"""
shake_engine.py - the part of the workflow that turns a displacement history into G-code.

Used by record_to_gcode.py; not meant to be run on its own. It holds:
  * readers for PEER .AT2/.VT2/.DT2 files and CSV tables,
  * filtering / integration to a band-limited displacement history,
  * the printer profile of the Ender 3 S1 Pro's stock firmware,
  * a model of that firmware's motion planner, used to choose each move's speed so the
    bed keeps the record's timing,
  * the G-code writer and the target-vs-predicted report and plot.

Printer firmware only understands straight moves with a speed (G1 Y.. F..). Its planner adds
its own acceleration ramps, so played back naively a record falls behind. The engine cuts the
record into short moves (split at every reversal), runs the planner model, and adjusts each
move's speed until every move ends when the record says it should.

Copied from shake_table/shake_table.py, which produced the sine demo that ran on the printer.
"""
from __future__ import annotations

import datetime as _dt
import math
import re
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

G_MM_S2 = 9806.65  # 1 g in mm/s^2
LAG_GAIN = 0.15    # share of the accumulated lag each move tries to take back

UNIT_FACTORS = {
    "accel": {"g": G_MM_S2, "m/s2": 1000.0, "cm/s2": 10.0, "gal": 10.0, "mm/s2": 1.0},
    "vel": {"m/s": 1000.0, "cm/s": 10.0, "mm/s": 1.0},
    "disp": {"m": 1000.0, "cm": 10.0, "mm": 1.0},
}
DEFAULT_UNITS = {"accel": "g", "vel": "mm/s", "disp": "mm"}
PEER_EXTENSIONS = {".at2": ("accel", "g"), ".vt2": ("vel", "cm/s"), ".dt2": ("disp", "cm")}


# --------------------------------------------------------------------------- printers

@dataclass(frozen=True)
class Profile:
    """What the generator needs to know about a printer + firmware combination."""
    name: str
    firmware: str          # "marlin" or "klipper"
    classic_jerk: bool     # Marlin CLASSIC_JERK planner (stock Creality firmware)
    steps_per_mm: float
    y_min: float           # default usable bed travel, absolute Y (mm)
    y_max: float
    max_feedrate: float    # highest Y speed the firmware accepts (mm/s)
    jerk_max: float        # highest Y jerk M205 accepts (classic jerk only)
    lookahead: int         # planned moves the planner sees beyond the one executing
    s_curve: bool          # S_CURVE_ACCELERATION
    min_steps: int         # shorter moves are silently dropped by the firmware
    ramp_floor: float      # lowest start/end speed of an acceleration ramp (mm/s)
    dt_min: float          # default shortest move (s)
    dt_max: float          # default longest move (s)
    min_move: float        # default shortest move (mm)
    v_quiet: float         # slower average speeds become pause + move (mm/s)
    max_accel: float       # default acceleration limit for the record (mm/s^2)
    restore: tuple = ()    # G-code that puts the firmware defaults back afterwards


PROFILES = {
    # Creality Ender-3 S1 / S1 Pro stock firmware (Marlin 2.0.x fork). Values from
    # github.com/CrealityOfficial/Ender-3S1 (Configuration.h, Configuration_adv.h,
    # planner.cpp): CLASSIC_JERK with Y jerk editable up to 10 mm/s, 8-block
    # planner buffer, S-curve acceleration, MIN_STEPS_PER_SEGMENT 6,
    # MINIMAL_STEP_RATE 120 steps/s, 80 steps/mm, soft endstops Y -8..220.
    "s1pro-stock": Profile(
        name="s1pro-stock", firmware="marlin", classic_jerk=True, steps_per_mm=80.0,
        y_min=5.0, y_max=215.0, max_feedrate=600.0, jerk_max=10.0, lookahead=6,
        s_curve=True, min_steps=6, ramp_floor=120.0 / 80.0,
        dt_min=0.005, dt_max=0.04, min_move=0.1, v_quiet=2.0, max_accel=1500.0,
        restore=("M201 Y500", "M203 Y300", "M204 P500 T1000", "M205 B20000 Y5"),
    ),
    # Klipper: junction-deviation planner, long look-ahead, constant-acceleration
    # ramps. Not tested on hardware.
    "klipper": Profile(
        name="klipper", firmware="klipper", classic_jerk=False, steps_per_mm=80.0,
        y_min=5.0, y_max=215.0, max_feedrate=600.0, jerk_max=0.0, lookahead=200,
        s_curve=False, min_steps=1, ramp_floor=0.0,
        dt_min=0.01, dt_max=0.01, min_move=0.02, v_quiet=0.5, max_accel=3000.0,
    ),
}


# --------------------------------------------------------------------------- input files

_FLOAT = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


def load_peer(path):
    """PEER NGA strong-motion file (.AT2/.VT2/.DT2): 4 header lines, then values."""
    lines = Path(path).read_text(errors="replace").splitlines()
    head = lines[3] if len(lines) > 3 else ""
    m_n = re.search(r"NPTS\s*=?\s*(\d+)", head, re.I)
    m_dt = re.search(r"DT\s*=?\s*(" + _FLOAT + ")", head, re.I)
    if m_n and m_dt:
        npts, dt = int(m_n.group(1)), float(m_dt.group(1))
    else:  # older layout: "  4000   0.0050   NPTS, DT"
        nums = re.findall(_FLOAT, head)
        if len(nums) < 2:
            raise ValueError(f"{path}: can't find NPTS and DT on header line 4")
        npts, dt = int(float(nums[0])), float(nums[1])
    values = np.array([float(v) for line in lines[4:] for v in re.findall(_FLOAT, line)])
    if len(values) < npts:
        print(f"warning: {path} has {len(values)} values, header says {npts}", file=sys.stderr)
    values = values[:npts]
    kind, units = PEER_EXTENSIONS[Path(path).suffix.lower()]
    return np.arange(len(values)) * dt, values, kind, units


def load_table(path, dt=None, time_col=0, value_col=None):
    """CSV / TXT with a time column and a value column, or one value column + dt."""
    rows = []
    for line in Path(path).read_text(errors="replace").splitlines():
        s = line.strip()
        if not s or s[0] in "#%!":
            continue
        if ";" in s:  # European CSV: ';' separates columns, ',' is the decimal mark
            parts = [p.strip().replace(",", ".") for p in s.split(";")]
        else:
            parts = [p for p in re.split(r"[,\t ]+", s) if p]
        try:
            rows.append([float(p) for p in parts if p != ""])
        except ValueError:
            continue  # header or text line
    if not rows:
        raise ValueError(f"{path}: no numeric rows found")
    widths = [len(r) for r in rows]
    ncol = max(set(widths), key=widths.count)
    data = np.array([r for r in rows if len(r) == ncol])
    if ncol == 1 or dt is not None:
        if dt is None:
            raise ValueError(f"{path} has a single column: give the sample interval with --dt")
        col = value_col if value_col is not None else (0 if ncol == 1 else 1)
        y = data[:, col]
        return np.arange(len(y)) * dt, y
    col = value_col if value_col is not None else (1 if time_col == 0 else 0)
    t, y = data[:, time_col], data[:, col]
    order = np.argsort(t, kind="stable")
    t, y = t[order], y[order]
    keep = np.concatenate(([True], np.diff(t) > 0))
    t, y = t[keep], y[keep]
    step = float(np.median(np.diff(t)))
    if np.max(np.abs(np.diff(t) - step)) > 0.01 * step:  # uneven sampling: resample
        tu = np.arange(t[0], t[-1] + 0.5 * step, step)
        y, t = np.interp(tu, t, y), tu
    return t - t[0], y


# --------------------------------------------------------------------------- signal processing

def _taper(n, m):
    """Window that is 1 in the middle with half-cosine ends of m samples."""
    w = np.ones(n)
    m = min(int(m), n // 2)
    if m > 0:
        r = 0.5 * (1.0 - np.cos(np.pi * np.arange(m) / m))
        w[:m] = r
        w[n - m:] = r[::-1]
    return w


def _spectral(y, dt, n_int, f_hp, f_lp, dt_out):
    """Zero-phase Butterworth (4th order) filtering, integration and resampling via FFT."""
    n = len(y)
    big_n = 1 << int(math.ceil(math.log2(2 * n)))  # >= 2n, so filter tails don't wrap round
    spec = np.fft.rfft(y, big_n)
    f = np.fft.rfftfreq(big_n, dt)
    fpos = np.where(f > 0, f, 1.0)
    h = np.ones_like(f, dtype=complex)
    if f_hp > 0:
        h = np.where(f > 0, 1.0 / np.sqrt(1.0 + (f_hp / fpos) ** 8), 0.0).astype(complex)
    if f_lp > 0:
        h /= np.sqrt(1.0 + (f / f_lp) ** 8)
    if n_int:
        h /= (2j * np.pi * fpos) ** n_int
        h[0] = 0.0
    spec = spec * h
    n_out = int(round(big_n * dt / dt_out))
    m = n_out // 2 + 1
    spec_out = np.zeros(m, dtype=complex)
    k = min(m, len(spec))
    spec_out[:k] = spec[:k]
    out = np.fft.irfft(spec_out, n_out) * (n_out / big_n)
    dt_new = big_n * dt / n_out
    keep = int(math.floor((n - 1) * dt / dt_new)) + 1
    return out[:keep], dt_new


def fft_lowpass(y, dt, f_lp):
    base = np.linspace(y[0], y[-1], len(y))  # make the ends meet: no FFT edge ringing
    out, _ = _spectral(y - base, dt, 0, 0.0, f_lp, dt)
    return out[: len(y)] + base


def to_displacement(y, dt, kind, f_hp, f_lp, dt_out):
    """Displacement (mm) on a uniform grid of about dt_out, starting and ending at rest."""
    y = np.asarray(y, float)
    n = len(y)
    span = (n - 1) * dt
    keep_offset = kind == "disp" and f_hp <= 0
    if kind == "disp":
        y0, y1 = y[0], y[-1]
        y = y - (y0 + (y1 - y0) * np.arange(n) / max(n - 1, 1))  # ends at 0, no FFT ringing
    else:
        if f_hp <= 0:
            raise ValueError("velocity and acceleration records need --highpass > 0 "
                             "(otherwise the integrated displacement drifts away)")
        y = (y - y.mean()) * _taper(n, min(0.05 * n, 1.0 / dt))
    pad_s = (max(2.0, 3.0 / f_hp) if f_hp > 0 else 0.5)
    pad = int(round(pad_s / dt))
    yp = np.concatenate([np.zeros(pad), y, np.zeros(pad)])
    x, dt_new = _spectral(yp, dt, {"accel": 2, "vel": 1, "disp": 0}[kind], f_hp, f_lp, dt_out)
    t = -pad * dt + np.arange(len(x)) * dt_new
    if keep_offset:
        x = x + y0 + (y1 - y0) * (np.clip(t / span, 0.0, 1.0) if span > 0 else 0.0)
    else:
        x = x * _taper(len(x), 0.5 * pad * dt / dt_new)  # fade to rest inside the padding
    return t, x, dt_new


def trim_quiet(t, x, thr=0.005, margin_s=0.3):
    """Cut leading/trailing stretches where the bed would not move."""
    dt = t[1] - t[0]
    moving = np.nonzero((np.abs(x - x[0]) > thr) | (np.abs(x - x[-1]) > thr))[0]
    if len(moving) == 0:
        raise ValueError("the processed record has no motion above 0.005 mm")
    first = np.nonzero(np.abs(x - x[0]) > thr)[0]
    last = np.nonzero(np.abs(x - x[-1]) > thr)[0]
    m = int(round(margin_s / dt))
    i0 = max(0, (first[0] if len(first) else 0) - m)
    i1 = min(len(x) - 1, (last[-1] if len(last) else len(x) - 1) + m)
    return t[i0:i1 + 1] - t[i0], x[i0:i1 + 1]


@dataclass
class Target:
    t: np.ndarray
    x: np.ndarray          # displacement relative to the motion's own zero (mm)
    dt: float
    label: str
    notes: list = field(default_factory=list)
    v: np.ndarray = None
    a: np.ndarray = None

    def derive(self):
        self.v = np.gradient(self.x, self.dt)
        self.a = np.gradient(self.v, self.dt)
        return self

    def scale(self, s):
        self.x = self.x * s
        return self.derive()


# --------------------------------------------------------------------------- segmentation

@dataclass
class Item:
    kind: str       # "move" or "dwell"
    T: float        # duration the record asks for (s)
    t_end: float    # record time at the end of the item (s)
    y: float = 0.0  # absolute bed Y at the end of a move (mm)
    T_out: float = None  # pause length actually written (re-synchronises the timing)

    @property
    def dwell_time(self):
        return self.T if self.T_out is None else self.T_out


def segment(tgt, offset, prof, opts):
    """Cut the displacement history into firmware-sized moves.

    A move ends when it is long enough (opts.dt_max), at every turning point,
    or (classic-jerk firmware) when the speed has changed by 60 % of the jerk
    limit, because Marlin brakes hard between moves whose speeds differ by more
    than the jerk. Moves are never shorter than opts.dt_min or opts.min_move
    (the firmware drops moves of < 6 steps).
    """
    t, x, v = tgt.t, tgt.x, tgt.v
    n = len(x)
    kmin = max(1, int(round(opts.dt_min / tgt.dt)))
    kmax = max(kmin, int(round(opts.dt_max / tgt.dt)))
    dv_seg = 0.6 * opts.jerk if prof.classic_jerk else math.inf
    sgn = np.sign(np.diff(x))
    idx = np.where(sgn != 0, np.arange(len(sgn)), 0)
    np.maximum.accumulate(idx, out=idx)
    sgn = sgn[idx]
    ext = np.zeros(n, bool)
    ext[1:-1] = (sgn[1:] != sgn[:-1]) & (sgn[1:] != 0) & (sgn[:-1] != 0)
    xl, vl, el, tl = x.tolist(), v.tolist(), ext.tolist(), t.tolist()
    mm = opts.min_move

    bounds = [0]
    k0 = 0
    while k0 < n - 1:
        x0, v0 = xl[k0], vl[k0]
        end = n - 1
        for k in range(k0 + 1, n):
            dist = abs(xl[k] - x0)
            if el[k] and dist < mm and len(bounds) >= 2:
                # A turning point too close to the last cut: end the previous move
                # here instead, so the firmware never sees a reversal hidden inside
                # one move (it would plan that move as slow and start the next from rest).
                kp = bounds[-2]
                if abs(xl[k] - xl[kp]) >= mm and (xl[k] - xl[kp]) * (x0 - xl[kp]) > 0:
                    bounds.pop()
                    end = k
                    break
            if k - k0 >= kmin and dist >= mm:
                if k - k0 >= kmax or el[k] or abs(vl[k] - v0) >= dv_seg:
                    end = k
                    break
        bounds.append(end)
        k0 = end

    items = []
    for k0, end in zip(bounds[:-1], bounds[1:]):
        x0 = xl[k0]
        T, dist = tl[end] - tl[k0], abs(xl[end] - x0)
        if dist < opts.min_move:
            items.append(Item("dwell", T, tl[end]))
        else:
            tm = T
            if dist / T < opts.v_quiet and T - dist / opts.v_quiet > 0.002:
                tm = dist / opts.v_quiet  # too slow to step smoothly: wait, then move
                items.append(Item("dwell", T - tm, tl[end] - tm))
            items.append(Item("move", tm, tl[end], round(offset + xl[end], 3)))

    merged = []
    for it in items:  # join neighbouring pauses
        if merged and it.kind == "dwell" and merged[-1].kind == "dwell":
            merged[-1].T += it.T
            merged[-1].t_end = it.t_end
        else:
            merged.append(it)
    # pauses under 1 ms can't be written as G4; fold them into the next move
    out = []
    carry = 0.0
    for it in merged:
        if it.kind == "dwell" and it.T < 0.001:
            carry += it.T
            continue
        if carry and it.kind == "move":
            it.T += carry
        carry = 0.0
        out.append(it)
    return out


# --------------------------------------------------------------------------- planner model

def _cj_junction(vp, dp, v, d, maxj, safe_prev):
    """Marlin CLASSIC_JERK junction speed for one axis (planner.cpp, _populate_block)."""
    safe = v if v <= maxj else maxj
    if v < vp:
        vj, f = v, v / vp
    else:
        vj, f = vp, 1.0
    v_exit, v_entry = dp * vp * f, d * v
    jv = abs(v_exit - v_entry) if dp == d else max(abs(v_exit), abs(v_entry))
    if jv > maxj:
        vj *= maxj / jv
    if safe_prev > 0.99 * vj and safe > 0.99 * vj:
        vj = safe
    return vj, safe


def _classic_junctions(V, d, start, opts, jerk_max):
    """Junction speeds of stock Marlin, choosing the M205 Y jerk for every move.

    Classic jerk compares the *nominal* speeds of consecutive moves with the jerk
    limit. Speeding up by more than the jerk brakes hard at the joint; below the
    jerk speed it allows instant velocity steps instead of ramps. For each move we
    pick the jerk value that makes the joint behave like an ideal planner (no dip,
    no step, near-zero speed at reversals), and keep the previous value whenever
    it already works, so only a few M205 lines are needed.
    """
    n = len(V)
    Vl, dl, sl = V.tolist(), d.tolist(), start.tolist()
    vj, jk = [0.0] * n, [0.0] * n
    cur, safe_prev = opts.jerk, 0.0
    for i in range(n):
        if sl[i]:  # planner empty (first move or after a pause): starts from rest
            jk[i], vj[i] = cur, 0.0
            safe_prev = min(Vl[i], cur)
            continue
        vp, dp, v, di = Vl[i - 1], dl[i - 1], Vl[i], dl[i]
        ideal = min(vp, v) if di == dp else 0.0
        cands = (cur, opts.jerk,
                 min(jerk_max, max(opts.reversal_jerk, (v - vp) * 1.02 + 0.02)),
                 min(jerk_max, max(opts.reversal_jerk, vp)),
                 opts.reversal_jerk)
        best = None
        for c in cands:
            vjc, safe = _cj_junction(vp, dp, v, di, c, safe_prev)
            err = abs(vjc - ideal) + (0.0 if c == cur else 0.05)
            if best is None or err < best[0]:
                best = (err, c, vjc, safe)
        _, cur, vj[i], safe_prev = best
        jk[i] = cur
    return np.array(vj), np.array(jk)


def simulate(L, d, V, A, start, prof, opts):
    """Predict what the firmware does with moves of length L, direction d, speed V."""
    n = len(L)
    if prof.classic_jerk:
        vj, jk = _classic_junctions(V, d, start, opts, prof.jerk_max)
    else:
        Vp = np.concatenate(([0.0], V[:-1]))
        dp = np.concatenate(([0.0], d[:-1]))
        vj = np.where(d == dp, np.minimum(V, Vp), 0.0)
        vj[start] = 0.0
        jk = np.full(n, np.nan)

    # Look-ahead: a move may only start as fast as the machine can still stop by
    # the end of the moves already in the planner buffer.
    W = prof.lookahead
    Lp = np.concatenate((L, np.zeros(W)))
    vjp = np.concatenate((vj, np.zeros(W)))
    rev = np.zeros(n)
    base = np.arange(n)
    for m in range(W - 1, -1, -1):
        rev = np.minimum(vjp[base + m], np.sqrt(rev * rev + 2.0 * A * Lp[base + m]))

    # Forward pass: a move can't start faster than the previous one could reach.
    if prof.firmware == "marlin":  # Marlin skips this for moves that reach full speed
        skip = (V * V <= 2.0 * A * L).tolist()
    else:
        skip = [False] * n
    cap = np.minimum(vj, rev).tolist()
    Ll, sl = L.tolist(), start.tolist()
    entry = [0.0] * n
    for i in range(n):
        c = cap[i]
        if i and not sl[i] and not skip[i - 1]:
            c = min(c, math.sqrt(entry[i - 1] ** 2 + 2.0 * A * Ll[i - 1]))
        entry[i] = c
    entry = np.array(entry)
    exit_ = np.concatenate((entry[1:], [0.0]))

    # Trapezoid of each move (S-curve ramps take the same time as linear ones).
    fl = prof.ramp_floor
    e = np.minimum(np.maximum(entry, fl), V)
    xo = np.minimum(np.maximum(exit_, fl), V)
    two_a = 2.0 * A
    peak = np.minimum(V, np.sqrt(np.maximum((two_a * L + e * e + xo * xo) / 2.0, 0.0)))
    peak = np.maximum(peak, np.maximum(e, xo))
    reach = np.abs(xo * xo - e * e) / two_a <= L * (1 + 1e-9)
    t_acc = (peak - e) / A
    t_dec = (peak - xo) / A
    t_cru = np.maximum(L - (peak * peak - e * e) / two_a - (peak * peak - xo * xo) / two_a, 0.0) / peak
    dur = np.where(reach, t_acc + t_cru + t_dec, 2.0 * L / np.maximum(e + xo, 1e-9))
    return dict(V=V, vj=vj, jerk=jk, entry=entry, exit=exit_, e=e, xo=xo, peak=peak,
                t_acc=t_acc, t_cru=t_cru, t_dec=t_dec, reach=reach, dur=dur)


def plan(items, y_start, A, prof, opts):
    """Pick each move's speed so the emulated firmware hits the record's timing.

    Every move aims at its absolute end time in the record, so a move that runs
    late is made up by the following ones, and pauses are stretched or shortened
    to re-synchronise (a pause ends exactly when the record says it should).
    """
    moves = [it for it in items if it.kind == "move"]
    n = len(moves)
    start, pause_item, prev = [], [None] * n, None
    for it in items:
        if it.kind == "move":
            start.append(prev is None or prev.kind == "dwell")
            if prev is not None and prev.kind == "dwell":
                pause_item[len(start) - 1] = prev
        prev = it
    start = np.array(start, bool)
    steps = np.round(np.array([y_start] + [m.y for m in moves]) * prof.steps_per_mm)
    dl = np.diff(steps) / prof.steps_per_mm
    L, d = np.abs(dl), np.sign(dl)
    T = np.array([m.T for m in moves])
    t_end = np.array([m.t_end for m in moves])
    if np.any(np.abs(np.diff(steps)) < prof.min_steps):
        raise RuntimeError("internal: a move is shorter than the firmware minimum")

    def timeline(dur, gain):
        """Aimed durations, end-time drift and pause lengths for the given move times.

        A pause ends exactly at its time in the record (full re-sync). With
        gain > 0, each move also takes back a small share of the lag accumulated
        before it (a small share, so the moves don't all over-correct at once).
        """
        tc = 0.0
        aim, drift, pauses = np.empty(n), np.empty(n), [0.0] * n
        dl_, tl_, Tl_ = dur.tolist(), t_end.tolist(), T.tolist()
        for i in range(n):
            p = pause_item[i]
            if p is not None:
                pauses[i] = max(0.001, p.t_end - tc)
                tc += pauses[i]
            lag = tc - (tl_[i] - Tl_[i])
            aim[i] = min(max(Tl_[i] - gain * lag, 0.7 * Tl_[i]), 1.3 * Tl_[i])
            tc += dl_[i]
            drift[i] = tc - tl_[i]
        return aim, drift, pauses

    V = L / T
    vcap = np.minimum(prof.max_feedrate, np.maximum(3.0 * L / T, 1.0))
    same_dir = np.concatenate(([False], d[1:] == d[:-1])) & ~start
    best = (math.inf, V, None)
    gain = 0.0  # first make every move take its own time, then mop up what is left
    for it in range(opts.iterations):
        sim = simulate(L, d, V, A, start, prof, opts)
        aim, drift, pauses = timeline(sim["dur"], gain)
        worst = float(np.max(np.abs(drift)))
        if worst < best[0]:
            best = (worst, V.copy(), pauses)
        if worst < 5e-4:
            break
        ratio = sim["dur"] / aim
        if gain == 0.0 and (np.max(np.abs(ratio - 1.0)) < 2e-3 or it >= opts.iterations // 3):
            if worst < 2e-3:
                break
            gain = LAG_GAIN
            aim, drift, pauses = timeline(sim["dur"], gain)
            ratio = sim["dur"] / aim
        V = np.clip(V * np.clip(ratio, 0.8, 1.25) ** 0.7, 0.5 * L / T, vcap)
        if prof.classic_jerk:
            # Above the jerk speed, speeding up between two moves by more than the
            # jerk makes Marlin brake at the joint (no jerk value avoids it), so
            # never ask for that. Below it, the per-move jerk choice mostly copes.
            Vl, sd = V.tolist(), same_dir.tolist()
            lim = 0.95 * prof.jerk_max
            for i in range(1, len(Vl)):
                if sd[i] and Vl[i - 1] > lim and Vl[i] > Vl[i - 1] + lim:
                    Vl[i] = Vl[i - 1] + lim
            V = np.array(Vl)
    V, pauses = best[1], best[2]
    for i, p in enumerate(pause_item):
        if p is not None:
            p.T_out = pauses[i]
    sim = simulate(L, d, V, A, start, prof, opts)
    sim.update(L=L, d=d, T=T, start=start, iterations=it + 1, max_drift=best[0])
    return sim


def plan_best(items, y_start, A, prof, opts):
    """Plan, and for classic-jerk firmware also pick the jerk used at reversals.

    A low reversal jerk turns the bed round gently but can leave the next move
    unable to speed up in time (Marlin's jerk rule), a high one does the reverse.
    The effect is erratic, so we simply try a few values and keep the one whose
    predicted timing is best, preferring gentler reversals when they're as good.
    """
    if not prof.classic_jerk or opts.reversal_jerk is not None:
        return items, plan(items, y_start, A, prof, opts)
    trials = []
    for rj in (0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0):
        if rj > opts.jerk:
            continue
        opts.reversal_jerk = rj
        its = [replace(it) for it in items]
        trials.append((rj, its, plan(its, y_start, A, prof, opts)))
    best_drift = min(t[2]["max_drift"] for t in trials)
    rj, its, sim = next(t for t in trials if t[2]["max_drift"] <= best_drift + 0.001)
    opts.reversal_jerk = rj
    return its, sim


# --------------------------------------------------------------------------- predicted motion

def _phases(items, sim, y_start, offset):
    """Piecewise description of the emulated motion: (t0, T, v0, v1, p0, dir, ramp)."""
    rows = []
    tc, pc, mi = 0.0, y_start - offset, 0
    for it in items:
        if it.kind == "dwell":
            rows.append((tc, it.dwell_time, 0.0, 0.0, pc, 0.0, False))
            tc += it.dwell_time
            continue
        i, mi = mi, mi + 1
        dirn = sim["d"][i]
        if sim["reach"][i]:
            parts = ((sim["t_acc"][i], sim["e"][i], sim["peak"][i]),
                     (sim["t_cru"][i], sim["peak"][i], sim["peak"][i]),
                     (sim["t_dec"][i], sim["peak"][i], sim["xo"][i]))
        else:
            parts = ((sim["dur"][i], sim["e"][i], sim["xo"][i]),)
        for tp, a, b in parts:
            if tp > 1e-9:
                rows.append((tc, tp, a, b, pc, dirn, a != b))
                pc += dirn * 0.5 * (a + b) * tp
                tc += tp
        pc = it.y - offset  # snap to the exact end point
    return np.array(rows, dtype=float), tc


def evaluate(rows, times, s_curve):
    """Position, velocity and acceleration of the emulated motion at given times."""
    t0, T, v0, v1, p0, dirn, ramp = rows.T
    j = np.clip(np.searchsorted(t0, times, side="right") - 1, 0, len(t0) - 1)
    tau = np.clip((times - t0[j]) / T[j], 0.0, 1.0)
    sc = (ramp[j] > 0) & s_curve
    big_s = np.where(sc, tau ** 4 * (2.5 - 3 * tau + tau * tau), 0.5 * tau * tau)
    small_s = np.where(sc, tau ** 3 * (10 - 15 * tau + 6 * tau * tau), tau)
    slope = np.where(sc, 30 * tau * tau * (1 - tau) ** 2, 1.0)
    dv = v1[j] - v0[j]
    pos = p0[j] + dirn[j] * (v0[j] * T[j] * tau + dv * T[j] * big_s)
    vel = dirn[j] * (v0[j] + dv * small_s)
    acc = dirn[j] * dv / T[j] * slope
    return pos, vel, acc


def velocity_steps(rows):
    """Instant speed changes between consecutive pieces (jerk the firmware allows)."""
    end_v = rows[:-1, 3] * rows[:-1, 5]
    start_v = rows[1:, 2] * rows[1:, 5]
    return np.abs(start_v - end_v)


# --------------------------------------------------------------------------- G-code

def setup_lines(prof, vmax, accel, jerk, home, min_segment_us=0):
    if prof.firmware == "marlin":
        out = ["G21 ; millimetres",
               "G90 ; absolute positions",
               "M420 S0 ; bed-levelling correction off (no Z moves mixed into Y moves)",
               "M211 S1 ; software endstops on",
               f"M203 Y{vmax:.0f} ; max Y speed (mm/s)",
               f"M201 Y{accel:.0f} ; max Y acceleration (mm/s^2)",
               f"M204 P{accel:.0f} T{accel:.0f} ; acceleration for these moves"]
        jerk_txt = f" Y{jerk:g}" if prof.classic_jerk else ""
        why = ("no minimum move time (stops Marlin slowing short moves)" if min_segment_us == 0
               else "stock minimum move time")
        out.append(f"M205 B{min_segment_us} S0 T0{jerk_txt} ; {why}, Y jerk")
    else:
        out = ["G90",
               "BED_MESH_CLEAR",
               "SET_INPUT_SHAPER SHAPER_FREQ_X=0 SHAPER_FREQ_Y=0 ; input shaping would "
               "reshape the record (harmless if not configured)",
               f"SET_VELOCITY_LIMIT VELOCITY={vmax:.0f} ACCEL={accel:.0f} "
               f"MINIMUM_CRUISE_RATIO=0 SQUARE_CORNER_VELOCITY=5"]
    if home:
        if prof.firmware == "marlin":  # plain G28 Y would raise an unhomed Z by 10 mm every run
            out.append("G28 Y R0 ; home Y only, no Z raise, no probe needed")
        else:
            out.append("G28 Y ; home Y only")
    return out


def end_lines(prof, y_park):
    out = ["M400", f"G1 Y{y_park:.3f} F1200 ; park", "M400"]
    if prof.restore:
        out.append("; put the stock settings back (they are in RAM only, not saved)")
        out += list(prof.restore)
    else:
        out.append("; restart Klipper (or SET_VELOCITY_LIMIT) to restore your normal limits")
    return out


def motion_gcode(items, sim, prof, opts, accel, vmax, y_start, y_park, header):
    out = [f"; {line}" if line else ";" for line in header]
    out += setup_lines(prof, vmax, accel, opts.jerk, not opts.no_home)
    out += [f"G1 Y{y_start:.3f} F{opts.preposition_speed * 60:.0f} ; go to the start position",
            "M400",
            "M117 Shake table running",
            f"G4 S{opts.pre_dwell:g} ; settle",
            "M118 SHAKE START"]
    mi, cur_jerk, n_jerk = 0, opts.jerk, 0
    for it in items:
        if it.kind == "dwell":
            out.append(f"G4 P{it.dwell_time * 1000:.0f}")
            continue
        if prof.classic_jerk and abs(sim["jerk"][mi] - cur_jerk) > 1e-9:
            cur_jerk = sim["jerk"][mi]
            out.append(f"M205 Y{cur_jerk:.2f}")
            n_jerk += 1
        out.append(f"G1 Y{it.y:.3f} F{sim['V'][mi] * 60:.1f}")
        mi += 1
    out += ["M400", "M118 SHAKE END", "M117 Shake table done", "G4 S1"]
    out += end_lines(prof, y_park)
    return out, n_jerk


# --------------------------------------------------------------------------- record / sine command

def run_motion(tgt, args, prof):
    warnings = []
    y_min = args.y_min if args.y_min is not None else prof.y_min
    y_max = args.y_max if args.y_max is not None else prof.y_max
    max_vel = min(args.max_vel, prof.max_feedrate)

    # 1. limits
    def peaks():
        return (float(tgt.x.max() - tgt.x.min()), float(np.max(np.abs(tgt.v))),
                float(np.max(np.abs(tgt.a))))
    stroke, vpk, apk = peaks()
    fits = {"travel": (y_max - y_min) / max(stroke, 1e-9),
            "speed": max_vel / max(vpk, 1e-9),
            "acceleration": args.max_accel / max(apk, 1e-9)}
    s_fit = min(fits.values())
    limit_lines = [f"stroke {stroke:.1f} mm of {y_max - y_min:.0f} mm travel (Y {y_min:g}..{y_max:g})",
                   f"peak speed {vpk:.1f} of {max_vel:g} mm/s",
                   f"peak acceleration {apk:.0f} of {args.max_accel:g} mm/s^2 ({apk / G_MM_S2:.3f} g)"]
    if s_fit < 1.0:
        over = ", ".join(f"{k} x{1 / v:.2f}" for k, v in fits.items() if v < 1.0)
        if args.autoscale:
            tgt.scale(0.99 * s_fit)
            tgt.notes.append(f"autoscale    : amplitude x{0.99 * s_fit:.3f} to fit the limits ({over} too big)")
            stroke, vpk, apk = peaks()
        elif not args.force:
            print("The motion does NOT fit the machine limits:", file=sys.stderr)
            for line in limit_lines:
                print("  " + line, file=sys.stderr)
            print(f"  too big: {over}", file=sys.stderr)
            print(f"Options: --autoscale (amplitude x{0.99 * s_fit:.3f}), --scale, a stronger "
                  f"--highpass to cut the stroke, or raise the limits if your machine can take it.",
                  file=sys.stderr)
            raise SystemExit(2)
        else:
            warnings.append(f"limits exceeded ({over}); generated anyway because of --force")

    # 2. placement on the bed
    center = args.center if args.center is not None else 0.5 * (y_min + y_max)
    offset = center - 0.5 * (tgt.x.max() + tgt.x.min())
    if offset + tgt.x.min() < y_min - 1e-6 or offset + tgt.x.max() > y_max + 1e-6:
        if not args.force:
            raise SystemExit(f"error: with --center {center:g} the motion leaves Y {y_min:g}..{y_max:g}")
    y_start = round(offset + tgt.x[0], 3)
    if abs(tgt.v[0]) > max(prof.ramp_floor, 1.0):
        warnings.append(f"the record starts moving at {tgt.v[0]:.1f} mm/s; the bed starts from rest, "
                        f"so the first moment will be off")

    # 3. moves
    items = segment(tgt, offset, prof, args)
    moves = [it for it in items if it.kind == "move"]
    if not moves:
        raise SystemExit(f"error: the motion is too small to play: the firmware ignores moves shorter "
                         f"than about {args.min_move:.3f} mm ({prof.min_steps} motor steps). "
                         f"Increase the amplitude.")
    accel = args.planner_accel
    if accel is None:  # ramps must be ~3x quicker than the record to keep its timing
        accel = float(np.clip(100 * math.ceil(3.0 * apk / 100), 1000, args.planner_accel_max))

    # 4. timing correction through the planner model
    items, sim = plan_best(items, y_start, accel, prof, args)
    vmax = float(np.clip(math.ceil(1.1 * sim["V"].max() / 10) * 10, 300, prof.max_feedrate))

    # 5. predicted motion vs target
    rows, t_total = _phases(items, sim, y_start, offset)
    pos, vel, acc = evaluate(rows, tgt.t, prof.s_curve)
    item_end_pred = []
    k = 0
    tc = 0.0
    for it in items:  # end time of each item in the emulated run
        if it.kind == "dwell":
            tc += it.dwell_time
        else:
            tc += sim["dur"][k]
            k += 1
        item_end_pred.append(tc)
    drift = np.array(item_end_pred) - np.array([it.t_end for it in items])
    ex = pos - tgt.x
    ev = vel - tgt.v
    f_cmp = args.lowpass if args.lowpass else 20.0
    acc_lp = fft_lowpass(acc, tgt.dt, f_cmp)
    tgt_lp = fft_lowpass(tgt.a, tgt.dt, f_cmp)
    a_err = float(np.sqrt(np.mean((acc_lp - tgt_lp) ** 2)))
    steps = velocity_steps(rows)
    _, _, acc_fine = evaluate(rows, np.arange(0.0, t_total, 0.0002), prof.s_curve)

    # line rate
    starts = np.array([0.0] + item_end_pred[:-1])
    win = 0.25
    counts = np.searchsorted(starts, starts + win) - np.arange(len(starts))
    peak_rate = counts.max() / win
    avg_rate = len(items) / max(t_total, 1e-9)
    if prof.firmware == "marlin" and peak_rate > 250:
        warnings.append(f"up to {peak_rate:.0f} moves/s is a lot for this firmware, especially over "
                        f"USB; raise --dt-min if the run stutters")
    if np.any(~sim["reach"]):
        warnings.append(f"{int(np.sum(~sim['reach']))} moves need more than {accel:.0f} mm/s^2 "
                        f"between them (raise --planner-accel)")
    if np.max(np.abs(drift)) > 0.02:
        warnings.append(f"timing drifts up to {np.max(np.abs(drift)) * 1000:.0f} ms: the planner "
                        f"can't keep up (try a higher --planner-accel or lower amplitude)")
    if prof.classic_jerk and apk > 1600:
        warnings.append("the stock (classic-jerk) planner follows accelerations above ~1500 mm/s^2 "
                        "less closely; check the predicted errors above")

    report = list(tgt.notes) + [
        f"motion       : {tgt.t[-1]:.1f} s, stroke {stroke:.1f} mm (Y {offset + tgt.x.min():.1f} .. "
        f"{offset + tgt.x.max():.1f}), peak {vpk:.0f} mm/s, peak {apk:.0f} mm/s^2 ({apk / G_MM_S2:.2f} g)",
        f"limits       : Y {y_min:g}..{y_max:g}, {max_vel:g} mm/s, {args.max_accel:g} mm/s^2 -> "
        + ("OK" if s_fit >= 0.999 or args.autoscale else "EXCEEDED"),
        f"firmware     : {prof.name}, planner accel {accel:.0f} mm/s^2"
        + (f", jerk {args.jerk:g} mm/s (re-chosen per move, {args.reversal_jerk:g} at reversals)"
           if prof.classic_jerk else "")
        + (", S-curve ramps" if prof.s_curve else ""),
        f"G-code       : {len(moves)} moves, {len(items) - len(moves)} pauses, "
        f"avg {avg_rate:.0f} / peak {peak_rate:.0f} moves per second",
        "predicted (model of the firmware planner):",
        f"  timing       : end times within {np.max(np.abs(drift)) * 1000:.1f} ms of the record",
        f"  displacement : max error {np.max(np.abs(ex)):.3f} mm, rms {np.sqrt(np.mean(ex ** 2)):.3f} mm",
        f"  velocity     : rms error {np.sqrt(np.mean(ev ** 2)):.2f} mm/s",
        f"  acceleration : rms error below {f_cmp:g} Hz {a_err:.0f} mm/s^2 "
        f"({100 * a_err / max(apk, 1e-9):.0f} % of peak); raw peaks up to {np.max(np.abs(acc_fine)):.0f} mm/s^2"
        + (" (S-curve ramps)" if prof.s_curve else ""),
        f"  speed steps  : largest instant speed change {steps.max():.1f} mm/s",
    ]

    header = [f"Shake-table motion for {prof.name}, generated {_dt.datetime.now():%Y-%m-%d %H:%M} "
              f"by record_to_gcode.py", f"Target: {tgt.label}", ""] + report
    if warnings:
        header += ["", "WARNINGS:"] + ["  " + w for w in warnings]
    header += ["", "Print this file from the SD card. Keep a hand near the power switch the first time.",
               "", f"EXPECTED_RUN_S={t_total:.3f}", ""]
    lines, n_jerk = motion_gcode(items, sim, prof, args, accel, vmax, y_start, round(center, 3), header)
    Path(args.output).write_text("\n".join(lines) + "\n")

    print(f"Wrote {args.output} ({len(lines)} lines"
          + (f", {n_jerk} jerk changes" if prof.classic_jerk else "") + ")")
    for line in report:
        print("  " + line)
    for w in warnings:
        print("  WARNING: " + w)

    if args.csv:
        a_pred_lp = acc_lp
        data = np.column_stack([tgt.t, tgt.x, pos, tgt.v, vel, tgt.a, a_pred_lp])
        np.savetxt(args.csv, data, delimiter=",", fmt="%.6g",
                   header="t_s,x_target_mm,x_pred_mm,v_target_mm_s,v_pred_mm_s,"
                          "a_target_mm_s2,a_pred_lowpass_mm_s2", comments="")
        print(f"Wrote {args.csv}")
    if args.plot:
        make_plot(args.plot, tgt, pos, vel, acc_lp, tgt_lp, f_cmp, prof, tgt.label)
        print(f"Wrote {args.plot}")
    return {"pos": pos, "vel": vel, "acc_lp": acc_lp, "tgt_lp": tgt_lp, "f_cmp": f_cmp,
            "offset": offset, "report": report, "warnings": warnings, "t_total": t_total,
            "max_drift": float(np.max(np.abs(drift))), "max_x_err": float(np.max(np.abs(ex)))}


def make_plot(path, tgt, pos, vel, acc_lp, tgt_lp, f_cmp, prof, label):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping --plot", file=sys.stderr)
        return
    fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    ax[0].plot(tgt.t, tgt.x, label="target", lw=1.6)
    ax[0].plot(tgt.t, pos, label="predicted", lw=1, ls="--")
    ax[0].set_ylabel("displacement (mm)")
    ax[1].plot(tgt.t, tgt.v, lw=1.6)
    ax[1].plot(tgt.t, vel, lw=0.8, ls="--")
    ax[1].set_ylabel("velocity (mm/s)")
    ax[2].plot(tgt.t, tgt_lp, lw=1.6)
    ax[2].plot(tgt.t, acc_lp, lw=0.8, ls="--")
    ax[2].set_ylabel(f"accel < {f_cmp:g} Hz (mm/s^2)")
    ax[2].set_xlabel("time (s)")
    ax[0].legend(loc="upper right")
    for a in ax:
        a.grid(alpha=0.3)
    fig.suptitle(f"{label} - {prof.name}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)

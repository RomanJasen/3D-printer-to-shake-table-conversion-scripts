#!/usr/bin/env python3
"""
synthesize_record.py - make synthetic earthquake-like ground motion records.

Step 1 of the shake-table workflow. Step 2 is record_to_gcode.py.

  python synthesize_record.py --duration 20 --pga 0.1g --plot
  python synthesize_record.py --duration 30 --pga 0.12g --pgv-max 100mm/s --pgd-max 40mm --count 5
  python synthesize_record.py --spectrum ec8 --ag 0.1g --ground C --duration 20 --plot
  python synthesize_record.py --spectrum asce7 --sds 0.5g --sd1 0.2g --plot
  python synthesize_record.py --spectrum my_spectrum.csv --spectrum-units g --periods 0.1 2

Values take units: 0.1g, 980mm/s2, 0.98m/s2, 98gal | 100mm/s, 10cm/s | 40mm, 4cm.
A bare number means mm, mm/s or mm/s^2.

The shaking is defined either by peak values (--pga, --pgv-max, --pgd-max) or by a target
response spectrum (--spectrum): Eurocode 8, ASCE 7, or a file of period / Sa pairs. A
spectrum-matched record starts from random phases and is adjusted, frequency by frequency,
until its response spectrum matches the target over --periods (the classic iterative
method of SIMQKE, Gasparini & Vanmarcke 1976).

Method for peak values (a standard stochastic ground-motion model):
  1. white noise, coloured by a Kanai-Tajimi ground filter (dominant frequency and damping),
  2. shaped in time by an envelope: build-up, strong shaking, exponential decay,
  3. band-limited (high-pass f_low, low-pass f_high) and integrated to displacement,
  4. displacement faded to rest at both ends; velocity and acceleration are its exact
     derivatives, so all three start and end at zero and match each other,
  5. scaled so the peak acceleration equals --pga. If the peak velocity or displacement
     would exceed --pgv-max / --pgd-max, f_low is raised (this removes slow drift and
     barely changes the acceleration) until they fit.

Output CSV columns: time_s, accel_mm_s2, vel_mm_s, disp_mm.
Requires numpy. matplotlib is only needed for --plot.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import math
import re
import secrets
import sys
from pathlib import Path

import numpy as np

G = 9806.65  # mm/s^2

UNITS = {
    "accel": {"": 1.0, "mm/s2": 1.0, "cm/s2": 10.0, "gal": 10.0, "m/s2": 1000.0, "g": G},
    "vel": {"": 1.0, "mm/s": 1.0, "cm/s": 10.0, "m/s": 1000.0},
    "disp": {"": 1.0, "mm": 1.0, "cm": 10.0, "m": 1000.0},
}


def quantity(kind, allow_none=False):
    """argparse type for '0.1g', '100mm/s', '40', or 'none'."""
    def parse(text):
        if allow_none and text.strip().lower() in ("none", "off"):
            return None
        m = re.fullmatch(r"\s*(\d*\.?\d+(?:[eE][-+]?\d+)?)\s*([a-zA-Z/^0-9]*)\s*", text)
        unit = m.group(2).lower().replace("^", "") if m else None
        if not m or unit not in UNITS[kind]:
            units = ", ".join(u for u in UNITS[kind] if u)
            raise argparse.ArgumentTypeError(f"'{text}': give a number with one of: {units}")
        return float(m.group(1)) * UNITS[kind][unit]
    return parse


# --------------------------------------------------------------------------- model

def fft_len(n):
    return 1 << int(math.ceil(math.log2(2 * n)))  # >= 2n: filtering doesn't wrap round


def kanai_tajimi(f, fg, zeta):
    """Amplitude of the Kanai-Tajimi ground filter: a soil layer resonating at fg."""
    r2 = (f / fg) ** 2
    return np.sqrt((1 + 4 * zeta * zeta * r2) / ((1 - r2) ** 2 + 4 * zeta * zeta * r2))


def butterworth(f, f_hp, f_lp, order=4):
    """Zero-phase band-pass amplitude."""
    fpos = np.where(f > 0, f, 1.0)
    h = np.where(f > 0, 1.0 / np.sqrt(1 + (f_hp / fpos) ** (2 * order)), 0.0)
    if f_lp > 0:
        h = h / np.sqrt(1 + (f / f_lp) ** (2 * order))
    return h


def envelope(t, duration, rise, strong, end_level=0.05):
    """Build-up (t/rise)^2, strong shaking at 1, then exponential decay to end_level."""
    e = np.ones_like(t)
    up = t < rise
    e[up] = (t[up] / rise) ** 2
    t2 = rise + strong
    if duration > t2:
        c = -math.log(end_level) / (duration - t2)
        down = t > t2
        e[down] = np.exp(-c * (t[down] - t2))
    return e


def smooth_fade(n, m):
    """1 in the middle; ends rise with zero slope and zero curvature (quintic smoothstep)."""
    w = np.ones(n)
    m = min(int(m), n // 2)
    if m > 1:
        s = np.arange(m) / m
        r = s ** 3 * (10 - 15 * s + 6 * s * s)
        w[:m] = r
        w[n - m:] = r[::-1]
    return w


def synthesize(duration, dt, pga, pgv_max, pgd_max, fg, zeta, f_low, f_high, rise, strong,
               seed, limit_mode="filter"):
    n = int(round(duration / dt)) + 1
    t = np.arange(n) * dt
    rng = np.random.default_rng(seed)
    big_n = fft_len(n)
    f = np.fft.rfftfreq(big_n, dt)
    w = 2 * np.pi * f

    noise = rng.standard_normal(n)
    coloured = np.fft.irfft(np.fft.rfft(noise, big_n) * kanai_tajimi(f, fg, zeta), big_n)[:n]
    spec_a = np.fft.rfft(coloured * envelope(t, duration, rise, strong), big_n)
    fade = smooth_fade(n, round(min(0.15 * duration, max(1.0, 0.5 / f_low)) / dt))
    d1 = 1j * w
    d1[0] = 0.0
    d2 = -(w ** 2)
    inv2 = np.zeros_like(w)
    inv2[1:] = -1.0 / w[1:] ** 2

    def build(fl):
        x = np.fft.irfft(spec_a * butterworth(f, fl, f_high) * inv2, big_n)[:n] * fade
        spec_x = np.fft.rfft(x, big_n)
        v = np.fft.irfft(spec_x * d1, big_n)[:n]
        a = np.fft.irfft(spec_x * d2, big_n)[:n]
        return a, v, x

    notes = []
    fl = f_low
    fl_cap = 0.7 * fg
    while True:
        a, v, x = build(fl)
        k = pga / np.max(np.abs(a))
        pgv, pgd = k * np.max(np.abs(v)), k * np.max(np.abs(x))
        fits = (pgv_max is None or pgv <= pgv_max) and (pgd_max is None or pgd <= pgd_max)
        if fits or limit_mode == "scale" or fl * 1.15 > fl_cap:
            break
        fl *= 1.15
    a, v, x = a * k, v * k, x * k
    if fl > f_low:
        notes.append(f"f_low raised from {f_low:g} to {fl:.3g} Hz to keep PGV/PGD within the limits")
    if not fits:
        s = min(pgv_max / pgv if pgv_max else 1.0, pgd_max / pgd if pgd_max else 1.0)
        a, v, x = a * s, v * s, x * s
        notes.append(f"PGA lowered to {np.max(np.abs(a)) / G:.3f} g: the PGV/PGD limits couldn't be met "
                     f"at {pga / G:.3f} g" + (" without changing the frequency content too much"
                                               if limit_mode == "filter" else ""))
    return t, a, v, x, fl, notes


# --------------------------------------------------------------------------- measures

def arias_and_duration(a, dt):
    """Arias intensity (m/s) and 5-95 % significant duration (s)."""
    a_si = a / 1000.0
    cum = np.cumsum(a_si ** 2) * dt
    ia = math.pi / (2 * 9.80665) * cum[-1]
    i5, i95 = np.searchsorted(cum, [0.05 * cum[-1], 0.95 * cum[-1]])
    return ia, (i95 - i5) * dt


def dominant_frequency(a, dt):
    spec = np.abs(np.fft.rfft(a))
    f = np.fft.rfftfreq(len(a), dt)
    k = max(1, len(spec) // 200)
    smooth = np.convolve(spec, np.ones(k) / k, mode="same")
    return f[np.argmax(smooth)]


def response_spectrum(a, dt, periods, zeta=0.05):
    """5 %-damped pseudo-spectral acceleration (Newmark average acceleration, SDOF bank)."""
    om = 2 * np.pi / periods
    c, kk = 2 * zeta * om, om ** 2
    gam, beta = 0.5, 0.25
    k_hat = kk + gam / (beta * dt) * c + 1 / (beta * dt * dt)
    ca = 1 / (beta * dt) + gam / beta * c
    cb = 1 / (2 * beta) + dt * (gam / (2 * beta) - 1) * c
    u = np.zeros_like(om)
    ud = np.zeros_like(om)
    udd = -a[0] * np.ones_like(om)
    umax = np.zeros_like(om)
    dp = -np.diff(a)
    for i in range(len(dp)):
        du = (dp[i] + ca * ud + cb * udd) / k_hat
        dud = gam / (beta * dt) * du - gam / beta * ud + dt * (1 - gam / (2 * beta)) * udd
        dudd = du / (beta * dt * dt) - ud / (beta * dt) - udd / (2 * beta)
        u, ud, udd = u + du, ud + dud, udd + dudd
        np.maximum(umax, np.abs(u), out=umax)
    return kk * umax


def fast_spectrum(a, dt, periods, zeta=0.05):
    """Pseudo-spectral acceleration from the exact frequency response of each oscillator.

    Same result as response_spectrum (to within its time-stepping error) but much faster:
    used inside the spectrum-matching loop.
    """
    n = len(a)
    ring = int(math.ceil(4.0 * max(periods) / (2 * math.pi * zeta) / dt))  # free vibration dies out
    big_n = 1 << int(math.ceil(math.log2(n + ring)))
    om = 2 * np.pi * np.fft.rfftfreq(big_n, dt)
    spec = np.fft.rfft(a, big_n)
    out = np.empty(len(periods))
    for i in range(0, len(periods), 16):  # a few oscillators at a time keeps memory small
        w = 2 * np.pi / np.asarray(periods[i:i + 16])[:, None]
        u = np.fft.irfft(spec * (-1.0 / (w * w - om * om + 2j * zeta * w * om)), big_n, axis=1)
        out[i:i + 16] = w[:, 0] ** 2 * np.abs(u).max(axis=1)
    return out


# --------------------------------------------------------------------------- target spectra

# Eurocode 8 (EN 1998-1) Tables 3.2 and 3.3: soil factor S and corner periods TB, TC, TD (s)
EC8 = {
    1: {"A": (1.0, 0.15, 0.4, 2.0), "B": (1.2, 0.15, 0.5, 2.0), "C": (1.15, 0.20, 0.6, 2.0),
        "D": (1.35, 0.20, 0.8, 2.0), "E": (1.4, 0.15, 0.5, 2.0)},
    2: {"A": (1.0, 0.05, 0.25, 1.2), "B": (1.35, 0.05, 0.25, 1.2), "C": (1.5, 0.10, 0.25, 1.2),
        "D": (1.8, 0.10, 0.30, 1.2), "E": (1.6, 0.05, 0.25, 1.2)},
}


def ec8_spectrum(ag, ground, kind, xi):
    """EC8 elastic horizontal spectrum Se(T) (same units as ag), damping xi in %."""
    s, tb, tc, td = EC8[kind][ground]
    eta = max(math.sqrt(10.0 / (5.0 + xi)), 0.55)

    def se(T):
        T = np.maximum(np.asarray(T, float), 1e-9)
        return ag * s * np.where(T <= tb, 1 + T / tb * (2.5 * eta - 1),
                                 np.where(T <= tc, 2.5 * eta,
                                          np.where(T <= td, 2.5 * eta * tc / T, 2.5 * eta * tc * td / T ** 2)))
    desc = (f"EC8 type {kind}, ground {ground}, ag {ag / G:.3g} g (S {s:g}, TB {tb:g}, TC {tc:g}, "
            f"TD {td:g} s)")
    return se, desc


def asce7_spectrum(sds, sd1, tl):
    """ASCE 7 (11.4.6) design response spectrum Sa(T)."""
    t0, ts = 0.2 * sd1 / sds, sd1 / sds

    def sa(T):
        T = np.maximum(np.asarray(T, float), 1e-9)
        return np.where(T < t0, sds * (0.4 + 0.6 * T / t0),
                        np.where(T <= ts, sds, np.where(T <= tl, sd1 / T, sd1 * tl / T ** 2)))
    desc = f"ASCE 7, SDS {sds / G:.3g} g, SD1 {sd1 / G:.3g} g, TL {tl:g} s"
    return sa, desc


def file_spectrum(path, units):
    """Spectrum from a file: rows of period (s) and Sa, interpolated in log-log."""
    rows = []
    for line in Path(path).read_text(errors="replace").splitlines():
        parts = [p for p in re.split(r"[,;\t ]+", line.strip()) if p]
        try:
            vals = [float(p) for p in parts[:2]]
        except ValueError:
            continue  # header or text
        if len(vals) == 2 and vals[0] > 0 and vals[1] > 0:
            rows.append(vals)
    if len(rows) < 2:
        raise ValueError(f"{path}: need at least two rows of 'period, Sa' (both above zero)")
    data = np.array(sorted(rows))
    lt, ls = np.log(data[:, 0]), np.log(data[:, 1] * UNITS["accel"][units])

    def sa(T):
        return np.exp(np.interp(np.log(np.maximum(np.asarray(T, float), 1e-9)), lt, ls))
    return sa, f"{Path(path).name} ({len(data)} points, Sa in {units or 'mm/s2'})", (data[0, 0], data[-1, 0])


def match_spectrum(duration, dt, target, t_lo, t_hi, f_low, f_high, rise, strong, seed, zeta,
                   iterations=40):
    """Spectrum-compatible record: random phases, envelope, Fourier amplitudes adjusted until the
    response spectrum matches target(T) between t_lo and t_hi. Returns t, a, v, x and a summary."""
    n = int(round(duration / dt)) + 1
    t = np.arange(n) * dt
    rng = np.random.default_rng(seed)
    big_n = fft_len(n)
    f = np.fft.rfftfreq(big_n, dt)
    w = 2 * np.pi * f
    phase = np.exp(1j * np.angle(np.fft.rfft(rng.standard_normal(n), big_n)))
    env = envelope(t, duration, rise, strong)
    band = butterworth(f, f_low, f_high)
    fade = smooth_fade(n, round(min(0.15 * duration, max(1.0, 0.5 / f_low)) / dt))
    d1 = 1j * w
    inv2 = np.zeros_like(w)
    inv2[1:] = -1.0 / w[1:] ** 2
    periods = np.logspace(math.log10(t_lo), math.log10(t_hi), 60)
    goal = target(periods)
    log_p, log_f = np.log(periods), np.log(1.0 / np.maximum(f, 1e-9))

    def build(amp):
        stationary = np.fft.irfft(amp * phase, big_n)[:n]
        x = np.fft.irfft(np.fft.rfft(stationary * env, big_n) * band * inv2, big_n)[:n] * fade
        spec_x = np.fft.rfft(x, big_n)
        return np.fft.irfft(spec_x * -(w ** 2), big_n)[:n], np.fft.irfft(spec_x * d1, big_n)[:n], x

    amp = np.ones_like(f)
    best = None
    for it in range(iterations + 1):
        a, v, x = build(amp)
        ratio = goal / np.maximum(fast_spectrum(a, dt, periods, zeta), 1e-12)
        misfit = float(np.max(np.abs(np.log(ratio))))
        if best is None or misfit < best[0]:
            best = (misfit, a, v, x, ratio, it)
        if it == iterations:
            break
        # each frequency is corrected by the misfit at the matching period (end values outside the range)
        amp = amp * np.interp(log_f, log_p, ratio)
    _, a, v, x, ratio, it = best
    return t, a, v, x, {"min": float(1 / ratio.max()), "max": float(1 / ratio.min()), "iterations": it}


# --------------------------------------------------------------------------- output

def write_csv(path, t, a, v, x, header):
    lines = [f"# {h}" for h in header] + ["time_s,accel_mm_s2,vel_mm_s,disp_mm"]
    body = "\n".join(f"{ti:.4f},{ai:.4f},{vi:.5f},{xi:.6f}" for ti, ai, vi, xi in zip(t, a, v, x))
    Path(path).write_text("\n".join(lines) + "\n" + body + "\n")


def write_at2(path, a, dt, title):
    """PEER NGA format (acceleration in g), readable by most earthquake software."""
    vals = [f"{v:15.7E}" for v in a / G]
    rows = ["".join(vals[i:i + 5]) for i in range(0, len(vals), 5)]
    head = ["PEER NGA STRONG MOTION DATABASE RECORD (synthetic)", title,
            "ACCELERATION TIME SERIES IN UNITS OF G", f"NPTS= {len(a)}, DT= {dt:.4f} SEC"]
    Path(path).write_text("\n".join(head + rows) + "\n")


def draw_spectrum(ax, a, dt, spectrum=None, colors=(None, None)):
    """Response spectrum of the record; with a target spectrum, the target and the matched range too."""
    zeta = spectrum["zeta"] if spectrum else 0.05
    periods = np.logspace(np.log10(0.05), np.log10(5.0), 80)
    ax.semilogx(periods, response_spectrum(a, dt, periods, zeta) / G, color=colors[0], lw=1.4, label="record")
    if spectrum:
        lo, hi = spectrum["range"]
        ax.axvspan(lo, hi, color="0.5", alpha=0.08, lw=0)
        ax.semilogx(periods, spectrum["fn"](periods) / G, color=colors[1], lw=1.2, ls=(0, (4, 2)),
                    label="target")
        ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.set_xlabel("period (s)")
    ax.set_ylabel(f"spectral acceleration, {100 * zeta:g} % damping (g)")


def write_plot(path, t, a, v, x, dt, title, spectrum=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping --plot", file=sys.stderr)
        return False
    fig = plt.figure(figsize=(11, 8))
    gs = fig.add_gridspec(3, 2, width_ratios=[2.2, 1])
    axes = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    for ax, y, label in zip(axes, (a / G, v, x), ("acceleration (g)", "velocity (mm/s)", "displacement (mm)")):
        ax.plot(t, y, lw=0.8)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[2].set_xlabel("time (s)")
    sp = fig.add_subplot(gs[:, 1])
    draw_spectrum(sp, a, dt, spectrum)
    sp.grid(alpha=0.3, which="both")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True


# --------------------------------------------------------------------------- command line

def make_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="Units: 0.1g, 980mm/s2, 98gal | 100mm/s, 10cm/s | 40mm, 4cm. "
                                       "Bare numbers are mm-based. 'none' switches a limit off.")
    g = p.add_argument_group("record defined by peak values")
    g.add_argument("--duration", type=float, default=20.0, help="record length, s (default 20)")
    g.add_argument("--pga", type=quantity("accel"),
                   help="peak acceleration the record reaches (default 0.1g)")
    g.add_argument("--pgv-max", type=quantity("vel", True), default=120.0,
                   help="upper limit for peak velocity (default 120mm/s)")
    g.add_argument("--pgd-max", type=quantity("disp", True), default=60.0,
                   help="upper limit for peak displacement (default 60mm)")
    g.add_argument("--limit-mode", choices=["filter", "scale"], default="filter",
                   help="meet PGV/PGD limits by raising f_low (keeps PGA, default) or by scaling "
                        "the whole record down (keeps frequency content)")
    g = p.add_argument_group("record defined by a target response spectrum (instead of the peak values)")
    g.add_argument("--spectrum",
                   help="'ec8' (Eurocode 8), 'asce7', or a CSV/TXT file with columns period (s), Sa")
    g.add_argument("--ag", type=quantity("accel"), default=0.1 * G,
                   help="EC8: design ground acceleration on type A ground, ag = gamma_I * agR (default 0.1g)")
    g.add_argument("--ground", choices=list("ABCDE"), default="C", help="EC8 ground type (default C)")
    g.add_argument("--ec8-type", type=int, choices=[1, 2], default=1,
                   help="EC8 spectrum type: 1 (larger, distant earthquakes, Ms > 5.5) or 2 (default 1)")
    g.add_argument("--sds", type=quantity("accel"), help="ASCE 7: SDS, e.g. 0.5g")
    g.add_argument("--sd1", type=quantity("accel"), help="ASCE 7: SD1, e.g. 0.2g")
    g.add_argument("--tl", type=float, default=8.0, help="ASCE 7: long-period transition TL, s (default 8)")
    g.add_argument("--spectrum-units", default="g", help="units of Sa in a spectrum file (default g)")
    g.add_argument("--periods", type=float, nargs=2, metavar=("T_MIN", "T_MAX"), default=[0.1, 2.0],
                   help="period range to match, s (default 0.1 2)")
    g.add_argument("--spectrum-damping", type=float, default=5.0,
                   help="damping of the target spectrum, %% (default 5)")
    g.add_argument("--iterations", type=int, default=40, help="matching iterations (default 40)")
    g = p.add_argument_group("character of the shaking")
    g.add_argument("--f-dominant", type=float, default=2.5,
                   help="dominant (ground) frequency, Hz (default 2.5)")
    g.add_argument("--damping", type=float, default=0.6, help="ground damping ratio (default 0.6)")
    g.add_argument("--f-low", type=float, default=0.2,
                   help="high-pass corner, Hz: removes slow drift (default 0.2)")
    g.add_argument("--f-high", type=float, default=15.0,
                   help="low-pass corner, Hz (default 15, about what the printer can follow)")
    g.add_argument("--rise", type=float, help="build-up time, s (default 15 %% of the duration)")
    g.add_argument("--strong", type=float, help="strong-shaking time, s (default 35 %% of the duration)")
    g = p.add_argument_group("output")
    g.add_argument("-o", "--output", default="records/synthetic.csv",
                   help="CSV file (default records/synthetic.csv; numbered when --count > 1)")
    g.add_argument("--count", type=int, default=1, help="how many records (different seeds)")
    g.add_argument("--seed", type=int, help="random seed, for repeatable records (default: random)")
    g.add_argument("--dt", type=float, default=0.005, help="sample interval, s (default 0.005)")
    g.add_argument("--at2", action="store_true", help="also write a PEER .AT2 file (acceleration in g)")
    g.add_argument("--plot", action="store_true", help="also write a PNG with the record and its spectrum")
    return p


def check_settings(p, args):
    """Reject impossible settings (through p.error); return the envelope times (rise, strong) in s."""
    if args.duration < 2:
        p.error("--duration must be at least 2 s")
    if not 0 < args.dt <= 0.02:
        p.error("--dt must be above 0 and at most 0.02 s")
    if args.spectrum:
        if not 0 < args.f_low < args.f_high:
            p.error("need 0 < --f-low < --f-high")
        check_spectrum(p, args)
    elif not 0 < args.f_low < args.f_dominant < args.f_high:
        p.error("need 0 < --f-low < --f-dominant < --f-high")
    if args.pga is None:
        args.pga = 0.1 * G
    if args.f_high >= 0.4 / args.dt:
        p.error(f"--f-high must be below {0.4 / args.dt:g} Hz for --dt {args.dt:g}")
    if not 0 < args.damping < 5:
        p.error("--damping must be between 0 and 5")
    rise = args.rise if args.rise is not None else 0.15 * args.duration
    strong = args.strong if args.strong is not None else 0.35 * args.duration
    if rise <= 0 or strong < 0 or rise + strong > args.duration:
        p.error("--rise must be > 0 and --rise + --strong at most --duration")
    return rise, strong


def check_spectrum(p, args):
    """Build the target spectrum from the settings: sets args.target (function of T), args.target_desc."""
    kind = args.spectrum.strip()
    t_lo, t_hi = args.periods
    if not 0 < t_lo < t_hi:
        p.error("--periods: need 0 < T_MIN < T_MAX")
    if t_lo < 1.0 / args.f_high:
        p.error(f"--periods: T_MIN must be at least 1/--f-high = {1 / args.f_high:.3g} s (the record has "
                f"no content above --f-high)")
    if not 0 < args.spectrum_damping < 50:
        p.error("--spectrum-damping must be between 0 and 50 %")
    if args.iterations < 1:
        p.error("--iterations must be at least 1")
    if kind.lower() == "ec8":
        if args.ag <= 0:
            p.error("--ag must be above zero")
        args.target, args.target_desc = ec8_spectrum(args.ag, args.ground, args.ec8_type, args.spectrum_damping)
    elif kind.lower() == "asce7":
        if not args.sds or not args.sd1:
            p.error("ASCE 7 needs --sds and --sd1 (both above zero)")
        args.target, args.target_desc = asce7_spectrum(args.sds, args.sd1, args.tl)
    else:
        units = args.spectrum_units.strip().lower().replace("^", "")
        if units not in UNITS["accel"]:
            p.error(f"--spectrum-units: one of {', '.join(u for u in UNITS['accel'] if u)}")
        try:
            args.target, args.target_desc, (f_lo, f_hi) = file_spectrum(kind, units)
        except OSError:
            p.error(f"--spectrum: can't read '{kind}' (give ec8, asce7 or a spectrum file)")
        except ValueError as e:
            p.error(f"--spectrum: {e}")
        if t_lo < f_lo * 0.999 or t_hi > f_hi * 1.001:
            p.error(f"--periods: the file covers {f_lo:g}-{f_hi:g} s; match inside that range")


def make_record(args, seed, rise, strong):
    """Synthesize one record. Returns a dict: t, a, v, x (mm-based), seed, peaks and the header lines."""
    spectrum = None
    if args.spectrum:
        t_lo, t_hi = args.periods
        fl = min(args.f_low, 0.5 / t_hi)  # keep the content the longest matched period needs
        notes = [f"f_low lowered from {args.f_low:g} to {fl:.3g} Hz to reach periods up to {t_hi:g} s"] \
            if fl < args.f_low else []
        zeta = args.spectrum_damping / 100
        t, a, v, x, info = match_spectrum(args.duration, args.dt, args.target, t_lo, t_hi, fl, args.f_high,
                                          rise, strong, seed, zeta, args.iterations)
        spectrum = {"fn": args.target, "range": (t_lo, t_hi), "zeta": zeta, "desc": args.target_desc,
                    "min": info["min"], "max": info["max"]}
        model = (f"model: spectrum-matched (random phases, {args.iterations} iterations); band "
                 f"{fl:.3g}-{args.f_high:g} Hz; envelope rise {rise:g} s, strong {strong:g} s")
        notes = [f"target spectrum: {args.target_desc}, {args.spectrum_damping:g} % damping",
                 f"matched {t_lo:g}-{t_hi:g} s: record spectrum {100 * (info['min'] - 1):+.0f} % to "
                 f"{100 * (info['max'] - 1):+.0f} % of the target"] + notes
    else:
        t, a, v, x, fl, notes = synthesize(args.duration, args.dt, args.pga, args.pgv_max, args.pgd_max,
                                           args.f_dominant, args.damping, args.f_low, args.f_high,
                                           rise, strong, seed, args.limit_mode)
        model = (f"model: Kanai-Tajimi, dominant frequency {args.f_dominant:g} Hz, damping {args.damping:g}; "
                 f"band {fl:.3g}-{args.f_high:g} Hz; envelope rise {rise:g} s, strong {strong:g} s")
    pga, pgv, pgd = np.max(np.abs(a)), np.max(np.abs(v)), np.max(np.abs(x))
    ia, d595 = arias_and_duration(a, args.dt)
    header = [
        f"Synthetic ground motion record, synthesize_record.py, {_dt.datetime.now():%Y-%m-%d %H:%M}",
        f"seed {seed}, duration {args.duration:g} s, dt {args.dt:g} s",
        model,
        f"PGA {pga:.1f} mm/s^2 ({pga / G:.4f} g), PGV {pgv:.2f} mm/s, PGD {pgd:.3f} mm "
        f"(stroke {x.max() - x.min():.2f} mm)",
        f"Arias intensity {ia:.4f} m/s, significant duration (5-95 %) {d595:.2f} s, "
        f"strongest frequency {dominant_frequency(a, args.dt):.2f} Hz",
    ] + notes
    return {"t": t, "a": a, "v": v, "x": x, "dt": args.dt, "seed": seed, "pga": pga, "pgv": pgv,
            "pgd": pgd, "stroke": float(x.max() - x.min()), "notes": notes, "header": header,
            "spectrum": spectrum}


def save_record(path, rec, at2=False, plot=False):
    """Write the CSV (and optionally the .AT2 and PNG) for a record from make_record."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(path, rec["t"], rec["a"], rec["v"], rec["x"], rec["header"])
    print(f"Wrote {path}")
    for h in rec["header"][1:]:
        print("  " + h)
    if at2:
        write_at2(path.with_suffix(".AT2"), rec["a"], rec["dt"], f"{path.stem}, seed {rec['seed']}")
        print(f"Wrote {path.with_suffix('.AT2')}")
    if plot and write_plot(path.with_suffix(".png"), rec["t"], rec["a"], rec["v"], rec["x"], rec["dt"],
                           f"{path.stem}: PGA {rec['pga'] / G:.3f} g, PGV {rec['pgv']:.0f} mm/s, "
                           f"PGD {rec['pgd']:.1f} mm", rec.get("spectrum")):
        print(f"Wrote {path.with_suffix('.png')}")


def main(argv=None):
    p = make_parser()
    args = p.parse_args(argv)
    rise, strong = check_settings(p, args)
    base_seed = args.seed if args.seed is not None else secrets.randbelow(10 ** 6)
    out = Path(args.output)
    for i in range(args.count):
        path = out if args.count == 1 else out.with_name(f"{out.stem}_{i + 1:02d}{out.suffix}")
        save_record(path, make_record(args, base_seed + i, rise, strong), args.at2, args.plot)
    return 0


if __name__ == "__main__":
    sys.exit(main())

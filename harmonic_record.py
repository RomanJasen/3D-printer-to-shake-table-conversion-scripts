#!/usr/bin/env python3
"""
harmonic_record.py - make harmonic (sine) records, optionally sweeping frequency and/or amplitude.

Step 1 of the shake-table workflow, next to synthesize_record.py (earthquake-like records).
Step 2 is record_to_gcode.py.

  python harmonic_record.py --define disp --amplitude 5mm --freq 2 --duration 20
  python harmonic_record.py --define accel --amplitude 0.05g --freq 1 --freq-end 8 --duration 60
  python harmonic_record.py --define vel --amplitude 20mm/s --amplitude-end 80mm/s --freq 3 --duration 30

--define says which quantity the amplitude is given for: displacement, velocity or acceleration.
A sweep runs from the start value to the end value over the whole duration:
  frequency: log (equal time per octave, the usual shake-table sweep) or linear (constant Hz/s),
  amplitude: linear or log (constant ratio per second).
Frequency and amplitude can sweep at the same time. The motion fades in and out smoothly
(--ramp), so the bed starts and ends at rest.

How it is built: the displacement is x(t) = fade(t) * D(t) * sin(phase(t)), with D = A for
displacement, A/w for velocity and A/w^2 for acceleration (w = 2 pi f(t)). Velocity and
acceleration are its exact derivatives, so the three always match and nothing drifts. During
a sweep the amplitude of the chosen quantity is then held to within a few per cent; the file
header says how close it came.

Output CSV columns: time_s, accel_mm_s2, vel_mm_s, disp_mm (the same as synthesize_record.py).
Values take units: 0.05g, 98gal, 490mm/s2 | 30mm/s, 3cm/s | 5mm, 0.5cm. A bare number is mm-based.
For sines above about 8 Hz convert with record_to_gcode.py --lowpass 0, so the converter's
default 15 Hz filter doesn't trim the amplitude.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import math
import sys
from pathlib import Path

import numpy as np

import synthesize_record as syn

G = syn.G
NAMES = {"disp": "displacement", "vel": "velocity", "accel": "acceleration"}
POWER = {"disp": 0, "vel": 1, "accel": 2}   # D = A / w^POWER
UNIT = {"disp": "mm", "vel": "mm/s", "accel": "g"}
DEFAULT_AMP = {"disp": "5mm", "vel": "30mm/s", "accel": "0.05g"}
DEFAULT_AMP_END = {"disp": "15mm", "vel": "90mm/s", "accel": "0.15g"}


def show_amp(kind, value):
    """Amplitude in the unit people think in: g for acceleration, mm(/s) otherwise."""
    return f"{value / G:.4g} g" if kind == "accel" else f"{value:.4g} {UNIT[kind]}"


def tag_amp(kind, value):
    """Amplitude for a file name."""
    return f"{value / G:.3g}g" if kind == "accel" else f"{value:.3g}" + ("mmps" if kind == "vel" else "mm")


# --------------------------------------------------------------------------- signal

def smooth_step(s):
    """0 -> 1 with zero slope and zero curvature at both ends (quintic)."""
    s = np.clip(s, 0.0, 1.0)
    return s ** 3 * (10 - 15 * s + 6 * s * s)


def frequency_and_phase(t, duration, f0, f1, law):
    if f1 is None or f1 == f0:
        return np.full_like(t, f0), 2 * np.pi * f0 * t
    s = t / duration
    if law == "linear":
        return f0 + (f1 - f0) * s, 2 * np.pi * (f0 * t + (f1 - f0) * t * t / (2 * duration))
    k = math.log(f1 / f0)  # exponential (log) sweep: the same time for every octave
    return f0 * np.exp(k * s), 2 * np.pi * f0 * duration * np.expm1(k * s) / k


def amplitude(t, duration, a0, a1, law):
    if a1 is None or a1 == a0:
        return np.full_like(t, a0)
    s = t / duration
    return a0 + (a1 - a0) * s if law == "linear" else a0 * (a1 / a0) ** s


def analytic_envelope(y, n_fft):
    """Instantaneous amplitude (magnitude of the analytic signal)."""
    n = len(y)
    h = np.zeros(n_fft)
    h[0] = h[n_fft // 2] = 1.0
    h[1:n_fft // 2] = 2.0
    return np.abs(np.fft.ifft(np.fft.fft(y, n_fft) * h)[:n])


def harmonic(duration, dt, define, a0, a1, a_law, f0, f1, f_law, ramp_in, ramp_out):
    """Returns t, a, v, x (mm-based), frequency f(t), the set amplitude and the amplitude reached
    (both of the defined quantity), and the fade."""
    n = int(round(duration / dt)) + 1
    t = np.arange(n) * dt
    duration = t[-1]
    f, phase = frequency_and_phase(t, duration, f0, f1, f_law)
    amp = amplitude(t, duration, a0, a1, a_law)
    fade = np.ones(n)
    if ramp_in > 0:
        fade *= smooth_step(t / ramp_in)
    if ramp_out > 0:
        fade *= smooth_step((duration - t) / ramp_out)
    x = fade * amp / (2 * np.pi * f) ** POWER[define] * np.sin(phase)
    # exact derivatives: x and its first two derivatives are zero at both ends, so the
    # zero-padded spectrum has no edge effects
    big_n = syn.fft_len(n)
    w = 2 * np.pi * np.fft.rfftfreq(big_n, dt)
    spec = np.fft.rfft(x, big_n)
    v = np.fft.irfft(spec * 1j * w, big_n)[:n]
    a = np.fft.irfft(spec * -(w ** 2), big_n)[:n]
    reached = analytic_envelope({"disp": x, "vel": v, "accel": a}[define], big_n)
    return t, a, v, x, f, fade * amp, reached, fade


# --------------------------------------------------------------------------- record

def make_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="Units: 0.05g, 98gal, 490mm/s2 | 30mm/s, 3cm/s | 5mm, 0.5cm. "
                                       "Bare numbers are mm-based.")
    g = p.add_argument_group("signal")
    g.add_argument("--define", choices=["disp", "vel", "accel"], default="disp",
                   help="which quantity --amplitude is given for (default disp)")
    g.add_argument("--amplitude", help="amplitude of that quantity (default 5mm, 30mm/s or 0.05g)")
    g.add_argument("--freq", type=float, default=2.0, help="frequency, Hz (default 2)")
    g.add_argument("--duration", type=float, default=20.0, help="length, s (default 20)")
    g.add_argument("--ramp", type=float,
                   help="fade-in and fade-out time, s (default 2 cycles, at most 20 %% of the duration)")
    g = p.add_argument_group("sweeps (each runs over the whole duration)")
    g.add_argument("--freq-end", type=float, help="sweep the frequency to this, Hz")
    g.add_argument("--freq-law", choices=["log", "linear"], default="log",
                   help="log = equal time per octave (default), linear = constant Hz per second")
    g.add_argument("--amplitude-end", help="sweep the amplitude to this (same quantity as --amplitude)")
    g.add_argument("--amplitude-law", choices=["linear", "log"], default="linear",
                   help="linear (default) or log = constant ratio per second")
    g = p.add_argument_group("output")
    g.add_argument("-o", "--output", help="CSV file (default records/<description>.csv)")
    g.add_argument("--dt", type=float, default=0.005, help="sample interval, s (default 0.005)")
    g.add_argument("--at2", action="store_true", help="also write a PEER .AT2 file (acceleration in g)")
    g.add_argument("--plot", action="store_true", help="also write a PNG of the record")
    return p


def check_settings(p, args):
    """Parse the amplitudes, reject impossible settings (through p.error); return the fade times."""
    kind = args.define

    def amp(text, opt):
        try:
            return syn.quantity(kind)(text)
        except argparse.ArgumentTypeError as e:
            p.error(f"{opt}: {e}")

    args.amp0 = amp(args.amplitude or DEFAULT_AMP[kind], "--amplitude")
    args.amp1 = amp(args.amplitude_end, "--amplitude-end") if args.amplitude_end else None
    if args.amp0 <= 0 or (args.amp1 is not None and args.amp1 <= 0):
        p.error("--amplitude and --amplitude-end must be above zero")
    if not 0 < args.dt <= 0.02:
        p.error("--dt must be above 0 and at most 0.02 s")
    if args.duration < 1:
        p.error("--duration must be at least 1 s")
    f_top = 0.4 / args.dt
    for opt, f in (("--freq", args.freq), ("--freq-end", args.freq_end)):
        if f is not None and not 0 < f <= f_top:
            p.error(f"{opt} must be above 0 and at most {f_top:g} Hz with --dt {args.dt:g}")
    f_end = args.freq_end or args.freq
    if args.ramp is None:
        return min(2 / args.freq, 0.2 * args.duration), min(2 / f_end, 0.2 * args.duration)
    if not 0 <= 2 * args.ramp <= args.duration:
        p.error("--ramp must be between 0 and half the duration")
    return args.ramp, args.ramp


def make_record(args, ramp_in, ramp_out):
    """One harmonic record as a dict (same keys as synthesize_record.make_record, plus extras)."""
    kind, a0, a1, f0, f1 = args.define, args.amp0, args.amp1, args.freq, args.freq_end
    t, a, v, x, f, amp_set, amp_got, fade = harmonic(
        args.duration, args.dt, kind, a0, a1, args.amplitude_law, f0, f1, args.freq_law, ramp_in, ramp_out)
    pga, pgv, pgd = float(np.max(np.abs(a))), float(np.max(np.abs(v))), float(np.max(np.abs(x)))
    sweep_f = f1 is not None and f1 != f0
    sweep_a = a1 is not None and a1 != a0
    f_txt = f"{f0:g} -> {f1:g} Hz ({args.freq_law} sweep)" if sweep_f else f"{f0:g} Hz"
    a_txt = (f"{show_amp(kind, a0)} -> {show_amp(kind, a1)} ({args.amplitude_law} sweep)" if sweep_a
             else show_amp(kind, a0))
    shape = {(False, False): "sine", (True, False): "frequency sweep", (False, True): "amplitude sweep",
             (True, True): "frequency and amplitude sweep"}[(sweep_f, sweep_a)]
    header = [
        f"Harmonic record, harmonic_record.py, {_dt.datetime.now():%Y-%m-%d %H:%M}",
        f"{shape}: {NAMES[kind]} amplitude {a_txt}; frequency {f_txt}",
        f"duration {t[-1]:g} s, fade in {ramp_in:.3g} s and out {ramp_out:.3g} s, dt {args.dt:g} s",
        f"peak acceleration {pga:.1f} mm/s^2 ({pga / G:.4f} g), peak velocity {pgv:.2f} mm/s, "
        f"peak displacement {pgd:.3f} mm (stroke {x.max() - x.min():.2f} mm)",
    ]
    inside = fade > 0.999
    if inside.sum() > 10:
        dev = float(np.max(np.abs(amp_got[inside] / amp_set[inside] - 1)))
        header.append(f"{NAMES[kind]} amplitude reached: within {100 * dev:.1f} % of the set value "
                      f"(outside the fades)")
    f_max = float(f.max())
    disp_amp = amp_set / (2 * np.pi * f) ** POWER[kind]  # displacement amplitude over time
    smallest = float(disp_amp[inside].min()) if inside.any() else pgd
    if smallest < 0.3:
        header.append(f"displacement amplitude down to {smallest:.2f} mm: the firmware's smallest move is "
                      f"about 0.1 mm, so that part will be coarse or won't play")
    if f_max > 15:
        header.append("above about 15 Hz the bed can't follow well")
    if f_max > 8:
        header.append("convert with Low-pass 0 (--lowpass 0) so the 15 Hz filter doesn't trim the amplitude")
    a_short = (f"{show_amp(kind, a0)} to {show_amp(kind, a1)} ({args.amplitude_law})" if sweep_a
               else show_amp(kind, a0))
    title = (f"{'Sweep' if sweep_f else 'Sine'} {f0:g}" + (f" to {f1:g} Hz ({args.freq_law})" if sweep_f else " Hz")
             + f", {NAMES[kind]} amplitude {a_short}")
    name = ("sweep" if sweep_f else "sine") + f"_{f0:g}" + (f"-{f1:g}" if sweep_f else "") + "Hz_" \
        + tag_amp(kind, a0) + (f"-{tag_amp(kind, a1)}" if sweep_a else "") + f"_{t[-1]:g}s"
    return {"type": "harmonic", "t": t, "a": a, "v": v, "x": x, "dt": args.dt, "seed": None,
            "pga": pga, "pgv": pgv, "pgd": pgd, "stroke": float(x.max() - x.min()), "header": header,
            "f": f, "amp_set": amp_set, "amp_got": amp_got, "define": kind, "f_max": f_max,
            "title": title, "name": name}


def draw_record(fig, rec, colors=None):
    """Time histories on the left; frequency and amplitude against time on the right."""
    c1, c2 = colors or (None, None)
    t, kind = rec["t"], rec["define"]
    gs = fig.add_gridspec(3, 2, width_ratios=[2.3, 1])
    axes = [fig.add_subplot(gs[0, 0])]
    axes += [fig.add_subplot(gs[i, 0], sharex=axes[0]) for i in (1, 2)]
    for ax, y, label in zip(axes, (rec["a"] / G, rec["v"], rec["x"]),
                            ("acceleration (g)", "velocity (mm/s)", "displacement (mm)")):
        ax.plot(t, y, color=c1, lw=0.8)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    for ax in axes[:2]:
        ax.tick_params(labelbottom=False)
    axes[2].set_xlabel("time (s)")
    right = gs[:, 1].subgridspec(2, 1)
    af = fig.add_subplot(right[0], sharex=axes[0])
    af.plot(t, rec["f"], color=c1, lw=1.6)
    af.set_ylim(0, 1.1 * rec["f_max"])
    af.set_ylabel("frequency (Hz)")
    af.tick_params(labelbottom=False)
    af.grid(alpha=0.3)
    am = fig.add_subplot(right[1], sharex=axes[0])
    k = 1 / G if kind == "accel" else 1.0
    am.plot(t, rec["amp_set"] * k, color=c1, lw=1.6, label="set")
    am.plot(t, rec["amp_got"] * k, color=c2, lw=1.0, ls=(0, (4, 2)), label="reached")
    am.set_ylim(0, 1.15 * k * max(rec["amp_set"].max(), rec["amp_got"].max()))
    am.set_ylabel(f"{NAMES[kind]} amplitude ({UNIT[kind]})")
    am.set_xlabel("time (s)")
    am.legend(loc="lower center", frameon=False, fontsize=8, ncols=2)
    am.grid(alpha=0.3)
    fig.suptitle(rec["title"])


def write_plot(path, rec):
    try:
        from matplotlib.figure import Figure
    except ImportError:
        print("matplotlib is not installed; skipping --plot", file=sys.stderr)
        return False
    fig = Figure(figsize=(11, 8), layout="constrained")
    draw_record(fig, rec)
    fig.savefig(path, dpi=110)
    return True


def save_record(path, rec, at2=False, plot=False):
    """Write the CSV (and optionally the .AT2 and PNG) for a record from make_record."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    syn.write_csv(path, rec["t"], rec["a"], rec["v"], rec["x"], rec["header"])
    print(f"Wrote {path}")
    for h in rec["header"][1:]:
        print("  " + h)
    if at2:
        syn.write_at2(path.with_suffix(".AT2"), rec["a"], rec["dt"], rec["title"])
        print(f"Wrote {path.with_suffix('.AT2')}")
    if plot and write_plot(path.with_suffix(".png"), rec):
        print(f"Wrote {path.with_suffix('.png')}")


def main(argv=None):
    p = make_parser()
    args = p.parse_args(argv)
    ramp_in, ramp_out = check_settings(p, args)
    rec = make_record(args, ramp_in, ramp_out)
    save_record(args.output or Path("records") / f"{rec['name']}.csv", rec, args.at2, args.plot)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
record_to_gcode.py - turn a ground-motion record into G-code for the Ender 3 S1 Pro bed.

Step 2 of the shake-table workflow. Step 1 is synthesize_record.py (earthquake-like records)
or harmonic_record.py (sines and sweeps; above ~8 Hz convert those with --lowpass 0).

  python record_to_gcode.py records/synthetic.csv                 check, then write records/synthetic.gcode
  python record_to_gcode.py records/synthetic.csv --fit --plot    shrink the record if it doesn't fit
  python record_to_gcode.py RSN1234.AT2 --fit -o gcode/rsn1234.gcode   real records work too

First it checks the record against the bed: travel (stroke), peak speed and peak
acceleration. If anything is too big it stops and says by how much, unless you pass
--fit (scale the amplitude down until everything fits) or --scale. Then it writes G-code
that compensates for how the stock firmware plans moves, and reports how closely the bed
is predicted to follow. Print the .gcode from the SD card.

Inputs: CSV from synthesize_record.py (uses its displacement column), other CSV/TXT
tables (--type accel|vel|disp, --units, --column), PEER .AT2/.VT2/.DT2 files.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

import shake_engine as eng

UNIT_NAMES = {"g": "g", "m_s2": "m/s2", "cm_s2": "cm/s2", "mm_s2": "mm/s2", "gal": "gal",
              "m_s": "m/s", "cm_s": "cm/s", "mm_s": "mm/s", "m": "m", "cm": "cm", "mm": "mm"}
COLUMN_KINDS = (("disp", "disp"), ("vel", "vel"), ("acc", "accel"))  # preferred first


def named_columns(path):
    """Find a header like 'time_s,accel_mm_s2,vel_mm_s,disp_mm' and pick the best column.

    Displacement is preferred: it is exactly what the bed does, so nothing has to be
    integrated or filtered.
    """
    for line in Path(path).read_text(errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        names = [n.strip().lower() for n in re.split(r"[,;\t]", s)]
        if not any(re.search(r"[a-z]", n) for n in names):
            return None
        for prefix, kind in COLUMN_KINDS:
            for i, name in enumerate(names):
                m = re.fullmatch(prefix + r"[a-z]*_([a-z0-9_]+)", name)
                if m and m.group(1) in UNIT_NAMES:
                    return i, kind, UNIT_NAMES[m.group(1)]
        return None
    return None


def build_target(args):
    """Read the record and turn it into a displacement history on a 1 ms grid."""
    path = Path(args.input)
    if path.suffix.lower() in eng.PEER_EXTENSIONS:
        t, y, kind, units = eng.load_peer(path)
        kind, units = args.type or kind, args.units or units
    else:
        found = None if args.type else named_columns(path)
        if found:
            col, kind, units = found
            if args.column is not None:
                col = args.column
        else:
            if not args.type:
                raise SystemExit("error: can't tell what the file contains; add --type accel|vel|disp "
                                 "(and --units, --column)")
            kind, units, col = args.type, args.units or eng.DEFAULT_UNITS[args.type], args.column
        units = args.units or units
        t, y = eng.load_table(path, args.dt, 0, col)
    if units not in eng.UNIT_FACTORS[kind]:
        raise SystemExit(f"error: units '{units}' don't fit {kind}; use one of "
                         f"{', '.join(eng.UNIT_FACTORS[kind])}")
    if args.trim:
        sel = (t >= args.trim[0]) & (t <= args.trim[1])
        t, y = t[sel] - args.trim[0], y[sel]
    dt = float(t[1] - t[0]) * args.time_scale
    y = y * eng.UNIT_FACTORS[kind][units] * args.scale
    f_hp = args.highpass if args.highpass is not None else (0.0 if kind == "disp" else 0.2)
    f_lp = args.lowpass if args.lowpass and args.lowpass < 0.5 / dt else 0.0
    tt, x, dt_f = eng.to_displacement(y, dt, kind, f_hp, f_lp, 0.001)  # 1 ms grid
    # drop quiet lead-in / tail (e.g. pre-event noise) where the bed would move < 0.5 % of its peak
    tt, x = eng.trim_quiet(tt, x, thr=max(0.02, 0.005 * np.max(np.abs(x - x[0]))))
    tgt = eng.Target(tt, x, dt_f, f"{path.name} ({kind} column, {units})")
    proc = [f"high-pass {f_hp:g} Hz" if f_hp > 0 else "no high-pass",
            f"low-pass {f_lp:g} Hz" if f_lp else "no low-pass"]
    if args.scale != 1:
        proc.append(f"amplitude x{args.scale:g}")
    if args.time_scale != 1:
        proc.append(f"time x{args.time_scale:g}")
    tgt.notes.append(f"source       : {path.name}, {kind} in {units}, {len(y)} samples at dt="
                     f"{dt / args.time_scale:g} s ({', '.join(proc)})")
    return tgt.derive()


def bounds(tgt, y_min, y_max, max_vel, max_accel):
    """The record's peaks against the bed's limits.

    Returns (rows, factor, binding): rows of (name, value, limit, unit), the amplitude factor
    that would make the tightest limit exactly 100 % used, and the name of that limit.
    """
    stroke = float(tgt.x.max() - tgt.x.min())
    vpk, apk = float(np.max(np.abs(tgt.v))), float(np.max(np.abs(tgt.a)))
    rows = (("bed travel (stroke)", stroke, y_max - y_min, "mm"),
            ("peak velocity", vpk, max_vel, "mm/s"),
            ("peak acceleration", apk, max_accel, "mm/s^2"))
    factor, binding = min((limit / max(value, 1e-9), name) for name, value, limit, _ in rows)
    return rows, factor, binding


def bounds_table(tgt, y_min, y_max, max_vel, max_accel):
    """Print the record's peaks against the bed's limits; return (fit factor, binding limit)."""
    rows, factor, binding = bounds(tgt, y_min, y_max, max_vel, max_accel)
    print(f"  {'':22}{'record':>14}{'limit':>14}{'used':>8}")
    for name, value, limit, unit in rows:
        used = value / limit
        flag = "OK" if used <= 1.0 else "TOO BIG"
        print(f"  {name:22}{value:9.1f} {unit:<6}{limit:7.0f} {unit:<6}{used:6.0%}  {flag}")
    return factor, binding


def engine_settings(args, prof):
    """Fill in the settings shake_engine.run_motion expects that aren't on the command line."""
    args.dt_min = args.dt_min or prof.dt_min
    args.dt_max = max(args.dt_max or prof.dt_max, args.dt_min)
    args.min_move = max(args.min_move or prof.min_move, (prof.min_steps + 1.5) / prof.steps_per_mm)
    args.v_quiet = max(prof.v_quiet, 1.2 * prof.ramp_floor)
    args.jerk = prof.jerk_max
    if args.reversal_jerk is not None:
        args.reversal_jerk = min(max(args.reversal_jerk, 0.1), args.jerk)
    args.iterations, args.pre_dwell, args.preposition_speed = 60, 3.0, 20.0
    args.autoscale = False
    args.max_vel = min(args.max_vel, prof.max_feedrate)
    return args


def make_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="record: CSV from synthesize_record.py, CSV/TXT table or PEER .AT2/.VT2/.DT2")
    p.add_argument("-o", "--output", help="G-code file (default: the record's name with .gcode)")
    g = p.add_argument_group("bed limits (Ender 3 S1 Pro, stock firmware)")
    g.add_argument("--y-min", type=float, default=5.0, help="lowest bed Y the motion may use, mm (default 5)")
    g.add_argument("--y-max", type=float, default=215.0, help="highest bed Y the motion may use, mm (default 215)")
    g.add_argument("--center", type=float, help="bed Y at the middle of the motion (default: middle of the range)")
    g.add_argument("--max-vel", type=float, default=150.0, help="speed limit, mm/s (default 150)")
    g.add_argument("--max-accel", type=float, default=1500.0,
                   help="acceleration limit, mm/s^2 (default 1500; the stock firmware tracks less well above)")
    g = p.add_argument_group("scaling")
    g.add_argument("--fit", action="store_true",
                   help="if the record is too big, shrink its amplitude until it fits (to 98 %% of the tightest limit)")
    g.add_argument("--scale", type=float, default=1.0, help="multiply the amplitude by this first")
    g.add_argument("--time-scale", type=float, default=1.0,
                   help="stretch time by this (>1 slower: speed /k, acceleration /k^2, displacement unchanged)")
    g.add_argument("--force", action="store_true", help="write G-code even if the record doesn't fit")
    g = p.add_argument_group("reading the input")
    g.add_argument("--type", choices=["accel", "vel", "disp"], help="what the file contains (auto for our CSV)")
    g.add_argument("--units", help="g, m/s2, cm/s2, gal, mm/s2 | m/s, cm/s, mm/s | m, cm, mm")
    g.add_argument("--column", type=int, help="value column, 0-based (time is column 0)")
    g.add_argument("--dt", type=float, help="sample interval for files without a time column, s")
    g.add_argument("--trim", type=float, nargs=2, metavar=("START", "END"), help="use only this part, s")
    g.add_argument("--highpass", type=float, help="high-pass corner, Hz (default 0.2 for accel/vel, none for disp)")
    g.add_argument("--lowpass", type=float, default=15.0, help="low-pass corner, Hz (default 15; 0 = off)")
    g = p.add_argument_group("output")
    g.add_argument("--plot", nargs="?", const="auto",
                   help="PNG of target vs predicted motion (default name: <gcode>_predicted.png)")
    g.add_argument("--csv", nargs="?", const="auto", help="CSV of target vs predicted motion")
    g.add_argument("--no-home", action="store_true", help="don't home Y first")
    g = p.add_argument_group("planner (normally leave alone)")
    g.add_argument("--planner-accel", type=float, help="acceleration given to the firmware (default 3 x record peak)")
    g.add_argument("--planner-accel-max", type=float, default=4000.0,
                   help="cap for the automatic planner acceleration (default 4000; lower it if the bed loses steps)")
    g.add_argument("--dt-min", type=float, help=argparse.SUPPRESS)
    g.add_argument("--dt-max", type=float, help=argparse.SUPPRESS)
    g.add_argument("--min-move", type=float, help=argparse.SUPPRESS)
    g.add_argument("--reversal-jerk", type=float, help=argparse.SUPPRESS)
    return p


def main(argv=None):
    args = make_parser().parse_args(argv)
    prof = eng.PROFILES["s1pro-stock"]
    out = Path(args.output) if args.output else Path(args.input).with_suffix(".gcode")
    out.parent.mkdir(parents=True, exist_ok=True)
    args.output = str(out)
    args.plot = str(out.with_name(out.stem + "_predicted.png")) if args.plot == "auto" else args.plot
    args.csv = str(out.with_name(out.stem + "_predicted.csv")) if args.csv == "auto" else args.csv
    engine_settings(args, prof)

    tgt = build_target(args)
    print(f"Record: {tgt.label}, {tgt.t[-1]:.1f} s")
    print("Bounds check:")
    factor, binding = bounds_table(tgt, args.y_min, args.y_max, args.max_vel, args.max_accel)
    if factor < 1.0:
        s = 0.98 * factor
        if args.fit:
            tgt.scale(s)
            tgt.notes.append(f"scaled       : amplitude x{s:.3f} by --fit ({binding} was the limit)")
            print(f"Scaled the amplitude by x{s:.3f} to fit ({binding} was the limit). Now:")
            bounds_table(tgt, args.y_min, args.y_max, args.max_vel, args.max_accel)
        elif not args.force:
            print(f"\nThe record doesn't fit: {binding} is x{1 / factor:.2f} too big.\n"
                  f"Re-run with --fit to shrink it automatically (amplitude x{s:.3f}), or choose your "
                  f"own --scale. With --time-scale k > 1 the speed drops by k and the acceleration "
                  f"by k^2, but the stroke stays the same.")
            return 2
        else:
            args.force = True
    else:
        print(f"Fits: the tightest limit ({binding}) is {1 / factor:.0%} used.")
    print()
    eng.run_motion(tgt, args, prof)
    print(f"\nCopy {out.name} to the SD card and print it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
shake_gui.py - a window for the shake-table workflow.

Double-click "Shake table GUI.bat", or run:  python shake_gui.py

Tab 1 makes a synthetic earthquake record (the same code as synthesize_record.py).
Tab 2 checks a record against the bed's limits, scales it if needed and writes the
G-code (the same code as record_to_gcode.py). The command-line scripts stay the way
to make many records at once or to script things.
"""
from __future__ import annotations

import contextlib
import io
import os
import queue
import re
import secrets
import subprocess
import sys
import threading
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import tkinter as tk  # noqa: E402
from tkinter import filedialog, messagebox, ttk  # noqa: E402
from tkinter.scrolledtext import ScrolledText  # noqa: E402

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

import harmonic_record as hr  # noqa: E402
import record_to_gcode as rtg  # noqa: E402
import shake_engine as eng  # noqa: E402
import synthesize_record as syn  # noqa: E402

G = syn.G
PROFILE = eng.PROFILES["s1pro-stock"]
RECORDS, GCODE = HERE / "records", HERE / "gcode"

# chart colours: light surface, recessive axes, blue = record, orange = predicted
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
GOOD, CRITICAL = "#0ca30c", "#d03b3b"

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
    "axes.prop_cycle": matplotlib.cycler(color=[BLUE, ORANGE]),  # also used by the saved PNGs
})


# --------------------------------------------------------------------------- helpers

class InputError(Exception):
    """A setting the user has to fix."""


def gui_parser(make_parser):
    """The script's own argument parser, raising InputError instead of exiting."""
    p = make_parser()

    def error(message):
        raise InputError(message)

    p.error = error
    return p


def friendly(exc, labels):
    """Turn an error from the scripts into a message that names the fields in this window."""
    if isinstance(exc, SystemExit):
        msg = exc.code if isinstance(exc.code, str) else "stopped"
    else:
        msg = str(exc)
    msg = re.sub(r"^error:\s*", "", msg.strip())
    msg = re.sub(r"^argument (?:-\w/)?(--[a-z0-9-]+)", lambda m: labels.get(m.group(1), m.group(1)), msg)
    msg = re.sub(r"--[a-z0-9-]+", lambda m: f"'{labels[m.group(0)]}'" if m.group(0) in labels
                 else m.group(0), msg)
    return msg[:1].upper() + msg[1:]


def resolve(text):
    """A path typed in the window; relative paths are relative to this folder."""
    p = Path(text.strip().strip('"'))
    return p if p.is_absolute() else HERE / p


def shown(path):
    """Path as shown in the window: relative to this folder when it is inside it."""
    try:
        return str(Path(path).resolve().relative_to(HERE))
    except ValueError:
        return str(path)


def open_folder(path):
    if sys.platform == "win32":
        os.startfile(path)  # noqa: S606 - opens Explorer on our own output folder
    else:
        subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(path)])


class QueueWriter(io.TextIOBase):
    """stdout/stderr replacement for the worker thread: hands text to the window."""

    def __init__(self, q):
        self.q = q

    def write(self, s):
        if s:
            self.q.put(("log", s))
        return len(s)


def style_axes(ax, ylabel, xlabel=None):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=AXIS, labelcolor=INK_2, labelsize=8, length=3)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_2, fontsize=9)


def show_limits(ax, data_lo, data_hi, lo, hi):
    """Dashed lines at the machine limits, when the data come near them."""
    span = max(data_hi - data_lo, 1e-9)
    y0, y1 = data_lo - 0.08 * span, data_hi + 0.08 * span
    for y in (lo, hi):
        if data_lo - 0.5 * span <= y <= data_hi + 0.5 * span:
            ax.axhline(y, color=MUTED, lw=0.9, ls=(0, (4, 3)))
            ax.text(0.995, y, "limit", transform=ax.get_yaxis_transform(), ha="right",
                    va="bottom" if y >= 0.5 * (data_lo + data_hi) else "top", fontsize=7, color=MUTED)
            y0, y1 = min(y0, y - 0.06 * span), max(y1, y + 0.06 * span)
    ax.set_ylim(y0, y1)


class PlotPanel(ttk.Frame):
    """A matplotlib figure with the zoom/pan toolbar (it also shows the value under the mouse)."""

    def __init__(self, parent):
        super().__init__(parent)
        self.fig = Figure(figsize=(8, 6), dpi=100, facecolor=SURFACE, layout="constrained")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        bar = NavigationToolbar2Tk(self.canvas, self, pack_toolbar=False)
        bar.update()
        bar.pack(side="bottom", fill="x")
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

    def message(self, text):
        self.fig.clear()
        self.fig.text(0.5, 0.5, text, ha="center", va="center", color=MUTED, fontsize=11, wrap=True)
        self.canvas.draw_idle()


class Form:
    """Rows of label / field / hint on a grid."""

    def __init__(self, frame):
        self.frame = frame
        self.row = 0
        frame.columnconfigure(1, weight=1)

    def section(self, text):
        ttk.Label(self.frame, text=text, style="Section.TLabel").grid(
            row=self.row, column=0, columnspan=3, sticky="w", pady=(12 if self.row else 0, 3))
        self.row += 1

    def _hint(self, hint):
        self.last_hint = None
        if hint:
            self.last_hint = ttk.Label(self.frame, text=hint, style="Hint.TLabel")
            self.last_hint.grid(row=self.row, column=2, sticky="w", padx=(6, 0))

    def entry(self, label, var, hint="", width=11):
        ttk.Label(self.frame, text=label).grid(row=self.row, column=0, sticky="w", padx=(0, 8), pady=2)
        e = ttk.Entry(self.frame, textvariable=var, width=width)
        e.grid(row=self.row, column=1, sticky="we", pady=2)
        self._hint(hint)
        self.row += 1
        return e

    def combo(self, label, var, values, hint="", width=11):
        ttk.Label(self.frame, text=label).grid(row=self.row, column=0, sticky="w", padx=(0, 8), pady=2)
        c = ttk.Combobox(self.frame, textvariable=var, values=values, state="readonly", width=width)
        c.grid(row=self.row, column=1, sticky="we", pady=2)
        self._hint(hint)
        self.row += 1
        return c

    def pair(self, label, var_a, var_b, hint=""):
        ttk.Label(self.frame, text=label).grid(row=self.row, column=0, sticky="w", padx=(0, 8), pady=2)
        f = ttk.Frame(self.frame)
        f.grid(row=self.row, column=1, sticky="we", pady=2)
        ttk.Entry(f, textvariable=var_a, width=5).pack(side="left", fill="x", expand=True)
        ttk.Label(f, text=" to ").pack(side="left")
        ttk.Entry(f, textvariable=var_b, width=5).pack(side="left", fill="x", expand=True)
        self._hint(hint)
        self.row += 1

    def check(self, text, var, command=None):
        c = ttk.Checkbutton(self.frame, text=text, variable=var, command=command)
        c.grid(row=self.row, column=0, columnspan=3, sticky="w", pady=2)
        self.row += 1
        return c

    def file(self, label, var, browse):
        ttk.Label(self.frame, text=label).grid(row=self.row, column=0, columnspan=3, sticky="w", pady=(2, 0))
        self.row += 1
        f = ttk.Frame(self.frame)
        f.grid(row=self.row, column=0, columnspan=3, sticky="we", pady=(0, 2))
        ttk.Entry(f, textvariable=var).pack(side="left", fill="x", expand=True)
        ttk.Button(f, text="Browse…", command=browse).pack(side="left", padx=(4, 0))
        self.row += 1


# --------------------------------------------------------------------------- tab 1: make a record

class RecordTab(ttk.Frame):
    LABELS = {"--duration": "Duration", "--pga": "Peak acceleration", "--pgv-max": "Velocity limit",
              "--pgd-max": "Displacement limit", "--limit-mode": "Meet the limits by",
              "--f-dominant": "Dominant frequency", "--damping": "Ground damping", "--f-low": "Low cut",
              "--f-high": "High cut", "--rise": "Build-up time", "--strong": "Strong shaking",
              "--dt": "Sample interval", "--seed": "Seed"}
    REQUIRED = (("duration", "--duration"), ("pga", "--pga"), ("pgv_max", "--pgv-max"),
                ("pgd_max", "--pgd-max"), ("f_dominant", "--f-dominant"), ("damping", "--damping"),
                ("f_low", "--f-low"), ("f_high", "--f-high"), ("dt", "--dt"))
    MODES = ("filter: keep the PGA", "scale: keep the spectrum")
    HLABELS = {"--define": "Amplitude is of", "--amplitude": "Amplitude", "--freq": "Frequency",
               "--duration": "Duration", "--ramp": "Fade in / out", "--dt": "Sample interval",
               "--freq-end": "End frequency", "--freq-law": "Frequency sweep law",
               "--amplitude-end": "End amplitude", "--amplitude-law": "Amplitude sweep law"}
    QUANTITIES = {"displacement": "disp", "velocity": "vel", "acceleration": "accel"}
    AMP_HINTS = {"disp": "mm, cm", "vel": "mm/s, cm/s", "accel": "g, gal, mm/s2"}
    F_LAWS = ("log", "linear")
    A_LAWS = ("linear", "log")

    def __init__(self, app, parent):
        super().__init__(parent, padding=10)
        self.app = app
        self.kind = tk.StringVar(value="quake")
        self.recs, self.saved = {}, {}  # per record type: the last record made, and where it was saved
        self.vars = {k: tk.StringVar(value=d) for k, d in (
            ("duration", "20"), ("pga", "0.1g"), ("pgv_max", "120mm/s"), ("pgd_max", "60mm"),
            ("f_dominant", "2.5"), ("damping", "0.6"), ("f_low", "0.2"), ("f_high", "15"),
            ("rise", ""), ("strong", ""), ("dt", "0.005"), ("seed", ""))}
        self.mode = tk.StringVar(value=self.MODES[0])
        self.random_seed = tk.BooleanVar(value=True)
        self.save_png = tk.BooleanVar(value=True)
        self.save_at2 = tk.BooleanVar(value=False)

        left = ttk.Frame(self)
        left.pack(side="left", fill="y", padx=(0, 12))
        top = ttk.Frame(left)
        top.pack(fill="x")
        ttk.Label(top, text="Type of record", style="Section.TLabel").pack(anchor="w")
        radios = ttk.Frame(top)
        radios.pack(fill="x", pady=(3, 0))
        for text, value in (("Earthquake-like", "quake"), ("Harmonic / sweep", "harmonic")):
            ttk.Radiobutton(radios, text=text, value=value, variable=self.kind,
                            command=self.switch_kind).pack(side="left", padx=(0, 16))
        self.quake_frame = ttk.Frame(left)
        self.harm_frame = ttk.Frame(left)
        self.common = ttk.Frame(left)
        self.build_harmonic_form(self.harm_frame)
        self.quake_frame.pack(fill="x", pady=(10, 0))
        self.common.pack(fill="x", pady=(12, 0))
        f = Form(self.quake_frame)
        v = self.vars
        f.section("Record")
        f.entry("Duration", v["duration"], "s")
        f.entry("Peak acceleration", v["pga"], "0.1g, 98gal, 980mm/s2")
        f.entry("Velocity limit", v["pgv_max"], "100mm/s, 10cm/s, none")
        f.entry("Displacement limit", v["pgd_max"], "40mm, 4cm, none")
        f.combo("Meet the limits by", self.mode, self.MODES, width=20)
        f.section("Character of the shaking")
        f.entry("Dominant frequency", v["f_dominant"], "Hz, of the ground")
        f.entry("Ground damping", v["damping"], "0.6 is typical")
        f.entry("Low cut", v["f_low"], "Hz, removes slow drift")
        f.entry("High cut", v["f_high"], "Hz, 15 ≈ what the bed follows")
        f.entry("Build-up time", v["rise"], "s, blank = 15 % of duration")
        f.entry("Strong shaking", v["strong"], "s, blank = 35 % of duration")
        f.entry("Sample interval", v["dt"], "s")
        f.section("Seed")
        self.seed_entry = f.entry("Seed", v["seed"], "same seed = same record")
        f.check("New random seed each time", self.random_seed, self._seed_state)
        self._seed_state()
        ttk.Label(self.quake_frame, text="For many records at once: synthesize_record.py --count",
                  style="Hint.TLabel").grid(row=f.row, column=0, columnspan=3, sticky="w", pady=(4, 0))

        f = Form(self.common)
        f.section("Save")
        f.check("Also save a plot (PNG)", self.save_png)
        f.check("Also save a PEER .AT2 file", self.save_at2)
        buttons = ttk.Frame(self.common)
        buttons.grid(row=f.row, column=0, columnspan=3, sticky="we", pady=(14, 0))
        ttk.Button(buttons, text="Generate", style="Primary.TButton",
                   command=self.generate).pack(fill="x")
        row = ttk.Frame(buttons)
        row.pack(fill="x", pady=(6, 0))
        ttk.Button(row, text="Save record…", command=self.save).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Use in step 2 →", command=self.to_step2).pack(
            side="left", fill="x", expand=True, padx=(6, 0))

        right = ttk.Frame(self)
        right.pack(side="left", fill="both", expand=True)
        self.summary = ttk.Label(right, text="", justify="left", style="Summary.TLabel")
        self.summary.pack(side="top", fill="x", pady=(0, 6))
        self.bed_line = ttk.Label(right, text="", justify="left", style="Summary.TLabel")
        self.bed_line.pack(side="top", fill="x", pady=(0, 6))
        self.plot = PlotPanel(right)
        self.plot.pack(side="top", fill="both", expand=True)
        right.bind("<Configure>", lambda e: (self.summary.configure(wraplength=e.width - 10),
                                             self.bed_line.configure(wraplength=e.width - 10)))

    def build_harmonic_form(self, frame):
        self.hvars = {k: tk.StringVar(value=d) for k, d in (
            ("amp", hr.DEFAULT_AMP["disp"]), ("freq", "2"), ("duration", "20"), ("ramp", ""),
            ("dt", "0.005"), ("freq_end", "8"), ("amp_end", hr.DEFAULT_AMP_END["disp"]))}
        self.quantity = tk.StringVar(value="displacement")
        self.sweep_f, self.sweep_a = tk.BooleanVar(value=False), tk.BooleanVar(value=False)
        self.f_law, self.a_law = tk.StringVar(value=self.F_LAWS[0]), tk.StringVar(value=self.A_LAWS[0])
        v = self.hvars
        f = Form(frame)
        f.section("Harmonic signal")
        f.combo("Amplitude is of", self.quantity, tuple(self.QUANTITIES))
        f.entry("Amplitude", v["amp"], self.AMP_HINTS["disp"])
        self.amp_hint = f.last_hint
        f.entry("Frequency", v["freq"], "Hz; the bed follows up to ~15")
        f.entry("Duration", v["duration"], "s")
        f.entry("Fade in / out", v["ramp"], "s, blank = 2 cycles")
        f.entry("Sample interval", v["dt"], "s")
        f.section("Frequency sweep")
        f.check("Sweep the frequency (start → end over the duration)", self.sweep_f, self.sweep_state)
        self.sweep_f_widgets = [f.entry("End frequency", v["freq_end"], "Hz"),
                                f.combo("Sweep law", self.f_law, self.F_LAWS,
                                        "log = same time per octave")]
        f.section("Amplitude sweep")
        f.check("Sweep the amplitude (start → end over the duration)", self.sweep_a, self.sweep_state)
        end_amp = f.entry("End amplitude", v["amp_end"], self.AMP_HINTS["disp"])
        self.amp_end_hint = f.last_hint
        self.sweep_a_widgets = [end_amp, f.combo("Sweep law", self.a_law, self.A_LAWS,
                                                 "log = same ratio per second")]
        self.quantity.trace_add("write", self.quantity_changed)
        self.sweep_state()

    def sweep_state(self):
        for on, widgets in ((self.sweep_f.get(), self.sweep_f_widgets), (self.sweep_a.get(), self.sweep_a_widgets)):
            for w in widgets:
                w.configure(state=("readonly" if isinstance(w, ttk.Combobox) else "normal") if on else "disabled")

    def quantity_changed(self, *_):
        """New quantity: new unit hints, and default amplitudes where the old ones don't fit."""
        kind = self.QUANTITIES[self.quantity.get()]
        for label in (self.amp_hint, self.amp_end_hint):
            label.configure(text=self.AMP_HINTS[kind])
        for key, default in (("amp", hr.DEFAULT_AMP[kind]), ("amp_end", hr.DEFAULT_AMP_END[kind])):
            m = re.fullmatch(r"\s*[\d.eE+-]+\s*([a-zA-Z/^0-9]*)\s*", self.hvars[key].get())
            unit = m.group(1).lower().replace("^", "") if m else ""
            if not unit or unit not in syn.UNITS[kind]:
                self.hvars[key].set(default)

    def _seed_state(self):
        self.seed_entry.configure(state="readonly" if self.random_seed.get() else "normal")

    @property
    def rec(self):
        return self.recs.get(self.kind.get())

    def switch_kind(self):
        harmonic = self.kind.get() == "harmonic"
        (self.quake_frame if harmonic else self.harm_frame).pack_forget()
        (self.harm_frame if harmonic else self.quake_frame).pack(fill="x", pady=(10, 0), before=self.common)
        if self.rec is None:
            self.generate()
        else:
            self.draw()

    def read_harmonic(self):
        v = {k: var.get().strip() for k, var in self.hvars.items()}
        required = [("amp", "--amplitude"), ("freq", "--freq"), ("duration", "--duration"), ("dt", "--dt")]
        if self.sweep_f.get():
            required.append(("freq_end", "--freq-end"))
        if self.sweep_a.get():
            required.append(("amp_end", "--amplitude-end"))
        argv = ["--define=" + self.QUANTITIES[self.quantity.get()]]
        for key, opt in required:
            if not v[key]:
                raise InputError(f"{self.HLABELS[opt]}: enter a value")
            argv.append(f"{opt}={v[key]}")
        if v["ramp"]:
            argv.append(f"--ramp={v['ramp']}")
        if self.sweep_f.get():
            argv.append("--freq-law=" + self.f_law.get().split(":")[0])
        if self.sweep_a.get():
            argv.append("--amplitude-law=" + self.a_law.get().split(":")[0])
        p = gui_parser(hr.make_parser)
        args = p.parse_args(argv)
        ramps = hr.check_settings(p, args)
        if args.duration > 900:
            raise InputError("Duration: at most 900 s here")
        return args, ramps

    def read_quake(self):
        v = {k: var.get().strip() for k, var in self.vars.items()}
        argv = []
        for key, opt in self.REQUIRED:
            if not v[key]:
                raise InputError(f"{self.LABELS[opt]}: enter a value")
            argv.append(f"{opt}={v[key]}")
        for key, opt in (("rise", "--rise"), ("strong", "--strong")):
            if v[key]:
                argv.append(f"{opt}={v[key]}")
        argv.append("--limit-mode=" + self.mode.get().split(":")[0])
        p = gui_parser(syn.make_parser)
        args = p.parse_args(argv)
        rise, strong = syn.check_settings(p, args)
        if args.pga <= 0:
            raise InputError("Peak acceleration must be above zero")
        if args.duration > 900:
            raise InputError("Duration: at most 900 s here")
        if self.random_seed.get() or not v["seed"]:
            seed = secrets.randbelow(10 ** 6)
        else:
            try:
                seed = int(v["seed"])
            except ValueError:
                raise InputError("Seed: a whole number") from None
        return args, rise, strong, seed

    def generate(self):
        kind = self.kind.get()
        try:
            if kind == "quake":
                args, rise, strong, seed = self.read_quake()
            else:
                args, ramps = self.read_harmonic()
        except InputError as e:
            msg = friendly(e, self.LABELS if kind == "quake" else self.HLABELS)
            self.app.status(msg, error=True)
            messagebox.showerror("Can't make the record", msg, parent=self)
            return False
        self.app.busy_cursor(True)
        try:
            if kind == "quake":
                self.vars["seed"].set(str(seed))
                rec = syn.make_record(args, seed, rise, strong)
                rec.update(type="quake", name=f"quake_{rec['pga'] / G:.2f}g_{args.duration:g}s_seed{seed}",
                           title=f"Seed {seed}: PGA {rec['pga'] / G:.3f} g, PGV {rec['pgv']:.0f} mm/s, "
                                 f"PGD {rec['pgd']:.1f} mm")
                done = f"Made a {args.duration:g} s record with seed {seed}."
            else:
                rec = hr.make_record(args, *ramps)
                done = f"Made: {rec['title']}, {rec['t'][-1]:g} s."
            self.recs[kind], self.saved[kind] = rec, None
            self.draw()
        finally:
            self.app.busy_cursor(False)
        self.app.status(done + " Save it, or use it in step 2.")
        return True

    def draw(self):
        rec = self.rec
        self.summary.configure(text="\n".join(rec["header"][1:]).replace("mm/s^2", "mm/s²"))
        self.update_bed_line()
        fig = self.plot.fig
        fig.clear()
        if rec["type"] == "harmonic":
            hr.draw_record(fig, rec, (BLUE, ORANGE))
            for ax in fig.axes:
                style_axes(ax, ax.get_ylabel(), ax.get_xlabel() or None)
            fig.suptitle(rec["title"], color=INK, fontsize=10)
            self.plot.canvas.draw_idle()
            return
        gs = fig.add_gridspec(3, 2, width_ratios=[2.3, 1])
        axes = [fig.add_subplot(gs[0, 0])]
        axes += [fig.add_subplot(gs[i, 0], sharex=axes[0]) for i in (1, 2)]
        for ax, y, label in zip(axes, (rec["a"] / G, rec["v"], rec["x"]),
                                ("acceleration (g)", "velocity (mm/s)", "displacement (mm)")):
            ax.plot(rec["t"], y, color=BLUE, lw=0.9)
            style_axes(ax, label)
        for ax in axes[:2]:
            ax.tick_params(labelbottom=False)
        axes[2].set_xlabel("time (s)", color=INK_2, fontsize=9)
        sp = fig.add_subplot(gs[:, 1])
        periods = np.logspace(np.log10(0.05), np.log10(5.0), 80)
        sp.semilogx(periods, syn.response_spectrum(rec["a"], rec["dt"], periods) / G, color=BLUE, lw=1.6)
        style_axes(sp, "spectral acceleration, 5 % damping (g)", "period (s)")
        sp.grid(which="minor", color=GRID, lw=0.4)
        fig.suptitle(rec["title"], color=INK, fontsize=10)
        self.plot.canvas.draw_idle()

    def update_bed_line(self):
        if self.rec is None:
            return
        y_min, y_max, max_vel, max_accel = self.app.convert.limits()
        used = (self.rec["stroke"] / max(y_max - y_min, 1e-9), self.rec["pgv"] / max_vel,
                self.rec["pga"] / max_accel)
        verdict = ("fits." if max(used) <= 1 else
                   f"too big (×{max(used):.2f}); step 2 can scale it to fit.")
        self.bed_line.configure(
            text=f"On the bed (limits in step 2): travel {used[0]:.0%}, speed {used[1]:.0%}, "
                 f"acceleration {used[2]:.0%} of the limit; {verdict}")

    def save(self):
        if self.rec is None and not self.generate():
            return None
        rec, kind = self.rec, self.kind.get()
        RECORDS.mkdir(exist_ok=True)
        path = filedialog.asksaveasfilename(
            parent=self, title="Save the record", initialdir=RECORDS, initialfile=rec["name"] + ".csv",
            defaultextension=".csv", filetypes=[("CSV record", "*.csv")])
        if not path:
            return None
        writer = hr.save_record if rec["type"] == "harmonic" else syn.save_record
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            writer(path, rec, self.save_at2.get(), self.save_png.get())
        self.saved[kind] = Path(path)
        extra = [ext for ext, on in ((".png", self.save_png.get()), (".AT2", self.save_at2.get())) if on]
        self.app.status(f"Saved {shown(path)}" + (f" (and {', '.join(extra)})" if extra else ""))
        return self.saved[kind]

    def to_step2(self):
        if self.rec is None and not self.generate():
            return
        path = self.saved.get(self.kind.get()) or self.save()
        if path:
            # a harmonic record is already clean: the converter's 15 Hz low-pass would only trim
            # the amplitude of faster sines. Earthquake records get the usual 15 Hz.
            self.app.convert.load_file(path, lowpass="0" if self.rec["type"] == "harmonic" else "15")
            self.app.notebook.select(self.app.convert)


# --------------------------------------------------------------------------- tab 2: convert to G-code

class Meter:
    """One row of the bounds check: value, bar against the limit, % used, OK / too big."""

    W, H = 170, 12

    def __init__(self, parent, row, name, scale):
        self.w, self.h = int(self.W * scale), int(self.H * scale)
        bg = ttk.Style().lookup("TFrame", "background")
        ttk.Label(parent, text=name).grid(row=row, column=0, sticky="w", padx=(0, 10), pady=2)
        self.value = ttk.Label(parent, text="", style="Num.TLabel")
        self.value.grid(row=row, column=1, sticky="e", padx=(0, 10))
        self.bar = tk.Canvas(parent, width=self.w, height=self.h, highlightthickness=0, bg=bg)
        self.bar.grid(row=row, column=2, padx=(0, 8))
        self.pct = ttk.Label(parent, text="", width=6, anchor="e", style="Num.TLabel")
        self.pct.grid(row=row, column=3, sticky="e")
        self.dot = tk.Label(parent, text="", bg=bg, font=("Segoe UI", 9))
        self.dot.grid(row=row, column=4, padx=(8, 2))
        self.flag = ttk.Label(parent, text="")
        self.flag.grid(row=row, column=5, sticky="w")

    def set(self, value, limit, unit):
        used = value / max(limit, 1e-9)
        self.value.configure(text=f"{value:,.1f} of {limit:,.0f} {unit}")
        self.pct.configure(text=f"{used:.0%}")
        c, w, h = self.bar, self.w, self.h
        c.delete("all")
        full = 1.5  # the bar spans 0..150 %, the tick marks 100 %
        c.create_rectangle(0, h * 0.25, w, h * 0.75, fill=GRID, width=0)
        c.create_rectangle(0, 0, w * min(used, full) / full, h, width=0,
                           fill=BLUE if used <= 1 else CRITICAL)
        x100 = w / full
        c.create_line(x100, 0, x100, h, fill=INK, width=2)
        ok = used <= 1
        self.dot.configure(text="●" if ok else "▲", fg=GOOD if ok else CRITICAL)
        self.flag.configure(text="OK" if ok else f"too big ×{used:.2f}")

    def clear(self):
        for w in (self.value, self.pct, self.flag):
            w.configure(text="")
        self.dot.configure(text="")
        self.bar.delete("all")


class ConvertTab(ttk.Frame):
    LABELS = {"--y-min": "Y min", "--y-max": "Y max", "--center": "Centre", "--max-vel": "Max speed",
              "--max-accel": "Max acceleration", "--scale": "Amplitude scale", "--time-scale": "Time scale",
              "--type": "File contains", "--units": "Units", "--column": "Column",
              "--dt": "Sample interval", "--trim": "Use only", "--highpass": "High-pass",
              "--lowpass": "Low-pass", "--planner-accel-max": "Planner accel cap",
              "--planner-accel": "Planner accel cap", "--autoscale": "Scale to fit",
              "--fit": "Scale to fit", "--force": "Scale to fit"}
    REQUIRED = (("y_min", "--y-min"), ("y_max", "--y-max"), ("max_vel", "--max-vel"),
                ("max_accel", "--max-accel"), ("scale", "--scale"), ("time_scale", "--time-scale"),
                ("lowpass", "--lowpass"), ("planner_accel_max", "--planner-accel-max"))
    UNITS = ("auto", "g", "m/s2", "cm/s2", "gal", "mm/s2", "m/s", "cm/s", "mm/s", "m", "cm", "mm")

    def __init__(self, app, parent):
        super().__init__(parent, padding=10)
        self.app = app
        self.busy = False
        self.tgt = self.args = self.fit = None
        self._after = None
        self.vars = {k: tk.StringVar(value=d) for k, d in (
            ("input", ""), ("output", ""), ("y_min", "5"), ("y_max", "215"), ("center", ""),
            ("max_vel", "150"), ("max_accel", "1500"), ("scale", "1"), ("time_scale", "1"),
            ("column", ""), ("dt", ""), ("trim_from", ""), ("trim_to", ""), ("highpass", ""),
            ("lowpass", "15"), ("planner_accel_max", "4000"))}
        self.kind = tk.StringVar(value="auto")
        self.units = tk.StringVar(value="auto")
        self.save_png = tk.BooleanVar(value=True)
        self.save_csv = tk.BooleanVar(value=False)

        left = ttk.Frame(self)
        left.pack(side="left", fill="y", padx=(0, 12))
        basic = ttk.Frame(left)
        basic.pack(fill="x")
        f = Form(basic)
        v = self.vars
        f.section("Files")
        f.file("Record (CSV, TXT or PEER .AT2/.VT2/.DT2)", v["input"], self.browse_input)
        f.file("G-code to write", v["output"], self.browse_output)
        f.section("Bed limits")
        f.entry("Y min", v["y_min"], "mm, lowest bed Y to use")
        f.entry("Y max", v["y_max"], "mm, highest bed Y to use")
        f.entry("Centre", v["center"], "mm, blank = middle")
        f.entry("Max speed", v["max_vel"], "mm/s")
        f.entry("Max acceleration", v["max_accel"], "mm/s², stock tracks well ≤ 1500")
        f.section("Scaling")
        f.entry("Amplitude scale", v["scale"], "× displacement, speed, accel")
        f.entry("Time scale", v["time_scale"], "> 1 slower: speed ÷k, accel ÷k²")

        self.adv_open = tk.BooleanVar(value=False)
        self.adv_button = ttk.Checkbutton(left, text="▸ Reading the file, filters, planner",
                                          variable=self.adv_open, style="Toolbutton",
                                          command=self.toggle_advanced)
        self.adv_button.pack(fill="x", pady=(10, 0))
        self.adv = ttk.Frame(left)
        f = Form(self.adv)
        f.combo("File contains", self.kind, ("auto", "accel", "vel", "disp"), "auto for step-1 CSVs")
        f.combo("Units", self.units, self.UNITS, "auto for step-1 CSVs, PEER")
        f.entry("Column", v["column"], "0-based, blank = auto")
        f.entry("Sample interval", v["dt"], "s, files without a time column")
        f.pair("Use only", v["trim_from"], v["trim_to"], "s, blank = all")
        f.entry("High-pass", v["highpass"], "Hz, blank = 0.2 (none for disp)")
        f.entry("Low-pass", v["lowpass"], "Hz, 0 = off")
        f.entry("Planner accel cap", v["planner_accel_max"], "mm/s², lower if steps are lost")

        self.bottom = ttk.Frame(left)
        self.bottom.pack(fill="x", pady=(10, 0))
        ttk.Checkbutton(self.bottom, text="Also save the predicted-motion plot (PNG)",
                        variable=self.save_png).pack(anchor="w")
        ttk.Checkbutton(self.bottom, text="Also save the predicted motion (CSV)",
                        variable=self.save_csv).pack(anchor="w")
        self.write_button = ttk.Button(self.bottom, text="Write G-code", style="Primary.TButton",
                                       command=self.write)
        self.write_button.pack(fill="x", pady=(12, 0))
        self.folder_button = ttk.Button(self.bottom, text="Open the G-code folder", state="disabled",
                                        command=lambda: open_folder(self.out_folder))
        self.folder_button.pack(fill="x", pady=(6, 0))
        self.progress = ttk.Progressbar(self.bottom, mode="indeterminate")
        self.out_folder = GCODE

        right = ttk.Frame(self)
        right.pack(side="left", fill="both", expand=True)
        box = ttk.Frame(right)
        box.pack(side="top", fill="x")
        ttk.Label(box, text="Bounds check", style="Section.TLabel").grid(row=0, column=0, columnspan=6,
                                                                       sticky="w", pady=(0, 4))
        scale = float(self.tk.call("tk", "scaling")) / (96 / 72)
        self.meters = [Meter(box, 1 + i, name, scale) for i, name in
                       enumerate(("Bed travel (stroke)", "Peak velocity", "Peak acceleration"))]
        line = ttk.Frame(right)
        line.pack(side="top", fill="x", pady=(6, 4))
        self.verdict = ttk.Label(line, text="", style="Summary.TLabel")
        self.verdict.pack(side="left")
        self.fit_button = ttk.Button(line, text="Scale to fit (98 %)", command=self.scale_to_fit,
                                     state="disabled")
        self.fit_button.pack(side="right")

        self.log_box = ScrolledText(right, height=9, font=("Consolas", 9), wrap="none", state="disabled",
                                    relief="flat", borderwidth=1, background=SURFACE, foreground=INK)
        self.log_box.pack(side="bottom", fill="x", pady=(6, 0))
        self.plot = PlotPanel(right)
        self.plot.pack(side="top", fill="both", expand=True)
        self.plot.message("Choose a record file (Browse…), or make one in step 1 and press "
                          "'Use in step 2 →'.")

        for key, var in self.vars.items():
            if key != "output":
                var.trace_add("write", self.schedule_check)
        for var in (self.kind, self.units):
            var.trace_add("write", self.schedule_check)
        for key in ("y_min", "y_max", "max_vel", "max_accel"):
            self.vars[key].trace_add("write", lambda *_: self.app.record.update_bed_line())

    # ---- layout and files

    def toggle_advanced(self):
        if self.adv_open.get():
            self.adv.pack(fill="x", pady=(6, 0), before=self.bottom)
            self.adv_button.configure(text="▾ Reading the file, filters, planner")
        else:
            self.adv.pack_forget()
            self.adv_button.configure(text="▸ Reading the file, filters, planner")

    def browse_input(self):
        path = filedialog.askopenfilename(
            parent=self, title="Choose a record", initialdir=RECORDS if RECORDS.is_dir() else HERE,
            filetypes=[("Records", "*.csv *.txt *.dat *.AT2 *.VT2 *.DT2 *.at2 *.vt2 *.dt2"),
                       ("All files", "*.*")])
        if path:
            self.load_file(path)

    def browse_output(self):
        GCODE.mkdir(exist_ok=True)
        current = self.output_path()
        path = filedialog.asksaveasfilename(
            parent=self, title="Write the G-code to", initialdir=current.parent if current else GCODE,
            initialfile=current.name if current else "", defaultextension=".gcode",
            filetypes=[("G-code", "*.gcode")])
        if path:
            self.vars["output"].set(shown(path))

    def load_file(self, path, lowpass=None):
        """Open a record. lowpass: set the Low-pass field too (and say so in the log if it changed)."""
        self.vars["output"].set(shown(GCODE / (Path(path).stem + ".gcode")))
        if lowpass is not None and self.vars["lowpass"].get().strip() != lowpass:
            self.vars["lowpass"].set(lowpass)
            self.log("Low-pass (under 'Reading the file, filters, planner') set to "
                     + ("0 = off: a harmonic record is already clean, and the 15 Hz filter would trim "
                        "the amplitude of sines above ~8 Hz.\n" if lowpass == "0" else
                        f"{lowpass} Hz, the usual value for earthquake records.\n"), clear=True)
        self.vars["input"].set(shown(path))

    def output_path(self):
        text = self.vars["output"].get().strip()
        if text:
            out = resolve(text)
            return out if out.suffix else out.with_suffix(".gcode")
        inp = self.vars["input"].get().strip()
        return GCODE / (Path(inp).stem + ".gcode") if inp else None

    def limits(self):
        out = []
        for key, default in (("y_min", 5.0), ("y_max", 215.0), ("max_vel", 150.0), ("max_accel", 1500.0)):
            try:
                out.append(float(self.vars[key].get()))
            except ValueError:
                out.append(default)
        return out

    # ---- reading the settings and checking the record

    def read_args(self):
        v = {k: var.get().strip() for k, var in self.vars.items()}
        if not v["input"]:
            raise InputError("Choose a record file first")
        path = resolve(v["input"])
        if not path.is_file():
            raise InputError(f"Record file not found: {v['input']}")
        argv = [str(path)]
        for key, opt in self.REQUIRED:
            if not v[key]:
                raise InputError(f"{self.LABELS[opt]}: enter a value")
            argv.append(f"{opt}={v[key]}")
        for key, opt in (("center", "--center"), ("column", "--column"), ("dt", "--dt"),
                         ("highpass", "--highpass")):
            if v[key]:
                argv.append(f"{opt}={v[key]}")
        if self.kind.get() != "auto":
            argv.append("--type=" + self.kind.get())
        if self.units.get() != "auto":
            argv.append("--units=" + self.units.get())
        if v["trim_from"] or v["trim_to"]:
            argv += ["--trim", v["trim_from"] or "0", v["trim_to"] or "1e9"]
        args = gui_parser(rtg.make_parser).parse_args(argv)
        if args.y_max - args.y_min < 1:
            raise InputError("Y max must be above Y min")
        if args.y_min < -8 or args.y_max > 220:
            raise InputError("Y min/Y max: the printer's soft limits are Y -8 to 220 mm")
        if args.center is not None and not args.y_min < args.center < args.y_max:
            raise InputError("Centre must lie between Y min and Y max")
        for name, val in (("Max speed", args.max_vel), ("Max acceleration", args.max_accel),
                          ("Amplitude scale", args.scale), ("Time scale", args.time_scale),
                          ("Planner accel cap", args.planner_accel_max)):
            if val <= 0:
                raise InputError(f"{name} must be above zero")
        out = self.output_path()
        args.output = str(out)
        args.plot = str(out.with_name(out.stem + "_predicted.png")) if self.save_png.get() else None
        args.csv = str(out.with_name(out.stem + "_predicted.csv")) if self.save_csv.get() else None
        rtg.engine_settings(args, PROFILE)
        return args

    def schedule_check(self, *_):
        if self.busy:
            return
        if self._after:
            self.after_cancel(self._after)
        self._after = self.after(500, self.check)

    def check(self):
        """Read the record with the current settings, show it against the limits. None if it can't."""
        self._after = None
        if self.busy:
            return None
        if not self.vars["input"].get().strip():
            return None
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                args = self.read_args()
                tgt = rtg.build_target(args)
        except (InputError, SystemExit, ValueError, OSError) as e:
            self.tgt = self.fit = None
            for m in self.meters:
                m.clear()
            self.fit_button.configure(state="disabled")
            msg = friendly(e, self.LABELS)
            self.verdict.configure(text="⚠  " + msg)
            self.app.status(msg, error=True)
            self.plot.message(msg)
            return None
        rows, factor, binding = rtg.bounds(tgt, args.y_min, args.y_max, args.max_vel, args.max_accel)
        for meter, (name, value, limit, unit) in zip(self.meters, rows):
            meter.set(value, limit, unit.replace("^2", "²"))
        if factor >= 1:
            text = f"Fits. The tightest limit ({binding}) is {1 / factor:.0%} used."
        else:
            text = f"Too big: {binding} is ×{1 / factor:.2f} over the limit."
        self.verdict.configure(text=f"{text}   Record: {tgt.t[-1]:.1f} s from {tgt.label}")
        self.fit_button.configure(state="normal")
        self.tgt, self.args, self.fit = tgt, args, (factor, binding)
        self.draw(tgt, args)
        self.app.status("Bounds checked. " + text)
        return args, tgt

    def scale_to_fit(self):
        if self.fit is None and self.check() is None:
            return
        factor, binding = self.fit
        try:
            current = float(self.vars["scale"].get())
        except ValueError:
            current = 1.0
        new = current * 0.98 * factor
        self.vars["scale"].set(f"{new:.4g}")
        self.check()
        self.app.status(f"Amplitude scale set to ×{new:.4g}: {binding} now uses 98 % of its limit.")

    # ---- plots and log

    def draw(self, tgt, args, result=None):
        fig = self.plot.fig
        fig.clear()
        axes = fig.subplots(3, 1, sharex=True)
        center = args.center if args.center is not None else 0.5 * (args.y_min + args.y_max)
        offset = center - 0.5 * (tgt.x.max() + tgt.x.min())
        t = tgt.t
        if result is None:
            series = [(offset + tgt.x, tgt.v, tgt.a, "record", BLUE, 1.2)]
            acc_label = "acceleration (mm/s²)"
        else:
            series = [(offset + tgt.x, tgt.v, result["tgt_lp"], "record (target)", BLUE, 1.6),
                      (offset + result["pos"], result["vel"], result["acc_lp"], "predicted bed motion",
                       ORANGE, 1.0)]
            acc_label = f"accel. below {result['f_cmp']:g} Hz (mm/s²)"
        for x, vel, acc, label, color, lw in series:
            axes[0].plot(t, x, color=color, lw=lw, label=label)
            axes[1].plot(t, vel, color=color, lw=lw)
            axes[2].plot(t, acc, color=color, lw=lw)
        lims = ((args.y_min, args.y_max), (-args.max_vel, args.max_vel), (-args.max_accel, args.max_accel))
        for i, (ax, ylabel, (lo, hi)) in enumerate(zip(
                axes, ("bed position Y (mm)", "velocity (mm/s)", acc_label), lims)):
            data = np.concatenate([s[i] for s in series])
            style_axes(ax, ylabel)
            show_limits(ax, float(data.min()), float(data.max()), lo, hi)
        axes[2].set_xlabel("time (s)", color=INK_2, fontsize=9)
        if result is not None:
            axes[0].legend(loc="upper right", frameon=False, fontsize=8, ncols=2)
        title = (f"{tgt.label}: record vs predicted bed motion" if result is not None
                 else f"{tgt.label} on the bed")
        fig.suptitle(title, color=INK, fontsize=10)
        self.plot.canvas.draw_idle()

    def log(self, text, clear=False):
        self.log_box.configure(state="normal")
        if clear:
            self.log_box.delete("1.0", "end")
        self.log_box.insert("end", text)
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    # ---- writing the G-code

    def write(self):
        if self.busy:
            return
        if self._after:
            self.after_cancel(self._after)
            self._after = None
        if not self.vars["input"].get().strip():
            messagebox.showinfo("No record", "Choose a record file first (Browse…).", parent=self)
            return
        res = self.check()
        if res is None:
            messagebox.showerror("Can't read the record", self.verdict.cget("text").lstrip("⚠ "),
                                 parent=self)
            return
        factor, binding = self.fit
        if factor < 1:
            s = 0.98 * factor
            if not messagebox.askyesno(
                    "The record doesn't fit",
                    f"{binding[:1].upper() + binding[1:]} is ×{1 / factor:.2f} over the limit set on the "
                    f"left.\n\nShrink the amplitude to fit (×{s:.3f}) and write the G-code?\n\n"
                    f"(If your machine can take more, press No and raise the limits instead.)",
                    parent=self):
                return
            self.scale_to_fit()
            res = self.check()
            if res is None:
                return
        args, tgt = res
        out = Path(args.output)
        if out.exists() and not messagebox.askyesno(
                "Replace the file?", f"{out.name} already exists in\n{out.parent}\n\nReplace it?", parent=self):
            return
        out.parent.mkdir(parents=True, exist_ok=True)
        self.set_busy(True)
        self.log(f"Writing {shown(out)} ...\n\n", clear=True)
        self.app.status(f"Planning the moves for {out.name}; this takes a few seconds ...")
        q = self.app.queue

        def work():
            writer = QueueWriter(q)
            try:
                with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
                    print(f"Record: {tgt.label}, {tgt.t[-1]:.1f} s")
                    print("Bounds check:")
                    rtg.bounds_table(tgt, args.y_min, args.y_max, args.max_vel, args.max_accel)
                    print()
                    result = eng.run_motion(tgt, args, PROFILE)
                q.put(("done", (args, tgt, result)))
            except SystemExit as e:
                q.put(("failed", friendly(e, self.LABELS)))
            except Exception:  # show anything unexpected in the log rather than losing it
                q.put(("failed", traceback.format_exc()))

        threading.Thread(target=work, daemon=True).start()

    def set_busy(self, busy):
        self.busy = busy
        self.write_button.configure(state="disabled" if busy else "normal")
        self.fit_button.configure(state="disabled" if busy or self.fit is None else "normal")
        if busy:
            self.progress.pack(fill="x", pady=(8, 0))
            self.progress.start(12)
        else:
            self.progress.stop()
            self.progress.pack_forget()
        self.app.busy_cursor(busy)

    def finished(self, args, tgt, result):
        self.set_busy(False)
        out = Path(args.output)
        self.out_folder = out.parent
        self.folder_button.configure(state="normal")
        self.draw(tgt, args, result)
        self.log(f"\nDone. Copy {out.name} to the SD card and print it.\n")
        warn = f"  {len(result['warnings'])} warning(s), see the log." if result["warnings"] else ""
        self.app.status(f"Wrote {shown(out)}: predicted timing within {result['max_drift'] * 1000:.1f} ms, "
                        f"position within {result['max_x_err']:.2f} mm. Copy it to the SD card.{warn}",
                        error=bool(warn))

    def failed(self, message):
        self.set_busy(False)
        self.log("\n" + message + "\n")
        self.app.status("Couldn't write the G-code; see the log.", error=True)
        messagebox.showerror("Couldn't write the G-code", message.strip().splitlines()[-1], parent=self)


# --------------------------------------------------------------------------- window

class App:
    def __init__(self, root):
        self.root = root
        root.title("Shake table: Ender 3 S1 Pro")
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        w, h = min(1360, int(0.95 * sw)), min(900, int(0.88 * sh))
        root.geometry(f"{w}x{h}+{(sw - w) // 2}+{max(0, (sh - h) // 3)}")
        root.minsize(min(1000, w), min(640, h))
        style = ttk.Style(root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Section.TLabel", font=("Segoe UI", 10, "bold"))
        style.configure("Hint.TLabel", foreground=MUTED)
        style.configure("Summary.TLabel", foreground=INK_2)
        style.configure("Num.TLabel", font=("Segoe UI", 9))
        style.configure("Primary.TButton", font=("Segoe UI", 10, "bold"), padding=6)
        style.configure("Error.TLabel", foreground=CRITICAL)
        self.queue = queue.Queue()

        status = ttk.Frame(root, padding=(10, 4))
        status.pack(fill="x", side="bottom")  # packed first so the tabs can't squeeze it out
        self.status_label = ttk.Label(status, text="")
        self.status_label.pack(side="left")
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True, padx=8, pady=(8, 0))
        self.convert = ConvertTab(self, self.notebook)
        self.record = RecordTab(self, self.notebook)
        self.notebook.add(self.record, text="   1  Make a record   ")
        self.notebook.add(self.convert, text="   2  Convert to G-code   ")
        root.after(100, self.poll)
        root.after(150, self.record.generate)

    def status(self, text, error=False):
        self.status_label.configure(text=("⚠  " if error else "") + text,
                                    style="Error.TLabel" if error else "TLabel")

    def busy_cursor(self, on):
        self.root.configure(cursor="watch" if on else "")
        self.root.update_idletasks()

    def poll(self):
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                if kind == "log":
                    self.convert.log(payload)
                elif kind == "done":
                    self.convert.finished(*payload)
                elif kind == "failed":
                    self.convert.failed(payload)
        except queue.Empty:
            pass
        self.root.after(100, self.poll)


def main():
    if sys.platform == "win32":  # sharp text on scaled displays
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()

"""Figures only: every function here takes finished arrays and draws/saves them.

No FFTs, beamforming or file reading happen here; those live in lfm_utils,
rdc_utils and rdc_io. Common layout: a plot on the right and a parameter table
("info panel") on the left, built from ``info_rows`` (list of (key, value)).
"""

import os
import shutil
import subprocess
import tempfile
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PLOT_RECT = [0.34, 0.12, 0.58, 0.80]
INFO_RECT = [0.015, 0.12, 0.216, 0.80]
LFM_OVERLAY_COLOR = "#18d7ff"


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def new_figure(figsize=(14, 6)):
    fig = plt.figure(figsize=figsize)
    return fig, fig.add_axes(PLOT_RECT)


def add_info_panel(fig, info_rows=None, notes=None, footer=None):
    """Parameter table on the left. ``notes`` are extra "key: value" lines under "Analysis"."""
    rows = list(info_rows or [])
    if notes:
        rows.append(("Analysis", ""))
        for line in str(notes).splitlines():
            key, _, value = line.partition(":")
            rows.append((key.strip(), value.strip()) if value else ("", line.strip()))
    ax = fig.add_axes(INFO_RECT)
    ax.axis("off")
    if rows:
        cells = [[str(k), ""] if v == "" else ["\n".join(textwrap.wrap(str(k), 13) or [""]),
                                                "\n".join(textwrap.wrap(str(v), 32) or [""])] for k, v in rows]
        lines = [1] + [max(k.count("\n"), v.count("\n")) + 1 for k, v in cells]   # header + wrapped rows
        n = sum(lines)
        size = 6.4 if n > 24 else (6.9 if n > 18 else 7.5)
        table = ax.table(cellText=cells, colLabels=["Parameter", "Value"], cellLoc="left", colLoc="left",
                         loc="upper left", colWidths=[0.34, 0.66])
        table.auto_set_font_size(False)
        table.set_fontsize(size)
        unit = min(0.045, 0.95 / n)
        for (r, c), cell in table.get_celld().items():
            cell.set_height(unit * (lines[r] if lines[r] == 1 else lines[r] * 0.85 + 0.3))
            cell.set_linewidth(0.35)
            cell.set_edgecolor("#c9c9c9")
            header = r == 0 or (c == 0 and cells[r - 1][1] == "" and cells[r - 1][0])
            cell.set_facecolor("#eeeeee" if r == 0 else ("#f6f6f6" if header else "white"))
            if header:
                cell.set_text_props(weight="bold")
        if footer:
            fig.canvas.draw()
            box = table.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
            fig.text(INFO_RECT[0], max(0.015, box.y0 - 0.012), "\n".join(textwrap.wrap(str(footer), 48)),
                     ha="left", va="top", fontsize=size,
                     bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 3})
    elif footer:
        fig.text(INFO_RECT[0], INFO_RECT[1] + INFO_RECT[3], "\n".join(textwrap.wrap(str(footer), 48)),
                 ha="left", va="top", fontsize=7.5)


def colorbar(fig, mappable, ax, label, pad=0.16):
    cb = fig.colorbar(mappable, ax=ax, pad=pad)
    cb.set_label(label)
    return cb


def save(fig, path, dpi=300):
    path = os.path.expanduser(path)
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def _finish(fig, ax, title, output, info_rows, notes, footer):
    ax.set_title(title)
    add_info_panel(fig, info_rows, notes, footer)
    return save(fig, output) if output else plt.show()


def delay_peaks_note(delays_ms):
    return "\n".join(f"Delay peak {i}: {d:.3f} ms" for i, d in enumerate(delays_ms, 1)) or None


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_spectrogram(freq_hz, time_s, power_db, title, output=None, vmin=None, vmax=None,
                     info_rows=None, notes=None, footer=None, lfm_overlay=None):
    """Frequency (x, kHz) vs time (y). ``lfm_overlay``: dict with sweep_period_s, f_start_hz, f_end_hz, first_sweep_start_s."""
    fig, ax = new_figure()
    pcm = ax.pcolormesh(freq_hz / 1e3, time_s, power_db.T, shading="gouraud", cmap="inferno", vmin=vmin, vmax=vmax)
    ax.set(xlabel="Frequency [kHz]", ylabel="Time [s]")
    colorbar(fig, pcm, ax, "Power [dB]", pad=0.02)
    if lfm_overlay and lfm_overlay.get("sweep_period_s"):
        period, off = lfm_overlay["sweep_period_s"], lfm_overlay.get("first_sweep_start_s") or 0.0
        t0, t1 = float(time_s[0]), float(time_s[-1])
        for start in np.arange(off - np.ceil((off - t0) / period) * period, t1 + period, period):
            t = np.clip([start, start + period], t0, t1)
            f = lfm_overlay["f_start_hz"] + (t - start) / period * (lfm_overlay["f_end_hz"] - lfm_overlay["f_start_hz"])
            ax.plot(f / 1e3, t, color=LFM_OVERLAY_COLOR, lw=0.6, alpha=0.45, ls="--")
    return _finish(fig, ax, title, output, info_rows, notes, footer)


def plot_average_spectrum(freq_hz, power_db, title, output=None, vmin=None, vmax=None, info_rows=None, footer=None):
    fig, ax = new_figure()
    ax.plot(freq_hz / 1e3, power_db)
    ax.set(xlabel="Frequency [kHz]", ylabel="Power [dB/RBW]", ylim=(vmin, vmax))
    ax.grid(True, alpha=0.3)
    return _finish(fig, ax, title, output, info_rows, None, footer)


def plot_delay_time(time_s, delay_ms, power_db, title, output=None, vmin=None, vmax=None,
                    info_rows=None, notes=None, footer=None):
    """Range-time intensity: time (x) vs delay (y); ``power_db`` is (time, delay)."""
    fig, ax = new_figure()
    pcm = ax.pcolormesh(time_s, delay_ms, power_db.T, shading="nearest", cmap="inferno", vmin=vmin, vmax=vmax)
    ax.set(xlabel="Time [s]", ylabel="Delay [ms]")
    colorbar(fig, pcm, ax, "Power [dB]")
    return _finish(fig, ax, title, output, info_rows, notes, footer)


def plot_power_delay(delay_ms, power_db, title, output=None, vmin=None, vmax=None, info_rows=None, notes=None, footer=None):
    fig, ax = new_figure(figsize=(14, 4.8))
    ax.plot(delay_ms, power_db)
    ax.set(xlabel="Delay [ms]", ylabel="Mean power [dB]")
    ax.set_ylim(bottom=vmin, top=vmax)
    ax.grid(True)
    return _finish(fig, ax, title, output, info_rows, notes, footer)


def plot_delay_doppler(doppler_hz, delay_ms, power_db, title, output=None, vmin=None, vmax=None,
                       info_rows=None, notes=None, footer=None):
    """Doppler (x) vs delay (y); ``power_db`` is (delay, doppler)."""
    fig, ax = new_figure()
    pcm = ax.pcolormesh(doppler_hz, delay_ms, power_db, shading="nearest", cmap="inferno", vmin=vmin, vmax=vmax)
    ax.set(xlabel="Doppler Frequency [Hz]", ylabel="Delay [ms]")
    colorbar(fig, pcm, ax, "Power [dB]")
    return _finish(fig, ax, title, output, info_rows, notes, footer)


def plot_doppler_time(time_s, doppler_hz, power_db, title, output=None, vmin=None, vmax=None,
                      info_rows=None, notes=None, footer=None):
    """Time (x) vs Doppler (y); ``power_db`` is (time, doppler)."""
    fig, ax = new_figure()
    pcm = ax.pcolormesh(time_s, doppler_hz, power_db.T, shading="nearest", cmap="inferno", vmin=vmin, vmax=vmax)
    ax.set(xlabel="Time [s]", ylabel="Doppler Frequency [Hz]")
    colorbar(fig, pcm, ax, "Power [dB]")
    return _finish(fig, ax, title, output, info_rows, notes, footer)


def plot_phase_doppler_delay_time(time_s, delay_ms, fd_hz, fd_lim, title, output=None,
                                  info_rows=None, notes=None, footer=None):
    """Delay vs time coloured by phase-derived Doppler; ``fd_hz`` is (time, delay), NaN = masked."""
    fig, ax = new_figure()
    ax.set_facecolor("#050505")
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#050505")
    pcm = ax.pcolormesh(time_s, delay_ms, np.ma.masked_invalid(fd_hz.T), shading="nearest", cmap=cmap,
                        vmin=fd_lim[0], vmax=fd_lim[1])
    ax.set(xlabel="Time [s]", ylabel="Delay [ms]")
    colorbar(fig, pcm, ax, "Phase-derived Doppler [Hz]")
    return _finish(fig, ax, title, output, info_rows, notes, footer)


def plot_phase_doppler_time(time_s, fd_centers, power_db, title, output=None, vmin=None, vmax=None,
                            info_rows=None, notes=None, footer=None):
    """Time (x) vs phase-derived Doppler bins (y); ``power_db`` is (time, bins)."""
    fig, ax = new_figure()
    ax.set_facecolor("#050505")
    cmap = plt.get_cmap("inferno").copy()
    cmap.set_bad("#050505")
    pcm = ax.pcolormesh(time_s, fd_centers, np.ma.masked_invalid(power_db.T), shading="nearest", cmap=cmap,
                        vmin=vmin, vmax=vmax)
    ax.set(xlabel="Time [s]", ylabel="Phase-derived Doppler [Hz]")
    colorbar(fig, pcm, ax, "Power [dB]")
    return _finish(fig, ax, title, output, info_rows, notes, footer)


def plot_phase_doppler_frame(fd_hz, delay_ms, power_db, fd_lim, title, output=None, vmin=None, vmax=None,
                             info_rows=None, notes=None, footer=None):
    """One estimate: scatter of (Doppler, delay) coloured by power; NaN entries are skipped."""
    fig, ax = new_figure()
    ax.set_facecolor("#050505")
    ok = np.isfinite(fd_hz) & np.isfinite(power_db)
    if ok.any():
        sc = ax.scatter(fd_hz[ok], delay_ms[ok], c=power_db[ok], s=18, cmap="inferno", vmin=vmin, vmax=vmax, linewidths=0)
        colorbar(fig, sc, ax, "Power [dB]")
    else:
        ax.text(0.5, 0.5, "No valid phase-Doppler estimates", transform=ax.transAxes, ha="center", color="white")
    ax.set(xlim=fd_lim, ylim=(float(np.min(delay_ms)), float(np.max(delay_ms))),
           xlabel="Phase-derived Doppler [Hz]", ylabel="Delay [ms]")
    return _finish(fig, ax, title, output, info_rows, notes, footer)


def plot_angle_maps(maps, az_deg, el_deg, labels, title, output, range_db=30.0,
                    info_rows=None, notes=None, footer=None, markers=None, annotations=None, side_lines=None):
    """Polar sky maps (north up, east right, zenith at centre), each normalised to its own peak.

    ``maps`` is a list of (el, az) linear power arrays. ``markers`` optionally gives
    one (az, el) refined peak per map, drawn as a cross. ``annotations`` optionally
    gives per map ``None`` or ``dict(label=str, points=[(az, el), ...])``: the label
    is written at the cross and the points (e.g. station bearings) are drawn as
    small triangles. ``side_lines`` adds a text panel on the right.
    """
    az_deg = np.asarray(az_deg, float)
    el_deg = np.asarray(el_deg, float)
    if az_deg.size > 1 and np.isclose(abs(az_deg[-1] - az_deg[0]), 360.0):
        az_deg, maps = az_deg[:-1], [m[:, :-1] for m in maps]
    theta = np.deg2rad(_edges(az_deg))
    radius = _edges(90.0 - el_deg[::-1])
    n = len(maps)
    cols = min(3, n)
    rows_n = int(np.ceil(n / cols))
    fig = plt.figure(figsize=(18 if side_lines else 14, max(5.2, 4.3 * rows_n)))
    left, bottom, width, height = [0.255, PLOT_RECT[1], 0.46, PLOT_RECT[3]] if side_lines else PLOT_RECT
    w = (width - 0.08 * (cols - 1)) / cols
    h = (height - 0.09 * (rows_n - 1)) / rows_n
    pcm = None
    for i, m in enumerate(maps):
        r, c = divmod(i, cols)
        ax = fig.add_axes([left + c * (w + 0.08), bottom + (rows_n - 1 - r) * (h + 0.09), w, h], projection="polar")
        peak = np.nanmax(m)
        db = 10 * np.log10(np.maximum(m / peak, 1e-30)) if np.isfinite(peak) and peak > 0 else np.full(m.shape, np.nan)
        pcm = ax.pcolormesh(theta, radius, db[::-1], shading="auto", cmap="inferno", vmin=-abs(range_db), vmax=0)
        if markers is not None and markers[i] is not None:
            ax.plot(np.deg2rad(markers[i][0]), 90 - markers[i][1], "c+", ms=10, mew=1.5)
        note = annotations[i] if annotations is not None and i < len(annotations) else None
        if note:
            for paz, pel in note.get("points", []):
                ax.plot(np.deg2rad(paz), 90 - pel, marker="^", ms=4, color="white", mec="black", mew=0.4, ls="none")
            if markers is not None and markers[i] is not None and note.get("label"):
                ax.annotate("\n".join(textwrap.wrap(note["label"], 22)), (np.deg2rad(markers[i][0]), 90 - markers[i][1]),
                            xytext=(8, 8), textcoords="offset points", fontsize=7, color="white",
                            bbox={"facecolor": "black", "alpha": 0.6, "edgecolor": "none", "pad": 1.5})
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_rlim(0, 90)
        ax.set_rticks([0, 30, 60, 90])
        ax.set_yticklabels(["90", "60", "30", "0"])
        ax.set_rlabel_position(135)
        ax.grid(color="white", alpha=0.28, lw=0.6)
        ax.tick_params(labelsize=8.5, pad=1)
        ax.set_title(labels[i] if labels else "", fontsize=9)
    if pcm is not None:
        if side_lines:
            cb = fig.colorbar(pcm, cax=fig.add_axes([left + width + 0.02, bottom, 0.01, height]))
        else:
            cb = fig.colorbar(pcm, ax=fig.axes, pad=0.08)
        cb.set_label("Power relative to map peak [dB]")
    if side_lines:
        fig.text(left + width + 0.075, bottom + height, "\n".join(side_lines), ha="left", va="top",
                 fontsize=6.6, family="monospace")
    fig.suptitle(title)
    add_info_panel(fig, info_rows, notes, footer)
    return save(fig, output)


# ---------------------------------------------------------------------------
# Wavefront tracking (plot_wavefront.py)
# ---------------------------------------------------------------------------

def plot_wavefront_summary(t_s, series, uv, title, output, info_rows=None, notes=None, footer=None):
    """Stacked time series sharing one time axis, plus the arrival-direction track.

    ``series``: list of (label, y, extra) where extra is an optional
    (label, y) drawn as grey dots on the same axis (e.g. the coarse grid fit).
    ``uv``: (u, v, u_ref, v_ref) direction cosines (east, north) for the track panel.
    """
    t_s = np.asarray(t_s, float)
    fig = plt.figure(figsize=(18, max(8.0, 1.55 * len(series) + 1.5)))
    left, bottom, width, height = 0.30, PLOT_RECT[1], 0.40, PLOT_RECT[3]
    gap = 0.012
    h = (height - gap * (len(series) - 1)) / len(series)
    axes = []
    for i, (label, y, extra) in enumerate(series):
        ax = fig.add_axes([left, bottom + (len(series) - 1 - i) * (h + gap), width, h],
                          sharex=axes[0] if axes else None)
        if extra is not None:
            ax.plot(t_s, extra[1], ".", ms=3, color="#9a9a9a", label=extra[0])
        ax.plot(t_s, y, "-", lw=1.4, color="#1f5fa8", label=label if extra is not None else None)
        ax.set_ylabel(label, fontsize=8.5)
        ax.grid(alpha=0.25, lw=0.6)
        ax.tick_params(labelsize=8)
        if extra is not None:
            ax.legend(fontsize=7, loc="upper right", frameon=False)
        if i < len(series) - 1:
            ax.tick_params(labelbottom=False)
        axes.append(ax)
    axes[-1].set_xlabel("Time [s]")
    axes[0].set_title(title)
    u, v, u_ref, v_ref = uv
    ax = fig.add_axes([0.76, bottom + 0.22, 0.19, 0.45])
    ok = np.isfinite(u) & np.isfinite(v)
    sc = ax.scatter(np.asarray(u)[ok] - u_ref, np.asarray(v)[ok] - v_ref, c=t_s[ok], s=9, cmap="viridis")
    ax.plot(np.asarray(u)[ok] - u_ref, np.asarray(v)[ok] - v_ref, "-", lw=0.6, color="#777777", alpha=0.6)
    ax.plot(0, 0, "+", color="black", ms=10, mew=1.2)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set(xlabel="East direction cosine change", ylabel="North direction cosine change")
    ax.set_title("Arrival-direction track\n(+ = session mean)", fontsize=9)
    ax.ticklabel_format(style="sci", scilimits=(-2, 2))
    ax.tick_params(labelsize=7.5)
    ax.grid(alpha=0.25, lw=0.6)
    cb = fig.colorbar(sc, ax=ax, pad=0.04, fraction=0.06)
    cb.set_label("Time [s]", fontsize=8)
    add_info_panel(fig, info_rows, notes, footer)
    return save(fig, output)


def _stand_map(ax, xy, values, vmax, cmap="RdBu_r", size=22):
    ok = np.isfinite(values)
    ax.scatter(xy[~ok, 0], xy[~ok, 1], s=size * 0.6, facecolors="none", edgecolors="#b5b5b5", lw=0.6)
    sc = ax.scatter(xy[ok, 0], xy[ok, 1], c=values[ok], s=size, cmap=cmap, vmin=-vmax, vmax=vmax,
                    edgecolors="#555555", linewidths=0.3)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.2, lw=0.5)
    return sc


def plot_phase_maps(xy, maps, labels, title, output, static=None, unit="rad", vmax=None,
                    info_rows=None, notes=None, footer=None):
    """Small multiples of per-stand phase on the ground (x east, y north [m]).

    ``maps``: list of (N,) arrays sharing one diverging colour scale (symmetric,
    ``vmax`` or the 99th percentile). ``static``: optional (N,) map drawn first
    with its own scale (the per-stand error left after calibration). Hollow
    circles are stands without a value."""
    xy = np.asarray(xy, float)
    panels = ([("static", static)] if static is not None else []) + list(zip(labels, maps))
    cols = min(4, len(panels))
    rows_n = int(np.ceil(len(panels) / cols))
    fig = plt.figure(figsize=(18, max(5.5, 3.9 * rows_n + 1.0)))
    left, bottom, width, height = 0.29, PLOT_RECT[1], 0.62, PLOT_RECT[3]
    w = (width - 0.035 * (cols - 1)) / cols
    h = (height - 0.07 * (rows_n - 1)) / rows_n
    allv = np.concatenate([np.ravel(m) for m in maps]) if maps else np.array([np.nan])
    scale = vmax or (np.nanpercentile(np.abs(allv), 99) if np.isfinite(allv).any() else 1.0) or 1.0
    shared = None
    for i, (label, values) in enumerate(panels):
        r, c = divmod(i, cols)
        ax = fig.add_axes([left + c * (w + 0.035), bottom + (rows_n - 1 - r) * (h + 0.07), w, h])
        values = np.asarray(values, float)
        if label == "static" and static is not None:
            s_scale = np.nanpercentile(np.abs(values), 99) if np.isfinite(values).any() else 1.0
            sc = _stand_map(ax, xy, values, s_scale or 1.0)
            ax.set_title("Static residual (time mean minus its plane):\nper-stand error left after calibration",
                         fontsize=8)
            cb = fig.colorbar(sc, ax=ax, orientation="horizontal", fraction=0.05, pad=0.16 if r == rows_n - 1 else 0.06)
            cb.set_label(f"Static residual [{unit}]", fontsize=7.5)
            cb.ax.tick_params(labelsize=7)
        else:
            shared = _stand_map(ax, xy, values, scale)
            ax.set_title(label, fontsize=8.5)
        if r == rows_n - 1:
            ax.set_xlabel("East [m]", fontsize=8)
        if c == 0:
            ax.set_ylabel("North [m]", fontsize=8)
    if shared is not None:
        cb = fig.colorbar(shared, cax=fig.add_axes([left + width + 0.012, bottom, 0.009, height]))
        cb.set_label(f"Phase [{unit}]")
    fig.suptitle(title)
    add_info_panel(fig, info_rows, notes, footer)
    return save(fig, output)


def plot_phase_surface(xy, z, color, plane, title, output, zlim, clim, color_label, info_rows=None,
                       notes=None, footer=None, cmap="RdBu_r"):
    """3D view: stands at (east, north) [m], height ``z`` [rad], coloured by ``color``.

    ``plane``: (du_term_e, du_term_n, offset) so the fitted plane is
    offset + a*x + b*y [rad], drawn as a translucent sheet."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the projection)
    xy = np.asarray(xy, float)
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_axes([0.28, 0.06, 0.62, 0.86], projection="3d")
    ok = np.isfinite(z)
    if plane is not None and all(np.isfinite(plane)):
        gx = np.linspace(xy[:, 0].min(), xy[:, 0].max(), 12)
        gy = np.linspace(xy[:, 1].min(), xy[:, 1].max(), 12)
        GX, GY = np.meshgrid(gx, gy)
        GZ = plane[2] + plane[0] * GX + plane[1] * GY
        GZ = np.where((GZ >= zlim[0]) & (GZ <= zlim[1]), GZ, np.nan)      # keep the sheet inside the box
        ax.plot_surface(GX, GY, GZ, color="#7d7d7d", alpha=0.18,
                        linewidth=0, antialiased=True)
    if ok.any():
        for x, y, h in zip(xy[ok, 0], xy[ok, 1], np.asarray(z)[ok]):
            ax.plot([x, x], [y, y], [zlim[0], h], color="#c4c4c4", lw=0.4)
        cvals = np.asarray(color)[ok]
        sc = ax.scatter(xy[ok, 0], xy[ok, 1], np.asarray(z)[ok], c=cvals, cmap=cmap, vmin=clim[0], vmax=clim[1],
                        s=26, edgecolors="#444444", linewidths=0.3, depthshade=False)
        cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.08)
        cb.set_label(color_label)
    ax.set_zlim(*zlim)
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_zlabel("Phase change [rad]")
    ax.view_init(elev=24, azim=-58)
    ax.set_title(title)
    add_info_panel(fig, info_rows, notes, footer)
    return save(fig, output, dpi=160)


def _edges(centers):
    c = np.asarray(centers, float)
    if c.size == 1:
        return np.array([c[0] - 0.5, c[0] + 0.5])
    mid = 0.5 * (c[:-1] + c[1:])
    return np.concatenate([[c[0] - (mid[0] - c[0])], mid, [c[-1] + (c[-1] - mid[-1])]])


# ---------------------------------------------------------------------------
# Videos
# ---------------------------------------------------------------------------

class FrameVideo:
    """Collect numbered PNG frames in a private folder and encode them with ffmpeg."""

    def __init__(self, output_dir, name, framerate=2, keep_frames=False):
        self.output_dir = os.path.expanduser(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self.dir = tempfile.mkdtemp(prefix=f"{name}_frames_", dir=self.output_dir)
        self.name, self.framerate, self.keep, self.count = name, framerate, keep_frames, 0

    def next_path(self):
        path = os.path.join(self.dir, f"frame_{self.count:04d}.png")
        self.count += 1
        return path

    def finish(self):
        if not self.count:
            shutil.rmtree(self.dir, ignore_errors=True)
            raise ValueError("No frames were produced")
        video = os.path.join(self.output_dir, f"{self.name}.mp4")
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(self.framerate),
               "-i", os.path.join(self.dir, "frame_%04d.png"), "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", video]
        try:
            subprocess.run(cmd, check=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            print(f"Video encoding failed ({exc}); frames kept in {self.dir}", flush=True)
            return None
        print(f"Video: {video}", flush=True)
        if self.keep:
            print(f"Frames: {self.dir}", flush=True)
        else:
            shutil.rmtree(self.dir, ignore_errors=True)
        return video

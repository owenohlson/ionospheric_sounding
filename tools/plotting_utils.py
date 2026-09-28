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
        n = len(cells) + 1
        size = 6.4 if n > 24 else (6.9 if n > 18 else 7.5)
        table = ax.table(cellText=cells, colLabels=["Parameter", "Value"], cellLoc="left", colLoc="left",
                         loc="upper left", colWidths=[0.34, 0.66])
        table.auto_set_font_size(False)
        table.set_fontsize(size)
        height = min(0.045, 0.9 / n)
        for (r, c), cell in table.get_celld().items():
            cell.set_height(height)
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
                    info_rows=None, notes=None, footer=None, markers=None):
    """Polar sky maps (north up, east right, zenith at centre), each normalised to its own peak.

    ``maps`` is a list of (el, az) linear power arrays. ``markers`` optionally gives
    one (az, el) refined peak per map, drawn as a cross.
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
    fig = plt.figure(figsize=(14, max(5.2, 4.3 * rows_n)))
    left, bottom, width, height = PLOT_RECT
    w = (width - 0.035 * (cols - 1)) / cols
    h = (height - 0.09 * (rows_n - 1)) / rows_n
    pcm = None
    for i, m in enumerate(maps):
        r, c = divmod(i, cols)
        ax = fig.add_axes([left + c * (w + 0.035), bottom + (rows_n - 1 - r) * (h + 0.09), w, h], projection="polar")
        peak = np.nanmax(m)
        db = 10 * np.log10(np.maximum(m / peak, 1e-30)) if np.isfinite(peak) and peak > 0 else np.full(m.shape, np.nan)
        pcm = ax.pcolormesh(theta, radius, db[::-1], shading="auto", cmap="inferno", vmin=-abs(range_db), vmax=0)
        if markers is not None and markers[i] is not None:
            ax.plot(np.deg2rad(markers[i][0]), 90 - markers[i][1], "c+", ms=10, mew=1.5)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_rlim(0, 90)
        ax.set_rticks([0, 30, 60, 90])
        ax.set_yticklabels(["90", "60", "30", "0"])
        ax.set_rlabel_position(135)
        ax.grid(color="white", alpha=0.28, lw=0.6)
        ax.set_title(labels[i] if labels else "", fontsize=9)
    if pcm is not None:
        cb = fig.colorbar(pcm, ax=fig.axes, pad=0.08)
        cb.set_label("Power relative to map peak [dB]")
    fig.suptitle(title)
    add_info_panel(fig, info_rows, notes, footer)
    return save(fig, output)


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

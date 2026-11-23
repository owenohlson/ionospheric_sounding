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
from matplotlib.lines import Line2D

PLOT_RECT = [0.34, 0.12, 0.58, 0.80]
INFO_RECT = [0.015, 0.12, 0.216, 0.80]
LFM_OVERLAY_COLOR = "#18d7ff"
SIDE_PANEL_CHARS = 64        # width of the sky map's station panel; longer lines wrap, so the layout never moves


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def new_figure(figsize=(14, 6)):
    fig = plt.figure(figsize=figsize)
    return fig, fig.add_axes(PLOT_RECT)


def add_info_panel(fig, info_rows=None, notes=None, footer=None, rect=None):
    """Parameter table on the left. ``notes`` are extra "key: value" lines under "Analysis".
    ``rect`` (figure fractions) overrides INFO_RECT."""
    rect = list(rect or INFO_RECT)
    rows = list(info_rows or [])
    if notes:
        rows.append(("Analysis", ""))
        for line in str(notes).splitlines():
            key, _, value = line.partition(":")
            rows.append((key.strip(), value.strip()) if value else ("", line.strip()))
    ax = fig.add_axes(rect)
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
            fig.text(rect[0], max(0.015, box.y0 - 0.012), "\n".join(textwrap.wrap(str(footer), 48)),
                     ha="left", va="top", fontsize=size,
                     bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 3})
    elif footer:
        fig.text(rect[0], rect[1] + rect[3], "\n".join(textwrap.wrap(str(footer), 48)),
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
# Power scales (as in Peter Driessen's lwa_radar.py)
# ---------------------------------------------------------------------------
# Absolute dB of the processed data are counts^2 times processing gains (sweep length,
# sweeps per frame, channels summed), so they change with every setting. Maps and
# profiles are therefore drawn in dB above their own median (the noise floor for
# sparse echoes), colour range 0 dB to the map's peak (at least 10 dB; Peter used the 99.99th
# percentile, which clips to the noise when an echo covers only a few cells);
# spectra and spectrograms in dB below their peak, -40 to 0 dB. --vmin/--vmax are
# on these scales. Only the drawing changes: findings and SNRs use the data.

MEDIAN_LABEL = "Power above map median [dB]"
PEAK_LABEL = "Power re peak [dB]"          # callers say whose peak


def above_median_db(power_db):
    """dB above the median of the finite values (10 log10(P / median P))."""
    p = np.array(power_db, dtype=float)
    finite = np.isfinite(p)
    if finite.any():
        p -= float(np.median(p[finite]))
    return p


def re_peak_db(power_db):
    """dB relative to the largest finite value."""
    p = np.array(power_db, dtype=float)
    finite = np.isfinite(p)
    if finite.any():
        p -= float(np.max(p[finite]))
    return p


def median_limits(arrays, vmin=None, vmax=None):
    """Colour range of maps above their median: 0 dB to max(10 dB, the peak) unless given."""
    vmin = 0.0 if vmin is None else float(vmin)
    if vmax is None:
        tops = [float(np.nanmax(a)) for a in arrays if a is not None and np.any(np.isfinite(a))]
        vmax = max([vmin + 10.0] + tops)
    return vmin, float(vmax)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_spectrogram(freq_hz, time_s, power_db, title, output=None, vmin=None, vmax=None,
                     info_rows=None, notes=None, footer=None, lfm_overlay=None):
    """Frequency (x, kHz) vs time (y). ``lfm_overlay``: dict with sweep_period_s, f_start_hz, f_end_hz, first_sweep_start_s."""
    fig, ax = new_figure()
    power_db = re_peak_db(power_db)
    vmin, vmax = (-40.0 if vmin is None else vmin), (0.0 if vmax is None else vmax)
    pcm = ax.pcolormesh(freq_hz / 1e3, time_s, power_db.T, shading="gouraud", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set(xlabel="Frequency [kHz]", ylabel="Time [s]")
    colorbar(fig, pcm, ax, "Power re spectrogram peak [dB]", pad=0.02)
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
    ax.plot(freq_hz / 1e3, re_peak_db(power_db))
    ax.set(xlabel="Frequency [kHz]", ylabel="Power re spectrum peak [dB]", ylim=(vmin, vmax))
    ax.grid(True, alpha=0.3)
    return _finish(fig, ax, title, output, info_rows, None, footer)


def plot_power_delay_profile(time_s, delay_ms, power_db, mean_db, title, output=None, vmin=None, vmax=None,
                             info_rows=None, notes=None, footer=None, mean_label="Time average"):
    """Power vs time (x) and delay (y), with the time-averaged profile beside it on the same delay axis.

    ``power_db`` is (time, delay); ``mean_db`` is (delay,). Both are drawn in dB above the median
    of the image, so they read on one scale; ``vmin``/``vmax`` (on that scale) set the colour range
    and the profile's power axis."""
    ref = float(np.nanmedian(power_db)) if np.any(np.isfinite(power_db)) else 0.0
    power_db, mean_db = np.asarray(power_db, float) - ref, np.asarray(mean_db, float) - ref
    vmin, vmax = median_limits([power_db], vmin, vmax)
    fig = plt.figure(figsize=(14, 6))
    x0, y0, w, h = PLOT_RECT
    ax = fig.add_axes([x0, y0, 0.66 * w, h])
    cax = fig.add_axes([x0 + 0.675 * w, y0, 0.012, h])
    prof = fig.add_axes([x0 + 0.83 * w, y0, 0.17 * w, h], sharey=ax)
    pcm = ax.pcolormesh(time_s, delay_ms, power_db.T, shading="nearest", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set(xlabel="Time [s]", ylabel="Delay [ms]")
    fig.colorbar(pcm, cax=cax).set_label("Power above image median [dB]")
    prof.plot(mean_db, delay_ms, lw=1.2, color="#1f5fa8")
    prof.set_xlim(vmin, max(vmax, float(np.nanmax(mean_db)) + 1.0) if np.any(np.isfinite(mean_db)) else vmax)
    prof.set_xlabel("Mean power [dB]\nabove image median")
    prof.set_title(mean_label, fontsize=9)
    prof.tick_params(labelleft=False)
    prof.grid(True, alpha=0.3)
    return _finish(fig, ax, title, output, info_rows, notes, footer)


def plot_lfm_scan(lag_k, lag_power, lag_peaks, table_marks, found, title, output=None, info_rows=None,
                  notes=None, footer=None, measurements=()):
    """Waveform scan (scan_lfm.py): lag-product spectrum; sweep rate from the resets; bandwidth scan;
    delay profiles.

    ``found``: scan_lfm.Candidate objects. ``table_marks``: [dict(label, k)] chirp rates of the station-table
    waveforms in the band (dotted). ``measurements`` (lfm.measure_sweep_rate results of the lag peaks) is
    accepted but not used: the lag spectrum is always drawn against chirp rate (bandwidth x sweep rate),
    which it measures before any sweep rate is chosen."""
    fig = plt.figure(figsize=(16, 10.5))
    x0, y0, w, h = PLOT_RECT[0], 0.055, PLOT_RECT[2], 0.86
    gap = 0.075
    ph = (h - 2 * gap) / 3
    half = (w - 0.06) / 2
    ax_lag = fig.add_axes([x0, y0 + 2 * (ph + gap), w, ph])
    ax_reset = fig.add_axes([x0, y0 + ph + gap, half, ph])
    ax_fine = fig.add_axes([x0 + half + 0.06, y0 + ph + gap, half, ph])
    ax_prof = fig.add_axes([x0, y0, w, ph])
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    col = {id(c): colors[i % len(colors)] for i, c in enumerate(found)}

    def who(c):
        codes = [x for s in c.sources if s.startswith("table") for x in s[6:].split("/") if x and x != "..."]
        if c.kind == "not in table":
            return "not in table"
        out = "/".join(codes[:3]) + ("/..." if len(codes) > 3 else "") if codes else "documented"
        return out + (", table sign flipped" if c.kind == "table, sign flipped" else "")

    if lag_k is not None:
        scale = 1.0                                               # x = chirp rate (no sweep rate assumed yet)
        med = np.median(lag_power[lag_power > 0]) if np.any(lag_power > 0) else 1.0
        ax_lag.plot(lag_k / scale / 1e3, 10 * np.log10(np.maximum(lag_power, 1e-30) / med), lw=0.6, color="#555555")
        lo, hi = ax_lag.get_ylim()
        for m in {round(m["k"]): m for m in table_marks}.values():
            if lag_k[0] <= m["k"] <= lag_k[-1]:
                ax_lag.axvline(m["k"] / scale / 1e3, color="#9a9a9a", lw=0.6, ls=":")
        for k, db in lag_peaks:
            ax_lag.plot(k / scale / 1e3, db, "v", color="#555555", ms=5)
        handles = [Line2D([], [], color="#9a9a9a", lw=0.8, ls=":", label="station-table waveforms")]
        if len(lag_peaks):
            handles.append(Line2D([], [], marker="v", ls="", color="#555555", ms=5,
                                  label="spectrum peaks: candidate chirp rates, each tested"))
        for i, c in enumerate(found):
            xb = c.chirp_rate / scale / 1e3
            ax_lag.axvline(xb, color=col[id(c)], lw=1.2, alpha=0.7)
            ax_lag.annotate(f"#{i + 1}", (xb, 1.0), xycoords=ax_lag.get_xaxis_transform(), xytext=(3, -10),
                            textcoords="offset points", fontsize=8, color=col[id(c)], fontweight="bold")
            handles.insert(i, Line2D([], [], color=col[id(c)], lw=1.2,
                                     label=f"#{i + 1} {c.best_bandwidth / 1e3:+.3f} kHz, {c.rate:g} Hz ({who(c)})"))
        ax_lag.legend(handles=handles, loc="upper right", fontsize=7.5, framealpha=0.9)
        ax_lag.set_ylim(lo, hi + 0.3 * (hi - lo))                   # room for the legend
        ks = [abs(k) for k, _ in lag_peaks] + [abs(c.chirp_rate) for c in found]
        if ks:
            lim = min(1.4 * max(ks), abs(lag_k).max()) / scale / 1e3
            ax_lag.set_xlim(-lim, lim)
        ax_lag.set_xlabel("Chirp rate [kHz/s]")
        ax_lag.set_ylabel("Power re spectrum median [dB]")
        ax_lag.set_title("Lag-product spectrum", fontsize=10)
    else:
        ax_lag.text(0.5, 0.5, "Lag-product spectrum skipped (--no-blind)", ha="center", va="center",
                    transform=ax_lag.transAxes)
        ax_lag.set_axis_off()

    copy_marks = residual_marks = cleaned_any = False
    for i, c in enumerate(found):
        tag = f"#{i + 1}"
        if c.fine is not None:
            ax_fine.plot(c.bandwidth + c.fine[0] - c.best_bandwidth, c.fine[1], color=col[id(c)], lw=1, label=tag)
        m = c.rate_measurement
        if m is not None and len(m.get("rates", [])):
            ax_reset.plot(m["rates"], m["coherence_curve"], color=col[id(c)], lw=1,
                          label=f"{tag} {m['rate_hz']:.3f} Hz" if m.get("rate_hz") else f"{tag} not measured")
        d = np.asarray(c.delay_ms, float)
        y = 10 * np.log10(np.maximum(c.profile, 1e-30) / np.median(c.profile))
        step = np.median(np.diff(d)) if d.size > 1 else 0
        cut = np.flatnonzero(np.diff(d) > 5 * step) + 1          # delays the chirp band does not reach
        raw = getattr(c, "raw_profile", None)
        if raw is not None and (getattr(c, "comb", None) or {}).get("detected"):
            cleaned_any = True                                   # before the sideband CLEAN, same reference
            yr = 10 * np.log10(np.maximum(raw, 1e-30) / np.median(c.profile))
            ax_prof.plot(np.insert(d, cut, np.nan), np.insert(yr, cut, np.nan), color=col[id(c)], lw=0.5,
                         alpha=0.3)
        ax_prof.plot(np.insert(d, cut, np.nan), np.insert(y, cut, np.nan), color=col[id(c)], lw=0.7,
                     label=f"{tag} {c.label()}")
        for dd, _ in c.echoes:
            j = int(np.argmin(np.abs(d - dd)))
            ax_prof.annotate(f"{dd:.1f}", (dd, y[j]), xytext=(0, 4), textcoords="offset points", ha="center",
                             fontsize=7, color=col[id(c)])
        cps = [cp for cp in getattr(c, "copies", []) if not cp.get("suppressed")]
        res = [cp for cp in getattr(c, "copies", []) if cp.get("suppressed")]
        if cps:
            copy_marks = True
            ax_prof.plot([cp["delay_ms"] for cp in cps], [cp["db"] + 1.5 for cp in cps], "x", color=col[id(c)],
                         ms=4, mew=1)
        if res:
            residual_marks = True
            ax_prof.plot([cp["delay_ms"] for cp in res], [cp["db"] + 1.5 for cp in res], "o", mfc="none",
                         color=col[id(c)], ms=4, mew=0.8)
    ax_fine.set(xlabel="Bandwidth offset [Hz]", ylabel="Contrast: peak re profile median [dB]")
    ax_fine.set_title("Bandwidth scan", fontsize=10)
    ax_reset.set_xscale("log")
    ax_reset.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax_reset.set_xticks([0.25, 0.5, 1, 2, 4])
    ax_reset.set_xticklabels(["0.25", "0.5", "1", "2", "4"])
    ax_reset.set(xlabel="Trial sweep rate [Hz]", ylabel="Reset coherence", ylim=(0, 1.05))
    ax_reset.set_title("Sweep rate from the transmitter resets", fontsize=10)
    ax_prof.set(xlabel="Delay mod sweep [ms]", ylabel="Power re profile median [dB]")
    ax_prof.set_title("Delay profile, gating sidebands removed (faint: before)" if cleaned_any else "Delay profile",
                      fontsize=10)
    for ax in (ax_fine, ax_reset, ax_prof, ax_lag):
        ax.grid(True, alpha=0.3)
    if found:
        ax_fine.legend(fontsize=8, loc="lower center")
        handles, labels = ax_prof.get_legend_handles_labels()
        if cleaned_any:
            handles.append(Line2D([], [], color="#555555", lw=0.5, alpha=0.3))
            labels.append("before the sideband CLEAN")
        if residual_marks:
            handles.append(Line2D([], [], marker="o", ls="", mfc="none", color="#555555", ms=4, mew=0.8))
            labels.append("suppressed copies (residual after the CLEAN)")
        if copy_marks:
            handles.append(Line2D([], [], marker="x", ls="", color="#555555", ms=4, mew=1))
            labels.append("copies not removed (gating / sweep wrap)" if cleaned_any
                          else "copies (gating / sweep wrap)")
        ax_prof.legend(handles, labels, fontsize=8, loc="upper right")
        ax_reset.legend(fontsize=8, loc="upper left")
    else:
        for ax in (ax_reset, ax_fine, ax_prof):
            ax.text(0.5, 0.5, "No waveform compressed", ha="center", va="center", transform=ax.transAxes)
    fig.suptitle(title, x=x0 + w / 2, y=0.995, fontsize=12)
    add_info_panel(fig, info_rows, notes, footer)
    return save(fig, output) if output else plt.show()


def plot_delay_doppler(doppler_hz, delay_ms, power_db, title, output=None, vmin=None, vmax=None,
                       info_rows=None, notes=None, footer=None):
    """Doppler (x) vs delay (y); ``power_db`` is (delay, doppler), drawn in dB above its median."""
    fig, ax = new_figure()
    power_db = above_median_db(power_db)
    vmin, vmax = median_limits([power_db], vmin, vmax)
    pcm = ax.pcolormesh(doppler_hz, delay_ms, power_db, shading="nearest", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set(xlabel="Doppler Frequency [Hz]", ylabel="Delay [ms]")
    colorbar(fig, pcm, ax, MEDIAN_LABEL)
    return _finish(fig, ax, title, output, info_rows, notes, footer)


def plot_delay_doppler_beams(doppler_hz, delay_ms, panels, title, output=None, vmin=None, vmax=None,
                             info_rows=None, notes=None, footer=None):
    """One delay-Doppler panel per beam, side by side, sharing axes and colour scale.

    ``panels``: [(power_db or None, label)]; None draws an empty panel with the label as its text.
    Each panel is drawn in dB above its own median; the colour range is common to all panels."""
    n = len(panels)
    margin, table_in, panel_in, gap_in, lead_in, cb_in = 0.2, 3.9, 4.2, 0.35, 0.9, 1.2
    fig_w = margin + table_in + lead_in + n * panel_in + (n - 1) * gap_in + cb_in + margin
    fig_h = 6.0
    fig = plt.figure(figsize=(fig_w, fig_h))
    fx = lambda inches: inches / fig_w                    # noqa: E731
    bottom, height = PLOT_RECT[1], PLOT_RECT[3] - 0.06
    panels = [(None if p is None else above_median_db(p), label) for p, label in panels]
    vmin, vmax = median_limits([p for p, _ in panels], vmin, vmax)
    pcm, axes = None, []
    for j, (pdb, label) in enumerate(panels):
        x = margin + table_in + lead_in + j * (panel_in + gap_in)
        ax = fig.add_axes([fx(x), bottom, fx(panel_in), height], sharey=axes[0] if axes else None)
        axes.append(ax)
        if pdb is None:
            ax.text(0.5, 0.5, "\n".join(textwrap.wrap(label, 30)), ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="#666666")
            ax.set_xlim(doppler_hz[0], doppler_hz[-1])
        else:
            pcm = ax.pcolormesh(doppler_hz, delay_ms, pdb, shading="nearest", cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_title(label, fontsize=9.5)
        ax.set_xlabel("Doppler Frequency [Hz]")
        if j == 0:
            ax.set_ylabel("Delay [ms]")
        else:
            ax.tick_params(labelleft=False)
    if pcm is not None:
        cb = fig.colorbar(pcm, cax=fig.add_axes([fx(fig_w - margin - cb_in + 0.25), bottom, fx(0.14), height]))
        cb.set_label("Power above each panel's median [dB]")
    fig.suptitle(title, x=fx(margin + table_in + lead_in + (n * panel_in + (n - 1) * gap_in) / 2), y=0.97)
    add_info_panel(fig, info_rows, notes, footer, rect=[fx(margin), INFO_RECT[1], fx(table_in), INFO_RECT[3]])
    return save(fig, output) if output else plt.show()


def plot_doppler_time(time_s, doppler_hz, power_db, title, output=None, vmin=None, vmax=None,
                      info_rows=None, notes=None, footer=None):
    """Time (x) vs Doppler (y); ``power_db`` is (time, doppler), drawn in dB above its median."""
    fig, ax = new_figure()
    power_db = above_median_db(power_db)
    vmin, vmax = median_limits([power_db], vmin, vmax)
    pcm = ax.pcolormesh(time_s, doppler_hz, power_db.T, shading="nearest", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set(xlabel="Time [s]", ylabel="Doppler Frequency [Hz]")
    colorbar(fig, pcm, ax, MEDIAN_LABEL)
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
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#050505")
    pcm = ax.pcolormesh(time_s, fd_centers, np.ma.masked_invalid(power_db.T), shading="nearest", cmap=cmap,
                        vmin=vmin, vmax=vmax)
    ax.set(xlabel="Time [s]", ylabel="Phase-derived Doppler [Hz]")
    colorbar(fig, pcm, ax, "Power above median of all estimates [dB]")
    return _finish(fig, ax, title, output, info_rows, notes, footer)


def plot_phase_doppler_frame(fd_hz, delay_ms, power_db, fd_lim, title, output=None, vmin=None, vmax=None,
                             info_rows=None, notes=None, footer=None):
    """One estimate: scatter of (Doppler, delay) coloured by power; NaN entries are skipped."""
    fig, ax = new_figure()
    ax.set_facecolor("#050505")
    ok = np.isfinite(fd_hz) & np.isfinite(power_db)
    if ok.any():
        sc = ax.scatter(fd_hz[ok], delay_ms[ok], c=power_db[ok], s=18, cmap="viridis", vmin=vmin, vmax=vmax, linewidths=0)
        colorbar(fig, sc, ax, "Power above median of all estimates [dB]")
    else:
        ax.text(0.5, 0.5, "No valid phase-Doppler estimates", transform=ax.transAxes, ha="center", color="white")
    ax.set(xlim=fd_lim, ylim=(float(np.min(delay_ms)), float(np.max(delay_ms))),
           xlabel="Phase-derived Doppler [Hz]", ylabel="Delay [ms]")
    return _finish(fig, ax, title, output, info_rows, notes, footer)


def wrap_side_lines(lines, width=SIDE_PANEL_CHARS):
    """Wrap panel lines to ``width`` characters; a continuation is indented 2 past its line's own indent."""
    out = []
    for line in lines:
        if len(line) <= width:
            out.append(line)
            continue
        indent = " " * (len(line) - len(line.lstrip()) + 2)
        out += textwrap.wrap(line, width, subsequent_indent=indent) or [""]
    return out


def plot_angle_maps(maps, az_deg, el_deg, labels, title, output, range_db=30.0,
                    info_rows=None, notes=None, footer=None, markers=None, annotations=None, side_lines=None):
    """Polar sky maps (north up, east right, zenith at centre), each normalised to its own peak.

    ``maps`` is a list of (el, az) linear power arrays. ``markers`` optionally gives
    per map one (az, el) refined peak, or a list of them (strongest first), drawn as
    numbered crosses. ``annotations`` optionally gives per map ``None``, a
    ``dict(label=str, points=[(az, el), ...])``, or a list of those (one per peak):
    the label is written at its cross and the points (e.g. station bearings) are
    drawn as small triangles. ``side_lines`` adds a text panel on the right, always
    SIDE_PANEL_CHARS wide (longer lines are wrapped), so frames of a video keep one layout
    whatever the text; pass an empty list to keep the panel's room with nothing in it.
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
    # Laid out in inches, left to right: parameter table, maps, colour bar, station panel; the figure
    # is as wide as that needs, so nothing floats in empty space whatever the number of maps.
    margin, table_in, map_in, map_gap = 0.2, 3.9, 4.0 if n == 1 else 3.6, 0.95
    lead_in = 0.75                                        # left of a map: its 270 deg label
    cb_gap, cb_w, cb_label = 0.5, 0.14, 0.75              # right of a map: its 90 deg label; colour bar
    if side_lines is not None:
        side_lines = wrap_side_lines(side_lines)
    side_in = (0.3 + 0.058 * SIDE_PANEL_CHARS) if side_lines is not None else 0.0   # 6.6 pt monospace
    label_in = 0.85 if labels and any(labels) else 0.45   # map labels need room above the 0 deg mark
    under_in, top_in, bottom_in = 0.35, 0.6, 0.35
    row_in = map_in + label_in + under_in
    fig_h = max(5.2, bottom_in + rows_n * row_in + top_in)
    maps_x = margin + table_in + lead_in
    cb_x = maps_x + cols * map_in + (cols - 1) * map_gap + cb_gap
    fig_w = cb_x + cb_w + cb_label + side_in + margin
    fig = plt.figure(figsize=(fig_w, fig_h))
    fx = lambda inches: inches / fig_w                    # noqa: E731
    left, w = fx(maps_x), fx(map_in)
    width = fx(cols * map_in + (cols - 1) * map_gap)
    bottom = (bottom_in + under_in) / fig_h
    cell, h = row_in / fig_h, map_in / fig_h
    height = (rows_n - 1) * cell + h                       # bottom of the lowest map to top of the highest
    pcm = None
    for i, m in enumerate(maps):
        r, c = divmod(i, cols)
        ax = fig.add_axes([left + c * fx(map_in + map_gap), bottom + (rows_n - 1 - r) * cell, w, h], projection="polar")
        peak = np.nanmax(m)
        db = 10 * np.log10(np.maximum(m / peak, 1e-30)) if np.isfinite(peak) and peak > 0 else np.full(m.shape, np.nan)
        pcm = ax.pcolormesh(theta, radius, db[::-1], shading="auto", cmap="viridis", vmin=-abs(range_db), vmax=0)
        peaks = markers[i] if markers is not None and i < len(markers) and markers[i] is not None else []
        if peaks and np.isscalar(peaks[0]):
            peaks = [peaks]
        notes_i = annotations[i] if annotations is not None and i < len(annotations) else None
        if isinstance(notes_i, dict) or notes_i is None:
            notes_i = [notes_i]
        for j, (paz, pel) in enumerate(peaks):
            ax.plot(np.deg2rad(paz), 90 - pel, "c+", ms=10, mew=1.5)
            if len(peaks) > 1:
                ax.annotate(str(j + 1), (np.deg2rad(paz), 90 - pel), xytext=(-9, -11), textcoords="offset points",
                            fontsize=7.5, color="cyan", fontweight="bold")
            note = notes_i[j] if j < len(notes_i) else None
            if note:
                for qaz, qel in note.get("points", []):
                    ax.plot(np.deg2rad(qaz), 90 - qel, marker="^", ms=4, color="white", mec="black", mew=0.4,
                            ls="none")
                if note.get("label"):
                    ax.annotate("\n".join(textwrap.wrap(note["label"], 22)), (np.deg2rad(paz), 90 - pel),
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
        if labels and labels[i]:
            ax.set_title(labels[i], fontsize=9, pad=14)          # clear of the 0 deg label
    if pcm is not None:
        cb = fig.colorbar(pcm, cax=fig.add_axes([fx(cb_x), bottom, fx(cb_w), height]))
        cb.set_label("Power relative to map peak [dB]")
    top = bottom + height + (label_in - 0.15) / fig_h      # top of the highest map label
    if side_lines:
        fig.text(fx(cb_x + cb_w + cb_label + 0.3), top, "\n".join(side_lines), ha="left", va="top",
                 fontsize=6.6, family="monospace")
    fig.suptitle(title, x=left + width / 2, y=1 - 0.2 / fig_h, va="top")
    add_info_panel(fig, info_rows, notes, footer,
                   rect=[fx(margin), bottom_in / fig_h, fx(table_in), 1 - (bottom_in + top_in) / fig_h])
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

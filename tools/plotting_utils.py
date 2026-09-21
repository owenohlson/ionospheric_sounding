# plotting_utils.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
import os
import textwrap

from lfm_utils import LFMWaveform, dechirp_delay_axis_ms, window_from_arg


C_M_PER_S = 299_792_458.0

CODAR_SWEEP_FREQUENCIES_HZ = np.array([
    0.26000000,
    0.43333000,
    1.0,
    2.0,
    2.00773500,
    3.0,
    4.0,
], dtype=float)

CODAR_BANDWIDTHS_HZ = np.array([
    -22.05764000,
    -24.98300000,
    -25.73391300,
    -47.79155300,
    -49.62968800,
    -51.46782700,
    -75.07324200,
    -99.25937700,
    -100.17845200,
    -149.80813600,
    -150.00199900,
    -150.72720300,
    -200.35690300,
    -219.65733300,
    -299.61627200,
    -600.00085400,
    -601.07067900,
    24.98300000,
    25.73391300,
    89.49000000,
    99.25937700,
    99.93100000,
    100.00000000,
    101.09751900,
    220.57639900,
    600.15161100,
], dtype=float) * 1e3

INFO_PANEL_RECT = [0.34, 0.12, 0.58, 0.80]
INFO_PANEL_AX_RECT = [0.015, 0.12, 0.216, 0.80]
GENERATED_LFM_COLOR = "#18d7ff"


def _add_corner_note(corner_note: str = None, fig=None):
    if not corner_note:
        return
    if fig is None:
        fig = plt.gcf()
    wrapped_note = "\n".join(
        wrapped_line
        for line in str(corner_note).splitlines()
        for wrapped_line in (textwrap.wrap(line, width=52) or [""])
    )
    fig.text(
        0.99,
        0.01,
        wrapped_note,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 3},
    )


def _add_lower_left_note(note: str = None, fig=None):
    if not note:
        return
    if fig is None:
        fig = plt.gcf()
    fig.text(
        0.01,
        0.01,
        note,
        ha="left",
        va="bottom",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 3},
    )


def _normalize_info_rows(info_rows=None):
    rows = []
    if info_rows:
        rows.extend(info_rows)
    return rows


def _append_note_rows(rows, header, note):
    if not note:
        return rows
    rows = list(rows)
    rows.append((header, ""))
    for line in str(note).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            rows.append((key.strip(), value.strip()))
        elif line.strip():
            rows.append(("", line.strip()))
    return rows


def _add_info_panel(
    fig,
    info_rows=None,
    corner_note=None,
    lower_note=None,
    generated_lfm=False,
    footer_note=None,
):
    rows = _normalize_info_rows(info_rows)
    rows = _append_note_rows(rows, "Metadata", corner_note)
    if lower_note or generated_lfm:
        rows.append(("Analysis", ""))
        if generated_lfm:
            rows.append(("Generated LFM", "- - - -"))
        if lower_note:
            for line in str(lower_note).splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    rows.append((key.strip(), value.strip()))
                elif line.strip():
                    rows.append(("", line.strip()))
    if not rows:
        if footer_note:
            wrapped_footer = "\n".join(textwrap.wrap(str(footer_note), width=48) or [""])
            fig.text(
                INFO_PANEL_AX_RECT[0],
                INFO_PANEL_AX_RECT[1] + INFO_PANEL_AX_RECT[3],
                wrapped_footer,
                ha="left",
                va="top",
                fontsize=7.5,
                bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 3},
            )
        return

    ax_info = fig.add_axes(INFO_PANEL_AX_RECT)
    ax_info.axis("off")
    table_data = []
    for key, value in rows:
        if value == "":
            table_data.append([str(key), ""])
        elif str(key) == "Generated LFM":
            table_data.append([str(key), str(value)])
        else:
            table_data.append([
                "\n".join(textwrap.wrap(str(key), width=13) or [""]),
                "\n".join(textwrap.wrap(str(value), width=32) or [""]),
            ])
    row_count = len(table_data) + 1
    if row_count > 24:
        font_size = 6.4
    elif row_count > 18:
        font_size = 6.9
    else:
        font_size = 7.5
    row_height = min(0.045, 0.9 / max(row_count, 1))

    table = ax_info.table(
        cellText=table_data,
        colLabels=["Parameter", "Value"],
        cellLoc="left",
        colLoc="left",
        loc="upper left",
        colWidths=[0.34, 0.66],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    for (row, col), cell in table.get_celld().items():
        cell.set_height(row_height)
        cell.set_linewidth(0.35)
        cell.set_edgecolor("#c9c9c9")
        if row == 0:
            cell.set_facecolor("#eeeeee")
            cell.set_text_props(weight="bold")
        elif col == 0 and cell.get_text().get_text() and not table_data[row - 1][1]:
            cell.set_facecolor("#f6f6f6")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("white")
        if row > 0 and table_data[row - 1][0] == "Generated LFM" and col == 1:
            cell.get_text().set_text("- - - -")
            cell.get_text().set_color(GENERATED_LFM_COLOR)
            cell.get_text().set_fontweight("bold")
    if footer_note:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        table_bbox = table.get_window_extent(renderer).transformed(fig.transFigure.inverted())
        footer_y = max(0.015, table_bbox.y0 - 0.012)
        wrapped_footer = "\n".join(textwrap.wrap(str(footer_note), width=48) or [""])
        fig.text(
            INFO_PANEL_AX_RECT[0],
            footer_y,
            wrapped_footer,
            ha="left",
            va="top",
            fontsize=font_size,
            bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 3},
        )


def _new_figure_with_info(figsize=(14, 6)):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(INFO_PANEL_RECT)
    return fig, ax


def _tight_layout_with_info(fig, has_info=False):
    if has_info:
        return
    fig.tight_layout()


def _lfm_info_rows(lfm_config=None):
    if lfm_config is None:
        return []
    rows = [
        ("Sample rate", f"{lfm_config.sample_rate / 1e3:.3f} kHz"),
        ("Sweep freq", f"{lfm_config.sweep_frequency:.8g} Hz"),
        ("Bandwidth", f"{lfm_config.bandwidth / 1e3:.6g} kHz"),
    ]
    if getattr(lfm_config, "reference_gate_frequency", None) is not None:
        rows.extend([
            ("Gate freq", f"{lfm_config.reference_gate_frequency:.8g} Hz"),
            ("Gate duty", f"{lfm_config.reference_gate_duty * 100:.1f}%"),
            ("Gate phase", f"{lfm_config.reference_gate_phase * 1e3:.3f} ms"),
        ])
    return rows


def _add_delay_range_axis(ax, label="Virtual range [km]", two_way=True):
    scale = C_M_PER_S * 1e-3 / 1000.0
    if two_way:
        scale /= 2.0
    secax = ax.secondary_yaxis(
        "right",
        functions=(lambda d_ms: d_ms * scale, lambda km: km / scale),
    )
    secax.set_ylabel(label)
    return secax


def _add_delay_range_xaxis(ax, label="Range [km]", two_way=True):
    scale = C_M_PER_S * 1e-3 / 1000.0
    if two_way:
        scale /= 2.0
    secax = ax.secondary_xaxis(
        "top",
        functions=(lambda d_ms: d_ms * scale, lambda km: km / scale),
    )
    secax.set_xlabel(label)
    return secax


def _add_colorbar(fig, mappable, ax, label, pad=0.16):
    cbar = fig.colorbar(mappable, ax=ax, pad=pad)
    cbar.set_label(label)
    return cbar


def _savefig(output_file: str, dpi=300):
    full_path = os.path.expanduser(output_file)
    output_dir = os.path.dirname(full_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(full_path, dpi=dpi)
    return full_path


def _strongest_delay_bin_ms(power, delay_ms):
    if power.size == 0 or delay_ms.size == 0:
        return None
    if not np.any(np.isfinite(power)):
        return None
    _, delay_idx = np.unravel_index(np.nanargmax(power), power.shape)
    return float(delay_ms[delay_idx])


def _strongest_delay_bins_ms(power, delay_ms, n=5, min_separation_ms=None):
    if power.size == 0 or delay_ms.size == 0:
        return []
    if power.shape[-1] != delay_ms.size:
        return []

    if power.ndim == 1:
        delay_power = power.astype(float, copy=False)
    else:
        delay_power = np.nanmax(power, axis=tuple(range(power.ndim - 1)))

    delay_power = np.asarray(delay_power, dtype=float)
    if not np.any(np.isfinite(delay_power)):
        return []

    if min_separation_ms is None:
        if delay_ms.size > 1:
            delay_step = float(np.nanmedian(np.abs(np.diff(delay_ms))))
        else:
            delay_step = 0.0
        min_separation_ms = max(0.5, 3.0 * delay_step)

    work = delay_power.copy()
    peaks = []
    for _ in range(n):
        if not np.any(np.isfinite(work)):
            break
        idx = int(np.nanargmax(work))
        if not np.isfinite(work[idx]):
            break
        peaks.append((float(delay_ms[idx]), float(delay_power[idx])))
        mask = np.abs(delay_ms - delay_ms[idx]) <= min_separation_ms
        work[mask] = np.nan
    return peaks


def _format_delay_peaks_note(power, delay_ms, n=5):
    peaks = _strongest_delay_bins_ms(power, delay_ms, n=n)
    if not peaks:
        return None
    lines = []
    for idx, (delay, _) in enumerate(peaks, start=1):
        rng_km = delay * C_M_PER_S * 1e-6 / 2.0
        lines.append(f"Delay peak {idx}: {delay:.3f} ms ({rng_km:.2f} km)")
    return "\n".join(lines)


def _estimate_lfm_from_stft(freq_hz, time_s, zxx_db, utc_fractional_second=None, estimate_gate=True):
    if zxx_db.size == 0 or time_s.size < 4 or freq_hz.size < 4:
        return None

    ridge_idx = np.nanargmax(zxx_db, axis=0)
    ridge_freq = freq_hz[ridge_idx]
    ridge_power = zxx_db[ridge_idx, np.arange(zxx_db.shape[1])]

    good = ridge_power >= np.nanpercentile(ridge_power, 55)
    if np.count_nonzero(good) < 4:
        good = np.ones_like(ridge_power, dtype=bool)

    f_good = ridge_freq[good]
    span = float(np.nanpercentile(f_good, 95) - np.nanpercentile(f_good, 5))
    if span <= 0:
        return None

    df = np.diff(ridge_freq)
    jump_threshold = 0.45 * span
    jump_idx = np.flatnonzero(np.abs(df) >= jump_threshold)
    jump_idx = jump_idx[np.argsort(np.abs(df[jump_idx]))[::-1]]

    resets = []
    min_reset_gap = max(1, int(round(0.04 * len(time_s))))
    for idx in jump_idx:
        if all(abs(int(idx) - existing) >= min_reset_gap for existing in resets):
            resets.append(int(idx))
    resets = np.array(sorted(resets), dtype=int)

    sweep_period = None
    sweep_frequency = None
    first_sweep_start = None
    direction = "unknown"
    autocorr_period = _estimate_period_from_autocorr(time_s, ridge_freq)
    if resets.size >= 2:
        reset_times = time_s[resets + 1]
        periods = np.diff(reset_times)
        periods = periods[periods > 0]
        if periods.size:
            sweep_period = float(np.median(periods))
            reset_jump_sign = float(np.median(df[resets]))
            direction = "down" if reset_jump_sign > 0 else "up"
            if autocorr_period is not None and sweep_period / autocorr_period > 1.35:
                ratio = max(1, int(round(sweep_period / autocorr_period)))
                sweep_period = sweep_period / ratio
            first_sweep_start = _circular_mean_phase(reset_times, sweep_period)
    elif autocorr_period is not None:
        sweep_period = autocorr_period

    if direction == "unknown" and sweep_period is not None:
        phase = np.mod(time_s, sweep_period) / sweep_period
        middle = (phase > 0.15) & (phase < 0.85)
        if np.count_nonzero(middle) >= 4:
            slope = np.polyfit(time_s[middle], ridge_freq[middle], 1)[0]
            direction = "up" if slope >= 0 else "down"

    raw_bandwidth = -span if direction == "down" else span
    raw_sweep_frequency = None if sweep_period is None else 1.0 / sweep_period
    candidate_fit = _fit_codar_lfm_candidates(
        freq_hz=freq_hz,
        time_s=time_s,
        zxx_db=zxx_db,
        ridge_freq_hz=ridge_freq,
        raw_sweep_frequency=raw_sweep_frequency,
        direction=direction,
        initial_offset=first_sweep_start,
    )
    if candidate_fit:
        bandwidth = candidate_fit["bandwidth_hz"]
        sweep_frequency = candidate_fit["sweep_frequency_hz"]
        sweep_period = 1.0 / sweep_frequency
        first_sweep_start = candidate_fit["first_sweep_start_s"]
    else:
        bandwidth = _snap_codar_bandwidth(raw_bandwidth)
        if raw_sweep_frequency is not None:
            sweep_frequency = float(CODAR_SWEEP_FREQUENCIES_HZ[
                np.argmin(np.abs(CODAR_SWEEP_FREQUENCIES_HZ - raw_sweep_frequency))
            ])
            sweep_period = 1.0 / sweep_frequency
    if bandwidth < 0:
        f_start = float(abs(bandwidth) / 2.0)
        f_end = float(-abs(bandwidth) / 2.0)
    else:
        f_start = float(-abs(bandwidth) / 2.0)
        f_end = float(abs(bandwidth) / 2.0)

    utc_offset = None
    if first_sweep_start is not None and utc_fractional_second is not None and sweep_period is not None:
        utc_offset = float((utc_fractional_second + first_sweep_start) % sweep_period)

    gate = None
    if estimate_gate and sweep_period is not None:
        gate = _estimate_gate_from_ridge(time_s, ridge_power, sweep_period)

    return {
        "bandwidth_hz": bandwidth,
        "raw_bandwidth_hz": raw_bandwidth,
        "sweep_frequency_hz": sweep_frequency,
        "raw_sweep_frequency_hz": raw_sweep_frequency,
        "sweep_period_s": sweep_period,
        "first_sweep_start_s": first_sweep_start,
        "utc_offset_s": utc_offset,
        "fit_score": None if not candidate_fit else candidate_fit["score"],
        "f_start_hz": f_start,
        "f_end_hz": f_end,
        "ridge_time_s": time_s,
        "ridge_freq_hz": ridge_freq,
        "utc_fractional_second": utc_fractional_second,
        "gate": gate,
    }


def _snap_codar_bandwidth(raw_bandwidth_hz):
    if raw_bandwidth_hz is None or not np.isfinite(raw_bandwidth_hz):
        return None
    sign = -1 if raw_bandwidth_hz < 0 else 1
    candidates = CODAR_BANDWIDTHS_HZ[np.sign(CODAR_BANDWIDTHS_HZ) == sign]
    if candidates.size == 0:
        candidates = CODAR_BANDWIDTHS_HZ

    raw_abs = 2.0 * abs(raw_bandwidth_hz)
    cand_abs = np.abs(candidates)
    wider = cand_abs >= raw_abs
    if np.any(wider):
        return float(candidates[wider][np.argmin(cand_abs[wider] - raw_abs)])
    return float(candidates[np.argmin(np.abs(cand_abs - raw_abs))])


def _fit_codar_lfm_candidates(freq_hz, time_s, zxx_db, ridge_freq_hz,
                              raw_sweep_frequency=None, direction="unknown", initial_offset=None):
    if time_s.size < 4 or zxx_db.size == 0:
        return None
    freq_bin_hz = float(np.median(np.diff(freq_hz)))
    if not np.isfinite(freq_bin_hz) or freq_bin_hz == 0:
        return None
    freq_bin_hz = abs(freq_bin_hz)

    power_floor = float(np.nanpercentile(zxx_db, 25))
    power_span = max(float(np.nanpercentile(zxx_db, 99) - power_floor), 1.0)
    z_norm = (zxx_db - power_floor) / power_span

    if raw_sweep_frequency is None:
        sweep_candidates = CODAR_SWEEP_FREQUENCIES_HZ
    else:
        # Keep enough candidates to recover when the raw period lands on a harmonic.
        order = np.argsort(np.abs(CODAR_SWEEP_FREQUENCIES_HZ - raw_sweep_frequency))
        sweep_candidates = CODAR_SWEEP_FREQUENCIES_HZ[order[:4]]

    if direction == "down":
        bandwidth_candidates = CODAR_BANDWIDTHS_HZ[CODAR_BANDWIDTHS_HZ < 0]
    elif direction == "up":
        bandwidth_candidates = CODAR_BANDWIDTHS_HZ[CODAR_BANDWIDTHS_HZ > 0]
    else:
        bandwidth_candidates = CODAR_BANDWIDTHS_HZ

    if time_s.size > 1800:
        stride = int(np.ceil(time_s.size / 1800))
        eval_indices = np.arange(0, time_s.size, stride, dtype=int)
    else:
        eval_indices = np.arange(time_s.size, dtype=int)
    t_eval = time_s[eval_indices]
    ridge_eval = ridge_freq_hz[eval_indices]

    best = None
    for sweep_frequency in sweep_candidates:
        period = 1.0 / float(sweep_frequency)
        if initial_offset is None:
            offset_grid = np.linspace(0.0, period, 48, endpoint=False)
        else:
            half_window = min(period / 2.0, 0.08)
            offset_grid = (initial_offset + np.linspace(-half_window, half_window, 41)) % period
        for bandwidth in bandwidth_candidates:
            half_bw = abs(float(bandwidth)) / 2.0
            if half_bw > max(abs(freq_hz[0]), abs(freq_hz[-1])) * 1.20:
                continue
            for offset in offset_grid:
                phase = np.mod(t_eval - offset, period) / period
                pred_freq = -float(bandwidth) / 2.0 + float(bandwidth) * phase
                inband = (pred_freq >= freq_hz[0]) & (pred_freq <= freq_hz[-1])
                if np.count_nonzero(inband) < max(4, int(0.35 * pred_freq.size)):
                    continue

                pred_idx = np.clip(np.rint((pred_freq[inband] - freq_hz[0]) / freq_bin_hz).astype(int), 0, len(freq_hz) - 1)
                ridge_error_bins = np.abs(pred_freq[inband] - ridge_eval[inband]) / max(freq_bin_hz, 1.0)
                ridge_score = np.exp(-0.5 * (ridge_error_bins / 2.5) ** 2)
                power_score = z_norm[pred_idx, eval_indices[inband]]
                score = float(np.nanmean(power_score + 0.6 * ridge_score))

                if best is None or score > best["score"]:
                    best = {
                        "score": score,
                        "sweep_frequency_hz": float(sweep_frequency),
                        "bandwidth_hz": float(bandwidth),
                        "first_sweep_start_s": float(offset),
                    }
    return best


def _apply_lfm_overrides(estimate, sweep_frequency=None, bandwidth=None, offset=None,
                         gate_frequency=None, gate_duty=None, gate_phase=None):
    if estimate is None:
        estimate = {}
    estimate = dict(estimate)
    if sweep_frequency is not None:
        estimate["sweep_frequency_hz"] = float(sweep_frequency)
        estimate["sweep_period_s"] = 1.0 / float(sweep_frequency)
    if bandwidth is not None:
        estimate["bandwidth_hz"] = float(bandwidth)
        half = abs(float(bandwidth)) / 2.0
        if float(bandwidth) < 0:
            estimate["f_start_hz"] = half
            estimate["f_end_hz"] = -half
        else:
            estimate["f_start_hz"] = -half
            estimate["f_end_hz"] = half
    if offset is not None:
        estimate["first_sweep_start_s"] = float(offset)
    if gate_frequency is not None or gate_duty is not None or gate_phase is not None:
        gate = dict(estimate.get("gate") or {})
        if gate_frequency is not None:
            gate["frequency_hz"] = float(gate_frequency)
            gate["period_s"] = 1.0 / float(gate_frequency)
        if gate_duty is not None:
            gate["duty"] = float(gate_duty)
        if gate_phase is not None:
            gate["phase_s"] = float(gate_phase)
        estimate["gate"] = gate
    if (
        estimate.get("first_sweep_start_s") is not None
        and estimate.get("sweep_period_s") is not None
        and estimate.get("utc_fractional_second") is not None
    ):
        estimate["utc_offset_s"] = (
            estimate["utc_fractional_second"] + estimate["first_sweep_start_s"]
        ) % estimate["sweep_period_s"]
    if (
        sweep_frequency is not None
        or bandwidth is not None
        or offset is not None
        or gate_frequency is not None
        or gate_duty is not None
        or gate_phase is not None
    ):
        estimate["manual_overlay"] = True
    return estimate


def _estimate_period_from_autocorr(time_s, ridge_freq):
    if time_s.size > 6000:
        stride = int(np.ceil(time_s.size / 6000))
        time_s = time_s[::stride]
        ridge_freq = ridge_freq[::stride]
    dt = float(np.median(np.diff(time_s)))
    if not np.isfinite(dt) or dt <= 0 or time_s.size < 8:
        return None
    x = ridge_freq - np.nanmean(ridge_freq)
    x = np.nan_to_num(x, nan=0.0)
    if np.nanstd(x) <= 0:
        return None

    nfft = 1 << int(np.ceil(np.log2(2 * len(x) - 1)))
    X = np.fft.fft(x, n=nfft)
    corr = np.fft.ifft(X * np.conj(X)).real[:len(x)]
    if corr[0] <= 0:
        return None
    corr = corr / corr[0]

    min_lag = max(3, int(round(0.025 / dt)))
    max_lag = min(len(corr) - 2, int(round((time_s[-1] - time_s[0]) * 0.75 / dt)))
    if max_lag <= min_lag:
        return None
    candidates = []
    for lag in range(min_lag, max_lag + 1):
        if corr[lag] > 0.2 and corr[lag] >= corr[lag - 1] and corr[lag] >= corr[lag + 1]:
            candidates.append((corr[lag], lag))
    if not candidates:
        return None
    best_corr = max(c[0] for c in candidates)
    near_best = [lag for value, lag in candidates if value >= 0.75 * best_corr]
    return float(min(near_best) * dt)


def _circular_mean_phase(times, period):
    if period is None or period <= 0 or len(times) == 0:
        return None
    angles = 2.0 * np.pi * np.mod(times, period) / period
    mean_angle = np.angle(np.mean(np.exp(1j * angles)))
    if mean_angle < 0:
        mean_angle += 2.0 * np.pi
    return float(mean_angle * period / (2.0 * np.pi))


def _estimate_gate_from_ridge(time_s, ridge_power_db, sweep_period):
    if time_s.size > 5000:
        stride = int(np.ceil(time_s.size / 5000))
        time_s = time_s[::stride]
        ridge_power_db = ridge_power_db[::stride]
    dt = float(np.median(np.diff(time_s)))
    if not np.isfinite(dt) or dt <= 0 or sweep_period <= 0:
        return None
    x = ridge_power_db - np.nanmedian(ridge_power_db)
    x = np.nan_to_num(x, nan=0.0)
    if np.nanstd(x) < 1.5:
        return None

    corr = np.correlate(x, x, mode="full")[len(x) - 1:]
    if corr[0] <= 0:
        return None
    corr = corr / corr[0]

    min_lag = max(1, int(round(2 * dt / dt)))
    max_lag = min(len(corr) - 1, int(round(0.5 * sweep_period / dt)))
    if max_lag <= min_lag:
        return None

    lag = int(min_lag + np.argmax(corr[min_lag:max_lag + 1]))
    if corr[lag] < 0.25:
        return None

    gate_period = lag * dt
    phase = np.mod(time_s, gate_period)
    nbins = max(8, min(64, int(round(gate_period / dt))))
    edges = np.linspace(0, gate_period, nbins + 1)
    folded = np.full(nbins, np.nan)
    for i in range(nbins):
        mask = (phase >= edges[i]) & (phase < edges[i + 1])
        if np.any(mask):
            folded[i] = np.nanmedian(ridge_power_db[mask])
    if np.count_nonzero(np.isfinite(folded)) < 4:
        return None

    lo = np.nanpercentile(folded, 20)
    hi = np.nanpercentile(folded, 80)
    if hi - lo < 4.0:
        return None
    on = folded >= (lo + 0.45 * (hi - lo))
    duty = float(np.count_nonzero(on) / on.size)
    phase_center = float(np.nanmean((edges[:-1] + edges[1:]) * 0.5))
    if np.any(on):
        phase_center = float(np.mean((edges[:-1][on] + edges[1:][on]) * 0.5))

    return {
        "frequency_hz": 1.0 / gate_period,
        "period_s": gate_period,
        "duty": duty,
        "phase_s": phase_center % gate_period,
        "confidence": float(corr[lag]),
    }


def _format_lfm_estimate_note(estimate):
    if not estimate:
        return None
    lines = []
    suffix = "" if estimate.get("manual_overlay") else " (est.)"
    if estimate.get("bandwidth_hz") is not None:
        lines.append(f"BW{suffix}: {estimate['bandwidth_hz'] / 1e3:.2f} kHz")
    if estimate.get("sweep_frequency_hz") is not None:
        lines.append(f"Sweep frequency{suffix}: {estimate['sweep_frequency_hz']:.4g} Hz")
    if estimate.get("first_sweep_start_s") is not None:
        lines.append(f"Local offset: {estimate['first_sweep_start_s'] * 1e3:.2f} ms")
    if estimate.get("utc_offset_s") is not None:
        lines.append(f"UTC offset: {estimate['utc_offset_s'] * 1e3:.2f} ms")
    gate = estimate.get("gate")
    if gate:
        lines.append(
            f"Gate: {gate['frequency_hz']:.4g} Hz, "
            f"duty {gate['duty'] * 100:.0f}%, phase {gate['phase_s'] * 1e3:.2f} ms"
        )
    else:
        lines.append("Gate: not detected")
    return "\n".join(lines)


def _overlay_lfm_estimate(ax, estimate, t_min, t_max):
    if (
        not estimate
        or estimate.get("sweep_period_s") is None
        or estimate.get("f_start_hz") is None
        or estimate.get("f_end_hz") is None
    ):
        return
    period = estimate["sweep_period_s"]
    offset = estimate.get("first_sweep_start_s") or 0.0
    f_start = estimate["f_start_hz"] / 1e3
    f_end = estimate["f_end_hz"] / 1e3

    first = offset - np.ceil((offset - t_min) / period) * period
    starts = np.arange(first, t_max + period, period)
    for start in starts:
        end = start + period
        x0 = f_start
        x1 = f_end
        y0 = start
        y1 = end
        if y1 < t_min or y0 > t_max:
            continue
        if y0 < t_min:
            frac = (t_min - y0) / period
            x0 = f_start + frac * (f_end - f_start)
            y0 = t_min
        if y1 > t_max:
            frac = (t_max - start) / period
            x1 = f_start + frac * (f_end - f_start)
            y1 = t_max
        ax.plot(
            [x0, x1],
            [y0, y1],
            color=GENERATED_LFM_COLOR,
            linewidth=0.6,
            alpha=0.45,
            linestyle="--",
        )


def _auto_delay_bounds(delay_ms, power, d_min, d_max, half_width_ms=1.5, enabled=True):
    strongest_delay_ms = _strongest_delay_bin_ms(power, delay_ms)
    if enabled and d_min is None and d_max is None and strongest_delay_ms is not None:
        d_min = max(float(np.min(delay_ms)), strongest_delay_ms - half_width_ms)
        d_max = min(float(np.max(delay_ms)), strongest_delay_ms + half_width_ms)
        print(
            f"No --d-min/--d-max provided; auto-centering on strongest delay bin "
            f"{strongest_delay_ms:.3f} ms with range {d_min:.3f} to {d_max:.3f} ms",
            flush=True,
        )
    return d_min, d_max, strongest_delay_ms


def _timestamp_fractional_second(timestamp) -> float:
    if timestamp is None:
        return None

    if hasattr(timestamp, "utc_datetime"):
        dt = timestamp.utc_datetime
    else:
        dt = timestamp

    if hasattr(dt, "microsecond"):
        return dt.microsecond / 1e6

    return None


def _timestamp_sweep_offset_samples(timestamp, lfm_config, sweep_offset=0.0) -> int:
    frac = _timestamp_fractional_second(timestamp)
    if frac is None:
        return 0

    sweep_period = 1.0 / lfm_config.sweep_frequency
    offset_s = (float(sweep_offset) - frac) % sweep_period
    offset_samples = int(round(offset_s * lfm_config.sample_rate))
    if offset_samples >= lfm_config.sweep_length:
        offset_samples = 0
    return offset_samples


def plot_iq_spectrogram(
        iq: np.ndarray,
        fs: float,
        plot_title: str,
        vmin: float,
        vmax: float,
        tstart: float = 0.0,
        tend: float = None,
        window: str = "hann",
        window_size: int = 1024,
        hop_size: int = 512,
        output_file: str = None,
        corner_note: str = None,
        estimate_lfm: bool = False,
        estimate_lfm_gate: bool = True,
        overlay_estimated_lfm: bool = False,
        utc_fractional_second: float = None,
        lfm_overlay_sweep_frequency: float = None,
        lfm_overlay_bandwidth: float = None,
        lfm_overlay_offset: float = None,
        lfm_overlay_gate_frequency: float = None,
        lfm_overlay_gate_duty: float = None,
        lfm_overlay_gate_phase: float = None,
        info_rows=None,
        info_panel_footer_note: str = None,
        sample_time_origin: float = 0.0,
):
    """
    Plots a vertical STFT-based spectrogram of a complex IQ audio file.

    Parameters:
        filename (str): Path to the stereo IQ audio file (e.g., .wav or .flac).
        plot_title (str): Title for the spectrogram.
        vmin (float): Minimum dB level for color scale.
        vmax (float): Maximum dB level for color scale.
        tstart (float, optional): Start offset in seconds. Default: 0.0s.
        tend (float, optional): End time in seconds. Default: None (entire file).
        window_size (int, optional): STFT window size in samples. Default: 1024.
        hop_size (int, optional): STFT hop size in samples. Default: 512.
        output_file (str, optional): If set, saves the plot to this file path instead of showing.
    """

    # Handle duration and tstart
    start_sample = int(tstart * fs)
    end_sample = int(start_sample + (tend - tstart) * fs) if tend is not None else np.asarray(iq).shape[-1]
    iq = np.asarray(iq)
    single_stream = iq.ndim == 1
    if single_stream:
        iq = iq[None, :]
    elif iq.ndim != 2:
        raise ValueError("iq must be 1D or 2D (n_streams, n_samples)")
    sliced_iq = iq[:, start_sample:end_sample]
    tstart += sample_time_origin

    noverlap = window_size - hop_size

    # Compute STFT
    f = t = None
    power = None
    for stream in sliced_iq:
        f, t, Zxx = stft(stream, fs=fs, window=window, nperseg=window_size, noverlap=noverlap, return_onesided=False)
        stream_power = np.abs(Zxx) ** 2
        power = stream_power if power is None else power + stream_power
    f = np.fft.fftshift(f)
    power = np.fft.fftshift(power, axes=0)
    Zxx_dB = 10 * np.log10(power + 1e-12)
    lfm_estimate = None
    if estimate_lfm:
        lfm_estimate = _estimate_lfm_from_stft(
            f,
            t + tstart,
            Zxx_dB,
            utc_fractional_second=utc_fractional_second,
            estimate_gate=estimate_lfm_gate,
        )
    if overlay_estimated_lfm and (
        lfm_overlay_sweep_frequency is not None
        or lfm_overlay_bandwidth is not None
        or lfm_overlay_offset is not None
    ):
        lfm_estimate = _apply_lfm_overrides(
            lfm_estimate,
            sweep_frequency=lfm_overlay_sweep_frequency,
            bandwidth=lfm_overlay_bandwidth,
            offset=lfm_overlay_offset,
            gate_frequency=lfm_overlay_gate_frequency,
            gate_duty=lfm_overlay_gate_duty,
            gate_phase=lfm_overlay_gate_phase,
        )

    # Plot vertical spectrogram
    lower_note = _format_lfm_estimate_note(lfm_estimate) if lfm_estimate else None
    has_info = bool(info_rows or corner_note or lower_note or info_panel_footer_note)
    if has_info:
        fig, ax = _new_figure_with_info(figsize=(14, 6))
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
    pcm = ax.pcolormesh(f / 1e3, t + tstart, Zxx_dB.T, shading='gouraud',
                        cmap='inferno', vmin=vmin, vmax=vmax)

    # Labels and title
    ax.set_xlabel('Frequency [kHz]', fontsize=12)
    ax.set_ylabel('Time [s]', fontsize=12)
    ax.set_title(plot_title, fontsize=14, fontweight='bold')

    # Colorbar
    cbar = fig.colorbar(pcm, ax=ax, orientation='vertical', pad=0.02)
    cbar.set_label('Power [dB]', fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    # Ticks and layout
    ax.tick_params(axis='both', which='major', labelsize=10)
    if overlay_estimated_lfm:
        _overlay_lfm_estimate(ax, lfm_estimate, float(t[0] + tstart), float(t[-1] + tstart))
    _add_info_panel(
        fig,
        info_rows=info_rows,
        corner_note=corner_note,
        lower_note=lower_note,
        generated_lfm=overlay_estimated_lfm,
        footer_note=info_panel_footer_note,
    )
    _tight_layout_with_info(fig, has_info=has_info)

    if output_file:
        full_path = _savefig(output_file, dpi=300)
        plt.close()
        print(f"Saved spectrogram to '{full_path}'")
    else:
        plt.show(block=True)


def plot_dechirp_profile(delay_ms, spectra, lfm_config, title="Dechirped Power–Delay Profile",
                         output_file=None, time_units="ms", vmin=None, vmax=None,
                         corner_note=None, info_rows=None, info_panel_footer_note=None):
    """Sum polarization powers, then average valid sweep powers before taking dB."""
    spectra = np.asarray(spectra)
    if spectra.ndim == 2:
        spectra = spectra[None, ...]
    if spectra.ndim != 3 or spectra.shape[1] == 0:
        raise ValueError("Profile requires at least one complete sweep")
    power = np.sum(np.abs(spectra) ** 2, axis=0)
    valid = np.isfinite(power)
    counts = valid.sum(axis=0)
    mean_power = np.divide(np.where(valid, power, 0).sum(axis=0), counts,
                           out=np.full(power.shape[1], np.nan), where=counts > 0)
    if not np.any(counts):
        raise ValueError("No valid sweeps available for a dechirped profile")
    rows = list(info_rows or []) + _lfm_info_rows(lfm_config)
    fig, ax = _new_figure_with_info(figsize=(14, 4.8))
    ax.plot(np.asarray(delay_ms) / (1000 if time_units == "s" else 1),
            10 * np.log10(mean_power + 1e-12))
    ax.set(xlabel=f"Delay [{time_units}]", ylabel="Mean power [dB]", title=title)
    ax.set_ylim(bottom=vmin, top=vmax)
    ax.grid(True)
    _add_info_panel(fig, info_rows=rows, corner_note=corner_note,
                    footer_note=info_panel_footer_note)
    _tight_layout_with_info(fig, has_info=True)
    if output_file:
        _savefig(output_file, dpi=300)
    else:
        plt.show()
    plt.close(fig)
    return ax




def plot_dechirp(stretch_result: np.ndarray,
                 lfm_config: LFMWaveform,
                 vmin: float = None,
                 vmax: float = None,
                 title: str = "Stretch Processed Range-Time Plot",
                 save_path: str = None,
                 navg: int = 1,
                 d_min: float = None,
                 d_max: float = None,
                 tstart: float = 0.0,
                 corner_note: str = None,
                 auto_delay_window: bool = True,
                 info_rows=None,
                 info_panel_footer_note: str = None,
                 selected_delay_ms=None,
):
    T = 1 / lfm_config.sweep_frequency
    k = lfm_config.bandwidth / T
    if k == 0:
        raise ValueError("Cannot convert dechirp beat frequency to delay when bandwidth is zero")
    chirp_len = stretch_result.shape[1]
    slow_time_len = stretch_result.shape[0]
    
    slow_time = np.linspace(0, slow_time_len / lfm_config.sweep_frequency, slow_time_len)

    power_db = 10 * np.log10(stretch_result + 1e-12)
    if selected_delay_ms is None:
        time_delays, delay_order = dechirp_delay_axis_ms(lfm_config, n_bins=chirp_len)
        power_db = power_db[:, delay_order]
    else:
        time_delays = np.asarray(selected_delay_ms)
        if time_delays.shape != (chirp_len,):
            raise ValueError("Stored delay axis does not match the data")
    should_auto_delay_window = auto_delay_window and d_min is None and d_max is None
    d_min, d_max, strongest_delay_ms = _auto_delay_bounds(
        time_delays,
        power_db,
        d_min,
        d_max,
        enabled=auto_delay_window,
    )

    d_mask = np.ones_like(time_delays, dtype=bool)
    if d_min is not None:
        d_mask = d_mask & (time_delays >= d_min)
    if d_max is not None:
        d_mask = d_mask & (time_delays <= d_max)
    time_delays = time_delays[d_mask]
    power_db = power_db[:, d_mask]
    if not should_auto_delay_window:
        strongest_delay_ms = _strongest_delay_bin_ms(power_db, time_delays)

    if navg > 1:
        nrows = (power_db.shape[0] // navg) * navg
        power_db = power_db[:nrows].reshape(-1, navg, power_db.shape[1]).mean(axis=1)
        slow_time_len = power_db.shape[0]
        slow_time = tstart + np.arange(slow_time_len) * navg / lfm_config.sweep_frequency
    else:
        slow_time = tstart + np.arange(slow_time_len) / lfm_config.sweep_frequency

    rows = list(info_rows or []) + _lfm_info_rows(lfm_config)
    lower_note = _format_delay_peaks_note(power_db, time_delays, n=5)
    has_info = bool(rows or corner_note or lower_note or info_panel_footer_note)
    fig, ax = _new_figure_with_info(figsize=(14, 6)) if has_info else plt.subplots(figsize=(10, 6))
    pcm = ax.pcolormesh(slow_time, time_delays, power_db.T, shading='nearest', cmap='inferno', vmin=vmin, vmax=vmax)
    ax.set_ylabel('Delay [ms]')
    ax.set_xlabel('Time [s]')
    ax.set_title(title)
    _add_delay_range_axis(ax)
    _add_colorbar(fig, pcm, ax, 'Power [dB]')
    _add_info_panel(
        fig,
        info_rows=rows,
        corner_note=corner_note,
        lower_note=lower_note,
        footer_note=info_panel_footer_note,
    )
    _tight_layout_with_info(fig, has_info=has_info)

    if save_path:
        _savefig(save_path, dpi=300)
    plt.show()
    



def plot_delay_doppler_dechirp(
    dechirp_spectra: np.ndarray,   # (num_chirps, n_bins), complex, fftshifted
    lfm_config,
    title: str,
    output_file: str = None,
    vmin: float = None,
    vmax: float = None,
    window_slow: str = "hann",
    nfft_doppler: int = None,
    fd_max: float = None,
    fd_min: float = None,
    d_max: float = None,
    d_min: float = None,
    interactive: bool = False,
    positive_delay_axis: bool = True,
    corner_note: str = None,
    auto_delay_window: bool = True,
    info_rows=None,
    info_panel_footer_note: str = None,
    selected_delay_ms=None,
    stored_rdc2=None,
    stored_doppler_hz=None,
):
    PRF = lfm_config.sweep_frequency

    dechirp_spectra = np.asarray(dechirp_spectra)
    if dechirp_spectra.ndim == 2:
        spectra = dechirp_spectra[None, :, :]
    elif dechirp_spectra.ndim == 3:
        spectra = dechirp_spectra
    else:
        raise ValueError("dechirp_spectra must be 2D (num_chirps, n_bins) or 3D (n_streams, num_chirps, n_bins)")
    
    _, slow_len, n_bins = spectra.shape

    if stored_rdc2 is None:
        from rdc_utils import doppler_fft_rdc1
        cube, fd = doppler_fft_rdc1(spectra.transpose(2, 1, 0), PRF,
                                    window=window_slow, nfft=nfft_doppler)
        DD = cube.transpose(2, 1, 0)
    else:
        DD = np.asarray(stored_rdc2).transpose(2, 1, 0)
        fd = np.asarray(stored_doppler_hz)
        if fd.shape != (DD.shape[1],):
            raise ValueError("Stored Doppler axis does not match RDC2")

    power_db = 10.0 * np.log10(np.sum(np.abs(DD) ** 2, axis=0) + 1e-12)

    if selected_delay_ms is None:
        delay_ms, delay_order = dechirp_delay_axis_ms(
            lfm_config, n_bins=n_bins, positive_delay_axis=positive_delay_axis,
        )
        power_db = power_db[:, delay_order]
    else:
        delay_ms = np.asarray(selected_delay_ms)
        if delay_ms.shape != (power_db.shape[1],):
            raise ValueError("Stored delay axis does not match the data")
    should_auto_delay_window = auto_delay_window and d_min is None and d_max is None
    d_min, d_max, strongest_delay_ms = _auto_delay_bounds(
        delay_ms,
        power_db,
        d_min,
        d_max,
        enabled=auto_delay_window,
    )

    # Apply delay mask
    d_mask = np.ones_like(delay_ms, dtype=bool)

    if d_max is not None:
        d_mask = d_mask & (delay_ms <= d_max)

    if d_min is not None:
        d_mask = d_mask & (delay_ms >= d_min)

    d_plot = delay_ms[d_mask]
    power_db = power_db[:, d_mask]
    if d_plot.size == 0:
        raise ValueError(
            "No delay bins selected; check --d-min/--d-max. "
            f"Available delay range is {delay_ms.min():.3f} to {delay_ms.max():.3f} ms."
        )
    if not should_auto_delay_window:
        strongest_delay_ms = _strongest_delay_bin_ms(power_db, d_plot)

    # Optional Doppler zoom
    if fd_max is not None or fd_min is not None:
        fd_mask = np.ones_like(fd, dtype=bool)
        if fd_max is not None:
            fd_mask = fd_mask & (fd <= fd_max)
        if fd_min is not None:
            fd_mask = fd_mask & (fd >= fd_min)
        fd_plot = fd[fd_mask]
        power_db = power_db[fd_mask, :]
    else:
        fd_plot = fd

    rows = list(info_rows or []) + _lfm_info_rows(lfm_config)
    lower_note = _format_delay_peaks_note(power_db, d_plot, n=5)
    has_info = bool(rows or corner_note or lower_note or info_panel_footer_note)
    fig, ax = _new_figure_with_info(figsize=(14, 6)) if has_info else plt.subplots(figsize=(10, 6))
    pcm = ax.pcolormesh(fd_plot, d_plot, power_db.T, shading="nearest", cmap="inferno", vmin=vmin, vmax=vmax)
    ax.set_xlabel("Doppler Frequency [Hz]")
    ax.set_ylabel("Delay [ms]")
    ax.set_title(title)
    _add_delay_range_axis(ax)
    _add_colorbar(fig, pcm, ax, "Power [dB]")
    _add_info_panel(
        fig,
        info_rows=rows,
        corner_note=corner_note,
        lower_note=lower_note,
        footer_note=info_panel_footer_note,
    )
    _tight_layout_with_info(fig, has_info=has_info)
    if output_file:
        # Expand the ~ if it exists
        _savefig(output_file, dpi=300)
        if interactive == True:
            plt.show()
        plt.close()
    else:
        plt.show()


def plot_doppler_time_from_delay_band(
    dechirp_spectra: np.ndarray,
    lfm_config,
    title: str,
    output_file: str = None,
    vmin: float = None,
    vmax: float = None,
    d_min: float = None,
    d_max: float = None,
    fd_min: float = None,
    fd_max: float = None,
    integration_time: float = 60.0,
    hop_time: float = 10.0,
    slow_window: str = "hann",
    nfft_doppler: int = None,
    combine: str = "incoherent",
    tstart: float = 0.0,
    selected_delay_ms: np.ndarray = None,
    corner_note: str = None,
    info_rows=None,
    info_panel_footer_note: str = None,
):
    prf = lfm_config.sweep_frequency
    dechirp_spectra = np.asarray(dechirp_spectra)
    if dechirp_spectra.ndim == 2:
        spectra_in = dechirp_spectra[None, :, :]
    elif dechirp_spectra.ndim == 3:
        spectra_in = dechirp_spectra
    else:
        raise ValueError("dechirp_spectra must be 2D (num_chirps, n_bins) or 3D (n_streams, num_chirps, n_bins)")
    if d_min is None or d_max is None:
        raise ValueError("Both --d-min and --d-max are required for Doppler-vs-time plots")
    if integration_time <= 0 or hop_time <= 0:
        raise ValueError("--integration-time and --hop-time must be > 0")
    if combine not in ("incoherent", "coherent", "peak"):
        raise ValueError("--combine must be one of: incoherent, coherent, peak")

    _, num_chirps, n_bins = spectra_in.shape

    if selected_delay_ms is None:
        delay_ms, delay_order = dechirp_delay_axis_ms(lfm_config, n_bins=n_bins)
        spectra = spectra_in[:, :, delay_order]

        d_mask = (delay_ms >= d_min) & (delay_ms <= d_max)
        selected_delays = delay_ms[d_mask]
        if selected_delays.size == 0:
            raise ValueError(
                "No delay bins selected; check --d-min/--d-max. "
                f"Available delay range is {delay_ms.min():.3f} to {delay_ms.max():.3f} ms."
            )
        band = spectra[:, :, d_mask]
    else:
        selected_delays = np.asarray(selected_delay_ms)
        if selected_delays.ndim != 1 or selected_delays.size != n_bins:
            raise ValueError("selected_delay_ms must be 1D and match the dechirp_spectra delay-bin axis")
        band = spectra_in

    window_len = int(round(integration_time * prf))
    hop_len = int(round(hop_time * prf))
    if window_len < 1:
        raise ValueError("--integration-time is shorter than one sweep")
    if hop_len < 1:
        raise ValueError("--hop-time is shorter than one sweep")
    if num_chirps < window_len:
        raise ValueError(
            f"Not enough sweeps for integration_time={integration_time:g}s "
            f"({num_chirps} available, {window_len} needed)"
        )

    if nfft_doppler is None:
        nfft_doppler = 1 << int(np.ceil(np.log2(max(window_len, 1))))
    if nfft_doppler < window_len:
        raise ValueError(
            f"Doppler FFT length ({nfft_doppler}) must be >= the number of sweeps ({window_len}); "
            "a shorter FFT truncates the slow-time window. Increase --nfft-doppler or omit it."
        )

    w = window_from_arg(window_len, slow_window).astype(np.float32, copy=False)
    coherent_gain = np.mean(w)
    if coherent_gain == 0:
        raise ValueError(f"Slow-time window '{slow_window}' has zero coherent gain")

    frame_starts = np.arange(0, num_chirps - window_len + 1, hop_len, dtype=int)
    fd = np.fft.fftshift(np.fft.fftfreq(nfft_doppler, d=1.0 / prf))

    power = np.empty((len(frame_starts), len(fd)), dtype=np.float32)
    for frame_idx, start in enumerate(frame_starts):
        x = band[:, start:start + window_len, :]
        xw = x * (w[None, :, None] / coherent_gain)
        X = np.fft.fftshift(np.fft.fft(xw, n=nfft_doppler, axis=1), axes=1)
        bin_power = np.abs(X) ** 2

        if combine == "incoherent":
            power[frame_idx] = np.mean(np.sum(bin_power, axis=0), axis=1)
        elif combine == "coherent":
            coherent = np.sum(X, axis=2)
            power[frame_idx] = np.sum(np.abs(coherent) ** 2, axis=0) / X.shape[2]
        else:
            strongest_bin = int(np.argmax(np.mean(np.sum(np.abs(x) ** 2, axis=0), axis=0)))
            power[frame_idx] = np.sum(bin_power[:, :, strongest_bin], axis=0)

    power_db = 10.0 * np.log10(power + 1e-12)
    time_plot = tstart + (frame_starts + window_len / 2.0) / prf

    fd_mask = np.ones_like(fd, dtype=bool)
    if fd_min is not None:
        fd_mask &= fd >= fd_min
    if fd_max is not None:
        fd_mask &= fd <= fd_max
    fd_plot = fd[fd_mask]
    power_db = power_db[:, fd_mask]

    title_text = (
        f"{title}\n"
    )
    rows = list(info_rows or []) + _lfm_info_rows(lfm_config) + [
        ("Delay band", f"{selected_delays.min():.3f}-{selected_delays.max():.3f} ms"),
        ("Delay range", f"{selected_delays.min() * C_M_PER_S * 1e-6 / 2:.2f}-{selected_delays.max() * C_M_PER_S * 1e-6 / 2:.2f} km"),
        ("Integration", f"{integration_time:.3f} s"),
        ("Hop", f"{hop_time:.3f} s"),
        ("Combine", combine),
    ]
    has_info = bool(rows or corner_note or info_panel_footer_note)
    fig, ax = _new_figure_with_info(figsize=(14, 6)) if has_info else plt.subplots(figsize=(10, 6))
    pcm = ax.pcolormesh(time_plot, fd_plot, power_db.T, shading="nearest",
                        cmap="inferno", vmin=vmin, vmax=vmax)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Doppler Frequency [Hz]")
    ax.set_title(title_text)
    _add_colorbar(fig, pcm, ax, "Power [dB]")
    _add_info_panel(fig, info_rows=rows, corner_note=corner_note, footer_note=info_panel_footer_note)
    _tight_layout_with_info(fig, has_info=has_info)

    if output_file:
        _savefig(output_file, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_phase_doppler_time_from_delay_band(
    dechirp_spectra: np.ndarray,
    selected_delay_ms: np.ndarray,
    lfm_config,
    title: str,
    output_file: str = None,
    vmin: float = None,
    vmax: float = None,
    fd_min: float = None,
    fd_max: float = None,
    estimator_sweeps: int = 1,
    fd_bins: int = 128,
    bin_combine: str = "max",
    power_floor_db: float = None,
    tstart: float = 0.0,
    corner_note: str = None,
    info_rows=None,
    info_panel_footer_note: str = None,
):
    selected_delay_ms = np.asarray(selected_delay_ms)
    spectra = np.asarray(dechirp_spectra)
    if selected_delay_ms.ndim != 1 or selected_delay_ms.size != spectra.shape[-1]:
        raise ValueError("selected_delay_ms must be 1D and match the dechirp_spectra delay-bin axis")
    if fd_bins < 2:
        raise ValueError("--fd-bins must be >= 2")
    if bin_combine not in ("max", "max2", "max3", "max4", "mean", "sum"):
        raise ValueError("--bin-combine must be one of: max, max2, max3, max4, mean, sum")

    prf = lfm_config.sweep_frequency
    time_centers, fd_hz, power_db = _phase_doppler_from_dechirp_spectra(
        spectra,
        prf,
        estimator_sweeps=estimator_sweeps,
    )
    selected_peak_note = None
    effective_combine = bin_combine
    if bin_combine in ("max2", "max3", "max4"):
        n_peaks = int(bin_combine.removeprefix("max"))
        peaks = _strongest_delay_bins_ms(power_db, selected_delay_ms, n=n_peaks)
        if not peaks:
            raise ValueError(f"Could not select strongest delay bins for --bin-combine={bin_combine}")
        peak_delays = np.asarray([delay for delay, _ in peaks])
        peak_indices = np.asarray([
            int(np.nanargmin(np.abs(selected_delay_ms - peak_delay)))
            for peak_delay in peak_delays
        ])
        fd_hz = fd_hz[:, peak_indices]
        power_db = power_db[:, peak_indices]
        selected_peak_note = ", ".join(f"{selected_delay_ms[idx]:.3f}" for idx in peak_indices)
        effective_combine = "max"
    time_plot = tstart + time_centers
    fd_min, fd_max, unambiguous_min, unambiguous_max = _phase_doppler_axis_limits(
        prf,
        fd_min=fd_min,
        fd_max=fd_max,
    )

    fd_edges = np.linspace(fd_min, fd_max, int(fd_bins) + 1)
    fd_centers = 0.5 * (fd_edges[:-1] + fd_edges[1:])
    if effective_combine == "max":
        power_grid = np.full((time_plot.size, fd_centers.size), np.nan, dtype=np.float32)
    else:
        power_sum = np.zeros((time_plot.size, fd_centers.size), dtype=np.float64)
        counts = np.zeros((time_plot.size, fd_centers.size), dtype=np.int32)

    for time_idx in range(time_plot.size):
        valid = np.isfinite(fd_hz[time_idx]) & np.isfinite(power_db[time_idx])
        valid &= (fd_hz[time_idx] >= fd_min) & (fd_hz[time_idx] <= fd_max)
        if power_floor_db is not None:
            valid &= power_db[time_idx] >= power_floor_db
        if not np.any(valid):
            continue

        bin_idx = np.digitize(fd_hz[time_idx, valid], fd_edges) - 1
        bin_idx = np.clip(bin_idx, 0, fd_centers.size - 1)
        p_linear = 10.0 ** (power_db[time_idx, valid] / 10.0)

        if effective_combine == "max":
            for idx, p_db in zip(bin_idx, power_db[time_idx, valid]):
                if not np.isfinite(power_grid[time_idx, idx]) or p_db > power_grid[time_idx, idx]:
                    power_grid[time_idx, idx] = p_db
        else:
            for idx, p in zip(bin_idx, p_linear):
                power_sum[time_idx, idx] += p
                counts[time_idx, idx] += 1

    if effective_combine == "mean":
        power_linear = np.full_like(power_sum, np.nan, dtype=np.float64)
        np.divide(power_sum, counts, out=power_linear, where=counts > 0)
        power_grid = 10.0 * np.log10(power_linear + 1e-12)
    elif effective_combine == "sum":
        power_linear = np.where(counts > 0, power_sum, np.nan)
        power_grid = 10.0 * np.log10(power_linear + 1e-12)

    rows = list(info_rows or []) + _lfm_info_rows(lfm_config) + [
        ("Delay band", f"{selected_delay_ms.min():.3f}-{selected_delay_ms.max():.3f} ms"),
        ("Delay range", f"{selected_delay_ms.min() * C_M_PER_S * 1e-6 / 2:.2f}-{selected_delay_ms.max() * C_M_PER_S * 1e-6 / 2:.2f} km"),
        ("Estimator", "sweep-to-sweep phase"),
        ("Estimator sweeps", str(estimator_sweeps)),
        ("Update interval", f"{1.0 / prf:.3f} s"),
        ("Unambiguous fd", f"{unambiguous_min:.3f} to {unambiguous_max:.3f} Hz"),
        ("Displayed fd", f"{fd_min:.3f} to {fd_max:.3f} Hz"),
        ("Doppler bins", str(fd_centers.size)),
        ("Bin combine", bin_combine),
    ]
    if power_floor_db is not None:
        rows.append(("Power floor", f"{power_floor_db:.3f} dB"))
    if selected_peak_note is not None:
        rows.append(("Selected delays", f"{selected_peak_note} ms"))

    power_cmap = plt.get_cmap("inferno").copy()
    power_cmap.set_bad("#050505")
    has_info = bool(rows or corner_note or info_panel_footer_note)
    fig, ax = _new_figure_with_info(figsize=(14, 6)) if has_info else plt.subplots(figsize=(10, 6))
    ax.set_facecolor("#050505")
    pcm = ax.pcolormesh(
        time_plot,
        fd_centers,
        np.ma.masked_invalid(power_grid.T),
        shading="nearest",
        cmap=power_cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Phase-derived Doppler [Hz]")
    ax.set_title(title)
    _add_colorbar(fig, pcm, ax, "Power [dB]")
    _add_info_panel(fig, info_rows=rows, corner_note=corner_note, footer_note=info_panel_footer_note)
    _tight_layout_with_info(fig, has_info=has_info)

    if output_file:
        _savefig(output_file, dpi=300)
        plt.close()
    else:
        plt.show()


def _phase_doppler_from_dechirp_spectra(dechirp_spectra, prf, estimator_sweeps=1):
    dechirp_spectra = np.asarray(dechirp_spectra)
    if dechirp_spectra.ndim == 2:
        spectra = dechirp_spectra[None, :, :]
    elif dechirp_spectra.ndim == 3:
        spectra = dechirp_spectra
    else:
        raise ValueError("dechirp_spectra must be 2D (num_chirps, n_bins) or 3D (n_streams, num_chirps, n_bins)")
    if estimator_sweeps < 1:
        raise ValueError("--estimator-sweeps must be >= 1")
    if spectra.shape[1] < estimator_sweeps + 1:
        raise ValueError(
            f"Need at least estimator_sweeps+1 sweeps "
            f"({estimator_sweeps + 1}) for phase-Doppler estimation"
        )

    products = spectra[:, 1:, :] * np.conj(spectra[:, :-1, :])
    pair_power = 0.5 * (
        np.abs(spectra[:, 1:, :]) ** 2
        + np.abs(spectra[:, :-1, :]) ** 2
    )
    valid = np.isfinite(products.real) & np.isfinite(products.imag) & np.isfinite(pair_power)

    _, num_pairs, n_bins = products.shape
    num_estimates = num_pairs - estimator_sweeps + 1
    fd_hz = np.full((num_estimates, n_bins), np.nan, dtype=np.float32)
    power = np.full((num_estimates, n_bins), np.nan, dtype=np.float32)

    dt = 1.0 / prf
    for i in range(num_estimates):
        p = products[:, i:i + estimator_sweeps, :]
        pw = pair_power[:, i:i + estimator_sweeps, :]
        ok = valid[:, i:i + estimator_sweeps, :]
        counts = np.sum(ok, axis=(0, 1))
        summed = np.sum(np.where(ok, p, 0.0), axis=(0, 1))
        good = (counts > 0) & (np.abs(summed) > 0)
        fd_hz[i, good] = np.angle(summed[good]) / (2.0 * np.pi * dt)
        power[i, good] = np.sum(np.where(ok, pw, 0.0), axis=(0, 1))[good] / counts[good]

    time_centers = (np.arange(num_estimates, dtype=float) + estimator_sweeps / 2.0) / prf
    power_db = 10.0 * np.log10(power + 1e-12)
    return time_centers, fd_hz, power_db


def _phase_doppler_axis_limits(prf, fd_min=None, fd_max=None):
    unambiguous_min = -0.5 * prf
    unambiguous_max = 0.5 * prf
    requested_min = unambiguous_min if fd_min is None else float(fd_min)
    requested_max = unambiguous_max if fd_max is None else float(fd_max)
    if requested_min > requested_max:
        raise ValueError("--fd-min must be <= --fd-max")

    clipped_min = max(requested_min, unambiguous_min)
    clipped_max = min(requested_max, unambiguous_max)
    if clipped_min > clipped_max:
        raise ValueError(
            "Requested Doppler display limits do not overlap the unambiguous "
            f"phase-Doppler range {unambiguous_min:.3f} to {unambiguous_max:.3f} Hz"
        )
    if clipped_min != requested_min or clipped_max != requested_max:
        print(
            f"Clipping phase-Doppler display limits to unambiguous range: "
            f"{clipped_min:.3f} to {clipped_max:.3f} Hz",
            flush=True,
        )
    return clipped_min, clipped_max, unambiguous_min, unambiguous_max


def plot_phase_doppler_time_delay(
    dechirp_spectra: np.ndarray,
    selected_delay_ms: np.ndarray,
    lfm_config,
    title: str,
    output_file: str = None,
    fd_min: float = None,
    fd_max: float = None,
    estimator_sweeps: int = 1,
    power_floor_db: float = None,
    tstart: float = 0.0,
    corner_note: str = None,
    info_rows=None,
    info_panel_footer_note: str = None,
):
    selected_delay_ms = np.asarray(selected_delay_ms)
    spectra = np.asarray(dechirp_spectra)
    delay_axis_len = spectra.shape[-1]
    if selected_delay_ms.ndim != 1 or selected_delay_ms.size != delay_axis_len:
        raise ValueError("selected_delay_ms must be 1D and match the dechirp_spectra delay-bin axis")

    prf = lfm_config.sweep_frequency
    time_centers, fd_hz, power_db = _phase_doppler_from_dechirp_spectra(
        dechirp_spectra,
        prf,
        estimator_sweeps=estimator_sweeps,
    )
    time_plot = tstart + time_centers
    fd_plot = fd_hz.copy()
    if power_floor_db is not None:
        fd_plot[power_db < power_floor_db] = np.nan

    fd_min, fd_max, unambiguous_min, unambiguous_max = _phase_doppler_axis_limits(
        prf,
        fd_min=fd_min,
        fd_max=fd_max,
    )

    rows = list(info_rows or []) + _lfm_info_rows(lfm_config) + [
        ("Delay band", f"{selected_delay_ms.min():.3f}-{selected_delay_ms.max():.3f} ms"),
        ("Delay range", f"{selected_delay_ms.min() * C_M_PER_S * 1e-6 / 2:.2f}-{selected_delay_ms.max() * C_M_PER_S * 1e-6 / 2:.2f} km"),
        ("Estimator", "sweep-to-sweep phase"),
        ("Estimator sweeps", str(estimator_sweeps)),
        ("Update interval", f"{1.0 / prf:.3f} s"),
        ("Unambiguous fd", f"{unambiguous_min:.3f} to {unambiguous_max:.3f} Hz"),
        ("Displayed fd", f"{fd_min:.3f} to {fd_max:.3f} Hz"),
    ]
    if power_floor_db is not None:
        rows.append(("Power floor", f"{power_floor_db:.3f} dB"))

    has_info = bool(rows or corner_note or info_panel_footer_note)
    fig, ax = _new_figure_with_info(figsize=(14, 6)) if has_info else plt.subplots(figsize=(10, 6))
    phase_cmap = plt.get_cmap("RdBu_r").copy()
    phase_cmap.set_bad("#050505")
    ax.set_facecolor("#050505")
    pcm = ax.pcolormesh(
        time_plot,
        selected_delay_ms,
        np.ma.masked_invalid(fd_plot.T),
        shading="nearest",
        cmap=phase_cmap,
        vmin=fd_min,
        vmax=fd_max,
    )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Delay [ms]")
    ax.set_title(title)
    _add_delay_range_axis(ax)
    _add_colorbar(fig, pcm, ax, "Phase-derived Doppler [Hz]")
    _add_info_panel(fig, info_rows=rows, corner_note=corner_note, footer_note=info_panel_footer_note)
    _tight_layout_with_info(fig, has_info=has_info)

    if output_file:
        _savefig(output_file, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_phase_doppler_delay_frame(
    fd_hz: np.ndarray,
    power_db: np.ndarray,
    selected_delay_ms: np.ndarray,
    lfm_config,
    title: str,
    output_file: str = None,
    fd_min: float = None,
    fd_max: float = None,
    power_floor_db: float = None,
    vmin: float = None,
    vmax: float = None,
    corner_note: str = None,
    info_rows=None,
    info_panel_footer_note: str = None,
):
    selected_delay_ms = np.asarray(selected_delay_ms)
    fd_hz = np.asarray(fd_hz)
    power_db = np.asarray(power_db)
    if selected_delay_ms.ndim != 1 or fd_hz.shape != selected_delay_ms.shape:
        raise ValueError("fd_hz and selected_delay_ms must be matching 1D arrays")
    if power_db.shape != selected_delay_ms.shape:
        raise ValueError("power_db and selected_delay_ms must be matching 1D arrays")

    prf = lfm_config.sweep_frequency
    fd_min, fd_max, unambiguous_min, unambiguous_max = _phase_doppler_axis_limits(
        prf,
        fd_min=fd_min,
        fd_max=fd_max,
    )

    valid = np.isfinite(fd_hz) & np.isfinite(power_db)
    if power_floor_db is not None:
        valid &= power_db >= power_floor_db

    rows = list(info_rows or []) + _lfm_info_rows(lfm_config) + [
        ("Estimator", "sweep-to-sweep phase"),
        ("Unambiguous fd", f"{unambiguous_min:.3f} to {unambiguous_max:.3f} Hz"),
        ("Displayed fd", f"{fd_min:.3f} to {fd_max:.3f} Hz"),
    ]
    if power_floor_db is not None:
        rows.append(("Power floor", f"{power_floor_db:.3f} dB"))

    has_info = bool(rows or corner_note or info_panel_footer_note)
    fig, ax = _new_figure_with_info(figsize=(14, 6)) if has_info else plt.subplots(figsize=(10, 6))
    ax.set_facecolor("#050505")
    if np.any(valid):
        power_cmap = plt.get_cmap("inferno").copy()
        power_cmap.set_bad("#050505")
        sc = ax.scatter(
            fd_hz[valid],
            selected_delay_ms[valid],
            c=power_db[valid],
            s=18,
            cmap=power_cmap,
            vmin=vmin,
            vmax=vmax,
            linewidths=0,
        )
        _add_colorbar(fig, sc, ax, "Power [dB]")
    else:
        ax.text(0.5, 0.5, "No valid phase-Doppler estimates", transform=ax.transAxes,
                ha="center", va="center", color="white")

    ax.set_xlim(fd_min, fd_max)
    ax.set_ylim(float(np.min(selected_delay_ms)), float(np.max(selected_delay_ms)))
    ax.set_xlabel("Phase-derived Doppler [Hz]")
    ax.set_ylabel("Delay [ms]")
    ax.set_title(title)
    _add_delay_range_axis(ax)
    _add_info_panel(fig, info_rows=rows, corner_note=corner_note, footer_note=info_panel_footer_note)
    _tight_layout_with_info(fig, has_info=has_info)

    if output_file:
        _savefig(output_file, dpi=300)
        plt.close()
    else:
        plt.show()

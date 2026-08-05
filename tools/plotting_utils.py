# plotting_utils.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
import os
import textwrap

from lfm_utils import LFMWaveform, dechirp_fft_complex, window_from_arg


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
    end_sample = int(start_sample + (tend - tstart) * fs) if tend is not None else len(iq)
    sliced_iq = iq[start_sample:end_sample]

    noverlap = window_size - hop_size

    # Compute STFT
    f, t, Zxx = stft(sliced_iq, fs=fs, window=window, nperseg=window_size, noverlap=noverlap, return_onesided=False)
    f = np.fft.fftshift(f)
    Zxx = np.fft.fftshift(Zxx, axes=0)
    Zxx_dB = 10 * np.log10(np.abs(Zxx) ** 2 + 1e-12)
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


def plot_matched_filter_output(
        lags: np.ndarray,
        magnitude_response: np.ndarray,
        fs: float,
        title: str = "Matched Filter Output",
        xlim: tuple = (None, None),
        ylim: tuple = (None, None),
        output_file: str = None,
        time_units: str = "s",
        corner_note: str = None,
        info_rows=None,
        info_panel_footer_note: str = None,

):
    """
    Plots the matched filter output as a function of lag time.

    Parameters:
        lags (np.ndarray): Array of sample lags (typically from correlation_lags).
        magnitude_response (np.ndarray): Power or magnitude response (e.g., |corr|² in dB).
        fs (float): Sampling frequency in Hz, used to convert lags to seconds.
        title (str): Plot title.
        xlim (tuple): Optional x-axis limits as (min, max).
        ylim (tuple): Optional y-axis limits as (min, max).
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
    """
    time = lags / fs
    if time_units == "ms":
        time = time * 1000

    has_info = bool(info_rows or corner_note or info_panel_footer_note)
    fig, ax = _new_figure_with_info(figsize=(14, 4.8)) if has_info else plt.subplots(figsize=(10, 4))
    ax.plot(time, magnitude_response)
    ax.grid(True)

    if ylim != (None, None):
        ax.set_ylim(*ylim)

    if xlim != (None, None):
        ax.set_xlim(*xlim)

    ax.set_xlabel(f'Time [{time_units}]')
    ax.set_ylabel('Power [dB]')
    ax.set_title(title)
    if time_units == "ms":
        _add_delay_range_xaxis(ax, label="Range [km]")
    _add_info_panel(
        fig,
        info_rows=info_rows,
        corner_note=corner_note,
        footer_note=info_panel_footer_note,
    )
    _tight_layout_with_info(fig, has_info=has_info)
    if output_file:
        _savefig(output_file, dpi=300)
        plt.show()
        plt.close()
    else:
        plt.show()


def plot_pdp(magnitude_response: np.ndarray, 
             lfm_config: LFMWaveform, 
             window_width: float, 
             title: str,
             output_file: str = None, 
             vmin: float = None,
             vmax: float = None, 
             tstart=None, 
             tend=None, 
             navg=4, 
             tcenter: float = None,
             delay_reference_note: str = None,
             corner_note: str = None,
             info_rows=None,
             info_panel_footer_note: str = None,
):
    if tstart is not None:
        magnitude_response = magnitude_response[int(tstart * lfm_config.sample_rate):]
    if tend is not None:
        magnitude_response = magnitude_response[:int(tend * lfm_config.sample_rate)]

    window_size = int(window_width * lfm_config.sample_rate / 2) * 2

    if tcenter is None:
        tcenter = np.argmax(magnitude_response[0:2 * lfm_config.sweep_length])
    else:
        tcenter = int(tcenter * lfm_config.sample_rate)
    t0 = int(tcenter - window_width * lfm_config.sample_rate / 2)

    # Handle wrap-around if t0 is negative
    if t0 < 0:
        t0 += lfm_config.sweep_length

    # Trim the input to start at t0
    trimmed = magnitude_response[t0:]

    # Create sliding windows
    num_sweeps = (len(trimmed) - window_size) // lfm_config.sweep_length

    sweeps = np.empty((num_sweeps, window_size), dtype=magnitude_response.dtype)

    for i in range(num_sweeps):
        start = i * lfm_config.sweep_length
        sweeps[i] = trimmed[start:start + window_size]

    num_rows = sweeps.shape[0]
    remainder = num_rows % navg

    # Trim array to a multiple of n if needed
    if remainder != 0:
        sweeps = sweeps[:num_rows - remainder]

    # Reshape and average
    averaged = sweeps.reshape(-1, navg, sweeps.shape[1]).mean(axis=1)

    slow_time_len, delay_time_len = averaged.shape

    lag_time = np.linspace(0, delay_time_len / lfm_config.sample_rate, delay_time_len) - (window_width / 2)
    slow_time = np.linspace(0, slow_time_len * navg / lfm_config.sweep_frequency, slow_time_len)

    rows = list(info_rows or []) + _lfm_info_rows(lfm_config)
    has_info = bool(rows or corner_note or delay_reference_note or info_panel_footer_note)
    fig, ax = _new_figure_with_info(figsize=(14, 6)) if has_info else plt.subplots(figsize=(10, 6))
    pcm = ax.pcolormesh(slow_time, lag_time * 1e3, averaged.T, shading='nearest', cmap='inferno', vmin=vmin, vmax=vmax)
    ax.set_ylabel('Relative Time Delay [ms]')
    ax.set_xlabel('Time [s]')
    ax.set_title(title)
    _add_delay_range_axis(ax, label="Relative virtual range [km]")
    _add_colorbar(fig, pcm, ax, 'Power [dB]')
    _add_info_panel(
        fig,
        info_rows=rows,
        corner_note=corner_note,
        lower_note=delay_reference_note,
        footer_note=info_panel_footer_note,
    )
    _tight_layout_with_info(fig, has_info=has_info)
    if output_file:
        _savefig(output_file, dpi=300)
        plt.show()
        plt.close()
    else:
        plt.show()


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
):
    T = 1 / lfm_config.sweep_frequency
    k = lfm_config.bandwidth / T
    if k == 0:
        raise ValueError("Cannot convert dechirp beat frequency to delay when bandwidth is zero")
    chirp_len = stretch_result.shape[1]
    slow_time_len = stretch_result.shape[0]
    
    slow_time = np.linspace(0, slow_time_len / lfm_config.sweep_frequency, slow_time_len)

    freqs = np.fft.fftshift(np.fft.fftfreq(chirp_len, d=1 / lfm_config.sample_rate))
    f_mask = np.abs(freqs) <= (np.abs(lfm_config.bandwidth / 2))
    freqs = freqs[f_mask]

    time_delays = np.mod(-freqs / k * 1e3, T * 1e3)  # milliseconds

    power_db = 10 * np.log10(stretch_result + 1e-12)
    power_db = power_db[:, f_mask]

    delay_order = np.argsort(time_delays)
    time_delays = time_delays[delay_order]
    power_db = power_db[:, delay_order]
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
    

def plot_dechirp_streaming(
        iq,
        lfm_config,
        vmin=None,
        vmax=None,
        title="Stretch-Processed Range-Time Plot",
        output_file=None,
        navg=1,
        d_min=None,
        d_max=None,
        dechirp_window="hann",
        tstart=0.0,
        start_offset_samples=0,
        corner_note=None,
        auto_delay_window=True,
        info_rows=None,
        info_panel_footer_note=None,
):
    chirp_len = lfm_config.sweep_length
    if start_offset_samples < 0:
        raise ValueError("start_offset_samples must be >= 0")
    iq = iq[start_offset_samples:]

    num_chirps = len(iq) // chirp_len
    if num_chirps < 1:
        raise ValueError("Not enough samples for one complete chirp")
    if navg < 1:
        raise ValueError("--navg must be >= 1")

    fs = lfm_config.sample_rate
    prf = lfm_config.sweep_frequency
    k = lfm_config.bandwidth / (1.0 / prf)
    if k == 0:
        raise ValueError("Cannot convert dechirp beat frequency to delay when bandwidth is zero")

    fb = np.fft.fftshift(np.fft.fftfreq(chirp_len, d=1.0 / fs))
    delay_ms = np.mod(-fb / k * 1e3, (1.0 / prf) * 1e3)
    delay_order = np.argsort(delay_ms)
    delay_ms = delay_ms[delay_order]

    reference_chirp = lfm_config.waveform.astype(iq.dtype)
    w = window_from_arg(chirp_len, dechirp_window).astype(np.float32, copy=False)
    coherent_gain = np.mean(w)
    if coherent_gain == 0:
        raise ValueError(f"Window '{dechirp_window}' has zero coherent gain")

    num_groups = num_chirps // navg
    should_auto_delay_window = auto_delay_window and d_min is None and d_max is None
    d_mask = np.ones_like(delay_ms, dtype=bool)
    if not should_auto_delay_window:
        if d_min is not None:
            d_mask = d_mask & (delay_ms >= d_min)
        if d_max is not None:
            d_mask = d_mask & (delay_ms <= d_max)
    selected_bins = delay_order[d_mask]
    delay_plot = delay_ms[d_mask]
    if selected_bins.size == 0:
        raise ValueError(
            "No dechirp delay bins selected; check --d-min/--d-max. "
            f"Available delay range is {delay_ms.min():.3f} to {delay_ms.max():.3f} ms "
            f"for sweep_frequency={prf:g} Hz "
            f"(chirp period={(1.0 / prf) * 1e3:.3f} ms)."
        )
    power = np.empty((num_groups, selected_bins.size), dtype=np.float32)

    for group_idx in range(num_groups):
        acc = np.zeros(selected_bins.size, dtype=np.float64)
        for j in range(navg):
            chirp_idx = group_idx * navg + j
            start = chirp_idx * chirp_len
            seg = iq[start:start + chirp_len]
            beat = seg * np.conj(reference_chirp)
            spectrum = np.fft.fftshift(np.fft.fft(beat * w) / coherent_gain)
            acc += np.abs(spectrum[selected_bins]) ** 2
        power[group_idx] = acc / navg

    power_db = 10.0 * np.log10(power + 1e-12)
    d_min, d_max, strongest_delay_ms = _auto_delay_bounds(
        delay_plot,
        power_db,
        d_min,
        d_max,
        enabled=auto_delay_window,
    )
    if should_auto_delay_window:
        d_mask = np.ones_like(delay_plot, dtype=bool)
        if d_min is not None:
            d_mask = d_mask & (delay_plot >= d_min)
        if d_max is not None:
            d_mask = d_mask & (delay_plot <= d_max)
        delay_plot = delay_plot[d_mask]
        power_db = power_db[:, d_mask]
        if delay_plot.size == 0:
            raise ValueError(
                "No dechirp delay bins selected; check --d-min/--d-max. "
                f"Available delay range is {delay_ms.min():.3f} to {delay_ms.max():.3f} ms "
                f"for sweep_frequency={prf:g} Hz "
                f"(chirp period={(1.0 / prf) * 1e3:.3f} ms)."
            )
    slow_time = tstart + np.arange(num_groups) * navg / prf

    rows = list(info_rows or []) + _lfm_info_rows(lfm_config)
    lower_note = _format_delay_peaks_note(power_db, delay_plot, n=5)
    has_info = bool(rows or corner_note or lower_note or info_panel_footer_note)
    fig, ax = _new_figure_with_info(figsize=(14, 6)) if has_info else plt.subplots(figsize=(10, 6))
    pcm = ax.pcolormesh(slow_time, delay_plot, power_db.T, shading="nearest",
                        cmap="inferno", vmin=vmin, vmax=vmax)
    ax.set_ylabel("Delay [ms]")
    ax.set_xlabel("Time [s]")
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
        _savefig(output_file, dpi=300)
        plt.close()
    else:
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
):
    B = lfm_config.bandwidth
    fs = lfm_config.sample_rate
    PRF = lfm_config.sweep_frequency
    T = 1.0 / PRF
    k = B / T
    if k == 0:
        raise ValueError("Cannot convert dechirp beat frequency to delay when bandwidth is zero")

    if dechirp_spectra.ndim != 2:
        raise ValueError("dechirp_spectra must be 2D: (num_chirps, n_bins)")
    
    slow_len, n_bins = dechirp_spectra.shape

    # Slow-time window
    w = window_from_arg(slow_len, window_slow)
    coherent_gain = np.mean(w)
    if coherent_gain == 0:
        raise ValueError(f"Slow-time window '{window_slow}' has zero coherent gain")
    x = dechirp_spectra * (w[:, None] / coherent_gain)

    # Doppler FFT across slow-time
    if nfft_doppler is None:
        nfft_doppler = 1 << int(np.ceil(np.log2(max(slow_len, 1))))

    print(f"Using nfft_doppler={nfft_doppler} for Doppler FFT (slow_len={slow_len})")

    DD = np.fft.fftshift(np.fft.fft(x, n=nfft_doppler, axis=0), axes=0)
    
    # Beat frequency axis
    fb = np.fft.fftshift(np.fft.fftfreq(n_bins, d=1.0 / fs))

    # Doppler frequency axis
    fd = np.fft.fftshift(np.fft.fftfreq(nfft_doppler, d=1.0 / PRF))

    power_db = 10.0 * np.log10(np.abs(DD) ** 2 + 1e-12)

    # Convert beat frequency to delay
    delay_s = -fb / k
    delay_ms = delay_s * 1e3
    if positive_delay_axis:
        delay_ms = np.mod(delay_ms, T * 1e3)

    delay_order = np.argsort(delay_ms)
    delay_ms = delay_ms[delay_order]
    power_db = power_db[:, delay_order]
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
    if dechirp_spectra.ndim != 2:
        raise ValueError("dechirp_spectra must be 2D: (num_chirps, n_bins)")
    if d_min is None or d_max is None:
        raise ValueError("Both --d-min and --d-max are required for Doppler-vs-time plots")
    if integration_time <= 0 or hop_time <= 0:
        raise ValueError("--integration-time and --hop-time must be > 0")
    if combine not in ("incoherent", "coherent", "peak"):
        raise ValueError("--combine must be one of: incoherent, coherent, peak")

    num_chirps, n_bins = dechirp_spectra.shape

    if selected_delay_ms is None:
        B = lfm_config.bandwidth
        fs = lfm_config.sample_rate
        T = 1.0 / prf
        k = B / T
        if k == 0:
            raise ValueError("Cannot convert dechirp beat frequency to delay when bandwidth is zero")

        fb = np.fft.fftshift(np.fft.fftfreq(n_bins, d=1.0 / fs))
        delay_ms = np.mod(-fb / k * 1e3, T * 1e3)
        delay_order = np.argsort(delay_ms)
        delay_ms = delay_ms[delay_order]
        spectra = dechirp_spectra[:, delay_order]

        d_mask = (delay_ms >= d_min) & (delay_ms <= d_max)
        selected_delays = delay_ms[d_mask]
        if selected_delays.size == 0:
            raise ValueError(
                "No delay bins selected; check --d-min/--d-max. "
                f"Available delay range is {delay_ms.min():.3f} to {delay_ms.max():.3f} ms."
            )
        band = spectra[:, d_mask]
    else:
        selected_delays = np.asarray(selected_delay_ms)
        if selected_delays.ndim != 1 or selected_delays.size != n_bins:
            raise ValueError("selected_delay_ms must be 1D and match dechirp_spectra.shape[1]")
        band = dechirp_spectra

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

    w = window_from_arg(window_len, slow_window).astype(np.float32, copy=False)
    coherent_gain = np.mean(w)
    if coherent_gain == 0:
        raise ValueError(f"Slow-time window '{slow_window}' has zero coherent gain")

    frame_starts = np.arange(0, num_chirps - window_len + 1, hop_len, dtype=int)
    fd = np.fft.fftshift(np.fft.fftfreq(nfft_doppler, d=1.0 / prf))

    power = np.empty((len(frame_starts), len(fd)), dtype=np.float32)
    for frame_idx, start in enumerate(frame_starts):
        x = band[start:start + window_len, :]
        xw = x * (w[:, None] / coherent_gain)
        X = np.fft.fftshift(np.fft.fft(xw, n=nfft_doppler, axis=0), axes=0)
        bin_power = np.abs(X) ** 2

        if combine == "incoherent":
            power[frame_idx] = np.mean(bin_power, axis=1)
        elif combine == "coherent":
            coherent = np.sum(X, axis=1)
            power[frame_idx] = np.abs(coherent) ** 2 / X.shape[1]
        else:
            strongest_bin = int(np.argmax(np.mean(np.abs(x) ** 2, axis=0)))
            power[frame_idx] = bin_power[:, strongest_bin]

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


def delay_doppler_process_window(iq_chunk, frame_idx, args, lfm_config, timestamps, start_timestamp=None):
    interactive = getattr(args, "interactive", False)
    auto_delay_window = frame_idx is None
    info_rows = []
    input_file = getattr(args, "input_file", None) or getattr(args, "filename", None)
    if input_file:
        info_rows.append(("File", os.path.basename(input_file)))
    tuning_frequency = getattr(args, "tuning_frequency", None)
    if tuning_frequency is not None:
        info_rows.append(("Tuning freq", f"{float(tuning_frequency) / 1e6:.6f} MHz"))
    if getattr(args, "stand", None) is not None or getattr(args, "pol", None) is not None:
        info_rows.append(("Stand/pol", f"{getattr(args, 'stand', None)}/{getattr(args, 'pol', None)}"))
    info_rows.append(("Integration", f"{getattr(args, 'integration_time', len(iq_chunk) / lfm_config.sample_rate):.3f} s"))
    if getattr(args, "hop_time", None) is not None:
        info_rows.append(("Hop", f"{args.hop_time:.3f} s"))
    if getattr(args, "gap_fill", None) is not None:
        info_rows.append(("Gap fill", args.gap_fill))
    if getattr(args, "fd_min", None) is not None or getattr(args, "fd_max", None) is not None:
        info_rows.append(("Doppler limits", f"{getattr(args, 'fd_min', None)} to {getattr(args, 'fd_max', None)} Hz"))
    if getattr(args, "d_min", None) is not None or getattr(args, "d_max", None) is not None:
        info_rows.append(("Delay limits", f"{getattr(args, 'd_min', None)} to {getattr(args, 'd_max', None)} ms"))

    if args.output is not None and frame_idx is not None:
        output_file = f"{args.output}/frame_{frame_idx:04d}.png"
    elif args.output is not None:
        output_file = args.output
    else:
        output_file = None
    info_panel_footer_note = getattr(args, "gap_fill_note", None)

    start_offset_samples = _timestamp_sweep_offset_samples(
        start_timestamp,
        lfm_config,
        sweep_offset=getattr(args, "offset", 0.0),
    )

    _, complex_spectra = dechirp_fft_complex(
        received_signal=iq_chunk,
        lfm_config=lfm_config,
        window=args.dechirp_window,
        start_offset_samples=start_offset_samples,
    )

    plot_delay_doppler_dechirp(
        dechirp_spectra=complex_spectra,
        lfm_config=lfm_config,
        title=args.title,
        output_file=output_file,
        vmin=args.vmin,
        vmax=args.vmax,
        window_slow=args.slow_window,
        nfft_doppler=args.nfft_doppler,
        fd_max=args.fd_max,
        fd_min=args.fd_min,
        d_max=args.d_max,
        d_min=args.d_min,
        interactive=interactive,
        positive_delay_axis=True,
        corner_note=timestamps,
        auto_delay_window=auto_delay_window,
        info_rows=info_rows,
        info_panel_footer_note=info_panel_footer_note,
    )


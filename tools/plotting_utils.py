# plotting_utils.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
import os
import textwrap

from lfm_utils import LFMWaveform, dechirp_fft_complex, lfm_matched_filtering, window_from_arg


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


def _strongest_delay_bin_ms(power, delay_ms):
    if power.size == 0 or delay_ms.size == 0:
        return None
    _, delay_idx = np.unravel_index(np.nanargmax(power), power.shape)
    return float(delay_ms[delay_idx])


def _auto_delay_bounds(delay_ms, power, d_min, d_max, half_width_ms=10.0):
    strongest_delay_ms = _strongest_delay_bin_ms(power, delay_ms)
    if d_min is None and d_max is None and strongest_delay_ms is not None:
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

    # Plot vertical spectrogram
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
    _add_corner_note(corner_note, fig=fig)
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300)
        plt.close()
        print(f"Saved spectrogram to '{output_file}'")
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

    plt.figure(figsize=(10, 4))
    plt.plot(time, magnitude_response)
    plt.grid(True)

    if ylim != (None, None):
        plt.ylim(*ylim)

    if xlim != (None, None):
        plt.xlim(*xlim)

    plt.xlabel(f'Time [{time_units}]')
    plt.ylabel('Power [dB]')
    plt.title(title)
    _add_corner_note(corner_note)
    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=300)
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

    plt.figure(figsize=(10, 6))
    plt.pcolormesh(slow_time, lag_time * 1e3, averaged.T, shading='nearest', cmap='inferno', vmin=vmin, vmax=vmax)
    plt.ylabel('Relative Time Delay [ms]')
    plt.xlabel('Time [s]')
    plt.title(title)
    plt.colorbar(label='Power [dB]')
    if delay_reference_note:
        wrapped_note = "\n".join(textwrap.wrap(delay_reference_note, width=115))
        plt.figtext(0.5, 0.01, wrapped_note, ha='center', va='bottom', fontsize=8)
        plt.tight_layout(rect=[0, 0.08, 1, 1])
    else:
        plt.tight_layout()
    _add_corner_note(corner_note)
    if output_file:
        plt.savefig(output_file, dpi=300)
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
    auto_delay_window = d_min is None and d_max is None
    d_min, d_max, strongest_delay_ms = _auto_delay_bounds(
        time_delays,
        power_db,
        d_min,
        d_max,
    )

    d_mask = np.ones_like(time_delays, dtype=bool)
    if d_min is not None:
        d_mask = d_mask & (time_delays >= d_min)
    if d_max is not None:
        d_mask = d_mask & (time_delays <= d_max)
    time_delays = time_delays[d_mask]
    power_db = power_db[:, d_mask]
    if not auto_delay_window:
        strongest_delay_ms = _strongest_delay_bin_ms(power_db, time_delays)

    if navg > 1:
        nrows = (power_db.shape[0] // navg) * navg
        power_db = power_db[:nrows].reshape(-1, navg, power_db.shape[1]).mean(axis=1)
        slow_time_len = power_db.shape[0]
        slow_time = tstart + np.arange(slow_time_len) * navg / lfm_config.sweep_frequency
    else:
        slow_time = tstart + np.arange(slow_time_len) / lfm_config.sweep_frequency

    plt.figure(figsize=(10, 6))
    plt.pcolormesh(slow_time, time_delays, power_db.T, shading='nearest', cmap='inferno', vmin=vmin, vmax=vmax)
    plt.ylabel('Delay [ms]')
    plt.xlabel('Time [s]')
    plt.title(title)
    plt.colorbar(label='Power [dB]')
    if strongest_delay_ms is not None:
        _add_lower_left_note(f"Strongest delay bin: {strongest_delay_ms:.3f} ms")
    _add_corner_note(corner_note)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
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
    auto_delay_window = d_min is None and d_max is None
    d_mask = np.ones_like(delay_ms, dtype=bool)
    if not auto_delay_window:
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
    )
    if auto_delay_window:
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

    plt.figure(figsize=(10, 6))
    plt.pcolormesh(slow_time, delay_plot, power_db.T, shading="nearest",
                   cmap="inferno", vmin=vmin, vmax=vmax)
    plt.ylabel("Delay [ms]")
    plt.xlabel("Time [s]")
    plt.title(title)
    plt.colorbar(label="Power [dB]")
    if strongest_delay_ms is not None:
        _add_lower_left_note(f"Strongest delay bin: {strongest_delay_ms:.3f} ms")
    _add_corner_note(corner_note)
    plt.tight_layout()

    if output_file:
        full_path = os.path.expanduser(output_file)
        output_dir = os.path.dirname(full_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(full_path, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_delay_doppler_mf(
    complex_response: np.ndarray,
    lfm_config,
    window_width: float,
    title: str,
    output_file: str = None,
    vmin: float = None,
    vmax: float = None,
    tstart=None,
    tend=None,
    tcenter: float = None,
    window_slow: str = "hann",
    nfft_doppler: int = None,
    fd_max: float = None,
    fd_min: float = None,
    d_max: float = None,
    d_min: float = None,
    interactive: bool = False,
    corner_note: str = None,
):
    """
    MF-based DD:
      - complex_response is matched-filter complex output vs time samples
      - window around each sweep (fast-time window)
      - stack one window per sweep
      - Doppler FFT across sweeps
      - y-axis is delay in ms, x-axis is Doppler in Hz
      ...
    """

    # Trim in time
    if tstart is not None:
        complex_response = complex_response[int(tstart * lfm_config.sample_rate):]
    if tend is not None:
        complex_response = complex_response[:int(tend * lfm_config.sample_rate)]

    if window_width is None:
        window_width = 1 / lfm_config.sweep_frequency

    fs = lfm_config.sample_rate
    window_size = int(window_width * fs / 2) * 2  # even

    if tcenter is None:
        search_len = min(int(2 * lfm_config.sweep_length), len(complex_response))
        tcenter = int(np.argmax(np.abs(complex_response[:search_len])))

    t0 = int(tcenter - window_width * fs / 2)

    # Handle wrap-around if t0 is negative
    if t0 < 0:
        t0 += lfm_config.sweep_length

    trimmed = complex_response[t0:]
    if len(trimmed) < window_size:
        raise ValueError(f"Not enough samples after trimming for window_size={window_size}")

    # all_windows = sliding_window_view(trimmed, window_shape=window_size) # (len(trimmed) - window_size + 1, window_size)
    # sweeps = all_windows[::lfm_config.sweep_length]  # (num_sweeps, window_size)

    num_sweeps = (len(trimmed) - window_size) // lfm_config.sweep_length

    sweeps = np.empty((num_sweeps, window_size), dtype=complex_response.dtype)

    for i in range(num_sweeps):
        start = i * lfm_config.sweep_length
        sweeps[i] = trimmed[start:start + window_size]

    slow_len, delay_len = sweeps.shape

    # Slow-time window
    w = window_from_arg(slow_len, window_slow)
    coherent_gain = np.mean(w)
    if coherent_gain == 0:
        raise ValueError(f"Slow-time window '{window_slow}' has zero coherent gain")

    x = sweeps * (w[:, None] / coherent_gain)

    PRF = lfm_config.sweep_frequency
    if nfft_doppler is None:
        nfft_doppler = 1 << int(np.ceil(np.log2(max(slow_len, 1))))

    DD = np.fft.fftshift(np.fft.fft(x, n=nfft_doppler, axis=0), axes=0)
    fd = np.fft.fftshift(np.fft.fftfreq(nfft_doppler, d=1.0 / PRF))

    power_db = 10 * np.log10(np.abs(DD) ** 2 + 1e-12)

    # Delay axis from delay samples 
    delay_s = ((np.arange(delay_len) - delay_len // 2) / fs) 
    delay_ms = delay_s * 1e3
    auto_delay_window = d_min is None and d_max is None
    d_min, d_max, strongest_delay_ms = _auto_delay_bounds(
        delay_ms,
        power_db,
        d_min,
        d_max,
    )

    # Apply delay mask
    d_mask = np.ones_like(delay_ms, dtype=bool)

    if d_max is not None:
        d_mask = d_mask & (delay_ms <= d_max)

    if d_min is not None:
        d_mask = d_mask & (delay_ms >= d_min)

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

    d_plot = delay_ms[d_mask]
    power_db = power_db[:, d_mask]
    if d_plot.size == 0:
        raise ValueError(
            "No delay bins selected; check --d-min/--d-max. "
            f"Available delay range is {delay_ms.min():.3f} to {delay_ms.max():.3f} ms."
        )
    if not auto_delay_window:
        strongest_delay_ms = _strongest_delay_bin_ms(power_db, d_plot)

    plt.figure(figsize=(10, 6))
    plt.pcolormesh(fd_plot, d_plot, power_db.T, shading="nearest", cmap="inferno", vmin=vmin, vmax=vmax)
    plt.xlabel("Doppler Frequency [Hz]")
    plt.ylabel("Delay [ms]")
    plt.title(title)
    plt.colorbar(label="Power [dB]")
    if strongest_delay_ms is not None:
        _add_lower_left_note(f"Strongest delay bin: {strongest_delay_ms:.3f} ms")
    _add_corner_note(corner_note)
    plt.tight_layout()
    if output_file:
        # Expand the ~ if it exists
        full_path = os.path.expanduser(output_file)
        
        # Create the directory if it's missing
        output_dir = os.path.dirname(full_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        plt.savefig(full_path, dpi=300)
        if interactive == True:
            plt.show()
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
    auto_delay_window = d_min is None and d_max is None
    d_min, d_max, strongest_delay_ms = _auto_delay_bounds(
        delay_ms,
        power_db,
        d_min,
        d_max,
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
    if not auto_delay_window:
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

    plt.figure(figsize=(10, 6))
    plt.pcolormesh(fd_plot, d_plot, power_db.T, shading="nearest", cmap="inferno", vmin=vmin, vmax=vmax)
    plt.xlabel("Doppler Frequency [Hz]")
    plt.ylabel("Delay [ms]")
    plt.title(title)
    plt.colorbar(label="Power [dB]")
    if strongest_delay_ms is not None:
        _add_lower_left_note(f"Strongest delay bin: {strongest_delay_ms:.3f} ms")
    _add_corner_note(corner_note)
    plt.tight_layout()
    if output_file:
        # Expand the ~ if it exists
        full_path = os.path.expanduser(output_file)
        
        # Create the directory if it's missing
        output_dir = os.path.dirname(full_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        plt.savefig(full_path, dpi=300)
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

    plt.figure(figsize=(10, 6))
    plt.pcolormesh(time_plot, fd_plot, power_db.T, shading="nearest",
                   cmap="inferno", vmin=vmin, vmax=vmax)
    plt.xlabel("Time [s]")
    plt.ylabel("Doppler Frequency [Hz]")
    plt.title(
        f"{title}\n"
        f"Delay band {selected_delays.min():.3f}-{selected_delays.max():.3f} ms, "
        f"{selected_delays.size} bins, {combine} combine"
    )
    plt.colorbar(label="Power [dB]")
    _add_corner_note(corner_note)
    plt.tight_layout()

    if output_file:
        full_path = os.path.expanduser(output_file)
        output_dir = os.path.dirname(full_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(full_path, dpi=300)
        plt.close()
    else:
        plt.show()


def delay_doppler_process_window(iq_chunk, frame_idx, args, lfm_config, timestamps, start_timestamp=None):
    interactive = getattr(args, "interactive", False)

    if args.output is not None and frame_idx is not None:
        output_file = f"{args.output}/frame_{frame_idx:04d}.png"
    elif args.output is not None:
        output_file = args.output
    else:
        output_file = None

    if args.method == "mf":
        _, _, complex_response = lfm_matched_filtering(iq_chunk, lfm_config)

        plot_delay_doppler_mf(
            complex_response=complex_response,
            lfm_config=lfm_config,
            window_width=getattr(args, "window_width", getattr(args, "window", None)),
            title=args.title,
            output_file=output_file,
            vmin=args.vmin,
            vmax=args.vmax,
            tcenter=args.window_center,
            window_slow=args.slow_window,
            nfft_doppler=args.nfft_doppler,
            fd_max=args.fd_max,
            fd_min=args.fd_min,
            d_max=args.d_max,
            d_min=args.d_min,
            interactive=interactive,
            corner_note=timestamps,
        )

    else:
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
        )


def plot_micro_doppler(
    dechirp_spectra: np.ndarray,   # (num_chirps, n_bins), complex, fftshifted
    lfm_config,
    title: str,
    output_file: str = None,
    vmin: float = None,
    vmax: float = None,
    # --- Micro-Doppler specific ---
    chirp_index: int = 0,           # Which chirp to analyse (or 'mean' across chirps)
    use_mean_chirp: bool = False,   # Average chirps before analysis (improves SNR)
    nperseg: int = 256,             # STFT window length (samples)
    noverlap: int = None,           # STFT overlap; defaults to 75% of nperseg
    nfft_stft: int = None,          # STFT FFT size; defaults to next power of 2 >= nperseg
    window_stft: str = "hann",
    beat_bin_window: int = 10,      # +/- bins around beat peak to extract phase track
    fd_max: float = None,         # Hz, optional zoom on micro-Doppler axis
    fd_min: float = None,
    d_max: float = None,   # delay axis limit in milliseconds
    d_min: float = None,
    # --- Phase track panel ---
    show_phase_track: bool = True,  # Also plot instantaneous phase vs. fast time
):
    """
    Micro-Doppler / within-sweep ionospheric scintillation analysis.

    Instead of the chirp-to-chirp (slow-time) Doppler FFT — which is limited to
    [-PRF/2, PRF/2] = [-0.5, 0.5] Hz at PRF=1 Hz — this function analyses the
    PHASE EVOLUTION of the beat-frequency signal within a single chirp sweep.

    The beat signal at the ionospheric target's delay bin has the form:
        s(t) = A(t) * exp(j * 2*pi * (fb + fd_ion(t)) * t)
    where fd_ion(t) is the instantaneous ionospheric Doppler (scintillation).

    An STFT along fast-time reveals fd_ion(t) at a temporal resolution of
    nperseg/fs seconds and a Doppler resolution of fs/nfft_stft Hz, both far
    better than anything achievable on the slow-time axis with PRF=1 Hz.

    Parameters
    ----------
    dechirp_spectra : (num_chirps, n_bins) complex array, fftshifted beat spectra
    lfm_config      : object with .bandwidth, .sample_rate, .sweep_frequency
    chirp_index     : which chirp row to use (ignored if use_mean_chirp=True)
    use_mean_chirp  : coherently average all chirps before STFT (boosts SNR,
                      smooths slow amplitude variations)
    nperseg         : STFT segment length in samples — controls time resolution.
                      Shorter → better time resolution, worse Doppler resolution.
                      Rule of thumb: nperseg ~ fs / max_expected_fd_ion
    noverlap        : STFT overlap in samples (default: 75% of nperseg)
    nfft_stft       : FFT size for STFT (default: next pow2 >= nperseg)
    beat_bin_window : half-width in bins around the peak beat bin used to extract
                      the complex envelope for phase-track analysis
    fd_limit        : clip the displayed Doppler axis to [-fd_limit, +fd_limit] Hz
    show_phase_track: if True, adds a second panel showing unwrapped phase vs time
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import stft as scipy_stft

    B   = lfm_config.bandwidth
    fs  = lfm_config.sample_rate
    PRF = lfm_config.sweep_frequency
    T   = 1.0 / PRF
    k   = B / T   # chirp rate Hz/s

    if dechirp_spectra.ndim != 2:
        raise ValueError("dechirp_spectra must be 2D: (num_chirps, n_bins)")
    num_chirps, n_bins = dechirp_spectra.shape

    # ------------------------------------------------------------------
    # 1.  Select / form the beat-domain signal to analyse
    # ------------------------------------------------------------------
    if use_mean_chirp:
        # Coherent average across chirps — maximises SNR for a stable target.
        # Incoherent scintillation amplitude variations are preserved in phase.
        beat_signal = np.mean(dechirp_spectra, axis=0)   # (n_bins,)
    else:
        beat_signal = dechirp_spectra[chirp_index, :]    # (n_bins,)

    # ------------------------------------------------------------------
    # 2.  Locate the dominant beat frequency bin (ionospheric echo peak)
    # ------------------------------------------------------------------
    peak_bin = int(np.argmax(np.abs(beat_signal)))

    # Beat frequency axis (fftshifted, so index 0 = most negative freq)
    fb_axis = np.fft.fftshift(np.fft.fftfreq(n_bins, d=1.0 / fs))
    fb_peak = fb_axis[peak_bin]
    delay_peak_ms = np.mod(-fb_peak / k * 1e3, T * 1e3)

    # ------------------------------------------------------------------
    # 3.  Extract the complex envelope around the peak bin
    #     Summing neighbouring bins improves SNR without distorting phase
    #     when the target is well-resolved.
    # ------------------------------------------------------------------
    lo = max(0, peak_bin - beat_bin_window)
    hi = min(n_bins, peak_bin + beat_bin_window + 1)
    # Phase-coherent sum: rotate each bin back to baseband first
    bin_offsets = np.arange(lo, hi) - peak_bin
    bin_freqs   = fb_axis[lo:hi] - fb_peak          # residual freq offsets
    # Build a time vector for the beat signal (fast-time within one sweep)
    t_fast = np.arange(n_bins) / fs                  # (n_bins,) seconds

    # Derotate each contributing bin to remove its carrier, then sum
    envelope = np.zeros(n_bins, dtype=complex)
    for i, b in enumerate(range(lo, hi)):
        phase_ramp = np.exp(-1j * 2 * np.pi * bin_freqs[i] * t_fast)
        envelope += beat_signal[b] * phase_ramp      # project to common phase centre

    # ------------------------------------------------------------------
    # 4.  STFT of the complex envelope → micro-Doppler spectrogram
    # ------------------------------------------------------------------
    if noverlap is None:
        noverlap = int(0.75 * nperseg)
    if nfft_stft is None:
        nfft_stft = 1 << int(np.ceil(np.log2(max(nperseg, 1))))

    window_map = {
        "hann": "hann", "hanning": "hann",
        "hamming": "hamming", "blackman": "blackman", "none": "boxcar"
    }
    win_name = window_map.get(window_stft.lower(), "hann")

    # scipy.signal.stft returns (freqs, times, Zxx)
    # We feed it the MAGNITUDE-normalised envelope so the spectrogram
    # shows Doppler structure rather than amplitude variations.
    f_stft, t_stft, Zxx = scipy_stft(
        envelope,
        fs=fs,
        window=win_name,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft_stft,
        return_onesided=False,   # complex signal → two-sided
    )

    # fftshift so DC (fd=0) is centred
    f_stft = np.fft.fftshift(f_stft)
    Zxx    = np.fft.fftshift(Zxx, axes=0)

    power_db = 10.0 * np.log10(np.abs(Zxx) ** 2 + 1e-12)

    # Convert fast-time axis to milliseconds for display
    t_stft_ms = t_stft * 1e3

    # Optional Doppler zoom
    fd_mask = np.ones(len(f_stft), dtype=bool)

    if fd_max is not None and fd_min is not None:
        fd_mask = (f_stft >= fd_min) & (f_stft <= fd_max)
    else:
        fd_mask = np.ones(len(f_stft), dtype=bool)

    f_plot  = f_stft[fd_mask]
    pow_plot = power_db[fd_mask, :]

    # ------------------------------------------------------------------
    # 5.  Instantaneous phase track (unwrapped)
    # ------------------------------------------------------------------
    inst_phase = np.unwrap(np.angle(envelope))   # radians, fast-time
    # Instantaneous frequency = d(phase)/dt  [Hz]
    inst_freq  = np.diff(inst_phase) / (2 * np.pi) * fs
    t_inst_ms  = np.arange(len(inst_freq)) / fs * 1e3

    # ------------------------------------------------------------------
    # 6.  Plot
    # ------------------------------------------------------------------
    n_panels = 2 if show_phase_track else 1
    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(11, 5 * n_panels),
        squeeze=False,
    )

    # --- Panel 1: Micro-Doppler STFT spectrogram ---
    ax0 = axes[0, 0]
    pcm = ax0.pcolormesh(
        t_stft_ms, f_plot, pow_plot,
        shading="nearest", cmap="inferno",
        vmin=vmin, vmax=vmax,
    )
    ax0.set_xlabel("Fast Time within Sweep [ms]")
    ax0.set_ylabel("Micro-Doppler Frequency [Hz]")
    ax0.set_title(
        f"{title}\n"
        f"Micro-Doppler STFT  |  Beat peak @ {fb_peak/1e3:.2f} kHz "
        f"(delay ≈ {delay_peak_ms:.3f} ms)  |  "
        f"Δf_res = {fs/nfft_stft:.1f} Hz  |  "
        f"Δt_res = {nperseg/fs*1e3:.2f} ms"
    )
    fig.colorbar(pcm, ax=ax0, label="Power [dB]")

    # --- Panel 2: Instantaneous frequency track ---
    if show_phase_track:
        ax1 = axes[1, 0]
        ax1.plot(t_inst_ms, inst_freq, color="orange", lw=0.8)
        if fd_min is not None and fd_max is not None:
            ax1.set_ylim(fd_min, fd_max)
        ax1.axhline(0, color="white" if plt.rcParams["axes.facecolor"] == "black" else "black",
                    lw=0.6, ls=":")
        ax1.set_xlabel("Fast Time [ms]")
        ax1.set_ylabel("Inst. Doppler [Hz]")
        ax1.set_title("Instantaneous Frequency Track (d phase/dt) — Scintillation Signature")
        ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=300)
        plt.show()
        plt.close()
    else:
        plt.show()

# tbn_utils.py

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for plotting
import matplotlib.pyplot as plt
import numpy as np
import os
import psutil
process = psutil.Process(os.getpid())

# LSL imports
from lsl.reader.ldp import LWADataFile, TBNFile
from lsl.correlator import fx as fxc
from lsl.reader import errors

# LFM utils
from lfm_utils import window_from_arg


# -----------------------------
# Helpers
# -----------------------------

def mem():
    print(f"Memory usage: {process.memory_info().rss / 1e9:.2f} GB", flush=True)

def best_freq_units(freq_hz: np.ndarray):
    """Return (freq_scaled, units) where units is Hz/kHz/MHz/GHz."""
    mx = float(np.max(np.abs(freq_hz))) if freq_hz.size else 1.0
    if mx >= 1e9:
        return freq_hz / 1e9, "GHz"
    if mx >= 1e6:
        return freq_hz / 1e6, "MHz"
    if mx >= 1e3:
        return freq_hz / 1e3, "kHz"
    return freq_hz, "Hz"

def pol_to_index(pol) -> int:
    p = str(pol).strip().lower()
    if p in ("x", "0"):
        return 0
    if p in ("y", "1"):
        return 1
    raise ValueError("--pol must be x/y or 0/1")

def lsl_open_tbn(filename: str):
    idf = LWADataFile(filename)
    if not isinstance(idf, TBNFile):
        raise RuntimeError(f"'{filename}' does not appear to be a valid TBN file (LSL type={type(idf)})")
    return idf

def lsl_print_metadata(idf: TBNFile):
    nFramesFile = idf.get_info("nframe")
    fs = idf.get_info("sample_rate")
    beginDate = idf.get_info("start_time").datetime
    central_freq = idf.get_info("freq1")
    print("=== LSL metadata ===")
    print(f"Opened as: {type(idf)}")
    print(f"Start time: {beginDate}")
    print(f"Sample rate: {fs} Hz")
    print(f"Tuning frequency (freq1): {central_freq:.3f} Hz")
    print(f"Frames in file: {nFramesFile}")
    print("====================")

def lsl_read_block_for_one_stream(idf: TBNFile, start_time, duration, stand_id=None, pol=None):
    """
    Robustly read enough samples for a spectrogram from one stand/pol stream.

    Returns:
        x   : complex samples (1D)
        t   : time vector for x (1D, seconds)
    """
    # Decide stream_index
    pol_idx = pol_to_index(pol)          # x/0 -> 0, y/1 -> 1
    stream_index = 2 * (int(stand_id) - 1) + pol_idx
    if stream_index < 0:
        raise ValueError("Invalid --stand/--pol combination (negative index)")

    # Position reader
    idf.offset(start_time)
    target_seconds = duration

    chunk_seconds = min(0.25, target_seconds)

    n_chunks = int(np.ceil(target_seconds / chunk_seconds))
    print("n_chunks = ", n_chunks)

    xs = []
    got = 0.0
    antpols_seen = None
    chunk = 1
    start_timestamp = None

    # print(f"Memory usage before reading: ", flush=True)
    # mem()

    while got < target_seconds:
        try:
            readT, start_timestamp, data = idf.read(min(chunk_seconds, target_seconds - got))

            # if start_timestamp is not None and chunk % 10 == 0:
            #     print(f"Start timestamp: {start_timestamp:.3f} s", flush=True)
            # else:
            #     print(f"Start timestamp: None", flush=True)

            # if chunk % 10 == 0:
            #     print(f"Chunk {chunk}/{n_chunks}: readT={readT:.3f} s, got={got:.3f} s total", flush=True)
            #     mem()

        except errors.EOFError:
            print("Reached EOF", flush=True)
            break
        except RuntimeError as e:
            if "Invalid timetag skip" in str(e):
                # Nudge forward a few ms and keep trying
                idf.offset(0.002)
                continue
            raise

        if data.ndim != 2:
            raise RuntimeError(f"Unexpected data shape from idf.read(): {data.shape}")

        antpols_seen = data.shape[0]
        if stream_index >= antpols_seen:
            raise ValueError(f"Requested stream_index={stream_index} but data has only antpols={antpols_seen}. "
                             f"(Check --stand/--pol)")

        xs.append(data[stream_index, :].copy())

        if len(xs) == 0:
            return None

        del data # free memory immediately

        chunk += 1
        got += readT

        if readT <= 0:
            break

    x = np.concatenate(xs) #if xs else None

    return x, start_timestamp


def lsl_average_spectrum_all_antpols(idf: TBNFile, args):
    """
    Compute PSD for ALL antpols using LSL SpecMaster,
    then average across antpols to produce ONE global spectrum.
    """
    args.offset = idf.offset(args.offset)
    fs = idf.get_info("sample_rate")
    window = window_from_arg(None, args)

    readT, t, data = idf.read(args.duration)
    freq, tempSpec = fxc.SpecMaster(
        data,
        LFFT=args.fft_length,
        window=window,
        pfb=args.pfb,
        verbose=args.verbose,
        sample_rate=fs,
    )

    # tempSpec: (antpols, LFFT)
    spec_avg = np.mean(tempSpec, axis=0)
    return fs, freq, spec_avg

# -----------------------------
# Averaged Spectrum
# -----------------------------
def plot_averaged_spectrum(freq_hz: np.ndarray, psd_linear: np.ndarray, center_freq_hz: float | None, out_png: str | None):
    # Convert to dB
    psd_db = 10.0 * np.log10(np.maximum(psd_linear, 1e-30))

    if center_freq_hz is not None:
        freq_plot = freq_hz + center_freq_hz
    else:
        freq_plot = freq_hz

    freq_scaled, units = best_freq_units(freq_plot)

    plt.figure()
    plt.plot(freq_scaled, psd_db)
    plt.xlabel(f"Frequency [{units}]")
    plt.ylabel("PSD [dB/RBW]")
    plt.title("Averaged Spectrum (single output)")
    plt.tight_layout()
    if out_png:
        plt.savefig(out_png, dpi=150)
    plt.show()


# # -----------------------------
# # MF Output
# # -----------------------------
# def plot_matched_filter_output(
#         lags: np.ndarray,
#         magnitude_response: np.ndarray,
#         fs: float,
#         title: str = "Matched Filter Output",
#         xlim: tuple = (None, None),
#         ylim: tuple = (None, None),
#         output_file: str = None,
#         time_units: str = "s"

# ):
#     """
#     Plots the matched filter output as a function of lag time.

#     Parameters:
#         lags (np.ndarray): Array of sample lags (typically from correlation_lags).
#         magnitude_response (np.ndarray): Power or magnitude response (e.g., |corr|² in dB).
#         fs (float): Sampling frequency in Hz, used to convert lags to seconds.
#         title (str): Plot title.
#         xlim (tuple): Optional x-axis limits as (min, max).
#         ylim (tuple): Optional y-axis limits as (min, max).
#         xlabel (str): Label for the x-axis.
#         ylabel (str): Label for the y-axis.
#     """
#     time = lags / fs
#     if time_units == "ms":
#         time = time * 1000

#     plt.figure(figsize=(10, 4))
#     plt.plot(time, magnitude_response)
#     plt.grid(True)

#     if ylim != (None, None):
#         plt.ylim(*ylim)

#     if xlim != (None, None):
#         plt.xlim(*xlim)

#     plt.xlabel(f'Time [{time_units}]')
#     plt.ylabel('Power [dB]')
#     plt.title(title)
#     plt.tight_layout()
#     if output_file:
#         plt.savefig(output_file, dpi=300)
#         plt.show()
#         plt.close()
#     else:
#         plt.show()


# # -----------------------------
# # Single-stream Spectrogram
# # -----------------------------
# def plot_spectrogram(iq: np.ndarray, fs: float, fc: float, nperseg: int, noverlap: int,
#                      title: str, out_png: str | None, vmin_pct: float, vmax_pct: float):
#     f, t, Z = stft(
#         iq,
#         fs=fs,
#         nperseg=nperseg,
#         noverlap=noverlap,
#         return_onesided=False
#     )
#     S = 20 * np.log10(np.abs(np.fft.fftshift(Z, axes=0)) + 1e-12)
#     ff = np.fft.fftshift(f)

#     vmin = np.percentile(S, vmin_pct)
#     vmax = np.percentile(S, vmax_pct)

#     plt.figure(figsize=(10,8))
#     plt.imshow(
#         S.T,
#         aspect="auto",
#         origin="lower",
#         extent=[(ff[0] + fc) / 1e6, (ff[-1] + fc) / 1e6, t[0], t[-1]],
#         vmin=vmin,
#         vmax=vmax,
#         cmap='inferno'
#     )
#     plt.xlabel("Frequency (MHz)")
#     plt.ylabel("Time (s)")
#     plt.title(title)
#     plt.colorbar(label="dB")
#     plt.tight_layout()

#     if out_png:
#         plt.savefig(out_png, dpi=150)
#     plt.show()


# # -----------------------------
# # Power Delay Profile
# # -----------------------------
# def plot_pdp(magnitude_response: np.ndarray, lfm_config: LFMWaveform, window_width: float, title: str,
#              output_file: str = None, vmin: float = None,
#              vmax: float = None, tstart=None, tend=None, navg=4, tcenter: float = None,):
#     if tstart is not None:
#         magnitude_response = magnitude_response[int(tstart * lfm_config.sample_rate):]
#     if tend is not None:
#         magnitude_response = magnitude_response[:int(tend * lfm_config.sample_rate)]

#     window_size = int(window_width * lfm_config.sample_rate / 2) * 2
#     if tcenter is None:
#         tcenter = np.argmax(magnitude_response[0:2 * lfm_config.sweep_length])
#     t0 = int(tcenter - window_width * lfm_config.sample_rate / 2)

#     # # Quick debug check
#     # peaks, _ = find_peaks(magnitude_response, distance=lfm_config.sweep_length*0.9, height=np.mean(magnitude_response)+10)
#     # diffs = np.diff(peaks)
#     # print(f"Average peak spacing: {np.mean(diffs)}, Expected: {lfm_config.sweep_length}")

#     # Trim the input to start at t0
#     trimmed = magnitude_response[t0:]

#     # Create sliding windows
#     all_windows = sliding_window_view(trimmed, window_shape=window_size)

#     # Select every `sweep_length`-th window
#     pdp_array = all_windows[::lfm_config.sweep_length]

#     num_rows = pdp_array.shape[0]
#     remainder = num_rows % navg

#     # Trim array to a multiple of n if needed
#     if remainder != 0:
#         pdp_array = pdp_array[:num_rows - remainder]

#     # Reshape and average
#     averaged = pdp_array.reshape(-1, navg, pdp_array.shape[1]).mean(axis=1)

#     slow_time_len, delay_time_len = averaged.shape

#     lag_time = np.linspace(0, delay_time_len / lfm_config.sample_rate, delay_time_len) - (window_width / 2)
#     slow_time = np.linspace(0, slow_time_len * navg / lfm_config.sweep_frequency, slow_time_len)

#     plt.figure(figsize=(10, 6))
#     plt.pcolormesh(slow_time, lag_time * 1e3, averaged.T, shading='gouraud', cmap='inferno', vmin=vmin, vmax=vmax)
#     plt.ylabel('Relative Time Delay [ms]')
#     plt.xlabel('Time [s]')
#     plt.title(title)
#     plt.colorbar(label='Power [dB]')
#     plt.tight_layout()
#     if output_file:
#         plt.savefig(output_file, dpi=300)
#         plt.show()
#         plt.close()
#     else:
#         plt.show()

# # -----------------------------
# # Delay-Doppler (MF)
# # -----------------------------
# def plot_delay_doppler_mf(
#     complex_response: np.ndarray,
#     lfm_config,
#     window_width: float,
#     title: str,
#     output_file: str = None,
#     vmin: float = None,
#     vmax: float = None,
#     tstart=None,
#     tend=None,
#     tcenter: float = None,
#     window_slow: str = "hann",
#     nfft_doppler: int = None,
#     fd_limit: float = None,
#     d_limit: float = None,   # delay axis limit in milliseconds
#     positive_only: bool = False,
# ):
#     """
#     MF-based DD:
#       - complex_response is matched-filter complex output vs time samples
#       - window around each sweep (fast-time window)
#       - stack one window per sweep
#       - Doppler FFT across sweeps
#       - y-axis is delay in ms, x-axis is Doppler in Hz
#       ...
#     """

#     # Trim in time
#     if tstart is not None:
#         complex_response = complex_response[int(tstart * lfm_config.sample_rate):]
#     if tend is not None:
#         complex_response = complex_response[:int(tend * lfm_config.sample_rate)]

#     fs = lfm_config.sample_rate
#     window_size = int(window_width * fs / 2) * 2  # even

#     if tcenter is None:
#         search_len = min(int(2 * lfm_config.sweep_length), len(complex_response))
#         tcenter = int(np.argmax(np.abs(complex_response[:search_len])))

#     t0 = int(tcenter - window_width * fs / 2)

#     # Handle wrap-around if t0 is negative
#     if t0 < 0:
#         t0 += lfm_config.sweep_length

#     trimmed = complex_response[t0:]
#     if len(trimmed) < window_size:
#         raise ValueError(f"Not enough samples after trimming for window_size={window_size}")

#     # all_windows = sliding_window_view(trimmed, window_shape=window_size) # (len(trimmed) - window_size + 1, window_size)
#     # sweeps = all_windows[::lfm_config.sweep_length]  # (num_sweeps, window_size)

#     num_sweeps = (len(trimmed) - window_size) // lfm_config.sweep_length

#     sweeps = np.empty((num_sweeps, window_size), dtype=complex_response.dtype)

#     for i in range(num_sweeps):
#         start = i * lfm_config.sweep_length
#         sweeps[i] = trimmed[start:start + window_size]

#     slow_len, delay_len = sweeps.shape

#     # Slow-time window
#     if window_slow is None or window_slow.lower() == "none":
#         w = np.ones(slow_len)
#     elif window_slow.lower() in ("hann", "hanning"):
#         w = np.hanning(slow_len)
#     elif window_slow.lower() == "hamming":
#         w = np.hamming(slow_len)
#     elif window_slow.lower() == "blackman":
#         w = np.blackman(slow_len)
#     else:
#         raise ValueError(f"Unknown window_slow='{window_slow}'")

#     x = sweeps * w[:, None]

#     PRF = lfm_config.sweep_frequency
#     if nfft_doppler is None:
#         nfft_doppler = 1 << int(np.ceil(np.log2(max(slow_len, 1))))

#     DD = np.fft.fftshift(np.fft.fft(x, n=nfft_doppler, axis=0), axes=0)
#     fd = np.fft.fftshift(np.fft.fftfreq(nfft_doppler, d=1.0 / PRF))

#     power_db = 10 * np.log10(np.abs(DD) ** 2 + 1e-12)

#     # Range axis from delay samples 
#     delay_s = ((np.arange(delay_len) - delay_len // 2) / fs) 
#     delay_ms = delay_s * 1e3

#     # Optional zooms
#     if fd_limit is not None:
#         fd_mask = (fd >= -fd_limit) & (fd <= fd_limit)
#         fd_plot = fd[fd_mask]
#         power_db = power_db[fd_mask, :]
#     else:
#         fd_plot = fd

#     if positive_only:
#         d_mask = delay_ms >= 0
#     else:
#         d_mask = np.ones_like(delay_ms, dtype=bool)

#     if d_limit is not None:
#         d_mask = (np.abs(delay_ms) <= d_limit)

#     d_plot = delay_ms[d_mask]
#     power_db = power_db[:, d_mask]

#     print("complex_response:", complex_response.shape, flush=True)
#     print("complex_response GB:", complex_response.nbytes / 1e9, flush=True)

#     print("sweeps shape:", sweeps.shape, flush=True)
#     print("sweeps GB:", sweeps.nbytes / 1e9, flush=True)

#     print("DD shape:", DD.shape, flush=True)
#     print("DD GB:", DD.nbytes / 1e9, flush=True)

#     plt.figure(figsize=(10, 6))
#     plt.pcolormesh(fd_plot, d_plot, power_db.T, shading="nearest", cmap="inferno", vmin=vmin, vmax=vmax)
#     plt.xlabel("Doppler Frequency [Hz]")
#     plt.ylabel("Delay [ms]")
#     plt.title(title)
#     plt.colorbar(label="Power [dB]")
#     plt.tight_layout()
#     if output_file:
#         # Expand the ~ if it exists
#         full_path = os.path.expanduser(output_file)
        
#         # Create the directory if it's missing
#         output_dir = os.path.dirname(full_path)
#         if output_dir and not os.path.exists(output_dir):
#             os.makedirs(output_dir)
            
#         plt.savefig(full_path, dpi=300)
#         plt.close()
#     else:
#         plt.show()

# # -----------------------------
# # Delay-Doppler (Dechirp)
# # -----------------------------
# def plot_delay_doppler_dechirp(
#     dechirp_spectra: np.ndarray,   # (num_chirps, n_bins), complex, fftshifted
#     lfm_config,
#     title: str,
#     output_file: str = None,
#     vmin: float = None,
#     vmax: float = None,
#     window_slow: str = "hann",
#     nfft_doppler: int = None,
#     fd_limit: float = None,
#     d_limit: float = None,        # ms, optional zoom on delay axis
#     positive_only: bool = False,
# ):
#     B = lfm_config.bandwidth
#     fs = lfm_config.sample_rate
#     PRF = lfm_config.sweep_frequency
#     T = 1.0 / PRF
#     k = B / T

#     if dechirp_spectra.ndim != 2:
#         raise ValueError("dechirp_spectra must be 2D: (num_chirps, n_bins)")
    
#     slow_len, n_bins = dechirp_spectra.shape

#     # Slow-time window
#     if window_slow is None or window_slow.lower() == "none":
#         w = np.ones(slow_len)
#     elif window_slow.lower() in ("hann", "hanning"):
#         w = np.hanning(slow_len)
#     elif window_slow.lower() == "hamming":
#         w = np.hamming(slow_len)
#     elif window_slow.lower() == "blackman":
#         w = np.blackman(slow_len)
#     else:
#         raise ValueError(f"Unknown window_slow='{window_slow}'")

#     x = dechirp_spectra * w[:, None]

#     # Doppler FFT across slow-time
#     if nfft_doppler is None:
#         nfft_doppler = 1 << int(np.ceil(np.log2(max(slow_len, 1))))

#     DD = np.fft.fftshift(np.fft.fft(x, n=nfft_doppler, axis=0), axes=0)
    
#     # Beat frequency axis
#     fb = np.fft.fftshift(np.fft.fftfreq(n_bins, d=1.0 / fs))

#     # peak_idx = np.argmax(np.abs(dechirp_spectra[0, :]))
#     # fb_peak = fb[peak_idx]
#     # delay_peak_s = fb_peak / k

#     # Doppler frequency axis
#     fd = np.fft.fftshift(np.fft.fftfreq(nfft_doppler, d=1.0 / PRF))

#     power_db = 10.0 * np.log10(np.abs(DD) ** 2 + 1e-12)
#     power_db = np.fliplr(power_db)  # flip beat frequency axis so negative is down, positive is up

#     # Convert beat frequency to range: r = (c * fb) / (k), where k is chirp rate (B/T)
#     delay_s = fb / k # + delay_peak_s  # Add delay of peak to center the plot around the main target response
#     delay_ms = delay_s * 1e3

#     # Apply delay mask
#     if positive_only:
#         d_mask = delay_ms >= 0
#     else:
#         d_mask = np.ones_like(delay_ms, dtype=bool)

#     if d_limit is not None:
#         d_mask = d_mask & (np.abs(delay_ms) <= d_limit)

#     d_plot = delay_ms[d_mask]
#     power_db = power_db[:, d_mask]

#     # Optional Doppler zoom
#     if fd_limit is not None:
#         fd_mask = (fd >= -fd_limit) & (fd <= fd_limit)
#         fd_plot = fd[fd_mask]
#         power_db = power_db[fd_mask, :]
#     else:
#         fd_plot = fd

#     print("dechirp_spectra:", dechirp_spectra.shape, flush=True)
#     print("dechirp_spectra GB:", dechirp_spectra.nbytes / 1e9, flush=True)

#     print("DD shape:", DD.shape, flush=True)
#     print("DD GB:", DD.nbytes / 1e9, flush=True)

#     plt.figure(figsize=(10, 6))
#     plt.pcolormesh(fd_plot, d_plot, power_db.T, shading="nearest", cmap="inferno", vmin=vmin, vmax=vmax)
#     plt.xlabel("Doppler Frequency [Hz]")
#     plt.ylabel("Delay [ms]")
#     plt.title(title)
#     plt.colorbar(label="Power [dB]")
#     plt.tight_layout()
#     if output_file:
#         # Expand the ~ if it exists
#         full_path = os.path.expanduser(output_file)
        
#         # Create the directory if it's missing
#         output_dir = os.path.dirname(full_path)
#         if output_dir and not os.path.exists(output_dir):
#             os.makedirs(output_dir)
            
#         plt.savefig(full_path, dpi=300)
#         plt.close()
#     else:
#         plt.show()


# def delay_doppler_process_window(iq_chunk, frame_idx, args, lfm_config, timestamps):
#     if args.method == "mf":
#         _, _, complex_response = lfm_matched_filtering(iq_chunk, lfm_config)

#         plot_delay_doppler_mf(
#             complex_response=complex_response,
#             lfm_config=lfm_config,
#             window_width=args.window_width,
#             title=f"{args.title} | {timestamps}",
#             output_file=f"{args.output}/frame_{frame_idx:04d}.png",
#             vmin=args.vmin,
#             vmax=args.vmax,
#             window_slow=args.slow_window,
#             nfft_doppler=args.nfft_doppler,
#             fd_limit=args.fd_limit,
#             d_limit=args.d_limit,
#             positive_only=args.d_positive_only,
#         )

#     else:
#         _, complex_spectra = dechirp_fft_complex(
#             received_signal=iq_chunk,
#             lfm_config=lfm_config,
#             window=None if args.dechirp_window == "none" else args.dechirp_window,
#         )

#         plot_delay_doppler_dechirp(
#             dechirp_spectra=complex_spectra,
#             lfm_config=lfm_config,
#             title=f"{args.title} | {timestamps}",
#             output_file=f"{args.output}/frame_{frame_idx:04d}.png",
#             vmin=args.vmin,
#             vmax=args.vmax,
#             window_slow=args.slow_window,
#             nfft_doppler=args.nfft_doppler,
#             fd_limit=args.fd_limit,
#             d_limit=args.d_limit,
#             positive_only=args.d_positive_only,
#         )


# Code graveyard

# def lsl_stream_one_stream(idf, stand_id=None, pol=None,
#                           tstart=None, tend=None,
#                           chunk_seconds=0.5):

#     pol_idx = pol_to_index(pol)
#     stream_index = 2 * (int(stand_id) - 1) + pol_idx

#     if tstart is not None:
#         idf.offset(tstart)

#     total_read = 0.0
#     target = None if tend is None else (tend - (tstart or 0.0))

#     while True:
#         if target is not None:
#             remaining = target - total_read
#             if remaining <= 0:
#                 break
#             read_len = min(chunk_seconds, remaining)
#         else:
#             read_len = chunk_seconds

#         try:
#             readT, _, data = idf.read(read_len)
#         except RuntimeError as e:
#             if "Invalid timetag skip" in str(e):
#                 idf.offset(0.006)
#                 continue
#             raise

#         if readT <= 0:
#             break

#         yield data[stream_index, :]

#         total_read += readT

# def compute_sweeps_streaming(
#     idf,
#     lfm_config,
#     args,
#     stand_id,
#     pol,
#     window_width,
#     chunk_seconds=0.5,
# ):
#     """
#     Stream IQ → matched filter → extract sweeps incrementally.

#     Returns:
#         sweeps: (num_sweeps, window_size)
#     """

#     fs = lfm_config.sample_rate
#     sweep_len = lfm_config.sweep_length
#     window_size = int(window_width * fs / 2) * 2

#     # Stream generator
#     stream = lsl_stream_one_stream(
#         idf,
#         stand_id=stand_id,
#         pol=pol,
#         tstart=args.tstart,
#         tend=args.tend,
#         chunk_seconds=chunk_seconds,
#     )

#     sweeps_list = []
#     carry = np.zeros(0, dtype=np.complex64)
#     sample_offset = 0

#     for chunk in stream:

#         # Add previous carry to current chunk for continuity
#         x = np.concatenate([carry, chunk])

#         # Matched filter chunk
#         _, _, resp = lfm_matched_filtering(x, lfm_config)

#         buffer = resp

#         # Extract sweeps from buffer, accounting for carry and offset
#         start = (-(sample_offset - len(carry))) % sweep_len

#         while start + window_size <= len(buffer):
#             sweeps_list.append(buffer[start:start + window_size])
#             start += sweep_len

#         sample_offset += len(chunk)

#         # keep just enough for next sweep continuity
#         carry = x[-sweep_len:]

#     if not sweeps_list:
#         raise ValueError("No sweeps extracted — check duration or parameters")

#     return np.array(sweeps_list)

# def plot_delay_doppler_from_sweeps(
#     sweeps,
#     lfm_config,
#     title,
#     output_file=None,
#     vmin=None,
#     vmax=None,
#     window_slow="hann",
#     nfft_doppler=None,
#     fd_limit=None,
#     d_limit=None,
#     positive_only=False,
# ):
#     slow_len, delay_len = sweeps.shape

#     # Slow-time window
#     if window_slow is None or window_slow.lower() == "none":
#         w = np.ones(slow_len)
#     elif window_slow.lower() in ("hann", "hanning"):
#         w = np.hanning(slow_len)
#     elif window_slow.lower() == "hamming":
#         w = np.hamming(slow_len)
#     elif window_slow.lower() == "blackman":
#         w = np.blackman(slow_len)
#     else:
#         raise ValueError(f"Unknown window_slow='{window_slow}'")

#     x = sweeps * w[:, None]

#     PRF = lfm_config.sweep_frequency

#     if nfft_doppler is None:
#         nfft_doppler = 1 << int(np.ceil(np.log2(max(slow_len, 1))))

#     DD = np.fft.fftshift(np.fft.fft(x, n=nfft_doppler, axis=0), axes=0)
#     fd = np.fft.fftshift(np.fft.fftfreq(nfft_doppler, d=1.0 / PRF))

#     power_db = 10 * np.log10(np.abs(DD) ** 2 + 1e-12)

#     fs = lfm_config.sample_rate
#     delay_s = ((np.arange(delay_len) - delay_len // 2) / fs)
#     delay_ms = delay_s * 1e3

#     # Optional zooms
#     if fd_limit is not None:
#         fd_mask = (fd >= -fd_limit) & (fd <= fd_limit)
#         fd_plot = fd[fd_mask]
#         power_db = power_db[fd_mask, :]
#     else:
#         fd_plot = fd

#     if positive_only:
#         d_mask = delay_ms >= 0
#     else:
#         d_mask = np.ones_like(delay_ms, dtype=bool)

#     if d_limit is not None:
#         d_mask = (delay_ms >= 0) & (delay_ms <= d_limit)

#     d_plot = delay_ms[d_mask]
#     power_db = power_db[:, d_mask]

#     plt.figure(figsize=(10, 6))
#     plt.pcolormesh(fd_plot, d_plot, power_db.T, shading="nearest",
#                    cmap="inferno", vmin=vmin, vmax=vmax)
#     plt.xlabel("Doppler Frequency [Hz]")
#     plt.ylabel("Delay [ms]")
#     plt.title(title)
#     plt.colorbar(label="Power [dB]")
#     plt.tight_layout()
#     if output_file:
#         # Expand the ~ if it exists
#         full_path = os.path.expanduser(output_file)
        
#         # Create the directory if it's missing
#         output_dir = os.path.dirname(full_path)
#         if output_dir and not os.path.exists(output_dir):
#             os.makedirs(output_dir)
            
#         plt.savefig(full_path, dpi=300)
#         plt.close()
#     else:
#         plt.show()
# lfm_utils.py

import dataclasses

import numpy as np
from scipy.signal import oaconvolve, correlation_lags
from scipy.signal.windows import chebwin
import soundfile as sf


@dataclasses.dataclass
class LFMWaveform:
    bandwidth: float
    sweep_frequency: float
    sample_rate: float
    reference_gate_frequency: float = None
    reference_gate_duty: float = 0.5
    reference_gate_phase: float = 0.0

    @property
    def waveform(self):
        T = 1 / self.sweep_frequency  # Sweep duration (s)
        t = np.arange(0, T, 1 / self.sample_rate)  # Time vector
        kf = self.bandwidth / T  # Chirp rate (Hz/s)

        # Complex LFM sweep centered at 0 Hz
        lfm = np.exp(1j * 2 * np.pi * ((-self.bandwidth / 2) * t + (kf / 2) * t ** 2))
        if self.reference_gate_frequency is not None:
            if self.reference_gate_frequency <= 0:
                raise ValueError("reference_gate_frequency must be > 0")
            if not 0 < self.reference_gate_duty <= 1:
                raise ValueError("reference_gate_duty must be in the interval (0, 1]")

            gate_period = 1.0 / self.reference_gate_frequency
            gate_time = (t - self.reference_gate_phase) % gate_period
            gate = gate_time < (self.reference_gate_duty * gate_period)
            lfm = lfm * gate.astype(lfm.dtype)
        return lfm

    @property
    def sweep_length(self):
        return int(self.sample_rate / self.sweep_frequency)


def load_iq_audio(filename, start: int = 0, stop: int = None):
    data, samplerate = sf.read(filename, start=start, stop=stop)
    iq = data[:, 0] + 1j * data[:, 1]
    return iq, samplerate


def reference_gate_frequency_from_args(args):
    gate_frequency = getattr(args, "reference_gate_frequency", None)
    gate_period = getattr(args, "reference_gate_period", None)

    if gate_frequency is not None and gate_period is not None:
        raise ValueError("Use only one of --reference-gate-frequency or --reference-gate-period")
    if gate_period is not None:
        if gate_period <= 0:
            raise ValueError("--reference-gate-period must be > 0")
        return 1.0 / gate_period
    return gate_frequency


def window_from_arg(len, arg):
    if arg is None or arg.lower() == "none":
        return np.ones(len)
    arg = arg.lower()
    if arg == "bartlett":
        return np.bartlett(len)
    if arg == "blackman":
        return np.blackman(len)
    if arg == "hann" or arg == "hanning":
        return np.hanning(len)
    if arg == "hamming":
        return np.hamming(len)
    if arg in ("cheb", "chebwin", "dolph-chebyshev", "cheb100"):
        return chebwin(len, at=100)
    if arg in ("cheb60", "cheb80", "cheb120"):
        return chebwin(len, at=float(arg.removeprefix("cheb")))
    raise ValueError(f"Unknown window type: {arg}")


def lfm_matched_filtering(iq, lfm_in: LFMWaveform):
    # Complex LFM sweep centered at 0 Hz
    lfm_waveform = lfm_in.waveform

    iq = np.asarray(iq)
    single_stream = iq.ndim == 1
    if single_stream:
        iq = iq[None, :]
    elif iq.ndim != 2:
        raise ValueError("iq must be 1D or 2D (n_streams, n_samples)")

    filter_rows = []
    for row in iq:
        filter_rows.append(oaconvolve(row, np.conj(lfm_waveform[::-1]), mode='valid'))
    filter_output = np.asarray(filter_rows)
    lags = correlation_lags(iq.shape[1], len(lfm_waveform), mode='valid')
    magnitude_response = 10 * np.log10(np.sum(np.abs(filter_output) ** 2, axis=0) + 1e-12)

    if single_stream:
        filter_output = filter_output[0]
    return magnitude_response, lags, filter_output


def dechirp_fft_complex(received_signal: np.ndarray,
                        lfm_config: LFMWaveform,
                        window: str = "hamming",
                        start_offset_samples: int = 0) -> np.ndarray:
    """
    Dechirp each chirp-length segment, then FFT it.
    Returns complex FFT spectra: shape (num_chirps, chirp_len)
    """
    chirp_len = len(lfm_config.waveform)
    if start_offset_samples < 0:
        raise ValueError("start_offset_samples must be >= 0")

    received_signal = np.asarray(received_signal)
    single_stream = received_signal.ndim == 1
    if single_stream:
        received_signal = received_signal[None, :]
    elif received_signal.ndim != 2:
        raise ValueError("received_signal must be 1D or 2D (n_streams, n_samples)")

    received_signal = received_signal[:, start_offset_samples:]
    num_streams = received_signal.shape[0]
    num_chirps = received_signal.shape[1] // chirp_len

    reference_chirp = lfm_config.waveform.astype(received_signal.dtype)

    w = window_from_arg(chirp_len, window).astype(np.float32, copy=False)
    coherent_gain = np.mean(w)
    if coherent_gain == 0:
        raise ValueError(f"Window '{window}' has zero coherent gain")

    out = np.zeros((num_streams, num_chirps, chirp_len), dtype=np.complex64)
    mag_out = np.zeros((num_chirps, chirp_len), dtype=np.float32)

    for i in range(num_chirps):
        seg = received_signal[:, i * chirp_len:(i + 1) * chirp_len]
        beat = seg * np.conj(reference_chirp)
        spectra = np.fft.fftshift(np.fft.fft(beat * w, axis=1) / coherent_gain, axes=1)
        out[:, i, :] = spectra
        mag_out[i, :] = np.sum(np.abs(spectra) ** 2, axis=0)

    if single_stream:
        return mag_out, out[0]
    return mag_out, out


def dechirp_delay_axis_ms(lfm_config: LFMWaveform,
                          n_bins: int = None,
                          positive_delay_axis: bool = True,
                          bandlimit: bool = True):
    if n_bins is None:
        n_bins = len(lfm_config.waveform)
    fs = lfm_config.sample_rate
    prf = lfm_config.sweep_frequency
    T = 1.0 / prf
    k = lfm_config.bandwidth / T
    if k == 0:
        raise ValueError("Cannot convert dechirp beat frequency to delay when bandwidth is zero")

    fb = np.fft.fftshift(np.fft.fftfreq(n_bins, d=1.0 / fs))
    bin_order = np.arange(n_bins)
    if bandlimit:
        df = fs / n_bins
        half_bandwidth = abs(lfm_config.bandwidth) / 2.0
        nyquist = fs / 2.0
        if half_bandwidth < nyquist:
            band_mask = np.abs(fb) <= (half_bandwidth + 0.5 * df)
            fb = fb[band_mask]
            bin_order = bin_order[band_mask]
    delay_ms = -fb / k * 1e3
    if positive_delay_axis:
        delay_ms = np.mod(delay_ms, T * 1e3)
    delay_order = np.argsort(delay_ms)
    return delay_ms[delay_order], bin_order[delay_order]


def auto_delay_bounds_from_iq(received_signal: np.ndarray,
                              lfm_config: LFMWaveform,
                              window: str = "hann",
                              start_offset_samples: int = 0,
                              preview_sweeps: int = 32,
                              half_width_ms: float = 1.5):
    if preview_sweeps < 1:
        raise ValueError("preview_sweeps must be >= 1")
    if half_width_ms <= 0:
        raise ValueError("half_width_ms must be > 0")

    chirp_len = len(lfm_config.waveform)
    received_signal = np.asarray(received_signal)
    single_stream = received_signal.ndim == 1
    if single_stream:
        received_signal = received_signal[None, :]
    elif received_signal.ndim != 2:
        raise ValueError("received_signal must be 1D or 2D (n_streams, n_samples)")

    available_samples = received_signal.shape[-1] - int(start_offset_samples)
    available_sweeps = available_samples // chirp_len
    n_preview = int(min(preview_sweeps, available_sweeps))
    if n_preview < 1:
        raise ValueError("Not enough samples to auto-estimate delay bounds")

    preview = received_signal[:, start_offset_samples:start_offset_samples + n_preview * chirp_len].copy()
    preview[~np.isfinite(preview)] = 0.0
    power, _ = dechirp_fft_complex(
        preview[0] if single_stream else preview,
        lfm_config,
        window=window,
        start_offset_samples=0,
    )
    delay_ms, delay_order = dechirp_delay_axis_ms(lfm_config, n_bins=power.shape[-1])
    power = power[:, delay_order]
    if not np.any(np.isfinite(power)):
        raise ValueError("Could not auto-estimate delay bounds because preview power is all non-finite")

    _, delay_idx = np.unravel_index(np.nanargmax(power), power.shape)
    strongest_delay_ms = float(delay_ms[delay_idx])
    d_min = max(float(np.min(delay_ms)), strongest_delay_ms - half_width_ms)
    d_max = min(float(np.max(delay_ms)), strongest_delay_ms + half_width_ms)
    print(
        f"No --d-min/--d-max provided; auto-centering on strongest delay bin "
        f"{strongest_delay_ms:.3f} ms using first {n_preview} sweep(s), "
        f"range {d_min:.3f} to {d_max:.3f} ms",
        flush=True,
    )
    return d_min, d_max, strongest_delay_ms


def dechirp_fft_delay_band_complex(received_signal: np.ndarray,
                                   lfm_config: LFMWaveform,
                                   d_min: float,
                                   d_max: float,
                                   window: str = "hann",
                                   start_offset_samples: int = 0):
    chirp_len = len(lfm_config.waveform)
    if start_offset_samples < 0:
        raise ValueError("start_offset_samples must be >= 0")
    if d_min is None or d_max is None:
        raise ValueError("Both d_min and d_max are required")

    received_signal = np.asarray(received_signal)
    single_stream = received_signal.ndim == 1
    if single_stream:
        received_signal = received_signal[None, :]
    elif received_signal.ndim != 2:
        raise ValueError("received_signal must be 1D or 2D (n_streams, n_samples)")

    received_signal = received_signal[:, start_offset_samples:]
    num_streams = received_signal.shape[0]
    num_chirps = received_signal.shape[1] // chirp_len

    delay_ms, delay_order = dechirp_delay_axis_ms(lfm_config, n_bins=chirp_len)
    d_mask = (delay_ms >= d_min) & (delay_ms <= d_max)
    selected_bins = delay_order[d_mask]
    selected_delay_ms = delay_ms[d_mask]
    if selected_bins.size == 0:
        raise ValueError(
            "No delay bins selected; check --d-min/--d-max. "
            f"Available delay range is {delay_ms.min():.3f} to {delay_ms.max():.3f} ms."
        )

    reference_chirp = lfm_config.waveform.astype(received_signal.dtype)
    w = window_from_arg(chirp_len, window).astype(np.float32, copy=False)
    coherent_gain = np.mean(w)
    if coherent_gain == 0:
        raise ValueError(f"Window '{window}' has zero coherent gain")

    out = np.empty((num_streams, num_chirps, selected_bins.size), dtype=np.complex64)
    for i in range(num_chirps):
        seg = received_signal[:, i * chirp_len:(i + 1) * chirp_len]
        beat = seg * np.conj(reference_chirp)
        spectrum = np.fft.fftshift(np.fft.fft(beat * w, axis=1) / coherent_gain, axes=1)
        out[:, i, :] = spectrum[:, selected_bins]

    if single_stream:
        return selected_delay_ms, out[0]
    return selected_delay_ms, out


# C = 299_792_458.0      # Speed of light (m/s)

# def dechirp(received_signal: np.ndarray,
#             reference_chirp: np.ndarray) -> np.ndarray:
#     chirp_len = len(reference_chirp)
#     num_chirps = len(received_signal) // chirp_len
#     result = []
#     window = np.hamming(len(reference_chirp))

#     for i in range(num_chirps):
#         segment = received_signal[i * chirp_len : (i + 1) * chirp_len]
#         if len(segment) < chirp_len:
#             break

#         beat = segment * np.conj(reference_chirp)

#         # beat *=window
#         fft_out = np.fft.fftshift(np.fft.fft(beat))
#         result.append(np.abs(fft_out)**2)

#     return np.array(result)

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

    filter_output = oaconvolve(iq, np.conj(lfm_waveform[::-1]), mode='valid')
    lags = correlation_lags(len(iq), len(lfm_waveform), mode='valid')
    magnitude_response = 10 * np.log10(np.abs(filter_output) ** 2)

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

    received_signal = received_signal[start_offset_samples:]
    num_chirps = len(received_signal) // chirp_len

    reference_chirp = lfm_config.waveform.astype(received_signal.dtype)

    w = window_from_arg(chirp_len, window).astype(np.float32, copy=False)
    coherent_gain = np.mean(w)
    if coherent_gain == 0:
        raise ValueError(f"Window '{window}' has zero coherent gain")

    out = np.zeros((num_chirps, chirp_len), dtype=np.complex64)
    mag_out = np.zeros((num_chirps, chirp_len), dtype=np.float32)

    for i in range(num_chirps):
        seg = received_signal[i * chirp_len:(i + 1) * chirp_len]
        beat = seg * np.conj(reference_chirp)
        beat = beat * w
        out[i, :] = np.fft.fftshift(np.fft.fft(beat) / coherent_gain)
        mag_out[i, :] = np.abs(out[i, :])**2

    return mag_out, out


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


# def bistatic_model(time_offset, h_km=500, v_kms=7.5, fc=29e6):
#     x_km = v_kms * time_offset
#     r_slant_km = np.sqrt(h_km**2 + x_km**2)
#     v_radial_kms = v_kms * (x_km / r_slant_km)
#     doppler_hz = -(v_radial_kms * 1000 / C) * fc
#     return r_slant_km, doppler_hz


# def direct_path_delay(time_offset, h_km=500, v_kms=7.5):
#     x_km = v_kms * time_offset
#     r_slant_km = np.sqrt(h_km**2 + x_km**2)
#     return (r_slant_km * 1000.0) / C


# def fractional_delay_shift(x, shift_samples):
#     """
#     Shift x by shift_samples (can be fractional).
#     Positive shift_samples delays the signal.
#     """
#     n = len(x)
#     freqs = np.fft.fftfreq(n, d=1.0)
#     phase = np.exp(-1j * 2.0 * np.pi * freqs * shift_samples)
#     return np.fft.ifft(np.fft.fft(x) * phase).astype(x.dtype)


# def motion_compensate_iq_per_sweep(
#     iq,
#     fs,
#     fc,
#     sweep_frequency,
#     delay_model,
#     time_offset_start,
# ):
#     chirp_len = int(round(fs / sweep_frequency))
#     num_sweeps = len(iq) // chirp_len

#     out = np.zeros(num_sweeps * chirp_len, dtype=np.complex64)

#     # Reference = first sweep center
#     t_ref = time_offset_start + 0.5 / sweep_frequency
#     tau_ref = delay_model(t_ref)

#     for i in range(num_sweeps):
#         start = i * chirp_len
#         end = start + chirp_len
#         seg = iq[start:end].astype(np.complex128)

#         # center-of-sweep time
#         t_i = time_offset_start + (i + 0.5) / sweep_frequency
#         tau_i = delay_model(t_i)

#         delta_tau = tau_i - tau_ref

#         # 1) remove delay walk
#         shift_samples = -delta_tau * fs
#         seg = fractional_delay_shift(seg, shift_samples)

#         # 2) remove direct-path carrier phase
#         seg *= np.exp(1j * 2.0 * np.pi * fc * delta_tau)

#         out[start:end] = seg.astype(np.complex64)

#     return out

# C0 = 299_792_458.0

# def bistatic_delay_doppler(time_offset, fc, h_km=500.0, v_kms=7.5):
#     """
#     Vectorized version.
#     time_offset can be scalar or numpy array, in seconds.
#     Returns:
#         tau_s : one-way direct-path delay [s]
#         fd_hz : direct-path Doppler [Hz]
#     """
#     time_offset = np.asarray(time_offset, dtype=np.float64)

#     x_km = v_kms * time_offset
#     r_slant_km = np.sqrt(h_km**2 + x_km**2)
#     v_radial_kms = v_kms * (x_km / r_slant_km)

#     tau_s = (r_slant_km * 1000.0) / C0
#     fd_hz = -(v_radial_kms * 1000.0 / C0) * fc
#     return tau_s, fd_hz


# def interp_complex_uniform(x, sample_pos):
#     """
#     Complex linear interpolation of x at fractional sample positions sample_pos.
#     sample_pos is in units of samples, not seconds.
#     """
#     n = np.arange(len(x), dtype=np.float64)
#     real = np.interp(sample_pos, n, x.real, left=0.0, right=0.0)
#     imag = np.interp(sample_pos, n, x.imag, left=0.0, right=0.0)
#     return real + 1j * imag


# def motion_compensate_iq_per_sample(
#     iq,
#     fs,
#     fc,
#     start_time_offset,
#     h_km,
#     v_kms,
#     tau_ref_mode="first",
# ):
#     """
#     Motion compensation on raw IQ, per sample.

#     iq                : complex IQ array
#     fs                : sample rate [Hz]
#     fc                : carrier used only to compute Doppler from geometry
#     start_time_offset : model time corresponding to iq[0], in seconds
#     h_km, v_kms       : geometry params for your current simplified model
#     tau_ref_mode      : 'first', 'min', or 'mean'
#     """
#     iq = np.asarray(iq)
#     N = len(iq)

#     # Absolute model time for each output sample
#     t_out = start_time_offset + np.arange(N, dtype=np.float64) / fs

#     # Predicted direct-path delay and Doppler at each output sample time
#     tau_s, fd_hz = bistatic_delay_doppler(t_out, fc=fc, h_km=h_km, v_kms=v_kms)

#     # Choose only a DELAY reference. This just defines where "zero excess delay" is.
#     if tau_ref_mode == "first":
#         tau_ref = tau_s[0]
#     elif tau_ref_mode == "min":
#         tau_ref = np.min(tau_s)
#     elif tau_ref_mode == "mean":
#         tau_ref = np.mean(tau_s)
#     else:
#         raise ValueError("tau_ref_mode must be 'first', 'min', or 'mean'")

#     delta_tau_s = tau_s - tau_ref

#     # Time-warp: output sample at t_out should come from input at t_out + delta_tau
#     t_in = t_out + delta_tau_s
#     sample_pos_in = (t_in - t_out[0]) * fs

#     # Interpolate raw IQ at those input positions
#     iq_warp = interp_complex_uniform(iq, sample_pos_in)

#     # Build continuous direct-path Doppler phase history
#     # This matches your simulator much better than exp(j 2π fc tau),
#     # because your raw IQ contains Doppler explicitly, not passband carrier phase.
#     phi = 2.0 * np.pi * np.cumsum(fd_hz) / fs
#     phi -= phi[0]

#     # Interpolate phase onto same warped input positions
#     phi_in = np.interp(sample_pos_in, np.arange(N, dtype=np.float64), phi,
#                        left=phi[0], right=phi[-1])

#     # Remove direct-path Doppler phase
#     iq_comp = iq_warp * np.exp(-1j * phi_in)

#     return iq_comp.astype(np.complex64), {
#         "tau_s": tau_s,
#         "fd_hz": fd_hz,
#         "delta_tau_s": delta_tau_s,
#         "phi_rad": phi,
#         "sample_pos_in": sample_pos_in,
#     }

# def latlon_to_ecef(lat_deg, lon_deg, alt_m=0.0):
#     R_earth = 6371e3  # meters

#     lat = np.deg2rad(lat_deg)
#     lon = np.deg2rad(lon_deg)

#     x = (R_earth + alt_m) * np.cos(lat) * np.cos(lon)
#     y = (R_earth + alt_m) * np.cos(lat) * np.sin(lon)
#     z = (R_earth + alt_m) * np.sin(lat)

#     return np.array([x, y, z])

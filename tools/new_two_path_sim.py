import numpy as np
import soundfile as sf
from lfm_utils import LFMWaveform

# Can you make one with a simulated 2-ray channel model where the 
# rays are of constant but unequal amplitude and the delay varies 
# with a triangle wave around a constant delay so that the the 
# Doppler would be constant up or down? in each direction?   
# Think of a target moving at constant velocity towards you and 
# then instantly turning around.

c = 299_792_458.0      # Speed of light (m/s)

# -------------- #
# LFM parameters #
# -------------- #
fc = 29.0e6              # Carrier frequency (Hz)
B = 100.0e3              # bandwidth
fsweep = 2.0             # sweep rate
fs = 100.0e3             # sample rate (Hz)
t_total = 7200.0         # total duration of signal (s)
noise_floor = 0.01       # Relative noise floor (0.01 ~ -20 dB relative to max signal)

num_sweeps = int(t_total * fsweep)
rng = np.random.default_rng(0)

# ------------ #
# Generate LFM #
# ------------ #
lfm = LFMWaveform(bandwidth=-B, sweep_frequency=fsweep, sample_rate=fs)
chirp = lfm.waveform.astype(np.complex64)
chirp_len = len(chirp)

# Allocate buffer (approximate length)
rx_len = num_sweeps * chirp_len + int(fs * 0.1) # extra buffer for delays
rx_signal = np.zeros(rx_len, dtype=np.complex64)

tx_signal = np.tile(chirp, num_sweeps)

# ---------------- #
# Channel Settings #
# ---------------- #

# Path 1 (fixed)
path_1_tau = 0.0  # 0 delay for direct path
path_1_gain = 1.0

# Path 2: triangle-wave delay around a constant mean delay.
path_2_tau_0 = path_1_tau + 0.005  # +5 ms extra delay
path_2_gain = 0.5

# Desired Doppler
doppler_mag = 0.90      # Hz
doppler_period = 7000.0  # seconds
doppler_half_period = doppler_period / 2.0
delay_lfo = "smooth_triangle"      # "triangle", "smooth_triangle", or "sine"
smooth_turn_time = 120.0           # seconds, only used by "smooth_triangle"

# For a delayed passband signal, baseband phase is exp(-j 2*pi*fc*tau(t)),
# so fD(t) = -(fc * d tau/dt).
delay_rate_mag = doppler_mag / fc
triangle_delay_amplitude = delay_rate_mag * doppler_period / 4.0
sine_delay_amplitude = doppler_mag * doppler_period / (2.0 * np.pi * fc)


def build_smooth_triangle_delay_table(num_points=200_000):
    """Build one period of a triangle-like delay with rounded Doppler reversals."""
    lfo_t = np.linspace(0.0, doppler_period, num_points, endpoint=False)
    raw_fd = np.where(lfo_t < doppler_half_period, doppler_mag, -doppler_mag)

    kernel_len = max(3, int(round(num_points * smooth_turn_time / doppler_period)))
    if kernel_len % 2 == 0:
        kernel_len += 1
    kernel = np.hanning(kernel_len)
    kernel /= np.sum(kernel)

    pad = kernel_len // 2
    padded_fd = np.concatenate((raw_fd[-pad:], raw_fd, raw_fd[:pad]))
    smooth_fd = np.convolve(padded_fd, kernel, mode="valid")
    smooth_fd -= np.mean(smooth_fd)

    dt = doppler_period / num_points
    delay_offset = -np.cumsum(smooth_fd) * dt / fc
    delay_offset -= np.mean(delay_offset)

    return (
        np.concatenate((lfo_t, [doppler_period])),
        np.concatenate((delay_offset, [delay_offset[0]])),
    )


smooth_triangle_t, smooth_triangle_delay = build_smooth_triangle_delay_table()


def triangle_delay(t):
    """Triangle-wave delay with +fD on the first half-cycle."""
    phase_t = np.mod(t, doppler_period)
    tau = np.empty_like(phase_t, dtype=np.float64)

    first_half = phase_t < doppler_half_period
    tau[first_half] = path_2_tau_0 + triangle_delay_amplitude - delay_rate_mag * phase_t[first_half]
    tau[~first_half] = (
        path_2_tau_0
        - triangle_delay_amplitude
        + delay_rate_mag * (phase_t[~first_half] - doppler_half_period)
    )
    return tau


def sine_delay(t):
    """Sinusoidal delay with smooth Doppler and peak |fD| = doppler_mag."""
    omega = 2.0 * np.pi / doppler_period
    return path_2_tau_0 - sine_delay_amplitude * np.sin(omega * t)


def smooth_triangle_delay_func(t):
    """Triangle-like delay with rounded corners to avoid Doppler discontinuities."""
    phase_t = np.mod(t, doppler_period)
    return path_2_tau_0 + np.interp(
        phase_t,
        smooth_triangle_t,
        smooth_triangle_delay,
    )


def path_2_delay(t):
    if delay_lfo == "triangle":
        return triangle_delay(t)
    if delay_lfo == "smooth_triangle":
        return smooth_triangle_delay_func(t)
    if delay_lfo == "sine":
        return sine_delay(t)
    raise ValueError(f"Unknown delay_lfo='{delay_lfo}'")


def selected_delay_amplitude():
    if delay_lfo == "triangle":
        return triangle_delay_amplitude
    if delay_lfo == "smooth_triangle":
        return np.max(np.abs(smooth_triangle_delay))
    if delay_lfo == "sine":
        return sine_delay_amplitude
    raise ValueError(f"Unknown delay_lfo='{delay_lfo}'")


def fractional_delay_lookup(x, sample_pos):
    """Linear interpolation of a complex sequence at fractional sample positions."""
    j0 = np.floor(sample_pos).astype(np.int64)
    frac = sample_pos - j0
    valid = (j0 >= 0) & (j0 + 1 < len(x))

    y = np.zeros(sample_pos.shape, dtype=np.complex64)
    y[valid] = (
        (1.0 - frac[valid]) * x[j0[valid]]
        + frac[valid] * x[j0[valid] + 1]
    )
    return y

# ---------------- #
# Simulate Channel #
# ---------------- #

path_1_delay_samples = int(np.round(path_1_tau * fs))
path_1_start = path_1_delay_samples
path_1_end = min(path_1_start + len(tx_signal), len(rx_signal))
rx_signal[path_1_start:path_1_end] += path_1_gain * tx_signal[:path_1_end - path_1_start]

chunk_size = 1_000_000
for start in range(0, len(rx_signal), chunk_size):
    end = min(start + chunk_size, len(rx_signal))
    out_idx = np.arange(start, end, dtype=np.float64)
    t_out = out_idx / fs

    tau = path_2_delay(t_out)
    sample_pos = (t_out - tau) * fs
    delayed = fractional_delay_lookup(tx_signal, sample_pos)
    carrier_phase = np.exp(-1j * 2.0 * np.pi * fc * tau).astype(np.complex64)

    rx_signal[start:end] += path_2_gain * delayed * carrier_phase

# Add noise
for start in range(0, len(rx_signal), chunk_size):
    end = min(start + chunk_size, len(rx_signal))
    noise = noise_floor * (
        rng.standard_normal(end - start) + 1j * rng.standard_normal(end - start)
    )
    rx_signal[start:end] += noise.astype(np.complex64)

# Normalize and Export
mx = np.max(np.abs(rx_signal))
if mx > 0: rx_signal /= mx

output_file = "/Users/owenohlson/Documents/Academics/UVic/ionospheric_sounding/data/new_two_path_sim.wav"
# reference_output_file = "/Users/owenohlson/Documents/Academics/UVic/ionospheric_sounding/data/new_two_path_sim_reference.wav"
sf.write(output_file, np.column_stack((rx_signal.real, rx_signal.imag)), int(fs))
# sf.write(reference_output_file, np.column_stack((tx_signal.real, tx_signal.imag)), int(fs))
print(
    f"Simulated two-path channel with {num_sweeps} sweeps, "
    f"{delay_lfo} delay {path_2_tau_0 * 1e3:.3f} ms +/- {selected_delay_amplitude() * 1e6:.3f} us, "
    f"and Doppler shift of +/-{doppler_mag} Hz over a {doppler_period}s period."
)

import numpy as np
import soundfile as sf
from lfm_utils import LFMWaveform # , bistatic_model

c = 299_792_458.0      # Speed of light (m/s)

# ---------------
# LFM parameters
# ---------------
fc = 29.0e6              # Carrier frequency (Hz)
B = 100.0e3              # bandwidth
fsweep = 2.0             # sweep rate
fs = 100.0e3             # sample rate (Hz)
num_sweeps = 240          # ~120 seconds of data
noise_floor = 0.5       # Relative noise floor (0.01 ~ -20 dB relative to max signal)

# -------------------------
# Cubesat Orbit Constants
# -------------------------
h_km = 500.0             # Altitude of satellite (km)
v_kms = 0.0              # Velocity (km/s)
start_time_offset = -30.0  # Start simulation 30 seconds before closest approach

# ------------ #
# Generate LFM #
# ------------ #
lfm = LFMWaveform(bandwidth=-B, sweep_frequency=fsweep, sample_rate=fs)
chirp = lfm.waveform.astype(np.complex64)
chirp_len = len(chirp)

# Allocate buffer (approximate length)
rx_len = num_sweeps * chirp_len + int(fs * 0.1) # extra buffer for delays
rx_signal = np.zeros(rx_len, dtype=np.complex64)

# ---------------- #
# Channel Settings #
# ---------------- #

# Direct path (fixed)
tau_direct = 0.0003  # 0.3 ms delay (~90 km)
direct_atten = 1.0

# Multipath baseline
tau_multipath_0 = tau_direct + 0.0005  # +0.5 ms extra delay

# Time axis for entire signal
t_global = np.arange(rx_len) / fs

# ----------------------------- #
# Define time-varying functions #
# ----------------------------- #

# Sinusoidal delay variation (ionospheric motion)
# delay_variation = 0.0025 * np.sin(2 * np.pi * 0.000035 * t_global)
# ±2.5 ms variation at 0.000035 Hz

# Linear delay variation to produce 0.25 Hz Doppler shift
# delay_variation = 0.00025 * t_global  # +0.25 ms over 100 seconds

# Precise linear delay slope for a -0.25 Hz Doppler shift
# dtau/dt = -f_d / f_c
doppler_target = -0.25
delay_slope = -doppler_target / fc  # ~8.6207e-9 seconds of delay per second
delay_variation = delay_slope * t_global

# Amplitude fading (slow fading)
amplitude_variation = 0.25 + 0.20 * np.sin(2 * np.pi * 0.11 * t_global)

# Optional: add randomness (scintillation-like)
# amplitude_variation *= (1 + 0.2 * np.random.randn(len(t_global)))

# Final multipath delay
tau_multipath = tau_multipath_0 + delay_variation

# ---------------- #
# Simulate Channel #
# ---------------- #

for i in range(num_sweeps):

    sweep_base = i * chirp_len

    # ---------- #
    # Direct path
    # ---------- #
    tau_samples = int(np.round(tau_direct * fs))
    start = sweep_base + tau_samples
    end = start + chirp_len

    if end <= len(rx_signal):
        rx_signal[start:end] += direct_atten * chirp

    # ---------------------- #
    # Multipath (time-varying)
    # ---------------------- #

    for n in range(chirp_len):
        idx = sweep_base + n

        if idx >= len(rx_signal):
            break

        # Current time
        t = idx / fs

        # Time-varying delay
        tau = tau_multipath[idx]
        delay_samples = int(np.round(tau * fs))

        target_idx = idx + delay_samples

        # if target_idx < len(rx_signal):
        #     amp =  amplitude_variation[idx]
        #     rx_signal[target_idx] += amp * chirp[n]

        if target_idx < len(rx_signal):
            amp = amplitude_variation[idx]
            
            # Apply the continuous phase rotation for the Doppler shift
            phase_rotation = np.exp(1j * 2 * np.pi * doppler_target * t)
            
            rx_signal[target_idx] += amp * chirp[n] * phase_rotation

# Add noise
noise = noise_floor * (np.random.randn(len(rx_signal)) + 1j * np.random.randn(len(rx_signal)))
rx_signal += noise.astype(np.complex64)

# Normalize and Export
mx = np.max(np.abs(rx_signal))
if mx > 0: rx_signal /= mx

output_file = "/Users/owenohlson/Documents/Academics/UVic/ionospheric_sounding/data/two_path_sim.wav"
sf.write(output_file, np.column_stack((rx_signal.real, rx_signal.imag)), int(fs))

# closest_approach_delay_ms = (bistatic_model(0, h_km, v_kms)[0] * 1000/c) * 1e3

print(f"Simulated cubesat pass starting at {start_time_offset}s offset.")
# print(f"Direct path delay at closest approach: {closest_approach_delay_ms:.2f} ms")
# print(f"Max Doppler predicted: {bistatic_model(start_time_offset, h_km, v_kms)[1]:.2f} Hz")
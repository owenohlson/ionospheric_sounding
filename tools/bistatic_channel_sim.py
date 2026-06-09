import numpy as np
import soundfile as sf
from lfm_utils import LFMWaveform, bistatic_model

c = 299_792_458.0      # Speed of light (m/s)

# ---------------
# LFM parameters
# ---------------
fc = 29.0e6              # Carrier frequency (Hz)
B = 100.0e3              # bandwidth
fsweep = 1.0             # sweep rate
fs = 100.0e3             # sample rate (Hz)
num_sweeps = 120         # ~240 seconds of data
noise_floor = 0.01       # Relative noise floor (0.01 ~ -20 dB relative to max signal)

# -------------------------
# Cubesat Orbit Constants
# -------------------------
h_km = 500.0             # Altitude of satellite (km)
v_kms = 10.0              # Velocity (km/s)
start_time_offset = -60.0  # Start simulation 120 seconds before closest approach

# Signal Strengths
direct_atten = 1.0     # The main signal from cubesat
iono_atten = 0.05      # A faint ionospheric multipath/reflection

# ------------ #
# Generate LFM #
# ------------ #
lfm = LFMWaveform(bandwidth=B, sweep_frequency=fsweep, sample_rate=fs)
chirp = lfm.waveform.astype(np.complex64)
chirp_len = len(chirp)

# Allocate buffer (approximate length)
rx_len = num_sweeps * chirp_len + int(fs * 0.1) # extra buffer for delays
rx_signal = np.zeros(rx_len, dtype=np.complex64)

# ---------------- #
# Simulate Channel #
# ---------------- #
for i in range(num_sweeps):
    # Time at the start of this specific chirp
    t_abs = (i) / fsweep 
    current_time_offset = start_time_offset + t_abs
    
    # Get Satellite Position & Doppler
    r_sat_km, fd_sat = bistatic_model(current_time_offset, h_km, v_kms, fc)
    
    # 1. Direct Path (Satellite to Ground)
    # Delay is one-way: tau = R / c
    tau_s = (r_sat_km * 1000.0) / c
    tau_samples = int(np.round(tau_s * fs))
    print(f"Chirp {i+1}/{num_sweeps}: Time offset {current_time_offset:.2f} s, Slant range {r_sat_km:.2f} km, Doppler {fd_sat:.2f} Hz, Direct path delay {tau_s*1e3:.2f} ms")
    
    sweep_start = i * chirp_len + tau_samples
    sweep_end = sweep_start + chirp_len
    
    if sweep_end <= len(rx_signal):
        n = np.arange(chirp_len)
        t_samples = (sweep_start + n) / fs
        # Apply the platform doppler
        doppler_phasor = np.exp(1j * 2.0 * np.pi * fd_sat * t_samples)
        rx_signal[sweep_start:sweep_end] += direct_atten * chirp * doppler_phasor

    # 2. Simulated Ionospheric Multipath (Scattering)
    # Let's say it's delayed by an extra 0.5ms (approx 150km extra path)
    tau_iono_s = tau_s + 0.0005 
    tau_iono_samples = int(np.round(tau_iono_s * fs))
    
    iono_start = i * chirp_len + tau_iono_samples
    iono_end = iono_start + chirp_len
    
    if iono_end <= len(rx_signal):
        # The ionosphere might have a slight additional Doppler (e.g. +0.25 Hz)
        fd_iono = fd_sat + 0.25 
        n = np.arange(chirp_len)
        t_samples_iono = (iono_start + n) / fs
        doppler_phasor_iono = np.exp(1j * 2.0 * np.pi * fd_iono * t_samples_iono)
        # rx_signal[iono_start:iono_end] += iono_atten * chirp * doppler_phasor_iono

# Add noise
noise = noise_floor * (np.random.randn(len(rx_signal)) + 1j * np.random.randn(len(rx_signal)))
rx_signal += noise.astype(np.complex64)

# Normalize and Export
mx = np.max(np.abs(rx_signal))
if mx > 0: rx_signal /= mx

output_file = "/Users/owenohlson/Documents/Academics/UVic/ionospheric_sounding/data/cubesat_sim.wav"
sf.write(output_file, np.column_stack((rx_signal.real, rx_signal.imag)), int(fs))

closest_approach_delay_ms = (bistatic_model(0, h_km, v_kms)[0] * 1000/c) * 1e3

print(f"Simulated cubesat pass starting at {start_time_offset}s offset.")
print(f"Direct path delay at closest approach: {closest_approach_delay_ms:.2f} ms")
print(f"Max Doppler predicted: {bistatic_model(start_time_offset, h_km, v_kms)[1]:.2f} Hz")
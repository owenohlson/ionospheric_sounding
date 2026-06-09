import numpy as np
import soundfile as sf
from lfm_utils import LFMWaveform

c = 299_792_458.0      # Speed of light (m/s)
pi = np.pi

# ---------------
# LFM parameters
# ---------------

fc = 29.0e6            # Carrier frequency (Hz) for Doppler modeling
# v = 100.0              # Relative radial velocity (m/s), positive = receding
initial_dist = 0.0     # Initial range to target (m)
B = 100e3              # 100 kHz bandwidth (baseband chirp)
fsweep = 2.0           # 2 Hz sweep rate
fs = 100e3             # sample rate (Hz)
num_sweeps = 1800       # number of chirps
delay_s = 0.0          # leading pad at beginning of signal (s)
direct_atten = 1.0
# ref_1_atten = 0.0
# ref_1_atten = 2**(-5)  # reflection attenuation (~ -15 dB)
noise_floor = 0.0

delay_samples = int(delay_s * fs)

# ------------ #
# Generate LFM #
# ------------ #
lfm = LFMWaveform(
    bandwidth=B,
    sweep_frequency=fsweep,
    sample_rate=fs
)

chirp = lfm.waveform.astype(np.complex64)
chirp_len = len(chirp)

# --------------------------- #
# Allocate buffer (no clipping)
# --------------------------- #
# Compute maximum delay over the entire record so the echo never truncates.
t_total = (num_sweeps - 1) / fsweep
# max_dist = initial_dist + max(v, 0) * t_total
# max_tau_s = (2 * max_dist) / c
# max_tau_samples = int(np.ceil(max_tau_s * fs))

# rx buffer long enough for last sweep's echo to fully fit
rx_len = num_sweeps * chirp_len
rx_signal = np.zeros(rx_len, dtype=np.complex64)

# ---------------- #
# Simulate channel #
# ---------------- #
for i in range(num_sweeps):
    sweep_start = i * chirp_len

    # Direct path (leakage / ground wave): full chirp, no Doppler
    rx_signal[sweep_start:sweep_start + chirp_len] += direct_atten * chirp

# -------------------- #
# Synthesize IQ Signal #
# -------------------- #
iq = np.zeros(len(rx_signal) + delay_samples, dtype=np.complex64)
iq[delay_samples:] = rx_signal

# Add noise
# noise = noise_floor * (np.random.randn(len(iq)) + 1j * np.random.randn(len(iq)))
# iq += noise.astype(np.complex64)

# Normalize for WAV export
mx = np.max(np.abs(iq))
if mx > 0:
    iq /= mx

# ----------------
# Write stereo IQ WAV
# ----------------
iq_stereo = np.column_stack((iq.real, iq.imag)).astype(np.float32)

output_file = "/Users/owenohlson/Documents/Academics/UVic/ionospheric_sounding/data/tx_before_channel_900s.wav"
sf.write(output_file, iq_stereo, int(fs), subtype="FLOAT")

print(f"Wrote {output_file}")
print(f"Samples: {len(iq)}")
print(f"Duration: {len(iq)/fs:.2f} s")
print(f"Chirp duration T: {1/fsweep:.3f} s, chirp_len: {chirp_len} samples, max range: {c / 2 / fsweep / 1e3:.1f} km")
print(f"Initial delay: {2*initial_dist/c*1e3:.3f} ms")
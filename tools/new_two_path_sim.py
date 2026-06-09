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
t_total = 900.0          # total duration of signal (s)
noise_floor = 0.01       # Relative noise floor (0.01 ~ -20 dB relative to max signal)

num_sweeps = int(t_total * fsweep)

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

# Path 2
path_2_tau_0 = path_1_tau + 0.005  # +5 ms extra delay
path_2_gain = 0.5

# Desired Doppler
doppler_mag = 0.25      # Hz
doppler_period = 240.0  # seconds

# Time axis for entire signal
t_global = np.arange(rx_len) / fs

fD = np.where(
    (t_global % doppler_period) < (doppler_period / 2),
    doppler_mag,
    -doppler_mag
)

phase = 2 * np.pi * np.cumsum(fD) / fs

# ---------------- #
# Simulate Channel #
# ---------------- #

for i in range(num_sweeps):

    sweep_base = i * chirp_len

    # -------- #
    #  Path 1  #
    # -------- #
    tau_samples = int(np.round(path_1_tau * fs))
    start = sweep_base + tau_samples
    end = start + chirp_len

    if end <= len(rx_signal):
        rx_signal[start:end] += path_1_gain * chirp

    # -------- #
    #  Path 2  #
    # -------- #

    for n in range(chirp_len):

        idx = sweep_base + n

        if idx >= len(rx_signal):
            break

        # Current time
        t = idx / fs

        delay_samples = int(np.round(path_2_tau_0 * fs))
        target_idx = idx + delay_samples


        if target_idx < len(rx_signal):

            # phase_in_cycle = (t % doppler_period) / doppler_period

            # if phase_in_cycle < 0.5:
            #     fD = doppler_mag
            # else:
            #     fD = -doppler_mag

            # # Apply the continuous phase rotation for the Doppler shift
            # phase_rotation = np.exp(
            #     1j * 2 * np.pi * fD * t
            # )            
            
            phase_rotation = np.exp(1j * phase[idx])
            
            rx_signal[target_idx] += (path_2_gain * 
                                      chirp[n] * 
                                      phase_rotation)

# Add noise
noise = noise_floor * (np.random.randn(len(rx_signal)) + 1j * np.random.randn(len(rx_signal)))
rx_signal += noise.astype(np.complex64)

# Normalize and Export
mx = np.max(np.abs(rx_signal))
if mx > 0: rx_signal /= mx

output_file = "/Users/owenohlson/Documents/Academics/UVic/ionospheric_sounding/data/new_two_path_sim.wav"
reference_output_file = "/Users/owenohlson/Documents/Academics/UVic/ionospheric_sounding/data/new_two_path_sim_reference.wav"
sf.write(output_file, np.column_stack((rx_signal.real, rx_signal.imag)), int(fs))
sf.write(reference_output_file, np.column_stack((tx_signal.real, tx_signal.imag)), int(fs))
print(f"Simulated two-path channel with {num_sweeps} sweeps and a Doppler shift of ±{doppler_mag} Hz over a {doppler_period}s period.")
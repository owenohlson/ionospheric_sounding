import argparse
import os

import numpy as np
import soundfile as sf

# Simulate a 2-ray channel model where the rays are of constant but unequal
# amplitude and the variable path delay varies with a triangle or sine wave
# around a constant delay.  For the triangle-wave case, this is like a target
# moving at constant radial velocity and then turning around.


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a two-path LFM simulation with a sinusoidal or triangular variable delay.",
        formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=35),
    )
    parser.add_argument("--output", default="data/new_two_path_sim.wav",
                        help="Output stereo IQ WAV file")
    parser.add_argument("--duration", type=float, default=900.0,
                        help="Total simulated duration in seconds")
    parser.add_argument("--fc", type=float, default=29.0e6,
                        help="Carrier frequency used for Doppler phase term (Hz)")
    parser.add_argument("--bandwidth", type=float, default=-100.0e3,
                        help="Baseband LFM sweep bandwidth in Hz; sign controls chirp direction")
    parser.add_argument("--sweep-frequency", type=float, default=2.0,
                        help="Sweep repetition rate in Hz")
    parser.add_argument("--sample-rate", type=float, default=100.0e3,
                        help="Output sample rate in Hz")

    parser.add_argument("--path1-delay", type=float, default=0.0,
                        help="Fixed path delay in seconds")
    parser.add_argument("--path1-gain", type=float, default=1.0,
                        help="Fixed path complex amplitude magnitude")
    parser.add_argument("--path2-delay", type=float, default=5.0e-3,
                        help="Mean variable-path delay in seconds")
    parser.add_argument("--path2-gain", type=float, default=0.5,
                        help="Variable path complex amplitude magnitude")

    parser.add_argument("--delay-lfo", choices=["triangle", "smooth_triangle", "sine"], default="sine",
                        help="Delay modulation shape")
    parser.add_argument("--doppler-mag", type=float, default=0.9,
                        help="Peak Doppler shift of variable path in Hz")
    parser.add_argument("--doppler-period", type=float, default=900.0,
                        help="Period of the delay/Doppler modulation in seconds")
    parser.add_argument("--smooth-turn-time", type=float, default=30.0,
                        help="Approximate rounded reversal time for smooth_triangle mode")

    parser.add_argument("--noise-floor", type=float, default=0.005,
                        help="Complex Gaussian noise amplitude before normalization")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    parser.add_argument("--chunk-seconds", type=float, default=10.0,
                        help="Chunk duration used while writing the WAV")
    parser.add_argument("--subtype", default="FLOAT",
                        help="SoundFile WAV subtype, e.g. FLOAT or PCM_16")
    return parser.parse_args()


def chirp_samples(t, bandwidth, sweep_frequency):
    """Periodic complex LFM samples at absolute times t."""
    sweep_period = 1.0 / sweep_frequency
    t_fast = np.mod(t, sweep_period)
    chirp_rate = bandwidth / sweep_period
    phase_cycles = (-bandwidth / 2.0) * t_fast + 0.5 * chirp_rate * t_fast**2
    return np.exp(1j * 2.0 * np.pi * phase_cycles)


def build_smooth_triangle_delay_table(doppler_mag, doppler_period, fc, smooth_turn_time, num_points=200_000):
    """Build one period of a triangle-like delay with rounded Doppler reversals."""
    half_period = doppler_period / 2.0
    lfo_t = np.linspace(0.0, doppler_period, num_points, endpoint=False)
    raw_fd = np.where(lfo_t < half_period, doppler_mag, -doppler_mag)

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


def make_delay_function(args):
    half_period = args.doppler_period / 2.0
    delay_rate_mag = args.doppler_mag / args.fc
    triangle_amp = delay_rate_mag * args.doppler_period / 4.0
    sine_amp = args.doppler_mag * args.doppler_period / (2.0 * np.pi * args.fc)

    smooth_t, smooth_delay = build_smooth_triangle_delay_table(
        args.doppler_mag,
        args.doppler_period,
        args.fc,
        args.smooth_turn_time,
    )

    def triangle_delay(t):
        phase_t = np.mod(t, args.doppler_period)
        tau = np.empty_like(phase_t, dtype=np.float64)
        first_half = phase_t < half_period
        tau[first_half] = args.path2_delay + triangle_amp - delay_rate_mag * phase_t[first_half]
        tau[~first_half] = (
            args.path2_delay
            - triangle_amp
            + delay_rate_mag * (phase_t[~first_half] - half_period)
        )
        return tau

    def sine_delay(t):
        omega = 2.0 * np.pi / args.doppler_period
        return args.path2_delay - sine_amp * np.sin(omega * t)

    def smooth_triangle_delay(t):
        phase_t = np.mod(t, args.doppler_period)
        return args.path2_delay + np.interp(phase_t, smooth_t, smooth_delay)

    if args.delay_lfo == "triangle":
        return triangle_delay, triangle_amp
    if args.delay_lfo == "smooth_triangle":
        return smooth_triangle_delay, float(np.max(np.abs(smooth_delay)))
    return sine_delay, sine_amp


def delayed_periodic_chirp(t_out, tau, args):
    t_tx = t_out - tau
    valid = (t_tx >= 0.0) & (t_tx < args.duration)
    y = np.zeros(t_out.shape, dtype=np.complex64)
    if np.any(valid):
        y[valid] = chirp_samples(
            t_tx[valid],
            args.bandwidth,
            args.sweep_frequency,
        ).astype(np.complex64)
    return y


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    path2_delay, delay_amplitude = make_delay_function(args)

    output_file = os.path.expanduser(args.output)
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    sample_rate = int(round(args.sample_rate))
    total_samples = int(round(args.duration * sample_rate))
    chunk_samples = max(1, int(round(args.chunk_seconds * sample_rate)))

    peak_bound = abs(args.path1_gain) + abs(args.path2_gain) + 6.0 * args.noise_floor
    if peak_bound <= 0:
        peak_bound = 1.0

    with sf.SoundFile(
        output_file,
        mode="w",
        samplerate=sample_rate,
        channels=2,
        subtype=args.subtype,
    ) as wav:
        for start in range(0, total_samples, chunk_samples):
            end = min(start + chunk_samples, total_samples)
            idx = np.arange(start, end, dtype=np.float64)
            t_out = idx / sample_rate

            tau1 = np.full_like(t_out, args.path1_delay)
            path1 = delayed_periodic_chirp(t_out, tau1, args)

            tau2 = path2_delay(t_out)
            path2 = delayed_periodic_chirp(t_out, tau2, args)
            carrier_phase = np.exp(-1j * 2.0 * np.pi * args.fc * tau2).astype(np.complex64)

            rx = args.path1_gain * path1 + args.path2_gain * path2 * carrier_phase
            if args.noise_floor > 0:
                noise = args.noise_floor * (
                    rng.standard_normal(end - start) + 1j * rng.standard_normal(end - start)
                )
                rx = rx + noise.astype(np.complex64)

            rx = (rx / peak_bound).astype(np.complex64)
            wav.write(np.column_stack((rx.real, rx.imag)))

    print(
        f"Simulated two-path channel: duration={args.duration:.3f}s, "
        f"sweeps={int(args.duration * args.sweep_frequency)}, "
        f"{args.delay_lfo} delay {args.path2_delay * 1e3:.3f} ms "
        f"+/- {delay_amplitude * 1e6:.3f} us, "
        f"target Doppler +/-{args.doppler_mag:.3f} Hz, output='{output_file}'."
    )


if __name__ == "__main__":
    main()

# dechirp_plot.py

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from plotting_utils import plot_dechirp
from lfm_utils import LFMWaveform, load_iq_audio, dechirp_fft_complex, window_from_arg


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
):
    chirp_len = lfm_config.sweep_length
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
    delay_ms = -fb / k * 1e3
    delay_order = np.argsort(delay_ms)
    delay_ms = delay_ms[delay_order]

    d_mask = np.ones_like(delay_ms, dtype=bool)
    if d_min is not None:
        d_mask = d_mask & (delay_ms >= d_min)
    if d_max is not None:
        d_mask = d_mask & (delay_ms <= d_max)
    selected_bins = delay_order[d_mask]
    delay_plot = delay_ms[d_mask]
    if selected_bins.size == 0:
        raise ValueError("No dechirp delay bins selected; check --d-min/--d-max")

    reference_chirp = lfm_config.waveform.astype(iq.dtype)
    w = window_from_arg(chirp_len, dechirp_window).astype(np.float32, copy=False)
    coherent_gain = np.mean(w)
    if coherent_gain == 0:
        raise ValueError(f"Window '{dechirp_window}' has zero coherent gain")

    num_groups = num_chirps // navg
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
    slow_time = tstart + np.arange(num_groups) * navg / prf

    plt.figure(figsize=(10, 6))
    plt.pcolormesh(slow_time, delay_plot, power_db.T, shading="nearest",
                   cmap="inferno", vmin=vmin, vmax=vmax)
    plt.ylabel("Delay [ms]")
    plt.xlabel("Time [s]")
    plt.title(title)
    plt.colorbar(label="Power [dB]")
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


def main():
    parser = argparse.ArgumentParser(description="Plot a range-time profile from a dechirp output.",
                                     formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog,
                                                                                                         max_help_position=35))

    parser.add_argument("input_file", type=str, help="Path to the stereo IQ audio file (.wav, .flac)")
    # LFM waveform parameters
    parser.add_argument("--sweep-frequency", type=float, required=True, help="Sweep repetition rate (Hz)")
    parser.add_argument("--bandwidth", type=float, default=100e3, help="LFM sweep bandwidth in Hz")

    # Optional arguments
    parser.add_argument("--title", type=str, default="Stretch-Processed Range-Time Plot", help="Plot title")
    parser.add_argument("--output", type=str, default=None,
                        help="Output image file path, displays the plot without saving if omitted")
    parser.add_argument("--vmin", type=float, default=None, help="Minimum dB scale (Auto scales if not provided)")
    parser.add_argument("--vmax", type=float, default=None, help="Maximum dB scale (Auto scales if not provided)")
    parser.add_argument("--window", type=float, default=None, help="Sliding window width in seconds")
    parser.add_argument("--tstart", type=float, default=None,
                        help="Trim start time in seconds, plots the full signal if omitted")
    parser.add_argument("--tend", type=float, default=None,
                        help="Trim end time in seconds, plots the full signal if omitted")
    parser.add_argument("--navg", type=int, default=4, help="Number of sweeps to average before plotting")
    parser.add_argument("--window-center", type=float, default=None,
                        help="The time to center the window around each sweep, automatically calculates if omitted")
    parser.add_argument("--d-min", type=float, default=None, help="Min delay to display (ms)")
    parser.add_argument("--d-max", type=float, default=None, help="Max delay to display (ms)")
    parser.add_argument("--dechirp-window", type=str, default="hann",
                        choices=["hamming", "hann", "none"], help="Fast-time dechirp FFT window")
    parser.add_argument("--streaming", action="store_true",
                        help="Process one chirp at a time and store only displayed delay bins")


    args = parser.parse_args()


    info = sf.info(args.input_file)
    fs = info.samplerate

    # Set default tstart/tend if not provided
    if args.tstart is None:
        args.tstart = 0.0
        print(f"No --tstart provided, starting from beginning of file")
    if args.tend is None:
        duration = info.frames / fs
        args.tend = duration
        print(f"No --tend provided, using end time of file: {args.tend:.2f} seconds")

    start_idx = int(args.tstart * fs)
    end_idx = int(args.tend * fs)

    # Load only the requested time span; long simulations can be many GB.
    iq, fs = load_iq_audio(args.input_file, start=start_idx, stop=end_idx)

    # Construct LFM waveform config
    lfm_config = LFMWaveform(
        sample_rate=fs,
        sweep_frequency=args.sweep_frequency,
        bandwidth=args.bandwidth,
    )

    if args.streaming or args.d_min is not None or args.d_max is not None:
        plot_dechirp_streaming(
            iq=iq,
            lfm_config=lfm_config,
            vmin=args.vmin,
            vmax=args.vmax,
            title=args.title,
            output_file=args.output,
            navg=args.navg,
            d_min=args.d_min,
            d_max=args.d_max,
            dechirp_window=None if args.dechirp_window == "none" else args.dechirp_window,
            tstart=args.tstart,
        )
    else:
        dechirp_mag_out, _ = dechirp_fft_complex(
            iq,
            lfm_config,
            window=None if args.dechirp_window == "none" else args.dechirp_window,
        )

        plot_dechirp(
            stretch_result=dechirp_mag_out,
            lfm_config=lfm_config,
            vmin=args.vmin,
            vmax=args.vmax,
            title=args.title,
            save_path=args.output,
            navg=args.navg,
            tstart=args.tstart,
        )


if __name__ == "__main__":
    main()

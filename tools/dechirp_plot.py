# dechirp_plot.py

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from plotting_utils import plot_dechirp, plot_dechirp_streaming
from lfm_utils import LFMWaveform, load_iq_audio, dechirp_fft_complex, window_from_arg, reference_gate_frequency_from_args


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
                        choices=["hamming", "hann", "cheb60", "cheb80", "cheb100", "cheb120", "none"],
                        help="Fast-time dechirp FFT window")
    parser.add_argument("--reference-gate-frequency", type=float, default=None,
                        help="Gate the reference chirp at this frequency in Hz")
    parser.add_argument("--reference-gate-period", type=float, default=None,
                        help="Gate the reference chirp at this period in seconds; e.g. 0.005 for 5 ms")
    parser.add_argument("--reference-gate-duty", type=float, default=0.5,
                        help="Reference gate duty cycle in (0, 1]")
    parser.add_argument("--reference-gate-phase", type=float, default=0.0,
                        help="Reference gate phase/time offset in seconds")
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
        reference_gate_frequency=reference_gate_frequency_from_args(args),
        reference_gate_duty=args.reference_gate_duty,
        reference_gate_phase=args.reference_gate_phase,
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

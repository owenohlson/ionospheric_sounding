# pdp_plot.py

import argparse
import soundfile as sf

from plotting_utils import plot_pdp
from lfm_utils import LFMWaveform, load_iq_audio, lfm_matched_filtering, reference_gate_frequency_from_args


def main():
    parser = argparse.ArgumentParser(description="Plot a power delay profile from a matched filter magnitude response.",
                                     formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog,
                                                                                                         max_help_position=35))

    parser.add_argument("input_file", type=str, help="Path to the stereo IQ audio file (.wav, .flac)")
    # LFM waveform parameters
    parser.add_argument("--sweep-frequency", type=float, required=True, help="Sweep repetition rate (Hz)")
    parser.add_argument("--bandwidth", type=float, default=100e3, help="LFM sweep bandwidth in Hz")

    # Optional arguments
    parser.add_argument("--title", type=str, default="Power Delay Profile", help="Plot title")
    parser.add_argument("--output", type=str, default=None,
                        help="Output image file path, displays the plot without saving if omitted")
    parser.add_argument("--vmin", type=float, default=None, help="Minimum dB scale (Auto scales if not provided)")
    parser.add_argument("--vmax", type=float, default=None, help="Maximum dB scale (Auto scales if not provided)")
    parser.add_argument("--window", type=float, default=5e-3, help="Sliding window width in seconds")
    parser.add_argument("--tstart", type=float, default=None,
                        help="Trim start time in seconds, plots the full signal if omitted")
    parser.add_argument("--tend", type=float, default=None,
                        help="Trim end time in seconds, plots the full signal if omitted")
    parser.add_argument("--navg", type=int, default=4, help="Number of sweeps to average before plotting")
    parser.add_argument("--window-center", type=float, default=None,
                        help="The time to center the window around each sweep, automatically calculates if omitted")
    parser.add_argument("--reference-gate-frequency", type=float, default=None,
                        help="Gate the reference chirp at this frequency in Hz")
    parser.add_argument("--reference-gate-period", type=float, default=None,
                        help="Gate the reference chirp at this period in seconds; e.g. 0.005 for 5 ms")
    parser.add_argument("--reference-gate-duty", type=float, default=0.5,
                        help="Reference gate duty cycle in (0, 1]")
    parser.add_argument("--reference-gate-phase", type=float, default=0.0,
                        help="Reference gate phase/time offset in seconds")


    args = parser.parse_args()

    info = sf.info(args.input_file)
    fs = info.samplerate

    if args.tstart is None:
        args.tstart = 0.0
        print("No --tstart provided, starting from beginning of file")
    if args.tend is None:
        duration = info.frames / fs
        args.tend = duration
        print(f"No --tend provided, using end time of file: {args.tend:.2f} seconds")

    start_idx = int(args.tstart * fs)
    end_idx = int(args.tend * fs)

    # Load only the requested time span; full-file matched filtering is expensive.
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

    magnitude_response, _, _ = lfm_matched_filtering(iq, lfm_config)

    if args.window_center is None:
        delay_reference_note = (
            "Relative delay 0 ms = auto matched-filter peak/window center; "
            "no WAV GPS timestamp used, so integer-second offset is unknown."
        )
    else:
        delay_reference_note = (
            f"Relative delay 0 ms = --window-center {args.window_center * 1e3:.3f} ms "
            "within each matched-filter sweep; no WAV GPS timestamp used."
        )
    print(delay_reference_note)

    # Call plotting function
    plot_pdp(
        magnitude_response=magnitude_response,
        lfm_config=lfm_config,
        window_width=args.window,
        title=args.title,
        output_file=args.output,
        vmin=args.vmin,
        vmax=args.vmax,
        tstart=None,
        tend=None,
        navg=args.navg,
        tcenter=args.window_center,
        delay_reference_note=delay_reference_note,
    )


if __name__ == "__main__":
    main()

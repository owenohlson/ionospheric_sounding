# phase_doppler_plot.py

import argparse
import os

import soundfile as sf

from lfm_utils import (
    LFMWaveform,
    auto_delay_bounds_from_iq,
    dechirp_fft_delay_band_complex,
    load_iq_audio,
    reference_gate_frequency_from_args,
)
from plotting_utils import plot_phase_doppler_time_delay


def main():
    parser = argparse.ArgumentParser(
        description="Plot sweep-to-sweep phase-derived Doppler vs time for each selected delay bin.",
        formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=35),
    )

    parser.add_argument("input_file", type=str, help="Path to the stereo IQ audio file (.wav, .flac)")
    parser.add_argument("--sweep-frequency", type=float, required=True, help="Sweep repetition rate / PRF (Hz)")
    parser.add_argument("--bandwidth", type=float, default=100e3, help="Chirp bandwidth (Hz)")

    parser.add_argument("--title", type=str, default="Phase Doppler vs Time", help="Plot title")
    parser.add_argument("--output", type=str, default=None, help="Output PNG filename; displays if omitted")
    parser.add_argument("--tstart", type=float, default=None)
    parser.add_argument("--tend", type=float, default=None)

    parser.add_argument("--d-min", type=float, default=None, help="Minimum delay band edge (ms)")
    parser.add_argument("--d-max", type=float, default=None, help="Maximum delay band edge (ms)")
    parser.add_argument("--auto-delay-sweeps", type=int, default=32,
                        help="Number of initial sweeps used when auto-selecting delay bounds")
    parser.add_argument("--auto-delay-half-width", type=float, default=1.5,
                        help="Half-width in ms around the strongest preview delay bin when auto-selecting")
    parser.add_argument("--fd-min", type=float, default=None,
                        help="Minimum Doppler color scale (Hz), clipped to +/- PRF/2")
    parser.add_argument("--fd-max", type=float, default=None,
                        help="Maximum Doppler color scale (Hz), clipped to +/- PRF/2")
    parser.add_argument("--estimator-sweeps", type=int, default=1,
                        help="Number of adjacent sweep-pair phase products to average")
    parser.add_argument("--power-floor-db", type=float, default=None,
                        help="Mask delay/time cells below this dechirped power level")

    parser.add_argument("--dechirp-window", type=str, default="hann",
                        choices=["hamming", "hann", "blackman", "cheb60", "cheb80", "cheb100", "cheb120", "none"])
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
        args.tend = info.frames / fs
        print(f"No --tend provided, using end time of file: {args.tend:.2f} seconds")

    start_idx = int(args.tstart * fs)
    end_idx = int(args.tend * fs)
    iq, fs = load_iq_audio(args.input_file, start=start_idx, stop=end_idx)

    lfm_config = LFMWaveform(
        sample_rate=fs,
        sweep_frequency=args.sweep_frequency,
        bandwidth=args.bandwidth,
        reference_gate_frequency=reference_gate_frequency_from_args(args),
        reference_gate_duty=args.reference_gate_duty,
        reference_gate_phase=args.reference_gate_phase,
    )
    dechirp_window = None if args.dechirp_window == "none" else args.dechirp_window
    auto_delay_used = args.d_min is None and args.d_max is None
    if (args.d_min is None) != (args.d_max is None):
        raise ValueError("Provide both --d-min and --d-max, or omit both for auto delay selection")
    if auto_delay_used:
        args.d_min, args.d_max, _ = auto_delay_bounds_from_iq(
            iq,
            lfm_config,
            window=dechirp_window,
            preview_sweeps=args.auto_delay_sweeps,
            half_width_ms=args.auto_delay_half_width,
        )

    selected_delay_ms, complex_spectra = dechirp_fft_delay_band_complex(
        received_signal=iq,
        lfm_config=lfm_config,
        d_min=args.d_min,
        d_max=args.d_max,
        window=dechirp_window,
    )

    info_rows = [
        ("File", os.path.basename(args.input_file)),
        ("Plot span", f"{args.tstart:.3f}-{args.tend:.3f} s"),
        ("Duration", f"{args.tend - args.tstart:.3f} s"),
        ("Delay band", f"{args.d_min:.3f}-{args.d_max:.3f} ms"),
        ("Dechirp window", args.dechirp_window),
    ]
    if auto_delay_used:
        info_rows.append(("Auto delay", f"first {args.auto_delay_sweeps} sweeps"))

    plot_phase_doppler_time_delay(
        dechirp_spectra=complex_spectra,
        selected_delay_ms=selected_delay_ms,
        lfm_config=lfm_config,
        title=args.title,
        output_file=args.output,
        fd_min=args.fd_min,
        fd_max=args.fd_max,
        estimator_sweeps=args.estimator_sweeps,
        power_floor_db=args.power_floor_db,
        tstart=args.tstart,
        info_rows=info_rows,
    )


if __name__ == "__main__":
    main()

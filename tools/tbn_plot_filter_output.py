# tbn_plot_filter_output.py

import argparse
import os

from lfm_utils import (
    LFMWaveform,
    auto_delay_bounds_from_iq,
    dechirp_fft_delay_band_complex,
    reference_gate_frequency_from_args,
)
from plotting_utils import (
    _timestamp_sweep_offset_samples,
    plot_dechirp_profile,
)
from tbn_utils import (
    gap_fill_note,
    lsl_open_tbn,
    lsl_print_metadata,
    lsl_read_block_for_one_stream,
    set_default_tbn_time_bounds,
    timestamp_range_note,
)


def main():
    parser = argparse.ArgumentParser(
        description="Dechirped power-versus-delay profile for TBN data (legacy script name)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("filename", help="TBN file to process")

    # Antenna selection
    parser.add_argument("-s", "--stand", type=int, default=1, help="stand ID to select")
    parser.add_argument("-p", "--pol", choices=["both", "x", "y", "0", "1"], default="both",
                        help="receiver polarization to use; both sums x and y power")

    # LFM waveform parameters
    parser.add_argument("--sweep-frequency", type=float, required=True, help="Sweep repetition rate (Hz)")
    parser.add_argument("--bandwidth", type=float, default=100e3, help="LFM sweep bandwidth in Hz")

    # Plot options
    parser.add_argument("--title", type=str, default=None, help="Plot title")
    parser.add_argument("--output", default=None, help="output PNG filename (optional)")
    parser.add_argument("--vmin", type=float, default=None, help="Minimum dB scale")
    parser.add_argument("--vmax", type=float, default=None, help="Maximum dB scale")
    parser.add_argument("--tstart", type=float, default=None)
    parser.add_argument("--tend", type=float, default=None)
    parser.add_argument("--d-min", type=float, default=None, help="Min delay to display (ms)")
    parser.add_argument("--d-max", type=float, default=None, help="Max delay to display (ms)")
    parser.add_argument("--units", choices=["s", "ms"], default="ms", help="Delay units")
    parser.add_argument("--auto-delay-sweeps", type=int, default=32)
    parser.add_argument("--auto-delay-half-width", type=float, default=1.5, help="Automatic delay half-width in ms")

    # Dechirp/reference options
    parser.add_argument("--dechirp-window", type=str, default="hann",
                        choices=["hamming", "hann", "blackman", "cheb60", "cheb80", "cheb100", "cheb120", "none"],
                        help="Fast-time dechirp FFT window")
    parser.add_argument("--offset", type=float, default=0.0,
                        help="Timestamp mode: sweep start offset in seconds after each integer-second boundary")
    parser.add_argument("--gap-fill", choices=["nan", "zero"], default="nan",
                        help="Fill TBN timetag gaps with NaNs for blank/corrupt regions or zeros for continuity")
    parser.add_argument("--no-timestamp-align", action="store_true",
                        help="Do not align dechirp chunks to timestamp second boundaries")
    parser.add_argument("--reference-gate-frequency", type=float, default=None,
                        help="Gate the reference chirp at this frequency in Hz")
    parser.add_argument("--reference-gate-period", type=float, default=None,
                        help="Gate the reference chirp at this period in seconds; e.g. 0.005 for 5 ms")
    parser.add_argument("--reference-gate-duty", type=float, default=0.5,
                        help="Reference gate duty cycle in (0, 1]")
    parser.add_argument("--reference-gate-phase", type=float, default=0.0,
                        help="Reference gate phase/time offset in seconds")

    args = parser.parse_args()

    idf = lsl_open_tbn(args.filename)
    lsl_print_metadata(idf)

    fs = float(idf.get_info("sample_rate"))
    fc = float(idf.get_info("freq1"))

    set_default_tbn_time_bounds(args, idf)

    duration = args.tend - args.tstart

    lfm_config = LFMWaveform(
        sample_rate=fs,
        sweep_frequency=args.sweep_frequency,
        bandwidth=args.bandwidth,
        reference_gate_frequency=reference_gate_frequency_from_args(args),
        reference_gate_duty=args.reference_gate_duty,
        reference_gate_phase=args.reference_gate_phase,
    )

    iq, start_timestamp, gap_info = lsl_read_block_for_one_stream(
        idf,
        args.tstart,
        duration,
        stand_id=args.stand,
        pol=args.pol,
        gap_fill=args.gap_fill,
        return_gap_info=True,
    )
    footer_note = gap_fill_note(gap_info)
    corner_note = timestamp_range_note(start_timestamp, iq.shape[-1] / fs)

    start_offset_samples = 0
    if not args.no_timestamp_align:
        start_offset_samples = _timestamp_sweep_offset_samples(
            start_timestamp,
            lfm_config,
            sweep_offset=args.offset,
        )

    if args.title is None:
        args.title = os.path.basename(args.output) if args.output else "Dechirped Power–Delay Profile"

    dechirp_window = None if args.dechirp_window == "none" else args.dechirp_window
    file_start = idf.get_info("start_time")
    info_rows = [
        ("File", os.path.basename(args.filename)),
        ("Sample rate", f"{fs / 1e3:.3f} kHz"),
        ("Tuning freq", f"{fc / 1e6:.6f} MHz"),
        ("Stand/pol", f"{args.stand}/{args.pol}"),
        ("Plot span", f"{args.tstart:.3f}-{args.tend:.3f} s"),
        ("Duration", f"{duration:.3f} s"),
        ("Averaging", "Linear power over selected sweeps"),
        ("Dechirp window", args.dechirp_window),
        ("Gap fill", args.gap_fill),
        ("Timestamp align", "no" if args.no_timestamp_align else "yes"),
    ]
    if file_start is not None:
        info_rows.insert(1, ("File start", str(getattr(file_start, "datetime", file_start))))
    if args.d_min is not None or args.d_max is not None:
        info_rows.append(("Delay limits", f"{args.d_min} to {args.d_max} ms"))

    if (args.d_min is None) != (args.d_max is None):
        raise ValueError("Provide both --d-min and --d-max, or omit both for automatic bounds")
    if args.d_min is None:
        args.d_min, args.d_max, _ = auto_delay_bounds_from_iq(
            iq, lfm_config, window=dechirp_window,
            start_offset_samples=start_offset_samples,
            preview_sweeps=args.auto_delay_sweeps,
            half_width_ms=args.auto_delay_half_width,
        )
    delay_ms, spectra = dechirp_fft_delay_band_complex(
        iq, lfm_config, args.d_min, args.d_max,
        window=dechirp_window, start_offset_samples=start_offset_samples,
    )
    plot_dechirp_profile(
        delay_ms, spectra, lfm_config, title=args.title,
        output_file=args.output, time_units=args.units,
        vmin=args.vmin, vmax=args.vmax, corner_note=corner_note,
        info_rows=info_rows, info_panel_footer_note=footer_note,
    )
    idf.close()


if __name__ == "__main__":
    main()

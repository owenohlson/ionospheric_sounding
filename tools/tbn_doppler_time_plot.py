# tbn_doppler_time_plot.py

import argparse

from lfm_utils import (
    LFMWaveform,
    dechirp_fft_delay_band_complex,
    reference_gate_frequency_from_args,
)
from plotting_utils import (
    _timestamp_sweep_offset_samples,
    plot_doppler_time_from_delay_band,
)
from tbn_utils import (
    lsl_open_tbn,
    lsl_print_metadata,
    lsl_read_block_for_one_stream,
    timestamp_range_note,
)


def main():
    parser = argparse.ArgumentParser(
        description="Plot slow-time Doppler vs time for a selected dechirp delay band in TBN data.",
        formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=35),
    )

    parser.add_argument("filename", help="TBN file to process")
    parser.add_argument("-s", "--stand", type=int, default=1, help="stand ID to select")
    parser.add_argument("-p", "--pol", type=str, default="x", help="pol to select (x/y)")

    parser.add_argument("--sweep-frequency", type=float, required=True, help="Sweep repetition rate / PRF (Hz)")
    parser.add_argument("--bandwidth", type=float, default=100e3, help="Chirp bandwidth (Hz)")

    parser.add_argument("--title", type=str, default=None, help="Plot title")
    parser.add_argument("--output", type=str, default=None, help="Output PNG filename; displays if omitted")
    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument("--tstart", type=float, default=None)
    parser.add_argument("--tend", type=float, default=None)

    parser.add_argument("--d-min", type=float, required=True, help="Minimum delay band edge (ms)")
    parser.add_argument("--d-max", type=float, required=True, help="Maximum delay band edge (ms)")
    parser.add_argument("--fd-min", type=float, default=None, help="Minimum Doppler to display (Hz)")
    parser.add_argument("--fd-max", type=float, default=None, help="Maximum Doppler to display (Hz)")
    parser.add_argument("--integration-time", type=float, default=60.0,
                        help="Slow-time Doppler integration time (s)")
    parser.add_argument("--hop-time", type=float, default=10.0,
                        help="Hop time between Doppler estimates (s)")
    parser.add_argument("--slow-window", type=str, default="hann",
                        choices=["hann", "hamming", "blackman", "cheb60", "cheb80", "cheb100", "cheb120", "none"])
    parser.add_argument("--nfft-doppler", type=int, default=None)
    parser.add_argument("--combine", type=str, default="incoherent",
                        choices=["incoherent", "coherent", "peak"],
                        help="How to combine selected delay bins before plotting Doppler power")

    parser.add_argument("--dechirp-window", type=str, default="hann",
                        choices=["hamming", "hann", "blackman", "cheb60", "cheb80", "cheb100", "cheb120", "none"])
    parser.add_argument("--offset", type=float, default=0.0,
                        help="Timestamp mode: sweep start offset in seconds after each integer-second boundary")
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

    if args.tstart is None:
        args.tstart = 0.0
        print("No --tstart provided, starting from beginning of file")
    if args.tend is None:
        n_frames_file = idf.get_info("nframe")
        args.tend = n_frames_file / fs
        print(f"No --tend provided, using end time of file: {args.tend:.2f} seconds")

    duration = args.tend - args.tstart

    lfm_config = LFMWaveform(
        sample_rate=fs,
        sweep_frequency=args.sweep_frequency,
        bandwidth=args.bandwidth,
        reference_gate_frequency=reference_gate_frequency_from_args(args),
        reference_gate_duty=args.reference_gate_duty,
        reference_gate_phase=args.reference_gate_phase,
    )

    iq, start_timestamp = lsl_read_block_for_one_stream(
        idf,
        args.tstart,
        duration,
        stand_id=args.stand,
        pol=args.pol,
    )
    corner_note = timestamp_range_note(start_timestamp, len(iq) / fs)

    start_offset_samples = 0
    if not args.no_timestamp_align:
        start_offset_samples = _timestamp_sweep_offset_samples(
            start_timestamp,
            lfm_config,
            sweep_offset=args.offset,
        )

    if args.title is None:
        args.title = (
            f"Doppler vs Time [fs={round(fs / 1000, 3)} kHz, "
            f"BW={round(args.bandwidth / 1e3, 3)} kHz, "
            f"sweep_freq={round(args.sweep_frequency, 3)} Hz, "
            f"stand={args.stand}, pol={args.pol}]"
        )

    selected_delay_ms, complex_spectra = dechirp_fft_delay_band_complex(
        received_signal=iq,
        lfm_config=lfm_config,
        d_min=args.d_min,
        d_max=args.d_max,
        window=None if args.dechirp_window == "none" else args.dechirp_window,
        start_offset_samples=start_offset_samples,
    )

    plot_doppler_time_from_delay_band(
        dechirp_spectra=complex_spectra,
        lfm_config=lfm_config,
        title=args.title,
        output_file=args.output,
        vmin=args.vmin,
        vmax=args.vmax,
        d_min=args.d_min,
        d_max=args.d_max,
        fd_min=args.fd_min,
        fd_max=args.fd_max,
        integration_time=args.integration_time,
        hop_time=args.hop_time,
        slow_window=args.slow_window,
        nfft_doppler=args.nfft_doppler,
        combine=args.combine,
        tstart=args.tstart + start_offset_samples / fs,
        selected_delay_ms=selected_delay_ms,
        corner_note=corner_note,
    )


if __name__ == "__main__":
    main()

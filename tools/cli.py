"""Shared command-line options for the export and plotting scripts.

Each ``add_*`` function adds one group of flags, so every script exposes the
same names with the same meaning. ``view_from_args`` opens an input as an
``RDCView`` with the selection flags applied.
"""

import argparse

from lfm_utils import WINDOW_CHOICES


def parser(description):
    return argparse.ArgumentParser(description=description,
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)


def add_input(p, raw_ok=False):
    p.add_argument("input", help="RDC HDF5 file" + (" or raw TBN (.dat/.tbn) / WAV recording" if raw_ok else ""))


def add_plot_output(p, video=False):
    g = p.add_argument_group("output")
    if video:
        g.add_argument("--output", required=True, help="Output directory for the video (and kept frames)")
        g.add_argument("--framerate", type=float, default=2.0)
        g.add_argument("--keep-frames", action="store_true")
        g.add_argument("--max-frames", type=int, default=None)
    else:
        g.add_argument("--output", required=True, help="Output PNG")
    g.add_argument("--title", default=None)
    g.add_argument("--vmin", type=float, default=None, help="Colour/axis minimum [dB]")
    g.add_argument("--vmax", type=float, default=None, help="Colour/axis maximum [dB]")


def add_selection(p, array=False, default_stands="all"):
    """Selection flags. ``array=None`` leaves pol/bad-stand/stand defaults to ``resolve_selection``."""
    g = p.add_argument_group("selection")
    if array is None:
        default_stands = None
    g.add_argument("--tstart", type=float, default=None, help="Start time [s] on the file's time axis")
    g.add_argument("--tend", type=float, default=None, help="End time [s] on the file's time axis")
    g.add_argument("-s", "--stands", "--stand", dest="stands", default=default_stands,
                   help='Stands to use: "5", "1-10,20", "first" (lowest stand in the input) or "all"')
    g.add_argument("-p", "--pol", choices=["both", "x", "y"], default=None if array is None else ("x" if array else "both"))
    g.add_argument("--bad-stands", choices=["none", "status"],
                   default=None if array is None else ("status" if array else "none"),
                   help="Drop stands: 'status' = flagged in the SSMIF (use --ssmif for the observation date)")
    g.add_argument("--gap-fill", choices=["nan", "zero"], default=None,
                   help="Default: the policy stored in the input")
    g.add_argument("--ssmif", default=None,
                   help="LWA station description (SSMIF or observation metadata .tgz) for the observation date; "
                        "default: LSL's bundled LWA-SV file (or $LWA_SSMIF)")


def add_delay(p):
    g = p.add_argument_group("delay window")
    g.add_argument("--d-min", type=float, default=None, help="Minimum delay [ms]")
    g.add_argument("--d-max", type=float, default=None, help="Maximum delay [ms]")
    g.add_argument("--full-delay", action="store_true",
                   help="Use every stored delay bin (default: +/-1.5 ms around the strongest bin)")


def add_doppler_limits(p):
    g = p.add_argument_group("Doppler window")
    g.add_argument("--fd-min", type=float, default=None, help="Minimum Doppler [Hz]")
    g.add_argument("--fd-max", type=float, default=None, help="Maximum Doppler [Hz]")


def add_range_processing(p):
    g = p.add_argument_group("range processing (applies only when the input is raw or RDC0)")
    g.add_argument("--bandwidth", type=float, default=None,
                   help="Chirp bandwidth [Hz], negative for down-chirps (default: stored value)")
    g.add_argument("--dechirp-window", choices=WINDOW_CHOICES, default=None, help="Default: stored value, else hann")
    g.add_argument("--reference-gate", default=None, metavar="FREQ[,DUTY[,PHASE]]",
                   help="Gate the reference chirp: frequency [Hz], duty (0-1), phase [s]")


def add_doppler_processing(p, defaults_from_file=True):
    g = p.add_argument_group("Doppler processing" + (" (default: stored RDC2 settings)" if defaults_from_file else ""))
    g.add_argument("--integration-time", type=float, default=None, help="Seconds per Doppler spectrum")
    g.add_argument("--hop-time", type=float, default=None, help="Seconds between spectra")
    g.add_argument("--slow-window", choices=WINDOW_CHOICES, default=None)
    g.add_argument("--nfft-doppler", type=int, default=None)


def add_phase(p):
    g = p.add_argument_group("phase-derived Doppler")
    g.add_argument("--estimator-sweeps", type=int, default=1, help="Sweep pairs averaged per estimate")
    g.add_argument("--power-floor-db", type=float, default=None, help="Hide estimates weaker than this")


def add_array(p):
    g = p.add_argument_group("array processing")
    g.add_argument("--beamformer", choices=["incoherent", "conventional", "mvdr"], default="incoherent",
                   help="incoherent = sum channel powers (no geometry); conventional/mvdr steer the array")
    g.add_argument("--combine", choices=["sum", "mean", "max"], default="sum",
                   help="How --beamformer incoherent combines channel powers")
    g.add_argument("--look-az", type=float, default=None, help="Beam azimuth [deg]; omit both to auto-select")
    g.add_argument("--look-el", type=float, default=None, help="Beam elevation [deg]")
    g.add_argument("--carrier-frequency", type=float, default=None, help="Default: stored tuning frequency")
    g.add_argument("--cable-correction", choices=["auto", "on", "off"], default="auto",
                   help="auto skips ideal simulations and already-corrected inputs")
    g.add_argument("--diagonal-loading", type=float, default=0.1)
    add_calibration(g)
    g.add_argument("--angle-output", default=None,
                   help="Also write angle (sky) maps: a PNG, or a directory for sliding videos")
    g.add_argument("--angle-cells", type=int, default=0,
                   help="0 = one map from the whole delay-Doppler window; N = maps for the N strongest cells")
    g.add_argument("--angle-method", choices=["conventional", "mvdr"], default="conventional")
    g.add_argument("--angle-step", type=float, default=2.0, help="Sky-map grid step [deg]")
    g.add_argument("--angle-range-db", type=float, default=30.0, help="Sky-map colour range below its peak")
    g.add_argument("--beam-pattern-output", default=None, help="PNG of the array beam pattern for the look direction")


def add_calibration(g):
    g.add_argument("--calibration", default=None, metavar="CAL.h5",
                   help="Per-stand self-calibration from calibrate_array.py (phases applied, flagged stands dropped)")


def load_calibration(args):
    if not getattr(args, "calibration", None):
        return None
    from calibration_utils import ArrayCalibration
    return ArrayCalibration.load(args.calibration)


def resolve_selection(args):
    """Defaults for plots that can be single-receiver or array: array -> all stands, x pol, drop SSMIF-flagged;
    otherwise the first stand, both pols, nothing dropped."""
    array = uses_array(args)
    if args.stands is None:
        args.stands = "all" if array else "first"
    if args.pol is None:
        args.pol = "x" if array else "both"
    if args.bad_stands is None:
        args.bad_stands = "status" if array else "none"
    return array


def uses_array(args):
    return getattr(args, "beamformer", "incoherent") != "incoherent" or bool(getattr(args, "angle_output", None)) \
        or bool(getattr(args, "beam_pattern_output", None))


def validate(p, args):
    for name in ("integration_time", "hop_time", "framerate", "estimator_sweeps", "max_frames"):
        value = getattr(args, name, None)
        if value is not None and value <= 0:
            p.error(f"--{name.replace('_', '-')} must be positive")
    if (getattr(args, "look_az", None) is None) != (getattr(args, "look_el", None) is None):
        p.error("Give both --look-az and --look-el, or neither")
    if getattr(args, "d_min", None) is not None and getattr(args, "d_max", None) is not None and args.d_min >= args.d_max:
        p.error("--d-min must be below --d-max")


def apply_ssmif(args):
    if getattr(args, "ssmif", None):
        import lwa_utils
        lwa_utils.use_ssmif(args.ssmif)


def view_from_args(args):
    from rdc_utils import RDCView
    apply_ssmif(args)
    return RDCView(args.input, stands=args.stands, pol=args.pol, bad_stands=args.bad_stands,
                   tstart=args.tstart, tend=args.tend, gap_fill=args.gap_fill,
                   bandwidth=getattr(args, "bandwidth", None), dechirp_window=getattr(args, "dechirp_window", None),
                   reference_gate=getattr(args, "reference_gate", None))


def gap_footer(had_gaps, gap_fill):
    if not had_gaps:
        return None
    return f"* {'Zeros' if gap_fill == 'zero' else 'NaNs'} fill gaps or corrupted samples in this view."


def run(main):
    """Print ValueErrors as one-line messages instead of tracebacks."""
    import sys
    try:
        main()
    except (ValueError, FileNotFoundError) as exc:
        sys.exit(f"error: {exc}")


# ---------------------------------------------------------------------------
# Export command lines (export_rdc0/1/2/2p.py are one-line launchers)
# ---------------------------------------------------------------------------

EXPORT_HELP = {
    "rdc0": "Cut a raw TBN/WAV recording into sweeps: RDC0 (adc, pulses, channels).",
    "rdc1": "Dechirp + range FFT: RDC1 (delay, pulses, channels). Input: raw recording or RDC0.",
    "rdc2": "Doppler FFT: RDC2 (frames, delay, doppler, channels), keeping the RDC1 it came from. "
            "Input: raw recording, RDC0, RDC1, or an RDC2 that retained RDC1.",
    "rdc2p": "Phase-derived Doppler: RDC2P (delay, estimates, channels) of sweep-to-sweep lag products. "
             "Input: raw recording, RDC0 or RDC1.",
}


def export_main(stage, argv=None):
    from rdc_utils import ExportSettings, export
    p = parser(EXPORT_HELP[stage])
    add_input(p, raw_ok=True)
    p.add_argument("--output", required=True, help=f"Output {stage.upper()} HDF5 file")
    add_selection(p)
    g = p.add_argument_group("raw input (TBN/WAV only)")
    g.add_argument("--sweep-frequency", type=float, default=None, help="Sweeps per second [Hz]")
    g.add_argument("--offset", type=float, default=0.0,
                   help="Sweep start offset [s] after each UTC second (use the observed echo delay minus a few ms)")
    g.add_argument("--no-timestamp-align", action="store_true", help="Start sweeps at the first sample instead")
    g.add_argument("--adc-start", type=int, default=0, help="First sample kept in each sweep")
    g.add_argument("--adc-count", type=int, default=None, help="Samples kept per sweep (default: rest of sweep)")
    g.add_argument("--start-utc", default=None, help="WAV only: UTC of sample zero (ISO 8601)")
    g.add_argument("--carrier-frequency", type=float, default=None, help="WAV only: RF centre frequency [Hz]")
    if stage != "rdc0":
        add_range_processing(p)
        add_delay(p)
    else:
        p.add_argument("--bandwidth", type=float, default=None, help="Record the chirp bandwidth [Hz] for later stages")
    if stage == "rdc2":
        add_doppler_processing(p, defaults_from_file=False)
        add_doppler_limits(p)
        p.add_argument("--no-keep-rdc1", action="store_true",
                       help="Do not store per-sweep RDC1 (integration/hop can then not be changed later)")
    if stage == "rdc2p":
        p.add_argument("--estimator-sweeps", type=int, default=1, help="Sweep pairs summed per estimate")
    g = p.add_argument_group("storage")
    g.add_argument("--dtype", choices=["float32", "float16"], default="float32", help="float16 is lossy")
    g.add_argument("--compression", choices=["gzip", "lzf", "none"], default="gzip")
    g.add_argument("--dry-run", action="store_true", help="Only report the output size")
    args = p.parse_args(argv)
    validate(p, args)
    s = ExportSettings(
        sweep_frequency=args.sweep_frequency, offset=args.offset, timestamp_align=not args.no_timestamp_align,
        start_utc=args.start_utc, carrier_frequency=args.carrier_frequency, adc_start=args.adc_start,
        adc_count=args.adc_count, tstart=args.tstart, tend=args.tend, stands=args.stands, pol=args.pol,
        bad_stands=args.bad_stands, gap_fill=args.gap_fill, bandwidth=args.bandwidth,
        dechirp_window=getattr(args, "dechirp_window", None), reference_gate=getattr(args, "reference_gate", None),
        d_min=None if getattr(args, "full_delay", False) else getattr(args, "d_min", None),
        d_max=None if getattr(args, "full_delay", False) else getattr(args, "d_max", None),
        integration_time=getattr(args, "integration_time", None), hop_time=getattr(args, "hop_time", None),
        slow_window=getattr(args, "slow_window", None) or "hann", nfft_doppler=getattr(args, "nfft_doppler", None),
        fd_min=getattr(args, "fd_min", None), fd_max=getattr(args, "fd_max", None),
        keep_rdc1=not getattr(args, "no_keep_rdc1", False), estimator_sweeps=getattr(args, "estimator_sweeps", 1),
        dtype=args.dtype, compression=args.compression, dry_run=args.dry_run)
    apply_ssmif(args)
    raw = not args.input.lower().endswith((".h5", ".hdf5"))
    if not raw:
        for flag in ("sweep_frequency", "start_utc", "carrier_frequency"):
            if getattr(args, flag) is not None:
                p.error(f"--{flag.replace('_', '-')} applies only to raw input; cached files store it")
        if args.offset or args.no_timestamp_align or args.adc_start or args.adc_count:
            p.error("--offset/--no-timestamp-align/--adc-* apply only when cutting a raw recording into sweeps")
    export(args.input, stage, args.output, s, tool=f"export_{stage}")

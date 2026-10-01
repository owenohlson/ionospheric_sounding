"""Beamform an array RDC2 into RDC3 (frames, delay, doppler, beams).

Input: any file that can produce RDC2 frames (RDC2, or RDC1/RDC0 with Doppler
settings). Each beam is steered with --look AZ,EL (repeatable); without --look,
one beam per frame is auto-steered to the strongest signal.
"""

import cli
import rdc_utils as ru


def main():
    p = cli.parser(__doc__)
    cli.add_input(p)
    p.add_argument("--output", required=True, help="Output RDC3 HDF5 file")
    cli.add_selection(p, array=True)
    cli.add_range_processing(p)
    cli.add_delay(p)
    cli.add_doppler_processing(p)
    cli.add_doppler_limits(p)
    g = p.add_argument_group("beams")
    g.add_argument("--beamformer", choices=["conventional", "mvdr"], default="conventional")
    g.add_argument("--look", action="append", default=None, metavar="AZ,EL", help="Beam direction [deg]; repeatable")
    g.add_argument("--carrier-frequency", type=float, default=None)
    g.add_argument("--cable-correction", choices=["auto", "on", "off"], default="auto")
    g.add_argument("--diagonal-loading", type=float, default=0.1)
    cli.add_calibration(g)
    g.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    g.add_argument("--compression", choices=["gzip", "lzf", "none"], default="gzip")
    args = p.parse_args()
    cli.validate(p, args)
    view = cli.view_from_args(args)
    view.select_delay(args.d_min, args.d_max, args.full_delay)
    processor = ru.ArrayProcessor(view.stands, view.pols, args.carrier_frequency or view.fc, args.beamformer,
                                  args.cable_correction, view.meta, args.diagonal_loading,
                                  calibration=cli.load_calibration(args))
    looks = None if not args.look else [tuple(float(v) for v in l.split(",")) for l in args.look]
    settings = dict(integration=args.integration_time, hop=args.hop_time, window=args.slow_window,
                    nfft=args.nfft_doppler, fd_min=args.fd_min, fd_max=args.fd_max)
    ru.export_rdc3(view, args.output, processor, looks, settings, args.dtype, args.compression)


if __name__ == "__main__":
    cli.run(main)

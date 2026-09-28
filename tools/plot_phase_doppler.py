"""Phase-derived Doppler (sweep-to-sweep phase change) over delay and time.

Input: RDC2P, or RDC0/RDC1 (computed on the fly). Channels are combined by
summing their lag-1 products, which needs no phase calibration.
--kind delay-time (default): delay vs time coloured by Doppler.
--kind doppler-time: Doppler vs time, binned and coloured by power.
"""

import numpy as np

import cli
import lfm_utils as lfm
import plotting_utils as plots
import rdc_utils as ru


def phase_products(args):
    view = cli.view_from_args(args)
    stage = "rdc2p" if view.file.has("rdc2p") and view.pulse_source() is None else None
    delay = view.select_delay(args.d_min, args.d_max, args.full_delay, stage=stage)
    pp = view.rdc2p(args.estimator_sweeps)
    fd = lfm.phase_doppler_hz(np.sum(pp["lag1"], axis=2), view.prf).T            # (est, delay)
    power_db = ru.to_db(np.nanmean(pp["power"], axis=2)).T
    if args.power_floor_db is not None:
        fd[power_db < args.power_floor_db] = np.nan
    lim = (max(args.fd_min if args.fd_min is not None else -view.prf / 2, -view.prf / 2),
           min(args.fd_max if args.fd_max is not None else view.prf / 2, view.prf / 2))
    rows = [("Estimator", "sweep-to-sweep phase"), ("Estimator sweeps", str(args.estimator_sweeps)),
            ("Unambiguous fd", f"+/-{view.prf / 2:g} Hz")]
    if args.power_floor_db is not None:
        rows.append(("Power floor", f"{args.power_floor_db:g} dB"))
    return view, delay, pp, fd, power_db, lim, rows


def build_parser(doc):
    p = cli.parser(doc)
    cli.add_input(p)
    return p


def add_common(p, video=False):
    cli.add_plot_output(p, video=video)
    cli.add_selection(p, default_stands="first")
    cli.add_range_processing(p)
    cli.add_delay(p)
    cli.add_doppler_limits(p)
    cli.add_phase(p)


def main():
    p = build_parser(__doc__)
    add_common(p)
    p.add_argument("--kind", choices=["delay-time", "doppler-time"], default="delay-time")
    p.add_argument("--fd-bins", type=int, default=128, help="doppler-time: number of Doppler bins")
    p.add_argument("--bin-combine", choices=["max", "sum", "mean"], default="max",
                   help="doppler-time: how estimates falling in one bin are combined")
    args = p.parse_args()
    cli.validate(p, args)
    view, delay, pp, fd, power_db, lim, extra = phase_products(args)
    t = pp["times"]
    rows = view.info_rows(t[0], t[-1]) + extra
    footer = cli.gap_footer(pp["gaps"].any(), view.gap_fill)
    if args.kind == "delay-time":
        plots.plot_phase_doppler_delay_time(t, delay, fd, lim, args.title or "Phase-derived Doppler", args.output,
                                            rows, footer=footer)
    else:
        edges = np.linspace(lim[0], lim[1], args.fd_bins + 1)
        grid = ru.phase_doppler_histogram(fd, power_db, edges, args.bin_combine)
        rows += [("Delay band", f"{delay[0]:.3f}-{delay[-1]:.3f} ms"), ("Bin combine", args.bin_combine)]
        plots.plot_phase_doppler_time(t, 0.5 * (edges[:-1] + edges[1:]), grid, args.title or "Phase-derived Doppler vs time",
                                      args.output, args.vmin, args.vmax, rows, footer=footer)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    cli.run(main)

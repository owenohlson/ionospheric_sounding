"""Delay vs time (range-time intensity) after dechirping. Input: RDC0 or RDC1 (including RDC1 kept in an RDC2 file).

Selected channels are combined by summing power; --navg averages consecutive sweeps.
"""

import numpy as np

import cli
import plotting_utils as plots
import rdc_utils as ru


def main():
    p = cli.parser(__doc__)
    cli.add_input(p)
    cli.add_plot_output(p)
    cli.add_selection(p, default_stands="first")
    cli.add_range_processing(p)
    cli.add_delay(p)
    p.add_argument("--navg", type=int, default=1, help="Sweeps averaged per plotted column")
    args = p.parse_args()
    cli.validate(p, args)
    view = cli.view_from_args(args)
    delay = view.select_delay(args.d_min, args.d_max, args.full_delay)
    power, gaps, times = view.rdc1_power()
    power = power.T                                          # (pulses, delay)
    n = (power.shape[0] // args.navg) * args.navg
    if n == 0:
        p.error("--navg exceeds the number of sweeps")
    power = power[:n].reshape(-1, args.navg, power.shape[1]).mean(axis=1)
    t = times[:n:args.navg] + (args.navg - 1) / (2 * view.prf)
    power_db = ru.to_db(power)
    rows = view.info_rows(times[0], times[-1] + 1 / view.prf) + [("Sweeps averaged", str(args.navg))]
    plots.plot_delay_time(t, delay, power_db, args.title or "Dechirped delay vs time", args.output,
                          args.vmin, args.vmax, rows, plots.delay_peaks_note(ru.strongest_delays(power_db, delay)),
                          cli.gap_footer(gaps.any(), view.gap_fill))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    cli.run(main)

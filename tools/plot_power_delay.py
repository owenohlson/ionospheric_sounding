"""Power-delay profile: dechirped power averaged over the selected sweeps. Input: RDC0 or RDC1.

Channel powers are summed, then averaged over valid sweeps before converting to dB.
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
    args = p.parse_args()
    cli.validate(p, args)
    view = cli.view_from_args(args)
    delay = view.select_delay(args.d_min, args.d_max, args.full_delay)
    rdc1, gaps, times = view.rdc1()
    power = np.sum(np.abs(rdc1) ** 2, axis=2)                # (delay, pulses)
    power_db = ru.to_db(np.nanmean(np.where(np.isfinite(power), power, np.nan), axis=1))
    rows = view.info_rows(times[0], times[-1] + 1 / view.prf) + [("Sweeps", str(len(times)))]
    plots.plot_power_delay(delay, power_db, args.title or "Power-delay profile", args.output, args.vmin, args.vmax,
                           rows, plots.delay_peaks_note(ru.strongest_delays(power_db, delay)),
                           cli.gap_footer(gaps.any(), view.gap_fill))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    cli.run(main)

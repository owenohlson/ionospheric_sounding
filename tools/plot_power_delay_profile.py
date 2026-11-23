"""Power-delay profile: dechirped power vs delay and time, with its time average beside it.

Input: a raw recording, RDC0 or RDC1 (including the RDC1 kept in an RDC2 file).
Selected channels are combined by summing power. --navg averages consecutive
sweeps in the image; the panel on the right averages every sweep without gaps.
"""

import warnings

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
    p.add_argument("--navg", type=int, default=1, help="Sweeps averaged per image column")
    args = p.parse_args()
    cli.validate(p, args)
    view = cli.view_from_args(args)
    delay = view.select_delay(args.d_min, args.d_max, args.full_delay)
    power, gaps, times = view.rdc1_power(skip_nan=False)     # (delay, pulses); sweeps with NaN stay NaN
    power = power.T                                          # (pulses, delay)
    n = (power.shape[0] // args.navg) * args.navg
    if n == 0:
        p.error("--navg exceeds the number of sweeps")
    finite = np.isfinite(power).all(axis=1)
    with warnings.catch_warnings():                          # all-gap columns are left blank
        warnings.simplefilter("ignore", RuntimeWarning)
        image = np.nanmean(power[:n].reshape(-1, args.navg, power.shape[1]), axis=1)
        mean = np.nanmean(power[finite], axis=0) if finite.any() else np.full(power.shape[1], np.nan)
    t = times[:n:args.navg] + (args.navg - 1) / (2 * view.prf)
    mean_db = ru.to_db(mean)
    rows = view.info_rows(times[0], times[-1] + 1 / view.prf) + [("Sweeps averaged", str(args.navg))]
    plots.plot_power_delay_profile(t, delay, ru.to_db(image), mean_db, args.title or "Power-delay profile",
                                   args.output, args.vmin, args.vmax, rows,
                                   plots.delay_peaks_note(ru.strongest_delays(mean_db, delay)),
                                   cli.gap_footer(gaps.any(), view.gap_fill),
                                   f"Time average ({int(finite.sum())} sweeps)")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    cli.run(main)

"""Doppler spectrum vs time for one delay band. Input: RDC0 or RDC1.

--combine incoherent averages channel-summed power over the delay band;
coherent sums channels as complex values first; peak uses the strongest delay bin.
"""

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
    cli.add_doppler_processing(p)
    cli.add_doppler_limits(p)
    p.add_argument("--combine", choices=["incoherent", "coherent", "peak"], default="incoherent")
    args = p.parse_args()
    cli.validate(p, args)
    view = cli.view_from_args(args)
    delay = view.select_delay(args.d_min, args.d_max, args.full_delay)
    integration = args.integration_time or view.meta.get("integration_time_s") or 60.0
    hop = args.hop_time or view.meta.get("hop_time_s") or 10.0
    window = args.slow_window or view.meta.get("slow_window") or "hann"
    t, fd, power, gaps, times = view.doppler_time(integration, hop, window, args.nfft_doppler, args.combine)
    keep = ru.range_mask(fd, args.fd_min, args.fd_max, "Doppler")
    rows = view.info_rows(times[0], times[-1] + 1 / view.prf) + [
        ("Delay band", f"{delay[0]:.3f}-{delay[-1]:.3f} ms"), ("Integration", f"{integration:g} s"),
        ("Hop", f"{hop:g} s"), ("Slow window", window), ("Combine", args.combine)]
    plots.plot_doppler_time(times[0] + t, fd[keep], ru.to_db(power[:, keep]), args.title or "Doppler vs time",
                            args.output, args.vmin, args.vmax, rows, footer=cli.gap_footer(gaps.any(), view.gap_fill))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    cli.run(main)

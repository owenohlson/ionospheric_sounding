"""Video of phase-derived Doppler: one frame per estimate, Doppler (x) vs delay (y) coloured by power.

Input: RDC2P, or RDC0/RDC1. --frame-step skips estimates between frames.
"""

import cli
import plot_phase_doppler as ppd
import plotting_utils as plots


def main():
    p = ppd.build_parser(__doc__)
    ppd.add_common(p, video=True)
    p.add_argument("--frame-step", type=int, default=1, help="Estimates between consecutive frames")
    args = p.parse_args()
    cli.validate(p, args)
    view, delay, pp, fd, power_db, lim, extra = ppd.phase_products(args)
    video = plots.FrameVideo(args.output, "phase_doppler", args.framerate, args.keep_frames)
    for n, i in enumerate(range(0, len(pp["times"]), max(1, args.frame_step))):
        if args.max_frames and n >= args.max_frames:
            break
        t = pp["times"][i]
        rows = view.info_rows(t, t) + extra
        plots.plot_phase_doppler_frame(fd[i], delay, power_db[i], lim, args.title or f"Phase-derived Doppler, t = {t:.2f} s",
                                       video.next_path(), args.vmin, args.vmax, rows,
                                       footer=cli.gap_footer(bool(pp["gaps"][i]), view.gap_fill))
    video.finish()


if __name__ == "__main__":
    cli.run(main)

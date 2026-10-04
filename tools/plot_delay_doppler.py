"""Delay-Doppler map from any RDC file, for one receiver, several receivers, or the array.

Input stage decides the path: RDC2 frames are used directly (or recomputed from
retained RDC1 when Doppler settings change); RDC1/RDC0 are dechirped and
Doppler-processed on the fly; RDC3 plots a stored beam.

Channel handling: ``--beamformer incoherent`` (default) sums channel powers;
``conventional``/``mvdr`` steer the array (one polarization) and enable sky
maps (``--angle-output``) and the beam pattern (``--beam-pattern-output``).
Without --stands, a single-receiver plot uses the first stand; array
processing uses all stands.
"""

import csv
import os

import numpy as np

import cli
import plotting_utils as plots
import rdc_utils as ru


def build_parser(video=False):
    p = cli.parser(__doc__)
    cli.add_input(p)
    cli.add_plot_output(p, video=video)
    cli.add_selection(p, array=None)
    cli.add_range_processing(p)
    cli.add_delay(p)
    cli.add_doppler_processing(p)
    cli.add_doppler_limits(p)
    cli.add_array(p)
    p.add_argument("--beam", type=int, default=0, help="RDC3 input: beam index to plot")
    return p


def frame_rows(frame):
    rows = [("Integration", f"{frame['integration']:.3f} s")]
    if frame.get("hop"):
        rows.append(("Hop", f"{float(frame['hop']):.3f} s"))
    rows += [("Slow window", str(frame.get("window"))), ("Doppler FFT", str(frame.get("nfft")))]
    return rows


def angle_maps(args, processor, cube, power_db, grid, delay_ms, doppler_hz):
    """Sky maps for the whole window or the strongest cells. Returns (maps, labels, markers, notes)."""
    az, el = grid
    maps, labels, markers, notes = [], [], [], []
    if args.angle_cells <= 0:
        blocks = [(cube.reshape(-1, cube.shape[-1]), "whole delay-Doppler window")]
    else:
        blocks = []
        for r, c in ru.pick_cells(power_db, args.angle_cells):
            block = cube[max(r - 1, 0):r + 2, max(c - 2, 0):c + 3]
            blocks.append((block.reshape(-1, cube.shape[-1]), (r, c)))
    for snaps, label in blocks:
        spec, R = processor.angle_map(snaps, az, el, args.angle_method)
        paz, pel, _ = ru.peak_direction(R, processor.positions, processor.fc, args.angle_method, args.angle_step)
        maps.append(spec)
        markers.append((paz, pel))
        if isinstance(label, tuple):
            r, c = label
            label = f"cell {len(maps)}: {delay_ms[r]:.3f} ms, {doppler_hz[c]:.3f} Hz\n({len(snaps)} snapshots)"
        labels.append(label)
        notes.append(f"Peak {len(maps)}: az {paz:.1f} deg, el {pel:.1f} deg")
    if args.angle_method == "mvdr" and min(len(s) for s, _ in blocks) < 2 * len(processor.positions):
        notes.append(f"MVDR note: fewer than 2N snapshots; relying on diagonal loading {args.diagonal_loading:g}")
    return maps, labels, markers, "\n".join(notes)



def run(args, video=False):
    array = cli.resolve_selection(args)
    view = cli.view_from_args(args)
    only_beams = view.file.has("rdc3") and len(view.file.stages) == 1
    delay = view.select_delay(args.d_min, args.d_max, args.full_delay)
    if only_beams:
        frames = view.rdc3_frames(args.beam, args.fd_min, args.fd_max)
        array = False
    else:
        frames = view.rdc2_frames(args.integration_time, args.hop_time, args.slow_window, args.nfft_doppler,
                                  args.fd_min, args.fd_max)
    processor = None
    if array:
        processor = ru.ArrayProcessor(view.stands, view.pols, args.carrier_frequency or view.fc,
                                      "conventional" if args.beamformer == "incoherent" else args.beamformer,
                                      args.cable_correction, view.meta, args.diagonal_loading,
                                      calibration=cli.load_calibration(args))
    grid = (np.arange(-180, 180, args.angle_step), np.arange(0, 90 + 0.5 * args.angle_step, args.angle_step))
    title = args.title or ("Delay-Doppler" + (f" ({args.beamformer} beam)" if array and args.beamformer != "incoherent" else ""))
    dd_video = plots.FrameVideo(args.output, "delay_doppler", args.framerate, args.keep_frames) if video else None
    angle_video = (plots.FrameVideo(args.angle_output, "angle_maps", args.framerate, args.keep_frames)
                   if video and args.angle_output else None)
    looks = []
    for frame in frames:
        if (not video and frame["index"] > 0) or (video and args.max_frames and frame["index"] >= args.max_frames):
            break
        cube, fd = frame["cube"], frame["doppler_hz"]
        rows = view.info_rows(frame["t0"], frame["t1"]) + frame_rows(frame)
        if processor is not None:
            cube = processor.prepare(cube)
            if args.look_az is not None:
                az, el, how = args.look_az, args.look_el, "fixed"
            else:
                (az, el), how = processor.look_direction(cube, args.angle_step), "auto"
            looks.append((frame["t0"], frame["t1"], az, el))
            rows += [("Array", f"{len(processor.stands)} stands, pol {'xy'[int(processor.pols[0])]}"),
                     ("Cable phase", processor.station_note)]
            if processor.calibration_note:
                rows.append(("Self-cal", processor.calibration_note))
            if args.beamformer == "incoherent":
                power = ru.incoherent_power(cube, args.combine)
                rows.append(("Beamformer", f"incoherent {args.combine}"))
            else:
                power = np.abs(processor.beamform(cube, az, el)) ** 2
                rows += [("Beamformer", args.beamformer), ("Look dir", f"{how}: az {az:.1f}, el {el:.1f} deg")]
                if args.beamformer == "mvdr":
                    rows.append(("MVDR cov", f"quietest 80% of cells, loading {args.diagonal_loading:g}"))
        elif only_beams:
            power = np.abs(cube[..., 0]) ** 2
            az = view.file.axis("rdc3", "beam_az_deg")
            rows.append(("Beam", f"{args.beam}: az {az[frame['index'], args.beam]:.1f} deg" if az is not None else str(args.beam)))
        else:
            power = ru.incoherent_power(cube, args.combine)
            rows.append(("Channel combine", f"incoherent {args.combine}"))
        power_db = ru.to_db(power)
        notes = plots.delay_peaks_note(ru.strongest_delays(power_db.T, delay))
        footer = cli.gap_footer(frame["gaps"], view.gap_fill)
        utc = view.utc_note(frame["t0"], frame["t1"])
        if utc:
            rows += [(k.strip(), v.strip()) for k, v in (line.split(":", 1) for line in utc.splitlines())]
        out = dd_video.next_path() if video else args.output
        plots.plot_delay_doppler(fd, delay, power_db, title, out, args.vmin, args.vmax, rows, notes, footer)
        if processor is not None and args.angle_output:
            maps, labels, markers, anotes = angle_maps(args, processor, cube, power_db, grid, delay, fd)
            path = angle_video.next_path() if video else args.angle_output
            plots.plot_angle_maps(maps, grid[0], grid[1], labels, f"Sky map ({args.angle_method})", path,
                                  args.angle_range_db, rows, anotes, footer, markers)
        if processor is not None and args.beam_pattern_output and len(looks) == 1:
            az, el = looks[0][2:]
            pattern = ru.beam_pattern(processor.positions, processor.fc, az, el, *grid)
            plots.plot_angle_maps([pattern], grid[0], grid[1], [f"look az {az:.1f}, el {el:.1f} deg"],
                                  "Array beam pattern (conventional)", args.beam_pattern_output,
                                  args.angle_range_db, rows[:6])
        print(f"Frame {frame['index'] + 1}: {frame['t0']:.3f}-{frame['t1']:.3f} s", flush=True)
    if video:
        dd_video.finish()
        if angle_video:
            angle_video.finish()
        if looks:
            with open(os.path.join(os.path.expanduser(args.output), "look_directions.csv"), "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["t_start_s", "t_end_s", "look_az_deg", "look_el_deg"])
                w.writerows(looks)
    else:
        print(f"Wrote {args.output}", flush=True)


def main(video=False):
    p = build_parser(video)
    args = p.parse_args()
    cli.validate(p, args)
    run(args, video)


if __name__ == "__main__":
    cli.run(main)

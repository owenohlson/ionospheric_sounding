"""Per-antenna phase self-calibration of an LWA array from strong echo cells.

The SSMIF (``--ssmif``) gives stand positions and cable delays. This command
measures what remains: one phase per stand, estimated from the strongest
delay-Doppler cells, which hold a single plane wave (see ``calibration_utils``
for the model and the alternating fit). It writes a small calibration file that
every array command accepts with ``--calibration``.

Steps
  1. Bring the input to RDC2 frames, apply the SSMIF cable correction.
  2. Pick the strongest echo cells in every frame (per-channel noise normalised,
     non-maximum suppressed, at least --min-snr-db above the median cell, and
     localised in delay so narrowband interference spread over all delays is
     skipped).
  3. Flag stands whose noise power is more than --max-power-db from the median.
  4. Fit the per-stand phases. If some cells get *worse* with them, fit those
     cells separately and flag stands whose phase differs between the two
     classes of echo (one phase per stand cannot serve both; on 2023-12-09 the
     classes look like the O and X magnetoionic modes). Flag stands whose phase
     does not follow the fit (consistency below half the median). Refit.
  5. Validate on held-out cells: fit on half (alternate frames when there are
     several, otherwise alternate cells), measure the plane-wave fit eta on the
     other half before and after, and swap.

Pointing is not changed by the calibration: a linear phase ramp across the array
is indistinguishable from a different arrival direction and is kept out of the
gains, so absolute directions still come from the SSMIF geometry.

Example:
    python tools/calibrate_array.py clean_rdc2.h5 --ssmif data/LD021_0001.tgz \
        --output cal_20231209_13p5.h5 --csv cal_stands.csv
    python tools/plot_delay_doppler.py clean_rdc2.h5 --ssmif data/LD021_0001.tgz \
        --beamformer conventional --calibration cal_20231209_13p5.h5 --output dd.png
"""

import csv
import os

import numpy as np

import calibration_utils as cu
import cli
import lwa_utils as lwa
import rdc_utils as ru


def build_parser():
    p = cli.parser(__doc__.split("\n\n")[0])
    cli.add_input(p)
    p.add_argument("--output", required=True, help="Calibration HDF5 file to write")
    cli.add_selection(p, array=True)
    cli.add_range_processing(p)
    cli.add_delay(p)
    cli.add_doppler_processing(p)
    cli.add_doppler_limits(p)
    g = p.add_argument_group("calibration")
    g.add_argument("--cells", type=int, default=40, help="Strongest cells used (over all frames)")
    g.add_argument("--min-snr-db", type=float, default=15.0, help="Minimum cell power above the median cell")
    g.add_argument("--max-power-db", type=float, default=10.0,
                   help="Flag stands whose noise power differs from the median by more than this")
    g.add_argument("--carrier-frequency", type=float, default=None, help="Default: stored tuning frequency")
    g.add_argument("--cable-correction", choices=["auto", "on", "off"], default="auto")
    g.add_argument("--csv", default=None, help="Optional per-stand table")
    return p


def strong_cells(frames, processor, n_cells, min_snr_db):
    """Echo cells from every frame (delay-localised, see ``calibration_utils.echo_cells``).

    Returns the chosen cells (strongest first, each with its normalised snapshot)
    and the per-channel noise power [dB re median].
    """
    candidates, noise = [], []
    for i, fr in enumerate(frames):
        cube = processor.prepare(fr["cube"]).astype(np.complex128)
        power, sigma2 = cu.normalized_power(cube)
        noise.append(sigma2)
        for r, c, snr, contrast in cu.echo_cells(power, n_cells, min_snr_db):
            candidates.append(dict(frame=i, t0=fr["t0"], t1=fr["t1"], delay_index=r, doppler_index=c,
                                   doppler_hz=float(fr["doppler_hz"][c]), snr_db=snr, contrast_db=contrast,
                                   x=cube[r, c] / np.sqrt(np.maximum(sigma2, 1e-30))))
    candidates.sort(key=lambda d: -d["snr_db"])
    noise_db = 10 * np.log10(np.nanmedian(np.array(noise), axis=0))
    return candidates[:n_cells], noise_db - np.nanmedian(noise_db)


def main(argv=None):
    p = build_parser()
    args = p.parse_args(argv)
    cli.validate(p, args)
    if args.gap_fill is None:
        args.gap_fill = "zero"
    view = cli.view_from_args(args)
    view.select_delay(args.d_min, args.d_max, args.full_delay)
    frames = list(view.rdc2_frames(args.integration_time, args.hop_time, args.slow_window, args.nfft_doppler,
                                   args.fd_min, args.fd_max))
    if not frames:
        raise ValueError("No RDC2 frames in the selection")
    fc = float(args.carrier_frequency or view.fc)
    processor = ru.ArrayProcessor(view.stands, view.pols, fc, cable_correction=args.cable_correction,
                                  metadata=view.meta)
    cells, power_db = strong_cells(frames, processor, args.cells, args.min_snr_db)
    if len(cells) < 3:
        raise ValueError(f"Only {len(cells)} echo cell(s) are {args.min_snr_db:g} dB above the median; "
                         "lower --min-snr-db or widen the delay/Doppler window")
    X = np.array([c["x"] for c in cells])
    frame_ids = np.array([c["frame"] for c in cells])
    groups = frame_ids % 2 if len(np.unique(frame_ids)) >= 2 and len(np.unique(frame_ids % 2)) == 2 else None
    split = "alternate frames" if groups is not None else "alternate cells"

    print(f"Input {args.input}: {len(frames)} frame(s), {len(processor.stands)} stands, pol "
          f"{'xy'[int(processor.pols[0])]}, {fc / 1e6:.6f} MHz, station {lwa.SSMIF_PATH or 'LSL bundled LWA-SV'}")
    print(f"Using {len(cells)} echo cells, {cells[-1]['snr_db']:.1f}-{cells[0]['snr_db']:.1f} dB above the median "
          f"cell; validation by {split}")
    res = cu.calibrate(X, processor.positions, fc, power_db, groups, args.max_power_db)
    v = res["validation"]

    stands, pols = processor.stands, processor.pols
    digitizers = [int(a.digitizer) for a in lwa.selected_antennas(stands, pols)]
    for c, (az, el), eta, b in zip(cells, res["directions"], res["eta"], res["class_b_cells"]):
        c.update(az=float(az), el=float(el), eta_after=float(eta), class_b=bool(b))
        c.pop("x")
    meta = dict(kind="array_calibration", created_utc=cu.now_utc(), tool="calibrate_array",
                input=os.path.abspath(args.input), ssmif=lwa.SSMIF_PATH, frequency_hz=fc,
                pol=int(pols[0]), cable_correction=processor.cable_note, phase_only=True,
                n_cells=len(cells), min_snr_db=args.min_snr_db, max_power_db=args.max_power_db,
                consistency_threshold=res["consistency_threshold"], iterations=res["iterations"],
                ramp_removed_du=res["ramp"], validation=v, validation_split=split,
                flag_reasons={str(int(stands[n])): r for n, r in res["reasons"].items()},
                delay_window_ms=[float(view.delay_ms[0]), float(view.delay_ms[-1])], cells=cells)
    cal = cu.ArrayCalibration(stands, pols, res["gain"], res["flagged"], digitizers, res["consistency"],
                              power_db, meta)
    cal.save(args.output)

    print("\nPlane-wave fit eta (phases only; 1 = perfect plane wave, "
          f"{v['noise_eta']:.4f} = random phases):")
    if v["eta_before_median"] is not None:
        print(f"  held-out cells ({v['held_out_cells']}):  SSMIF only {v['eta_power_flags_only_median']:.3f}  ->  "
              f"without flagged stands {v['eta_before_median']:.3f}  ->  with self-cal {v['eta_after_median']:.3f}"
              f"   ({v['cells_made_worse']} cell(s) made worse)")
    print(f"  fitted cells (optimistic):  {v['in_sample_eta_before_median']:.3f}  ->  "
          f"{v['in_sample_eta_after_median']:.3f}")
    az, el = np.median(res["directions"], axis=0)
    print(f"  median direction after calibration: az {az:.1f} deg, el {el:.1f} deg "
          f"(ramp kept out of the gains: du = {res['ramp'][0]:+.4f} E, {res['ramp'][1]:+.4f} N)")
    phase = np.degrees(np.angle(res["gain"]))
    used = ~res["flagged"]
    print(f"\nPer-stand phase corrections: rms {np.sqrt(np.mean(phase[used] ** 2)):.1f} deg over "
          f"{int(used.sum())} stands; largest:")
    for n in np.argsort(-np.abs(phase) * used)[:10]:
        print(f"  stand {int(stands[n]):3d}  {phase[n]:+7.1f} deg  consistency {res['consistency'][n]:.2f}")
    nb = int(res["class_b_cells"].sum())
    if nb:
        print(f"\n{nb} of {len(cells)} cells were made worse by the first-pass phases (a second class of echo, "
              "e.g. the other\nmagnetoionic mode); stands that respond differently to the two classes are flagged below.")
    print(f"\nFlagged {int(res['flagged'].sum())} stand(s) (dropped when the calibration is applied):")
    for n, reason in sorted(res["reasons"].items(), key=lambda kv: int(stands[kv[0]])):
        print(f"  stand {int(stands[n]):3d}  {reason}")
    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["stand", "pol", "digitizer", "phase_deg", "consistency", "power_db_re_median",
                        "flagged", "reason"])
            for n in range(len(stands)):
                w.writerow([int(stands[n]), int(pols[n]), digitizers[n], f"{phase[n]:.2f}",
                            f"{res['consistency'][n]:.3f}", f"{power_db[n]:.2f}", int(res["flagged"][n]),
                            res["reasons"].get(n, "")])
        print(f"Wrote {args.csv}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    cli.run(main)

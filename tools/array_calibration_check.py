"""Check whether array phases in an RDC2 cache behave like a calibrated plane wave.

For the strongest delay-Doppler cells, this script asks how much of the per-channel
power a single plane wave can explain:

    eta = max_direction |a^H x|^2 / (N * x^H x),   1/N <= eta <= 1

eta near 1 means the channels are phase-coherent across the array for some
direction (good calibration, one dominant path in the cell). eta near 1/N means
the phases look random. The same cells are tested with several cable-correction
variants so the sign convention and the LSL delays can be compared on real data:

    none        no cable correction
    lsl         the pipeline's correction, exp(+j 2 pi f d)
    lsl_flip    the opposite sign, exp(-j 2 pi f d)
    conj_lsl    conjugated I/Q followed by the pipeline's correction

It also reports per-stand phase residuals after removing the best plane wave.
Stands whose residuals are consistent across many independent cells are
candidates for mapping or calibration problems (or a self-calibration offset).

Example:
    python tools/array_calibration_check.py --rdc2-input 4747_rdc2_x.h5 --pol x \
        --top-cells 20 --csv stand_residuals.csv
"""

import argparse
import csv
import os
import types

import numpy as np

C = 299_792_458.0


def unit_vectors(az_deg, el_deg):
    az = np.deg2rad(np.asarray(az_deg, dtype=float))
    el = np.deg2rad(np.asarray(el_deg, dtype=float))
    return np.stack([np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)], axis=-1)


def steering_matrix(positions_m, frequency_hz, az_deg, el_deg):
    """Rows are steering vectors with the same convention as rdc_utils.steering_vector."""
    u = unit_vectors(az_deg, el_deg)
    return np.exp(1j * 2.0 * np.pi * frequency_hz / C * (u @ np.asarray(positions_m, dtype=float).T))


def pick_cells(power, n_cells, delay_guard=3, doppler_guard=5):
    """Top cells with simple non-maximum suppression, so cells are not all one feature's neighbors."""
    work = np.array(power, dtype=float, copy=True)
    work[~np.isfinite(work)] = -np.inf
    cells = []
    for _ in range(int(n_cells)):
        flat = int(np.argmax(work))
        if not np.isfinite(work.flat[flat]):
            break
        r, c = np.unravel_index(flat, work.shape)
        cells.append((int(r), int(c)))
        work[max(0, r - delay_guard):r + delay_guard + 1, max(0, c - doppler_guard):c + doppler_guard + 1] = -np.inf
    return cells


def best_plane_wave(x, positions_m, frequency_hz, coarse_step=2.0, fine_step=0.25):
    x = np.asarray(x, dtype=np.complex128)
    n = x.size
    energy = float(np.real(np.vdot(x, x)))
    if energy <= 0 or not np.isfinite(energy):
        return np.nan, np.nan, np.nan
    az = np.arange(-180.0, 180.0, coarse_step)
    el = np.arange(0.0, 90.0 + 0.5 * coarse_step, coarse_step)
    AZ, EL = np.meshgrid(az, el)
    A = steering_matrix(positions_m, frequency_hz, AZ.ravel(), EL.ravel())
    p = np.abs(np.conj(A) @ x) ** 2
    i = int(np.argmax(p))
    az0, el0 = AZ.ravel()[i], EL.ravel()[i]
    az_f = np.arange(az0 - coarse_step, az0 + coarse_step + 1e-9, fine_step)
    el_f = np.clip(np.arange(el0 - coarse_step, el0 + coarse_step + 1e-9, fine_step), 0.0, 90.0)
    AZ, EL = np.meshgrid(az_f, el_f)
    A = steering_matrix(positions_m, frequency_hz, AZ.ravel(), EL.ravel())
    p = np.abs(np.conj(A) @ x) ** 2
    i = int(np.argmax(p))
    eta = float(p[i] / (n * energy))
    return eta, float(((AZ.ravel()[i] + 180.0) % 360.0) - 180.0), float(EL.ravel()[i])


def correction_variants(x, cable_delays_s, frequency_hz):
    d = np.asarray(cable_delays_s, dtype=float)
    d = d - np.nanmin(d)
    lsl = np.exp(2j * np.pi * frequency_hz * d)
    return {
        "none": x,
        "lsl": x * lsl,
        "lsl_flip": x * np.conj(lsl),
        "conj_lsl": np.conj(x) * lsl,
    }


def calibration_report(rdc2, positions_m, frequency_hz, cable_delays_s, stands,
                       n_cells=20, n_noise=20, seed=0):
    """rdc2: (delay, doppler, channels) complex, channel order matching positions/cables/stands."""
    rdc2 = np.asarray(rdc2)
    finite = np.all(np.isfinite(rdc2.real) & np.isfinite(rdc2.imag), axis=2)
    power = np.where(finite, np.sum(np.abs(rdc2) ** 2, axis=2), np.nan)
    cells = pick_cells(power, n_cells)
    rng = np.random.default_rng(seed)
    quiet = np.argwhere(finite & (power <= np.nanpercentile(power, 50)))
    noise_cells = [tuple(map(int, quiet[i])) for i in rng.choice(len(quiet), min(n_noise, len(quiet)), replace=False)]
    floor = np.nanmedian(power)

    results = {}
    for label, cell_list in (("signal", cells), ("noise", noise_cells)):
        for r, c in cell_list:
            for name, xv in correction_variants(rdc2[r, c, :], cable_delays_s, frequency_hz).items():
                eta, az, el = best_plane_wave(xv, positions_m, frequency_hz)
                results.setdefault((label, name), []).append(
                    dict(cell=(r, c), eta=eta, az=az, el=el, snr_db=10 * np.log10(power[r, c] / floor)))

    # Per-stand residual phase after removing each strong cell's best plane wave (pipeline correction).
    # Only cells that are clearly plane-wave-like are used, so noise cells do not dominate.
    noise_eta = np.nanmedian([r["eta"] for r in results.get(("noise", "lsl"), [])] or [np.nan])
    eta_min = max(0.3, 3.0 * noise_eta) if np.isfinite(noise_eta) else 0.3
    resid = []
    for rec in results[("signal", "lsl")]:
        if not rec["eta"] >= eta_min:
            continue
        r, c = rec["cell"]
        xv = correction_variants(rdc2[r, c, :], cable_delays_s, frequency_hz)["lsl"]
        a = steering_matrix(positions_m, frequency_hz, [rec["az"]], [rec["el"]])[0]
        z = xv * np.conj(a)
        z = z / (np.abs(z) + 1e-30)
        z = z * np.conj(np.sum(z) / (np.abs(np.sum(z)) + 1e-30))  # remove common phase
        resid.append(z)
    resid = np.asarray(resid)
    mean_vec = resid.mean(axis=0) if resid.size else np.full(len(stands), np.nan)
    n_used = len(resid)
    per_stand = [dict(stand=int(s), cells_used=n_used, mean_residual_deg=float(np.degrees(np.angle(m))),
                      consistency=float(np.abs(m)),
                      rms_residual_deg=float(np.degrees(np.sqrt(np.mean(np.angle(resid[:, i]) ** 2)))) if resid.size else np.nan)
                 for i, (s, m) in enumerate(zip(stands, mean_vec))]
    return results, per_stand


def print_report(results, per_stand, n_channels):
    print(f"\nChannels: {n_channels}   (eta = 1 for a perfect plane wave; ~1/N = {1.0 / n_channels:.4f} for random phases)")
    print(f"{'cells':7s} {'variant':9s} {'median eta':>11s} {'max eta':>8s}   best directions (az, el) of first 5 cells")
    for label in ("signal", "noise"):
        for name in ("none", "lsl", "lsl_flip", "conj_lsl"):
            recs = results.get((label, name), [])
            if not recs:
                continue
            etas = np.array([r["eta"] for r in recs])
            dirs = ", ".join(f"({r['az']:.0f},{r['el']:.0f})" for r in recs[:5])
            print(f"{label:7s} {name:9s} {np.nanmedian(etas):11.4f} {np.nanmax(etas):8.4f}   {dirs}")
    print("\nInterpretation: 'lsl' should give the largest signal eta, clearly above the noise cells, "
          "with best directions that agree between cells of the same propagation mode.")
    if not per_stand or per_stand[0]["cells_used"] == 0:
        print("\nNo cell was plane-wave-like enough (eta >= max(0.3, 3x noise)) to estimate per-stand residuals.")
        return
    print(f"\nPer-stand residuals use {per_stand[0]['cells_used']} plane-wave-like cell(s).")
    ranked = sorted(per_stand, key=lambda d: -d["consistency"] * abs(d["mean_residual_deg"]))
    print("\nStands with the most consistent large phase residual (possible mapping/cable/calibration issues):")
    print(f"{'stand':>6s} {'mean resid [deg]':>17s} {'consistency':>12s}")
    for d in ranked[:12]:
        print(f"{d['stand']:6d} {d['mean_residual_deg']:17.1f} {d['consistency']:12.2f}")


def main():
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from rdc_cache import prepare_rdc2_view, iter_rdc2_view_chunks
    from rdc_utils import lwasv_selected_antennas
    from array_delay_doppler import contiguous_axis_selection, select_array_channels

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rdc2-input", required=True)
    p.add_argument("--frame-index", type=int, default=0)
    p.add_argument("--integration-time", type=float, default=None)
    p.add_argument("--hop-time", type=float, default=None)
    p.add_argument("--pol", default="x", choices=["x", "y", "0", "1"])
    p.add_argument("--keep-bad-stands", action="store_true")
    p.add_argument("--operational-only", action="store_true")
    p.add_argument("--d-min", type=float, default=None)
    p.add_argument("--d-max", type=float, default=None)
    p.add_argument("--fd-min", type=float, default=None)
    p.add_argument("--fd-max", type=float, default=None)
    p.add_argument("--top-cells", type=int, default=20)
    p.add_argument("--carrier-frequency", type=float, default=None)
    p.add_argument("--csv", default=None, help="Optional per-stand residual CSV")
    args = p.parse_args()

    view_args = types.SimpleNamespace(rdc2_input=args.rdc2_input, integration_time=args.integration_time,
                                      hop_time=args.hop_time, tstart=None, tend=None, duration=None,
                                      nfft_doppler=None, slow_window="hann", gap_fill="zero")
    view = prepare_rdc2_view(view_args)
    meta = view["metadata"]
    fc = args.carrier_frequency or meta.get("tuning_frequency_hz") or meta.get("carrier_frequency_hz")
    if fc is None:
        raise ValueError("No tuning frequency in metadata; pass --carrier-frequency")
    idx, stands, pols, positions, _ = select_array_channels(
        view["channel_stands"], view["channel_pols"], args.pol,
        drop_bad=not args.keep_bad_stands, require_positions=True, operational_only=args.operational_only)
    dsel, _ = contiguous_axis_selection(view["delay_ms"], args.d_min, args.d_max, "delay")
    fsel, _ = contiguous_axis_selection(view["doppler_hz"], args.fd_min, args.fd_max, "Doppler")
    chunks = [ch[0] for _, _, ch in iter_rdc2_view_chunks(view, 16, channel_indices=idx,
                                                         frame_start=args.frame_index, frame_stop=args.frame_index + 1,
                                                         delay_selection=dsel, doppler_selection=fsel)]
    rdc2 = np.concatenate(chunks, axis=2)
    if meta.get("cable_phase_corrected", False):
        print("NOTE: metadata says cable phase is already corrected; the 'none' variant is the corrected data.")
    delays = np.array([a.cable.delay(float(fc)) for a in lwasv_selected_antennas(stands, pols)])
    results, per_stand = calibration_report(rdc2, positions, float(fc), delays, stands, n_cells=args.top_cells)
    print(f"File {args.rdc2_input}, frame {args.frame_index}, fc {float(fc) / 1e6:.6f} MHz, cube {rdc2.shape}")
    print_report(results, per_stand, rdc2.shape[2])
    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(per_stand[0].keys()))
            w.writeheader()
            w.writerows(per_stand)
        print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()

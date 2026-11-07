"""Spectrogram (frequency vs time) or averaged spectrum of raw samples. Input: raw TBN/WAV, or RDC0 with complete sweeps.

--average plots one spectrum averaged over time and the selected channels.
--estimate-lfm fits the sweep rate, bandwidth, start offset and gating to the
spectrogram ridge (snapped to known CODAR settings); --overlay-lfm also draws it.
"""

import numpy as np
from scipy.signal import periodogram

import cli
import lfm_utils as lfm
import plotting_utils as plots
import rdc_utils as ru


def main():
    p = cli.parser(__doc__)
    cli.add_input(p)
    cli.add_plot_output(p)
    cli.add_selection(p, default_stands="first")
    p.add_argument("--window", choices=lfm.WINDOW_CHOICES, default="hann")
    p.add_argument("--window-size", type=int, default=1024)
    p.add_argument("--hop-size", type=int, default=512)
    p.add_argument("--average", action="store_true", help="Plot the time-averaged spectrum instead")
    p.add_argument("--estimate-lfm", action="store_true")
    p.add_argument("--overlay-lfm", action="store_true", help="Draw the estimated sweep (implies --estimate-lfm)")
    args = p.parse_args()
    cli.validate(p, args)
    view = cli.view_from_args(args, raw_stage="rdc0")
    iq, t0, gaps = view.raw_samples()
    fc = float(view.fc or 0.0)
    rows = view.info_rows(t0, t0 + iq.shape[1] / view.fs) + [("Window", f"{args.window}, {args.window_size}")]
    footer = cli.gap_footer(gaps, view.gap_fill)
    if args.average:
        n = iq.shape[1] // args.window_size
        seg = iq[:, :n * args.window_size].reshape(-1, args.window_size)
        seg = seg[np.all(np.isfinite(seg), axis=1)]
        if not len(seg):
            p.error("No complete finite blocks for the averaged spectrum")
        w = lfm.window_from_arg(args.window_size, args.window)
        f, pw = periodogram(seg, fs=view.fs, window=w, detrend=False, return_onesided=False, scaling="spectrum", axis=-1)
        order = np.argsort(f)
        plots.plot_average_spectrum(f[order] + fc, ru.to_db(pw.mean(axis=0)[order]), args.title or "Averaged spectrum",
                                    args.output, args.vmin, args.vmax, rows + [("Blocks", str(len(seg)))], footer)
    else:
        f, t, power = lfm.stft_power(np.nan_to_num(iq), view.fs, args.window, args.window_size, args.hop_size)
        power_db = ru.to_db(power)
        estimate = None
        if args.estimate_lfm or args.overlay_lfm:
            zero = view.utc_at(t0)
            frac = None if zero is None else (zero.microsecond / 1e6)
            estimate = lfm.estimate_lfm_from_stft(f, t0 + t, power_db, frac)
        notes = None
        if estimate:
            notes = "\n".join(filter(None, [
                f"BW (est.): {estimate['bandwidth_hz'] / 1e3:.2f} kHz",
                f"Sweep frequency (est.): {estimate['sweep_frequency_hz']:.4g} Hz" if estimate["sweep_frequency_hz"] else None,
                f"Local offset: {estimate['first_sweep_start_s'] * 1e3:.2f} ms" if estimate["first_sweep_start_s"] is not None else None,
                f"UTC offset: {estimate['utc_offset_s'] * 1e3:.2f} ms" if estimate["utc_offset_s"] is not None else None,
                (f"Gate: {estimate['gate']['frequency_hz']:.4g} Hz, duty {estimate['gate']['duty'] * 100:.0f}%, "
                 f"phase {estimate['gate']['phase_s'] * 1e3:.2f} ms") if estimate.get("gate") else "Gate: not detected"]))
        plots.plot_spectrogram(f, t0 + t, power_db, args.title or "Spectrogram", args.output, args.vmin, args.vmax,
                               rows, notes, footer, estimate if args.overlay_lfm else None)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    cli.run(main)

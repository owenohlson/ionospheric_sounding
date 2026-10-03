"""Check a raw TBN recording, and optionally cut it into time ranges (replaces splitTBNfix.py).

The export commands read raw TBN directly (``lwa_utils.TBNReader``): every frame
is placed by its timetag, so late or shuffled frames land in the right place and
missing captures become NaN (or zero with --gap-fill zero) at the right time.
Trimming is therefore optional. Use this command to:

* see the data quality before processing (``--report-only``): complete,
  partial, overfull (duplicate frames) and missing captures, the number and
  longest of the gaps, late frame groups and bad sync words;
* cut a time range or fixed-length chunks. Output is written in timetag order
  with late frames rejoining their capture, bad-sync frames and partial edge
  captures dropped (all partial captures with --drop-incomplete). Missing
  captures stay missing, so LSL's own reader still refuses files with gaps.

The scan reads frame headers only (no LSL). Output goes next to the input
unless --output is given.
"""

import os
import time

import cli
import lwa_utils


def main():
    p = cli.parser(__doc__)
    p.add_argument("input", help="Raw TBN file")
    p.add_argument("--output", default=None, help="Output TBN file (default: <input>_<tstart>-<tend>s.dat)")
    p.add_argument("--tstart", type=float, default=0.0, help="Start [s] after the first capture")
    p.add_argument("--tend", type=float, default=None, help="End [s] after the first capture (default: end of file)")
    p.add_argument("--chunk", type=float, default=None, help="Write consecutive files of this many seconds instead")
    p.add_argument("--drop-incomplete", action="store_true", help="Drop every capture with missing/extra frames")
    p.add_argument("--report-only", action="store_true", help="Only scan and report; write nothing")
    args = p.parse_args()

    began = time.time()
    scan = lwa_utils.scan_tbn(args.input)
    info = lwa_utils.summarize_tbn_scan(scan)
    print(f"Scanned {scan['n_frames']} frames in {time.time() - began:.1f} s")
    for key, value in info.items():
        print(f"  {key:20s} {value:.6g}" if isinstance(value, float) else f"  {key:20s} {value}")
    if args.report_only:
        return
    end = info["duration_s"] if args.tend is None else min(args.tend, info["duration_s"])
    if end <= args.tstart:
        p.error("--tend must be after --tstart and inside the file")
    step = end - args.tstart if args.chunk is None else args.chunk
    base = os.path.splitext(args.output or args.input)[0]
    t0 = args.tstart
    while t0 < end - 1e-9:
        t1 = min(t0 + step, end)
        out = args.output if (args.output and args.chunk is None) else f"{base}_{t0:.0f}-{t1:.0f}s.dat"
        captures, frames = lwa_utils.write_tbn_range(args.input, out, scan, t0, t1, args.drop_incomplete)
        print(f"Wrote {out}: {captures} captures, {frames} frames ({t0:.3f}-{t1:.3f} s)")
        t0 = t1


if __name__ == "__main__":
    cli.run(main)

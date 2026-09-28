"""Repair and trim a raw TBN recording before export (replaces splitTBNfix.py).

1. Skip forward to the first complete capture (a frame from stand 1, pol 0).
2. Probe captures until the stream metadata is readable with a ~100 kHz sample rate.
3. Copy whole captures for [--tstart, --tend), checking that every capture starts
   on stand 1 / pol 0 and resynchronising (dropping the partial capture) if not.

Missing or corrupted samples are not reconstructed; later stages fill timing
gaps with NaN/zero. Output goes next to the input unless --output is given.
"""

import os
import time

import cli


def main():
    p = cli.parser(__doc__)
    p.add_argument("input", help="Raw TBN file")
    p.add_argument("--output", default=None, help="Output TBN file (default: <input>_<tstart>-<tend>s.dat)")
    p.add_argument("--tstart", type=float, default=0.0, help="Seconds to skip from the start of the file")
    p.add_argument("--tend", type=float, default=None, help="End time [s] from the start of the file (default: end)")
    p.add_argument("--chunk", type=float, default=None,
                   help="Instead of one output, write consecutive files of this many seconds")
    p.add_argument("--no-verify", action="store_true", help="Copy captures without checking frame alignment")
    args = p.parse_args()
    from lsl.reader import tbn
    import tbn_utils

    size = os.path.getsize(args.input)
    with open(args.input, "rb") as fh:
        skipped = tbn_utils.find_capture_start(fh)
        if skipped:
            print(f"Skipped {skipped} frame(s) to reach the first capture boundary")
        pos, rate, per_capture = tbn_utils.probe_stream_layout(fh)
        capture_s = tbn_utils.TBN_SAMPLES_PER_FRAME / rate
        available = (size - pos) // (tbn.FRAME_SIZE * per_capture)
        print(f"Sample rate {rate / 1e3:.3f} kHz, {per_capture} frames per capture, "
              f"{available} captures ({available * capture_s:.3f} s) from byte {pos}")
        first = int(round(args.tstart / capture_s))
        last = available if args.tend is None else min(available, int(round(args.tend / capture_s)))
        if last <= first:
            p.error("--tend must be after --tstart and inside the file")
        fh.seek(pos + first * tbn.FRAME_SIZE * per_capture)
        step = last - first if args.chunk is None else max(1, int(round(args.chunk / capture_s)))
        base = os.path.splitext(args.input)[0]
        for k, start in enumerate(range(first, last, step)):
            n = min(step, last - start)
            t0, t1 = start * capture_s, (start + n) * capture_s
            out = args.output if (args.output and args.chunk is None) else (
                (os.path.splitext(args.output)[0] if args.output else base) + f"_{t0:.0f}-{t1:.0f}s.dat")
            began = time.time()
            with open(out, "wb") as fo:
                stats = tbn_utils.copy_captures(fh, fo, n, per_capture, verify=not args.no_verify)
            mb = os.path.getsize(out) / 1024 ** 2
            print(f"Wrote {out}: {stats['captures']} captures ({t0:.3f}-{t1:.3f} s), "
                  f"{stats['resyncs']} resync(s), {mb:.1f} MB in {time.time() - began:.1f} s")


if __name__ == "__main__":
    main()

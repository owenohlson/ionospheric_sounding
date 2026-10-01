"""TBN frame scanning/trimming on a synthetic file (no LSL needed)."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

import helpers  # noqa: F401
import lwa_utils

STEP = int(512 * 196e6 / 100e3)


def frame(tag, row, fill):
    f = np.zeros(lwa_utils.TBN_FRAME_SIZE, np.uint8)
    f[0:4] = [0xDE, 0xC0, 0xDE, 0x5C]
    f[12:14] = np.frombuffer(np.array([row + 1], ">u2").tobytes(), np.uint8)
    f[16:24] = np.frombuffer(np.array([tag], ">u8").tobytes(), np.uint8)
    f[24:] = fill
    return f


class TBNScanTests(unittest.TestCase):
    def test_scan_and_trim(self):
        rng = np.random.default_rng(0)
        per, frames = 16, []
        for g in range(50):
            if g == 20:
                continue                                   # missing capture (timetag jump)
            rows = rng.permutation(per)[: per - 1 if g == 30 else per]   # shuffled; one short capture
            frames += [frame(10 ** 12 + g * STEP, r, g % 256) for r in rows]
        frames = frames[5:]                                # file starts mid-capture
        frames[100][0] = 0                                 # one corrupted sync word
        with tempfile.TemporaryDirectory() as d:
            src, out = Path(d) / "in.dat", Path(d) / "out.dat"
            np.concatenate(frames).tofile(src)
            scan = lwa_utils.scan_tbn(src, block_frames=37)
            info = lwa_utils.summarize_tbn_scan(scan)
            self.assertAlmostEqual(info["sample_rate_hz"], 100e3)
            self.assertEqual((info["frames_per_capture"], info["missing_captures"], info["bad_sync_frames"],
                              info["incomplete_captures"]), (16, 1, 1, 2))
            captures, n = lwa_utils.write_tbn_range(src, out, scan, 0.0, 0.1)
            self.assertEqual(captures, 19)                 # partial first capture dropped
            self.assertEqual(n, 19 * 16 - 1)               # bad-sync frame dropped
            again = lwa_utils.summarize_tbn_scan(lwa_utils.scan_tbn(out))
            self.assertEqual(again["bad_sync_frames"], 0)


if __name__ == "__main__":
    unittest.main()

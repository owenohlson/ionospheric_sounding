"""TBN frame scanning/trimming on a synthetic file (no LSL needed)."""

import tempfile
import unittest
from pathlib import Path

from dataclasses import replace as dataclasses_replace

import numpy as np

import helpers  # noqa: F401
import lwa_utils

STEP = int(512 * 196e6 / 100e3)


TUNING_WORD = int(round(38.1e6 / 196e6 * 2 ** 32))


def frame(tag, row, fill, tuning_word=TUNING_WORD):
    """One TBN frame; ``fill`` is a scalar byte or 512 complex samples (int8 I/Q)."""
    f = np.zeros(lwa_utils.TBN_FRAME_SIZE, np.uint8)
    f[0:4] = [0xDE, 0xC0, 0xDE, 0x5C]
    f[8:12] = np.frombuffer(np.array([tuning_word], ">u4").tobytes(), np.uint8)
    f[12:14] = np.frombuffer(np.array([row + 1], ">u2").tobytes(), np.uint8)
    f[16:24] = np.frombuffer(np.array([tag], ">u8").tobytes(), np.uint8)
    if np.ndim(fill):
        f[24:] = np.column_stack([fill.real, fill.imag]).astype(np.int8).ravel().view(np.uint8)
    else:
        f[24:] = fill
    return f


def messy_tbn(path, per=16, captures=60, seed=0, missing_capture=25, partial=(40, 3), stray=False):
    """Synthetic TBN with shuffled frames, a late group, a duplicate, a missing and a partial capture.

    Returns the true samples (per rows, captures * 512) with NaN where frames are absent.
    """
    rng = np.random.default_rng(seed)
    t0 = 1702137594 * 196_000_000 + 37 * STEP           # 2023-12-09 15:59:54 plus a few captures
    truth = (rng.integers(-60, 60, (per, captures * 512)) + 1j * rng.integers(-60, 60, (per, captures * 512)))
    truth = truth.astype(np.complex64)
    groups = []
    for g in range(captures):
        rows = rng.permutation(per)
        groups.append([(g, r) for r in rows])
    groups[missing_capture] = []
    groups[partial[0]] = [x for x in groups[partial[0]] if x[1] != partial[1]]
    late = [x for x in groups[10] if x[1] in (1, 2)]    # rows 1, 2 of capture 10 arrive after capture 11
    groups[10] = [x for x in groups[10] if x[1] not in (1, 2)]
    groups[11] = groups[11] + late
    groups[12] = groups[12] + [groups[12][0]]           # duplicated frame
    frames = [frame(t0 + g * STEP, r, truth[r, g * 512:(g + 1) * 512]) for grp in groups for g, r in grp]
    if stray:   # leftovers of a previous setup: other tuning, off the timetag grid, 5.8 s earlier
        other = int(round(13.5e6 / 196e6 * 2 ** 32))
        frames = [frame(t0 - int(5.8 * 196e6) - STEP // 2, r, 7, other) for r in range(5)] + frames
    np.concatenate(frames).tofile(path)
    truth[:, missing_capture * 512:(missing_capture + 1) * 512] = np.nan
    truth[partial[1], partial[0] * 512:(partial[0] + 1) * 512] = np.nan
    return truth, t0


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
                              info["partial_captures"]), (16, 1, 1, 2))
            captures, n = lwa_utils.write_tbn_range(src, out, scan, 0.0, 0.1)
            self.assertEqual(captures, 19)                 # partial first capture dropped
            self.assertEqual(n, 19 * 16 - 1)               # bad-sync frame dropped
            again = lwa_utils.summarize_tbn_scan(lwa_utils.scan_tbn(out))
            self.assertEqual(again["bad_sync_frames"], 0)


    def test_reader_places_frames_by_timetag(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "messy.dat"
            truth, t0 = messy_tbn(src)
            scan = lwa_utils.scan_tbn(src, block_frames=100)
            q = lwa_utils.summarize_tbn_scan(scan)
            self.assertEqual((q["missing_captures"], q["partial_captures"], q["gaps"], q["out_of_order_groups"],
                              q["overfull_captures"]), (1, 1, 1, 1, 1))
            r = lwa_utils.TBNReader(src, scan)
            self.assertAlmostEqual(r.tuning_frequency_hz, 38.1e6, delta=1.0)
            self.assertEqual(r.utc_at(0).isoformat(), "2023-12-09T15:59:54.189440+00:00")
            rows = np.array([0, 3, 1, 15])
            got = np.concatenate([b for b, _ in r.read_blocks(rows, 0.0, None, chunk_seconds=0.03)], axis=1)
            np.testing.assert_array_equal(got, truth[rows])
            part = np.concatenate([b for b, _ in r.read_blocks(rows, 0.1, 0.05)], axis=1)
            first = int(np.floor(0.1 * 100e3 / 512))
            np.testing.assert_array_equal(part, truth[rows, first * 512:first * 512 + part.shape[1]])
            out = Path(d) / "fixed.dat"
            lwa_utils.write_tbn_range(src, out, scan)
            fixed = lwa_utils.summarize_tbn_scan(lwa_utils.scan_tbn(out))
            self.assertEqual(fixed["out_of_order_groups"], 0)        # late frames rejoin their capture
            again = np.concatenate([b for b, _ in lwa_utils.TBNReader(out).read_blocks(rows)], axis=1)
            np.testing.assert_array_equal(again, truth[rows])


    def test_leftover_frames_from_another_setup_are_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "stray.dat"
            truth, t0 = messy_tbn(src, stray=True)
            q = lwa_utils.summarize_tbn_scan(lwa_utils.scan_tbn(src))
            self.assertEqual((q["ignored_frames"], q["ignored_tunings_mhz"], q["missing_captures"], q["gaps"]),
                             (5, "13.5", 1, 1))
            self.assertAlmostEqual(q["tuning_mhz"], 38.1, places=3)
            r = lwa_utils.TBNReader(src)
            self.assertEqual(r.utc_at(0).isoformat(), "2023-12-09T15:59:54.189440+00:00")
            got = np.concatenate([b for b, _ in r.read_blocks(np.arange(16))], axis=1)
            np.testing.assert_array_equal(got, truth)

    def test_export_rdc0_from_messy_tbn(self):
        import rdc_io
        import rdc_utils
        station = helpers.FakeStation(n=24)
        with station.patch(), tempfile.TemporaryDirectory() as d:
            src, out = Path(d) / "messy.dat", Path(d) / "r0.h5"
            truth, t0 = messy_tbn(src, per=48, captures=420, seed=2, missing_capture=200, partial=(300, 46))
            s = rdc_utils.ExportSettings(sweep_frequency=2.0, stands="1-3", pol="both", gap_fill="nan",
                                         compression="none")
            import contextlib, io
            with contextlib.redirect_stdout(io.StringIO()):
                rdc_utils.export(str(src), "rdc0", str(out), s)
            f = rdc_io.RDCFile(out)
            data = f.read("rdc0")                                  # (adc, pulses, channels)
            digs = f.channel_digitizers("rdc0")
            stands, pols = f.channel_ids("rdc0")
            for k, dig in enumerate(digs):                         # identity follows the station's digitizer map
                a = station.lookup[(int(stands[k]), int(pols[k]))]
                self.assertEqual(a.digitizer, dig)
            skip = int(f.meta["start_offset_samples"])
            n = data.shape[0] * data.shape[1]
            for k, dig in enumerate(digs):
                expect = truth[dig - 1, skip:skip + n].reshape(data.shape[1], data.shape[0]).T
                np.testing.assert_array_equal(data[..., k], expect)
            self.assertTrue(np.isnan(data).any())                  # the gap stays a gap, in the right place
            # streaming raw -> RDC1 matches RDC0 -> RDC1
            s1 = rdc_utils.ExportSettings(bandwidth=-99.259377e3, d_min=10, d_max=20, compression="none")
            raw1 = dataclasses_replace(s, bandwidth=-99.259377e3, d_min=10, d_max=20)
            with contextlib.redirect_stdout(io.StringIO()):
                rdc_utils.export(str(out), "rdc1", str(Path(d) / "via0.h5"), s1)
                rdc_utils.export(str(src), "rdc1", str(Path(d) / "direct.h5"), raw1)
            a = rdc_io.RDCFile(Path(d) / "direct.h5").read("rdc1")
            b = rdc_io.RDCFile(Path(d) / "via0.h5").read("rdc1")
            np.testing.assert_allclose(a, b, atol=1e-3, equal_nan=True)
            frac = ((t0 + skip * 1960) % 196_000_000) / 196e6       # first sweep on a half-second boundary
            self.assertAlmostEqual(frac % 0.5, 0.0, places=6)


if __name__ == "__main__":
    unittest.main()

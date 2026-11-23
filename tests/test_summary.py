"""Processing summary: shared log, replacement on re-runs, merge into one row per signal, recording commands."""

import contextlib
import csv
import io
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

import helpers
import summary_utils as su


def rows_of(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = Path(self.tmp.name)

    def test_paths(self):
        self.assertEqual(su.summary_path(self.d / "plots" / "a.png"), str(self.d / "plots" / su.LOG_NAME))
        self.assertEqual(su.summary_path(self.d / "plots" / "video_dir"), str(self.d / "plots" / su.LOG_NAME))
        self.assertIsNone(su.summary_path("a.png", "none"))
        self.assertEqual(su.file_id("../data/060287_010274515_rdc1.h5"), "060287_010274515")
        self.assertEqual(su.file_id("/raw/060287_010274515"), "060287_010274515")

    def test_record_replace_and_merge(self):
        log = str(self.d / su.LOG_NAME)
        su.record(log, "scan_lfm", "/raw/060287_010274515", self.d / "scan.png", [
            dict(kind="waveform", bandwidth_hz=-99259.2, sweep_rate_hz=2, delay_ms=15.70, snr_db=39.5,
                 tuning_mhz=13.5, candidate_stations="ARG1 PTC1"),
            dict(kind="waveform", bandwidth_hz=-51467.5, sweep_rate_hz=2, delay_ms=25.59, snr_db=29.5)])
        dd = [dict(kind="echo", bandwidth_hz=-99259.377, sweep_rate_hz=2, delay_ms=15.60 + 0.01 * i, doppler_hz=-0.2,
                   snr_db=30 + i, frame_t0_s=i * 30, look_az_deg=-85, look_el_deg=20) for i in range(3)]
        dd += [dict(kind="doa", bandwidth_hz=-99259.377, sweep_rate_hz=2, delay_ms=499.9, az_deg=a, el_deg=20,
                    frame_t0_s=0, station="SLO: ARG1/PTC1", network="SLO", station_lat=34.58, station_lon=-120.65,
                    hops=1, virtual_height_km=250, layer="F") for a in (179.0, -179.0)]
        su.record(log, "plot_delay_doppler", "../data/060287_010274515_rdc1.h5", self.d / "dd.png", dd)
        merged = rows_of(su.merged_path(log))
        self.assertEqual(len(merged), 3)        # 15.7 ms echo, 499.9 ms sky-map cell, -51.5 kHz waveform
        main = next(m for m in merged if m["scan_snr_db"] == "39.5")
        self.assertEqual((main["bandwidth_hz"], main["dd_snr_db"], main["candidate_stations"]),
                         ("-99259.2", "31", "ARG1 PTC1"))
        self.assertAlmostEqual(float(main["az_deg"]), -85)            # beam look directions as the fallback
        sky = next(m for m in merged if m["station"])
        self.assertAlmostEqual(abs(float(sky["az_deg"])), 180, places=6)   # circular median across +-180
        self.assertEqual(sky["station_lat"], "34.58")
        # re-running a command replaces its rows
        su.record(log, "plot_delay_doppler", "../data/060287_010274515_rdc1.h5", self.d / "dd.png", dd[:1])
        self.assertEqual(sum(r["script"] == "plot_delay_doppler" for r in rows_of(log)), 1)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            su.main([log, "--min-db", "35"])
        self.assertIn("-99259.2", out.getvalue())
        self.assertNotIn("-51467.5", out.getvalue())

    def test_copies_fold_into_their_echo(self):
        log = str(self.d / su.LOG_NAME)
        rows = [dict(kind="echo", bandwidth_hz=-99259.377, sweep_rate_hz=2, sample_rate_hz=1e5, delay_ms=d,
                     doppler_hz=-0.2, snr_db=db, frame_t0_s=0) for d, db in ((15.7, 30), (10.52, 27), (20.88, 26.5),
                                                                                  (19.43, 18))]
        rows.append(dict(kind="doa", bandwidth_hz=-99259.377, sweep_rate_hz=2, delay_ms=10.52, doppler_hz=-0.2,
                         az_deg=-85, el_deg=20, station="SLO: ARG1/PTC1", frame_t0_s=0))
        su.record(log, "plot_delay_doppler", "060287_010274515_rdc1.h5", self.d / "dd.png", rows)
        merged = rows_of(su.merged_path(log))
        self.assertEqual(len(merged), 1)
        m = merged[0]
        self.assertAlmostEqual(float(m["delay_ms"]), 15.7)
        self.assertEqual(m["copies"].count("gating"), 2)
        self.assertIn("sweep wrap", m["copies"])
        self.assertEqual((m["az_deg"], m["station"]), ("-85", "SLO: ARG1/PTC1"))   # taken from its copy

    def test_delay_doppler_findings_report_local_snr(self):
        """SNR = (P - N)/N with N the local mean noise, not the map median (the floor varies with delay)."""
        import plot_delay_doppler as pdd
        rng = np.random.default_rng(3)
        delay = np.linspace(0, 100, 1000)
        floor = 10 ** (np.where(delay < 50, 6.0, 0.0) / 10)            # +6 dB near the echo, as on a real profile
        p = floor[:, None] * rng.exponential(size=(1000, 64))
        p[155, 20] = 101 * floor[155]                                   # 20 dB SNR at 15.5 ms
        info = dict(_period=500.0, _offset=0.0, bandwidth_hz=-99259.377, sweep_rate_hz=2.0, sample_rate_hz=1e5)
        power_db = 10 * np.log10(p)
        rows = pdd.echo_findings(power_db, delay, np.linspace(-1, 1, 64), info, n=1)
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["delay_ms"], delay[155])
        self.assertAlmostEqual(rows[0]["snr_db"], 20.0, delta=0.6)
        self.assertFalse(any(k.startswith("_") for k in rows[0]))
        doa = pdd.doa_findings([(-85.0, 21.0)], [(155, 20)], power_db, delay, np.linspace(-1, 1, 64), info, [])
        self.assertAlmostEqual(doa[0]["snr_db"], rows[0]["snr_db"], delta=0.3)

    def test_a_long_video_gives_a_few_rows(self):
        """60 frames x (3 echoes + their pairs of gating copies + a noise cell + 2 sky-map peaks) -> 3 rows."""
        log = str(self.d / su.LOG_NAME)
        rng = np.random.default_rng(1)
        rows = []
        for i in range(60):
            base = dict(bandwidth_hz=-99259.377, sweep_rate_hz=2, sample_rate_hz=1e5, frame_t0_s=10 * i,
                        frame_t1_s=10 * i + 30)
            for d, db, dop in ((15.70, 30, -0.2), (19.15, 26, 0.1), (14.54, 24, 0.3)):
                d += 0.05 * np.sin(i / 9)                              # a TID moving the echo a little
                rows.append(dict(base, kind="echo", delay_ms=d, snr_db=db + rng.normal(), doppler_hz=dop,
                                 look_az_deg=-85, look_el_deg=20))
                rows += [dict(base, kind="echo", delay_ms=d + s, snr_db=db - 3, doppler_hz=dop) for s in (-5.18, 5.18)]
            rows.append(dict(base, kind="echo", delay_ms=rng.uniform(30, 400), snr_db=12, doppler_hz=1.0))
            rows += [dict(base, kind="doa", delay_ms=15.70, doppler_hz=-0.2, az_deg=-84 + rng.normal(), el_deg=21,
                          station="SLO: ARG1/PTC1", candidate_stations="ARG1 PTC1"),
                     dict(base, kind="doa", delay_ms=19.15, doppler_hz=0.1, az_deg=150, el_deg=30)]
        su.record(log, "sliding_delay_doppler", "060287_010275465_rdc1.h5", self.d / "video", rows, total_frames=60)
        logged = rows_of(log)
        np.testing.assert_allclose(sorted(float(r["delay_ms"]) for r in logged), [14.54, 15.70, 19.15], atol=0.06)
        main = next(r for r in logged if round(float(r["delay_ms"]), 1) == 15.7)
        self.assertEqual((main["n_frames"], main["station"]), ("60", "SLO: ARG1/PTC1"))
        self.assertAlmostEqual(float(main["az_deg"]), -84, delta=0.5)
        self.assertIn("gating +5.18 ms", main["copies"])
        merged = rows_of(su.merged_path(log))
        self.assertEqual(len(merged), 3)
        self.assertEqual(merged[0]["frames"], "60")
        # a log written before consolidation (one row per cell per frame) is compacted at the next write
        with open(log, "w", newline="") as fh:
            w = csv.DictWriter(fh, su.LOG_COLUMNS)
            w.writeheader()
            for r in rows:
                w.writerow({c: su._fmt(r.get(c)) for c in su.LOG_COLUMNS} | dict(
                    script="sliding_delay_doppler", input="/x/060287_010275465_rdc1.h5", output="/x/video",
                    file="060287_010275465"))
        su.record(log, "scan_lfm", "/raw/060287_010275465", self.d / "scan.png", [
            dict(kind="waveform", bandwidth_hz=-99259.2, sweep_rate_hz=2, delay_ms=15.70, snr_db=39.5)])
        self.assertEqual(len(rows_of(log)), 4)

    def test_delay_doppler_sky_map_and_station_map_record(self):
        import plot_delay_doppler
        import plot_station_map
        from test_array import received, write_rdc0
        from test_codar_stations import TABLE
        table = self.d / "c.csv"
        table.write_text(TABLE.replace("13.49971", "24.55000").replace("13.50000", "24.55000"))
        station = helpers.FakeStation(n=48, seed=3)
        with station.patch():
            x = received(station.positions, -85.2, 21.0, 0.00437, station.delays[:, 0], snr_db=15, sweeps=16)
            write_rdc0(self.d / "060287_010274515.h5", x, np.arange(1, 49), np.zeros(48, int))
            args = plot_delay_doppler.build_parser().parse_args([
                str(self.d / "060287_010274515.h5"), "--pol", "x", "--beamformer", "conventional", "--d-min", "3.5",
                "--d-max", "5.5", "--angle-output", str(self.d / "sky.png"), "--angle-cells", "1",
                "--output", str(self.d / "dd.png"), "--codar-stations", str(table)])
            with contextlib.redirect_stdout(io.StringIO()):
                plot_delay_doppler.run(args)
        log = self.d / su.LOG_NAME
        logged = rows_of(log)
        self.assertEqual({r["kind"] for r in logged}, {"echo"})      # sky-map peaks join their echo's row
        doa = next(r for r in logged if r["az_deg"])
        self.assertAlmostEqual(float(doa["az_deg"]), -85.2, delta=3)
        self.assertEqual(doa["station"], "SLO: ARG1/NEW1/PTC1")
        self.assertAlmostEqual(float(doa["delay_ms"]), 4.37, delta=0.05)
        self.assertTrue(doa["reflection_points"])
        with contextlib.redirect_stdout(io.StringIO()):
            plot_station_map.main(["--stations", "PCYC@11.78", "--output", str(self.d / "map.png")])
        merged = rows_of(su.merged_path(str(log)))
        self.assertTrue(any(m["station"].startswith("SLO") and m["file"] == "060287_010274515" for m in merged))
        pcyc = next(m for m in merged if m["station"] == "PCYC")
        self.assertEqual((pcyc["hops"], pcyc["file"]), ("1", ""))


if __name__ == "__main__":
    unittest.main()

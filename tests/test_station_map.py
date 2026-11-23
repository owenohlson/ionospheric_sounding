"""Station map: outline file building (shapefile/GeoJSON), great circles, projection, hop points."""

import json
import struct
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

import helpers  # noqa: F401
import map_utils as mu
import plot_station_map as psm
import codar_stations as cs


def shp_bytes(lines, kind=3):
    """A minimal .shp holding polylines (``kind`` 3) or polygons (5)."""
    recs = b""
    for i, ln in enumerate(lines, 1):
        pts = np.asarray(ln, "<f8")
        body = struct.pack("<i4d2i", kind, *pts[:, 0].min(keepdims=True), *pts[:, 1].min(keepdims=True),
                           pts[:, 0].max(), pts[:, 1].max(), 1, len(pts)) + struct.pack("<i", 0) + pts.tobytes()
        recs += struct.pack(">2i", i, len(body) // 2) + body
    header = struct.pack(">7i", 9994, 0, 0, 0, 0, 0, (100 + len(recs)) // 2) + struct.pack("<2i", 1000, kind) \
        + b"\0" * 64
    return header + recs


class MapTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = Path(self.tmp.name)

    def test_build_and_load_outlines(self):
        coast = [[(-125.0, 48.0), (-124.0, 40.0), (-117.123, 32.5)], [(-80.0, 25.0), (10.0, 25.0)]]
        with zipfile.ZipFile(self.d / "ne_50m_coastline.zip", "w") as z:
            z.writestr("ne_50m_coastline.shp", shp_bytes(coast))
        states = {"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": {
            "type": "MultiLineString", "coordinates": [[[-109.05, 31.33], [-109.05, 37.0], [-109.05, 41.0]]]}}]}
        (self.d / "ne_50m_admin_1_states_provinces_lines.geojson").write_text(json.dumps(states))
        out = self.d / "outlines.npz"
        counts = mu.build_outlines([str(self.d / "ne_50m_coastline.zip"),
                                    str(self.d / "ne_50m_admin_1_states_provinces_lines.geojson")], str(out))
        self.assertEqual(counts["coast"], 3 + 2)             # second line is cut at the region's east edge
        layers = mu.load_outlines(str(out))
        self.assertAlmostEqual(np.nanmax(layers["coast"][:, 0]), -48.0, places=2)
        self.assertAlmostEqual(layers["coast"][2, 1], 32.5, places=2)        # rounded to 0.01 deg
        self.assertEqual(int(np.isnan(layers["states"][:, 0]).sum()), 1)
        self.assertEqual(mu.load_outlines(str(self.d / "missing.npz")), {})
        with self.assertRaises(ValueError):
            mu._layer_of("rivers.zip")

    def test_great_circle_points(self):
        lat, lon = mu.intermediate(0.0, 0.0, 0.0, 90.0, [0.0, 0.5, 1.0])
        np.testing.assert_allclose(lon, [0, 45, 90], atol=1e-9)
        # the LWA1-PCYC midpoint matches the professor's 32.53 N, 98.24 W
        pcyc = next(c for c in cs.load_tables() if c.code == "PCYC")
        (sky_lat, sky_lon), ground = mu.hop_points(*cs.RECEIVERS["lwa1"], pcyc.lat_deg, pcyc.lon_deg, 1)
        self.assertAlmostEqual(sky_lat[0], 32.53, delta=0.02)
        self.assertAlmostEqual(sky_lon[0], -98.24, delta=0.02)
        self.assertEqual(len(ground[0]), 0)
        (sky_lat, _), (g_lat, _) = mu.hop_points(0.0, 0.0, 0.0, 90.0, 2)
        self.assertEqual((len(sky_lat), len(g_lat)), (2, 1))

    def test_projection_is_conformal_near_the_centre(self):
        proj = mu.LambertConformal.for_extent(-120, -80, 25, 45)
        x0, y0 = proj(-100, 35)
        xe, _ = proj(-99.99, 35)
        _, yn = proj(-100, 35.01)
        scale_e = (xe - x0) / np.radians(0.01) / np.cos(np.radians(35))
        scale_n = (yn - y0) / np.radians(0.01)
        self.assertAlmostEqual(scale_e / scale_n, 1.0, places=3)

    def test_map_rows_and_plot(self):
        rows = psm.path_rows(cs.load_tables(), psm.parse_stations("PCYC@11.78,FOCF@10.3,SCRZ:2,SCI1"),
                             *cs.RECEIVERS["lwa1"])
        by = {r["code"]: r for r in rows}
        self.assertEqual((by["PCYC"]["hops"], by["FOCF"]["hops"], by["SCRZ"]["hops"], by["SCI1"]["hops"]),
                         (1, 2, 2, 1))
        self.assertAlmostEqual(by["FOCF"]["height"], 212, delta=2)
        self.assertEqual(len(by["FOCF"]["ground"]), 1)
        self.assertEqual(psm.parse_points("A:1,2;B:3,-4"), [("A", 1.0, 2.0), ("B", 3.0, -4.0)])
        out = self.d / "map.png"
        psm.main(["--stations", "PCYC@11.78,FOCF@10.3", "--band", "25.6", "--bandwidth", "-150727.2",
                  "--date", "2023-12-09", "--points", "AU930:30.4,-97.7", "--output", str(out),
                  "--csv", str(self.d / "paths.csv")])
        self.assertTrue(out.exists())
        self.assertIn("PCYC", (self.d / "paths.csv").read_text())


    def test_elevations_from_the_processing_summary(self):
        """--from-summary: stations and elevations come from signals that array processing gave a direction."""
        import summary_utils as su
        stations = cs.load_tables()
        arg1 = next(c for c in stations if c.code == "ARG1")
        bearing, _ = cs.bearing_distance(*cs.RECEIVERS["lwa1"], arg1.lat_deg, arg1.lon_deg)
        log = self.d / su.LOG_NAME
        base = dict(file="060287_010274515", utc_start="2023-12-09T16:00:00+00:00", tuning_mhz=13.5,
                    bandwidth_hz=-99259.377, sweep_rate_hz=2, sample_rate_hz=1e5)
        su.record(str(log), "sliding_delay_doppler", "/x/060287_010274515_rdc1.h5", str(self.d / "video"), [
            dict(base, kind="echo", delay_ms=15.7, snr_db=30, doppler_hz=-0.2, az_deg=bearing + 1, el_deg=20,
                 n_frames=40),
            dict(base, kind="echo", delay_ms=60.0, snr_db=25, doppler_hz=0.1, look_az_deg=bearing, look_el_deg=40,
                 n_frames=40),
            dict(base, kind="echo", delay_ms=90.0, snr_db=25, doppler_hz=0.1, n_frames=40)])    # no direction
        su.record(str(log), "scan_lfm", "/raw/060287_010274635", str(self.d / "scan.png"), [
            dict(base, file="060287_010274635", kind="waveform", delay_ms=15.0, snr_db=35)])     # no direction
        out = self.d / "map.png"
        psm.main(["--from-summary", "--output", str(out), "--csv", str(self.d / "paths.csv")])
        self.assertTrue(out.exists())
        merged = su.merge(su.read_log(str(log)))
        sigs = psm.summary_signals(merged, stations, *cs.RECEIVERS["lwa1"])
        self.assertEqual(sorted(s["el"] for s in sigs), [20, 40])
        self.assertIn("ARG1", sigs[0]["codes"])
        rows = psm.summary_paths(sigs, stations, *cs.RECEIVERS["lwa1"])
        arg = [r for r in rows if r["code"] == "ARG1"]
        self.assertEqual(sorted(r["elevation"] for r in arg), [20, 40])           # one path per hop count
        self.assertEqual(sorted(r["hops"] for r in arg), [1, 2])
        self.assertEqual(psm.summary_signals(merged, stations, *cs.RECEIVERS["lwa1"], files="4635"), [])
        with self.assertRaises(SystemExit) as cm:                                  # no directions: skipped (3)
            psm.main(["--from-summary", str(log), "--files", "060287_010274635", "--skip-empty",
                      "--output", str(self.d / "none.png")])
        self.assertEqual(cm.exception.code, 3)
        self.assertFalse((self.d / "none.png").exists())
        self.assertEqual(len(psm.summary_signals(merged, stations, *cs.RECEIVERS["lwa1"], min_snr=28)), 1)
        paths = [r for r in su.read_log(str(log)) if r["kind"] == "path" and r["station"] == "ARG1"]
        self.assertTrue(paths and "060287_010274515" in paths[0]["notes"])
        # with ionosonde data the hop count comes from the measured profile at the file's time
        import test_ionosonde
        (self.d / "giro").mkdir()
        (self.d / "giro" / "BC840_2023-12-09.txt").write_text(test_ionosonde.fastchar(
            [(f"16:{m:02d}", 100, 9.0, 3.5, 235.0) for m in range(0, 30, 5)]))
        iono = __import__("ionosonde").load_soundings(str(self.d / "giro"))
        rows = psm.summary_paths(sigs, stations, *cs.RECEIVERS["lwa1"], iono=iono)
        arg = next(r for r in rows if r["code"] == "ARG1" and r["elevation"] == 20)
        self.assertEqual((arg["hops"], arg["ionosonde"]), (1, "BC840"))
        self.assertTrue(arg["how"].startswith("ionosonde BC840"))
        psm.main(["--from-summary", "--iono-data", str(self.d / "giro"), "--output", str(self.d / "map2.png"),
                  "--summary", "none"])
        self.assertTrue((self.d / "map2.png").exists())


if __name__ == "__main__":
    unittest.main()

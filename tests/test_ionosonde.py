"""Ionosonde-based hop counts: GIRO FastChar parsing, profile, virtual height, oblique solutions."""

import math
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import helpers  # noqa: F401
import codar_stations as cs
import ionosonde as io

HEADER = """# Global Ionospheric Radio Observatory (GIRO)
# Location: GEO ( 40.0 N   254.7 E ), URSI-Code BC840, BOULDER
# Instrument: Ionosonde, Model: [DGS-256]
#
# Time                    CS   foF2 QD  foF1 QD   foE QD  foEs QD MUF(D) QD M(D) QD   hmF2 QD   hmF1 QD    hmE QD    B0 QD   B1 QD
"""


def fastchar(rows):
    lines = [f"2023-12-09T{t}:05.000Z {cs_:3d} {f2:6.3f} //   --- __  2.80 //  2.80 // 32.000 // {md:4.2f} // "
             f"{hm:6.1f} //    --- __  108.0 //  60.0 // 1.80 //" for t, cs_, f2, md, hm in rows]
    return HEADER + "\n".join(lines) + "\n"


class IonosondeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = Path(self.tmp.name)
        rows = [(f"16:{m:02d}", 100, 9.0, 3.5, 235.0) for m in range(0, 60, 5)]
        rows[6] = ("16:30", 100, 9.0, 2.5, 390.0)              # a mis-scaled trace
        rows[8] = ("16:40", 30, 9.0, 3.5, 235.0)               # low confidence
        (self.d / "BC840_2023-12-09.txt").write_text(fastchar(rows))

    def test_read_and_clean(self):
        s = io.load_soundings(str(self.d))["BC840"]
        self.assertEqual((s.name, s.lat_deg, round(s.lon_deg, 1)), ("BOULDER", 40.0, -105.3))
        self.assertEqual(len(s.dropped), 2)
        p = s.at(datetime(2023, 12, 9, 16, 30, tzinfo=timezone.utc))
        self.assertAlmostEqual(p["hmF2"], 235.0)                # interpolated across the dropped row
        self.assertTrue(math.isnan(p["foF1"]))
        self.assertTrue(math.isnan(s.at(datetime(2023, 12, 9, 18, 0, tzinfo=timezone.utc))["foF2"]))

    def test_virtual_height(self):
        p = dict(foF2=9.0, hmF2=240.0, foE=3.0, hmE=110.0, B0=60.0, B1=2.0)
        h = [io.virtual_height(f, p) for f in (2.0, 5.0, 8.0, 8.9)]
        self.assertLess(h[0], 140)                              # E layer
        self.assertTrue(140 < h[1] < h[2] < h[3])               # F trace rises towards foF2
        self.assertTrue(math.isnan(io.virtual_height(9.5, p)))  # penetrates
        # thin linear layer: fN^2 = foF2^2 (h - h0) / L gives h' = h0 + 2 L f^2 / foF2^2 exactly
        lin = dict(foF2=9.0, hmF2=300.0)
        f2 = lambda h, q: np.clip((np.asarray(h) - 200.0) / 100.0, 0, 1) * 81.0         # noqa: E731
        orig = io.plasma_frequency_sq
        io.plasma_frequency_sq = f2
        try:
            self.assertAlmostEqual(io.virtual_height(6.0, lin), 200 + 2 * 100 * 36 / 81, delta=0.5)
        finally:
            io.plasma_frequency_sq = orig

    def test_hop_counts_follow_the_measured_ionosphere(self):
        soundings = io.load_soundings(str(self.d))
        when = datetime(2023, 12, 9, 16, 20, tzinfo=timezone.utc)
        lwa = cs.RECEIVERS["lwa1"]
        # a west-coast path at 13.5 MHz: one F hop near 16 deg (the low ray first)
        opts = io.predict_hops(*lwa, 34.47, -120.48, 13.5, when, soundings)
        one = [o for o in opts if o.hops == 1 and o.layer == "F"]
        self.assertEqual(one[0].ray, "low")
        self.assertTrue(12 < one[0].elevation_deg < 20 and 190 < one[0].height_km < 240)
        best, _ = io.choose(opts, 17.0)
        self.assertEqual((best.hops, best.station), (1, "BC840"))
        # a 3300 km path at 13.5 MHz measured at 10 deg: two F hops
        best, _ = io.choose(io.predict_hops(*lwa, 41.5, -70.7, 13.5, when, soundings), 10.0)
        self.assertEqual((best.hops, best.layer), (2, "F"))
        # far above the MUF: nothing; no ionosonde in range: None
        self.assertEqual(io.predict_hops(*lwa, 34.47, -120.48, 60.0, when, soundings), [])
        self.assertIsNone(io.predict_hops(*lwa, 34.47, -120.48, 13.5, when, soundings, max_km=100))

    def test_station_list(self):
        p = self.d / "list.txt"
        p.write_text("1\tAU930\tAUSTIN\t30.40\t262.30\n2\tPA836\tPT ARGUELLO\t34.80\t239.50\nnot a station\n")
        st = io.load_station_list(str(p))
        self.assertEqual([s.code for s in st], ["AU930", "PA836"])
        self.assertAlmostEqual(st[1].lon_deg, -120.5)
        self.assertGreater(len(io.load_station_list()), 100)          # the shipped list


if __name__ == "__main__":
    unittest.main()

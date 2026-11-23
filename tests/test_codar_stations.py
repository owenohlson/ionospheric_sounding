"""Station identification: table parsing, geometry, waveform/band/date filters and the labelling rule."""

import tempfile
import unittest
from datetime import date
from pathlib import Path

import helpers  # noqa: F401
import codar_stations as cs

LWASV = (34.348358, -106.885783)
TABLE = """Station,"Station Name",Network,Frequency,Bandwidth,SweepRate,"Date Added","Last Radial File",Latitude,Longitude,Manufacturer
ARG1,"Pt. Arguello, CA",SLO,13.49971,-99.25937700,2.00000000,2010-07-21,2025-07-03,34.576950,-120.650500,"CODAR"
PTC1,"Point Conception, CA",SLO,13.49971,-99.25937700,2.00000000,2010-05-13,2025-07-03,34.448667,-120.471650,"CODAR"
SCCI,"Catalina Island, CA",USC,13.50000,-99.25937700,2.00000000,2006-07-18,2025-08-12,33.446867,-118.478217,"CODAR"
COP1,"Coal Oil Pt.",UCSB,13.50000,99.25937700,2.00000000,2004-10-25,2025-07-03,34.407467,-119.878783,"CODAR"
BML1,"Bodega Marine Laboratory, CA",BML,13.50000,-51.46782700,2.00000000,2006-08-19,2025-09-03,38.317383,-123.072317,"CODAR"
SNI1,"San Nicolas Island, CA",UCSB,13.43965,-99.25937700,2.00000000,2009-09-10,2023-05-04,33.280217,-119.522500,"CODAR"
NEW1,"Added later",SLO,13.50000,-99.25937700,2.00000000,2024-01-01,2025-07-03,35.0,-120.0,"CODAR"
SMOA,"Humboldt Bay, Samoa, CA",BML,13.50000,-99.25937700,2.00000000,2022-07-22,2023-11-02,40.768783,-124.218767,"CODAR"
KAP,"Kapolei, HI",UH,13.50000,-9999.00000000,2.30769000,2015-09-22,2025-08-28,21.311350,-158.117000,"GHFDR"
"""
LATER = TABLE.replace("ARG1,\"Pt. Arguello, CA\",SLO,13.49971,-99.25937700", "ARG1,\"Pt. Arguello, CA\",SLO,13.52500,-47.79155300")


class StationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        a, b = Path(self.tmp.name) / "a.csv", Path(self.tmp.name) / "b.csv"
        a.write_text(TABLE)
        b.write_text(LATER)
        self.stations = cs.load_tables([str(a), str(b)])
        self.rx = cs.Receiver(*LWASV, 13.5e6, 100e3, 2.0, -99.259377e3, date(2023, 12, 9))

    def test_tables_keep_every_configuration_and_skip_unknown_waveforms(self):
        codes = [c.code for c in self.stations]
        self.assertEqual(codes.count("ARG1"), 2)              # both snapshots' configurations
        self.assertNotIn("KAP", codes)                        # -9999 waveform
        self.assertTrue(all(c.bandwidth_hz != 0 for c in self.stations))

    def test_geometry(self):
        brg, dist = cs.bearing_distance(*LWASV, 34.576950, -120.650500)
        self.assertAlmostEqual((brg + 180) % 360 - 180, -84.9, delta=0.2)
        self.assertAlmostEqual(dist, 1261, delta=5)
        for hops in (1, 2):
            el, path = cs.skywave(1261, 300, hops)
            self.assertAlmostEqual(cs.height_for_elevation(1261, el, hops), 300, delta=0.5)
            self.assertGreater(path, 1261)
        self.assertAlmostEqual(cs.skywave(1261, 300)[0], 21.6, delta=0.5)

    def test_filters(self):
        codes = sorted(c.code for c in cs.candidates(self.stations, self.rx))
        # COP1 is listed sweeping the other way: kept, after the others and flagged; BML1 has another
        # bandwidth, ARG1's later setup is 47.8 kHz, SNI1 is mostly outside the band and stopped before
        # the date, NEW1 was added later, SMOA stopped earlier
        self.assertEqual(codes, ["ARG1", "COP1", "PTC1", "SCCI"])
        cands = cs.candidates(self.stations, self.rx)
        self.assertEqual([c.code for c in cands if cs.sign_flipped(c)], ["COP1"])
        self.assertEqual(cands[-1].code, "COP1")
        self.assertAlmostEqual(next(c for c in cands if c.code == "COP1").bandwidth_hz, -99259.377, places=3)
        self.assertEqual(len(cs.candidates(self.stations, cs.Receiver(*LWASV, 13.5e6, 100e3, 1.0,
                                                                      -99.259377e3, date(2023, 12, 9)))), 0)

    def test_labels_only_when_unambiguous(self):
        cands = cs.candidates(self.stations, self.rx)
        slo = cs.identify(-85.2, 21.0, cands, self.rx)          # between ARG1 and PTC1, SCCI 7 deg away
        self.assertEqual((slo.status, slo.label), ("network", "SLO: ARG1/PTC1"))
        usc = cs.identify(-92.0, 23.0, cands, self.rx)
        self.assertEqual((usc.status, usc.label), ("station", "SCCI (USC)"))
        mid = cs.identify(-88.8, 22.0, cands, self.rx)          # halfway: both networks fit
        self.assertEqual(mid.status, "ambiguous")
        self.assertFalse(mid.confident)
        self.assertEqual(cs.identify(-60.0, 20.0, cands, self.rx).status, "none")
        self.assertEqual(cs.identify(-85.0, 75.0, cands, self.rx).status, "none")   # no sky-wave path fits
        # COP1 (listed +99 kHz) only names a direction when no station listed with the sign fits there
        brg, _ = cs.bearing_distance(*LWASV, 34.407467, -119.878783)
        brg = (brg + 180) % 360 - 180
        near = cs.identify(brg, 21.0, cands, self.rx)
        self.assertNotIn("COP1", near.label)
        only = cs.identify(brg, 21.0, [c for c in cands if c.code == "COP1"], self.rx)
        self.assertEqual((only.status, only.label), ("station", "COP1 (UCSB)"))
        self.assertTrue(any("COP1~" in line for line in cs.describe(only)))


    def test_hop_count_prefers_a_plausible_virtual_height(self):
        # FOCF-like: 3,325 km at 10.3 deg needs 560 km for one hop (unphysical), 212 km for two
        best, alts = cs.choose_hops(cs.hop_options(3325, 10.3))
        self.assertEqual((best.hops, best.layer), (2, "F"))
        self.assertAlmostEqual(best.height_km, 212, delta=2)
        self.assertEqual([(a.hops, a.layer) for a in alts], [(3, "E")])
        # PCYC-like: one F hop at ~256 km, with two E hops at ~110 km as the alternative
        best, alts = cs.choose_hops(cs.hop_options(1777, 11.78))
        self.assertEqual((best.hops, best.layer, [a.hops for a in alts]), (1, "F", [2]))
        self.assertAlmostEqual(best.height_km, 256, delta=2)
        # nothing fits exactly: the tolerance can still admit a path, flagged as such
        opts = cs.hop_options(1261, 1.0, el_tol=5.0)
        best, alts = cs.choose_hops(opts)
        self.assertTrue(best.within_tol and not best.fits)
        self.assertEqual(cs.choose_hops(cs.hop_options(1261, 80.0)), (None, []))

    def test_identify_reports_the_hop_count(self):
        cands = cs.candidates(self.stations, self.rx)
        ident = cs.identify(-92.0, 23.0, cands, self.rx)
        scci = ident.matches[0]
        self.assertEqual((scci.config.code, scci.hops, scci.layer), ("SCCI", 1, "F"))
        self.assertTrue(any(" 1F " in line for line in cs.describe(ident)))
        lines = cs.hop_table("SCCI", 23.0, self.stations, receiver="lwasv")
        self.assertIn("chosen", lines[2])

    def test_builtin_table_and_merging(self):
        builtin = cs.load_tables()                              # no paths: the table shipped with the code
        self.assertGreater(len(builtin), 200)
        # stations HFRnet lists without a waveform (-9999) stay in the table but are never candidates
        every = cs.load_tables(keep_unknown=True)
        unknown = [c for c in every if None in (c.frequency_hz, c.bandwidth_hz, c.sweep_rate_hz)]
        self.assertTrue(unknown and len(every) == len(builtin) + len(unknown))
        out = Path(self.tmp.name) / "roundtrip.csv"
        cs.write_table(unknown, str(out))
        self.assertEqual(len(cs.load_tables([str(out)], keep_unknown=True)), len(unknown))
        self.assertEqual(cs.load_tables([str(out)]), [])
        rx = cs.Receiver(*LWASV, 13.5e6, 100e3, 2.0, -99.259377e3, date(2023, 12, 9))
        cands = cs.candidates(builtin, rx)
        # the 2023-11-21 list has the UCSB 13.5 MHz stations on -99.259 kHz too: from New Mexico they are
        # 1-4 deg from the SLO ones, so direction alone cannot pick the network
        west = cs.identify(-84.0, 19.0, cands, rx)
        self.assertEqual(west.status, "ambiguous")
        self.assertEqual({m.config.network for m in west.matches if m.passed}, {"SLO", "UCSB"})
        self.assertEqual(cs.identify(-92.0, 23.0, cands, rx).matches[0].config.code, "SCCI")
        table = Path(self.tmp.name) / "merged.csv"
        before, after = cs.merge_into_builtin([str(Path(self.tmp.name) / "a.csv")], str(table))
        self.assertEqual((before, after), (0, 9))               # KAP's -9999 row is kept in the table
        before, after = cs.merge_into_builtin([str(Path(self.tmp.name) / "b.csv")], str(table))
        self.assertEqual((before, after), (9, 10))              # ARG1's newer configuration is added
        again = cs.merge_into_builtin([str(Path(self.tmp.name) / "b.csv")], str(table))
        self.assertEqual(again, (10, 10))                       # merging is idempotent
        self.assertNotIn("KAP", {c.code for c in cs.load_tables([str(table)])})
        arg1 = [c for c in cs.load_tables([str(table)]) if c.code == "ARG1"]
        self.assertEqual(sorted(c.sources for c in arg1), [("a",), ("b",)])
        both = [c for c in cs.load_tables([str(table)]) if c.code == "PTC1"][0]
        self.assertEqual(both.sources, ("a", "b"))

    def test_sky_map_with_station_labels(self):
        import contextlib
        import io
        import numpy as np
        import plot_delay_doppler
        from test_array import received, write_rdc0
        table = Path(self.tmp.name) / "c.csv"
        table.write_text(TABLE.replace("13.49971", "24.55000").replace("13.50000", "24.55000"))
        station = helpers.FakeStation(n=48, seed=3)
        with station.patch():
            x = received(station.positions, -85.2, 21.0, 0.00437, station.delays[:, 0], snr_db=15, sweeps=16)
            write_rdc0(Path(self.tmp.name) / "a.h5", x, np.arange(1, 49), np.zeros(48, int))
            out = Path(self.tmp.name) / "sky.png"
            args = plot_delay_doppler.build_parser().parse_args([
                str(Path(self.tmp.name) / "a.h5"), "--pol", "x", "--beamformer", "conventional", "--d-min", "3.5",
                "--d-max", "5.5", "--angle-output", str(out), "--output", str(Path(self.tmp.name) / "dd.png"),
                "--codar-stations", str(table)])
            plot_delay_doppler.cli.resolve_selection(args)
            ctx = plot_delay_doppler.station_context(args, plot_delay_doppler.cli.view_from_args(args))
            annotations, lines = plot_delay_doppler.identify_sources(ctx, 0.0, [(-85.0, 21.5)], args)
            with contextlib.redirect_stdout(io.StringIO()):
                plot_delay_doppler.run(args)
        self.assertTrue(out.exists())
        bare = plot_delay_doppler.build_parser().parse_args([str(Path(self.tmp.name) / "a.h5"), "--output", "x.png",
                                                             "--codar-stations"])
        self.assertEqual(bare.codar_stations, [])               # bare flag -> built-in table
        # the synthetic file has no UTC, so no date filter: NEW1 (added later) stays a candidate
        self.assertEqual(annotations[0][0]["label"], "SLO: ARG1/NEW1/PTC1")
        self.assertTrue(any("unknown date" in line for line in lines))


if __name__ == "__main__":
    unittest.main()

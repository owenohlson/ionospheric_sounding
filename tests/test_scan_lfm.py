"""Waveform scan (scan_lfm.py): two band-limited chirps with different bandwidths in one recording."""

import unittest

import numpy as np

import helpers  # noqa: F401
import lfm_utils as lfm
import scan_lfm

FS = 100e3


def chirp(t, bandwidth, rate, delay, doppler=0.0):
    """Periodic down/up chirp centred on 0 Hz, kept only inside the receive band (like TBN)."""
    k, period = bandwidth * rate, 1.0 / rate
    tm = np.mod(t - delay, period)
    f = -bandwidth / 2 + k * tm
    s = np.exp(1j * 2 * np.pi * ((-bandwidth / 2) * tm + k / 2 * tm ** 2 + doppler * t))
    return s * (np.abs(f) < 0.475 * FS)


def recording(seconds=8.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * FS)) / FS
    x = 0.5 * chirp(t, -150727.2, 2.0, 0.0215, 0.2) + 0.7 * chirp(t, -99259.377, 2.0, 0.2182, -0.1)
    x = x + (rng.normal(size=t.size) + 1j * rng.normal(size=t.size)) / np.sqrt(2)
    return x[None].astype(np.complex64)


class ScanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.iq = recording()

    def test_lag_product_finds_both_chirp_rates(self):
        k, p = lfm.lag_product_spectrum(self.iq, FS)
        peaks = [x for x, _ in lfm.spectrum_peaks(k, p, 2, min_db=8, exclude_below=2e3, min_separation=3e3)]
        self.assertEqual(len(peaks), 2)
        for true in (-150727.2 * 2, -99259.377 * 2):
            self.assertLess(min(abs(x - true) for x in peaks), 0.01 * abs(true))

    def test_compression_needs_the_right_bandwidth(self):
        _, right = lfm.compression_profile(self.iq, FS, -99259.377, 2.0)
        _, wrong = lfm.compression_profile(self.iq, FS, -99259.377 + 400, 2.0)
        self.assertGreater(lfm.contrast_db(right) - lfm.contrast_db(wrong), 6)
        self.assertEqual(len(lfm.scan_windows(FS, 2.0)), 20)

    def test_sweep_rate_from_resets(self):
        for bandwidth, rate in ((-25733.9, 1.0), (-49629.7, 4.0)):
            t = np.arange(int(8 * FS)) / FS
            rng = np.random.default_rng(4)
            x = 0.3 * chirp(t, bandwidth, rate, 0.123) + (rng.normal(size=t.size) + 1j * rng.normal(size=t.size)) / 2 ** 0.5
            m = lfm.measure_sweep_rate(x[None], FS, bandwidth * rate)
            self.assertEqual(lfm.snap_sweep_rate(m["rate_hz"], m["rate_err_hz"]), rate)
        self.assertEqual(lfm.snap_sweep_rate(2.005, 0.003), 2.0)          # 2.007735 Hz only when 2 Hz does not fit
        self.assertEqual(lfm.snap_sweep_rate(2.0077, 0.0005), 2.007735)
        noise = (np.random.default_rng(5).normal(size=400000) + 0j)[None]
        self.assertIsNone(lfm.measure_sweep_rate(noise, FS, -198518.8)["rate_hz"])

    def test_copies_are_not_echoes(self):
        peaks = [dict(delay_ms=d, db=db) for d, db in ((15.70, 39.5), (10.52, 36.2), (20.87, 36.0), (26.05, 30.0),
                                                      (19.43, 25.0), (60.0, 20.0))]
        echoes, copies = lfm.classify_copies(peaks, 500.0, 100e3, -99259.377, 2.0)
        self.assertEqual([e["delay_ms"] for e in echoes], [15.70, 60.0])
        self.assertEqual(sorted(c["kind"] for c in copies), ["gating", "gating", "gating", "sweep wrap"])
        # copies share the echo's Doppler: with a different Doppler on one side there is no pair
        e2, _ = lfm.classify_copies([dict(p, doppler_hz=0.0) for p in peaks[:2]] +
                                    [dict(peaks[2], doppler_hz=0.5)], 500.0)
        self.assertEqual(len(e2), 3)
        e3, _ = lfm.classify_copies([dict(p, doppler_hz=0.0) for p in peaks[:3]], 500.0)
        self.assertEqual(len(e3), 1)

    def test_scan_recovers_both_waveforms_and_delays(self):
        opts = scan_lfm.ScanOptions(use_table=False, sweep_rates=(1.0, 2.0, 2.007735, 4.0), fine_seconds=4,
                                    verbose=False, candidates=[(-150727.2, 2.0), (-75363.6, 4.0)])
        r = scan_lfm.scan(self.iq, FS, opts=opts)
        got = {round(c.best_bandwidth / 1e3): c for c in r.found}
        self.assertEqual(sorted(got), [-151, -99])
        # the documented waveform is used exactly; the other is worked out blind and scanned
        self.assertEqual(got[-151].best_bandwidth, -150727.2)
        self.assertEqual(got[-151].kind, "table")
        self.assertLess(abs(got[-151].scan_peak_hz), 3)                 # the bandwidth scan agrees with the table
        self.assertEqual(got[-99].kind, "not in table")
        self.assertAlmostEqual(got[-99].best_bandwidth, -99259.4, delta=5)
        self.assertTrue(all(c.rate == 2.0 for c in r.found))          # never the rare 2.007735 Hz
        self.assertAlmostEqual(got[-99].echoes[0][0], 218.2, delta=0.05)
        self.assertAlmostEqual(got[-151].echoes[0][0], 21.5, delta=0.05)
        # same chirp rate documented at 2 and 4 Hz: the resets pick 2 Hz
        self.assertTrue(got[-151].rate_basis.startswith("measured"), got[-151].rate_basis)
        self.assertTrue(got[-99].rate_basis.startswith("measured"))
        for c in r.found:
            self.assertAlmostEqual(c.rate_measurement["rate_hz"], 2.0, delta=0.01)
            self.assertTrue(any(t["rate"] == 1.0 and np.isfinite(t["snr_db"]) for t in c.rate_test))
        self.assertTrue(any("#1" in line for line in scan_lfm.report_lines(r, opts.min_contrast_db)))

    def test_documented_waveform_without_a_lag_peak_and_flat_profiles(self):
        opts = scan_lfm.ScanOptions(use_table=False, blind=False, fine_seconds=4, verbose=False,
                                    candidates=[(-99259.377, 2.0), (-25733.9, 1.0)])
        r = scan_lfm.scan(self.iq, FS, opts=opts)
        self.assertEqual([(c.kind, c.best_bandwidth) for c in r.found], [("table, no lag peak", -99259.377)])
        self.assertEqual([c.bandwidth for c in r.rejected], [-25733.9])

    def test_documented_with_the_opposite_sign(self):
        # +99 kHz documented, -99 kHz received: matched with the sign reversed; next to a same-sign
        # entry for -151 kHz, a +151 kHz one is listed as a flipped source
        opts = scan_lfm.ScanOptions(use_table=False, fine_seconds=4, verbose=False,
                                    candidates=[(99259.377, 2.0), (-150727.2, 2.0), (150727.2, 2.0)])
        r = scan_lfm.scan(self.iq, FS, opts=opts)
        got = {round(c.best_bandwidth / 1e3): c for c in r.found}
        self.assertEqual(sorted(got), [-151, -99])
        self.assertEqual((got[-99].kind, got[-99].best_bandwidth), ("table, sign flipped", -99259.377))
        self.assertEqual(got[-151].kind, "table")
        self.assertIn("user (sign flipped)", got[-151].sources)
        self.assertFalse(any(c.best_bandwidth > 0 for c in r.candidates))    # the listed signs are not tried


class StrongWaveformTests(unittest.TestCase):
    def test_pick_and_name(self):
        import strong_waveforms as sw
        res = dict(waveforms=[
            dict(bandwidth_hz=-51467.827, sweep_rate_hz=2, kind="table",
                 echoes=[dict(delay_ms=25.6, snr_db=22.6), dict(delay_ms=23.5, snr_db=13.6)]),
            dict(bandwidth_hz=-99259.377, sweep_rate_hz=2, kind="table",
                 echoes=[dict(delay_ms=15.7, snr_db=35.7), dict(delay_ms=19.1, snr_db=31.1),
                         dict(delay_ms=16.9, snr_db=20.0), dict(delay_ms=1.0, snr_db=20.0)]),
            dict(bandwidth_hz=-220576.4, sweep_rate_hz=4, echoes=[dict(delay_ms=200.0, snr_db=12.0)]),
            dict(bandwidth_hz=-49629.688, sweep_rate_hz=4, echoes=[dict(delay_ms=248.0, db=18.0)])])
        skipped = []
        got = sw.strong(res, max_n=3, min_snr=15, skipped=skipped)
        self.assertEqual([w["tag"] for w in got], ["bw99p259_fsweep2", "bw51p468_fsweep2", "bw49p630_fsweep4"])
        self.assertEqual((got[0]["d_min_ms"], got[0]["d_max_ms"]), (14.2, 18.2))      # strongest -1.5 / +2.5
        self.assertEqual(got[0]["echoes_ms"], [15.7, 16.9])                           # strongest first, inside
        self.assertEqual(sorted(d for _, d, _ in skipped), [1.0, 19.1, 23.5])           # outside: left out
        self.assertEqual(got[2]["d_max_ms"], 250.0)                                     # clipped at the sweep
        self.assertEqual(got[0]["label"], "bw99p259_fsweep2")
        self.assertEqual(got[0]["window"], "d14p2-18p2ms")
        self.assertEqual(len(sw.strong(res, max_n=2)), 2)
        self.assertEqual(sw.tag(-150727.2, 2.007735), "bw150p727_fsweep2p00773")
        near0 = dict(waveforms=[dict(bandwidth_hz=-99259.377, sweep_rate_hz=2, echoes=[dict(delay_ms=0.5, snr_db=30)])])
        self.assertEqual(sw.strong(near0)[0]["d_min_ms"], 0.0)                          # clipped at 0

    def test_untabled_needs_more_snr(self):
        import strong_waveforms as sw
        res = dict(waveforms=[
            dict(bandwidth_hz=877233.28, sweep_rate_hz=2, kind="not in table",
                 echoes=[dict(delay_ms=45.8, snr_db=15.5), dict(delay_ms=3.07, snr_db=15.4)]),
            dict(bandwidth_hz=-150727.2, sweep_rate_hz=2, kind="table", echoes=[dict(delay_ms=138.0, snr_db=49.9)])])
        self.assertEqual([w["tag"] for w in sw.strong(res)], ["bw150p727_fsweep2"])
        self.assertEqual(len(sw.strong(res, min_snr_untabled=15)), 2)
        res["waveforms"][0]["echoes"][0]["snr_db"] = 30.0
        self.assertEqual(len(sw.strong(res)), 2)

    def test_hand_set_windows(self):
        import strong_waveforms as sw
        res = dict(waveforms=[dict(bandwidth_hz=-51467.827, sweep_rate_hz=2, echoes=[
            dict(delay_ms=19.0, snr_db=30.0), dict(delay_ms=23.0, snr_db=25.0),
            dict(delay_ms=401.0, snr_db=20.0), dict(delay_ms=150.0, snr_db=10.0)])])
        hand = sw.strong(res, windows=sw.parse_windows(["bw51:10:200"]))
        self.assertEqual([(w["d_min_ms"], w["d_max_ms"], w["echoes_ms"], w["label"]) for w in hand],
                         [(10.0, 200.0, [19.0, 23.0, 150.0], "bw51p468_fsweep2")])
        two = sw.strong(res, windows=sw.parse_windows(["bw51:17:25", "bw51:395:405"]))
        self.assertEqual([w["label"] for w in two], ["bw51p468_fsweep2_d17p0-25p0ms", "bw51p468_fsweep2_d395p0-405p0ms"])
        empty = sw.strong(res, windows=sw.parse_windows(["bw51:300:350"]))
        self.assertEqual(empty[0]["echoes_ms"], [])
        self.assertEqual(sw.strong(res, windows=sw.parse_windows(["bw99:1:2"]))[0]["d_min_ms"], 17.5)  # other tag
        with self.assertRaises(ValueError):
            sw.parse_windows(["bw51:30:12"])

    def test_no_delay_bins_message(self):
        import lfm_utils as lfm
        w = lfm.LFMWaveform(877233.28, 2.0, 100000.0, range_method="dechirp")
        with self.assertRaises(ValueError) as cm:
            lfm.delay_band(w, 50000, 41.1, 48.8)
        self.assertIn("reach 0.000-28.4", str(cm.exception))
        self.assertIn("and 471.5", str(cm.exception))


if __name__ == "__main__":
    unittest.main()

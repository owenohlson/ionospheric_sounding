"""Gating-sideband detection and CLEAN removal (lfm_utils.detect_gating_comb / clean_gating_comb)."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np

import helpers
import lfm_utils as lfm
import rdc_io
import rdc_utils as ru

FS, SF, BW = 20000.0, 2.0, -19851.875          # |B| / fs as for the 99.26 kHz CODARs at 100 kHz
N = int(FS / SF)
GATE_HZ = 200.0                               # 50% duty gate
SPACING_MS = GATE_HZ / abs(BW * SF) * 1e3     # sideband spacing in the compressed output, 5.04 ms
MAIN = (0.0137, 0.3, 1.0)                     # delay [s], Doppler [Hz], amplitude
WEAK = (0.0137 - SPACING_MS * 1e-3, -0.4, 0.1)  # a real, weaker echo right on the main echo's -1 order


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def signal(*echoes, sweeps=40, noise=0.01, seed=0):
    t = np.arange(N * sweeps) / FS
    x = np.zeros(t.size, complex)
    for tau, fd, amp in echoes:
        tt = t - tau
        gate = np.mod(tt, 1 / GATE_HZ) < 0.5 / GATE_HZ
        x += amp * helpers.lfm_periodic(tt, BW, 1 / SF) * gate * np.exp(2j * np.pi * fd * t)
    rng = np.random.default_rng(seed)
    return x + noise * (rng.normal(size=t.size) + 1j * rng.normal(size=t.size))


def compress(x):
    rdc0 = x.reshape(-1, N).T[:, :, None].astype(np.complex64)
    return lfm.range_compress(rdc0, lfm.LFMWaveform(BW, SF, FS), "hann")


def power(Y, lag):
    return float(np.mean(np.abs(Y[int(lag) % N]) ** 2))


def doppler_power(Y, lag):
    return np.abs(np.fft.fft(Y[int(lag) % N, :, 0] * np.hanning(Y.shape[1]))) ** 2


class SidebandTests(unittest.TestCase):
    def test_detects_the_comb(self):
        comb = lfm.detect_gating_comb(compress(signal(MAIN)), FS)
        self.assertTrue(comb["detected"])
        self.assertAlmostEqual(comb["peak_ms"], 13.7, delta=0.06)
        self.assertAlmostEqual(comb["spacing_ms"], SPACING_MS, delta=0.06)
        first = [o for o in comb["orders"] if abs(o["n"]) == 1]
        self.assertEqual(len(first), 2)
        for o in first:
            self.assertTrue(-10 < o["level_db"] < -1, o)
            self.assertGreater(o["coherence_db"], -10)

    def test_no_comb_without_gating(self):
        t = np.arange(N * 40) / FS
        x = helpers.lfm_periodic(t - 0.0137, BW, 1 / SF) + 0.01 * np.random.default_rng(1).normal(size=t.size)
        self.assertFalse(lfm.detect_gating_comb(compress(x), FS)["detected"])

    def test_clean_removes_copies_and_keeps_echoes(self):
        Y = compress(signal(MAIN, WEAK))
        comb = lfm.detect_gating_comb(Y, FS)
        Yc, info = lfm.clean_gating_comb(Y, comb, FS, n_lags=N, block=20)
        self.assertEqual(info["blocks"], 2)
        pk = comb["peak_bin"]
        plus1 = next(o["offset_bins"] for o in comb["orders"] if o["n"] == 1)
        self.assertLess(10 * np.log10(power(Yc, pk + plus1) / power(Y, pk + plus1)), -15)   # the copy goes
        self.assertLess(abs(10 * np.log10(power(Yc, pk) / power(Y, pk))), 0.05)            # the echo stays
        # the weaker echo sitting on the -1 order keeps its own Doppler line, except for the known cross
        # term: its own +1 copy sits on the main echo, so subtracting main x r(-1) also removes r(+1) r(-1)
        # of it. With these (ideal 50% gate, -3.9 dB) orders that is -4 dB; with orders 10 dB down, < 1 dB.
        lag = round(WEAK[0] * FS)
        alone = doppler_power(compress(signal(WEAK)), lag)
        line = int(np.argmax(alone))
        r = {o["n"]: complex(o["ratio_re"], o["ratio_im"]) for o in comb["orders"]}
        expect = 20 * np.log10(abs(1 - r[-1] * r[1]))
        got = 10 * np.log10(doppler_power(Yc, lag)[line] / alone[line])
        self.assertLess(abs(got - expect), 0.5)

    def test_clean_on_kept_rows_equals_full_axis(self):
        Y = compress(signal(MAIN, WEAK))
        comb = lfm.detect_gating_comb(Y, FS)
        full, _ = lfm.clean_gating_comb(Y, comb, FS, n_lags=N)
        out = np.arange(round(0.005 * FS), round(0.025 * FS))
        s = lfm._ms_bins(lfm.SIDEBAND_SEARCH_MS, FS, 0)
        keep = np.union1d(out, (comb["peak_bin"] + np.arange(-s, s + 1)) % N)
        part, _ = lfm.clean_gating_comb(Y[keep], comb, FS, n_lags=N, lags=keep)
        np.testing.assert_allclose(part, full[keep], rtol=1e-5, atol=1e-2)


class SidebandExportTests(unittest.TestCase):
    def setUp(self):
        import soundfile as sf
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = Path(self.tmp.name)
        self.wav = self.d / "gated.wav"
        x = signal(MAIN, sweeps=24)
        sf.write(str(self.wav), np.column_stack([x.real, x.imag]), int(FS), subtype="FLOAT")

    def export(self, name, **kw):
        s = ru.ExportSettings(sweep_frequency=SF, bandwidth=BW, d_min=5, d_max=25, **kw)
        quiet(ru.export, self.wav, "rdc1", self.d / name, s)
        f = rdc_io.RDCFile(self.d / name)
        return f, f.read("rdc1")[:, :, 0], np.asarray(f.axis("rdc1", "delay_ms"))

    def test_export_with_suppression(self):
        f0, plain, delay = self.export("plain.h5")
        f1, clean, _ = self.export("clean.h5", suppress_sidebands=True)
        at = lambda A, ms: float(np.mean(np.abs(A[np.argmin(np.abs(delay - ms))]) ** 2))  # noqa: E731
        self.assertLess(10 * np.log10(at(clean, 13.7 + SPACING_MS) / at(plain, 13.7 + SPACING_MS)), -12)
        self.assertLess(abs(10 * np.log10(at(clean, 13.7) / at(plain, 13.7))), 0.1)
        sb = f1.meta["sideband_suppression"]
        self.assertTrue(sb["detected"])
        self.assertAlmostEqual(sb["spacing_ms"], SPACING_MS, delta=0.06)
        self.assertNotIn("sideband_suppression", f0.meta)
        self.assertIn(("Sidebands", f"CLEAN: {sb['spacing_ms']:.3f} ms comb of the {sb['main_ms']:.3f} ms echo"),
                      ru.RDCView(self.d / "clean.h5").info_rows())
        with self.assertRaises(ValueError):          # needs the raw recording
            quiet(ru.export, self.d / "plain.h5", "rdc2", self.d / "x.h5",
                  ru.ExportSettings(suppress_sidebands=True, integration_time=4))
        with self.assertRaises(ValueError):          # needs the matched filter
            self.export("d.h5", suppress_sidebands=True, range_method="dechirp")


class ScanSidebandTests(unittest.TestCase):
    """scan_lfm's delay profiles with the gating comb removed (lfm.compression_profile_clean)."""

    @classmethod
    def setUpClass(cls):
        cls.x = signal(MAIN, sweeps=24)[None].astype(np.complex64)

    def level(self, d, p, ms):
        """Peak within +/-0.1 ms of ``ms`` (mod one sweep), dB above the profile median."""
        off = np.abs((d - ms + 250.0) % 500.0 - 250.0)                     # sweep period 500 ms
        return 10 * np.log10(p[off <= 0.1].max() / np.median(p))

    def test_profile_loses_the_comb_and_keeps_the_echo(self):
        d, clean, raw, comb = lfm.compression_profile_clean(np.repeat(self.x, 3, axis=0), FS, BW, SF)
        self.assertTrue(comb["detected"])
        _, ref = lfm.compression_profile(np.repeat(self.x, 3, axis=0), FS, BW, SF)
        np.testing.assert_allclose(raw, ref, rtol=1e-4)                      # "before" is the usual profile
        self.assertLess(abs(10 * np.log10(clean.max() / raw.max())), 0.05)   # the echo stays
        self.assertGreater(max(o["n"] for o in comb["orders"]), 8)          # every order in half a sweep
        for n in (-5, -1, 1, 5):                                              # sub-bin spacing reaches order 5
            at = 13.7 + n * SPACING_MS
            self.assertGreater(self.level(d, raw, at), 10)
            self.assertLess(self.level(d, clean, at), self.level(d, raw, at) - 10)
        # streams compressed in groups give the same profile
        d2, clean2, _, _ = lfm.compression_profile_clean(np.repeat(self.x, 3, axis=0), FS, BW, SF,
                                                         budget_bytes=N * 24 * 8)
        np.testing.assert_allclose(clean2, clean, rtol=1e-3, atol=1e-6 * clean.max())

    def test_dechirped_sweeps_are_left_alone(self):
        _, clean, raw, comb = lfm.compression_profile_clean(self.x, FS, BW, 2.007735)
        self.assertFalse(comb["detected"])
        self.assertIs(clean, raw)

    def test_scan_marks_fewer_copies(self):
        import scan_lfm
        opts = dict(use_table=False, blind=False, verbose=False, candidates=[(BW, SF)], fine_seconds=12)
        kept = quiet(scan_lfm.scan, self.x, FS, opts=scan_lfm.ScanOptions(suppress_sidebands=False, **opts)).found[0]
        clean = quiet(scan_lfm.scan, self.x, FS, opts=scan_lfm.ScanOptions(**opts)).found[0]
        self.assertIsNone(kept.comb)
        self.assertGreaterEqual(len([x for x in kept.copies if x["kind"] == "gating"]), 2)
        self.assertTrue(clean.comb["detected"])
        # what is left of the subtracted orders is marked as suppressed, not as an ordinary copy
        self.assertFalse(any(x.get("suppressed") for x in kept.copies))
        for x in clean.copies:
            n = round(x["offset_ms"] / clean.comb["spacing_ms"])
            measured = {o["n"]: o["coherence_db"] for o in clean.comb["orders"]}
            if x["kind"] == "gating" and abs(x["of_delay_ms"] - 13.7) < 0.1 and n in measured:
                self.assertEqual(x["suppressed"], measured[n] >= lfm.SIDEBAND_MIN_COHERENCE_DB)
        self.assertAlmostEqual(clean.echoes[0][0], 13.7, delta=0.06)
        self.assertLess(len(clean.copies), len(kept.copies))
        self.assertTrue(any("sideband CLEAN" in line for line in scan_lfm.report_lines(
            scan_lfm.ScanResult([clean], [], [clean]), 10.0)))


if __name__ == "__main__":
    unittest.main()

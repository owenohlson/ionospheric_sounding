"""End-to-end export and view tests on synthetic WAV recordings (no LSL needed)."""

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

import helpers  # noqa: F401  (sets sys.path)
import h5py
import lfm_utils as lfm
import rdc_io
import rdc_utils as ru

FS, SF, BW = 2000, 2.0, -1985.1875


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = Path(self.tmp.name)
        self.wav = self.d / "s.wav"
        self.iq = helpers.write_wav(self.wav, FS, SF, BW, gap=(5000, 5100))
        self.raw_settings = ru.ExportSettings(sweep_frequency=SF, bandwidth=BW, gap_fill="zero")
        self.r0 = self.d / "r0.h5"
        quiet(ru.export, self.wav, "rdc0", self.r0, self.raw_settings)

    def cube(self, path, stage):
        return rdc_io.RDCFile(path).read(stage)

    def test_rdc0_holds_the_samples(self):
        f = rdc_io.RDCFile(self.r0)
        x = f.read("rdc0")[:, :, 0].T.ravel()
        expect = np.where(np.isfinite(self.iq), self.iq, 0)
        np.testing.assert_allclose(x, expect, atol=1e-6)
        self.assertEqual(f.channel_ids("rdc0")[1].tolist(), [-1])
        self.assertEqual(f.meta["polarization_mode"], "unknown")
        self.assertEqual(f.invalid_counts("rdc0").sum(), 100)
        self.assertEqual(f.meta["history"][0]["step"], "export_rdc0")

    def test_chained_equals_direct(self):
        quiet(ru.export, self.r0, "rdc1", self.d / "r1.h5", ru.ExportSettings(d_min=5, d_max=30))
        quiet(ru.export, self.d / "r1.h5", "rdc2", self.d / "r2.h5", ru.ExportSettings(integration_time=8, hop_time=4))
        direct = ru.ExportSettings(**{**self.raw_settings.__dict__, "d_min": 5, "d_max": 30,
                                      "integration_time": 8, "hop_time": 4})
        quiet(ru.export, self.wav, "rdc2", self.d / "d2.h5", direct)
        np.testing.assert_allclose(self.cube(self.d / "r2.h5", "rdc2"), self.cube(self.d / "d2.h5", "rdc2"), atol=1e-3)
        f = rdc_io.RDCFile(self.d / "r2.h5")
        self.assertEqual(f.stages, ["rdc1", "rdc2"])
        self.assertEqual([h["step"] for h in f.meta["history"]], ["export_rdc0", "export_rdc1", "export_rdc2"])
        np.testing.assert_allclose(f.axis("rdc2", "frame_tstart_s"), np.arange(0, 13, 4))

    def test_raw_streams_straight_to_rdc1(self):
        """Raw -> RDC1 without an RDC0 gives the same cube, time axis and UTC as going through RDC0."""
        quiet(ru.export, self.r0, "rdc1", self.d / "via0.h5", ru.ExportSettings(d_min=5, d_max=30))
        streamed = ru.ExportSettings(**{**self.raw_settings.__dict__, "d_min": 5, "d_max": 30})
        quiet(ru.export, self.wav, "rdc1", self.d / "direct.h5", streamed)
        np.testing.assert_allclose(self.cube(self.d / "direct.h5", "rdc1"), self.cube(self.d / "via0.h5", "rdc1"),
                                   atol=1e-5)
        a, b = rdc_io.RDCFile(self.d / "direct.h5"), rdc_io.RDCFile(self.d / "via0.h5")
        np.testing.assert_allclose(a.axis("rdc1", "delay_ms"), b.axis("rdc1", "delay_ms"))
        self.assertEqual(a.time_zero_utc, b.time_zero_utc)
        np.testing.assert_array_equal(a.invalid_counts("rdc1", [0], slice(None)),
                                      b.invalid_counts("rdc1", [0], slice(None)))
        v = ru.RDCView(self.d / "direct.h5")
        quiet(v.select_delay)
        self.assertEqual(next(v.rdc2_frames(8, 4))["cube"].shape[0], v.delay_ms.size)

    def test_streamed_plot_products_match_in_memory_math(self):
        """Frames, Doppler-time and summed power are streamed by channel chunk / frame; same numbers."""
        quiet(ru.export, self.r0, "rdc1", self.d / "s1.h5", ru.ExportSettings(d_min=5, d_max=30))
        v = ru.RDCView(self.d / "s1.h5")
        quiet(v.select_delay, 5, 30)
        rdc1, _, times = v.rdc1()
        for combine in ("incoherent", "coherent", "peak"):
            t1, _, p1 = ru.doppler_time(rdc1, v.prf, 4, 2, "hann", None, combine)
            t2, _, p2, _, _ = v.doppler_time(4, 2, "hann", None, combine)
            np.testing.assert_allclose(p2, p1, rtol=1e-4)
            np.testing.assert_allclose(t2, t1)
        np.testing.assert_allclose(v.rdc1_power()[0], ru.incoherent_power(rdc1), rtol=1e-5)
        count = int(4 * v.prf)
        for i, frame in enumerate(v.rdc2_frames(4, 2)):
            s = 2 * i * int(v.prf)
            expect, _ = lfm.doppler_fft(rdc1[:, s:s + count], v.prf, "hann", None)
            np.testing.assert_allclose(frame["cube"], expect, rtol=1e-4, atol=1e-3)
            self.assertAlmostEqual(frame["t0"], times[s])

    def test_rdc2_matches_manual_math(self):
        quiet(ru.export, self.r0, "rdc2", self.d / "r2.h5", ru.ExportSettings(d_min=10, d_max=20, integration_time=10))
        f = rdc_io.RDCFile(self.d / "r2.h5")
        raw = rdc_io.RDCFile(self.r0).read("rdc0")
        cfg = lfm.LFMWaveform(BW, SF, FS)
        bins, delay = lfm.delay_band(cfg, raw.shape[0], 10, 20)
        rdc1 = lfm.dechirp_range_fft(raw, cfg, "hann", 0, bins)
        rdc2, fd = lfm.doppler_fft(rdc1[:, :20], SF, "hann")
        np.testing.assert_allclose(f.read("rdc2", frames=0), rdc2, atol=1e-3)
        np.testing.assert_allclose(f.axis("rdc2", "delay_ms"), delay)
        peak = np.unravel_index(np.argmax(np.abs(rdc2[..., 0])), rdc2.shape[:2])
        self.assertAlmostEqual(delay[peak[0]], 13.7, delta=0.6)
        self.assertAlmostEqual(fd[peak[1]], 0.3, delta=0.1)

    def test_view_recomputes_from_retained_rdc1_and_uses_stored_frames(self):
        quiet(ru.export, self.r0, "rdc2", self.d / "r2.h5", ru.ExportSettings(integration_time=8, hop_time=4))
        v = ru.RDCView(self.d / "r2.h5")
        quiet(v.select_delay, 10, 20)
        stored = list(v.rdc2_frames())
        self.assertEqual(len(stored), 4)
        again = list(v.rdc2_frames(integration=8, hop=4, window="hann"))
        np.testing.assert_allclose(stored[1]["cube"], again[1]["cube"], atol=1e-3)
        other = list(v.rdc2_frames(integration=5, hop=5))
        self.assertEqual(len(other), 4)
        self.assertAlmostEqual(other[1]["t0"], 5.0)

    def test_rdc2_without_rdc1_is_fixed(self):
        s = ru.ExportSettings(integration_time=8, hop_time=4, keep_rdc1=False)
        quiet(ru.export, self.r0, "rdc2", self.d / "r2.h5", s)
        v = ru.RDCView(self.d / "r2.h5")
        quiet(v.select_delay, 10, 20)
        self.assertEqual(len(list(v.rdc2_frames())), 4)
        with self.assertRaises(ValueError):
            list(v.rdc2_frames(integration=5))

    def test_conflicting_bandwidth_is_rejected(self):
        with self.assertRaises(ValueError):
            ru.RDCView(self.r0, bandwidth=-1000)

    def test_rdc2p_matches_pulse_pair_and_doppler(self):
        quiet(ru.export, self.r0, "rdc2p", self.d / "p.h5", ru.ExportSettings(d_min=10, d_max=20, estimator_sweeps=3))
        f = rdc_io.RDCFile(self.d / "p.h5")
        v = ru.RDCView(self.r0)
        quiet(v.select_delay, 10, 20)
        rdc1, _, _ = v.rdc1()
        lag1, _, _ = lfm.pulse_pair(rdc1, 3)
        np.testing.assert_allclose(f.read("rdc2p"), lag1, rtol=1e-4, atol=1e-2)
        fd = lfm.phase_doppler_hz(lag1.sum(axis=2), SF)
        strongest = np.argmax(np.abs(lag1).sum(axis=(1, 2)))
        self.assertAlmostEqual(float(np.nanmedian(fd[strongest])), 0.3, delta=0.02)
        self.assertAlmostEqual(float(f.axis("rdc2p", "time_s")[0]), 2 / SF, places=6)

    def test_time_selection_and_utc(self):
        s = ru.ExportSettings(**{**self.raw_settings.__dict__, "start_utc": "2023-12-09T12:00:00.25Z",
                                 "offset": 0.1})
        quiet(ru.export, self.wav, "rdc0", self.d / "u0.h5", s)
        quiet(ru.export, self.d / "u0.h5", "rdc1", self.d / "u1.h5", ru.ExportSettings(tstart=4, tend=10, d_min=5, d_max=30))
        f0, f1 = rdc_io.RDCFile(self.d / "u0.h5"), rdc_io.RDCFile(self.d / "u1.h5")
        # sweeps start 0.1 s after a half-second boundary: 12:00:00.25 -> first sweep at 12:00:00.60
        self.assertEqual(f0.time_zero_utc.isoformat(), "2023-12-09T12:00:00.600000+00:00")
        self.assertEqual(f1.time_zero_utc, f0.time_zero_utc)
        self.assertAlmostEqual(f1.time_origin_s("rdc1"), 4.0)
        self.assertEqual(f1.shape("rdc1")[1], 12)

    def test_legacy_3d_rdc2_and_missing_stage_are_readable(self):
        path = self.d / "legacy.h5"
        with h5py.File(path, "w") as h5:
            g = h5.create_group("rdc2")
            data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
            g.create_dataset("real", data=data)
            g.create_dataset("imag", data=-data)
            for name, value in (("delay_ms", [1.0, 2.0]), ("doppler_hz", [-1, 0, 1]), ("frame_tstart_s", [0.0]),
                                ("frame_tend_s", [10.0]), ("channel_stands", [1, 1, 2, 2]), ("channel_pols", [0, 1, 0, 1])):
                g.create_dataset(name, data=np.asarray(value))
            h5.attrs["metadata_json"] = json.dumps(dict(sweep_frequency_hz=2.0, sample_rate_hz=100e3,
                                                        synthetic=True, hop_time_s=10.0))
        f = rdc_io.RDCFile(path)
        self.assertEqual(f.shape("rdc2"), (1, 2, 3, 4))
        self.assertEqual(f.meta["rdc_stage"], 2)
        v = ru.RDCView(path, stands="2", pol="y")
        quiet(v.select_delay, full=True)
        frame = next(v.rdc2_frames())
        np.testing.assert_allclose(frame["cube"][..., 0], data[..., 3] - 1j * data[..., 3])

    def test_dry_run_writes_nothing(self):
        quiet(ru.export, self.r0, "rdc1", self.d / "no.h5", ru.ExportSettings(dry_run=True))
        self.assertFalse((self.d / "no.h5").exists())

    def test_failed_export_keeps_existing_output(self):
        out = self.d / "keep.h5"
        out.write_bytes(b"old")
        with self.assertRaises(ValueError):
            quiet(ru.export, self.r0, "rdc2", out, ru.ExportSettings(integration_time=1000))
        self.assertEqual(out.read_bytes(), b"old")


class ScriptTests(unittest.TestCase):
    """Every command line runs on a small WAV-derived cache."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.d = Path(cls.tmp.name)
        helpers.write_wav(cls.d / "s.wav", FS, SF, BW)
        cls.env = dict(os.environ, MPLBACKEND="Agg")
        stubs = os.environ.get("RDC_TEST_STUBS")
        if stubs:
            cls.env["PYTHONPATH"] = stubs + os.pathsep + cls.env.get("PYTHONPATH", "")
        cls.run_tool("export_rdc0.py", cls.d / "s.wav", "--sweep-frequency", SF, f"--bandwidth={BW}", "--output", cls.d / "r0.h5")
        cls.run_tool("export_rdc2.py", cls.d / "r0.h5", "--d-min", 5, "--d-max", 30, "--integration-time", 8,
                     "--hop-time", 4, "--output", cls.d / "r2.h5")
        cls.run_tool("export_rdc2p.py", cls.d / "r0.h5", "--d-min", 5, "--d-max", 30, "--output", cls.d / "p.h5")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @classmethod
    def run_tool(cls, script, *args):
        cmd = [sys.executable, str(helpers.TOOLS / script)] + [str(a) for a in args]
        r = subprocess.run(cmd, capture_output=True, text=True, env=cls.env)
        if r.returncode:
            raise AssertionError(f"{script} failed:\n{r.stdout[-1500:]}\n{r.stderr[-3000:]}")
        return r.stdout

    def test_plots(self):
        d = self.d
        cases = [
            ("plot_spectrogram.py", d / "r0.h5", "--estimate-lfm", "--overlay-lfm"),
            ("plot_spectrogram.py", d / "r0.h5", "--average"),
            ("plot_dechirp.py", d / "r0.h5", "--navg", 2),
            ("plot_dechirp.py", d / "r2.h5"),
            ("plot_power_delay.py", d / "r2.h5", "--full-delay"),
            ("plot_delay_doppler.py", d / "r0.h5", "--integration-time", 10),
            ("plot_delay_doppler.py", d / "r2.h5"),
            ("plot_delay_doppler.py", d / "r2.h5", "--integration-time", 6, "--fd-min", -0.5, "--fd-max", 0.5),
            ("plot_doppler_time.py", d / "r2.h5", "--integration-time", 6, "--hop-time", 2),
            ("plot_phase_doppler.py", d / "p.h5"),
            ("plot_phase_doppler.py", d / "r0.h5", "--kind", "doppler-time", "--estimator-sweeps", 2),
        ]
        for i, (script, *args) in enumerate(cases):
            out = d / f"plot{i}.png"
            self.run_tool(script, *args, "--output", out)
            self.assertTrue(out.exists(), script)

    def test_videos(self):
        import shutil
        if shutil.which("ffmpeg") is None:
            self.skipTest("ffmpeg not installed")
        self.run_tool("sliding_delay_doppler.py", self.d / "r2.h5", "--output", self.d / "vid")
        self.assertTrue((self.d / "vid" / "delay_doppler.mp4").exists())
        self.run_tool("sliding_phase_doppler.py", self.d / "p.h5", "--output", self.d / "pvid", "--frame-step", 8)
        self.assertTrue((self.d / "pvid" / "phase_doppler.mp4").exists())


if __name__ == "__main__":
    unittest.main()

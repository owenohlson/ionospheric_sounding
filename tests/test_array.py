"""Array processing checked against a physical model: each antenna receives the chirp
delayed in time (geometry + cable), with the matching RF carrier phase."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import helpers
import h5py
import lfm_utils as lfm
import rdc_io
import rdc_utils as ru
import lwa_utils

FS, SF, BW, FC = 100e3, 2.0, -99.259377e3, 24.55e6


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def received(positions, az, el, tau0, cable=None, fd=0.3, sweeps=8, snr_db=10, seed=0):
    rng = np.random.default_rng(seed)
    u = ru.unit_vectors(az, el)
    t = np.arange(int(FS / SF) * sweeps) / FS
    total = tau0 - positions @ u / helpers.C + (0 if cable is None else cable)
    x = np.stack([helpers.lfm_periodic(t - d, BW, 1 / SF) * np.exp(-2j * np.pi * FC * d) for d in total])
    x *= np.exp(2j * np.pi * fd * t)
    x += 10 ** (-snr_db / 20) * (rng.normal(size=x.shape) + 1j * rng.normal(size=x.shape)) / np.sqrt(2)
    return x.astype(np.complex64)


def write_rdc0(path, x, stands, pols, synthetic=False):
    n = int(FS / SF)
    rdc0 = x.reshape(x.shape[0], -1, n).transpose(2, 1, 0)
    with h5py.File(path, "w") as h5:
        w = rdc_io.StageWriter(h5, "rdc0", rdc0.shape, compression="none")
        w.write((slice(None),) * 3, rdc0)
        w.channels(stands, pols)
        rdc_io.write_metadata(h5, dict(format_version=3, rdc_stage=0, sample_rate_hz=FS, sweep_frequency_hz=SF,
                                       bandwidth_hz=BW, tuning_frequency_hz=FC, channel_id_kind="physical_stand",
                                       station="lwasv", synthetic=synthetic, cable_delays_injected=not synthetic))


class ArrayTests(unittest.TestCase):
    def setUp(self):
        self.station = helpers.FakeStation(n=24, bad_status=(3,))
        p = self.station.patch()
        p.start()
        self.addCleanup(p.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = Path(self.tmp.name)
        self.stands = np.arange(1, 25)
        self.cable = self.station.delays[:, 0]

    def test_direction_recovered_through_the_whole_pipeline(self):
        x = received(self.station.positions, 137.0, 31.0, 0.00437, self.cable)
        write_rdc0(self.d / "a.h5", x, self.stands, np.zeros(24, int))
        v = ru.RDCView(self.d / "a.h5", pol="x")
        quiet(v.select_delay, 3.5, 5.5)
        frame = next(v.rdc2_frames())
        proc = ru.ArrayProcessor(v.stands, v.pols, FC, "conventional", "auto", v.meta)
        cube = proc.prepare(frame["cube"])
        az, el = proc.look_direction(cube)
        self.assertLess(abs(az - 137.0), 1.0)
        self.assertLess(abs(el - 31.0), 1.0)
        off = ru.ArrayProcessor(v.stands, v.pols, FC, "conventional", "off", v.meta)
        az2, el2 = off.look_direction(off.prepare(frame["cube"]))
        self.assertFalse(abs(az2 - 137) < 3 and abs(el2 - 31) < 3, "cable phases must matter")

    def test_mvdr_background_covariance_keeps_the_signal(self):
        x = received(self.station.positions, -64.0, 47.0, 0.00437, self.cable, snr_db=0)
        write_rdc0(self.d / "b.h5", x, self.stands, np.zeros(24, int))
        v = ru.RDCView(self.d / "b.h5", pol="x")
        quiet(v.select_delay, 3.0, 6.0)
        cube = next(v.rdc2_frames())["cube"]
        conv = ru.ArrayProcessor(v.stands, v.pols, FC, "conventional", metadata=v.meta)
        mvdr = ru.ArrayProcessor(v.stands, v.pols, FC, "mvdr", metadata=v.meta)
        cube = conv.prepare(cube)
        p_conv = np.abs(conv.beamform(cube, -64, 47)) ** 2
        p_mvdr = np.abs(mvdr.beamform(cube, -64, 47)) ** 2
        self.assertAlmostEqual(np.log10(p_mvdr.max() / p_conv.max()), 0.0, delta=0.15)

    def test_angle_spectrum_and_beam_pattern(self):
        a = ru.steering_matrix(self.station.positions, FC, [80.0], [40.0])[0]
        rng = np.random.default_rng(3)
        snaps = rng.normal(size=(200, 1)) * a[None, :] + 0.05 * rng.normal(size=(200, 24))
        R = ru.spatial_covariance(snaps, 0.1)
        for method in ("conventional", "mvdr"):
            az, el, _ = ru.peak_direction(R, self.station.positions, FC, method)
            self.assertLess(abs(az - 80) + abs(el - 40), 1.0, method)
        pattern = ru.beam_pattern(self.station.positions, FC, 80, 40, np.arange(-180, 180, 2), np.arange(0, 91, 2))
        self.assertAlmostEqual(pattern.max(), 1.0, places=2)

    def test_channel_selection_and_bad_stands(self):
        stands, pols = np.repeat(self.stands, 2), np.tile([0, 1], 24)
        idx = ru.select_channels(stands, pols, "1-5", "x", "status")
        self.assertEqual(stands[idx].tolist(), [1, 2, 4, 5])
        rows, s, p = lwa_utils.tbn_stream_indices([1, 24], "both")
        self.assertEqual(rows.tolist(), [46, 47, 0, 1])
        self.assertEqual(ru.select_channels([1], [-1], "first", "both").tolist(), [0])

    def test_legacy_digitizer_labels_are_resolved(self):
        meta = {}
        s, p = lwa_utils.resolve_legacy_channel_ids([24, 24, 1, 1], [0, 1, 0, 1], meta)
        self.assertEqual(s.tolist(), [1, 1, 24, 24])
        self.assertEqual(meta["channel_id_kind"], "physical_stand")

    def test_plot_and_rdc3_export_with_array_options(self):
        import plot_delay_doppler
        x = received(self.station.positions, 137.0, 31.0, 0.00437, self.cable)
        write_rdc0(self.d / "a.h5", x, self.stands, np.zeros(24, int))
        args = plot_delay_doppler.build_parser().parse_args([
            str(self.d / "a.h5"), "--output", str(self.d / "dd.png"), "--beamformer", "conventional",
            "--d-min", "3.5", "--d-max", "5.5", "--angle-output", str(self.d / "sky.png"), "--angle-cells", "2",
            "--beam-pattern-output", str(self.d / "bp.png")])
        quiet(plot_delay_doppler.run, args)
        for name in ("dd.png", "sky.png", "bp.png"):
            self.assertTrue((self.d / name).exists(), name)
        v = ru.RDCView(self.d / "a.h5", pol="x")
        quiet(v.select_delay, 3.5, 5.5)
        proc = ru.ArrayProcessor(v.stands, v.pols, FC, "conventional", metadata=v.meta)
        quiet(ru.export_rdc3, v, self.d / "b.h5", proc, looks=[(137.0, 31.0), (0.0, 90.0)])
        f = rdc_io.RDCFile(self.d / "b.h5")
        beams = f.read("rdc3")
        self.assertEqual(beams.shape[-1], 2)
        self.assertGreater(np.abs(beams[..., 0]).max(), 3 * np.abs(beams[..., 1]).max())
        args = plot_delay_doppler.build_parser().parse_args([str(self.d / "b.h5"), "--output", str(self.d / "b.png")])
        quiet(plot_delay_doppler.run, args)
        self.assertTrue((self.d / "b.png").exists())

    def test_calibration_check_flags_a_bad_stand(self):
        import check_array_calibration as cac
        cable = self.cable.copy()
        cable[5] += 9e-9
        x = received(self.station.positions, 137.0, 31.0, 0.00437, cable, snr_db=20,
                     sweeps=16) + received(self.station.positions, 250, 55, 0.0052, cable, fd=-0.4, sweeps=16, seed=4)
        write_rdc0(self.d / "c.h5", x, self.stands, np.zeros(24, int))
        v = ru.RDCView(self.d / "c.h5", pol="x")
        quiet(v.select_delay, 3.5, 6.0)
        cube = next(v.rdc2_frames())["cube"]
        results, per_stand = cac.calibration_report(cube, self.station.positions, FC, self.cable, self.stands, n_cells=6)
        lsl = np.median([r["eta"] for r in results[("signal", "lsl")]])
        flip = np.median([r["eta"] for r in results[("signal", "lsl_flip")]])
        self.assertGreater(lsl, flip)
        worst = max(per_stand, key=lambda d: d["consistency"] * abs(d["mean_residual_deg"]))
        self.assertEqual(worst["stand"], 6)


class ForwardModelSignTests(unittest.TestCase):
    """Sweep-wrap replica: an echo far into the sweep leaves a copy at +(fs-|B|)/|k|."""

    def test_wrap_replica(self):
        cfg = lfm.LFMWaveform(BW, SF, FS)
        offset_ms = (FS - abs(BW)) / abs(cfg.chirp_rate) * 1e3
        pos = np.zeros((1, 3))
        for tau, expect in ((0.21835, True), (0.002, False)):
            x = received(pos, 0.0, 90.0, tau, fd=0.0, sweeps=3, snr_db=60)
            rdc0 = x.reshape(1, 3, -1).transpose(2, 1, 0)
            delay, order = lfm.delay_axis_ms(cfg)
            p = np.mean(np.abs(lfm.dechirp_range_fft(rdc0, cfg, "hann", 0, order)[:, 1:, 0]) ** 2, axis=1)
            main = p[np.argmin(np.abs(delay - tau * 1e3))]
            copy = p[np.argmin(np.abs(delay - (tau * 1e3 + offset_ms)))]
            rel = 10 * np.log10(copy / main)
            self.assertGreater(rel, -10) if expect else self.assertLess(rel, -60)


if __name__ == "__main__":
    unittest.main()

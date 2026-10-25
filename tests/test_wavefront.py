"""Wavefront tracking: a slowly rocking plane wave is followed through the whole pipeline."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np

import helpers
import h5py
import calibration_utils as cu
import rdc_io
import rdc_utils as ru
import wavefront as wf

FS, SF, BW, FC = 10e3, 2.0, -9.9e3, 13.5e6
PERIOD_S = 12.0


def truth(t):
    return 137.0 + 1.5 * np.sin(2 * np.pi * t / PERIOD_S), 31.0 + 1.0 * np.cos(2 * np.pi * t / PERIOD_S)


def rocking_echo(positions, cable, sweeps=40, tau0=0.00437, fd=-0.4, snr_db=12, seed=3):
    """Chirp echo whose arrival direction changes continuously with time (per sample)."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(FS / SF) * sweeps) / FS
    az, el = truth(t)
    u = ru.unit_vectors(az, el)                                    # (T, 3)
    total = tau0 - (positions @ u.T) / helpers.C + cable[:, None]  # (N, T)
    x = helpers.lfm_periodic(t[None, :] - total, BW, 1 / SF) * np.exp(-2j * np.pi * FC * total)
    x *= np.exp(2j * np.pi * fd * t)
    x += 10 ** (-snr_db / 20) * (rng.normal(size=x.shape) + 1j * rng.normal(size=x.shape)) / np.sqrt(2)
    return x.astype(np.complex64)


def write_rdc0(path, x, stands, pols):
    n = int(FS / SF)
    rdc0 = x.reshape(x.shape[0], -1, n).transpose(2, 1, 0)
    with h5py.File(path, "w") as h5:
        w = rdc_io.StageWriter(h5, "rdc0", rdc0.shape, compression="none")
        w.write((slice(None),) * 3, rdc0)
        w.channels(stands, pols)
        rdc_io.write_metadata(h5, dict(format_version=3, rdc_stage=0, sample_rate_hz=FS, sweep_frequency_hz=SF,
                                       bandwidth_hz=BW, tuning_frequency_hz=FC, channel_id_kind="physical_stand",
                                       station="lwasv", synthetic=False, cable_delays_injected=True))


class TrackerTests(unittest.TestCase):
    """The estimator alone, on (frames, stands) snapshots."""

    def setUp(self):
        rng = np.random.default_rng(1)
        n = 120
        r, th = np.sqrt(rng.uniform(0, 1, n)) * 55, rng.uniform(0, 2 * np.pi, n)
        self.pos = np.column_stack([r * np.cos(th), r * np.sin(th), rng.normal(0, 0.5, n)])
        self.t = np.arange(240) * 0.5
        self.fc = 24.55e6
        self.az = -84 + 6 * np.sin(2 * np.pi * self.t / 60)
        self.el = 19 + 3 * np.cos(2 * np.pi * self.t / 45)
        A = cu.steering(self.pos, self.fc, self.az, self.el)
        common = np.exp(1j * (2 * np.pi * 0.7 * self.t + rng.uniform(0, 6, len(self.t))))
        self.err = np.radians(rng.normal(0, 30, n))
        noise = (rng.normal(size=A.shape) + 1j * rng.normal(size=A.shape)) * 0.2
        self.X = A * np.exp(1j * self.err) * common[:, None] + noise
        self.X[:, 5] = noise[:, 5] * 5                     # a dead stand: noise only
        self.X[100:110, 9] = 0                             # a stand missing for a while

    def test_direction_changes_are_recovered_without_calibration(self):
        ref_az, ref_el, *_ = wf.reference_direction(self.X, self.pos, self.fc)
        out = wf.track(self.X, self.pos, self.fc, ref_az, ref_el)
        self.assertTrue(out["noisy"][5])
        self.assertEqual(out["slips"], 0)
        daz = (out["az_deg"] - out["az_deg"].mean()) - (self.az - self.az.mean())
        dele = (out["el_deg"] - out["el_deg"].mean()) - (self.el - self.el.mean())
        self.assertLess(np.sqrt(np.mean(daz ** 2)), 0.15)
        self.assertLess(np.sqrt(np.mean(dele ** 2)), 0.4)
        # the static residual is the injected per-stand error (minus its plane and common phase)
        good = ~out["noisy"]
        e = cu.remove_ramp(np.exp(1j * self.err), self.pos, self.fc, good)[0]
        diff = np.angle(np.exp(1j * out["static"][good]) * e[good].conj())
        diff = np.angle(np.exp(1j * (diff - np.angle(np.mean(np.exp(1j * diff))))))
        self.assertLess(np.degrees(np.sqrt(np.mean(diff ** 2))), 5.0)

    def test_phase_change_far_beyond_pi_is_unwrapped_in_time(self):
        # the azimuth swing moves edge stands through more than a full cycle
        k = wf.wavenumber(self.fc)
        u, v = wf.direction_cosines(self.az, self.el)
        excursion = np.ptp(k * (np.outer(u, self.pos[:, 0]) + np.outer(v, self.pos[:, 1])), axis=0)
        self.assertGreater(excursion.max(), 1.5 * np.pi)
        out = wf.track(self.X, self.pos, self.fc, *wf.reference_direction(self.X, self.pos, self.fc)[:2])
        n = int(np.argmax(excursion))
        self.assertLess(abs(np.ptp(out["change"][:, n]) - excursion[n]), 0.6)

    def test_plane_fit_wrapped(self):
        # an offset near pi splits the stands across the wrap; a straight fit of the wrapped values fails
        xy = self.pos[:, :2]
        k = wf.wavenumber(self.fc)
        phase = wf.wrap(3.0 + k * (0.01 * xy[:, 0] - 0.008 * xy[:, 1]))
        _, du_naive, _, _, _ = wf.fit_plane(phase, xy, k, robust=False)
        c, du, dv, res, _ = wf.fit_plane_wrapped(phase, xy, k)
        self.assertGreater(abs(du_naive - 0.01), 1e-3)
        self.assertAlmostEqual(du, 0.01, places=6)
        self.assertAlmostEqual(dv, -0.008, places=6)
        self.assertLess(np.max(np.abs(res)), 1e-6)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.station = helpers.FakeStation(n=24, bad_status=(3,))
        p = self.station.patch()
        p.start()
        self.addCleanup(p.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = Path(self.tmp.name)
        rng = np.random.default_rng(4)
        cable = self.station.delays[:, 0] + rng.normal(0, 3e-9, 24)    # errors the SSMIF does not know
        x = rocking_echo(self.station.positions, cable)
        write_rdc0(self.d / "rock.h5", x, np.arange(1, 25), np.zeros(24, int))

    def test_track_through_files(self):
        import plot_wavefront
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            res = plot_wavefront.main([str(self.d / "rock.h5"), "--output", str(self.d / "wf.png"),
                                       "--d-min", "3.5", "--d-max", "5.5", "--integration-time", "2",
                                       "--hop-time", "0.5", "--maps-output", str(self.d / "maps.png"),
                                       "--surface-output", str(self.d / "surf.png"), "--csv", str(self.d / "wf.csv"),
                                       "--npz", str(self.d / "wf.npz")])
        for name in ("wf.png", "maps.png", "surf.png", "wf.csv", "wf.npz"):
            self.assertTrue((self.d / name).exists(), name)
        self.assertIn("Mean direction", out.getvalue())
        data = np.load(self.d / "wf.npz")
        used = data["used"]
        self.assertGreater(used.sum(), 30)
        t = data["t_mid_s"][used]
        az_true, el_true = truth(t)
        # each frame averages the direction over its 2 s, which barely shrinks a 12 s period
        daz = (data["az_deg"][used] - data["az_deg"][used].mean()) - (az_true - az_true.mean())
        dele = (data["el_deg"][used] - data["el_deg"][used].mean()) - (el_true - el_true.mean())
        self.assertLess(np.sqrt(np.mean(daz ** 2)), 0.25)
        self.assertLess(np.sqrt(np.mean(dele ** 2)), 0.25)
        self.assertLess(abs(np.mean(data["az_deg"][used]) - 137.0) + abs(np.mean(data["el_deg"][used]) - 31.0), 3.0)
        self.assertNotIn(3, data["stands"].tolist())           # SSMIF-flagged stand dropped
        self.assertTrue(np.allclose(data["delay_ms"][used], data["delay_ms"][used][0], atol=0.2))


if __name__ == "__main__":
    unittest.main()

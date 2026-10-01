"""Per-antenna self-calibration: recovers injected per-stand phase errors without moving the pointing."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np

import helpers
import calibration_utils as cu
import rdc_utils as ru
from test_array import FC, quiet, received, write_rdc0


def gain_error_deg(estimate, truth, positions, fc, use=None):
    """rms phase difference after removing the common phase and linear ramp, which self-cal cannot see."""
    use = np.ones(len(truth), bool) if use is None else use
    truth = cu.remove_ramp(truth, positions, fc, use)[0]
    estimate = cu.remove_ramp(estimate, positions, fc, use)[0]
    return wrapped_rms_deg(np.angle(estimate * truth.conj())[use])


def wrapped_rms_deg(phase):
    phase = np.angle(np.exp(1j * (phase - np.angle(np.mean(np.exp(1j * phase))))))
    return float(np.degrees(np.sqrt(np.mean(phase ** 2))))


class SnapshotTests(unittest.TestCase):
    """The estimator alone, on (cells, stands) snapshots."""

    def setUp(self):
        rng = np.random.default_rng(1)
        n = 60
        r, th = np.sqrt(rng.uniform(0, 1, n)) * 55, rng.uniform(0, 2 * np.pi, n)
        self.pos = np.column_stack([r * np.cos(th), r * np.sin(th), rng.normal(0, 0.5, n)])
        self.fc = 13.5e6
        self.g = np.exp(1j * np.radians(rng.normal(0, 40, n)))
        self.g[[4, 9]] = np.exp(1j * np.radians([120, -150]))
        self.g = cu.remove_ramp(self.g, self.pos, self.fc)[0]     # a ramp would legitimately move the pointing
        k = 24
        self.az, self.el = -86 + rng.normal(0, 3, k), 21 + rng.normal(0, 2, k)
        a = cu.steering(self.pos, self.fc, self.az, self.el)
        s = np.exp(2j * np.pi * rng.uniform(size=k))
        noise = (rng.normal(size=(k, n)) + 1j * rng.normal(size=(k, n))) / np.sqrt(2)
        self.x = self.g * a * s[:, None] + 0.6 * noise
        self.x[:, 30] = noise[:, 30] * 3          # a stand that sees only noise
        self.power_db = np.zeros(n)
        self.power_db[40] = -18.0                 # a nearly dead stand

    def test_gains_flags_and_validation(self):
        res = cu.calibrate(self.x, self.pos, self.fc, self.power_db)
        used = ~res["flagged"]
        self.assertEqual(sorted(res["reasons"]), [30, 40])
        self.assertLess(gain_error_deg(res["gain"], self.g, self.pos, self.fc, used), 8.0)
        v = res["validation"]
        self.assertGreater(v["eta_after_median"], v["eta_before_median"] + 0.15)
        az, el = np.median(res["directions"], axis=0)
        self.assertLess(abs(az - np.median(self.az)) + abs(el - np.median(self.el)), 3.0)

    def test_a_linear_ramp_is_not_absorbed(self):
        """A pure pointing offset (linear phase across the array) must stay in the directions, not the gains."""
        k = 2 * np.pi * self.fc / helpers.C
        ramp = np.exp(1j * k * self.pos[:, :2] @ np.array([0.02, -0.015]))
        res = cu.self_calibrate(self.x[:, :30] * ramp[:30], self.pos[:30], self.fc)
        res0 = cu.self_calibrate(self.x[:, :30], self.pos[:30], self.fc)
        self.assertLess(wrapped_rms_deg(np.angle(res["gain"] * res0["gain"].conj())), 2.0)


class EndToEndTests(unittest.TestCase):
    """RDC0 from a physical model with extra per-stand delays -> calibrate_array.py -> array commands."""

    def setUp(self):
        self.station = helpers.FakeStation(n=24, bad_status=(3,))
        p = self.station.patch()
        p.start()
        self.addCleanup(p.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = Path(self.tmp.name)
        rng = np.random.default_rng(7)
        self.error = rng.normal(0, 4.5e-9, 24)          # ~40 deg rms at 24.55 MHz, not in the "SSMIF"
        cable = self.station.delays[:, 0] + self.error
        pos = self.station.positions
        x = received(pos, 137.0, 31.0, 0.00437, cable, snr_db=15, sweeps=16, seed=1) \
            + received(pos, 140.0, 35.0, 0.0050, cable, fd=-0.4, snr_db=15, sweeps=16, seed=2)
        write_rdc0(self.d / "a.h5", x, np.arange(1, 25), np.zeros(24, int))

    def test_calibrate_and_apply(self):
        import calibrate_array
        import check_array_calibration
        import plot_delay_doppler
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            calibrate_array.main([str(self.d / "a.h5"), "--output", str(self.d / "cal.h5"), "--d-min", "3.5",
                                  "--d-max", "6", "--integration-time", "2", "--hop-time", "2",
                                  "--min-snr-db", "10", "--csv", str(self.d / "cal.csv")])
        cal = cu.ArrayCalibration.load(self.d / "cal.h5")
        self.assertNotIn(3, cal.stands.tolist())                 # SSMIF-flagged stand never calibrated
        self.assertEqual(cal.meta["validation_split"], "alternate frames")
        v = cal.meta["validation"]
        self.assertGreater(v["eta_after_median"], v["eta_before_median"])
        truth = np.exp(-2j * np.pi * FC * self.error[cal.stands - 1])
        good = ~cal.flagged
        self.assertLess(gain_error_deg(cal.gain, truth, self.station.positions[cal.stands - 1], FC, good), 10.0)

        view = ru.RDCView(self.d / "a.h5", pol="x", bad_stands="status")
        quiet(view.select_delay, 3.5, 6.0)
        cube = next(view.rdc2_frames())["cube"]
        plain = ru.ArrayProcessor(view.stands, view.pols, FC, metadata=view.meta)
        calibrated = ru.ArrayProcessor(view.stands, view.pols, FC, metadata=view.meta, calibration=cal)
        self.assertIn("self-cal", calibrated.cable_note)
        def peak(proc):
            c = proc.prepare(cube)
            return (np.abs(proc.beamform(c, *proc.look_direction(c))) ** 2).max()
        # eta 0.73 -> ~1 means the coherent beam gains ~1/0.73 = 1.37x
        self.assertGreater(peak(calibrated), 1.25 * peak(plain))
        az, el = calibrated.look_direction(calibrated.prepare(cube))
        self.assertLess(abs(az - 137) + abs(el - 31), 4.0)   # top cells mix both echoes (137,31) and (140,35)

        args = plot_delay_doppler.build_parser().parse_args([
            str(self.d / "a.h5"), "--output", str(self.d / "dd.png"), "--beamformer", "conventional",
            "--d-min", "3.5", "--d-max", "6", "--calibration", str(self.d / "cal.h5")])
        quiet(plot_delay_doppler.run, args)
        self.assertTrue((self.d / "dd.png").exists())
        with self.assertRaises(ValueError):
            cal.factors(cal.stands, cal.pols, FC + 1e6)
        import sys
        argv = sys.argv
        sys.argv = ["check_array_calibration.py", str(self.d / "a.h5"), "--d-min", "3.5", "--d-max", "6",
                    "--top-cells", "4", "--calibration", str(self.d / "cal.h5")]
        try:
            text = io.StringIO()
            with contextlib.redirect_stdout(text):
                check_array_calibration.main()
        finally:
            sys.argv = argv
        self.assertIn("selfcal", text.getvalue())


if __name__ == "__main__":
    unittest.main()

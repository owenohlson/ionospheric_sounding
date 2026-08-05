import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from doppler_compensation import (  # noqa: E402
    CartesianStateSeries,
    compensate_iq_doppler,
    geodetic_to_ecef,
    one_way_doppler_hz,
    one_way_range_and_rate,
    phase_history_from_frequency,
)


class DopplerCompensationTests(unittest.TestCase):
    def test_wgs84_equator(self):
        actual = geodetic_to_ecef(0.0, 0.0, 0.0)
        np.testing.assert_allclose(actual, [6_378_137.0, 0.0, 0.0], atol=1e-6)

    def test_receding_satellite_has_negative_doppler(self):
        _, range_rate = one_way_range_and_rate(
            satellite_position_m=[7_000_000.0, 0.0, 0.0],
            satellite_velocity_m_s=[1_000.0, 0.0, 0.0],
            receiver_position_m=[6_378_137.0, 0.0, 0.0],
        )
        self.assertAlmostEqual(float(range_rate), 1_000.0)
        self.assertLess(float(one_way_doppler_hz(range_rate, 29e6)), 0.0)

    def test_constant_doppler_is_removed(self):
        sample_rate = 10_000.0
        sample_count = 20_000
        doppler_hz = np.full(sample_count, 37.25)
        phase = phase_history_from_frequency(doppler_hz, sample_rate)
        received = np.exp(1j * phase).astype(np.complex64)

        corrected, predicted_phase = compensate_iq_doppler(
            received, doppler_hz, sample_rate
        )

        np.testing.assert_allclose(predicted_phase, phase, atol=1e-12)
        np.testing.assert_allclose(corrected, np.ones(sample_count), atol=2e-7)

    def test_time_varying_doppler_is_removed(self):
        sample_rate = 2_000.0
        time_s = np.arange(10_000) / sample_rate
        doppler_hz = 20.0 + 3.0 * np.sin(2.0 * np.pi * 0.2 * time_s)
        phase = phase_history_from_frequency(doppler_hz, sample_rate)
        received = np.exp(1j * phase)
        corrected, _ = compensate_iq_doppler(received, doppler_hz, sample_rate)
        np.testing.assert_allclose(corrected, np.ones(time_s.size), atol=2e-12)

    def test_trajectory_rejects_extrapolation(self):
        trajectory = CartesianStateSeries(
            epoch_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
            time_s=np.array([0.0, 1.0]),
            position_m=np.array([[7e6, 0.0, 0.0], [7e6, 1.0, 0.0]]),
            velocity_m_s=np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]),
        )
        with self.assertRaisesRegex(ValueError, "outside the ephemeris"):
            trajectory.interpolate([-0.1])

    def test_trajectory_velocity_is_position_derivative(self):
        trajectory = CartesianStateSeries(
            epoch_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
            time_s=np.array([0.0, 2.0]),
            position_m=np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]]),
            velocity_m_s=np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]]),
        )
        position, velocity = trajectory.interpolate([1.0])
        self.assertAlmostEqual(position[0, 0], 1.0)
        self.assertAlmostEqual(velocity[0, 0], 2.0)


if __name__ == "__main__":
    unittest.main()

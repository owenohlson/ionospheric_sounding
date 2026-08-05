"""Geometry-based Doppler prediction and compensation for complex IQ data.

The functions in this module deliberately do not know about WAV, TBN, or any
particular ephemeris file format.  File readers should translate their inputs
to the small array-based interface used here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
from scipy.interpolate import CubicHermiteSpline


SPEED_OF_LIGHT_M_S = 299_792_458.0


def _as_xyz(name: str, values) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape[-1:] != (3,):
        raise ValueError(f"{name} must have a final dimension of length 3")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains a non-finite value")
    return array


def parse_utc(value: str | datetime) -> datetime:
    """Return an aware UTC datetime from an ISO-8601 string or datetime."""
    if isinstance(value, datetime):
        result = value
    else:
        text = str(value).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        result = datetime.fromisoformat(text)

    if result.tzinfo is None:
        raise ValueError("UTC timestamps must include a timezone (usually 'Z')")
    return result.astimezone(timezone.utc)


def geodetic_to_ecef(latitude_deg: float, longitude_deg: float, height_m: float) -> np.ndarray:
    """Convert WGS-84 geodetic coordinates to ECEF metres."""
    semi_major_m = 6_378_137.0
    eccentricity_squared = 6.69437999014e-3

    latitude = np.deg2rad(float(latitude_deg))
    longitude = np.deg2rad(float(longitude_deg))
    height = float(height_m)
    prime_vertical_radius = semi_major_m / np.sqrt(
        1.0 - eccentricity_squared * np.sin(latitude) ** 2
    )

    x = (prime_vertical_radius + height) * np.cos(latitude) * np.cos(longitude)
    y = (prime_vertical_radius + height) * np.cos(latitude) * np.sin(longitude)
    z = (
        prime_vertical_radius * (1.0 - eccentricity_squared) + height
    ) * np.sin(latitude)
    return np.array([x, y, z], dtype=np.float64)


@dataclass(frozen=True)
class CartesianStateSeries:
    """Time-tagged positions and velocities in one Cartesian frame.

    ``time_s`` is measured from ``epoch_utc``.  For the initial MarmotSat
    adapter the frame is ECEF/ITRF, positions are metres, and velocities are
    metres per second in the rotating ECEF frame.
    """

    epoch_utc: datetime
    time_s: np.ndarray
    position_m: np.ndarray
    velocity_m_s: np.ndarray
    frame: str = "ecef"

    def __post_init__(self):
        epoch = parse_utc(self.epoch_utc)
        time_s = np.asarray(self.time_s, dtype=np.float64)
        position_m = _as_xyz("position_m", self.position_m)
        velocity_m_s = _as_xyz("velocity_m_s", self.velocity_m_s)

        if time_s.ndim != 1 or time_s.size < 2:
            raise ValueError("time_s must contain at least two samples")
        if position_m.shape != (time_s.size, 3):
            raise ValueError("position_m must have shape (len(time_s), 3)")
        if velocity_m_s.shape != (time_s.size, 3):
            raise ValueError("velocity_m_s must have shape (len(time_s), 3)")
        if not np.all(np.isfinite(time_s)):
            raise ValueError("time_s contains a non-finite value")
        if np.any(np.diff(time_s) <= 0):
            raise ValueError("time_s must be strictly increasing")
        if self.frame.lower() not in ("ecef", "itrf"):
            raise ValueError("the initial geometry model requires ECEF/ITRF states")

        object.__setattr__(self, "epoch_utc", epoch)
        object.__setattr__(self, "time_s", time_s)
        object.__setattr__(self, "position_m", position_m)
        object.__setattr__(self, "velocity_m_s", velocity_m_s)
        object.__setattr__(self, "frame", self.frame.lower())

    def seconds_at(self, timestamp: str | datetime) -> float:
        return (parse_utc(timestamp) - self.epoch_utc).total_seconds()

    def interpolate(self, query_time_s) -> tuple[np.ndarray, np.ndarray]:
        """Hermite-interpolate states, rejecting extrapolation.

        Position and velocity are interpolated together so the returned
        velocity remains the derivative of the returned position curve.
        """
        query = np.asarray(query_time_s, dtype=np.float64)
        if not np.all(np.isfinite(query)):
            raise ValueError("query times contain a non-finite value")
        if np.any(query < self.time_s[0]) or np.any(query > self.time_s[-1]):
            raise ValueError(
                "recording time is outside the ephemeris interval; extrapolation is disabled"
            )

        interpolator = CubicHermiteSpline(
            self.time_s,
            self.position_m,
            self.velocity_m_s,
            axis=0,
        )
        position = interpolator(query)
        velocity = interpolator(query, nu=1)
        return position, velocity


def one_way_range_and_rate(
    satellite_position_m,
    satellite_velocity_m_s,
    receiver_position_m,
    receiver_velocity_m_s=(0.0, 0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate one-way geometric range and range rate in a common frame."""
    satellite_position = _as_xyz("satellite_position_m", satellite_position_m)
    satellite_velocity = _as_xyz("satellite_velocity_m_s", satellite_velocity_m_s)
    receiver_position = _as_xyz("receiver_position_m", receiver_position_m)
    receiver_velocity = _as_xyz("receiver_velocity_m_s", receiver_velocity_m_s)

    displacement = satellite_position - receiver_position
    distance = np.linalg.norm(displacement, axis=-1)
    if np.any(distance == 0.0):
        raise ValueError("satellite and receiver positions coincide")
    line_of_sight = displacement / np.expand_dims(distance, axis=-1)
    relative_velocity = satellite_velocity - receiver_velocity
    range_rate = np.sum(relative_velocity * line_of_sight, axis=-1)
    return distance, range_rate


def one_way_doppler_hz(range_rate_m_s, carrier_frequency_hz: float) -> np.ndarray:
    """Narrowband one-way Doppler; positive range rate gives negative Doppler."""
    carrier = float(carrier_frequency_hz)
    if not np.isfinite(carrier) or carrier <= 0.0:
        raise ValueError("carrier_frequency_hz must be positive and finite")
    return -carrier * np.asarray(range_rate_m_s, dtype=np.float64) / SPEED_OF_LIGHT_M_S


def predict_one_way_doppler(
    trajectory: CartesianStateSeries,
    sample_time_s,
    receiver_position_m,
    carrier_frequency_hz: float,
    receiver_velocity_m_s=(0.0, 0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return predicted Doppler, range, and range rate at ephemeris-relative times."""
    satellite_position, satellite_velocity = trajectory.interpolate(sample_time_s)
    distance, range_rate = one_way_range_and_rate(
        satellite_position,
        satellite_velocity,
        receiver_position_m,
        receiver_velocity_m_s,
    )
    doppler = one_way_doppler_hz(range_rate, carrier_frequency_hz)
    return doppler, distance, range_rate


def phase_history_from_frequency(
    frequency_hz,
    sample_rate_hz: float,
    initial_phase_rad: float = 0.0,
) -> np.ndarray:
    """Integrate instantaneous frequency using trapezoids.

    The returned phase is the phase at each corresponding input sample.  Its
    first element is ``initial_phase_rad``.
    """
    frequency = np.asarray(frequency_hz, dtype=np.float64)
    if frequency.ndim != 1:
        raise ValueError("frequency_hz must be one-dimensional")
    sample_rate = float(sample_rate_hz)
    if not np.isfinite(sample_rate) or sample_rate <= 0.0:
        raise ValueError("sample_rate_hz must be positive and finite")
    if not np.all(np.isfinite(frequency)):
        raise ValueError("frequency_hz contains a non-finite value")

    phase = np.empty(frequency.size, dtype=np.float64)
    if frequency.size == 0:
        return phase
    phase[0] = float(initial_phase_rad)
    if frequency.size > 1:
        increments = np.pi * (frequency[:-1] + frequency[1:]) / sample_rate
        phase[1:] = phase[0] + np.cumsum(increments)
    return phase


def compensate_iq_doppler(
    iq,
    predicted_doppler_hz,
    sample_rate_hz: float,
    initial_phase_rad: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove a predicted Doppler history from one complex IQ stream.

    If the input contains ``exp(+j * integral(2*pi*f_d dt))``, compensation
    multiplies it by the conjugate phase history.  The second return value is
    the predicted Doppler phase, useful for diagnostics and chunk continuity.
    """
    samples = np.asarray(iq)
    if samples.ndim != 1:
        raise ValueError("iq must be a one-dimensional complex stream")
    if not np.iscomplexobj(samples):
        raise ValueError("iq must contain complex samples")

    doppler = np.asarray(predicted_doppler_hz, dtype=np.float64)
    if doppler.ndim == 0:
        doppler = np.full(samples.size, float(doppler), dtype=np.float64)
    if doppler.shape != (samples.size,):
        raise ValueError("predicted_doppler_hz must be scalar or have one value per IQ sample")

    phase = phase_history_from_frequency(doppler, sample_rate_hz, initial_phase_rad)
    corrected = samples * np.exp(-1j * phase)
    output_dtype = np.complex64 if samples.dtype == np.complex64 else np.complex128
    return corrected.astype(output_dtype, copy=False), phase

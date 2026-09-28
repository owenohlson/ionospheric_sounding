"""Shared fixtures: synthetic recordings and a fake LWA-SV station (no LSL needed)."""

import os
import sys
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
if os.environ.get("RDC_TEST_STUBS"):          # sandbox without h5py/soundfile
    sys.path.insert(0, os.environ["RDC_TEST_STUBS"])

import soundfile as sf  # noqa: E402

C = 299_792_458.0


def lfm_periodic(t, bandwidth, period):
    k = bandwidth / period
    tm = np.mod(t, period)
    return np.exp(1j * 2 * np.pi * ((-bandwidth / 2) * tm + (k / 2) * tm ** 2))


def write_wav(path, fs=2000, sweep_frequency=2.0, bandwidth=-1985.1875, sweeps=40, delay=0.0137,
              doppler=0.3, noise=0.1, gap=None, seed=1):
    """Stereo I/Q WAV of a delayed, Doppler-shifted periodic chirp."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(fs / sweep_frequency) * sweeps) / fs
    iq = lfm_periodic(t - delay, bandwidth, 1 / sweep_frequency) * np.exp(2j * np.pi * doppler * t)
    iq = iq + noise * (rng.normal(size=t.size) + 1j * rng.normal(size=t.size))
    if gap is not None:
        iq[gap[0]:gap[1]] = np.nan
    sf.write(str(path), np.column_stack([iq.real, iq.imag]), fs, subtype="FLOAT")
    return iq


class FakeStation:
    """Patch station_utils.lwasv_antennas with a small random array."""

    def __init__(self, n=24, seed=5, bad_status=()):
        rng = np.random.default_rng(seed)
        self.positions = np.column_stack([rng.uniform(-50, 50, n), rng.uniform(-50, 50, n), rng.normal(0, 1, n)])
        self.delays = rng.uniform(150e-9, 900e-9, (n, 2))
        self.n = n
        self.lookup = {}
        for i in range(n):
            stand = i + 1
            for pol in (0, 1):
                self.lookup[(stand, pol)] = types.SimpleNamespace(
                    stand=types.SimpleNamespace(id=stand, x=self.positions[i, 0], y=self.positions[i, 1],
                                                z=self.positions[i, 2]),
                    pol=pol, digitizer=2 * (n - 1 - i) + pol + 1,
                    combined_status=0 if stand in bad_status else 33,
                    cable=types.SimpleNamespace(delay=lambda f, d=self.delays[i, pol]: d))

    def patch(self):
        import station_utils
        return patch.object(station_utils, "lwasv_antennas", return_value=self.lookup)

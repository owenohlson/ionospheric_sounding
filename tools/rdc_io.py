"""HDF5 storage for radar data cubes (RDC0, RDC1, RDC2, RDC2P, RDC3).

One file may hold several stages (e.g. an RDC2 export keeps the RDC1 it was
made from). Each stage is a group with ``real``/``imag`` datasets, its axis
datasets, and ``channel_stands``/``channel_pols``. The root attribute
``metadata_json`` records every irreversible processing choice, plus a
``history`` list with one entry per export step.

In-memory (canonical) layouts returned by ``RDCFile.read``:

=======  ==========================================
rdc0     (adc, pulses, channels)       raw sweeps
rdc1     (delay, pulses, channels)     range spectra per sweep
rdc2     (frames, delay, doppler, channels)
rdc2p    (delay, estimates, channels)  summed lag-1 products (+ ``power``)
rdc3     (frames, delay, doppler, beams)
=======  ==========================================

Older caches stay readable: 3-D RDC2 without frames, RDC2 without retained
RDC1, files without ``rdc_stage``/``invalid_counts``, and caches whose channels
were labelled by digitizer pair (resolved through LSL).
"""

import json
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone

import numpy as np

FORMAT_VERSION = 3
STAGES = ("rdc0", "rdc1", "rdc2", "rdc2p", "rdc3")
DIMS = {
    "rdc0": ("adc", "pulses", "channels"),
    "rdc1": ("delay", "pulses", "channels"),
    "rdc2": ("frames", "delay", "doppler", "channels"),
    "rdc2p": ("delay", "estimates", "channels"),
    "rdc3": ("frames", "delay", "doppler", "beams"),
}


def stage_name(stage):
    """Accept 0, "1", "2p", "RDC2", "rdc3" ... and return the group name."""
    s = str(stage).strip().lower()
    s = s if s.startswith("rdc") else "rdc" + s
    if s not in STAGES:
        raise ValueError(f"Unknown RDC stage '{stage}'")
    return s


def _h5py():
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required to read or write RDC files") from exc
    return h5py


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

@contextmanager
def atomic_output(path):
    """Yield a temporary path; replace ``path`` only if the block finishes without error."""
    path = os.path.expanduser(path)
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=os.path.basename(path) + ".", suffix=".tmp", dir=directory)
    os.close(fd)
    try:
        yield tmp
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _compression(compression):
    if compression in (None, "none"):
        return {}
    if compression == "lzf":
        return {"compression": "lzf"}
    return {"compression": "gzip", "compression_opts": 4}


class StageWriter:
    """Complex cube writer for one stage group inside an open HDF5 file."""

    def __init__(self, h5, stage, shape, dtype="float32", compression="gzip", chunks=None, maxshape=None):
        self.stage = stage_name(stage)
        self.group = h5.require_group(self.stage)
        shape = tuple(int(n) for n in shape)
        if chunks is None:
            chunks = tuple(max(1, min(n, c)) for n, c in zip(shape, _default_chunks(self.stage, len(shape))))
        opts = dict(shape=shape, dtype=np.dtype(dtype), chunks=chunks, **_compression(compression))
        if maxshape is not None:
            opts["maxshape"] = maxshape
        self.real = self.group.create_dataset("real", **opts)
        self.imag = self.group.create_dataset("imag", **opts)
        self.group.attrs["dimension_order"] = ",".join(DIMS[self.stage][-len(shape):])

    def write(self, selection, values):
        values = np.asarray(values)
        self.real[selection] = values.real.astype(self.real.dtype, copy=False)
        self.imag[selection] = values.imag.astype(self.imag.dtype, copy=False)

    def dataset(self, name, data=None, shape=None, dtype=None, maxshape=None):
        if data is not None:
            return self.group.create_dataset(name, data=np.asarray(data))
        if maxshape is not None:
            return self.group.create_dataset(name, shape=shape, dtype=dtype, maxshape=maxshape)
        return self.group.create_dataset(name, shape=shape, dtype=dtype)

    def resize(self, shape):
        self.real.resize(shape)
        self.imag.resize(shape)

    def channels(self, stands, pols):
        self.dataset("channel_stands", np.asarray(stands, np.int32))
        self.dataset("channel_pols", np.asarray(pols, np.int32))


def _default_chunks(stage, ndim):
    return {"rdc0": (8192, 1, 32), "rdc1": (32, 256, 8), "rdc2": (1, 32, 256, 8),
            "rdc2p": (32, 256, 8), "rdc3": (1, 32, 256, 8)}[stage][-ndim:]


def write_metadata(h5, metadata):
    h5.attrs["metadata_json"] = json.dumps(_jsonable(metadata), indent=2, sort_keys=True)


def history_entry(step, **params):
    return dict(step=step, time_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                params=_jsonable(params))


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, datetime):
        return value.isoformat()
    return value


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

class RDCFile:
    """Read access to every stage stored in one RDC HDF5 file."""

    def __init__(self, path):
        self.path = os.path.expanduser(path)
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"RDC file not found: {self.path}")
        h5py = _h5py()
        try:
            with h5py.File(self.path, "r") as h5:
                self.meta = json.loads(h5.attrs.get("metadata_json", "{}"))
                self.stages = [s for s in STAGES if s in h5]
                if not self.stages:
                    raise ValueError(f"'{self.path}' contains no rdc0/rdc1/rdc2/rdc2p/rdc3 group")
                self._shapes = {s: tuple(h5[s]["real"].shape) for s in self.stages}
                self._axes = {s: {k: h5[s][k][...] for k in h5[s].keys()
                                  if k not in ("real", "imag", "invalid_counts", "channel_stands",
                                               "channel_pols", "power", "n_valid")}
                              for s in self.stages}
                self._attrs = {s: dict(h5[s].attrs) for s in self.stages}
                self._has_invalid = {s: "invalid_counts" in h5[s] for s in self.stages}
                raw_ids = {s: (h5[s]["channel_stands"][...], h5[s]["channel_pols"][...])
                           for s in self.stages if "channel_stands" in h5[s]}
        except OSError as exc:
            raise OSError(f"Could not open '{self.path}' as HDF5 (truncated or not an RDC file?)") from exc
        self._normalize()
        self._ids = {}
        for s, (stands, pols) in raw_ids.items():
            if self.meta.get("channel_id_kind") == "legacy_input_pair" or (
                    "channel_id_kind" not in self.meta and not self.meta.get("synthetic")):
                from lwa_utils import resolve_legacy_channel_ids
                stands, pols = resolve_legacy_channel_ids(stands, pols, dict(self.meta))
            self._ids[s] = (np.asarray(stands, int), np.asarray(pols, int))
        if "rdc1" in self.stages:
            for attr, key in (("tstart_s", "rdc1_tstart_s"), ("sweep_frequency_hz", "sweep_frequency_hz")):
                if attr in self._attrs["rdc1"]:
                    self.meta.setdefault(key, float(self._attrs["rdc1"][attr]))

    # -- metadata -------------------------------------------------------
    def _normalize(self):
        m = self.meta
        if "rdc_stage" not in m:
            m["rdc_stage"] = int(self.stages[-1][3]) if self.stages[-1] != "rdc2p" else "2p"
        if "gap_fill" not in m and isinstance(m.get("gap_info"), dict):
            m["gap_fill"] = m["gap_info"].get("mode", "nan")
        m.setdefault("gap_fill", "nan")
        if "rdc0" in self.stages:
            m.setdefault("adc_start", 0)
            m.setdefault("adc_count", self._shapes["rdc0"][0])

    @property
    def time_zero_utc(self):
        """UTC datetime at t = 0 of this file's time axes, or None if unknown.

        New files store ``time_zero_utc`` directly. Older files stored the UTC of
        their own first sweep (``pulse0_utc``) together with a time origin.
        """
        if self.meta.get("time_zero_utc"):
            return datetime.fromisoformat(self.meta["time_zero_utc"])
        if not self.meta.get("pulse0_utc"):
            return None
        from datetime import timedelta
        origin = 0.0
        for stage in ("rdc1", "rdc2", "rdc3"):
            if stage in self.stages:
                origin = self.time_origin_s(stage)
                break
        return datetime.fromisoformat(self.meta["pulse0_utc"]) - timedelta(seconds=origin)

    def has(self, stage):
        return stage_name(stage) in self.stages

    def shape(self, stage):
        s = stage_name(stage)
        shape = self._shapes[s]
        return (1,) + shape if s == "rdc2" and len(shape) == 3 else shape

    def axis(self, stage, name, default=None):
        return self._axes[stage_name(stage)].get(name, default)

    def channel_ids(self, stage):
        return self._ids[stage_name(stage)]

    @property
    def is_receiver(self):
        return self.meta.get("channel_id_kind") == "receiver" or self.meta.get("source_format") == "wav"

    def time_origin_s(self, stage):
        """Start time (s, relative to the original export's zero) of pulse 0 / frame 0."""
        s = stage_name(stage)
        if s == "rdc1" and "tstart_s" in self._attrs.get("rdc1", {}):
            return float(self._attrs["rdc1"]["tstart_s"])
        if s in ("rdc2", "rdc3"):
            starts = self.axis(s, "frame_tstart_s")
            return float(starts[0]) if starts is not None and len(starts) else 0.0
        if s == "rdc2p":
            return float(self._attrs[s].get("tstart_s", self.meta.get("tstart", 0.0) or 0.0))
        return float(self.meta.get("tstart", 0.0) or 0.0) if s == "rdc1" else 0.0

    # -- data -----------------------------------------------------------
    def read(self, stage, channels=None, **selection):
        """Read a complex block in canonical layout.

        ``selection`` keys are dimension names from ``DIMS`` (e.g. ``pulses=slice(0, 32)``,
        ``delay=slice(...)``, ``frames=3``). ``channels`` is an increasing index list.
        """
        s = stage_name(stage)
        h5py = _h5py()
        with h5py.File(self.path, "r") as h5:
            real, imag = h5[s]["real"], h5[s]["imag"]
            legacy_3d = s == "rdc2" and real.ndim == 3
            dims = DIMS[s][1:] if legacy_3d else DIMS[s]
            index = []
            for d in dims:
                if d in ("channels", "beams"):
                    index.append(slice(None) if channels is None else np.asarray(channels, int))
                else:
                    index.append(selection.get(d, slice(None)))
            index = tuple(index)
            data = real[index] + 1j * imag[index]
        data = np.asarray(data, np.complex64)
        if legacy_3d:
            frame = selection.get("frames", slice(None))
            data = data[None] if not isinstance(frame, (int, np.integer)) else data
        return data

    def read_real(self, stage, name, channels=None, **selection):
        """Read a real-valued companion dataset (e.g. rdc2p ``power``) with the cube's layout."""
        s = stage_name(stage)
        with _h5py().File(self.path, "r") as h5:
            ds = h5[s][name]
            index = tuple((slice(None) if channels is None else np.asarray(channels, int))
                          if d in ("channels", "beams") else selection.get(d, slice(None)) for d in DIMS[s])
            return np.asarray(ds[index])

    def invalid_counts(self, stage, channels=None, rows=slice(None)):
        """Invalid-sample counts per (pulse or frame, channel), or None if not stored."""
        s = stage_name(stage)
        if not self._has_invalid[s]:
            return None
        with _h5py().File(self.path, "r") as h5:
            ds = h5[s]["invalid_counts"]
            return np.asarray(ds[rows, slice(None) if channels is None else np.asarray(channels, int)])

    def iter_channels(self, stage, channels, chunk=8, **selection):
        """Yield ``(start, stop, block)`` over channel chunks (positions within ``channels``)."""
        channels = np.asarray(channels, int)
        for start in range(0, len(channels), int(chunk)):
            stop = min(start + int(chunk), len(channels))
            yield start, stop, self.read(stage, channels=channels[start:stop], **selection)

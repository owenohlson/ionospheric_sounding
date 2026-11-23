"""Scratch: save cable-corrected array frames + calibration files as one .npz for offline debugging."""
import sys, os
sys.path.insert(0, "tools")
import numpy as np
import lwa_utils as lwa
import rdc_utils as ru
from calibration_utils import ArrayCalibration

out = {}
view = ru.RDCView("calib_checks/array_rdc0.h5", "all", "x", "status", gap_fill="zero")
view.select_delay()
proc = ru.ArrayProcessor(view.stands, view.pols, view.fc, metadata=view.meta)
out.update(stands=proc.stands, pols=proc.pols, positions=proc.positions, fc=proc.fc,
           cable=proc.cable, delay_ms=view.delay_ms,
           digitizers=np.array([a.digitizer for a in lwa.selected_antennas(proc.stands, proc.pols)]))
for tag, kw in (("f25", dict(integration=2.5, hop=2.5)), ("whole", dict(integration=None, hop=None))):
    for i, fr in enumerate(view.rdc2_frames(**kw)):
        out[f"{tag}_{i}_cube"] = proc.prepare(fr["cube"]).astype(np.complex64)
        out[f"{tag}_{i}_t"] = np.array([fr["t0"], fr["t1"]])
        out[f"{tag}_{i}_doppler"] = fr["doppler_hz"]
        print(tag, i, fr["t0"], fr["t1"], fr["cube"].shape)
late = ru.RDCView("calib_checks/array_rdc0.h5", "all", "x", "status", tstart=5, gap_fill="zero")
late.select_delay(float(view.delay_ms[0]), float(view.delay_ms[-1]))
assert np.array_equal(late.stands, view.stands)
for i, fr in enumerate(late.rdc2_frames()):
    out[f"late_{i}_cube"] = proc.prepare(fr["cube"]).astype(np.complex64)
    out[f"late_{i}_t"] = np.array([fr["t0"], fr["t1"]])
    out[f"late_{i}_doppler"] = fr["doppler_hz"]
    print("late", i, fr["t0"], fr["t1"], fr["cube"].shape)
for name in ("cal_13p5", "cal_first_half"):
    c = ArrayCalibration.load(f"calib_checks/{name}.h5")
    out[f"{name}_stands"], out[f"{name}_gain"], out[f"{name}_flagged"] = c.stands, c.gain, c.flagged
np.savez_compressed("calib_checks/frames.npz", **out)
print("saved", os.path.getsize("calib_checks/frames.npz") / 1e6, "MB")

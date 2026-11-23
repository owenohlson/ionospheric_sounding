"""Scratch check: the native TBN reader returns exactly what LSL returns on a clean file."""
import sys
sys.path.insert(0, "tools")
import numpy as np
from lsl.reader.ldp import LWADataFile
import lwa_utils as L

path = sys.argv[1] if len(sys.argv) > 1 else "calib_checks/clean.dat"
idf = LWADataFile(path)
freq = idf.get_info("freq1")
_, stamp, lsl_data = idf.read(2.0)
idf.close()
r = L.TBNReader(path)
rows = np.arange(lsl_data.shape[0])
ours = np.concatenate([b for b, _ in r.read_blocks(rows, 0.0, 2.0)], axis=1)[:, :lsl_data.shape[1]]
print("LSL start", stamp.utc_datetime, " native start", r.utc_at(0.0))
print("LSL tuning", freq, " native", r.tuning_frequency_hz)
print("shapes", lsl_data.shape, ours.shape, " identical:", np.array_equal(lsl_data, ours),
      " max |diff|", float(np.nanmax(np.abs(lsl_data - ours))))

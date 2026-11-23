#!/bin/bash
# Self-calibration check on the 2023-12-09 TBN file. Run from the repo root: bash calib_checks.sh
# Writes everything to calib_checks/ (about 1.5 GB; delete it afterwards). Re-runs reuse the RDC0.
cd "$(dirname "$0")"
PY="${PY:-lwa_env/bin/python}"
OUT=calib_checks
export LWA_SSMIF=tools/resources/ssmif/LD021_0001.tgz          # station description for the observation date
mkdir -p "$OUT"
{
  echo "== 1. unit tests"
  "$PY" -m unittest discover -s tests 2>&1 | tail -3

  if [ ! -f "$OUT/array_rdc0.h5" ]; then
    echo; echo "== 2. clean TBN and all-stand x-pol RDC0 (SSMIF-flagged stands dropped)"
    "$PY" tools/preprocess_tbn.py data/060287_010274515_s0010.dat --output "$OUT/clean.dat" | tail -2
    "$PY" tools/export_rdc0.py "$OUT/clean.dat" --sweep-frequency 2 --bandwidth=-99.259377e3 \
        --pol x --bad-stands status --gap-fill zero --output "$OUT/array_rdc0.h5"
  fi

  echo; echo "== 3. self-calibration on the whole file (held-out validation inside)"
  "$PY" tools/calibrate_array.py "$OUT/array_rdc0.h5" --integration-time 2.5 --hop-time 2.5 \
      --output "$OUT/cal_13p5.h5" --csv "$OUT/cal_13p5.csv"

  echo; echo "== 4. independent time test: calibrate on 0-5 s, check on 5 s onwards"
  "$PY" tools/calibrate_array.py "$OUT/array_rdc0.h5" --tend 5 --integration-time 2.5 --hop-time 2.5 \
      --output "$OUT/cal_first_half.h5" | grep -A4 "Plane-wave fit"
  "$PY" tools/check_array_calibration.py "$OUT/array_rdc0.h5" --tstart 5 --calibration "$OUT/cal_first_half.h5"

  echo; echo "== 5. beamformed delay-Doppler and sky maps, without and with calibration"
  for tag in plain cal; do
    extra=(); [ "$tag" = cal ] && extra=(--calibration "$OUT/cal_13p5.h5")
    "$PY" tools/plot_delay_doppler.py "$OUT/array_rdc0.h5" --beamformer conventional "${extra[@]}" \
        --angle-output "$OUT/sky_$tag.png" --angle-cells 4 --output "$OUT/dd_$tag.png" | grep -iv "^frame"
  done
} 2>&1 | tee "$OUT/log2.txt"

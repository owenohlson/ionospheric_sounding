# Geometry-based Doppler compensation

The implementation is intentionally array-first. `doppler_compensation.py`
contains the geometry and signal processing, while `doppler_compensate.py`
adapts WAV/TBN files to complex NumPy arrays and writes a compensated WAV.
The existing functions in `plotting_utils.py` already accept arrays, so a
larger Python workflow can pass the compensated result directly to them and
skip the output file.

## Initial input contract

Receiver/recording metadata is JSON. A WAV needs all three fields; TBN normally
provides the start time and tuning frequency itself, but explicit JSON values or
command-line overrides may still be used.

```json
{
  "recording_start_utc": "2026-08-05T12:00:00Z",
  "carrier_frequency_hz": 29000000.0,
  "receiver_geodetic": {
    "latitude_deg": 48.4634,
    "longitude_deg": -123.3117,
    "height_m": 25.0
  }
}
```

`receiver_ecef_m: [x, y, z]` can replace `receiver_geodetic`.

The temporary ephemeris contract is CSV with ECEF/ITRF coordinates:

```csv
utc,x_m,y_m,z_m,vx_m_s,vy_m_s,vz_m_s
2026-08-05T12:00:00Z,1000000,2000000,6800000,-7000,2000,500
2026-08-05T12:00:01Z,993000,2002000,6800500,-7001,1999,499
```

Positions are metres and velocities are metres per second in the rotating ECEF
frame. The ephemeris must cover the entire selected recording interval;
extrapolation is rejected.

## Command line

```sh
python tools/doppler_compensate.py \
  recording.wav receiver.json marmotsat_ecef.csv compensated.wav
```

For TBN, select one stream using `--stand` and `--pol`. `--tstart` and `--tend`
limit either input type. The output is a stereo 32-bit float WAV (I in the first
channel, Q in the second) plus `compensated.wav.json`, which records the timing,
carrier, model, source files, and predicted Doppler/range extrema.

## Model and limitations

The current model is narrowband, one-way geometric Doppler:

`f_d = -(f_c / c) * range_rate`

Positive range rate means MarmotSat is receding and therefore predicts negative
Doppler. Compensation integrates that frequency history and conjugate-rotates
the IQ samples. This removes carrier Doppler but does not yet correct the much
smaller wideband time scaling or propagation-delay walk across an LFM sweep.

The first implementation loads the selected interval into memory. The core API
was separated so a streaming block adapter can be added without changing the
geometry or plotting code.

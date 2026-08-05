#!/usr/bin/env python3
"""Apply one-way geometry-based Doppler compensation to WAV or TBN IQ data."""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import timedelta, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

from doppler_compensation import (
    CartesianStateSeries,
    compensate_iq_doppler,
    geodetic_to_ecef,
    parse_utc,
    predict_one_way_doppler,
)
from lfm_utils import load_iq_audio


def load_metadata(filename: str) -> dict:
    with open(filename, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise ValueError("receiver metadata must be a JSON object")
    return metadata


def receiver_ecef_from_metadata(metadata: dict) -> np.ndarray:
    if "receiver_ecef_m" in metadata:
        position = np.asarray(metadata["receiver_ecef_m"], dtype=np.float64)
        if position.shape != (3,):
            raise ValueError("receiver_ecef_m must be [x, y, z]")
        return position

    geodetic = metadata.get("receiver_geodetic")
    if not isinstance(geodetic, dict):
        raise ValueError(
            "metadata must contain receiver_ecef_m or receiver_geodetic"
        )
    return geodetic_to_ecef(
        geodetic["latitude_deg"],
        geodetic["longitude_deg"],
        geodetic.get("height_m", 0.0),
    )


def load_ecef_ephemeris_csv(filename: str) -> CartesianStateSeries:
    required = ("utc", "x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s")
    timestamps = []
    states = []
    with open(filename, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [name for name in required if name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"ephemeris CSV is missing columns: {', '.join(missing)}")
        for row_number, row in enumerate(reader, start=2):
            try:
                timestamps.append(parse_utc(row["utc"]))
                states.append([float(row[name]) for name in required[1:]])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid ephemeris row {row_number}: {exc}") from exc

    if len(timestamps) < 2:
        raise ValueError("ephemeris CSV must contain at least two data rows")
    epoch = timestamps[0]
    time_s = np.array([(timestamp - epoch).total_seconds() for timestamp in timestamps])
    state_array = np.asarray(states, dtype=np.float64)
    return CartesianStateSeries(
        epoch_utc=epoch,
        time_s=time_s,
        position_m=state_array[:, :3],
        velocity_m_s=state_array[:, 3:],
        frame="ecef",
    )


def timestamp_to_datetime(timestamp):
    if timestamp is None:
        return None
    if hasattr(timestamp, "utc_datetime"):
        timestamp = timestamp.utc_datetime
    elif hasattr(timestamp, "datetime"):
        timestamp = timestamp.datetime
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


def load_wav(args, metadata: dict):
    info = sf.info(args.input_file)
    start_s = 0.0 if args.tstart is None else args.tstart
    end_s = info.frames / info.samplerate if args.tend is None else args.tend
    if not 0.0 <= start_s < end_s <= info.frames / info.samplerate:
        raise ValueError("requested WAV interval is outside the file")

    start_sample = int(round(start_s * info.samplerate))
    end_sample = int(round(end_s * info.samplerate))
    iq, sample_rate = load_iq_audio(args.input_file, start=start_sample, stop=end_sample)
    file_start = parse_utc(args.start_utc or metadata["recording_start_utc"])
    recording_start = file_start + timedelta(seconds=start_sample / sample_rate)
    carrier = args.carrier_frequency or metadata.get("carrier_frequency_hz")
    if carrier is None:
        raise ValueError("WAV input requires carrier_frequency_hz in metadata or --carrier-frequency")
    return iq, float(sample_rate), recording_start, float(carrier), None


def load_tbn(args, metadata: dict):
    from tbn_utils import (
        lsl_open_tbn,
        lsl_read_block_for_one_stream,
        tbn_usable_duration,
    )

    data_file = lsl_open_tbn(args.input_file)
    sample_rate = float(data_file.get_info("sample_rate"))
    start_s = 0.0 if args.tstart is None else args.tstart
    end_s = tbn_usable_duration(data_file) if args.tend is None else args.tend
    if not 0.0 <= start_s < end_s <= tbn_usable_duration(data_file):
        raise ValueError("requested TBN interval is outside the usable file duration")
    iq, timestamp, gap_info = lsl_read_block_for_one_stream(
        data_file,
        start_s,
        end_s - start_s,
        stand_id=args.stand,
        pol=args.pol,
        gap_fill=args.gap_fill,
        return_gap_info=True,
    )
    recording_start = timestamp_to_datetime(timestamp)
    if args.start_utc:
        recording_start = parse_utc(args.start_utc)
    if recording_start is None:
        recording_start = parse_utc(metadata["recording_start_utc"]) + timedelta(seconds=start_s)
    carrier = args.carrier_frequency or metadata.get("carrier_frequency_hz")
    if carrier is None:
        carrier = float(data_file.get_info("freq1"))
    return iq, sample_rate, recording_start, float(carrier), gap_info


def write_complex_wav(filename: str, iq: np.ndarray, sample_rate_hz: float):
    output = np.column_stack((iq.real, iq.imag)).astype(np.float32, copy=False)
    sf.write(filename, output, int(round(sample_rate_hz)), subtype="FLOAT")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Remove predicted one-way MarmotSat/receiver Doppler from complex IQ.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_file", help="Stereo IQ WAV or LWA TBN input")
    parser.add_argument("receiver_metadata", help="Receiver/recording JSON metadata")
    parser.add_argument("ephemeris_csv", help="ECEF MarmotSat state CSV")
    parser.add_argument("output_wav", help="Compensated stereo float WAV output")
    parser.add_argument("--tstart", type=float, default=None, help="Start offset in input seconds")
    parser.add_argument("--tend", type=float, default=None, help="End offset in input seconds")
    parser.add_argument("--start-utc", default=None, help="Override UTC time of selected input start")
    parser.add_argument("--carrier-frequency", type=float, default=None, help="Actual RF carrier in Hz")
    parser.add_argument("--stand", type=int, default=1, help="TBN stand ID")
    parser.add_argument("--pol", choices=("x", "y", "0", "1"), default="x", help="TBN polarization")
    parser.add_argument("--gap-fill", choices=("zero", "nan"), default="zero", help="TBN gap fill")
    return parser


def main():
    args = build_parser().parse_args()
    metadata = load_metadata(args.receiver_metadata)
    receiver_position_m = receiver_ecef_from_metadata(metadata)
    trajectory = load_ecef_ephemeris_csv(args.ephemeris_csv)

    if Path(args.input_file).suffix.lower() == ".tbn":
        iq, sample_rate, recording_start, carrier, gap_info = load_tbn(args, metadata)
    else:
        iq, sample_rate, recording_start, carrier, gap_info = load_wav(args, metadata)

    first_ephemeris_second = trajectory.seconds_at(recording_start)
    sample_time_s = first_ephemeris_second + np.arange(iq.size, dtype=np.float64) / sample_rate
    doppler_hz, range_m, range_rate_m_s = predict_one_way_doppler(
        trajectory,
        sample_time_s,
        receiver_position_m,
        carrier,
    )
    compensated, _ = compensate_iq_doppler(iq, doppler_hz, sample_rate)
    write_complex_wav(args.output_wav, compensated, sample_rate)

    sidecar = {
        "format_version": 1,
        "source_file": os.path.abspath(args.input_file),
        "ephemeris_file": os.path.abspath(args.ephemeris_csv),
        "receiver_metadata_file": os.path.abspath(args.receiver_metadata),
        "recording_start_utc": recording_start.isoformat().replace("+00:00", "Z"),
        "sample_rate_hz": sample_rate,
        "sample_count": int(compensated.size),
        "carrier_frequency_hz": carrier,
        "geometry_model": "one_way_ecef_narrowband",
        "doppler_hz_min": float(np.min(doppler_hz)),
        "doppler_hz_max": float(np.max(doppler_hz)),
        "range_m_min": float(np.min(range_m)),
        "range_m_max": float(np.max(range_m)),
        "range_rate_m_s_min": float(np.min(range_rate_m_s)),
        "range_rate_m_s_max": float(np.max(range_rate_m_s)),
        "tbn_gap_info": gap_info,
    }
    sidecar_path = f"{args.output_wav}.json"
    with open(sidecar_path, "w", encoding="utf-8") as handle:
        json.dump(sidecar, handle, indent=2, default=str)
        handle.write("\n")

    print(f"Wrote {args.output_wav}")
    print(f"Wrote {sidecar_path}")
    print(f"Predicted Doppler range: {doppler_hz.min():.3f} to {doppler_hz.max():.3f} Hz")


if __name__ == "__main__":
    main()

# rdc_utils.py

import csv
import json
import os
from datetime import datetime, timedelta

import numpy as np

from lfm_utils import dechirp_delay_axis_ms, window_from_arg


BAD_LWA_STANDS = [
    10, 16, 27, 76, 108, 133, 228, 229, 232, 164,
    93, 60, 20, 163, 155, 116, 6, 110, 178, 21,
]

LWA_STAND_POSITIONS_M = {
    1: (-37.116, 26.191, 2.503),
    2: (-35.415, 16.967, 2.089),
    3: (-34.649, 7.982, 1.733),
    4: (-37.380, 2.164, 1.510),
    5: (-35.214, -2.950, 1.295),
    6: (-34.015, -9.858, 1.047),
    7: (-30.027, 10.815, 1.770),
    8: (-28.241, 3.739, 1.349),
    9: (-28.607, -2.446, 1.279),
    10: (-29.746, -7.293, 1.060),
    11: (-23.081, -3.313, 1.072),
    12: (-24.591, -8.983, 1.037),
    13: (-27.789, -12.842, 0.912),
    14: (-29.159, -18.666, 0.768),
    15: (-19.391, -10.384, 0.848),
    16: (-14.446, -12.824, 0.602),
    17: (-22.565, -18.236, 0.622),
    18: (-26.046, -22.631, 0.752),
    19: (-18.222, -21.537, 0.428),
    20: (-19.875, -27.212, 0.624),
    21: (-12.788, -20.833, 0.396),
    22: (-7.948, -22.433, 0.253),
    23: (-12.562, -28.202, 0.408),
    24: (-16.697, -31.083, 0.582),
    25: (-9.245, -35.138, 0.236),
    26: (-7.769, -29.538, 0.216),
    27: (-2.667, -34.971, 0.030),
    28: (0.269, -39.160, 0.067),
    29: (-32.961, 38.956, 2.572),
    30: (-35.368, 32.892, 2.497),
    31: (-30.383, 33.437, 2.320),
    32: (-26.056, 30.856, 2.149),
    33: (-30.464, 26.220, 2.272),
    34: (-33.107, 21.387, 2.149),
    35: (-25.167, 25.915, 2.000),
    36: (-19.767, 25.513, 1.875),
    37: (-28.050, 21.843, 2.042),
    38: (-27.214, 16.877, 1.743),
    39: (-21.737, 19.767, 1.716),
    40: (-16.723, 19.990, 1.588),
    41: (-20.916, 12.116, 1.470),
    42: (-23.334, 6.907, 1.356),
    43: (-22.691, 1.916, 1.229),
    44: (-16.620, 14.979, 1.474),
    45: (-11.639, 13.161, 1.653),
    46: (-15.529, 10.049, 1.518),
    47: (-15.650, 4.986, 1.289),
    48: (-17.898, 0.597, 1.051),
    49: (-17.758, -4.604, 0.915),
    50: (-9.978, 8.644, 1.514),
    51: (-12.810, -0.379, 1.157),
    52: (-13.514, -7.230, 0.830),
    53: (-9.769, 3.587, 1.303),
    54: (-5.671, -0.265, 1.266),
    55: (-0.708, -3.946, 1.176),
    56: (-9.075, -3.979, 1.073),
    57: (-7.351, -9.308, 0.817),
    58: (-8.441, -16.344, 0.445),
    59: (-1.919, -9.550, 0.922),
    60: (-3.702, -14.475, 0.666),
    61: (-2.531, -19.414, 0.398),
    62: (2.136, -12.956, 0.935),
    63: (6.819, -14.674, 0.943),
    64: (3.252, -18.353, 0.558),
    65: (1.622, -25.636, 0.026),
    66: (-0.610, -30.156, 0.012),
    67: (6.566, -22.201, 0.294),
    68: (11.484, -23.195, 0.186),
    69: (4.111, -32.129, -0.165),
    70: (8.586, -27.912, -0.075),
    71: (12.172, -31.544, -0.269),
    72: (8.732, -35.673, -0.225),
    73: (7.338, -40.428, 0.031),
    74: (13.368, -40.001, -0.262),
    75: (16.662, -33.966, -0.409),
    76: (20.057, -37.840, -0.519),
    77: (-30.395, 47.180, 2.637),
    78: (-25.769, 53.752, 2.372),
    79: (-27.913, 42.785, 2.559),
    80: (-25.915, 36.471, 2.299),
    81: (-23.145, 46.941, 2.545),
    82: (-19.990, 39.217, 2.091),
    83: (-19.250, 31.459, 1.853),
    84: (-13.640, 31.873, 1.810),
    85: (-13.105, 25.034, 1.760),
    86: (-8.393, 27.575, 1.795),
    87: (-11.740, 19.678, 1.732),
    88: (-7.674, 22.650, 1.686),
    89: (-7.556, 16.193, 1.530),
    90: (-6.435, 12.096, 1.579),
    91: (-1.257, 14.585, 1.167),
    92: (-1.949, 9.532, 1.317),
    93: (-4.994, 5.045, 1.512),
    94: (3.036, 12.261, 0.773),
    95: (-0.473, 3.000, 1.365),
    96: (2.515, 7.286, 0.849),
    97: (6.695, 4.623, 0.738),
    98: (3.814, 0.401, 0.906),
    99: (6.398, -4.037, 1.000),
    100: (4.304, -8.465, 1.111),
    101: (9.302, -8.314, 0.843),
    102: (11.259, -3.762, 0.648),
    103: (11.980, -13.038, 0.928),
    104: (13.915, -17.388, 0.599),
    105: (15.636, -8.954, 0.404),
    106: (17.489, -13.614, 0.306),
    107: (17.716, -22.340, 0.030),
    108: (23.326, -17.488, -0.085),
    109: (22.835, -25.800, -0.417),
    110: (18.079, -29.157, -0.280),
    111: (21.667, -32.606, -0.555),
    112: (28.497, -28.198, -0.704),
    113: (29.601, -33.940, -0.758),
    114: (26.749, -21.225, -0.485),
    115: (-20.675, 55.615, 2.144),
    116: (-17.257, 60.821, 1.975),
    117: (-14.781, 56.436, 1.783),
    118: (-17.631, 46.381, 2.365),
    119: (-13.867, 51.556, 1.850),
    120: (-9.279, 52.466, 1.647),
    121: (-12.501, 44.423, 2.025),
    122: (-6.234, 48.484, 1.516),
    123: (-11.453, 36.454, 1.871),
    124: (-6.052, 40.325, 1.687),
    125: (0.801, 38.817, 1.437),
    126: (3.307, 34.522, 1.432),
    127: (-6.985, 33.139, 1.827),
    128: (-3.642, 29.397, 1.713),
    129: (-1.934, 34.098, 1.632),
    130: (-1.788, 24.694, 1.287),
    131: (-1.283, 19.608, 1.164),
    132: (1.945, 28.903, 1.266),
    133: (2.960, 22.230, 0.957),
    134: (3.196, 17.364, 0.775),
    135: (6.384, 26.267, 0.953),
    136: (8.809, 20.803, 0.491),
    137: (7.453, 14.669, 0.542),
    138: (7.251, 9.700, 0.615),
    139: (11.772, 17.182, 0.421),
    140: (16.263, 14.598, 0.210),
    141: (11.689, 12.129, 0.302),
    142: (11.577, 7.018, 0.321),
    143: (10.857, 1.904, 0.431),
    144: (16.370, 9.556, 0.143),
    145: (15.594, 4.099, 0.122),
    146: (15.377, -0.839, 0.313),
    147: (20.139, 6.268, 0.061),
    148: (24.754, 2.127, -0.016),
    149: (18.387, -4.853, 0.177),
    150: (20.331, -0.188, 0.005),
    151: (24.124, -3.450, -0.019),
    152: (22.131, -10.791, 0.084),
    153: (27.396, -7.248, -0.173),
    154: (32.444, -7.090, -0.189),
    155: (26.604, -13.138, -0.207),
    156: (31.452, -12.024, -0.337),
    157: (30.453, -18.196, -0.503),
    158: (34.163, -26.369, -0.695),
    159: (37.018, -18.137, -0.603),
    160: (42.258, -21.947, -0.754),
    161: (-9.709, 60.230, 1.726),
    162: (-6.573, 67.046, 1.904),
    163: (1.102, 68.071, 1.914),
    164: (-3.188, 61.867, 1.722),
    165: (-3.089, 55.079, 1.508),
    166: (-0.415, 50.926, 1.430),
    167: (-1.945, 46.173, 1.394),
    168: (4.601, 53.299, 1.406),
    169: (4.264, 47.214, 1.275),
    170: (6.045, 41.226, 1.209),
    171: (8.727, 49.186, 1.182),
    172: (12.415, 44.935, 0.949),
    173: (7.956, 36.172, 1.082),
    174: (7.025, 31.168, 1.160),
    175: (10.940, 40.187, 0.966),
    176: (15.528, 37.509, 0.790),
    177: (12.657, 32.850, 0.716),
    178: (11.029, 28.008, 0.586),
    179: (11.384, 22.953, 0.440),
    180: (18.089, 32.781, 0.603),
    181: (23.795, 32.104, 0.527),
    182: (15.579, 25.641, 0.395),
    183: (15.676, 20.419, 0.321),
    184: (20.688, 27.366, 0.426),
    185: (26.416, 26.276, 0.371),
    186: (20.603, 21.170, 0.399),
    187: (21.115, 16.038, 0.203),
    188: (21.845, 10.992, 0.082),
    189: (26.029, 20.596, 0.214),
    190: (31.022, 20.560, 0.195),
    191: (26.003, 13.981, 0.141),
    192: (25.537, 7.622, 0.011),
    193: (30.640, 15.521, 0.103),
    194: (36.528, 14.684, 0.087),
    195: (31.237, 10.601, 0.020),
    196: (37.544, 9.732, -0.059),
    197: (43.299, 6.214, -0.221),
    198: (33.048, 5.910, -0.057),
    199: (31.899, -1.408, -0.119),
    200: (38.242, 3.139, -0.183),
    201: (45.437, 1.618, -0.286),
    202: (50.423, 1.424, -0.450),
    203: (50.330, 6.373, -0.365),
    204: (38.262, -9.115, -0.379),
    205: (39.354, -1.778, -0.226),
    206: (43.013, -5.166, -0.330),
    207: (49.241, -4.994, -0.636),
    208: (42.625, -11.424, -0.593),
    209: (45.262, -16.079, -0.756),
    210: (48.681, -10.261, -0.599),
    211: (6.559, 69.600, 1.752),
    212: (8.380, 64.988, 1.673),
    213: (4.930, 61.471, 1.582),
    214: (11.866, 68.774, 1.575),
    215: (12.729, 60.853, 1.397),
    216: (8.568, 57.998, 1.390),
    217: (11.394, 53.865, 1.256),
    218: (18.951, 60.111, 1.315),
    219: (16.635, 55.308, 1.212),
    220: (15.649, 48.758, 0.998),
    221: (20.745, 46.365, 0.871),
    222: (17.694, 42.383, 0.861),
    223: (21.432, 52.743, 1.115),
    224: (26.781, 51.341, 0.948),
    225: (22.372, 37.925, 0.679),
    226: (27.391, 37.947, 0.616),
    227: (27.183, 43.232, 0.713),
    228: (34.550, 40.306, 0.589),
    229: (29.111, 30.514, 0.395),
    230: (31.903, 25.780, 0.277),
    231: (33.592, 35.390, 0.523),
    232: (37.691, 31.841, 0.344),
    233: (36.766, 26.713, 0.309),
    234: (37.923, 19.426, 0.113),
    235: (41.694, 27.600, 0.212),
    236: (41.393, 15.887, -0.002),
    237: (42.986, 11.188, -0.124),
    238: (44.071, 20.526, 0.006),
    239: (49.973, 20.084, -0.060),
    240: (51.139, 15.109, -0.230),
    241: (19.040, 66.377, 1.377),
    242: (23.870, 65.181, 1.284),
    243: (25.358, 60.130, 1.145),
    244: (28.739, 55.986, 1.043),
    245: (32.934, 58.672, 1.053),
    246: (34.757, 54.066, 0.929),
    247: (31.774, 49.309, 0.850),
    248: (36.585, 45.515, 0.733),
    249: (38.607, 58.520, 1.039),
    250: (42.856, 48.658, 0.772),
    251: (42.414, 43.849, 0.548),
    252: (40.058, 38.100, 0.489),
    253: (44.420, 33.227, 0.427),
    254: (49.244, 25.112, -0.064),
    255: (48.167, 37.230, 0.300),
}


def pol_to_indices(pol):
    p = str(pol).strip().lower()
    if p == "both":
        return [0, 1]
    if p in ("x", "0"):
        return [0]
    if p in ("y", "1"):
        return [1]
    raise ValueError("--pol must be both, x/y, or 0/1")


def _timestamp_datetime(timestamp):
    if timestamp is None:
        return None
    if hasattr(timestamp, "utc_datetime"):
        return timestamp.utc_datetime
    if hasattr(timestamp, "datetime"):
        return timestamp.datetime
    return timestamp


def _timestamp_at(start_timestamp, seconds_after_start):
    start_dt = _timestamp_datetime(start_timestamp)
    if isinstance(start_dt, datetime):
        return start_dt + timedelta(seconds=float(seconds_after_start))
    return None


def _timestamp_delta_seconds(later_timestamp, earlier_timestamp):
    later_dt = _timestamp_datetime(later_timestamp)
    earlier_dt = _timestamp_datetime(earlier_timestamp)
    if isinstance(later_dt, datetime) and isinstance(earlier_dt, datetime):
        return (later_dt - earlier_dt).total_seconds()
    return None


def _format_timestamp(timestamp):
    dt = _timestamp_datetime(timestamp)
    if dt is None:
        return "unknown UTC"
    if isinstance(dt, datetime):
        return f"{dt.strftime('%H:%M:%S.%f')[:-3]} UTC"
    return f"{dt} UTC"


def _gap_warning(reason, plot_time, timestamp):
    print(f"{reason} at local plot time {plot_time:.3f}s ({_format_timestamp(timestamp)})", flush=True)


def _gap_fill_array(n_samples, gap_fill):
    if gap_fill == "zero":
        return np.zeros(n_samples, dtype=np.complex64)
    if gap_fill == "nan":
        return np.full(n_samples, np.nan + 1j * np.nan, dtype=np.complex64)
    raise ValueError("--gap-fill must be 'nan' or 'zero'")


def _gap_fill_block(n_streams, n_samples, gap_fill):
    block = _gap_fill_array(n_samples, gap_fill)
    if n_streams == 1:
        return block[None, :]
    return np.tile(block[None, :], (n_streams, 1))


def parse_stands(stand_spec="all", max_stand=256, drop_bad=False):
    if stand_spec is None or str(stand_spec).strip().lower() == "all":
        stands = list(range(1, int(max_stand) + 1))
    else:
        stands = []
        for part in str(stand_spec).split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                lo, hi = part.split("-", 1)
                stands.extend(range(int(lo), int(hi) + 1))
            else:
                stands.append(int(part))
    stands = sorted(dict.fromkeys(stands))
    if drop_bad:
        bad = set(BAD_LWA_STANDS)
        stands = [stand for stand in stands if stand not in bad]
    return stands


def channel_selection(stands, pol):
    pol_indices = pol_to_indices(pol)
    stream_indices = []
    channel_stands = []
    channel_pols = []
    for stand in stands:
        for pol_idx in pol_indices:
            stream_indices.append(2 * (int(stand) - 1) + pol_idx)
            channel_stands.append(int(stand))
            channel_pols.append(int(pol_idx))
    return np.asarray(stream_indices, dtype=int), np.asarray(channel_stands, dtype=int), np.asarray(channel_pols, dtype=int)


def polarization_metadata(channel_pols):
    channel_pols = np.asarray(channel_pols, dtype=int)
    unique_pols = sorted(int(pol) for pol in np.unique(channel_pols))
    if unique_pols == [0]:
        mode = "x"
    elif unique_pols == [1]:
        mode = "y"
    elif unique_pols == [0, 1]:
        mode = "both"
    else:
        mode = "unknown"
    return {
        "polarization_mode": mode,
        "polarization_streams": unique_pols,
        "n_polarizations": len(unique_pols),
    }


def estimate_rdc0_bytes(adc_count, pulses, channels, dtype=np.float32):
    return int(adc_count) * int(pulses) * int(channels) * np.dtype(dtype).itemsize * 2


def estimate_rdc2_bytes(frames, delay_bins, doppler_bins, channels, dtype=np.float32):
    return int(frames) * int(delay_bins) * int(doppler_bins) * int(channels) * np.dtype(dtype).itemsize * 2


def format_bytes(n_bytes):
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(n_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0


def reshape_samples_to_rdc0(samples, sweep_length, start_offset_samples=0, adc_start=0, adc_count=None):
    samples = np.asarray(samples)
    if samples.ndim == 1:
        samples = samples[None, :]
    if samples.ndim != 2:
        raise ValueError("samples must be 1D or 2D (channels, samples)")
    if start_offset_samples < 0:
        raise ValueError("start_offset_samples must be >= 0")
    if adc_start < 0:
        raise ValueError("adc_start must be >= 0")
    if adc_count is None:
        adc_count = sweep_length - adc_start
    if adc_count <= 0 or adc_start + adc_count > sweep_length:
        raise ValueError("adc_start/adc_count must select samples inside one sweep")

    samples = samples[:, start_offset_samples:]
    pulses = samples.shape[-1] // sweep_length
    trimmed = samples[:, :pulses * sweep_length]
    cube = trimmed.reshape(samples.shape[0], pulses, sweep_length)
    cube = cube[:, :, adc_start:adc_start + adc_count]
    return np.transpose(cube, (2, 1, 0))


def dechirp_range_fft_rdc0(rdc0, lfm_config, window="hann", adc_start=0):
    """RDC0 (adc, pulses, channels) -> RDC1 (ranges, pulses, channels)."""
    rdc0 = np.asarray(rdc0)
    if rdc0.ndim != 3:
        raise ValueError("rdc0 must have shape (adc, pulses, channels)")
    adc_len = rdc0.shape[0]
    reference = lfm_config.waveform[int(adc_start):int(adc_start) + adc_len].astype(rdc0.dtype, copy=False)
    if reference.size != adc_len:
        raise ValueError("adc_start plus RDC0 ADC length exceeds the reference sweep length")
    w = window_from_arg(adc_len, window).astype(np.float32, copy=False)
    coherent_gain = np.mean(w)
    if coherent_gain == 0:
        raise ValueError(f"Window '{window}' has zero coherent gain")

    beat = rdc0 * np.conj(reference)[:, None, None]
    return np.fft.fftshift(np.fft.fft(beat * w[:, None, None], axis=0) / coherent_gain, axes=0).astype(np.complex64)


def dechirp_range_fft_delay_band_rdc0(rdc0, lfm_config, d_min, d_max, window="hann", adc_start=0):
    """RDC0 -> selected RDC1 delay band without keeping the full RDC1 after selection."""
    rdc0 = np.asarray(rdc0)
    if rdc0.ndim != 3:
        raise ValueError("rdc0 must have shape (adc, pulses, channels)")
    if d_min is None or d_max is None:
        raise ValueError("d_min and d_max are required for delay-band RDC1 processing")

    adc_len = rdc0.shape[0]
    delay_ms, delay_order = dechirp_delay_axis_ms(lfm_config, n_bins=adc_len)
    mask = (delay_ms >= float(d_min)) & (delay_ms <= float(d_max))
    if not np.any(mask):
        raise ValueError(
            "No delay bins selected. "
            f"Available delay range is {delay_ms.min():.3f} to {delay_ms.max():.3f} ms."
        )

    reference = lfm_config.waveform[int(adc_start):int(adc_start) + adc_len].astype(rdc0.dtype, copy=False)
    if reference.size != adc_len:
        raise ValueError("adc_start plus RDC0 ADC length exceeds the reference sweep length")
    w = window_from_arg(adc_len, window).astype(np.float32, copy=False)
    coherent_gain = np.mean(w)
    if coherent_gain == 0:
        raise ValueError(f"Window '{window}' has zero coherent gain")

    beat = rdc0 * np.conj(reference)[:, None, None]
    spectra = np.fft.fftshift(np.fft.fft(beat * w[:, None, None], axis=0) / coherent_gain, axes=0)
    return spectra[delay_order[mask], :, :].astype(np.complex64), delay_ms[mask]


def select_rdc1_delay_band(rdc1, lfm_config, d_min=None, d_max=None):
    delay_ms, delay_order = dechirp_delay_axis_ms(lfm_config, n_bins=rdc1.shape[0])
    ordered = rdc1[delay_order, :, :]
    mask = np.ones(delay_ms.shape, dtype=bool)
    if d_min is not None:
        mask &= delay_ms >= d_min
    if d_max is not None:
        mask &= delay_ms <= d_max
    if not np.any(mask):
        raise ValueError(
            "No delay bins selected. "
            f"Available delay range is {delay_ms.min():.3f} to {delay_ms.max():.3f} ms."
        )
    return ordered[mask, :, :], delay_ms[mask]


def doppler_fft_rdc1(rdc1, prf, window="hann", nfft=None):
    """RDC1 (ranges, pulses, channels) -> RDC2 (ranges, doppler, channels)."""
    rdc1 = np.asarray(rdc1)
    if rdc1.ndim != 3:
        raise ValueError("rdc1 must have shape (ranges, pulses, channels)")
    pulses = rdc1.shape[1]
    if nfft is None:
        nfft = 1 << int(np.ceil(np.log2(max(pulses, 1))))
    w = window_from_arg(pulses, window).astype(np.float32, copy=False)
    coherent_gain = np.mean(w)
    if coherent_gain == 0:
        raise ValueError(f"Window '{window}' has zero coherent gain")
    x = rdc1 * (w[None, :, None] / coherent_gain)
    rdc2 = np.fft.fftshift(np.fft.fft(x, n=nfft, axis=1), axes=1).astype(np.complex64)
    fd_hz = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / float(prf)))
    return rdc2, fd_hz


def delay_doppler_power_db(rdc2, combine="sum"):
    power = np.abs(rdc2) ** 2
    if combine == "sum":
        out = np.sum(power, axis=2)
    elif combine == "mean":
        out = np.mean(power, axis=2)
    elif combine == "max":
        out = np.max(power, axis=2)
    else:
        raise ValueError("--channel-combine must be one of: sum, mean, max")
    return 10.0 * np.log10(out + 1e-12)


def load_lwa_antenna_positions(channel_stands, strict=True, return_mask=False):
    positions = []
    missing = []
    mask = []
    for stand in channel_stands:
        stand = int(stand)
        if stand not in LWA_STAND_POSITIONS_M:
            missing.append(stand)
            mask.append(False)
        else:
            positions.append(LWA_STAND_POSITIONS_M[stand])
            mask.append(True)
    if missing and strict:
        raise ValueError(
            "Built-in LWA position map is missing stand(s): "
            f"{sorted(set(missing))}. The bundled coordInfo files contained stands 1-255."
        )
    positions = np.asarray(positions, dtype=np.float64)
    if return_mask:
        return positions, np.asarray(mask, dtype=bool), sorted(set(missing))
    return positions


def load_antenna_positions_csv(path, channel_stands):
    positions_by_stand = {}
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            stand = int(row["stand"])
            positions_by_stand[stand] = [float(row["x_m"]), float(row["y_m"]), float(row.get("z_m", 0.0))]
    positions = []
    missing = []
    for stand in channel_stands:
        if int(stand) not in positions_by_stand:
            missing.append(int(stand))
        else:
            positions.append(positions_by_stand[int(stand)])
    if missing:
        raise ValueError(f"Antenna position CSV is missing stand(s): {sorted(set(missing))}")
    return np.asarray(positions, dtype=np.float64)


def steering_vector(positions_m, wavelength_m, az_deg, el_deg):
    az = np.deg2rad(az_deg)
    el = np.deg2rad(el_deg)
    direction = np.array([
        np.cos(el) * np.sin(az),
        np.cos(el) * np.cos(az),
        np.sin(el),
    ])
    phase = 2.0 * np.pi / wavelength_m * (positions_m @ direction)
    return np.exp(1j * phase)


def capon_mvdr_spectrum(
    snapshot,
    positions_m,
    wavelength_m,
    az_grid_deg,
    el_grid_deg,
    diagonal_loading=1e-2,
    method="mvdr",
):
    """Spatial spectrum from one or more snapshots for a range-Doppler cell."""
    x = np.asarray(snapshot, dtype=np.complex128)
    if x.ndim == 1:
        x = x[None, :]
    elif x.ndim != 2:
        x = x.reshape(-1, x.shape[-1])

    good_snapshots = np.all(np.isfinite(x.real) & np.isfinite(x.imag), axis=1)
    if np.count_nonzero(good_snapshots) < 1:
        return np.full((len(el_grid_deg), len(az_grid_deg)), np.nan, dtype=np.float32)
    x = x[good_snapshots]

    good_channels = np.all(np.isfinite(x.real) & np.isfinite(x.imag), axis=0)
    if np.count_nonzero(good_channels) < 2:
        return np.full((len(el_grid_deg), len(az_grid_deg)), np.nan, dtype=np.float32)
    x = x[:, good_channels]
    pos = np.asarray(positions_m)[good_channels]

    R = (x.T @ np.conj(x)) / max(x.shape[0], 1)
    loading = float(diagonal_loading) * (np.trace(R).real / max(R.shape[0], 1) + 1e-12)
    R = R + loading * np.eye(R.shape[0])
    method = str(method).lower()
    R_inv = np.linalg.pinv(R) if method == "mvdr" else None

    spec = np.empty((len(el_grid_deg), len(az_grid_deg)), dtype=np.float32)
    for ei, el in enumerate(el_grid_deg):
        for ai, az in enumerate(az_grid_deg):
            a = steering_vector(pos, wavelength_m, az, el)
            if method == "mvdr":
                denom = np.conj(a) @ R_inv @ a
                spec[ei, ai] = float(1.0 / np.maximum(np.real(denom), 1e-30))
            elif method == "conventional":
                spec[ei, ai] = float(np.real(np.conj(a) @ R @ a) / max(len(a) ** 2, 1))
            else:
                raise ValueError("angle spectrum method must be mvdr or conventional")
    return spec


def write_complex_dataset(group, name, array, compression="gzip", compression_opts=4):
    compression_arg = None if compression == "none" else compression
    opts = None if compression in ("none", "lzf") else compression_opts
    group.create_dataset(f"{name}_real", data=np.asarray(array.real), compression=compression_arg, compression_opts=opts)
    group.create_dataset(f"{name}_imag", data=np.asarray(array.imag), compression=compression_arg, compression_opts=opts)


def read_complex_dataset(group, name):
    return group[f"{name}_real"][...] + 1j * group[f"{name}_imag"][...]


def _open_hdf5_read(path, h5py):
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"RDC0 HDF5 file not found: {path}")
    try:
        return h5py.File(path, "r")
    except OSError as exc:
        raise OSError(
            f"Could not open RDC HDF5 file '{path}'. This usually means the file is "
            "not a valid HDF5/RDC cache, is truncated/corrupted, or was written by a job "
            "that did not finish cleanly. Re-run the RDC exporter or point the input "
            "flag at a valid .h5 file."
        ) from exc


def _rdc0_group(h5, path):
    if "rdc0" not in h5:
        raise ValueError(f"HDF5 file '{path}' does not contain the required '/rdc0' group")
    grp = h5["rdc0"]
    required = ("real", "imag", "channel_stands", "channel_pols")
    missing = [name for name in required if name not in grp]
    if missing:
        raise ValueError(f"HDF5 file '{path}' is missing required RDC0 dataset(s): {', '.join(missing)}")
    if grp["real"].shape != grp["imag"].shape:
        raise ValueError(f"HDF5 file '{path}' has mismatched RDC0 real/imag dataset shapes")
    return grp


def _rdc2_group(h5, path):
    if "rdc2" not in h5:
        raise ValueError(f"HDF5 file '{path}' does not contain the required '/rdc2' group")
    grp = h5["rdc2"]
    required = (
        "real",
        "imag",
        "delay_ms",
        "doppler_hz",
        "frame_tstart_s",
        "frame_tend_s",
        "channel_stands",
        "channel_pols",
    )
    missing = [name for name in required if name not in grp]
    if missing:
        raise ValueError(f"HDF5 file '{path}' is missing required RDC2 dataset(s): {', '.join(missing)}")
    if grp["real"].shape != grp["imag"].shape:
        raise ValueError(f"HDF5 file '{path}' has mismatched RDC2 real/imag dataset shapes")
    if len(grp["real"].shape) != 4:
        raise ValueError(f"HDF5 file '{path}' RDC2 datasets must have shape (frames, delay, doppler, channels)")
    return grp


def load_rdc0_hdf5(path):
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required to load RDC HDF5 files") from exc
    path = os.path.expanduser(path)
    with _open_hdf5_read(path, h5py) as h5:
        grp = _rdc0_group(h5, path)
        rdc0 = grp["real"][...] + 1j * grp["imag"][...]
        channel_stands = grp["channel_stands"][...]
        channel_pols = grp["channel_pols"][...]
        adc_sample_index = grp["adc_sample_index"][...] if "adc_sample_index" in grp else np.arange(rdc0.shape[0])
        metadata = {}
        if "metadata_json" in h5.attrs:
            metadata = json.loads(h5.attrs["metadata_json"])
    return {
        "rdc0": rdc0.astype(np.complex64, copy=False),
        "channel_stands": channel_stands,
        "channel_pols": channel_pols,
        "adc_sample_index": adc_sample_index,
        "metadata": metadata,
    }


def load_rdc0_hdf5_metadata(path):
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required to load RDC HDF5 files") from exc
    path = os.path.expanduser(path)
    with _open_hdf5_read(path, h5py) as h5:
        grp = _rdc0_group(h5, path)
        shape = tuple(grp["real"].shape)
        channel_stands = grp["channel_stands"][...]
        channel_pols = grp["channel_pols"][...]
        adc_sample_index = grp["adc_sample_index"][...] if "adc_sample_index" in grp else np.arange(shape[0])
        metadata = {}
        if "metadata_json" in h5.attrs:
            metadata = json.loads(h5.attrs["metadata_json"])
    return {
        "shape": shape,
        "channel_stands": channel_stands,
        "channel_pols": channel_pols,
        "adc_sample_index": adc_sample_index,
        "metadata": metadata,
    }


def iter_rdc0_hdf5_channel_chunks(
    path,
    channel_chunk_size=8,
    channel_indices=None,
    pulse_start=None,
    pulse_stop=None,
):
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required to load RDC HDF5 files") from exc
    path = os.path.expanduser(path)
    channel_chunk_size = int(channel_chunk_size)
    if channel_chunk_size < 1:
        raise ValueError("channel_chunk_size must be >= 1")

    with _open_hdf5_read(path, h5py) as h5:
        grp = _rdc0_group(h5, path)
        real = grp["real"]
        imag = grp["imag"]
        n_channels = real.shape[2]
        pulse_start = 0 if pulse_start is None else int(pulse_start)
        pulse_stop = real.shape[1] if pulse_stop is None else int(pulse_stop)
        if pulse_start < 0 or pulse_stop > real.shape[1] or pulse_stop <= pulse_start:
            raise ValueError("pulse_start/pulse_stop select an invalid RDC0 pulse range")
        if channel_indices is None:
            selected = np.arange(n_channels, dtype=int)
        else:
            selected = np.asarray(channel_indices, dtype=int)
            if selected.ndim != 1:
                raise ValueError("channel_indices must be a 1D sequence")
            if selected.size and (np.min(selected) < 0 or np.max(selected) >= n_channels):
                raise ValueError("channel_indices contains an out-of-range channel")
        for start in range(0, selected.size, channel_chunk_size):
            stop = min(start + channel_chunk_size, selected.size)
            idx = selected[start:stop]
            chunk = real[:, pulse_start:pulse_stop, idx] + 1j * imag[:, pulse_start:pulse_stop, idx]
            yield start, stop, chunk.astype(np.complex64, copy=False)


def load_rdc2_hdf5_metadata(path):
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required to load RDC HDF5 files") from exc
    path = os.path.expanduser(path)
    with _open_hdf5_read(path, h5py) as h5:
        grp = _rdc2_group(h5, path)
        metadata = {}
        if "metadata_json" in h5.attrs:
            metadata = json.loads(h5.attrs["metadata_json"])
        return {
            "shape": tuple(grp["real"].shape),
            "delay_ms": grp["delay_ms"][...],
            "doppler_hz": grp["doppler_hz"][...],
            "frame_tstart_s": grp["frame_tstart_s"][...],
            "frame_tend_s": grp["frame_tend_s"][...],
            "channel_stands": grp["channel_stands"][...],
            "channel_pols": grp["channel_pols"][...],
            "metadata": metadata,
        }


def iter_rdc2_hdf5_channel_chunks(
    path,
    channel_chunk_size=8,
    channel_indices=None,
    frame_start=None,
    frame_stop=None,
    delay_selection=None,
    doppler_selection=None,
):
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required to load RDC HDF5 files") from exc
    path = os.path.expanduser(path)
    channel_chunk_size = int(channel_chunk_size)
    if channel_chunk_size < 1:
        raise ValueError("channel_chunk_size must be >= 1")

    with _open_hdf5_read(path, h5py) as h5:
        grp = _rdc2_group(h5, path)
        real = grp["real"]
        imag = grp["imag"]
        n_frames = real.shape[0]
        n_channels = real.shape[3]
        frame_start = 0 if frame_start is None else int(frame_start)
        frame_stop = n_frames if frame_stop is None else int(frame_stop)
        if frame_start < 0 or frame_stop > n_frames or frame_stop <= frame_start:
            raise ValueError("frame_start/frame_stop select an invalid RDC2 frame range")
        delay_selection = slice(None) if delay_selection is None else delay_selection
        doppler_selection = slice(None) if doppler_selection is None else doppler_selection
        if channel_indices is None:
            selected = np.arange(n_channels, dtype=int)
        else:
            selected = np.asarray(channel_indices, dtype=int)
            if selected.ndim != 1:
                raise ValueError("channel_indices must be a 1D sequence")
            if selected.size and (np.min(selected) < 0 or np.max(selected) >= n_channels):
                raise ValueError("channel_indices contains an out-of-range channel")
        for start in range(0, selected.size, channel_chunk_size):
            stop = min(start + channel_chunk_size, selected.size)
            idx = selected[start:stop]
            chunk = (
                real[frame_start:frame_stop, delay_selection, doppler_selection, idx]
                + 1j * imag[frame_start:frame_stop, delay_selection, doppler_selection, idx]
            )
            yield start, stop, chunk.astype(np.complex64, copy=False)


def metadata_json(**kwargs):
    return json.dumps(kwargs, indent=2, sort_keys=True)


def export_tbn_rdc0_hdf5(
    idf,
    output_file,
    tstart,
    duration,
    sweep_frequency,
    stands,
    pol,
    start_offset_samples=0,
    adc_start=0,
    adc_count=None,
    gap_fill="nan",
    dtype="float32",
    compression="gzip",
    compression_level=4,
    chunk_pulses=1,
):
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required for RDC HDF5 export") from exc

    fs = float(idf.get_info("sample_rate"))
    try:
        tuning_frequency_hz = float(idf.get_info("freq1"))
    except Exception:
        tuning_frequency_hz = None
    sweep_length = int(round(fs / float(sweep_frequency)))
    if adc_count is None:
        adc_count = sweep_length - int(adc_start)
    total_pulses = int(np.floor((duration * fs - start_offset_samples) / sweep_length))
    if total_pulses < 1:
        raise ValueError("Requested duration does not contain one complete sweep after alignment")

    stream_indices, channel_stands, channel_pols = channel_selection(stands, pol)
    n_channels = len(stream_indices)
    out_dtype = np.dtype(dtype)
    chunks = (int(adc_count), int(chunk_pulses), min(n_channels, 32))
    compression_arg = None if compression == "none" else compression
    opts = None if compression in ("none", "lzf") else int(compression_level)

    output_file = os.path.expanduser(output_file)
    os.makedirs(os.path.dirname(os.path.abspath(output_file)) or ".", exist_ok=True)
    idf.offset(float(tstart))
    chunk_seconds = min(0.25, float(duration))
    target_seconds = float(duration)
    timestamp_gap_tolerance = max(2.0 / fs, 0.001)
    timetag_skip_seconds = 0.002

    buffer = np.empty((n_channels, 0), dtype=np.complex64)
    got = 0.0
    timeline_elapsed = 0.0
    first_timestamp = None
    first_requested_timestamp = None
    pulse_idx = 0
    offset_pending = int(start_offset_samples)
    gap_info = {"mode": gap_fill, "events": 0, "samples": 0, "sample_rate": fs}

    with h5py.File(output_file, "w") as h5:
        grp = h5.create_group("rdc0")
        dset_real = grp.create_dataset(
            "real",
            shape=(int(adc_count), total_pulses, n_channels),
            maxshape=(int(adc_count), total_pulses, n_channels),
            dtype=out_dtype,
            chunks=chunks,
            compression=compression_arg,
            compression_opts=opts,
        )
        dset_imag = grp.create_dataset(
            "imag",
            shape=(int(adc_count), total_pulses, n_channels),
            maxshape=(int(adc_count), total_pulses, n_channels),
            dtype=out_dtype,
            chunks=chunks,
            compression=compression_arg,
            compression_opts=opts,
        )

        while got < target_seconds and pulse_idx < total_pulses:
            try:
                readT, start_timestamp, data = idf.read(min(chunk_seconds, target_seconds - got))
                if first_timestamp is None:
                    if got > 0:
                        first_timestamp = _timestamp_at(start_timestamp, -got)
                        first_requested_timestamp = first_timestamp
                    else:
                        first_timestamp = start_timestamp
                        first_requested_timestamp = start_timestamp
                else:
                    observed_elapsed = _timestamp_delta_seconds(start_timestamp, first_timestamp)
                    if observed_elapsed is not None:
                        timestamp_gap = observed_elapsed - timeline_elapsed
                        if timestamp_gap > timestamp_gap_tolerance:
                            gap_samples = int(round(min(timestamp_gap, target_seconds - got) * fs))
                            if gap_samples > 0:
                                _gap_warning("Timestamp gap; inserting RDC0 fill", tstart + got, _timestamp_at(first_requested_timestamp, got))
                                buffer = np.concatenate([buffer, _gap_fill_block(n_channels, gap_samples, gap_fill)], axis=-1)
                                gap_info["events"] += 1
                                gap_info["samples"] += gap_samples
                                gap_seconds = gap_samples / fs
                                got += gap_seconds
                                timeline_elapsed += gap_seconds
            except RuntimeError as exc:
                if "Invalid timetag skip" not in str(exc):
                    raise
                gap_seconds = min(timetag_skip_seconds, target_seconds - got)
                gap_samples = int(round(gap_seconds * fs))
                _gap_warning("Invalid timetag skip; inserting RDC0 fill", tstart + got, _timestamp_at(first_requested_timestamp, got))
                buffer = np.concatenate([buffer, _gap_fill_block(n_channels, gap_samples, gap_fill)], axis=-1)
                gap_info["events"] += 1
                gap_info["samples"] += gap_samples
                got += gap_seconds
                timeline_elapsed += gap_seconds
                idf.offset(gap_seconds)
                continue

            if data.ndim != 2:
                raise RuntimeError(f"Unexpected data shape from idf.read(): {data.shape}")
            if np.max(stream_indices) >= data.shape[0]:
                raise ValueError(f"Requested stream index {np.max(stream_indices)} but read data has {data.shape[0]} antpols")
            keep_samples = min(data.shape[1], int(round((target_seconds - got) * fs)))
            if keep_samples <= 0:
                break
            buffer = np.concatenate([buffer, data[stream_indices, :keep_samples].astype(np.complex64, copy=False)], axis=-1)
            del data

            read_seconds = keep_samples / fs
            got += read_seconds
            timeline_elapsed += read_seconds

            if offset_pending > 0 and buffer.shape[1] > 0:
                drop = min(offset_pending, buffer.shape[1])
                buffer = buffer[:, drop:]
                offset_pending -= drop

            while buffer.shape[1] >= sweep_length and pulse_idx < total_pulses:
                sweep = buffer[:, :sweep_length]
                adc = sweep[:, adc_start:adc_start + adc_count].T
                dset_real[:, pulse_idx, :] = adc.real.astype(out_dtype, copy=False)
                dset_imag[:, pulse_idx, :] = adc.imag.astype(out_dtype, copy=False)
                buffer = buffer[:, sweep_length:]
                pulse_idx += 1
                if pulse_idx % 10 == 0:
                    print(f"Wrote RDC0 pulse {pulse_idx}/{total_pulses}", flush=True)

        if pulse_idx < total_pulses:
            dset_real.resize((int(adc_count), pulse_idx, n_channels))
            dset_imag.resize((int(adc_count), pulse_idx, n_channels))

        grp.create_dataset("channel_stands", data=channel_stands)
        grp.create_dataset("channel_pols", data=channel_pols)
        grp.create_dataset("adc_sample_index", data=np.arange(adc_start, adc_start + adc_count, dtype=np.int32))
        grp.create_dataset("pulse_index", data=np.arange(pulse_idx, dtype=np.int32))
        h5.attrs["metadata_json"] = metadata_json(
            shape_adc_pulses_channels=[int(adc_count), int(pulse_idx), int(n_channels)],
            tstart=float(tstart),
            duration_requested_s=float(duration),
            sample_rate_hz=fs,
            tuning_frequency_hz=tuning_frequency_hz,
            sweep_frequency_hz=float(sweep_frequency),
            sweep_length_samples=int(sweep_length),
            start_offset_samples=int(start_offset_samples),
            adc_start=int(adc_start),
            adc_count=int(adc_count),
            stands=[int(s) for s in stands],
            pol=str(pol),
            n_channels=int(n_channels),
            n_stands=int(len(np.unique(channel_stands))),
            **polarization_metadata(channel_pols),
            gap_info=gap_info,
        )

    return {
        "output_file": output_file,
        "shape": (int(adc_count), int(pulse_idx), int(n_channels)),
        "bytes_uncompressed": estimate_rdc0_bytes(adc_count, pulse_idx, n_channels, out_dtype),
        "gap_info": gap_info,
    }

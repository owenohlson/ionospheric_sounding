"""Reading WAV recordings: mono = real samples, stereo = I (track 1) and Q (track 2).

A WAV file is one receiver of unknown polarization. It has no reliable UTC
clock unless the user supplies the time of sample zero.
"""

from datetime import datetime, timezone

import numpy as np


def parse_utc(text):
    """ISO 8601 -> aware UTC datetime; a time without zone is taken as UTC."""
    if not text:
        return None
    dt = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def wav_info(path):
    import soundfile as sf
    info = sf.info(path)
    if info.channels not in (1, 2):
        raise ValueError("WAV must be mono real samples or stereo I/Q (I first, Q second)")
    return dict(sample_rate_hz=float(info.samplerate), n_samples=int(info.frames),
                duration_s=info.frames / float(info.samplerate), layout="iq" if info.channels == 2 else "real")


def read_wav_blocks(path, start_sample, n_samples, block=1 << 18):
    """Yield complex64 blocks of shape (1, n) covering [start_sample, start_sample + n_samples)."""
    import soundfile as sf
    with sf.SoundFile(path) as wav:
        wav.seek(int(start_sample))
        remaining = int(n_samples)
        while remaining > 0:
            data = wav.read(min(block, remaining), dtype="float32", always_2d=True)
            if not len(data):
                return
            iq = data[:, 0].astype(np.complex64)
            if data.shape[1] == 2:
                iq.imag = data[:, 1]
            remaining -= len(iq)
            yield iq[None, :], 0

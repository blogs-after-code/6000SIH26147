"""
File ingest: .wav and raw .IQ -> one common representation (complex64 array + fs).

WHY: every later stage (spectrum, demod, FEC) must not care where the samples
came from. This module is the only place that knows about file formats.

Important facts to defend:
  * A raw IQ file has NO header -> sampling rate cannot be read from it.
    The operator must supply it (or we later estimate only the ratio
    fs/symbol_rate = samples per symbol, never the absolute fs).
  * A WAV header DOES carry fs.
  * Mono WAV = real signal -> we build the analytic (complex) signal with the
    Hilbert transform so the same pipeline works.
"""
import io
import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import hilbert

RAW_FORMATS = ("cf32", "ci16", "ci8", "cu8")
EXT_TO_FMT = {".cf32": "cf32", ".cfile": "cf32", ".cs16": "ci16", ".cs8": "ci8", ".cu8": "cu8"}


def load_raw_iq(data: bytes, fmt: str = "cf32") -> np.ndarray:
    if fmt not in RAW_FORMATS:
        raise ValueError(f"fmt must be one of {RAW_FORMATS}")
    dt = {"cf32": np.float32, "ci16": np.int16, "ci8": np.int8, "cu8": np.uint8}[fmt]
    size = np.dtype(dt).itemsize
    data = data[: len(data) // size * size]
    a = np.frombuffer(data, dtype=dt).astype(np.float32)
    if fmt == "ci16":
        a /= 32768.0
    elif fmt == "ci8":
        a /= 128.0
    elif fmt == "cu8":            # RTL-SDR style: unsigned, centred at 127.5
        a = (a - 127.5) / 127.5
    if a.size % 2:
        a = a[:-1]
    return (a[0::2] + 1j * a[1::2]).astype(np.complex64)


def load_wav(data: bytes):
    fs, arr = wavfile.read(io.BytesIO(data))
    notes = []
    if arr.dtype == np.int16:
        a = arr.astype(np.float32) / 32768.0
    elif arr.dtype == np.int32:
        a = arr.astype(np.float32) / 2147483648.0
    elif arr.dtype == np.uint8:
        a = (arr.astype(np.float32) - 128.0) / 128.0
    else:
        a = arr.astype(np.float32)
    if a.ndim == 2 and a.shape[1] >= 2:
        x = (a[:, 0] + 1j * a[:, 1]).astype(np.complex64)
        notes.append("WAV treated as stereo I/Q (left=I, right=Q)")
    else:
        a = a.reshape(-1)
        x = hilbert(a).astype(np.complex64)
        notes.append("Mono WAV treated as a real signal; analytic signal built with Hilbert transform")
    return x, float(fs), notes


def load_file(filename: str, data: bytes, fmt: str = "cf32", fs=None):
    """Return (x, fs, info dict)."""
    ext = os.path.splitext(filename.lower())[1]
    if ext == ".wav":
        x, wav_fs, notes = load_wav(data)
        if fs:
            notes.append(f"User fs {fs} Hz overrides WAV header fs {wav_fs} Hz")
        return x, float(fs or wav_fs), dict(kind="wav", fmt="wav", notes=notes)
    fmt = EXT_TO_FMT.get(ext, fmt)
    if not fs:
        raise ValueError("Sampling rate (fs) is required for raw IQ files - they have no header")
    x = load_raw_iq(data, fmt)
    return x, float(fs), dict(kind="raw", fmt=fmt, notes=[f"Raw IQ parsed as {fmt}"])

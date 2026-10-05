"""
Synthetic signal generator (ground truth source).

WHY: real recordings have no answer key. A generator lets us create signals whose
parameters we KNOW, so every estimator we write can be scored against the truth
(accuracy vs SNR graphs for the PPT / finals).

HOW (linear modulations BPSK/QPSK/8PSK/16QAM):
  bits -> symbols (constellation) -> upsample by sps -> RRC pulse shaping
  -> carrier frequency offset -> AWGN noise.
HOW (FSK): each symbol picks a frequency; phase = integral of frequency
  (phase-continuous, so the spectrum stays compact).
"""
import numpy as np
from scipy.io import wavfile

MOD_ORDER = {"BPSK": 2, "QPSK": 4, "8PSK": 8, "16QAM": 16, "2FSK": 2, "4FSK": 4}


def rrc_taps(beta: float, sps: int, span: int = 10) -> np.ndarray:
    """Root-raised-cosine filter. Two of them (TX + RX) make a raised cosine,
    which has zero inter-symbol interference at the symbol instants."""
    half = span * sps // 2
    t = np.arange(-half, half + 1) / sps
    taps = np.zeros_like(t)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-12:
            taps[i] = 1.0 - beta + 4.0 * beta / np.pi
        elif beta > 0 and abs(abs(ti) - 1.0 / (4.0 * beta)) < 1e-9:
            taps[i] = (beta / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
            )
        else:
            num = np.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))
            den = np.pi * ti * (1 - (4 * beta * ti) ** 2)
            taps[i] = num / den
    return taps / np.sqrt(np.sum(taps ** 2))


def constellation(mod: str) -> np.ndarray:
    """Symbol table. Index = integer value of the bit group (natural binary)."""
    if mod == "BPSK":
        return np.array([1, -1], dtype=complex)
    if mod == "QPSK":
        return np.exp(1j * (2 * np.pi * np.arange(4) / 4 + np.pi / 4))
    if mod == "8PSK":
        return np.exp(1j * 2 * np.pi * np.arange(8) / 8)
    if mod == "16QAM":
        lv = np.array([-3, -1, 1, 3])
        pts = np.array([a + 1j * b for a in lv for b in lv])
        return pts / np.sqrt(10)
    raise ValueError(f"no constellation for {mod}")


def _gray_encode(i):
    return i ^ (i >> 1)


def _gray_decode(v):
    i = v
    s = v >> 1
    while s:
        i ^= s
        s >>= 1
    return i


def label_to_index(mod, v, gray):
    """Bit-group value -> constellation point index. Natural binary or Gray labelling."""
    v = np.asarray(v)
    if not gray:
        return v
    f = np.vectorize(_gray_decode)
    if mod == "16QAM":
        return 4 * f(v >> 2) + f(v & 3)
    return f(v)


def index_to_label(mod, idx, gray):
    idx = np.asarray(idx)
    if not gray:
        return idx
    if mod == "16QAM":
        return (_gray_encode(idx >> 2) << 2) | _gray_encode(idx & 3)
    return _gray_encode(idx)


def gen_signal(mod="QPSK", fs=100e3, symbol_rate=12.5e3, n_symbols=4000,
               snr_db=15.0, cfo_hz=0.0, beta=0.35, seed=None, bits=None, fsk_h=1.0, gray=False):
    """Return (complex64 IQ, truth dict).

    snr_db is FULL-BAND SNR: signal power / total noise power across the whole
    sampling bandwidth fs. (In-band SNR is higher by about fs/bandwidth.)
    """
    mod = mod.upper()
    if mod not in MOD_ORDER:
        raise ValueError(f"unsupported modulation {mod}")
    rng = np.random.default_rng(seed)
    M = MOD_ORDER[mod]
    k = int(np.log2(M))
    sps_f = fs / symbol_rate
    if abs(sps_f - round(sps_f)) > 1e-9:
        raise ValueError("fs / symbol_rate must be an integer (samples per symbol)")
    sps = int(round(sps_f))

    if bits is None:
        bits = rng.integers(0, 2, n_symbols * k, dtype=np.uint8)
    else:
        bits = np.asarray(bits, dtype=np.uint8)
        bits = bits[: len(bits) // k * k]
    idx = bits.reshape(-1, k).astype(int) @ (1 << np.arange(k - 1, -1, -1))
    idx = label_to_index(mod, idx, gray)
    n_sym = len(idx)

    if mod.endswith("FSK"):
        freqs = (idx - (M - 1) / 2.0) * fsk_h * symbol_rate
        phase = 2 * np.pi * np.cumsum(np.repeat(freqs, sps)) / fs
        x = np.exp(1j * phase)
    else:
        sym = constellation(mod)[idx]
        up = np.zeros(n_sym * sps, dtype=complex)
        up[::sps] = sym
        taps = rrc_taps(beta, sps)
        delay = (len(taps) - 1) // 2
        x = np.convolve(up, taps)[delay: delay + n_sym * sps]

    x = x / np.sqrt(np.mean(np.abs(x) ** 2))          # unit power
    t = np.arange(len(x)) / fs
    x = x * np.exp(1j * 2 * np.pi * cfo_hz * t)        # carrier offset
    sigma2 = 10 ** (-snr_db / 10.0)                    # noise power (signal power = 1)
    noise = np.sqrt(sigma2 / 2) * (rng.standard_normal(len(x)) + 1j * rng.standard_normal(len(x)))
    x = (x + noise).astype(np.complex64)

    truth = dict(mod=mod, fs=fs, symbol_rate=symbol_rate, sps=sps, beta=beta,
                 snr_db_fullband=snr_db, cfo_hz=cfo_hz, n_symbols=int(n_sym), gray=bool(gray),
                 bits=bits)
    return x, truth


def save_cf32(path, x):
    """Raw interleaved float32 I,Q,I,Q... (GNU Radio 'file sink' format)."""
    np.stack([x.real, x.imag], axis=1).astype(np.float32).tofile(path)


def save_wav_iq(path, x, fs):
    """Stereo 16-bit WAV: left = I, right = Q."""
    peak = max(np.abs(x.real).max(), np.abs(x.imag).max())
    data = (np.stack([x.real, x.imag], axis=1) * (0.9 / peak) * 32767).astype(np.int16)
    wavfile.write(path, int(fs), data)

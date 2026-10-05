"""
Phase 2 core: channelise -> symbol rate -> timing -> carrier -> classify -> demodulate.

Design choice (defend this): every estimator is FEED-FORWARD (block based), not a
feedback loop. No loop gains to tune, no cycle slips, works on short captures.
  * symbol rate  : spectral line of |x|^2 (cyclostationarity of linear modulations)
  * timing       : Oerder-Meyr - the same |x|^2 line, but its PHASE gives the sampling instant
  * carrier      : M-th power line gives frequency offset, windowed M-th power gives phase
"""
from fractions import Fraction
import numpy as np
from scipy.signal import firwin, fftconvolve, resample_poly
from scipy.interpolate import CubicSpline
from scipy.cluster.vq import kmeans2

from .generator import rrc_taps, constellation, MOD_ORDER, index_to_label

SPS = 8                      # internal samples per symbol after resampling
LINEAR = ("BPSK", "QPSK", "8PSK", "16QAM")


# ---------------------------------------------------------------- channel
def channelize(x, fs, fc, bw):
    """Shift the detected signal to 0 Hz and low-pass it to the occupied band."""
    n = np.arange(len(x))
    xb = x * np.exp(-2j * np.pi * fc * n / fs).astype(np.complex64)
    cutoff = float(min(0.45 * fs, 0.75 * bw))
    ntaps = int(np.clip(6 * fs / cutoff, 63, 1023)) | 1
    taps = firwin(ntaps, cutoff, fs=fs)
    return fftconvolve(xb, taps, mode="same").astype(np.complex64)


def envelope_cv(xb):
    a = np.abs(xb)
    return float(a.std() / (a.mean() + 1e-12))


# ---------------------------------------------------------------- symbol rate
def _parabolic(mag, i):
    if i <= 0 or i >= len(mag) - 1:
        return float(i)
    a, b, c = np.log(mag[i - 1] + 1e-30), np.log(mag[i] + 1e-30), np.log(mag[i + 1] + 1e-30)
    d = a - 2 * b + c
    return float(i) if d == 0 else i + 0.5 * (a - c) / d


def _line_search(sig, fs, lo, hi):
    """Strongest spectral line of `sig` in [lo, hi] Hz. Returns (freq, peak-to-median ratio)."""
    sig = sig - sig.mean()
    nfft = 1 << int(np.ceil(np.log2(max(len(sig) * 4, 1024))))
    nfft = min(nfft, 1 << 20)
    S = np.abs(np.fft.rfft(sig, nfft))
    f = np.fft.rfftfreq(nfft, 1 / fs)
    m = (f >= lo) & (f <= hi)
    idx = np.where(m)[0]
    k = idx[np.argmax(S[idx])]
    pmr = float(S[k] / (np.median(S[idx]) + 1e-30))
    kf = _parabolic(S, k)
    return float(kf * fs / nfft), pmr, S, f


def estimate_symbol_rate_linear(xb, fs, bw):
    e = np.abs(xb) ** 2
    fr, pmr, S, f = _line_search(e, fs, 0.35 * bw, 1.1 * bw)
    spec = (f, S)
    # harmonic guard: if a line also exists at fr/2 it is the true fundamental
    half = fr / 2
    if half > 0.2 * bw:
        w = (f > half * 0.97) & (f < half * 1.03)
        pk = S[(f > fr * 0.97) & (f < fr * 1.03)].max()
        if w.any() and S[w].max() > 0.4 * pk:
            fr = half
    return fr, pmr, spec


def estimate_symbol_rate_fsk(xb, fs, bw):
    d = np.angle(xb[1:] * np.conj(xb[:-1]))            # rad/sample, noisy
    k = max(3, int(round(fs / max(bw, 1) * 0.6)))
    u = np.convolve(d, np.ones(k) / k, mode="same")
    g = np.abs(u[2:] - u[:-2])                          # large where the tone changes
    fr, pmr, S, f = _line_search(g, fs, 0.15 * bw, 1.0 * bw)
    return fr, pmr, (f, S)


# ---------------------------------------------------------------- timing (linear)
def resample_to_sps(xb, fs, Rs, sps=SPS):
    frac = Fraction(Rs * sps / fs).limit_denominator(2000)
    y = resample_poly(xb, frac.numerator, frac.denominator)
    fs2 = fs * frac.numerator / frac.denominator
    return y.astype(np.complex64), fs2


def recover_symbols(xb, fs, Rs, beta=0.35, sps=SPS, block_syms=64):
    """Matched filter + Oerder-Meyr feed-forward timing. Returns (symbols, info)."""
    y, fs2 = resample_to_sps(xb, fs, Rs, sps)
    y = fftconvolve(y, rrc_taps(beta, sps), mode="same")
    B = block_syms * sps
    nb = len(y) // B
    if nb < 2:
        raise ValueError("Signal too short for timing recovery")
    e = np.abs(y[: nb * B]) ** 2
    e = e.reshape(nb, B)
    e = e - e.mean(axis=1, keepdims=True)
    c = (e * np.exp(-2j * np.pi * np.arange(B) / sps)).sum(axis=1)
    phi = np.unwrap(np.angle(c))
    w = np.abs(c)
    A = np.stack([np.ones(nb), np.arange(nb)], axis=1) * w[:, None]
    a, s = np.linalg.lstsq(A, phi * w, rcond=None)[0]
    sps_eff = sps / (1.0 + s * sps / (2 * np.pi * B))
    p0 = (-a * sps / (2 * np.pi)) % sps + 3 * sps
    n_sym = int((len(y) - 4 - p0) / sps_eff)
    pos = p0 + np.arange(n_sym) * sps_eff
    cs = CubicSpline(np.arange(len(y)), y)
    syms = cs(pos).astype(np.complex64)
    lock = float(np.abs(c.sum()) / (w.sum() + 1e-30))
    Rs_ref = fs2 / sps_eff
    # eye diagram: 2-symbol-wide traces of the in-phase component around sampling instants
    ntr = min(150, max(n_sym - 4, 1))
    sel = pos[2:2 + ntr]
    offs = np.arange(-sps, sps + 1)
    eye = cs(sel[:, None] + offs[None, :]).real
    eye = eye / (np.abs(eye).max() + 1e-12)
    return syms, dict(symbol_rate_refined=float(Rs_ref), timing_lock=lock,
                      sps_eff=float(sps_eff), eye=eye.astype(np.float32))


# ---------------------------------------------------------------- carrier
def mpower_line(y, M):
    """Spectral line of y^M. Returns (cycles/symbol of y^M, peak-to-median ratio)."""
    z = y ** M
    z = z - 0 * z.mean()
    nfft = 1 << int(np.ceil(np.log2(len(z) * 8)))
    Z = np.abs(np.fft.fft(z, nfft))
    k = int(np.argmax(Z))
    kk = _parabolic(np.concatenate([Z[-1:], Z, Z[:1]]), k + 1) - 1
    f = kk / nfft
    if f > 0.5:
        f -= 1.0
    return float(f), float(Z[k] / (np.median(Z) + 1e-30))


def line_features(y):
    out = {}
    for M in (2, 4, 8):
        f, pmr = mpower_line(y, M)
        out[M] = (f, pmr)
    return out


def ref_phase(mod, M):
    return float(np.angle(np.mean(constellation(mod) ** M)))


PSK_M = {"BPSK": 2, "QPSK": 4, "8PSK": 8, "16QAM": 4}


def carrier_recover(y, mod, win=None):
    """Frequency offset from M-th power line, then windowed M-th power for phase.
    Returns corrected symbols and info. Result is correct up to a 2*pi/M rotation."""
    M = PSK_M[mod]
    if win is None:                      # noisier M-th power (QAM, 8PSK) needs a longer window
        win = 129 if mod in ("16QAM", "8PSK") else 65
    f, pmr = mpower_line(y, M)
    cfo = f / M                                          # cycles per symbol
    k = np.arange(len(y))
    y1 = y * np.exp(-2j * np.pi * cfo * k)
    z = y1 ** M
    kern = np.ones(win) / win
    zs = np.convolve(z, kern, mode="same")
    th = (np.unwrap(np.angle(zs)) - ref_phase(mod, M)) / M
    y2 = y1 * np.exp(-1j * th)
    # decision-directed refinement (tightens phase for 8PSK / 16QAM where M-th power is noisy)
    for _ in range(2):
        yn = y2 / np.sqrt(np.mean(np.abs(y2) ** 2))
        C = constellation(mod)
        dec = C[np.abs(yn[:, None] - C[None, :]).argmin(axis=1)]
        e = np.angle(yn * np.conj(dec))
        es = np.convolve(np.exp(1j * e), np.ones(win) / win, mode="same")
        y2 = y2 * np.exp(-1j * np.angle(es))
    return y2, dict(cfo_cycles_per_symbol=float(cfo), mth_power_pmr=float(pmr), M=M)


def agc(y):
    return y / np.sqrt(np.mean(np.abs(y) ** 2) + 1e-30)


def demap_linear(y, mod, gray=False):
    C = constellation(mod)
    d = np.abs(y[:, None] - C[None, :]) ** 2
    idx = d.argmin(axis=1)
    k = int(np.log2(len(C)))
    lab = index_to_label(mod, idx, gray)
    bits = ((lab[:, None] >> np.arange(k - 1, -1, -1)) & 1).astype(np.uint8).reshape(-1)
    evm = float(np.sqrt(np.mean(d[np.arange(len(y)), idx])))
    return bits, evm


# ---------------------------------------------------------------- FSK
def demod_fsk(xb, fs, Rs):
    """Frequency discriminator -> per-symbol mean frequency -> cluster into tones."""
    d = np.angle(xb[1:] * np.conj(xb[:-1])) * fs / (2 * np.pi)       # Hz
    sps = fs / Rs
    k = max(3, int(round(sps * 0.5)))
    u = np.convolve(d, np.ones(k) / k, mode="same")
    # timing: transitions of u have line at Rs; its phase gives boundary instants
    # refine using explicit scan of candidate offsets: choose offset maximising tone separability
    best = None
    for off in np.linspace(0, sps, int(min(sps, 32)), endpoint=False):
        pos = off + sps / 2 + np.arange(int((len(u) - off) / sps) - 1) * sps
        pos = pos[(pos > 1) & (pos < len(u) - 2)]
        s = u[pos.astype(int)]
        score = np.mean(np.abs(s - np.median(s)))        # largest when sampling mid-symbol
        if best is None or score > best[0]:
            best = (score, pos)
    pos = best[1]
    s = u[pos.astype(int)]
    return s, dict()


def fsk_levels(s):
    """Decide 2 vs 4 tones and return (idx array, centres, M)."""
    best = None
    res = {}
    for K in (2, 4):
        c0 = np.quantile(s, (np.arange(K) + 0.5) / K)
        cent, lab = kmeans2(s.reshape(-1, 1), c0.reshape(-1, 1), minit="matrix", iter=30)
        order = np.argsort(cent[:, 0])
        res[K] = (cent[order, 0], np.argsort(order)[lab])
    c4 = res[4][0]
    gaps = np.diff(c4)
    ratio = gaps.min() / (gaps.max() + 1e-12)
    M = 4 if ratio > 0.6 else 2
    cent, idx = res[M]
    return idx.astype(int), cent, M

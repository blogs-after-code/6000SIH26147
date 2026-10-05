"""
Spectral analysis: PSD, waterfall, centre frequency, bandwidth, SNR.

WHY: before demodulating anything we must know WHERE the signal is, HOW WIDE it
is and HOW NOISY it is. Everything in Phase 2 (filtering, decimation, carrier
recovery) is configured from these numbers.

HOW:
  PSD   = Welch's method: split into overlapping windowed segments, FFT each,
          average the |FFT|^2. Averaging cuts the random variance of one FFT.
  Noise = iterative estimate of the flat floor: start from a low percentile,
          keep only bins below 2x floor (i.e. bins that look like noise), take
          their mean, repeat. Signal bins stop contaminating the estimate.
  Signal power = sum of (PSD - floor) over bins clearly above the floor.
  Centre = power-weighted mean frequency of that excess power.
  Bandwidth = width containing 99% of the excess power (99% occupied BW).
  SNR   = signal power / noise power (reported both full-band and in-band).

Known limit (say it in the defence): if the signal fills almost the whole
sampling band there are no "empty" bins to measure the noise floor from.
"""
import numpy as np
from scipy.signal import welch, spectrogram


def _pow2_floor(n: int, cap: int = 1024) -> int:
    p = 64
    while p * 2 <= min(n, cap):
        p *= 2
    return p


def estimate_noise_floor(psd: np.ndarray) -> float:
    noise = np.percentile(psd, 20)
    for _ in range(4):
        sel = psd[psd < 2.0 * noise]
        if sel.size == 0:
            break
        noise = sel.mean()
    return float(noise)


def analyze(x: np.ndarray, fs: float, max_samples: int = 2_000_000) -> dict:
    x = np.asarray(x[:max_samples], dtype=np.complex64)
    n = len(x)
    if n < 256:
        raise ValueError("Need at least 256 samples")
    nfft = _pow2_floor(n)

    f, psd = welch(x, fs=fs, window="hann", nperseg=nfft, noverlap=nfft // 2,
                   return_onesided=False, detrend=False, scaling="density")
    f = np.fft.fftshift(f)
    psd = np.fft.fftshift(psd)
    df = fs / nfft

    noise = estimate_noise_floor(psd)
    sig_mask = psd >= 2.0 * noise
    excess = np.where(sig_mask, psd - noise, 0.0)
    total_excess = excess.sum()

    if total_excess <= 0:
        params = dict(center_freq_hz=None, bandwidth_hz=None, snr_db_fullband=None,
                      snr_db_inband=None, note="No signal found above the noise floor")
    else:
        center = float((f * excess).sum() / total_excess)
        cum = np.cumsum(excess) / total_excess
        f_lo = float(f[np.searchsorted(cum, 0.005)])
        f_hi = float(f[min(np.searchsorted(cum, 0.995), len(f) - 1)])
        bw = max(f_hi - f_lo, df)
        p_sig = total_excess * df
        snr_full = p_sig / (noise * fs)
        snr_in = p_sig / (noise * bw)
        params = dict(center_freq_hz=center, bandwidth_hz=float(bw),
                      band_low_hz=f_lo, band_high_hz=f_hi,
                      snr_db_fullband=float(10 * np.log10(snr_full)),
                      snr_db_inband=float(10 * np.log10(snr_in)))
    params["noise_floor_db"] = float(10 * np.log10(noise + 1e-30))
    params["power_dbfs"] = float(10 * np.log10(np.mean(np.abs(x) ** 2) + 1e-30))
    params["dc_offset"] = float(abs(np.mean(x)))

    # ---- waterfall (time x frequency), downsampled for the browser ----
    hop = max(nfft // 2, n // 200)
    _, _, S = spectrogram(x, fs=fs, window="hann", nperseg=nfft, noverlap=nfft - hop,
                          return_onesided=False, detrend=False, mode="psd")
    S = np.fft.fftshift(S, axes=0)                  # freq x time
    target_f = 256
    fac = max(nfft // target_f, 1)
    S = S[: (S.shape[0] // fac) * fac].reshape(-1, fac, S.shape[1]).mean(axis=1)
    wf_db = 10 * np.log10(S.T + 1e-20)              # time x freq
    f_ds = f[: (len(f) // fac) * fac].reshape(-1, fac).mean(axis=1)

    return dict(
        n_samples=int(n), fs=float(fs), duration_s=float(n / fs), nfft=int(nfft),
        params={k: (round(v, 3) if isinstance(v, float) else v) for k, v in params.items()},
        psd=dict(freq=np.round(f, 2).tolist(), db=np.round(10 * np.log10(psd + 1e-20), 2).tolist()),
        waterfall=dict(freq=np.round(f_ds, 2).tolist(),
                       rows=int(wf_db.shape[0]), cols=int(wf_db.shape[1]),
                       db=np.round(wf_db, 1).tolist(),
                       t_end=float(n / fs)),
    )

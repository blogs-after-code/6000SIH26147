"""
Bit-stream correlation: find frame length, sync word, and the header / payload split.

HOW (all blind):
 1. Frame length  : autocorrelation of the +/-1 bit stream. Frames repeat the same
    sync word, so bits one frame apart agree more than chance -> a peak at the frame length.
    (ASCII text also correlates at multiples of 8, so each lag is judged against the
    other lags with the same residue mod 8.)
 2. Sync word     : fold the stream into rows of one frame length. Columns whose bit is
    (almost) always the same across frames form a long constant run = the sync word.
 3. Header        : right after the sync word, columns that are PREDICTABLE from the same
    column in the previous two frames (constants, counters) are header. Random-looking
    columns are payload. The boundary is rounded down to a byte.
"""
import numpy as np

KNOWN_ASM = {0x1ACFFC1D: "CCSDS ASM", 0xEB90: "AOS/TM sync 0xEB90", 0x7E: "HDLC flag 0x7E"}


def _bits_to_hex(b):
    b = np.asarray(b, np.uint8)
    pad = (-len(b)) % 4
    b = np.concatenate([b, np.zeros(pad, np.uint8)])
    return "".join(format(int("".join(map(str, b[i:i + 4])), 2), "x") for i in range(0, len(b), 4)).upper()


def bits_to_bytes(b):
    b = np.asarray(b, np.uint8)
    m = len(b) // 8 * 8
    return np.packbits(b[:m]).tobytes()


def fold_period(bits, min_lag=24, min_frames=8):
    """Frame length by FOLDING: for each candidate length L, stack the stream into rows of L
    bits and measure how many columns hold the same bit in (almost) every row.
    T(L) = sum_j (F*p_j^2 - 1)/(F-1)   (p_j = column mean of the +/-1 bits; ~0 for random data)
    Frames repeat their sync word, so T jumps at the true frame length. More sensitive than
    plain autocorrelation because every pair of frames contributes, not just neighbours.
    ASCII payloads add constant columns (bit 7 = 0) at every multiple of 8, so each L is
    judged against its neighbours of the same residue mod 8 (a running median)."""
    from scipy.ndimage import median_filter
    x = (2.0 * np.asarray(bits, np.float32) - 1.0)
    N = len(x)
    max_lag = N // min_frames
    if max_lag <= min_lag + 40:
        return None, 0.0, np.zeros(1)
    lags = np.arange(min_lag, max_lag)
    T = np.zeros(len(lags))
    for i, L in enumerate(lags):
        F = N // L
        p = x[: F * L].reshape(F, L).mean(axis=0)
        T[i] = np.sum((F * p * p - 1.0) / (F - 1.0))
    res = np.zeros(len(lags))
    for r in range(8):
        sel = np.where(lags % 8 == r)[0]
        if len(sel) < 9:
            continue
        base = median_filter(T[sel], size=min(31, len(sel) | 1), mode="nearest")
        res[sel] = T[sel] - base
    sigma = 1.4826 * np.median(np.abs(res - np.median(res))) + 1e-9
    z = res / sigma
    zmax = float(z.max())
    if zmax < 6:
        return None, zmax, z
    kbest = int(np.argmax(z))
    lag = int(lags[kbest])
    # multiples of the true frame length peak too (T grows with the number of sync copies per row).
    # A divisor d of the strongest lag is the real period if folding at d still shows a long constant run.
    while True:
        shrunk = False
        for m in range(8, 1, -1):
            if lag % m == 0 and lag // m >= min_lag:
                d = lag // m
                Fd = N // d
                if Fd < min_frames:
                    continue
                p = x[: Fd * d].reshape(Fd, d).mean(axis=0)
                _, run = _longest_circular_run(np.abs(p) > max(0.7, 3.0 / np.sqrt(Fd)))
                if run >= 16:
                    lag, shrunk = d, True
                    break
        if not shrunk:
            break
    return lag, zmax, z


def _longest_circular_run(mask):
    m = len(mask)
    dbl = np.concatenate([mask, mask])
    best = cur = 0; bs = 0; cs = 0
    for i in range(2 * m):
        if dbl[i]:
            if cur == 0:
                cs = i
            cur += 1
            if cur > best and cur <= m:
                best, bs = cur, cs
        else:
            cur = 0
    return bs % m, best


def _predictability(X):
    """Per-column accuracy of the best order-2 predictor from the previous two frames."""
    F, L = X.shape
    if F < 6:
        return np.zeros(L)
    ctx = (2 * X[:-2] + X[1:-1]).astype(np.int64)    # (F-2, L) values 0..3
    tgt = X[2:]
    acc = np.zeros(L)
    cols = np.arange(L)[None, :].repeat(F - 2, 0)
    key = (cols * 4 + ctx).reshape(-1)
    n1 = np.bincount(key, weights=tgt.reshape(-1), minlength=L * 4).reshape(L, 4)
    nt = np.bincount(key, minlength=L * 4).reshape(L, 4)
    best = np.maximum(n1, nt - n1).sum(axis=1)
    acc2 = best / (F - 2)
    pers = np.mean(X[1:] == X[:-1], axis=0)             # slowly varying bits (counters) stay put
    return np.where(pers >= 0.7, np.maximum(acc2, 0.95), acc2)


def detect(bits, min_sync=8, tol_bits=4):
    """Full frame analysis. Returns dict (found=False if no periodic structure)."""
    b = np.asarray(bits, np.uint8)
    lag, z, curve = fold_period(b)
    out = dict(found=False, z=round(float(z), 1))
    if lag is None:
        return out
    L = lag
    F = len(b) // L
    if F < 6:
        return out
    X = b[: F * L].reshape(F, L).astype(np.float64)
    p = 2 * X.mean(axis=0) - 1
    const = np.abs(p) > max(0.7, 3.0 / np.sqrt(F))
    start, run = _longest_circular_run(const)
    if run < min_sync or run > L - 2:
        return out
    run_bits = (p[(start + np.arange(run)) % L] > 0).astype(np.uint8)
    # known sync words (and their inversions) inside the constant run
    q, sync_len, known = 0, min(run, 32), None
    for val, name in KNOWN_ASM.items():
        nb = max(8, (val.bit_length() + 7) // 8 * 8)
        pat = np.array([(val >> (nb - 1 - i)) & 1 for i in range(nb)], np.uint8)
        for inv in (0, 1):
            pp = pat ^ inv
            for qq in range(0, run - nb + 1):
                if np.array_equal(run_bits[qq:qq + nb], pp):
                    q, sync_len, known = qq, nb, (val, name)
                    break
            if known:
                break
        if known:
            break
    start = (start + q) % L
    sync_bits = run_bits[q:q + sync_len]
    Xr = np.roll(X, -start, axis=1)
    pred = _predictability(Xr)
    j = sync_len
    while j < L and pred[j] >= 0.9:
        j += 1
    hdr = ((j - sync_len) // 8) * 8
    sync_val = int(_bits_to_hex(sync_bits), 16) if sync_len <= 64 else None
    run = sync_len
    # locate every frame start by matched filtering the sync word over the stream
    xs = 2.0 * b.astype(np.float64) - 1.0
    pat = 2.0 * sync_bits - 1.0
    corr = np.correlate(xs, pat, mode="valid")
    hits = np.where(corr >= run - 2 * tol_bits)[0]
    starts = []
    for h in hits:
        if not starts or h - starts[-1] >= L // 2:
            starts.append(int(h))
    # column curves for the UI (rotated order), downsampled to <= 600 columns
    bias = np.abs(2 * Xr.mean(axis=0) - 1)
    step = max(1, L // 600)
    Lc = (L // step) * step
    out.update(
        found=True, frame_len=int(L), n_frames=int(F), sync_len=int(run), sync_offset=int((start) % L),
        sync_hex=_bits_to_hex(sync_bits), sync_value=sync_val,
        sync_name=known[1] if known else None,
        header_bits=int(hdr), payload_bits=int(L - run - hdr), frame_starts=starts[:4000],
        columns=dict(bias=np.round(bias[:Lc].reshape(-1, step).mean(axis=1), 3).tolist(),
                     pred=np.round(pred[:Lc].reshape(-1, step).mean(axis=1), 3).tolist(), step=int(step)),
        autocorr=np.round(curve[: min(len(curve), 4000)][::max(1, len(curve) // 800)], 2).tolist(),
    )
    return out


def extract_frames(bits, det, max_frames=8):
    """Slice frames out of the stream using detection results. Returns list of dicts."""
    b = np.asarray(bits, np.uint8)
    L, run, hdr = det["frame_len"], det["sync_len"], det["header_bits"]
    frames = []
    for s in det["frame_starts"][:max_frames]:
        if s + L > len(b):
            break
        fr = b[s: s + L]
        h = fr[run: run + hdr]
        pl = fr[run + hdr:]
        pb = bits_to_bytes(pl)
        frames.append(dict(start=int(s), header_hex=_bits_to_hex(h), payload_hex=pb.hex().upper(),
                           payload_ascii="".join(chr(c) if 32 <= c < 127 else "." for c in pb)))
    return frames


def printable_fraction(data: bytes):
    if not data:
        return 0.0
    return sum(1 for c in data if 32 <= c < 127 or c in (10, 13)) / len(data)

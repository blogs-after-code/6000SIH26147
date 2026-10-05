"""
Interleavers and de-interleavers (bit level) plus a blind scanner.

block(R,C)       write rows of an R x C matrix, read columns.
diagonal(R,C)    write rows, read along diagonals (c - r) mod C.
convolutional(B,M)  Forney: B branches, branch j delays its bits by j*M (de-interleaver uses (B-1-j)*M).
pseudo-random(N,seed)  fixed random permutation of N bits. NOT blindly detectable - needs the seed.

Alignment: a captured stream does not start on an interleaver block boundary, so
align_deinterleave() tries every start offset and keeps the one after which structure
(a convolutional code or a repeating frame) appears. The interleaver TYPE and SIZE are
operator-supplied; blind discovery of them is an open research problem (see docs).
"""
import time
import numpy as np


# ------------------------------------------------------------------ permutations
def perm_block(R, C):
    return np.arange(R * C).reshape(R, C).T.reshape(-1)


def perm_diag(R, C):
    r, c = np.divmod(np.arange(R * C), C)
    order = np.lexsort((r, (c - r) % C))               # primary key: diagonal, secondary: row
    return order


def perm_prand(N, seed):
    return np.random.default_rng(seed).permutation(N)


def apply_perm(bits, perm, inverse=False):
    bits = np.asarray(bits)
    N = len(perm)
    nb = len(bits) // N
    if nb == 0:
        return bits[:0]
    blocks = bits[: nb * N].reshape(nb, N)
    if inverse:
        out = np.empty_like(blocks)
        out[:, perm] = blocks
    else:
        out = blocks[:, perm]
    return out.reshape(-1)


# ------------------------------------------------------------------ convolutional (Forney)
def conv_interleave(bits, B, M, deinterleave=False):
    bits = np.asarray(bits)
    out = np.zeros_like(bits)
    for j in range(B):
        d = ((B - 1 - j) if deinterleave else j) * M
        br = bits[j::B]
        sh = np.concatenate([np.zeros(d, bits.dtype), br])[: len(br)]
        out[j::B] = sh
    return out


# ------------------------------------------------------------------ generic front door
def interleave(bits, kind, a=None, b=None, seed=1):
    if kind in (None, "none"):
        return np.asarray(bits)
    if kind == "block":
        return apply_perm(bits, perm_block(a, b))
    if kind == "diagonal":
        return apply_perm(bits, perm_diag(a, b))
    if kind == "convolutional":
        return conv_interleave(bits, a, b)
    if kind == "prandom":
        return apply_perm(bits, perm_prand(a * b, seed))
    raise ValueError(kind)


def deinterleave(bits, kind, a=None, b=None, seed=1):
    if kind in (None, "none"):
        return np.asarray(bits)
    if kind == "block":
        return apply_perm(bits, perm_block(a, b), inverse=True)
    if kind == "diagonal":
        return apply_perm(bits, perm_diag(a, b), inverse=True)
    if kind == "convolutional":
        return conv_interleave(bits, a, b, deinterleave=True)
    if kind == "prandom":
        return apply_perm(bits, perm_prand(a * b, seed), inverse=True)
    raise ValueError(kind)


def n_offsets(kind, a, b):
    """How many start alignments must be tried (the stream does not begin on a block boundary)."""
    return (a if kind == "convolutional" else a * b)


def align_deinterleave(bits, kind, a, b, score_fn, seed=1, threshold=0.95, budget_s=25.0, probe=3000):
    """Find the start offset that makes structure appear after de-interleaving.
    score_fn(bits)->0..1.  Returns (offset, score, deinterleaved_full_stream)."""
    bits = np.asarray(bits, np.uint8)
    t0 = time.time()
    best = (-1, -1.0)
    N = n_offsets(kind, a, b)
    for off in range(N):
        if time.time() - t0 > budget_s:
            break
        seg = bits[off: off + max(probe, 3 * a * b) + (a * a * b if kind == "convolutional" else 0)]
        d = deinterleave(seg, kind, a, b, seed)
        if kind == "convolutional":
            d = d[a * a * b:]                       # drop the filler transient of the delay lines
        if len(d) < 400:
            continue
        sc = score_fn(d[:probe])
        if sc > best[1]:
            best = (off, sc)
        if sc >= threshold:
            break
    off, sc = best
    full = deinterleave(bits[max(off, 0):], kind, a, b, seed)
    if kind == "convolutional":
        full = full[a * a * b:]
    return off, sc, full

"""
Convolutional codes: encoder, hard-decision Viterbi decoder, BLIND identification.

Generator convention in this file ("delay form"): bit k of a generator = tap on the
input delayed by k steps (bit 0 = current input).  Textbook octal numbers
(e.g. 171, 133 for K=7) put the current-input tap in the MOST significant bit; the
helpers to_octal / from_octal convert between the two.

BLIND IDENTIFICATION (rate 1/n), the key idea
---------------------------------------------
Let y1, y2 be the two output streams, y_i = g_i * x (GF(2) convolution).  Then
        g2 * y1  =  g1 * y2      (both sides equal g1*g2*x)
so for the right pair (g1, g2) the sequence  g2*y1 + g1*y2  is ALL ZERO on an
error-free stream.  We compute g*y for every candidate g at once and compare all
pairs with one matrix product (a +/-1 correlation).  The pair with correlation ~1 is
the code.  Bit errors only lower the correlation a little.  A wrong pair gives ~0.
"""
import numpy as np

QUICK_K = 7
FULL_K = 9


def to_octal(g, K):
    rev = int(format(g, f"0{K}b")[::-1], 2)
    return format(rev, "o")


def from_octal(s, K):
    v = int(str(s), 8)
    return int(format(v, f"0{K}b")[::-1], 2)


def degree_K(gens):
    return max(g.bit_length() for g in gens)


# ------------------------------------------------------------------ encoder
def conv_encode(bits, gens, K=None):
    """gens in delay form. Returns interleaved output bits (n per input bit)."""
    bits = np.asarray(bits, dtype=np.uint8)
    K = K or degree_K(gens)
    out = np.zeros((len(bits), len(gens)), dtype=np.uint8)
    for gi, g in enumerate(gens):
        acc = np.zeros(len(bits), dtype=np.uint8)
        for k in range(K):
            if (g >> k) & 1:
                acc[k:] ^= bits[: len(bits) - k]
        out[:, gi] = acc
    return out.reshape(-1)


# ------------------------------------------------------------------ Viterbi (hard decision)
def viterbi_decode(rx, gens, K=None):
    """rx: received bit stream (length multiple of n). Returns decoded info bits."""
    rx = np.asarray(rx, dtype=np.uint8)
    n = len(gens)
    K = K or degree_K(gens)
    rx = rx[: len(rx) // n * n].reshape(-1, n)
    T = len(rx)
    S = 1 << (K - 1)
    ns_all = np.arange(S)
    u = ns_all & 1
    p0 = ns_all >> 1                       # predecessor with dropped bit 0
    p1 = p0 | (1 << (K - 2)) if K > 1 else p0
    # expected output bits for each branch
    def outputs(prev):
        reg = u | (prev << 1)
        o = np.zeros((S, n), dtype=np.uint8)
        for gi, g in enumerate(gens):
            x = reg & g
            par = np.zeros(S, dtype=np.uint8)
            while x.any():
                par ^= (x & 1).astype(np.uint8); x = x >> 1
            o[:, gi] = par
        return o
    E0, E1 = outputs(p0), outputs(p1)
    # branch metric lookup by received n-bit symbol
    syms = np.arange(1 << n)
    sbits = ((syms[:, None] >> np.arange(n - 1, -1, -1)) & 1).astype(np.uint8)
    BM0 = (E0[None, :, :] != sbits[:, None, :]).sum(axis=2).astype(np.int32)   # (2^n, S)
    BM1 = (E1[None, :, :] != sbits[:, None, :]).sum(axis=2).astype(np.int32)
    rsym = (rx * (1 << np.arange(n - 1, -1, -1))).sum(axis=1)
    pm = np.zeros(S, dtype=np.int32)
    dec = np.zeros((T, S), dtype=np.uint8)
    for t in range(T):
        a = pm[p0] + BM0[rsym[t]]
        b = pm[p1] + BM1[rsym[t]]
        sel = b < a
        dec[t] = sel
        pm = np.where(sel, b, a)
        pm -= pm.min()
    st = int(np.argmin(pm))
    out = np.zeros(T, dtype=np.uint8)
    for t in range(T - 1, -1, -1):
        out[t] = st & 1
        b = dec[t, st]
        st = (st >> 1) | (int(b) << (K - 2))
    return out


# ------------------------------------------------------------------ blind identification
def _gmul_table(y, Kmax):
    """T[g] = g * y for every g in 1..2^Kmax-1 (delay form), valid region only. Returns +/-1 float32."""
    N = len(y)
    Nv = N - (Kmax - 1)
    Ys = [y[(Kmax - 1 - k): N - k] for k in range(Kmax)]
    G = 1 << Kmax
    T = np.zeros((G, Nv), dtype=np.uint8)
    for g in range(1, G):
        low = (g & -g).bit_length() - 1
        T[g] = T[g & (g - 1)] ^ Ys[low]
    return (1 - 2 * T[1:].astype(np.float32))          # row index = g-1


def identify_pair(ya, yb, Kmax, tol=0.03):
    """Find (ga, gb) with ga*yb == gb*ya. Returns (score, ga, gb)."""
    A = _gmul_table(yb, Kmax)
    B = _gmul_table(ya, Kmax)
    M = np.abs(A @ B.T) / A.shape[1]
    best = M.max()
    cand = np.argwhere(M >= best - tol)
    # among near-ties prefer the SMALLEST code (multiples of the true pair also satisfy the relation)
    cand = sorted(cand.tolist(), key=lambda ab: (max(ab[0] + 1, ab[1] + 1).bit_length(), -M[ab[0], ab[1]]))
    a, b = cand[0]
    return float(M[a, b]), a + 1, b + 1


def identify(bits, Kmax=QUICK_K, rates=(2, 3), max_bits=6000):
    """Blind rate-1/n identification. Returns best result dict or None (score is 0..1)."""
    bits = np.asarray(bits, dtype=np.uint8)[:max_bits]
    if bits.std() < 0.2 or len(bits) < 200:
        return None
    best = None
    for n in rates:
        for off in range(n):
            s = bits[off:]
            m = len(s) // n * n
            if m < 150 * n:
                continue
            streams = s[:m].reshape(-1, n).T
            sc, g0, g1 = identify_pair(streams[0], streams[1], Kmax)
            gens = [g0, g1]
            scores = [sc]
            if n >= 3:
                for j in range(2, n):
                    sc_j, ga, gb = identify_pair(streams[0], streams[j], Kmax)
                    gens.append(gb); scores.append(sc_j)
                    if ga != g0:
                        scores[-1] = 0.0           # inconsistent first generator -> reject
            score = float(min(scores))
            K = degree_K(gens)
            # prefer higher score; for ties (alignment mirror) prefer smaller K
            key = (round(score, 2), -K, -n)
            if best is None or key > best["_key"]:
                best = dict(_key=key, n=n, offset=off, gens=gens, K=K, score=score)
    if best:
        best.pop("_key")
        best["octal"] = [to_octal(g, best["K"]) for g in best["gens"]]
        best["rate"] = f"1/{best['n']}"
    return best

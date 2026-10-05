"""
Reed-Solomon over GF(256): encoder, decoder (Berlekamp-Massey + Chien + Forney),
and a blind identifier that finds (field polynomial, first root, parity length)
from an aligned byte stream.

WHY blind identification works: every valid codeword c(x) is a multiple of the
generator g(x) = (x - a^fcr)(x - a^(fcr+1)) ... (x - a^(fcr+2t-1)).  So c(a^j) = 0
for exactly 2t consecutive exponents j.  Evaluate c at ALL 255 powers of a and look
for a run of zeros: its start is fcr, its length is n-k = 2t.  Random data gives a
zero at any one root with probability 1/256, so a run of 4+ zeros is no accident.
"""
import numpy as np

PRIM_POLYS = [0x11D, 0x12B, 0x12D, 0x14D, 0x15F, 0x163, 0x165, 0x169,
              0x171, 0x187, 0x18D, 0x1A9, 0x1C3, 0x1CF, 0x1E7, 0x1F5]
COMMON_N = [255, 204, 188, 160, 128, 100, 64, 63, 32, 31]


class GF256:
    def __init__(self, prim=0x11D):
        self.prim = prim
        exp = np.zeros(512, dtype=np.int32)
        log = np.zeros(256, dtype=np.int32)
        x = 1
        for i in range(255):
            exp[i] = x
            log[x] = i
            x <<= 1
            if x & 0x100:
                x ^= prim
            if x == 1 and i < 254:
                raise ValueError("polynomial is not primitive")
        exp[255:510] = exp[:255]
        self.exp, self.log = exp, log

    def mul(self, a, b):
        if a == 0 or b == 0:
            return 0
        return int(self.exp[self.log[a] + self.log[b]])

    def inv(self, a):
        return int(self.exp[255 - self.log[a]])

    def vmul(self, a, b):
        """Vectorised multiply of integer arrays."""
        a = np.asarray(a); b = np.asarray(b)
        r = self.exp[self.log[a] + self.log[b]]
        return np.where((a == 0) | (b == 0), 0, r)


def _poly_mul(gf, p, q):
    r = [0] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        for j, b in enumerate(q):
            r[i + j] ^= gf.mul(a, b)
    return r


def generator_poly(gf, nsym, fcr=0):
    g = [1]
    for i in range(nsym):
        g = _poly_mul(gf, g, [1, int(gf.exp[(fcr + i) % 255])])
    return g          # highest degree first


def rs_encode(data, n, k, prim=0x11D, fcr=0):
    """Systematic encode: returns n bytes = data (k bytes) + (n-k) parity bytes."""
    gf = GF256(prim)
    nsym = n - k
    assert len(data) == k
    g = generator_poly(gf, nsym, fcr)
    buf = list(data) + [0] * nsym
    for i in range(k):
        c = buf[i]
        if c:
            for j in range(1, len(g)):
                buf[i + j] ^= gf.mul(g[j], c)
    return list(data) + buf[k:]


def _eval(gf, poly, x):
    y = 0
    for c in poly:                       # highest first (Horner)
        y = gf.mul(y, x) ^ c
    return y


def syndromes(gf, word, nsym, fcr):
    return [_eval(gf, word, int(gf.exp[(fcr + j) % 255])) for j in range(nsym)]


def rs_decode(word, n, k, prim=0x11D, fcr=0):
    """Decode one codeword (list of n bytes). Returns (data k bytes, n_errors) or (None, -1)."""
    gf = GF256(prim)
    nsym = n - k
    w = list(word)
    S = syndromes(gf, w, nsym, fcr)
    if not any(S):
        return w[:k], 0
    # Berlekamp-Massey -> error locator polynomial Lambda (lowest degree first)
    C, B = [1], [1]
    L, m, b = 0, 1, 1
    for i in range(nsym):
        d = S[i]
        for j in range(1, L + 1):
            if j < len(C):
                d ^= gf.mul(C[j], S[i - j])
        if d == 0:
            m += 1
        else:
            T = C[:]
            coef = gf.mul(d, gf.inv(b))
            if len(C) < len(B) + m:
                C += [0] * (len(B) + m - len(C))
            for j, bj in enumerate(B):
                C[j + m] ^= gf.mul(coef, bj)
            if 2 * L <= i:
                L, B, b, m = i + 1 - L, T, d, 1
            else:
                m += 1
    if L > nsym // 2:
        return None, -1
    # Chien search: position p (0 = first byte) has locator X = a^(n-1-p)
    err_pos = []
    for p in range(n):
        Xinv = int(gf.exp[(255 - (n - 1 - p)) % 255])
        v = 0
        for c in reversed(C):            # evaluate Lambda(Xinv), Horner on lowest-first reversed
            v = gf.mul(v, Xinv) ^ c
        if v == 0:
            err_pos.append(p)
    if len(err_pos) != L:
        return None, -1
    # Forney: Omega = S(x) Lambda(x) mod x^nsym
    omega = [0] * nsym
    for i in range(nsym):
        for j, cj in enumerate(C):
            if i - j >= 0:
                omega[i] ^= gf.mul(S[i - j], cj)
    for p in err_pos:
        X = int(gf.exp[(n - 1 - p) % 255])
        Xinv = gf.inv(X)
        om = 0
        for c in reversed(omega):
            om = gf.mul(om, Xinv) ^ c
        dl = 0                           # formal derivative of Lambda at Xinv (odd terms)
        for j in range(1, len(C), 2):
            dl ^= gf.mul(C[j], int(gf.exp[(255 - (j - 1) * gf.log[X]) % 255]) if X else 0)
        if dl == 0:
            return None, -1
        mag = gf.mul(gf.mul(om, int(gf.exp[((1 - fcr) * gf.log[X]) % 255])), gf.inv(dl))
        w[p] ^= mag
    if any(syndromes(gf, w, nsym, fcr)):
        return None, -1
    return w[:k], len(err_pos)


# ------------------------------------------------------------------ blind identification
def _zero_runs(zmask):
    """Longest cyclic run of True per row. Returns (start, length) arrays."""
    c, m = zmask.shape
    dbl = np.concatenate([zmask, zmask], axis=1)
    starts = np.zeros(c, int); lens = np.zeros(c, int)
    for r in range(c):
        best = cur = 0; bs = 0
        for i in range(2 * m):
            if dbl[r, i]:
                if cur == 0:
                    cs = i
                cur += 1
                if cur > best and cur <= m:
                    best, bs = cur, cs
            else:
                cur = 0
        starts[r], lens[r] = bs % m, min(best, m - 1)
    return starts, lens


def identify(words, n, polys=None, min_run=4, min_fraction=0.2):
    """words: uint8 array (c, n) of codeword-aligned bytes. Returns dict or None."""
    polys = polys or PRIM_POLYS
    words = np.asarray(words, dtype=np.int32)
    if words.shape[0] == 0:
        return None
    best = None
    for prim in polys:
        try:
            gf = GF256(prim)
        except ValueError:
            continue
        roots = gf.exp[:255].astype(np.int32)                       # a^j for j = 0..254
        acc = np.zeros((words.shape[0], 255), dtype=np.int32)
        for i in range(n):                                          # Horner across all roots at once
            acc = gf.vmul(acc, roots[None, :]) ^ words[:, i:i + 1]
        z = acc == 0
        starts, lens = _zero_runs(z)
        ok = lens >= min_run
        if ok.sum() < max(1, int(min_fraction * len(words))):
            continue
        votes = {}
        for s, l in zip(starts[ok], lens[ok]):
            votes[(int(s), int(l))] = votes.get((int(s), int(l)), 0) + 1
        (fcr, nsym), cnt = max(votes.items(), key=lambda kv: kv[1])
        if nsym >= n:
            continue
        score = cnt / len(words)
        if best is None or score > best["score"]:
            best = dict(n=n, k=n - nsym, nsym=nsym, prim=prim, fcr=fcr, score=round(score, 3))
    return best


# ------------------------------------------------------------------ error-tolerant identification
def _bm(gf, seq):
    """Berlekamp-Massey. Returns (L, C) with C the connection polynomial (C[0] = 1, lowest first)."""
    C, B = [1], [1]
    L, m, b = 0, 1, 1
    for i, d0 in enumerate(seq):
        d = d0
        for j in range(1, L + 1):
            if j < len(C):
                d ^= gf.mul(C[j], seq[i - j])
        if d == 0:
            m += 1
            continue
        T = C[:]
        coef = gf.mul(d, gf.inv(b))
        if len(C) < len(B) + m:
            C += [0] * (len(B) + m - len(C))
        for j, bj in enumerate(B):
            C[j + m] ^= gf.mul(coef, bj)
        if 2 * L <= i:
            L, B, b, m = i + 1 - L, T, d, 1
        else:
            m += 1
    return L, C


def _linear_complexity(gf, seq):
    return _bm(gf, seq)[0]


def _extend(gf, S, start, W, step):
    """Starting from the window S[start .. start+W) (cyclic, direction `step`), learn the shortest
    LFSR and count how many further elements it predicts exactly. Returns the count."""
    n = len(S)
    win = [S[(start + step * i) % n] for i in range(W)]
    L, C = _bm(gf, win)
    cnt = 0
    for i in range(W, n - 1):
        pred = 0
        for j in range(1, L + 1):
            if j < len(C):
                pred ^= gf.mul(C[j], S[(start + step * (i - j)) % n])
        if pred != S[(start + step * i) % n]:
            break
        cnt += 1
    return cnt


def identify_noisy(words, n, polys=None, Ws=(16, 12, 10), max_words=8):
    """Tolerates symbol errors. Inside the code's 2t parity roots a corrupted word's syndromes are
    S_j = e(a^j): a sequence of LOW linear complexity (= number of errors) that obeys ONE fixed
    recurrence (the error locator). Outside those roots the syndromes also contain the data and look
    random. Find a low-complexity window, then extend the recurrence both ways until it stops
    predicting: the stretch is the run of parity roots (start = fcr, length = 2t)."""
    polys = polys or PRIM_POLYS
    words = np.asarray(words, dtype=np.int32)[:max_words]
    if words.shape[0] < 2:
        return None
    best = None
    tables = {}
    for prim in polys:
        try:
            gf = GF256(prim)
        except ValueError:
            continue
        roots = gf.exp[:255].astype(np.int32)
        acc = np.zeros((words.shape[0], 255), dtype=np.int32)
        for i in range(n):
            acc = gf.vmul(acc, roots[None, :]) ^ words[:, i:i + 1]
        for W in Ws:                                    # shorter windows catch codes with few parity roots
            max_low = W // 2 - 2
            votes = {}
            for w in range(words.shape[0]):
                S = [int(v) for v in acc[w]]
                comp = np.array([_linear_complexity(gf, [S[(j0 + i) % 255] for i in range(W)]) for j0 in range(255)])
                low = comp <= max_low
                if low.all() or low.sum() < 3:
                    continue
                starts, lens = _zero_runs(low[None, :])
                centre = (int(starts[0]) + int(lens[0]) // 2) % 255
                fwd = _extend(gf, S, centre, W, +1)
                bwd = _extend(gf, S, (centre + W - 1) % 255, W, -1)
                first = (centre - bwd) % 255
                span = W + fwd + bwd
                if 6 <= span < n:
                    votes[(first, span)] = votes.get((first, span), 0) + 1
            if not votes:
                continue
            (fcr, nsym), cnt = max(votes.items(), key=lambda kv: kv[1])
            if cnt < 2 or nsym >= n or nsym < 4:
                continue
            ok = 0
            for w in words:
                d, _ = rs_decode(list(w), n, n - nsym, prim, fcr)
                ok += d is not None
            sc = ok / len(words)
            if sc >= 0.4 and (best is None or sc > best["score"]):
                best = dict(n=n, k=n - nsym, nsym=nsym, prim=prim, fcr=fcr, score=round(sc, 3))
            if best and best["score"] >= 0.8:
                return best
    return best

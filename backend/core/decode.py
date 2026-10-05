"""
Phase 3 receiver. Starts from the demodulated symbols and works out, blindly:
   bit mapping / phase ambiguity -> (de-interleave) -> convolutional code + Viterbi -> LDPC
   -> frame sync, header / payload -> Reed-Solomon -> message.

Every step reports what it tried and how confident it is, so the operator can see
(and override) each decision.
"""
import time
import numpy as np

from . import framing as FR
from .demod import SPS
from .generator import MOD_ORDER, constellation, index_to_label
from .fec import conv as C
from .fec import interleave as I
from .fec import ldpc as LD
from .fec import rs as R

STANDARD_CONV = {("7", "5"), ("171", "133"), ("561", "753"), ("171", "133", "165"), ("23", "35"),
                 ("133", "171"), ("753", "561"), ("133", "171", "165"), ("35", "23")}
CONV_THR = 0.35        # blind conv-code score above this = code present (random data scores ~0.05)


# ------------------------------------------------------------------ bit-stream candidates
def candidates(internal):
    """All bit streams the demodulator could have produced given its phase ambiguity and the
    unknown bit labelling (natural vs Gray). Returns list of (label, bits)."""
    out, seen = [], set()

    def add(label, bits):
        key = hash(bits.tobytes())
        if key not in seen:
            seen.add(key)
            out.append((label, bits))

    if internal["kind"] == "linear":
        mod, M, k = internal["mod"], internal["M"], internal["k"]
        Cc = constellation(mod)
        for gray in (False, True):
            for r in range(M):
                y = internal["syms"] * np.exp(2j * np.pi * r / M)
                idx = np.abs(y[:, None] - Cc[None, :]).argmin(axis=1)
                lab = index_to_label(mod, idx, gray)
                bits = ((lab[:, None] >> np.arange(k - 1, -1, -1)) & 1).astype(np.uint8).reshape(-1)
                add(f"{'Gray' if gray else 'natural'} labelling, rotation {int(round(360 * r / M))} deg", bits)
    else:
        idx, M, k = internal["idx"], internal["M"], internal["k"]
        for rev in (False, True):
            for gray in ((False, True) if M == 4 else (False,)):
                i2 = (M - 1 - idx) if rev else idx
                lab = index_to_label("4FSK", i2, gray) if gray else i2
                bits = ((lab[:, None] >> np.arange(k - 1, -1, -1)) & 1).astype(np.uint8).reshape(-1)
                add(f"{'Gray' if gray else 'natural'} labelling, tones {'reversed' if rev else 'ascending'}", bits)
    return out


# ------------------------------------------------------------------ evidence
def conv_quick(bits, max_bits=3000, deep=False):
    r = C.identify(bits, Kmax=C.QUICK_K, rates=(2, 3), max_bits=max_bits)
    if deep and (not r or r["score"] < CONV_THR):          # codes with constraint length 8-9
        r9 = C.identify(bits, Kmax=C.FULL_K, rates=(2, 3), max_bits=4000)
        if r9 and (not r or r9["score"] > r["score"]):
            r = r9
    return r if r else None


def frame_run_score(bits, L):
    """Length of the longest constant-column run when folded at frame length L, scaled so 64 bits = 1.
    Uncapped on purpose: only the exact de-interleaver alignment keeps the whole preamble contiguous."""
    F = len(bits) // L
    if F < 6:
        return 0.0
    p = 2 * bits[: F * L].reshape(F, L).astype(np.float64).mean(axis=0) - 1
    _, run = FR._longest_circular_run(np.abs(p) > max(0.7, 3.0 / np.sqrt(F)))
    return min(run, 64) / 64.0


PROBE_BITS, PROBE_WEIGHT = 2400, 1.0


def probe_ber(bits, r):
    """Residual error rate of a Viterbi decode of the first PROBE_BITS received bits, using the identified code r."""
    gens, K, n, off = r["gens"], r["K"], r["n"], r["offset"]
    seg = bits[off: off + PROBE_BITS]
    if len(seg) < 20 * n * K:
        return 0.0
    dec = C.viterbi_decode(seg, gens, K)
    re = C.conv_encode(dec, gens, K)
    m = min(len(re), len(seg))
    return float(np.mean(re[K * n: m] != seg[K * n: m]))


def conv_score(bits):
    r = conv_quick(bits)
    return r["score"] if r else 0.0


def structure(bits):
    """How much structure does this stream show? (conv-code score, frame-fold z)."""
    cs = conv_score(bits)
    fz = FR.fold_period(bits[:60000])[1] if len(bits) > 600 else 0.0
    return cs, fz


# ------------------------------------------------------------------ main entry
def ldpc_library(custom=None):
    """Codes the receiver can recognise: the built-in demo codes plus any parity-check matrix the operator uploaded."""
    return list(custom or []) + LD.builtin_library()


def decode(internal, interleaver=None, truth=None, budget_s=120.0, ldpc_codes=None):
    t0 = time.time()
    lib = ldpc_library(ldpc_codes)
    log, notes = [], []
    res = dict(steps=[], warnings=notes)

    # 1. candidates -------------------------------------------------------------
    cands = candidates(internal)
    rows = []
    bonus = []
    for label, bits in cands:
        cs, fz = structure(bits) if not interleaver else (0.0, 0.0)
        r = conv_quick(bits) if (cs >= CONV_THR and not interleaver) else None
        # Equivalent codes exist (a linear re-labelling of the bits gives different generators).
        # Prefer the labelling that yields a textbook / standard code.
        b = 0.1 if r and tuple(r["octal"]) in STANDARD_CONV else 0.0
        # Some phase rotations give the bitwise COMPLEMENT of the data. The code structure and fold strength look the
        # same, but Viterbi decoding of the complement is bad (11-15 % residual errors) and framing then fails. Rank by
        # the residual error rate of a short decoded probe, the quantity that actually matters.
        b += PROBE_WEIGHT * (1.0 - min(1.0, 4.0 * probe_ber(bits, r))) if r else 0.0
        lz = 0.0
        if not r and not interleaver:
            # no convolutional code: does this labelling show an LDPC code from the library? (share of violated checks)
            hit = LD.identify(bits, lib, n_blocks=3)
            if hit:
                lz = hit["z"]
                b += 1.0 + min(lz, 60.0) / 60.0
        bonus.append(b)
        rows.append(dict(label=label, conv=round(cs, 3), frame_z=round(fz, 1), ldpc_z=round(float(lz), 1)))
    if interleaver:
        best_i = 0
        rows = [dict(label=l, conv=None, frame_z=None) for l, _ in cands]
    else:
        keys = [(r["conv"] >= CONV_THR) * 2 + (r["frame_z"] >= 10) for r in rows]
        score = [r["conv"] + min(r["frame_z"], 100) / 100 + bonus[i] for i, r in enumerate(rows)]
        best_i = int(np.argmax(score))
    label, bits = cands[best_i]
    for i, r in enumerate(rows):
        r["chosen"] = (i == best_i)
    res["candidates"] = rows

    # 2. de-interleave (operator supplied) --------------------------------------
    stream = bits
    il = dict(kind="none")
    if interleaver and interleaver.get("kind", "none") != "none":
        kind, a, b = interleaver["kind"], int(interleaver["a"]), int(interleaver["b"])
        seed = int(interleaver.get("seed", 1))
        # try every candidate bit stream until structure appears
        bestc = None
        for ci, (lab, bt) in enumerate(cands):
            if time.time() - t0 > budget_s:
                break
            def sf(x):
                r = conv_quick(x, max_bits=1500)
                if r and r["score"] >= CONV_THR:
                    return r["score"]
                L, z, _ = FR.fold_period(x)                          # frame structure: strict + sensitive parts
                run = frame_run_score(x, L) if L else 0.0
                return 0.9 * (0.8 * run + 0.2 * min(z, 300.0) / 300.0)
            off, sc, full = I.align_deinterleave(bt, kind, a, b, sf, seed=seed, budget_s=min(45.0, budget_s / 2), probe=6000)
            if bestc is None or sc > bestc[0]:
                bestc = (sc, ci, off, full)
            if sc >= 0.5:
                break
        sc, ci, off, full = bestc
        label, bits = cands[ci]
        for i, r in enumerate(rows):
            r["chosen"] = (i == ci)
        stream = full
        il = dict(kind=kind, a=a, b=b, offset=int(off), score=round(float(sc), 3), found=sc >= 0.3)
        if not il["found"]:
            notes.append("No structure appeared after de-interleaving: wrong interleaver type or size?")
    res["interleaver"] = il
    res["bit_mapping"] = label

    # 3. convolutional code ------------------------------------------------------
    quick = conv_quick(stream, deep=True)
    conv_found = bool(quick and quick["score"] >= CONV_THR)
    info = stream
    cinfo = dict(found=False)
    if conv_found:
        full = C.identify(stream, Kmax=C.FULL_K, rates=(quick["n"],), max_bits=8000) or quick
        gens, K, n, off = full["gens"], full["K"], full["n"], full["offset"]
        dec = C.viterbi_decode(stream[off:], gens, K)
        # channel BER estimate: re-encode the decoded bits, compare with what was received
        re = C.conv_encode(dec, gens, K)
        m = min(len(re), len(stream) - off)
        ber = float(np.mean(re[K * n: m] != stream[off + K * n: off + m]))
        info = dec[K:]
        cinfo = dict(found=True, rate=full["rate"], K=int(K), generators_octal=full["octal"],
                     score=round(float(full["score"]), 3), channel_ber=round(ber, 5),
                     decoded_bits=int(len(dec)))
    res["conv"] = cinfo

    # 3b. LDPC (between the convolutional decoder and the framing, mirroring the transmit chain) -------------
    ldpc_info = dict(found=False)
    hit = LD.identify(info, lib, n_blocks=4) if len(info) >= 3 * min(c.n for c in lib) else None
    if hit:
        code = hit["code"]
        st = LD.decode_stream(info, code, hit["offset"])
        info = st.pop("info")
        ldpc_info = dict(found=True, **code.info(), offset=int(hit["offset"]), checks_violated_before=round(hit["sat_fraction"], 3),
                         z=round(hit["z"], 1), **st)
        if st["failed"]:
            notes.append(f"{st['failed']} of {st['n_blocks']} LDPC blocks did not converge (too many bit errors).")
    res["ldpc"] = ldpc_info

    # 4. frames --------------------------------------------------------------------
    det = FR.detect(info)
    res["frame"] = {k: v for k, v in det.items() if k not in ("frame_starts",)}
    res["frame"]["found"] = bool(det.get("found"))
    rs_info = dict(found=False)
    data_stream, data_det = info, det

    # 5. Reed-Solomon -----------------------------------------------------------------
    if det.get("found") and det["frame_len"] % 8 == 0 and 16 <= det["frame_len"] // 8 <= 255:
        n_bytes = det["frame_len"] // 8
        s0 = det["frame_starts"][0] if det["frame_starts"] else det["sync_offset"]
        nw = (len(info) - s0) // det["frame_len"]
        wb = np.packbits(info[s0: s0 + nw * det["frame_len"]]).reshape(nw, n_bytes)
        ident = R.identify(wb[: min(nw, 10)], n_bytes)
        if not ident or ident["score"] < 0.2:
            ident = R.identify_noisy(wb, n_bytes)         # symbol errors present: slower, tolerant method
        if ident and ident["score"] >= 0.2:
            n, k, prim, fcr = ident["n"], ident["k"], ident["prim"], ident["fcr"]
            datas, corrected, failed = [], 0, 0
            for w in wb:
                d, ne = R.rs_decode(list(w), n, k, prim, fcr)
                if d is None:
                    failed += 1
                    datas.append(bytes(w[:k]))
                else:
                    corrected += ne
                    datas.append(bytes(d))
            rs_info = dict(found=True, n=n, k=k, parity=n - k, t=(n - k) // 2, prim=hex(prim), fcr=fcr,
                           codewords=int(nw), uncorrectable=int(failed), corrected_symbols=int(corrected),
                           score=ident["score"])
            data_stream = np.unpackbits(np.frombuffer(b"".join(datas), np.uint8))
            data_det = FR.detect(data_stream)
            if not data_det.get("found"):
                data_det = dict(det, frame_len=k * 8, payload_bits=det["payload_bits"] - (n - k) * 8)
                data_det["frame_starts"] = [i * k * 8 for i in range(nw)]
    res["rs"] = rs_info

    # 6. polarity + message ------------------------------------------------------------
    msg = dict(found=False)
    if data_det.get("found"):
        best = None
        for inv in (0, 1):
            ds = data_stream ^ inv
            dd = FR.detect(ds) if inv else data_det
            if not dd.get("found"):
                continue
            frames = FR.extract_frames(ds, dd, 400)
            payload = b"".join(bytes.fromhex(f["payload_hex"]) for f in frames)
            known = 1.0 if dd.get("sync_name") else 0.0
            sc = known + FR.printable_fraction(payload)
            if best is None or sc > best[0]:
                best = (sc, inv, ds, dd, frames, payload)
        if best:
            _, inv, ds, dd, frames, payload = best
            res["frame"] = {k: v for k, v in dd.items() if k not in ("frame_starts",)}
            res["frame"]["found"] = True
            res["frame"]["polarity_inverted"] = bool(inv)
            res["frames"] = frames[:8]
            txt = "".join(chr(c) if 32 <= c < 127 else "." for c in payload)
            msg = dict(found=True, n_frames=len(frames), bytes=len(payload), text=txt[:1500],
                       hex=payload[:256].hex().upper(), printable=round(FR.printable_fraction(payload), 3))
            res["_payload"] = payload
            res["_bits"] = ds
    else:
        notes.append("No repeating frame structure found in the decoded stream. Needs >= ~12 frames with a "
                     "fixed sync word; short captures or streams without sync words cannot be framed.")
    res["message"] = msg

    # 7. ground truth (demo signals) -------------------------------------------------------
    if truth is not None and msg["found"]:
        tm = truth["message"]
        got = res["_payload"]
        n = min(len(tm), len(got))
        # first received frame may be partial: align on the longest matching prefix offset
        best = 0.0
        for off in range(0, min(len(got), truth["payload_len"] * 3)):
            seg = got[off: off + n - off]
            tm_seg = tm[: len(seg)]
            if len(seg) < 16:
                continue
            best = max(best, float(np.mean(np.frombuffer(seg, np.uint8) == np.frombuffer(tm_seg, np.uint8))))
        # frames may be dropped at the start; also test shifted truth
        for shift in range(0, 4):
            seg = got[: len(tm) - shift * truth["payload_len"]]
            tm_seg = tm[shift * truth["payload_len"]: shift * truth["payload_len"] + len(seg)]
            if len(seg) >= 16:
                best = max(best, float(np.mean(np.frombuffer(seg, np.uint8) == np.frombuffer(tm_seg, np.uint8))))
        res["ground_truth_check"] = dict(payload_byte_accuracy=round(best, 4),
                                         fec=truth["fec"], conv=truth["conv"], interleaver=truth["interleaver"],
                                         ldpc=truth.get("ldpc", "none"))
    res["elapsed_s"] = round(time.time() - t0, 1)
    return res

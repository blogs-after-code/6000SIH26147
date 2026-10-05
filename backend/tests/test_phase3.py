import random
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.core import framing as FR
from backend.core import decode as DEC
from backend.core.chain import build_chain
from backend.core.fec import conv as C
from backend.core.fec import interleave as I
from backend.core.fec import rs as R
from backend.main import app

client = TestClient(app)


# ------------------------------------------------------------------ Reed-Solomon
@pytest.mark.parametrize("prim,fcr,n,k", [(0x11D, 0, 255, 223), (0x187, 112, 255, 223), (0x11D, 1, 204, 188), (0x12B, 1, 64, 48)])
def test_rs_corrects_up_to_t(prim, fcr, n, k):
    rnd = random.Random(1)
    data = [rnd.randrange(256) for _ in range(k)]
    cw = R.rs_encode(data, n, k, prim, fcr)
    w = cw[:]
    for p in rnd.sample(range(n), (n - k) // 2):
        w[p] ^= rnd.randrange(1, 256)
    out, ne = R.rs_decode(w, n, k, prim, fcr)
    assert out == data and ne == (n - k) // 2
    w2 = cw[:]
    for p in rnd.sample(range(n), (n - k) // 2 + 4):                # beyond t: the decoder must not return the original
        w2[p] ^= rnd.randrange(1, 256)
    out2, _ = R.rs_decode(w2, n, k, prim, fcr)
    assert out2 is None or out2 != data


def test_rs_matches_reference_library():
    reedsolo = pytest.importorskip("reedsolo")
    data = bytes(random.Random(2).randrange(256) for _ in range(223))
    mine = R.rs_encode(list(data), 255, 223, 0x11D, 0)
    ref = list(reedsolo.RSCodec(32, nsize=255, fcr=0, prim=0x11D, generator=2).encode(data))
    assert mine == ref


def test_rs_blind_identification():
    rnd = random.Random(3)
    words = np.array([R.rs_encode([rnd.randrange(256) for _ in range(188)], 204, 188, 0x187, 112) for _ in range(6)])
    r = R.identify(words, 204)
    assert (r["k"], r["prim"], r["fcr"]) == (188, 0x187, 112)
    assert R.identify(np.random.randint(0, 256, (6, 204)), 204) is None


# ------------------------------------------------------------------ convolutional codes
CODES = [(("7", "5"), 3), (("23", "35"), 5), (("171", "133"), 7), (("171", "133", "165"), 7), (("561", "753"), 9)]


@pytest.mark.parametrize("octs,K", CODES)
def test_conv_blind_identification_and_viterbi(octs, K):
    rng = np.random.default_rng(0)
    info = rng.integers(0, 2, 6000, dtype=np.uint8)
    gens = [C.from_octal(o, K) for o in octs]
    rx = C.conv_encode(info, gens, K)[1:].copy()                    # unknown start offset
    rx[rng.random(len(rx)) < 0.01] ^= 1                             # 1% channel errors
    r = C.identify(rx, Kmax=9 if K == 9 else 7)
    assert sorted(r["octal"]) == sorted(octs) and r["score"] > 0.5
    dec = C.viterbi_decode(rx[r["offset"]:], r["gens"], r["K"])
    assert min(np.mean(dec[50:3000] != info[50 + s:3000 + s]) for s in range(4)) < 0.002


def test_conv_identify_rejects_random_data():
    r = C.identify(np.random.default_rng(1).integers(0, 2, 6000, dtype=np.uint8))
    assert r is None or r["score"] < DEC.CONV_THR


# ------------------------------------------------------------------ interleavers
@pytest.mark.parametrize("kind,a,b", [("block", 12, 20), ("diagonal", 9, 16), ("prandom", 10, 24)])
def test_interleaver_roundtrip(kind, a, b):
    x = np.random.default_rng(2).integers(0, 2, 5000, dtype=np.uint8)
    z = I.deinterleave(I.interleave(x, kind, a, b), kind, a, b)
    assert np.array_equal(z, x[: len(z)])


def test_convolutional_interleaver_roundtrip():
    x = np.random.default_rng(2).integers(0, 2, 5000, dtype=np.uint8)
    z = I.deinterleave(I.interleave(x, "convolutional", 6, 4), "convolutional", 6, 4)
    d = 5 * 6 * 4
    assert np.array_equal(z[d:], x[: len(z) - d])


# ------------------------------------------------------------------ framing
@pytest.mark.parametrize("P,nf,off,ber", [(24, 40, 0, 0), (24, 40, 37, 0.01), (100, 20, 5, 0.003), (180, 16, 3, 0)])
def test_frame_detection(P, nf, off, ber):
    b = build_chain(payload_len=P, n_frames=nf)[0][off:].copy()
    b[np.random.default_rng(1).random(len(b)) < ber] ^= 1
    d = FR.detect(b)
    assert d["found"] and d["frame_len"] == (8 + P) * 8
    assert d["sync_hex"] == "1ACFFC1D" and d["header_bits"] == 32 and d["payload_bits"] == P * 8


# ------------------------------------------------------------------ whole receiver at bit level
def _decode(ber=0.0, il=None, **kw):
    bits, truth = build_chain(**kw)
    rx = bits[11:].copy()
    rx[np.random.default_rng(0).random(len(rx)) < ber] ^= 1
    internal = dict(kind="fsk", idx=rx.astype(int), M=2, k=1)
    return DEC.decode(internal, interleaver=il, truth=truth)


def test_decoder_conv_only():
    r = _decode(conv="conv_k7", n_frames=60, ber=0.01)
    assert r["conv"]["generators_octal"] == ["171", "133"]
    assert r["ground_truth_check"]["payload_byte_accuracy"] == 1.0


def test_decoder_rs_with_symbol_errors():
    r = _decode(fec="rs_255_223", n_frames=14, ber=0.0015)
    assert r["rs"]["found"] and r["rs"]["corrected_symbols"] > 0
    assert r["ground_truth_check"]["payload_byte_accuracy"] == 1.0


def test_decoder_concatenated_rs_conv():
    r = _decode(fec="rs_204_188", conv="conv_k9", n_frames=30, ber=0.01)
    assert r["conv"]["K"] == 9 and r["rs"]["n"] == 204
    assert r["ground_truth_check"]["payload_byte_accuracy"] == 1.0


def test_decoder_block_interleaver_and_conv():
    r = _decode(conv="conv_k7", interleaver="block", ia=12, ib=20, n_frames=80, ber=0.005,
                il=dict(kind="block", a=12, b=20))
    assert r["interleaver"]["found"] and r["ground_truth_check"]["payload_byte_accuracy"] == 1.0


def test_decoder_reports_no_structure_on_random_bits():
    rx = np.random.default_rng(4).integers(0, 2, 20000)
    r = DEC.decode(dict(kind="fsk", idx=rx, M=2, k=1))
    assert not r["frame"]["found"] and not r["conv"]["found"] and not r["message"]["found"] and r["warnings"]


# ------------------------------------------------------------------ the whole thing, from IQ samples
def _e2e(**demo):
    d = client.post("/api/demo", json=demo).json()
    m = client.post(f"/api/demod/{d['session']}", json={}).json()
    dec = client.post(f"/api/decode/{d['session']}", json={}).json()
    return d, m, dec


def test_e2e_qpsk_rs_plus_conv():
    d, m, dec = _e2e(mod="QPSK", snr_db=12, cfo_hz=4000, fec="rs_204_188", conv="conv_k7", n_frames=14)
    assert m["modulation"] == "QPSK"
    assert dec["conv"]["generators_octal"] == ["171", "133"] and dec["rs"]["k"] == 188
    assert dec["ground_truth_check"]["payload_byte_accuracy"] == 1.0
    assert client.get(f"/api/payload/{d['session']}").status_code == 200


def test_e2e_gray_16qam_rs():
    d, m, dec = _e2e(mod="16QAM", snr_db=14, cfo_hz=2000, fec="rs_255_223", n_frames=12, gray=True)
    assert m["modulation"] == "16QAM" and "Gray" in dec["bit_mapping"]
    assert dec["ground_truth_check"]["payload_byte_accuracy"] == 1.0


def test_e2e_2fsk_plain_frames():
    d, m, dec = _e2e(mod="2FSK", snr_db=12, cfo_hz=2000, n_frames=40)
    assert m["modulation"] == "2FSK" and dec["ground_truth_check"]["payload_byte_accuracy"] == 1.0


def test_api_decode_guards():
    d = client.post("/api/demo", json={"mod": "QPSK"}).json()
    assert client.post(f"/api/decode/{d['session']}", json={}).status_code == 400        # demod not run yet
    client.post(f"/api/demod/{d['session']}", json={})
    bad = client.post(f"/api/decode/{d['session']}", json={"interleaver": {"kind": "zigzag", "a": 4, "b": 4}})
    assert bad.status_code == 400
    assert client.post("/api/decode/nope", json={}).status_code == 404

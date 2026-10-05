"""LDPC: code construction, decoder, alignment search, alist round-trip, pipeline integration, false positives."""
import numpy as np
import pytest

from backend.core import decode as DEC
from backend.core.chain import build_chain
from backend.core.fec import ldpc as L


def test_builtin_codes_are_valid_and_encodable():
    for nm in L.BUILTIN:
        c = L.get_builtin(nm)
        u = np.random.default_rng(0).integers(0, 2, 3 * c.k).astype(np.uint8)
        cw = c.encode(u)
        assert c.syndrome(cw.reshape(-1, c.n)).sum() == 0
        assert np.array_equal(cw.reshape(-1, c.n)[:, :c.k].reshape(-1), u)          # systematic


def test_decoder_corrects_errors():
    c = L.get_builtin("ldpc_512_256")
    rng = np.random.default_rng(1)
    cw = c.encode(rng.integers(0, 2, 8 * c.k).astype(np.uint8))
    rx = cw.copy(); rx[rng.random(len(rx)) < 0.03] ^= 1
    dec, ok, _ = c.decode(rx.reshape(-1, c.n))
    assert ok.all() and np.array_equal(dec.reshape(-1), cw)


def test_alignment_finds_offset_and_rejects_random():
    c = L.get_builtin("ldpc_512_256")
    rng = np.random.default_rng(2)
    cw = c.encode(rng.integers(0, 2, 8 * c.k).astype(np.uint8))
    rx = np.concatenate([rng.integers(0, 2, 77).astype(np.uint8), cw]); rx[rng.random(len(rx)) < 0.03] ^= 1
    off, fr, z = c.align(rx)
    assert off == 77 and fr < L.SAT_MAX and z > L.Z_MIN
    assert L.identify(rng.integers(0, 2, 6000).astype(np.uint8), L.builtin_library()) is None


def test_alist_round_trip_and_dense_parse():
    c = L.get_builtin("ldpc_256_128")
    c2 = L.parse_matrix(L.to_alist(c), name="x")
    assert c2.n == c.n and c2.m == c.m and all(np.array_equal(a, b) for a, b in zip(c.rows, c2.rows))
    H = np.zeros((c.m, c.n), int)
    for i, r in enumerate(c.rows):
        H[i, r] = 1
    c3 = L.parse_matrix("\n".join("".join(map(str, r)) for r in H))
    assert all(np.array_equal(a, b) for a, b in zip(c.rows, c3.rows))
    with pytest.raises(ValueError):
        L.parse_matrix("not a matrix")


@pytest.mark.parametrize("kw", [dict(ldpc="ldpc_512_256"), dict(ldpc="ldpc_1024_512"),
                                dict(ldpc="ldpc_512_256", conv="conv_k7"),
                                dict(ldpc="ldpc_512_256", fec="rs_204_188", n_frames=14)])
def test_pipeline_decodes_ldpc_chains(kw):
    kw = dict(dict(n_frames=16), **kw)
    bits, truth = build_chain(seed=3, **kw)
    rx = bits[23:].copy(); rx[np.random.default_rng(1).random(len(rx)) < 0.02] ^= 1
    r = DEC.decode(dict(kind="fsk", idx=rx.astype(int), M=2, k=1), truth=truth)
    assert r["ldpc"]["found"] and r["ldpc"]["name"] == kw["ldpc"] and r["ldpc"]["failed"] == 0
    assert r["message"]["found"] and r["ground_truth_check"]["payload_byte_accuracy"] == 1.0


def test_uploaded_matrix_is_used():
    custom = L.make_ira(384, 192, seed=99, name="mine")                      # not in the built-in library
    bits, truth = build_chain(seed=3, n_frames=16)
    cw = custom.encode(bits)
    rx = cw[31:].copy(); rx[np.random.default_rng(1).random(len(rx)) < 0.02] ^= 1
    internal = dict(kind="fsk", idx=rx.astype(int), M=2, k=1)
    assert not DEC.decode(internal)["ldpc"]["found"]                         # unknown without the matrix
    r = DEC.decode(internal, ldpc_codes=[L.parse_matrix(L.to_alist(custom), name="mine")], truth=truth)
    assert r["ldpc"]["found"] and r["ldpc"]["name"] == "mine" and r["ldpc"]["source"] == "uploaded"
    assert r["message"]["found"] and r["ground_truth_check"]["payload_byte_accuracy"] == 1.0


@pytest.mark.parametrize("kw", [dict(), dict(conv="conv_k7"), dict(fec="rs_204_188", n_frames=14)])
def test_no_false_ldpc_on_other_chains(kw):
    bits, truth = build_chain(seed=4, **dict(dict(n_frames=16), **kw))
    r = DEC.decode(dict(kind="fsk", idx=bits[17:].astype(int), M=2, k=1), truth=truth)
    assert not r["ldpc"]["found"]
    assert r["message"]["found"]


def test_ldpc_behind_a_block_interleaver():
    bits, truth = build_chain(seed=3, ldpc="ldpc_512_256", interleaver="block", ia=12, ib=20, n_frames=16)
    r = DEC.decode(dict(kind="fsk", idx=bits[23:].astype(int), M=2, k=1), interleaver=dict(kind="block", a=12, b=20, seed=1), truth=truth, budget_s=120)
    assert r["ldpc"]["found"] and r["message"]["found"] and r["ground_truth_check"]["payload_byte_accuracy"] == 1.0


def test_report_contains_ldpc_row():
    from backend.core.report import build_report
    from backend.core.spectral import analyze
    from backend.core.generator import gen_signal
    bits, truth = build_chain(seed=3, ldpc="ldpc_512_256", n_frames=16)
    x, _ = gen_signal("QPSK", fs=100e3, symbol_rate=12.5e3, snr_db=14, cfo_hz=0, seed=2, bits=bits)
    from backend.core.pipeline import demodulate
    spec = analyze(x, 100e3)["params"]
    res, b, internal = demodulate(x, 100e3, spec)
    dec = DEC.decode(internal, truth=truth)
    s = dict(x=x, fs=100e3, spec=spec, source=dict(filename="t", kind="synthetic"), demod=res, decode_res=_clean(dec))
    pdf = build_report(s, analyze)
    assert pdf[:4] == b"%PDF" and len(pdf) > 20000


def _clean(o):
    from backend.main import _clean as c
    return c(o)

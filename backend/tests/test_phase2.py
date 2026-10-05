import numpy as np
import pytest
from scipy.signal import resample_poly
from fastapi.testclient import TestClient

from backend.core.generator import gen_signal, MOD_ORDER
from backend.core.spectral import analyze
from backend.core.pipeline import demodulate
from backend.core.evaluate import ber_aligned
from backend.core import demod as D
from backend.main import app

client = TestClient(app)
FS = 100e3


def _ber(mod, snr, cfo=5e3, fs=FS, rs=12.5e3, beta=0.35, seed=3, **kw):  # kw -> demodulate()
    x, t = gen_signal(mod, fs=fs, symbol_rate=rs, snr_db=snr, cfo_hz=cfo, beta=beta, seed=seed, n_symbols=4000)
    spec = analyze(x, fs)["params"]
    res, bits, internal = demodulate(x, fs, spec, **kw)
    k = internal["k"]
    if res["modulation"] != mod:
        return res, 1.0
    best = 1.0
    for r in range(internal["M"]):
        if internal["kind"] == "linear":
            b, _ = D.demap_linear(internal["syms"] * np.exp(2j * np.pi * r / internal["M"]), internal["mod"])
        else:
            b = bits
        best = min(best, ber_aligned(t["bits"], b, k)[0])
    return res, best


@pytest.mark.parametrize("mod,snr", [("BPSK", 8), ("QPSK", 10), ("8PSK", 12), ("16QAM", 12), ("2FSK", 10), ("4FSK", 14)])
def test_blind_end_to_end(mod, snr):
    res, ber = _ber(mod, snr)
    assert res["modulation"] == mod
    assert abs(res["symbol_rate"] - 12.5e3) < 30
    assert ber < 0.01, (mod, ber)


def test_non_integer_samples_per_symbol():
    """Resample so fs / Rs is not an integer: the pipeline must still lock."""
    x, t = gen_signal("QPSK", fs=100e3, symbol_rate=12.5e3, snr_db=12, cfo_hz=4e3, seed=5, n_symbols=4000)
    y = resample_poly(x, 97, 100).astype(np.complex64)          # new fs = 97 kHz
    fs2 = 97e3
    spec = analyze(y, fs2)["params"]
    res, bits, internal = demodulate(y, fs2, spec)
    assert res["modulation"] == "QPSK"
    assert abs(res["symbol_rate"] - 12.5e3) < 30
    best = min(ber_aligned(t["bits"], D.demap_linear(internal["syms"] * np.exp(2j * np.pi * r / 4), "QPSK")[0], 2)[0] for r in range(4))
    assert best < 0.01


def test_roll_off_mismatch_tolerated():
    _, ber = _ber("QPSK", 12, beta=0.5)
    assert ber < 0.01


def _manual():
    x, t = gen_signal("QPSK", snr_db=12, cfo_hz=5e3, seed=3, n_symbols=4000)
    spec = analyze(x, FS)["params"]
    res, bits, internal = demodulate(x, FS, spec, mod="QPSK", symbol_rate=12.5e3)
    best = min(ber_aligned(t["bits"], D.demap_linear(internal["syms"] * np.exp(2j * np.pi * r / 4), "QPSK")[0], 2)[0] for r in range(4))
    return res, best


def test_manual_override():
    res, ber = _manual()
    assert res["modulation_source"] == "override" and ber < 0.01


def test_classifier_accuracy_on_fresh_signals():
    ok = n = 0
    for mod in ("BPSK", "QPSK", "8PSK", "16QAM"):
        for seed in range(100, 108):
            x, _ = gen_signal(mod, snr_db=11, cfo_hz=2e3, seed=seed, n_symbols=3000)
            res, _, _ = demodulate(x, FS, analyze(x, FS)["params"])
            n += 1
            ok += res["modulation"] == mod
    assert ok / n >= 0.9, ok / n


def test_noise_only_is_rejected_cleanly():
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(30000) + 1j * rng.standard_normal(30000)).astype(np.complex64)
    spec = analyze(x, FS)["params"]
    try:
        res, _, _ = demodulate(x, FS, spec)
        assert res["warnings"], "noise should at least raise warnings"
    except (ValueError, RuntimeError):
        pass


def test_api_flow_and_bits_download():
    d = client.post("/api/demo", json={"mod": "16QAM", "snr_db": 14, "cfo_hz": -6000}).json()
    r = client.post(f"/api/demod/{d['session']}", json={})
    assert r.status_code == 200
    j = r.json()
    assert j["modulation"] == "16QAM" and j["ground_truth_check"]["ber"] < 0.01
    assert len(j["plots"]["constellation"]) > 100
    b = client.get(f"/api/bits/{d['session']}?fmt=bin")
    assert b.status_code == 200 and len(b.content) > 500
    assert client.get("/api/bits/nope").status_code == 404


def test_api_rejects_bad_input():
    d = client.post("/api/demo", json={"mod": "QPSK"}).json()
    assert client.post(f"/api/demod/{d['session']}", json={"mod": "WXYZ"}).status_code == 400
    assert client.post("/api/demod/doesnotexist", json={}).status_code == 404

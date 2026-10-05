import numpy as np
import pytest
from backend.core.generator import gen_signal, save_cf32, save_wav_iq, rrc_taps
from backend.core.loader import load_file, load_raw_iq
from backend.core.spectral import analyze

FS = 100e3


def test_rrc_unit_energy():
    assert abs(np.sum(rrc_taps(0.35, 8) ** 2) - 1) < 1e-9


@pytest.mark.parametrize("mod", ["BPSK", "QPSK", "8PSK", "16QAM", "2FSK"])
def test_center_bw_snr(mod):
    cfo, snr = 5e3, 10.0
    x, truth = gen_signal(mod, fs=FS, symbol_rate=12.5e3, n_symbols=4000,
                          snr_db=snr, cfo_hz=cfo, seed=1)
    r = analyze(x, FS)["params"]
    assert abs(r["center_freq_hz"] - cfo) < 600, r
    assert abs(r["snr_db_fullband"] - snr) < 2.0, r
    if mod != "2FSK":
        expected_bw = 12.5e3 * 1.35
        assert abs(r["bandwidth_hz"] - expected_bw) / expected_bw < 0.2, r


def test_cf32_roundtrip(tmp_path):
    x, _ = gen_signal("QPSK", seed=2)
    p = tmp_path / "a.cf32"
    save_cf32(p, x)
    y, fs, info = load_file("a.cf32", p.read_bytes(), fs=FS)
    assert np.allclose(x, y) and fs == FS


def test_wav_roundtrip(tmp_path):
    x, _ = gen_signal("QPSK", seed=3, cfo_hz=-7e3)
    p = tmp_path / "a.wav"
    save_wav_iq(p, x, FS)
    y, fs, info = load_file("a.wav", p.read_bytes())
    assert fs == FS
    assert abs(analyze(y, fs)["params"]["center_freq_hz"] + 7e3) < 600


def test_raw_needs_fs():
    with pytest.raises(ValueError):
        load_file("a.iq", b"\x00" * 64)


def test_cu8_and_ci16():
    a = np.array([0, 255, 127, 128], dtype=np.uint8).tobytes()
    assert load_raw_iq(a, "cu8").shape == (2,)
    b = np.array([1000, -1000], dtype=np.int16).tobytes()
    assert abs(load_raw_iq(b, "ci16")[0] - (1000 / 32768 - 1j * 1000 / 32768)) < 1e-6

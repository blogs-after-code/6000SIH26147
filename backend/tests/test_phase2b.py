import json
import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
from onnx import TensorProto, helper, numpy_helper

from backend.core import dl_classifier as DL
from backend.core import demod as D
from backend.core.generator import gen_signal
from backend.core.spectral import analyze

LABELS = ["BPSK", "QPSK", "QAM16", "GFSK", "QAM64"]


def _make_model(directory, L=128, labels=LABELS):
    """A tiny stand-in network with the same input/output contract as the Kaggle model."""
    C = len(labels)
    rng = np.random.default_rng(0)
    W = numpy_helper.from_array(rng.normal(size=(2, C)).astype(np.float32), "W")
    b = numpy_helper.from_array(np.zeros(C, np.float32), "b")
    nodes = [helper.make_node("ReduceMean", ["iq"], ["m"], axes=[2], keepdims=0),
             helper.make_node("MatMul", ["m", "W"], ["z"]), helper.make_node("Add", ["z", "b"], ["logits"])]
    g = helper.make_graph(nodes, "tiny", [helper.make_tensor_value_info("iq", TensorProto.FLOAT, ["batch", 2, L])],
                          [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", C])], [W, b])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.save(m, str(directory / DL.ONNX_NAME))
    json.dump(dict(labels=labels, input_len=L, sps=8, dataset="test stand-in"), open(directory / DL.META_NAME, "w"))


def test_label_mapping():
    assert DL.to_app_label("QAM16") == "16QAM" and DL.to_app_label("16QAM") == "16QAM"
    assert DL.to_app_label("GFSK") == "FSK" and DL.to_app_label("CPFSK") == "FSK"
    assert DL.to_app_label("QAM64") is None and DL.to_app_label("AM-DSB") is None and DL.to_app_label("WBFM") is None
    assert DL.to_app_label("8PSK") == "8PSK" and DL.to_app_label("BPSK") == "BPSK"


def test_missing_model_is_reported_not_fatal(tmp_path):
    c = DL.DeepClassifier(str(tmp_path))
    assert not c.available and "not found" in c.reason


def test_corrupt_model_is_reported_not_fatal(tmp_path):
    (tmp_path / DL.ONNX_NAME).write_bytes(b"not an onnx file")
    json.dump(dict(labels=["A"], input_len=128), open(tmp_path / DL.META_NAME, "w"))
    c = DL.DeepClassifier(str(tmp_path))
    assert not c.available and "could not load" in c.reason


def test_inference_contract_and_agreement(tmp_path):
    _make_model(tmp_path)
    c = DL.DeepClassifier(str(tmp_path))
    assert c.available
    x, _ = gen_signal("QPSK", snr_db=14, cfo_hz=3000, seed=1, n_symbols=3000)
    spec = analyze(x, 100e3)["params"]
    xb = D.channelize(x, 100e3, spec["center_freq_hz"], spec["bandwidth_hz"])
    out = c.predict(xb, 100e3, 12.5e3)
    assert out["n_frames"] > 1 and len(out["top"]) == 3
    assert abs(sum(t["prob"] for t in out["top"]) - 1) < 1.0 and out["top"][0]["prob"] >= out["top"][1]["prob"]
    assert c.agrees("QAM16", "16QAM") is True and c.agrees("QAM16", "QPSK") is False
    assert c.agrees("QAM64", "16QAM") is None and c.agrees("GFSK", "2FSK") is True


def test_too_short_capture_raises(tmp_path):
    _make_model(tmp_path, L=1024)
    c = DL.DeepClassifier(str(tmp_path))
    with pytest.raises(ValueError):
        c.predict(np.zeros(500, np.complex64), 100e3, 12.5e3)


def test_pipeline_and_api_expose_cnn_opinion(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from backend.main import app
    _make_model(tmp_path)
    monkeypatch.setattr(DL, "_instance", DL.DeepClassifier(str(tmp_path)))
    c = TestClient(app)
    assert c.get("/api/health").json()["cnn_ready"] is True
    m = c.get("/api/models").json()
    assert m["cnn_available"] and m["cnn"]["dataset"] == "test stand-in" and m["rf"] is not None
    d = c.post("/api/demo", json={"mod": "QPSK", "snr_db": 14, "cfo_hz": 3000}).json()
    r = c.post(f"/api/demod/{d['session']}", json={}).json()
    assert r["deep"]["available"] and len(r["deep"]["top"]) == 3


def test_api_without_model(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from backend.main import app
    monkeypatch.setattr(DL, "_instance", DL.DeepClassifier(str(tmp_path)))
    c = TestClient(app)
    assert c.get("/api/models").json()["cnn_available"] is False
    d = c.post("/api/demo", json={"mod": "QPSK", "snr_db": 14}).json()
    r = c.post(f"/api/demod/{d['session']}", json={}).json()
    assert r["deep"]["available"] is False and "not found" in r["deep"]["reason"]

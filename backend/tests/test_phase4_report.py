"""Phase 4: PDF report export, CNN gating and label-fix helper."""
import json
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def _session(**demo):
    r = client.post("/api/demo", json=dict(mod="QPSK", snr_db=15, cfo_hz=2000, **demo))
    assert r.status_code == 200
    return r.json()["session"]


def test_report_after_demod_is_a_pdf_with_content():
    sid = _session()
    assert client.post(f"/api/demod/{sid}", json={}).status_code == 200
    r = client.get(f"/api/report/{sid}")
    assert r.status_code == 200 and r.headers["content-type"] == "application/pdf"
    assert r.content[:5] == b"%PDF-" and len(r.content) > 20_000          # includes the plots


def test_report_with_full_decode_contains_the_message():
    sid = _session(fec="rs_204_188", conv="conv_k7", n_frames=20)
    client.post(f"/api/demod/{sid}", json={})
    d = client.post(f"/api/decode/{sid}", json={}).json()
    assert d["message"]["found"]
    r = client.get(f"/api/report/{sid}")
    assert r.status_code == 200 and r.content[:5] == b"%PDF-"
    pytest.importorskip("pypdf")
    from pypdf import PdfReader
    import io
    text = " ".join(p.extract_text() for p in PdfReader(io.BytesIO(r.content)).pages)
    assert "QPSK" in text and "Reed-Solomon" in text and "RS(204,188)" in text and "Method limits" in text
    assert d["message"]["text"][:20].strip() in text.replace("\n", " ")


def test_report_unknown_session_is_404_and_works_before_demod():
    assert client.get("/api/report/doesnotexist").status_code == 404
    sid = _session()
    assert client.get(f"/api/report/{sid}").status_code == 200           # capture + parameters only


def test_fix_cnn_labels_remaps_by_index():
    from scripts.fix_cnn_labels import relabel, FIXED, CLASSES_TXT
    old = [f"'{n}'," for n in CLASSES_TXT]
    meta = dict(labels=old, class_accuracy_ge10dB={n: i / 100 for i, n in enumerate(old)},
                top_confusions_ge10dB=[[5, old[20], old[19]]], app_map={})
    new = relabel(meta, FIXED, "fixed", False)
    assert new["labels"] == FIXED and new["labels_verified"] is False
    assert list(new["class_accuracy_ge10dB"].items())[20] == (FIXED[20], 0.20)       # value follows the index
    assert new["top_confusions_ge10dB"] == [[5, FIXED[20], FIXED[19]]]
    assert new["app_map"]["QPSK"] == "QPSK" and new["app_map"]["16QAM"] == "16QAM" and new["app_map"]["FM"] is None


def test_cnn_warnings_need_verified_labels_and_measured_transfer(tmp_path):
    from backend.core import dl_classifier as DL
    def make(verified, agreement):
        d = tmp_path / f"m{verified}{agreement}"; d.mkdir()
        import shutil, os
        shutil.copy(os.path.join(DL.MODELS_DIR, DL.ONNX_NAME), d / DL.ONNX_NAME)
        meta = json.load(open(os.path.join(DL.MODELS_DIR, DL.META_NAME)))
        meta["labels_verified"] = verified
        json.dump(meta, open(d / DL.META_NAME, "w"))
        if agreement is not None:
            json.dump(dict(agreement=agreement), open(d / DL.TRANSFER_NAME, "w"))
        return DL.DeepClassifier(str(d))
    assert make(True, 0.9).warnings_trusted is True
    assert make(False, 0.9).warnings_trusted is False          # labels not confirmed
    assert make(True, 0.3).warnings_trusted is False           # poor agreement on our own signals
    assert make(True, None).warnings_trusted is False          # never measured


@pytest.mark.parametrize("n_frames", [17, 20])
def test_large_capture_decodes_regression_complemented_rotation(n_frames):
    """Regression: with >= 17 frames the decoder used to pick a phase rotation whose bits are the complement of the data
    (Viterbi left 11-15 % errors, framing failed). Candidates are now ranked by decoded probe error rate."""
    sid = _session(fec="rs_204_188", conv="conv_k7", n_frames=n_frames)
    client.post(f"/api/demod/{sid}", json={})
    d = client.post(f"/api/decode/{sid}", json={}).json()
    assert d["conv"]["channel_ber"] == 0.0 and d["frame"]["found"] and d["rs"]["found"]
    assert d["ground_truth_check"]["payload_byte_accuracy"] == 1.0

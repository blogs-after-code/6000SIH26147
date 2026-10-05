"""
Deep-learning modulation classifier (second opinion), trained on a PUBLIC dataset (RadioML) on Kaggle.

The network is trained offline (kaggle/train_modulation_cnn.ipynb) and shipped as an ONNX file, so the app needs
only `onnxruntime` (no PyTorch). Files expected in backend/models/:
    radioml_cnn.onnx   input  "iq"     float32 (batch, 2, L)   channel 0 = I, channel 1 = Q, every frame scaled to unit RMS
                       output "logits" float32 (batch, n_classes)
    radioml_cnn.json   labels, input_len L, samples-per-symbol, accuracy by SNR, mapping to this app's labels

Pre-processing contract (must match training): our channelised capture is resampled to the model's samples-per-symbol
(RadioML: about 8) using the symbol rate already estimated by the pipeline, cut into frames of L samples, each frame
normalised to unit RMS. Softmax probabilities are averaged over up to 24 frames spread across the capture.
"""
import json
import os
import numpy as np

from . import demod as D

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
ONNX_NAME, META_NAME = "radioml_cnn.onnx", "radioml_cnn.json"
TRANSFER_NAME = "cnn_transfer.json"
TRANSFER_MIN = 0.90
APP_LABELS = ("BPSK", "QPSK", "8PSK", "16QAM", "2FSK", "4FSK")


def normalise_label(name: str) -> str:
    """RadioML names -> this app's names where an equivalent exists; otherwise return the cleaned name."""
    n = "".join(ch for ch in str(name).upper() if ch.isalnum())
    fixed = {"QAM16": "16QAM", "QAM64": "64QAM", "QAM32": "32QAM", "QAM128": "128QAM", "QAM256": "256QAM",
             "PSK8": "8PSK", "PSK16": "16PSK", "PSK32": "32PSK"}
    return fixed.get(n, n)


def to_app_label(name: str):
    """Model label -> one of APP_LABELS (or 'FSK' for the FSK family), or None if the app cannot demodulate it."""
    n = normalise_label(name)
    if n in ("CPFSK", "GFSK", "FSK", "2FSK", "4FSK"):
        return "FSK"
    return n if n in APP_LABELS else None


class DeepClassifier:
    def __init__(self, directory=MODELS_DIR):
        self.dir = directory
        self.meta = None
        self.session = None
        self.reason = None
        self._load()

    def _load(self):
        onnx_path, meta_path = (os.path.join(self.dir, ONNX_NAME), os.path.join(self.dir, META_NAME))
        if not (os.path.exists(onnx_path) and os.path.exists(meta_path)):
            self.reason = f"model files not found (expected backend/models/{ONNX_NAME} and {META_NAME})"
            return
        try:
            import onnxruntime as ort
        except ImportError:
            self.reason = "onnxruntime is not installed (pip install onnxruntime)"
            return
        try:
            self.meta = json.load(open(meta_path))
            so = ort.SessionOptions()
            so.intra_op_num_threads = 2
            self.session = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
            self.in_name = self.session.get_inputs()[0].name
            self.L = int(self.meta["input_len"])
            self.sps = float(self.meta.get("sps", 8))
            self.labels = list(self.meta["labels"])
            self.noise_snr = self.meta.get("input_noise_snr_db")        # None = feed the capture as is
            self.labels_verified = bool(self.meta.get("labels_verified", True))   # False = order inferred, not confirmed
            self.transfer = self._load_transfer()
        except Exception as e:                                   # corrupt file, wrong format, ...
            self.session, self.reason = None, f"could not load model: {e}"

    def _load_transfer(self):
        """Measured agreement on THIS project's simulator (scripts/evaluate_pipeline.py). None until measured."""
        p = os.path.join(self.dir, TRANSFER_NAME)
        try:
            return json.load(open(p))
        except Exception:
            return None

    @property
    def warnings_trusted(self):
        """CNN disagreement may raise warnings only when its labels are confirmed AND it was measured to agree with
        ground truth on our own simulator at least TRANSFER_MIN of the time. Otherwise it is informational only."""
        t = self.transfer or {}
        return bool(self.labels_verified and t.get("agreement", 0.0) >= TRANSFER_MIN)

    @property
    def available(self):
        return self.session is not None

    def agrees(self, label, chosen):
        a = to_app_label(label)
        if a is None:
            return None                                        # outside the app's supported set
        if a == "FSK":
            return chosen in ("2FSK", "4FSK")
        return a == chosen

    def predict(self, xb, fs, symbol_rate, max_frames=24):
        """xb: channelised complex capture. Returns dict with top-3 and diagnostics, or raises ValueError."""
        y, _ = D.resample_to_sps(xb, fs, symbol_rate, sps=int(round(self.sps)))
        L = self.L
        if len(y) < L:
            raise ValueError("capture too short for the CNN input length")
        n = int(min(max_frames, len(y) // L))
        starts = np.linspace(0, len(y) - L, n).astype(int)
        frames = np.stack([y[s:s + L] for s in starts])
        frames = frames / (np.sqrt(np.mean(np.abs(frames) ** 2, axis=1, keepdims=True)) + 1e-9)
        if self.noise_snr is not None:
            # RadioML frames always carry wideband noise; our channelised capture has none outside the signal band, which is
            # out of distribution for the network. Add seeded white noise at the configured SNR (documented in the JSON).
            rng = np.random.default_rng(0)
            noise = (rng.standard_normal(frames.shape) + 1j * rng.standard_normal(frames.shape)) * np.sqrt(10 ** (-self.noise_snr / 10) / 2)
            frames = frames + noise
            frames = frames / (np.sqrt(np.mean(np.abs(frames) ** 2, axis=1, keepdims=True)) + 1e-9)
        X = np.stack([frames.real, frames.imag], axis=1).astype(np.float32)
        logits = self.session.run(None, {self.in_name: X})[0]
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        p = e / e.sum(axis=1, keepdims=True)
        mean = p.mean(axis=0)
        order = np.argsort(-mean)[:3]
        votes = np.bincount(p.argmax(axis=1), minlength=len(self.labels)) / n
        top = [dict(label=self.labels[i], app_label=to_app_label(self.labels[i]), prob=round(float(mean[i]), 4),
                    vote_fraction=round(float(votes[i]), 3)) for i in order]
        return dict(available=True, top=top, n_frames=n, frame_len=L, dataset=self.meta.get("dataset"),
                    supported=top[0]["app_label"] is not None, labels_verified=self.labels_verified,
                    warnings_trusted=self.warnings_trusted,
                    transfer_agreement=(self.transfer or {}).get("agreement"), input_noise_snr_db=self.noise_snr)


_instance = None


def get_deep():
    global _instance
    if _instance is None:
        _instance = DeepClassifier()
    return _instance


def reset_deep():
    global _instance
    _instance = None

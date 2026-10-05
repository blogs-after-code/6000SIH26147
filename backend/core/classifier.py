"""
Modulation classifier for linear modulations (BPSK, QPSK, 8PSK, 16QAM).

WHY features + Random Forest (not a deep net): the features are classical,
explainable signal-processing quantities, the model trains in seconds on a
laptop, and every decision can be defended in terms of the feature values.

FEATURES (computed on timing-recovered symbols y, BEFORE carrier recovery, so
they must not depend on the carrier phase or frequency):
  pmr2, pmr4, pmr8 : strength of the spectral line of y^2, y^4, y^8 (dB).
                     BPSK -> lines at 2 and 4; QPSK -> 4; 8PSK -> 8; 16QAM -> weak 4.
  m4, m6           : E|y|^4 and E|y|^6 of the unit-power symbols (amplitude spread).
                     PSK has constant modulus (m4 ~ 1); 16QAM has three rings (m4 ~ 1.32).
  cv               : coefficient of variation of |y|.
"""
import os
import numpy as np

from . import demod as D

LABELS = ["BPSK", "QPSK", "8PSK", "16QAM"]
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "mod_rf.joblib")
FEATURE_NAMES = ["pmr2_dB", "pmr4_dB", "pmr8_dB", "m4", "m6", "cv"]


def features(y):
    y = np.asarray(y, np.complex128)
    y = y / np.sqrt(np.mean(np.abs(y) ** 2) + 1e-30)
    lf = D.line_features(y)
    a = np.abs(y)
    return np.array([10 * np.log10(lf[2][1]), 10 * np.log10(lf[4][1]), 10 * np.log10(lf[8][1]),
                     np.mean(a ** 4), np.mean(a ** 6), a.std() / (a.mean() + 1e-12)])


class ModClassifier:
    def __init__(self, path=MODEL_PATH):
        import joblib
        self.path = path
        self.model = joblib.load(path) if os.path.exists(path) else None

    @property
    def available(self):
        return self.model is not None

    def predict(self, y):
        if not self.available:
            raise RuntimeError("Classifier model missing - run: python scripts/train_classifier.py")
        f = features(y).reshape(1, -1)
        p = self.model.predict_proba(f)[0]
        classes = list(self.model.classes_)
        probs = {c: float(pr) for c, pr in zip(classes, p)}
        best = max(probs, key=probs.get)
        return best, probs, dict(zip(FEATURE_NAMES, [round(float(v), 3) for v in f[0]]))

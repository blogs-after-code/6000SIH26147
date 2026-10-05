"""Train the modulation classifier (Random Forest) on synthetic signals.

Run from project root:   python scripts/train_classifier.py
Takes a few minutes on a laptop (no GPU needed). Writes:
  backend/models/mod_rf.joblib      the model
  backend/models/eval_report.json   hold-out accuracy per SNR + confusion matrix
Options:  --n 300 (training signals per class)   --test 80 (hold-out per class)
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.core.generator import gen_signal
from backend.core.spectral import analyze
from backend.core.pipeline import extract_features_from_iq
from backend.core.classifier import LABELS, FEATURE_NAMES, MODEL_PATH

SPS_CHOICES = [4, 5, 8, 10, 16]


def make_one(mod, seed):
    rng = np.random.default_rng(seed)
    sps = int(rng.choice(SPS_CHOICES))
    fs = float(rng.choice([48e3, 96e3, 100e3, 200e3]))
    Rs = fs / sps
    esn0 = float(rng.uniform(6, 26))
    beta = float(rng.uniform(0.2, 0.5))
    cfo = float(rng.uniform(-0.25, 0.25) * fs)
    x, _ = gen_signal(mod, fs=fs, symbol_rate=Rs, n_symbols=2000, snr_db=esn0 - 10 * np.log10(sps),
                      cfo_hz=cfo, beta=beta, seed=int(rng.integers(1 << 30)))
    spec = analyze(x, fs)["params"]
    return extract_features_from_iq(x, fs, spec), esn0


def build(n, seed0):
    X, y, e = [], [], []
    t0 = time.time()
    for li, lab in enumerate(LABELS):
        for i in range(n):
            try:
                f, esn0 = make_one(lab, seed0 + li * 100000 + i)
            except Exception:
                continue
            X.append(f); y.append(lab); e.append(esn0)
        print(f"  {lab}: done ({time.time() - t0:.0f}s)", flush=True)
    return np.array(X), np.array(y), np.array(e)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--test", type=int, default=80)
    a = ap.parse_args()
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import confusion_matrix
    import joblib
    print("Generating training data..."); Xtr, ytr, _ = build(a.n, 1)
    print("Generating hold-out data (different seeds)..."); Xte, yte, ete = build(a.test, 900000)
    clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=0, n_jobs=-1)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    acc = float(np.mean(pred == yte))
    bins = [(6, 10), (10, 14), (14, 18), (18, 22), (22, 26)]
    by_snr = []
    for lo, hi in bins:
        m = (ete >= lo) & (ete < hi)
        by_snr.append(dict(esn0_range=[lo, hi], n=int(m.sum()), accuracy=round(float(np.mean(pred[m] == yte[m])), 4) if m.any() else None))
    cm = confusion_matrix(yte, pred, labels=LABELS).tolist()
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    rep = dict(labels=LABELS, features=FEATURE_NAMES, n_train=int(len(ytr)), n_test=int(len(yte)),
               accuracy=round(acc, 4), accuracy_by_esn0_db=by_snr, confusion_matrix=cm,
               feature_importance=dict(zip(FEATURE_NAMES, np.round(clf.feature_importances_, 3).tolist())),
               note="Hold-out set = synthetic signals with unseen seeds/parameters. Real-world accuracy will be lower.")
    json.dump(rep, open(os.path.join(os.path.dirname(MODEL_PATH), "eval_report.json"), "w"), indent=2)
    print(json.dumps(rep, indent=2))

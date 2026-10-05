"""
Confirm the class order of GOLD_XYZ_OSC.0001_1024.hdf5 from the DATA itself (about 1 minute, no GPU needed).

Paste into a Kaggle notebook cell that has the radioml2018 dataset attached (Add Input), run it, and send me the printed
VERDICT line. It does not train anything.

Idea: constant-envelope signals (FM, GMSK) have an almost flat |x|, on-off keying (OOK) has the most variable |x|.
We measure the envelope variation of the frames behind every one-hot column (high SNR only) and see which columns
are the flattest and which is the most variable. That tells us which name list belongs to the file.
"""
import glob
import os
import sys

import numpy as np

FIXED = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK', '16APSK', '32APSK', '64APSK', '128APSK',
         '16QAM', '32QAM', '64QAM', '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC', 'FM',
         'GMSK', 'OQPSK']
CLASSES_TXT = ['32PSK', '16APSK', '32QAM', 'FM', 'GMSK', '32APSK', 'OQPSK', '8ASK', 'BPSK', '8PSK', 'AM-SSB-SC',
               '4ASK', '16PSK', '64APSK', '128QAM', '128APSK', 'AM-DSB-SC', 'AM-SSB-WC', '64QAM', 'QPSK', '256QAM',
               'AM-DSB-WC', 'OOK', '16QAM']
CANDIDATES = {"fixed": FIXED, "classes_txt": CLASSES_TXT}


def envelope_cv_by_column(X, Y, Z, per_class=150, min_snr=20):
    """Mean (std/mean) of |x| per one-hot column, using only frames with SNR >= min_snr."""
    cls = Y.argmax(axis=1)
    cv = np.full(Y.shape[1], np.nan)
    for c in range(Y.shape[1]):
        idx = np.where((cls == c) & (Z >= min_snr))[0][:per_class]
        if len(idx) == 0:
            continue
        a = np.hypot(X[idx, :, 0].astype(np.float32), X[idx, :, 1].astype(np.float32))
        cv[c] = float(np.mean(a.std(axis=1) / (a.mean(axis=1) + 1e-9)))
    return cv


def verdict(cv):
    """Return (name_of_matching_order or None, details). Uses only names that must be extreme in envelope variation."""
    order = np.argsort(cv)
    flat5, var1 = set(order[:5].tolist()), int(order[-1])   # AM-SSB with carrier is also flat, so allow 5
    good = []
    for name, labels in CANDIDATES.items():
        fm, gmsk, ook = labels.index("FM"), labels.index("GMSK"), labels.index("OOK")
        if fm in flat5 and gmsk in flat5 and var1 == ook:
            good.append(name)
    return (good[0] if len(good) == 1 else None), dict(flattest5=order[:5].tolist(), most_variable=var1)


def load_and_measure(path):
    import h5py
    with h5py.File(path, "r") as f:
        Y, Z = f["Y"][:], f["Z"][:].reshape(-1).astype(int)
        cls = Y.argmax(axis=1)
        take = np.concatenate([np.where((cls == c) & (Z >= 20))[0][:150] for c in range(Y.shape[1])])
        take.sort()
        X = f["X"][take.tolist()]
    return envelope_cv_by_column(X, Y[take], Z[take])


if __name__ == "__main__":
    hits = glob.glob("/kaggle/input/**/GOLD_XYZ_*.hdf5", recursive=True) if len(sys.argv) < 2 else [sys.argv[1]]
    if not hits:
        sys.exit("No GOLD_XYZ_*.hdf5 found. Add the radioml2018 dataset under Add Input.")
    cv = load_and_measure(hits[0])
    print("column  envelope-variation (low = constant envelope)   name if 'fixed' / if 'classes_txt'")
    for i in np.argsort(cv):
        print(f"{i:5d}   {cv[i]:.3f}   {FIXED[i]:10s} / {CLASSES_TXT[i]}")
    name, info = verdict(cv)
    print("\nVERDICT:", name if name else "UNDECIDED (neither list fits; send me the table above)", info)

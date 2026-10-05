"""
Re-label a trained RadioML CNN's metadata JSON by output index, without retraining.

Why this exists: the first Kaggle run parsed `classes.txt` line by line, so two labels carried junk
("classes = ['32PSK'," and "'16QAM']") and, more importantly, the name order looked like the classes.txt order
rather than the order of the one-hot column in the HDF5 file. The network weights are fine: only the NAME
attached to each output index is in question.

    python scripts/fix_cnn_labels.py IN.json OUT.json --order fixed
    python scripts/fix_cnn_labels.py IN.json OUT.json --order classes_txt      # same names, cleaned of junk

Use `--verified` ONLY after kaggle/verify_class_order.py has printed a verdict that matches --order.
Every list that depends on names (labels, class_accuracy_ge10dB, top_confusions_ge10dB, app_map) is rebuilt by index.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.core.dl_classifier import to_app_label  # noqa: E402

# Order of the one-hot column Y in GOLD_XYZ_OSC.0001_1024.hdf5 (RadioML 2018.01A paper table order).
FIXED = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK', '16APSK', '32APSK', '64APSK', '128APSK',
         '16QAM', '32QAM', '64QAM', '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC', 'FM',
         'GMSK', 'OQPSK']
# Order of the shipped classes.txt.
CLASSES_TXT = ['32PSK', '16APSK', '32QAM', 'FM', 'GMSK', '32APSK', 'OQPSK', '8ASK', 'BPSK', '8PSK', 'AM-SSB-SC',
               '4ASK', '16PSK', '64APSK', '128QAM', '128APSK', 'AM-DSB-SC', 'AM-SSB-WC', '64QAM', 'QPSK', '256QAM',
               'AM-DSB-WC', 'OOK', '16QAM']
ORDERS = {"fixed": FIXED, "classes_txt": CLASSES_TXT}


def relabel(meta: dict, new: list, order_name: str, verified: bool) -> dict:
    old = list(meta["labels"])
    assert len(old) == len(new) == 24, "expected 24 classes"
    idx_of_old = {n: i for i, n in enumerate(old)}
    acc_by_idx = list(meta["class_accuracy_ge10dB"].values())            # dict preserves index order
    assert len(acc_by_idx) == 24
    out = dict(meta)
    out["labels"] = list(new)
    out["class_accuracy_ge10dB"] = {new[i]: acc_by_idx[i] for i in range(24)}
    out["top_confusions_ge10dB"] = [[c, new[idx_of_old[a]], new[idx_of_old[b]]]
                                    for c, a, b in meta["top_confusions_ge10dB"]]
    out["app_map"] = {n: to_app_label(n) for n in new}
    out["label_order"] = order_name
    out["labels_verified"] = bool(verified)
    out["label_note"] = ("Labels were re-attached by output index after training (weights unchanged). "
                         + ("Order confirmed against the dataset with kaggle/verify_class_order.py."
                            if verified else
                            "Order INFERRED from the confusion pattern; not yet confirmed against the dataset."))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src"), ap.add_argument("dst")
    ap.add_argument("--order", choices=list(ORDERS), required=True)
    ap.add_argument("--verified", action="store_true")
    a = ap.parse_args()
    meta = json.load(open(a.src))
    json.dump(relabel(meta, ORDERS[a.order], a.order, a.verified), open(a.dst, "w"), indent=2)
    print("wrote", a.dst, "| order:", a.order, "| verified:", a.verified)

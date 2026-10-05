"""
Accuracy graphs for the PPT and the defence (backlog item 3).  Everything here is measured on THIS PROJECT'S OWN
SIMULATOR (backend/core/generator.py and chain.py), so it is an upper bound for real captures. Say so when you quote it.

    python scripts/evaluate_pipeline.py                 # full run, about 10 to 15 minutes
    python scripts/evaluate_pipeline.py --quick         # about 2 minutes, coarse grid (use to check it works)
    python scripts/evaluate_pipeline.py --only demod    # or: --only decode

Writes results/pipeline_eval.json and results/*.png.  Also writes backend/models/cnn_transfer.json (how often the
RadioML CNN agrees with ground truth on our simulator), which the app uses to decide whether CNN warnings are trusted.

Experiments
  demod : for each modulation and full-band SNR, N random captures (random seed, carrier offset up to +/-8 kHz).
          Reports modulation-ID accuracy of the app, bit error rate of the demodulated bits (after the app's own
          ambiguity handling, scored against the transmitted bits), and CNN agreement for the four linear types.
  decode: bit-level channel with a given raw bit error rate (random flips) fed into the blind decoder, for a
          convolutional code, an RS code, the RS + convolutional concatenation, an LDPC code and RS + LDPC. Success = every payload byte right.
"""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from backend.core import decode as DEC                      # noqa: E402
from backend.core import demod as D                         # noqa: E402
from backend.core.chain import build_chain                  # noqa: E402
from backend.core.dl_classifier import get_deep, TRANSFER_NAME, MODELS_DIR   # noqa: E402
from backend.core.evaluate import ber_aligned               # noqa: E402
from backend.core.generator import gen_signal               # noqa: E402
from backend.core.pipeline import demodulate                # noqa: E402
from backend.core.spectral import analyze                   # noqa: E402

FS, RS = 100e3, 12.5e3
LINEAR = ("BPSK", "QPSK", "8PSK", "16QAM")
MODS = LINEAR + ("2FSK", "4FSK")
OUT = os.path.join(ROOT, "results")


def one_capture(mod, snr, seed):
    rng = np.random.default_rng(1000 + seed)
    cfo = float(rng.uniform(-8e3, 8e3))
    x, t = gen_signal(mod, fs=FS, symbol_rate=RS, snr_db=snr, cfo_hz=cfo, seed=seed, n_symbols=4000)
    out = dict(id_ok=False, ber=0.5, cnn_ok=None)
    try:
        spec = analyze(x, FS)["params"]
        res, bits, internal = demodulate(x, FS, spec)
    except Exception:
        return out                                                   # failure to demodulate counts as a miss
    out["id_ok"] = res["modulation"] == mod
    d = res.get("deep") or {}
    if mod in LINEAR and d.get("top"):
        out["cnn_ok"] = d["top"][0]["app_label"] == mod
    if out["id_ok"]:
        k, best = internal["k"], 0.5
        for r in range(internal["M"]):
            b = D.demap_linear(internal["syms"] * np.exp(2j * np.pi * r / internal["M"]), internal["mod"])[0] \
                if internal["kind"] == "linear" else bits
            best = min(best, ber_aligned(t["bits"], b, k)[0])
        out["ber"] = float(best)
    return out


def run_demod(snrs, n):
    rows = []
    for mod in MODS:
        for snr in snrs:
            rs = [one_capture(mod, snr, s) for s in range(n)]
            ok_ber = [r["ber"] for r in rs if r["id_ok"]]
            cnn = [r["cnn_ok"] for r in rs if r["cnn_ok"] is not None]
            rows.append(dict(mod=mod, snr_fullband_db=snr, n=n, id_accuracy=float(np.mean([r["id_ok"] for r in rs])),
                             ber_median=float(np.median(ok_ber)) if ok_ber else None,
                             cnn_agreement=float(np.mean(cnn)) if cnn else None))
            print(f"  {mod:5s} {snr:>3} dB  id={rows[-1]['id_accuracy']:.2f}  ber={rows[-1]['ber_median']}  cnn={rows[-1]['cnn_agreement']}", flush=True)
    return rows


CHAINS = {"Convolutional K=7 (171/133)": dict(conv="conv_k7", n_frames=60),
          "Reed-Solomon (255,223)": dict(fec="rs_255_223", n_frames=14),
          "RS (204,188) + conv K=9": dict(fec="rs_204_188", conv="conv_k9", n_frames=30),
          "LDPC (512,256)": dict(ldpc="ldpc_512_256", n_frames=16),
          "RS (204,188) + LDPC (512,256)": dict(fec="rs_204_188", ldpc="ldpc_512_256", n_frames=14)}


def run_decode(bers, n):
    rows = []
    for name, kw in CHAINS.items():
        for ber in bers[name]:
            ok = []
            for s in range(n):
                bits, truth = build_chain(seed=1 + s, **kw)
                rx = bits[int(np.random.default_rng(s).integers(0, 40)):].copy()
                rx[np.random.default_rng(50 + s).random(len(rx)) < ber] ^= 1
                try:
                    r = DEC.decode(dict(kind="fsk", idx=rx.astype(int), M=2, k=1), truth=truth)
                    ok.append(r["ground_truth_check"]["payload_byte_accuracy"] == 1.0)
                except Exception:
                    ok.append(False)
            rows.append(dict(chain=name, channel_ber=ber, n=n, success=float(np.mean(ok))))
            print(f"  {name:30s} ber={ber:<7} success={rows[-1]['success']:.2f}", flush=True)
    return rows


def plots(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(OUT, exist_ok=True)
    if res.get("demod"):
        rows = res["demod"]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for mod in MODS:
            r = [x for x in rows if x["mod"] == mod]
            ax[0].plot([x["snr_fullband_db"] for x in r], [x["id_accuracy"] for x in r], "o-", label=mod)
            rb = [x for x in r if x["ber_median"] is not None and x["ber_median"] > 0]
            ax[1].semilogy([x["snr_fullband_db"] for x in rb], [x["ber_median"] for x in rb], "o-", label=mod)
        ax[0].set(xlabel="full-band SNR (dB)", ylabel="modulation identified correctly", title="Modulation ID vs SNR", ylim=(-.02, 1.02))
        ax[1].set(xlabel="full-band SNR (dB)", ylabel="median bit error rate", title="Demodulated BER vs SNR (correctly identified captures)")
        ax[1].set_ylim(1e-4, 1)
        for a in ax: a.grid(alpha=.3); a.legend(fontsize=8)
        fig.suptitle("Own simulator, not real captures", fontsize=9, y=0.02)
        fig.tight_layout(rect=(0, .04, 1, 1)); fig.savefig(os.path.join(OUT, "demod_vs_snr.png"), dpi=150); plt.close(fig)
    if res.get("decode"):
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for name in CHAINS:
            r = [x for x in res["decode"] if x["chain"] == name]
            ax.plot([x["channel_ber"] for x in r], [x["success"] for x in r], "o-", label=name)
        ax.set(xlabel="raw channel bit error rate (random flips)", ylabel="message fully recovered", title="Blind decode success vs channel BER", ylim=(-.02, 1.02))
        ax.grid(alpha=.3); ax.legend(fontsize=8)
        fig.text(0.5, 0.01, "Own simulator and own transmit chain, not real captures", ha="center", fontsize=8)
        fig.tight_layout(rect=(0, .04, 1, 1)); fig.savefig(os.path.join(OUT, "decode_vs_ber.png"), dpi=150); plt.close(fig)


def write_cnn_transfer(rows, snr_min=8):
    deep = get_deep()
    if not deep.available:
        return
    vals = [(r["cnn_agreement"], r["n"]) for r in rows if r["mod"] in LINEAR and r["snr_fullband_db"] >= snr_min and r["cnn_agreement"] is not None]
    if not vals:
        return
    agreement = float(sum(a * n for a, n in vals) / sum(n for _, n in vals))
    info = dict(agreement=round(agreement, 4), n_captures=int(sum(n for _, n in vals)), modulations=list(LINEAR),
                min_fullband_snr_db=snr_min, source="scripts/evaluate_pipeline.py on backend/core/generator.py signals",
                model_sha256=deep.meta.get("sha256"), labels_verified=deep.labels_verified,
                created=time.strftime("%Y-%m-%dT%H:%M:%S"))
    json.dump(info, open(os.path.join(MODELS_DIR, TRANSFER_NAME), "w"), indent=2)
    print("CNN agreement with ground truth on our simulator:", info["agreement"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=["demod", "decode"])
    ap.add_argument("--n", type=int, help="captures per (modulation, SNR) cell for demod (default 8, quick 3)")
    a = ap.parse_args()
    snrs = [0, 6, 12, 20] if a.quick else [-2, 0, 2, 4, 6, 8, 12, 16, 20]
    n = a.n or (3 if a.quick else 8)
    bers = ({k: [0.0, 0.02, 0.06] for k in CHAINS} if a.quick else
            {"Convolutional K=7 (171/133)": [0, .01, .02, .03, .05, .08],
             "Reed-Solomon (255,223)": [0, .001, .002, .003, .005, .01],
             "RS (204,188) + conv K=9": [0, .01, .02, .03, .05, .08],
             "LDPC (512,256)": [0, .01, .02, .03, .04, .05, .06],
             "RS (204,188) + LDPC (512,256)": [0, .01, .02, .03, .04, .05, .06]})
    res = {"note": "Measured on this project's own simulator; an upper bound for real captures.",
           "date": time.strftime("%Y-%m-%d"), "quick": a.quick}
    prev = os.path.join(OUT, "pipeline_eval.json")
    if a.only and not a.quick and os.path.exists(prev):          # keep the other experiment's earlier results
        old = json.load(open(prev))
        res.update({k: old[k] for k in ("demod", "decode") if k in old})
    if a.only in (None, "demod"):
        print("demod ..."); res["demod"] = run_demod(snrs, n); res["demod_n_per_cell"] = n
        if not a.quick:
            write_cnn_transfer(res["demod"])
    if a.only in (None, "decode"):
        print("decode ..."); res["decode"] = run_decode(bers, 3 if a.quick else 6)
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "pipeline_eval" + ("_quick" if a.quick else "") + ".json"), "w"), indent=1)
    plots(res)
    print("wrote", OUT)

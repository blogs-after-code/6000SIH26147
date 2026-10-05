# %% [markdown]
# # SanketSetu: train the modulation CNN on RadioML (public dataset) on Kaggle
#
# **What this notebook does:** trains a small 1-D ResNet on the public RadioML dataset (2018.01A or 2016.10a),
# measures accuracy by SNR on frames it has never seen, and exports `radioml_cnn.onnx` + `radioml_cnn.json`.
# Copy those two files into `signal-lab/backend/models/` and the app shows the CNN as a second opinion next to the
# feature-based classifier.
#
# **Before you run (one time):**
# 1. Add the dataset: *Add Input* -> search `radioml2018` (file `GOLD_XYZ_OSC.0001_1024.hdf5`, about 20 GB) **or**
#    `radioml2016` (file `RML2016.10a_dict.pkl`, about 600 MB). The 2016 set is faster to try first.
# 2. Settings -> Accelerator -> **GPU T4 x2 / P100**.
# 3. Settings -> Internet -> **On** (only needed to pip-install onnxruntime for the final check).
# 4. Run all cells. Set `SMOKE = True` first to check everything works in a couple of minutes.
#
# **Honest scope:** RadioML is simulated (GNU Radio channel model), so accuracy on it is an upper bound for real
# captures. The CNN is a *second opinion*: if it disagrees with the main pipeline the app warns the operator.

# %% [code]
# ----------------------------------------------------------------------------- CONFIG
SMOKE = False            # True = tiny quick run (1 epoch, few frames) to verify the notebook works end to end
DATASET = "auto"         # "auto" | "2016" | "2018"
N_PER_CELL_2018 = 800    # frames kept per (modulation, SNR) cell from the 2.5M-frame 2018 file (800 -> ~500k frames)
EPOCHS = None            # None -> 25 for 2018, 40 for 2016 (1 in smoke mode)
BATCH = 512
WIDTH = 64               # network width (64 -> about 1.2 M parameters)
SPS = 8                  # nominal samples per symbol of RadioML frames (the app resamples captures to this)
SEED = 0
INPUT_ROOT = "/kaggle/input"
OUT_DIR = "/kaggle/working/sanketsetu_model"

# %% [code]
# ----------------------------------------------------------------------------- LOCATE THE DATASET
import os, re, glob, json, ast, pickle, time, math, hashlib, datetime, shutil
import numpy as np

rng = np.random.default_rng(SEED)
os.makedirs(OUT_DIR, exist_ok=True)


def find_dataset(root, want):
    hits = {"2018": [], "2016": []}
    for dp, _, files in os.walk(root):
        for fn in files:
            low = fn.lower()
            if low.endswith((".hdf5", ".h5")) and "gold_xyz" in low:
                hits["2018"].append(os.path.join(dp, fn))
            if low.endswith((".pkl", ".dat")) and ("rml2016" in low or "radioml" in low):
                hits["2016"].append(os.path.join(dp, fn))
    order = [want] if want in ("2016", "2018") else ["2018", "2016"]
    for k in order:
        if hits[k]:
            return k, sorted(hits[k])[0]
    raise FileNotFoundError(
        f"No RadioML file found under {root}. Use 'Add Input' and search for 'radioml2018' or 'radioml2016'. "
        f"Looked for GOLD_XYZ_*.hdf5 (2018) or RML2016.10a_dict.pkl (2016).")


KIND, PATH = find_dataset(INPUT_ROOT, DATASET)
print(f"Dataset: RadioML {KIND}  ->  {PATH}")

# %% [code]
# ----------------------------------------------------------------------------- LOAD (numpy / h5py only)
# Two published orders exist. The one-hot column of the HDF5 file follows the paper's table ("fixed"); the shipped
# classes.txt lists the same names in a different order. We do NOT trust either: the data decides (see choose_names).
CLASSES_FIXED = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK', '16APSK', '32APSK', '64APSK',
                 '128APSK', '16QAM', '32QAM', '64QAM', '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC',
                 'AM-DSB-SC', 'FM', 'GMSK', 'OQPSK']
CLASSES_TXT = ['32PSK', '16APSK', '32QAM', 'FM', 'GMSK', '32APSK', 'OQPSK', '8ASK', 'BPSK', '8PSK', 'AM-SSB-SC',
               '4ASK', '16PSK', '64APSK', '128QAM', '128APSK', 'AM-DSB-SC', 'AM-SSB-WC', '64QAM', 'QPSK', '256QAM',
               'AM-DSB-WC', 'OOK', '16QAM']


def candidate_orders(path):
    """Name lists to test: the two published orders plus any classes*.{txt,json} file next to the dataset.
    Files are parsed by pulling out quoted tokens, so 'classes = [...]' style files give clean names."""
    cands = {"fixed": CLASSES_FIXED, "classes_txt": CLASSES_TXT}
    for d in (os.path.dirname(path), os.path.dirname(os.path.dirname(path))):
        for p in sorted(glob.glob(os.path.join(d, "classes*"))):
            try:
                names = [a or b for a, b in re.findall(r"'([^']+)'|\"([^\"]+)\"", open(p).read())]
            except Exception:
                continue
            if len(names) == 24:
                print("Found classes file:", p)
                cands["file:" + os.path.basename(p)] = names
    return cands


def load_2016(path):
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="latin1")                     # {(modulation, snr): array (1000, 2, 128)}
    mods = sorted({k[0] for k in d})
    X, y, z = [], [], []
    for (m, s), arr in sorted(d.items()):
        if SMOKE:
            arr = arr[:40]
        X.append(np.asarray(arr, np.float32)); y += [mods.index(m)] * len(arr); z += [int(s)] * len(arr)
    return np.concatenate(X), np.array(y), np.array(z), mods         # X: (N, 2, 128)


def load_2018(path, n_per_cell, rng):
    import h5py
    with h5py.File(path, "r") as f:
        Y = f["Y"][:]
        Z = f["Z"][:].reshape(-1).astype(int)
        cls = Y.argmax(axis=1)
        take = []
        for c in range(Y.shape[1]):
            for s in np.unique(Z):
                idx = np.where((cls == c) & (Z == s))[0]
                if len(idx) == 0:
                    continue
                k = min(n_per_cell, len(idx))
                take.append(np.sort(rng.choice(idx, size=k, replace=False)))
        sel = np.sort(np.concatenate(take))                          # h5py needs increasing indices
        print(f"Reading {len(sel):,} of {len(cls):,} frames from the HDF5 file ...")
        X = np.empty((len(sel), 1024, 2), np.float16)
        for a in range(0, len(sel), 20000):                          # chunked reads keep memory flat
            ids = sel[a:a + 20000]
            X[a:a + len(ids)] = f["X"][ids.tolist()].astype(np.float16)
    return X.transpose(0, 2, 1), cls[sel], Z[sel], None             # X: (N, 2, 1024); names chosen below


if KIND == "2018":
    X, y, z, NAMES = load_2018(PATH, 20 if SMOKE else N_PER_CELL_2018, rng)
else:
    X, y, z, NAMES = load_2016(PATH)
N, _, L = X.shape
print(f"Loaded X={X.shape} dtype={X.dtype}  SNR range {z.min()}..{z.max()} dB")

# %% [code]
# ----------------------------------------------------------------------------- CHOOSE CLASS NAMES FROM THE DATA (2018)
# Constant-envelope signals (FM, GMSK) have the flattest |x|; OOK has the most variable |x|. A name list is accepted
# only if it passes this test. If none passes we STOP: wrong names silently ruin every number and plot downstream.
def envelope_cv(X, y, z):
    hi, cv = z >= 20, {}
    for c in range(int(y.max()) + 1):
        sel = np.where(hi & (y == c))[0][:200]
        if len(sel):
            a = np.sqrt(X[sel, 0].astype(np.float32) ** 2 + X[sel, 1].astype(np.float32) ** 2)
            cv[c] = float(np.mean(a.std(axis=1) / (a.mean(axis=1) + 1e-9)))
    return cv


def choose_names(X, y, z, candidates):
    cv = envelope_cv(X, y, z)
    order = sorted(cv, key=cv.get)
    flat3, most_variable = set(order[:5]), order[-1]       # AM-SSB with carrier is flat too
    print("Envelope variation, flattest first (column: value):", [f"{c}:{cv[c]:.2f}" for c in order])
    passing = []
    for tag, names in candidates.items():
        if all(n in names for n in ("FM", "GMSK", "OOK")) and names.index("FM") in flat3 \
                and names.index("GMSK") in flat3 and names.index("OOK") == most_variable:
            passing.append(tag)
    print("Name lists that fit the data:", passing or "NONE")
    assert passing, ("None of the candidate class-name orders fits the data (FM and GMSK must be among the 5 flattest, OOK the "
                     "most variable). Do not train. Send the table above to the project owner.")
    if len(passing) > 1:
        print("Several lists fit; using the first:", passing[0])
    return list(candidates[passing[0]]), passing[0]


if KIND == "2018":
    NAMES, NAMES_SOURCE = choose_names(X, y, z, candidate_orders(PATH))
    print("Using class names from:", NAMES_SOURCE)
else:
    NAMES_SOURCE = "dataset keys"
print("Classes:", NAMES)

# %% [code]
# ----------------------------------------------------------------------------- PREPARE: unit-RMS frames + stratified split
def unit_rms(X, chunk=50000):
    out = np.empty(X.shape, np.float16)
    for a in range(0, len(X), chunk):
        b = X[a:a + chunk].astype(np.float32)
        r = np.sqrt(np.mean(b[:, 0] ** 2 + b[:, 1] ** 2, axis=1, keepdims=True))[:, None, :] + 1e-9
        out[a:a + chunk] = (b / r).astype(np.float16)
    return out


X = unit_rms(X)
split = np.zeros(len(y), np.int8)                                 # 0 train, 1 val, 2 test
for c in np.unique(y):
    for s in np.unique(z):
        idx = np.where((y == c) & (z == s))[0]
        rng.shuffle(idx)
        n = len(idx)
        split[idx[int(0.8 * n):int(0.9 * n)]] = 1
        split[idx[int(0.9 * n):]] = 2
Xtr, ytr, ztr = X[split == 0], y[split == 0], z[split == 0]
Xva, yva, zva = X[split == 1], y[split == 1], z[split == 1]
Xte, yte, zte = X[split == 2], y[split == 2], z[split == 2]
print(f"train {len(ytr):,}  val {len(yva):,}  test {len(yte):,}   frame length {L}")

# %% [code]
# ----------------------------------------------------------------------------- MODEL
import torch, torch.nn as nn, torch.nn.functional as F

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "(no GPU: enable one under Settings -> Accelerator)")
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = True


class ResBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv1d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm1d(cout)
        self.c2 = nn.Conv1d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm1d(cout)
        self.sc = None
        if stride != 1 or cin != cout:
            self.sc = nn.Sequential(nn.Conv1d(cin, cout, 1, stride, bias=False), nn.BatchNorm1d(cout))

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(y + (x if self.sc is None else self.sc(x)))


class ModNet(nn.Module):
    """Input (batch, 2, L): I and Q rows. Works for any L (global average pooling)."""

    def __init__(self, n_classes, width=64):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(2, width, 7, 1, 3, bias=False), nn.BatchNorm1d(width), nn.ReLU())
        chans = [width, width, 2 * width, 2 * width, 4 * width]
        blocks, cin = [], width
        for i, c in enumerate(chans):
            blocks.append(ResBlock(cin, c, 1 if i == 0 else 2))
            cin = c
        self.body = nn.Sequential(*blocks)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(cin, n_classes)

    def forward(self, x):
        x = self.body(self.stem(x))
        return self.fc(self.drop(x.mean(dim=-1)))


model = ModNet(len(NAMES), WIDTH).to(dev)
n_params = sum(p.numel() for p in model.parameters())
print(f"ModNet: {n_params:,} parameters")

# %% [code]
# ----------------------------------------------------------------------------- TRAIN
EP = 1 if SMOKE else (EPOCHS or (25 if KIND == "2018" else 40))
use_amp = dev == "cuda"
Xt = torch.from_numpy(Xtr).to(dev)                                # float16 on the GPU: 2 GB for 500k x 2 x 1024
yt = torch.from_numpy(ytr).long().to(dev)
steps = math.ceil(len(yt) / BATCH)
opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=EP * steps, pct_start=0.2)
try:
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
except Exception:
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)


def augment(x):
    """Label-preserving: random carrier phase and a small random frequency offset per frame."""
    B, _, Ln = x.shape
    th = torch.rand(B, 1, device=x.device) * 2 * math.pi
    f = (torch.rand(B, 1, device=x.device) - 0.5) * 0.02
    t = torch.arange(Ln, device=x.device).view(1, Ln)
    ph = th + 2 * math.pi * f * t
    c, s = torch.cos(ph), torch.sin(ph)
    i, q = x[:, 0], x[:, 1]
    return torch.stack([i * c - q * s, i * s + q * c], dim=1)


@torch.no_grad()
def predict(Xnp, bs=2048):
    model.eval()
    out = []
    for a in range(0, len(Xnp), bs):
        xb = torch.from_numpy(Xnp[a:a + bs]).to(dev).float()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            out.append(model(xb).float().cpu())
    return torch.cat(out).argmax(dim=1).numpy()


best = dict(acc=-1.0, state=None, epoch=0)
t0 = time.time()
for ep in range(EP):
    model.train()
    perm = torch.randperm(len(yt), device=dev)
    run_loss = run_ok = seen = 0
    for k in range(steps):
        idx = perm[k * BATCH:(k + 1) * BATCH]
        xb = augment(Xt[idx].float())
        yb = yt[idx]
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(xb)
            loss = F.cross_entropy(logits, yb, label_smoothing=0.05)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        run_loss += loss.item() * len(idx)
        run_ok += (logits.argmax(1) == yb).sum().item()
        seen += len(idx)
    va = float(np.mean(predict(Xva) == yva))
    if va > best["acc"]:
        best = dict(acc=va, epoch=ep + 1, state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    print(f"epoch {ep + 1:02d}/{EP}  loss {run_loss / seen:.3f}  train acc {run_ok / seen:.3f}  "
          f"val acc {va:.3f}  best {best['acc']:.3f}  ({time.time() - t0:.0f}s)")
model.load_state_dict(best["state"])
print(f"Best validation accuracy {best['acc']:.4f} at epoch {best['epoch']}")

# %% [code]
# ----------------------------------------------------------------------------- EVALUATE ON THE HELD-OUT TEST SET
pred = predict(Xte)
test_acc = float(np.mean(pred == yte))
snrs = sorted(int(s) for s in np.unique(zte))
acc_by_snr = [dict(snr=s, acc=round(float(np.mean(pred[zte == s] == yte[zte == s])), 4), n=int((zte == s).sum())) for s in snrs]
ge10 = zte >= 10
class_acc_ge10 = {NAMES[c]: round(float(np.mean(pred[ge10 & (yte == c)] == c)), 4) for c in range(len(NAMES)) if (ge10 & (yte == c)).any()}
cm = np.zeros((len(NAMES), len(NAMES)), int)
for a, b in zip(yte[ge10], pred[ge10]):
    cm[a, b] += 1
print(f"Test accuracy (all SNR): {test_acc:.4f}   at >= 10 dB: {np.mean(pred[ge10] == yte[ge10]):.4f}")
for r in acc_by_snr:
    print(f"  SNR {r['snr']:>4} dB  acc {r['acc']:.3f}  (n={r['n']})")
confusions = sorted(((int(cm[a, b]), NAMES[a], NAMES[b]) for a in range(len(NAMES)) for b in range(len(NAMES)) if a != b), reverse=True)[:8]
print("Most common confusions at >= 10 dB (count, true, predicted):", confusions)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot([r["snr"] for r in acc_by_snr], [r["acc"] for r in acc_by_snr], marker="o")
ax.set_xlabel("SNR (dB)"); ax.set_ylabel("accuracy"); ax.set_ylim(0, 1.02); ax.grid(alpha=.3)
ax.set_title(f"RadioML {KIND}: test accuracy by SNR")
fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "accuracy_by_snr.png"), dpi=130); plt.show()
fig, ax = plt.subplots(figsize=(8, 7))
ax.imshow(cm / np.maximum(cm.sum(1, keepdims=True), 1), cmap="viridis")
ax.set_xticks(range(len(NAMES))); ax.set_xticklabels(NAMES, rotation=90, fontsize=7)
ax.set_yticks(range(len(NAMES))); ax.set_yticklabels(NAMES, fontsize=7)
ax.set_title("Confusion matrix, SNR >= 10 dB (rows = true class)")
fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "confusion_ge10.png"), dpi=130); plt.show()

# %% [code]
# ----------------------------------------------------------------------------- EXPORT: ONNX + metadata
def to_app_label(name):
    n = "".join(ch for ch in str(name).upper() if ch.isalnum())
    n = {"QAM16": "16QAM", "PSK8": "8PSK"}.get(n, n)
    if n in ("CPFSK", "GFSK", "FSK"):
        return "FSK"
    return n if n in ("BPSK", "QPSK", "8PSK", "16QAM") else None


model.eval().cpu()
onnx_path = os.path.join(OUT_DIR, "radioml_cnn.onnx")
dummy = torch.randn(1, 2, L)
export_kwargs = dict(input_names=["iq"], output_names=["logits"],
                     dynamic_axes={"iq": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17)
try:
    torch.onnx.export(model, dummy, onnx_path, dynamo=False, **export_kwargs)      # classic exporter (torch >= 2.5)
except TypeError:
    torch.onnx.export(model, dummy, onnx_path, **export_kwargs)
print("Wrote", onnx_path, f"({os.path.getsize(onnx_path) / 1e6:.1f} MB)")

# Verify the ONNX file reproduces the PyTorch output (this is the file the app will run)
try:
    try:
        import onnxruntime as ort
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "onnxruntime"])
        import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    probe = Xte[:64].astype(np.float32)
    with torch.no_grad():
        ref = model(torch.from_numpy(probe)).numpy()
    got = sess.run(None, {"iq": probe})[0]
    diff = float(np.abs(ref - got).max())
    print(f"ONNX vs PyTorch max |difference| = {diff:.2e}", "OK" if diff < 1e-3 else "TOO LARGE - do not use this file")
    onnx_ok = diff < 1e-3
except Exception as e:
    print("ONNX verification skipped:", e)
    onnx_ok = None

meta = dict(
    labels=[str(n) for n in NAMES], label_order=NAMES_SOURCE, labels_verified=(KIND == "2018"), input_len=int(L), sps=SPS, input_layout="(batch, 2, L): I then Q, unit RMS per frame",
    dataset=f"RadioML {KIND}" + (".01A" if KIND == "2018" else ".10a"), source="Kaggle",
    arch=f"1-D ResNet, width {WIDTH}, {n_params} parameters", epochs=EP, best_epoch=best["epoch"],
    best_val_acc=round(best["acc"], 4), test_accuracy=round(test_acc, 4), test_accuracy_ge10dB=round(float(np.mean(pred[ge10] == yte[ge10])), 4),
    accuracy_by_snr=acc_by_snr, class_accuracy_ge10dB=class_acc_ge10, top_confusions_ge10dB=confusions,
    app_map={str(n): to_app_label(n) for n in NAMES}, n_train=int(len(ytr)), n_test=int(len(yte)), seed=SEED,
    onnx_matches_torch=onnx_ok, created=datetime.datetime.utcnow().isoformat() + "Z",
    sha256=hashlib.sha256(open(onnx_path, "rb").read()).hexdigest(),
    note="Accuracy is on simulated RadioML frames never seen in training. Real captures will score lower.")
json.dump(meta, open(os.path.join(OUT_DIR, "radioml_cnn.json"), "w"), indent=2, default=str)
shutil.make_archive("/kaggle/working/sanketsetu_model", "zip", OUT_DIR)
print("\nDONE. Download /kaggle/working/sanketsetu_model.zip  (Output tab on the right -> the .zip file).")
print("Unzip it and copy radioml_cnn.onnx and radioml_cnn.json into signal-lab/backend/models/, then restart the app.")

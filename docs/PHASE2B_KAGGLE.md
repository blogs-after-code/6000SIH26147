# Phase 2b: train the modulation CNN on Kaggle (public data)

**Goal:** a second, independent modulation classifier trained on a public dataset (RadioML), shipped as a small ONNX file.
The app shows it next to the feature-based classifier and warns the operator when the two disagree.

**Status of this step (read first)**
- Trained on Kaggle (RadioML 2018.01A, 25 epochs): 60.5 % overall, 96.2 % at 10 dB and above. Files are in `backend/models/`.
- The first run attached class names in the wrong order (and two junk names). The weights are fine. Names were re-attached by
  index with `scripts/fix_cnn_labels.py --order fixed`. The order was then **confirmed from the dataset** by the developer's run of
  `kaggle/verify_class_order.py` (FM and GMSK among the five flattest envelopes, OOK the most variable), so `labels_verified` is true.
  No further action is needed. (If you ever train again, run the verifier on the new file and use `--verified` only when it agrees.)
- The notebook now picks the order from the data and stops if none fits (the old check silently skipped itself). Re-training is NOT needed.
- Domain gap: see `docs/PHASE4_EXPLAINED.md` section 4. The app adds seeded 5 dB noise before scoring (`input_noise_snr_db`) and treats the CNN as informational.

## Steps (about 20 minutes of your time, then Kaggle does the work)

1. Go to kaggle.com, create a notebook, then **File, Import notebook** and upload `kaggle/train_modulation_cnn.ipynb`.
2. **Add Input** (right panel), search `radioml2018` (file `GOLD_XYZ_OSC.0001_1024.hdf5`, about 20 GB, the better dataset)
   or `radioml2016` (file `RML2016.10a_dict.pkl`, about 600 MB, quick start). Several uploads exist; any that contains
   those file names works.
3. Right panel, **Session options**: Accelerator = **GPU T4 x2** (or P100), Internet = **On**.
4. In the first code cell set `SMOKE = True`. **Run all.** It should finish in a few minutes and end with
   "DONE". If a cell fails, copy the error text and send it to me (or another AI agent) together with
   `PROJECT_HANDOFF.md`.
5. Set `SMOKE = False`, **Restart and run all**. Typical time on a T4: about 10 to 25 minutes for 2018 (25 epochs on
   about 500 000 frames), less for 2016.
6. When it prints DONE, open the **Output** tab and download `sanketsetu_model.zip`. Unzip it.
7. Copy `radioml_cnn.onnx` and `radioml_cnn.json` into `signal-lab/backend/models/`.
8. `pip install onnxruntime` (already in requirements.txt), restart `uvicorn`. Open the Models tab: the CNN card now
   shows accuracy by SNR, and every analysis shows both opinions.

## What the notebook does

| Cell | Job |
|---|---|
| Config | switches: `SMOKE`, `DATASET`, frames per cell, epochs, width |
| Locate | finds the dataset under `/kaggle/input` and says which version it found |
| Load | 2018: reads a stratified subset (N frames per modulation and SNR) from the HDF5 file in chunks; 2016: loads the pickle |
| Sanity check | 2018 only: confirms the class-name order by checking that FM has the flattest envelope and OOK the most variable. Stops if the order looks wrong, because wrong labels would silently ruin the model |
| Prepare | unit-RMS normalisation per frame, stratified 80/10/10 split per (modulation, SNR) cell |
| Model | 1-D ResNet (about 1.2 M parameters), global average pooling so any frame length works |
| Train | AdamW, one-cycle learning rate, label smoothing, mixed precision, random phase and small frequency-offset augmentation |
| Evaluate | test accuracy overall, per SNR, per class at 10 dB and above, top confusions; saves two plots |
| Export | writes ONNX, checks it reproduces PyTorch output, writes metadata (labels, accuracy by SNR, hash, dataset, date) |

## How the app uses the model (contract)
- Input name `iq`, shape `(batch, 2, L)`: row 0 = I, row 1 = Q, every frame scaled to unit RMS. `L` comes from the JSON.
- The capture is channelised, resampled to the model's samples-per-symbol (8) using the symbol rate the pipeline
  already estimated, cut into up to 24 frames, and the softmax outputs are averaged.
- Output name `logits`, shape `(batch, n_classes)`. Labels come from the JSON; `app_map` shows which ones the app can
  demodulate (BPSK, QPSK, 8PSK, 16QAM). A confident prediction outside that set raises a warning
  ("a modulation this app cannot demodulate yet").
- The CNN never overrides the main result. It adds a warning when it disagrees with 60% or more confidence.

## Things to know before you quote a number
- RadioML is **simulated**. Accuracy on it is an upper bound for real captures. Say "on RadioML 2018.01A test frames".
- RadioML has **no FSK** in the 2018 set (the 2016 set has CPFSK and GFSK). The app's FSK path uses tone clustering anyway.
- The 2018 set is about 8 samples per symbol with a channel model (fading, offsets); our captures are resampled to match
  but there is a **domain gap**. Check the Models tab on your own test files: if the CNN disagrees with a signal you
  know is QPSK, that is a finding worth writing down, not something to hide.
- Class names for 2018 are read from the dataset's classes file when present; otherwise the published order is assumed
  and the sanity check tests that assumption.

## If something goes wrong
| Symptom | Likely cause and fix |
|---|---|
| `No RadioML file found` | dataset not added under Add Input, or a different file name: check `/kaggle/input` with `!ls -R /kaggle/input | head` |
| Out of memory while loading 2018 | lower `N_PER_CELL_2018` (800 to 400) |
| `Class-name sanity check FAILED` | class order differs from the assumption; read the dataset's own classes file and set `NAMES` by hand |
| `torch.onnx.export` error about `dynamo` | the cell already retries without that argument; if it still fails, send the full traceback |
| ONNX difference above 1e-3 | do not use the file; re-export, or send the output |
| App says "could not load model" | the file is damaged or was exported with an unsupported opset; re-download and check the file size |

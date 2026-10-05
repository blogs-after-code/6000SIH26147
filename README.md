# SanketSetu

Blind analysis of IQ and WAV radio captures: spectrum, modulation, demodulation, error-correcting codes, interleaving,
framing and message recovery, with the evidence shown at every step. Smart India Hackathon 2026 prototype.

New here (human or AI)? Read `PROJECT_HANDOFF.md`. Project status: Phases 1, 2, 3, 4 and 5 (LDPC) done; Phase 2b (Kaggle-trained CNN) done with caveats;
PPT and demo video are the operator's to finish. **Nothing has been validated on real captures yet.**

## Run it
```bash
cd signal-lab
python -m venv venv
# Windows: venv\Scripts\activate        Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
python -m pytest backend/tests -q           # expect: 80 passed (about 80 s)
python scripts/make_samples.py              # example captures + truth files in samples/
uvicorn backend.main:app --reload           # open http://127.0.0.1:8000
```

## Deploy (Render + Vercel)
See `DEPLOY.md`.

## Try it (5 minutes)
1. Open the app, click **Analyze a capture**.
2. Preset **Reed-Solomon + convolutional**, click **Generate and analyze**. The summary row fills in: QPSK, 12.5 kHz,
   Conv 1/2 K7 + RS(204,188), 1504-bit frames, message recovered. **Decoding** tab shows how.
3. Drag `samples/coded_qpsk_rs204_conv_k7.cf32` into the drop zone, sampling rate `100000`. Compare with the matching
   `.truth.json`.
4. Try preset "Convolutional + block interleaver": the interleaver type and size must be given in Decoding tab, then
   Re-run decoding. Try Gray labelling, lower SNR, other modulations. Read the warnings.
5. **Models** tab: feature classifier report, and instructions to train the CNN on Kaggle.

## Train the CNN on Kaggle (Phase 2b)
See `docs/PHASE2B_KAGGLE.md`. Notebook: `kaggle/train_modulation_cnn.ipynb`. Put the two output files into
`backend/models/` and restart.

## Layout
`backend/` signal processing and API, `frontend/` UI (HTML, CSS, JS in separate files), `kaggle/` training notebook,
`scripts/` tools, `docs/` algorithm explanations and finals Q&A, `samples/` generated captures.

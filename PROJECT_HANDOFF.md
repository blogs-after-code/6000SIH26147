# SanketSetu: project handoff

**Read this first if you are an AI agent or a collaborator joining the project.** It explains the goal, what exists, how it
was verified, what is not done, and how to work on it without breaking things. Last updated after Phase 5 (LDPC) and the
SanketSetu rebrand / UI refresh. Short version of the rules is in `AGENTS.md`.

---------------------------------------------------------------------------------------------------------------------

## 1. The project in one paragraph

SanketSetu is a web application for the **Smart India Hackathon 2026** that takes a raw radio capture (`.wav` or raw
IQ) and, with no prior knowledge of the signal, finds the signal, measures its parameters, identifies the modulation,
demodulates it, works out the error-correcting code and interleaving, finds the frame structure and recovers the message.
It shows its evidence and confidence at every step. It is built by **one student developer** who must also produce an
idea PPT (PDF) and a demo video, and who must be able to **explain and defend every part at the finals**. Prefer methods
that can be explained from first principles over black boxes.

## 2. The problem statement (as given)

*Background.* Raw data from terrestrial HF, VHF and UHF signals is analysed manually to identify signal parameters.
The resulting data is often insufficient for fine-grained analysis (modulation type, sampling rate, FEC, interleaving).

*Description.* Captures arrive as `.wav` or `.IQ` from different sensors and locations, so parameters vary and the two
formats need different processing. The expected solution may use GNU Radio, Python or C++ and should demodulate signals.
A GUI tool must accept `.IQ` or `.wav` and:

1. identify signal parameters (sampling frequency, modulation, FEC, interleaving; more if feasible)
2. demodulate (FSK, QAM, PSK)
3. de-interleave (block, convolutional, diagonal, pseudo-random)
4. FEC decode (short-constraint convolutional codes with Viterbi, RS block codes, concatenated codes, LDPC)
5. bit-stream correlation (to identify header and payload)

*Expected solution.* Better feature visibility through the GUI (sampling frequency, constellation plot, waterfall),
automated analysis, demodulation, de-interleaving and error correction, and bit-stream correlation for header/payload.

The problem owner organisation was **not stated by the developer**. Earlier conversation text assumed one; treat it as
unknown and do not put an organisation name in the product or the PPT unless the developer confirms it.

## 3. Working agreement with the developer (important)

- **Honest over impressive.** The developer explicitly wants direct, critical feedback, not validation. Never claim real-world
  performance from simulated data. State limits plainly (they are listed in section 9 and on the landing page).
- **Short, crisp communication.** No long explanations unless asked. When guiding them to run something, give numbered,
  simple steps with exact commands.
- **They must be able to defend it.** Every algorithm needs a plain explanation (what, why, how). `docs/PHASE*_EXPLAINED.md`
  hold these; extend them when you add or change an algorithm.
- **HTML, CSS and JavaScript must stay in separate files** (strong preference).
- Their laptop is weak: **no model training locally**. Training happens on Kaggle (free GPU). Inference must be light
  (ONNX Runtime on CPU, small models).
- They provided a design reference (`spectrascope.html`, dark neutral palette with one orange accent) but said it was a
  *reference, not to be copied*. The current UI is its own design system in the same spirit. The product was renamed from
  "Spectrascope" to **SanketSetu**; the developer chose the original orange accent (`#ff8a3d` dark, `#c2410c` light) and the
  Inter + JetBrains Mono fonts (the mono was changed from IBM Plex Mono at the owner's request) over a trial of `#9F3F17` and other fonts. Do not change them again unless asked. The UI must **not** imitate a
  government site (no ministry names, emblems or mottos).
- Deliverables still owed by the developer: PPT as **PDF** (max 6 slides including the title, template in section 11) and a
  demo video. The developer asked us to STOP working on the PPT: `deliverables/PPT_BRIEF.md` is a self-contained brief to
  give to another Claude account to generate the deck. The old slides draft was removed.

## 4. Status at a glance

| Area | State | Verified how |
|---|---|---|
| Phase 1: ingest (.wav, raw IQ), PSD, waterfall, centre frequency, bandwidth, SNR, signal generator | **Done** | 10 automated tests |
| Phase 2: symbol rate, timing, carrier recovery, modulation classification, demodulation (BPSK, QPSK, 8PSK, 16QAM, 2FSK, 4FSK) | **Done** | 13 automated tests, bit error rate checked against generator truth |
| Phase 2b: CNN on public data (Kaggle) | **Done, with caveats.** Trained (96.2 % at >=10 dB on RadioML). Labels re-attached by index, order verified. Informational only in the app (agreement with truth on our simulator below the 90 % trust bar). | 7 tests + label/gating tests; measured agreement with ground truth on our simulator 71 % (optimistic) |
| Phase 3: bit-mapping resolution, convolutional + Reed-Solomon blind identification and decoding, de-interleaving, framing | **Done for the supported scope** (section 9) | 29 automated tests including end-to-end from IQ samples |
| Front end: landing page (feature map to the problem statement, PDF-report section, boot screen, scroll animations), analyzer workspace with tabs, light/dark theme, mobile layout | **Done** (third iteration, SanketSetu branding) | Manual headless-browser runs and screenshots; **no automated UI tests** |
| Phase 4: accuracy graphs, PDF report, PPT brief, demo script, defence Q&A | **Drafted.** `scripts/evaluate_pipeline.py` + `results/`, `/api/report/{sid}`, `deliverables/PPT_BRIEF.md` (the deck itself is built elsewhere), `docs/DEMO_SCRIPT.md`, `docs/DEFENCE_QA.md`. Demo video itself not recorded. | report tests; evaluation run on own simulator |
| Validation on real captures | **Not done** | n/a |
| LDPC (Phase 5) | **Done for the supported scope**: min-sum decoder, 3 built-in IRA codes, upload of any parity-check matrix, automatic code and start-offset search, works alone, with RS and with a convolutional code | 12 tests in `test_phase5_ldpc.py`; evaluation curves for LDPC and RS+LDPC |

Total automated tests: **80 expected passing** (59 original + 7 in `test_phase4_report.py` + 14 in `test_phase5_ldpc.py`; about 75 s).

## 5. How to run it

```bash
cd signal-lab
python -m venv venv && source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m pytest backend/tests -q                      # expect: 80 passed
python scripts/make_samples.py                         # writes samples/ (cf32 + wav + truth json)
uvicorn backend.main:app --reload                      # open http://127.0.0.1:8000
```
In the UI: Analyzer, choose preset "Reed-Solomon + convolutional", Generate and analyze. Expect QPSK, conv generators
171/133, RS(204,188), 1504-bit frames with the CCSDS sync word, the recovered text, and ground truth 100 %.
Or drag `samples/coded_qpsk_rs204_conv_k7.cf32` in (sampling rate 100000).

## 6. Repository map

```
signal-lab/
  PROJECT_HANDOFF.md   this file          AGENTS.md  short rules for agents        README.md  run instructions
  requirements.txt
  backend/
    main.py              FastAPI app; in-memory session store; all HTTP routes; serves frontend/ statically
    core/
      loader.py          .wav / raw IQ (cf32, ci16, ci8, cu8) -> complex64 + fs
      spectral.py        Welch PSD, waterfall, noise floor, centre, 99% bandwidth, SNR
      generator.py       synthetic signals (RRC-shaped PSK/QAM, phase-continuous FSK), Gray/natural labelling, file writers
      demod.py           channelise, symbol rate, Oerder-Meyr timing, M-th power carrier recovery, demap, FSK discriminator
      classifier.py      feature extraction + Random Forest wrapper (BPSK/QPSK/8PSK/16QAM)
      dl_classifier.py   ONNX Runtime wrapper for the Kaggle-trained CNN (optional, absent until trained)
      pipeline.py        orchestrates demodulation, family detection (FSK vs linear), constellation-fit verification, CNN opinion
      decode.py          Phase 3 receiver: candidates -> de-interleave -> conv -> frames -> RS -> message
      framing.py         frame length by folding, sync word, header/payload split, frame extraction
      chain.py           TRANSMIT-side builder (frames -> RS -> conv -> interleaver) used to make test signals with known answers
      evaluate.py        BER with alignment helper
      fec/conv.py        conv encoder, Viterbi (hard), blind identification (matrix of GF(2) correlations)
      fec/rs.py          GF(256), RS encode/decode (Berlekamp-Massey, Chien, Forney), blind identification (two methods)
      fec/interleave.py  block / diagonal / convolutional (Forney) / seeded pseudo-random, start-offset alignment search
      fec/ldpc.py        LDPC: built-in IRA codes + encoder, uploaded-H parser (alist/dense/pairs), min-sum decoder, code + offset search
      report.py          PDF report builder (reportlab + matplotlib)
    models/              mod_rf.joblib (trained RF), eval_report.json; radioml_cnn.onnx + .json go here after Kaggle training
    tests/               test_phase1.py, test_phase2.py, test_phase2b.py, test_phase3.py, test_phase4_report.py, test_phase5_ldpc.py
  frontend/              index.html (landing + analyzer, hash routes), css/{style,loader}.css, js/{boot,loader,reveal,api,plots,hero,app}.js,
                         favicon.svg/.ico/png, assets/ (sample report + page previews)
  kaggle/                train_modulation_cnn.py (source of truth), .ipynb (generated by make_notebook.py)
  scripts/               make_samples.py, train_classifier.py (RF, runs locally in minutes)
  docs/                  PHASE1/2/3/4_EXPLAINED.md, PHASE5_LDPC.md (algorithms + finals Q&A), PHASE2B_KAGGLE.md, DEFENCE_QA.md,
                         DEMO_SCRIPT.md, REAL_CAPTURE_VALIDATION.md
  deliverables/          PPT_BRIEF.md (give to another Claude to build the deck), sample_report.pdf
  samples/               generated example captures (regenerate with scripts/make_samples.py)
```

## 7. Data flow and API

```
upload / demo ──> loader ──> spectral.analyze ──> [session stored: complex samples, fs, params]
                                  │
POST /api/demod/{sid} ──> pipeline.demodulate:
      channelize(fc, bw) ─> family (FSK if constant envelope + separable tones) ─┬─ linear: symbol rate (|x|^2 line) ─> RRC matched filter
                                                                                  │   ─> Oerder-Meyr timing ─> RF classifier ─> carrier recovery (M-th power + DD)
                                                                                  │   ─> demap ─> verify by constellation fit ─> CNN second opinion
                                                                                  └─ FSK: discriminator ─> symbol rate ─> sample mid-symbol ─> k-means tones
POST /api/decode/{sid} ──> decode.decode: bit candidates (rotations x natural/Gray) ─> [align + de-interleave, operator-supplied type/size]
      ─> conv code blind ID (K<=9, rate 1/2, 1/3) + Viterbi ─> LDPC (library + uploaded H, offset search, min-sum)
      ─> frame detection (fold) ─> RS blind ID + decode ─> polarity ─> message
```

| Route | Purpose | Notes |
|---|---|---|
| `GET /api/health` | status | `classifier_ready`, `cnn_ready`, `cnn_reason` |
| `GET /api/models` | RF report + CNN metadata | used by the Models tab and landing metrics |
| `GET /api/model-report` | RF hold-out report | |
| `POST /api/upload` | multipart `file`, `fmt`, `fs` | `fs` required for raw IQ. Returns spectral analysis + `session` |
| `POST /api/demo` | JSON, see `DemoRequest` in `main.py` | synthetic capture, optional transmit chain (`fec`, `conv`, `ldpc`, `interleaver`, `ia`, `ib`, `n_frames`, `gray`) |
| `POST /api/demod/{sid}` | JSON `mod?`, `symbol_rate?`, `beta` | overrides optional. Returns modulation, rate, plots, quality, bits, warnings, `deep` |
| `POST /api/decode/{sid}` | JSON `interleaver?: {kind,a,b,seed}` | slow (5 to 30 s); returns candidates, conv, rs, frame, frames, message, warnings |
| `POST /api/ldpc/{sid}` | multipart `file` (alist / dense 0/1 / row-col pairs) | registers a parity-check matrix for the session; the UI then re-runs decoding. `DELETE` clears them |
| `GET /api/ldpc-codes`, `GET /api/ldpc-codes/{name}.alist` | built-in LDPC library | alist download shows the accepted format |
| `GET /api/report/{sid}` | PDF report of the session | |
| `GET /api/payload/{sid}` | decoded payload bytes | |
| `GET /api/bits/{sid}?fmt=txt|bin` | demodulated bits | |

Sessions live in an in-memory `OrderedDict` (last 8). Single user, no auth, no persistence. Result field names are the
contract with `frontend/js/app.js`; if you change one, change both and the tests.

## 8. Key methods (one line each; details in `docs/`)

- **Symbol rate:** spectral line of |x|^2 (cyclostationarity of shaped linear modulations); FSK uses a transition signal.
- **Timing:** Oerder-Meyr, block-wise phase of the same line, linear fit gives clock drift and refines the rate.
- **Carrier:** M-th power gives frequency offset and phase; windowed estimate then decision-directed refinement. All
  estimators are **feed-forward** (no loops to tune). Result has an M-fold phase ambiguity.
- **Modulation:** Random Forest on spectral-line and amplitude features (`classifier.py`), verified by constellation fit
  (structured text payloads can bias the features). CNN second opinion when installed.
- **Blind conv code:** relation `g2*y1 = g1*y2`; all candidate polynomials tested at once with one matrix product.
- **Blind RS:** codewords are zero at 2t consecutive powers of alpha; noisy variant finds the low-complexity syndrome stretch.
- **LDPC:** share of violated parity checks for every start offset (random 0.5, right code < 0.4, z >= 7); normalised min-sum on hard bits. Details in `docs/PHASE5_LDPC.md`.
- **Frame length:** folding statistic with divisor search; sync word = long constant run; header = predictable columns.
- **Bit mapping:** all rotations x natural/Gray tried; scored by structure; prefers textbook generator sets on ties.

## 9. Known limitations and risks (be honest with judges)

1. **No real capture has been tested.** All validation uses the project's own generator and a self-written transmit chain.
   Highest-priority risk. A bug shared by transmitter and receiver would be invisible.
2. **CNN is informational only.** Trained and label order confirmed, but RadioML is simulated and agrees with ground truth on only ~71 % of our simulated captures, so its warnings are switched off (`TRANSFER_MIN` 0.90).
3. **LDPC needs a known parity-check matrix.** Built-in demo codes (256/128, 512/256, 1024/512) are found automatically; other codes (CCSDS, 5G, Wi-Fi, DVB-S2) work only after the operator uploads their H (alist, dense 0/1 or row-col pairs; systematic, information bits first). The built-in codes are NOT the standard ones. Blind recovery of an unknown H is an open research problem. Decoder takes hard bits (no soft decisions).
4. **Interleaver type and size must be supplied.** Only the start offset is found blindly. Pseudo-random needs the seed.
5. **Framing needs a repeating sync word and about 12+ long frames.** Streams without one cannot be aligned for RS.
6. **Not supported:** scramblers/whitening, puncturing, soft-decision decoding, RS dual basis, RS interleave depth above 1,
   conv codes above K=9 or rates other than 1/2 and 1/3, OFDM, frequency hopping, spread spectrum, 32/64/256-QAM,
   multiple carriers in one file, MSK/GMSK, analog modulations.
7. Sampling rate of **raw IQ cannot be recovered** from the file; the operator supplies it.
8. Carrier capture range after centring is about +/- Rs/(2M); a bad centre estimate breaks carrier recovery.
9. Random-forest accuracy (98 %) is on its own simulator: an upper bound.
10. Decoding time up to ~30 s (interleaver alignment); no progress streaming.
11. UI has no automated tests; verified manually in headless Chromium and screenshots. Fonts load from Google Fonts
    (fall back to system fonts offline).
12. Developed and tested on Linux; the developer will run on their own machine (likely Windows). Path or dependency
    issues are possible; report any.

## 9b. Added in the Phase 4 session
- Decoder fix: candidates ranked by Viterbi probe error (captures with >= 17 frames used to fail). See `docs/PHASE4_EXPLAINED.md` section 3.
- **Open weakness:** BPSK is called FSK in 15-30 % of simulated captures at 6 dB and above (tone-cluster test fires on noise). Backlog item 0.
- CNN label order confirmed from the dataset by the developer's run of `kaggle/verify_class_order.py` (`labels_verified` true).
- Python 3.13 / sklearn 1.9 warns that `mod_rf.joblib` was pickled with 1.8; retrain with `scripts/train_classifier.py` if it matters.

## 9c. Added in the Phase 5 session
- LDPC decoding (`fec/ldpc.py`, `decode.py` step 3b, `/api/ldpc*` routes, UI card in the Decoding tab, report row, 14 tests).
- Rebrand to SanketSetu; landing page rebuilt around the five tasks in the problem statement with Working / Partly / Not yet labels.
- Static front-end files are served with `Cache-Control: no-cache` and script URLs carry `?v=6`. Reason: a stale cached `api.js`
  once made the new `app.js` fail with "API.ldpcCodes is not a function". Bump the `?v=` number when front-end files change.
- Evaluation curves now include LDPC(512,256) and RS(204,188)+LDPC(512,256): full recovery up to ~3-4 % raw BER (own simulator).

## 9d. Added in the Phase 6 session (frontend showcase and mobile)
- Landing now follows the problem statement: `#problem` (background/description), `#solution` (expected solution mapped to views, with real screenshots in `frontend/assets/shots/*.webp`), `#coverage`, `#model` (CNN and RF accuracy, training recipe, per-class bars, honest caveat; numbers filled from `/api/models`, which now also returns `cnn_transfer` and `cnn_size_bytes`), `#report`.
- Colours: orange main accent, `--blue` (CNN/data), `--green` (classifier/pass) tokens in `style.css`; only via variables.
- New files: `css/sections.css`, `css/responsive.css` (breakpoints 1100/860/600/400), `js/nav.js` (hamburger menu). Cache version is `?v=7`.
- Mobile checked with Playwright at 360, 390, 768, 1366 px in both themes: no horizontal overflow on landing or any analyzer tab. Not tested on a physical phone.
- Screenshots were captured from the running analyzer (LDPC+RS preset, 14 dB, generated signal); recapture if the UI changes.

## 9e. Deployment (kept intact in Phase 6)
- The project was deployed as Render (backend, `render.yaml`, `requirements-prod.txt`) plus Vercel (`frontend/`, `vercel.json`, `build-config.mjs` writes `js/config.js` from `API_BASE_URL`). See `DEPLOY.md`.
- Rules for any frontend change: every backend link or download must go through `API.url(...)`; `js/config.js` must load before `js/api.js`; new static assets live under `frontend/` so Vercel serves them. CORS allows `*.vercel.app` plus `CORS_ORIGINS`.
- Phase 6 verified in split-origin mode (static server on one port, API on another with CORS): status, model section, a full analysis and the report link all work.

## 9f. Decoder time budget
- `decode()` searches for the interleaver alignment under a time budget (now 120 s, inner search up to 45 s; it stops early once a clear match is found, so normal speed is unchanged). With the old 40 s / 15 s budget a slow or busy machine (the free Render CPU, a user's laptop under load) ran out of time and reported no message. Reproduced by running the test with 8 busy loops on 2 cores.

## 9g. Landing navigation and pipeline (frontend only)
- Nav labels equal the section eyebrows: Overview, Problem, Solution, Capabilities, AI Model, PDF Report, How it works, Limitations, Open Analyzer. Section ids are unchanged (`home, problem, solution, coverage, model, report, how, limits`). The old `#evidence` section was removed; its numbers live in AI Model.
- `js/nav.js` has a scroll-spy: the highlighted link follows the section under the top bar; clicking a link scrolls there even if the URL already matches. The hamburger now appears below 1060 px because nine links do not fit.
- Analysis pipeline: rail shows 7 stages with description while running, result text when done, a progress bar and count; a stage banner shows the stage and, for decoding (stages 5 to 7), the options being tried. The report bar stays hidden until decoding ends. Cache version `?v=8`.
- Fixed: `.grow` rule was hiding alert text on phones.

## 9h. Landing trim and polish
- Landing shortened: fewer images (analyzer, waterfall, constellation, one report page), shorter copy, compact capability cards, no gradient progress bars (pipeline uses seven solid segments), the hero PDF pill removed. Wave divider now reveals with a clip, not a dash animation (the dash version left the line cut short). Cache version `?v=9`.
- Mono font is JetBrains Mono everywhere (CSS, canvas plots, loader); canvases redraw when web fonts finish loading.

## 10. Backlog (priority order, with acceptance criteria)

**P0, needed for the submission**
0. *Fix BPSK-as-FSK*: evaluate both families by constellation/tone fit and keep the better one; re-run `evaluate_pipeline.py --only demod`.
1. *Run the Kaggle training* (`docs/PHASE2B_KAGGLE.md`). Done when `backend/models/radioml_cnn.onnx` + `.json` exist, the
   Models tab shows accuracy by SNR, and a test file produces a CNN opinion. Fix any bugs the first real run reveals
   (training/export cells are untested).
2. *Validate on a real or public IQ capture* (RTL-SDR recording, or a public dataset with known parameters). Done when
   there is a short written result: what worked, what failed, with screenshots. Update section 9 accordingly.
3. *Accuracy graphs* (`scripts/evaluate_pipeline.py`, new): BER vs SNR per modulation; modulation-ID accuracy vs SNR;
   decode success vs channel bit error rate for conv, RS, concatenated. Output JSON + PNG. Done when the PPT can cite them.
4. *PDF report export* (`GET /api/report/{sid}`): parameters, plots, decoded message, warnings, timestamp, tool version.
5. *PPT (PDF)* and the demo video: owned by the developer. `deliverables/PPT_BRIEF.md` is the brief for generating the deck;
   `docs/DEMO_SCRIPT.md` is the video script; `docs/DEFENCE_QA.md` is the Q&A sheet (all drafted).

**P1, strengthens the project**
6. (Done in Phase 5) Next for LDPC: ship the standard CCSDS and 802.11 matrices as files, soft-decision input, non-systematic output positions.
7. Scrambler support (CCSDS PN, additive and self-synchronising) as an extra hypothesis in `decode.py`.
8. Puncturing patterns for conv codes (2/3, 3/4); soft-decision Viterbi (needs soft demod output).
9. Progress streaming (server-sent events) for the decode step; cancel button.
10. SigMF metadata ingest (sampling rate and centre frequency from the `.sigmf-meta` file).
11. Automated UI tests (Playwright) for the main flow.

**P2, nice to have**
12. Persist sessions; batch mode over a folder; Docker image; GNU Radio flowgraph export; i18n (Hindi).
13. Blind interleaver-parameter search using the decoded structure (research item; document negative results too).

## 11. Submission requirements (from the uploaded SIH template, `SIH2026-IDEA-Presentation-Format.pptx`)

Slides: 1 title (problem statement ID, title, theme, category, team ID and name), 2 **Idea** (proposed solution, how it
addresses the problem, innovation and uniqueness), 3 **Technical approach** (technologies, methodology, flow charts, working
prototype), 4 **Feasibility and viability** (feasibility, challenges and risks, strategies), 5 **Impact and benefits**,
6 **Research and references**. Rules: **maximum 6 slides including the title**, points/diagrams not paragraphs, precise,
**saved and uploaded as PDF only**, use the provided template without changing its headings.

## 12. How to work on this codebase

- Run `python -m pytest backend/tests -q` before and after every change. Do not lower an assertion to make a test pass;
  if a threshold is wrong, say why.
- New algorithm = new module or function + tests with a **known answer** (use `generator.py` / `chain.py`) + an entry in
  `docs/PHASE*_EXPLAINED.md` (what, why, how, honest limits).
- Adding a **modulation**: constellation in `generator.py`, demap and carrier recovery in `demod.py`, label in
  `classifier.LABELS` and retrain (`python scripts/train_classifier.py`), plots in `frontend/js/plots.js`, tests.
- Adding a **FEC code**: encoder in `chain.py` (to test with), blind identification + decoder in `fec/`, a step in
  `decode.py` with a result block, UI rows in `app.js` (`renderTexts`), tests, limits updated.
- Keep API field names stable or update `frontend/js/app.js` and tests together.
- Frontend: plain HTML, CSS, JS in separate files; no build step; colours only through CSS variables (so dark/light work);
  every plot reads colours at draw time; keep text contrast high and warnings visible.
- Never fabricate numbers in the UI. Every metric shown must come from a measurement or a labelled report file.
- Do not commit trained models of unknown origin. Record dataset, date and hash in the model JSON (the notebook does).
- Be careful with long-running tests (decode tests take seconds each). Do not run `pkill -f uvicorn` inside a shell that
  also needs to keep running other commands.

## 13. Development environment notes (what the previous agent could and could not do)

- Python 3.12, numpy/scipy/scikit-learn/FastAPI/onnxruntime available; headless Chromium via Playwright worked (used for
  screenshots). Network access only to package indexes: **Kaggle datasets and Google Fonts were not reachable**.
- **PyTorch could not be installed** (needs large CUDA libraries); therefore the Kaggle training code was never executed.
- ONNX consumption was tested with a hand-built stand-in graph (`backend/tests/test_phase2b.py`), not a trained model.

## 14. Decision log (why things are the way they are)

| Decision | Reason |
|---|---|
| Feed-forward estimators instead of PLL / Costas / Gardner loops | no gains to tune, no cycle slips, simple to explain; works on short captures |
| Random Forest on explainable features first, CNN second | trains in seconds on a weak laptop; explainable; the CNN adds a public-data cross-check |
| ONNX Runtime for the CNN | no PyTorch needed on the developer's laptop |
| Blind conv identification by pairwise GF(2) correlation | exact, fast for K up to 9, easy to explain |
| Interleaver type/size supplied by operator | blind discovery is an open problem; honest scope |
| Constellation-fit check after classification | structured payloads fooled the classifier (16QAM called BPSK) |
| Own transmit chain (`chain.py`) | gives ground truth; also the biggest validity risk (section 9.1) |
| Single-file in-memory sessions | demo scope; persistence is backlog |
| Neutral product styling | not a government site; no official branding |

## 15. Glossary

IQ: complex baseband samples (I = real, Q = imaginary). Symbol rate Rs: symbols per second. sps: samples per symbol.
RRC: root-raised-cosine pulse filter. CFO: carrier frequency offset. EVM: error vector magnitude. Es/N0: symbol energy to
noise density. FEC: forward error correction. K: constraint length of a convolutional code. RS(n,k): Reed-Solomon with n
symbols per codeword, k data symbols. t: correctable symbol errors, (n-k)/2. ASM: attached sync marker. Gray labelling:
neighbouring constellation points differ by one bit. Fold: stacking a bit stream into rows of a trial frame length.

## 16. Prompt starters for another AI agent

- "Read PROJECT_HANDOFF.md and AGENTS.md. Run the tests. Then implement backlog item 3 (scripts/evaluate_pipeline.py)
  with a known-answer test. Do not change existing assertions."
- "Read PROJECT_HANDOFF.md section 9. Review `backend/core/decode.py` for ways it could fail on a real capture that the
  self-made transmit chain would not reveal, and write tests for the three most likely."
- "Here is the traceback from running kaggle/train_modulation_cnn.ipynb on Kaggle: ... Fix the notebook (edit the .py
  source and regenerate with kaggle/make_notebook.py)."
- "Read deliverables/PPT_BRIEF.md and build the six-slide deck exactly as specified."

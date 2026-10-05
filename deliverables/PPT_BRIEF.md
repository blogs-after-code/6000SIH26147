# PPT brief for SanketSetu (give this file to Claude on another account)

**Task for Claude:** build the Smart India Hackathon 2026 idea presentation described below. Output a **.pptx** (and, if you
can, also export a **PDF**; the portal accepts PDF only). Build exactly the six slides specified, with the headings given.
Do not add slides. Do not invent facts, numbers or team details: everything you may state is in this file. Where a value is
marked `[FILL IN]`, leave a clearly visible placeholder.

## 0. Hard rules from the SIH template
- **Maximum 6 slides including the title slide.** Slide order and headings are fixed (section 3).
- Points and diagrams, **not paragraphs**. Short, precise bullets (about 8 to 14 words each, 4 to 7 bullets per slide).
- Keep the template's headings unchanged. If the developer attaches the official `SIH2026-IDEA-Presentation-Format.pptx`,
  **use it as the template** and only fill it in. If it is not attached, use the design in section 4.
- No ministry names, emblems, mottos or government branding anywhere. Do not name the problem owner organisation.

## 1. The project in one paragraph (for your understanding, not to paste)
**SanketSetu** (Hindi: संकेत सेतु, "signal bridge") is a web tool. The user uploads a radio recording (.wav or raw .IQ).
With no prior knowledge of the signal it measures the signal parameters, identifies the modulation, demodulates it,
finds the forward-error-correction (FEC) code and interleaving, correlates the bit stream to separate header from payload,
and recovers the message. Every step shows its evidence (spectrum, constellation, confidence, warnings). Every analysis can be
downloaded as a **PDF report**. Theme: **Space Technology**.

## 2. The problem statement (what the tool must do)
"The GUI based model will have features to take .IQ or .wav file as input data and perform the following tasks":
1. Identify signal parameters (sampling frequency, modulation, FEC, interleaving). Additional features if feasible.
2. Demodulate signals (FSK, QAM, PSK).
3. Carry out de-interleaving (block, convolutional, diagonal, pseudo-random).
4. FEC: short-constraint convolutional codes with Viterbi decoding, RS block codes, concatenated codes, LDPC.
5. Bit-stream correlation.

Context: terrestrial HF/VHF/UHF captures come from different sensors and places. Today analysts work out sampling rate,
modulation, FEC and interleaving by hand, one signal at a time; raw IQ has no header; IQ and WAV need separate handling;
manual results are hard to reproduce and carry no confidence score.

## 3. Slide-by-slide content

### Slide 1: Title
- Small label: `SMART INDIA HACKATHON 2026 | IDEA PRESENTATION`
- Big title: **SanketSetu**
- Subtitle: *Automated analysis of .IQ and .wav files with signal parameter extraction*
- Table/fields (placeholders): Problem Statement ID `[FILL IN]`, Problem Statement Title `[FILL IN from the SIH portal]`,
  Theme: `Space Technology`, Category `[FILL IN]`, Team ID and Team Name `[FILL IN]`.
- Logo (file `favicon.svg`, a suspension bridge with a signal wave as its deck) top left.

### Slide 2: Idea
Heading **Idea**. Sub-blocks (keep the template's wording for the sub-headings):
- **Proposed solution:** web tool; upload .wav or raw IQ; reports parameters, modulation, FEC, interleaving, frame structure; recovers the message.
- **How it addresses the problem:** replaces manual analysis with one automated chain; shows spectrum, constellation, confidence at every step; one-click PDF report.
- **Coverage of the five tasks (small table, 5 rows; this is the key visual of the slide):**

| Task in the brief | What SanketSetu does | Status |
|---|---|---|
| i. Signal parameters | centre frequency, bandwidth, SNR, symbol rate, modulation, FEC, frame length, sync word | Working |
| ii. Demodulate FSK / QAM / PSK | 2FSK, 4FSK, BPSK, QPSK, 8PSK, 16QAM | Working (QAM: 16QAM only) |
| iii. De-interleave | block, convolutional, diagonal, pseudo-random | Working (type and size entered by operator; start offset found automatically) |
| iv. FEC | convolutional + Viterbi, Reed-Solomon, concatenated (RS + conv, RS + LDPC), LDPC | Working (LDPC: built-in codes found automatically; other codes need their parity-check matrix uploaded) |
| v. Bit-stream correlation | frame length, sync word, header vs payload | Working (needs a repeating sync word, about 12 frames) |

- **Innovation and uniqueness:**
  - blind identification of the convolutional code and the Reed-Solomon code from the bits alone
  - LDPC code and block start found automatically from the share of violated parity checks
  - feed-forward estimators: no loops to tune, every step explainable from first principles
  - second opinion from a CNN trained on public RadioML data
  - **PDF evidence report for every analysis** (call this out visibly, e.g. a highlighted badge "Key feature")

### Slide 3: Technical approach
Heading **Technical approach**.
- **Flow chart** (left to right, one box each; draw it as a diagram, not text):
  `.wav / .IQ` > `Ingest` > `Spectrum: centre frequency, bandwidth, SNR` > `Symbol rate, timing, carrier recovery` >
  `Modulation (feature classifier + CNN)` > `Demodulate to bits` > `De-interleave` > `Convolutional (Viterbi) > LDPC > Reed-Solomon` >
  `Frames: sync, header, payload` > `Message + PDF report`
- **Technologies:** Python, NumPy / SciPy, scikit-learn, ONNX Runtime, FastAPI, plain HTML / CSS / JavaScript, ReportLab (PDF).
- **Methods (one short line each):**
  - symbol rate from the spectral line of |x|^2; Oerder-Meyr timing; M-th power carrier recovery
  - convolutional code identified by GF(2) correlation, decoded with Viterbi
  - Reed-Solomon identified from consecutive zero syndromes, decoded with Berlekamp-Massey
  - LDPC: normalised min-sum belief propagation; code and start found from violated-check share
  - frame length by folding the bit stream; sync word = long constant run across frames
- **Working prototype:** web app with 80 automated tests; screenshot of the analyzer results page (see section 5, file `analyzer.png`).
- Image: `results/demod_vs_snr.png` (caption: "Own simulator, 20 captures per point. Not real captures.").

### Slide 4: Feasibility and viability
Heading **Feasibility and viability**.
- **Feasibility:** runs on a laptop CPU; CNN is a 2.3 MB ONNX file; open-source stack; no GPU or licence at run time.
- **Measured on our own simulator (say "simulator" every time):**
  - bit error rate 0 at 8 dB full-band SNR for BPSK, QPSK, 8PSK and 2FSK
  - convolutional K=7 recovers the message up to about 2 % raw bit errors
  - Reed-Solomon (255,223) up to about 0.3 %
  - RS (204,188) + convolutional K=9 up to about 3 %
  - LDPC (512,256) up to about 3 to 4 %; RS + LDPC up to about 4 %
- **Challenges and risks (state plainly; judges reward honesty):**
  - not yet validated on real captures; transmitter and receiver were written by the same developer
  - BPSK is sometimes mistaken for FSK in simulation (15 to 30 % of captures at 6 dB and above)
  - the CNN agrees with ground truth on 71 % of simulated captures, so it is shown as informational only
  - LDPC for codes other than the built-in demo codes needs the parity-check matrix; no scramblers, puncturing or soft decisions
  - interleaver type and size are entered by the operator
- **Strategies to address them:** validate on public and RTL-SDR captures; ship the standard CCSDS and Wi-Fi LDPC matrices; soft-decision decoding; run both modulation families and keep the better fit.
- Image: `results/decode_vs_ber.png` (caption: "Blind decode success vs raw channel bit error rate. Own simulator and transmit chain.").

### Slide 5: Impact and benefits
Heading **Impact and benefits**.
- **Analysts:** minutes instead of hours for first-pass parameter identification; every conclusion shows its evidence.
- **Trust:** confidence, warnings and limits are shown; estimates are never presented as certain.
- **Reproducible:** one-click PDF report (capture, parameters, plots, decoded message, warnings, limits) to file, share or attach.
- **Reuse:** works on files from different sensors (.wav, raw IQ in four sample formats).
- **Training and research:** each algorithm is explainable and documented with known-answer tests.
- **Low cost:** open-source, laptop-class hardware.
- **Roadmap:** standard LDPC matrices, scramblers, SigMF metadata, batch mode, GNU Radio flowgraph export.
- Image: the two PDF report pages side by side (`report-page-1.png`, `report-page-2.png`), caption "Downloadable PDF report".

### Slide 6: Research and references
Heading **Research and references**.
- O'Shea, Roy, Clancy. "Over-the-Air Deep Learning Based Radio Signal Classification". IEEE J. Sel. Topics Signal Process., 2018 (RadioML 2018.01A dataset).
- Oerder, Meyr. "Digital Filter and Square Timing Recovery". IEEE Trans. Commun., 1988.
- Viterbi. "Error Bounds for Convolutional Codes and an Asymptotically Optimum Decoding Algorithm". IEEE Trans. Inf. Theory, 1967.
- Reed, Solomon. "Polynomial Codes over Certain Finite Fields". J. SIAM, 1960.
- Gallager. "Low-Density Parity-Check Codes". IRE Trans. Inf. Theory, 1962.
- CCSDS 131.0-B: TM Synchronization and Channel Coding (attached sync marker 1ACFFC1D, RS(255,223), K=7 convolutional code).
- Tools: FastAPI, ONNX Runtime, scikit-learn, NumPy / SciPy, ReportLab.
- Footer line: *All accuracy figures come from this project's own simulator. Dataset: RadioML 2018.01A (simulated).*

## 4. Design (use if no official template is attached)
- 16:9. Dark background `#0c0d0f`, panels `#131519`, text `#eceef2`, muted text `#a3abb7`. **Orange `#ff8a3d` is the main accent; blue `#5aa9ff` marks the CNN/data and green `#3ecf8e` marks the feature classifier and passing results**
  (use `#c2410c` if you make a light version). Status colours only for the status column: green `#4cc38a`, amber `#f2b84b`, red `#ff6b6b`.
- Fonts: **Inter** for text, **JetBrains Mono** for numbers, code and labels. Titles 32 to 40 pt bold, body 16 to 20 pt. Nothing below 14 pt.
- One idea per slide, generous spacing, thin 1 px panel borders, no clip art, no stock photos, no gradients on text.
- Every figure has a caption stating its source. Use the logo (`favicon.svg`) on the title slide only.

## 5. Files the developer should attach together with this brief
| File | Where in the project | Use |
|---|---|---|
| `demod_vs_snr.png` | `results/` | slide 3 |
| `decode_vs_ber.png` | `results/` | slide 4 |
| `report-page-1.png`, `report-page-2.png` | `frontend/assets/` | slide 5 |
| `favicon.svg` | `frontend/` | logo, slide 1 |
| `analyzer.png` | take a screenshot yourself: open the app, preset "Reed-Solomon + LDPC", Generate and analyze, capture the results page | slide 3 |
| `SIH2026-IDEA-Presentation-Format.pptx` | the official template, if you have it | use as the template |

## 6. Checks before you finish
1. Exactly 6 slides; headings match section 3; title slide has the placeholders.
2. Every number on a slide appears in this file; every performance number says "simulator" or has the footer.
3. No paragraphs; no text below 14 pt; nothing overflows its box (render and look at every slide).
4. No government names or emblems. Mention that the PDF report is a key feature on slides 2 and 5.
5. Export the PDF as well and check that the page count is 6.


## Model facts for the model slide (all from `backend/models/*.json`; label every figure as simulated)
- CNN: 1-D ResNet, 588,440 parameters, ONNX about 2.2 MB. Dataset RadioML 2018.01A, 24 classes, 26 SNRs (-20 to 30 dB), 800 frames per class and SNR (499,200 total, 80/10/10 split), 1024 I/Q samples, unit RMS. AdamW, one-cycle LR, label smoothing 0.05, dropout 0.3, phase and frequency augmentation, 25 epochs on a Kaggle GPU.
- CNN accuracy: 96.2% at 10 dB and above, 60.5% over all SNRs. Weak spots: AM-SSB/DSB with and without carrier, 256QAM vs 128QAM.
- Feature classifier (Random Forest, 6 features): 98.1% on 480 held-out generated signals, 4 classes.
- Honest caveat to state on the slide: on our own simulator the CNN agrees with the truth 71% of the time (320 captures), so it is a cross-check only. No real capture tested.
- Landing page sections to screenshot: #problem, #solution, #model (live at `/#/model`).

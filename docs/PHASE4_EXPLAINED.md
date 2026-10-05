# Phase 4 explained: evaluation, report, CNN integration, decoder fix

All numbers below are measured on this project's OWN simulator (`generator.py`, `chain.py`). They are upper bounds for
real captures. Reproduce: `python scripts/evaluate_pipeline.py` (results in `results/`).

## 1. Accuracy curves (`results/demod_vs_snr.png`, `results/decode_vs_ber.png`)
- Setup: 100 kHz sampling, 12.5 kBd, random carrier offset within +/-8 kHz, 20 captures per modulation and SNR (full-band SNR;
  in-band SNR is about 7.7 dB higher).
- Bit error rate is the median over correctly identified captures, scored against the transmitted bits.
- Practical reading: at 8 dB full-band SNR and above BER is 0 for BPSK/QPSK/8PSK/2FSK, ~1e-3 for 16QAM, ~1e-4 for 4FSK.
  At -2 dB BPSK/QPSK/4FSK are not even detected (identification fails).
- Blind decode success (random bit flips, no soft decisions): conv K=7 to ~2 % raw BER, RS(255,223) to ~0.3 %,
  RS(204,188)+conv K=9 to ~3 %, LDPC(512,256) to ~3-4 %, RS(204,188)+LDPC(512,256) to ~4 % (LDPC added in Phase 5). Success falls off a cliff past those points (expected for hard-decision Viterbi / RS).

## 2. Known weaknesses the evaluation exposed (say these first)
1. **BPSK is called FSK in roughly 15-30 % of captures at 6 dB and above** (QPSK about 5-10 % at 16-20 dB). Cause: the FSK
   test accepts "four separable clusters of instantaneous frequency" on its own, and noise on a BPSK signal sometimes
   forms them. Envelope variation (about 0.40) says "not constant envelope", but 4FSK at 0 dB has the same value, so a
   threshold cannot fix it without losing low-SNR FSK. Proper fix (backlog): run both families and keep the one with the better
   constellation / tone fit.
2. Nothing is detected at about -2 dB full-band SNR.
3. 16QAM, 64QAM, 256QAM style confusions are expected for any classifier at low SNR.

## 3. Decoder bug found and fixed
Captures with 17 or more frames failed to frame while 14 frames worked. Two of the four phase rotations give the bitwise
complement of the data. The code structure and fold strength look identical, and Viterbi on the complement leaves 11-15 %
errors, so framing failed. Candidates are now ranked by the residual error of a short Viterbi probe
(`decode.probe_ber`). Regression tests: `test_phase4_report.py::test_large_capture_decodes_...`.

## 4. CNN on public data (RadioML 2018.01A)
- Trained on Kaggle: 60.5 % overall test accuracy, 96.2 % at 10 dB and above (normal for this dataset).
- Weak classes are exactly the physically hard ones: 64/128/256QAM and the AM variants.
- **Label order**: the first run attached names in the `classes.txt` order, which is not the order of the HDF5 one-hot column.
  The weights are fine; names were re-attached by index (`scripts/fix_cnn_labels.py`). The corrected order was *inferred* first
  and then **confirmed from the dataset itself** by the developer's run of `kaggle/verify_class_order.py` (FM and GMSK among the flattest
  envelopes, OOK the most variable: the fixed order). `labels_verified` is true in `radioml_cnn.json`.
- **Domain gap**: clean channelised input is out of distribution (0 of 36 correct). Adding seeded white noise at 5 dB brings
  agreement to 71 % on our simulator (16QAM is always called 256QAM). That setting was tuned on the same simulator, so the
  figure is optimistic. The CNN is shown as informational; it raises warnings only when labels are verified AND measured
  agreement is at least 90 % (`dl_classifier.warnings_trusted`).

## 5. PDF report
`GET /api/report/{session}` (buttons "Download PDF report" at the top of the results and "PDF report" under the bitstream): capture info, measured parameters, spectrum and waterfall,
modulation results and constellation, decoding results and recovered message, warnings, and a method-limits footer.

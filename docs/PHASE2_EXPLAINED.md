# Phase 2 explained (what / why / how) + finals Q&A

## Pipeline (backend/core/pipeline.py calls these in order)
1. **channelize** (`demod.py`): shift the detected centre frequency to 0 Hz, low-pass to the occupied band. Removes everything that is not our signal.
2. **Family decision** (`pipeline.detect_family`): constant envelope + separable tones = FSK, otherwise linear (PSK/QAM).
3. **Symbol rate**: linear -> spectral line of |x|^2; FSK -> spectral line of the frequency-transition signal.
4. **Matched filter + timing**: root-raised-cosine matched filter, then Oerder-Meyr timing from the phase of the same |x|^2 line.
5. **Modulation classifier**: Random Forest on 6 explainable features.
6. **Carrier recovery**: M-th power gives frequency offset and phase; a decision-directed pass tightens it.
7. **Demap**: nearest constellation point -> bits.

## Key ideas
**Why |x|^2 reveals the symbol rate.** A filtered linear modulation has a *periodic* power envelope: it rises and falls once per symbol. That periodicity is a spectral line at exactly the symbol rate. It exists because of the excess bandwidth (roll-off); with zero roll-off it vanishes.

**Why timing comes from the same line.** The *phase* of that line says where in each symbol period the power peaks, which is the best sampling instant. We measure it in blocks of 64 symbols and fit a straight line through the phase. The slope is clock drift, which also refines the symbol-rate estimate.

**Why feed-forward, not loops.** No loop gains to tune, no cycle slips, works on short captures. Cost: it needs a block of data and assumes slow drift.

**M-th power.** Raise symbols to the power M (2 for BPSK, 4 for QPSK/16QAM, 8 for 8PSK). The modulation disappears (all points map to the same angle) leaving a pure tone at M x the frequency offset. Its frequency gives the offset, its phase gives the carrier phase. Cost: an M-fold phase ambiguity, which Phase 3 resolves with frame sync.

**Classifier features.** pmr2/pmr4/pmr8 = how strong the line of y^2/y^4/y^8 is. BPSK has lines at 2 and 4, QPSK at 4, 8PSK at 8. m4, m6, cv measure amplitude spread; 16QAM has three amplitude rings, PSK has one. All are insensitive to carrier phase and frequency, so the classifier can run before carrier recovery.

**FSK.** A frequency discriminator turns phase change into instantaneous frequency. Each symbol is a plateau at one of M tone frequencies. We sample mid-symbol and cluster. 2 vs 4 tones: fit 4 clusters and check whether the gaps are equal (true 4FSK) or very unequal (a 2FSK cloud split in half).

## Measured results (on our generator)
- 6 modulations, blind, bit error rate below 1 % at the SNRs in the tests (see `test_phase2.py`).
- Classifier (4 linear types) hold-out accuracy is in `backend/models/eval_report.json` and shown in the workspace.
- Works with non-integer samples per symbol (tested with fs / Rs = 7.76), roll-off mismatch (true 0.5 vs assumed 0.35), carrier offsets up to tens of kHz (cleared by Phase 1 centring).

## Honest limits (say these first)
1. **Synthetic only.** The classifier is trained and tested on our own generator. Real captures have fading, interference, drifting oscillators and filter shapes we did not model. Expect lower accuracy; do not claim real-world numbers. Close this gap by testing on a real RTL-SDR or public recording.
2. **Carrier capture range** is about +/- Rs/(2M) after Phase 1 centring. A badly wrong centre estimate breaks M-th power.
3. **Roll-off is assumed** (0.35 default) for the matched filter; mismatch costs a little SNR. You can override it in the UI.
4. **Bits have an unknown start and an M-fold phase ambiguity** until Phase 3 sync.
5. FSK assumes phase-continuous tones and a symbol rate that is not tiny relative to the bandwidth. 4FSK with very close tones will be unreliable (the UI warns).
6. Higher-order QAM (32/64/256), OFDM, spread-spectrum and burst/hopping signals are not handled.
7. Single signal per file: overlapping carriers are not separated.

## Likely questions
- **How do you know the symbol rate without being told?** The |x|^2 spectral line; show the plot.
- **Why not a CNN?** Features + Random Forest trains in seconds on a laptop, is explainable, and reaches high accuracy on this task. A CNN is the upgrade path once real labelled data exists.
- **Why does accuracy drop at low SNR?** The lines and amplitude rings drown in noise; shown in the accuracy-by-SNR bars.
- **What is the phase ambiguity?** After M-th power you cannot tell which of M equivalent rotations is true. Frame sync words disambiguate.
- **How did you validate?** Generator ground truth, bit error rate against true bits, unit tests, hold-out set with unseen seeds.
- **What if the modulation is wrong?** The UI shows probabilities and warnings; the operator can override and re-run.

## Phase 3 preview
Sync-word and frame detection (resolves phase ambiguity and start offset), de-interleaving (block first), Viterbi + Reed-Solomon with known parameters.

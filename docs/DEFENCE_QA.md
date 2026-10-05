# Defence Q&A sheet (consolidated; details in PHASE*_EXPLAINED.md)

**State limits first:** all validation is on our own simulator; no real capture has been tested; scramblers, puncturing,
soft decisions are not supported; LDPC needs the parity-check matrix for any code that is not one of our three built-in demo codes; interleaver type/size are operator-supplied.

## Problem and approach
- *What does the tool do?* Raw .wav/.IQ in; finds the signal, measures parameters, identifies modulation, demodulates,
  de-interleaves, FEC-decodes, finds frames and recovers the message, with evidence at each step.
- *Why not GNU Radio?* We need blind analysis with explainable estimators and a web GUI; every stage is plain Python we can derive.
- *What is new?* Blind end-to-end chain with feed-forward estimators (no loops to tune), blind convolutional and RS identification,
  and evidence shown at every step.

## Signal processing
- *Symbol rate without being told?* Spectral line of |x|^2 (cyclostationarity of shaped linear modulations); FSK uses a transition signal.
- *Why feed-forward timing/carrier?* No gains to tune, no cycle slips, works on short captures, easy to explain.
- *Phase ambiguity?* M-th power recovers carrier up to M rotations; we try all rotations and natural/Gray labelling and pick by structure.
- *How is SNR defined?* Full-band: signal power over total noise across the sampling band; in-band is higher by about fs/bandwidth.

## FEC
- *How do you find the convolutional code blind?* Relation g2*y1 = g1*y2 over GF(2); all candidate polynomials tested with one matrix product.
- *Why (171,133)?* CCSDS/NASA standard; equivalent codes exist after bit re-labelling, we prefer textbook ones.
- *Blind Reed-Solomon?* Codewords are zero at 2t consecutive powers of alpha; a noisy variant finds the low-complexity syndrome stretch.
- *Frame length/sync?* Folding statistic; sync word = long constant run across frames; header = predictable columns.
- *What if there are too many errors?* RS reports uncorrectable words instead of returning wrong data.

## LDPC and the PDF report
- *How do you decode LDPC?* Normalised min-sum belief propagation on hard bits, up to 60 iterations, against a parity-check matrix H.
- *Is it blind?* Partly. Which library code is present and where its blocks start are found automatically (share of violated
  parity checks: random bits 50 %, right code and offset below 40 % and 7 standard deviations away). H itself is not recovered:
  built-in demo codes, or the operator uploads the standard matrix (alist, dense 0/1, row-col pairs).
- *Are your built-in LDPC codes the CCSDS / 5G ones?* No. They are our own rate-1/2 IRA codes (256/128, 512/256, 1024/512).
  Say so. Standard codes work once their matrix is uploaded.
- *How strong is it?* On our simulator LDPC(512,256) recovers the whole message up to ~3 to 4 % raw bit errors; RS(204,188)+LDPC
  about the same; both fail by 6 %. Hard decisions only (soft input would gain 1.5 to 2 dB). Not a real-capture result.
- *Why a PDF report?* An analyst can hand on, file or attach one document: capture details, measured parameters with plots,
  modulation and quality, decoded FEC and framing, recovered message, all warnings and a statement of the method's limits.

## Machine learning
- *Why a Random Forest first?* Trains in seconds, explainable features, 98 % on its own simulator (an upper bound).
- *What is the CNN for?* Independent second opinion trained on public RadioML 2018. 96 % at 10 dB and above on RadioML test frames.
- *Does it work on your signals?* Only partly: it needs noise conditioning and agrees with ground truth on 71 % of our simulated captures
  (optimistic, tuned on the same simulator). So it is informational only. Say this yourself.
- *Why is overall RadioML accuracy only 60 %?* It includes -20 to 0 dB frames where no classifier can succeed.

## Honest weak points
- BPSK is mistaken for FSK in a noticeable share of captures (see PHASE4_EXPLAINED section 2).
- A decoder bug (complemented rotation chosen on large captures) was found by our evaluation and fixed; regression-tested.
- Transmit chain and receiver were written by the same developer: a shared bug would be invisible. Real captures are the fix.
- Sampling rate of raw IQ cannot be recovered from the file.
- Interleaver type and size are entered by the operator; only the start offset is found.
- QAM is 16QAM only; 64QAM and above are not supported.
- LDPC needs the parity-check matrix unless the code is one of our three built-in demo codes.

# Phase 3 explained (what / why / how) + finals Q&A

## What the decoder does, in order (backend/core/decode.py)
1. **Bit-mapping hypotheses.** Demodulation leaves an M-fold phase ambiguity and the bit labelling (natural or Gray) is unknown. We build every candidate bit stream (e.g. QPSK: 4 rotations x 2 labellings) and score each for structure.
2. **De-interleaving** (only if the operator supplies the type and size): try every start offset, keep the one where structure appears.
3. **Convolutional code, blind** (`fec/conv.py`): find the rate, constraint length and generator polynomials, then Viterbi-decode.
4. **Frame detection** (`framing.py`): frame length, sync word, header / payload split.
5. **Reed-Solomon, blind** (`fec/rs.py`): find (n, k), field polynomial and first root, then decode every codeword.
6. **Polarity + message**: pick the polarity that matches a known sync word or reads as text; extract frames and the payload.

## Key ideas (these are the answers to "how did you do that?")
**Blind convolutional-code identification.** For a rate-1/2 code, outputs are y1 = g1*x and y2 = g2*x (GF(2) convolution). So g2*y1 = g1*y2 (both equal g1*g2*x). For the right pair the sequence g2*y1 + g1*y2 is all zeros. We compute g*y for EVERY candidate g (up to K=9: 511 of them) in one dynamic-programming pass, turn them into +/-1 vectors, and one matrix product compares all pairs. The pair with correlation near 1 is the code; noise only lowers it a little; a wrong pair scores about 0.05. Multiples of the true pair also satisfy the relation, so we take the smallest. Rate 1/3 repeats the idea with a third stream.

**Viterbi.** Hard-decision, 2^(K-1) states, add-compare-select per step, traceback from the best final state. The re-encoded decoded bits are compared with what was received to estimate the channel bit error rate (shown in the UI).

**Reed-Solomon blind identification.** Every codeword is a multiple of g(x) = (x-a^fcr)...(x-a^(fcr+2t-1)), so it evaluates to zero at 2t consecutive powers of a. Evaluate each codeword at all 255 powers and look for a run of zeros: start = first root, length = 2t = n-k. Repeat for each of the 16 primitive field polynomials. With symbol errors the zeros vanish, so a second method uses the fact that inside the parity roots the syndromes of a corrupted word form a low-complexity sequence (Berlekamp-Massey length = number of errors) obeying one fixed recurrence; we find where that recurrence stops predicting. Own GF(256) arithmetic, encoder and Berlekamp-Massey / Chien / Forney decoder; the encoder is checked against the `reedsolo` library in the tests.

**Frame length by folding.** For every candidate length L stack the stream into rows of L bits and measure how many columns hold the same bit in every row. The statistic jumps at the true frame length because the sync word repeats. Multiples of the true length also jump, so we test divisors. ASCII text adds constant columns (bit 7 = 0) at every multiple of 8, so each length is compared with its neighbours of the same residue mod 8.

**Header vs payload.** Right after the sync word, columns that are constant, slowly changing (counter bits) or predictable from the previous two frames are header; random-looking columns are payload. The boundary is rounded to a byte.

**Why the structure score is invariant to polarity.** XOR-ing the stream with a constant changes the check sequence only by a constant, so the +/-1 correlation (taken in absolute value) is unchanged. That is why several candidate rotations score the same in the UI table; the final polarity is chosen from a known sync word or from how text-like the payload is.

**Modulation verification (added in this phase).** Structured data (text, constant headers) biases symbol statistics and can fool the Phase 2 classifier, which assumes near-random data. If the symbol SNR looks poor, every modulation hypothesis is tried and the lowest-order one that fits the constellation about as well as the best wins. In testing this fixed a 16QAM signal being called BPSK.

## Honest limits (say these before a judge finds them)
1. **Tested only on our own generated signals.** The transmit chain (chain.py) and the receiver are written by us, so the 100 % message recovery in the tests is evidence the algorithms work, not that they work on real operational captures. Real captures need testing before any claim.
2. **LDPC was not implemented in this phase.** It was added in Phase 5 (see `PHASE5_LDPC.md`).
3. **Interleaver type and size are operator-supplied.** We find the start offset blindly, but not the type or size. Pseudo-random interleavers also need the seed.
4. **Frames need a repeating sync word, and enough of them.** Roughly 12 or more frames for long frames (e.g. RS(255,223) needs about 12 x 2040 bits). Streams with no sync word cannot be framed or aligned for Reed-Solomon.
5. **Convolutional codes:** rate 1/2 and 1/3, constraint length up to 9, no puncturing, hard-decision decoding only, soft decisions would gain about 2 dB.
6. **Reed-Solomon:** GF(256), consecutive roots with step 1, codeword aligned to a frame start (sync word at the start of the codeword). No dual-basis / conventional-basis conversion (CCSDS), no RS interleaving depth > 1.
7. **No scrambler / whitening handling.** Many real systems scramble the stream; that is not undone here.
8. **Header fields that look random** (CRC, timestamps) are classed as payload.
9. Decoding can take 5-30 s on long captures; interleaver alignment is the slow part.

## Likely questions
- **How do you find the code without being told?** Show the relation g2*y1 = g1*y2 and the matrix of correlations.
- **Why do you get (171,133)?** It is the CCSDS / NASA standard K=7 code; several equivalent codes exist after a linear re-labelling of bits, so the decoder prefers a textbook one.
- **What if there are too many errors?** Viterbi output degrades; Reed-Solomon reports uncorrectable words (shown in the UI) instead of returning wrong data.
- **How do you know the header boundary?** From predictability across frames; it is a measurement, not a guess, and it is rounded to a byte.
- **Why not deep learning here?** The structure is algebraic; exact identification is both faster and explainable.

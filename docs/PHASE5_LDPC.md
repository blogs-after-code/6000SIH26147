# Phase 5: LDPC decoding (explained simply)

## What an LDPC code is
A message is split into blocks of k bits. For each block the sender adds n-k extra "parity" bits, so every block sent
is n bits long. The extra bits obey many small rules (parity checks): "bits 4, 19, 77 and 130 must add up to an even
number". The list of all rules is the **parity-check matrix H**. The receiver checks the rules; broken rules point at
the bit errors, and a belief-propagation decoder repairs them.

## What the tool does with it (pipeline step 3b)
Order in the receiver: demodulate -> (de-interleave) -> convolutional decode -> **LDPC** -> framing -> Reed-Solomon.
This mirrors a sender that did frames -> RS -> LDPC -> convolutional -> interleaver.

1. **Which code, and where does a block start?** For every code in the library and every possible start position, the
   tool counts the share of parity checks a block breaks. Random bits break about half. The right code at the right
   position breaks far fewer (for example 12 % at 3 % bit errors). A hit needs less than 40 % broken AND a gap of
   at least 7 standard deviations from random. This also picks the correct bit labelling and phase rotation of the
   demodulated stream.
2. **Decode.** Normalised min-sum belief propagation (scale 0.8, up to 60 iterations) on the hard bits. Blocks that
   still break rules are counted and reported ("did not converge").
3. **Hand on.** The first k bits of each block (systematic code) go on to framing and Reed-Solomon.

## What is blind and what is not
| Part | Status |
|---|---|
| Which library code is present | found automatically |
| Where the first block starts | found automatically |
| Bit labelling / phase ambiguity | found automatically (the broken-checks share also ranks the hypotheses) |
| The matrix H itself | **must be known**: built-in demo codes, or uploaded by the operator |
| Soft decisions | not available; the decoder takes hard bits |

## Built-in codes
`ldpc_256_128`, `ldpc_512_256`, `ldpc_1024_512`: rate 1/2 IRA codes (dual-diagonal parity part, information columns
of weight 3 chosen with a seeded construction that avoids 4-cycles). They are NOT CCSDS, 5G, Wi-Fi or DVB-S2 codes
and must never be described as such. Their matrices can be downloaded in alist form from the Decoding tab
(`/api/ldpc-codes/<name>.alist`).

## Uploading a matrix
Decoding tab -> LDPC codes -> Choose matrix file. Accepted: alist; plain 0/1 text, one matrix row per line; or lines of
`row col` pairs (0-based, optional `rows cols` header). The code must be systematic with the information bits first
(k = columns - rows). The decoder re-runs automatically. API: `POST /api/ldpc/{session}` (multipart `file`),
`DELETE /api/ldpc/{session}`.

## Measured (own simulator, hard decisions)
See `results/decode_vs_ber.png`. Typical: LDPC(512,256) fully recovers the message up to about 5 % raw bit error rate;
the min-sum decoder fails around 7 %. Reed-Solomon outside LDPC extends the range slightly. These numbers are for our
own transmit chain, not for real captures.

## Tests (`backend/tests/test_phase5_ldpc.py`)
Valid and systematic encoding; error correction; offset search and rejection of random data; alist and dense round trip;
end-to-end decoding of LDPC, LDPC + convolutional and RS + LDPC chains; an uploaded matrix that is unknown to the
built-in library; no false LDPC detection on uncoded, convolutional or RS-only streams.

## Honest limits
* Standard-code matrices are not shipped; upload them.
* Hard-decision input costs about 1.5 to 2 dB against soft decisions.
* Non-systematic codes and codes with punctured/shortened blocks are not handled.
* Frame structure inside the LDPC payload is found the usual way and still needs a repeating sync word.

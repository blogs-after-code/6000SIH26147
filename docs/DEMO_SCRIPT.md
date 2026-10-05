# Demo video script (4 to 5 minutes)

Feature map to say out loud (the landing page has the same five cards): i. signal parameters, ii. demodulation FSK/PSK/QAM,
iii. de-interleaving block/convolutional/diagonal/pseudo-random, iv. FEC convolutional+Viterbi / RS / concatenated / LDPC,
v. bit-stream correlation (frame length, sync word, header vs payload).

| Time | Screen | Say |
|---|---|---|
| 0:00 | Landing page | Problem: raw HF/VHF/UHF captures are analysed by hand; parameters are often unknown. |
| 0:25 | Analyzer, preset "Reed-Solomon + convolutional", Generate | Upload or generate a capture. Server measures centre frequency, bandwidth, SNR from the spectrum. |
| 0:55 | Spectrum + waterfall | Detected band highlighted; noise floor and SNR measured, not assumed. |
| 1:20 | Demodulate | Symbol rate from the |x|^2 line, modulation classified, constellation shown. Point at EVM and the warnings. |
| 1:55 | Models tab | Two opinions: feature classifier and CNN trained on RadioML. CNN is informational; explain why. |
| 2:20 | Decode | Convolutional code 171/133 found blind, frame structure with CCSDS sync word, RS(204,188), recovered text. |
| 2:50 | Preset "Reed-Solomon + LDPC", Decoding tab | LDPC code and block start found automatically; show "checks violated before decoding" against 50 % for random bits. Mention: other LDPC codes need their matrix uploaded. |
| 3:20 | **Download PDF report** button (top of results) | The key feature: whole analysis in one document. Open the PDF. |
| 3:40 | Limits slide/landing text | Simulator-validated only; LDPC needs the matrix for non-built-in codes; interleaver supplied by operator. Close. |

# Validating on a real or public capture (P0 item 2; needs you, not possible from the sandbox)

1. Get a capture with known parameters: an RTL-SDR recording of a known signal, or a public SigMF/IQ file with its metadata.
2. Convert if needed to cf32/ci16/ci8/cu8 or wav. Note the true sampling rate.
3. Upload in Analyzer, enter the rate for raw IQ. Run Demodulate (and Decode if coded).
4. Record in a table: true vs measured centre, bandwidth, symbol rate, modulation; CNN opinion; what failed.
5. Screenshot every stage. Report failures plainly and update section 9 of PROJECT_HANDOFF.md.
Expect problems (frequency drift, fading, non-integer samples/symbol); a documented failure is a valid result.

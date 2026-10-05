# Phase 1 explained (what / why / how) + finals Q&A

## Files
| File | Job |
|---|---|
| `backend/core/generator.py` | Makes test signals with known parameters (the answer key) |
| `backend/core/loader.py` | .wav / raw IQ -> complex samples + fs |
| `backend/core/spectral.py` | PSD, waterfall, centre freq, bandwidth, SNR |
| `backend/main.py` | REST API: `/api/analyze`, `/api/demo` |
| `frontend/*` | Upload, parameter table, PSD plot, waterfall (plain HTML/CSS/JS, no libraries) |

## Core ideas
**IQ samples.** A radio signal is stored as complex numbers I + jQ. The complex form keeps
phase and lets us tell +f from -f. A mono WAV is real-only, so we build the complex
(analytic) signal with the Hilbert transform.

**Why a generator first.** Real files have no answer key. With a generator we can say
"SNR was 12 dB, offset 5 kHz" and measure how close the estimator gets. That is how
the accuracy graphs are made.

**Generator chain (linear mods).** bits -> group into symbols -> map to constellation ->
insert zeros between symbols (upsample by sps) -> root-raised-cosine filter -> multiply by
e^(j2*pi*f0*t) (carrier offset) -> add Gaussian noise. FSK instead sets the frequency per
symbol and integrates it to phase, so phase is continuous.

**Why RRC.** It limits bandwidth to about Rs*(1+beta). Matched RX + TX RRC = raised cosine,
which has no inter-symbol interference at the sampling instants.

**PSD (Welch).** One FFT of noisy data is itself noisy. Welch cuts the data into overlapping
Hann-windowed blocks, FFTs each, averages the power. More averages = smoother estimate.

**Noise floor.** Start at the 20th percentile of PSD bins, keep only bins below 2x that,
take their mean, repeat 4x. Signal bins get excluded so they do not bias the floor.

**Outputs.** Signal power = sum of (PSD - floor) over bins clearly above the floor.
Centre = power-weighted mean frequency. Bandwidth = width holding 99% of that power.
SNR = signal power / noise power (full-band = over all fs; in-band = over the signal BW).

**Waterfall.** Same FFT, repeated over time; rows = time, columns = frequency, colour = power.
Shows bursts, hopping, drifting carriers that a single PSD would average away.

## Honest limits (say these yourself before a judge finds them)
1. Raw IQ has no header, so **absolute sampling rate cannot be recovered from the file**; the
   operator supplies it. Later phases estimate samples-per-symbol (fs / symbol rate), not fs.
2. 99% bandwidth reads about 10-15% low on our QPSK test: edge bins fall below the 2x-floor
   threshold and are not counted. It is a known bias, accepted for robustness.
3. If the signal fills nearly the whole band, no empty bins exist to measure the noise floor.
4. Only the first 2M samples are analysed.

## Likely questions
- **Why complex IQ and not real samples?** Preserves phase, distinguishes +/- frequency, needs half the rate for the same bandwidth.
- **Why Welch instead of one big FFT?** Variance. A single periodogram's error does not shrink with more data; averaging does.
- **Why does SNR differ full-band vs in-band?** Noise power scales with bandwidth; in-band SNR is higher by about fs / BW.
- **Why do you need fs supplied for .iq?** Headerless format; fs is metadata that is not in the samples.
- **How do you know your estimates are right?** Generator ground truth + pytest assertions on centre (+/-600 Hz), SNR (+/-2 dB), bandwidth (+/-20%).

## Phase 2 preview
Symbol-rate estimation (cyclostationary spectral lines), carrier + timing recovery (Costas, Gardner),
constellation plot, demodulation, modulation classifier (features + Random Forest locally; optional CNN on Colab/Kaggle).

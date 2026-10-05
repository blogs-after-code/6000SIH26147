# Phase 6: showcase sections, model highlight and mobile layout

## What changed
- Landing page order: hero, `#problem`, `#solution`, `#coverage`, `#model`, `#report`, `#how`, `#evidence`, `#limits`.
- `#solution` lists each line of the expected solution next to the view that delivers it, with real screenshots (`frontend/assets/shots/*.webp`) taken from the running analyzer on a generated test signal.
- `#model` shows the CNN (96.2% at 10 dB SNR and above, 60.5% over all SNRs, RadioML 2018.01A, 24 classes) and the Random Forest (98.1%, 4 classes), the training recipe, the training pipeline, per-class accuracy and the caveat that the CNN agrees with truth only 71% on our own simulator. Numbers are read from `/api/models` (now with `cnn_transfer` and `cnn_size_bytes`), not typed into the page.
- Colours: orange stays the main accent. `--blue` marks the CNN and data, `--green` marks the feature classifier and passing results. Tokens live in `css/style.css`.

## Mobile
- `css/responsive.css` (breakpoints 1100, 860, 600, 400), `js/nav.js` (hamburger menu below 860 px).
- Checked with Playwright at 360, 390, 768 and 1366 px in dark and light themes: no horizontal scroll on the landing page or any analyzer tab. Not tested on a physical phone.

## Recapturing screenshots
Run the server, open `#/analyze`, choose the LDPC + RS preset at 14 dB, run it, and screenshot the cards named in `index.html`.

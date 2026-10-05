# AGENTS.md: rules for AI agents working on this repository

Full context is in `PROJECT_HANDOFF.md`. Read it before changing anything. Summary of the rules:

1. **Run the tests first and last:** `python -m pytest backend/tests -q` (80 should pass, about 80 s).
2. **Never weaken a test or an assertion to get green.** If a threshold is wrong, explain why in the change.
3. **Never present simulated results as real-world performance.** Everything validated so far uses our own generator.
   Label sources of every number shown in the UI or docs.
4. **New algorithm = code + known-answer test (use `backend/core/generator.py` and `chain.py`) + entry in `docs/`**
   (what, why, how, honest limits). The developer must be able to defend it at the finals.
5. **Frontend:** plain HTML, CSS and JS in **separate files**, no build step, colours only via CSS variables, high text
   contrast, no government branding. API field names are a contract with `frontend/js/app.js`.
6. **Communication with the developer:** short, direct, critical feedback; numbered steps with exact commands; do not pad.
7. **Weak laptop:** no local model training. Training goes to Kaggle; inference stays light (ONNX Runtime, CPU).
8. **Do not invent facts** about the problem owner, datasets, or results. Unknowns go in the handoff's limits section.

Quick commands
```bash
pip install -r requirements.txt
python -m pytest backend/tests -q
python scripts/make_samples.py
uvicorn backend.main:app --reload          # http://127.0.0.1:8000
python scripts/train_classifier.py --n 500 --test 120     # retrain the Random Forest (minutes, CPU)
python kaggle/make_notebook.py             # regenerate the notebook from kaggle/train_modulation_cnn.py
```
Where things are: backend/core (signal processing), backend/core/fec (codes), backend/main.py (API), frontend/ (UI),
kaggle/ (CNN training), docs/ (algorithm explanations), scripts/ (tools).

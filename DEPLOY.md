# Deploying SanketSetu (demo link for the judges)

Backend (FastAPI) on **Render**, front end (static) on **Vercel**. Both watch the GitHub repo, so every `git push` to `main` redeploys.

```
Judges' browser -> Vercel (frontend/)  --API calls-->  Render (backend/, models, signal processing)
```

## 1. Push to GitHub (once)
Run inside the `signal-lab` folder (this folder must be the repo root, so `render.yaml` is at the top):
```bash
git init
git add .
git commit -m "SanketSetu: deployable version"
git branch -M main
# create an empty repo on github.com first, then:
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
```
`.gitignore` already keeps `venv/`, caches and `.env` out. The trained models in `backend/models/` ARE committed (the app needs them).

## 2. Backend on Render
1. render.com > **New + > Blueprint** > connect GitHub > pick the repo. It reads `render.yaml` and creates `sanketsetu-api`.
2. Wait for the build (about 5 to 10 minutes the first time). Open `https://<service-name>.onrender.com/api/health`: you should see `{"status":"ok",...}`.
3. That same URL, opened without `/api/health`, already serves the full app. Keep it as your **backup demo link**.
4. Copy the service URL (no trailing slash). If the name `sanketsetu-api` was taken, Render adds a suffix.

## 3. Front end on Vercel
1. vercel.com > **Add New > Project** > import the same repo.
2. **Root Directory: `frontend`** (important). Framework Preset: Other. Leave build settings alone; `frontend/vercel.json` sets them.
3. **Environment Variables:** `API_BASE_URL` = the Render URL from step 2 (e.g. `https://sanketsetu-api.onrender.com`, with `https://`, no trailing slash).
4. Deploy. The `*.vercel.app` URL is your **main demo link**. CORS for any `*.vercel.app` address is already allowed in the backend.

## 4. Updates
Edit code, then `git add . && git commit -m "..." && git push`. Render rebuilds the backend and Vercel rebuilds the front end by themselves (Render about 3 to 8 minutes, Vercel under a minute). Pull requests / other branches get Vercel preview URLs automatically; only `main` goes live on Render.
If you change the backend URL, update `API_BASE_URL` in Vercel and click Redeploy.

## 5. Before showing the judges (read this)
- **Free Render sleeps after 15 minutes idle** and takes about a minute to wake. The page pings the server as soon as it opens and shows "server waking up". Open the link yourself 5 minutes before the judges do and run one demo.
- **Free Render has 0.1 CPU.** Decoding takes 5 to 30 s on a laptop and may be several times slower there. Test the Decode step on the deployed link. If it is too slow, switch the service to **Starter ($7/month, 0.5 CPU, no sleeping)** in Render > Settings > Instance Type, and cancel after judging.
- Sessions live in the server's memory (last 8). A restart or redeploy loses them: re-run the analysis, do not push code during the judging window.
- Uploads: stay with small files (the `samples/` ones). Free memory is 512 MB.
- Everything shown is simulated or public-dataset data, same as your docs. Say so.

## Local run (unchanged)
`pip install -r requirements.txt && uvicorn backend.main:app --reload` opens the whole app on http://127.0.0.1:8000.

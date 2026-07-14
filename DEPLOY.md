# Deploying to Streamlit Community Cloud

The app (`app/app.py`) reads only precomputed artifacts already checked into
this repo — `config/universe.json`, `results/final_comparison.csv`, and
`results/predictions/*.csv`. It never retrains a model or calls the Yahoo
Finance API, so there is **no secrets/config to set up** and page loads stay
fast regardless of traffic.

## One-time setup

1. **Push this repo to GitHub** (already done —
   [pranavchauhann/sector-comparative-equity-lstm](https://github.com/pranavchauhann/sector-comparative-equity-lstm),
   public, required for the free tier).
2. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in /
   connect your GitHub account (already done).
3. Click **"New app"**, then:
   - **Repository:** `pranavchauhann/sector-comparative-equity-lstm`
   - **Branch:** `main`
   - **Main file path:** `app/app.py`
4. Open **"Advanced settings"** before deploying and set:
   - **Python dependencies file:** `app/requirements.txt`

     This matters: the repo-root `requirements.txt` includes the full
     notebook stack (TensorFlow, `pandas-ta`, `statsmodels`, Jupyter — some
     of which require Python ≥3.12 and add real build time/size) needed to
     *reproduce the analysis*, but the deployed app itself only needs
     `streamlit`, `pandas`, and `plotly` to read the precomputed CSVs. Using
     the lean `app/requirements.txt` keeps the Cloud build fast and avoids
     dragging in dependencies the app doesn't touch at runtime.
   - **Python version:** 3.11 or 3.12 both work for the lean app file (only
     the notebook stack's `pandas-ta` needs ≥3.12).
5. Click **Deploy**.

## Secrets / config

None required. All data the app reads is static and already committed:

- `config/universe.json` — locked, dated ticker universe
- `results/final_comparison.csv` — precomputed metrics for all 4 models × 40 stocks
- `results/predictions/*.csv` — precomputed test-period predictions per stock

If any of these files are missing from the repo (check `.gitignore` — only
`data/` is excluded; `config/` and `results/` are intentionally tracked), the
app will error on load rather than silently trying to regenerate them live.

## After deploying

1. Confirm the public URL loads and the sector/stock dropdowns populate.
2. Spot-check a couple of stocks per sector — chart renders, metrics panel
   shows non-null values, the model comparison expander and sector summary
   table both render.
3. Record the live URL in the README's "Live demo" section.

## Redeploying after changes

Streamlit Community Cloud auto-redeploys on every push to `main`. If you
change `results/*.csv` (e.g. by re-running a notebook), just commit and push
— no separate action needed on the Streamlit Cloud side.

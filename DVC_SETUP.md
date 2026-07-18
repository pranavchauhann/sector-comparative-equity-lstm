# DVC Setup — Data & Model Versioning

**Phase 1 of the MLOps upgrade.** This document explains what DVC does in this
repo, how to pull the tracked data/models after cloning, and how the pipeline
DAG works — including proof that `dvc repro` correctly re-runs only the
stages a change actually affects.

## Why DVC, concretely

Before this phase, "which exact data was this model trained on?" had no real
answer — `data/raw/`, `data/processed/`, and the trained models were all
gitignored and regenerated ad hoc. DVC version-controls those large
files the same way Git version-controls code: a small `.dvc`/`dvc.lock`
pointer file goes into Git, the actual bytes go into DVC's own
content-addressed cache and a remote (Google Drive here), and `dvc pull`
resolves a pointer back into a real file.

## What's tracked, and how

| Path | Tracked via | Contents |
|---|---|---|
| `data/raw/` | pipeline output (`fetch_data` stage) | 40 tickers' cached OHLCV, 5yr |
| `data/processed/` | pipeline output (`engineer_features` stage) | 40 tickers' engineered features |
| `models/baselines/` | pipeline output (`train_baselines` stage) | per-stock `LinearRegression` + `ARIMA` fits (joblib `.pkl`) |
| `models/lstm/` | pipeline output (`train_lstm` stage) | per-stock returns-target LSTM (`.keras`) |
| `results/baseline_metrics.csv`, `results/lstm_metrics.csv` | pipeline output, `cache: false` | metrics CSVs — deliberately kept in **Git**, not DVC (see below) |

**Deliberate deviation from "just DVC-add everything":** `data/raw` and
`data/processed` are **pipeline outputs** (`dvc.yaml` `outs:`), not manually
`dvc add`-ed static files. Pipeline outputs are automatically versioned by
`dvc repro`/`dvc commit` — running `dvc add` on a path that's already a stage
output would conflict with DVC ("output already tracked"). This is the
DVC-idiomatic way to track a pipeline's data, and it's what makes selective
re-execution (below) possible.

**Deliberate deviation #2:** `results/baseline_metrics.csv` and
`results/lstm_metrics.csv` are declared in `dvc.yaml` as `metrics:` with
`cache: false`. This means DVC still tracks them as *pipeline outputs* for
staleness detection (if `train_lstm`'s deps change, `dvc status` correctly
flags `results/lstm_metrics.csv` as stale) — but the file content itself stays
in **Git**, not DVC's cache. This project's prior phases treated `results/`
CSVs as portfolio deliverables readable directly on GitHub; moving them into
DVC's cache would hide them behind a `dvc pull`. `cache: false` gets DVC's DAG
benefits without that tradeoff.

**ARIMA model size, honestly:** each `models/baselines/<ticker>_arima.pkl` is
~8MB (linreg pickles are ~1KB). This is inherent to `statsmodels`'
`ARIMAResultsWrapper` — its state-space Kalman filter/smoother results retain
substantial diagnostic arrays even for a single `.fit()` call with no
walk-forward appends (verified: a bare fit pickles to the same ~8.5MB).
`statsmodels` offers `.remove_data()` to shrink this, but it strips the
`endog` array the model needs to determine its forecast index — trying it
broke `.forecast()` entirely. Since ~320MB total for 40 ARIMA models lives
entirely in DVC's cache/remote (never touches Git), this is exactly the kind
of size DVC exists to absorb; it was not worth breaking forecasting to save
space Git will never see.

## Pipeline DAG

```
                  +------------+
                  | fetch_data |
                  +------------+
                         *
                         *
                         *
              +-------------------+
              | engineer_features |
              +-------------------+
                ***            ***
              **                  **
            **                      **
+-----------------+              +------------+
| train_baselines |              | train_lstm |
+-----------------+              +------------+
```
(exact `dvc dag` output, `dvc.yaml` stages)

- **`fetch_data`** — `scripts/fetch_data.py`. Deps: itself, `src/data_loader.py`,
  `src/universe.py`, `config/universe.json`. Out: `data/raw/`.
- **`engineer_features`** — `scripts/engineer_features.py`. Deps: itself,
  `src/features.py`, `src/scaling.py`, `data/raw/`, `config/universe.json`.
  Out: `data/processed/`.
- **`train_baselines`** — `scripts/train_baselines.py`. Deps: itself,
  `src/baselines.py`, `src/evaluate.py`, `src/scaling.py`, `data/processed/`,
  `config/universe.json`. Outs: `models/baselines/`,
  `results/baseline_metrics.csv` (metric, uncached).
- **`train_lstm`** — `scripts/train_lstm.py`. Deps: itself, `src/lstm_model.py`,
  `src/evaluate.py`, `src/scaling.py`, `data/processed/`, `config/universe.json`.
  Outs: `models/lstm/`, `results/lstm_metrics.csv` (metric, uncached).

`train_baselines` and `train_lstm` **fan out in parallel** from
`engineer_features` — neither depends on the other, so changing one never
invalidates the other. This is the property proven below.

## Proof: selective re-execution works

This is the part that separates "added a config file" from "the DAG actually
tracks dependencies correctly." Starting from a fully up-to-date pipeline
(`dvc repro --dry` reports all 4 stages unchanged), one line was added to
`scripts/train_baselines.py` (an extra ARIMA fallback order):

```diff
- ARIMA_ORDERS = [(5, 1, 0), (2, 1, 0), (1, 1, 0), (1, 1, 1)]
+ ARIMA_ORDERS = [(5, 1, 0), (2, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1)]
```

`dvc status` immediately and correctly identified the *only* affected stage:

```
train_baselines:
	changed deps:
		modified:           scripts/train_baselines.py
```

`dvc repro` then produced this — the actual terminal output:

```
Stage 'fetch_data' didn't change, skipping
Stage 'engineer_features' didn't change, skipping
Running stage 'train_baselines':
> .venv/bin/python scripts/train_baselines.py
...
Stage 'train_lstm' didn't change, skipping
```

**`train_lstm` was correctly skipped** even though it's a "sibling" stage
that also consumes `data/processed/` — DVC reasoned that it has no dependency
on `src/baselines.py` or `scripts/train_baselines.py`, so it stayed untouched.
That's the ~7-minute LSTM training stage *not* re-run for a change that has
nothing to do with it — the entire point of dependency-aware reproducibility.

## Push/pull round-trip — verified

Google Drive requires an interactive OAuth flow (a browser + a Google
account) that can't be completed in a headless setup, so the push/pull
*mechanics* were verified end-to-end against a local DVC remote first:

1. `dvc push` → `204 files pushed`.
2. Simulated a fresh clone: deleted `.dvc/cache/`, `data/raw/`,
   `data/processed/`, `models/` entirely.
3. `dvc pull` → `204 files fetched and 200 files added`. Every raw CSV,
   processed CSV, baseline pickle, and LSTM `.keras` file came back byte-for-byte.
4. `dvc status` → `Data and pipelines are up to date.` — no retraining
   needed; the pulled artifacts satisfy `dvc.lock` exactly.

The repo's actual remote is configured for **Google Drive** (see below) —
the local-remote test above only proved the DVC mechanics work; you still
need to authenticate your own Google account to push/pull the real remote.

## Setting up your own Google Drive remote

```bash
# 1. Create (or pick) a Google Drive folder to hold the DVC cache.
#    Open it in a browser; the folder ID is the last segment of its URL:
#    https://drive.google.com/drive/folders/<FOLDER_ID>

# 2. Point the existing gdrive remote at your folder (replace the ID):
dvc remote modify gdrive url gdrive://<YOUR_FOLDER_ID>

# 3. Push. The first push opens a browser for Google OAuth consent —
#    authenticate once, and DVC caches the token locally.
dvc push
```

If you'd rather not use Google Drive, any DVC remote type works the same way
— e.g. a local path (`dvc remote add -d local /path/to/remote`) or S3
(`dvc remote add -d s3remote s3://bucket/path`, needs AWS credentials).

## Reproducing the whole pipeline from scratch

```bash
git clone https://github.com/pranavchauhann/sector-comparative-equity-lstm.git
cd sector-comparative-equity-lstm
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Option A: pull already-trained data/models (fast, seconds)
dvc pull

# Option B: regenerate everything from raw data (slow, ~10-12 min — fetches
# fresh 5yr data, re-engineers features, refits baselines, retrains 40 LSTMs)
dvc repro
```

`dvc repro` is always safe to run — any stage whose dependencies haven't
changed since the last run is skipped automatically (see the proof above).

## Everyday commands

| Command | What it does |
|---|---|
| `dvc dag` | Print the pipeline stage graph |
| `dvc status` | Show which stages/files are stale vs. `dvc.lock` |
| `dvc repro` | Re-run only the stale stages |
| `dvc repro <stage>` | Re-run one stage (and its stale dependents) |
| `dvc push` | Upload the local cache to the configured remote |
| `dvc pull` | Download tracked files from the remote into the local cache/workspace |
| `dvc metrics show` | Print `results/*.csv` metrics (cache:false files still tracked as metrics) |

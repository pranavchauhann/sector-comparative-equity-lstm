# Experiment Tracking with MLflow (Phase 2)

Phase 1 (DVC) made the pipeline reproducible. Phase 2 makes it **comparable**:
every training run — baseline or LSTM, with any hyperparameter change — is
now automatically logged to MLflow, so no result is ever lost when a retrain
overwrites `results/*.csv`, and any two runs can be compared side by side.

## Why MLflow *and* DVC — they are not redundant

| | DVC (Phase 1) | MLflow (Phase 2) |
|---|---|---|
| **Tracks** | Data + model *artifacts* and their lineage (which code + data hashes produced them) | *Experiments*: which hyperparameters produced which metrics |
| **Answers** | "Can I reproduce exactly this model?" | "Which of my 12 configurations was best, and why?" |
| **Unit** | Pipeline stage output | Training run |
| **Storage** | Content-addressed cache, remotable | Local SQLite (`mlflow.db`) + artifact dir (`mlruns/`) |

The same training event feeds both: running `dvc repro train_lstm` still
writes the DVC-tracked `models/lstm/` + `results/lstm_metrics.csv` exactly as
in Phase 1, **and** logs one MLflow run per stock as a side effect. DVC
guarantees the artifact is reproducible; MLflow remembers what was tried and
how it scored. Delete `mlruns/` and the pipeline still reproduces; delete the
DVC cache and MLflow still remembers every experiment's numbers.

## Launching the dashboard

```bash
source .venv/bin/activate
mlflow ui --backend-store-uri sqlite:///mlflow.db   # from the repo root
# open http://localhost:5000
```

> MLflow 3.x put the plain `./mlruns` filesystem backend into maintenance
> mode, so this project uses the recommended local SQLite store. Both
> `mlflow.db` (run metadata) and `mlruns/` (logged artifacts) are gitignored —
> they are per-machine experiment history, not repo state.

## What gets logged

**One MLflow experiment per sector** (Information Technology, Banking &
Financial Services, Energy, FMCG) — so a sector's baseline and LSTM runs sit
side by side rather than 160+ runs landing in one flat list.

Every **LSTM** run ([scripts/train_lstm.py](../scripts/train_lstm.py)) logs:
- **Params**: `seq_length`, `units`, `dropout`, `learning_rate`, `batch_size`,
  `max_epochs`, `patience`, ticker, sector, target type, feature count
- **Metrics**: `train_loss` / `val_loss` **per epoch** (so training curves are
  comparable in the UI), plus final test `RMSE`, `MAE`, `MAPE`,
  `directional_accuracy`, `epochs_trained`
- **Artifacts**: the training/validation loss-curve plot (`loss_curve.png`)
  and the trained Keras model (`mlflow.tensorflow.log_model`)

Every **baseline** run ([scripts/train_baselines.py](../scripts/train_baselines.py))
logs one run per (stock, model):
- **Params**: model type (Naive / LinearRegression / ARIMA), ARIMA order used,
  ticker, sector
- **Metrics**: `RMSE`, `MAE`, `MAPE`, `directional_accuracy`
- **Artifacts**: the fitted model — `mlflow.sklearn.log_model` for
  LinearRegression; the joblib pickle for ARIMA (MLflow 3 dropped the
  statsmodels flavor); none for the parameter-free naive forecast

## Reproducing the comparison runs

The DVC pipeline produces the canonical seq-30 runs for all 40 stocks:

```bash
dvc repro          # trains everything stale; each stock logs an MLflow run
```

The hyperparameter sweeps use `--sweep` mode: runs log to MLflow but do
**not** overwrite the DVC-tracked `models/` and `results/` outputs — that
separation is what keeps the two systems from fighting over the same files.

```bash
# 3 LSTM configurations on one representative stock per sector:
python scripts/train_lstm.py --sweep --seq-length 15 \
    --tickers TCS.NS HDFCBANK.NS RELIANCE.NS HINDUNILVR.NS
python scripts/train_lstm.py --sweep --seq-length 60 \
    --tickers TCS.NS HDFCBANK.NS RELIANCE.NS HINDUNILVR.NS
# (seq-length 30 comes from the dvc repro above)
```

Other knobs work the same way, e.g.
`--units 32`, `--dropout 0.3`, `--learning-rate 0.0005`.

## Comparing runs in the UI

1. Open an experiment (e.g. *Information Technology*), filter the run list
   (e.g. `params.ticker = 'TCS.NS'`), tick the runs to compare, and press
   **Compare** — the comparison view shows params and metrics side by side
   and a parallel-coordinates plot of the hyperparameters against metrics.
2. To compare an LSTM run against a baseline, tick e.g. `LSTM-TCS.NS-seq30-u64`
   and `ARIMA-TCS.NS` / `Naive-TCS.NS` in the same experiment — they share the
   same metric names (`RMSE`, `MAE`, `MAPE`, `directional_accuracy`) on the
   same test split, so the comparison is apples-to-apples.

Screenshots of both views: [results/mlflow_screenshots/](../results/mlflow_screenshots/).

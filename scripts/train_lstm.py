"""
scripts/train_lstm.py
=======================
DVC pipeline stage 4: train_lstm — now MLflow-instrumented (Phase 2).

Trains the returns-target LSTM (the current best-performing architecture —
see notebooks/05_lstm_returns.ipynb) for next-day prediction, one model per
stock in the locked universe, using the engineered features from stage 2
(data/processed/). Writes results/lstm_metrics.csv and saves each trained
model to models/lstm/<TICKER>.keras.

Every per-stock training run is also logged to MLflow (local ./mlruns store,
one experiment per sector): hyperparameters, per-epoch train/val loss, final
test metrics, the loss-curve plot, and the model artifact. DVC still tracks
the stage outputs exactly as before — MLflow logging is a side effect.

Run directly:  python scripts/train_lstm.py
Run via DVC:   dvc repro train_lstm

Experiment sweeps (MLflow-only, does NOT overwrite DVC-tracked outputs):
    python scripts/train_lstm.py --sweep --seq-length 15 \
        --tickers TCS.NS HDFCBANK.NS RELIANCE.NS HINDUNILVR.NS
"""

from __future__ import annotations

# CRITICAL: tensorflow must be imported before pandas/sklearn in this
# process — see src/lstm_model.py's module docstring for the deadlock this
# avoids on TF 2.21 / Keras 3.15 / macOS arm64. mlflow imports pandas, so it
# must come after tensorflow too.
import tensorflow as tf
tf.random.set_seed(42)

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

from src.data_loader import load_universe, flat_tickers, ticker_to_sector
from src.features import FEATURE_COLUMNS
from src.scaling import chronological_split, scale_features
from src.evaluate import compute_metrics
from src.tracking import setup_mlflow
from src.lstm_model import (
    build_sequences, build_sequences_for_split, scale_target,
    build_lstm_model, train_lstm, reconstruct_price_from_returns,
)

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models" / "lstm"

np.random.seed(42)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seq-length", type=int, default=30)
    p.add_argument("--units", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--tickers", nargs="+", default=None,
                   help="Subset of tickers (default: full locked universe)")
    p.add_argument("--sweep", action="store_true",
                   help="Experiment-only mode: log to MLflow but do NOT "
                        "overwrite the DVC-tracked models/ and results/ outputs")
    return p.parse_args()


def loss_curve_figure(hist, tk: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(hist.history["loss"], label="train loss")
    ax.plot(hist.history["val_loss"], label="val loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE (scaled returns)")
    ax.set_title(f"{tk} — LSTM training curve")
    ax.legend()
    fig.tight_layout()
    return fig


def build_return_target(feats: pd.DataFrame) -> pd.DataFrame:
    out = feats.copy()
    out["target_next_return"] = out["Adj Close"].pct_change().shift(-1)
    out["target_next_close"] = out["Adj Close"].shift(-1)
    return out.dropna(subset=["target_next_return", "target_next_close"])


def train_one_stock(tk: str, sector: str, args: argparse.Namespace) -> dict:
    """Train the LSTM for one stock, log the run to MLflow, return metrics row."""
    feats = pd.read_csv(PROCESSED_DIR / f"{tk.replace('.', '_')}.csv",
                        index_col="Date", parse_dates=True)
    feats = build_return_target(feats)

    train, val, test = chronological_split(feats, 0.70, 0.15)
    train_s, val_s, test_s, _ = scale_features(train, val, test, FEATURE_COLUMNS)

    y_train_sc, y_val_sc, y_test_sc, t_scaler = scale_target(
        train["target_next_return"], val["target_next_return"],
        test["target_next_return"])
    train_s = train_s.assign(target_scaled=y_train_sc)
    val_s = val_s.assign(target_scaled=y_val_sc)

    seq = args.seq_length
    X_train, Y_train, _ = build_sequences(train_s, FEATURE_COLUMNS,
                                           "target_scaled", seq)
    X_val, Y_val, _ = build_sequences_for_split(train_s, val_s, FEATURE_COLUMNS,
                                                 "target_scaled", seq)
    test_sc = test_s.assign(target_scaled=y_test_sc)
    X_test, _, _ = build_sequences_for_split(
        pd.concat([train_s, val_s]), test_sc, FEATURE_COLUMNS,
        "target_scaled", seq)

    setup_mlflow(sector)
    with mlflow.start_run(run_name=f"LSTM-{tk}-seq{seq}-u{args.units}"):
        mlflow.log_params({
            "model": "LSTM", "ticker": tk, "sector": sector,
            "seq_length": seq, "units": args.units, "dropout": args.dropout,
            "learning_rate": args.learning_rate, "batch_size": args.batch_size,
            "max_epochs": args.epochs, "patience": args.patience,
            "target": "next_day_return", "n_features": len(FEATURE_COLUMNS),
        })

        model = build_lstm_model((seq, len(FEATURE_COLUMNS)),
                                 units=args.units, dropout=args.dropout,
                                 learning_rate=args.learning_rate)
        model, hist = train_lstm(model, X_train, Y_train, X_val, Y_val,
                                  epochs=args.epochs, batch_size=args.batch_size,
                                  patience=args.patience)

        # Per-epoch curves — this is what makes runs comparable epoch-by-epoch
        # in the MLflow UI, not just by their final numbers.
        for epoch, (tr, vl) in enumerate(zip(hist.history["loss"],
                                             hist.history["val_loss"])):
            mlflow.log_metric("train_loss", tr, step=epoch)
            mlflow.log_metric("val_loss", vl, step=epoch)

        pred_ret_scaled = model.predict(X_test, verbose=0).flatten()
        pred_ret = t_scaler.inverse_transform(
            pred_ret_scaled.reshape(-1, 1)).flatten()
        pred_price = reconstruct_price_from_returns(test["Adj Close"].values,
                                                    pred_ret)

        m = compute_metrics(test["target_next_close"], pred_price,
                            test["Adj Close"])
        mlflow.log_metrics({
            "RMSE": m["rmse"], "MAE": m["mae"], "MAPE": m["mape"],
            "directional_accuracy": m["directional_accuracy"],
            "epochs_trained": len(hist.history["loss"]),
        })

        fig = loss_curve_figure(hist, tk)
        mlflow.log_figure(fig, "loss_curve.png")
        plt.close(fig)

        mlflow.tensorflow.log_model(model, name="model")

    row = {
        "ticker": tk, "sector": sector,
        "lstm_RMSE": m["rmse"], "lstm_MAE": m["mae"],
        "lstm_MAPE": m["mape"], "lstm_DirAcc": m["directional_accuracy"],
        "epochs_trained": len(hist.history["loss"]),
    }
    return row, model


def main() -> None:
    args = parse_args()
    universe = load_universe()
    tickers = args.tickers or flat_tickers(universe)
    tk2sec = ticker_to_sector(universe)

    unknown = [tk for tk in tickers if tk not in tk2sec]
    if unknown:
        sys.exit(f"tickers not in locked universe: {unknown}")

    if not args.sweep:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for i, tk in enumerate(tickers, 1):
        row, model = train_one_stock(tk, tk2sec[tk], args)
        results.append(row)
        if not args.sweep:
            model.save(MODELS_DIR / f"{tk.replace('.', '_')}.keras")
        print(f"[{i:>2}/{len(tickers)}] {tk:<16} "
              f"epochs={row['epochs_trained']:>3}  MAPE={row['lstm_MAPE']:.2f}%  "
              f"DirAcc={row['lstm_DirAcc']:.1f}%")

    if args.sweep:
        print(f"\ntrain_lstm (sweep mode): {len(results)} stocks logged to "
              f"MLflow only — DVC outputs untouched")
        return

    results_df = pd.DataFrame(results).set_index("ticker")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULTS_DIR / "lstm_metrics.csv")
    print(f"\ntrain_lstm: {len(results)} stocks -> "
          f"{RESULTS_DIR / 'lstm_metrics.csv'}")
    print(f"Model artifacts -> {MODELS_DIR}")


if __name__ == "__main__":
    main()

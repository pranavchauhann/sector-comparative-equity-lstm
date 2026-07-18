"""
scripts/train_lstm.py
=======================
DVC pipeline stage 4: train_lstm.

Trains the returns-target LSTM (the current best-performing architecture —
see notebooks/05_lstm_returns.ipynb) for next-day prediction, one model per
stock in the locked universe, using the engineered features from stage 2
(data/processed/). Writes results/lstm_metrics.csv and saves each trained
model to models/lstm/<TICKER>.keras.

Run directly:  python scripts/train_lstm.py
Run via DVC:   dvc repro train_lstm
"""

from __future__ import annotations

# CRITICAL: tensorflow must be imported before pandas/sklearn in this
# process — see src/lstm_model.py's module docstring for the deadlock this
# avoids on TF 2.21 / Keras 3.15 / macOS arm64.
import tensorflow as tf
tf.random.set_seed(42)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.data_loader import load_universe, flat_tickers, ticker_to_sector
from src.features import FEATURE_COLUMNS
from src.scaling import chronological_split, scale_features
from src.evaluate import compute_metrics
from src.lstm_model import (
    build_sequences, build_sequences_for_split, scale_target,
    build_lstm_model, train_lstm, reconstruct_price_from_returns,
)

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models" / "lstm"
SEQ_LENGTH = 30

np.random.seed(42)


def build_return_target(feats: pd.DataFrame) -> pd.DataFrame:
    out = feats.copy()
    out["target_next_return"] = out["Adj Close"].pct_change().shift(-1)
    out["target_next_close"] = out["Adj Close"].shift(-1)
    return out.dropna(subset=["target_next_return", "target_next_close"])


def main() -> None:
    universe = load_universe()
    tickers = flat_tickers(universe)
    tk2sec = ticker_to_sector(universe)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for i, tk in enumerate(tickers, 1):
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

        X_train, Y_train, _ = build_sequences(train_s, FEATURE_COLUMNS,
                                               "target_scaled", SEQ_LENGTH)
        X_val, Y_val, _ = build_sequences_for_split(train_s, val_s, FEATURE_COLUMNS,
                                                     "target_scaled", SEQ_LENGTH)
        test_sc = test_s.assign(target_scaled=y_test_sc)
        X_test, _, _ = build_sequences_for_split(
            pd.concat([train_s, val_s]), test_sc, FEATURE_COLUMNS,
            "target_scaled", SEQ_LENGTH)

        model = build_lstm_model((SEQ_LENGTH, len(FEATURE_COLUMNS)))
        model, hist = train_lstm(model, X_train, Y_train, X_val, Y_val,
                                  epochs=100, patience=10)

        pred_ret_scaled = model.predict(X_test, verbose=0).flatten()
        pred_ret = t_scaler.inverse_transform(pred_ret_scaled.reshape(-1, 1)).flatten()
        pred_price = reconstruct_price_from_returns(test["Adj Close"].values, pred_ret)

        m = compute_metrics(test["target_next_close"], pred_price, test["Adj Close"])
        results.append({
            "ticker": tk, "sector": tk2sec[tk],
            "lstm_RMSE": m["rmse"], "lstm_MAE": m["mae"],
            "lstm_MAPE": m["mape"], "lstm_DirAcc": m["directional_accuracy"],
            "epochs_trained": len(hist.history["loss"]),
        })
        model.save(MODELS_DIR / f"{tk.replace('.', '_')}.keras")
        print(f"[{i:>2}/{len(tickers)}] {tk:<16} "
              f"epochs={len(hist.history['loss']):>3}  MAPE={m['mape']:.2f}%  "
              f"DirAcc={m['directional_accuracy']:.1f}%")

    results_df = pd.DataFrame(results).set_index("ticker")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULTS_DIR / "lstm_metrics.csv")
    print(f"\ntrain_lstm: {len(results)} stocks -> "
          f"{RESULTS_DIR / 'lstm_metrics.csv'}")
    print(f"Model artifacts -> {MODELS_DIR}")


if __name__ == "__main__":
    main()

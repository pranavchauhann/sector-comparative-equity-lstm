"""
src/log_predictions.py
=======================
Phase 4 shadow-evaluation log: the live error stream drift detection needs.

Every run does two things per stock:

1. **Backfill** — any logged prediction whose target date has since been
   realised in ``data/processed/`` gets its actual price and absolute
   percentage error (APE) filled in.
2. **Append** — a fresh next-trading-day prediction from the deployed LSTM
   (``models/lstm/<TICKER>.keras``), logged with a timestamp *before* the
   actual is knowable. That ordering is the point: the log is an honest
   record of what the model said in advance, not a retrospective backtest.

``--replay N`` bootstraps the log by walking the most recent N trading days
as if they had been live (predict from data through day t, realise day t+1).
This is still shadow evaluation — the models were trained with a 70/15/15
chronological split, so the recent window replayed here lies in their test
period, not their training data.

Scaling note: feature/target scalers are re-fit on the same 70% train split
the model was trained on (the split boundary only moves when new data is
appended, and train-split statistics are stable), exactly mirroring
scripts/train_lstm.py.

Output: ``data/prediction_log.csv`` with columns
    ticker, sector, prediction_date, target_date, predicted_price,
    actual_price, ape_pct, logged_at

Run:  python src/log_predictions.py [--replay 60] [--tickers TCS.NS ...]
"""

from __future__ import annotations

# TF before pandas/sklearn — see src/lstm_model.py.
import tensorflow as tf

import argparse
import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.data_loader import load_universe, flat_tickers, ticker_to_sector
from src.features import FEATURE_COLUMNS
from src.scaling import chronological_split, scale_features
from src.lstm_model import scale_target

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models" / "lstm"
LOG_PATH = ROOT / "data" / "prediction_log.csv"
SEQ_LENGTH = 30

LOG_COLUMNS = ["ticker", "sector", "prediction_date", "target_date",
               "predicted_price", "actual_price", "ape_pct", "logged_at"]


def load_log() -> pd.DataFrame:
    if LOG_PATH.exists():
        return pd.read_csv(LOG_PATH, dtype={"ticker": str})
    return pd.DataFrame(columns=LOG_COLUMNS)


def prepare_stock(tk: str):
    """Load processed features + fitted-as-at-training scalers for one stock.

    Returns (feats, scaled feature frame covering all rows, target scaler).
    """
    feats = pd.read_csv(PROCESSED_DIR / f"{tk.replace('.', '_')}.csv",
                        index_col="Date", parse_dates=True)
    tgt = feats["Adj Close"].pct_change().shift(-1)

    train, val, test = chronological_split(feats, 0.70, 0.15)
    train_s, val_s, test_s, _ = scale_features(train, val, test, FEATURE_COLUMNS)
    all_scaled = pd.concat([train_s, val_s, test_s])

    # Target scaler fit on train only (as in training); dropna mirrors the
    # training target construction.
    tr = tgt.loc[train.index].dropna()
    _, _, _, t_scaler = scale_target(tr, tr, tr)
    return feats, all_scaled, t_scaler


def predict_for_dates(model, all_scaled: pd.DataFrame, t_scaler,
                      feats: pd.DataFrame, dates: list) -> dict:
    """Predicted next-day price for each 'today' date, one batched predict."""
    X = np.stack([
        all_scaled.loc[:d, FEATURE_COLUMNS].values[-SEQ_LENGTH:]
        for d in dates
    ])
    pred_scaled = model.predict(X, verbose=0).flatten()
    pred_ret = t_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    today_price = feats.loc[dates, "Adj Close"].values
    return dict(zip(dates, today_price * (1.0 + pred_ret)))


def backfill(log: pd.DataFrame, feats: pd.DataFrame, tk: str) -> int:
    """Fill actual_price/ape for this ticker's rows whose target has realised."""
    filled = 0
    mask = (log["ticker"] == tk) & (log["actual_price"].isna())
    for idx in log.index[mask]:
        pred_date = pd.Timestamp(log.at[idx, "prediction_date"])
        later = feats.index[feats.index > pred_date]
        if len(later) == 0:
            continue
        target = later[0]
        actual = float(feats.at[target, "Adj Close"])
        pred = float(log.at[idx, "predicted_price"])
        log.at[idx, "target_date"] = target.date().isoformat()
        log.at[idx, "actual_price"] = actual
        log.at[idx, "ape_pct"] = abs(pred - actual) / actual * 100
        filled += 1
    return filled


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=int, default=0, metavar="N",
                        help="also log predictions for the last N trading "
                             "days (bootstrap the shadow history)")
    parser.add_argument("--tickers", nargs="+", default=None)
    args = parser.parse_args()

    universe = load_universe()
    tickers = args.tickers or flat_tickers(universe)
    tk2sec = ticker_to_sector(universe)
    log = load_log()
    now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")

    new_rows, backfilled = [], 0
    for i, tk in enumerate(tickers, 1):
        model_path = MODELS_DIR / f"{tk.replace('.', '_')}.keras"
        if not model_path.exists():
            print(f"[{i:>2}/{len(tickers)}] {tk:<16} SKIP (no model artifact)")
            continue
        feats, all_scaled, t_scaler = prepare_stock(tk)
        backfilled += backfill(log, feats, tk)

        model = tf.keras.models.load_model(model_path)
        dates = list(feats.index[-(args.replay + 1):]) if args.replay else [feats.index[-1]]
        already = set(pd.to_datetime(
            log.loc[log["ticker"] == tk, "prediction_date"]).dt.date)
        dates = [d for d in dates if d.date() not in already]
        if dates:
            preds = predict_for_dates(model, all_scaled, t_scaler, feats, dates)
            for d, p in preds.items():
                new_rows.append({
                    "ticker": tk, "sector": tk2sec[tk],
                    "prediction_date": d.date().isoformat(),
                    "target_date": None, "predicted_price": round(float(p), 4),
                    "actual_price": None, "ape_pct": None, "logged_at": now,
                })
        print(f"[{i:>2}/{len(tickers)}] {tk:<16} +{len(dates)} predictions")

    log = pd.concat([log, pd.DataFrame(new_rows, columns=LOG_COLUMNS)],
                    ignore_index=True)
    # Backfill rows appended via --replay in this same run.
    for tk in tickers:
        f = PROCESSED_DIR / f"{tk.replace('.', '_')}.csv"
        if f.exists() and (log["ticker"] == tk).any():
            feats = pd.read_csv(f, index_col="Date", parse_dates=True)
            backfilled += backfill(log, feats, tk)

    log = log.sort_values(["prediction_date", "ticker"]).reset_index(drop=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(LOG_PATH, index=False)
    n_pending = int(log["actual_price"].isna().sum())
    print(f"\nprediction log: {len(log)} rows (+{len(new_rows)} new, "
          f"{backfilled} backfilled, {n_pending} awaiting actuals) -> {LOG_PATH}")


if __name__ == "__main__":
    main()

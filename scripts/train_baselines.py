"""
scripts/train_baselines.py
============================
DVC pipeline stage 3: train_baselines.

Runs naive / linear regression / ARIMA on every stock in the locked
universe, using the engineered features from stage 2 (data/processed/).
Writes results/baseline_metrics.csv (per-stock RMSE/MAE/MAPE/directional
accuracy for all three models) and persists each stock's fitted
LinearRegression and ARIMA model to models/baselines/ via joblib — the
notebook version (03_baselines.ipynb) never serialized these; this script
adds that so DVC has real model artifacts to version.

Run directly:  python scripts/train_baselines.py
Run via DVC:   dvc repro train_baselines
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.data_loader import load_universe, flat_tickers, ticker_to_sector
from src.features import FEATURE_COLUMNS
from src.scaling import chronological_split, scale_features
from src.baselines import naive_forecast, linear_regression_forecast, arima_forecast
from src.evaluate import compute_metrics

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models" / "baselines"

ARIMA_ORDERS = [(5, 1, 0), (2, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1)]


def build_target(feats: pd.DataFrame) -> pd.DataFrame:
    out = feats.copy()
    out["target_next_close"] = out["Adj Close"].shift(-1)
    return out.dropna(subset=["target_next_close"])


def run_arima_with_fallback(history, test_actuals, orders=ARIMA_ORDERS):
    errors = []
    for order in orders:
        try:
            preds, fitted = arima_forecast(history, test_actuals, order=order,
                                            return_model=True)
            return preds, order, fitted, errors
        except Exception as exc:  # noqa: BLE001
            errors.append(f"order={order}: {type(exc).__name__}: {exc}")
    return None, None, None, errors


def main() -> None:
    universe = load_universe()
    tickers = flat_tickers(universe)
    tk2sec = ticker_to_sector(universe)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    results, arima_failures = [], []

    for i, tk in enumerate(tickers, 1):
        feats = pd.read_csv(PROCESSED_DIR / f"{tk.replace('.', '_')}.csv",
                            index_col="Date", parse_dates=True)
        feats = build_target(feats)

        train, val, test = chronological_split(feats, 0.70, 0.15)
        train_s, val_s, test_s, _ = scale_features(train, val, test, FEATURE_COLUMNS)

        y_train, y_test = train["target_next_close"], test["target_next_close"]
        y_prev_test = test["Adj Close"]

        naive_pred = naive_forecast(test["Adj Close"])
        naive_m = compute_metrics(y_test, naive_pred, y_prev_test)

        lr_model = LinearRegression()
        lr_model.fit(train_s[FEATURE_COLUMNS], y_train)
        lr_pred = pd.Series(lr_model.predict(test_s[FEATURE_COLUMNS]), index=test.index)
        lr_m = compute_metrics(y_test, lr_pred, y_prev_test)
        joblib.dump(lr_model, MODELS_DIR / f"{tk.replace('.', '_')}_linreg.pkl")

        history = pd.concat([train["Adj Close"], val["Adj Close"]])
        arima_pred, order_used, arima_fitted, errors = run_arima_with_fallback(
            history, test["Adj Close"])
        if arima_pred is not None:
            arima_m = compute_metrics(y_test, arima_pred, y_prev_test)
            joblib.dump(arima_fitted, MODELS_DIR / f"{tk.replace('.', '_')}_arima.pkl")
        else:
            arima_m = {"rmse": None, "mae": None, "mape": None,
                       "directional_accuracy": None}
            arima_failures.append({"ticker": tk, "errors": errors})

        results.append({
            "ticker": tk, "sector": tk2sec[tk],
            "naive_RMSE": naive_m["rmse"], "naive_MAE": naive_m["mae"],
            "naive_MAPE": naive_m["mape"], "naive_DirAcc": naive_m["directional_accuracy"],
            "linreg_RMSE": lr_m["rmse"], "linreg_MAE": lr_m["mae"],
            "linreg_MAPE": lr_m["mape"], "linreg_DirAcc": lr_m["directional_accuracy"],
            "arima_RMSE": arima_m["rmse"], "arima_MAE": arima_m["mae"],
            "arima_MAPE": arima_m["mape"], "arima_DirAcc": arima_m["directional_accuracy"],
            "arima_order": order_used,
        })
        print(f"[{i:>2}/{len(tickers)}] {tk:<16} naive_RMSE={naive_m['rmse']:.2f}  "
              f"linreg_RMSE={lr_m['rmse']:.2f}  "
              f"arima={'FAILED' if arima_pred is None else f'order={order_used}'}")

    results_df = pd.DataFrame(results).set_index("ticker")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULTS_DIR / "baseline_metrics.csv")
    print(f"\ntrain_baselines: {len(results)} stocks, "
          f"{len(arima_failures)} ARIMA failures -> {RESULTS_DIR / 'baseline_metrics.csv'}")
    print(f"Model artifacts -> {MODELS_DIR}")


if __name__ == "__main__":
    main()

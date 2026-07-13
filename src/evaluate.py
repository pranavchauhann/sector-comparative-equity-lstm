"""
src/evaluate.py
================
Shared evaluation metrics, used identically for baselines (Phase 3) and the
LSTM (Phase 4) so results are directly comparable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    y_prev: pd.Series | np.ndarray | None = None,
) -> dict:
    """RMSE, MAE, MAPE, and directional accuracy for a set of predictions.

    Parameters
    ----------
    y_true : actual next-day Adj Close values.
    y_pred : predicted next-day Adj Close values (same index/order as y_true).
    y_prev : the "today" actual close each prediction is a forecast *from*
        (e.g. ``test_df['Adj Close']``). Required to define "direction":
        actual direction = sign(y_true - y_prev), predicted direction =
        sign(y_pred - y_prev). If omitted, falls back to
        ``pd.Series(y_true).shift(1)`` — note this drops the first
        observation (no prior value) and is only a reasonable substitute if
        y_true is itself a contiguous daily series; passing y_prev explicitly
        is strongly preferred and is what every caller in this project does.

    Returns
    -------
    dict with keys: rmse, mae, mape, directional_accuracy, n_obs.
    ``directional_accuracy`` is a percentage (0-100). NaNs in any of
    y_true/y_pred/y_prev are dropped pairwise before scoring.
    """
    y_true = pd.Series(y_true).reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)

    if y_prev is None:
        y_prev = y_true.shift(1)
    else:
        y_prev = pd.Series(y_prev).reset_index(drop=True)

    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "y_prev": y_prev}).dropna()
    if df.empty:
        raise ValueError("compute_metrics: no overlapping non-NaN observations")

    err = df["y_true"] - df["y_pred"]
    rmse = float(np.sqrt((err ** 2).mean()))
    mae = float(err.abs().mean())
    mape = float((err.abs() / df["y_true"].abs()).mean() * 100)

    actual_dir = np.sign(df["y_true"] - df["y_prev"])
    pred_dir = np.sign(df["y_pred"] - df["y_prev"])
    directional_accuracy = float((actual_dir == pred_dir).mean() * 100)

    return {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "directional_accuracy": directional_accuracy,
        "n_obs": int(len(df)),
    }


if __name__ == "__main__":
    y_true = pd.Series([100, 102, 101, 105, 103])
    y_pred = pd.Series([99, 103, 100, 104, 104])
    y_prev = pd.Series([98, 100, 102, 101, 105])
    print(compute_metrics(y_true, y_pred, y_prev))

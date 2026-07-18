"""
src/lstm_model.py
==================
Sequence construction and the LSTM architecture/training loop for next-day
`Adj Close` prediction.

Design notes
------------
* **Same target as the baselines.** The LSTM predicts ``target_next_close``
  (``Adj Close`` shifted -1), exactly as Phase 3's naive/linear-regression/
  ARIMA baselines do, so `RMSE`/`MAE`/`MAPE`/directional-accuracy are directly
  comparable across all four models.
* **30-day input window.** Each training example is the trailing 30 days of
  *scaled* engineered features (never raw OHLCV) predicting the next day's
  close.
* **No rows lost at split boundaries.** Building sequences naively on each of
  train/val/test independently would throw away the first 29 rows of val and
  test (no 30-day lookback available at the start of a split). Instead,
  ``build_sequences_for_split`` borrows the trailing ``seq_length - 1`` rows
  from the *previous, chronologically earlier* split as lookback context —
  those feature values already happened before the split starts, so this is
  not leakage (the same principle Phase 3's ARIMA walk-forward relies on: use
  of already-realised past data to build "today's" input is legitimate). The
  targets and evaluation dates returned always belong to the split itself, so
  val/test sequence counts match Phase 2/3's split sizes exactly.
* **Target scaling.** LSTM optimisation is unstable when regressing directly
  onto raw price-level targets (hundreds-to-thousands of INR) while inputs
  are standardised. We fit a *second* `StandardScaler` on the train target
  only (never val/test) and inverse-transform predictions before computing
  metrics, extending Phase 2's "scaler fit on train only" rule to the target.
"""

from __future__ import annotations

# NOTE: TensorFlow/Keras must be imported before pandas/sklearn in this
# process. On this environment (TF 2.21 + Keras 3.15, Python 3.12, macOS
# arm64), importing pandas or scikit-learn (both pull in a BLAS backend via
# numpy/scipy) *before* tensorflow.keras causes model.fit() to deadlock at
# 0% CPU as soon as a callback (e.g. EarlyStopping) is used with epochs > a
# handful — reproducible and confirmed via isolated testing. Importing
# tensorflow.keras first avoids the conflict entirely. Any notebook/script
# using this module must likewise import tensorflow (or this module) before
# importing pandas.
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def build_sequences(
    df: pd.DataFrame, feature_cols: list[str], target_col: str, seq_length: int = 30,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Convert a chronologically-sorted feature DataFrame into (X, y) sequences.

    ``X[i]`` is the window of ``feature_cols`` values for the ``seq_length``
    rows ending at (and including) the row that produced ``y[i]``; ``y[i]`` is
    ``target_col`` at that same row (i.e. the *next-day* close already stored
    there, matching the baselines' target convention). ``dates[i]`` is that
    row's own date — the "today" the prediction is made from.

    Returns
    -------
    X : np.ndarray, shape (n_samples, seq_length, len(feature_cols))
    y : np.ndarray, shape (n_samples,)
    dates : pd.DatetimeIndex, shape (n_samples,)
    """
    values = df[feature_cols].values
    targets = df[target_col].values
    dates = df.index

    X, y, out_dates = [], [], []
    for i in range(seq_length - 1, len(df)):
        X.append(values[i - seq_length + 1: i + 1])
        y.append(targets[i])
        out_dates.append(dates[i])

    return np.array(X), np.array(y), pd.DatetimeIndex(out_dates)


def build_sequences_for_split(
    history_df: pd.DataFrame,
    split_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    seq_length: int = 30,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Build sequences for ``split_df`` (val or test) with no rows lost.

    Borrows the trailing ``seq_length - 1`` rows of ``history_df`` (the
    immediately preceding split — e.g. train, or train+val) purely as
    lookback context for the first few input windows. The returned
    ``y``/``dates`` always correspond 1:1 to ``split_df``'s own rows.
    """
    buffer = history_df.tail(seq_length - 1)
    combined = pd.concat([buffer, split_df])
    X, y, dates = build_sequences(combined, feature_cols, target_col, seq_length)
    assert len(dates) == len(split_df), (
        f"expected {len(split_df)} sequences, got {len(dates)} — "
        f"history_df has fewer than {seq_length - 1} rows to borrow from?"
    )
    return X, y, dates


def scale_target(
    train_target: pd.Series, val_target: pd.Series, test_target: pd.Series,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Fit a StandardScaler on the train target only; transform all three splits."""
    scaler = StandardScaler()
    y_train = scaler.fit_transform(train_target.values.reshape(-1, 1)).flatten()
    y_val = scaler.transform(val_target.values.reshape(-1, 1)).flatten()
    y_test = scaler.transform(test_target.values.reshape(-1, 1)).flatten()
    return y_train, y_val, y_test, scaler


def build_lstm_model(input_shape: tuple[int, int], units: int = 64) -> Sequential:
    """2 stacked LSTM layers (50-100 units each) + Dropout(0.2) + Dense output.

    ``input_shape`` is ``(seq_length, num_features)``. A second Dropout after
    the final LSTM layer (before the Dense head) is included in addition to
    the required dropout *between* the two LSTM layers — standard practice
    for regularising the layer that feeds the output directly.
    """
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units),
        Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def reconstruct_price_from_returns(
    today_prices: pd.Series | np.ndarray, pred_returns: np.ndarray,
) -> np.ndarray:
    """Convert predicted next-day *returns* back to next-day *price levels*.

    ``pred_price[t+1] = today_price[t] * (1 + pred_return[t+1])``.

    This is the evaluation bridge for the returns-target LSTM: the model
    regresses on scale-free returns (avoiding the price-level extrapolation
    failure documented in Phase 4 — a scaled-price target cannot exceed the
    training range), but metrics are still computed on reconstructed prices
    so RMSE/MAE/MAPE are directly comparable with the price-level models.
    Directional accuracy is unaffected by the reconstruction since
    ``sign(pred_price - today_price) == sign(pred_return)``.
    """
    today = np.asarray(today_prices, dtype=float)
    return today * (1.0 + np.asarray(pred_returns, dtype=float))


def train_lstm(
    model: Sequential,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    epochs: int = 100, batch_size: int = 32, patience: int = 10,
):
    """Train with early stopping on validation loss; returns (model, history).

    ``restore_best_weights=True`` means the returned model holds the weights
    from the epoch with the lowest validation loss, not necessarily the final
    epoch's weights.
    """
    early_stop = EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True,
    )
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs, batch_size=batch_size,
        callbacks=[early_stop], verbose=0,
    )
    return model, history

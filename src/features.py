"""
src/features.py
================
Feature engineering for a single stock's raw OHLCV history.

``compute_features(df)`` takes a raw price DataFrame (the shape produced by
``src.data_loader.download_history`` / ``load_cached`` — columns
``Open, High, Low, Close, Adj Close, Volume``) and returns it with the
project's engineered features appended:

    daily_return   - % change in Adj Close (pandas-ta percent_return)
    ma_5           - 5-day  SMA of Adj Close
    ma_20          - 20-day SMA of Adj Close
    ma_50          - 50-day SMA of Adj Close
    rsi_14         - 14-day RSI
    macd           - MACD line (12, 26)
    macd_signal    - MACD signal line (9)
    volatility_20  - 20-day rolling std of daily_return
    volume_ratio   - Volume / 20-day average Volume

All indicator math goes through ``pandas_ta`` rather than hand-rolled
formulas, per the project spec — less error-prone and matches the standard
definitions (e.g. Wilder's RSI smoothing) that any reader will recognise.

Rows created by an indicator's warm-up period (the first `window - 1` rows of
a rolling/EWM computation) are NaN and are dropped before the feature set is
considered usable. ``compute_features`` reports how many rows were dropped and
why via its return value's ``.attrs["warmup_rows_dropped"]``.
"""

from __future__ import annotations

import pandas as pd
import pandas_ta as ta

# Longest warm-up in the feature set: MA_50 needs 50 Adj Close observations
# before it produces its first non-NaN value. RSI(14), MACD(12,26,9), and
# MA_5/MA_20/volatility_20 all warm up faster, so MA_50 sets the drop count.
FEATURE_COLUMNS = [
    "daily_return", "ma_5", "ma_20", "ma_50",
    "rsi_14", "macd", "macd_signal",
    "volatility_20", "volume_ratio",
]


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` with engineered features appended and warm-up NaNs dropped.

    Parameters
    ----------
    df : raw OHLCV DataFrame for a single stock, indexed by Date, with an
         ``Adj Close`` column (and ``Volume``).

    Returns
    -------
    A copy of ``df`` with the columns in ``FEATURE_COLUMNS`` added, sorted by
    date, with leading warm-up-NaN rows removed. ``result.attrs`` carries:
      - "warmup_rows_dropped": int, rows removed for indicator warm-up
      - "input_rows": int, rows in ``df`` before processing
      - "output_rows": int, rows in the returned frame
    """
    if "Adj Close" not in df.columns:
        raise ValueError("compute_features requires an 'Adj Close' column")

    out = df.sort_index().copy()
    input_rows = len(out)

    # Any fully-empty trailing rows (e.g. an unsettled latest session where
    # Yahoo has not posted Adj Close yet) can't produce features either;
    # treat them the same as warm-up NaNs and drop them below.
    adj = out["Adj Close"]
    vol = out["Volume"]

    out["daily_return"] = ta.percent_return(adj)
    out["ma_5"] = ta.sma(adj, length=5)
    out["ma_20"] = ta.sma(adj, length=20)
    out["ma_50"] = ta.sma(adj, length=50)
    out["rsi_14"] = ta.rsi(adj, length=14)

    macd_df = ta.macd(adj, fast=12, slow=26, signal=9)
    out["macd"] = macd_df["MACD_12_26_9"]
    out["macd_signal"] = macd_df["MACDs_12_26_9"]

    out["volatility_20"] = out["daily_return"].rolling(window=20).std()

    vol_ma_20 = ta.sma(vol, length=20)
    out["volume_ratio"] = vol / vol_ma_20

    before_dropna = len(out)
    out = out.dropna(subset=FEATURE_COLUMNS)
    dropped = before_dropna - len(out)

    out.attrs["warmup_rows_dropped"] = dropped
    out.attrs["input_rows"] = input_rows
    out.attrs["output_rows"] = len(out)
    return out


if __name__ == "__main__":
    # Smoke test against a cached ticker.
    from pathlib import Path

    sample = Path(__file__).resolve().parents[1] / "data" / "raw" / "RELIANCE_NS.csv"
    raw = pd.read_csv(sample, index_col="Date", parse_dates=True)
    feats = compute_features(raw)
    print(f"input_rows={feats.attrs['input_rows']}  "
          f"dropped={feats.attrs['warmup_rows_dropped']}  "
          f"output_rows={feats.attrs['output_rows']}")
    print(feats[FEATURE_COLUMNS].tail(3))

"""
src/scaling.py
===============
Chronological train/val/test split and leakage-safe feature scaling.

Non-negotiable rules this module exists to enforce:
  * **Chronological splits only.** Random splitting on time series leaks
    future information into training.
  * **Scaler fit on train only.** Fitting ``StandardScaler`` on the full
    dataset (or on train+val) leaks the distribution of future data into the
    training signal, even though no individual future *value* is copied.

Typical use
-----------
    from src.scaling import chronological_split, scale_features

    train, val, test = chronological_split(feat_df, train_frac=0.70, val_frac=0.15)
    train_s, val_s, test_s, scaler = scale_features(train, val, test, FEATURE_COLUMNS)
"""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import StandardScaler

DEFAULT_TRAIN_FRAC = 0.70
DEFAULT_VAL_FRAC = 0.15
# test_frac is implicitly 1 - train_frac - val_frac (0.15 by default)


def chronological_split(
    df: pd.DataFrame,
    train_frac: float = DEFAULT_TRAIN_FRAC,
    val_frac: float = DEFAULT_VAL_FRAC,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a date-sorted DataFrame chronologically into train/val/test.

    The first ``train_frac`` of rows (by position, i.e. by date since the
    frame must already be sorted ascending) become train, the next
    ``val_frac`` become validation, and the remainder becomes test. No
    shuffling, ever — row order is time order.
    """
    if not df.index.is_monotonic_increasing:
        raise ValueError("chronological_split requires a date-sorted index")
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must leave a nonzero test slice")

    n = len(df)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)

    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()
    return train, val, test


def scale_features(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Fit a ``StandardScaler`` on ``train[feature_cols]`` only, and apply it
    to train/val/test alike.

    Returns copies of train/val/test with ``feature_cols`` replaced by their
    scaled values (all other columns, e.g. Adj Close, are left untouched so
    the raw price is still available for evaluation/plotting), plus the
    fitted scaler (needed later to inverse-transform predictions or to scale
    genuinely new data with the *same* train-derived mean/std).
    """
    scaler = StandardScaler()
    scaler.fit(train[feature_cols])

    train_s, val_s, test_s = train.copy(), val.copy(), test.copy()
    train_s[feature_cols] = scaler.transform(train[feature_cols])
    val_s[feature_cols] = scaler.transform(val[feature_cols])
    test_s[feature_cols] = scaler.transform(test[feature_cols])
    return train_s, val_s, test_s, scaler


if __name__ == "__main__":
    from pathlib import Path
    from src.features import compute_features, FEATURE_COLUMNS

    sample = Path(__file__).resolve().parents[1] / "data" / "raw" / "RELIANCE_NS.csv"
    raw = pd.read_csv(sample, index_col="Date", parse_dates=True)
    feats = compute_features(raw)

    train, val, test = chronological_split(feats)
    print(f"train={len(train)}  val={len(val)}  test={len(test)}  "
          f"(train ends {train.index.max().date()}, "
          f"test starts {test.index.min().date()})")

    train_s, val_s, test_s, scaler = scale_features(train, val, test, FEATURE_COLUMNS)
    print("\nTrain (scaled) feature means (~0) / stds (~1):")
    print(train_s[FEATURE_COLUMNS].mean().round(3))
    print(train_s[FEATURE_COLUMNS].std().round(3))

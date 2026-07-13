"""
src/baselines.py
=================
Three baseline "predict tomorrow's closing price" models, evaluated fairly
against each other and (in later phases) against the LSTM.

**Common information cutoff.** Every model here predicts ``Adj Close`` at
date ``t+1`` using only information available through date ``t`` ("today").
This is the same cutoff the LSTM will use in Phase 4, so comparisons across
models are apples-to-apples:

  * ``naive_forecast``            -> tomorrow's close = today's close
  * ``linear_regression_forecast`` -> today's engineered features -> tomorrow's close
  * ``arima_forecast``            -> today's realised close appended to
                                      history -> one-step-ahead forecast

ARIMA walk-forward design
--------------------------
The spec's "ARIMA doesn't use engineered features, just the price series
itself" is honoured literally. To make ARIMA's prediction directly comparable
to naive/linear regression's "predict day t+1 using info through day t", we
do a **walk-forward one-step forecast**: fit once on the pre-test history,
then for each test date, append that date's *actual, already-realised* close
(today's real value — not future information) to the fitted model's state via
``append(refit=False)`` and forecast one step ahead. This avoids the
compounding error of a single static multi-step forecast over the whole test
horizon, which would not really be a "next-day" prediction for day 2 onward.
``refit=False`` re-uses the already-estimated ARMA coefficients (cheap Kalman
filter update only) rather than re-optimising them at every step, which is
both standard practice for a classical baseline and necessary for this to
run in reasonable time across 40 stocks.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA


def naive_forecast(today_actuals: pd.Series) -> pd.Series:
    """Naive "tomorrow = today" forecast.

    Parameters
    ----------
    today_actuals : today's actual Adj Close for each evaluation date (e.g.
        ``test_df['Adj Close']``). Predicting tomorrow's close as today's
        close means the prediction *is* this series, unchanged.

    Returns
    -------
    A copy of ``today_actuals``, representing the predicted close for the day
    *after* each index date — directly comparable to a ``target_next_close``
    series built the same way for the other models.
    """
    return today_actuals.copy()


def linear_regression_forecast(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame,
) -> np.ndarray:
    """Fit sklearn ``LinearRegression`` on engineered features, predict X_test.

    ``X_train``/``X_test`` should already be scaled (train-only-fit
    ``StandardScaler``, per ``src.scaling``) engineered features — never raw
    OHLCV. ``y_train`` is the next-day ``Adj Close`` target.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model.predict(X_test)


def arima_forecast(
    history: pd.Series,
    test_actuals: pd.Series,
    order: tuple[int, int, int] = (5, 1, 0),
) -> pd.Series:
    """Walk-forward one-step-ahead ARIMA forecast over a test period.

    Parameters
    ----------
    history : Adj Close series for all dates strictly BEFORE the test period
        (train + val). Used for the initial fit.
    test_actuals : Adj Close series for the test period (today's actual close
        for each test date). Each value is appended to the model's state
        (not used as a feature at fit time) immediately before forecasting
        the following day, so every forecast uses only information available
        "as of today."
    order : (p, d, q) ARIMA order. Not grid-searched — this is a classical
        baseline, not the model under study.

    Returns
    -------
    pd.Series indexed like ``test_actuals``, each value the one-step-ahead
    forecast for the day *after* that index date (comparable to the other
    baselines' ``target_next_close`` predictions).

    Raises
    ------
    Whatever exception statsmodels raises on a failed fit (e.g.
    ``numpy.linalg.LinAlgError``, ``ValueError``) — the caller is expected to
    catch this, try a fallback order, and log/exclude the stock if all orders
    fail. This function does not silently swallow failures.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = ARIMA(history.values, order=order).fit()

        preds = []
        for actual_today in test_actuals.values:
            res = res.append([actual_today], refit=False)
            preds.append(res.forecast(1)[0])

    return pd.Series(preds, index=test_actuals.index, name="arima_pred")


if __name__ == "__main__":
    # Smoke test against a cached, feature-engineered ticker.
    from pathlib import Path
    from src.scaling import chronological_split, scale_features
    from src.features import FEATURE_COLUMNS

    sample = Path(__file__).resolve().parents[1] / "data" / "processed" / "TCS_NS.csv"
    feats = pd.read_csv(sample, index_col="Date", parse_dates=True)
    feats["target_next_close"] = feats["Adj Close"].shift(-1)
    feats = feats.dropna(subset=["target_next_close"])

    train, val, test = chronological_split(feats, 0.70, 0.15)
    train_s, val_s, test_s, scaler = scale_features(train, val, test, FEATURE_COLUMNS)

    naive_pred = naive_forecast(test["Adj Close"])
    lr_pred = linear_regression_forecast(
        train_s[FEATURE_COLUMNS], train["target_next_close"], test_s[FEATURE_COLUMNS]
    )
    history = pd.concat([train["Adj Close"], val["Adj Close"]])
    arima_pred = arima_forecast(history, test["Adj Close"], order=(5, 1, 0))

    print("naive  head:", naive_pred.head(3).values)
    print("lr     head:", lr_pred[:3])
    print("arima  head:", arima_pred.head(3).values)
    print("actual head:", test["target_next_close"].head(3).values)

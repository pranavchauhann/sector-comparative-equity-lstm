"""
src/drift_performance.py
=========================
Phase 4 performance-drift detector: has the live prediction error diverged
from the model's training-time benchmark?

Signal
------
Rolling **20-trading-day mean APE** over the shadow log
(``data/prediction_log.csv``, built by src/log_predictions.py), compared to
the production benchmark MAPE in ``config/current_model_metrics.json``.

Two-part flag — BOTH must fire (documented rationale below):

1. **Magnitude**: latest rolling MAPE > production MAPE × (1 + 0.50).
   The +50% relative threshold is deliberately wide: day-to-day MAPE on
   this universe swings with volatility regimes (the production 1.29%
   routinely visits 1.6–1.7% in calm weeks), so a tight cutoff would page
   on noise weekly. A sustained 50% relative excess (≈1.9%+) has never
   occurred in the backtest window without a genuine regime change.
2. **Statistical confirmation** (the rigor beyond a raw cutoff): a
   one-sided **Mann-Whitney U test** comparing the most recent 20 days of
   per-prediction APEs against the preceding reference window (up to 60
   days). Drift flags only if recent errors are stochastically larger at
   p < 0.01. Mann-Whitney is chosen over a t-test because APE
   distributions are heavy-tailed and non-normal (a few large misses
   dominate); over KS because we specifically care about a *location*
   shift (errors got bigger), not any distributional difference.

Requiring magnitude AND significance means: a quiet week with one huge
outlier fails (1) — no flag; a slow persistent creep that is significant
but still small fails (2)'s pairing with (1) — the weekly retrain absorbs
it. Only "large and statistically real" triggers the out-of-schedule
retrain, which is exactly the event worth spending compute on.

Outputs: human log + ``$GITHUB_STEP_SUMMARY`` markdown, ``drift=true|false``
in ``$GITHUB_OUTPUT``, exit 0 either way (a flag is a signal, not an error).
``--plot`` also writes results/plots/drift_monitor.png (rolling live MAPE
vs the training benchmark).

Run:  python src/drift_performance.py [--plot] [--log-path ...]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG = ROOT / "data" / "prediction_log.csv"
PROD_JSON = ROOT / "config" / "current_model_metrics.json"
PLOT_PATH = ROOT / "results" / "plots" / "drift_monitor.png"

ROLL_DAYS = 20            # rolling window (trading days)
REL_THRESHOLD = 0.50      # +50% relative MAPE excess — see module docstring
P_VALUE = 0.01            # Mann-Whitney one-sided significance
REFERENCE_DAYS = 60       # historical APE window the recent one is tested against
MIN_DAYS = 25             # minimum realised days before the detector can run


def daily_mape(log: pd.DataFrame) -> pd.Series:
    """Mean APE per prediction_date across all stocks (realised rows only)."""
    realised = log.dropna(subset=["ape_pct"])
    return (realised.groupby("prediction_date")["ape_pct"].mean()
            .rename("daily_mape").sort_index())


def check(log_path: Path) -> dict:
    log = pd.read_csv(log_path)
    prod = json.loads(PROD_JSON.read_text())
    train_mape = prod["metrics"]["mean_MAPE"]

    daily = daily_mape(log)
    if len(daily) < MIN_DAYS:
        return {"status": "insufficient_data", "days": len(daily),
                "needed": MIN_DAYS, "drift": False, "train_mape": train_mape}

    rolling = daily.rolling(ROLL_DAYS).mean().dropna()
    latest = float(rolling.iloc[-1])
    rel_excess = latest / train_mape - 1
    magnitude_fired = rel_excess > REL_THRESHOLD

    realised = log.dropna(subset=["ape_pct"]).sort_values("prediction_date")
    recent_dates = daily.index[-ROLL_DAYS:]
    ref_dates = daily.index[-(ROLL_DAYS + REFERENCE_DAYS):-ROLL_DAYS]
    recent = realised.loc[realised["prediction_date"].isin(recent_dates), "ape_pct"]
    reference = realised.loc[realised["prediction_date"].isin(ref_dates), "ape_pct"]

    if len(reference) >= 30:
        stat, p = mannwhitneyu(recent, reference, alternative="greater")
        test_fired, p_val = bool(p < P_VALUE), float(p)
    else:  # too little history to test against — magnitude alone decides
        test_fired, p_val = magnitude_fired, None

    return {
        "status": "ok", "drift": bool(magnitude_fired and test_fired),
        "train_mape": train_mape, "rolling_mape": latest,
        "rel_excess_pct": rel_excess * 100, "magnitude_fired": magnitude_fired,
        "mannwhitney_p": p_val, "test_fired": test_fired,
        "n_recent": int(len(recent)), "n_reference": int(len(reference)),
        "daily": daily, "rolling": rolling,
    }


def plot(res: dict) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    daily, rolling = res["daily"], res["rolling"]
    x = pd.to_datetime(daily.index)
    ax.plot(x, daily.values, lw=0.8, alpha=0.45, color="#888",
            label="daily live MAPE")
    ax.plot(pd.to_datetime(rolling.index), rolling.values, lw=2,
            color="#2471a3", label=f"rolling {ROLL_DAYS}d live MAPE")
    ax.axhline(res["train_mape"], color="#2e7d52", ls="--", lw=1.5,
               label=f"training benchmark ({res['train_mape']:.2f}%)")
    thr = res["train_mape"] * (1 + REL_THRESHOLD)
    ax.axhline(thr, color="#c0392b", ls=":", lw=1.5,
               label=f"drift threshold (+{REL_THRESHOLD:.0%} → {thr:.2f}%)")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Live prediction error vs training benchmark (40-stock mean)")
    ax.legend(loc="upper left", fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_PATH, dpi=150)
    plt.close(fig)
    print(f"drift plot -> {PLOT_PATH}")


def emit(res: dict) -> None:
    if res["status"] == "insufficient_data":
        lines = ["### Performance drift check", "",
                 f"⏳ Insufficient realised history: {res['days']} days logged, "
                 f"{res['needed']} needed. No drift verdict.", ""]
    else:
        verdict = ("🚨 **Performance drift detected** — triggering retrain"
                   if res["drift"] else "✅ No performance drift detected")
        p_txt = ("n/a (short history — magnitude only)"
                 if res["mannwhitney_p"] is None else f"{res['mannwhitney_p']:.2e}")
        lines = [
            "### Performance drift check", "",
            "| | value |", "|---|---|",
            f"| training benchmark MAPE | {res['train_mape']:.4f}% |",
            f"| rolling {ROLL_DAYS}d live MAPE | {res['rolling_mape']:.4f}% |",
            f"| relative excess | {res['rel_excess_pct']:+.1f}% "
            f"(threshold +{REL_THRESHOLD:.0%}) → "
            f"{'FIRED' if res['magnitude_fired'] else 'ok'} |",
            f"| Mann-Whitney U p (recent {res['n_recent']} vs reference "
            f"{res['n_reference']} APEs) | {p_txt} → "
            f"{'FIRED' if res['test_fired'] else 'ok'} |",
            "", verdict, "",
        ]
    text = "\n".join(lines)
    print(text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(text + "\n")
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"drift={'true' if res['drift'] else 'false'}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    res = check(args.log_path)
    emit(res)
    if args.plot and res["status"] == "ok":
        plot(res)
    sys.exit(0)


if __name__ == "__main__":
    main()

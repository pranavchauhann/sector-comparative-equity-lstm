"""
src/drift_data.py
==================
Phase 4 data-drift detector: are the model's *inputs* still distributed the
way they were at training time?

This catches problems performance drift cannot see yet — a broken upstream
feed (stale prices → volatility collapses to ~0), a market regime change,
a corporate action mangling volume — because it needs **no realised
actuals**: it fires the moment the incoming feature distribution shifts,
often days before enough live errors accumulate to move a rolling MAPE.

Method: Population Stability Index (PSI), per feature, per stock
----------------------------------------------------------------
**Stationarity first** — the step naive drift checks skip: price-level
features (the moving averages, MACD and its signal line) trend with the
price itself, so their raw distribution *always* diverges from a
years-old training window on any trending stock. Raw PSI on them fires on
every feature every day — a permanently-red dashboard nobody trusts. So
level features are first mapped to stationary forms (divided by that day's
``Adj Close``: ``ma_20/price`` measures "how far below trend are we", not
"what rupee level is the average"), while ``daily_return``, ``rsi_14``,
``volatility_20`` and ``volume_ratio`` are scale-free already and used
as-is. What PSI then detects is a genuine change in market *dynamics*
(vol regime, trend shape, volume behaviour), not the trivial fact that
prices moved.

For each transformed feature, compare the **most recent 30 trading days**
against that stock's **training split** (the chronological 70% the
scalers/models were fit on):

1. Cut the training values into 10 equal-population (decile) bins.
2. Compute each bin's share in the recent window.
3. ``PSI = Σ (recent% − train%) · ln(recent% / train%)`` with the standard
   0.5/N smoothing for empty bins.

PSI is preferred over a KS test here because the recent window is small
(30 obs): KS p-values at n=30 are either insensitive or, across
40 stocks × 9 features = 360 simultaneous tests, drown in multiple-testing
noise. PSI is a stable *effect-size* measure.

Empirically calibrated threshold — why not the textbook 0.25
------------------------------------------------------------
The industry bands (0.10 moderate / 0.25 significant) assume the compared
sample is an *independent* draw from the reference. Thirty **consecutive**
trading days are nothing of the sort: these features are strongly
autocorrelated, so any single month lives in a narrow slice of the
multi-year distribution. Measured on this universe, 30-day windows drawn
*from inside the training period itself* show median PSIs of 0.2–1.6 —
the fixed 0.25 cutoff would flag permanent drift on data the model was
literally trained on. (Verified before choosing this design; a detector
that always fires is a detector nobody trusts.)

So the threshold is calibrated per stock × feature against its own null:
slide a 30-day window through the training split (10-day step), compute
the PSI of each — that is the distribution of "PSI when nothing is wrong"
— and flag the recent window only if it exceeds the **95th percentile** of
that null distribution.

Aggregation and flag
--------------------
A feature is **drifted** when **more than half of the 40 stocks** exceed
their own null-p95 for it. The majority vote deliberately requires the
shift to be universe-wide: one stock's idiosyncratic move (a split, a
single-name crash) should not trigger a full retrain, but a shift
affecting half the universe is a regime change by construction. The
detector reports *which* features drifted (diagnostic, per this project's
"diagnose why, not just that" rule), plus each drifted feature's worst
offenders by PSI excess over null.

Global flag: **any feature drifted** → ``drift=true``.

Outputs: human log + ``$GITHUB_STEP_SUMMARY``, ``drift=true|false`` in
``$GITHUB_OUTPUT``, exit 0 either way.

Run:  python src/drift_data.py [--recent-days 30] [--data-dir ...]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import load_universe, flat_tickers
from src.features import FEATURE_COLUMNS
from src.scaling import chronological_split

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data" / "processed"

N_BINS = 10
RECENT_DAYS = 30
NULL_STEP = 10           # stride of the sliding in-train null windows
NULL_PCTL = 95           # flag above this percentile of the null PSIs
MAJORITY = 0.5           # fraction of stocks that must exceed their null

# Price-level features → stationary form by dividing by that day's price;
# the rest are scale-free already. See module docstring.
PRICE_RELATIVE = {"ma_5", "ma_20", "ma_50", "macd", "macd_signal"}


def stationary_features(feats: pd.DataFrame) -> pd.DataFrame:
    out = feats[FEATURE_COLUMNS].copy()
    for col in PRICE_RELATIVE:
        out[col] = feats[col] / feats["Adj Close"]
    return out


def psi(train_values: np.ndarray, recent_values: np.ndarray,
        n_bins: int = N_BINS) -> float:
    """Population Stability Index of recent vs train, decile bins on train."""
    edges = np.quantile(train_values, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    edges = np.unique(edges)  # guard against ties collapsing bins

    t_counts, _ = np.histogram(train_values, bins=edges)
    r_counts, _ = np.histogram(recent_values, bins=edges)
    # 0.5-count smoothing so empty bins don't blow up the log term
    t_pct = (t_counts + 0.5) / (t_counts.sum() + 0.5 * len(t_counts))
    r_pct = (r_counts + 0.5) / (r_counts.sum() + 0.5 * len(r_counts))
    return float(np.sum((r_pct - t_pct) * np.log(r_pct / t_pct)))


def null_psis(train: pd.DataFrame, col: str, recent_days: int) -> np.ndarray:
    """PSI of every sliding in-train window vs the full train split."""
    vals = train[col].values
    out = []
    for start in range(0, len(vals) - recent_days + 1, NULL_STEP):
        out.append(psi(vals, vals[start:start + recent_days]))
    return np.array(out)


def check(data_dir: Path, recent_days: int = RECENT_DAYS) -> dict:
    tickers = flat_tickers(load_universe())
    rows = []
    for tk in tickers:
        f = data_dir / f"{tk.replace('.', '_')}.csv"
        if not f.exists():
            continue
        feats = pd.read_csv(f, index_col="Date", parse_dates=True)
        stat = stationary_features(feats)
        train, _, _ = chronological_split(stat, 0.70, 0.15)
        recent = stat.tail(recent_days)
        for col in FEATURE_COLUMNS:
            null = null_psis(train, col, recent_days)
            p95 = float(np.percentile(null, NULL_PCTL))
            recent_psi = psi(train[col].values, recent[col].values)
            rows.append({"ticker": tk, "feature": col, "psi": recent_psi,
                         "null_p95": p95, "excess": recent_psi - p95,
                         "exceeds": recent_psi > p95})
    df = pd.DataFrame(rows)

    by_feature = (df.groupby("feature")
                  .agg(share_exceeding=("exceeds", "mean"),
                       median_psi=("psi", "median"),
                       median_null_p95=("null_p95", "median"))
                  .sort_values("share_exceeding", ascending=False))
    drifted = by_feature[by_feature["share_exceeding"] > MAJORITY]
    worst = {feat: df[df["feature"] == feat].nlargest(3, "excess")
             for feat in drifted.index}
    return {"by_feature": by_feature, "drifted": drifted, "worst": worst,
            "drift": bool(len(drifted) > 0), "n_stocks": df["ticker"].nunique()}


def emit(res: dict, recent_days: int) -> None:
    lines = [f"### Data drift check (PSI vs per-stock empirical null, recent "
             f"{recent_days}d, {res['n_stocks']} stocks)", "",
             "| feature | stocks over own null p95 | median PSI "
             "(null p95) | verdict |", "|---|---|---|---|"]
    for feat, r in res["by_feature"].iterrows():
        verdict = ("🚨 drifted" if r["share_exceeding"] > MAJORITY else "ok")
        lines.append(
            f"| {feat} | {r['share_exceeding']:.0%} | {r['median_psi']:.2f} "
            f"({r['median_null_p95']:.2f}) | {verdict} |")
    lines.append("")
    if res["drift"]:
        feats = ", ".join(res["drifted"].index)
        lines.append(f"🚨 **Data drift detected** in: {feats} — triggering retrain")
        for feat, w in res["worst"].items():
            offenders = ", ".join(
                f"{r.ticker} (PSI {r.psi:.2f} vs null p95 {r.null_p95:.2f})"
                for r in w.itertuples())
            lines.append(f"- {feat} worst offenders: {offenders}")
    else:
        lines.append("✅ No data drift detected — no feature exceeds its "
                     f"empirical null on more than {MAJORITY:.0%} of stocks")
    lines.append("")
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
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--recent-days", type=int, default=RECENT_DAYS)
    args = parser.parse_args()

    res = check(args.data_dir, args.recent_days)
    emit(res, args.recent_days)
    sys.exit(0)


if __name__ == "__main__":
    main()

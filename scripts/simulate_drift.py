"""
scripts/simulate_drift.py
==========================
Phase 4 proof harness: inject synthetic drift and confirm BOTH detectors
flag it (and that they correctly pass on the unmodified data first).

Real drift may not occur naturally inside a testing window, so this script
manufactures the two failure modes the detectors exist to catch:

* **Data drift** — a crash-regime shock applied to the last 30 days of a
  temp copy of ``data/processed``: volatility trebled, returns shifted
  −2%/day, RSI collapsed toward oversold, volume ratio doubled. This is
  the "market changed sharply on a Tuesday" scenario.
* **Performance drift** — the last 20 days of a temp copy of the shadow
  prediction log get their APEs multiplied ×2.5 (rolling MAPE ≈ 3.1% vs
  the 1.29% benchmark, > +50% threshold), simulating a model gone stale.

Exit code 0 only if all four assertions hold:
  clean data → no data drift     | shocked data → data drift
  clean log  → no perf drift     | inflated log → perf drift

Run:  python scripts/simulate_drift.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.data_loader import load_universe, flat_tickers
from src import drift_data, drift_performance

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
LOG_PATH = ROOT / "data" / "prediction_log.csv"
SHOCK_DAYS = 30
APE_INFLATION = 2.5

np.random.seed(42)


def make_shocked_data(tmp: Path) -> Path:
    """Copy processed data, apply a crash-regime shock to the last 30 days."""
    out = tmp / "processed_shocked"
    shutil.copytree(PROCESSED_DIR, out)
    for tk in flat_tickers(load_universe()):
        f = out / f"{tk.replace('.', '_')}.csv"
        df = pd.read_csv(f, index_col="Date", parse_dates=True)
        tail = df.index[-SHOCK_DAYS:]
        df.loc[tail, "volatility_20"] *= 3.0
        df.loc[tail, "daily_return"] -= 0.02
        df.loc[tail, "rsi_14"] = df.loc[tail, "rsi_14"].clip(upper=35) * 0.7
        df.loc[tail, "volume_ratio"] *= 2.0
        df.to_csv(f)
    return out


def make_inflated_log(tmp: Path) -> Path:
    """Copy the shadow log, multiply the most recent 20 days' APEs by 2.5."""
    out = tmp / "prediction_log_inflated.csv"
    log = pd.read_csv(LOG_PATH)
    dates = sorted(log["prediction_date"].dropna().unique())
    recent = set(dates[-20:])
    mask = log["prediction_date"].isin(recent) & log["ape_pct"].notna()
    log.loc[mask, "ape_pct"] *= APE_INFLATION
    log.to_csv(out, index=False)
    return out


def main() -> None:
    results = {}
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        print("=" * 70)
        print("1/4  Data drift on CLEAN data (expect: no drift)")
        print("=" * 70)
        res = drift_data.check(PROCESSED_DIR)
        drift_data.emit(res, drift_data.RECENT_DAYS)
        results["clean_data_passes"] = not res["drift"]

        print("=" * 70)
        print("2/4  Data drift on SHOCKED data (expect: drift)")
        print("=" * 70)
        res = drift_data.check(make_shocked_data(tmp))
        drift_data.emit(res, drift_data.RECENT_DAYS)
        results["shocked_data_flagged"] = res["drift"]

        print("=" * 70)
        print("3/4  Performance drift on CLEAN log (expect: no drift)")
        print("=" * 70)
        res = drift_performance.check(LOG_PATH)
        drift_performance.emit(res)
        results["clean_log_passes"] = not res["drift"]

        print("=" * 70)
        print("4/4  Performance drift on INFLATED log (expect: drift)")
        print("=" * 70)
        res = drift_performance.check(make_inflated_log(tmp))
        drift_performance.emit(res)
        results["inflated_log_flagged"] = res["drift"]

    print("=" * 70)
    ok = all(results.values())
    for name, passed in results.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    print(f"\nsimulate_drift: {'ALL CHECKS PASSED' if ok else 'FAILURES ABOVE'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

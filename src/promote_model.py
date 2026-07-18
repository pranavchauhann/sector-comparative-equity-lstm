"""
src/promote_model.py
=====================
Phase 3 promotion gate: decide whether a freshly retrained set of LSTM
models may replace the current production models.

Why a gate at all
-----------------
"Always accept the newest model" is a dangerous default: a bad retraining
run — a Yahoo Finance outage returning truncated/garbage data, a silent
feature-engineering regression, an unlucky training seed — would silently
replace a good production model with a worse one every week. The gate makes
degradation impossible to miss: a candidate must *prove* it is at least as
good as production on the held-out test window, or nothing changes.

Promotion criteria (both must hold)
-----------------------------------
Metrics are the mean over the 40-stock locked universe, from the pipeline's
``results/lstm_metrics.csv`` (the same numbers every run also logs to
MLflow; the CSV is the DVC-tracked copy of record and exists even when the
CI job's local MLflow store does not).

1. **Directional accuracy must not regress**:
   ``new_diracc >= current_diracc - DIRACC_TOLERANCE_PP``
   Default tolerance: **0.0 percentage points** — "equal or better", exactly.
2. **MAPE must not meaningfully worsen**:
   ``new_mape <= current_mape * (1 + MAPE_TOLERANCE)``
   Default tolerance: **+5% relative** (e.g. production MAPE 1.285% allows
   up to 1.350%). Rationale: week-to-week MAPE jitters a few basis points
   from retraining nondeterminism alone; 5% relative is wide enough not to
   reject noise, tight enough to catch genuine degradation.

A rejected candidate is a **normal, expected outcome**, not an error: the
script always exits 0 unless something is actually broken (missing files),
and communicates the decision via stdout, ``$GITHUB_OUTPUT`` (promoted=
true/false) and a markdown table in ``$GITHUB_STEP_SUMMARY``.

Both tolerances can be overridden via environment variables
(``PROMOTE_DIRACC_TOLERANCE_PP``, ``PROMOTE_MAPE_TOLERANCE``) — used in CI
to deliberately test the reject path with an impossibly strict gate.

Usage
-----
    python src/promote_model.py            # decide + update JSON if promoted
    python src/promote_model.py --dry-run  # decide only, never write
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METRICS_CSV = ROOT / "results" / "lstm_metrics.csv"
PROD_JSON = ROOT / "config" / "current_model_metrics.json"

# Defaults — see module docstring for rationale. Env vars override (CI uses
# this to force-test the reject path).
DIRACC_TOLERANCE_PP = float(os.environ.get("PROMOTE_DIRACC_TOLERANCE_PP", "0.0"))
MAPE_TOLERANCE = float(os.environ.get("PROMOTE_MAPE_TOLERANCE", "0.05"))


def candidate_metrics() -> dict:
    # Rounded to the same 6-dp precision the production JSON stores, so an
    # identical model compares exactly equal (no float-dust rejections).
    df = pd.read_csv(METRICS_CSV)
    return {
        "mean_MAPE": round(float(df["lstm_MAPE"].mean()), 6),
        "mean_directional_accuracy": round(float(df["lstm_DirAcc"].mean()), 6),
        "mean_RMSE": round(float(df["lstm_RMSE"].mean()), 6),
        "mean_MAE": round(float(df["lstm_MAE"].mean()), 6),
        "n_stocks": int(len(df)),
    }


def production_metrics() -> dict:
    with open(PROD_JSON) as f:
        return json.load(f)


def decide(cand: dict, prod: dict) -> tuple[bool, list[str]]:
    """Apply the two-part gate; returns (promote?, human-readable checks)."""
    cur = prod["metrics"]
    checks = []

    diracc_floor = cur["mean_directional_accuracy"] - DIRACC_TOLERANCE_PP
    diracc_ok = cand["mean_directional_accuracy"] >= diracc_floor - 1e-9
    checks.append(
        f"{'PASS' if diracc_ok else 'FAIL'}  directional accuracy: "
        f"{cand['mean_directional_accuracy']:.3f}% vs required >= {diracc_floor:.3f}% "
        f"(production {cur['mean_directional_accuracy']:.3f}%, "
        f"tolerance {DIRACC_TOLERANCE_PP}pp)")

    mape_ceiling = cur["mean_MAPE"] * (1 + MAPE_TOLERANCE)
    mape_ok = cand["mean_MAPE"] <= mape_ceiling + 1e-9
    checks.append(
        f"{'PASS' if mape_ok else 'FAIL'}  MAPE: "
        f"{cand['mean_MAPE']:.4f}% vs allowed <= {mape_ceiling:.4f}% "
        f"(production {cur['mean_MAPE']:.4f}%, tolerance +{MAPE_TOLERANCE:.0%} relative)")

    return diracc_ok and mape_ok, checks


def git_short_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, cwd=ROOT,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001 — CI checkouts can be shallow/odd
        return "unknown"


def write_summary(cand: dict, prod: dict, promoted: bool, checks: list[str]) -> None:
    """Markdown table for $GITHUB_STEP_SUMMARY (also printed to stdout)."""
    cur = prod["metrics"]
    verdict = ("✅ **Promoted** — production model updated" if promoted else
               "🛑 **Not promoted** — new model did not meet promotion "
               "criteria; production model unchanged (this is a normal outcome)")
    lines = [
        "## Weekly retraining: promotion gate result", "",
        f"Production benchmark from `{prod.get('source_commit', '?')}` "
        f"(promoted {prod.get('promoted_at', '?')}); candidate = this run.", "",
        "| metric (mean of 40 stocks) | production | candidate |",
        "|---|---|---|",
        f"| MAPE | {cur['mean_MAPE']:.4f}% | {cand['mean_MAPE']:.4f}% |",
        f"| Directional accuracy | {cur['mean_directional_accuracy']:.3f}% "
        f"| {cand['mean_directional_accuracy']:.3f}% |",
        f"| RMSE (INR) | {cur['mean_RMSE']:.2f} | {cand['mean_RMSE']:.2f} |",
        f"| MAE (INR) | {cur['mean_MAE']:.2f} | {cand['mean_MAE']:.2f} |", "",
        *(f"- `{c}`" for c in checks), "",
        verdict, "",
    ]
    text = "\n".join(lines)
    print(text)
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a") as f:
            f.write(text + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="decide and report, but never update the "
                             "production pointer")
    args = parser.parse_args()

    cand = candidate_metrics()
    prod = production_metrics()
    promoted, checks = decide(cand, prod)

    write_summary(cand, prod, promoted, checks)

    if promoted and not args.dry_run:
        prod_out = {
            "description": prod["description"],
            "promoted_at": datetime.datetime.now(datetime.timezone.utc)
                            .isoformat(timespec="seconds"),
            "source_commit": git_short_sha(),
            "n_stocks": cand["n_stocks"],
            "metrics": {k: round(cand[k], 6) for k in
                        ("mean_MAPE", "mean_directional_accuracy",
                         "mean_RMSE", "mean_MAE")},
        }
        with open(PROD_JSON, "w") as f:
            json.dump(prod_out, f, indent=2)
            f.write("\n")
        print(f"production pointer updated: {PROD_JSON}")
    elif not promoted:
        print("New model did not meet promotion criteria — production model "
              "unchanged")

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"promoted={'true' if promoted else 'false'}\n")

    sys.exit(0)  # a rejected candidate is not a failure


if __name__ == "__main__":
    main()

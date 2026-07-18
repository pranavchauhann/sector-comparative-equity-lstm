"""
scripts/engineer_features.py
=============================
DVC pipeline stage 2: engineer_features.

Applies src.features.compute_features to every ticker in the locked
universe, using the raw data cached by stage 1 (data/raw/), and writes the
engineered per-stock CSVs to data/processed/. Extends the single-stock smoke
test in src/features.py's __main__ to the full 40-stock universe (this is
the same logic notebooks/02_feature_engineering.ipynb runs interactively).

Run directly:  python scripts/engineer_features.py
Run via DVC:   dvc repro engineer_features
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import load_universe, flat_tickers, load_cached
from src.features import compute_features

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"


def main() -> None:
    universe = load_universe()
    tickers = flat_tickers(universe)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    n_ok, n_failed = 0, []
    for tk in tickers:
        try:
            raw = load_cached(tk)
        except FileNotFoundError as exc:
            n_failed.append((tk, str(exc)))
            continue
        feats = compute_features(raw)
        feats.to_csv(PROCESSED_DIR / f"{tk.replace('.', '_')}.csv")
        n_ok += 1

    print(f"engineer_features: {n_ok}/{len(tickers)} tickers written to "
          f"{PROCESSED_DIR}")
    if n_failed:
        print(f"  failed (no raw cache): {n_failed}")


if __name__ == "__main__":
    main()

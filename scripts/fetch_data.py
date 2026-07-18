"""
scripts/fetch_data.py
======================
DVC pipeline stage 1: fetch_data.

Pulls 5 years of daily adjusted OHLCV for every ticker in the locked
universe (config/universe.json) and caches it to data/raw/. Thin CLI
wrapper around src.data_loader — the universe is read, never re-ranked,
consistent with the project's "universe locked and dated" rule.

Run directly:  python scripts/fetch_data.py
Run via DVC:   dvc repro fetch_data
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import load_universe, flat_tickers, download_history, adj_close_panel


def main() -> None:
    universe = load_universe()
    tickers = flat_tickers(universe)
    data = download_history(tickers, years=5)
    panel = adj_close_panel(data)
    print(f"\nfetch_data: {panel.shape[0]} rows x {panel.shape[1]} tickers "
          f"cached to data/raw/")


if __name__ == "__main__":
    main()

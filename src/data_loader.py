"""
src/data_loader.py
==================
Reusable helpers to pull, cache, and load historical daily price data for the
project universe.

Design notes
------------
* **Adjusted prices.** We download with ``auto_adjust=False`` and keep the
  ``Adj Close`` column. Indian equities have frequent bonus issues and splits;
  ``Adj Close`` back-adjusts for these so a price series is continuous and
  comparable through time. All modelling should use ``Adj Close``, not ``Close``.
* **Universe comes from config, not a live fetch.** ``load_universe`` reads the
  locked ``config/universe.json`` produced by ``src/universe.py``. This module
  never re-ranks or re-selects tickers — the universe stays frozen between
  explicit refreshes (a Non-Negotiable Rule of the project).
* **Local cache.** Raw per-ticker data is cached under ``data/raw/`` as CSV.
  ``data/`` is gitignored (regenerable). Re-running is cheap: pass
  ``force=False`` (default) to reuse the cache.

Typical use
-----------
    from src.data_loader import load_universe, download_history, adj_close_panel

    uni = load_universe()
    tickers = flat_tickers(uni)
    data = download_history(tickers, years=5)      # dict[ticker] -> DataFrame
    panel = adj_close_panel(data)                   # wide Adj Close DataFrame
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "universe.json"
RAW_DIR = ROOT / "data" / "raw"

# Columns we keep from Yahoo, in a tidy order.
KEEP_COLS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


# --------------------------------------------------------------------------- #
# Universe
# --------------------------------------------------------------------------- #
def load_universe(path: Path | str = CONFIG_PATH) -> dict:
    """Load the locked universe dict written by src/universe.py."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python src/universe.py` first to build "
            "and lock the universe."
        )
    return json.loads(path.read_text())


def flat_tickers(universe: dict) -> list[str]:
    """All tickers across all sectors, as a flat de-duplicated list (order kept)."""
    seen: dict[str, None] = {}
    for rows in universe["sectors"].values():
        for row in rows:
            seen.setdefault(row["ticker"], None)
    return list(seen)


def ticker_to_sector(universe: dict) -> dict[str, str]:
    """Map each ticker to its sector name."""
    return {
        row["ticker"]: sector
        for sector, rows in universe["sectors"].items()
        for row in rows
    }


# --------------------------------------------------------------------------- #
# Download / cache
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str) -> Path:
    return RAW_DIR / f"{ticker.replace('.', '_')}.csv"


def _download_one(ticker: str, start: str, end: str,
                  retries: int = 3, pause: float = 1.5) -> pd.DataFrame | None:
    """Download a single ticker's OHLCV+AdjClose, with retries. None on failure."""
    for attempt in range(1, retries + 1):
        try:
            df = yf.download(
                ticker, start=start, end=end,
                auto_adjust=False, progress=False, threads=False,
            )
            if df is None or df.empty:
                raise ValueError("empty frame returned")
            # yfinance may return a MultiIndex (field, ticker) for single tickers.
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[[c for c in KEEP_COLS if c in df.columns]].copy()
            df.index.name = "Date"
            return df
        except Exception as exc:  # noqa: BLE001
            print(f"    [{ticker}] attempt {attempt}/{retries} failed: {exc}")
            time.sleep(pause * attempt)
    return None


def download_history(
    tickers: list[str],
    years: int = 5,
    end: str | None = None,
    save_dir: Path | str = RAW_DIR,
    force: bool = False,
) -> dict[str, pd.DataFrame]:
    """Pull ``years`` of daily data for ``tickers``, caching per-ticker CSVs.

    Parameters
    ----------
    tickers : list of Yahoo tickers (e.g. ``["RELIANCE.NS", ...]``).
    years   : lookback window in years (default 5).
    end     : end date ``YYYY-MM-DD`` (default: today).
    save_dir: cache directory (default ``data/raw``).
    force   : if True, re-download even when a cache file exists.

    Returns
    -------
    dict mapping ticker -> DataFrame indexed by Date with columns from KEEP_COLS.
    Tickers that fail entirely are omitted from the result (and reported).
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    end_ts = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()
    start_ts = end_ts - pd.DateOffset(years=years)
    start, end_s = start_ts.strftime("%Y-%m-%d"), end_ts.strftime("%Y-%m-%d")

    out: dict[str, pd.DataFrame] = {}
    failed: list[str] = []
    print(f"Fetching {len(tickers)} tickers | {start} -> {end_s} "
          f"(force={force})")

    for i, tk in enumerate(tickers, 1):
        cache = _cache_path(tk)
        if cache.exists() and not force:
            df = pd.read_csv(cache, index_col="Date", parse_dates=True)
            out[tk] = df
            print(f"  [{i:>2}/{len(tickers)}] {tk:<16} cached  ({len(df)} rows)")
            continue

        df = _download_one(tk, start, end_s)
        if df is None:
            failed.append(tk)
            print(f"  [{i:>2}/{len(tickers)}] {tk:<16} FAILED")
            continue
        df.to_csv(cache)
        out[tk] = df
        print(f"  [{i:>2}/{len(tickers)}] {tk:<16} downloaded ({len(df)} rows)")
        time.sleep(0.3)  # polite pacing for Yahoo

    if failed:
        print(f"\nWARNING: {len(failed)} ticker(s) failed: {failed}")
    print(f"Done: {len(out)}/{len(tickers)} tickers available.")
    return out


def load_cached(ticker: str, save_dir: Path | str = RAW_DIR) -> pd.DataFrame:
    """Load a single cached ticker CSV (raises if not cached)."""
    path = Path(save_dir) / f"{ticker.replace('.', '_')}.csv"
    if not path.exists():
        raise FileNotFoundError(f"No cache for {ticker} at {path}. "
                                "Run download_history first.")
    return pd.read_csv(path, index_col="Date", parse_dates=True)


# --------------------------------------------------------------------------- #
# Convenience transforms
# --------------------------------------------------------------------------- #
def adj_close_panel(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Wide DataFrame of Adj Close, one column per ticker, aligned on Date."""
    series = {tk: df["Adj Close"] for tk, df in data.items() if "Adj Close" in df}
    return pd.DataFrame(series).sort_index()


if __name__ == "__main__":
    # Smoke test: load the locked universe and pull everything into the cache.
    uni = load_universe()
    tks = flat_tickers(uni)
    data = download_history(tks, years=5)
    panel = adj_close_panel(data)
    print(f"\nAdj Close panel: {panel.shape[0]} rows x {panel.shape[1]} tickers")
    print(panel.tail(3).iloc[:, :4])

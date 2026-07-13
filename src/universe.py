"""
src/universe.py
===============
Build the investable universe for the project: the top 10 stocks by market
capitalisation in each of four NSE sectors, and lock the result to
``config/universe.json`` with a source attribution and a fetch date.

Sourcing approach (see README for the full rationale)
-----------------------------------------------------
The *ideal* source is the official NSE sectoral-index constituent CSVs
(Nifty IT / Nifty Bank / Nifty Financial Services / Nifty Energy / Nifty FMCG).
In practice ``nseindia.com`` aggressively blocks programmatic access
(the request returns no response from data-centre / CI IPs), so we cannot rely
on scraping it inside a reproducible script.

Instead we take a two-step, defensible approach:

1.  **Candidate pools** — a curated superset of each Nifty sectoral index's
    constituents (hard-coded below, dated). This is the "which names belong to
    this sector" decision, which is stable and comes from the NSE index
    definitions. It is a *superset*: we deliberately list more than 10 names per
    sector so the market-cap ranking has something to choose from.

2.  **Live market-cap ranking** — for every candidate we pull the current market
    capitalisation from Yahoo Finance (``yfinance``) and keep the top 10. This is
    the part that must not be stale, and it *is* fetched programmatically every
    time this script runs.

The final list should still be cross-checked by a human against Moneycontrol /
ET Markets before being relied upon — the candidate pools are refreshed by hand
whenever the indices are reconstituted (NSE reviews them semi-annually).

Run
---
    python src/universe.py                 # writes config/universe.json
    python src/universe.py --top-n 5       # 5 per sector instead of 10
    python src/universe.py --dry-run       # print, don't write

Idempotent: safe to re-run. It always overwrites config/universe.json with a
fresh market-cap snapshot and a new fetch date. Downstream code (data_loader,
notebooks, training) must READ config/universe.json and never call this module,
so the universe stays locked between explicit refreshes.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import yfinance as yf

# --------------------------------------------------------------------------- #
# Candidate pools per sector.
#
# Basis: constituents of the corresponding NSE sectoral indices, as published on
# nseindia.com. These are SUPERSETS (12-16 names) — the market-cap ranking below
# trims each to the top 10. Tickers use Yahoo Finance's ".NS" (NSE) suffix.
#
# Last hand-refreshed against the Nifty index factsheets: see CANDIDATE_AS_OF.
# --------------------------------------------------------------------------- #
CANDIDATE_AS_OF = "2026-07-14"

CANDIDATE_POOLS: dict[str, list[str]] = {
    "Information Technology": [
        "TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS",
        "LTIM.NS", "PERSISTENT.NS", "COFORGE.NS", "MPHASIS.NS", "LTTS.NS",
        "OFSS.NS", "TATAELXSI.NS", "KPITTECH.NS",
    ],
    "Banking & Financial Services": [
        "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS",
        "BAJFINANCE.NS", "BAJAJFINSV.NS", "INDUSINDBK.NS", "HDFCLIFE.NS",
        "SBILIFE.NS", "SHRIRAMFIN.NS", "CHOLAFIN.NS", "PFC.NS", "RECLTD.NS",
        "JIOFIN.NS", "ICICIGI.NS",
    ],
    "Energy": [
        "RELIANCE.NS", "ONGC.NS", "NTPC.NS", "POWERGRID.NS", "COALINDIA.NS",
        "BPCL.NS", "IOC.NS", "GAIL.NS", "TATAPOWER.NS", "ADANIGREEN.NS",
        "ADANIENSOL.NS", "ADANIPOWER.NS", "NHPC.NS",
    ],
    "FMCG": [
        "HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "VBL.NS", "BRITANNIA.NS",
        "TATACONSUM.NS", "GODREJCP.NS", "DABUR.NS", "MARICO.NS", "COLPAL.NS",
        "UBL.NS", "PGHH.NS", "EMAMILTD.NS",
    ],
}

SECTOR_INDEX_SOURCE = {
    "Information Technology": "Nifty IT index constituents (nseindia.com)",
    "Banking & Financial Services": (
        "Nifty Bank + Nifty Financial Services index constituents (nseindia.com)"
    ),
    "Energy": "Nifty Energy index constituents (nseindia.com)",
    "FMCG": "Nifty FMCG index constituents (nseindia.com)",
}

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "universe.json"


def fetch_market_cap(ticker: str, retries: int = 3, pause: float = 1.0) -> float | None:
    """Return the current market cap for a Yahoo ticker, or None if unavailable.

    Tries the lightweight ``fast_info`` endpoint first, then falls back to the
    heavier ``.info`` dict. Retries a few times because Yahoo rate-limits
    (HTTP 429) under bursty access.
    """
    for attempt in range(1, retries + 1):
        try:
            t = yf.Ticker(ticker)
            mcap = None
            try:
                mcap = t.fast_info.get("marketCap")
            except Exception:
                mcap = None
            if not mcap:
                info = t.info  # heavier, but has marketCap when fast_info lacks it
                mcap = info.get("marketCap")
            if mcap:
                return float(mcap)
        except Exception as exc:  # noqa: BLE001 - report and retry
            print(f"    [{ticker}] attempt {attempt}/{retries} failed: {exc}")
        time.sleep(pause * attempt)  # linear backoff
    return None


# 1 lakh crore = 1e5 crore = 1e5 * 1e7 rupees = 1e12 rupees.
RUPEES_PER_LAKH_CRORE = 1e12


def rank_sector(sector: str, candidates: list[str],
                top_n: int) -> tuple[list[dict], list[str]]:
    """Fetch market caps for a sector's candidates and return (top_n rows, missing).

    ``missing`` is the list of candidate tickers Yahoo could not price (e.g. a
    symbol Yahoo does not carry); these are excluded from the ranking and
    recorded in the universe metadata for transparency.
    """
    print(f"\n[{sector}] ranking {len(candidates)} candidates by market cap ...")
    rows: list[dict] = []
    for tk in candidates:
        mcap = fetch_market_cap(tk)
        status = (f"{mcap/RUPEES_PER_LAKH_CRORE:,.2f} lakh-cr"
                  if mcap else "UNAVAILABLE")
        print(f"    {tk:<16} {status}")
        rows.append({"ticker": tk, "market_cap": mcap})
        time.sleep(0.4)  # be polite to Yahoo

    ranked = [r for r in rows if r["market_cap"] is not None]
    ranked.sort(key=lambda r: r["market_cap"], reverse=True)
    missing = [r["ticker"] for r in rows if r["market_cap"] is None]
    if missing:
        print(f"    WARNING: no market cap for {missing} (excluded from ranking)")

    if len(ranked) < top_n:
        raise RuntimeError(
            f"[{sector}] only {len(ranked)} candidates had a market cap; "
            f"need at least {top_n}. Check network / candidate pool."
        )

    top = ranked[:top_n]
    for rank, r in enumerate(top, start=1):
        r["rank"] = rank
        r["market_cap_lakh_cr"] = round(r["market_cap"] / RUPEES_PER_LAKH_CRORE, 2)
    return top, missing


def build_universe(top_n: int = 10) -> dict:
    """Build the full universe dict (all sectors) with metadata."""
    fetched_at = datetime.now(timezone.utc).astimezone()
    sectors: dict[str, list[dict]] = {}
    excluded: dict[str, list[str]] = {}
    for sector, candidates in CANDIDATE_POOLS.items():
        top, missing = rank_sector(sector, candidates, top_n)
        sectors[sector] = top
        if missing:
            excluded[sector] = missing

    universe = {
        "metadata": {
            "description": (
                "Top stocks by market capitalisation per NSE sector for the "
                "Sector-Comparative Indian Equity Forecasting project."
            ),
            "top_n_per_sector": top_n,
            "n_sectors": len(sectors),
            "n_tickers": sum(len(v) for v in sectors.values()),
            "fetch_date": fetched_at.strftime("%Y-%m-%d"),
            "fetch_timestamp": fetched_at.isoformat(),
            "candidate_pool_as_of": CANDIDATE_AS_OF,
            "market_cap_source": "Yahoo Finance via yfinance (live at fetch time)",
            "candidate_pool_source": SECTOR_INDEX_SOURCE,
            "excluded_no_market_cap": excluded,
            "excluded_note": (
                "Candidates Yahoo Finance could not price (e.g. a symbol Yahoo "
                "does not carry, such as LTIMindtree/LTIM). Excluded from ranking "
                "because we also cannot fetch their price history downstream."
            ),
            "sourcing_note": (
                "Sector membership comes from NSE sectoral-index constituents "
                "(hand-curated superset). Market caps are pulled live from Yahoo "
                "Finance and used to rank each pool to the top N. Cross-check "
                "against Moneycontrol / ET Markets before relying on this list."
            ),
            "currency": "INR",
        },
        "sectors": sectors,
    }
    return universe


def flat_ticker_list(universe: dict) -> list[str]:
    """Convenience: all tickers across all sectors as a flat list."""
    return [row["ticker"] for rows in universe["sectors"].values() for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-n", type=int, default=10,
                        help="stocks per sector (default 10)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the result but do not write config/universe.json")
    args = parser.parse_args()

    universe = build_universe(top_n=args.top_n)

    print("\n" + "=" * 60)
    print("FINAL UNIVERSE")
    print("=" * 60)
    for sector, rows in universe["sectors"].items():
        print(f"\n{sector}:")
        for r in rows:
            print(f"  {r['rank']:>2}. {r['ticker']:<16} "
                  f"Rs {r['market_cap_lakh_cr']:>6,.2f} lakh-cr")
    total = universe["metadata"]["n_tickers"]
    print(f"\nTotal tickers: {total}")

    if args.dry_run:
        print("\n[dry-run] not writing config/universe.json")
        return

    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(universe, indent=2))
    print(f"\nWrote {CONFIG_PATH.relative_to(Path.cwd()) if CONFIG_PATH.is_relative_to(Path.cwd()) else CONFIG_PATH}")


if __name__ == "__main__":
    main()

# Sector-Comparative Indian Equity Forecasting with LSTM

Predict the **next-day closing price** for the top 10 stocks by market cap in each
of four NSE sectors — **Information Technology, Banking & Financial Services,
Energy, and FMCG** — and honestly compare an LSTM against simpler baselines
(naive, linear regression, ARIMA) to see where deep learning actually helps.

> **Not investment advice.** This is a portfolio project demonstrating ML
> technique on Indian equity data. Stock-price predictability is fundamentally
> limited by market efficiency; the point of the project is a *rigorous, honest*
> model comparison, not a trading signal.

### Why this framing
Most stock-LSTM portfolio projects skip baselines and report inflated accuracy.
The differentiator here is intellectual honesty: strong baselines, chronological
splits, no data leakage, and a willingness to report "the LSTM barely beat naive"
where that is what the data shows.

---

## Project status

| Phase | Scope | Status |
|-------|-------|--------|
| **1** | Universe selection & EDA | ✅ **Done** |
| 2 | Feature engineering & scaling pipeline | ⏳ Next |
| 3 | Baselines (naive, linear regression, ARIMA) | ⬜ |
| 4 | LSTM (2 stacked layers, early stopping) | ⬜ |
| 5 | Streamlit app + Community Cloud deploy | ⬜ |
| 6 | README polish + final GitHub push | ⬜ |

---

## Phase 1 — Universe & EDA (complete)

### How the 40 tickers were selected

**Two-step approach: NSE sector membership + live market-cap ranking.**

1. **Sector membership (candidate pools).** For each sector I start from the
   constituents of the corresponding **NSE sectoral index** — Nifty IT, Nifty
   Bank / Nifty Financial Services, Nifty Energy, Nifty FMCG. These are curated
   as *superset* candidate pools (12–16 names each) in
   [`src/universe.py`](src/universe.py), dated `candidate_pool_as_of`.
2. **Ranking (top 10).** `src/universe.py` then pulls the **current market
   capitalisation for every candidate live from Yahoo Finance** and keeps the
   top 10 per sector. This is the part that must not be stale, and it is fetched
   programmatically on every run.

The result is written to [`config/universe.json`](config/universe.json) with the
market-cap source, the fetch date, and per-ticker market caps — then **locked**.
Downstream code reads that file and never re-ranks, so the universe stays frozen
between explicit refreshes (`python src/universe.py`).

**Why not scrape NSE directly?** The spec's first-choice source is the NSE
sectoral-index constituent CSVs. In practice `nseindia.com` blocks programmatic
access (requests from non-browser / data-centre IPs return nothing), so scraping
it is not reproducible inside a script. Rather than hard-code a stale list, I use
the NSE index *definitions* for sector membership (which change only at
semi-annual reconstitution) and do the volatile part — the market-cap ranking —
live via Yahoo. This is honest, dated, and mostly programmatic. **Cross-check the
final list against Moneycontrol / ET Markets before relying on it.**

**Market-cap snapshot: `fetch_date` in `config/universe.json` (built 2026-07-14).**

<details>
<summary><b>The locked universe (top 10 per sector, by market cap)</b></summary>

| IT | Banking & Fin. Svcs | Energy | FMCG |
|----|--------------------|--------|------|
| TCS | HDFCBANK | RELIANCE | HINDUNILVR |
| INFY | ICICIBANK | ADANIPOWER | ITC |
| HCLTECH | SBIN | NTPC | NESTLEIND |
| WIPRO | BAJFINANCE | ONGC | VBL |
| TECHM | AXISBANK | POWERGRID | BRITANNIA |
| OFSS | KOTAKBANK | COALINDIA | MARICO |
| PERSISTENT | BAJAJFINSV | ADANIGREEN | GODREJCP |
| COFORGE | SHRIRAMFIN | IOC | TATACONSUM |
| MPHASIS | SBILIFE | ADANIENSOL | DABUR |
| LTTS | JIOFIN | BPCL | COLPAL |

Tickers use the Yahoo `.NS` (NSE) suffix in code and data.
</details>

**Known deviation — LTIMindtree (LTIM).** Yahoo Finance carries no data for
LTIM.NS under any symbol variant (confirmed 404 for `LTIM.NS`, `LTIMINDTREE.NS`,
`MINDTREE.NS`, `LTI.NS`). A large IT name that would otherwise rank ~top-5 is
therefore excluded, and **OFSS** takes the 10th IT slot. Since we could not fetch
LTIM's *price history* either, it could not be modelled regardless. This is
recorded in `config/universe.json` under `metadata.excluded_no_market_cap`.

### Data

- **Source:** Yahoo Finance via `yfinance` (`.NS` tickers).
- **Window:** 5 years of daily data ending on the fetch date (2021-07-14 →
  2026-07-10; the most recent unsettled session is dropped).
- **Adjusted prices:** downloaded with `auto_adjust=False`, modelled on
  **`Adj Close`** so splits and bonus issues are handled correctly.
- **Cache:** one CSV per ticker under `data/raw/` (gitignored — regenerable via
  `src/data_loader.py`).

### EDA highlights (see [`notebooks/01_universe_and_eda.ipynb`](notebooks/01_universe_and_eda.ipynb))

- **Coverage:** 39 of 40 tickers have the full ~1,236 trading days. **JIOFIN**
  (Jio Financial Services) is the exception at ~715 days — it listed in
  **Aug 2023** after demerging from Reliance. Flagged now so train/val/test
  sizing accounts for it.
- **Split/bonus adjustment works:** historical `Adj Close / Close` diverges most
  for **Coal India (−34%)**, BPCL, ONGC, IOC — consistent with heavy
  dividend/bonus histories — and converges to 1.0 at the latest date.
- **Volatility:** Adani names are the most volatile (~50% annualised); FMCG
  staples (Nestlé, ITC, HUL) the least (~19–20%).
- **Structure:** mean **intra-sector** return correlation **0.42** clearly
  exceeds mean **inter-sector 0.19** (IT most cohesive at 0.55), which justifies
  the sector-comparative framing.

Plots saved to [`results/plots/`](results/plots/): price history by sector,
return distributions, the adjustment check, and the correlation heatmap.

---

## Repository structure

```
├── README.md
├── requirements.txt
├── .gitignore
├── config/
│   └── universe.json          # locked, dated top-10-per-sector ticker list
├── data/                      # raw per-ticker CSVs (gitignored, regenerable)
├── src/
│   ├── universe.py            # fetch top 10 per sector by live market cap
│   └── data_loader.py         # reusable historical-data loader + cache
├── notebooks/
│   └── 01_universe_and_eda.ipynb
└── results/
    └── plots/                 # EDA figures (Phase 1); metrics to follow
```
*(Later phases add `src/features.py`, `baselines.py`, `lstm_model.py`,
`evaluate.py`, notebooks 02–04, and `app/app.py`.)*

## Reproduce Phase 1

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python src/universe.py                     # build & lock config/universe.json
python src/data_loader.py                  # cache 5yr data for all 40 tickers
jupyter nbconvert --to notebook --execute \
  --inplace notebooks/01_universe_and_eda.ipynb   # or open interactively
```

The notebook runs top-to-bottom with no manual intervention. It **reads** the
locked universe; to refresh the market-cap snapshot, re-run `src/universe.py`.

### Non-negotiable rules honoured
No fabricated results · chronological splits only (Phase 3+) · scaler fit on
train only (Phase 2) · universe locked and dated · everything checked into Git as
we go · every notebook runs end-to-end.

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
| **2** | Feature engineering & scaling pipeline | ✅ **Done** |
| 3 | Baselines (naive, linear regression, ARIMA) | ⏳ Next |
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

## Phase 2 — Feature Engineering (complete)

### Features

Per stock, 9 engineered columns computed via **`pandas_ta`** (not hand-rolled
formulas — [`src/features.py`](src/features.py)):

| Feature | Description |
|---|---|
| `daily_return` | % change in `Adj Close` |
| `ma_5`, `ma_20`, `ma_50` | Simple moving averages of `Adj Close` |
| `rsi_14` | 14-day RSI |
| `macd`, `macd_signal` | MACD line (12,26) and signal line (9) |
| `volatility_20` | 20-day rolling std of `daily_return` |
| `volume_ratio` | Volume ÷ 20-day average volume |

**Warm-up:** every stock loses **49–51 rows** to indicator warm-up — `MA_50`
(needing 50 prior observations) is the binding constraint; a trailing
unsettled-session row adds one more where present. Documented per stock in
[`notebooks/02_feature_engineering.ipynb`](notebooks/02_feature_engineering.ipynb).
Engineered CSVs are saved to `data/processed/` (gitignored, regenerable).

**Redundancy check:** `ma_5`/`ma_20`/`ma_50` are highly correlated (r ≈ 0.94–0.98)
and so are `macd`/`macd_signal` (r ≈ 0.95) — expected, since they're smoothed
derivatives of the same price series. Kept for now; revisiting via ablation is a
later-phase decision, not a Phase 2 one.

**Sanity checks (asserted in the notebook, not just eyeballed):** `rsi_14` stays
within `[0, 100]` across all 40 stocks (observed range ≈ [10.7, 92.1]);
`volatility_20` is non-negative throughout.

### Scaling — chronological split, train-only fit

[`src/scaling.py`](src/scaling.py) implements:
- `chronological_split(df, train_frac=0.70, val_frac=0.15)` — **no shuffling**,
  first 70% of dates → train, next 15% → val, remaining 15% → test.
- `scale_features(train, val, test, feature_cols)` — fits `StandardScaler` on
  **train only**, applies the same fitted scaler to val/test. Scaled train
  features have mean ≈ 0 / std ≈ 1 by construction; scaled test features do
  **not** (confirmed in the notebook) — which is exactly the leakage-free
  behaviour we want, since test is scaled with train's statistics, not its own.

### Leakage guard

The notebook's **final cell** prints the exact 9-column feature list and
asserts it is disjoint from the raw OHLCV columns (`Open/High/Low/Close/Volume`).
`Adj Close` itself is also excluded from modeling features — it's kept only as
the prediction target / for plotting.

### Environment note

`pandas-ta` (as of `0.4.71b0` on PyPI) requires **Python ≥3.12**; the project
venv was rebuilt from 3.11 → **3.12** to support it. This also required
switching `tensorflow-macos` (deprecated by Apple, stuck at 2.16.2) to plain
**`tensorflow`** (native arm64 wheels from 2.16+), and relaxing the `numpy` pin
so `pip` could resolve a consistent set across `pandas-ta`, `scipy`, and
`tensorflow` together. See [`requirements.txt`](requirements.txt).

See [`notebooks/02_feature_engineering.ipynb`](notebooks/02_feature_engineering.ipynb)
for the full run, including the correlation heatmap and RSI/volatility
distribution plots (saved to `results/plots/`).

---

## Repository structure

```
├── README.md
├── requirements.txt
├── .gitignore
├── config/
│   └── universe.json          # locked, dated top-10-per-sector ticker list
├── data/                      # raw + processed per-ticker CSVs (gitignored, regenerable)
├── src/
│   ├── universe.py            # fetch top 10 per sector by live market cap
│   ├── data_loader.py         # reusable historical-data loader + cache
│   ├── features.py            # compute_features(): engineered indicators via pandas-ta
│   └── scaling.py             # chronological split + train-only StandardScaler
├── notebooks/
│   ├── 01_universe_and_eda.ipynb
│   └── 02_feature_engineering.ipynb
└── results/
    └── plots/                 # EDA + feature-engineering figures; metrics to follow
```
*(Later phases add `baselines.py`, `lstm_model.py`, `evaluate.py`,
notebooks 03–04, and `app/app.py`.)*

## Reproduce Phase 1 + 2

```bash
python3.12 -m venv .venv && source .venv/bin/activate   # pandas-ta requires Python >=3.12
pip install -r requirements.txt

python src/universe.py                     # build & lock config/universe.json
python src/data_loader.py                  # cache 5yr data for all 40 tickers
jupyter nbconvert --to notebook --execute \
  --inplace notebooks/01_universe_and_eda.ipynb          # Phase 1 EDA
jupyter nbconvert --to notebook --execute \
  --inplace notebooks/02_feature_engineering.ipynb       # Phase 2 features + scaling demo
```

Both notebooks run top-to-bottom with no manual intervention. They **read** the
locked universe; to refresh the market-cap snapshot, re-run `src/universe.py`.

### Non-negotiable rules honoured
No fabricated results · chronological splits only (Phase 3+) · scaler fit on
train only (Phase 2) · universe locked and dated · everything checked into Git as
we go · every notebook runs end-to-end.

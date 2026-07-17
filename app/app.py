"""
app/app.py
==========
Streamlit dashboard for the Sector-Comparative Indian Equity Forecasting
project. Reads only precomputed artifacts (config/universe.json and the
results/ CSVs) — it never retrains a model or fetches live data from Yahoo
Finance. This keeps page loads fast and avoids rate-limiting real users, per
the project's architecture rule.

Run locally:  streamlit run app/app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
UNIVERSE_PATH = ROOT / "config" / "universe.json"
FINAL_COMPARISON_PATH = ROOT / "results" / "final_comparison_with_returns_lstm.csv"
MULTI_HORIZON_PATH = ROOT / "results" / "multi_horizon_comparison.csv"
HORIZON_PRED_DIR = ROOT / "results" / "predictions_horizons"
LATEST_FORECASTS_PATH = ROOT / "results" / "latest_forecasts.csv"

HORIZON_LABELS = {          # UI label -> directory/key in the results CSV
    "Next day": "next_day",
    "Next month": "next_month",
    "Next year": "next_year",
}
HORIZON_DAYS = {"next_day": 1, "next_month": 21, "next_year": 252}

MODEL_COLORS = {
    "actual": "#111111",
    "naive": "#9c9c9c",
    "linreg": "#e07b39",
    "lstm": "#2471a3",
}

st.set_page_config(
    page_title="Sector-Comparative Indian Equity Forecasting",
    page_icon="\U0001F4C8",
    layout="wide",
)


@st.cache_data
def load_universe() -> dict:
    return json.loads(UNIVERSE_PATH.read_text())


@st.cache_data
def load_final_comparison() -> pd.DataFrame:
    return pd.read_csv(FINAL_COMPARISON_PATH, index_col="ticker")


@st.cache_data
def load_multi_horizon() -> pd.DataFrame:
    return pd.read_csv(MULTI_HORIZON_PATH)


@st.cache_data
def load_latest_forecasts() -> pd.DataFrame:
    return pd.read_csv(LATEST_FORECASTS_PATH,
                       parse_dates=["as_of_date", "target_date"])


@st.cache_data
def load_horizon_predictions(horizon: str, ticker: str) -> pd.DataFrame | None:
    path = HORIZON_PRED_DIR / horizon / f"{ticker.replace('.', '_')}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=["date"])


@st.cache_data
def sector_verdict(final_df: pd.DataFrame) -> pd.DataFrame:
    """Phase 4/4b honest per-sector verdict, recomputed from the results CSV.

    Cheap pandas aggregation over an already-precomputed file — not a
    retrain — so it can't drift out of sync with the underlying numbers.
    """
    rows = []
    for sector, sub in final_df.groupby("sector"):
        lstm_dir = sub["lstm_ret_DirAcc"].mean()
        best_baseline_dir = sub[["linreg_DirAcc", "arima_DirAcc"]].mean().max()
        gap = lstm_dir - best_baseline_dir
        if gap > 3:
            verdict = "LSTM meaningfully beat baselines"
        elif gap < -3:
            verdict = "LSTM underperformed baselines"
        else:
            verdict = "LSTM tied baselines"
        rows.append({
            "Sector": sector,
            "LSTM (returns) dir. acc. (%)": round(lstm_dir, 1),
            "Best baseline dir. acc. (%)": round(best_baseline_dir, 1),
            "Gap (pp)": round(gap, 1),
            "LSTM (returns) MAPE (%)": round(sub["lstm_ret_MAPE"].mean(), 2),
            "Naive MAPE (%)": round(sub["naive_MAPE"].mean(), 2),
            "Verdict": verdict,
        })
    return pd.DataFrame(rows).set_index("Sector")


def render_disclaimer() -> None:
    st.warning(
        "**This is a portfolio project demonstrating ML techniques on Indian "
        "equity data (NSE). Not investment advice.** Past model performance "
        "does not indicate future accuracy, and stock markets are inherently "
        "difficult to predict.",
        icon="⚠️",
    )


def main() -> None:
    st.title("Sector-Comparative Indian Equity Forecasting")
    st.caption(
        "LSTM vs. naive / linear regression / ARIMA baselines for "
        "next-day, next-month, and next-year price prediction across "
        "40 NSE stocks in 4 sectors."
    )
    render_disclaimer()

    universe = load_universe()
    final_df = load_final_comparison()
    mh = load_multi_horizon()
    meta = universe["metadata"]

    st.sidebar.header("Select")
    sectors = list(universe["sectors"].keys())
    sector = st.sidebar.selectbox("Sector", sectors)
    sector_tickers = [row["ticker"] for row in universe["sectors"][sector]]
    ticker = st.sidebar.selectbox("Stock", sector_tickers)
    horizon_label = st.sidebar.radio("Forecast horizon", list(HORIZON_LABELS))
    horizon = HORIZON_LABELS[horizon_label]

    st.sidebar.caption(
        f"Universe: top {meta['top_n_per_sector']} per sector by market cap, "
        f"as of {meta['fetch_date']} (source: {meta['market_cap_source']})."
    )

    # ---- Forecast hero: the prediction itself, front and centre ----
    fc = load_latest_forecasts()
    fc_tk = fc[fc["ticker"] == ticker].sort_values("h_days")
    if not fc_tk.empty:
        as_of = fc_tk["as_of_date"].iloc[0]
        last_price = fc_tk["last_price"].iloc[0]
        st.markdown(f"### {ticker} — LSTM forecast")
        cols = st.columns(4)
        cols[0].metric("Last close", f"₹{last_price:,.2f}",
                       help=f"As of {as_of.date()}")
        hz_title = {"next_day": "Next day", "next_month": "Next month",
                    "next_year": "Next year"}
        by_hz = {r["horizon"]: r for _, r in fc_tk.iterrows()}
        for col, hz in zip(cols[1:], ["next_day", "next_month", "next_year"]):
            if hz in by_hz:
                r = by_hz[hz]
                col.metric(hz_title[hz], f"₹{r['pred_price']:,.2f}",
                           delta=f"{r['pred_return_pct']:+.2f}%",
                           help=f"Target date ≈ {r['target_date'].date()}")
            else:
                col.metric(hz_title[hz], "—",
                           help="Insufficient history to train this horizon")
        st.caption(
            f"Forecasts generated from data through {as_of.date()}. They are "
            "unverifiable until their target dates arrive, and the backtest "
            "below shows this model has **no directional edge over simple "
            "baselines** — treat these as illustrations of the method, not "
            "as advice."
        )
        st.divider()

    if horizon == "next_year":
        st.info(
            "**Next-year caveat:** with 5 years of daily data there are only "
            "~2 non-overlapping one-year windows per stock to learn from. "
            "These forecasts had no early-stopping validation split (it was "
            "consumed by leakage purging) and are illustrative, not "
            "validated. Directional accuracy must beat the *'always up'* "
            "share of the sample — not just 50% — to mean anything.",
            icon="📉",
        )

    preds = load_horizon_predictions(horizon, ticker)
    row_q = mh[(mh["ticker"] == ticker) & (mh["horizon"] == horizon)]

    if preds is None or row_q.empty:
        st.error(
            f"No {horizon_label.lower()} predictions for {ticker} — this "
            "stock's history is too short to train at this horizon after "
            "leakage purging (e.g. JIOFIN listed Aug-2023). See notebook 06."
        )
        return
    row = row_q.iloc[0]

    h_days = HORIZON_DAYS[horizon]
    st.subheader(
        f"{ticker} — Actual vs. LSTM-predicted price "
        f"{h_days} trading day{'s' if h_days > 1 else ''} ahead"
    )
    st.caption(
        "Each point compares the price the LSTM predicted for a date with "
        "the price actually realised on that date. The LSTM predicts the "
        f"{horizon_label.lower()} *return* and the price is reconstructed as "
        "today's price × (1 + predicted return)."
    )
    col_a, col_b = st.columns(2)
    show_baselines = col_a.checkbox(
        "Also show baselines (naive 'price unchanged' / linear regression)",
        value=False,
    )
    show_forward = col_b.checkbox(
        "Show forward forecasts from the latest data (next day / month / year)",
        value=True,
    )

    fig = go.Figure()
    # Invisible baseline slightly below the data so the area fill hugs the
    # chart bottom (Chart.js-style `fill: true`) instead of stretching to 0.
    baseline = float(min(preds["actual"].min(), preds["lstm"].min())) * 0.985
    fig.add_trace(go.Scatter(
        x=preds["date"], y=[baseline] * len(preds), mode="lines",
        line=dict(width=0), hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=preds["date"], y=preds["actual"], name="Actual",
        line=dict(color=MODEL_COLORS["actual"], width=2.2, shape="spline",
                  smoothing=0.4),
        fill="tonexty", fillcolor="rgba(0, 0, 0, 0.05)",
    ))
    fig.add_trace(go.Scatter(
        x=preds["date"], y=preds["lstm"], name="LSTM (returns-target)",
        line=dict(color=MODEL_COLORS["lstm"], width=1.6, dash="dash"),
        opacity=0.9,
    ))
    if show_baselines:
        fig.add_trace(go.Scatter(
            x=preds["date"], y=preds["naive"], name="Naive (price unchanged)",
            line=dict(color=MODEL_COLORS["naive"], width=1.2, dash="dot"),
            opacity=0.75,
        ))
        fig.add_trace(go.Scatter(
            x=preds["date"], y=preds["linreg"], name="Linear Regression",
            line=dict(color=MODEL_COLORS["linreg"], width=1.2, dash="dot"),
            opacity=0.75,
        ))
    if show_forward:
        fc = load_latest_forecasts()
        fc_tk = fc[fc["ticker"] == ticker].sort_values("h_days")
        if not fc_tk.empty:
            as_of = fc_tk["as_of_date"].iloc[0]
            last_price = fc_tk["last_price"].iloc[0]
            hz_short = {"next_day": "+1 day", "next_month": "+1 month",
                        "next_year": "+1 year"}
            # Dotted path from the last actual price through the three
            # forward predictions, plus labelled diamond markers.
            path_x = [as_of] + list(fc_tk["target_date"])
            path_y = [last_price] + list(fc_tk["pred_price"])
            fig.add_trace(go.Scatter(
                x=path_x, y=path_y, mode="lines+markers",
                name="Forward forecast (LSTM)",
                line=dict(color="#8e44ad", width=2, dash="dot",
                          shape="spline", smoothing=0.6),
                marker=dict(size=9, color="#8e44ad",
                            line=dict(color="#ffffff", width=2)),
                hoverinfo="skip",
            ))
            # Per-horizon label placement: +1d and +1m sit almost on the same
            # pixel at full zoom, so stagger them vertically; +1y hugs the
            # right edge, so its label goes to the left of the point.
            label_pos = {"next_day": "top left", "next_month": "bottom right",
                         "next_year": "middle left"}
            fig.add_trace(go.Scatter(
                x=fc_tk["target_date"], y=fc_tk["pred_price"],
                mode="text", showlegend=False,
                text=[f"{hz_short[h]}: ₹{p:,.0f} ({r:+.1f}%)"
                      for h, p, r in zip(fc_tk["horizon"], fc_tk["pred_price"],
                                          fc_tk["pred_return_pct"])],
                textposition=[label_pos[h] for h in fc_tk["horizon"]],
                textfont=dict(size=11, color="#6c3483"),
                hovertemplate="%{text}<extra>Forward forecast</extra>",
            ))
            missing = set(HORIZON_DAYS) - set(fc_tk["horizon"])
            note = (f" No {'/'.join(sorted(missing)).replace('_', ' ')} "
                    f"forecast for this stock (insufficient history)."
                    if missing else "")
            st.caption(
                f"◆ Forward forecasts made from data as of "
                f"{as_of.date()} (last close ₹{last_price:,.2f}). These are "
                f"unverifiable until their target dates arrive — illustrative "
                f"only, not investment advice.{note}"
            )

    fig.update_layout(
        yaxis=dict(title=None, tickprefix="₹", tickformat=",.0f",
                   gridcolor="#e9ecef", zeroline=False),
        xaxis=dict(title=None, showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0,
                    font=dict(size=13)),
        font=dict(family="Inter, -apple-system, 'Segoe UI', sans-serif",
                  color="#333333"),
        plot_bgcolor="#ffffff", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=30, b=10), height=460,
        hovermode="x unified",
    )
    st.plotly_chart(fig, width="stretch")

    st.subheader(f"LSTM performance — {horizon_label.lower()}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RMSE (₹)", f"{row['lstm_RMSE']:.2f}")
    c2.metric("MAPE", f"{row['lstm_MAPE']:.2f}%")
    c3.metric("Directional accuracy", f"{row['lstm_DirAcc']:.1f}%")
    c4.metric('"Always up" trivial bar', f"{row['always_up_acc']:.1f}%",
              help="Share of test windows where the price actually rose over "
                   "this horizon. A model's directional accuracy is only "
                   "meaningful if it beats this, not just 50%.")

    with st.expander(f"Model comparison — {horizon_label.lower()}, this stock",
                     expanded=False):
        comp_rows = [
            {"Model": "Naive (price unchanged)",
             "RMSE (₹)": round(row["naive_RMSE"], 2),
             "MAPE (%)": round(row["naive_MAPE"], 2),
             "Dir. acc. (%)": round(row["naive_DirAcc"], 1)},
            {"Model": "Linear Regression",
             "RMSE (₹)": round(row["linreg_RMSE"], 2),
             "MAPE (%)": round(row["linreg_MAPE"], 2),
             "Dir. acc. (%)": round(row["linreg_DirAcc"], 1)},
            {"Model": "LSTM (returns-target)",
             "RMSE (₹)": round(row["lstm_RMSE"], 2),
             "MAPE (%)": round(row["lstm_MAPE"], 2),
             "Dir. acc. (%)": round(row["lstm_DirAcc"], 1)},
        ]
        if horizon == "next_day" and ticker in final_df.index:
            f = final_df.loc[ticker]
            comp_rows.insert(2, {
                "Model": "ARIMA (Phase 3, next-day only)",
                "RMSE (₹)": round(f["arima_RMSE"], 2),
                "MAPE (%)": round(f["arima_MAPE"], 2),
                "Dir. acc. (%)": round(f["arima_DirAcc"], 1)})
            comp_rows.append({
                "Model": "LSTM price-target (Phase 4 — superseded)",
                "RMSE (₹)": round(f["lstm_RMSE"], 2),
                "MAPE (%)": round(f["lstm_MAPE"], 2),
                "Dir. acc. (%)": round(f["lstm_DirAcc"], 1)})
        st.dataframe(pd.DataFrame(comp_rows).set_index("Model"), width="stretch")
        st.caption(
            "Naive's near-zero directional accuracy is structural: it always "
            "predicts zero change, so it never commits to a direction. Low "
            "RMSE alone is not predictive skill."
        )

    st.divider()
    st.subheader("Horizon summary — all 40 stocks")
    st.markdown(
        "Error grows with horizon for **every** model — predicting further "
        "out is fundamentally harder. The LSTM's directional accuracy should "
        "be read against the *'always up'* bar at longer horizons, since a "
        "mostly-rising market makes 'up' a trivially good guess."
    )
    hsum = (mh.groupby("horizon")
              .agg(**{"Stocks": ("ticker", "count"),
                      "Naive MAPE (%)": ("naive_MAPE", "mean"),
                      "LinReg MAPE (%)": ("linreg_MAPE", "mean"),
                      "LSTM MAPE (%)": ("lstm_MAPE", "mean"),
                      "LSTM dir. acc. (%)": ("lstm_DirAcc", "mean"),
                      "'Always up' bar (%)": ("always_up_acc", "mean")})
              .round(2)
              .reindex(["next_day", "next_month", "next_year"])
              .rename(index={"next_day": "Next day",
                             "next_month": "Next month",
                             "next_year": "Next year"}))
    st.dataframe(hsum, width="stretch")

    st.subheader("Sector summary — the honest next-day finding")
    st.markdown(
        "The returns-target LSTM (Phase 4b) fixed the price-level LSTM's "
        "error problem (MAPE 4.3% → 1.3%, improving all 40 stocks) but its "
        "**directional accuracy remains a coin flip in every sector** — "
        "matching, not beating, the simple baselines."
    )
    st.dataframe(sector_verdict(final_df), width="stretch")

    st.caption(
        "All metrics are precomputed by notebooks 03-06 and read from "
        "results/ CSVs — this app does not retrain models or fetch live "
        "prices."
    )


if __name__ == "__main__":
    main()

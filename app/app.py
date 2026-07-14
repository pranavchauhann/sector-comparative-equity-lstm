"""
app/app.py
==========
Streamlit dashboard for the Sector-Comparative Indian Equity Forecasting
project. Reads only precomputed artifacts (config/universe.json,
results/final_comparison.csv, results/predictions/<TICKER>.csv) — it never
retrains a model or fetches live data from Yahoo Finance. This keeps page
loads fast and avoids rate-limiting real users, per the project's
architecture rule.

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
FINAL_COMPARISON_PATH = ROOT / "results" / "final_comparison.csv"
PREDICTIONS_DIR = ROOT / "results" / "predictions"

MODEL_LABELS = {
    "naive": "Naive (tomorrow = today)",
    "linreg": "Linear Regression",
    "arima": "ARIMA",
    "lstm": "LSTM",
}
MODEL_COLORS = {
    "actual": "#111111",
    "naive": "#9c9c9c",
    "linreg": "#e07b39",
    "arima": "#3f8f5f",
    "lstm": "#c0392b",
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
def load_predictions(ticker: str) -> pd.DataFrame:
    path = PREDICTIONS_DIR / f"{ticker.replace('.', '_')}.csv"
    return pd.read_csv(path, parse_dates=["date"])


@st.cache_data
def sector_verdict(final_df: pd.DataFrame) -> pd.DataFrame:
    """Recompute the Phase 4 honest per-sector verdict from final_comparison.csv.

    This is cheap pandas aggregation over an already-precomputed CSV — not a
    live retrain or refetch — so it stays in sync with the results file
    without violating the "no retraining at load time" rule.
    """
    rows = []
    for sector, sub in final_df.groupby("sector"):
        lstm_dir = sub["lstm_DirAcc"].mean()
        best_baseline_dir = sub[["linreg_DirAcc", "arima_DirAcc"]].mean().max()
        lstm_rmse = sub["lstm_RMSE"].mean()
        best_baseline_rmse = sub[["naive_RMSE", "linreg_RMSE", "arima_RMSE"]].mean(axis=1).mean()
        gap = lstm_dir - best_baseline_dir
        if gap > 3:
            verdict = "LSTM meaningfully beat baselines"
        elif gap < -3:
            verdict = "LSTM underperformed baselines"
        else:
            verdict = "LSTM tied baselines"
        rows.append({
            "Sector": sector,
            "LSTM dir. acc. (%)": round(lstm_dir, 1),
            "Best baseline dir. acc. (%)": round(best_baseline_dir, 1),
            "Gap (pp)": round(gap, 1),
            "LSTM mean RMSE": round(lstm_rmse, 2),
            "Best baseline mean RMSE": round(best_baseline_rmse, 2),
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
        "LSTM vs. naive / linear regression / ARIMA baselines for next-day "
        "closing-price prediction across 40 NSE stocks in 4 sectors."
    )
    render_disclaimer()

    universe = load_universe()
    final_df = load_final_comparison()
    meta = universe["metadata"]

    st.sidebar.header("Select a stock")
    sectors = list(universe["sectors"].keys())
    sector = st.sidebar.selectbox("Sector", sectors)

    sector_tickers = [row["ticker"] for row in universe["sectors"][sector]]
    ticker = st.sidebar.selectbox("Stock", sector_tickers)

    st.sidebar.caption(
        f"Universe: top {meta['top_n_per_sector']} per sector by market cap, "
        f"as of {meta['fetch_date']} (source: {meta['market_cap_source']})."
    )

    if ticker not in final_df.index:
        st.error(f"No results found for {ticker}.")
        return
    row = final_df.loc[ticker]
    preds = load_predictions(ticker)

    # --- Main chart: actual vs LSTM predicted, in INR ---
    st.subheader(f"{ticker} — Actual vs. LSTM Prediction (test period)")
    show_baselines = st.checkbox(
        "Also show baseline model predictions (naive / linear regression / ARIMA)",
        value=False,
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=preds["date"], y=preds["actual"], name="Actual",
        line=dict(color=MODEL_COLORS["actual"], width=2.2),
    ))
    fig.add_trace(go.Scatter(
        x=preds["date"], y=preds["lstm"], name="LSTM",
        line=dict(color=MODEL_COLORS["lstm"], width=1.6, dash="dash"),
    ))
    if show_baselines:
        for m in ["naive", "linreg", "arima"]:
            if m in preds.columns:
                fig.add_trace(go.Scatter(
                    x=preds["date"], y=preds[m], name=MODEL_LABELS[m],
                    line=dict(color=MODEL_COLORS[m], width=1.2, dash="dot"),
                    opacity=0.75,
                ))
    fig.update_layout(
        yaxis_title="Adj Close (₹)", xaxis_title="Date",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=30, b=10), height=460,
        hovermode="x unified",
    )
    st.plotly_chart(fig, width="stretch")

    # --- Metrics panel for the LSTM on this stock ---
    st.subheader("LSTM performance on this stock")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RMSE (₹)", f"{row['lstm_RMSE']:.2f}")
    c2.metric("MAE (₹)", f"{row['lstm_MAE']:.2f}")
    c3.metric("MAPE", f"{row['lstm_MAPE']:.2f}%")
    c4.metric("Directional accuracy", f"{row['lstm_DirAcc']:.1f}%")

    # --- Model comparison table/expander ---
    with st.expander("Model comparison — all 4 models on this stock", expanded=False):
        comp_rows = []
        for m in ["naive", "linreg", "arima", "lstm"]:
            comp_rows.append({
                "Model": MODEL_LABELS[m],
                "RMSE (₹)": round(row[f"{m}_RMSE"], 2),
                "MAE (₹)": round(row[f"{m}_MAE"], 2),
                "MAPE (%)": round(row[f"{m}_MAPE"], 2),
                "Directional accuracy (%)": round(row[f"{m}_DirAcc"], 1),
            })
        comp_df = pd.DataFrame(comp_rows).set_index("Model")
        st.dataframe(comp_df, width="stretch")
        best_rmse_model = comp_df["RMSE (₹)"].idxmin()
        st.caption(
            f"Lowest RMSE on {ticker}: **{best_rmse_model}**. Note: low RMSE "
            "does not by itself imply predictive skill — see the Sector "
            "Summary tab for the directional-accuracy comparison, which is "
            "the fairer test (Phase 3/4 finding)."
        )

    # --- Sector summary tab ---
    st.divider()
    st.subheader("Sector Summary — the honest Phase 4 finding")
    st.markdown(
        "Naive wins on raw RMSE most often across the universe because "
        "day-to-day price moves are small relative to price level — but its "
        "directional accuracy is close to 0% by construction (its predicted "
        "change is always exactly zero). Linear regression and ARIMA land "
        "near a 50% coin-flip on direction. **The LSTM's job was to beat "
        "that ~50% floor — the table below shows whether it did, per "
        "sector.**"
    )
    verdict_df = sector_verdict(final_df)
    st.dataframe(verdict_df, width="stretch")

    st.caption(
        "Metrics are precomputed from the Phase 3/4 notebooks and read from "
        "`results/final_comparison.csv` — this app does not retrain models "
        "or fetch live prices."
    )


if __name__ == "__main__":
    main()

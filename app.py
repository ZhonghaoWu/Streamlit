import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


def compute_rsi(prices: pd.Series, period: int) -> pd.Series:
    delta = prices.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.rolling(window=period, min_periods=period).mean()
    avg_loss = losses.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_metrics(returns: pd.Series, periods_per_year: int = 252) -> dict[str, float]:
    returns = returns.dropna()
    if returns.empty:
        return {"Total Return": np.nan, "CAGR": np.nan, "Sharpe": np.nan}

    cumulative = (1 + returns).prod() - 1
    years = len(returns) / periods_per_year
    cagr = (1 + cumulative) ** (1 / years) - 1 if years > 0 else np.nan
    sharpe = (
        returns.mean() / returns.std() * np.sqrt(periods_per_year)
        if returns.std() != 0
        else np.nan
    )
    return {
        "Total Return": cumulative,
        "CAGR": cagr,
        "Sharpe": sharpe,
    }


st.set_page_config(page_title="Strategy Comparison", layout="wide")

st.title("Lightweight Strategy Comparison")
st.write(
    "Compare a moving average crossover with an RSI strategy using daily prices from Yahoo Finance."
)

with st.sidebar:
    st.header("Inputs")
    tickers = st.multiselect(
        "Select tickers",
        options=[
            "AAPL",
            "MSFT",
            "SPY",
            "TSLA",
            "NVDA",
            "AMZN",
            "GOOGL",
            "META",
        ],
        default=["AAPL", "MSFT"],
    )
    end_date = st.date_input("End date", value=dt.date.today())
    start_date = st.date_input("Start date", value=end_date - dt.timedelta(days=365 * 2))

    st.subheader("Moving Average Crossover")
    short_window = st.number_input("Short MA window", min_value=5, max_value=60, value=20)
    long_window = st.number_input("Long MA window", min_value=30, max_value=200, value=100)

    st.subheader("RSI Strategy")
    rsi_period = st.number_input("RSI period", min_value=5, max_value=30, value=14)
    rsi_low = st.slider("RSI oversold", min_value=10, max_value=40, value=30)
    rsi_high = st.slider("RSI overbought", min_value=60, max_value=90, value=70)

if not tickers:
    st.info("Pick at least one ticker to begin.")
    st.stop()

if start_date >= end_date:
    st.error("Start date must be earlier than end date.")
    st.stop()

@st.cache_data(show_spinner=False)
def load_prices(selected_tickers: tuple[str, ...], start: dt.date, end: dt.date) -> pd.DataFrame:
    data = yf.download(
        tickers=list(selected_tickers),
        start=start,
        end=end + dt.timedelta(days=1),
        progress=False,
        auto_adjust=True,
    )
    if data.empty:
        return pd.DataFrame()
    prices = data["Close"].copy()
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(selected_tickers[0])
    return prices.dropna(how="all")


prices = load_prices(tuple(tickers), start_date, end_date)
if prices.empty:
    st.warning("No price data returned. Try a different date range or ticker.")
    st.stop()

returns = prices.pct_change().fillna(0)

ma_results = {}
ma_signals = pd.DataFrame(index=prices.index)
for ticker in tickers:
    short_ma = prices[ticker].rolling(window=short_window).mean()
    long_ma = prices[ticker].rolling(window=long_window).mean()
    signal = (short_ma > long_ma).astype(int)
    ma_signals[ticker] = signal
    ma_returns = returns[ticker] * signal.shift(1).fillna(0)
    ma_results[ticker] = ma_returns

rsi_results = {}
rsi_signals = pd.DataFrame(index=prices.index)
for ticker in tickers:
    rsi = compute_rsi(prices[ticker], rsi_period)
    signal = pd.Series(index=prices.index, dtype=float)
    signal[rsi < rsi_low] = 1
    signal[rsi > rsi_high] = 0
    signal = signal.ffill().fillna(0)
    rsi_signals[ticker] = signal
    rsi_returns = returns[ticker] * signal.shift(1).fillna(0)
    rsi_results[ticker] = rsi_returns

buy_hold_results = {ticker: returns[ticker] for ticker in tickers}

st.subheader("Cumulative Returns")
for ticker in tickers:
    cumulative = pd.DataFrame(
        {
            "Buy & Hold": (1 + buy_hold_results[ticker]).cumprod(),
            "MA Crossover": (1 + ma_results[ticker]).cumprod(),
            "RSI Strategy": (1 + rsi_results[ticker]).cumprod(),
        }
    )
    st.markdown(f"**{ticker}**")
    st.line_chart(cumulative)

st.subheader("Performance Summary")
summary_rows = []
for ticker in tickers:
    summary_rows.append(
        {
            "Ticker": ticker,
            "Strategy": "Buy & Hold",
            **compute_metrics(buy_hold_results[ticker]),
        }
    )
    summary_rows.append(
        {
            "Ticker": ticker,
            "Strategy": "MA Crossover",
            **compute_metrics(ma_results[ticker]),
        }
    )
    summary_rows.append(
        {
            "Ticker": ticker,
            "Strategy": "RSI",
            **compute_metrics(rsi_results[ticker]),
        }
    )

summary_df = pd.DataFrame(summary_rows)
summary_df["Total Return"] = summary_df["Total Return"].map(lambda x: f"{x:.2%}" if pd.notna(x) else "-")
summary_df["CAGR"] = summary_df["CAGR"].map(lambda x: f"{x:.2%}" if pd.notna(x) else "-")
summary_df["Sharpe"] = summary_df["Sharpe"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "-")

st.dataframe(summary_df, use_container_width=True, hide_index=True)

st.caption(
    "Signals are evaluated at the close and applied on the next day. Results are gross of fees and slippage."
)

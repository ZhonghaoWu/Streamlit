# Develop a lightweight Streamlit app that compares the performance of two simple strategies (e.g., moving average crossover vs. RSI-based). Let the user select tickers and strategy parameters via sidebar controls.
# Streamlit Strategy Comparison App

This lightweight Streamlit app compares two simple strategies—moving-average crossover and RSI—across user-selected tickers.

## Features
- Select tickers and date range from the sidebar.
- Tune moving-average and RSI parameters.
- View cumulative return charts and a performance summary table.

## Run
```bash
streamlit run app.py
```

> Data is sourced from Yahoo Finance via `yfinance`.

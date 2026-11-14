import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

st.set_page_config(
    page_title="AI Stock Analysis Assistant",
    page_icon="📈",
    layout="wide"
)

st.title("📈 AI Stock Analysis Assistant")
st.caption(
    "Analyze US and Indian stocks: price trend, returns, volatility, and moving averages."
)

# ---------------------------------------------------------------
# Market settings
# Yahoo Finance uses ".NS" for NSE and ".BO" for BSE stocks.
# ---------------------------------------------------------------
MARKETS = {
    "US stocks": {
        "suffix": "",
        "default": "AAPL",
        "examples": "AAPL, TSLA, MSFT, NVDA",
    },
    "India - NSE": {
        "suffix": ".NS",
        "default": "RELIANCE",
        "examples": "RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK, SBIN, ITC",
    },
    "India - BSE": {
        "suffix": ".BO",
        "default": "RELIANCE",
        "examples": "RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK, SBIN, ITC",
    },
}

# Sidebar controls
st.sidebar.header("Settings")

market = st.sidebar.selectbox("Market", list(MARKETS.keys()))
cfg = MARKETS[market]

raw_ticker = st.sidebar.text_input(
    "Stock symbol",
    value=cfg["default"],
    key=f"ticker_{market}",
    help=f"Examples: {cfg['examples']}"
)

if market != "US stocks":
    st.sidebar.caption(
        f"Type just the symbol (like TCS). The {cfg['suffix']} ending is added automatically."
    )

period = st.sidebar.selectbox(
    "History period",
    ["1y", "2y", "5y"],
    index=0
)

analyze_button = st.sidebar.button("Analyze stock", type="primary")


# ---------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------
def resolve_ticker(raw, suffix):
    ticker = raw.strip().upper()
    if not ticker:
        raise ValueError("Please enter a stock symbol.")
    if suffix and not (ticker.endswith(".NS") or ticker.endswith(".BO")):
        ticker += suffix
    return ticker


def currency_symbol(ticker):
    if ticker.endswith(".NS") or ticker.endswith(".BO"):
        return "₹"
    return "$"


def money(value, cur):
    if value is None:
        return "N/A"
    return f"{cur}{value:,.2f}"


def pct(value):
    if value is None:
        return "N/A"
    return f"{value:.2f}%"


@st.cache_data(ttl=3600, show_spinner=False)
def get_price_data(ticker, period):
    data = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        raise ValueError(
            f"No price data found for '{ticker}'. Check the symbol and the market selected."
        )

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume"
    })

    return data[["open", "high", "low", "close", "volume"]].dropna()


def moving_average(close, window):
    if len(close) < window:
        return None
    return float(close.rolling(window).mean().iloc[-1])


def calculate_metrics(data):
    close = data["close"]
    returns = close.pct_change().dropna()

    latest_close = float(close.iloc[-1])
    first_close = float(close.iloc[0])

    total_return = (latest_close / first_close - 1) * 100
    annual_volatility = float(returns.std() * np.sqrt(252) * 100)

    running_max = close.cummax()
    max_drawdown = float(((close / running_max) - 1).min() * 100)

    ma_20 = moving_average(close, 20)
    ma_50 = moving_average(close, 50)
    ma_200 = moving_average(close, 200)

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_value = None if pd.isna(rsi.iloc[-1]) else float(rsi.iloc[-1])

    if ma_50 is None or ma_200 is None:
        trend = "Not enough data"
    elif latest_close > ma_50 and ma_50 > ma_200:
        trend = "Bullish"
    elif latest_close < ma_50 and ma_50 < ma_200:
        trend = "Bearish"
    else:
        trend = "Mixed"

    return {
        "latest_close": latest_close,
        "total_return": total_return,
        "annual_volatility": annual_volatility,
        "max_drawdown": max_drawdown,
        "ma_20": ma_20,
        "ma_50": ma_50,
        "ma_200": ma_200,
        "rsi": rsi_value,
        "trend": trend
    }


def create_ai_report(ticker, metrics, cur):
    rsi_text = "N/A" if metrics["rsi"] is None else f"{metrics['rsi']:.2f}"

    if metrics["trend"] == "Not enough data":
        trend_text = (
            "There is not enough price history to calculate the 50-day and 200-day "
            "moving averages, so the trend cannot be classified. Try a longer history period."
        )
    else:
        trend_text = (
            f"The stock is currently showing a **{metrics['trend'].lower()}** trend "
            "based on its price relative to the 50-day and 200-day moving averages."
        )

    return f"""
### {ticker} Analysis

**Latest close:** {money(metrics['latest_close'], cur)}  
**Period return:** {pct(metrics['total_return'])}  
**Annualized volatility:** {pct(metrics['annual_volatility'])}  
**Maximum drawdown:** {pct(metrics['max_drawdown'])}  
**Trend:** {metrics['trend']}  

### Moving averages

| Indicator | Value |
|---|---:|
| 20-day moving average | {money(metrics['ma_20'], cur)} |
| 50-day moving average | {money(metrics['ma_50'], cur)} |
| 200-day moving average | {money(metrics['ma_200'], cur)} |
| 14-day RSI | {rsi_text} |

### Interpretation

{trend_text}

A price above both moving averages generally suggests stronger medium- and long-term momentum. A price below both can suggest weakening momentum.

An RSI above 70 may indicate short-term overbought conditions, while an RSI below 30 may indicate oversold conditions. These are technical indicators, not buy or sell signals.

**Disclaimer:** This is educational analysis, not financial advice.
"""


# ---------------------------------------------------------------
# Main app logic
# Results are kept in session_state so they stay on screen after
# clicking the download button.
# ---------------------------------------------------------------
if analyze_button:
    try:
        symbol = resolve_ticker(raw_ticker, cfg["suffix"])
        with st.spinner("Fetching and analyzing stock data..."):
            data = get_price_data(symbol, period)
            metrics = calculate_metrics(data)

        st.session_state["result"] = {
            "ticker": symbol,
            "data": data,
            "metrics": metrics,
            "cur": currency_symbol(symbol),
        }
    except Exception as error:
        st.session_state.pop("result", None)
        st.error(f"Could not analyze '{raw_ticker.strip().upper()}': {error}")

result = st.session_state.get("result")

if result:
    ticker = result["ticker"]
    data = result["data"]
    metrics = result["metrics"]
    cur = result["cur"]

    st.success(f"Analysis complete for {ticker}")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Latest close", money(metrics["latest_close"], cur))
    col2.metric("Period return", pct(metrics["total_return"]))
    col3.metric("Annual volatility", pct(metrics["annual_volatility"]))
    col4.metric("Trend", metrics["trend"])

    st.subheader("Price chart")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(data.index, data["close"], label="Close price", linewidth=2)
    ax.plot(data.index, data["close"].rolling(20).mean(), label="20-day MA")
    ax.plot(data.index, data["close"].rolling(50).mean(), label="50-day MA")
    ax.plot(data.index, data["close"].rolling(200).mean(), label="200-day MA")

    ax.set_title(f"{ticker} Price and Moving Averages")
    ax.set_ylabel(f"Price ({cur})")
    ax.legend()
    ax.grid(alpha=0.3)

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("AI analysis")
    st.markdown(create_ai_report(ticker, metrics, cur))

    st.subheader("Download data")
    csv_data = data.to_csv().encode("utf-8")

    st.download_button(
        label="Download price history as CSV",
        data=csv_data,
        file_name=f"{ticker}_price_history.csv",
        mime="text/csv"
    )

elif not analyze_button:
    st.info("Choose a market and symbol in the sidebar, then click **Analyze stock**.")

"""Market Dashboard - Streamlit Module"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf


def show():
    """Display the Market Dashboard page."""
    st.header("🏠 Market Dashboard")
    st.markdown("Real-time overview of major market indices and trending stocks")

    # Major indices
    st.subheader("📊 Major Indices")

    indices = {
        '^GSPC': 'S&P 500',
        '^DJI': 'Dow Jones',
        '^IXIC': 'NASDAQ',
        '^RUT': 'Russell 2000'
    }

    cols = st.columns(4)

    for idx, (symbol, name) in enumerate(indices.items()):
        with cols[idx]:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='2d')

                if len(hist) >= 2:
                    current = hist['Close'].iloc[-1]
                    previous = hist['Close'].iloc[-2]
                    change = current - previous
                    change_pct = (change / previous) * 100

                    st.metric(
                        label=name,
                        value=f"${current:,.2f}",
                        delta=f"{change_pct:+.2f}%"
                    )
                else:
                    st.metric(label=name, value="N/A")
            except Exception as e:
                st.metric(label=name, value="Error")

    st.markdown("---")

    # Stock lookup section
    st.subheader("🔍 Stock Lookup")

    col1, col2 = st.columns([2, 1])

    with col1:
        symbol = st.text_input(
            "Enter stock symbol",
            value="AAPL",
            help="e.g., AAPL, MSFT, GOOGL"
        ).upper()

    with col2:
        period = st.selectbox(
            "Time Period",
            options=["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"],
            index=4  # Default to 6mo
        )

    if symbol:
        try:
            with st.spinner(f"Fetching data for {symbol}..."):
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period)
                info = ticker.info

                if len(hist) > 0:
                    # Stock info cards
                    st.markdown(f"### {info.get('longName', symbol)}")

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        current_price = hist['Close'].iloc[-1]
                        st.metric("Current Price", f"${current_price:.2f}")

                    with col2:
                        day_high = hist['High'].iloc[-1]
                        st.metric("Day High", f"${day_high:.2f}")

                    with col3:
                        day_low = hist['Low'].iloc[-1]
                        st.metric("Day Low", f"${day_low:.2f}")

                    with col4:
                        volume = hist['Volume'].iloc[-1]
                        st.metric("Volume", f"{volume:,.0f}")

                    # Price chart
                    st.subheader("📈 Price Chart")

                    fig = go.Figure()

                    # Candlestick chart
                    fig.add_trace(go.Candlestick(
                        x=hist.index,
                        open=hist['Open'],
                        high=hist['High'],
                        low=hist['Low'],
                        close=hist['Close'],
                        name=symbol
                    ))

                    fig.update_layout(
                        title=f"{symbol} Price Chart",
                        yaxis_title="Price ($)",
                        xaxis_title="Date",
                        template="plotly_white",
                        height=500,
                        xaxis_rangeslider_visible=False
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Volume chart
                    st.subheader("📊 Volume")

                    fig_vol = go.Figure()

                    colors = ['red' if hist['Close'].iloc[i] < hist['Open'].iloc[i]
                             else 'green' for i in range(len(hist))]

                    fig_vol.add_trace(go.Bar(
                        x=hist.index,
                        y=hist['Volume'],
                        marker_color=colors,
                        name='Volume'
                    ))

                    fig_vol.update_layout(
                        title=f"{symbol} Trading Volume",
                        yaxis_title="Volume",
                        xaxis_title="Date",
                        template="plotly_white",
                        height=300
                    )

                    st.plotly_chart(fig_vol, use_container_width=True)

                    # Additional info
                    with st.expander("📋 Additional Information"):
                        col1, col2 = st.columns(2)

                        with col1:
                            if 'marketCap' in info:
                                st.write(f"**Market Cap:** ${info['marketCap']:,.0f}")
                            if 'sector' in info:
                                st.write(f"**Sector:** {info['sector']}")
                            if 'industry' in info:
                                st.write(f"**Industry:** {info['industry']}")

                        with col2:
                            if 'fiftyTwoWeekHigh' in info:
                                st.write(f"**52-Week High:** ${info['fiftyTwoWeekHigh']:.2f}")
                            if 'fiftyTwoWeekLow' in info:
                                st.write(f"**52-Week Low:** ${info['fiftyTwoWeekLow']:.2f}")
                            if 'averageVolume' in info:
                                st.write(f"**Avg Volume:** {info['averageVolume']:,.0f}")
                else:
                    st.warning(f"No data available for {symbol}")

        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {str(e)}")

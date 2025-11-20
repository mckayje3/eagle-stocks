"""News & Sentiment Analysis - Streamlit Module"""

import streamlit as st
import yfinance as yf
from textblob import TextBlob
from datetime import datetime
import pandas as pd


def analyze_sentiment(text):
    """Analyze sentiment of text using TextBlob."""
    try:
        blob = TextBlob(text)
        return blob.sentiment.polarity
    except:
        return 0.0


def get_sentiment_label(score):
    """Convert sentiment score to label."""
    if score > 0.2:
        return "Positive"
    elif score < -0.2:
        return "Negative"
    else:
        return "Neutral"


def get_sentiment_color(score):
    """Get color for sentiment score."""
    if score > 0.2:
        return "green"
    elif score < -0.2:
        return "red"
    else:
        return "gray"


def show():
    """Display the News & Sentiment page."""
    st.header("📰 News & Sentiment Analysis")
    st.markdown("Latest news with AI-powered sentiment analysis")

    # Top buttons
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        if st.button("📊 Top Market Stories", use_container_width=True):
            st.session_state.news_mode = 'market'

    with col2:
        symbol = st.text_input(
            "Stock Symbol",
            value=st.session_state.get('news_symbol', ''),
            placeholder="e.g., AAPL"
        ).upper()

    with col3:
        if st.button("🔍 Fetch Stock News", use_container_width=True, disabled=not symbol):
            st.session_state.news_mode = 'stock'
            st.session_state.news_symbol = symbol

    st.markdown("---")

    # Initialize news mode
    if 'news_mode' not in st.session_state:
        st.session_state.news_mode = 'market'

    # Fetch and display news
    try:
        if st.session_state.news_mode == 'market':
            st.subheader("📈 Top Market Stories")
            ticker_symbol = "^GSPC"
            source = "S&P 500 Market News"
        else:
            st.subheader(f"📊 {st.session_state.news_symbol} News")
            ticker_symbol = st.session_state.news_symbol
            source = f"{st.session_state.news_symbol} News"

        with st.spinner(f"Fetching {source.lower()}..."):
            ticker = yf.Ticker(ticker_symbol)
            news = ticker.news

            if news:
                # Process news and calculate sentiment
                news_data = []
                sentiments = []

                for article in news[:20]:
                    # Skip if article is None or empty
                    if not article:
                        continue

                    # Handle nested content structure
                    if 'content' in article:
                        article_data = article['content']
                    else:
                        article_data = article

                    # Skip if article_data is None
                    if not article_data:
                        continue

                    title = article_data.get('title', 'No title') if isinstance(article_data, dict) else 'No title'

                    # Extract publisher
                    provider = article_data.get('provider', {}) if isinstance(article_data, dict) else {}
                    publisher = provider.get('displayName', 'Unknown') if isinstance(provider, dict) else 'Unknown'

                    # Extract link
                    click_through = article_data.get('clickThroughUrl', {}) if isinstance(article_data, dict) else {}
                    link = click_through.get('url', '') if isinstance(click_through, dict) else ''

                    # Parse date
                    pub_date = article_data.get('pubDate') if isinstance(article_data, dict) else None
                    if isinstance(pub_date, str):
                        try:
                            dt = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                            date = dt.strftime('%Y-%m-%d %H:%M')
                        except:
                            date = 'Unknown date'
                    else:
                        date = 'Unknown date'

                    # Sentiment analysis
                    sentiment = analyze_sentiment(title)
                    sentiments.append(sentiment)

                    news_data.append({
                        'title': title,
                        'publisher': publisher,
                        'date': date,
                        'link': link,
                        'sentiment': sentiment
                    })

                # Display overall sentiment
                if sentiments:
                    avg_sentiment = sum(sentiments) / len(sentiments)
                    overall_label = get_sentiment_label(avg_sentiment)

                    if st.session_state.news_mode == 'market':
                        label_text = "Overall Market Sentiment"
                    else:
                        label_text = f"{st.session_state.news_symbol} Sentiment"

                    # Sentiment indicator
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        if avg_sentiment > 0.2:
                            st.success(f"**{label_text}:** {overall_label} ({avg_sentiment:.3f})")
                        elif avg_sentiment < -0.2:
                            st.error(f"**{label_text}:** {overall_label} ({avg_sentiment:.3f})")
                        else:
                            st.info(f"**{label_text}:** {overall_label} ({avg_sentiment:.3f})")

                st.markdown("---")

                # Display news articles
                for article in news_data:
                    with st.container():
                        # Header
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"**{article['publisher']}**")
                        with col2:
                            st.markdown(f"*{article['date']}*")

                        # Title
                        st.markdown(f"### {article['title']}")

                        # Footer
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            sentiment_label = get_sentiment_label(article['sentiment'])
                            color = get_sentiment_color(article['sentiment'])

                            if color == 'green':
                                st.success(f"Sentiment: {sentiment_label} ({article['sentiment']:.3f})")
                            elif color == 'red':
                                st.error(f"Sentiment: {sentiment_label} ({article['sentiment']:.3f})")
                            else:
                                st.info(f"Sentiment: {sentiment_label} ({article['sentiment']:.3f})")

                        with col2:
                            if article['link']:
                                st.markdown(f"[Read More →]({article['link']})")

                        st.markdown("---")

            else:
                st.warning(f"No news found for {ticker_symbol}")

    except Exception as e:
        st.error(f"Error fetching news: {str(e)}")

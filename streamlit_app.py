# streamlit_app.py
"""
Free Crypto-Intelligence Tool – Streamlit front-end
Author: <you>
License: MIT
---------------------------------------------------
This single file lets non-coders launch an interactive
dashboard that:
• pulls market data from CoinGecko's free API
• performs basic sentiment + technical analysis
• ranks coins with the Opportunity Score described in
  the earlier blueprint
Everything runs on Streamlit Community Cloud's free tier.
"""

import os
import time
import requests
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

# -------------- CONFIG ------------------------------------------------------

COINS = [
    "bitcoin", "ethereum", "cardano", "solana",
    "dogecoin", "polkadot", "litecoin", "chainlink",
    "avalanche-2", "polygon"
]
VS_CURRENCY = "usd"
NEWS_FEEDS = [
    "https://feeds.feedburner.com/CoinDesk",
    "https://cointelegraph.com/rss",
    "https://bitcoinmagazine.com/.rss/full/"
]
REFRESH_EVERY_MIN = 15          # Streamlit caching TTL
OPPORTUNITY_WEIGHTING = {
    "sentiment": 0.35,
    "technical": 0.35,
    "volume": 0.15,
    "volatility": 0.15,
}

# -------------- HELPER FUNCTIONS -------------------------------------------

@st.cache_data(ttl=REFRESH_EVERY_MIN * 60)
def fetch_market_data(coins):
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": VS_CURRENCY,
        "ids": ",".join(coins),
        "price_change_percentage": "1h,24h,7d"
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    return df


def basic_technical_scores(df):
    """Very lightweight technical 'score' 0-100 based on RSI proxy."""
    tech_scores = []
    for _, row in df.iterrows():
        # crude proxy: percent change 24h vs 7d volatility
        momentum = row["price_change_percentage_24h"]
        volatility = abs(row["price_change_percentage_7d"])
        if volatility == 0:
            score = 50
        else:
            score = 50 + (momentum / volatility) * 50
        score = max(0, min(100, score))
        tech_scores.append(score)
    return tech_scores


def fetch_news_sentiment():
    """Very small demo sentiment: counts positive vs negative keywords."""
    import feedparser, re
    positive = re.compile(r"\b(bullish|surge|gain|rise|record)\b", re.I)
    negative = re.compile(r"\b(hack|fall|plunge|bearish|loss)\b", re.I)
    latest = []
    for feed in NEWS_FEEDS:
        parsed = feedparser.parse(feed)
        latest.extend(parsed.entries[:20])

    sentiments = {}
    for entry in latest:
        txt = f"{entry.title} {entry.summary}"
        pos = bool(positive.search(txt))
        neg = bool(negative.search(txt))
        tokens = re.findall(r"\b[a-z]{2,}\b", txt.lower())
        for coin in COINS:
            symbol = coin.split("-")[0]  # rough match
            if symbol in tokens:
                prev = sentiments.get(coin, 0)
                if pos and not neg:
                    sentiments[coin] = prev + 1
                elif neg and not pos:
                    sentiments[coin] = prev - 1
    # scale to 0-100
    scores = {c: (v + 5) * 10 for c, v in sentiments.items()}
    return scores


def compute_opportunity(df, tech_scores, sent_scores):
    opp = []
    for idx, row in df.iterrows():
        coin = row["id"]
        sentiment = sent_scores.get(coin, 50)
        technical = tech_scores[idx]
        volume_z = 50  # placeholder (needs historic vol)
        volat_z = 50   # placeholder (needs ATR/Bollinger)
        score = (
            OPPORTUNITY_WEIGHTING["sentiment"] * sentiment +
            OPPORTUNITY_WEIGHTING["technical"] * technical +
            OPPORTUNITY_WEIGHTING["volume"] * volume_z +
            OPPORTUNITY_WEIGHTING["volatility"] * volat_z
        )
        opp.append(round(score, 2))
    return opp


def grade(score):
    if score >= 90:
        return "A+"
    if score >= 80:
        return "A"
    if score >= 70:
        return "B"
    if score >= 60:
        return "C"
    if score >= 50:
        return "D"
    return "F"


# -------------- STREAMLIT LAYOUT -------------------------------------------

st.set_page_config(page_title="Free Crypto-Intelligence Tool",
                   page_icon="💹",
                   layout="wide")

st.title("💹 Crypto Intelligence Dashboard (100% Free)")

with st.sidebar:
    st.header("Settings")
    selected_coins = st.multiselect(
        "Choose coins to analyse",
        COINS,
        default=COINS[:5]
    )
    st.markdown("Data refreshes every **15 minutes**.")
    if st.button("Force refresh now"):
        st.cache_data.clear()

# ---------------- DATA PIPELINE --------------------------------------------

try:
    market_df = fetch_market_data(selected_coins)
except Exception as e:
    st.error(f"Failed to fetch market data: {e}")
    st.stop()

tech_scores = basic_technical_scores(market_df)
sent_scores = fetch_news_sentiment()
opp_scores = compute_opportunity(market_df, tech_scores, sent_scores)

market_df["Tech Score"] = tech_scores
market_df["Sentiment"] = market_df["id"].map(sent_scores).fillna(50)
market_df["Opportunity"] = opp_scores
market_df["Grade"] = market_df["Opportunity"].apply(grade)

# ---------------- DISPLAY ---------------------------------------------------

st.subheader("📈 Market Overview")
st.dataframe(
    market_df[[
        "symbol", "current_price", "market_cap",
        "price_change_percentage_24h",
        "Tech Score", "Sentiment", "Opportunity", "Grade"
    ]].rename(columns={
        "symbol": "Symbol",
        "current_price": "Price ($)",
        "market_cap": "Mkt Cap ($)",
        "price_change_percentage_24h": "24h Δ (%)"
    }).set_index("Symbol"),
    height=350
)

st.subheader("🏆 Top Opportunities")
leader = market_df.sort_values("Opportunity", ascending=False).head(10)
st.table(
    leader[["symbol", "Grade", "Opportunity"]]
    .rename(columns={"symbol": "Symbol"})
    .set_index("Symbol")
)

st.caption(
    "Sentiment scores use simple keyword heuristics for demonstration. "
    "For production-grade analysis, plug in the full VADER/TextBlob "
    "pipeline described in the detailed blueprint."
)

st.markdown("---")
st.markdown(
    "Built with ❤️ using only free APIs (CoinGecko, RSS) and "
    "Streamlit Community Cloud.  |  **Not financial advice.**"
)
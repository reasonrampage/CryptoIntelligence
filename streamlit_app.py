# CryptoIntelligence - Fixed Streamlit App
# This version fixes the API integration issues and adds proper error handling

import streamlit as st
import pandas as pd
import requests
import feedparser
import numpy as np
import os
from datetime import datetime, timedelta
import time
import json
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas_ta as ta

# Page configuration
st.set_page_config(
    page_title="CryptoIntelligence Dashboard",
    page_icon="💰",
    layout="wide"
)

# Initialize VADER analyzer
analyzer = SentimentIntensityAnalyzer()

def check_configuration():
    """Check if all required configurations are present"""
    config_status = {
        'coingecko_api': os.getenv('COINGECKO_API_KEY') is not None,
        'news_api': os.getenv('NEWS_API_KEY') is not None,
        'google_sheets': os.path.exists('google_credentials.json') if os.getenv('GOOGLE_CREDENTIALS_PATH') else False
    }
    return config_status

def fetch_crypto_market_data():
    """
    Fetch cryptocurrency market data from CoinGecko API
    Fixed to handle the correct column names and API parameters
    """
    try:
        url = "https://api.coingecko.com/api/v3/coins/markets"
        
        # Add API key if available
        headers = {}
        api_key = os.getenv('COINGECKO_API_KEY')
        if api_key:
            headers['x-cg-demo-api-key'] = api_key
        
        # Parameters to get complete market data including 7-day changes
        params = {
            'vs_currency': 'usd',
            'order': 'market_cap_desc',
            'per_page': 100,
            'page': 1,
            'sparkline': False,
            'price_change_percentage': '1h,24h,7d,14d,30d'  # This is crucial for 7d data
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        df = pd.DataFrame(data)
        
        # Fix column name issue - map the correct column names
        column_mapping = {
            'price_change_percentage_7d_in_currency': 'price_change_percentage_7d',
            'price_change_percentage_1h_in_currency': 'price_change_percentage_1h',
            'price_change_percentage_24h_in_currency': 'price_change_percentage_24h_alt'
        }
        
        # Rename columns if they exist
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df[new_col] = df[old_col]
        
        # Ensure required columns exist with default values if missing
        required_columns = ['price_change_percentage_7d', 'price_change_percentage_24h']
        for col in required_columns:
            if col not in df.columns:
                if col == 'price_change_percentage_7d':
                    # If 7d data is not available, use a fallback or NaN
                    df[col] = np.nan
                    st.warning(f"⚠️ 7-day price change data not available. Using fallback calculations.")
        
        return df
        
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Failed to fetch market data: {e}")
        # Return empty DataFrame with expected structure
        return pd.DataFrame(columns=['id', 'name', 'symbol', 'current_price', 
                                   'price_change_percentage_24h', 'price_change_percentage_7d'])
    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
        return pd.DataFrame()

def basic_technical_scores(df):
    """
    Calculate basic technical analysis scores
    Fixed to handle missing data gracefully
    """
    scores = {}
    
    if df.empty:
        return scores
    
    for _, row in df.iterrows():
        try:
            coin_id = row.get('id', 'unknown')
            
            # Get momentum (24h change) safely
            momentum = row.get("price_change_percentage_24h", 0)
            if pd.isna(momentum):
                momentum = 0
            
            # Get volatility (7d change) safely  
            volatility_7d = row.get("price_change_percentage_7d", None)
            if pd.isna(volatility_7d) or volatility_7d is None:
                # Fallback: use 24h change as proxy or default to low volatility
                volatility = abs(momentum) if momentum != 0 else 1
            else:
                volatility = abs(volatility_7d)
            
            # Calculate technical score (0-100)
            if volatility == 0:
                score = 50  # Neutral score for no volatility
            else:
                # Simple scoring: positive momentum vs volatility
                score = max(0, min(100, 50 + (momentum / max(volatility, 0.1)) * 10))
            
            scores[coin_id] = {
                'technical_score': round(score, 2),
                'momentum_24h': round(momentum, 2),
                'volatility_7d': round(volatility, 2)
            }
            
        except Exception as e:
            st.warning(f"⚠️ Error calculating technical score for {row.get('name', 'unknown')}: {e}")
            continue
    
    return scores

def fetch_news_sentiment():
    """
    Fetch and analyze cryptocurrency news sentiment
    Using RSS feeds as fallback when API keys are not available
    """
    try:
        sentiment_scores = {}
        
        # Try RSS feeds first (free option)
        rss_sources = [
            {'name': 'CoinDesk', 'url': 'https://feeds.coindesk.com/rss'},
            {'name': 'CoinTelegraph', 'url': 'https://cointelegraph.com/rss'},
            {'name': 'Bitcoin Magazine', 'url': 'https://bitcoinmagazine.com/.rss/full/'}
        ]
        
        all_articles = []
        
        for source in rss_sources:
            try:
                feed = feedparser.parse(source['url'])
                for entry in feed.entries[:5]:  # Limit to 5 articles per source
                    all_articles.append({
                        'title': entry.get('title', ''),
                        'summary': entry.get('summary', entry.get('description', '')),
                        'source': source['name'],
                        'published': entry.get('published', ''),
                        'link': entry.get('link', '')
                    })
            except Exception as e:
                st.warning(f"⚠️ Could not fetch from {source['name']}: {e}")
                continue
        
        if not all_articles:
            st.warning("⚠️ No news articles retrieved")
            return {'overall_sentiment': 0, 'article_count': 0, 'sources': []}
        
        # Analyze sentiment
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        sentiment_sum = 0
        
        for article in all_articles:
            text = f"{article['title']} {article['summary']}"
            
            # Use VADER sentiment analysis
            sentiment_score = analyzer.polarity_scores(text)
            compound_score = sentiment_score['compound']
            sentiment_sum += compound_score
            
            if compound_score > 0.1:
                positive_count += 1
            elif compound_score < -0.1:
                negative_count += 1
            else:
                neutral_count += 1
        
        # Calculate overall sentiment
        avg_sentiment = sentiment_sum / len(all_articles) if all_articles else 0
        overall_score = max(0, min(100, 50 + (avg_sentiment * 50)))
        
        return {
            'overall_sentiment': round(overall_score, 2),
            'article_count': len(all_articles),
            'positive_articles': positive_count,
            'negative_articles': negative_count,
            'neutral_articles': neutral_count,
            'sources': list(set([article['source'] for article in all_articles])),
            'latest_articles': all_articles[:5]  # Return latest 5 for display
        }
        
    except Exception as e:
        st.error(f"❌ Error fetching news sentiment: {e}")
        return {'overall_sentiment': 50, 'article_count': 0, 'sources': []}

def compute_opportunity(market_df, tech_scores, sent_scores):
    """
    Compute investment opportunity scores
    """
    opportunity_scores = {}
    
    if market_df.empty or not tech_scores:
        return opportunity_scores
    
    try:
        for _, row in market_df.head(20).iterrows():  # Top 20 by market cap
            coin_id = row.get('id')
            if not coin_id or coin_id not in tech_scores:
                continue
            
            # Get scores
            tech_score = tech_scores[coin_id]['technical_score']
            sentiment_weight = sent_scores.get('overall_sentiment', 50) / 100
            
            # Market cap factor (prefer established coins)
            market_cap = row.get('market_cap', 0)
            market_cap_score = min(100, market_cap / 1e9)  # Normalized to billions
            
            # Volume factor (prefer liquid markets)
            volume = row.get('total_volume', 0)
            volume_score = min(100, volume / 1e8)  # Normalized
            
            # Combined opportunity score
            opportunity = (
                tech_score * 0.4 +           # Technical analysis weight
                sentiment_weight * 50 * 0.3 + # Sentiment weight  
                market_cap_score * 0.2 +      # Market cap weight
                volume_score * 0.1            # Volume weight
            )
            
            opportunity_scores[coin_id] = {
                'opportunity_score': round(opportunity, 2),
                'coin_name': row.get('name', coin_id),
                'symbol': row.get('symbol', '').upper(),
                'current_price': row.get('current_price', 0),
                'market_cap': market_cap,
                'volume_24h': volume,
                'change_24h': row.get('price_change_percentage_24h', 0)
            }
    
    except Exception as e:
        st.error(f"❌ Error computing opportunity scores: {e}")
    
    return opportunity_scores

def display_configuration_guide():
    """Display configuration guide when APIs are not set up"""
    st.markdown("## 🔧 Configuration Required")
    
    config_status = check_configuration()
    
    with st.expander("📋 Configuration Status", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### API Keys Status")
            status_icon = "✅" if config_status['coingecko_api'] else "❌"
            st.markdown(f"{status_icon} **CoinGecko API**: {'Configured' if config_status['coingecko_api'] else 'Missing'}")
            
            status_icon = "✅" if config_status['news_api'] else "❌"
            st.markdown(f"{status_icon} **News API**: {'Configured' if config_status['news_api'] else 'Using RSS fallback'}")
            
            status_icon = "✅" if config_status['google_sheets'] else "❌"
            st.markdown(f"{status_icon} **Google Sheets**: {'Configured' if config_status['google_sheets'] else 'Not configured'}")
        
        with col2:
            st.markdown("### Quick Setup")
            st.markdown("""
            1. **CoinGecko API** (Recommended):
               - Visit [coingecko.com/api](https://www.coingecko.com/en/api)
               - Create free account
               - Get API key
               - Add to Streamlit secrets
            
            2. **News API** (Optional):
               - Visit [newsapi.org](https://newsapi.org)
               - Get free API key
            
            3. **Google Sheets** (Optional):
               - Follow Google Cloud setup guide
               - Download credentials JSON
            """)
    
    return config_status

# Main App Layout
def main():
    st.title("💰 CryptoIntelligence Dashboard")
    st.markdown("*Automated cryptocurrency market analysis and opportunity detection*")
    
    # Check configuration
    config_status = check_configuration()
    
    # Show configuration guide if needed
    if not any(config_status.values()):
        display_configuration_guide()
        st.info("📝 The app works with basic functionality using free data sources. Configure APIs for enhanced features.")
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Market Overview", "📈 Technical Analysis", "📰 News Sentiment", "🎯 Opportunities"])
    
    # Fetch data
    with st.spinner("📡 Fetching market data..."):
        market_df = fetch_crypto_market_data()
    
    with tab1:
        st.header("📊 Market Overview")
        
        if not market_df.empty:
            # Display top cryptocurrencies
            st.subheader("Top 20 Cryptocurrencies by Market Cap")
            
            # Select columns to display
            display_columns = ['name', 'symbol', 'current_price', 'market_cap', 
                             'price_change_percentage_24h', 'price_change_percentage_7d']
            
            # Filter columns that exist
            available_columns = [col for col in display_columns if col in market_df.columns]
            
            if available_columns:
                display_df = market_df[available_columns].head(20).copy()
                
                # Format the dataframe for better display
                if 'current_price' in display_df.columns:
                    display_df['current_price'] = display_df['current_price'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A")
                
                if 'market_cap' in display_df.columns:
                    display_df['market_cap'] = display_df['market_cap'].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A")
                
                for col in ['price_change_percentage_24h', 'price_change_percentage_7d']:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
                
                st.dataframe(display_df, use_container_width=True)
            else:
                st.error("No market data columns available")
        else:
            st.error("Unable to fetch market data. Please check your internet connection.")
    
    with tab2:
        st.header("📈 Technical Analysis")
        
        if not market_df.empty:
            with st.spinner("🔍 Calculating technical scores..."):
                tech_scores = basic_technical_scores(market_df)
            
            if tech_scores:
                st.subheader("Technical Analysis Scores")
                
                # Convert to DataFrame for display
                tech_df = pd.DataFrame.from_dict(tech_scores, orient='index')
                tech_df = tech_df.reset_index()
                tech_df.columns = ['Coin ID'] + list(tech_df.columns[1:])
                
                # Sort by technical score
                tech_df = tech_df.sort_values('technical_score', ascending=False)
                
                st.dataframe(tech_df.head(15), use_container_width=True)
                
                # Show score distribution
                st.subheader("Score Distribution")
                fig_data = tech_df['technical_score'].values
                st.bar_chart(fig_data[:10])  # Top 10
            else:
                st.warning("No technical analysis data available")
        else:
            st.error("No market data available for technical analysis")
    
    with tab3:
        st.header("📰 News Sentiment Analysis")
        
        with st.spinner("📰 Analyzing news sentiment..."):
            sent_scores = fetch_news_sentiment()
        
        if sent_scores and sent_scores.get('article_count', 0) > 0:
            # Display overall sentiment
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Overall Sentiment", f"{sent_scores['overall_sentiment']:.1f}/100")
            
            with col2:
                st.metric("Articles Analyzed", sent_scores['article_count'])
            
            with col3:
                sentiment_label = "Bullish" if sent_scores['overall_sentiment'] > 60 else "Bearish" if sent_scores['overall_sentiment'] < 40 else "Neutral"
                st.metric("Market Outlook", sentiment_label)
            
            # Sentiment breakdown
            st.subheader("Sentiment Breakdown")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Positive Articles", sent_scores.get('positive_articles', 0), delta="Bullish")
            
            with col2:
                st.metric("Neutral Articles", sent_scores.get('neutral_articles', 0))
            
            with col3:
                st.metric("Negative Articles", sent_scores.get('negative_articles', 0), delta="Bearish")
            
            # Latest articles
            if 'latest_articles' in sent_scores:
                st.subheader("Latest News Headlines")
                for article in sent_scores['latest_articles']:
                    with st.expander(f"📰 {article['title'][:100]}..."):
                        st.markdown(f"**Source:** {article['source']}")
                        st.markdown(f"**Summary:** {article['summary'][:200]}...")
                        if article.get('link'):
                            st.markdown(f"[Read full article]({article['link']})")
        else:
            st.warning("No news sentiment data available")
    
    with tab4:
        st.header("🎯 Investment Opportunities")
        
        if not market_df.empty:
            with st.spinner("🎯 Computing opportunity scores..."):
                tech_scores = basic_technical_scores(market_df)
                sent_scores = fetch_news_sentiment()
                opp_scores = compute_opportunity(market_df, tech_scores, sent_scores)
            
            if opp_scores:
                st.subheader("Top Investment Opportunities")
                
                # Convert to DataFrame and sort
                opp_df = pd.DataFrame.from_dict(opp_scores, orient='index')
                opp_df = opp_df.sort_values('opportunity_score', ascending=False)
                
                # Display top opportunities
                for i, (coin_id, data) in enumerate(opp_df.head(10).iterrows()):
                    with st.container():
                        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                        
                        with col1:
                            st.markdown(f"**#{i+1} {data['coin_name']} ({data['symbol']})**")
                        
                        with col2:
                            st.metric("Opportunity Score", f"{data['opportunity_score']:.1f}/100")
                        
                        with col3:
                            st.metric("Current Price", f"${data['current_price']:,.2f}")
                        
                        with col4:
                            change_24h = data['change_24h']
                            st.metric("24h Change", f"{change_24h:.2f}%", delta=f"{change_24h:.2f}%")
            else:
                st.warning("No opportunity scores available")
        else:
            st.error("No market data available for opportunity analysis")
    
    # Footer
    st.markdown("---")
    st.markdown("*CryptoIntelligence Dashboard - Built with Streamlit*")
    st.markdown("⚠️ **Disclaimer**: This is for educational purposes only. Not financial advice.")

if __name__ == "__main__":
    main()
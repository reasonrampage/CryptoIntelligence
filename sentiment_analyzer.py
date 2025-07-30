"""
Free Sentiment Analyzer for Crypto News
Uses only free, open-source sentiment analysis tools:
- VADER Sentiment (completely free)
- TextBlob (completely free)  
- NLTK (completely free)
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import re
import math

# Free sentiment analysis libraries
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import nltk

class FreeSentimentAnalyzer:
    """Analyzes sentiment using only free, open-source tools."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Download required NLTK data (one-time setup)
        self._setup_nltk()
        
        # Initialize VADER analyzer (completely free)
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Source credibility weights (based on domain reputation)
        self.source_weights = {
            'cointelegraph.com': 1.0,
            'coindesk.com': 1.0,
            'bitcoinist.com': 0.9,
            'newsbtc.com': 0.9,
            'cryptopotato.com': 0.8,
            'decrypt.co': 0.9,
            '99bitcoins.com': 0.8,
            'cryptobriefing.com': 0.9,
            'crypto.news': 0.8,
            'zycrypto.com': 0.7,
            'unknown': 0.5
        }
        
        # Crypto-specific sentiment boosters/dampeners
        self.crypto_sentiment_modifiers = {
            # Positive boosters
            'bullish': 0.3, 'moon': 0.4, 'pump': 0.3, 'surge': 0.2,
            'rally': 0.2, 'breakout': 0.3, 'adoption': 0.2, 'partnership': 0.2,
            'upgrade': 0.2, 'milestone': 0.2, 'breakthrough': 0.3,
            
            # Negative dampeners
            'bearish': -0.3, 'dump': -0.3, 'crash': -0.4, 'plunge': -0.3,
            'correction': -0.2, 'sell-off': -0.3, 'decline': -0.2, 'drop': -0.2,
            'hack': -0.4, 'scam': -0.4, 'ban': -0.3, 'regulation': -0.1,
            
            # Neutral/context dependent
            'volatile': 0.0, 'trading': 0.0, 'analysis': 0.0
        }
    
    def _setup_nltk(self):
        """Download required NLTK data."""
        try:
            nltk_downloads = ['punkt', 'vader_lexicon', 'stopwords']
            for item in nltk_downloads:
                try:
                    nltk.data.find(f'tokenizers/{item}')
                except LookupError:
                    nltk.download(item, quiet=True)
        except Exception as e:
            self.logger.warning(f"NLTK setup issue: {e}")
    
    def _clean_text_for_sentiment(self, text: str) -> str:
        """Clean text specifically for sentiment analysis."""
        if not text:
            return ""
        
        # Convert to lowercase for analysis
        text = text.lower()
        
        # Handle crypto ticker symbols (keep them as they carry sentiment)
        text = re.sub(r'\$([a-z]{2,10})', r'\1', text)  # $BTC -> btc
        
        # Remove URLs but keep the text flow
        text = re.sub(r'http[s]?://\S+', ' ', text)
        
        # Clean up extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def analyze_vader_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using VADER (free)."""
        if not text:
            return {'compound': 0.0, 'pos': 0.0, 'neu': 1.0, 'neg': 0.0}
        
        clean_text = self._clean_text_for_sentiment(text)
        scores = self.vader_analyzer.polarity_scores(clean_text)
        
        return scores
    
    def analyze_textblob_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using TextBlob (free)."""
        if not text:
            return {'polarity': 0.0, 'subjectivity': 0.0}
        
        clean_text = self._clean_text_for_sentiment(text)
        blob = TextBlob(clean_text)
        
        return {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }
    
    def apply_crypto_modifiers(self, base_sentiment: float, text: str) -> float:
        """Apply crypto-specific sentiment modifiers."""
        if not text:
            return base_sentiment
        
        text_lower = text.lower()
        modifier_sum = 0.0
        modifier_count = 0
        
        for term, modifier in self.crypto_sentiment_modifiers.items():
            if term in text_lower:
                modifier_sum += modifier
                modifier_count += 1
        
        if modifier_count > 0:
            avg_modifier = modifier_sum / modifier_count
            # Apply modifier with diminishing returns
            adjusted_sentiment = base_sentiment + (avg_modifier * 0.5)
            # Keep within bounds [-1, 1]
            adjusted_sentiment = max(-1.0, min(1.0, adjusted_sentiment))
            return adjusted_sentiment
        
        return base_sentiment
    
    def calculate_composite_sentiment(self, text: str) -> Dict[str, float]:
        """Calculate composite sentiment using multiple free methods."""
        # Get VADER sentiment
        vader_scores = self.analyze_vader_sentiment(text)
        
        # Get TextBlob sentiment
        textblob_scores = self.analyze_textblob_sentiment(text)
        
        # Combine scores (weighted average)
        # VADER is better for social media/informal text
        # TextBlob is better for formal text
        vader_weight = 0.7
        textblob_weight = 0.3
        
        composite_sentiment = (
            vader_scores['compound'] * vader_weight +
            textblob_scores['polarity'] * textblob_weight
        )
        
        # Apply crypto-specific modifiers
        adjusted_sentiment = self.apply_crypto_modifiers(composite_sentiment, text)
        
        return {
            'composite_sentiment': adjusted_sentiment,
            'vader_compound': vader_scores['compound'],
            'vader_pos': vader_scores['pos'],
            'vader_neu': vader_scores['neu'],
            'vader_neg': vader_scores['neg'],
            'textblob_polarity': textblob_scores['polarity'],
            'textblob_subjectivity': textblob_scores['subjectivity'],
            'crypto_adjusted': adjusted_sentiment != composite_sentiment
        }
    
    def calculate_recency_weight(self, published_date: datetime, half_life_hours: float = 12.0) -> float:
        """Calculate time-based weight with exponential decay."""
        if not published_date:
            return 0.1  # Very low weight for articles without dates
        
        hours_ago = (datetime.utcnow() - published_date).total_seconds() / 3600
        
        # Exponential decay: weight = exp(-ln(2) * hours_ago / half_life)
        decay_factor = math.exp(-math.log(2) * hours_ago / half_life_hours)
        
        # Ensure minimum weight of 0.1 and maximum of 1.0
        return max(0.1, min(1.0, decay_factor))
    
    def get_source_weight(self, source_domain: str) -> float:
        """Get credibility weight for news source."""
        return self.source_weights.get(source_domain.lower(), 0.5)
    
    def analyze_news_sentiment(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Analyze sentiment for a DataFrame of news articles."""
        if news_df.empty:
            return news_df
        
        self.logger.info(f"Analyzing sentiment for {len(news_df)} articles...")
        
        results = []
        
        for idx, row in news_df.iterrows():
            # Combine title and description for analysis
            text_content = f"{row.get('title', '')} {row.get('description', '')}"
            
            # Calculate sentiment scores
            sentiment_scores = self.calculate_composite_sentiment(text_content)
            
            # Calculate weights
            source_weight = self.get_source_weight(row.get('source_domain', ''))
            
            published_date = row.get('published')
            if pd.isna(published_date):
                published_date = datetime.utcnow() - timedelta(hours=1)
            
            recency_weight = self.calculate_recency_weight(published_date)
            
            # Calculate final weighted sentiment
            weighted_sentiment = (
                sentiment_scores['composite_sentiment'] *
                source_weight *
                recency_weight
            )
            
            # Compile results
            result = {
                'article_id': idx,
                'title': row.get('title', ''),
                'source_domain': row.get('source_domain', ''),
                'published': published_date,
                'text_length': len(text_content),
                
                # Raw sentiment scores
                'raw_sentiment': sentiment_scores['composite_sentiment'],
                'vader_compound': sentiment_scores['vader_compound'],
                'textblob_polarity': sentiment_scores['textblob_polarity'],
                
                # Weights
                'source_weight': source_weight,
                'recency_weight': recency_weight,
                
                # Final weighted sentiment
                'weighted_sentiment': weighted_sentiment,
                
                # Classification
                'sentiment_label': self._classify_sentiment(weighted_sentiment),
                'confidence': abs(weighted_sentiment),
                
                # Additional metrics
                'subjectivity': sentiment_scores['textblob_subjectivity'],
                'crypto_adjusted': sentiment_scores['crypto_adjusted'],
                
                'analysis_timestamp': datetime.utcnow()
            }
            
            results.append(result)
        
        sentiment_df = pd.DataFrame(results)
        self.logger.info(f"Sentiment analysis completed for {len(sentiment_df)} articles")
        
        return sentiment_df
    
    def _classify_sentiment(self, sentiment_score: float) -> str:
        """Classify sentiment into categories."""
        if sentiment_score >= 0.05:
            return 'positive'
        elif sentiment_score <= -0.05:
            return 'negative'
        else:
            return 'neutral'
    
    def calculate_market_sentiment_metrics(self, sentiment_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate overall market sentiment metrics."""
        if sentiment_df.empty:
            return {}
        
        # Overall metrics
        total_articles = len(sentiment_df)
        avg_sentiment = sentiment_df['weighted_sentiment'].mean()
        sentiment_std = sentiment_df['weighted_sentiment'].std()
        
        # Classification counts
        sentiment_counts = sentiment_df['sentiment_label'].value_counts()
        positive_pct = sentiment_counts.get('positive', 0) / total_articles * 100
        negative_pct = sentiment_counts.get('negative', 0) / total_articles * 100
        neutral_pct = sentiment_counts.get('neutral', 0) / total_articles * 100
        
        # Recent sentiment (last 6 hours)
        six_hours_ago = datetime.utcnow() - timedelta(hours=6)
        recent_mask = sentiment_df['published'] >= six_hours_ago
        recent_sentiment = sentiment_df[recent_mask]['weighted_sentiment'].mean()
        
        # Sentiment momentum (comparing last 6h vs previous 6h)
        twelve_hours_ago = datetime.utcnow() - timedelta(hours=12)
        previous_mask = (sentiment_df['published'] >= twelve_hours_ago) & (sentiment_df['published'] < six_hours_ago)
        previous_sentiment = sentiment_df[previous_mask]['weighted_sentiment'].mean()
        
        momentum = recent_sentiment - previous_sentiment if not pd.isna(previous_sentiment) else 0
        
        # Top sources sentiment
        source_sentiment = sentiment_df.groupby('source_domain')['weighted_sentiment'].agg(['mean', 'count'])
        top_sources = source_sentiment[source_sentiment['count'] >= 3].sort_values('mean', ascending=False)
        
        metrics = {
            'total_articles': total_articles,
            'overall_sentiment': avg_sentiment,
            'sentiment_volatility': sentiment_std,
            'positive_percentage': positive_pct,
            'negative_percentage': negative_pct,
            'neutral_percentage': neutral_pct,
            'recent_sentiment_6h': recent_sentiment,
            'sentiment_momentum': momentum,
            'bullish_articles': int(sentiment_counts.get('positive', 0)),
            'bearish_articles': int(sentiment_counts.get('negative', 0)),
            'confidence_avg': sentiment_df['confidence'].mean(),
            'high_confidence_articles': len(sentiment_df[sentiment_df['confidence'] >= 0.5])
        }
        
        return metrics
    
    def save_sentiment_analysis(self, sentiment_df: pd.DataFrame, output_dir: str = './data/') -> str:
        """Save sentiment analysis results to CSV."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{output_dir}sentiment_analysis_{timestamp}.csv"
        
        sentiment_df.to_csv(filename, index=False)
        self.logger.info(f"Saved sentiment analysis to {filename}")
        return filename


def main():
    """Example usage of the Free Sentiment Analyzer."""
    # Sample news data for testing
    sample_news = pd.DataFrame([
        {
            'title': 'Bitcoin Surges to New All-Time High Amid Institutional Adoption',
            'description': 'Bitcoin has reached unprecedented levels as major corporations continue to adopt cryptocurrency.',
            'source_domain': 'cointelegraph.com',
            'published': datetime.utcnow() - timedelta(hours=2)
        },
        {
            'title': 'Crypto Market Faces Regulatory Uncertainty',
            'description': 'Government officials express concerns about cryptocurrency regulations, causing market volatility.',
            'source_domain': 'coindesk.com',
            'published': datetime.utcnow() - timedelta(hours=5)
        },
        {
            'title': 'Ethereum Network Upgrade Shows Promise',
            'description': 'The latest Ethereum upgrade demonstrates improved scalability and reduced transaction fees.',
            'source_domain': 'decrypt.co',
            'published': datetime.utcnow() - timedelta(hours=1)
        }
    ])
    
    # Initialize analyzer
    analyzer = FreeSentimentAnalyzer()
    
    # Analyze sentiment
    sentiment_results = analyzer.analyze_news_sentiment(sample_news)
    
    # Calculate market metrics
    market_metrics = analyzer.calculate_market_sentiment_metrics(sentiment_results)
    
    # Display results
    print("Sentiment Analysis Results:")
    print("=" * 50)
    
    for idx, row in sentiment_results.iterrows():
        print(f"\nTitle: {row['title'][:60]}...")
        print(f"Source: {row['source_domain']}")
        print(f"Sentiment: {row['sentiment_label']} ({row['weighted_sentiment']:.3f})")
        print(f"Confidence: {row['confidence']:.3f}")
        print(f"Source Weight: {row['source_weight']:.2f}")
        print(f"Recency Weight: {row['recency_weight']:.2f}")
    
    print(f"\nMarket Sentiment Metrics:")
    print("=" * 30)
    for metric, value in market_metrics.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.3f}")
        else:
            print(f"{metric}: {value}")


if __name__ == "__main__":
    main()
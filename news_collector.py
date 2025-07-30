"""
Free News Collector for Crypto Intelligence
Uses only free sources: RSS feeds and free news APIs
"""

import requests
import feedparser
import pandas as pd
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import re
from urllib.parse import urlparse

class FreeNewsCollector:
    """Collects crypto news using only free sources."""
    
    def __init__(self, newsdata_api_key: Optional[str] = None):
        self.newsdata_api_key = newsdata_api_key
        self.session = requests.Session()
        
        # Free RSS feeds for crypto news
        self.rss_feeds = [
            'https://cointelegraph.com/rss',
            'https://bitcoinist.com/feed',
            'https://newsbtc.com/feed',
            'https://cryptopotato.com/feed',
            'https://99bitcoins.com/feed',
            'https://cryptobriefing.com/feed',
            'https://crypto.news/feed',
            'https://zycrypto.com/feed',
            'https://www.coindesk.com/arc/outboundfeeds/rss/',
            'https://decrypt.co/feed',
        ]
        
        # Rate limiting for NewsData.io free tier (500 calls/month)
        self.newsdata_calls_made = 0
        self.newsdata_daily_limit = 15  # Conservative limit
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL for source identification."""
        try:
            return urlparse(url).netloc.lower()
        except:
            return 'unknown'
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove common RSS feed artifacts
        text = re.sub(r'The post .* appeared first on .*\.', '', text)
        text = re.sub(r'Continue reading .*', '', text)
        
        return text
    
    def collect_rss_news(self, hours_back: int = 24) -> pd.DataFrame:
        """Collect news from free RSS feeds."""
        self.logger.info(f"Collecting RSS news from last {hours_back} hours...")
        
        all_articles = []
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
        
        for feed_url in self.rss_feeds:
            try:
                self.logger.info(f"Parsing RSS feed: {feed_url}")
                feed = feedparser.parse(feed_url)
                
                source_domain = self._extract_domain(feed_url)
                
                for entry in feed.entries:
                    # Parse publish date
                    pub_date = None
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        pub_date = datetime(*entry.published_parsed[:6])
                    elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                        pub_date = datetime(*entry.updated_parsed[:6])
                    
                    # Skip old articles
                    if pub_date and pub_date < cutoff_time:
                        continue
                    
                    # Extract article data
                    article = {
                        'title': self._clean_text(getattr(entry, 'title', '')),
                        'description': self._clean_text(getattr(entry, 'description', '')),
                        'summary': self._clean_text(getattr(entry, 'summary', '')),
                        'link': getattr(entry, 'link', ''),
                        'published': pub_date,
                        'source_domain': source_domain,
                        'source_feed': feed_url,
                        'authors': getattr(entry, 'author', ''),
                        'tags': [tag.term for tag in getattr(entry, 'tags', [])],
                        'collected_at': datetime.utcnow()
                    }
                    
                    all_articles.append(article)
                
                # Small delay to be respectful
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error parsing RSS feed {feed_url}: {e}")
                continue
        
        df = pd.DataFrame(all_articles)
        self.logger.info(f"Collected {len(df)} articles from RSS feeds")
        return df
    
    def collect_newsdata_crypto_news(self, hours_back: int = 24) -> pd.DataFrame:
        """Collect crypto news from NewsData.io free API."""
        if not self.newsdata_api_key:
            self.logger.warning("NewsData.io API key not provided, skipping...")
            return pd.DataFrame()
        
        if self.newsdata_calls_made >= self.newsdata_daily_limit:
            self.logger.warning("NewsData.io daily limit reached, skipping...")
            return pd.DataFrame()
        
        self.logger.info("Collecting crypto news from NewsData.io...")
        
        # Calculate date range
        to_date = datetime.utcnow()
        from_date = to_date - timedelta(hours=hours_back)
        
        params = {
            'apikey': self.newsdata_api_key,
            'q': 'cryptocurrency OR bitcoin OR ethereum OR crypto OR blockchain',
            'language': 'en',
            'category': 'business,technology',
            'from_date': from_date.strftime('%Y-%m-%d'),
            'to_date': to_date.strftime('%Y-%m-%d'),
            'size': 50  # Max 50 articles per request
        }
        
        try:
            response = self.session.get(
                "https://newsdata.io/api/1/news", 
                params=params, 
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            self.newsdata_calls_made += 1
            
            if data.get('status') != 'success':
                self.logger.warning(f"NewsData.io API returned: {data}")
                return pd.DataFrame()
            
            articles = []
            for item in data.get('results', []):
                article = {
                    'title': self._clean_text(item.get('title', '')),
                    'description': self._clean_text(item.get('description', '')),
                    'content': self._clean_text(item.get('content', '')),
                    'link': item.get('link', ''),
                    'published': pd.to_datetime(item.get('pubDate')),
                    'source_domain': item.get('source_id', ''),
                    'source_name': item.get('source_name', ''),
                    'country': item.get('country', []),
                    'category': item.get('category', []),
                    'language': item.get('language', ''),
                    'collected_at': datetime.utcnow()
                }
                articles.append(article)
            
            df = pd.DataFrame(articles)
            self.logger.info(f"Collected {len(df)} articles from NewsData.io")
            return df
            
        except Exception as e:
            self.logger.error(f"Error collecting from NewsData.io: {e}")
            return pd.DataFrame()
    
    def filter_crypto_relevant(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter articles for crypto relevance."""
        if df.empty:
            return df
        
        crypto_keywords = [
            'bitcoin', 'btc', 'ethereum', 'eth', 'cryptocurrency', 'crypto',
            'blockchain', 'defi', 'nft', 'altcoin', 'trading', 'price',
            'market', 'binance', 'coinbase', 'wallet', 'mining', 'staking',
            'dogecoin', 'cardano', 'solana', 'polygon', 'chainlink',
            'decentralized', 'digital currency', 'web3', 'metaverse'
        ]
        
        # Create text column for searching
        df['full_text'] = (
            df.get('title', '').fillna('') + ' ' +
            df.get('description', '').fillna('') + ' ' +
            df.get('summary', '').fillna('') + ' ' +
            df.get('content', '').fillna('')
        ).str.lower()
        
        # Filter for crypto relevance
        crypto_pattern = '|'.join(crypto_keywords)
        crypto_mask = df['full_text'].str.contains(crypto_pattern, case=False, na=False)
        
        filtered_df = df[crypto_mask].copy()
        filtered_df.drop('full_text', axis=1, inplace=True)
        
        self.logger.info(f"Filtered to {len(filtered_df)} crypto-relevant articles")
        return filtered_df
    
    def collect_all_news(self, hours_back: int = 24) -> pd.DataFrame:
        """Collect news from all free sources."""
        self.logger.info("Starting comprehensive news collection...")
        
        all_dataframes = []
        
        # Collect RSS news
        rss_news = self.collect_rss_news(hours_back)
        if not rss_news.empty:
            rss_news['source_type'] = 'rss'
            all_dataframes.append(rss_news)
        
        # Collect NewsData.io news
        newsdata_news = self.collect_newsdata_crypto_news(hours_back)
        if not newsdata_news.empty:
            newsdata_news['source_type'] = 'newsdata_api'
            all_dataframes.append(newsdata_news)
        
        if not all_dataframes:
            self.logger.warning("No news data collected from any source")
            return pd.DataFrame()
        
        # Combine all sources
        combined_df = pd.concat(all_dataframes, ignore_index=True, sort=False)
        
        # Filter for crypto relevance
        crypto_news = self.filter_crypto_relevant(combined_df)
        
        # Remove duplicates based on title similarity
        crypto_news = self._remove_duplicates(crypto_news)
        
        # Add collection metadata
        crypto_news['collection_timestamp'] = datetime.utcnow()
        crypto_news = crypto_news.sort_values('published', ascending=False)
        
        self.logger.info(f"Final collection: {len(crypto_news)} unique crypto articles")
        return crypto_news
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate articles based on title similarity."""
        if df.empty:
            return df
        
        # Simple deduplication based on similar titles
        df['title_clean'] = df['title'].str.lower().str.replace(r'[^\w\s]', '', regex=True)
        df = df.drop_duplicates(subset=['title_clean'], keep='first')
        df.drop('title_clean', axis=1, inplace=True)
        
        return df
    
    def save_news_data(self, df: pd.DataFrame, output_dir: str = './data/') -> str:
        """Save news data to CSV file."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{output_dir}crypto_news_{timestamp}.csv"
        
        df.to_csv(filename, index=False)
        self.logger.info(f"Saved {len(df)} articles to {filename}")
        return filename
    
    def get_news_summary_stats(self, df: pd.DataFrame) -> Dict:
        """Get summary statistics of collected news."""
        if df.empty:
            return {}
        
        stats = {
            'total_articles': len(df),
            'unique_sources': df['source_domain'].nunique(),
            'date_range': {
                'earliest': df['published'].min(),
                'latest': df['published'].max()
            },
            'top_sources': df['source_domain'].value_counts().head(5).to_dict(),
            'articles_per_hour': len(df) / 24,  # Assuming 24-hour collection
        }
        
        return stats


def main():
    """Example usage of the Free News Collector."""
    # Initialize collector (API key is optional)
    collector = FreeNewsCollector()
    
    # Collect news from last 24 hours
    news_df = collector.collect_all_news(hours_back=24)
    
    if not news_df.empty:
        # Save to CSV
        filename = collector.save_news_data(news_df)
        
        # Print summary
        stats = collector.get_news_summary_stats(news_df)
        print("\nNews Collection Summary:")
        print(f"Total articles: {stats.get('total_articles', 0)}")
        print(f"Unique sources: {stats.get('unique_sources', 0)}")
        print(f"Articles per hour: {stats.get('articles_per_hour', 0):.1f}")
        
        print("\nTop sources:")
        for source, count in stats.get('top_sources', {}).items():
            print(f"  {source}: {count} articles")
        
        print(f"\nSample articles:")
        print(news_df[['title', 'source_domain', 'published']].head())
    else:
        print("No news articles collected.")


if __name__ == "__main__":
    main()
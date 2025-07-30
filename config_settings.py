# Configuration settings for Free Crypto Trading Intelligence Tool
# All APIs used are completely free with generous limits

import os
from typing import List, Dict

# =====================
# FREE API CONFIGURATIONS
# =====================

# CoinGecko API (Free - 30 calls/min, 10,000 calls/month)
COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"
COINGECKO_API_KEY = os.getenv('COINGECKO_API_KEY', '')  # Optional for free tier

# Free Crypto News RSS Feeds
FREE_CRYPTO_NEWS_FEEDS = [
    'https://cointelegraph.com/rss',
    'https://bitcoinist.com/feed',
    'https://newsbtc.com/feed',
    'https://cryptopotato.com/feed',
    'https://99bitcoins.com/feed',
    'https://cryptobriefing.com/feed',
    'https://crypto.news/feed',
    'https://zycrypto.com/feed',
]

# NewsData.io API (Free - 500 calls/month)
NEWSDATA_API_KEY = os.getenv('NEWSDATA_API_KEY', '')
NEWSDATA_BASE_URL = "https://newsdata.io/api/1/news"

# Google Sheets API (Free with limits)
GOOGLE_SHEETS_CREDENTIALS_FILE = os.getenv('GOOGLE_SHEETS_CREDENTIALS', 'credentials.json')
GOOGLE_SHEETS_ID = os.getenv('GOOGLE_SHEETS_ID', '')

# =====================
# ANALYSIS SETTINGS
# =====================

# Cryptocurrencies to analyze (top 50 by market cap)
TARGET_CRYPTOCURRENCIES = [
    'bitcoin', 'ethereum', 'binancecoin', 'cardano', 'solana',
    'xrp', 'polkadot', 'dogecoin', 'avalanche-2', 'shiba-inu',
    'polygon', 'chainlink', 'litecoin', 'bitcoin-cash', 'algorand',
    'stellar', 'vechain', 'filecoin', 'tron', 'ethereum-classic',
    'monero', 'eos', 'aave', 'cosmos', 'theta-token',
    'neo', 'pancakeswap-token', 'dash', 'zcash', 'maker',
    'compound', 'sushiswap', '1inch', 'yearn-finance', 'uma',
    'synthetix-network-token', 'curve-dao-token', 'uniswap',
    'celsius-degree-token', 'ren', 'kyber-network-crystal',
    'loopring', 'basic-attention-token', 'enjincoin', 'chiliz',
    'decentraland', 'the-sandbox', 'axie-infinity', 'flow', 'helium'
]

# Technical Indicators Configuration
TECHNICAL_INDICATORS = {
    'sma_periods': [20, 50, 200],
    'ema_periods': [12, 26, 50],
    'rsi_period': 14,
    'macd_config': {'fast': 12, 'slow': 26, 'signal': 9},
    'bollinger_bands': {'period': 20, 'std': 2},
    'volume_sma': 20,
}

# Sentiment Analysis Configuration
SENTIMENT_CONFIG = {
    'news_lookback_hours': 24,
    'sentiment_threshold_positive': 0.05,
    'sentiment_threshold_negative': -0.05,
    'source_weights': {
        'cointelegraph.com': 1.0,
        'bitcoinist.com': 0.9,
        'newsbtc.com': 0.9,
        'cryptopotato.com': 0.8,
        '99bitcoins.com': 0.8,
        'cryptobriefing.com': 0.9,
        'crypto.news': 0.8,
        'zycrypto.com': 0.7,
        'unknown': 0.5
    }
}

# Opportunity Score Weights (must sum to 1.0)
OPPORTUNITY_SCORE_WEIGHTS = {
    'sentiment_weight': 0.35,
    'technical_weight': 0.35,
    'volume_weight': 0.15,
    'volatility_weight': 0.15
}

# =====================
# GOOGLE SHEETS CONFIGURATION
# =====================

SHEETS_CONFIG = {
    'raw_data_sheet': 'RawData',
    'sentiment_sheet': 'NewsSentiment',
    'technical_sheet': 'TechIndicators',
    'opportunity_sheet': 'OpportunityScores',
    'leaderboard_sheet': 'Leaderboard',
    'update_frequency_minutes': 60
}

# =====================
# RATE LIMITING & SAFETY
# =====================

RATE_LIMITS = {
    'coingecko_calls_per_minute': 25,  # Stay under 30/min limit
    'newsdata_calls_per_day': 15,     # Stay under 500/month limit
    'sheets_writes_per_minute': 10,   # Conservative Google Sheets limit
    'request_delay_seconds': 2        # Delay between API calls
}

# =====================
# GITHUB ACTIONS SCHEDULE
# =====================

# Cron schedule for GitHub Actions (free 2,000 minutes/month)
GITHUB_ACTIONS_SCHEDULE = "0 */4 * * *"  # Every 4 hours

# =====================
# DATA RETENTION
# =====================

DATA_RETENTION_DAYS = 30  # Keep data for 30 days to stay within free limits

# =====================
# LOGGING
# =====================

LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'crypto_intelligence.log'
}

# =====================
# BACKUP CONFIGURATION
# =====================

BACKUP_CONFIG = {
    'enabled': True,
    'local_backup_path': './backups/',
    'max_backup_files': 10
}

def validate_configuration():
    """Validate that all required configuration is present."""
    required_env_vars = ['GOOGLE_SHEETS_ID']
    
    missing_vars = []
    for var in required_env_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {missing_vars}")
    
    # Validate weights sum to 1.0
    weight_sum = sum(OPPORTUNITY_SCORE_WEIGHTS.values())
    if abs(weight_sum - 1.0) > 0.001:
        raise ValueError(f"Opportunity score weights must sum to 1.0, got {weight_sum}")
    
    print("✅ Configuration validation passed!")

if __name__ == "__main__":
    validate_configuration()
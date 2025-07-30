"""
Free Crypto Data Collector
Uses only free APIs: CoinGecko (30 calls/min, 10k/month free)
"""

import requests
import pandas as pd
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json

class FreeCryptoDataCollector:
    """Collects cryptocurrency data using free APIs only."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.base_url = "https://api.coingecko.com/api/v3"
        self.api_key = api_key  # Optional for free tier
        self.session = requests.Session()
        
        # Rate limiting for free tier (30 calls/min)
        self.calls_per_minute = 25  # Stay under limit
        self.last_call_times = []
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _rate_limit(self):
        """Implement rate limiting for free API."""
        now = time.time()
        
        # Remove calls older than 1 minute
        self.last_call_times = [t for t in self.last_call_times if now - t < 60]
        
        # If we've made too many calls, wait
        if len(self.last_call_times) >= self.calls_per_minute:
            sleep_time = 60 - (now - self.last_call_times[0])
            if sleep_time > 0:
                self.logger.info(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
        
        self.last_call_times.append(now)
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make a rate-limited request to CoinGecko API."""
        self._rate_limit()
        
        url = f"{self.base_url}/{endpoint}"
        if params is None:
            params = {}
        
        # Add API key if available (optional for free tier)
        if self.api_key:
            params['x_cg_demo_api_key'] = self.api_key
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed: {e}")
            return None
    
    def get_coin_list(self) -> List[Dict]:
        """Get list of all available coins (free endpoint)."""
        self.logger.info("Fetching coin list...")
        data = self._make_request("coins/list")
        return data if data else []
    
    def get_market_data(self, coin_ids: List[str], vs_currency: str = 'usd') -> pd.DataFrame:
        """Get current market data for specified coins."""
        self.logger.info(f"Fetching market data for {len(coin_ids)} coins...")
        
        # CoinGecko allows up to 250 coins per request
        all_data = []
        batch_size = 100  # Conservative batch size
        
        for i in range(0, len(coin_ids), batch_size):
            batch = coin_ids[i:i + batch_size]
            params = {
                'ids': ','.join(batch),
                'vs_currency': vs_currency,
                'include_market_cap': 'true',
                'include_24hr_vol': 'true',
                'include_24hr_change': 'true',
                'include_last_updated_at': 'true'
            }
            
            data = self._make_request("coins/markets", params)
            if data:
                all_data.extend(data)
        
        if not all_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_data)
        df['timestamp'] = datetime.utcnow()
        return df
    
    def get_historical_data(self, coin_id: str, days: int = 30, vs_currency: str = 'usd') -> pd.DataFrame:
        """Get historical price data for a coin."""
        self.logger.info(f"Fetching {days} days of historical data for {coin_id}...")
        
        params = {
            'vs_currency': vs_currency,
            'days': str(days),
            'interval': 'daily' if days > 1 else 'hourly'
        }
        
        data = self._make_request(f"coins/{coin_id}/market_chart", params)
        if not data:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame()
        if 'prices' in data:
            prices = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
            prices['timestamp'] = pd.to_datetime(prices['timestamp'], unit='ms')
            df = prices.set_index('timestamp')
        
        if 'market_caps' in data:
            market_caps = pd.DataFrame(data['market_caps'], columns=['timestamp', 'market_cap'])
            market_caps['timestamp'] = pd.to_datetime(market_caps['timestamp'], unit='ms')
            market_caps = market_caps.set_index('timestamp')
            df = df.join(market_caps, how='outer')
        
        if 'total_volumes' in data:
            volumes = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])
            volumes['timestamp'] = pd.to_datetime(volumes['timestamp'], unit='ms')
            volumes = volumes.set_index('timestamp')
            df = df.join(volumes, how='outer')
        
        df['coin_id'] = coin_id
        return df.reset_index()
    
    def get_trending_coins(self) -> List[Dict]:
        """Get trending coins (free endpoint)."""
        self.logger.info("Fetching trending coins...")
        data = self._make_request("search/trending")
        return data.get('coins', []) if data else []
    
    def get_global_market_data(self) -> Dict:
        """Get global cryptocurrency market data."""
        self.logger.info("Fetching global market data...")
        data = self._make_request("global")
        return data.get('data', {}) if data else {}
    
    def get_fear_greed_index(self) -> Dict:
        """Get Fear & Greed Index (alternative free API)."""
        try:
            response = requests.get("https://api.alternative.me/fng/", timeout=10)
            data = response.json()
            return data.get('data', [{}])[0] if data else {}
        except Exception as e:
            self.logger.error(f"Failed to fetch Fear & Greed Index: {e}")
            return {}
    
    def collect_comprehensive_data(self, coin_ids: List[str]) -> Dict[str, pd.DataFrame]:
        """Collect comprehensive data for analysis."""
        self.logger.info(f"Starting comprehensive data collection for {len(coin_ids)} coins...")
        
        results = {}
        
        # Current market data
        market_data = self.get_market_data(coin_ids)
        if not market_data.empty:
            results['market_data'] = market_data
        
        # Historical data (last 30 days for technical analysis)
        historical_data = []
        for coin_id in coin_ids[:10]:  # Limit to top 10 to stay within rate limits
            hist_data = self.get_historical_data(coin_id, days=30)
            if not hist_data.empty:
                historical_data.append(hist_data)
        
        if historical_data:
            results['historical_data'] = pd.concat(historical_data, ignore_index=True)
        
        # Trending coins
        trending = self.get_trending_coins()
        if trending:
            results['trending_coins'] = pd.DataFrame(trending)
        
        # Global market data
        global_data = self.get_global_market_data()
        if global_data:
            results['global_market'] = pd.DataFrame([global_data])
        
        # Fear & Greed Index
        fg_index = self.get_fear_greed_index()
        if fg_index:
            results['fear_greed_index'] = pd.DataFrame([fg_index])
        
        self.logger.info(f"Data collection completed. Collected {len(results)} datasets.")
        return results
    
    def save_data_to_csv(self, data_dict: Dict[str, pd.DataFrame], output_dir: str = './data/'):
        """Save collected data to CSV files."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for name, df in data_dict.items():
            filename = f"{output_dir}{name}_{timestamp}.csv"
            df.to_csv(filename, index=False)
            self.logger.info(f"Saved {name} data to {filename}")


def main():
    """Example usage of the Free Crypto Data Collector."""
    # Initialize collector (no API key needed for free tier)
    collector = FreeCryptoDataCollector()
    
    # Top 20 cryptocurrencies by market cap
    top_coins = [
        'bitcoin', 'ethereum', 'binancecoin', 'cardano', 'solana',
        'xrp', 'polkadot', 'dogecoin', 'avalanche-2', 'shiba-inu',
        'polygon', 'chainlink', 'litecoin', 'bitcoin-cash', 'algorand',
        'stellar', 'vechain', 'filecoin', 'tron', 'ethereum-classic'
    ]
    
    # Collect comprehensive data
    data = collector.collect_comprehensive_data(top_coins)
    
    # Save to CSV files
    collector.save_data_to_csv(data)
    
    # Display summary
    for name, df in data.items():
        print(f"\n{name.upper()}:")
        print(f"Shape: {df.shape}")
        if not df.empty:
            print(f"Columns: {list(df.columns)}")
            print(df.head(2))


if __name__ == "__main__":
    main()
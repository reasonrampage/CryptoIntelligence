"""
Free Technical Analysis Module
Uses only free, open-source libraries:
- TA-Lib (open source)
- pandas_ta (free)
- Custom implementations for additional indicators
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings

# Free technical analysis libraries
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    warnings.warn("TA-Lib not available. Using pandas_ta and custom implementations.")

try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False
    warnings.warn("pandas_ta not available. Using TA-Lib and custom implementations.")

class FreeTechnicalAnalyzer:
    """Technical analysis using only free, open-source tools."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Indicator parameters based on crypto trading best practices
        self.indicator_params = {
            'sma_periods': [10, 20, 50, 200],
            'ema_periods': [12, 26, 50],
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'bb_period': 20,
            'bb_std': 2,
            'stoch_k': 14,
            'stoch_d': 3,
            'volume_sma': 20,
            'atr_period': 14
        }
    
    def _validate_ohlcv_data(self, df: pd.DataFrame) -> bool:
        """Validate that DataFrame has required OHLCV columns."""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Check for various naming conventions
        df_columns_lower = [col.lower() for col in df.columns]
        
        for req_col in required_columns:
            variations = [req_col, req_col.upper(), req_col.capitalize()]
            if not any(var in df.columns or var.lower() in df_columns_lower for var in variations):
                self.logger.error(f"Missing required column: {req_col}")
                return False
        return True
    
    def _standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to lowercase."""
        df = df.copy()
        
        # Common column name mappings
        column_mapping = {}
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in ['open', 'high', 'low', 'close', 'volume']:
                column_mapping[col] = col_lower
            elif 'price' in col_lower:
                column_mapping[col] = 'close'
        
        df = df.rename(columns=column_mapping)
        return df
    
    def calculate_sma(self, close_prices: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average."""
        if TALIB_AVAILABLE:
            return pd.Series(talib.SMA(close_prices.values, timeperiod=period), index=close_prices.index)
        elif PANDAS_TA_AVAILABLE:
            return ta.sma(close_prices, length=period)
        else:
            # Custom implementation
            return close_prices.rolling(window=period).mean()
    
    def calculate_ema(self, close_prices: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        if TALIB_AVAILABLE:
            return pd.Series(talib.EMA(close_prices.values, timeperiod=period), index=close_prices.index)
        elif PANDAS_TA_AVAILABLE:
            return ta.ema(close_prices, length=period)
        else:
            # Custom implementation
            return close_prices.ewm(span=period).mean()
    
    def calculate_rsi(self, close_prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        if TALIB_AVAILABLE:
            return pd.Series(talib.RSI(close_prices.values, timeperiod=period), index=close_prices.index)
        elif PANDAS_TA_AVAILABLE:
            return ta.rsi(close_prices, length=period)
        else:
            # Custom implementation
            delta = close_prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
    
    def calculate_macd(self, close_prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """Calculate MACD indicators."""
        if TALIB_AVAILABLE:
            macd, signal_line, histogram = talib.MACD(close_prices.values, fastperiod=fast, slowperiod=slow, signalperiod=signal)
            return {
                'macd': pd.Series(macd, index=close_prices.index),
                'signal': pd.Series(signal_line, index=close_prices.index),
                'histogram': pd.Series(histogram, index=close_prices.index)
            }
        elif PANDAS_TA_AVAILABLE:
            macd_data = ta.macd(close_prices, fast=fast, slow=slow, signal=signal)
            return {
                'macd': macd_data[f'MACD_{fast}_{slow}_{signal}'],
                'signal': macd_data[f'MACDs_{fast}_{slow}_{signal}'],
                'histogram': macd_data[f'MACDh_{fast}_{slow}_{signal}']
            }
        else:
            # Custom implementation
            ema_fast = self.calculate_ema(close_prices, fast)
            ema_slow = self.calculate_ema(close_prices, slow)
            macd = ema_fast - ema_slow
            signal_line = self.calculate_ema(macd, signal)
            histogram = macd - signal_line
            
            return {
                'macd': macd,
                'signal': signal_line,
                'histogram': histogram
            }
    
    def calculate_bollinger_bands(self, close_prices: pd.Series, period: int = 20, std_dev: float = 2) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands."""
        if TALIB_AVAILABLE:
            upper, middle, lower = talib.BBANDS(close_prices.values, timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev)
            return {
                'upper': pd.Series(upper, index=close_prices.index),
                'middle': pd.Series(middle, index=close_prices.index),
                'lower': pd.Series(lower, index=close_prices.index)
            }
        elif PANDAS_TA_AVAILABLE:
            bb_data = ta.bbands(close_prices, length=period, std=std_dev)
            return {
                'upper': bb_data[f'BBU_{period}_{std_dev}'],
                'middle': bb_data[f'BBM_{period}_{std_dev}'],
                'lower': bb_data[f'BBL_{period}_{std_dev}']
            }
        else:
            # Custom implementation
            sma = self.calculate_sma(close_prices, period)
            std = close_prices.rolling(window=period).std()
            
            return {
                'upper': sma + (std * std_dev),
                'middle': sma,
                'lower': sma - (std * std_dev)
            }
    
    def calculate_stochastic(self, high_prices: pd.Series, low_prices: pd.Series, 
                           close_prices: pd.Series, k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
        """Calculate Stochastic Oscillator."""
        if TALIB_AVAILABLE:
            k_percent, d_percent = talib.STOCH(high_prices.values, low_prices.values, close_prices.values,
                                            fastk_period=k_period, slowk_period=d_period, slowd_period=d_period)
            return {
                'k_percent': pd.Series(k_percent, index=close_prices.index),
                'd_percent': pd.Series(d_percent, index=close_prices.index)
            }
        elif PANDAS_TA_AVAILABLE:
            stoch_data = ta.stoch(high_prices, low_prices, close_prices, k=k_period, d=d_period)
            return {
                'k_percent': stoch_data[f'STOCHk_{k_period}_{d_period}_{d_period}'],
                'd_percent': stoch_data[f'STOCHd_{k_period}_{d_period}_{d_period}']
            }
        else:
            # Custom implementation
            lowest_low = low_prices.rolling(window=k_period).min()
            highest_high = high_prices.rolling(window=k_period).max()
            
            k_percent = 100 * ((close_prices - lowest_low) / (highest_high - lowest_low))
            d_percent = k_percent.rolling(window=d_period).mean()
            
            return {
                'k_percent': k_percent,
                'd_percent': d_percent
            }
    
    def calculate_atr(self, high_prices: pd.Series, low_prices: pd.Series, 
                     close_prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        if TALIB_AVAILABLE:
            return pd.Series(talib.ATR(high_prices.values, low_prices.values, close_prices.values, timeperiod=period),
                           index=close_prices.index)
        elif PANDAS_TA_AVAILABLE:
            return ta.atr(high_prices, low_prices, close_prices, length=period)
        else:
            # Custom implementation
            high_low = high_prices - low_prices
            high_close_prev = np.abs(high_prices - close_prices.shift(1))
            low_close_prev = np.abs(low_prices - close_prices.shift(1))
            
            true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
            return true_range.rolling(window=period).mean()
    
    def calculate_volume_indicators(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate volume-based indicators."""
        volume = df['volume']
        close = df['close']
        
        indicators = {}
        
        # Volume SMA
        indicators['volume_sma'] = self.calculate_sma(volume, self.indicator_params['volume_sma'])
        
        # Volume Rate of Change
        indicators['volume_roc'] = volume.pct_change(periods=10) * 100
        
        # On-Balance Volume (OBV)
        if TALIB_AVAILABLE:
            indicators['obv'] = pd.Series(talib.OBV(close.values, volume.values), index=close.index)
        else:
            # Custom OBV implementation
            price_change = close.diff()
            volume_direction = np.where(price_change > 0, volume,
                                      np.where(price_change < 0, -volume, 0))
            indicators['obv'] = pd.Series(volume_direction).cumsum()
        
        return indicators
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators for a cryptocurrency."""
        if not self._validate_ohlcv_data(df):
            return pd.DataFrame()
        
        df = self._standardize_column_names(df)
        
        if len(df) < 50:  # Need sufficient data for indicators
            self.logger.warning(f"Insufficient data for technical analysis: {len(df)} rows")
            return df
        
        result_df = df.copy()
        
        # Moving Averages
        for period in self.indicator_params['sma_periods']:
            result_df[f'sma_{period}'] = self.calculate_sma(df['close'], period)
        
        for period in self.indicator_params['ema_periods']:
            result_df[f'ema_{period}'] = self.calculate_ema(df['close'], period)
        
        # RSI
        result_df['rsi'] = self.calculate_rsi(df['close'], self.indicator_params['rsi_period'])
        
        # MACD
        macd_data = self.calculate_macd(df['close'], 
                                      self.indicator_params['macd_fast'],
                                      self.indicator_params['macd_slow'],
                                      self.indicator_params['macd_signal'])
        result_df['macd'] = macd_data['macd']
        result_df['macd_signal'] = macd_data['signal']
        result_df['macd_histogram'] = macd_data['histogram']
        
        # Bollinger Bands
        bb_data = self.calculate_bollinger_bands(df['close'], 
                                               self.indicator_params['bb_period'],
                                               self.indicator_params['bb_std'])
        result_df['bb_upper'] = bb_data['upper']
        result_df['bb_middle'] = bb_data['middle']
        result_df['bb_lower'] = bb_data['lower']
        
        # Bollinger Band Width and Position
        result_df['bb_width'] = (bb_data['upper'] - bb_data['lower']) / bb_data['middle']
        result_df['bb_position'] = (df['close'] - bb_data['lower']) / (bb_data['upper'] - bb_data['lower'])
        
        # Stochastic
        stoch_data = self.calculate_stochastic(df['high'], df['low'], df['close'],
                                             self.indicator_params['stoch_k'],
                                             self.indicator_params['stoch_d'])
        result_df['stoch_k'] = stoch_data['k_percent']
        result_df['stoch_d'] = stoch_data['d_percent']
        
        # ATR
        result_df['atr'] = self.calculate_atr(df['high'], df['low'], df['close'], 
                                            self.indicator_params['atr_period'])
        
        # Volume indicators
        volume_indicators = self.calculate_volume_indicators(df)
        for name, series in volume_indicators.items():
            result_df[name] = series
        
        # Price-based calculations
        result_df['price_change'] = df['close'].pct_change()
        result_df['volatility'] = result_df['price_change'].rolling(window=20).std() * np.sqrt(365) * 100
        
        # Support and Resistance levels (simple implementation)
        result_df['support'] = df['low'].rolling(window=20).min()
        result_df['resistance'] = df['high'].rolling(window=20).max()
        
        # Add timestamp
        result_df['calculation_timestamp'] = datetime.utcnow()
        
        return result_df
    
    def generate_trading_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate simple trading signals based on technical indicators."""
        if df.empty:
            return df
        
        signals_df = df.copy()
        
        # Initialize signal columns
        signals_df['signal_rsi'] = 0  # 1: oversold, -1: overbought
        signals_df['signal_macd'] = 0  # 1: bullish, -1: bearish
        signals_df['signal_bb'] = 0   # 1: oversold, -1: overbought
        signals_df['signal_stoch'] = 0  # 1: oversold, -1: overbought
        
        # RSI signals
        if 'rsi' in signals_df.columns:
            signals_df.loc[signals_df['rsi'] < 30, 'signal_rsi'] = 1  # Oversold
            signals_df.loc[signals_df['rsi'] > 70, 'signal_rsi'] = -1  # Overbought
        
        # MACD signals
        if all(col in signals_df.columns for col in ['macd', 'macd_signal']):
            signals_df.loc[signals_df['macd'] > signals_df['macd_signal'], 'signal_macd'] = 1
            signals_df.loc[signals_df['macd'] < signals_df['macd_signal'], 'signal_macd'] = -1
        
        # Bollinger Bands signals
        if all(col in signals_df.columns for col in ['close', 'bb_upper', 'bb_lower']):
            signals_df.loc[signals_df['close'] < signals_df['bb_lower'], 'signal_bb'] = 1  # Oversold
            signals_df.loc[signals_df['close'] > signals_df['bb_upper'], 'signal_bb'] = -1  # Overbought
        
        # Stochastic signals
        if 'stoch_k' in signals_df.columns:
            signals_df.loc[signals_df['stoch_k'] < 20, 'signal_stoch'] = 1  # Oversold
            signals_df.loc[signals_df['stoch_k'] > 80, 'signal_stoch'] = -1  # Overbought
        
        # Composite signal (majority vote)
        signal_columns = ['signal_rsi', 'signal_macd', 'signal_bb', 'signal_stoch']
        signals_df['composite_signal'] = signals_df[signal_columns].sum(axis=1)
        
        # Normalize composite signal
        signals_df['composite_signal'] = np.clip(signals_df['composite_signal'], -1, 1)
        
        return signals_df
    
    def calculate_technical_score(self, df: pd.DataFrame) -> float:
        """Calculate a technical analysis score (0-100)."""
        if df.empty or len(df) < 2:
            return 50.0  # Neutral score
        
        latest = df.iloc[-1]
        score_components = []
        
        # RSI component (inverted - lower RSI = higher score)
        if 'rsi' in df.columns and not pd.isna(latest['rsi']):
            if latest['rsi'] < 30:
                rsi_score = 80 + (30 - latest['rsi'])  # Oversold bonus
            elif latest['rsi'] > 70:
                rsi_score = 20 - (latest['rsi'] - 70)  # Overbought penalty
            else:
                rsi_score = 50 + ((50 - latest['rsi']) * 0.5)  # Neutral zone
            score_components.append(max(0, min(100, rsi_score)))
        
        # MACD component
        if all(col in df.columns for col in ['macd', 'macd_signal']) and not pd.isna(latest['macd']):
            if latest['macd'] > latest['macd_signal']:
                macd_score = 70
            else:
                macd_score = 30
            score_components.append(macd_score)
        
        # Bollinger Bands component
        if all(col in df.columns for col in ['bb_position']) and not pd.isna(latest['bb_position']):
            if latest['bb_position'] < 0.2:  # Near lower band
                bb_score = 80
            elif latest['bb_position'] > 0.8:  # Near upper band
                bb_score = 20
            else:
                bb_score = 50
            score_components.append(bb_score)
        
        # Price momentum component
        if 'price_change' in df.columns:
            recent_returns = df['price_change'].tail(5).mean()
            if not pd.isna(recent_returns):
                momentum_score = 50 + (recent_returns * 1000)  # Scale and center
                score_components.append(max(0, min(100, momentum_score)))
        
        # Volume component
        if all(col in df.columns for col in ['volume', 'volume_sma']) and not pd.isna(latest['volume']):
            volume_ratio = latest['volume'] / latest['volume_sma']
            if volume_ratio > 1.5:  # High volume
                volume_score = 70
            elif volume_ratio < 0.5:  # Low volume
                volume_score = 30
            else:
                volume_score = 50
            score_components.append(volume_score)
        
        # Calculate final score
        if score_components:
            final_score = np.mean(score_components)
        else:
            final_score = 50.0  # Neutral if no indicators available
        
        return max(0, min(100, final_score))


def main():
    """Example usage of the Free Technical Analyzer."""
    # Generate sample OHLCV data for testing
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    
    # Simulate price data
    np.random.seed(42)
    price = 100
    prices = []
    
    for i in range(100):
        price += np.random.normal(0, 2)
        price = max(50, price)  # Floor price
        
        # Create OHLCV data
        open_price = price
        high_price = price * (1 + abs(np.random.normal(0, 0.02)))
        low_price = price * (1 - abs(np.random.normal(0, 0.02)))
        close_price = open_price + np.random.normal(0, 1)
        volume = np.random.lognormal(10, 0.5)
        
        prices.append({
            'date': dates[i],
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
        price = close_price
    
    sample_data = pd.DataFrame(prices)
    sample_data.set_index('date', inplace=True)
    
    # Initialize analyzer
    analyzer = FreeTechnicalAnalyzer()
    
    # Calculate all indicators
    print("Calculating technical indicators...")
    technical_data = analyzer.calculate_all_indicators(sample_data)
    
    # Generate trading signals
    signals_data = analyzer.generate_trading_signals(technical_data)
    
    # Calculate technical score
    tech_score = analyzer.calculate_technical_score(technical_data)
    
    # Display results
    print(f"\nTechnical Analysis Results:")
    print(f"Data points: {len(technical_data)}")
    print(f"Technical Score: {tech_score:.1f}/100")
    
    print(f"\nLatest indicators:")
    latest = technical_data.iloc[-1]
    
    indicators_to_show = ['close', 'rsi', 'macd', 'bb_position', 'stoch_k', 'volume_ratio']
    for indicator in indicators_to_show:
        if indicator in technical_data.columns:
            value = latest[indicator]
            if not pd.isna(value):
                print(f"{indicator}: {value:.3f}")
    
    print(f"\nLatest signals:")
    latest_signals = signals_data.iloc[-1]
    signal_cols = ['signal_rsi', 'signal_macd', 'signal_bb', 'signal_stoch', 'composite_signal']
    for signal in signal_cols:
        if signal in signals_data.columns:
            print(f"{signal}: {latest_signals[signal]}")


if __name__ == "__main__":
    main()
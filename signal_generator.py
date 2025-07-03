import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from enum import Enum
import logging
from config import TradingConfig

class SignalType(Enum):
    """Enumeration for signal types"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

class CrossoverSignalGenerator:
    """Class to generate golden and death crossover signals"""
    
    def __init__(self):
        self.config = TradingConfig()
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def detect_crossovers(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Detect golden and death crossovers
        
        Args:
            data: DataFrame with EMA values
            
        Returns:
            DataFrame with crossover signals
        """
        df = data.copy()
        
        ema_short_col = f'EMA_{self.config.EMA_SHORT}'
        ema_long_col = f'EMA_{self.config.EMA_LONG}'
        
        # Ensure EMA columns exist
        if ema_short_col not in df.columns or ema_long_col not in df.columns:
            raise ValueError(f"EMA columns not found in data. Required: {ema_short_col}, {ema_long_col}")
        
        # Calculate the difference between EMAs
        df['EMA_Diff'] = df[ema_short_col] - df[ema_long_col]
        df['EMA_Diff_Prev'] = df['EMA_Diff'].shift(1)
        
        # Detect crossovers
        # Golden crossover: short EMA crosses above long EMA (bullish)
        df['Golden_Cross'] = (
            (df['EMA_Diff'] > 0) & 
            (df['EMA_Diff_Prev'] <= 0) & 
            (df[ema_short_col].notna()) & 
            (df[ema_long_col].notna())
        )
        
        # Death crossover: short EMA crosses below long EMA (bearish)
        df['Death_Cross'] = (
            (df['EMA_Diff'] < 0) & 
            (df['EMA_Diff_Prev'] >= 0) & 
            (df[ema_short_col].notna()) & 
            (df[ema_long_col].notna())
        )
        
        return df
    
    def generate_signals(self, data: pd.DataFrame, include_filters: bool = True) -> pd.DataFrame:
        """
        Generate trading signals based on crossovers
        
        Args:
            data: DataFrame with stock data and EMAs
            include_filters: Whether to include additional filters
            
        Returns:
            DataFrame with signals
        """
        df = self.detect_crossovers(data)
        
        # Initialize signal column
        df['Signal'] = SignalType.HOLD.value
        df['Signal_Strength'] = 0.0
        df['Entry_Price'] = np.nan
        df['Stop_Loss'] = np.nan
        df['Take_Profit'] = np.nan
        
        # Apply additional filters if requested
        if include_filters:
            df = self._apply_signal_filters(df)
        
        # Generate basic signals
        buy_condition = df['Golden_Cross']
        sell_condition = df['Death_Cross']
        
        # Apply filters
        if include_filters:
            buy_condition = buy_condition & df['Volume_Filter'] & df['Trend_Filter']
            sell_condition = sell_condition & df['Volume_Filter']
        
        # Set signals
        df.loc[buy_condition, 'Signal'] = SignalType.BUY.value
        df.loc[sell_condition, 'Signal'] = SignalType.SELL.value
        
        # Calculate signal strength (0-1)
        df.loc[buy_condition, 'Signal_Strength'] = self._calculate_signal_strength(df[buy_condition], 'BUY')
        df.loc[sell_condition, 'Signal_Strength'] = self._calculate_signal_strength(df[sell_condition], 'SELL')
        
        # Set entry prices and risk management levels
        df.loc[buy_condition, 'Entry_Price'] = df.loc[buy_condition, 'Close']
        df.loc[sell_condition, 'Entry_Price'] = df.loc[sell_condition, 'Close']
        
        # Calculate stop loss and take profit levels
        df.loc[buy_condition, 'Stop_Loss'] = df.loc[buy_condition, 'Close'] * (1 - self.config.STOP_LOSS_PCT)
        df.loc[buy_condition, 'Take_Profit'] = df.loc[buy_condition, 'Close'] * (1 + self.config.TAKE_PROFIT_PCT)
        
        df.loc[sell_condition, 'Stop_Loss'] = df.loc[sell_condition, 'Close'] * (1 + self.config.STOP_LOSS_PCT)
        df.loc[sell_condition, 'Take_Profit'] = df.loc[sell_condition, 'Close'] * (1 - self.config.TAKE_PROFIT_PCT)
        
        return df
    
    def _apply_signal_filters(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply additional filters to improve signal quality
        
        Args:
            data: DataFrame with basic signals
            
        Returns:
            DataFrame with filters applied
        """
        df = data.copy()
        
        # Volume filter: Above average volume
        df['Volume_Filter'] = df['Volume'] > df['Volume_MA']
        
        # Trend filter: Price above/below longer-term trend
        df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
        df['Trend_Filter'] = True  # Default to True
        
        # For buy signals: price should be above 200 EMA (uptrend)
        if 'EMA_200' in df.columns:
            df['Trend_Filter'] = df['Close'] > df['EMA_200']
        
        # Volatility filter: Avoid extremely volatile periods
        volatility_threshold = df['Volatility'].quantile(0.8)  # Top 20% volatility
        df['Volatility_Filter'] = df['Volatility'] < volatility_threshold
        
        # Momentum filter: Check if EMAs are diverging
        df['EMA_Momentum'] = df['EMA_Diff'] - df['EMA_Diff'].shift(5)
        df['Momentum_Filter'] = df['EMA_Momentum'] > 0  # EMAs diverging in favor of signal
        
        return df
    
    def _calculate_signal_strength(self, data: pd.DataFrame, signal_type: str) -> pd.Series:
        """
        Calculate signal strength based on multiple factors
        
        Args:
            data: DataFrame with signal data
            signal_type: 'BUY' or 'SELL'
            
        Returns:
            Series with signal strength values (0-1)
        """
        strength = pd.Series(0.5, index=data.index)  # Base strength
        
        if len(data) == 0:
            return strength
        
        # Factor 1: EMA divergence magnitude
        ema_diff_abs = abs(data['EMA_Diff'])
        ema_diff_norm = ema_diff_abs / data['Close']
        strength += ema_diff_norm * 0.3
        
        # Factor 2: Volume confirmation
        volume_ratio = data['Volume'] / data['Volume_MA']
        volume_strength = np.clip(volume_ratio / 2, 0, 0.2)
        strength += volume_strength
        
        # Factor 3: Trend alignment
        if 'EMA_200' in data.columns:
            if signal_type == 'BUY':
                trend_align = (data['Close'] > data['EMA_200']).astype(float) * 0.2
            else:
                trend_align = (data['Close'] < data['EMA_200']).astype(float) * 0.2
            strength += trend_align
        
        # Factor 4: Momentum
        if 'EMA_Momentum' in data.columns:
            momentum_strength = np.clip(abs(data['EMA_Momentum']) / data['Close'], 0, 0.1)
            strength += momentum_strength
        
        # Normalize to 0-1 range
        strength = np.clip(strength, 0, 1)
        
        return strength
    
    def get_signal_summary(self, data: pd.DataFrame, symbol: str) -> Dict:
        """
        Get summary of signals for a stock
        
        Args:
            data: DataFrame with signals
            symbol: Stock symbol
            
        Returns:
            Dictionary with signal summary
        """
        signals = data[data['Signal'] != SignalType.HOLD.value].copy()
        
        if len(signals) == 0:
            return {
                'symbol': symbol,
                'total_signals': 0,
                'buy_signals': 0,
                'sell_signals': 0,
                'latest_signal': None,
                'latest_signal_date': None,
                'avg_signal_strength': 0
            }
        
        buy_signals = signals[signals['Signal'] == SignalType.BUY.value]
        sell_signals = signals[signals['Signal'] == SignalType.SELL.value]
        latest_signal = signals.iloc[-1]
        
        return {
            'symbol': symbol,
            'total_signals': len(signals),
            'buy_signals': len(buy_signals),
            'sell_signals': len(sell_signals),
            'latest_signal': latest_signal['Signal'],
            'latest_signal_date': latest_signal.name,
            'latest_signal_strength': latest_signal['Signal_Strength'],
            'avg_signal_strength': signals['Signal_Strength'].mean(),
            'golden_crosses': len(data[data['Golden_Cross']]),
            'death_crosses': len(data[data['Death_Cross']])
        }
    
    def filter_high_quality_signals(self, data: pd.DataFrame, min_strength: float = 0.7) -> pd.DataFrame:
        """
        Filter signals based on minimum strength threshold
        
        Args:
            data: DataFrame with signals
            min_strength: Minimum signal strength threshold
            
        Returns:
            DataFrame with filtered signals
        """
        df = data.copy()
        
        # Keep only high-quality signals
        low_quality_mask = (
            (df['Signal'] != SignalType.HOLD.value) & 
            (df['Signal_Strength'] < min_strength)
        )
        
        df.loc[low_quality_mask, 'Signal'] = SignalType.HOLD.value
        df.loc[low_quality_mask, 'Signal_Strength'] = 0
        
        return df
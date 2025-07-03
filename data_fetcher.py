import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from config import TradingConfig

class DataFetcher:
    """Class to fetch and preprocess Indian stock market data"""
    
    def __init__(self):
        self.config = TradingConfig()
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def fetch_stock_data(self, symbol: str, period: str = None, 
                        start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Fetch stock data from Yahoo Finance
        
        Args:
            symbol: Stock symbol (e.g., 'RELIANCE.NS')
            period: Period for data ('1y', '2y', etc.)
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            DataFrame with stock data
        """
        try:
            symbol = self.config.get_stock_symbol(symbol)
            ticker = yf.Ticker(symbol)
            
            if start_date and end_date:
                data = ticker.history(start=start_date, end=end_date, interval=self.config.DATA_INTERVAL)
            else:
                period = period or self.config.DATA_PERIOD
                data = ticker.history(period=period, interval=self.config.DATA_INTERVAL)
            
            if data.empty:
                self.logger.warning(f"No data found for symbol: {symbol}")
                return pd.DataFrame()
            
            # Clean the data
            data = data.dropna()
            data.index = pd.to_datetime(data.index)
            
            self.logger.info(f"Fetched {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def fetch_multiple_stocks(self, symbols: List[str], period: str = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple stocks
        
        Args:
            symbols: List of stock symbols
            period: Period for data
            
        Returns:
            Dictionary with symbol as key and DataFrame as value
        """
        data_dict = {}
        period = period or self.config.DATA_PERIOD
        
        for symbol in symbols:
            self.logger.info(f"Fetching data for {symbol}")
            data = self.fetch_stock_data(symbol, period=period)
            if not data.empty:
                data_dict[symbol] = data
            else:
                self.logger.warning(f"Skipping {symbol} due to no data")
        
        return data_dict
    
    def calculate_ema(self, data: pd.DataFrame, period: int, column: str = 'Close') -> pd.Series:
        """
        Calculate Exponential Moving Average
        
        Args:
            data: DataFrame with stock data
            period: EMA period
            column: Column to calculate EMA on
            
        Returns:
            Series with EMA values
        """
        return data[column].ewm(span=period, adjust=False).mean()
    
    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the data
        
        Args:
            data: DataFrame with stock data
            
        Returns:
            DataFrame with added technical indicators
        """
        df = data.copy()
        
        # Calculate EMAs
        df[f'EMA_{self.config.EMA_SHORT}'] = self.calculate_ema(df, self.config.EMA_SHORT)
        df[f'EMA_{self.config.EMA_LONG}'] = self.calculate_ema(df, self.config.EMA_LONG)
        
        # Calculate additional indicators
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Price_Change'] = df['Close'].pct_change()
        df['Volatility'] = df['Price_Change'].rolling(window=20).std() * np.sqrt(252)
        
        # Calculate support and resistance levels
        df['High_20'] = df['High'].rolling(window=20).max()
        df['Low_20'] = df['Low'].rolling(window=20).min()
        
        return df
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current price of a stock
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Current price or None if not available
        """
        try:
            symbol = self.config.get_stock_symbol(symbol)
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d", interval="1m")
            
            if not data.empty:
                return float(data['Close'].iloc[-1])
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting current price for {symbol}: {str(e)}")
            return None
    
    def validate_data_quality(self, data: pd.DataFrame, symbol: str) -> bool:
        """
        Validate data quality
        
        Args:
            data: DataFrame to validate
            symbol: Stock symbol
            
        Returns:
            True if data quality is good, False otherwise
        """
        if data.empty:
            self.logger.warning(f"Empty data for {symbol}")
            return False
        
        # Check for minimum data points
        min_required = max(self.config.EMA_SHORT, self.config.EMA_LONG) * 2
        if len(data) < min_required:
            self.logger.warning(f"Insufficient data points for {symbol}: {len(data)} < {min_required}")
            return False
        
        # Check for excessive missing values
        missing_pct = data.isnull().sum().sum() / (len(data) * len(data.columns))
        if missing_pct > 0.1:  # More than 10% missing
            self.logger.warning(f"Too many missing values for {symbol}: {missing_pct:.2%}")
            return False
        
        # Check for zero or negative prices
        if (data['Close'] <= 0).any():
            self.logger.warning(f"Invalid price data for {symbol}")
            return False
        
        return True
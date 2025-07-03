# Trading Algorithm Configuration
import os
from typing import Dict, List

class TradingConfig:
    """Configuration class for the EMA crossover trading algorithm"""
    
    # EMA Parameters
    EMA_SHORT = 20  # Short period EMA
    EMA_LONG = 50   # Long period EMA
    
    # Trading Parameters
    INITIAL_CAPITAL = 100000  # Starting capital in INR
    RISK_PER_TRADE = 0.02     # Risk 2% per trade
    STOP_LOSS_PCT = 0.05      # 5% stop loss
    TAKE_PROFIT_PCT = 0.10    # 10% take profit
    
    # Data Parameters
    DATA_PERIOD = "2y"        # 2 years of historical data
    DATA_INTERVAL = "1d"      # Daily data
    
    # Indian Stock Symbols (NSE)
    DEFAULT_STOCKS = [
        "RELIANCE.NS",   # Reliance Industries
        "TCS.NS",        # Tata Consultancy Services
        "HDFCBANK.NS",   # HDFC Bank
        "INFY.NS",       # Infosys
        "HINDUNILVR.NS", # Hindustan Unilever
        "ICICIBANK.NS",  # ICICI Bank
        "KOTAKBANK.NS",  # Kotak Mahindra Bank
        "LT.NS",         # Larsen & Toubro
        "SBIN.NS",       # State Bank of India
        "BHARTIARTL.NS", # Bharti Airtel
        "ASIANPAINT.NS", # Asian Paints
        "MARUTI.NS",     # Maruti Suzuki
        "TITAN.NS",      # Titan Company
        "NESTLEIND.NS",  # Nestle India
        "WIPRO.NS"       # Wipro
    ]
    
    # Nifty 50 Index for benchmark
    BENCHMARK = "^NSEI"
    
    # File paths
    DATA_DIR = "data"
    RESULTS_DIR = "results"
    LOGS_DIR = "logs"
    
    @classmethod
    def get_stock_symbol(cls, symbol: str) -> str:
        """Convert stock symbol to NSE format if needed"""
        if not symbol.endswith('.NS'):
            return f"{symbol}.NS"
        return symbol
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        for directory in [cls.DATA_DIR, cls.RESULTS_DIR, cls.LOGS_DIR]:
            os.makedirs(directory, exist_ok=True)
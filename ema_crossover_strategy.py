import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from config import TradingConfig
from data_fetcher import DataFetcher
from signal_generator import CrossoverSignalGenerator, SignalType
from backtester import Backtester
from visualizer import TradingVisualizer

class EMACrossoverStrategy:
    """
    Complete EMA Crossover Trading Strategy for Indian Stock Market
    
    This class implements a golden and death crossover strategy based on
    EMA 20 and EMA 50 for Indian stocks listed on NSE.
    """
    
    def __init__(self, initial_capital: float = None, enable_logging: bool = True):
        """
        Initialize the trading strategy
        
        Args:
            initial_capital: Starting capital for backtesting
            enable_logging: Whether to enable detailed logging
        """
        self.config = TradingConfig()
        self.config.create_directories()
        
        # Initialize components
        self.data_fetcher = DataFetcher()
        self.signal_generator = CrossoverSignalGenerator()
        self.backtester = Backtester(initial_capital)
        self.visualizer = TradingVisualizer()
        
        # Setup logging
        if enable_logging:
            self._setup_logging()
        
        # Storage for results
        self.stock_data: Dict[str, pd.DataFrame] = {}
        self.signals_data: Dict[str, pd.DataFrame] = {}
        self.backtest_results: Dict[str, Dict] = {}
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("EMA Crossover Strategy initialized")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_file = os.path.join(self.config.LOGS_DIR, 
                               f"trading_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def fetch_data(self, symbols: List[str] = None, period: str = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch stock data for analysis
        
        Args:
            symbols: List of stock symbols (default: config default stocks)
            period: Data period (default: config period)
            
        Returns:
            Dictionary with stock data
        """
        symbols = symbols or self.config.DEFAULT_STOCKS
        period = period or self.config.DATA_PERIOD
        
        self.logger.info(f"Fetching data for {len(symbols)} stocks over {period}")
        
        # Fetch raw data
        raw_data = self.data_fetcher.fetch_multiple_stocks(symbols, period)
        
        # Add technical indicators and validate data quality
        for symbol, data in raw_data.items():
            if self.data_fetcher.validate_data_quality(data, symbol):
                # Add technical indicators
                processed_data = self.data_fetcher.add_technical_indicators(data)
                self.stock_data[symbol] = processed_data
                self.logger.info(f"Successfully processed data for {symbol}: {len(processed_data)} records")
            else:
                self.logger.warning(f"Skipping {symbol} due to poor data quality")
        
        return self.stock_data
    
    def generate_signals(self, include_filters: bool = True, 
                        min_signal_strength: float = 0.6) -> Dict[str, pd.DataFrame]:
        """
        Generate trading signals for all stocks
        
        Args:
            include_filters: Whether to apply additional signal filters
            min_signal_strength: Minimum signal strength threshold
            
        Returns:
            Dictionary with signal data
        """
        self.logger.info("Generating trading signals")
        
        for symbol, data in self.stock_data.items():
            try:
                # Generate signals
                signals_df = self.signal_generator.generate_signals(data, include_filters)
                
                # Filter by signal strength
                if min_signal_strength > 0:
                    signals_df = self.signal_generator.filter_high_quality_signals(
                        signals_df, min_signal_strength)
                
                self.signals_data[symbol] = signals_df
                
                # Log signal summary
                summary = self.signal_generator.get_signal_summary(signals_df, symbol)
                self.logger.info(f"Signals for {symbol}: {summary['total_signals']} total, "
                               f"{summary['buy_signals']} buy, {summary['sell_signals']} sell")
                
            except Exception as e:
                self.logger.error(f"Error generating signals for {symbol}: {str(e)}")
        
        return self.signals_data
    
    def run_backtest(self, symbols: List[str] = None) -> Dict[str, Dict]:
        """
        Run backtest for selected stocks
        
        Args:
            symbols: List of symbols to backtest (default: all available)
            
        Returns:
            Dictionary with backtest results
        """
        symbols = symbols or list(self.signals_data.keys())
        
        self.logger.info(f"Running backtest for {len(symbols)} stocks")
        
        for symbol in symbols:
            if symbol not in self.signals_data:
                self.logger.warning(f"No signal data available for {symbol}")
                continue
            
            try:
                # Reset backtester for each stock
                self.backtester.reset()
                
                # Run backtest
                results = self.backtester.run_backtest(self.signals_data[symbol], symbol)
                self.backtest_results[symbol] = results
                
                # Log key results
                self.logger.info(f"Backtest completed for {symbol}: "
                               f"Return: {results['total_return_pct']:.1f}%, "
                               f"Trades: {results['total_trades']}, "
                               f"Win Rate: {results['win_rate']:.1f}%")
                
            except Exception as e:
                self.logger.error(f"Error in backtest for {symbol}: {str(e)}")
        
        return self.backtest_results
    
    def analyze_single_stock(self, symbol: str, period: str = None, 
                           show_plots: bool = True, save_plots: bool = False) -> Dict:
        """
        Complete analysis for a single stock
        
        Args:
            symbol: Stock symbol
            period: Data period
            show_plots: Whether to display plots
            save_plots: Whether to save plots
            
        Returns:
            Dictionary with complete analysis results
        """
        self.logger.info(f"Starting complete analysis for {symbol}")
        
        # Fetch and process data
        data_dict = self.fetch_data([symbol], period)
        if symbol not in data_dict:
            self.logger.error(f"Failed to fetch data for {symbol}")
            return {}
        
        # Generate signals
        signals_dict = self.generate_signals()
        if symbol not in signals_dict:
            self.logger.error(f"Failed to generate signals for {symbol}")
            return {}
        
        # Run backtest
        backtest_dict = self.run_backtest([symbol])
        if symbol not in backtest_dict:
            self.logger.error(f"Failed to run backtest for {symbol}")
            return {}
        
        # Create visualizations
        if show_plots or save_plots:
            self._create_stock_visualizations(symbol, show_plots, save_plots)
        
        # Compile complete results
        results = {
            'symbol': symbol,
            'data_points': len(data_dict[symbol]),
            'date_range': (data_dict[symbol].index[0], data_dict[symbol].index[-1]),
            'signal_summary': self.signal_generator.get_signal_summary(signals_dict[symbol], symbol),
            'backtest_results': backtest_dict[symbol],
            'current_signal': self._get_current_signal(symbol),
            'risk_metrics': self._calculate_additional_metrics(symbol)
        }
        
        return results
    
    def scan_all_stocks(self, show_summary: bool = True) -> pd.DataFrame:
        """
        Scan all configured stocks and return summary
        
        Args:
            show_summary: Whether to display summary dashboard
            
        Returns:
            DataFrame with scan results
        """
        self.logger.info("Scanning all configured stocks")
        
        # Fetch data for all stocks
        self.fetch_data()
        
        # Generate signals
        self.generate_signals()
        
        # Run backtests
        self.run_backtest()
        
        # Create summary DataFrame
        summary_data = []
        for symbol in self.backtest_results.keys():
            results = self.backtest_results[symbol]
            signal_summary = self.signal_generator.get_signal_summary(
                self.signals_data[symbol], symbol)
            
            current_signal = self._get_current_signal(symbol)
            
            summary_data.append({
                'Symbol': symbol.replace('.NS', ''),
                'Current_Signal': current_signal['signal'],
                'Signal_Strength': current_signal['strength'],
                'Signal_Date': current_signal['date'],
                'Total_Return_%': results['total_return_pct'],
                'Total_Trades': results['total_trades'],
                'Win_Rate_%': results['win_rate'],
                'Profit_Factor': results['profit_factor'],
                'Max_Drawdown_%': results['max_drawdown'],
                'Sharpe_Ratio': results['sharpe_ratio'],
                'Golden_Crosses': signal_summary['golden_crosses'],
                'Death_Crosses': signal_summary['death_crosses']
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Sort by total return
        summary_df = summary_df.sort_values('Total_Return_%', ascending=False)
        
        # Display summary dashboard
        if show_summary and len(self.backtest_results) > 1:
            self.visualizer.create_summary_dashboard(self.backtest_results)
        
        return summary_df
    
    def get_current_signals(self, min_strength: float = 0.7) -> List[Dict]:
        """
        Get current trading signals for all stocks
        
        Args:
            min_strength: Minimum signal strength filter
            
        Returns:
            List of current signals
        """
        current_signals = []
        
        for symbol in self.signals_data.keys():
            signal_info = self._get_current_signal(symbol)
            
            if (signal_info['signal'] != SignalType.HOLD.value and 
                signal_info['strength'] >= min_strength):
                
                current_signals.append({
                    'symbol': symbol,
                    'signal': signal_info['signal'],
                    'strength': signal_info['strength'],
                    'date': signal_info['date'],
                    'price': signal_info['price'],
                    'stop_loss': signal_info['stop_loss'],
                    'take_profit': signal_info['take_profit']
                })
        
        # Sort by signal strength
        current_signals.sort(key=lambda x: x['strength'], reverse=True)
        
        return current_signals
    
    def _get_current_signal(self, symbol: str) -> Dict:
        """Get current signal for a stock"""
        if symbol not in self.signals_data:
            return {'signal': SignalType.HOLD.value, 'strength': 0, 'date': None, 
                   'price': None, 'stop_loss': None, 'take_profit': None}
        
        data = self.signals_data[symbol]
        latest = data.iloc[-1]
        
        return {
            'signal': latest['Signal'],
            'strength': latest['Signal_Strength'],
            'date': latest.name,
            'price': latest['Close'],
            'stop_loss': latest.get('Stop_Loss'),
            'take_profit': latest.get('Take_Profit')
        }
    
    def _create_stock_visualizations(self, symbol: str, show_plots: bool, save_plots: bool):
        """Create visualizations for a stock"""
        try:
            data = self.signals_data[symbol]
            
            # Base paths for saving
            base_path = os.path.join(self.config.RESULTS_DIR, symbol.replace('.NS', ''))
            
            # Price and signals chart
            save_path = f"{base_path}_signals.png" if save_plots else None
            self.visualizer.plot_price_and_signals(data, symbol, save_path, show_plots)
            
            # Signal analysis
            save_path = f"{base_path}_analysis.png" if save_plots else None
            self.visualizer.plot_signal_analysis(data, symbol, save_path, show_plots)
            
            # Backtest results
            if symbol in self.backtest_results:
                results = self.backtest_results[symbol]
                equity_curve = self.backtester.equity_curve if hasattr(self.backtester, 'equity_curve') else pd.DataFrame()
                save_path = f"{base_path}_backtest.png" if save_plots else None
                self.visualizer.plot_backtest_results(results, equity_curve, symbol, save_path, show_plots)
                
        except Exception as e:
            self.logger.error(f"Error creating visualizations for {symbol}: {str(e)}")
    
    def _calculate_additional_metrics(self, symbol: str) -> Dict:
        """Calculate additional risk metrics"""
        if symbol not in self.signals_data:
            return {}
        
        data = self.signals_data[symbol]
        
        # Current price relative to EMAs
        current_price = data['Close'].iloc[-1]
        ema_short = data[f'EMA_{self.config.EMA_SHORT}'].iloc[-1]
        ema_long = data[f'EMA_{self.config.EMA_LONG}'].iloc[-1]
        
        return {
            'current_price': current_price,
            'ema_20': ema_short,
            'ema_50': ema_long,
            'price_vs_ema20': ((current_price / ema_short) - 1) * 100,
            'price_vs_ema50': ((current_price / ema_long) - 1) * 100,
            'ema_trend': 'Bullish' if ema_short > ema_long else 'Bearish',
            'current_volatility': data['Volatility'].iloc[-1] if 'Volatility' in data.columns else None
        }
    
    def save_results(self, filename: str = None) -> str:
        """
        Save all results to files
        
        Args:
            filename: Base filename (default: auto-generated)
            
        Returns:
            Path to saved file
        """
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"ema_crossover_results_{timestamp}"
        
        # Save summary as CSV
        if self.backtest_results:
            summary_df = self.scan_all_stocks(show_summary=False)
            csv_path = os.path.join(self.config.RESULTS_DIR, f"{filename}.csv")
            summary_df.to_csv(csv_path, index=False)
            self.logger.info(f"Results saved to {csv_path}")
            return csv_path
        
        return ""
    
    def get_portfolio_recommendation(self, max_positions: int = 5, 
                                   min_signal_strength: float = 0.8) -> Dict:
        """
        Get portfolio recommendations based on current signals
        
        Args:
            max_positions: Maximum number of positions
            min_signal_strength: Minimum signal strength
            
        Returns:
            Dictionary with portfolio recommendations
        """
        current_signals = self.get_current_signals(min_signal_strength)
        
        # Filter and rank signals
        buy_signals = [s for s in current_signals if s['signal'] == SignalType.BUY.value]
        sell_signals = [s for s in current_signals if s['signal'] == SignalType.SELL.value]
        
        # Limit to max positions
        recommended_buys = buy_signals[:max_positions]
        recommended_sells = sell_signals[:max_positions]
        
        # Calculate position sizes
        capital_per_position = self.config.INITIAL_CAPITAL / max_positions if recommended_buys else 0
        
        portfolio = {
            'timestamp': datetime.now(),
            'recommended_buys': recommended_buys,
            'recommended_sells': recommended_sells,
            'capital_per_position': capital_per_position,
            'total_recommendations': len(recommended_buys) + len(recommended_sells),
            'portfolio_allocation': {
                signal['symbol']: capital_per_position / signal['price'] 
                for signal in recommended_buys
            }
        }
        
        return portfolio
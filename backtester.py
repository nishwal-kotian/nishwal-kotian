import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import logging
from config import TradingConfig
from signal_generator import SignalType

@dataclass
class Trade:
    """Class to represent a single trade"""
    symbol: str
    entry_date: datetime
    entry_price: float
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    quantity: int = 0
    trade_type: str = "LONG"  # LONG or SHORT
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    exit_reason: str = ""
    pnl: float = 0.0
    pnl_pct: float = 0.0

class Backtester:
    """Class to backtest the EMA crossover strategy"""
    
    def __init__(self, initial_capital: float = None):
        self.config = TradingConfig()
        self.initial_capital = initial_capital or self.config.INITIAL_CAPITAL
        self.current_capital = self.initial_capital
        self.trades: List[Trade] = []
        self.positions: Dict[str, Trade] = {}
        self.portfolio_value = []
        self.equity_curve = pd.DataFrame()
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def calculate_position_size(self, price: float, stop_loss: float) -> int:
        """
        Calculate position size based on risk management
        
        Args:
            price: Entry price
            stop_loss: Stop loss price
            
        Returns:
            Number of shares to buy
        """
        if stop_loss <= 0 or price <= 0:
            return 0
        
        risk_amount = self.current_capital * self.config.RISK_PER_TRADE
        risk_per_share = abs(price - stop_loss)
        
        if risk_per_share <= 0:
            return 0
        
        position_size = int(risk_amount / risk_per_share)
        
        # Ensure we don't exceed available capital
        max_shares = int(self.current_capital * 0.95 / price)  # Use 95% of capital max
        position_size = min(position_size, max_shares)
        
        return max(position_size, 0)
    
    def enter_position(self, symbol: str, date: datetime, price: float, 
                      signal_type: str, stop_loss: float, take_profit: float) -> bool:
        """
        Enter a new position
        
        Args:
            symbol: Stock symbol
            date: Entry date
            price: Entry price
            signal_type: BUY or SELL
            stop_loss: Stop loss price
            take_profit: Take profit price
            
        Returns:
            True if position entered successfully
        """
        # Check if already in position
        if symbol in self.positions:
            return False
        
        trade_type = "LONG" if signal_type == SignalType.BUY.value else "SHORT"
        
        # Calculate position size
        quantity = self.calculate_position_size(price, stop_loss)
        
        if quantity <= 0:
            self.logger.warning(f"Cannot enter position for {symbol}: insufficient capital or invalid parameters")
            return False
        
        # Calculate required capital
        required_capital = quantity * price
        
        if required_capital > self.current_capital:
            self.logger.warning(f"Insufficient capital for {symbol}: required {required_capital}, available {self.current_capital}")
            return False
        
        # Create trade
        trade = Trade(
            symbol=symbol,
            entry_date=date,
            entry_price=price,
            quantity=quantity,
            trade_type=trade_type,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        # Update capital
        self.current_capital -= required_capital
        self.positions[symbol] = trade
        
        self.logger.info(f"Entered {trade_type} position: {symbol} @ {price}, quantity: {quantity}")
        return True
    
    def exit_position(self, symbol: str, date: datetime, price: float, reason: str = "") -> bool:
        """
        Exit an existing position
        
        Args:
            symbol: Stock symbol
            date: Exit date
            price: Exit price
            reason: Reason for exit
            
        Returns:
            True if position exited successfully
        """
        if symbol not in self.positions:
            return False
        
        trade = self.positions[symbol]
        trade.exit_date = date
        trade.exit_price = price
        trade.exit_reason = reason
        
        # Calculate P&L
        if trade.trade_type == "LONG":
            trade.pnl = (price - trade.entry_price) * trade.quantity
        else:  # SHORT
            trade.pnl = (trade.entry_price - price) * trade.quantity
        
        trade.pnl_pct = trade.pnl / (trade.entry_price * trade.quantity) * 100
        
        # Update capital
        if trade.trade_type == "LONG":
            self.current_capital += price * trade.quantity
        else:  # SHORT
            self.current_capital += (2 * trade.entry_price - price) * trade.quantity
        
        # Store completed trade
        self.trades.append(trade)
        del self.positions[symbol]
        
        self.logger.info(f"Exited position: {symbol} @ {price}, P&L: {trade.pnl:.2f} ({trade.pnl_pct:.2f}%)")
        return True
    
    def check_exit_conditions(self, symbol: str, current_data: pd.Series) -> Tuple[bool, str]:
        """
        Check if exit conditions are met for a position
        
        Args:
            symbol: Stock symbol
            current_data: Current market data
            
        Returns:
            Tuple of (should_exit, reason)
        """
        if symbol not in self.positions:
            return False, ""
        
        trade = self.positions[symbol]
        current_price = current_data['Close']
        
        # Check stop loss
        if trade.trade_type == "LONG":
            if current_price <= trade.stop_loss:
                return True, "Stop Loss"
            if current_price >= trade.take_profit:
                return True, "Take Profit"
        else:  # SHORT
            if current_price >= trade.stop_loss:
                return True, "Stop Loss"
            if current_price <= trade.take_profit:
                return True, "Take Profit"
        
        # Check signal reversal
        if current_data['Signal'] != SignalType.HOLD.value:
            if trade.trade_type == "LONG" and current_data['Signal'] == SignalType.SELL.value:
                return True, "Signal Reversal"
            elif trade.trade_type == "SHORT" and current_data['Signal'] == SignalType.BUY.value:
                return True, "Signal Reversal"
        
        return False, ""
    
    def run_backtest(self, data: pd.DataFrame, symbol: str) -> Dict:
        """
        Run backtest on a single stock
        
        Args:
            data: DataFrame with signals and market data
            symbol: Stock symbol
            
        Returns:
            Dictionary with backtest results
        """
        self.logger.info(f"Running backtest for {symbol}")
        
        portfolio_values = []
        
        for date, row in data.iterrows():
            # Check exit conditions for existing positions
            should_exit, exit_reason = self.check_exit_conditions(symbol, row)
            if should_exit:
                self.exit_position(symbol, date, row['Close'], exit_reason)
            
            # Check entry conditions
            if row['Signal'] in [SignalType.BUY.value, SignalType.SELL.value]:
                if symbol not in self.positions:  # Not already in position
                    self.enter_position(
                        symbol=symbol,
                        date=date,
                        price=row['Entry_Price'],
                        signal_type=row['Signal'],
                        stop_loss=row['Stop_Loss'],
                        take_profit=row['Take_Profit']
                    )
            
            # Calculate current portfolio value
            portfolio_value = self.current_capital
            for pos_symbol, trade in self.positions.items():
                if trade.trade_type == "LONG":
                    portfolio_value += trade.quantity * row['Close']
                else:  # SHORT
                    portfolio_value += trade.quantity * (2 * trade.entry_price - row['Close'])
            
            portfolio_values.append({
                'Date': date,
                'Portfolio_Value': portfolio_value,
                'Cash': self.current_capital,
                'Positions': len(self.positions)
            })
        
        # Close any remaining positions
        if self.positions:
            last_date = data.index[-1]
            last_price = data['Close'].iloc[-1]
            for symbol in list(self.positions.keys()):
                self.exit_position(symbol, last_date, last_price, "End of backtest")
        
        # Create equity curve
        self.equity_curve = pd.DataFrame(portfolio_values).set_index('Date')
        
        return self.calculate_performance_metrics()
    
    def calculate_performance_metrics(self) -> Dict:
        """
        Calculate comprehensive performance metrics
        
        Returns:
            Dictionary with performance metrics
        """
        if len(self.trades) == 0:
            return {
                'total_trades': 0,
                'total_return': 0,
                'total_return_pct': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'calmar_ratio': 0
            }
        
        trades_df = pd.DataFrame([{
            'symbol': t.symbol,
            'entry_date': t.entry_date,
            'exit_date': t.exit_date,
            'entry_price': t.entry_price,
            'exit_price': t.exit_price,
            'quantity': t.quantity,
            'trade_type': t.trade_type,
            'pnl': t.pnl,
            'pnl_pct': t.pnl_pct,
            'exit_reason': t.exit_reason
        } for t in self.trades])
        
        # Basic metrics
        total_trades = len(self.trades)
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] < 0]
        
        total_return = sum(t.pnl for t in self.trades)
        total_return_pct = (self.current_capital - self.initial_capital) / self.initial_capital * 100
        
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
        
        # Profit factor
        gross_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
        gross_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Risk metrics
        if not self.equity_curve.empty:
            returns = self.equity_curve['Portfolio_Value'].pct_change().dropna()
            
            # Maximum drawdown
            rolling_max = self.equity_curve['Portfolio_Value'].expanding().max()
            drawdown = (self.equity_curve['Portfolio_Value'] - rolling_max) / rolling_max
            max_drawdown = abs(drawdown.min()) * 100
            
            # Sharpe ratio (assuming 252 trading days)
            if len(returns) > 1:
                sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
            else:
                sharpe_ratio = 0
            
            # Calmar ratio
            if max_drawdown > 0:
                annual_return = total_return_pct
                calmar_ratio = annual_return / max_drawdown
            else:
                calmar_ratio = 0
        else:
            max_drawdown = 0
            sharpe_ratio = 0
            calmar_ratio = 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'calmar_ratio': calmar_ratio,
            'final_capital': self.current_capital,
            'trades_df': trades_df
        }
    
    def reset(self):
        """Reset backtester for new run"""
        self.current_capital = self.initial_capital
        self.trades = []
        self.positions = {}
        self.portfolio_value = []
        self.equity_curve = pd.DataFrame()
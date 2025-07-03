# EMA Crossover Trading Algorithm - Project Summary

## 🎯 What Was Built

A **complete algorithmic trading system** for the Indian stock market that captures **Golden and Death crossovers** using EMA 20 and EMA 50. This is a professional-grade solution with multiple components working together.

## 📦 Core Components Created

### 1. **Configuration System** (`config.py`)
- Centralized trading parameters (EMA periods, risk management, etc.)
- Pre-configured list of 15 major NSE stocks
- Flexible settings for capital, stop-loss, take-profit

### 2. **Data Management** (`data_fetcher.py`)
- Real-time data fetching from Yahoo Finance
- Support for Indian NSE stocks (`.NS` symbols)
- Data validation and quality checks
- Technical indicators calculation (EMAs, volume MA, volatility)

### 3. **Signal Generation** (`signal_generator.py`)
- **Golden Cross Detection**: EMA 20 crosses above EMA 50 (Buy signal)
- **Death Cross Detection**: EMA 20 crosses below EMA 50 (Sell signal)
- Advanced signal filtering (volume, trend, volatility, momentum)
- Signal strength scoring (0-1 scale)

### 4. **Backtesting Engine** (`backtester.py`)
- Complete portfolio simulation
- Risk-based position sizing (2% risk per trade)
- Automatic stop-loss and take-profit execution
- Comprehensive performance metrics (Sharpe ratio, max drawdown, etc.)

### 5. **Visualization Suite** (`visualizer.py`)
- Price charts with EMA lines and signals
- Interactive Plotly charts
- Backtest performance analysis
- Multi-stock comparison dashboards
- Signal analysis plots

### 6. **Main Strategy Class** (`ema_crossover_strategy.py`)
- Unified interface for all operations
- Single-stock and multi-stock analysis
- Real-time signal scanning
- Portfolio recommendation system

### 7. **Demo & Testing** (`demo.py`, `test_strategy.py`)
- Complete demonstration script
- Component testing suite
- Usage examples

## 🚀 Key Features

### Trading Strategy
- **EMA 20/50 Crossover** signals
- **Risk Management**: 2% risk per trade, 5% stop-loss, 10% take-profit
- **Signal Quality Filters**: Volume confirmation, trend alignment, volatility filters
- **Position Sizing**: Automatic calculation based on risk parameters

### Indian Market Focus
- **Pre-configured NSE stocks**: RELIANCE, TCS, HDFC BANK, INFY, etc.
- **INR currency** display
- **Market-specific** data handling

### Performance Analysis
- **Comprehensive Metrics**: Total return, win rate, profit factor, Sharpe ratio
- **Visual Analysis**: Equity curves, trade distribution, monthly returns
- **Risk Assessment**: Maximum drawdown, volatility analysis

### Real-time Capabilities
- **Current Signal Detection**: Scan all stocks for active signals
- **Portfolio Recommendations**: Automated stock selection
- **Live Data Integration**: Real-time price and signal updates

## 📈 Usage Examples

### Quick Start
```python
from ema_crossover_strategy import EMACrossoverStrategy

# Initialize strategy
strategy = EMACrossoverStrategy(initial_capital=100000)

# Analyze single stock
results = strategy.analyze_single_stock("RELIANCE.NS", show_plots=True)

# Scan all stocks
summary = strategy.scan_all_stocks()

# Get current signals
signals = strategy.get_current_signals(min_strength=0.8)
```

### Running the Demo
```bash
python3 demo.py
```

## 📊 Performance Metrics Calculated

- **Total Return** (absolute and percentage)
- **Win Rate** (percentage of profitable trades)
- **Profit Factor** (gross profit / gross loss)
- **Maximum Drawdown** (largest peak-to-trough decline)
- **Sharpe Ratio** (risk-adjusted return)
- **Calmar Ratio** (annual return / max drawdown)

## 🎨 Visualizations Generated

1. **Price & Signals Chart**: Price with EMA lines and trade signals
2. **Signal Analysis**: EMA differences, volatility, signal strength
3. **Backtest Results**: Equity curve, trade distribution, monthly returns
4. **Portfolio Dashboard**: Multi-stock performance comparison

## 🔧 Configuration Options

All parameters are easily configurable in `config.py`:
- EMA periods (default: 20, 50)
- Risk per trade (default: 2%)
- Stop loss (default: 5%)
- Take profit (default: 10%)
- Initial capital (default: ₹100,000)
- Stock universe (15 major NSE stocks)

## 📁 File Structure
```
├── config.py              # Trading configuration
├── data_fetcher.py         # Data fetching & indicators
├── signal_generator.py     # Golden/death cross detection
├── backtester.py          # Portfolio simulation
├── visualizer.py          # Charts & analysis
├── ema_crossover_strategy.py # Main strategy class
├── demo.py                # Usage examples
├── test_strategy.py       # Component tests
├── requirements.txt       # Dependencies
├── README.md              # Detailed documentation
├── data/                  # Downloaded data storage
├── results/              # Analysis outputs
└── logs/                 # Execution logs
```

## 🎯 Trading Logic Summary

1. **Data Collection**: Fetch historical price data for NSE stocks
2. **Indicator Calculation**: Compute EMA 20, EMA 50, volume MA, volatility
3. **Signal Detection**: Identify golden/death crossovers
4. **Signal Filtering**: Apply volume, trend, and volatility filters
5. **Signal Scoring**: Rate signal quality (0-1 scale)
6. **Position Management**: Calculate position sizes based on risk
7. **Risk Management**: Implement stop-loss and take-profit levels
8. **Performance Tracking**: Monitor and analyze results

## ⚠️ Important Notes

- **Educational Purpose**: This algorithm is for learning and research
- **Risk Warning**: Past performance doesn't guarantee future results
- **Data Source**: Uses Yahoo Finance (subject to delays)
- **Market Hours**: Some features depend on market availability

## 🚀 Ready to Use!

The algorithm is complete and ready for:
- **Backtesting** historical performance
- **Paper trading** with virtual capital
- **Signal generation** for current market conditions
- **Educational research** into crossover strategies

**Run `python3 demo.py` to see it in action!**
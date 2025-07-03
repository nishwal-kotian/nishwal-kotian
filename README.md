# EMA Crossover Trading Algorithm for Indian Stock Market

A comprehensive algorithmic trading system that implements golden and death crossover strategies using Exponential Moving Averages (EMA 20 and EMA 50) specifically designed for the Indian stock market (NSE).

## 🎯 Overview

This trading algorithm captures **Golden Crossovers** (bullish signals) and **Death Crossovers** (bearish signals) using EMA 20 and EMA 50 for Indian stocks. It includes data fetching, signal generation, backtesting, visualization, and portfolio management capabilities.

### Key Features

- **� Golden & Death Crossover Detection**: Automatically identifies EMA 20/50 crossovers
- **🇮🇳 Indian Market Focus**: Pre-configured for NSE stocks with `.NS` symbols
- **🔍 Advanced Signal Filtering**: Volume, trend, and volatility filters for signal quality
- **📊 Comprehensive Backtesting**: Full portfolio simulation with risk management
- **📈 Rich Visualizations**: Interactive charts and performance dashboards
- **⚡ Real-time Scanning**: Current signal detection across multiple stocks
- **💼 Portfolio Management**: Automated position sizing and risk management
- **� Results Export**: Save analysis results and charts

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd ema-crossover-trading
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the demo:**
```bash
python demo.py
```

### Basic Usage

```python
from ema_crossover_strategy import EMACrossoverStrategy

# Initialize the strategy
strategy = EMACrossoverStrategy(initial_capital=100000)

# Analyze a single stock
results = strategy.analyze_single_stock("RELIANCE.NS", period="1y", show_plots=True)

# Scan multiple stocks
summary = strategy.scan_all_stocks()

# Get current trading signals
signals = strategy.get_current_signals(min_strength=0.7)
```

## 📋 Strategy Details

### EMA Crossover Logic

- **Golden Cross (Buy Signal)**: EMA 20 crosses above EMA 50
- **Death Cross (Sell Signal)**: EMA 20 crosses below EMA 50
- **Signal Strength**: 0-1 scale based on multiple factors

### Signal Filters

1. **Volume Filter**: Signal must occur on above-average volume
2. **Trend Filter**: Buy signals only in uptrend (price > EMA 200)
3. **Volatility Filter**: Avoid extremely volatile periods
4. **Momentum Filter**: Check for EMA divergence strength

### Risk Management

- **Position Sizing**: 2% risk per trade (configurable)
- **Stop Loss**: 5% (configurable)
- **Take Profit**: 10% (configurable)
- **Maximum Positions**: Configurable portfolio limits

## 📁 Project Structure

```
├── config.py              # Trading parameters and configuration
├── data_fetcher.py         # Data fetching and preprocessing
├── signal_generator.py     # Signal generation and filtering
├── backtester.py          # Backtesting engine
├── visualizer.py          # Charts and visualization
├── ema_crossover_strategy.py # Main strategy class
├── demo.py                # Demo and examples
├── requirements.txt       # Python dependencies
├── data/                  # Downloaded data storage
├── results/              # Analysis results and charts
└── logs/                 # Execution logs
```

## 🔧 Configuration

Edit `config.py` to customize the strategy:

```python
class TradingConfig:
    # EMA Parameters
    EMA_SHORT = 20          # Short period EMA
    EMA_LONG = 50           # Long period EMA
    
    # Trading Parameters
    INITIAL_CAPITAL = 100000    # Starting capital (INR)
    RISK_PER_TRADE = 0.02      # Risk 2% per trade
    STOP_LOSS_PCT = 0.05       # 5% stop loss
    TAKE_PROFIT_PCT = 0.10     # 10% take profit
    
    # Data Parameters
    DATA_PERIOD = "2y"         # Historical data period
    DATA_INTERVAL = "1d"       # Daily data
```

## 📈 Usage Examples

### 1. Single Stock Analysis
```python
strategy = EMACrossoverStrategy()

# Comprehensive analysis of Reliance
results = strategy.analyze_single_stock(
    symbol="RELIANCE.NS",
    period="1y",
    show_plots=True,
    save_plots=True
)

print(f"Total return: {results['backtest_results']['total_return_pct']:.1f}%")
print(f"Win rate: {results['backtest_results']['win_rate']:.1f}%")
```

### 2. Multi-Stock Scanning
```python
# Scan all configured stocks
summary_df = strategy.scan_all_stocks()
print(summary_df)

# Get top performers
top_performers = summary_df.nlargest(5, 'Total_Return_%')
```

### 3. Current Signal Detection
```python
# Get high-quality current signals
signals = strategy.get_current_signals(min_strength=0.8)

for signal in signals:
    print(f"{signal['symbol']} - {signal['signal']} @ ₹{signal['price']:.2f}")
```

### 4. Portfolio Recommendations
```python
portfolio = strategy.get_portfolio_recommendation(
    max_positions=5,
    min_signal_strength=0.8
)

print(f"Recommended buys: {len(portfolio['recommended_buys'])}")
```

## � Default Stock Universe

The algorithm comes pre-configured with 15 major NSE stocks:

- RELIANCE.NS (Reliance Industries)
- TCS.NS (Tata Consultancy Services)
- HDFCBANK.NS (HDFC Bank)
- INFY.NS (Infosys)
- HINDUNILVR.NS (Hindustan Unilever)
- ICICIBANK.NS (ICICI Bank)
- KOTAKBANK.NS (Kotak Mahindra Bank)
- LT.NS (Larsen & Toubro)
- SBIN.NS (State Bank of India)
- BHARTIARTL.NS (Bharti Airtel)
- ASIANPAINT.NS (Asian Paints)
- MARUTI.NS (Maruti Suzuki)
- TITAN.NS (Titan Company)
- NESTLEIND.NS (Nestle India)
- WIPRO.NS (Wipro)

## 📈 Performance Metrics

The algorithm calculates comprehensive performance metrics:

- **Total Return**: Absolute and percentage returns
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / Gross loss
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Sharpe Ratio**: Risk-adjusted return measure
- **Calmar Ratio**: Annual return / Maximum drawdown

## 🎨 Visualizations

The system generates multiple types of charts:

1. **Price & Signals Chart**: Price with EMA lines and trade signals
2. **Signal Analysis**: EMA differences, volatility, signal strength
3. **Backtest Results**: Equity curve, trade distribution, monthly returns
4. **Portfolio Dashboard**: Multi-stock performance comparison

## ⚠️ Important Notes

### Disclaimer
This is for educational and research purposes only. Not financial advice. Always do your own research and consider consulting with a financial advisor before making investment decisions.

### Data Source
- Uses Yahoo Finance via `yfinance` library
- Real-time data subject to delays and availability
- Ensure stable internet connection for data fetching

### Performance Considerations
- Initial run may take time to download historical data
- Consider using smaller stock lists for faster execution
- Data is cached to improve subsequent runs

## 🔧 Advanced Usage

### Custom Stock Lists
```python
my_stocks = ["TCS.NS", "INFY.NS", "WIPRO.NS"]
strategy.fetch_data(my_stocks, period="6m")
```

### Backtesting Specific Periods
```python
# Fetch data for specific date range
data = strategy.data_fetcher.fetch_stock_data(
    "RELIANCE.NS", 
    start_date="2023-01-01",
    end_date="2023-12-31"
)
```

### Signal Sensitivity Analysis
```python
# Test different signal strength thresholds
for threshold in [0.5, 0.6, 0.7, 0.8]:
    signals = strategy.get_current_signals(min_strength=threshold)
    print(f"Threshold {threshold}: {len(signals)} signals")
```

## 📞 Support

For questions, issues, or contributions:
- Create an issue in the repository
- Review the logs in the `logs/` directory for debugging
- Check the `results/` directory for saved analysis

## 🚀 Future Enhancements

Potential improvements and features:
- Multiple timeframe analysis
- Additional technical indicators
- Machine learning signal enhancement
- Real-time trading integration
- Portfolio optimization algorithms
- Risk parity position sizing

---

**Happy Trading! 📈**

*Remember: Past performance does not guarantee future results. Always practice proper risk management.*



#!/usr/bin/env python3
"""
Demo script for EMA Crossover Trading Strategy
Demonstrates how to use the trading algorithm for Indian stock market analysis
"""

import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from ema_crossover_strategy import EMACrossoverStrategy
from config import TradingConfig

def main():
    """Main demo function"""
    print("🚀 EMA Crossover Trading Strategy Demo")
    print("=" * 50)
    
    # Initialize the strategy
    print("Initializing trading strategy...")
    strategy = EMACrossoverStrategy(initial_capital=100000, enable_logging=True)
    
    # Demo 1: Analyze a single stock
    print("\n📊 Demo 1: Single Stock Analysis")
    print("-" * 30)
    
    symbol = "RELIANCE.NS"  # Reliance Industries
    print(f"Analyzing {symbol}...")
    
    try:
        results = strategy.analyze_single_stock(
            symbol=symbol,
            period="1y",  # 1 year of data
            show_plots=True,
            save_plots=True
        )
        
        if results:
            print(f"✅ Analysis completed for {symbol}")
            print(f"   📈 Data points: {results['data_points']}")
            print(f"   📅 Date range: {results['date_range'][0].date()} to {results['date_range'][1].date()}")
            print(f"   🎯 Total signals: {results['signal_summary']['total_signals']}")
            print(f"   💰 Total return: {results['backtest_results']['total_return_pct']:.1f}%")
            print(f"   🎲 Win rate: {results['backtest_results']['win_rate']:.1f}%")
            print(f"   📊 Sharpe ratio: {results['backtest_results']['sharpe_ratio']:.2f}")
            
            # Current signal
            current = results['current_signal']
            print(f"   🔔 Current signal: {current['signal']} (Strength: {current['strength']:.2f})")
        
    except Exception as e:
        print(f"❌ Error in single stock analysis: {str(e)}")
    
    # Demo 2: Scan multiple stocks
    print("\n📈 Demo 2: Multi-Stock Scan")
    print("-" * 30)
    
    # Use a smaller subset for demo
    demo_stocks = [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", 
        "INFY.NS", "HINDUNILVR.NS"
    ]
    
    try:
        print(f"Scanning {len(demo_stocks)} stocks...")
        
        # Fetch data for all stocks
        strategy.fetch_data(demo_stocks, period="6m")  # 6 months
        
        # Generate signals
        strategy.generate_signals(include_filters=True, min_signal_strength=0.6)
        
        # Run backtests
        strategy.run_backtest()
        
        # Get summary
        summary_df = strategy.scan_all_stocks(show_summary=True)
        
        print("\n📋 Scan Results Summary:")
        print(summary_df.to_string(index=False))
        
    except Exception as e:
        print(f"❌ Error in multi-stock scan: {str(e)}")
    
    # Demo 3: Current trading signals
    print("\n🎯 Demo 3: Current Trading Signals")
    print("-" * 30)
    
    try:
        current_signals = strategy.get_current_signals(min_strength=0.7)
        
        if current_signals:
            print(f"Found {len(current_signals)} high-quality signals:")
            for i, signal in enumerate(current_signals[:5], 1):  # Show top 5
                print(f"   {i}. {signal['symbol'].replace('.NS', '')} - "
                      f"{signal['signal']} (Strength: {signal['strength']:.2f}) "
                      f"@ ₹{signal['price']:.2f}")
        else:
            print("   No high-quality signals found at the moment")
            
    except Exception as e:
        print(f"❌ Error getting current signals: {str(e)}")
    
    # Demo 4: Portfolio recommendations
    print("\n💼 Demo 4: Portfolio Recommendations")
    print("-" * 30)
    
    try:
        portfolio = strategy.get_portfolio_recommendation(
            max_positions=3, 
            min_signal_strength=0.8
        )
        
        print(f"Portfolio recommendations as of {portfolio['timestamp'].strftime('%Y-%m-%d %H:%M')}:")
        
        if portfolio['recommended_buys']:
            print(f"   📈 BUY Recommendations:")
            for rec in portfolio['recommended_buys']:
                shares = int(portfolio['capital_per_position'] / rec['price'])
                print(f"      • {rec['symbol'].replace('.NS', '')} - "
                      f"{shares} shares @ ₹{rec['price']:.2f} "
                      f"(Strength: {rec['strength']:.2f})")
        
        if portfolio['recommended_sells']:
            print(f"   📉 SELL Recommendations:")
            for rec in portfolio['recommended_sells']:
                print(f"      • {rec['symbol'].replace('.NS', '')} @ ₹{rec['price']:.2f} "
                      f"(Strength: {rec['strength']:.2f})")
        
        if not portfolio['recommended_buys'] and not portfolio['recommended_sells']:
            print("   No recommendations at current signal strength threshold")
            
    except Exception as e:
        print(f"❌ Error getting portfolio recommendations: {str(e)}")
    
    # Demo 5: Save results
    print("\n💾 Demo 5: Saving Results")
    print("-" * 30)
    
    try:
        saved_file = strategy.save_results()
        if saved_file:
            print(f"✅ Results saved to: {saved_file}")
        else:
            print("   No results to save")
            
    except Exception as e:
        print(f"❌ Error saving results: {str(e)}")
    
    print("\n🎉 Demo completed!")
    print("=" * 50)
    print("📁 Check the 'results' folder for saved charts and data")
    print("📄 Check the 'logs' folder for detailed execution logs")

def quick_scan():
    """Quick function to scan for current signals"""
    print("🔍 Quick Signal Scan")
    print("=" * 20)
    
    strategy = EMACrossoverStrategy(enable_logging=False)
    
    # Quick scan with just a few stocks
    quick_stocks = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]
    
    try:
        strategy.fetch_data(quick_stocks, period="3m")
        strategy.generate_signals()
        signals = strategy.get_current_signals(min_strength=0.6)
        
        if signals:
            print(f"Found {len(signals)} signals:")
            for signal in signals:
                print(f"  {signal['symbol'].replace('.NS', '')} - {signal['signal']} "
                      f"(Strength: {signal['strength']:.2f})")
        else:
            print("No signals found")
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    # Uncomment the function you want to run:
    
    # Full demo (takes longer, shows charts)
    main()
    
    # Quick scan (faster, no charts)
    # quick_scan()
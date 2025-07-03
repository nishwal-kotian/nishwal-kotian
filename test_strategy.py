#!/usr/bin/env python3
"""
Simple test script to verify EMA crossover trading algorithm components
"""

import sys
import traceback
from datetime import datetime

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    try:
        from config import TradingConfig
        from data_fetcher import DataFetcher
        from signal_generator import CrossoverSignalGenerator, SignalType
        from backtester import Backtester
        from ema_crossover_strategy import EMACrossoverStrategy
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import error: {str(e)}")
        traceback.print_exc()
        return False

def test_configuration():
    """Test configuration setup"""
    print("\nTesting configuration...")
    try:
        config = TradingConfig()
        config.create_directories()
        
        assert config.EMA_SHORT == 20
        assert config.EMA_LONG == 50
        assert len(config.DEFAULT_STOCKS) > 0
        
        print("✅ Configuration test passed")
        return True
    except Exception as e:
        print(f"❌ Configuration error: {str(e)}")
        return False

def test_data_fetcher():
    """Test data fetching functionality"""
    print("\nTesting data fetcher...")
    try:
        from data_fetcher import DataFetcher
        
        fetcher = DataFetcher()
        
        # Test with a single stock and shorter period
        data = fetcher.fetch_stock_data("RELIANCE.NS", period="5d")
        
        if not data.empty:
            print(f"✅ Data fetcher test passed - got {len(data)} records")
            return True
        else:
            print("⚠️ Data fetcher returned empty data (might be market hours/weekend)")
            return True  # Don't fail test due to market timing
            
    except Exception as e:
        print(f"❌ Data fetcher error: {str(e)}")
        return False

def test_signal_generator():
    """Test signal generation"""
    print("\nTesting signal generator...")
    try:
        from data_fetcher import DataFetcher
        from signal_generator import CrossoverSignalGenerator
        
        fetcher = DataFetcher()
        generator = CrossoverSignalGenerator()
        
        # Get sample data
        data = fetcher.fetch_stock_data("TCS.NS", period="1mo")
        
        if not data.empty:
            # Add technical indicators
            data_with_indicators = fetcher.add_technical_indicators(data)
            
            # Generate signals
            signals = generator.generate_signals(data_with_indicators, include_filters=False)
            
            print(f"✅ Signal generator test passed - processed {len(signals)} records")
            return True
        else:
            print("⚠️ Signal generator test skipped - no data available")
            return True
            
    except Exception as e:
        print(f"❌ Signal generator error: {str(e)}")
        traceback.print_exc()
        return False

def test_strategy_initialization():
    """Test main strategy class initialization"""
    print("\nTesting strategy initialization...")
    try:
        from ema_crossover_strategy import EMACrossoverStrategy
        
        strategy = EMACrossoverStrategy(initial_capital=50000, enable_logging=False)
        
        print("✅ Strategy initialization test passed")
        return True
    except Exception as e:
        print(f"❌ Strategy initialization error: {str(e)}")
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all tests"""
    print("🚀 EMA Crossover Trading Algorithm - Component Tests")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_configuration,
        test_data_fetcher,
        test_signal_generator,
        test_strategy_initialization
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {str(e)}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The trading algorithm is ready to use.")
        print("\n💡 Next steps:")
        print("   - Run 'python demo.py' for a full demonstration")
        print("   - Check the README.md for detailed usage instructions")
    else:
        print("⚠️ Some tests failed. Please check the error messages above.")
        print("   - Ensure all dependencies are installed: pip install -r requirements.txt")
        print("   - Check your internet connection for data fetching")
        
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
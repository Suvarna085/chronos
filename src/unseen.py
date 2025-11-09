import yfinance as yf
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

# Import from existing modules
from transformer import MultiFeatureChronosModel
from tokenizer import MultiFeatureScaler, MultiFeatureTokenizer
from backtest import ChronosBacktester


def add_simple_features(df):
    """Add basic technical indicators (same as training data)"""
    data = df.copy()
    
    # Returns
    data['returns'] = data['Close'].pct_change() * 100
    
    # Moving averages
    data['sma_7'] = data['Close'].rolling(window=7).mean()
    data['sma_21'] = data['Close'].rolling(window=21).mean()
    
    # RSI (simple version)
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)
    data['rsi'] = 100 - (100 / (1 + rs))
    
    # Volume ratio (current vs average)
    data['volume_ratio'] = data['Volume'] / data['Volume'].rolling(window=20).mean()
    
    # Drop NaN rows
    data = data.dropna()
    
    return data


def download_and_prepare_stock(symbol, start_date='2020-01-01', end_date='2024-12-31'):
    """Download and prepare a new stock for testing"""
    print(f"\nDownloading {symbol}...")
    
    try:
        df = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=True)
        
        if len(df) == 0:
            print(f"❌ No data found for {symbol}")
            return None
        
        print(f"✓ Downloaded {len(df)} records")
        
        # Add features
        data = add_simple_features(df)
        
        # Select columns (must match training data)
        keep_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 
                     'returns', 'sma_7', 'sma_21', 'rsi', 'volume_ratio']
        
        data = data[keep_cols]
        
        print(f"✓ Processed {len(data)} clean records with {len(keep_cols)} features")
        
        # Save temporarily
        temp_dir = '../datasets/temp_test'
        os.makedirs(temp_dir, exist_ok=True)
        
        filepath = os.path.join(temp_dir, f'{symbol}_processed.csv')
        data.to_csv(filepath)
        print(f"✓ Saved to {filepath}")
        
        return filepath
        
    except Exception as e:
        print(f"❌ Error downloading {symbol}: {e}")
        return None


def test_generalization():
    """Test model on completely unseen companies"""
    
    print("=" * 70)
    print("CHRONOS GENERALIZATION TEST - UNSEEN COMPANIES")
    print("=" * 70)
    
    # Show trained companies
    print("\n📚 Model was trained on:")
    trained_stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA']
    print(f"   {', '.join(trained_stocks)}")
    
    # Suggest test companies from different sectors
    suggestions = {
        'Tech': ['META', 'NFLX', 'ADBE', 'CRM', 'ORCL'],
        'Finance': ['JPM', 'BAC', 'GS', 'MS', 'V'],
        'Healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV', 'TMO'],
        'Consumer': ['WMT', 'HD', 'NKE', 'SBUX', 'MCD'],
        'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG'],
        'Industrial': ['BA', 'CAT', 'GE', 'HON', 'UPS']
    }
    
    print("\n💡 Suggested companies to test (not in training data):")
    for sector, stocks in suggestions.items():
        print(f"   {sector:12s}: {', '.join(stocks)}")
    
    print("\n" + "=" * 70)
    print("Enter stock symbol(s) to test:")
    print("  - Single: META")
    print("  - Multiple: META,JPM,WMT")
    print("  - By sector: Enter sector name (Tech, Finance, etc.)")
    
    user_input = input("\nYour choice: ").strip()
    
    # Parse input
    test_stocks = []
    
    # Check if it's a sector name
    if user_input.title() in suggestions:
        test_stocks = suggestions[user_input.title()]
        print(f"\n✓ Testing all {user_input.title()} sector stocks: {', '.join(test_stocks)}")
    else:
        # Parse as stock symbols
        test_stocks = [s.strip().upper() for s in user_input.split(',')]
        
        # Check if any are in training set
        overlap = [s for s in test_stocks if s in trained_stocks]
        if overlap:
            print(f"\n⚠️  Warning: {', '.join(overlap)} were in the training data!")
            proceed = input("Continue anyway? (y/n): ").strip().lower()
            if proceed != 'y':
                print("Exiting...")
                return
    
    if not test_stocks:
        print("❌ No valid stocks entered. Exiting.")
        return
    
    print(f"\n🔬 Testing generalization on {len(test_stocks)} unseen companies...")
    print("=" * 70)
    
    # Setup paths
    SRC_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SRC_DIR)
    MODEL_PATH = os.path.join(PROJECT_ROOT, 'trained_models', 'chronos_best.pt')
    TOKENIZED_DATA_DIR = os.path.join(PROJECT_ROOT, 'tokenized_data')
    
    # Initialize backtester
    backtester = ChronosBacktester(MODEL_PATH, TOKENIZED_DATA_DIR)
    
    # Download and test each stock
    all_results = []
    successful_tests = []
    
    for symbol in test_stocks:
        print("\n" + "=" * 70)
        
        # Download and prepare
        filepath = download_and_prepare_stock(symbol)
        
        if filepath is None:
            print(f"⏭️  Skipping {symbol}")
            continue
        
        # Run backtest
        try:
            result = backtester.backtest_stock(filepath)
            
            if result:
                all_results.append(result)
                successful_tests.append(symbol)
                
                # Save plots with "unseen" prefix
                output_dir = os.path.join(PROJECT_ROOT, 'backtest_results', 'unseen_companies')
                backtester.plot_predictions(result, output_dir=output_dir)
                
        except Exception as e:
            print(f"❌ Error testing {symbol}: {e}")
            continue
    
    if not all_results:
        print("\n❌ No successful tests completed.")
        return
    
    # Compare with training set performance
    print("\n" + "=" * 70)
    print("GENERALIZATION TEST RESULTS")
    print("=" * 70)
    
    summary_df = pd.DataFrame([
        {
            'Stock': r['stock'],
            'Sector': 'Unseen',
            'MAE': r['metrics']['MAE'],
            'MAPE': r['metrics']['MAPE'],
            'RMSE': r['metrics']['RMSE'],
            'Direction_Acc': r['metrics']['Direction_Accuracy'],
            'R²': r['metrics']['R2']
        }
        for r in all_results
    ])
    
    print("\n" + summary_df.to_string(index=False))
    
    # Calculate averages
    print("\n" + "=" * 70)
    print("UNSEEN COMPANIES AVERAGE PERFORMANCE:")
    print(f"  Average MAE:  ${summary_df['MAE'].mean():.2f}")
    print(f"  Average MAPE: {summary_df['MAPE'].mean():.2f}%")
    print(f"  Average RMSE: ${summary_df['RMSE'].mean():.2f}")
    print(f"  Average Direction Accuracy: {summary_df['Direction_Acc'].mean():.1f}%")
    print(f"  Average R²: {summary_df['R²'].mean():.4f}")
    
    # Comparison guidance
    print("\n" + "=" * 70)
    print("📊 INTERPRETATION GUIDE:")
    print("=" * 70)
    print("If unseen company performance is similar to training companies:")
    print("  ✅ Model has learned generalizable patterns")
    print("  ✅ Can be applied to new stocks confidently")
    print("\nIf unseen company performance is much worse:")
    print("  ⚠️  Model may have overfit to training companies")
    print("  ⚠️  Consider retraining with more diverse stocks")
    print("\nGood generalization benchmarks:")
    print("  • Direction Accuracy: > 60% (baseline is 50%)")
    print("  • MAPE: < 10%")
    print("  • R²: > 0.3")
    print("=" * 70)
    
    # Save results
    output_dir = os.path.join(PROJECT_ROOT, 'backtest_results', 'unseen_companies')
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_filename = f"generalization_test_{timestamp}.csv"
    summary_path = os.path.join(output_dir, summary_filename)
    summary_df.to_csv(summary_path, index=False)
    
    print(f"\n✓ Results saved to {summary_path}")
    print(f"✓ Plots saved to {output_dir}/")
    
    return all_results, summary_df


if __name__ == "__main__":
    test_generalization()
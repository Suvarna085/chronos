import yfinance as yf
import pandas as pd
import numpy as np
import os

# Configuration
STOCKS = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA']
START_DATE = '2020-01-01'
END_DATE = '2024-12-31'

def collect_stock_data(symbols, start, end):
    """Download stock data from Yahoo Finance"""
    print("Collecting stock data...")
    all_data = {}
    
    for symbol in symbols:
        print(f"Downloading {symbol}...")
        try:
            df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
            all_data[symbol] = df
            print(f"✓ {symbol}: {len(df)} records")
        except Exception as e:
            print(f"✗ Failed to download {symbol}: {e}")
    
    return all_data

def add_simple_features(df):
    """Add basic technical indicators"""
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

def preprocess_data(stock_data):
    """Preprocess stock data with features"""
    print("\nPreprocessing data...")
    processed = {}
    
    for symbol, df in stock_data.items():
        print(f"Processing {symbol}...")
        
        # Add features
        data = add_simple_features(df)
        
        # Select columns we want
        # OHLCV + technical indicators
        keep_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 
                     'returns', 'sma_7', 'sma_21', 'rsi', 'volume_ratio']
        
        data = data[keep_cols]
        
        processed[symbol] = data
        print(f"✓ {symbol}: {len(data)} clean records with {len(keep_cols)} features")
    
    return processed

def save_data(processed_data):
    """Save processed data to files"""
    print(f"\nSaving data...")
    
    output_dir = '../datasets'
    os.makedirs(output_dir, exist_ok=True)
    
    for symbol, df in processed_data.items():
        filename = f'{output_dir}/{symbol}_processed.csv'
        df.to_csv(filename)
        print(f"✓ Saved {filename}")

def get_statistics(processed_data):
    """Print basic statistics"""
    print("\n=== Statistics ===")
    for symbol, df in processed_data.items():
        print(f"\n{symbol}:")
        print(f"  Records: {len(df)}")
        print(f"  Features: {len(df.columns)}")
        print(f"  Date range: {df.index[0]} to {df.index[-1]}")
        print(f"  Mean close: ${float(df['Close'].mean()):.2f}")

# Main execution
if __name__ == "__main__":
    
    # Step 1: Collect data
    raw_data = collect_stock_data(STOCKS, START_DATE, END_DATE)
    
    # Step 2: Preprocess and add features
    processed_data = preprocess_data(raw_data)
    
    # Step 3: Show statistics
    get_statistics(processed_data)
    
    # Step 4: Save data
    save_data(processed_data)
    
    print("\n✓ All done! Now run tokenizer.py to create training data.")
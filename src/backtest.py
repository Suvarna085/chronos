import torch
import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import yfinance as yf
from calendar import monthrange

# Import from your existing modules
from transformer import MultiFeatureChronosModel
from tokenizer import MultiFeatureScaler, MultiFeatureTokenizer


def add_simple_features(df):
    """Add basic technical indicators (same as training)"""
    data = df.copy()
    
    # Returns
    data['returns'] = data['Close'].pct_change() * 100
    
    # Moving averages
    data['sma_7'] = data['Close'].rolling(window=7).mean()
    data['sma_21'] = data['Close'].rolling(window=21).mean()
    
    # RSI
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)
    data['rsi'] = 100 - (100 / (1 + rs))
    
    # Volume ratio
    data['volume_ratio'] = data['Volume'] / data['Volume'].rolling(window=20).mean()
    
    data = data.dropna()
    return data


class MonthlyForecaster:
    """Forecast a specific month and compare with actual results"""
    
    def __init__(self, model_path, tokenized_data_dir, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model
        print("\n=== Loading Model ===")
        checkpoint = torch.load(model_path, map_location=self.device)
        self.feature_names = checkpoint['feature_names']
        self.vocab_size = checkpoint['vocab_size']
        self.num_features = checkpoint['num_features']
        
        config = checkpoint['config']
        self.model = MultiFeatureChronosModel(
            vocab_size=self.vocab_size,
            num_features=self.num_features,
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            n_layers=config['n_layers'],
            d_ff=config['d_ff'],
            dropout=0.0
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"✓ Model loaded")
        
        # Load preprocessing artifacts
        print("\n=== Loading Preprocessing Artifacts ===")
        with open(os.path.join(tokenized_data_dir, 'scaler.pkl'), 'rb') as f:
            self.scaler = pickle.load(f)
        print("✓ Scaler loaded")
        
        with open(os.path.join(tokenized_data_dir, 'tokenizer.pkl'), 'rb') as f:
            tokenizer_data = pickle.load(f)
            self.tokenizer = MultiFeatureTokenizer()
            self.tokenizer.n_tokens = tokenizer_data['n_tokens']
            self.tokenizer.tokenizers = tokenizer_data['tokenizers']
        print("✓ Tokenizer loaded")
        
        self.context_length = 128
        
        # Check scaler format and extract base feature names
        sample_key = list(self.scaler.scalers.keys())[0]
        self.scaler_uses_tuples = isinstance(sample_key, tuple)
        
        # Extract just the feature names (without symbol)
        if self.scaler_uses_tuples:
            # feature_names like [('Open', 'AAPL'), ('High', 'AAPL'), ...]
            # Extract just ['Open', 'High', ...]
            self.base_feature_names = [f[0] if isinstance(f, tuple) else f for f in self.feature_names]
        else:
            self.base_feature_names = self.feature_names
    
    def get_context_and_target_data(self, symbol, year, month):
        """Download historical context data and target month data"""
        
        # Target month dates
        target_start = datetime(year, month, 1)
        _, last_day = monthrange(year, month)
        target_end = datetime(year, month, last_day)
        
        # Download 1 year before target month to ensure enough context
        download_start = target_start - timedelta(days=365)
        download_end = target_end + timedelta(days=5)
        
        print(f"\n{'='*70}")
        print(f"DOWNLOADING DATA FOR {symbol}")
        print(f"{'='*70}")
        print(f"Target Month: {target_start.strftime('%B %Y')}")
        print(f"Download Range: {download_start.strftime('%Y-%m-%d')} to {download_end.strftime('%Y-%m-%d')}")
        
        try:
            df = yf.download(symbol, 
                           start=download_start.strftime('%Y-%m-%d'),
                           end=download_end.strftime('%Y-%m-%d'),
                           progress=False,
                           auto_adjust=True)
            
            if len(df) == 0:
                print(f"❌ No data available for {symbol}")
                return None, None, None
            
            print(f"✓ Downloaded {len(df)} total days")
            
            # Add technical features
            df = add_simple_features(df)
            
            # Keep only needed columns
            keep_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 
                        'returns', 'sma_7', 'sma_21', 'rsi', 'volume_ratio']
            df = df[keep_cols]
            
            # Extract target month data
            target_mask = (df.index >= target_start) & (df.index <= target_end)
            target_data = df[target_mask].copy()
            
            # Extract context data (128 days before target month)
            context_data = df[df.index < target_start].tail(self.context_length)
            
            # Validation
            if len(target_data) == 0:
                print(f"❌ No trading data in {target_start.strftime('%B %Y')}")
                return None, None, None
            
            if len(context_data) < self.context_length:
                print(f"❌ Insufficient context: only {len(context_data)} days (need {self.context_length})")
                print(f"   Try a later month or a stock with more history")
                return None, None, None
            
            num_trading_days = len(target_data)
            
            print(f"\n✓ Context Data: {len(context_data)} days")
            print(f"  From: {context_data.index[0].strftime('%Y-%m-%d')}")
            print(f"  To:   {context_data.index[-1].strftime('%Y-%m-%d')}")
            print(f"\n✓ Target Month: {num_trading_days} trading days")
            print(f"  From: {target_data.index[0].strftime('%Y-%m-%d')}")
            print(f"  To:   {target_data.index[-1].strftime('%Y-%m-%d')}")
            
            return context_data, target_data, num_trading_days
            
        except Exception as e:
            print(f"❌ Download error: {e}")
            return None, None, None
    
    def generate_forecast(self, context_data, num_days, symbol):
        """Generate predictions for the specified number of days"""
        
        print(f"\n{'='*70}")
        print(f"GENERATING {num_days}-DAY FORECAST")
        print(f"{'='*70}")
        
        # Prepare features - use the EXACT format from training
        features_data = {}
        for feat_name in self.feature_names:
            if isinstance(feat_name, tuple):
                # Feature name is like ('Open', 'AAPL')
                base_feat = feat_name[0]
                features_data[feat_name] = context_data[base_feat].values
            else:
                # Feature name is just 'Open'
                features_data[feat_name] = context_data[feat_name].values
        
        # Scale and tokenize context
        scaled = self.scaler.transform(features_data)
        tokens = self.tokenizer.tokenize(scaled)
        
        # Debug: check token shapes
        print(f"Token shapes: {[(feat, tokens[feat].shape) for feat in self.feature_names]}")
        
        # Prepare input tensor - ensure all features have same length
        context_list = []
        for feat in self.feature_names:
            feat_tokens = tokens[feat]
            if len(feat_tokens.shape) == 1:
                context_list.append(feat_tokens)
            else:
                # If 2D, take first dimension
                context_list.append(feat_tokens.flatten())
        
        # Verify all same length
        lengths = [len(t) for t in context_list]
        if len(set(lengths)) > 1:
            print(f"⚠️  Warning: Token length mismatch: {dict(zip(self.feature_names, lengths))}")
            # Trim all to minimum length
            min_len = min(lengths)
            context_list = [t[:min_len] for t in context_list]
            print(f"   Trimmed all to length: {min_len}")
        
        context_array = np.stack(context_list, axis=1)
        context_tensor = torch.LongTensor(context_array).unsqueeze(0).to(self.device)
        
        # Generate predictions day by day
        predictions = {feat: [] for feat in self.feature_names}
        current_seq = context_tensor.clone()
        
        print("Forecasting", end="", flush=True)
        with torch.no_grad():
            for step in range(num_days):
                if step % 5 == 0:
                    print(".", end="", flush=True)
                
                outputs = self.model(current_seq)
                
                next_tokens = []
                for feat_idx, logits in enumerate(outputs):
                    last_logits = logits[:, -1, :]
                    next_token = torch.argmax(last_logits, dim=-1)
                    next_tokens.append(next_token)
                    predictions[self.feature_names[feat_idx]].append(next_token.item())
                
                next_tokens_tensor = torch.stack(next_tokens, dim=1).unsqueeze(1)
                current_seq = torch.cat([current_seq, next_tokens_tensor], dim=1)
        
        print(" Done!")
        
        # Convert to arrays
        for feat in self.feature_names:
            predictions[feat] = np.array(predictions[feat])
        
        # Detokenize and inverse scale
        pred_scaled = self.tokenizer.detokenize(predictions)
        pred_values = self.scaler.inverse_transform(pred_scaled)
        
        print(f"✓ Generated {num_days} days of predictions")
        
        return pred_values
    
    def compare_and_evaluate(self, predictions, actual_data, symbol, year, month):
        """Compare predictions with actual data and calculate metrics"""
        
        print(f"\n{'='*70}")
        print(f"EVALUATION RESULTS")
        print(f"{'='*70}")
        
        # Extract actual values - ensure 1D arrays
        actual_close = actual_data['Close'].values.flatten()
        
        # Get predictions for Close - handle different shapes
        pred_close_raw = predictions['Close']
        if len(pred_close_raw.shape) > 1:
            pred_close = pred_close_raw.flatten()[:len(actual_close)]
        else:
            pred_close = pred_close_raw[:len(actual_close)]
        
        # Price metrics
        mae = np.mean(np.abs(pred_close - actual_close))
        mape = np.mean(np.abs((pred_close - actual_close) / (actual_close + 1e-8))) * 100
        rmse = np.sqrt(np.mean((pred_close - actual_close) ** 2))
        
        # Direction prediction
        pred_change = pred_close[-1] - pred_close[0]
        actual_change = actual_close[-1] - actual_close[0]
        pred_direction = "UP" if pred_change > 0 else "DOWN"
        actual_direction = "UP" if actual_change > 0 else "DOWN"
        direction_correct = (pred_direction == actual_direction)
        
        # Price change percentages
        pred_pct = (pred_change / pred_close[0]) * 100
        actual_pct = (actual_change / actual_close[0]) * 100
        
        # Display results
        month_name = datetime(year, month, 1).strftime('%B %Y')
        print(f"\n📅 Period: {month_name}")
        print(f"📈 Stock: {symbol}")
        print(f"📊 Trading Days: {len(actual_close)}")
        
        print(f"\n{'─'*70}")
        print("PRICE ACCURACY METRICS")
        print(f"{'─'*70}")
        print(f"  Mean Absolute Error (MAE):  ${mae:.2f}")
        print(f"  Mean Abs Percentage Error:   {mape:.2f}%")
        print(f"  Root Mean Squared Error:     ${rmse:.2f}")
        
        print(f"\n{'─'*70}")
        print("PRICE MOVEMENT COMPARISON")
        print(f"{'─'*70}")
        print(f"  Starting Price:  ${actual_close[0]:.2f}")
        print(f"  Ending Price:    ${actual_close[-1]:.2f}")
        print(f"  Actual Change:   ${actual_change:.2f} ({actual_pct:+.2f}%)")
        print(f"  Predicted Change: ${pred_change:.2f} ({pred_pct:+.2f}%)")
        
        print(f"\n{'─'*70}")
        print("DIRECTION PREDICTION")
        print(f"{'─'*70}")
        print(f"  Predicted: {pred_direction} {'⬆️' if pred_direction == 'UP' else '⬇️'}")
        print(f"  Actual:    {actual_direction} {'⬆️' if actual_direction == 'UP' else '⬇️'}")
        print(f"  Result:    {'✅ CORRECT' if direction_correct else '❌ INCORRECT'}")
        
        # Create visualization
        self.create_comparison_plot(pred_close, actual_close, actual_data.index, 
                                   symbol, year, month, mae, mape, direction_correct)
        
        return {
            'symbol': symbol,
            'period': month_name,
            'trading_days': len(actual_close),
            'mae': mae,
            'mape': mape,
            'rmse': rmse,
            'actual_change_pct': actual_pct,
            'pred_change_pct': pred_pct,
            'direction_correct': direction_correct,
            'pred_direction': pred_direction,
            'actual_direction': actual_direction
        }
    
    def create_comparison_plot(self, predictions, actuals, dates, symbol, year, month, 
                              mae, mape, direction_correct):
        """Create detailed comparison visualization"""
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        month_name = datetime(year, month, 1).strftime('%B %Y')
        status = "✓ CORRECT" if direction_correct else "✗ WRONG"
        fig.suptitle(f'{symbol} - {month_name} Forecast vs Actual [{status}]', 
                    fontsize=16, fontweight='bold')
        
        # Plot 1: Price comparison
        x = range(len(actuals))
        ax1.plot(x, predictions, label='Predicted', marker='o', linewidth=2.5, 
                markersize=6, color='#2E86AB', alpha=0.8)
        ax1.plot(x, actuals, label='Actual', marker='s', linewidth=2.5, 
                markersize=6, color='#A23B72', alpha=0.8)
        
        ax1.set_xlabel('Trading Day', fontsize=12)
        ax1.set_ylabel('Close Price ($)', fontsize=12)
        ax1.set_title(f'Price Forecast (MAE: ${mae:.2f}, MAPE: {mape:.2f}%)', fontsize=13)
        ax1.legend(fontsize=11, loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Add date labels
        date_labels = [d.strftime('%m/%d') for d in dates]
        step = max(1, len(date_labels) // 10)
        ax1.set_xticks(x[::step])
        ax1.set_xticklabels(date_labels[::step], rotation=45)
        
        # Plot 2: Prediction errors
        errors = predictions - actuals
        colors = ['#d62728' if e < 0 else '#2ca02c' for e in errors]
        ax2.bar(x, errors, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax2.axhline(0, color='black', linestyle='--', linewidth=1.5)
        
        ax2.set_xlabel('Trading Day', fontsize=12)
        ax2.set_ylabel('Error ($)', fontsize=12)
        ax2.set_title('Daily Prediction Errors', fontsize=13)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_xticks(x[::step])
        ax2.set_xticklabels(date_labels[::step], rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        output_dir = '../backtest_results/monthly_forecasts'
        os.makedirs(output_dir, exist_ok=True)
        filename = f'{symbol}_{year}_{month:02d}_forecast.png'
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ Chart saved: {filepath}")


def main():
    print("="*70)
    print("MONTHLY STOCK FORECAST VALIDATOR")
    print("="*70)
    print("\nForecasts a specific month and compares predictions with actual results")
    print("(Works best for months in 2025 - outside training data 2020-2024)")
    
    # User input
    print("\n" + "-"*70)
    symbol = input("Stock Symbol (e.g., AAPL, TSLA, MSFT): ").strip().upper()
    
    print("\nTarget Month to Forecast:")
    year = int(input("  Year (e.g., 2025): ").strip())
    month = int(input("  Month (1-12): ").strip())
    
    # Validation
    if not (1 <= month <= 12):
        print("❌ Invalid month. Must be between 1-12.")
        return
    
    target_date = datetime(year, month, 1)
    current_date = datetime.now()
    
    # Check if month is in the future
    if target_date > current_date:
        print(f"\n❌ Cannot forecast {target_date.strftime('%B %Y')} - it's in the future!")
        print(f"   Current date: {current_date.strftime('%B %Y')}")
        return
    
    # Warn if overlapping with training data
    if year <= 2024:
        print(f"\n⚠️  WARNING: {year} may overlap with training data (2020-2024)")
        print("   Results may be overfitted. Consider testing on 2025 data.")
        proceed = input("   Continue anyway? (y/n): ").strip().lower()
        if proceed != 'y':
            return
    
    # Setup paths
    SRC_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SRC_DIR)
    MODEL_PATH = os.path.join(PROJECT_ROOT, 'trained_models', 'chronos_best.pt')
    TOKENIZED_DATA_DIR = os.path.join(PROJECT_ROOT, 'tokenized_data')
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"\n❌ Model not found at: {MODEL_PATH}")
        return
    
    # Initialize forecaster
    forecaster = MonthlyForecaster(MODEL_PATH, TOKENIZED_DATA_DIR)
    
    # Step 1: Get data
    context_data, target_data, num_days = forecaster.get_context_and_target_data(
        symbol, year, month
    )
    
    if context_data is None:
        print("\n❌ Failed to retrieve data. Exiting.")
        return
    
    # Step 2: Generate forecast
    predictions = forecaster.generate_forecast(context_data, num_days, symbol)
    
    # Step 3: Compare and evaluate
    metrics = forecaster.compare_and_evaluate(predictions, target_data, symbol, year, month)
    
    # Summary
    print(f"\n{'='*70}")
    print("✓ FORECAST VALIDATION COMPLETE")
    print(f"{'='*70}")
    print(f"\nResults saved to: ../backtest_results/monthly_forecasts/")


if __name__ == "__main__":
    main()
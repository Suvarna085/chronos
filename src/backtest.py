import torch
import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import yfinance as yf
from calendar import monthrange
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

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
                base_feat = feat_name[0]
                features_data[feat_name] = context_data[base_feat].values
            else:
                features_data[feat_name] = context_data[feat_name].values
        
        # Scale and tokenize context
        scaled = self.scaler.transform(features_data)
        tokens = self.tokenizer.tokenize(scaled)
        
        # Prepare input tensor
        context_list = []
        for feat in self.feature_names:
            feat_tokens = tokens[feat]
            if len(feat_tokens.shape) == 1:
                context_list.append(feat_tokens)
            else:
                context_list.append(feat_tokens.flatten())
        
        # Verify all same length
        lengths = [len(t) for t in context_list]
        if len(set(lengths)) > 1:
            min_len = min(lengths)
            context_list = [t[:min_len] for t in context_list]
        
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
    
    def calculate_classification_metrics(self, pred_close, actual_close, threshold=0.5):
        """
        Calculate precision, recall, F1 for daily direction prediction
        
        Args:
            pred_close: Predicted prices
            actual_close: Actual prices
            threshold: Minimum % change to be considered UP/DOWN (default 0.5%)
        """
        # Calculate daily returns
        pred_returns = np.diff(pred_close) / pred_close[:-1] * 100
        actual_returns = np.diff(actual_close) / actual_close[:-1] * 100
        
        # Binary classification: UP (1) vs DOWN (0)
        # Apply threshold to filter out very small movements
        pred_direction = (pred_returns > threshold).astype(int)
        actual_direction = (actual_returns > threshold).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(actual_direction, pred_direction)
        
        # Handle case where there are no positive predictions
        try:
            precision = precision_score(actual_direction, pred_direction, zero_division=0)
            recall = recall_score(actual_direction, pred_direction, zero_division=0)
            f1 = f1_score(actual_direction, pred_direction, zero_division=0)
        except:
            precision = recall = f1 = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(actual_direction, pred_direction)
        
        # Additional metrics
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        return {
            'daily_accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'confusion_matrix': cm,
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'total_days': len(actual_direction),
            'actual_up_days': np.sum(actual_direction),
            'predicted_up_days': np.sum(pred_direction)
        }
    
    def compare_and_evaluate(self, predictions, actual_data, symbol, year, month):
        """Compare predictions with actual data and calculate comprehensive metrics"""
        
        print(f"\n{'='*70}")
        print(f"EVALUATION RESULTS")
        print(f"{'='*70}")
        
        # Extract actual values
        actual_close = actual_data['Close'].values.flatten()
        
        # Get predictions for Close
        pred_close_raw = predictions['Close']
        if len(pred_close_raw.shape) > 1:
            pred_close = pred_close_raw.flatten()[:len(actual_close)]
        else:
            pred_close = pred_close_raw[:len(actual_close)]
        
        # === REGRESSION METRICS ===
        mae = np.mean(np.abs(pred_close - actual_close))
        mape = np.mean(np.abs((pred_close - actual_close) / (actual_close + 1e-8))) * 100
        rmse = np.sqrt(np.mean((pred_close - actual_close) ** 2))
        
        # R-squared
        ss_res = np.sum((actual_close - pred_close) ** 2)
        ss_tot = np.sum((actual_close - np.mean(actual_close)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # === DIRECTIONAL METRICS ===
        # Overall month direction
        pred_change = pred_close[-1] - pred_close[0]
        actual_change = actual_close[-1] - actual_close[0]
        pred_direction = "UP" if pred_change > 0 else "DOWN"
        actual_direction = "UP" if actual_change > 0 else "DOWN"
        monthly_direction_correct = (pred_direction == actual_direction)
        
        # Price change percentages
        pred_pct = (pred_change / pred_close[0]) * 100
        actual_pct = (actual_change / actual_close[0]) * 100
        
        # === CLASSIFICATION METRICS (Daily Direction) ===
        classification_metrics = self.calculate_classification_metrics(pred_close, actual_close)
        
        # Display results
        month_name = datetime(year, month, 1).strftime('%B %Y')
        print(f"\n📅 Period: {month_name}")
        print(f"📈 Stock: {symbol}")
        print(f"📊 Trading Days: {len(actual_close)}")
        
        print(f"\n{'─'*70}")
        print("PRICE ACCURACY METRICS (Regression)")
        print(f"{'─'*70}")
        print(f"  Mean Absolute Error (MAE):       ${mae:.2f}")
        print(f"  Mean Abs Percentage Error (MAPE): {mape:.2f}%")
        print(f"  Root Mean Squared Error (RMSE):   ${rmse:.2f}")
        print(f"  R² Score:                         {r2:.4f}")
        
        print(f"\n{'─'*70}")
        print("MONTHLY PRICE MOVEMENT")
        print(f"{'─'*70}")
        print(f"  Starting Price:   ${actual_close[0]:.2f}")
        print(f"  Ending Price:     ${actual_close[-1]:.2f}")
        print(f"  Actual Change:    ${actual_change:.2f} ({actual_pct:+.2f}%)")
        print(f"  Predicted Change: ${pred_change:.2f} ({pred_pct:+.2f}%)")
        
        print(f"\n{'─'*70}")
        print("MONTHLY DIRECTION PREDICTION")
        print(f"{'─'*70}")
        print(f"  Predicted: {pred_direction} {'⬆️' if pred_direction == 'UP' else '⬇️'}")
        print(f"  Actual:    {actual_direction} {'⬆️' if actual_direction == 'UP' else '⬇️'}")
        print(f"  Result:    {'✅ CORRECT' if monthly_direction_correct else '❌ INCORRECT'}")
        
        print(f"\n{'─'*70}")
        print("DAILY DIRECTION METRICS (Classification)")
        print(f"{'─'*70}")
        print(f"  Accuracy:   {classification_metrics['daily_accuracy']:.2%}")
        print(f"  Precision:  {classification_metrics['precision']:.2%}")
        print(f"  Recall:     {classification_metrics['recall']:.2%}")
        print(f"  F1-Score:   {classification_metrics['f1_score']:.2%}")
        print(f"  Specificity: {classification_metrics['specificity']:.2%}")
        
        print(f"\n{'─'*70}")
        print("CONFUSION MATRIX (Daily Directions)")
        print(f"{'─'*70}")
        cm = classification_metrics['confusion_matrix']
        print(f"                 Predicted DOWN  Predicted UP")
        print(f"  Actual DOWN         {cm[0,0]:3d}           {cm[0,1]:3d}")
        print(f"  Actual UP           {cm[1,0]:3d}           {cm[1,1]:3d}")
        print(f"\n  True Positives:  {classification_metrics['true_positives']} (correctly predicted UP days)")
        print(f"  True Negatives:  {classification_metrics['true_negatives']} (correctly predicted DOWN days)")
        print(f"  False Positives: {classification_metrics['false_positives']} (predicted UP, was DOWN)")
        print(f"  False Negatives: {classification_metrics['false_negatives']} (predicted DOWN, was UP)")
        
        print(f"\n{'─'*70}")
        print("TRADING SIGNAL ANALYSIS")
        print(f"{'─'*70}")
        print(f"  Actual UP days:    {classification_metrics['actual_up_days']}/{classification_metrics['total_days']}")
        print(f"  Predicted UP days: {classification_metrics['predicted_up_days']}/{classification_metrics['total_days']}")
        
        # Create visualization
        self.create_comparison_plot(pred_close, actual_close, actual_data.index, 
                                   symbol, year, month, mae, mape, 
                                   monthly_direction_correct, classification_metrics)
        
        return {
            'symbol': symbol,
            'period': month_name,
            'trading_days': len(actual_close),
            # Regression metrics
            'mae': mae,
            'mape': mape,
            'rmse': rmse,
            'r2': r2,
            # Monthly direction
            'actual_change_pct': actual_pct,
            'pred_change_pct': pred_pct,
            'monthly_direction_correct': monthly_direction_correct,
            'pred_direction': pred_direction,
            'actual_direction': actual_direction,
            # Classification metrics
            **classification_metrics
        }
    
    def create_comparison_plot(self, predictions, actuals, dates, symbol, year, month, 
                              mae, mape, monthly_correct, class_metrics):
        """Create detailed comparison visualization with metrics"""
        
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        month_name = datetime(year, month, 1).strftime('%B %Y')
        status = "✓" if monthly_correct else "✗"
        fig.suptitle(f'{symbol} - {month_name} Forecast [{status} Monthly Direction]', 
                    fontsize=16, fontweight='bold')
        
        # Plot 1: Price comparison
        ax1 = fig.add_subplot(gs[0, :])
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
        
        date_labels = [d.strftime('%m/%d') for d in dates]
        step = max(1, len(date_labels) // 10)
        ax1.set_xticks(x[::step])
        ax1.set_xticklabels(date_labels[::step], rotation=45)
        
        # Plot 2: Prediction errors
        ax2 = fig.add_subplot(gs[1, 0])
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
        
        # Plot 3: Daily returns comparison
        ax3 = fig.add_subplot(gs[1, 1])
        pred_returns = np.diff(predictions) / predictions[:-1] * 100
        actual_returns = np.diff(actuals) / actuals[:-1] * 100
        
        x_returns = range(len(actual_returns))
        ax3.scatter(x_returns, actual_returns, label='Actual Returns', 
                   alpha=0.6, s=60, color='#A23B72')
        ax3.scatter(x_returns, pred_returns, label='Predicted Returns', 
                   alpha=0.6, s=60, color='#2E86AB', marker='^')
        ax3.axhline(0, color='black', linestyle='--', linewidth=1)
        
        ax3.set_xlabel('Trading Day', fontsize=12)
        ax3.set_ylabel('Daily Return (%)', fontsize=12)
        ax3.set_title('Daily Returns Comparison', fontsize=13)
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Confusion matrix heatmap
        ax4 = fig.add_subplot(gs[2, 0])
        cm = class_metrics['confusion_matrix']
        im = ax4.imshow(cm, cmap='Blues', aspect='auto')
        
        ax4.set_xticks([0, 1])
        ax4.set_yticks([0, 1])
        ax4.set_xticklabels(['Pred DOWN', 'Pred UP'])
        ax4.set_yticklabels(['Actual DOWN', 'Actual UP'])
        
        # Add text annotations
        for i in range(2):
            for j in range(2):
                text = ax4.text(j, i, cm[i, j], ha="center", va="center", 
                              color="white" if cm[i, j] > cm.max()/2 else "black",
                              fontsize=20, fontweight='bold')
        
        ax4.set_title('Confusion Matrix (Daily Directions)', fontsize=13)
        plt.colorbar(im, ax=ax4)
        
        # Plot 5: Metrics summary
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.axis('off')
        
        metrics_text = f"""
Classification Metrics (Daily Direction):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Accuracy:    {class_metrics['daily_accuracy']:.1%}
Precision:   {class_metrics['precision']:.1%}
Recall:      {class_metrics['recall']:.1%}
F1-Score:    {class_metrics['f1_score']:.1%}
Specificity: {class_metrics['specificity']:.1%}

Trading Days: {class_metrics['total_days']}
Actual UP:    {class_metrics['actual_up_days']}
Predicted UP: {class_metrics['predicted_up_days']}

True Positives:  {class_metrics['true_positives']}
True Negatives:  {class_metrics['true_negatives']}
False Positives: {class_metrics['false_positives']}
False Negatives: {class_metrics['false_negatives']}
        """
        
        ax5.text(0.1, 0.9, metrics_text, transform=ax5.transAxes,
                fontsize=11, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
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
    
    # Save metrics to CSV
    output_dir = '../backtest_results/monthly_forecasts'
    os.makedirs(output_dir, exist_ok=True)
    metrics_df = pd.DataFrame([metrics])
    csv_path = os.path.join(output_dir, f'{symbol}_{year}_{month:02d}_metrics.csv')
    metrics_df.to_csv(csv_path, index=False)
    print(f"✓ Metrics saved to: {csv_path}")


if __name__ == "__main__":
    main()
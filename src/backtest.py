import torch
import numpy as np
import pandas as pd
import pickle
import os
import glob
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import sys

# Import from your existing modules
from transformer import MultiFeatureChronosModel
from tokenizer import MultiFeatureScaler, MultiFeatureTokenizer


class ChronosBacktester:
    """Backtest the Chronos model on historical data"""
    
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
            dropout=0.0  # No dropout for inference
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"✓ Model loaded (val_loss: {checkpoint['val_loss']:.4f})")
        
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
        self.prediction_length = 30
    
    def tokenize(self, data_dict):
        """Tokenize continuous values"""
        return self.tokenizer.tokenize(data_dict)
    
    def detokenize(self, tokens_dict):
        """Convert tokens back to continuous values"""
        return self.tokenizer.detokenize(tokens_dict)
    
    def predict(self, context_tokens):
        """
        Generate predictions given context tokens
        context_tokens: dict of {feature_name: np.array of shape [context_length]}
        Returns: dict of {feature_name: np.array of shape [prediction_length]}
        """
        with torch.no_grad():
            # Prepare input [1, context_length, num_features]
            context_list = [context_tokens[feat] for feat in self.feature_names]
            context_array = np.stack(context_list, axis=1)  # [context_length, num_features]
            context_tensor = torch.LongTensor(context_array).unsqueeze(0).to(self.device)
            
            # Generate predictions autoregressively
            predictions = {feat: [] for feat in self.feature_names}
            
            current_seq = context_tensor.clone()
            
            for step in range(self.prediction_length):
                # Forward pass
                outputs = self.model(current_seq)  # List of [1, seq_len, vocab_size]
                
                # Sample next token for each feature
                next_tokens = []
                for feat_idx, logits in enumerate(outputs):
                    # Get logits for last position
                    last_logits = logits[:, -1, :]  # [1, vocab_size]
                    
                    # Greedy decoding (can also use sampling)
                    next_token = torch.argmax(last_logits, dim=-1)  # [1]
                    next_tokens.append(next_token)
                    
                    # Store prediction
                    predictions[self.feature_names[feat_idx]].append(next_token.item())
                
                # Append to sequence - stack tokens and add sequence dimension
                next_tokens_tensor = torch.stack(next_tokens, dim=1)  # [1, num_features]
                next_tokens_tensor = next_tokens_tensor.unsqueeze(1)  # [1, 1, num_features]
                current_seq = torch.cat([current_seq, next_tokens_tensor], dim=1)
            
            # Convert lists to arrays
            for feat in self.feature_names:
                predictions[feat] = np.array(predictions[feat])
        
        return predictions
    
    def backtest_stock(self, stock_file):
        """
        Backtest on a single stock
        Returns: dict of metrics
        """
        stock_name = os.path.basename(stock_file).split('_processed.csv')[0]
        print(f"\n=== Backtesting {stock_name} ===")
        
        # Load data
        df = pd.read_csv(stock_file, index_col=0)
        
        # Handle the index - skip if first row is 'Ticker' or non-date
        if df.index[0] == 'Ticker' or not df.index[0].replace('-', '').replace('/', '').isdigit():
            df = df.iloc[1:]  # Skip first row
        
        # Now convert to datetime
        df.index = pd.to_datetime(df.index, errors='coerce')
        df = df[df.index.notna()]  # Remove any rows with invalid dates
        
        # Prepare features
        features_data = {}
        for col in df.columns:
            values = pd.to_numeric(df[col], errors='coerce').values
            values = values[~np.isnan(values)]
            features_data[col] = values
        
        # Scale and tokenize
        scaled = self.scaler.transform(features_data)
        tokens = self.tokenize(scaled)
        
        # Generate test windows
        total_len = len(list(tokens.values())[0])
        num_tests = (total_len - self.context_length - self.prediction_length) // self.prediction_length
        
        if num_tests < 1:
            print(f"⚠ Not enough data for backtesting {stock_name}")
            return None
        
        # Store results
        all_predictions = []
        all_actuals = []
        test_dates = []
        
        print(f"Running {num_tests} backtest windows...")
        for i in tqdm(range(num_tests)):
            start_idx = i * self.prediction_length
            end_idx = start_idx + self.context_length
            pred_end_idx = end_idx + self.prediction_length
            
            if pred_end_idx > total_len:
                break
            
            # Extract context
            context = {feat: tokens[feat][start_idx:end_idx] for feat in self.feature_names}
            
            # Extract actual future values (tokens)
            actual_tokens = {feat: tokens[feat][end_idx:pred_end_idx] for feat in self.feature_names}
            
            # Predict
            pred_tokens = self.predict(context)
            
            # Detokenize predictions and actuals
            pred_scaled = self.detokenize(pred_tokens)
            actual_scaled = self.detokenize(actual_tokens)
            
            # Inverse transform to original scale
            pred_values = self.scaler.inverse_transform(pred_scaled)
            actual_values = self.scaler.inverse_transform(actual_scaled)
            
            all_predictions.append(pred_values)
            all_actuals.append(actual_values)
            
            # Get date for this prediction
            date_idx = end_idx - 1  # Last context point
            if date_idx < len(df):
                test_dates.append(df.index[date_idx])
        
        # Calculate metrics
        metrics = self.calculate_metrics(all_predictions, all_actuals, stock_name)
        
        return {
            'stock': stock_name,
            'metrics': metrics,
            'predictions': all_predictions,
            'actuals': all_actuals,
            'dates': test_dates,
            'df': df
        }
    
    def calculate_metrics(self, predictions, actuals, stock_name):
        """Calculate prediction accuracy metrics"""
        metrics = {}
        
        # Focus on Close price for main metrics
        close_idx = self.feature_names.index('Close')
        
        pred_close = np.array([p['Close'] for p in predictions])  # [num_tests, pred_len]
        actual_close = np.array([a['Close'] for a in actuals])
        
        # 1. Mean Absolute Error (MAE)
        mae = np.mean(np.abs(pred_close - actual_close))
        metrics['MAE'] = mae
        
        # 2. Mean Absolute Percentage Error (MAPE)
        mape = np.mean(np.abs((pred_close - actual_close) / (actual_close + 1e-8))) * 100
        metrics['MAPE'] = mape
        
        # 3. Root Mean Square Error (RMSE)
        rmse = np.sqrt(np.mean((pred_close - actual_close) ** 2))
        metrics['RMSE'] = rmse
        
        # 4. Direction Accuracy (did we predict up/down correctly?)
        # Compare first and last values
        pred_direction = np.sign(pred_close[:, -1] - pred_close[:, 0])
        actual_direction = np.sign(actual_close[:, -1] - actual_close[:, 0])
        direction_accuracy = np.mean(pred_direction == actual_direction) * 100
        metrics['Direction_Accuracy'] = direction_accuracy
        
        # 5. R² Score
        ss_res = np.sum((actual_close - pred_close) ** 2)
        ss_tot = np.sum((actual_close - np.mean(actual_close)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        metrics['R2'] = r2
        
        # Print metrics
        print(f"\n{stock_name} Metrics:")
        print(f"  MAE:  ${mae:.2f}")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  RMSE: ${rmse:.2f}")
        print(f"  Direction Accuracy: {direction_accuracy:.1f}%")
        print(f"  R²: {r2:.4f}")
        
        return metrics
    
    def plot_predictions(self, backtest_results, output_dir='../backtest_results'):
        """Create visualization plots"""
        os.makedirs(output_dir, exist_ok=True)
        
        stock_name = backtest_results['stock']
        predictions = backtest_results['predictions']
        actuals = backtest_results['actuals']
        dates = backtest_results['dates']
        
        # Plot 1: First few predictions vs actuals
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{stock_name} - Sample Predictions vs Actuals', fontsize=16)
        
        for idx, ax in enumerate(axes.flat):
            if idx >= len(predictions):
                break
            
            pred = predictions[idx]['Close']
            actual = actuals[idx]['Close']
            
            ax.plot(pred, label='Predicted', marker='o', linewidth=2)
            ax.plot(actual, label='Actual', marker='s', linewidth=2)
            ax.set_title(f'Test Window {idx+1} ({dates[idx].strftime("%Y-%m-%d")})')
            ax.set_xlabel('Days Ahead')
            ax.set_ylabel('Close Price ($)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{stock_name}_sample_predictions.png', dpi=150)
        plt.close()
        
        # Plot 2: Error distribution
        all_errors = []
        for pred, actual in zip(predictions, actuals):
            errors = pred['Close'] - actual['Close']
            all_errors.extend(errors)
        
        plt.figure(figsize=(10, 6))
        plt.hist(all_errors, bins=50, edgecolor='black', alpha=0.7)
        plt.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        plt.xlabel('Prediction Error ($)')
        plt.ylabel('Frequency')
        plt.title(f'{stock_name} - Prediction Error Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{output_dir}/{stock_name}_error_distribution.png', dpi=150)
        plt.close()
        
        print(f"✓ Plots saved to {output_dir}/")


def run_full_backtest(model_path, tokenized_data_dir, dataset_dir):
    """Run backtest on all stocks"""
    
    print("=" * 60)
    print("CHRONOS MODEL BACKTESTING")
    print("=" * 60)
    
    # Initialize backtester
    backtester = ChronosBacktester(model_path, tokenized_data_dir)
    
    # Find all stock files
    stock_files = glob.glob(os.path.join(dataset_dir, '*_processed.csv'))
    
    if not stock_files:
        print(f"❌ No stock files found in {dataset_dir}")
        return None, None
    
    # Extract stock names
    available_stocks = {}
    for stock_file in stock_files:
        stock_name = os.path.basename(stock_file).split('_processed.csv')[0]
        available_stocks[stock_name.upper()] = stock_file
    
    # Display available stocks
    print(f"\nAvailable stocks for backtesting:")
    for i, stock_name in enumerate(sorted(available_stocks.keys()), 1):
        print(f"  {i}. {stock_name}")
    
    # Ask user which stock to backtest
    print(f"\nEnter stock symbol(s) to backtest:")
    print("  - Single stock: AAPL")
    print("  - Multiple stocks: AAPL,GOOGL,MSFT")
    print("  - All stocks: ALL")
    
    user_input = input("\nYour choice: ").strip().upper()
    
    # Determine which stocks to backtest
    stocks_to_test = []
    if user_input == "ALL":
        stocks_to_test = list(available_stocks.keys())
        print(f"\n✓ Backtesting all {len(stocks_to_test)} stocks")
    else:
        requested_stocks = [s.strip() for s in user_input.split(',')]
        for stock in requested_stocks:
            if stock in available_stocks:
                stocks_to_test.append(stock)
            else:
                print(f"⚠ Warning: {stock} not found in dataset, skipping...")
        
        if not stocks_to_test:
            print("❌ No valid stocks selected. Exiting.")
            return None, None
        
        print(f"\n✓ Backtesting {len(stocks_to_test)} stock(s): {', '.join(stocks_to_test)}")
    
    # Run backtest on selected stocks
    all_results = []
    for stock_name in stocks_to_test:
        stock_file = available_stocks[stock_name]
        result = backtester.backtest_stock(stock_file)
        if result:
            all_results.append(result)
            backtester.plot_predictions(result)
    
    if not all_results:
        print("\n❌ No successful backtests completed.")
        return None, None
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    
    summary_df = pd.DataFrame([
        {
            'Stock': r['stock'],
            'MAE': r['metrics']['MAE'],
            'MAPE': r['metrics']['MAPE'],
            'RMSE': r['metrics']['RMSE'],
            'Direction_Acc': r['metrics']['Direction_Accuracy'],
            'R²': r['metrics']['R2']
        }
        for r in all_results
    ])
    
    print("\n" + summary_df.to_string(index=False))
    
    # Overall averages (only if multiple stocks)
    if len(all_results) > 1:
        print("\n" + "=" * 60)
        print("OVERALL AVERAGES:")
        print(f"  Average MAE:  ${summary_df['MAE'].mean():.2f}")
        print(f"  Average MAPE: {summary_df['MAPE'].mean():.2f}%")
        print(f"  Average RMSE: ${summary_df['RMSE'].mean():.2f}")
        print(f"  Average Direction Accuracy: {summary_df['Direction_Acc'].mean():.1f}%")
        print(f"  Average R²: {summary_df['R²'].mean():.4f}")
        print("=" * 60)
    
    # Save summary
    output_dir = os.path.join(os.path.dirname(dataset_dir), 'backtest_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename with timestamp and stock names
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    stock_names = '_'.join(sorted(stocks_to_test))
    if len(stock_names) > 50:  # If too long, use count
        stock_names = f"{len(stocks_to_test)}stocks"
    
    summary_filename = f"summary_{stock_names}_{timestamp}.csv"
    summary_path = os.path.join(output_dir, summary_filename)
    summary_df.to_csv(summary_path, index=False)
    print(f"\n✓ Summary saved to {summary_path}")
    
    return all_results, summary_df


if __name__ == "__main__":
    
    # Setup paths
    SRC_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SRC_DIR)
    
    MODEL_PATH = os.path.join(PROJECT_ROOT, 'trained_models', 'chronos_best.pt')
    TOKENIZED_DATA_DIR = os.path.join(PROJECT_ROOT, 'tokenized_data')
    DATASET_DIR = os.path.join(PROJECT_ROOT, 'datasets')
    
    # Run backtest
    results, summary = run_full_backtest(MODEL_PATH, TOKENIZED_DATA_DIR, DATASET_DIR)
    
    print("\n✓ Backtesting complete!")
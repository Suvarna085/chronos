import numpy as np
import pandas as pd
import pickle
import os
import glob

class MultiFeatureScaler:
    """Scales multiple features independently"""
    
    def __init__(self):
        self.scalers = {}  # One scaler per feature
        self.epsilon = 1e-8
    
    def fit(self, data_dict):
        """
        Learn scaling parameters for each feature
        data_dict: {feature_name: array_of_values}
        """
        for feature_name, values in data_dict.items():
            mean = np.mean(values)
            std = np.std(values)
            self.scalers[feature_name] = {'mean': mean, 'std': std}
        return self
    
    def transform(self, data_dict):
        """Scale the data"""
        scaled = {}
        for feature_name, values in data_dict.items():
            mean = self.scalers[feature_name]['mean']
            std = self.scalers[feature_name]['std']
            scaled[feature_name] = (values - mean) / (std + self.epsilon)
        return scaled
    
    def inverse_transform(self, scaled_dict):
        """Convert back to original scale"""
        unscaled = {}
        for feature_name, values in scaled_dict.items():
            mean = self.scalers[feature_name]['mean']
            std = self.scalers[feature_name]['std']
            unscaled[feature_name] = values * (std + self.epsilon) + mean
        return unscaled


class MultiFeatureTokenizer:
    """Tokenizes multiple features independently"""
    
    def __init__(self, n_tokens=2048):
        self.n_tokens = n_tokens
        self.tokenizers = {}  # One tokenizer per feature
    
    def fit(self, data_dict):
        """
        Learn quantile boundaries for each feature
        data_dict: {feature_name: array_of_values}
        """
        print(f"Fitting tokenizers with {self.n_tokens} tokens per feature...")
        
        for feature_name, values in data_dict.items():
            percentiles = np.linspace(0, 100, self.n_tokens + 1)
            quantiles = np.percentile(values, percentiles)
            quantiles = np.unique(quantiles)
            
            self.tokenizers[feature_name] = {
                'quantiles': quantiles,
                'n_tokens': len(quantiles) - 1
            }
            print(f"  ✓ {feature_name}: {len(quantiles)-1} tokens")
        
        return self
    
    def tokenize(self, data_dict):
        """Convert continuous values to token IDs for each feature"""
        tokens = {}
        for feature_name, values in data_dict.items():
            quantiles = self.tokenizers[feature_name]['quantiles']
            n_tokens = self.tokenizers[feature_name]['n_tokens']
            
            feature_tokens = np.digitize(values, quantiles[1:])
            feature_tokens = np.clip(feature_tokens, 0, n_tokens - 1)
            tokens[feature_name] = feature_tokens
        
        return tokens
    
    def detokenize(self, tokens_dict):
        """Convert token IDs back to continuous values"""
        values = {}
        for feature_name, token_ids in tokens_dict.items():
            quantiles = self.tokenizers[feature_name]['quantiles']
            n_tokens = self.tokenizers[feature_name]['n_tokens']
            
            token_ids = np.clip(token_ids, 0, n_tokens - 1)
            lower = quantiles[token_ids]
            upper = quantiles[token_ids + 1]
            values[feature_name] = (lower + upper) / 2
        
        return values
    
    def save(self, filepath):
        """Save tokenizer"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'n_tokens': self.n_tokens,
                'tokenizers': self.tokenizers
            }, f)
        print(f"✓ Saved tokenizer to {filepath}")
    
    def load(self, filepath):
        """Load tokenizer"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.n_tokens = data['n_tokens']
            self.tokenizers = data['tokenizers']
        print(f"✓ Loaded tokenizer from {filepath}")
        return self


class MultiFeatureWindowBuilder:
    """Creates sliding windows from multivariate time series"""
    
    def __init__(self, context_length=128, prediction_length=30):
        self.context_length = context_length
        self.prediction_length = prediction_length
    
    def create_windows(self, tokens_dict):
        """
        Create windows from tokenized multi-feature data
        tokens_dict: {feature_name: array_of_tokens}
        """
        windows = []
        
        # Get length from first feature
        first_feature = list(tokens_dict.keys())[0]
        seq_length = len(tokens_dict[first_feature])
        
        total_length = self.context_length + self.prediction_length
        
        if seq_length < total_length:
            return []
        
        for i in range(seq_length - total_length + 1):
            window = {
                'context': {},
                'target': {}
            }
            
            # For each feature, extract context and target
            for feature_name, tokens in tokens_dict.items():
                window['context'][feature_name] = tokens[i:i + self.context_length]
                window['target'][feature_name] = tokens[i + self.context_length:i + total_length]
            
            windows.append(window)
        
        return windows


if __name__ == "__main__":
    
    # Setup paths
    SRC_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SRC_DIR)
    DATASET_DIR = os.path.join(PROJECT_ROOT, 'datasets')
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'tokenized_data')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Dataset dir: {DATASET_DIR}")
    print(f"Output dir:  {OUTPUT_DIR}")
    
    # Find all processed CSV files
    stock_files = glob.glob(os.path.join(DATASET_DIR, '*_processed.csv'))
    if not stock_files:
        raise FileNotFoundError(f"No *_processed.csv files found in {DATASET_DIR}")
    
    print(f"\nFound {len(stock_files)} stock files.")
    
    # --- Step 1: Collect ALL data for fitting ---
    print("\n=== Step 1: Fitting Scaler & Tokenizer ===")
    
    all_features_data = {}  # {feature_name: list_of_all_values}
    
    for stock_file in stock_files:
        df = pd.read_csv(stock_file, index_col=0)
        
        for col in df.columns:
            if col not in all_features_data:
                all_features_data[col] = []
            
            values = pd.to_numeric(df[col], errors='coerce').values
            values = values[~np.isnan(values)]
            all_features_data[col].extend(values)
    
    # Convert to numpy arrays
    for feature in all_features_data:
        all_features_data[feature] = np.array(all_features_data[feature])
        print(f"  {feature}: {len(all_features_data[feature])} values")
    
    # Fit scaler
    scaler = MultiFeatureScaler()
    scaler.fit(all_features_data)
    
    # Scale all data, then fit tokenizer
    scaled_data = scaler.transform(all_features_data)
    tokenizer = MultiFeatureTokenizer(n_tokens=2048)
    tokenizer.fit(scaled_data)
    
    # --- Step 2: Create windows ---
    print("\n=== Step 2: Creating Windows ===")
    
    all_windows = []
    window_builder = MultiFeatureWindowBuilder(context_length=128, prediction_length=30)
    
    for stock_file in stock_files:
        stock_name = os.path.basename(stock_file).split('_processed.csv')[0]
        
        df = pd.read_csv(stock_file, index_col=0)
        
        # Extract each feature
        features_data = {}
        for col in df.columns:
            values = pd.to_numeric(df[col], errors='coerce').values
            values = values[~np.isnan(values)]
            features_data[col] = values
        
        # Scale
        scaled = scaler.transform(features_data)
        
        # Tokenize
        tokens = tokenizer.tokenize(scaled)
        
        # Create windows
        windows = window_builder.create_windows(tokens)
        all_windows.extend(windows)
        print(f"✓ {stock_name}: {len(windows)} windows")
    
    print(f"\nTotal windows created: {len(all_windows)}")
    
    # --- Step 3: Save everything ---
    print("\n=== Step 3: Saving Artifacts ===")
    
    # Save feature names
    feature_names = list(all_features_data.keys())
    features_path = os.path.join(OUTPUT_DIR, 'feature_names.pkl')
    with open(features_path, 'wb') as f:
        pickle.dump(feature_names, f)
    print(f"✓ Saved {len(feature_names)} feature names")
    
    # Save scaler
    scaler_path = os.path.join(OUTPUT_DIR, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✓ Saved scaler")
    
    # Save tokenizer
    tokenizer_path = os.path.join(OUTPUT_DIR, 'tokenizer.pkl')
    tokenizer.save(tokenizer_path)
    
    # Save windows
    windows_path = os.path.join(OUTPUT_DIR, 'all_windows.pkl')
    with open(windows_path, 'wb') as f:
        pickle.dump(all_windows, f)
    print(f"✓ Saved {len(all_windows)} windows")
    
    print("\n✓ Dataset build complete!")
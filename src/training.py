import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import pickle
import os
import random

from transformer import MultiFeatureChronosModel, ChronosConfig


class MultiFeatureDataset(Dataset):
    """Dataset for multi-feature time series windows"""
    
    def __init__(self, windows, feature_names):
        self.windows = windows
        self.feature_names = feature_names
        self.num_features = len(feature_names)
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        window = self.windows[idx]
        
        # Stack all features for context
        context_list = [window['context'][feat] for feat in self.feature_names]
        context = np.stack(context_list, axis=1)  # [seq_len, num_features]
        
        # Stack all features for target
        target_list = [window['target'][feat] for feat in self.feature_names]
        target = np.stack(target_list, axis=1)  # [pred_len, num_features]
        
        return torch.LongTensor(context), torch.LongTensor(target)


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for context, target in pbar:
        context = context.to(device)  # [batch, seq, features]
        target = target.to(device)    # [batch, pred_len, features]
        
        # Concatenate context and target
        input_seq = torch.cat([context, target], dim=1)  # [batch, seq+pred, features]
        
        # Forward pass
        outputs = model(input_seq[:, :-1, :])  # List of [batch, seq-1, vocab]
        
        # Calculate loss for each feature
        loss = 0
        for feat_idx, logits in enumerate(outputs):
            # Get target for this feature
            target_feat = input_seq[:, 1:, feat_idx]  # [batch, seq-1]
            
            # Only compute loss on prediction region
            pred_start = context.size(1) - 1
            target_logits = logits[:, pred_start:, :]  # [batch, pred_len, vocab]
            target_flat = target[:, :, feat_idx].reshape(-1)  # [batch*pred_len]
            
            # Flatten logits
            target_logits_flat = target_logits.reshape(-1, target_logits.size(-1))
            
            # Compute loss
            feat_loss = criterion(target_logits_flat, target_flat)
            loss += feat_loss
        
        # Average loss across features
        loss = loss / len(outputs)
        
        if torch.isnan(loss) or torch.isinf(loss):
            print("Warning: NaN/Inf loss, skipping batch")
            continue
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    
    return total_loss / max(num_batches, 1)


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation", leave=False)
        for context, target in pbar:
            context = context.to(device)
            target = target.to(device)
            
            input_seq = torch.cat([context, target], dim=1)
            outputs = model(input_seq[:, :-1, :])
            
            loss = 0
            for feat_idx, logits in enumerate(outputs):
                pred_start = context.size(1) - 1
                target_logits = logits[:, pred_start:, :]
                target_flat = target[:, :, feat_idx].reshape(-1)
                target_logits_flat = target_logits.reshape(-1, target_logits.size(-1))
                
                feat_loss = criterion(target_logits_flat, target_flat)
                loss += feat_loss
            
            loss = loss / len(outputs)
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
    
    return total_loss / max(len(dataloader), 1)


def train_chronos(epochs=50, batch_size=16, learning_rate=3e-4):
    """Main training function"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Setup paths
    SRC_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SRC_DIR)
    TOKENIZED_DATA_DIR = os.path.join(PROJECT_ROOT, 'tokenized_data')
    MODEL_OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'trained_models')
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    
    print(f"Loading data from: {TOKENIZED_DATA_DIR}")
    
    # Load data
    print("=== Loading Data ===")
    
    # Load feature names
    with open(os.path.join(TOKENIZED_DATA_DIR, 'feature_names.pkl'), 'rb') as f:
        feature_names = pickle.load(f)
    print(f"✓ Features: {feature_names}")
    
    # Load windows
    with open(os.path.join(TOKENIZED_DATA_DIR, 'all_windows.pkl'), 'rb') as f:
        all_windows = pickle.load(f)
    print(f"✓ Loaded {len(all_windows)} windows")
    
    # Load tokenizer to get vocab size
    with open(os.path.join(TOKENIZED_DATA_DIR, 'tokenizer.pkl'), 'rb') as f:
        tokenizer_data = pickle.load(f)
    vocab_size = tokenizer_data['n_tokens']
    print(f"✓ Vocab size: {vocab_size}")
    
    # Shuffle and split
    random.seed(42)
    random.shuffle(all_windows)
    
    split_idx = int(0.9 * len(all_windows))
    train_windows = all_windows[:split_idx]
    val_windows = all_windows[split_idx:]
    
    train_dataset = MultiFeatureDataset(train_windows, feature_names)
    val_dataset = MultiFeatureDataset(val_windows, feature_names)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Create model
    config = ChronosConfig.SMALL
    model = MultiFeatureChronosModel(
        vocab_size=vocab_size,
        num_features=len(feature_names),
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        d_ff=config['d_ff'],
        dropout=0.2
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    print("\n=== Training ===")
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} - Train: {train_loss:.4f}, Val: {val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.1e}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            save_path = os.path.join(MODEL_OUTPUT_DIR, 'chronos_best.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'config': config,
                'vocab_size': vocab_size,
                'num_features': len(feature_names),
                'feature_names': feature_names
            }, save_path)
            print(f"✓ Saved best model (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= 5:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    print("\n✓ Training complete!")
    return model


if __name__ == "__main__":
    model = train_chronos(
        epochs=100,
        batch_size=32,
        learning_rate=3e-4
    )
    
    if model:
        print("Training finished successfully!")
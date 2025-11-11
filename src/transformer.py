import torch
import torch.nn as nn
import math
import os

class PositionalEncoding(nn.Module):
    """Adds positional information to embeddings"""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
    
#so basically transformer doesn't have a sense of direction, in RNN it's inplicit, but in transformer, it sees all the tokens together, if we want our transformer to learn abt relationship which depend on order of tokens, we add  positional encoding


class TransformerBlock(nn.Module):
    """Single transformer layer"""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        attn_out, _ = self.attention(x, x, x, attn_mask=mask, need_weights=False)
        x = self.norm1(x + self.dropout(attn_out))
        
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        return x


class MultiFeatureChronosModel(nn.Module):
    """Chronos model that handles multiple input features"""
    
    def __init__(self, vocab_size, num_features, d_model=256, n_heads=8, 
                 n_layers=6, d_ff=1024, dropout=0.1, max_len=1024):
        super().__init__()
        
        self.d_model = d_model 
        self.num_features = num_features
        
        # Token embedding (shared across all features)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Feature type embedding (to distinguish between features)
        self.feature_embedding = nn.Embedding(num_features, d_model)
        
        # Positional encoding (need more space for flattened sequence)
        self.pos_encoding = PositionalEncoding(d_model, max_len * num_features)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Separate output heads for each feature
        self.output_heads = nn.ModuleList([
            nn.Linear(d_model, vocab_size)
            for _ in range(num_features)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, mask=None):
        """
        x: [batch_size, seq_len, num_features] - token indices for each feature
        Returns: list of logits, one per feature [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len, num_features = x.shape
        
        # Embed tokens for all features
        # x: [batch, seq, features] -> [batch, seq, features, d_model]
        token_embeds = self.token_embedding(x)
        
        # Create feature embeddings
        feature_ids = torch.arange(num_features, device=x.device).unsqueeze(0).unsqueeze(0)
        feature_ids = feature_ids.expand(batch_size, seq_len, -1)
        feature_embeds = self.feature_embedding(feature_ids)
        
        # Combine: token + feature embeddings
        combined = token_embeds + feature_embeds
        
        # Flatten to [batch, seq * features, d_model]
        combined = combined.reshape(batch_size, seq_len * num_features, self.d_model)
        
        # Scale and add positional encoding
        combined = combined * math.sqrt(self.d_model)
        combined = self.pos_encoding(combined)
        combined = self.dropout(combined)
        
        # Pass through transformer
        for block in self.transformer_blocks:
            combined = block(combined, mask)
        
        # Reshape back to [batch, seq, features, d_model]
        combined = combined.reshape(batch_size, seq_len, num_features, self.d_model)
        
        # Apply separate output head for each feature
        outputs = []
        for i in range(num_features):
            feature_output = combined[:, :, i, :]  # [batch, seq, d_model]
            logits = self.output_heads[i](feature_output)  # [batch, seq, vocab_size]
            outputs.append(logits)
        
        return outputs  # List of tensors
    
    def generate_causal_mask(self, seq_len):
        """Creates causal mask for autoregressive generation"""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return mask


class ChronosConfig:
    """Configuration for model variants"""
    
    TINY = {
        'd_model': 128,
        'n_heads': 4,
        'n_layers': 2,
        'd_ff': 512
    }
    
    SMALL = {
        'd_model': 256,
        'n_heads': 8,
        'n_layers': 4,
        'd_ff': 1024
    }
    
    BASE = {
        'd_model': 512,
        'n_heads': 8,
        'n_layers': 6,
        'd_ff': 2048
    }


# Test
if __name__ == "__main__":
    VOCAB_SIZE = 2048
    NUM_FEATURES = 10  # OHLCV + 5 indicators
    BATCH_SIZE = 4
    SEQ_LEN = 128
    
    print("=== Creating Multi-Feature Chronos Model ===")
    
    config = ChronosConfig.SMALL
    model = MultiFeatureChronosModel(
        vocab_size=VOCAB_SIZE,
        num_features=NUM_FEATURES,
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        d_ff=config['d_ff']
    )
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created with {n_params:,} parameters")
    
    # Test forward pass
    print("\n=== Testing Forward Pass ===")
    dummy_tokens = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN, NUM_FEATURES))
    
    with torch.no_grad():
        outputs = model(dummy_tokens)
    
    print(f"Input shape: {dummy_tokens.shape}")
    print(f"Number of output heads: {len(outputs)}")
    print(f"Each output shape: {outputs[0].shape}")
    
    print("\n✓ Model test passed!")
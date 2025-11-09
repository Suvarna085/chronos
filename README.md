# Stock Price Prediction using Chronos Transformers

A deep learning project that predicts stock prices using a transformer model inspired by Amazon's Chronos framework.

## What This Does

Predicts stock prices 30 days into the future using historical data and technical indicators. Uses a transformer model that treats stock prices like language tokens.

## Results

**Training Stocks (TSLA, MSFT):**
- TSLA: 97% direction accuracy, 4% error
- MSFT: 74% direction accuracy, 2.7% error

**Unseen Stock (META):**
- 43% direction accuracy, 9.4% error
- **Conclusion:** Model learned specific stock patterns, not general market behavior

## Quick Start

### Install
```bash
pip install torch pandas numpy yfinance matplotlib tqdm
```

### Run
```bash
# 1. Download data
python src/load_data.py

# 2. Create training data
python src/tokenizer.py

# 3. Train model
python src/train.py

# 4. Test predictions
python src/backtest.py
```

## Project Structure
```
chronos/
├── src/
│   ├── load_data.py      # Get stock data
│   ├── tokenizer.py         # Process data
│   ├── transformer.py       # Model architecture
│   ├── train.py            # Train model
│   └── backtest.py         # Evaluate results
├── datasets/               # Stock CSV files
├── trained_models/         # Saved model
└── backtest_results/       # Plots & metrics
```

## How It Works

1. **Data Collection:** Downloads OHLCV data + calculates technical indicators (RSI, moving averages)
2. **Tokenization:** Converts prices to discrete tokens (like words in NLP)
3. **Model:** Transformer with 4 layers, 8 attention heads, ~2M parameters
4. **Training:** Learns patterns from 128-day windows to predict next 30 days
5. **Prediction:** Generates forecasts autoregressively (one day at a time)

## Metrics Explained

- **Direction Accuracy:** Did we predict up/down correctly? (Random = 50%)
- **MAPE:** Average % error in predictions
- **R²:** How well model explains price movements (1.0 = perfect)

## Key Findings

✅ **Good:**
- Works great on training data (74-97% direction accuracy)
- Low prediction errors (~3-4% MAPE)

❌ **Problem:**
- Poor on new stocks (43% accuracy = worse than guessing)
- Model overfitted to specific companies

## Why It Failed on New Stocks

- Only trained on 6 similar tech stocks
- Learned company-specific patterns instead of general market dynamics
- Needs more diverse training data (different sectors, market caps)

## Improvements Needed

1. **More Training Data:** Add 40+ stocks from finance, healthcare, energy sectors
2. **Longer Training:** 200+ epochs instead of 100
3. **Better Features:** Add market-wide indicators (VIX, sector indices)

**Expected Result:** 60-70% accuracy on unseen stocks (vs current 43%)

## Tech Stack

- **PyTorch:** Model training
- **yfinance:** Stock data
- **Pandas/NumPy:** Data processing
- **Matplotlib:** Visualization

## Testing on New Companies

```bash
python src/generalization_test.py
# Enter: META (or any stock not in training data)
```

## References

- [Chronos Paper](https://arxiv.org/abs/2403.07815) - Amazon's time series foundation model
- Transformer architecture for sequence prediction
- Quantile-based tokenization for continuous data

## Author

College project demonstrating deep learning for financial forecasting

---

**Note:** This is a research project. Not for actual trading decisions.

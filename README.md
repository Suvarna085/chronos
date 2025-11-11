# **Multi-Feature Time Series Transformer**

This is an end-to-end Python project for stock market forecasting using a multi-feature, Chronos-style Transformer model.

The core idea is to adapt the [Chronos methodology](https://www.google.com/search?q=https://huggingface.co/blog/chronos) (which tokenizes a single time series using quantiles) and extend it to a multi-feature context. This model simultaneously learns from 10 different financial features (OHLCV, SMA, RSI, etc.) by tokenizing each one independently and feeding them into a custom Transformer architecture.

The project is structured as a complete pipeline:

1. **Data Collection** (load\_data.py)  
2. **Preprocessing & Tokenization** (tokenizer.py)  
3. **Model Training** (training.py)  
4. **Backtesting & Evaluation** (backtest.py)

## **Key Features**

* **Multi-Feature Model:** The MultiFeatureChronosModel in transformer.py is a custom Transformer that uses feature embeddings to learn from 10 different time series features at once.  
* **Chronos-style Tokenization:** Implements a quantile-based tokenizer (MultiFeatureTokenizer) to convert continuous scaled data into discrete integer tokens, which is ideal for a Transformer's embedding layer.  
* **End-to-End Pipeline:** All scripts are designed to run in sequence, creating the necessary data, artifacts, and models at each step.  
* **Powerful Backtesting:** The backtest.py script is a powerful evaluation tool. It prompts you for a stock, year, and month, downloads fresh data, and runs a forecast. It then generates a detailed report comparing the forecast to the actual data, including:  
  * **Regression Metrics:** MAE, MAPE, RMSE, and R².  
  * **Directional Metrics:** Daily directional accuracy, precision, recall, and F1-score to see how well the model predicts "UP" vs. "DOWN" days.  
  * **Visualization:** Saves a detailed chart and a CSV of the metrics.

## **Project Structure**

The project assumes the following directory structure, which the scripts will create:
  
    .  
    ├── src/  
    │   ├── load\_data.py         \# 1\. Downloads and processes stock data  
    │   ├── tokenizer.py         \# 2\. Scales, tokenizes, and windows the data  
    │   ├── transformer.py       \# Defines the model architecture  
    │   ├── training.py          \# 3\. Trains the model  
    │   └── backtest.py          \# 4\. Evaluates the model on new data  
    │  
    ├── datasets/                \# Output of load\_data.py  
    │   └── AAPL\_processed.csv  
    │   └── ...  
    │  
    ├── tokenized\_data/          \# Output of tokenizer.py (artifacts)  
    │   ├── all\_windows.pkl  
    │   ├── feature\_names.pkl  
    │   ├── scaler.pkl  
    │   └── tokenizer.pkl  
    │  
    ├── trained\_models/          \# Output of training.py  
    │   └── chronos\_best.pt  
    │  
    └── backtest\_results/        \# Output of backtest.py  
        └── monthly\_forecasts/  
            └── AAPL\_2025\_01\_forecast.png  
            └── AAPL\_2025\_01\_metrics.csv  
            └── ...

## **Requirements**

You will need the following Python libraries. You can install them using pip:

    pip install torch numpy pandas yfinance scikit-learn matplotlib tqdm

## **How to Use (The 4-Step Workflow)**

Run the scripts from the command line in this order.

### **Step 1: Load and Process Data**

This script downloads data for the stocks defined in STOCKS (default: 'AAPL', 'GOOGL', 'MSFT', etc.) from 2020 to 2024\. It adds technical indicators and saves the results to the ../datasets/ directory.

    python src/load\_data.py

### **Step 2: Tokenize Data and Build Windows**

This script reads all the processed CSVs from ../datasets/. It fits a MultiFeatureScaler and MultiFeatureTokenizer on all the data, then creates sliding windows (context\_length=128, prediction\_length=30). It saves all artifacts (scaler, tokenizer, windows) to the ../tokenized\_data/ directory.

    python src/tokenizer.py

### **Step 3: Train the Model**

This script loads the windows and artifacts from ../tokenized\_data/ and trains the MultiFeatureChronosModel. It uses a validation split, early stopping, and saves the best-performing model to ../trained\_models/chronos\_best.pt.

**Note:** This step will take a significant amount of time, especially if you don't have a CUDA-enabled GPU.

    python src/training.py

### **Step 4: Run a Backtest and Evaluate**

This is the fun part\! Once the model is trained, you can run this script to test its performance on any stock for any past month. It works best on data *outside* the training range (e.g., any month in 2025).

The script will prompt you for input:

    python src/backtest.py

\# \--- Example Interaction \---  
Stock Symbol (e.g., AAPL, TSLA, MSFT): MSFT
    
    Target Month to Forecast:  
      Year (e.g., 2025): 2025  
      Month (1-12): 1

The script will then:

1. Download the required data (e.g., MSFT for Jan 2025 and 128 days of context before it).  
2. Generate a day-by-day forecast for the entire month.  
3. Compare the forecast to the actual data.  
4. Print a detailed metrics report to the console.  
5. Save a chart and a metrics CSV to ../backtest\_results/monthly\_forecasts/.

## **Code Overview**

* src/load\_data.py: Fetches raw stock data from yfinance and adds 5 technical indicators: returns, sma\_7, sma\_21, rsi, and volume\_ratio. Total features: 10 (OHLCV \+ 5 indicators).  
* src/tokenizer.py: Defines classes to scale, tokenize, and window the multi-feature data. This is the core of the data-processing pipeline.  
* src/transformer.py: Defines the neural network architecture. MultiFeatureChronosModel is the key class, which includes token embeddings, feature-type embeddings, and positional encodings, followed by standard Transformer blocks and separate output heads for each feature.  
* src/training.py: Standard PyTorch training loop. It loads the MultiFeatureDataset and trains the model, saving the best checkpoint based on validation loss.  
* src/backtest.py: The user-facing evaluation script. It loads the trained model and artifacts to perform a comprehensive forecast analysis on new, unseen data.

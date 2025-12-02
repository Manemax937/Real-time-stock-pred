📌 Overview

This project builds an end-to-end machine learning pipeline to predict the next-day direction of NIFTY 50 using:

Price-based features (returns, lags)

Technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands)

Volatility features

XGBoost Classifier as the core ML model

The system also includes a daily auto-update module that fetches the latest NIFTY data, updates the dataset, loads the trained model, and produces a final Buy / No Buy recommendation.

This project is ideal for:

Algorithmic trading students

Data science learners

Portfolio analysts

Anyone exploring ML for financial time series

🚀 Features
1. Daily Data Update

Automatically downloads the latest NIFTY OHLCV data and appends new rows to the existing dataset.

2. Feature Engineering

Generates over 20 technical features including:

Returns: ret_1, ret_2, ret_3

Moving averages: sma_5, sma_10, sma_20

Exponential averages: ema_12, ema_26

RSI, MACD, MACD signal, MACD histogram

Bollinger Band width

Volume-based features

Lag features: close_lag1, close_lag2, close_lag3, close_lag5

3. Train / Validation / Test Split

Data is chronologically split as:

Train: up to 2022

Validation: 2023

Test: 2024+

4. Model Training

Uses XGBoost Classifier with tuned hyperparameters:

n_estimators = 300

max_depth = 4

learning_rate = 0.05

subsample = 0.8

colsample_bytree = 0.8

5. Real-Time Prediction

Produces:

Probability of upward movement (P(up))

Technical indicator checks (SMA, MACD, RSI)

Final Recommendation:

STRONG BUY

NO BUY / AVOID

6. Model Saving & Loading

Trained models are saved using joblib and can be loaded anytime for prediction.

🗂 Project Structure
📦 NIFTY-Prediction-Project
│
├── data/
│   └── nifty_final_for_model.csv        # cleaned dataset
│
├── models/
│   └── xgb_model.pkl                    # trained XGBoost model
│
├── notebooks/
│   └── training_pipeline.ipynb          # feature engineering + model training
│   └── auto_predict.ipynb               # daily prediction engine
│
├── src/
│   ├── data_loader.py                   # fetch, clean, update data
│   ├── features.py                      # indicator & feature generator
│   ├── model_train.py                   # training script
│   └── predict.py                       # final prediction logic
│
└── README.md

🔧 Installation
1. Install dependencies
pip install pandas numpy xgboost scikit-learn joblib matplotlib yfinance

2. Run the training notebook

Contains:

Feature engineering

Train/validation split

Model training

Model saving

3. Run auto_predict

Fetches today’s NIFTY data and prints:

Latest available bar date: YYYY-MM-DD
Model P(up) = 0.4435  (threshold = 0.6)
TA checks: { ... }
Recommendation: NO BUY / AVOID

📊 Outputs Explained
Model Probability
P(up) → probability NIFTY closes higher tomorrow

Threshold Rule

If P(up) >= 0.6 → Strong Buy
Else → No Buy

Technical Filter

The system checks:

Price > SMA50

MACD positive

RSI not overbought

Final Recommendation

STRONG BUY → Consider long entry

NO BUY / AVOID → Stay out / reduce position

🔍 Example Output
Local data last date: 2025-11-24
Remote data available: 2025-11-24 to 2025-12-02
Found 6 new rows. Updating dataset...

Loaded model: xgb_model.pkl

---- AUTO PREDICT TOMORROW ----
Latest bar: 2025-12-02
Model P(up) = 0.4435  (threshold = 0.6)
TA checks: {'close_gt_sma50': True, 'macd_pos': True, 'rsi_ok': True}

Recommendation: NO BUY / AVOID
Advice: Use position sizing and stop-loss for risk control.

🧠 What You Can Improve Later

Includes future planned enhancements:

Add India VIX (volatility-based filter)

Add SHAP explainability

Add LSTM/Transformer model for comparison

Improve hyperparameter tuning

Add walk-forward validation

Build a Streamlit dashboard

📄 License

This project is open-source for educational and research purposes only.
Not intended for live trading without risk management & proper validation.

🙋‍♂️ Author

Pratik Mane
Machine Learning & Trading Enthusiast
Data Science • Algorithmic Trading • Flutter & Firebase Developer

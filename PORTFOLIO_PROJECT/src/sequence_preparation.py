# src/sequence_preparation.py

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle 
# ==============================
# PARAMETERS
# ==============================
SEQ_LEN = 30           # number of past days in each sequence
TEST_RATIO = 0.2       # last 20% for testing
RAW_PATH = "data/processed"
OUTPUT_PATH = "data/processed"
os.makedirs(OUTPUT_PATH, exist_ok=True)

FEATURE_COLS = [
    'Open','High','Low','Close','Volume',
    'Daily_Return','MA7','MA30',
    'Volatility_7','Volatility_14',
    'RSI','MACD','MACD_Signal','MACD_Hist','NIFTY_Return'
]

# ==============================
# LOAD FINAL DATASET
# ==============================
print("📂 Loading final dataset...")
final_df = pd.read_csv(os.path.join(RAW_PATH, "final_dataset.csv"), parse_dates=['Date'])
final_df = final_df.sort_values(['Ticker','Date']).reset_index(drop=True)

# ==============================
# CREATE SEQUENCES
# ==============================
X_all, y_all = [], []
scalers = {}

tickers = final_df['Ticker'].unique()
for ticker in tickers:
    df_t = final_df[final_df['Ticker']==ticker].copy()
    
    # Features
    df_features = df_t[FEATURE_COLS]
    
    # Normalize
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df_features)
    scalers[ticker] = scaler
    
    # Target
    y = df_t['Target'].values
    
    # Create sequences
    for i in range(SEQ_LEN, len(df_scaled)):
        X_all.append(df_scaled[i-SEQ_LEN:i])
        y_all.append(y[i])

# Convert to numpy arrays
X_all = np.array(X_all)
y_all = np.array(y_all)

print(f"✅ Total sequences: {len(X_all)}")
print(f"Shape of X_all: {X_all.shape}")
print(f"Shape of y_all: {y_all.shape}")

# ==============================
# TRAIN-TEST SPLIT
# ==============================
num_samples = len(X_all)
split_idx = int(num_samples * (1 - TEST_RATIO))

X_train, X_test = X_all[:split_idx], X_all[split_idx:]
y_train, y_test = y_all[:split_idx], y_all[split_idx:]

print("Train shapes:", X_train.shape, y_train.shape)
print("Test shapes:", X_test.shape, y_test.shape)

# ==============================
# SAVE ARRAYS FOR MODELING
# ==============================
np.save(os.path.join(OUTPUT_PATH, "X_train.npy"), X_train)
np.save(os.path.join(OUTPUT_PATH, "y_train.npy"), y_train)
np.save(os.path.join(OUTPUT_PATH, "X_test.npy"), X_test)
np.save(os.path.join(OUTPUT_PATH, "y_test.npy"), y_test)

# ==============================
# SAVE SCALERS FOR LATER USE
# ==============================
print("💾 Saving scalers for each ticker...")
with open(os.path.join(OUTPUT_PATH, "scalers.pkl"), 'wb') as f:
    pickle.dump(scalers, f)
print(f"✅ Scalers saved for {len(scalers)} tickers")
print(f"📁 Scalers file: {os.path.join(OUTPUT_PATH, 'scalers.pkl')}")
print(f"🎉 Arrays saved to {OUTPUT_PATH}")

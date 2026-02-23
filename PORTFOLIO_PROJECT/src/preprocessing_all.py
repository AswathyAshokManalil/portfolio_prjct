import os
import pandas as pd
import numpy as np
import yfinance as yf

# ==============================
# PATH SETTINGS
# ==============================
RAW_PATH = "data/raw"          # folder containing all CSV files
PROCESSED_PATH = "data/processed"
os.makedirs(PROCESSED_PATH, exist_ok=True)

# ==============================
# FUNCTION: ADD TECHNICAL INDICATORS
# ==============================
def add_technical_indicators(df):
    df['Daily_Return'] = df['Close'].pct_change()
    df['MA7'] = df['Close'].rolling(7).mean()
    df['MA30'] = df['Close'].rolling(30).mean()
    df['Volatility_7'] = df['Daily_Return'].rolling(7).std()
    df['Volatility_14'] = df['Daily_Return'].rolling(14).std()

    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    return df

# ==============================
# STEP 1: LOAD & CLEAN NIFTY
# ==============================
print("📊 Processing NIFTY50...")

nifty = pd.read_csv(os.path.join(RAW_PATH, "nifty50.csv"), skiprows=2)
nifty.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
nifty['Date'] = pd.to_datetime(nifty['Date'], dayfirst=True, errors='coerce')
nifty = nifty.dropna(subset=['Date'])
nifty = nifty.set_index('Date')
nifty['NIFTY_Return'] = nifty['Close'].pct_change()
nifty = nifty[['NIFTY_Return']]  # keep only the return

print("✅ NIFTY processed successfully")

# ==============================
# STEP 2: PROCESS ALL STOCKS
# ==============================
all_data = []

print("\n📦 Processing all stock files...\n")

for file in os.listdir(RAW_PATH):
    if file.endswith(".csv") and file.lower() != "nifty50.csv":
        print(f"Processing {file}...")
        try:
            df = pd.read_csv(os.path.join(RAW_PATH, file), skiprows=2)
            df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
            df = df.dropna(subset=['Date'])
            df = df.set_index('Date')
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            df = df.apply(pd.to_numeric, errors='coerce')
            df = df.sort_index().dropna()

            # Add technical indicators
            df = add_technical_indicators(df)

            # Merge NIFTY return using LEFT join to keep all stock rows
            df = df.merge(nifty, how='left', left_index=True, right_index=True)
            df['NIFTY_Return'] = df['NIFTY_Return'].fillna(method='ffill')  # forward-fill missing market data

            # Target: next day price up or not
            df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

            # Add ticker column
            ticker = file.replace(".csv", "")
            df['Ticker'] = ticker

            # Drop only rows with NaNs in critical features
            df = df.dropna(subset=['MA7', 'MA30', 'RSI', 'MACD', 'NIFTY_Return', 'Target'])

            all_data.append(df)

        except Exception as e:
            print(f"❌ Error processing {file}: {e}")

# ==============================
# STEP 3: COMBINE ALL STOCKS
# ==============================
if not all_data:
    raise ValueError("❌ No valid stock data processed!")

final_dataset = pd.concat(all_data)
final_dataset = final_dataset.reset_index()

# Save final dataset
final_path = os.path.join(PROCESSED_PATH, "final_dataset.csv")
final_dataset.to_csv(final_path, index=False)

print("\n🎉 All stocks processed successfully!")
print("Final dataset shape:", final_dataset.shape)
print(f"📁 Saved to: {final_path}")

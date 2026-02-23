
#create 50 seperate csv file for each companies. This is the actual data.
# src/data_collection.py
import yfinance as yf
import os

# Path to save CSVs
raw_data_path = 'data/raw'
os.makedirs(raw_data_path, exist_ok=True)  # create folder if it doesn't exist

# List of NIFTY 50 tickers
nifty50_tickers = [
    'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'HINDUNILVR.NS',
    'ICICIBANK.NS', 'KOTAKBANK.NS', 'SBIN.NS', 'ITC.NS', 'AXISBANK.NS',
    'LT.NS', 'BAJAJFINSV.NS', 'BHARTIARTL.NS', 'MARUTI.NS', 'HCLTECH.NS',
    'ASIANPAINT.NS', 'NTPC.NS', 'SUNPHARMA.NS', 'WIPRO.NS', 'TITAN.NS',
    'TECHM.NS', 'ULTRACEMCO.NS', 'ONGC.NS', 'POWERGRID.NS', 'NESTLEIND.NS',
    'BRITANNIA.NS', 'GRASIM.NS', 'DIVISLAB.NS', 'HDFCLIFE.NS', 'JSWSTEEL.NS',
    'BAJAJ-AUTO.NS', 'COALINDIA.NS', 'HEROMOTOCO.NS', 'TATASTEEL.NS', 'BPCL.NS',
    'EICHERMOT.NS', 'DRREDDY.NS', 'HINDALCO.NS', 'M&M.NS', 'TATAMOTORS.NS',
    'ADANIPORTS.NS', 'SBILIFE.NS', 'IOC.NS', 'INDUSINDBK.NS', 'HDFC.NS',
    'ULTRACEMCO.NS', 'SHREECEM.NS', 'APOLLOHOSP.NS', 'TECHM.NS', 'CIPLA.NS',
    'VEDL.NS', 'BPCL.NS'
]

# Download data for each ticker
for ticker in nifty50_tickers:
    print(f'Downloading {ticker}...')
    df = yf.download(ticker, start='2015-01-01', end='2024-01-01')
    
    # Save as CSV
    csv_file = os.path.join(raw_data_path, f'{ticker}.csv')
    df.to_csv(csv_file)
    print(f'{ticker} saved to {csv_file}')

print('All NIFTY 50 stock data downloaded successfully!')

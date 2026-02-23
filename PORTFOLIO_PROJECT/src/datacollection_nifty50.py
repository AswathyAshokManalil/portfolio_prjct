import yfinance as yf

nifty = yf.download("^NSEI", start="2015-01-01", end="2023-12-31")

nifty.to_csv("data/raw/nifty50.csv")
#stock prices are influenced by both company-specific factors and overall market conditions.
#  By adding the NIFTY_Return feature, the model can understand how much of a stock’s movement is
#  due to the general market trend and how much is due to the stock’s own performance. This helps 
# the model separate the market effect (how the overall market moved today) from the stock-specific
#  effect (how the individual company performed relative to the market).
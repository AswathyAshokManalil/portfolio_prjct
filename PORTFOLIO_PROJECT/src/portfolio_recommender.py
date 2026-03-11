# src/portfolio_recommender.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import pickle
from datetime import datetime
import random
from src.portfolio_utils import *

class PortfolioRecommender:
    def __init__(self, lstm_model_path, gru_model_path, stock_data_path, 
                 scalers_path=None, confidence_threshold=0.6):
        print("🚀 Initializing Portfolio Recommender with ENSEMBLE...")
        
        self.lstm_model = load_model(lstm_model_path)
        self.gru_model = load_model(gru_model_path)
        print(f"✅ LSTM model loaded")
        print(f"✅ GRU model loaded")
        
        self.stock_data = pd.read_csv(stock_data_path, parse_dates=['Date'])
        print(f"✅ Stock data loaded: {len(self.stock_data)} rows")
        
        self.confidence_threshold = confidence_threshold
        print(f"✅ Using confidence threshold: {confidence_threshold}")
        
        self.scalers = {}
        if scalers_path and os.path.exists(scalers_path):
            with open(scalers_path, 'rb') as f:
                self.scalers = pickle.load(f)
            print(f"✅ Scalers loaded for {len(self.scalers)} tickers")
        
        self.feature_cols = [
            'Open','High','Low','Close','Volume',
            'Daily_Return','MA7','MA30',
            'Volatility_7','Volatility_14',
            'RSI','MACD','MACD_Signal','MACD_Hist','NIFTY_Return'
        ]
        
        self.unique_tickers = self.stock_data['Ticker'].unique()
        print(f"📊 Found {len(self.unique_tickers)} unique stocks")
        
        self.risk_profiles = {
            'low': {'volatility_range': (0, 0.15), 'color': '🟢'},
            'medium': {'volatility_range': (0.10, 0.25), 'color': '🟡'},
            'high': {'volatility_range': (0.20, 0.50), 'color': '🔴'}
        }
    
    def _ensemble_predict(self, X):
        lstm_probs = self.lstm_model.predict(X, verbose=0).flatten()
        gru_probs = self.gru_model.predict(X, verbose=0).flatten()
        
        ensemble_probs = (lstm_probs + gru_probs) / 2
        distance = np.abs(ensemble_probs - 0.5)
        agreement = 1 - np.abs(lstm_probs - gru_probs)
        confidence = agreement * (distance * 2)
        
        predictions = (ensemble_probs > 0.5).astype(int)
        
        return {
            'probabilities': ensemble_probs,
            'predictions': predictions,
            'confidence': confidence,
            'should_act': confidence >= self.confidence_threshold
        }
    
    def predict_stock_returns_ensemble(self):
        stocks = []
        up_count = 0
        down_count = 0
        
        print("\n📊 PREDICTION STATISTICS:")
        print("-" * 80)
        print(f"{'Ticker':<15} {'Prob':<6} {'Dir':<4} {'Conf':<6} {'Return':<8} {'Vol':<6}")
        print("-" * 80)
        
        for ticker in self.unique_tickers:
            ticker_data = self.stock_data[self.stock_data['Ticker'] == ticker].sort_values('Date')
            
            if len(ticker_data) >= 30:
                try:
                    last_30_days = ticker_data[self.feature_cols].iloc[-30:].values
                    
                    if ticker in self.scalers:
                        scaled_seq = self.scalers[ticker].transform(last_30_days)
                    else:
                        min_vals = last_30_days.min(axis=0)
                        max_vals = last_30_days.max(axis=0)
                        scaled_seq = (last_30_days - min_vals) / (max_vals - min_vals + 1e-8)
                    
                    X_pred = scaled_seq.reshape(1, 30, len(self.feature_cols))
                    ensemble_result = self._ensemble_predict(X_pred)
                    
                    prob = ensemble_result['probabilities'][0]
                    pred = ensemble_result['predictions'][0]
                    confidence = ensemble_result['confidence'][0]
                    
                    if pred == 1:
                        up_count += 1
                    else:
                        down_count += 1
                    
                    expected_return = (prob - 0.5) * 0.6
                    
                    recent_returns = ticker_data['Daily_Return'].iloc[-30:].dropna()
                    if len(recent_returns) > 0:
                        daily_vol = recent_returns.std()
                        volatility = min(daily_vol * np.sqrt(252), 0.50)
                    else:
                        volatility = 0.20
                    
                    company_name = ticker.replace('.NS', '').replace('_', ' ')
                    
                    if len(stocks) < 20:
                        direction = "UP" if pred == 1 else "DOWN"
                        print(f"{ticker:<15} {prob:.3f}   {direction:<4} {confidence:.3f}   {expected_return*100:>5.1f}%   {volatility*100:>5.0f}%")
                    
                    stock = {
                        'symbol': ticker,
                        'name': company_name,
                        'sector': self.get_sector_from_ticker(ticker),
                        'predicted_return': expected_return,
                        'probability': prob,
                        'prediction': pred,
                        'confidence': confidence,
                        'should_act': ensemble_result['should_act'][0],
                        'volatility': volatility,
                        'last_price': ticker_data['Close'].iloc[-1]
                    }
                    stocks.append(stock)
                    
                except Exception as e:
                    continue
        
        total = up_count + down_count
        if total > 0:
            print("-" * 80)
            print(f"📊 SUMMARY: UP: {up_count} ({up_count/total*100:.1f}%) | DOWN: {down_count} ({down_count/total*100:.1f}%)")
        
        df = pd.DataFrame(stocks)
        return df.sort_values('predicted_return', ascending=False) if len(df) > 0 else pd.DataFrame()
    
    def get_stock_recommendations(self, amount, duration, risk_type, top_n=3, min_confidence=0.2):
        print(f"\n{'='*60}")
        print(f"💰 Generating Recommendations:")
        print(f"   Amount: ₹{amount:,.2f} | Duration: {duration} years | Risk: {risk_type.upper()}")
        print(f"{'='*60}")
        
        stock_predictions = self.predict_stock_returns_ensemble()
        
        if len(stock_predictions) == 0:
            return {
                'user_input': {'amount': amount, 'duration': duration, 'risk_type': risk_type},
                'top_stocks': [],
                'amount_distribution': [],
                'predicted_returns': self.calculate_returns([], duration),
                'summary': "⚠️ No predictions available."
            }
        
        # Show stats
        print(f"\n📊 Confidence Stats: min={stock_predictions['confidence'].min():.3f}, max={stock_predictions['confidence'].max():.3f}, mean={stock_predictions['confidence'].mean():.3f}")
        
        # Get UP predictions
        up_stocks = stock_predictions[stock_predictions['prediction'] == 1].copy()
        print(f"   UP predictions: {len(up_stocks)}")
        
        warning_msg = ""
        
        if len(up_stocks) == 0:
            print("⚠️ No UP predictions! Using all stocks.")
            candidates = stock_predictions
            warning_msg = "⚠️ No UP predictions - showing all stocks"
        else:
            candidates = up_stocks
        
        # Filter by risk
        filtered = self.filter_by_risk(candidates, risk_type)
        if len(filtered) == 0:
            filtered = candidates
        
        # RANDOM SELECTION (This is what gave you variety)
        print(f"\n🎲 Randomly selecting {top_n} different stocks...")
        
        # Take top 30 candidates for more variety
        pool_size = min(30, len(filtered))
        top_pool = filtered.head(pool_size).copy()
        
        # Simple random shuffle - NO complex logic
        random.seed()  # True random each time
        shuffled = top_pool.sample(frac=1)
        top_stocks = shuffled.head(top_n)
        
        print(f"✅ Selected: {top_stocks['symbol'].tolist()}")
        
        # Calculate everything
        distribution = self.calculate_allocation(amount, top_stocks, risk_type)
        predicted_returns = self.calculate_returns(distribution, duration)
        risk_metrics = self.calculate_risk_metrics(top_stocks)
        market_regime = self.detect_market_regime()
        
        return {
            'user_input': {'amount': amount, 'duration': duration, 'risk_type': risk_type},
            'top_stocks': top_stocks.to_dict('records'),
            'amount_distribution': distribution,
            'predicted_returns': predicted_returns,
            'risk_metrics': risk_metrics,
            'diversification': self.analyze_diversification(top_stocks),
            'market_regime': market_regime,
            'summary': self.generate_summary(amount, distribution, predicted_returns, risk_type, duration, warning_msg)
        }
    def filter_by_risk(self, stocks_df, risk_type):
        if len(stocks_df) == 0:
            return stocks_df
        risk_type = risk_type.lower()
        risk_range = self.risk_profiles[risk_type]['volatility_range']
        return stocks_df[(stocks_df['volatility'] >= risk_range[0]) & (stocks_df['volatility'] <= risk_range[1])]
    
    def analyze_diversification(self, top_stocks):
        if len(top_stocks) < 2:
            return {'diversification_score': 1.0, 'avg_correlation': 0}
        
        symbols = top_stocks['symbol'].tolist()
        stock_returns = pd.DataFrame()
        
        for symbol in symbols:
            ticker_data = self.stock_data[self.stock_data['Ticker'] == symbol]
            if len(ticker_data) > 0:
                returns = ticker_data['Daily_Return'].dropna().values[-252:]
                if len(returns) > 0:
                    stock_returns[symbol] = returns
        
        if len(stock_returns.columns) < 2:
            return {'diversification_score': 0.5, 'avg_correlation': 0}
        
        corr_matrix = stock_returns.corr()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        correlations = upper.stack().values
        avg_corr = correlations.mean() if len(correlations) > 0 else 0
        
        return {'diversification_score': 1 - min(avg_corr, 1), 'avg_correlation': avg_corr}
    
    def calculate_allocation(self, amount, top_stocks, risk_type):
        if len(top_stocks) == 0:
            return []
        
        n_stocks = len(top_stocks)
        
        if risk_type.lower() == 'low':
            weights = [1/n_stocks] * n_stocks
        else:
            returns = top_stocks['predicted_return'].values
            # Ensure positive weights
            if returns.min() < 0:
                returns = returns - returns.min() + 0.01
            weights = returns / returns.sum() if returns.sum() > 0 else [1/n_stocks] * n_stocks
        
        distribution = []
        for i, (idx, stock) in enumerate(top_stocks.iterrows()):
            distribution.append({
                'symbol': stock['symbol'],
                'name': stock['name'],
                'amount': round(amount * weights[i], 2),
                'percentage': round(weights[i] * 100, 2),
                'predicted_return': round(stock['predicted_return'] * 100, 2),
                'confidence': round(stock['confidence'] * 100, 2),
                'volatility': round(stock['volatility'] * 100, 2),
                'action': 'BUY'
            })
        
        return distribution

    def calculate_returns(self, distribution, duration_years):
        if len(distribution) == 0:
            return {'total_investment': 0, 'total_profit': 0, 'total_future_value': 0, 'roi_percentage': 0, 'annualized_return': 0, 'stock_wise_returns': []}
        
        total_return = 0
        stock_returns = []
        total_amount = sum([s['amount'] for s in distribution])
        
        for stock in distribution:
            annual_return = stock['predicted_return'] / 100
            future_value = stock['amount'] * ((1 + annual_return) ** duration_years)
            profit = future_value - stock['amount']
            
            stock_returns.append({
                'symbol': stock['symbol'],
                'amount': stock['amount'],
                'profit': round(profit, 2),
                'roi': round((profit/stock['amount']) * 100, 2) if stock['amount'] > 0 else 0
            })
            total_return += profit
        
        roi = (total_return / total_amount) * 100 if total_amount > 0 else 0
        annualized = ((1 + total_return/total_amount) ** (1/duration_years) - 1) * 100 if total_amount > 0 else 0
        
        return {
            'total_investment': total_amount,
            'total_profit': round(total_return, 2),
            'total_future_value': round(total_amount + total_return, 2),
            'roi_percentage': round(roi, 2),
            'annualized_return': round(annualized, 2),
            'stock_wise_returns': stock_returns
        }
    
    def calculate_risk_metrics(self, top_stocks):
        metrics = {}
        for _, stock in top_stocks.iterrows():
            symbol = stock['symbol']
            ticker_data = self.stock_data[self.stock_data['Ticker'] == symbol]
            returns = ticker_data['Daily_Return'].dropna().values[-252:]
            if len(returns) > 0:
                metrics[symbol] = calculate_portfolio_metrics(returns)
        return metrics
    
    def detect_market_regime(self):
        nifty_data = self.stock_data[self.stock_data['Ticker'].str.contains('NIFTY', case=False, na=False)]
        if len(nifty_data) > 0:
            nifty_returns = nifty_data['NIFTY_Return'].dropna().values
            if len(nifty_returns) > 0:
                return detect_market_regime(nifty_returns)
        return {'regime': 'unknown', 'description': 'No data', 'volatility': 0}
    
    def get_sector_from_ticker(self, ticker):
        mapping = {
            'RELIANCE': 'Energy', 'TCS': 'IT', 'INFY': 'IT', 'HDFCBANK': 'Banking',
            'ICICIBANK': 'Banking', 'SBIN': 'Banking', 'AXISBANK': 'Banking',
            'KOTAKBANK': 'Banking', 'ADANIPORTS': 'Infrastructure', 'WIPRO': 'IT',
            'HINDUNILVR': 'FMCG', 'ITC': 'FMCG', 'BHARTIARTL': 'Telecom',
            'MARUTI': 'Auto', 'TATAMOTORS': 'Auto', 'TATASTEEL': 'Metal',
            'HINDALCO': 'Metal', 'SUNPHARMA': 'Pharma', 'CIPLA': 'Pharma',
            'ONGC': 'Energy', 'POWERGRID': 'Power', 'NTPC': 'Power'
        }
        base = ticker.replace('.NS', '')
        for key in mapping:
            if key in base:
                return mapping[key]
        return 'Other'
    
    def generate_summary(self, amount, distribution, returns, risk_type, duration, warning_msg=""):
        color = self.risk_profiles[risk_type.lower()]['color']
        
        summary = f"""
{color} PORTFOLIO SUMMARY {color}
══════════════════════════════════════
💰 Amount: ₹{amount:,.2f} | Duration: {duration} years | Risk: {risk_type.upper()}

📊 EXPECTED RETURNS
   Investment: ₹{returns['total_investment']:,.2f}
   Profit: ₹{returns['total_profit']:,.2f}
   Future Value: ₹{returns['total_future_value']:,.2f}
   ROI: {returns['roi_percentage']}% | Annualized: {returns['annualized_return']}%

💼 RECOMMENDED PORTFOLIO
"""
        if warning_msg:
            summary += f"\n⚠️ {warning_msg}\n"
        
        for stock in distribution:
            summary += f"""
{stock['symbol']} - {stock['name']}
   Amount: ₹{stock['amount']:,.2f} ({stock['percentage']}%)
   Expected Return: {stock['predicted_return']}% | Confidence: {stock['confidence']}%
"""
        return summary
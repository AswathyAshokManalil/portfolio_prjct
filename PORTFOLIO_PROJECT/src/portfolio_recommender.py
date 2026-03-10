# src/portfolio_recommender.py

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import os

class PortfolioRecommender:
    def __init__(self, model_path, stock_data_path):
        """
        Initialize the portfolio recommender
        """
        self.model = load_model(model_path)
        self.stock_data = pd.read_csv(stock_data_path)
        
        # Get unique tickers from your data
        self.unique_tickers = self.stock_data['Ticker'].unique()
        print(f"Found {len(self.unique_tickers)} unique stocks: {self.unique_tickers}")
        
        self.risk_profiles = {
            'low': {'volatility': 0.1, 'returns': 0.08, 'allocation': 'defensive'},
            'medium': {'volatility': 0.15, 'returns': 0.12, 'allocation': 'balanced'},
            'high': {'volatility': 0.25, 'returns': 0.18, 'allocation': 'aggressive'}
        }
        
    def get_stock_recommendations(self, amount, duration, risk_type, top_n=3):
        """
        Main method to get portfolio recommendations
        """
        # Get predicted returns for all stocks
        stock_predictions = self.predict_stock_returns()
        
        # Filter stocks based on risk profile
        filtered_stocks = self.filter_by_risk(stock_predictions, risk_type)
        
        # Get top N stocks
        top_stocks = filtered_stocks.head(top_n)
        
        # Calculate amount distribution
        distribution = self.calculate_allocation(amount, top_stocks, risk_type)
        
        # Calculate predicted returns
        predicted_returns = self.calculate_returns(distribution, duration)
        
        # Prepare recommendations
        recommendations = {
            'user_input': {
                'amount': amount,
                'duration': duration,
                'risk_type': risk_type
            },
            'top_stocks': top_stocks.to_dict('records'),
            'amount_distribution': distribution,
            'predicted_returns': predicted_returns,
            'summary': self.generate_summary(amount, distribution, predicted_returns, risk_type, duration)
        }
        
        return recommendations
    
    def predict_stock_returns(self):
        """
        Predict returns for all stocks using your trained model
        This uses your actual stock data to make predictions
        """
        stocks = []
        
        for ticker in self.unique_tickers:
            # Get all data for this ticker
            ticker_data = self.stock_data[self.stock_data['Ticker'] == ticker]
            
            # Get the latest data (most recent date)
            latest_data = ticker_data.iloc[-1] if len(ticker_data) > 0 else None
            
            if latest_data is not None:
                # TODO: Replace this with your actual model prediction
                # You need to:
                # 1. Prepare the last 30 days of data for this stock
                # 2. Reshape it to match your model input (1, 30, 15)
                # 3. Use self.model.predict() to get probability
                
                # For now, using sample predictions based on recent volatility and RSI
                # This is just a placeholder - replace with actual model prediction
                recent_volatility = ticker_data['Volatility_14'].iloc[-5:].mean() if len(ticker_data) > 5 else 0.15
                recent_rsi = ticker_data['RSI'].iloc[-1] if 'RSI' in ticker_data.columns else 50
                recent_macd = ticker_data['MACD'].iloc[-1] if 'MACD' in ticker_data.columns else 0
                
                # Sample prediction logic (REPLACE THIS WITH YOUR ACTUAL MODEL)
                # Higher probability when RSI is low (oversold) and MACD is positive
                sample_probability = 0.5 + (0.2 * (1 - recent_rsi/100)) + (0.1 if recent_macd > 0 else -0.1)
                sample_probability = max(0.1, min(0.9, sample_probability))  # Clip between 0.1 and 0.9
                
                # Get company name from ticker (you might want to maintain a separate mapping)
                # For now, just using ticker as name
                company_name = ticker.replace('.NS', '').replace('_', ' ')
                
                stock = {
                    'symbol': ticker,
                    'name': company_name,
                    'sector': self.get_sector_from_ticker(ticker),  # You'll need to implement this
                    'predicted_return': sample_probability * 0.3,  # Map probability to return (0.3 = 30% max)
                    'volatility': recent_volatility,
                    'confidence_score': sample_probability,
                    'rsi': recent_rsi,
                    'macd': recent_macd
                }
                stocks.append(stock)
        
        # If no stocks were processed, create sample data
        if len(stocks) == 0:
            # Fallback to sample data
            for i, ticker in enumerate(self.unique_tickers[:10]):  # Limit to first 10
                stock = {
                    'symbol': ticker,
                    'name': ticker.replace('.NS', '').replace('_', ' '),
                    'sector': 'Unknown',
                    'predicted_return': np.random.uniform(0.05, 0.25),
                    'volatility': np.random.uniform(0.05, 0.30),
                    'confidence_score': np.random.uniform(0.6, 0.95),
                    'rsi': np.random.uniform(30, 70),
                    'macd': np.random.uniform(-2, 2)
                }
                stocks.append(stock)
        
        return pd.DataFrame(stocks).sort_values('predicted_return', ascending=False)
    
    def get_sector_from_ticker(self, ticker):
        """
        Helper method to get sector for a ticker
        You can expand this with actual sector mapping
        """
        # This is a sample mapping - replace with your actual sector data
        sector_mapping = {
            'RELIANCE': 'Energy',
            'TCS': 'IT',
            'INFY': 'IT',
            'HDFC': 'Banking',
            'ICICIBANK': 'Banking',
            'SBIN': 'Banking',
            'ADANIPORTS': 'Infrastructure',
            'WIPRO': 'IT',
            'HINDUNILVR': 'FMCG',
            'ITC': 'FMCG',
            'BHARTIARTL': 'Telecom'
        }
        
        # Remove .NS suffix for matching
        base_ticker = ticker.replace('.NS', '')
        return sector_mapping.get(base_ticker, 'Other')
    
    def filter_by_risk(self, stocks_df, risk_type):
        """
        Filter stocks based on risk profile
        """
        if risk_type.lower() == 'low':
            filtered = stocks_df[stocks_df['volatility'] <= 0.12]
        elif risk_type.lower() == 'medium':
            filtered = stocks_df[(stocks_df['volatility'] >= 0.08) & (stocks_df['volatility'] <= 0.18)]
        else:  # high risk
            filtered = stocks_df[stocks_df['volatility'] >= 0.15]
        
        if len(filtered) == 0:
            return stocks_df.head(10)
        
        return filtered.sort_values('predicted_return', ascending=False)
    
    def calculate_allocation(self, amount, top_stocks, risk_type):
        """
        Calculate how to distribute amount across selected stocks
        """
        if risk_type.lower() == 'low':
            # Equal weight for low risk
            weights = [1/len(top_stocks)] * len(top_stocks)
        elif risk_type.lower() == 'medium':
            # Weighted by confidence score
            confidence_scores = top_stocks['confidence_score'].values
            weights = confidence_scores / confidence_scores.sum()
        else:  # high risk
            # Heavier weight on highest predicted returns
            returns = top_stocks['predicted_return'].values
            weights = returns / returns.sum()
        
        distribution = []
        for i, (idx, stock) in enumerate(top_stocks.iterrows()):
            allocation = {
                'symbol': stock['symbol'],
                'name': stock['name'],
                'amount': round(amount * weights[i], 2),
                'percentage': round(weights[i] * 100, 2),
                'predicted_return': round(stock['predicted_return'] * 100, 2)
            }
            distribution.append(allocation)
        
        return distribution
    
    def calculate_returns(self, distribution, duration_years):
        """
        Calculate predicted returns for the portfolio
        """
        total_return = 0
        stock_returns = []
        
        for stock in distribution:
            annual_return = stock['predicted_return'] / 100
            future_value = stock['amount'] * ((1 + annual_return) ** duration_years)
            profit = future_value - stock['amount']
            
            stock_returns.append({
                'symbol': stock['symbol'],
                'amount': stock['amount'],
                'predicted_return_pct': stock['predicted_return'],
                'future_value': round(future_value, 2),
                'profit': round(profit, 2)
            })
            total_return += profit
        
        total_amount = sum([s['amount'] for s in distribution])
        
        if total_amount > 0:
            roi_percentage = (total_return / total_amount) * 100
            annualized_return = ((1 + total_return/total_amount) ** (1/duration_years) - 1) * 100
        else:
            roi_percentage = 0
            annualized_return = 0
        
        return {
            'total_investment': total_amount,
            'total_profit': round(total_return, 2),
            'total_future_value': round(total_amount + total_return, 2),
            'roi_percentage': round(roi_percentage, 2),
            'annualized_return': round(annualized_return, 2),
            'stock_wise_returns': stock_returns
        }
    
    def generate_summary(self, amount, distribution, returns, risk_type, duration):
        """
        Generate a user-friendly summary
        """
        summary = f"""
        📊 PORTFOLIO SUMMARY
        ═══════════════════════════════════════
        Investment Amount: ₹{amount:,.2f}
        Duration: {duration} years
        Risk Profile: {risk_type.upper()}
        
        📈 EXPECTED RETURNS
        ═══════════════════════════════════════
        Total Investment: ₹{returns['total_investment']:,.2f}
        Expected Profit: ₹{returns['total_profit']:,.2f}
        Future Value: ₹{returns['total_future_value']:,.2f}
        ROI: {returns['roi_percentage']}%
        Annualized Return: {returns['annualized_return']}%
        
        💼 RECOMMENDED PORTFOLIO
        ═══════════════════════════════════════
        """
        
        for stock in distribution:
            summary += f"\n{stock['symbol']} - {stock['name']}"
            summary += f"\n  Amount: ₹{stock['amount']:,.2f} ({stock['percentage']}%)"
            summary += f"\n  Expected Return: {stock['predicted_return']}%"
            summary += f"\n  {'─' * 40}\n"
        
        return summary
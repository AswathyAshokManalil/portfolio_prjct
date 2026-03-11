"""
Utility functions for portfolio analysis and risk management
"""

import numpy as np
import pandas as pd
from scipy import stats

# ============================================
# RISK METRICS FUNCTIONS
# ============================================

def calculate_var(returns, confidence_level=0.95):
    """
    Calculate Value at Risk (VaR)
    """
    if len(returns) == 0:
        return 0
    return np.percentile(returns, (1 - confidence_level) * 100)

def calculate_cvar(returns, confidence_level=0.95):
    """
    Calculate Conditional Value at Risk (Expected Shortfall)
    """
    if len(returns) == 0:
        return 0
    var = calculate_var(returns, confidence_level)
    return returns[returns <= var].mean()

def calculate_max_drawdown(returns):
    """
    Calculate maximum drawdown from returns series
    FIXED: Now works with both numpy arrays and pandas Series
    """
    if len(returns) == 0:
        return 0
    
    # Convert to pandas Series if it's a numpy array
    if isinstance(returns, np.ndarray):
        returns = pd.Series(returns)
    
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return abs(drawdown.min())

def calculate_sharpe_ratio(returns, risk_free_rate=0.05):
    """
    Calculate Sharpe Ratio (annualized)
    """
    if len(returns) == 0:
        return 0
    returns_std = returns.std()
    if returns_std == 0:
        return 0
    excess_returns = returns.mean() * 252 - risk_free_rate
    return excess_returns / (returns_std * np.sqrt(252))

def calculate_sortino_ratio(returns, risk_free_rate=0.05, target=0):
    """
    Calculate Sortino Ratio (only considers downside deviation)
    """
    if len(returns) == 0:
        return 0
    downside_returns = returns[returns < target]
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0
    excess_return = returns.mean() * 252 - risk_free_rate
    downside_deviation = downside_returns.std() * np.sqrt(252)
    return excess_return / downside_deviation

def calculate_portfolio_metrics(returns):
    """
    Calculate all risk metrics for a portfolio
    FIXED: Converts numpy array to pandas Series if needed
    """
    # Convert to pandas Series if it's a numpy array
    if isinstance(returns, np.ndarray):
        returns = pd.Series(returns)
    
    return {
        'var_95': calculate_var(returns, 0.95),
        'cvar_95': calculate_cvar(returns, 0.95),
        'max_drawdown': calculate_max_drawdown(returns),
        'sharpe_ratio': calculate_sharpe_ratio(returns),
        'sortino_ratio': calculate_sortino_ratio(returns),
        'annual_return': returns.mean() * 252,
        'annual_volatility': returns.std() * np.sqrt(252)
    }

# ============================================
# DIVERSIFICATION ANALYSIS
# ============================================

def analyze_diversification(stock_returns, stock_symbols):
    """
    Analyze correlation between selected stocks
    """
    if len(stock_symbols) < 2:
        return {
            'diversification_score': 1.0, 
            'avg_correlation': 0,
            'warnings': ['Only one stock selected - no diversification analysis needed'],
            'high_correlation_warnings': []
        }
    
    # Calculate correlation matrix
    corr_matrix = stock_returns[stock_symbols].corr()
    
    # Calculate average correlation (lower is better for diversification)
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    correlations = upper_triangle.stack().values
    avg_correlation = correlations.mean() if len(correlations) > 0 else 0
    
    # Diversification score (0-1, higher is better)
    diversification_score = 1 - min(avg_correlation, 1) if not pd.isna(avg_correlation) else 0.5
    
    # Identify highly correlated pairs
    high_corr_pairs = []
    warnings = []
    
    for i in range(len(stock_symbols)):
        for j in range(i+1, len(stock_symbols)):
            corr = corr_matrix.iloc[i, j]
            if abs(corr) > 0.7:
                warning = f"⚠️ {stock_symbols[i]} and {stock_symbols[j]} are highly correlated ({corr:.2f})"
                warnings.append(warning)
                high_corr_pairs.append({
                    'stock1': stock_symbols[i],
                    'stock2': stock_symbols[j],
                    'correlation': corr
                })
    
    return {
        'diversification_score': diversification_score,
        'avg_correlation': avg_correlation,
        'correlation_matrix': corr_matrix,
        'warnings': warnings,
        'high_correlation_warnings': high_corr_pairs
    }

# ============================================
# MARKET REGIME DETECTION
# ============================================

def detect_market_regime(nifty_returns, lookback=60):
    """
    Detect current market regime (bull/bear/sideways)
    """
    if nifty_returns is None or len(nifty_returns) == 0:
        return {
            'regime': 'unknown',
            'description': 'No market data available',
            'trend_strength': 0,
            'volatility': 0,
            'r_squared': 0
        }
    
    if len(nifty_returns) < lookback:
        lookback = len(nifty_returns)
    
    if lookback < 20:  # Need minimum data
        return {
            'regime': 'unknown',
            'description': 'Insufficient data for regime detection',
            'trend_strength': 0,
            'volatility': 0,
            'r_squared': 0
        }
    
    recent_returns = nifty_returns[-lookback:]
    
    # Calculate trend strength (slope of linear regression)
    x = np.arange(len(recent_returns))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, recent_returns)
    trend_strength = slope * lookback  # Normalized trend
    
    # Calculate recent volatility
    volatility = recent_returns.std() * np.sqrt(252)
    
    # Classify regime
    if trend_strength > 0.05 and volatility < 0.20:
        regime = "bull"
        description = "Strong uptrend with moderate volatility"
    elif trend_strength > 0.02 and volatility < 0.25:
        regime = "moderate_bull"
        description = "Moderate uptrend"
    elif trend_strength < -0.03 and volatility > 0.20:
        regime = "bear"
        description = "Downtrend with high volatility"
    elif trend_strength < -0.01:
        regime = "mild_bear"
        description = "Slight downtrend"
    else:
        regime = "sideways"
        description = "No clear trend"
    
    return {
        'regime': regime,
        'description': description,
        'trend_strength': trend_strength,
        'volatility': volatility,
        'r_squared': r_value ** 2
    }

# ============================================
# REBALANCING LOGIC
# ============================================

def check_rebalancing_needed(current_weights, target_weights, threshold=0.05):
    """
    Check if portfolio needs rebalancing
    """
    current_weights = np.array(current_weights)
    target_weights = np.array(target_weights)
    
    drift = abs(current_weights - target_weights)
    needs_rebalancing = any(drift > threshold)
    max_drift = drift.max()
    
    rebalancing_suggestions = []
    if needs_rebalancing:
        for i, (curr, target, drift_val) in enumerate(zip(current_weights, target_weights, drift)):
            if drift_val > threshold:
                action = "BUY" if curr < target else "SELL"
                rebalancing_suggestions.append({
                    'stock_index': i,
                    'action': action,
                    'current_pct': curr * 100,
                    'target_pct': target * 100,
                    'adjustment_pct': abs(curr - target) * 100,
                    'adjustment_amount': abs(curr - target)
                })
    
    return {
        'needs_rebalancing': needs_rebalancing,
        'max_drift': max_drift,
        'suggestions': rebalancing_suggestions
    }

# ============================================
# WALK-FORWARD VALIDATION
# ============================================

def walk_forward_validation(data, model_func, train_window=252, test_window=21):
    """
    Perform walk-forward validation for backtesting
    """
    results = []
    dates = []
    
    n_samples = len(data)
    
    for i in range(0, n_samples - train_window - test_window, test_window):
        train_start = i
        train_end = i + train_window
        test_start = train_end
        test_end = min(test_start + test_window, n_samples)
        
        # Get train and test data
        train_data = data.iloc[train_start:train_end]
        test_data = data.iloc[test_start:test_end]
        
        if len(test_data) == 0:
            break
            
        # Train model (model_func should return predictions)
        try:
            predictions = model_func(train_data, test_data)
            
            # Calculate performance metrics
            if len(predictions) == len(test_data):
                accuracy = np.mean((predictions > 0) == (test_data.values > 0))
            else:
                accuracy = 0
            
            # Store results
            results.append({
                'window': len(results) + 1,
                'train_start': data.index[train_start],
                'train_end': data.index[train_end-1],
                'test_start': data.index[test_start],
                'test_end': data.index[test_end-1],
                'predictions': predictions,
                'actual': test_data.values,
                'accuracy': accuracy
            })
            dates.append(data.index[test_start])
            
        except Exception as e:
            print(f"Warning: Walk-forward iteration failed at {data.index[test_start]}: {e}")
            continue
    
    return results

# ============================================
# HELPER FUNCTIONS
# ============================================

def calculate_confidence_interval(data, confidence=0.95):
    """
    Calculate confidence interval for a dataset
    """
    if len(data) < 2:
        return np.mean(data) if len(data) > 0 else 0, 0
    
    n = len(data)
    mean = np.mean(data)
    sem = stats.sem(data)  # Standard error of the mean
    margin = sem * stats.t.ppf((1 + confidence) / 2., n-1)
    
    return mean, margin

def calculate_rolling_sharpe(returns, window=252):
    """
    Calculate rolling Sharpe ratio
    """
    rolling_sharpe = []
    for i in range(window, len(returns) + 1):
        window_returns = returns[i-window:i]
        if window_returns.std() > 0:
            sharpe = window_returns.mean() * 252 / (window_returns.std() * np.sqrt(252))
        else:
            sharpe = 0
        rolling_sharpe.append(sharpe)
    
    return np.array(rolling_sharpe)

def calculate_beta(stock_returns, market_returns):
    """
    Calculate beta (market sensitivity) of a stock
    """
    if len(stock_returns) != len(market_returns) or len(stock_returns) < 2:
        return 1
    
    covariance = np.cov(stock_returns, market_returns)[0, 1]
    variance = np.var(market_returns)
    
    if variance == 0:
        return 1
    
    return covariance / variance
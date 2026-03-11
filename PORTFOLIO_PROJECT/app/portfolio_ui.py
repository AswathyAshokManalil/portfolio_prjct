# portfolio_ui.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.portfolio_recommender import PortfolioRecommender

st.set_page_config(page_title="AI Portfolio Advisor", page_icon="📈", layout="wide")

st.markdown("""
<style>
.main-header { font-size: 3rem; color: #1E88E5; text-align: center; }
.metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white; padding: 1rem; border-radius: 10px; text-align: center; }
.stock-card { background-color: #f0f2f6; padding: 1rem; border-radius: 10px;
    border-left: 5px solid #1E88E5; margin: 0.5rem 0; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_recommender():
    return PortfolioRecommender(
        lstm_model_path='C:/DS_mainproject/portfolio_prjct/PORTFOLIO_PROJECT/models/lstm_model.keras',
        gru_model_path='C:/DS_mainproject/portfolio_prjct/PORTFOLIO_PROJECT/models/gru_model.keras',
        stock_data_path='C:/DS_mainproject/portfolio_prjct/PORTFOLIO_PROJECT/data/processed/final_dataset.csv',
        scalers_path='C:/DS_mainproject/portfolio_prjct/PORTFOLIO_PROJECT/data/processed/scalers.pkl',
        confidence_threshold=0.6
    )

st.markdown("<h1 class='main-header'>🤖 AI Portfolio Advisor</h1>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 📝 Your Investment Profile")
    amount = st.number_input("💰 Amount (₹)", 1000, 10000000, 100000, step=1000)
    duration = st.slider("⏱️ Years", 1, 30, 5)
    risk_type = st.select_slider("🎯 Risk", ['Low', 'Medium', 'High'], value='Medium').lower()
    
    with st.expander("⚙️ Advanced"):
        conf_thresh = st.slider("Confidence Threshold", 0.1, 0.8, 0.2, 0.05)
    
    generate_btn = st.button("🚀 Generate Portfolio", use_container_width=True)

if generate_btn:
    with st.spinner("🤖 Analyzing market data..."):
        recommender = load_recommender()
        recommendations = recommender.get_stock_recommendations(
            amount=amount, duration=duration, risk_type=risk_type,
            top_n=3, min_confidence=conf_thresh
        )
        
        # Top metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"<div class='metric-card'><h3>💰 Investment</h3><h2>₹{recommendations['predicted_returns']['total_investment']:,.2f}</h2></div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='metric-card'><h3>📈 Profit</h3><h2>₹{recommendations['predicted_returns']['total_profit']:,.2f}</h2></div>", unsafe_allow_html=True)
        with col3:
            st.markdown(f"<div class='metric-card'><h3>🎯 Future Value</h3><h2>₹{recommendations['predicted_returns']['total_future_value']:,.2f}</h2></div>", unsafe_allow_html=True)
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("ROI", f"{recommendations['predicted_returns']['roi_percentage']}%")
        with col2: st.metric("Annualized", f"{recommendations['predicted_returns']['annualized_return']}%")
        with col3: st.metric("Risk", risk_type.upper())
        
        st.markdown("---")
        st.markdown("## 🏆 Top 3 Recommended Stocks")
        
        # Stock cards
        cols = st.columns(3)
        for idx, stock in enumerate(recommendations['top_stocks']):
            with cols[idx]:
                st.markdown(f"""
                <div class='stock-card'>
                    <h3>{stock['symbol']}</h3>
                    <p>{stock['name']} | {stock['sector']}</p>
                    <p><b>Expected Return:</b> {stock['predicted_return'] * 100:.1f}%</p>
                    <p><b>Confidence:</b> {stock['confidence'] * 100:.1f}%</p>
                    
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            dist_df = pd.DataFrame(recommendations['amount_distribution'])
            fig = go.Figure(data=[go.Pie(labels=dist_df['symbol'], values=dist_df['amount'], hole=0.3)])
            fig.update_layout(title="Investment Allocation")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = go.Figure(data=[
                go.Bar(name='Investment', x=dist_df['symbol'], y=dist_df['amount'], marker_color='lightblue'),
                go.Bar(name='Expected Profit', x=dist_df['symbol'], 
                       y=[r['profit'] for r in recommendations['predicted_returns']['stock_wise_returns']], 
                       marker_color='lightgreen')
            ])
            fig.update_layout(title="Investment vs Profit", barmode='group')
            st.plotly_chart(fig, use_container_width=True)
        
        # Summary
        st.markdown("---")
        st.markdown("### 📝 Summary")
        st.info(recommendations['summary'])
        
        # Download
        csv = pd.DataFrame(recommendations['amount_distribution']).to_csv(index=False)
        st.download_button("📥 Download Portfolio", csv, f"portfolio.csv", "text/csv")

else:
    st.markdown("""
    <div style='text-align: center; padding: 3rem;'>
        <h2>👋 Welcome</h2>
        <p>Enter your details and click "Generate Portfolio" for AI-powered recommendations</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1: st.markdown("### 🧠 Ensemble AI\nLSTM + GRU for better predictions")
    with col2: st.markdown("### 🎯 Max Return\nSelects stocks with highest expected return")
    with col3: st.markdown("### 📊 Risk-Based\nPersonalized to your risk profile")
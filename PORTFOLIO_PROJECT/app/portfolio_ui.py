# portfolio_ui.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import sys
import os

# Add the parent directory to path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import from src
from src.portfolio_recommender import PortfolioRecommender

# Page configuration
st.set_page_config(
    page_title="AI Portfolio Advisor",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .card {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #f8f9fa;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .stock-recommendation {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize recommender (cache to avoid reloading)
@st.cache_resource
def load_recommender():
    return PortfolioRecommender(
        model_path='C:/DS_mainproject/portfolio_prjct/PORTFOLIO_PROJECT/models/best_model_gru.keras',
        stock_data_path='C:/DS_mainproject/portfolio_prjct/PORTFOLIO_PROJECT/data/processed/final_dataset.csv'
    )

# Title
st.markdown("<h1 class='main-header'>🤖 AI-Powered Portfolio Advisor</h1>", unsafe_allow_html=True)

# Sidebar for user input
with st.sidebar:
    st.markdown("<h2 class='sub-header'>📝 Your Investment Profile</h2>", unsafe_allow_html=True)
    
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        
        # Investment Amount
        amount = st.number_input(
            "💰 Investment Amount (₹)",
            min_value=1000,
            max_value=10000000,
            value=100000,
            step=1000,
            format="%d"
        )
        
        # Investment Duration
        duration = st.slider(
            "⏱️ Investment Duration (Years)",
            min_value=1,
            max_value=30,
            value=5,
            step=1
        )
        
        # Risk Profile
        risk_type = st.select_slider(
            "🎯 Risk Tolerance",
            options=['Low', 'Medium', 'High'],
            value='Medium'
        )
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Risk Profile Descriptions
        with st.expander("📊 Risk Profile Guide"):
            st.markdown("""
            **🟢 Low Risk**
            - Stable returns (8-10% annually)
            - Lower volatility
            - Blue-chip companies, bonds
            
            **🟡 Medium Risk**
            - Balanced returns (12-15% annually)
            - Moderate volatility
            - Mix of growth and value stocks
            
            **🔴 High Risk**
            - High returns (18-25% annually)
            - Higher volatility
            - Growth stocks, emerging markets
            """)
        
        # Generate Portfolio Button
        generate_btn = st.button("🚀 Generate Portfolio", use_container_width=True)

# Main content area
if generate_btn:
    with st.spinner("🤖 AI is analyzing market data and building your portfolio..."):
        # Load recommender
        recommender = load_recommender()
        
        # Get recommendations
        recommendations = recommender.get_stock_recommendations(amount, duration, risk_type)
        
        # Display results in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class='metric-card'>
                <h3>💰 Total Investment</h3>
                <h2>₹{:,.2f}</h2>
            </div>
            """.format(recommendations['predicted_returns']['total_investment']), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class='metric-card'>
                <h3>📈 Expected Profit</h3>
                <h2>₹{:,.2f}</h2>
            </div>
            """.format(recommendations['predicted_returns']['total_profit']), unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class='metric-card'>
                <h3>🎯 Future Value</h3>
                <h2>₹{:,.2f}</h2>
            </div>
            """.format(recommendations['predicted_returns']['total_future_value']), unsafe_allow_html=True)
        
        # Key Metrics Row
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("ROI", f"{recommendations['predicted_returns']['roi_percentage']}%")
        
        with col2:
            st.metric("Annualized Return", f"{recommendations['predicted_returns']['annualized_return']}%")
        
        with col3:
            st.metric("Risk Profile", risk_type.upper())
        
        st.markdown("---")
        
        # Top Stocks Recommendation
        st.markdown("<h2 class='sub-header'>🏆 Top 3 Recommended Stocks</h2>", unsafe_allow_html=True)
        
        cols = st.columns(3)
        for idx, stock in enumerate(recommendations['top_stocks']):
            with cols[idx]:
                st.markdown(f"""
                <div class='stock-recommendation'>
                    <h3>{stock['symbol']}</h3>
                    <p>{stock['name']}</p>
                    <p><b>Sector:</b> {stock['sector']}</p>
                    <p><b>Expected Return:</b> {stock['predicted_return']*100:.1f}%</p>
                    <p><b>Volatility:</b> {stock['volatility']*100:.1f}%</p>
                    <p><b>Confidence:</b> {stock['confidence_score']*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Portfolio Distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<h3>💰 Amount Distribution</h3>", unsafe_allow_html=True)
            
            # Create distribution dataframe
            dist_df = pd.DataFrame(recommendations['amount_distribution'])
            
            # Pie chart
            fig = go.Figure(data=[go.Pie(
                labels=dist_df['symbol'],
                values=dist_df['amount'],
                hole=.3,
                marker_colors=px.colors.qualitative.Set3
            )])
            fig.update_layout(title="Investment Allocation")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("<h3>📊 Expected Returns Breakdown</h3>", unsafe_allow_html=True)
            
            # Bar chart
            fig = go.Figure(data=[
                go.Bar(
                    name='Investment',
                    x=dist_df['symbol'],
                    y=dist_df['amount'],
                    marker_color='lightblue'
                ),
                go.Bar(
                    name='Expected Profit',
                    x=dist_df['symbol'],
                    y=[r['profit'] for r in recommendations['predicted_returns']['stock_wise_returns']],
                    marker_color='lightgreen'
                )
            ])
            fig.update_layout(
                title="Investment vs Expected Profit",
                barmode='group',
                yaxis_title="Amount (₹)"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed Distribution Table
        st.markdown("<h3>📋 Detailed Portfolio Breakdown</h3>", unsafe_allow_html=True)
        
        detailed_df = pd.DataFrame(recommendations['amount_distribution'])
        returns_df = pd.DataFrame(recommendations['predicted_returns']['stock_wise_returns'])
        
        detailed_df['predicted_return'] = detailed_df['predicted_return'].apply(lambda x: f"{x}%")
        detailed_df['amount'] = detailed_df['amount'].apply(lambda x: f"₹{x:,.2f}")
        detailed_df['percentage'] = detailed_df['percentage'].apply(lambda x: f"{x}%")
        
        st.dataframe(
            detailed_df[['symbol', 'name', 'amount', 'percentage', 'predicted_return']],
            use_container_width=True,
            hide_index=True
        )
        
        # Summary Text
        st.markdown("---")
        st.markdown("### 📝 Portfolio Summary")
        st.info(recommendations['summary'])
        
        # Download Button
        csv = detailed_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Portfolio Details",
            data=csv,
            file_name=f"portfolio_{risk_type}_{amount}.csv",
            mime="text/csv"
        )
        
else:
    # Welcome screen
    st.markdown("""
    <div style='text-align: center; padding: 3rem;'>
        <h2>👋 Welcome to AI Portfolio Advisor</h2>
        <p style='font-size: 1.2rem; color: #666;'>
            Enter your investment details in the sidebar and click "Generate Portfolio" 
            to get AI-powered stock recommendations tailored to your risk profile.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Features
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style='text-align: center; padding: 1rem;'>
            <h3>🧠 AI-Powered</h3>
            <p>Using deep learning (LSTM/GRU) to predict stock returns</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 1rem;'>
            <h3>🎯 Personalized</h3>
            <p>Tailored recommendations based on your risk profile</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='text-align: center; padding: 1rem;'>
            <h3>📊 Comprehensive</h3>
            <p>Detailed analysis with visualizations and projections</p>
        </div>
        """, unsafe_allow_html=True)  # ← Fixed this line!
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os

st.set_page_config(page_title="Quantitative Risk & Alpha Engine", layout="wide")
st.title("📈 Quantitative Portfolio Risk & Alpha Engine")
st.write("A production-grade interface tracking cross-asset correlation networks and directional predictive analytics.")

# Tracked portfolio assets
assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']

@st.cache
def load_portfolio_data():
    portfolio = {}
    for asset in assets:
        # LOOK HERE: We removed 'data/processed/' because files are in your main GitHub folder now!
        path = f'{asset}_market_data.csv'
        if os.path.exists(path):
            portfolio[asset] = pd.read_csv(path, parse_dates=['Date'], index_col='Date')
    return portfolio

portfolio_data = load_portfolio_data()

# Sidebar Navigation Panel
st.sidebar.header("Platform Navigation")
app_mode = st.sidebar.selectbox("Choose Engine Display", ["📊 Multi-Asset Risk Analytics (BI)", "🔮 Directional Movement Alpha (ML)"])

if app_mode == "📊 Multi-Asset Risk Analytics (BI)":
    st.subheader("Asset Performance & Correlation Framework")
    
    if not portfolio_data:
        st.error("No processed data files mapped. Make sure your CSV files are in the main repository folder!")
    else:
        close_prices = pd.DataFrame({ticker: df['Close'] for ticker, df in portfolio_data.items()})
        returns_df = close_prices.pct_change().dropna()
        
        col1, col2 = st.beta_columns(2)
        
        with col1:
            st.write("### Trailing Asset Volatility")
            annual_vol = returns_df.std() * np.sqrt(252) * 100
            for ticker, vol in annual_vol.items():
                st.metric(label=f"{ticker} Annualized Volatility", value=f"{vol:.1f}%")
                
        with col2:
            st.write("### Cross-Asset Correlation Map")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.set_theme(style="white")
            sns.heatmap(returns_df.corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1, center=0, linewidths=0.5, ax=ax)
            st.pyplot(fig)

elif app_mode == "🔮 Directional Movement Alpha (ML)":
    st.subheader("Predictive Market Signal System")
    st.write("Select an equity position to evaluate real-time feature structures and run machine learning classification models.")
    
    selected_asset = st.selectbox("Choose Target Equity Track", options=assets)
    
    if selected_asset in portfolio_data:
        asset_df = portfolio_data[selected_asset]
        latest_row = asset_df.iloc[-1]
        
        c1, c2, c3, c4 = st.beta_columns(4)
        c1.metric("Last Close Price", f"${latest_row['Close']:.2f}")
        c2.metric("10-Day Moving Average", f"${latest_row['SMA_10']:.2f}")
        c3.metric("50-Day Moving Average", f"${latest_row['SMA_50']:.2f}")
        c4.metric("20-Day Market Volatility", f"{latest_row['Volatility_20d']:.4f}")
        
        st.write("---")
        if st.button(f"Generate Next-Day Movement Vector for {selected_asset}"):
            # LOOK HERE: Checking the main folder for the model files
            model_path = f'{selected_asset}_model.pkl'
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
                
            features_input = [[latest_row['SMA_10'], latest_row['SMA_50'], latest_row['Volatility_20d'], latest_row['Volume']]]
            prediction = model.predict(features_input)
            probability = model.predict_proba(features_input)
            
            if prediction == 1:
                st.success(f"🚀 **Bullish Signal:** Model projects a positive closing trend tomorrow. Confidence level: **{(probability*100):.1f}%**")
            else:
                st.error(f"🐻 **Bearish Signal:** Model projects a downward closing trend tomorrow. Confidence level: **{(probability*100):.1f}%**")

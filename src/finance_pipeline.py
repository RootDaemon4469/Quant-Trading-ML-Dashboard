import os
import logging
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import pickle

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_indicators(ticker_symbol):
    logging.info(f"Fetching historical market records for {ticker_symbol}...")
    ticker = yf.Ticker(ticker_symbol)
    df = ticker.history(period="5y")
    if df.empty:
        raise ValueError(f"No market data returned for {ticker_symbol}")
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['Daily_Return'] = df['Close'].pct_change()
    df['Volatility_20d'] = df['Daily_Return'].rolling(window=20).std()
    df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    df = df.dropna()
    return df

def train_market_model(df, ticker_symbol):
    logging.info(f"Training predictive market model for {ticker_symbol}...")
    features = ['SMA_10', 'SMA_50', 'Volatility_20d', 'Volume']
    X = df[features]
    y = df['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)
    processed_dir = os.path.join('data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    df.to_csv(os.path.join(processed_dir, f'{ticker_symbol}_market_data.csv'))
    with open(os.path.join(processed_dir, f'{ticker_symbol}_model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    logging.info(f"Model saved. Test Accuracy: {model.score(X_test, y_test):.2f}")

if __name__ == '__main__':
    assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
    for asset in assets:
        try:
            data = generate_indicators(asset)
            train_market_model(data, asset)
        except Exception as e:
            logging.error(f"Failed to process asset {asset}: {str(e)}")

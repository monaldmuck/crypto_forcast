# data.py: Download and preprocess price data
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Download historical data using yfinance
def download_data(symbol='BTC-USD', period='3y', interval='1d'):
    df = yf.download(symbol, period=period, interval=interval)
    df = df[['Close']].dropna()  # Keep only 'Close' prices
    return df

# Create sequences of a fixed length for LSTM input
def create_sequences(data, seq_length=60):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:i+seq_length])
        ys.append(data[i+seq_length])
    return np.array(xs), np.array(ys)

# Preprocess the data: scale and split into training and test sets
def preprocess_data(df, seq_length=60):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    X, y = create_sequences(scaled, seq_length)
    split = int(0.8 * len(X))  # 80/20 train-test split
    return X[:split], X[split:], y[:split], y[split:], scaler
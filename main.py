# main.py: Multi-asset forecasting pipeline
from data import download_data, preprocess_data
from model import build_model
from train import train_model
from evaluate import evaluate
from plot import plot_predictions
import pandas as pd
import numpy as np
import os

def main():
    symbols = ['BTC-USD', 'ETH-USD', 'QQQ', 'VOX']
    symbol_categories = {
        'BTC-USD': 'Crypto',
        'ETH-USD': 'Crypto',
        'QQQ': 'ETF',
        'VOX': 'ETF'
    }

    all_metrics = {}

    os.makedirs('forecasts', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    for symbol in symbols:
        print(f"\nProcessing {symbol}...")

        # Step 1: Data
        df = download_data(symbol)
        X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

        # Step 2: Model
        model = build_model(seq_length=X_train.shape[1])
        train_model(model, X_train, y_train, epochs=10)

        # Step 3: Predictions
        predictions = model.predict(X_test)
        y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
        predictions_inv = scaler.inverse_transform(predictions)

        # Step 4: Save forecast CSV
        forecast_df = pd.DataFrame({
            'Date': df.index[-len(y_test):].strftime('%Y-%m-%d'),
            'True Price': y_test_inv.flatten(),
            'Predicted Price': predictions_inv.flatten()
        })
        forecast_df.to_csv(f'forecasts/{symbol}_forecast.csv', index=False)

        # Step 5: Evaluate & Save Metrics
        metrics = evaluate(y_test_inv, predictions_inv, return_dict=True)
        all_metrics[symbol] = metrics

        # Step 6: Save model
        model.save(f'models/{symbol}_lstm_model.h5')

        # Step 7: Plot
        plot_predictions(df.index[-len(y_test):], y_test_inv, predictions_inv, symbol, save=True)

    # Save metrics summary
    summary_df = pd.DataFrame.from_dict(all_metrics, orient='index')
    summary_df['Category'] = summary_df.index.map(symbol_categories)
    summary_df.to_csv('forecasts/summary_metrics.csv')

if __name__ == '__main__':
    main()


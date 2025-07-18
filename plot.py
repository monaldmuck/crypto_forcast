# plot.py: Visualize predictions
import matplotlib.pyplot as plt
import os

# Plot true vs predicted values
def plot_predictions(dates, y_true, y_pred, symbol='BTC-USD', save=False):
    plt.figure(figsize=(14, 6))
    plt.plot(dates, y_true, label='True Price')
    plt.plot(dates, y_pred, label='Predicted Price')
    plt.title(f'{symbol} Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    if save:
        os.makedirs('plots', exist_ok=True)
        plt.savefig(f'plots/{symbol}_forecast.png')
    plt.show()
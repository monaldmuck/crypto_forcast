
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate(y_true, y_pred, return_dict=False):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    mape = (abs((y_true - y_pred) / y_true)).mean()

    if return_dict:
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape
        }
    
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.4%}")


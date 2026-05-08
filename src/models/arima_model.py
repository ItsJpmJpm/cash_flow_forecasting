"""Cash Flow Forecasting - ARIMA Model"""

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')
import os

# Get project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
sys.path.insert(0, PROJECT_ROOT)

from src import config


class ARIMAModel:
    """ARIMA/SARIMA model for cash flow forecasting."""

    def __init__(self, p: int = 1, d: int = 1, q: int = 1,
                 seasonal_order: Tuple = (1, 1, 1, 12)):
        """Initialize ARIMA model."""
        self.p = p
        self.d = d
        self.q = q
        self.seasonal_order = seasonal_order
        self.model = None
        self.results = None
        self.aic = None
        self.bic = None

    def fit(self, train_data: pd.Series) -> 'ARIMAModel':
        """Fit ARIMA model to training data."""
        print(f"Fitting SARIMA({self.p},{self.d},{self.q})"
              f"({self.seasonal_order[0]},{self.seasonal_order[1]},{self.seasonal_order[2]})"
              f" s={self.seasonal_order[3]}")

        self.model = SARIMAX(
            train_data,
            order=(self.p, self.d, self.q),
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        self.results = self.model.fit(disp=False)
        self.aic = self.results.aic
        self.bic = self.results.bic

        print(f"  AIC: {self.aic:.2f}")
        print(f"  BIC: {self.bic:.2f}")

        return self

    def forecast(self, n_periods: int = 6,
                 return_confidence: bool = True) -> Dict:
        """Generate forecast."""
        if self.results is None:
            raise ValueError("Model not fitted. Call fit() first.")

        forecast = self.results.get_forecast(steps=n_periods)
        mean_forecast = forecast.predicted_mean

        if return_confidence:
            conf_int = forecast.conf_int()
            return {
                'forecast': mean_forecast.values,
                'lower_ci': conf_int.iloc[:, 0].values,
                'upper_ci': conf_int.iloc[:, 1].values,
            }
        else:
            return {'forecast': mean_forecast.values}

    def get_residuals(self) -> pd.Series:
        """Get model residuals."""
        if self.results is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.results.resid

    def summary(self) -> str:
        """Get model summary as string."""
        if self.results is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return str(self.results.summary())


def check_stationarity(series: pd.Series) -> Dict:
    """Check if a time series is stationary using ADF test."""
    result = adfuller(series.dropna())

    return {
        'adf_statistic': result[0],
        'p_value': result[1],
        'is_stationary': result[1] < 0.05,
        'critical_values': result[4]
    }


def train_arima(df: pd.DataFrame, use_best_params: bool = False) -> Dict:
    """Train ARIMA model on cash flow data."""
    # Prepare training data
    train_size = config.TRAIN_SIZE
    train_data = df.set_index('fecha')['cash_flow_real'][:train_size]
    test_data = df.set_index('fecha')['cash_flow_real'][train_size:]

    # Check stationarity
    print("\n" + "=" * 60)
    print("ARIMA MODEL TRAINING")
    print("=" * 60)
    print("\nChecking stationarity...")
    stationarity = check_stationarity(train_data)
    print(f"  ADF Statistic: {stationarity['adf_statistic']:.4f}")
    print(f"  P-value: {stationarity['p_value']:.4f}")
    print(f"  Stationary: {stationarity['is_stationary']}")

    # Train model
    print("\nUsing parameters from config...")
    arima_config = config.MODEL_CONFIG['arima']
    model = ARIMAModel(
        p=arima_config['p'],
        d=arima_config['d'],
        q=arima_config['q'],
        seasonal_order=arima_config['seasonal_order']
    )
    model.fit(train_data)

    # Generate forecast
    n_test = len(test_data)
    forecast_results = model.forecast(n_periods=n_test)

    # Print predictions
    print("\n" + "=" * 60)
    print("PREDICTIONS VS ACTUAL")
    print("=" * 60)
    print(f"{'Date':<12} {'Actual (€)':>15} {'Forecast (€)':>15} {'Lower CI':>12} {'Upper CI':>12}")
    print("-" * 70)

    for i, (date, actual) in enumerate(test_data.items()):
        forecast = forecast_results['forecast'][i]
        lower = forecast_results.get('lower_ci', [0]*n_test)[i]
        upper = forecast_results.get('upper_ci', [0]*n_test)[i]
        print(f"{str(date)[:10]:<12} {actual:>15,.2f} {forecast:>15,.2f} {lower:>12,.2f} {upper:>12,.2f}")

    # Calculate metrics
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mae = mean_absolute_error(test_data.values, forecast_results['forecast'])
    rmse = np.sqrt(mean_squared_error(test_data.values, forecast_results['forecast']))
    r2 = r2_score(test_data.values, forecast_results['forecast'])

    # MAPE
    non_zero = test_data.values != 0
    mape = np.mean(np.abs((test_data.values[non_zero] - forecast_results['forecast'][non_zero]) / test_data.values[non_zero])) * 100

    print("-" * 70)
    print(f"{'MAE:':<15} {mae:>15,.2f}")
    print(f"{'RMSE:':<15} {rmse:>15,.2f}")
    print(f"{'MAPE:':<15} {mape:>15.2f}%")
    print(f"{'R²:':<15} {r2:>15.4f}")

    return {
        'model': model,
        'train_data': train_data,
        'test_data': test_data,
        'forecast': forecast_results['forecast'],
        'lower_ci': forecast_results.get('lower_ci'),
        'upper_ci': forecast_results.get('upper_ci'),
        'residuals': model.get_residuals(),
        'metrics': {'mae': mae, 'rmse': rmse, 'mape': mape, 'r2': r2}
    }


if __name__ == "__main__":
    from src.data_generator import generate_cash_flow_data

    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    df = generate_cash_flow_data()
    print(f"Loaded {len(df)} months of data")

    print("\nTraining ARIMA model...")
    results = train_arima(df, use_best_params=False)

    # Save results
    import pickle
    os.makedirs("models/saved", exist_ok=True)
    with open("models/saved/arima_results.pkl", 'wb') as f:
        pickle.dump(results, f)
    print("\nResults saved to models/saved/arima_results.pkl")
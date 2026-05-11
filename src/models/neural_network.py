"""Cash Flow Forecasting - Neural Network Model (MLP)"""

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')
import os
import pickle

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
sys.path.insert(0, PROJECT_ROOT)

from src import config


class CashFlowMLP:
    """Multi-Layer Perceptron for Cash Flow Forecasting."""

    def __init__(self,
                 hidden_layers: tuple = (64, 32),
                 activation: str = 'relu',
                 alpha: float = 0.001,
                 learning_rate: float = 0.001,
                 max_iter: int = 500,
                 early_stopping: bool = True,
                 validation_fraction: float = 0.2,
                 n_iter_no_change: int = 50,
                 random_state: int = 42):
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.random_state = random_state

        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.feature_names = None
        self._X_test = None
        self._y_test = None

    def fit(self, df: pd.DataFrame) -> 'CashFlowMLP':
        """Train the MLP model."""
        # Create features inline - use only pre-computed lag features
        # These are available in the data from data_generator
        df_feat = df.copy()
        
        # Temporal features - these capture seasonality well
        df_feat['mes_sin'] = np.sin(2 * np.pi * df_feat['mes'] / 12)
        df_feat['mes_cos'] = np.cos(2 * np.pi * df_feat['mes'] / 12)
        df_feat['trimestre_sin'] = np.sin(2 * np.pi * df_feat['trimestre'] / 4)
        df_feat['trimestre_cos'] = np.cos(2 * np.pi * df_feat['trimestre'] / 4)
        
        # Trend (normalized month number)
        df_feat['tendencia'] = (df_feat['año'] - 2021) * 12 + df_feat['mes']
        min_t, max_t = df_feat['tendencia'].min(), df_feat['tendencia'].max()
        df_feat['tendencia_normalized'] = (df_feat['tendencia'] - min_t) / (max_t - min_t) if max_t > min_t else 0
        
        # Ratio features from the data
        df_feat['ratio_cobros_pagos'] = df_feat['clientes_activos'] / (df_feat['proyectos_pendientes'] + 1)
        
        # One-hot encode mes for better seasonal pattern capture
        for m in range(1, 13):
            df_feat[f'mes_{m}'] = (df_feat['mes'] == m).astype(int)
        
        # Feature columns - use original features only, not the pre-computed lags
        exclude = ['fecha', 'cash_flow_real', 'año', 'mes', 'trimestre', 'is_low_demand',
                   'ingresos_totales', 'egresos_totales', 'num_transacciones',
                   'cash_flow_lag1', 'cash_flow_lag2', 'media_movil_3', 'variacion_mes_anterior']
        self.feature_names = [col for col in df_feat.columns if col not in exclude]

        X = df_feat[self.feature_names].values
        y = df_feat['cash_flow_real'].values

        # Split: 24 train, 12 test
        X_train, X_test = X[:24], X[24:36]
        y_train, y_test = y[:24], y[24:36]

        print(f"\n  Features ({len(self.feature_names)}): {self.feature_names[:10]}...")
        print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"  y_train range: {y_train.min():.2f} - {y_train.max():.2f}")
        print(f"  y_test range: {y_test.min():.2f} - {y_test.max():.2f}")

        # Scale features
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        X_test_scaled = self.scaler_X.transform(X_test)

        # Scale target (fit on training data)
        y_train_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

        print("\n  Building and training MLP model...")
        print("-" * 50)

        # Use a larger network with more capacity
        self.model = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),  # Larger architecture
            activation='relu',
            solver='adam',
            alpha=0.0001,  # Less regularization
            learning_rate_init=self.learning_rate,
            max_iter=self.max_iter,
            early_stopping=self.early_stopping,
            validation_fraction=self.validation_fraction,
            n_iter_no_change=self.n_iter_no_change,
            random_state=self.random_state,
            verbose=True
        )

        self.model.fit(X_train_scaled, y_train_scaled)

        print("-" * 50)
        print(f"  Training completed in {self.model.n_iter_} iterations")
        if hasattr(self.model, 'loss_curve_'):
            print(f"  Final loss: {self.model.loss_curve_[-1]:.6f}")

        self._X_test = X_test_scaled
        self._y_test = y_test

        return self

    def predict(self) -> np.ndarray:
        """Generate predictions."""
        if self.model is None:
            raise ValueError("Model not trained.")

        y_pred_scaled = self.model.predict(self._X_test)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

        return y_pred

    def get_training_loss_curve(self) -> list:
        if hasattr(self.model, 'loss_curve_'):
            return self.model.loss_curve_
        return []

    def save(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler_X': self.scaler_X,
                'scaler_y': self.scaler_y,
                'feature_names': self.feature_names,
                'config': {
                    'hidden_layers': self.hidden_layers,
                    'activation': self.activation,
                    'alpha': self.alpha
                }
            }, f)
        print(f"  Model saved to: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'CashFlowMLP':
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        instance = cls(
            hidden_layers=data['config']['hidden_layers'],
            activation=data['config']['activation'],
            alpha=data['config']['alpha']
        )
        instance.model = data['model']
        instance.scaler_X = data['scaler_X']
        instance.scaler_y = data['scaler_y']
        instance.feature_names = data['feature_names']

        return instance


class CashFlowGBM:
    """Gradient Boosting for Cash Flow Forecasting - handles small datasets better."""

    def __init__(self, n_estimators: int = 200, max_depth: int = 3, 
                 learning_rate: float = 0.05, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state

        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.feature_names = None
        self._X_test = None
        self._y_test = None

    def fit(self, df: pd.DataFrame) -> 'CashFlowGBM':
        """Train the GBM model."""
        df_feat = df.copy()
        
        # Temporal features
        df_feat['mes_sin'] = np.sin(2 * np.pi * df_feat['mes'] / 12)
        df_feat['mes_cos'] = np.cos(2 * np.pi * df_feat['mes'] / 12)
        df_feat['trimestre_sin'] = np.sin(2 * np.pi * df_feat['trimestre'] / 4)
        df_feat['trimestre_cos'] = np.cos(2 * np.pi * df_feat['trimestre'] / 4)
        
        # Trend
        df_feat['tendencia'] = (df_feat['año'] - 2021) * 12 + df_feat['mes']
        min_t, max_t = df_feat['tendencia'].min(), df_feat['tendencia'].max()
        df_feat['tendencia_normalized'] = (df_feat['tendencia'] - min_t) / (max_t - min_t) if max_t > min_t else 0
        
        # Ratio
        df_feat['ratio_cobros_pagos'] = df_feat['clientes_activos'] / (df_feat['proyectos_pendientes'] + 1)
        
        # One-hot encode mes
        for m in range(1, 13):
            df_feat[f'mes_{m}'] = (df_feat['mes'] == m).astype(int)
        
        # Feature columns
        exclude = ['fecha', 'cash_flow_real', 'año', 'mes', 'trimestre', 'is_low_demand',
                   'ingresos_totales', 'egresos_totales', 'num_transacciones',
                   'cash_flow_lag1', 'cash_flow_lag2', 'media_movil_3', 'variacion_mes_anterior']
        self.feature_names = [col for col in df_feat.columns if col not in exclude]

        X = df_feat[self.feature_names].values
        y = df_feat['cash_flow_real'].values

        # Split: 24 train, 12 test
        X_train, X_test = X[:24], X[24:36]
        y_train, y_test = y[:24], y[24:36]

        print(f"\n  Features ({len(self.feature_names)}): {self.feature_names[:10]}...")
        print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")

        # Scale features
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        X_test_scaled = self.scaler_X.transform(X_test)

        # Scale target
        y_train_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

        print("\n  Building and training GBM model...")
        print("-" * 50)

        self.model = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            subsample=0.8,
            min_samples_split=3,
            min_samples_leaf=2
        )

        self.model.fit(X_train_scaled, y_train_scaled)

        print("-" * 50)
        print(f"  Training completed with {self.model.n_estimators_} trees")
        print(f"  Feature importances: {dict(zip(self.feature_names[:5], self.model.feature_importances_[:5]))}")

        self._X_test = X_test_scaled
        self._y_test = y_test

        return self

    def predict(self) -> np.ndarray:
        """Generate predictions."""
        if self.model is None:
            raise ValueError("Model not trained.")

        y_pred_scaled = self.model.predict(self._X_test)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

        return y_pred

    def save(self, filepath: str):
        """Save the GBM model to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler_X': self.scaler_X,
                'scaler_y': self.scaler_y,
                'feature_names': self.feature_names,
                'config': {
                    'n_estimators': self.n_estimators,
                    'max_depth': self.max_depth,
                    'learning_rate': self.learning_rate
                }
            }, f)
        print(f"  Model saved to: {filepath}")


def train_gbm(df: pd.DataFrame, save_path: str = None) -> dict:
    """Train GBM model on cash flow data."""
    print("\n" + "=" * 60)
    print("GRADIENT BOOSTING TRAINING")
    print("=" * 60)

    model = CashFlowGBM(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        random_state=config.RANDOM_SEED
    )

    model.fit(df)

    y_pred = model.predict()
    y_test_orig = model._y_test

    # Calculate metrics
    mae = mean_absolute_error(y_test_orig, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred))
    r2 = r2_score(y_test_orig, y_pred)

    non_zero = y_test_orig != 0
    mape = np.mean(np.abs((y_test_orig[non_zero] - y_pred[non_zero]) / y_test_orig[non_zero])) * 100 if non_zero.sum() > 0 else np.nan

    print("\n" + "=" * 60)
    print("PREDICTIONS VS ACTUAL")
    print("=" * 60)
    print(f"{'Date':<12} {'Actual (€)':>15} {'Predicted (€)':>15} {'Error (€)':>15}")
    print("-" * 60)

    test_dates = df['fecha'].iloc[24:36].values
    for i, (actual, pred) in enumerate(zip(y_test_orig, y_pred)):
        error = pred - actual
        date_str = str(test_dates[i])[:10]
        print(f"{date_str:<12} {actual:>15,.2f} {pred:>15,.2f} {error:>15,.2f}")

    print("-" * 60)
    print(f"{'MAE:':<15} {mae:>15,.2f}")
    print(f"{'RMSE:':<15} {rmse:>15,.2f}")
    print(f"{'MAPE:':<15} {mape:>15.2f}%")
    print(f"{'R²:':<15} {r2:>15.4f}")

    results = {
        'model': model,
        'predictions': y_pred,
        'actual': y_test_orig,
        'feature_names': model.feature_names,
        'metrics': {'mae': mae, 'rmse': rmse, 'mape': mape, 'r2': r2}
    }

    if save_path:
        model.save(save_path)

    return results


class CashFlowRF:
    """Random Forest for Cash Flow Forecasting - robust for small datasets."""

    def __init__(self, n_estimators: int = 100, max_depth: int = 5, 
                 min_samples_split: int = 2, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state

        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.feature_names = None
        self._X_test = None
        self._y_test = None

    def fit(self, df: pd.DataFrame) -> 'CashFlowRF':
        """Train the RF model."""
        df_feat = df.copy()
        
        # Temporal features
        df_feat['mes_sin'] = np.sin(2 * np.pi * df_feat['mes'] / 12)
        df_feat['mes_cos'] = np.cos(2 * np.pi * df_feat['mes'] / 12)
        df_feat['trimestre_sin'] = np.sin(2 * np.pi * df_feat['trimestre'] / 4)
        df_feat['trimestre_cos'] = np.cos(2 * np.pi * df_feat['trimestre'] / 4)
        
        # Trend
        df_feat['tendencia'] = (df_feat['año'] - 2021) * 12 + df_feat['mes']
        min_t, max_t = df_feat['tendencia'].min(), df_feat['tendencia'].max()
        df_feat['tendencia_normalized'] = (df_feat['tendencia'] - min_t) / (max_t - min_t) if max_t > min_t else 0
        
        # Ratio
        df_feat['ratio_cobros_pagos'] = df_feat['clientes_activos'] / (df_feat['proyectos_pendientes'] + 1)
        
        # One-hot encode mes
        for m in range(1, 13):
            df_feat[f'mes_{m}'] = (df_feat['mes'] == m).astype(int)
        
        # Feature columns
        exclude = ['fecha', 'cash_flow_real', 'año', 'mes', 'trimestre', 'is_low_demand',
                   'ingresos_totales', 'egresos_totales', 'num_transacciones',
                   'cash_flow_lag1', 'cash_flow_lag2', 'media_movil_3', 'variacion_mes_anterior']
        self.feature_names = [col for col in df_feat.columns if col not in exclude]

        X = df_feat[self.feature_names].values
        y = df_feat['cash_flow_real'].values

        # Split: 24 train, 12 test
        X_train, X_test = X[:24], X[24:36]
        y_train, y_test = y[:24], y[24:36]

        print(f"\n  Features ({len(self.feature_names)}): {self.feature_names[:10]}...")
        print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")

        # Scale features
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        X_test_scaled = self.scaler_X.transform(X_test)

        # Scale target
        y_train_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

        print("\n  Building and training RF model...")
        print("-" * 50)

        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            random_state=self.random_state,
            bootstrap=True,
            max_features='sqrt'
        )

        self.model.fit(X_train_scaled, y_train_scaled)

        print("-" * 50)
        print(f"  Training completed with {self.model.n_estimators} trees")
        print(f"  Feature importances: {dict(zip(self.feature_names[:5], self.model.feature_importances_[:5]))}")

        self._X_test = X_test_scaled
        self._y_test = y_test

        return self

    def predict(self) -> np.ndarray:
        """Generate predictions."""
        if self.model is None:
            raise ValueError("Model not trained.")

        y_pred_scaled = self.model.predict(self._X_test)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

        return y_pred

    def save(self, filepath: str):
        """Save the RF model to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler_X': self.scaler_X,
                'scaler_y': self.scaler_y,
                'feature_names': self.feature_names,
                'config': {
                    'n_estimators': self.n_estimators,
                    'max_depth': self.max_depth,
                    'min_samples_split': self.min_samples_split
                }
            }, f)
        print(f"  Model saved to: {filepath}")


def train_rf(df: pd.DataFrame, save_path: str = None) -> dict:
    """Train RF model on cash flow data."""
    print("\n" + "=" * 60)
    print("RANDOM FOREST TRAINING")
    print("=" * 60)

    model = CashFlowRF(
        n_estimators=100,
        max_depth=5,
        min_samples_split=2,
        random_state=config.RANDOM_SEED
    )

    model.fit(df)

    y_pred = model.predict()
    y_test_orig = model._y_test

    # Calculate metrics
    mae = mean_absolute_error(y_test_orig, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred))
    r2 = r2_score(y_test_orig, y_pred)

    non_zero = y_test_orig != 0
    mape = np.mean(np.abs((y_test_orig[non_zero] - y_pred[non_zero]) / y_test_orig[non_zero])) * 100 if non_zero.sum() > 0 else np.nan

    print("\n" + "=" * 60)
    print("PREDICTIONS VS ACTUAL")
    print("=" * 60)
    print(f"{'Date':<12} {'Actual (€)':>15} {'Predicted (€)':>15} {'Error (€)':>15}")
    print("-" * 60)

    test_dates = df['fecha'].iloc[24:36].values
    for i, (actual, pred) in enumerate(zip(y_test_orig, y_pred)):
        error = pred - actual
        date_str = str(test_dates[i])[:10]
        print(f"{date_str:<12} {actual:>15,.2f} {pred:>15,.2f} {error:>15,.2f}")

    print("-" * 60)
    print(f"{'MAE:':<15} {mae:>15,.2f}")
    print(f"{'RMSE:':<15} {rmse:>15,.2f}")
    print(f"{'MAPE:':<15} {mape:>15.2f}%")
    print(f"{'R²:':<15} {r2:>15.4f}")

    results = {
        'model': model,
        'predictions': y_pred,
        'actual': y_test_orig,
        'feature_names': model.feature_names,
        'metrics': {'mae': mae, 'rmse': rmse, 'mape': mape, 'r2': r2}
    }

    if save_path:
        model.save(save_path)

    return results


def train_mlp(df: pd.DataFrame, save_path: str = None) -> dict:
    print("=" * 60)
    print("NEURAL NETWORK (MLP) TRAINING")
    print("=" * 60)

    nn_config = config.MODEL_CONFIG['neural_network']

    model = CashFlowMLP(
        hidden_layers=tuple(nn_config['hidden_layers']),
        alpha=0.001,
        learning_rate=nn_config['learning_rate'],
        max_iter=nn_config['epochs'],
        n_iter_no_change=nn_config['patience'],
        validation_fraction=nn_config['validation_split'],
        random_state=config.RANDOM_SEED
    )

    model.fit(df)

    y_pred = model.predict()
    y_test_orig = model._y_test

    # Calculate metrics
    mae = mean_absolute_error(y_test_orig, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred))
    r2 = r2_score(y_test_orig, y_pred)

    non_zero = y_test_orig != 0
    mape = np.mean(np.abs((y_test_orig[non_zero] - y_pred[non_zero]) / y_test_orig[non_zero])) * 100 if non_zero.sum() > 0 else np.nan

    print("\n" + "=" * 60)
    print("PREDICTIONS VS ACTUAL")
    print("=" * 60)
    print(f"{'Date':<12} {'Actual (€)':>15} {'Predicted (€)':>15} {'Error (€)':>15}")
    print("-" * 60)

    test_dates = df['fecha'].iloc[24:36].values
    for i, (actual, pred) in enumerate(zip(y_test_orig, y_pred)):
        error = pred - actual
        date_str = str(test_dates[i])[:10]
        print(f"{date_str:<12} {actual:>15,.2f} {pred:>15,.2f} {error:>15,.2f}")

    print("-" * 60)
    print(f"{'MAE:':<15} {mae:>15,.2f}")
    print(f"{'RMSE:':<15} {rmse:>15,.2f}")
    print(f"{'MAPE:':<15} {mape:>15.2f}%")
    print(f"{'R²:':<15} {r2:>15.4f}")

    results = {
        'model': model,
        'predictions': y_pred,
        'actual': y_test_orig,
        'loss_curve': model.get_training_loss_curve(),
        'feature_names': model.feature_names,
        'metrics': {'mae': mae, 'rmse': rmse, 'mape': mape, 'r2': r2}
    }

    if save_path:
        model.save(save_path)

    return results


if __name__ == "__main__":
    from src.data_generator import generate_cash_flow_data

    print("Loading data...")
    df = generate_cash_flow_data()
    print(f"Loaded {len(df)} months of data")

    print("\nTraining MLP...")
    results = train_mlp(df, save_path="models/saved/mlp_model.pkl")

    with open("models/saved/nn_results.pkl", 'wb') as f:
        pickle.dump({
            'predictions': results['predictions'],
            'actual': results['actual'],
            'metrics': results['metrics']
        }, f)
    print("\nResults saved to models/saved/nn_results.pkl")
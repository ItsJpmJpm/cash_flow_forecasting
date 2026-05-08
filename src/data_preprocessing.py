"""Cash Flow Forecasting - Data Preprocessing"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class CashFlowPreprocessor:
    """Preprocessing pipeline for cash flow forecasting data."""

    def __init__(self):
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.feature_names = None

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features from raw data."""
        df = df.copy()

        # Temporal features (cyclical encoding)
        df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
        df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)
        df['trimestre_sin'] = np.sin(2 * np.pi * df['trimestre'] / 4)
        df['trimestre_cos'] = np.cos(2 * np.pi * df['trimestre'] / 4)

        # Trend (normalized month number)
        df['tendencia'] = (df['año'] - 2021) * 12 + df['mes']
        min_t = df['tendencia'].min()
        max_t = df['tendencia'].max()
        df['tendencia_normalized'] = (df['tendencia'] - min_t) / (max_t - min_t) if max_t > min_t else 0

        # Ratio features
        df['ratio_ingresos_egresos'] = df['ingresos_totales'] / (df['egresos_totales'] + 1)

        # Nomina day
        df['dia_nomina'] = 1

        # One-hot encode mes
        for m in range(1, 13):
            df[f'mes_{m}'] = (df['mes'] == m).astype(int)

        return df

    def get_feature_columns(self, df: pd.DataFrame) -> list:
        """Get list of feature columns for model input."""
        exclude = ['fecha', 'cash_flow_real', 'año', 'mes', 'trimestre',
                   'is_low_demand', 'ingresos_totales', 'egresos_totales']
        features = [col for col in df.columns if col not in exclude]
        return features

    def fit_transform(self, df: pd.DataFrame) -> tuple:
        """
        Fit preprocessors and transform data.
        Split: Train (2021-2022, 24 months), Test (2023, 12 months)
        Returns: X_train, X_test, y_train_scaled, y_test, y_train_orig, y_test_orig
        """
        # Create features
        df_features = self.create_features(df)

        # Get feature columns
        self.feature_names = self.get_feature_columns(df_features)

        # Prepare X and y
        X = df_features[self.feature_names].values
        y = df_features['cash_flow_real'].values

        # Split: 24 months for training, 12 for test
        train_size = 24
        test_size = 12

        X_train = X[:train_size]
        X_test = X[train_size:train_size + test_size]
        y_train = y[:train_size]
        y_test = y[train_size:train_size + test_size]

        print(f"  Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        print(f"  Test: {X_test.shape[0]} samples")
        print(f"  y_train range: {y_train.min():.2f} - {y_train.max():.2f}")
        print(f"  y_test range: {y_test.min():.2f} - {y_test.max():.2f}")

        # Scale features
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        X_test_scaled = self.scaler_X.transform(X_test)

        # Scale target (fit on training data only)
        y_train_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

        # Return unscaled y_train and y_test for later use
        return X_train_scaled, X_test_scaled, y_train_scaled, y_test, y_train, y_test


if __name__ == "__main__":
    import sys
    import os
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, PROJECT_ROOT)
    
    from src.data_generator import generate_cash_flow_data

    print("Testing preprocessor...")
    df = generate_cash_flow_data()

    preprocessor = CashFlowPreprocessor()
    X_train, X_test, y_train_scaled, y_test, y_train_orig, y_test_orig = preprocessor.fit_transform(df)

    print(f"\nX_train: {X_train.shape}, y_train_scaled: {y_train_scaled.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
    print(f"y_test_orig: {y_test_orig.shape}")
    print(f"Feature names: {len(preprocessor.feature_names)}")
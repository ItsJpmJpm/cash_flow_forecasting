"""
Cash Flow Forecasting - Configuration
TechSolutions Toledo S.L.
"""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models" / "saved"
DOCS_DIR = PROJECT_ROOT / "docs"

# Company configuration
COMPANY_NAME = "TechSolutions Toledo S.L."
COMPANY_LOCATION = "Toledo, España"
COMPANY_SECTOR = "Servicios IT / Consultoría tecnológica"
COMPANY_EMPLOYEES = 50

# Date range
START_DATE = "2021-01-01"
END_DATE = "2023-12-31"
FORECAST_HORIZON = 6  # months

# Financial ranges (in €)
INGRESOS_MIN = 180000
INGRESOS_MAX = 450000
EGRESOS_MIN = 150000
EGRESOS_MAX = 400000
CASH_FLOW_MIN = -50000
CASH_FLOW_MAX = 150000

# Model hyperparameters
MODEL_CONFIG = {
    "arima": {
        "p": 1,  # AR order
        "d": 1,  # Differencing order
        "q": 1,  # MA order
        "seasonal_order": (1, 1, 1, 12),  # Monthly seasonality
    },
    "neural_network": {
        "input_dim": 12,
        "hidden_layers": [64, 32],
        "dropout": 0.3,
        "learning_rate": 0.001,
        "epochs": 500,
        "batch_size": 32,
        "patience": 50,
        "validation_split": 0.2,
    }
}

# Telegram bot configuration
TELEGRAM_CONFIG = {
    "alert_threshold": 50000,  # € - Cash flow below this triggers alert
    "low_demand_threshold": 0.7,  # Ratio relative to monthly average
    "check_interval_hours": 24,  # How often to check for alerts
}

# Train/test split
TRAIN_SIZE = 24  # 2021-2022 for training
VAL_SIZE = 6     # 01/2023 - 06/2023 for validation
TEST_SIZE = 6    # 07/2023 - 12/2023 for testing

# Random seed for reproducibility
RANDOM_SEED = 42

# Seasonal factors (multipliers for each month)
SEASONAL_FACTORS = {
    1: 0.65,   # Enero - muy bajo, post-navidad
    2: 0.75,   # Febrero - recuperación lenta
    3: 0.85,   # Marzo - mejora
    4: 0.90,   # Abril - Semana Santa impacta
    5: 1.00,   # Mayo - normal
    6: 1.05,   # Junio - fin fiscal fuerte
    7: 0.70,   # Julio - vacaciones cliente
    8: 0.60,   # Agosto - mínimo anual
    9: 0.85,   # Septiembre - vuelta vacaciones
    10: 1.00,  # Octubre - inicio Q4
    11: 1.10,  # Noviembre - pre-navidad activo
    12: 0.80,  # Diciembre - navidades
}

# External shock factors (COVID, crisis, etc.)
SHOCK_FACTORS = {
    "2021-01": 0.75,  # COVID aún afectando
    "2021-02": 0.80,
    "2021-03": 0.85,
    "2022-10": 0.92,  # Crisis energética empieza
    "2022-11": 0.90,
    "2022-12": 0.88,
}

# Evaluation thresholds
EVALUATION_THRESHOLDS = {
    "mape_max": 10,  # percentage
    "mae_max": 15000,  # euros
    "r2_min": 0.70,
}
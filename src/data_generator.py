"""Cash Flow Forecasting - Data Generator for TechSolutions Toledo S.L."""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
sys.path.append(str(__file__).replace("data_generator.py", ""))

import config


def set_random_seed(seed=config.RANDOM_SEED):
    """Set random seed for reproducibility."""
    np.random.seed(seed)


def get_seasonal_factor(month: int, year: int) -> float:
    """Get seasonal factor for a given month."""
    base_factor = config.SEASONAL_FACTORS.get(month, 1.0)

    # Check for external shocks
    date_key = f"{year}-{month:02d}"
    shock = config.SHOCK_FACTORS.get(date_key, 1.0)

    return base_factor * shock


def generate_cash_flow_data(
    start_date: str = config.START_DATE,
    end_date: str = config.END_DATE,
    num_transactions: int = 500
) -> pd.DataFrame:
    """
    Generate synthetic cash flow data for TechSolutions Toledo S.L.

    Args:
        start_date: Start date of the data
        end_date: End date of the data
        num_transactions: Approximate number of transactions per month

    Returns:
        DataFrame with monthly aggregated cash flow data
    """
    set_random_seed()

    # Generate monthly dates
    date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
    data = []

    for date in date_range:
        month = date.month
        year = date.year

        # Base values (mean of ranges)
        base_ingresos = (config.INGRESOS_MIN + config.INGRESOS_MAX) / 2
        base_egresos = (config.EGRESOS_MIN + config.EGRESOS_MAX) / 2

        # Apply seasonal factor
        seasonal = get_seasonal_factor(month, year)

        # Add yearly trend (slight growth)
        years_since_start = year - 2021
        trend_factor = 1 + (years_since_start * 0.03)  # 3% yearly growth

        # Calculate monthly values with noise
        noise_scale = 0.15  # 15% noise
        ingresos = base_ingresos * seasonal * trend_factor * np.random.normal(1, noise_scale)
        egresos = base_egresos * seasonal * trend_factor * np.random.normal(1, noise_scale * 0.8)

        # Ensure positive values
        ingresos = max(ingresos, config.INGRESOS_MIN * 0.5)
        egresos = max(egresos, config.EGRESOS_MIN * 0.5)

        # Cash flow
        cash_flow = ingresos - egresos

        # Additional features
        clientes_activos = int(np.random.normal(60, 10))
        clientes_activos = max(40, min(80, clientes_activos))

        proyectos_pendientes = int(np.random.normal(20, 5))
        proyectos_pendientes = max(10, min(30, proyectos_pendientes))

        dias_cartera = np.random.normal(45, 8)
        dias_cartera = max(30, min(65, dias_cartera))

        # Ratio cobros/pagos
        ratio_cobros = np.random.normal(1.0, 0.2)
        ratio_cobros = max(0.5, min(2.0, ratio_cobros))

        # Determine if it's a low demand month
        is_low_demand = 1 if (seasonal < 0.75 or cash_flow < config.TELEGRAM_CONFIG["alert_threshold"]) else 0

        data.append({
            'fecha': date,
            'año': year,
            'mes': month,
            'trimestre': (month - 1) // 3 + 1,
            'ingresos_totales': round(ingresos, 2),
            'egresos_totales': round(egresos, 2),
            'cash_flow_real': round(cash_flow, 2),
            'clientes_activos': clientes_activos,
            'proyectos_pendientes': proyectos_pendientes,
            'dias_cartera': round(dias_cartera, 1),
            'ratio_cobros': round(ratio_cobros, 3),
            'estacionalidad': round(seasonal, 3),
            'is_low_demand': is_low_demand,
            'num_transacciones': num_transactions + int(np.random.normal(0, 50)),
        })

    df = pd.DataFrame(data)

    # Add derived features
    df['cash_flow_real'] = df['ingresos_totales'] - df['egresos_totales']

    return df


def generate_transaction_data(n_transactions: int = 15000) -> pd.DataFrame:
    """
    Generate detailed transaction-level data (optional, for granular analysis).

    Args:
        n_transactions: Total number of transactions to generate

    Returns:
        DataFrame with individual transactions
    """
    set_random_seed()

    transactions = []

    # Define transaction types with probabilities
    transaction_types = {
        'consultoria': {'prob': 0.45, 'min': 500, 'max': 50000},
        'mantenimiento': {'prob': 0.30, 'min': 200, 'max': 5000},
        'licencias': {'prob': 0.15, 'min': 100, 'max': 10000},
        'formacion': {'prob': 0.05, 'min': 300, 'max': 8000},
        'proveedores': {'prob': 0.25, 'min': 50, 'max': 15000},
        'nominas': {'prob': 0.10, 'min': 3000, 'max': 15000},
        'cloud_infra': {'prob': 0.08, 'min': 100, 'max': 5000},
        'alquiler': {'prob': 0.05, 'min': 2000, 'max': 3000},
    }

    categories = list(transaction_types.keys())
    probs = [transaction_types[t]['prob'] for t in categories]

    for _ in range(n_transactions):
        # Select transaction type
        tx_type = np.random.choice(categories, p=probs)
        tx_config = transaction_types[tx_type]

        # Generate amount
        amount = np.random.uniform(tx_config['min'], tx_config['max'])

        # Determine if ingreso or egreso
        if tx_type in ['consultoria', 'mantenimiento', 'licencias', 'formacion']:
            tipo = 'ingreso'
            # Add some days delay (negative = received, positive = pending)
            dias_vencimiento = int(np.random.normal(45, 15))
        else:
            tipo = 'egreso'
            dias_vencimiento = int(np.random.normal(-30, 5))  # Usually pay within 30 days

        # Random date within range
        start = datetime(2021, 1, 1)
        end = datetime(2023, 12, 31)
        delta = end - start
        random_days = np.random.randint(0, delta.days)
        fecha = start + timedelta(days=random_days)

        transactions.append({
            'fecha': fecha,
            'tipo': tipo,
            'categoria': tx_type,
            'cantidad': round(amount, 2),
            'dias_vencimiento': dias_vencimiento,
            'cliente_proveedor': f"{tx_type}_{np.random.randint(1, 20):03d}",
            'estado': 'completado' if np.random.random() > 0.1 else 'pendiente',
        })

    df = pd.DataFrame(transactions)
    df = df.sort_values('fecha').reset_index(drop=True)

    return df


def save_data(df: pd.DataFrame, filename: str = "cash_flow_data.csv") -> str:
    """Save generated data to CSV."""
    filepath = config.DATA_RAW / filename
    df.to_csv(filepath, index=False)
    return str(filepath)


if __name__ == "__main__":
    # Generate and save data
    print("Generating cash flow data...")
    df = generate_cash_flow_data()
    filepath = save_data(df)
    print(f"Data saved to: {filepath}")
    print(f"Shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nStatistics:")
    print(df.describe())
"""Cash Flow Forecasting - Evaluation Module"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import sys
sys.path.append(str(__file__).replace("evaluation.py", ""))

import config


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate evaluation metrics for forecast.

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        Dictionary with metrics
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    # MAPE - handle zeros
    non_zero_mask = y_true != 0
    if non_zero_mask.sum() > 0:
        mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
    else:
        mape = np.nan

    r2 = r2_score(y_true, y_pred)

    # Direction accuracy (trend prediction)
    if len(y_true) > 1:
        true_direction = np.sign(np.diff(y_true))
        pred_direction = np.sign(np.diff(y_pred))
        direction_accuracy = np.mean(true_direction == pred_direction) * 100
    else:
        direction_accuracy = np.nan

    # Theil's U (relative to naive forecast)
    if len(y_true) > 1:
        naive_errors = np.abs(np.diff(y_true))
        forecast_errors = np.abs(y_true[1:] - y_pred[1:])
        theil_u = np.sqrt(np.sum(forecast_errors**2)) / np.sqrt(np.sum(naive_errors**2))
    else:
        theil_u = np.nan

    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'mape': mape,
        'r2': r2,
        'direction_accuracy': direction_accuracy,
        'theil_u': theil_u
    }


def compare_models(df: pd.DataFrame,
                   arima_results: Dict,
                   nn_results: Dict,
                   gbm_results: Dict = None) -> pd.DataFrame:
    """
    Compare ARIMA, Neural Network, and GBM models.

    Args:
        df: Original DataFrame
        arima_results: Results from train_arima
        nn_results: Results from train_mlp
        gbm_results: Results from train_gbm (optional)

    Returns:
        DataFrame with comparison table
    """
    # Calculate metrics for ARIMA
    arima_metrics = calculate_metrics(
        arima_results['test_data'].values,
        arima_results['forecast']
    )

    # Calculate metrics for NN
    nn_metrics = calculate_metrics(
        nn_results['actual'],
        nn_results['predictions']
    )

    # Create comparison DataFrame
    comparison_data = {
        'Metric': ['MAE (€)', 'RMSE (€)', 'MAPE (%)', 'R²', 'Direction Accuracy (%)', "Theil's U"],
        'ARIMA': [
            arima_metrics['mae'],
            arima_metrics['rmse'],
            arima_metrics['mape'],
            arima_metrics['r2'],
            arima_metrics['direction_accuracy'],
            arima_metrics['theil_u']
        ],
        'Neural Network': [
            nn_metrics['mae'],
            nn_metrics['rmse'],
            nn_metrics['mape'],
            nn_metrics['r2'],
            nn_metrics['direction_accuracy'],
            nn_metrics['theil_u']
        ]
    }

    # Add GBM if provided
    if gbm_results:
        gbm_metrics = calculate_metrics(
            gbm_results['actual'],
            gbm_results['predictions']
        )
        comparison_data['GBM'] = [
            gbm_metrics['mae'],
            gbm_metrics['rmse'],
            gbm_metrics['mape'],
            gbm_metrics['r2'],
            gbm_metrics['direction_accuracy'],
            gbm_metrics['theil_u']
        ]

    comparison = pd.DataFrame(comparison_data)

    # Add winner column
    def get_winner(row):
        models = ['ARIMA', 'Neural Network']
        if 'GBM' in comparison.columns:
            models.append('GBM')

        if 'RMSE' in row['Metric'] or 'MAE' in row['Metric'] or 'MAPE' in row['Metric'] or 'U' in row['Metric']:
            # Lower is better
            values = {m: row[m] for m in models}
            min_model = min(values, key=values.get)
            return min_model
        elif 'R²' in row['Metric'] or 'Accuracy' in row['Metric']:
            # Higher is better
            values = {m: row[m] for m in models}
            max_model = max(values, key=values.get)
            return max_model
        return '-'

    comparison['Winner'] = comparison.apply(get_winner, axis=1)

    return comparison


def generate_forecast_report(df: pd.DataFrame,
                             arima_results: Dict,
                             nn_results: Dict) -> Dict:
    """
    Generate detailed forecast report.

    Args:
        df: Original DataFrame
        arima_results: ARIMA results
        nn_results: Neural Network results

    Returns:
        Dictionary with report data
    """
    report = {
        'period': f"{df['fecha'].min()} to {df['fecha'].max()}",
        'test_period': {
            'start': str(arima_results['test_data'].index[0])[:10],
            'end': str(arima_results['test_data'].index[-1])[:10]
        },
        'arima': calculate_metrics(
            arima_results['test_data'].values,
            arima_results['forecast']
        ),
        'nn': calculate_metrics(
            nn_results['actual'],
            nn_results['predictions']
        ),
        'predictions': {
            'arima': arima_results['forecast'].tolist(),
            'nn': nn_results['predictions'].tolist(),
            'actual': nn_results['actual'].tolist(),
            'dates': [str(d)[:10] for d in arima_results['test_data'].index]
        }
    }

    return report


def print_comparison_report(
    comparison: pd.DataFrame,
    arima_metrics: Dict = None,
    nn_metrics: Dict = None,
    gbm_metrics: Dict = None,
    thresholds: Dict = None
):
    """
    Print formatted comparison report.

    Args:
        comparison: Comparison DataFrame
        arima_metrics: Dict with ARIMA metrics
        nn_metrics: Dict with NN metrics
        gbm_metrics: Dict with GBM metrics
        thresholds: Optional evaluation thresholds
    """
    if thresholds is None:
        thresholds = config.EVALUATION_THRESHOLDS

    print("\n" + "=" * 80)
    print("MODEL COMPARISON REPORT - Cash Flow Forecasting")
    print("=" * 80)

    # Build header based on available models
    header = f"{'Metric':<25} {'ARIMA':>15} {'Neural Network':>18}"
    if gbm_metrics:
        header += " {'GBM':>12}"
    header += " {'Winner':>10}"
    print(header)
    print("-" * 80)

    for _, row in comparison.iterrows():
        line = f"{row['Metric']:<25} {row['ARIMA']:>15.2f} {row['Neural Network']:>18.2f}"
        if 'GBM' in row:
            line += f" {row['GBM']:>12.2f}"
        line += f" {row['Winner']:>10}"
        print(line)

    # Targets evaluation
    if arima_metrics and nn_metrics:
        print("\n" + "-" * 80)
        print("EVALUATION AGAINST TARGETS")
        print("-" * 80)

        header = f"{'Metric':<25} {'Target':>15} {'ARIMA':>15} {'NN':>15}"
        if gbm_metrics:
            header += " {'GBM':>15}"
        print(header)
        print("-" * 80)

        target_results = [
            ('MAE', arima_metrics['mae'], nn_metrics['mae'], 15000, True, gbm_metrics['mae'] if gbm_metrics else None),
            ('RMSE', arima_metrics['rmse'], nn_metrics['rmse'], 20000, True, gbm_metrics['rmse'] if gbm_metrics else None),
            ('R²', arima_metrics['r2'], nn_metrics['r2'], 0.70, False, gbm_metrics['r2'] if gbm_metrics else None),
        ]

        for name, arima_val, nn_val, target, is_lower_better, gbm_val in target_results:
            if is_lower_better:
                arima_pass = "[OK]" if arima_val < target else "[X]"
                nn_pass = "[OK]" if nn_val < target else "[X]"
                gbm_pass = "[OK]" if gbm_val < target else "[X]" if gbm_val else ""
            else:
                arima_pass = "[OK]" if arima_val > target else "[X]"
                nn_pass = "[OK]" if nn_val > target else "[X]"
                gbm_pass = "[OK]" if gbm_val > target else "[X]" if gbm_val else ""

            line = f"{name:<25} {target:>15} {arima_val:>12.2f} {arima_pass:<4} {nn_val:>12.2f} {nn_pass:<4}"
            if gbm_val is not None:
                line += f" {gbm_val:>12.2f} {gbm_pass:<4}"
            print(line)

    print("\n" + "=" * 80)


def evaluate_low_demand_detection(df: pd.DataFrame,
                                   predictions: np.ndarray,
                                   actual: np.ndarray,
                                   threshold: float = 50000) -> Dict:
    """
    Evaluate low demand detection for Telegram alerts.

    Args:
        df: Original DataFrame
        predictions: Model predictions
        actual: Actual cash flow values
        threshold: Low demand threshold in €

    Returns:
        Dictionary with detection metrics
    """
    # Detect low demand in actual
    actual_low = actual < threshold
    pred_low = predictions < threshold

    # Calculate detection metrics
    true_positives = np.sum(actual_low & pred_low)
    false_positives = np.sum(~actual_low & pred_low)
    false_negatives = np.sum(actual_low & ~pred_low)
    true_negatives = np.sum(~actual_low & ~pred_low)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (true_positives + true_negatives) / len(actual)

    return {
        'threshold': threshold,
        'true_positives': int(true_positives),
        'false_positives': int(false_positives),
        'false_negatives': int(false_negatives),
        'true_negatives': int(true_negatives),
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'accuracy': accuracy
    }


if __name__ == "__main__":
    from data_generator import generate_cash_flow_data
    from models.arima_model import train_arima
    from models.neural_network import train_mlp

    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    print("Generating data...")
    df = generate_cash_flow_data()

    print("\nTraining models...")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    arima_results = train_arima(df)
    nn_results = train_mlp(df)

    print("\nGenerating comparison...")
    comparison = compare_models(df, arima_results, nn_results)
    print_comparison_report(comparison)

    print("\nLow demand detection evaluation:")
    detection = evaluate_low_demand_detection(
        df, nn_results['predictions'], nn_results['actual']
    )
    print(f"Threshold: {detection['threshold']} €")
    print(f"Precision: {detection['precision']:.2%}")
    print(f"Recall: {detection['recall']:.2%}")
    print(f"F1 Score: {detection['f1_score']:.2%}")
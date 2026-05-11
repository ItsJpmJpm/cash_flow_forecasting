"""Cash Flow Forecasting - Main Script
Execute all models and generate comparison report
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.data_generator import generate_cash_flow_data, save_data
from src import config

def main():
    print("=" * 70)
    print("CASH FLOW FORECASTING - FULL PIPELINE")
    print("TechSolutions Toledo S.L.")
    print("=" * 70)

    # 1. Load/Generate Data
    print("\n[1/5] Generating synthetic data...")
    df = generate_cash_flow_data()
    data_path = save_data(df)
    print(f"  Data saved to: {data_path}")
    print(f"  Total months: {len(df)}")

    # 2. Train ARIMA
    print("\n[2/5] Training ARIMA model...")
    from src.models.arima_model import train_arima
    arima_results = train_arima(df)

    # Save ARIMA results
    os.makedirs("models/saved", exist_ok=True)
    with open("models/saved/arima_results.pkl", 'wb') as f:
        pickle.dump(arima_results, f)
    print("  ARIMA results saved to models/saved/arima_results.pkl")

    # 3. Train Neural Network (MLP)
    print("\n[3/5] Training Neural Network (MLP)...")
    from src.models.neural_network import train_mlp, train_gbm, train_rf
    nn_results = train_mlp(df, save_path="models/saved/mlp_model.pkl")
    print("  MLP model saved to models/saved/mlp_model.pkl")

    # 3b. Train Gradient Boosting (GBM) - better for small datasets
    print("\n[3b/5] Training Gradient Boosting (GBM)...")
    gbm_results = train_gbm(df, save_path="models/saved/gbm_model.pkl")
    print("  GBM model saved to models/saved/gbm_model.pkl")

    # 3c. Train Random Forest (RF) - robust ensemble
    print("\n[3c/5] Training Random Forest (RF)...")
    rf_results = train_rf(df, save_path="models/saved/rf_model.pkl")
    print("  RF model saved to models/saved/rf_model.pkl")

    # 4. Compare models
    print("\n[4/7] Comparing models...")
    from src.evaluation import compare_models, print_comparison_report

    comparison = compare_models(df, arima_results, nn_results, gbm_results, rf_results)
    print_comparison_report(
        comparison,
        arima_metrics=arima_results['metrics'],
        nn_metrics=nn_results['metrics'],
        gbm_metrics=gbm_results['metrics'],
        rf_metrics=rf_results['metrics'],
        thresholds=config.EVALUATION_THRESHOLDS
    )

# 5. Generate summary report

    print("\n[5/7] Generating summary report...")

    # Get test dates
    test_dates = df['fecha'].iloc[24:36].values
    test_dates_str = [str(d)[:10] for d in test_dates]

    # Create summary CSV
    summary_df = pd.DataFrame({
        'fecha': test_dates_str,
        'cash_flow_real': nn_results['actual'],
        'arima_prediccion': arima_results['forecast'],
        'nn_prediccion': nn_results['predictions'],
        'gbm_prediccion': gbm_results['predictions'],
        'rf_prediccion': rf_results['predictions'],
        'arima_error': arima_results['forecast'] - nn_results['actual'],
        'nn_error': nn_results['predictions'] - nn_results['actual'],
        'gbm_error': gbm_results['predictions'] - nn_results['actual'],
        'rf_error': rf_results['predictions'] - nn_results['actual']
    })
    summary_df.to_csv('models/saved/predictions_comparison.csv', index=False)
    print("  Summary saved to models/saved/predictions_comparison.csv")

    # Save final comparison
    comparison.to_csv('models/saved/model_comparison.csv', index=False)

    # 6. Print final summary
    print("\n[6/7] Final summary...")

    # Print final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"\n{'Model':<25} {'MAE (€)':>15} {'RMSE (€)':>15} {'MAPE (%)':>15}")
    print("-" * 70)
    print(f"{'ARIMA':<25} {arima_results['metrics']['mae']:>15,.2f} "
          f"{arima_results['metrics']['rmse']:>15,.2f} "
          f"{arima_results['metrics']['mape']:>15.2f}")
    print(f"{'Neural Network (MLP)':<25} {nn_results['metrics']['mae']:>15,.2f} "
          f"{nn_results['metrics']['rmse']:>15,.2f} "
          f"{nn_results['metrics']['mape']:>15.2f}")
    print(f"{'Gradient Boosting (GBM)':<25} {gbm_results['metrics']['mae']:>15,.2f} "
          f"{gbm_results['metrics']['rmse']:>15,.2f} "
          f"{gbm_results['metrics']['mape']:>15.2f}")
    print(f"{'Random Forest (RF)':<25} {rf_results['metrics']['mae']:>15,.2f} "
          f"{rf_results['metrics']['rmse']:>15,.2f} "
          f"{rf_results['metrics']['mape']:>15.2f}")

    # Determine winner
    arima_wins = 0
    nn_wins = 0
    gbm_wins = 0
    for _, row in comparison.iterrows():
        if row['Winner'] == 'ARIMA':
            arima_wins += 1
        elif row['Winner'] == 'NN':
            nn_wins += 1
        elif row['Winner'] == 'GBM':
            gbm_wins += 1

    print("\n" + "-" * 70)
    print(f"ARIMA wins: {arima_wins} metrics")
    print(f"Neural Network wins: {nn_wins} metrics")
    print(f"Gradient Boosting wins: {gbm_wins} metrics")

    # Find best model based on total metrics
    all_metrics = {
        'ARIMA': arima_results['metrics'],
        'Neural Network': nn_results['metrics'],
        'Gradient Boosting': gbm_results['metrics']
    }
    
    # Count how many metrics each model meets target
    targets = config.EVALUATION_THRESHOLDS
    best_model = None
    best_score = -1
    for name, metrics in all_metrics.items():
        score = 0
        if metrics['mae'] <= targets['mae_max']:
            score += 1
        if metrics['rmse'] <= targets['mae_max'] * 1.33:  # ~20k
            score += 1
        if metrics['mape'] <= targets['mape_max']:
            score += 1
        if metrics['r2'] >= targets['r2_min']:
            score += 1
        if score > best_score:
            best_score = score
            best_model = name
    
    if best_score >= 3:
        print(f"\n==> WINNER: {best_model} (meets {best_score}/4 targets)")
    else:
        print(f"\n==> RESULT: {best_model} is best but doesn't meet targets ({best_score}/4)")
    
    if gbm_wins >= nn_wins and gbm_wins >= arima_wins:
        print("==> GBM selected for production use")
    elif nn_wins > arima_wins:
        print("==> MLP selected for production use")
    else:
        print("==> ARIMA selected for production use")

    print("\n" + "=" * 70)
    print("EXECUTION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
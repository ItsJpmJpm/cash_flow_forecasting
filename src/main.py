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

    # 3. Train Neural Network
    print("\n[3/5] Training Neural Network (MLP)...")
    from src.models.neural_network import train_mlp
    nn_results = train_mlp(df, save_path="models/saved/mlp_model.pkl")
    print("  MLP model saved to models/saved/mlp_model.pkl")

    # 4. Compare models
    print("\n[4/5] Comparing models...")
    from src.evaluation import compare_models, print_comparison_report

    comparison = compare_models(df, arima_results, nn_results)
    print_comparison_report(
        comparison,
        arima_metrics=arima_results['metrics'],
        nn_metrics=nn_results['metrics'],
        thresholds=config.EVALUATION_THRESHOLDS
    )

    # 5. Generate summary report
    print("\n[5/5] Generating summary report...")

    # Get test dates
    test_dates = df['fecha'].iloc[24:36].values
    test_dates_str = [str(d)[:10] for d in test_dates]

    # Create summary CSV
    summary_df = pd.DataFrame({
        'fecha': test_dates_str,
        'cash_flow_real': nn_results['actual'],
        'arima_prediccion': arima_results['forecast'],
        'nn_prediccion': nn_results['predictions'],
        'arima_error': arima_results['forecast'] - nn_results['actual'],
        'nn_error': nn_results['predictions'] - nn_results['actual']
    })
    summary_df.to_csv('models/saved/predictions_comparison.csv', index=False)
    print("  Summary saved to models/saved/predictions_comparison.csv")

    # Save final comparison
    comparison.to_csv('models/saved/model_comparison.csv', index=False)

    # Print final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"\n{'Model':<20} {'MAE (€)':>15} {'RMSE (€)':>15} {'MAPE (%)':>15}")
    print("-" * 65)
    print(f"{'ARIMA':<20} {arima_results['metrics']['mae']:>15,.2f} "
          f"{arima_results['metrics']['rmse']:>15,.2f} "
          f"{arima_results['metrics']['mape']:>15.2f}")
    print(f"{'Neural Network':<20} {nn_results['metrics']['mae']:>15,.2f} "
          f"{nn_results['metrics']['rmse']:>15,.2f} "
          f"{nn_results['metrics']['mape']:>15.2f}")

    # Determine winner
    arima_wins = 0
    nn_wins = 0
    for _, row in comparison.iterrows():
        if row['Winner'] == 'ARIMA':
            arima_wins += 1
        elif row['Winner'] == 'NN':
            nn_wins += 1

    print("\n" + "-" * 65)
    print(f"ARIMA wins: {arima_wins} metrics")
    print(f"Neural Network wins: {nn_wins} metrics")

    if nn_wins > arima_wins:
        print("\n==> WINNER: Neural Network (MLP)")
    elif arima_wins > nn_wins:
        print("\n==> WINNER: ARIMA")
    else:
        print("\n==> RESULT: Draw - use ensemble")

    print("\n" + "=" * 70)
    print("EXECUTION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
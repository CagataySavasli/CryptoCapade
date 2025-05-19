import os
import json
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from lib import LSTMForecaster, TransformerForecaster
from lib.utils import (
    load_and_preprocess,
    create_feature_target,
    evaluate_sklearn_model,
    evaluate_arima,
    evaluate_pytorch_model
)
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


def main():
    # Settings
    SYMBOL = 'BTC-USD'
    START_DATE = '2018-01-01'
    END_DATE = datetime.today().strftime('%Y-%m-%d')
    WINDOW_SIZE = 30
    HORIZON = 1
    CV_SPLITS = 5

    # Load data and create features
    df = load_and_preprocess(SYMBOL, START_DATE, END_DATE)
    X, y = create_feature_target(df, WINDOW_SIZE, HORIZON)

    # Define models and hyperparameter grids
    sklearn_models = {
        'LinearRegression': LinearRegression(),
        'Lasso': Lasso(),
        'Ridge': Ridge(),
        'RandomForest': RandomForestRegressor(random_state=42),
        'XGB': XGBRegressor(objective='reg:squarederror', random_state=42)
    }
    sklearn_grids = {
        'Lasso': {'alpha': [0.001, 0.01, 0.1, 1]},
        'Ridge': {'alpha': [0.01, 0.1, 1, 10]},
        'RandomForest': {'n_estimators': [50, 100], 'max_depth': [5, 10]},
        'XGB': {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.05]}
    }
    lstm_grid = {
        'epochs': [20, 50],
        'batch_size': [16, 32],
        'lr': [1e-3, 5e-4],
        'hidden_size': [16, 32],
        'num_layers': [1, 2],
        'bidirectional': [False, True]
    }
    transformer_grid = {
        'epochs': [20, 50],
        'batch_size': [16, 32],
        'lr': [1e-3, 5e-4],
        'num_heads': [2, 4],
        'hidden_dim': [16, 32],
        'num_layers': [1, 2]
    }
    arima_order = (5, 1, 0)

    best_params = {}

    # Hyperparameter tuning for sklearn models
    for name, model in sklearn_models.items():
        if name in sklearn_grids:
            best_score = float('inf')
            best_p = None
            for combo in product(*sklearn_grids[name].values()):
                params = dict(zip(sklearn_grids[name].keys(), combo))
                model.set_params(**params)
                rmses = evaluate_sklearn_model(model, X, y, CV_SPLITS)
                score = np.mean(rmses)
                if score < best_score:
                    best_score = score
                    best_p = params
            best_params[name] = best_p
            print(f"Best params for {name}: {best_p}, RMSE={best_score:.4f}")
        else:
            best_params[name] = {}

    # ARIMA parameters
    best_params['ARIMA'] = {'order': arima_order}
    print(f"Using ARIMA order: {arima_order}")

    # Hyperparameter tuning for LSTM
    best_score = float('inf')
    best_p = None
    for combo in tqdm(list(product(*lstm_grid.values())), desc='LSTM tuning'):
        params = dict(zip(lstm_grid.keys(), combo))
        rmses = evaluate_pytorch_model(
            LSTMForecaster,
            X, y,
            CV_SPLITS,
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            lr=params['lr'],
            input_size=1,
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            # bidirectional=params['bidirectional']
        )
        score = np.mean(rmses)
        if score < best_score:
            best_score = score
            best_p = params
    best_params['LSTM'] = best_p
    print(f"Best params for LSTM: {best_p}, RMSE={best_score:.4f}")

    # Hyperparameter tuning for Transformer
    best_score = float('inf')
    best_p = None
    for combo in tqdm(list(product(*transformer_grid.values())), desc='Transformer tuning'):
        params = dict(zip(transformer_grid.keys(), combo))
        rmses = evaluate_pytorch_model(
            TransformerForecaster,
            X, y,
            CV_SPLITS,
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            lr=params['lr'],
            input_size=1,
            num_heads=params['num_heads'],
            hidden_dim=params['hidden_dim'],
            num_layers=params['num_layers']
        )
        score = np.mean(rmses)
        if score < best_score:
            best_score = score
            best_p = params
    best_params['Transformer'] = best_p
    print(f"Best params for Transformer: {best_p}, RMSE={best_score:.4f}")

    # Save best parameters
    os.makedirs('../output', exist_ok=True)
    with open('../output/best_parameters.json', 'w') as f:
        json.dump(best_params, f, indent=4)

    # Evaluation of best models
    cv_results = {}
    for name, model in sklearn_models.items():
        model.set_params(**best_params.get(name, {}))
        cv_results[name] = evaluate_sklearn_model(model, X, y, CV_SPLITS)
    cv_results['ARIMA'] = evaluate_arima(df['log_return'].reset_index(drop=True), arima_order, CV_SPLITS)
    cv_results['LSTM'] = evaluate_pytorch_model(
        LSTMForecaster, X, y, CV_SPLITS,
        epochs=best_params['LSTM']['epochs'],
        batch_size=best_params['LSTM']['batch_size'],
        lr=best_params['LSTM']['lr'],
        input_size=1,
        hidden_size=best_params['LSTM']['hidden_size'],
        num_layers=best_params['LSTM']['num_layers'],
        # bidirectional=best_params['LSTM']['bidirectional']
    )
    cv_results['Transformer'] = evaluate_pytorch_model(
        TransformerForecaster, X, y, CV_SPLITS,
        epochs=best_params['Transformer']['epochs'],
        batch_size=best_params['Transformer']['batch_size'],
        lr=best_params['Transformer']['lr'],
        input_size=1,
        num_heads=best_params['Transformer']['num_heads'],
        hidden_dim=best_params['Transformer']['hidden_dim'],
        num_layers=best_params['Transformer']['num_layers']
    )
    # Save best parameters
    os.makedirs('../output', exist_ok=True)
    with open('../output/best_params.json', 'w') as f:
        json.dump(best_params, f, indent=4)


    # Summary
    summary = pd.DataFrame({
        'Model': list(cv_results.keys()),
        'Avg_RMSE': [np.mean(v) for v in cv_results.values()]
    }).sort_values('Avg_RMSE')
    summary.to_csv('output/comparison_summary.csv', index=False)

    # Plots
    plt.figure()
    plt.bar(summary['Model'], summary['Avg_RMSE'])
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Avg CV RMSE')
    plt.title('Model Comparison')
    plt.tight_layout()
    plt.savefig('output/comparison_bar.png')

    plt.figure()
    plt.boxplot([cv_results[m] for m in summary['Model']], labels=summary['Model'])
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('CV RMSE Distribution')
    plt.title('RMSE Distribution')
    plt.tight_layout()
    plt.savefig('output/comparison_box.png')

    print("Finished model comparison.")

if __name__ == '__main__':
    main()

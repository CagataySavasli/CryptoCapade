import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

from lib import (
# Models:
    LSTMForecaster,
    TransformerForecaster,
# Data Download
    DataDownloader,
    YFinanceDataSource,
# Preprocessing
    DataPreprocessor,
    LogReturnStep,
    DropNaNStep,
# Dataset
    TimeSeriesDataset
)

from lib.utils import (
    load_and_preprocess,
    create_feature_target,
    evaluate_sklearn_model,
    evaluate_arima,
    evaluate_pytorch_model
)
def main():
    SYMBOL = 'BTC-USD'
    START_DATE = '2018-01-01'
    END_DATE = datetime.today().strftime('%Y-%m-%d')
    WINDOW_SIZE = 30
    HORIZON = 1
    CV_SPLITS = 5

    df = load_and_preprocess(SYMBOL, START_DATE, END_DATE)
    X, y = create_feature_target(df, WINDOW_SIZE, HORIZON)

    # Define models
    sklearn_models = {
        'LinearRegression': LinearRegression(),
        'Lasso': Lasso(),
        'Ridge': Ridge(),
        'RandomForest': RandomForestRegressor(random_state=42),
        'XGB': XGBRegressor(objective='reg:squarederror', random_state=42)
    }

    cv_results: Dict[str, list] = {}

    # Evaluate sklearn models
    for name, model in sklearn_models.items():
        rmses = evaluate_sklearn_model(model, X, y, CV_SPLITS)
        mean_rmse = np.mean(rmses)
        print(f'{name}: CV Mean RMSE = {mean_rmse:.6f} | folds = {rmses}')
        cv_results[name] = rmses

    # Evaluate ARIMA
    series = df['log_return'].reset_index(drop=True)
    arima_order = (5, 1, 0)
    arima_rmses = evaluate_arima(series, arima_order, CV_SPLITS)
    print(f'ARIMA{arima_order}: CV Mean RMSE = {np.mean(arima_rmses):.6f} | folds = {arima_rmses}')
    cv_results[f'ARIMA{arima_order}'] = arima_rmses

    # Evaluate LSTM
    lstm_rmses = evaluate_pytorch_model(
        LSTMForecaster, X, y, CV_SPLITS,
        epochs=5, batch_size=32, lr=1e-3,
        input_size=1, hidden_size=16, num_layers=1
    )
    print(f'LSTM: CV Mean RMSE = {np.mean(lstm_rmses):.6f} | folds = {lstm_rmses}')
    cv_results['LSTM'] = lstm_rmses

    # Evaluate Transformer
    trans_rmses = evaluate_pytorch_model(
        TransformerForecaster, X, y, CV_SPLITS,
        epochs=5, batch_size=32, lr=1e-3,
        input_size=1, num_heads=2, hidden_dim=16, num_layers=1
    )
    print(f'Transformer: CV Mean RMSE = {np.mean(trans_rmses):.6f} | folds = {trans_rmses}')
    cv_results['Transformer'] = trans_rmses

    # Ensure output directory exists
    root_output_dir = os.path.join('..', 'output')
    output_dir = os.path.join('..', 'output', 'results')
    os.makedirs(output_dir, exist_ok=True)

    # Save CV results for tables and plots
    with open(os.path.join(output_dir, 'cv_results.json'), 'w') as fp:
        json.dump(cv_results, fp, indent=4)
    print(f'Saved CV results to {os.path.join(output_dir, "cv_results.json")}')

    # Generate summary table
    df_summary = pd.DataFrame({
        'Model': list(cv_results.keys()),
        'Avg_CV_RMSE': [np.mean(v) for v in cv_results.values()]
    })
    summary_path = os.path.join(output_dir, 'performance_summary.csv')
    df_summary.to_csv(summary_path, index=False)
    print(f'Saved summary CSV to {summary_path}')

    # Bar chart of average RMSE
    plt.figure()
    plt.bar(df_summary['Model'], df_summary['Avg_CV_RMSE'])
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('CV RMSE')
    plt.title('Average CV RMSE by Model')
    plt.tight_layout()
    bar_path = os.path.join(output_dir, 'avg_rmse_bar.png')
    plt.savefig(bar_path)
    plt.close()
    print(f'Saved bar chart to {bar_path}')

    # Boxplot of RMSE distributions
    plt.figure()
    plt.boxplot([cv_results[m] for m in df_summary['Model']], labels=df_summary['Model'])
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('CV RMSE')
    plt.title('RMSE Distribution by Model')
    plt.tight_layout()
    box_path = os.path.join(output_dir, 'rmse_boxplot.png')
    plt.savefig(box_path)
    plt.close()
    print(f'Saved boxplot to {box_path}')

    # Select top 3 models by mean RMSE
    sorted_models = sorted(df_summary['Model'], key=lambda m: df_summary.loc[df_summary['Model']==m, 'Avg_CV_RMSE'].iloc[0])
    top3 = sorted_models[:3]
    print(f'Top 3 models: {top3}')

    # Hyperparameter grids
    param_grids: Dict[str, Any] = {
        'Lasso': {'alpha': [0.001, 0.01, 0.1, 1]},
        'Ridge': {'alpha': [0.01, 0.1, 1, 10]},
        'RandomForest': {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 20]},
        'XGB': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]}
    }

    best_params: Dict[str, Any] = {}
    for name in top3:
        if name in param_grids:
            grid = GridSearchCV(
                sklearn_models[name], param_grids[name],
                cv=TimeSeriesSplit(n_splits=CV_SPLITS), scoring='neg_root_mean_squared_error'
            )
            grid.fit(X, y)
            best_params[name] = grid.best_params_
            print(f'{name} best params: {grid.best_params_}')
        elif name.startswith('ARIMA'):
            best_params[name] = {'order': arima_order}
        elif name == 'LSTM':
            best_params[name] = {
                'input_size': 1, 'hidden_size': 16, 'num_layers': 1,
                'lr': 1e-3, 'epochs': 5, 'batch_size': 32
            }
        elif name == 'Transformer':
            best_params[name] = {
                'input_size': 1, 'num_heads': 2, 'hidden_dim': 16,
                'num_layers': 1, 'lr': 1e-3, 'epochs': 5, 'batch_size': 32
            }

    params_path = os.path.join(root_output_dir, 'best_params.json')
    with open(params_path, 'w') as fp:
        json.dump(best_params, fp, indent=4)
    print(f'Saved best parameters to {params_path}')


if __name__ == '__main__':
    main()
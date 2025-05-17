import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA

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
    evaluate_pytorch_model,
    train_lstm_full,
    train_transformer_full,
)
# Reuse model definitions from compare_models
def main():
    SYMBOL = 'BTC-USD'
    START_DATE = '2018-01-01'
    END_DATE = datetime.today().strftime('%Y-%m-%d')
    WINDOW_SIZE = 30
    HORIZON = 1

    # Load best parameters
    params_path = os.path.join('..', 'output', 'best_params.json')
    with open(params_path, 'r') as fp:
        best_params = json.load(fp)

    df = load_and_preprocess(SYMBOL, START_DATE, END_DATE)
    X, y = create_feature_target(df, WINDOW_SIZE, HORIZON)

    for name, params in best_params.items():
        if name in ['LinearRegression', 'Lasso', 'Ridge', 'RandomForest', 'XGB']:
            model_map = {
                'LinearRegression': LinearRegression(),
                'Lasso': Lasso(),
                'Ridge': Ridge(),
                'RandomForest': RandomForestRegressor(random_state=42),
                'XGB': XGBRegressor(objective='reg:squarederror', random_state=42)
            }
            model = model_map[name].set_params(**params)
            model.fit(X, y)
            joblib.dump(model, os.path.join('..', 'output', 'model', f'{name}_best.pkl'))
        elif name.startswith('ARIMA'):
            order = tuple(params['order'])
            series = df['log_return'].reset_index(drop=True)
            model = ARIMA(series, order=order).fit()
            model.save(os.path.join('..', 'output', 'model', f'{name}_best.pkl'))
        elif name == 'LSTM':
            model = train_lstm_full(X, y, **params)
            torch.save(model.state_dict(), os.path.join('..', 'output', 'model', 'LSTM_best.pt'))
        elif name == 'Transformer':
            model = train_transformer_full(X, y, **params)
            torch.save(model.state_dict(), os.path.join('..', 'output', 'model', 'Transformer_best.pt'))
        print(f'{name} trained and saved.')

if __name__ == '__main__':
    main()
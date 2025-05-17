import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from statsmodels.tsa.arima.model import ARIMA
import torch
import torch.nn as nn
import torch.optim as optim

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

def load_and_preprocess(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    downloader = DataDownloader(YFinanceDataSource())
    raw = downloader.download(symbol, start_date, end_date)
    raw['symbol'] = symbol
    preprocessor = DataPreprocessor([LogReturnStep(), DropNaNStep()])
    return preprocessor.run(raw)


def create_feature_target(df: pd.DataFrame, window_size: int, horizon: int):
    dataset = TimeSeriesDataset(
        df,
        symbol_col='symbol', date_col='date',
        feature_cols=['log_return'], target_col='log_return',
        window_size=window_size, horizon=horizon
    )
    X = np.array([x.numpy().flatten() for x, _ in dataset])
    y = np.array([y.item() for _, y in dataset])
    return X, y


def evaluate_sklearn_model(model, X: np.ndarray, y: np.ndarray, cv_splits: int) -> list:
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    rmses = []
    for train_idx, test_idx in tscv.split(X):
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[test_idx])
        rmses.append(np.sqrt(np.mean((preds - y[test_idx])**2)))
    return rmses


def evaluate_arima(series: pd.Series, order: tuple, cv_splits: int) -> list:
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    rmses = []
    for train_idx, test_idx in tscv.split(series):
        train_data = series.iloc[train_idx]
        test_data = series.iloc[test_idx]
        model = ARIMA(train_data, order=order).fit()
        preds = model.forecast(steps=len(test_data))
        rmses.append(np.sqrt(np.mean((preds - test_data)**2)))
    return rmses


def evaluate_pytorch_model(
    model_class, X: np.ndarray, y: np.ndarray,
    cv_splits: int, epochs: int, batch_size: int, lr: float, **model_kwargs
) -> list:
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rmses = []
    for train_idx, test_idx in tscv.split(X):
        X_train = torch.tensor(X[train_idx], dtype=torch.float32).unsqueeze(-1).to(device)
        y_train = torch.tensor(y[train_idx], dtype=torch.float32).unsqueeze(-1).to(device)
        X_test = torch.tensor(X[test_idx], dtype=torch.float32).unsqueeze(-1).to(device)
        y_test = torch.tensor(y[test_idx], dtype=torch.float32).unsqueeze(-1).to(device)

        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

        model = model_class(**model_kwargs).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        model.train()
        for _ in range(epochs):
            for xb, yb in train_loader:
                optimizer.zero_grad()
                preds = model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            preds = model(X_test)
        rmses.append(float(torch.sqrt(torch.mean((preds - y_test)**2)).item()))
    return rmses

def train_lstm_full(X, y, input_size, hidden_size, num_layers, lr, epochs, batch_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(-1).to(device)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(-1).to(device)
    dataset = torch.utils.data.TensorDataset(X_t, y_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = LSTMForecaster(input_size, hidden_size, num_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
    return model


def train_transformer_full(X, y, input_size, num_heads, hidden_dim, num_layers, lr, epochs, batch_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(-1).to(device)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(-1).to=device
    dataset = torch.utils.data.TensorDataset(X_t, y_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = TransformerForecaster(input_size, num_heads, hidden_dim, num_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
    return model
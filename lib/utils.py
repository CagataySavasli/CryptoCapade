import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from statsmodels.tsa.arima.model import ARIMA
import torch
import torch.nn as nn
import torch.optim as optim


from tqdm import trange

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
    model_class,
    X,                     # shape: (n_samples, seq_len)
    y,                     # shape: (n_samples,) or (n_samples, 1)
    cv_splits: int,
    epochs: int,
    batch_size: int,
    lr: float,
    **model_kwargs
):
    device = torch.device('cpu') #'torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rmses = []
    tscv = TimeSeriesSplit(n_splits=cv_splits)

    for train_idx, test_idx in tscv.split(X):
        # prepare data
        X_train = X[train_idx]
        y_train = y[train_idx]

        X_test = X[test_idx]
        y_test = y[test_idx]

        # convert to tensors
        X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1).to(device)
        y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1).to(device)
        X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1).to(device)
        y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1).to(device)

        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

        # instantiate model
        model = model_class(**model_kwargs).to(device)
        opt   = optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        # training loop
        model.train()
        for _ in range(epochs):
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                # print(xb.shape)
                # xb = xb.view(-1, xb.size(1), 1)  # Reshape to (batch_size, seq_len, input_size)
                output = model(xb)
                loss = loss_fn(output, yb)
                loss.backward()
                opt.step()

        # evaluation
        model.eval()
        with torch.no_grad():
            preds = model(X_test)
            rmse = torch.sqrt(torch.mean((preds - y_test) ** 2)).item()
        rmses.append(rmse)

        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return rmses


def train_lstm_full(X, y, input_size, hidden_size, num_layers, lr, epochs, batch_size):
    device = torch.device('cpu') #'torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    device = torch.device('cpu') #'torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
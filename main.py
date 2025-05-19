import os
import json
from datetime import datetime, timedelta
from typing import Dict

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import yfinance as yf
import torch
from statsmodels.tsa.arima.model import ARIMAResults
from sklearn.metrics import mean_squared_error

from lib import LSTMForecaster, TransformerForecaster

# Constants
MODEL_DIR = os.path.join('output', 'model')
RESULT_DIR = os.path.join('output')
SYMBOL = 'BTC-USD'
WINDOW_SIZE = 30
SKL_MODELS = ['LinearRegression', 'Lasso', 'Ridge', 'RandomForest', 'XGB']


class Service:
    def run(self):
        raise NotImplementedError("Service must implement run method")


class AIPredictionService(Service):
    def __init__(self):
        self.models = self._load_models()

    def _load_models(self) -> Dict[str, object]:
        params_path = os.path.join(RESULT_DIR, 'best_parameters.json')
        with open(params_path, 'r') as fp:
            best_params = json.load(fp)

        loaded = {}
        for name, params in best_params.items():
            if name in SKL_MODELS:
                model_file = os.path.join(MODEL_DIR, f"{name}_best.pkl")
                loaded[name] = joblib.load(model_file)
            elif name.startswith('ARIMA'):
                model_file = os.path.join(MODEL_DIR, f"{name}_best.pkl")
                loaded[name] = ARIMAResults.load(model_file)
            elif name == 'LSTM':
                model = LSTMForecaster(
                    input_size=1,
                    hidden_size=params['hidden_size'],
                    num_layers=params['num_layers']
                )
                model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'LSTM_best.pt')))
                model.eval()
                loaded[name] = model
            elif name == 'Transformer':
                model = TransformerForecaster(
                    input_size=1,
                    num_heads=params['num_heads'],
                    hidden_dim=params['hidden_dim'],
                    num_layers=params['num_layers']
                )
                model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'Transformer_best.pt')))
                model.eval()
                loaded[name] = model
        return loaded

    def _fetch_data(self) -> pd.DataFrame:
        end = datetime.today()
        start = end - timedelta(days=WINDOW_SIZE * 3)
        df = yf.download(
            SYMBOL,
            start=start.strftime('%Y-%m-%d'),
            end=end.strftime('%Y-%m-%d')
        )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[['Close']].dropna()
        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
        return df.dropna()

    def _fetch_data_range(self, start_date: datetime.date, end_date: datetime.date) -> pd.DataFrame:
        df = yf.download(
            SYMBOL,
            start=start_date.strftime('%Y-%m-%d'),
            end=(end_date + timedelta(days=1)).strftime('%Y-%m-%d')
        )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[['Close']].dropna()
        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
        return df.dropna()

    def _predict_historical(self, model_name: str, df: pd.DataFrame) -> pd.Series:
        returns = df['log_return'].values
        dates = df.index
        model = self.models[model_name]

        hist_preds = []
        prices = df['Close'].values

        for idx in range(len(dates)):
            if idx < WINDOW_SIZE:
                hist_preds.append(np.nan)
            else:
                window = returns[idx - WINDOW_SIZE: idx]
                if model_name.startswith('ARIMA'):
                    out = model.forecast(steps=1)
                    pred_ret = float(out.iloc[0])
                elif model_name in SKL_MODELS:
                    X_last = window.reshape(1, -1)
                    pred_ret = float(model.predict(X_last)[0])
                else:
                    tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
                    with torch.no_grad():
                        pred_ret = float(model(tensor).item())
                prev_price = prices[idx - 1]
                hist_preds.append(prev_price * np.exp(pred_ret))

        return pd.Series(data=hist_preds, index=dates)

    def _forecast(self, horizon: int) -> pd.DataFrame:
        df = self._fetch_data()
        last_returns = df['log_return'].values[-WINDOW_SIZE:]
        last_price = df['Close'].values[-1]
        dates = [datetime.today() + timedelta(days=i + 1) for i in range(horizon)]

        predictions = {}
        for name, model in self.models.items():
            preds = []
            window = np.array(last_returns, copy=True)

            for _ in range(horizon):
                if name.startswith('ARIMA'):
                    out = model.forecast(steps=1)
                    pred = float(out.iloc[0])
                elif name in SKL_MODELS:
                    X_last = window.reshape(1, -1)
                    pred = float(model.predict(X_last)[0])
                else:
                    tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
                    with torch.no_grad():
                        pred = float(model(tensor).item())
                preds.append(pred)
                window = np.roll(window, -1)
                window[-1] = pred

            prices = []
            price = last_price
            for r in preds:
                price = price * np.exp(r)
                prices.append(price)

            predictions[name] = prices

        return pd.DataFrame(predictions, index=pd.to_datetime(dates))

    def run(self):
        st.header("AI-Powered BTC Forecasting 🧠📈")
        with st.expander("How to use 💡"):
            st.write(
                "Select your model, choose a historical date range for backtesting, then view real vs. predicted prices and forecast future change percentages."
            )

        model_name = st.sidebar.selectbox("Choose Model", list(self.models.keys()))

        df_full = self._fetch_data()
        min_date = df_full.index.min().date()
        max_date = df_full.index.max().date()
        start_date = st.sidebar.date_input(
            "Backtesting start date",
            value=min_date,
            min_value=min_date,
            max_value=max_date,
            key="bt_start"
        )
        end_date = st.sidebar.date_input(
            "Backtesting end date",
            value=max_date,
            min_value=min_date,
            max_value=max_date,
            key="bt_end"
        )
        if start_date > end_date:
            st.sidebar.error("⛔ Start date must be on or before end date.")
            return

        horizon = st.sidebar.slider("Forecast Horizon (days)", 1, 30, 7)

        if st.sidebar.button("Run Forecast"):
            df_hist = self._fetch_data_range(start_date, end_date)
            hist_pred = self._predict_historical(model_name, df_hist)
            real = df_hist['Close'].loc[hist_pred.index]

            st.subheader(f"Backtesting Real vs Predicted ({model_name})")
            backtest_df = pd.DataFrame({
                "Real": real,
                "Predicted": hist_pred
            }).dropna()
            st.line_chart(backtest_df)

            real_clean = backtest_df['Real']
            pred_clean = backtest_df['Predicted']
            mse = mean_squared_error(real_clean, pred_clean)
            mape = np.mean(np.abs((real_clean - pred_clean) / real_clean)) * 100
            col1, col2 = st.columns(2)
            col1.metric("MAPE (%)", f"{mape:.2f}")
            col2.metric("RMSE", f"{np.sqrt(mse):.4f}")

            df_preds = self._forecast(horizon)
            st.subheader("Forecast Change (%)")
            pct = (df_preds[model_name] - df_preds[model_name].iloc[0]) / df_preds[model_name].iloc[0] * 100
            st.line_chart(pct.to_frame(name="Percent Change"))


class TrendTrackingService(Service):
    def _fetch_data(self) -> pd.DataFrame:
        end = datetime.today()
        start = end - timedelta(days=180)
        df = yf.download(
            SYMBOL,
            start=start.strftime('%Y-%m-%d'),
            end=end.strftime('%Y-%m-%d')
        )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df[['Close']].dropna()

    def run(self):
        st.header("BTC Trend Tracking 📊")
        df = self._fetch_data()
        df['MA7'] = df['Close'].rolling(7).mean()
        df['MA30'] = df['Close'].rolling(30).mean()
        st.line_chart(df)


def main():
    st.set_page_config(page_title="CryptoCapade", layout="wide")
    st.title("CryptoCapade 💎: AI-Enhanced BTC Services")
    text = """
    Welcome to CryptoCapade! 🚀
    It is a web application designed to provide AI-powered services for Bitcoin (BTC) trading and analysis.

    NOTE: Due to memory constraints, not all models can be deployed simultaneously.

    Developed as: CS552 - Data Science Project
    Developed by: Ahmet Çağatay Savaşlı, Nazar Aktosun
    """
    st.text(text)

    services = {
        "AI Powered Prediction": AIPredictionService,
        "Trend Tracking": TrendTrackingService
    }
    choice = st.sidebar.selectbox("Select a service", list(services.keys()))
    services[choice]().run()


if __name__ == "__main__":
    main()

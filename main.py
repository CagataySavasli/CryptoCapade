# streamlit_app.py
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

from lib import LSTMForecaster, TransformerForecaster

# Constants
MODEL_DIR = os.path.join('output', 'model')
RESULT_DIR = os.path.join('output')
SYMBOL = 'BTC-USD'
WINDOW_SIZE = 30

# Service base class
class Service:
    def run(self):
        raise NotImplementedError("Service must implement run method")

# AI-Powered Prediction Service
class AIPredictionService(Service):
    def __init__(self):
        self.models = self._load_models()

    def _load_models(self) -> Dict[str, object]:
        # Load best parameters to know which models
        params_path = os.path.join(RESULT_DIR, 'best_params.json')
        with open(params_path, 'r') as fp:
            best_params = json.load(fp)

        loaded = {}
        for name, params in best_params.items():
            if name in ['LinearRegression', 'Lasso', 'Ridge', 'RandomForest', 'XGB']:
                model_file = os.path.join(MODEL_DIR, f"{name}_best.pkl")
                loaded[name] = joblib.load(model_file)
            elif name.startswith('ARIMA'):
                model_file = os.path.join(MODEL_DIR, f"{name}_best.pkl")
                loaded[name] = ARIMAResults.load(model_file)
            elif name == 'LSTM':
                model = LSTMForecaster(
                    input_size=1, hidden_size=params['hidden_size'], num_layers=params['num_layers']
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
        df = yf.download(SYMBOL, start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'))
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[['Close']].dropna()
        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
        return df.dropna()

    def _forecast(self, horizon: int) -> pd.DataFrame:
        df = self._fetch_data()
        last_returns = df['log_return'].values[-WINDOW_SIZE:]
        last_price = df['Close'].values[-1]
        dates = [datetime.today() + timedelta(days=i+1) for i in range(horizon)]

        predictions = {}
        for name, model in self.models.items():
            preds = []
            window = np.array(last_returns, copy=True)

            for _ in range(horizon):
                if name.startswith('ARIMA'):
                    out = model.forecast(steps=1)
                    pred = float(out.iloc[0])
                elif name in ['LinearRegression', 'Lasso', 'Ridge', 'RandomForest', 'XGB']:
                    X_last = window.reshape(1, -1)
                    pred = float(model.predict(X_last)[0])
                else:  # PyTorch models
                    tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
                    with torch.no_grad():
                        pred = float(model(tensor).item())
                preds.append(pred)
                window = np.roll(window, -1)
                window[-1] = pred

            # Convert log-returns to price forecast
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
                "Choose a forecast horizon (in days) and view model predictions for BTC-USD. "
                "The top 3 pre-trained models generate log-return forecasts, converted to price projections."
            )

        horizon = st.sidebar.slider("Forecast Horizon (days)", min_value=1, max_value=30, value=7)
        if st.sidebar.button("Run Forecast"):
            df_preds = self._forecast(horizon)
            st.subheader("Forecasted Prices")
            st.table(df_preds.style.format("{:.2f}"))

            st.subheader("Forecast Plot")
            st.line_chart(df_preds)

# Trend Tracking Service
class TrendTrackingService(Service):
    def _fetch_data(self) -> pd.DataFrame:
        end = datetime.today()
        start = end - timedelta(days=180)
        df = yf.download(SYMBOL, start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'))
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df[['Close']].dropna()

    def run(self):
        st.header("BTC Trend Tracking 📊")
        with st.expander("How to use 💡"):
            st.write(
                "View recent BTC-USD price trend with moving averages. "
                "Compare raw price to 7-day and 30-day moving averages."
            )

        df = self._fetch_data()

        df['MA7'] = df['Close'].rolling(7).mean()
        df['MA30'] = df['Close'].rolling(30).mean()

        st.subheader("Price and Moving Averages")
        st.line_chart(df)

# Main App
def main():
    st.set_page_config(page_title="CryptoCapade", layout="wide")
    st.title("CryptoCapade 💎: AI-Enhanced BTC Financial Services")

    services = {
        "AI Powered Prediction": AIPredictionService,
        "Trend Tracking": TrendTrackingService
    }

    choice = st.sidebar.selectbox("Select a service", list(services.keys()))
    services[choice]().run()

if __name__ == "__main__":
    main()

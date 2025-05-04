
# utils/data_fetcher.py
import pandas as pd
import numpy as np
from datetime import datetime
import yfinance as yf
import time
import streamlit as st

@st.cache_data(ttl=24 * 3600, persist="disk")
def _download_crypto(
    ticker: str,
    start: str,
    end: str
) -> pd.DataFrame:
    """
    Download OHLCV data for a ticker with simple retry on empty results,
    and cache to disk for 24 hours so all processes reuse it.
    """
    for attempt in range(3):
        df = yf.download(
            tickers=ticker,
            start=start,
            end=end,
            progress=False,
            auto_adjust=True,
            threads=False,
            proxy=None,
        )
        # If we got no data, assume a transient rate limit or network hiccup
        if df.empty:
            time.sleep(2 ** attempt)
            continue
        return df
    # Final attempt; if still empty, raise
    df = yf.download(
        tickers=ticker,
        start=start,
        end=end,
        progress=False,
        auto_adjust=True,
        threads=False,
        proxy=None,
    )
    if df.empty:
        raise RuntimeError(f"Failed to fetch data for {ticker} after retries.")
    return df

class DataFetcher:
    """
    DataFetcher fetches cryptocurrency data using yfinance
    and returns a clean DataFrame ready for feature building.
    """

    def __init__(self):
        pass

    def get_crypto_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        compute_log_returns: bool = True,
        n_lags: int = 5
    ) -> pd.DataFrame:
        """
        Fetch historical crypto data, coerce types, set the Date index, and optionally add log returns.
        """
        ticker = f"{symbol}-USD"
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        # Download (cached, retried, and persisted to disk)
        df = _download_crypto(ticker, start_str, end_str)

        # Handle MultiIndex returned by yf.download
        if isinstance(df.columns, pd.MultiIndex) and "Ticker" in df.columns.names:
            df.columns = df.columns.droplevel("Ticker")

        # Select and coerce columns
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop missing rows
        df.dropna(subset=["Open", "High", "Low", "Close", "Volume"], inplace=True)

        # Compute log returns if requested
        if compute_log_returns:
            df["Log Returns"] = np.log(df["Close"] / df["Close"].shift(1))
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(subset=["Log Returns"], inplace=True)

        # Finalize index
        df.index.name = "Date"
        df.sort_index(inplace=True)

        return df


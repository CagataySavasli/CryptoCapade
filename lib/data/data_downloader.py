from abc import ABC, abstractmethod
import pandas as pd
import yfinance as yf


class DataSource(ABC):
    """
    Strategy interface for fetching time series data for a given symbol and date range.
    """
    @abstractmethod
    def fetch(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch data for `symbol` between `start_date` and `end_date`.

        Args:
            symbol (str): Asset ticker (e.g., 'BTC-USD').
            start_date (str): Inclusive start date in 'YYYY-MM-DD'.
            end_date (str): Exclusive end date in 'YYYY-MM-DD'.

        Returns:
            pd.DataFrame: DataFrame with columns ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', ...]
        """
        pass


class YFinanceDataSource(DataSource):
    """
    Concrete DataSource using yfinance as the backend.
    """
    def fetch(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        df = yf.download(
            tickers=symbol,
            start=start_date,
            end=end_date,
            interval="1d",
            progress=False,
            auto_adjust=True,
        )
        # Flatten multi-index columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        # Ensure date is a column
        df = df.reset_index()
        df.rename(columns={
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }, inplace=True)
        return df


class DataDownloader:
    """
    Context class that uses a DataSource strategy to download data.
    """
    def __init__(self, source: DataSource) -> None:
        self._source = source

    def download(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Download data for the specified symbol and date range.

        Returns a DataFrame with standardized column names and a 'date' column.
        """
        return self._source.fetch(symbol, start_date, end_date)
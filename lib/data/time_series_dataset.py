import torch
from torch.utils.data import Dataset
import pandas as pd
from typing import List, Tuple

class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for sliding-window time series forecasting per symbol.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        symbol_col: str,
        date_col: str,
        feature_cols: List[str],
        target_col: str,
        window_size: int,
        horizon: int = 1,
    ) -> None:
        """
        Args:
            df (pd.DataFrame): DataFrame containing symbol, date, features, and target.
            symbol_col (str): Column name for each time series identifier.
            date_col (str): Column name for chronological ordering.
            feature_cols (List[str]): Input feature columns.
            target_col (str): Column to predict.
            window_size (int): Number of past steps for X.
            horizon (int): Steps ahead to forecast.
        """
        self.symbol_col = symbol_col
        self.date_col = date_col
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.window_size = window_size
        self.horizon = horizon

        # Group by symbol and build index map
        self._data: dict[str, pd.DataFrame] = {}
        self._index_map: List[Tuple[str, int]] = []
        for symbol, grp in df.groupby(self.symbol_col):
            grp_sorted = grp.sort_values(by=self.date_col).reset_index(drop=True)
            self._data[symbol] = grp_sorted
            max_start = len(grp_sorted) - window_size - horizon + 1
            for start in range(max_start):
                self._index_map.append((symbol, start))

    def __len__(self) -> int:
        return len(self._index_map)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        symbol, start = self._index_map[idx]
        df_sym = self._data[symbol]

        # Positional indexing with iloc
        start_idx = start
        end_idx = start_idx + self.window_size
        x_np = df_sym[self.feature_cols].iloc[start_idx:end_idx].to_numpy(dtype=float)
        y_idx = start_idx + self.window_size + self.horizon - 1
        # Extract scalar value to avoid FutureWarning
        y_value = df_sym[self.target_col].iloc[y_idx].item()

        x_tensor = torch.tensor(x_np, dtype=torch.float32)
        y_tensor = torch.tensor(y_value, dtype=torch.float32)
        return x_tensor, y_tensor

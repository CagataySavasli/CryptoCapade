from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


class PreprocessingStep(ABC):
    """
    Strategy interface for a single preprocessing operation on a DataFrame.
    """
    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


class LogReturnStep(PreprocessingStep):
    """
    Adds a log-returns column to the DataFrame.
    """
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        return df


class DropNaNStep(PreprocessingStep):
    """
    Drops any rows containing NaN values.
    """
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.dropna().reset_index(drop=True)


class DataPreprocessor:
    """
    Orchestrates a pipeline of preprocessing steps.
    """
    def __init__(self, steps: list[PreprocessingStep]) -> None:
        self._steps = steps

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply each step in sequence and return the processed DataFrame.
        """
        for step in self._steps:
            df = step.apply(df)
        return df
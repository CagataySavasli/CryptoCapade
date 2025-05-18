# lib/model/lstm_forecaster.py

import torch
import torch.nn as nn

class LSTMForecaster(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        # batch_first=True olduğuna dikkat
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        # fc katmanı artık hidden_size → 1
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_len, input_size)
        out: (batch_size, seq_len, hidden_size)
        """
        out, _ = self.lstm(x)
        # Sadece son zaman adımının gizli durumunu al
        last_hidden = out[:, -1, :]             # → (batch_size, hidden_size)
        return self.fc(last_hidden)             # → (batch_size, 1)

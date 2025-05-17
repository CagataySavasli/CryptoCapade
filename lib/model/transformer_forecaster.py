import torch
import torch.nn as nn

class TransformerForecaster(nn.Module):
    def __init__(self, input_size: int, num_heads: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = self.input_proj(x)
        output = self.transformer(x_proj)
        return self.fc(output[:, -1, :])
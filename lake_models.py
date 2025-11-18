# lake_models.py
import torch
import torch.nn as nn


def make_rnn(cell_type: str, input_size: int, hidden_size: int, num_layers: int = 1):
    ct = cell_type.lower()
    if ct == "gru":
        rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
    elif ct == "lstm":
        rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
    elif ct in ("rnn", "vanilla"):
        rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                     nonlinearity="tanh", batch_first=True)
    else:
        raise ValueError(f"Unknown cell_type: {cell_type}")
    return rnn, ct


class LakeProfileSeq(nn.Module):
    """
    Sequence encoder (RNN family) over L×Du drivers window + param fusion → Dz profile next day.
    """
    def __init__(self, driver_dim: int, param_dim: int, depth_out: int,
                 hidden: int = 128, cell_type: str = "gru",
                 num_layers: int = 1, dropout_fc: float = 0.0):
        super().__init__()
        self.rnn, _ = make_rnn(cell_type, driver_dim, hidden, num_layers)
        self.head = nn.Sequential(
            nn.Linear(hidden + param_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_fc),
            nn.Linear(256, depth_out),
        )

    def forward(self, drivers_win: torch.Tensor, params_vec: torch.Tensor) -> torch.Tensor:
        """
        drivers_win: (B, L, Du)
        params_vec : (B, P)
        returns    : (B, Dz)
        """
        out, _ = self.rnn(drivers_win)        # (B, L, H)
        h_last = out[:, -1, :]                # (B, H)
        fused = torch.cat([h_last, params_vec], dim=-1)
        return self.head(fused)

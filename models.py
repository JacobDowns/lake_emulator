# models.py
import torch
import torch.nn as nn
from typing import Iterable, List, Tuple, Union, Optional

# ---------------------------
# Utilities
# ---------------------------
def build_mlp(
    in_dim: int,
    hidden_dims: Union[int, Iterable[int]] = 256,
    out_dim: int = 8,
    activation: nn.Module = nn.ReLU,
    dropout: float = 0.0,
) -> nn.Sequential:
    """
    Build a simple MLP:
      in_dim -> [hidden_dims...] -> out_dim
    activation is applied between hidden layers; dropout (if >0) after each activation.
    """
    if isinstance(hidden_dims, int):
        hidden_dims = [hidden_dims]
    layers: List[nn.Module] = []
    prev = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(activation())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)

class _SqueezeSingleStepMixin:
    """
    If W_y == 1 and squeeze_output=True, convert (B, 1, Dz) -> (B, Dz) for a consistent API.
    """
    def _maybe_squeeze(self, y: torch.Tensor, W_y: int, squeeze_output: bool) -> torch.Tensor:
        if squeeze_output and y.dim() == 3 and W_y == 1:
            return y[:, 0, :]
        return y

# ---------------------------
# Models
# ---------------------------
class ModelMLP(nn.Module, _SqueezeSingleStepMixin):
    """
    Flatten (B, W_x, Du) + params -> MLP -> (B, W_y, Dz) or (B, Dz) if W_y==1 and squeeze_output=True
    """
    def __init__(
        self,
        W_x: int,
        Du: int,
        P: int,
        Dz: int,
        W_y: int = 1,
        mlp_hidden: Union[int, Iterable[int]] = (256, 256),
        mlp_dropout: float = 0.0,
        activation: nn.Module = nn.ReLU,
        squeeze_output: bool = True,
    ):
        super().__init__()
        self.W_x, self.Du, self.P, self.Dz, self.W_y = W_x, Du, P, Dz, W_y
        self.squeeze_output = squeeze_output
        self.head = build_mlp(
            in_dim=W_x * Du + P,
            hidden_dims=mlp_hidden,
            out_dim=W_y * Dz,
            activation=activation,
            dropout=mlp_dropout,
        )

    def forward(self, x_win: torch.Tensor, p_vec: torch.Tensor) -> torch.Tensor:
        # x_win: (B, W_x, Du), p_vec: (B, P)
        B = x_win.shape[0]
        x_flat = x_win.reshape(B, self.W_x * self.Du)
        z = torch.cat([x_flat, p_vec], dim=-1)
        y = self.head(z).reshape(B, self.W_y, self.Dz)
        return self._maybe_squeeze(y, self.W_y, self.squeeze_output)

class ModelRNN(nn.Module, _SqueezeSingleStepMixin):
    def __init__(self,
                 cell="gru", 
                 Du=7, 
                 P=6, 
                 Dz=8,
                 hidden=128, 
                 num_layers=2, 
                 rnn_dropout=0.0,
                 W_y=1,
                 head_hidden=256,
                 head_dropout=0.0,
                 activation=nn.ReLU,
                 squeeze_output=True,
                 param_in_rnn: bool = True,
                 param_in_head: bool = True
        ):
        super().__init__()
        self.param_in_rnn = param_in_rnn
        self.param_in_head = param_in_head
        self.W_y, self.Dz, self.P = W_y, Dz, P
        self.squeeze_output = squeeze_output

        rnn_input_dim = Du + (P if param_in_rnn else 0)

        cell = cell.lower()
        if cell == "gru":
            self.rnn = nn.GRU(rnn_input_dim, hidden, num_layers=num_layers,
                              batch_first=True,
                              dropout=rnn_dropout if num_layers > 1 else 0.0)
        elif cell == "lstm":
            self.rnn = nn.LSTM(rnn_input_dim, hidden, num_layers=num_layers,
                               batch_first=True,
                               dropout=rnn_dropout if num_layers > 1 else 0.0)
        elif cell in ("rnn", "vanilla"):
            self.rnn = nn.RNN(rnn_input_dim, hidden, nonlinearity="tanh",
                              num_layers=num_layers,
                              batch_first=True,
                              dropout=rnn_dropout if num_layers > 1 else 0.0)
        else:
            raise ValueError("cell must be 'gru', 'lstm', or 'rnn'")

        head_in_dim = hidden + (P if param_in_head else 0)

        self.head = build_mlp(
            in_dim=head_in_dim,
            hidden_dims=head_hidden,
            out_dim=Dz,
            activation=activation,
            dropout=head_dropout,
        )

    def forward(self, x_win, p_vec):
        # x_win: (B, W_x, Du), p_vec: (B, P)
        if self.param_in_rnn:
            p_rep_rnn = p_vec.unsqueeze(1).expand(-1, x_win.size(1), -1)  # (B, W_x, P)
            x_in = torch.cat([x_win, p_rep_rnn], dim=-1)                   # (B, W_x, Du+P)
        else:
            x_in = x_win

        out, _ = self.rnn(x_in)           # (B, W_x, H)
        last_seq = out[:, -self.W_y:, :]  # (B, W_y, H)

        if self.param_in_head:
            p_rep_head = p_vec.unsqueeze(1).expand(-1, self.W_y, -1)  # (B, W_y, P)
            fused = torch.cat([last_seq, p_rep_head], dim=-1)         # (B, W_y, H+P)
        else:
            fused = last_seq                                          # (B, W_y, H)

        B, Wy, F = fused.shape
        y = self.head(fused.reshape(B * Wy, F)).reshape(B, Wy, self.Dz)
        return self._maybe_squeeze(y, self.W_y, self.squeeze_output)

class ModelCNN1D(nn.Module, _SqueezeSingleStepMixin):
    """
    Temporal 1D CNN over x_win; take last W_y steps; concat params per step; MLP head -> Dz
    Returns (B, W_y, Dz) or (B, Dz) if W_y==1 and squeeze_output=True
    """
    def __init__(
        self,
        Du: int = 7,
        P: int = 6,
        Dz: int = 8,
        W_y: int = 1,
        layers: int = 2,
        channels: int = 128,
        kernel: int = 5,
        conv_activation: nn.Module = nn.ReLU,
        head_hidden: Union[int, Iterable[int]] = 256,
        head_dropout: float = 0.0,
        activation: nn.Module = nn.ReLU,
        squeeze_output: bool = True,
    ):
        super().__init__()
        self.W_y, self.Dz, self.P = W_y, Dz, P
        self.squeeze_output = squeeze_output

        ks = kernel
        pad = ks // 2  # "same" padding (stride=1)
        convs: List[nn.Module] = []
        c_in = Du
        for _ in range(layers):
            convs += [nn.Conv1d(c_in, channels, ks, padding=pad), conv_activation()]
            c_in = channels
        self.conv = nn.Sequential(*convs)  # (B, C, W_x)

        # Head maps (channels + P) -> Dz per step
        self.head = build_mlp(
            in_dim=channels + P,
            hidden_dims=head_hidden,
            out_dim=Dz,
            activation=activation,
            dropout=head_dropout,
        )

    def forward(self, x_win: torch.Tensor, p_vec: torch.Tensor) -> torch.Tensor:
        # x_win: (B, W_x, Du) -> (B, Du, W_x)
        x = x_win.transpose(1, 2)        # (B, Du, W_x)
        f = self.conv(x)                 # (B, C, W_x)
        f = f.transpose(1, 2)            # (B, W_x, C)
        last = f[:, -self.W_y:, :]       # (B, W_y, C)

        # Tile params across steps and fuse
        p_rep = p_vec.unsqueeze(1).expand(-1, last.size(1), -1)      # (B, W_y, P)
        fused = torch.cat([last, p_rep], dim=-1)                     # (B, W_y, C+P)

        # Apply head per step
        B, Wy, CP = fused.shape
        y = self.head(fused.reshape(B * Wy, CP)).reshape(B, Wy, self.Dz)  # (B, W_y, Dz)
        return self._maybe_squeeze(y, self.W_y, self.squeeze_output)

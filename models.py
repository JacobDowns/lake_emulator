# models.py
import torch
import torch.nn as nn
from typing import Iterable, List, Union


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

    activation is applied between hidden layers; dropout (if >0)
    is applied after each activation.
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
    If W_y == 1 and squeeze_output=True, convert (B, 1, Dz) -> (B, Dz)
    so the rest of the code can always expect (B, Dz) for single-step. 
    """
    def _maybe_squeeze(self, y: torch.Tensor, W_y: int, squeeze_output: bool) -> torch.Tensor:
        if squeeze_output and y.dim() == 3 and W_y == 1:
            return y[:, 0, :]
        return y


# ---------------------------
# MLP model (baseline)
# ---------------------------

class ModelMLP(nn.Module, _SqueezeSingleStepMixin):
    """
    Baseline fully-connected model:
      Flatten (B, W_x, Du) + params -> MLP -> (B, W_y, Dz)
      or (B, Dz) if W_y==1 and squeeze_output=True.
    """
    def __init__(
        self,
        W_x: int,
        Du: int,
        P: int,
        Dz: int,
        W_y: int = 30,
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


# ---------------------------
# RNN/GRU/LSTM model (main one)
# ---------------------------

class ModelRNN(nn.Module, _SqueezeSingleStepMixin):
    """
    Temporal RNN/GRU/LSTM model with optional parameter-conditioned attention.

    - RNN processes the window x_win -> hidden states h_t.
    - If use_attention=False:
        use last W_y hidden states as before.
    - If use_attention=True:
        compute an attention-weighted sum over all W_x states, conditioned on p_vec,
        and use that context vector (repeated W_y times) instead.

    Input:
      x_win: (B, W_x, Du)
      p_vec: (B, P)

    Output:
      (B, W_y, Dz) or (B, Dz) if W_y==1 and squeeze_output=True.
    """
    def __init__(
        self,
        cell: str = "gru",
        Du: int = 7,
        P: int = 6,
        Dz: int = 8,
        hidden: int = 128,
        num_layers: int = 3,
        rnn_dropout: float = 0.1,
        W_y: int = 30,
        head_hidden: Union[int, Iterable[int]] = 128,
        head_dropout: float = 0.1,
        activation: nn.Module = nn.ReLU,
        squeeze_output: bool = True,
        param_in_rnn: bool = True,
        param_in_head: bool = True,
    ):
        super().__init__()
        self.param_in_rnn = param_in_rnn
        self.param_in_head = param_in_head
        self.W_y, self.Dz, self.P = W_y, Dz, P
        self.squeeze_output = squeeze_output

        # RNN input dimension
        rnn_input_dim = Du + (P if param_in_rnn else 0)

        self.rnn = nn.GRU(
            rnn_input_dim,
            hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=rnn_dropout if num_layers > 1 else 0.0,
        )
            
        # MLP head: maps [hidden (+params)] -> Dz per step
        head_in_dim = hidden + (P if param_in_head else 0)
        self.head = build_mlp(
            in_dim=head_in_dim,
            hidden_dims=head_hidden,
            out_dim=Dz,
            activation=activation,
            dropout=head_dropout,
        )

    def forward(self, x_win: torch.Tensor, p_vec: torch.Tensor) -> torch.Tensor:
        # x_win: (B, W_x, Du), p_vec: (B, P)
        if self.param_in_rnn:
            # Tile parameters across time steps and concatenate to inputs
            p_rep_rnn = p_vec.unsqueeze(1).expand(-1, x_win.size(1), -1)  # (B, W_x, P)
            x_in = torch.cat([x_win, p_rep_rnn], dim=-1)                   # (B, W_x, Du+P)
        else:
            x_in = x_win

        out, _ = self.rnn(x_in)           # (B, W_x, H)

        #Take the last W_y hidden states
        last_seq = out[:, -self.W_y:, :]  # (B, W_y, H)

        if self.param_in_head:
            p_rep_head = p_vec.unsqueeze(1).expand(-1, self.W_y, -1)  # (B, W_y, P)
            fused = torch.cat([last_seq, p_rep_head], dim=-1)         # (B, W_y, H+P)
        else:
            fused = last_seq                                          # (B, W_y, H)

        B, Wy, F = fused.shape
        y = self.head(fused.reshape(B * Wy, F)).reshape(B, Wy, self.Dz)
        return self._maybe_squeeze(y, self.W_y, self.squeeze_output)

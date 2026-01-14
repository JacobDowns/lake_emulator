# models.py
from __future__ import annotations
from typing import Iterable, List, Union
import torch
import torch.nn as nn


def build_mlp(
    in_dim: int,
    hidden_dims: Union[int, Iterable[int]] = (256, 256),
    out_dim: int = 1,
    activation: nn.Module = nn.ReLU,
    dropout: float = 0.0,
) -> nn.Sequential:
    if isinstance(hidden_dims, int):
        hidden_dims = [hidden_dims]

    layers: List[nn.Module] = []
    prev = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, int(h)))
        layers.append(activation())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        prev = int(h)
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class ModelRNNDepth(nn.Module):
    """
    GRU over time + depth-conditioned decoder.

    Inputs:
      x_win: (B, Wx, Du)
      p: (B, P)
      depth_feat: (B, Dz_max, Dd)

    Output:
      y: (B, Wy, Dz_max)   (or (B, Dz_max) if Wy==1 and squeeze_output=True)
    """

    def __init__(
        self,
        Du: int,
        P: int,
        Dd: int,
        hidden: int = 64,
        num_layers: int = 2,
        rnn_dropout: float = 0.0,
        Wy: int = 30,
        head_hidden: Union[int, Iterable[int]] = (256, 256),
        head_dropout: float = 0.0,
        activation: nn.Module = nn.ReLU,
        squeeze_output: bool = True,
        param_in_rnn: bool = True,
        param_in_head: bool = True,
    ):
        super().__init__()
        self.Du = int(Du)
        self.P = int(P)
        self.Dd = int(Dd)
        self.hidden = int(hidden)
        self.num_layers = int(num_layers)
        self.Wy = int(Wy)
        self.squeeze_output = bool(squeeze_output)
        self.param_in_rnn = bool(param_in_rnn)
        self.param_in_head = bool(param_in_head)

        rnn_in = self.Du + (self.P if self.param_in_rnn else 0)
        self.rnn = nn.GRU(
            input_size=rnn_in,
            hidden_size=self.hidden,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=float(rnn_dropout) if self.num_layers > 1 else 0.0,
        )

        head_in = self.hidden
        if self.param_in_head:
            head_in += self.P
        head_in += self.Dd  # depth conditioning

        self.head = build_mlp(
            in_dim=head_in,
            hidden_dims=head_hidden,
            out_dim=1,
            activation=activation,
            dropout=float(head_dropout),
        )

    def forward(self, x_win: torch.Tensor, p: torch.Tensor, depth_feat: torch.Tensor) -> torch.Tensor:
        """
        x_win: (B, Wx, Du)
        p: (B, P)
        depth_feat: (B, Dz, Dd)  (padded Dz_max)
        """
        B, Wx, _ = x_win.shape
        Dz = depth_feat.shape[1]
        Wy = self.Wy

        if self.param_in_rnn:
            p_rep = p.unsqueeze(1).expand(-1, Wx, -1)  # (B, Wx, P)
            x_in = torch.cat([x_win, p_rep], dim=-1)   # (B, Wx, Du+P)
        else:
            x_in = x_win

        out, _ = self.rnn(x_in)                        # (B, Wx, H)
        last_seq = out[:, -Wy:, :]                     # (B, Wy, H)

        # Expand across depths -> (B, Wy, Dz, H)
        h = last_seq.unsqueeze(2).expand(-1, -1, Dz, -1)

        parts = [h]
        if self.param_in_head:
            p_rep2 = p.unsqueeze(1).unsqueeze(2).expand(-1, Wy, Dz, -1)  # (B, Wy, Dz, P)
            parts.append(p_rep2)

        d_rep = depth_feat.unsqueeze(1).expand(-1, Wy, -1, -1)  # (B, Wy, Dz, Dd)
        parts.append(d_rep)

        fused = torch.cat(parts, dim=-1)  # (B, Wy, Dz, F)

        B2, Wy2, Dz2, F = fused.shape
        y = self.head(fused.reshape(B2 * Wy2 * Dz2, F)).reshape(B2, Wy2, Dz2)

        if self.squeeze_output and Wy == 1:
            return y[:, 0, :]  # (B, Dz)
        return y

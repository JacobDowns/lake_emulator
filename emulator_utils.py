"""
emulator_utils.py

Small, reusable utilities shared across evaluation/analysis scripts.

This module intentionally keeps:
- checkpoint loading (local path or MLflow run artifacts)
- (de)normalization helpers that match `multi_lake_dataset.load_multi_lake_data(..., normalize=True)`
- model reconstruction from a saved training config dict
- memory-friendly, batched windowed inference over a single time series

It intentionally does *not* include plotting code or dataset-specific logic.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from mlflow.tracking import MlflowClient

from models import ModelRNNDepth


# =========================================================
# Checkpoint loading
# =========================================================

def load_checkpoint(
    ckpt_or_run_id: str,
    *,
    dst_dir: str = "downloaded_artifacts",
    artifact_path: str = "checkpoints/best.pt",
) -> Tuple[str, Dict[str, Any]]:
    """
    Load a PyTorch checkpoint dict from either:
    - a local file path, or
    - an MLflow run id (downloads `artifact_path` into `dst_dir`)

    Returns `(local_path, checkpoint_dict)`.
    """
    os.makedirs(dst_dir, exist_ok=True)

    if os.path.isfile(ckpt_or_run_id):
        local_path = ckpt_or_run_id
    else:
        client = MlflowClient()
        local_path = client.download_artifacts(ckpt_or_run_id, artifact_path, dst_dir)

    # In PyTorch >=2.6, weights_only default changed; we want full dict (config + arrays).
    try:
        ckpt = torch.load(local_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(local_path, map_location="cpu")

    if not isinstance(ckpt, dict):
        raise TypeError(f"Expected checkpoint dict at {local_path}, got {type(ckpt)}")

    return local_path, ckpt


def load_checkpoint_from_mlflow(
    run_id: str,
    *,
    dst_dir: str = "downloaded_artifacts",
    artifact_path: str = "checkpoints/best.pt",
) -> Tuple[str, Dict[str, Any]]:
    """
    Convenience wrapper around `load_checkpoint` for callers that always pass an MLflow `run_id`.
    """
    return load_checkpoint(run_id, dst_dir=dst_dir, artifact_path=artifact_path)


# =========================================================
# Normalization / denormalization
# =========================================================

def _norms_2d(norms_arr: np.ndarray) -> np.ndarray:
    """
    Normalize norms array shapes to `(1, D)`.

    Accepts:
    - `(D,)`
    - `(1, D)`
    - `(1, 1, D)`
    """
    a = np.array(norms_arr)
    if a.ndim in (1, 2, 3):
        return a.reshape(1, -1).astype(np.float32)
    raise ValueError(f"Unexpected norms shape: {a.shape}")


def denorm_weather(drivers_norm: np.ndarray, norms: Optional[Dict[str, Any]]) -> np.ndarray:
    """
    Denormalize weather drivers.

    `drivers_norm`: `(..., Du)` normalized
    """
    if norms is None:
        return drivers_norm
    mu = _norms_2d(norms["weather_mean"])  # (1,Du)
    sd = _norms_2d(norms["weather_std"])   # (1,Du)
    return drivers_norm * sd + mu


def denorm_params(params_norm: np.ndarray, norms: Optional[Dict[str, Any]]) -> np.ndarray:
    """
    Denormalize simulator parameter vectors.

    `params_norm`: `(..., P)` normalized
    """
    if norms is None:
        return params_norm
    mu = _norms_2d(norms["params_mean"])  # (1,P)
    sd = _norms_2d(norms["params_std"])   # (1,P)
    return params_norm * sd + mu


def denorm_outputs(
    y_norm: np.ndarray,
    norms: Optional[Dict[str, Any]],
    *,
    Dz: Optional[int] = None,
) -> np.ndarray:
    """
    Denormalize lake temperature outputs.

    `y_norm`: `(..., Dz_lake)` or `(..., Dzmax)` normalized
    `norms["output_mean"/"output_std"]` are stored for the training global `Dzmax`.
    If `Dz` is provided, the norms are sliced to the first `Dz` entries.
    """
    if norms is None:
        return y_norm
    mu = _norms_2d(norms["output_mean"])  # (1,Dzmax)
    sd = _norms_2d(norms["output_std"])   # (1,Dzmax)
    if Dz is not None:
        mu = mu[:, :Dz]
        sd = sd[:, :Dz]
    return y_norm * sd + mu


# =========================================================
# Split indexing
# =========================================================

def split_time_indices(year: np.ndarray, split: str, split_years: Tuple[int, int, int]) -> np.ndarray:
    """
    Return indices into a full time axis given a year array and split definition.

    Supported splits: `train`, `val`, `test`, `valtest`, `all`.
    """
    tr_end, va_end, _ = split_years
    if split == "train":
        return np.where(year <= tr_end)[0]
    if split == "val":
        return np.where((year > tr_end) & (year <= va_end))[0]
    if split == "test":
        return np.where(year > va_end)[0]
    if split == "valtest":
        idx_val = np.where((year > tr_end) & (year <= va_end))[0]
        idx_test = np.where(year > va_end)[0]
        return np.concatenate([idx_val, idx_test], axis=0)
    if split == "all":
        return np.arange(year.shape[0], dtype=np.int32)
    raise ValueError("split must be one of train|val|test|valtest|all")


# =========================================================
# Model reconstruction from checkpoint config
# =========================================================

def parse_hidden_list(s: Any) -> List[int]:
    """
    Parse a hidden layer list from either:
    - list/tuple of ints
    - comma-separated string (e.g. "256,256")
    """
    if isinstance(s, (list, tuple)):
        return [int(x) for x in s]
    s = str(s)
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_bool(x: Any, default: bool) -> bool:
    if x is None:
        return bool(default)
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, np.integer)):
        return bool(int(x))
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("1", "true", "t", "yes", "y"):
            return True
        if s in ("0", "false", "f", "no", "n"):
            return False
    return bool(x)


def build_model_from_ckpt_config(cfg: Dict[str, Any], Du: int, P: int, Dd: int) -> ModelRNNDepth:
    """
    Reconstruct a `ModelRNNDepth` instance from a checkpoint `config` dict.

    The training script typically logs `vars(args)` so values may be strings; this helper
    defensively casts types and handles common boolean encodings.
    """
    Wy = int(cfg.get("Wy", 30))
    hidden = int(cfg.get("hidden", 64))
    num_layers = int(cfg.get("num_layers", 2))
    rnn_dropout = float(cfg.get("rnn_dropout", 0.0))
    head_dropout = float(cfg.get("head_dropout", 0.0))
    head_hidden = parse_hidden_list(cfg.get("head_hidden", "256,256"))

    param_in_rnn = _parse_bool(cfg.get("param_in_rnn", True), True)
    param_in_head = _parse_bool(cfg.get("param_in_head", True), True)

    return ModelRNNDepth(
        Du=Du,
        P=P,
        Dd=Dd,
        hidden=hidden,
        num_layers=num_layers,
        rnn_dropout=rnn_dropout,
        Wy=Wy,
        head_hidden=head_hidden,
        head_dropout=head_dropout,
        param_in_rnn=param_in_rnn,
        param_in_head=param_in_head,
        squeeze_output=False,  # keep (B,Wy,Dz) for accumulation
    )


# =========================================================
# Batched window inference for one sim
# =========================================================

@torch.no_grad()
def predict_timeseries_for_sim_batched(
    model: torch.nn.Module,
    drivers: np.ndarray,  # (T, Du) normalized
    p_vec: np.ndarray,  # (P,) normalized
    depth_feat_padded: np.ndarray,  # (Dzmax, Dd)
    *,
    Wx: int,
    Wy: int,
    stride: int,
    device: str,
    batch_windows: int,
) -> np.ndarray:
    """
    Memory-friendly sliding-window inference over a single simulation's time series.

    Returns:
      `pred`: `(T, Dzmax)` normalized, filled with `NaN` where no prediction is available.

    Window alignment:
      Each window starting at `s` predicts the time indices
      `[s + (Wx - Wy) ... s + Wx - 1]` (length `Wy`).
    """
    T, _ = drivers.shape
    Dzmax = depth_feat_padded.shape[0]
    if T < Wx:
        raise ValueError(f"T={T} < Wx={Wx}")

    starts = np.arange(0, T - Wx + 1, stride, dtype=np.int32)
    W = starts.shape[0]

    preds_sum = np.zeros((T, Dzmax), dtype=np.float32)
    preds_cnt = np.zeros((T,), dtype=np.int32)

    # constant tensors
    p_t = torch.from_numpy(p_vec.astype(np.float32, copy=False)).to(device=device).unsqueeze(0)  # (1,P)
    df_t = (
        torch.from_numpy(depth_feat_padded.astype(np.float32, copy=False)).to(device=device).unsqueeze(0)
    )  # (1,Dzmax,Dd)

    for i0 in range(0, W, batch_windows):
        i1 = min(W, i0 + batch_windows)
        batch_starts = starts[i0:i1]  # (B,)

        x_batch = np.stack([drivers[s : s + Wx, :] for s in batch_starts], axis=0).astype(np.float32, copy=False)
        x_t = torch.from_numpy(np.ascontiguousarray(x_batch)).to(device=device)  # (B,Wx,Du)
        B = int(x_t.shape[0])

        p_bt = p_t.expand(B, -1)  # (B,P)
        df_bt = df_t.expand(B, -1, -1)  # (B,Dzmax,Dd)

        y = model(x_t, p_bt, df_bt)  # (B,Wy,Dzmax)
        if y.dim() == 2:
            y = y.unsqueeze(1)

        y_np = y.detach().cpu().numpy().astype(np.float32, copy=False)  # (B,Wy,Dzmax)

        for bi, s in enumerate(batch_starts):
            base_t = int(s + (Wx - Wy))
            t0 = base_t
            t1 = base_t + Wy
            if t0 < 0 or t1 > T:
                continue
            preds_sum[t0:t1, :] += y_np[bi]
            preds_cnt[t0:t1] += 1

    pred = np.full((T, Dzmax), np.nan, dtype=np.float32)
    mask = preds_cnt > 0
    pred[mask, :] = preds_sum[mask, :] / preds_cnt[mask, None]
    return pred


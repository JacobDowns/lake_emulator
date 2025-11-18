# lake_utils.py
import subprocess
import torch

def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())

def log_git_tags_safe(mlflow):
    try:
        commit = subprocess.check_output(["git","rev-parse","HEAD"], text=True).strip()
        dirty  = subprocess.check_output(["git","status","--porcelain"], text=True).strip()
        mlflow.set_tag("git.commit", commit)
        mlflow.set_tag("git.dirty", "1" if dirty else "0")
    except Exception:
        pass

def per_depth_rmse_mean(loader, model, device) -> float:
    sse = None; n = 0
    with torch.no_grad():
        for x, p, y in loader:
            x = x.float().to(device); p = p.float().to(device); y = y.float().to(device)
            pred = model(x, p)
            err2 = (pred - y) ** 2
            sse = err2.sum(dim=0) if sse is None else sse + err2.sum(dim=0)
            n += y.shape[0]
    rmse_depth = torch.sqrt(sse / max(1, n))
    return float(rmse_depth.mean().item())

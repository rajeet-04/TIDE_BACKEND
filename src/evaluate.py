"""Phase 4: Performance metrics from the paper (eqs 28-31).

bias (b)        : mean(predicted - observed)
RMSE            : sqrt(mean((predicted - observed)^2))
R (Pearson)     : correlation coefficient
SR (sym. slope) : sqrt(sum(predicted^2) / sum(observed^2))
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Metrics:
    n: int
    bias: float
    rmse: float
    r: float
    sr: float

    def as_dict(self) -> dict[str, float]:
        return {
            "n": self.n,
            "bias_m": self.bias,
            "rmse_m": self.rmse,
            "r": self.r,
            "sr": self.sr,
        }


def evaluate(predicted: np.ndarray, observed: np.ndarray) -> Metrics:
    pred = np.asarray(predicted, dtype=float)
    obs = np.asarray(observed, dtype=float)
    if pred.shape != obs.shape:
        raise ValueError(f"shape mismatch: {pred.shape} vs {obs.shape}")
    n = pred.size
    diff = pred - obs
    bias = float(diff.mean())
    rmse = float(np.sqrt(np.mean(diff**2)))
    pred_dm = pred - pred.mean()
    obs_dm = obs - obs.mean()
    denom = np.sqrt((pred_dm**2).sum() * (obs_dm**2).sum())
    r = float((pred_dm * obs_dm).sum() / denom) if denom > 0 else float("nan")
    sr = float(np.sqrt((pred**2).sum() / (obs**2).sum())) if (obs**2).sum() > 0 else float("nan")
    return Metrics(n=n, bias=bias, rmse=rmse, r=r, sr=sr)

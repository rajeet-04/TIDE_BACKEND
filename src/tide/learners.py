"""Regressors for model B's corrections (spec 4.3, 4.5).

LightGBM, the legacy calibrator's ExtraTrees settings, XGBoost and a small PyTorch network,
plus weighted blends of these. Each trains on the CPU with one thread and fixed seeds, so
identical inputs give identical backtest metrics (spec 8) and a model chosen here can be
refit unchanged on the CPU VM.
"""
from __future__ import annotations

import re

import numpy as np

LGBM = dict(n_estimators=500, learning_rate=0.03, subsample=0.8, subsample_freq=1, colsample_bytree=0.9,
            random_state=1, deterministic=True, force_row_wise=True, n_jobs=1, verbose=-1)

def make_learner(name: str):
    """An unfitted regressor with fit(X, y) and predict(X), by name:
    - lgbm-l<leaves>-m<min child samples>, optionally ending in -huber (Huber loss, delta 0.3)
    - et: the legacy ExtraTrees settings
    - xgb-d<max depth>
    - mlp: the small PyTorch network
    - blend(<name>=<weight>,...): a weighted average; weights sum to 1"""
    if match := re.fullmatch(r"lgbm-l(\d+)-m(\d+)(-huber)?", name):
        from lightgbm import LGBMRegressor

        huber = {"objective": "huber", "alpha": 0.3} if match.group(3) else {}
        return LGBMRegressor(num_leaves=int(match.group(1)), min_child_samples=int(match.group(2)), **LGBM, **huber)
    if name == "et":
        from sklearn.ensemble import ExtraTreesRegressor

        return ExtraTreesRegressor(n_estimators=160, min_samples_leaf=15, max_features=0.9, random_state=1, n_jobs=1)
    if match := re.fullmatch(r"xgb-d(\d+)", name):
        from xgboost import XGBRegressor

        return XGBRegressor(n_estimators=500, learning_rate=0.03, max_depth=int(match.group(1)), subsample=0.8,
                            colsample_bytree=0.9, tree_method="hist", device="cpu", random_state=1, n_jobs=1)
    if name == "mlp":
        return MLP()
    if match := re.fullmatch(r"blend\((.+)\)", name):
        return Blend(_members(match.group(1)))
    raise ValueError(f"unknown learner {name!r}")

def _members(text: str) -> list[tuple[str, float]]:
    members = []
    for part in text.split(","):
        name, _, weight = part.rpartition("=")
        members.append((name, float(weight)))
    if abs(sum(weight for _, weight in members) - 1.0) > 1e-9 or any(n.startswith("blend") for n, _ in members):
        raise ValueError(f"a blend needs plain learners with weights summing to 1: {text!r}")
    return members

class Blend:
    """A weighted average of plain learners, each fitted on the same data (spec 4.5)."""

    def __init__(self, members: list[tuple[str, float]]):
        self.members = members

    def fit(self, X, y) -> "Blend":
        self.fitted_ = [(make_learner(name).fit(X, y), weight) for name, weight in self.members]
        return self

    def predict(self, X) -> np.ndarray:
        return sum(weight * model.predict(X) for model, weight in self.fitted_)

class MLP:
    """Two hidden layers of 64 units on standardised inputs and target: Adam, 30 epochs,
    batches of 1024, seed 1, deterministic single-thread CPU arithmetic."""

    def __init__(self, hidden: int = 64, epochs: int = 30, batch: int = 1024, rate: float = 1e-3):
        self.hidden, self.epochs, self.batch, self.rate = hidden, epochs, batch, rate

    def fit(self, X, y) -> "MLP":
        import torch

        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(1)
        x = np.asarray(X, dtype=float)
        target = np.asarray(y, dtype=float)
        self.x_mean_, self.x_scale_ = x.mean(axis=0), x.std(axis=0) + 1e-9
        self.y_mean_, self.y_scale_ = float(target.mean()), float(target.std()) + 1e-9
        inputs = torch.tensor((x - self.x_mean_) / self.x_scale_, dtype=torch.float32)
        outputs = torch.tensor((target - self.y_mean_) / self.y_scale_, dtype=torch.float32)[:, None]
        self.net_ = torch.nn.Sequential(torch.nn.Linear(x.shape[1], self.hidden), torch.nn.SiLU(),
                                        torch.nn.Linear(self.hidden, self.hidden), torch.nn.SiLU(),
                                        torch.nn.Linear(self.hidden, 1))
        optimiser = torch.optim.Adam(self.net_.parameters(), lr=self.rate)
        order = torch.Generator().manual_seed(1)
        for _ in range(self.epochs):
            for batch in torch.randperm(len(inputs), generator=order).split(self.batch):
                optimiser.zero_grad()
                torch.nn.functional.mse_loss(self.net_(inputs[batch]), outputs[batch]).backward()
                optimiser.step()
        return self

    def predict(self, X) -> np.ndarray:
        import torch

        with torch.no_grad():
            inputs = torch.tensor((np.asarray(X, dtype=float) - self.x_mean_) / self.x_scale_, dtype=torch.float32)
            return self.net_(inputs).numpy()[:, 0].astype(float) * self.y_scale_ + self.y_mean_

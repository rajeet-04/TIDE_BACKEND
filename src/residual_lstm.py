"""Phase 6: LSTM residual model.

Trained on (observed - LSM_prediction) to capture non-astronomical signals:
storm surge, monsoon discharge, atmospheric pressure variation.

Architecture
------------
Input window: previous 168 hourly residuals (1 week).
Per-step features per timestep:
    [residual_t,
     hour_sin, hour_cos,           # hour of day (period 24)
     day_sin, day_cos,             # day of year (period 365.25)
     month_sin, month_cos]         # month (period 12)

The LSTM emits a single scalar: the predicted residual for the next hour.
At inference time we run autoregressively, appending each prediction to the
window for the next step.

Training details
----------------
- Walk-forward: chronological 80/10/10 train/val/test split of the residuals.
- Adam, MSE loss, early stopping on validation MSE.
- LSM mean is added back at the end so the final number is a tidal height.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluate as eval_mod
import lsm
import train_predict as tp

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
MODEL_DIR = OUTPUT_DIR / "lstm_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

WINDOW = 168            # 1 week of hourly history
N_FEATURES = 7
HIDDEN = 96
LAYERS = 2
DROPOUT = 0.15
BATCH = 1024
EPOCHS = 40
LR = 1e-3
PATIENCE = 5


def _time_features(dt: pd.Series) -> np.ndarray:
    """Return (n, 6) of sin/cos features for hour-of-day, day-of-year, month."""
    hour = dt.dt.hour.to_numpy()
    doy = dt.dt.dayofyear.to_numpy()
    month = dt.dt.month.to_numpy()
    h_rad = 2 * np.pi * hour / 24.0
    d_rad = 2 * np.pi * doy / 365.25
    m_rad = 2 * np.pi * month / 12.0
    return np.stack([
        np.sin(h_rad), np.cos(h_rad),
        np.sin(d_rad), np.cos(d_rad),
        np.sin(m_rad), np.cos(m_rad),
    ], axis=1).astype(np.float32)


class ResidualWindowDataset(Dataset):
    """Sliding-window dataset over residuals + time features."""

    def __init__(self, residuals: np.ndarray, time_feat: np.ndarray, window: int = WINDOW):
        if residuals.shape[0] != time_feat.shape[0]:
            raise ValueError("residuals and time_feat must have same length")
        # Per-step feature: [residual, hour_sin, hour_cos, day_sin, day_cos, month_sin, month_cos]
        feats = np.concatenate([residuals[:, None], time_feat], axis=1).astype(np.float32)
        self.X = feats
        self.r = residuals.astype(np.float32)
        self.window = window

    def __len__(self) -> int:
        return max(0, self.X.shape[0] - self.window)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        xs = self.X[idx: idx + self.window]
        y = self.r[idx + self.window]
        return torch.from_numpy(xs), torch.tensor(y, dtype=torch.float32)


class ResidualLSTM(nn.Module):
    def __init__(self, n_features: int = N_FEATURES, hidden: int = HIDDEN,
                 layers: int = LAYERS, dropout: float = DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


def _build_residuals(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, datetime]:
    """Run LSM on the full series, return residuals (obs - pred) and time features."""
    fit = lsm.fit(df["t_hours"].to_numpy(), df["height_m"].to_numpy())
    pred = lsm.predict(df["t_hours"].to_numpy(), fit)
    residuals = df["height_m"].to_numpy() - pred
    return residuals, _time_features(df["datetime_ist"]), fit


def _split_indices(n: int, train: float = 0.8, val: float = 0.1) -> tuple[int, int]:
    cut1 = int(n * train)
    cut2 = int(n * (train + val))
    return cut1, cut2


def train_port(port_slug: str, device: torch.device) -> dict:
    csv_path = DATA_DIR / f"{port_slug}.csv"
    df = tp.load_series(csv_path)
    residuals, time_feat, fit = _build_residuals(df)

    print(f"[{port_slug}] residuals: mean={residuals.mean():+.4f} m  "
          f"std={residuals.std():.4f} m  abs_max={np.abs(residuals).max():.3f} m")

    n = residuals.size
    cut_train, cut_val = _split_indices(n)

    # Standardize residuals using the train segment so the network sees ~unit-variance input
    train_mean = float(residuals[:cut_train].mean())
    train_std = float(residuals[:cut_train].std() + 1e-8)
    res_norm = (residuals - train_mean) / train_std

    train_ds = ResidualWindowDataset(res_norm[:cut_train], time_feat[:cut_train])
    val_ds = ResidualWindowDataset(res_norm[cut_train - WINDOW: cut_val],
                                   time_feat[cut_train - WINDOW: cut_val])
    test_ds = ResidualWindowDataset(res_norm[cut_val - WINDOW:],
                                    time_feat[cut_val - WINDOW:])

    print(f"  windows: train={len(train_ds):,}  val={len(val_ds):,}  test={len(test_ds):,}")

    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0,
                          pin_memory=(device.type == "cuda"))
    val_dl = DataLoader(val_ds, batch_size=BATCH * 4, shuffle=False, num_workers=0,
                        pin_memory=(device.type == "cuda"))
    test_dl = DataLoader(test_ds, batch_size=BATCH * 4, shuffle=False, num_workers=0,
                         pin_memory=(device.type == "cuda"))

    model = ResidualLSTM().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    bad = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        n_seen = 0
        for xb, yb in train_dl:
            xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
            opt.zero_grad()
            yp = model(xb)
            loss = loss_fn(yp, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            train_loss += loss.item() * xb.size(0)
            n_seen += xb.size(0)
        train_loss /= max(1, n_seen)

        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
                val_loss += loss_fn(model(xb), yb).item() * xb.size(0)
                n_val += xb.size(0)
        val_loss /= max(1, n_val)

        marker = ""
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
            marker = "*"
        else:
            bad += 1
        print(f"  epoch {epoch:>2d}  train_mse={train_loss:.5f}  val_mse={val_loss:.5f}{marker}")
        if bad >= PATIENCE:
            print(f"  early stop after {epoch} epochs")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Test set evaluation: compare LSM-only vs LSM+LSTM
    model.eval()
    preds_norm = []
    with torch.no_grad():
        for xb, _ in test_dl:
            preds_norm.append(model(xb.to(device, non_blocking=True)).cpu().numpy())
    preds_norm = np.concatenate(preds_norm)
    pred_residuals = preds_norm * train_std + train_mean

    # Aligned slice of true residuals & observations for the test segment
    true_residuals = residuals[cut_val:]
    if pred_residuals.size != true_residuals.size:
        # In case of off-by-one alignment, trim
        m = min(pred_residuals.size, true_residuals.size)
        pred_residuals = pred_residuals[:m]
        true_residuals = true_residuals[:m]

    obs_test = df["height_m"].to_numpy()[cut_val: cut_val + true_residuals.size]
    lsm_test = lsm.predict(df["t_hours"].to_numpy()[cut_val: cut_val + true_residuals.size], fit)

    lsm_only_metrics = eval_mod.evaluate(lsm_test, obs_test)
    combined_metrics = eval_mod.evaluate(lsm_test + pred_residuals, obs_test)

    print(f"  TEST  LSM-only      RMSE={lsm_only_metrics.rmse:.3f} m  R={lsm_only_metrics.r:.4f}")
    print(f"  TEST  LSM + LSTM    RMSE={combined_metrics.rmse:.3f} m  R={combined_metrics.r:.4f}  "
          f"({100*(lsm_only_metrics.rmse - combined_metrics.rmse)/lsm_only_metrics.rmse:+.1f}% RMSE)")

    # Save model + scaler
    torch.save({
        "state_dict": model.state_dict(),
        "config": {
            "window": WINDOW, "n_features": N_FEATURES,
            "hidden": HIDDEN, "layers": LAYERS, "dropout": DROPOUT,
        },
        "scaler": {"mean": train_mean, "std": train_std},
    }, MODEL_DIR / f"{port_slug}.pt")
    print(f"  -> {MODEL_DIR / (port_slug + '.pt')}")

    # Save side-by-side comparison CSV
    comp = pd.DataFrame({
        "datetime_ist": df["datetime_ist"].to_numpy()[cut_val: cut_val + true_residuals.size],
        "observed_m": obs_test,
        "lsm_predicted_m": lsm_test,
        "lstm_residual_m": pred_residuals,
        "combined_predicted_m": lsm_test + pred_residuals,
        "true_residual_m": true_residuals,
    })
    comp.to_csv(OUTPUT_DIR / f"lstm_test_{port_slug}.csv", index=False)

    return {
        "port": port_slug,
        "lsm_only_test": lsm_only_metrics.as_dict(),
        "lsm_plus_lstm_test": combined_metrics.as_dict(),
        "scaler": {"mean": train_mean, "std": train_std},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=None,
                        help="port slug (default: train all ports in data/)")
    parser.add_argument("--cpu", action="store_true", help="force CPU")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"device: {device}")

    summaries = []
    csvs = ([DATA_DIR / f"{args.port}.csv"] if args.port
            else sorted(DATA_DIR.glob("*.csv")))
    for csv_path in csvs:
        slug = csv_path.stem
        print(f"\n=== {slug.upper()} ===")
        summaries.append(train_port(slug, device))

    out = OUTPUT_DIR / "lstm_summary.json"
    # Merge with any existing summary so single-port runs accumulate
    existing = []
    if out.exists():
        try:
            existing = json.loads(out.read_text())
        except Exception:
            existing = []
    by_port = {r["port"]: r for r in existing}
    for r in summaries:
        by_port[r["port"]] = r
    merged = sorted(by_port.values(), key=lambda r: r["port"])
    out.write_text(json.dumps(merged, indent=2))
    print(f"\nSummary -> {out}")


if __name__ == "__main__":
    main()

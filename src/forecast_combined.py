"""Combined forecasting: LSM (deterministic future tide) + LSTM (residual model
that captures storm surge / monsoon discharge).

Two modes:
1. Hindcast: predict over a historical range we have observations for. The LSTM
   gets fed real residuals from the past as input, so accuracy is high. Useful
   for evaluation and for back-filling gappy observations.

2. Forecast: predict the future. The LSTM is run autoregressively, feeding its
   own predicted residuals back in. Quality decays the further you go: the
   first ~24 hours are typically very good, after that the LSTM contribution
   smoothly tends towards zero (we don't have future weather), so the model
   converges to the LSM-only prediction.

Usage:
  uv run python src/forecast_combined.py --port haldia \
      --start 2024-12-01 --end 2024-12-08 --mode hindcast --out tmp.csv

  uv run python src/forecast_combined.py --port haldia \
      --start 2026-06-01 --hours 72 --mode forecast --out forecast.csv
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

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lsm
import train_predict as tp
from constituents import CONSTITUENTS, by_name
from residual_lstm import ResidualLSTM, WINDOW, _time_features

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
MODEL_DIR = OUTPUT_DIR / "lstm_models"


def _load_lsm_model(port_slug: str):
    """Reconstruct an LSM HarmonicFit from the saved CSV + summary."""
    summary = json.loads((OUTPUT_DIR / "summary.json").read_text())
    rec = next(r for r in summary if r["port"] == port_slug)
    epoch = datetime.fromisoformat(rec["epoch"])
    mean_level = float(rec["mean_level_m"])
    hc = pd.read_csv(OUTPUT_DIR / f"harmonic_constants_{port_slug}.csv")

    constituents = [by_name(n) for n in hc["name"]]
    fit = lsm.HarmonicFit(
        mean_level=mean_level,
        constituents=constituents,
        amplitudes=hc["amplitude_m"].to_numpy(),
        phases_deg=hc["phase_deg"].to_numpy(),
        cos_coef=hc["A_cos"].to_numpy(),
        sin_coef=hc["B_sin"].to_numpy(),
    )
    return fit, epoch


def _load_lstm_model(port_slug: str, device: torch.device):
    ck = torch.load(MODEL_DIR / f"{port_slug}.pt", map_location=device, weights_only=False)
    cfg = ck["config"]
    model = ResidualLSTM(
        n_features=cfg["n_features"],
        hidden=cfg["hidden"],
        layers=cfg["layers"],
        dropout=cfg["dropout"],
    ).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck["scaler"], cfg


def _build_lstm_input(residuals_norm: np.ndarray, time_feat: np.ndarray) -> np.ndarray:
    """Concatenate per-timestep [residual, hour_sin, hour_cos, day_sin, day_cos, month_sin, month_cos]."""
    return np.concatenate([residuals_norm[:, None], time_feat], axis=1).astype(np.float32)


def hindcast(port_slug: str, start: datetime, end: datetime,
             device: torch.device | None = None) -> pd.DataFrame:
    """Predict over a historical range using observed residuals as LSTM context."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fit, epoch = _load_lsm_model(port_slug)
    model, scaler, _ = _load_lstm_model(port_slug, device)

    # Need WINDOW hours of history before `start`. Pull from the observed CSV.
    df = tp.load_series(DATA_DIR / f"{port_slug}.csv")
    history_start = start - timedelta(hours=WINDOW)
    mask = (df["datetime_ist"] >= history_start) & (df["datetime_ist"] < end)
    seg = df.loc[mask].copy().reset_index(drop=True)
    if seg.empty:
        raise SystemExit("No observed data covers the requested range; use --mode forecast")

    obs = seg["height_m"].to_numpy()
    t = seg["t_hours"].to_numpy()
    lsm_pred = lsm.predict(t, fit)
    residuals = obs - lsm_pred
    residuals_norm = (residuals - scaler["mean"]) / scaler["std"]
    tf = _time_features(seg["datetime_ist"])
    feat = _build_lstm_input(residuals_norm, tf)

    n_targets = feat.shape[0] - WINDOW
    if n_targets <= 0:
        raise SystemExit("Range too short; need at least WINDOW+1 hours of context.")
    # Build batch: each row is one window
    X = np.lib.stride_tricks.sliding_window_view(feat, (WINDOW, feat.shape[1]))[:n_targets, 0].copy()
    X_t = torch.from_numpy(X).to(device)
    with torch.no_grad():
        preds_norm = model(X_t).cpu().numpy()
    pred_residuals = preds_norm * scaler["std"] + scaler["mean"]

    # Targets correspond to seg.iloc[WINDOW:]
    out = seg.iloc[WINDOW:].copy().reset_index(drop=True)
    out["lsm_predicted_m"] = lsm_pred[WINDOW:]
    out["lstm_residual_m"] = pred_residuals
    out["combined_predicted_m"] = out["lsm_predicted_m"] + out["lstm_residual_m"]
    return out[["datetime_ist", "height_m", "lsm_predicted_m",
                "lstm_residual_m", "combined_predicted_m"]]


def forecast(port_slug: str, start: datetime, end: datetime,
             device: torch.device | None = None) -> pd.DataFrame:
    """Predict the future autoregressively. LSTM residual decays as we go further out."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fit, epoch = _load_lsm_model(port_slug)
    model, scaler, _ = _load_lstm_model(port_slug, device)

    df = tp.load_series(DATA_DIR / f"{port_slug}.csv")
    last_obs = df["datetime_ist"].iloc[-1]
    if start <= last_obs:
        # Caller probably wants hindcast
        print(f"  note: start ({start}) <= last observation ({last_obs}); "
              "using observed history for the LSTM context")
    # Take the WINDOW hours immediately before `start` (or the last available hours
    # if start is in the future).
    ctx_end = min(start, last_obs + timedelta(hours=1))
    ctx_start = ctx_end - timedelta(hours=WINDOW)
    ctx_mask = (df["datetime_ist"] >= ctx_start) & (df["datetime_ist"] < ctx_end)
    ctx = df.loc[ctx_mask].copy().reset_index(drop=True)
    if len(ctx) < WINDOW:
        raise SystemExit(f"Need {WINDOW} hours of context before {start}; only {len(ctx)} available.")

    ctx_t = ctx["t_hours"].to_numpy()
    ctx_lsm = lsm.predict(ctx_t, fit)
    ctx_residuals = ctx["height_m"].to_numpy() - ctx_lsm
    ctx_resn = (ctx_residuals - scaler["mean"]) / scaler["std"]
    ctx_tf = _time_features(ctx["datetime_ist"])
    rolling = _build_lstm_input(ctx_resn, ctx_tf)  # shape (WINDOW, 7)

    # Hourly forecast loop
    timestamps = pd.date_range(start, end, freq="h", inclusive="left")
    n = len(timestamps)
    if n == 0:
        raise SystemExit("Empty forecast range.")
    t_hours = np.array([(ts.to_pydatetime() - epoch).total_seconds() / 3600.0
                        for ts in timestamps])
    lsm_pred = lsm.predict(t_hours, fit)
    tf = _time_features(pd.Series(timestamps))
    pred_residuals = np.zeros(n, dtype=float)

    rolling_t = torch.from_numpy(rolling).unsqueeze(0).to(device)  # (1, WINDOW, 7)
    with torch.no_grad():
        for i in range(n):
            yhat_norm = model(rolling_t).item()
            pred_residuals[i] = yhat_norm * scaler["std"] + scaler["mean"]
            # Append new step, drop oldest
            new_row = np.concatenate([[yhat_norm], tf[i]]).astype(np.float32)
            rolling_t = torch.cat([
                rolling_t[:, 1:, :],
                torch.from_numpy(new_row).view(1, 1, -1).to(device),
            ], dim=1)

    out = pd.DataFrame({
        "datetime_ist": timestamps,
        "lsm_predicted_m": lsm_pred,
        "lstm_residual_m": pred_residuals,
        "combined_predicted_m": lsm_pred + pred_residuals,
    })
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", required=True)
    p.add_argument("--start", required=True)
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--end")
    grp.add_argument("--hours", type=int)
    grp.add_argument("--days", type=int)
    p.add_argument("--mode", choices=["hindcast", "forecast"], default="forecast")
    p.add_argument("--out", type=Path)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    start = datetime.fromisoformat(args.start)
    if args.end:
        end = datetime.fromisoformat(args.end)
    elif args.hours:
        end = start + timedelta(hours=args.hours)
    else:
        end = start + timedelta(days=args.days)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"device: {device}  mode: {args.mode}  range: {start} -> {end}")

    if args.mode == "hindcast":
        df = hindcast(args.port, start, end, device)
    else:
        df = forecast(args.port, start, end, device)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"wrote {len(df):,} rows -> {args.out}")
    else:
        print(df.head(24).to_string(index=False))


if __name__ == "__main__":
    main()

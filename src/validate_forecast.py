"""Validate future forecasts by comparing against real observations.

This tool:
1. Generates a forecast for a historical range we have observations for
2. Compares the forecast against the real observations
3. Reports accuracy metrics (RMSE, R, bias)
4. Shows how accuracy degrades over the forecast horizon
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import evaluate as eval_mod
import forecast_combined as fc
import train_predict as tp

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"


def validate_forecast(port_slug: str, start: datetime, end: datetime,
                      device_str: str = "cuda") -> dict:
    """Generate a forecast for a historical range and compare to observations."""
    import torch
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    # Load observations
    df_obs = tp.load_series(DATA_DIR / f"{port_slug}.csv")
    obs_mask = (df_obs["datetime_ist"] >= start) & (df_obs["datetime_ist"] < end)
    obs_seg = df_obs.loc[obs_mask].copy()

    if obs_seg.empty:
        raise SystemExit(f"No observations in range {start} to {end}")

    print(f"[{port_slug}] validating forecast from {start} to {end}")
    print(f"  observations: {len(obs_seg):,} hours")

    # Generate forecast (autoregressive)
    forecast_df = fc.forecast(port_slug, start, end, device)

    # Align and compare
    merged = pd.merge(
        obs_seg[["datetime_ist", "height_m"]].reset_index(drop=True),
        forecast_df[["datetime_ist", "lsm_predicted_m", "lstm_residual_m",
                     "combined_predicted_m"]].reset_index(drop=True),
        on="datetime_ist",
        how="inner"
    )

    if merged.empty:
        raise SystemExit("No matching timestamps between forecast and observations")

    obs = merged["height_m"].to_numpy()
    lsm_pred = merged["lsm_predicted_m"].to_numpy()
    combined_pred = merged["combined_predicted_m"].to_numpy()

    # Overall metrics
    lsm_metrics = eval_mod.evaluate(lsm_pred, obs)
    combined_metrics = eval_mod.evaluate(combined_pred, obs)

    print(f"\n  Overall (all {len(merged):,} hours):")
    print(f"    LSM only      RMSE={lsm_metrics.rmse:.3f} m  R={lsm_metrics.r:.4f}  bias={lsm_metrics.bias:+.4f}")
    print(f"    LSM + LSTM    RMSE={combined_metrics.rmse:.3f} m  R={combined_metrics.r:.4f}  bias={combined_metrics.bias:+.4f}  "
          f"({100*(lsm_metrics.rmse - combined_metrics.rmse)/lsm_metrics.rmse:+.1f}%)")

    # Horizon-based breakdown (how accuracy degrades over time)
    print(f"\n  Accuracy by forecast horizon:")
    horizons = [6, 12, 24, 48, 72]
    for h in horizons:
        if h > len(merged):
            continue
        mask = np.arange(len(merged)) < h
        h_obs = obs[mask]
        h_lsm = lsm_pred[mask]
        h_comb = combined_pred[mask]
        h_lsm_m = eval_mod.evaluate(h_lsm, h_obs)
        h_comb_m = eval_mod.evaluate(h_comb, h_obs)
        print(f"    {h:>2d}h  LSM={h_lsm_m.rmse:.3f}m  LSM+LSTM={h_comb_m.rmse:.3f}m  "
              f"({100*(h_lsm_m.rmse - h_comb_m.rmse)/h_lsm_m.rmse:+.1f}%)")

    # Save detailed comparison
    out_csv = OUTPUT_DIR / f"validation_forecast_{port_slug}_{start.date()}_to_{end.date()}.csv"
    merged["lsm_error_m"] = lsm_pred - obs
    merged["combined_error_m"] = combined_pred - obs
    merged.to_csv(out_csv, index=False)
    print(f"\n  -> {out_csv}")

    return {
        "port": port_slug,
        "range": f"{start} to {end}",
        "n_hours": len(merged),
        "lsm_metrics": lsm_metrics.as_dict(),
        "combined_metrics": combined_metrics.as_dict(),
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="Validate forecast accuracy by comparing against real observations."
    )
    p.add_argument("--port", required=True, help="Port slug (haldia or diamond_harbour)")
    p.add_argument("--start", required=True, help="Start date (ISO format, e.g. 2024-06-01)")
    p.add_argument("--days", type=int, default=7, help="Number of days to validate (default 7)")
    p.add_argument("--cpu", action="store_true", help="Force CPU (default: use GPU if available)")
    args = p.parse_args()

    start = datetime.fromisoformat(args.start)
    end = start + timedelta(days=args.days)
    device = "cpu" if args.cpu else "cuda"

    result = validate_forecast(args.port, start, end, device)
    print(f"\nValidation complete.")


if __name__ == "__main__":
    main()

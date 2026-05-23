"""Phase 3+: Train LSM on 90% of each port's series, validate on 10%,
report metrics, persist harmonic constants, and emit forecast helpers.
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running this file directly: `python src/train_predict.py`
sys.path.insert(0, str(Path(__file__).resolve().parent))

from constituents import CONSTITUENTS
import datums as datums_mod
import evaluate as eval_mod
import lsm

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Common epoch so all ports/predictions share one time axis
EPOCH = datetime(2000, 1, 1, 0, 0, 0)


def load_series(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["datetime_ist"])
    df = df.sort_values("datetime_ist").drop_duplicates("datetime_ist")
    df["t_hours"] = (df["datetime_ist"] - EPOCH).dt.total_seconds() / 3600.0
    return df.reset_index(drop=True)


def chronological_split(df: pd.DataFrame, train_frac: float = 0.9) -> tuple[pd.DataFrame, pd.DataFrame]:
    cut = int(len(df) * train_frac)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def fit_and_evaluate(df: pd.DataFrame, port_slug: str) -> dict:
    train, val = chronological_split(df, train_frac=0.9)
    print(f"[{port_slug}] train: {len(train):,} rows "
          f"({train['datetime_ist'].iloc[0]} -> {train['datetime_ist'].iloc[-1]})")
    print(f"[{port_slug}] valid: {len(val):,} rows "
          f"({val['datetime_ist'].iloc[0]} -> {val['datetime_ist'].iloc[-1]})")

    fit = lsm.fit(train["t_hours"].to_numpy(), train["height_m"].to_numpy())

    # Save harmonic constants
    rows = []
    for c, H, g, A, B in zip(fit.constituents, fit.amplitudes, fit.phases_deg,
                             fit.cos_coef, fit.sin_coef):
        rows.append({
            "name": c.name, "kind": c.kind,
            "period_hours": c.period_hours,
            "speed_deg_per_hour": c.speed_deg_per_hour,
            "amplitude_m": float(H),
            "phase_deg": float(g),
            "A_cos": float(A),
            "B_sin": float(B),
        })
    hc_path = OUTPUT_DIR / f"harmonic_constants_{port_slug}.csv"
    pd.DataFrame(rows).to_csv(hc_path, index=False)
    print(f"  -> {hc_path}")

    # Predict on train and validation
    train_pred = lsm.predict(train["t_hours"].to_numpy(), fit)
    val_pred = lsm.predict(val["t_hours"].to_numpy(), fit)

    train_metrics = eval_mod.evaluate(train_pred, train["height_m"].to_numpy())
    val_metrics = eval_mod.evaluate(val_pred, val["height_m"].to_numpy())

    print(f"  TRAIN  n={train_metrics.n:>7,}  RMSE={train_metrics.rmse:.3f} m  "
          f"R={train_metrics.r:.4f}  bias={train_metrics.bias:+.4f}  SR={train_metrics.sr:.4f}")
    print(f"  VALID  n={val_metrics.n:>7,}  RMSE={val_metrics.rmse:.3f} m  "
          f"R={val_metrics.r:.4f}  bias={val_metrics.bias:+.4f}  SR={val_metrics.sr:.4f}")

    # Save validation predictions for inspection
    val_out = val.copy()
    val_out["predicted_m"] = val_pred
    val_out["residual_m"] = val_out["predicted_m"] - val_out["height_m"]
    val_out[["datetime_ist", "height_m", "predicted_m", "residual_m"]].to_csv(
        OUTPUT_DIR / f"validation_{port_slug}.csv", index=False
    )

    # 19-year forward prediction for tidal datums
    last_obs = df["datetime_ist"].iloc[-1]
    future_start = last_obs + timedelta(hours=1)
    future_end = future_start + timedelta(days=365 * 19 + 5)
    future_t = np.arange(
        (future_start - EPOCH).total_seconds() / 3600.0,
        (future_end - EPOCH).total_seconds() / 3600.0,
        1.0,
    )
    future_h = lsm.predict(future_t, fit)
    d = datums_mod.compute(future_t, future_h)
    print(f"  19yr DATUMS  HAT={d.HAT:+.2f}  MHWS={d.MHWS:+.2f}  MHW={d.MHW:+.2f}  "
          f"MSL={d.MSL:+.2f}  MLW={d.MLW:+.2f}  MLWS={d.MLWS:+.2f}  LAT={d.LAT:+.2f}  "
          f"RA={d.RA:.2f}")

    # Persist a small future-forecast sample (next 30 days, hourly)
    sample_t = future_t[: 24 * 30]
    sample_h = future_h[: 24 * 30]
    sample_dt = [EPOCH + timedelta(hours=float(t)) for t in sample_t]
    pd.DataFrame({
        "datetime_ist": sample_dt,
        "predicted_height_m": sample_h,
    }).to_csv(OUTPUT_DIR / f"forecast_30days_{port_slug}.csv", index=False)

    return {
        "port": port_slug,
        "epoch": EPOCH.isoformat(),
        "n_train": len(train),
        "n_valid": len(val),
        "train_range": [train["datetime_ist"].iloc[0].isoformat(),
                        train["datetime_ist"].iloc[-1].isoformat()],
        "valid_range": [val["datetime_ist"].iloc[0].isoformat(),
                        val["datetime_ist"].iloc[-1].isoformat()],
        "train_metrics": train_metrics.as_dict(),
        "valid_metrics": val_metrics.as_dict(),
        "datums_19yr": d.as_dict(),
        "mean_level_m": fit.mean_level,
    }


def main() -> None:
    summaries = []
    for csv_path in sorted(DATA_DIR.glob("*.csv")):
        port_slug = csv_path.stem
        print(f"\n=== {port_slug.upper()} ===")
        df = load_series(csv_path)
        print(f"  {len(df):,} hourly observations "
              f"({df['datetime_ist'].iloc[0]} -> {df['datetime_ist'].iloc[-1]})")
        summaries.append(fit_and_evaluate(df, port_slug))

    summary_path = OUTPUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2, default=str))
    print(f"\nSummary -> {summary_path}")


if __name__ == "__main__":
    main()

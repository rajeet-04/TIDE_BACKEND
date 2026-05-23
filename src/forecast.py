"""CLI for generating future tidal predictions from saved harmonic constants.

Usage:
  python src/forecast.py --port haldia --start 2026-01-01 --end 2026-01-08
  python src/forecast.py --port diamond_harbour --start 2026-06-01 --hours 168 \
      --out output/my_forecast.csv

Reads:
  output/harmonic_constants_<port>.csv  (created by train_predict.py)
  output/summary.json                   (for mean_level_m and epoch)
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output"


def _load_model(port_slug: str) -> tuple[float, datetime, pd.DataFrame]:
    summary = json.loads((OUTPUT_DIR / "summary.json").read_text())
    rec = next((r for r in summary if r["port"] == port_slug), None)
    if rec is None:
        raise SystemExit(f"No model for port '{port_slug}'. Run train_predict.py first.")
    epoch = datetime.fromisoformat(rec["epoch"])
    mean_level = float(rec["mean_level_m"])
    hc = pd.read_csv(OUTPUT_DIR / f"harmonic_constants_{port_slug}.csv")
    return mean_level, epoch, hc


def predict(port_slug: str, start: datetime, end: datetime, freq_minutes: int = 60) -> pd.DataFrame:
    mean_level, epoch, hc = _load_model(port_slug)

    step_h = freq_minutes / 60.0
    start_h = (start - epoch).total_seconds() / 3600.0
    end_h = (end - epoch).total_seconds() / 3600.0
    t = np.arange(start_h, end_h, step_h)

    h = np.full(t.size, mean_level, dtype=float)
    for _, row in hc.iterrows():
        omega = np.deg2rad(row["speed_deg_per_hour"])
        h += row["A_cos"] * np.cos(omega * t) + row["B_sin"] * np.sin(omega * t)

    timestamps = [epoch + timedelta(hours=float(x)) for x in t]
    return pd.DataFrame({"datetime_ist": timestamps, "predicted_height_m": h})


def main() -> None:
    p = argparse.ArgumentParser(description="Forecast tidal heights with the trained LSM model.")
    p.add_argument("--port", required=True,
                   help="port slug, e.g. 'haldia' or 'diamond_harbour'")
    p.add_argument("--start", required=True, help="start datetime, ISO format e.g. 2026-01-01T00:00")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--end", help="end datetime, ISO format")
    grp.add_argument("--hours", type=int, help="number of hours forward from start")
    grp.add_argument("--days", type=int, help="number of days forward from start")
    p.add_argument("--freq-minutes", type=int, default=60, help="sample interval (default 60)")
    p.add_argument("--out", type=Path, default=None, help="output CSV (default stdout)")
    args = p.parse_args()

    start = datetime.fromisoformat(args.start)
    if args.end:
        end = datetime.fromisoformat(args.end)
    elif args.hours:
        end = start + timedelta(hours=args.hours)
    else:
        end = start + timedelta(days=args.days)

    df = predict(args.port, start, end, args.freq_minutes)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"wrote {len(df):,} rows -> {args.out}")
    else:
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()

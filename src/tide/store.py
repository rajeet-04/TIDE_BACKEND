"""Canonical observation tables (spec 3.2): time_utc, height_m, source, qc_flag."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from tide.ports import DATA_DIR, IST, port, trainable

QC_DIR = DATA_DIR / "qc"
COLUMNS = ["time_utc", "height_m", "source", "qc_flag"]

def read_gauge_csv(path: Path) -> pd.DataFrame:
    """Survey of India hourly heights (port, datetime_ist, height_m) as an unflagged canonical table.

    Rows are sorted by time; a repeated time keeps its last reading."""
    raw = pd.read_csv(path, parse_dates=["datetime_ist"])
    times = pd.DatetimeIndex(raw["datetime_ist"]).tz_localize(IST).tz_convert("UTC")
    table = pd.DataFrame({"time_utc": times, "height_m": raw["height_m"].astype(float).to_numpy(),
                          "source": "soi_gauge", "qc_flag": ""})
    table = table.drop_duplicates("time_utc", keep="last")
    return table.sort_values("time_utc", kind="stable").reset_index(drop=True)

def flags_path(port_slug: str) -> Path:
    return QC_DIR / f"{port_slug}_flags.csv"

def load_gauge(port_slug: str, *, passed_only: bool = False) -> pd.DataFrame:
    """Gauge readings for a port with the committed QC flags applied."""
    p = port(port_slug)
    if not trainable(p.slug, "soi_gauge"):
        raise ValueError(f"the {p.slug} gauge datum is unknown; its readings cannot be used")
    table = read_gauge_csv(DATA_DIR / f"{p.slug}.csv")
    path = flags_path(p.slug)
    if path.exists():
        flags = pd.read_csv(path)
        flagged = pd.Series(flags["qc_flag"].to_numpy(), index=pd.to_datetime(flags["time_utc"], utc=True))
        table["qc_flag"] = table["time_utc"].map(flagged).fillna("")
    return table[table["qc_flag"] == ""].reset_index(drop=True) if passed_only else table

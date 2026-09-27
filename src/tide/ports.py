"""Ports, time conventions and the datum registry.

All code works in timezone-aware UTC. IST (UTC+05:30, no daylight saving) appears only
in outputs, and naive input times are read as IST, the local convention at these ports.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
IST = timezone(timedelta(hours=5, minutes=30), "IST")
_SEASON_BY_MONTH = np.array(["dry"] * 3 + ["pre_monsoon"] * 2 + ["monsoon"] * 4 + ["post_monsoon"] * 3, dtype=object)

@dataclass(frozen=True)
class Port:
    slug: str
    name: str
    lat: float
    lon: float

# Coordinates as printed on the Survey of India 2026 tide tables.
PORTS = {
    "haldia": Port("haldia", "HALDIA", 22 + 2 / 60, 88 + 6 / 60),
    "diamond_harbour": Port("diamond_harbour", "DIAMOND HARBOUR", 22 + 12 / 60, 88 + 10 / 60),
}

@dataclass(frozen=True)
class Datum:
    name: str
    offset_to_chart_datum_m: float | None  # None: unknown
    evidence: str

_SOI = ("Survey of India gauge heights and tables are referred to chart datum; "
        "confirmation of the gauge datums has been requested from Survey of India.")
DATUMS = {
    ("haldia", "soi_gauge"): Datum("chart datum", 0.0, _SOI),
    ("diamond_harbour", "soi_gauge"): Datum("chart datum", 0.0, _SOI),
    ("haldia", "soi_tables"): Datum("chart datum", 0.0, _SOI),
    ("diamond_harbour", "soi_tables"): Datum("chart datum", 0.0, _SOI),
    ("garden_reach", "incois"): Datum("unknown", None, "INCOIS has not published this gauge's datum."),
}

def port(slug: str) -> Port:
    if slug not in PORTS:
        raise ValueError(f"unknown port {slug!r}; choose from {sorted(PORTS)}")
    return PORTS[slug]

def datum_label(port_slug: str, source: str = "soi_gauge") -> str:
    d = DATUMS[(port_slug, source)]
    return "unknown datum" if d.offset_to_chart_datum_m is None else f"{d.name} (Survey of India)"

def trainable(port_slug: str, source: str) -> bool:
    """Only sources with a known datum may be used for training."""
    return DATUMS[(port_slug, source)].offset_to_chart_datum_m is not None

def to_utc(value) -> pd.Timestamp:
    """A timestamp in UTC; naive values are read as IST."""
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize(IST)
    return stamp.tz_convert("UTC")

def ist_year_start(year: int) -> pd.Timestamp:
    """00:00 IST on 1 January, in UTC."""
    return pd.Timestamp(year=year, month=1, day=1, tz=IST).tz_convert("UTC")

def ist_years(times) -> np.ndarray:
    return np.asarray(pd.DatetimeIndex(times).tz_convert(IST).year)

def season_of(times) -> np.ndarray:
    return _SEASON_BY_MONTH[np.asarray(pd.DatetimeIndex(times).tz_convert(IST).month) - 1]

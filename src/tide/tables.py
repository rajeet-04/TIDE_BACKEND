"""Official Survey of India prediction tables: benchmark and monitoring only, never training data."""
from __future__ import annotations

import json
import urllib.request

import pandas as pd

from tide.ports import DATA_DIR, port

API = "http://117.250.29.124:5008/api/tide/predicted-tide-data"
API_PORT_IDS = {"haldia": "ef43eb28-8fe3-400d-a837-6d5a69e5ab59",
                "diamond_harbour": "6fdb84c4-1459-4f5a-960e-a5c50acfdeba"}
# The API labels times 'Z', but they are IST minus 11 h (measured on 478 events on 2026-09-27),
# so the true UTC time is the label plus 5 h 30 min.
API_LABEL_TO_UTC = pd.Timedelta(hours=5, minutes=30)
TABLES_DIR = DATA_DIR / "tables"
COLUMNS = ["state", "time_utc", "height_m", "source"]

def parse_api_rows(rows: list[dict], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """API event rows as table events in [start, end), with the clock label corrected."""
    raw = pd.DataFrame(rows).drop_duplicates("date_time").reset_index(drop=True)
    events = pd.DataFrame({"state": raw["tide_type"].str.title(),
                           "time_utc": pd.to_datetime(raw["date_time"], utc=True) + API_LABEL_TO_UTC,
                           "height_m": raw["tide_height_m"].astype(float), "source": "soi_api"})
    inside = (events["time_utc"] >= start) & (events["time_utc"] < end)
    return events[inside].sort_values("time_utc").reset_index(drop=True)

def fetch_api(port_slug: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Table events in [start, end) from the prediction API, in 30-day requests (its limit)."""
    rows: list[dict] = []
    for chunk in pd.date_range(start - pd.Timedelta(days=1), end + pd.Timedelta(days=1), freq="30D"):
        body = json.dumps({"port_id": API_PORT_IDS[port(port_slug).slug], "start_date": chunk.date().isoformat(),
                           "days": "30"}).encode()
        request = urllib.request.Request(API, body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=60) as response:
            rows += json.load(response)["data"]
    return parse_api_rows(rows, start, end)

def tables_path(port_slug: str):
    return TABLES_DIR / f"{port_slug}.csv"

def load_tables(port_slug: str) -> pd.DataFrame:
    path = tables_path(port_slug)
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    stored = pd.read_csv(path)
    stored["time_utc"] = pd.to_datetime(stored["time_utc"], utc=True)
    return stored

def save_tables(port_slug: str, events: pd.DataFrame) -> pd.DataFrame:
    """Merge events into data/tables/<port>.csv; a repeated (time, state) keeps the newer row."""
    frames = [f for f in (load_tables(port_slug), events[COLUMNS]) if len(f)]
    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)
    merged = merged.drop_duplicates(["time_utc", "state"], keep="last").sort_values("time_utc").reset_index(drop=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(tables_path(port_slug), index=False)
    return merged

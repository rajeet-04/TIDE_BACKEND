import io
import json

import numpy as np
import pandas as pd

from tide import backtest, tables
from tide.backtest import joint_pct
from tide.events import find_events
from tide.ports import ist_year_start

class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

def test_api_labels_are_shifted_to_true_utc(monkeypatch):
    payload = {"data": [{"date_time": "2024-01-01T01:30:00.000Z", "tide_type": "HIGH", "tide_height_m": 5.1},
                        {"date_time": "2024-01-01T07:45:00.000Z", "tide_type": "LOW", "tide_height_m": 1.2}]}
    monkeypatch.setattr(tables.urllib.request, "urlopen",
                        lambda request, timeout: FakeResponse(json.dumps(payload).encode()))
    events = tables.fetch_api("haldia", pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-01-02", tz="UTC"))
    assert events["state"].tolist() == ["High", "Low"]
    assert events["time_utc"].tolist() == [pd.Timestamp("2024-01-01 07:00", tz="UTC"),
                                           pd.Timestamp("2024-01-01 13:15", tz="UTC")]

def test_saved_tables_merge_without_duplicates(tmp_path, monkeypatch):
    monkeypatch.setattr(tables, "TABLES_DIR", tmp_path)
    first = pd.DataFrame({"state": ["High"], "time_utc": [pd.Timestamp("2024-01-01 07:00", tz="UTC")],
                          "height_m": [5.1], "source": ["soi_api"]})
    tables.save_tables("haldia", first)
    merged = tables.save_tables("haldia", first.assign(height_m=5.2))
    assert len(merged) == 1 and merged["height_m"].iloc[0] == 5.2
    assert tables.load_tables("haldia")["time_utc"].iloc[0] == pd.Timestamp("2024-01-01 07:00", tz="UTC")

def test_fixed_predictions_are_scored_on_shared_gauge_years(monkeypatch):
    start = ist_year_start(2024)
    times = pd.date_range(start, ist_year_start(2025), freq="1h", inclusive="left")
    heights = 3.0 + 2.0 * np.cos(2 * np.pi * np.asarray((times - start) / pd.Timedelta(hours=1)) / 12.4206)
    gauge = pd.DataFrame({"time_utc": times, "height_m": heights, "source": "soi_gauge", "qc_flag": ""})
    monkeypatch.setattr(backtest, "load_gauge", lambda port_slug, passed_only=False: gauge)
    official = find_events(times, heights, step_minutes=60).assign(source="soi_api")
    rows = backtest.score_fixed("haldia", official)
    assert set(rows["test_year"]) == {2024} and joint_pct(rows) == 100.0

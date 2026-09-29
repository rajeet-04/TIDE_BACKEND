import inspect

import numpy as np
import pandas as pd

from tide import features
from tide.events import find_events
from tide.features import EVENT_FEATURES, LEVEL_FEATURES, astronomical, event_features, ist_hour_grid, level_features

def at(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz="UTC")

def test_moon_and_sun_features_match_known_dates():
    frame = astronomical(pd.DatetimeIndex([at("2024-01-25 17:54"), at("2024-01-11 11:57"), at("2024-06-20 20:51")]))
    assert list(frame.columns) == features.ASTRONOMICAL
    assert frame["moon_phase_cos"].iloc[0] < -0.95                 # full moon
    assert frame["moon_phase_cos"].iloc[1] > 0.95                  # new moon
    assert (frame["spring_neap_cos"].iloc[:2] > 0.9).all()         # both bring spring tides
    assert abs(np.degrees(frame["sun_declination"].iloc[2]) - 23.44) < 1.0   # June solstice

def test_hour_grid_is_whole_ist_hours_with_margins():
    grid = ist_hour_grid(at("2024-03-01 00:00"), at("2024-03-02 00:00"))
    assert (grid.minute == 30).all() and str(grid.tz) == "UTC"
    assert grid[0] <= at("2024-03-01 00:00") - pd.Timedelta(hours=40)
    assert grid[-1] >= at("2024-03-02 00:00") + pd.Timedelta(hours=40)

def test_level_features_locate_the_tide():
    grid = pd.date_range("2024-03-01 00:30", periods=120, freq="1h", tz="UTC")
    hours = np.arange(len(grid), dtype=float)
    frame = level_features(grid, 3.0 + 2.0 * np.cos(2 * np.pi * (hours - 24.0) / 12.42))   # high water at hour 24
    assert list(frame.columns) == LEVEL_FEATURES and frame.notna().all().all()
    row = frame.iloc[25]                                                                   # an hour after it
    assert abs(row["since_high"] - 1.0) < 0.05 and abs(row["until_low"] - 5.21) < 0.05
    assert abs(row["tide_range"] - 4.0) < 0.02 and row["slope"] < 0
    assert row["stage_cos"] < -0.8 and row["stage_sin"] < 0                                # early in the fall

def test_event_features_describe_each_event_and_its_neighbours():
    grid = pd.date_range("2024-03-01 00:30", periods=72, freq="1h", tz="UTC")
    hours = np.arange(len(grid), dtype=float)
    events = find_events(grid, 3.0 + 2.0 * np.cos(2 * np.pi * hours / 12.42), step_minutes=60)
    frame = event_features(events.sample(frac=1.0, random_state=0))                        # order does not matter
    assert list(frame.columns) == EVENT_FEATURES and len(frame) == len(events)
    assert abs(frame["prev_interval"].iloc[3] - 6.21) < 0.05 and abs(frame["prev_range"].iloc[3] - 4.0) < 0.05
    assert event_features(events.iloc[:0]).empty

def test_features_use_only_times_and_model_a():
    allowed = {"times", "grid", "levels", "events", "start", "end", "margin"}
    for function in (astronomical, ist_hour_grid, level_features, event_features):
        assert set(inspect.signature(function).parameters) <= allowed, function.__name__
    source = inspect.getsource(features)
    assert "tide.store" not in source and "load_gauge" not in source and "read_gauge_csv" not in source

def test_no_feature_follows_a_cycle_longer_than_a_year():
    """The lunar node (18.6 y) and perigee (8.85 y) cycles let learners tell years apart and replay
    one year's weather-driven level anomaly onto a later year in the same phase (plan 1b, Task 11).
    Model A's nodal corrections already carry their tidal effect."""
    assert not [c for c in features.ASTRONOMICAL if c.startswith(("node", "perigee"))]
    # the Moon's declination at the same point of its orbit must not drift with the node
    frame = astronomical(pd.date_range("2000-01-01", "2020-01-01", freq="1D", tz="UTC"))
    peaks = frame["moon_declination"].rolling(28).max().iloc[28:]
    assert peaks.max() - peaks.min() < np.radians(1.0)

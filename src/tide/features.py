"""Known-in-advance features for model B (spec 4.3, 5.6).

Every function here takes only times and model A's predicted curve or events, never gauge
readings, so a correction can be computed for any future time. A unit test enforces this.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from utide.astronomy import ut_astron

from tide.events import find_events
from tide.harmonic import datenum
from tide.ports import IST

OBLIQUITY = np.radians(23.44)
MOON_ECCENTRICITY = 0.0549
EARTH_ECCENTRICITY = 0.0167
CLIP_HOURS = 30.0
MARGIN = pd.Timedelta(hours=40)
ASTRONOMICAL = ["moon_phase_sin", "moon_phase_cos", "spring_neap_sin", "spring_neap_cos", "moon_declination",
                "moon_distance", "sun_declination", "sun_distance", "day_sin", "day_cos", "hour_sin", "hour_cos"]
SHAPE = ["level", "slope", "curvature", "since_high", "until_high", "since_low", "until_low", "tide_range",
         "stage_sin", "stage_cos"]
CONTEXT = ["is_high", "height", "prev_height", "next_height", "prev_range", "next_range", "prev_interval",
           "next_interval"]
LEVEL_FEATURES = SHAPE + ASTRONOMICAL
EVENT_FEATURES = CONTEXT + ASTRONOMICAL

def astronomical(times) -> pd.DataFrame:
    """Moon and sun state from UTide's mean longitudes: lunar phase and its spring/neap
    harmonic, approximate declinations and distances, and the day of year and hour of day in IST.
    Nothing on a cycle longer than a year: those let learners tell years apart (see the tests)."""
    times = pd.DatetimeIndex(times)
    astro, _ = ut_astron(datenum(times))
    s, h, perigee, perihelion = astro[1], astro[2], astro[3], astro[5]  # cycles
    turn = 2 * np.pi
    phase = turn * (s - h)
    ist = times.tz_convert(IST)
    hour = np.asarray(ist.hour + ist.minute / 60.0, dtype=float)
    day = (np.asarray(ist.dayofyear, dtype=float) - 1 + hour / 24.0) / 365.25
    return pd.DataFrame({
        "moon_phase_sin": np.sin(phase), "moon_phase_cos": np.cos(phase),
        "spring_neap_sin": np.sin(2 * phase), "spring_neap_cos": np.cos(2 * phase),
        "moon_declination": np.arcsin(np.sin(OBLIQUITY) * np.sin(turn * s)),   # on the ecliptic: no 18.6-year drift
        "moon_distance": 1 - MOON_ECCENTRICITY * np.cos(turn * (s - perigee)),
        "sun_declination": np.arcsin(np.sin(OBLIQUITY) * np.sin(turn * h)),
        "sun_distance": 1 - EARTH_ECCENTRICITY * np.cos(turn * (h - perihelion)),
        "day_sin": np.sin(turn * day), "day_cos": np.cos(turn * day),
        "hour_sin": np.sin(turn * hour / 24.0), "hour_cos": np.cos(turn * hour / 24.0),
    })

def ist_hour_grid(start, end, margin: pd.Timedelta = MARGIN) -> pd.DatetimeIndex:
    """Whole IST hours from start - margin to end + margin, in UTC: the hours the gauge reads,
    and the knots model B's level correction is evaluated on."""
    first = (pd.Timestamp(start).tz_convert(IST) - margin).floor("h")
    last = (pd.Timestamp(end).tz_convert(IST) + margin).ceil("h")
    return pd.date_range(first, last, freq="1h").tz_convert("UTC")

def level_features(grid, levels) -> pd.DataFrame:
    """LEVEL_FEATURES at each hour of an hourly grid, from model A's levels on that grid:
    level, slope and curvature; hours since and until A's high and low waters (clipped at 30);
    the range of the current tide; the stage within it; and the astronomical state."""
    grid = pd.DatetimeIndex(grid)
    a = np.asarray(levels, dtype=float)
    events = find_events(grid, a, step_minutes=60)
    if len(events) < 2:
        raise ValueError("level features need at least two tides on the grid")
    hours = datenum(grid) * 24.0
    event_hours = datenum(events["time_utc"]) * 24.0
    shape = {"level": a, "slope": np.gradient(a), "curvature": np.gradient(np.gradient(a))}
    for state in ("High", "Low"):
        mine = event_hours[(events["state"] == state).to_numpy()]
        shape[f"since_{state.lower()}"], shape[f"until_{state.lower()}"] = _since_until(hours, mine)
    k = np.searchsorted(event_hours, hours, side="right")
    before, after = np.clip(k - 1, 0, len(events) - 1), np.clip(k, 0, len(events) - 1)
    heights = events["height_m"].to_numpy()
    shape["tide_range"] = np.abs(heights[after] - heights[before])
    span = event_hours[after] - event_hours[before]
    share = np.divide(hours - event_hours[before], span, out=np.full(len(hours), 0.5), where=span > 0)
    rising = events["state"].to_numpy()[before] == "Low"
    stage = np.pi * np.clip(share, 0.0, 1.0) + np.where(rising, 0.0, np.pi)
    shape["stage_sin"], shape["stage_cos"] = np.sin(stage), np.cos(stage)
    return pd.concat([pd.DataFrame(shape), astronomical(grid)], axis=1)

def _since_until(hours: np.ndarray, event_hours: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    k = np.searchsorted(event_hours, hours)
    last = len(event_hours) - 1
    since = np.where(k > 0, hours - event_hours[np.clip(k - 1, 0, last)], CLIP_HOURS)
    until = np.where(k <= last, event_hours[np.clip(k, 0, last)] - hours, CLIP_HOURS)
    return np.clip(since, 0.0, CLIP_HOURS), np.clip(until, 0.0, CLIP_HOURS)

def event_features(events) -> pd.DataFrame:
    """EVENT_FEATURES for each event in time order: its state and height, its neighbours'
    heights and the intervals to them (the legacy calibrator's context), and the astronomical
    state. The first and last events stand in for their missing neighbour, 6 hours away."""
    ev = pd.DataFrame(events).sort_values("time_utc", kind="stable").reset_index(drop=True)
    if ev.empty:
        return pd.DataFrame(columns=EVENT_FEATURES, dtype=float)
    h = ev["height_m"].to_numpy(dtype=float)
    gaps = np.diff(datenum(ev["time_utc"]) * 24.0)
    prev_h, next_h = np.r_[h[:1], h[:-1]], np.r_[h[1:], h[-1:]]
    context = pd.DataFrame({
        "is_high": (ev["state"] == "High").to_numpy(dtype=float), "height": h,
        "prev_height": prev_h, "next_height": next_h,
        "prev_range": np.abs(h - prev_h), "next_range": np.abs(h - next_h),
        "prev_interval": np.clip(np.r_[6.0, gaps], 0.0, 24.0), "next_interval": np.clip(np.r_[gaps, 6.0], 0.0, 24.0),
    })
    return pd.concat([context, astronomical(ev["time_utc"])], axis=1)

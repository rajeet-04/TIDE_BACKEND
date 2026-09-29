"""High and low waters, event matching and accuracy scores (spec 4.6 and 5.3)."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

PROMINENCE_M = 0.5
MIN_SEPARATION_H = 4.0
STAND_TOLERANCE_M = 0.01
MAX_MATCH_MINUTES = 180
TIME_GATE_MINUTES = 30
HEIGHT_GATE_M = 0.30
PAD = pd.Timedelta(hours=12)
EVENT_COLUMNS = ["state", "time_utc", "height_m"]
MATCH_COLUMNS = ["state", "observed_time_utc", "observed_height_m", "predicted_index", "predicted_time_utc",
                 "predicted_height_m", "time_error_minutes", "height_error_m"]

def find_events(times, heights, step_minutes: int) -> pd.DataFrame:
    """Turning points of a regularly sampled series, split into blocks at gaps.

    Hourly observations use quadratic refinement. On a one-minute model curve a high water
    is the maximum sample, and a low water is the centre of the interval within
    STAND_TOLERANCE_M of the minimum, which fixes the time of long low-water stands."""
    times = pd.DatetimeIndex(times)
    heights = np.asarray(heights, dtype=float)
    if len(times) != len(heights):
        raise ValueError("times and heights differ in length")
    step = pd.Timedelta(minutes=step_minutes)
    breaks = np.flatnonzero(np.diff(times.values) != np.timedelta64(step_minutes, "m")) + 1
    distance = max(1, round(MIN_SEPARATION_H * 60 / step_minutes))
    rows = []
    for block in np.split(np.arange(len(times)), breaks):
        if len(block) < 5:
            continue
        values = heights[block]
        for state, sign in (("High", 1.0), ("Low", -1.0)):
            peaks, _ = find_peaks(sign * values, distance=distance, prominence=PROMINENCE_M)
            for p in peaks:
                if step_minutes == 1:
                    position = _stand_centre(values, p) if state == "Low" else float(p)
                    height = float(values[p])
                else:
                    offset, height = _vertex(values[p - 1], values[p], values[p + 1])
                    position = p + offset
                rows.append((state, times[block[0]] + position * step, height))
    if not rows:
        return pd.DataFrame({"state": pd.Series(dtype=object), "time_utc": pd.Series(dtype="datetime64[ns, UTC]"),
                             "height_m": pd.Series(dtype=float)})
    events = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    return events.sort_values("time_utc", kind="stable").reset_index(drop=True)

def _vertex(previous: float, current: float, following: float) -> tuple[float, float]:
    """Sub-sample offset and height of the parabola through three samples."""
    a = (previous - 2 * current + following) / 2
    b = (following - previous) / 2
    offset = float(np.clip(-b / (2 * a), -1, 1)) if a else 0.0
    return offset, float(current + b * offset + a * offset**2)

def _stand_centre(values: np.ndarray, i: int) -> float:
    lo = hi = i
    while lo > 0 and values[lo - 1] - values[i] <= STAND_TOLERANCE_M:
        lo -= 1
    while hi < len(values) - 1 and values[hi + 1] - values[i] <= STAND_TOLERANCE_M:
        hi += 1
    return (lo + hi) / 2

def model_events(levels: Callable[[pd.DatetimeIndex], np.ndarray], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Events of a model's one-minute curve in [start, end), padded so edge tides are found.
    The curve is sampled on whole minutes, so overlapping windows see the same samples."""
    times = pd.date_range((start - PAD).floor("min"), (end + PAD).ceil("min"), freq="1min", inclusive="left")
    events = find_events(times, levels(times), step_minutes=1)
    inside = (events["time_utc"] >= start) & (events["time_utc"] < end)
    return events[inside].reset_index(drop=True)

def _utc_ns(times) -> np.ndarray:
    index = pd.DatetimeIndex(times)
    if index.tz is not None:
        index = index.tz_convert("UTC").tz_localize(None)
    return index.to_numpy(dtype="datetime64[ns]")

def match_events(observed: pd.DataFrame, predicted: pd.DataFrame, max_minutes: float = MAX_MATCH_MINUTES) -> pd.DataFrame:
    """One row per observed event, paired with the nearest unused same-state prediction within
    max_minutes. Errors are predicted minus observed; unmatched rows have NaN errors and hit False."""
    predicted = predicted.reset_index(drop=True)
    rows = []
    for state in ("High", "Low"):
        obs = observed[observed["state"] == state].sort_values("time_utc")
        cand = predicted[predicted["state"] == state]
        cand_times = _utc_ns(cand["time_utc"])
        free = np.ones(len(cand), dtype=bool)
        for t, h, t_ns in zip(obs["time_utc"], obs["height_m"], _utc_ns(obs["time_utc"])):
            gap = (cand_times - t_ns) / np.timedelta64(1, "m")
            usable = free & (np.abs(gap) <= max_minutes)
            row = {"state": state, "observed_time_utc": t, "observed_height_m": float(h), "predicted_index": np.nan,
                   "predicted_time_utc": pd.NaT, "predicted_height_m": np.nan,
                   "time_error_minutes": np.nan, "height_error_m": np.nan}
            if usable.any():
                k = np.flatnonzero(usable)[np.argmin(np.abs(gap[usable]))]
                free[k] = False
                row.update(predicted_index=int(cand.index[k]), predicted_time_utc=cand["time_utc"].iloc[k],
                           predicted_height_m=float(cand["height_m"].iloc[k]), time_error_minutes=float(gap[k]),
                           height_error_m=float(cand["height_m"].iloc[k] - h))
            rows.append(row)
    matches = pd.DataFrame(rows, columns=MATCH_COLUMNS)
    matches["hit"] = ((matches["time_error_minutes"].abs() <= TIME_GATE_MINUTES)
                      & (matches["height_error_m"].abs() <= HEIGHT_GATE_M))
    return matches.sort_values("observed_time_utc", kind="stable").reset_index(drop=True)

def coverage(times, step_minutes: int = 60, margin: pd.Timedelta = pd.Timedelta(hours=2)) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Spans of regularly sampled observations, shrunk by margin at both ends."""
    times = pd.DatetimeIndex(times)
    if len(times) == 0:
        return []
    breaks = np.flatnonzero(np.diff(times.values) != np.timedelta64(step_minutes, "m")) + 1
    spans = []
    for block in np.split(np.arange(len(times)), breaks):
        first, last = times[block[0]] + margin, times[block[-1]] - margin
        if first < last:
            spans.append((first, last))
    return spans

def extra_events(predicted: pd.DataFrame, matches: pd.DataFrame, spans) -> pd.DataFrame:
    """Predicted events inside observed spans that no observed event claimed."""
    predicted = predicted.reset_index(drop=True)
    used = set(matches["predicted_index"].dropna().astype(int))
    inside = np.zeros(len(predicted), dtype=bool)
    for first, last in spans:
        inside |= ((predicted["time_utc"] >= first) & (predicted["time_utc"] <= last)).to_numpy()
    return predicted[inside & ~predicted.index.isin(list(used))].reset_index(drop=True)

def summarize(matches: pd.DataFrame, extra: int = 0) -> dict:
    """Scores for observed events (spec 5.3); joint_pct is the share of observed events hit."""
    n = len(matches)
    found = matches.dropna(subset=["time_error_minutes"])
    te, he = found["time_error_minutes"], found["height_error_m"]
    return {"observed": n, "matched": len(found), "missed": n - len(found), "extra": int(extra),
            "joint_pct": 100.0 * matches["hit"].sum() / n if n else np.nan,
            "time_mae_min": te.abs().mean(), "time_p95_min": te.abs().quantile(0.95), "time_bias_min": te.mean(),
            "height_mae_m": he.abs().mean(), "height_p95_m": he.abs().quantile(0.95), "height_bias_m": he.mean()}

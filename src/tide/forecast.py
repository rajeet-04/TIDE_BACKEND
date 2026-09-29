"""Forecasts from the current model (spec 6.1, 6.2): levels with components and 90% ranges,
high and low waters with their ranges, and the frequency summary."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from tide.ports import IST, datum_label, ist_years, port, season_of, to_utc
from tide.ranges import bounds
from tide.registry import load_current

BEYOND_HORIZON = "beyond_publishable_horizon"
NO_BACKTEST = "no_backtest"
NO_RANGES = "no_ranges"

@dataclass
class Forecast:
    port: str
    version: str
    datum: str
    levels: pd.DataFrame      # time_utc, time_ist, height_m, astronomical_m, seasonal_m, correction_m, lower_90, upper_90, flags
    events: pd.DataFrame      # state, time_utc, time_ist, height_m, raw_*, corrections, time/height lower and upper, flags
    frequency: pd.DataFrame   # state, count, average_interval_hours, event_times_ist, event_times_utc

def predict(port_slug: str, start, end=None, hours: float | None = None, step_minutes: int = 60) -> Forecast:
    """Forecast for [start, end) or [start, start + hours). Naive times are read as IST."""
    if (end is None) == (hours is None):
        raise ValueError("give exactly one of end or hours")
    start_utc = to_utc(start)
    end_utc = to_utc(end) if end is not None else start_utc + pd.Timedelta(hours=hours)
    if end_utc <= start_utc:
        raise ValueError("end must be after start")
    if step_minutes < 1:
        raise ValueError("step_minutes must be at least 1")
    version, fitted, metadata, ranges = load_current(port(port_slug).slug)
    times = pd.date_range(start_utc, end_utc, freq=f"{step_minutes}min", inclusive="left")
    parts = fitted.components(times)
    levels = pd.DataFrame({"time_utc": times, "time_ist": times.tz_convert(IST), "height_m": parts["height_m"].to_numpy(),
                           "astronomical_m": parts["astronomical_m"].to_numpy(), "seasonal_m": parts["seasonal_m"].to_numpy(),
                           "correction_m": parts["correction_m"].to_numpy()})
    lower, upper = _bounds(ranges, "level", levels["time_utc"], metadata)
    levels["lower_90"], levels["upper_90"] = levels["height_m"] + lower, levels["height_m"] + upper
    levels["flags"] = _flags(levels["time_utc"], metadata, ranges)
    events = fitted.event_detail(start_utc, end_utc)
    events.insert(2, "time_ist", events["time_utc"].dt.tz_convert(IST))
    minutes, metres = np.full((2, len(events)), np.nan), np.full((2, len(events)), np.nan)
    for state, prefix in (("High", "high"), ("Low", "low")):
        chosen = (events["state"] == state).to_numpy()
        minutes[:, chosen] = _bounds(ranges, f"{prefix}_time", events.loc[chosen, "time_utc"], metadata)
        metres[:, chosen] = _bounds(ranges, f"{prefix}_height", events.loc[chosen, "time_utc"], metadata)
    events["time_lower_utc"] = events["time_utc"] + pd.to_timedelta(minutes[0], unit="min")
    events["time_upper_utc"] = events["time_utc"] + pd.to_timedelta(minutes[1], unit="min")
    events["height_lower_m"], events["height_upper_m"] = events["height_m"] + metres[0], events["height_m"] + metres[1]
    events["flags"] = _flags(events["time_utc"], metadata, ranges)
    return Forecast(port_slug, version, datum_label(port_slug), levels, events, frequency(events))

def _horizon(times: pd.Series, metadata: dict) -> np.ndarray:
    """IST years after the last training year: 1 is one year ahead; 0 or less is inside training."""
    return ist_years(times) - pd.Timestamp(metadata["training_end_utc"]).tz_convert(IST).year

def _bounds(ranges: pd.DataFrame | None, output: str, times: pd.Series, metadata: dict) -> tuple[np.ndarray, np.ndarray]:
    """Range bounds for the times, from the cell of their horizon (clipped to 1-3) and season."""
    if ranges is None or len(times) == 0:
        return np.full(len(times), np.nan), np.full(len(times), np.nan)
    return bounds(ranges, output, _horizon(times, metadata), season_of(times))

def _flags(times: pd.Series, metadata: dict, ranges: pd.DataFrame | None) -> np.ndarray:
    """'|'-joined warnings per time: no_backtest when the version records no publishable
    horizons; beyond_publishable_horizon for forecast years outside them; no_ranges when the
    version has no error ranges."""
    publishable = metadata.get("publishable_horizons")
    horizon = _horizon(times, metadata) if len(times) else np.array([], dtype=int)
    if not publishable:
        flags = np.full(len(times), NO_BACKTEST, dtype=object)
    else:
        flags = np.where((horizon >= 1) & ~np.isin(horizon, publishable), BEYOND_HORIZON, "").astype(object)
    if ranges is None:
        flags = np.array([f"{flag}|{NO_RANGES}" if flag else NO_RANGES for flag in flags], dtype=object)
    return flags

def frequency(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for state in ("High", "Low"):
        times = events.loc[events["state"] == state, "time_utc"].sort_values()
        gaps = times.diff().dropna().dt.total_seconds() / 3600
        rows.append({"state": state, "count": len(times),
                     "average_interval_hours": float(gaps.mean()) if len(gaps) else np.nan,
                     "event_times_ist": ", ".join(times.dt.tz_convert(IST).dt.strftime("%H:%M")),
                     "event_times_utc": ", ".join(times.dt.strftime("%H:%M"))})
    return pd.DataFrame(rows)

def _naive(times: pd.Series) -> pd.Series:
    return times.dt.tz_localize(None)

def _ist(times: pd.Series) -> pd.Series:
    return _naive(times.dt.tz_convert(IST))

def write_outputs(forecast: Forecast, out_dir: Path) -> None:
    """The current file names and leading columns, with new columns appended (spec 6.2)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = {"datum": forecast.datum, "model_version": forecast.version}
    lv, ev = forecast.levels, forecast.events
    pd.DataFrame({"datetime_ist": _naive(lv["time_ist"]), "datetime_utc": _naive(lv["time_utc"]),
                  "predicted_height_m": lv["height_m"], "astronomical_m": lv["astronomical_m"],
                  "seasonal_m": lv["seasonal_m"], **tag, "flags": lv["flags"], "correction_m": lv["correction_m"],
                  "lower_90_m": lv["lower_90"], "upper_90_m": lv["upper_90"]}
                 ).to_csv(out_dir / "hourly_water_levels.csv", index=False)
    pd.DataFrame({"state": ev["state"], "datetime_ist": _naive(ev["time_ist"]), "datetime_utc": _naive(ev["time_utc"]),
                  "height_m": ev["height_m"], "raw_datetime_ist": _ist(ev["raw_time_utc"]), "raw_height_m": ev["raw_height_m"],
                  "time_correction_minutes": ev["time_correction_minutes"], "height_correction_m": ev["height_correction_m"],
                  **tag, "flags": ev["flags"], "time_lower_ist": _ist(ev["time_lower_utc"]),
                  "time_upper_ist": _ist(ev["time_upper_utc"]), "height_lower_m": ev["height_lower_m"],
                  "height_upper_m": ev["height_upper_m"]}
                 ).to_csv(out_dir / "predicted_tide_events.csv", index=False)
    forecast.frequency.to_csv(out_dir / "tide_frequency.csv", index=False)
    plot(forecast, out_dir / "water_level_forecast.png")

def plot(forecast: Forecast, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5))
    when = _naive(forecast.levels["time_ist"])
    if forecast.levels["lower_90"].notna().any():
        ax.fill_between(when, forecast.levels["lower_90"], forecast.levels["upper_90"], color="#006d77", alpha=0.15,
                        label="90% range")
    ax.plot(when, forecast.levels["height_m"], color="#006d77", marker="o", markersize=3, label="Water level")
    for state, color in (("High", "#d1495b"), ("Low", "#0077b6")):
        chosen = forecast.events[forecast.events["state"] == state]
        ax.scatter(_naive(chosen["time_ist"]), chosen["height_m"], color=color, s=55, zorder=3, label=f"{state} water")
    ax.set_title(f"{port(forecast.port).name}: water-level forecast (model {forecast.version})")
    ax.set_xlabel("Time (IST)")
    ax.set_ylabel(f"Height above {forecast.datum} (m)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %H:%M"))
    ax.grid(alpha=0.2)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

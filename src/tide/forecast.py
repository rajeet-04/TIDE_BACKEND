"""Forecasts from the current model (spec 6.1, 6.2): levels, high and low waters, frequency summary."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from tide.events import model_events
from tide.ports import IST, datum_label, ist_years, port, to_utc
from tide.registry import load_current

BEYOND_HORIZON = "beyond_publishable_horizon"

@dataclass
class Forecast:
    port: str
    version: str
    datum: str
    levels: pd.DataFrame      # time_utc, time_ist, height_m, astronomical_m, seasonal_m, flags
    events: pd.DataFrame      # state, time_utc, time_ist, height_m, flags
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
    version, model, metadata = load_current(port(port_slug).slug)
    times = pd.date_range(start_utc, end_utc, freq=f"{step_minutes}min", inclusive="left")
    parts = model.predict(times, components=True)
    levels = pd.DataFrame({"time_utc": times, "time_ist": times.tz_convert(IST),
                           "height_m": parts["height_m"].to_numpy(), "astronomical_m": parts["astronomical_m"].to_numpy(),
                           "seasonal_m": parts["seasonal_m"].to_numpy()})
    levels["flags"] = _flags(levels["time_utc"], metadata)
    events = model_events(model.predict, start_utc, end_utc)
    events.insert(2, "time_ist", events["time_utc"].dt.tz_convert(IST))
    events["flags"] = _flags(events["time_utc"], metadata)
    return Forecast(port_slug, version, datum_label(port_slug), levels, events, frequency(events))

def _flags(times: pd.Series, metadata: dict) -> np.ndarray:
    """Mark times beyond the publishable horizon, counted in IST years after the last training year."""
    publishable = metadata.get("publishable_horizons")
    if not publishable or len(times) == 0:
        return np.full(len(times), "", dtype=object)
    last_training_year = pd.Timestamp(metadata["training_end_utc"]).tz_convert(IST).year
    return np.where(ist_years(times) - last_training_year > max(publishable), BEYOND_HORIZON, "").astype(object)

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

def write_outputs(forecast: Forecast, out_dir: Path) -> None:
    """The current file names and leading columns, with new columns appended (spec 6.2).

    Model A has no event calibration, so raw_* equal the final values and the corrections are 0."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = {"datum": forecast.datum, "model_version": forecast.version}
    lv, ev = forecast.levels, forecast.events
    pd.DataFrame({"datetime_ist": _naive(lv["time_ist"]), "datetime_utc": _naive(lv["time_utc"]),
                  "predicted_height_m": lv["height_m"], "astronomical_m": lv["astronomical_m"],
                  "seasonal_m": lv["seasonal_m"], **tag, "flags": lv["flags"]}
                 ).to_csv(out_dir / "hourly_water_levels.csv", index=False)
    pd.DataFrame({"state": ev["state"], "datetime_ist": _naive(ev["time_ist"]), "datetime_utc": _naive(ev["time_utc"]),
                  "height_m": ev["height_m"], "raw_datetime_ist": _naive(ev["time_ist"]), "raw_height_m": ev["height_m"],
                  "time_correction_minutes": 0.0, "height_correction_m": 0.0, **tag, "flags": ev["flags"]}
                 ).to_csv(out_dir / "predicted_tide_events.csv", index=False)
    forecast.frequency.to_csv(out_dir / "tide_frequency.csv", index=False)
    plot(forecast, out_dir / "water_level_forecast.png")

def plot(forecast: Forecast, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(_naive(forecast.levels["time_ist"]), forecast.levels["height_m"], color="#006d77", marker="o",
            markersize=3, label="Water level")
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

"""Generate a 24-hour tide report and graphical error plots.

The forecast input is the hourly CSV emitted by forecast_combined.py. An
optional observed-events CSV enables event-level timing and height errors.

Usage:
  uv run python src/tide_report.py \
      --forecast output/forecast.csv \
      --start 2026-06-01 \
      --observed-events data/reference_tides_haldia_2026-06-01.csv \
      --out-dir output/tide_report_haldia_2026-06-01
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HEIGHT_COLUMNS = ("combined_predicted_m", "predicted_height_m", "predicted_m")


@dataclass(frozen=True)
class TideEvent:
    state: str
    datetime_ist: pd.Timestamp
    height_m: float
    is_boundary_estimate: bool = False


def _height_column(df: pd.DataFrame, requested: str | None) -> str:
    if requested:
        if requested not in df.columns:
            raise SystemExit(f"Missing requested height column: {requested}")
        return requested
    for column in HEIGHT_COLUMNS:
        if column in df.columns:
            return column
    raise SystemExit(
        "No forecast height column found. Expected one of: "
        + ", ".join(HEIGHT_COLUMNS)
    )


def _interpolated_event(
    state: str,
    timestamp: pd.Timestamp,
    previous: float,
    current: float,
    following: float,
) -> TideEvent:
    """Estimate a turning point using a parabola through three hourly samples."""
    a = (previous - 2 * current + following) / 2
    b = (following - previous) / 2
    offset_hours = float(np.clip(-b / (2 * a), -1, 1)) if a else 0.0
    height_m = current + b * offset_hours + a * offset_hours**2
    event_time = timestamp + pd.to_timedelta(offset_hours, unit="h")
    return TideEvent(state, event_time, float(height_m))


def find_tide_events(forecast: pd.DataFrame, height_column: str) -> list[TideEvent]:
    """Find high and low tides in an hourly forecast curve."""
    times = forecast["datetime_ist"].reset_index(drop=True)
    heights = forecast[height_column].to_numpy(dtype=float)
    if len(heights) < 2:
        raise SystemExit("Forecast needs at least two hourly rows.")

    events: list[TideEvent] = []
    if heights[0] > heights[1]:
        events.append(TideEvent("High", times.iloc[0], float(heights[0]), True))
    elif heights[0] < heights[1]:
        events.append(TideEvent("Low", times.iloc[0], float(heights[0]), True))

    for index in range(1, len(heights) - 1):
        previous, current, following = heights[index - 1 : index + 2]
        if current >= previous and current > following:
            events.append(
                _interpolated_event(
                    "High", times.iloc[index], previous, current, following
                )
            )
        elif current <= previous and current < following:
            events.append(
                _interpolated_event(
                    "Low", times.iloc[index], previous, current, following
                )
            )

    return events


def _events_frame(events: list[TideEvent]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "state": event.state,
                "datetime_ist": event.datetime_ist,
                "height_m": event.height_m,
                "is_boundary_estimate": event.is_boundary_estimate,
            }
            for event in events
        ]
    )


def _mark_hourly_events(hourly: pd.DataFrame, events: list[TideEvent]) -> pd.DataFrame:
    out = hourly.rename(columns={"forecast_height_m": "water_level_height_m"}).copy()
    out["tide_state"] = ""
    for event in events:
        distance = (out["datetime_ist"] - event.datetime_ist).abs()
        index = distance.idxmin()
        out.loc[index, "tide_state"] = event.state
    return out


def summarize_frequency(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for state in ("High", "Low"):
        state_events = events.loc[events["state"] == state, "datetime_ist"].sort_values()
        intervals = state_events.diff().dropna().dt.total_seconds() / 3600
        rows.append(
            {
                "state": state,
                "count_next_24h": len(state_events),
                "average_interval_hours": (
                    float(intervals.mean()) if not intervals.empty else np.nan
                ),
                "event_times": ", ".join(ts.strftime("%H:%M") for ts in state_events),
            }
        )
    return pd.DataFrame(rows)


def load_observed_events(path: Path, start: datetime, end: datetime) -> pd.DataFrame:
    observed = pd.read_csv(path, parse_dates=["datetime_ist"])
    required = {"state", "datetime_ist", "height_m"}
    missing = required - set(observed.columns)
    if missing:
        raise SystemExit(f"Observed events CSV missing columns: {sorted(missing)}")
    observed["state"] = observed["state"].str.strip().str.title()
    observed["height_m"] = pd.to_numeric(observed["height_m"])
    mask = (observed["datetime_ist"] >= start) & (observed["datetime_ist"] < end)
    return observed.loc[mask, ["state", "datetime_ist", "height_m"]].reset_index(
        drop=True
    )


def match_events(observed: pd.DataFrame, predicted: pd.DataFrame) -> pd.DataFrame:
    """Match each observed turning point to the nearest unused prediction of its type."""
    rows = []
    used: set[int] = set()
    for _, obs in observed.sort_values("datetime_ist").iterrows():
        candidates = predicted[
            (predicted["state"] == obs["state"]) & (~predicted.index.isin(used))
        ]
        if candidates.empty:
            continue
        distances = (candidates["datetime_ist"] - obs["datetime_ist"]).abs()
        predicted_index = int(distances.idxmin())
        pred = predicted.loc[predicted_index]
        used.add(predicted_index)
        time_error = (
            pred["datetime_ist"] - obs["datetime_ist"]
        ).total_seconds() / 60
        height_error = float(pred["height_m"] - obs["height_m"])
        rows.append(
            {
                "state": obs["state"],
                "observed_datetime_ist": obs["datetime_ist"],
                "predicted_datetime_ist": pred["datetime_ist"],
                "time_error_minutes": time_error,
                "absolute_time_error_minutes": abs(time_error),
                "observed_height_m": obs["height_m"],
                "predicted_height_m": pred["height_m"],
                "height_error_m": height_error,
                "absolute_height_error_m": abs(height_error),
                "prediction_is_boundary_estimate": pred["is_boundary_estimate"],
            }
        )
    return pd.DataFrame(rows)


def error_metrics(errors: pd.DataFrame) -> dict[str, float | int]:
    if errors.empty:
        return {"matched_events": 0}
    return {
        "matched_events": len(errors),
        "mean_absolute_time_error_minutes": float(
            errors["absolute_time_error_minutes"].mean()
        ),
        "time_rmse_minutes": float(
            np.sqrt(np.mean(errors["time_error_minutes"] ** 2))
        ),
        "mean_absolute_height_error_m": float(
            errors["absolute_height_error_m"].mean()
        ),
        "height_rmse_m": float(np.sqrt(np.mean(errors["height_error_m"] ** 2))),
        "height_bias_m": float(errors["height_error_m"].mean()),
    }


def plot_water_levels(
    hourly: pd.DataFrame,
    predicted: pd.DataFrame,
    observed: pd.DataFrame | None,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        hourly["datetime_ist"],
        hourly["forecast_height_m"],
        color="#006d77",
        marker="o",
        markersize=3,
        linewidth=1.5,
        label="Predicted hourly water level",
    )
    colors = {"High": "#d1495b", "Low": "#0077b6"}
    for state in ("High", "Low"):
        points = predicted[predicted["state"] == state]
        ax.scatter(
            points["datetime_ist"],
            points["height_m"],
            color=colors[state],
            s=55,
            zorder=3,
            label=f"Predicted {state.lower()} tide",
        )
    if observed is not None and not observed.empty:
        ax.scatter(
            observed["datetime_ist"],
            observed["height_m"],
            color="#222222",
            marker="x",
            s=70,
            linewidths=2,
            zorder=4,
            label="Observed tide events",
        )
    ax.set_title("24-hour water level forecast and tide turning points")
    ax.set_ylabel("Water level height (m)")
    ax.set_xlabel("Time (IST)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.grid(alpha=0.25)
    ax.legend(ncols=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_errors(errors: pd.DataFrame, out_path: Path) -> None:
    labels = [
        f"{row.state} {row.observed_datetime_ist:%H:%M}"
        for row in errors.itertuples()
    ]
    colors = ["#d1495b" if state == "High" else "#0077b6" for state in errors["state"]]
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].bar(labels, errors["time_error_minutes"], color=colors)
    axes[0].axhline(0, color="#222222", linewidth=0.8)
    axes[0].set_ylabel("Time error (minutes)")
    axes[0].set_title(
        "High/low tide prediction errors "
        f"(time MAE: {errors['absolute_time_error_minutes'].mean():.1f} min)"
    )
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(labels, errors["height_error_m"], color=colors)
    axes[1].axhline(0, color="#222222", linewidth=0.8)
    axes[1].set_ylabel("Height error (m)")
    axes[1].set_title(
        f"Water level errors (height MAE: {errors['absolute_height_error_m'].mean():.3f} m)"
    )
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a 24-hour tide table, turning points, and error plots."
    )
    parser.add_argument("--forecast", required=True, type=Path)
    parser.add_argument("--start", required=True, help="Report start time in ISO format")
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--height-column")
    parser.add_argument("--observed-events", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("output/tide_report"))
    args = parser.parse_args()

    start = datetime.fromisoformat(args.start)
    end = start + timedelta(hours=args.hours)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    forecast = pd.read_csv(args.forecast, parse_dates=["datetime_ist"])
    forecast = forecast.sort_values("datetime_ist").reset_index(drop=True)
    height_column = _height_column(forecast, args.height_column)
    forecast["forecast_height_m"] = pd.to_numeric(forecast[height_column])

    all_predicted = _events_frame(find_tide_events(forecast, "forecast_height_m"))
    mask = (forecast["datetime_ist"] >= start) & (forecast["datetime_ist"] < end)
    hourly = forecast.loc[mask, ["datetime_ist", "forecast_height_m"]].reset_index(
        drop=True
    )
    if hourly.empty:
        raise SystemExit(f"No forecast rows between {start} and {end}")

    event_mask = (all_predicted["datetime_ist"] >= start) & (
        all_predicted["datetime_ist"] < end
    )
    report_events = all_predicted.loc[event_mask].reset_index(drop=True)
    hourly_report = _mark_hourly_events(hourly, list(report_events.itertuples(index=False)))
    frequency = summarize_frequency(report_events)

    hourly_report.to_csv(args.out_dir / "hourly_water_levels.csv", index=False)
    report_events.to_csv(args.out_dir / "predicted_tide_events.csv", index=False)
    frequency.to_csv(args.out_dir / "tide_frequency.csv", index=False)

    observed = None
    metrics = None
    if args.observed_events:
        observed = load_observed_events(args.observed_events, start, end)
        errors = match_events(observed, all_predicted)
        errors.to_csv(args.out_dir / "tide_event_errors.csv", index=False)
        metrics = error_metrics(errors)
        (args.out_dir / "error_metrics.json").write_text(
            json.dumps(metrics, indent=2) + "\n"
        )
        if not errors.empty:
            plot_errors(errors, args.out_dir / "tide_event_errors.png")

    plot_water_levels(
        hourly, report_events, observed, args.out_dir / "water_level_forecast.png"
    )

    print(f"24-hour report: {start} -> {end}")
    print(f"hourly rows: {len(hourly_report)}")
    print("\npredicted tide events:")
    print(report_events.to_string(index=False))
    print("\nfrequency:")
    print(frequency.to_string(index=False))
    if metrics is not None:
        print("\nerror metrics:")
        print(json.dumps(metrics, indent=2))
    print(f"\nwrote report files -> {args.out_dir}")


if __name__ == "__main__":
    main()

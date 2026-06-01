"""Train, forecast, and validate future-safe UTide event models.

The existing residual LSTM is useful for short-horizon hindcasts, but it cannot
forecast environmental residuals years after the last observation. This model
uses UTide's astronomical nodal corrections and applies event-level calibration
learned only from historical hourly tide tables.

Accuracy gates are explicit:
  - timing: predicted high/low event is within 30 minutes
  - amplitude: predicted event height is within 0.30 metres
  - joint: both conditions hold for the same event

Usage:
  uv run python src/utide_event_model.py train
  uv run python src/utide_event_model.py validate
  uv run python src/utide_event_model.py forecast \
      --port haldia --start 2026-06-01 --hours 24
"""
from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.ensemble import ExtraTreesRegressor
from utide import reconstruct, solve

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
MODEL_DIR = OUTPUT_DIR / "utide_models"
VALIDATION_DIR = OUTPUT_DIR / "utide_validation_2026"

PORTS = {
    "haldia": {"name": "HALDIA", "latitude": 22 + 2 / 60},
    "diamond_harbour": {"name": "DIAMOND HARBOUR", "latitude": 22 + 12 / 60},
}

PROMINENCE_M = 0.5
MIN_EVENT_SEPARATION_HOURS = 4
MAX_MATCH_ERROR_MINUTES = 180
TIME_GATE_MINUTES = 30
HEIGHT_GATE_M = 0.30
MOON_EPOCH = pd.Timestamp("2000-01-07 00:00:00")
SYNODIC_MONTH_DAYS = 29.530588853
IST_OFFSET = pd.Timedelta(hours=5, minutes=30)


def _reconstruct_chunked(
    timestamps: pd.Series | pd.DatetimeIndex,
    coefficients,
    chunk_size: int = 50_000,
) -> np.ndarray:
    """Reconstruct in bounded-memory chunks for multi-decade hourly archives."""
    timestamps = pd.DatetimeIndex(timestamps)
    chunks = []
    for start in range(0, len(timestamps), chunk_size):
        chunk = timestamps[start : start + chunk_size]
        chunks.append(
            reconstruct(
                chunk,
                coefficients,
                min_SNR=0,
                min_PE=0,
                verbose=False,
            ).h
        )
    return np.concatenate(chunks)


def _quadratic_vertex(previous: float, current: float, following: float) -> tuple[float, float]:
    """Return sub-sample turning-point offset and height for three samples."""
    a = (previous - 2 * current + following) / 2
    b = (following - previous) / 2
    offset = float(np.clip(-b / (2 * a), -1, 1)) if a else 0.0
    return offset, float(current + b * offset + a * offset**2)


def find_events(
    timestamps: pd.Series | pd.DatetimeIndex,
    heights: np.ndarray,
    *,
    expected_step_minutes: int,
) -> pd.DataFrame:
    """Extract prominent tide turning points from regularly sampled blocks."""
    timestamps = pd.DatetimeIndex(timestamps)
    heights = np.asarray(heights, dtype=float)
    if len(timestamps) != len(heights):
        raise ValueError("timestamp and height lengths differ")

    differences = np.diff(timestamps.values).astype("timedelta64[m]").astype(int)
    block_starts = np.where(differences != expected_step_minutes)[0] + 1
    minimum_distance = max(
        1, round(MIN_EVENT_SEPARATION_HOURS * 60 / expected_step_minutes)
    )
    rows = []
    for indices in np.split(np.arange(len(timestamps)), block_starts):
        if len(indices) < 5:
            continue
        values = heights[indices]
        high_indices, _ = find_peaks(
            values, distance=minimum_distance, prominence=PROMINENCE_M
        )
        low_indices, _ = find_peaks(
            -values, distance=minimum_distance, prominence=PROMINENCE_M
        )
        for state, peaks in (("High", high_indices), ("Low", low_indices)):
            for local_index in peaks:
                index = int(indices[local_index])
                offset, height = _quadratic_vertex(
                    heights[index - 1], heights[index], heights[index + 1]
                )
                rows.append(
                    {
                        "state": state,
                        "datetime_ist": timestamps[index]
                        + pd.to_timedelta(
                            offset * expected_step_minutes, unit="minutes"
                        ),
                        "height_m": height,
                    }
                )
    return pd.DataFrame(rows).sort_values("datetime_ist").reset_index(drop=True)


def match_events(observed: pd.DataFrame, predicted: pd.DataFrame) -> pd.DataFrame:
    """Bounded nearest-neighbour event match by tide state."""
    rows = []
    used: set[int] = set()
    for obs in observed.sort_values("datetime_ist").itertuples():
        candidates = predicted[
            (predicted["state"] == obs.state) & (~predicted.index.isin(used))
        ]
        if candidates.empty:
            continue
        distances = (
            (candidates["datetime_ist"] - obs.datetime_ist)
            .abs()
            .dt.total_seconds()
            / 60
        )
        distances = distances[distances <= MAX_MATCH_ERROR_MINUTES]
        if distances.empty:
            continue
        predicted_index = int(distances.idxmin())
        pred = candidates.loc[predicted_index]
        used.add(predicted_index)
        time_error = (
            pred["datetime_ist"] - obs.datetime_ist
        ).total_seconds() / 60
        height_error = float(pred["height_m"] - obs.height_m)
        rows.append(
            {
                "predicted_index": predicted_index,
                "state": obs.state,
                "observed_datetime_ist": obs.datetime_ist,
                "predicted_datetime_ist": pred["datetime_ist"],
                "time_error_minutes": time_error,
                "absolute_time_error_minutes": abs(time_error),
                "observed_height_m": float(obs.height_m),
                "predicted_height_m": float(pred["height_m"]),
                "height_error_m": height_error,
                "absolute_height_error_m": abs(height_error),
            }
        )
    return pd.DataFrame(rows)


def _serialize_month_state(series: pd.Series) -> dict[str, float]:
    return {
        f"{month}:{state}": float(value)
        for (month, state), value in series.items()
    }


def _event_features(events: pd.DataFrame) -> pd.DataFrame:
    """Build future-known event context, including explicit lunar phase."""
    events = events.sort_values("datetime_ist").reset_index(drop=True)
    timestamps = events["datetime_ist"]
    heights = events["height_m"]
    previous_heights = heights.shift(1).fillna(heights)
    following_heights = heights.shift(-1).fillna(heights)
    previous_intervals = (
        (timestamps - timestamps.shift(1))
        .dt.total_seconds()
        .div(3600)
        .fillna(6)
        .clip(0, 24)
    )
    following_intervals = (
        (timestamps.shift(-1) - timestamps)
        .dt.total_seconds()
        .div(3600)
        .fillna(6)
        .clip(0, 24)
    )
    day_of_year = (
        timestamps.dt.dayofyear
        + timestamps.dt.hour / 24
        + timestamps.dt.minute / (24 * 60)
    )
    hour = timestamps.dt.hour + timestamps.dt.minute / 60
    lunar_angle = (
        (timestamps - MOON_EPOCH).dt.total_seconds()
        / (24 * 60 * 60)
        / SYNODIC_MONTH_DAYS
        * 2
        * np.pi
    )
    elapsed_years = (
        (timestamps - pd.Timestamp("2000-01-01")).dt.total_seconds()
        / (365.25 * 24 * 60 * 60)
    )
    return pd.DataFrame(
        {
            "state_high": (events["state"] == "High").astype(float),
            "height": heights,
            "previous_height": previous_heights,
            "following_height": following_heights,
            "previous_range": (heights - previous_heights).abs(),
            "following_range": (heights - following_heights).abs(),
            "previous_interval": previous_intervals,
            "following_interval": following_intervals,
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "day_of_year_sin": np.sin(2 * np.pi * day_of_year / 365.25),
            "day_of_year_cos": np.cos(2 * np.pi * day_of_year / 365.25),
            "moon_sin": np.sin(lunar_angle),
            "moon_cos": np.cos(lunar_angle),
            "moon_second_harmonic_sin": np.sin(2 * lunar_angle),
            "moon_second_harmonic_cos": np.cos(2 * lunar_angle),
            "elapsed_years": elapsed_years,
        }
    )


def _extra_trees() -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=160,
        min_samples_leaf=15,
        max_features=0.9,
        n_jobs=-1,
        random_state=1,
    )


def _historical_calibration(
    history: pd.DataFrame, predicted_heights: np.ndarray
) -> tuple[ExtraTreesRegressor, ExtraTreesRegressor, dict]:
    observed_events = find_events(
        history["datetime_ist"], history["height_m"], expected_step_minutes=60
    )
    predicted_events = find_events(
        history["datetime_ist"], predicted_heights, expected_step_minutes=60
    )
    errors = match_events(observed_events, predicted_events)

    last_observation = history["datetime_ist"].max()
    recent_start = datetime(last_observation.year - 3, 1, 1)
    recent_errors = errors[errors["observed_datetime_ist"] >= recent_start]
    features = _event_features(predicted_events).loc[
        errors["predicted_index"]
    ].reset_index(drop=True)
    recent_mask = errors["observed_datetime_ist"] >= recent_start
    time_calibrator = _extra_trees()
    height_calibrator = _extra_trees()
    time_calibrator.fit(features.loc[recent_mask], errors.loc[recent_mask, "time_error_minutes"])
    height_calibrator.fit(features, errors["height_error_m"])
    return (
        time_calibrator,
        height_calibrator,
        {
            "observed_historical_events": len(observed_events),
            "matched_historical_events": len(errors),
            "recent_calibration_start": recent_start.isoformat(),
            "recent_calibration_events": len(recent_errors),
        },
    )


def train_port(port_slug: str) -> Path:
    port = PORTS[port_slug]
    history_path = DATA_DIR / f"{port_slug}.csv"
    history = pd.read_csv(history_path, parse_dates=["datetime_ist"])
    history = history.sort_values("datetime_ist").reset_index(drop=True)
    print(
        f"[{port_slug}] solving UTide from {len(history):,} hourly rows "
        f"({history['datetime_ist'].min()} -> {history['datetime_ist'].max()})"
    )
    coefficients = solve(
        history["datetime_ist"],
        history["height_m"].to_numpy(),
        lat=port["latitude"],
        constit="auto",
        method="ols",
        conf_int="none",
        trend=True,
        nodal=True,
        verbose=False,
    )
    predicted_heights = _reconstruct_chunked(history["datetime_ist"], coefficients)
    time_calibrator, height_calibrator, calibration = _historical_calibration(
        history, predicted_heights
    )
    artifact = {
        "version": 2,
        "port_slug": port_slug,
        "port_name": port["name"],
        "latitude": port["latitude"],
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "history_start": history["datetime_ist"].min().isoformat(),
        "history_end": history["datetime_ist"].max().isoformat(),
        "history_rows": len(history),
        "constituent_count": len(coefficients.name),
        "coefficients": coefficients,
        "calibrator": "extra_trees_lunar_event_context",
        "time_calibrator": time_calibrator,
        "height_calibrator": height_calibrator,
        "calibration": calibration,
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"{port_slug}.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(artifact, handle)
    print(f"  constituents: {len(coefficients.name)}")
    print(f"  calibrator: {artifact['calibrator']}")
    print(f"  -> {model_path}")
    return model_path


def load_model(port_slug: str) -> dict:
    model_path = MODEL_DIR / f"{port_slug}.pkl"
    if not model_path.exists():
        raise SystemExit(f"Missing {model_path}; run the train command first.")
    with model_path.open("rb") as handle:
        return pickle.load(handle)


def _apply_event_calibration(events: pd.DataFrame, model: dict) -> pd.DataFrame:
    out = events.rename(
        columns={"datetime_ist": "raw_datetime_ist", "height_m": "raw_height_m"}
    ).copy()
    features = _event_features(events)
    out["time_correction_minutes"] = model["time_calibrator"].predict(features)
    out["height_correction_m"] = model["height_calibrator"].predict(features)
    out["datetime_ist"] = out["raw_datetime_ist"] - pd.to_timedelta(
        out["time_correction_minutes"], unit="minutes"
    )
    out["height_m"] = out["raw_height_m"] - out["height_correction_m"]
    return out[
        [
            "state",
            "datetime_ist",
            "height_m",
            "raw_datetime_ist",
            "raw_height_m",
            "time_correction_minutes",
            "height_correction_m",
        ]
    ]


def predict_events(model: dict, start: datetime, end: datetime) -> pd.DataFrame:
    padded_start = start - timedelta(hours=12)
    padded_end = end + timedelta(hours=12)
    timestamps = pd.date_range(padded_start, padded_end, freq="1min")
    heights = _reconstruct_chunked(timestamps, model["coefficients"])
    raw_events = find_events(timestamps, heights, expected_step_minutes=1)
    events = _apply_event_calibration(raw_events, model)
    mask = (events["datetime_ist"] >= start) & (events["datetime_ist"] < end)
    return events.loc[mask].reset_index(drop=True)


def predict_hourly(model: dict, start: datetime, end: datetime) -> pd.DataFrame:
    timestamps = pd.date_range(start, end, freq="1h", inclusive="left")
    heights = _reconstruct_chunked(timestamps, model["coefficients"])
    return pd.DataFrame({"datetime_ist": timestamps, "predicted_height_m": heights})


def metrics(errors: pd.DataFrame, observed_count: int) -> dict[str, float | int | bool]:
    time_hits = errors["absolute_time_error_minutes"] <= TIME_GATE_MINUTES
    height_hits = errors["absolute_height_error_m"] <= HEIGHT_GATE_M
    matched_count = len(errors)
    denominator = observed_count
    return {
        "observed_events": observed_count,
        "matched_events": matched_count,
        "match_rate_percent": 100 * matched_count / denominator,
        "time_mae_minutes": float(errors["absolute_time_error_minutes"].mean()),
        "time_rmse_minutes": float(
            np.sqrt(np.mean(errors["time_error_minutes"] ** 2))
        ),
        "time_accuracy_percent": float(100 * time_hits.sum() / denominator),
        "height_mae_m": float(errors["absolute_height_error_m"].mean()),
        "height_rmse_m": float(np.sqrt(np.mean(errors["height_error_m"] ** 2))),
        "height_accuracy_percent": float(100 * height_hits.sum() / denominator),
        "joint_accuracy_percent": float(100 * (time_hits & height_hits).sum() / denominator),
        "time_gate_minutes": TIME_GATE_MINUTES,
        "height_gate_m": HEIGHT_GATE_M,
        "passes_time_accuracy_gate": bool(100 * time_hits.sum() / denominator >= 95),
        "passes_height_accuracy_gate": bool(100 * height_hits.sum() / denominator >= 95),
        "passes_joint_accuracy_gate": bool(
            100 * (time_hits & height_hits).sum() / denominator >= 95
        ),
    }


def _plot_errors(errors: pd.DataFrame, port_name: str, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].scatter(
        errors["observed_datetime_ist"], errors["time_error_minutes"], s=10, alpha=0.7
    )
    axes[0].axhline(0, color="#222222", linewidth=0.8)
    axes[0].axhline(TIME_GATE_MINUTES, color="#d1495b", linestyle="--", linewidth=0.8)
    axes[0].axhline(-TIME_GATE_MINUTES, color="#d1495b", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("Time error (minutes)")
    axes[0].grid(alpha=0.2)

    axes[1].scatter(
        errors["observed_datetime_ist"], errors["height_error_m"], s=10, alpha=0.7
    )
    axes[1].axhline(0, color="#222222", linewidth=0.8)
    axes[1].axhline(HEIGHT_GATE_M, color="#d1495b", linestyle="--", linewidth=0.8)
    axes[1].axhline(-HEIGHT_GATE_M, color="#d1495b", linestyle="--", linewidth=0.8)
    axes[1].set_ylabel("Height error (m)")
    axes[1].set_xlabel("Observed tide event date (IST)")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    axes[1].grid(alpha=0.2)

    fig.suptitle(f"{port_name}: April-June 2026 tide event errors")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _event_frequency(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for state in ("High", "Low"):
        state_events = events.loc[
            events["state"] == state, "datetime_ist"
        ].sort_values()
        intervals = state_events.diff().dropna().dt.total_seconds() / 3600
        rows.append(
            {
                "state": state,
                "count": len(state_events),
                "average_interval_hours": (
                    float(intervals.mean()) if not intervals.empty else np.nan
                ),
                "event_times_ist": ", ".join(
                    timestamp.strftime("%H:%M") for timestamp in state_events
                ),
                "event_times_utc": ", ".join(
                    (timestamp - IST_OFFSET).strftime("%H:%M")
                    for timestamp in state_events
                ),
            }
        )
    return pd.DataFrame(rows)


def _add_utc_column(
    frame: pd.DataFrame, ist_column: str = "datetime_ist"
) -> pd.DataFrame:
    out = frame.copy()
    out.insert(
        out.columns.get_loc(ist_column) + 1,
        "datetime_utc",
        out[ist_column] - IST_OFFSET,
    )
    return out


def _plot_forecast(
    hourly: pd.DataFrame, events: pd.DataFrame, port_name: str, out_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        hourly["datetime_ist"],
        hourly["predicted_height_m"],
        color="#006d77",
        marker="o",
        markersize=3,
        label="Hourly UTide water level",
    )
    colors = {"High": "#d1495b", "Low": "#0077b6"}
    for state in ("High", "Low"):
        state_events = events[events["state"] == state]
        ax.scatter(
            state_events["datetime_ist"],
            state_events["height_m"],
            color=colors[state],
            s=55,
            zorder=3,
            label=f"Calibrated {state.lower()} tide",
        )
    ax.set_title(f"{port_name}: 24-hour water-level forecast")
    ax.set_xlabel("Time (IST)")
    ax.set_ylabel("Water level height (m)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def validate(reference_path: Path, out_dir: Path) -> dict:
    references = pd.read_csv(reference_path, parse_dates=["datetime_ist"])
    out_dir.mkdir(parents=True, exist_ok=True)
    all_errors = []
    report = {}
    for port_slug, port in PORTS.items():
        model = load_model(port_slug)
        observed = references[references["port"] == port["name"]].copy()
        start = observed["datetime_ist"].min().normalize().to_pydatetime()
        end = (
            observed["datetime_ist"].max().normalize() + pd.Timedelta(days=1)
        ).to_pydatetime()
        predicted = predict_events(model, start, end)
        errors = match_events(observed, predicted)
        errors.insert(0, "port", port["name"])
        all_errors.append(errors)
        port_metrics = metrics(errors, len(observed))
        report[port_slug] = port_metrics
        _plot_errors(errors, port["name"], out_dir / f"{port_slug}_event_errors.png")
        predicted.to_csv(out_dir / f"{port_slug}_predicted_events.csv", index=False)
        print(f"[{port_slug}] {json.dumps(port_metrics, indent=2)}")

    errors = pd.concat(all_errors, ignore_index=True)
    errors.to_csv(out_dir / "event_errors.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote validation artifacts -> {out_dir}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train")
    train_parser.add_argument("--port", choices=[*PORTS, "all"], default="all")

    validate_parser = commands.add_parser("validate")
    validate_parser.add_argument(
        "--reference-events",
        type=Path,
        default=DATA_DIR / "reference_tides_2026.csv",
    )
    validate_parser.add_argument("--out-dir", type=Path, default=VALIDATION_DIR)

    forecast_parser = commands.add_parser("forecast")
    forecast_parser.add_argument("--port", choices=PORTS, required=True)
    forecast_parser.add_argument("--start", required=True)
    forecast_parser.add_argument("--hours", type=int, default=24)
    forecast_parser.add_argument("--out-dir", type=Path)

    args = parser.parse_args()
    if args.command == "train":
        port_slugs = PORTS if args.port == "all" else [args.port]
        for port_slug in port_slugs:
            train_port(port_slug)
    elif args.command == "validate":
        validate(args.reference_events, args.out_dir)
    else:
        start = datetime.fromisoformat(args.start)
        end = start + timedelta(hours=args.hours)
        out_dir = args.out_dir or OUTPUT_DIR / f"utide_forecast_{args.port}_{start.date()}"
        out_dir.mkdir(parents=True, exist_ok=True)
        model = load_model(args.port)
        hourly = predict_hourly(model, start, end)
        events = predict_events(model, start, end)
        frequency = _event_frequency(events)
        hourly_output = _add_utc_column(hourly)
        events_output = _add_utc_column(events)
        hourly_output.to_csv(out_dir / "hourly_water_levels.csv", index=False)
        events_output.to_csv(out_dir / "predicted_tide_events.csv", index=False)
        frequency.to_csv(out_dir / "tide_frequency.csv", index=False)
        _plot_forecast(hourly, events, model["port_name"], out_dir / "water_level_forecast.png")
        print(hourly_output.to_string(index=False))
        print("\npredicted tide events:")
        print(
            events_output[
                ["state", "datetime_ist", "datetime_utc", "height_m"]
            ].to_string(index=False)
        )
        print("\ntide frequency:")
        print(frequency.to_string(index=False))
        print(f"\nwrote forecast artifacts -> {out_dir}")


if __name__ == "__main__":
    main()

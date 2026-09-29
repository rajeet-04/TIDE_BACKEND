import numpy as np
import pandas as pd
import pytest

from tide import forecast, registry
from tide.harmonic import HarmonicConfig, HarmonicModel
from tide.ports import SEASONS
from tide.ranges import OUTPUTS
from tide.stack import FittedStack

def fitted_a() -> tuple[FittedStack, pd.DatetimeIndex]:
    times = pd.date_range("2023-12-31 18:30", periods=2 * 8766, freq="1h", tz="UTC")
    hours = np.arange(len(times), dtype=float)
    heights = 3.2 + 2.0 * np.cos(2 * np.pi * hours / 12.4206) + 0.4 * np.cos(2 * np.pi * hours / 12.0)
    model = HarmonicModel(22.03, HarmonicConfig(constituents=("M2", "S2", "K1", "M4"), seasonal=1)).fit(times, heights)
    return FittedStack(model), times

def flat_ranges(width: float = 0.2) -> pd.DataFrame:
    """Ranges of ±width (m) for levels and heights, ±100·width minutes for times, horizon 1 wider."""
    return pd.DataFrame([{"output": o, "horizon": h, "season": s, "lower": -width * (100 if "time" in o else 1) * (2 if h == 1 else 1),
                          "upper": width * (100 if "time" in o else 1) * (2 if h == 1 else 1), "n": 500, "source": "cell"}
                         for o in OUTPUTS for h in (1, 2, 3) for s in SEASONS])

@pytest.fixture
def current_model(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "MODELS_DIR", tmp_path / "models")
    fitted, times = fitted_a()
    version = registry.save_version("haldia", fitted, {"training_end_utc": str(times[-1]), "publishable_horizons": [1]},
                                    flat_ranges())
    registry.set_current("haldia", version)
    return version

def test_naive_start_is_ist_and_events_stay_inside_the_window(current_model):
    result = forecast.predict("haldia", "2026-10-01", hours=48)
    start, end = pd.Timestamp("2026-09-30 18:30", tz="UTC"), pd.Timestamp("2026-10-02 18:30", tz="UTC")
    assert result.levels["time_utc"].iloc[0] == start and len(result.levels) == 48
    assert result.events["time_utc"].between(start, end, inclusive="left").all()
    assert 7 <= len(result.events) <= 8
    assert result.frequency["count"].sum() == len(result.events)
    assert result.datum == "chart datum (Survey of India)" and result.version == current_model
    assert (result.levels["flags"] == "").all()

def test_times_beyond_the_publishable_horizon_are_flagged(current_model):
    result = forecast.predict("haldia", "2028-01-01", hours=6)
    assert (result.levels["flags"] == forecast.BEYOND_HORIZON).all()

def test_files_keep_the_current_names_and_leading_columns(current_model, tmp_path):
    out = tmp_path / "out"
    forecast.write_outputs(forecast.predict("haldia", "2026-10-01T00:00+05:30", end="2026-10-02T00:00+05:30"), out)
    hourly = pd.read_csv(out / "hourly_water_levels.csv")
    events = pd.read_csv(out / "predicted_tide_events.csv")
    assert list(hourly.columns[:3]) == ["datetime_ist", "datetime_utc", "predicted_height_m"]
    assert list(events.columns[:8]) == ["state", "datetime_ist", "datetime_utc", "height_m", "raw_datetime_ist",
                                        "raw_height_m", "time_correction_minutes", "height_correction_m"]
    assert {"datum", "model_version", "flags"} <= set(hourly.columns) & set(events.columns)
    assert hourly["datetime_ist"].iloc[0] == "2026-10-01 00:00:00"
    assert (out / "tide_frequency.csv").exists() and (out / "water_level_forecast.png").exists()

def test_bad_requests_are_clear_errors(current_model):
    with pytest.raises(ValueError, match="after start"):
        forecast.predict("haldia", "2026-10-02", end="2026-10-01")
    with pytest.raises(ValueError, match="exactly one"):
        forecast.predict("haldia", "2026-10-02")
    with pytest.raises(FileNotFoundError):
        registry.set_current("haldia", "20000101-missing")

def test_refitting_never_overwrites_a_saved_version(current_model):
    before = (registry.MODELS_DIR / "haldia" / current_model / "a.json").read_text()
    model = registry.load_current("haldia")[1]
    again = registry.save_version("haldia", model, {"note": "second fit"})
    assert again != current_model
    assert (registry.MODELS_DIR / "haldia" / current_model / "a.json").read_text() == before
    assert "note" not in registry.load_current("haldia")[2]

def test_levels_and_events_carry_their_90_percent_ranges(current_model):
    result = forecast.predict("haldia", "2026-10-01", hours=48)        # 2026 is horizon 1 after 2025
    np.testing.assert_allclose(result.levels["upper_90"] - result.levels["height_m"], 0.4)
    np.testing.assert_allclose(result.levels["height_m"] - result.levels["lower_90"], 0.4)
    width = (result.events["time_upper_utc"] - result.events["time_lower_utc"]).dt.total_seconds() / 60
    np.testing.assert_allclose(width, 80.0)
    assert (result.events["height_lower_m"] < result.events["height_m"]).all()
    assert (result.levels["correction_m"] == 0).all()

def test_far_forecasts_use_the_last_horizon_and_hindcasts_are_not_flagged(current_model):
    far = forecast.predict("haldia", "2029-03-01", hours=6)
    np.testing.assert_allclose(far.levels["upper_90"] - far.levels["height_m"], 0.2)     # horizon 3's cell
    assert (far.levels["flags"] == forecast.BEYOND_HORIZON).all()
    assert (forecast.predict("haldia", "2025-03-01", hours=6).levels["flags"] == "").all()

def test_versions_without_a_backtest_or_ranges_are_flagged(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "MODELS_DIR", tmp_path / "models")
    fitted, times = fitted_a()
    folder = tmp_path / "models" / "haldia" / "20260927-plan-1a-version"
    folder.mkdir(parents=True)
    fitted.save(folder)                                                  # a.json only, as plan 1a saved
    (folder / "metadata.json").write_text(f'{{"training_end_utc": "{times[-1]}"}}')
    registry.set_current("haldia", folder.name)
    result = forecast.predict("haldia", "2026-10-01", hours=6)
    assert (result.levels["flags"] == "no_backtest|no_ranges").all()
    assert result.levels["lower_90"].isna().all() and result.events["time_lower_utc"].isna().all()

def test_saved_versions_reproduce_their_snapshots(current_model):
    for base in (registry.MODELS_DIR, registry.ROOT / "models"):
        for folder in sorted(base.glob("*/*/snapshot_levels.csv")):
            fitted = FittedStack.load(folder.parent)
            levels, events = registry.snapshot(fitted)
            saved = pd.read_csv(folder)
            np.testing.assert_allclose(levels["height_m"], saved["height_m"], atol=1e-9)
            saved_events = pd.read_csv(folder.parent / "snapshot_events.csv")
            assert events["state"].tolist() == saved_events["state"].tolist()
            gap = pd.to_datetime(saved_events["time_utc"], utc=True) - events["time_utc"].reset_index(drop=True)
            assert gap.abs().max() <= pd.Timedelta(microseconds=1)
            np.testing.assert_allclose(events["height_m"], saved_events["height_m"], atol=1e-9)

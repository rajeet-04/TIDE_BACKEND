import json

import numpy as np
import pandas as pd
import pytest

from tide import backtest, crossfit
from tide.backtest import Fold
from tide.candidates import candidate
from tide.events import find_events, match_events, model_events, summarize
from tide.harmonic import HarmonicConfig
from tide.ports import ist_year_start
from tide.stack import FittedStack, StackCandidate, StackConfig

CONFIG = HarmonicConfig(constituents=("M2", "S2", "K1", "O1", "M4"), seasonal=1, trend=False)

def distorted(first: int, last: int, seed: int = 0) -> pd.DataFrame:
    """Synthetic readings whose sharp high waters no harmonic set of A's size can hold."""
    times = pd.date_range(ist_year_start(first), ist_year_start(last + 1), freq="1h", inclusive="left")
    hours = np.arange(len(times), dtype=float)
    phase = 2 * np.pi * hours / 12.4206
    tide = (3.2 + 2.0 * np.cos(phase) + 0.6 * np.cos(2 * np.pi * hours / 12.0 + 1.0)
            + 0.3 * np.cos(2 * np.pi * hours / 23.9345 + 2.0) + 0.4 * np.maximum(np.cos(phase), 0.0) ** 12)
    noise = np.random.default_rng(seed).normal(0.0, 0.03, len(times))
    return pd.DataFrame({"time_utc": times, "height_m": tide + noise, "source": "soi_gauge", "qc_flag": ""})

@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    saved = crossfit.CACHE_DIR
    crossfit.CACHE_DIR = tmp_path_factory.mktemp("crossfit")
    readings = distorted(2010, 2013)
    train = readings[readings["time_utc"] < ist_year_start(2013)]
    yield readings, StackCandidate(StackConfig(CONFIG, level="lgbm-l15-m20", event="lgbm-l15-m20")).fit(train, "haldia")
    crossfit.CACHE_DIR = saved

def test_stack_names_round_trip():
    config = StackConfig(HarmonicConfig(window_years=8), level="lgbm-l31-m100", event="blend(lgbm-l31-m40=0.5,et=0.5)")
    assert config.name == "A-w8-auto-noside-trend/lv:lgbm-l31-m100/ev:blend(lgbm-l31-m40=0.5,et=0.5)"
    assert StackConfig.from_name(config.name) == config
    assert StackConfig.from_name("A-wall-auto-noside-trend") == StackConfig(HarmonicConfig())
    assert candidate(config.name) == StackCandidate(config)
    with pytest.raises(ValueError):
        StackConfig.from_name("A-wall-auto-noside-trend/xx:et")

def test_event_correction_learns_what_model_a_cannot_hold(trained):
    readings, both = trained
    start, end = ist_year_start(2013), ist_year_start(2014)
    test = readings[readings["time_utc"] >= start]
    truth = find_events(test["time_utc"], test["height_m"], step_minutes=60)
    plain = summarize(match_events(truth, model_events(both.model.predict, start, end)))
    corrected = summarize(match_events(truth, both.events(start, end)))
    assert corrected["height_mae_m"] < 0.6 * plain["height_mae_m"]
    assert corrected["joint_pct"] >= plain["joint_pct"]

def test_level_correction_lowers_the_hourly_error(trained):
    readings, both = trained
    test = readings[readings["time_utc"] >= ist_year_start(2013)]
    times, observed = pd.DatetimeIndex(test["time_utc"]), test["height_m"].to_numpy()
    plain = np.sqrt(np.mean((observed - both.model.predict(times)) ** 2))
    assert np.sqrt(np.mean((observed - both.levels(times)) ** 2)) < 0.8 * plain

def test_a_level_is_the_same_at_any_request_step(trained):
    _, both = trained
    start = ist_year_start(2013) + pd.Timedelta(hours=5)
    hourly = pd.date_range(start, periods=48, freq="1h")
    minutes = pd.date_range(start, periods=48 * 60, freq="1min")
    np.testing.assert_allclose(both.levels(minutes)[::60], both.levels(hourly), atol=1e-9)

def test_adjacent_windows_neither_lose_nor_repeat_corrected_events(trained):
    _, both = trained
    a = ist_year_start(2013)
    b, c = a + pd.Timedelta(days=3, hours=7, minutes=13), a + pd.Timedelta(days=7)
    joined = pd.concat([both.events(a, b), both.events(b, c)], ignore_index=True)
    pd.testing.assert_frame_equal(joined, both.events(a, c))

def test_components_add_up_with_the_correction(trained):
    _, both = trained
    times = pd.date_range(ist_year_start(2013), periods=24, freq="1h")
    parts = both.components(times)
    np.testing.assert_allclose(parts["astronomical_m"] + parts["seasonal_m"] + parts["correction_m"], parts["height_m"])
    np.testing.assert_allclose(parts["height_m"], both.levels(times))

def test_a_saved_stack_predicts_identically_and_plan_1a_folders_still_load(trained, tmp_path):
    _, both = trained
    both.save(tmp_path)
    again = FittedStack.load(tmp_path)
    start = ist_year_start(2013)
    times = pd.date_range(start, periods=72, freq="1h")
    np.testing.assert_array_equal(again.levels(times), both.levels(times))
    pd.testing.assert_frame_equal(again.event_detail(start, times[-1]), both.event_detail(start, times[-1]))
    old = tmp_path / "old"
    old.mkdir()
    (old / "a.json").write_text(json.dumps(both.model.to_dict()))
    alone = FittedStack.load(old)
    assert alone.level is None and alone.event is None
    np.testing.assert_array_equal(alone.levels(times), both.model.predict(times))

def test_stack_backtests_repeat_exactly(tmp_path, monkeypatch):
    readings = distorted(2010, 2013)
    monkeypatch.setattr(backtest, "load_gauge", lambda port_slug, passed_only=False: readings)
    stack = StackCandidate(StackConfig(CONFIG, event="lgbm-l15-m20"))
    results = []
    for run_dir in ("one", "two"):
        monkeypatch.setattr(backtest, "CACHE_DIR", tmp_path / run_dir / "backtest")
        monkeypatch.setattr(crossfit, "CACHE_DIR", tmp_path / run_dir / "crossfit")
        results.append(backtest.run("haldia", stack, [Fold(2013, 2013)]))
    pd.testing.assert_frame_equal(results[0][0], results[1][0])
    pd.testing.assert_frame_equal(results[0][1], results[1][1])

import json

import numpy as np
import pandas as pd
import pytest

from tide import selection
from tide.harmonic import HarmonicConfig
from tide.ports import ist_year_start
from tide.stack import StackConfig

A = HarmonicConfig(window_years=8)

def rows(joint: int, rmse: float):
    """Backtest rows whose joint share is `joint` % and hourly RMSE is `rmse` m."""
    times = pd.date_range("2015-01-01", periods=100, freq="12h", tz="UTC")
    tag = {"origin": 2015, "test_year": 2015, "horizon": 1, "final": False}
    events = pd.DataFrame({"kind": "observed", "state": "High", "time_utc": times, "observed_time_utc": times,
                           "hit": np.arange(100) < joint, "time_error_minutes": 10.0, "height_error_m": 0.1, **tag})
    hours = pd.DataFrame({"time_utc": times, "observed_m": 3.0, "predicted_m": 3.0 + rmse, **tag})
    return events, hours

@pytest.fixture
def scored(tmp_path, monkeypatch):
    """Point select_b at fake backtests: set joint shares (by event learner) and RMSEs (by level learner)."""
    table = {"event": {}, "level": {}, "a_joint": 88, "level_joint": 89}

    def fake_run(port_slug, candidate, fold_list, n_jobs=1):
        config = candidate.config
        joint = (table["event"].get(config.event, 90) if config.event
                 else table["level_joint"] if config.level else table["a_joint"])
        return rows(joint, table["level"].get(config.level, 0.19) if config.level else 0.20)

    readings = pd.DataFrame({"time_utc": pd.date_range(ist_year_start(2000), ist_year_start(2020), freq="30D")})
    monkeypatch.setattr(selection, "run", fake_run)
    monkeypatch.setattr(selection, "load_gauge", lambda port_slug, passed_only=False: readings)
    monkeypatch.setattr(selection, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(selection, "selected_path", lambda port_slug: tmp_path / "selected.json")
    return table

def test_the_best_event_and_level_corrections_are_combined(scored, tmp_path):
    scored["event"].update({"lgbm-l31-m40": 93, "et": 92})
    scored["level"].update({"lgbm-l63-m400": 0.15, "mlp": 0.16})
    chosen = selection.select_b("haldia", A)
    assert chosen == StackConfig(A, level="lgbm-l63-m400", event="lgbm-l31-m40")
    saved = json.loads((tmp_path / "selected.json").read_text())
    assert saved["candidate"] == chosen.name and saved["a_candidate"] == A.name and saved["selection_joint_pct"] == 93
    ranking = pd.read_csv(tmp_path / "backtest" / "haldia" / "selection_b.csv")
    assert "A-w8-auto-noside-trend/ev:blend(lgbm-l31-m40=0.25,et=0.75)" in set(ranking["candidate"])

def test_a_blend_wins_when_it_scores_best(scored):
    scored["event"].update({"lgbm-l31-m40": 93, "et": 92, "blend(lgbm-l31-m40=0.5,et=0.5)": 94})
    assert selection.select_b("haldia", A).event == "blend(lgbm-l31-m40=0.5,et=0.5)"

def test_corrections_that_do_not_help_are_left_out(scored):
    scored["event"].update(dict.fromkeys(selection.EVENT_LEARNERS, 85))
    scored["event"].update({"blend(lgbm-l15-m20=0.25,lgbm-l31-m40=0.75)": 85, "blend(lgbm-l15-m20=0.5,lgbm-l31-m40=0.5)": 85,
                            "blend(lgbm-l15-m20=0.75,lgbm-l31-m40=0.25)": 85})
    scored["level_joint"] = 89
    assert selection.select_b("haldia", A) == StackConfig(A, level="lgbm-l31-m100")   # events from the level curve
    scored["level_joint"] = 87
    assert selection.select_b("haldia", A) == StackConfig(A)                           # joint share comes first

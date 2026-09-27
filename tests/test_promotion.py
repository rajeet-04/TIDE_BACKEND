import pandas as pd
import pytest

from tide.backtest import promotion
from tide.ports import IST

def fake(hits_by_month: dict[str, list[bool]], horizon: int = 1) -> pd.DataFrame:
    rows = []
    for month, hits in hits_by_month.items():
        first = pd.Timestamp(f"{month}-01 06:00", tz=IST).tz_convert("UTC")
        for k, hit in enumerate(hits):
            t = first + pd.Timedelta(hours=12 * k)
            rows.append({"kind": "observed", "origin": 2020, "test_year": int(month[:4]), "horizon": horizon,
                         "final": True, "state": "High", "observed_time_utc": t, "time_utc": t, "hit": hit,
                         "time_error_minutes": 10.0 if hit else 50.0, "height_error_m": 0.1 if hit else 0.4})
    return pd.DataFrame(rows)

def year_of(pattern) -> dict[str, list[bool]]:
    return {f"2021-{m:02d}": pattern(m) for m in range(1, 13)}

def test_identical_candidates_are_not_promoted():
    events = fake(year_of(lambda m: [True] * 15 + [False] * 5))
    result = promotion(events, events)
    assert result["joint"]["diff"] == 0.0 and not result["checks"]["joint_better"]
    assert result["checks"]["time_mae_not_worse"] and result["checks"]["no_season_worse_than_1pp"]
    assert not result["passed"]

def test_a_consistently_better_candidate_passes_every_check():
    base = fake(year_of(lambda m: [True] * 12 + [False] * 8))
    cand = fake(year_of(lambda m: [True] * 16 + [False] * 4))
    result = promotion(base, cand)
    assert result["joint"]["diff"] == pytest.approx(20.0) and result["joint"]["lo"] > 0
    assert result["passed"]

def test_a_worse_monsoon_blocks_promotion():
    base = fake(year_of(lambda m: [True] * 16 + [False] * 4))
    cand = fake(year_of(lambda m: [True] * 14 + [False] * 6 if 6 <= m <= 9 else [True] * 20))
    result = promotion(base, cand)
    assert result["season_diff_pp"]["monsoon"] == pytest.approx(-10.0)
    assert not result["checks"]["no_season_worse_than_1pp"] and not result["passed"]

def test_publishable_horizons_are_within_two_points_of_one_year_ahead():
    cand = pd.concat([fake({"2021-01": [True] * 18 + [False] * 2}, horizon=1),
                      fake({"2021-01": [True] * 18 + [False] * 2}, horizon=2),
                      fake({"2021-01": [True] * 17 + [False] * 3}, horizon=3)], ignore_index=True)
    assert promotion(cand, cand)["publishable_horizons"] == [1, 2]

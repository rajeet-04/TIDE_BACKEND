import pandas as pd
import pytest

from tide.backtest import promotion
from tide.ports import IST
from tide.ranges import OUTPUTS

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

def in_band(pct: float = 90.0, horizons=(1, 2, 3), **exceptions) -> pd.DataFrame:
    """Coverage rows as ranges.coverage returns them; exceptions map 'output@horizon' to a share."""
    rows = [{"output": o, "horizon": h, "covered_pct": exceptions.get(f"{o}@{h}", pct), "n": 100}
            for o in OUTPUTS for h in ("all", *horizons)]
    return pd.DataFrame(rows).astype({"horizon": object})

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
    result = promotion(base, cand, coverage=in_band())
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
    assert promotion(cand, cand, coverage=in_band())["publishable_horizons"] == [1, 2]

def test_coverage_outside_the_band_blocks_promotion_and_publication():
    base = fake(year_of(lambda m: [True] * 12 + [False] * 8))
    cand = pd.concat([fake(year_of(lambda m: [True] * 16 + [False] * 4), horizon=h) for h in (1, 2)], ignore_index=True)
    assert not promotion(base, cand)["checks"]["coverage_88_92"]                      # no ranges at all
    low = promotion(base, cand, coverage=in_band(**{"high_time@all": 85.0, "high_time@2": 84.0}))
    assert not low["checks"]["coverage_88_92"] and not low["passed"]
    assert low["publishable_horizons"] == [1]

def test_official_tables_must_be_beaten_where_they_overlap():
    base = fake(year_of(lambda m: [True] * 12 + [False] * 8))
    cand = fake(year_of(lambda m: [True] * 16 + [False] * 4))
    tables = fake(year_of(lambda m: [True] * 18 + [False] * 2)).assign(horizon=0)
    result = promotion(base, cand, coverage=in_band(), tables=tables)
    assert result["versus_tables_pp"] == {2021: pytest.approx(-10.0)}
    assert not result["checks"]["beats_tables"] and not result["passed"]
    assert promotion(base, cand, coverage=in_band(), tables=tables.iloc[:0])["checks"]["beats_tables"]

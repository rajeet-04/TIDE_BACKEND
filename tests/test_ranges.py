import numpy as np
import pandas as pd

from tide.ports import IST
from tide.ranges import OUTPUTS, bounds, calibrate, coverage

def backtest_rows(per_month: int, seed: int, miss_every: int = 0, horizon: int = 1):
    """Backtest-shaped event and hour rows with known error distributions."""
    rng = np.random.default_rng(seed)
    events, hours = [], []
    for year in (2010, 2011):
        for month in range(1, 13):
            first = pd.Timestamp(f"{year}-{month:02d}-01 06:00", tz=IST).tz_convert("UTC")
            for k in range(per_month):
                t = first + pd.Timedelta(hours=12.42 * k)
                missed = bool(miss_every) and k % miss_every == 0
                events.append({"kind": "observed", "state": "High" if k % 2 else "Low", "time_utc": t,
                               "horizon": horizon,
                               "time_error_minutes": np.nan if missed else rng.normal(3.0, 10.0),
                               "height_error_m": np.nan if missed else rng.normal(-0.05, 0.1)})
                hours.append({"time_utc": t, "horizon": horizon, "observed_m": 3.0 + rng.normal(0.0, 0.2),
                              "predicted_m": 3.0})
    return pd.DataFrame(events), pd.DataFrame(hours)

def test_ranges_hold_ninety_percent_of_fresh_errors():
    table = calibrate(*backtest_rows(50, seed=0))
    assert len(table) == len(OUTPUTS) * 3 * 4
    high_time = table[(table["output"] == "high_time") & (table["horizon"] == 1)]
    assert (high_time["upper"] + high_time["lower"]).abs().max() < 2 * 3.0 + 6.0   # centred near -3 min
    fresh = coverage(*backtest_rows(50, seed=1), table)
    pooled = fresh[fresh["horizon"] == "all"].set_index("output")["covered_pct"]
    assert set(pooled.index) == set(OUTPUTS) and pooled.between(86.0, 94.0).all()
    seasons = coverage(*backtest_rows(50, seed=1), table, by="season")
    assert set(seasons["season"]) == {"all", "dry", "pre_monsoon", "monsoon", "post_monsoon"}

def test_small_cells_borrow_from_the_horizon_then_the_output():
    table = calibrate(*backtest_rows(6, seed=0))
    sources = table.groupby("output")["source"].agg(set)
    assert sources["high_time"] == {"output"}            # 72 high waters in all
    assert sources["level"] == {"horizon", "output"}     # 144 hours at horizon 1; none at 2 and 3

def test_missed_events_count_as_outside():
    table = calibrate(*backtest_rows(50, seed=0))
    missing = coverage(*backtest_rows(50, seed=1, miss_every=5), table)
    pooled = missing[missing["horizon"] == "all"].set_index("output")["covered_pct"]
    assert pooled["high_time"] < 76.0 and pooled["level"] > 86.0

def test_bounds_clip_horizons_and_follow_seasons():
    table = calibrate(*backtest_rows(50, seed=0))
    lower, upper = bounds(table, "level", [0, 1, 7], ["dry", "monsoon", "dry"])
    cell = table[table["output"] == "level"].set_index(["horizon", "season"])
    assert lower[0] == cell.loc[(1, "dry"), "lower"] and upper[1] == cell.loc[(1, "monsoon"), "upper"]
    assert lower[2] == cell.loc[(3, "dry"), "lower"]

import numpy as np
import pandas as pd
import pytest

from tide import crossfit
from tide.crossfit import cross_fit
from tide.features import EVENT_FEATURES, LEVEL_FEATURES
from tide.harmonic import HarmonicConfig
from tide.ports import IST, ist_year_start, ist_years

CONFIG = HarmonicConfig(constituents=("M2", "S2", "K1", "O1", "M4"), seasonal=1, trend=False)

def gauge(first: int, last: int, seed: int = 0) -> pd.DataFrame:
    """Synthetic hourly readings on whole IST hours for IST years first to last."""
    times = pd.date_range(ist_year_start(first), ist_year_start(last + 1), freq="1h", inclusive="left")
    hours = np.arange(len(times), dtype=float)
    tide = (3.2 + 2.0 * np.cos(2 * np.pi * hours / 12.4206) + 0.6 * np.cos(2 * np.pi * hours / 12.0 + 1.0)
            + 0.3 * np.cos(2 * np.pi * hours / 23.9345 + 2.0) + 0.25 * np.cos(2 * np.pi * hours / 6.2103 + 0.5))
    return pd.DataFrame({"time_utc": times, "height_m": tide + np.random.default_rng(seed).normal(0.0, 0.03, len(times))})

@pytest.fixture(autouse=True)
def private_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(crossfit, "CACHE_DIR", tmp_path / "crossfit")

def test_each_year_is_predicted_by_a_fit_that_never_saw_it():
    readings = gauge(2010, 2013)
    readings.loc[ist_years(readings["time_utc"]) == 2012, "height_m"] += 1.0   # a re-levelled year
    table = cross_fit(readings, "haldia", HarmonicConfig(constituents=CONFIG.constituents, seasonal=1, trend=False,
                                                          robust=False)).level_table()
    # an in-sample fit would absorb a quarter of the step; the fit without 2012 absorbs none
    assert table.loc[table["year"] == 2012, "target"].mean() > 0.95

def test_missing_years_add_no_rows_and_sparse_years_only_their_readings():
    readings = gauge(2010, 2014)
    years = ist_years(readings["time_utc"])
    march = (readings["time_utc"].dt.tz_convert(IST).dt.month == 3).to_numpy()
    keep = (years != 2012) & ((years != 2013) | march)
    material = cross_fit(readings[keep], "haldia", CONFIG)
    level, events = material.level_table(), material.event_table()
    assert set(material.curves["year"]) == {2010, 2011, 2013, 2014}
    assert 2012 not in set(level["year"]) and 2012 not in set(events["year"])
    assert (level["year"] == 2013).sum() == ((years == 2013) & march).sum()
    assert level[LEVEL_FEATURES + ["target"]].notna().all().all()
    assert events[EVENT_FEATURES + ["time_error_minutes", "height_error_m"]].notna().all().all()
    assert 100 < (events["year"] == 2013).sum() < 125               # March's tides only

def test_results_are_cached_and_identical(tmp_path):
    readings = gauge(2010, 2012)
    first = cross_fit(readings, "haldia", CONFIG)
    assert len(list((tmp_path / "crossfit" / "haldia").glob("*.pkl"))) == 1
    second = cross_fit(readings, "haldia", CONFIG)
    pd.testing.assert_frame_equal(first.level_table(), second.level_table())
    pd.testing.assert_frame_equal(first.event_table(), second.event_table())

def test_one_year_of_readings_cannot_be_cross_fitted():
    with pytest.raises(ValueError, match="two years"):
        cross_fit(gauge(2010, 2010), "haldia", CONFIG)

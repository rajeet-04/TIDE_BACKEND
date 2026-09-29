import pandas as pd
import pytest

from tide.candidates import CurrentPipeline, HarmonicCandidate, candidate, window
from tide.harmonic import HarmonicConfig
from tide.ports import ist_year_start
from tide.store import load_gauge

def test_window_keeps_the_most_recent_years():
    times = pd.date_range(ist_year_start(2010), ist_year_start(2020), freq="1D", inclusive="left")
    train = pd.DataFrame({"time_utc": times, "height_m": 1.0})
    recent = window(train, 5)
    assert recent["time_utc"].min() > ist_year_start(2014) and recent["time_utc"].max() == times[-1]
    assert window(train, None) is train

def test_candidates_are_found_by_name():
    assert candidate("A-w8-auto-side-trend") == HarmonicCandidate(HarmonicConfig(window_years=8, side_terms=True))
    assert candidate("current_pipeline").name == "current_pipeline"
    assert candidate("utide_only").name == "utide_only"
    with pytest.raises(ValueError):
        candidate("B-lightgbm")

@pytest.mark.slow
def test_current_pipeline_refits_and_returns_utc_events():
    gauge = load_gauge("haldia", passed_only=True)
    train = gauge[(gauge["time_utc"] >= ist_year_start(2021)) & (gauge["time_utc"] < ist_year_start(2023))]
    fitted = CurrentPipeline().fit(train, "haldia")
    start = ist_year_start(2023)
    end = start + pd.Timedelta(days=7)
    events = fitted.events(start, end)
    assert 24 <= len(events) <= 30 and str(events["time_utc"].dt.tz) == "UTC"
    assert events["time_utc"].between(start, end, inclusive="left").all()
    assert len(fitted.levels(pd.date_range(start, periods=24, freq="1h"))) == 24

@pytest.mark.slow
def test_model_a_candidate_fits_on_its_window():
    gauge = load_gauge("haldia", passed_only=True)
    fitted = HarmonicCandidate(HarmonicConfig(window_years=2)).fit(gauge[gauge["time_utc"] < ist_year_start(2023)], "haldia")
    start = ist_year_start(2023)
    assert 24 <= len(fitted.events(start, start + pd.Timedelta(days=7))) <= 30

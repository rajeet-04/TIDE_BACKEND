import numpy as np
import pandas as pd

from tide.events import coverage, extra_events, find_events, match_events, model_events, summarize

START = pd.Timestamp("2024-01-01", tz="UTC")

def at(hours: float) -> pd.Timestamp:
    return START + pd.Timedelta(hours=hours)

def cosine(times, period_h: float = 12.0, amp: float = 2.0, mean: float = 3.0) -> np.ndarray:
    hours = np.asarray((pd.DatetimeIndex(times) - START) / pd.Timedelta(hours=1))
    return mean + amp * np.cos(2 * np.pi * hours / period_h)

def test_minute_curve_events_fall_on_the_true_extremes():
    times = pd.date_range(START, periods=48 * 60, freq="1min")
    events = find_events(times, cosine(times), step_minutes=1)
    assert events.loc[events.state == "High", "time_utc"].tolist() == [at(12), at(24), at(36)]
    assert events.loc[events.state == "Low", "time_utc"].tolist() == [at(6), at(18), at(30), at(42)]

def test_low_water_stand_is_timed_at_its_centre():
    times = pd.date_range(START, periods=13 * 60, freq="1min")
    hours = np.asarray((times - START) / pd.Timedelta(hours=1))
    level = np.interp(hours, [0, 3, 4, 5, 7, 10, 13], [1.0, 3.0, 2.4, 0.600, 0.605, 3.0, 1.0])
    low = find_events(times, level, step_minutes=1).query("state == 'Low'").iloc[0]
    assert low.time_utc == at(6)          # the minimum itself is at 05:00
    assert low.height_m == 0.600

def test_hourly_observations_use_quadratic_refinement():
    times = pd.date_range(START, periods=72, freq="1h")
    first_high = find_events(times, cosine(times, period_h=12.42), step_minutes=60).query("state == 'High'").iloc[0]
    assert abs((first_high.time_utc - START) / pd.Timedelta(minutes=1) - 12.42 * 60) < 5
    assert abs(first_high.height_m - 5.0) < 0.02

def test_events_are_not_invented_across_gaps():
    times = pd.date_range(START, periods=72, freq="1h")
    heights = cosine(times, period_h=12.42)
    keep = (times < at(20)) | (times > at(28))
    events = find_events(times[keep], heights[keep], step_minutes=60)
    assert not events.time_utc.between(at(20), at(28)).any()
    assert (events.state == "High").sum() == 4      # 12.42 h, 37.26 h, 49.68 h, 62.1 h; 24.84 h is in the gap

def test_model_events_find_edge_tides_and_stay_inside_the_window():
    events = model_events(cosine, at(12), at(36))
    assert events.time_utc.tolist() == [at(12), at(18), at(24), at(30)]

def test_matching_counts_hits_misses_and_extras():
    observed = pd.DataFrame({"state": ["High", "Low", "High"], "time_utc": [at(0), at(6), at(12.5)],
                             "height_m": [3.0, 1.0, 3.2]})
    predicted = pd.DataFrame({"state": ["High", "Low", "High"], "time_utc": [at(0) + pd.Timedelta(minutes=20), at(6.75), at(20)],
                              "height_m": [3.1, 1.0, 3.0]})
    matches = match_events(observed, predicted)
    assert matches["hit"].tolist() == [True, False, False]   # 20 min fine; 45 min late; nothing within 3 h
    assert matches["time_error_minutes"].tolist()[:2] == [20.0, 45.0]
    extras = extra_events(predicted, matches, [(at(0), at(24))])
    assert extras["time_utc"].tolist() == [at(20)]
    scores = summarize(matches, extra=len(extras))
    assert (scores["observed"], scores["missed"], scores["extra"]) == (3, 1, 1)
    assert round(scores["joint_pct"], 1) == 33.3

def test_coverage_spans_shrink_at_gaps():
    times = pd.date_range(START, periods=10, freq="1h").append(pd.date_range(at(20), periods=10, freq="1h"))
    assert coverage(times) == [(at(2), at(7)), (at(22), at(27))]

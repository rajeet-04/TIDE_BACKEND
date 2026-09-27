import numpy as np
import pandas as pd

from tide.ports import ist_year_start
from tide.qc import qc_report, review_list, run_qc

NO_REVIEW = pd.DataFrame(columns=["start", "end", "action", "reason", "evidence"])

def synthetic_gauge(years: int = 3) -> pd.DataFrame:
    times = pd.date_range(ist_year_start(2010), periods=years * 8766, freq="1h")
    hours = np.arange(len(times), dtype=float)
    tide = (3.2 + 2.0 * np.cos(2 * np.pi * hours / 12.4206) + 0.6 * np.cos(2 * np.pi * hours / 12.0 + 1.0)
            + 0.3 * np.cos(2 * np.pi * hours / 23.9345 + 2.0) + 0.25 * np.cos(2 * np.pi * hours / 6.2103 + 0.5))
    noise = np.random.default_rng(0).normal(0.0, 0.03, len(times))
    return pd.DataFrame({"time_utc": times, "height_m": np.round(tide + noise, 2), "source": "soi_gauge", "qc_flag": ""})

def flagged_with(table: pd.DataFrame, rule: str) -> set[int]:
    return set(np.flatnonzero(table["qc_flag"].str.contains(rule, regex=False)))

def test_rules_flag_injected_faults_but_not_a_smooth_surge():
    gauge = synthetic_gauge()
    h = gauge["height_m"].to_numpy().copy()
    h[10_000] += 1.5                                         # single-hour spike
    h[15_000:15_006] = 0.0                                   # six hours of zeros
    h[20_000:20_005] = h[20_000]                             # stuck gauge
    k = np.arange(-72, 73)
    h[25_000 + k] += 1.5 * np.exp(-0.5 * (k / 12.0) ** 2)    # storm surge: real, must pass
    gauge["height_m"] = h
    table = run_qc(gauge, "haldia", review=NO_REVIEW)
    assert 10_000 in flagged_with(table, "spike")
    assert set(range(15_000, 15_006)) <= flagged_with(table, "range")
    assert set(range(20_000, 20_005)) <= flagged_with(table, "flat")
    assert table["qc_flag"].iloc[25_000 - 72:25_000 + 73].eq("").all()
    assert table.loc[[9_999, 10_001, 14_999, 15_006], "qc_flag"].eq("").all()
    assert (table["qc_flag"] != "").mean() < 0.001

def test_review_rows_exclude_and_keep():
    gauge = synthetic_gauge(years=2)
    h = gauge["height_m"].to_numpy().copy()
    h[5_000] += 1.5
    gauge["height_m"] = h
    t = gauge["time_utc"]
    review = pd.DataFrame({"start": [t[100].isoformat(), t[5_000].isoformat()],
                           "end": [t[110].isoformat(), t[5_000].isoformat()],
                           "action": ["exclude", "keep"], "reason": ["audit", "real surge"], "evidence": ["test", "test"]})
    table = run_qc(gauge, "haldia", review=review)
    assert flagged_with(table, "manual") == set(range(100, 111))
    assert table["qc_flag"].iloc[5_000] == ""

def test_report_and_review_list_summarise_flags():
    times = pd.date_range(ist_year_start(2011), periods=8, freq="1h")
    table = pd.DataFrame({"time_utc": times, "height_m": [1.0, 0.0, 0.0, 2.0, 9.9, 2.0, 2.1, 2.2],
                          "qc_flag": ["", "range|flat", "range|flat", "", "spike", "", "", ""],
                          "qc_residual_m": [0.4, -2.0, -2.1, 0.5, 6.0, 0.6, 0.5, 0.4]})
    report = qc_report(table)
    assert (report.loc[0, "readings"], report.loc[0, "flagged"], report.loc[0, "range"]) == (8, 3, 2)
    stretches = review_list(table)
    assert stretches["hours"].tolist() == [2, 1]
    assert stretches["rules"].tolist() == ["flat|range", "spike"]
    assert stretches["context_residual_m"].tolist() == [0.5, 0.5]

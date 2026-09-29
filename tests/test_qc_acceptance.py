"""Acceptance checks on the committed QC flags (spec 3.3)."""
import pytest

from tide.ports import IST, to_utc
from tide.qc import qc_report
from tide.store import flags_path, load_gauge

KNOWN_BAD = [  # IST times from the residual scan recorded in spec section 2
    ("haldia", "2021-12-05 18:00"), ("haldia", "2021-12-05 19:00"), ("haldia", "2021-12-05 20:00"),
    ("haldia", "2007-08-25 10:00"), ("diamond_harbour", "2007-04-08 17:00"),
]
# Years allowed above 1% flagged, each with a decision-log entry explaining why.
EXPLAINED_YEARS: dict[str, set[int]] = {"haldia": set(), "diamond_harbour": set()}

@pytest.mark.parametrize("port_slug", ["haldia", "diamond_harbour"])
def test_flags_are_committed(port_slug):
    assert flags_path(port_slug).exists()

@pytest.mark.parametrize("port_slug, when", KNOWN_BAD)
def test_known_bad_readings_are_flagged(port_slug, when):
    gauge = load_gauge(port_slug)
    row = gauge[gauge["time_utc"] == to_utc(when)]
    assert len(row) == 1 and row["qc_flag"].iloc[0] != ""

def test_haldia_zero_run_of_2006_04_26_is_flagged():
    gauge = load_gauge("haldia")
    day = gauge["time_utc"].dt.tz_convert(IST).dt.strftime("%Y-%m-%d") == "2006-04-26"
    zeros = gauge[day & (gauge["height_m"] == 0.0)]
    assert len(zeros) >= 1 and zeros["qc_flag"].ne("").all()

@pytest.mark.parametrize("port_slug", ["haldia", "diamond_harbour"])
def test_at_most_one_percent_flagged_per_year_unless_explained(port_slug):
    report = qc_report(load_gauge(port_slug))
    assert set(report.loc[report["flagged_pct"] > 1.0, "year"]) <= EXPLAINED_YEARS[port_slug]

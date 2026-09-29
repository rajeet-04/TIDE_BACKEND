import pandas as pd
import pytest

from tide.ports import datum_label, ist_year_start, ist_years, port, season_of, to_utc, trainable

def test_naive_times_are_read_as_ist():
    assert to_utc("2026-10-01 05:30") == pd.Timestamp("2026-10-01 00:00", tz="UTC")
    assert to_utc("2026-10-01") == pd.Timestamp("2026-09-30 18:30", tz="UTC")

def test_times_with_a_zone_keep_it():
    assert to_utc("2026-10-01T00:00Z") == pd.Timestamp("2026-10-01 00:00", tz="UTC")
    assert to_utc("2026-10-01T00:00+05:30") == pd.Timestamp("2026-09-30 18:30", tz="UTC")

def test_years_and_seasons_follow_the_ist_calendar():
    assert ist_year_start(2024) == pd.Timestamp("2023-12-31 18:30", tz="UTC")
    times = pd.DatetimeIndex(["2023-12-31 18:30", "2024-03-31 19:00", "2024-06-01 00:00"], tz="UTC")
    assert list(ist_years(times)) == [2024, 2024, 2024]
    assert list(season_of(times)) == ["dry", "pre_monsoon", "monsoon"]

def test_datums_are_labelled_and_unknown_ones_are_not_trainable():
    assert datum_label("haldia") == "chart datum (Survey of India)"
    assert trainable("diamond_harbour", "soi_gauge")
    assert not trainable("garden_reach", "incois")

def test_unknown_port_is_a_clear_error():
    with pytest.raises(ValueError, match="unknown port"):
        port("sagar")

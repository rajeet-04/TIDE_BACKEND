import pandas as pd

from tide import store

def test_gauge_csv_becomes_sorted_unique_utc(tmp_path):
    path = tmp_path / "x.csv"
    path.write_text(
        "port,datetime_ist,height_m\n"
        "X,2024-01-01T01:00:00,2.0\n"
        "X,2024-01-01T00:00:00,1.0\n"
        "X,2024-01-01T01:00:00,2.5\n"
    )
    table = store.read_gauge_csv(path)
    assert list(table.columns) == store.COLUMNS
    assert table["time_utc"].tolist() == [pd.Timestamp("2023-12-31 18:30", tz="UTC"),
                                          pd.Timestamp("2023-12-31 19:30", tz="UTC")]
    assert table["height_m"].tolist() == [1.0, 2.5]
    assert (table["qc_flag"] == "").all()

def test_haldia_record_starts_at_ist_midnight_2000():
    table = store.load_gauge("haldia")
    assert table["time_utc"].iloc[0] == pd.Timestamp("1999-12-31 18:30", tz="UTC")
    assert table["time_utc"].is_monotonic_increasing and table["time_utc"].is_unique

def test_committed_flags_mark_readings_and_can_be_dropped(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "QC_DIR", tmp_path)
    first = store.load_gauge("haldia")["time_utc"].iloc[0]
    pd.DataFrame({"time_utc": [first], "height_m": [2.22], "qc_flag": ["spike"]}).to_csv(
        tmp_path / "haldia_flags.csv", index=False)
    assert store.load_gauge("haldia")["qc_flag"].iloc[0] == "spike"
    assert store.load_gauge("haldia", passed_only=True)["time_utc"].iloc[0] > first

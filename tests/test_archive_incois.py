"""Checks for src/archive_incois.py. Run: python tests/test_archive_incois.py (pytest also works)."""
import sys
import ssl
import tempfile
from unittest.mock import MagicMock, patch
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import archive_incois as ai

MS_2026_09_26_0827 = -58167819180000  # first sample of the captured GARDENREACH_1.json


def test_tls_context_keeps_verification_and_loads_missing_intermediate():
    context = ai.tls_context()
    assert context.check_hostname
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert any(
        ("commonName", "GlobalSign RSA OV SSL CA 2018") in group
        for cert in context.get_ca_certs() for group in cert["subject"]
    )


def test_fetch_uses_verified_context_and_retries_network_errors():
    response = MagicMock()
    response.__enter__.return_value.read.return_value = b"readings"
    with patch.object(ai.urllib.request, "urlopen", side_effect=[OSError("temporary"), response]) as urlopen, patch.object(ai.time, "sleep") as sleep:
        assert ai.fetch(f"{ai.BASE}/homexmls/TideStations.xml") == b"readings"
        assert urlopen.call_count == 2
        sleep.assert_called_once_with(10)
        context = urlopen.call_args.kwargs["context"]
        assert context.check_hostname and context.verify_mode == ssl.CERT_REQUIRED


def test_decode_time_fixes_incois_year_offset():
    assert ai.decode_time(MS_2026_09_26_0827) == datetime(2026, 9, 26, 8, 27)


def test_rows_skip_nulls_and_tag_predicted_residual_with_their_sensor():
    series = [
        {"name": "RAD", "data": [[MS_2026_09_26_0827, 4.6], [MS_2026_09_26_0827 + 60000, None]]},
        {"name": "Predicted", "data": [[MS_2026_09_26_0827, None]]},
        {"name": "Residual", "data": [[MS_2026_09_26_0827, 0.1]]},
        {"name": "PRS", "data": [[MS_2026_09_26_0827, 20.0]]},
    ]
    assert sorted(ai.rows_from_series(series)) == [
        ("2026-09-26 08:27", "PRS", "20.0"),
        ("2026-09-26 08:27", "RAD", "4.6"),
        ("2026-09-26 08:27", "RAD:Residual", "0.1"),
    ]


def test_merge_splits_by_utc_day_dedupes_and_reports_only_changed_files():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        first = [("2026-09-26 23:59", "RAD", "4.1"), ("2026-09-27 00:00", "RAD", "4.2")]
        changed = ai.merge_rows(out, "gard", first)
        day1, day2 = out / "gard/2026/2026-09-26.csv", out / "gard/2026/2026-09-27.csv"
        assert sorted(changed) == [day1, day2]
        assert day2.read_text() == "time_utc,sensor,value\n2026-09-27 00:00,RAD,4.2\n"

        # overlapping fetch: one repeated row, one new row -> only day 2 changes, no duplicates
        changed = ai.merge_rows(out, "gard", [("2026-09-27 00:00", "RAD", "4.2"), ("2026-09-27 00:01", "RAD", "4.3")])
        assert changed == [day2]
        assert day2.read_text().count("2026-09-27 00:00") == 1
        assert ai.merge_rows(out, "gard", first) == []
        assert ai.latest_time(out, "gard") == datetime(2026, 9, 27, 0, 1)


def test_backfill_needed_when_archive_empty_or_window_leaves_a_gap():
    window_start = datetime(2026, 9, 27, 8, 0)
    assert ai.needs_backfill(None, window_start)
    assert not ai.needs_backfill(datetime(2026, 9, 27, 9, 0), window_start)   # overlap
    assert ai.needs_backfill(datetime(2026, 9, 27, 7, 0), window_start)       # 1 h hole


def test_parse_stations_reads_code_name_position_and_json_name():
    xml = ('<?xml version="1.0" encoding="UTF-8" standalone="no"?><stations><station status="Reporting">'
           '<latitude>22.55</latitude><longitude>88.3</longitude><html>&lt;b&gt;x&lt;/b&gt;</html>'
           '<date>2026-Sep-27 08:43</date><statname>gard</statname><country>India</country><owner>INCOIS</owner>'
           '<colorClass>GREEN</colorClass><statrealName>Gardenreach</statrealName></station></stations>')
    assert ai.parse_stations(xml) == [{
        "code": "gard", "name": "Gardenreach", "json_name": "GARDENREACH", "latitude": "22.55",
        "longitude": "88.3", "status": "Reporting", "last_report": "2026-Sep-27 08:43", "owner": "INCOIS",
    }]


if __name__ == "__main__":
    tests = [f for name, f in dict(globals()).items() if name.startswith("test_")]
    for t in tests:
        t()
    print(f"{len(tests)} checks passed")

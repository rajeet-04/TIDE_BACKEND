"""Archive INCOIS real-time tide gauge readings for every station into daily CSV files.

INCOIS only serves rolling windows (<NAME>_1.json = last 24 h, <NAME>_30.json = last 30 days),
so this runs on a schedule and merges into <out>/<code>/<YYYY>/<YYYY-MM-DD>.csv with columns
time_utc,sensor,value. Values are raw (no QC: sentinels such as 20.00 are kept). The 24 h file is
used normally; the 30-day file backfills a station whose archive has a gap. Stdlib only.

Usage:
  python src/archive_incois.py --out incois
"""
from __future__ import annotations

import argparse
import csv
import json
import ssl
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path

BASE = "https://tsunami.incois.gov.in/itews"
USER_AGENT = "tide-backend-archiver/1.0 (daily tide gauge archive)"
HEADER = "time_utc,sensor,value\n"
TIME_FORMAT = "%Y-%m-%d %H:%M"
GAP_TOLERANCE = timedelta(minutes=5)
EPOCH = datetime(1970, 1, 1)
STATION_FIELDS = ["code", "name", "json_name", "latitude", "longitude", "status", "last_report", "owner"]


def tls_context() -> ssl.SSLContext:
    """Complete INCOIS's missing intermediate chain without disabling TLS verification."""
    context = ssl.create_default_context()
    context.load_verify_locations(cafile=str(Path(__file__).with_name("certs") / "globalsign-rsa-ov-ssl-ca-2018.pem"))
    return context


def decode_time(ms: int) -> datetime:
    """INCOIS builds timestamps with JS Date.UTC(year - 1900, ...), so year 126 means 2026."""
    t = EPOCH + timedelta(milliseconds=ms)
    return t.replace(year=t.year + 1900)


def rows_from_series(series: list[dict]) -> list[tuple[str, str, str]]:
    """Flatten chart series to (time_utc, sensor, value); Predicted/Residual follow their sensor."""
    rows, sensor = [], ""
    for s in series:
        if s["name"] in ("Predicted", "Residual"):
            label = f"{sensor}:{s['name']}"
        else:
            sensor = label = s["name"]
        rows += [(decode_time(ms).strftime(TIME_FORMAT), label, str(v)) for ms, v in s["data"] if v is not None]
    return rows


def merge_rows(out: Path, code: str, rows: list[tuple[str, str, str]]) -> list[Path]:
    """Merge rows into per-day files (newest value wins); return the files whose content changed."""
    by_day: dict[str, dict[tuple[str, str], str]] = {}
    for t, sensor, value in rows:
        by_day.setdefault(t[:10], {})[(t, sensor)] = value
    changed = []
    for day, new in sorted(by_day.items()):
        path = out / code / day[:4] / f"{day}.csv"
        old = path.read_text() if path.exists() else HEADER
        merged = {(t, s): v for t, s, v in csv.reader(old.splitlines()[1:])}
        merged.update(new)
        text = HEADER + "".join(f"{t},{s},{v}\n" for (t, s), v in sorted(merged.items()))
        if text != old:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, newline="\n")
            changed.append(path)
    return changed


def latest_time(out: Path, code: str) -> datetime | None:
    files = sorted((out / code).glob("*/*.csv"))
    if not files:
        return None
    return datetime.strptime(files[-1].read_text().splitlines()[-1].split(",")[0], TIME_FORMAT)


def needs_backfill(latest: datetime | None, window_start: datetime) -> bool:
    return latest is None or latest < window_start - GAP_TOLERANCE


def parse_stations(xml_text: str | bytes) -> list[dict]:
    root = ET.fromstring(xml_text.encode("utf-8") if isinstance(xml_text, str) else xml_text)
    stations = []
    for st in root.iter("station"):
        name = st.findtext("statrealName", "").strip()
        stations.append({
            "code": st.findtext("statname", "").strip(), "name": name, "json_name": name.upper().replace(" ", ""),
            "latitude": st.findtext("latitude", ""), "longitude": st.findtext("longitude", ""),
            "status": st.get("status", ""), "last_report": st.findtext("date", ""), "owner": st.findtext("owner", ""),
        })
    return stations


def fetch(url: str, retries: int = 3) -> bytes:
    context = tls_context()
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT}), timeout=60, context=context) as r:
                return r.read()
        except OSError:  # URLError, HTTPError and timeouts
            if attempt == retries - 1:
                raise
            time.sleep(10 * (attempt + 1))


def archive(out: Path, pause: float) -> tuple[int, int]:
    """Archive every station once; return (failed stations, total stations)."""
    stations = parse_stations(fetch(f"{BASE}/homexmls/TideStations.xml"))
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "stations.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, STATION_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(stations)
    failures = 0
    for st in stations:
        try:
            rows = rows_from_series(json.loads(fetch(f"{BASE}/JSONS/{st['json_name']}_1.json") or b"[]"))
            backfilled = False
            if rows and needs_backfill(latest_time(out, st["code"]), datetime.strptime(min(r[0] for r in rows), TIME_FORMAT)):
                rows = rows_from_series(json.loads(fetch(f"{BASE}/JSONS/{st['json_name']}_30.json"))) + rows
                backfilled = True
            changed = merge_rows(out, st["code"], rows)
            print(f"{st['code']:7} {len(rows):6} rows, {len(changed)} day files changed{' (30-day backfill)' if backfilled else ''}")
        except Exception as e:  # one broken station must not stop the others
            failures += 1
            print(f"{st['code']:7} FAILED: {e}", file=sys.stderr)
        time.sleep(pause)
    print(f"{len(stations)} stations, {failures} failed")
    return failures, len(stations)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("incois"))
    parser.add_argument("--pause", type=float, default=1.0, help="seconds between requests")
    args = parser.parse_args()
    failures, total = archive(args.out, args.pause)
    sys.exit(1 if failures == total else 0)  # all failed (or no stations) = INCOIS down or format changed


if __name__ == "__main__":
    main()

"""Phase 1: Extract hourly tidal heights from Survey of India PDFs.

Each PDF contains 12 monthly pages. Each page has 24 hourly columns and one row
per day. Output: tidy CSV with columns [datetime_ist, height_m] per port.

Uses layout-aware extraction (word positions) so partially blank days
(e.g. Diamond Harbour 2021 Jan days 2-9) don't break parsing.
"""
from __future__ import annotations

import calendar
import csv
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pdfplumber

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "TIDAL_DATA"
OUT_DIR = ROOT / "data"
OUT_DIR.mkdir(exist_ok=True)

MONTH_NAMES = [
    "JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE",
    "JULY", "AUGUST", "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER",
]
# Build prefix lookup so abbreviations (SEPT, AUG, NOV, etc.) all resolve.
MONTHS: dict[str, int] = {}
for _i, _name in enumerate(MONTH_NAMES, start=1):
    for _length in range(3, len(_name) + 1):
        MONTHS.setdefault(_name[:_length], _i)

HEADER_RE = re.compile(
    r"NAME OF PORT:-\s*(?P<port>[A-Z][A-Za-z0-9 ()\-]+?)\s+MONTH\s+(?P<month>[A-Za-z]+)\s+YEAR\s+(?P<year>\d{4})",
    re.IGNORECASE,
)
NUMBER_RE = re.compile(r"-?\d+\.\d+")


def group_words_into_rows(words: list[dict], y_tol: float = 3.0) -> list[list[dict]]:
    """Group words by their vertical position into rows."""
    rows: dict[int, list[dict]] = defaultdict(list)
    for w in words:
        y = round(float(w["top"]) / y_tol) * y_tol
        rows[y].append(w)
    out = []
    for y in sorted(rows):
        line = sorted(rows[y], key=lambda w: float(w["x0"]))
        out.append(line)
    return out


def parse_page(page) -> tuple[str, int, int, list[tuple[datetime, float]]] | None:
    text = page.extract_text() or ""
    m = HEADER_RE.search(text)
    if not m:
        return None
    port = m.group("port").strip().upper()
    # Normalize variants like "DIAMOND HARBOUR (ROY CHAK)"
    port = re.sub(r"\s*\([^)]*\)\s*", "", port).strip()
    mkey = m.group("month").upper()
    if mkey not in MONTHS:
        return None
    month = MONTHS[mkey]
    year = int(m.group("year"))
    days_in_month = calendar.monthrange(year, month)[1]

    words = page.extract_words(use_text_flow=False, keep_blank_chars=False)
    rows = group_words_into_rows(words, y_tol=3.0)

    records: list[tuple[datetime, float]] = []
    found_days: set[int] = set()

    for line in rows:
        if not line:
            continue
        # Each row should start with an integer day number 1..days_in_month
        first_text = line[0]["text"]
        if not first_text.isdigit():
            continue
        day = int(first_text)
        if not (1 <= day <= days_in_month) or day in found_days:
            continue
        # Collect floats from the rest of the line
        floats: list[float] = []
        for w in line[1:]:
            t = w["text"]
            if NUMBER_RE.fullmatch(t):
                floats.append(float(t))
        if len(floats) != 24:
            # Partial row -- skip this day (likely missing source data)
            found_days.add(day)
            continue
        found_days.add(day)
        for hour, h in enumerate(floats):
            records.append((datetime(year, month, day, hour), h))

    return port, year, month, records


def extract_pdf(path: Path) -> tuple[list[tuple[str, datetime, float]], int]:
    """Returns (records, num_unparsed_pages)."""
    out: list[tuple[str, datetime, float]] = []
    failures = 0
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            parsed = parse_page(page)
            if parsed is None:
                failures += 1
                continue
            port, _y, _m, records = parsed
            for dt, h in records:
                out.append((port, dt, h))
    return out, failures


def extract_port(port_dir: Path, out_csv: Path) -> tuple[int, int]:
    pdfs = sorted(port_dir.glob("*.pdf"))
    print(f"\n[{port_dir.name}] {len(pdfs)} PDF files")
    by_dt: dict[datetime, tuple[str, float]] = {}
    fail_total = 0
    for pdf_path in pdfs:
        recs, fails = extract_pdf(pdf_path)
        for port, dt, h in recs:
            by_dt[dt] = (port, h)
        status = f"({fails} unparsed pages)" if fails else ""
        print(f"  {pdf_path.name}: {len(recs):,} records {status}")
        fail_total += fails

    rows = sorted(by_dt.items())
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["port", "datetime_ist", "height_m"])
        for dt, (port, h) in rows:
            w.writerow([port, dt.isoformat(), f"{h:.3f}"])
    print(f"  -> {out_csv} ({len(rows):,} rows)")
    return len(rows), fail_total


def main() -> None:
    grand = 0
    for port_dir in sorted(DATA_DIR.iterdir()):
        if not port_dir.is_dir():
            continue
        slug = port_dir.name.lower().replace(" ", "_")
        n, _ = extract_port(port_dir, OUT_DIR / f"{slug}.csv")
        grand += n
    print(f"\nDone. Total records across ports: {grand:,}")


if __name__ == "__main__":
    main()

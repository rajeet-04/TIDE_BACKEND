"""Extract high/low tide events from monthly 2026 reference-table PDFs.

These PDFs use two narrow day columns and their text layer does not preserve
reading order. Parsing positioned characters is more reliable than raw text.

Usage:
  uv run python src/extract_reference_events.py \
      --data-dir data \
      --out data/reference_tides_2026.csv
"""
from __future__ import annotations

import argparse
import calendar
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pdfplumber
import pandas as pd

MONTHS = {
    name.upper(): number for number, name in enumerate(calendar.month_name) if name
}
HEADER_RE = re.compile(r"\bYEAR\s+(?P<year>\d{4})\s+(?P<month>[A-Z]+)\b")
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


@dataclass(frozen=True)
class TableColumn:
    day_x: tuple[float, float]
    time_x: tuple[float, float]
    height_x: tuple[float, float]


TABLE_COLUMNS = (
    TableColumn((234.0, 247.0), (252.0, 270.0), (271.0, 287.0)),
    TableColumn((294.0, 309.0), (313.0, 331.0), (332.0, 348.0)),
)


def _group_chars(chars: list[dict], x_range: tuple[float, float], min_top: float = 140.0) -> list[tuple[float, str]]:
    selected = [
        char
        for char in chars
        if x_range[0] <= float(char["x0"]) < x_range[1]
        and float(char["top"]) >= min_top
        and (char["text"].isdigit() or char["text"] == ".")
    ]
    rows: list[list[dict]] = []
    for char in sorted(selected, key=lambda item: (float(item["top"]), float(item["x0"]))):
        if not rows or abs(float(char["top"]) - float(rows[-1][0]["top"])) > 1.0:
            rows.append([char])
        else:
            rows[-1].append(char)
    return [
        (
            sum(float(char["top"]) for char in row) / len(row),
            "".join(char["text"] for char in sorted(row, key=lambda item: float(item["x0"]))),
        )
        for row in rows
    ]


def _day_labels(chars: list[dict], column: TableColumn) -> list[tuple[float, int]]:
    large_chars = [
        char
        for char in chars
        if column.day_x[0] <= float(char["x0"]) < column.day_x[1]
        and float(char["top"]) >= 140.0
        and float(char["size"]) >= 9.0
        and char["text"].isdigit()
    ]
    rows: list[list[dict]] = []
    for char in sorted(large_chars, key=lambda item: (float(item["top"]), float(item["x0"]))):
        if not rows or abs(float(char["top"]) - float(rows[-1][0]["top"])) > 1.5:
            rows.append([char])
        else:
            rows[-1].append(char)
    labels = []
    for row in rows:
        value = "".join(char["text"] for char in sorted(row, key=lambda item: float(item["x0"])))
        if value.isdigit():
            labels.append((sum(float(char["top"]) for char in row) / len(row), int(value)))
    return labels


def _valid_time(value: str) -> bool:
    return len(value) == 4 and value.isdigit() and int(value[:2]) < 24 and int(value[2:]) < 60


def _column_events(chars: list[dict], column: TableColumn) -> list[tuple[int, str, float]]:
    days = _day_labels(chars, column)
    times = [(top, value) for top, value in _group_chars(chars, column.time_x) if _valid_time(value)]
    heights = [
        (top, float(value))
        for top, value in _group_chars(chars, column.height_x)
        if NUMBER_RE.fullmatch(value) and 0.0 <= float(value) <= 15.0
    ]
    events = []
    for index, (day_top, day) in enumerate(days):
        next_top = days[index + 1][0] if index + 1 < len(days) else float("inf")
        day_times = [(top, value) for top, value in times if day_top - 2.0 <= top < next_top - 2.0]
        for time_top, time_value in day_times:
            candidates = [(abs(top - time_top), value) for top, value in heights if abs(top - time_top) <= 1.0]
            if not candidates:
                raise ValueError(f"No height found for day {day} time {time_value}")
            _, height = min(candidates)
            events.append((day, time_value, height))
    return events


def _classify_states(events: pd.DataFrame) -> pd.DataFrame:
    heights = events["height_m"].to_numpy()
    states = []
    for index, height in enumerate(heights):
        neighbor = heights[index + 1] if index == 0 else heights[index - 1]
        states.append("High" if height > neighbor else "Low")
    out = events.copy()
    out.insert(2, "state", states)
    return out


def extract_pdf(path: Path) -> pd.DataFrame:
    with pdfplumber.open(path) as pdf:
        if len(pdf.pages) != 1:
            raise ValueError(f"Expected one page in {path}, got {len(pdf.pages)}")
        page = pdf.pages[0]
        text = page.extract_text() or ""
        header = HEADER_RE.search(text)
        if not header:
            raise ValueError(f"Could not find month/year header in {path}")
        month_name = header.group("month").upper()
        if month_name not in MONTHS:
            raise ValueError(f"Unsupported month {month_name!r} in {path}")
        month = MONTHS[month_name]
        year = int(header.group("year"))
        chars = page.chars

    rows = []
    for column in TABLE_COLUMNS:
        for day, hhmm, height in _column_events(chars, column):
            rows.append(
                {
                    "port": path.stem.upper(),
                    "datetime_ist": datetime(
                        year, month, day, int(hhmm[:2]), int(hhmm[2:])
                    ),
                    "height_m": height,
                    "source_pdf": str(path),
                }
            )
    events = pd.DataFrame(rows).sort_values("datetime_ist").reset_index(drop=True)
    if events.empty:
        raise ValueError(f"No events extracted from {path}")
    if not events["datetime_ist"].is_unique:
        raise ValueError(f"Duplicate event times extracted from {path}")
    return _classify_states(events)


def _validate_month(events: pd.DataFrame, path: Path) -> None:
    counts = events.groupby(events["datetime_ist"].dt.day).size()
    expected_days = calendar.monthrange(
        events["datetime_ist"].iloc[0].year, events["datetime_ist"].iloc[0].month
    )[1]
    if len(counts) != expected_days:
        raise ValueError(f"{path}: extracted {len(counts)} days, expected {expected_days}")
    if not counts.between(3, 5).all():
        raise ValueError(f"{path}: invalid daily event counts: {counts.to_dict()}")
    states = events["state"].tolist()
    if any(left == right for left, right in zip(states, states[1:])):
        raise ValueError(f"{path}: extracted tide states do not alternate")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("data/reference_tides_2026.csv"))
    args = parser.parse_args()

    paths = [
        args.data_dir / f"{month} 2026" / f"{port}.pdf"
        for month in ("APRIL", "MAY", "JUNE")
        for port in ("HALDIA", "DIAMOND HARBOUR")
    ]
    frames = []
    for path in paths:
        events = extract_pdf(path)
        _validate_month(events, path)
        frames.append(events)
        print(f"{path}: {len(events)} events")

    combined = pd.concat(frames, ignore_index=True).sort_values(
        ["port", "datetime_ist"]
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.out, index=False)
    print(f"wrote {len(combined)} events -> {args.out}")


if __name__ == "__main__":
    main()

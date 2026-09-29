"""Model versions in models/<port>/<version>/ with a `current` pointer (spec 7.1, minimal until plan 3).

A version holds model A's constants (a.json), B's learners when present (level.joblib,
event.joblib), the 90% error ranges (ranges.json), metadata.json, and a regression snapshot
(snapshot_levels.csv, snapshot_events.csv) that tests replay on every run (spec 8).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from tide.ports import ROOT
from tide.stack import FittedStack

MODELS_DIR = ROOT / "models"
SNAPSHOT_START = pd.Timestamp("2026-01-01T00:00+05:30")
SNAPSHOT_HOURS = 72

def save_version(port_slug: str, fitted: FittedStack, metadata: dict, ranges: pd.DataFrame | None = None) -> str:
    """Save a new, immutable version: written to a temporary folder, then renamed into place.
    An existing version is never overwritten."""
    name = re.sub(r"[^A-Za-z0-9+._=-]", "_", metadata.get("candidate", fitted.model.config.name))
    stamp = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{name}"
    version, n = stamp, 1
    while (MODELS_DIR / port_slug / version).exists():
        n += 1
        version = f"{stamp}-{n}"
    temporary = MODELS_DIR / port_slug / f".{version}.tmp"
    temporary.mkdir(parents=True)
    fitted.save(temporary)
    if ranges is not None:
        (temporary / "ranges.json").write_text(json.dumps(ranges.to_dict("records"), indent=2) + "\n")
    (temporary / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    levels, events = snapshot(FittedStack.load(temporary))
    levels.to_csv(temporary / "snapshot_levels.csv", index=False)
    events.to_csv(temporary / "snapshot_events.csv", index=False)
    os.rename(temporary, MODELS_DIR / port_slug / version)
    return version

def snapshot(fitted: FittedStack) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hourly levels and events for SNAPSHOT_HOURS from SNAPSHOT_START: a fixed window whose
    predictions a saved version must always reproduce."""
    start = SNAPSHOT_START.tz_convert("UTC")
    end = start + pd.Timedelta(hours=SNAPSHOT_HOURS)
    times = pd.date_range(start, end, freq="1h", inclusive="left")
    return pd.DataFrame({"time_utc": times, "height_m": fitted.levels(times)}), fitted.events(start, end)

def set_current(port_slug: str, version: str) -> None:
    """Point `current` at a saved version; the pointer file is replaced atomically."""
    if not (MODELS_DIR / port_slug / version / "a.json").exists():
        raise FileNotFoundError(f"no saved version {version} for {port_slug}")
    pointer = MODELS_DIR / port_slug / "current"
    temporary = pointer.with_suffix(".tmp")
    temporary.write_text(version + "\n")
    os.replace(temporary, pointer)

def load_current(port_slug: str) -> tuple[str, FittedStack, dict, pd.DataFrame | None]:
    """The live version: its name, fitted model, metadata and ranges (None if it has none)."""
    pointer = MODELS_DIR / port_slug / "current"
    if not pointer.exists():
        raise FileNotFoundError(f"no current model for {port_slug}; run `uv run tide train --port {port_slug}`")
    version = pointer.read_text().strip()
    folder = MODELS_DIR / port_slug / version
    ranges_path = folder / "ranges.json"
    ranges = pd.DataFrame(json.loads(ranges_path.read_text())) if ranges_path.exists() else None
    return version, FittedStack.load(folder), json.loads((folder / "metadata.json").read_text()), ranges

def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def git_commit() -> str:
    """The checked-out commit, marked -dirty when tracked files have uncommitted changes."""
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
        changed = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                                 capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return head + ("-dirty" if changed else "")

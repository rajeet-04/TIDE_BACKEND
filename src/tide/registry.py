"""Model versions in models/<port>/<version>/ with a `current` pointer (spec 7.1, minimal until plan 3)."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import date
from pathlib import Path

from tide.harmonic import HarmonicModel
from tide.ports import ROOT

MODELS_DIR = ROOT / "models"

def save_version(port_slug: str, model: HarmonicModel, metadata: dict) -> str:
    version = f"{date.today():%Y%m%d}-{model.config.name}"
    folder = MODELS_DIR / port_slug / version
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "a.json").write_text(json.dumps(model.to_dict()) + "\n")
    (folder / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    return version

def set_current(port_slug: str, version: str) -> None:
    """Point `current` at a saved version; the pointer file is replaced atomically."""
    if not (MODELS_DIR / port_slug / version / "a.json").exists():
        raise FileNotFoundError(f"no saved version {version} for {port_slug}")
    pointer = MODELS_DIR / port_slug / "current"
    temporary = pointer.with_suffix(".tmp")
    temporary.write_text(version + "\n")
    os.replace(temporary, pointer)

def load_current(port_slug: str) -> tuple[str, HarmonicModel, dict]:
    pointer = MODELS_DIR / port_slug / "current"
    if not pointer.exists():
        raise FileNotFoundError(f"no current model for {port_slug}; run `uv run tide fit --port {port_slug} --promote`")
    version = pointer.read_text().strip()
    folder = MODELS_DIR / port_slug / version
    model = HarmonicModel.from_dict(json.loads((folder / "a.json").read_text()))
    return version, model, json.loads((folder / "metadata.json").read_text())

def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"

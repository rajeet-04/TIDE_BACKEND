"""Fitting and promoting versions, and the full training run (spec 6.1 `train`, 5.7).

Collaborators are reached through their modules (report.final_report, store.load_gauge, ...)
so tests can replace them.
"""
from __future__ import annotations

import json

import pandas as pd

from tide import audit, ports, qc, registry, report, selection, store
from tide.stack import StackCandidate, StackConfig

EVENT_DEFINITION = ("high water: maximum of the 1-minute curve; low water: centre of the interval within 1 cm "
                    "of the minimum; with an event correction, the corrected time and height of that event")

def fit_version(port_slug: str, promote: bool = False, n_jobs: int = 4) -> str:
    """Fit the selected stack on every QC-passed reading and save it as a new version, with the
    ranges recalibrated on all folds. With promote, re-run the final report first and make the
    version current only if the promotion checks pass (otherwise raise SystemExit)."""
    config = StackConfig.from_name(json.loads(report.selected_path(port_slug).read_text())["candidate"])
    if promote:
        result = report.final_report(port_slug, config, n_jobs)
        if result["promotion"]["passed"] is not True:
            raise SystemExit(f"refusing to promote {port_slug}: fresh backtest promotion checks did not pass")
    else:
        path = ports.OUTPUT_DIR / "backtest" / port_slug / "report.json"
        saved = json.loads(path.read_text()) if path.exists() else {}
        result = saved if saved.get("candidate") == config.name else {}   # never another model's scores
    readings = store.load_gauge(port_slug, passed_only=True)
    fitted = StackCandidate(config).fit(readings, port_slug)
    promotion = result.get("promotion", {})
    ranges = pd.DataFrame(result["ranges_production"]) if result.get("ranges_production") else None
    metadata = {"port": port_slug, "candidate": config.name,
                "training_start_utc": str(readings["time_utc"].min()), "training_end_utc": str(readings["time_utc"].max()),
                "training_readings": len(readings),
                "gauge_csv_sha256": registry.file_sha256(ports.DATA_DIR / f"{port_slug}.csv"),
                "qc_flags_sha256": registry.file_sha256(store.flags_path(port_slug)), "code_commit": registry.git_commit(),
                "publishable_horizons": promotion.get("publishable_horizons"), "backtest_promotion": promotion,
                "ranges": "split-conformal 90%, recalibrated on all folds" if ranges is not None else None,
                "event_definition": EVENT_DEFINITION}
    version = registry.save_version(port_slug, fitted, metadata, ranges)
    if promote:
        registry.set_current(port_slug, version)
    return version

def train(port_slug: str, n_jobs: int = 4, quality_control: bool = True) -> dict:
    """Spec 6.1: QC and audit the gauge record, choose model A and model B on the selection
    folds, score the final folds against the promotion rule, then save a version and make it
    current only if it passed."""
    if quality_control:
        qc.write_qc(port_slug, qc.run_qc(store.read_gauge_csv(ports.DATA_DIR / f"{port_slug}.csv"), port_slug))
        audit.write_audit(port_slug, audit.audit(audit.year_constants(store.load_gauge(port_slug), port_slug)))
    a = report.select(port_slug, n_jobs)
    config = selection.select_b(port_slug, a, n_jobs)
    passed = report.final_report(port_slug, config, n_jobs)["promotion"]["passed"] is True
    version = fit_version(port_slug, promote=passed, n_jobs=n_jobs)
    return {"port": port_slug, "candidate": config.name, "passed": passed, "version": version, "promoted": passed}

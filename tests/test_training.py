import pytest

from tide import audit, qc, report, selection, training
from tide.harmonic import HarmonicConfig
from tide.stack import StackConfig

A = HarmonicConfig()
STACK = StackConfig(A, event="lgbm-l31-m40")

@pytest.fixture
def steps(monkeypatch):
    """Replace every step of a training run with a recorder; returns the call log."""
    calls = []
    outcome = {"passed": True}
    monkeypatch.setattr(qc, "run_qc", lambda gauge, port_slug: calls.append("qc") or gauge)
    monkeypatch.setattr(qc, "write_qc", lambda port_slug, table: calls.append("write_qc"))
    monkeypatch.setattr(audit, "year_constants", lambda gauge, port_slug: calls.append("audit") or gauge)
    monkeypatch.setattr(audit, "audit", lambda constants: constants)
    monkeypatch.setattr(audit, "write_audit", lambda port_slug, table: calls.append("write_audit"))
    monkeypatch.setattr(report, "select", lambda port_slug, n_jobs=4: calls.append("select_a") or A)
    monkeypatch.setattr(selection, "select_b", lambda port_slug, a, n_jobs=4: calls.append("select_b") or STACK)
    monkeypatch.setattr(report, "final_report",
                        lambda port_slug, config, n_jobs=4: calls.append(("final", config.name)) or
                        {"promotion": {"passed": outcome["passed"]}})
    monkeypatch.setattr(training, "fit_version",
                        lambda port_slug, promote=False, n_jobs=4: calls.append(("fit", promote)) or "v1")
    return calls, outcome

def test_a_training_run_goes_through_every_step_and_promotes_a_pass(steps):
    calls, _ = steps
    summary = training.train("haldia")
    assert calls == ["qc", "write_qc", "audit", "write_audit", "select_a", "select_b", ("final", STACK.name), ("fit", True)]
    assert summary == {"port": "haldia", "candidate": STACK.name, "passed": True, "version": "v1", "promoted": True}

def test_a_failed_candidate_is_saved_but_not_promoted(steps):
    calls, outcome = steps
    outcome["passed"] = False
    summary = training.train("haldia", quality_control=False)
    assert "qc" not in calls and calls[-1] == ("fit", False) and not summary["promoted"]

def test_a_version_records_the_readings_its_window_actually_fitted(tmp_path, monkeypatch):
    import json

    import numpy as np
    import pandas as pd

    from tide import ports, registry, store
    from tide.ports import ist_year_start

    times = pd.date_range(ist_year_start(2016), ist_year_start(2020), freq="1h", inclusive="left")
    hours = np.arange(len(times), dtype=float)
    readings = pd.DataFrame({"time_utc": times, "height_m": 3.0 + 2.0 * np.cos(2 * np.pi * hours / 12.4206)})
    (tmp_path / "selected.json").write_text(json.dumps({"candidate": "A-w2-auto-noside-notrend"}))
    monkeypatch.setattr(report, "selected_path", lambda port_slug: tmp_path / "selected.json")
    monkeypatch.setattr(store, "load_gauge", lambda port_slug, passed_only=False: readings)
    monkeypatch.setattr(registry, "file_sha256", lambda path: "0" * 64)
    monkeypatch.setattr(registry, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(ports, "OUTPUT_DIR", tmp_path / "output")
    version = training.fit_version("haldia")
    saved = json.loads((tmp_path / "models" / "haldia" / version / "metadata.json").read_text())
    kept = readings[readings["time_utc"] > times[-1] - pd.DateOffset(years=2)]
    assert saved["training_readings"] == len(kept)
    assert pd.Timestamp(saved["training_start_utc"]) == kept["time_utc"].min()

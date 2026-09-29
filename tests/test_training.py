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

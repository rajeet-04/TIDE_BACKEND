import pytest

from tide import cli

def test_help_names_the_program(capsys):
    with pytest.raises(SystemExit) as stopped:
        cli.main(["--help"])
    assert stopped.value.code == 0
    assert "usage: tide" in capsys.readouterr().out

def test_predict_command_parses():
    args = cli.build_parser().parse_args(["predict", "--port", "haldia", "--start", "2026-10-01", "--hours", "24"])
    assert args.handler.__name__ == "_predict" and args.hours == 24.0 and args.step == 60

def _selected(tmp_path, monkeypatch, name="A-wall-auto-noside-notrend"):
    import json

    from tide import report
    path = tmp_path / "selected.json"
    path.write_text(json.dumps({"candidate": name}))
    monkeypatch.setattr(report, "selected_path", lambda port_slug: path)

def test_promote_refuses_when_fresh_checks_fail(tmp_path, monkeypatch):
    from tide import registry, report
    _selected(tmp_path, monkeypatch)
    monkeypatch.setattr(registry, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(report, "final_report", lambda port_slug, config, n_jobs=4: {"promotion": {"passed": False}})
    with pytest.raises(SystemExit, match="refusing to promote"):
        cli.main(["fit", "--port", "haldia", "--promote"])
    assert not (tmp_path / "models" / "haldia" / "current").exists()

def test_fit_ignores_a_report_for_another_candidate(tmp_path, monkeypatch):
    import json

    import numpy as np
    import pandas as pd

    from tide import ports, registry, store
    _selected(tmp_path, monkeypatch)
    monkeypatch.setattr(registry, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(ports, "OUTPUT_DIR", tmp_path)
    (tmp_path / "backtest" / "haldia").mkdir(parents=True)
    (tmp_path / "backtest" / "haldia" / "report.json").write_text(json.dumps(
        {"candidate": "A-w5-auto-side-trend", "promotion": {"passed": True, "publishable_horizons": [1, 2, 3]}}))
    times = pd.date_range("2022-01-01", periods=8766, freq="1h", tz="UTC")
    gauge = pd.DataFrame({"time_utc": times, "height_m": 3 + 2 * np.cos(np.arange(len(times)) * 2 * np.pi / 12.42),
                          "source": "soi_gauge", "qc_flag": ""})
    monkeypatch.setattr(store, "load_gauge", lambda port_slug, passed_only=False: gauge)
    cli.main(["fit", "--port", "haldia"])
    (folder,) = (tmp_path / "models" / "haldia").iterdir()
    metadata = json.loads((folder / "metadata.json").read_text())
    assert metadata["backtest_promotion"] == {} and metadata["publishable_horizons"] is None

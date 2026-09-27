import json

import pandas as pd

from tide import report
from tide.harmonic import HarmonicConfig
from tide.ports import IST, ist_year_start

def test_grid_covers_every_setting_once():
    names = [c.name for c in report.GRID]
    assert len(names) == 32 == len(set(names))
    assert all(HarmonicConfig.from_name(n) == c for n, c in zip(names, report.GRID))

def fake_results(hit_share: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, hours = [], []
    for origin, year in [(2020, 2020), (2020, 2021), (2021, 2021)]:
        first = pd.Timestamp(f"{year}-01-01 06:00", tz=IST).tz_convert("UTC")
        tag = {"origin": origin, "test_year": year, "horizon": year - origin + 1, "final": True}
        for k in range(700):
            t = first + pd.Timedelta(hours=12.42 * k)
            hit = (k % 100) < hit_share * 100
            rows.append({"kind": "observed", **tag, "state": "High" if k % 2 else "Low", "observed_time_utc": t,
                         "time_utc": t, "hit": hit, "time_error_minutes": 10.0 if hit else 45.0, "height_error_m": 0.1})
            hours.append({"time_utc": t, "observed_m": 3.0, "predicted_m": 3.1, **tag})
    return pd.DataFrame(rows), pd.DataFrame(hours)

def test_final_report_writes_markdown_and_json(tmp_path, monkeypatch):
    results = {"A-w8-auto-noside-trend": fake_results(0.9), "current_pipeline": fake_results(0.8),
               "utide_only": fake_results(0.7)}
    monkeypatch.setattr(report, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(report, "load_gauge", lambda port_slug, passed_only=False: pd.DataFrame(
        {"time_utc": pd.date_range(ist_year_start(2018), ist_year_start(2022), freq="30D")}))
    monkeypatch.setattr(report, "run", lambda port_slug, cand, fold_list, n_jobs=1: results[cand.name])
    monkeypatch.setattr(report, "load_tables", lambda port_slug: pd.DataFrame())
    result = report.final_report("haldia", HarmonicConfig(window_years=8), n_jobs=1)
    assert result["candidate"] == "A-w8-auto-noside-trend" and result["promotion"]["checks"]["joint_better"]
    text = (tmp_path / "backtest" / "haldia" / "report.md").read_text(encoding="utf-8")
    assert "Promotion checks against the current pipeline" in text and "joint_better: PASS" in text
    assert json.loads((tmp_path / "backtest" / "haldia" / "report.json").read_text())["pooled"]

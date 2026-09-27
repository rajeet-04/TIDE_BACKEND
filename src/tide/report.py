"""Model A selection on the selection folds and the final-set report (spec 5.2, 5.5, 5.7)."""
from __future__ import annotations

import itertools
import json
from datetime import date

import pandas as pd

from tide.backtest import SELECTION_LAST_YEAR, folds, promotion, run, score_fixed, summary
from tide.candidates import CurrentPipeline, HarmonicCandidate, UTideOnly
from tide.harmonic import HarmonicConfig
from tide.ports import OUTPUT_DIR, ROOT, ist_years
from tide.store import load_gauge
from tide.tables import load_tables

GRID = [HarmonicConfig(window_years=w, constituents=c, side_terms=s, trend=t)
        for w, c, s, t in itertools.product((5, 8, 12, None), ("auto", "auto+shallow"), (False, True), (False, True))]
SHOWN = ["candidate", "observed", "joint_pct", "time_mae_min", "time_p95_min", "time_bias_min",
         "height_mae_m", "height_p95_m", "height_bias_m", "missed", "extra", "hourly_rmse_m"]

def selected_path(port_slug: str):
    return ROOT / "models" / port_slug / "selected.json"

def _years(port_slug: str) -> set[int]:
    return set(ist_years(load_gauge(port_slug, passed_only=True)["time_utc"]))

def select(port_slug: str, n_jobs: int = 4) -> HarmonicConfig:
    """Score every grid setting on the selection folds; record and return the best joint share
    (ties go to the lower timing error)."""
    selection = [f for f in folds(port_slug, _years(port_slug)) if not f.final]
    rows = []
    for config in GRID:
        events, hours = run(port_slug, HarmonicCandidate(config), selection, n_jobs=n_jobs)
        rows.append({"candidate": config.name, **summary(events, hours, by=()).iloc[0].to_dict()})
    ranking = pd.DataFrame(rows).sort_values(["joint_pct", "time_mae_min"], ascending=[False, True]).reset_index(drop=True)
    out = OUTPUT_DIR / "backtest" / port_slug
    out.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(out / "selection.csv", index=False)
    best = HarmonicConfig.from_name(ranking["candidate"].iloc[0])
    path = selected_path(port_slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"candidate": best.name,
                                "selection_joint_pct": round(float(ranking["joint_pct"].iloc[0]), 2),
                                "selected_on": date.today().isoformat()}, indent=2) + "\n")
    return best

def final_report(port_slug: str, config: HarmonicConfig, n_jobs: int = 4) -> dict:
    """Score the selected model and both baselines on the final folds, compare with the official
    tables where they overlap, run the promotion checks and write report.md and report.json."""
    final = [f for f in folds(port_slug, _years(port_slug)) if f.final]
    results = {c.name: run(port_slug, c, final, n_jobs=n_jobs)
               for c in (HarmonicCandidate(config), CurrentPipeline(), UTideOnly())}

    def table(by) -> pd.DataFrame:
        return pd.concat([summary(ev, hr, by=by).assign(candidate=name) for name, (ev, hr) in results.items()],
                         ignore_index=True)

    pooled, by_horizon, by_season = table(()), table(("horizon",)), table(("season",))
    versus_tables = _versus_tables(port_slug, results)
    check = promotion(results["current_pipeline"][0], results[config.name][0])
    result = {"port": port_slug, "candidate": config.name, "promotion": check,
              "pooled": pooled.to_dict("records"), "by_horizon": by_horizon.to_dict("records"),
              "by_season": by_season.to_dict("records"), "versus_tables": versus_tables.to_dict("records")}
    out = OUTPUT_DIR / "backtest" / port_slug
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(result, indent=2, default=_plain) + "\n")
    (out / "report.md").write_text(_markdown(result, pooled, by_horizon, by_season, versus_tables), encoding="utf-8")
    return result

def _versus_tables(port_slug: str, results: dict) -> pd.DataFrame:
    """The official tables against each candidate's one-year-ahead forecast, per shared test year."""
    official = score_fixed(port_slug, load_tables(port_slug))
    if official.empty:
        return pd.DataFrame()
    frames = []
    for year in sorted(set(official["test_year"])):
        frames.append(summary(official[official["test_year"] == year], pd.DataFrame(), by=())
                      .assign(candidate="official_tables", test_year=year))
        for name, (ev, hr) in results.items():
            same = (ev["test_year"] == year) & (ev["horizon"] == 1)
            if same.any():
                hours = hr[(hr["test_year"] == year) & (hr["horizon"] == 1)]
                frames.append(summary(ev[same], hours, by=()).assign(candidate=name, test_year=year))
    return pd.concat(frames, ignore_index=True)

def _plain(value):
    return value.item() if hasattr(value, "item") else str(value)

def _markdown(result: dict, pooled, by_horizon, by_season, versus_tables) -> str:
    check = result["promotion"]
    lines = [f"# Backtest report: {result['port']}", "",
             f"Selected model A: `{result['candidate']}`. Selection set: test years up to {SELECTION_LAST_YEAR}; "
             "final set: later years. Truth: QC-passed gauge readings. A hit is within ±30 min and ±0.30 m.", ""]
    sections = (("Final set, all horizons", pooled, []), ("Final set by horizon", by_horizon, ["horizon"]),
                ("Final set by season", by_season, ["season"]),
                ("Against the official tables (one year ahead)", versus_tables, ["test_year"]))
    for title, frame, keys in sections:
        if len(frame):
            shown = frame[[*keys, *[c for c in SHOWN if c in frame.columns]]]
            lines += [f"## {title}", "", "```text", shown.round(3).to_string(index=False), "```", ""]
    lines += ["## Promotion checks against the current pipeline", ""]
    lines += [f"- {name}: {'PASS' if passed else 'FAIL'}" for name, passed in check["checks"].items()]
    joint = check["joint"]
    seasons = ", ".join(f"{k} {v:+.1f}" for k, v in check["season_diff_pp"].items())
    lines += ["", f"Joint share difference: {joint['diff']:+.2f} points "
                  f"(95% interval {joint['lo']:+.2f} to {joint['hi']:+.2f}).",
              f"Season differences (points): {seasons}.",
              f"Publishable horizons: {check['publishable_horizons']}. Coverage check: {check['coverage_check']}.",
              f"Overall: {'PASSED' if check['passed'] else 'NOT PASSED'}.", ""]
    return "\n".join(lines)

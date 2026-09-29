"""Model A selection on the selection folds and the final-set report (spec 4.7, 5.2, 5.5, 5.7)."""
from __future__ import annotations

import itertools
import json
from datetime import date

import pandas as pd

from tide.backtest import SELECTION_LAST_YEAR, folds, promotion, run, score_fixed, summary
from tide.candidates import CurrentPipeline, HarmonicCandidate, UTideOnly
from tide.harmonic import HarmonicConfig
from tide.ports import OUTPUT_DIR, ROOT, ist_years
from tide.ranges import calibrate, coverage
from tide.stack import StackCandidate, StackConfig
from tide.store import load_gauge
from tide.tables import load_tables

GRID = [HarmonicConfig(window_years=w, constituents=c, side_terms=s, trend=t)
        for w, c, s, t in itertools.product((5, 8, 12, None), ("auto", "auto+shallow"), (False, True), (False, True))]
SHOWN = ["candidate", "observed", "joint_pct", "time_mae_min", "time_p95_min", "time_bias_min",
         "height_mae_m", "height_p95_m", "height_bias_m", "missed", "extra", "hourly_rmse_m"]
# Findings from the per-year audit that every final score of a port must carry (decision log 2026-09-27).
CAVEATS = {"diamond_harbour": "tides in the 2021–2023 final years run about 10 min later than in 2000–2016 "
                              "(M2 phase about +5° against the same months), consistent with a gauge-site change; "
                              "final scores there understate a model fitted to earlier years."}

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

def final_report(port_slug: str, config: StackConfig, n_jobs: int = 4) -> dict:
    """Score the selected model, model A alone and both baselines on the final folds. Calibrate
    the selected model's 90% ranges on its selection folds and check their coverage on the final
    folds; compare with the official tables; run the promotion checks; write report.md and
    report.json. report.json also keeps the ranges recalibrated on all folds, which a saved
    version serves (spec 4.7)."""
    chosen = StackCandidate(config)
    every = folds(port_slug, _years(port_slug))
    final, selection = [f for f in every if f.final], [f for f in every if not f.final]
    rivals = [chosen, CurrentPipeline(), UTideOnly()] + ([HarmonicCandidate(config.a)] if config.level or config.event else [])
    results = {c.name: run(port_slug, c, final, n_jobs=n_jobs) for c in rivals}
    calibration = run(port_slug, chosen, selection, n_jobs=n_jobs)
    ranges = calibrate(*calibration)
    covered = coverage(*results[chosen.name], ranges)
    covered_seasons = coverage(*results[chosen.name], ranges, by="season")
    production = calibrate(*(pd.concat([a, b], ignore_index=True) for a, b in zip(calibration, results[chosen.name])))
    official = score_fixed(port_slug, load_tables(port_slug))

    def table(by) -> pd.DataFrame:
        return pd.concat([summary(ev, hr, by=by).assign(candidate=name) for name, (ev, hr) in results.items()],
                         ignore_index=True)

    pooled, by_horizon, by_season = table(()), table(("horizon",)), table(("season",))
    versus_tables = _versus_tables(official, results)
    check = promotion(results["current_pipeline"][0], results[chosen.name][0], coverage=covered, tables=official)
    result = {"port": port_slug, "candidate": chosen.name, "promotion": check,
              "pooled": pooled.to_dict("records"), "by_horizon": by_horizon.to_dict("records"),
              "by_season": by_season.to_dict("records"), "versus_tables": versus_tables.to_dict("records"),
              "coverage": covered.to_dict("records"), "coverage_by_season": covered_seasons.to_dict("records"),
              "ranges_selection": ranges.to_dict("records"),
              "ranges_production": production.to_dict("records")}
    out = OUTPUT_DIR / "backtest" / port_slug
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(result, indent=2, default=_plain) + "\n")
    (out / "report.md").write_text(_markdown(result, pooled, by_horizon, by_season, versus_tables, covered, covered_seasons),
                                   encoding="utf-8")
    return result

def _versus_tables(official: pd.DataFrame, results: dict) -> pd.DataFrame:
    """The official tables against each candidate's one-year-ahead forecast, per shared test year."""
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

def _markdown(result: dict, pooled, by_horizon, by_season, versus_tables, covered, covered_seasons) -> str:
    check = result["promotion"]
    lines = [f"# Backtest report: {result['port']}", "",
             f"Selected model: `{result['candidate']}`. Selection set: test years up to {SELECTION_LAST_YEAR}; "
             "final set: later years. Truth: QC-passed gauge readings. A hit is within ±30 min and ±0.30 m.", ""]
    if result["port"] in CAVEATS:
        lines += [f"**Caveat:** {CAVEATS[result['port']]}", ""]
    sections = (("Final set, all horizons", pooled, []), ("Final set by horizon", by_horizon, ["horizon"]),
                ("Final set by season", by_season, ["season"]),
                ("Against the official tables (one year ahead)", versus_tables, ["test_year"]))
    for title, frame, keys in sections:
        if len(frame):
            shown = frame[[*keys, *[c for c in SHOWN if c in frame.columns]]]
            lines += [f"## {title}", "", "```text", shown.round(3).to_string(index=False), "```", ""]
    seasons_grid = covered_seasons[covered_seasons["season"] != "all"].pivot(index="output", columns="season",
                                                                            values="covered_pct")
    grid = covered.pivot(index="output", columns="horizon", values="covered_pct").join(seasons_grid)
    lines += ["## Error-range coverage on the final set (%)", "",
              "90% ranges calibrated on the selection folds; the band is 88-92. Columns: all, each horizon, "
              "each season.", "",
              "```text", grid.round(1).to_string(), "```", ""]
    lines += ["## Promotion checks against the current pipeline", ""]
    lines += [f"- {name}: {'PASS' if passed else 'FAIL'}" for name, passed in check["checks"].items()]
    joint = check["joint"]
    seasons = ", ".join(f"{k} {v:+.1f}" for k, v in check["season_diff_pp"].items())
    tables = ", ".join(f"{year} {diff:+.1f}" for year, diff in check["versus_tables_pp"].items()) or "no overlap"
    lines += ["", f"Joint share difference: {joint['diff']:+.2f} points "
                  f"(95% interval {joint['lo']:+.2f} to {joint['hi']:+.2f}).",
              f"Season differences (points): {seasons}.",
              f"Against the official tables (points): {tables}.",
              f"Publishable horizons: {check['publishable_horizons']}.",
              f"Overall: {'PASSED' if check['passed'] else 'NOT PASSED'}.", ""]
    return "\n".join(lines)

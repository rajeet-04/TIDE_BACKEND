"""Choosing model B on the selection folds (spec 4.3, 4.5, 5.2).

On top of the selected model A, every event learner is ranked by joint share (then timing
MAE) and every level learner by hourly RMSE; a blend of the top two of each competes as its
own learner. The chosen stack is the best of four by joint share, then hourly RMSE: A alone,
A with the best level correction, A with the best event correction, and A with both (whose
events are the event correction's and whose levels are the level correction's).
"""
from __future__ import annotations

import json
from datetime import date

import pandas as pd

from tide.backtest import folds, run, summary
from tide.harmonic import HarmonicConfig
from tide.ports import OUTPUT_DIR, ist_years
from tide.report import selected_path
from tide.stack import StackCandidate, StackConfig
from tide.store import load_gauge

EVENT_LEARNERS = ("lgbm-l15-m20", "lgbm-l31-m40", "lgbm-l63-m40", "lgbm-l31-m100", "et", "xgb-d4", "xgb-d6", "mlp")
LEVEL_LEARNERS = ("lgbm-l31-m100", "lgbm-l63-m400", "lgbm-l31-m100-huber", "xgb-d6", "mlp")
BLEND_WEIGHTS = (0.25, 0.5, 0.75)

def select_b(port_slug: str, a: HarmonicConfig, n_jobs: int = 4) -> StackConfig:
    """Score the B candidates on the selection folds, write selection_b.csv and selected.json,
    and return the chosen stack."""
    selection = [f for f in folds(port_slug, set(ist_years(load_gauge(port_slug, passed_only=True)["time_utc"])))
                 if not f.final]
    rows: dict[str, dict] = {}

    def score(config: StackConfig) -> dict:
        if config.name not in rows:
            events, hours = run(port_slug, StackCandidate(config), selection, n_jobs=n_jobs)
            rows[config.name] = {"candidate": config.name, **summary(events, hours, by=()).iloc[0].to_dict()}
        return rows[config.name]

    event = _best(a, "event", EVENT_LEARNERS, score, key=lambda row: (-row["joint_pct"], row["time_mae_min"]))
    level = _best(a, "level", LEVEL_LEARNERS, score, key=lambda row: row["hourly_rmse_m"])
    options = [StackConfig(a), StackConfig(a, level=level), StackConfig(a, event=event),
               StackConfig(a, level=level, event=event)]
    chosen = min(options, key=lambda config: (-score(config)["joint_pct"], score(config)["hourly_rmse_m"]))
    out = OUTPUT_DIR / "backtest" / port_slug
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows.values()).sort_values(["joint_pct", "time_mae_min"], ascending=[False, True]).to_csv(
        out / "selection_b.csv", index=False)
    selected_path(port_slug).write_text(json.dumps({
        "candidate": chosen.name, "a_candidate": a.name,
        "selection_joint_pct": round(float(rows[chosen.name]["joint_pct"]), 2),
        "selection_hourly_rmse_m": round(float(rows[chosen.name]["hourly_rmse_m"]), 4),
        "a_selection_joint_pct": round(float(rows[a.name]["joint_pct"]), 2),
        "selected_on": date.today().isoformat()}, indent=2) + "\n")
    return chosen

def _best(a: HarmonicConfig, slot: str, learners, score, key) -> str:
    """The best learner for one correction, after its top two also compete as blends."""
    def config(learner: str) -> StackConfig:
        return StackConfig(a, **{slot: learner})

    ranked = sorted(learners, key=lambda learner: key(score(config(learner))))
    first, second = ranked[0], ranked[1]
    blends = [f"blend({first}={w:g},{second}={1 - w:g})" for w in BLEND_WEIGHTS]
    return min([*ranked, *blends], key=lambda learner: key(score(config(learner))))

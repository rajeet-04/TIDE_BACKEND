"""Split-conformal 90% error ranges (spec 4.7) and their coverage (spec 1.2 criterion 4).

Ranges come from a candidate's own backtest errors, per output, horizon (1-3) and season:
- in the backtest, calibrated on the selection folds and checked on the final folds;
- in production, recalibrated on all folds.
A range bounds "observed minus predicted": the outcome lies in [prediction + lower, prediction + upper].
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tide.ports import SEASONS, season_of

ALPHA = 0.10
HORIZONS = (1, 2, 3)
OUTPUTS = ("level", "high_time", "high_height", "low_time", "low_height")
MIN_CELL = 100

def differences(events: pd.DataFrame, hours: pd.DataFrame) -> pd.DataFrame:
    """Observed minus predicted for every outcome in backtest rows, as output, horizon, season,
    difference (minutes for times, metres for heights and levels). Missed events are NaN."""
    frames = []
    observed = events[events["kind"] == "observed"]
    for state, prefix in (("High", "high"), ("Low", "low")):
        part = observed[observed["state"] == state]
        tags = {"horizon": part["horizon"].to_numpy(dtype=int), "season": season_of(part["time_utc"])}
        frames.append(pd.DataFrame({"output": f"{prefix}_time", **tags,
                                    "difference": -part["time_error_minutes"].to_numpy(dtype=float)}))
        frames.append(pd.DataFrame({"output": f"{prefix}_height", **tags,
                                    "difference": -part["height_error_m"].to_numpy(dtype=float)}))
    if len(hours):
        frames.append(pd.DataFrame({"output": "level", "horizon": hours["horizon"].to_numpy(dtype=int),
                                    "season": season_of(hours["time_utc"]),
                                    "difference": (hours["observed_m"] - hours["predicted_m"]).to_numpy(dtype=float)}))
    return pd.concat(frames, ignore_index=True)

def calibrate(events: pd.DataFrame, hours: pd.DataFrame, alpha: float = ALPHA) -> pd.DataFrame:
    """Range bounds for every output, horizon and season: output, horizon, season, lower,
    upper, n, source. A cell with fewer than MIN_CELL errors borrows its output's errors at
    that horizon (source 'horizon'), then at every horizon (source 'output')."""
    diffs = differences(events, hours).dropna(subset=["difference"])
    rows = []
    for output in OUTPUTS:
        mine = diffs[diffs["output"] == output]
        if mine.empty:
            continue
        for horizon in HORIZONS:
            for season in SEASONS:
                choices = (("cell", mine[(mine["horizon"] == horizon) & (mine["season"] == season)]),
                           ("horizon", mine[mine["horizon"] == horizon]), ("output", mine))
                source, cell = next((s, c) for s, c in choices if len(c) >= MIN_CELL or s == "output")
                lower, upper = _conformal(cell["difference"].to_numpy(), alpha)
                rows.append({"output": output, "horizon": horizon, "season": season, "lower": lower,
                             "upper": upper, "n": len(cell), "source": source})
    return pd.DataFrame(rows)

def _conformal(values: np.ndarray, alpha: float) -> tuple[float, float]:
    """Split-conformal bounds holding 1 - alpha of new outcomes (alpha/2 on each side)."""
    n = len(values)
    low = min(max(np.floor((n + 1) * alpha / 2) / n, 0.0), 1.0)
    high = min(np.ceil((n + 1) * (1 - alpha / 2)) / n, 1.0)
    return float(np.quantile(values, low, method="lower")), float(np.quantile(values, high, method="higher"))

def coverage(events: pd.DataFrame, hours: pd.DataFrame, table: pd.DataFrame, by: str = "horizon") -> pd.DataFrame:
    """Share (%) of outcomes inside their range: output, <by> ('all', then each horizon or
    season), covered_pct, n. A missed event counts as outside."""
    diffs = differences(events, hours).merge(table[["output", "horizon", "season", "lower", "upper"]],
                                             on=["output", "horizon", "season"], how="left")
    diffs["inside"] = (diffs["difference"] >= diffs["lower"]) & (diffs["difference"] <= diffs["upper"])
    pooled = diffs.groupby("output")["inside"].agg(["mean", "size"]).reset_index().assign(**{by: "all"})
    per = diffs.groupby(["output", by])["inside"].agg(["mean", "size"]).reset_index()
    both = pd.concat([pooled, per], ignore_index=True).astype({by: object})
    return pd.DataFrame({"output": both["output"], by: both[by], "covered_pct": 100.0 * both["mean"],
                         "n": both["size"]})

def bounds(table: pd.DataFrame, output: str, horizon, season) -> tuple[np.ndarray, np.ndarray]:
    """Lower and upper bounds for each (horizon, season) pair; horizons are clipped to 1-3."""
    keys = pd.DataFrame({"horizon": np.clip(np.asarray(horizon, dtype=int), 1, 3), "season": np.asarray(season)})
    found = keys.merge(table[table["output"] == output], on=["horizon", "season"], how="left")
    return found["lower"].to_numpy(dtype=float), found["upper"].to_numpy(dtype=float)

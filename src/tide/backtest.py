"""Rolling-origin yearly backtest (spec 5): folds, cached parallel scoring and summaries."""
from __future__ import annotations

import hashlib
import os
import pickle
import re
from importlib.metadata import version
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config

from tide.events import coverage, extra_events, find_events, match_events, model_events, summarize
from tide.ports import IST, ROOT, SEASONS, ist_year_start, ist_years, season_of
from tide.store import load_gauge

CACHE_DIR = ROOT / ".cache" / "backtest"
SELECTION_LAST_YEAR = 2019  # moves forward with the final set when new gauge years arrive (spec 5.2)
ORIGINS = {"haldia": list(range(2008, 2025)), "diamond_harbour": list(range(2008, 2017)) + [2021]}
HORIZONS = 3
RESULT_MODULES = ("ports", "store", "harmonic", "events", "backtest", "candidates", "features", "crossfit",
                  "learners", "stack")
LIBRARIES = ("numpy", "pandas", "scipy", "scikit-learn", "utide", "lightgbm", "xgboost", "torch")

@dataclass(frozen=True)
class Fold:
    origin: int      # train on readings before 00:00 IST on 1 January of this year
    test_year: int

    @property
    def horizon(self) -> int:
        return self.test_year - self.origin + 1

    @property
    def final(self) -> bool:
        return self.test_year > SELECTION_LAST_YEAR

def folds(port_slug: str, data_years: set[int]) -> list[Fold]:
    return [Fold(o, y) for o in ORIGINS[port_slug] for y in range(o, o + HORIZONS) if y in data_years]

class Fitted(Protocol):
    def levels(self, times: pd.DatetimeIndex) -> np.ndarray: ...
    def events(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame: ...

class Candidate(Protocol):
    name: str
    def fit(self, train: pd.DataFrame, port_slug: str) -> Fitted: ...

@dataclass
class FittedLevels:
    """A fitted model known by its level function; its events come from the shared routine (spec 4.6)."""
    levels: Callable[[pd.DatetimeIndex], np.ndarray]

    def events(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        return model_events(self.levels, start, end)

def run(port_slug: str, candidate: Candidate, fold_list: list[Fold], *, n_jobs: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score a candidate on the folds, fitting once per origin.

    Results are cached under .cache/backtest, keyed by the code and the QC-passed data, so a
    change to either refits. Returns two tables:
    - event rows: kind 'observed' for each observed event, 'extra' for each unclaimed prediction
    - hourly rows: observed and predicted level at every QC-passed reading"""
    gauge = load_gauge(port_slug, passed_only=True)
    by_origin: dict[int, list[Fold]] = {}
    for fold in fold_list:
        by_origin.setdefault(fold.origin, []).append(fold)
    if not by_origin:
        return pd.DataFrame(), pd.DataFrame()
    config = getattr(candidate, "config", None)  # names do not cover every setting
    variant = hashlib.sha256(repr(asdict(config)).encode()).hexdigest()[:8] if config is not None else "fixed"
    folder = re.sub(r"[^A-Za-z0-9+._=-]", "_", candidate.name)   # names hold / : ( ) , which paths cannot
    cache = CACHE_DIR / port_slug / _cache_key(gauge) / f"{folder}-{variant}"
    jobs = [delayed(_run_origin)(port_slug, candidate, fs, gauge, cache) for _, fs in sorted(by_origin.items())]
    threads = max(1, (os.cpu_count() or 1) // max(1, n_jobs))
    with parallel_config(backend="loky", inner_max_num_threads=threads):
        parts = Parallel(n_jobs=n_jobs)(jobs)
    return (pd.concat([p[0] for p in parts], ignore_index=True),
            pd.concat([p[1] for p in parts], ignore_index=True))

def _cache_key(gauge: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for library in LIBRARIES:  # upgrades change results
        digest.update(f"{library}={version(library)}".encode())
    for path in [ROOT / "src" / "tide" / f"{m}.py" for m in RESULT_MODULES] + [ROOT / "src" / "utide_event_model.py"]:
        if path.exists():
            digest.update(path.read_bytes())
    digest.update(pd.util.hash_pandas_object(gauge[["time_utc", "height_m"]], index=False).to_numpy().tobytes())
    return digest.hexdigest()[:16]

def _run_origin(port_slug: str, candidate: Candidate, fs: list[Fold], gauge: pd.DataFrame, cache) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = cache / f"{fs[0].origin}_{'-'.join(str(f.test_year) for f in fs)}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    fitted = candidate.fit(gauge[gauge["time_utc"] < ist_year_start(fs[0].origin)].reset_index(drop=True), port_slug)
    events, hours = [], []
    for fold in fs:
        start, end = ist_year_start(fold.test_year), ist_year_start(fold.test_year + 1)
        observed = gauge[(gauge["time_utc"] >= start) & (gauge["time_utc"] < end)].reset_index(drop=True)
        truth = find_events(observed["time_utc"], observed["height_m"], step_minutes=60)
        predicted = fitted.events(start, end)
        matches = match_events(truth, predicted)
        extras = extra_events(predicted, matches, coverage(observed["time_utc"]))
        tag = {"origin": fold.origin, "test_year": fold.test_year, "horizon": fold.horizon, "final": fold.final}
        events.append(matches.assign(kind="observed", time_utc=matches["observed_time_utc"], **tag))
        events.append(extras.assign(kind="extra", hit=False, **tag))
        hours.append(pd.DataFrame({"time_utc": observed["time_utc"], "observed_m": observed["height_m"],
                                   "predicted_m": fitted.levels(pd.DatetimeIndex(observed["time_utc"]))}).assign(**tag))
    result = (pd.concat(events, ignore_index=True), pd.concat(hours, ignore_index=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")   # an interrupted run never leaves half a file
    temporary.write_bytes(pickle.dumps(result))
    os.replace(temporary, path)
    return result

def summary(events: pd.DataFrame, hours: pd.DataFrame, by=("horizon", "season", "state")) -> pd.DataFrame:
    """Scores per group. 'season' comes from each row's IST month; other keys are columns."""
    keys = ["all", *by]
    ev = events.assign(season=season_of(events["time_utc"]), all="all")
    rows = []
    for key, part in ev.groupby(keys):
        stats = summarize(part[part["kind"] == "observed"], extra=int((part["kind"] == "extra").sum()))
        rows.append({**dict(zip(keys, key)), **stats})
    table = pd.DataFrame(rows)
    level_keys = [k for k in keys if k != "state"]
    if hours.empty:
        table["hourly_rmse_m"] = np.nan
    else:
        squared = hours.assign(season=season_of(hours["time_utc"]), all="all",
                               sq=(hours["predicted_m"] - hours["observed_m"]) ** 2)
        rmse = np.sqrt(squared.groupby(level_keys)["sq"].mean()).rename("hourly_rmse_m").reset_index()
        table = table.merge(rmse, on=level_keys, how="left")
    return table.drop(columns="all")

def joint_pct(events: pd.DataFrame) -> float:
    observed = events[events["kind"] == "observed"]
    return 100.0 * observed["hit"].sum() / len(observed) if len(observed) else float("nan")

MATCH_KEYS = ["origin", "test_year", "horizon", "state", "observed_time_utc"]

def bootstrap_diff(base: pd.DataFrame, cand: pd.DataFrame, metric: str, n: int = 1000, seed: int = 0) -> dict:
    """Candidate minus base on the observed events both scored, with a 95% interval from
    resampling calendar months (spec 5.4). joint: percentage points; *_mae: minutes or metres."""
    both = (base[base["kind"] == "observed"].set_index(MATCH_KEYS)
            .join(cand[cand["kind"] == "observed"].set_index(MATCH_KEYS), how="inner", lsuffix="_base", rsuffix="_cand"))
    if metric == "joint":
        value, scale = both["hit_cand"].astype(float) - both["hit_base"].astype(float), 100.0
    else:
        column = {"time_mae": "time_error_minutes", "height_mae": "height_error_m"}[metric]
        both = both.dropna(subset=[f"{column}_base", f"{column}_cand"])
        value, scale = both[f"{column}_cand"].abs() - both[f"{column}_base"].abs(), 1.0
    if both.empty:
        return {"diff": float("nan"), "lo": float("nan"), "hi": float("nan")}
    months = pd.DatetimeIndex(both.index.get_level_values("observed_time_utc")).tz_convert(IST).strftime("%Y-%m")
    sums = pd.DataFrame({"value": value.to_numpy(dtype=float), "count": 1.0}).groupby(np.asarray(months)).sum()
    draws = np.random.default_rng(seed).integers(0, len(sums), size=(n, len(sums)))
    stats = scale * sums["value"].to_numpy()[draws].sum(axis=1) / sums["count"].to_numpy()[draws].sum(axis=1)
    return {"diff": float(scale * sums["value"].sum() / sums["count"].sum()),
            "lo": float(np.percentile(stats, 2.5)), "hi": float(np.percentile(stats, 97.5))}

COVERAGE_BAND = (88.0, 92.0)

def promotion(base: pd.DataFrame, cand: pd.DataFrame, coverage: pd.DataFrame | None = None,
              tables: pd.DataFrame | None = None) -> dict:
    """Spec 5.7 on the final folds, against the base (the current pipeline):
    1. joint share better, with the 95% interval of the difference above zero
    2. timing and height MAE not worse beyond their intervals
    3. no season worse by more than 1 point
    4. 90% range coverage within 88-92% for every output (ranges.coverage rows; fails without them)
    Also spec 1.2: beat the official tables (score_fixed rows) in every year they overlap, and
    publish only horizons within 2 points of one year ahead whose coverage is also in the band."""
    b, c = base[base["final"].astype(bool)], cand[cand["final"].astype(bool)]
    joint = bootstrap_diff(b, c, "joint")
    time_mae = bootstrap_diff(b, c, "time_mae")
    height_mae = bootstrap_diff(b, c, "height_mae")
    seasons = {}
    for season in SEASONS:
        in_b, in_c = b[season_of(b["time_utc"]) == season], c[season_of(c["time_utc"]) == season]
        if (in_b["kind"] == "observed").any() and (in_c["kind"] == "observed").any():
            seasons[season] = joint_pct(in_c) - joint_pct(in_b)
    by_horizon = {int(h): joint_pct(part) for h, part in c.groupby("horizon")}
    first = by_horizon.get(1)
    publishable = [h for h, score in by_horizon.items()
                   if first is not None and score >= first - 2.0 and _covered(coverage, h)]
    versus_tables = {}
    for year, official in (tables.groupby("test_year") if tables is not None and len(tables) else []):
        mine = cand[(cand["test_year"] == year) & (cand["horizon"] == 1)]
        if (mine["kind"] == "observed").any():
            versus_tables[int(year)] = joint_pct(mine) - joint_pct(official)
    checks = {"joint_better": bool(joint["lo"] > 0),
              "time_mae_not_worse": bool(time_mae["lo"] <= 0),
              "height_mae_not_worse": bool(height_mae["lo"] <= 0),
              "no_season_worse_than_1pp": all(v >= -1.0 for v in seasons.values()),
              "coverage_88_92": _covered(coverage, "all"),
              "beats_tables": all(v > 0 for v in versus_tables.values())}
    return {"passed": all(checks.values()), "checks": checks, "joint": joint, "time_mae": time_mae,
            "height_mae": height_mae, "season_diff_pp": seasons, "joint_by_horizon": by_horizon,
            "publishable_horizons": publishable, "versus_tables_pp": versus_tables}

def _covered(coverage: pd.DataFrame | None, horizon) -> bool:
    """Whether every output's coverage at this horizon ('all' pools the horizons) lies in the band."""
    if coverage is None:
        return False
    part = coverage[coverage["horizon"] == horizon]
    return len(part) > 0 and bool(part["covered_pct"].between(*COVERAGE_BAND).all())

def score_fixed(port_slug: str, predicted: pd.DataFrame) -> pd.DataFrame:
    """Event rows for predictions made outside the backtest (the official tables), scored on
    every IST year they share with the QC-passed gauge, within the span they cover."""
    if predicted.empty:
        return pd.DataFrame()
    gauge = load_gauge(port_slug, passed_only=True)
    rows = []
    for year in sorted(set(ist_years(predicted["time_utc"])) & set(ist_years(gauge["time_utc"]))):
        start, end = ist_year_start(year), ist_year_start(year + 1)
        inside = predicted[(predicted["time_utc"] >= start) & (predicted["time_utc"] < end)].reset_index(drop=True)
        observed = gauge[(gauge["time_utc"] >= start) & (gauge["time_utc"] < end)]
        truth = find_events(observed["time_utc"], observed["height_m"], step_minutes=60)
        truth = truth[truth["time_utc"].between(inside["time_utc"].min(), inside["time_utc"].max())]
        matches = match_events(truth, inside)
        extras = extra_events(inside, matches, coverage(observed["time_utc"]))
        tag = {"origin": -1, "test_year": year, "horizon": 0, "final": year > SELECTION_LAST_YEAR}
        rows += [matches.assign(kind="observed", time_utc=matches["observed_time_utc"], **tag),
                 extras.assign(kind="extra", hit=False, **tag)]
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

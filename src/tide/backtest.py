"""Rolling-origin yearly backtest (spec 5): folds, cached parallel scoring and summaries."""
from __future__ import annotations

import hashlib
import os
import pickle
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config

from tide.events import coverage, extra_events, find_events, match_events, model_events, summarize
from tide.ports import IST, ROOT, ist_year_start, ist_years, season_of
from tide.store import load_gauge

CACHE_DIR = ROOT / ".cache" / "backtest"
SELECTION_LAST_YEAR = 2019  # moves forward with the final set when new gauge years arrive (spec 5.2)
ORIGINS = {"haldia": list(range(2008, 2025)), "diamond_harbour": list(range(2008, 2017)) + [2021]}
HORIZONS = 3
RESULT_MODULES = ("ports", "store", "harmonic", "events", "backtest", "candidates")

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
    cache = CACHE_DIR / port_slug / _cache_key(gauge) / candidate.name
    jobs = [delayed(_run_origin)(port_slug, candidate, fs, gauge, cache) for _, fs in sorted(by_origin.items())]
    threads = max(1, (os.cpu_count() or 1) // max(1, n_jobs))
    with parallel_config(backend="loky", inner_max_num_threads=threads):
        parts = Parallel(n_jobs=n_jobs)(jobs)
    return (pd.concat([p[0] for p in parts], ignore_index=True),
            pd.concat([p[1] for p in parts], ignore_index=True))

def _cache_key(gauge: pd.DataFrame) -> str:
    digest = hashlib.sha256()
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
    path.write_bytes(pickle.dumps(result))
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

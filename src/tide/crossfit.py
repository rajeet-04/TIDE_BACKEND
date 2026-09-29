"""Cross-fitted model A: the material model B learns from (spec 4.3).

For each training year, A's hourly curve and 1-minute events come from a fit that left that
year out, so B learns from errors as large as A's real forecast errors. Results are cached
under .cache/crossfit, keyed by the readings, the settings, the code and library versions.
"""
from __future__ import annotations

import hashlib
import os
import pickle
from dataclasses import asdict, dataclass
from importlib.metadata import version

import pandas as pd

from tide.events import find_events, match_events, model_events
from tide.features import event_features, ist_hour_grid, level_features
from tide.harmonic import HarmonicConfig, HarmonicModel, leave_one_year_out
from tide.ports import ROOT, ist_year_start, ist_years, port

CACHE_DIR = ROOT / ".cache" / "crossfit"
EVENT_PAD = pd.Timedelta(hours=13)   # events on each side of a year, so edge events have neighbours
SOURCES = ("harmonic", "events", "crossfit")

@dataclass
class CrossFit:
    model: HarmonicModel     # A fitted on every training reading
    observed: pd.DataFrame   # the training readings: time_utc, height_m
    curves: pd.DataFrame     # year, time_utc, level_m: the year's hourly curve (whole IST hours ± 40 h), fitted without it
    events: pd.DataFrame     # year, state, time_utc, height_m: the year's 1-minute events, fitted without it

    def level_table(self) -> pd.DataFrame:
        """One row per training reading: LEVEL_FEATURES from its year's cross-fitted curve,
        target (observed minus that curve) and year. Years without readings add no rows."""
        years = ist_years(self.observed["time_utc"])
        parts = []
        for year, curve in self.curves.groupby("year"):
            grid = pd.DatetimeIndex(curve["time_utc"])
            rows = level_features(grid, curve["level_m"].to_numpy())
            rows.index = grid
            observed = self.observed[years == year]
            rows = rows.reindex(pd.DatetimeIndex(observed["time_utc"]))
            rows["target"] = observed["height_m"].to_numpy() - rows["level"].to_numpy()
            rows["year"] = year
            parts.append(rows.dropna())
        return pd.concat(parts, ignore_index=True)

    def event_table(self) -> pd.DataFrame:
        """One row per observed event matched to its year's cross-fitted A event: EVENT_FEATURES
        of that A event, its errors (predicted minus observed) and the year."""
        years = ist_years(self.observed["time_utc"])
        parts = []
        for year, predicted in self.events.groupby("year"):
            observed = self.observed[years == year]
            truth = find_events(observed["time_utc"], observed["height_m"], step_minutes=60)
            predicted = predicted.drop(columns="year").sort_values("time_utc", kind="stable").reset_index(drop=True)
            matches = match_events(truth, predicted).dropna(subset=["predicted_index"])
            rows = event_features(predicted).iloc[matches["predicted_index"].astype(int).to_numpy()].reset_index(drop=True)
            rows["time_error_minutes"] = matches["time_error_minutes"].to_numpy()
            rows["height_error_m"] = matches["height_error_m"].to_numpy()
            rows["year"] = year
            parts.append(rows)
        return pd.concat(parts, ignore_index=True)

def cross_fit(train: pd.DataFrame, port_slug: str, config: HarmonicConfig) -> CrossFit:
    """A fitted on the training readings (time_utc, height_m), with each year's curve and events
    from a refit without that year. Needs readings from at least two IST years."""
    path = CACHE_DIR / port_slug / f"{_key(train, port_slug, config)}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    observed = train[["time_utc", "height_m"]].reset_index(drop=True)
    times = pd.DatetimeIndex(observed["time_utc"])
    heights = observed["height_m"].to_numpy(dtype=float)
    model = HarmonicModel(port(port_slug).lat, config).fit(times, heights)
    curves, events = [], []
    for year, coef in sorted(leave_one_year_out(model, times, heights, ist_years(times)).items()):
        refit = model.with_coef(coef)
        start, end = ist_year_start(year), ist_year_start(year + 1)
        grid = ist_hour_grid(start, end)
        curves.append(pd.DataFrame({"year": year, "time_utc": grid, "level_m": refit.predict(grid)}))
        events.append(model_events(refit.predict, start - EVENT_PAD, end + EVENT_PAD).assign(year=year))
    result = CrossFit(model, observed, pd.concat(curves, ignore_index=True), pd.concat(events, ignore_index=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_bytes(pickle.dumps(result))
    os.replace(temporary, path)
    return result

def _key(train: pd.DataFrame, port_slug: str, config: HarmonicConfig) -> str:
    digest = hashlib.sha256()
    for library in ("numpy", "pandas", "scipy", "utide"):
        digest.update(f"{library}={version(library)}".encode())
    for module in SOURCES:
        digest.update((ROOT / "src" / "tide" / f"{module}.py").read_bytes())
    digest.update(repr((port_slug, asdict(config))).encode())
    digest.update(pd.util.hash_pandas_object(train[["time_utc", "height_m"]], index=False).to_numpy().tobytes())
    return digest.hexdigest()[:16]

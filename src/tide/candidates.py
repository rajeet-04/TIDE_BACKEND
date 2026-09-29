"""Backtest candidates: model A settings and the baselines (spec 4.2, 5.5)."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from tide.backtest import FittedLevels
from tide.harmonic import HarmonicConfig, HarmonicModel
from tide.ports import IST, port

def window(train: pd.DataFrame, years: int | None) -> pd.DataFrame:
    """The last `years` years of training readings (all of them when None)."""
    if years is None or train.empty:
        return train
    return train[train["time_utc"] > train["time_utc"].max() - pd.DateOffset(years=years)]

@dataclass(frozen=True)
class HarmonicCandidate:
    config: HarmonicConfig

    @property
    def name(self) -> str:
        return self.config.name

    def fit(self, train: pd.DataFrame, port_slug: str) -> FittedLevels:
        recent = window(train, self.config.window_years)
        model = HarmonicModel(port(port_slug).lat, self.config).fit(recent["time_utc"], recent["height_m"])
        return FittedLevels(model.predict)

def _naive_ist(times) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(times).tz_convert(IST).tz_localize(None)

def _utide_fit(train: pd.DataFrame, port_slug: str):
    """The current pipeline's UTide solve, on naive IST times as it has always run."""
    from utide import solve

    history = pd.DataFrame({"datetime_ist": _naive_ist(train["time_utc"]), "height_m": train["height_m"].to_numpy()})
    coef = solve(history["datetime_ist"], history["height_m"].to_numpy(), lat=port(port_slug).lat,
                 constit="auto", method="ols", conf_int="none", trend=True, nodal=True, verbose=False)
    return history, coef

class UTideOnly:
    """The current pipeline's UTide fit without its event calibration."""
    name = "utide_only"

    def fit(self, train: pd.DataFrame, port_slug: str) -> FittedLevels:
        from utide_event_model import _reconstruct_chunked

        _, coef = _utide_fit(train, port_slug)
        return FittedLevels(lambda times: _reconstruct_chunked(_naive_ist(times), coef))

class CurrentPipeline:
    """UTide plus ExtraTrees event calibration (src/utide_event_model.py), refit in every fold."""
    name = "current_pipeline"

    def fit(self, train: pd.DataFrame, port_slug: str) -> "_CalibratedEvents":
        from utide_event_model import _historical_calibration, _reconstruct_chunked

        history, coef = _utide_fit(train, port_slug)
        time_cal, height_cal, _ = _historical_calibration(history, _reconstruct_chunked(history["datetime_ist"], coef))
        return _CalibratedEvents({"coefficients": coef, "time_calibrator": time_cal, "height_calibrator": height_cal})

@dataclass
class _CalibratedEvents:
    model: dict

    def levels(self, times):
        from utide_event_model import _reconstruct_chunked

        return _reconstruct_chunked(_naive_ist(times), self.model["coefficients"])

    def events(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        from utide_event_model import predict_events

        begin, finish = (pd.Timestamp(t).tz_convert(IST).tz_localize(None).to_pydatetime() for t in (start, end))
        ev = predict_events(self.model, begin, finish)
        times = pd.DatetimeIndex(ev["datetime_ist"]).tz_localize(IST).tz_convert("UTC")
        return pd.DataFrame({"state": ev["state"].to_numpy(), "time_utc": times, "height_m": ev["height_m"].to_numpy()})

def candidate(name: str):
    """A candidate by name: 'current_pipeline', 'utide_only', a model A name such as
    A-w8-auto-side-trend, or model A with B's corrections such as
    A-wall-auto+shallow-side-notrend/lv:lgbm-l31-m100/ev:lgbm-l31-m40."""
    if name == "current_pipeline":
        return CurrentPipeline()
    if name == "utide_only":
        return UTideOnly()
    if "/" in name:
        from tide.stack import StackCandidate, StackConfig

        return StackCandidate(StackConfig.from_name(name))
    return HarmonicCandidate(HarmonicConfig.from_name(name))

"""Model A with model B's corrections, as one candidate (spec 4.1-4.3).

- Level correction: a learner predicts A's hourly residual from LEVEL_FEATURES. It is
  evaluated on whole IST hours and joined by a cubic spline, so every step down to one
  minute sees the same smooth curve.
- Event correction: two learners predict the timing and height errors of A's one-minute
  events from EVENT_FEATURES, and each event is moved by them (the legacy calibrator's idea).

Both learn from cross-fitted material (tide.crossfit). Planning checks on nine selection
fold-years found the event correction the strongest path for events and the level correction
the best for hourly levels, and that training the event correction on level-corrected events
makes it worse; so events always start from A's own curve, and the two are fitted independently.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

from tide.candidates import window
from tide.crossfit import EVENT_PAD, cross_fit
from tide.events import EVENT_COLUMNS, model_events
from tide.features import EVENT_FEATURES, LEVEL_FEATURES, event_features, ist_hour_grid, level_features
from tide.harmonic import HarmonicConfig, HarmonicModel, datenum
from tide.learners import make_learner
from tide.ports import port

@dataclass(frozen=True)
class StackConfig:
    a: HarmonicConfig
    level: str | None = None   # learner for the hourly-residual correction
    event: str | None = None   # learner for the event timing and height corrections

    @property
    def name(self) -> str:
        """For example A-wall-auto+shallow-side-notrend/lv:lgbm-l31-m100/ev:lgbm-l31-m40."""
        tags = [f"{tag}:{learner}" for tag, learner in (("lv", self.level), ("ev", self.event)) if learner]
        return "/".join([self.a.name, *tags])

    @classmethod
    def from_name(cls, name: str) -> "StackConfig":
        head, *parts = name.split("/")
        found: dict[str, str | None] = {"lv": None, "ev": None}
        for part in parts:
            tag, _, learner = part.partition(":")
            if tag not in found or not learner or found[tag]:
                raise ValueError(f"not a stack name: {name!r}")
            found[tag] = learner
        return cls(HarmonicConfig.from_name(head), level=found["lv"], event=found["ev"])

@dataclass(frozen=True)
class StackCandidate:
    config: StackConfig

    @property
    def name(self) -> str:
        return self.config.name

    def fit(self, train: pd.DataFrame, port_slug: str) -> "FittedStack":
        recent = window(train, self.config.a.window_years)
        if not (self.config.level or self.config.event):
            return FittedStack(HarmonicModel(port(port_slug).lat, self.config.a).fit(recent["time_utc"], recent["height_m"]))
        material = cross_fit(recent, port_slug, self.config.a)
        level = event = None
        if self.config.level:
            table = material.level_table()
            level = make_learner(self.config.level).fit(table[LEVEL_FEATURES], table["target"].to_numpy())
        if self.config.event:
            table = material.event_table()
            event = tuple(make_learner(self.config.event).fit(table[EVENT_FEATURES], table[target].to_numpy())
                          for target in ("time_error_minutes", "height_error_m"))
        return FittedStack(material.model, level, event)

@dataclass
class FittedStack:
    model: HarmonicModel
    level: object | None = None   # fitted learner for the hourly residual
    event: tuple | None = None    # fitted (timing, height) learners

    def correction(self, times) -> np.ndarray:
        """The level correction (m) at the times; zeros without one."""
        times = pd.DatetimeIndex(times)
        if self.level is None or len(times) == 0:
            return np.zeros(len(times))
        knots = ist_hour_grid(times.min(), times.max())
        values = self.level.predict(level_features(knots, self.model.predict(knots))[LEVEL_FEATURES])
        return CubicSpline(datenum(knots), values)(datenum(times))

    def levels(self, times) -> np.ndarray:
        return self.model.predict(times) + self.correction(times)

    def components(self, times) -> pd.DataFrame:
        """height_m = astronomical_m + seasonal_m + correction_m, indexed by time."""
        parts = self.model.predict(times, components=True)
        correction = self.correction(times)
        return parts.assign(height_m=parts["height_m"].to_numpy() + correction, correction_m=correction)

    def event_detail(self, start, end) -> pd.DataFrame:
        """High and low waters in [start, end): state, time_utc, height_m, then the raw event
        (raw_time_utc, raw_height_m) and the corrections applied to it
        (time = raw - time_correction_minutes; height = raw - height_correction_m)."""
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        if self.event is None:
            found = model_events(self.levels, start, end)
            return found.assign(raw_time_utc=found["time_utc"], raw_height_m=found["height_m"],
                                time_correction_minutes=0.0, height_correction_m=0.0)
        raw = model_events(self.model.predict, start - EVENT_PAD, end + EVENT_PAD)
        if raw.empty:
            return raw.assign(raw_time_utc=raw["time_utc"], raw_height_m=raw["height_m"],
                              time_correction_minutes=0.0, height_correction_m=0.0)
        features = event_features(raw)[EVENT_FEATURES]
        minutes, metres = (learner.predict(features) for learner in self.event)
        detail = pd.DataFrame({"state": raw["state"], "time_utc": raw["time_utc"] - pd.to_timedelta(minutes, unit="min"),
                               "height_m": raw["height_m"].to_numpy() - metres, "raw_time_utc": raw["time_utc"],
                               "raw_height_m": raw["height_m"], "time_correction_minutes": minutes,
                               "height_correction_m": metres})
        inside = (detail["time_utc"] >= start) & (detail["time_utc"] < end)
        return detail[inside].reset_index(drop=True)

    def events(self, start, end) -> pd.DataFrame:
        return self.event_detail(start, end)[EVENT_COLUMNS]

    def save(self, folder: Path) -> None:
        """A's constants as JSON (a.json), and the learners with joblib when present."""
        (folder / "a.json").write_text(json.dumps(self.model.to_dict()) + "\n")
        if self.level is not None:
            joblib.dump(self.level, folder / "level.joblib")
        if self.event is not None:
            joblib.dump(self.event, folder / "event.joblib")

    @classmethod
    def load(cls, folder: Path) -> "FittedStack":
        """Loads what save wrote; a folder with only a.json (plan 1a versions) loads as A alone."""
        model = HarmonicModel.from_dict(json.loads((folder / "a.json").read_text()))
        level = joblib.load(folder / "level.joblib") if (folder / "level.joblib").exists() else None
        event = joblib.load(folder / "event.joblib") if (folder / "event.joblib").exists() else None
        return cls(model, level, event)

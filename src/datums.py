"""Phase 5: Tidal datums from a long predicted series (peak approach).

Datums computed:
  HAT   - Highest Astronomical Tide  (max over series)
  MHWS  - Mean High Water Springs    (mean of highs in the top 10%)
  MHW   - Mean High Water            (mean of all daily highs)
  MSL   - Mean Sea Level             (mean of series)
  MLW   - Mean Low Water             (mean of all daily lows)
  MLWS  - Mean Low Water Springs     (mean of lows in the bottom 10%)
  LAT   - Lowest Astronomical Tide   (min over series)
  RA    - Tidal Range                (HAT - LAT)

The paper computes spring values by averaging at spring tide timestamps; we use
the equivalent percentile approach which is robust without needing explicit
moon-phase data. Both yield very similar magnitudes because spring tides
generate the largest peaks.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Datums:
    HAT: float
    MHWS: float
    MHW: float
    MSL: float
    MLW: float
    MLWS: float
    LAT: float
    RA: float

    def as_dict(self) -> dict[str, float]:
        return self.__dict__.copy()


def _daily_extrema(t_hours: np.ndarray, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (daily_max, daily_min) arrays. Assumes hourly samples."""
    days = (t_hours // 24).astype(np.int64)
    uniq = np.unique(days)
    highs = np.empty(uniq.size)
    lows = np.empty(uniq.size)
    for i, d in enumerate(uniq):
        mask = days == d
        highs[i] = h[mask].max()
        lows[i] = h[mask].min()
    return highs, lows


def compute(t_hours: np.ndarray, h: np.ndarray, spring_pct: float = 10.0) -> Datums:
    daily_high, daily_low = _daily_extrema(t_hours, h)
    spring_high_thresh = np.percentile(daily_high, 100 - spring_pct)
    spring_low_thresh = np.percentile(daily_low, spring_pct)
    mhws = float(daily_high[daily_high >= spring_high_thresh].mean())
    mlws = float(daily_low[daily_low <= spring_low_thresh].mean())
    return Datums(
        HAT=float(h.max()),
        MHWS=mhws,
        MHW=float(daily_high.mean()),
        MSL=float(h.mean()),
        MLW=float(daily_low.mean()),
        MLWS=mlws,
        LAT=float(h.min()),
        RA=float(h.max() - h.min()),
    )

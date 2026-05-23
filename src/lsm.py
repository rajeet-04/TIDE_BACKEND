"""Phase 2: Least Squares Method for tidal harmonic analysis.

Given an observed series h_m at times t_m (hours from epoch) and a list of K
constituents with angular speeds omega_i (deg/h), solve:

    h(t) = h_o + sum_i [A_i cos(omega_i t) + B_i sin(omega_i t)]

for the 2K+1 unknowns {h_o, A_i, B_i}. Then recover amplitude H_i and phase g_i
via H_i = sqrt(A_i^2 + B_i^2), g_i = arctan2(B_i, A_i).

We solve via numpy.linalg.lstsq for numerical stability (paper uses Gauss-Jordan
on the normal equations which is equivalent).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from constituents import CONSTITUENTS, Constituent


@dataclass
class HarmonicFit:
    mean_level: float                     # h_o
    constituents: list[Constituent]       # in order
    amplitudes: np.ndarray                # H_i (m), shape (K,)
    phases_deg: np.ndarray                # g_i (deg), shape (K,)
    cos_coef: np.ndarray                  # A_i
    sin_coef: np.ndarray                  # B_i


def _design_matrix(t_hours: np.ndarray, constituents: list[Constituent]) -> np.ndarray:
    """Build [1, cos(w1 t), sin(w1 t), cos(w2 t), sin(w2 t), ...]."""
    n = t_hours.size
    k = len(constituents)
    X = np.empty((n, 1 + 2 * k), dtype=float)
    X[:, 0] = 1.0
    for i, c in enumerate(constituents):
        omega_rad = np.deg2rad(c.speed_deg_per_hour)
        X[:, 1 + 2 * i] = np.cos(omega_rad * t_hours)
        X[:, 2 + 2 * i] = np.sin(omega_rad * t_hours)
    return X


def fit(
    t_hours: np.ndarray,
    h: np.ndarray,
    constituents: list[Constituent] | None = None,
) -> HarmonicFit:
    """Solve LSM for the given observation times and heights."""
    if constituents is None:
        constituents = list(CONSTITUENTS)

    X = _design_matrix(t_hours, constituents)
    coef, _residuals, _rank, _sv = np.linalg.lstsq(X, h, rcond=None)
    h_o = float(coef[0])
    k = len(constituents)
    A = coef[1::2][:k]
    B = coef[2::2][:k]
    H = np.sqrt(A**2 + B**2)
    # Phase in degrees, in [0, 360)
    g = np.rad2deg(np.arctan2(B, A)) % 360.0
    return HarmonicFit(
        mean_level=h_o,
        constituents=constituents,
        amplitudes=H,
        phases_deg=g,
        cos_coef=A,
        sin_coef=B,
    )


def predict(t_hours: np.ndarray, fit_result: HarmonicFit) -> np.ndarray:
    """Predict tidal heights at given times (hours from same epoch as fit)."""
    out = np.full_like(t_hours, fit_result.mean_level, dtype=float)
    for i, c in enumerate(fit_result.constituents):
        omega_rad = np.deg2rad(c.speed_deg_per_hour)
        out += fit_result.cos_coef[i] * np.cos(omega_rad * t_hours)
        out += fit_result.sin_coef[i] * np.sin(omega_rad * t_hours)
    return out

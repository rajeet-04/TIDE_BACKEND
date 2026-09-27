"""Model A: harmonic core (spec 4.2).

Weighted least squares on hourly heights. Constituents carry UTide's nodal corrections
(f, u) and astronomical arguments (V). All times are UTC, so phases are Greenwich phases.
Optional terms:
- mean-level seasonal cycles (Sa, Ssa, Sta)
- seasonal side-terms (±1 cycle per year on the main constituents)
- a linear trend
Robust fitting is bisquare IRLS with a small ridge penalty.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from utide._ut_constants import constit_index_dict, ut_constants
from utide.harmonics import FUV, linearized_freqs

CONST = ut_constants.const
UNIX_DATENUM = 719163.0      # UTide datenum (days since 0000-12-31) of 1970-01-01
EPOCH_DATENUM = 730120.0     # 2000-01-01 00:00 UTC: trend and seasonal phase count from here
TROPICAL_YEAR_DAYS = 365.2422
SEASONAL_NAMES = ("Sa", "Ssa", "Sta")
SIDE_CARRIERS = ("M2", "S2", "N2", "K1", "O1", "M4", "MS4")
# Shallow-water constituents in UTide's table that its automatic selection never picks.
EXTRA_SHALLOW = (
    "2PO1", "2NS2", "O2", "SNK2", "OP2", "2SK2", "2SM2", "SKM2", "2SN2", "NO3", "NK3", "SP3",
    "N4", "3MS4", "KN4", "SL4", "MNO5", "2MO5", "3MP5", "MNK5", "2MP5", "MSK5", "3KM5",
    "2NM6", "MSN6", "MKN6", "NSK6", "S6", "M7", "3MN8", "3MS8", "3MK8", "4MK9", "M10", "M12",
)
BISQUARE_C = 4.685
CHUNK = 100_000

@dataclass(frozen=True)
class HarmonicConfig:
    window_years: int | None = None                # training years before the origin; None = all
    constituents: str | tuple[str, ...] = "auto"   # "auto", "auto+shallow" or explicit UTide names
    side_terms: bool = False
    trend: bool = True
    seasonal: int = 3                              # mean-level seasonal harmonics, 0-3
    robust: bool = True
    ridge: float = 1e-4                            # per-observation penalty on all columns but the mean

    @property
    def name(self) -> str:
        window = "all" if self.window_years is None else str(self.window_years)
        const = self.constituents if isinstance(self.constituents, str) else f"{len(self.constituents)}c"
        side = "side" if self.side_terms else "noside"
        trend = "trend" if self.trend else "notrend"
        return f"A-w{window}-{const}-{side}-{trend}"

    @classmethod
    def from_name(cls, name: str) -> "HarmonicConfig":
        parts = name.split("-")
        if len(parts) != 5 or parts[0] != "A" or not parts[1].startswith("w") or parts[2] not in ("auto", "auto+shallow"):
            raise ValueError(f"not a model A name: {name!r}")
        _, window, const, side, trend = parts
        return cls(window_years=None if window == "wall" else int(window[1:]), constituents=const,
                   side_terms=side == "side", trend=trend == "trend")

def datenum(times) -> np.ndarray:
    """UTide datenum (days since 0000-12-31) of timezone-aware times."""
    utc = pd.DatetimeIndex(times).tz_convert("UTC").tz_localize(None)
    return np.asarray((utc - pd.Timestamp("1970-01-01")) / pd.Timedelta(days=1), dtype=float) + UNIX_DATENUM

def select_constituents(spec, record_hours: float, tref: float) -> list[str]:
    """UTide's automatic set for the record length (Sa and Ssa are left to the seasonal
    terms), optionally plus the extra shallow-water constituents the record resolves."""
    if not isinstance(spec, str):
        return list(spec)
    if spec not in ("auto", "auto+shallow"):
        raise ValueError(f"unknown constituent set {spec!r}")
    minres = 1.0 / record_hours
    with np.errstate(invalid="ignore"):
        chosen = CONST.df >= minres
    names = [str(n) for n in CONST.name[chosen] if n not in ("SA", "SSA")]
    if spec == "auto+shallow":
        freqs = linearized_freqs(tref)
        taken = [freqs[constit_index_dict[n]] for n in names]
        for name in EXTRA_SHALLOW:
            f = freqs[constit_index_dict[name]]
            if min(abs(f - t) for t in taken) >= minres:
                names.append(name)
                taken.append(f)
    return names

class HarmonicModel:
    """h(t) = Z0 [+ trend] + seasonal cycles + Σ f(t)·(a cos + b sin)(u + V) [+ side-terms]."""

    def __init__(self, lat: float, config: HarmonicConfig = HarmonicConfig()):
        if not 0 <= config.seasonal <= len(SEASONAL_NAMES):
            raise ValueError("seasonal must be between 0 and 3")
        self.lat = lat
        self.config = config
        self.names: list[str] = []
        self.side: list[tuple[str, int]] = []
        self.columns: list[str] = []
        self.coef: np.ndarray | None = None

    def fit(self, times, heights) -> "HarmonicModel":
        dn = datenum(times)
        y = np.asarray(heights, dtype=float)
        if len(y) != len(dn) or len(y) < 10 or not np.isfinite(y).all():
            raise ValueError("need at least 10 finite heights, one per time")
        self._prepare(dn)
        X = np.vstack([self._design(dn[i:i + CHUNK]) for i in range(0, len(dn), CHUNK)])
        self.coef = _irls(X, y, self.config.ridge, self.config.robust)
        return self

    def predict(self, times, components: bool = False):
        """Heights at the times. With components, a frame that also splits them into
        astronomical_m (constituents and side-terms) and seasonal_m (mean, trend, seasonal cycles)."""
        if self.coef is None:
            raise RuntimeError("fit the model first")
        dn = datenum(times)
        n_mean = 1 + int(self.config.trend) + 2 * self.config.seasonal
        height = np.empty(len(dn))
        seasonal = np.empty(len(dn))
        for i in range(0, len(dn), CHUNK):
            X = self._design(dn[i:i + CHUNK])
            height[i:i + CHUNK] = X @ self.coef
            seasonal[i:i + CHUNK] = X[:, :n_mean] @ self.coef[:n_mean]
        if not components:
            return height
        return pd.DataFrame({"height_m": height, "astronomical_m": height - seasonal, "seasonal_m": seasonal},
                            index=pd.DatetimeIndex(times))

    def constants(self) -> pd.DataFrame:
        """Amplitude (m) and Greenwich phase (degrees) of each constituent."""
        if self.coef is None:
            raise RuntimeError("fit the model first")
        where = {c: i for i, c in enumerate(self.columns)}
        a = np.array([self.coef[where[f"{n}_cos"]] for n in self.names])
        b = np.array([self.coef[where[f"{n}_sin"]] for n in self.names])
        return pd.DataFrame({"amplitude_m": np.hypot(a, b), "phase_deg": np.degrees(np.arctan2(b, a)) % 360},
                            index=pd.Index(self.names, name="name"))

    @property
    def mean_level(self) -> float:
        return float(self.coef[0])

    def to_dict(self) -> dict:
        return {"kind": "harmonic", "lat": self.lat, "config": asdict(self.config), "names": self.names,
                "side": [list(s) for s in self.side], "columns": self.columns, "coef": self.coef.tolist()}

    @classmethod
    def from_dict(cls, data: dict) -> "HarmonicModel":
        config = dict(data["config"])
        if not isinstance(config["constituents"], str):
            config["constituents"] = tuple(config["constituents"])
        model = cls(data["lat"], HarmonicConfig(**config))
        model.names = list(data["names"])
        model.side = [(carrier, int(sign)) for carrier, sign in data["side"]]
        model.columns = list(data["columns"])
        model.coef = np.asarray(data["coef"], dtype=float)
        return model

    def _prepare(self, dn: np.ndarray) -> None:
        record_hours = max((dn.max() - dn.min()) * 24.0, 1.0)
        tref = 0.5 * (dn.max() + dn.min())
        names = select_constituents(self.config.constituents, record_hours, tref)
        self.names = [n for n in names if not (self.config.seasonal and n in ("SA", "SSA"))]
        self.side = self._side_terms(record_hours, tref) if self.config.side_terms else []
        self.columns = self._column_names()

    def _side_terms(self, record_hours: float, tref: float) -> list[tuple[str, int]]:
        """(carrier, ±1) pairs whose frequency is not already resolved by a chosen constituent
        (UTide's set already holds H1/H2 = M2∓Sa, T2/R2 = S2∓Sa, S1/PSI1 = K1∓Sa)."""
        freqs = linearized_freqs(tref)
        taken = [freqs[constit_index_dict[n]] for n in self.names]
        annual = 1.0 / (TROPICAL_YEAR_DAYS * 24.0)
        side = []
        for carrier in SIDE_CARRIERS:
            if carrier not in self.names:
                continue
            for sign in (1, -1):
                f = freqs[constit_index_dict[carrier]] + sign * annual
                if min(abs(f - t) for t in taken) >= 1.0 / record_hours:
                    side.append((carrier, sign))
                    taken.append(f)
        return side

    def _column_names(self) -> list[str]:
        cols = ["Z0"] + (["trend"] if self.config.trend else [])
        for k in range(self.config.seasonal):
            cols += [f"{SEASONAL_NAMES[k]}_cos", f"{SEASONAL_NAMES[k]}_sin"]
        for n in self.names:
            cols += [f"{n}_cos", f"{n}_sin"]
        for carrier, sign in self.side:
            label = f"{carrier}{'+' if sign > 0 else '-'}Sa"
            cols += [f"{label}_cos", f"{label}_sin"]
        return cols

    def _nodal(self, dn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Nodal factor f and phase u + V (radians) per time and constituent.

        FUV runs once per whole hour; within the hour the phase advances at the
        constituent's frequency (error below 1e-5 rad), which keeps 1-minute curves cheap."""
        lind = [constit_index_dict[n] for n in self.names]
        knots = np.floor(dn * 24.0) / 24.0
        unique, inverse = np.unique(knots, return_inverse=True)
        F, U, V = FUV(unique, unique.mean(), lind, self.lat, [False, False, False, False])
        freq = linearized_freqs(unique.mean())[lind]  # cycles per hour
        phase = 2 * np.pi * ((U + V)[inverse] + np.outer((dn - knots) * 24.0, freq))
        return F[inverse], phase

    def _design(self, dn: np.ndarray) -> np.ndarray:
        if len(dn) == 0:
            return np.empty((0, len(self.columns)))
        years = (dn - EPOCH_DATENUM) / TROPICAL_YEAR_DAYS
        cols = [np.ones_like(dn)]
        if self.config.trend:
            cols.append(years)
        for k in range(1, self.config.seasonal + 1):
            cols += [np.cos(2 * np.pi * k * years), np.sin(2 * np.pi * k * years)]
        if self.names:
            F, phase = self._nodal(dn)
            for j in range(len(self.names)):
                cols += [F[:, j] * np.cos(phase[:, j]), F[:, j] * np.sin(phase[:, j])]
            where = {n: j for j, n in enumerate(self.names)}
            for carrier, sign in self.side:
                j = where[carrier]
                shifted = phase[:, j] + sign * 2 * np.pi * years
                cols += [F[:, j] * np.cos(shifted), F[:, j] * np.sin(shifted)]
        return np.column_stack(cols)

def _irls(X: np.ndarray, y: np.ndarray, ridge: float, robust: bool, max_iter: int = 30) -> np.ndarray:
    """Least squares, then bisquare reweighting (c = 4.685 MAD-scaled) until coefficients settle."""
    weights = np.ones(len(y))
    coef = _wls(X, y, weights, ridge)
    for _ in range(max_iter if robust else 0):
        resid = y - X @ coef
        scale = 1.4826 * np.median(np.abs(resid - np.median(resid)))
        if scale < 1e-9:
            break
        u = resid / (BISQUARE_C * scale)
        weights = np.where(np.abs(u) < 1.0, (1.0 - u**2) ** 2, 0.0)
        new = _wls(X, y, weights, ridge)
        converged = np.max(np.abs(new - coef)) < 1e-6
        coef = new
        if converged:
            break
    return coef

def _wls(X: np.ndarray, y: np.ndarray, weights: np.ndarray, ridge: float) -> np.ndarray:
    Xw = X * weights[:, None]
    gram = X.T @ Xw
    rest = np.arange(1, gram.shape[0])
    gram[rest, rest] += ridge * len(y)
    return np.linalg.solve(gram, Xw.T @ y)

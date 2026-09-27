"""Quality control of gauge readings (spec 3.3). Readings are flagged, never deleted."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter
from scipy.signal import find_peaks

from tide.harmonic import HarmonicConfig, HarmonicModel
from tide.ports import DATA_DIR, IST, OUTPUT_DIR, ist_years, port, to_utc
from tide.store import flags_path

RULES = ("range", "spike", "rate", "flat", "manual")
RANGE_MARGIN_M = 1.0
SPIKE_SD = 5.0
SPIKE_MIN_M = 0.5
RATE_FACTOR = 1.2
FLAT_RUN = 4
FLAT_TIDE_CHANGE_M = 0.3
CONTEXT = pd.Timedelta(hours=6)
QC_MODEL = HarmonicConfig(constituents="auto", seasonal=3, trend=True, robust=True)
REVIEW_COLUMNS = ["start", "end", "action", "reason", "evidence"]

def review_path(port_slug: str):
    return DATA_DIR / "qc" / f"{port_slug}_review.csv"

def load_review(port_slug: str) -> pd.DataFrame:
    """Manual decisions: start, end (naive times are IST), action (exclude or keep), reason, evidence."""
    path = review_path(port_slug)
    return pd.read_csv(path) if path.exists() else pd.DataFrame(columns=REVIEW_COLUMNS)

def cross_fitted_prediction(times: pd.DatetimeIndex, heights: np.ndarray, lat: float, usable: np.ndarray) -> np.ndarray:
    """Prediction for each IST year from a robust harmonic fit to all other years, so a
    year's truth is never judged by a model that saw it."""
    years = ist_years(times)
    prediction = np.full(len(times), np.nan)
    for year in np.unique(years):
        this = years == year
        train = usable & ~this
        model = HarmonicModel(lat, QC_MODEL).fit(times[train], heights[train])
        prediction[this] = model.predict(times[this])
    return prediction

def _tidal_phase(prediction: np.ndarray, times: pd.DatetimeIndex, grid: pd.DatetimeIndex) -> np.ndarray:
    """Phase class of each hour on the grid: whole hours since the predicted low water (0-13)
    times 10, plus the spring/neap tercile of the predicted 25-hour range (0-2)."""
    p = pd.Series(prediction, index=times).reindex(grid).interpolate(limit_area="inside")
    values = p.to_numpy()
    lows, _ = find_peaks(-np.nan_to_num(values, nan=float(np.nanmean(values))), distance=4)
    hours = np.arange(len(values))
    if len(lows) == 0:
        return np.zeros(len(values), dtype=int)
    last = lows[np.clip(np.searchsorted(lows, hours, side="right") - 1, 0, None)]
    since = np.clip(hours - last, 0, 13)
    window = p.rolling(25, center=True, min_periods=12)
    tercile = pd.qcut(window.max() - window.min(), 3, labels=False).fillna(1).to_numpy(dtype=int)
    return since * 10 + tercile

def _phase_spread(centred: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """Spread of the centred residual by phase class: the 90th percentile of its size over 1.645
    (the SD for Gaussian noise; it grows when 10% or more of a phase departs, as at a bore),
    taken as the largest over neighbouring classes (±1 hour, ±1 tercile) since events drift between them."""
    q90 = pd.Series(np.abs(centred)).groupby(phase).quantile(0.9) / 1.645
    table = np.full((14, 3), np.nan)
    table[q90.index // 10, q90.index % 10] = q90.to_numpy()
    table = maximum_filter(np.nan_to_num(table, nan=0.0), size=3, mode="nearest")
    return table[phase // 10, phase % 10]

def run_qc(gauge: pd.DataFrame, port_slug: str, review: pd.DataFrame | None = None) -> pd.DataFrame:
    """The gauge table plus qc_flag ('' when passed, else '|'-joined rule names) and qc_residual_m."""
    table = gauge.sort_values("time_utc").reset_index(drop=True)
    times = pd.DatetimeIndex(table["time_utc"])
    h = table["height_m"].to_numpy(dtype=float)
    flags = {rule: np.zeros(len(table), dtype=bool) for rule in RULES}

    # 1. impossible values
    positive = h[h > 0]
    low = np.quantile(positive, 1e-4) - RANGE_MARGIN_M
    high = np.quantile(positive, 1 - 1e-4) + RANGE_MARGIN_M
    flags["range"] = (h <= 0) | (h < low) | (h > high)

    # 2. spikes: residual minus its 7-hour median, against the larger of the spread of its 7-day
    #    neighbourhood and the spread at the same tidal phase (the spring flood onset is steep)
    prediction = cross_fitted_prediction(times, h, port(port_slug).lat, ~flags["range"])
    residual = h - prediction
    grid = pd.Series(np.where(flags["range"], np.nan, residual), index=times).asfreq("1h")
    local = grid - grid.rolling(7, center=True, min_periods=4).median()
    centred = local - local.rolling(169, center=True, min_periods=48).median()
    week = 1.4826 * centred.abs().rolling(169, center=True, min_periods=48).median()
    phase = pd.Series(_phase_spread(centred.to_numpy(), _tidal_phase(prediction, times, local.index)), index=local.index)
    spread = np.maximum(week, phase)
    spike = (local.abs() > SPIKE_SD * spread) & (local.abs() > SPIKE_MIN_M)
    flags["spike"] = spike.reindex(times, fill_value=False).to_numpy(dtype=bool)

    # 3. impossible rate of change: of the two readings, flag the one further from the prediction
    consecutive = np.diff(times.values) == np.timedelta64(1, "h")
    change = np.abs(np.diff(h))
    clean = ~(flags["range"] | flags["spike"])
    limit = RATE_FACTOR * np.quantile(change[consecutive & clean[:-1] & clean[1:]], 0.9999)
    jumps = np.flatnonzero(consecutive & (change > limit))
    distance = np.abs(residual)
    flags["rate"][np.where(distance[jumps] >= distance[jumps + 1], jumps, jumps + 1)] = True

    # 4. flat stretches while the tide should move
    run = np.cumsum(np.r_[True, (np.diff(h) != 0) | ~consecutive])
    by_run = pd.Series(prediction).groupby(run)
    moved = (by_run.transform("max") - by_run.transform("min")).to_numpy()
    flags["flat"] = (by_run.transform("size").to_numpy() >= FLAT_RUN) & (moved > FLAT_TIDE_CHANGE_M)

    # 5. manual decisions from the review file
    decisions = load_review(port_slug) if review is None else review
    for row in decisions.itertuples():
        inside = np.asarray((times >= to_utc(row.start)) & (times <= to_utc(row.end)))
        if row.action == "exclude":
            flags["manual"] |= inside
        elif row.action == "keep":
            for rule in RULES[:-1]:
                flags[rule] = flags[rule] & ~inside  # not in place: pandas 3 arrays are read-only
        else:
            raise ValueError(f"review action must be 'exclude' or 'keep', not {row.action!r}")

    names = np.array(RULES)
    stacked = np.column_stack([flags[rule] for rule in RULES])
    table["qc_flag"] = ["|".join(names[marks]) for marks in stacked]
    table["qc_residual_m"] = residual
    return table

def qc_report(table: pd.DataFrame) -> pd.DataFrame:
    """Readings, flagged count and share, and the count per rule, for each IST year."""
    frame = pd.DataFrame({"year": ist_years(table["time_utc"]), "flagged": (table["qc_flag"] != "").to_numpy()})
    for rule in RULES:
        frame[rule] = table["qc_flag"].str.contains(rule, regex=False).to_numpy()
    report = frame.groupby("year").agg(readings=("flagged", "size"), flagged=("flagged", "sum"),
                                       **{rule: (rule, "sum") for rule in RULES})
    report.insert(2, "flagged_pct", (100 * report["flagged"] / report["readings"]).round(3))
    return report.reset_index()

def review_list(table: pd.DataFrame) -> pd.DataFrame:
    """Flagged stretches (consecutive flagged hours) for manual inspection, times in IST.

    context_residual_m is the median residual of passed readings within 6 h. A large value
    with the same sign as mean_residual_m means the flagged hours ride on a genuine surge."""
    columns = ["start_ist", "end_ist", "hours", "rules", "min_height_m", "max_height_m",
               "mean_residual_m", "max_abs_residual_m", "context_residual_m"]
    flagged = table[table["qc_flag"] != ""]
    if flagged.empty:
        return pd.DataFrame(columns=columns)
    passed = table[table["qc_flag"] == ""]
    passed_times = pd.DatetimeIndex(passed["time_utc"])
    passed_resid = passed["qc_residual_m"].to_numpy()
    stretch = np.cumsum(np.r_[True, np.diff(pd.DatetimeIndex(flagged["time_utc"]).values) > np.timedelta64(1, "h")])
    rows = []
    for _, part in flagged.groupby(stretch):
        first, last = part["time_utc"].iloc[0], part["time_utc"].iloc[-1]
        lo = passed_times.searchsorted(first - CONTEXT)
        hi = passed_times.searchsorted(last + CONTEXT, side="right")
        rows.append({"start_ist": first.tz_convert(IST).strftime("%Y-%m-%d %H:%M"),
                     "end_ist": last.tz_convert(IST).strftime("%Y-%m-%d %H:%M"),
                     "hours": len(part),
                     "rules": "|".join(sorted({r for flag in part["qc_flag"] for r in flag.split("|")})),
                     "min_height_m": part["height_m"].min(), "max_height_m": part["height_m"].max(),
                     "mean_residual_m": round(float(part["qc_residual_m"].mean()), 2),
                     "max_abs_residual_m": round(float(part["qc_residual_m"].abs().max()), 2),
                     "context_residual_m": round(float(np.median(passed_resid[lo:hi])), 2) if hi > lo else np.nan})
    return pd.DataFrame(rows, columns=columns)

def write_qc(port_slug: str, table: pd.DataFrame) -> pd.DataFrame:
    """Commit-ready flags (data/qc) plus the report and review list (output/qc); returns the report."""
    path = flags_path(port_slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.loc[table["qc_flag"] != "", ["time_utc", "height_m", "qc_flag"]].to_csv(path, index=False)
    out = OUTPUT_DIR / "qc"
    out.mkdir(parents=True, exist_ok=True)
    report = qc_report(table)
    report.to_csv(out / f"{port_slug}_report.csv", index=False)
    review_list(table).to_csv(out / f"{port_slug}_review_list.csv", index=False)
    return report

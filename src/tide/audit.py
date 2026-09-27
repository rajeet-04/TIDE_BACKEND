"""Per-year harmonic audit (spec 3.4): years that depart from their neighbours."""
from __future__ import annotations

import numpy as np
import pandas as pd

from tide.harmonic import HarmonicConfig, HarmonicModel
from tide.ports import OUTPUT_DIR, ist_years, port

TRACKED = ("M2", "S2", "K1", "O1", "M4")
SPARSE_SET = ("M2", "S2", "N2", "K1", "O1", "M4", "MS4", "MN4", "M6")
FULL_YEAR_READINGS = 0.6 * 8766
MIN_READINGS = 720
NEIGHBOURS = 6
LIMIT_SD = 3.0

def year_constants(gauge: pd.DataFrame, port_slug: str) -> pd.DataFrame:
    """Mean level and the tracked amplitudes and phases from a separate fit to each IST year.

    Years with under 60% coverage are 'sparse' and get a small constituent set and no
    seasonal terms, so their mean level carries the seasonal bias of the months present."""
    lat = port(port_slug).lat
    passed = gauge[gauge["qc_flag"] == ""]
    rows = []
    for year, part in passed.groupby(ist_years(passed["time_utc"])):
        if len(part) < MIN_READINGS:
            continue
        sparse = len(part) < FULL_YEAR_READINGS
        config = HarmonicConfig(constituents=SPARSE_SET if sparse else "auto", seasonal=0 if sparse else 3, trend=False)
        model = HarmonicModel(lat, config).fit(part["time_utc"], part["height_m"])
        consts = model.constants()
        row = {"year": int(year), "hours": len(part), "sparse": sparse, "mean_level_m": model.mean_level}
        for name in TRACKED:
            row[f"{name}_amp_m"] = consts.loc[name, "amplitude_m"]
            row[f"{name}_phase_deg"] = consts.loc[name, "phase_deg"]
        rows.append(row)
    return pd.DataFrame(rows)

def audit(constants: pd.DataFrame) -> pd.DataFrame:
    """Each metric's departure from the nearest full years (median of the differences,
    wrapped for phases) and its robust z-score; |z| > 3 marks the year as suspect."""
    table = constants.sort_values("year").reset_index(drop=True)
    full = table[~table["sparse"].astype(bool)]
    metrics = ["mean_level_m"] + [f"{n}_amp_m" for n in TRACKED] + [f"{n}_phase_deg" for n in TRACKED]
    suspects: list[list[str]] = [[] for _ in range(len(table))]
    for metric in metrics:
        departures = []
        for _, row in table.iterrows():
            others = full[full["year"] != row["year"]]
            near = others.iloc[np.argsort(np.abs(others["year"].to_numpy() - row["year"]), kind="stable")[:NEIGHBOURS]]
            diff = row[metric] - near[metric].to_numpy(dtype=float)
            if metric.endswith("_phase_deg"):
                diff = (diff + 180.0) % 360.0 - 180.0
            departures.append(float(np.median(diff)))
        departures = np.array(departures)
        reference = departures[~table["sparse"].astype(bool).to_numpy()]
        scale = 1.4826 * np.median(np.abs(reference - np.median(reference)))
        z = departures / scale if scale > 0 else np.zeros(len(departures))
        table[f"{metric}_departure"] = departures
        table[f"{metric}_z"] = z
        for i in np.flatnonzero(np.abs(z) > LIMIT_SD):
            suspects[i].append(metric)
    table["suspect"] = ["|".join(s) for s in suspects]
    return table

def write_audit(port_slug: str, table: pd.DataFrame) -> None:
    out = OUTPUT_DIR / "audit"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / f"{port_slug}_audit.csv", index=False)

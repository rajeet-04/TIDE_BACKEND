import numpy as np
import pandas as pd

from tide.audit import TRACKED, audit, year_constants
from tide.harmonic import HarmonicConfig, HarmonicModel, datenum
from tide.ports import PORTS, ist_year_start, ist_years

def planted_tide(times) -> np.ndarray:
    model = HarmonicModel(PORTS["haldia"].lat,
                          HarmonicConfig(constituents=("M2", "S2", "K1", "O1", "M4"), seasonal=0, trend=False))
    model._prepare(datenum(times))
    model.coef = np.array([3.2, 1.2, 1.5, 0.5, 0.3, 0.3, 0.1, 0.2, 0.15, 0.1, 0.2])
    return model.predict(times)

def test_audit_singles_out_a_relevelled_year_and_a_clock_error():
    times = pd.date_range(ist_year_start(2005), ist_year_start(2015), freq="1h", inclusive="left")
    year = ist_years(times)
    lag = pd.to_timedelta(np.where(year == 2011, 40, 0), unit="min")
    heights = (planted_tide(times - lag) + np.where(year == 2008, 0.3, 0.0)
               + np.random.default_rng(0).normal(0.0, 0.02, len(times)))
    gauge = pd.DataFrame({"time_utc": times, "height_m": heights, "qc_flag": ""})
    table = audit(year_constants(gauge, "haldia")).set_index("year")
    assert table["mean_level_m_z"].abs().idxmax() == 2008 and abs(table.loc[2008, "mean_level_m_z"]) > 3
    assert table["M2_phase_deg_z"].abs().idxmax() == 2011 and abs(table.loc[2011, "M2_phase_deg_z"]) > 3
    assert "mean_level_m" in table.loc[2008, "suspect"] and "M2_phase_deg" in table.loc[2011, "suspect"]

def test_phase_departures_wrap_around_north():
    constants = pd.DataFrame({"year": range(2000, 2010), "hours": 8760, "sparse": False, "mean_level_m": 3.0})
    for name in TRACKED:
        constants[f"{name}_amp_m"] = 1.0
        constants[f"{name}_phase_deg"] = 10.0
    constants["M2_phase_deg"] = [359.9, 0.1] * 4 + [359.9, 90.0]
    table = audit(constants).set_index("year")
    assert table.loc[2009, "suspect"] == "M2_phase_deg"
    assert (table.drop(index=2009)["suspect"] == "").all()

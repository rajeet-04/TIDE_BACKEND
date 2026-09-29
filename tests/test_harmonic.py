import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from utide import reconstruct, solve
from utide._ut_constants import constit_index_dict
from utide.harmonics import FUV

from tide.harmonic import HarmonicConfig, HarmonicModel, datenum, select_constituents
from tide.ports import PORTS, ist_year_start
from tide.store import load_gauge

LAT = PORTS["haldia"].lat
MAJOR = ["M2", "S2", "N2", "K1", "O1", "M4", "MS4"]

def hourly(start: str, years: float) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=int(years * 8766), freq="1h", tz="UTC") + pd.Timedelta(minutes=30)

def planted(config: HarmonicConfig, times, seed: int = 0) -> HarmonicModel:
    """A model with known random coefficients, prepared for these times."""
    model = HarmonicModel(LAT, config)
    model._prepare(datenum(times))
    model.coef = np.random.default_rng(seed).normal(0.0, 0.2, len(model.columns))
    model.coef[0] = 3.0
    return model

@pytest.mark.slow
def test_ols_fit_reproduces_utide_constants_and_predictions():
    gauge = load_gauge("haldia")
    two_years = gauge[(gauge["time_utc"] >= ist_year_start(2019)) & (gauge["time_utc"] < ist_year_start(2021))]
    times, heights = two_years["time_utc"], two_years["height_m"].to_numpy()
    naive = pd.DatetimeIndex(times).tz_localize(None)
    ref = solve(naive, heights, lat=LAT, constit="auto", method="ols", conf_int="none",
                trend=False, nodal=True, verbose=False)
    config = HarmonicConfig(constituents=tuple(ref.name), seasonal=0, trend=False, robust=False, ridge=0.0)
    model = HarmonicModel(LAT, config).fit(times, heights)
    ours = model.constants()
    theirs = pd.DataFrame({"amplitude_m": ref.A, "phase_deg": ref.g}, index=list(ref.name))
    for name in MAJOR:
        assert ours.loc[name, "amplitude_m"] == pytest.approx(theirs.loc[name, "amplitude_m"], abs=1e-4)
        gap = (ours.loc[name, "phase_deg"] - theirs.loc[name, "phase_deg"] + 180) % 360 - 180
        assert abs(gap) < 0.01
    np.testing.assert_allclose(model.predict(times), reconstruct(naive, ref, verbose=False).h, atol=1e-4)

def test_minute_phases_match_direct_utide_nodal_corrections():
    names = ["M2", "K1", "M4", "MS4", "MN4"]
    model = HarmonicModel(LAT, HarmonicConfig(constituents=tuple(names), seasonal=0, trend=False))
    model.names = names
    dn = datenum(pd.date_range("2024-03-01 00:00:17", periods=600, freq="1min", tz="UTC"))
    F, phase = model._nodal(dn)
    F0, U0, V0 = FUV(dn, dn.mean(), [constit_index_dict[n] for n in names], LAT, [False] * 4)
    np.testing.assert_allclose(F, F0, atol=1e-5)
    np.testing.assert_allclose(np.cos(phase), np.cos(2 * np.pi * (U0 + V0)), atol=1e-5)
    np.testing.assert_allclose(np.sin(phase), np.sin(2 * np.pi * (U0 + V0)), atol=1e-5)

def test_fit_recovers_seasonal_side_and_trend_terms():
    times = hourly("2010-01-01", 6)
    config = HarmonicConfig(constituents=tuple(MAJOR), side_terms=True, trend=True, seasonal=3, robust=False, ridge=0.0)
    truth = planted(config, times)
    fitted = HarmonicModel(LAT, config).fit(times, truth.predict(times))
    assert fitted.columns == truth.columns
    assert len(fitted.side) == 2 * len(MAJOR)
    np.testing.assert_allclose(fitted.coef, truth.coef, atol=1e-6)

def test_bisquare_weights_ignore_spikes():
    times = hourly("2015-01-01", 2)
    config = HarmonicConfig(constituents=("M2", "S2", "K1", "O1", "M4"), trend=False, seasonal=1)
    truth = planted(config, times)
    rng = np.random.default_rng(1)
    y = truth.predict(times) + rng.normal(0.0, 0.05, len(times))
    y[rng.choice(len(times), size=len(times) // 100, replace=False)] += 4.0
    robust = HarmonicModel(LAT, config).fit(times, y)
    plain = HarmonicModel(LAT, replace(config, robust=False)).fit(times, y)
    assert np.abs(robust.coef - truth.coef).max() < 0.01
    assert abs(plain.coef[0] - truth.coef[0]) > 0.03

def test_auto_sets_leave_sa_ssa_to_seasonal_terms_and_add_resolvable_shallow_water():
    record_hours = 5 * 8766.0
    tref = datenum(pd.DatetimeIndex(["2015-01-01"], tz="UTC"))[0]
    auto = select_constituents("auto", record_hours, tref)
    assert len(auto) == 66 and "M2" in auto and not {"SA", "SSA"} & set(auto)
    extra = select_constituents("auto+shallow", record_hours, tref)
    assert set(auto) < set(extra) and {"M10", "M12"} <= set(extra)

def test_side_terms_skip_lines_the_auto_set_already_resolves():
    model = HarmonicModel(LAT, HarmonicConfig(side_terms=True))
    model._prepare(datenum(hourly("2010-01-01", 5)))
    assert not {("M2", 1), ("M2", -1), ("S2", 1), ("S2", -1)} & set(model.side)
    assert {("M4", 1), ("O1", -1)} <= set(model.side)

def test_long_requests_are_chunked_without_changing_results():
    config = HarmonicConfig(constituents=("M2", "K1", "M4"), seasonal=1, trend=False, robust=False, ridge=0.0)
    times = pd.date_range("2024-01-01", periods=250_001, freq="1min", tz="UTC")
    model = planted(config, times)
    whole = model.predict(times)
    parts = np.concatenate([model.predict(times[:120_000]), model.predict(times[120_000:])])
    np.testing.assert_allclose(whole, parts, atol=1e-9)

def test_json_round_trip_predicts_identically():
    times = hourly("2012-01-01", 2)
    config = HarmonicConfig(constituents=("M2", "S2", "K1"), side_terms=True, seasonal=2)
    model = HarmonicModel(LAT, config).fit(times, planted(config, times).predict(times))
    copy = HarmonicModel.from_dict(json.loads(json.dumps(model.to_dict())))
    np.testing.assert_array_equal(copy.predict(times), model.predict(times))

def test_grid_names_round_trip():
    config = HarmonicConfig(window_years=8, constituents="auto+shallow", side_terms=True, trend=False)
    assert config.name == "A-w8-auto+shallow-side-notrend"
    assert HarmonicConfig.from_name(config.name) == config
    assert HarmonicConfig.from_name("A-wall-auto-noside-trend").window_years is None

def test_components_add_up():
    times = hourly("2012-01-01", 1)
    config = HarmonicConfig(constituents=("M2", "K1"), seasonal=2, trend=True, robust=False, ridge=0.0)
    model = planted(config, times)
    parts = model.predict(times, components=True)
    np.testing.assert_allclose(parts["astronomical_m"] + parts["seasonal_m"], parts["height_m"])
    np.testing.assert_allclose(parts["height_m"], model.predict(times))

def test_leaving_a_year_out_equals_refitting_without_it():
    from tide.harmonic import leave_one_year_out
    from tide.ports import ist_years

    times = hourly("2010-01-01", 4)
    config = HarmonicConfig(constituents=("M2", "S2", "K1", "M4"), seasonal=1, trend=False, robust=False)
    heights = planted(config, times).predict(times) + np.random.default_rng(2).normal(0.0, 0.05, len(times))
    years = ist_years(times)
    model = HarmonicModel(LAT, config).fit(times, heights)
    refits = leave_one_year_out(model, times, heights, years)
    assert sorted(refits) == sorted(set(years.tolist()))
    keep = years != 2011
    np.testing.assert_allclose(refits[2011], HarmonicModel(LAT, config).fit(times[keep], heights[keep]).coef, atol=1e-8)

def test_robust_cross_fit_stays_within_millimetres_of_a_refit():
    from tide.harmonic import leave_one_year_out
    from tide.ports import ist_years

    times = hourly("2010-01-01", 4)
    config = HarmonicConfig(constituents=("M2", "S2", "K1", "M4"), seasonal=1, trend=False)
    rng = np.random.default_rng(3)
    heights = planted(config, times).predict(times) + rng.normal(0.0, 0.05, len(times))
    heights[rng.choice(len(times), size=len(times) // 200, replace=False)] += 2.0
    years = ist_years(times)
    model = HarmonicModel(LAT, config).fit(times, heights)
    held = years == 2012
    refit = HarmonicModel(LAT, config).fit(times[~held], heights[~held])
    gap = model.with_coef(leave_one_year_out(model, times, heights, years)[2012]).predict(times[held]) - refit.predict(times[held])
    assert np.sqrt(np.mean(gap**2)) < 0.005

def test_normal_equations_do_not_depend_on_the_chunk_size(monkeypatch):
    from tide import harmonic

    times = hourly("2012-01-01", 1)
    config = HarmonicConfig(constituents=("M2", "K1"), seasonal=1, trend=False)
    heights = planted(config, times).predict(times) + np.random.default_rng(4).normal(0.0, 0.05, len(times))
    whole = HarmonicModel(LAT, config).fit(times, heights).coef
    monkeypatch.setattr(harmonic, "CHUNK", 997)
    np.testing.assert_allclose(HarmonicModel(LAT, config).fit(times, heights).coef, whole, atol=1e-10)

def test_with_coef_keeps_the_terms_and_swaps_the_coefficients():
    times = hourly("2012-01-01", 1)
    model = planted(HarmonicConfig(constituents=("M2",), seasonal=0, trend=False), times)
    twin = model.with_coef(np.zeros(len(model.coef)))
    assert twin.columns == model.columns and model.coef[0] == 3.0
    np.testing.assert_array_equal(twin.predict(times), np.zeros(len(times)))

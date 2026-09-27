import numpy as np
import pandas as pd
import pandas.testing as pdt

from tide import backtest
from tide.backtest import FittedLevels, Fold, folds
from tide.harmonic import HarmonicConfig, HarmonicModel
from tide.ports import ist_year_start

START = ist_year_start(2001)

def truth(times) -> np.ndarray:
    hours = np.asarray((pd.DatetimeIndex(times) - START) / pd.Timedelta(hours=1))
    return 3.0 + 2.0 * np.cos(2 * np.pi * hours / 12.4206) + 0.5 * np.cos(2 * np.pi * hours / 12.0)

def use_synthetic_gauge(monkeypatch, cache) -> None:
    times = pd.date_range(START, ist_year_start(2003), freq="1h", inclusive="left")
    gauge = pd.DataFrame({"time_utc": times, "height_m": truth(times), "source": "soi_gauge", "qc_flag": ""})
    monkeypatch.setattr(backtest, "load_gauge", lambda port_slug, passed_only=False: gauge)
    monkeypatch.setattr(backtest, "CACHE_DIR", cache)

class Perfect:
    name = "perfect"

    def fit(self, train, port_slug):
        assert train["time_utc"].max() < ist_year_start(2002)
        return FittedLevels(truth)

class TwoConstituents:
    name = "two-constituents"

    def fit(self, train, port_slug):
        config = HarmonicConfig(constituents=("M2", "S2"), seasonal=0, trend=False)
        return FittedLevels(HarmonicModel(22.03, config).fit(train["time_utc"], train["height_m"]).predict)

def test_folds_follow_the_spec():
    haldia = folds("haldia", set(range(2000, 2025)))
    assert {Fold(2008, 2008), Fold(2022, 2024), Fold(2024, 2024)} <= set(haldia)
    assert Fold(2024, 2025) not in haldia and Fold(2007, 2008) not in haldia
    assert {f.test_year for f in haldia if f.final} == set(range(2020, 2025))
    assert all(f.test_year <= 2019 for f in haldia if not f.final)
    dh = folds("diamond_harbour", set(range(2000, 2017)) | {2021, 2022, 2023})
    assert {(f.origin, f.test_year) for f in dh if f.final} == {(2021, 2021), (2021, 2022), (2021, 2023)}
    assert max(f.test_year for f in dh if not f.final) == 2016
    assert Fold(2021, 2023).horizon == 3

def test_perfect_candidate_hits_every_tide(tmp_path, monkeypatch):
    use_synthetic_gauge(monkeypatch, tmp_path)
    events, hours = backtest.run("haldia", Perfect(), [Fold(2002, 2002)])
    pooled = backtest.summary(events, hours, by=()).iloc[0]
    assert pooled["joint_pct"] == 100.0 and pooled["missed"] == 0 and pooled["extra"] == 0
    assert pooled["hourly_rmse_m"] < 1e-9
    table = backtest.summary(events, hours)
    assert set(table["season"]) == {"dry", "pre_monsoon", "monsoon", "post_monsoon"}
    assert set(table["state"]) == {"High", "Low"} and (table["horizon"] == 1).all()
    assert table["observed"].sum() == (events["kind"] == "observed").sum()

def test_scores_are_reproducible(tmp_path, monkeypatch):
    use_synthetic_gauge(monkeypatch, tmp_path / "first")
    first = backtest.run("haldia", TwoConstituents(), [Fold(2002, 2002)])
    monkeypatch.setattr(backtest, "CACHE_DIR", tmp_path / "second")
    second = backtest.run("haldia", TwoConstituents(), [Fold(2002, 2002)])
    pdt.assert_frame_equal(first[0], second[0])
    pdt.assert_frame_equal(first[1], second[1])

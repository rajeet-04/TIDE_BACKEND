import numpy as np
import pandas as pd
import pytest

from tide.learners import make_learner

def data(n: int = 20_000, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 6)), columns=[f"f{i}" for i in range(6)])
    y = 2.0 * np.sin(X["f0"]) + X["f1"] * X["f2"] + rng.normal(0.0, 0.1, n)
    return X, y.to_numpy()

@pytest.mark.parametrize("name", ["lgbm-l15-m20", "lgbm-l31-m40-huber", "et", "xgb-d4", "mlp",
                                  "blend(lgbm-l15-m20=0.25,et=0.75)"])
def test_learners_learn_and_repeat_exactly(name):
    X, y = data()
    first = make_learner(name).fit(X, y).predict(X)
    np.testing.assert_array_equal(first, make_learner(name).fit(X, y).predict(X))
    assert np.corrcoef(first, y)[0, 1] > 0.8

def test_a_blend_is_the_weighted_average_of_its_members():
    X, y = data(5_000)
    blend = make_learner("blend(lgbm-l15-m20=0.25,et=0.75)").fit(X, y).predict(X)
    parts = 0.25 * make_learner("lgbm-l15-m20").fit(X, y).predict(X) + 0.75 * make_learner("et").fit(X, y).predict(X)
    np.testing.assert_allclose(blend, parts)

@pytest.mark.parametrize("name", ["gbm", "lgbm-l31", "blend(et=0.5,mlp=0.4)", "blend(blend(et=1)=1)"])
def test_bad_names_are_clear_errors(name):
    with pytest.raises(ValueError):
        make_learner(name)

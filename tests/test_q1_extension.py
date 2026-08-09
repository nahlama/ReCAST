import numpy as np
import pandas as pd

from recast_surv.q1_extension import _bh, _net_benefit, _normalized_coefficients


def test_normalized_coefficients_are_nonnegative_and_sum_to_one():
    result = _normalized_coefficients(np.array([-1.0, 2.0, 3.0]))
    assert np.all(result >= 0)
    assert np.isclose(result.sum(), 1.0)
    assert np.allclose(result, [0.0, 0.4, 0.6])


def test_net_benefit_matches_weighted_definition():
    outcome = np.array([1, 0, 1, 0])
    weights = np.ones(4)
    risk = np.array([0.8, 0.7, 0.2, 0.1])
    observed = _net_benefit(outcome, weights, risk, threshold=0.5)
    assert np.isclose(observed, 0.0)


def test_bh_preserves_missing_and_monotonic_ranking():
    adjusted = _bh(pd.Series([0.01, 0.04, np.nan, 0.03]))
    assert np.isnan(adjusted.iloc[2])
    assert adjusted.iloc[0] <= adjusted.iloc[3] <= adjusted.iloc[1]

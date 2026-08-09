import numpy as np
import pandas as pd

from recast_surv.biology import _specificity_effect, benjamini_hochberg


def test_benjamini_hochberg_is_monotone_in_p_value_order():
    values = pd.Series([0.04, 0.001, np.nan, 0.02], index=list("abcd"))
    adjusted = benjamini_hochberg(values)
    ordered = adjusted.loc[values.dropna().sort_values().index].to_numpy()
    assert np.all(np.diff(ordered) >= 0)
    assert np.isnan(adjusted.loc["c"])
    assert adjusted.loc["b"] == 0.003


def test_specificity_effect_is_bounded_rank_contrast():
    effects = _specificity_effect(np.asarray([[1, 2, 3], [3, 2, 1]], dtype=float))
    assert effects.shape == (2, 3)
    assert np.max(np.abs(effects)) <= 1.0

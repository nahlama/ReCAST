import numpy as np

from recast_surv.metrics import (
    cumulative_dynamic_auc,
    integrated_brier_score,
    ipcw_brier_score,
    ipcw_calibration,
    uno_c_index,
    harrell_c_index,
)


def test_concordance_rewards_correct_ranking():
    time = np.array([1.0, 2.0, 3.0, 4.0])
    event = np.ones(4, dtype=int)
    correct_risk = -time
    reversed_risk = time
    assert harrell_c_index(time, event, correct_risk) == 1.0
    assert harrell_c_index(time, event, reversed_risk) == 0.0
    assert uno_c_index(time, event, time, event, correct_risk) == 1.0


def test_uno_uses_training_censoring_distribution():
    train_time = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    train_event = np.array([1, 0, 1, 0, 1])
    test_time = np.array([1.5, 2.5, 3.5])
    test_event = np.array([1, 1, 0])
    score = uno_c_index(train_time, train_event, test_time, test_event, np.array([3.0, 2.0, 1.0]))
    assert np.isclose(score, 1.0)


def test_probability_metrics_reward_correct_predictions():
    train_time = np.arange(1.0, 9.0)
    train_event = np.ones(8, dtype=int)
    test_time = np.array([1.0, 2.0, 5.0, 7.0])
    test_event = np.ones(4, dtype=int)
    horizons = np.array([3.0, 6.0])
    perfect = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    poor = 1.0 - perfect
    assert ipcw_brier_score(train_time, train_event, test_time, test_event, perfect[:, 0], 3.0) == 0.0
    assert integrated_brier_score(
        train_time, train_event, test_time, test_event, perfect, horizons
    ) < integrated_brier_score(train_time, train_event, test_time, test_event, poor, horizons)
    risk = -test_time
    assert cumulative_dynamic_auc(train_time, train_event, test_time, test_event, risk, 3.0) == 1.0


def test_ipcw_calibration_returns_finite_parameters():
    train_time = np.arange(1.0, 13.0)
    train_event = np.ones(12, dtype=int)
    test_time = np.arange(1.0, 9.0)
    test_event = np.ones(8, dtype=int)
    survival = np.linspace(0.1, 0.9, 8)
    intercept, slope = ipcw_calibration(
        train_time, train_event, test_time, test_event, survival, 4.5
    )
    assert np.isfinite(intercept)
    assert np.isfinite(slope)

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def harrell_c_index(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> float:
    """Harrell concordance where a larger prediction means greater risk."""
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=bool)
    risk = np.asarray(risk, dtype=float)
    concordant = 0.0
    comparable = 0.0
    for i in range(len(time)):
        if not event[i]:
            continue
        later = time > time[i]
        if not np.any(later):
            continue
        differences = risk[i] - risk[later]
        concordant += float(np.sum(differences > 0) + 0.5 * np.sum(differences == 0))
        comparable += float(np.sum(later))
    return concordant / comparable if comparable else float("nan")


def _censoring_survival_at(
    train_time: np.ndarray,
    train_event: np.ndarray,
    query_time: np.ndarray,
) -> np.ndarray:
    """Right-continuous Kaplan-Meier estimate of P(censoring time > t)."""
    train_time = np.asarray(train_time, dtype=float)
    censored = 1 - np.asarray(train_event, dtype=int)
    unique_times = np.unique(train_time)
    survival = 1.0
    step_times: list[float] = []
    step_survival: list[float] = []
    for value in unique_times:
        at_risk = int(np.sum(train_time >= value))
        events = int(np.sum((train_time == value) & (censored == 1)))
        if at_risk:
            survival *= 1.0 - events / at_risk
        step_times.append(float(value))
        step_survival.append(survival)
    indices = np.searchsorted(np.asarray(step_times), np.asarray(query_time), side="right") - 1
    result = np.ones(len(np.asarray(query_time)), dtype=float)
    valid = indices >= 0
    result[valid] = np.asarray(step_survival)[indices[valid]]
    return result


def uno_c_index(
    train_time: np.ndarray,
    train_event: np.ndarray,
    test_time: np.ndarray,
    test_event: np.ndarray,
    risk: np.ndarray,
) -> float:
    """IPCW Uno concordance using censoring estimated only on training data."""
    test_time = np.asarray(test_time, dtype=float)
    test_event = np.asarray(test_event, dtype=bool)
    risk = np.asarray(risk, dtype=float)
    censoring_survival = _censoring_survival_at(train_time, train_event, test_time)
    numerator = 0.0
    denominator = 0.0
    for i in range(len(test_time)):
        if not test_event[i] or censoring_survival[i] <= 1e-8:
            continue
        later = test_time > test_time[i]
        if not np.any(later):
            continue
        weight = 1.0 / (censoring_survival[i] ** 2)
        differences = risk[i] - risk[later]
        numerator += weight * float(np.sum(differences > 0) + 0.5 * np.sum(differences == 0))
        denominator += weight * float(np.sum(later))
    return numerator / denominator if denominator else float("nan")


def ipcw_brier_score(
    train_time: np.ndarray,
    train_event: np.ndarray,
    test_time: np.ndarray,
    test_event: np.ndarray,
    survival_probability: np.ndarray,
    evaluation_time: float,
) -> float:
    """Graf IPCW Brier score at one time, using training-fold censoring only."""
    test_time = np.asarray(test_time, dtype=float)
    test_event = np.asarray(test_event, dtype=bool)
    survival_probability = np.asarray(survival_probability, dtype=float)
    horizon = float(evaluation_time)
    g_observed = _censoring_survival_at(train_time, train_event, test_time)
    g_horizon = float(_censoring_survival_at(train_time, train_event, np.asarray([horizon]))[0])
    if g_horizon <= 1e-8:
        return float("nan")
    weights = np.zeros(len(test_time), dtype=float)
    failed = (test_time <= horizon) & test_event
    survived = test_time > horizon
    weights[failed] = 1.0 / np.maximum(g_observed[failed], 1e-8)
    weights[survived] = 1.0 / g_horizon
    outcome = survived.astype(float)
    return float(np.mean(weights * np.square(outcome - survival_probability)))


def integrated_brier_score(
    train_time: np.ndarray,
    train_event: np.ndarray,
    test_time: np.ndarray,
    test_event: np.ndarray,
    survival_probabilities: np.ndarray,
    evaluation_times: np.ndarray,
) -> float:
    times = np.asarray(evaluation_times, dtype=float)
    if len(times) < 2:
        return float("nan")
    scores = np.asarray(
        [
            ipcw_brier_score(
                train_time,
                train_event,
                test_time,
                test_event,
                survival_probabilities[:, index],
                horizon,
            )
            for index, horizon in enumerate(times)
        ]
    )
    return float(np.trapezoid(scores, times) / (times[-1] - times[0]))


def cumulative_dynamic_auc(
    train_time: np.ndarray,
    train_event: np.ndarray,
    test_time: np.ndarray,
    test_event: np.ndarray,
    risk: np.ndarray,
    evaluation_time: float,
) -> float:
    """IPCW cumulative/dynamic AUC: failures by t versus event-free at t."""
    test_time = np.asarray(test_time, dtype=float)
    test_event = np.asarray(test_event, dtype=bool)
    risk = np.asarray(risk, dtype=float)
    horizon = float(evaluation_time)
    cases = np.flatnonzero((test_time <= horizon) & test_event)
    controls = np.flatnonzero(test_time > horizon)
    if not len(cases) or not len(controls):
        return float("nan")
    g_cases = _censoring_survival_at(train_time, train_event, test_time[cases])
    weights = 1.0 / np.maximum(g_cases, 1e-8)
    numerator = 0.0
    denominator = 0.0
    for case, weight in zip(cases, weights):
        differences = risk[case] - risk[controls]
        numerator += weight * float(np.sum(differences > 0) + 0.5 * np.sum(differences == 0))
        denominator += weight * len(controls)
    return numerator / denominator if denominator else float("nan")


def ipcw_calibration(
    train_time: np.ndarray,
    train_event: np.ndarray,
    test_time: np.ndarray,
    test_event: np.ndarray,
    survival_probability: np.ndarray,
    evaluation_time: float,
) -> tuple[float, float]:
    """Weighted logistic calibration intercept/slope for event risk by time t."""
    test_time = np.asarray(test_time, dtype=float)
    test_event = np.asarray(test_event, dtype=bool)
    horizon = float(evaluation_time)
    g_observed = _censoring_survival_at(train_time, train_event, test_time)
    g_horizon = float(_censoring_survival_at(train_time, train_event, np.asarray([horizon]))[0])
    failed = (test_time <= horizon) & test_event
    survived = test_time > horizon
    usable = failed | survived
    if failed.sum() < 2 or survived.sum() < 2 or g_horizon <= 1e-8:
        return float("nan"), float("nan")
    weights = np.zeros(len(test_time), dtype=float)
    weights[failed] = 1.0 / np.maximum(g_observed[failed], 1e-8)
    weights[survived] = 1.0 / g_horizon
    outcome = failed.astype(float)[usable]
    predicted_event = np.clip(1.0 - np.asarray(survival_probability, dtype=float)[usable], 1e-6, 1 - 1e-6)
    linear_predictor = np.log(predicted_event / (1.0 - predicted_event))
    fit_weights = weights[usable]

    def objective(parameters: np.ndarray) -> float:
        logits = parameters[0] + parameters[1] * linear_predictor
        return float(
            np.sum(fit_weights * (np.logaddexp(0.0, logits) - outcome * logits))
            / np.sum(fit_weights)
        )

    result = minimize(objective, x0=np.asarray([0.0, 1.0]), method="BFGS")
    if not result.success or not np.isfinite(result.x).all():
        return float("nan"), float("nan")
    return float(result.x[0]), float(result.x[1])

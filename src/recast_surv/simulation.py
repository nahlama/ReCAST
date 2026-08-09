from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr

from .recast import ReCASTEstimator


SCENARIOS = {
    "clean": {"noise": 0.01, "power_low": 1.0, "power_high": 1.0, "unknown_high": 0.0},
    "platform_shift": {"noise": 0.05, "power_low": 0.6, "power_high": 1.4, "unknown_high": 0.0},
    "unknown_component": {"noise": 0.05, "power_low": 0.6, "power_high": 1.4, "unknown_high": 0.4},
}


def _make_estimator(backend: str, settings: dict[str, Any]) -> ReCASTEstimator:
    return ReCASTEstimator(
        backend=backend,
        markers_per_state=int(settings["markers_per_state"]),
        min_state_donors=int(settings["min_state_donors"]),
        robust_iterations=int(settings["robust_iterations"]),
        huber_delta=float(settings["huber_delta"]),
        transport_epsilon=float(settings.get("transport_epsilon", 0.15)),
        mass_penalty=float(settings.get("mass_penalty", 0.8)),
        target_mass_penalty=float(settings.get("target_mass_penalty", 0.0)),
        transport_iterations=int(settings.get("transport_iterations", 500)),
        transport_tolerance=float(settings.get("transport_tolerance", 1e-7)),
        protein_prior_strength=float(settings.get("protein_prior_strength", 0.25)),
    )


def run_transport_simulation(
    reference_metadata: pd.DataFrame,
    reference_profiles: pd.DataFrame,
    settings: dict[str, Any],
    gene_reliability: dict[str, float] | None,
    mixtures_per_scenario: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    estimators = {}
    for backend in ("unbalanced_ot", "robust_nnls"):
        estimators[backend] = _make_estimator(backend, settings).fit(
            reference_metadata,
            reference_profiles,
            set(reference_profiles.columns),
            gene_reliability=gene_reliability,
        )
    primary = estimators["unbalanced_ot"]
    if primary.generative_signature_ is None:
        raise ValueError("Reference generative signature was not retained")
    signature = primary.generative_signature_.to_numpy(dtype=float)
    genes = primary.generative_signature_.index.tolist()
    states = primary.generative_signature_.columns.tolist()
    if estimators["robust_nnls"].marker_genes_ != genes:
        raise ValueError("Backends selected different marker genes")

    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for scenario_name, scenario in SCENARIOS.items():
        truth = rng.dirichlet(np.full(len(states), 0.5), size=int(mixtures_per_scenario))
        unknown_truth = rng.uniform(0.0, scenario["unknown_high"], size=len(truth))
        mixtures = []
        for proportions, unknown_fraction in zip(truth, unknown_truth):
            signal = signature @ proportions
            unknown = rng.lognormal(mean=-0.2, sigma=0.8, size=len(genes))
            unknown = unknown / np.mean(unknown) * np.mean(signal)
            mixed = (1.0 - unknown_fraction) * signal + unknown_fraction * unknown
            power = rng.uniform(scenario["power_low"], scenario["power_high"])
            shifted = np.power(np.clip(mixed, 1e-8, None), power)
            shifted += rng.normal(0.0, scenario["noise"] * np.std(shifted), size=len(shifted))
            mixtures.append(np.clip(shifted, 0.0, None))
        bulk = pd.DataFrame(mixtures, columns=genes)

        for backend, estimator in estimators.items():
            scores = estimator.transform(bulk)
            state_columns = [f"state__{state.replace('/', '_')}" for state in states]
            estimated = scores[state_columns].to_numpy(dtype=float)
            estimated /= np.maximum(estimated.sum(axis=1, keepdims=True), 1e-12)
            for index in range(len(truth)):
                rows.append(
                    {
                        "scenario": scenario_name,
                        "backend": backend,
                        "mixture": index,
                        "state_mae": float(np.mean(np.abs(estimated[index] - truth[index]))),
                        "jensen_shannon": float(
                            jensenshannon(
                                np.clip(estimated[index], 1e-12, None),
                                np.clip(truth[index], 1e-12, None),
                            )
                            ** 2
                        ),
                        "unknown_truth": float(unknown_truth[index]),
                        "unknown_estimate": float(scores.iloc[index]["recast__unknown_score"]),
                        "reconstruction_cosine": float(
                            scores.iloc[index]["recast__reconstruction_cosine"]
                        ),
                    }
                )
    results = pd.DataFrame(rows)
    summaries = []
    for (scenario, backend), block in results.groupby(["scenario", "backend"], sort=True):
        if block["unknown_truth"].nunique() > 1:
            correlation = float(
                spearmanr(block["unknown_truth"], block["unknown_estimate"]).statistic
            )
        else:
            correlation = float("nan")
        summaries.append(
            {
                "scenario": scenario,
                "backend": backend,
                "mixtures": len(block),
                "state_mae_mean": float(block["state_mae"].mean()),
                "state_mae_std": float(block["state_mae"].std()),
                "jensen_shannon_mean": float(block["jensen_shannon"].mean()),
                "unknown_spearman": correlation,
                "reconstruction_cosine_mean": float(block["reconstruction_cosine"].mean()),
            }
        )
    return results, pd.DataFrame(summaries)

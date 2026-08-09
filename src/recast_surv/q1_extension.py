from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from scipy.optimize import minimize, nnls
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr, wilcoxon
from sklearn.svm import NuSVR

from .features import read_hpa_protein_priors
from .metrics import _censoring_survival_at, harrell_c_index, uno_c_index
from .recast import ReCASTEstimator


METHOD_LABELS = {
    "robust_nnls_hpa": "HPA-weighted robust NNLS (ReCAST)",
    "ordinary_nnls": "Ordinary NNLS",
    "ridge_nnls": "Ridge NNLS",
    "simplex_nnls": "Simplex-constrained NNLS",
    "nu_svr": "CIBERSORT-style nu-SVR",
    "unbalanced_ot_hpa": "HPA-weighted unbalanced OT",
}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _bh(values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.dropna().sort_values()
    if valid.empty:
        return result
    adjusted = valid * len(valid) / np.arange(1, len(valid) + 1)
    adjusted = np.minimum.accumulate(adjusted.iloc[::-1])[::-1].clip(upper=1.0)
    result.loc[valid.index] = adjusted
    return result


def _estimator(settings: dict[str, Any], backend: str) -> ReCASTEstimator:
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


def _normalize_columns(design: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(design, axis=0, keepdims=True)
    return np.divide(design, norms, out=np.zeros_like(design), where=norms > 0)


def _normalized_coefficients(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=float), 0.0, None)
    total = float(values.sum())
    return values / total if total > 0 else np.full(len(values), 1.0 / len(values))


def _matched_scorers(primary: ReCASTEstimator, uot: ReCASTEstimator) -> dict[str, Callable[[np.ndarray], np.ndarray]]:
    if primary.generative_signature_ is None or primary.signature_ is None:
        raise ValueError("Primary estimator did not retain its signature")
    if primary.marker_weights_ is None:
        raise ValueError("Primary estimator did not retain marker weights")
    raw_design = _normalize_columns(primary.generative_signature_.to_numpy(dtype=float))
    weighted_design = primary.signature_
    weights = primary.marker_weights_

    def prepare(values: np.ndarray, weighted: bool = True) -> np.ndarray:
        ranked = primary._rank_vector(values)
        if weighted:
            ranked = ranked * weights
        return ranked / (np.linalg.norm(ranked) + 1e-12)

    def robust(values: np.ndarray) -> np.ndarray:
        return _normalized_coefficients(primary._score_nnls(values)[0])

    def ordinary(values: np.ndarray) -> np.ndarray:
        coefficients, _ = nnls(raw_design, prepare(values, weighted=False))
        return _normalized_coefficients(coefficients)

    def ridge(values: np.ndarray) -> np.ndarray:
        alpha = 0.10
        augmented_design = np.vstack([weighted_design, np.sqrt(alpha) * np.eye(len(primary.states_))])
        augmented_target = np.concatenate([prepare(values), np.zeros(len(primary.states_))])
        coefficients, _ = nnls(augmented_design, augmented_target)
        return _normalized_coefficients(coefficients)

    def simplex(values: np.ndarray) -> np.ndarray:
        target = prepare(values)
        start = np.full(len(primary.states_), 1.0 / len(primary.states_))
        result = minimize(
            lambda coefficient: float(np.square(weighted_design @ coefficient - target).sum()),
            start,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * len(start),
            constraints={"type": "eq", "fun": lambda coefficient: float(coefficient.sum() - 1.0)},
            options={"maxiter": 500, "ftol": 1e-10},
        )
        return _normalized_coefficients(result.x if result.success else start)

    def nu_svr(values: np.ndarray) -> np.ndarray:
        model = NuSVR(kernel="linear", C=1.0, nu=0.5, tol=1e-5, max_iter=10000)
        model.fit(weighted_design, prepare(values))
        return _normalized_coefficients(np.asarray(model.coef_).reshape(-1))

    def unbalanced_ot(values: np.ndarray) -> np.ndarray:
        return _normalized_coefficients(uot._score_unbalanced_ot(values)[0])

    return {
        "robust_nnls_hpa": robust,
        "ordinary_nnls": ordinary,
        "ridge_nnls": ridge,
        "simplex_nnls": simplex,
        "nu_svr": nu_svr,
        "unbalanced_ot_hpa": unbalanced_ot,
    }


def _cluster_bootstrap_mean(values: pd.DataFrame, value_column: str, seed: int, iterations: int = 2000) -> tuple[float, float]:
    donors = values["heldout_donor"].drop_duplicates().to_numpy()
    if not len(donors):
        return float("nan"), float("nan")
    donor_means = values.groupby("heldout_donor")[value_column].mean()
    rng = np.random.default_rng(seed)
    estimates = np.asarray(
        [donor_means.loc[rng.choice(donors, size=len(donors), replace=True)].mean() for _ in range(iterations)]
    )
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def run_donor_heldout_projection_validation(
    reference_metadata: pd.DataFrame,
    reference_profiles: pd.DataFrame,
    settings: dict[str, Any],
    protein_priors: dict[str, float],
    output_dir: str | Path,
    mixtures_per_donor: int,
    seed: int,
    max_donors: int | None = None,
) -> dict[str, Any]:
    """Validate projection on pseudo-bulk mixtures from completely held-out donors."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = reference_metadata.set_index("profile_id").loc[reference_profiles.index].copy()
    donor_sources = metadata.groupby("patient")["source"].first()
    donors = sorted(metadata["patient"].astype(str).unique())
    if max_donors is not None:
        donors = donors[: int(max_donors)]
    rng = np.random.default_rng(seed)
    metric_rows: list[dict[str, Any]] = []
    composition_rows: list[dict[str, Any]] = []

    for donor_index, donor in enumerate(donors):
        train_mask = metadata["patient"].astype(str).ne(donor)
        heldout_mask = ~train_mask
        train_metadata = metadata.loc[train_mask].reset_index()
        train_profiles = reference_profiles.loc[train_mask]
        primary = _estimator(settings, "robust_nnls").fit(
            train_metadata,
            train_profiles,
            set(reference_profiles.columns),
            gene_reliability=protein_priors,
        )
        uot = _estimator(settings, "unbalanced_ot").fit(
            train_metadata,
            train_profiles,
            set(reference_profiles.columns),
            gene_reliability=protein_priors,
        )
        if primary.marker_genes_ != uot.marker_genes_ or primary.states_ != uot.states_:
            raise ValueError(f"Projection comparators selected incompatible anchors for held-out donor {donor}")
        scorers = _matched_scorers(primary, uot)
        heldout = reference_profiles.loc[heldout_mask].copy()
        heldout["_state"] = metadata.loc[heldout_mask, "state"].astype(str).to_numpy()
        heldout = heldout.groupby("_state", sort=True).median()
        available_states = [state for state in primary.states_ if state in heldout.index]
        if len(available_states) < 2:
            continue
        library_scaled = heldout.loc[available_states, reference_profiles.columns].astype(float)
        library_scaled = library_scaled.div(library_scaled.mean(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)

        for mixture_index in range(int(mixtures_per_donor)):
            component_count = int(rng.integers(2, min(6, len(available_states)) + 1))
            components = sorted(rng.choice(available_states, size=component_count, replace=False).tolist())
            weights_truth = rng.dirichlet(np.full(component_count, 0.7))
            truth = pd.Series(0.0, index=primary.states_)
            truth.loc[components] = weights_truth
            base = weights_truth @ library_scaled.loc[components].to_numpy(dtype=float)
            scenario_values = {
                "heldout_donor_clean": np.clip(base, 0.0, None),
                "heldout_donor_platform_shift": np.clip(
                    np.power(np.clip(base, 1e-8, None), rng.uniform(0.70, 1.30))
                    * rng.lognormal(mean=0.0, sigma=0.08, size=len(base))
                    + rng.normal(0.0, 0.02 * np.std(base), size=len(base)),
                    0.0,
                    None,
                ),
            }
            for scenario, mixture in scenario_values.items():
                marker_values = pd.Series(mixture, index=reference_profiles.columns).loc[primary.marker_genes_].to_numpy()
                for method, scorer in scorers.items():
                    estimated = scorer(marker_values)
                    mae = float(np.mean(np.abs(estimated - truth.to_numpy(dtype=float))))
                    js = float(jensenshannon(np.clip(estimated, 1e-12, None), np.clip(truth, 1e-12, None)) ** 2)
                    correlation = float(spearmanr(estimated, truth.to_numpy(dtype=float)).statistic)
                    key = {
                        "heldout_donor": donor,
                        "source": donor_sources.get(donor, ""),
                        "donor_index": donor_index,
                        "mixture": mixture_index,
                        "scenario": scenario,
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "component_count": component_count,
                    }
                    metric_rows.append({**key, "state_mae": mae, "jensen_shannon": js, "state_spearman": correlation})
                    for state, true_value, estimate in zip(primary.states_, truth, estimated):
                        composition_rows.append(
                            {
                                **key,
                                "state": state,
                                "truth": float(true_value),
                                "estimate": float(estimate),
                            }
                        )

    metrics = pd.DataFrame(metric_rows)
    composition = pd.DataFrame(composition_rows)
    metrics.to_csv(output_dir / "mixture_metrics.csv", index=False)
    composition.to_csv(output_dir / "composition_long.csv", index=False)
    summaries: list[dict[str, Any]] = []
    for group_index, ((scenario, method), block) in enumerate(metrics.groupby(["scenario", "method"], sort=True)):
        record: dict[str, Any] = {
            "scenario": scenario,
            "method": method,
            "method_label": METHOD_LABELS[method],
            "donors": int(block["heldout_donor"].nunique()),
            "mixtures": int(len(block)),
        }
        for metric in ("state_mae", "jensen_shannon", "state_spearman"):
            low, high = _cluster_bootstrap_mean(block, metric, seed + group_index * 17 + len(metric))
            record[f"{metric}_mean"] = float(block[metric].mean())
            record[f"{metric}_ci95_low"] = low
            record[f"{metric}_ci95_high"] = high
        summaries.append(record)
    summary = pd.DataFrame(summaries)
    summary.to_csv(output_dir / "summary.csv", index=False)

    paired_rows: list[dict[str, Any]] = []
    primary = metrics[metrics["method"] == "robust_nnls_hpa"]
    keys = ["heldout_donor", "mixture", "scenario"]
    for scenario in sorted(metrics["scenario"].unique()):
        for method in sorted(set(metrics["method"]) - {"robust_nnls_hpa"}):
            comparator = metrics[metrics["method"] == method]
            paired = primary.merge(comparator, on=keys, suffixes=("_primary", "_comparator"), validate="one_to_one")
            paired = paired[paired["scenario"] == scenario].copy()
            donor_delta = paired.groupby("heldout_donor").apply(
                lambda frame: float((frame["state_mae_comparator"] - frame["state_mae_primary"]).mean()),
                include_groups=False,
            )
            rng_pair = np.random.default_rng(seed + len(paired_rows) * 101)
            bootstrap = [float(rng_pair.choice(donor_delta, size=len(donor_delta), replace=True).mean()) for _ in range(5000)]
            try:
                p_value = float(wilcoxon(donor_delta).pvalue)
            except ValueError:
                p_value = float("nan")
            paired_rows.append(
                {
                    "scenario": scenario,
                    "comparator": method,
                    "comparator_label": METHOD_LABELS[method],
                    "donors": int(len(donor_delta)),
                    "mean_mae_improvement_primary": float(donor_delta.mean()),
                    "ci95_low": float(np.quantile(bootstrap, 0.025)),
                    "ci95_high": float(np.quantile(bootstrap, 0.975)),
                    "wilcoxon_p": p_value,
                }
            )
    paired = pd.DataFrame(paired_rows)
    paired["fdr_bh"] = paired.groupby("scenario", group_keys=False)["wilcoxon_p"].apply(_bh)
    paired.to_csv(output_dir / "paired_comparisons.csv", index=False)
    result = {
        "analysis_role": "locked_secondary_q1_extension",
        "validation_unit": "completely held-out donor",
        "pseudo_bulk_source": "linear mixtures of donor-state profiles derived from genuine cell-level GEO inputs",
        "donors_evaluated": int(metrics["heldout_donor"].nunique()),
        "mixtures_per_donor": int(mixtures_per_donor),
        "scenarios": sorted(metrics["scenario"].unique()),
        "methods": list(METHOD_LABELS),
        "primary_method_locked_before_extension": "robust_nnls_hpa",
        "seed": int(seed),
    }
    _write_json(output_dir / "summary.json", result)
    return result


def _ipcw_binary_weights(
    train_time: np.ndarray,
    train_event: np.ndarray,
    test_time: np.ndarray,
    test_event: np.ndarray,
    horizon: float,
) -> tuple[np.ndarray, np.ndarray]:
    g_observed = _censoring_survival_at(train_time, train_event, test_time)
    g_horizon = float(_censoring_survival_at(train_time, train_event, np.asarray([horizon]))[0])
    cases = (test_time <= horizon) & test_event.astype(bool)
    controls = test_time > horizon
    outcome = cases.astype(int)
    weights = np.zeros(len(test_time), dtype=float)
    weights[cases] = 1.0 / np.maximum(g_observed[cases], 1e-8)
    if g_horizon > 1e-8:
        weights[controls] = 1.0 / g_horizon
    return outcome, weights


def _net_benefit(outcome: np.ndarray, weights: np.ndarray, risk: np.ndarray, threshold: float) -> float:
    positive = risk >= threshold
    true_positive = float(np.sum(weights * positive * (outcome == 1)))
    false_positive = float(np.sum(weights * positive * (outcome == 0)))
    return true_positive / len(risk) - false_positive / len(risk) * threshold / (1.0 - threshold)


def _km_risk(time: pd.Series, event: pd.Series, horizon: float) -> tuple[float, float, float]:
    km = KaplanMeierFitter().fit(time, event_observed=event)
    survival = float(km.predict(horizon))
    ci = km.confidence_interval_survival_function_
    index = ci.index.searchsorted(horizon, side="right") - 1
    if index < 0:
        return 1.0 - survival, float("nan"), float("nan")
    lower_survival = float(ci.iloc[index, 0])
    upper_survival = float(ci.iloc[index, 1])
    return 1.0 - survival, 1.0 - upper_survival, 1.0 - lower_survival


def run_clinical_utility_analysis(root: str | Path, output_dir: str | Path, seed: int) -> dict[str, Any]:
    root = Path(root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clinical = pd.read_csv(root / "cohort" / "clinical.csv").set_index("sample_id")
    predictions = pd.read_csv(root / "benchmark" / "outer_predictions.csv")
    predictions = predictions[predictions["panel"] == "clinical"].copy()
    challenger_path = root / "challengers" / "survpfn" / "outer_predictions.csv"
    if challenger_path.exists():
        predictions = pd.concat([predictions, pd.read_csv(challenger_path)], ignore_index=True)
    horizons = [365, 1095]
    thresholds = np.round(np.arange(0.05, 0.61, 0.025), 3)
    dca_rows: list[dict[str, Any]] = []
    for (model, repeat), block in predictions.groupby(["model", "repeat"], sort=True):
        for horizon in horizons:
            pooled: list[pd.DataFrame] = []
            for fold, test in block.groupby("fold"):
                train = clinical.loc[~clinical.index.isin(test["sample_id"])]
                outcome, weights = _ipcw_binary_weights(
                    train["time_days"].to_numpy(),
                    train["event"].to_numpy(),
                    test["time_days"].to_numpy(),
                    test["event"].to_numpy(),
                    horizon,
                )
                pooled.append(
                    pd.DataFrame(
                        {
                            "outcome": outcome,
                            "weight": weights,
                            "risk": 1.0 - test[f"survival_t{horizon}"].to_numpy(dtype=float),
                        }
                    )
                )
            pooled_frame = pd.concat(pooled, ignore_index=True)
            for threshold in thresholds:
                dca_rows.append(
                    {
                        "dataset": "TCGA_ESCA_outer_test",
                        "model": model,
                        "repeat": int(repeat),
                        "horizon_days": horizon,
                        "threshold_probability": threshold,
                        "net_benefit": _net_benefit(
                            pooled_frame["outcome"].to_numpy(),
                            pooled_frame["weight"].to_numpy(),
                            pooled_frame["risk"].to_numpy(),
                            threshold,
                        ),
                        "treat_all_net_benefit": _net_benefit(
                            pooled_frame["outcome"].to_numpy(),
                            pooled_frame["weight"].to_numpy(),
                            np.ones(len(pooled_frame)),
                            threshold,
                        ),
                    }
                )
    dca = pd.DataFrame(dca_rows)
    dca.to_csv(output_dir / "decision_curve_internal.csv", index=False)

    external_parts = []
    for model, path in {
        "survivalpfn": root / "external_validation" / "clinical_survivalpfn" / "predictions.csv",
        "elastic_net_cox": root / "external_validation" / "clinical_elastic_net_cox" / "predictions.csv",
        "survpfn": root / "challengers" / "survpfn" / "external_predictions.csv",
    }.items():
        if path.exists():
            part = pd.read_csv(path)
            part["model"] = model
            external_parts.append(part)
    external = pd.concat(external_parts, ignore_index=True)
    external_dca_rows: list[dict[str, Any]] = []
    for model, block in external.groupby("model"):
        for horizon in horizons:
            outcome, weights = _ipcw_binary_weights(
                block["time_days"].to_numpy(),
                block["event"].to_numpy(),
                block["time_days"].to_numpy(),
                block["event"].to_numpy(),
                horizon,
            )
            risk = 1.0 - block[f"survival_t{horizon}"].to_numpy(dtype=float)
            for threshold in thresholds:
                external_dca_rows.append(
                    {
                        "dataset": "GSE53625_locked_external",
                        "model": model,
                        "horizon_days": horizon,
                        "threshold_probability": threshold,
                        "net_benefit": _net_benefit(outcome, weights, risk, threshold),
                        "treat_all_net_benefit": _net_benefit(outcome, weights, np.ones(len(risk)), threshold),
                    }
                )
    pd.DataFrame(external_dca_rows).to_csv(output_dir / "decision_curve_external.csv", index=False)

    averaged = predictions.groupby(["sample_id", "model"], as_index=False).agg(
        time_days=("time_days", "first"), event=("event", "first"), risk=("risk", "mean")
    )
    averaged = averaged.merge(clinical[["histology"]], left_on="sample_id", right_index=True, validate="many_to_one")
    subgroup_rows: list[dict[str, Any]] = []
    interaction_rows: list[dict[str, Any]] = []
    for model, block in averaged.groupby("model"):
        for histology, subset in block.groupby("histology"):
            subgroup_rows.append(
                {
                    "model": model,
                    "histology": histology,
                    "patients": int(len(subset)),
                    "events": int(subset["event"].sum()),
                    "harrell_c": harrell_c_index(
                        subset["time_days"].to_numpy(), subset["event"].to_numpy(), subset["risk"].to_numpy()
                    ),
                    "uno_c": uno_c_index(
                        subset["time_days"].to_numpy(), subset["event"].to_numpy(),
                        subset["time_days"].to_numpy(), subset["event"].to_numpy(), subset["risk"].to_numpy(),
                    ),
                }
            )
        interaction = block[["time_days", "event", "risk", "histology"]].copy()
        interaction["risk"] = (interaction["risk"] - interaction["risk"].mean()) / (interaction["risk"].std() + 1e-12)
        interaction["histology_ESCC"] = interaction["histology"].eq("ESCC").astype(float)
        interaction["risk_x_ESCC"] = interaction["risk"] * interaction["histology_ESCC"]
        fit = CoxPHFitter(penalizer=0.01).fit(
            interaction[["time_days", "event", "risk", "histology_ESCC", "risk_x_ESCC"]],
            "time_days",
            "event",
        )
        interaction_rows.append(
            {
                "model": model,
                "interaction_log_hr": float(fit.params_["risk_x_ESCC"]),
                "interaction_hr": float(np.exp(fit.params_["risk_x_ESCC"])),
                "ci95_low": float(np.exp(fit.confidence_intervals_.loc["risk_x_ESCC"].iloc[0])),
                "ci95_high": float(np.exp(fit.confidence_intervals_.loc["risk_x_ESCC"].iloc[1])),
                "p_value": float(fit.summary.loc["risk_x_ESCC", "p"]),
            }
        )
    subgroup = pd.DataFrame(subgroup_rows)
    interactions = pd.DataFrame(interaction_rows)
    interactions["fdr_bh"] = _bh(interactions["p_value"])
    subgroup.to_csv(output_dir / "histology_subgroup_performance.csv", index=False)
    interactions.to_csv(output_dir / "histology_interactions.csv", index=False)

    calibration_rows: list[dict[str, Any]] = []
    internal_risk = predictions.groupby(["sample_id", "model"], as_index=False).agg(
        **{f"survival_t{horizon}": (f"survival_t{horizon}", "mean") for horizon in horizons}
    )
    for model, external_block in external.groupby("model"):
        training_block = internal_risk[internal_risk["model"] == model]
        if training_block.empty:
            continue
        for horizon in horizons:
            risk_column = f"survival_t{horizon}"
            cutpoints = np.quantile(1.0 - training_block[risk_column], [1 / 3, 2 / 3])
            risk = 1.0 - external_block[risk_column]
            groups = pd.cut(risk, [-np.inf, cutpoints[0], cutpoints[1], np.inf], labels=["low", "intermediate", "high"])
            for group in ["low", "intermediate", "high"]:
                subset = external_block.loc[groups == group]
                observed, low, high = _km_risk(subset["time_days"], subset["event"], horizon)
                calibration_rows.append(
                    {
                        "model": model,
                        "horizon_days": horizon,
                        "risk_group": group,
                        "tcga_cutpoint_low": float(cutpoints[0]),
                        "tcga_cutpoint_high": float(cutpoints[1]),
                        "patients": int(len(subset)),
                        "events_total_followup": int(subset["event"].sum()),
                        "mean_predicted_risk": float((1.0 - subset[risk_column]).mean()),
                        "km_observed_risk": observed,
                        "km_ci95_low": low,
                        "km_ci95_high": high,
                    }
                )
    calibration = pd.DataFrame(calibration_rows)
    calibration.to_csv(output_dir / "external_risk_group_calibration.csv", index=False)
    result = {
        "analysis_role": "secondary_exploratory_clinical_utility",
        "internal_dca": "pooled outer-test predictions within repeat with fold-training censoring weights",
        "external_dca": "locked predictions without recalibration; evaluation-cohort censoring weights",
        "risk_group_thresholds": "tertiles defined from averaged TCGA out-of-fold risks and transferred unchanged",
        "subgroups": ["EAC", "ESCC"],
        "horizons_days": horizons,
        "seed": seed,
    }
    _write_json(output_dir / "summary.json", result)
    return result


def make_probast_ai_self_assessment(output_dir: str | Path) -> pd.DataFrame:
    """Domain-level author self-assessment; not a substitute for independent PROBAST+AI review."""

    rows = [
        ("Participants and data sources", "Some concerns", "Retrospective public cohorts; eligibility and overlap exclusions are reproducible.", "Report selection mechanisms and cohort differences; avoid deployment claims."),
        ("Predictors", "Some concerns", "Clinical predictors are available in both cohorts, but external gene annotation is untraceable.", "Restrict external claims to clinical variables until versioned reannotation passes."),
        ("Outcome", "Low", "Overall survival time and death status are defined consistently and without predictor knowledge.", "Report administrative differences in follow-up."),
        ("Sample size and data complexity", "High", "The development set has 182 patients and 77 deaths relative to multiple candidate models.", "Use compact predictors, penalization, nested validation, and acknowledge imprecision."),
        ("Missing data", "Low", "Median imputation is fitted inside each training fold.", "Publish fold-local preprocessing code and missingness counts."),
        ("Model development", "Some concerns", "Candidate models and tuning are prespecified, but complex learners remain unstable in small samples.", "Retain elastic-net Cox and prohibit post-result model switching."),
        ("Internal evaluation", "Low", "Ten repeated five-fold outer splits are locked; preprocessing and tuning are training-only.", "Report paired fold uncertainty and all models, including negative results."),
        ("External evaluation", "Some concerns", "The 60-patient test is non-overlapping but small and ESCC-only; calibration is poor.", "No external recalibration or deployment claim; show confidence intervals and applicability limits."),
        ("Performance measures", "Low", "Uno C, Harrell C, dynamic AUC, IPCW Brier/IBS, calibration and decision curves are reported.", "Interpret clinical utility as exploratory because thresholds were not prospectively chosen."),
        ("Subgroups and fairness", "High", "Sex, ancestry and histology strata are too small for reliable fairness conclusions.", "Provide descriptive EAC/ESCC results and interaction tests; state that fairness is unresolved."),
        ("Applicability", "High", "Public retrospective cohorts and platform discordance limit transportability to contemporary clinical practice.", "Frame as a methodological study, not a clinical decision aid."),
        ("Overall predictive-performance risk of bias", "High", "Strong validation safeguards do not eliminate small-sample and external-applicability limitations.", "Require independent prospective or larger multicentre evaluation before clinical use."),
    ]
    frame = pd.DataFrame(rows, columns=["domain", "author_judgment", "evidence", "mitigation_or_claim_boundary"])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "probast_ai_domain_self_assessment.csv", index=False)
    (output_dir / "README.md").write_text(
        "# PROBAST+AI author self-assessment\n\n"
        "This is a domain-level evidence map prepared by the study authors. It does not reproduce the official tool, "
        "does not constitute independent assessment, and must not be described as certification or low risk of bias.\n",
        encoding="utf-8",
    )
    return frame


def write_q1_extension_protocol(output_dir: str | Path, seed: int) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "title": "Locked secondary protocol for Q1-strengthening analyses",
        "frozen_on": "2026-07-22",
        "role": "secondary and exploratory; does not replace the completed confirmatory benchmark",
        "primary_projection_method": "HPA-weighted robust NNLS fixed before these analyses",
        "projection_primary_metric": "donor-clustered mean state-composition MAE",
        "projection_validation": "leave-one-donor-out anchors; pseudo-bulk mixtures from the held-out donor only",
        "comparators": list(METHOD_LABELS),
        "clinical_utility_horizons_days": [365, 1095],
        "decision_threshold_range": [0.05, 0.60],
        "external_policy": "no outcome-driven feature selection, threshold tuning, or recalibration in GSE53625",
        "subgroup_policy": "EAC/ESCC descriptive metrics plus interaction tests; no subgroup superiority claims",
        "external_omics_policy": "requires a versioned sequence-to-gene mapping before any gene-level confirmation",
        "seed": int(seed),
    }
    path = output_dir / "Q1_EXTENSION_PROTOCOL.json"
    _write_json(path, payload)
    return path


def write_q1_artifact_manifest(output_dir: str | Path) -> pd.DataFrame:
    output_dir = Path(output_dir)
    rows = []
    for path in sorted(item for item in output_dir.rglob("*") if item.is_file() and item.name != "manifest.csv"):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        rows.append(
            {
                "artifact": str(path.relative_to(output_dir)),
                "bytes": path.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(output_dir / "manifest.csv", index=False)
    return manifest

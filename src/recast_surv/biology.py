from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceWarning
from scipy.stats import rankdata
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .features import read_hpa_protein_priors


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    """Benjamini-Hochberg adjusted values, preserving missing entries."""
    result = pd.Series(np.nan, index=p_values.index, dtype=float)
    finite = p_values.dropna().astype(float)
    if finite.empty:
        return result
    order = np.argsort(finite.to_numpy())
    ranked = finite.to_numpy()[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1].clip(0.0, 1.0)
    result.loc[finite.index[order]] = adjusted
    return result


def _collapse_donor_state(metadata: pd.DataFrame, profiles: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    indexed = metadata.set_index("profile_id").loc[profiles.index]
    annotated = profiles.copy()
    annotated["_state"] = indexed["state"].astype(str).to_numpy()
    annotated["_patient"] = indexed["patient"].astype(str).to_numpy()
    collapsed = annotated.groupby(["_state", "_patient"], sort=True).median()
    return collapsed, indexed


def _specificity_effect(state_profiles: np.ndarray) -> np.ndarray:
    """Bounded within-state rank contrast against the other state anchors."""
    ranked = np.vstack(
        [rankdata(row, method="average") / max(len(row), 1) for row in state_profiles]
    )
    other_median = np.empty_like(ranked)
    for index in range(ranked.shape[0]):
        other_median[index] = np.median(np.delete(ranked, index, axis=0), axis=0)
    return ranked - other_median


def donor_bootstrap_markers(
    metadata: pd.DataFrame,
    profiles: pd.DataFrame,
    bulk_genes: set[str],
    hpa_path: Path,
    markers_per_state: int,
    candidate_multiplier: int,
    bootstrap_iterations: int,
    stable_frequency: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    collapsed, _ = _collapse_donor_state(metadata, profiles)
    donor_counts = collapsed.groupby(level=0).size()
    states = donor_counts.loc[donor_counts >= 2].index.astype(str).tolist()
    genes = [gene for gene in profiles.columns if gene in bulk_genes]
    donor_arrays = {
        state: collapsed.loc[state, genes].to_numpy(dtype=np.float32) for state in states
    }
    baseline_states = np.vstack([np.median(donor_arrays[state], axis=0) for state in states])
    baseline_effect = _specificity_effect(baseline_states)
    pool_size = markers_per_state * candidate_multiplier
    candidate_indices = sorted(
        set(
            np.argsort(baseline_effect[state_index])[::-1][:pool_size].tolist()[position]
            for state_index in range(len(states))
            for position in range(min(pool_size, len(genes)))
        )
    )
    candidate_genes = [genes[index] for index in candidate_indices]
    baseline_candidate = baseline_effect[:, candidate_indices]
    selected_counts = np.zeros_like(baseline_candidate, dtype=np.int32)
    effect_sum = np.zeros_like(baseline_candidate, dtype=np.float64)
    effect_square_sum = np.zeros_like(baseline_candidate, dtype=np.float64)
    effects = np.empty(
        (bootstrap_iterations, len(states), len(candidate_genes)), dtype=np.float32
    )
    rng = np.random.default_rng(seed)
    for iteration in range(bootstrap_iterations):
        sampled_states = []
        for state in states:
            values = donor_arrays[state][:, candidate_indices]
            draw = rng.integers(0, len(values), size=len(values))
            sampled_states.append(np.median(values[draw], axis=0))
        sampled_effect = _specificity_effect(np.vstack(sampled_states)).astype(np.float32)
        effects[iteration] = sampled_effect
        effect_sum += sampled_effect
        effect_square_sum += np.square(sampled_effect)
        for state_index in range(len(states)):
            positive = np.flatnonzero(sampled_effect[state_index] > 0)
            order = positive[np.argsort(sampled_effect[state_index, positive])[::-1]]
            selected_counts[state_index, order[:markers_per_state]] += 1
        if (iteration + 1) % 100 == 0:
            print(f"marker bootstrap {iteration + 1}/{bootstrap_iterations}", flush=True)
    hpa = pd.read_csv(hpa_path, sep="\t", usecols=["Gene", "Evidence"], low_memory=False)
    evidence = hpa.drop_duplicates("Gene").set_index("Gene")["Evidence"]
    weights = pd.Series(read_hpa_protein_priors(hpa_path)).reindex(candidate_genes).fillna(1.0)
    rows = []
    for state_index, state in enumerate(states):
        for gene_index, gene in enumerate(candidate_genes):
            frequency = selected_counts[state_index, gene_index] / bootstrap_iterations
            effect_values = effects[:, state_index, gene_index]
            baseline = float(baseline_candidate[state_index, gene_index])
            rows.append(
                {
                    "state": state,
                    "gene": gene,
                    "donors": int(donor_counts[state]),
                    "baseline_specificity_effect": baseline,
                    "bootstrap_selection_frequency": frequency,
                    "bootstrap_effect_mean": float(effect_sum[state_index, gene_index] / bootstrap_iterations),
                    "bootstrap_effect_sd": float(
                        np.sqrt(
                            max(
                                effect_square_sum[state_index, gene_index] / bootstrap_iterations
                                - np.square(effect_sum[state_index, gene_index] / bootstrap_iterations),
                                0.0,
                            )
                        )
                    ),
                    "bootstrap_effect_ci95_low": float(np.quantile(effect_values, 0.025)),
                    "bootstrap_effect_ci95_high": float(np.quantile(effect_values, 0.975)),
                    "hpa_evidence": str(evidence.get(gene, "Not listed")),
                    "hpa_weight": float(weights.loc[gene]),
                    "prominence_score": float(frequency * max(baseline, 0.0) * weights.loc[gene]),
                    "stable_marker": bool(
                        frequency >= stable_frequency and np.quantile(effect_values, 0.025) > 0
                    ),
                }
            )
    full = pd.DataFrame(rows)
    full["state_rank"] = full.groupby("state")["prominence_score"].rank(
        method="first", ascending=False
    ).astype(int)
    prominent = full.loc[full["state_rank"].le(10)].sort_values(["state", "state_rank"])
    summary = {
        "bootstrap_iterations": bootstrap_iterations,
        "seed": seed,
        "states": len(states),
        "shared_genes": len(genes),
        "candidate_pool_genes": len(candidate_genes),
        "markers_per_state_per_bootstrap": markers_per_state,
        "stable_frequency_threshold": stable_frequency,
        "stable_state_gene_pairs": int(full["stable_marker"].sum()),
        "unique_prominent_genes": int(prominent["gene"].nunique()),
        "method": "donor-state median; candidate pool top 5x markers by bounded within-state rank contrast; donor bootstrap",
    }
    return full, prominent, summary


def _clinical_frame(features: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in features if column.startswith("clinical__")]
    return features[columns].copy()


def adjusted_cox_associations(
    candidates: pd.DataFrame,
    clinical: pd.DataFrame,
    outcomes: pd.DataFrame,
    family: str,
) -> pd.DataFrame:
    rows = []
    aligned_outcomes = outcomes.loc[clinical.index]
    base = clinical.copy()
    base = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(base),
        index=base.index,
        columns=base.columns,
    )
    age_columns = [column for column in base if column.endswith("age_years")]
    for column in age_columns:
        base[column] = (base[column] - base[column].mean()) / (base[column].std(ddof=0) + 1e-8)
    for candidate in candidates.columns:
        values = pd.to_numeric(candidates.loc[base.index, candidate], errors="coerce")
        values = values.fillna(values.median())
        sd = float(values.std(ddof=0))
        if not np.isfinite(sd) or sd <= 1e-10:
            rows.append({"feature": candidate, "family": family, "error": "constant_feature"})
            continue
        frame = base.copy()
        frame["candidate"] = (values - values.mean()) / sd
        frame["time_days"] = aligned_outcomes["time_days"].to_numpy(dtype=float)
        frame["event"] = aligned_outcomes["event"].to_numpy(dtype=int)
        try:
            model = CoxPHFitter(penalizer=0.0)
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                model.fit(frame, duration_col="time_days", event_col="event", robust=True)
            item = model.summary.loc["candidate"]
            rows.append(
                {
                    "feature": candidate,
                    "family": family,
                    "log_hazard_ratio_per_sd": float(item["coef"]),
                    "hazard_ratio_per_sd": float(item["exp(coef)"]),
                    "ci95_low": float(item["exp(coef) lower 95%"]),
                    "ci95_high": float(item["exp(coef) upper 95%"]),
                    "p_value": float(item["p"]),
                    "n": len(frame),
                    "events": int(frame["event"].sum()),
                    "error": "",
                }
            )
        except (ValueError, ArithmeticError, ConvergenceWarning) as exc:
            rows.append({"feature": candidate, "family": family, "error": f"{type(exc).__name__}: {exc}"})
    result = pd.DataFrame(rows)
    result["fdr_bh"] = benjamini_hochberg(result.get("p_value", pd.Series(dtype=float)))
    return result


def split_effect_stability(
    candidates: pd.DataFrame,
    clinical: pd.DataFrame,
    outcomes: pd.DataFrame,
    split_manifest: pd.DataFrame,
    full_effects: pd.Series,
) -> pd.DataFrame:
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sksurv.util import Surv

    rows = []
    pairs = split_manifest[["repeat", "fold"]].drop_duplicates().sort_values(["repeat", "fold"])
    for feature_number, candidate in enumerate(candidates.columns, start=1):
        coefficients = []
        for repeat, fold in pairs.itertuples(index=False):
            block = split_manifest.loc[
                split_manifest["repeat"].eq(repeat) & split_manifest["fold"].eq(fold)
            ]
            ids = block.loc[block["role"].eq("train"), "sample_id"].astype(str)
            frame = pd.concat([clinical.loc[ids], candidates.loc[ids, [candidate]]], axis=1)
            imputed = SimpleImputer(strategy="median").fit_transform(frame)
            keep = np.std(imputed, axis=0) > 1e-10
            if not keep[-1]:
                continue
            scaled = StandardScaler().fit_transform(imputed[:, keep])
            y = Surv.from_arrays(
                outcomes.loc[ids, "event"].astype(bool).to_numpy(),
                outcomes.loc[ids, "time_days"].to_numpy(dtype=float),
            )
            try:
                model = CoxPHSurvivalAnalysis(alpha=0.1, n_iter=200).fit(scaled, y)
                coefficients.append(float(model.coef_[-1]))
            except (ValueError, ArithmeticError):
                continue
        values = np.asarray(coefficients, dtype=float)
        full = float(full_effects.get(candidate, np.nan))
        rows.append(
            {
                "feature": candidate,
                "valid_training_splits": len(values),
                "split_log_hr_median": float(np.median(values)) if len(values) else np.nan,
                "split_log_hr_q25": float(np.quantile(values, 0.25)) if len(values) else np.nan,
                "split_log_hr_q75": float(np.quantile(values, 0.75)) if len(values) else np.nan,
                "direction_consistency": float(np.mean(np.sign(values) == np.sign(full)))
                if len(values) and np.isfinite(full) and full != 0
                else np.nan,
            }
        )
        if feature_number % 25 == 0:
            print(f"effect stability {feature_number}/{len(candidates.columns)}", flush=True)
    return pd.DataFrame(rows)


def run_biology_analysis(
    root: Path,
    hpa_path: Path,
    settings: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    output = root / "biology"
    output.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(root / "reference/metadata.csv")
    profiles = pd.read_parquet(root / "reference/profiles.parquet")
    expression = pd.read_parquet(root / "cohort/expression.parquet")
    features = pd.read_parquet(root / "features/features.parquet")
    outcomes = pd.read_csv(root / "features/outcomes.csv", index_col="sample_id")
    splits = pd.read_csv(root / "benchmark/split_manifest.csv")
    marker_full, prominent, marker_summary = donor_bootstrap_markers(
        metadata,
        profiles,
        set(expression.columns),
        hpa_path,
        markers_per_state=int(settings.get("markers_per_state", 40)),
        candidate_multiplier=int(settings.get("candidate_multiplier", 5)),
        bootstrap_iterations=int(settings.get("marker_bootstrap_iterations", 500)),
        stable_frequency=float(settings.get("stable_frequency", 0.70)),
        seed=seed,
    )
    marker_full.to_csv(output / "marker_stability_all.csv", index=False)
    prominent.to_csv(output / "prominent_markers.csv", index=False)
    (output / "marker_stability_summary.json").write_text(
        json.dumps(marker_summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    gene_names = prominent["gene"].drop_duplicates().tolist()
    gene_frame = expression[gene_names]
    clinical = _clinical_frame(features)
    gene_associations = adjusted_cox_associations(gene_frame, clinical, outcomes, "gene")
    gene_stability = split_effect_stability(
        gene_frame,
        clinical,
        outcomes,
        splits,
        gene_associations.set_index("feature")["log_hazard_ratio_per_sd"],
    )
    gene_associations = gene_associations.merge(gene_stability, on="feature", how="left")
    marker_annotation = prominent.sort_values("prominence_score", ascending=False).drop_duplicates("gene")
    gene_associations = gene_associations.merge(
        marker_annotation[
            [
                "gene",
                "state",
                "state_rank",
                "prominence_score",
                "bootstrap_selection_frequency",
                "stable_marker",
                "hpa_evidence",
            ]
        ],
        left_on="feature",
        right_on="gene",
        how="left",
    ).sort_values(["fdr_bh", "p_value"], na_position="last")
    gene_associations.to_csv(output / "exploratory_gene_survival.csv", index=False)

    pathway_columns = [column for column in features if column.startswith("pathway__")]
    pathway_frame = features[pathway_columns]
    pathway_associations = adjusted_cox_associations(pathway_frame, clinical, outcomes, "pathway")
    pathway_stability = split_effect_stability(
        pathway_frame,
        clinical,
        outcomes,
        splits,
        pathway_associations.set_index("feature")["log_hazard_ratio_per_sd"],
    )
    pathway_associations = pathway_associations.merge(pathway_stability, on="feature", how="left")
    pathway_associations = pathway_associations.sort_values(["fdr_bh", "p_value"], na_position="last")
    pathway_associations.to_csv(output / "exploratory_pathway_survival.csv", index=False)
    summary = {
        **marker_summary,
        "genes_tested_for_survival": len(gene_associations),
        "gene_fdr_lt_0_05": int(gene_associations["fdr_bh"].lt(0.05).sum()),
        "pathways_tested_for_survival": len(pathway_associations),
        "pathway_fdr_lt_0_05": int(pathway_associations["fdr_bh"].lt(0.05).sum()),
        "survival_analysis_status": "exploratory_internal_only",
        "external_gene_validation": "blocked_untraceable_platform_annotation",
        "association_model": "clinical-adjusted Cox; effect per one SD; BH FDR within gene and pathway families",
        "stability_model": "ridge Cox coefficients across the 50 locked outer training sets",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary

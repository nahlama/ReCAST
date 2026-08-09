from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .pathways import rank_pathway_scores, read_gmt, select_pathways
from .recast import ReCASTEstimator


def read_hpa_protein_priors(path: Path) -> dict[str, float]:
    table = pd.read_csv(path, sep="\t", usecols=["Gene", "Evidence"], low_memory=False)
    evidence_weights = {
        "Evidence at protein level": 1.15,
        "Evidence at transcript level": 1.00,
        "Predicted": 0.75,
        "Uncertain": 0.60,
    }
    weights = table["Evidence"].map(evidence_weights).fillna(1.0)
    return dict(zip(table["Gene"].astype(str), weights.astype(float)))


def clinical_features(cohort: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=cohort.index)
    result["clinical__age_years"] = pd.to_numeric(cohort["age_years"], errors="coerce")
    result["clinical__sex_male"] = pd.to_numeric(cohort["sex_male"], errors="coerce")
    # Reference coding avoids exact collinearity in small training folds.
    result["clinical__histology_ESCC"] = cohort["histology"].eq("ESCC").astype(float)
    for stage in ("II", "III", "IV", "UNKNOWN"):
        result[f"clinical__stage_{stage}"] = cohort["stage_group"].eq(stage).astype(float)
    return result.astype(np.float32)


def compact_recast_features(scores: pd.DataFrame) -> pd.DataFrame:
    """Convert constrained state masses into prespecified compositional balances."""
    state_columns = [column for column in scores if column.startswith("state__")]
    if not state_columns:
        raise ValueError("ReCAST scores contain no state masses")
    states = scores[state_columns].clip(lower=0.0)

    def mass(names: tuple[str, ...]) -> pd.Series:
        columns = [f"state__{name}" for name in names if f"state__{name}" in states]
        return states[columns].sum(axis=1) if columns else pd.Series(0.0, index=states.index)

    lymphoid = mass(("B", "CD4Tconv", "CD8Tex", "Plasma", "Tprolif", "Treg"))
    myeloid = mass(("DC", "Mast", "Mono_Macro"))
    stromal = mass(("Endothelial", "Fibroblasts", "Pericytes"))
    malignant = mass(("Malignant",))
    regulatory = mass(("CD8Tex", "Treg"))
    conventional_t = mass(("CD4Tconv", "Tprolif"))
    epsilon = 1e-6
    result = pd.DataFrame(index=scores.index)
    result["balance__tumour_vs_microenvironment"] = np.log(
        (malignant + epsilon) / (lymphoid + myeloid + stromal + epsilon)
    )
    result["balance__immune_vs_stromal"] = np.log(
        (lymphoid + myeloid + epsilon) / (stromal + epsilon)
    )
    result["balance__myeloid_vs_lymphoid"] = np.log(
        (myeloid + epsilon) / (lymphoid + epsilon)
    )
    result["balance__regulatory_exhausted_vs_conventional_t"] = np.log(
        (regulatory + epsilon) / (conventional_t + epsilon)
    )
    state_total = states.sum(axis=1)
    proportions = states.div(state_total.replace(0.0, np.nan), axis=0).fillna(0.0)
    entropy = -(proportions * np.log(proportions.clip(lower=epsilon))).sum(axis=1)
    result["balance__state_diversity"] = entropy / np.log(max(len(state_columns), 2))
    result["recast__matched_mass"] = state_total
    for column in (
        "recast__unknown_score",
        "recast__reconstruction_cosine",
        "recast__solver_residual",
    ):
        if column in scores:
            result[column] = scores[column]
    return result.astype(np.float32)


def build_features(
    cohort: pd.DataFrame,
    expression: pd.DataFrame,
    reference_metadata: pd.DataFrame,
    reference_profiles: pd.DataFrame,
    pathway_paths: list[Path],
    settings: dict[str, Any],
    protein_prior_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    recast = ReCASTEstimator(
        backend=str(settings.get("backend", "unbalanced_ot")),
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
    protein_priors = read_hpa_protein_priors(protein_prior_path) if protein_prior_path else None
    recast.fit(
        reference_metadata,
        reference_profiles,
        set(expression.columns),
        gene_reliability=protein_priors,
    )
    raw_recast_scores = recast.transform(expression)
    recast_scores = compact_recast_features(raw_recast_scores)
    interpretation_scores = raw_recast_scores[
        [column for column in raw_recast_scores if column.startswith("state__")]
    ].rename(columns=lambda column: column.replace("state__", "interpretation_state__", 1))

    pathway_library = read_gmt(pathway_paths)
    selected = select_pathways(
        pathway_library,
        set(expression.columns),
        min_genes=int(settings["min_pathway_genes"]),
        max_genes=int(settings["max_pathway_genes"]),
        max_pathways=settings.get("max_pathways"),
    )
    pathway_scores = rank_pathway_scores(expression, selected)
    clinical = clinical_features(cohort)
    features = pd.concat([clinical, pathway_scores, recast_scores, interpretation_scores], axis=1)
    if features.columns.duplicated().any():
        raise ValueError("Feature names are not unique")
    outcomes = cohort[["time_days", "event", "histology", "patient"]].copy()
    diagnostics = {
        "samples": int(len(features)),
        "features": int(features.shape[1]),
        "clinical_features": int(clinical.shape[1]),
        "pathway_features": int(pathway_scores.shape[1]),
        "recast_features": int(recast_scores.shape[1]),
        "interpretation_state_features": int(interpretation_scores.shape[1]),
        "selected_pathways": list(selected),
        "recast": recast.diagnostics().to_dict(),
        "missing_values": int(features.isna().sum().sum()),
    }
    return features, outcomes, diagnostics

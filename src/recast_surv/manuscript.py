from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


MODEL_LABELS = {
    "elastic_net_cox": "Elastic-net Cox",
    "survivalpfn": "SurvivalPFN",
    "survpfn": "SurvPFN (June 2026)",
    "random_survival_forest": "Random survival forest",
    "xgb_aft": "XGBoost-AFT",
}


def _median_iqr(values: pd.Series) -> str:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return "NA"
    return f"{numeric.median():.1f} ({numeric.quantile(0.25):.1f}-{numeric.quantile(0.75):.1f})"


def _count_percent(mask: pd.Series) -> str:
    count = int(mask.fillna(False).sum())
    return f"{count} ({100.0 * count / len(mask):.1f}%)"


def _external_subset(audited: pd.DataFrame) -> pd.DataFrame:
    overlap = set(
        audited.loc[audited["cohort"].eq("GSE53624"), "patient_id"].astype(str)
    )
    return audited.loc[
        audited["cohort"].eq("GSE53625")
        & ~audited["patient_id"].astype(str).isin(overlap)
    ].copy()


def cohort_characteristics(tcga: pd.DataFrame, external: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []

    def add(characteristic: str, tcga_value: str, external_value: str) -> None:
        rows.append(
            {
                "characteristic": characteristic,
                "TCGA_ESCA_development": tcga_value,
                "GSE53625_locked_external": external_value,
            }
        )

    add("Patients, n", str(len(tcga)), str(len(external)))
    add("Deaths, n (%)", _count_percent(tcga["event"].eq(1)), _count_percent(external["event"].eq(1)))
    add("Age, median (IQR), years", _median_iqr(tcga["age_years"]), _median_iqr(external["age"]))
    add("Male, n (%)", _count_percent(tcga["sex_male"].eq(1)), _count_percent(external["sex"].str.casefold().eq("male")))
    for histology in ("ESCC", "EAC"):
        add(
            f"Histology: {histology}, n (%)",
            _count_percent(tcga["histology"].eq(histology)),
            _count_percent(external["histology"].eq(histology)),
        )
    for stage in ("I", "II", "III", "IV", "UNKNOWN"):
        tcga_stage = tcga["stage_group"].fillna("UNKNOWN").astype(str).str.upper()
        external_stage = external["stage"].fillna("UNKNOWN").astype(str).str.upper()
        add(
            f"Stage {stage}, n (%)",
            _count_percent(tcga_stage.eq(stage)),
            _count_percent(external_stage.eq(stage)),
        )
    add("Observed time, median (IQR), days", _median_iqr(tcga["time_days"]), _median_iqr(external["time_days"]))
    return pd.DataFrame(rows)


def internal_model_table(root: Path) -> pd.DataFrame:
    challenger = root / "challengers/survpfn/clinical_model_comparison_internal.csv"
    if challenger.exists():
        data = pd.read_csv(challenger)
    else:
        data = pd.read_csv(root / "benchmark/summary.csv").query("panel == 'clinical'")
    columns = [
        "model",
        "harrell_c_mean",
        "harrell_c_std",
        "uno_c_mean",
        "uno_c_std",
        "integrated_brier_mean",
        "integrated_brier_std",
        "uno_c_count",
    ]
    table = data[columns].copy()
    table.insert(0, "model_label", table["model"].map(MODEL_LABELS).fillna(table["model"]))
    table["uno_c_rank"] = table["uno_c_mean"].rank(method="min", ascending=False).astype(int)
    return table.sort_values(["uno_c_rank", "model_label"])


def incremental_table(root: Path) -> pd.DataFrame:
    summary = pd.read_csv(root / "benchmark/summary.csv")
    table = summary.loc[
        summary["model"].eq("survivalpfn"),
        ["panel", "uno_c_mean", "uno_c_std", "integrated_brier_mean", "integrated_brier_std"],
    ].copy()
    paired = pd.read_csv(root / "benchmark/paired_comparisons.csv")
    for metric in ("uno_c", "integrated_brier"):
        block = paired.loc[
            paired["model"].eq("survivalpfn") & paired["metric"].eq(metric),
            ["candidate_panel", "mean_improvement", "ci95_low", "ci95_high"],
        ].rename(
            columns={
                "candidate_panel": "panel",
                "mean_improvement": f"paired_{metric}_improvement",
                "ci95_low": f"paired_{metric}_ci95_low",
                "ci95_high": f"paired_{metric}_ci95_high",
            }
        )
        table = table.merge(block, on="panel", how="left")
    return table.sort_values("panel")


def external_model_table(root: Path) -> pd.DataFrame:
    extended = root / "challengers/survpfn/clinical_model_comparison_external.csv"
    source = extended if extended.exists() else root / "external_validation/clinical_model_comparison.csv"
    data = pd.read_csv(source)
    keep = [
        "model",
        "n_external",
        "events_external",
        "harrell_c",
        "uno_c",
        "uno_c_ci95_low",
        "uno_c_ci95_high",
        "integrated_brier",
    ]
    table = data[keep].copy()
    table.insert(0, "model_label", table["model"].map(MODEL_LABELS).fillna(table["model"]))
    return table.sort_values("uno_c", ascending=False)


def _write_markdown_table(frame: pd.DataFrame, path: Path) -> None:
    path.write_text(frame.to_markdown(index=False, floatfmt=".3f") + "\n", encoding="utf-8")


def _methods_text(counts: dict[str, Any]) -> str:
    return f"""# Methods draft

## Study design and data roles

This retrospective methodological study evaluated whether donor-aware cell-state reference information improved overall-survival prediction in TCGA esophageal carcinoma. The method-literature eligibility cutoff was 30 June 2026. Patient-level TCGA expression and outcomes were used for model development; single-cell datasets were used only to construct outcome-free reference anchors; HPA evidence supplied marker-reliability priors; and MSigDB 2026.1 gene sets supplied pathway definitions.

## Cohort and endpoint

One primary tumour per TCGA patient was retained. Overall survival was represented by time in days and a death indicator. The final development cohort contained {counts['tcga_n']} patients and {counts['tcga_events']} deaths. The external clinical evaluation used {counts['external_n']} GSE53625 patients absent from GSE53624, including {counts['external_events']} deaths.

## Donor-hierarchical reference construction

Cell-level GSE160269 counts and GSE154763 normalized expression were summarized within donor and annotated state before state-level anchors were calculated. This prevented donors with more cells from dominating the reference. The reference summarized {counts['reference_cells']:,} cells from {counts['reference_patients']} patients into {counts['reference_profiles']} donor/state profiles, with {counts['reference_annotated_states']} annotated states and {counts['reference_modeled_states']} states passing the modeling threshold.

## Robust reference projection

Bulk expression was projected onto the reference using non-negative least squares with HPA-derived marker weights and iterative Huber residual reweighting. A normalized reconstruction residual was retained as an unknown-expression score; it was not interpreted as an unknown-cell fraction. Entropic unbalanced optimal transport was retained as a prespecified technical comparator. Estimators were compared in clean, platform-shift, and unknown-component simulations with 500 mixtures per scenario.

## Prediction features and models

The clinical panel comprised age, sex, histology, and stage indicators. Compact ReCAST features contained compositional balances, diversity, matched-reference support, reconstruction cosine, and the residual unknown-expression score. Twenty-four outcome-independent pathway scores were optional. Raw state proportions were interpretation-only. SurvivalPFN was the prespecified advanced small-sample model. Elastic-net Cox, random survival forest, XGBoost-AFT, and the cutoff-eligible June-2026 SurvPFN were comparators. SurvPFN was evaluated only on its supported seven-variable clinical panel.

## Internal validation and metrics

Models were evaluated using 10 repetitions of five-fold patient-level cross-validation. A locked split manifest ensured that every patient appeared once in a test fold per repetition. Imputation, constant filtering, scaling, and hyperparameter selection were fitted using training data only. The primary endpoint was Uno's IPCW concordance with censoring estimated from the corresponding outer training set. Harrell concordance, time-dependent AUC, IPCW Brier scores, integrated Brier score, and calibration intercept and slope were safeguards. Paired outer-fold differences were summarized with 5,000 bootstrap replicates.

## External evaluation

GSE53625 was audited as a SuperSeries containing all GSE53624 patients. Those overlapping patients were excluded before model scoring. Models were fitted using the complete TCGA development cohort and evaluated once on the remaining 60 patients. External outcomes were not used for feature processing, model selection, or recalibration. Confidence intervals used 5,000 patient-level bootstrap samples. External expression was not analyzed because supplied gene-level matrices were not traceable to a validated platform annotation.

## Reproducibility and claim boundary

Source-file hashes, patient splits, software environments, model-source commits, checkpoint hashes, fold predictions, and external predictions were retained. Results are interpreted as methodological and exploratory. The analysis does not establish clinical utility, deployment readiness, causal cell-state effects, or superiority of a single survival model.
{counts.get('biology_methods', '')}
"""


def _biology_results(tables: dict[str, pd.DataFrame], summary: dict[str, Any] | None) -> str:
    if not summary:
        return ""
    genes = tables["gene_associations"].dropna(subset=["p_value"]).sort_values("p_value")
    pathways = tables["pathway_associations"].dropna(subset=["p_value"]).sort_values("p_value")
    gene = genes.iloc[0]
    pathway = pathways.iloc[0]
    pathway_name = str(pathway["feature"]).replace("pathway__", "").replace("_", " ")
    return f"""

## Stable markers and exploratory biological associations

Across 500 donor-bootstrap replicates, {summary['stable_state_gene_pairs']} state-gene pairs passed the prespecified stability threshold and {summary['unique_prominent_genes']} unique genes appeared among the ten highest-ranked markers per state. Prominent markers included canonical lineage signals such as VPREB3 and CD19 for B cells, GZMK/GZMM for conventional T cells, KRT15/CLDN4 for malignant epithelial cells, and TPSAB1/GATA2 for mast cells.

No tested gene or pathway passed Benjamini-Hochberg FDR < 0.05. The smallest nominal gene association was {gene['feature']} (clinical-adjusted HR {gene['hazard_ratio_per_sd']:.2f}, 95% CI {gene['ci95_low']:.2f}-{gene['ci95_high']:.2f}, nominal P={gene['p_value']:.3f}, FDR={gene['fdr_bh']:.3f}). The smallest nominal pathway association was {pathway_name} (HR {pathway['hazard_ratio_per_sd']:.2f}, 95% CI {pathway['ci95_low']:.2f}-{pathway['ci95_high']:.2f}, nominal P={pathway['p_value']:.3f}, FDR={pathway['fdr_bh']:.3f}). Pathway labels describe database gene sets and must not be interpreted as evidence of the named infection or disease process. These associations are hypothesis-generating and lack external expression validation.
"""


def _results_text(
    tables: dict[str, pd.DataFrame], biology_summary: dict[str, Any] | None = None
) -> str:
    internal = tables["internal"].set_index("model")
    external = tables["external"].set_index("model")
    increment = tables["incremental"].set_index("panel")
    return f"""# Results draft

## Technical estimator selection

Robust NNLS passed the prespecified technical gate and outperformed unbalanced optimal transport in clean, platform-shift, and unknown-component simulations. The single-cell-derived representation was therefore constructed with robust NNLS; UOT was retained as a negative ablation.

## Internal survival performance

Clinical elastic-net Cox produced the highest mean Uno C ({internal.loc['elastic_net_cox', 'uno_c_mean']:.3f}). SurvivalPFN and June-2026 SurvPFN produced mean Uno C values of {internal.loc['survivalpfn', 'uno_c_mean']:.3f} and {internal.loc['survpfn', 'uno_c_mean']:.3f}, respectively. Their integrated Brier scores were {internal.loc['survivalpfn', 'integrated_brier_mean']:.3f} and {internal.loc['survpfn', 'integrated_brier_mean']:.3f}. Thus, the newer SurvPFN did not improve the locked internal evaluation.

## Incremental value of ReCAST

For SurvivalPFN, adding ReCAST changed mean Uno C from {increment.loc['clinical', 'uno_c_mean']:.3f} to {increment.loc['clinical_plus_recast', 'uno_c_mean']:.3f} and changed IBS from {increment.loc['clinical', 'integrated_brier_mean']:.3f} to {increment.loc['clinical_plus_recast', 'integrated_brier_mean']:.3f}. The cell-state-derived representation therefore failed the confirmatory incremental-prediction gate.

## Locked external evaluation

The locked external subset contained 60 non-overlapping patients. SurvivalPFN achieved Uno C {external.loc['survivalpfn', 'uno_c']:.3f}, elastic-net Cox {external.loc['elastic_net_cox', 'uno_c']:.3f}, and SurvPFN {external.loc['survpfn', 'uno_c']:.3f}. The corresponding IBS values were {external.loc['survivalpfn', 'integrated_brier']:.3f}, {external.loc['elastic_net_cox', 'integrated_brier']:.3f}, and {external.loc['survpfn', 'integrated_brier']:.3f}. Rankings differed between internal and external analyses, uncertainty was substantial, and calibration was inadequate; no model can be described as most accurate or clinically ready.
{_biology_results(tables, biology_summary)}
"""


def _checklist() -> pd.DataFrame:
    rows = [
        ("Title/abstract", "Partial", "Add structured abstract after journal selection"),
        ("Data sources and study dates", "Complete", "MANUSCRIPT_DESIGN.md and Table 1"),
        ("Participants and eligibility", "Complete", "Cohort code and cohort/summary.json"),
        ("Outcome definition", "Complete", "Overall survival time and death indicator"),
        ("Predictor definition and availability", "Complete", "features/diagnostics.json and methods draft"),
        ("Sample size and event count", "Complete", "182 patients and 77 deaths"),
        ("Missing-data handling", "Complete", "Training-fold median imputation"),
        ("Model specification and tuning", "Complete", "Locked config, source code, inner selection artifacts"),
        ("Internal validation", "Complete", "10 x 5-fold locked patient splits"),
        ("External evaluation", "Complete", "60 non-overlapping patients; 33 deaths"),
        ("Performance measures and uncertainty", "Complete", "Uno/Harrell C, AUC, Brier/IBS, calibration, bootstrap CI"),
        ("Model output availability", "Complete", "Per-fold and external prediction CSV files"),
        ("Fairness/subgroup evaluation", "Complete (exploratory)", "EAC/ESCC descriptive metrics and interaction tests; underpowered for fairness claims"),
        ("Clinical utility/decision curve", "Complete (exploratory)", "IPCW decision curves at 1 and 3 years; thresholds were not prospectively selected"),
        ("External molecular evaluation", "Complete", "GENCODE 50 exact probe remap; 75 prespecified genes and 24 pathways tested; none passed FDR"),
        ("Risk-of-bias assessment", "Complete author assessment", "Domain-level PROBAST+AI evidence map; independent assessor still required"),
        ("Registration/protocol", "Partial", "Versioned protocol and locked config exist; no prospective registration"),
        ("Code/data availability", "Partial", "Repository is reproducible locally; public archive/DOI pending"),
        ("Funding/conflicts/author roles", "Pending author input", "Cannot be inferred from analysis files"),
    ]
    return pd.DataFrame(rows, columns=["reporting_domain", "status", "evidence_or_action"])


def _q1_extension_methods() -> str:
    return """

## Locked secondary Q1-strengthening analyses

These analyses were frozen after completion of the primary benchmark and were designated secondary or exploratory. Projection generalization was evaluated by excluding each of 67 reference donors from anchor construction, generating five pseudo-bulk mixtures from that donor's state profiles, and evaluating clean and platform-shift scenarios. The locked HPA-weighted robust NNLS estimator was compared with ordinary NNLS, ridge NNLS, simplex-constrained NNLS, CIBERSORT-style linear nu-SVR, and HPA-weighted unbalanced optimal transport using identical held-out mixtures. State-composition MAE was the primary projection metric, and uncertainty used donor-clustered bootstrap resampling.

GPL18109 probe sequences were remapped by full-length exact sequence matching on both strands against GENCODE 50 transcript sequences (GRCh38.p14). Only probe sequences mapping to exactly one GENCODE gene were retained. Multiple uniquely mapped probes for one gene were collapsed by their median. The 92 internally prespecified prominent genes and 24 pathways were tested in the locked 60-patient GSE53625 subset using clinical-adjusted ridge Cox models with a fixed penalizer of 0.05. No external feature selection or recalibration was performed, and BH correction was applied within the gene and pathway families.

Exploratory IPCW decision curves were estimated at one and three years. Internal curves pooled outer-test predictions within each repeat while using the corresponding fold-training censoring distribution. External curves used locked predictions without recalibration. EAC and ESCC discrimination was summarized descriptively, and risk-by-histology interaction terms were tested. External low, intermediate, and high risk groups used tertile cutpoints derived exclusively from TCGA out-of-fold predictions. A domain-level PROBAST+AI author self-assessment documented residual bias and applicability concerns; it was not treated as independent certification.
"""


def _q1_extension_results(root: Path) -> str:
    projection = pd.read_csv(root / "q1_extension/projection_validation/summary.csv").set_index(["scenario", "method"])
    mapping = json.loads((root / "q1_extension/external_reannotation/mapping_summary.json").read_text(encoding="utf-8"))
    confirmation = json.loads((root / "q1_extension/external_reannotation/confirmation_summary.json").read_text(encoding="utf-8"))
    genes = pd.read_csv(root / "q1_extension/external_reannotation/external_prominent_gene_confirmation.csv").sort_values("p_value")
    interactions = pd.read_csv(root / "q1_extension/clinical_utility/histology_interactions.csv")
    clean = "heldout_donor_clean"
    shifted = "heldout_donor_platform_shift"
    top_gene = genes.iloc[0]
    return f"""

## Donor-held-out projection validation and comparator benchmark

Across 67 completely held-out donors, simplex-constrained NNLS produced the lowest mean state-composition MAE in both clean ({projection.loc[(clean, 'simplex_nnls'), 'state_mae_mean']:.3f}) and platform-shift ({projection.loc[(shifted, 'simplex_nnls'), 'state_mae_mean']:.3f}) mixtures. The corresponding locked robust-NNLS MAE values were {projection.loc[(clean, 'robust_nnls_hpa'), 'state_mae_mean']:.3f} and {projection.loc[(shifted, 'robust_nnls_hpa'), 'state_mae_mean']:.3f}. Robust NNLS remained markedly better than unbalanced optimal transport but did not outperform every matched comparator. Therefore, the extension supports the donor-held-out validation framework, not a universal ReCAST superiority claim.

## Versioned external molecular confirmation

Exact matching against GENCODE 50 assigned {mapping['status_counts']['unique_gene_exact_match']:,} probe features uniquely to one gene. The locked external matrix contained {confirmation['mapped_external_genes']:,} mapped genes, including {confirmation['tcga_external_gene_overlap']:,} shared with TCGA and {confirmation['prominent_genes_available']} of 92 prespecified prominent genes. Direction was concordant for {100 * confirmation['direction_concordance_fraction']:.1f}% of testable genes. No gene or pathway passed BH FDR < 0.05. The smallest nominal external gene association was {top_gene['feature']} (ridge-Cox HR {top_gene['hazard_ratio_per_sd']:.2f}, 95% CI {top_gene['ci95_low']:.2f}-{top_gene['ci95_high']:.2f}, nominal P={top_gene['p_value']:.3f}, FDR={top_gene['fdr_bh']:.3f}). Thus, reannotation resolved the technical mapping barrier but did not validate a prognostic molecular signature.

## Exploratory clinical utility and subgroup evaluation

Decision curves showed threshold-dependent net benefit without consistent dominance by one model, and external predictions were not recalibrated. Discrimination was descriptively higher in ESCC than EAC for all evaluated models, but none of {len(interactions)} risk-by-histology interaction tests passed BH FDR < 0.05. Because the external cohort contained ESCC only, histology transportability and fairness remain unresolved. The PROBAST+AI-aligned author assessment judged overall predictive-performance risk of bias and clinical applicability as high, principally because of sample size, subgroup limitations, retrospective design, and inadequate external calibration.
"""


def make_manuscript_package(root: str | Path, output_dir: str | Path) -> pd.DataFrame:
    root = Path(root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tcga = pd.read_csv(root / "cohort/clinical.csv")
    audited = pd.read_csv(root / "external_validation/audit/external_clinical.csv")
    external = _external_subset(audited)
    tables = {
        "cohort": cohort_characteristics(tcga, external),
        "technical": pd.read_csv(root / "technical_validation/summary.csv"),
        "internal": internal_model_table(root),
        "incremental": incremental_table(root),
        "external": external_model_table(root),
        "checklist": _checklist(),
    }
    filenames = {
        "cohort": "table1_cohort_characteristics",
        "technical": "table2_technical_validation",
        "internal": "table3_internal_model_performance",
        "incremental": "table4_incremental_recast",
        "external": "table5_external_validation",
        "checklist": "reporting_checklist_tripod_ai_probast_ai",
    }
    biology_summary = None
    if (root / "biology/summary.json").exists():
        biology_summary = json.loads((root / "biology/summary.json").read_text(encoding="utf-8"))
        prominent = pd.read_csv(root / "biology/prominent_markers.csv")
        tables["prominent_markers"] = prominent.loc[prominent["state_rank"].le(3)]
        tables["gene_associations"] = pd.read_csv(
            root / "biology/exploratory_gene_survival.csv"
        )
        tables["pathway_associations"] = pd.read_csv(
            root / "biology/exploratory_pathway_survival.csv"
        )
        filenames.update(
            {
                "prominent_markers": "table6_prominent_state_markers",
                "gene_associations": "supplementary_table1_exploratory_gene_survival",
                "pathway_associations": "supplementary_table2_exploratory_pathway_survival",
            }
        )
    q1_available = (root / "q1_extension" / "external_reannotation" / "confirmation_summary.json").exists()
    if q1_available:
        tables.update(
            {
                "projection_validation": pd.read_csv(root / "q1_extension/projection_validation/summary.csv"),
                "projection_comparisons": pd.read_csv(root / "q1_extension/projection_validation/paired_comparisons.csv"),
                "external_gene_confirmation": pd.read_csv(root / "q1_extension/external_reannotation/external_prominent_gene_confirmation.csv"),
                "external_pathway_confirmation": pd.read_csv(root / "q1_extension/external_reannotation/external_pathway_confirmation.csv"),
                "histology_subgroups": pd.read_csv(root / "q1_extension/clinical_utility/histology_subgroup_performance.csv"),
                "histology_interactions": pd.read_csv(root / "q1_extension/clinical_utility/histology_interactions.csv"),
                "external_risk_group_calibration": pd.read_csv(root / "q1_extension/clinical_utility/external_risk_group_calibration.csv"),
                "probast_ai_domain_assessment": pd.read_csv(root / "q1_extension/probast_ai/probast_ai_domain_self_assessment.csv"),
            }
        )
        filenames.update(
            {
                "projection_validation": "supplementary_table3_donor_heldout_projection",
                "projection_comparisons": "supplementary_table4_projection_comparators",
                "external_gene_confirmation": "supplementary_table5_external_gene_confirmation",
                "external_pathway_confirmation": "supplementary_table6_external_pathway_confirmation",
                "histology_subgroups": "supplementary_table7_histology_subgroups",
                "histology_interactions": "supplementary_table8_histology_interactions",
                "external_risk_group_calibration": "supplementary_table9_external_risk_group_calibration",
                "probast_ai_domain_assessment": "supplementary_table10_probast_ai_author_assessment",
            }
        )
    manifest_rows = []
    for name, frame in tables.items():
        stem = filenames[name]
        frame.to_csv(output_dir / f"{stem}.csv", index=False)
        _write_markdown_table(frame, output_dir / f"{stem}.md")
        manifest_rows.append({"artifact": name, "csv": f"{stem}.csv", "markdown": f"{stem}.md", "rows": len(frame)})
    reference = json.loads((root / "reference/summary.json").read_text(encoding="utf-8"))
    counts = {
        "tcga_n": len(tcga),
        "tcga_events": int(tcga["event"].sum()),
        "external_n": len(external),
        "external_events": int(external["event"].sum()),
        "reference_cells": int(reference["cells_summarized"]),
        "reference_patients": int(reference["patients"]),
        "reference_profiles": int(reference["profiles"]),
        "reference_annotated_states": int(reference["states"]),
        "reference_modeled_states": int(biology_summary["states"]) if biology_summary else 13,
    }
    if biology_summary:
        counts["biology_methods"] = """

## Marker stability and exploratory biological associations

State-marker prominence was assessed without survival outcomes. Donor/state profiles were resampled with replacement in 500 bootstrap replicates. Within-state gene ranks were contrasted with the median rank across other states. Each replicate selected 40 positive markers per state from a prespecified candidate pool containing five times that number. Stable markers required selection frequency at least 0.70 and a positive 95% bootstrap effect interval. Clinical-adjusted Cox models then evaluated the prominent genes and 24 prespecified pathway scores one at a time. Effects were expressed per standard deviation, Benjamini-Hochberg correction was applied separately to the gene and pathway families, and coefficient direction was evaluated across the 50 locked outer training sets using ridge Cox models. These outcome associations were exploratory and were not used to modify the prediction models.
"""
    methods_text = _methods_text(counts)
    if q1_available:
        methods_text += _q1_extension_methods()
    (output_dir / "METHODS_DRAFT.md").write_text(methods_text, encoding="utf-8")
    results_text = _results_text(tables, biology_summary)
    if q1_available:
        results_text += _q1_extension_results(root)
    (output_dir / "RESULTS_DRAFT.md").write_text(
        results_text, encoding="utf-8"
    )
    (output_dir / "REPORTING_STANDARDS.md").write_text(
        """# Reporting standards

This package is aligned to the TRIPOD+AI reporting domains for studies that develop or evaluate clinical prediction models and uses PROBAST+AI as a risk-of-bias and applicability framework.

- TRIPOD+AI: https://www.bmj.com/content/385/bmj-2023-078378
- PROBAST+AI: https://www.bmj.com/content/388/bmj-2024-082505
- EQUATOR TRIPOD page: https://www.equator-network.org/reporting-guidelines/tripod-statement/

The generated checklist is an evidence map, not a claim of formal compliance. Independent risk-of-bias assessment, author disclosures, public archiving, and journal-specific formatting remain pending.
""",
        encoding="utf-8",
    )
    manifest_rows.extend(
        [
            {"artifact": "methods_draft", "csv": "", "markdown": "METHODS_DRAFT.md", "rows": ""},
            {"artifact": "results_draft", "csv": "", "markdown": "RESULTS_DRAFT.md", "rows": ""},
            {"artifact": "reporting_standards", "csv": "", "markdown": "REPORTING_STANDARDS.md", "rows": ""},
        ]
    )
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(output_dir / "manifest.csv", index=False)
    return manifest

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .figures import COLORS, _bold_ticks, _panel, _save, configure_manuscript_style


MODEL_LABELS = {
    "elastic_net_cox": "Elastic-net Cox",
    "random_survival_forest": "Random survival forest",
    "survivalpfn": "SurvivalPFN",
    "xgb_aft": "XGBoost-AFT",
    "survpfn": "SurvPFN (June 2026)",
}


def _box(ax: plt.Axes, x: float, y: float, width: float, height: float, text: str, color: str) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        facecolor=color,
        edgecolor=color,
        alpha=0.14,
        linewidth=1.4,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", color=COLORS["navy"], fontsize=9.2)


def _arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.5,
            color=COLORS["navy"],
        )
    )


def figure_s1_cohort_flow(root: Path, output: Path) -> None:
    cohort = json.loads((root / "cohort" / "summary.json").read_text(encoding="utf-8"))
    reference = json.loads((root / "reference" / "summary.json").read_text(encoding="utf-8"))
    audit = json.loads((root / "external_validation" / "audit" / "audit_report.json").read_text(encoding="utf-8"))
    external_metrics = json.loads(
        (root / "external_validation" / "clinical_survivalpfn" / "metrics.json").read_text(encoding="utf-8")
    )

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.2))
    titles = ["TCGA discovery cohort", "Single-cell reference", "Locked external cohort"]
    for label, title, ax in zip("ABC", titles, axes):
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 6)
        ax.axis("off")
        ax.set_title(title, fontsize=12)
        _panel(ax, label)

    _box(axes[0], 0.35, 4.65, 3.3, 0.85, "TCGA-ESCA primary-tumour RNA-seq\nand clinical metadata", COLORS["blue"])
    _arrow(axes[0], (2.0, 4.63), (2.0, 3.82))
    _box(axes[0], 0.35, 2.95, 3.3, 0.85, "One primary tumour per patient\nOS time > 0; eligible EAC or ESCC", COLORS["teal"])
    _arrow(axes[0], (2.0, 2.93), (2.0, 2.12))
    _box(
        axes[0],
        0.35,
        1.05,
        3.3,
        1.05,
        f"Locked discovery cohort\nn = {cohort['samples']} | deaths = {cohort['events']}\nEAC = {cohort['eac']} | ESCC = {cohort['escc']}",
        COLORS["purple"],
    )

    _box(axes[1], 0.15, 4.65, 1.7, 0.85, "GSE160269\ncell-level inputs", COLORS["blue"])
    _box(axes[1], 2.15, 4.65, 1.7, 0.85, "GSE154763\ncell-level inputs", COLORS["gold"])
    _arrow(axes[1], (1.0, 4.63), (1.75, 3.8))
    _arrow(axes[1], (3.0, 4.63), (2.25, 3.8))
    _box(axes[1], 0.35, 2.85, 3.3, 0.95, "Donor-aware state aggregation\nbefore construction of state anchors", COLORS["teal"])
    _arrow(axes[1], (2.0, 2.83), (2.0, 2.02))
    _box(
        axes[1],
        0.35,
        0.95,
        3.3,
        1.05,
        f"Reference resource\n{reference['cells_summarized']:,} cells | {reference['patients']} patients\n{reference['profiles']} donor-state profiles | 13 modeled states",
        COLORS["purple"],
    )

    overlap = int(audit["cohort_patient_overlap"]["GSE53624__GSE53625"])
    _box(axes[2], 0.35, 4.65, 3.3, 0.85, "GSE53625 SuperSeries\n179 tumour patients", COLORS["blue"])
    _arrow(axes[2], (2.0, 4.63), (2.0, 3.82))
    _box(axes[2], 0.35, 2.85, 3.3, 0.95, f"Remove {overlap} patients overlapping\nwith GSE53624", COLORS["red"])
    _arrow(axes[2], (2.0, 2.83), (2.0, 2.02))
    _box(
        axes[2],
        0.35,
        0.95,
        3.3,
        1.05,
        f"Locked non-overlapping test set\nn = {external_metrics['n_external']} | deaths = {external_metrics['events_external']}\nClinical validation only",
        COLORS["purple"],
    )

    fig.suptitle("Supplementary Figure S1 | Data-resource inclusion and independence audit", fontsize=14, y=1.02)
    fig.tight_layout(w_pad=1.5)
    _save(fig, output)


def figure_s2_reference_composition(root: Path, output: Path) -> None:
    data = pd.read_csv(root / "reference" / "metadata.csv")
    states = sorted(data["state"].unique())
    sources = sorted(data["source"].unique())
    source_colors = [COLORS["blue"], COLORS["gold"]]
    y = np.arange(len(states))
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 6.8))

    left = np.zeros(len(states))
    for source, color in zip(sources, source_colors):
        values = data[data["source"] == source].groupby("state")["patient"].nunique().reindex(states, fill_value=0)
        axes[0].barh(y, values, left=left, color=color, alpha=0.9, label=source)
        left += values.to_numpy()
    axes[0].set_yticks(y, states)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Source-specific donor counts")
    axes[0].set_title("Donor coverage by annotated state")
    axes[0].legend(frameon=False, fontsize=9)

    cells = data.groupby("state")["n_cells"].sum().reindex(states)
    axes[1].barh(y, cells, color=COLORS["teal"], alpha=0.88)
    axes[1].set_yticks(y, states)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Cells summarized")
    axes[1].set_title("Cell representation by state")
    axes[1].ticklabel_format(style="plain", axis="x")

    values = [data.loc[data["state"] == state, "n_cells"].to_numpy() for state in states]
    bp = axes[2].boxplot(values, vert=False, tick_labels=states, showfliers=False, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor(COLORS["purple"])
        patch.set_alpha(0.55)
    axes[2].set_xscale("log")
    axes[2].invert_yaxis()
    axes[2].set_xlabel("Cells per donor-state profile (log scale)")
    axes[2].set_title("Profile-size distribution")

    for label, ax in zip("ABC", axes):
        ax.grid(axis="x", alpha=0.2, linewidth=0.8)
        _panel(ax, label)
        _bold_ticks(ax)
    fig.suptitle("Supplementary Figure S2 | Composition of the donor-aware reference", fontsize=14, y=1.01)
    fig.tight_layout(w_pad=2.0)
    _save(fig, output)


def _grouped_boxplot(
    ax: plt.Axes,
    data: pd.DataFrame,
    metric: str,
    scenarios: list[str],
    backends: list[str],
    ylabel: str,
) -> None:
    positions: list[float] = []
    values: list[np.ndarray] = []
    colors: list[str] = []
    offsets = [-0.18, 0.18]
    palette = [COLORS["teal"], COLORS["red"]]
    for index, scenario in enumerate(scenarios):
        for offset, backend, color in zip(offsets, backends, palette):
            positions.append(index + offset)
            values.append(data.loc[(data["scenario"] == scenario) & (data["backend"] == backend), metric].dropna().to_numpy())
            colors.append(color)
    bp = ax.boxplot(values, positions=positions, widths=0.30, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    ax.set_xticks(range(len(scenarios)), ["Clean", "Platform shift", "Unknown component"])
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.2, linewidth=0.8)


def figure_s3_simulation_distributions(root: Path, output: Path) -> None:
    data = pd.read_csv(root / "technical_validation" / "mixture_results.csv")
    scenarios = ["clean", "platform_shift", "unknown_component"]
    backends = ["robust_nnls", "unbalanced_ot"]
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.2))
    _grouped_boxplot(axes[0, 0], data, "state_mae", scenarios, backends, "State-composition MAE")
    _grouped_boxplot(axes[0, 1], data, "jensen_shannon", scenarios, backends, "Jensen-Shannon divergence")
    _grouped_boxplot(axes[1, 0], data, "reconstruction_cosine", scenarios, backends, "Reconstruction cosine")
    axes[0, 0].scatter([], [], color=COLORS["teal"], label="Robust NNLS")
    axes[0, 0].scatter([], [], color=COLORS["red"], label="Unbalanced OT")
    axes[0, 0].legend(frameon=False, fontsize=9)

    unknown = data[data["scenario"] == "unknown_component"]
    for backend, color, label in zip(backends, [COLORS["teal"], COLORS["red"]], ["Robust NNLS", "Unbalanced OT"]):
        block = unknown[unknown["backend"] == backend]
        axes[1, 1].scatter(block["unknown_truth"], block["unknown_estimate"], s=15, alpha=0.25, color=color, label=label)
    bounds = [0, float(max(unknown["unknown_truth"].max(), unknown["unknown_estimate"].max())) * 1.04]
    axes[1, 1].plot(bounds, bounds, linestyle="--", color=COLORS["grey"], linewidth=1.2)
    axes[1, 1].set_xlim(bounds)
    axes[1, 1].set_ylim(bounds)
    axes[1, 1].set_xlabel("True unknown fraction")
    axes[1, 1].set_ylabel("Estimated unknown score")
    axes[1, 1].set_title("Unknown-component recovery")
    axes[1, 1].legend(frameon=False, fontsize=9)
    axes[1, 1].grid(alpha=0.2, linewidth=0.8)

    for label, ax in zip("ABCD", axes.ravel()):
        _panel(ax, label)
        _bold_ticks(ax)
    fig.suptitle("Supplementary Figure S3 | Full simulated-mixture error distributions (500 mixtures per setting)", fontsize=14, y=1.0)
    fig.tight_layout(h_pad=2.2, w_pad=2.0)
    _save(fig, output)


def _clinical_metrics(root: Path) -> pd.DataFrame:
    primary = pd.read_csv(root / "benchmark" / "fold_metrics.csv")
    primary = primary[primary["panel"] == "clinical"].copy()
    challenger_path = root / "challengers" / "survpfn" / "fold_metrics.csv"
    if challenger_path.exists():
        primary = pd.concat([primary, pd.read_csv(challenger_path)], ignore_index=True)
    return primary


def _single_metric_boxes(ax: plt.Axes, data: pd.DataFrame, metric: str, ylabel: str) -> None:
    models = ["elastic_net_cox", "random_survival_forest", "survivalpfn", "xgb_aft", "survpfn"]
    models = [model for model in models if model in set(data["model"])]
    values = [data.loc[data["model"] == model, metric].dropna().to_numpy() for model in models]
    bp = ax.boxplot(values, patch_artist=True, showfliers=False, tick_labels=[MODEL_LABELS[item] for item in models])
    palette = [COLORS["blue"], COLORS["teal"], COLORS["purple"], COLORS["gold"], COLORS["red"]]
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.62)
    ax.tick_params(axis="x", rotation=22)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.2, linewidth=0.8)


def figure_s4_fold_distributions(root: Path, output: Path) -> None:
    clinical = _clinical_metrics(root)
    all_metrics = pd.read_csv(root / "benchmark" / "fold_metrics.csv")
    fig, axes = plt.subplots(2, 2, figsize=(14.2, 9.5))
    _single_metric_boxes(axes[0, 0], clinical, "uno_c", "Uno C-index")
    axes[0, 0].set_title("Clinical-only discrimination across 50 folds")
    _single_metric_boxes(axes[0, 1], clinical, "integrated_brier", "Integrated Brier score")
    axes[0, 1].set_title("Clinical-only prediction error across 50 folds")

    base = all_metrics[(all_metrics["panel"] == "clinical") & (all_metrics["model"] == "survivalpfn")]
    recast = all_metrics[(all_metrics["panel"] == "clinical_plus_recast") & (all_metrics["model"] == "survivalpfn")]
    paired = base.merge(recast, on=["repeat", "fold"], suffixes=("_clinical", "_recast"), validate="one_to_one")
    deltas = [
        paired["uno_c_recast"] - paired["uno_c_clinical"],
        paired["integrated_brier_clinical"] - paired["integrated_brier_recast"],
    ]
    bp = axes[1, 0].boxplot(deltas, patch_artist=True, showfliers=False, tick_labels=["Delta Uno C", "Delta IBS improvement"])
    for patch, color in zip(bp["boxes"], [COLORS["purple"], COLORS["gold"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    axes[1, 0].axhline(0, color=COLORS["grey"], linestyle="--", linewidth=1.1)
    axes[1, 0].set_ylabel("Paired fold difference (positive favors ReCAST)")
    axes[1, 0].set_title("Incremental ReCAST effect for SurvivalPFN")
    axes[1, 0].grid(axis="y", alpha=0.2, linewidth=0.8)

    axes[1, 1].scatter(paired["uno_c_clinical"], paired["uno_c_recast"], color=COLORS["purple"], alpha=0.7, s=30)
    plotted = paired[["uno_c_clinical", "uno_c_recast"]].dropna()
    bounds = [float(plotted.min().min()) - 0.02, float(plotted.max().max()) + 0.02]
    axes[1, 1].plot(bounds, bounds, color=COLORS["grey"], linestyle="--", linewidth=1.1)
    axes[1, 1].set_xlim(bounds)
    axes[1, 1].set_ylim(bounds)
    axes[1, 1].set_xlabel("Clinical-only Uno C")
    axes[1, 1].set_ylabel("Clinical + ReCAST Uno C")
    axes[1, 1].set_title("Fold-level paired discrimination")
    axes[1, 1].grid(alpha=0.2, linewidth=0.8)

    for label, ax in zip("ABCD", axes.ravel()):
        _panel(ax, label)
        _bold_ticks(ax)
    fig.suptitle("Supplementary Figure S4 | Outer-fold performance and paired ReCAST differences", fontsize=14, y=1.0)
    fig.tight_layout(h_pad=2.4, w_pad=2.0)
    _save(fig, output)


def _time_metric_plot(ax: plt.Axes, data: pd.DataFrame, prefix: str, ylabel: str, title: str, ideal: float | None = None) -> None:
    horizons = [365, 1095, 1825]
    models = ["elastic_net_cox", "survivalpfn", "survpfn"]
    colors = [COLORS["blue"], COLORS["purple"], COLORS["red"]]
    for model, color in zip(models, colors):
        block = data[data["model"] == model]
        means, errors = [], []
        for horizon in horizons:
            values = block[f"{prefix}_t{horizon}"].dropna()
            means.append(values.mean())
            errors.append(values.std())
        ax.errorbar([1, 3, 5], means, yerr=errors, marker="o", linewidth=1.6, capsize=3, color=color, label=MODEL_LABELS[model])
    if ideal is not None:
        ax.axhline(ideal, color=COLORS["grey"], linestyle="--", linewidth=1.0)
    ax.set_xticks([1, 3, 5], ["1 year", "3 years", "5 years"])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.2, linewidth=0.8)


def figure_s5_time_metrics_calibration(root: Path, output: Path) -> None:
    data = _clinical_metrics(root)
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.0))
    _time_metric_plot(axes[0, 0], data, "auc", "Dynamic AUC (mean +/- SD)", "Time-specific discrimination")
    _time_metric_plot(axes[0, 1], data, "brier", "IPCW Brier score (mean +/- SD)", "Time-specific prediction error")
    _time_metric_plot(axes[1, 0], data, "calibration_slope", "Calibration slope (mean +/- SD)", "Calibration slope", ideal=1.0)
    _time_metric_plot(axes[1, 1], data, "calibration_intercept", "Calibration intercept (mean +/- SD)", "Calibration intercept", ideal=0.0)
    axes[0, 0].legend(frameon=False, fontsize=9)
    for label, ax in zip("ABCD", axes.ravel()):
        _panel(ax, label)
        _bold_ticks(ax)
    fig.suptitle("Supplementary Figure S5 | Time-specific performance and calibration in outer test folds", fontsize=14, y=1.0)
    fig.text(0.5, 0.005, "Error bars use valid folds only; calibration estimates are unavailable in folds lacking sufficient horizon events.", ha="center", fontsize=9)
    fig.tight_layout(h_pad=2.2, w_pad=2.0, rect=(0, 0.03, 1, 1))
    _save(fig, output)


def _volcano(ax: plt.Axes, data: pd.DataFrame, label_column: str, title: str, color: str) -> None:
    clean = data.dropna(subset=["log_hazard_ratio_per_sd", "p_value"]).copy()
    clean["minus_log10_p"] = -np.log10(clean["p_value"].clip(lower=np.finfo(float).tiny))
    ax.scatter(clean["log_hazard_ratio_per_sd"], clean["minus_log10_p"], color=color, alpha=0.62, s=28)
    ax.axvline(0, color=COLORS["grey"], linestyle="--", linewidth=1.0)
    ax.axhline(-np.log10(0.05), color=COLORS["grey"], linestyle=":", linewidth=1.0)
    for _, row in clean.nsmallest(5, "p_value").iterrows():
        ax.annotate(str(row[label_column]), (row["log_hazard_ratio_per_sd"], row["minus_log10_p"]), xytext=(3, 4), textcoords="offset points", fontsize=7.7)
    ax.set_xlabel("Log hazard ratio per 1 SD")
    ax.set_ylabel("-log10(nominal P)")
    ax.set_title(title)
    ax.grid(alpha=0.15, linewidth=0.7)
    ax.text(0.98, 0.96, "No BH FDR < 0.05", transform=ax.transAxes, ha="right", va="top", color=COLORS["red"], fontsize=9)


def figure_s6_biology_full(root: Path, output: Path) -> None:
    markers = pd.read_csv(root / "biology" / "marker_stability_all.csv")
    genes = pd.read_csv(root / "biology" / "exploratory_gene_survival.csv")
    pathways = pd.read_csv(root / "biology" / "exploratory_pathway_survival.csv")
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.0))

    candidate = markers[markers["bootstrap_selection_frequency"] > 0]
    axes[0, 0].hist(candidate["bootstrap_selection_frequency"], bins=np.linspace(0, 1, 21), color=COLORS["teal"], alpha=0.82)
    axes[0, 0].axvline(0.70, color=COLORS["red"], linestyle="--", linewidth=1.3, label="Stability threshold = 0.70")
    axes[0, 0].set_xlabel("Donor-bootstrap selection frequency")
    axes[0, 0].set_ylabel("State-gene pairs")
    axes[0, 0].set_title("Selection-frequency distribution")
    axes[0, 0].legend(frameon=False, fontsize=9)

    counts = markers.groupby("state")["stable_marker"].sum().sort_values()
    axes[0, 1].barh(np.arange(len(counts)), counts.to_numpy(), color=COLORS["purple"], alpha=0.82)
    axes[0, 1].set_yticks(np.arange(len(counts)), counts.index)
    axes[0, 1].set_xlabel("Stable state-gene pairs")
    axes[0, 1].set_title("Stable markers across all modeled states")
    axes[0, 1].grid(axis="x", alpha=0.2, linewidth=0.8)

    _volcano(axes[1, 0], genes, "gene", "All 92 prominent-gene associations", COLORS["blue"])
    pathways = pathways.copy()
    pathways["short_label"] = pathways["feature"].str.replace("pathway__REACTOME_", "", regex=False).str.replace("_", " ", regex=False).str.title()
    _volcano(axes[1, 1], pathways, "short_label", "All 24 pathway associations", COLORS["gold"])

    for label, ax in zip("ABCD", axes.ravel()):
        _panel(ax, label)
        _bold_ticks(ax)
    fig.suptitle("Supplementary Figure S6 | Complete marker stability and exploratory association analysis", fontsize=14, y=1.0)
    fig.text(0.5, 0.006, "Associations are internal and exploratory. The Reactome Dengue Virus Infection label denotes a curated gene set, not evidence of dengue infection.", ha="center", fontsize=9)
    fig.tight_layout(h_pad=2.5, w_pad=2.0, rect=(0, 0.035, 1, 1))
    _save(fig, output)


def figure_s7_external_audit(root: Path, output: Path) -> None:
    audit_root = root / "external_validation" / "audit"
    cohorts = pd.read_csv(audit_root / "cohort_audit.csv")
    report = json.loads((audit_root / "audit_report.json").read_text(encoding="utf-8"))
    mapping = report["platform_mapping"]
    overlap = int(report["cohort_patient_overlap"]["GSE53624__GSE53625"])
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 9.2))

    x = np.arange(len(cohorts))
    axes[0, 0].bar(x, cohorts["tumor_patients"], color=[COLORS["blue"], COLORS["purple"]], alpha=0.85)
    axes[0, 0].bar(x, cohorts["events"], color=COLORS["red"], alpha=0.75, label="Deaths")
    axes[0, 0].set_xticks(x, cohorts["cohort"])
    axes[0, 0].set_ylabel("Patients")
    axes[0, 0].set_title("External clinical cohorts")
    axes[0, 0].legend(frameon=False, fontsize=9)
    axes[0, 0].text(0.5, 0.95, f"Patient overlap = {overlap}; locked non-overlap = 60", transform=axes[0, 0].transAxes, ha="center", va="top", fontsize=9)

    labels = ["Platform\nfeatures", "With\nsequence", "Unique\nmatches", "Direct TCGA\ngene overlap"]
    values = [mapping["feature_rows"], mapping["feature_rows_with_sequence"], mapping["feature_rows_with_unique_match"], mapping["direct_tcga_gene_overlap"]]
    colors = [COLORS["blue"], COLORS["teal"], COLORS["gold"], COLORS["red"]]
    axes[0, 1].bar(np.arange(4), values, color=colors, alpha=0.85)
    axes[0, 1].set_xticks(np.arange(4), labels)
    axes[0, 1].set_ylabel("Rows")
    axes[0, 1].set_title("Sequence crosswalk audit")
    for idx, value in enumerate(values):
        axes[0, 1].text(idx, value + 1800, f"{value:,}", ha="center", fontsize=8.5)

    width = 0.34
    axes[1, 0].bar(x - width / 2, cohorts["probe_rows"], width, color=COLORS["blue"], label="Probe rows")
    axes[1, 0].bar(x + width / 2, cohorts["gene_rows"], width, color=COLORS["red"], label="Supplied gene rows")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xticks(x, cohorts["cohort"])
    axes[1, 0].set_ylabel("Rows (log scale)")
    axes[1, 0].set_title("Probe-level versus supplied gene matrices")
    axes[1, 0].legend(frameon=False, fontsize=9)

    ax = axes[1, 1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    _box(ax, 0.5, 3.8, 4.0, 1.25, "Clinical endpoint and covariates\nREADY for locked validation", COLORS["teal"])
    _box(ax, 5.5, 3.8, 4.0, 1.25, "Gene-level external replication\nBLOCKED by untraceable annotation", COLORS["red"])
    _box(ax, 2.4, 1.05, 5.2, 1.35, "Claim boundary\nExternal clinical validation is reported;\nexternal omics confirmation is not claimed.", COLORS["purple"])
    _arrow(ax, (2.5, 3.75), (4.3, 2.42))
    _arrow(ax, (7.5, 3.75), (5.7, 2.42))
    ax.set_title("Pre-specified external-validation decision")

    for label, ax in zip("ABCD", axes.ravel()):
        _panel(ax, label)
        _bold_ticks(ax)
    fig.suptitle("Supplementary Figure S7 | External-cohort overlap and platform-annotation audit", fontsize=14, y=1.0)
    fig.tight_layout(h_pad=2.4, w_pad=2.0)
    _save(fig, output)


def make_supplementary_figures(root: str | Path, output_dir: str | Path) -> pd.DataFrame:
    configure_manuscript_style()
    root = Path(root)
    output_dir = Path(output_dir)
    definitions: list[tuple[str, Any, str]] = [
        ("figureS1_cohort_flow.svg", figure_s1_cohort_flow, "cohort/reference summaries and external overlap audit"),
        ("figureS2_reference_composition.svg", figure_s2_reference_composition, "reference/metadata.csv"),
        ("figureS3_simulation_distributions.svg", figure_s3_simulation_distributions, "technical_validation/mixture_results.csv"),
        ("figureS4_fold_distributions.svg", figure_s4_fold_distributions, "benchmark and SurvPFN challenger fold metrics"),
        ("figureS5_time_metrics_calibration.svg", figure_s5_time_metrics_calibration, "clinical outer-fold time-specific metrics"),
        ("figureS6_biology_full.svg", figure_s6_biology_full, "complete marker stability and exploratory association tables"),
        ("figureS7_external_audit.svg", figure_s7_external_audit, "external cohort overlap and platform annotation audit"),
    ]
    rows = []
    for filename, function, sources in definitions:
        path = output_dir / filename
        function(root, path)
        rows.append(
            {
                "figure": filename,
                "path": str(path),
                "source_artifacts": sources,
                "format": "editable_svg",
                "font_family": "Times New Roman",
                "font_weight": "bold",
            }
        )
    manifest = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_dir / "supplementary_figure_manifest.csv", index=False)
    return manifest

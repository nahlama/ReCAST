from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from .figures import COLORS, _bold_ticks, _panel, _save, configure_manuscript_style
from .q1_extension import METHOD_LABELS


MODEL_LABELS = {
    "elastic_net_cox": "Elastic-net Cox",
    "random_survival_forest": "Random survival forest",
    "survivalpfn": "SurvivalPFN",
    "survpfn": "SurvPFN (June 2026)",
    "xgb_aft": "XGBoost-AFT",
}


def figure_s8_heldout_projection(root: Path, output: Path) -> None:
    directory = root / "q1_extension" / "projection_validation"
    metrics = pd.read_csv(directory / "mixture_metrics.csv")
    paired = pd.read_csv(directory / "paired_comparisons.csv")
    composition = pd.read_csv(directory / "composition_long.csv")
    methods = list(METHOD_LABELS)
    short = ["Robust NNLS", "Ordinary NNLS", "Ridge NNLS", "Simplex NNLS", "nu-SVR", "Unbalanced OT"]
    colors = [COLORS["purple"], COLORS["blue"], COLORS["gold"], COLORS["teal"], COLORS["navy"], COLORS["red"]]
    fig, axes = plt.subplots(2, 2, figsize=(14.2, 9.6))
    for ax, scenario, title in zip(
        axes[0],
        ["heldout_donor_clean", "heldout_donor_platform_shift"],
        ["Clean held-out donor mixtures", "Held-out mixtures with platform shift"],
    ):
        values = [metrics.loc[(metrics["scenario"] == scenario) & (metrics["method"] == method), "state_mae"] for method in methods]
        bp = ax.boxplot(values, patch_artist=True, showfliers=False, tick_labels=short)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.62)
        ax.tick_params(axis="x", rotation=25)
        for label in ax.get_xticklabels():
            label.set_ha("right")
        ax.set_ylabel("State-composition MAE")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)

    clean = paired[paired["scenario"] == "heldout_donor_clean"].set_index("comparator").reindex(methods[1:])
    y = np.arange(len(clean))
    effect = clean["mean_mae_improvement_primary"].to_numpy()
    axes[1, 0].errorbar(
        effect,
        y,
        xerr=[effect - clean["ci95_low"].to_numpy(), clean["ci95_high"].to_numpy() - effect],
        fmt="o",
        color=COLORS["purple"],
        capsize=3,
    )
    axes[1, 0].axvline(0, color=COLORS["grey"], linestyle="--")
    axes[1, 0].set_yticks(y, short[1:])
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_xlabel("Comparator MAE - ReCAST MAE\n(positive favors ReCAST)")
    axes[1, 0].set_title("Donor-clustered paired differences (95% CI)")
    axes[1, 0].grid(axis="x", alpha=0.2)

    block = composition[
        (composition["scenario"] == "heldout_donor_clean")
        & (composition["method"].isin(["robust_nnls_hpa", "simplex_nnls"]))
    ]
    for method, color, label in [
        ("robust_nnls_hpa", COLORS["purple"], "ReCAST robust NNLS"),
        ("simplex_nnls", COLORS["teal"], "Simplex NNLS"),
    ]:
        item = block[block["method"] == method]
        axes[1, 1].scatter(item["truth"], item["estimate"], s=9, alpha=0.16, color=color, label=label)
    axes[1, 1].plot([0, 1], [0, 1], color=COLORS["grey"], linestyle="--")
    axes[1, 1].set_xlim(-0.02, 1.02)
    axes[1, 1].set_ylim(-0.02, 1.02)
    axes[1, 1].set_xlabel("True state fraction")
    axes[1, 1].set_ylabel("Estimated state fraction")
    axes[1, 1].set_title("All held-out donor-state estimates")
    axes[1, 1].legend(frameon=False, fontsize=9)
    axes[1, 1].grid(alpha=0.2)

    for label, ax in zip("ABCD", axes.ravel()):
        _panel(ax, label)
        _bold_ticks(ax)
    fig.suptitle("Supplementary Figure S8 | Donor-held-out pseudo-bulk validation with matched comparators", fontsize=14, y=1.0)
    fig.text(0.5, 0.005, "Sixty-seven donors were excluded from anchor construction one at a time; five mixtures per donor and scenario were evaluated.", ha="center", fontsize=9)
    fig.tight_layout(h_pad=2.5, w_pad=2.0, rect=(0, 0.035, 1, 1))
    _save(fig, output)


def _volcano(ax: plt.Axes, data: pd.DataFrame, title: str, color: str) -> None:
    clean = data.dropna(subset=["log_hazard_ratio_per_sd", "p_value"]).copy()
    clean["minus_log10_p"] = -np.log10(clean["p_value"].clip(lower=1e-300))
    ax.scatter(clean["log_hazard_ratio_per_sd"], clean["minus_log10_p"], color=color, alpha=0.65, s=28)
    ax.axvline(0, color=COLORS["grey"], linestyle="--")
    ax.axhline(-np.log10(0.05), color=COLORS["grey"], linestyle=":")
    label_count = 3 if clean["feature"].astype(str).str.startswith("pathway__").any() else 5
    for _, row in clean.nsmallest(label_count, "p_value").iterrows():
        raw_label = str(row["feature"])
        label = (
            raw_label.replace("pathway__REACTOME_", "").replace("_", " ").title()
            if raw_label.startswith("pathway__")
            else raw_label
        )
        ax.annotate(label, (row["log_hazard_ratio_per_sd"], row["minus_log10_p"]), xytext=(3, 4), textcoords="offset points", fontsize=7.5)
    ax.set_xlabel("External log hazard ratio per 1 SD")
    ax.set_ylabel("-log10(nominal P)")
    ax.set_title(title)
    ax.text(0.98, 0.95, "No BH FDR < 0.05", transform=ax.transAxes, ha="right", va="top", color=COLORS["red"], fontsize=9)
    ax.grid(alpha=0.16)


def figure_s9_external_reannotation(root: Path, output: Path) -> None:
    directory = root / "q1_extension" / "external_reannotation"
    mapping = pd.read_csv(directory / "probe_to_gencode50.csv.gz")
    genes = pd.read_csv(directory / "external_prominent_gene_confirmation.csv")
    pathways = pd.read_csv(directory / "external_pathway_confirmation.csv")
    fig, axes = plt.subplots(2, 2, figsize=(14.2, 9.6))

    counts = mapping["mapping_status"].value_counts().reindex(
        ["unique_gene_exact_match", "ambiguous_gene_exact_match", "no_exact_transcript_match", "duplicated_feature_sequence"]
    )
    axes[0, 0].bar(np.arange(4), counts, color=[COLORS["teal"], COLORS["gold"], COLORS["grey"], COLORS["red"]], alpha=0.85)
    axes[0, 0].set_xticks(np.arange(4), ["Unique gene", "Ambiguous gene", "No exact match", "Duplicated sequence"], rotation=18, ha="right")
    axes[0, 0].set_ylabel("Probe features")
    axes[0, 0].set_title("GENCODE 50 full-length exact mapping")
    for index, value in enumerate(counts):
        axes[0, 0].text(index, value + 650, f"{int(value):,}", ha="center", fontsize=8.5)

    valid = genes.dropna(subset=["internal_log_hr", "log_hazard_ratio_per_sd"])
    concordant = valid["direction_concordant"].astype(bool)
    axes[0, 1].scatter(valid.loc[~concordant, "internal_log_hr"], valid.loc[~concordant, "log_hazard_ratio_per_sd"], color=COLORS["red"], alpha=0.65, label="Discordant")
    axes[0, 1].scatter(valid.loc[concordant, "internal_log_hr"], valid.loc[concordant, "log_hazard_ratio_per_sd"], color=COLORS["teal"], alpha=0.65, label="Concordant")
    axes[0, 1].axhline(0, color=COLORS["grey"], linestyle="--")
    axes[0, 1].axvline(0, color=COLORS["grey"], linestyle="--")
    axes[0, 1].set_xlabel("Internal TCGA log hazard ratio")
    axes[0, 1].set_ylabel("External GSE53625 log hazard ratio")
    axes[0, 1].set_title("Direction of 75 testable prominent genes")
    axes[0, 1].legend(frameon=False, fontsize=9)
    axes[0, 1].grid(alpha=0.16)

    _volcano(axes[1, 0], genes, "External prominent-gene confirmation", COLORS["blue"])
    _volcano(axes[1, 1], pathways, "External prespecified-pathway confirmation", COLORS["gold"])
    for label, ax in zip("ABCD", axes.ravel()):
        _panel(ax, label)
        _bold_ticks(ax)
    fig.suptitle("Supplementary Figure S9 | Versioned external molecular reannotation and confirmation", fontsize=14, y=1.0)
    fig.text(0.5, 0.005, "Strict mapping used GENCODE 50 (GRCh38.p14); external ridge-Cox associations were prespecified and not used for selection.", ha="center", fontsize=9)
    fig.tight_layout(h_pad=2.5, w_pad=2.0, rect=(0, 0.035, 1, 1))
    _save(fig, output)


def _dca_panel(ax: plt.Axes, data: pd.DataFrame, horizon: int, title: str) -> None:
    colors = [COLORS["blue"], COLORS["purple"], COLORS["red"]]
    for model, color in zip(["elastic_net_cox", "survivalpfn", "survpfn"], colors):
        block = data[(data["model"] == model) & (data["horizon_days"] == horizon)]
        if "repeat" in block and block["repeat"].notna().any():
            block = block.groupby("threshold_probability", as_index=False)["net_benefit"].mean()
        ax.plot(block["threshold_probability"], block["net_benefit"], color=color, linewidth=1.8, label=MODEL_LABELS[model])
    reference = data[data["horizon_days"] == horizon]
    if "repeat" in reference and reference["repeat"].notna().any():
        reference = reference.groupby("threshold_probability", as_index=False)["treat_all_net_benefit"].mean()
    else:
        reference = reference.drop_duplicates("threshold_probability")
    ax.plot(reference["threshold_probability"], reference["treat_all_net_benefit"], color=COLORS["grey"], linestyle="--", label="Treat all")
    ax.axhline(0, color="black", linewidth=1.0, label="Treat none")
    ax.set_xlabel("Risk threshold")
    ax.set_ylabel("IPCW net benefit")
    ax.set_title(title)
    ax.grid(alpha=0.18)


def figure_s10_decision_curves(root: Path, output: Path) -> None:
    directory = root / "q1_extension" / "clinical_utility"
    internal = pd.read_csv(directory / "decision_curve_internal.csv")
    external = pd.read_csv(directory / "decision_curve_external.csv")
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.2))
    _dca_panel(axes[0, 0], internal, 365, "Internal outer-test decision curves | 1 year")
    _dca_panel(axes[0, 1], internal, 1095, "Internal outer-test decision curves | 3 years")
    _dca_panel(axes[1, 0], external, 365, "Locked external decision curves | 1 year")
    _dca_panel(axes[1, 1], external, 1095, "Locked external decision curves | 3 years")
    axes[0, 0].legend(frameon=False, fontsize=8.5)
    for label, ax in zip("ABCD", axes.ravel()):
        _panel(ax, label)
        _bold_ticks(ax)
    fig.suptitle("Supplementary Figure S10 | Exploratory IPCW decision-curve analysis", fontsize=14, y=1.0)
    fig.text(0.5, 0.005, "External predictions were not recalibrated. Clinical utility remains exploratory because thresholds were not prospectively selected.", ha="center", fontsize=9)
    fig.tight_layout(h_pad=2.3, w_pad=2.0, rect=(0, 0.035, 1, 1))
    _save(fig, output)


def figure_s11_subgroups_calibration(root: Path, output: Path) -> None:
    directory = root / "q1_extension" / "clinical_utility"
    subgroup = pd.read_csv(directory / "histology_subgroup_performance.csv")
    interaction = pd.read_csv(directory / "histology_interactions.csv")
    calibration = pd.read_csv(directory / "external_risk_group_calibration.csv")
    models = ["elastic_net_cox", "random_survival_forest", "survivalpfn", "xgb_aft", "survpfn"]
    fig, axes = plt.subplots(2, 2, figsize=(14.2, 9.5))
    x = np.arange(len(models))
    width = 0.34
    for offset, histology, color in [(-width / 2, "EAC", COLORS["blue"]), (width / 2, "ESCC", COLORS["gold"])]:
        values = subgroup[subgroup["histology"] == histology].set_index("model").reindex(models)["uno_c"]
        axes[0, 0].bar(x + offset, values, width, color=color, alpha=0.85, label=histology)
    axes[0, 0].axhline(0.5, color=COLORS["grey"], linestyle="--")
    axes[0, 0].set_xticks(x, [MODEL_LABELS[item] for item in models], rotation=22, ha="right")
    axes[0, 0].set_ylabel("Descriptive Uno C")
    axes[0, 0].set_title("Out-of-fold discrimination by histology")
    axes[0, 0].legend(frameon=False)

    block = interaction.set_index("model").reindex(models)
    y = np.arange(len(models))
    hr = block["interaction_hr"].to_numpy()
    axes[0, 1].errorbar(hr, y, xerr=[hr - block["ci95_low"].to_numpy(), block["ci95_high"].to_numpy() - hr], fmt="o", color=COLORS["purple"], capsize=3)
    axes[0, 1].axvline(1, color=COLORS["grey"], linestyle="--")
    axes[0, 1].set_yticks(y, [MODEL_LABELS[item] for item in models])
    axes[0, 1].invert_yaxis()
    axes[0, 1].set_xlabel("Risk x ESCC interaction HR (95% CI)")
    axes[0, 1].set_title("Formal histology interaction tests")
    axes[0, 1].grid(axis="x", alpha=0.2)

    for ax, horizon, title in zip(axes[1], [365, 1095], ["External risk groups | 1 year", "External risk groups | 3 years"]):
        groups = ["low", "intermediate", "high"]
        for model, color, marker in zip(["elastic_net_cox", "survivalpfn", "survpfn"], [COLORS["blue"], COLORS["purple"], COLORS["red"]], ["o", "s", "^"]):
            item = calibration[(calibration["model"] == model) & (calibration["horizon_days"] == horizon)].set_index("risk_group").reindex(groups)
            ax.plot(item["mean_predicted_risk"], item["km_observed_risk"], marker=marker, color=color, label=MODEL_LABELS[model])
        ax.plot([0, 1], [0, 1], color=COLORS["grey"], linestyle="--")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted risk")
        ax.set_ylabel("Kaplan-Meier observed risk")
        ax.set_title(title)
        ax.grid(alpha=0.2)
    axes[1, 0].legend(frameon=False, fontsize=8.5)
    for label, ax in zip("ABCD", axes.ravel()):
        _panel(ax, label)
        _bold_ticks(ax)
    fig.suptitle("Supplementary Figure S11 | Histology subgroups and transferred external risk groups", fontsize=14, y=1.0)
    fig.text(0.5, 0.005, "Subgroup analyses are exploratory; external risk-group cutpoints were defined exclusively from TCGA out-of-fold predictions.", ha="center", fontsize=9)
    fig.tight_layout(h_pad=2.5, w_pad=2.0, rect=(0, 0.035, 1, 1))
    _save(fig, output)


def figure_s12_probast(root: Path, output: Path) -> None:
    data = pd.read_csv(root / "q1_extension" / "probast_ai" / "probast_ai_domain_self_assessment.csv")
    palette = {"Low": COLORS["teal"], "Some concerns": COLORS["gold"], "High": COLORS["red"]}
    fig, ax = plt.subplots(figsize=(13.8, 7.4))
    y = np.arange(len(data))
    ax.barh(y, np.ones(len(data)), color=[palette[item] for item in data["author_judgment"]], alpha=0.82)
    ax.set_yticks(y, data["domain"])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    for index, row in data.iterrows():
        ax.text(0.02, index, row["author_judgment"], va="center", color="white" if row["author_judgment"] != "Some concerns" else COLORS["navy"], fontsize=9.5)
    handles = [Patch(facecolor=color, edgecolor=color, label=judgment) for judgment, color in palette.items()]
    ax.legend(handles=handles, frameon=False, ncol=3, loc="lower right", bbox_to_anchor=(1, 1.01))
    ax.set_title("Domain-level author assessment; independent reviewer assessment remains required")
    _panel(ax, "A")
    _bold_ticks(ax)
    fig.suptitle("Supplementary Figure S12 | PROBAST+AI-aligned risk-of-bias and applicability evidence map", fontsize=14, y=1.0)
    fig.text(0.5, 0.012, "This evidence map is not certification and does not reproduce or replace the official PROBAST+AI assessment tool.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, output)


def make_q1_extension_figures(root: str | Path, output_dir: str | Path) -> pd.DataFrame:
    configure_manuscript_style()
    root = Path(root)
    output_dir = Path(output_dir)
    definitions: list[tuple[str, Any, str]] = [
        ("figureS8_heldout_projection.svg", figure_s8_heldout_projection, "q1_extension/projection_validation"),
        ("figureS9_external_reannotation.svg", figure_s9_external_reannotation, "q1_extension/external_reannotation"),
        ("figureS10_decision_curves.svg", figure_s10_decision_curves, "q1_extension/clinical_utility decision curves"),
        ("figureS11_subgroups_calibration.svg", figure_s11_subgroups_calibration, "q1_extension/clinical_utility subgroup and calibration tables"),
        ("figureS12_probast.svg", figure_s12_probast, "q1_extension/probast_ai domain assessment"),
    ]
    rows = []
    for filename, function, source in definitions:
        path = output_dir / filename
        function(root, path)
        rows.append({"figure": filename, "path": str(path), "source_artifacts": source, "format": "editable_svg", "font_family": "Times New Roman", "font_weight": "bold"})
    manifest = pd.DataFrame(rows)
    manifest.to_csv(output_dir / "q1_extension_figure_manifest.csv", index=False)
    return manifest

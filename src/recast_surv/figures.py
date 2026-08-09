from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


COLORS = {
    "navy": "#143D59",
    "blue": "#2F6B9A",
    "teal": "#2A9D8F",
    "gold": "#E9A23B",
    "red": "#C84C4C",
    "purple": "#7A5AA6",
    "grey": "#6B7280",
    "light": "#E9EEF3",
}


def configure_manuscript_style() -> None:
    """Editable SVG typography: all text remains bold Times New Roman."""

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "figure.titleweight": "bold",
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.1,
            "xtick.major.width": 1.1,
            "ytick.major.width": 1.1,
        }
    )


def _bold_ticks(ax: plt.Axes) -> None:
    for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        label.set_fontweight("bold")
        label.set_fontfamily("Times New Roman")


def _panel(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.11,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        fontfamily="Times New Roman",
        va="top",
    )


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        format="svg",
        bbox_inches="tight",
        metadata={"Creator": "ReCAST-Surv reproducible figure pipeline"},
    )
    plt.close(fig)


def figure_workflow(root: Path, output: Path) -> None:
    cohort = json.loads((root / "cohort" / "summary.json").read_text(encoding="utf-8"))
    reference = json.loads((root / "reference" / "summary.json").read_text(encoding="utf-8"))
    diagnostics = json.loads((root / "features" / "diagnostics.json").read_text(encoding="utf-8"))

    fig, ax = plt.subplots(figsize=(13.0, 4.6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5)
    ax.axis("off")
    boxes = [
        (
            0.2,
            COLORS["blue"],
            "Patient cohort",
            f"TCGA-ESCA\n{cohort['samples']} tumours | {cohort['events']} deaths\n{cohort['genes']:,} genes",
        ),
        (
            3.45,
            COLORS["teal"],
            "Donor-aware reference",
            f"{reference['cells_summarized']:,} cells | {reference['patients']} patients\n"
            f"{reference['profiles']} profiles | {reference['states']} annotated states",
        ),
        (
            6.7,
            COLORS["gold"],
            "Robust projection",
            f"HPA-weighted NNLS\n{diagnostics['recast_features']} compact ReCAST + "
            f"{diagnostics['pathway_features']} pathway features\nUnknown-expression score",
        ),
        (
            9.95,
            COLORS["purple"],
            "Leakage-safe survival",
            "SurvivalPFN + Cox comparison\nNested patient-level validation\nLocked external overlap audit",
        ),
    ]
    for index, (x, color, heading, body) in enumerate(boxes):
        patch = FancyBboxPatch(
            (x, 1.25),
            2.75,
            2.45,
            boxstyle="round,pad=0.06,rounding_size=0.12",
            facecolor=color,
            edgecolor=color,
            linewidth=1.5,
            alpha=0.14,
        )
        ax.add_patch(patch)
        ax.text(x + 1.375, 3.18, heading, ha="center", va="center", fontsize=12, color=color)
        ax.text(x + 1.375, 2.25, body, ha="center", va="center", fontsize=10, color=COLORS["navy"])
        ax.text(x + 0.13, 3.52, chr(ord("A") + index), fontsize=13, color=color)
        if index < len(boxes) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (x + 2.82, 2.47),
                    (boxes[index + 1][0] - 0.08, 2.47),
                    arrowstyle="-|>",
                    mutation_scale=15,
                    linewidth=1.7,
                    color=COLORS["navy"],
                )
            )
    ax.text(
        6.5,
        0.55,
        "Reference construction is outcome-free; all patient-derived preprocessing is fitted inside training folds.",
        ha="center",
        fontsize=10.5,
        color=COLORS["navy"],
    )
    _save(fig, output)


def figure_technical_validation(root: Path, output: Path) -> None:
    data = pd.read_csv(root / "technical_validation" / "summary.csv")
    scenarios = ["clean", "platform_shift", "unknown_component"]
    labels = ["Clean", "Platform shift", "Unknown component"]
    backends = ["robust_nnls", "unbalanced_ot"]
    backend_labels = ["Robust NNLS", "Unbalanced OT"]
    colors = [COLORS["teal"], COLORS["red"]]
    x = np.arange(len(scenarios))
    width = 0.34
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3))
    for backend_index, (backend, label, color) in enumerate(zip(backends, backend_labels, colors)):
        block = data.set_index(["scenario", "backend"]).loc[[(item, backend) for item in scenarios]]
        offset = (backend_index - 0.5) * width
        axes[0].bar(
            x + offset,
            block["state_mae_mean"],
            width,
            yerr=block["state_mae_std"],
            label=label,
            color=color,
            alpha=0.88,
            capsize=3,
        )
        axes[1].bar(
            x + offset,
            block["jensen_shannon_mean"],
            width,
            label=label,
            color=color,
            alpha=0.88,
        )
    unknown = data.loc[data["scenario"].eq("unknown_component")].set_index("backend")
    axes[2].bar(
        np.arange(2),
        [unknown.loc[item, "unknown_spearman"] for item in backends],
        color=colors,
        width=0.62,
        alpha=0.88,
    )
    axes[0].set_ylabel("State proportion MAE")
    axes[1].set_ylabel("Jensen-Shannon divergence")
    axes[2].set_ylabel("Unknown-score Spearman rho")
    axes[0].set_title("Mixture recovery")
    axes[1].set_title("Compositional divergence")
    axes[2].set_title("Unknown-component detection")
    for ax in axes[:2]:
        ax.set_xticks(x, labels, rotation=18, ha="right")
        ax.grid(axis="y", alpha=0.2, linewidth=0.8)
    axes[2].set_xticks(np.arange(2), backend_labels, rotation=18, ha="right")
    axes[2].set_ylim(0.75, 1.0)
    axes[2].grid(axis="y", alpha=0.2, linewidth=0.8)
    axes[0].legend(frameon=False, prop={"family": "Times New Roman", "weight": "bold"})
    for label, ax in zip("ABC", axes):
        _panel(ax, label)
        _bold_ticks(ax)
    fig.tight_layout(w_pad=2.2)
    _save(fig, output)


def figure_internal_benchmark(root: Path, output: Path) -> None:
    data = pd.read_csv(root / "benchmark" / "summary.csv")
    challenger = root / "challengers" / "survpfn" / "summary.csv"
    if challenger.exists():
        data = pd.concat([data, pd.read_csv(challenger)], ignore_index=True)
    panels = ["clinical", "clinical_plus_recast", "clinical_plus_pathway", "full"]
    panel_labels = ["Clinical", "Clinical + ReCAST", "Clinical + pathway", "Full"]
    models = ["survivalpfn", "survpfn", "elastic_net_cox", "xgb_aft", "random_survival_forest"]
    model_labels = ["SurvivalPFN", "SurvPFN (June 2026)", "Elastic-net Cox", "XGB-AFT", "RSF"]
    colors = [COLORS["purple"], COLORS["red"], COLORS["blue"], COLORS["gold"], COLORS["teal"]]
    markers = ["o", "P", "s", "^", "D"]
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8))
    x = np.arange(len(panels))
    for model_index, (model, label, color, marker) in enumerate(
        zip(models, model_labels, colors, markers)
    ):
        block = data.loc[data["model"].eq(model)].set_index("panel").reindex(panels)
        offset = (model_index - 2.0) * 0.07
        axes[0].errorbar(
            x + offset,
            block["uno_c_mean"],
            yerr=block["uno_c_std"],
            fmt=marker,
            color=color,
            capsize=3,
            markersize=6,
            linewidth=1.3,
            label=label,
        )
        axes[1].errorbar(
            x + offset,
            block["integrated_brier_mean"],
            yerr=block["integrated_brier_std"],
            fmt=marker,
            color=color,
            capsize=3,
            markersize=6,
            linewidth=1.3,
            label=label,
        )
    axes[0].axhline(0.5, color=COLORS["grey"], linewidth=1.0, linestyle="--")
    axes[0].set_ylabel("Uno IPCW C-index")
    axes[1].set_ylabel("Integrated Brier score")
    axes[0].set_title("Discrimination (higher is better)")
    axes[1].set_title("Prediction error (lower is better)")
    for ax in axes:
        ax.set_xticks(x, panel_labels, rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.2, linewidth=0.8)
        _bold_ticks(ax)
    axes[0].legend(frameon=False, prop={"family": "Times New Roman", "weight": "bold"})
    _panel(axes[0], "A")
    _panel(axes[1], "B")
    fig.tight_layout(w_pad=2.2)
    _save(fig, output)


def figure_external_validation(root: Path, output: Path) -> None:
    extended = root / "challengers" / "survpfn" / "clinical_model_comparison_external.csv"
    source = extended if extended.exists() else root / "external_validation" / "clinical_model_comparison.csv"
    data = pd.read_csv(source)
    labels = data["model"].map(
        {
            "survivalpfn": "SurvivalPFN",
            "survpfn": "SurvPFN (June 2026)",
            "elastic_net_cox": "Elastic-net Cox",
        }
    ).fillna(data["model"])
    color_map = {
        "survivalpfn": COLORS["purple"],
        "survpfn": COLORS["red"],
        "elastic_net_cox": COLORS["blue"],
    }
    colors = [color_map.get(model, COLORS["grey"]) for model in data["model"]]
    y = np.arange(len(data))
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.3), gridspec_kw={"width_ratios": [1.35, 1.05, 1.15]})
    low = data["uno_c"] - data["uno_c_ci95_low"]
    high = data["uno_c_ci95_high"] - data["uno_c"]
    axes[0].errorbar(
        data["uno_c"],
        y,
        xerr=np.vstack([low, high]),
        fmt="o",
        markersize=7,
        capsize=4,
        color=COLORS["navy"],
        ecolor=COLORS["navy"],
    )
    axes[0].axvline(0.5, color=COLORS["grey"], linestyle="--", linewidth=1.0)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("External Uno C-index (95% bootstrap CI)")
    axes[0].set_xlim(0.42, 0.80)
    axes[0].grid(axis="x", alpha=0.2, linewidth=0.8)

    axes[1].barh(y, data["integrated_brier"], color=colors, alpha=0.88)
    axes[1].set_yticks(y, labels)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("External IBS (1–3 years)")
    axes[1].grid(axis="x", alpha=0.2, linewidth=0.8)

    axes[2].axis("off")
    outer = plt.Circle((0.5, 0.52), 0.36, transform=axes[2].transAxes, color=COLORS["teal"], alpha=0.16)
    inner = plt.Circle((0.40, 0.52), 0.25, transform=axes[2].transAxes, color=COLORS["blue"], alpha=0.30)
    axes[2].add_patch(outer)
    axes[2].add_patch(inner)
    axes[2].text(0.37, 0.52, "119\noverlap", transform=axes[2].transAxes, ha="center", va="center")
    axes[2].text(0.72, 0.52, "60\nexternal", transform=axes[2].transAxes, ha="center", va="center")
    axes[2].text(0.50, 0.93, "GSE53625 SuperSeries (n = 179)", transform=axes[2].transAxes, ha="center")
    axes[2].text(0.50, 0.10, "Locked test: non-overlapping patients only\n33 deaths", transform=axes[2].transAxes, ha="center")
    for label, ax in zip("ABC", axes):
        _panel(ax, label)
        _bold_ticks(ax)
    fig.tight_layout(w_pad=2.0)
    _save(fig, output)


def figure_marker_stability(root: Path, output: Path) -> None:
    prominent = pd.read_csv(root / "biology" / "prominent_markers.csv")
    all_markers = pd.read_csv(root / "biology" / "marker_stability_all.csv")
    states = sorted(prominent["state"].unique())
    top = prominent.loc[prominent["state_rank"].le(3)].copy()
    frequencies = top.pivot(index="state_rank", columns="state", values="bootstrap_selection_frequency").reindex(
        index=[1, 2, 3], columns=states
    )
    genes = top.pivot(index="state_rank", columns="state", values="gene").reindex(
        index=[1, 2, 3], columns=states
    )
    stable_counts = (
        all_markers.loc[all_markers["stable_marker"]]
        .groupby("state")
        .size()
        .reindex(states, fill_value=0)
    )
    fig, axes = plt.subplots(2, 1, figsize=(14.2, 7.6), gridspec_kw={"height_ratios": [1.2, 1.0]})
    image = axes[0].imshow(frequencies.to_numpy(dtype=float), cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")
    for row in range(3):
        for column in range(len(states)):
            gene = genes.iloc[row, column]
            frequency = frequencies.iloc[row, column]
            axes[0].text(
                column,
                row,
                f"{gene}\n{frequency:.2f}",
                ha="center",
                va="center",
                fontsize=7.5,
                color="white" if frequency >= 0.72 else COLORS["navy"],
            )
    axes[0].set_xticks(np.arange(len(states)), states, rotation=28, ha="right")
    axes[0].set_yticks(np.arange(3), ["Rank 1", "Rank 2", "Rank 3"])
    axes[0].set_title("Top donor-bootstrap markers by cell state")
    colorbar = fig.colorbar(image, ax=axes[0], fraction=0.025, pad=0.015)
    colorbar.set_label("Selection frequency")
    _bold_ticks(colorbar.ax)

    colors = [COLORS["teal"] if value > 0 else COLORS["grey"] for value in stable_counts]
    axes[1].bar(np.arange(len(states)), stable_counts.to_numpy(), color=colors, alpha=0.88)
    axes[1].set_xticks(np.arange(len(states)), states, rotation=28, ha="right")
    axes[1].set_ylabel("Stable state-gene pairs")
    axes[1].set_title("Markers passing frequency >= 0.70 and positive bootstrap effect")
    axes[1].grid(axis="y", alpha=0.2, linewidth=0.8)
    for index, value in enumerate(stable_counts):
        axes[1].text(index, value + 0.5, str(int(value)), ha="center", va="bottom", fontsize=8)
    for label, ax in zip("AB", axes):
        _panel(ax, label)
        _bold_ticks(ax)
    fig.tight_layout(h_pad=2.0)
    _save(fig, output)


def _association_forest(ax: plt.Axes, data: pd.DataFrame, labels: list[str], title: str) -> None:
    y = np.arange(len(data))
    hazard = data["hazard_ratio_per_sd"].to_numpy(dtype=float)
    low = data["ci95_low"].to_numpy(dtype=float)
    high = data["ci95_high"].to_numpy(dtype=float)
    colors = [COLORS["red"] if value > 1 else COLORS["blue"] for value in hazard]
    for index in range(len(data)):
        ax.errorbar(
            hazard[index],
            y[index],
            xerr=[[hazard[index] - low[index]], [high[index] - hazard[index]]],
            fmt="o",
            color=colors[index],
            ecolor=colors[index],
            capsize=3,
            markersize=6,
        )
    ax.axvline(1.0, color=COLORS["grey"], linestyle="--", linewidth=1.0)
    ax.set_xscale("log")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Clinical-adjusted hazard ratio per 1 SD (95% CI)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.2, linewidth=0.8)
    _bold_ticks(ax)


def figure_exploratory_associations(root: Path, output: Path) -> None:
    genes = pd.read_csv(root / "biology" / "exploratory_gene_survival.csv").dropna(
        subset=["hazard_ratio_per_sd", "ci95_low", "ci95_high"]
    ).head(12)
    pathways = pd.read_csv(root / "biology" / "exploratory_pathway_survival.csv").dropna(
        subset=["hazard_ratio_per_sd", "ci95_low", "ci95_high"]
    ).head(10)
    gene_labels = [f"{gene} ({state})" for gene, state in zip(genes["feature"], genes["state"])]
    pathway_labels = [
        value.replace("pathway__REACTOME_", "").replace("_", " ").title()
        for value in pathways["feature"]
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.4), gridspec_kw={"width_ratios": [1.0, 1.25]})
    _association_forest(axes[0], genes, gene_labels, "Prominent state-marker associations")
    _association_forest(axes[1], pathways, pathway_labels, "Selected pathway associations")
    _panel(axes[0], "A")
    _panel(axes[1], "B")
    fig.suptitle(
        "Exploratory internal analysis: no gene or pathway passed BH FDR < 0.05",
        fontsize=13,
        y=1.01,
    )
    fig.tight_layout(w_pad=2.3)
    _save(fig, output)


def make_manuscript_figures(root: str | Path, output_dir: str | Path) -> pd.DataFrame:
    configure_manuscript_style()
    root = Path(root)
    output_dir = Path(output_dir)
    definitions: list[tuple[str, Any, str]] = [
        ("figure1_workflow.svg", figure_workflow, "cohort, reference, and feature summaries"),
        ("figure2_technical_validation.svg", figure_technical_validation, "technical_validation/summary.csv"),
        (
            "figure3_internal_benchmark.svg",
            figure_internal_benchmark,
            "benchmark/summary.csv and challengers/survpfn/summary.csv",
        ),
        (
            "figure4_external_validation.svg",
            figure_external_validation,
            "extended external clinical model comparison and audit",
        ),
    ]
    if (root / "biology" / "summary.json").exists():
        definitions.extend(
            [
                (
                    "figure5_marker_stability.svg",
                    figure_marker_stability,
                    "biology/prominent_markers.csv and marker_stability_all.csv",
                ),
                (
                    "figure6_exploratory_associations.svg",
                    figure_exploratory_associations,
                    "biology exploratory gene and pathway survival tables",
                ),
            ]
        )
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
    manifest.to_csv(output_dir / "figure_manifest.csv", index=False)
    return manifest

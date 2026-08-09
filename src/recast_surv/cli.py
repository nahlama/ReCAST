from __future__ import annotations

import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import click
import pandas as pd

from .benchmark import paired_representation_comparisons, run_nested_benchmark
from .biology import run_biology_analysis
from .cohort import build_tcga_cohort
from .config import artifacts_dir, load_config, resolve_path
from .external import audit_external_cohorts, run_locked_external_clinical_validation
from .features import build_features, read_hpa_protein_priors
from .figures import make_manuscript_figures
from .manuscript import make_manuscript_package
from .provenance import validate_source_manifest, write_run_manifest
from .reference import build_gse154763_reference, build_gse160269_reference, combine_references
from .simulation import run_transport_simulation
from .supplementary_figures import make_supplementary_figures
from .external_reannotation import map_probe_sequences_to_gencode, run_external_gene_confirmation
from .q1_extension import (
    make_probast_ai_self_assessment,
    run_clinical_utility_analysis,
    run_donor_heldout_projection_validation,
    write_q1_artifact_manifest,
    write_q1_extension_protocol,
)
from .q1_figures import make_q1_extension_figures


Loaded = tuple[dict[str, Any], Path, Path]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load(path: str) -> Loaded:
    config, workspace = load_config(path)
    return config, workspace, artifacts_dir(config, workspace)


def _validate(config: dict[str, Any], workspace: Path, root: Path) -> None:
    manifest_path = resolve_path(workspace, config["data"]["source_manifest"])
    result = validate_source_manifest(manifest_path, workspace)
    output = root / "provenance" / "source_validation.tsv"
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, sep="\t", index=False)
    write_run_manifest(root / "provenance" / "validate_manifest.json", "validate", [manifest_path])


def _build_cohort(config: dict[str, Any], workspace: Path, root: Path) -> None:
    metadata_path = resolve_path(workspace, config["data"]["tcga_metadata"])
    vst_path = resolve_path(workspace, config["data"]["tcga_vst"])
    settings = config["cohort"]
    cohort, expression, summary = build_tcga_cohort(
        metadata_path,
        vst_path,
        min_time_days=float(settings["min_time_days"]),
        include_histologies=tuple(settings["include_histologies"]),
    )
    output = root / "cohort"
    output.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(output / "clinical.csv", index=False)
    expression.to_parquet(output / "expression.parquet", index=True)
    _write_json(output / "summary.json", summary)
    write_run_manifest(output / "run_manifest.json", "build-cohort", [metadata_path, vst_path])


def _build_reference(config: dict[str, Any], workspace: Path, root: Path) -> None:
    data = config["data"]
    settings = config["reference"]
    parts = []
    inputs: list[Path] = []
    if settings.get("include_gse160269", True):
        directory = resolve_path(workspace, data["gse160269_dir"])
        tisch = resolve_path(workspace, data["gse160269_tisch_metadata"])
        parts.append(
            build_gse160269_reference(
                directory,
                tisch,
                state_field=str(settings["state_field"]),
                max_genes=settings.get("max_genes"),
                max_cells_per_compartment=settings.get("max_cells_per_compartment"),
            )
        )
        inputs.extend([tisch, *directory.glob("*.gz")])
    if settings.get("include_gse154763", True):
        directory = resolve_path(workspace, data["gse154763_dir"])
        parts.append(
            build_gse154763_reference(
                directory,
                max_genes=settings.get("max_genes"),
                chunksize=int(settings["gse154763_chunksize"]),
            )
        )
        inputs.extend(directory.glob("*.gz"))
    if not parts:
        raise ValueError("At least one single-cell reference must be enabled")
    metadata, profiles = combine_references(parts)
    output = root / "reference"
    output.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(output / "metadata.csv", index=False)
    profiles.to_parquet(output / "profiles.parquet", index=True)
    summary = {
        "profiles": int(len(profiles)),
        "genes": int(profiles.shape[1]),
        "states": int(metadata["state"].nunique()),
        "patients": int(metadata["patient"].nunique()),
        "cells_summarized": int(metadata["n_cells"].sum()),
        "sources": metadata["source"].value_counts().to_dict(),
    }
    _write_json(output / "summary.json", summary)
    write_run_manifest(output / "run_manifest.json", "build-reference", inputs)


def _build_features(config: dict[str, Any], workspace: Path, root: Path) -> None:
    cohort = pd.read_csv(root / "cohort" / "clinical.csv", index_col="sample_id")
    expression = pd.read_parquet(root / "cohort" / "expression.parquet")
    reference_metadata = pd.read_csv(root / "reference" / "metadata.csv")
    reference_profiles = pd.read_parquet(root / "reference" / "profiles.parquet")
    paths = [resolve_path(workspace, item) for item in config["data"]["pathways"]]
    protein_prior_path = resolve_path(workspace, config["data"]["hpa_proteinatlas"])
    features, outcomes, diagnostics = build_features(
        cohort,
        expression,
        reference_metadata,
        reference_profiles,
        paths,
        config["features"],
        protein_prior_path=protein_prior_path,
    )
    output = root / "features"
    output.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output / "features.parquet", index=True)
    outcomes.to_csv(output / "outcomes.csv", index=True)
    _write_json(output / "diagnostics.json", diagnostics)
    write_run_manifest(output / "run_manifest.json", "build-features", [*paths, protein_prior_path])


def _benchmark(config: dict[str, Any], workspace: Path, root: Path) -> None:
    features = pd.read_parquet(root / "features" / "features.parquet")
    outcomes = pd.read_csv(root / "features" / "outcomes.csv", index_col=0)
    _, _, metrics = run_nested_benchmark(
        features,
        outcomes,
        config["benchmark"],
        workspace,
        root / "benchmark",
        seed=int(config["project"]["seed"]),
    )
    excluded = {"repeat", "fold", "n_test", "events_test"}
    metric_columns = [
        column
        for column in metrics.select_dtypes(include="number").columns
        if column not in excluded
    ]
    summary = metrics.groupby(["panel", "model"])[metric_columns].agg(["mean", "std", "count"]).reset_index()
    summary.columns = [
        "_".join(part for part in item if part) if isinstance(item, tuple) else item
        for item in summary.columns
    ]
    summary.to_csv(root / "benchmark" / "summary.csv", index=False)


def _benchmark_ablations(config: dict[str, Any], workspace: Path, root: Path) -> None:
    cohort = pd.read_csv(root / "cohort" / "clinical.csv", index_col="sample_id")
    expression = pd.read_parquet(root / "cohort" / "expression.parquet")
    reference_metadata = pd.read_csv(root / "reference" / "metadata.csv")
    reference_profiles = pd.read_parquet(root / "reference" / "profiles.parquet")
    pathway_paths = [resolve_path(workspace, item) for item in config["data"]["pathways"]]
    hpa_path = resolve_path(workspace, config["data"]["hpa_proteinatlas"])
    definitions = config.get("ablations", {}).get("representations", [])
    if not definitions:
        raise ValueError("No ablations.representations are configured")
    ablation_models = config.get("ablations", {}).get("models")
    combined_metrics = []
    ablation_root = root / "ablations"
    for definition in definitions:
        name = str(definition["name"])
        feature_settings = dict(config["features"])
        for key in (
            "backend",
            "protein_prior_strength",
            "transport_epsilon",
            "mass_penalty",
            "target_mass_penalty",
        ):
            if key in definition:
                feature_settings[key] = definition[key]
        use_hpa = bool(definition.get("use_protein_priors", True))
        features, outcomes, diagnostics = build_features(
            cohort,
            expression,
            reference_metadata,
            reference_profiles,
            pathway_paths,
            feature_settings,
            protein_prior_path=hpa_path if use_hpa else None,
        )
        drop_features = list(definition.get("drop_features", []))
        if drop_features:
            missing = sorted(set(drop_features) - set(features.columns))
            if missing:
                raise ValueError(f"Ablation {name} cannot drop absent features: {missing}")
            features = features.drop(columns=drop_features)
            diagnostics["dropped_features"] = drop_features
            diagnostics["features"] = int(features.shape[1])
        variant_root = ablation_root / name
        variant_root.mkdir(parents=True, exist_ok=True)
        features.to_parquet(variant_root / "features.parquet", index=True)
        _write_json(variant_root / "diagnostics.json", diagnostics)
        benchmark_settings = dict(config["benchmark"])
        if ablation_models:
            benchmark_settings["models"] = list(ablation_models)
        _, _, metrics = run_nested_benchmark(
            features,
            outcomes,
            benchmark_settings,
            workspace,
            variant_root / "benchmark",
            seed=int(config["project"]["seed"]),
        )
        metrics.insert(0, "representation", name)
        combined_metrics.append(metrics)
    combined = pd.concat(combined_metrics, ignore_index=True)
    combined.to_csv(ablation_root / "combined_fold_metrics.csv", index=False)
    baseline = str(config.get("ablations", {}).get("primary_representation", definitions[0]["name"]))
    comparisons = paired_representation_comparisons(
        combined,
        baseline_representation=baseline,
        seed=int(config["project"]["seed"]),
    )
    comparisons.to_csv(ablation_root / "representation_comparisons.csv", index=False)


def _benchmark_transport(config: dict[str, Any], workspace: Path, root: Path) -> None:
    metadata = pd.read_csv(root / "reference" / "metadata.csv")
    profiles = pd.read_parquet(root / "reference" / "profiles.parquet")
    hpa_path = resolve_path(workspace, config["data"]["hpa_proteinatlas"])
    priors = read_hpa_protein_priors(hpa_path)
    validation = config.get("technical_validation", {})
    results, summary = run_transport_simulation(
        metadata,
        profiles,
        config["features"],
        gene_reliability=priors,
        mixtures_per_scenario=int(validation.get("mixtures_per_scenario", 250)),
        seed=int(config["project"]["seed"]),
    )
    output = root / "technical_validation"
    output.mkdir(parents=True, exist_ok=True)
    results.to_csv(output / "mixture_results.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)


def _audit_external(config: dict[str, Any], workspace: Path, root: Path) -> None:
    settings = config["external_validation"]
    audit_external_cohorts(
        resolve_path(workspace, settings["prepared_root"]),
        list(settings["accessions"]),
        resolve_path(workspace, settings["gpl19748_soft"]),
        resolve_path(workspace, config["data"]["tcga_vst"]),
        root / "external_validation" / "audit",
    )


def _validate_external_clinical(config: dict[str, Any], workspace: Path, root: Path) -> None:
    settings = config["external_validation"]
    training_features = pd.read_parquet(root / "features" / "features.parquet")
    training_outcomes = pd.read_csv(root / "features" / "outcomes.csv", index_col=0)
    clinical = pd.read_csv(root / "external_validation" / "audit" / "external_clinical.csv")
    results = []
    for model_name in settings.get("clinical_models", ["survivalpfn"]):
        result = run_locked_external_clinical_validation(
            training_features,
            training_outcomes,
            clinical,
            workspace,
            config["benchmark"].get("survivalpfn", {}),
            str(settings["primary_series"]),
            str(settings["overlap_series"]),
            root / "external_validation" / f"clinical_{model_name}",
            seed=int(config["project"]["seed"]),
            evaluation_times=list(config["benchmark"]["time_grid_days"]),
            bootstrap_iterations=int(settings.get("bootstrap_iterations", 2000)),
            model_name=str(model_name),
            inner_folds=int(config["benchmark"].get("inner_folds", 4)),
        )
        row = {
            "model": result["model"],
            "n_external": result["n_external"],
            "events_external": result["events_external"],
            "selected_params": json.dumps(result["selected_params"], sort_keys=True),
            **result["metrics"],
        }
        for metric, interval in result["bootstrap_95_ci"].items():
            row[f"{metric}_ci95_low"] = interval["low"]
            row[f"{metric}_ci95_high"] = interval["high"]
        results.append(row)
    comparison_path = root / "external_validation" / "clinical_model_comparison.csv"
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(comparison_path, index=False)


def _make_figures(config: dict[str, Any], workspace: Path, root: Path) -> None:
    del config, workspace
    make_manuscript_figures(root, root / "figures")


def _make_supplementary_figures(config: dict[str, Any], workspace: Path, root: Path) -> None:
    del config, workspace
    make_supplementary_figures(root, root / "figures" / "supplementary")


def _make_manuscript(config: dict[str, Any], workspace: Path, root: Path) -> None:
    del config, workspace
    make_manuscript_package(root, root / "manuscript")


def _make_toolkit_figures(config: dict[str, Any], workspace: Path, root: Path) -> None:
    """Render the publication-toolkit figures (main Fig 3-7 and supplements S1-S5).

    These live under ``Manuscript/`` and use the shared style contract
    (``Manuscript/mpl_style.py``). They read only committed artifacts under
    ``root`` and write editable SVG+PNG+PDF into ``root/figures/toolkit``.
    Figure 7 consumes the locked IPCW decision-curve artifact produced by
    ``run-q1-extension`` (q1_extension/clinical_utility/decision_curve_external.csv).
    """
    del config
    manuscript_dir = workspace / "Manuscript"
    toolkit_out = root / "figures" / "toolkit"
    env = dict(os.environ)
    env["RECAST_ARTIFACTS_ROOT"] = str(root)
    env["RECAST_FIGURES_OUT"] = str(toolkit_out)
    env["RECAST_SUPP_FIGURES_OUT"] = str(toolkit_out / "supplementary")
    for script in ("make_manuscript_figures.py", "make_supplementary_figures.py"):
        subprocess.run(
            [sys.executable, script],
            cwd=str(manuscript_dir),
            env=env,
            check=True,
        )


def _make_toolkit_tables(config: dict[str, Any], workspace: Path, root: Path) -> None:
    """Render the publication-ready manuscript + supplementary tables.

    Reads the pipeline's computed tables under ``root/manuscript`` (from
    ``make-manuscript``), formats them, and writes CSV+Markdown plus an index
    into ``root/manuscript/tables_formatted`` (and ``Manuscript/Tables`` when
    run standalone).
    """
    del config
    manuscript_dir = workspace / "Manuscript"
    env = dict(os.environ)
    env["RECAST_ARTIFACTS_ROOT"] = str(root)
    env["RECAST_TABLES_OUT"] = str(root / "manuscript" / "tables_formatted")
    subprocess.run(
        [sys.executable, "make_tables.py"],
        cwd=str(manuscript_dir),
        env=env,
        check=True,
    )


def _analyze_biology(config: dict[str, Any], workspace: Path, root: Path) -> None:
    run_biology_analysis(
        root,
        resolve_path(workspace, config["data"]["hpa_proteinatlas"]),
        config.get("biology", {}),
        seed=int(config["project"]["seed"]),
    )


def _run_q1_extension(config: dict[str, Any], workspace: Path, root: Path) -> None:
    settings = config.get("q1_extension", {})
    output = root / "q1_extension"
    write_q1_extension_protocol(output, seed=int(config["project"]["seed"]))
    metadata = pd.read_csv(root / "reference" / "metadata.csv")
    profiles = pd.read_parquet(root / "reference" / "profiles.parquet")
    priors = read_hpa_protein_priors(resolve_path(workspace, config["data"]["hpa_proteinatlas"]))
    run_donor_heldout_projection_validation(
        metadata,
        profiles,
        config["features"],
        priors,
        output / "projection_validation",
        mixtures_per_donor=int(settings.get("mixtures_per_donor", 5)),
        seed=int(config["project"]["seed"]),
    )
    run_clinical_utility_analysis(
        root,
        output / "clinical_utility",
        seed=int(config["project"]["seed"]),
    )
    make_probast_ai_self_assessment(output / "probast_ai")
    write_q1_artifact_manifest(output)


def _reannotate_external_omics(config: dict[str, Any], workspace: Path, root: Path) -> None:
    settings = config["q1_extension"]
    fasta = resolve_path(workspace, settings["gencode_transcripts"])
    if not fasta.exists():
        raise FileNotFoundError(
            f"GENCODE transcript FASTA is missing: {fasta}. Download the frozen release before reannotation."
        )
    output = root / "q1_extension" / "external_reannotation"
    mapping_path = output / "probe_to_gencode50.csv.gz"
    if mapping_path.exists():
        mapping = pd.read_csv(mapping_path, dtype={"feature_id": str})
    else:
        mapping, _ = map_probe_sequences_to_gencode(
            root / "external_validation" / "audit" / "GPL18109_to_GPL19748_sequence_crosswalk.csv.gz",
            fasta,
            output,
        )
    run_external_gene_confirmation(
        root,
        resolve_path(workspace, config["external_validation"]["prepared_root"]),
        mapping,
        [resolve_path(workspace, item) for item in config["data"]["pathways"]],
        output,
    )
    write_q1_artifact_manifest(root / "q1_extension")


def _make_q1_figures(config: dict[str, Any], workspace: Path, root: Path) -> None:
    del config, workspace
    make_q1_extension_figures(root, root / "figures" / "supplementary")


@click.group()
@click.option("--config", "config_path", default="configs/default.yaml", show_default=True, type=click.Path(exists=True))
@click.pass_context
def main(context: click.Context, config_path: str) -> None:
    """ReCAST-Surv reproducible pipeline."""
    context.ensure_object(dict)
    context.obj["loaded"] = _load(config_path)


@main.command("validate")
@click.pass_context
def validate_command(context: click.Context) -> None:
    _validate(*context.obj["loaded"])
    click.echo("Source manifest validated.")


@main.command("build-cohort")
@click.pass_context
def cohort_command(context: click.Context) -> None:
    _build_cohort(*context.obj["loaded"])
    click.echo("TCGA cohort built.")


@main.command("build-reference")
@click.pass_context
def reference_command(context: click.Context) -> None:
    _build_reference(*context.obj["loaded"])
    click.echo("Single-cell reference built.")


@main.command("build-features")
@click.pass_context
def features_command(context: click.Context) -> None:
    _build_features(*context.obj["loaded"])
    click.echo("ReCAST and pathway features built.")


@main.command("benchmark")
@click.option("--outer-repeats", type=click.IntRange(min=1), default=None, help="Override configured repeats.")
@click.option("--models", multiple=True, help="Override configured model list; may be repeated.")
@click.pass_context
def benchmark_command(context: click.Context, outer_repeats: int | None, models: tuple[str, ...]) -> None:
    config, workspace, root = context.obj["loaded"]
    runtime_config = deepcopy(config)
    if outer_repeats is not None:
        runtime_config["benchmark"]["outer_repeats"] = int(outer_repeats)
    if models:
        runtime_config["benchmark"]["models"] = list(models)
    _benchmark(runtime_config, workspace, root)
    click.echo("Nested benchmark completed.")


@main.command("benchmark-ablations")
@click.pass_context
def benchmark_ablations_command(context: click.Context) -> None:
    _benchmark_ablations(*context.obj["loaded"])
    click.echo("Representation ablations completed.")


@main.command("benchmark-transport")
@click.pass_context
def benchmark_transport_command(context: click.Context) -> None:
    _benchmark_transport(*context.obj["loaded"])
    click.echo("Transport technical validation completed.")


@main.command("audit-external")
@click.pass_context
def audit_external_command(context: click.Context) -> None:
    _audit_external(*context.obj["loaded"])
    click.echo("External validation cohorts audited.")


@main.command("validate-external-clinical")
@click.pass_context
def validate_external_clinical_command(context: click.Context) -> None:
    _validate_external_clinical(*context.obj["loaded"])
    click.echo("Locked external clinical model validation completed.")


@main.command("make-figures")
@click.pass_context
def make_figures_command(context: click.Context) -> None:
    _make_figures(*context.obj["loaded"])
    click.echo("Editable SVG manuscript figures created.")


@main.command("make-supplementary-figures")
@click.pass_context
def make_supplementary_figures_command(context: click.Context) -> None:
    _make_supplementary_figures(*context.obj["loaded"])
    click.echo("Editable SVG supplementary figures created.")


@main.command("make-manuscript")
@click.pass_context
def make_manuscript_command(context: click.Context) -> None:
    _make_manuscript(*context.obj["loaded"])
    click.echo("Manuscript tables, drafts, and reporting checklist created.")


@main.command("make-toolkit-figures")
@click.pass_context
def make_toolkit_figures_command(context: click.Context) -> None:
    _make_toolkit_figures(*context.obj["loaded"])
    click.echo("Publication-toolkit figures (Fig 3-7, S1-S12) created under figures/toolkit.")


@main.command("make-toolkit-tables")
@click.pass_context
def make_toolkit_tables_command(context: click.Context) -> None:
    _make_toolkit_tables(*context.obj["loaded"])
    click.echo("Publication tables (Table 1-6, Supplementary 1-10) created under manuscript/tables_formatted.")


@main.command("analyze-biology")
@click.pass_context
def analyze_biology_command(context: click.Context) -> None:
    _analyze_biology(*context.obj["loaded"])
    click.echo("Outcome-free marker stability and exploratory survival associations completed.")


@main.command("run-q1-extension")
@click.pass_context
def run_q1_extension_command(context: click.Context) -> None:
    _run_q1_extension(*context.obj["loaded"])
    click.echo("Locked Q1-strengthening projection, utility, subgroup, and PROBAST+AI analyses completed.")


@main.command("reannotate-external-omics")
@click.pass_context
def reannotate_external_omics_command(context: click.Context) -> None:
    _reannotate_external_omics(*context.obj["loaded"])
    click.echo("GENCODE 50 external probe reannotation and locked molecular confirmation completed.")


@main.command("make-q1-figures")
@click.pass_context
def make_q1_figures_command(context: click.Context) -> None:
    _make_q1_figures(*context.obj["loaded"])
    click.echo("Editable SVG Q1-extension figures S8-S12 created.")


@main.command("run-all")
@click.pass_context
def run_all_command(context: click.Context) -> None:
    loaded = context.obj["loaded"]
    _validate(*loaded)
    _build_cohort(*loaded)
    _build_reference(*loaded)
    _build_features(*loaded)
    _benchmark_transport(*loaded)
    _benchmark(*loaded)
    _audit_external(*loaded)
    _validate_external_clinical(*loaded)
    _analyze_biology(*loaded)
    _make_figures(*loaded)
    _make_supplementary_figures(*loaded)
    _make_manuscript(*loaded)
    _make_toolkit_figures(*loaded)
    _make_toolkit_tables(*loaded)
    click.echo("Pipeline completed.")


if __name__ == "__main__":
    main()

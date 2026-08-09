from __future__ import annotations

import gzip
import hashlib
import json
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .biology import benjamini_hochberg
from .external import external_clinical_features
from .manuscript import _external_subset
from .pathways import rank_pathway_scores, read_gmt


GENCODE_RELEASE = "50"
GENCODE_ASSEMBLY = "GRCh38.p14"
GENCODE_TRANSCRIPT_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/"
    "release_50/gencode.v50.transcripts.fa.gz"
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reverse_complement(sequence: str) -> str:
    return sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def _fasta_records(path: Path) -> Iterator[tuple[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    header = ""
    sequence: list[str] = []
    with opener(path, "rt", encoding="ascii", errors="strict") as handle:
        for line in handle:
            if line.startswith(">"):
                if header:
                    yield header, "".join(sequence).upper()
                header = line[1:].strip()
                sequence = []
            else:
                sequence.append(line.strip())
    if header:
        yield header, "".join(sequence).upper()


def map_probe_sequences_to_gencode(
    crosswalk_path: str | Path,
    transcript_fasta: str | Path,
    output_dir: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Strict full-length exact probe mapping to unique GENCODE genes on either strand."""

    try:
        import ahocorasick
    except ImportError as exc:  # pragma: no cover - exercised by the CLI environment gate
        raise RuntimeError("pyahocorasick is required for versioned probe reannotation") from exc

    crosswalk_path = Path(crosswalk_path)
    transcript_fasta = Path(transcript_fasta)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(
        crosswalk_path,
        usecols=["feature_id", "sequence_key", "feature_sequence_count"],
        dtype={"feature_id": str, "sequence_key": str},
    ).drop_duplicates(["feature_id", "sequence_key"])
    source["sequence_key"] = source["sequence_key"].str.upper().str.strip()
    source["valid_sequence"] = source["sequence_key"].str.fullmatch(r"[ACGT]+", na=False) & source["sequence_key"].str.len().ge(25)
    source["unique_feature_sequence"] = source["feature_sequence_count"].eq(1)
    eligible = source[source["valid_sequence"] & source["unique_feature_sequence"]].copy()

    pattern_features: dict[str, list[str]] = defaultdict(list)
    for feature_id, sequence in eligible[["feature_id", "sequence_key"]].itertuples(index=False):
        pattern_features[sequence].append(feature_id)
        pattern_features[_reverse_complement(sequence)].append(feature_id)
    automaton = ahocorasick.Automaton()
    for pattern, features in pattern_features.items():
        automaton.add_word(pattern, (pattern, tuple(sorted(set(features)))))
    automaton.make_automaton()

    gene_hits: dict[str, set[tuple[str, str]]] = defaultdict(set)
    transcripts_scanned = 0
    for header, sequence in _fasta_records(transcript_fasta):
        fields = header.split("|")
        if len(fields) < 6:
            continue
        gene_id = fields[1].split(".")[0]
        gene_symbol = fields[5]
        transcripts_scanned += 1
        matched_features: set[str] = set()
        for _, (_, feature_ids) in automaton.iter(sequence):
            matched_features.update(feature_ids)
        for feature_id in matched_features:
            gene_hits[feature_id].add((gene_id, gene_symbol))

    rows: list[dict[str, Any]] = []
    for row in source.itertuples(index=False):
        hits = sorted(gene_hits.get(str(row.feature_id), set()))
        gene_ids = sorted({item[0] for item in hits})
        symbols = sorted({item[1] for item in hits})
        if not row.valid_sequence:
            status = "invalid_or_short_sequence"
        elif not row.unique_feature_sequence:
            status = "duplicated_feature_sequence"
        elif not hits:
            status = "no_exact_transcript_match"
        elif len(gene_ids) == 1 and len(symbols) == 1:
            status = "unique_gene_exact_match"
        else:
            status = "ambiguous_gene_exact_match"
        rows.append(
            {
                "feature_id": str(row.feature_id),
                "probe_sequence": row.sequence_key,
                "probe_length": len(row.sequence_key),
                "mapping_status": status,
                "gencode_gene_ids": "|".join(gene_ids),
                "gene_symbols": "|".join(symbols),
                "matched_gene_count": len(gene_ids),
            }
        )
    mapping = pd.DataFrame(rows)
    mapping.to_csv(output_dir / "probe_to_gencode50.csv.gz", index=False)
    summary = {
        "reference": f"GENCODE {GENCODE_RELEASE} ({GENCODE_ASSEMBLY})",
        "reference_url": GENCODE_TRANSCRIPT_URL,
        "reference_sha256": _sha256(transcript_fasta),
        "reference_bytes": transcript_fasta.stat().st_size,
        "mapping_rule": "full-length exact transcript match on forward or reverse-complement strand; retain one GENCODE gene only",
        "feature_rows": int(len(mapping)),
        "eligible_unique_sequences": int(len(eligible)),
        "transcripts_scanned": transcripts_scanned,
        "status_counts": mapping["mapping_status"].value_counts().to_dict(),
        "unique_gene_symbols": int(mapping.loc[mapping["mapping_status"].eq("unique_gene_exact_match"), "gene_symbols"].nunique()),
    }
    _write_json(output_dir / "mapping_summary.json", summary)
    return mapping, summary


def aggregate_external_probe_expression(
    probe_matrix_path: str | Path,
    mapping: pd.DataFrame,
    sample_ids: list[str],
    output_path: str | Path,
) -> pd.DataFrame:
    unique = mapping[mapping["mapping_status"] == "unique_gene_exact_match"][["feature_id", "gene_symbols"]].copy()
    unique["feature_id"] = unique["feature_id"].astype(str)
    expression = pd.read_csv(
        probe_matrix_path,
        usecols=["feature_id", *sample_ids],
        dtype={"feature_id": str},
        low_memory=False,
    )
    expression = expression.merge(unique, on="feature_id", how="inner", validate="many_to_one")
    numeric = expression[sample_ids].apply(pd.to_numeric, errors="coerce")
    numeric["gene_symbol"] = expression["gene_symbols"].to_numpy()
    gene_expression = numeric.groupby("gene_symbol", sort=True).median()
    gene_expression.to_parquet(output_path)
    return gene_expression


def _penalized_external_associations(
    candidates: pd.DataFrame,
    clinical: pd.DataFrame,
    outcomes: pd.DataFrame,
    family: str,
    penalizer: float = 0.05,
) -> pd.DataFrame:
    """Small-sample ridge Cox confirmation with all continuous inputs standardized."""

    base_values = SimpleImputer(strategy="median").fit_transform(clinical)
    base = pd.DataFrame(
        StandardScaler().fit_transform(base_values),
        index=clinical.index,
        columns=clinical.columns,
    )
    rows: list[dict[str, Any]] = []
    for candidate in candidates.columns:
        values = pd.to_numeric(candidates.loc[base.index, candidate], errors="coerce")
        values = values.fillna(values.median())
        sd = float(values.std(ddof=0))
        if not np.isfinite(sd) or sd <= 1e-10:
            rows.append({"feature": candidate, "family": family, "error": "constant_feature"})
            continue
        frame = base.copy()
        frame["candidate"] = (values - values.mean()) / sd
        frame["time_days"] = outcomes.loc[frame.index, "time_days"].to_numpy(dtype=float)
        frame["event"] = outcomes.loc[frame.index, "event"].to_numpy(dtype=int)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                model = CoxPHFitter(penalizer=penalizer, l1_ratio=0.0).fit(
                    frame,
                    duration_col="time_days",
                    event_col="event",
                    robust=True,
                )
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
                    "penalizer": penalizer,
                    "n": len(frame),
                    "events": int(frame["event"].sum()),
                    "error": "",
                }
            )
        except (ValueError, ArithmeticError) as exc:
            rows.append({"feature": candidate, "family": family, "error": f"{type(exc).__name__}: {exc}"})
    result = pd.DataFrame(rows)
    result["fdr_bh"] = benjamini_hochberg(result.get("p_value", pd.Series(dtype=float)))
    return result


def run_external_gene_confirmation(
    root: str | Path,
    prepared_root: str | Path,
    mapping: pd.DataFrame,
    pathway_paths: list[Path],
    output_dir: str | Path,
) -> dict[str, Any]:
    root = Path(root)
    prepared_root = Path(prepared_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audited = pd.read_csv(root / "external_validation" / "audit" / "external_clinical.csv")
    external = _external_subset(audited).set_index("sample_id")
    sample_ids = external.index.astype(str).tolist()
    expression = aggregate_external_probe_expression(
        prepared_root / "GSE53625" / "processed" / "expression_probe_matrix.csv",
        mapping,
        sample_ids,
        output_dir / "GSE53625_locked_gene_expression.parquet",
    )
    expression = expression.T.reindex(sample_ids)
    tcga_genes = set(pd.read_parquet(root / "cohort" / "expression.parquet").columns)
    mapped_genes = set(expression.columns)
    overlap = sorted(mapped_genes & tcga_genes)

    internal = pd.read_csv(root / "biology" / "exploratory_gene_survival.csv")
    requested = internal["feature"].drop_duplicates().astype(str).tolist()
    available = [gene for gene in requested if gene in expression.columns]
    clinical = external_clinical_features(external.reset_index())
    clinical = clinical.loc[:, clinical.std(ddof=0).gt(1e-10)]
    outcomes = external[["time_days", "event"]].copy()
    outcomes.index.name = "sample_id"
    gene_results = _penalized_external_associations(expression[available], clinical, outcomes, "external_gene")
    gene_results = gene_results.merge(
        internal[["feature", "log_hazard_ratio_per_sd", "fdr_bh"]].rename(
            columns={"log_hazard_ratio_per_sd": "internal_log_hr", "fdr_bh": "internal_fdr_bh"}
        ),
        on="feature",
        how="left",
    )
    if "log_hazard_ratio_per_sd" not in gene_results:
        gene_results["log_hazard_ratio_per_sd"] = np.nan
    gene_results["direction_concordant"] = np.sign(gene_results["log_hazard_ratio_per_sd"]) == np.sign(gene_results["internal_log_hr"])
    gene_results.to_csv(output_dir / "external_prominent_gene_confirmation.csv", index=False)

    library = read_gmt(pathway_paths)
    diagnostics = json.loads((root / "features" / "diagnostics.json").read_text(encoding="utf-8"))
    selected = {
        name: [gene for gene in library[name] if gene in mapped_genes]
        for name in diagnostics["selected_pathways"]
        if name in library and len([gene for gene in library[name] if gene in mapped_genes]) >= 8
    }
    pathway_scores = rank_pathway_scores(expression, selected)
    pathway_results = _penalized_external_associations(pathway_scores, clinical, outcomes, "external_pathway")
    pathway_results.to_csv(output_dir / "external_pathway_confirmation.csv", index=False)

    summary = {
        "analysis_role": "locked external molecular confirmation after versioned reannotation",
        "external_patients": int(len(external)),
        "external_events": int(external["event"].sum()),
        "mapped_external_genes": int(len(mapped_genes)),
        "tcga_external_gene_overlap": int(len(overlap)),
        "prominent_genes_requested": len(requested),
        "prominent_genes_available": len(available),
        "external_gene_fdr_lt_0_05": int(gene_results["fdr_bh"].lt(0.05).sum()),
        "direction_concordance_fraction": float(gene_results.loc[gene_results["log_hazard_ratio_per_sd"].notna(), "direction_concordant"].mean()),
        "external_association_model": "ridge Cox penalizer=0.05; clinical adjusted; effect per one SD; BH FDR by family",
        "external_pathways_tested": int(len(pathway_results)),
        "external_pathway_fdr_lt_0_05": int(pathway_results["fdr_bh"].lt(0.05).sum()),
        "claim_boundary": "external associations are confirmatory only for prespecified genes/pathways; no external feature selection or recalibration",
    }
    _write_json(output_dir / "confirmation_summary.json", summary)
    pd.DataFrame({"gene": overlap}).to_csv(output_dir / "tcga_external_gene_overlap.csv", index=False)
    return summary

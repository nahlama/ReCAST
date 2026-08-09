from __future__ import annotations

import gzip
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


IDENTIFIER_COLUMNS = ["profile_id", "source", "patient", "sample", "tissue", "state", "n_cells"]


def harmonize_cell_state(value: str) -> str:
    """Map GSE154763 myeloid subclusters onto the shared major-state ontology."""
    text = str(value)
    if text.startswith("M01_"):
        return "Mast"
    if text.startswith(("M02_", "M03_", "M04_", "M05_")):
        return "DC"
    if text.startswith(("M06_", "M07_", "M08_", "M09_", "M10_", "M11_")):
        return "Mono/Macro"
    return text


def _balanced_positions(metadata: pd.DataFrame, limit: int | None) -> np.ndarray:
    if limit is None or len(metadata) <= limit:
        return np.arange(len(metadata), dtype=int)
    group = metadata["sample"].astype(str) + "::" + metadata["state"].astype(str)
    per_group = max(1, math.ceil(limit / group.nunique()))
    selected = metadata.assign(_group=group).groupby("_group", sort=True).head(per_group)
    return np.sort(selected.index.to_numpy(dtype=int)[:limit])


def _read_tisch_crosswalk(path: Path, state_field: str) -> pd.DataFrame:
    table = pd.read_csv(path, sep="\t")
    required = {"Cell", "Patient", "Sample", "Tissue", state_field}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"TISCH metadata missing columns: {sorted(missing)}")
    result = pd.DataFrame(
        {
            "raw_cell": table["Cell"].astype(str).str.split("@", n=1).str[-1],
            "patient": table["Patient"].astype(str),
            "sample": table["Sample"].astype(str),
            "tissue": table["Tissue"].astype(str),
            "state": table[state_field].astype(str),
        }
    )
    if result["raw_cell"].duplicated().any():
        raise ValueError("TISCH crosswalk contains duplicated raw cell identifiers")
    return result.set_index("raw_cell")


def _gse160269_cell_metadata(
    cells_path: Path,
    crosswalk: pd.DataFrame,
) -> pd.DataFrame:
    coarse = pd.read_csv(cells_path, sep=r"\s+", compression="gzip")
    if list(coarse.columns) != ["cell", "sample", "annotated_type"]:
        raise ValueError(f"Unexpected GSE160269 annotation header in {cells_path.name}")
    joined = coarse.join(
        crosswalk,
        on="cell",
        how="left",
        validate="one_to_one",
        lsuffix="_geo",
        rsuffix="_tisch",
    )
    # GEO contains one cell omitted by TISCH2; retain its coarse label.
    joined["patient"] = joined["patient"].fillna(
        joined["sample_geo"].str.replace(r"[TN]$", "", regex=True)
    )
    joined["sample"] = joined["sample_tisch"].fillna(joined["sample_geo"])
    inferred_tissue = pd.Series(
        np.where(joined["sample_geo"].str.endswith("N"), "Normal", "Tumor"),
        index=joined.index,
    )
    joined["tissue"] = joined["tissue"].fillna(inferred_tissue)
    joined["state"] = joined["state"].fillna(joined["annotated_type"])
    return joined[["cell", "patient", "sample", "tissue", "state", "annotated_type"]].reset_index(drop=True)


def _stream_gse160269_matrix(
    matrix_path: Path,
    metadata: pd.DataFrame,
    max_genes: int | None,
    max_cells: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    positions = _balanced_positions(metadata, max_cells)
    selected = metadata.iloc[positions].reset_index(drop=True)
    group_keys = list(
        zip(
            selected["patient"].astype(str),
            selected["sample"].astype(str),
            selected["tissue"].astype(str),
            selected["state"].astype(str),
        )
    )
    unique_groups = sorted(set(group_keys))
    group_to_code = {key: index for index, key in enumerate(unique_groups)}
    codes = np.asarray([group_to_code[key] for key in group_keys], dtype=np.int32)
    group_cell_counts = np.bincount(codes, minlength=len(unique_groups)).astype(int)

    genes: list[str] = []
    columns: list[np.ndarray] = []
    with gzip.open(matrix_path, "rb") as handle:
        matrix_cells = handle.readline().decode("utf-8-sig").strip().split()
        if matrix_cells != metadata["cell"].astype(str).tolist():
            raise ValueError(f"Matrix columns do not match cell annotations: {matrix_path.name}")
        for raw_line in handle:
            if not raw_line.strip():
                continue
            gene_raw, values_raw = raw_line.split(maxsplit=1)
            gene = gene_raw.decode("utf-8")
            values = np.fromstring(values_raw.decode("ascii"), sep=" ", dtype=np.float64)
            if len(values) != len(metadata):
                raise ValueError(f"Malformed row for {gene} in {matrix_path.name}")
            aggregated = np.bincount(codes, weights=values[positions], minlength=len(unique_groups))
            genes.append(gene)
            columns.append(aggregated)
            if max_genes is not None and len(genes) >= max_genes:
                break

    counts = np.vstack(columns).T
    library = counts.sum(axis=1, keepdims=True)
    normalized = np.log1p(np.divide(counts, library, out=np.zeros_like(counts), where=library > 0) * 10000.0)
    profiles = pd.DataFrame(normalized.astype(np.float32), columns=genes)
    records = []
    for index, (patient, sample, tissue, state) in enumerate(unique_groups):
        profile_id = f"GSE160269::{sample}::{state}"
        records.append(
            {
                "profile_id": profile_id,
                "source": "GSE160269",
                "patient": patient,
                "sample": sample,
                "tissue": tissue,
                "state": state,
                "n_cells": int(group_cell_counts[index]),
            }
        )
    return pd.DataFrame(records), profiles


def build_gse160269_reference(
    directory: Path,
    tisch_metadata_path: Path,
    state_field: str,
    max_genes: int | None,
    max_cells_per_compartment: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    crosswalk = _read_tisch_crosswalk(tisch_metadata_path, state_field)
    metadata_parts = []
    profile_parts = []
    for compartment in ("CD45neg", "CD45pos"):
        cell_path = directory / f"GSE160269_{compartment}_cells.txt.gz"
        matrix_path = directory / f"GSE160269_{compartment}_UMIs.txt.gz"
        metadata = _gse160269_cell_metadata(cell_path, crosswalk)
        group_meta, profiles = _stream_gse160269_matrix(
            matrix_path,
            metadata,
            max_genes=max_genes,
            max_cells=max_cells_per_compartment,
        )
        metadata_parts.append(group_meta)
        profile_parts.append(profiles)
    combined_meta = pd.concat(metadata_parts, ignore_index=True)
    # Never encode an unmeasured gene as a biological zero. Compartments are
    # harmonized only over genes measured in both matrices.
    combined_profiles = pd.concat(profile_parts, ignore_index=True, join="inner")
    combined_profiles.index = combined_meta["profile_id"]
    return combined_meta, combined_profiles


def build_gse154763_reference(
    directory: Path,
    max_genes: int | None,
    chunksize: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata_path = directory / "GSE154763_ESCA_metadata.csv.gz"
    expression_path = directory / "GSE154763_ESCA_normalized_expression.csv.gz"
    metadata = pd.read_csv(metadata_path, compression="gzip")
    metadata = metadata.rename(columns={"index": "cell"})
    metadata["state"] = metadata["MajorCluster"].astype(str).map(harmonize_cell_state)
    metadata["sample"] = metadata["library_id"].astype(str)
    metadata["tissue"] = metadata["tissue"].map({"N": "Normal", "T": "Tumor"}).fillna(metadata["tissue"])
    lookup = metadata.set_index("cell")[["patient", "sample", "tissue", "state"]]

    header = pd.read_csv(expression_path, compression="gzip", nrows=0).columns.tolist()
    genes = header[1 : None if max_genes is None else max_genes + 1]
    usecols = ["index", *genes]
    sums: dict[tuple[str, str, str, str], np.ndarray] = {}
    counts: dict[tuple[str, str, str, str], int] = {}
    observed_cells: list[str] = []
    for chunk in pd.read_csv(
        expression_path,
        compression="gzip",
        usecols=usecols,
        chunksize=int(chunksize),
    ):
        observed_cells.extend(chunk["index"].astype(str).tolist())
        joined = chunk.join(lookup, on="index", how="left", validate="many_to_one")
        if joined["state"].isna().any():
            raise ValueError("GSE154763 expression contains cells absent from metadata")
        for key, block in joined.groupby(["patient", "sample", "tissue", "state"], sort=False):
            vector = block[genes].to_numpy(dtype=np.float64).sum(axis=0)
            sums[key] = sums.get(key, np.zeros(len(genes), dtype=np.float64)) + vector
            counts[key] = counts.get(key, 0) + len(block)
    if observed_cells != metadata["cell"].astype(str).tolist():
        raise ValueError("GSE154763 expression rows do not match metadata order")

    records = []
    vectors = []
    for patient, sample, tissue, state in sorted(sums):
        profile_id = f"GSE154763::{sample}::{state}"
        records.append(
            {
                "profile_id": profile_id,
                "source": "GSE154763",
                "patient": patient,
                "sample": sample,
                "tissue": tissue,
                "state": state,
                "n_cells": counts[(patient, sample, tissue, state)],
            }
        )
        vectors.append(sums[(patient, sample, tissue, state)] / counts[(patient, sample, tissue, state)])
    group_meta = pd.DataFrame(records)
    profiles = pd.DataFrame(np.vstack(vectors).astype(np.float32), columns=genes, index=group_meta["profile_id"])
    return group_meta, profiles


def combine_references(parts: Iterable[tuple[pd.DataFrame, pd.DataFrame]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata_parts = []
    profile_parts = []
    for metadata, profiles in parts:
        metadata_parts.append(metadata)
        profile_parts.append(profiles)
    metadata = pd.concat(metadata_parts, ignore_index=True)
    # Platform integration is restricted to jointly measured genes. An outer
    # merge with zero-filling would manufacture study-specific marker signal.
    profiles = pd.concat(profile_parts, axis=0, join="inner")
    profiles = profiles.loc[metadata["profile_id"]]
    return metadata, profiles.astype(np.float32)

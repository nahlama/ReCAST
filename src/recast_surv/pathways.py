from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def read_gmt(paths: Iterable[Path]) -> dict[str, list[str]]:
    pathways: dict[str, list[str]] = {}
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 3:
                    continue
                name = fields[0]
                genes = list(dict.fromkeys(gene.strip() for gene in fields[2:] if gene.strip()))
                if name in pathways:
                    raise ValueError(f"Duplicated pathway name across GMT files: {name}")
                pathways[name] = genes
    return pathways


def safe_feature_name(prefix: str, value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    return f"{prefix}{cleaned}"


def select_pathways(
    pathways: dict[str, list[str]],
    available_genes: set[str],
    min_genes: int,
    max_genes: int,
    max_pathways: int | None,
) -> dict[str, list[str]]:
    candidates = []
    for order, (name, members) in enumerate(pathways.items()):
        overlap = [gene for gene in members if gene in available_genes]
        if min_genes <= len(overlap) <= max_genes:
            candidates.append((name, overlap, order))
    # Prefer well-covered pathways; retain source order as deterministic tie-break.
    candidates.sort(key=lambda item: (-len(item[1]), item[2]))
    if max_pathways is not None:
        candidates = candidates[: int(max_pathways)]
    return {name: genes for name, genes, _ in candidates}


def rank_pathway_scores(
    expression: pd.DataFrame,
    pathways: dict[str, list[str]],
) -> pd.DataFrame:
    if expression.empty:
        raise ValueError("Expression matrix is empty")
    ranked = expression.rank(axis=1, method="average", pct=True)
    scores = {}
    for name, genes in pathways.items():
        present = [gene for gene in genes if gene in ranked.columns]
        if not present:
            continue
        scores[safe_feature_name("pathway__", name)] = ranked[present].mean(axis=1) - 0.5
    return pd.DataFrame(scores, index=expression.index, dtype=np.float32)


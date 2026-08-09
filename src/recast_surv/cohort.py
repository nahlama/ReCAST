from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


def _missing(value: object) -> bool:
    return pd.isna(value) or str(value).strip().lower() in {
        "",
        "na",
        "nan",
        "not reported",
        "unknown",
    }


def normalize_histology(row: pd.Series) -> str:
    paper = str(row.get("paper_Histological.Type", "")).strip().upper()
    if paper == "ESCC":
        return "ESCC"
    if paper in {"AC", "EAC"}:
        return "EAC"
    diagnosis = str(row.get("primary_diagnosis", "")).lower()
    if "squamous" in diagnosis:
        return "ESCC"
    if "adenocarcinoma" in diagnosis:
        return "EAC"
    return "OTHER_OR_UNRESOLVED"


def normalize_stage(row: pd.Series) -> str:
    for field in ("paper_Pathologic.stage", "ajcc_pathologic_stage", "ajcc_clinical_stage"):
        value = row.get(field)
        if _missing(value):
            continue
        text = str(value).upper().replace("STAGE", "").strip()
        match = re.match(r"(IV|III|II|I)", text)
        if match:
            return match.group(1)
    return "UNKNOWN"


def build_tcga_cohort(
    metadata_path: Path,
    vst_path: Path,
    min_time_days: float = 1.0,
    include_histologies: tuple[str, ...] = ("ESCC", "EAC"),
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    metadata = pd.read_csv(metadata_path, low_memory=False)
    if "barcode" not in metadata or "shortLetterCode" not in metadata:
        raise ValueError("TCGA metadata lacks barcode or shortLetterCode")
    cohort = metadata.loc[metadata["shortLetterCode"].eq("TP")].copy()
    if cohort["patient"].duplicated().any():
        dup = cohort.loc[cohort["patient"].duplicated(False), "patient"].tolist()
        raise ValueError(f"Multiple primary tumours found for patients: {dup[:10]}")

    cohort["histology"] = cohort.apply(normalize_histology, axis=1)
    cohort = cohort.loc[cohort["histology"].isin(include_histologies)].copy()
    cohort["event"] = cohort["vital_status"].astype(str).str.lower().eq("dead").astype(int)

    death = pd.to_numeric(cohort["days_to_death"], errors="coerce")
    follow = pd.to_numeric(cohort["days_to_last_follow_up"], errors="coerce")
    cohort["time_days_raw"] = np.where(cohort["event"].eq(1), death, follow)
    if cohort["time_days_raw"].isna().any():
        missing = cohort.loc[cohort["time_days_raw"].isna(), "barcode"].tolist()
        raise ValueError(f"Missing overall-survival time for {len(missing)} samples")
    cohort["time_days"] = cohort["time_days_raw"].clip(lower=float(min_time_days))
    cohort["stage_group"] = cohort.apply(normalize_stage, axis=1)
    cohort["age_years"] = pd.to_numeric(cohort["age_at_diagnosis"], errors="coerce") / 365.25
    cohort["sex_male"] = cohort["gender"].astype(str).str.lower().eq("male").astype(int)
    cohort["sample_id"] = cohort["barcode"].astype(str)

    expression = pd.read_csv(vst_path, index_col=0)
    if expression.index.duplicated().any():
        expression = expression.groupby(level=0, sort=False).mean()
    missing_expr = sorted(set(cohort["sample_id"]) - set(expression.columns))
    if missing_expr:
        raise ValueError(f"Missing expression columns for {len(missing_expr)} cohort samples")
    expression = expression.loc[:, cohort["sample_id"]].T
    expression.index.name = "sample_id"
    expression = expression.astype(np.float32)

    keep = [
        "sample_id",
        "patient",
        "histology",
        "event",
        "time_days",
        "time_days_raw",
        "stage_group",
        "age_years",
        "sex_male",
        "race",
    ]
    cohort = cohort[keep].set_index("sample_id", drop=False)
    cohort = cohort.loc[expression.index]
    summary = {
        "samples": int(len(cohort)),
        "events": int(cohort["event"].sum()),
        "escc": int(cohort["histology"].eq("ESCC").sum()),
        "eac": int(cohort["histology"].eq("EAC").sum()),
        "times_clipped": int((cohort["time_days"] != cohort["time_days_raw"]).sum()),
        "genes": int(expression.shape[1]),
    }
    return cohort, expression, summary


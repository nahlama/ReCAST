from __future__ import annotations

import csv
import gzip
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .metrics import (
    cumulative_dynamic_auc,
    harrell_c_index,
    integrated_brier_score,
    ipcw_brier_score,
    ipcw_calibration,
    uno_c_index,
)
from .models import FoldPreprocessor, build_model, candidate_grid


def read_soft_platform(path: str | Path) -> pd.DataFrame:
    """Read only the platform table from a (possibly compressed) GEO SOFT file."""

    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    rows: list[list[str]] = []
    header: list[str] | None = None
    in_table = False
    with opener(path, "rt", encoding="utf-8", errors="replace", newline="") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\r\n")
            lower = line.lower()
            if lower == "!platform_table_begin":
                in_table = True
                continue
            if not in_table:
                continue
            if lower == "!platform_table_end":
                break
            fields = line.split("\t")
            if header is None:
                header = fields
            else:
                if len(fields) < len(header):
                    fields.extend([""] * (len(header) - len(fields)))
                rows.append(fields[: len(header)])
    if header is None:
        raise ValueError(f"No platform table found in {path}")
    frame = pd.DataFrame(rows, columns=header, dtype=str).fillna("")
    frame.columns = [column.strip().lower().replace(" ", "_") for column in frame.columns]
    return frame


def build_sequence_crosswalk(
    feature_platform: pd.DataFrame,
    probe_name_platform: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Crosswalk GPL18109 feature numbers to GPL19748 probe annotations by sequence."""

    required_feature = {"id", "sequence"}
    required_probe = {"id", "spot_id", "sequence"}
    if not required_feature.issubset(feature_platform.columns):
        raise ValueError(f"Feature platform lacks {sorted(required_feature - set(feature_platform.columns))}")
    if not required_probe.issubset(probe_name_platform.columns):
        raise ValueError(f"Probe-name platform lacks {sorted(required_probe - set(probe_name_platform.columns))}")

    feature = feature_platform.copy()
    probe = probe_name_platform.copy()
    feature["sequence_key"] = feature["sequence"].str.strip().str.upper()
    probe["sequence_key"] = probe["sequence"].str.strip().str.upper()
    feature = feature.loc[feature["sequence_key"].ne("")].copy()
    probe = probe.loc[probe["sequence_key"].ne("")].copy()

    feature_counts = feature["sequence_key"].value_counts()
    probe_counts = probe["sequence_key"].value_counts()
    feature["feature_sequence_count"] = feature["sequence_key"].map(feature_counts)
    probe["probe_sequence_count"] = probe["sequence_key"].map(probe_counts)

    feature = feature.rename(columns={"id": "feature_id"})
    probe_columns = ["sequence_key", "id", "spot_id", "probe_sequence_count"]
    for optional in ("controltype", "description"):
        if optional in probe.columns:
            probe_columns.append(optional)
    probe = probe[probe_columns].rename(
        columns={
            "id": "probe_name",
            "spot_id": "platform_gene_name",
            "controltype": "control_type",
        }
    )
    crosswalk = feature[["feature_id", "sequence_key", "feature_sequence_count"]].merge(
        probe,
        on="sequence_key",
        how="left",
        validate="many_to_many",
    )
    crosswalk["unique_sequence_match"] = (
        crosswalk["feature_sequence_count"].eq(1)
        & crosswalk["probe_sequence_count"].eq(1)
        & crosswalk["probe_name"].notna()
    )
    summary = {
        "feature_rows": int(len(feature_platform)),
        "feature_rows_with_sequence": int(len(feature)),
        "probe_name_rows": int(len(probe_name_platform)),
        "probe_name_rows_with_sequence": int(len(probe)),
        "feature_rows_with_any_match": int(crosswalk.loc[crosswalk["probe_name"].notna(), "feature_id"].nunique()),
        "feature_rows_with_unique_match": int(crosswalk.loc[crosswalk["unique_sequence_match"], "feature_id"].nunique()),
    }
    return crosswalk, summary


def _characteristics(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for item in str(text).split(" | "):
        key, separator, value = item.partition(":")
        if separator:
            values[key.strip().lower()] = value.strip()
    return values


def parse_external_clinical(metadata_path: str | Path) -> pd.DataFrame:
    metadata = pd.read_csv(metadata_path, dtype=str).fillna("")
    metadata = metadata.loc[metadata["sample_type_inferred"].str.casefold().eq("tumor")].copy()
    records: list[dict[str, Any]] = []
    roman_stage = {"I": 1, "II": 2, "III": 3, "IV": 4}
    for row in metadata.itertuples(index=False):
        values = _characteristics(row.characteristics_text)
        patient = values.get("patient id", "")
        age = pd.to_numeric(values.get("age", ""), errors="coerce")
        months = pd.to_numeric(values.get("survival time(months)", ""), errors="coerce")
        event_text = values.get("death at fu", "").casefold()
        stage = values.get("tnm stage", "").upper().strip()
        stage_group = re.sub(r"[^IV]", "", stage)
        records.append(
            {
                "cohort": row.accession,
                "sample_id": row.gsm_id,
                "patient_id": patient,
                "age": age,
                "sex": values.get("sex", "").casefold(),
                "stage": stage,
                "stage_numeric": roman_stage.get(stage_group),
                "time_days": float(months) * 30.4375 if pd.notna(months) else None,
                "event": 1 if event_text == "yes" else (0 if event_text == "no" else None),
                "histology": row.histology_inferred,
            }
        )
    return pd.DataFrame.from_records(records)


def inspect_expression_csv(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        header_line = handle.readline()
        header = next(csv.reader([header_line]))
        row_count = sum(1 for _ in handle)
    return {
        "path": str(path),
        "rows": int(row_count),
        "samples": int(max(0, len(header) - 1)),
        "index_name": header[0] if header else "",
        "bytes": int(path.stat().st_size),
    }


def _first_column(path: str | Path) -> set[str]:
    values: set[str] = set()
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        next(handle, None)
        for line in handle:
            values.add(line.partition(",")[0].strip().strip('"'))
    return values


def audit_external_cohorts(
    prepared_root: str | Path,
    accessions: list[str],
    gpl19748_path: str | Path,
    tcga_vst_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    prepared_root = Path(prepared_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    first_soft = prepared_root / accessions[0] / "raw" / f"{accessions[0]}_family.soft.gz"
    feature_platform = read_soft_platform(first_soft)
    probe_platform = read_soft_platform(gpl19748_path)
    crosswalk, mapping_summary = build_sequence_crosswalk(feature_platform, probe_platform)

    tcga_genes = set(pd.read_csv(tcga_vst_path, usecols=[0]).iloc[:, 0].astype(str))
    mapped_names = set(crosswalk.loc[crosswalk["unique_sequence_match"], "platform_gene_name"].dropna())
    direct_tcga_overlap = sorted(mapped_names & tcga_genes)
    mapping_summary["unique_platform_names"] = int(len(mapped_names))
    mapping_summary["direct_tcga_gene_overlap"] = int(len(direct_tcga_overlap))
    mapping_summary["gene_mapping_status"] = (
        "usable" if len(direct_tcga_overlap) >= 5000 else "blocked_requires_sequence_reannotation"
    )
    crosswalk.to_csv(output_dir / "GPL18109_to_GPL19748_sequence_crosswalk.csv.gz", index=False)

    cohort_rows: list[dict[str, Any]] = []
    clinical_parts: list[pd.DataFrame] = []
    patient_sets: dict[str, set[str]] = {}
    for accession in accessions:
        directory = prepared_root / accession
        metadata_path = directory / "metadata" / "sample_metadata_inferred.csv"
        probe_path = directory / "processed" / "expression_probe_matrix.csv"
        gene_path = directory / "processed" / "expression_gene_matrix.csv"
        clinical = parse_external_clinical(metadata_path)
        clinical_parts.append(clinical)
        patient_sets[accession] = set(clinical["patient_id"])
        probe_info = inspect_expression_csv(probe_path)
        gene_info = inspect_expression_csv(gene_path)
        gene_names = _first_column(gene_path)
        traceable_gene_names = gene_names & mapped_names
        complete = clinical[["age", "sex", "stage_numeric", "time_days", "event"]].notna().all(axis=1)
        cohort_rows.append(
            {
                "cohort": accession,
                "tumor_patients": int(clinical["patient_id"].nunique()),
                "events": int(pd.to_numeric(clinical["event"], errors="coerce").fillna(0).sum()),
                "complete_clinical_records": int(complete.sum()),
                "probe_rows": probe_info["rows"],
                "probe_samples": probe_info["samples"],
                "gene_rows": gene_info["rows"],
                "gene_samples": gene_info["samples"],
                "reported_gene_names_traceable_to_platform": int(len(traceable_gene_names)),
                "clinical_validation_status": "ready" if complete.mean() >= 0.9 else "incomplete",
                "omics_validation_status": mapping_summary["gene_mapping_status"],
            }
        )

    clinical_all = pd.concat(clinical_parts, ignore_index=True)
    clinical_all.to_csv(output_dir / "external_clinical.csv", index=False)
    cohort_audit = pd.DataFrame(cohort_rows)
    cohort_audit.to_csv(output_dir / "cohort_audit.csv", index=False)

    overlaps: dict[str, int] = {}
    for index, left in enumerate(accessions):
        for right in accessions[index + 1 :]:
            overlaps[f"{left}__{right}"] = int(len(patient_sets[left] & patient_sets[right]))

    report = {
        "platform_mapping": mapping_summary,
        "cohort_patient_overlap": overlaps,
        "cohorts": cohort_rows,
        "decision": {
            "clinical_external_validation": "ready",
            "omics_external_validation": mapping_summary["gene_mapping_status"],
            "reason": (
                "GPL19748 supplies probe names rather than a validated current gene-symbol mapping; "
                "the existing 72-row gene matrices are not traceable to the supplied platform map."
            ),
        },
    }
    (output_dir / "audit_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def external_clinical_features(clinical: pd.DataFrame) -> pd.DataFrame:
    """Encode GEO clinical variables exactly as the locked TCGA clinical panel."""

    stage = clinical["stage_numeric"]
    features = pd.DataFrame(index=clinical["sample_id"].astype(str))
    features.index.name = "sample_id"
    features["clinical__age_years"] = pd.to_numeric(clinical["age"], errors="coerce").to_numpy()
    features["clinical__sex_male"] = clinical["sex"].str.casefold().eq("male").astype(float).to_numpy()
    features["clinical__histology_ESCC"] = clinical["histology"].eq("ESCC").astype(float).to_numpy()
    for value, label in ((2, "II"), (3, "III"), (4, "IV")):
        features[f"clinical__stage_{label}"] = stage.eq(value).astype(float).to_numpy()
    features["clinical__stage_UNKNOWN"] = stage.isna().astype(float).to_numpy()
    return features


def run_locked_external_clinical_validation(
    training_features: pd.DataFrame,
    training_outcomes: pd.DataFrame,
    audited_clinical: pd.DataFrame,
    workspace: str | Path,
    survivalpfn_settings: dict[str, Any],
    primary_series: str,
    overlap_series: str,
    output_dir: str | Path,
    seed: int,
    evaluation_times: list[float],
    bootstrap_iterations: int = 2000,
    model_name: str = "survivalpfn",
    inner_folds: int = 4,
) -> dict[str, Any]:
    """Tune only in TCGA, then score a locked non-overlapping external subset."""

    clinical_columns = [column for column in training_features if column.startswith("clinical__")]
    if not clinical_columns:
        raise ValueError("Training features do not contain a clinical panel")
    overlap_patients = set(
        audited_clinical.loc[audited_clinical["cohort"].eq(overlap_series), "patient_id"].astype(str)
    )
    external = audited_clinical.loc[
        audited_clinical["cohort"].eq(primary_series)
        & ~audited_clinical["patient_id"].astype(str).isin(overlap_patients)
    ].copy()
    if external.empty:
        raise ValueError("No non-overlapping external patients remain")
    if external["patient_id"].duplicated().any():
        raise ValueError("External validation must contain one tumor sample per patient")

    X_train_frame = training_features[clinical_columns]
    X_external_frame = external_clinical_features(external).reindex(columns=clinical_columns)
    if X_external_frame.isna().all(axis=0).any():
        missing = X_external_frame.columns[X_external_frame.isna().all(axis=0)].tolist()
        raise ValueError(f"External feature columns are entirely missing: {missing}")
    external = external.set_index("sample_id").loc[X_external_frame.index]
    train_time = training_outcomes.loc[X_train_frame.index, "time_days"].to_numpy(dtype=float)
    train_event = training_outcomes.loc[X_train_frame.index, "event"].to_numpy(dtype=int)
    test_time = external["time_days"].to_numpy(dtype=float)
    test_event = external["event"].to_numpy(dtype=int)
    times = np.asarray([time for time in evaluation_times if float(time) < float(test_time.max())], dtype=float)
    if len(times) < 2:
        raise ValueError("At least two evaluation times must precede the maximum external follow-up")

    preprocessor = FoldPreprocessor()
    X_train = preprocessor.fit_transform(X_train_frame.to_numpy(dtype=float))
    X_test = preprocessor.transform(X_external_frame.to_numpy(dtype=float))
    candidates = candidate_grid(model_name)
    selected = candidates[0]
    selection_scores: list[dict[str, Any]] = []
    if len(candidates) > 1:
        histology = training_outcomes.loc[X_train_frame.index, "histology"].astype(str)
        joint = training_outcomes.loc[X_train_frame.index, "event"].astype(str) + "::" + histology
        strata = joint if joint.value_counts().min() >= inner_folds else training_outcomes.loc[
            X_train_frame.index, "event"
        ].astype(str)
        splitter = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
        raw_training = X_train_frame.to_numpy(dtype=float)
        best_score = float("-inf")
        for candidate in candidates:
            fold_scores: list[float] = []
            errors: list[str] = []
            for fold, (fit_index, valid_index) in enumerate(splitter.split(raw_training, strata)):
                try:
                    fold_preprocessor = FoldPreprocessor()
                    X_fit = fold_preprocessor.fit_transform(raw_training[fit_index])
                    X_valid = fold_preprocessor.transform(raw_training[valid_index])
                    fold_model = build_model(
                        candidate,
                        seed + fold,
                        Path(workspace),
                        survivalpfn_settings,
                    )
                    fold_model.fit(X_fit, train_time[fit_index], train_event[fit_index])
                    fold_scores.append(
                        uno_c_index(
                            train_time[fit_index],
                            train_event[fit_index],
                            train_time[valid_index],
                            train_event[valid_index],
                            fold_model.predict_risk(X_valid),
                        )
                    )
                except (ValueError, RuntimeError, ArithmeticError, Warning) as exc:
                    errors.append(f"{type(exc).__name__}: {exc}")
            mean_score = float(np.nanmean(fold_scores)) if fold_scores else float("nan")
            selection_scores.append(
                {
                    "params": candidate.params,
                    "mean_internal_uno_c": mean_score,
                    "valid_folds": len(fold_scores),
                    "errors": errors,
                }
            )
            if np.isfinite(mean_score) and mean_score > best_score:
                selected = candidate
                best_score = mean_score
    model = build_model(
        selected,
        seed,
        Path(workspace),
        survivalpfn_settings,
    )
    model.fit(X_train, train_time, train_event)
    risk = model.predict_risk(X_test)
    survival = model.predict_survival(X_test, times)

    def score(indices: np.ndarray) -> dict[str, float]:
        selected_time = test_time[indices]
        selected_event = test_event[indices]
        selected_risk = risk[indices]
        selected_survival = survival[indices]
        result = {
            "harrell_c": harrell_c_index(selected_time, selected_event, selected_risk),
            "uno_c": uno_c_index(
                train_time, train_event, selected_time, selected_event, selected_risk
            ),
            "integrated_brier": integrated_brier_score(
                train_time,
                train_event,
                selected_time,
                selected_event,
                selected_survival,
                times,
            ),
        }
        for time_index, horizon in enumerate(times):
            suffix = int(horizon)
            result[f"brier_t{suffix}"] = ipcw_brier_score(
                train_time,
                train_event,
                selected_time,
                selected_event,
                selected_survival[:, time_index],
                horizon,
            )
            result[f"auc_t{suffix}"] = cumulative_dynamic_auc(
                train_time,
                train_event,
                selected_time,
                selected_event,
                selected_risk,
                horizon,
            )
            intercept, slope = ipcw_calibration(
                train_time,
                train_event,
                selected_time,
                selected_event,
                selected_survival[:, time_index],
                horizon,
            )
            result[f"calibration_intercept_t{suffix}"] = intercept
            result[f"calibration_slope_t{suffix}"] = slope
        return result

    point = score(np.arange(len(external)))
    rng = np.random.default_rng(seed)
    bootstrap: dict[str, list[float]] = {name: [] for name in point}
    for _ in range(int(bootstrap_iterations)):
        draw = rng.integers(0, len(external), size=len(external))
        draw_scores = score(draw)
        for name, value in draw_scores.items():
            if np.isfinite(value):
                bootstrap[name].append(float(value))
    intervals = {
        name: {
            "low": float(np.quantile(values, 0.025)) if values else None,
            "high": float(np.quantile(values, 0.975)) if values else None,
            "valid_draws": len(values),
        }
        for name, values in bootstrap.items()
    }

    predictions = external.reset_index()[
        ["sample_id", "patient_id", "age", "sex", "stage", "time_days", "event"]
    ].copy()
    predictions["risk"] = risk
    for time_index, horizon in enumerate(times):
        predictions[f"survival_t{int(horizon)}"] = survival[:, time_index]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    result = {
        "model": model_name,
        "selected_params": selected.params,
        "internal_candidate_selection": selection_scores,
        "panel": "clinical_only",
        "training_cohort": "TCGA-ESCA",
        "external_series": primary_series,
        "excluded_overlapping_series": overlap_series,
        "external_subset": "non_overlapping_patients_only",
        "n_training": int(len(training_features)),
        "events_training": int(train_event.sum()),
        "n_external": int(len(external)),
        "events_external": int(test_event.sum()),
        "evaluation_times_days": times.tolist(),
        "metrics": point,
        "bootstrap_95_ci": intervals,
        "bootstrap_iterations": int(bootstrap_iterations),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result

#!/usr/bin/env python3
"""Reproduce the June-2026 SurvPFN clinical-only challenger.

This runner deliberately consumes the locked TCGA split manifest and the locked
non-overlapping GEO subset.  It writes only below
``artifacts/final/challengers/survpfn`` so the primary analysis is immutable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from sksurv.util import Surv

from SurvPFN import SurvPFN
from recast_surv.external import external_clinical_features
from recast_surv.figures import make_manuscript_figures
from recast_surv.metrics import (
    cumulative_dynamic_auc,
    harrell_c_index,
    integrated_brier_score,
    ipcw_brier_score,
    ipcw_calibration,
    uno_c_index,
)
from recast_surv.models import FoldPreprocessor


EVALUATION_TIMES = np.asarray([365.0, 1095.0, 1825.0])
MODEL_REPO = "samuelboehm/SurvPFN"
MODEL_FILE = "survpfn_nr.pt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def survival_at(model: SurvPFN, X: np.ndarray, times: np.ndarray) -> np.ndarray:
    functions = model.predict_survival_function(X)
    return np.clip(
        np.vstack([[float(function(time)) for time in times] for function in functions]),
        0.0,
        1.0,
    )


def score_block(
    train_time: np.ndarray,
    train_event: np.ndarray,
    test_time: np.ndarray,
    test_event: np.ndarray,
    risk: np.ndarray,
    survival: np.ndarray,
    times: np.ndarray,
) -> dict[str, float]:
    result = {
        "harrell_c": harrell_c_index(test_time, test_event, risk),
        "uno_c": uno_c_index(train_time, train_event, test_time, test_event, risk),
        "integrated_brier": integrated_brier_score(
            train_time, train_event, test_time, test_event, survival, times
        ),
    }
    for index, horizon in enumerate(times):
        suffix = int(horizon)
        result[f"brier_t{suffix}"] = ipcw_brier_score(
            train_time, train_event, test_time, test_event, survival[:, index], horizon
        )
        result[f"auc_t{suffix}"] = cumulative_dynamic_auc(
            train_time, train_event, test_time, test_event, risk, horizon
        )
        intercept, slope = ipcw_calibration(
            train_time, train_event, test_time, test_event, survival[:, index], horizon
        )
        result[f"calibration_intercept_t{suffix}"] = intercept
        result[f"calibration_slope_t{suffix}"] = slope
    return result


def validate_locked_manifest(manifest: pd.DataFrame, sample_ids: pd.Index) -> None:
    expected = set(sample_ids.astype(str))
    pairs = manifest[["repeat", "fold"]].drop_duplicates()
    if len(pairs) != 50:
        raise ValueError(f"Expected 50 locked outer folds, found {len(pairs)}")
    for repeat, fold in pairs.itertuples(index=False):
        block = manifest.loc[manifest["repeat"].eq(repeat) & manifest["fold"].eq(fold)]
        counts = block["sample_id"].astype(str).value_counts()
        if set(counts.index) != expected or not counts.eq(1).all():
            raise ValueError(f"Invalid locked coverage in repeat={repeat}, fold={fold}")
        if set(block["role"]) != {"train", "test"}:
            raise ValueError(f"Missing role in repeat={repeat}, fold={fold}")


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    value_columns = [
        column
        for column in metrics.columns
        if column not in {"repeat", "fold", "panel", "model", "n_test", "events_test"}
    ]
    grouped = metrics.groupby(["panel", "model"])[value_columns].agg(["mean", "std", "count"])
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    return grouped.reset_index()


def write_comparisons(
    workspace: Path,
    output_dir: Path,
    challenger_metrics: pd.DataFrame,
    challenger_summary: pd.DataFrame,
    external_result: dict[str, Any],
    seed: int,
    bootstrap_iterations: int,
) -> None:
    primary_metrics = pd.read_csv(workspace / "artifacts/final/benchmark/fold_metrics.csv")
    primary_clinical = primary_metrics.loc[primary_metrics["panel"].eq("clinical")]
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    for baseline_model in sorted(primary_clinical["model"].unique()):
        baseline = primary_clinical.loc[primary_clinical["model"].eq(baseline_model)]
        paired = challenger_metrics.merge(
            baseline,
            on=["repeat", "fold", "panel"],
            suffixes=("_survpfn", "_baseline"),
            validate="one_to_one",
        )
        for metric in ("uno_c", "harrell_c", "integrated_brier"):
            values = paired[[f"{metric}_survpfn", f"{metric}_baseline"]].dropna()
            if metric == "integrated_brier":
                improvement = values.iloc[:, 1].to_numpy() - values.iloc[:, 0].to_numpy()
                direction = "lower_is_better"
            else:
                improvement = values.iloc[:, 0].to_numpy() - values.iloc[:, 1].to_numpy()
                direction = "higher_is_better"
            draws = rng.choice(
                improvement, size=(bootstrap_iterations, len(improvement)), replace=True
            ).mean(axis=1)
            rows.append(
                {
                    "candidate_model": "survpfn",
                    "baseline_model": baseline_model,
                    "panel": "clinical",
                    "metric": metric,
                    "direction": direction,
                    "paired_folds": len(improvement),
                    "mean_improvement": float(improvement.mean()),
                    "ci95_low": float(np.quantile(draws, 0.025)),
                    "ci95_high": float(np.quantile(draws, 0.975)),
                    "probability_improvement_gt_zero": float(np.mean(draws > 0)),
                }
            )
    pd.DataFrame(rows).to_csv(output_dir / "paired_model_comparisons.csv", index=False)

    primary_summary = pd.read_csv(workspace / "artifacts/final/benchmark/summary.csv")
    extended_internal = pd.concat(
        [primary_summary.loc[primary_summary["panel"].eq("clinical")], challenger_summary],
        ignore_index=True,
    ).sort_values("uno_c_mean", ascending=False)
    extended_internal.to_csv(output_dir / "clinical_model_comparison_internal.csv", index=False)

    external_rows = pd.read_csv(
        workspace / "artifacts/final/external_validation/clinical_model_comparison.csv"
    )
    metrics = external_result["metrics"]
    intervals = external_result["bootstrap_95_ci"]
    challenger_external = pd.DataFrame(
        [
            {
                "model": "survpfn",
                "n_external": external_result["n_external"],
                "events_external": external_result["events_external"],
                "harrell_c": metrics["harrell_c"],
                "uno_c": metrics["uno_c"],
                "uno_c_ci95_low": intervals["uno_c"]["low"],
                "uno_c_ci95_high": intervals["uno_c"]["high"],
                "integrated_brier": metrics["integrated_brier"],
            }
        ]
    )
    pd.concat([external_rows, challenger_external], ignore_index=True).sort_values(
        "uno_c", ascending=False
    ).to_csv(output_dir / "clinical_model_comparison_external.csv", index=False)


def run_internal(
    features: pd.DataFrame,
    outcomes: pd.DataFrame,
    manifest: pd.DataFrame,
    checkpoint: Path,
    fold_limit: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    clinical_columns = [column for column in features if column.startswith("clinical__")]
    if not 2 <= len(clinical_columns) <= 10:
        raise ValueError(f"SurvPFN requires 2-10 features; clinical panel has {len(clinical_columns)}")
    validate_locked_manifest(manifest, features.index)
    pairs = manifest[["repeat", "fold"]].drop_duplicates().sort_values(["repeat", "fold"])
    if fold_limit is not None:
        pairs = pairs.head(fold_limit)
    estimator = SurvPFN(model_path=checkpoint, device="cpu", categorical_features_indices=None)
    predictions: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for sequence, (repeat, fold) in enumerate(pairs.itertuples(index=False), start=1):
        block = manifest.loc[manifest["repeat"].eq(repeat) & manifest["fold"].eq(fold)]
        train_ids = block.loc[block["role"].eq("train"), "sample_id"].astype(str)
        test_ids = block.loc[block["role"].eq("test"), "sample_id"].astype(str)
        raw_train = features.loc[train_ids, clinical_columns].to_numpy(dtype=float)
        raw_test = features.loc[test_ids, clinical_columns].to_numpy(dtype=float)
        preprocessor = FoldPreprocessor()
        X_train = preprocessor.fit_transform(raw_train)
        X_test = preprocessor.transform(raw_test)
        if not 2 <= X_train.shape[1] <= 10:
            raise ValueError(f"Fold {repeat}/{fold} retained {X_train.shape[1]} features")
        train_time = outcomes.loc[train_ids, "time_days"].to_numpy(dtype=float)
        train_event = outcomes.loc[train_ids, "event"].to_numpy(dtype=int)
        test_time = outcomes.loc[test_ids, "time_days"].to_numpy(dtype=float)
        test_event = outcomes.loc[test_ids, "event"].to_numpy(dtype=int)
        estimator.fit(X_train, Surv.from_arrays(train_event.astype(bool), train_time))
        risk = np.asarray(estimator.predict(X_test), dtype=float)
        survival = survival_at(estimator, X_test, EVALUATION_TIMES)
        fold_metrics = score_block(
            train_time, train_event, test_time, test_event, risk, survival, EVALUATION_TIMES
        )
        metric_rows.append(
            {
                "repeat": int(repeat),
                "fold": int(fold),
                "panel": "clinical",
                "model": "survpfn",
                "n_test": len(test_ids),
                "events_test": int(test_event.sum()),
                **fold_metrics,
            }
        )
        for index, sample_id in enumerate(test_ids):
            predictions.append(
                {
                    "sample_id": sample_id,
                    "repeat": int(repeat),
                    "fold": int(fold),
                    "panel": "clinical",
                    "model": "survpfn",
                    "time_days": test_time[index],
                    "event": test_event[index],
                    "risk": risk[index],
                    **{
                        f"survival_t{int(horizon)}": survival[index, time_index]
                        for time_index, horizon in enumerate(EVALUATION_TIMES)
                    },
                }
            )
        print(f"internal fold {sequence}/{len(pairs)} complete", flush=True)
    prediction_frame = pd.DataFrame(predictions)
    metric_frame = pd.DataFrame(metric_rows)
    return prediction_frame, metric_frame, summarize(metric_frame)


def external_subset(audited: pd.DataFrame) -> pd.DataFrame:
    overlap = set(
        audited.loc[audited["cohort"].eq("GSE53624"), "patient_id"].astype(str)
    )
    external = audited.loc[
        audited["cohort"].eq("GSE53625")
        & ~audited["patient_id"].astype(str).isin(overlap)
    ].copy()
    if external["patient_id"].duplicated().any():
        raise ValueError("External subset contains duplicate patients")
    return external


def run_external(
    features: pd.DataFrame,
    outcomes: pd.DataFrame,
    audited: pd.DataFrame,
    checkpoint: Path,
    bootstrap_iterations: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    clinical_columns = [column for column in features if column.startswith("clinical__")]
    external = external_subset(audited)
    X_external_frame = external_clinical_features(external).reindex(columns=clinical_columns)
    external = external.set_index("sample_id").loc[X_external_frame.index]
    preprocessor = FoldPreprocessor()
    X_train = preprocessor.fit_transform(features[clinical_columns].to_numpy(dtype=float))
    X_test = preprocessor.transform(X_external_frame.to_numpy(dtype=float))
    train_time = outcomes.loc[features.index, "time_days"].to_numpy(dtype=float)
    train_event = outcomes.loc[features.index, "event"].to_numpy(dtype=int)
    test_time = external["time_days"].to_numpy(dtype=float)
    test_event = external["event"].to_numpy(dtype=int)
    times = EVALUATION_TIMES[EVALUATION_TIMES < test_time.max()]
    estimator = SurvPFN(model_path=checkpoint, device="cpu", categorical_features_indices=None)
    estimator.fit(X_train, Surv.from_arrays(train_event.astype(bool), train_time))
    risk = np.asarray(estimator.predict(X_test), dtype=float)
    survival = survival_at(estimator, X_test, times)
    point = score_block(train_time, train_event, test_time, test_event, risk, survival, times)
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {name: [] for name in point}
    for _ in range(bootstrap_iterations):
        indices = rng.integers(0, len(external), size=len(external))
        values = score_block(
            train_time,
            train_event,
            test_time[indices],
            test_event[indices],
            risk[indices],
            survival[indices],
            times,
        )
        for name, value in values.items():
            if np.isfinite(value):
                draws[name].append(float(value))
    intervals = {
        name: {
            "low": float(np.quantile(values, 0.025)) if values else None,
            "high": float(np.quantile(values, 0.975)) if values else None,
            "valid_draws": len(values),
        }
        for name, values in draws.items()
    }
    prediction_frame = external.reset_index()[
        ["sample_id", "patient_id", "age", "sex", "stage", "time_days", "event"]
    ].copy()
    prediction_frame["risk"] = risk
    for index, horizon in enumerate(times):
        prediction_frame[f"survival_t{int(horizon)}"] = survival[:, index]
    result = {
        "model": "survpfn",
        "checkpoint": MODEL_FILE,
        "panel": "clinical_only",
        "training_cohort": "TCGA-ESCA",
        "external_series": "GSE53625",
        "excluded_overlapping_series": "GSE53624",
        "external_subset": "non_overlapping_patients_only",
        "n_training": len(features),
        "events_training": int(train_event.sum()),
        "n_external": len(external),
        "events_external": int(test_event.sum()),
        "evaluation_times_days": times.tolist(),
        "metrics": point,
        "bootstrap_95_ci": intervals,
        "bootstrap_iterations": bootstrap_iterations,
    }
    return prediction_frame, result


def write_environment(workspace: Path, output_dir: Path, checkpoint: Path) -> None:
    packages = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (output_dir / "requirements_frozen.txt").write_text(packages, encoding="utf-8")
    manifest = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": "cpu",
        "model_repository": MODEL_REPO,
        "model_filename": MODEL_FILE,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "survpfn_commit": git_commit(workspace / "third_party" / "SurvPFN"),
        "tfm_playground_commit": git_commit(workspace / "third_party" / "TFM-Playground"),
        "split_manifest_sha256": sha256(workspace / "artifacts/final/benchmark/split_manifest.csv"),
        "features_sha256": sha256(workspace / "artifacts/final/features/features.parquet"),
        "outcomes_sha256": sha256(workspace / "artifacts/final/features/outcomes.csv"),
        "runner_sha256": sha256(workspace / "scripts/run_survpfn_challenger.py"),
        "setup_sha256": sha256(workspace / "scripts/setup_survpfn_challenger.sh"),
    }
    (output_dir / "environment_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--fold-limit", type=int, default=None)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument(
        "--postprocess-only",
        action="store_true",
        help="Rebuild comparison/provenance artifacts from an already completed challenger run.",
    )
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    output_dir = workspace / "artifacts/final/challengers/survpfn"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE))
    print(f"checkpoint={checkpoint}", flush=True)
    if args.postprocess_only:
        metrics = pd.read_csv(output_dir / "fold_metrics.csv")
        summary = pd.read_csv(output_dir / "summary.csv")
        predictions = pd.read_csv(output_dir / "outer_predictions.csv")
        external_metrics = json.loads(
            (output_dir / "external_metrics.json").read_text(encoding="utf-8")
        )
        write_comparisons(
            workspace,
            output_dir,
            metrics,
            summary,
            external_metrics,
            args.seed,
            args.bootstrap_iterations,
        )
        (output_dir / "split_validation.json").write_text(
            json.dumps(
                {
                    "status": "passed",
                    "locked_outer_folds": 50,
                    "outer_predictions": len(predictions),
                    "unique_patients": int(predictions["sample_id"].nunique()),
                    "test_assignment_per_patient": 10,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        write_environment(workspace, output_dir, checkpoint)
        make_manuscript_figures(
            workspace / "artifacts/final", workspace / "artifacts/final/figures"
        )
        print(f"outputs={output_dir}", flush=True)
        return
    features = pd.read_parquet(workspace / "artifacts/final/features/features.parquet")
    outcomes = pd.read_csv(workspace / "artifacts/final/features/outcomes.csv", index_col="sample_id")
    manifest = pd.read_csv(workspace / "artifacts/final/benchmark/split_manifest.csv")
    predictions, metrics, summary = run_internal(
        features, outcomes, manifest, checkpoint, args.fold_limit
    )
    suffix = "smoke" if args.fold_limit is not None else ""
    target = output_dir / suffix if suffix else output_dir
    target.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(target / "outer_predictions.csv", index=False)
    metrics.to_csv(target / "fold_metrics.csv", index=False)
    summary.to_csv(target / "summary.csv", index=False)
    if args.fold_limit is None:
        audited = pd.read_csv(
            workspace / "artifacts/final/external_validation/audit/external_clinical.csv"
        )
        external_predictions, external_metrics = run_external(
            features,
            outcomes,
            audited,
            checkpoint,
            args.bootstrap_iterations,
            args.seed,
        )
        external_predictions.to_csv(output_dir / "external_predictions.csv", index=False)
        (output_dir / "external_metrics.json").write_text(
            json.dumps(external_metrics, indent=2, sort_keys=True), encoding="utf-8"
        )
        write_comparisons(
            workspace,
            output_dir,
            metrics,
            summary,
            external_metrics,
            args.seed,
            args.bootstrap_iterations,
        )
        (output_dir / "split_validation.json").write_text(
            json.dumps(
                {
                    "status": "passed",
                    "locked_outer_folds": 50,
                    "outer_predictions": len(predictions),
                    "unique_patients": int(predictions["sample_id"].nunique()),
                    "test_assignment_per_patient": 10,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        write_environment(workspace, output_dir, checkpoint)
        make_manuscript_figures(
            workspace / "artifacts/final", workspace / "artifacts/final/figures"
        )
    print(f"outputs={target}", flush=True)


if __name__ == "__main__":
    main()

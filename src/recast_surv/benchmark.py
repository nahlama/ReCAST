from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from .metrics import (
    cumulative_dynamic_auc,
    harrell_c_index,
    integrated_brier_score,
    ipcw_brier_score,
    ipcw_calibration,
    uno_c_index,
)
from .models import FoldPreprocessor, ModelCandidate, build_model, candidate_grid


def _strata(outcomes: pd.DataFrame) -> np.ndarray:
    joint = outcomes["event"].astype(str) + "::" + outcomes["histology"].astype(str)
    if joint.value_counts().min() >= 2:
        return joint.to_numpy()
    return outcomes["event"].astype(str).to_numpy()


def _feature_panels(features: pd.DataFrame) -> dict[str, list[str]]:
    clinical = [column for column in features if column.startswith("clinical__")]
    pathways = [column for column in features if column.startswith("pathway__")]
    recast = [column for column in features if column.startswith(("balance__", "recast__"))]
    panels = {
        "clinical": clinical,
        "clinical_plus_pathway": clinical + pathways,
        "clinical_plus_recast": clinical + recast,
        "full": clinical + pathways + recast,
    }
    return {name: list(dict.fromkeys(columns)) for name, columns in panels.items() if columns}


def _score_candidate(
    candidate: ModelCandidate,
    X: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    strata: np.ndarray,
    folds: int,
    seed: int,
    workspace: Path,
    survivalpfn_settings: dict[str, Any],
) -> float:
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = []
    for fold, (train_index, valid_index) in enumerate(splitter.split(X, strata)):
        preprocessor = FoldPreprocessor()
        X_train = preprocessor.fit_transform(X[train_index])
        X_valid = preprocessor.transform(X[valid_index])
        model = build_model(candidate, seed + fold, workspace, survivalpfn_settings)
        model.fit(X_train, time[train_index], event[train_index])
        risk = model.predict_risk(X_valid)
        scores.append(
            uno_c_index(
                time[train_index], event[train_index], time[valid_index], event[valid_index], risk
            )
        )
    finite = np.asarray(scores, dtype=float)
    return float(np.nanmean(finite)) if np.isfinite(finite).any() else float("nan")


def paired_fold_comparisons(
    metrics: pd.DataFrame,
    seed: int,
    n_bootstrap: int = 5000,
    baseline_panel: str = "clinical",
) -> pd.DataFrame:
    """Paired outer-fold uncertainty; positive improvement always favors candidate."""
    rng = np.random.default_rng(seed)
    metric_names = [
        column
        for column in metrics.columns
        if column in {"uno_c", "harrell_c", "integrated_brier"} or column.startswith("auc_t")
    ]
    rows: list[dict[str, Any]] = []
    keys = ["repeat", "fold"]
    for model_name in sorted(metrics["model"].unique()):
        model_metrics = metrics.loc[metrics["model"].eq(model_name)]
        baseline = model_metrics.loc[model_metrics["panel"].eq(baseline_panel), [*keys, *metric_names]]
        for panel in sorted(set(model_metrics["panel"]) - {baseline_panel}):
            candidate = model_metrics.loc[model_metrics["panel"].eq(panel), [*keys, *metric_names]]
            paired = candidate.merge(
                baseline,
                on=keys,
                suffixes=("_candidate", "_baseline"),
                validate="one_to_one",
            )
            for metric in metric_names:
                values = paired[[f"{metric}_candidate", f"{metric}_baseline"]].dropna()
                if values.empty:
                    continue
                if metric == "integrated_brier":
                    improvement = values.iloc[:, 1].to_numpy() - values.iloc[:, 0].to_numpy()
                    direction = "lower_is_better"
                else:
                    improvement = values.iloc[:, 0].to_numpy() - values.iloc[:, 1].to_numpy()
                    direction = "higher_is_better"
                draws = rng.choice(improvement, size=(int(n_bootstrap), len(improvement)), replace=True).mean(axis=1)
                rows.append(
                    {
                        "model": model_name,
                        "candidate_panel": panel,
                        "baseline_panel": baseline_panel,
                        "metric": metric,
                        "direction": direction,
                        "paired_folds": len(improvement),
                        "mean_improvement": float(np.mean(improvement)),
                        "ci95_low": float(np.quantile(draws, 0.025)),
                        "ci95_high": float(np.quantile(draws, 0.975)),
                        "probability_improvement_gt_zero": float(np.mean(draws > 0)),
                    }
                )
    return pd.DataFrame(rows)


def paired_representation_comparisons(
    metrics: pd.DataFrame,
    baseline_representation: str,
    seed: int,
    n_bootstrap: int = 5000,
) -> pd.DataFrame:
    """Compare a primary representation with ablations on identical outer folds."""
    rng = np.random.default_rng(seed)
    metric_names = [
        column
        for column in metrics.columns
        if column in {"uno_c", "harrell_c", "integrated_brier"} or column.startswith("auc_t")
    ]
    keys = ["repeat", "fold", "panel", "model"]
    baseline = metrics.loc[
        metrics["representation"].eq(baseline_representation), [*keys, *metric_names]
    ]
    rows: list[dict[str, Any]] = []
    for ablation in sorted(set(metrics["representation"]) - {baseline_representation}):
        candidate = metrics.loc[metrics["representation"].eq(ablation), [*keys, *metric_names]]
        paired = baseline.merge(
            candidate,
            on=keys,
            suffixes=("_primary", "_ablation"),
            validate="one_to_one",
        )
        for (panel, model_name), block in paired.groupby(["panel", "model"], sort=True):
            for metric in metric_names:
                values = block[[f"{metric}_primary", f"{metric}_ablation"]].dropna()
                if values.empty:
                    continue
                if metric == "integrated_brier":
                    improvement = values.iloc[:, 1].to_numpy() - values.iloc[:, 0].to_numpy()
                    direction = "lower_is_better"
                else:
                    improvement = values.iloc[:, 0].to_numpy() - values.iloc[:, 1].to_numpy()
                    direction = "higher_is_better"
                draws = rng.choice(improvement, size=(int(n_bootstrap), len(improvement)), replace=True).mean(axis=1)
                rows.append(
                    {
                        "primary_representation": baseline_representation,
                        "ablation_representation": ablation,
                        "panel": panel,
                        "model": model_name,
                        "metric": metric,
                        "direction": direction,
                        "paired_folds": len(improvement),
                        "mean_primary_improvement": float(np.mean(improvement)),
                        "ci95_low": float(np.quantile(draws, 0.025)),
                        "ci95_high": float(np.quantile(draws, 0.975)),
                        "probability_primary_better": float(np.mean(draws > 0)),
                    }
                )
    return pd.DataFrame(rows)


def validate_split_manifest(split_manifest: pd.DataFrame, sample_ids: pd.Index, repeats: int, folds: int) -> None:
    expected = set(sample_ids.astype(str))
    for repeat in range(repeats):
        repeat_block = split_manifest.loc[split_manifest["repeat"].eq(repeat)]
        test_counts = repeat_block.loc[repeat_block["role"].eq("test"), "sample_id"].astype(str).value_counts()
        if set(test_counts.index) != expected or not test_counts.eq(1).all():
            raise ValueError(f"Invalid test assignment in repeat {repeat}")
        for fold in range(folds):
            block = repeat_block.loc[repeat_block["fold"].eq(fold)]
            counts = block["sample_id"].astype(str).value_counts()
            if set(counts.index) != expected or not counts.eq(1).all():
                raise ValueError(f"Invalid split coverage in repeat {repeat}, fold {fold}")


def run_nested_benchmark(
    features: pd.DataFrame,
    outcomes: pd.DataFrame,
    settings: dict[str, Any],
    workspace: Path,
    output_dir: Path,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not features.index.equals(outcomes.index):
        outcomes = outcomes.loc[features.index]
    if outcomes["event"].nunique() < 2:
        raise ValueError("Both events and censored observations are required")
    panels = _feature_panels(features)
    if "clinical" not in panels or "full" not in panels:
        raise ValueError("Benchmark requires clinical and full feature panels")

    time = outcomes["time_days"].to_numpy(dtype=float)
    event = outcomes["event"].to_numpy(dtype=int)
    strata = _strata(outcomes)
    outer = RepeatedStratifiedKFold(
        n_splits=int(settings["outer_folds"]),
        n_repeats=int(settings["outer_repeats"]),
        random_state=seed,
    )
    prediction_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    models = list(settings["models"])
    survivalpfn_settings = settings.get("survivalpfn", {})
    evaluation_times = np.asarray(settings.get("time_grid_days", [365, 1095, 1825]), dtype=float)

    for split_number, (train_index, test_index) in enumerate(outer.split(features, strata)):
        repeat = split_number // int(settings["outer_folds"])
        fold = split_number % int(settings["outer_folds"])
        train_strata = strata[train_index]
        for role, indices in (("train", train_index), ("test", test_index)):
            split_rows.extend(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "sample_id": features.index[position],
                    "role": role,
                    "event": int(event[position]),
                    "histology": outcomes.iloc[position]["histology"],
                }
                for position in indices
            )
        for panel_name, columns in panels.items():
            X = features[columns].to_numpy(dtype=float)
            for model_name in models:
                candidates = candidate_grid(model_name)
                candidate_scores = []
                if len(candidates) == 1:
                    selected = candidates[0]
                    selection_rows.append(
                        {
                            "repeat": repeat,
                            "fold": fold,
                            "panel": panel_name,
                            "model": model_name,
                            "params": json.dumps(selected.params, sort_keys=True),
                            "inner_uno_c": float("nan"),
                            "error": "",
                            "selection_note": "single_candidate_no_tuning",
                        }
                    )
                else:
                    selected = None
                for candidate in candidates:
                    if selected is not None:
                        break
                    try:
                        score = _score_candidate(
                            candidate,
                            X[train_index],
                            time[train_index],
                            event[train_index],
                            train_strata,
                            int(settings["inner_folds"]),
                            seed + 10000 * repeat + 100 * fold,
                            workspace,
                            survivalpfn_settings,
                        )
                    except (ValueError, RuntimeError, ArithmeticError, Warning) as exc:
                        score = float("nan")
                        error = f"{type(exc).__name__}: {exc}"
                    else:
                        error = ""
                    candidate_scores.append((score, candidate, error))
                    selection_rows.append(
                        {
                            "repeat": repeat,
                            "fold": fold,
                            "panel": panel_name,
                            "model": model_name,
                            "params": json.dumps(candidate.params, sort_keys=True),
                            "inner_uno_c": score,
                            "error": error,
                            "selection_note": "nested_tuning",
                        }
                    )
                if selected is None:
                    valid = [item for item in candidate_scores if np.isfinite(item[0])]
                    if not valid:
                        continue
                    _, selected, _ = max(valid, key=lambda item: item[0])
                preprocessor = FoldPreprocessor()
                X_train = preprocessor.fit_transform(X[train_index])
                X_test = preprocessor.transform(X[test_index])
                model = build_model(
                    selected,
                    seed + 10000 * repeat + 100 * fold,
                    workspace,
                    survivalpfn_settings,
                )
                model.fit(X_train, time[train_index], event[train_index])
                risk = model.predict_risk(X_test)
                survival = model.predict_survival(X_test, evaluation_times)
                if survival.shape != (len(test_index), len(evaluation_times)):
                    raise ValueError(
                        f"{model_name} returned survival shape {survival.shape}; "
                        f"expected {(len(test_index), len(evaluation_times))}"
                    )
                if not np.isfinite(survival).all() or np.any((survival < 0) | (survival > 1)):
                    raise ValueError(f"{model_name} returned invalid survival probabilities")
                for position, sample_position in enumerate(test_index):
                    record = {
                        "sample_id": features.index[sample_position],
                        "repeat": repeat,
                        "fold": fold,
                        "panel": panel_name,
                        "model": model_name,
                        "time_days": time[sample_position],
                        "event": event[sample_position],
                        "risk": float(risk[position]),
                        "selected_params": json.dumps(selected.params, sort_keys=True),
                    }
                    record.update(
                        {
                            f"survival_t{int(horizon)}": float(survival[position, time_index])
                            for time_index, horizon in enumerate(evaluation_times)
                        }
                    )
                    prediction_rows.append(record)

    predictions = pd.DataFrame(prediction_rows)
    selections = pd.DataFrame(selection_rows)
    metric_rows = []
    for (repeat, fold, panel, model_name), block in predictions.groupby(
        ["repeat", "fold", "panel", "model"], sort=True
    ):
        # Censoring distribution is estimated from the actual outer training set.
        outer_train = outcomes.loc[~outcomes.index.isin(block["sample_id"])]
        train_time = outer_train["time_days"].to_numpy(dtype=float)
        train_event = outer_train["event"].to_numpy(dtype=int)
        test_time = block["time_days"].to_numpy(dtype=float)
        test_event = block["event"].to_numpy(dtype=int)
        risk = block["risk"].to_numpy(dtype=float)
        survival = block[[f"survival_t{int(item)}" for item in evaluation_times]].to_numpy(dtype=float)
        record = {
            "repeat": repeat,
            "fold": fold,
            "panel": panel,
            "model": model_name,
            "n_test": len(block),
            "events_test": int(block["event"].sum()),
            "harrell_c": harrell_c_index(test_time, test_event, risk),
            "uno_c": uno_c_index(train_time, train_event, test_time, test_event, risk),
            "integrated_brier": integrated_brier_score(
                train_time,
                train_event,
                test_time,
                test_event,
                survival,
                evaluation_times,
            ),
        }
        for time_index, horizon in enumerate(evaluation_times):
            calibration_intercept, calibration_slope = ipcw_calibration(
                train_time,
                train_event,
                test_time,
                test_event,
                survival[:, time_index],
                horizon,
            )
            suffix = int(horizon)
            record[f"brier_t{suffix}"] = ipcw_brier_score(
                train_time,
                train_event,
                test_time,
                test_event,
                survival[:, time_index],
                horizon,
            )
            record[f"auc_t{suffix}"] = cumulative_dynamic_auc(
                train_time,
                train_event,
                test_time,
                test_event,
                risk,
                horizon,
            )
            record[f"calibration_intercept_t{suffix}"] = calibration_intercept
            record[f"calibration_slope_t{suffix}"] = calibration_slope
        metric_rows.append(record)
    metrics = pd.DataFrame(metric_rows)
    split_manifest = pd.DataFrame(split_rows)
    validate_split_manifest(
        split_manifest,
        features.index,
        repeats=int(settings["outer_repeats"]),
        folds=int(settings["outer_folds"]),
    )
    comparisons = paired_fold_comparisons(metrics, seed=seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "outer_predictions.csv", index=False)
    selections.to_csv(output_dir / "inner_selection.csv", index=False)
    metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    split_manifest.to_csv(output_dir / "split_manifest.csv", index=False)
    comparisons.to_csv(output_dir / "paired_comparisons.csv", index=False)
    return predictions, selections, metrics

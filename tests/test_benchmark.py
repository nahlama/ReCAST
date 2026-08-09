import pandas as pd

from recast_surv.benchmark import (
    paired_fold_comparisons,
    paired_representation_comparisons,
    validate_split_manifest,
)


def test_split_manifest_validation_and_paired_improvement():
    samples = pd.Index(["a", "b", "c", "d"])
    rows = []
    for fold, test_ids in enumerate((["a", "b"], ["c", "d"])):
        for sample in samples:
            rows.append(
                {
                    "repeat": 0,
                    "fold": fold,
                    "sample_id": sample,
                    "role": "test" if sample in test_ids else "train",
                }
            )
    validate_split_manifest(pd.DataFrame(rows), samples, repeats=1, folds=2)

    metrics = pd.DataFrame(
        {
            "repeat": [0, 0, 0, 0],
            "fold": [0, 1, 0, 1],
            "panel": ["clinical", "clinical", "full", "full"],
            "model": ["m", "m", "m", "m"],
            "uno_c": [0.5, 0.6, 0.6, 0.7],
            "harrell_c": [0.5, 0.6, 0.6, 0.7],
            "integrated_brier": [0.25, 0.24, 0.20, 0.19],
        }
    )
    comparisons = paired_fold_comparisons(metrics, seed=1, n_bootstrap=100)
    assert (comparisons["mean_improvement"] > 0).all()
    assert comparisons["paired_folds"].eq(2).all()

    primary = metrics.copy()
    primary["representation"] = "primary"
    ablation = metrics.copy()
    ablation["representation"] = "ablation"
    ablation.loc[ablation["panel"].eq("full"), "uno_c"] -= 0.1
    representation = paired_representation_comparisons(
        pd.concat([primary, ablation]), baseline_representation="primary", seed=1, n_bootstrap=100
    )
    full_uno = representation.loc[
        representation["panel"].eq("full") & representation["metric"].eq("uno_c")
    ]
    assert full_uno["mean_primary_improvement"].iloc[0] > 0

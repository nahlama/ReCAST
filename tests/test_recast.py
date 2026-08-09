import numpy as np
import pandas as pd

from recast_surv.recast import ReCASTEstimator
from recast_surv.features import compact_recast_features


def test_recast_recovers_reference_state_direction():
    genes = [f"G{i}" for i in range(12)]
    metadata = pd.DataFrame(
        {
            "profile_id": ["a1", "a2", "b1", "b2"],
            "patient": ["p1", "p2", "p3", "p4"],
            "state": ["A", "A", "B", "B"],
        }
    )
    profiles = pd.DataFrame(
        [
            [10] * 6 + [1] * 6,
            [9] * 6 + [2] * 6,
            [1] * 6 + [10] * 6,
            [2] * 6 + [9] * 6,
        ],
        index=metadata["profile_id"],
        columns=genes,
        dtype=float,
    )
    bulk = pd.DataFrame(
        [[8] * 6 + [1] * 6, [1] * 6 + [8] * 6],
        index=["mostly_a", "mostly_b"],
        columns=genes,
    )
    estimator = ReCASTEstimator(
        backend="robust_nnls", markers_per_state=6, min_state_donors=2
    ).fit(
        metadata, profiles, set(genes)
    )
    scores = estimator.transform(bulk)
    assert scores.loc["mostly_a", "state__A"] > scores.loc["mostly_a", "state__B"]
    assert scores.loc["mostly_b", "state__B"] > scores.loc["mostly_b", "state__A"]
    assert np.all((scores["recast__unknown_score"] >= 0) & (scores["recast__unknown_score"] <= 1))


def test_unbalanced_transport_is_finite_and_mass_bounded():
    genes = [f"G{i}" for i in range(12)]
    metadata = pd.DataFrame(
        {
            "profile_id": ["a1", "a2", "b1", "b2"],
            "patient": ["p1", "p2", "p3", "p4"],
            "state": ["A", "A", "B", "B"],
        }
    )
    profiles = pd.DataFrame(
        [[10] * 6 + [1] * 6, [9] * 6 + [2] * 6, [1] * 6 + [10] * 6, [2] * 6 + [9] * 6],
        index=metadata["profile_id"],
        columns=genes,
        dtype=float,
    )
    estimator = ReCASTEstimator(
        backend="unbalanced_ot", markers_per_state=6, min_state_donors=2
    ).fit(metadata, profiles, set(genes))
    scores = estimator.transform(profiles.iloc[:2])
    mass = scores[["state__A", "state__B", "recast__unknown_score"]].sum(axis=1)
    assert np.allclose(mass, 1.0)
    assert np.isfinite(scores.to_numpy()).all()
    assert (scores["recast__solver_residual"] < 1e-5).all()
    compact = compact_recast_features(scores)
    assert "balance__myeloid_vs_lymphoid" in compact
    assert "recast__matched_mass" in compact
    assert np.isfinite(compact.to_numpy()).all()

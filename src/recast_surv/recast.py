from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import rankdata

from .pathways import safe_feature_name


@dataclass
class ReCASTDiagnostics:
    backend: str
    states: list[str]
    marker_genes: list[str]
    donors_per_state: dict[str, int]
    protein_prior_genes: int
    unknown_reference_baseline: float

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "states": self.states,
            "marker_genes": self.marker_genes,
            "donors_per_state": self.donors_per_state,
            "protein_prior_genes": self.protein_prior_genes,
            "unknown_reference_baseline": self.unknown_reference_baseline,
        }


class ReCASTEstimator:
    """Robust reference-state scoring with an explicit unmatched residual.

    This implementation is deliberately described as robust decomposition, not
    as exact optimal transport. It estimates non-negative reference-state
    scores from platform-robust within-sample ranks and reports unexplained
    expression as a normalized ``unknown_score``.
    """

    def __init__(
        self,
        backend: str = "unbalanced_ot",
        markers_per_state: int = 40,
        min_state_donors: int = 2,
        robust_iterations: int = 5,
        huber_delta: float = 1.5,
        transport_epsilon: float = 0.15,
        mass_penalty: float = 0.8,
        target_mass_penalty: float = 0.0,
        transport_iterations: int = 500,
        transport_tolerance: float = 1e-7,
        protein_prior_strength: float = 0.25,
    ) -> None:
        if backend not in {"unbalanced_ot", "robust_nnls"}:
            raise ValueError(f"Unknown ReCAST backend: {backend}")
        self.backend = backend
        self.markers_per_state = int(markers_per_state)
        self.min_state_donors = int(min_state_donors)
        self.robust_iterations = int(robust_iterations)
        self.huber_delta = float(huber_delta)
        self.transport_epsilon = float(transport_epsilon)
        self.mass_penalty = float(mass_penalty)
        self.target_mass_penalty = float(target_mass_penalty)
        self.transport_iterations = int(transport_iterations)
        self.transport_tolerance = float(transport_tolerance)
        self.protein_prior_strength = float(protein_prior_strength)
        self.states_: list[str] = []
        self.marker_genes_: list[str] = []
        self.signature_: np.ndarray | None = None
        self.transport_kernel_: np.ndarray | None = None
        self.generative_signature_: pd.DataFrame | None = None
        self.marker_weights_: np.ndarray | None = None
        self.protein_prior_genes_ = 0
        self.unknown_reference_baseline_ = 0.0
        self.donors_per_state_: dict[str, int] = {}

    @staticmethod
    def _rank_vector(values: np.ndarray) -> np.ndarray:
        ranked = rankdata(values, method="average").astype(np.float64)
        ranked /= max(len(ranked), 1)
        return ranked

    def fit(
        self,
        reference_metadata: pd.DataFrame,
        reference_profiles: pd.DataFrame,
        available_bulk_genes: set[str],
        gene_reliability: dict[str, float] | None = None,
    ) -> "ReCASTEstimator":
        metadata = reference_metadata.set_index("profile_id").loc[reference_profiles.index]
        donor_counts = metadata.groupby("state")["patient"].nunique()
        states = donor_counts.loc[donor_counts >= self.min_state_donors].index.astype(str).tolist()
        if len(states) < 2:
            raise ValueError("At least two reference states pass the donor-count threshold")
        common = [gene for gene in reference_profiles.columns if gene in available_bulk_genes]
        if len(common) < max(10, len(states)):
            raise ValueError(f"Only {len(common)} genes overlap between reference and bulk expression")

        selected = metadata["state"].isin(states)
        donor_state_profiles = reference_profiles.loc[selected, common].copy()
        donor_state_profiles["_state"] = metadata.loc[selected, "state"].astype(str).values
        donor_state_profiles["_patient"] = metadata.loc[selected, "patient"].astype(str).values
        # Collapse within donor before forming state anchors so a donor with
        # several samples or fine subclusters cannot dominate a state median.
        donor_state_profiles = donor_state_profiles.groupby(
            ["_state", "_patient"], sort=True
        ).median()
        state_profiles = donor_state_profiles.groupby(level="_state", sort=True).median()
        states = state_profiles.index.tolist()
        ranked = pd.DataFrame(
            np.vstack([self._rank_vector(row) for row in state_profiles.to_numpy(dtype=float)]),
            index=states,
            columns=common,
        )
        reliability = pd.Series(gene_reliability or {}, dtype=float)
        reliability = reliability.reindex(common).fillna(1.0).clip(lower=0.25, upper=1.25)
        markers: list[str] = []
        for state in states:
            others = ranked.drop(index=state).median(axis=0)
            contrast = ((ranked.loc[state] - others) * reliability.pow(self.protein_prior_strength)).sort_values(
                ascending=False
            )
            positive = contrast.loc[contrast > 0].head(self.markers_per_state).index.tolist()
            markers.extend(positive)
        marker_genes = list(dict.fromkeys(markers))
        if len(marker_genes) < len(states):
            raise ValueError("Reference marker selection yielded too few genes")

        marker_weights = reliability.loc[marker_genes].pow(self.protein_prior_strength).to_numpy(dtype=np.float64)
        self.generative_signature_ = ranked.loc[states, marker_genes].T.copy()
        design = ranked.loc[states, marker_genes].T.to_numpy(dtype=np.float64) * marker_weights[:, None]
        norms = np.linalg.norm(design, axis=0, keepdims=True)
        design = np.divide(design, norms, out=np.zeros_like(design), where=norms > 0)
        self.states_ = states
        self.marker_genes_ = marker_genes
        self.signature_ = design
        self.marker_weights_ = marker_weights
        self.protein_prior_genes_ = int(np.sum(marker_weights != 1.0))
        # Gene-to-state cost uses relative state specificity, not absolute
        # platform intensity. This makes the transport geometry invariant to a
        # multiplicative change in any marker gene.
        specificity = np.divide(
            design,
            design.sum(axis=1, keepdims=True),
            out=np.full_like(design, 1.0 / len(states)),
            where=design.sum(axis=1, keepdims=True) > 0,
        )
        cost = -np.log(np.clip(specificity, 1e-12, None))
        self.transport_kernel_ = np.exp(-cost / self.transport_epsilon)
        self.donors_per_state_ = {state: int(donor_counts[state]) for state in states}
        scorer = self._score_unbalanced_ot if self.backend == "unbalanced_ot" else self._score_nnls
        reference_unknown = [
            scorer(row)[1]
            for row in reference_profiles.loc[selected, marker_genes].to_numpy(dtype=float)
        ]
        self.unknown_reference_baseline_ = float(np.median(reference_unknown))
        return self

    def _score_nnls(self, values: np.ndarray) -> tuple[np.ndarray, float, float, float]:
        if self.signature_ is None:
            raise ValueError("Estimator has not been fitted")
        x = self._rank_vector(values)
        if self.marker_weights_ is not None:
            x = x * self.marker_weights_
        x_norm = np.linalg.norm(x)
        if x_norm <= 0:
            return np.zeros(len(self.states_)), 1.0, 0.0, 0.0
        x = x / x_norm
        row_weights = np.ones_like(x)
        coefficients = np.zeros(len(self.states_))
        for _ in range(max(1, self.robust_iterations)):
            weighted_signature = self.signature_ * np.sqrt(row_weights)[:, None]
            weighted_x = x * np.sqrt(row_weights)
            coefficients, _ = nnls(weighted_signature, weighted_x)
            residual = x - self.signature_ @ coefficients
            scale = np.median(np.abs(residual - np.median(residual))) * 1.4826 + 1e-8
            standardized = np.abs(residual) / scale
            row_weights = np.where(
                standardized <= self.huber_delta,
                1.0,
                self.huber_delta / np.maximum(standardized, 1e-8),
            )
        prediction = self.signature_ @ coefficients
        residual = x - prediction
        raw_unknown = float(
            np.clip(np.linalg.norm(residual) / (np.linalg.norm(x) + 1e-8), 0.0, 1.0)
        )
        unknown_score = raw_unknown
        cosine = float(np.dot(x, prediction) / ((np.linalg.norm(x) * np.linalg.norm(prediction)) + 1e-8))
        if coefficients.sum() > 0:
            coefficients = coefficients / coefficients.sum() * (1.0 - unknown_score)
        return coefficients, unknown_score, cosine, 0.0

    def _score_unbalanced_ot(self, values: np.ndarray) -> tuple[np.ndarray, float, float, float]:
        if self.signature_ is None or self.transport_kernel_ is None:
            raise ValueError("Estimator has not been fitted")
        source = self._rank_vector(values)
        if self.marker_weights_ is not None:
            source = source * self.marker_weights_
        source = np.maximum(source, 1e-12)
        source /= source.sum()
        target = np.full(len(self.states_), 1.0 / len(self.states_), dtype=np.float64)
        kernel = self.transport_kernel_
        source_exponent = self.mass_penalty / (self.mass_penalty + self.transport_epsilon)
        target_exponent = self.target_mass_penalty / (
            self.target_mass_penalty + self.transport_epsilon
        )
        u = np.ones(len(source), dtype=np.float64)
        v = np.ones(len(target), dtype=np.float64)
        residual = float("inf")
        for _ in range(max(1, self.transport_iterations)):
            previous_u = u.copy()
            previous_v = v.copy()
            u = np.power(source / np.maximum(kernel @ v, 1e-300), source_exponent)
            if self.target_mass_penalty > 0:
                v = np.power(
                    target / np.maximum(kernel.T @ u, 1e-300), target_exponent
                )
            else:
                # Semi-relaxed transport: target/state masses are learned from
                # the sample rather than shrunk toward an arbitrary uniform mix.
                v.fill(1.0)
            residual = max(
                float(np.max(np.abs(u - previous_u) / np.maximum(np.abs(previous_u), 1e-12))),
                float(np.max(np.abs(v - previous_v) / np.maximum(np.abs(previous_v), 1e-12))),
            )
            if residual <= self.transport_tolerance:
                break
        plan = u[:, None] * kernel * v[None, :]
        state_mass = plan.sum(axis=0)
        if state_mass.sum() > 0:
            state_mass = state_mass / state_mass.sum()
        prediction = self.signature_ @ state_mass
        ranked = self._rank_vector(values)
        if self.marker_weights_ is not None:
            ranked = ranked * self.marker_weights_
        ranked = ranked / (np.linalg.norm(ranked) + 1e-12)
        raw_unknown = float(
            np.clip(
                np.linalg.norm(ranked - prediction) / (np.linalg.norm(ranked) + 1e-8),
                0.0,
                1.0,
            )
        )
        unknown_score = raw_unknown
        cosine = float(
            np.dot(ranked, prediction)
            / ((np.linalg.norm(ranked) * np.linalg.norm(prediction)) + 1e-8)
        )
        state_mass = state_mass * (1.0 - unknown_score)
        return state_mass, unknown_score, cosine, residual

    def transform(self, bulk_expression: pd.DataFrame) -> pd.DataFrame:
        missing = sorted(set(self.marker_genes_) - set(bulk_expression.columns))
        if missing:
            raise ValueError(f"Bulk expression is missing {len(missing)} fitted marker genes")
        rows = []
        for _, sample in bulk_expression[self.marker_genes_].iterrows():
            scorer = self._score_unbalanced_ot if self.backend == "unbalanced_ot" else self._score_nnls
            coefficients, unknown_score, cosine, residual = scorer(sample.to_numpy(dtype=float))
            record = {
                safe_feature_name("state__", state): float(value)
                for state, value in zip(self.states_, coefficients)
            }
            record["recast__unknown_score"] = unknown_score
            record["recast__reconstruction_cosine"] = cosine
            if self.backend == "unbalanced_ot":
                record["recast__solver_residual"] = residual
            rows.append(record)
        return pd.DataFrame(rows, index=bulk_expression.index, dtype=np.float32)

    def diagnostics(self) -> ReCASTDiagnostics:
        return ReCASTDiagnostics(
            backend=self.backend,
            states=list(self.states_),
            marker_genes=list(self.marker_genes_),
            donors_per_state=dict(self.donors_per_state_),
            protein_prior_genes=self.protein_prior_genes_,
            unknown_reference_baseline=self.unknown_reference_baseline_,
        )

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceWarning
from scipy.linalg import LinAlgWarning
from scipy.stats import norm
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


class FoldPreprocessor:
    """Median imputation, constant filtering and scaling fitted per training fold."""

    def __init__(self) -> None:
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.keep_: np.ndarray | None = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        values = self.imputer.fit_transform(np.asarray(X, dtype=float))
        self.keep_ = np.nanstd(values, axis=0) > 1e-10
        if not np.any(self.keep_):
            raise ValueError("No non-constant features remain in this training fold")
        return self.scaler.fit_transform(values[:, self.keep_]).astype(np.float32)

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.keep_ is None:
            raise ValueError("Preprocessor is not fitted")
        values = self.imputer.transform(np.asarray(X, dtype=float))
        return self.scaler.transform(values[:, self.keep_]).astype(np.float32)


class RiskModel:
    def fit(self, X: np.ndarray, time: np.ndarray, event: np.ndarray) -> "RiskModel":
        raise NotImplementedError

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict_survival(self, X: np.ndarray, times: np.ndarray) -> np.ndarray:
        """Return P(T > t | X) with shape (samples, times)."""
        raise NotImplementedError


class ElasticNetCox(RiskModel):
    def __init__(self, penalizer: float, l1_ratio: float) -> None:
        self.penalizer = float(penalizer)
        self.l1_ratio = float(l1_ratio)
        self.model: CoxPHFitter | None = None
        self.columns: list[str] = []

    def fit(self, X: np.ndarray, time: np.ndarray, event: np.ndarray) -> "ElasticNetCox":
        self.columns = [f"x{i}" for i in range(X.shape[1])]
        frame = pd.DataFrame(X, columns=self.columns)
        frame["time"] = np.asarray(time, dtype=float)
        frame["event"] = np.asarray(event, dtype=int)
        self.model = CoxPHFitter(penalizer=self.penalizer, l1_ratio=self.l1_ratio)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            warnings.simplefilter("error", LinAlgWarning)
            self.model.fit(frame, duration_col="time", event_col="event", show_progress=False)
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model has not been fitted")
        frame = pd.DataFrame(X, columns=self.columns)
        return np.log(np.asarray(self.model.predict_partial_hazard(frame), dtype=float).reshape(-1) + 1e-12)

    def predict_survival(self, X: np.ndarray, times: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model has not been fitted")
        frame = pd.DataFrame(X, columns=self.columns)
        curves = self.model.predict_survival_function(frame, times=np.asarray(times, dtype=float))
        return np.asarray(curves, dtype=float).T


class XGBoostAFT(RiskModel):
    def __init__(self, params: dict[str, Any], seed: int) -> None:
        self.params = dict(params)
        self.seed = int(seed)
        self.model: Any = None
        self.distribution_scale = float(self.params.get("aft_loss_distribution_scale", 1.0))

    def fit(self, X: np.ndarray, time: np.ndarray, event: np.ndarray) -> "XGBoostAFT":
        import xgboost as xgb

        matrix = xgb.DMatrix(X)
        lower = np.asarray(time, dtype=float)
        upper = np.where(np.asarray(event, dtype=bool), lower, np.inf)
        matrix.set_float_info("label_lower_bound", lower)
        matrix.set_float_info("label_upper_bound", upper)
        settings = {
            "objective": "survival:aft",
            "eval_metric": "aft-nloglik",
            "aft_loss_distribution": "normal",
            "aft_loss_distribution_scale": 1.0,
            "tree_method": "hist",
            "nthread": 1,
            "seed": self.seed,
            **self.params,
        }
        rounds = int(settings.pop("num_boost_round", 200))
        self.model = xgb.train(settings, matrix, num_boost_round=rounds, verbose_eval=False)
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        import xgboost as xgb

        if self.model is None:
            raise ValueError("Model has not been fitted")
        predicted_time = np.asarray(self.model.predict(xgb.DMatrix(X)), dtype=float)
        return -np.log(np.maximum(predicted_time, 1e-8))

    def predict_survival(self, X: np.ndarray, times: np.ndarray) -> np.ndarray:
        import xgboost as xgb

        if self.model is None:
            raise ValueError("Model has not been fitted")
        location = np.log(np.maximum(self.model.predict(xgb.DMatrix(X)), 1e-8))
        log_times = np.log(np.maximum(np.asarray(times, dtype=float), 1e-8))
        standardized = (log_times[None, :] - location[:, None]) / self.distribution_scale
        return np.clip(norm.sf(standardized), 0.0, 1.0)


class RandomSurvivalForestModel(RiskModel):
    def __init__(self, params: dict[str, Any], seed: int) -> None:
        self.params = dict(params)
        self.seed = int(seed)
        self.model: Any = None

    def fit(self, X: np.ndarray, time: np.ndarray, event: np.ndarray) -> "RandomSurvivalForestModel":
        try:
            from sksurv.ensemble import RandomSurvivalForest
        except ImportError as exc:
            raise RuntimeError("random_survival_forest requires the optional scikit-survival dependency") from exc
        outcome = np.empty(len(time), dtype=[("event", "?"), ("time", "<f8")])
        outcome["event"] = np.asarray(event, dtype=bool)
        outcome["time"] = np.asarray(time, dtype=float)
        self.model = RandomSurvivalForest(random_state=self.seed, n_jobs=1, **self.params)
        self.model.fit(X, outcome)
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model has not been fitted")
        return np.asarray(self.model.predict(X), dtype=float)

    def predict_survival(self, X: np.ndarray, times: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model has not been fitted")
        functions = self.model.predict_survival_function(X, return_array=False)
        return np.vstack([[float(function(time)) for time in times] for function in functions])


class SurvivalPFNModel(RiskModel):
    def __init__(self, settings: dict[str, Any], workspace: Path) -> None:
        self.settings = dict(settings)
        self.workspace = workspace
        self.model: Any = None

    def fit(self, X: np.ndarray, time: np.ndarray, event: np.ndarray) -> "SurvivalPFNModel":
        vendor = (self.workspace / self.settings.get("vendor_path", "third_party/SurvivalPFN")).resolve()
        if str(vendor) not in sys.path:
            sys.path.insert(0, str(vendor))
        try:
            from survivalpfn import SurvivalEstimator
        except ImportError as exc:
            raise RuntimeError("SurvivalPFN could not be imported from the configured vendor path") from exc
        self.model = SurvivalEstimator(
            device=self.settings.get("device", "cpu"),
            model_path=self.settings.get("model_path", "shi-ang/SurvivalPFN"),
        )
        self.model.fit(X=X, T=np.asarray(time, dtype=np.float32), delta=np.asarray(event, dtype=np.float32))
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model has not been fitted")
        # A lower predicted median event time denotes higher risk.
        predicted_time = np.asarray(self.model.predict_event_time(X, type="median"), dtype=float)
        return -predicted_time

    def predict_survival(self, X: np.ndarray, times: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model has not been fitted")
        return np.clip(
            np.asarray(self.model.S(X, np.asarray(times, dtype=np.float32)), dtype=float),
            0.0,
            1.0,
        )


@dataclass(frozen=True)
class ModelCandidate:
    name: str
    params: dict[str, Any]


def candidate_grid(model_name: str) -> list[ModelCandidate]:
    if model_name == "elastic_net_cox":
        elastic = [
            ModelCandidate(model_name, {"penalizer": penalty, "l1_ratio": ratio})
            for penalty in (0.01, 0.1, 1.0)
            for ratio in (0.0, 0.5)
        ]
        # Very weak pure-lasso penalties are numerically unstable in lifelines;
        # retain lasso only at regularization strengths that are identifiable.
        return elastic + [
            ModelCandidate(model_name, {"penalizer": penalty, "l1_ratio": 1.0})
            for penalty in (0.1, 1.0)
        ]
    if model_name == "xgb_aft":
        return [
            ModelCandidate(
                model_name,
                {
                    "eta": eta,
                    "max_depth": depth,
                    "min_child_weight": 5,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "lambda": 2.0,
                    "num_boost_round": 150,
                },
            )
            for eta in (0.03, 0.1)
            for depth in (1, 2)
        ]
    if model_name == "random_survival_forest":
        return [
            ModelCandidate(model_name, {"n_estimators": 500, "min_samples_leaf": leaf, "max_features": feature})
            for leaf in (5, 10)
            for feature in ("sqrt", 0.5)
        ]
    if model_name == "survivalpfn":
        return [ModelCandidate(model_name, {})]
    raise ValueError(f"Unknown model: {model_name}")


def build_model(
    candidate: ModelCandidate,
    seed: int,
    workspace: Path,
    survivalpfn_settings: dict[str, Any],
) -> RiskModel:
    if candidate.name == "elastic_net_cox":
        return ElasticNetCox(**candidate.params)
    if candidate.name == "xgb_aft":
        return XGBoostAFT(candidate.params, seed)
    if candidate.name == "random_survival_forest":
        return RandomSurvivalForestModel(candidate.params, seed)
    if candidate.name == "survivalpfn":
        return SurvivalPFNModel(survivalpfn_settings, workspace)
    raise ValueError(candidate.name)

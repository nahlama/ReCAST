import numpy as np

from recast_surv.models import (
    ElasticNetCox,
    FoldPreprocessor,
    RandomSurvivalForestModel,
    XGBoostAFT,
)


def _synthetic_survival(seed=7):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(48, 4))
    event_time = np.exp(2.0 - X[:, 0] + rng.normal(scale=0.25, size=len(X)))
    censor_time = rng.exponential(12.0, size=len(X))
    return X, np.minimum(event_time, censor_time), (event_time <= censor_time).astype(int)


def test_fold_preprocessor_drops_constant_columns():
    X = np.array([[1.0, 2.0], [1.0, np.nan], [1.0, 4.0]])
    processor = FoldPreprocessor()
    transformed = processor.fit_transform(X)
    assert transformed.shape == (3, 1)
    assert np.isfinite(transformed).all()


def test_survival_models_fit_and_predict_finite_risk():
    X, time, event = _synthetic_survival()
    cox = ElasticNetCox(penalizer=0.1, l1_ratio=0.5).fit(X, time, event)
    aft = XGBoostAFT({"max_depth": 1, "eta": 0.1, "num_boost_round": 5}, seed=1).fit(
        X, time, event
    )
    assert np.isfinite(cox.predict_risk(X[:4])).all()
    assert np.isfinite(aft.predict_risk(X[:4])).all()
    times = np.array([2.0, 5.0, 10.0])
    for model in (cox, aft):
        survival = model.predict_survival(X[:4], times)
        assert survival.shape == (4, 3)
        assert np.isfinite(survival).all()
        assert np.all((survival >= 0) & (survival <= 1))
        assert np.all(np.diff(survival, axis=1) <= 1e-8)


def test_random_survival_forest_adapter():
    X, time, event = _synthetic_survival()
    model = RandomSurvivalForestModel(
        {"n_estimators": 10, "min_samples_leaf": 3, "max_features": "sqrt"}, seed=1
    ).fit(X, time, event)
    survival = model.predict_survival(X[:4], np.array([2.0, 5.0, 10.0]))
    assert np.isfinite(model.predict_risk(X[:4])).all()
    assert survival.shape == (4, 3)
    assert np.all((survival >= 0) & (survival <= 1))

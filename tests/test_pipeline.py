import numpy as np
import pandas as pd

from pipeline import FEATURES, build_final_model, make_features


def test_negative_weights_become_positive(synthetic_loads):
    X = make_features(synthetic_loads)
    assert (X["weight"].dropna() >= 0).all()


def test_missing_values_are_kept_for_the_imputer(synthetic_loads):
    X = make_features(synthetic_loads)
    assert X["weight"].isna().sum() == 10
    assert X["market_index"].isna().sum() == 10


def test_features_have_expected_columns_and_values(synthetic_loads):
    X = make_features(synthetic_loads)
    assert list(X.columns) == FEATURES
    assert np.allclose(X["log_distance"], np.log(synthetic_loads["distance"]))
    first_weekday = pd.to_datetime(synthetic_loads["date"].iloc[0]).dayofweek
    assert X["dow"].iloc[0] == first_weekday


def test_make_features_does_not_modify_input(synthetic_loads):
    before = synthetic_loads.copy()
    make_features(synthetic_loads)
    pd.testing.assert_frame_equal(synthetic_loads, before)


def test_model_predicts_positive_finite_rates(synthetic_loads):
    X = make_features(synthetic_loads)
    model = build_final_model().fit(X, synthetic_loads["posted_rate"])
    pred = model.predict(X)
    assert len(pred) == len(X)
    assert np.isfinite(pred).all() and (pred > 0).all()
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 23

FEATURES = ["distance", "log_distance", "weight", "market_index", "quote_signal",
            "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "dow", "equipment"]
NUM = ["distance", "log_distance", "weight", "market_index", "quote_signal",
       "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]
CAT = ["equipment", "dow"]

# Best settings found by the Optuna search (see notebooks/02_modeling.ipynb)
BEST_HGB_PARAMS = {
    "loss": "absolute_error",
    "learning_rate": 0.15968975616292932,
    "max_iter": 274,
    "max_leaf_nodes": 26,
    "min_samples_leaf": 98,
    "l2_regularization": 0.034086932716089424,
}
BEST_KNN_K = 11


def make_features(frame):
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["weight"] = out["weight"].abs()  # negative weights are sign errors
    out["log_distance"] = np.log(out["distance"])
    out["dow"] = out["date"].dt.dayofweek
    return out[FEATURES]


def make_preprocessor(k=BEST_KNN_K):
    encode_scale = ColumnTransformer(
        [("num", StandardScaler(), NUM),
         ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT)]
    )
    return Pipeline([("encode_scale", encode_scale),
                     ("knn_impute", KNNImputer(n_neighbors=k))])


def build_final_model():
    regressor = HistGradientBoostingRegressor(random_state=SEED, **BEST_HGB_PARAMS)
    pipe = Pipeline([("prep", make_preprocessor()), ("model", regressor)])
    return TransformedTargetRegressor(regressor=pipe, func=np.log, inverse_func=np.exp)
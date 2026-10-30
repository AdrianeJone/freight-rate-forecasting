import joblib
import pytest
from fastapi.testclient import TestClient

import api
from pipeline import build_final_model, make_features

VALID_LOAD = {
    "pickup_lat": 36.99152,
    "pickup_lon": -84.99876,
    "delivery_lat": 41.31561,
    "delivery_lon": -85.36206,
    "distance": 360,
    "equipment": "Dry Van",
    "weight": 32000,
    "date": "2025-12-03",
    "market_index": 0.994063,
    "quote_signal": 2.053293,
}


@pytest.fixture(scope="module")
def client(synthetic_loads, tmp_path_factory):
    model_path = tmp_path_factory.mktemp("model") / "model.joblib"
    model = build_final_model().fit(make_features(synthetic_loads), synthetic_loads["posted_rate"])
    joblib.dump(model, model_path)

    patch = pytest.MonkeyPatch()
    patch.setattr(api, "MODEL_PATH", model_path)
    with TestClient(api.app) as test_client:  # runs the startup that loads the model
        yield test_client
    patch.undo()


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_predict_returns_positive_rate(client):
    response = client.post("/predict", json=VALID_LOAD)
    assert response.status_code == 200
    assert response.json()["predicted_rate"] > 0


def test_predict_without_optional_fields(client):
    load = {k: v for k, v in VALID_LOAD.items() if k not in ("weight", "market_index")}
    response = client.post("/predict", json=load)
    assert response.status_code == 200
    assert response.json()["predicted_rate"] > 0


@pytest.mark.parametrize("field, bad_value", [
    ("equipment", "Container"),
    ("distance", 0),
    ("date", "not-a-date"),
    ("pickup_lat", 120),
])
def test_invalid_values_are_rejected(client, field, bad_value):
    response = client.post("/predict", json={**VALID_LOAD, field: bad_value})
    assert response.status_code == 422


def test_missing_required_field_is_rejected(client):
    load = {k: v for k, v in VALID_LOAD.items() if k != "quote_signal"}
    response = client.post("/predict", json=load)
    assert response.status_code == 422
from main import app
import httpx
from fastapi.testclient import TestClient


client = TestClient(app)


def test_start_page():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"msg": "Welcome to the Projet 7 API !"}


def test_get_group():
    response = client.get("/group/?feature=FLAG_PHONE")
    assert response.status_code == 200
    response_json = response.json()
    assert "feature" in response_json
    assert "y_pred" in response_json


def test_get_feat_imp():
    response = client.get("/feat_imp/")
    assert response.status_code == 200
    response_json = response.json()
    assert "features_importance" in response_json


def test_get_predict():
    id_to_test = 0

    response = client.get(f"/customer/?id={id_to_test}")
    assert response.status_code == 200

    response_json = response.json()
    assert "prediction" in response_json
    assert "probability" in response_json
    assert "infos" in response_json
from fastapi.testclient import TestClient
import pytest
from main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

#client = TestClient(app)


def test_api_locally_get_root(client):
    """
    Test root endpoint with get request
    """
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {
        "endpoints": "Available endpoint: POST: /predict - prediction, \
        GET: /docs - documentation"
    }


def test_api_locally_post_predict_above50k(client):
    """
    Test predict endpoint with the post request
    """
    json_body = {
        "age": 40,
        "workclass": "Private",
        "fnlgt": 54545,
        "education": "Doctorate",
        "educationNum": 16,
        "maritalStatus": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capitalGain": 0,
        "capitalLoss": 0,
        "hoursPerWeek": 40,
        "nativeCountry": "United-States",
    }
    r = client.post("/predict", json=json_body)
    assert r.status_code == 200
    assert r.json() == {"salary": ">50K"}


def test_api_locally_post_predict_below50k(client):
    """
    Test predict endpoint with the post request
    """
    json_body = {
        "age": 42,
        "workclass": "Private",
        "fnlgt": 124692,
        "education": "HS-grad",
        "educationNum": 9,
        "maritalStatus": "Married-civ-spouse",
        "occupation": "Handlers-cleaners",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capitalGain": 0,
        "capitalLoss": 0,
        "hoursPerWeek": 40,
        "nativeCountry": "United-States",
    }
    r = client.post("/predict", json=json_body)
    assert r.status_code == 200
    assert r.json() == {"salary": "<=50K"}


def test_api_locally_post_predict_missing_fields(client):

    """
    Test predict endpoint with missing field in the request body
    Missing field: "workclass"
    """
    json_body = {
        "age": 40,
        "education": "Doctorate",
        "educationNum": 16,
        "maritalStatus": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capitalGain": 0,
        "capitalLoss": 0,
        "hoursPerWeek": 40,
        "nativeCountry": "United-States",
    }
    r = client.post("/predict", json=json_body)
    assert r.status_code == 422
    assert r.json()["detail"][0]["loc"] == ["body", "workclass"]
    assert r.json()["detail"][0]["msg"] == "field required"


if __name__ == "__main__":
    test_api_locally_post_predict_above50k(client)
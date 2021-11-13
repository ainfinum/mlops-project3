from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_api_locally_get_root():
    """
    Test root endpoint with get request
    """
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {
        "endpoints": "Available endpoint: POST: /predict - prediction, \
        GET: /docs - documentation"
    }


def test_api_locally_post_predict():
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


def test_api_locally_post_predict_missing_fields():

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

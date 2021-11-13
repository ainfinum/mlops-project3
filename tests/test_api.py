from fastapi.testclient import TestClient

# Import our app from main.py.
from main import app

client = TestClient(app)

def test_api_locally_get_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"endpoints": "Available endpoints: /predict"}
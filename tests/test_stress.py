import pytest

@pytest.fixture
def auth_token(client):
    client.post(
        "/register",
        json={"email": "stress_test@example.com", "password": "Password123"}
    )
    response = client.post(
        "/login",
        json={"email": "stress_test@example.com", "password": "Password123"}
    )
    return response.json()["access_token"]

valid_stress_payload = {
    "sdnn": 50.0, "median_rr": 800.0, "cv_rr": 0.05, "rmssd": 30.0,
    "pnn50": 15.0, "pnn20": 45.0, "mean_hr": 75.0, "std_hr": 5.0,
    "min_hr": 60.0, "max_hr": 90.0, "hr_range": 30.0
}

def test_stress_prediction_unauthorized(client):
    response = client.post("/stress-predict", json=valid_stress_payload)
    assert response.status_code == 401

def test_stress_prediction_authorized(client, auth_token):
    response = client.post(
        "/stress-predict",
        headers={"Authorization": f"Bearer {auth_token}"},
        json=valid_stress_payload
    )
    assert response.status_code in [200, 503]
    if response.status_code == 200:
        data = response.json()
        assert "stress_level_pct" in data

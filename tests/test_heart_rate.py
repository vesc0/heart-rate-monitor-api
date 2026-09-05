import pytest

@pytest.fixture
def auth_token(client):
    client.post(
        "/register",
        json={"email": "hr_test@example.com", "password": "Password123"}
    )
    response = client.post(
        "/login",
        json={"email": "hr_test@example.com", "password": "Password123"}
    )
    return response.json()["access_token"]

def test_record_heart_rate(client, auth_token):
    response = client.post(
        "/heart-rate",
        headers={"Authorization": f"Bearer {auth_token}"},
        json={
            "bpm": 85,
            "recorded_at": "2023-10-27T10:00:00Z",
            "activity_state": "resting"
        }
    )
    assert response.status_code == 201
    data = response.json()
    assert data["bpm"] == 85
    assert data["activity_state"] == "resting"

def test_get_heart_rate_records(client, auth_token):
    # Record first
    client.post(
        "/heart-rate",
        headers={"Authorization": f"Bearer {auth_token}"},
        json={
            "bpm": 70,
            "recorded_at": "2023-10-27T10:05:00Z"
        }
    )
    
    response = client.get(
        "/heart-rate",
        headers={"Authorization": f"Bearer {auth_token}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 1
    assert data[0]["bpm"] in [85, 70]

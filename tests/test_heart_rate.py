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

def test_stress_explanation_round_trips(client, auth_token):
    """The app stores the LLM explanation server-side, so a refresh must return it."""
    headers = {"Authorization": f"Bearer {auth_token}"}
    entry = {
        "id": "3f6c1b3e-6f4e-4a7c-9f2a-2b8d4e5a1c90",
        "bpm": 78,
        "recorded_at": "2023-10-27T10:10:00Z",
        "stress_level": 42,
        "stress_explanation": "Your heart rate stayed steady and variability was healthy.",
    }
    assert client.post("/heart-rate", headers=headers, json=entry).status_code == 201

    listed = client.get("/heart-rate", headers=headers).json()
    stored = next(item for item in listed if item["id"] == entry["id"])
    assert stored["stress_level"] == 42
    assert stored["stress_explanation"] == entry["stress_explanation"]

    # Re-posting the same id upserts rather than duplicating.
    entry["stress_explanation"] = "Updated explanation."
    assert client.post("/heart-rate", headers=headers, json=entry).status_code == 201
    listed = client.get("/heart-rate", headers=headers).json()
    matches = [item for item in listed if item["id"] == entry["id"]]
    assert len(matches) == 1 and matches[0]["stress_explanation"] == "Updated explanation."

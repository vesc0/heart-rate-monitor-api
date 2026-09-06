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

def test_explanation_unavailable_returns_503(client, auth_token, monkeypatch):
    """A model SHAP cannot explain must degrade to 503, not crash with a 500."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from services.stress_service import stress_model

    columns = stress_model.feature_columns
    features = [[valid_stress_payload.get(column, 0.0) for column in columns]]
    # Any non-tree model: TreeExplainer cannot explain it, but predict still works.
    linear = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
    linear.fit(features * 2, [0, 1])

    monkeypatch.setattr(stress_model, "_model", linear)
    monkeypatch.setattr(stress_model, "_explainer", None)

    response = client.post(
        "/stress-predict/explain",
        headers={"Authorization": f"Bearer {auth_token}"},
        json=valid_stress_payload,
    )
    assert response.status_code == 503
    assert "LogisticRegression" in response.json()["detail"]


def test_explanation_normalized_for_display():
    """Typographic spaces and hyphens from the LLM must not reach the app."""
    from services.llm_explainer import StressExplanationLLM

    raw = "First\u202fline.  \n\n\n\nSecond\u00a0camera\u2011based line.  "
    assert StressExplanationLLM._normalize(raw) == (
        "First line.\n\nSecond camera-based line."
    )

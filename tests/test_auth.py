def test_register_user(client):
    response = client.post(
        "/register",
        json={"email": "test@example.com", "password": "Password123"}
    )
    assert response.status_code == 201
    data = response.json()
    assert data["email"] == "test@example.com"
    assert "access_token" in data

def test_login_user(client):
    # Register first
    client.post(
        "/register",
        json={"email": "test2@example.com", "password": "Password123"}
    )
    
    # Login
    response = client.post(
        "/login",
        json={"email": "test2@example.com", "password": "Password123"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"

def test_login_invalid_password(client):
    # Register first
    client.post(
        "/register",
        json={"email": "test3@example.com", "password": "Password123"}
    )
    
    # Login with wrong password
    response = client.post(
        "/login",
        json={"email": "test3@example.com", "password": "WrongPassword123"}
    )
    assert response.status_code == 401

def test_login_rate_limited_per_client(client, monkeypatch):
    from core.limiter import limiter

    monkeypatch.setattr(limiter, "enabled", True)
    statuses = [
        client.post(
            "/login",
            # The proxy reports a new source port per connection; the IP is the key.
            headers={"X-Forwarded-For": f"203.0.113.7:{50000 + attempt}"},
            json={"email": "nobody@example.com", "password": "WrongPassword123"},
        ).status_code
        for attempt in range(11)
    ]
    assert statuses == [401] * 10 + [429]

def test_logout_revokes_token(client):
    credentials = {"email": "logout@example.com", "password": "Password123"}
    token = client.post("/register", json=credentials).json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    assert client.post("/logout", headers=headers).status_code == 200
    assert client.get("/me", headers=headers).status_code == 401

    fresh = client.post("/login", json=credentials).json()["access_token"]
    assert client.get("/me", headers={"Authorization": f"Bearer {fresh}"}).status_code == 200

def test_token_issued_before_versioning_still_works(client):
    from datetime import datetime, timedelta, timezone
    from jose import jwt
    from core.config import settings

    token = client.post(
        "/register", json={"email": "legacy@example.com", "password": "Password123"}
    ).json()["access_token"]
    claims = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
    legacy = jwt.encode(
        {"sub": claims["sub"], "exp": datetime.now(timezone.utc) + timedelta(minutes=5)},
        settings.SECRET_KEY,
        algorithm=settings.ALGORITHM,
    )
    assert client.get("/me", headers={"Authorization": f"Bearer {legacy}"}).status_code == 200

def test_delete_account(client):
    token = client.post(
        "/register",
        json={"email": "delete_me@example.com", "password": "Password123"}
    ).json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    client.post("/heart-rate", headers=headers, json={"bpm": 70, "recorded_at": "2023-10-27T10:00:00Z"})

    assert client.delete("/me", headers=headers).status_code == 204
    # The token has not expired, but its account is gone.
    assert client.get("/me", headers=headers).status_code == 401
    assert client.post("/stress-predict", headers=headers, json={}).status_code == 401

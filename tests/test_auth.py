import pytest

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

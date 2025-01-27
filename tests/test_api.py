import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to LLM Agent API"}

def test_query():
    response = client.post("/query", json={"question": "What is AI?"})
    assert response.status_code == 200
    assert "answer" in response.json()

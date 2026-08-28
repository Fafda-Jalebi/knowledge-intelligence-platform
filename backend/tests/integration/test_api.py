"""Integration tests for the full API."""

from datetime import datetime

import pytest
from httpx import AsyncClient, ASGITransport
from kip.api import app


@pytest.fixture
async def client(test_db):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


@pytest.fixture
async def auth_token(client: AsyncClient):
    # Register a user
    resp = await client.post("/api/auth/register", json={
        "email": "test@example.com",
        "password": "StrongPass-2024!Secure"
    })
    assert resp.status_code == 201
    return resp.json()["access_token"]


@pytest.fixture
async def auth_headers(auth_token: str):
    return {"Authorization": f"Bearer {auth_token}"}


class TestAuthFlow:
    async def test_register_and_login(self, client: AsyncClient):
        # Register
        resp = await client.post("/api/auth/register", json={
            "email": "newuser@example.com",
            "password": "AnotherStrong-2024!Pass"
        })
        assert resp.status_code == 201
        data = resp.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

        # Login with same credentials
        resp = await client.post("/api/auth/login", json={
            "email": "newuser@example.com",
            "password": "AnotherStrong-2024!Pass"
        })
        assert resp.status_code == 200
        assert "access_token" in resp.json()

    async def test_duplicate_registration_fails(self, client: AsyncClient):
        await client.post("/api/auth/register", json={
            "email": "dup@example.com",
            "password": "StrongPass-2024!Secure"
        })
        resp = await client.post("/api/auth/register", json={
            "email": "dup@example.com",
            "password": "StrongPass-2024!Secure"
        })
        assert resp.status_code == 400

    async def test_wrong_password_fails(self, client: AsyncClient, auth_headers):
        resp = await client.post("/api/auth/login", json={
            "email": "test@example.com",
            "password": "WrongPassword123"
        })
        assert resp.status_code == 401

    async def test_get_me(self, client: AsyncClient, auth_headers):
        resp = await client.get("/api/auth/me", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["email"] == "test@example.com"
        assert "id" in data

    async def test_change_password(self, client: AsyncClient, auth_headers):
        resp = await client.post("/api/auth/change-password", headers=auth_headers, json={
            "current_password": "StrongPass-2024!Secure",
            "new_password": "NewStrong-2024!Pass"
        })
        assert resp.status_code == 200

        # Verify new password works
        resp = await client.post("/api/auth/login", json={
            "email": "test@example.com",
            "password": "NewStrong-2024!Pass"
        })
        assert resp.status_code == 200

    async def test_change_password_wrong_current_password_returns_400(self, client: AsyncClient, auth_headers):
        resp = await client.post("/api/auth/change-password", headers=auth_headers, json={
            "current_password": "WrongPassword123!",
            "new_password": "NewStrong-2024!Pass"
        })
        assert resp.status_code == 400


class TestDocumentsFlow:
    async def test_upload_document(self, client: AsyncClient, auth_headers):
        # Create a simple text file
        content = b"This is a test document about mango drying at 60 degrees."
        files = {"file": ("test.txt", content, "text/plain")}

        resp = await client.post("/api/documents/upload", headers=auth_headers, files=files)
        assert resp.status_code == 201
        data = resp.json()
        assert "id" in data
        assert data["filename"] == "test.txt"
        assert data["status"] == "ready"
        assert data["created_at"]
        datetime.fromisoformat(data["created_at"])

    async def test_list_documents(self, client: AsyncClient, auth_headers):
        resp = await client.get("/api/documents", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "documents" in data
        assert "total" in data

    async def test_get_document(self, client: AsyncClient, auth_headers):
        # First upload
        content = b"Test document content."
        files = {"file": ("get_test.txt", content, "text/plain")}
        upload_resp = await client.post("/api/documents/upload", headers=auth_headers, files=files)
        doc_id = upload_resp.json()["id"]

        # Get document
        resp = await client.get(f"/api/documents/{doc_id}", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == doc_id

    async def test_get_document_chunks(self, client: AsyncClient, auth_headers):
        content = b"Chunk listing should return document text for the owner."
        files = {"file": ("chunks.txt", content, "text/plain")}
        upload_resp = await client.post("/api/documents/upload", headers=auth_headers, files=files)
        doc_id = upload_resp.json()["id"]

        resp = await client.get(f"/api/documents/{doc_id}/chunks", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] > 0
        assert data["chunks"][0]["body"]

    async def test_document_chunks_are_scoped_to_owner(self, client: AsyncClient, auth_headers):
        content = b"Private owner-only chunk text."
        files = {"file": ("private.txt", content, "text/plain")}
        upload_resp = await client.post("/api/documents/upload", headers=auth_headers, files=files)
        doc_id = upload_resp.json()["id"]

        other_resp = await client.post("/api/auth/register", json={
            "email": "other@example.com",
            "password": "OtherStrong-2024!Pass"
        })
        assert other_resp.status_code == 201
        other_headers = {"Authorization": f"Bearer {other_resp.json()['access_token']}"}

        resp = await client.get(f"/api/documents/{doc_id}/chunks", headers=other_headers)
        assert resp.status_code == 404

    async def test_delete_document(self, client: AsyncClient, auth_headers):
        # Upload
        content = b"To be deleted."
        files = {"file": ("delete_me.txt", content, "text/plain")}
        upload_resp = await client.post("/api/documents/upload", headers=auth_headers, files=files)
        doc_id = upload_resp.json()["id"]

        # Delete
        resp = await client.delete(f"/api/documents/{doc_id}", headers=auth_headers)
        assert resp.status_code == 200

        # Verify gone
        resp = await client.get(f"/api/documents/{doc_id}", headers=auth_headers)
        assert resp.status_code == 404

    async def test_rejects_executable(self, client: AsyncClient, auth_headers):
        files = {"file": ("evil.exe", b"MZ\x90\x00", "application/octet-stream")}
        resp = await client.post("/api/documents/upload", headers=auth_headers, files=files)
        assert resp.status_code == 415  # UnsupportedMediaTypeError


class TestChatFlow:
    async def test_ask_question(self, client: AsyncClient, auth_headers):
        # First upload a document
        content = b"Mango slices are dried at 60 C for eight hours. Water activity below 0.6 prevents mold."
        files = {"file": ("mango.txt", content, "text/plain")}
        await client.post("/api/documents/upload", headers=auth_headers, files=files)

        # Ask a question
        resp = await client.post("/api/chat/ask", headers=auth_headers, json={
            "question": "At what temperature are mango slices dried?"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert "60" in data["answer"]
        assert not data["refused"]
        assert len(data["citations"]) > 0

    async def test_refuses_unanswerable(self, client: AsyncClient, auth_headers):
        resp = await client.post("/api/chat/ask", headers=auth_headers, json={
            "question": "What is the tensile strength of steel?"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["refused"]
        # Empty corpus -> no passages retrieved -> no_passages refusal
        assert data["refusal_reason"] in ("no_passages", "weak_match", "model_refused")

    async def test_conversation_history(self, client: AsyncClient, auth_headers):
        # Upload doc
        content = b"Mango drying at 60 C."
        files = {"file": ("conv.txt", content, "text/plain")}
        await client.post("/api/documents/upload", headers=auth_headers, files=files)

        # First question
        resp1 = await client.post("/api/chat/ask", headers=auth_headers, json={
            "question": "What temperature for mango drying?"
        })
        assert resp1.status_code == 200
        conv_id = resp1.json()["conversation_id"]

        # Second question in same conversation
        resp2 = await client.post("/api/chat/ask", headers=auth_headers, json={
            "question": "And for how many hours?",
            "conversation_id": conv_id
        })
        assert resp2.status_code == 200
        assert resp2.json()["conversation_id"] == conv_id

    async def test_list_conversations(self, client: AsyncClient, auth_headers):
        resp = await client.get("/api/chat/conversations", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)

    async def test_get_conversation(self, client: AsyncClient, auth_headers):
        # Create conversation
        content = b"Test document."
        files = {"file": ("conv2.txt", content, "text/plain")}
        await client.post("/api/documents/upload", headers=auth_headers, files=files)

        resp1 = await client.post("/api/chat/ask", headers=auth_headers, json={
            "question": "Test question?"
        })
        conv_id = resp1.json()["conversation_id"]

        resp = await client.get(f"/api/chat/conversations/{conv_id}", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == conv_id
        assert len(data["messages"]) >= 2


class TestSettings:
    async def test_get_settings(self, client: AsyncClient, auth_headers):
        resp = await client.get("/api/settings", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "embedding_provider" in data
        assert "llm_provider" in data

    async def test_get_providers(self, client: AsyncClient, auth_headers):
        for endpoint in [
            "/api/settings/embedding-providers",
            "/api/settings/llm-providers",
            "/api/settings/rerankers",
            "/api/settings/vector-stores",
            "/api/settings/keyword-indexes",
        ]:
            resp = await client.get(endpoint, headers=auth_headers)
            assert resp.status_code == 200
            data = resp.json()
            assert isinstance(data, list)
            assert len(data) > 0


class TestHealth:
    async def test_health_check(self, client: AsyncClient):
        resp = await client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    async def test_readiness(self, client: AsyncClient):
        resp = await client.get("/api/health/ready")
        assert resp.status_code == 200
        assert "database" in resp.json()

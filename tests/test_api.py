import pytest

fastapi = pytest.importorskip("fastapi", reason="api extra not installed")

from fastapi.testclient import TestClient  # noqa: E402

from document_parser.api.main import create_app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app()) as test_client:
        yield test_client


def test_parse_endpoint(client):
    response = client.post(
        "/v1/parse", files={"file": ("doc.txt", b"hello\n\nworld", "text/plain")}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["format"] == "txt"
    assert [el["text"] for el in body["elements"]] == ["hello", "world"]


def test_parse_unsupported_format(client):
    response = client.post("/v1/parse", files={"file": ("doc.xyz", b"data", "text/plain")})
    assert response.status_code == 415


def test_chunk_endpoint(client):
    payload = {
        "segments": [{"text": "some text to chunk", "metadata": {"source": "x"}}],
        "config": {"chunk_size": 100, "chunk_overlap": 10},
    }
    response = client.post("/v1/chunk", json=payload)
    assert response.status_code == 200
    chunks = response.json()
    assert chunks[0]["text"] == "some text to chunk"
    assert chunks[0]["metadata"] == {"source": "x"}


def test_ingest_endpoint(client):
    response = client.post(
        "/v1/ingest",
        params={"chunk_size": 100, "chunk_overlap": 10},
        files={"file": ("doc.txt", b"para one\n\npara two", "text/plain")},
    )
    assert response.status_code == 200
    assert [chunk["text"] for chunk in response.json()] == ["para one", "para two"]

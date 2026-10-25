"""Web API access control (the agent itself is not started: no lifespan, no network)."""

from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

import web.app as web_app


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(web_app, "settings", replace(web_app.settings, access_code="letmein"))
    return TestClient(web_app.app)  # not used as a context manager, so lifespan doesn't run


def test_config_reports_access_code_requirement(client: TestClient) -> None:
    assert client.get("/api/config").json() == {"access_code_required": True}


@pytest.mark.parametrize("headers", [{}, {"X-Access-Code": "wrong"}])
def test_calls_rejected_without_valid_code(client: TestClient, headers: dict[str, str]) -> None:
    assert client.post("/api/call", headers=headers).status_code == 401
    assert client.post("/api/end", data={"session_id": "x"}, headers=headers).status_code == 401


def test_valid_code_passes_the_gate(client: TestClient) -> None:
    # Past the gate the agent isn't running in this test, so 503 (starting) proves access was granted.
    assert client.post("/api/call", headers={"X-Access-Code": "letmein"}).status_code == 503


def test_no_code_configured_means_open_access(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(web_app, "settings", replace(web_app.settings, access_code=""))
    client = TestClient(web_app.app)
    assert client.get("/api/config").json() == {"access_code_required": False}
    assert client.post("/api/call").status_code == 503

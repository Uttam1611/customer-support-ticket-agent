from __future__ import annotations

from datetime import datetime, timezone

from fastapi.testclient import TestClient

from src.api.server import app, pipeline
from src.models import Ticket


def test_health_returns_ready_when_pipeline_is_ready(
    monkeypatch,
) -> None:
    async def fake_initialize() -> None:
        pipeline.ready = True

    async def fake_shutdown() -> None:
        pipeline.ready = False

    monkeypatch.setattr(
        pipeline,
        "initialize",
        fake_initialize,
    )

    monkeypatch.setattr(
        pipeline,
        "shutdown",
        fake_shutdown,
    )

    with TestClient(app) as client:
        response = client.get(
            "/health"
        )

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready"
    }


def test_health_returns_503_when_not_ready(
    monkeypatch,
) -> None:
    async def fake_initialize() -> None:
        pipeline.ready = False

    async def fake_shutdown() -> None:
        pipeline.ready = False

    monkeypatch.setattr(
        pipeline,
        "initialize",
        fake_initialize,
    )

    monkeypatch.setattr(
        pipeline,
        "shutdown",
        fake_shutdown,
    )

    with TestClient(app) as client:
        response = client.get(
            "/health"
        )

    assert response.status_code == 503


def test_chat_rejects_blank_message(
    monkeypatch,
) -> None:
    async def fake_initialize() -> None:
        pipeline.ready = True

    async def fake_shutdown() -> None:
        pipeline.ready = False

    monkeypatch.setattr(
        pipeline,
        "initialize",
        fake_initialize,
    )

    monkeypatch.setattr(
        pipeline,
        "shutdown",
        fake_shutdown,
    )

    with TestClient(
        app,
        raise_server_exceptions=False,
    ) as client:
        response = client.post(
            "/chat",
            json={
                "session_id": "demo-session",
                "message": "   ",
            },
        )

    assert response.status_code == 422


def test_unknown_ticket_returns_404(
    monkeypatch,
) -> None:
    async def fake_initialize() -> None:
        pipeline.ready = True

    async def fake_shutdown() -> None:
        pipeline.ready = False

    monkeypatch.setattr(
        pipeline,
        "initialize",
        fake_initialize,
    )

    monkeypatch.setattr(
        pipeline,
        "shutdown",
        fake_shutdown,
    )

    with TestClient(app) as client:
        response = client.get(
            "/tickets/CST-2026-9999"
        )

    assert response.status_code == 404
    assert response.json()["detail"] == "Ticket not found"
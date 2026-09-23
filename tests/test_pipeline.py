from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.server import app, pipeline
from src.config import Settings
from src.models import ChatResponse
from src.pipeline import SupportPipeline
from src.utils.errors import (
    ComponentNotReadyError,
)


def settings() -> Settings:
    return Settings(
        llm_base_url="http://localhost:11434/v1",
        llm_api_key="not-required",
        llm_model="test-model",
        vector_db_path=".data/test-vector-db",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        rag_collection="test-support",
        rag_top_k=3,
        rag_relevance_threshold=0.35,
        api_host="127.0.0.1",
        api_port=8000,
        streamlit_host="127.0.0.1",
        streamlit_port=8501,
    )


@pytest.mark.asyncio
async def test_uninitialized_pipeline_rejects_chat(
    tmp_path: Path,
) -> None:
    pipeline = SupportPipeline(
        settings(),
        tmp_path,
    )

    with pytest.raises(
        ComponentNotReadyError
    ):
        await pipeline.process(
            "session-1",
            "hello",
        )


@pytest.mark.asyncio
async def test_blank_session_is_rejected(
    tmp_path: Path,
) -> None:
    pipeline = SupportPipeline(
        settings(),
        tmp_path,
    )

    pipeline.ready = True

    with pytest.raises(
        ValueError,
        match="session_id must not be blank",
    ):
        await pipeline.process(
            "   ",
            "hello",
        )


@pytest.mark.asyncio
async def test_blank_message_is_rejected(
    tmp_path: Path,
) -> None:
    pipeline = SupportPipeline(
        settings(),
        tmp_path,
    )

    pipeline.ready = True

    with pytest.raises(
        ValueError,
        match="message must not be blank",
    ):
        await pipeline.process(
            "session-1",
            "   ",
        )

def test_chat_returns_pipeline_response(
    monkeypatch,
) -> None:
    async def fake_initialize() -> None:
        pipeline.ready = True

    async def fake_shutdown() -> None:
        pipeline.ready = False

    async def fake_process(
        session_id: str,
        message: str,
    ) -> ChatResponse:
        assert session_id == "demo-session"
        assert message == "Where is my order?"

        return ChatResponse(
            session_id=session_id,
            response="Please check your tracking information.",
            sources=["shipping.md"],
            ticket_id=None,
        )

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

    monkeypatch.setattr(
        pipeline,
        "process",
        fake_process,
    )

    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={
                "session_id": "demo-session",
                "message": "Where is my order?",
            },
        )

    assert response.status_code == 200

    assert response.json() == {
        "success": True,
        "session_id": "demo-session",
        "response": (
            "Please check your tracking information."
        ),
        "sources": ["shipping.md"],
        "ticket_id": None,
    }
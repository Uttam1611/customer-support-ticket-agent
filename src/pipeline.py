from __future__ import annotations

from pathlib import Path
from typing import Any

from src.config import Settings
from src.llm.client import build_chat_model
from src.llm.workflow import build_support_workflow
from src.models import ChatResponse
from src.rag.retriever import KnowledgeRetriever
from src.sessions.store import SessionStore
from src.tools.ticket_tool import TicketRepository
from src.utils.errors import (
    AgentProcessingError,
    ComponentNotReadyError,
)


class SupportPipeline:
    """Application-level orchestration boundary.

    FastAPI knows only about request/response models. The pipeline owns
    session state, workflow execution, ticket consistency, and conversion
    into the public ChatResponse contract.
    """

    def __init__(
        self,
        settings: Settings,
        documents_dir: Path,
    ) -> None:
        self.settings = settings

        self.model = build_chat_model(settings)

        self.retriever = KnowledgeRetriever(
            settings,
            documents_dir,
        )

        self.sessions = SessionStore()
        self.tickets = TicketRepository()

        self.workflow = None
        self.ready = False

    async def initialize(self) -> None:
        """Initialize expensive shared components once."""

        self.ready = False

        try:
            await self.retriever.initialize()

            self.workflow = build_support_workflow(
                model=self.model,
                retriever=self.retriever,
                sessions=self.sessions,
                tickets=self.tickets,
            )

            self.ready = True

        except Exception:
            self.workflow = None
            self.ready = False
            raise

    async def shutdown(self) -> None:
        """Mark the pipeline unavailable during application shutdown."""

        self.ready = False
        self.workflow = None

    @staticmethod
    def _clean_text(
        value: str,
        field_name: str,
    ) -> str:
        """Normalize and validate user-controlled text."""

        if not isinstance(value, str):
            raise ValueError(
                f"{field_name} must be a string"
            )

        cleaned = value.strip()

        if not cleaned:
            raise ValueError(
                f"{field_name} must not be blank"
            )

        return cleaned

    @staticmethod
    def _normalize_sources(
        sources: Any,
    ) -> list[str]:
        """Return unique, non-empty source filenames."""

        if not sources:
            return []

        normalized: list[str] = []

        for source in sources:
            value = str(source).strip()

            if not value:
                continue

            if value not in normalized:
                normalized.append(value)

        return normalized

    async def process(
        self,
        session_id: str,
        message: str,
    ) -> ChatResponse:
        """Process one customer turn through the support workflow."""

        if not self.ready or self.workflow is None:
            raise ComponentNotReadyError(
                "Support pipeline is not ready"
            )

        clean_session = self._clean_text(
            session_id,
            "session_id",
        )

        clean_message = self._clean_text(
            message,
            "message",
        )

        session = self.sessions.get_or_create(
            clean_session
        )

        # Store only normalized user input.
        session.history.append(
            {
                "role": "user",
                "content": clean_message,
            }
        )

        initial_state = {
            "session_id": clean_session,
            "customer_message": clean_message,
            "messages": [],
            "retrieved_chunks": [],
            "route": "",
            "extracted_fields": {},
            "response_text": "",
            "sources": [],
            "ticket_id": session.ticket_id,
        }

        try:
            result = await self.workflow.ainvoke(
                initial_state
            )

        except AgentProcessingError:
            # Preserve known application errors so FastAPI can map
            # them to the appropriate HTTP response.
            raise

        except Exception as exc:
            raise AgentProcessingError(
                "Unable to process the support request."
            ) from exc

        response_text = str(
            result.get(
                "response_text",
                "",
            )
        ).strip()

        if not response_text:
            raise AgentProcessingError(
                "The agent returned an empty response."
            )

        sources = self._normalize_sources(
            result.get("sources", [])
        )

        ticket_id_raw = result.get(
            "ticket_id"
        )

        ticket_id: str | None = None

        if ticket_id_raw is not None:
            ticket_id = str(
                ticket_id_raw
            ).strip()

            if not ticket_id:
                ticket_id = None

        # If the workflow claims to have created/returned a ticket,
        # verify that it actually exists in the repository.
        if ticket_id is not None:
            ticket = self.tickets.get(
                ticket_id
            )

            if ticket is None:
                raise AgentProcessingError(
                    "The agent returned an unknown ticket ID."
                )

            # Keep the session's ticket state synchronized with
            # the authoritative repository record.
            session.ticket_id = ticket.ticket_id

        session.history.append(
            {
                "role": "assistant",
                "content": response_text,
            }
        )

        return ChatResponse(
            success=True,
            session_id=clean_session,
            response=response_text,
            sources=sources,
            ticket_id=ticket_id,
        )
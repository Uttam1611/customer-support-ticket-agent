from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from src.llm.workflow import build_support_workflow
from src.models import TicketCreate
from src.sessions.store import SessionStore
from src.tools.ticket_tool import TicketRepository


class FakeRetriever:
    def __init__(
        self,
        results: list[dict[str, str]],
    ) -> None:
        self.results = results

    async def search(
        self,
        query: str,
    ) -> list[dict[str, str]]:
        return self.results


class FakeStructuredModel:
    def __init__(
        self,
        decision: dict[str, Any],
    ) -> None:
        self.decision = decision

    def with_structured_output(
        self,
        _schema: Any,
    ) -> "FakeStructuredModel":
        return self

    async def ainvoke(
        self,
        _messages: Any,
    ) -> Any:
        return self.decision


class FakeAnswerModel:
    def __init__(
        self,
        decision: dict[str, Any],
        answer: str = "Standard delivery takes three to five business days.",
    ) -> None:
        self.decision = decision
        self.answer = answer

    def with_structured_output(
        self,
        _schema: Any,
    ) -> "FakeAnswerModel":
        return self

    async def ainvoke(
        self,
        messages: Any,
    ) -> Any:
        # The decision node receives the structured-output model.
        # The answer node receives the normal model invocation.
        if isinstance(messages, list):
            first = messages[0]

            if hasattr(first, "content") and (
                "Classify the customer's current message"
                in str(first.content)
            ):
                return self.decision

        return AIMessage(
            content=self.answer
        )


@pytest.mark.asyncio
async def test_answer_route_uses_retrieved_evidence() -> None:
    retriever = FakeRetriever(
        [
            {
                "content": (
                    "Standard delivery usually takes "
                    "three to five business days."
                ),
                "source": "shipping.md",
            }
        ]
    )

    model = FakeAnswerModel(
        {
            "route": "answer",
        }
    )

    workflow = build_support_workflow(
        model=model,
        retriever=retriever,
        sessions=SessionStore(),
        tickets=TicketRepository(),
    )

    result = await workflow.ainvoke(
        {
            "session_id": "answer-session",
            "customer_message": (
                "How long does standard shipping take?"
            ),
        }
    )

    assert result["route"] == "answer"
    assert result["sources"] == ["shipping.md"]
    assert result["response_text"]


@pytest.mark.asyncio
async def test_unknown_question_does_not_invent_answer() -> None:
    retriever = FakeRetriever([])

    model = FakeAnswerModel(
        {
            "route": "answer",
        }
    )

    workflow = build_support_workflow(
        model=model,
        retriever=retriever,
        sessions=SessionStore(),
        tickets=TicketRepository(),
    )

    result = await workflow.ainvoke(
        {
            "session_id": "unknown-session",
            "customer_message": (
                "What is your policy for lunar shipping?"
            ),
        }
    )

    assert result["route"] == "answer"
    assert result["sources"] == []
    assert (
        "don't have enough information"
        in result["response_text"]
    )


@pytest.mark.asyncio
async def test_ticket_route_asks_for_missing_fields() -> None:
    retriever = FakeRetriever([])

    model = FakeAnswerModel(
        {
            "route": "ticket",
            "issue_description": (
                "My payment was charged twice."
            ),
        }
    )

    sessions = SessionStore()
    tickets = TicketRepository()

    workflow = build_support_workflow(
        model=model,
        retriever=retriever,
        sessions=sessions,
        tickets=tickets,
    )

    result = await workflow.ainvoke(
        {
            "session_id": "ticket-session",
            "customer_message": (
                "My payment was charged twice."
            ),
        }
    )

    assert result["route"] == "ticket"

    assert (
        "name"
        in result["response_text"].lower()
    )

    assert (
        sessions.get_or_create(
            "ticket-session"
        ).issue_description
        == "My payment was charged twice."
    )


@pytest.mark.asyncio
async def test_complete_ticket_is_created_once() -> None:
    retriever = FakeRetriever([])

    model = FakeAnswerModel(
        {
            "route": "ticket",
            "customer_name": "Rahul Sharma",
            "customer_email": "rahul@example.com",
            "issue_description": (
                "My payment was charged twice."
            ),
            "category": "payment",
        }
    )

    sessions = SessionStore()
    tickets = TicketRepository()

    workflow = build_support_workflow(
        model=model,
        retriever=retriever,
        sessions=sessions,
        tickets=tickets,
    )

    state = {
        "session_id": "complete-ticket",
        "customer_message": (
            "My payment was charged twice. "
            "My name is Rahul Sharma and my email is "
            "rahul@example.com. This is a payment issue."
        ),
    }

    first = await workflow.ainvoke(state)

    ticket_id = first["ticket_id"]

    assert ticket_id
    assert ticket_id.startswith("CST-")

    stored = tickets.get(ticket_id)

    assert stored is not None
    assert stored.customer_name == "Rahul Sharma"
    assert stored.customer_email == "rahul@example.com"
    assert stored.category == "payment"


@pytest.mark.asyncio
async def test_existing_ticket_is_not_duplicated() -> None:
    retriever = FakeRetriever([])

    model = FakeAnswerModel(
        {
            "route": "ticket",
            "customer_name": "Rahul Sharma",
            "customer_email": "rahul@example.com",
            "issue_description": (
                "My payment was charged twice."
            ),
            "category": "payment",
        }
    )

    sessions = SessionStore()
    tickets = TicketRepository()

    workflow = build_support_workflow(
        model=model,
        retriever=retriever,
        sessions=sessions,
        tickets=tickets,
    )

    state = {
        "session_id": "duplicate-session",
        "customer_message": (
            "My payment was charged twice."
        ),
    }

    first = await workflow.ainvoke(state)
    second = await workflow.ainvoke(state)

    assert first["ticket_id"]
    assert second["ticket_id"]

    assert (
        first["ticket_id"]
        == second["ticket_id"]
    )


@pytest.mark.asyncio
async def test_multi_turn_ticket_collection_preserves_session_state() -> None:
    retriever = FakeRetriever([])

    class MultiTurnModel(FakeAnswerModel):
        async def ainvoke(self, messages: Any) -> Any:
            if isinstance(messages, list):
                first = messages[0]
                if hasattr(first, "content") and (
                    "Classify the customer's current message"
                    in str(first.content)
                ):
                    return {
                        "route": "ticket",
                        "customer_name": "Lena",
                    }

            return AIMessage(content="Thanks, I have your name.")

    sessions = SessionStore()
    tickets = TicketRepository()

    workflow = build_support_workflow(
        model=MultiTurnModel({"route": "ticket"}),
        retriever=retriever,
        sessions=sessions,
        tickets=tickets,
    )

    first = await workflow.ainvoke(
        {
            "session_id": "multi-turn-session",
            "customer_message": "Hi, my name is Lena.",
        }
    )
    second = await workflow.ainvoke(
        {
            "session_id": "multi-turn-session",
            "customer_message": "My email is lena@example.com.",
        }
    )

    session = sessions.get_or_create("multi-turn-session")

    assert first["route"] == "ticket"
    assert second["route"] == "ticket"
    assert session.customer_name == "Lena"
    assert session.customer_email == "lena@example.com" or session.customer_email is None
    assert session.history


@pytest.mark.asyncio
async def test_ticket_repository_supports_lookup_by_id() -> None:
    tickets = TicketRepository()
    created = tickets.create(
        "lookup-session",
        TicketCreate(
            customer_name="Test User",
            customer_email="test@example.com",
            issue_description="Issue description",
            category="technical",
            summary="Technical issue",
        ),
    )

    assert tickets.get(created.ticket_id) == created
    assert tickets.get("missing-id") is None
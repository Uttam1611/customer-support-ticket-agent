from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import (
    BaseModel,
    EmailStr,
    Field,
    field_validator,
)


TicketCategory = Literal[
    "order",
    "payment",
    "account",
    "technical",
    "other",
]


class ChatRequest(BaseModel):
    """Public POST /chat request contract."""

    session_id: str = Field(
        min_length=1
    )

    message: str = Field(
        min_length=1
    )

    @field_validator(
        "session_id",
        "message",
    )
    @classmethod
    def reject_blank_text(
        cls,
        value: str,
    ) -> str:
        value = value.strip()

        if not value:
            raise ValueError(
                "value must not be blank"
            )

        return value


class TicketCreate(BaseModel):
    """Validated data allowed to cross into ticket creation."""

    customer_name: str = Field(
        min_length=1
    )

    customer_email: EmailStr

    issue_description: str = Field(
        min_length=5
    )

    category: TicketCategory

    summary: str = Field(
        min_length=5,
        max_length=160,
    )

    @field_validator(
        "customer_name",
        "issue_description",
        "summary",
    )
    @classmethod
    def reject_blank_text(
        cls,
        value: str,
    ) -> str:
        value = value.strip()

        if not value:
            raise ValueError(
                "value must not be blank"
            )

        return value


class Ticket(TicketCreate):
    """Repository-owned ticket record."""

    ticket_id: str

    status: Literal["open"] = "open"

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(
            timezone.utc
        )
    )


class ChatResponse(BaseModel):
    """Public POST /chat response contract."""

    success: bool = True

    session_id: str

    response: str

    sources: list[str] = Field(
        default_factory=list
    )

    ticket_id: str | None = None
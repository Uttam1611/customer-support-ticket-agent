from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ConversationState:
    customer_name: str | None = None
    customer_email: str | None = None
    issue_description: str | None = None
    category: str | None = None
    ticket_id: str | None = None
    history: list[dict[str, str]] = field(default_factory=list)

    def missing_ticket_fields(self) -> list[str]:
        values = {
            "customer_name": self.customer_name,
            "customer_email": self.customer_email,
            "issue_description": self.issue_description,
            "category": self.category,
        }

        return [
            name
            for name, value in values.items()
            if not value
        ]

    def update_fields(
        self,
        values: dict[str, str | None],
    ) -> None:
        for field_name in (
            "customer_name",
            "customer_email",
            "issue_description",
            "category",
        ):
            value = values.get(field_name)

            if value is not None and str(value).strip():
                setattr(
                    self,
                    field_name,
                    str(value).strip(),
                )


class SessionStore:
    """In-memory storage for per-session conversation state."""

    def __init__(self) -> None:
        self._sessions: dict[str, ConversationState] = {}

    def get_or_create(self, session_id: str) -> ConversationState:
        clean_session_id = str(session_id).strip()

        if not clean_session_id:
            raise ValueError("session_id must not be blank")

        session = self._sessions.get(clean_session_id)

        if session is None:
            session = ConversationState()
            self._sessions[clean_session_id] = session

        return session

    def get(self, session_id: str) -> ConversationState | None:
        clean_session_id = str(session_id).strip()

        if not clean_session_id:
            return None

        return self._sessions.get(clean_session_id)

    def all(self) -> list[ConversationState]:
        return list(self._sessions.values())
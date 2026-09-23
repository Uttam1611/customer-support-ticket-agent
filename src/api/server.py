from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException

from src.config import load_settings
from src.models import (
    ChatRequest,
    ChatResponse,
    Ticket,
)
from src.pipeline import SupportPipeline
from src.utils.errors import (
    AgentProcessingError,
    ComponentNotReadyError,
)


settings = load_settings()

pipeline = SupportPipeline(
    settings,
    Path(__file__).resolve().parents[2]
    / "knowledge_base",
)


@asynccontextmanager
async def lifespan(
    _: FastAPI,
):
    try:
        await pipeline.initialize()
        yield
    finally:
        await pipeline.shutdown()


app = FastAPI(
    title="Customer Support Ticket Agent",
    version="1.0.0",
    description=(
        "Session-aware customer-support agent using "
        "grounded RAG and ticket creation."
    ),
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, str]:
    if not pipeline.ready:
        raise HTTPException(
            status_code=503,
            detail="Support pipeline is not ready",
        )

    return {"status": "ready"}


@app.post(
    "/chat",
    response_model=ChatResponse,
)
async def chat(
    request: ChatRequest,
) -> ChatResponse:
    try:
        return await pipeline.process(
            request.session_id,
            request.message,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except ComponentNotReadyError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    except AgentProcessingError as exc:
        raise HTTPException(
            status_code=502,
            detail=str(exc),
        ) from exc


@app.get(
    "/tickets/{ticket_id}",
    response_model=Ticket,
)
async def get_ticket(
    ticket_id: str,
) -> Ticket:
    clean_ticket_id = ticket_id.strip()

    if not clean_ticket_id:
        raise HTTPException(
            status_code=422,
            detail="ticket_id must not be blank",
        )

    ticket = pipeline.tickets.get(
        clean_ticket_id
    )

    if ticket is None:
        raise HTTPException(
            status_code=404,
            detail="Ticket not found",
        )

    return ticket
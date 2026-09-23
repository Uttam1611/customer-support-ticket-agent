from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response

from src.config import load_settings
from src.models import ChatRequest, ChatResponse
from src.pipeline import SupportPipeline
from src.utils.errors import AgentProcessingError, ComponentNotReadyError
from src.voice.models import SynthesisRequest, TranscriptionResponse
from src.voice.pipeline import VoicePipeline
from src.voice.stt import GoogleSTTService
from src.voice.tts import EdgeTTSService


settings = load_settings()

pipeline = SupportPipeline(
    settings,
    Path(__file__).resolve().parents[2] / "knowledge_base",
)

voice_pipeline = VoicePipeline(
    GoogleSTTService(),
    EdgeTTSService(),
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    await pipeline.initialize()

    try:
        await voice_pipeline.initialize()
        yield
    finally:
        await voice_pipeline.cleanup()
        await pipeline.shutdown()


app = FastAPI(
    title="Customer Support Ticket Agent",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, str]:
    if not pipeline.ready:
        raise HTTPException(
            status_code=503,
            detail="Backend is not ready",
        )

    return {"status": "ready"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        return await pipeline.process(
            request.session_id,
            request.message,
        )

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


@app.get("/tickets/{ticket_id}")
async def get_ticket(ticket_id: str):
    ticket = pipeline.tickets.get(ticket_id)

    if ticket is None:
        raise HTTPException(
            status_code=404,
            detail="Ticket not found",
        )

    return ticket


@app.post(
    "/voice/transcribe",
    response_model=TranscriptionResponse,
)
async def transcribe_voice(
    audio: UploadFile = File(...),
) -> TranscriptionResponse:
    try:
        audio_bytes = await audio.read()

        transcript, processing_time = (
            await voice_pipeline.transcribe(
                audio_bytes,
                audio.content_type or "",
            )
        )

        return TranscriptionResponse(
            transcript=transcript,
            processing_time_ms=processing_time,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="Speech transcription failed",
        ) from exc


@app.post("/voice/synthesize")
async def synthesize_voice(
    request: SynthesisRequest,
) -> Response:
    try:
        audio, media_type, _ = (
            await voice_pipeline.synthesize(request.text)
        )

        return Response(
            content=audio,
            media_type=media_type,
            headers={
                "X-Message-ID": request.message_id,
            },
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="Text-to-speech synthesis failed",
        ) from exc
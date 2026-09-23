import pytest

from src.voice.contracts import STTService, TTSService
from src.voice.pipeline import VoicePipeline


class FakeSTT(STTService):
    async def initialize(self) -> None:
        return None

    async def transcribe(
        self,
        audio_bytes: bytes,
        media_type: str,
    ) -> str:
        return "My payment was charged twice"


class EmptySTT(STTService):
    async def initialize(self) -> None:
        return None

    async def transcribe(
        self,
        audio_bytes: bytes,
        media_type: str,
    ) -> str:
        return ""


class FakeTTS(TTSService):
    async def initialize(self) -> None:
        return None

    async def synthesize(
        self,
        text: str,
    ) -> tuple[bytes, str]:
        return b"fake-audio", "audio/mpeg"


@pytest.mark.asyncio
async def test_successful_transcription() -> None:
    pipeline = VoicePipeline(
        FakeSTT(),
        FakeTTS(),
    )

    transcript, elapsed = await pipeline.transcribe(
        b"audio",
        "audio/wav",
    )

    assert transcript == "My payment was charged twice"
    assert elapsed >= 0


@pytest.mark.asyncio
async def test_empty_audio_rejected() -> None:
    pipeline = VoicePipeline(
        FakeSTT(),
        FakeTTS(),
    )

    with pytest.raises(ValueError, match="empty"):
        await pipeline.transcribe(
            b"",
            "audio/wav",
        )


@pytest.mark.asyncio
async def test_empty_transcript_rejected() -> None:
    pipeline = VoicePipeline(
        EmptySTT(),
        FakeTTS(),
    )

    with pytest.raises(
        ValueError,
        match="No understandable speech",
    ):
        await pipeline.transcribe(
            b"audio",
            "audio/wav",
        )


@pytest.mark.asyncio
async def test_successful_synthesis() -> None:
    pipeline = VoicePipeline(
        FakeSTT(),
        FakeTTS(),
    )

    audio, media_type, elapsed = (
        await pipeline.synthesize(
            "Agent response"
        )
    )

    assert audio == b"fake-audio"
    assert media_type == "audio/mpeg"
    assert elapsed >= 0


@pytest.mark.asyncio
async def test_empty_tts_text_rejected() -> None:
    pipeline = VoicePipeline(
        FakeSTT(),
        FakeTTS(),
    )

    with pytest.raises(ValueError, match="empty"):
        await pipeline.synthesize("")
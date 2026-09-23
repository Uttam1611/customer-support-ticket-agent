from __future__ import annotations

import time

from src.voice.contracts import STTService, TTSService


class VoicePipeline:
    """Simple wrapper around speech-to-text and text-to-speech services."""

    def __init__(
        self,
        stt_service: STTService,
        tts_service: TTSService,
    ) -> None:
        self.stt = stt_service
        self.tts = tts_service

    async def initialize(self) -> None:
        await self.stt.initialize()
        await self.tts.initialize()

    async def cleanup(self) -> None:
        await self.stt.cleanup()
        await self.tts.cleanup()

    async def transcribe(
        self,
        audio_bytes: bytes,
        media_type: str,
    ) -> tuple[str, int]:
        if not audio_bytes:
            raise ValueError("Audio input is empty")

        started_at = time.monotonic()
        transcript = await self.stt.transcribe(audio_bytes, media_type)

        clean_transcript = transcript.strip()
        if not clean_transcript:
            raise ValueError("No understandable speech was detected")

        elapsed_ms = max(0, int((time.monotonic() - started_at) * 1000))
        return clean_transcript, elapsed_ms

    async def synthesize(
        self,
        text: str,
    ) -> tuple[bytes, str, int]:
        clean_text = text.strip()

        if not clean_text:
            raise ValueError("Text input is empty")

        started_at = time.monotonic()
        audio, media_type = await self.tts.synthesize(clean_text)

        if not audio:
            raise RuntimeError("Text-to-speech returned empty audio")

        elapsed_ms = max(0, int((time.monotonic() - started_at) * 1000))
        return audio, media_type, elapsed_ms
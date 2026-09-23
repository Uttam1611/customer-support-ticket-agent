from __future__ import annotations

import os

import edge_tts

from .contracts import TTSService


class EdgeTTSService(TTSService):
    """Text-to-speech using Edge TTS."""

    def __init__(self) -> None:
        self.voice = os.getenv(
            "TTS_VOICE",
            "en-US-AriaNeural",
        )

    async def initialize(self) -> None:
        return None

    async def synthesize(
        self,
        text: str,
    ) -> tuple[bytes, str]:
        clean_text = text.strip()

        if not clean_text:
            raise ValueError("Text input is empty")

        communicator = edge_tts.Communicate(
            clean_text,
            self.voice,
        )

        chunks: list[bytes] = []

        try:
            async for chunk in communicator.stream():
                if chunk.get("type") == "audio":
                    chunks.append(chunk["data"])
        except Exception as exc:
            raise RuntimeError(
                "Text-to-speech synthesis failed"
            ) from exc

        audio = b"".join(chunks)

        if not audio:
            raise RuntimeError(
                "Text-to-speech returned empty audio"
            )

        return audio, "audio/mpeg"
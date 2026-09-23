from __future__ import annotations

from abc import ABC, abstractmethod


class STTService(ABC):
    @abstractmethod
    async def initialize(self) -> None:
        """Prepare the STT service."""

    @abstractmethod
    async def transcribe(self, audio_bytes: bytes, media_type: str) -> str:
        """Convert audio bytes into text."""

    async def cleanup(self) -> None:
        """Release STT resources."""


class TTSService(ABC):
    @abstractmethod
    async def initialize(self) -> None:
        """Prepare the TTS service."""

    @abstractmethod
    async def synthesize(self, text: str) -> tuple[bytes, str]:
        """Convert text into playable audio."""

    async def cleanup(self) -> None:
        """Release TTS resources."""
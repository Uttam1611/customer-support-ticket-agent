from __future__ import annotations

import io
import wave

import speech_recognition as sr

from .contracts import STTService


class GoogleSTTService(STTService):
    """Speech-to-text using the individual SpeechRecognition Google adapter."""

    async def initialize(self) -> None:
        return None

    async def transcribe(
        self,
        audio_bytes: bytes,
        media_type: str,
    ) -> str:
        if not audio_bytes:
            raise ValueError("Audio input is empty")

        if media_type.lower() not in {
            "audio/wav",
            "audio/x-wav",
            "audio/wave",
        }:
            raise ValueError(
                f"Unsupported audio type: {media_type}"
            )

        try:
            with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
                frames = wav_file.readframes(wav_file.getnframes())
                sample_rate = wav_file.getframerate()
                sample_width = wav_file.getsampwidth()

            audio = sr.AudioData(
                frames,
                sample_rate,
                sample_width,
            )

            recognizer = sr.Recognizer()

            transcript = recognizer.recognize_google(audio)

        except sr.UnknownValueError as exc:
            raise ValueError(
                "No understandable speech was detected"
            ) from exc

        except sr.RequestError as exc:
            raise RuntimeError(
                "Speech recognition service is unavailable"
            ) from exc

        except (wave.Error, ValueError) as exc:
            raise ValueError(
                "The uploaded audio could not be processed"
            ) from exc

        return transcript.strip()
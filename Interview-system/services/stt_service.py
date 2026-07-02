"""
Phase 3 — Speech-to-Text Service
AI Voice Interview System

Local Whisper STT using faster-whisper for zero-cost transcription.
Falls back to Deepgram API if configured.

Usage:
    stt = create_stt_service()
    transcript = await stt.transcribe(audio_bytes)
"""

from __future__ import annotations

import asyncio
import io
import logging
import tempfile
import wave
from abc import ABC, abstractmethod

import numpy as np

log = logging.getLogger("phase3.stt")


class BaseSTT(ABC):
    """Abstract STT interface."""

    @abstractmethod
    async def transcribe(self, audio_data: bytes, sample_rate: int = 16000) -> str:
        """Transcribe audio bytes to text."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Clean up resources."""
        ...


class WhisperSTT(BaseSTT):
    """
    Local STT using faster-whisper (CTranslate2 backend).
    Completely free — no API calls, no limits.
    """

    def __init__(self, model_size: str = "base", device: str = "cpu"):
        from faster_whisper import WhisperModel

        log.info("Loading Whisper model '%s' on %s...", model_size, device)
        compute_type = "float16" if device == "cuda" else "int8"
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )
        log.info("Whisper model loaded")

    async def transcribe(self, audio_data: bytes, sample_rate: int = 16000) -> str:
        """
        Transcribe raw PCM audio bytes.
        Runs Whisper in a thread pool to avoid blocking the event loop.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._transcribe_sync, audio_data, sample_rate
        )

    def _transcribe_sync(self, audio_data: bytes, sample_rate: int) -> str:
        """Synchronous transcription."""
        # Convert raw bytes to numpy array
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

        if len(audio_np) < 1600:  # less than 0.1s
            return ""

        # Write to temp WAV file (faster-whisper needs file or numpy)
        segments, info = self.model.transcribe(
            audio_np,
            beam_size=5,
            language="en",
            vad_filter=True,
        )

        # Collect all segments
        text_parts = []
        for segment in segments:
            text_parts.append(segment.text.strip())

        transcript = " ".join(text_parts).strip()
        if transcript:
            log.info("Whisper transcript: '%s' (%.1fs audio)", transcript, len(audio_np) / sample_rate)
        return transcript

    async def close(self) -> None:
        del self.model
        log.info("Whisper model unloaded")


class DeepgramSTT(BaseSTT):
    """
    Cloud STT using Deepgram API.
    Uses free trial credits — use sparingly.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        log.info("Deepgram STT initialised (API mode)")

    async def transcribe(self, audio_data: bytes, sample_rate: int = 16000) -> str:
        """Transcribe audio via Deepgram API."""
        try:
            from deepgram import DeepgramClient, PrerecordedOptions

            client = DeepgramClient(self.api_key)

            # Write audio to WAV buffer
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(audio_data)

            wav_buffer.seek(0)
            payload = {"buffer": wav_buffer.read()}

            options = PrerecordedOptions(
                model="nova-3",
                smart_format=True,
                language="en",
            )

            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.listen.rest.v("1").transcribe_file(payload, options),
            )

            transcript = response.results.channels[0].alternatives[0].transcript
            if transcript:
                log.info("Deepgram transcript: '%s'", transcript)
            return transcript

        except Exception as exc:
            log.error("Deepgram transcription failed: %s", exc)
            return ""

    async def close(self) -> None:
        log.info("Deepgram STT closed")


def create_stt_service() -> BaseSTT:
    """Factory: create the configured STT service."""
    from config import STT_PROVIDER, WHISPER_MODEL, WHISPER_DEVICE, DEEPGRAM_API_KEY

    if STT_PROVIDER == "deepgram" and DEEPGRAM_API_KEY:
        return DeepgramSTT(api_key=DEEPGRAM_API_KEY)
    else:
        return WhisperSTT(model_size=WHISPER_MODEL, device=WHISPER_DEVICE)

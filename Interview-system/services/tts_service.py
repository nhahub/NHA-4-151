"""
Phase 3 — Text-to-Speech Service
AI Voice Interview System

Edge TTS (free, unlimited) as default, ElevenLabs as optional upgrade.

Usage:
    tts = create_tts_service()
    audio_bytes = await tts.synthesize("Hello, tell me about your experience.")
"""

from __future__ import annotations

import asyncio
import io
import logging
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import AsyncIterator

log = logging.getLogger("phase3.tts")


class BaseTTS(ABC):
    """Abstract TTS interface."""

    @abstractmethod
    async def synthesize(self, text: str) -> bytes:
        """Convert text to audio bytes (MP3 format)."""
        ...

    @abstractmethod
    async def stream_audio(self, text: str) -> AsyncIterator[bytes]:
        """Convert text to audio bytes and stream chunks."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Clean up resources."""
        ...


class EdgeTTS(BaseTTS):
    """
    Free TTS using Microsoft Edge's neural voices.
    No API key needed, unlimited usage, good quality.
    """

    def __init__(self, voice: str = "en-US-JennyNeural"):
        self.voice = voice
        log.info("Edge TTS initialised (voice=%s)", voice)

    async def synthesize(self, text: str) -> bytes:
        """Convert text to MP3 audio bytes."""
        import edge_tts

        communicate = edge_tts.Communicate(text, self.voice)

        # Collect all audio chunks
        audio_chunks = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_chunks.append(chunk["data"])

        audio_data = b"".join(audio_chunks)
        log.info("Edge TTS: synthesised %d bytes for '%s...'", len(audio_data), text[:50])
        return audio_data

    async def stream_audio(self, text: str) -> AsyncIterator[bytes]:
        """Stream audio chunks directly."""
        import edge_tts
        communicate = edge_tts.Communicate(text, self.voice)
        log.info("Edge TTS: streaming audio for '%s...'", text[:50])
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                yield chunk["data"]

    async def close(self) -> None:
        log.info("Edge TTS closed")


class ElevenLabsTTS(BaseTTS):
    """
    Premium TTS using ElevenLabs API.
    Uses free trial credits — limited characters/month.
    """

    def __init__(self, api_key: str, voice_id: str, model_id: str = "eleven_flash_v2_5"):
        self.api_key = api_key
        self.voice_id = voice_id
        self.model_id = model_id
        log.info("ElevenLabs TTS initialised (voice=%s, model=%s)", voice_id, model_id)

    async def synthesize(self, text: str) -> bytes:
        """Convert text to MP3 audio bytes via ElevenLabs API."""
        try:
            from elevenlabs.client import ElevenLabs

            client = ElevenLabs(api_key=self.api_key)

            # Run in thread pool (SDK is synchronous)
            loop = asyncio.get_event_loop()
            audio_iter = await loop.run_in_executor(
                None,
                lambda: client.text_to_speech.convert(
                    text=text,
                    voice_id=self.voice_id,
                    model_id=self.model_id,
                    output_format="mp3_44100_128",
                ),
            )

            # Collect audio bytes
            if isinstance(audio_iter, bytes):
                audio_data = audio_iter
            else:
                audio_data = b"".join(audio_iter)

            log.info("ElevenLabs TTS: synthesised %d bytes", len(audio_data))
            return audio_data

        except Exception as exc:
            log.error("ElevenLabs TTS failed: %s — falling back to Edge TTS", exc)
            fallback = EdgeTTS()
            return await fallback.synthesize(text)

    async def stream_audio(self, text: str) -> AsyncIterator[bytes]:
        """Stream audio chunks directly from ElevenLabs API."""
        try:
            from elevenlabs.client import ElevenLabs
            client = ElevenLabs(api_key=self.api_key)

            # This is synchronous blocking, so we run it in an executor and 
            # simulate streaming by yielding the chunks
            loop = asyncio.get_event_loop()
            audio_iter = await loop.run_in_executor(
                None,
                lambda: client.text_to_speech.convert(
                    text=text,
                    voice_id=self.voice_id,
                    model_id=self.model_id,
                    output_format="mp3_44100_128",
                ),
            )

            if isinstance(audio_iter, bytes):
                yield audio_iter
            else:
                for chunk in audio_iter:
                    yield chunk
                    await asyncio.sleep(0.01) # allow event loop to run

        except Exception as exc:
            log.error("ElevenLabs TTS streaming failed: %s — falling back", exc)
            fallback = EdgeTTS()
            async for chunk in fallback.stream_audio(text):
                yield chunk

    async def close(self) -> None:
        log.info("ElevenLabs TTS closed")


def create_tts_service() -> BaseTTS:
    """Factory: create the configured TTS service."""
    from config import TTS_PROVIDER, ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID, ELEVENLABS_MODEL, EDGE_TTS_VOICE

    if TTS_PROVIDER == "elevenlabs" and ELEVENLABS_API_KEY:
        return ElevenLabsTTS(
            api_key=ELEVENLABS_API_KEY,
            voice_id=ELEVENLABS_VOICE_ID,
            model_id=ELEVENLABS_MODEL,
        )
    else:
        return EdgeTTS(voice=EDGE_TTS_VOICE)

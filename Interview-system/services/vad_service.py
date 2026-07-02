"""
Phase 3 — Voice Activity Detection Service
AI Voice Interview System

Silero VAD for detecting when the candidate stops speaking.
Runs locally using the pre-trained Silero model (torch).

Usage:
    vad = SileroVAD()
    for chunk in audio_chunks:
        if vad.process_chunk(chunk):
            print("End of speech detected!")
            vad.reset()
"""

from __future__ import annotations

import logging
import time

import numpy as np
import torch

log = logging.getLogger("phase3.vad")


class SileroVAD:
    """
    End-of-speech detection using Silero VAD.

    Processes 16kHz audio chunks and detects when the speaker
    has been silent for longer than `silence_ms` milliseconds.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        silence_ms: int = 1500,
        sample_rate: int = 16000,
    ):
        self.threshold = threshold
        self.silence_ms = silence_ms
        self.sample_rate = sample_rate

        # Load Silero VAD model
        log.info("Loading Silero VAD model...")
        self.model, self.utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        self.model.eval()
        log.info("Silero VAD model loaded")

        # State tracking
        self._is_speaking = False
        self._silence_start: float | None = None
        self._speech_detected = False

    def process_chunk(self, audio_chunk: bytes) -> bool:
        """
        Process a raw PCM audio chunk (16-bit, 16kHz, mono).

        Returns True when end-of-speech is detected:
        - Speech was detected at some point
        - Silence has lasted longer than silence_ms

        The chunk should be 30-96ms of audio (480-1536 samples at 16kHz).
        """
        # Convert bytes to float tensor
        audio_np = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        audio_tensor = torch.from_numpy(audio_np)

        # Silero VAD expects specific chunk sizes: 256, 512, or 768 samples for 16kHz
        # Pad or truncate to 512 samples
        target_len = 512
        if len(audio_tensor) < target_len:
            audio_tensor = torch.nn.functional.pad(audio_tensor, (0, target_len - len(audio_tensor)))
        elif len(audio_tensor) > target_len:
            audio_tensor = audio_tensor[:target_len]

        # Get speech probability
        with torch.no_grad():
            speech_prob = self.model(audio_tensor, self.sample_rate).item()

        now = time.time()

        if speech_prob >= self.threshold:
            # Speech detected
            self._is_speaking = True
            self._speech_detected = True
            self._silence_start = None
            return False

        else:
            # Silence
            if self._speech_detected and self._is_speaking:
                # Transition from speech to silence
                if self._silence_start is None:
                    self._silence_start = now
                    self._is_speaking = False

            if self._silence_start is not None:
                silence_duration_ms = (now - self._silence_start) * 1000
                if silence_duration_ms >= self.silence_ms and self._speech_detected:
                    log.info(
                        "End-of-speech detected (%.0fms silence, threshold=%dms)",
                        silence_duration_ms, self.silence_ms,
                    )
                    return True

        return False

    def reset(self) -> None:
        """Reset state for the next utterance."""
        self._is_speaking = False
        self._silence_start = None
        self._speech_detected = False
        self.model.reset_states()

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking

    @property
    def speech_detected(self) -> bool:
        return self._speech_detected

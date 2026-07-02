"""
Phase 3 — Centralised Configuration
AI Voice Interview System

Loads all environment variables from .env file.
All services import config values from here.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
_env_path = Path(__file__).parent / ".env"
load_dotenv(_env_path)

# ── Groq (Phase 2 LLM) ───────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ── Deepgram STT (optional — fallback to local Whisper) ──────────────────────
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")

# ── ElevenLabs TTS (optional — fallback to edge-tts) ─────────────────────────
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # Rachel
ELEVENLABS_MODEL = os.getenv("ELEVENLABS_MODEL", "eleven_flash_v2_5")

# ── LiveKit (optional — for production WebRTC) ───────────────────────────────
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "")

# ── STT Configuration ────────────────────────────────────────────────────────
# "whisper" (local, free) or "deepgram" (API, paid)
STT_PROVIDER = os.getenv("STT_PROVIDER", "whisper")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")  # tiny, base, small, medium, large-v3
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")  # cpu or cuda

# ── TTS Configuration ────────────────────────────────────────────────────────
# "edge" (free, unlimited) or "elevenlabs" (API, paid)
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "edge")
EDGE_TTS_VOICE = os.getenv("EDGE_TTS_VOICE", "en-US-JennyNeural")  # Natural female voice

# ── VAD Configuration ────────────────────────────────────────────────────────
VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", "0.5"))
VAD_SILENCE_MS = int(os.getenv("VAD_SILENCE_MS", "1500"))  # 1.5s silence = end of speech

# ── Server Configuration ─────────────────────────────────────────────────────
SERVER_HOST = os.getenv("SERVER_HOST", "127.0.0.1")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# ── Audio Configuration ──────────────────────────────────────────────────────
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_CHUNK_MS = 30  # 30ms chunks for VAD

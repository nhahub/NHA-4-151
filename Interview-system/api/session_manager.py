"""
Phase 3 — Session Manager
AI Voice Interview System

Bridges the WebSocket audio layer with Phase 2's LangGraph engine.
Each interview session has its own graph instance, STT/VAD/TTS services,
and audio buffer.

Usage:
    manager = SessionManager()
    session = await manager.create_session("EXP-SOFT-SEN")
    result = await manager.process_answer(session.session_id, "My answer is...")
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from langgraph.checkpoint.memory import MemorySaver

from core.graph import build_graph
from services.stt_service import BaseSTT, create_stt_service
from services.tts_service import BaseTTS, create_tts_service
from services.vad_service import SileroVAD

log = logging.getLogger("phase3.session")


class SessionStatus(str, Enum):
    INITIALIZING = "initializing"
    READY = "ready"          # Question asked, waiting for candidate
    LISTENING = "listening"  # Receiving audio
    PROCESSING = "processing"  # Evaluating answer
    SPEAKING = "speaking"    # Playing TTS
    COMPLETE = "complete"    # Interview done


@dataclass
class InterviewSession:
    """A single interview session with its own graph + audio state."""
    session_id: str
    role_id: str
    graph: Any  # CompiledGraph
    config: dict
    status: SessionStatus = SessionStatus.INITIALIZING
    state: dict = field(default_factory=dict)
    audio_buffer: bytearray = field(default_factory=bytearray)
    current_question_text: str = ""
    current_question_audio: bytes = b""
    turn: int = 0
    report: dict | None = None
    vad: SileroVAD | None = None
    barge_in_triggered: bool = False
    ready_time: float | None = None
    latency_ms: float = 0.0


class SessionManager:
    """
    Manages concurrent interview sessions.
    Each session has its own LangGraph instance and audio pipeline.
    """

    def __init__(self):
        self._sessions: dict[str, InterviewSession] = {}
        self._stt: BaseSTT | None = None
        self._tts: BaseTTS | None = None
        self._vad_template: dict = {}
        log.info("SessionManager initialised")

    async def _ensure_services(self) -> None:
        """Lazy-init shared STT and TTS services."""
        if self._stt is None:
            self._stt = create_stt_service()
        if self._tts is None:
            self._tts = create_tts_service()

    async def create_session(
        self,
        role_id: str = "",
        jd_profile: dict | None = None,
        cv_profile: dict | None = None,
    ) -> InterviewSession:
        """
        Create a new interview session.

        Modes:
          A) JD/CV-driven: pass jd_profile (and optionally cv_profile)
          B) Legacy: pass role_id (e.g. "EXP-SOFT-SEN")
        """
        await self._ensure_services()

        session_id = str(uuid.uuid4())
        graph = build_graph()
        config = {"configurable": {"thread_id": session_id}}

        session = InterviewSession(
            session_id=session_id,
            role_id=role_id,
            graph=graph,
            config=config,
            vad=SileroVAD(),
        )
        self._sessions[session_id] = session

        mode = "JD" if jd_profile else "role_id"
        log.info("Creating session %s (%s mode)", session_id[:8], mode)

        # Build initial state
        initial_state = {
            "role_id": role_id,
            "session_id": session_id,
            "jd_profile": jd_profile,
            "cv_profile": cv_profile,
        }
        state = await asyncio.to_thread(graph.invoke, initial_state, config)
        session.state = state

        # Extract the first question
        question = state.get("current_question", {})
        question_text = question.get("question_text", "Tell me about yourself.")
        session.current_question_text = question_text
        session.turn = state.get("turn", 1)

        # Generate TTS audio for greeting + first question
        role_title = state.get("role_title", "this")
        greeting = (
            f"Welcome to your interview for the {role_title} position. "
            f"I'll be asking you a series of questions. "
            f"Take your time to answer each question thoroughly. "
            f"Let's begin. {question_text}"
        )

        try:
            session.current_question_audio = await self._tts.synthesize(greeting)
        except Exception as exc:
            log.error("TTS failed for greeting: %s", exc)
            session.current_question_audio = b""

        session.status = SessionStatus.READY
        session.ready_time = __import__("time").time()
        log.info(
            "Session %s ready: role='%s', first Q='%s'",
            session_id[:8], role_title, question_text[:60],
        )

        return session

    async def process_audio_chunk(self, session_id: str, audio_chunk: bytes) -> dict:
        """
        Process an audio chunk from the WebSocket.
        Appends to buffer. VAD runs on each chunk.

        Returns:
            {"status": "listening"} — still collecting audio
            {"status": "processing", "transcript": "..."} — end of speech, processing
            {"status": "error", "message": "..."} — error
        """
        session = self._sessions.get(session_id)
        if not session:
            return {"status": "error", "message": "Session not found"}

        session.audio_buffer.extend(audio_chunk)
        
        # Track latency if this is the first chunk after question was ready
        if session.status == SessionStatus.READY and session.ready_time is not None:
            import time
            session.latency_ms = (time.time() - session.ready_time) * 1000
            log.info("Voice latency for turn %d: %.0f ms", session.turn, session.latency_ms)
        
        # Run VAD to detect barge-in (user talking while AI is talking/ready)
        barge_in = False
        if session.vad:
            is_end = session.vad.process_chunk(audio_chunk)
            # If user just started speaking and we haven't triggered barge_in yet
            if session.vad.is_speaking and not session.barge_in_triggered:
                if session.status in (SessionStatus.READY, SessionStatus.SPEAKING):
                    session.barge_in_triggered = True
                    barge_in = True
            
            # If end of speech detected via VAD automatically
            if is_end:
                return {"status": "end_of_speech"}

        session.status = SessionStatus.LISTENING

        return {"status": "listening", "barge_in": barge_in}

    async def process_answer(self, session_id: str, transcript: str) -> dict:
        """
        Process a complete candidate answer (from STT).

        1. Inject answer into graph state
        2. Resume graph (answer_eval → router → next question or report)
        3. Generate TTS for next question (if any)

        Returns:
            {
                "status": "next_question" | "complete",
                "turn": int,
                "phase": str,
                "score": int,           # score for the answer just evaluated
                "rationale": str,       # why that score
                "question_text": str,   # next question (if not complete)
                "question_audio": bytes # TTS audio (if not complete)
                "report": dict          # final report (if complete)
            }
        """
        session = self._sessions.get(session_id)
        if not session:
            return {"status": "error", "message": "Session not found"}

        # Track text latency BEFORE changing status (otherwise check is dead code)
        if session.status == SessionStatus.READY and session.ready_time is not None:
            import time
            session.latency_ms = (time.time() - session.ready_time) * 1000
            log.info("Text latency for turn %d: %.0f ms", session.turn, session.latency_ms)

        session.status = SessionStatus.PROCESSING
        log.info("Session %s: processing answer (turn %d)", session_id[:8], session.turn)

        # Get current question info
        question = session.state.get("current_question", {})

        # Inject candidate answer into graph state
        await asyncio.to_thread(
            session.graph.update_state,
            session.config,
            {
                "transcript": [{
                    "role": "candidate",
                    "content": transcript,
                    "turn": session.turn,
                    "question_id": question.get("question_id", ""),
                    "score": None,
                    "latency_ms": session.latency_ms,
                }],
            },
        )

        # Resume graph — runs answer_eval → router → next node
        state = await asyncio.to_thread(session.graph.invoke, None, session.config)
        session.state = state

        # Extract evaluation results from transcript
        score = None
        rationale = ""
        for entry in reversed(state.get("transcript", [])):
            if entry.get("role") == "evaluation":
                score = entry.get("score")
                rationale = entry.get("content", "")
                break

        # Check if interview is complete
        report = state.get("report")
        if report:
            session.report = report
            session.status = SessionStatus.COMPLETE
            log.info(
                "Session %s: interview complete, overall=%.1f/10",
                session_id[:8], report.get("overall_score", 0),
            )
            return {
                "status": "complete",
                "turn": session.turn,
                "phase": state.get("phase", ""),
                "score": score,
                "rationale": rationale,
                "report": report,
            }

        # Get next question
        next_question = state.get("current_question", {})
        next_text = next_question.get("question_text", "")
        session.current_question_text = next_text
        session.turn = state.get("turn", session.turn + 1)

        # Generate TTS for next question
        session.status = SessionStatus.SPEAKING
        try:
            session.current_question_audio = await self._tts.synthesize(next_text)
        except Exception as exc:
            log.error("TTS failed: %s", exc)
            session.current_question_audio = b""

        session.status = SessionStatus.READY
        session.ready_time = __import__("time").time()
        session.barge_in_triggered = False
        if session.vad:
            session.vad.reset()

        log.info(
            "Session %s: turn %d, phase='%s', score=%s, next Q='%s'",
            session_id[:8], session.turn, state.get("phase"),
            score, next_text[:50],
        )

        # ── Enrich response with memory data ─────────────────────────────
        memory_summary = self._get_memory_summary(state)

        return {
            "status": "next_question",
            "turn": session.turn,
            "phase": state.get("phase", ""),
            "score": score,
            "rationale": rationale,
            "question_text": next_text,
            "source": next_question.get("source", "adaptive"),
            "memory": memory_summary,
        }

    async def transcribe_buffer(self, session_id: str) -> str:
        """Transcribe the accumulated audio buffer for a session."""
        session = self._sessions.get(session_id)
        if not session:
            return ""

        audio_data = bytes(session.audio_buffer)
        session.audio_buffer.clear()

        if len(audio_data) < 3200:  # less than 0.1s of audio
            return ""

        transcript = await self._stt.transcribe(audio_data)
        return transcript

    def get_session(self, session_id: str) -> InterviewSession | None:
        return self._sessions.get(session_id)

    def get_session_info(self, session_id: str) -> dict:
        """Get serialisable session info with memory metrics."""
        session = self._sessions.get(session_id)
        if not session:
            return {"error": "Session not found"}

        memory_summary = self._get_memory_summary(session.state)

        return {
            "session_id": session.session_id,
            "role_id": session.role_id,
            "status": session.status.value,
            "turn": session.turn,
            "phase": session.state.get("phase", ""),
            "role_title": session.state.get("role_title", ""),
            "seniority": session.state.get("seniority", ""),
            "max_turns": session.state.get("max_turns", 0),
            "current_question": session.current_question_text,
            "report": session.report,
            "memory": memory_summary,
        }

    @staticmethod
    def _get_memory_summary(state: dict) -> dict:
        """Extract a concise memory summary from state for API responses."""
        try:
            from intelligence.interview_memory import InterviewMemory
            memory = InterviewMemory.from_dict(state.get("interview_memory", {}))
            ap = memory.answer_patterns
            return {
                "avg_specificity": round(ap.avg_specificity, 1),
                "avg_realism": round(ap.avg_realism, 1),
                "total_claims": len(memory.factual_claims),
                "total_contradictions": len(memory.contradictions),
                "unresolved_contradictions": len(memory.get_unresolved_contradictions()),
                "vague_answers": ap.vague_count,
                "specific_answers": ap.specific_count,
                "skills_depth": {
                    skill: {
                        "questions": ds.questions_asked,
                        "deep_enough": ds.is_sufficiently_deep,
                        "has_real_project": ds.has_real_project,
                        "difficulty": ds.max_difficulty_reached,
                    }
                    for skill, ds in memory.depth_signals.items()
                },
            }
        except Exception:
            return {}

    async def end_session(self, session_id: str) -> dict:
        """Force-end a session and get the report."""
        session = self._sessions.get(session_id)
        if not session:
            return {"error": "Session not found"}

        if session.report:
            return session.report

        # Force report generation by maxing out turn count
        session.graph.update_state(
            session.config,
            {"turn": session.state.get("max_turns", 10)},
        )
        state = session.graph.invoke(None, session.config)
        session.state = state
        session.report = state.get("report", {})
        session.status = SessionStatus.COMPLETE

        return session.report

    async def cleanup(self) -> None:
        """Clean up all services."""
        if self._stt:
            await self._stt.close()
        if self._tts:
            await self._tts.close()
        self._sessions.clear()
        log.info("SessionManager cleaned up")

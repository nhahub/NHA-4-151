"""
cv_agent.api
=============
FastAPI application, session manager, request/response schemas, rate limiter.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Callable, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from cv_agent.cache import get_cache
from cv_agent.config import PipelineConfig, _FASTAPI_AVAILABLE, logger
from cv_agent.config import (
    _FAISS_AVAILABLE, _SENTENCE_AVAILABLE, _SKLEARN_AVAILABLE,
    _REPORTLAB_AVAILABLE, _MISTUNE_AVAILABLE,
)
from cv_agent.gpu_queue import _gpu_queue
from cv_agent.pipeline import run_pipeline
from cv_agent.schemas import PipelineResult, UserProfile


# ==============================================================================
# SESSION MANAGER
# ==============================================================================

class SessionStatus:
    PENDING   = "pending"
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"


class SessionRecord(BaseModel):
    session_id:    str
    status:        str = SessionStatus.PENDING
    created_at:    str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at:    str = Field(default_factory=lambda: datetime.now().isoformat())
    progress_msgs: List[str] = Field(default_factory=list)
    result:        Optional[PipelineResult] = None
    error:         Optional[str] = None


class SessionManager:
    """Thread-safe in-memory registry of active and completed sessions."""

    def __init__(self, max_sessions: int = 1000) -> None:
        self._sessions: Dict[str, SessionRecord] = {}
        self._lock = threading.Lock()
        self._max = max_sessions

    def create(self, session_id: str) -> SessionRecord:
        rec = SessionRecord(session_id=session_id)
        with self._lock:
            if len(self._sessions) >= self._max:
                completed = [
                    (k, v) for k, v in self._sessions.items()
                    if v.status in (SessionStatus.COMPLETED, SessionStatus.FAILED)
                ]
                if completed:
                    oldest = min(completed, key=lambda x: x[1].updated_at)
                    del self._sessions[oldest[0]]
            self._sessions[session_id] = rec
        return rec

    def get(self, session_id: str) -> Optional[SessionRecord]:
        with self._lock:
            return self._sessions.get(session_id)

    def update_status(self, session_id: str, status: str, msg: Optional[str] = None) -> None:
        with self._lock:
            rec = self._sessions.get(session_id)
            if rec:
                rec.status = status
                rec.updated_at = datetime.now().isoformat()
                if msg:
                    rec.progress_msgs.append(msg)

    def complete(self, session_id: str, result: PipelineResult) -> None:
        with self._lock:
            rec = self._sessions.get(session_id)
            if rec:
                rec.status = SessionStatus.COMPLETED
                rec.result = result
                rec.updated_at = datetime.now().isoformat()

    def fail(self, session_id: str, error: str) -> None:
        with self._lock:
            rec = self._sessions.get(session_id)
            if rec:
                rec.status = SessionStatus.FAILED
                rec.error = error
                rec.updated_at = datetime.now().isoformat()

    def list_sessions(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [
                {
                    "session_id": v.session_id, "status": v.status,
                    "created_at": v.created_at, "updated_at": v.updated_at,
                }
                for v in self._sessions.values()
            ]


_session_manager = SessionManager()


# ==============================================================================
# FASTAPI APPLICATION
# ==============================================================================

if _FASTAPI_AVAILABLE:
    from fastapi import FastAPI, HTTPException, Request, status
    from fastapi.middleware.cors import CORSMiddleware

    # ------------------------------------------------------------------
    # Request / Response schemas
    # ------------------------------------------------------------------

    class GenerateRequest(BaseModel):
        full_name:        str = Field(..., min_length=2, max_length=120)
        target_role:      str = Field(..., min_length=2, max_length=120)
        target_industry:  str = ""
        years_experience: str = ""
        summary:          str = ""
        tone:             Literal["professional", "creative", "technical"] = "professional"
        email:            str = ""
        phone:            str = ""
        linkedin:         str = ""
        education:        List[str] = Field(default_factory=list)
        skills:           List[str] = Field(..., min_length=1)
        experiences:      List[str] = Field(..., min_length=1)
        achievements:     List[str] = Field(default_factory=list)
        certifications:   List[str] = Field(default_factory=list)
        job_description:  str = ""
        parsed_resume:    str = ""
        max_iterations:   Optional[int] = Field(default=None, ge=1, le=20)
        score_threshold:  Optional[int] = Field(default=None, ge=50, le=100)
        num_candidates:   Optional[int] = Field(default=None, ge=1, le=5)
        session_id:       str = ""

        @field_validator("skills", "experiences")
        @classmethod
        def non_empty_list(cls, v: List[str]) -> List[str]:
            if not v:
                raise ValueError("must contain at least one item")
            return v

    class GenerateResponse(BaseModel):
        session_id: str
        status:     str
        message:    str

    class StatusResponse(BaseModel):
        session_id:    str
        status:        str
        created_at:    str
        updated_at:    str
        progress_msgs: List[str]
        error:         Optional[str] = None

    class ResultResponse(BaseModel):
        session_id:       str
        status:           str
        candidate_name:   str = ""
        target_role:      str = ""
        total_iterations: int = 0
        final_cv:         str = ""
        final_scores:     Optional[Dict[str, Any]] = None
        score_trajectory: List[float] = Field(default_factory=list)
        jd_keywords:      List[str]   = Field(default_factory=list)
        node_errors:      List[str]   = Field(default_factory=list)
        total_latency_ms: int = 0
        content_check:    Optional[Dict[str, Any]] = None
        error:            Optional[str] = None

    class HealthResponse(BaseModel):
        status:               str
        cache_stats:          Dict[str, int]
        active_sessions:      int
        gpu_queue_size:       int
        gpu_worker_alive:     bool = True
        gpu_worker_restarts:  int  = 0
        features:             Dict[str, bool]

    # ------------------------------------------------------------------
    # Thread pool for background pipeline execution
    # ------------------------------------------------------------------

    _pipeline_executor = ThreadPoolExecutor(
        max_workers=int(os.getenv("PIPELINE_WORKERS", "2")),
        thread_name_prefix="pipeline",
    )

    def _run_pipeline_background(
        session_id: str, profile: UserProfile,
        job_description: str, parsed_resume: str, cfg: PipelineConfig,
    ) -> None:
        _session_manager.update_status(session_id, SessionStatus.RUNNING)

        def _cb(msg: str) -> None:
            _session_manager.update_status(session_id, SessionStatus.RUNNING, msg)

        try:
            result = run_pipeline(
                profile=profile, job_description=job_description,
                parsed_resume=parsed_resume, config=cfg,
                session_id=session_id, status_callback=_cb,
            )
            _session_manager.complete(session_id, result)
        except Exception as exc:
            err_msg = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            logger.error("Pipeline failed session=%s: %s", session_id, err_msg)
            _session_manager.fail(session_id, err_msg)

    # ------------------------------------------------------------------
    # App factory
    # ------------------------------------------------------------------

    def create_app(default_cfg: Optional[PipelineConfig] = None) -> "FastAPI":
        _cfg = default_cfg or PipelineConfig()

        app = FastAPI(
            title="CV Agent SaaS API",
            description="Agentic CV Generation — LangGraph + Ensemble Judges + FastAPI",
            version="7.0.0",
        )

        # CORS — FIX M-2a
        _raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8080")
        _allowed_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]
        app.add_middleware(
            CORSMiddleware,
            allow_origins=_allowed_origins,
            allow_credentials=False,
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=["Content-Type", "Authorization", "X-Request-ID"],
        )

        # Rate limiter — FIX M-2b
        _RATE_LIMIT_RPM = int(os.getenv("RATE_LIMIT_RPM", "20"))
        _rate_buckets: Dict[str, List[float]] = {}
        _rate_lock = threading.Lock()

        async def _check_rate_limit(request: Request) -> None:
            if _RATE_LIMIT_RPM <= 0:
                return
            client_ip = (request.headers.get("X-Forwarded-For") or
                         (request.client.host if request.client else "unknown"))
            now = time.monotonic()
            window = 60.0
            with _rate_lock:
                timestamps = _rate_buckets.get(client_ip, [])
                timestamps = [t for t in timestamps if now - t < window]
                if len(timestamps) >= _RATE_LIMIT_RPM:
                    _rate_buckets[client_ip] = timestamps
                    raise HTTPException(
                        status_code=429,
                        detail=f"Rate limit exceeded: max {_RATE_LIMIT_RPM} requests/min.",
                        headers={"Retry-After": "60"},
                    )
                timestamps.append(now)
                _rate_buckets[client_ip] = timestamps

        @app.middleware("http")
        async def _request_timing(request: Request, call_next: Any) -> Any:
            t0 = time.perf_counter()
            res = await call_next(request)
            ms = int((time.perf_counter() - t0) * 1000)
            res.headers["X-Response-Time-Ms"] = str(ms)
            logger.info("HTTP %s %s → %d (%dms)", request.method, request.url.path, res.status_code, ms)
            return res

        # ── Endpoints ──

        @app.get("/health", response_model=HealthResponse, tags=["system"])
        async def health() -> HealthResponse:
            gpu_alive = _gpu_queue.is_alive
            gpu_q_size = _gpu_queue._q.qsize()
            health_status = "ok" if gpu_alive else "degraded"
            if not gpu_alive:
                logger.warning("/health: GPU worker is not alive — status=degraded")
            return HealthResponse(
                status=health_status,
                cache_stats=get_cache(_cfg).stats(),
                active_sessions=len(_session_manager.list_sessions()),
                gpu_queue_size=gpu_q_size,
                gpu_worker_alive=gpu_alive,
                gpu_worker_restarts=_gpu_queue._restart_count,
                features={
                    "faiss": _FAISS_AVAILABLE,
                    "sentence_transformers": _SENTENCE_AVAILABLE,
                    "sklearn": _SKLEARN_AVAILABLE,
                    "pdf_export": _REPORTLAB_AVAILABLE,
                    "mistune": _MISTUNE_AVAILABLE,
                },
            )

        @app.post("/generate", response_model=GenerateResponse, status_code=status.HTTP_202_ACCEPTED, tags=["pipeline"])
        async def generate(req: GenerateRequest, request: Request) -> GenerateResponse:
            await _check_rate_limit(request)
            profile = UserProfile(
                full_name=req.full_name, target_role=req.target_role,
                target_industry=req.target_industry, years_experience=req.years_experience,
                summary=req.summary, tone=req.tone, email=req.email,
                phone=req.phone, linkedin=req.linkedin, education=req.education,
                skills=req.skills, experiences=req.experiences,
                achievements=req.achievements, certifications=req.certifications,
            )
            if not profile.is_complete():
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Profile must include full_name, target_role, skills, and experiences.",
                )

            sid = req.session_id or str(uuid.uuid4())[:12]
            _session_manager.create(sid)

            req_cfg = PipelineConfig()
            req_cfg.db_path = _cfg.db_path
            req_cfg.output_dir = _cfg.output_dir
            if req.max_iterations is not None:
                req_cfg.max_iterations = req.max_iterations
            if req.score_threshold is not None:
                req_cfg.score_threshold = req.score_threshold
            if req.num_candidates is not None:
                req_cfg.num_candidates = req.num_candidates

            loop = asyncio.get_event_loop()
            loop.run_in_executor(
                _pipeline_executor, _run_pipeline_background,
                sid, profile, req.job_description, req.parsed_resume, req_cfg,
            )

            return GenerateResponse(
                session_id=sid, status=SessionStatus.PENDING,
                message=f"Pipeline started. Poll GET /status/{sid} for progress.",
            )

        @app.get("/status/{session_id}", response_model=StatusResponse, tags=["pipeline"])
        async def get_status(session_id: str) -> StatusResponse:
            rec = _session_manager.get(session_id)
            if rec is None:
                raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
            return StatusResponse(
                session_id=rec.session_id, status=rec.status,
                created_at=rec.created_at, updated_at=rec.updated_at,
                progress_msgs=rec.progress_msgs, error=rec.error,
            )

        @app.get("/result/{session_id}", response_model=ResultResponse, tags=["pipeline"])
        async def get_result(session_id: str) -> ResultResponse:
            rec = _session_manager.get(session_id)
            if rec is None:
                raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
            if rec.status == SessionStatus.FAILED:
                return ResultResponse(session_id=session_id, status=rec.status, error=rec.error)
            if rec.status != SessionStatus.COMPLETED:
                raise HTTPException(
                    status_code=status.HTTP_202_ACCEPTED,
                    detail=f"Session '{session_id}' is still {rec.status}.",
                )
            r = rec.result
            assert r is not None
            return ResultResponse(
                session_id=r.session_id, status=SessionStatus.COMPLETED,
                candidate_name=r.candidate_name, target_role=r.target_role,
                total_iterations=r.total_iterations, final_cv=r.final_cv,
                final_scores=r.final_scores.model_dump() if r.final_scores else None,
                score_trajectory=r.score_trajectory, jd_keywords=r.jd_keywords,
                node_errors=r.node_errors, total_latency_ms=r.total_latency_ms,
                content_check=r.content_check.model_dump() if r.content_check else None,
            )

        @app.get("/sessions", tags=["system"])
        async def list_sessions() -> Dict[str, Any]:
            return {"sessions": _session_manager.list_sessions()}

        @app.get("/cache/stats", tags=["system"])
        async def cache_stats() -> Dict[str, int]:
            return get_cache(_cfg).stats()

        @app.delete("/cache", tags=["system"])
        async def clear_cache() -> Dict[str, str]:
            get_cache(_cfg).clear()
            return {"message": "Cache cleared."}

        return app

    # Instantiate the global app
    app = create_app()

else:
    app = None  # type: ignore[assignment]

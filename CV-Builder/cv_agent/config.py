"""
cv_agent.config
===============
Pipeline configuration, structured logging, and feature-flag detection.

This module is a **leaf node** — it has no intra-package imports,
making it safe to import from any other module without circular dependency risk.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict


# ==============================================================================
# FEATURE FLAGS — graceful third-party import detection
# ==============================================================================

_SKLEARN_AVAILABLE    = False
_FAISS_AVAILABLE      = False
_SENTENCE_AVAILABLE   = False
_REPORTLAB_AVAILABLE  = False
_MISTUNE_AVAILABLE    = False
_PDFPLUMBER_AVAILABLE = False
_FASTAPI_AVAILABLE    = False

try:
    import sklearn; _SKLEARN_AVAILABLE = True  # noqa: E702
except ImportError:
    pass

try:
    import faiss; _FAISS_AVAILABLE = True  # noqa: E702,F401
except ImportError:
    pass

try:
    from sentence_transformers import SentenceTransformer; _SENTENCE_AVAILABLE = True  # noqa: E702,F401
except ImportError:
    pass

try:
    import reportlab; _REPORTLAB_AVAILABLE = True  # noqa: E702,F401
except ImportError:
    pass

try:
    import mistune; _MISTUNE_AVAILABLE = True  # noqa: E702,F401
except ImportError:
    pass

try:
    import pdfplumber; _PDFPLUMBER_AVAILABLE = True  # noqa: E702,F401
except ImportError:
    pass

try:
    from fastapi import FastAPI, BackgroundTasks, HTTPException, Request, status  # noqa: F401
    from fastapi.middleware.cors import CORSMiddleware  # noqa: F401
    from fastapi.responses import JSONResponse  # noqa: F401
    import uvicorn  # noqa: F401
    _FASTAPI_AVAILABLE = True
except ImportError:
    pass

# Required dependencies — fail fast if missing
try:
    from pydantic import BaseModel, Field, ValidationError, field_validator  # noqa: F401
except ImportError as _e:
    raise ImportError("pydantic>=2.0 required: pip install pydantic>=2.0") from _e

try:
    from langgraph.graph import StateGraph, START, END  # noqa: F401
except ImportError as _e:
    raise ImportError("langgraph required: pip install langgraph>=0.2") from _e

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ==============================================================================
# STRUCTURED LOGGING — JSON formatter
# ==============================================================================

class _JSONFormatter(logging.Formatter):
    """Emit log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        log_obj: Dict[str, Any] = {
            "ts":      self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
            "level":   record.levelname,
            "logger":  record.name,
            "msg":     record.getMessage(),
        }
        if record.exc_info:
            log_obj["exc"] = self.formatException(record.exc_info)
        for key in ("session_id", "node", "duration_ms", "score", "iteration"):
            if hasattr(record, key):
                log_obj[key] = getattr(record, key)
        return json.dumps(log_obj, ensure_ascii=False)


def _setup_logging(debug: bool = False, json_logs: bool = True) -> logging.Logger:
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if debug else logging.INFO)
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JSONFormatter() if json_logs else logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    ))
    root.addHandler(handler)
    return logging.getLogger("cv_agent")


_DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
_JSON_LOGS  = os.getenv("JSON_LOGS",  "true").lower()  == "true"
logger      = _setup_logging(debug=_DEBUG_MODE, json_logs=_JSON_LOGS)


# ==============================================================================
# PIPELINE CONFIGURATION
# ==============================================================================

@dataclass
class PipelineConfig:
    """All tunable knobs. Override via environment variables or pass directly."""

    judge_base:        str   = field(default_factory=lambda: os.getenv("JUDGE_BASE_MODEL",   "OsamaHayba/qwen-ats-merged-stage1"))
    judge_adapter:     str   = field(default_factory=lambda: os.getenv("JUDGE_ADAPTER_PATH", "OsamaHayba/cv-analysis-final-stage2"))
    writer_model:      str   = field(default_factory=lambda: os.getenv("WRITER_MODEL",       "Qwen/Qwen2.5-7B-Instruct"))
    hr_judge_model:    str   = field(default_factory=lambda: os.getenv("HR_JUDGE_MODEL",     "Qwen/Qwen2.5-7B-Instruct"))
    hf_token:          str   = field(default_factory=lambda: os.getenv("HUGGINGFACE_TOKEN",  ""))

    writer_max_tokens: int   = field(default_factory=lambda: int(os.getenv("WRITER_MAX_TOKENS", "3000")))
    judge_max_tokens:  int   = field(default_factory=lambda: int(os.getenv("JUDGE_MAX_TOKENS",  "1200")))
    num_candidates:    int   = field(default_factory=lambda: int(os.getenv("NUM_CANDIDATES",    "3")))

    max_iterations:    int   = field(default_factory=lambda: int(os.getenv("MAX_ITERATIONS",    "5")))
    score_threshold:   int   = field(default_factory=lambda: int(os.getenv("SCORE_THRESHOLD",   "82")))
    plateau_threshold: float = field(default_factory=lambda: float(os.getenv("PLATEAU_THRESHOLD", "2.0")))
    stagnation_limit:  int   = field(default_factory=lambda: int(os.getenv("STAGNATION_LIMIT",  "2")))

    ats_weight:        float = field(default_factory=lambda: float(os.getenv("ATS_WEIGHT",  "0.40")))
    hr_weight:         float = field(default_factory=lambda: float(os.getenv("HR_WEIGHT",   "0.35")))
    rule_weight:       float = field(default_factory=lambda: float(os.getenv("RULE_WEIGHT", "0.25")))

    load_in_4bit:      bool  = field(default_factory=lambda: os.getenv("LOAD_IN_4BIT", "true").lower() == "true")
    device:            str   = field(default_factory=lambda: os.getenv("DEVICE", "auto"))

    db_path:           str   = field(default_factory=lambda: os.getenv("SESSION_DB",  "cv_sessions.db"))
    output_dir:        Path  = field(default_factory=lambda: Path(os.getenv("OUTPUT_DIR", "output")))
    log_dir:           Path  = field(default_factory=lambda: Path(os.getenv("LOG_DIR",    "logs")))
    ontology_path:     str   = field(default_factory=lambda: os.getenv("ONTOLOGY_PATH", ""))

    # Cache
    cache_ttl_seconds: int   = field(default_factory=lambda: int(os.getenv("CACHE_TTL_SECONDS", "3600")))
    cache_max_size:    int   = field(default_factory=lambda: int(os.getenv("CACHE_MAX_SIZE",    "256")))

    # Concurrency
    writer_workers:    int   = field(default_factory=lambda: int(os.getenv("WRITER_WORKERS", "4")))

    # RAG
    embedding_model:   str   = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))

    judge_max_retries: int   = 3
    debug_mode:        bool  = field(default_factory=lambda: os.getenv("DEBUG_MODE", "false").lower() == "true")
    hallucination_penalty: float = field(default_factory=lambda: float(os.getenv("HALLUCINATION_PENALTY", "15.0")))

    # CV Template
    cv_template:       str   = field(default_factory=lambda: os.getenv("CV_TEMPLATE", "classic"))

    # Content Checker — inline fixing
    # Minimum number of issues before an LLM pass is used to apply fixes (rule-based fixes always run).
    content_fix_llm_threshold: int  = field(default_factory=lambda: int(os.getenv("CONTENT_FIX_LLM_THRESHOLD", "3")))
    # Master toggle: set to false to disable inline content fixing (reverts to post-pipeline report only).
    enable_inline_content_fix: bool = field(default_factory=lambda: os.getenv("ENABLE_INLINE_CONTENT_FIX", "true").lower() == "true")

    def __post_init__(self) -> None:
        """Validate that the three ensemble judge weights sum to 1.0."""
        total = self.ats_weight + self.hr_weight + self.rule_weight
        if abs(total - 1.0) >= 1e-6:
            raise AssertionError(
                f"Ensemble weights must sum to 1.0, got {total:.6f} "
                f"(ats_weight={self.ats_weight}, hr_weight={self.hr_weight}, "
                f"rule_weight={self.rule_weight})"
            )

"""
cv_agent — Agentic CV Builder v7.0
===================================
Modular package for LangGraph-based CV generation with ensemble judges,
hallucination guard, and FastAPI serving layer.

Public API::

    from cv_agent import run_pipeline, UserProfile, PipelineConfig
    from cv_agent import HallucinationGuard, GPUQueue
    from cv_agent import app  # FastAPI application (or None if FastAPI absent)
"""

from cv_agent.config import PipelineConfig, logger
from cv_agent.schemas import (
    UserProfile, JudgeOutput, EnsembleResult, JDContext,
    GuardResult, LoopDecision, IterationRecord,
    CVAgentState, PipelineResult,
)
from cv_agent.cache import LRUCache, get_cache
from cv_agent.gpu_queue import GPUQueue, _gpu_queue
from cv_agent.model_manager import ModelManager
from cv_agent.hallucination_guard import HallucinationGuard
from cv_agent.judges import RuleJudge, run_ensemble
from cv_agent.memory import MemoryModule
from cv_agent.rag import RAGModule
from cv_agent.routing import adaptive_decide
from cv_agent.pipeline import run_pipeline, build_graph
from cv_agent.file_parsing import parse_resume_bytes, parse_resume_file
from cv_agent.api import app, SessionManager, SessionStatus

__all__ = [
    "PipelineConfig", "logger",
    "UserProfile", "JudgeOutput", "EnsembleResult", "JDContext",
    "GuardResult", "LoopDecision", "IterationRecord",
    "CVAgentState", "PipelineResult",
    "LRUCache", "get_cache",
    "GPUQueue", "_gpu_queue",
    "ModelManager",
    "HallucinationGuard",
    "RuleJudge", "run_ensemble",
    "MemoryModule", "RAGModule",
    "adaptive_decide",
    "run_pipeline", "build_graph",
    "parse_resume_bytes", "parse_resume_file",
    "app", "SessionManager", "SessionStatus",
]

"""
Phase 2 — LLM Configuration
AI Voice Interview System

Centralised LLM factory using Groq for fast inference.
All LangGraph nodes that need an LLM import from here.

Requires:
    export GROQ_API_KEY="gsk_..."
    pip install langchain-groq
"""

from __future__ import annotations

import os
import logging

from langchain_groq import ChatGroq

log = logging.getLogger("phase2.llm")

# ── Default models ────────────────────────────────────────────────────────────
# Groq model catalogue (fast inference):
#   - llama-3.3-70b-versatile   → best quality, good for evaluation
#   - llama-3.1-8b-instant      → fastest, good for simple tasks
#   - mixtral-8x7b-32768        → 32k context, good for reports

EVAL_MODEL   = os.getenv("EVAL_MODEL",   "llama-3.3-70b-versatile")
REPORT_MODEL = os.getenv("REPORT_MODEL", "llama-3.3-70b-versatile")


def get_eval_llm() -> ChatGroq:
    """
    LLM for answer evaluation — needs structured output + consistency.
    Low temperature for reproducible scoring.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        log.warning("GROQ_API_KEY not set — LLM calls will fail")
    return ChatGroq(
        model=EVAL_MODEL,
        temperature=0.1,
        max_tokens=1024,
        api_key=api_key,
    )


def get_report_llm() -> ChatGroq:
    """
    LLM for report generation — needs longer output for summaries.
    Slightly higher temperature for more natural language.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        log.warning("GROQ_API_KEY not set — LLM calls will fail")
    return ChatGroq(
        model=REPORT_MODEL,
        temperature=0.3,
        max_tokens=4096,
        api_key=api_key,
    )

"""
cv_agent.utils
==============
Shared utility functions used across the package.

Centralises regex helpers, JSON parsing, timing decorators, and hash functions
so they are defined once and imported everywhere — eliminates duplicate code.
"""

from __future__ import annotations

import json
import re
import time
from functools import wraps
from hashlib import sha256
from typing import Any, Callable, Dict

from cv_agent.config import logger


# ==============================================================================
# TEXT NORMALISATION
# ==============================================================================

def normalise(text: str) -> str:
    """Strip non-alphanumeric characters and lowercase for comparison."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


# ==============================================================================
# ROBUST JSON PARSING
# ==============================================================================

def parse_json_robust(raw: str) -> dict:
    """
    Parse a JSON string with multiple fallback strategies.

    Handles: markdown fences, trailing text, truncated JSON, json_repair library.
    """
    cleaned = re.sub(r"```(?:json)?", "", raw, flags=re.IGNORECASE).replace("```", "").strip()

    for candidate in (cleaned, cleaned.split("\n\n")[0]):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
        blob = match.group()
        for end in range(len(blob), 0, -1):
            try:
                return json.loads(blob[:end])
            except json.JSONDecodeError:
                continue

    try:
        from json_repair import repair_json  # type: ignore[import]
        repaired = repair_json(raw)
        if repaired:
            return json.loads(repaired)
    except (ImportError, json.JSONDecodeError, Exception):
        pass

    logger.warning("parse_json_robust: all strategies exhausted for: %.120s", raw)
    return {}


# ==============================================================================
# TIMING DECORATOR — node-level observability
# ==============================================================================

def timed_node(node_name: str) -> Callable:
    """Decorator that logs node start/end with duration in ms."""
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            t0  = time.perf_counter()
            sid = ""
            if args and hasattr(args[0], "session_id"):
                sid = args[0].session_id
            extra = {"node": node_name, "session_id": sid}
            logger.info("NODE_START", extra=extra)
            try:
                result = fn(*args, **kwargs)
                ms = int((time.perf_counter() - t0) * 1000)
                logger.info("NODE_END", extra={**extra, "duration_ms": ms})
                return result
            except Exception as exc:
                ms = int((time.perf_counter() - t0) * 1000)
                logger.error(
                    "NODE_ERROR: %s", exc,
                    extra={**extra, "duration_ms": ms},
                    exc_info=True,
                )
                raise
        return wrapper
    return decorator


# ==============================================================================
# HASH HELPERS
# ==============================================================================

def jd_hash(jd_text: str) -> str:
    """Short SHA-256 hash of job description text for cache keys."""
    return sha256(jd_text.encode()).hexdigest()[:16]


# ==============================================================================
# MARKDOWN-TO-REPORTLAB HELPER
# ==============================================================================

def md_to_rl(text: str) -> str:
    """Convert basic markdown inline formatting to ReportLab XML tags."""
    # 1) Extract markdown links BEFORE escaping (they contain < > & in URLs)
    _link_placeholders: list = []
    def _save_link(m):
        idx = len(_link_placeholders)
        link_text = m.group(1)
        url = m.group(2)
        _link_placeholders.append((link_text, url))
        return f"__LINK_{idx}__"
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', _save_link, text)

    # 2) Escape XML entities
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # 3) Restore links as ReportLab <a> tags
    for idx, (link_text, url) in enumerate(_link_placeholders):
        safe_text = link_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        text = text.replace(f"__LINK_{idx}__", f'<a href="{url}" color="blue">{safe_text}</a>')

    # 4) Convert bare URLs to clickable links (after entity escaping)
    text = re.sub(
        r'(?<!["\'>=/a-zA-Z])(https?://[^\s<>&;]+(?:\.[^\s<>&;]+)*)',
        r'<a href="\1" color="blue">\1</a>',
        text,
    )

    # 5) Bold / italic
    text = re.sub(r'\*\*\*(.+?)\*\*\*', r'<b><i>\1</i></b>', text)
    text = re.sub(r'\*\*(.+?)\*\*',     r'<b>\1</b>',         text)
    text = re.sub(r'\*(.+?)\*',         r'<i>\1</i>',          text)
    text = re.sub(r'_(.+?)_',           r'<i>\1</i>',          text)
    return text

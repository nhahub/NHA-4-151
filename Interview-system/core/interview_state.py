"""
Phase 2 — Interview State Definition
AI Voice Interview System

The InterviewState TypedDict is the single source of truth passed between
all LangGraph nodes. Every node reads from and writes partial updates to
this state.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class InterviewState(TypedDict):
    """
    Full interview session state — flows through the LangGraph.

    Convention: nodes return *partial* dicts containing only the keys
    they want to update.  LangGraph merges them automatically.
    """

    # ── Session identity ──────────────────────────────────────────────────
    role_id: str                    # e.g. "EXP-SOFT-SEN"
    session_id: str                 # UUID per interview session

    # ── Interview progress ────────────────────────────────────────────────
    phase: str                      # current phase: "Technical", "Behavioral", …
    turn: int                       # global turn counter (0-indexed)
    max_turns: int                  # hard cap from role_expectations

    # ── Skill tracking ────────────────────────────────────────────────────
    # {domain: {sub_skill: [score1, score2, ...]}}
    skill_scores: dict[str, dict[str, list[float]]]
    # {domain: {sub_skill: avg_score}}  — recomputed after each answer_eval
    coverage_map: dict[str, dict[str, float]]

    # ── Question chain state ──────────────────────────────────────────────
    chain_cursor: dict | None       # {chain_id, base_question_id, order, max_depth, scoring_threshold}
    current_question: dict | None   # the active question dict

    # answered question_ids — uses Annotated + operator.add so that
    # returning ["Q00005"] from a node *appends* rather than replaces
    answered_ids: Annotated[list[str], operator.add]

    # ── Transcript ────────────────────────────────────────────────────────
    # Each entry: {role: "interviewer"|"candidate", content: str,
    #              turn: int, question_id: str|None, score: float|None}
    transcript: Annotated[list[dict], operator.add]
    transcript_summary: str         # AI summary of older turns

    # ── Role context (loaded at session_init) ─────────────────────────────
    role_expectations: list[dict]   # from data_layer.get_role_expectations()
    role_title: str                 # human-readable role name
    seniority: str                  # "Junior", "Mid", "Senior"
    phases: list[str]               # ordered interview phases for this role
    pass_threshold_pct: float       # overall pass threshold (0-100)
    depth_threshold: int            # score threshold to advance to next phase

    # ── LLM interaction ───────────────────────────────────────────────────
    messages: list                  # LangChain message history for LLM calls

    # ── Final output ──────────────────────────────────────────────────────
    report: dict | None             # final evaluation report (populated by report_gen)

    # ── JD / CV driven interview (Phase 3) ────────────────────────────
    jd_profile: dict | None         # parsed JD: {title, required_skills, ...}
    cv_profile: dict | None         # parsed CV: {name, skills, experience, ...}
    jd_skills_tested: Annotated[list[str], operator.add]   # skills from JD that have been tested
    cv_claims_verified: Annotated[list[str], operator.add] # CV claims that have been probed
    pending_skills: list[str]       # skills still to be tested (ordered by priority)
    confidence_score: float         # AI's confidence it has enough data (0.0-1.0)

    # ── Soft Skills ───────────────────────────────────────────────────────
    soft_skills_scores: Annotated[list[dict], operator.add]  # per-answer soft skill evals

    # ── Topic Rotation Guard ──────────────────────────────────────────────
    consecutive_probes_on_skill: dict[str, int]  # {skill_name: consecutive_probe_count}

    # ── Phrasing Diversity (Anti-Repetition) ──────────────────────────────
    # Stores the last N question opener phrases (normalized: strip().lower())
    # so the LLM prompt can blacklist them and force structural variety.
    recent_openers: list[str]

    # ── Non-Linear Topic Flow ─────────────────────────────────────────────
    # Records the actual order in which skills were tested/revisited,
    # enabling the callback logic to know which skills to circle back to.
    skills_tested_order: list[str]

    # ── Coverage Dimensions ───────────────────────────────────────────────
    # Tracks how many questions tested each assessment dimension
    # Keys: "technical_depth", "business_understanding", "communication",
    #        "problem_solving", "real_world_decision_making"
    dimensions_covered: dict[str, int]

    # ── Interview Memory (Cross-Turn Intelligence) ────────────────────────
    # Serialized InterviewMemory object — tracks claims, contradictions,
    # depth signals, and answer patterns across the full interview
    interview_memory: dict

    # Detected contradictions between candidate answers (accumulated)
    contradiction_flags: Annotated[list[dict], operator.add]

    # Per-answer specificity and realism scores (accumulated)
    specificity_scores: Annotated[list[dict], operator.add]

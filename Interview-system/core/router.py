"""
Phase 2 — Conditional Router & Transition Nodes
AI Voice Interview System

Decision logic that determines the next step after answer evaluation:
  - chain_follow_up  → continue a question chain sequence
  - advance_phase    → move to the next interview phase
  - question_gen     → ask another question in current phase
  - report_gen       → end interview and generate report
"""

from __future__ import annotations

import logging
from typing import Literal

from langchain_core.messages import HumanMessage

from core.interview_state import InterviewState
from data_layer.phase1_data_layer import get_data_layer

log = logging.getLogger("phase2.router")


# ─────────────────────────────────────────────────────────────────────────────
# Conditional Router
# ─────────────────────────────────────────────────────────────────────────────

def interview_router(
    state: InterviewState,
) -> Literal["chain_follow_up", "advance_phase", "question_gen", "report_gen"]:
    """
    AI-controlled routing — LLM decides when the interview is complete.

    Decision order:
      1. Hard safety cap (max_turns)
      2. Chain follow-up (if active)
      3. LLM Confidence Evaluator — the AI decides if it knows enough
      4. Phase coverage check
      5. Default: keep questioning
    """
    from evaluation.confidence_evaluator import evaluate_confidence

    turn = state.get("turn", 0)
    max_turns = state.get("max_turns", 15)  # safety cap only
    chain_cursor = state.get("chain_cursor")
    coverage_map = state.get("coverage_map", {})
    phase = state.get("phase", "")
    phases = state.get("phases", [])
    depth_threshold = state.get("depth_threshold", 5)

    # ── 1. Hard safety cap ────────────────────────────────────────────────
    if turn >= max_turns:
        log.info("router: safety cap turn %d >= max %d -> report_gen", turn, max_turns)
        return "report_gen"

    # ── 2. Chain follow-up (always honor active chains) ───────────────────
    if chain_cursor:
        order = chain_cursor.get("order", 0)
        max_depth = chain_cursor.get("max_depth", 3)
        if order <= max_depth:
            base_qid = chain_cursor.get("base_question_id", "")
            log.info(
                "router: chain active (base=%s, order=%d/%d) -> chain_follow_up",
                base_qid, order, max_depth,
            )
            return "chain_follow_up"
        else:
            log.info("router: chain exhausted (order=%d > max=%d)", order, max_depth)

    # ── 3. LLM Confidence Evaluator — smart decision ─────────────────────
    #    The AI evaluates: do I know enough about this candidate?
    #    - Are all JD skills tested?
    #    - Were weak areas probed sufficiently?
    #    - Is there a consistent score pattern?
    #    - Would more questions change the hiring decision?
    if turn >= 3:  # minimum questions before AI can decide to stop
        try:
            conf_result = evaluate_confidence(state)
            confidence = conf_result.get("confidence", 0.0)
            reason = conf_result.get("reason", "")

            if conf_result.get("should_stop"):
                log.info(
                    "router: AI DECIDED to stop (confidence=%.2f, reason='%s') -> report_gen",
                    confidence, reason[:80],
                )
                return "report_gen"
            else:
                log.info(
                    "router: AI wants to continue (confidence=%.2f, reason='%s')",
                    confidence, reason[:80],
                )
                # If AI identified weak areas, we could prioritize those
                weak_areas = conf_result.get("weak_areas", [])
                if weak_areas:
                    log.info("router: weak areas to probe: %s", weak_areas)

        except Exception as exc:
            log.warning("router: confidence eval failed: %s — using fallbacks", exc)
            # Fallback: simple rule
            pending = state.get("pending_skills", [])
            if not pending and turn >= 5:
                log.info("router: fallback — no pending skills, turn=%d -> report_gen", turn)
                return "report_gen"

    # ── 4. Check phase coverage ───────────────────────────────────────────
    phase_domain = phase
    if phase_domain in coverage_map:
        domain_scores = coverage_map[phase_domain]
        if domain_scores:
            all_covered = all(
                score >= depth_threshold
                for score in domain_scores.values()
            )
            if all_covered:
                current_idx = phases.index(phase) if phase in phases else -1
                if current_idx < len(phases) - 1:
                    log.info(
                        "router: phase '%s' fully covered (all >= %d) -> advance_phase",
                        phase, depth_threshold,
                    )
                    return "advance_phase"
                else:
                    log.info("router: all phases completed -> report_gen")
                    return "report_gen"

    # ── 5. Default: more questions ────────────────────────────────────────
    log.info("router: continue in phase '%s' -> question_gen", phase)
    return "question_gen"


# ─────────────────────────────────────────────────────────────────────────────
# Chain Follow-Up Node
# ─────────────────────────────────────────────────────────────────────────────

def chain_follow_up(state: InterviewState) -> dict:
    """
    Load the next question in the current follow-up chain.

    Reads chain_cursor to find the next chain question, increments
    the cursor order, and returns it as the current_question.
    """
    dl = get_data_layer()
    chain_cursor = state.get("chain_cursor")
    turn = state.get("turn", 0)

    if not chain_cursor:
        log.warning("chain_follow_up: no chain_cursor — falling through")
        return {"chain_cursor": None}

    base_qid = chain_cursor["base_question_id"]
    order = chain_cursor["order"]
    max_depth = chain_cursor["max_depth"]

    log.info("chain_follow_up: base=%s, order=%d/%d", base_qid, order, max_depth)

    # Load the chain question at this order
    chain_q = dl.get_chain_question_at(base_qid, order)

    if not chain_q:
        log.info("chain_follow_up: no question at order %d — clearing chain", order)
        return {"chain_cursor": None}

    # Build the question dict
    question = {
        "question_id": chain_q.get("chain_id", f"CHAIN-{base_qid}-{order}"),
        "question_text": chain_q.get("question_text", ""),
        "domain": chain_q.get("domain", state.get("phase", "")),
        "sub_skill": chain_q.get("sub_skill", ""),
        "difficulty": chain_q.get("difficulty", "Medium"),
        "phase": state.get("phase", ""),
        "chain_type": chain_q.get("chain_type", "follow_up"),
    }

    # Advance chain cursor
    next_order = order + 1
    if next_order > max_depth:
        new_cursor = None  # chain exhausted after this question
    else:
        new_cursor = {
            **chain_cursor,
            "order": next_order,
        }

    interviewer_msg = question["question_text"]

    log.info(
        "chain_follow_up: asking chain Q (type=%s, difficulty=%s)",
        question["chain_type"], question["difficulty"],
    )

    return {
        "current_question": question,
        "chain_cursor": new_cursor,
        "turn": turn + 1,
        "messages": [HumanMessage(content=interviewer_msg)],
        "transcript": [
            {
                "role": "interviewer",
                "content": interviewer_msg,
                "turn": turn + 1,
                "question_id": question["question_id"],
                "score": None,
            }
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Advance Phase Node
# ─────────────────────────────────────────────────────────────────────────────

def advance_phase(state: InterviewState) -> dict:
    """
    Transition to the next interview phase.

    Clears the chain cursor and advances the phase pointer.
    """
    phases = state.get("phases", [])
    current_phase = state.get("phase", "")

    current_idx = phases.index(current_phase) if current_phase in phases else 0
    next_idx = current_idx + 1

    if next_idx < len(phases):
        next_phase = phases[next_idx]
        log.info("advance_phase: '%s' → '%s'", current_phase, next_phase)
    else:
        # This shouldn't happen (router checks), but safety fallback
        next_phase = current_phase
        log.warning("advance_phase: no more phases after '%s'", current_phase)

    return {
        "phase": next_phase,
        "chain_cursor": None,
    }

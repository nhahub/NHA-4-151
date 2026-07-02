"""
Confidence Evaluator — LLM-based interview termination decision.

Instead of using simple turn counts, the AI evaluates whether it has
gathered enough information about the candidate to make a confident
hiring decision. It considers:
  - Coverage of JD required skills
  - Depth of probing per skill (were follow-ups done?)
  - Score consistency/variance
  - Whether weak areas were explored
  - Overall confidence in the assessment
"""

from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from core.llm_config import get_eval_llm

log = logging.getLogger("phase3.confidence")

CONFIDENCE_PROMPT = """You are a senior hiring manager reviewing an ongoing interview.

Decide whether the interview has gathered ENOUGH information to make a confident hiring decision, or if MORE questions are needed.

## Interview Context
- Position: {title} ({seniority})
- Required Skills: {required_skills}
- Total questions asked so far: {turn}
- Minimum questions before early stop: 3

## Skills Tested So Far
{skills_summary}

## Score History
{score_history}

## Coverage Dimensions
1. technical_depth — algorithms, tools, implementation details
2. business_understanding — how technical choices impact business metrics
3. communication — ability to explain concepts clearly
4. problem_solving — debugging approaches, handling ambiguity
5. real_world_decision_making — production trade-offs, monitoring, deployment

Current dimension coverage: {dimensions_covered}
Dimensions with ZERO coverage are gaps that should be addressed before stopping.

## Memory-Based Signals
{memory_signals}

## Decision Criteria (apply in order)

### MUST CONTINUE (override any stop decision):
- Unresolved contradictions exist → MUST probe them first
- Average specificity < 4.0 → answers too vague, need to push harder
- Fewer than 3 skills tested with real depth

### CAN STOP EARLY (even before max turns):
- Average specificity >= 7.0 AND all critical skills have depth signals
- Candidate has clearly demonstrated their level (consistent high or low scores)
- No unresolved contradictions AND strong depth signals across skills

### STANDARD CRITERIA:
- Have ALL critical required skills been tested?
- Are there skills with concerning scores (< 5/10) that need probing?
- Has the candidate shown a consistent pattern?
- Would additional questions significantly change the hiring decision?

Return ONLY valid JSON:
{{
    "decision": "continue" or "stop",
    "confidence": 0.0 to 1.0,
    "reason": "Brief explanation",
    "untested_critical_skills": ["skill1", "skill2"],
    "weak_areas_to_probe": ["area1"]
}}"""


def evaluate_confidence(state: dict) -> dict:
    """
    LLM-based decision on whether to continue or stop the interview.

    Returns:
        {
            "should_stop": bool,
            "confidence": float (0-1),
            "reason": str,
            "untested_skills": list[str],
            "weak_areas": list[str],
        }
    """
    turn = state.get("turn", 0)

    # Always ask at least 3 questions before considering stopping
    if turn < 3:
        return {
            "should_stop": False,
            "confidence": 0.0,
            "reason": "Minimum 3 questions not reached yet",
            "untested_skills": [],
            "weak_areas": [],
        }

    jd_profile = state.get("jd_profile")
    if not jd_profile:
        # Legacy mode — use simple turn-based logic
        max_turns = state.get("max_turns", 10)
        return {
            "should_stop": turn >= max_turns,
            "confidence": min(1.0, turn / max_turns),
            "reason": f"Turn {turn}/{max_turns}",
            "untested_skills": [],
            "weak_areas": [],
        }

    # Build context for LLM
    required_skills = jd_profile.get("required_skills", [])
    tested_skills = state.get("jd_skills_tested", [])
    pending_skills = state.get("pending_skills", [])
    transcript = state.get("transcript", [])

    # ── Build memory signals ──────────────────────────────────────────────
    memory_signals_parts = []
    contradiction_flags = state.get("contradiction_flags", [])
    specificity_scores = state.get("specificity_scores", [])

    unresolved_count = len(contradiction_flags)
    memory_signals_parts.append(f"- Unresolved contradictions: {unresolved_count}")

    if specificity_scores:
        avg_spec = sum(s.get("specificity", 5) for s in specificity_scores) / len(specificity_scores)
        avg_real = sum(s.get("realism", 5) for s in specificity_scores) / len(specificity_scores)
        memory_signals_parts.append(f"- Average specificity: {avg_spec:.1f}/10")
        memory_signals_parts.append(f"- Average realism: {avg_real:.1f}/10")
    else:
        memory_signals_parts.append("- No specificity data yet")

    memory_signals = "\n".join(memory_signals_parts)

    # Build skills summary
    skill_scores = state.get("skill_scores", {})
    skills_summary_parts = []
    for domain, skills in skill_scores.items():
        for skill, scores in skills.items():
            if scores:
                avg = sum(scores) / len(scores)
                skills_summary_parts.append(
                    f"  - {skill}: tested {len(scores)}x, avg score {avg:.1f}/10"
                )
            else:
                skills_summary_parts.append(f"  - {skill}: NOT TESTED YET")

    # Add tested skills not in skill_scores
    for skill in tested_skills:
        if not any(skill in line for line in skills_summary_parts):
            skills_summary_parts.append(f"  - {skill}: tested (from JD)")

    for skill in pending_skills:
        if not any(skill in line for line in skills_summary_parts):
            skills_summary_parts.append(f"  - {skill}: PENDING (not tested)")

    skills_summary = "\n".join(skills_summary_parts) if skills_summary_parts else "No skills tracked yet"

    # Build score history
    eval_entries = [
        e for e in transcript
        if e.get("role") == "evaluation" and e.get("score") is not None
    ]
    score_history = ", ".join(
        f"Q{i+1}: {e['score']}/10" for i, e in enumerate(eval_entries)
    ) if eval_entries else "No scores yet"

    try:
        from pydantic import BaseModel, Field
        from typing import Literal
        
        class ConfidenceEvaluation(BaseModel):
            decision: Literal["continue", "stop"]
            confidence: float = Field(ge=0.0, le=1.0)
            reason: str
            untested_critical_skills: list[str]
            weak_areas_to_probe: list[str]

        llm = get_eval_llm()
        structured_llm = llm.with_structured_output(ConfidenceEvaluation)
        
        prompt = CONFIDENCE_PROMPT.format(
            title=jd_profile.get("title", "Unknown"),
            seniority=jd_profile.get("seniority", "Mid"),
            required_skills=", ".join(required_skills),
            turn=turn,
            skills_summary=skills_summary,
            score_history=score_history,
            dimensions_covered=", ".join(
                f"{k}: {v}" for k, v in state.get("dimensions_covered", {}).items()
            ) or "none tracked yet",
            memory_signals=memory_signals,
        )

        evaluation = structured_llm.invoke([
            SystemMessage(content="You are a senior hiring manager making interview decisions."),
            HumanMessage(content=prompt),
        ])

        should_stop = evaluation.decision == "stop"
        confidence = float(evaluation.confidence)
        reason = evaluation.reason or ""
        untested = evaluation.untested_critical_skills or []
        weak = evaluation.weak_areas_to_probe or []

        log.info(
            "Confidence eval: decision=%s, confidence=%.2f, reason='%s'",
            "STOP" if should_stop else "CONTINUE", confidence, reason[:80],
        )

        return {
            "should_stop": should_stop,
            "confidence": confidence,
            "reason": reason,
            "untested_skills": untested,
            "weak_areas": weak,
        }

    except Exception as exc:
        log.warning("Confidence eval failed: %s — defaulting to continue", exc)
        # Fallback: stop if all skills tested
        if pending_skills:
            return {
                "should_stop": False,
                "confidence": 0.5,
                "reason": f"Fallback: {len(pending_skills)} skills remaining",
                "untested_skills": pending_skills,
                "weak_areas": [],
            }
        else:
            return {
                "should_stop": True,
                "confidence": 0.8,
                "reason": "Fallback: all skills tested",
                "untested_skills": [],
                "weak_areas": [],
            }

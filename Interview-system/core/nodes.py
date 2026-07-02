"""
Phase 2 — LangGraph Nodes
AI Voice Interview System

Four core nodes that plug into the LangGraph StateGraph:
  1. session_init   — load role, seed state
  2. question_gen   — pick the next best question
  3. answer_eval    — score the candidate's answer via LLM
  4. report_gen     — produce the final evaluation report
"""

from __future__ import annotations

import json
import logging
import uuid
from statistics import mean
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from core.interview_state import InterviewState
from core.llm_config import get_eval_llm, get_report_llm
from data_layer.phase1_data_layer import get_data_layer

log = logging.getLogger("phase2.nodes")

# Maximum consecutive follow-up probes on the same skill before forcing a pivot
MAX_CONSECUTIVE_PROBES = 3

# Minimum follow-ups per skill before allowing pivot (ensures we dig into answers)
MIN_PROBES_PER_SKILL = 1

# Coverage dimensions the interview should balance across
COVERAGE_DIMENSIONS = [
    "technical_depth",
    "business_understanding",
    "communication",
    "problem_solving",
    "real_world_decision_making",
]


# ─────────────────────────────────────────────────────────────────────────────
# 1. SESSION INIT
# ─────────────────────────────────────────────────────────────────────────────

def session_init(state: InterviewState) -> dict:
    """
    Boot the interview session.

    Two modes:
      A) JD/CV-driven (new): jd_profile + cv_profile provided
      B) Legacy: role_id provided → load from role_expectations

    Builds coverage map, sets phases, and queues pending skills.
    """
    dl = get_data_layer()
    jd_profile = state.get("jd_profile")
    cv_profile = state.get("cv_profile")
    role_id = state.get("role_id", "")

    # ── Mode A: JD/CV driven ──────────────────────────────────────────────
    if jd_profile:
        role_title = jd_profile.get("title", "Unknown Role")
        seniority = jd_profile.get("seniority", "Mid")
        required_skills = jd_profile.get("required_skills", [])
        nice_to_haves = jd_profile.get("nice_to_haves", [])
        domain = jd_profile.get("domain", "Technical")

        # Build phases from JD domain + standard phases
        phases = [domain]
        if domain != "Behavioral":
            phases.append("Behavioral")

        # Build coverage map from JD skills
        coverage_map: dict[str, dict[str, float]] = {}
        for skill in required_skills:
            coverage_map.setdefault(domain, {})[skill] = 0.0
        for skill in nice_to_haves:
            coverage_map.setdefault(domain, {})[skill] = 0.0

        # Queue all skills: required first, then nice-to-haves, then CV-specific
        pending_skills = list(required_skills)
        for s in nice_to_haves:
            if s not in pending_skills:
                pending_skills.append(s)

        # Add CV skills that overlap with JD for verification
        cv_skills_to_verify = []
        if cv_profile:
            cv_skills = cv_profile.get("skills", [])
            jd_skills_lower = {s.lower() for s in required_skills + nice_to_haves}
            for s in cv_skills:
                if s.lower() in jd_skills_lower and s not in pending_skills:
                    pending_skills.append(s)
                cv_skills_to_verify.append(s)

        # Determine max turns: ~1.5 questions per required skill, min 5, max 15
        max_turns = max(5, min(15, int(len(required_skills) * 1.5) + 2))
        pass_threshold_pct = 60.0
        depth_threshold = 5

        # Initialize skill_scores
        skill_scores: dict[str, dict[str, list[float]]] = {}
        for d, skills in coverage_map.items():
            skill_scores[d] = {sub: [] for sub in skills}

        session_id = state.get("session_id") or str(uuid.uuid4())

        log.info(
            "session_init (JD mode): title='%s' (%s), %d required skills, %d pending, max_turns=%d",
            role_title, seniority, len(required_skills), len(pending_skills), max_turns,
        )

        return {
            "session_id": session_id,
            "role_expectations": [],
            "role_title": role_title,
            "seniority": seniority,
            "phases": phases,
            "phase": phases[0],
            "turn": 0,
            "max_turns": max_turns,
            "pass_threshold_pct": pass_threshold_pct,
            "depth_threshold": depth_threshold,
            "skill_scores": skill_scores,
            "coverage_map": coverage_map,
            "chain_cursor": None,
            "current_question": None,
            "answered_ids": [],
            "transcript": [],
            "transcript_summary": "",
            "messages": [],
            "report": None,
            "jd_profile": jd_profile,
            "cv_profile": cv_profile,
            "jd_skills_tested": [],
            "cv_claims_verified": [],
            "pending_skills": pending_skills,
            "confidence_score": 0.0,
            "soft_skills_scores": [],
            "consecutive_probes_on_skill": {},
            "dimensions_covered": {d: 0 for d in COVERAGE_DIMENSIONS},
            "interview_memory": {},
            "contradiction_flags": [],
            "specificity_scores": [],
            "recent_openers": [],
            "skills_tested_order": [],
        }

    # ── Mode B: Legacy role_id ────────────────────────────────────────────
    log.info("session_init: loading role '%s'", role_id)

    expectations = dl.get_role_expectations(role_id)
    if not expectations:
        raise ValueError(f"No role expectations found for role_id='{role_id}'")

    first = expectations[0]
    role_title = first.get("title", "Unknown Role")
    seniority = first.get("seniority", "Unknown")

    raw_domains = first.get("domain", "Technical")
    if isinstance(raw_domains, str) and "|" in raw_domains:
        phases = [p.strip() for p in raw_domains.split("|")]
    elif isinstance(raw_domains, str):
        phases = [raw_domains.strip()]
    else:
        phases = ["Technical"]

    try:
        max_turns = int(first.get("num_questions_recommended", 10))
    except (ValueError, TypeError):
        max_turns = 10

    try:
        pass_threshold_pct = float(first.get("pass_threshold_pct", 60))
    except (ValueError, TypeError):
        pass_threshold_pct = 60.0

    try:
        depth_threshold = int(first.get("depth_decision_threshold", 5))
    except (ValueError, TypeError):
        depth_threshold = 5

    coverage_map = dl.get_coverage_map(role_id)

    skill_scores: dict[str, dict[str, list[float]]] = {}
    for d, skills in coverage_map.items():
        skill_scores[d] = {sub: [] for sub in skills}

    session_id = state.get("session_id") or str(uuid.uuid4())

    log.info(
        "session_init complete: role='%s' (%s), phases=%s, max_turns=%d",
        role_title, seniority, phases, max_turns,
    )

    return {
        "session_id": session_id,
        "role_expectations": expectations,
        "role_title": role_title,
        "seniority": seniority,
        "phases": phases,
        "phase": phases[0],
        "turn": 0,
        "max_turns": max_turns,
        "pass_threshold_pct": pass_threshold_pct,
        "depth_threshold": depth_threshold,
        "skill_scores": skill_scores,
        "coverage_map": coverage_map,
        "chain_cursor": None,
        "current_question": None,
        "answered_ids": [],
        "transcript": [],
        "transcript_summary": "",
        "messages": [],
        "report": None,
        "jd_profile": jd_profile,
        "cv_profile": cv_profile,
        "jd_skills_tested": [],
        "cv_claims_verified": [],
        "pending_skills": [],
        "confidence_score": 0.0,
        "soft_skills_scores": [],
        "consecutive_probes_on_skill": {},
        "dimensions_covered": {d: 0 for d in COVERAGE_DIMENSIONS},
        "interview_memory": {},
        "contradiction_flags": [],
        "specificity_scores": [],
        "recent_openers": [],
        "skills_tested_order": [],
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. QUESTION GENERATION — Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_opener(question_text: str) -> str:
    """
    Extract and normalize the first ~15 words of a question.
    Normalization: strip().lower() to prevent bypass via capitalization or whitespace.
    """
    words = question_text.strip().split()[:15]
    return " ".join(words).strip().lower()


def _select_next_skill(
    pending_skills: list[str],
    skills_tested_order: list[str],
    depth_signals: dict,
    memory_claims: list,
) -> dict:
    """
    Smart skill selection with non-linear topic flow.

    Instead of always picking pending_skills[0], this function introduces
    human-like non-linearity:
      - ~25% chance of CALLBACK: revisit a previously-tested skill that
        needs more depth, always referencing a specific earlier claim
      - ~15% chance of SKIP-AHEAD: pick a later skill from the queue
      - ~60% default: next skill in order

    Returns:
        {
            "skill": str,
            "is_callback": bool,
            "revisit_context": str,  # specific claim/detail for contextual callback
            "remaining": list[str],  # updated pending list
        }
    """
    import random

    # Build list of skills eligible for callback (tested but not deep enough)
    callback_candidates = []
    for skill in skills_tested_order:
        ds = depth_signals.get(skill)
        if ds and not ds.get("is_sufficiently_deep", True):
            # Find specific claims about this skill for contextual reference
            skill_claims = [
                c for c in memory_claims
                if c.get("skill", "").lower() == skill.lower()
            ]
            if skill_claims:
                callback_candidates.append((skill, skill_claims))

    roll = random.random()

    # ── ~25% CALLBACK to a previous skill ────────────────────────────────
    if roll < 0.25 and callback_candidates and pending_skills:
        chosen_skill, claims = random.choice(callback_candidates)
        # Pick the most recent claim for the revisit reference
        latest_claim = claims[-1]
        revisit_context = (
            f"REVISITING '{chosen_skill}' — the candidate previously claimed: "
            f"\"{latest_claim.get('claim', '')}\""
            f" (Turn {latest_claim.get('turn', '?')}, category: {latest_claim.get('category', 'general')}). "
            f"Dig deeper into THIS specific claim."
        )
        log.info(
            "question_gen: CALLBACK to '%s' — revisiting claim: '%s'",
            chosen_skill, latest_claim.get("claim", "")[:60],
        )
        return {
            "skill": chosen_skill,
            "is_callback": True,
            "revisit_context": revisit_context,
            "remaining": pending_skills,  # don't consume from queue
        }

    # ── ~15% SKIP-AHEAD (pick index 1-3 instead of 0) ───────────────────
    if roll < 0.40 and len(pending_skills) > 2:
        skip_idx = random.randint(1, min(3, len(pending_skills) - 1))
        chosen = pending_skills[skip_idx]
        remaining = [s for i, s in enumerate(pending_skills) if i != skip_idx]
        log.info(
            "question_gen: SKIP-AHEAD to '%s' (index %d instead of 0)",
            chosen, skip_idx,
        )
        return {
            "skill": chosen,
            "is_callback": False,
            "revisit_context": "",
            "remaining": remaining,
        }

    # ── ~60% DEFAULT — next in order ─────────────────────────────────────
    if pending_skills:
        return {
            "skill": pending_skills[0],
            "is_callback": False,
            "revisit_context": "",
            "remaining": pending_skills[1:],
        }

    return {
        "skill": "",
        "is_callback": False,
        "revisit_context": "",
        "remaining": [],
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. QUESTION GENERATION
# ─────────────────────────────────────────────────────────────────────────────


def question_gen(state: InterviewState) -> dict:
    """
    Generate the next question — memory-aware, adaptive, and conversational.

    Strategy:
      1. Load interview memory for cross-turn context
      2. If follow-up → adaptive based on candidate's answer + full memory
      3. If pivot → connect to something already discussed
      4. Memory-based depth signals replace crude probe counters
    """
    from intelligence.adaptive_followup import (
        analyze_answer_and_generate_followup,
        generate_opening_question,
    )
    from intelligence.interview_memory import InterviewMemory

    phase = state.get("phase", "Technical")
    role_id = state.get("role_id", "")
    role_title = state.get("role_title", role_id)
    turn = state.get("turn", 0)
    jd_profile = state.get("jd_profile")
    cv_profile = state.get("cv_profile")
    pending_skills = list(state.get("pending_skills", []))
    transcript = state.get("transcript", [])

    # ── Load interview memory ─────────────────────────────────────────────
    memory = InterviewMemory.from_dict(state.get("interview_memory", {}))
    memory_context = memory.get_memory_context()

    # ── Load recent openers for anti-repetition ─────────────────────────
    recent_openers = list(state.get("recent_openers", []))
    skills_tested_order = list(state.get("skills_tested_order", []))

    log.info("question_gen: phase='%s', turn=%d, pending=%d, openers_tracked=%d",
             phase, turn, len(pending_skills), len(recent_openers))

    question_text = ""
    target_skill = None
    updated_pending = pending_skills
    source = "adaptive"
    revisit_context = ""

    # ── Get the last answer + score from transcript ───────────────────────
    last_answer = ""
    last_score = 0
    last_question_text = ""
    last_skill = ""

    for entry in reversed(transcript):
        if entry.get("role") == "candidate" and not last_answer:
            last_answer = entry.get("content", "")
        if entry.get("role") == "evaluation" and last_score == 0:
            last_score = entry.get("score", 5)
        if entry.get("role") == "interviewer" and not last_question_text:
            last_question_text = entry.get("content", "")

    current_q = state.get("current_question") or {}
    last_skill = current_q.get("sub_skill", "")

    # ── Topic rotation state ────────────────────────────────────────────────
    probes_map = dict(state.get("consecutive_probes_on_skill", {}))
    dimensions_covered = dict(state.get("dimensions_covered", {d: 0 for d in COVERAGE_DIMENSIONS}))
    current_probes = probes_map.get(last_skill, 0) if last_skill else 0

    # ── Use memory-based depth signal for smarter pivoting ────────────────
    depth_signal = memory.get_depth_signal(last_skill) if last_skill else None
    skill_is_deep_enough = depth_signal.is_sufficiently_deep if depth_signal else False

    # ── Decide: adapt from previous answer or start fresh ─────────────────
    if last_answer and jd_profile and turn > 0:
        # Adaptive follow-up based on what the candidate said + full memory
        adaptive = analyze_answer_and_generate_followup(
            question_text=last_question_text,
            answer_text=last_answer,
            score=last_score,
            current_skill=last_skill,
            remaining_skills=pending_skills,
            jd_profile=jd_profile,
            consecutive_probes=current_probes,
            dimensions_covered=dimensions_covered,
            memory_context=memory_context,
            recent_openers=recent_openers,
        )

        action = adaptive["action"]
        confidence = adaptive.get("confidence_in_skill", 0.5)

        # ── Guard 1: Force FOLLOW-UP if we haven't probed this skill enough ──
        #    A real interviewer NEVER asks one question and moves on.
        #    Minimum 1 follow-up per skill (unless candidate scored ≤ 2 = zero knowledge)
        if action == "pivot" and current_probes < MIN_PROBES_PER_SKILL and last_score > 2:
            log.info(
                "question_gen: BLOCKED PIVOT — only %d probes on '%s' (min=%d), "
                "forcing follow-up from adaptive engine",
                current_probes, last_skill, MIN_PROBES_PER_SKILL,
            )
            # Use the adaptive question if available (LLM generated one even though it said pivot)
            if adaptive.get("question"):
                action = "probe_deeper"
            else:
                # Generate a follow-up ourselves based on the answer
                action = "probe_deeper"
                adaptive["question"] = (
                    f"Got it, interesting points. I'd love to hear about a specific project or situation "
                    f"where you actually applied {last_skill} in practice — what did that look like?"
                )

        # ── Guard 2: Force PIVOT after MAX_CONSECUTIVE_PROBES ──
        #    But allow contradiction_probe to override (must resolve contradictions)
        non_contradiction_actions = ("probe_deeper", "clarify", "challenge", "scenario", "depth_check")
        if action in non_contradiction_actions and current_probes >= MAX_CONSECUTIVE_PROBES:
            # Memory-based override: if depth signal says not enough, allow one more
            if not skill_is_deep_enough and current_probes == MAX_CONSECUTIVE_PROBES:
                log.info(
                    "question_gen: EXTENDED PROBING — depth signal says '%s' needs more, allowing one extra probe",
                    last_skill,
                )
            else:
                log.info(
                    "question_gen: FORCED PIVOT — %d probes on '%s' (max=%d), overriding action='%s'",
                    current_probes, last_skill, MAX_CONSECUTIVE_PROBES, action,
                )
                action = "pivot"

        stay_actions = ("probe_deeper", "clarify", "challenge", "scenario", "contradiction_probe", "depth_check")

        # ── Guard 3: Post-LLM skill-mention detection ─────────────────────────
        # The LLM sometimes returns probe_deeper even when the candidate explicitly
        # mentioned a pending skill. Scan the answer and override to new_topic.
        if action in stay_actions and pending_skills:
            answer_lower = last_answer.lower()
            for _candidate_skill in pending_skills:
                if _candidate_skill.lower() in answer_lower:
                    log.info(
                        "question_gen: SKILL MENTION DETECTED — candidate mentioned '%s' "
                        "in answer but LLM chose '%s'; overriding to new_topic",
                        _candidate_skill, action,
                    )
                    action = "new_topic"
                    adaptive["detected_topic"] = _candidate_skill
                    break

        if action in stay_actions and adaptive["question"]:
            # Stay on current skill — use adaptive follow-up question
            question_text = adaptive["question"]
            target_skill = last_skill
            source = f"adaptive_{action}"
            # Increment probe counter for this skill
            probes_map[last_skill] = current_probes + 1
            log.info(
                "question_gen: %s on '%s' (confidence=%.2f, probes=%d/%d, deep_enough=%s)",
                action, last_skill, confidence, probes_map[last_skill],
                MAX_CONSECUTIVE_PROBES, skill_is_deep_enough,
            )

        elif action == "new_topic" and adaptive.get("detected_topic"):
            # Candidate mentioned a new skill — ask a proper OPENING question for it,
            # NOT the probe-style follow-up generated by the analysis prompt.
            new_topic = adaptive["detected_topic"]
            cv_context = ""
            if cv_profile:
                for exp in cv_profile.get("experience", []):
                    if new_topic.lower() in str(exp).lower():
                        cv_context = str(exp.get("description", ""))
                        break
            question_text = generate_opening_question(
                skill=new_topic,
                jd_profile=jd_profile,
                cv_context=cv_context,
                memory_context=memory_context,
                recent_openers=recent_openers,
            )
            target_skill = new_topic
            source = "adaptive_new_topic"
            # Reset probe counter for old skill; start fresh for the new topic
            if last_skill:
                probes_map[last_skill] = 0
            probes_map[new_topic] = 0
            # Remove new_topic from pending so it isn't asked again later
            updated_pending = [s for s in updated_pending if s.lower() != new_topic.lower()]
            log.info("question_gen: exploring new topic '%s' mentioned by candidate (opening question)", new_topic)

        else:
            # Pivot — move to next skill
            action = "pivot"

        if action == "pivot":
            # Reset probe counter for old skill, start fresh for new one
            if last_skill:
                probes_map[last_skill] = 0
            if pending_skills:
                # ── Smart skill selection with non-linear flow ─────────
                memory_data = state.get("interview_memory", {})
                ds_raw = memory_data.get("depth_signals", {})
                claims_raw = memory_data.get("factual_claims", [])
                selection = _select_next_skill(
                    pending_skills=pending_skills,
                    skills_tested_order=skills_tested_order,
                    depth_signals=ds_raw,
                    memory_claims=claims_raw,
                )
                target_skill = selection["skill"]
                updated_pending = selection["remaining"]
                revisit_context = selection["revisit_context"]

                cv_context = ""
                if cv_profile:
                    for exp in cv_profile.get("experience", []):
                        if target_skill.lower() in str(exp).lower():
                            cv_context = str(exp.get("description", ""))
                            break

                question_text = generate_opening_question(
                    skill=target_skill,
                    jd_profile=jd_profile,
                    cv_context=cv_context,
                    memory_context=memory_context,
                    recent_openers=recent_openers,
                    revisit_context=revisit_context,
                )
                source = "adaptive_callback" if selection["is_callback"] else "adaptive_pivot"
                log.info("question_gen: %s to skill '%s'", source, target_skill)
            else:
                # No more skills — generate a wrap-up question
                question_text = generate_opening_question(
                    skill=last_skill,
                    jd_profile=jd_profile,
                    memory_context=memory_context,
                    recent_openers=recent_openers,
                )
                target_skill = last_skill
                source = "adaptive_wrapup"

    # ── First question (turn 0) or no JD — opening question ───────────────
    elif jd_profile and pending_skills:
        target_skill = pending_skills[0]
        updated_pending = pending_skills[1:]

        cv_context = ""
        if cv_profile:
            for exp in cv_profile.get("experience", []):
                if target_skill.lower() in str(exp).lower():
                    cv_context = str(exp.get("description", ""))
                    break

        question_text = generate_opening_question(
            skill=target_skill,
            jd_profile=jd_profile,
            cv_context=cv_context,
            memory_context=memory_context,
            recent_openers=recent_openers,
        )
        source = "opening"
        log.info("question_gen: opening question for skill '%s'", target_skill)

    # ── Fallback ──────────────────────────────────────────────────────────
    if not question_text:
        target_skill = target_skill or "general experience"
        role_context = f" in your {role_title} work" if role_title and role_title != role_id else ""
        question_text = (
            f"Tell me about your experience with {target_skill}{role_context}. "
            f"What's a challenging project where you applied it?"
        )
        source = "fallback"
        log.warning("question_gen: using fallback question")

    # Build question dict
    import uuid
    question = {
        "question_id": f"ADQ-{uuid.uuid4().hex[:8]}",
        "question_text": question_text,
        "domain": phase,
        "sub_skill": target_skill or phase,
        "difficulty": "Medium",
        "phase": phase,
        "source": source,
    }

    log.info(
        "question_gen: Q='%s' skill='%s' source='%s'",
        question["question_id"], question["sub_skill"], source,
    )

    interviewer_msg = question["question_text"]

    # ── Update opener history (normalized) ─────────────────────────────────
    new_opener = _extract_opener(interviewer_msg)
    updated_openers = list(recent_openers)
    updated_openers.append(new_opener)
    # Keep only the last 3 openers
    if len(updated_openers) > 3:
        updated_openers = updated_openers[-3:]

    # ── Update skills tested order ────────────────────────────────────────
    updated_skills_order = list(skills_tested_order)
    effective_skill = target_skill or phase
    if effective_skill not in updated_skills_order:
        updated_skills_order.append(effective_skill)

    result = {
        "current_question": question,
        "turn": turn + 1,
        "pending_skills": updated_pending,
        "consecutive_probes_on_skill": probes_map,
        "recent_openers": updated_openers,
        "skills_tested_order": updated_skills_order,
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

    # Track that this JD skill was tested (both opening and follow-up probes)
    if target_skill and jd_profile:
        already_tested = state.get("jd_skills_tested", [])
        if target_skill not in already_tested:
            result["jd_skills_tested"] = [target_skill]

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 3. ANSWER EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

EVAL_SYSTEM_PROMPT = """You are a SENIOR technical interviewer at a top tech company evaluating a candidate's answer.
You have MEMORY of all previous answers in this interview.

You will be given:
1. The interview question
2. The candidate's answer
3. Interview memory context (what the candidate said in previous turns)
4. Scoring rubric criteria
5. Calibration examples (gold-standard scored answers)

## SCORING DIMENSIONS (score each 1-10):

### 1. technical_score (1-10): Technical accuracy and correctness
### 2. depth_score (1-10): Depth of understanding
- 1-3: Surface-level, textbook definition only
- 4-6: Moderate understanding, some details
- 7-10: Deep understanding with nuance, edge cases, tradeoffs

### 3. specificity_score (1-10): How specific and concrete is the answer?
VAGUENESS PENALTIES (apply strictly):
- "I would use best practices" → 2
- "We optimized the model" without HOW → 3
- "I have experience with X" without ANY details → 2
- Textbook definition without personal experience → 3
- Generic answer that could apply to any tool/framework → 3
SPECIFICITY REWARDS:
- Mentioning specific metrics: "reduced latency from 200ms to 50ms" → 9
- Naming tools with config: "XGBoost with learning_rate=0.01, max_depth=6" → 9
- Describing real decisions: "We chose PostgreSQL over MongoDB because..." → 8

### 4. realism_score (1-10): Does this sound like REAL experience?
- 1-3: Clearly rehearsed, AI-generated, or textbook knowledge
- 4-6: Plausible but lacking personal details, no failures/surprises
- 7-10: Real experience (mentions failures, surprises, specific decisions, lessons learned)

### 5. consistency_score (1-10): Consistency with previous answers
- Check against the memory context for contradictions
- 10 if no contradictions, reduce for each inconsistency found

## OVERALL SCORE FORMULA
overall = (technical * 0.25) + (depth * 0.25) + (specificity * 0.25) + (realism * 0.15) + (consistency * 0.10)

## STRENGTH/GAP RULES
For each gap, name the SPECIFIC concepts, techniques, or tools the candidate missed.
BAD:  "Needs improvement in advanced imputation"
GOOD: "Did not mention KNN Imputation, MICE (IterativeImputer), or handling MNAR data patterns"

For each strength, be equally specific.
BAD:  "Good understanding of ML"
GOOD: "Correctly explained gradient boosting ensemble mechanics and referenced XGBoost regularisation parameters"

## RESPONSE FORMAT (JSON only):
{{
    "score": <integer 1-10, computed from formula above>,
    "technical_score": <integer 1-10>,
    "depth_score": <integer 1-10>,
    "specificity_score": <integer 1-10>,
    "realism_score": <integer 1-10>,
    "consistency_score": <integer 1-10>,
    "rationale": "<2-3 sentences explaining the score>",
    "strengths": ["<specific string>", ...],
    "gaps": ["<specific string>", ...],
    "missing_concepts": ["<string>", ...],
    "dimension": "technical_depth" | "business_understanding" | "communication" | "problem_solving" | "real_world_decision_making"
}}

Respond ONLY with the JSON object, no other text."""

class AnswerEvaluation(BaseModel):
    score: int = Field(description="Overall score from 1 to 10")
    technical_score: int = Field(default=5, description="Technical accuracy 1-10")
    depth_score: int = Field(default=5, description="Depth of understanding 1-10")
    specificity_score: int = Field(default=5, description="Specificity and concreteness 1-10")
    realism_score: int = Field(default=5, description="Real experience vs textbook 1-10")
    consistency_score: int = Field(default=10, description="Consistency with previous answers 1-10")
    rationale: str = Field(description="Brief explanation of the score")
    strengths: list[str] = Field(description="Specific concepts/techniques demonstrated well")
    gaps: list[str] = Field(description="Specific concepts/techniques missing from the answer")
    missing_concepts: list[str] = Field(
        default_factory=list,
        description="Named techniques/tools the candidate should have mentioned"
    )
    dimension: str = Field(
        default="technical_depth",
        description="Which coverage dimension this Q&A primarily tests"
    )

def answer_eval(state: InterviewState) -> dict:
    """
    Evaluate the candidate's answer using LLM + memory + rubric.

    Enhanced with:
      - Multi-dimensional scoring (technical, depth, specificity, realism, consistency)
      - Interview memory integration (records claims, checks contradictions)
      - Vagueness penalties and specificity rewards
    """
    from intelligence.interview_memory import InterviewMemory

    dl = get_data_layer()
    question = state.get("current_question")
    transcript = state.get("transcript", [])

    if not question:
        log.error("answer_eval: no current_question in state")
        return {}

    # Find the candidate's answer (last candidate entry in transcript)
    candidate_answer = ""
    for entry in reversed(transcript):
        if entry.get("role") == "candidate":
            candidate_answer = entry["content"]
            break

    if not candidate_answer:
        log.error("answer_eval: no candidate answer found in transcript")
        return {}

    question_id = question["question_id"]
    domain = question["domain"]
    sub_skill = question["sub_skill"]

    log.info("answer_eval: scoring Q='%s' (domain=%s, sub_skill=%s)", question_id, domain, sub_skill)

    # ── Load interview memory for context ─────────────────────────────────
    memory = InterviewMemory.from_dict(state.get("interview_memory", {}))
    memory_context = memory.get_memory_context(sub_skill)

    # ── Load rubric ───────────────────────────────────────────────────────
    try:
        rubric = dl.find_rubric_criteria(candidate_answer, domain, sub_skill)
    except Exception:
        try:
            rubric = dl.get_rubric(domain, sub_skill)
        except Exception:
            rubric = []

    rubric_text = ""
    if rubric:
        for r in rubric[:3]:
            if isinstance(r, dict):
                rubric_text += f"- {r.get('criteria', r.get('document', ''))}\n"

    # ── Load calibration examples ─────────────────────────────────────────
    try:
        calibration = dl.get_calibration_examples(domain, sub_skill, n=3)
    except Exception:
        calibration = []

    calibration_text = ""
    for cal in calibration:
        cal_score = cal.get("score", "?")
        cal_answer = cal.get("transcript_excerpt", "")[:200]
        cal_rationale = cal.get("rationale", "")[:150]
        calibration_text += (
            f"  Score {cal_score}: \"{cal_answer}...\"\n"
            f"  Rationale: {cal_rationale}\n\n"
        )

    # ── Build evaluation prompt with memory context ───────────────────────
    eval_prompt = f"""**Question:** {question['question_text']}

**Candidate's Answer:** {candidate_answer}

**Interview Memory (previous answers and claims):**
{memory_context[:2000]}

**Rubric Criteria:**
{rubric_text or 'Use general interview evaluation standards.'}

**Calibration Examples (gold-standard):**
{calibration_text or 'No calibration examples available.'}

Score the candidate's answer across all dimensions and provide your evaluation."""

    # ── Call LLM ──────────────────────────────────────────────────────────
    llm = get_eval_llm()
    structured_llm = llm.with_structured_output(AnswerEvaluation)
    specificity_score_val = 5
    realism_score_val = 5
    try:
        evaluation = structured_llm.invoke([
            SystemMessage(content=EVAL_SYSTEM_PROMPT),
            HumanMessage(content=eval_prompt),
        ])

        score = max(1, min(10, evaluation.score))
        rationale = evaluation.rationale or "No rationale provided."
        strengths = evaluation.strengths or []
        gaps = evaluation.gaps or []
        missing_concepts = evaluation.missing_concepts or []
        dimension = evaluation.dimension or "technical_depth"
        specificity_score_val = max(1, min(10, evaluation.specificity_score))
        realism_score_val = max(1, min(10, evaluation.realism_score))

    except Exception as exc:
        log.error("LLM evaluation failed: %s — using default score", exc)
        score = 5
        rationale = f"LLM evaluation failed: {exc}"
        strengths = []
        gaps = []
        missing_concepts = []
        dimension = "technical_depth"

    log.info(
        "answer_eval: score=%d specificity=%d realism=%d for Q='%s' (dimension=%s)",
        score, specificity_score_val, realism_score_val, question_id, dimension,
    )

    # ── Record turn in interview memory ───────────────────────────────────
    turn = state.get("turn", 0)
    memory_result = memory.record_turn(
        question=question["question_text"],
        answer=candidate_answer,
        skill=sub_skill,
        turn=turn,
        score=score,
        strengths=strengths,
        gaps=gaps,
    )

    # Build contradiction flags from memory
    new_contradictions = []
    for c in memory.get_unresolved_contradictions():
        if c.turn_b == turn:  # Only flag new ones from this turn
            new_contradictions.append({
                "claim_a": c.claim_a,
                "claim_b": c.claim_b,
                "turn_a": c.turn_a,
                "turn_b": c.turn_b,
                "skill": c.skill,
                "severity": c.severity,
            })

    # ── Update skill scores (deep-copy to avoid mutating state in-place) ──
    import copy
    skill_scores = copy.deepcopy(state.get("skill_scores", {}))
    if domain not in skill_scores:
        skill_scores[domain] = {}
    if sub_skill not in skill_scores[domain]:
        skill_scores[domain][sub_skill] = []
    skill_scores[domain][sub_skill].append(float(score))

    # ── Recompute coverage map (deep-copy) ────────────────────────────────
    coverage_map = copy.deepcopy(state.get("coverage_map", {}))
    if domain not in coverage_map:
        coverage_map[domain] = {}
    coverage_map[domain][sub_skill] = mean(skill_scores[domain][sub_skill])

    # ── Mark question answered (only for dataset questions, not adaptive) ─
    if not question_id.startswith("ADQ-"):
        try:
            dl.mark_question_answered(question_id)
        except Exception as exc:
            log.warning("Could not mark question answered: %s", exc)

    # ── Determine if chain follow-up is available ─────────────────────────
    chain_cursor = None
    if not question_id.startswith(("GEN-", "ADQ-", "CHAIN-")):
        try:
            chains = dl.get_chain_questions(question_id)
            if chains:
                first_chain = chains[0]
                threshold = float(first_chain.get("scoring_threshold", 6))
                if score >= threshold:
                    chain_cursor = {
                        "base_question_id": question_id,
                        "order": int(first_chain.get("order_in_chain", 2)),
                        "max_depth": int(first_chain.get("max_depth", 3)),
                        "scoring_threshold": threshold,
                    }
                    log.info(
                        "answer_eval: chain triggered for Q='%s' (score=%d >= threshold=%d)",
                        question_id, score, threshold,
                    )
        except Exception as exc:
            log.debug("No chains for Q='%s': %s", question_id, exc)

    # ── Soft Skills Evaluation ────────────────────────────────────────────
    soft_result = {}
    try:
        from evaluation.soft_skills_evaluator import SoftSkillsEvaluator
        sse = SoftSkillsEvaluator()
        soft = sse.evaluate(question["question_text"], candidate_answer)
        soft_result = soft.to_dict()
    except Exception as exc:
        log.warning("Soft skills eval failed: %s", exc)

    # ── Update dimensions_covered counter ───────────────────────────────────
    dimensions_covered = dict(state.get("dimensions_covered", {d: 0 for d in COVERAGE_DIMENSIONS}))
    if dimension in dimensions_covered:
        dimensions_covered[dimension] = dimensions_covered.get(dimension, 0) + 1
    else:
        dimensions_covered[dimension] = 1

    return {
        "skill_scores": skill_scores,
        "coverage_map": coverage_map,
        "chain_cursor": chain_cursor,
        "answered_ids": [question_id],
        "soft_skills_scores": [soft_result] if soft_result else [],
        "dimensions_covered": dimensions_covered,
        "interview_memory": memory.to_dict(),
        "contradiction_flags": new_contradictions,
        "specificity_scores": [{
            "turn": turn,
            "skill": sub_skill,
            "specificity": specificity_score_val,
            "realism": realism_score_val,
        }],
        "transcript": [
            {
                "role": "evaluation",
                "content": rationale,
                "turn": turn,
                "question_id": question_id,
                "score": score,
                "strengths": strengths,
                "gaps": gaps,
                "missing_concepts": missing_concepts,
                "dimension": dimension,
                "specificity_score": specificity_score_val,
                "realism_score": realism_score_val,
                "soft_skills": soft_result,
            }
        ],
        "messages": [],
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3.5 CONTEXT SUMMARIZATION
# ─────────────────────────────────────────────────────────────────────────────

def summarize_context(state: InterviewState) -> dict:
    """Summarize older transcript entries to prevent context window bloat.
    
    Now memory-aware — passes memory signals to the context manager
    so factual claims, contradictions, and depth signals survive summarization.
    """
    from intelligence.context_manager import summarize_transcript
    from intelligence.interview_memory import InterviewMemory
    
    turn = state.get("turn", 0)
    # Only summarize every 3 turns
    if turn % 3 != 0 or turn == 0:
        return {}
        
    transcript = state.get("transcript", [])
    current_summary = state.get("transcript_summary", "")
    
    # 3 turns = ~9 entries (interviewer, candidate, evaluation)
    latest_turns = transcript[-9:]
    
    # Load memory signals for preservation during summarization
    memory = InterviewMemory.from_dict(state.get("interview_memory", {}))
    memory_signals = memory.get_memory_context()
    
    new_summary = summarize_transcript(
        current_summary, 
        latest_turns,
        memory_signals=memory_signals,
    )
    log.info("summarize_context: generated new summary (len=%d) at turn %d", len(new_summary), turn)
    
    return {"transcript_summary": new_summary}


# ─────────────────────────────────────────────────────────────────────────────
# 4. REPORT GENERATION
# ─────────────────────────────────────────────────────────────────────────────

REPORT_SYSTEM_PROMPT = """You are generating a final interview evaluation report as a senior hiring manager.

You will receive:
1. Role expectations and interview metadata
2. Full interview transcript with per-answer scores
3. Coverage map (average scores per domain/skill)
4. Memory analysis: contradictions, specificity scores, depth signals, strongest/weakest moments

Generate a comprehensive evaluation report in this JSON format:
{{
    "overall_score": <float>,
    "recommendation": "STRONG_HIRE" | "HIRE" | "LEAN_HIRE" | "LEAN_NO_HIRE" | "NO_HIRE",
    "scoring_breakdown": {{
        "technical_accuracy": <float 1-10>,
        "depth_of_understanding": <float 1-10>,
        "specificity": <float 1-10>,
        "realism": <float 1-10>,
        "consistency": <float 1-10>
    }},
    "domain_scores": {{"<domain>": {{"score": <float>, "summary": "<string>"}}}},
    "strengths": ["<string>", ...],
    "areas_for_improvement": ["<string>", ...],
    "red_flags": ["<string describing any red flags>", ...],
    "hiring_confidence": <float 0-1>,
    "executive_summary": "<string 2-3 paragraphs — professional and specific>"
}}

IMPORTANT:
- If contradictions were detected, mention them specifically in the executive summary
- If specificity is low, note that the candidate gave vague answers
- Reference SPECIFIC moments from the interview (strongest and weakest)
- The executive summary should read like a senior interviewer's debrief, not generic text

Respond ONLY with the JSON object, no other text."""

class DomainScore(BaseModel):
    score: float
    summary: str

class ScoringBreakdown(BaseModel):
    technical_accuracy: float = 5.0
    depth_of_understanding: float = 5.0
    specificity: float = 5.0
    realism: float = 5.0
    consistency: float = 10.0

class ReportEvaluation(BaseModel):
    overall_score: float
    recommendation: Literal["STRONG_HIRE", "HIRE", "LEAN_HIRE", "LEAN_NO_HIRE", "NO_HIRE"]
    scoring_breakdown: ScoringBreakdown = Field(default_factory=ScoringBreakdown)
    domain_scores: dict[str, DomainScore]
    strengths: list[str]
    areas_for_improvement: list[str]
    red_flags: list[str] = Field(default_factory=list)
    hiring_confidence: float = Field(ge=0.0, le=1.0)
    executive_summary: str

def report_gen(state: InterviewState) -> dict:
    """
    Generate the final interview evaluation report.

    Computes weighted scores, checks pass thresholds, and uses LLM
    to generate qualitative summaries.
    """
    coverage_map = state.get("coverage_map", {})
    skill_scores = state.get("skill_scores", {})
    role_expectations = state.get("role_expectations", [])
    transcript = state.get("transcript", [])
    role_title = state.get("role_title", "Unknown")
    seniority = state.get("seniority", "Unknown")
    pass_threshold_pct = state.get("pass_threshold_pct", 60.0)

    log.info("report_gen: generating final report")

    # ── Compute weighted overall score ────────────────────────────────────
    total_weight = 0.0
    weighted_sum = 0.0
    domain_scores = {}

    # First: build domain scores directly from coverage_map (which has actual scored data)
    for domain, skills in coverage_map.items():
        scored_skills = {sk: sc for sk, sc in skills.items() if sc > 0}
        if scored_skills:
            domain_avg = mean(scored_skills.values())
            domain_scores[domain] = {
                "score": round(domain_avg, 1),
                "skills": {sk: round(sc, 1) for sk, sc in scored_skills.items()},
            }

    # Second: compute weighted overall using role_expectations weights
    if role_expectations:
        # Legacy mode — role_expectations may have pipe-separated domains
        for exp in role_expectations:
            raw_domain = str(exp.get("domain", ""))
            weight = float(exp.get("weight", 1.0))
            domains = [d.strip() for d in raw_domain.split("|")] if "|" in raw_domain else [raw_domain.strip()]

            for domain in domains:
                if domain in coverage_map:
                    scored = [s for s in coverage_map[domain].values() if s > 0]
                    if scored:
                        domain_avg = mean(scored)
                        weighted_sum += domain_avg * weight
                        total_weight += weight

        overall_score = round(weighted_sum / total_weight, 1) if total_weight > 0 else 0.0
    else:
        # JD mode — compute overall directly from all scored skills in coverage_map
        all_scored = []
        for domain, skills in coverage_map.items():
            for sk_score in skills.values():
                if sk_score > 0:
                    all_scored.append(sk_score)
        overall_score = round(mean(all_scored), 1) if all_scored else 0.0
        log.info("report_gen (JD mode): computed overall=%.1f from %d scored skills",
                 overall_score, len(all_scored))

    # ── Determine pass/fail ───────────────────────────────────────────────
    max_possible = 10.0
    score_pct = (overall_score / max_possible) * 100
    passed = score_pct >= pass_threshold_pct

    # ── Build transcript summary for LLM ──────────────────────────────────
    transcript_summary = ""
    for entry in transcript:
        role = entry.get("role", "unknown")
        content = entry.get("content", "")[:300]
        score_val = entry.get("score")
        score_str = f" [Score: {score_val}]" if score_val is not None else ""
        transcript_summary += f"{role.upper()}: {content}{score_str}\n"

    # ── Build memory analysis for report prompt ──────────────────────────
    from intelligence.interview_memory import InterviewMemory
    memory = InterviewMemory.from_dict(state.get("interview_memory", {}))
    memory_report = memory.get_report_data()

    contradiction_text = "None detected."
    if memory_report.get("contradictions"):
        parts = []
        for c in memory_report["contradictions"]:
            parts.append(
                f"  - Turn {c['turn_a']}: \"{c['claim_a']}\" vs Turn {c['turn_b']}: \"{c['claim_b']}\" [{c['severity']}]"
            )
        contradiction_text = "\n".join(parts)

    specificity_text = "No data."
    spec_scores = memory_report.get("specificity_scores", [])
    if spec_scores:
        avg_spec = sum(s.get("specificity", 5) for s in spec_scores) / len(spec_scores)
        avg_real = sum(s.get("realism", 5) for s in spec_scores) / len(spec_scores)
        specificity_text = f"Average specificity: {avg_spec:.1f}/10, Average realism: {avg_real:.1f}/10"

    strongest_text = "None recorded."
    if memory_report.get("strongest_moments"):
        parts = [f"  - Turn {m['turn']} ({m['skill']}): score {m['score']}" for m in memory_report["strongest_moments"][:3]]
        strongest_text = "\n".join(parts)

    weakest_text = "None recorded."
    if memory_report.get("weakest_moments"):
        parts = [f"  - Turn {m['turn']} ({m['skill']}): score {m['score']}" for m in memory_report["weakest_moments"][:3]]
        weakest_text = "\n".join(parts)

    # ── LLM report generation ─────────────────────────────────────────────
    llm = get_report_llm()
    structured_llm = llm.with_structured_output(ReportEvaluation)
    report_prompt = f"""**Role:** {role_title} ({seniority})
**Overall Score:** {overall_score}/10 ({score_pct:.0f}%)
**Pass Threshold:** {pass_threshold_pct}%
**Result:** {"PASSED" if passed else "DID NOT PASS"}

**Domain Scores:**
{json.dumps(domain_scores, indent=2)}

**Contradictions Detected:**
{contradiction_text}

**Specificity & Realism:**
{specificity_text}

**Strongest Moments:**
{strongest_text}

**Weakest Moments:**
{weakest_text}

**Interview Transcript:**
{transcript_summary[:3000]}

Generate the final evaluation report."""

    try:
        report_data = structured_llm.invoke([
            SystemMessage(content=REPORT_SYSTEM_PROMPT),
            HumanMessage(content=report_prompt),
        ])
        
        report = report_data.model_dump()
        # Convert nested DomainScore models to dicts if needed
        for d, ds in report.get("domain_scores", {}).items():
            if hasattr(ds, "model_dump"):
                report["domain_scores"][d] = ds.model_dump()

    except Exception as exc:
        log.error("LLM report generation failed: %s — using computed report", exc)
        if passed:
            if score_pct >= 80:
                recommendation = "STRONG_HIRE"
            elif score_pct >= 70:
                recommendation = "HIRE"
            else:
                recommendation = "LEAN_HIRE"
        else:
            if score_pct >= 50:
                recommendation = "LEAN_NO_HIRE"
            else:
                recommendation = "NO_HIRE"

        report = {
            "overall_score": overall_score,
            "recommendation": recommendation,
            "domain_scores": domain_scores,
            "strengths": [],
            "areas_for_improvement": [],
            "hiring_confidence": score_pct / 100,
            "executive_summary": (
                f"Candidate interviewed for {role_title} ({seniority}). "
                f"Overall score: {overall_score}/10 ({score_pct:.0f}%). "
                f"{'Passed' if passed else 'Did not pass'} the {pass_threshold_pct}% threshold."
            ),
        }

    # ── Soft Skills Aggregate ──────────────────────────────────────────────
    soft_scores_list = state.get("soft_skills_scores", [])
    soft_summary = {}
    if soft_scores_list:
        try:
            from evaluation.soft_skills_evaluator import SoftSkillsEvaluator
            sse = SoftSkillsEvaluator()
            soft_summary = sse.aggregate_session(soft_scores_list)
            log.info(
                "report_gen: soft skills overall=%.1f (%s)",
                soft_summary.get("overall_soft_score", 0),
                soft_summary.get("feedback", "")[:60],
            )
        except Exception as exc:
            log.warning("Soft skills aggregation failed: %s", exc)

    # ── Security & Proctoring ──────────────────────────────────────────────
    proctoring = {
        "flags": [],
        "avg_latency_ms": 0.0,
        "suspicious_turns": []
    }
    
    latencies = []
    for entry in transcript:
        if entry.get("role") == "candidate":
            lat = entry.get("latency_ms", 0.0)
            if lat > 0:
                latencies.append(lat)
                if lat > 20000:
                    proctoring["flags"].append("High Latency")
                    proctoring["suspicious_turns"].append({
                        "turn": entry.get("turn"),
                        "reason": f"Response took {lat/1000:.1f}s",
                        "latency_ms": lat
                    })
                elif lat < 1500 and len(entry.get("content", "")) > 100:
                    proctoring["flags"].append("Suspiciously Fast")
                    proctoring["suspicious_turns"].append({
                        "turn": entry.get("turn"),
                        "reason": f"Long answer provided in {lat/1000:.1f}s (possible copy-paste/reading)",
                        "latency_ms": lat
                    })
                    
    if latencies:
        proctoring["avg_latency_ms"] = sum(latencies) / len(latencies)
        
    proctoring["flags"] = list(set(proctoring["flags"]))

    # Add metadata
    report["role_id"] = state.get("role_id", "")
    report["role_title"] = role_title
    report["seniority"] = seniority
    report["session_id"] = state.get("session_id", "")
    report["total_turns"] = state.get("turn", 0)
    report["phases_covered"] = list(set(
        e.get("question_id", "").split("-")[0]
        for e in transcript if e.get("role") == "interviewer"
    ))
    report["soft_skills"] = soft_summary
    report["proctoring"] = proctoring

    # ── Memory-Based Analysis ─────────────────────────────────────────────
    report["contradiction_analysis"] = {
        "total_detected": len(memory_report.get("contradictions", [])),
        "contradictions": memory_report.get("contradictions", []),
    }
    report["specificity_breakdown"] = memory_report.get("specificity_scores", [])
    report["depth_map"] = memory_report.get("depth_signals", {})
    report["answer_patterns"] = memory_report.get("answer_patterns", {})
    report["strongest_moments"] = memory_report.get("strongest_moments", [])
    report["weakest_moments"] = memory_report.get("weakest_moments", [])

    # ── Dimension Coverage Analysis ────────────────────────────────────────
    dimensions_covered = state.get("dimensions_covered", {})
    dimension_analysis = {
        "counts": dimensions_covered,
        "total_questions": sum(dimensions_covered.values()) if dimensions_covered else 0,
        "gaps": [d for d in COVERAGE_DIMENSIONS if dimensions_covered.get(d, 0) == 0],
        "strongest": max(dimensions_covered, key=dimensions_covered.get) if dimensions_covered else "N/A",
    }
    report["dimension_coverage"] = dimension_analysis

    if dimension_analysis["gaps"]:
        log.info(
            "report_gen: dimension gaps detected — untested dimensions: %s",
            ", ".join(dimension_analysis["gaps"]),
        )

    log.info(
        "report_gen: overall=%.1f/10 (%s), recommendation=%s, contradictions=%d",
        report.get("overall_score", 0),
        "PASS" if passed else "FAIL",
        report.get("recommendation", "N/A"),
        len(memory_report.get("contradictions", [])),
    )

    return {"report": report}

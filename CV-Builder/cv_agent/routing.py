"""
cv_agent.routing
================
Adaptive loop controller, candidate generation, and selection logic.
"""

from __future__ import annotations

import re
import threading
import time
from hashlib import md5
from typing import Any, Callable, Dict, List, Optional, Tuple

from concurrent.futures import ThreadPoolExecutor
from langgraph.graph import END

from cv_agent.cache import LRUCache, get_cache
from cv_agent.config import PipelineConfig, logger
from cv_agent.content_checker import apply_content_fixes, check_content, format_issues_as_feedback
from cv_agent.gpu_queue import _gpu_queue
from cv_agent.hallucination_guard import _guard
from cv_agent.judges import run_ensemble
from cv_agent.memory import MemoryModule
from cv_agent.model_manager import ModelManager, chat
from cv_agent.prompts import WRITER_SYSTEM, REVISION_PROMPTS, WRITER_SYSTEM_NO_EXP, REVISION_PROMPTS_NO_EXP
from cv_agent.schemas import (
    CVAgentState, JDContext, JudgeOutput, LoopDecision,
    UserProfile, EnsembleResult,
)
from cv_agent.utils import jd_hash as _jd_hash


# ==============================================================================
# CANDIDATE CONFIGS
# ==============================================================================

CANDIDATE_CONFIGS: List[Dict[str, Any]] = [
    {"label": "A", "strategy": "revise",      "temperature": 0.70},
    {"label": "B", "strategy": "restructure", "temperature": 0.65},
    {"label": "C", "strategy": "keywords",    "temperature": 0.75},
    {"label": "D", "strategy": "revise",      "temperature": 0.80},
    {"label": "E", "strategy": "regenerate",  "temperature": 0.90},
]


# ==============================================================================
# ADAPTIVE LOOP CONTROLLER
# ==============================================================================

def adaptive_decide(state: CVAgentState, cfg: PipelineConfig) -> LoopDecision:
    scores = state.score_history
    weighted = state.last_scores

    if not weighted:
        return LoopDecision(action="revise", reason="no scores yet")

    # Priority 1: hard stop
    if state.iteration >= state.max_iterations:
        return LoopDecision(
            action="finalize",
            reason=f"max iterations ({state.max_iterations}) reached",
        )

    # Priority 2: hallucination
    if state.has_hallucination:
        return LoopDecision(action="revise", reason="hallucination detected")

    # Priority 3: score below threshold
    if not weighted.passes(state.score_threshold):
        strategy: str = {
            "ATS readiness":    "keywords",
            "Skills relevance": "keywords",
            "Impact":           "revise",
            "Structure":        "restructure",
            "Clarity":          "revise",
        }.get(weighted.lowest_metric(), "revise")
        return LoopDecision(
            action=strategy,  # type: ignore[arg-type]
            reason=f"targeting lowest metric: '{weighted.lowest_metric()}'",
        )

    # Priority 4: plateau detection
    delta = (scores[-1] - scores[-2]) if len(scores) >= 2 else 999.0
    new_stagnation = (state.stagnation_count + 1) if delta < cfg.plateau_threshold else 0

    if new_stagnation >= cfg.stagnation_limit:
        return LoopDecision(
            action="regenerate",
            reason=f"score plateau: delta={delta:.1f} for {new_stagnation} consecutive iterations",
        )

    # Priority 5: all good
    return LoopDecision(
        action="finalize",
        reason="all conditions satisfied (score + no hallucination)",
    )


def _route_decision(state: CVAgentState) -> str:
    """Return the END sentinel or 'writer_node'."""
    if state.loop_complete or state.error:
        return END
    return "writer_node"


# ==============================================================================
# WRITER PROMPT BUILDER
# ==============================================================================

def _build_writer_prompt(
    state: CVAgentState, jd_context: JDContext,
    last_scores: Optional[JudgeOutput], strategy: str,
    current_cv: str, memory_block: str,
) -> Tuple[str, str]:
    profile = state.user_profile

    # Select the right prompt set based on experience
    if profile.has_no_experience:
        writer_sys = WRITER_SYSTEM_NO_EXP
        revision_prompts = REVISION_PROMPTS_NO_EXP
    else:
        writer_sys = WRITER_SYSTEM
        revision_prompts = REVISION_PROMPTS

    system = revision_prompts.get(strategy, revision_prompts["revise"]) if current_cv else writer_sys

    if not current_cv:
        user = f"Write a professional CV for:\n\n{profile.to_context()}{jd_context.inject_block()}{memory_block}"
    else:
        def ul(items: List[str]) -> str:
            return "\n".join(f"  - {i}" for i in items)

        scores_block = ""
        if last_scores:
            scores_block = (
                f"\nScores: Clarity={last_scores.clarity_score} "
                f"Structure={last_scores.structure_score} Impact={last_scores.impact_score} "
                f"Skills={last_scores.skills_relevance_score} ATS={last_scores.ats_readiness_score} "
                f"Overall={last_scores.overall_score}\n"
                f"Weaknesses:\n{ul(last_scores.weaknesses)}\n"
                f"Suggestions:\n{ul(last_scores.improvement_suggestions)}\n"
                f"Rewrites:\n{ul(last_scores.rewrite_suggestions)}"
            )

        hallucination_block = ""
        if state.has_hallucination and state.hallucination_issues:
            hallucination_block = (
                "\n\nCRITICAL ERRORS (FIX THESE):\n"
                + "\n".join(f"- {i}" for i in state.hallucination_issues[:5])
                + "\nRemove ALL hallucinated words listed above.\n"
                + "Do NOT paraphrase them.\nDo NOT replace them with synonyms.\n"
                + "Only use technologies, skills, and terminology explicitly present in the candidate profile.\n"
            )

        # Content quality feedback from the previous iteration's inline check.
        # Strongly worded so the writer treats every item as a mandatory fix.
        content_feedback_block = ""
        if state.content_feedback:
            content_feedback_block = f"\n\n{state.content_feedback}\n"

        user = (
            f"Revise this CV.\n\nCURRENT CV:\n{current_cv}\n\n"
            f"FEEDBACK:\n{scores_block}{hallucination_block}{content_feedback_block}\n\n"
            f"PROFILE:\n{profile.to_context()}{jd_context.inject_block()}{memory_block}"
        )

    return system, user


# ==============================================================================
# POST-GENERATION SANITIZER (strict no-experience enforcement)
# ==============================================================================

_EXP_SECTION_RE = re.compile(
    r'(^|\n)(#{1,3}\s*(?:Professional\s+Experience|Work\s+Experience|Employment\s+History)'
    r'[^\n]*\n)'
    r'(.*?)'
    r'(?=\n#{1,3}\s|\Z)',
    re.IGNORECASE | re.DOTALL,
)


def _sanitize_no_exp_cv(cv_text: str, profile: 'UserProfile') -> str:
    """Strip any work experience section from the CV if the profile has no experience.

    This is the hard enforcement layer — even if the LLM ignores prompt
    instructions and generates a work experience section, it gets removed.
    """
    if not profile.has_no_experience:
        return cv_text

    cleaned = _EXP_SECTION_RE.sub('', cv_text)
    if cleaned != cv_text:
        logger.warning("Sanitizer removed work experience section from no-experience CV")
    return cleaned.strip()


# ==============================================================================
# CANDIDATE GENERATION
# ==============================================================================

def _generate_one_candidate(
    state: CVAgentState, jd_context: JDContext,
    last_scores: Optional[JudgeOutput], current_cv: str,
    memory_block: str, config: Dict[str, Any], cfg: PipelineConfig,
    profile_hash: str, jd_hash_val: str, iteration: int, cache: LRUCache,
) -> Tuple[str, str]:
    label = config["label"]
    hallucination_hash = md5(
        "|".join(state.hallucination_issues).encode()
    ).hexdigest()[:8] if state.hallucination_issues else "00000000"
    cache_key = LRUCache.make_key(
        profile_hash, jd_hash_val, iteration, label,
        namespace="cv", hallucination_hash=hallucination_hash,
    )
    cached = cache.get(cache_key)
    if cached is not None:
        logger.debug("Cache HIT — candidate %s iter=%d", label, iteration)
        return cached, config["strategy"]

    system, user = _build_writer_prompt(
        state, jd_context, last_scores, config["strategy"],
        current_cv, memory_block,
    )
    mm = ModelManager.get_instance()
    pipe = mm.writer_pipe(cfg)
    cv_text = _gpu_queue.submit(chat, pipe, system, user, config["temperature"])
    # Strict enforcement: strip any work experience section for no-experience profiles
    cv_text = _sanitize_no_exp_cv(cv_text, state.user_profile)
    cache.set(cache_key, cv_text)
    return cv_text, config["strategy"]


def generate_candidates(
    state: CVAgentState, n: int, memory: MemoryModule,
    cfg: PipelineConfig,
    status_callback: Optional[Callable[[str], None]] = None,
) -> Tuple[List[str], bool, List[str], Dict[int, Any], str]:
    """Generate n CV candidates + hallucination signal + per-candidate guard results.

    Returns:
        candidates: List of generated CV texts (length <= n).
        has_hallucination: True if any candidate was flagged by the guard.
        all_issues: Deduplicated list of all guard issues found.
        guard_map: Dict mapping candidate index (0-based) to its GuardResult,
                   so select_best_candidate can reuse these without re-running the guard.
        content_feedback: Formatted feedback string for the next iteration's prompt
                          (empty string when inline content fix is disabled).
    """
    profile = state.user_profile
    cache = get_cache(cfg)
    profile_hash = profile.profile_hash()
    jd_hash_val = _jd_hash(state.job_description)
    memory_block = memory.build_memory_block(state.session_id, profile.target_role)

    # FIX: Build exactly n configs. In regenerate mode take CANDIDATE_CONFIGS[:n]
    # and overwrite the first 1-2 entries with the regenerate config (index 4).
    # This guarantees len(configs)==n and keeps labels unique (no cache collisions).
    if state.current_strategy == "regenerate":
        configs = list(CANDIDATE_CONFIGS[:n])
        configs[0] = CANDIDATE_CONFIGS[4]
        if n >= 2:
            configs[1] = CANDIDATE_CONFIGS[4]
    else:
        configs = list(CANDIDATE_CONFIGS[:n])

    raw_candidates: List[Tuple[str, str, str]] = []
    for conf in configs:
        label = conf["label"]
        try:
            cv_text, strategy = _generate_one_candidate(
                state, state.jd_context, state.last_scores,
                state.current_cv, memory_block, conf, cfg,
                profile_hash, jd_hash_val, state.iteration, cache,
            )
            raw_candidates.append((cv_text, strategy, label))
        except Exception as e:
            msg = f"Candidate {label} generation failed: {e}"
            logger.error(msg)
            if status_callback:
                status_callback(f"ERROR: {msg}")

    if not raw_candidates:
        return [], False, [], {}

    # Guard validation (FIX C-3: thread-safe mutable container)
    all_issues: List[str] = []
    _guard_lock = threading.Lock()
    _hallu_flag: Dict[str, bool] = {"value": False}
    # Maps original list index → GuardResult so select_best_candidate can reuse them
    _guard_results: Dict[int, Any] = {}

    def _validate(args: Tuple[int, Tuple[str, str, str]]) -> Optional[str]:
        idx, (cv_text, strategy, label) = args
        guard = _guard.validate(cv_text, profile, state.jd_context)
        with _guard_lock:
            _guard_results[idx] = guard
        if not guard.passed:
            with _guard_lock:
                _hallu_flag["value"] = True
                all_issues.extend(guard.issues)
            msg = f"Candidate {label} flagged by hallucination guard: {guard.issues[:2]}"
            logger.warning(msg)
            if status_callback:
                status_callback(f"WARNING: {msg}")
        return cv_text

    with ThreadPoolExecutor(max_workers=min(len(raw_candidates), cfg.writer_workers)) as pool:
        validated = list(pool.map(_validate, enumerate(raw_candidates)))

    results = [v for v in validated if v is not None]
    has_hallucination = _hallu_flag["value"]
    all_issues = list(dict.fromkeys(all_issues))

    # ── Inline content check + fix ────────────────────────────────────────────
    # Run content quality analysis on each validated candidate and apply
    # rule-based + optional LLM fixes before handing them to the judge.
    # We also capture the first candidate's content result to build the
    # feedback block that will be injected into the NEXT iteration's prompt.
    content_feedback = ""
    if cfg.enable_inline_content_fix and results:
        fixed_results: List[str] = []
        first_content_result = None
        for idx, cv_text in enumerate(results):
            try:
                cr = check_content(cv_text, cfg)
                if first_content_result is None:
                    first_content_result = cr
                fixed_cv = apply_content_fixes(cv_text, cr, cfg)
                fixed_results.append(fixed_cv)
                logger.info(
                    "generate_candidates: candidate %d content fix — quality=%s issues=%d",
                    idx, cr.overall_quality, len(cr.issues),
                )
                if status_callback and cr.issues:
                    status_callback(
                        f"[iter {state.iteration+1}] Content fix candidate {idx}: "
                        f"{cr.overall_quality} ({len(cr.issues)} issue(s) fixed)"
                    )
            except Exception as e:
                logger.warning("generate_candidates: content fix failed for candidate %d: %s", idx, e)
                fixed_results.append(cv_text)
        results = fixed_results
        # Build judgmental feedback block for the next iteration
        if first_content_result:
            content_feedback = format_issues_as_feedback(first_content_result)
            logger.debug("generate_candidates: content_feedback block = %d chars", len(content_feedback))

    return results, has_hallucination, all_issues, _guard_results, content_feedback


def select_best_candidate(
    candidates: List[str], jd_context: JDContext, cfg: PipelineConfig,
    profile_hash: str = "", jd_hash_val: str = "", iteration: int = 0,
    profile: Optional[UserProfile] = None,
    guard_results: Optional[Dict[int, Any]] = None,
) -> Tuple[str, EnsembleResult]:
    """Score candidates sequentially (GPU-safe) and return the best. FIX C-4.

    Args:
        guard_results: Optional pre-computed guard results from generate_candidates
                       (Dict mapping candidate index → GuardResult).  When provided,
                       the hallucination guard is NOT re-run, eliminating the double
                       validation that was previously happening on every candidate.
    """
    best_cv = candidates[0]
    best_result: Optional[EnsembleResult] = None
    best_score = -1

    for idx, cv in enumerate(candidates):
        try:
            result = run_ensemble(
                cv, jd_context, cfg,
                cache_key_prefix=f"{profile_hash}:{jd_hash_val}:{iteration}",
                profile=profile,
            )
            raw_score = result.weighted.overall_score
            effective_score = raw_score
            if profile is not None:
                # Reuse pre-computed guard result when available; otherwise run guard
                if guard_results is not None and idx in guard_results:
                    guard = guard_results[idx]
                else:
                    guard = _guard.validate(cv, profile, jd_context)
                if not guard.passed:
                    penalty = int(cfg.hallucination_penalty)
                    effective_score = max(0, raw_score - penalty)
                    logger.debug(
                        "select_best_candidate: penalised %d pts (raw=%d → effective=%d)",
                        penalty, raw_score, effective_score,
                    )
            if effective_score > best_score:
                best_score, best_cv, best_result = effective_score, cv, result
        except Exception as e:
            logger.warning("Candidate scoring failed: %s", e)

    if best_result is None:
        best_result = run_ensemble(candidates[0], jd_context, cfg, profile=profile)

    return best_cv, best_result

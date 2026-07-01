"""
cv_agent.pipeline
=================
LangGraph nodes (writer, judge, router), graph builder, and pipeline runner.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Callable, Dict, Optional

from langgraph.graph import StateGraph, START, END

from cv_agent.cache import get_cache
from cv_agent.config import PipelineConfig, logger
from cv_agent.content_checker import auto_categorize_skills, check_content
from cv_agent.gpu_queue import _gpu_queue
from cv_agent.memory import MemoryModule
from cv_agent.rag import RAGModule
from cv_agent.routing import (
    adaptive_decide, _route_decision,
    generate_candidates, select_best_candidate,
)
from cv_agent.schemas import (
    CVAgentState, IterationRecord, LoopDecision,
    PipelineResult, UserProfile,
)
from cv_agent.utils import timed_node, jd_hash as _jd_hash


# ==============================================================================
# LANGGRAPH NODES
# ==============================================================================

@timed_node("writer_node")
def writer_node(
    state: CVAgentState, cfg: PipelineConfig,
    memory: MemoryModule,
    status_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    if state.loop_complete or state.iteration >= state.max_iterations:
        logger.info(
            "writer_node skipped — loop complete (loop_complete=%s, iteration=%d, max=%d)",
            state.loop_complete, state.iteration, state.max_iterations,
        )
        return {"loop_complete": True, "iteration": state.iteration}

    iteration = state.iteration + 1
    logger.info(
        "writer_node iter=%d strategy=%s session=%s",
        iteration, state.current_strategy, state.session_id,
    )

    msg = f"[iter {iteration}] Generating {cfg.num_candidates} candidate(s) (strategy: {state.current_strategy})..."
    if status_callback:
        status_callback(msg)

    candidates, any_hallucination, hallucination_issues, guard_results, content_feedback = generate_candidates(
        state, cfg.num_candidates, memory, cfg, status_callback
    )

    if not candidates:
        err = "All candidates failed generation — aborting writer node."
        logger.error(err)
        return {
            "iteration": iteration, "node_errors": [err],
            "error": err, "loop_complete": True,
        }

    logger.info("writer_node generated %d candidate(s)", len(candidates))
    return {
        "iteration": iteration,
        "candidates": candidates,
        "has_hallucination": any_hallucination,
        "hallucination_issues": hallucination_issues,
        "guard_results": guard_results,
        "content_feedback": content_feedback,
    }


@timed_node("judge_node")
def judge_node(
    state: CVAgentState, cfg: PipelineConfig,
    memory: MemoryModule,
    status_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    logger.info(
        "judge_node iter=%d candidates=%d session=%s",
        state.iteration, len(state.candidates), state.session_id,
    )

    if not state.candidates:
        err = "judge_node received no candidates — skipping."
        logger.warning(err)
        return {"node_errors": [err]}

    msg = f"[iter {state.iteration}] Scoring {len(state.candidates)} candidate(s)..."
    if status_callback:
        status_callback(msg)

    profile_hash = state.user_profile.profile_hash()
    jd_hash_val = _jd_hash(state.job_description)

    t0 = time.time()
    best_cv, ensemble = select_best_candidate(
        state.candidates, state.jd_context, cfg,
        profile_hash=profile_hash, jd_hash_val=jd_hash_val,
        iteration=state.iteration, profile=state.user_profile,
        guard_results=state.guard_results or None,
    )
    weighted = ensemble.weighted

    if state.has_hallucination:
        effective_score = max(0, weighted.overall_score - int(cfg.hallucination_penalty))
        weighted = weighted.model_copy(update={"overall_score": effective_score})

    duration = round(time.time() - t0, 2)
    score_history = state.score_history + [float(weighted.overall_score)]

    if state.session_id:
        memory.record_iteration(
            state.session_id, state.iteration, weighted, state.current_strategy
        )

    logger.info(
        "judge_node iter=%d overall_score=%d duration=%.1fs",
        state.iteration, weighted.overall_score, duration,
        extra={"score": weighted.overall_score, "iteration": state.iteration, "duration_ms": int(duration * 1000)},
    )

    updates: dict = {
        "current_cv": best_cv,
        "last_ensemble": ensemble,
        "last_scores": weighted,
        "score_history": score_history,
        "iteration_history": [
            IterationRecord(
                iteration=state.iteration, cv_text=best_cv,
                ensemble=ensemble,
                decision=LoopDecision(action="revise", reason="pending router decision"),
                strategy=state.current_strategy, duration_s=duration,
            )
        ],
    }

    # ── Best-CV tracking: keep the highest-scoring CV seen across all iterations ──
    current_score = float(weighted.overall_score)
    if current_score > state.best_cv_score:
        logger.info(
            "judge_node: new best CV — score %.1f (was %.1f) at iter=%d",
            current_score, state.best_cv_score, state.iteration,
        )
        updates["best_cv_score"] = current_score
        updates["best_cv_text"] = best_cv
    else:
        logger.info(
            "judge_node: score %.1f ≤ best %.1f — best_cv_text unchanged (iter=%d)",
            current_score, state.best_cv_score, state.iteration,
        )

    return updates


@timed_node("router_node")
def router_node(
    state: CVAgentState, cfg: PipelineConfig,
    memory: MemoryModule,
    status_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    logger.info(
        "router_node iter=%d score=%s hallucination=%s session=%s",
        state.iteration,
        state.last_scores.overall_score if state.last_scores else "N/A",
        state.has_hallucination, state.session_id,
    )

    decision = adaptive_decide(state, cfg)
    scores = state.score_history
    delta = (scores[-1] - scores[-2]) if len(scores) >= 2 else 999.0
    new_stagnation = (state.stagnation_count + 1) if delta < cfg.plateau_threshold else 0
    finalize = decision.action == "finalize"

    if finalize and state.has_hallucination and state.iteration < state.max_iterations:
        logger.error(
            "router_node: finalize with active hallucination at iter=%d",
            state.iteration,
        )

    msg = (
        f"[iter {state.iteration}] Decision: {decision.action} — {decision.reason} "
        f"| score={state.last_scores.overall_score if state.last_scores else 'N/A'}/100"
        f" | hallucination={state.has_hallucination}"
    )
    if status_callback:
        status_callback(msg)

    updates: Dict[str, Any] = {
        "current_strategy": decision.action,
        "stagnation_count": new_stagnation if not finalize else state.stagnation_count,
        "delta_history": state.delta_history + [delta],
        "loop_complete": finalize,
    }

    if finalize:
        if (
            state.last_scores
            and state.last_scores.overall_score >= state.score_threshold
            and state.session_id
        ):
            memory.record_success(
                state.user_profile.target_role,
                state.last_scores.overall_score,
                state.current_cv[:800],
            )
        # Prefer the best-ever CV over the last-iteration CV when finalizing.
        # This ensures a regression in the final iteration doesn't degrade the output.
        best_for_final = state.best_cv_text or state.current_cv
        updates["final_cv"] = best_for_final
        logger.info(
            "router_node FINALIZE — %d iterations, final_score=%s, "
            "best_cv_score=%.1f, using_best=%s",
            state.iteration,
            state.last_scores.overall_score if state.last_scores else "N/A",
            state.best_cv_score,
            best_for_final != state.current_cv,
        )

    return updates


# ==============================================================================
# GRAPH BUILDER
# ==============================================================================

def build_graph(
    cfg: PipelineConfig, memory: MemoryModule,
    status_callback: Optional[Callable[[str], None]] = None,
) -> Any:
    def _writer(state: CVAgentState) -> Dict[str, Any]:
        return writer_node(state, cfg, memory, status_callback)

    def _judge(state: CVAgentState) -> Dict[str, Any]:
        return judge_node(state, cfg, memory, status_callback)

    def _router(state: CVAgentState) -> Dict[str, Any]:
        return router_node(state, cfg, memory, status_callback)

    graph = StateGraph(CVAgentState)
    graph.add_node("writer_node", _writer)
    graph.add_node("judge_node", _judge)
    graph.add_node("router_node", _router)

    graph.add_edge(START, "writer_node")
    graph.add_edge("writer_node", "judge_node")
    graph.add_edge("judge_node", "router_node")
    graph.add_conditional_edges(
        "router_node", _route_decision,
        {"writer_node": "writer_node", END: END},
    )

    logger.info("LangGraph compiled: writer_node -> judge_node -> router_node -> (conditional)")
    return graph.compile()


# ==============================================================================
# PIPELINE RUNNER
# ==============================================================================

def run_pipeline(
    profile: UserProfile, job_description: str = "",
    parsed_resume: str = "", config: Optional[PipelineConfig] = None,
    session_id: str = "",
    status_callback: Optional[Callable[[str], None]] = None,
) -> PipelineResult:
    cfg = config or PipelineConfig()
    sid = session_id or str(uuid.uuid4())[:8]
    t_wall = time.perf_counter()

    logger.info(
        "run_pipeline START candidate=%s role=%s session=%s max_iter=%d",
        profile.full_name, profile.target_role, sid, cfg.max_iterations,
        extra={"session_id": sid},
    )

    _gpu_queue.start()
    get_cache(cfg)

    # ── Auto-categorize skills via LLM if not already categorized ──
    if profile.skills and not profile.skill_categories:
        if status_callback:
            status_callback("[pre] Auto-categorizing skills via LLM...")
        profile = profile.model_copy(
            update={"skill_categories": auto_categorize_skills(
                profile.skills, profile.target_role, profile.target_industry, cfg,
            )}
        )
        logger.info(
            "run_pipeline: auto-categorized %d skills into %d categories",
            len(profile.skills),
            len(profile.skill_categories) if profile.skill_categories else 0,
        )

    rag = RAGModule(cfg.ontology_path)
    memory = MemoryModule(cfg.db_path, session_id=sid)
    jd_context = rag.extract(job_description, profile.target_role, cfg)

    initial_state = CVAgentState(
        user_profile=UserProfile.model_construct(**profile.model_dump()),
        job_description=job_description,
        parsed_resume=parsed_resume,
        jd_context=jd_context,
        max_iterations=cfg.max_iterations,
        score_threshold=cfg.score_threshold,
        session_id=sid,
        intake_complete=True,
    )

    compiled = build_graph(cfg, memory, status_callback)
    config_dict = {"recursion_limit": cfg.max_iterations * 4 + 10}
    raw_state = compiled.invoke(initial_state, config=config_dict)

    if isinstance(raw_state, dict):
        final_state = CVAgentState(**raw_state)
    else:
        final_state = raw_state

    # ── Post-pipeline content quality check ──
    # Use the best-ever CV (not just the last iteration's output).
    # final_state.final_cv is already set to best_cv_text by router_node,
    # but we add a second fallback chain for safety.
    final_cv_text = (
        final_state.final_cv
        or final_state.best_cv_text
        or final_state.current_cv
    )
    content_result = None
    if final_cv_text:
        if status_callback:
            status_callback("[post] Running content quality check...")
        try:
            content_result = check_content(final_cv_text, cfg)
            logger.info(
                "run_pipeline: content check — quality=%s issues=%d",
                content_result.overall_quality, len(content_result.issues),
            )
        except Exception as e:
            logger.warning("run_pipeline: content check failed: %s", e)

    latency_ms = int((time.perf_counter() - t_wall) * 1000)
    final_score = (
        final_state.last_scores.overall_score if final_state.last_scores else "N/A"
    )
    logger.info(
        "run_pipeline END session=%s iterations=%d final_score=%s latency_ms=%d",
        sid, final_state.iteration, final_score, latency_ms,
        extra={"session_id": sid, "duration_ms": latency_ms},
    )

    return PipelineResult(
        session_id=sid,
        candidate_name=profile.full_name,
        target_role=profile.target_role,
        total_iterations=final_state.iteration,
        final_cv=final_cv_text,
        final_scores=final_state.last_scores,
        score_trajectory=final_state.score_history,
        node_errors=final_state.node_errors,
        jd_keywords=final_state.jd_context.keywords[:20],
        total_latency_ms=latency_ms,
        content_check=content_result,
    )

"""
Phase 2 — Orchestration Entry Point
AI Voice Interview System

CLI for running, testing, and simulating the LangGraph interview flow.

Usage:
    python phase2_orchestration.py                              # interactive CLI interview
    python phase2_orchestration.py --role EXP-SOFT-SEN          # specific role
    python phase2_orchestration.py --simulate                   # auto-simulate with canned answers
    python phase2_orchestration.py --simulate --turns 5         # simulate 5 turns
    python phase2_orchestration.py --validate                   # run Gate 2 validation
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
import uuid

# Fix Windows console encoding for Unicode output
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from langgraph.checkpoint.memory import MemorySaver

from core.interview_state import InterviewState
from core.graph import build_graph

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("phase2")


# ─────────────────────────────────────────────────────────────────────────────
# Simulated answers for --simulate mode
# ─────────────────────────────────────────────────────────────────────────────

SIMULATED_ANSWERS = [
    "Python's GIL is the Global Interpreter Lock. It prevents multiple native threads "
    "from executing Python bytecodes simultaneously. This becomes a bottleneck in "
    "CPU-bound multi-threaded programs. For I/O-bound tasks, the GIL is released "
    "during I/O operations so threading still helps. For CPU-bound work, you'd use "
    "multiprocessing to bypass the GIL entirely, or use C extensions.",

    "I would design a distributed rate limiter using a sliding window approach with "
    "Redis. Each request increments a counter in a time-bucketed key. For distributed "
    "consistency, I'd use Redis Lua scripts for atomic check-and-increment. The trade-off "
    "is between strict accuracy and performance — a token bucket algorithm would give "
    "smoother rate limiting but requires more state management.",

    "In my previous role, I identified that our deployment process took 45 minutes "
    "and was error-prone. I led the initiative to implement CI/CD with GitHub Actions, "
    "containerized our services with Docker, and set up automated testing. The result "
    "was reducing deployment time to 8 minutes with zero-downtime deployments. "
    "I learned the importance of getting buy-in from the team early.",

    "For database indexing, I consider the query patterns first. B-tree indexes work "
    "well for range queries and equality checks. For full-text search, I'd use inverted "
    "indexes. Composite indexes should follow the leftmost prefix rule. I always check "
    "the query execution plan with EXPLAIN ANALYZE to verify index usage.",

    "I handle conflicts by first seeking to understand the other person's perspective. "
    "In one case, a colleague and I disagreed on the architecture for a new service. "
    "I suggested we each prototype our approach for a day, then compare with objective "
    "criteria like latency, maintainability, and team familiarity. We ended up combining "
    "the best parts of both approaches.",

    "For system design, I'd start with requirements clarification — both functional and "
    "non-functional. Then I'd do a back-of-the-envelope estimation for traffic and storage. "
    "I'd design the API first, then the data model, then the high-level architecture with "
    "load balancers, application servers, caching layer, and database. Finally, I'd deep-dive "
    "into the most critical component.",

    "I approach testing with a pyramid strategy: many unit tests, fewer integration tests, "
    "and minimal E2E tests. For unit tests, I use pytest with fixtures and parametrize. "
    "I aim for high coverage on business logic but don't obsess over 100% coverage. "
    "Mocking is important but I prefer testing real integrations where practical.",

    "When I receive critical feedback, I try not to be defensive. I take time to process it, "
    "look for valid points, and create an action plan. For example, my manager once pointed "
    "out that I was making technical decisions without enough team input. I started holding "
    "RFC-style reviews for significant changes, which improved both the decisions and team morale.",

    "I ensure code quality through code reviews, linting, type checking, and automated tests. "
    "I use pre-commit hooks for formatting and basic checks. For Python, I use mypy for static "
    "typing, ruff for linting, and pytest for testing. I also do periodic refactoring sessions "
    "to address tech debt before it accumulates.",

    "My approach to learning new technologies starts with understanding the problem it solves "
    "and its trade-offs compared to alternatives. Then I read the official docs, build a small "
    "prototype, and only then start using it in production. I also contribute to internal "
    "knowledge sharing by writing tech notes and giving presentations.",
]


# ─────────────────────────────────────────────────────────────────────────────
# Interactive CLI Interview
# ─────────────────────────────────────────────────────────────────────────────

def run_interactive(role_id: str) -> dict:
    """Run an interactive CLI interview — type answers at the prompt."""
    graph = build_graph()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print("\n" + "=" * 60)
    print("  AI Voice Interview System -- Phase 2 CLI")
    print("=" * 60)
    print(f"  Role: {role_id}")
    print(f"  Session: {thread_id}")
    print("  Type your answers. Type 'quit' to end early.\n")

    # Initial invoke — runs session_init + question_gen, then pauses
    initial_state = {"role_id": role_id, "session_id": thread_id}
    state = graph.invoke(initial_state, config)

    while True:
        # Show the current question
        question = state.get("current_question")
        if not question:
            print("\n[No more questions. Generating report...]")
            break

        turn = state.get("turn", 0)
        phase = state.get("phase", "?")
        print(f"\n{'-' * 60}")
        print(f"  Turn {turn} | Phase: {phase} | Domain: {question.get('domain', '?')}")
        print(f"  Skill: {question.get('sub_skill', '?')} | Difficulty: {question.get('difficulty', '?')}")
        print(f"{'-' * 60}")
        print(f"\n  Q: {question['question_text']}\n")

        # Get candidate answer
        answer = input("  Your answer: ").strip()
        if answer.lower() in ("quit", "exit", "q"):
            print("\n[Ending interview early...]")
            # Force report generation by setting turn to max
            state = graph.update_state(
                config,
                {"transcript": [{"role": "candidate", "content": "Interview ended by candidate.", "turn": turn, "question_id": question.get("question_id"), "score": None}],
                 "turn": state.get("max_turns", 10)},
            )
            state = graph.invoke(None, config)
            break

        if not answer:
            answer = "I'm not sure about that. Could you rephrase the question?"

        # Inject candidate answer into state and resume
        graph.update_state(
            config,
            {
                "transcript": [{
                    "role": "candidate",
                    "content": answer,
                    "turn": turn,
                    "question_id": question.get("question_id"),
                    "score": None,
                }],
            },
        )

        # Resume graph — runs answer_eval → router → next node
        state = graph.invoke(None, config)

        # Check if report is ready
        if state.get("report"):
            break

    # Print report
    report = state.get("report")
    if report:
        print_report(report)

    return state


# ─────────────────────────────────────────────────────────────────────────────
# Simulation Mode
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation(role_id: str, max_turns: int = 5) -> dict:
    """Run an automated simulation with pre-canned answers."""
    graph = build_graph()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print("\n" + "=" * 60)
    print("  AI Voice Interview System -- SIMULATION MODE")
    print("=" * 60)
    print(f"  Role: {role_id}")
    print(f"  Max turns: {max_turns}")
    print(f"  Session: {thread_id}\n")

    # Initial invoke
    initial_state = {"role_id": role_id, "session_id": thread_id, "max_turns": max_turns}
    state = graph.invoke(initial_state, config)

    sim_idx = 0
    while True:
        question = state.get("current_question")
        if not question:
            print("\n[No more questions]")
            break

        turn = state.get("turn", 0)
        phase = state.get("phase", "?")
        print(f"\n{'-' * 50}")
        print(f"  Turn {turn} | Phase: {phase}")
        print(f"  Q: {question['question_text'][:100]}...")

        # Pick a simulated answer
        answer = SIMULATED_ANSWERS[sim_idx % len(SIMULATED_ANSWERS)]
        sim_idx += 1
        print(f"  A: {answer[:100]}...")

        # Inject answer and resume
        graph.update_state(
            config,
            {
                "transcript": [{
                    "role": "candidate",
                    "content": answer,
                    "turn": turn,
                    "question_id": question.get("question_id"),
                    "score": None,
                }],
            },
        )

        state = graph.invoke(None, config)

        # Show evaluation
        transcript = state.get("transcript", [])
        for entry in reversed(transcript):
            if entry.get("role") == "evaluation":
                score = entry.get("score", "?")
                rationale = entry.get("content", "")[:100]
                print(f"  -> Score: {score}/10 -- {rationale}")
                break

        if state.get("report"):
            break

    # Print report
    report = state.get("report")
    if report:
        print_report(report)

    return state


# ─────────────────────────────────────────────────────────────────────────────
# Report Printer
# ─────────────────────────────────────────────────────────────────────────────

def print_report(report: dict) -> None:
    """Pretty-print the interview evaluation report."""
    print("\n" + "=" * 60)
    print("  FINAL INTERVIEW REPORT")
    print("=" * 60)

    print(f"\n  Role:            {report.get('role_title', 'N/A')} ({report.get('seniority', '')})")
    print(f"  Overall Score:   {report.get('overall_score', 0)}/10")
    print(f"  Recommendation:  {report.get('recommendation', 'N/A')}")
    print(f"  Confidence:      {report.get('hiring_confidence', 0):.0%}")
    print(f"  Total Turns:     {report.get('total_turns', 0)}")

    domain_scores = report.get("domain_scores", {})
    if domain_scores:
        print(f"\n  {'Domain Scores':}")
        for domain, info in domain_scores.items():
            if isinstance(info, dict):
                score = info.get("score", 0)
                print(f"    {domain}: {score}/10")
                skills = info.get("skills", {})
                for skill, s in skills.items():
                    print(f"      +-- {skill}: {s}/10")
            else:
                print(f"    {domain}: {info}")

    strengths = report.get("strengths", [])
    if strengths:
        print(f"\n  Strengths:")
        for s in strengths[:5]:
            print(f"    [+] {s}")

    improvements = report.get("areas_for_improvement", [])
    if improvements:
        print(f"\n  Areas for Improvement:")
        for a in improvements[:5]:
            print(f"    [-] {a}")

    summary = report.get("executive_summary", "")
    if summary:
        print(f"\n  Executive Summary:")
        # Word wrap at 55 chars
        words = summary.split()
        line = "    "
        for word in words:
            if len(line) + len(word) > 60:
                print(line)
                line = "    "
            line += word + " "
        if line.strip():
            print(line)

    print("\n" + "=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Gate 2 Validation
# ─────────────────────────────────────────────────────────────────────────────

def run_validation() -> bool:
    """
    Gate 2 validation — verify graph structure, state transitions,
    and router logic without requiring LLM calls.
    """
    results: list[tuple[str, bool, str]] = []

    def check(label: str, fn):
        try:
            ok, note = fn()
            results.append((label, ok, note))
            sym = "✓" if ok else "✗"
            log.info("  %s  %-45s  %s", sym, label, note)
        except Exception as exc:
            results.append((label, False, str(exc)))
            log.error("  ✗  %-45s  ERROR: %s", label, exc)

    log.info("\n── Gate 2 Validation ─────────────────────────────────")

    # 1. Graph compiles
    def test_graph_compiles():
        graph = build_graph()
        return True, f"{len(graph.nodes)} nodes"
    check("Graph compiles", test_graph_compiles)

    # 2. Session init produces valid state
    def test_session_init():
        from nodes import session_init as si
        state = si({"role_id": "EXP-SOFT-SEN"})
        required = ["session_id", "phases", "coverage_map", "skill_scores", "max_turns"]
        missing = [k for k in required if k not in state]
        return len(missing) == 0, f"missing={missing}" if missing else "all keys present"
    check("session_init produces valid state", test_session_init)

    # 3. Router returns valid routes
    def test_router_report():
        from router import interview_router
        state = {
            "turn": 10, "max_turns": 10, "chain_cursor": None,
            "coverage_map": {}, "phase": "Technical", "phases": ["Technical"],
            "depth_threshold": 5,
        }
        route = interview_router(state)
        return route == "report_gen", f"route={route}"
    check("Router: turn limit → report_gen", test_router_report)

    def test_router_chain():
        from router import interview_router
        state = {
            "turn": 2, "max_turns": 10,
            "chain_cursor": {"base_question_id": "Q1", "order": 2, "max_depth": 3},
            "coverage_map": {"Technical": {"Python": 3.0}},
            "phase": "Technical", "phases": ["Technical", "Behavioral"],
            "depth_threshold": 5,
        }
        route = interview_router(state)
        return route == "chain_follow_up", f"route={route}"
    check("Router: active chain → chain_follow_up", test_router_chain)

    def test_router_advance():
        from router import interview_router
        state = {
            "turn": 3, "max_turns": 10, "chain_cursor": None,
            "coverage_map": {"Technical": {"Python": 8.0, "Java": 7.0}},
            "phase": "Technical", "phases": ["Technical", "Behavioral"],
            "depth_threshold": 5,
        }
        route = interview_router(state)
        return route == "advance_phase", f"route={route}"
    check("Router: phase covered → advance_phase", test_router_advance)

    def test_router_continue():
        from router import interview_router
        state = {
            "turn": 2, "max_turns": 10, "chain_cursor": None,
            "coverage_map": {"Technical": {"Python": 3.0, "Java": 2.0}},
            "phase": "Technical", "phases": ["Technical", "Behavioral"],
            "depth_threshold": 5,
        }
        route = interview_router(state)
        return route == "question_gen", f"route={route}"
    check("Router: gaps remain → question_gen", test_router_continue)

    # 4. Advance phase works
    def test_advance_phase():
        from router import advance_phase as ap
        state = {"phases": ["Technical", "Behavioral", "Problem Solving"], "phase": "Technical"}
        result = ap(state)
        return result["phase"] == "Behavioral", f"next_phase={result['phase']}"
    check("advance_phase transitions correctly", test_advance_phase)

    # Summary
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    log.info("\n── Gate 2 Result: %d/%d checks passed ─────────────────", passed, total)
    return passed == total


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 — LangGraph Orchestration")
    parser.add_argument("--role", default="EXP-SOFT-SEN",
                        help="Role ID to interview for (default: EXP-SOFT-SEN)")
    parser.add_argument("--simulate", action="store_true",
                        help="Run automated simulation with canned answers")
    parser.add_argument("--turns", type=int, default=5,
                        help="Max turns for simulation (default: 5)")
    parser.add_argument("--validate", action="store_true",
                        help="Run Gate 2 validation suite")
    args = parser.parse_args()

    if args.validate:
        passed = run_validation()
        raise SystemExit(0 if passed else 1)

    if args.simulate:
        run_simulation(role_id=args.role, max_turns=args.turns)
    else:
        run_interactive(role_id=args.role)


if __name__ == "__main__":
    main()

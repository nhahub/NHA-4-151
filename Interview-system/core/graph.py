"""
Phase 2 — LangGraph Assembly
AI Voice Interview System

Wires all nodes and edges into a compiled LangGraph StateGraph.
The graph pauses (interrupt_before) at answer_eval so the Voice I/O
layer (Phase 3) or CLI can inject the candidate's answer.

Usage:
    from graph import build_graph
    graph = build_graph()
    # invoke: graph.invoke({"role_id": "EXP-SOFT-SEN"})
"""

from __future__ import annotations

import logging

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from core.interview_state import InterviewState
from core.nodes import session_init, question_gen, answer_eval, report_gen, summarize_context
from core.router import interview_router, chain_follow_up, advance_phase

log = logging.getLogger("phase2.graph")


def build_graph(checkpointer=None) -> StateGraph:
    """
    Build and compile the interview orchestration graph.

    Flow:
        session_init → question_gen → [INTERRUPT] → answer_eval
                                                         ↓
                                                  interview_router
                                               /       |         \\
                                    chain_follow_up  advance   question_gen
                                        ↓           phase ↓
                                   [INTERRUPT]   question_gen
                                        ↓
                                   answer_eval
                                        ↓              report_gen → END
    """
    builder = StateGraph(InterviewState)

    # ── Add nodes ─────────────────────────────────────────────────────────
    builder.add_node("session_init", session_init)
    builder.add_node("question_gen", question_gen)
    builder.add_node("answer_eval", answer_eval)
    builder.add_node("summarize_context", summarize_context)
    builder.add_node("chain_follow_up", chain_follow_up)
    builder.add_node("advance_phase", advance_phase)
    builder.add_node("report_gen", report_gen)

    # ── Entry point ───────────────────────────────────────────────────────
    builder.set_entry_point("session_init")

    # ── Edges ─────────────────────────────────────────────────────────────
    # session_init always leads to first question
    builder.add_edge("session_init", "question_gen")

    # After question is asked, we wait for candidate answer then evaluate
    # The interrupt_before=["answer_eval"] makes this a human-in-the-loop step
    builder.add_edge("question_gen", "answer_eval")

    # Chain follow-up also leads to answer_eval (after interrupt for answer)
    builder.add_edge("chain_follow_up", "answer_eval")

    # After evaluation, summarize context if needed
    builder.add_edge("answer_eval", "summarize_context")

    # After summarize_context, the router decides next step
    builder.add_conditional_edges(
        "summarize_context",
        interview_router,
        {
            "chain_follow_up": "chain_follow_up",
            "advance_phase": "advance_phase",
            "question_gen": "question_gen",
            "report_gen": "report_gen",
        },
    )

    # After advancing phase, go to question_gen for the new phase
    builder.add_edge("advance_phase", "question_gen")

    # Report generation is the terminal node
    builder.add_edge("report_gen", END)

    # ── Compile ───────────────────────────────────────────────────────────
    # interrupt_before=["answer_eval"] pauses after a question is presented,
    # allowing the Voice I/O layer (Phase 3) to collect the candidate's answer
    if checkpointer is None:
        try:
            import sqlite3
            conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
            checkpointer = SqliteSaver(conn)
            # We also need to run setup() to create tables if they don't exist
            checkpointer.setup()
        except Exception as exc:
            log.warning("Could not initialize SqliteSaver, falling back to MemorySaver: %s", exc)
            checkpointer = MemorySaver()

    graph = builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["answer_eval"],
    )

    log.info("Interview graph compiled with %d nodes", len(builder.nodes))
    return graph

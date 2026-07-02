from typing import List, Optional
from langgraph.graph import StateGraph, START, END

from core.state import AgentState, DATE_FILTER_LABELS
from core.nodes import (
    extract_resume_info,
    quality_check,
    route_after_quality_check,
    discover_target_sites,
    generate_search_queries,
    search_jobs,
    match_and_score,
    deep_research_top_jobs,
    compile_report,
    generate_cover_letters,
    generate_pdf_report
)

def build_agent() -> object:
    """Build and compile the LangGraph research pipeline."""
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("extract_resume_info",    extract_resume_info)
    graph.add_node("quality_check",          quality_check)
    graph.add_node("discover_target_sites",  discover_target_sites)
    graph.add_node("generate_search_queries", generate_search_queries)
    graph.add_node("search_jobs",            search_jobs)
    graph.add_node("match_and_score",        match_and_score)
    graph.add_node("deep_research_top_jobs", deep_research_top_jobs)
    graph.add_node("compile_report",         compile_report)
    graph.add_node("generate_cover_letters", generate_cover_letters)
    graph.add_node("generate_pdf_report",    generate_pdf_report)

    # ---- Edges ----
    graph.add_edge(START, "extract_resume_info")
    graph.add_edge("extract_resume_info", "quality_check")

    graph.add_conditional_edges(
        "quality_check",
        route_after_quality_check,
        {
            "extract_resume_info":    "extract_resume_info",
            "generate_search_queries": "discover_target_sites",
        },
    )

    graph.add_edge("discover_target_sites",   "generate_search_queries")
    graph.add_edge("generate_search_queries", "search_jobs")
    graph.add_edge("search_jobs",             "match_and_score")
    graph.add_edge("match_and_score",         "deep_research_top_jobs")
    graph.add_edge("deep_research_top_jobs",  "compile_report")
    graph.add_edge("compile_report",          "generate_cover_letters")
    graph.add_edge("generate_cover_letters",  "generate_pdf_report")
    graph.add_edge("generate_pdf_report",     END)

    return graph.compile()

def run_agent(
    resume_text:      str,
    target_sites:     Optional[List[str]] = None,
    date_filter_days: Optional[int]       = 7,
    location:         str                 = ""
) -> AgentState:
    """
    Main entry point for the research agent.
    """
    agent         = build_agent()
    recency_label = DATE_FILTER_LABELS.get(date_filter_days, "any time")

    default_sites = [
        "linkedin.com/jobs",
        "indeed.com",
        "glassdoor.com",
        "wuzzuf.net",          # MENA
        "bayt.com",            # MENA
        "forasna.com",         # Egypt
        "naukrigulf.com",      # Gulf
        "weworkremotely.com",
        "remoteok.com",
        "wellfound.com",
    ]

    initial_state: AgentState = {
        "resume_text":          resume_text,
        "extracted_skills":     [],
        "soft_skills":          [],
        "experience_level":     "",
        "job_titles":           [],
        "education":            "",
        "search_queries":       [],
        "raw_job_results":      [],
        "matched_jobs":         [],
        "deep_research_results": [],
        "research_report":      "",
        "cover_letters":        [],
        "pdf_report_path":      "",
        "target_sites":         target_sites or default_sites,
        "date_filter_days":     date_filter_days,
        "location":             location,
        "status_log": [
            f"🚀 Agent initialised — searching jobs {recency_label}" + (f" in {location}" if location else "") + "...",
        ],
        "iteration": 0,
        "messages":  [],
    }

    return agent.invoke(initial_state)

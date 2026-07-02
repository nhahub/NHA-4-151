from typing import TypedDict, List, Annotated, Optional
from langgraph.graph.message import add_messages

DATE_FILTER_OPTIONS = {
    "Last 24 hours": 1,
    "Last 7 days": 7,
    "Last 30 days": 30,
    "Last 3 months": 90,
    "Any time": None,
}

DATE_FILTER_LABELS = {
    1:    "posted in the last 24 hours",
    7:    "posted in the last 7 days",
    30:   "posted in the last 30 days",
    90:   "posted in the last 3 months",
    None: "any time",
}

MAX_EXTRACTION_RETRIES = 2

class AgentState(TypedDict):
    resume_text:          str
    extracted_skills:     List[str]
    soft_skills:          List[str]
    experience_level:     str
    job_titles:           List[str]
    education:            str
    search_queries:       List[str]
    raw_job_results:      List[dict]
    matched_jobs:         List[dict]
    deep_research_results: List[dict]
    research_report:      str
    cover_letters:        List[dict]
    pdf_report_path:      str
    target_sites:         List[str]
    date_filter_days:     Optional[int]
    location:             str                # Added location filter
    status_log:           List[str]
    iteration:            int
    messages:             Annotated[List, add_messages]

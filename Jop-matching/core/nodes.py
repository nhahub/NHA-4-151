import json
import re
import asyncio
from datetime import datetime
from typing import List, Literal

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from json_repair import repair_json
from fpdf import FPDF

from core.state import AgentState, MAX_EXTRACTION_RETRIES, DATE_FILTER_LABELS
from core.utils import get_llm, get_tavily, with_retry

# ---------------------------------------------------------------------------
# Node 1 – Resume Extraction
# ---------------------------------------------------------------------------

def extract_resume_info(state: AgentState) -> AgentState:
    llm = get_llm()
    log = state.get("status_log", [])
    iteration = state.get("iteration", 0)

    if iteration == 0:
        log.append("🔍 Analysing resume and extracting skills...")
    else:
        log.append(f"🔁 Re-extracting resume info (quality-check retry {iteration})...")

    system_prompt = """You are an expert technical recruiter and career analyst.
Extract comprehensive information from the resume provided.
Return ONLY valid JSON with this exact schema:
{
  "technical_skills": ["skill1", "skill2", ...],
  "soft_skills": ["skill1", "skill2", ...],
  "experience_level": "junior|mid|senior|lead|executive",
  "years_of_experience": "X years",
  "job_titles": ["most suitable job title 1", "job title 2", "job title 3"],
  "education": "highest degree and field",
  "industries": ["industry1", "industry2"],
  "key_achievements": ["achievement1", "achievement2"],
  "languages": ["language1", "language2"],
  "certifications": ["cert1", "cert2"]
}
Be specific and comprehensive. Infer multiple relevant job titles."""

    extra = " Be extra thorough — a previous attempt produced incomplete data." if iteration > 0 else ""

    @with_retry
    def call_llm():
        return llm.invoke([
            SystemMessage(content=system_prompt + extra),
            HumanMessage(content=f"Extract information from this resume:\n\n{state['resume_text']}"),
        ])

    try:
        response = call_llm()
        raw = response.content
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        data = json.loads(json_match.group()) if json_match else {}
    except Exception:
        data = {}

    if not data:
        data = {
            "technical_skills": ["Python", "Machine Learning"],
            "soft_skills": ["Communication", "Problem Solving"],
            "experience_level": "mid",
            "years_of_experience": "3 years",
            "job_titles": ["Software Engineer", "ML Engineer"],
            "education": "Bachelor's in Computer Science",
            "industries": ["Technology"],
            "key_achievements": [],
            "languages": ["English"],
            "certifications": [],
        }

    log.append(f"✅ Extracted {len(data.get('technical_skills', []))} technical skills")
    log.append(f"✅ Identified role: {data.get('experience_level', 'mid').title()} level")

    return {
        **state,
        "extracted_skills":  data.get("technical_skills", []),
        "soft_skills":       data.get("soft_skills", []),
        "experience_level":  data.get("experience_level", "mid"),
        "job_titles":        data.get("job_titles", []),
        "education":         data.get("education", ""),
        "status_log":        log,
        "messages": [AIMessage(content=f"Resume analysed. Found skills: {', '.join(data.get('technical_skills', [])[:8])}...")],
    }

# ---------------------------------------------------------------------------
# Quality-Check node + conditional router
# ---------------------------------------------------------------------------

def quality_check(state: AgentState) -> AgentState:
    log = state.get("status_log", [])
    skills   = state.get("extracted_skills", [])
    titles   = state.get("job_titles", [])
    level    = state.get("experience_level", "")

    is_complete = len(skills) >= 3 and len(titles) >= 1 and level != ""

    if is_complete:
        log.append("✅ Quality check passed — extraction data is sufficient.")
        return {**state, "status_log": log}
    else:
        iteration = state.get("iteration", 0) + 1
        log.append(f"⚠️  Quality check FAILED (attempt {iteration}): incomplete extraction data.")
        return {**state, "iteration": iteration, "status_log": log}

def route_after_quality_check(state: AgentState) -> Literal["generate_search_queries", "extract_resume_info"]:
    skills  = state.get("extracted_skills", [])
    titles  = state.get("job_titles", [])
    level   = state.get("experience_level", "")
    is_ok   = len(skills) >= 3 and len(titles) >= 1 and level != ""

    if is_ok or state.get("iteration", 0) >= MAX_EXTRACTION_RETRIES:
        return "generate_search_queries"
    return "extract_resume_info"

# ---------------------------------------------------------------------------
# Node 2 – Dynamic Site Discovery
# ---------------------------------------------------------------------------

def discover_target_sites(state: AgentState) -> AgentState:
    llm = get_llm()
    log = state.get("status_log", [])
    log.append("🌐 Discovering niche job boards for your profile...")

    titles    = ", ".join(state.get("job_titles", [])[:3])
    industries = state.get("extracted_skills", [])
    level      = state.get("experience_level", "mid")
    location   = state.get("location", "Remote/Anywhere")

    system_prompt = (
        "You are a job market expert. Return ONLY a JSON array of domain strings "
        "(e.g. ['remoteok.com', 'wellfound.com']). No markdown, no explanation."
    )
    user_prompt = (
        f"Suggest 5-8 niche or mainstream job board domains (bare domain only, no paths) "
        f"best suited for a {level}-level professional targeting roles: {titles} in location: {location}. "
        f"Include at least one remote-first board and one regional/international board if relevant."
    )

    @with_retry
    def call_llm():
        return llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])

    discovered: List[str] = []
    try:
        resp = call_llm()
        arr  = re.search(r'\[.*?\]', resp.content, re.DOTALL)
        if arr:
            discovered = json.loads(arr.group())
    except Exception:
        pass

    existing = state.get("target_sites", [])
    merged = existing + [s for s in discovered if s not in existing]

    log.append(f"✅ Site list expanded to {len(merged)} boards: {', '.join(merged)}")
    return {**state, "target_sites": merged, "status_log": log}

# ---------------------------------------------------------------------------
# Node 3 – Generate Search Queries
# ---------------------------------------------------------------------------

def generate_search_queries(state: AgentState) -> AgentState:
    llm = get_llm()
    log = state.get("status_log", [])
    log.append("🧠 Generating targeted job search queries...")

    skills_str    = ", ".join(state["extracted_skills"][:10])
    titles_str    = ", ".join(state["job_titles"][:3])
    level         = state["experience_level"]
    sites         = state.get("target_sites", ["linkedin.com/jobs", "indeed.com", "glassdoor.com"])
    current_year  = datetime.now().year
    days          = state.get("date_filter_days")
    recency_label = DATE_FILTER_LABELS.get(days, "any time")
    location      = state.get("location", "").strip()
    location_str  = f" in {location}" if location else ""

    system_prompt = (
        "You are a job search strategist. Generate targeted search queries for job boards. "
        "Return ONLY valid JSON array of query strings. No explanation, no markdown. "
        "Mix remote and on-site queries. Mix title-focused and skill-focused queries. "
        "NEVER hardcode a year — always use the current year value provided in the prompt."
    )

    user_prompt = (
        f"Generate 8-10 Tavily web search queries to find job postings for:\n"
        f"- Skills: {skills_str}\n"
        f"- Job Titles: {titles_str}\n"
        f"- Seniority: {level}\n"
        f"- Location: {location if location else 'Anywhere/Remote'}\n"
        f"- Target sites: {', '.join(sites)}\n"
        f"- Current year: {current_year}\n"
        f"- Recency requirement: jobs {recency_label}\n\n"
        f'Example format: ["site:linkedin.com/jobs {titles_str} {level}{location_str} {current_year}", ...]\n'
        f'Lean into recency language (e.g. "hiring now", "new opening") to match the {recency_label} filter.'
    )

    @with_retry
    def call_llm():
        return llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])

    queries: List[str] = []
    try:
        response    = call_llm()
        array_match = re.search(r'\[.*?\]', response.content, re.DOTALL)
        queries     = json.loads(array_match.group()) if array_match else []
    except Exception:
        pass

    if not queries:
        first_title = state["job_titles"][0] if state["job_titles"] else "Software Engineer"
        first_skill = state["extracted_skills"][0] if state["extracted_skills"] else "Python"
        queries = [
            f'site:linkedin.com/jobs {first_title} {level}{location_str}',
            f'site:indeed.com {first_skill} developer jobs{location_str}',
            f'site:glassdoor.com {first_title} {level} job opening{location_str}',
            f'{first_title} remote job {current_year} {first_skill}{location_str}',
            f'{first_title} {level} position hiring {current_year}{location_str}',
        ]

    log.append(f"✅ Generated {len(queries)} search queries across {len(sites)} platforms")
    return {**state, "search_queries": queries, "status_log": log}

# ---------------------------------------------------------------------------
# Node 4 – Async Parallel Job Search
# ---------------------------------------------------------------------------

def search_jobs(state: AgentState) -> AgentState:
    log  = state.get("status_log", [])
    days = state.get("date_filter_days")

    raw_sites      = state.get("target_sites", ["linkedin.com", "indeed.com", "glassdoor.com"])
    forced_domains = list({s.split("/")[0] for s in raw_sites})
    recency_label  = DATE_FILTER_LABELS.get(days, "any time")

    log.append("⚡ Launching parallel job searches...")
    log.append(f"Domains: {', '.join(forced_domains)}")
    if days:
        log.append(f"Date filter active: {recency_label} (days={days})")

    async def fetch_query(query: str, tavily) -> List[dict]:
        loop = asyncio.get_running_loop()
        search_kwargs = dict(
            query=query,
            search_depth="advanced",
            max_results=5,
            include_answer=True,
            include_raw_content=False,
            include_domains=forced_domains,
        )
        if days is not None:
            search_kwargs["days"] = days

        @with_retry
        def _sync_search():
            return tavily.search(**search_kwargs)

        try:
            results = await loop.run_in_executor(None, _sync_search)
            return results.get("results", [])
        except Exception as e:
            log.append(f"Query failed: {str(e)[:60]}")
            return []

    async def gather_all(queries: List[str]) -> List[dict]:
        tavily = get_tavily()
        tasks  = [fetch_query(q, tavily) for q in queries]
        nested = await asyncio.gather(*tasks)
        return [item for sublist in nested for item in sublist]

    raw_flat = asyncio.run(gather_all(state["search_queries"]))

    seen_urls: set = set()
    all_results: List[dict] = []
    for r in raw_flat:
        url = r.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            matched_domain = next((d for d in forced_domains if d in url), "unknown")
            all_results.append({
                "title":          r.get("title", ""),
                "url":            url,
                "content":        r.get("content", ""),
                "score":          r.get("score", 0),
                "source_domain":  matched_domain,
                "recency_window": recency_label,
            })

    domain_counts: dict = {}
    for r in all_results:
        d = r.get("source_domain", "unknown")
        domain_counts[d] = domain_counts.get(d, 0) + 1
    summary = " | ".join(f"{d}: {c}" for d, c in sorted(domain_counts.items()))
    log.append(f"✅ Found {len(all_results)} unique listings — {summary}")

    return {**state, "raw_job_results": all_results, "status_log": log}

# ---------------------------------------------------------------------------
# Node 5 – Match & Score
# ---------------------------------------------------------------------------

def match_and_score(state: AgentState) -> AgentState:
    llm  = get_llm()
    log  = state.get("status_log", [])
    log.append("🎯 Matching your skills to job opportunities...")

    raw_results = state["raw_job_results"]
    if not raw_results:
        log.append("No raw results to match.")
        return {**state, "matched_jobs": [], "status_log": log}

    batch_size  = 10
    batches     = [raw_results[i:i+batch_size] for i in range(0, len(raw_results), batch_size)]
    all_matched: List[dict] = []

    system_prompt = """You are an expert job-skills matcher. Analyse job listings vs. candidate skills.
Return ONLY a valid JSON array. Each item must have:
{
  "title": "job title",
  "company": "company name or Unknown",
  "url": "original url",
  "match_score": 0-100,
  "matching_skills": ["skill1", "skill2"],
  "missing_skills": ["skill1", "skill2"],
  "job_type": "remote|hybrid|onsite|unknown",
  "salary_range": "if mentioned or Unknown",
  "location": "city/country or Remote",
  "why_good_fit": "2 sentence explanation",
  "seniority_match": "perfect|close|stretch",
  "source": "platform name"
}
Be strict: only include actual job postings, not news/blog articles."""

    for batch_idx, batch in enumerate(batches):
        log.append(f"Processing batch {batch_idx+1}/{len(batches)}...")
        results_text = json.dumps(
            [{"title": r["title"], "url": r["url"], "content": r["content"][:600]} for r in batch],
            indent=2,
        )

        user_prompt = (
            f"Candidate Profile:\n"
            f"- Technical Skills: {', '.join(state['extracted_skills'][:15])}\n"
            f"- Soft Skills: {', '.join(state['soft_skills'][:5])}\n"
            f"- Seniority: {state['experience_level']}\n"
            f"- Target Roles: {', '.join(state['job_titles'][:3])}\n"
            f"- Education: {state['education']}\n\n"
            f"Job Listings to Analyse:\n{results_text}\n\n"
            "Score each job 0-100 based on skill match. Only include actual job postings."
        )

        @with_retry
        def call_llm():
            return llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ])

        try:
            response    = call_llm()
            array_match = re.search(r'\[.*\]', response.content, re.DOTALL)
            if array_match:
                repaired = repair_json(array_match.group())
                all_matched.extend(json.loads(repaired))
        except Exception as e:
            log.append(f"Matching error: {str(e)[:60]}")

    all_matched.sort(key=lambda x: x.get("match_score", 0), reverse=True)
    top_matches = all_matched[:20]
    log.append(f"✅ Ranked {len(top_matches)} matched job opportunities")

    return {**state, "matched_jobs": top_matches, "status_log": log}

# ---------------------------------------------------------------------------
# Node 6 – Deep Research
# ---------------------------------------------------------------------------

def deep_research_top_jobs(state: AgentState) -> AgentState:
    tavily = get_tavily()
    llm    = get_llm()
    log    = state.get("status_log", [])
    top_jobs = state["matched_jobs"][:5]
    log.append(f"🔬 Deep-researching top {len(top_jobs)} matched positions...")

    enriched: List[dict] = []

    for job in top_jobs:
        company = job.get("company", "")
        if not company or company == "Unknown":
            enriched.append(job)
            continue

        @with_retry
        def fetch_company_info(co=company):
            return tavily.search(
                query=f"{co} company culture glassdoor reviews {datetime.now().year}",
                search_depth="advanced",
                max_results=3,
            )

        try:
            research     = fetch_company_info()
            company_info = " ".join([r.get("content", "")[:300] for r in research.get("results", [])])

            @with_retry
            def synthesise(info=company_info, co=company):
                return llm.invoke([
                    SystemMessage(content="Summarise company insights in 2 sentences. Be factual and concise."),
                    HumanMessage(content=f"Company: {co}\nInfo: {info[:800]}"),
                ])

            synthesis = synthesise()
            enriched.append({**job, "company_insights": synthesis.content, "research_depth": "deep"})
            log.append(f"Researched: {company[:30]}")
        except Exception:
            enriched.append({**job, "company_insights": "Research unavailable.", "research_depth": "basic"})

    for job in state["matched_jobs"][5:]:
        enriched.append({**job, "research_depth": "basic"})

    log.append("✅ Deep research phase complete")
    return {**state, "deep_research_results": enriched, "status_log": log}

# ---------------------------------------------------------------------------
# Node 7 – Compile Research Report
# ---------------------------------------------------------------------------

def compile_report(state: AgentState) -> AgentState:
    llm = get_llm()
    log = state.get("status_log", [])
    log.append("📊 Compiling your personalised deep research report...")

    jobs_summary  = json.dumps(state["deep_research_results"][:10], indent=2)
    skills        = ", ".join(state["extracted_skills"][:15])
    missing_skills = list({
        skill
        for job in state["matched_jobs"][:10]
        for skill in job.get("missing_skills", [])
    })[:10]

    system_prompt = (
        "You are a senior career strategist writing a deep research report. "
        "Write in markdown with clear sections. Be specific, actionable, and insightful. "
        "Include data-backed observations from the job results provided."
    )

    report_prompt = f"""Generate a comprehensive Career Research Report with these sections:

## Executive Summary
Brief overview of the candidate's profile and market position.

## Top Job Matches
For each of the top 5-8 jobs (from the data), include:
- Company + Role + Match Score
- Key matching skills
- Why it's a strong fit
- Direct application link

## Market Intelligence
- Demand patterns for the candidate's skill set
- Salary range insights
- Remote vs onsite trends
- Hot companies hiring in this space

## Skill Gap Analysis
- Current skills inventory strength
- Missing skills that appear frequently in job listings: {', '.join(missing_skills)}
- Learning roadmap recommendations

## Action Plan
- Top 3 immediate application targets
- Skills to develop in next 30/60/90 days
- Networking and outreach strategy

---
CANDIDATE DATA:
Skills: {skills}
Level: {state['experience_level']}
Target Roles: {', '.join(state['job_titles'])}

JOB MATCHES DATA:
{jobs_summary}"""

    @with_retry
    def call_llm():
        return llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=report_prompt),
        ])

    response = call_llm()
    log.append("✅ Report compiled successfully!")

    return {
        **state,
        "research_report": response.content,
        "status_log":      log,
        "messages": [AIMessage(content="Deep research report ready. " + response.content[:200] + "...")],
    }

# ---------------------------------------------------------------------------
# Node 8 – Cover Letter Generation
# ---------------------------------------------------------------------------

def generate_cover_letters(state: AgentState) -> AgentState:
    llm  = get_llm()
    log  = state.get("status_log", [])
    log.append("✉️  Generating tailored cover letters for top 3 jobs...")

    top_jobs      = state["deep_research_results"][:3]
    cover_letters: List[dict] = []

    system_prompt = (
        "You are an expert career coach and professional writer. "
        "Write compelling, tailored cover letters that highlight relevant skills and enthusiasm. "
        "Keep letters to 3 paragraphs. Avoid generic platitudes. Be specific and confident."
    )

    for job in top_jobs:
        company = job.get("company", "the company")
        title   = job.get("title",   "this role")
        fit     = job.get("why_good_fit", "")
        skills  = ", ".join(job.get("matching_skills", state["extracted_skills"][:6]))

        user_prompt = (
            f"Write a tailored cover letter for:\n"
            f"- Role: {title} at {company}\n"
            f"- My matching skills: {skills}\n"
            f"- Why I'm a good fit: {fit}\n"
            f"- My experience level: {state['experience_level']}\n"
            f"- My background: {state['education']}\n\n"
            "Begin with a strong opening hook. Second paragraph highlights relevant achievements. "
            "Third paragraph expresses enthusiasm and calls to action."
        )

        @with_retry
        def call_llm(p=user_prompt):
            return llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=p),
            ])

        try:
            response = call_llm()
            cover_letters.append({
                "company": company,
                "title":   title,
                "url":     job.get("url", ""),
                "letter":  response.content,
            })
            log.append(f"✅ Cover letter drafted for: {company[:30]} – {title[:30]}")
        except Exception as e:
            log.append(f"Cover letter failed for {company}: {str(e)[:50]}")

    return {**state, "cover_letters": cover_letters, "status_log": log}

# ---------------------------------------------------------------------------
# Node 9 – PDF Report Generation
# ---------------------------------------------------------------------------

def generate_pdf_report(state: AgentState) -> AgentState:
    log = state.get("status_log", [])
    log.append("📄 Generating PDF report...")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"/tmp/career_report_{timestamp}.pdf"

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    def heading(text: str, size: int = 14):
        pdf.set_font("Helvetica", "B", size)
        pdf.set_text_color(30, 60, 120)
        pdf.multi_cell(0, 8, text)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(2)

    def body(text: str, size: int = 10):
        pdf.set_font("Helvetica", "", size)
        clean = re.sub(r'[#*`]', '', text)
        pdf.multi_cell(0, 6, clean)
        pdf.ln(2)

    def divider():
        pdf.set_draw_color(180, 180, 200)
        pdf.line(pdf.get_x(), pdf.get_y(), pdf.get_x() + 190, pdf.get_y())
        pdf.ln(4)

    heading("Career Research Report", size=20)
    body(f"Generated: {datetime.now().strftime('%B %d, %Y  %H:%M')}")
    body(f"Candidate Level : {state.get('experience_level', '').title()}")
    body(f"Target Roles    : {', '.join(state.get('job_titles', []))}")
    body(f"Target Location : {state.get('location', 'Anywhere/Remote')}")
    body(f"Top Skills      : {', '.join(state.get('extracted_skills', [])[:10])}")
    divider()

    heading("Deep Research Report", size=14)
    body(state.get("research_report", "No report generated."))
    divider()

    if state.get("cover_letters"):
        heading("Tailored Cover Letters", size=14)
        for cl in state["cover_letters"]:
            pdf.add_page()
            heading(f"{cl['title']} @ {cl['company']}", size=12)
            if cl.get("url"):
                body(f"Application URL: {cl['url']}")
            body(cl["letter"])
            divider()

    pdf.output(output_path)
    log.append(f"✅ PDF saved to: {output_path}")

    return {**state, "pdf_report_path": output_path, "status_log": log}

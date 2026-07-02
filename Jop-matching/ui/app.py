"""
Resume-to-Job Deep Research Agent — Streamlit UI
Aesthetic: Dark cosmic / mission control — bold, purposeful, technical
"""

import streamlit as st
import os
import time
import json
import threading
from pypdf import PdfReader
import io
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.state import DATE_FILTER_OPTIONS, DATE_FILTER_LABELS

# ── Page Config ─────────────────────────────────────────────────
st.set_page_config(
    page_title="JobAgent AI",
    page_icon="🛸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Outfit:wght@300;400;500;600;700;800&display=swap');

/* Reset & base */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #050A14 !important;
    color: #E8EDF5 !important;
    font-family: 'Outfit', sans-serif !important;
}

[data-testid="stSidebar"] {
    background: #080E1C !important;
    border-right: 1px solid #1a2540 !important;
}

/* Header */
.hero-header {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
    position: relative;
}
.hero-title {
    font-family: 'Space Mono', monospace !important;
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(135deg, #00D4FF 0%, #7B61FF 50%, #FF6B9D 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -1px;
    line-height: 1.1;
    margin-bottom: 0.5rem;
}
.hero-subtitle {
    font-size: 1rem;
    color: #5A7099;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    font-weight: 400;
}

/* Cards */
.card {
    background: linear-gradient(145deg, #0D1628, #0A1220);
    border: 1px solid #1E2D4A;
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.25rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.3s ease, transform 0.2s ease;
}
.card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, #00D4FF33, transparent);
}
.card:hover {
    border-color: #00D4FF44;
    transform: translateY(-1px);
}

/* Job cards */
.job-card {
    background: linear-gradient(145deg, #0D1628, #0A1220);
    border: 1px solid #1E2D4A;
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
    transition: all 0.25s ease;
}
.job-card:hover {
    border-color: #7B61FF55;
    background: linear-gradient(145deg, #101c35, #0d1628);
}
.job-card.top-match {
    border-color: #00D4FF44;
    background: linear-gradient(145deg, #091524, #06101e);
}
.job-title { font-size: 1.1rem; font-weight: 600; color: #E8EDF5; margin-bottom: 0.25rem; }
.job-company { font-size: 0.9rem; color: #7B93BB; font-family: 'Space Mono', monospace; }
.job-meta { display: flex; gap: 0.75rem; margin-top: 0.75rem; flex-wrap: wrap; }

/* Badges */
.badge {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 99px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.badge-skill { background: #7B61FF22; color: #9B84FF; border: 1px solid #7B61FF33; }
.badge-remote { background: #00D4FF22; color: #00D4FF; border: 1px solid #00D4FF33; }
.badge-missing { background: #FF6B9D22; color: #FF6B9D; border: 1px solid #FF6B9D33; }
.badge-level { background: #22D9A022; color: #22D9A0; border: 1px solid #22D9A033; }

/* Score ring */
.score-ring {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
}
.score-high { color: #22D9A0; }
.score-mid  { color: #FFD166; }
.score-low  { color: #FF6B9D; }

/* Status log */
.log-container {
    background: #060C18;
    border: 1px solid #1a2540;
    border-radius: 12px;
    padding: 1rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    color: #4A90D9;
    max-height: 280px;
    overflow-y: auto;
    line-height: 2;
}
.log-line { margin: 0; padding: 0; }

/* Section labels */
.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    color: #3A5480;
    text-transform: uppercase;
    letter-spacing: 0.2em;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, #1E2D4A, transparent);
}

/* Stat cards */
.stat-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
.stat-card {
    flex: 1;
    min-width: 120px;
    background: #0D1628;
    border: 1px solid #1E2D4A;
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.stat-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: #00D4FF;
    line-height: 1;
    margin-bottom: 0.25rem;
}
.stat-label { font-size: 0.72rem; color: #5A7099; text-transform: uppercase; letter-spacing: 0.1em; }

/* Streamlit overrides */
.stButton > button {
    background: linear-gradient(135deg, #00D4FF22, #7B61FF22) !important;
    color: #E8EDF5 !important;
    border: 1px solid #7B61FF55 !important;
    border-radius: 10px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.85rem !important;
    padding: 0.6rem 1.2rem !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}
.stButton > button:hover {
    border-color: #7B61FF !important;
    background: linear-gradient(135deg, #00D4FF33, #7B61FF33) !important;
    transform: translateY(-1px) !important;
}
.launch-btn > button {
    background: linear-gradient(135deg, #00D4FF, #7B61FF) !important;
    color: #050A14 !important;
    border: none !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.8rem !important;
    letter-spacing: 0.05em !important;
}

.stTextArea textarea {
    background: #0D1628 !important;
    border: 1px solid #1E2D4A !important;
    border-radius: 12px !important;
    color: #E8EDF5 !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.9rem !important;
}
.stFileUploader {
    background: #0D1628 !important;
    border: 1px dashed #1E2D4A !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}
.stProgress > div > div > div { background: linear-gradient(90deg, #00D4FF, #7B61FF) !important; }
.stMultiSelect [data-baseweb="tag"] { background: #7B61FF33 !important; }
div[data-testid="stMarkdownContainer"] h1,
div[data-testid="stMarkdownContainer"] h2,
div[data-testid="stMarkdownContainer"] h3 { color: #E8EDF5 !important; }
.stAlert { border-radius: 12px !important; }

/* Sidebar widgets */
[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] .stTextInput textarea {
    background: #0D1628 !important;
    border-color: #1E2D4A !important;
    color: #E8EDF5 !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown p {
    color: #7B93BB !important;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────────

def extract_text_from_pdf(file_bytes) -> str:
    
    pdf_stream = io.BytesIO(file_bytes)
    reader = PdfReader(pdf_stream)
    
    text = ""
    for page in reader.pages:
        
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
            
    return text.strip()


def score_color(score: int) -> str:
    if score >= 75: return "score-high"
    if score >= 50: return "score-mid"
    return "score-low"


def render_skill_chart(skills: list, matched_jobs: list):
    if not skills or not matched_jobs: return

    skill_demand = {}
    for skill in skills:
        count = sum(1 for j in matched_jobs if skill.lower() in str(j.get("matching_skills", [])).lower())
        if count > 0:
            skill_demand[skill] = count

    if not skill_demand: return

    df = pd.DataFrame(list(skill_demand.items()), columns=["Skill", "Demand"]).sort_values("Demand", ascending=True)
    fig = px.bar(df, x="Demand", y="Skill", orientation="h",
                 color="Demand",
                 color_continuous_scale=["#7B61FF", "#00D4FF"],
                 template="plotly_dark")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Space Mono", color="#7B93BB", size=11),
        showlegend=False,
        coloraxis_showscale=False,
        margin=dict(l=10, r=10, t=10, b=10),
        height=max(200, len(skill_demand) * 32),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False),
    )
    fig.update_traces(marker_line_width=0)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_match_score_gauge(score: int):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={"x": [0, 1], "y": [0, 1]},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#3A5480",
                     "tickfont": {"color": "#5A7099", "size": 9}},
            "bar": {"color": "#00D4FF" if score >= 75 else "#FFD166" if score >= 50 else "#FF6B9D"},
            "bgcolor": "#0D1628",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 50], "color": "#FF6B9D11"},
                {"range": [50, 75], "color": "#FFD16611"},
                {"range": [75, 100], "color": "#00D4FF11"},
            ],
            "threshold": {"line": {"color": "#E8EDF5", "width": 1}, "thickness": 0.75, "value": score},
        },
        number={"font": {"family": "Space Mono", "size": 32, "color": "#E8EDF5"}, "suffix": "%"}
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=5, r=5, t=5, b=5),
        height=140,
    )
    return fig


# ── Main App ─────────────────────────────────────────────────────

def main():
    # ── Sidebar ──────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="padding: 1rem 0 0.5rem;">
            <p style="font-family: 'Space Mono', monospace; font-size: 0.65rem; 
                      color: #3A5480; letter-spacing: 0.25em; text-transform: uppercase;">
                MISSION CONTROL
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**🔑 API Keys**")
        groq_key = st.text_input("Groq API Key", type="password",
                                  value=os.getenv("GROQ_API_KEY", ""),
                                  placeholder="gsk_...")
        tavily_key = st.text_input("Tavily API Key", type="password",
                                    value=os.getenv("TAVILY_API_KEY", ""),
                                    placeholder="tvly-...")

        if groq_key: os.environ["GROQ_API_KEY"] = groq_key
        if tavily_key: os.environ["TAVILY_API_KEY"] = tavily_key

        st.markdown("---")
        st.markdown("**🌍 Target Location**")
        target_location = st.text_input("City, Country, or Remote", value="", placeholder="e.g., Cairo, Dubai, Remote")

        st.markdown("---")
        st.markdown("**🎯 Target Job Sites**")
        default_sites = [
            "linkedin.com/jobs",
            "indeed.com",
            "glassdoor.com",
            "wuzzuf.net",
            "bayt.com",
            "forasna.com",
            "naukrigulf.com",
            "weworkremotely.com",
            "remoteok.com",
            "wellfound.com",
        ]
        selected_sites = st.multiselect(
            "Select platforms to search",
            options=default_sites + ["levels.fyi", "ycombinator.com/jobs", "stackoverflow.com/jobs"],
            default=default_sites[:4],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("**📅 Posting Recency**")

        date_option = st.radio(
            "Posting recency",
            options=list(DATE_FILTER_OPTIONS.keys()),
            index=1,          # default → Last 7 days
            label_visibility="collapsed",
        )
        selected_days = DATE_FILTER_OPTIONS[date_option]
        recency_desc = DATE_FILTER_LABELS.get(selected_days, "any time")

        # Live feedback line
        tip = {
            1:    "⚡ Only the freshest postings — may return fewer results",
            7:    "✅ Balanced: fresh enough, broad enough",
            30:   "📦 Good volume — catches slower-moving roles",
            90:   "🔭 Maximum coverage — includes older postings",
            None: "♾ No date restriction applied",
        }
        st.markdown(
            f'<div style="font-family:Space Mono,monospace;font-size:0.68rem;'
            f'color:#00D4FF;margin:0.3rem 0 0.1rem;">→ Jobs {recency_desc}</div>'
            f'<div style="font-size:0.68rem;color:#3A5480;">{tip.get(selected_days,"")}</div>',
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown("**⚙️ Model**")
        st.markdown("""
        <div class="card" style="padding: 1rem;">
            <div style="font-family: 'Space Mono', monospace; font-size: 0.78rem; color: #00D4FF;">
                🦙 LLaMA 3.3 70B
            </div>
            <div style="font-size: 0.72rem; color: #5A7099; margin-top: 0.3rem;">
                via Groq · Best open-source for agentic reasoning
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        <div style="font-size: 0.7rem; color: #3A5480; font-family: 'Space Mono', monospace; line-height: 1.8;">
            PIPELINE<br>
            ① Resume Parser<br>
            <span style="color:#FFD166;">↺ Quality Check Loop</span><br>
            ② Site Discovery<br>
            ③ Query Generator<br>
            ④ Parallel Web Search<br>
            ⑤ Skill Matcher<br>
            ⑥ Deep Researcher<br>
            ⑦ Report Compiler<br>
            ⑧ Cover Letters<br>
            ⑨ PDF Export
        </div>
        """, unsafe_allow_html=True)

    # ── Hero ─────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">JOBAGENT AI</div>
        <div class="hero-subtitle">Deep Research · Skill Extraction · Job Matching · Cover Letters · Market Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Input Section ─────────────────────────────────────────────
    col_input, col_gap, col_preview = st.columns([1, 0.05, 1])

    with col_input:
        st.markdown('<div class="section-label">📄 RESUME INPUT</div>', unsafe_allow_html=True)

        input_method = st.radio("Input method", ["📁 Upload PDF", "✏️ Paste Text"],
                                 horizontal=True, label_visibility="collapsed")

        resume_text = ""

        if input_method == "📁 Upload PDF":
            uploaded = st.file_uploader("Upload resume", type=["pdf"],
                                         label_visibility="collapsed")
            if uploaded:
                with st.spinner("Extracting text from PDF..."):
                    resume_text = extract_text_from_pdf(uploaded.read())
                st.success(f"✅ Extracted {len(resume_text.split())} words from PDF")
        else:
            resume_text = st.text_area(
                "Paste your resume",
                height=320,
                placeholder="Paste your full resume text here...\n\nName, contact info, skills, experience, education...",
                label_visibility="collapsed",
            )

    with col_preview:
        st.markdown('<div class="section-label">🔭 PREVIEW</div>', unsafe_allow_html=True)

        if resume_text:
            st.markdown(f"""
            <div class="card">
                <div style="font-family: 'Space Mono', monospace; font-size: 0.72rem; color: #3A5480; margin-bottom: 0.75rem;">
                    RESUME LOADED
                </div>
                <div style="font-size: 0.85rem; color: #7B93BB; line-height: 1.7; max-height: 260px; overflow: hidden;">
                    {resume_text[:600].replace(chr(10), '<br>')}{'...' if len(resume_text) > 600 else ''}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="card" style="min-height: 200px; display: flex; align-items: center; justify-content: center;">
                <div style="text-align: center; color: #3A5480;">
                    <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🛸</div>
                    <div style="font-family: 'Space Mono', monospace; font-size: 0.75rem;">
                        Awaiting resume upload
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Launch Button ─────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    col_btn, col_mid, col_r = st.columns([1, 2, 1])

    with col_mid:
        apis_ready = bool(os.getenv("GROQ_API_KEY")) and bool(os.getenv("TAVILY_API_KEY"))
        resume_ready = bool(resume_text and resume_text.strip())

        if not apis_ready:
            st.warning("⚡ Add your Groq + Tavily API keys in the sidebar to launch.")
        elif not resume_ready:
            st.info("📄 Upload or paste your resume to begin.")

        with st.container():
            st.markdown('<div class="launch-btn">', unsafe_allow_html=True)
            launch = st.button(
                "🚀 LAUNCH DEEP RESEARCH MISSION",
                disabled=not (apis_ready and resume_ready),
                use_container_width=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)

    # ── Agent Execution ───────────────────────────────────────────
    if launch and resume_ready and apis_ready:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">⚡ MISSION IN PROGRESS</div>', unsafe_allow_html=True)

        log_placeholder = st.empty()
        progress_bar = st.progress(0)
        status_text = st.empty()

        steps = [
            "Parsing resume & extracting skills...",
            "Running quality check on extraction...",
            "Discovering niche job boards...",
            "Generating targeted search queries...",
            "Searching job boards in parallel...",
            "Matching skills to opportunities...",
            "Deep-researching top matches...",
            "Compiling research report...",
            "Drafting tailored cover letters...",
            "Generating PDF report...",
        ]

        # Simulate early progress
        for i, step in enumerate(steps):
            status_text.markdown(f"<p style='color:#5A7099; font-size:0.85rem;'>{step}</p>",
                                  unsafe_allow_html=True)
            progress_bar.progress((i + 1) / (len(steps) + 1))
            time.sleep(0.3)

        status_text.markdown(
            "<p style='color:#00D4FF; font-size:0.85rem;'>🔄 Agent running...</p>",
            unsafe_allow_html=True
        )

        try:
            from core.agent import run_agent
            result = run_agent(
                resume_text,
                target_sites=selected_sites,
                date_filter_days=selected_days,
                location=target_location,
            )

            progress_bar.progress(1.0)
            status_text.markdown(
                "<p style='color:#22D9A0; font-size:0.9rem;'>✅ Mission complete!</p>",
                unsafe_allow_html=True
            )

            # Show log
            log_html = "".join([f'<p class="log-line">{line}</p>' for line in result.get("status_log", [])])
            log_placeholder.markdown(
                f'<div class="log-container">{log_html}</div>',
                unsafe_allow_html=True
            )

            # ── Results ──────────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)

            # Stats row
            matched = result.get("matched_jobs", [])
            skills = result.get("extracted_skills", [])
            cover_letters = result.get("cover_letters", [])
            avg_score = int(sum(j.get("match_score", 0) for j in matched) / len(matched)) if matched else 0
            top_score = max((j.get("match_score", 0) for j in matched), default=0)
            active_recency = DATE_FILTER_LABELS.get(selected_days, "any time")
            discovered_sites = result.get("target_sites", selected_sites)

            st.markdown(f"""
            <div class="stat-row">
                <div class="stat-card">
                    <div class="stat-value">{len(skills)}</div>
                    <div class="stat-label">Skills Found</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{len(result.get('raw_job_results', []))}</div>
                    <div class="stat-label">Jobs Crawled</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{len(matched)}</div>
                    <div class="stat-label">Matched Jobs</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{avg_score}%</div>
                    <div class="stat-label">Avg Match</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{top_score}%</div>
                    <div class="stat-label">Best Match</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{len(cover_letters)}</div>
                    <div class="stat-label">Cover Letters</div>
                </div>
            </div>
            <div style="margin-bottom:1.25rem; display:flex; flex-wrap:wrap; gap:0.5rem;">
                <span class="badge badge-remote" style="font-size:0.75rem;padding:0.3rem 0.9rem;">
                    📅 Filtered: {active_recency}
                </span>
                <span class="badge badge-skill" style="font-size:0.75rem;padding:0.3rem 0.9rem;">
                    🔒 Sites: {', '.join(s.split('/')[0] for s in selected_sites[:3])}{'...' if len(selected_sites)>3 else ''}
                </span>
                <span class="badge badge-level" style="font-size:0.75rem;padding:0.3rem 0.9rem;">
                    🌐 Boards discovered: {len(discovered_sites)}
                </span>
            </div>
            """, unsafe_allow_html=True)

            # Tabs
            tab_jobs, tab_skills, tab_covers, tab_report = st.tabs([
                "🎯 Job Matches", "🔧 Skills Analysis", "✉️ Cover Letters", "📊 Full Report"
            ])

            with tab_jobs:
                st.markdown('<div class="section-label">TOP JOB MATCHES</div>', unsafe_allow_html=True)

                jobs_to_show = result.get("deep_research_results", matched)[:15]
                for i, job in enumerate(jobs_to_show):
                    score = job.get("match_score", 0)
                    score_cls = score_color(score)
                    is_top = i < 3

                    matching = ", ".join(job.get("matching_skills", [])[:5])
                    missing = job.get("missing_skills", [])[:3]
                    jtype = job.get("job_type", "unknown")
                    company_info = job.get("company_insights", "")

                    missing_badges = "".join([
                        f'<span class="badge badge-missing">{s}</span>' for s in missing
                    ])
                    remote_badge = f'<span class="badge badge-remote">{jtype}</span>' if jtype != "unknown" else ""
                    seniority_badge = f'<span class="badge badge-level">{job.get("seniority_match","")}</span>'

                    rank_icon = ["🥇", "🥈", "🥉"][i] if i < 3 else f"#{i+1}"

                    url = job.get("url", "#")
                    link = f'<a href="{url}" target="_blank" style="color:#00D4FF; font-size:0.78rem; font-family: Space Mono, monospace; text-decoration:none;">↗ View Posting</a>'

                    html_parts = [
                        f"<div class='job-card {'top-match' if is_top else ''}'>",
                        f"<div style='display:flex; justify-content:space-between; align-items:flex-start;'>",
                        f"<div style='flex:1;'>",
                        f"<div class='job-title'>{rank_icon} {job.get('title', 'Unknown Role')}</div>",
                        f"<div class='job-company'>🏢 {job.get('company','Unknown')} · 📍 {job.get('location','Unknown')}</div>",
                        f"<div class='job-meta'>",
                        f"{remote_badge} {seniority_badge}",
                        f"<span style='font-size:0.78rem; color:#3A5480;'>💰 {job.get('salary_range','Unknown')}</span>",
                        f"</div>",
                        f"</div>",
                        f"<div style='text-align:center; min-width:70px;'>",
                        f"<div class='score-ring {score_cls}'>{score}</div>",
                        f"<div style='font-size:0.65rem; color:#3A5480; font-family: Space Mono; margin-top:2px;'>MATCH%</div>",
                        f"</div>",
                        f"</div>"
                    ]
                    
                    if job.get('why_good_fit'):
                        html_parts.append(f"<div style='margin-top:0.75rem; font-size:0.82rem; color:#7B93BB;'>💡 {job.get('why_good_fit','')}</div>")
                    if company_info and len(company_info) > 50:
                        html_parts.append(f"<div style='margin-top:0.5rem; font-size:0.78rem; color:#5A7099; font-style:italic;'>🏢 {company_info[:200]}...</div>")
                        
                    html_parts.append(f"<div style='margin-top:0.75rem;'>")
                    html_parts.append(f"<span style='font-size:0.72rem; color:#3A5480; margin-right:0.5rem;'>SKILLS MATCH:</span>")
                    html_parts.append(matching or "<span style='color:#3A5480; font-size:0.78rem;'>—</span>")
                    html_parts.append(f"</div>")
                    
                    if missing:
                        html_parts.append(f"<div style='margin-top:0.5rem;'><span style='font-size:0.72rem; color:#3A5480; margin-right:0.5rem;'>GAP:</span>{missing_badges}</div>")
                        
                    html_parts.append(f"<div style='margin-top:0.75rem;'>{link}</div>")
                    html_parts.append(f"</div>")
                    
                    st.markdown("".join(html_parts), unsafe_allow_html=True)

            with tab_skills:
                st.markdown('<div class="section-label">SKILL DEMAND ANALYSIS</div>', unsafe_allow_html=True)

                col_s1, col_s2 = st.columns([1, 1])
                with col_s1:
                    st.markdown("**Your Technical Skills**")
                    skills_html = " ".join([
                        f'<span class="badge badge-skill" style="margin:3px; display:inline-block;">{s}</span>'
                        for s in skills
                    ])
                    st.markdown(f'<div style="line-height:2.5;">{skills_html}</div>', unsafe_allow_html=True)

                    if result.get("soft_skills"):
                        st.markdown("<br>**Soft Skills**")
                        soft_html = " ".join([
                            f'<span class="badge badge-level" style="margin:3px; display:inline-block;">{s}</span>'
                            for s in result["soft_skills"][:10]
                        ])
                        st.markdown(f'<div style="line-height:2.5;">{soft_html}</div>', unsafe_allow_html=True)

                with col_s2:
                    st.markdown("**Skill Demand in Matched Jobs**")
                    render_skill_chart(skills, matched)

                # Missing skills
                all_missing = list(set([
                    s for job in matched[:15]
                    for s in job.get("missing_skills", [])
                ]))[:15]
                if all_missing:
                    st.markdown("<br>**Frequently Requested Skills You Could Add**")
                    missing_html = " ".join([
                        f'<span class="badge badge-missing" style="margin:3px; display:inline-block;">{s}</span>'
                        for s in all_missing
                    ])
                    st.markdown(f'<div style="line-height:2.5;">{missing_html}</div>', unsafe_allow_html=True)

            # ── Cover Letters Tab ─────────────────────────────────
            with tab_covers:
                st.markdown('<div class="section-label">TAILORED COVER LETTERS</div>', unsafe_allow_html=True)

                if not cover_letters:
                    st.markdown("""
                    <div class="card" style="text-align:center; padding:2rem; color:#3A5480;">
                        <div style="font-size:2rem; margin-bottom:0.5rem;">✉️</div>
                        <div style="font-family:'Space Mono',monospace; font-size:0.78rem;">
                            No cover letters generated in this run.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(
                        f'<p style="font-size:0.82rem; color:#5A7099; margin-bottom:1rem;">'
                        f'{len(cover_letters)} personalised cover letters generated for your top matches.</p>',
                        unsafe_allow_html=True,
                    )
                    for cl in cover_letters:
                        company = cl.get("company", "Unknown")
                        title   = cl.get("title", "Unknown Role")
                        url     = cl.get("url", "")
                        letter  = cl.get("letter", "")

                        with st.expander(f"✉️ {title} @ {company}", expanded=False):
                            if url:
                                st.markdown(
                                    f'<a href="{url}" target="_blank" style="color:#00D4FF; '
                                    f'font-family:Space Mono,monospace; font-size:0.78rem; '
                                    f'text-decoration:none;">↗ View Job Posting</a>',
                                    unsafe_allow_html=True,
                                )
                                st.markdown("<br>", unsafe_allow_html=True)
                            st.markdown(
                                f'<div style="font-size:0.88rem; color:#C8D8F0; line-height:1.9; '
                                f'white-space:pre-wrap;">{letter}</div>',
                                unsafe_allow_html=True,
                            )
                            st.download_button(
                                f"⬇️ Download — {company[:20]}",
                                data=letter,
                                file_name=f"cover_letter_{company.replace(' ','_')[:20]}.txt",
                                mime="text/plain",
                                key=f"cl_{company}_{title}",
                            )

            with tab_report:
                st.markdown('<div class="section-label">DEEP RESEARCH REPORT</div>', unsafe_allow_html=True)
                report = result.get("research_report", "")
                if report:
                    st.markdown(report)
                    col_dl1, col_dl2 = st.columns(2)
                    with col_dl1:
                        st.download_button(
                            "⬇️ Download Report (.md)",
                            data=report,
                            file_name="job_research_report.md",
                            mime="text/markdown",
                        )
                    with col_dl2:
                        pdf_path = result.get("pdf_report_path", "")
                        if pdf_path:
                            try:
                                with open(pdf_path, "rb") as f:
                                    pdf_bytes = f.read()
                                st.download_button(
                                    "📄 Download Full PDF Report",
                                    data=pdf_bytes,
                                    file_name="job_research_report.pdf",
                                    mime="application/pdf",
                                )
                            except Exception:
                                st.caption("PDF file not available for download.")

        except ImportError:
            st.error("⚠️ Could not import agent module. Make sure agent.py is in the same directory.")
        except Exception as e:
            progress_bar.progress(1.0)
            st.error(f"❌ Agent error: {str(e)}")
            st.info("Check your API keys and try again. Common issues: invalid Groq/Tavily keys, rate limits.")
            with st.expander("Debug Info"):
                st.exception(e)


if __name__ == "__main__":
    main()
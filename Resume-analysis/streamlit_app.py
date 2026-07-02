"""
Resume Analysis — Streamlit Frontend

Premium dark glassmorphism UI with animated score gauges,
PDF upload, text paste, and comprehensive analysis dashboard.
"""

import streamlit as st
import time
import json
import math

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════
# CUSTOM CSS — Premium Dark Glassmorphism Theme
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

:root {
    --bg-primary: #0a0e1a;
    --bg-secondary: #111827;
    --bg-card: rgba(17, 24, 39, 0.7);
    --bg-glass: rgba(255, 255, 255, 0.03);
    --accent-blue: #3b82f6;
    --accent-purple: #8b5cf6;
    --accent-cyan: #06b6d4;
    --accent-emerald: #10b981;
    --accent-amber: #f59e0b;
    --accent-rose: #f43f5e;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --border: rgba(148, 163, 184, 0.1);
    --glow-blue: rgba(59, 130, 246, 0.15);
    --glow-purple: rgba(139, 92, 246, 0.15);
}

.stApp {
    background: linear-gradient(135deg, var(--bg-primary) 0%, #0f172a 50%, #1a0a2e 100%);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0e1a 0%, #1a0a2e 100%) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .stMarkdown h1 {
    background: linear-gradient(135deg, #60a5fa, #a78bfa, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 1.4rem;
    font-weight: 800;
}

/* Hero Banner */
.hero-banner {
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.08), rgba(139, 92, 246, 0.08), rgba(244, 63, 94, 0.05));
    border: 1px solid var(--border);
    border-radius: 24px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(139, 92, 246, 0.1) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-banner h1 {
    background: linear-gradient(135deg, #60a5fa, #a78bfa, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.2rem;
    font-weight: 900;
    margin: 0 0 0.5rem 0;
    letter-spacing: -0.02em;
}
.hero-banner p {
    color: var(--text-secondary);
    font-size: 1.05rem;
    margin: 0;
    line-height: 1.6;
}

/* Score Gauge */
.gauge-container {
    background: var(--bg-glass);
    backdrop-filter: blur(20px);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 1.8rem 1.2rem;
    text-align: center;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}
.gauge-container:hover {
    transform: translateY(-6px);
    border-color: rgba(59, 130, 246, 0.3);
    box-shadow: 0 12px 40px rgba(59, 130, 246, 0.1);
}
.gauge-svg {
    width: 120px;
    height: 120px;
    margin: 0 auto 0.8rem;
    display: block;
}
.gauge-score {
    font-size: 2rem;
    font-weight: 900;
    letter-spacing: -0.02em;
}
.gauge-label {
    color: var(--text-secondary);
    font-size: 0.8rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.3rem;
}

/* Overall Score Badge */
.overall-badge {
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(139, 92, 246, 0.15));
    border: 2px solid rgba(139, 92, 246, 0.3);
    border-radius: 24px;
    padding: 2.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.overall-badge::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: conic-gradient(from 0deg, transparent, rgba(139, 92, 246, 0.05), transparent);
    animation: spin 8s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
.overall-value {
    font-size: 4rem;
    font-weight: 900;
    position: relative;
    z-index: 1;
}
.overall-label {
    color: var(--text-secondary);
    font-size: 1rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    position: relative;
    z-index: 1;
}

/* Analysis Cards */
.analysis-card {
    background: var(--bg-glass);
    backdrop-filter: blur(20px);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: all 0.3s ease;
}
.analysis-card:hover {
    border-color: rgba(59, 130, 246, 0.2);
}
.card-title {
    font-size: 1.1rem;
    font-weight: 700;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.card-item {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.5rem;
    color: var(--text-primary);
    font-size: 0.9rem;
    line-height: 1.5;
    transition: all 0.2s ease;
}
.card-item:hover {
    background: rgba(255, 255, 255, 0.04);
    border-color: rgba(255, 255, 255, 0.1);
}

/* Strength items */
.strength-item {
    border-left: 3px solid var(--accent-emerald);
}
.strength-item::before { content: '✓ '; color: var(--accent-emerald); font-weight: 700; }

/* Weakness items */
.weakness-item {
    border-left: 3px solid var(--accent-amber);
}
.weakness-item::before { content: '⚠ '; }

/* Suggestion items */
.suggestion-item {
    border-left: 3px solid var(--accent-blue);
}
.suggestion-item::before { content: '💡 '; }

/* Rewrite items */
.rewrite-item {
    border-left: 3px solid var(--accent-purple);
}
.rewrite-item::before { content: '✏️ '; }

/* Upload Zone */
.upload-zone {
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.05), rgba(139, 92, 246, 0.05));
    border: 2px dashed rgba(139, 92, 246, 0.3);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    transition: all 0.3s ease;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.7rem 2.5rem !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.02em !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(59, 130, 246, 0.2) !important;
}
.stButton > button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 30px rgba(59, 130, 246, 0.35) !important;
}

/* Inputs */
.stTextArea > div > div > textarea,
.stTextInput > div > div > input {
    background: rgba(17, 24, 39, 0.8) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 12px !important;
    font-family: 'Inter', sans-serif !important;
}
.stTextArea > div > div > textarea:focus,
.stTextInput > div > div > input:focus {
    border-color: rgba(139, 92, 246, 0.5) !important;
    box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.1) !important;
}

/* Section Headers */
.section-header {
    font-size: 1.3rem;
    font-weight: 800;
    color: var(--text-primary);
    margin: 2rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid var(--border);
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* Status Pill */
.status-pill {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 50px;
    font-weight: 700;
    font-size: 0.75rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.status-success { background: rgba(16, 185, 129, 0.15); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.3); }
.status-warning { background: rgba(245, 158, 11, 0.15); color: #f59e0b; border: 1px solid rgba(245, 158, 11, 0.3); }
.status-error { background: rgba(244, 63, 94, 0.15); color: #f43f5e; border: 1px solid rgba(244, 63, 94, 0.3); }

/* Processing Animation */
.processing-pulse {
    animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

/* Divider */
.glow-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(139, 92, 246, 0.3), transparent);
    margin: 2rem 0;
    border: none;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════

def get_score_color(score: int) -> str:
    """Return color based on score threshold."""
    if score >= 80:
        return "#10b981"
    elif score >= 60:
        return "#f59e0b"
    else:
        return "#f43f5e"


def get_score_gradient(score: int) -> str:
    """Return gradient string based on score."""
    if score >= 80:
        return "linear-gradient(135deg, #10b981, #06b6d4)"
    elif score >= 60:
        return "linear-gradient(135deg, #f59e0b, #f97316)"
    else:
        return "linear-gradient(135deg, #f43f5e, #e11d48)"


def render_gauge_svg(score: int, color: str) -> str:
    """Generate an SVG radial gauge for a score."""
    radius = 45
    circumference = 2 * math.pi * radius
    offset = circumference - (score / 100) * circumference

    return f"""
    <svg class="gauge-svg" viewBox="0 0 120 120">
        <defs>
            <linearGradient id="grad-{score}" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stop-color="{color}" stop-opacity="0.8"/>
                <stop offset="100%" stop-color="{color}" stop-opacity="1"/>
            </linearGradient>
            <filter id="glow-{score}">
                <feGaussianBlur stdDeviation="3" result="glow"/>
                <feMerge><feMergeNode in="glow"/><feMergeNode in="SourceGraphic"/></feMerge>
            </filter>
        </defs>
        <circle cx="60" cy="60" r="{radius}" fill="none" stroke="rgba(255,255,255,0.05)" stroke-width="8"/>
        <circle cx="60" cy="60" r="{radius}" fill="none" stroke="url(#grad-{score})"
                stroke-width="8" stroke-linecap="round"
                stroke-dasharray="{circumference}" stroke-dashoffset="{offset}"
                transform="rotate(-90 60 60)" filter="url(#glow-{score})"
                style="transition: stroke-dashoffset 1.5s ease-out;"/>
        <text x="60" y="60" text-anchor="middle" dominant-baseline="central"
              font-size="24" font-weight="900" fill="{color}" font-family="Inter, sans-serif">
            {score}
        </text>
    </svg>
    """


def render_score_card(score: int, label: str, icon: str = ""):
    """Render a single score gauge card."""
    color = get_score_color(score)
    svg = render_gauge_svg(score, color)
    st.markdown(f"""
    <div class="gauge-container">
        {svg}
        <div class="gauge-label">{icon} {label}</div>
    </div>
    """, unsafe_allow_html=True)


def render_analysis_list(items: list, item_class: str, title: str, icon: str):
    """Render a list of analysis items as styled cards."""
    items_html = "".join([
        f'<div class="card-item {item_class}">{item}</div>'
        for item in items
    ])
    st.markdown(f"""
    <div class="analysis-card">
        <div class="card-title">{icon} {title}</div>
        {items_html}
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("# 🔬 AI Resume Analyzer")
    st.caption("Powered by Fine-tuned Qwen2 + LoRA")
    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

    st.markdown("**Model Configuration**")
    st.code("Base: qwen-ats-merged-stage1", language=None)
    st.code("Adapter: cv-analysis-final-stage2", language=None)

    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

    st.markdown("**Analysis Schema**")
    schema_items = [
        "📊 Clarity Score",
        "🏗️ Structure Score",
        "💥 Impact Score",
        "🎯 Skills Relevance",
        "🤖 ATS Readiness",
        "⭐ Overall Score",
        "💪 Strengths",
        "⚠️ Weaknesses",
        "💡 Improvements",
        "✏️ Rewrites",
    ]
    for item in schema_items:
        st.caption(item)

    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
    st.caption("v1.0 — Career Pilot Suite")


# ══════════════════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════════════════

# ── Hero Banner ────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <h1>🔬 AI-Powered Resume Analysis</h1>
    <p>Upload your resume or paste the text below. Our fine-tuned Qwen2 model will analyze it across
    5 key dimensions — clarity, structure, impact, skills relevance, and ATS readiness — and provide
    actionable feedback to help you land your dream job.</p>
</div>
""", unsafe_allow_html=True)


# ── Input Section ──────────────────────────────────────────
st.markdown('<div class="section-header">📄 Resume Input</div>', unsafe_allow_html=True)

input_method = st.radio(
    "Choose input method:",
    ["📝 Paste Text", "📎 Upload PDF"],
    horizontal=True,
    label_visibility="collapsed",
)

resume_text = ""

if input_method == "📝 Paste Text":
    resume_text = st.text_area(
        "Paste your resume text here:",
        height=300,
        placeholder="Paste your full resume content here...\n\nInclude sections like:\n• Contact Information\n• Professional Summary\n• Work Experience\n• Education\n• Skills\n• Certifications",
    )
else:
    uploaded_file = st.file_uploader(
        "Upload your resume PDF",
        type=["pdf"],
        help="Max 10 MB. Text-based PDFs only (not scanned images).",
    )
    if uploaded_file:
        try:
            from core.pdf_parser import extract_text_from_pdf
            pdf_bytes = uploaded_file.read()
            resume_text = extract_text_from_pdf(pdf_bytes) or ""
            if resume_text:
                st.success(f"✅ Extracted {len(resume_text):,} characters from PDF")
                with st.expander("👁️ Preview extracted text"):
                    st.text(resume_text[:2000] + ("..." if len(resume_text) > 2000 else ""))
            else:
                st.error("❌ Could not extract text from PDF. Please try pasting the text directly.")
        except Exception as e:
            st.error(f"❌ PDF processing error: {str(e)}")

# ── Optional Job Description ───────────────────────────────
with st.expander("🎯 Add Target Job Description (Optional — improves skills relevance scoring)"):
    job_description = st.text_area(
        "Job Description:",
        height=150,
        placeholder="Paste the target job description here for contextual analysis...",
    )

# ── Analyze Button ─────────────────────────────────────────
st.markdown("", unsafe_allow_html=True)
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    analyze_clicked = st.button("🚀 Analyze Resume", use_container_width=True, type="primary")


# ══════════════════════════════════════════════════════════════
# ANALYSIS EXECUTION & RESULTS
# ══════════════════════════════════════════════════════════════

if analyze_clicked:
    if not resume_text or len(resume_text.strip()) < 50:
        st.error("⚠️ Please provide at least 50 characters of resume text.")
    else:
        # ── Processing State ───────────────────────────────
        with st.spinner(""):
            progress_placeholder = st.empty()
            progress_placeholder.markdown("""
            <div style="text-align:center; padding: 3rem;">
                <div class="processing-pulse">
                    <div style="font-size: 3rem; margin-bottom: 1rem;">🔬</div>
                    <div style="font-size: 1.2rem; font-weight: 700; color: #a78bfa;">Analyzing your resume...</div>
                    <div style="color: #64748b; margin-top: 0.5rem;">Fine-tuned Qwen2 model is evaluating 5 dimensions</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            start_time = time.time()

            try:
                from core.analyzer import analyze_resume
                job_desc = job_description if "job_description" in dir() else ""
                result = analyze_resume(
                    resume_text=resume_text,
                    job_description=job_desc,
                )
                elapsed = time.time() - start_time
                progress_placeholder.empty()

            except Exception as e:
                progress_placeholder.empty()
                st.error(f"❌ Analysis failed: {str(e)}")
                st.info("💡 Make sure the model is loaded. Check your GPU/CUDA setup if using GPU acceleration.")
                st.stop()

        # ── Success Banner ─────────────────────────────────
        st.markdown(f"""
        <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3);
                    border-radius: 12px; padding: 1rem 1.5rem; display: flex; align-items: center;
                    justify-content: space-between; margin-bottom: 1.5rem;">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="font-size: 1.3rem;">✅</span>
                <span style="color: #10b981; font-weight: 700;">Analysis Complete</span>
            </div>
            <span class="status-pill status-success">⏱️ {elapsed:.1f}s</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

        # ── Overall Score ──────────────────────────────────
        st.markdown('<div class="section-header">⭐ Overall Score</div>', unsafe_allow_html=True)

        col_left, col_center, col_right = st.columns([1, 2, 1])
        with col_center:
            overall_color = get_score_color(int(result.overall_score))
            overall_gradient = get_score_gradient(int(result.overall_score))
            st.markdown(f"""
            <div class="overall-badge">
                <div class="overall-value" style="background: {overall_gradient};
                     -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    {result.overall_score:.0f}
                </div>
                <div class="overall-label">Overall Resume Score</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

        # ── Dimension Scores ───────────────────────────────
        st.markdown('<div class="section-header">📊 Dimension Scores</div>', unsafe_allow_html=True)

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            render_score_card(result.clarity_score, "Clarity", "📝")
        with c2:
            render_score_card(result.structure_score, "Structure", "🏗️")
        with c3:
            render_score_card(result.impact_score, "Impact", "💥")
        with c4:
            render_score_card(result.skills_relevance_score, "Skills", "🎯")
        with c5:
            render_score_card(result.ats_readiness_score, "ATS Ready", "🤖")

        st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

        # ── Detailed Analysis ──────────────────────────────
        st.markdown('<div class="section-header">📋 Detailed Analysis</div>', unsafe_allow_html=True)

        col_a, col_b = st.columns(2)

        with col_a:
            render_analysis_list(
                result.strengths, "strength-item",
                "Strengths", "💪"
            )

        with col_b:
            render_analysis_list(
                result.weaknesses, "weakness-item",
                "Weaknesses", "⚠️"
            )

        st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

        col_c, col_d = st.columns(2)

        with col_c:
            render_analysis_list(
                result.improvement_suggestions, "suggestion-item",
                "Improvement Suggestions", "💡"
            )

        with col_d:
            render_analysis_list(
                result.rewrite_suggestions, "rewrite-item",
                "Rewrite Suggestions", "✏️"
            )

        st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

        # ── Raw JSON Output ────────────────────────────────
        with st.expander("🔍 Raw JSON Output"):
            raw_dict = result.model_dump()
            st.json(raw_dict)

        # ── Download Button ────────────────────────────────
        st.markdown('<div class="section-header">📥 Export</div>', unsafe_allow_html=True)

        col_dl1, col_dl2, col_dl3 = st.columns([1, 2, 1])
        with col_dl2:
            json_str = json.dumps(result.model_dump(), indent=2, ensure_ascii=False)
            st.download_button(
                label="📥 Download Analysis (JSON)",
                data=json_str,
                file_name="resume_analysis.json",
                mime="application/json",
                use_container_width=True,
            )

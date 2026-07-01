"""CV Agent SaaS v7.0 — Streamlit Dashboard"""
import streamlit as st, time, json, os, tempfile
from pathlib import Path

st.set_page_config(page_title="CV Agent SaaS", page_icon="📄", layout="wide", initial_sidebar_state="expanded")

# ── Custom CSS ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
:root{--bg:#0f172a;--card:#1e293b;--accent:#3b82f6;--accent2:#8b5cf6;--success:#10b981;
--warn:#f59e0b;--err:#ef4444;--text:#f1f5f9;--muted:#94a3b8;--border:#334155;}
.stApp{background:var(--bg);font-family:'Inter',sans-serif;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0f172a 0%,#1e1b4b 100%);border-right:1px solid var(--border);}
[data-testid="stSidebar"] .stMarkdown h1{background:linear-gradient(135deg,#3b82f6,#8b5cf6);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-size:1.5rem;}
.metric-card{background:linear-gradient(135deg,var(--card),#1a1a2e);border:1px solid var(--border);border-radius:16px;padding:1.5rem;text-align:center;transition:transform .2s,box-shadow .2s;}
.metric-card:hover{transform:translateY(-4px);box-shadow:0 8px 32px rgba(59,130,246,.15);}
.metric-value{font-size:2rem;font-weight:800;background:linear-gradient(135deg,#3b82f6,#8b5cf6);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.metric-label{color:var(--muted);font-size:.85rem;margin-top:.3rem;}
.hero{background:linear-gradient(135deg,rgba(59,130,246,.1),rgba(139,92,246,.1));border:1px solid var(--border);border-radius:20px;padding:2.5rem;margin-bottom:2rem;}
.hero h2{background:linear-gradient(135deg,#60a5fa,#a78bfa);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-size:2rem;margin:0;}
.hero p{color:var(--muted);font-size:1rem;margin-top:.5rem;}
.score-pill{display:inline-block;padding:4px 14px;border-radius:50px;font-weight:700;font-size:.85rem;margin:2px;}
.score-high{background:rgba(16,185,129,.15);color:#10b981;border:1px solid rgba(16,185,129,.3);}
.score-mid{background:rgba(245,158,11,.15);color:#f59e0b;border:1px solid rgba(245,158,11,.3);}
.score-low{background:rgba(239,68,68,.15);color:#ef4444;border:1px solid rgba(239,68,68,.3);}
.feature-badge{display:inline-block;padding:3px 10px;border-radius:6px;font-size:.75rem;font-weight:600;margin:2px;}
.feat-on{background:rgba(16,185,129,.15);color:#10b981;}.feat-off{background:rgba(239,68,68,.15);color:#ef4444;}
.cv-output{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:1.5rem;color:var(--text);line-height:1.7;}
.guard-pass{background:rgba(16,185,129,.1);border:1px solid rgba(16,185,129,.3);border-radius:12px;padding:1rem;color:#10b981;}
.guard-fail{background:rgba(239,68,68,.1);border:1px solid rgba(239,68,68,.3);border-radius:12px;padding:1rem;color:#ef4444;}
div[data-testid="stForm"]{background:var(--card);border:1px solid var(--border);border-radius:16px;padding:1.5rem;}
.stButton>button{background:linear-gradient(135deg,#3b82f6,#8b5cf6)!important;color:white!important;border:none!important;border-radius:10px!important;padding:.6rem 2rem!important;font-weight:600!important;transition:all .2s!important;}
.stButton>button:hover{transform:translateY(-2px)!important;box-shadow:0 6px 20px rgba(59,130,246,.3)!important;}
.stTextInput>div>div>input,.stTextArea>div>div>textarea,.stSelectbox>div>div{background:var(--bg)!important;border:1px solid var(--border)!important;color:var(--text)!important;border-radius:10px!important;}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──
with st.sidebar:
    st.markdown("# 📄 CV Agent SaaS")
    st.caption("v7.0 — Agentic CV Builder")
    st.divider()
    page = st.radio("Navigation", ["🏠 Dashboard", "✍️ Generate CV", "🛡️ Guard Test", "📊 Scores", "⚙️ System"], label_visibility="collapsed")
    st.divider()
    st.markdown("**Quick Stats**")
    try:
        from cv_agent.cache import get_cache
        stats = get_cache().stats()
        st.metric("Cache Size", stats["size"])
        st.metric("Cache Hits", stats["hits"])
    except Exception:
        st.caption("Cache not initialized")

# ── Imports ──
from cv_agent.config import PipelineConfig, _FASTAPI_AVAILABLE, _SKLEARN_AVAILABLE, _FAISS_AVAILABLE, _SENTENCE_AVAILABLE, _REPORTLAB_AVAILABLE, _MISTUNE_AVAILABLE
from cv_agent.schemas import UserProfile, JudgeOutput, JDContext
from cv_agent.hallucination_guard import HallucinationGuard
from cv_agent.judges import RuleJudge
from cv_agent.cache import LRUCache, get_cache

def score_pill(val, label=""):
    cls = "score-high" if val >= 80 else "score-mid" if val >= 60 else "score-low"
    return f'<span class="score-pill {cls}">{label}{val}</span>'

# ══════════════════════════════════════════
# PAGE: Dashboard
# ══════════════════════════════════════════
if page == "🏠 Dashboard":
    st.markdown('<div class="hero"><h2>Agentic CV Generation Platform</h2><p>LangGraph + Ensemble Judges + Hallucination Guard + FastAPI</p></div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    features = {"FAISS": _FAISS_AVAILABLE, "Sentence-T": _SENTENCE_AVAILABLE, "sklearn": _SKLEARN_AVAILABLE, "ReportLab": _REPORTLAB_AVAILABLE, "Mistune": _MISTUNE_AVAILABLE, "FastAPI": _FASTAPI_AVAILABLE}
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">17</div><div class="metric-label">Modules</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">52</div><div class="metric-label">Unit Tests</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-value">v7.0</div><div class="metric-label">Version</div></div>', unsafe_allow_html=True)
    with c4:
        on = sum(1 for v in features.values() if v)
        st.markdown(f'<div class="metric-card"><div class="metric-value">{on}/{len(features)}</div><div class="metric-label">Features Active</div></div>', unsafe_allow_html=True)

    st.markdown("### 🔌 Feature Flags")
    badges = " ".join(f'<span class="feature-badge {"feat-on" if v else "feat-off"}">{"✓" if v else "✗"} {k}</span>' for k, v in features.items())
    st.markdown(badges, unsafe_allow_html=True)

    st.markdown("### 🏗️ Architecture")
    st.markdown("""
    ```
    ┌─────────────┐    ┌──────────────┐    ┌──────────────┐
    │  Streamlit   │───▶│  cv_agent/   │───▶│  GPUQueue    │
    │  Dashboard   │    │  pipeline.py │    │  (Serial)    │
    └─────────────┘    └──────┬───────┘    └──────────────┘
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
              ┌──────────┐      ┌──────────────┐
              │  Judges  │      │  Hallucination│
              │ ATS+HR+  │      │    Guard      │
              │  Rule    │      │  (Standalone) │
              └──────────┘      └──────────────┘
    ```""")

# ══════════════════════════════════════════
# PAGE: Generate CV
# ══════════════════════════════════════════
elif page == "✍️ Generate CV":
    st.markdown("## ✍️ Generate CV")
    st.caption("Fill in the candidate profile to generate an optimized CV")

    # ── Template Selection ──
    st.markdown("### 🎨 Choose Your CV Template")
    st.caption("Select a design template for your CV output")

    # Template CSS
    st.markdown("""
    <style>
    .template-grid{display:flex;gap:1.2rem;margin:1rem 0 1.5rem 0;}
    .template-card{flex:1;background:linear-gradient(145deg,#1e293b,#0f172a);border:2px solid #334155;
        border-radius:16px;padding:1.3rem;cursor:pointer;transition:all .3s cubic-bezier(.4,0,.2,1);position:relative;overflow:hidden;}
    .template-card:hover{transform:translateY(-6px);box-shadow:0 12px 40px rgba(59,130,246,.2);}
    .template-card.selected{border-color:#3b82f6;box-shadow:0 0 20px rgba(59,130,246,.25);}
    .template-card.selected::after{content:'✓ Selected';position:absolute;top:10px;right:10px;
        background:linear-gradient(135deg,#3b82f6,#8b5cf6);color:white;font-size:.65rem;font-weight:700;
        padding:3px 10px;border-radius:20px;}
    .tpl-name{font-size:1.1rem;font-weight:700;margin-bottom:.3rem;}
    .tpl-desc{font-size:.78rem;color:#94a3b8;margin-bottom:.8rem;line-height:1.4;}
    .tpl-preview{background:#f8fafc;border-radius:8px;padding:10px 12px;min-height:130px;}

    /* Classic preview */
    .preview-classic .pv-name{text-align:center;font-weight:800;font-size:.85rem;color:#0f172a;margin-bottom:2px;}
    .preview-classic .pv-contact{text-align:center;font-size:.55rem;color:#475569;margin-bottom:6px;}
    .preview-classic .pv-section{font-weight:700;font-size:.6rem;color:#2563eb;border-bottom:1.5px solid #2563eb;
        padding-bottom:2px;margin:5px 0 3px 0;text-transform:uppercase;}
    .preview-classic .pv-line{height:4px;background:#e2e8f0;border-radius:2px;margin:3px 0;width:85%;}
    .preview-classic .pv-line.short{width:60%;}

    /* Modern preview */
    .preview-modern .pv-name{text-align:left;font-weight:800;font-size:.85rem;color:#0f3460;margin-bottom:2px;}
    .preview-modern .pv-contact{text-align:left;font-size:.55rem;color:#0d9488;margin-bottom:6px;}
    .preview-modern .pv-section{font-weight:700;font-size:.6rem;color:#0d9488;border-bottom:1.5px solid #0d9488;
        padding-bottom:2px;margin:5px 0 3px 0;text-transform:uppercase;}
    .preview-modern .pv-line{height:4px;background:#e2e8f0;border-radius:2px;margin:3px 0;width:90%;}
    .preview-modern .pv-line.short{width:65%;}
    .preview-modern{border-left:3px solid #0d9488;}

    /* Monochrome preview */
    .preview-mono .pv-name{text-align:center;font-weight:800;font-size:.85rem;color:#000;margin-bottom:2px;}
    .preview-mono .pv-contact{text-align:center;font-size:.55rem;color:#2a2a2a;margin-bottom:6px;}
    .preview-mono .pv-section{font-weight:700;font-size:.6rem;color:#000;border-bottom:1.5px solid #333;
        padding-bottom:2px;margin:5px 0 3px 0;text-transform:uppercase;}
    .preview-mono .pv-line{height:4px;background:#d4d4d4;border-radius:2px;margin:3px 0;width:85%;}
    .preview-mono .pv-line.short{width:60%;}

    /* CV Output template-specific styles */
    .cv-classic{color:#1e293b;}
    .cv-classic h1,.cv-classic h2,.cv-classic h3{color:#0f172a;}
    .cv-classic h2{color:#2563eb;border-bottom:2px solid #2563eb;padding-bottom:4px;}

    .cv-modern{color:#1a1a2e;}
    .cv-modern h1{color:#0f3460;font-size:1.8rem;}
    .cv-modern h2{color:#0d9488;border-bottom:2px solid #0d9488;padding-bottom:4px;text-transform:uppercase;font-size:1rem;}
    .cv-modern h3{color:#0f3460;}

    .cv-mono{color:#1a1a1a;}
    .cv-mono h1,.cv-mono h2,.cv-mono h3{color:#000;}
    .cv-mono h2{border-bottom:1.5px solid #333;padding-bottom:4px;}
    </style>
    """, unsafe_allow_html=True)

    # Initialize template in session state
    if "cv_template" not in st.session_state:
        st.session_state.cv_template = "classic"

    # Template preview cards
    preview_block = """
    <div class="template-grid">
        <div class="template-card {sel_classic}" onclick="window.parent.postMessage({type:'streamlit:setComponentValue',value:'classic'},'*')">
            <div class="tpl-name" style="color:#60a5fa;">📋 Classic</div>
            <div class="tpl-desc">Centered name, blue section headers, professional & balanced layout</div>
            <div class="tpl-preview preview-classic">
                <div class="pv-name">Jessie Smith</div>
                <div class="pv-contact">email@mail.com | +1 234 567 | linkedin.com/in/jessie</div>
                <div class="pv-section">Professional Summary</div>
                <div class="pv-line"></div>
                <div class="pv-line short"></div>
                <div class="pv-section">Experience</div>
                <div class="pv-line"></div>
                <div class="pv-line short"></div>
                <div class="pv-line"></div>
            </div>
        </div>
        <div class="template-card {sel_modern}">
            <div class="tpl-name" style="color:#2dd4bf;">🎯 Modern</div>
            <div class="tpl-desc">Left-aligned with teal accent sidebar, compact & sleek design</div>
            <div class="tpl-preview preview-modern">
                <div class="pv-name">Jessie Smith</div>
                <div class="pv-contact">email@mail.com | +1 234 567 | linkedin.com/in/jessie</div>
                <div class="pv-section">Professional Summary</div>
                <div class="pv-line"></div>
                <div class="pv-line short"></div>
                <div class="pv-section">Experience</div>
                <div class="pv-line"></div>
                <div class="pv-line short"></div>
                <div class="pv-line"></div>
            </div>
        </div>
        <div class="template-card {sel_mono}">
            <div class="tpl-name" style="color:#e2e8f0;">⬛ Monochrome</div>
            <div class="tpl-desc">Pure black & white — zero colors, clean minimal design</div>
            <div class="tpl-preview preview-mono">
                <div class="pv-name">Jessie Smith</div>
                <div class="pv-contact">email@mail.com | +1 234 567 | linkedin.com/in/jessie</div>
                <div class="pv-section">Professional Summary</div>
                <div class="pv-line"></div>
                <div class="pv-line short"></div>
                <div class="pv-section">Experience</div>
                <div class="pv-line"></div>
                <div class="pv-line short"></div>
                <div class="pv-line"></div>
            </div>
        </div>
    </div>
    """.replace(
        "{sel_classic}", "selected" if st.session_state.cv_template == "classic" else ""
    ).replace(
        "{sel_modern}", "selected" if st.session_state.cv_template == "modern" else ""
    ).replace(
        "{sel_mono}", "selected" if st.session_state.cv_template == "monochrome" else ""
    )
    st.markdown(preview_block, unsafe_allow_html=True)

    # Actual selection via Streamlit radio (functional)
    tpl_options = {"📋 Classic": "classic", "🎯 Modern": "modern", "⬛ Monochrome": "monochrome"}
    tpl_display = list(tpl_options.keys())
    current_idx = list(tpl_options.values()).index(st.session_state.cv_template)
    selected_display = st.radio(
        "Select Template", tpl_display, index=current_idx,
        horizontal=True, key="tpl_radio", label_visibility="collapsed"
    )
    st.session_state.cv_template = tpl_options[selected_display]

    st.divider()

    with st.form("cv_form"):
        col1, col2 = st.columns(2)
        with col1:
            full_name = st.text_input("Full Name *", placeholder="Ahmed Hassan")
            target_role = st.text_input("Target Role *", placeholder="Senior Data Scientist")
            industry = st.text_input("Industry", placeholder="Technology")
            years = st.text_input("Years of Experience", placeholder="5 years")
            tone = st.selectbox("Tone", ["professional", "creative", "technical"])
        with col2:
            skills = st.text_area("Skills * (Categorized or comma-separated)", placeholder="Programming: Python, SQL\nData Analysis: Pandas, NumPy\nOR just comma-separated: Python, SQL, Pandas", height=80, help="Enter skills comma-separated. They will be auto-categorized by AI. You can also pre-categorize using 'Category: skill1, skill2' format.")
            experiences = st.text_area("Work Experience (semicolon-separated)", placeholder="ML Engineer at Google 2020-2024; Data Analyst at Meta 2018-2020", height=100, help="Leave empty if you have no work experience — use Projects below instead")
            projects = st.text_area("Projects (semicolon-separated)", placeholder="E-commerce site using React and Node.js; ML sentiment analysis model with Python and TensorFlow", height=100, help="If you have no work experience, add your projects here")
            education = st.text_area("Education (semicolon-separated)", placeholder="MSc CS, MIT 2018; BSc Math, Stanford 2016", height=68)

        st.markdown("**Contact Information**")
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            email = st.text_input("Email *", placeholder="ahmed@example.com")
        with cc2:
            phone = st.text_input("Phone Number *", placeholder="+1 234 567 8900")
        with cc3:
            linkedin = st.text_input("LinkedIn Profile *", placeholder="linkedin.com/in/ahmed")

        col3, col4 = st.columns(2)
        with col3:
            achievements = st.text_area("Achievements (semicolon-separated)", placeholder="Improved model accuracy by 30%", height=68)
            certifications = st.text_input("Certifications (comma-separated)", placeholder="AWS ML Specialty, GCP Data Engineer")
        with col4:
            summary = st.text_area("Summary Notes", placeholder="Brief career narrative...", height=68)
            jd = st.text_area("Job Description (optional)", placeholder="Paste JD for keyword alignment...", height=68)

        st.divider()
        acol1, acol2, acol3 = st.columns(3)
        with acol1:
            max_iter = st.slider("Max Iterations", 1, 10, 5)
        with acol2:
            threshold = st.slider("Score Threshold", 50, 100, 82)
        with acol3:
            n_candidates = st.slider("Candidates per Round", 1, 5, 3)

        submitted = st.form_submit_button("🚀 Generate CV", use_container_width=True)

    if submitted:
        has_exp_or_projects = bool(experiences.strip() or projects.strip())
        if not full_name or not target_role or not skills or not has_exp_or_projects or not email or not phone or not linkedin:
            if not has_exp_or_projects:
                st.error("Please provide either Work Experience or Projects (at least one is required)")
            else:
                st.error("Please fill in all required fields (marked with *)")
        else:
            # Parse categorized vs flat skills
            parsed_skills = []
            parsed_categories = {}
            for line in skills.split('\n'):
                line = line.strip()
                if not line:
                    continue
                if ':' in line:
                    cat, cat_skills_str = line.split(':', 1)
                    cat_skills = [s.strip() for s in cat_skills_str.split(',') if s.strip()]
                    if cat_skills:
                        parsed_categories[cat.strip()] = cat_skills
                        parsed_skills.extend(cat_skills)
                else:
                    flat = [s.strip() for s in line.split(',') if s.strip()]
                    parsed_skills.extend(flat)

            profile = UserProfile(
                full_name=full_name, target_role=target_role,
                target_industry=industry, years_experience=years,
                tone=tone, summary=summary,
                email=email, phone=phone, linkedin=linkedin,
                skills=parsed_skills,
                skill_categories=parsed_categories if parsed_categories else None,
                experiences=[e.strip() for e in experiences.split(";") if e.strip()],
                projects=[p.strip() for p in projects.split(";") if p.strip()],
                education=[e.strip() for e in education.split(";") if e.strip()],
                achievements=[a.strip() for a in achievements.split(";") if a.strip()],
                certifications=[c.strip() for c in certifications.split(",") if c.strip()],
            )
            cfg = PipelineConfig()
            cfg.max_iterations = max_iter
            cfg.score_threshold = threshold
            cfg.num_candidates = n_candidates
            cfg.cv_template = st.session_state.cv_template

            progress = st.empty()
            status_box = st.empty()
            msgs = []

            from streamlit.runtime.scriptrunner import get_script_run_ctx, add_script_run_ctx
            ctx = get_script_run_ctx()

            def _cb(msg):
                add_script_run_ctx(ctx=ctx)
                msgs.append(msg)
                status_box.code("\n".join(msgs[-8:]), language="text")

            progress.info("⏳ Pipeline starting... This may take several minutes with GPU inference.")
            try:
                from cv_agent.pipeline import run_pipeline
                t0 = time.time()
                result = run_pipeline(profile=profile, job_description=jd, config=cfg, status_callback=_cb)
                elapsed = time.time() - t0
                progress.success(f"✅ Done in {elapsed:.1f}s — {result.total_iterations} iteration(s)")

                if result.final_scores:
                    s = result.final_scores
                    st.markdown("### 📊 Final Scores")
                    pills = " ".join([
                        score_pill(s.overall_score, "Overall "), score_pill(s.clarity_score, "Clarity "),
                        score_pill(s.structure_score, "Structure "), score_pill(s.impact_score, "Impact "),
                        score_pill(s.skills_relevance_score, "Skills "), score_pill(s.ats_readiness_score, "ATS "),
                    ])
                    st.markdown(pills, unsafe_allow_html=True)

                # Apply template-specific CSS class to CV output
                tpl_class = {"classic": "cv-classic", "modern": "cv-modern", "monochrome": "cv-mono"}.get(st.session_state.cv_template, "cv-classic")
                st.markdown("### 📄 Generated CV")
                st.markdown(f'<div class="cv-output {tpl_class}">{result.final_cv}</div>', unsafe_allow_html=True)

                if result.content_check:
                    c = result.content_check
                    with st.expander("📝 Content Quality Report", expanded=False):
                        st.markdown(f"**Overall Quality:** `{c.overall_quality.upper()}`")
                        st.caption(c.summary)
                        
                        if c.repeated_words:
                            st.markdown("**🔄 Repeated Words (4+ times):**")
                            rep_pills = " ".join([f'<span class="feature-badge feat-off">{w} ({cnt})</span>' for w, cnt in c.repeated_words.items()])
                            st.markdown(rep_pills, unsafe_allow_html=True)
                            
                        if c.issues:
                            st.markdown("**⚠️ Detected Issues:**")
                            for i, issue in enumerate(c.issues, 1):
                                icon = "❌" if issue.severity == "error" else "⚠️" if issue.severity == "warning" else "💡"
                                st.markdown(f"**{i}. {icon} [{issue.category}] in *{issue.location}***")
                                st.markdown(f"> \"{issue.text}\"")
                                st.markdown(f"**Suggestion:** {issue.suggestion}")
                                st.divider()


                selected_template = st.session_state.cv_template
                dcol1, dcol2 = st.columns(2)
                with dcol1:
                    st.download_button("📥 Download Markdown", result.final_cv, f"{full_name}_CV.md", "text/markdown", use_container_width=True)
                with dcol2:
                    try:
                        from cv_agent.pdf_export import export_pdf_bytes
                        pdf = export_pdf_bytes(result.final_cv, full_name, template=selected_template)
                        st.download_button("📥 Download PDF", pdf, f"{full_name}_CV.pdf", "application/pdf", use_container_width=True)
                    except Exception as e:
                        st.warning(f"PDF export unavailable: {e}")

            except Exception as e:
                progress.error(f"❌ Pipeline failed: {e}")
                st.exception(e)

# ══════════════════════════════════════════
# PAGE: Guard Test
# ══════════════════════════════════════════
elif page == "🛡️ Guard Test":
    st.markdown("## 🛡️ Hallucination Guard Tester")
    st.caption("Test CV text against a profile to detect hallucinations")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Candidate Profile**")
        g_name = st.text_input("Name", "Ahmed Hassan", key="g_name")
        g_role = st.text_input("Role", "Data Scientist", key="g_role")
        g_years = st.text_input("Years", "3 years", key="g_years")
        g_skills = st.text_area("Skills (comma-sep)", "Python, SQL, TensorFlow, Pandas, scikit-learn", key="g_skills", height=80)
        g_exp = st.text_area("Experience (semicolon-sep)", "ML Engineer at Acme Corp 2021-2024", key="g_exp", height=80)
        g_jd_kw = st.text_input("JD Keywords (comma-sep, optional)", "", key="g_jd")
    with col2:
        st.markdown("**CV Text to Validate**")
        g_cv = st.text_area("Paste CV text here", height=350, key="g_cv", value="""## Professional Summary
Data Scientist with 3 years experience in ML.

## SKILLS
Python, SQL, TensorFlow, Pandas, Kubernetes, Apache Kafka

## Professional Experience
ML Engineer at Acme Corp 2021-2024
- Improved model accuracy by 15%
- Generated $50 Billion in revenue
- Extensive experience leading teams""")

    ontology_path = st.text_input("Ontology File Path (optional)", "", help="Path to JSON/YAML ontology file")

    if st.button("🔍 Run Hallucination Guard", use_container_width=True):
        profile = UserProfile(
            full_name=g_name, target_role=g_role, years_experience=g_years,
            skills=[s.strip() for s in g_skills.split(",") if s.strip()],
            experiences=[e.strip() for e in g_exp.split(";") if e.strip()],
        )
        jd_ctx = JDContext(keywords=[k.strip() for k in g_jd_kw.split(",") if k.strip()]) if g_jd_kw else None
        guard = HallucinationGuard(ontology_path=ontology_path) if ontology_path else HallucinationGuard()
        result = guard.validate(g_cv, profile, jd_context=jd_ctx)

        if result.passed:
            st.markdown('<div class="guard-pass">✅ <strong>PASSED</strong> — No hallucinations detected</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="guard-fail">❌ <strong>FAILED</strong> — {len(result.issues)} issue(s) detected</div>', unsafe_allow_html=True)
            for i, issue in enumerate(result.issues, 1):
                st.warning(f"**Issue {i}:** {issue}")

        with st.expander("🔎 Debug Info"):
            tokens = profile.authorised_tokens()
            st.markdown(f"**Profile hash:** `{profile.profile_hash()}`")
            st.markdown(f"**Authorised tokens:** {len(tokens)} tokens")
            st.markdown(f"**Ontology size:** {len(guard.TECH_ONTOLOGY)} tech, {len(guard.PROJECT_CONCEPTS)} projects, {len(guard.METHODOLOGIES)} methods")

    st.divider()
    st.markdown("### 📏 Rule Judge (Heuristic)")
    if st.button("Run Rule Judge on CV above"):
        rj = RuleJudge()
        out = rj.evaluate(g_cv)
        pills = " ".join([
            score_pill(out.overall_score, "Overall "), score_pill(out.clarity_score, "Clarity "),
            score_pill(out.structure_score, "Structure "), score_pill(out.impact_score, "Impact "),
            score_pill(out.skills_relevance_score, "Skills "), score_pill(out.ats_readiness_score, "ATS "),
        ])
        st.markdown(pills, unsafe_allow_html=True)
        if out.weaknesses:
            st.markdown("**Weaknesses:**")
            for w in out.weaknesses:
                st.caption(f"⚠️ {w}")
        if out.improvement_suggestions:
            st.markdown("**Suggestions:**")
            for s in out.improvement_suggestions:
                st.caption(f"💡 {s}")

# ══════════════════════════════════════════
# PAGE: Scores
# ══════════════════════════════════════════
elif page == "📊 Scores":
    st.markdown("## 📊 Score Analysis")
    st.caption("Analyze JudgeOutput scoring and thresholds")

    st.markdown("### Score Simulator")
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        sim_clarity = st.slider("Clarity", 0, 100, 80)
        sim_structure = st.slider("Structure", 0, 100, 75)
    with sc2:
        sim_impact = st.slider("Impact", 0, 100, 85)
        sim_skills = st.slider("Skills Relevance", 0, 100, 70)
    with sc3:
        sim_ats = st.slider("ATS Readiness", 0, 100, 90)
        sim_threshold = st.slider("Pass Threshold", 50, 100, 82)

    sim = JudgeOutput(clarity_score=sim_clarity, structure_score=sim_structure, impact_score=sim_impact,
                      skills_relevance_score=sim_skills, ats_readiness_score=sim_ats,
                      overall_score=int((sim_clarity+sim_structure+sim_impact+sim_skills+sim_ats)/5))

    st.divider()
    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{sim.overall_score}</div><div class="metric-label">Overall</div></div>', unsafe_allow_html=True)
    with mc2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{sim.average_score():.1f}</div><div class="metric-label">Average</div></div>', unsafe_allow_html=True)
    with mc3:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{sim.lowest_metric()}</div><div class="metric-label">Weakest Area</div></div>', unsafe_allow_html=True)
    with mc4:
        passes = sim.passes(sim_threshold)
        cls = "score-high" if passes else "score-low"
        st.markdown(f'<div class="metric-card"><span class="score-pill {cls}">{"PASS ✓" if passes else "FAIL ✗"}</span><div class="metric-label">@ threshold {sim_threshold}</div></div>', unsafe_allow_html=True)

    st.markdown("### All Scores")
    pills = " ".join([
        score_pill(sim_clarity, "Clarity "), score_pill(sim_structure, "Structure "),
        score_pill(sim_impact, "Impact "), score_pill(sim_skills, "Skills "),
        score_pill(sim_ats, "ATS "), score_pill(sim.overall_score, "Overall "),
    ])
    st.markdown(pills, unsafe_allow_html=True)

# ══════════════════════════════════════════
# PAGE: System
# ══════════════════════════════════════════
elif page == "⚙️ System":
    st.markdown("## ⚙️ System & Configuration")

    st.markdown("### Pipeline Configuration")
    cfg = PipelineConfig()
    cfg_data = {
        "Writer Model": cfg.writer_model, "Judge Base": cfg.judge_base,
        "Judge Adapter": cfg.judge_adapter, "HR Judge": cfg.hr_judge_model,
        "Max Iterations": cfg.max_iterations, "Score Threshold": cfg.score_threshold,
        "Num Candidates": cfg.num_candidates, "ATS Weight": cfg.ats_weight,
        "HR Weight": cfg.hr_weight, "Rule Weight": cfg.rule_weight,
        "4-bit Quant": cfg.load_in_4bit, "Cache TTL": f"{cfg.cache_ttl_seconds}s",
        "Cache Max Size": cfg.cache_max_size, "Ontology Path": cfg.ontology_path or "(none)",
        "CV Template": cfg.cv_template,
    }
    left, right = st.columns(2)
    items = list(cfg_data.items())
    for k, v in items[:len(items)//2]:
        left.markdown(f"**{k}:** `{v}`")
    for k, v in items[len(items)//2:]:
        right.markdown(f"**{k}:** `{v}`")

    st.divider()
    st.markdown("### 📦 Cache Management")
    cache = get_cache(cfg)
    stats = cache.stats()
    cc1, cc2, cc3 = st.columns(3)
    cc1.metric("Entries", stats["size"])
    cc2.metric("Hits", stats["hits"])
    cc3.metric("Misses", stats["misses"])
    if st.button("🗑️ Clear Cache"):
        cache.clear()
        st.success("Cache cleared!")
        st.rerun()

    st.divider()
    st.markdown("### 🧪 Module Health Check")
    modules = [
        ("cv_agent.config", "config"), ("cv_agent.schemas", "schemas"),
        ("cv_agent.cache", "cache"), ("cv_agent.gpu_queue", "gpu_queue"),
        ("cv_agent.model_manager", "model_manager"), ("cv_agent.prompts", "prompts"),
        ("cv_agent.hallucination_guard", "hallucination_guard"),
        ("cv_agent.judges", "judges"), ("cv_agent.rag", "rag"),
        ("cv_agent.memory", "memory"), ("cv_agent.routing", "routing"),
        ("cv_agent.pipeline", "pipeline"), ("cv_agent.api", "api"),
        ("cv_agent.cli", "cli"), ("cv_agent.pdf_export", "pdf_export"),
        ("cv_agent.file_parsing", "file_parsing"),
    ]
    results = []
    for mod, name in modules:
        try:
            __import__(mod)
            results.append((name, True))
        except Exception as e:
            results.append((name, False))

    badges = " ".join(
        f'<span class="feature-badge {"feat-on" if ok else "feat-off"}">{"✓" if ok else "✗"} {n}</span>'
        for n, ok in results
    )
    st.markdown(badges, unsafe_allow_html=True)

    st.divider()
    st.markdown("### 🧬 GPU Queue Status")
    try:
        from cv_agent.gpu_queue import _gpu_queue
        st.markdown(f"**Worker Alive:** {'🟢 Yes' if _gpu_queue.is_alive else '🔴 No'}")
        st.markdown(f"**Queue Size:** {_gpu_queue._q.qsize()}")
        st.markdown(f"**Restart Count:** {_gpu_queue._restart_count}")
    except Exception as e:
        st.warning(f"Could not check GPU queue: {e}")

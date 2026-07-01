# NHA-4-151
<div align="center">

# 🚀 Career Pilot

### AI-Powered Career Management Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agentic_AI-FF6F00?style=for-the-badge)](https://python.langchain.com/docs/langgraph)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)

**Career Pilot** is a modular, AI-driven platform that covers the entire career lifecycle — from **analyzing your resume**, to **building an optimized CV**, to **matching you with relevant jobs**, and **conducting AI-powered voice interviews**.

[Resume Analysis](#-1-resume-analysis) · [CV Builder](#-2-cv-builder-agent) · [Job Matching](#-3-job-matching-agent) · [Interview System](#-4-ai-voice-interview-system)

</div>

---

## 📑 Table of Contents

- [Platform Overview](#-platform-overview)
- [High-Level Architecture](#-high-level-architecture)
- [Sub-Projects](#-sub-projects)
  - [1. Resume Analysis](#-1-resume-analysis)
  - [2. CV Builder Agent](#-2-cv-builder-agent)
  - [3. Job Matching Agent](#-3-job-matching-agent)
  - [4. AI Voice Interview System](#-4-ai-voice-interview-system)
- [Technology Stack](#-technology-stack)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🌐 Platform Overview

Career Pilot is composed of **4 independent but complementary sub-projects**, each tackling a critical stage in the job-seeking process:

| # | Sub-Project | Description | Core Tech |
|---|-------------|-------------|-----------|
| 1 | **Resume Analysis** | AI-powered resume scoring using a fine-tuned Qwen2 model with LoRA | Transformers, PEFT, FastAPI |
| 2 | **CV Builder Agent** | Agentic CV generation with ensemble judges & hallucination guards | LangGraph, FAISS, ReportLab |
| 3 | **Job Matching Agent** | Deep job research, skill matching & cover letter generation | LangGraph, Tavily, FPDF |
| 4 | **Voice Interview** | Real-time AI voice interview with adaptive follow-ups | LangGraph, WebSocket, Whisper |

---

## 🏗 High-Level Architecture

```mermaid
graph TB
    subgraph Career_Pilot["🚀 Career Pilot Platform"]
        direction TB

        subgraph RA["📊 Resume Analysis"]
            RA_UI["Streamlit UI"] --> RA_API["FastAPI Backend"]
            RA_API --> RA_MODEL["Qwen2 + LoRA<br/>Fine-tuned Model"]
        end

        subgraph CV["📄 CV Builder"]
            CV_UI["Streamlit Dashboard"] --> CV_PIPE["LangGraph Pipeline"]
            CV_PIPE --> CV_W["Writer Agent"]
            CV_PIPE --> CV_J["Ensemble Judges"]
            CV_PIPE --> CV_G["Hallucination Guard"]
        end

        subgraph JM["🔍 Job Matching"]
            JM_UI["Streamlit UI"] --> JM_AGENT["LangGraph Agent"]
            JM_AGENT --> JM_SEARCH["Tavily Search<br/>10+ Job Boards"]
            JM_AGENT --> JM_REPORT["PDF Report +<br/>Cover Letters"]
        end

        subgraph IV["🎙️ Voice Interview"]
            IV_UI["Web UI"] --> IV_API["FastAPI + WebSocket"]
            IV_API --> IV_GRAPH["LangGraph<br/>Interview Engine"]
            IV_API --> IV_VOICE["STT / TTS / VAD<br/>Audio Pipeline"]
        end
    end

    USER((👤 User)) --> RA_UI
    USER --> CV_UI
    USER --> JM_UI
    USER --> IV_UI

    style Career_Pilot fill:#0d1117,stroke:#58a6ff,stroke-width:2px,color:#c9d1d9
    style RA fill:#1a1f2e,stroke:#7c3aed,stroke-width:1px,color:#e2e8f0
    style CV fill:#1a1f2e,stroke:#06b6d4,stroke-width:1px,color:#e2e8f0
    style JM fill:#1a1f2e,stroke:#f59e0b,stroke-width:1px,color:#e2e8f0
    style IV fill:#1a1f2e,stroke:#10b981,stroke-width:1px,color:#e2e8f0
```

---

## 📦 Sub-Projects

---

### 📊 1. Resume Analysis

> AI-powered resume evaluation using a **fine-tuned Qwen2** model with **LoRA adapter** for structured CV scoring.

#### Features

- 🤖 **Fine-Tuned Model** — Custom Qwen2 model (`OsamaHayba/qwen-ats-merged-stage1`) with LoRA adapter (`OsamaHayba/cv-analysis-final-stage2`)
- 📊 **10-Field Structured Output** — Clarity, Structure, Impact, Skills Relevance, ATS Readiness scores + strengths, weaknesses, and suggestions
- ⚡ **4-bit Quantization** — Runs efficiently on consumer GPUs (≥6 GB VRAM) via bitsandbytes NF4
- 📄 **PDF Parsing** — Direct PDF upload and text extraction via PyMuPDF
- 🌐 **Dual Mode** — Run as Streamlit-only (direct inference) or FastAPI + Streamlit

#### Architecture

```mermaid
graph LR
    A["📄 Resume<br/>(PDF/Text)"] --> B["pdf_parser.py<br/>Text Extraction"]
    B --> C["analyzer.py<br/>Prompt Engineering"]
    C --> D["model_loader.py<br/>Qwen2 + LoRA"]
    D --> E["4-bit Inference<br/>(GPU/CPU)"]
    E --> F["schemas.py<br/>Pydantic Validation"]
    F --> G["📊 Structured<br/>JSON Report"]

    style A fill:#7c3aed,stroke:#7c3aed,color:#fff
    style D fill:#4f46e5,stroke:#4f46e5,color:#fff
    style G fill:#06b6d4,stroke:#06b6d4,color:#fff
```

#### Data Flow

```mermaid
sequenceDiagram
    participant U as 👤 User
    participant S as Streamlit UI
    participant API as FastAPI
    participant M as Qwen2 + LoRA
    participant V as Pydantic Validator

    U->>S: Upload Resume (PDF)
    S->>API: POST /analyze/pdf
    API->>API: Extract text (PyMuPDF)
    API->>M: Build chat prompt (Qwen template)
    M->>M: Generate analysis (4-bit inference)
    M->>V: Raw JSON output
    V->>V: Multi-strategy JSON extraction
    V->>V: Schema validation
    V->>API: ResumeAnalysis object
    API->>S: 10-field structured result
    S->>U: Scores + Suggestions Dashboard
```

#### Output Schema

| Field | Type | Description |
|-------|------|-------------|
| `clarity_score` | int (0-100) | How clear and readable |
| `structure_score` | int (0-100) | How well-organized |
| `impact_score` | int (0-100) | How effectively achievements are communicated |
| `skills_relevance_score` | int (0-100) | How relevant skills are to target role |
| `ats_readiness_score` | int (0-100) | ATS optimization level |
| `overall_score` | float (0-100) | Weighted overall score |
| `strengths` | list[str] | Resume strengths |
| `weaknesses` | list[str] | Resume weaknesses |
| `improvement_suggestions` | list[str] | Actionable improvements |
| `rewrite_suggestions` | list[str] | Specific rewrite recommendations |

#### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/analyze` | Analyze resume text |
| `POST` | `/analyze/pdf` | Upload & analyze PDF |
| `GET` | `/health` | Health check + model status |

<details>
<summary>📁 Directory Structure</summary>

```text
resume-analysis/
├── app.py                  # FastAPI backend
├── streamlit_app.py        # Streamlit frontend
├── requirements.txt
├── .env.example
├── core/
│   ├── __init__.py
│   ├── model_loader.py     # Model + LoRA adapter loading (4-bit)
│   ├── analyzer.py         # Prompt engineering + JSON parsing
│   ├── schemas.py          # Pydantic output schemas
│   └── pdf_parser.py       # PDF text extraction (PyMuPDF)
└── README.md
```

</details>

---

### 📄 2. CV Builder Agent

> An **agentic, multi-iteration** CV generation platform with ensemble judges, hallucination guards, and semantic matching — powered by **LangGraph**.

#### Features

- 🤖 **Agentic CV Generation** — Multi-iteration write → judge → revise loop using LangGraph's StateGraph
- ⚖️ **Ensemble Judges** — ATS Judge, HR Judge, and Rule-based Judge score Clarity, Structure, Impact, and Skills Relevance
- 🛡️ **Hallucination Guard** — Ontology matching + semantic verification to prevent fabricated skills or metrics
- 🎨 **Multi-Template PDF Export** — Classic, Modern, Monochrome templates via ReportLab
- 🧠 **RAG & Semantic Matching** — FAISS + Sentence-Transformers for JD-to-skill semantic matching
- 🚀 **GPU Queue & Caching** — Serial GPU queue for concurrency + LRU cache for fast re-evaluations
- 📊 **Best-CV Tracking** — Automatically selects the highest-scoring CV across all iterations

#### LangGraph Pipeline

```mermaid
graph TD
    START(("▶ START")) --> W["✍️ Writer Node<br/>Generate N Candidates"]
    W --> J["⚖️ Judge Node<br/>Ensemble Scoring"]
    J --> R{"🔀 Router Node<br/>Adaptive Decision"}

    R -->|"revise / strategy change"| W
    R -->|"finalize ✅"| FINAL["📄 Final CV<br/>(Best Score)"]
    FINAL --> DONE(("⏹ END"))

    style START fill:#22c55e,stroke:#22c55e,color:#fff
    style W fill:#3b82f6,stroke:#3b82f6,color:#fff
    style J fill:#f59e0b,stroke:#f59e0b,color:#fff
    style R fill:#8b5cf6,stroke:#8b5cf6,color:#fff
    style FINAL fill:#06b6d4,stroke:#06b6d4,color:#fff
    style DONE fill:#ef4444,stroke:#ef4444,color:#fff
```

#### Detailed Pipeline Flow

```mermaid
sequenceDiagram
    participant U as 👤 User
    participant D as Streamlit Dashboard
    participant P as Pipeline Runner
    participant RAG as RAG Module (FAISS)
    participant W as Writer Node
    participant G as Hallucination Guard
    participant J as Ensemble Judges
    participant R as Router Node

    U->>D: Input Profile + JD
    D->>P: run_pipeline(profile, jd)
    P->>RAG: Extract JD context & keywords
    P->>P: Auto-categorize skills via LLM

    loop Iteration Loop (max N)
        P->>W: Generate candidate CVs
        W->>G: Check for hallucinations
        G-->>W: Pass/Fail + Issues
        W->>J: Score candidates
        J->>J: ATS + HR + Rule judges
        J->>R: Weighted ensemble score
        R->>R: Adaptive decision (revise/finalize)
        alt Score ≥ threshold OR max iterations
            R-->>P: Finalize (best CV)
        else Needs improvement
            R-->>W: Revise with new strategy
        end
    end

    P->>D: PipelineResult (CV + scores + trajectory)
    D->>U: Generated CV + PDF Export
```

#### Writer → Judge → Router Components

```mermaid
graph LR
    subgraph Writer["✍️ Writer Node"]
        W1["Prompt<br/>Engineering"] --> W2["Multi-Candidate<br/>Generation"]
        W2 --> W3["Content<br/>Checking"]
        W3 --> W4["Hallucination<br/>Guard"]
    end

    subgraph Judges["⚖️ Judge Node"]
        J1["ATS Judge<br/>(Keyword Match)"]
        J2["HR Judge<br/>(Readability)"]
        J3["Rule Judge<br/>(Format Check)"]
        J1 --> J4["Weighted<br/>Ensemble"]
        J2 --> J4
        J3 --> J4
    end

    subgraph Router["🔀 Router Node"]
        R1["Score Delta<br/>Analysis"]
        R2["Stagnation<br/>Detection"]
        R3["Strategy<br/>Selection"]
        R1 --> R3
        R2 --> R3
    end

    Writer --> Judges --> Router

    style Writer fill:#1e3a5f,stroke:#3b82f6,color:#e2e8f0
    style Judges fill:#3d2e0a,stroke:#f59e0b,color:#e2e8f0
    style Router fill:#2d1b4e,stroke:#8b5cf6,color:#e2e8f0
```

<details>
<summary>📁 Directory Structure</summary>

```text
cv-builder/
├── streamlit_app.py           # Main Streamlit frontend
├── requirements.txt
├── .env
├── cv_sessions.db             # Session cache database
├── cv_agent/                  # Core agentic backend
│   ├── __init__.py
│   ├── api.py                 # FastAPI integration
│   ├── cache.py               # LRU caching mechanism
│   ├── config.py              # Pipeline & system configuration
│   ├── content_checker.py     # Content quality verification
│   ├── gpu_queue.py           # Serial GPU queue management
│   ├── hallucination_guard.py # Hallucination detection logic
│   ├── judges.py              # AI evaluators (HR, ATS, Rule)
│   ├── memory.py              # Agent memory management
│   ├── model_manager.py       # LLM initialization and routing
│   ├── pdf_export.py          # PDF generation (ReportLab)
│   ├── pipeline.py            # LangGraph execution pipeline
│   ├── prompts.py             # System prompts for agents
│   ├── rag.py                 # FAISS retrieval-augmented generation
│   ├── routing.py             # Semantic routing & strategy logic
│   ├── schemas.py             # Pydantic data models
│   └── utils.py               # Utility functions
└── tests/                     # Unit & integration tests
    ├── test_api.py
    ├── test_cache.py
    ├── test_content_checker.py
    ├── test_hallucination_guard.py
    ├── test_hallucination_guard_edge.py
    ├── test_integration.py
    ├── test_routing.py
    └── test_schemas.py
```

</details>

---

### 🔍 3. Job Matching Agent

> An intelligent **LangGraph agent** that automates deep job research, skill matching, cover letter generation, and PDF report export — searching across **10+ global and MENA job boards**.

#### Features

- 🔎 **Multi-Board Search** — LinkedIn, Indeed, Glassdoor, Wuzzuf, Bayt, Forasna, NaukriGulf, WeWorkRemotely, RemoteOK, Wellfound
- 🧠 **Smart Resume Extraction** — LLM-powered skill, experience, and role extraction with quality-check retry loop
- 🌐 **Dynamic Site Discovery** — AI discovers niche job boards relevant to your profile
- ⚡ **Async Parallel Search** — Concurrent Tavily API searches across all platforms
- 🎯 **AI Skill Matching** — LLM-scored job matching (0-100) with gap analysis
- 🔬 **Deep Company Research** — Glassdoor reviews, culture insights for top matches
- ✉️ **Auto Cover Letters** — Tailored 3-paragraph cover letters for top 3 positions
- 📄 **PDF Report** — Full career research report with FPDF

#### LangGraph Agent Pipeline

```mermaid
graph TD
    START(("▶ START")) --> N1["📄 Extract Resume Info<br/>(LLM Parser)"]
    N1 --> N2{"✅ Quality Check"}

    N2 -->|"❌ Fail (retry ≤ 2)"| N1
    N2 -->|"✅ Pass"| N3["🌐 Discover<br/>Target Sites"]

    N3 --> N4["🧠 Generate<br/>Search Queries"]
    N4 --> N5["⚡ Search Jobs<br/>(Async Parallel)"]
    N5 --> N6["🎯 Match & Score<br/>(LLM Scoring)"]
    N6 --> N7["🔬 Deep Research<br/>Top 5 Jobs"]
    N7 --> N8["📊 Compile<br/>Research Report"]
    N8 --> N9["✉️ Generate<br/>Cover Letters"]
    N9 --> N10["📄 Generate<br/>PDF Report"]
    N10 --> DONE(("⏹ END"))

    style START fill:#22c55e,stroke:#22c55e,color:#fff
    style N2 fill:#f59e0b,stroke:#f59e0b,color:#fff
    style N5 fill:#3b82f6,stroke:#3b82f6,color:#fff
    style N6 fill:#8b5cf6,stroke:#8b5cf6,color:#fff
    style N10 fill:#06b6d4,stroke:#06b6d4,color:#fff
    style DONE fill:#ef4444,stroke:#ef4444,color:#fff
```

#### Detailed Workflow

```mermaid
sequenceDiagram
    participant U as 👤 User
    participant UI as Streamlit UI
    participant A as LangGraph Agent
    participant LLM as Groq LLM
    participant T as Tavily Search API
    participant PDF as FPDF Generator

    U->>UI: Upload Resume + Set Filters
    UI->>A: invoke(resume_text, location, date_filter)

    rect rgb(30, 40, 60)
        Note over A,LLM: Phase 1 — Resume Extraction
        A->>LLM: Extract skills, titles, level
        LLM->>A: Structured JSON
        A->>A: Quality check (retry if incomplete)
    end

    rect rgb(30, 50, 40)
        Note over A,T: Phase 2 — Job Discovery
        A->>LLM: Discover niche job boards
        A->>LLM: Generate 8-10 search queries
        A->>T: Async parallel search (all boards)
        T->>A: Raw job listings (deduplicated)
    end

    rect rgb(50, 30, 50)
        Note over A,LLM: Phase 3 — Matching & Research
        A->>LLM: Score each job (0-100 match)
        A->>T: Deep research top 5 companies
        A->>LLM: Compile career research report
    end

    rect rgb(40, 40, 30)
        Note over A,PDF: Phase 4 — Deliverables
        A->>LLM: Generate 3 cover letters
        A->>PDF: Build PDF report
        PDF->>A: career_report.pdf
    end

    A->>UI: Final state (report + jobs + letters)
    UI->>U: Interactive results dashboard
```

#### Agent State

```mermaid
classDiagram
    class AgentState {
        +str resume_text
        +List~str~ extracted_skills
        +List~str~ soft_skills
        +str experience_level
        +List~str~ job_titles
        +str education
        +List~str~ search_queries
        +List~dict~ raw_job_results
        +List~dict~ matched_jobs
        +List~dict~ deep_research_results
        +str research_report
        +List~dict~ cover_letters
        +str pdf_report_path
        +List~str~ target_sites
        +Optional~int~ date_filter_days
        +str location
        +List~str~ status_log
        +int iteration
    }
```

<details>
<summary>📁 Directory Structure</summary>

```text
jop-matching/
├── main.py               # Entry point
├── requirements.txt
├── .env.example
├── core/
│   ├── agent.py           # LangGraph graph builder + runner
│   ├── nodes.py           # All 9 pipeline nodes
│   ├── state.py           # AgentState TypedDict
│   └── utils.py           # LLM & Tavily helpers
└── ui/
    └── app.py             # Streamlit web interface (34KB)
```

</details>

---

### 🎙️ 4. AI Voice Interview System

> An intelligent, real-time **AI voice interview** platform with adaptive questioning, multi-phase evaluation, and comprehensive candidate assessment — powered by **LangGraph**, **WebSockets**, and local/cloud audio processing.

#### Features

- 📄 **Smart Document Parsing** — Extracts skills from JD and matches against uploaded CV (PDF/DOCX/TXT)
- 🧠 **Dynamic Orchestration** — LangGraph-driven interview flow that adapts based on candidate answers
- 🗣️ **Real-Time Voice I/O** — WebSocket streaming with barge-in detection and VAD
- 🎙️ **Flexible Audio Engines** — STT (Faster-Whisper / Deepgram) + TTS (Edge-TTS / ElevenLabs)
- 📊 **Multi-Dimensional Evaluation** — Technical proficiency, soft skills, confidence assessment
- 🧠 **Interview Memory** — Cross-turn memory tracking claims, contradictions, and depth
- 🔗 **3-Tier Question Sourcing** — Dataset (5000+ questions) → Web Search → LLM Generation
- 📋 **5-Phase Interview** — Opening → Technical → Behavioral → Situational → Closing

#### LangGraph Interview Engine

```mermaid
graph TD
    START(("▶ START")) --> SI["🔧 Session Init<br/>Parse CV + JD<br/>Build Coverage Map"]

    SI --> QG["❓ Question Gen<br/>3-Tier Sourcing"]

    QG --> AE["📝 Answer Eval<br/>⏸️ INTERRUPT<br/>(Wait for candidate)"]

    AE --> SC["📋 Summarize<br/>Context"]

    SC --> IR{"🔀 Interview Router"}

    IR -->|"Follow-up needed"| CF["🔗 Chain Follow-Up<br/>Memory-Aware"]
    IR -->|"New topic"| QG
    IR -->|"Phase complete"| AP["📈 Advance Phase"]
    IR -->|"Interview done"| RG["📊 Report Gen<br/>Final Assessment"]

    CF --> AE
    AP --> QG
    RG --> DONE(("⏹ END"))

    style START fill:#22c55e,stroke:#22c55e,color:#fff
    style SI fill:#3b82f6,stroke:#3b82f6,color:#fff
    style QG fill:#8b5cf6,stroke:#8b5cf6,color:#fff
    style AE fill:#f59e0b,stroke:#f59e0b,color:#fff
    style IR fill:#ec4899,stroke:#ec4899,color:#fff
    style CF fill:#06b6d4,stroke:#06b6d4,color:#fff
    style AP fill:#14b8a6,stroke:#14b8a6,color:#fff
    style RG fill:#f97316,stroke:#f97316,color:#fff
    style DONE fill:#ef4444,stroke:#ef4444,color:#fff
```

#### 3-Phase System Architecture

```mermaid
graph TB
    subgraph Phase1["💾 Phase 1 — Data Layer"]
        D1["ChromaDB<br/>Vector Store"]
        D2["CSV Knowledge Base<br/>5000+ Questions"]
        D3["Embeddings<br/>(Sentence-Transformers)"]
        D1 --- D2 --- D3
    end

    subgraph Phase2["🧠 Phase 2 — Orchestration"]
        O1["LangGraph<br/>StateGraph"]
        O2["7 Nodes<br/>(init, question, eval, ...)"]
        O3["Conditional<br/>Router"]
        O4["SQLite<br/>Checkpointer"]
        O1 --- O2 --- O3 --- O4
    end

    subgraph Phase3["🎙️ Phase 3 — Voice I/O & API"]
        V1["FastAPI<br/>Server"]
        V2["WebSocket<br/>Audio Stream"]
        V3["STT<br/>(Whisper/Deepgram)"]
        V4["TTS<br/>(Edge/ElevenLabs)"]
        V5["VAD<br/>(Voice Activity)"]
        V1 --- V2
        V2 --- V3
        V2 --- V4
        V2 --- V5
    end

    Phase1 --> Phase2 --> Phase3

    style Phase1 fill:#1a1f2e,stroke:#7c3aed,stroke-width:2px,color:#e2e8f0
    style Phase2 fill:#1a1f2e,stroke:#3b82f6,stroke-width:2px,color:#e2e8f0
    style Phase3 fill:#1a1f2e,stroke:#10b981,stroke-width:2px,color:#e2e8f0
```

#### Real-Time Voice Interaction Flow

```mermaid
sequenceDiagram
    participant C as 🎤 Candidate
    participant WS as WebSocket
    participant VAD as VAD Service
    participant STT as STT (Whisper)
    participant LG as LangGraph Engine
    participant MEM as Interview Memory
    participant EVAL as Evaluators
    participant TTS as TTS (Edge-TTS)

    C->>WS: Connect to /ws/{session_id}
    WS->>TTS: Generate question audio
    TTS->>WS: Audio (base64)
    WS->>C: 🔊 Play question

    loop Interview Loop
        C->>WS: 🎤 Audio chunks (binary)
        WS->>VAD: Detect speech activity
        VAD->>WS: End-of-speech signal
        WS->>STT: Transcribe audio buffer
        STT->>WS: Transcript text

        WS->>LG: Process answer
        LG->>MEM: Update memory (claims, depth)
        LG->>EVAL: Score answer
        EVAL->>EVAL: Technical + Soft Skills + Confidence
        EVAL->>LG: Evaluation result

        LG->>LG: Router decides next step
        alt Follow-up needed
            LG->>LG: Generate adaptive follow-up
        else New topic
            LG->>LG: Pick next question (3-tier)
        else Interview complete
            LG->>LG: Generate final report
        end

        LG->>TTS: Next question text
        TTS->>WS: Audio response
        WS->>C: 🔊 Next question + scores
    end
```

#### Intelligence Layer

```mermaid
graph LR
    subgraph Intelligence["🤖 Intelligence Layer"]
        AF["Adaptive<br/>Follow-Up"]
        SQ["Smart Question<br/>Generator"]
        IM["Interview<br/>Memory"]
        CM["Context<br/>Manager"]
    end

    subgraph Sources["📚 Question Sources"]
        S1["CSV Dataset<br/>(5000+ Q's)"]
        S2["Web Search<br/>(Dynamic)"]
        S3["LLM Generation<br/>(Contextual)"]
    end

    subgraph Memory["🧠 Memory Tracks"]
        M1["Claims Tracker"]
        M2["Contradiction<br/>Detector"]
        M3["Depth Tracker"]
        M4["Skill Coverage<br/>Map"]
    end

    Sources --> SQ
    SQ --> AF
    IM --> AF
    CM --> AF
    IM --> Memory

    style Intelligence fill:#1a1f2e,stroke:#8b5cf6,stroke-width:2px,color:#e2e8f0
    style Sources fill:#1a1f2e,stroke:#06b6d4,stroke-width:2px,color:#e2e8f0
    style Memory fill:#1a1f2e,stroke:#f59e0b,stroke-width:2px,color:#e2e8f0
```

#### API Endpoints

**REST API**

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serves the frontend web UI |
| `POST` | `/api/sessions` | Create session (JD text + CV file upload) |
| `GET` | `/api/sessions/{id}` | Get current session state |
| `POST` | `/api/sessions/{id}/text-answer` | Submit a text-based answer |

**WebSocket API**

| Endpoint | Direction | Format |
|----------|-----------|--------|
| `WS /ws/{session_id}` | Client → Server | Binary audio chunks or JSON text answers |
| `WS /ws/{session_id}` | Server → Client | Audio responses, transcriptions, scores, phase changes |

#### Knowledge Base (CSV Datasets)

| Dataset | Description |
|---------|-------------|
| `questions_master.csv` | 5000+ interview questions across domains |
| `domain_rubrics.csv` | Scoring rubrics per domain |
| `answer_calibration.csv` | Score calibration data |
| `question_chains.csv` | Follow-up question chains |
| `role_expectations.csv` | Role-specific requirements |
| `skill_hierarchy.csv` | Skill taxonomy & hierarchy |

<details>
<summary>📁 Directory Structure</summary>

```text
interview-system/
├── server.py                    # FastAPI entry point
├── config.py                    # Centralized configuration
├── requirements.txt
├── .env.example
│
├── core/                        # LangGraph Orchestration Engine
│   ├── interview_state.py       # TypedDict shared state
│   ├── llm_config.py            # LLM model configuration (Groq)
│   ├── graph.py                 # Graph assembly (7 nodes)
│   ├── nodes.py                 # session_init, question_gen, answer_eval, report_gen
│   └── router.py                # Conditional routing logic
│
├── data_layer/                  # Phase 1 — Data & Knowledge Layer
│   ├── phase1_data_layer.py     # ChromaDB vector store + Pandas engine
│   └── phase2_orchestration.py  # CLI testing tool
│
├── parsers/                     # Document Parsing
│   ├── cv_parser.py             # CV extraction (PDF/DOCX)
│   └── jd_parser.py             # JD requirements extraction
│
├── services/                    # Audio I/O Services
│   ├── stt_service.py           # Speech-to-Text (Whisper/Deepgram)
│   ├── tts_service.py           # Text-to-Speech (Edge/ElevenLabs)
│   └── vad_service.py           # Voice Activity Detection
│
├── evaluation/                  # Answer Evaluation
│   ├── confidence_evaluator.py  # Interview confidence scoring
│   ├── soft_skills_evaluator.py # Communication & clarity rating
│   └── analyze_code.py          # Code answer analysis
│
├── intelligence/                # Interview Intelligence
│   ├── adaptive_followup.py     # Memory-aware follow-up generation
│   ├── smart_question_gen.py    # 3-tier question sourcing
│   ├── interview_memory.py      # Cross-turn memory management
│   └── context_manager.py       # Transcript summarization
│
├── api/                         # API & Session Management
│   ├── session_manager.py       # WebSocket ↔ LangGraph bridge
│   └── admin_dashboard.py       # Streamlit monitoring dashboard
│
├── frontend/static/             # Web UI
│   └── index.html
│
├── data/                        # Knowledge Base (CSV)
│   ├── questions_master.csv     # 5000+ questions
│   ├── domain_rubrics.csv
│   ├── answer_calibration.csv
│   ├── question_chains.csv
│   ├── role_expectations.csv
│   └── skill_hierarchy.csv
│
└── tests/                       # Test Suite
    ├── test_api.py
    ├── test_bugs.py
    ├── test_full_flow.py
    └── check_q.py
```

</details>

---

## 🛠 Technology Stack

```mermaid
graph LR
    subgraph AI["🤖 AI & ML"]
        LG["LangGraph"]
        LC["LangChain"]
        PT["PyTorch"]
        HF["HuggingFace<br/>Transformers"]
        PEFT["PEFT / LoRA"]
        ST["Sentence<br/>Transformers"]
    end

    subgraph Backend["⚙️ Backend"]
        FA["FastAPI"]
        UV["Uvicorn"]
        WS["WebSockets"]
        PD["Pydantic"]
    end

    subgraph Frontend["🖥️ Frontend"]
        SL["Streamlit"]
        HTML["HTML/JS"]
    end

    subgraph Data["💾 Data & Search"]
        FAISS["FAISS"]
        CHROMA["ChromaDB"]
        TAVILY["Tavily API"]
        SQLITE["SQLite"]
    end

    subgraph Audio["🎙️ Audio"]
        WHISPER["Faster-Whisper"]
        EDGE["Edge-TTS"]
        VADX["VAD"]
    end

    subgraph Export["📄 Export"]
        RL["ReportLab"]
        FPDF["FPDF"]
        PYMUPDF["PyMuPDF"]
    end

    style AI fill:#1a1f2e,stroke:#7c3aed,color:#e2e8f0
    style Backend fill:#1a1f2e,stroke:#3b82f6,color:#e2e8f0
    style Frontend fill:#1a1f2e,stroke:#ef4444,color:#e2e8f0
    style Data fill:#1a1f2e,stroke:#f59e0b,color:#e2e8f0
    style Audio fill:#1a1f2e,stroke:#10b981,color:#e2e8f0
    style Export fill:#1a1f2e,stroke:#06b6d4,color:#e2e8f0
```

| Category | Technologies |
|----------|-------------|
| **LLM Orchestration** | LangGraph, LangChain, Groq API |
| **AI / ML Models** | PyTorch, HuggingFace Transformers, PEFT, Accelerate, bitsandbytes |
| **Embeddings & RAG** | FAISS, Sentence-Transformers, ChromaDB |
| **Backend** | FastAPI, Uvicorn, WebSockets, Pydantic |
| **Frontend** | Streamlit, HTML/JS |
| **Audio** | Faster-Whisper (STT), Edge-TTS (TTS), VAD, Deepgram, ElevenLabs |
| **Search** | Tavily API |
| **PDF & Parsing** | ReportLab, FPDF, PyMuPDF, python-docx |
| **Data** | Pandas, SQLite, Redis |
| **Testing** | pytest |

---

## ⚡ Quick Start

### Prerequisites

- **Python** 3.10+
- **GPU** (recommended): NVIDIA GPU with ≥6 GB VRAM + CUDA
- **ffmpeg** (for Interview System audio processing)

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/career-pilot.git
cd career-pilot
```

### 2. Choose a Sub-Project

Each sub-project is self-contained with its own `requirements.txt` and `.env`:

```bash
# Resume Analysis
cd resume-analysis
pip install -r requirements.txt
streamlit run streamlit_app.py

# CV Builder
cd cv-builder
pip install -r requirements.txt
streamlit run streamlit_app.py

# Job Matching
cd jop-matching
pip install -r requirements.txt
python main.py

# Interview System
cd interview-system
pip install -r requirements.txt
python server.py
```

### 3. Configure Environment

Each sub-project requires API keys in a `.env` file:

```env
# Common
GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_token

# Job Matching
TAVILY_API_KEY=your_tavily_api_key

# Interview System (Optional)
STT_PROVIDER=whisper          # whisper | deepgram
TTS_PROVIDER=edge             # edge | elevenlabs
```

> ⚠️ **Never commit your `.env` files!** Use `.env.example` as a template.

---

## 📁 Project Structure

```text
career-pilot/
│
├── 📊 resume-analysis/        # Fine-tuned Qwen2 resume scorer
│   ├── app.py                 # FastAPI backend
│   ├── streamlit_app.py       # Streamlit frontend
│   └── core/                  # Model loader, analyzer, schemas, parser
│
├── 📄 cv-builder/             # Agentic CV generation platform
│   ├── streamlit_app.py       # Dashboard UI
│   └── cv_agent/              # 15+ modules (pipeline, judges, guard, RAG...)
│
├── 🔍 jop-matching/           # Deep job research agent
│   ├── main.py                # Entry point
│   ├── core/                  # LangGraph agent (9 nodes)
│   └── ui/                    # Streamlit interface
│
├── 🎙️ interview-system/       # AI voice interview platform
│   ├── server.py              # FastAPI + WebSocket server
│   ├── core/                  # LangGraph engine (7 nodes)
│   ├── services/              # STT, TTS, VAD audio services
│   ├── intelligence/          # Adaptive follow-up, memory, question gen
│   ├── evaluation/            # Confidence, soft skills, code analysis
│   ├── data_layer/            # ChromaDB + knowledge base
│   ├── parsers/               # CV & JD document parsing
│   └── data/                  # 5000+ questions, rubrics, calibration
│
└── 📖 README.md               # This file
```

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** your changes: `git commit -m 'Add amazing feature'`
4. **Push** to the branch: `git push origin feature/amazing-feature`
5. **Open** a Pull Request

### Guidelines

- Each sub-project is independent — changes should be scoped accordingly
- Add tests for new features (see `tests/` in each sub-project)
- Follow existing code style and naming conventions
- Update the relevant sub-project README if adding new modules

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with ❤️ using LangGraph, PyTorch & FastAPI**

[⬆ Back to Top](#-career-pilot)

</div>


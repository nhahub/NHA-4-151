# 🔬 AI Resume Analyzer

> AI-powered resume analysis using a fine-tuned **Qwen2** model with **LoRA adapter** for structured CV evaluation.

Part of the **Career Pilot** suite.

---

## 🏗️ Architecture

```
┌─────────────────────┐       ┌──────────────────────────┐
│   Streamlit UI      │──────▶│   FastAPI Backend         │
│   (streamlit_app.py)│◀──────│   (app.py)               │
└─────────────────────┘       └────────┬─────────────────┘
                                       │
                              ┌────────▼─────────────────┐
                              │   Core Engine             │
                              │   ├── model_loader.py     │
                              │   ├── analyzer.py         │
                              │   ├── schemas.py          │
                              │   └── pdf_parser.py       │
                              └────────┬─────────────────┘
                                       │
                              ┌────────▼─────────────────┐
                              │   Qwen2 + LoRA Adapter    │
                              │   (HuggingFace)           │
                              └──────────────────────────┘
```

## 🤖 Model

| Component | Value |
|-----------|-------|
| **Base Model** | `OsamaHayba/qwen-ats-merged-stage1` |
| **LoRA Adapter** | `OsamaHayba/cv-analysis-final-stage2` |
| **Architecture** | Qwen2ForCausalLM |
| **Quantization** | 4-bit (bitsandbytes, NF4) |
| **Framework** | Transformers + PEFT |

## 📊 Output Schema

The model returns **exactly 10 fields**:

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

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd resume-analysis
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and set your HF_TOKEN
```

### 3. Run (Option A: Streamlit Only — Direct Inference)

```bash
streamlit run streamlit_app.py
```

### 3. Run (Option B: FastAPI Backend + Streamlit)

```bash
# Terminal 1: API
uvicorn app:app --host 0.0.0.0 --port 8000

# Terminal 2: UI
streamlit run streamlit_app.py
```

## 📁 Project Structure

```
resume-analysis/
├── app.py                  # FastAPI backend
├── streamlit_app.py        # Streamlit frontend
├── core/
│   ├── __init__.py
│   ├── model_loader.py     # Model + adapter loading
│   ├── analyzer.py         # Prompt engineering + JSON parsing
│   ├── schemas.py          # Pydantic output schemas
│   └── pdf_parser.py       # PDF text extraction
├── requirements.txt
├── .env                    # Your config (gitignored)
├── .env.example            # Template
└── README.md
```

## ⚙️ System Requirements

- **GPU (recommended)**: NVIDIA GPU with ≥6 GB VRAM + CUDA
- **CPU (fallback)**: Works but slow (~2-5 min per analysis)
- **RAM**: ≥16 GB recommended
- **Disk**: ~8 GB for model weights
- **Python**: 3.10+

## 📡 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/analyze` | Analyze resume text |
| `POST` | `/analyze/pdf` | Upload & analyze PDF |
| `GET` | `/health` | Health check + model status |

## 📄 License

Part of the Career Pilot project.

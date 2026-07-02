# JobAgent AI

JobAgent AI is an intelligent agent built using LangGraph and Streamlit that automates deep job research, skills matching, and personalized cover letter generation. 

It analyzes your resume, identifies skill gaps and top matches from numerous global and MENA-specific job boards (like Wuzzuf, Bayt, Forasna, LinkedIn, Indeed), and exports a comprehensive PDF report.

## Setup

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   Copy `.env.example` to `.env` and provide your API keys. You can also provide these directly in the UI.

## Usage

Run the Streamlit application using:
```bash
python main.py
```
Or via Streamlit CLI:
```bash
streamlit run ui/app.py
```

## Structure
- `core/`: LangGraph agent logic (state, nodes, graph assembly).
- `ui/`: Streamlit web interface.
- `main.py`: Entry point for the application.

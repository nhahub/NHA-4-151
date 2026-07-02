"""
JD Parser — Extract structured profile from Job Description text.

Uses Groq LLM to parse raw JD text into skills, responsibilities,
and requirements that drive the interview question selection.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict

from langchain_core.messages import HumanMessage, SystemMessage
from core.llm_config import get_eval_llm

log = logging.getLogger("phase3.jd_parser")

JD_PARSE_PROMPT = """You are an expert HR analyst. Extract structured information from this job description.

Return ONLY valid JSON in this exact format:
{
    "title": "job title",
    "seniority": "Junior|Mid|Senior|Lead|Staff|Principal",
    "required_skills": ["skill1", "skill2"],
    "responsibilities": ["resp1", "resp2"],
    "nice_to_haves": ["skill1", "skill2"],
    "domain": "Technical|Data Science|Product|DevOps|ML",
    "years_experience_min": 0,
    "key_technologies": ["tech1", "tech2"]
}

Rules:
- required_skills: extract ALL technical skills, tools, languages, frameworks mentioned as required
- nice_to_haves: skills mentioned as "preferred", "bonus", "nice to have"
- key_technologies: specific tools/platforms (e.g. "AWS", "Docker", "PostgreSQL")
- If seniority is unclear, infer from years of experience or responsibilities
- Respond ONLY with JSON, no other text."""


@dataclass
class JDProfile:
    title: str = "Unknown Role"
    seniority: str = "Mid"
    required_skills: list[str] = field(default_factory=list)
    responsibilities: list[str] = field(default_factory=list)
    nice_to_haves: list[str] = field(default_factory=list)
    domain: str = "Technical"
    years_experience_min: int = 0
    key_technologies: list[str] = field(default_factory=list)
    raw_text: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def all_skills(self) -> list[str]:
        """All skills combined (required + nice-to-haves + technologies)."""
        seen = set()
        result = []
        for s in self.required_skills + self.nice_to_haves + self.key_technologies:
            if s.lower() not in seen:
                seen.add(s.lower())
                result.append(s)
        return result


class JDParser:
    """Parse raw job description text into structured JDProfile."""

    def __init__(self):
        self.llm = get_eval_llm()

    def parse(self, jd_text: str) -> JDProfile:
        """Extract structured profile from job description text."""
        if not jd_text or len(jd_text.strip()) < 20:
            log.warning("JD text too short, using defaults")
            return JDProfile(raw_text=jd_text)

        log.info("Parsing JD (%d chars)...", len(jd_text))

        try:
            response = self.llm.invoke([
                SystemMessage(content=JD_PARSE_PROMPT),
                HumanMessage(content=f"Job Description:\n\n{jd_text[:4000]}"),
            ])
            raw = response.content.strip()

            # Strip markdown code fences robustly
            if raw.startswith("```"):
                # Remove opening fence (```json or ```)
                first_newline = raw.find("\n")
                if first_newline != -1:
                    raw = raw[first_newline + 1:]
                # Remove closing fence
                if raw.rstrip().endswith("```"):
                    raw = raw.rstrip()[:-3]
                raw = raw.strip()

            data = json.loads(raw, strict=False)

            profile = JDProfile(
                title=data.get("title", "Unknown Role"),
                seniority=data.get("seniority", "Mid"),
                required_skills=data.get("required_skills", []),
                responsibilities=data.get("responsibilities", []),
                nice_to_haves=data.get("nice_to_haves", []),
                domain=data.get("domain", "Technical"),
                years_experience_min=data.get("years_experience_min", 0),
                key_technologies=data.get("key_technologies", []),
                raw_text=jd_text,
            )

            log.info(
                "JD parsed: title='%s', seniority='%s', %d required skills, %d technologies",
                profile.title, profile.seniority,
                len(profile.required_skills), len(profile.key_technologies),
            )
            return profile

        except Exception as exc:
            log.error("JD parsing failed: %s — extracting basic info", exc)
            return self._fallback_parse(jd_text)

    def _fallback_parse(self, jd_text: str) -> JDProfile:
        """Simple keyword extraction when LLM fails."""
        text_lower = jd_text.lower()
        common_skills = [
            "Python", "Java", "JavaScript", "TypeScript", "C++", "Go", "Rust",
            "React", "Angular", "Vue", "Node.js", "Django", "Flask", "FastAPI",
            "SQL", "PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch",
            "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Terraform",
            "Git", "CI/CD", "REST API", "GraphQL", "Machine Learning",
            "TensorFlow", "PyTorch", "Pandas", "NumPy", "Scikit-learn",
            "Data Science", "Deep Learning", "Statistics", "EDA",
        ]
        found = [s for s in common_skills if s.lower() in text_lower]

        # ── Infer seniority ──────────────────────────────────────────────
        seniority = "Mid"
        if "senior" in text_lower or "sr." in text_lower:
            seniority = "Senior"
        elif "junior" in text_lower or "jr." in text_lower:
            seniority = "Junior"
        elif "lead" in text_lower or "principal" in text_lower:
            seniority = "Lead"
        elif "staff" in text_lower:
            seniority = "Staff"

        # ── Infer job title from JD text (first non-empty line or common patterns) ──
        title = "Unknown Role"
        # Try to extract from the first few lines
        for line in jd_text.splitlines():
            stripped = line.strip()
            if stripped and len(stripped) < 80:
                # Likely a title line
                title = stripped
                break

        # Override with known role patterns (order matters — most specific first)
        role_patterns = [
            ("data scientist",       "Data Scientist"),
            ("machine learning engineer", "Machine Learning Engineer"),
            ("ml engineer",          "Machine Learning Engineer"),
            ("data engineer",        "Data Engineer"),
            ("data analyst",         "Data Analyst"),
            ("devops engineer",      "DevOps Engineer"),
            ("site reliability",     "Site Reliability Engineer"),
            ("product manager",      "Product Manager"),
            ("frontend engineer",    "Frontend Engineer"),
            ("backend engineer",     "Backend Engineer"),
            ("full stack",           "Full Stack Engineer"),
            ("fullstack",            "Full Stack Engineer"),
            ("software engineer",    "Software Engineer"),
            ("software developer",   "Software Developer"),
        ]
        for keyword, role_name in role_patterns:
            if keyword in text_lower:
                title = f"{seniority} {role_name}" if seniority != "Mid" else role_name
                break

        # ── Infer domain ─────────────────────────────────────────────────
        domain = "Technical"
        if any(kw in text_lower for kw in [
            "data scientist", "machine learning", "deep learning",
            "data science", "ml model", "neural network", "statistics",
            "pandas", "numpy", "scikit", "tensorflow", "pytorch", "eda",
        ]):
            domain = "Data Science"
        elif any(kw in text_lower for kw in [
            "devops", "site reliability", "infrastructure", "kubernetes",
            "terraform", "ci/cd", "deployment pipeline",
        ]):
            domain = "DevOps"
        elif any(kw in text_lower for kw in [
            "product manager", "product owner", "roadmap", "stakeholder",
        ]):
            domain = "Product"

        log.info(
            "_fallback_parse: title='%s', seniority='%s', domain='%s', %d skills found",
            title, seniority, domain, len(found),
        )

        return JDProfile(
            title=title,
            seniority=seniority,
            required_skills=found,
            domain=domain,
            raw_text=jd_text,
        )

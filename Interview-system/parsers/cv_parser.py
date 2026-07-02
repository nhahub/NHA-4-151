"""
CV Parser — Extract structured profile from uploaded CV/resume files.

Supports PDF and DOCX formats. Uses text extraction + Groq LLM
to parse into skills, experience, and education.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict

from langchain_core.messages import HumanMessage, SystemMessage
from core.llm_config import get_eval_llm

log = logging.getLogger("phase3.cv_parser")

CV_PARSE_PROMPT = """You are an expert recruiter analyzing a candidate's resume/CV.

Extract structured information and return ONLY valid JSON:
{
    "name": "candidate name",
    "skills": ["skill1", "skill2"],
    "experience": [
        {"role": "Job Title", "company": "Company", "duration": "2 years", "description": "key achievements"}
    ],
    "education": [
        {"degree": "BS Computer Science", "institution": "MIT", "year": "2020"}
    ],
    "projects": ["project description 1"],
    "years_of_experience": 5,
    "certifications": ["cert1"],
    "claimed_strengths": ["strength1"]
}

Rules:
- skills: extract ALL technical skills, tools, languages, frameworks mentioned
- experience: list work experience with key responsibilities/achievements
- claimed_strengths: things the candidate claims to be good at
- Respond ONLY with JSON, no other text."""


@dataclass
class CVProfile:
    name: str = "Candidate"
    skills: list[str] = field(default_factory=list)
    experience: list[dict] = field(default_factory=list)
    education: list[dict] = field(default_factory=list)
    projects: list[str] = field(default_factory=list)
    years_of_experience: int = 0
    certifications: list[str] = field(default_factory=list)
    claimed_strengths: list[str] = field(default_factory=list)
    raw_text: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def all_claims(self) -> list[str]:
        """All verifiable claims from the CV."""
        claims = list(self.skills)
        for exp in self.experience:
            if exp.get("description"):
                claims.append(exp["description"])
        claims.extend(self.claimed_strengths)
        return claims


class CVParser:
    """Parse CV files (PDF/DOCX) into structured CVProfile."""

    def __init__(self):
        self.llm = get_eval_llm()

    def parse_file(self, file_bytes: bytes, filename: str) -> CVProfile:
        """Parse a CV file into structured profile."""
        ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

        if ext == "pdf":
            text = self._extract_pdf(file_bytes)
        elif ext in ("docx", "doc"):
            text = self._extract_docx(file_bytes)
        elif ext == "txt":
            text = file_bytes.decode("utf-8", errors="replace")
        else:
            log.warning("Unknown file type '%s', treating as text", ext)
            text = file_bytes.decode("utf-8", errors="replace")

        if not text or len(text.strip()) < 20:
            log.warning("Could not extract text from CV")
            return CVProfile(raw_text="[extraction failed]")

        return self.parse_text(text)

    def parse_text(self, cv_text: str) -> CVProfile:
        """Parse raw CV text into structured profile."""
        log.info("Parsing CV (%d chars)...", len(cv_text))

        try:
            response = self.llm.invoke([
                SystemMessage(content=CV_PARSE_PROMPT),
                HumanMessage(content=f"Resume/CV:\n\n{cv_text[:4000]}"),
            ])
            raw = response.content.strip()

            # Strip markdown code fences robustly
            if raw.startswith("```"):
                first_newline = raw.find("\n")
                if first_newline != -1:
                    raw = raw[first_newline + 1:]
                if raw.rstrip().endswith("```"):
                    raw = raw.rstrip()[:-3]
                raw = raw.strip()

            data = json.loads(raw, strict=False)

            profile = CVProfile(
                name=data.get("name", "Candidate"),
                skills=data.get("skills", []),
                experience=data.get("experience", []),
                education=data.get("education", []),
                projects=data.get("projects", []),
                years_of_experience=data.get("years_of_experience", 0),
                certifications=data.get("certifications", []),
                claimed_strengths=data.get("claimed_strengths", []),
                raw_text=cv_text,
            )

            log.info(
                "CV parsed: name='%s', %d skills, %d years exp, %d positions",
                profile.name, len(profile.skills),
                profile.years_of_experience, len(profile.experience),
            )
            return profile

        except Exception as exc:
            log.error("CV parsing failed: %s", exc)
            return CVProfile(raw_text=cv_text)

    def _extract_pdf(self, file_bytes: bytes) -> str:
        """Extract text from PDF bytes."""
        try:
            import fitz  # pymupdf
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            text_parts = []
            for page in doc:
                text_parts.append(page.get_text())
            doc.close()
            return "\n".join(text_parts)
        except ImportError:
            log.error("pymupdf not installed — cannot parse PDF")
            return ""
        except Exception as exc:
            log.error("PDF extraction failed: %s", exc)
            return ""

    def _extract_docx(self, file_bytes: bytes) -> str:
        """Extract text from DOCX bytes."""
        try:
            import io
            from docx import Document
            doc = Document(io.BytesIO(file_bytes))
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except ImportError:
            log.error("python-docx not installed — cannot parse DOCX")
            return ""
        except Exception as exc:
            log.error("DOCX extraction failed: %s", exc)
            return ""

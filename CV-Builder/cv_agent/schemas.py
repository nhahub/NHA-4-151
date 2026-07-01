"""
cv_agent.schemas
================
All Pydantic models and dataclass schemas used across the pipeline.

Includes profile-hash–based caching for ``authorised_tokens()`` to avoid
redundant token extraction during hallucination validation.
"""

from __future__ import annotations

import json
import operator
import re
from hashlib import md5
from typing import (
    Annotated, Any, Dict, List, Literal, Optional, Tuple, Union,
)

from pydantic import BaseModel, Field, field_validator

from cv_agent.config import PipelineConfig  # noqa: F401 — re-exported for convenience


# ==============================================================================
# authorised_tokens cache (keyed by profile_hash)
# ==============================================================================

_authorised_tokens_cache: Dict[str, set] = {}


# ==============================================================================
# JUDGE OUTPUT
# ==============================================================================

class JudgeOutput(BaseModel):
    clarity_score:           int = Field(ge=0, le=100, default=0)
    structure_score:         int = Field(ge=0, le=100, default=0)
    impact_score:            int = Field(ge=0, le=100, default=0)
    skills_relevance_score:  int = Field(ge=0, le=100, default=0)
    ats_readiness_score:     int = Field(ge=0, le=100, default=0)
    overall_score:           int = Field(ge=0, le=100, default=0)
    strengths:               List[str] = Field(default_factory=list)
    weaknesses:              List[str] = Field(default_factory=list)
    improvement_suggestions: List[str] = Field(default_factory=list)
    rewrite_suggestions:     List[str] = Field(default_factory=list)

    @field_validator("strengths", "weaknesses", "improvement_suggestions", "rewrite_suggestions", mode="before")
    @classmethod
    def _coerce_list_of_strings(cls, v):
        if not isinstance(v, list):
            return v
        res = []
        for item in v:
            if isinstance(item, dict):
                res.append(json.dumps(item))
            elif not isinstance(item, str):
                res.append(str(item))
            else:
                res.append(item)
        return res

    @classmethod
    def fallback(cls, reason: str = "parse failure") -> "JudgeOutput":
        return cls(weaknesses=[f"Judge output could not be parsed: {reason}"])

    def average_score(self) -> float:
        return (
            self.clarity_score + self.structure_score + self.impact_score
            + self.skills_relevance_score + self.ats_readiness_score
        ) / 5

    def lowest_metric(self) -> str:
        m = {
            "Clarity":          self.clarity_score,
            "Structure":        self.structure_score,
            "Impact":           self.impact_score,
            "Skills relevance": self.skills_relevance_score,
            "ATS readiness":    self.ats_readiness_score,
        }
        return min(m, key=m.get)  # type: ignore[arg-type]

    def passes(self, threshold: int) -> bool:
        return all(
            v >= threshold for v in [
                self.clarity_score, self.structure_score, self.impact_score,
                self.skills_relevance_score, self.ats_readiness_score,
            ]
        ) and self.overall_score >= threshold


# ==============================================================================
# ENSEMBLE RESULT
# ==============================================================================

class EnsembleResult(BaseModel):
    model_config = {"arbitrary_types_allowed": True, "revalidate_instances": "never"}
    ats_output:  JudgeOutput
    hr_output:   JudgeOutput
    rule_output: JudgeOutput
    weighted:    JudgeOutput
    cv_text:     str = ""


# ==============================================================================
# JD CONTEXT
# ==============================================================================

class JDContext(BaseModel):
    keywords:         List[str] = Field(default_factory=list)
    requirements:     List[str] = Field(default_factory=list)
    canonical_skills: List[str] = Field(default_factory=list)

    def inject_block(self) -> str:
        if not self.keywords:
            return ""
        return (
            f"\n\nJD KEYWORDS: {', '.join(self.keywords[:20])}\n"
            f"Requirements: {'; '.join(self.requirements[:8])}\n"
            f"Canonical skills: {', '.join(self.canonical_skills[:15])}"
        )


# ==============================================================================
# USER PROFILE
# ==============================================================================

class UserProfile(BaseModel):
    model_config = {"arbitrary_types_allowed": True, "revalidate_instances": "never"}
    full_name:          str = ""
    target_role:        str = ""
    target_industry:    str = ""
    years_experience:   str = ""
    summary:            str = ""
    tone:               str = "professional"
    email:              str = ""
    phone:              str = ""
    linkedin:           str = ""
    education:          List[str] = Field(default_factory=list)
    skills:             List[str] = Field(default_factory=list)
    skill_categories:   Optional[Dict[str, List[str]]] = Field(default=None, description="Auto-categorized skills by LLM. Keys are category names, values are skill lists. e.g. {'Programming': ['Python', 'SQL']}")
    experiences:        List[str] = Field(default_factory=list)
    projects:           List[str] = Field(default_factory=list)
    achievements:       List[str] = Field(default_factory=list)
    certifications:     List[str] = Field(default_factory=list)
    needs_more_info:    bool = False
    followup_questions: List[str] = Field(default_factory=list)

    # Regex for entries that look like non-employment (self-taught, student, etc.)
    _NON_EMPLOYMENT_RE = re.compile(
        r'(?i)(?:self[- ]?taught|self[- ]?employed|freelance|freelancer|personal\s+project|'
        r'side\s+project|student|open\s+source|volunteer|academic|capstone|'
        r'thesis|bootcamp|hackathon|course\s+project|research\s+assistant|'
        r'teaching\s+assistant)',
    )

    @property
    def has_no_experience(self) -> bool:
        """True when the user has no real professional work experience.

        Detects self-taught, freelance, student, and similar non-employment
        entries so they are treated as projects rather than work history.
        """
        # Check years_experience first
        yrs = self.years_experience.strip().lower()
        no_yrs = yrs in ("", "0", "0 years", "none", "no experience", "no", "n/a")

        # If experiences list is empty, clearly no experience
        if not self.experiences:
            return True

        # Check if ALL experience entries are non-employment
        all_non_employment = all(
            self._NON_EMPLOYMENT_RE.search(exp) for exp in self.experiences
        )
        if all_non_employment:
            return True

        # If years field explicitly says 0 / none and no convincing entries
        return False

    def is_complete(self) -> bool:
        has_content = bool(self.experiences or self.projects)
        return bool(self.full_name and self.target_role and self.skills and has_content)

    def to_context(self) -> str:
        def ul(items: List[str]) -> str:
            return "\n".join(f"  - {i}" for i in items) if items else "  (none)"

        # Build skills block — categorized when skill_categories provided
        if self.skill_categories:
            skills_lines = []
            for category, cat_skills in self.skill_categories.items():
                skills_lines.append(f"  {category}: {', '.join(cat_skills)}")
            skills_block = f"Skills (CATEGORIZED — preserve these exact categories in the CV):\n" + "\n".join(skills_lines)
        else:
            skills_block = f"Skills:\n{ul(self.skills)}"

        # Build experience or projects block depending on user profile
        if self.has_no_experience:
            # Merge any self-taught "experience" entries into the projects list
            merged_projects = list(self.projects)
            for exp in self.experiences:
                if exp.strip() and exp.strip() not in merged_projects:
                    merged_projects.append(exp.strip())
            exp_block = f"Projects:\n{ul(merged_projects)}"
        else:
            exp_block = f"Work Experience:\n{ul(self.experiences)}"
            if self.projects:
                exp_block += f"\n\nProjects:\n{ul(self.projects)}"

        return (
            f"Name: {self.full_name}\nTarget Role: {self.target_role}\n"
            f"Email: {self.email or 'Not specified'}\nPhone: {self.phone or 'Not specified'}\n"
            f"LinkedIn: {self.linkedin or 'Not specified'}\n"
            f"Industry: {self.target_industry or 'Not specified'}\n"
            f"Experience: {self.years_experience or 'Not specified'}\nTone: {self.tone}\n\n"
            f"Education:\n{ul(self.education)}\n\n{skills_block}\n\n"
            f"{exp_block}\n\nAchievements:\n{ul(self.achievements)}\n\n"
            f"Certifications:\n{ul(self.certifications)}\n\nSummary Notes: {self.summary or 'None'}"
        )

    def profile_hash(self) -> str:
        return md5(self.model_dump_json().encode()).hexdigest()[:16]

    def authorised_tokens(self) -> set:
        """
        Build the set of tokens the HallucinationGuard considers legitimate.

        Cached per profile_hash — identical profiles skip recomputation.

        FIX M-3 (v7): Two-level tokenisation.
        1. Phrase level — 1–4-word n-grams from raw text.
        2. Token level — single alphanumeric tokens as fallback.
        Both sets are unioned for maximum fuzzy-match surface area.
        """
        key = self.profile_hash()
        cached = _authorised_tokens_cache.get(key)
        if cached is not None:
            return cached

        # Collect all skill tokens including from skill_categories
        all_skill_tokens = list(self.skills)
        if self.skill_categories:
            for cat_skills in self.skill_categories.values():
                all_skill_tokens.extend(cat_skills)

        raw = " ".join([
            self.full_name, self.target_role, self.target_industry, self.summary,
            self.email, self.phone, self.linkedin,
            " ".join(all_skill_tokens), " ".join(self.experiences),
            " ".join(self.projects), " ".join(self.achievements),
            " ".join(self.education), " ".join(self.certifications),
        ])

        # Level 1: individual alphanumeric tokens (original behaviour)
        single_tokens = {t.lower() for t in re.findall(r"[a-zA-Z0-9&+#.]{2,}", raw)}

        # Level 2: sliding n-gram phrases (1–4 words)
        words = re.findall(r"[a-zA-Z0-9][a-zA-Z0-9&+#.'\\\-]*", raw)
        phrase_tokens: set = set()
        for n in range(1, 5):
            for i in range(len(words) - n + 1):
                phrase = " ".join(words[i:i + n]).lower()
                if len(phrase) >= 2:
                    phrase_tokens.add(phrase)

        result = single_tokens | phrase_tokens
        _authorised_tokens_cache[key] = result
        return result


# ==============================================================================
# GUARD RESULT
# ==============================================================================

class GuardResult(BaseModel):
    passed: bool
    issues: List[str] = Field(default_factory=list)


# ==============================================================================
# CONTENT QUALITY MODELS
# ==============================================================================

class ContentIssue(BaseModel):
    """A single content-quality issue detected in the CV text."""
    category: Literal["repetition", "spelling", "grammar", "weak_verb", "tense", "vague"]
    severity: Literal["error", "warning", "info"] = "warning"
    text:       str = ""              # The problematic text
    suggestion: str = ""              # The suggested fix
    location:   str = ""              # e.g. "Professional Summary", "bullet 3"


class ContentCheckResult(BaseModel):
    """Aggregated content-quality report for a finished CV."""
    issues:          List[ContentIssue]   = Field(default_factory=list)
    repeated_words:  Dict[str, int]       = Field(default_factory=dict)  # word → count
    overall_quality: Literal["excellent", "good", "needs_work", "poor"] = "good"
    summary:         str = ""


# ==============================================================================
# LOOP DECISION
# ==============================================================================

class LoopDecision(BaseModel):
    action: Literal["revise", "restructure", "keywords", "regenerate", "finalize"]
    reason: str


# ==============================================================================
# ITERATION RECORD
# ==============================================================================

class IterationRecord(BaseModel):
    iteration:  int
    cv_text:    str
    ensemble:   EnsembleResult
    decision:   LoopDecision
    strategy:   str
    duration_s: float = 0.0


# ==============================================================================
# LANGGRAPH STATE
# ==============================================================================

class CVAgentState(BaseModel):
    model_config = {"arbitrary_types_allowed": True, "revalidate_instances": "never"}

    user_profile:      Any       = Field(default_factory=UserProfile)
    job_description:   str       = ""
    parsed_resume:     str       = ""
    jd_context:        JDContext = Field(default_factory=JDContext)
    session_id:        str       = ""

    current_cv:        str       = ""
    candidates:        List[str] = Field(default_factory=list)

    last_ensemble:     Optional[EnsembleResult] = None
    last_scores:       Optional[JudgeOutput]    = None

    iteration:         int       = 0
    max_iterations:    int       = 5
    score_threshold:   int       = 82
    score_history:     List[float] = Field(default_factory=list)
    delta_history:     List[float] = Field(default_factory=list)
    stagnation_count:  int       = 0
    current_strategy:  str       = "revise"
    loop_complete:     bool      = False
    intake_complete:   bool      = False
    recurring_issues:  List[str] = Field(default_factory=list)

    final_cv:          str       = ""
    output_pdf_path:   str       = ""

    # Best-CV tracking: always points to the highest-scoring CV across all iterations.
    # If a later iteration produces a worse score, the pipeline reverts to this.
    best_cv_text:      str       = ""
    best_cv_score:     float     = -1.0

    # Inline content feedback forwarded from writer_node to the next iteration's prompt.
    # Populated by generate_candidates after apply_content_fixes().
    content_feedback:  str       = ""

    error:             str       = ""
    node_errors:       Annotated[List[str],             operator.add] = Field(default_factory=list)
    iteration_history: Annotated[List[IterationRecord], operator.add] = Field(default_factory=list)
    has_hallucination:    bool      = False
    hallucination_issues: List[str] = Field(default_factory=list)
    # Per-candidate guard results from generate_candidates, keyed by candidate index.
    # Passed to select_best_candidate to avoid re-running the guard a second time.
    guard_results:        Dict[int, Any] = Field(default_factory=dict)


# ==============================================================================
# PIPELINE RESULT
# ==============================================================================

class PipelineResult(BaseModel):
    """Serialisable result — safe to JSON-encode and return over HTTP."""
    model_config = {"arbitrary_types_allowed": True}

    session_id:        str
    candidate_name:    str
    target_role:       str
    total_iterations:  int
    final_cv:          str
    final_scores:      Optional[JudgeOutput]
    score_trajectory:  List[float]
    node_errors:       List[str]
    jd_keywords:       List[str]
    total_latency_ms:  int = 0
    cv_template:       str = "classic"
    content_check:     Optional["ContentCheckResult"] = None

    def to_report_dict(self) -> dict:
        return {
            "session_id":       self.session_id,
            "candidate":        self.candidate_name,
            "target_role":      self.target_role,
            "total_iterations": self.total_iterations,
            "final_scores":     self.final_scores.model_dump() if self.final_scores else {},
            "score_trajectory": self.score_trajectory,
            "jd_keywords":      self.jd_keywords[:20],
            "node_errors":      self.node_errors,
            "total_latency_ms": self.total_latency_ms,
        }

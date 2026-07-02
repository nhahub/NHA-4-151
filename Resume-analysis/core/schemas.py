"""
Pydantic schemas for the Resume Analysis output.

Enforces EXACTLY the 10 required keys from the fine-tuned model.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List


class ResumeAnalysis(BaseModel):
    """
    Structured resume analysis output.

    All score fields are integers 0-100.
    List fields contain actionable text items.
    """

    clarity_score: int = Field(
        ..., ge=0, le=100,
        description="How clear and readable the resume is (0-100)"
    )
    structure_score: int = Field(
        ..., ge=0, le=100,
        description="How well-organized the resume sections are (0-100)"
    )
    impact_score: int = Field(
        ..., ge=0, le=100,
        description="How effectively achievements and impact are communicated (0-100)"
    )
    skills_relevance_score: int = Field(
        ..., ge=0, le=100,
        description="How relevant the listed skills are to the target role (0-100)"
    )
    ats_readiness_score: int = Field(
        ..., ge=0, le=100,
        description="How well-optimized the resume is for ATS systems (0-100)"
    )
    overall_score: float = Field(
        ..., ge=0, le=100,
        description="Weighted overall score across all dimensions"
    )
    strengths: List[str] = Field(
        default_factory=list,
        description="List of identified resume strengths"
    )
    weaknesses: List[str] = Field(
        default_factory=list,
        description="List of identified resume weaknesses"
    )
    improvement_suggestions: List[str] = Field(
        default_factory=list,
        description="Actionable suggestions for improving the resume"
    )
    rewrite_suggestions: List[str] = Field(
        default_factory=list,
        description="Specific rewrite recommendations for sections or bullet points"
    )

    @field_validator(
        "clarity_score", "structure_score", "impact_score",
        "skills_relevance_score", "ats_readiness_score",
        mode="before"
    )
    @classmethod
    def clamp_scores(cls, v):
        """Clamp scores to 0-100 range."""
        if isinstance(v, (int, float)):
            return max(0, min(100, int(v)))
        return v

    @field_validator("overall_score", mode="before")
    @classmethod
    def clamp_overall(cls, v):
        """Clamp overall score to 0-100 range."""
        if isinstance(v, (int, float)):
            return max(0.0, min(100.0, float(v)))
        return v

    @field_validator(
        "strengths", "weaknesses",
        "improvement_suggestions", "rewrite_suggestions",
        mode="before"
    )
    @classmethod
    def ensure_list(cls, v):
        """Convert string to single-item list if needed."""
        if isinstance(v, str):
            return [v]
        if v is None:
            return []
        return v


class AnalyzeRequest(BaseModel):
    """Request body for the /analyze endpoint."""

    resume_text: str = Field(
        ..., min_length=50,
        description="The full text content of the resume"
    )
    job_description: str = Field(
        default="",
        description="Optional job description for contextual scoring"
    )


class AnalyzeResponse(BaseModel):
    """Response body wrapping the analysis result."""

    success: bool = True
    analysis: ResumeAnalysis
    model_info: str = ""
    processing_time_seconds: float = 0.0

"""
Resume Analyzer — Prompt engineering + structured JSON parsing.

Builds chat messages using Qwen's <|im_start|>/<|im_end|> template,
sends resume text to the model, and parses the structured JSON output
into a validated ResumeAnalysis schema.
"""

import json
import re
import logging
import os
import torch
from typing import Optional

from .schemas import ResumeAnalysis
from .model_loader import get_model

logger = logging.getLogger(__name__)

MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "2048"))

# ── System Prompt ──────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert Resume/CV Analyst and ATS (Applicant Tracking System) specialist.

Your task is to analyze the provided resume thoroughly and return a structured JSON evaluation.

You MUST return EXACTLY the following keys in your JSON response:
- clarity_score (integer 0-100): How clear and readable the resume is
- structure_score (integer 0-100): How well-organized the resume sections are
- impact_score (integer 0-100): How effectively achievements and impact are communicated
- skills_relevance_score (integer 0-100): How relevant the listed skills are to the target role
- ats_readiness_score (integer 0-100): How well-optimized the resume is for ATS systems
- overall_score (float 0-100): Weighted overall score
- strengths (list of strings): Key strengths identified in the resume
- weaknesses (list of strings): Key weaknesses or gaps identified
- improvement_suggestions (list of strings): Actionable suggestions for improvement
- rewrite_suggestions (list of strings): Specific rewrite recommendations for sections or bullet points

IMPORTANT RULES:
1. Return ONLY valid JSON — no markdown, no extra text before or after the JSON.
2. All scores must be integers between 0 and 100.
3. Each list must contain at least 2 items.
4. Be specific and actionable in your suggestions.
5. Consider ATS compatibility, keyword optimization, and industry standards."""


def _build_user_prompt(resume_text: str, job_description: str = "") -> str:
    """Build the user message with resume text and optional job description."""
    prompt = f"Analyze the following resume:\n\n{resume_text}"

    if job_description.strip():
        prompt += f"\n\n--- TARGET JOB DESCRIPTION ---\n{job_description}"

    prompt += "\n\nReturn your analysis as a JSON object with exactly the required keys."
    return prompt


def _extract_json_from_text(text: str) -> Optional[dict]:
    """
    Extract JSON from model output using multiple strategies.

    Strategy 1: Direct JSON parse
    Strategy 2: Find JSON block between curly braces
    Strategy 3: Find JSON in markdown code blocks
    Strategy 4: Regex extraction of individual fields
    """

    # Clean up common issues
    text = text.strip()

    # Strategy 1: Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Find the outermost { ... } block
    try:
        # Find the first { and last }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            json_str = text[start:end + 1]
            return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    # Strategy 3: Markdown code block
    try:
        pattern = r"```(?:json)?\s*\n?(.*?)\n?```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return json.loads(match.group(1).strip())
    except (json.JSONDecodeError, AttributeError):
        pass

    # Strategy 4: Regex extraction of individual fields
    try:
        result = {}
        score_fields = [
            "clarity_score", "structure_score", "impact_score",
            "skills_relevance_score", "ats_readiness_score", "overall_score"
        ]
        list_fields = [
            "strengths", "weaknesses",
            "improvement_suggestions", "rewrite_suggestions"
        ]

        for field in score_fields:
            match = re.search(rf'"{field}"\s*:\s*(\d+(?:\.\d+)?)', text)
            if match:
                result[field] = float(match.group(1))

        for field in list_fields:
            match = re.search(rf'"{field}"\s*:\s*\[(.*?)\]', text, re.DOTALL)
            if match:
                items = re.findall(r'"([^"]+)"', match.group(1))
                result[field] = items

        if len(result) >= 6:  # At least the score fields
            return result
    except Exception:
        pass

    logger.error("All JSON extraction strategies failed.")
    return None


def _build_default_analysis() -> dict:
    """Return a default analysis when model output cannot be parsed."""
    return {
        "clarity_score": 50,
        "structure_score": 50,
        "impact_score": 50,
        "skills_relevance_score": 50,
        "ats_readiness_score": 50,
        "overall_score": 50.0,
        "strengths": ["Resume was submitted for analysis"],
        "weaknesses": ["Unable to fully parse model output — using defaults"],
        "improvement_suggestions": ["Please try again or provide a clearer resume format"],
        "rewrite_suggestions": ["Consider restructuring with standard sections: Summary, Experience, Skills, Education"],
    }


def analyze_resume(
    resume_text: str,
    job_description: str = "",
) -> ResumeAnalysis:
    """
    Analyze a resume using the fine-tuned Qwen2 model.

    Args:
        resume_text: Full text content of the resume.
        job_description: Optional job description for contextual scoring.

    Returns:
        ResumeAnalysis: Validated analysis with all 10 required fields.
    """
    model, tokenizer, device = get_model()

    # ── Build Messages ─────────────────────────────────────
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_prompt(resume_text, job_description)},
    ]

    # ── Tokenize with Chat Template ────────────────────────
    try:
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception as e:
        logger.warning(f"Chat template failed: {e}. Using manual template.")
        input_text = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{_build_user_prompt(resume_text, job_description)}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    inputs = tokenizer(input_text, return_tensors="pt")

    # Move to correct device
    if device == "cuda":
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    # ── Generate ───────────────────────────────────────────
    logger.info("Generating analysis...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # ── Decode ─────────────────────────────────────────────
    # Only decode the new tokens (skip the input)
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_length:]
    raw_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    logger.info(f"Raw model output length: {len(raw_output)} chars")
    logger.debug(f"Raw output: {raw_output[:500]}...")

    # ── Parse JSON ─────────────────────────────────────────
    parsed = _extract_json_from_text(raw_output)

    if parsed is None:
        logger.warning("Could not parse model output. Using defaults.")
        parsed = _build_default_analysis()

    # ── Validate with Pydantic ─────────────────────────────
    # Fill missing fields with defaults
    defaults = _build_default_analysis()
    for key in defaults:
        if key not in parsed:
            parsed[key] = defaults[key]

    try:
        analysis = ResumeAnalysis(**parsed)
    except Exception as e:
        logger.error(f"Pydantic validation failed: {e}. Using full defaults.")
        analysis = ResumeAnalysis(**_build_default_analysis())

    return analysis

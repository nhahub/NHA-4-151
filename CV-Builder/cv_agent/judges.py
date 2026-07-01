"""
cv_agent.judges
===============
RuleJudge (deterministic heuristic), LLM judge runner, and ensemble scorer.
"""

from __future__ import annotations

import re
from hashlib import md5
from copy import copy
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from cv_agent.cache import LRUCache, get_cache
from cv_agent.config import PipelineConfig, logger
from cv_agent.gpu_queue import _gpu_queue
from cv_agent.model_manager import ModelManager, chat
from cv_agent.prompts import ATS_JUDGE_SYSTEM, HR_JUDGE_SYSTEM
from cv_agent.schemas import EnsembleResult, JDContext, JudgeOutput, UserProfile
from cv_agent.utils import parse_json_robust


class RuleJudge:
    """Deterministic, heuristic-based judge — no LLM required. FIX M-1 (v7)."""

    REQUIRED_SECTIONS = ["experience", "education", "skills", "summary"]
    WEAK_VERBS = frozenset({
        "worked on", "helped with", "was responsible for",
        "involved in", "assisted with", "participated in",
    })

    def evaluate(self, cv_text: str, profile: Optional[UserProfile] = None) -> JudgeOutput:
        text_lower = cv_text.lower()
        required = list(self.REQUIRED_SECTIONS)
        # For fresh grads / no-experience, require "projects" instead of "experience"
        if profile and profile.has_no_experience:
            if "experience" in required:
                required.remove("experience")
            if "projects" not in required:
                required.append("projects")
        missing = [s for s in required if s not in text_lower]
        weak_count = sum(1 for v in self.WEAK_VERBS if v in text_lower)
        metric_count = len(re.findall(r'\d+%|\$[\d,]+|\d+x\b|\d+ years?', cv_text))
        word_count = len(cv_text.split())
        bullet_count = cv_text.count("\n-") + cv_text.count("\n•") + cv_text.count("\n*")
        has_contact = bool(re.search(
            r'\[email\]|\[phone\]|email@|linkedin\.com|@[a-z]+\.[a-z]|\+\d[\d\s\-]{7,}',
            cv_text, re.I,
        ))
        has_links = bool(re.search(r'github\.com|linkedin\.com|portfolio|http', cv_text, re.I))

        section_score = max(0, 100 - len(missing) * 25)
        bullet_score = min(80, bullet_count * 6)
        contact_bonus = 12 if has_contact else 0
        link_bonus = 8 if has_links else 0
        structure_score = min(100, bullet_score + contact_bonus + link_bonus)
        metric_score = min(100, int((metric_count / 6) * 100)) if metric_count < 6 else 100
        verb_score = max(0, 100 - weak_count * 20)

        if 400 <= word_count <= 900:
            length_score = 100
        elif word_count < 300:
            length_score = 40
        elif word_count < 400:
            length_score = 70
        elif word_count <= 1100:
            length_score = 80
        else:
            length_score = 55

        ats_score = int(length_score * 0.7 + (15 if has_contact else 0) + (15 if has_links else 0))
        ats_score = min(100, ats_score)

        overall = int(
            section_score * 0.25 + structure_score * 0.20
            + metric_score * 0.20 + verb_score * 0.15 + ats_score * 0.20
        )

        weaknesses: List[str] = []
        suggestions: List[str] = []
        if missing:
            weaknesses.append(f"Missing required sections: {', '.join(missing)}")
            suggestions.append(f"Add the following sections: {', '.join(missing)}")
        if weak_count > 0:
            weaknesses.append(f"Found {weak_count} weak action verb phrase(s)")
            suggestions.append("Replace passive phrases with strong action verbs")
        if metric_count < 3:
            weaknesses.append(f"Only {metric_count} quantified metric(s) detected — aim for 4+")
            suggestions.append("Add quantified achievements: %, $, time saved, team size")
        if word_count < 400:
            weaknesses.append(f"CV is too brief ({word_count} words) — aim for 400–900")
        if word_count > 1100:
            weaknesses.append(f"CV may be too long ({word_count} words) — aim for 400–900")
        if not has_contact:
            weaknesses.append("No contact information detected")
            suggestions.append("Add email, phone, and LinkedIn URL to the header")

        return JudgeOutput(
            clarity_score=section_score, structure_score=structure_score,
            impact_score=metric_score, skills_relevance_score=verb_score,
            ats_readiness_score=ats_score, overall_score=overall,
            strengths=(
                (["Quantified achievements present"] if metric_count >= 3 else [])
                + (["Good structure with bullet points"] if bullet_count >= 8 else [])
            ),
            weaknesses=weaknesses, improvement_suggestions=suggestions,
        )


_rule_judge = RuleJudge()


def _run_llm_judge(
    pipe: Any, system: str, cv_text: str,
    jd_context: JDContext, cfg: PipelineConfig,
) -> JudgeOutput:
    """Run a single LLM judge via GPU queue with retry and robust JSON parsing."""
    user_prompt = f"Analyze this CV:\n\n{cv_text}{jd_context.inject_block()}"
    for attempt in range(cfg.judge_max_retries):
        try:
            prompt = user_prompt
            if attempt > 0:
                prompt = (
                    "IMPORTANT: Your previous output was not valid JSON. "
                    f"Return ONLY a JSON object, nothing else.\n\n{user_prompt}"
                )
            raw = _gpu_queue.submit(chat, pipe, system, prompt)
            d = parse_json_robust(raw)
            if not d:
                raise ValueError("empty parse result")
            return JudgeOutput(**d)
        except (ValidationError, ValueError, Exception) as e:
            logger.warning("Judge attempt %d/%d failed: %s", attempt + 1, cfg.judge_max_retries, e)
            if attempt == cfg.judge_max_retries - 1:
                return JudgeOutput.fallback(str(e))
    return JudgeOutput.fallback("max retries exceeded")


def run_ensemble(
    cv_text: str, jd_context: JDContext, cfg: PipelineConfig,
    cache_key_prefix: str = "",
    profile: Optional[UserProfile] = None,
) -> EnsembleResult:
    """Run ATS → HR → Rule judge sequentially through the GPU queue."""
    cache = get_cache(cfg)
    cv_hash = md5(cv_text.encode()).hexdigest()[:12]
    ens_key = LRUCache.make_key(
        cache_key_prefix or cv_hash, cv_hash, 0, "ensemble", namespace="judge"
    )
    cached_ens = cache.get(ens_key)
    if cached_ens is not None:
        logger.debug("Cache HIT — ensemble for cv_hash=%s", cv_hash)
        return cached_ens

    mm = ModelManager.get_instance()
    logger.info("Ensemble [1/3] ATS judge")
    ats_out = _run_llm_judge(mm.ats_judge_pipe(cfg), ATS_JUDGE_SYSTEM, cv_text, jd_context, cfg)
    logger.info("Ensemble [2/3] HR judge")
    hr_out = _run_llm_judge(mm.hr_judge_pipe(cfg), HR_JUDGE_SYSTEM, cv_text, jd_context, cfg)
    logger.info("Ensemble [3/3] Rule judge (CPU)")
    rule_out = _rule_judge.evaluate(cv_text, profile=profile)

    def wavg(a: int, h: int, r: int) -> int:
        return int(a * cfg.ats_weight + h * cfg.hr_weight + r * cfg.rule_weight)

    weighted = JudgeOutput(
        clarity_score=wavg(ats_out.clarity_score, hr_out.clarity_score, rule_out.clarity_score),
        structure_score=wavg(ats_out.structure_score, hr_out.structure_score, rule_out.structure_score),
        impact_score=wavg(ats_out.impact_score, hr_out.impact_score, rule_out.impact_score),
        skills_relevance_score=wavg(ats_out.skills_relevance_score, hr_out.skills_relevance_score, rule_out.skills_relevance_score),
        ats_readiness_score=wavg(ats_out.ats_readiness_score, hr_out.ats_readiness_score, rule_out.ats_readiness_score),
        overall_score=wavg(ats_out.overall_score, hr_out.overall_score, rule_out.overall_score),
        strengths=list(dict.fromkeys(ats_out.strengths + hr_out.strengths + rule_out.strengths))[:5],
        weaknesses=list(dict.fromkeys(ats_out.weaknesses + hr_out.weaknesses + rule_out.weaknesses))[:8],
        improvement_suggestions=list(dict.fromkeys(
            ats_out.improvement_suggestions + hr_out.improvement_suggestions + rule_out.improvement_suggestions
        ))[:6],
        rewrite_suggestions=list(dict.fromkeys(ats_out.rewrite_suggestions + hr_out.rewrite_suggestions))[:4],
    )

    result = EnsembleResult(
        ats_output=ats_out, hr_output=hr_out,
        rule_output=rule_out, weighted=weighted, cv_text=cv_text,
    )
    cache.set(ens_key, result)
    return result

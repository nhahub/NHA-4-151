"""
Soft Skills Evaluator — Communication, clarity, confidence analysis.

Evaluates candidate's soft skills from their answer text:
  - Communication clarity (structure, coherence)
  - Confidence level (hedging words, assertiveness)
  - Structured thinking (logical flow, STAR method)
  - Conciseness (verbosity vs. clarity ratio)
  - Professionalism (vocabulary, tone)
  - Filler words (um, uh, like, you know)

Each answer gets a soft_skills_scores dict appended to the transcript.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field, asdict

from langchain_core.messages import HumanMessage, SystemMessage
from core.llm_config import get_eval_llm

log = logging.getLogger("phase3.soft_skills")

# ── Filler / hedging word patterns ────────────────────────────────────────────
FILLER_WORDS = [
    r"\bum\b", r"\buh\b", r"\blike\b", r"\byou know\b",
    r"\bbasically\b", r"\bactually\b", r"\bliterally\b",
    r"\bso yeah\b", r"\bi mean\b", r"\bkind of\b", r"\bsort of\b",
]

# Words that signal genuine uncertainty → penalise confidence
STRONG_HEDGING = [
    r"\bi guess\b", r"\bnot sure\b", r"\bi have no idea\b", r"\bno idea\b",
    r"\bi don't know\b", r"\bidunno\b", r"\bmaybe\b", r"\bi'm unsure\b",
    r"\bnot really sure\b",
]

# Words that reflect valid technical caution — do NOT penalise
WEAK_HEDGING = [
    r"\bprobably\b", r"\bperhaps\b", r"\bi think\b", r"\bi believe\b",
    r"\bit depends\b", r"\bcould be\b", r"\bmight be\b",
    r"\btypically\b", r"\busually\b", r"\bgenerally\b",
    r"\bin most cases\b", r"\bin general\b",
]

# Legacy alias so other modules that import HEDGING_WORDS still work
HEDGING_WORDS = STRONG_HEDGING + WEAK_HEDGING

SOFT_SKILLS_PROMPT = """You are an expert communication coach evaluating a job interview answer.

Evaluate the candidate's SOFT SKILLS (not technical accuracy) for this answer.

Question: {question}
Answer: {answer}

IMPORTANT: Do NOT penalise candidates for using scientifically cautious language such as
"typically", "usually", "in most cases", "it depends on the context", "generally".
These demonstrate technical maturity and appropriate epistemic humility — they are STRENGTHS.
Only flag GENUINE uncertainty markers like: "I guess", "not sure", "I have no idea", "I don't know".

Focus on: answer consistency, technical clarity, logical reasoning quality, and depth of explanation.

Score each dimension 1-10 and provide a brief note:

Return ONLY valid JSON:
{{
    "communication_clarity": {{"score": 7, "note": "Clear structure but could be more concise"}},
    "structured_thinking": {{"score": 8, "note": "Good use of examples and logical flow"}},
    "confidence": {{"score": 7, "note": "Speaks with appropriate technical caution"}},
    "professionalism": {{"score": 8, "note": "Professional vocabulary and tone"}},
    "reasoning_quality": {{"score": 8, "note": "Well-reasoned, logical argument with concrete examples"}},
    "overall_soft_score": 7.5,
    "communication_feedback": "Brief 1-line feedback on communication style"
}}"""


@dataclass
class SoftSkillsResult:
    """Result of soft skills evaluation for a single answer."""
    communication_clarity: float = 5.0
    structured_thinking: float = 5.0
    confidence: float = 5.0
    professionalism: float = 5.0
    conciseness: float = 5.0
    reasoning_quality: float = 5.0
    filler_word_count: int = 0
    hedging_word_count: int = 0     # legacy: total weak + strong
    strong_hedging_count: int = 0   # genuine uncertainty markers
    weak_hedging_count: int = 0     # valid technical qualifiers
    word_count: int = 0
    overall_soft_score: float = 5.0
    communication_feedback: str = ""
    notes: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class SoftSkillsEvaluator:
    """
    Evaluates candidate's communication and soft skills.
    Combines rule-based text analysis with LLM-based assessment.
    """

    def __init__(self):
        self.llm = get_eval_llm()

    def evaluate(self, question_text: str, answer_text: str) -> SoftSkillsResult:
        """
        Full soft skills evaluation of a single answer.
        Combines rule-based metrics + LLM assessment.
        """
        # ── Rule-based analysis ───────────────────────────────────────────
        result = self._analyze_text(answer_text)

        # ── LLM-based assessment ──────────────────────────────────────────
        llm_result = self._llm_evaluate(question_text, answer_text)
        if llm_result:
            result.communication_clarity = llm_result.get("communication_clarity", {}).get("score", result.communication_clarity)
            result.structured_thinking = llm_result.get("structured_thinking", {}).get("score", result.structured_thinking)
            result.confidence = llm_result.get("confidence", {}).get("score", result.confidence)
            result.professionalism = llm_result.get("professionalism", {}).get("score", result.professionalism)
            result.reasoning_quality = llm_result.get("reasoning_quality", {}).get("score", result.reasoning_quality)
            result.overall_soft_score = llm_result.get("overall_soft_score", result.overall_soft_score)
            result.communication_feedback = llm_result.get("communication_feedback", "")
            result.notes = {
                k: v.get("note", "") for k, v in llm_result.items()
                if isinstance(v, dict) and "note" in v
            }

            # Only penalise STRONG hedging (genuine uncertainty), not weak technical qualifiers
            if result.strong_hedging_count > 3:
                result.confidence = max(1, result.confidence - 2)
            elif result.strong_hedging_count > 1:
                result.confidence = max(1, result.confidence - 1)
            # weak_hedging: no penalty — technical maturity

            # Adjust conciseness based on word count
            if result.word_count > 300:
                result.conciseness = max(3, result.conciseness - 2)
            elif result.word_count < 20:
                result.conciseness = max(2, result.conciseness - 3)

        log.info(
            "Soft skills: clarity=%.0f, structure=%.0f, confidence=%.0f, "
            "fillers=%d, hedging=%d, words=%d, overall=%.1f",
            result.communication_clarity, result.structured_thinking,
            result.confidence, result.filler_word_count,
            result.hedging_word_count, result.word_count,
            result.overall_soft_score,
        )
        return result

    def _analyze_text(self, text: str) -> SoftSkillsResult:
        """Rule-based text analysis for filler words, hedging, length."""
        text_lower = text.lower()
        words = text.split()
        word_count = len(words)

        filler_count = sum(
            len(re.findall(pattern, text_lower)) for pattern in FILLER_WORDS
        )
        # Count strong and weak hedging separately
        strong_count = sum(
            len(re.findall(pattern, text_lower)) for pattern in STRONG_HEDGING
        )
        weak_count = sum(
            len(re.findall(pattern, text_lower)) for pattern in WEAK_HEDGING
        )
        hedging_count = strong_count + weak_count  # legacy total

        # Basic conciseness score
        if 50 <= word_count <= 200:
            conciseness = 8.0
        elif 30 <= word_count <= 300:
            conciseness = 6.0
        elif word_count < 30:
            conciseness = 3.0  # too short
        else:
            conciseness = 4.0  # too verbose

        # Confidence: only based on STRONG hedging ratio (genuine uncertainty)
        strong_ratio = strong_count / max(1, word_count) * 100
        if strong_ratio < 1:
            confidence = 8.0
        elif strong_ratio < 3:
            confidence = 6.0
        else:
            confidence = 4.0

        return SoftSkillsResult(
            word_count=word_count,
            filler_word_count=filler_count,
            hedging_word_count=hedging_count,
            strong_hedging_count=strong_count,
            weak_hedging_count=weak_count,
            conciseness=conciseness,
            confidence=confidence,
        )

    def _llm_evaluate(self, question: str, answer: str) -> dict | None:
        """LLM-based soft skills assessment."""
        try:
            prompt = SOFT_SKILLS_PROMPT.format(
                question=question[:500],
                answer=answer[:1000],
            )
            response = self.llm.invoke([
                SystemMessage(content="You are an expert communication and soft skills evaluator."),
                HumanMessage(content=prompt),
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
            return json.loads(raw, strict=False)
        except Exception as exc:
            log.warning("LLM soft skills eval failed: %s", exc)
            return None

    def aggregate_session(self, soft_scores_list: list[dict]) -> dict:
        """
        Aggregate soft skills scores across all answers in the session.
        Returns a summary for the final report.
        """
        if not soft_scores_list:
            return {"overall_soft_score": 0, "feedback": "No answers evaluated"}

        dims = [
            "communication_clarity", "structured_thinking",
            "confidence", "professionalism", "conciseness", "reasoning_quality",
        ]
        avg = {}
        for dim in dims:
            values = [s.get(dim, 5) for s in soft_scores_list if dim in s]
            avg[dim] = round(sum(values) / len(values), 1) if values else 5.0

        total_fillers = sum(s.get("filler_word_count", 0) for s in soft_scores_list)
        total_strong_hedging = sum(s.get("strong_hedging_count", 0) for s in soft_scores_list)
        total_weak_hedging = sum(s.get("weak_hedging_count", 0) for s in soft_scores_list)
        total_hedging = sum(s.get("hedging_word_count", 0) for s in soft_scores_list)
        avg_words = sum(s.get("word_count", 0) for s in soft_scores_list) / len(soft_scores_list)

        overall = round(sum(avg.values()) / len(avg), 1)

        # Build feedback (only flag genuine uncertainty, not scientific caution)
        feedback_parts = []
        if avg.get("communication_clarity", 5) >= 7:
            feedback_parts.append("Clear communicator")
        elif avg.get("communication_clarity", 5) <= 4:
            feedback_parts.append("Needs improvement in clarity")

        if avg.get("confidence", 5) >= 7:
            feedback_parts.append("Projects confidence")
        elif avg.get("confidence", 5) <= 4:
            feedback_parts.append("Shows genuine uncertainty (excessive 'I guess', 'not sure')")

        if avg.get("reasoning_quality", 5) >= 7:
            feedback_parts.append("Strong reasoning and logical depth")
        elif avg.get("reasoning_quality", 5) <= 4:
            feedback_parts.append("Reasoning could be more structured and concrete")

        if total_fillers > 10:
            feedback_parts.append(f"High filler word usage ({total_fillers} total)")
        if total_strong_hedging > 8:
            feedback_parts.append(
                f"Frequent genuine uncertainty language ({total_strong_hedging} instances of 'I guess', 'not sure', etc.)"
            )

        if avg.get("structured_thinking", 5) >= 7:
            feedback_parts.append("Well-structured responses")

        return {
            "scores": avg,
            "overall_soft_score": overall,
            "total_filler_words": total_fillers,
            "total_hedging_words": total_hedging,
            "total_strong_hedging": total_strong_hedging,
            "total_weak_hedging": total_weak_hedging,
            "avg_answer_length": round(avg_words),
            "feedback": ". ".join(feedback_parts) if feedback_parts else "Average communication skills",
        }

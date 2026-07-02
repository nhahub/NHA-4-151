"""
Interview Memory — Cross-Turn Memory for Realistic Interviewer Behavior.

Tracks everything a senior interviewer would remember across the full interview:
  - Factual claims the candidate makes (tools, projects, metrics, years)
  - Per-skill depth signals (not just a counter, but quality-based)
  - Detected contradictions between answers
  - Answer patterns (vague vs specific, theoretical vs practical)
  - Topics the candidate brings up organically (for natural follow-ups)
  - Strongest and weakest moments for the final report

This module is the "brain" that makes the interviewer feel human.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from core.llm_config import get_eval_llm

log = logging.getLogger("memory.core")


# ─────────────────────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FactualClaim:
    """A concrete, verifiable claim the candidate made."""
    claim: str              # "Used XGBoost with learning_rate=0.01"
    skill: str              # "Machine Learning"
    turn: int               # When they said it
    source_quote: str       # The exact sentence from the answer
    category: str = "tool"  # "tool", "metric", "project", "experience", "technique"


@dataclass
class Contradiction:
    """A detected inconsistency between two answers."""
    claim_a: str            # What they said first
    claim_b: str            # What they said later
    turn_a: int             # When they said claim_a
    turn_b: int             # When they said claim_b
    skill: str              # Related skill
    severity: str = "minor" # "minor", "moderate", "major"
    resolved: bool = False  # Set to True if candidate explains it
    resolution: str = ""    # How they resolved it (if they did)


@dataclass
class DepthSignal:
    """Quality-based depth assessment for a skill (not just a counter)."""
    skill: str
    questions_asked: int = 0
    answers_with_specifics: int = 0    # Answers that included real details
    answers_vague: int = 0             # Answers that were generic/textbook
    max_difficulty_reached: str = "basic" # "basic", "intermediate", "advanced", "expert"
    has_real_project: bool = False      # Did they describe a real project?
    has_edge_case: bool = False         # Did they handle an edge case well?
    has_tradeoff_reasoning: bool = False # Did they reason about tradeoffs?
    confidence_in_assessment: float = 0.0  # 0-1: how confident we are in our assessment

    @property
    def is_sufficiently_deep(self) -> bool:
        """Determine if we have enough signal on this skill."""
        # Need at least 2 questions AND at least one of: real project, edge case, or tradeoff
        if self.questions_asked < 2:
            return False
        # If they've given specific answers with real experience
        if self.has_real_project and self.answers_with_specifics >= 2:
            return True
        # If they've shown tradeoff reasoning AND handled edge cases
        if self.has_tradeoff_reasoning and self.has_edge_case:
            return True
        # If they failed badly (all vague), 2 questions is enough signal
        if self.answers_vague >= 2 and self.answers_with_specifics == 0:
            return True
        # If we've asked 3+ questions, that's enough regardless
        if self.questions_asked >= 3:
            return True
        return False


@dataclass
class AnswerPatterns:
    """Tracks recurring patterns across all answers."""
    total_answers: int = 0
    vague_count: int = 0         # "I have experience with X"
    specific_count: int = 0      # "I used X with config Y on project Z"
    theoretical_count: int = 0   # Textbook answers
    practical_count: int = 0     # Real experience answers
    avg_specificity: float = 5.0
    avg_realism: float = 5.0
    buzzword_heavy: bool = False  # Uses lots of buzzwords without substance


# ─────────────────────────────────────────────────────────────────────────────
# LLM Prompts
# ─────────────────────────────────────────────────────────────────────────────

CLAIM_EXTRACTION_PROMPT = """You are analyzing a candidate's interview answer to extract factual claims.

**Question:** {question}
**Answer:** {answer}
**Skill being tested:** {skill}

Extract ALL concrete, verifiable claims the candidate makes. Focus on:
- Specific tools/technologies they say they used (e.g., "XGBoost", "Docker", "PostgreSQL")
- Specific metrics or numbers they mention (e.g., "reduced latency by 40%", "processed 1M rows")
- Project descriptions (e.g., "built a recommendation engine for e-commerce")
- Experience claims (e.g., "3 years with Python", "led a team of 5")
- Techniques or approaches they describe (e.g., "used SMOTE for class imbalance")

Also assess:
- specificity_score (1-10): How specific and detailed is this answer? 
  * 1-3 = vague/generic ("I have experience with ML")
  * 4-6 = moderate ("I've used sklearn for classification tasks")
  * 7-10 = highly specific ("Used XGBoost with max_depth=6, ran 5-fold CV, achieved 0.92 AUC")
- realism_score (1-10): Does this sound like real experience or textbook knowledge?
  * 1-3 = clearly rehearsed/textbook
  * 4-6 = plausible but lacking details
  * 7-10 = clearly from real experience (mentions failures, surprises, specific decisions)
- is_theoretical (bool): Is this a textbook definition rather than practical experience?
- has_real_project (bool): Did they describe a real project they worked on?
- has_tradeoff (bool): Did they reason about tradeoffs or just give one approach?

Return ONLY valid JSON:
{{
    "claims": [
        {{"claim": "string", "category": "tool|metric|project|experience|technique", "source_quote": "exact quote"}}
    ],
    "specificity_score": 1-10,
    "realism_score": 1-10,
    "is_theoretical": true/false,
    "has_real_project": true/false,
    "has_tradeoff": true/false,
    "topics_mentioned": ["topic1", "topic2"]
}}"""


CONTRADICTION_CHECK_PROMPT = """You are a senior interviewer checking for contradictions in a candidate's answers.

**Previous claims the candidate has made:**
{previous_claims}

**New answer just given:**
Question: {question}
Answer: {answer}

Check if the new answer CONTRADICTS any previous claim. A contradiction means:
- They said they used tool X before, but now say they never use it
- They claimed Y years of experience before, but now the timeline doesn't match
- They described approach A before, but now describe doing the opposite
- They said they worked on project P before, but now describe a conflicting detail

DO NOT flag as contradictions:
- Adding new information that doesn't conflict
- Discussing different projects with different approaches
- Clarifying or expanding on a previous point
- Using different but compatible tools in different contexts

Return ONLY valid JSON:
{{
    "has_contradiction": true/false,
    "contradictions": [
        {{
            "claim_a": "what they said before",
            "claim_b": "what they said now",
            "turn_a": <turn number>,
            "severity": "minor|moderate|major",
            "explanation": "why this is contradictory"
        }}
    ]
}}"""


# ─────────────────────────────────────────────────────────────────────────────
# Interview Memory Class
# ─────────────────────────────────────────────────────────────────────────────

class InterviewMemory:
    """
    Cross-turn memory system for the interview.

    This is the core intelligence that makes the interviewer feel human.
    It tracks claims, detects contradictions, evaluates depth quality,
    and provides rich context for follow-up generation.
    """

    def __init__(self):
        self.factual_claims: list[FactualClaim] = []
        self.contradictions: list[Contradiction] = []
        self.depth_signals: dict[str, DepthSignal] = {}
        self.answer_patterns = AnswerPatterns()
        self.topics_mentioned: dict[str, int] = {}  # topic → turn first mentioned
        self.strongest_moments: list[dict] = []      # {turn, skill, quote, score}
        self.weakest_moments: list[dict] = []        # {turn, skill, quote, score}
        self.per_answer_specificity: list[dict] = [] # {turn, skill, specificity, realism}

    def record_turn(
        self,
        question: str,
        answer: str,
        skill: str,
        turn: int,
        score: int,
        strengths: list[str] | None = None,
        gaps: list[str] | None = None,
    ) -> dict:
        """
        Record a complete turn (question + answer + evaluation).
        Extracts claims, checks contradictions, updates depth signals.

        Returns a dict of extracted info for logging/debugging.
        """
        result = {}

        # ── 1. Extract claims and assess specificity ──────────────────────
        extraction = self._extract_claims(question, answer, skill, turn)
        result["extraction"] = extraction

        if extraction:
            # Store factual claims
            for claim_data in extraction.get("claims", []):
                fc = FactualClaim(
                    claim=claim_data.get("claim", ""),
                    skill=skill,
                    turn=turn,
                    source_quote=claim_data.get("source_quote", ""),
                    category=claim_data.get("category", "tool"),
                )
                self.factual_claims.append(fc)

            # Store specificity/realism scores
            specificity = extraction.get("specificity_score", 5)
            realism = extraction.get("realism_score", 5)
            self.per_answer_specificity.append({
                "turn": turn,
                "skill": skill,
                "specificity": specificity,
                "realism": realism,
            })

            # Update answer patterns
            self.answer_patterns.total_answers += 1
            if specificity <= 3:
                self.answer_patterns.vague_count += 1
            elif specificity >= 7:
                self.answer_patterns.specific_count += 1
            if extraction.get("is_theoretical"):
                self.answer_patterns.theoretical_count += 1
            else:
                self.answer_patterns.practical_count += 1

            # Recalculate averages
            if self.per_answer_specificity:
                self.answer_patterns.avg_specificity = (
                    sum(a["specificity"] for a in self.per_answer_specificity)
                    / len(self.per_answer_specificity)
                )
                self.answer_patterns.avg_realism = (
                    sum(a["realism"] for a in self.per_answer_specificity)
                    / len(self.per_answer_specificity)
                )

            # Track organically mentioned topics
            for topic in extraction.get("topics_mentioned", []):
                if topic.lower() not in {t.lower() for t in self.topics_mentioned}:
                    self.topics_mentioned[topic] = turn

            # ── 2. Update depth signals ───────────────────────────────────
            ds = self.depth_signals.get(skill, DepthSignal(skill=skill))
            ds.questions_asked += 1
            if specificity >= 6:
                ds.answers_with_specifics += 1
            if specificity <= 3:
                ds.answers_vague += 1
            if extraction.get("has_real_project"):
                ds.has_real_project = True
            if extraction.get("has_tradeoff"):
                ds.has_tradeoff_reasoning = True
            # Update difficulty based on question type
            if score >= 7 and ds.questions_asked >= 2:
                ds.max_difficulty_reached = "advanced"
            elif score >= 5:
                ds.max_difficulty_reached = max(
                    ds.max_difficulty_reached, "intermediate",
                    key=["basic", "intermediate", "advanced", "expert"].index,
                )
            # Confidence in our assessment
            ds.confidence_in_assessment = min(1.0, ds.questions_asked * 0.25 +
                                               (0.2 if ds.has_real_project else 0) +
                                               (0.1 if ds.has_tradeoff_reasoning else 0))
            self.depth_signals[skill] = ds

        # ── 3. Track strongest/weakest moments ────────────────────────────
        moment = {
            "turn": turn,
            "skill": skill,
            "score": score,
            "strengths": strengths or [],
            "gaps": gaps or [],
            "answer_preview": answer[:200],
        }
        if score >= 8:
            self.strongest_moments.append(moment)
        elif score <= 4:
            self.weakest_moments.append(moment)

        # ── 4. Check for contradictions ───────────────────────────────────
        if self.factual_claims and turn > 1:
            new_contradictions = self._check_contradictions(question, answer, turn)
            if new_contradictions:
                self.contradictions.extend(new_contradictions)
                result["contradictions_found"] = len(new_contradictions)
                log.info(
                    "Memory: %d contradiction(s) detected at turn %d",
                    len(new_contradictions), turn,
                )

        log.info(
            "Memory: turn %d recorded — %d claims total, %d contradictions, "
            "specificity=%.1f, depth_signal(%s)=%s",
            turn, len(self.factual_claims), len(self.contradictions),
            self.answer_patterns.avg_specificity, skill,
            "sufficient" if self.depth_signals.get(skill, DepthSignal(skill=skill)).is_sufficiently_deep else "needs_more",
        )

        return result

    def get_depth_signal(self, skill: str) -> DepthSignal:
        """Get depth signal for a skill (creates default if not tracked yet)."""
        return self.depth_signals.get(skill, DepthSignal(skill=skill))

    def get_unresolved_contradictions(self) -> list[Contradiction]:
        """Get contradictions that haven't been addressed yet."""
        return [c for c in self.contradictions if not c.resolved]

    def get_memory_context(self, current_skill: str = "") -> str:
        """
        Generate a concise context string for the follow-up engine.
        This is what makes the interviewer "remember" previous answers.
        """
        parts = []

        # Skills tested and depth
        if self.depth_signals:
            parts.append("## Skills Assessed So Far")
            for skill, ds in self.depth_signals.items():
                depth_label = "sufficient" if ds.is_sufficiently_deep else "needs more probing"
                details = []
                if ds.has_real_project:
                    details.append("gave real project example")
                if ds.has_tradeoff_reasoning:
                    details.append("showed tradeoff reasoning")
                if ds.answers_vague > ds.answers_with_specifics:
                    details.append("mostly vague answers")
                detail_str = f" ({', '.join(details)})" if details else ""
                parts.append(
                    f"- {skill}: {ds.questions_asked} Qs, "
                    f"depth={depth_label}{detail_str}"
                )

        # Key factual claims (last 10 most recent, grouped)
        if self.factual_claims:
            parts.append("\n## Key Claims Made by Candidate")
            # Group by skill, show last 10
            recent = self.factual_claims[-15:]
            for claim in recent:
                parts.append(
                    f"- [Turn {claim.turn}, {claim.skill}] \"{claim.claim}\""
                )

        # Unresolved contradictions
        unresolved = self.get_unresolved_contradictions()
        if unresolved:
            parts.append("\n## ⚠ UNRESOLVED CONTRADICTIONS")
            for c in unresolved:
                parts.append(
                    f"- Turn {c.turn_a}: \"{c.claim_a}\" vs Turn {c.turn_b}: \"{c.claim_b}\" "
                    f"[{c.severity}]"
                )

        # Answer patterns
        if self.answer_patterns.total_answers > 0:
            ap = self.answer_patterns
            parts.append("\n## Candidate Answer Patterns")
            parts.append(
                f"- Specificity: {ap.avg_specificity:.1f}/10 "
                f"({ap.specific_count} specific, {ap.vague_count} vague out of {ap.total_answers})"
            )
            parts.append(
                f"- Style: {ap.practical_count} practical, "
                f"{ap.theoretical_count} theoretical"
            )

        # Topics mentioned organically (for natural bridges)
        if self.topics_mentioned:
            parts.append("\n## Topics Candidate Mentioned Organically")
            for topic, first_turn in sorted(self.topics_mentioned.items(), key=lambda x: x[1]):
                parts.append(f"- {topic} (first at turn {first_turn})")

        return "\n".join(parts) if parts else "No memory yet — this is the first question."

    def get_report_data(self) -> dict:
        """Get structured memory data for the final report."""
        return {
            "factual_claims": [asdict(c) for c in self.factual_claims],
            "contradictions": [asdict(c) for c in self.contradictions],
            "depth_signals": {
                skill: asdict(ds) for skill, ds in self.depth_signals.items()
            },
            "answer_patterns": asdict(self.answer_patterns),
            "strongest_moments": self.strongest_moments,
            "weakest_moments": self.weakest_moments,
            "specificity_scores": self.per_answer_specificity,
            "topics_mentioned": self.topics_mentioned,
        }

    def to_dict(self) -> dict:
        """Serialize the entire memory for state storage."""
        return {
            "factual_claims": [asdict(c) for c in self.factual_claims],
            "contradictions": [asdict(c) for c in self.contradictions],
            "depth_signals": {
                skill: asdict(ds) for skill, ds in self.depth_signals.items()
            },
            "answer_patterns": asdict(self.answer_patterns),
            "topics_mentioned": self.topics_mentioned,
            "strongest_moments": self.strongest_moments,
            "weakest_moments": self.weakest_moments,
            "per_answer_specificity": self.per_answer_specificity,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "InterviewMemory":
        """Deserialize from state storage."""
        mem = cls()
        if not data:
            return mem

        # Restore factual claims
        for c in data.get("factual_claims", []):
            mem.factual_claims.append(FactualClaim(**c))

        # Restore contradictions
        for c in data.get("contradictions", []):
            mem.contradictions.append(Contradiction(**c))

        # Restore depth signals
        for skill, ds_data in data.get("depth_signals", {}).items():
            mem.depth_signals[skill] = DepthSignal(**ds_data)

        # Restore answer patterns
        ap_data = data.get("answer_patterns", {})
        if ap_data:
            mem.answer_patterns = AnswerPatterns(**ap_data)

        # Restore simple fields
        mem.topics_mentioned = data.get("topics_mentioned", {})
        mem.strongest_moments = data.get("strongest_moments", [])
        mem.weakest_moments = data.get("weakest_moments", [])
        mem.per_answer_specificity = data.get("per_answer_specificity", [])

        return mem

    # ── Private methods ───────────────────────────────────────────────────

    def _extract_claims(self, question: str, answer: str, skill: str, turn: int) -> dict:
        """Use LLM to extract factual claims and assess specificity."""
        try:
            llm = get_eval_llm()
            prompt = CLAIM_EXTRACTION_PROMPT.format(
                question=question[:500],
                answer=answer[:1500],
                skill=skill,
            )
            response = llm.invoke([
                SystemMessage(content="You extract factual claims from interview answers. Return only JSON."),
                HumanMessage(content=prompt),
            ])
            raw = response.content.strip()
            # Strip markdown code fences
            if raw.startswith("```"):
                first_nl = raw.find("\n")
                if first_nl != -1:
                    raw = raw[first_nl + 1:]
                if raw.rstrip().endswith("```"):
                    raw = raw.rstrip()[:-3]
                raw = raw.strip()
            return json.loads(raw, strict=False)
        except Exception as exc:
            log.warning("Claim extraction failed: %s", exc)
            return {
                "claims": [],
                "specificity_score": 5,
                "realism_score": 5,
                "is_theoretical": False,
                "has_real_project": False,
                "has_tradeoff": False,
                "topics_mentioned": [],
            }

    def _check_contradictions(self, question: str, answer: str, turn: int) -> list[Contradiction]:
        """Use LLM to check if the new answer contradicts any previous claims."""
        # Only check if we have enough claims to compare against
        if len(self.factual_claims) < 2:
            return []

        try:
            # Build previous claims summary (grouped by skill)
            claims_by_skill: dict[str, list[str]] = {}
            for claim in self.factual_claims:
                claims_by_skill.setdefault(claim.skill, []).append(
                    f"[Turn {claim.turn}] {claim.claim}"
                )
            claims_text = ""
            for skill, claims in claims_by_skill.items():
                claims_text += f"\n{skill}:\n" + "\n".join(f"  - {c}" for c in claims) + "\n"

            llm = get_eval_llm()
            prompt = CONTRADICTION_CHECK_PROMPT.format(
                previous_claims=claims_text[:2000],
                question=question[:500],
                answer=answer[:1500],
            )
            response = llm.invoke([
                SystemMessage(content="You detect contradictions in interview answers. Return only JSON."),
                HumanMessage(content=prompt),
            ])
            raw = response.content.strip()
            if raw.startswith("```"):
                first_nl = raw.find("\n")
                if first_nl != -1:
                    raw = raw[first_nl + 1:]
                if raw.rstrip().endswith("```"):
                    raw = raw.rstrip()[:-3]
                raw = raw.strip()

            result = json.loads(raw, strict=False)

            if result.get("has_contradiction"):
                contradictions = []
                for c in result.get("contradictions", []):
                    contradictions.append(Contradiction(
                        claim_a=c.get("claim_a", ""),
                        claim_b=c.get("claim_b", ""),
                        turn_a=c.get("turn_a", 0),
                        turn_b=turn,
                        skill=c.get("skill", ""),
                        severity=c.get("severity", "minor"),
                    ))
                return contradictions

        except Exception as exc:
            log.warning("Contradiction check failed: %s", exc)

        return []

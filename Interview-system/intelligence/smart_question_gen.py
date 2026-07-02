"""
Smart Question Generator — 3-tier question sourcing.

Priority: Dataset (5000 Q's) → Semantic Search → LLM Generation (last resort)

Questions are grounded in the JD requirements and CV claims to create
a targeted, relevant interview experience.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage
from core.llm_config import get_eval_llm
from data_layer.phase1_data_layer import get_data_layer

log = logging.getLogger("phase3.smart_qgen")

QUESTION_GEN_PROMPT = """You are an expert technical interviewer. Generate ONE interview question.

Context:
- Position: {title} ({seniority})
- Target Skill: {skill}
- Domain: {domain}
- Candidate background: {cv_context}

Requirements:
- The question should test the candidate's practical knowledge of {skill}
- Make it specific and scenario-based, not generic
- Appropriate difficulty for {seniority} level

Return ONLY the question text, nothing else."""


@dataclass
class Question:
    question_id: str
    question_text: str
    domain: str
    sub_skill: str
    difficulty: str
    phase: str
    source: str  # "dataset", "semantic", or "llm_generated"


class SmartQuestionGen:
    """
    3-tier question sourcing:
    1. Structured filter from dataset (Pandas)
    2. Semantic search from dataset (ChromaDB)
    3. LLM generation (last resort only)
    """

    def __init__(self):
        self.dl = get_data_layer()
        self.llm = get_eval_llm()

    def get_question(
        self,
        skill: str,
        domain: str,
        phase: str,
        jd_title: str = "",
        jd_seniority: str = "Mid",
        cv_context: str = "",
        answered_ids: list[str] | None = None,
        role_title: str = "",
    ) -> Question:
        """
        Get the best question for a skill, trying dataset first.

        Args:
            skill: target skill to test (e.g. "Python", "Docker")
            domain: interview domain (e.g. "Technical", "Behavioral")
            phase: current interview phase
            jd_title: job title from JD
            jd_seniority: seniority from JD
            cv_context: relevant CV text for context
            answered_ids: already-asked question IDs
            role_title: role title for dataset filtering
        """
        answered = answered_ids or []

        # ── Tier 1: Structured filter from dataset ────────────────────────
        question = self._try_structured_filter(skill, domain, role_title, answered)
        if question:
            log.info("Tier 1 (dataset): found Q='%s' for skill='%s'", question.question_id, skill)
            return question

        # ── Tier 2: Semantic search from dataset ──────────────────────────
        question = self._try_semantic_search(skill, domain, role_title, answered)
        if question:
            log.info("Tier 2 (semantic): found Q='%s' for skill='%s'", question.question_id, skill)
            return question

        # ── Tier 3: LLM generation (last resort) ─────────────────────────
        question = self._generate_with_llm(skill, domain, phase, jd_title, jd_seniority, cv_context)
        log.info("Tier 3 (LLM generated): Q='%s' for skill='%s'", question.question_id, skill)
        return question

    def _try_structured_filter(
        self, skill: str, domain: str, role_title: str, answered: list[str]
    ) -> Question | None:
        """Tier 1: Filter questions from Pandas DataFrame."""
        import pandas as pd

        # Try progressively broader filters
        filter_attempts = [
            # 1. Exact: actual role + domain + skill
            {"role_id": role_title, "phase": domain, "domain": domain, "sub_skill": skill},
            # 2. Actual role, just skill
            {"role_id": role_title, "phase": domain, "sub_skill": skill},
            # 3. Broad: any role with domain + skill
            {"role_id": "Software Engineer", "phase": domain, "domain": domain, "sub_skill": skill},
            # 4. Broadest: any role, domain only
            {"role_id": "Software Engineer", "phase": domain, "domain": domain},
        ]

        for filters in filter_attempts:
            try:
                candidates = self.dl.filter_questions(
                    exclude_ids=answered,
                    **filters,
                )
                # Handle DataFrame result
                if isinstance(candidates, pd.DataFrame) and not candidates.empty:
                    row = candidates.iloc[0]
                    return Question(
                        question_id=str(row.get("question_id", "")),
                        question_text=str(row.get("question_text", "")),
                        domain=str(row.get("domain", domain)),
                        sub_skill=str(row.get("sub_skill", skill)),
                        difficulty=str(row.get("difficulty", "Medium")),
                        phase=domain,
                        source="dataset",
                    )
                # Handle list result
                elif isinstance(candidates, list) and candidates:
                    q = candidates[0]
                    return Question(
                        question_id=q.get("question_id", ""),
                        question_text=q.get("question_text", ""),
                        domain=q.get("domain", domain),
                        sub_skill=q.get("sub_skill", skill),
                        difficulty=q.get("difficulty", "Medium"),
                        phase=domain,
                        source="dataset",
                    )
            except Exception:
                continue

        return None

    def _try_semantic_search(
        self, skill: str, domain: str, role_title: str, answered: list[str]
    ) -> Question | None:
        """Tier 2: Semantic search via ChromaDB."""
        try:
            query = f"{domain} {skill} interview question"
            results = self.dl.find_questions(
                query=query,
                role_id=role_title,
                phase=domain,
                n_results=10,
                domain=domain,
            )
            for r in results:
                qid = r.get("id") or r.get("question_id")
                if qid and qid not in answered:
                    return Question(
                        question_id=qid,
                        question_text=r.get("document") or r.get("question_text", ""),
                        domain=r.get("metadata", {}).get("domain", domain),
                        sub_skill=r.get("metadata", {}).get("sub_skill", skill),
                        difficulty=r.get("metadata", {}).get("difficulty", "Medium"),
                        phase=domain,
                        source="semantic",
                    )
        except Exception as exc:
            log.debug("Semantic search failed for skill='%s': %s", skill, exc)
        return None

    def _generate_with_llm(
        self, skill: str, domain: str, phase: str,
        jd_title: str, jd_seniority: str, cv_context: str,
    ) -> Question:
        """Tier 3: Generate question with LLM (last resort)."""
        try:
            prompt = QUESTION_GEN_PROMPT.format(
                title=jd_title or "this role",
                seniority=jd_seniority or "Mid",
                skill=skill,
                domain=domain,
                cv_context=cv_context[:500] if cv_context else "No CV provided",
            )
            response = self.llm.invoke([
                SystemMessage(content="You are an expert technical interviewer."),
                HumanMessage(content=prompt),
            ])
            question_text = response.content.strip()
            if question_text.startswith('"') and question_text.endswith('"'):
                question_text = question_text[1:-1]

        except Exception as exc:
            log.error("LLM question generation failed: %s", exc)
            question_text = f"Tell me about your experience with {skill} and how you've applied it in your work."

        return Question(
            question_id=f"LLM-{uuid.uuid4().hex[:8]}",
            question_text=question_text,
            domain=domain,
            sub_skill=skill,
            difficulty="Medium",
            phase=phase,
            source="llm_generated",
        )

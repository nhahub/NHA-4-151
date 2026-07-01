"""
cv_agent.content_checker
========================
LLM-driven skill auto-categorisation and CV content quality analysis.

Two main entry points:
* ``auto_categorize_skills()`` — groups a flat skill list into semantic categories.
* ``check_content()``         — analyses a finished CV for repetition, weak verbs,
                                 spelling/grammar, tense issues, and vague phrasing.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Optional

from cv_agent.config import PipelineConfig, logger
from cv_agent.schemas import ContentCheckResult, ContentIssue
from cv_agent.utils import parse_json_robust


# ==============================================================================
# CONSTANTS — action-verb thesaurus & stop words
# ==============================================================================

STOP_WORDS: frozenset = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "this", "that",
    "these", "those", "i", "we", "you", "he", "she", "it", "they", "me",
    "us", "him", "her", "them", "my", "our", "your", "his", "its", "their",
    "as", "if", "not", "no", "so", "up", "out", "about", "into", "over",
    "after", "before", "between", "under", "through", "during", "more",
    "most", "other", "some", "such", "than", "too", "very", "just", "also",
    "all", "each", "every", "both", "few", "own", "same", "which", "who",
    "whom", "what", "when", "where", "how", "while", "per", "via",
})

# Weak verb phrases → suggested strong alternatives
WEAK_VERB_REPLACEMENTS: Dict[str, List[str]] = {
    "worked on":            ["developed", "engineered", "built", "crafted"],
    "helped with":          ["facilitated", "contributed to", "supported", "enabled"],
    "was responsible for":  ["managed", "led", "directed", "oversaw"],
    "involved in":          ["contributed to", "participated in", "drove", "executed"],
    "assisted with":        ["supported", "enabled", "strengthened", "bolstered"],
    "participated in":      ["contributed to", "collaborated on", "engaged in"],
    "dealt with":           ["managed", "resolved", "addressed", "handled"],
    "took care of":         ["managed", "administered", "maintained", "oversaw"],
    "was in charge of":     ["directed", "led", "managed", "headed"],
    "tasked with":          ["spearheaded", "drove", "orchestrated", "executed"],
    "worked with":          ["collaborated with", "partnered with", "engaged with"],
    "did":                  ["executed", "accomplished", "performed", "delivered"],
    "made":                 ["created", "developed", "designed", "produced"],
    "got":                  ["achieved", "attained", "secured", "obtained"],
    "put together":         ["assembled", "compiled", "organized", "constructed"],
    "looked at":            ["analyzed", "evaluated", "assessed", "reviewed"],
    "used":                 ["leveraged", "utilized", "employed", "applied"],
}

# Passive voice patterns
PASSIVE_RE = re.compile(
    r'\b(?:was|were|been|being)\s+'
    r'(?:responsible|involved|tasked|assigned|given|asked|told|made|used)\b',
    re.IGNORECASE,
)

# Vague filler phrases
VAGUE_PHRASES: Dict[str, str] = {
    "various tasks":       "Specify the actual tasks (e.g., 'data pipeline maintenance, model evaluation')",
    "different things":    "Replace with specific deliverables",
    "many things":         "Replace with specific items",
    "etc.":                "Remove 'etc.' and list 2–3 concrete examples instead",
    "and so on":           "Remove and list specific items",
    "stuff":               "Replace with specific technologies or deliverables",
    "a lot of":            "Quantify with a specific number or percentage",
    "some experience":     "Specify years or depth of experience",
    "good knowledge":      "Specify proficiency level or demonstrate with achievements",
    "familiar with":       "Use 'proficient in' or describe hands-on usage",
    "basic understanding": "Describe what you actually built or accomplished with the technology",
    "strong skills":       "Demonstrate the skill with a quantified achievement instead",
    "team player":         "Show collaboration through a specific example or project outcome",
    "hard worker":         "Demonstrate work ethic through quantified achievements",
    "detail-oriented":     "Show attention to detail through a concrete example",
    "responsible for":     "Replace with an action verb: 'managed', 'led', 'executed'",
}

# Section header regex (to detect which section an issue is in)
_SECTION_RE = re.compile(r'^#{1,3}\s+(.+)', re.MULTILINE)


# ==============================================================================
# HELPER: detect current section from character offset
# ==============================================================================

def _section_at_offset(text: str, offset: int) -> str:
    """Return the nearest section heading above the given character offset."""
    best = "General"
    for m in _SECTION_RE.finditer(text):
        if m.start() <= offset:
            best = m.group(1).strip().lstrip("#").strip()
        else:
            break
    return best


# ==============================================================================
# AUTO SKILL CATEGORIZATION (LLM-driven)
# ==============================================================================

def auto_categorize_skills(
    skills: List[str],
    target_role: str = "",
    target_industry: str = "",
    cfg: Optional[PipelineConfig] = None,
) -> Dict[str, List[str]]:
    """
    Use the LLM to group a flat list of skills into semantic categories.

    Falls back to ``{"General Skills": [...]}`` if the LLM call fails.
    """
    if not skills:
        return {}

    cfg = cfg or PipelineConfig()

    # Build user prompt
    skills_str = ", ".join(skills)
    user_prompt = f"Skills to categorize: {skills_str}"
    if target_role:
        user_prompt += f"\nTarget role: {target_role}"
    if target_industry:
        user_prompt += f"\nIndustry: {target_industry}"

    try:
        from cv_agent.gpu_queue import _gpu_queue
        from cv_agent.model_manager import ModelManager, chat
        from cv_agent.prompts import SKILL_CATEGORIZER_SYSTEM

        _gpu_queue.start()
        mm = ModelManager.get_instance()
        pipe = mm.writer_pipe(cfg)
        raw = _gpu_queue.submit(chat, pipe, SKILL_CATEGORIZER_SYSTEM, user_prompt, 0.3)
        parsed = parse_json_robust(raw)

        if not parsed or not isinstance(parsed, dict):
            logger.warning("auto_categorize_skills: LLM returned invalid JSON, using fallback")
            return {"General Skills": list(skills)}

        # Validate: every skill should appear somewhere
        categorized: Dict[str, List[str]] = {}
        seen: set = set()
        for cat_name, cat_skills in parsed.items():
            if isinstance(cat_skills, list):
                valid = [s for s in cat_skills if isinstance(s, str) and s.strip()]
                if valid:
                    categorized[str(cat_name)] = valid
                    seen.update(s.lower() for s in valid)

        # Add any missing skills to an "Other" bucket
        missing = [s for s in skills if s.lower() not in seen]
        if missing:
            categorized.setdefault("Other Skills", []).extend(missing)

        if not categorized:
            return {"General Skills": list(skills)}

        logger.info(
            "auto_categorize_skills: %d skills → %d categories",
            len(skills), len(categorized),
        )
        return categorized

    except Exception as e:
        logger.warning("auto_categorize_skills failed (%s), using fallback", e)
        return {"General Skills": list(skills)}


# ==============================================================================
# CONTENT QUALITY CHECK
# ==============================================================================

def _detect_repetition(text: str) -> tuple[Dict[str, int], List[ContentIssue]]:
    """Rule-based repetition detector. Returns word counts and issues."""
    # Tokenise into lowercase alpha words (3+ chars to skip noise)
    words = re.findall(r'[a-z]{3,}', text.lower())
    counts = Counter(w for w in words if w not in STOP_WORDS)

    repeated: Dict[str, int] = {}
    issues: List[ContentIssue] = []

    for word, count in counts.most_common():
        if count < 4:
            break
        repeated[word] = count
        issues.append(ContentIssue(
            category="repetition",
            severity="warning",
            text=word,
            suggestion=f"'{word}' appears {count} times. Use synonyms to vary your language.",
            location="Throughout CV",
        ))

    return repeated, issues


def _detect_weak_verbs(text: str) -> List[ContentIssue]:
    """Rule-based weak/passive verb detector."""
    issues: List[ContentIssue] = []
    text_lower = text.lower()

    # Check explicit weak verb phrases
    for phrase, replacements in WEAK_VERB_REPLACEMENTS.items():
        # Find all occurrences
        start = 0
        while True:
            idx = text_lower.find(phrase, start)
            if idx == -1:
                break
            section = _section_at_offset(text, idx)
            suggestion_str = ", ".join(f"'{r}'" for r in replacements[:3])
            issues.append(ContentIssue(
                category="weak_verb",
                severity="warning",
                text=phrase,
                suggestion=f"Replace with a stronger verb: {suggestion_str}",
                location=section,
            ))
            start = idx + len(phrase)

    # Check passive voice patterns
    for m in PASSIVE_RE.finditer(text):
        section = _section_at_offset(text, m.start())
        issues.append(ContentIssue(
            category="weak_verb",
            severity="warning",
            text=m.group(),
            suggestion="Rewrite in active voice with a strong action verb",
            location=section,
        ))

    return issues


def _detect_vague_phrases(text: str) -> List[ContentIssue]:
    """Rule-based vague/filler phrase detector."""
    issues: List[ContentIssue] = []
    text_lower = text.lower()

    for phrase, fix in VAGUE_PHRASES.items():
        start = 0
        while True:
            idx = text_lower.find(phrase, start)
            if idx == -1:
                break
            section = _section_at_offset(text, idx)
            issues.append(ContentIssue(
                category="vague",
                severity="info",
                text=phrase,
                suggestion=fix,
                location=section,
            ))
            start = idx + len(phrase)

    return issues


def _llm_content_check(cv_text: str, cfg: PipelineConfig) -> List[ContentIssue]:
    """LLM-based spelling, grammar, and tense checker."""
    try:
        from cv_agent.gpu_queue import _gpu_queue
        from cv_agent.model_manager import ModelManager, chat
        from cv_agent.prompts import CONTENT_CHECKER_SYSTEM

        _gpu_queue.start()
        mm = ModelManager.get_instance()
        pipe = mm.writer_pipe(cfg)
        raw = _gpu_queue.submit(chat, pipe, CONTENT_CHECKER_SYSTEM, f"CV to check:\n\n{cv_text}", 0.2)
        parsed = parse_json_robust(raw)

        if not parsed or "issues" not in parsed:
            logger.warning("_llm_content_check: LLM returned no issues block")
            return []

        issues: List[ContentIssue] = []
        valid_categories = {"spelling", "grammar", "tense", "weak_verb", "vague"}
        valid_severities = {"error", "warning", "info"}

        for item in parsed.get("issues", []):
            if not isinstance(item, dict):
                continue
            cat = item.get("category", "grammar")
            if cat not in valid_categories:
                cat = "grammar"
            sev = item.get("severity", "warning")
            if sev not in valid_severities:
                sev = "warning"
            issues.append(ContentIssue(
                category=cat,
                severity=sev,
                text=str(item.get("text", "")),
                suggestion=str(item.get("suggestion", "")),
                location=str(item.get("location", "")),
            ))

        logger.info("_llm_content_check: found %d issues", len(issues))
        return issues

    except Exception as e:
        logger.warning("_llm_content_check failed: %s", e)
        return []


def check_content(
    cv_text: str,
    cfg: Optional[PipelineConfig] = None,
) -> ContentCheckResult:
    """
    Run full content quality analysis on a finished CV.

    Combines rule-based checks (repetition, weak verbs, vague phrasing)
    with an LLM pass (spelling, grammar, tense inconsistencies).
    """
    cfg = cfg or PipelineConfig()

    # 1. Repetition
    repeated_words, rep_issues = _detect_repetition(cv_text)

    # 2. Weak verbs (rule-based)
    weak_issues = _detect_weak_verbs(cv_text)

    # 3. Vague phrases (rule-based)
    vague_issues = _detect_vague_phrases(cv_text)

    # 4. LLM check (spelling, grammar, tense + may also catch weak verbs/vague)
    llm_issues = _llm_content_check(cv_text, cfg)

    # Merge all issues, deduplicating by (category, text)
    all_issues: List[ContentIssue] = []
    seen_keys: set = set()
    for issue in rep_issues + weak_issues + vague_issues + llm_issues:
        key = (issue.category, issue.text.lower().strip())
        if key not in seen_keys:
            seen_keys.add(key)
            all_issues.append(issue)

    # Determine overall quality
    error_count = sum(1 for i in all_issues if i.severity == "error")
    warning_count = sum(1 for i in all_issues if i.severity == "warning")

    if error_count >= 3 or (error_count + warning_count) >= 8:
        quality = "poor"
    elif error_count >= 1 or warning_count >= 4:
        quality = "needs_work"
    elif warning_count >= 1 or len(all_issues) > 0:
        quality = "good"
    else:
        quality = "excellent"

    # Build summary
    parts: List[str] = []
    if error_count:
        parts.append(f"{error_count} error(s)")
    if warning_count:
        parts.append(f"{warning_count} warning(s)")
    info_count = sum(1 for i in all_issues if i.severity == "info")
    if info_count:
        parts.append(f"{info_count} suggestion(s)")
    summary = f"Content check found {', '.join(parts)}." if parts else "No issues found — excellent content quality."

    return ContentCheckResult(
        issues=all_issues,
        repeated_words=repeated_words,
        overall_quality=quality,
        summary=summary,
    )


# ==============================================================================
# FEEDBACK FORMATTER — turns issues into a judgmental writer prompt block
# ==============================================================================

def format_issues_as_feedback(content_result: "ContentCheckResult") -> str:
    """
    Convert a ContentCheckResult into a strongly-worded feedback block
    that gets injected into the next writer prompt so the LLM knows exactly
    what to fix.

    The output is intentionally direct and judgmental so the model treats
    these as mandatory corrections, not optional suggestions.
    """
    if not content_result.issues:
        return ""

    severity_label = {"error": "🔴 ERROR", "warning": "⚠️  WARNING", "info": "ℹ️  SUGGESTION"}

    lines = [
        "CONTENT QUALITY ISSUES — FIX ALL OF THESE (MANDATORY):",
        "  These were detected in the previous iteration. Every single one must be resolved.",
    ]

    # Group by severity: errors first, then warnings, then info
    for sev in ("error", "warning", "info"):
        issues_for_sev = [i for i in content_result.issues if i.severity == sev]
        for issue in issues_for_sev:
            label = severity_label.get(sev, sev.upper())
            loc = f" ({issue.location})" if issue.location else ""
            lines.append(
                f"  {label} [{issue.category}]{loc}: \"{issue.text}\" → {issue.suggestion}"
            )

    # Add repetition summary
    bad_repeats = {w: c for w, c in content_result.repeated_words.items() if c >= 4}
    if bad_repeats:
        repeats_str = ", ".join(f'"{w}" ×{c}' for w, c in list(bad_repeats.items())[:6])
        lines.append(
            f"  ⚠️  WARNING [repetition]: Overused words — {repeats_str}. "
            "Use diverse synonyms throughout."
        )

    return "\n".join(lines)


# ==============================================================================
# INLINE CV FIXER — rule-based + optional LLM surgical pass
# ==============================================================================

def apply_content_fixes(
    cv_text: str,
    content_result: "ContentCheckResult",
    cfg: Optional[PipelineConfig] = None,
) -> str:
    """
    Apply content fixes directly to the CV text.

    Phase 1 — Rule-based (always runs):
      • Replaces weak verb phrases with the first strong alternative.
      • Removes or rewrites vague filler phrases.
      • Does NOT touch repetition here (writer prompt handles that via feedback).

    Phase 2 — LLM surgical pass (runs when issue count >= cfg.content_fix_llm_threshold):
      • Sends the full CV + issue list to the CONTENT_FIXER_SYSTEM prompt.
      • The LLM fixes ONLY the listed problems; it is instructed not to rewrite anything else.

    Returns the fixed CV text (or the original if all fixes fail).
    """
    cfg = cfg or PipelineConfig()
    fixed = cv_text

    # ── Phase 1: rule-based weak-verb substitution ───────────────────────────
    weak_issues = [i for i in content_result.issues if i.category == "weak_verb"]
    for issue in weak_issues:
        phrase = issue.text.lower()
        # Find a strong replacement from our thesaurus
        replacements = WEAK_VERB_REPLACEMENTS.get(phrase)
        if replacements:
            replacement = replacements[0]
            # Case-insensitive replacement preserving surrounding text
            pattern = re.compile(re.escape(issue.text), re.IGNORECASE)
            fixed = pattern.sub(replacement, fixed, count=1)
            logger.debug("apply_content_fixes: replaced weak verb '%s' → '%s'", issue.text, replacement)

    # ── Phase 1: rule-based vague phrase removal ─────────────────────────────
    # We only remove phrases where the fix is simple enough for a regex swap
    simple_vague_fixes: dict = {
        "etc.":         "",          # strip trailing etc.
        "and so on":    "",
        "stuff":        "resources",
        "a lot of":     "numerous",
        "various tasks": "key tasks",
    }
    for phrase, simple_fix in simple_vague_fixes.items():
        if phrase in fixed.lower():
            pattern = re.compile(re.escape(phrase), re.IGNORECASE)
            fixed = pattern.sub(simple_fix, fixed)
            logger.debug("apply_content_fixes: replaced vague phrase '%s' → '%s'", phrase, simple_fix)

    # ── Phase 2: LLM surgical pass ───────────────────────────────────────────
    total_issues = len(content_result.issues)
    if total_issues < cfg.content_fix_llm_threshold:
        logger.info(
            "apply_content_fixes: skipping LLM pass (issues=%d < threshold=%d)",
            total_issues, cfg.content_fix_llm_threshold,
        )
        return fixed

    # Build the issues list for the prompt
    issue_lines = []
    for i, issue in enumerate(content_result.issues, 1):
        loc = f" in {issue.location}" if issue.location else ""
        issue_lines.append(
            f'{i}. [{issue.category}]{loc}: find "{issue.text}" → fix: {issue.suggestion}'
        )

    issues_block = "\n".join(issue_lines)
    user_prompt = (
        f"Fix ONLY these {total_issues} issue(s) in the CV below. "
        f"Do NOT change anything else.\n\n"
        f"ISSUES TO FIX:\n{issues_block}\n\n"
        f"CV TO FIX:\n{fixed}"
    )

    try:
        from cv_agent.gpu_queue import _gpu_queue
        from cv_agent.model_manager import ModelManager, chat
        from cv_agent.prompts import CONTENT_FIXER_SYSTEM

        _gpu_queue.start()
        mm = ModelManager.get_instance()
        pipe = mm.writer_pipe(cfg)
        result = _gpu_queue.submit(chat, pipe, CONTENT_FIXER_SYSTEM, user_prompt, 0.1)

        # Basic sanity check: result must look like a CV (has a heading or bullet)
        if result and len(result) >= len(cv_text) * 0.5 and ("#" in result or "-" in result):
            logger.info(
                "apply_content_fixes: LLM pass applied %d fix(es), CV length %d→%d chars",
                total_issues, len(fixed), len(result),
            )
            return result
        else:
            logger.warning(
                "apply_content_fixes: LLM pass returned suspicious output (len=%d), keeping rule-based result",
                len(result) if result else 0,
            )
            return fixed

    except Exception as e:
        logger.warning("apply_content_fixes: LLM pass failed (%s), using rule-based result", e)
        return fixed


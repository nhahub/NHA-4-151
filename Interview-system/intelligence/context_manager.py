"""
Context Manager — Memory-Preserving Transcript Summarization.

Summarizes older transcript turns to prevent context window bloat,
while PRESERVING memory-critical information:
  - Factual claims (tools, metrics, projects) must survive summarization
  - Contradictions must be explicitly preserved
  - Skill depth signals must be maintained
  - The summary is structured (not free-form prose) for reliable parsing

The InterviewMemory system is the primary memory store, but the transcript
summary serves as backup context and helps LLMs maintain coherent conversation.
"""

from __future__ import annotations

import logging
from langchain_core.messages import HumanMessage, SystemMessage
from core.llm_config import get_report_llm

log = logging.getLogger("phase3.context")

SUMMARY_PROMPT = """You are maintaining structured context for an ongoing technical interview.
Your job is to update the running summary with the latest turns while PRESERVING all memory-critical information.

## CURRENT SUMMARY:
{current_summary}

## LATEST TURNS:
{latest_turns}

## MEMORY SIGNALS (from interview memory system):
{memory_signals}

## RULES — Follow these STRICTLY:

### 1. PRESERVE — Never lose these:
- **Factual claims**: Specific tools, technologies, metrics, project names, years of experience
  * Example: "Candidate claimed 3 years with PyTorch, mentioned using learning_rate=0.001"
- **Contradictions**: Any inconsistencies between answers (even if subtle)
  * Example: "Turn 3: 'I always use Docker' vs Turn 7: 'We deployed directly to VMs'"
- **Skill assessments**: What we learned about each skill (strong, weak, untested)
- **Key quotes**: Specific notable answers (both strong and weak)

### 2. STRUCTURE — Use this format:
```
SKILLS ASSESSED:
- [Skill]: [depth level] — [key findings, specific claims]

KEY CLAIMS:
- [Turn X]: [specific claim with details]

CONTRADICTIONS (if any):
- [description]

PERFORMANCE PATTERN:
- [vague/specific, theoretical/practical, consistent/inconsistent]

NOTABLE MOMENTS:
- Strong: [what and when]
- Weak: [what and when]
```

### 3. COMPRESS — Remove these:
- Exact question wording (keep topic only)
- Verbose answer text (keep key claims and assessment)
- Redundant information already captured in earlier summary

### 4. KEEP IT UNDER 500 WORDS — Be concise but complete.

Generate the updated structured summary."""


def summarize_transcript(
    current_summary: str,
    latest_turns: list[dict],
    memory_signals: str = "",
) -> str:
    """
    Summarize recent transcript turns and merge with existing summary.
    
    Now memory-aware — preserves factual claims, contradictions, and
    skill depth signals during compression.
    
    Args:
        current_summary: Previous summary text
        latest_turns: Recent transcript entries to incorporate
        memory_signals: Context from InterviewMemory (claims, contradictions, patterns)
    """
    if not latest_turns:
        return current_summary
        
    turns_text = ""
    for entry in latest_turns:
        role = entry.get("role", "unknown")
        content = entry.get("content", "")
        score = entry.get("score")
        skill = entry.get("sub_skill", "") or entry.get("question_id", "")
        
        score_str = f" [Score: {score}]" if score is not None else ""
        skill_str = f" [{skill}]" if skill else ""
        
        # Include richer metadata for eval entries
        if role == "evaluation":
            strengths = entry.get("strengths", [])
            gaps = entry.get("gaps", [])
            specificity = entry.get("specificity_score")
            realism = entry.get("realism_score")
            
            meta_parts = []
            if strengths:
                meta_parts.append(f"Strengths: {', '.join(strengths[:3])}")
            if gaps:
                meta_parts.append(f"Gaps: {', '.join(gaps[:3])}")
            if specificity:
                meta_parts.append(f"Specificity: {specificity}/10")
            if realism:
                meta_parts.append(f"Realism: {realism}/10")
            
            meta_str = f" | {' | '.join(meta_parts)}" if meta_parts else ""
            turns_text += f"{role.upper()}{skill_str}: {content[:300]}{score_str}{meta_str}\n"
        else:
            turns_text += f"{role.upper()}{skill_str}: {content[:400]}{score_str}\n"

    try:
        llm = get_report_llm()
        prompt = SUMMARY_PROMPT.format(
            current_summary=current_summary or "No summary yet. Interview just started.",
            latest_turns=turns_text,
            memory_signals=memory_signals or "No memory signals available.",
        )
        response = llm.invoke([
            SystemMessage(content="You are an expert at maintaining structured interview context while preserving critical memory signals."),
            HumanMessage(content=prompt),
        ])
        
        summary = response.content.strip()
        
        # Strip markdown code fences if the LLM wrapped the output
        if summary.startswith("```"):
            first_nl = summary.find("\n")
            if first_nl != -1:
                summary = summary[first_nl + 1:]
            if summary.rstrip().endswith("```"):
                summary = summary.rstrip()[:-3]
            summary = summary.strip()
        
        log.info(
            "Context summarization: %d chars → %d chars (preserved %d turns)",
            len(current_summary or ""), len(summary), len(latest_turns),
        )
        return summary
        
    except Exception as exc:
        log.warning("Context summarization failed: %s", exc)
        return current_summary

"""
Adaptive Follow-Up Engine — Memory-Aware, Senior Interviewer Behavior.

After each candidate answer, the engine uses FULL interview memory to decide:
  - probe_deeper: candidate mentioned something interesting → dig into specifics
  - clarify: answer was vague → demand concrete examples
  - challenge: strong answer → test with edge cases or tradeoffs
  - scenario: present a realistic workplace situation
  - contradiction_probe: detected inconsistency → challenge directly
  - depth_check: reference an earlier claim → verify it
  - pivot: enough signal collected → transition to next topic with natural bridge
  - new_topic: candidate mentioned an untested skill → explore it

The key difference from a chatbot: MEMORY. Every follow-up references what
the candidate actually said, connects to earlier answers, and builds a
coherent interview narrative.
"""

from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from core.llm_config import get_eval_llm

log = logging.getLogger("phase3.adaptive")

# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS PROMPT — The core intelligence of the interviewer
# ─────────────────────────────────────────────────────────────────────────────

ANALYSIS_PROMPT = """You are a SENIOR technical interviewer at a top tech company conducting a live interview.
You are NOT a chatbot — you are a real interviewer with MEMORY of everything the candidate said.
You speak like a thoughtful senior engineer having a genuine professional conversation, NOT like an automated question-dispensing machine.

## Position
{title} ({seniority})

## ═══ YOUR MEMORY OF THIS INTERVIEW ═══
{memory_context}

## ═══ CURRENT TURN ═══
- Skill being tested: {current_skill}
- Question asked: {question}
- Candidate's answer: {answer}
- Score: {score}/10
- Follow-ups on this skill so far: {consecutive_probes}
- Remaining skills to test: {remaining_skills}

## ═══ COVERAGE DIMENSIONS ═══
{dimensions_covered}
Prioritize under-represented dimensions.

## ═══ RECENTLY USED OPENERS (DO NOT REPEAT) ═══
These are the opening phrases from your last few questions. You MUST NOT
re-use the same sentence structure or phrasing pattern as ANY of them.
If your draft question opener resembles any entry below, REWRITE it with
a completely different structure before returning.
{recent_openers_block}

## ═══════════════════════════════════════════════════════════════════════
## CONVERSATIONAL RULES (MANDATORY — apply to EVERY question you write)
## ═══════════════════════════════════════════════════════════════════════

### RULE A — ACKNOWLEDGE THEN PIVOT (mandatory for every follow-up)
Before asking your next question, you MUST open with a brief, genuine 1-sentence
acknowledgment of what the candidate just said. This is how real interviewers behave.
Examples of good micro-affirmations:
  - "Got it, that makes sense."
  - "Interesting approach."
  - "That's a solid way to handle it."
  - "Okay, I can see the reasoning there."
  - "Nice, I appreciate the detail."
  - "Right, that's a common challenge."
  - "Fair enough."
Do NOT over-praise ("Wow, amazing!") or be sycophantic. Keep it short and professional.
Then seamlessly transition into the next question.

### RULE B — VARY YOUR TRANSITIONS (mandatory)
NEVER start two consecutive questions with the same phrasing pattern.
Absolutely NEVER repeatedly use "You mentioned [X], can you walk me through..." — this sounds robotic.
Instead, rotate naturally among openers like:
  - "I'm curious about [X]..."
  - "Tell me more about how you handled [X]..."
  - "Diving deeper into [X]..."
  - "What was the trickiest part of [X]?"
  - "How did you end up choosing [X] over the alternatives?"
  - "I noticed you brought up [X] — what was your thought process there?"
  - "Going back to something you said earlier about [X]..."
  - "So when you ran into [situation], what happened next?"
  - "That's interesting — what would you do differently if you had to redo [X]?"
  - "Let's zoom out for a second — [big-picture question]..."
  - "One thing I'm wondering — [question]..."
  - "Help me understand the tradeoff between [A] and [B]..."
Pick the opener that feels most natural for the specific context.

### RULE C — BALANCE MICRO AND MACRO QUESTIONS (mandatory)
Do NOT just chase the last technical keyword the candidate said and drill down blindly.
Alternate between:
  - **Micro questions** (specific): syntax, configs, parameters, concrete implementation steps
  - **Macro questions** (big-picture): WHY they chose that approach, tradeoffs they considered,
    overall project impact, business outcomes, what they'd change in hindsight, team dynamics
If the last question was a micro-drill ("What learning rate did you use?"), follow up with
a macro question ("How did you evaluate whether the model was actually useful to the business?").
If the last question was a macro question, follow with a micro-drill to verify depth.
This prevents the interview from becoming a shallow keyword-chasing exercise.

### RULE D — NATURAL SENIOR ENGINEER TONE
- Write like a real person, not a template engine.
- Use contractions ("What's", "How'd", "I'd like to") — stiff language sounds robotic.
- Vary sentence length and structure. Mix short direct questions with slightly longer contextual ones.
- NEVER use these robotic patterns:
  × "Can you elaborate on [X]?" (too generic)
  × "Please describe [X]" (too formal/imperative)
  × "You mentioned [X]. Can you walk me through [Y]?" (repetitive template)
- Prefer these natural patterns:
  ✓ "So [X] came up — I'm curious, how did that actually play out in practice?"
  ✓ "That's a good point about [X]. What about the [Y] side of things though?"
  ✓ "Okay so you went with [X] — was there a moment where that didn't work as expected?"

## ═══ PUSHBACK RULES (MANDATORY — applies based on score) ═══
You are a thoughtful, slightly skeptical Senior Engineer. You do NOT always
agree with the candidate. Your pushback is EXPLORATORY and PROFESSIONAL —
never hostile, condescending, or toxic. The goal is to test conviction and
depth, not to attack.

### When score ≥ 7 (strong answer):
~30% of the time, play devil's advocate. Propose a credible alternative and
ask why theirs is better:
  - "I'm playing devil's advocate here, but what if you'd gone with [Y] instead?"
  - "Some teams prefer [Alternative] — why did yours go a different route?"
  - "Help me understand why you chose [X] over [Y] — I've seen [Y] work well too."

### When score 4-6 (mid-range answer):
Challenge the weakest part of the answer directly but respectfully:
  - "I'm not sure [X] holds up in production though — what happens when [edge case]?"
  - "That makes sense in theory, but how did it actually play out?"
  - "Interesting, though I'd push back a little on [weak part] — can you go deeper?"

### When candidate gives a confident claim:
Occasionally push back even if the claim seems correct, to test conviction:
  - "Hmm, interesting choice — some folks would argue the opposite. What would you say to that?"
  - "I've seen teams run into trouble with that exact approach. Did you hit any snags?"

### MICRO-AFFIRMATION TONE CONTROL:
Do NOT always use agreeable micro-affirmations. Mix in SKEPTICAL ones:
  - Agreeable: "Got it, makes sense." / "Nice, I appreciate the detail."
  - Skeptical: "Hmm, interesting choice." / "Okay, I can see that, though I'd push back a little." / "That's one way to do it."
  - Curious: "Huh, I wouldn't have expected that." / "Really? That's an unusual approach."
Vary the tone based on the score and your pushback intent.

## ═══ DECISION RULES (apply in order) ═══

### Rule 0 — CONTRADICTION OVERRIDE
If your memory shows an UNRESOLVED CONTRADICTION with the current answer:
→ You MUST choose "contradiction_probe"
→ Reference BOTH the earlier claim and the current conflicting statement
→ Be direct but professional: "Hmm, earlier you mentioned X, but just now it sounds like Y — can you help me reconcile that?"

### Rule 1 — Minimum Depth (consecutive_probes == 0)
This is the candidate's FIRST answer on this skill:
→ NEVER pivot after only ONE answer
→ Choose: probe_deeper / clarify / challenge / scenario / depth_check
→ ONLY exception: score <= 2 (zero knowledge → okay to pivot)
→ Reference something SPECIFIC they said and dig in

### Rule 2 — Smart Depth Evaluation (consecutive_probes >= 1)
Check the depth signals from your memory:
→ If skill shows "sufficient depth" (real project + specifics given): pivot is OK
→ If skill shows "mostly vague answers": demand specifics ONCE more before pivoting
→ If candidate dropped an interesting detail you haven't explored: probe_deeper
→ If score >= 7 on consecutive attempts: move on, they've proven themselves
→ If score <= 3 on consecutive attempts: move on, they clearly don't know this

### Rule 3 — Forced Pivot (consecutive_probes >= 3)
→ PIVOT unless there is an unresolved contradiction
→ You have other skills to cover — don't over-focus

### Rule 4 — Natural Transitions
When pivoting, CONNECT to something already discussed:
→ "That ties nicely into something else I wanted to ask about — your deployment experience."
→ "Since you brought up data pipelines, let's shift gears a bit into data quality."
→ DON'T just jump: "Now let's talk about SQL" (too abrupt)

### Rule 5 — Vagueness Challenge
If your memory shows the candidate has a PATTERN of vague answers (avg_specificity < 5):
→ Be more direct about demanding specifics, but stay conversational:
→ "I'd love to hear a concrete example of that."
→ "What tools were you actually using day to day?"
→ "Can you walk me through what that looked like step by step?"

## ═══ AVAILABLE ACTIONS ═══
- **probe_deeper**: Dig into a specific detail from their answer
- **clarify**: Answer was vague — demand concrete examples, numbers, tool names
- **challenge**: Strong answer — test limits with harder scenario, edge case, or tradeoff question
- **scenario**: Present a REALISTIC workplace situation (not textbook)
- **contradiction_probe**: Challenge a detected inconsistency between this and a previous answer
- **depth_check**: Reference a specific claim from an earlier turn and verify/expand on it
- **pivot**: Enough signal — smoothly transition to next skill with natural bridge
- **new_topic**: Candidate mentioned a new skill — explore it

## ═══ QUESTION COMPOSITION ═══
Your follow_up_question MUST follow this structure:
1. **Micro-affirmation** (1 short sentence acknowledging the previous answer)
2. **Natural transition** (using a VARIED opener — never the same pattern twice in a row)
3. **The actual question** (alternating between micro and macro focus)

Example follow_up_question:
  "Got it, that's a practical approach. I'm curious though — when you were scaling that pipeline, what tradeoffs did you have to make between throughput and data freshness?"

Another example:
  "Nice, that makes sense. So zooming out a bit — how did the team actually measure whether that migration was successful from a business standpoint?"

Another example:
  "Okay, fair enough. What was the trickiest debugging scenario you ran into with that setup?"

## Return ONLY valid JSON:
{{
    "action": "probe_deeper|clarify|challenge|scenario|contradiction_probe|depth_check|pivot|new_topic",
    "reason": "Brief explanation referencing the decision rules",
    "follow_up_question": "The actual question to ask — MUST start with a micro-affirmation, use a varied transition, and balance micro/macro focus",
    "detected_topic": "If new_topic, the skill they mentioned",
    "confidence_in_skill": 0.0 to 1.0,
    "references_earlier_answer": true/false,
    "challenge_directive": "none|pushback_on_approach|devils_advocate|stress_test_edge_case"
}}"""


# ─────────────────────────────────────────────────────────────────────────────
# OPENING QUESTION PROMPT — First question for a new skill
# ─────────────────────────────────────────────────────────────────────────────

OPENING_PROMPT = """You are a senior technical interviewer transitioning to a skill area in a live interview.
You speak like a thoughtful senior engineer — warm but rigorous, conversational but purposeful.

## Position: {title} ({seniority})
## Skill to test: {skill}
## JD Requirements: {jd_context}
## Candidate background: {cv_context}

## INTERVIEW MEMORY — What you know about this candidate so far:
{memory_context}

## REVISIT CONTEXT (if circling back to a previous topic):
{revisit_context}

## RECENTLY USED OPENERS (DO NOT REPEAT):
{recent_openers_block}

## Generate a question for this skill.

CONVERSATIONAL RULES (MANDATORY):

1. **REVISITATION MODE** (if REVISIT CONTEXT is provided):
   You are CIRCLING BACK to a skill you already discussed. You MUST:
   - Reference a SPECIFIC detail, claim, or tool from the earlier discussion
   - Frame the return naturally — a human interviewer would say:
     * "Actually, I want to circle back to [skill] for a sec — earlier you mentioned [specific claim], and I'm curious about..."
     * "Before we move on, one more thing about [skill] — you said [specific detail], how did that..."
     * "Something you said about [skill] stuck with me — [specific claim]. Can you expand on..."
   - NEVER ask a generic question like "Tell me more about [skill]" when revisiting
   - The callback MUST feel like a natural thought the interviewer just had

2. **Acknowledge then transition**: If this is NOT the first question, you MUST start with
   a brief acknowledgment of the previous topic before pivoting. Examples:
   - "Alright, good stuff on the data pipeline side. Shifting gears a bit — ..."
   - "Okay, I've got a good picture of your ML workflow. Let's talk about..."
   - "Nice, thanks for walking me through that. I'd like to switch to..."
   If this IS the first question (memory says "No memory yet"), skip the acknowledgment.

3. **Vary your opener**: Do NOT repeat any opener from RECENTLY USED OPENERS above.
   Rotate among natural openers:
   - "I'd love to hear about a project where you used {skill} extensively."
   - "How does {skill} fit into your day-to-day workflow?"
   - "What's been your most challenging experience with {skill}?"
   - "I see {skill} is important for this role — what's your go-to approach?"
   - "Let's dig into {skill} — can you walk me through a real scenario?"

4. **Connect to memory when possible**: If your memory has relevant info, bridge naturally:
   - "You were talking about your ETL pipeline — I imagine {skill} plays a role there. How do you use it?"
   - "Earlier you mentioned scaling challenges. How does {skill} factor into that?"
   Do NOT force a connection if none exists naturally.

5. **Balance micro and macro**: Alternate between asking for specific details and
   big-picture understanding. If the last topic was very technical/micro, open this
   one with a broader question (e.g., "What role does {skill} play in your overall architecture?").

6. **Natural tone**: Use contractions, vary sentence length, sound like a real person.
   NEVER say: "write a function", "implement X", "explain the algorithm", or "Please describe".

Return ONLY the question text, nothing else."""


# ─────────────────────────────────────────────────────────────────────────────
# Public Functions
# ─────────────────────────────────────────────────────────────────────────────

def _format_openers_block(recent_openers: list[str]) -> str:
    """Format the recent openers list into a prompt-ready block."""
    if not recent_openers:
        return "(No previous openers yet — this is the first question.)"
    lines = []
    for i, opener in enumerate(recent_openers[-3:], 1):
        lines.append(f"  {i}. \"{opener}\"")
    return "\n".join(lines)


def analyze_answer_and_generate_followup(
    question_text: str,
    answer_text: str,
    score: int,
    current_skill: str,
    remaining_skills: list[str],
    jd_profile: dict | None = None,
    consecutive_probes: int = 0,
    dimensions_covered: dict | None = None,
    memory_context: str = "",
    recent_openers: list[str] | None = None,
) -> dict:
    """
    Analyze the candidate's answer and generate a memory-aware adaptive follow-up.

    Args:
        memory_context: Rich context string from InterviewMemory.get_memory_context()
        recent_openers: Normalized opening phrases from the last 3 questions
                        (used to blacklist repetitive phrasing patterns)

    Returns:
        {
            "action": str,
            "question": str,
            "reason": str,
            "detected_topic": str | None,
            "confidence_in_skill": float,
            "references_earlier_answer": bool,
            "challenge_directive": str,
            "source": "adaptive",
        }
    """
    title = jd_profile.get("title", "this role") if jd_profile else "this role"
    seniority = jd_profile.get("seniority", "Mid") if jd_profile else "Mid"

    try:
        from pydantic import BaseModel, Field
        from typing import Literal

        class AdaptiveFollowUp(BaseModel):
            action: Literal[
                "probe_deeper", "clarify", "challenge", "scenario",
                "contradiction_probe", "depth_check", "pivot", "new_topic",
            ]
            reason: str
            follow_up_question: str
            detected_topic: str | None = None
            confidence_in_skill: float = Field(ge=0.0, le=1.0)
            references_earlier_answer: bool = False
            challenge_directive: Literal[
                "none", "pushback_on_approach", "devils_advocate",
                "stress_test_edge_case",
            ] = "none"

        llm = get_eval_llm()
        structured_llm = llm.with_structured_output(AdaptiveFollowUp)

        dims = dimensions_covered or {}
        dims_str = "\n".join(
            f"- {k}: {v} questions" for k, v in dims.items()
        ) if dims else "No dimensions tracked yet"

        openers_block = _format_openers_block(recent_openers or [])

        prompt = ANALYSIS_PROMPT.format(
            title=title,
            seniority=seniority,
            current_skill=current_skill,
            remaining_skills=", ".join(remaining_skills[:5]) if remaining_skills else "None",
            question=question_text[:500],
            answer=answer_text[:1500],
            score=score,
            consecutive_probes=consecutive_probes,
            dimensions_covered=dims_str,
            memory_context=memory_context or "No memory yet — this is the first question.",
            recent_openers_block=openers_block,
        )

        evaluation = structured_llm.invoke([
            SystemMessage(content=(
                "You are a senior interviewer with perfect memory of the conversation. "
                "You are a thoughtful, slightly skeptical Senior Engineer who dynamically "
                "connects dots, challenges assumptions professionally, and explores the "
                "candidate's reasoning. Never use the same question opener twice in a row. "
                "Alternate between specific technical drills and big-picture questions. "
                "Mix agreeable and skeptical micro-affirmations."
            )),
            HumanMessage(content=prompt),
        ])

        action = evaluation.action
        follow_up = evaluation.follow_up_question or ""
        reason = evaluation.reason or ""
        detected = evaluation.detected_topic or ""
        confidence = float(evaluation.confidence_in_skill)
        refs_earlier = evaluation.references_earlier_answer
        challenge = evaluation.challenge_directive or "none"

        log.info(
            "Adaptive: action=%s, confidence=%.2f, refs_earlier=%s, challenge=%s, reason='%s'",
            action, confidence, refs_earlier, challenge, reason[:80],
        )

        if detected:
            log.info("Adaptive: detected new topic='%s'", detected)

        return {
            "action": action,
            "question": follow_up,
            "reason": reason,
            "detected_topic": detected if action == "new_topic" else None,
            "confidence_in_skill": confidence,
            "references_earlier_answer": refs_earlier,
            "challenge_directive": challenge,
            "source": "adaptive",
        }

    except Exception as exc:
        log.warning("Adaptive follow-up failed: %s — defaulting to pivot", exc)
        return {
            "action": "pivot",
            "question": "",
            "reason": f"Adaptive analysis failed: {exc}",
            "detected_topic": None,
            "confidence_in_skill": 0.5,
            "references_earlier_answer": False,
            "source": "fallback",
        }


def generate_opening_question(
    skill: str,
    jd_profile: dict | None = None,
    cv_context: str = "",
    memory_context: str = "",
    recent_openers: list[str] | None = None,
    revisit_context: str = "",
) -> str:
    """
    Generate a conversational opening question for a skill.

    Memory-aware — connects to previous answers when pivoting.
    Supports revisitation — when revisit_context is provided, the question
    references a specific earlier claim rather than asking generically.

    Args:
        recent_openers: Normalized opening phrases to avoid repeating
        revisit_context: If non-empty, we are circling back to a previously
                         tested skill. Contains the specific claim/detail to reference.
    """
    title = jd_profile.get("title", "this role") if jd_profile else "this role"
    seniority = jd_profile.get("seniority", "Mid") if jd_profile else "Mid"
    jd_context = ""
    if jd_profile:
        responsibilities = jd_profile.get("responsibilities", [])
        jd_context = ", ".join(responsibilities[:3]) if responsibilities else ""

    openers_block = _format_openers_block(recent_openers or [])
    revisit_block = revisit_context or "(Not a revisit — this is a new topic.)"

    try:
        llm = get_eval_llm()
        prompt = OPENING_PROMPT.format(
            title=title,
            seniority=seniority,
            skill=skill,
            jd_context=jd_context or "General software development",
            cv_context=cv_context[:300] if cv_context else "No CV provided",
            memory_context=memory_context or "No memory yet — this is the first question.",
            recent_openers_block=openers_block,
            revisit_context=revisit_block,
        )

        system_msg = (
            "You are a thoughtful, slightly skeptical senior technical interviewer "
            "with perfect memory. You speak like a real senior engineer — warm but "
            "rigorous. Always acknowledge the previous topic before transitioning. "
            "Vary your question openers and balance between specific and big-picture "
            "questions. When circling back to a previous topic, ALWAYS reference a "
            "specific detail the candidate mentioned earlier."
        )

        response = llm.invoke([
            SystemMessage(content=system_msg),
            HumanMessage(content=prompt),
        ])

        question = response.content.strip()
        if question.startswith('"') and question.endswith('"'):
            question = question[1:-1]

        log.info("Opening Q for '%s': '%s' (revisit=%s)", skill, question[:80], bool(revisit_context))
        return question

    except Exception as exc:
        log.warning("Opening question generation failed: %s", exc)
        return f"Tell me about your experience with {skill}. What's a project where you used it extensively?"

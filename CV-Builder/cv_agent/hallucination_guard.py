"""
cv_agent.hallucination_guard
=============================
Standalone, reusable hallucination detection for CV text.

Uses a hybrid **semantic + rule-based** multi-layer classifier to
distinguish real technical skills from normal CV vocabulary, instead
of relying on a static blocklist of safe words.

Classifier layers:
  1. Tech Fingerprint  — structural analysis (CamelCase, special chars, etc.)
  2. Morphological     — English suffix/verb-form detection
  3. Semantic Category — pattern-based CV vocabulary classification
  4. Context           — skill-list vs. sentence analysis

Supports ontology loading from JSON/YAML files.
"""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from cv_agent.config import logger
from cv_agent.schemas import GuardResult, JDContext, UserProfile
from cv_agent.utils import normalise


class HallucinationGuard:
    """Validates CV text against a user profile to detect hallucinated content."""

    MAX_PERCENTAGE = 99
    MAX_REVENUE_B = 10
    MAX_MULTIPLIER = 50

    _DEFAULT_TECH_ONTOLOGY: Set[str] = {
        "python", "sql", "tensorflow", "keras", "pytorch",
        "scikit-learn", "sklearn", "xgboost", "langchain",
        "llamaindex", "openai", "transformers", "pandas",
        "numpy", "matplotlib", "seaborn", "plotly",
        "power bi", "powerbi", "tableau", "streamlit",
        "dash", "postgresql", "postgres", "mysql",
        "mongodb", "redis", "snowflake", "bigquery",
        "docker", "kubernetes", "mlflow", "airflow",
        "spark", "pyspark", "kafka", "flask",
        "django", "fastapi", "react", "graphql",
        "opencv", "git", "linux", "excel",
    }

    _DEFAULT_PROJECT_CONCEPTS: Set[str] = {
        "customer segmentation", "demand forecasting", "business intelligence",
        "revenue forecasting", "brain tumor detection", "predictive modeling",
        "machine learning", "deep learning", "time series forecasting",
        "computer vision", "natural language processing",
    }

    _DEFAULT_METHODOLOGIES: Set[str] = {"agile", "scrum", "kanban", "devops", "ci/cd", "oop"}

    # ── Fast-path safe words (small set — the classifier handles the rest) ──
    NON_SKILL_WORDS: Set[str] = {
        "tools", "tool", "libraries", "library", "education", "experience", "summary",
        "skills", "projects", "project", "professional", "expertise",
        "achievements", "achievement", "certifications", "certification",
        # Multi-word section headers common in generated markdown CVs
        "work experience", "professional experience", "professional summary",
        "core skills", "technical skills", "key skills", "tech stack",
        "project experience", "technical projects", "personal projects",
        "employment history", "career objective", "career summary",
    }

    # ── Layer 1: Tech Fingerprint patterns ────────────────────────────────
    _RE_DOT_TECH = re.compile(r"[a-zA-Z0-9]\.[a-zA-Z]")          # Node.js, ASP.NET
    _RE_SPECIAL_CHARS = re.compile(r"[+#]")                        # C++, C#, F#
    _RE_CAMEL_CASE = re.compile(r"[a-z][A-Z]")                     # TensorFlow, FastAPI
    _RE_ALL_CAPS_ABBR = re.compile(r"^[A-Z][A-Z0-9]{1,5}$")       # SQL, AWS, REST
    _RE_DIGIT_MIXED = re.compile(r"(?:[a-zA-Z]\d|\d[a-zA-Z])")    # H2O, OAuth2, S3
    _RE_TECH_HYPHEN = re.compile(                                   # scikit-learn, e2e
        r"^[a-z]+-[a-z]+$", re.IGNORECASE,
    )
    _RE_VERSION_SUFFIX = re.compile(r"\d+(?:\.\d+)*$")             # Python3, ES6

    # ── Layer 2: English morphology patterns ──────────────────────────────
    #    Single purely-alphabetical words ending in these suffixes are
    #    almost certainly NOT technology names.
    _ENGLISH_NOUN_SUFFIXES = (
        "tion", "sion", "ment", "ness", "ity", "ity", "ance", "ence",
        "ship", "dom", "ism", "ery", "ory", "ary", "ure",
    )
    _ENGLISH_ADJ_SUFFIXES = (
        "ous", "ious", "eous", "ive", "ative", "ible", "able",
        "ful", "less", "ical", "ular", "ic",
    )
    _ENGLISH_ADVERB_SUFFIXES = ("ly",)
    _ALL_ENGLISH_SUFFIXES = (
        _ENGLISH_NOUN_SUFFIXES + _ENGLISH_ADJ_SUFFIXES + _ENGLISH_ADVERB_SUFFIXES
    )
    _RE_VERB_FORM = re.compile(
        r"^[a-z]+(ed|ing|ize|ise|ized|ised|ates?|ated|ating)$"
    )
    _RE_PLURAL_NOUN = re.compile(r"^[a-z]{4,}(ies|es|s)$")

    # ── Layer 3: Semantic category patterns ───────────────────────────────
    #    Lightweight regex detectors for known non-skill categories.
    _CATEGORY_PATTERNS: Dict[str, re.Pattern] = {
        "cv_section": re.compile(
            r"(?i)^(responsibilities|qualifications|requirements|objective|overview|"
            r"references|contact|interests|hobbies|summary|profile|about|headline|"
            r"title|section|header|footer|description|details|information|background|"
            r"introduction|competencies|capabilities|attributes|strengths|weaknesses|"
            r"career|portfolio|publications|languages|activities|extracurricular)$"
        ),
        "location_geo": re.compile(
            r"(?i)^(location|city|state|country|region|remote|onsite|on-site|hybrid|"
            r"relocat\w*|based|headquarter\w*|address|worldwide|global|local|domestic|"
            r"international|national|abroad|overseas)$"
        ),
        "time_quantity": re.compile(
            r"(?i)^(years?|months?|weeks?|days?|hours?|daily|weekly|monthly|annually|"
            r"multiple|various|several|numerous|approximate\w*|current|present|ongoing|"
            r"previous|subsequent|consecutive|overall|total|average|minimum|maximum)$"
        ),
        "job_context": re.compile(
            r"(?i)^(team|stakeholders?|clients?|customers?|management|leadership|"
            r"operations?|department|organization|company|enterprise|industry|sector|"
            r"division|unit|role|position|full-time|part-time|contract|permanent|"
            r"temporary|senior|junior|mid-level|entry-level|staff|personnel|workforce|"
            r"supervisor|manager|director|coordinator|specialist|analyst|consultant|"
            r"executive|officer|associate|assistant|administrator|intern|trainee|"
            r"scientist|developer|engineer|architect|programmer|designer|"
            r"self-taught|data|lead|head|principal|freelance|freelancer)$"
        ),
        "soft_skill": re.compile(
            r"(?i)^(communication|teamwork|problem-solving|critical-thinking|"
            r"adaptability|creativity|initiative|motivation|flexibility|attention|"
            r"detail-oriented|self-motivated|multitasking|interpersonal|organizational|"
            r"time-management|prioritization|negotiation|presentation|mentoring|"
            r"decision-making|conflict-resolution|delegation|empathy|patience|"
            r"accountability|reliability|professionalism|enthusiasm|resilience)$"
        ),
        "action_context": re.compile(
            r"(?i)^(responsible|ensuring|providing|maintaining|delivering|supporting|"
            r"handling|overseeing|facilitating|contributing|participating|assisting|"
            r"preparing|conducting|reviewing|analyzing|evaluating|monitoring|"
            r"improving|enhancing|optimizing|streamlining|establishing|executing|"
            r"coordinating|collaborating|communicating|reporting|documenting|"
            r"managing|leading|driving|spearheading|architecting|mentoring)$"
        ),
    }

    OVERCLAIM_PHRASES: Set[str] = {
        "extensive experience", "seasoned professional", "industry veteran",
        "expert professional", "proven leadership", "highly experienced",
    }

    ENTRY_LEVEL_SIGNALS = re.compile(
        r"(?:internship|intern\b|freelance|freelancer|self[- ]?taught|research\s+assistant|teaching\s+assistant|"
        r"capstone|thesis|bootcamp|personal\s+project|side\s+project|open\s+source|volunteer|"
        r"student\s+project|academic\s+project|course\s+project|hackathon)",
        re.IGNORECASE,
    )

    EXPERIENCE_SECTION = re.compile(
        r"professional\s+experience|work\s+experience|employment\s+history", re.IGNORECASE,
    )

    class WeightedIssue:
        def __init__(self, message: str, confidence: float):
            self.message = message
            self.confidence = min(max(confidence, 0.0), 1.0)

    def __init__(self, ontology_path: str = "") -> None:
        self.TECH_ONTOLOGY: Set[str] = set(self._DEFAULT_TECH_ONTOLOGY)
        self.PROJECT_CONCEPTS: Set[str] = set(self._DEFAULT_PROJECT_CONCEPTS)
        self.METHODOLOGIES: Set[str] = set(self._DEFAULT_METHODOLOGIES)

        if ontology_path:
            loaded = self._load_ontology(ontology_path)
            self.TECH_ONTOLOGY |= loaded.get("tech", set())
            self.PROJECT_CONCEPTS |= loaded.get("projects", set())
            self.METHODOLOGIES |= loaded.get("methodologies", set())

    @staticmethod
    def _load_ontology(path: str) -> Dict[str, Set[str]]:
        file_path = Path(path)
        if not file_path.exists():
            logger.warning("Ontology file not found: %s — using defaults only", path)
            return {}
        try:
            raw_text = file_path.read_text(encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed to read ontology file %s: %s", path, exc)
            return {}

        data: Dict[str, Any] = {}
        suffix = file_path.suffix.lower()

        if suffix in (".json", ""):
            try:
                data = json.loads(raw_text)
            except json.JSONDecodeError as exc:
                logger.warning("Invalid JSON in ontology file %s: %s", path, exc)
                if suffix == ".json":
                    return {}

        if not data and suffix in (".yaml", ".yml", ""):
            try:
                import yaml
                data = yaml.safe_load(raw_text) or {}
            except ImportError:
                logger.warning("PyYAML not installed — skipping %s", path)
                return {}
            except Exception as exc:
                logger.warning("Invalid YAML in ontology file %s: %s", path, exc)
                return {}

        if not isinstance(data, dict):
            logger.warning("Ontology file %s did not produce a dict — ignoring", path)
            return {}

        result: Dict[str, Set[str]] = {}
        for key in ("tech", "projects", "methodologies"):
            if key in data and isinstance(data[key], list):
                result[key] = {str(item).lower().strip() for item in data[key] if item}
        loaded_count = sum(len(v) for v in result.values())
        if loaded_count > 0:
            logger.info("Ontology loaded from %s: %d entries", path, loaded_count)
        return result

    def _normalise(self, text: str) -> str:
        return normalise(text)

    def _separator_insensitive_match(self, tok: str, candidate: str) -> bool:
        return self._normalise(tok) == self._normalise(candidate)

    def _acronym_of(self, token: str, candidate: str) -> bool:
        if len(token) < 2 or len(token) > 6:
            return False
        words = re.findall(r"[a-z]+", candidate.lower())
        if not words:
            return False
        return token.lower() == "".join(w[0] for w in words)

    def _fuzzy_match(self, tok: str, allowed_tokens: Set[str], threshold: float = 0.88) -> bool:
        tl = tok.lower().strip()
        norm_tok = self._normalise(tl)
        for candidate in allowed_tokens:
            if self._separator_insensitive_match(tl, candidate):
                return True
            if self._acronym_of(tl, candidate):
                return True
            if norm_tok == self._normalise(candidate):
                return True
            if SequenceMatcher(None, norm_tok, self._normalise(candidate)).ratio() >= threshold:
                return True
        return False

    @staticmethod
    def _strip_markdown(text: str) -> str:
        """Remove markdown formatting that interferes with skill extraction.

        Strips heading markers (# / ## / ###), bold/italic markers (* / **),
        and leading bullet dashes so the downstream tokeniser only sees
        plain text.
        """
        # Remove heading markers at the start of lines: # Title -> Title
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
        # Remove bold / italic markers
        text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
        # Remove leading bullet dashes / asterisks used as list markers
        text = re.sub(r"^\s*[-*]\s+", "", text, flags=re.MULTILINE)
        return text

    def _extract_section(self, text: str, section_names: List[str]) -> str:
        """Extract section content, supporting both plain-text and markdown headers."""
        # Boundary: next plain-text header ("HEADING:") OR markdown header ("## Heading")
        boundary = r"(?=\n#{1,3}\s+|\n[A-Z][A-Z\s]{2,30}:|\Z)"
        pattern = (
            r"(?is)(?:#{1,3}\s+)?(?:" + "|".join(re.escape(s) for s in section_names)
            + r")\s*:?" + r"(.*?)" + boundary
        )
        matches = re.findall(pattern, text)
        return "\n".join(matches) if matches else ""

    # ── Layer 1: Tech Fingerprint ──────────────────────────────────────

    def _tech_fingerprint_score(self, token: str) -> float:
        """Score how 'tech-like' a token looks based on structural signals.

        Returns 0.0 (definitely not tech) to 1.0 (definitely tech).
        Uses the *original* casing of the token for CamelCase detection.
        """
        score = 0.0
        tl = token.strip()
        tl_lower = tl.lower()

        # Dot-separated tech names: Node.js, ASP.NET, Vue.js
        if self._RE_DOT_TECH.search(tl):
            score = max(score, 0.90)
        # Special characters: C++, C#, F#
        if self._RE_SPECIAL_CHARS.search(tl):
            score = max(score, 0.90)
        # CamelCase: TensorFlow, FastAPI, GraphQL
        if self._RE_CAMEL_CASE.search(tl):
            score = max(score, 0.70)
        # ALL-CAPS abbreviation: SQL, AWS, REST, API
        if self._RE_ALL_CAPS_ABBR.match(tl):
            score = max(score, 0.65)
        # Digits mixed with letters: H2O, OAuth2, S3, gRPC
        if self._RE_DIGIT_MIXED.search(tl_lower):
            score = max(score, 0.70)
        # Version suffix: Python3, ES6
        if self._RE_VERSION_SUFFIX.search(tl) and re.search(r"[a-zA-Z]", tl):
            score = max(score, 0.55)

        return score

    # ── Layer 2: English Morphology ───────────────────────────────────

    def _has_english_morphology(self, token: str) -> bool:
        """Return True if the token looks like a regular English word
        (not a technology name) based on its morphological structure.

        Only applies to single, purely-alphabetical words.
        """
        tl = token.lower().strip()
        # Only analyse single purely-alphabetical words
        if " " in tl or not re.fullmatch(r"[a-z]+", tl):
            return False
        if len(tl) <= 3:
            return False
        # Never filter out known tech ontology items
        if tl in self.TECH_ONTOLOGY:
            return False

        # Verb forms: managed, developing, optimized, etc.
        if self._RE_VERB_FORM.match(tl):
            return True

        # Common noun/adjective/adverb suffixes
        for suffix in self._ALL_ENGLISH_SUFFIXES:
            if tl.endswith(suffix) and len(tl) > len(suffix) + 2:
                return True

        # Plural nouns: responsibilities, requirements, strategies
        if self._RE_PLURAL_NOUN.match(tl):
            # Exclude known tech plurals
            singular_candidates = [tl[:-1], tl[:-2], tl[:-3] + "y"]
            for s in singular_candidates:
                if s in self.TECH_ONTOLOGY:
                    return False
            return True

        return False

    # ── Layer 3: Semantic Category Detection ──────────────────────────

    def _matches_non_skill_category(self, token: str) -> bool:
        """Return True if the token belongs to a known non-skill
        semantic category (location, section heading, soft skill, etc.).

        Uses pattern-based detection — not a static word list — so it
        generalises to unseen words within each category.
        """
        tl = token.strip()
        for _category_name, pattern in self._CATEGORY_PATTERNS.items():
            if pattern.match(tl):
                return True
        return False

    # ── Layer 4: Context Analysis ─────────────────────────────────────

    def _is_in_skill_list_context(self, token: str, context_line: str) -> bool:
        """Heuristic: is the token inside a comma/pipe-separated skill
        list or inside a flowing English sentence?

        Returns True  → skill-list context (more likely a real skill)
        Returns False → sentence context (less likely a real skill)
        """
        ctx = context_line.strip()
        if not ctx:
            return True  # no context available — assume skill list

        # Count structural delimiters typical of skill lists
        delimiter_count = len(re.findall(r"[,|;•]", ctx))
        word_count = len(ctx.split())

        # Skill lists have high delimiter-to-word ratio
        if word_count > 0 and delimiter_count / word_count >= 0.15:
            return True

        # Sentence indicators: starts with a verb or contains articles
        sentence_signals = re.search(
            r"(?i)\b(the|a|an|was|were|is|are|has|have|had|been|being|by|for|with|from|into|through)\b",
            ctx,
        )
        verb_start = re.match(
            r"(?i)^(managed|led|developed|designed|built|created|implemented|achieved|delivered|"
            r"coordinated|collaborated|spearheaded|drove|improved|established|maintained|conducted|"
            r"executed|streamlined|optimized|reduced|increased|launched|initiated|performed|applied|"
            r"utilized|leveraged|architected|engineered|orchestrated|deployed|automated|facilitated)",
            ctx,
        )
        if sentence_signals or verb_start:
            return False

        return True  # default: treat as skill list

    # ── Combined classifier ───────────────────────────────────────────

    def _is_likely_skill_token(self, token: str, context: str = "") -> bool:
        """Multi-layer token classifier.

        Decides whether *token* looks like a real technical skill/tool
        rather than regular CV vocabulary.

        Layers (evaluated in order, first decisive layer wins):
          1. Tech fingerprint  — structural signals → likely skill
          2. Morphological     — English suffix patterns → NOT skill
          3. Semantic category — CV vocabulary category → NOT skill
          4. Context analysis  — skill-list vs sentence → decide
        """
        tl = token.lower().strip()
        word_count = len(tl.split())

        # ── Basic filters (unchanged) ──
        if len(tl) <= 2 and self._tech_fingerprint_score(token) < 0.5:
            return False
        if word_count > 5:
            return False
        if tl in self.NON_SKILL_WORDS or tl in self.PROJECT_CONCEPTS or tl in self.METHODOLOGIES:
            return False
        if not re.search(r"[a-z0-9]", tl):
            return False

        # ── Layer 1: If it LOOKS like tech, accept it early ──
        if self._tech_fingerprint_score(token) >= 0.5:
            return True

        # ── Layer 2: Common English morphology → reject ──
        if self._has_english_morphology(tl):
            return False

        # ── Layer 3: Known non-skill semantic category → reject ──
        if self._matches_non_skill_category(tl):
            return False
        # For multi-word tokens, reject if every word is a non-skill word
        if word_count > 1:
            words = tl.split()
            if all(
                self._matches_non_skill_category(w) or w in self.NON_SKILL_WORDS
                for w in words
            ):
                return False

        # ── Layer 4: Context — skill list vs sentence ──
        if context and not self._is_in_skill_list_context(tl, context):
            return False

        return True

    def _extract_skill_tokens(self, text: str) -> Set[str]:
        # Strip markdown formatting before tokenising to avoid treating
        # section headers (## Education, ### Job Title) as skill tokens.
        text = self._strip_markdown(text)

        tokens: Set[str] = set()
        sorted_ontology = sorted(self.TECH_ONTOLOGY, key=len, reverse=True)
        remaining_text = text
        for phrase in sorted_ontology:
            pattern = re.compile(re.escape(phrase), re.IGNORECASE)
            if pattern.search(remaining_text):
                tokens.add(phrase)
                remaining_text = pattern.sub(" " * len(phrase), remaining_text)
        raw_items = re.split(r"[,|\n•;]+", remaining_text)
        for item in raw_items:
            item = item.strip()
            if not item:
                continue
            cleaned = re.sub(r"[^A-Za-z0-9+#.\- ]", "", item).strip()
            if len(cleaned.split()) > 5:
                continue
            parts = re.split(r"[/|]", cleaned)
            for part in parts:
                part = part.strip()
                if part and self._is_likely_skill_token(part, context=item):
                    tokens.add(part)
        return tokens

    def validate(self, cv_text: str, profile: UserProfile, jd_context: Optional[JDContext] = None) -> GuardResult:
        issues: List[str] = []
        weighted: List[HallucinationGuard.WeightedIssue] = []
        profile_context = profile.to_context().lower()
        authorised: Set[str] = {t.lower().strip() for t in profile.authorised_tokens()}
        jd_tokens: Set[str] = {t.lower().strip() for t in (jd_context.keywords if jd_context else [])}
        allowed: Set[str] = authorised | jd_tokens
        all_known: Set[str] = allowed | self.TECH_ONTOLOGY

        skills_text = self._extract_section(cv_text, [
            "SKILLS", "TECHNICAL SKILLS", "TECH STACK", "TOOLS",
            "TECHNOLOGIES", "FRAMEWORKS", "CORE SKILLS", "KEY SKILLS",
        ])
        projects_text = self._extract_section(cv_text, [
            "PROJECTS", "PROJECT EXPERIENCE", "TECHNICAL PROJECTS",
            "PORTFOLIO", "PERSONAL PROJECTS",
        ])
        validation_text = f"{skills_text}\n{projects_text}"

        # Fake experience detection
        has_exp_section = bool(self.EXPERIENCE_SECTION.search(cv_text))
        
        # Stricter enforcement: if the profile strictly has no experience, 
        # ANY presence of a Professional Experience section is a hallucination.
        if profile.has_no_experience and has_exp_section:
            weighted.append(self.WeightedIssue(
                "Professional experience section invented despite candidate having no experience",
                confidence=0.99))
        else:
            # Fallback heuristic for profiles that didn't strictly trigger has_no_experience 
            # but look like they shouldn't have an experience section.
            no_exp_signals = (
                "no experience" in profile_context
                or re.search(r"\b0\s*years?\b", profile_context)
                or re.search(r"fresh\s*(graduate|grad)\b", profile_context, re.I)
            )
            has_entry_level = bool(self.ENTRY_LEVEL_SIGNALS.search(cv_text))
            if no_exp_signals and has_exp_section and not has_entry_level:
                weighted.append(self.WeightedIssue(
                    "Professional experience section invented despite candidate having no experience",
                    confidence=0.90))

        # Overclaim detection
        cv_lower = cv_text.lower()
        for phrase in self.OVERCLAIM_PHRASES:
            if phrase in cv_lower:
                weighted.append(self.WeightedIssue(f"Potential overclaim detected: '{phrase}'", confidence=0.90))

        # Company validation
        cv_companies = re.findall(r"\bat\s+([A-Z][a-zA-Z\s&]{2,30}(?:Inc|LLC|Ltd|Corp|Co)?)", cv_text)
        for company in cv_companies:
            c_norm = company.strip().lower()
            if len(c_norm) <= 3:
                continue
            words = set(re.findall(r"[a-z]{3,}", c_norm))
            if not words.intersection(allowed):
                weighted.append(self.WeightedIssue(f"Unverified company in profile: '{company.strip()}'", confidence=0.70))

        # Percentage validation
        pcts = [int(m) for m in re.findall(r"(\d{1,3})%", cv_text)]
        flagged_pcts = [p for p in pcts if p > self.MAX_PERCENTAGE]
        if flagged_pcts:
            profile_pcts = [int(m) for m in re.findall(r"(\d{1,3})%", profile.to_context())]
            novel_pcts = [p for p in flagged_pcts if p not in profile_pcts]
            if novel_pcts:
                weighted.append(self.WeightedIssue(f"Implausible percentage metric(s) not in profile: {novel_pcts}", confidence=0.85))

        # Revenue validation
        billions = re.findall(r"\$(\d+(?:\.\d+)?)\s*[Bb]illion", cv_text)
        for b in billions:
            if float(b) > self.MAX_REVENUE_B and str(b) not in profile.to_context():
                weighted.append(self.WeightedIssue(f"Implausibly large revenue figure: ${b}B", confidence=0.88))

        # Multiplier validation
        multipliers = re.findall(r"(\d+(?:\.\d+)?)[xX]\s+(?:improvement|faster|increase)", cv_text)
        for m in multipliers:
            if float(m) > self.MAX_MULTIPLIER and str(m) not in profile.to_context():
                weighted.append(self.WeightedIssue(f"Implausible multiplier claim: {m}x", confidence=0.88))

        # Skill validation
        cv_skill_tokens = self._extract_skill_tokens(validation_text)
        for tok in cv_skill_tokens:
            tl = tok.lower().strip()
            if tl in self.PROJECT_CONCEPTS or tl in self.METHODOLOGIES or tl in allowed:
                continue
            if self._normalise(tl) in {self._normalise(a) for a in allowed}:
                continue
            if self._fuzzy_match(tok, all_known, threshold=0.88):
                continue

            has_special = bool(re.search(r"[0-9+#]", tl))
            is_multiword = " " in tl
            char_ratio = len(re.sub(r"[^a-z]", "", tl)) / max(len(tl), 1)
            tech_score = self._tech_fingerprint_score(tok)

            if tech_score >= 0.7:
                # High tech fingerprint → very likely a real skill mention
                confidence = 0.85
            elif has_special and not is_multiword:
                confidence = 0.80
            elif not is_multiword and char_ratio > 0.8:
                # Single alphabetical word with low tech score → lower confidence
                confidence = 0.55 if tech_score < 0.3 else 0.72
            elif is_multiword:
                confidence = 0.55
            else:
                confidence = 0.65
            weighted.append(self.WeightedIssue(f"Skill not present in profile: '{tok}'", confidence=confidence))

        # Confidence filtering
        CONFIDENCE_THRESHOLD = 0.65
        for wi in weighted:
            if wi.confidence >= CONFIDENCE_THRESHOLD:
                issues.append(wi.message)
        issues = list(dict.fromkeys(issues))
        return GuardResult(passed=len(issues) == 0, issues=issues)


_guard = HallucinationGuard()

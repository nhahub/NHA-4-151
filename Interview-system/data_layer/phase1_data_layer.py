"""
Phase 1 — Foundation & Data Layer
AI Voice Interview System

Stack:
  - Pandas  : structured CSV → DataFrame store (replaces PostgreSQL)
  - ChromaDB: vector search over questions, rubrics, role descriptions
  - Redis   : LangGraph checkpoints + query result cache (optional, degrades gracefully)

Usage:
  python phase1_data_layer.py              # boot store + run smoke tests
  python phase1_data_layer.py --validate   # run full validation suite
  python phase1_data_layer.py --shell      # drop into interactive query shell

Install:
  pip install chromadb pandas sentence-transformers redis pyarrow
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

# ── Optional deps: degrade gracefully when not installed ──────────────────────
try:
    import chromadb
    from chromadb.utils import embedding_functions
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    logging.warning("chromadb not installed — vector search disabled. pip install chromadb")

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("redis not installed — caching disabled. pip install redis")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("phase1")

# ── Constants ─────────────────────────────────────────────────────────────────
_THIS_FILE = Path(__file__).resolve()
DATA_DIR    = _THIS_FILE.parent.parent / "data"
CHROMA_DIR  = _THIS_FILE.parent.parent / ".chroma"
PARQUET_DIR = _THIS_FILE.parent.parent / ".parquet"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 384-dim, fast, good quality

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_TTL  = 3600  # 1 hour

CSV_FILES = {
    "skill_hierarchy":     "skill_hierarchy.csv",
    "domain_rubrics":      "domain_rubrics.csv",
    "questions_master":    "questions_master.csv",
    "role_expectations":   "role_expectations.csv",
    "calibration_examples":"answer_calibration.csv",
    "question_chains":     "question_chains.csv",
}

# ── Column rename maps (CSV actual → script expected) ─────────────────────────
COLUMN_RENAMES: dict[str, dict[str, str]] = {
    "skill_hierarchy": {
        "relationship_type": "domain",
        "skill": "sub_skill",
        "assessment_gate": "description",
        "profile_analyzer_weight": "weight",
    },
    "domain_rubrics": {
        "evaluation_criteria": "competency",
        "level_3_criteria": "criteria",
        "evaluator_notes": "example_evidence",
    },
    "questions_master": {
        "target_role": "role_id",
        "primary_skill": "sub_skill",
        "question_type": "phase",
    },
    "role_expectations": {
        "expectation_id": "role_id",
        "role": "title",
        "seniority_level": "seniority",
        "interview_domains": "domain",
        "core_skills_required": "sub_skill",
        "orchestrator_weight": "weight",
        "min_acceptable_score": "min_passing_score",
        "skill_depth": "description",
    },
    "calibration_examples": {
        "primary_skill": "sub_skill",
        "answer_text": "transcript_excerpt",
        "score_rationale": "rationale",
    },
    "question_chains": {
        "primary_skill": "sub_skill",
        "chain_question": "question_text",
    },
}

# ── DataFrame schemas (dtypes + required columns) ─────────────────────────────
SCHEMAS: dict[str, dict] = {
    "skill_hierarchy": {
        "required": ["domain", "sub_skill", "description", "weight"],
        "dtypes":   {"weight": float},
    },
    "domain_rubrics": {
        "required": ["domain", "competency", "criteria"],
        "dtypes":   {},
    },
    "questions_master": {
        "required": ["question_id", "role_id", "domain", "sub_skill", "phase",
                     "difficulty", "question_text"],
        "dtypes":   {"answered": bool},
        "defaults": {"answered": False},
    },
    "role_expectations": {
        "required": ["role_id", "title", "seniority", "domain", "sub_skill",
                     "weight", "min_passing_score", "description"],
        "dtypes":   {"weight": float, "min_passing_score": float},
    },
    "calibration_examples": {
        "required": ["calibration_id", "domain", "sub_skill", "score",
                     "transcript_excerpt", "rationale"],
        "dtypes":   {"score": int},
    },
    "question_chains": {
        "required": ["chain_id", "base_question_id", "chain_type", "question_text",
                     "trigger_condition", "difficulty", "domain", "sub_skill",
                     "order_in_chain", "max_depth"],
        "dtypes":   {"order_in_chain": int, "max_depth": int,
                     "adaptive_weight": float, "scoring_threshold": float},
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Pandas Store
# ─────────────────────────────────────────────────────────────────────────────

class PandasStore:
    """
    Loads all CSVs into typed DataFrames, validates schemas, and exposes
    fast indexed query helpers used by LangGraph nodes at runtime.
    """

    def __init__(self, data_dir: Path = DATA_DIR) -> None:
        self.data_dir = data_dir
        self._frames: dict[str, pd.DataFrame] = {}
        self._load_all()
        self._build_indexes()

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load_all(self) -> None:
        t0 = time.perf_counter()
        for name, filename in CSV_FILES.items():
            path = self.data_dir / filename
            if not path.exists():
                raise FileNotFoundError(
                    f"CSV not found: {path}\n"
                    f"Expected at: {path.resolve()}"
                )
            df = pd.read_csv(path)
            # Rename CSV columns to match script expectations
            renames = COLUMN_RENAMES.get(name, {})
            df = df.rename(columns=renames)
            df = self._validate_and_cast(name, df)
            self._frames[name] = df
            log.info("Loaded %-22s  %d rows", name, len(df))
        log.info("All CSVs loaded in %.1f ms", (time.perf_counter() - t0) * 1000)

    def _validate_and_cast(self, name: str, df: pd.DataFrame) -> pd.DataFrame:
        schema = SCHEMAS[name]

        # Apply defaults before validation
        for col, default in schema.get("defaults", {}).items():
            if col not in df.columns:
                df[col] = default

        # Check required columns
        missing = [c for c in schema["required"] if c not in df.columns]
        if missing:
            raise ValueError(f"[{name}] Missing required columns: {missing}")

        # Cast dtypes
        for col, dtype in schema.get("dtypes", {}).items():
            if col in df.columns:
                try:
                    df[col] = df[col].astype(dtype)
                except Exception as exc:
                    raise ValueError(f"[{name}] Cannot cast '{col}' to {dtype}: {exc}") from exc

        # Strip whitespace from string cols
        str_cols = df.select_dtypes(include="object").columns
        for col in str_cols:
            df[col] = df[col].str.strip()

        return df

    def _build_indexes(self) -> None:
        """Set meaningful indexes for O(1) lookup patterns."""
        self._frames["skill_hierarchy"] = (
            self._frames["skill_hierarchy"].set_index(["domain", "sub_skill"])
        )
        self._frames["domain_rubrics"] = (
            self._frames["domain_rubrics"].set_index(["domain"])
        )
        self._frames["questions_master"] = (
            self._frames["questions_master"].set_index("question_id")
        )
        self._frames["role_expectations"] = (
            self._frames["role_expectations"].set_index(["role_id"])
        )
        self._frames["question_chains"] = (
            self._frames["question_chains"].set_index("chain_id")
        )
        log.info("Indexes built on all DataFrames")

    # ── Public Query API ──────────────────────────────────────────────────────

    def get_role_expectations(self, role_id: str) -> pd.DataFrame:
        """Return all skill weights and thresholds for a role."""
        df = self._frames["role_expectations"]
        try:
            result = df.loc[[role_id]]
        except KeyError:
            raise KeyError(f"Unknown role_id: '{role_id}'")
        if result.empty:
            raise KeyError(f"Unknown role_id: '{role_id}'")
        return result.reset_index()

    def get_skill_info(self, domain: str, sub_skill: str) -> dict:
        """Return skill metadata dict for a domain/sub_skill pair."""
        df = self._frames["skill_hierarchy"]
        try:
            row = df.loc[(domain, sub_skill)]
            return row.to_dict()
        except KeyError:
            raise KeyError(f"Skill not found: domain='{domain}', sub_skill='{sub_skill}'")

    def filter_questions(
        self,
        role_id: str,
        phase: str,
        domain: str | None = None,
        sub_skill: str | None = None,
        difficulty: str | None = None,
        exclude_ids: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Return unanswered questions matching filters. Used by question_generator node.

        The 'phase' param maps to the 'question_type' column (renamed to 'phase').
        If no results match the exact phase, we fall back to filtering by domain only,
        since interview phases (Technical, Behavioral, System Design, etc.) map to
        the 'domain' column in questions_master.
        """
        df = self._frames["questions_master"].reset_index()
        base_mask = (df["role_id"] == role_id) & (~df["answered"])
        if exclude_ids:
            base_mask &= ~df["question_id"].isin(exclude_ids)

        # Try exact phase match first
        mask = base_mask & (df["phase"] == phase)
        if domain:
            mask &= df["domain"] == domain
        if sub_skill:
            mask &= df["sub_skill"] == sub_skill
        if difficulty:
            mask &= df["difficulty"] == difficulty

        result = df[mask]

        # If no results with phase filter, try matching domain column instead
        # (interview phases like "System Design" map to the domain column)
        if result.empty and domain:
            mask = base_mask & (df["domain"] == domain)
            if sub_skill:
                mask &= df["sub_skill"] == sub_skill
            if difficulty:
                mask &= df["difficulty"] == difficulty
            result = df[mask]

        # If still empty, try just domain = phase (e.g. phase="Behavioral" matches domain)
        if result.empty:
            mask = base_mask & (df["domain"] == phase)
            if sub_skill:
                mask &= df["sub_skill"] == sub_skill
            result = df[mask]

        return result.copy()

    def get_rubric(self, domain: str, competency: str) -> pd.DataFrame:
        """Return full rubric (all score levels) for a domain/competency."""
        df = self._frames["domain_rubrics"]
        try:
            result = df.loc[[domain]]
            if competency:
                mask = result["competency"].str.contains(competency, case=False, na=False)
                filtered = result[mask]
                if not filtered.empty:
                    result = filtered
            return result.reset_index()
        except KeyError:
            raise KeyError(f"No rubric for domain='{domain}', competency='{competency}'")

    def get_calibration_examples(
        self, domain: str, sub_skill: str, n: int = 3
    ) -> pd.DataFrame:
        """Return n calibration examples for a domain/sub_skill pair."""
        df = self._frames["calibration_examples"]
        result = df[(df["domain"] == domain) & (df["sub_skill"] == sub_skill)]
        return result.head(n).copy()

    def mark_question_answered(self, question_id: str) -> None:
        """Mutate the in-memory DataFrame to mark a question answered."""
        df = self._frames["questions_master"]
        if question_id not in df.index:
            raise KeyError(f"Unknown question_id: '{question_id}'")
        df.at[question_id, "answered"] = True

    def get_coverage_map(self, role_id: str) -> dict[str, dict[str, float]]:
        """
        Build initial coverage map: {domain: {sub_skill: 0.0}} for a role.
        Used by session_init LangGraph node to seed InterviewState.

        Handles pipe-separated domain and sub_skill values from role_expectations CSV.
        E.g. domain="Technical|Behavioral" → separate entries for each.
        """
        expectations = self.get_role_expectations(role_id)
        coverage: dict[str, dict[str, float]] = {}
        for _, row in expectations.iterrows():
            raw_domain = str(row["domain"])
            raw_skill = str(row["sub_skill"])

            # Split pipe-separated values
            domains = [d.strip() for d in raw_domain.split("|")] if "|" in raw_domain else [raw_domain.strip()]
            skills = [s.strip() for s in raw_skill.split("|")] if "|" in raw_skill else [raw_skill.strip()]

            # Cross-product: each domain gets all sub_skills
            for domain in domains:
                for sub_skill in skills:
                    coverage.setdefault(domain, {})[sub_skill] = 0.0
        return coverage

    def get_chain_questions(self, base_question_id: str) -> pd.DataFrame:
        """
        Return all follow-up chain questions for a base question,
        ordered by order_in_chain. Used by Phase 2 chain_follow_up node.
        """
        df = self._frames["question_chains"].reset_index()
        result = df[df["base_question_id"] == base_question_id]
        return result.sort_values("order_in_chain").copy()

    def get_chain_question_at(
        self, base_question_id: str, order: int
    ) -> dict | None:
        """
        Return a single chain question at a specific order position.
        Returns None if no chain question exists at that order.
        """
        df = self.get_chain_questions(base_question_id)
        match = df[df["order_in_chain"] == order]
        if match.empty:
            return None
        return match.iloc[0].to_dict()

    def persist_parquet(self, out_dir: Path = PARQUET_DIR) -> None:
        """Snapshot all DataFrames to Parquet for audit / restart."""
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, df in self._frames.items():
            path = out_dir / f"{name}.parquet"
            df.reset_index().to_parquet(path, index=False)
            log.info("Saved %s → %s", name, path)

    @property
    def frames(self) -> dict[str, pd.DataFrame]:
        return self._frames


# ─────────────────────────────────────────────────────────────────────────────
# ChromaDB Store
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ChromaStore:
    """
    Manages three ChromaDB collections:
      • questions         — embed question_text, metadata: role_id, phase, domain, sub_skill, difficulty
      • rubrics           — embed criteria, metadata: domain, competency, level, score
      • role_expectations — embed description, metadata: role_id, domain, sub_skill, seniority

    Populated from PandasStore DataFrames at startup.
    Provides semantic search methods used by LangGraph nodes.
    """

    chroma_dir: Path = CHROMA_DIR
    _client: Any = field(default=None, repr=False, init=False)
    _ef: Any = field(default=None, repr=False, init=False)
    _collections: dict[str, Any] = field(default_factory=dict, repr=False, init=False)

    def __post_init__(self) -> None:
        if not CHROMA_AVAILABLE:
            raise RuntimeError("chromadb not installed. pip install chromadb")
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.chroma_dir))
        self._ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        self._init_collections()
        log.info("ChromaDB ready at %s", self.chroma_dir)

    def _init_collections(self) -> None:
        for name in ("questions", "rubrics", "role_expectations"):
            self._collections[name] = self._client.get_or_create_collection(
                name=name,
                embedding_function=self._ef,
                metadata={"hnsw:space": "cosine"},
            )

    # ── Population ────────────────────────────────────────────────────────────

    def populate_from_store(self, store: PandasStore) -> None:
        """Upsert all documents from PandasStore into ChromaDB collections."""
        t0 = time.perf_counter()
        self._populate_questions(store)
        self._populate_rubrics(store)
        self._populate_role_expectations(store)
        log.info("ChromaDB populated in %.1f ms", (time.perf_counter() - t0) * 1000)

    def _populate_questions(self, store: PandasStore) -> None:
        df = store.frames["questions_master"].reset_index()
        col = self._collections["questions"]
        ids, docs, metas = [], [], []
        for _, row in df.iterrows():
            ids.append(row["question_id"])
            docs.append(row["question_text"])
            metas.append({
                "role_id":   str(row["role_id"]),
                "phase":     str(row["phase"]),
                "domain":    str(row["domain"]),
                "sub_skill": str(row["sub_skill"]),
                "difficulty":str(row["difficulty"]),
            })
        col.upsert(ids=ids, documents=docs, metadatas=metas)
        log.info("questions collection: %d documents", col.count())

    def _populate_rubrics(self, store: PandasStore) -> None:
        df = store.frames["domain_rubrics"].reset_index()
        col = self._collections["rubrics"]
        ids, docs, metas = [], [], []
        for _, row in df.iterrows():
            uid = f"{row['domain']}__{row['competency']}"
            ids.append(uid)
            docs.append(str(row["criteria"]))
            metas.append({
                "domain":     str(row["domain"]),
                "competency": str(row["competency"]),
            })
        col.upsert(ids=ids, documents=docs, metadatas=metas)
        log.info("rubrics collection: %d documents", col.count())

    def _populate_role_expectations(self, store: PandasStore) -> None:
        df = store.frames["role_expectations"].reset_index()
        col = self._collections["role_expectations"]
        ids, docs, metas = [], [], []
        for _, row in df.iterrows():
            uid = f"{row['role_id']}__{row['title']}__{row['domain']}"
            ids.append(uid)
            docs.append(str(row["description"]))
            metas.append({
                "role_id":   str(row["role_id"]),
                "domain":    str(row["domain"]),
                "sub_skill": str(row["sub_skill"]),
                "seniority": str(row["seniority"]),
                "weight":    float(row["weight"]),
            })
        col.upsert(ids=ids, documents=docs, metadatas=metas)
        log.info("role_expectations collection: %d documents", col.count())

    # ── Query API ─────────────────────────────────────────────────────────────

    def find_questions(
        self,
        query: str,
        role_id: str,
        phase: str,
        n_results: int = 5,
        domain: str | None = None,
    ) -> list[dict]:
        """
        Semantic search for questions relevant to a candidate's current state.
        Hard-filters by role_id and phase; optionally by domain.
        Used by question_generator node.
        """
        where: dict[str, Any] = {
            "$and": [
                {"role_id": {"$eq": role_id}},
                {"phase":   {"$eq": phase}},
            ]
        }
        if domain:
            where["$and"].append({"domain": {"$eq": domain}})

        results = self._collections["questions"].query(
            query_texts=[query],
            n_results=min(n_results, self._collections["questions"].count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        return self._unpack_results(results)

    def find_rubric_criteria(
        self, candidate_answer: str, domain: str, competency: str, n_results: int = 3
    ) -> list[dict]:
        """
        Find rubric levels whose criteria text is semantically closest to a
        candidate's answer. Used by answer_evaluator node to build scoring prompt.
        """
        results = self._collections["rubrics"].query(
            query_texts=[candidate_answer],
            n_results=min(n_results, self._collections["rubrics"].count()),
            where={"domain": {"$eq": domain}},
            include=["documents", "metadatas", "distances"],
        )
        return self._unpack_results(results)

    def find_role_context(self, query: str, role_id: str, n_results: int = 3) -> list[dict]:
        """
        Semantic search over role expectation descriptions.
        Used by session_init node to load the most relevant skill expectations.
        """
        results = self._collections["role_expectations"].query(
            query_texts=[query],
            n_results=min(n_results, self._collections["role_expectations"].count()),
            where={"role_id": {"$eq": role_id}},
            include=["documents", "metadatas", "distances"],
        )
        return self._unpack_results(results)

    @staticmethod
    def _unpack_results(results: dict) -> list[dict]:
        """Flatten ChromaDB result dict into a clean list of dicts."""
        out = []
        if not results or not results.get("ids"):
            return out
        ids        = results["ids"][0]
        docs       = results["documents"][0]
        metas      = results["metadatas"][0]
        distances  = results.get("distances", [[None] * len(ids)])[0]
        for qid, doc, meta, dist in zip(ids, docs, metas, distances):
            out.append({"id": qid, "document": doc, "metadata": meta, "distance": dist})
        return out

    def collection_counts(self) -> dict[str, int]:
        return {name: col.count() for name, col in self._collections.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Redis Cache Layer
# ─────────────────────────────────────────────────────────────────────────────

class CacheLayer:
    """
    Thin Redis wrapper. Used to cache:
      - Role expectation DataFrames (serialised to JSON)
      - ChromaDB query results
      - LangGraph checkpoint metadata (handled separately by LangGraph itself)

    Falls back to a no-op dict cache if Redis is unavailable.
    """

    def __init__(self) -> None:
        self._redis: Any = None
        self._fallback: dict[str, tuple[str, float]] = {}
        if REDIS_AVAILABLE:
            try:
                r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT,
                                decode_responses=True, socket_connect_timeout=2)
                r.ping()
                self._redis = r
                log.info("Redis connected at %s:%s", REDIS_HOST, REDIS_PORT)
            except Exception as exc:
                log.warning("Redis unavailable (%s) — using in-process fallback cache", exc)
        else:
            log.info("Redis not installed — using in-process fallback cache")

    def _make_key(self, namespace: str, **kwargs: Any) -> str:
        parts = ":".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
        digest = hashlib.md5(parts.encode()).hexdigest()[:8]
        return f"iv:{namespace}:{digest}"

    def get(self, namespace: str, **kwargs: Any) -> Any | None:
        key = self._make_key(namespace, **kwargs)
        if self._redis:
            raw = self._redis.get(key)
            return json.loads(raw) if raw else None
        val, exp = self._fallback.get(key, (None, 0))
        if val and time.time() < exp:
            return json.loads(val)
        return None

    def set(self, namespace: str, value: Any, ttl: int = REDIS_TTL, **kwargs: Any) -> None:
        key = self._make_key(namespace, **kwargs)
        serialised = json.dumps(value, default=str)
        if self._redis:
            self._redis.setex(key, ttl, serialised)
        else:
            self._fallback[key] = (serialised, time.time() + ttl)

    def invalidate(self, namespace: str, **kwargs: Any) -> None:
        key = self._make_key(namespace, **kwargs)
        if self._redis:
            self._redis.delete(key)
        else:
            self._fallback.pop(key, None)


# ─────────────────────────────────────────────────────────────────────────────
# DataLayer — unified facade used by LangGraph nodes
# ─────────────────────────────────────────────────────────────────────────────

class DataLayer:
    """
    Single entry point for all data access in Phase 1.
    LangGraph nodes import and call this object — they never touch
    PandasStore or ChromaStore directly.

    Example (inside a LangGraph node):
        from phase1_data_layer import data_layer

        expectations = data_layer.get_role_expectations("eng_senior")
        questions    = data_layer.find_questions(
                           "system design distributed", "eng_senior", "technical"
                       )
    """

    def __init__(self) -> None:
        log.info("Booting DataLayer …")
        self.pandas  = PandasStore()
        self.cache   = CacheLayer()
        self.chroma: ChromaStore | None = None
        if CHROMA_AVAILABLE:
            self.chroma = ChromaStore()
            self.chroma.populate_from_store(self.pandas)
        log.info("DataLayer ready")

    # ── Structured queries (Pandas) ───────────────────────────────────────────

    def get_role_expectations(self, role_id: str) -> list[dict]:
        """Cached fetch of role expectations as a list of dicts."""
        cached = self.cache.get("role_exp", role_id=role_id)
        if cached:
            return cached
        df = self.pandas.get_role_expectations(role_id)
        result = df.to_dict(orient="records")
        self.cache.set("role_exp", result, role_id=role_id)
        return result

    def get_coverage_map(self, role_id: str) -> dict[str, dict[str, float]]:
        return self.pandas.get_coverage_map(role_id)

    def filter_questions(
        self,
        role_id: str,
        phase: str,
        domain: str | None = None,
        sub_skill: str | None = None,
        difficulty: str | None = None,
        exclude_ids: list[str] | None = None,
    ) -> list[dict]:
        """
        Filter questions by structured criteria (no vector search).
        Fallback used when ChromaDB is unavailable or for exact matching.
        """
        df = self.pandas.filter_questions(
            role_id=role_id, phase=phase, domain=domain,
            sub_skill=sub_skill, difficulty=difficulty, exclude_ids=exclude_ids,
        )
        return df.to_dict(orient="records")

    def get_rubric(self, domain: str, competency: str) -> list[dict]:
        """Full rubric for a domain/competency pair (all score levels)."""
        cached = self.cache.get("rubric", domain=domain, competency=competency)
        if cached:
            return cached
        df = self.pandas.get_rubric(domain, competency)
        result = df.to_dict(orient="records")
        self.cache.set("rubric", result, domain=domain, competency=competency)
        return result

    def get_calibration_examples(
        self, domain: str, sub_skill: str, n: int = 3
    ) -> list[dict]:
        df = self.pandas.get_calibration_examples(domain, sub_skill, n)
        return df.to_dict(orient="records")

    def mark_question_answered(self, question_id: str) -> None:
        self.pandas.mark_question_answered(question_id)

    def get_chain_questions(self, base_question_id: str) -> list[dict]:
        """Return all follow-up chain questions for a base question."""
        df = self.pandas.get_chain_questions(base_question_id)
        return df.to_dict(orient="records")

    def get_chain_question_at(
        self, base_question_id: str, order: int
    ) -> dict | None:
        """Return a single chain question at a specific order."""
        return self.pandas.get_chain_question_at(base_question_id, order)

    # ── Vector queries (ChromaDB) ─────────────────────────────────────────────

    def find_questions(
        self,
        query: str,
        role_id: str,
        phase: str,
        n_results: int = 5,
        domain: str | None = None,
    ) -> list[dict]:
        """
        Semantic question search. Falls back to filter_questions() if ChromaDB
        is unavailable or returns no results.
        """
        if self.chroma:
            cache_key = hashlib.md5(
                f"{query}{role_id}{phase}{domain}".encode()
            ).hexdigest()[:12]
            cached = self.cache.get("q_search", key=cache_key)
            if cached:
                return cached
            results = self.chroma.find_questions(
                query=query, role_id=role_id, phase=phase,
                n_results=n_results, domain=domain,
            )
            if results:
                self.cache.set("q_search", results, key=cache_key)
                return results
        # Fallback to Pandas filter
        log.debug("find_questions: falling back to Pandas filter")
        return self.filter_questions(
            role_id=role_id, phase=phase, domain=domain
        )[:n_results]

    def find_rubric_criteria(
        self, candidate_answer: str, domain: str, competency: str
    ) -> list[dict]:
        """
        Semantic rubric matching against a candidate answer.
        Falls back to full rubric fetch if ChromaDB unavailable.
        """
        if self.chroma:
            return self.chroma.find_rubric_criteria(candidate_answer, domain, competency)
        return self.get_rubric(domain, competency)

    def find_role_context(self, query: str, role_id: str, n_results: int = 3) -> list[dict]:
        """Semantic search over role expectation descriptions."""
        if self.chroma:
            return self.chroma.find_role_context(query, role_id, n_results)
        return self.get_role_expectations(role_id)[:n_results]

    def health(self) -> dict:
        """Return health status dict — used by CI smoke test."""
        status: dict[str, Any] = {
            "pandas_frames": {k: len(df) for k, df in self.pandas.frames.items()},
            "redis": "connected" if (self.cache._redis and self.cache._redis.ping()) else "fallback",
            "chroma": self.chroma.collection_counts() if self.chroma else "unavailable",
        }
        return status


# ─────────────────────────────────────────────────────────────────────────────
# Validation Suite (Gate 1)
# ─────────────────────────────────────────────────────────────────────────────

def run_validation(layer: DataLayer) -> bool:
    """
    Gate 1 validation: all queries must return correct results within
    latency thresholds. Returns True if all checks pass.
    """
    results: list[tuple[str, bool, str]] = []

    def check(label: str, fn, *args, max_ms: float = 30, **kwargs):
        t0 = time.perf_counter()
        try:
            result = fn(*args, **kwargs)
            elapsed = (time.perf_counter() - t0) * 1000
            ok = result is not None and (len(result) > 0 if hasattr(result, "__len__") else True)
            latency_ok = elapsed <= max_ms
            status = ok and latency_ok
            note = f"{elapsed:.1f} ms" + ("" if latency_ok else f" EXCEEDS {max_ms} ms limit")
            results.append((label, status, note))
            sym = "✓" if status else "✗"
            log.info("  %s  %-45s  %s", sym, label, note)
        except Exception as exc:
            results.append((label, False, str(exc)))
            log.error("  ✗  %-45s  ERROR: %s", label, exc)

    log.info("\n── Gate 1 Validation ─────────────────────────────────")

    # Pandas queries (target <5 ms)
    check("get_role_expectations(EXP-SOFT-SEN)",
          layer.get_role_expectations, "EXP-SOFT-SEN", max_ms=5)
    check("get_coverage_map(EXP-SOFT-SEN)",
          layer.get_coverage_map, "EXP-SOFT-SEN", max_ms=5)
    check("filter_questions(Software Engineer, Technical)",
          layer.filter_questions, "Software Engineer", "Technical", max_ms=5)
    check("filter_questions with domain filter",
          layer.filter_questions, "Software Engineer", "Technical",
          domain="Technical", max_ms=5)
    check("get_rubric(Technical, Correctness)",
          layer.get_rubric, "Technical", "Correctness", max_ms=5)
    check("get_calibration_examples(Technical, Python)",
          layer.get_calibration_examples, "Technical", "Python", max_ms=5)

    # ChromaDB vector queries (target <30 ms after warm-up)
    if layer.chroma:
        check("find_questions semantic search",
              layer.find_questions,
              "design a scalable distributed system", "Software Engineer", "Technical", max_ms=200)
        check("find_rubric_criteria semantic match",
              layer.find_rubric_criteria,
              "I would use consistent hashing with a read replica per shard",
              "Technical", "Correctness", max_ms=200)
        check("find_role_context semantic search",
              layer.find_role_context,
              "architectural decision making", "EXP-SOFT-SEN", max_ms=200)

    # Referential integrity checks
    log.info("\n── Referential Integrity ─────────────────────────────")
    qdf = layer.pandas.frames["questions_master"].reset_index()
    rdf = layer.pandas.frames["role_expectations"].reset_index()
    valid_roles = set(rdf["role_id"].unique())
    bad_roles = set(qdf["role_id"].unique()) - valid_roles
    ok = len(bad_roles) == 0
    results.append(("questions → role_ids valid", ok, str(bad_roles) if bad_roles else "all valid"))
    sym = "✓" if ok else "✗"
    log.info("  %s  questions → role_ids valid: %s",
             sym, "all valid" if ok else f"INVALID: {bad_roles}")

    # Summary
    passed = sum(1 for _, ok, _ in results if ok)
    total  = len(results)
    log.info("\n── Result: %d/%d checks passed ─────────────────────────", passed, total)
    return passed == total


# ─────────────────────────────────────────────────────────────────────────────
# Interactive Shell
# ─────────────────────────────────────────────────────────────────────────────

def interactive_shell(layer: DataLayer) -> None:
    import code
    banner = (
        "\n╔══════════════════════════════════════════════════╗\n"
        "║  Phase 1 DataLayer — interactive shell           ║\n"
        "╠══════════════════════════════════════════════════╣\n"
        "║  layer.get_role_expectations('eng_senior')       ║\n"
        "║  layer.filter_questions('eng_senior','technical')║\n"
        "║  layer.find_questions('system design', ...)      ║\n"
        "║  layer.get_rubric('engineering','system_design') ║\n"
        "║  layer.health()                                  ║\n"
        "╚══════════════════════════════════════════════════╝\n"
    )
    code.interact(banner=banner, local={"layer": layer, "pd": pd})


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton (imported by LangGraph nodes)
# ─────────────────────────────────────────────────────────────────────────────

# Lazily instantiated so importing the module doesn't trigger full boot.
_layer_singleton: DataLayer | None = None

def get_data_layer() -> DataLayer:
    global _layer_singleton
    if _layer_singleton is None:
        _layer_singleton = DataLayer()
    return _layer_singleton

# Convenience alias for direct import:  from phase1_data_layer import data_layer
data_layer = None  # populated on first import via get_data_layer() if needed


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 — Data Layer")
    parser.add_argument("--validate", action="store_true",
                        help="Run Gate 1 validation suite")
    parser.add_argument("--shell", action="store_true",
                        help="Drop into interactive query shell")
    parser.add_argument("--persist", action="store_true",
                        help="Save DataFrames to Parquet snapshots")
    parser.add_argument("--health", action="store_true",
                        help="Print health status and exit")
    args = parser.parse_args()

    layer = DataLayer()

    if args.health:
        import pprint
        pprint.pprint(layer.health())
        return

    if args.persist:
        layer.pandas.persist_parquet()

    if args.validate:
        passed = run_validation(layer)
        raise SystemExit(0 if passed else 1)

    if args.shell:
        interactive_shell(layer)
        return

    # Default: boot, print health, run quick smoke test
    log.info("\n── Health ────────────────────────────────────────────")
    h = layer.health()
    for k, v in h.items():
        log.info("  %-20s %s", k, v)

    log.info("\n── Quick smoke test ─────────────────────────────────")
    exp = layer.get_role_expectations("EXP-SOFT-SEN")
    log.info("  role_expectations (EXP-SOFT-SEN): %d rows", len(exp))

    qs = layer.filter_questions("Software Engineer", "Technical", domain="Technical")
    log.info("  filter_questions (Software Engineer, Technical): %d rows", len(qs))

    rubric = layer.get_rubric("Technical", "Correctness")
    log.info("  rubric (Technical, Correctness): %d levels", len(rubric))

    cov = layer.get_coverage_map("EXP-SOFT-SEN")
    log.info("  coverage_map (EXP-SOFT-SEN): %s", json.dumps(cov, indent=2))

    if layer.chroma:
        found = layer.find_questions(
            "design a distributed rate limiter", "Software Engineer", "Technical", n_results=3
        )
        log.info("  find_questions (semantic): %d results — top: %s",
                 len(found), found[0]["metadata"].get("question_id", found[0]["id"]) if found else "none")

    log.info("\nPhase 1 boot complete. Run with --validate for Gate 1 checks.")


if __name__ == "__main__":
    main()

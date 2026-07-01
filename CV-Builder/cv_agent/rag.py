"""
cv_agent.rag
============
RAG Module — FAISS embedding retrieval + TF-IDF fallback + regex keyword extraction.
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from cv_agent.config import (
    PipelineConfig, logger,
    _SKLEARN_AVAILABLE, _FAISS_AVAILABLE, _SENTENCE_AVAILABLE,
)
from cv_agent.schemas import JDContext
from cv_agent.utils import jd_hash as _jd_hash


class RAGModule:
    """Retrieval-Augmented Generation module with FAISS + TF-IDF + regex fallback."""

    REQUIREMENT_PATTERNS = [
        r"(?:required|must have|mandatory)[:\s]+([^.\n]{10,80})",
        r"(\d+\+?\s+years?[^.\n]{0,50})",
        r"(?:bachelor|master|phd|degree)[^.\n]{0,60}",
    ]

    def __init__(self, ontology_path: str = "") -> None:
        self.ontology: Dict[str, List[str]] = {}
        if ontology_path and Path(ontology_path).exists():
            with open(ontology_path) as f:
                self.ontology = json.load(f)

        self._embed_model: Any = None
        self._embed_lock = threading.Lock()
        self._index_cache: Dict[str, Any] = {}
        self._chunk_cache: Dict[str, List[str]] = {}

    def _get_embed_model(self, cfg_model: str = "all-MiniLM-L6-v2") -> Optional[Any]:
        if not _SENTENCE_AVAILABLE or not _FAISS_AVAILABLE:
            return None
        if self._embed_model is None:
            with self._embed_lock:
                if self._embed_model is None:
                    try:
                        from sentence_transformers import SentenceTransformer
                        self._embed_model = SentenceTransformer(cfg_model)
                        logger.info("RAG: SentenceTransformer loaded (%s)", cfg_model)
                    except Exception as e:
                        logger.warning("RAG: embedding model load failed: %s", e)
        return self._embed_model

    def _build_faiss_index(self, chunks: List[str], model: Any, jd_h: str) -> Any:
        import faiss
        import numpy as np
        vecs = model.encode(chunks, normalize_embeddings=True, show_progress_bar=False)
        dim = vecs.shape[1]
        idx = faiss.IndexFlatIP(dim)
        idx.add(vecs.astype(np.float32))
        self._index_cache[jd_h] = idx
        self._chunk_cache[jd_h] = chunks
        return idx

    @staticmethod
    def _sentence_chunk(text: str, min_len: int = 20) -> List[str]:
        """Split text into sentence-level chunks for more granular retrieval.

        Splits on sentence boundaries (`.`, `!`, `?`, `\n`, `;`) and filters
        out fragments that are too short to be meaningful.
        """
        raw = re.split(r"(?<=[.!?;])\s+|\n+", text)
        return [s.strip() for s in raw if len(s.strip()) >= min_len]

    def extract(self, jd_text: str, target_role: str, cfg: Optional[PipelineConfig] = None) -> JDContext:
        if not jd_text.strip():
            return JDContext()
        keywords = self._extract_keywords(jd_text, cfg)
        requirements = self._extract_requirements(jd_text)
        canonical = self._ontology_align(keywords, target_role)
        return JDContext(keywords=keywords, requirements=requirements, canonical_skills=canonical)

    def semantic_search(
        self, query: str, jd_text: str, top_k: int = 5,
        cfg: Optional[PipelineConfig] = None,
        min_relevance: float = 0.3,
        scored: bool = False,
    ) -> List[Any]:
        """Search the JD for the most relevant chunks to the given query.

        Args:
            query: The search query string.
            jd_text: The full job description text to index.
            top_k: Maximum number of chunks to return.
            cfg: Pipeline config (used for embedding model name).
            min_relevance: Minimum cosine similarity score (0–1) to include a chunk.
            scored: If True, return ``List[Tuple[str, float]]`` (chunk, score).
                    If False (default), return ``List[str]`` for backward compatibility.

        Returns:
            List of matching chunks (optionally with relevance scores).
        """
        model = self._get_embed_model(cfg.embedding_model if cfg else "all-MiniLM-L6-v2")
        if model is None:
            return []
        import numpy as np
        h = _jd_hash(jd_text)
        if h not in self._index_cache:
            # Use sentence-based chunking for more granular retrieval
            sentences = self._sentence_chunk(jd_text)
            if not sentences:
                return []
            try:
                self._build_faiss_index(sentences, model, h)
            except Exception as e:
                logger.warning("FAISS index build failed: %s", e)
                return []
        try:
            q_vec = model.encode([query], normalize_embeddings=True, show_progress_bar=False)
            idx = self._index_cache[h]
            chunks = self._chunk_cache[h]
            scores_mat, ids_mat = idx.search(q_vec.astype(np.float32), min(top_k, len(chunks)))
            raw_scores: List[float] = scores_mat[0].tolist()
            raw_ids: List[int] = ids_mat[0].tolist()

            # Filter by min_relevance and collect valid (chunk, score) pairs
            results: List[Tuple[str, float]] = [
                (chunks[i], float(s))
                for i, s in zip(raw_ids, raw_scores)
                if 0 <= i < len(chunks) and float(s) >= min_relevance
            ]
            if scored:
                return results  # type: ignore[return-value]
            return [chunk for chunk, _ in results]
        except Exception as e:
            logger.warning("FAISS search failed: %s", e)
            return []

    @staticmethod
    def _tfidf_keywords(text: str, max_features: int = 60, top_n: int = 30) -> Optional[List[str]]:
        """Run TF-IDF on a list of sentence chunks and return the top-n keywords.

        Splits the text into sentences first so IDF can be computed across
        multiple documents, giving more meaningful term weights than a single
        document. Returns None if sklearn is unavailable or extraction fails.
        """
        if not _SKLEARN_AVAILABLE:
            return None
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            # Use sentence chunks as the corpus for better IDF computation
            sentences = [s.strip() for s in re.split(r"[.\n;]+", text) if len(s.strip()) > 10]
            corpus = sentences if len(sentences) >= 2 else [text]
            vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=max_features)
            mat = vec.fit_transform(corpus)
            # Sum TF-IDF scores across all sentences for each term
            scores = sorted(
                zip(vec.get_feature_names_out(), mat.toarray().sum(axis=0)),
                key=lambda x: -x[1],
            )
            return [w for w, _ in scores[:top_n]]
        except Exception:
            return None

    def _extract_keywords(self, text: str, cfg: Optional[PipelineConfig] = None) -> List[str]:
        """Extract the most relevant keywords from a job description.

        Strategy (in order of preference):
          1. TF-IDF on sentence chunks (max_features=60, top 30 terms)
          2. Regex fallback (capitalised tokens + common tech terms)
        """
        # Single call — removes the two nearly-identical duplicated TF-IDF blocks
        keywords = self._tfidf_keywords(text, max_features=60, top_n=30)
        if keywords is not None:
            return keywords

        logger.info("RAG: falling back to regex keyword extraction")
        tokens = re.findall(r'\b[A-Z][a-zA-Z+#]{2,}\b|\b(?:python|sql|aws|ml|ai|api)\b', text)
        seen: Dict[str, int] = {}
        for t in tokens:
            seen[t.lower()] = seen.get(t.lower(), 0) + 1
        return [k for k, _ in sorted(seen.items(), key=lambda x: -x[1])[:30]]


    def _extract_requirements(self, text: str) -> List[str]:
        reqs: List[str] = []
        for pat in self.REQUIREMENT_PATTERNS:
            reqs.extend(re.findall(pat, text, re.IGNORECASE))
        return [r.strip() for r in reqs[:10]]

    def _ontology_align(self, keywords: List[str], role: str) -> List[str]:
        canonical: List[str] = []
        role_key = role.lower()
        for k, skills in self.ontology.items():
            if k.lower() in role_key or role_key in k.lower():
                canonical.extend(skills)
        kw_lower = {k.lower() for k in keywords}
        return [s for s in canonical if any(s.lower() in kw for kw in kw_lower)][:15]

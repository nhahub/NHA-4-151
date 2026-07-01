"""
cv_agent.memory
===============
SQLite-backed session store with pattern injection for cross-session learning.
"""

from __future__ import annotations

import sqlite3
import threading
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from cv_agent.config import logger
from cv_agent.schemas import JudgeOutput


class MemoryModule:
    """SQLite-backed session store with pattern learning."""

    def __init__(self, db_path: str, session_id: Optional[str] = None) -> None:
        self._db_path = db_path
        self._write_lock = threading.Lock()
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.session_id = session_id or str(uuid.uuid4())
        self._init_schema()

    def _init_schema(self) -> None:
        with self._write_lock:
            self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS weakness_log (
                    session_id TEXT NOT NULL, iteration INTEGER,
                    category TEXT, weakness TEXT, ts TEXT
                );
                CREATE TABLE IF NOT EXISTS success_patterns (
                    role TEXT, score REAL, pattern_text TEXT, ts TEXT
                );
                CREATE TABLE IF NOT EXISTS session_meta (
                    session_id TEXT PRIMARY KEY, role TEXT,
                    created_at TEXT, last_updated TEXT
                );
                CREATE TABLE IF NOT EXISTS common_improvements (
                    role TEXT, category TEXT, suggestion TEXT,
                    freq INTEGER DEFAULT 1, ts TEXT
                );
                CREATE TABLE IF NOT EXISTS score_log (
                    session_id TEXT, iteration INTEGER,
                    score REAL, strategy TEXT, ts TEXT
                );
            """)
            self.conn.commit()

    def record_iteration(
        self, session_id: str, iteration: int,
        weighted: JudgeOutput, strategy: str = "",
    ) -> None:
        sid = session_id or self.session_id
        ts = datetime.now().isoformat()
        with self._write_lock:
            for w in weighted.weaknesses:
                self.conn.execute(
                    "INSERT INTO weakness_log VALUES (?,?,?,?,?)",
                    (sid, iteration, self._categorize(w), w, ts),
                )
            for sug in weighted.improvement_suggestions:
                cat = self._categorize(sug)
                existing = self.conn.execute(
                    "SELECT rowid, freq FROM common_improvements WHERE role=? AND suggestion=?",
                    ("", sug),
                ).fetchone()
                if existing:
                    self.conn.execute(
                        "UPDATE common_improvements SET freq=?, ts=? WHERE rowid=?",
                        (existing[1] + 1, ts, existing[0]),
                    )
                else:
                    self.conn.execute(
                        "INSERT INTO common_improvements VALUES (?,?,?,?,?)",
                        ("", cat, sug, 1, ts),
                    )
            self.conn.execute(
                "INSERT INTO score_log VALUES (?,?,?,?,?)",
                (sid, iteration, float(weighted.overall_score), strategy, ts),
            )
            self.conn.execute(
                "INSERT OR REPLACE INTO session_meta VALUES (?,?,?,?)",
                (sid, "", ts, ts),
            )
            self.conn.commit()

    def record_success(self, role: str, score: float, cv_excerpt: str) -> None:
        ts = datetime.now().isoformat()
        with self._write_lock:
            self.conn.execute(
                "INSERT INTO success_patterns VALUES (?,?,?,?)",
                (role, score, cv_excerpt[:1000], ts),
            )
            self.conn.commit()

    def get_recurring_issues(self, session_id: str, min_count: int = 2) -> List[str]:
        rows = self.conn.execute(
            "SELECT category, COUNT(*) as cnt FROM weakness_log "
            "WHERE session_id = ? GROUP BY category HAVING cnt >= ? ORDER BY cnt DESC",
            (session_id, min_count),
        ).fetchall()
        return [r[0] for r in rows]

    def get_successful_patterns(self, role: str, limit: int = 3) -> List[str]:
        rows = self.conn.execute(
            "SELECT pattern_text FROM success_patterns WHERE role LIKE ? ORDER BY score DESC LIMIT ?",
            (f"%{role.split()[0]}%", limit),
        ).fetchall()
        return [r[0] for r in rows]

    def get_common_improvements(self, limit: int = 5) -> List[str]:
        rows = self.conn.execute(
            "SELECT suggestion FROM common_improvements ORDER BY freq DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [r[0] for r in rows]

    def get_anti_patterns(self, session_id: str) -> List[str]:
        rows = self.conn.execute(
            "SELECT category, COUNT(*) as cnt FROM weakness_log "
            "WHERE session_id = ? GROUP BY category HAVING cnt >= 3 ORDER BY cnt DESC",
            (session_id,),
        ).fetchall()
        return [r[0] for r in rows]

    def get_session_score_history(self, session_id: str) -> List[Tuple[int, float, str]]:
        rows = self.conn.execute(
            "SELECT iteration, score, strategy FROM score_log WHERE session_id=? ORDER BY iteration",
            (session_id,),
        ).fetchall()
        return [(r[0], r[1], r[2]) for r in rows]

    def build_memory_block(self, session_id: str, role: str) -> str:
        recurring = self.get_recurring_issues(session_id)
        patterns = self.get_successful_patterns(role)
        improvements = self.get_common_improvements()
        anti = self.get_anti_patterns(session_id)

        block = ""
        if recurring:
            block += f"\n\nRECURRING ISSUES TO FIX: {', '.join(recurring)}"
        if anti:
            block += f"\n\nPERSISTENT WEAKNESSES (3+ iterations): {', '.join(anti)} — prioritize these"
        if improvements:
            block += f"\n\nCOMMON IMPROVEMENTS THAT WORK: {'; '.join(improvements[:3])}"
        if patterns:
            block += f"\n\nSUCCESSFUL PATTERN EXCERPT:\n{patterns[0][:600]}"
        return block

    @staticmethod
    def _categorize(text: str) -> str:
        w = text.lower()
        if any(k in w for k in ["metric", "quantif", "number", "%", "$"]):
            return "missing_metrics"
        if any(k in w for k in ["verb", "passive", "worked on", "helped"]):
            return "weak_verbs"
        if any(k in w for k in ["section", "missing", "header"]):
            return "missing_sections"
        if any(k in w for k in ["keyword", "ats", "alignment"]):
            return "ats_keywords"
        if any(k in w for k in ["clarity", "unclear", "vague"]):
            return "clarity"
        return "other"

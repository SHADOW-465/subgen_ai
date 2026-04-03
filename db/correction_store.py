"""
SubGEN AI — SQLite Correction Store.

Persists user-validated subtitle corrections with MFCC fingerprints.
Used by the transcriber for automatic correction lookup at inference time.

DB location: ~/.subgen_ai/corrections.db
Each connection is opened and closed per operation (thread-safe, Streamlit-compatible).
"""
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from subgen_ai.core.models import CorrectionRecord

# Default DB path (can be monkeypatched in tests)
DB_PATH: Path = Path.home() / ".subgen_ai" / "corrections.db"

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS corrections (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    segment_start   REAL NOT NULL,
    segment_end     REAL NOT NULL,
    original_text   TEXT NOT NULL,
    corrected_text  TEXT NOT NULL,
    language        TEXT NOT NULL DEFAULT 'en',
    mfcc_mean       TEXT NOT NULL,
    mfcc_var        TEXT NOT NULL,
    match_score     REAL NOT NULL DEFAULT 0.0,
    hw_used         INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_lang ON corrections(language);
"""


def init_db() -> sqlite3.Connection:
    """
    Initialise the database, creating the directory and table if needed.
    Returns an open connection — caller is responsible for closing it.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.executescript(CREATE_TABLE_SQL)
    conn.commit()
    return conn


def save_correction(record: CorrectionRecord) -> int:
    """
    Insert a new correction record into the database.
    Returns the auto-assigned row id.
    """
    conn = init_db()
    cur = conn.execute(
        """
        INSERT INTO corrections
            (segment_start, segment_end, original_text, corrected_text, language,
             mfcc_mean, mfcc_var, match_score, hw_used, created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?)
        """,
        (
            record.segment_start, record.segment_end,
            record.original_text, record.corrected_text, record.language,
            json.dumps(record.mfcc_mean), json.dumps(record.mfcc_var),
            record.match_score, int(record.hw_used),
            record.created_at or datetime.now().isoformat()
        )
    )
    conn.commit()
    rowid = cur.lastrowid
    conn.close()
    return rowid


def find_nearest_correction(query_mfcc_mean: list, language: str,
                             threshold: float = 0.80) -> Optional[CorrectionRecord]:
    """
    Find the best matching stored correction for a given audio fingerprint.

    Scans all corrections for the given language and returns the one with
    the highest cosine similarity score, provided it exceeds the threshold.

    Args:
        query_mfcc_mean: 12-float MFCC mean vector for the current segment.
        language:        ISO 639-1 language code to restrict the search.
        threshold:       Minimum cosine similarity to return a match (default 0.80).

    Returns:
        The best matching CorrectionRecord or None if no match found.
    """
    conn = init_db()
    rows = conn.execute(
        "SELECT * FROM corrections WHERE language = ?", (language,)
    ).fetchall()
    conn.close()

    best_score, best_row = 0.0, None
    for row in rows:
        stored_mean = json.loads(row[6])   # mfcc_mean column index
        score = _cosine_sim_arrays(query_mfcc_mean, stored_mean)
        if score > best_score:
            best_score, best_row = score, row

    if best_score >= threshold and best_row is not None:
        return CorrectionRecord(
            id=best_row[0],
            segment_start=best_row[1],
            segment_end=best_row[2],
            original_text=best_row[3],
            corrected_text=best_row[4],
            language=best_row[5],
            mfcc_mean=json.loads(best_row[6]),
            mfcc_var=json.loads(best_row[7]),
            match_score=best_score,
            hw_used=bool(best_row[9]),
            created_at=best_row[10]
        )
    return None


def get_db_stats() -> dict:
    """Return total correction count and per-language breakdown."""
    conn = init_db()
    total = conn.execute("SELECT COUNT(*) FROM corrections").fetchone()[0]
    by_lang = conn.execute(
        "SELECT language, COUNT(*) FROM corrections GROUP BY language"
    ).fetchall()
    conn.close()
    return {"total": total, "by_language": dict(by_lang)}


def delete_correction(correction_id: int) -> None:
    """Delete a single correction by its database id."""
    conn = init_db()
    conn.execute("DELETE FROM corrections WHERE id = ?", (correction_id,))
    conn.commit()
    conn.close()


def _cosine_sim_arrays(v1: list, v2: list) -> float:
    """
    Cosine similarity mapped from [-1, 1] to [0, 1].
    Returns 0.0 if either vector is near-zero.
    """
    a  = np.array(v1, dtype=np.float64)
    b  = np.array(v2, dtype=np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float((np.dot(a, b) / (na * nb) + 1.0) / 2.0)

"""Persistenza su SQLite: libreria, posizione di lettura, note, impostazioni."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from . import config

_local = threading.local()

SCHEMA = """
CREATE TABLE IF NOT EXISTS books (
    id          TEXT PRIMARY KEY,
    filename    TEXT NOT NULL,
    title       TEXT NOT NULL,
    author      TEXT,
    language    TEXT,
    n_chapters  INTEGER NOT NULL DEFAULT 0,
    added_at    REAL NOT NULL,
    opened_at   REAL
);

CREATE TABLE IF NOT EXISTS progress (
    book_id      TEXT PRIMARY KEY REFERENCES books(id) ON DELETE CASCADE,
    chapter_idx  INTEGER NOT NULL DEFAULT 0,
    sentence_idx INTEGER NOT NULL DEFAULT 0,
    word_idx     INTEGER NOT NULL DEFAULT 0,
    updated_at   REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS notes (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    book_id      TEXT NOT NULL REFERENCES books(id) ON DELETE CASCADE,
    chapter_idx  INTEGER NOT NULL,
    sentence_idx INTEGER,
    anchor       TEXT,
    body         TEXT NOT NULL,
    created_at   REAL NOT NULL,
    updated_at   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS notes_book ON notes(book_id, chapter_idx);

CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def conn() -> sqlite3.Connection:
    """Connessione per-thread: FastAPI serve le richieste da un pool di thread."""
    c = getattr(_local, "conn", None)
    if c is None:
        config.ensure_dirs()
        c = sqlite3.connect(config.DB_PATH, check_same_thread=False)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA foreign_keys=ON")
        c.executescript(SCHEMA)
        _local.conn = c
    return c


# --- libreria ---------------------------------------------------------------

def upsert_book(book_id: str, filename: str, title: str, author: str,
                language: str, n_chapters: int) -> None:
    c = conn()
    now = time.time()
    c.execute(
        """INSERT INTO books (id, filename, title, author, language, n_chapters, added_at)
           VALUES (?,?,?,?,?,?,?)
           ON CONFLICT(id) DO UPDATE SET
             filename=excluded.filename, title=excluded.title, author=excluded.author,
             language=excluded.language, n_chapters=excluded.n_chapters""",
        (book_id, filename, title, author, language, n_chapters, now),
    )
    c.commit()


def list_books() -> list[dict[str, Any]]:
    rows = conn().execute(
        """SELECT b.*, p.chapter_idx, p.sentence_idx
           FROM books b LEFT JOIN progress p ON p.book_id = b.id
           ORDER BY COALESCE(b.opened_at, b.added_at) DESC"""
    ).fetchall()
    return [dict(r) for r in rows]


def get_book(book_id: str) -> dict[str, Any] | None:
    row = conn().execute("SELECT * FROM books WHERE id=?", (book_id,)).fetchone()
    return dict(row) if row else None


def touch_book(book_id: str) -> None:
    c = conn()
    c.execute("UPDATE books SET opened_at=? WHERE id=?", (time.time(), book_id))
    c.commit()


def delete_book(book_id: str) -> None:
    c = conn()
    c.execute("DELETE FROM books WHERE id=?", (book_id,))
    c.commit()


# --- posizione di lettura ---------------------------------------------------

def get_progress(book_id: str) -> dict[str, int]:
    row = conn().execute("SELECT * FROM progress WHERE book_id=?", (book_id,)).fetchone()
    if not row:
        return {"chapter_idx": 0, "sentence_idx": 0, "word_idx": 0}
    return {"chapter_idx": row["chapter_idx"], "sentence_idx": row["sentence_idx"],
            "word_idx": row["word_idx"]}


def set_progress(book_id: str, chapter_idx: int, sentence_idx: int, word_idx: int = 0) -> None:
    c = conn()
    c.execute(
        """INSERT INTO progress (book_id, chapter_idx, sentence_idx, word_idx, updated_at)
           VALUES (?,?,?,?,?)
           ON CONFLICT(book_id) DO UPDATE SET
             chapter_idx=excluded.chapter_idx, sentence_idx=excluded.sentence_idx,
             word_idx=excluded.word_idx, updated_at=excluded.updated_at""",
        (book_id, chapter_idx, sentence_idx, word_idx, time.time()),
    )
    c.commit()


# --- note -------------------------------------------------------------------

def list_notes(book_id: str, chapter_idx: int | None = None) -> list[dict[str, Any]]:
    if chapter_idx is None:
        rows = conn().execute(
            "SELECT * FROM notes WHERE book_id=? ORDER BY chapter_idx, sentence_idx, id",
            (book_id,)).fetchall()
    else:
        rows = conn().execute(
            "SELECT * FROM notes WHERE book_id=? AND chapter_idx=? ORDER BY sentence_idx, id",
            (book_id, chapter_idx)).fetchall()
    return [dict(r) for r in rows]


def add_note(book_id: str, chapter_idx: int, sentence_idx: int | None,
             anchor: str, body: str) -> dict[str, Any]:
    c = conn()
    now = time.time()
    cur = c.execute(
        """INSERT INTO notes (book_id, chapter_idx, sentence_idx, anchor, body, created_at, updated_at)
           VALUES (?,?,?,?,?,?,?)""",
        (book_id, chapter_idx, sentence_idx, anchor, body, now, now))
    c.commit()
    row = c.execute("SELECT * FROM notes WHERE id=?", (cur.lastrowid,)).fetchone()
    return dict(row)


def update_note(note_id: int, body: str) -> None:
    c = conn()
    c.execute("UPDATE notes SET body=?, updated_at=? WHERE id=?", (body, time.time(), note_id))
    c.commit()


def delete_note(note_id: int) -> None:
    c = conn()
    c.execute("DELETE FROM notes WHERE id=?", (note_id,))
    c.commit()


# --- impostazioni -----------------------------------------------------------

def get_settings() -> dict[str, Any]:
    rows = conn().execute("SELECT key, value FROM settings").fetchall()
    return {r["key"]: json.loads(r["value"]) for r in rows}


def set_settings(values: dict[str, Any]) -> None:
    c = conn()
    c.executemany(
        """INSERT INTO settings (key, value) VALUES (?,?)
           ON CONFLICT(key) DO UPDATE SET value=excluded.value""",
        [(k, json.dumps(v)) for k, v in values.items()])
    c.commit()

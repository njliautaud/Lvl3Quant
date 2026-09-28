#!/usr/bin/env python3
"""
fts5_memory_index.py — Fast full-text-search index over the MCP memory store.

WHY THIS EXISTS:
- The MCP `memories` table grows monotonically and is queried via `recall` /
  `find_similar` / `list_memories`. Without an FTS index, every query is a
  LIKE-scan over the entire `content` column. As the store grows past a few
  thousand rows that becomes slow + token-expensive (full content gets pulled
  to compare).
- HERMES (Nous Research, Feb 2026) uses SQLite FTS5 for ~10ms recall over 10K+
  entries. This script ports that pattern onto our existing memory.db without
  touching the MCP server code.

DESIGN:
- ADDITIVE. We create a parallel `memories_fts` virtual table and three
  triggers on `memories`. The MCP server keeps reading/writing `memories` as
  before; FTS5 is kept in sync automatically by SQLite.
- IDEMPOTENT. `init` is safe to re-run. `rebuild` drops + repopulates the
  FTS table from scratch.
- READ-ONLY EXCEPT IN init/rebuild. The `search` and `status` commands open
  the DB read-only.

USAGE:
  python3 fts5_memory_index.py init             # one-time: create FTS table + triggers
  python3 fts5_memory_index.py rebuild          # nuke + repopulate FTS table
  python3 fts5_memory_index.py search "query"   # FTS5 query, ranked by bm25
  python3 fts5_memory_index.py status           # row counts, last-sync sanity

SAFETY:
- Uses BEGIN IMMEDIATE for writes so concurrent MCP writes are serialized.
- Triggers are AFTER INSERT/UPDATE/DELETE — original write always commits
  first; FTS sync is a follow-on.
- No DDL on the existing `memories` table. Only adds a virtual table + triggers.

EXIT CODES: 0 success, 1 user error, 2 DB error.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from pathlib import Path

DEFAULT_DB = Path("/home/jupiter/teleclaude-main/memory/memory.db")
FTS_TABLE = "memories_fts"
SOURCE_TABLE = "memories"
TRIGGER_PREFIX = "memories_fts_sync"


def _connect(db_path: Path, *, read_only: bool = False) -> sqlite3.Connection:
    if not db_path.exists():
        print(f"ERROR: DB not found: {db_path}", file=sys.stderr)
        sys.exit(2)
    if read_only:
        uri = f"file:{db_path}?mode=ro"
        return sqlite3.connect(uri, uri=True)
    return sqlite3.connect(str(db_path))


def _fts_exists(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (FTS_TABLE,),
    ).fetchone()
    return row is not None


def cmd_init(db_path: Path) -> int:
    """Create FTS table + triggers if missing. Safe to re-run."""
    conn = _connect(db_path)
    try:
        # Verify source table exists with expected columns
        cols = [r[1] for r in conn.execute(f"PRAGMA table_info({SOURCE_TABLE})")]
        if "id" not in cols or "content" not in cols:
            print(
                f"ERROR: source table `{SOURCE_TABLE}` lacks id/content cols. Got: {cols}",
                file=sys.stderr,
            )
            return 2

        conn.execute("BEGIN IMMEDIATE")

        if not _fts_exists(conn):
            # contentless FTS5: stores its own copy of content, indexed.
            # content='memories', content_rowid='id' would require id INTEGER;
            # our id is TEXT, so we mirror content explicitly.
            conn.execute(
                f"""
                CREATE VIRTUAL TABLE {FTS_TABLE} USING fts5(
                    mem_id UNINDEXED,
                    content,
                    priority UNINDEXED,
                    status UNINDEXED,
                    tokenize='porter unicode61'
                )
                """
            )
            print(f"  created virtual table {FTS_TABLE}")
        else:
            print(f"  {FTS_TABLE} already exists; leaving as-is")

        # Triggers — drop + recreate so they always match current schema.
        for action in ("ai", "au_old", "au_new", "ad"):
            conn.execute(f"DROP TRIGGER IF EXISTS {TRIGGER_PREFIX}_{action}")

        conn.execute(
            f"""
            CREATE TRIGGER {TRIGGER_PREFIX}_ai
            AFTER INSERT ON {SOURCE_TABLE}
            BEGIN
                INSERT INTO {FTS_TABLE}(mem_id, content, priority, status)
                VALUES (NEW.id, NEW.content, NEW.priority, NEW.status);
            END
            """
        )
        conn.execute(
            f"""
            CREATE TRIGGER {TRIGGER_PREFIX}_ad
            AFTER DELETE ON {SOURCE_TABLE}
            BEGIN
                DELETE FROM {FTS_TABLE} WHERE mem_id = OLD.id;
            END
            """
        )
        # FTS5 has no native row UPDATE — implemented as DELETE + INSERT.
        conn.execute(
            f"""
            CREATE TRIGGER {TRIGGER_PREFIX}_au_old
            AFTER UPDATE ON {SOURCE_TABLE}
            BEGIN
                DELETE FROM {FTS_TABLE} WHERE mem_id = OLD.id;
                INSERT INTO {FTS_TABLE}(mem_id, content, priority, status)
                VALUES (NEW.id, NEW.content, NEW.priority, NEW.status);
            END
            """
        )
        print(f"  installed 3 sync triggers on {SOURCE_TABLE}")

        conn.commit()
    except sqlite3.Error as e:
        conn.rollback()
        print(f"DB ERROR: {e}", file=sys.stderr)
        return 2
    finally:
        conn.close()
    print("init OK")
    return 0


def cmd_rebuild(db_path: Path) -> int:
    """Drop + repopulate FTS table. Triggers untouched (assumed in place)."""
    conn = _connect(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        if _fts_exists(conn):
            conn.execute(f"DELETE FROM {FTS_TABLE}")
            print(f"  cleared {FTS_TABLE}")
        else:
            print(f"  {FTS_TABLE} missing — run `init` first", file=sys.stderr)
            conn.rollback()
            return 1

        t0 = time.perf_counter()
        rows = conn.execute(
            f"SELECT id, content, priority, status FROM {SOURCE_TABLE}"
        ).fetchall()
        conn.executemany(
            f"INSERT INTO {FTS_TABLE}(mem_id, content, priority, status) "
            f"VALUES (?, ?, ?, ?)",
            rows,
        )
        conn.commit()
        dt = time.perf_counter() - t0
        print(f"  populated {len(rows)} rows into {FTS_TABLE} in {dt*1000:.1f} ms")
    except sqlite3.Error as e:
        conn.rollback()
        print(f"DB ERROR: {e}", file=sys.stderr)
        return 2
    finally:
        conn.close()
    print("rebuild OK")
    return 0


def cmd_search(db_path: Path, query: str, limit: int) -> int:
    """Run FTS5 query, return top N rows ranked by bm25 score."""
    if not query.strip():
        print("ERROR: empty query", file=sys.stderr)
        return 1
    conn = _connect(db_path, read_only=True)
    try:
        if not _fts_exists(conn):
            print(f"ERROR: {FTS_TABLE} missing — run `init` first", file=sys.stderr)
            return 1
        t0 = time.perf_counter()
        # bm25 lower = better; sort ascending. Snippet for context.
        sql = f"""
            SELECT mem_id, priority, status,
                   snippet({FTS_TABLE}, 1, '«', '»', '…', 12) AS snippet,
                   bm25({FTS_TABLE}) AS score
            FROM {FTS_TABLE}
            WHERE {FTS_TABLE} MATCH ?
            ORDER BY score
            LIMIT ?
        """
        rows = conn.execute(sql, (query, limit)).fetchall()
        dt = time.perf_counter() - t0
        print(f"  {len(rows)} hits in {dt*1000:.2f} ms for query: {query!r}\n")
        for i, (mid, prio, status, snip, score) in enumerate(rows, 1):
            print(f"  [{i}] score={score:.3f} prio={prio} status={status} id={mid}")
            print(f"       {snip}\n")
    except sqlite3.Error as e:
        print(f"DB ERROR: {e}", file=sys.stderr)
        return 2
    finally:
        conn.close()
    return 0


def cmd_status(db_path: Path) -> int:
    """Report row counts + sync sanity."""
    conn = _connect(db_path, read_only=True)
    try:
        src_n = conn.execute(f"SELECT COUNT(*) FROM {SOURCE_TABLE}").fetchone()[0]
        if not _fts_exists(conn):
            print(f"  {SOURCE_TABLE}: {src_n} rows")
            print(f"  {FTS_TABLE}: ABSENT — run `init` first")
            return 0
        fts_n = conn.execute(f"SELECT COUNT(*) FROM {FTS_TABLE}").fetchone()[0]
        print(f"  {SOURCE_TABLE}: {src_n} rows")
        print(f"  {FTS_TABLE}:  {fts_n} rows")
        if src_n != fts_n:
            print(
                f"  WARN: drift {src_n - fts_n} rows. Run `rebuild` to resync.",
                file=sys.stderr,
            )
        else:
            print("  in sync ✓")
        triggers = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger' AND name LIKE ?",
                (f"{TRIGGER_PREFIX}%",),
            )
        ]
        print(f"  triggers: {len(triggers)} installed: {triggers}")
    except sqlite3.Error as e:
        print(f"DB ERROR: {e}", file=sys.stderr)
        return 2
    finally:
        conn.close()
    return 0


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB,
        help=f"path to memory.db (default: {DEFAULT_DB})",
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init", help="create FTS5 table + sync triggers (idempotent)")
    sub.add_parser("rebuild", help="drop + repopulate FTS table from memories")
    sp = sub.add_parser("search", help="FTS5 query, ranked by bm25")
    sp.add_argument("query", help="FTS5 query string (e.g. 'fold AND crash')")
    sp.add_argument("-n", "--limit", type=int, default=10, help="max results (default 10)")
    sub.add_parser("status", help="row counts + drift check")

    args = p.parse_args(argv)
    if args.cmd == "init":
        return cmd_init(args.db)
    if args.cmd == "rebuild":
        return cmd_rebuild(args.db)
    if args.cmd == "search":
        return cmd_search(args.db, args.query, args.limit)
    if args.cmd == "status":
        return cmd_status(args.db)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

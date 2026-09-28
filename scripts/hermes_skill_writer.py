#!/usr/bin/env python3
"""
hermes_skill_writer.py — Create / update / search procedural skills (HERMES Win #1).

WHY THIS EXISTS:
- HERMES (Nous Research) "auto-skill generation" pattern: every non-trivial
  workflow we solve should crystallize into a reusable SKILL.md the next session
  can retrieve before re-deriving the answer from scratch.
- Skills live at ~/.claude/skills/<category>/<slug>/SKILL.md (Claude Code harness
  convention). YAML frontmatter + markdown body.
- We also mirror each skill into an FTS5-indexed SQLite at
  /home/jupiter/Lvl3Quant/data/hermes_skills.sqlite so retrieval is sub-ms.

DESIGN PRINCIPLES (same as scripts/fts5_memory_index.py):
- ADDITIVE: writing a skill file is the source of truth; the SQLite is a derived
  index. `--rebuild-index` walks ~/.claude/skills/ and repopulates.
- IDEMPOTENT: re-running create on an existing slug requires --update; re-running
  rebuild-index never corrupts state.
- STDLIB ONLY: sqlite3 + argparse + pathlib. No new pip deps.

USAGE:
  Create:
    python3 scripts/hermes_skill_writer.py \\
      --name fifo-canonical-replay \\
      --category trading \\
      --tags fifo,backtest,hc74 \\
      --problem "Midpoint backtest looks profitable but FIFO replay is negative" \\
      --solution-file /tmp/draft.md \\
      [--description "one-line summary"] \\
      [--when-to-use "trigger conditions text"] \\
      [--update]

  List:
    python3 scripts/hermes_skill_writer.py --list
    python3 scripts/hermes_skill_writer.py --list --category trading

  Search:
    python3 scripts/hermes_skill_writer.py --search "fifo cost model"

  Rebuild FTS index from on-disk SKILL.md files:
    python3 scripts/hermes_skill_writer.py --rebuild-index

EXIT CODES: 0 success, 1 user error, 2 IO/DB error.
"""

from __future__ import annotations

import argparse
import os
import re
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SKILLS_ROOT = Path(os.path.expanduser("~/.claude/skills"))
DB_PATH = Path("/home/jupiter/Lvl3Quant/data/hermes_skills.sqlite")
VALID_CATEGORIES = {"trading", "infra", "debug", "analysis", "general"}
SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,63}$")
DEFAULT_VERSION = "1.0.0"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _slug_ok(slug: str) -> bool:
    return bool(SLUG_RE.match(slug))


def _skill_path(category: str, slug: str) -> Path:
    return SKILLS_ROOT / category / slug / "SKILL.md"


def _connect(*, read_only: bool = False) -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if read_only and DB_PATH.exists():
        return sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    return sqlite3.connect(str(DB_PATH))


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Create skills table + FTS5 mirror + triggers. Idempotent."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS skills (
            slug      TEXT PRIMARY KEY,
            category  TEXT NOT NULL,
            name      TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            tags      TEXT NOT NULL DEFAULT '',
            version   TEXT NOT NULL DEFAULT '1.0.0',
            body      TEXT NOT NULL DEFAULT '',
            path      TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS skills_fts USING fts5(
            slug UNINDEXED,
            name,
            description,
            tags,
            body,
            category UNINDEXED,
            tokenize='porter unicode61'
        );

        CREATE TRIGGER IF NOT EXISTS skills_ai
        AFTER INSERT ON skills BEGIN
            INSERT INTO skills_fts(slug, name, description, tags, body, category)
            VALUES (NEW.slug, NEW.name, NEW.description, NEW.tags, NEW.body, NEW.category);
        END;

        CREATE TRIGGER IF NOT EXISTS skills_ad
        AFTER DELETE ON skills BEGIN
            DELETE FROM skills_fts WHERE slug = OLD.slug;
        END;

        CREATE TRIGGER IF NOT EXISTS skills_au
        AFTER UPDATE ON skills BEGIN
            DELETE FROM skills_fts WHERE slug = OLD.slug;
            INSERT INTO skills_fts(slug, name, description, tags, body, category)
            VALUES (NEW.slug, NEW.name, NEW.description, NEW.tags, NEW.body, NEW.category);
        END;
        """
    )
    conn.commit()


def _render_skill_md(
    *,
    slug: str,
    title: str,
    description: str,
    version: str,
    tags: list[str],
    category: str,
    when_to_use: str,
    procedure: str,
    pitfalls: str,
    verification: str,
) -> str:
    """Render YAML-frontmatter + markdown body matching Hermes SKILL.md format."""
    tag_yaml = ", ".join(tags) if tags else ""
    frontmatter = (
        "---\n"
        f"name: {slug}\n"
        f"description: {description}\n"
        f"version: {version}\n"
        "metadata:\n"
        "  hermes:\n"
        f"    tags: [{tag_yaml}]\n"
        f"    category: {category}\n"
        f"    created: {_now_iso()}\n"
        "---\n"
    )
    body = (
        f"# {title}\n\n"
        "## When to Use\n"
        f"{when_to_use.strip()}\n\n"
        "## Procedure\n"
        f"{procedure.strip()}\n\n"
        "## Pitfalls\n"
        f"{pitfalls.strip()}\n\n"
        "## Verification\n"
        f"{verification.strip()}\n"
    )
    return frontmatter + "\n" + body


def _parse_solution_file(text: str) -> dict:
    """
    Parse a draft solution markdown file. Recognized section headers
    (case-insensitive, ## prefix optional):
      title          -> H1 of body, OR first non-empty line
      description    -> one-line summary (## Description)
      when-to-use    -> ## When to Use
      procedure      -> ## Procedure
      pitfalls       -> ## Pitfalls
      verification   -> ## Verification
    Returns dict with those keys; missing keys default to "".
    """
    sections = {
        "title": "",
        "description": "",
        "when_to_use": "",
        "procedure": "",
        "pitfalls": "",
        "verification": "",
    }
    header_map = {
        "description": "description",
        "when to use": "when_to_use",
        "when-to-use": "when_to_use",
        "procedure": "procedure",
        "pitfalls": "pitfalls",
        "verification": "verification",
    }
    current = None
    buf: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if line.startswith("# ") and not sections["title"]:
            sections["title"] = line[2:].strip()
            continue
        m = re.match(r"^#{2,3}\s+(.*?)\s*$", line)
        if m:
            if current and buf:
                sections[current] = "\n".join(buf).strip()
            header = m.group(1).strip().lower()
            current = header_map.get(header)
            buf = []
            continue
        if current is not None:
            buf.append(line)
    if current and buf:
        sections[current] = "\n".join(buf).strip()
    return sections


def cmd_create(args: argparse.Namespace) -> int:
    if not _slug_ok(args.name):
        print(f"ERROR: invalid slug {args.name!r}; must match [a-z0-9-]{{2,64}}", file=sys.stderr)
        return 1
    if args.category not in VALID_CATEGORIES:
        print(
            f"ERROR: invalid category {args.category!r}; pick from {sorted(VALID_CATEGORIES)}",
            file=sys.stderr,
        )
        return 1

    sol_path = Path(args.solution_file)
    if not sol_path.exists():
        print(f"ERROR: --solution-file not found: {sol_path}", file=sys.stderr)
        return 1

    sol = _parse_solution_file(sol_path.read_text())
    title = sol["title"] or args.name.replace("-", " ").title()
    description = args.description or sol["description"] or (args.problem or "").strip()[:120]
    when_to_use = args.when_to_use or sol["when_to_use"] or (args.problem or "").strip()
    procedure = sol["procedure"] or "1. (fill in)\n"
    pitfalls = sol["pitfalls"] or "- (none documented yet)\n"
    verification = sol["verification"] or "- (fill in)\n"
    tags = [t.strip() for t in (args.tags or "").split(",") if t.strip()]

    out_path = _skill_path(args.category, args.name)
    if out_path.exists() and not args.update:
        print(f"ERROR: skill already exists at {out_path}; pass --update to overwrite", file=sys.stderr)
        return 1
    out_path.parent.mkdir(parents=True, exist_ok=True)

    md = _render_skill_md(
        slug=args.name,
        title=title,
        description=description,
        version=args.version,
        tags=tags,
        category=args.category,
        when_to_use=when_to_use,
        procedure=procedure,
        pitfalls=pitfalls,
        verification=verification,
    )
    out_path.write_text(md)
    print(f"  wrote {out_path} ({len(md)} bytes)")

    conn = _connect()
    try:
        _ensure_schema(conn)
        now = _now_iso()
        existing = conn.execute("SELECT created_at FROM skills WHERE slug=?", (args.name,)).fetchone()
        if existing:
            conn.execute(
                "UPDATE skills SET category=?, name=?, description=?, tags=?, version=?, body=?, path=?, updated_at=? WHERE slug=?",
                (args.category, title, description, ",".join(tags), args.version, md, str(out_path), now, args.name),
            )
        else:
            conn.execute(
                "INSERT INTO skills(slug, category, name, description, tags, version, body, path, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (args.name, args.category, title, description, ",".join(tags), args.version, md, str(out_path), now, now),
            )
        conn.commit()
    except sqlite3.Error as e:
        print(f"DB ERROR: {e}", file=sys.stderr)
        return 2
    finally:
        conn.close()
    print(f"  indexed slug={args.name} category={args.category} tags={tags}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    conn = _connect(read_only=DB_PATH.exists())
    try:
        _ensure_schema(conn) if not DB_PATH.exists() else None
        if args.category:
            rows = conn.execute(
                "SELECT slug, category, description, version, updated_at FROM skills WHERE category=? ORDER BY category, slug",
                (args.category,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT slug, category, description, version, updated_at FROM skills ORDER BY category, slug"
            ).fetchall()
        if not rows:
            print("  (no skills indexed; run --rebuild-index after creating files, or create one with --name ...)")
            return 0
        print(f"  {len(rows)} skill(s):\n")
        for slug, cat, desc, ver, upd in rows:
            print(f"  [{cat}/{slug}] v{ver}  (updated {upd})")
            print(f"      {desc}\n")
    finally:
        conn.close()
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    if not args.search.strip():
        print("ERROR: empty search query", file=sys.stderr)
        return 1
    if not DB_PATH.exists():
        print("ERROR: index DB missing; run --rebuild-index first", file=sys.stderr)
        return 1
    conn = _connect(read_only=True)
    try:
        t0 = time.perf_counter()
        rows = conn.execute(
            """
            SELECT slug, category, snippet(skills_fts, 4, '«', '»', '…', 12) AS snip,
                   bm25(skills_fts) AS score
            FROM skills_fts WHERE skills_fts MATCH ?
            ORDER BY score LIMIT ?
            """,
            (args.search, args.top),
        ).fetchall()
        dt = time.perf_counter() - t0
        print(f"  {len(rows)} hit(s) in {dt*1000:.2f} ms for {args.search!r}\n")
        for i, (slug, cat, snip, score) in enumerate(rows, 1):
            print(f"  [{i}] {cat}/{slug}  score={score:.3f}")
            print(f"       {snip}\n")
    except sqlite3.Error as e:
        print(f"DB ERROR: {e}", file=sys.stderr)
        return 2
    finally:
        conn.close()
    return 0


def _parse_skill_md(text: str) -> dict:
    """Best-effort parse of YAML frontmatter + body. Stdlib-only."""
    out = {"slug": "", "description": "", "version": DEFAULT_VERSION, "tags": [], "category": "general", "title": "", "body": text}
    if not text.startswith("---"):
        return out
    end = text.find("\n---", 4)
    if end < 0:
        return out
    fm = text[3:end].strip()
    body = text[end + 4 :].lstrip("\n")
    out["body"] = body
    for line in fm.splitlines():
        line = line.rstrip()
        if line.startswith("name:"):
            out["slug"] = line.split(":", 1)[1].strip()
        elif line.startswith("description:"):
            out["description"] = line.split(":", 1)[1].strip()
        elif line.startswith("version:"):
            out["version"] = line.split(":", 1)[1].strip()
        elif "category:" in line:
            out["category"] = line.split(":", 1)[1].strip()
        elif "tags:" in line:
            raw = line.split(":", 1)[1].strip().strip("[]")
            out["tags"] = [t.strip() for t in raw.split(",") if t.strip()]
    # title = first H1 in body
    for ln in body.splitlines():
        if ln.startswith("# "):
            out["title"] = ln[2:].strip()
            break
    return out


def cmd_rebuild_index(args: argparse.Namespace) -> int:
    if not SKILLS_ROOT.exists():
        print(f"  {SKILLS_ROOT} missing; nothing to index")
        SKILLS_ROOT.mkdir(parents=True, exist_ok=True)
    conn = _connect()
    try:
        _ensure_schema(conn)
        conn.execute("DELETE FROM skills")
        conn.commit()
        n = 0
        now = _now_iso()
        for md_path in SKILLS_ROOT.rglob("SKILL.md"):
            text = md_path.read_text()
            parsed = _parse_skill_md(text)
            slug = parsed["slug"] or md_path.parent.name
            category = parsed["category"] or md_path.parent.parent.name
            title = parsed["title"] or slug
            conn.execute(
                "INSERT INTO skills(slug, category, name, description, tags, version, body, path, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (slug, category, title, parsed["description"], ",".join(parsed["tags"]), parsed["version"], text, str(md_path), now, now),
            )
            n += 1
        conn.commit()
        print(f"  rebuilt index from {n} SKILL.md file(s) under {SKILLS_ROOT}")
    except sqlite3.Error as e:
        print(f"DB ERROR: {e}", file=sys.stderr)
        return 2
    finally:
        conn.close()
    return 0


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--name", help="skill slug (lowercase, hyphens)")
    p.add_argument("--category", choices=sorted(VALID_CATEGORIES), help="skill category")
    p.add_argument("--tags", help="comma-separated tags")
    p.add_argument("--problem", help="one-line problem statement (becomes default when-to-use)")
    p.add_argument("--solution-file", help="path to markdown file with sections Procedure/Pitfalls/Verification")
    p.add_argument("--description", help="one-line description (overrides solution-file Description)")
    p.add_argument("--when-to-use", help="trigger conditions text (overrides solution-file When to Use)")
    p.add_argument("--version", default=DEFAULT_VERSION, help="semver string (default 1.0.0)")
    p.add_argument("--update", action="store_true", help="allow overwriting an existing skill")

    p.add_argument("--list", action="store_true", help="list indexed skills (optional --category filter)")
    p.add_argument("--search", help="FTS5 query, return top-N ranked")
    p.add_argument("--top", type=int, default=5, help="search result limit (default 5)")
    p.add_argument("--rebuild-index", action="store_true", help="rescan ~/.claude/skills/ and rebuild SQLite index")

    args = p.parse_args(argv)

    if args.rebuild_index:
        return cmd_rebuild_index(args)
    if args.search:
        return cmd_search(args)
    if args.list:
        return cmd_list(args)
    if args.name or args.solution_file:
        missing = [k for k in ("name", "category", "solution_file") if not getattr(args, k)]
        if missing:
            print(f"ERROR: create mode requires --name --category --solution-file (missing: {missing})", file=sys.stderr)
            return 1
        return cmd_create(args)
    p.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

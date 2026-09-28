#!/usr/bin/env python3
"""
hermes_reflect.py — Closed learning loop (HERMES Win #3).

WHY THIS EXISTS:
- HERMES "closed learning loop": every ~15 tool calls or when the user wraps a
  long task, pause and ask "what did I learn that should become a SKILL.md?"
- Goal: convert recurring command sequences, recurring error→fix pairs, and
  recurring file-path/process patterns into CANDIDATE skill drafts that I
  (or a human) can then promote via `hermes_skill_writer.py`.

INPUTS:
- Session logs under /home/jupiter/teleclaude-main/logs/agent-*.log
  (the messaging-bridge daily log — this is what exists today)
- Bridge logs under /home/jupiter/teleclaude-main/logs/bridge-*.log
- Anything else passed via --log-glob

HEURISTICS:
- Repeated commands: same shell command (normalized) appearing 2+ times across
  the last N days = candidate procedural skill.
- Repeated error tokens: words like "FAIL", "ERROR", "denied", "zombie",
  "stale", "MLflow", "Ray" appearing in proximity to a fix line.
- Recurring HC references: HC #NNN mentioned 3+ times = candidate
  "remember-HC-NNN" skill.
- Recurring file paths: same absolute path touched 3+ times across the window.

OUTPUT:
- Markdown report to stdout (or --out file) with:
  ## Candidate N — <slug suggestion>
  Why: ...
  Evidence: ...
  Draft procedure: ...
  Suggested writer call: python3 scripts/hermes_skill_writer.py --name ... --category ... --solution-file ...

USAGE:
  python3 scripts/hermes_reflect.py                       # last 24h
  python3 scripts/hermes_reflect.py --since 72h --top 15
  python3 scripts/hermes_reflect.py --log-glob '/home/jupiter/teleclaude-main/logs/agent-*.log'
  python3 scripts/hermes_reflect.py --out /tmp/reflect.md

Stdlib only. No deps.
"""

from __future__ import annotations

import argparse
import glob
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

DEFAULT_GLOBS = [
    "/home/jupiter/teleclaude-main/logs/agent-*.log",
    "/home/jupiter/teleclaude-main/logs/bridge-*.log",
    "/home/jupiter/Lvl3Quant/logs/sessions/*.log",
]

HC_RE = re.compile(r"HC\s*#?\s*(\d{2,4})")
PATH_RE = re.compile(r"(/home/jupiter/[A-Za-z0-9_./-]+)")
CMD_RE = re.compile(r"`([^`\n]{3,200})`")
ERROR_TOKENS = (
    "ERROR", "FAIL", "denied", "zombie", "stale", "killed",
    "missing", "not found", "timeout", "drift", "leak",
)


def _parse_since(since: str) -> timedelta:
    """Accept '24h', '72h', '7d', '30m'. Returns timedelta."""
    m = re.match(r"^(\d+)\s*([smhd])$", since.strip().lower())
    if not m:
        raise ValueError(f"bad --since {since!r}; use e.g. 24h, 7d, 30m")
    n, unit = int(m.group(1)), m.group(2)
    return timedelta(seconds=n * {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit])


def _gather_log_files(globs: list[str], cutoff_ts: float) -> list[Path]:
    files: list[Path] = []
    for g in globs:
        for p in glob.glob(g):
            pp = Path(p)
            try:
                if pp.is_file() and pp.stat().st_mtime >= cutoff_ts:
                    files.append(pp)
            except OSError:
                pass
    return sorted(set(files), key=lambda p: p.stat().st_mtime)


def _normalize_cmd(cmd: str) -> str:
    """Strip timestamps, PIDs, hashes so 'same command' collisions group."""
    s = cmd.strip()
    s = re.sub(r"\b\d{4}-\d{2}-\d{2}T?\d*:?\d*:?\d*Z?\b", "<DATE>", s)
    s = re.sub(r"\b\d{6,}\b", "<NUM>", s)
    s = re.sub(r"\b[a-f0-9]{8,}\b", "<HASH>", s)
    s = re.sub(r"\s+", " ", s)
    return s[:200]


def _scan_file(path: Path, since_dt: datetime, agg: dict) -> int:
    """Update aggregator counters from this log file. Returns lines scanned."""
    try:
        text = path.read_text(errors="ignore")
    except OSError:
        return 0
    n_lines = 0
    for line in text.splitlines():
        n_lines += 1
        # HC mentions
        for m in HC_RE.finditer(line):
            agg["hc"][m.group(1)] += 1
        # paths
        for m in PATH_RE.finditer(line):
            p = m.group(1).rstrip(".,;:)")
            if len(p) <= 200:
                agg["paths"][p] += 1
        # commands (backticked)
        for m in CMD_RE.finditer(line):
            agg["cmds"][_normalize_cmd(m.group(1))] += 1
        # error proximity
        for tok in ERROR_TOKENS:
            if tok in line:
                agg["errors"][tok] += 1
                break
    return n_lines


def _build_candidates(agg: dict, top: int) -> list[dict]:
    cands: list[dict] = []

    # Repeated commands (>= 2 occurrences)
    for cmd, n in agg["cmds"].most_common(top):
        if n < 2:
            break
        cands.append({
            "kind": "repeated-command",
            "slug": f"cmd-{re.sub(r'[^a-z0-9]+', '-', cmd.lower())[:40].strip('-')}",
            "why": f"command ran {n} time(s) in the window",
            "evidence": cmd,
            "category": "infra",
            "procedure": f"1. Run: `{cmd}`\n2. Verify exit code 0.",
        })

    # Recurring HC references
    for hc, n in agg["hc"].most_common(top):
        if n < 3:
            break
        cands.append({
            "kind": "recurring-hc",
            "slug": f"hc-{hc}-remember",
            "why": f"HC #{hc} referenced {n} time(s)",
            "evidence": f"HC #{hc}",
            "category": "general",
            "procedure": f"1. Re-read DIRECTIVES.md section `HC #{hc}` at session start.\n2. Apply its binding rules before acting.",
        })

    # Recurring paths
    for path, n in agg["paths"].most_common(top):
        if n < 3:
            break
        cands.append({
            "kind": "recurring-path",
            "slug": f"path-{re.sub(r'[^a-z0-9]+', '-', Path(path).name.lower())[:40].strip('-') or 'file'}",
            "why": f"path touched {n} time(s)",
            "evidence": path,
            "category": "infra",
            "procedure": f"1. Inspect / update `{path}`.\n2. Document why it's hot.",
        })

    # Hot error tokens
    for tok, n in agg["errors"].most_common(5):
        if n < 3:
            continue
        cands.append({
            "kind": "recurring-error",
            "slug": f"error-{tok.lower()}-playbook",
            "why": f"'{tok}' appeared {n} time(s) — likely recurring failure mode",
            "evidence": tok,
            "category": "debug",
            "procedure": f"1. Grep recent logs for `{tok}`.\n2. Identify root cause.\n3. Apply fix.",
        })

    return cands[:top]


def _render_markdown(cands: list[dict], files: list[Path], since_h: float) -> str:
    out: list[str] = []
    out.append(f"# Hermes Reflection Report")
    out.append("")
    out.append(f"- Generated: {datetime.now(timezone.utc).isoformat()}")
    out.append(f"- Window: last {since_h:.1f}h")
    out.append(f"- Log files scanned: {len(files)}")
    out.append(f"- Candidate skills: {len(cands)}")
    out.append("")
    if not cands:
        out.append("_No candidate skills detected. Either the window is empty or thresholds (2+ commands / 3+ HCs / 3+ paths / 3+ errors) weren't met._")
        return "\n".join(out)
    for i, c in enumerate(cands, 1):
        out.append(f"## Candidate {i} — `{c['slug']}` ({c['kind']})")
        out.append(f"- **Why**: {c['why']}")
        out.append(f"- **Evidence**: `{c['evidence']}`")
        out.append(f"- **Suggested category**: {c['category']}")
        out.append("")
        out.append("**Draft procedure:**")
        out.append("")
        out.append("```")
        out.append(c["procedure"])
        out.append("```")
        out.append("")
        out.append("**Suggested writer call:**")
        out.append("")
        out.append("```bash")
        out.append(
            f"python3 /home/jupiter/Lvl3Quant/scripts/hermes_skill_writer.py \\\n"
            f"  --name {c['slug']} \\\n"
            f"  --category {c['category']} \\\n"
            f"  --tags reflect,{c['kind']} \\\n"
            f"  --problem {c['why']!r} \\\n"
            f"  --solution-file <path-to-draft.md>"
        )
        out.append("```")
        out.append("")
    return "\n".join(out)


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--since", default="24h", help="time window (e.g. 24h, 7d, 30m). Default 24h.")
    p.add_argument("--top", type=int, default=10, help="max candidates per kind (default 10).")
    p.add_argument("--log-glob", action="append", help="additional log glob (repeatable). Defaults to teleclaude-main + Lvl3Quant sessions.")
    p.add_argument("--out", help="write report here instead of stdout.")
    args = p.parse_args(argv)

    try:
        delta = _parse_since(args.since)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    cutoff_dt = datetime.now(timezone.utc) - delta
    cutoff_ts = cutoff_dt.timestamp()
    globs = list(args.log_glob) if args.log_glob else list(DEFAULT_GLOBS)
    files = _gather_log_files(globs, cutoff_ts)

    agg = {
        "cmds": Counter(),
        "hc": Counter(),
        "paths": Counter(),
        "errors": Counter(),
    }
    total_lines = 0
    t0 = time.perf_counter()
    for f in files:
        total_lines += _scan_file(f, cutoff_dt, agg)
    dt = time.perf_counter() - t0

    cands = _build_candidates(agg, args.top)
    report = _render_markdown(cands, files, delta.total_seconds() / 3600.0)
    report += f"\n\n_Scanned {total_lines} lines across {len(files)} file(s) in {dt*1000:.1f} ms._\n"

    if args.out:
        Path(args.out).write_text(report)
        print(f"  wrote report to {args.out} ({len(report)} bytes, {len(cands)} candidates)")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

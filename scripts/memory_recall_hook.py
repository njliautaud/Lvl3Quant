#!/usr/bin/env python3
"""
memory_recall_hook.py — UserPromptSubmit hook.

When the user sends a message, this hook extracts content-bearing keywords and
greps DIRECTIVES.md, SESSION_STATE.md, RUN_HISTORY.md, IMPROVEMENT_BACKLOG.md
for relevant prior decisions. Top hits (with surrounding context) are injected
back into Claude's context as a system reminder.

Goal: turn "memory recall" from a tool I have to remember to call into
something that automatically primes me with relevant prior decisions before
I respond to the user.

Output contract: stdout = JSON with hookSpecificOutput.additionalContext.
Silent (no output) if no useful hits found — avoid noise.
"""

import json
import os
import re
import sys

LVL3 = "/home/jupiter/Lvl3Quant"
SOURCES = [
    ("DIRECTIVES", f"{LVL3}/DIRECTIVES.md"),
    ("SESSION_STATE", f"{LVL3}/SESSION_STATE.md"),
    ("RUN_HISTORY", f"{LVL3}/RUN_HISTORY.md"),
    ("BACKLOG", f"{LVL3}/IMPROVEMENT_BACKLOG.md"),
]

# Stopwords to skip when extracting keywords from user message
STOP = set("""
a an and are as at be but by can could do does for from has have how i if in
is it its just like me my no not of on or our she so that the their them
then there these they this to too us was we were what when where which who
why will with would you your yours yes ok please now also any all how
hello hi hey thanks thank
""".split())


def extract_keywords(msg: str) -> list[str]:
    """Pull content-bearing tokens 4+ chars. Lowercase, dedup."""
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_\-]{3,}", msg.lower())
    seen = []
    for t in tokens:
        if t in STOP or t in seen:
            continue
        seen.append(t)
    return seen[:20]  # cap


def search_file(path: str, keywords: list[str], context_lines: int = 2) -> list[tuple[int, str]]:
    """Return (line_no, snippet) tuples for paragraphs/blocks that hit >= 2 keywords
    OR hit a single high-signal HC #N pattern from the message."""
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception:
        return []

    # Also pull explicit HC references from message keywords
    hc_pattern = re.compile(r"hc\s*#?\s*(\d{2,4})", re.I)
    hc_targets = set()
    for kw in keywords:
        m = hc_pattern.match(kw)
        if m:
            hc_targets.add(m.group(1))

    hits = []
    for i, line in enumerate(lines):
        low = line.lower()
        # count distinct keyword hits in this line
        kw_hits = sum(1 for kw in keywords if len(kw) >= 5 and kw in low)
        # also match explicit HC numbers
        hc_in_line = bool(re.search(r"hc\s*#\s*\d{2,4}", line, re.I))
        hc_match = any(f"#{n}" in line or f"# {n}" in line for n in hc_targets) if hc_targets else False

        # Relevance raised (token hygiene, EFFICIENCY_UPGRADE §4): need >=3 distinct
        # keyword hits, OR an explicit HC# match. Cuts low-signal noise.
        if kw_hits >= 3 or hc_match:
            start = max(0, i - context_lines)
            end = min(len(lines), i + context_lines + 1)
            snippet = "".join(lines[start:end]).rstrip()
            hits.append((i + 1, snippet, kw_hits + (5 if hc_match else 0)))

    # Sort by score, take top 2, dedup overlapping line ranges
    hits.sort(key=lambda x: -x[2])
    chosen = []
    chosen_lines = set()
    for line_no, snippet, _ in hits:
        if line_no in chosen_lines:
            continue
        chosen.append((line_no, snippet))
        # mark nearby lines as taken to avoid duplicates
        for n in range(line_no - 5, line_no + 5):
            chosen_lines.add(n)
        if len(chosen) >= 2:
            break
    return chosen


def main():
    try:
        payload = json.load(sys.stdin)
    except Exception:
        sys.exit(0)

    msg = payload.get("prompt", "") or payload.get("user_prompt", "") or ""
    if not msg or len(msg) < 8:
        sys.exit(0)

    keywords = extract_keywords(msg)
    if not keywords:
        sys.exit(0)

    # Token-hygiene caps (EFFICIENCY_UPGRADE §4): this injects into the CACHED PREFIX
    # and is re-read every subsequent turn, so keep it small + high-signal.
    SNIPPET_MAX = 250      # per-snippet char cap (was 600)
    TOTAL_MAX = 1800       # total injected char cap (~450 tokens)

    sections = []
    for label, path in SOURCES:
        hits = search_file(path, keywords)
        if not hits:
            continue
        block = f"### {label}\n"
        for line_no, snippet in hits:
            if len(snippet) > SNIPPET_MAX:
                snippet = snippet[:SNIPPET_MAX] + "…"
            block += f"  [L{line_no}] {snippet}\n"
        sections.append(block.rstrip())

    if not sections:
        sys.exit(0)

    body = "\n".join(sections)
    if len(body) > TOTAL_MAX:
        body = body[:TOTAL_MAX] + "…"

    reminder = (
        "[MEMORY RECALL] Possibly-relevant prior context (grep by your keywords):\n"
        + body
        + "\n[Ground your response on this; flag if the user contradicts a prior decision.]"
    )

    out = {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": reminder,
        }
    }
    json.dump(out, sys.stdout)


if __name__ == "__main__":
    main()

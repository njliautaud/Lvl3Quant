#!/usr/bin/env python3
"""
discord_message_lint.py — PreToolUse hook for mcp__discord__send_to_discord.

Enforces HC #433 (simple summaries, no garbage data) and HC #393 (no waiting
for user approval) BEFORE the message is sent. If banned patterns are found,
returns a JSON output that injects a system reminder telling Claude to rewrite.

Hook contract (Claude Code):
  stdin = JSON with {tool_name, tool_input, ...}
  stdout = JSON with {hookSpecificOutput: {additionalContext: "..."}}
  exit 0 = allow tool call to proceed
  exit 2 = block tool call (rare — use the additionalContext nudge first)

Strategy: NUDGE not BLOCK. The hook never silently blocks Discord sends —
the user must always hear from us. Instead it injects a strong reminder that
Claude must self-correct before the NEXT send.
"""

import json
import re
import sys

# --- HC #433: banned content in Discord messages ---
PATH_PATTERN = re.compile(
    r"(?:^|[\s/`'\"])"
    r"(?:/(?:home|tmp|var|usr|opt|mnt|root|nick|jupiter|claude)/[A-Za-z0-9_./\-]+"
    r"|[A-Z]:\\\\?[A-Za-z0-9_.\\\\\-]+"
    r"|[A-Za-z0-9_\-]+\.(?:py|sh|json|yaml|yml|csv|parquet|npz|pt|ckpt|log|txt|md|js|ts))"
)
PID_PATTERN = re.compile(r"\b(?:PID|pid)[\s:=]*\d{3,}\b|\bprocess\s+\d{4,}\b")
HASH_PATTERN = re.compile(r"\b[a-f0-9]{12,}\b")
RESET_COUNTER = re.compile(r"\b(?:reset|restart|run)\s*#\s*\d{2,}\b", re.I)
HEARTBEAT_TS = re.compile(r"\bheartbeat[\s:=]+\d", re.I)
BARE_HC = re.compile(r"\bHC\s*#\s*\d{2,3}\b(?!\s*[—\-:])")  # HC #433 with no plain-English follow-up

# --- HC #393: banned wait-for-approval patterns ---
WAIT_PATTERNS = [
    re.compile(r"awaiting\s+(?:your\s+)?(?:approval|decision|sign[\s\-]?off|reply|response)", re.I),
    re.compile(r"standing\s+by(?:\s+(?:on|for))?", re.I),
    re.compile(r"(?:want|do you want|should|would you like)\s+me\s+to\b", re.I),
    re.compile(r"pending\s+(?:your|user)\s+decision", re.I),
    re.compile(r"need\s+your\s+(?:sign[\s\-]?off|approval|call)", re.I),
    re.compile(r"\blet me know\b", re.I),
    re.compile(r"\bping me if\b", re.I),
    re.compile(r"\byour\s+call\b", re.I),
]

# Acceptable phrasings (HC #393 allows reporting + 10-min interrupt window)
ALLOWED_OVERRIDES = [
    re.compile(r"interrupt within \d+ ?min", re.I),
    re.compile(r"defaulting to [A-Z]", re.I),
    re.compile(r"will report", re.I),
]


def strip_quoted(text: str) -> str:
    """Remove content inside backticks (`...`) and code fences (```...```).
    Meta-discussion of the rules themselves should not trigger violations.
    Double-quoted strings are LEFT IN — those are usually user-facing content."""
    # Code fences first
    text = re.sub(r"```[\s\S]*?```", " ", text)
    # Inline backticks
    text = re.sub(r"`[^`\n]*`", " ", text)
    return text


def scan(message: str) -> list[str]:
    """Return list of plain-English violation descriptions."""
    violations = []
    # Scan the de-backticked version so meta-quotes of banned terms don't fire
    message = strip_quoted(message)

    # HC #433 checks
    paths = PATH_PATTERN.findall(message)
    if paths:
        sample = paths[0].strip()[:60]
        violations.append(f"contains file path/script name (e.g. '{sample}') — HC #433 forbids")

    if PID_PATTERN.search(message):
        violations.append("contains PID/process number — HC #433 forbids in Discord")

    hashes = HASH_PATTERN.findall(message)
    # Filter out obvious non-hash strings (timestamps, common words)
    real_hashes = [h for h in hashes if not h.isdigit()]
    if real_hashes:
        violations.append(f"contains hex hash ('{real_hashes[0][:12]}...') — HC #433 forbids")

    if RESET_COUNTER.search(message):
        violations.append("contains reset/restart counter — HC #433 forbids")

    if HEARTBEAT_TS.search(message):
        violations.append("contains heartbeat timestamp — HC #433 forbids")

    bare_hc = BARE_HC.findall(message)
    if bare_hc:
        violations.append(
            f"references {bare_hc[0]} without plain-English translation — HC #433 "
            f"requires e.g. 'the 40-day rule' not just 'HC #428'"
        )

    # HC #393 checks (only if no override clause present)
    has_override = any(p.search(message) for p in ALLOWED_OVERRIDES)
    if not has_override:
        for p in WAIT_PATTERNS:
            m = p.search(message)
            if m:
                violations.append(
                    f"contains wait-for-approval pattern ('{m.group(0)}') — HC #393 "
                    f"forbids. Use 'I'm doing X. Will report Y.' or default + interrupt window."
                )
                break  # one wait-pattern hit is enough to flag

    return violations


def main():
    try:
        payload = json.load(sys.stdin)
    except Exception:
        # Malformed input — allow through, don't break Discord
        sys.exit(0)

    tool_name = payload.get("tool_name", "")
    if tool_name != "mcp__discord__send_to_discord":
        sys.exit(0)

    tool_input = payload.get("tool_input", {}) or {}
    message = tool_input.get("message", "") or ""
    if not message.strip():
        sys.exit(0)

    violations = scan(message)
    if not violations:
        sys.exit(0)

    # Inject a strong reminder. Do NOT block — the message still goes out
    # (user always hears from us), but Claude gets a follow-up nudge.
    reminder = (
        "[DISCORD LINT] The message you just sent contains HC #433/#393 "
        "violations:\n  - "
        + "\n  - ".join(violations)
        + "\n\nThis message went through, but rewrite your NEXT message to fix this. "
        "Self-test every send_to_discord BEFORE calling the tool. "
        "Patterns to scan: file paths, PID numbers, hex hashes, reset counters, "
        "bare 'HC #N' references, 'awaiting/standing by/should I/want me to'."
    )

    out = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "additionalContext": reminder,
        }
    }
    json.dump(out, sys.stdout)
    sys.exit(0)


if __name__ == "__main__":
    main()

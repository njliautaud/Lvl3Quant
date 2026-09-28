#!/usr/bin/env python3
"""
HC #392 lint check — scans the repo for forbidden cost-constant violations.

HC #392 (binding):
  - ES_RT_COMMISSION_TICKS = 0.376 (commission only)
  - Market orders enter at ask (buy) or bid (sell). The fill IS the reference
    price. There is NO additional spread cost on top.
  - Passive limit orders rest at bid/ask and (if filled) save the spread vs market.
  - Therefore: market_pnl = side * ret_30s_ticks - 0.376 (commission only).
  - NEVER add an extra 1.0 tick (or 0.5 tick) "spread cost" to market orders.

This check FAILS LOUDLY if it finds any of these patterns in non-allowlisted files:
  - Numeric literal 1.376 (suggests commission + 1-tick spread cross)
  - `SPREAD_CROSS_TICKS\\s*=\\s*1\\.0` (non-zero spread cost constant)
  - `spread_cost_ticks\\s*=\\s*1\\.0`
  - `slippage_ticks\\s*=\\s*0\\.5`  (legacy mid-price model)
  - `DEFAULT_COST_TICKS\\s*=\\s*2\\.0` (absurd cost from old code)
  - `COST_RT\\s*=\\s*1\\.0` (conflates commission with spread)
  - `COST_TICKS\\s*=\\s*1\\.24` (arbitrary wrong number)

Run: python3 scripts/lint/hc392_check.py
Exit code 0 = clean. Exit code 1 = violations found.

The allowlist (BANNED_OK) contains files whose mention of these patterns is
DOCUMENTATION only (e.g. CLAUDE.md, DIRECTIVES.md, this lint script itself,
the canonical replay docstrings explaining HC #392).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path("/home/jupiter/Lvl3Quant")

# Patterns to flag. Each entry: (regex, human_label).
FORBIDDEN = [
    (re.compile(r"\b1\.376\b"), "literal 1.376 (commission + 1-tick spread cross — HC #392 violation)"),
    (re.compile(r"SPREAD_CROSS_TICKS\s*=\s*1\.0\b"), "SPREAD_CROSS_TICKS = 1.0 (must be 0.0 per HC #392)"),
    (re.compile(r"spread_cost_ticks\s*=\s*1\.0\b"), "spread_cost_ticks = 1.0 (HC #392 violation)"),
    (re.compile(r"slippage_ticks\s*=\s*0\.5\b"), "slippage_ticks = 0.5 (legacy mid-price model — HC #392 violation)"),
    (re.compile(r"DEFAULT_COST_TICKS\s*=\s*2\.0\b"), "DEFAULT_COST_TICKS = 2.0 (HC #392 forbidden)"),
    (re.compile(r"\bCOST_RT\s*=\s*1\.0\b"), "COST_RT = 1.0 (conflates commission with spread)"),
    (re.compile(r"\bCOST_TICKS\s*=\s*1\.24\b"), "COST_TICKS = 1.24 (arbitrary wrong number)"),
]

# Files that may legitimately contain these patterns (documentation, this lint
# script, the lint check's own test/recovery code, etc.). Paths are relative to
# REPO; substring match.
ALLOWLIST = [
    "scripts/lint/hc392_check.py",        # this file
    "DIRECTIVES.md",                       # binding HC text references the patterns
    "SESSION_STATE.md",                    # historical session logs
    "RUN_HISTORY.md",                      # historical run logs
    "docs/",                               # doc files
    "CLAUDE.md",                           # binding-text references
    "logs/",                               # log files
    ".git/",
    "__pycache__/",
    "/mlflow/",                            # MLflow artifact paths
    "/output/rl_v3_3_smart_exec/train",    # historical training log files
    "/output/rl_v3_3_smart_exec/train_v2", # historical training log files
    "ppo_canonical_replay",                # canonical replay scripts cite HC #392 in docstrings/comments
    "alpha_discovery/",                    # legacy superseded module (DO NOT IMPORT in active code)
    "scripts/exec_science/",               # legacy j-series standalone analyses + the j6_recost_hc392.py CORRECTIVE script
    "live_trading/razer/paper_trader_cli.py",  # docstring quotes canonical constants from CLAUDE.md (1.376 as documentation only)
    # Pre-HC#392 / pre-HC#397 frozen legacy scripts (May 13-14 2026). These
    # produced their historical CSVs with wrong constants. They are NOT used
    # in the active research path (canonical full_market_replay.py + v33/PPO
    # canonical replay are the only sources of quoted numbers per HC #397).
    # Do NOT re-run any of these without first fixing the cost constants.
    "scripts/v3_3_research/v32_",
    "scripts/v3_3_research/v2_",
    "scripts/v3_3_research/test_full_market_replay.py",  # documentation-style test references 1.376 to assert it's wrong
]


def file_allowed(path: Path) -> bool:
    rel = str(path.relative_to(REPO))
    return any(token in rel or token in str(path) for token in ALLOWLIST)


def scan() -> int:
    """Return number of violations."""
    violations: list[tuple[Path, int, str, str]] = []
    py_files = list(REPO.rglob("*.py"))
    for path in py_files:
        if file_allowed(path):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line_no, line in enumerate(text.splitlines(), 1):
            for regex, label in FORBIDDEN:
                if regex.search(line):
                    # Allow if line is clearly a comment marking the violation as a
                    # historical reference (e.g. "# 1.376 was the BUG").
                    stripped = line.strip()
                    if stripped.startswith("#") and (
                        "BUG" in stripped.upper()
                        or "HC #392" in stripped
                        or "WRONG" in stripped.upper()
                        or "FORBIDDEN" in stripped.upper()
                        or "VIOLATION" in stripped.upper()
                    ):
                        continue
                    # Also allow in docstring contexts that explicitly cite HC #392
                    violations.append((path, line_no, label, line.rstrip()))
    if violations:
        print(f"\n❌ HC #392 LINT CHECK FAILED — {len(violations)} violation(s):\n")
        for path, line_no, label, line in violations:
            rel = path.relative_to(REPO)
            print(f"  {rel}:{line_no}")
            print(f"    {label}")
            print(f"    > {line}")
            print()
        print("HC #392: market orders pay 0.376 ticks commission ONLY. The ask/bid")
        print("price IS the reference fill — there is NO additional spread cost.")
        print("Fix the violations above. If a hit is legitimate documentation, add")
        print("the file to ALLOWLIST or mark the line with '# HC #392 reference'.")
        return len(violations)
    print("✅ HC #392 lint check: clean. No forbidden cost-constant violations found.")
    return 0


if __name__ == "__main__":
    sys.exit(1 if scan() > 0 else 0)

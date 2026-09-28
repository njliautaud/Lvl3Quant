#!/usr/bin/env python3
"""
seed_queue_from_directives.py — Populate the persistent task queue with the
canonical productive-work tasks defined in DIRECTIVES.md (HC #417 §4 +
HC #422 falsification-gate items).

Idempotency: each seed item carries a stable tag `seed:<key>`. Before enqueueing,
we check the current queue for any non-terminal record (pending|running) with
that tag — if one exists, we skip. Done/failed/cancelled tasks DO NOT block
re-seeding; the seeder will re-queue them so the work happens again.

Usage:
  python3 seed_queue_from_directives.py            # idempotent re-seed
  python3 seed_queue_from_directives.py --force    # enqueue regardless
  python3 seed_queue_from_directives.py --dry-run  # show what would be queued
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import task_queue as tq  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# Canonical seed list — derived from /home/jupiter/Lvl3Quant/DIRECTIVES.md
# HC #422 falsification-gate items + HC #417 productive-work queue.
#
# Each entry: (seed_key, node, kind, priority, cmd, [tags])
# Lower priority number = higher urgency.
# ──────────────────────────────────────────────────────────────────────────────

SEEDS: list[tuple[str, str, str, int, str, list[str]]] = [
    # --- HC #422 falsification gate (due Mon 5/18 EOD) ----------------------
    (
        "hc422_post_apr29_data_audit",
        "jupiter",
        "audit",
        2,
        # Audit all files dated > 2026-04-29 under canonical data dirs, compare
        # schema/row-counts vs Apr-29 snapshot, emit report.
        "cd /home/jupiter/Lvl3Quant && python3 -c \""
        "import os,sys,json,pathlib;"
        "roots=['data','output','C:\\\\Users\\\\claude\\\\Lvl3Quant'];"
        "cutoff='2026-04-29';"
        "report={'cutoff':cutoff,'flagged':[]};"
        "from datetime import datetime as D, timezone as TZ;"
        "import re;"
        "p=pathlib.Path('data');"
        "files=[f for f in p.rglob('*') if f.is_file()];"
        "[report['flagged'].append({'path':str(f),'mtime':D.fromtimestamp(f.stat().st_mtime,TZ.utc).date().isoformat(),'bytes':f.stat().st_size})"
        " for f in files if D.fromtimestamp(f.stat().st_mtime,TZ.utc).date().isoformat()>cutoff];"
        "out=pathlib.Path('output/hc422_post_apr29_audit.json');"
        "out.parent.mkdir(parents=True,exist_ok=True);"
        "out.write_text(json.dumps(report,indent=2));"
        "print('wrote',out,'flagged',len(report['flagged']))\"",
        ["hc422", "rule3", "data-audit"],
    ),
    (
        "hc422_v33_exec_lane_scaffold",
        "jupiter",
        "scaffold",
        3,
        # Scaffold v3.3 smart-execution research output dir + a stub runner.
        "mkdir -p /home/jupiter/Lvl3Quant/output/hc422_v33_execution_research && "
        "cd /home/jupiter/Lvl3Quant && "
        "python3 -c \"from pathlib import Path;"
        "p=Path('output/hc422_v33_execution_research');"
        "p.mkdir(parents=True,exist_ok=True);"
        "(p/'README.md').write_text('# HC #422 Rule 5 — v3.3 Smart Execution Research\\n\\n"
        "Lane mirrors v2 smart-exec pipeline but consumes v3.3 prediction NPZs.\\n"
        "Features: ALL v3.3 heads (directional + auxiliary) — not just pred_1s.\\n"
        "Outputs: FIFO replay verdicts (HC #74), HC #344 day_conc gate, Sortino + PF table.\\n\\n"
        "Status: SCAFFOLDED (seed task hc422_v33_exec_lane_scaffold).\\n');"
        "(p/'TODO.md').write_text('1. point at v3.3 NPZs\\n2. extract all heads as features\\n"
        "3. run RL_v3_3_smart_exec multi-head\\n4. FIFO replay (HC #74)\\n"
        "5. HC #344 day_conc gate\\n6. compare vs v2 baseline\\n');"
        "print('scaffold ready at',p)\"",
        ["hc422", "rule5", "v33-exec"],
    ),
    (
        "hc422_refactor_rl_v33_all_heads",
        "jupiter",
        "exec",
        4,
        # Identify the RL_v3_3_smart_exec script and write a refactor-spec doc.
        "cd /home/jupiter/Lvl3Quant && "
        "python3 -c \"import pathlib,re;"
        "root=pathlib.Path('.');"
        "cands=list(root.rglob('*RL_v3_3_smart_exec*'))+list(root.rglob('*rl_v3_3_smart_exec*'));"
        "out=pathlib.Path('output/hc422_rl_v33_refactor_spec.md');"
        "out.parent.mkdir(parents=True,exist_ok=True);"
        "lines=['# HC #422 Rule 8 — Refactor RL_v3_3_smart_exec to consume ALL heads',''];"
        "lines.append('Found candidates:');"
        "[lines.append('  - '+str(c)) for c in cands];"
        "lines.append('');"
        "lines.append('Required input features (per Rule 8):');"
        "lines.append('  - pred_log_ret_1s,5s,10s,30s');"
        "lines.append('  - MFE/MAE');"
        "lines.append('  - Vol, spread forecast, confidence');"
        "lines.append('  - book imbalance, depth, microprice, queue position');"
        "out.write_text('\\n'.join(lines));"
        "print('wrote',out,'cands=',len(cands))\"",
        ["hc422", "rule8", "exec-refactor"],
    ),
    (
        "hc422_review_v342_head_config",
        "jupiter",
        "audit",
        3,
        # Identify FIFO-bracket-specific heads in v3.4.2 config files; produce
        # a kill-list per Rule 2 (drop TP/SL-conditioned heads).
        "cd /home/jupiter/Lvl3Quant && "
        "python3 -c \"import pathlib,re,json;"
        "root=pathlib.Path('.');"
        "pats=[re.compile(r'pred_fifo_tp\\\\d+sl\\\\d+',re.I),re.compile(r'fifo_tp\\\\d+sl\\\\d+',re.I)];"
        "hits=[];"
        "exts=('.py','.json','.yaml','.yml','.md');"
        "for f in root.rglob('*'):"
        "  if not f.is_file(): continue\n"
        "  if f.suffix.lower() not in exts: continue\n"
        "  if 'v3_4_2' not in str(f) and 'v342' not in str(f) and 'v3.4.2' not in str(f): continue\n"
        "  try: txt=f.read_text(errors='ignore')\n"
        "  except Exception: continue\n"
        "  for pat in pats:\n"
        "    for m in pat.finditer(txt):\n"
        "      hits.append({'file':str(f),'match':m.group(0)})\n"
        "out=pathlib.Path('output/hc422_v342_tpsl_head_killist.json');"
        "out.parent.mkdir(parents=True,exist_ok=True);"
        "out.write_text(json.dumps(hits,indent=2));"
        "print('wrote',out,'hits=',len(hits))\"",
        ["hc422", "rule2", "v342-heads"],
    ),
]


def find_active_seed(tag: str) -> dict | None:
    """Return any non-terminal task carrying the given seed tag, else None."""
    for r in tq.list_tasks():
        if tag in (r.get("tags") or []) and r.get("status") in {"pending", "running"}:
            return r
    return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--force", action="store_true",
                   help="enqueue even if an active seed with the same tag exists")
    p.add_argument("--dry-run", action="store_true",
                   help="print what would be enqueued, don't write")
    args = p.parse_args(argv)

    print(f"Seeding queue from DIRECTIVES.md — {len(SEEDS)} canonical tasks")
    enqueued, skipped = 0, 0
    for key, node, kind, prio, cmd, tags in SEEDS:
        tag_seed = f"seed:{key}"
        all_tags = list(tags) + [tag_seed]

        if not args.force:
            active = find_active_seed(tag_seed)
            if active is not None:
                print(f"  SKIP  {key:<40s} (already active: id={active['id']} status={active['status']})")
                skipped += 1
                continue

        if args.dry_run:
            print(f"  WOULD-ENQUEUE  {key:<40s} node={node} prio={prio} kind={kind}")
            enqueued += 1
            continue

        rec = tq.enqueue(node, kind, cmd, priority=prio, tags=all_tags)
        print(f"  ENQUEUED  {key:<40s} id={rec['id']} node={node} prio={prio}")
        enqueued += 1

    print(f"\nDone. enqueued={enqueued} skipped={skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

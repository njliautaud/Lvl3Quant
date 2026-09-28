# HERMES Procedural Memory — Quick Reference

Two-tool system for crystallizing recurring workflows into reusable SKILL.md files.

## What's installed
- `scripts/hermes_skill_writer.py` — create / list / search / rebuild-index over skills.
- `scripts/hermes_reflect.py` — scan recent session logs for candidate skills to write.
- `~/.claude/skills/<category>/<slug>/SKILL.md` — on-disk procedural memory (source of truth).
- `data/hermes_skills.sqlite` — FTS5-indexed mirror for sub-ms retrieval.

## Session-start ritual
At the top of every new Claude session, run:
```
python3 /home/jupiter/Lvl3Quant/scripts/hermes_skill_writer.py --list
```
This loads procedural memory into context. If a task matches a skill, follow the SKILL.md procedure rather than re-deriving the answer.

## Targeted retrieval mid-session
```
python3 /home/jupiter/Lvl3Quant/scripts/hermes_skill_writer.py --search "fifo cost model"
python3 /home/jupiter/Lvl3Quant/scripts/hermes_skill_writer.py --search "regime stratified oot" --top 3
```
FTS5 query syntax supported (`AND`, `OR`, `NEAR`, quoted phrases).

## Writing a new skill
1. Draft a markdown file with sections `## Description`, `## When to Use`, `## Procedure`, `## Pitfalls`, `## Verification`.
2. Call:
```
python3 /home/jupiter/Lvl3Quant/scripts/hermes_skill_writer.py \
  --name my-new-skill \
  --category trading \
  --tags tag1,tag2 \
  --problem "one-line what this solves" \
  --solution-file /tmp/draft.md
```
Categories: `trading | infra | debug | analysis | general`.
Pass `--update` to overwrite an existing slug.

## Reflection loop (HERMES Win #3)
After ~15 non-trivial tool calls or before context compression:
```
python3 /home/jupiter/Lvl3Quant/scripts/hermes_reflect.py --since 24h --top 10
```
Outputs candidate skills (repeated commands, recurring HCs/paths, hot error tokens). For the ones worth keeping, draft a markdown file and call the writer.

## Rebuilding the FTS index
If you edited a SKILL.md by hand or restored from backup:
```
python3 /home/jupiter/Lvl3Quant/scripts/hermes_skill_writer.py --rebuild-index
```
Walks `~/.claude/skills/` and repopulates the SQLite mirror from on-disk files.

## Seed skills shipped 2026-05-21
- `infra/recovery-cron-restore` — rebuild monitoring crons after session reset
- `trading/fifo-canonical-replay` — HC #74 / HC #470 R7 canonical fill validation
- `analysis/regime-stratified-oot` — HC #428 R1 40-day regime-balanced validation
- `debug/discord-message-rules` — HC #433 plain-English Discord messages
- `infra/ray-job-dispatch` — Ray-only remote dispatch (no SSH for routine)

Source of truth: the SKILL.md files. The SQLite is a derived index — safe to delete and rebuild.

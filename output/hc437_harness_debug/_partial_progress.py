"""Write bracket_47day_progress.json from completed dates in the bracket log."""
import json
import re
import sys
from pathlib import Path

OUT = Path("/home/jupiter/Lvl3Quant/output/hc437_harness_debug")
log = OUT / "logs" / "bracket_47day.log"

if not log.exists():
    print("no log yet"); sys.exit(0)

pat = re.compile(r"(\d{8}): (\d+) fills$")
done = []
for line in log.read_text().splitlines():
    m = pat.search(line)
    if m:
        done.append({"date": m.group(1), "fills": int(m.group(2))})

(OUT / "bracket_47day_progress.json").write_text(json.dumps({
    "mode": "hc413_bracket",
    "n_completed_days": len(done),
    "total_fills": sum(d["fills"] for d in done),
    "completed": done,
}, indent=2))
print(f"wrote {len(done)} completed dates, total fills={sum(d['fills'] for d in done)}")

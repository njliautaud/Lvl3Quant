#!/bin/bash
# HC #31 EOD completeness report — runs at 4:30 PM ET weekdays.
# Posts per-hour event counts + flagged gap buckets to Discord.

LOG=/home/jupiter/Lvl3Quant/logs/mbo_eod_completeness.log
mkdir -p $(dirname "$LOG")

# Suppression-flag guard (HC #518 posture). If the Razer idle suppression flag
# is active, the recorder is intentionally down — skip the alert so we don't
# nag the user every weekday at 4:30 with an already-known state.
SUPPRESS_FLAG=/home/jupiter/Lvl3Quant/logs/idle_watchdog/razer_idle_acceptable.flag
if [ -f "$SUPPRESS_FLAG" ]; then
    VALID_UNTIL=$(grep -oP 'valid_until=\K\d+' "$SUPPRESS_FLAG" 2>/dev/null || echo 0)
    NOW=$(date +%s)
    if [ "$VALID_UNTIL" -gt "$NOW" ]; then
        echo "[$(date)] HC #31 EOD check skipped: Razer suppression flag active (expires $(date -d @$VALID_UNTIL))" >> "$LOG"
        exit 0
    fi
fi

REPORT=$(python3 - <<'PYEOF' 2>&1
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

today = datetime.now().strftime("%Y%m%d")
f = Path(f"/home/jupiter/Lvl3Quant/data/processed/mbo_events/{today}_mbo_events.npz")

if not f.exists():
    msg = f"**🚨 HC #31 EOD CHECK FAILED — no NPZ for {today} exists.**"
else:
    d = np.load(f, allow_pickle=True)
    ts = d["timestamps"].astype(np.int64) // 1_000_000_000
    n = len(ts)

    # Per-hour ET counts
    lines = [f"**📊 HC #31 EOD MBO completeness — {today}**\n",
             f"Total events: {n:,}\n",
             f"First event: {datetime.fromtimestamp(int(ts.min()), tz=timezone.utc).astimezone():%H:%M ET}",
             f"Last event:  {datetime.fromtimestamp(int(ts.max()), tz=timezone.utc).astimezone():%H:%M ET}\n"]

    lines.append("| Hour ET | Events | Status |")
    lines.append("|---|---|---|")
    flag_count = 0
    for hour in range(24):
        h_unix_start = (int(ts.min()) // 86400) * 86400 + hour * 3600 - (4 * 3600)  # crude ET offset
        h_unix_end = h_unix_start + 3600
        cnt = int(((ts >= h_unix_start) & (ts < h_unix_end)).sum())
        if cnt == 0:
            continue
        # Flag if regular-session hour with very low count
        flag = ""
        if 9 <= hour < 16 and cnt < 50000:
            flag = "🚨 LOW"; flag_count += 1
        elif 16 <= hour < 18 and cnt < 5000:
            flag = "⚠️ LOW"; flag_count += 1
        lines.append(f"| {hour:02d}:00 | {cnt:,} | {flag} |")

    # 5-min gap buckets
    edges = list(range(int(ts.min() // 300 * 300), int(ts.max() // 300 * 300) + 600, 300))
    counts, _ = np.histogram(ts, bins=edges)
    gaps = []
    for i, c in enumerate(counts):
        if c < 50:
            t = datetime.fromtimestamp(edges[i], tz=timezone.utc).astimezone()
            h = t.hour
            # Skip overnight + maintenance
            if 9 <= h < 17 or (h == 17 and t.minute >= 30) or 18 <= h <= 23:
                gaps.append(f"{t:%H:%M}: {c} events")

    if gaps:
        lines.append(f"\n**5-min gap buckets during active hours (<50 events): {len(gaps)}**")
        for g in gaps[:10]:
            lines.append(f"  {g}")
        if len(gaps) > 10:
            lines.append(f"  ... +{len(gaps)-10} more")
    else:
        lines.append("\n✅ No 5-min gap buckets during active hours.")

    if flag_count > 0:
        lines.insert(0, f"**🚨 HC #31 ALERT — {flag_count} hours flagged as low-volume during active session.**\n")

    msg = "\n".join(lines)

# Send to inject endpoint
try:
    req = urllib.request.Request(
        "http://127.0.0.1:7731/inject",
        data=json.dumps({"message": msg}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    urllib.request.urlopen(req, timeout=15).read()
    print("Report dispatched to Discord")
except Exception as e:
    print(f"Discord dispatch failed: {e}")
    print(msg)
PYEOF
)

echo "[$(date)] EOD completeness report:" >> "$LOG"
echo "$REPORT" >> "$LOG"

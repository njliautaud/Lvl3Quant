#!/bin/bash
# HC #590 gap fix 2026-06-10: alert on silent pm2 restarts of critical daemons.
# Compares restart counts vs last run; on increase, sends Discord webhook alert.
STATE=/home/jupiter/Lvl3Quant/logs/pm2_restart_watch.state
WATCH="wheel-paper-engine wheel-paper-balanced alert-router qcc-daemon persistent-monitor mlflow-server"
touch "$STATE"
for app in $WATCH; do
  cur=$(pm2 describe "$app" 2>/dev/null | grep -oP 'restarts\s*│\s*\K[0-9]+' | head -1)
  [ -z "$cur" ] && continue
  prev=$(grep "^$app=" "$STATE" | cut -d= -f2)
  if [ -n "$prev" ] && [ "$cur" -gt "$prev" ]; then
    node /home/jupiter/teleclaude-main/utils/webhook_notifier.js \
      "⚠️ pm2 restart detected: $app restarted ($prev → $cur). Check logs if unexpected." 2>/dev/null
  fi
  grep -q "^$app=" "$STATE" && sed -i "s/^$app=.*/$app=$cur/" "$STATE" || echo "$app=$cur" >> "$STATE"
done

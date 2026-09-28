# Self-Audit on Every Deep Check (HC #492)

## When to trigger
On every deep-check / 15-min monitor cycle / recovery / session start.

## What to do
1. Run python3 /home/jupiter/Lvl3Quant/infra/self_audit.py FIRST
2. If any check returns FAIL:
   - GPU idle on weekday: dispatch work from SESSION_STATE Next field BEFORE reporting
   - PM2 processes dead: pm2 restart immediately
   - SESSION_STATE stale: update it with current cluster state
   - Discord spam detected: reduce alert frequency in offending watchdog
3. If any check returns CRITICAL: escalate to user via Discord
4. If all PASS: continue silently

## Edge validator
After any model retrain completes or new predictions are available:
  python3 /home/jupiter/Lvl3Quant/infra/edge_validator.py predictions.npz
If FAIL: the model has lost edge. Do not deploy. Alert user.

## Self-critical questions
- Am I actually advancing research, or just monitoring?
- Is any GPU idle that could be running an experiment?
- Have I reported results without acting on them?

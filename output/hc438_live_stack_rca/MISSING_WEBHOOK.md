# BLOCKER: No Discord Webhook URL Available For Trader Alerts

**Fix #2 (wire DISCORD_WEBHOOK_URL into the live trader) is BLOCKED until the user provides a webhook URL.**

## Where I Looked
- `/home/jupiter/Lvl3Quant/.env` -- does not exist
- `/home/jupiter/teleclaude-main/.env` -- has GEMINI/GITHUB/Rithmic creds, NO Discord webhook
- All source files under `/home/jupiter/Lvl3Quant` matching `discord.com/api/webhooks` -- 0 hits
- All source files under `/home/jupiter/teleclaude-main` matching `discord.com/api/webhooks` -- only `.env.example` and a `docs/ref_system.md` placeholder
- Razer Windows env vars (User + Machine scopes) for `DISCORD_WEBHOOK_URL` -- both empty
- Razer `C:\Users\claude\Lvl3Quant\live_trading\.env` -- Rithmic creds only

## What's Needed
A Discord channel webhook URL (format: `https://discord.com/api/webhooks/<id>/<token>`) for a channel the parent agent monitors. Suggested channels: `#system-status` or `#trading-alerts`.

## Where to Plumb It Once Provided
1. Append to `C:\Users\claude\Lvl3Quant\live_trading\.env` on Razer:
   ```
   DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
   ```
2. Also set as a Windows User-level env var (so it survives reboots and is picked up by scheduled tasks):
   ```powershell
   [Environment]::SetEnvironmentVariable('DISCORD_WEBHOOK_URL', '<URL>', 'User')
   ```
3. Restart the trader (kill PID + relaunch via `\ShadowV2Top05` scheduled task).
4. Verify with: trader log first line should change from `WARNING DiscordNotifier: no webhook URL` to no warning, and a `LIVE START` alert should arrive in Discord within 5 seconds of restart.

## Why This Matters
Without the webhook, every kill-switch alert (manual halt, drift, daily loss cap, IC drift) is silent. Today's silent failure was *enabled* by this gap. Even with the HC #438 kill-switch fix applied, the next genuine failure mode (e.g. daily-loss cap, IC drift detection) will still be invisible.

## Estimated Severity
HIGH. The trader has 9 kill-switches, all of which currently log to file but cannot reach the parent agent. Operating a live trading stack without out-of-band alerting is unsafe.

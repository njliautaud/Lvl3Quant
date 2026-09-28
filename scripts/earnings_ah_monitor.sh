#!/bin/bash
# Earnings After-Hours Monitor — Lightweight bash script
# Checks AAPL/AMZN post-earnings gaps and writes alert file if PEAD conditions met
# Run via: crontab at 16:30, 17:00, 17:30, 18:00 ET on Jul 30

ALERT_FILE="/home/jupiter/Lvl3Quant/state/earnings_alert.json"
LOG_FILE="/home/jupiter/Lvl3Quant/logs/earnings_ah_monitor.log"

echo "$(date '+%Y-%m-%d %H:%M:%S') — Checking AAPL/AMZN post-earnings" >> "$LOG_FILE"

python3 -c "
import json, sys
from datetime import datetime
try:
    import yfinance as yf
except ImportError:
    sys.exit(0)

results = []
for ticker in ['AAPL', 'AMZN']:
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period='2d', interval='1m', prepost=True)
        if hist.empty:
            continue

        # Get regular hours close vs latest after-hours
        info = t.fast_info
        last_price = info.last_price
        prev_close = info.previous_close

        if prev_close and prev_close > 0:
            gap_pct = (last_price - prev_close) / prev_close * 100
            results.append({
                'ticker': ticker,
                'prev_close': round(prev_close, 2),
                'current': round(last_price, 2),
                'gap_pct': round(gap_pct, 2),
                'pead_signal': abs(gap_pct) > 3.0,  # 3%+ gap = potential PEAD
                'direction': 'UP' if gap_pct > 0 else 'DOWN',
                'timestamp': datetime.now().isoformat()
            })
    except Exception as e:
        results.append({'ticker': ticker, 'error': str(e)})

# Write alert if any PEAD signals
pead_signals = [r for r in results if r.get('pead_signal')]
output = {
    'check_time': datetime.now().isoformat(),
    'results': results,
    'pead_alerts': len(pead_signals),
    'alert_triggered': len(pead_signals) > 0
}
with open('$ALERT_FILE', 'w') as f:
    json.dump(output, f, indent=2)

if pead_signals:
    for s in pead_signals:
        print(f'PEAD_ALERT: {s[\"ticker\"]} gapped {s[\"gap_pct\"]:+.1f}% to \${s[\"current\"]}')
else:
    print('No PEAD signals')
" 2>/dev/null >> "$LOG_FILE"

# If alert triggered, wake Claude via autonomy_inject
if python3 -c "
import json, sys
try:
    with open('$ALERT_FILE') as f:
        d = json.load(f)
    if d.get('alert_triggered'):
        # Build message
        alerts = [r for r in d['results'] if r.get('pead_signal')]
        msg = 'EARNINGS_PEAD_ALERT. '
        for a in alerts:
            msg += f'{a[\"ticker\"]} gapped {a[\"gap_pct\"]:+.1f}% AH. '
        msg += 'Assess PEAD trade opportunity. Account: \$667.73 agentic. Kill switch status matters.'
        print(msg)
        sys.exit(0)
    sys.exit(1)
except:
    sys.exit(1)
" 2>/dev/null; then
    ALERT_MSG=$(python3 -c "
import json
with open('$ALERT_FILE') as f:
    d = json.load(f)
alerts = [r for r in d['results'] if r.get('pead_signal')]
msg = 'EARNINGS_PEAD_ALERT. '
for a in alerts:
    msg += f'{a[\"ticker\"]} gapped {a[\"gap_pct\"]:+.1f}% AH. '
msg += 'Assess PEAD trade opportunity. Account: \$667.73 agentic. Kill switch status matters.'
print(msg)
" 2>/dev/null)
    echo "$(date '+%Y-%m-%d %H:%M:%S') — ALERT TRIGGERED: $ALERT_MSG" >> "$LOG_FILE"
    /home/jupiter/Lvl3Quant/scripts/autonomy_inject.sh "$ALERT_MSG" >> "$LOG_FILE" 2>&1
fi

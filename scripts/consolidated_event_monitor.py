#!/usr/bin/env python3
"""
Consolidated Event Monitor v1
==============================
Monitors ALL validated supplementary signals and produces a unified
watchlist for the agentic trading account.

Validated signal types (from research):
1. PEAD (Post-Earnings Announcement Drift) — 5d or 40d hold
2. Stock Split Pre-Run — buy after split announced, hold 10d
3. VIX Spike Fade — enter SPY/QQQ longs when VIX starts declining from spike
4. Sector Drawdown Recovery — buy recovering sector ETFs after >10% drawdown
5. 200-SMA Reclaim — buy stocks/ETFs reclaiming 200-SMA
6. MSFT/AAPL Pairs Reversion — partial mean-reversion alpha (supplementary)
7. Contrarian Sector Reversion — buy sector after mega-cap gap >3%

Kill switch: VIX >20 + SPY < 50-SMA = pause momentum/rotation strategies
"""

import yfinance as yf
import numpy as np
import pandas as pd
import json, os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

STATE_DIR = '/home/jupiter/Lvl3Quant/state'
os.makedirs(STATE_DIR, exist_ok=True)

print("=" * 60)
print("CONSOLIDATED EVENT MONITOR v1")
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M ET')}")
print("=" * 60)

# ── Market Regime ──
spy = yf.download('SPY', period='1y', progress=False)['Close'].dropna()
vix = yf.download('^VIX', period='5d', progress=False)['Close'].dropna()

spy_last = float(spy.iloc[-1].iloc[0] if hasattr(spy.iloc[-1], 'iloc') else spy.iloc[-1])
sma200 = float(spy.iloc[-200:].mean().iloc[0] if hasattr(spy.iloc[-200:].mean(), 'iloc') else spy.iloc[-200:].mean())
sma50 = float(spy.iloc[-50:].mean().iloc[0] if hasattr(spy.iloc[-50:].mean(), 'iloc') else spy.iloc[-50:].mean())
vix_last = float(vix.iloc[-1].iloc[0] if hasattr(vix.iloc[-1], 'iloc') else vix.iloc[-1])

bull_regime = spy_last > sma200
kill_switch = vix_last > 20 and spy_last < sma50

print(f"\n  SPY: ${spy_last:.2f} | 200-SMA: ${sma200:.2f} ({'ABOVE' if bull_regime else 'BELOW'})")
print(f"  VIX: {vix_last:.2f} | Kill Switch: {'ACTIVE' if kill_switch else 'INACTIVE'}")

events = []

# ── 1. VIX Spike Fade Check ──
print("\n--- VIX SPIKE FADE ---")
vix_long = yf.download('^VIX', period='30d', progress=False)['Close'].dropna()
if len(vix_long) >= 5:
    vix_5d_ago = float(vix_long.iloc[-5].iloc[0] if hasattr(vix_long.iloc[-5], 'iloc') else vix_long.iloc[-5])
    vix_peak = float(vix_long.iloc[-5:].max().iloc[0] if hasattr(vix_long.iloc[-5:].max(), 'iloc') else vix_long.iloc[-5:].max())
    vix_declining = vix_last < vix_peak * 0.9  # VIX declined 10% from recent peak

    if vix_peak > 22 and vix_declining:
        events.append({
            'type': 'VIX_SPIKE_FADE',
            'signal': f'VIX peaked at {vix_peak:.1f}, now declining to {vix_last:.1f}',
            'action': 'BUY SPY/QQQ when VIX confirms below 20',
            'confidence': 'SUPPLEMENTARY',
            'blocked': kill_switch
        })
        print(f"  ⚠️ VIX peaked at {vix_peak:.1f}, declining to {vix_last:.1f}")
        print(f"    Wait for VIX < 20 to enter SPY/QQQ longs")
    else:
        print(f"  No spike-fade signal (VIX peak: {vix_peak:.1f}, current: {vix_last:.1f})")

# ── 2. Sector Drawdown Recovery ──
print("\n--- SECTOR DRAWDOWN RECOVERY ---")
sectors = ['XLK', 'XLF', 'XLV', 'XLE', 'XLY', 'XLC', 'XLI', 'XLB', 'XLRE', 'XLU', 'XLP']
try:
    sector_data = yf.download(sectors, period='90d', progress=False)['Close']
    for sec in sectors:
        if sec not in sector_data.columns:
            continue
        prices = sector_data[sec].dropna()
        if len(prices) < 60:
            continue
        high_60d = float(prices.iloc[-60:].max())
        current = float(prices.iloc[-1])
        drawdown = (current - high_60d) / high_60d * 100

        # Check for recovery signal
        if drawdown < -10:
            # Check if recovering (2 consecutive up days)
            if len(prices) >= 3:
                d1 = float(prices.iloc[-1]) > float(prices.iloc[-2])
                d2 = float(prices.iloc[-2]) > float(prices.iloc[-3])
                recovering = d1 and d2
            else:
                recovering = False

            sma10 = float(prices.iloc[-10:].mean())
            above_sma10 = current > sma10

            status = 'RECOVERING' if (recovering and above_sma10) else 'STILL FALLING'
            events.append({
                'type': 'SECTOR_DRAWDOWN_RECOVERY',
                'ticker': sec,
                'drawdown_pct': round(drawdown, 1),
                'status': status,
                'action': f'BUY {sec} if recovery confirmed' if status == 'RECOVERING' else f'WATCH {sec} ({drawdown:.1f}% from high)',
                'confidence': 'VALIDATED (perm p=0.001)' if status == 'RECOVERING' else 'MONITORING',
                'blocked': kill_switch
            })
            print(f"  {sec}: {drawdown:.1f}% from 60d high — {status}")
except Exception as e:
    print(f"  Error: {e}")

# ── 3. 200-SMA Reclaim ──
print("\n--- 200-SMA RECLAIM ---")
growth_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD', 'NFLX', 'CRM']
try:
    stock_data = yf.download(growth_stocks, period='1y', progress=False)['Close']
    for sym in growth_stocks:
        if sym not in stock_data.columns:
            continue
        prices = stock_data[sym].dropna()
        if len(prices) < 200:
            continue
        current = float(prices.iloc[-1])
        sma200_stock = float(prices.iloc[-200:].mean())
        prev = float(prices.iloc[-2])
        prev_sma200 = float(prices.iloc[-201:-1].mean())

        # Reclaim: was below, now above
        reclaimed = current > sma200_stock and prev < prev_sma200

        if reclaimed:
            events.append({
                'type': '200SMA_RECLAIM',
                'ticker': sym,
                'price': round(current, 2),
                'sma200': round(sma200_stock, 2),
                'action': f'BUY {sym} — just reclaimed 200-SMA',
                'confidence': 'VALIDATED (perm p=0.015)',
                'blocked': kill_switch
            })
            print(f"  ✅ {sym}: reclaimed 200-SMA! Price ${current:.2f} > SMA ${sma200_stock:.2f}")
    else:
        if not any(e['type'] == '200SMA_RECLAIM' for e in events):
            print("  No 200-SMA reclaim events today.")
except Exception as e:
    print(f"  Error: {e}")

# ── 4. MSFT/AAPL Pairs ──
print("\n--- MSFT/AAPL PAIRS ---")
try:
    pair_data = yf.download(['MSFT', 'AAPL'], period='90d', progress=False)['Close']
    msft = pair_data['MSFT'].dropna()
    aapl = pair_data['AAPL'].dropna()
    ratio = msft / aapl
    zscore = (float(ratio.iloc[-1]) - float(ratio.iloc[-60:].mean())) / float(ratio.iloc[-60:].std())

    if abs(zscore) > 2:
        underperformer = 'MSFT' if zscore < -2 else 'AAPL'
        events.append({
            'type': 'PAIRS_REVERSION',
            'pair': 'MSFT/AAPL',
            'zscore': round(zscore, 2),
            'action': f'BUY {underperformer} (z-score: {zscore:.2f})',
            'confidence': 'SUPPLEMENTARY (3/4 adversarial)',
            'blocked': kill_switch
        })
        print(f"  ⚠️ MSFT/AAPL z-score: {zscore:.2f} — BUY {underperformer}")
    else:
        print(f"  No signal (z-score: {zscore:.2f}, need >2.0 or <-2.0)")
except Exception as e:
    print(f"  Error: {e}")

# ── 5. Contrarian Mega-Cap Gap ──
print("\n--- CONTRARIAN MEGA-CAP GAP ---")
mega_caps = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA']
sector_map = {'AAPL': 'XLK', 'MSFT': 'XLK', 'GOOGL': 'XLC', 'AMZN': 'XLY',
              'META': 'XLC', 'NVDA': 'XLK', 'TSLA': 'XLY'}
try:
    mega_data = yf.download(mega_caps, period='5d', progress=False)['Close']
    for sym in mega_caps:
        if sym in mega_data.columns:
            prices_s = mega_data[sym].dropna()
            if len(prices_s) >= 2:
                today_ret = (float(prices_s.iloc[-1]) / float(prices_s.iloc[-2]) - 1) * 100
                if today_ret < -3:
                    sector = sector_map.get(sym, 'SPY')
                    events.append({
                        'type': 'CONTRARIAN_GAP',
                        'trigger': sym,
                        'gap_pct': round(today_ret, 1),
                        'action': f'BUY {sector} for 3-day reversion',
                        'confidence': 'VALIDATED',
                        'blocked': False  # Contrarian works in all regimes
                    })
                    print(f"  ⚠️ {sym} gapped {today_ret:+.1f}% → BUY {sector}")
    if not any(e['type'] == 'CONTRARIAN_GAP' for e in events):
        print("  No mega-cap gaps >3% today.")
except Exception as e:
    print(f"  Error: {e}")

# ── Summary ──
print("\n" + "=" * 60)
print("SIGNAL SUMMARY")
print("=" * 60)

active_events = [e for e in events if not e.get('blocked')]
blocked_events = [e for e in events if e.get('blocked')]

if active_events:
    print(f"\n  {len(active_events)} ACTIONABLE SIGNALS:")
    for e in active_events:
        print(f"    [{e['type']}] {e['action']} ({e['confidence']})")
else:
    print("\n  No actionable signals.")

if blocked_events:
    print(f"\n  {len(blocked_events)} BLOCKED BY KILL SWITCH:")
    for e in blocked_events:
        print(f"    [{e['type']}] {e['action']}")

# Save state
output = {
    'timestamp': datetime.now().isoformat(),
    'regime': {
        'spy': spy_last,
        'sma200': sma200,
        'sma50': sma50,
        'vix': vix_last,
        'bull': bull_regime,
        'kill_switch': kill_switch
    },
    'events': events,
    'active_count': len(active_events),
    'blocked_count': len(blocked_events)
}

with open(os.path.join(STATE_DIR, 'consolidated_events.json'), 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"\n  State saved.")
print("Done.")

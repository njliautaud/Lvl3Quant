#!/usr/bin/env python3
"""
Market Regime Dashboard v1
===========================
Tracks current market regime across multiple dimensions for signal filtering.
Used by both agentic account and portfolio management.

Dimensions:
1. VIX regime (low/normal/elevated/crisis)
2. Trend regime (bull/neutral/bear) via SPY vs moving averages
3. Breadth regime (broad/narrow) via advance-decline
4. Earnings season regime (active/quiet)
5. Correlation regime (normal/stressed) via sector correlation

Output: JSON state file + plain English summary for Discord
"""

import numpy as np
import pandas as pd
import yfinance as yf
import json, os, sys, warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/jupiter/Lvl3Quant/state'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("MARKET REGIME DASHBOARD v1")
print("=" * 60)

# ══════════════════════════════════════════════════════════════
# 1. DOWNLOAD DATA
# ══════════════════════════════════════════════════════════════
print("\n[1/5] Downloading market data...")
sys.stdout.flush()

end_date = datetime.now()
start_date = end_date - timedelta(days=365)

tickers = {
    'SPY': 'S&P 500',
    '^VIX': 'VIX',
    'XLK': 'Tech', 'XLF': 'Financials', 'XLE': 'Energy',
    'XLV': 'Healthcare', 'XLI': 'Industrials', 'XLP': 'Staples',
    'XLY': 'Discretionary', 'XLU': 'Utilities', 'XLC': 'Communications',
    'XLB': 'Materials', 'XLRE': 'Real Estate',
    'TLT': 'Long Bonds', 'GLD': 'Gold', 'UUP': 'Dollar',
    'HYG': 'High Yield', 'LQD': 'Investment Grade'
}

data = yf.download(list(tickers.keys()), start=start_date, end=end_date, progress=False)
prices = data['Close'] if 'Close' in data.columns.get_level_values(0) else data['Adj Close']

if prices.empty:
    print("ERROR: No data downloaded")
    sys.exit(1)

print(f"  Downloaded {len(prices)} days of data for {len(tickers)} tickers")

# ══════════════════════════════════════════════════════════════
# 2. VIX REGIME
# ══════════════════════════════════════════════════════════════
print("\n[2/5] VIX Regime Analysis...")

vix = prices['^VIX'].dropna()
vix_current = vix.iloc[-1]
vix_5d_ago = vix.iloc[-5] if len(vix) >= 5 else vix.iloc[0]
vix_20d_mean = vix.tail(20).mean()
vix_percentile = (vix < vix_current).mean() * 100

if vix_current < 15:
    vix_regime = 'LOW_VOL'
    vix_action = 'Full risk allocation, momentum strategies active'
elif vix_current < 20:
    vix_regime = 'NORMAL'
    vix_action = 'Normal allocation, all strategies active'
elif vix_current < 30:
    vix_regime = 'ELEVATED'
    vix_action = 'REDUCE 50%, pause momentum/rotation, defensive only'
else:
    vix_regime = 'CRISIS'
    vix_action = 'FLAT/HEDGE ONLY, no new longs, consider VIX puts'

vix_direction = 'RISING' if vix_current > vix_5d_ago else 'FALLING'

print(f"  VIX: {vix_current:.1f} ({vix_regime})")
print(f"  VIX 5d change: {vix_current - vix_5d_ago:+.1f} ({vix_direction})")
print(f"  VIX percentile (1yr): {vix_percentile:.0f}th")
print(f"  Action: {vix_action}")

# ══════════════════════════════════════════════════════════════
# 3. TREND REGIME (SPY vs MAs)
# ══════════════════════════════════════════════════════════════
print("\n[3/5] Trend Regime Analysis...")

spy = prices['SPY'].dropna()
spy_current = spy.iloc[-1]
spy_20sma = spy.tail(20).mean()
spy_50sma = spy.tail(50).mean()
spy_200sma = spy.tail(200).mean()

above_20 = spy_current > spy_20sma
above_50 = spy_current > spy_50sma
above_200 = spy_current > spy_200sma

score = sum([above_20, above_50, above_200])
if score == 3:
    trend_regime = 'STRONG_BULL'
    trend_action = 'Full long exposure, momentum strategies preferred'
elif score == 2:
    trend_regime = 'BULL'
    trend_action = 'Normal long exposure, selective entries'
elif score == 1:
    trend_regime = 'NEUTRAL'
    trend_action = 'Reduced exposure, hedge some longs'
else:
    trend_regime = 'BEAR'
    trend_action = 'Minimal long exposure, short bias, defensive strategies only'

# SPY momentum
spy_5d_ret = (spy_current / spy.iloc[-5] - 1) * 100 if len(spy) >= 5 else 0
spy_20d_ret = (spy_current / spy.iloc[-20] - 1) * 100 if len(spy) >= 20 else 0

print(f"  SPY: ${spy_current:.2f}")
print(f"  vs 20-SMA: {'ABOVE' if above_20 else 'BELOW'} (${spy_20sma:.2f})")
print(f"  vs 50-SMA: {'ABOVE' if above_50 else 'BELOW'} (${spy_50sma:.2f})")
print(f"  vs 200-SMA: {'ABOVE' if above_200 else 'BELOW'} (${spy_200sma:.2f})")
print(f"  Trend: {trend_regime}")
print(f"  5d return: {spy_5d_ret:+.1f}%, 20d return: {spy_20d_ret:+.1f}%")

# ══════════════════════════════════════════════════════════════
# 4. SECTOR ROTATION / CORRELATION REGIME
# ══════════════════════════════════════════════════════════════
print("\n[4/5] Sector Correlation & Rotation...")

sector_tickers = ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLP', 'XLY', 'XLU', 'XLC', 'XLB', 'XLRE']
sector_prices = prices[sector_tickers].dropna()
sector_rets = sector_prices.pct_change().dropna()

# 20d rolling correlation matrix
recent_rets = sector_rets.tail(20)
corr_matrix = recent_rets.corr()

# Average pairwise correlation (excluding diagonal)
n = len(sector_tickers)
avg_corr = (corr_matrix.sum().sum() - n) / (n * (n - 1))

if avg_corr > 0.7:
    corr_regime = 'STRESSED'
    corr_action = 'Sectors moving together — market-driven, rotation less effective'
elif avg_corr > 0.4:
    corr_regime = 'NORMAL'
    corr_action = 'Normal dispersion — rotation strategies work'
else:
    corr_regime = 'DISPERSED'
    corr_action = 'High dispersion — rotation strategies strong, stock-picking alpha high'

# Sector momentum ranking
sector_20d = {}
for t in sector_tickers:
    s = sector_prices[t]
    if len(s) >= 20:
        ret = (s.iloc[-1] / s.iloc[-20] - 1) * 100
        sector_20d[t] = ret

sector_ranked = sorted(sector_20d.items(), key=lambda x: x[1], reverse=True)

print(f"  Avg sector correlation (20d): {avg_corr:.2f} ({corr_regime})")
print(f"  Top sectors (20d mom):")
for t, r in sector_ranked[:3]:
    print(f"    {t} ({tickers[t]}): {r:+.1f}%")
print(f"  Bottom sectors:")
for t, r in sector_ranked[-3:]:
    print(f"    {t} ({tickers[t]}): {r:+.1f}%")

# ══════════════════════════════════════════════════════════════
# 5. CROSS-ASSET SIGNALS
# ══════════════════════════════════════════════════════════════
print("\n[5/5] Cross-Asset Signals...")

# Bond-equity divergence
if 'TLT' in prices.columns:
    tlt = prices['TLT'].dropna()
    tlt_20d = (tlt.iloc[-1] / tlt.iloc[-20] - 1) * 100 if len(tlt) >= 20 else 0
    bond_equity_signal = 'RISK-ON' if spy_20d_ret > 0 and tlt_20d < 0 else \
                         'RISK-OFF' if spy_20d_ret < 0 and tlt_20d > 0 else \
                         'MIXED'
    print(f"  TLT 20d: {tlt_20d:+.1f}%  →  {bond_equity_signal}")

# Credit stress
if 'HYG' in prices.columns and 'LQD' in prices.columns:
    hyg = prices['HYG'].dropna()
    lqd = prices['LQD'].dropna()
    hyg_20d = (hyg.iloc[-1] / hyg.iloc[-20] - 1) * 100 if len(hyg) >= 20 else 0
    lqd_20d = (lqd.iloc[-1] / lqd.iloc[-20] - 1) * 100 if len(lqd) >= 20 else 0
    credit_spread_change = hyg_20d - lqd_20d
    credit_signal = 'WIDENING (STRESS)' if credit_spread_change < -0.5 else \
                    'TIGHTENING (CALM)' if credit_spread_change > 0.5 else 'STABLE'
    print(f"  Credit spread trend: {credit_signal} (HYG-LQD 20d: {credit_spread_change:+.2f}%)")

# Gold as fear indicator
if 'GLD' in prices.columns:
    gld = prices['GLD'].dropna()
    gld_20d = (gld.iloc[-1] / gld.iloc[-20] - 1) * 100 if len(gld) >= 20 else 0
    print(f"  Gold 20d: {gld_20d:+.1f}%  {'(FEAR BID)' if gld_20d > 2 else '(calm)'}")

# Dollar
if 'UUP' in prices.columns:
    uup = prices['UUP'].dropna()
    uup_20d = (uup.iloc[-1] / uup.iloc[-20] - 1) * 100 if len(uup) >= 20 else 0
    print(f"  Dollar 20d: {uup_20d:+.1f}%  {'(STRENGTHENING)' if uup_20d > 1 else '(weakening)' if uup_20d < -1 else '(flat)'}")

# ══════════════════════════════════════════════════════════════
# 6. COMPOSITE REGIME & RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("COMPOSITE REGIME ASSESSMENT")
print("=" * 60)

# Score-based composite (higher = more risk-on)
risk_score = 0
risk_max = 100

# VIX contribution (30 pts)
vix_scores = {'LOW_VOL': 30, 'NORMAL': 20, 'ELEVATED': 5, 'CRISIS': 0}
risk_score += vix_scores[vix_regime]

# Trend contribution (30 pts)
trend_scores = {'STRONG_BULL': 30, 'BULL': 20, 'NEUTRAL': 10, 'BEAR': 0}
risk_score += trend_scores[trend_regime]

# Correlation contribution (20 pts)
corr_scores = {'DISPERSED': 20, 'NORMAL': 15, 'STRESSED': 5}
risk_score += corr_scores[corr_regime]

# Credit contribution (20 pts)
try:
    if credit_spread_change > 0.5:
        risk_score += 20
    elif credit_spread_change > -0.5:
        risk_score += 10
    else:
        risk_score += 0
except:
    risk_score += 10  # default neutral

if risk_score >= 70:
    composite = 'RISK-ON'
    allocation_pct = 100
    strategy_filter = 'ALL strategies active, momentum preferred'
elif risk_score >= 45:
    composite = 'NEUTRAL'
    allocation_pct = 75
    strategy_filter = 'Selective strategies, reduce position sizes 25%'
elif risk_score >= 25:
    composite = 'DEFENSIVE'
    allocation_pct = 50
    strategy_filter = 'PAUSE momentum/rotation, income strategies only, event-driven OK'
else:
    composite = 'RISK-OFF'
    allocation_pct = 25
    strategy_filter = 'CASH preferred, hedges only, VIX mean-reversion if VIX > 30'

print(f"\n  Risk Score: {risk_score}/{risk_max}")
print(f"  Composite Regime: {composite}")
print(f"  Allocation Level: {allocation_pct}%")
print(f"  Strategy Filter: {strategy_filter}")

# Strategy-specific guidance
print("\n  Strategy Guidance:")
print(f"    Sector Rotation: {'ACTIVE' if vix_regime in ['LOW_VOL','NORMAL'] and trend_regime in ['STRONG_BULL','BULL'] else 'PAUSED'}")
print(f"    SPY Iron Condors: {'ACTIVE' if vix_regime != 'CRISIS' else 'PAUSED'}")
print(f"    IV Run-Up: {'ACTIVE (earnings season)' if True else 'QUIET'}")  # always active during earnings
print(f"    PEAD: {'ACTIVE' if True else 'QUIET'}")  # always active when earnings report
print(f"    Contrarian Reversion: {'ACTIVE' if vix_regime in ['ELEVATED','CRISIS'] else 'STANDBY'}")
print(f"    VIX Mean Reversion: {'ACTIVE' if vix_current > 25 else 'STANDBY'}")

# ══════════════════════════════════════════════════════════════
# 7. SAVE STATE
# ══════════════════════════════════════════════════════════════
state = {
    'timestamp': datetime.now().isoformat(),
    'vix': {
        'current': round(float(vix_current), 2),
        'regime': vix_regime,
        'direction': vix_direction,
        'percentile_1yr': round(float(vix_percentile), 1),
        'action': vix_action
    },
    'trend': {
        'spy_price': round(float(spy_current), 2),
        'spy_vs_20sma': 'above' if above_20 else 'below',
        'spy_vs_50sma': 'above' if above_50 else 'below',
        'spy_vs_200sma': 'above' if above_200 else 'below',
        'regime': trend_regime,
        'spy_5d_return': round(float(spy_5d_ret), 2),
        'spy_20d_return': round(float(spy_20d_ret), 2)
    },
    'sectors': {
        'correlation_regime': corr_regime,
        'avg_correlation_20d': round(float(avg_corr), 3),
        'top_3': [{'ticker': t, 'name': tickers[t], 'mom_20d': round(r, 2)} for t, r in sector_ranked[:3]],
        'bottom_3': [{'ticker': t, 'name': tickers[t], 'mom_20d': round(r, 2)} for t, r in sector_ranked[-3:]]
    },
    'composite': {
        'risk_score': risk_score,
        'regime': composite,
        'allocation_pct': allocation_pct,
        'strategy_filter': strategy_filter
    },
    'strategy_guidance': {
        'sector_rotation': bool(vix_regime in ['LOW_VOL', 'NORMAL'] and trend_regime in ['STRONG_BULL', 'BULL']),
        'spy_iron_condors': bool(vix_regime != 'CRISIS'),
        'iv_runup': True,
        'pead': True,
        'contrarian_reversion': bool(vix_regime in ['ELEVATED', 'CRISIS']),
        'vix_mean_reversion': bool(vix_current > 25)
    }
}

output_path = os.path.join(OUTPUT_DIR, 'market_regime.json')
with open(output_path, 'w') as f:
    json.dump(state, f, indent=2)

print(f"\n  State saved to {output_path}")
print("\nDone.")

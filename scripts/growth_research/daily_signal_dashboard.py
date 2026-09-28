#!/usr/bin/env python3
"""
Daily Signal Dashboard — GAMEPLAN v3 (Confluence-Gated Vol System)
==================================================================
Generates actionable signals based on validated research (entries 424-428):

1. Vol regime: UPRO / SPY / GLD based on 21-day vol (threshold 15%/30%)
2. 20/200 MA protection: SPY below 20/200 crossover → SPY
3. September hedge: Sep → SPY regardless
4. **NEW v3**: Confluence confirmation gate (entry 428):
   - 3-timeframe score (short/medium/long) → 0 to 3
   - Enter UPRO only when vol<15 AND score ≥ 2.5
   - Exit UPRO when score drops below 2.0
   - Prevents entering UPRO during brief low-vol windows before downturns
5. Risk score: 6-factor composite warning
6. Asset performance

Validated: Sharpe 2.388 vs v2 baseline 1.812 (+0.576). 13/13 years beats v2.
Permutation p=0.000. WF 5.95 mean OOS Sharpe. MaxDD -25.2% vs -31.3%.

Run daily before market open.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import json
import os
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/growth_research/daily_signals'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_data():
    """Get recent market data."""
    tickers = ['SPY', 'UPRO', 'TQQQ', 'GLD', 'TLT', 'SLV', 'VIXY',
               'HYG', 'LQD', 'IWM', 'UUP', 'SH', 'EEM', 'XLU']

    data = yf.download(tickers, period='2y', auto_adjust=True,
                       threads=True, progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        closes = data['Close']
    else:
        closes = data

    if hasattr(closes.columns, 'droplevel'):
        try:
            closes.columns = closes.columns.droplevel(1)
        except:
            pass

    return closes.dropna(how='all')

def compute_confluence_score(closes):
    """
    Compute 3-timeframe confluence score (0-3).
    SHORT:  5d momentum > 0 (+0.5), 10d RSI > 50 (+0.5)
    MEDIUM: 20-day > 50-day MA (+0.5), 21d vol < 15% (+0.5)
    LONG:   200d MA slope > 0 (+0.5), 63d vol trend < 0 (+0.5)
    """
    spy = closes['SPY']
    spy_ret = spy.pct_change()

    score = 0.0
    components = {}

    # SHORT: 5d momentum
    mom_5d = spy.pct_change(5).iloc[-1]
    components['mom_5d'] = float(mom_5d * 100) if not np.isnan(mom_5d) else 0
    if not np.isnan(mom_5d) and mom_5d > 0:
        score += 0.5
        components['mom_5d_pass'] = True
    else:
        components['mom_5d_pass'] = False

    # SHORT: 10d RSI
    delta = spy_ret.copy()
    gain = delta.where(delta > 0, 0).rolling(10).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(10).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi_10 = (100 - (100 / (1 + rs))).iloc[-1]
    components['rsi_10'] = float(rsi_10) if not np.isnan(rsi_10) else 50
    if not np.isnan(rsi_10) and rsi_10 > 50:
        score += 0.5
        components['rsi_10_pass'] = True
    else:
        components['rsi_10_pass'] = False

    # MEDIUM: 20/50 MA crossover
    sma_20 = spy.rolling(20).mean().iloc[-1]
    sma_50 = spy.rolling(50).mean().iloc[-1]
    components['sma_20'] = float(sma_20)
    components['sma_50'] = float(sma_50)
    if not np.isnan(sma_20) and not np.isnan(sma_50) and sma_20 > sma_50:
        score += 0.5
        components['ma_cross_pass'] = True
    else:
        components['ma_cross_pass'] = False

    # MEDIUM: 21d vol < 15%
    vol_21d = spy_ret.rolling(21).std().iloc[-1] * np.sqrt(252) * 100
    components['vol_21d_for_score'] = float(vol_21d) if not np.isnan(vol_21d) else 15
    if not np.isnan(vol_21d) and vol_21d < 15:
        score += 0.5
        components['vol_low_pass'] = True
    else:
        components['vol_low_pass'] = False

    # LONG: 200d MA slope (20-day change)
    sma_200 = spy.rolling(200).mean()
    sma_200_slope = sma_200.pct_change(20).iloc[-1]
    components['sma_200_slope_pct'] = float(sma_200_slope * 100) if not np.isnan(sma_200_slope) else 0
    if not np.isnan(sma_200_slope) and sma_200_slope > 0:
        score += 0.5
        components['slope_pass'] = True
    else:
        components['slope_pass'] = False

    # LONG: 63d vol trend < 0 (declining vol = bullish)
    vol_63d = spy_ret.rolling(63).std() * np.sqrt(252) * 100
    vol_63d_trend = (vol_63d - vol_63d.rolling(21).mean()).iloc[-1]
    components['vol_63d_trend'] = float(vol_63d_trend) if not np.isnan(vol_63d_trend) else 0
    if not np.isnan(vol_63d_trend) and vol_63d_trend < 0:
        score += 0.5
        components['vol_trend_pass'] = True
    else:
        components['vol_trend_pass'] = False

    return score, components


# Persistent state for dual gate hysteresis
_GATE_STATE_FILE = '/home/jupiter/Lvl3Quant/output/growth_research/daily_signals/gate_state.json'

def load_gate_state():
    try:
        with open(_GATE_STATE_FILE) as f:
            return json.load(f)
    except:
        return {'in_upro': False, 'last_date': None}

def save_gate_state(state):
    os.makedirs(os.path.dirname(_GATE_STATE_FILE), exist_ok=True)
    with open(_GATE_STATE_FILE, 'w') as f:
        json.dump(state, f)


def compute_signals(closes):
    """Compute all signals for today — Gameplan v3."""
    spy = closes['SPY']
    spy_ret = spy.pct_change()
    latest = closes.index[-1]

    signals = {
        'date': latest.strftime('%Y-%m-%d'),
        'spy_price': float(spy.iloc[-1]),
        'gameplan_version': 'v3 (confluence-gated)',
    }

    # === 1. VOL REGIME ===
    vol_21d = spy_ret.rolling(21).std().iloc[-1] * np.sqrt(252)
    vol_5d = spy_ret.rolling(5).std().iloc[-1] * np.sqrt(252)
    vol_63d = spy_ret.rolling(63).std().iloc[-1] * np.sqrt(252)

    signals['vol_5d'] = float(vol_5d * 100)
    signals['vol_21d'] = float(vol_21d * 100)
    signals['vol_63d'] = float(vol_63d * 100)

    if vol_21d < 0.15:  # v3 uses 15% threshold (validated in entry 424)
        signals['vol_regime'] = 'LOW_VOL'
        signals['vol_says'] = 'UPRO'
    elif vol_21d < 0.30:
        signals['vol_regime'] = 'MEDIUM_VOL'
        signals['vol_says'] = 'SPY'
    else:
        signals['vol_regime'] = 'HIGH_VOL'
        signals['vol_says'] = 'GLD'

    signals['vol_distance_to_15'] = float((0.15 - vol_21d) * 100)
    signals['vol_distance_to_30'] = float((0.30 - vol_21d) * 100)

    # === 2. PROTECTION OVERLAY (20/200 MA crossover — entry 410) ===
    sma20 = spy.rolling(20).mean().iloc[-1]
    sma200 = spy.rolling(200).mean().iloc[-1]
    sma50 = spy.rolling(50).mean().iloc[-1]

    signals['spy_sma20'] = float(sma20)
    signals['spy_sma50'] = float(sma50)
    signals['spy_sma200'] = float(sma200)
    signals['spy_above_sma50'] = bool(spy.iloc[-1] > sma50)
    signals['spy_above_sma200'] = bool(spy.iloc[-1] > sma200)
    signals['sma20_above_sma200'] = bool(sma20 > sma200)
    signals['protection_on'] = signals['sma20_above_sma200']

    # === 3. CONFLUENCE GATE (v3 addition — entry 428) ===
    conf_score, conf_components = compute_confluence_score(closes)
    signals['confluence_score'] = conf_score
    signals['confluence_components'] = conf_components

    ENTRY_GATE = 2.5
    EXIT_GATE = 2.0

    gate_state = load_gate_state()
    was_in_upro = gate_state.get('in_upro', False)

    if was_in_upro:
        confluence_allows_upro = conf_score >= EXIT_GATE
    else:
        confluence_allows_upro = conf_score >= ENTRY_GATE

    signals['confluence_allows_upro'] = confluence_allows_upro
    signals['confluence_entry_gate'] = ENTRY_GATE
    signals['confluence_exit_gate'] = EXIT_GATE
    signals['was_in_upro'] = was_in_upro

    # === FINAL ALLOCATION DECISION ===
    # September hedge
    is_september = latest.month == 9
    signals['september_hedge'] = is_september

    if is_september:
        signals['final_allocation'] = 'SPY'
        signals['allocation_reason'] = 'September hedge active (worst month for UPRO)'
        new_in_upro = False
    elif signals['vol_says'] == 'GLD':
        signals['final_allocation'] = 'GLD'
        signals['allocation_reason'] = f'High vol ({signals["vol_21d"]:.1f}% > 30%) → safe haven'
        new_in_upro = False
    elif signals['vol_says'] == 'SPY' or not signals['protection_on']:
        signals['final_allocation'] = 'SPY'
        if not signals['protection_on']:
            signals['allocation_reason'] = f'Protection OFF (20d MA below 200d MA)'
        else:
            signals['allocation_reason'] = f'Medium vol ({signals["vol_21d"]:.1f}% > 15%)'
        new_in_upro = False
    elif not confluence_allows_upro:
        signals['final_allocation'] = 'SPY'
        if was_in_upro:
            signals['allocation_reason'] = f'Confluence EXIT triggered (score {conf_score:.1f} < {EXIT_GATE})'
        else:
            signals['allocation_reason'] = f'Confluence gate blocked entry (score {conf_score:.1f} < {ENTRY_GATE})'
        new_in_upro = False
    else:
        signals['final_allocation'] = 'UPRO'
        signals['allocation_reason'] = f'All clear: vol {signals["vol_21d"]:.1f}% < 15%, confluence {conf_score:.1f} ≥ {"exit " + str(EXIT_GATE) if was_in_upro else "entry " + str(ENTRY_GATE)}, protection ON'
        new_in_upro = True

    # Legacy compat
    signals['vol_allocation'] = signals['final_allocation']

    # Save gate state
    save_gate_state({'in_upro': new_in_upro, 'last_date': signals['date']})
    signals['now_in_upro'] = new_in_upro

    # v2 comparison (what v2 would say)
    v2_allocation = 'UPRO' if vol_21d < 0.15 and sma20 > sma200 and not is_september else ('GLD' if vol_21d > 0.30 else 'SPY')
    signals['v2_would_say'] = v2_allocation
    signals['v3_differs_from_v2'] = signals['final_allocation'] != v2_allocation

    if not signals['protection_on']:
        signals['vol_allocation'] = 'SPY (protection overlay triggered — 20d MA below 200d MA)'

    # === 3. RISK SCORE ===
    risk_signals = []

    # Vol ratio (short/long)
    vol_ratio = vol_5d / vol_21d if vol_21d > 0 else 1.0
    risk_signals.append(('Vol ratio (5d/21d)', float(vol_ratio), vol_ratio > 1.2, 'Rising short-term vol'))

    # VIXY momentum
    if 'VIXY' in closes.columns:
        vixy = closes['VIXY']
        vixy_sma5 = vixy.rolling(5).mean().iloc[-1]
        vixy_rising = vixy.iloc[-1] > vixy_sma5
        signals['vixy_price'] = float(vixy.iloc[-1])
        signals['vixy_sma5'] = float(vixy_sma5)
        risk_signals.append(('VIXY > 5d SMA', float(vixy.iloc[-1] / vixy_sma5), vixy_rising, 'Fear rising'))

    # Credit spread
    if 'HYG' in closes.columns and 'LQD' in closes.columns:
        credit = closes['LQD'] / closes['HYG']
        credit_chg = credit.pct_change(5).iloc[-1]
        risk_signals.append(('Credit spread widening', float(credit_chg * 100), credit_chg > 0.005, 'Credit stress'))

    # SPY drawdown from peak
    spy_peak = spy.rolling(252).max().iloc[-1]
    spy_dd = (spy.iloc[-1] - spy_peak) / spy_peak
    signals['spy_dd_from_peak'] = float(spy_dd * 100)
    risk_signals.append(('SPY drawdown', float(spy_dd * 100), spy_dd < -0.05, f'SPY {spy_dd*100:.1f}% from 1yr high'))

    # Breadth (IWM vs SPY)
    if 'IWM' in closes.columns:
        breadth = closes['IWM'].pct_change(21).iloc[-1] - spy.pct_change(21).iloc[-1]
        risk_signals.append(('Breadth (IWM-SPY 21d)', float(breadth * 100), breadth < -0.03, 'Small caps lagging'))

    # Safe haven flows (gold outperforming)
    if 'GLD' in closes.columns:
        gold_vs_spy = closes['GLD'].pct_change(10).iloc[-1] - spy.pct_change(10).iloc[-1]
        risk_signals.append(('Gold vs SPY (10d)', float(gold_vs_spy * 100), gold_vs_spy > 0.02, 'Flight to safety'))

    # Count risk warnings
    n_warnings = sum(1 for _, _, triggered, _ in risk_signals if triggered)
    signals['risk_warnings'] = n_warnings
    signals['risk_level'] = 'LOW' if n_warnings <= 1 else ('MEDIUM' if n_warnings <= 3 else 'HIGH')
    signals['risk_details'] = [(name, val, triggered, desc) for name, val, triggered, desc in risk_signals]

    # === 4. ASSET PERFORMANCE ===
    perf = {}
    for ticker in ['SPY', 'UPRO', 'TQQQ', 'GLD', 'TLT', 'IWM']:
        if ticker in closes.columns:
            t = closes[ticker]
            perf[ticker] = {
                '1d': float(t.pct_change().iloc[-1] * 100),
                '5d': float(t.pct_change(5).iloc[-1] * 100),
                '21d': float(t.pct_change(21).iloc[-1] * 100),
                'ytd': float((t.iloc[-1] / t[t.index >= f'{latest.year}-01-01'].iloc[0] - 1) * 100) if len(t[t.index >= f'{latest.year}-01-01']) > 0 else 0,
            }
    signals['performance'] = perf

    # === 5. UPRO-SPECIFIC METRICS ===
    if 'UPRO' in closes.columns:
        upro = closes['UPRO']
        upro_peak = upro.rolling(252).max().iloc[-1]
        signals['upro_dd_from_peak'] = float((upro.iloc[-1] - upro_peak) / upro_peak * 100)
        signals['upro_price'] = float(upro.iloc[-1])

    return signals

def format_report(signals):
    """Format signals into a readable report."""
    lines = []
    lines.append("=" * 60)
    lines.append(f"GAMEPLAN v3 DAILY SIGNAL — {signals['date']}")
    lines.append("=" * 60)

    # Final allocation
    alloc = signals.get('final_allocation', signals['vol_allocation'])
    reason = signals.get('allocation_reason', '')
    lines.append(f"\n📊 TODAY'S ALLOCATION: {alloc}")
    lines.append(f"   Reason: {reason}")

    # v2 comparison
    if signals.get('v3_differs_from_v2'):
        lines.append(f"   ⚡ v2 would say: {signals['v2_would_say']} (v3 overrides)")
    else:
        lines.append(f"   v2 agrees: {signals.get('v2_would_say', 'N/A')}")

    # Vol details
    lines.append(f"\n📉 VOLATILITY:")
    lines.append(f"   Vol regime: {signals['vol_regime']} (21d vol = {signals['vol_21d']:.1f}%)")
    if signals['vol_regime'] == 'LOW_VOL':
        lines.append(f"   Room to 15% switch: {signals.get('vol_distance_to_15', signals.get('vol_distance_to_20', 0)):.1f}pp")
    elif signals['vol_regime'] == 'MEDIUM_VOL':
        d15 = signals.get('vol_distance_to_15', signals.get('vol_distance_to_20', 0))
        lines.append(f"   Distance to UPRO: {-d15:.1f}pp above 15%")
        lines.append(f"   Distance to safe haven: {signals.get('vol_distance_to_30', 0):.1f}pp below 30%")

    # Confluence gate
    lines.append(f"\n🎯 CONFLUENCE GATE (v3):")
    score = signals.get('confluence_score', 0)
    comp = signals.get('confluence_components', {})
    entry_g = signals.get('confluence_entry_gate', 2.5)
    exit_g = signals.get('confluence_exit_gate', 2.0)

    lines.append(f"   Score: {score:.1f} / 3.0 (entry ≥ {entry_g}, exit < {exit_g})")
    lines.append(f"   Gate status: {'OPEN ✓' if signals.get('confluence_allows_upro') else 'CLOSED ✗'}")
    lines.append(f"   State: {'IN UPRO' if signals.get('now_in_upro') else 'NOT in UPRO'} (was {'in' if signals.get('was_in_upro') else 'out'})")

    # Score breakdown
    checks = [
        ('5d Momentum > 0', comp.get('mom_5d_pass', False), f"{comp.get('mom_5d', 0):.2f}%"),
        ('10d RSI > 50', comp.get('rsi_10_pass', False), f"{comp.get('rsi_10', 50):.1f}"),
        ('20d > 50d MA', comp.get('ma_cross_pass', False), f"20d={comp.get('sma_20', 0):.0f} vs 50d={comp.get('sma_50', 0):.0f}"),
        ('21d Vol < 15%', comp.get('vol_low_pass', False), f"{comp.get('vol_21d_for_score', 15):.1f}%"),
        ('200d MA slope > 0', comp.get('slope_pass', False), f"{comp.get('sma_200_slope_pct', 0):.3f}%"),
        ('63d Vol trend < 0', comp.get('vol_trend_pass', False), f"{comp.get('vol_63d_trend', 0):.2f}"),
    ]
    for name, passed, val in checks:
        icon = "✅" if passed else "❌"
        lines.append(f"     {icon} {name}: {val}")

    # Protection
    lines.append(f"\n🛡️  PROTECTION:")
    lines.append(f"   20/200 MA: {'ON ✓' if signals['protection_on'] else 'OFF ✗ — STAY IN SPY'}")
    lines.append(f"   SPY ${signals['spy_price']:.2f} | SMA20 ${signals.get('spy_sma20', signals.get('spy_sma50', 0)):.2f} | SMA200 ${signals['spy_sma200']:.2f}")
    if signals.get('september_hedge'):
        lines.append(f"   🍂 SEPTEMBER HEDGE ACTIVE")

    # Risk score
    lines.append(f"\n⚠️  RISK LEVEL: {signals['risk_level']} ({signals['risk_warnings']}/6 warnings)")
    for name, val, triggered, desc in signals['risk_details']:
        icon = "🔴" if triggered else "🟢"
        lines.append(f"   {icon} {name}: {val:.2f} — {desc if triggered else 'OK'}")

    # Performance
    lines.append(f"\n📈 PERFORMANCE:")
    lines.append(f"   {'Asset':<8s} {'1d':>7s} {'5d':>7s} {'21d':>7s} {'YTD':>7s}")
    lines.append(f"   {'-'*37}")
    for ticker, p in signals.get('performance', {}).items():
        lines.append(f"   {ticker:<8s} {p['1d']:>+6.1f}% {p['5d']:>+6.1f}% {p['21d']:>+6.1f}% {p['ytd']:>+6.1f}%")

    # Key levels
    lines.append(f"\n📍 KEY LEVELS:")
    lines.append(f"   SPY: ${signals['spy_price']:.2f} | SMA50: ${signals['spy_sma50']:.2f} | SMA200: ${signals['spy_sma200']:.2f}")
    lines.append(f"   SPY from 1yr high: {signals['spy_dd_from_peak']:+.1f}%")
    if 'upro_price' in signals:
        lines.append(f"   UPRO: ${signals['upro_price']:.2f} | from 1yr high: {signals['upro_dd_from_peak']:+.1f}%")

    lines.append(f"\n{'=' * 60}")

    return '\n'.join(lines)

def main():
    print("Fetching market data...")
    closes = get_data()

    print("Computing signals...")
    signals = compute_signals(closes)

    report = format_report(signals)
    print(report)

    # Save
    output_path = os.path.join(OUTPUT_DIR, f"signals_{signals['date']}.json")
    with open(output_path, 'w') as f:
        json.dump(signals, f, indent=2, default=str)

    print(f"\nSignals saved to {output_path}")

    return signals

if __name__ == '__main__':
    main()

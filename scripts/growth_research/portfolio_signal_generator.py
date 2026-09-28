#!/usr/bin/env python3
"""
Portfolio Signal Generator — Three-Tier Framework
==================================================
Consolidates all 14 validated strategies into daily actionable signals.
Produces allocation recommendations for Growth tier (Tier 2).

Three Tiers:
  Tier 1 — Ultra High Risk (RH Agentic): Options, tick-level, VIX spikes
  Tier 2 — Growth (Main portfolios): Validated ML strategies + drawdown protection
  Tier 3 — Income (Future): Wheel, covered calls, carry — paper testing now

Validated Strategies (10 full 4/4, 4 partial 3/4):
  4/4: Yield Curve, Commodity Trend, Tail Risk Hedging, CTA Trend,
       Currency Carry, Bond Duration, Stat Arb, Gold/Silver, Vol Breakout, Sector Rotation
  3/4: Carry+Momentum, International Rotation, Small Cap Value, Thematic Rotation

Key Learnings Applied:
  - ML timing doesn't beat simple rules for leveraged ETFs (200MA > all ML)
  - Multi-signal drawdown protection is the real alpha
  - Static diversification beats ML allocation between strategies
  - Signal edge at tick level is real but decays fast
  - Regime-agnostic validation is mandatory
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import json
import os
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/growth_research/portfolio_signals'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# STRATEGY DEFINITIONS — All 14 validated strategies
# ============================================================

STRATEGIES = {
    # === 4/4 VALIDATED ===
    'yield_curve': {
        'tier': 2, 'gates': '4/4',
        'etfs': ['SHY', 'IEF', 'TLT'],
        'description': 'ML yield curve timing — rotate duration based on curve shape',
        'sharpe': 2.006, 'cagr': 0.318, 'maxdd': -0.151, 'spy_corr': -0.164,
        'weight_optimal': 0.08,
    },
    'commodity_trend': {
        'tier': 2, 'gates': '4/4',
        'etfs': ['DBC', 'USO', 'UNG', 'GLD', 'SLV', 'CPER', 'WEAT', 'CORN'],
        'description': 'Top-3 commodity momentum selection',
        'sharpe': 2.278, 'cagr': 0.549, 'maxdd': -0.185, 'spy_corr': 0.15,
        'weight_optimal': 0.06,
    },
    'tail_risk': {
        'tier': 2, 'gates': '4/4',
        'etfs': ['TAIL', 'SPY', 'SHY'],
        'description': 'ML tail risk hedging — dynamic hedge allocation',
        'sharpe': 4.12, 'cagr': 0.529, 'maxdd': -0.026, 'spy_corr': 0.30,
        'weight_optimal': 0.216,
    },
    'cta_trend': {
        'tier': 2, 'gates': '4/4',
        'etfs': ['SPY', 'TLT', 'GLD', 'UUP', 'USO', 'DBC', 'EFA', 'EEM'],
        'description': 'CTA-style trend following across 8 asset classes',
        'sharpe': 2.915, 'cagr': 0.198, 'maxdd': -0.046, 'spy_corr': 0.25,
        'weight_optimal': 0.213,
    },
    'currency_carry': {
        'tier': [2, 3], 'gates': '4/4',
        'etfs': ['FXA', 'FXB', 'FXC', 'FXE', 'FXY', 'UUP'],
        'description': 'FX carry trade — long high-yield, short low-yield currencies',
        'sharpe': 1.99, 'cagr': 0.10, 'maxdd': -0.043, 'spy_corr': 0.10,
        'weight_optimal': 0.194,
    },
    'bond_duration': {
        'tier': [2, 3], 'gates': '4/4',
        'etfs': ['SHY', 'IEF', 'TLT', 'BND'],
        'description': 'ML bond duration timing — Sortino 22.9',
        'sharpe': 2.00, 'cagr': 0.237, 'maxdd': -0.009, 'spy_corr': -0.05,
        'weight_optimal': 0.05,
    },
    'stat_arb': {
        'tier': [2, 3], 'gates': '4/4',
        'etfs': ['GLD', 'GDX', 'XLF', 'KRE', 'SPY', 'IWM'],
        'description': 'Market-neutral pairs trading on cointegrated ETFs',
        'sharpe': 0.807, 'cagr': 0.081, 'maxdd': -0.115, 'spy_corr': 0.043,
        'weight_optimal': 0.078,
    },
    'gold_silver': {
        'tier': 2, 'gates': '4/4',
        'etfs': ['GLD', 'SLV'],
        'description': 'ML gold/silver ratio timing',
        'sharpe': 1.278, 'cagr': 0.297, 'maxdd': -0.238, 'spy_corr': 0.12,
        'weight_optimal': 0.04,
    },
    'vol_breakout': {
        'tier': 1, 'gates': '4/4',
        'etfs': ['VIX_options'],
        'description': 'Vol breakout options strategy — VIX spike buying',
        'sharpe': 1.11, 'cagr': None, 'maxdd': -0.102, 'spy_corr': -0.30,
        'weight_optimal': 0.03,
    },
    'sector_rotation': {
        'tier': 2, 'gates': '4/4',
        'etfs': ['XLE', 'XLF', 'XLK', 'XLV', 'XLI', 'XLP', 'XLU', 'XLY', 'XLC', 'XLB', 'XLRE'],
        'description': 'Top-3 sector momentum rotation',
        'sharpe': 2.468, 'cagr': 0.206, 'maxdd': -0.074, 'spy_corr': 0.60,
        'weight_optimal': 0.05,
    },
    # === 3/4 VALIDATED (R1 fail acceptable for growth per HC #709) ===
    'carry_momentum': {
        'tier': 2, 'gates': '3/4',
        'etfs': ['SCHD', 'VYM', 'DGRO', 'QQQ', 'VUG', 'MTUM', 'USMV', 'QUAL'],
        'description': 'ML carry+momentum hybrid — dividend+growth allocation',
        'sharpe': 2.963, 'cagr': 0.445, 'maxdd': -0.070, 'spy_corr': 0.75,
        'weight_optimal': 0.078,
    },
    'intl_rotation': {
        'tier': 2, 'gates': '3/4',
        'etfs': ['EFA', 'EEM', 'VEA', 'VWO', 'IEMG', 'EWJ', 'FXI', 'EWZ'],
        'description': 'ML international country rotation',
        'sharpe': 2.20, 'cagr': 0.442, 'maxdd': -0.27, 'spy_corr': 0.87,
        'weight_optimal': 0.02,
    },
    'small_cap_value': {
        'tier': 2, 'gates': '3/4',
        'etfs': ['IWM', 'IWN', 'VBR', 'SLYV'],
        'description': 'ML small cap value timing',
        'sharpe': 1.75, 'cagr': 0.341, 'maxdd': -0.192, 'spy_corr': 0.80,
        'weight_optimal': 0.02,
    },
    'thematic_rotation': {
        'tier': 2, 'gates': '3/4',
        'etfs': ['ARKK', 'ICLN', 'TAN', 'KWEB', 'HACK', 'BOTZ', 'LIT'],
        'description': 'ML thematic ETF rotation',
        'sharpe': 2.59, 'cagr': 0.597, 'maxdd': -0.121, 'spy_corr': 0.65,
        'weight_optimal': 0.02,
    },
}

# ============================================================
# DRAWDOWN PROTECTION — Multi-signal kill switches (HC #709/710)
# Best composite: VIX<20 + SPY>50SMA + credit not stressed + breadth>50%
# Turned Sharpe 0.59 → 2.91, MaxDD -77% → -8.5%
# ============================================================

def get_drawdown_signals():
    """Pull current drawdown protection signals."""
    try:
        end = datetime.now()
        start = end - timedelta(days=120)

        tickers = {
            'SPY': 'spy', '^VIX': 'vix', 'HYG': 'hyg', 'LQD': 'lqd',
            'RSP': 'rsp',  # equal weight S&P for breadth proxy
        }

        data = yf.download(list(tickers.keys()), start=start, end=end, progress=False)
        if data.empty:
            return {'all_clear': None, 'signals': {}, 'error': 'No data'}

        close = data['Close'] if 'Close' in data.columns else data['Adj Close']

        signals = {}

        # 1. VIX < 20 (calm)
        if '^VIX' in close.columns:
            vix_now = close['^VIX'].dropna().iloc[-1]
            signals['vix_calm'] = bool(vix_now < 20)
            signals['vix_level'] = round(float(vix_now), 1)

        # 2. SPY > 50-day SMA
        if 'SPY' in close.columns:
            spy = close['SPY'].dropna()
            spy_now = spy.iloc[-1]
            spy_sma50 = spy.rolling(50).mean().iloc[-1]
            spy_sma200 = spy.rolling(200).mean().iloc[-1] if len(spy) >= 200 else spy_sma50
            signals['spy_above_50sma'] = bool(spy_now > spy_sma50)
            signals['spy_above_200sma'] = bool(spy_now > spy_sma200)
            signals['spy_price'] = round(float(spy_now), 2)
            signals['spy_50sma'] = round(float(spy_sma50), 2)

        # 3. Credit not stressed (HYG/LQD spread)
        if 'HYG' in close.columns and 'LQD' in close.columns:
            hyg = close['HYG'].dropna()
            lqd = close['LQD'].dropna()
            if len(hyg) > 20 and len(lqd) > 20:
                spread = (hyg / lqd).dropna()
                spread_now = spread.iloc[-1]
                spread_mean = spread.rolling(60).mean().iloc[-1]
                signals['credit_ok'] = bool(spread_now >= spread_mean * 0.98)

        # 4. Breadth > 50% (RSP/SPY ratio as proxy)
        if 'RSP' in close.columns and 'SPY' in close.columns:
            rsp = close['RSP'].dropna()
            spy = close['SPY'].dropna()
            if len(rsp) > 20:
                breadth_ratio = (rsp / spy).dropna()
                br_now = breadth_ratio.iloc[-1]
                br_mean = breadth_ratio.rolling(50).mean().iloc[-1]
                signals['breadth_ok'] = bool(br_now >= br_mean * 0.99)

        # Composite: ALL four must be true for "all clear"
        checks = ['vix_calm', 'spy_above_50sma', 'credit_ok', 'breadth_ok']
        all_available = all(k in signals for k in checks)
        if all_available:
            signals['all_clear'] = all(signals[k] for k in checks)
        else:
            signals['all_clear'] = None

        return signals

    except Exception as e:
        return {'all_clear': None, 'signals': {}, 'error': str(e)}


def get_sector_momentum():
    """Calculate sector momentum rankings for rotation signal."""
    sectors = {
        'XLE': 'Energy', 'XLF': 'Financials', 'XLK': 'Technology',
        'XLV': 'Healthcare', 'XLI': 'Industrials', 'XLP': 'Staples',
        'XLU': 'Utilities', 'XLY': 'Consumer Disc', 'XLC': 'Communication',
        'XLB': 'Materials', 'XLRE': 'Real Estate',
    }

    try:
        end = datetime.now()
        start = end - timedelta(days=200)
        data = yf.download(list(sectors.keys()), start=start, end=end, progress=False)
        if data.empty:
            return []

        close = data['Close'] if 'Close' in data.columns else data['Adj Close']

        results = []
        for ticker, name in sectors.items():
            if ticker not in close.columns:
                continue
            prices = close[ticker].dropna()
            if len(prices) < 130:
                continue

            mom_1m = float(prices.iloc[-1] / prices.iloc[-22] - 1) if len(prices) >= 22 else 0
            mom_3m = float(prices.iloc[-1] / prices.iloc[-66] - 1) if len(prices) >= 66 else 0
            mom_6m = float(prices.iloc[-1] / prices.iloc[-130] - 1) if len(prices) >= 130 else 0

            # Composite momentum score (skip most recent month to avoid reversal)
            composite = 0.4 * mom_6m + 0.4 * mom_3m + 0.2 * mom_1m

            results.append({
                'ticker': ticker, 'name': name,
                'mom_1m': round(mom_1m * 100, 1),
                'mom_3m': round(mom_3m * 100, 1),
                'mom_6m': round(mom_6m * 100, 1),
                'composite': round(composite * 100, 1),
            })

        results.sort(key=lambda x: x['composite'], reverse=True)
        return results

    except Exception as e:
        return [{'error': str(e)}]


def get_commodity_momentum():
    """Calculate commodity trend signals — top 3 selection."""
    commodities = {
        'DBC': 'Broad Commodities', 'USO': 'Crude Oil', 'UNG': 'Natural Gas',
        'GLD': 'Gold', 'SLV': 'Silver', 'CPER': 'Copper',
        'WEAT': 'Wheat', 'CORN': 'Corn',
    }

    try:
        end = datetime.now()
        start = end - timedelta(days=200)
        data = yf.download(list(commodities.keys()), start=start, end=end, progress=False)
        if data.empty:
            return []

        close = data['Close'] if 'Close' in data.columns else data['Adj Close']

        results = []
        for ticker, name in commodities.items():
            if ticker not in close.columns:
                continue
            prices = close[ticker].dropna()
            if len(prices) < 66:
                continue

            mom_1m = float(prices.iloc[-1] / prices.iloc[-22] - 1)
            mom_3m = float(prices.iloc[-1] / prices.iloc[-66] - 1)

            results.append({
                'ticker': ticker, 'name': name,
                'mom_1m': round(mom_1m * 100, 1),
                'mom_3m': round(mom_3m * 100, 1),
                'score': round((0.5 * mom_3m + 0.5 * mom_1m) * 100, 1),
            })

        results.sort(key=lambda x: x['score'], reverse=True)
        return results

    except Exception as e:
        return [{'error': str(e)}]


def get_regime_indicators():
    """Pull cross-asset regime signals."""
    try:
        end = datetime.now()
        start = end - timedelta(days=300)

        tickers = ['^VIX', 'GLD', 'TLT', 'UUP', 'SPY', 'BTC-USD', 'USO', 'HYG']
        data = yf.download(tickers, start=start, end=end, progress=False)
        if data.empty:
            return {}

        close = data['Close'] if 'Close' in data.columns else data['Adj Close']

        regime = {}

        for t in tickers:
            if t not in close.columns:
                continue
            prices = close[t].dropna()
            if len(prices) < 50:
                continue

            name = t.replace('^', '').replace('-', '_')
            current = float(prices.iloc[-1])
            sma50 = float(prices.rolling(50).mean().iloc[-1])
            mom_1m = float(prices.iloc[-1] / prices.iloc[-22] - 1) if len(prices) >= 22 else 0

            regime[name] = {
                'price': round(current, 2),
                'vs_50sma': round((current / sma50 - 1) * 100, 1),
                'mom_1m': round(mom_1m * 100, 1),
                'trend': 'UP' if current > sma50 else 'DOWN',
            }

        # Overall regime classification
        vix_low = regime.get('VIX', {}).get('price', 20) < 20
        spy_up = regime.get('SPY', {}).get('trend') == 'UP'
        tlt_up = regime.get('TLT', {}).get('trend') == 'UP'
        gld_up = regime.get('GLD', {}).get('trend') == 'UP'

        if spy_up and vix_low:
            regime['overall'] = 'RISK_ON'
        elif not spy_up and not vix_low:
            regime['overall'] = 'RISK_OFF'
        elif spy_up and not vix_low:
            regime['overall'] = 'CAUTIOUS_BULL'
        else:
            regime['overall'] = 'TRANSITIONAL'

        return regime

    except Exception as e:
        return {'error': str(e)}


def generate_daily_signals():
    """Master signal generator — produces daily portfolio signals."""
    print("=" * 70)
    print(f"PORTFOLIO SIGNAL GENERATOR — {datetime.now().strftime('%Y-%m-%d %H:%M ET')}")
    print("=" * 70)

    # 1. Drawdown protection
    print("\n[1/4] Drawdown Protection Signals...")
    dd_signals = get_drawdown_signals()
    print(f"  All Clear: {dd_signals.get('all_clear', 'N/A')}")
    for k, v in dd_signals.items():
        if k not in ('all_clear', 'error'):
            print(f"  {k}: {v}")

    # 2. Regime indicators
    print("\n[2/4] Cross-Asset Regime...")
    regime = get_regime_indicators()
    overall = regime.pop('overall', 'UNKNOWN')
    print(f"  Overall Regime: {overall}")
    for name, vals in regime.items():
        if isinstance(vals, dict) and 'price' in vals:
            print(f"  {name}: {vals['price']} ({vals['trend']}, {vals['vs_50sma']:+.1f}% vs 50SMA)")

    # 3. Sector momentum
    print("\n[3/4] Sector Momentum Rankings...")
    sectors = get_sector_momentum()
    for i, s in enumerate(sectors[:5]):
        marker = " ← TOP 3" if i < 3 else ""
        print(f"  #{i+1} {s['ticker']} ({s['name']}): {s['composite']:+.1f}% composite | 1M={s['mom_1m']:+.1f}% 3M={s['mom_3m']:+.1f}% 6M={s['mom_6m']:+.1f}%{marker}")

    # 4. Commodity momentum
    print("\n[4/4] Commodity Trend Rankings...")
    commodities = get_commodity_momentum()
    for i, c in enumerate(commodities[:5]):
        marker = " ← TOP 3" if i < 3 else ""
        print(f"  #{i+1} {c['ticker']} ({c['name']}): {c['score']:+.1f}% | 1M={c['mom_1m']:+.1f}% 3M={c['mom_3m']:+.1f}%{marker}")

    # === TIER RECOMMENDATIONS ===
    print("\n" + "=" * 70)
    print("TIER RECOMMENDATIONS")
    print("=" * 70)

    all_clear = dd_signals.get('all_clear', None)

    # TIER 2 — Growth
    print("\n--- TIER 2: GROWTH (Main Portfolios) ---")
    if all_clear:
        print("  ✅ ALL CLEAR — Full growth exposure")
        print("  • Leveraged ETFs (TQQQ/UPRO): ACTIVE with full allocation")
        if len(sectors) >= 3:
            top3 = [s['ticker'] for s in sectors[:3]]
            print(f"  • Sector Rotation: Top 3 = {', '.join(top3)}")
        if len(commodities) >= 3:
            top3c = [c['ticker'] for c in commodities[:3]]
            print(f"  • Commodity Trend: Top 3 = {', '.join(top3c)}")
    elif all_clear is False:
        failing = [k for k in ['vix_calm', 'spy_above_50sma', 'credit_ok', 'breadth_ok']
                   if k in dd_signals and not dd_signals[k]]
        print(f"  ⚠️  DRAWDOWN PROTECTION ACTIVE — Reduce exposure")
        print(f"  • Failing signals: {', '.join(failing)}")
        print("  • Leveraged ETFs: REDUCE to 50% or switch to unleveraged")
        print("  • Increase defensive allocation (TLT, GLD, SHY)")
    else:
        print("  ❓ Signals unavailable — maintain current allocation")

    # TIER 1 — Ultra High Risk
    print("\n--- TIER 1: ULTRA HIGH RISK (RH Agentic) ---")
    vix_level = dd_signals.get('vix_level', None)
    if vix_level and vix_level > 30:
        print(f"  🎯 VIX SPIKE ({vix_level}) — Deploy VIX puts per HC #714 R4")
    elif vix_level and vix_level > 25:
        print(f"  ⚡ VIX elevated ({vix_level}) — Watch for spike entry")
    else:
        print(f"  • VIX calm ({vix_level}) — Momentum plays only")
        if len(sectors) >= 1:
            print(f"  • Top sector: {sectors[0]['ticker']} ({sectors[0]['name']}) {sectors[0]['composite']:+.1f}%")

    # TIER 3 — Income
    print("\n--- TIER 3: INCOME (Paper Testing) ---")
    print("  • Wheel paper engines: 8 running (BPS, IC, V5, diversified, etc.)")
    print("  • Deploy when main portfolio reaches target size")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'regime': overall,
        'drawdown_protection': dd_signals,
        'sector_rankings': sectors[:5] if sectors else [],
        'commodity_rankings': commodities[:5] if commodities else [],
        'tier2_recommendation': 'FULL_GROWTH' if all_clear else 'REDUCE' if all_clear is False else 'HOLD',
        'vix_signal': 'SPIKE_BUY' if (vix_level and vix_level > 30) else 'WATCH' if (vix_level and vix_level > 25) else 'CALM',
    }

    outfile = os.path.join(OUTPUT_DIR, 'latest_signals.json')
    with open(outfile, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSignals saved to {outfile}")

    return output


if __name__ == '__main__':
    generate_daily_signals()

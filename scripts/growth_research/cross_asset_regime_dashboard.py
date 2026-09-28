#!/usr/bin/env python3
"""
Cross-Asset Regime Dashboard (HC #710 R2)
==========================================
Multi-asset momentum/trend dashboard that updates daily.
Incorporates broad market signals: Gold, Silver, Copper, Oil,
Dollar, Bonds, Credit, Bitcoin — per HC #710 mandate.

Outputs:
  1. Cross-asset momentum/trend table (HC #710 R2)
  2. Regime classification (risk-on / risk-off / transition)
  3. Correlation heatmap with SPY (HC #710 R3)
  4. Leading indicator signals
  5. Gameplan v2 allocation signal (enhanced with cross-asset context)
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import json
import os
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/growth_research/cross_asset_dashboard'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# HC #710 R1 full asset list
ASSETS = {
    # Equities
    'SPY': 'S&P 500',
    'QQQ': 'Nasdaq 100',
    'IWM': 'Russell 2000',
    'EEM': 'Emerging Markets',
    # Leveraged
    'UPRO': '3x S&P 500',
    'TQQQ': '3x Nasdaq',
    # Commodities
    'GLD': 'Gold',
    'SLV': 'Silver',
    'COPX': 'Copper Miners',  # Proxy for copper
    'USO': 'Oil',
    'DBA': 'Agriculture',
    # Fixed Income
    'TLT': '20+ Yr Bonds',
    'IEF': '7-10 Yr Bonds',
    'SHY': '1-3 Yr Bonds',
    # Credit
    'HYG': 'High Yield',
    'LQD': 'Inv Grade',
    # Dollar
    'UUP': 'US Dollar',
    # Crypto
    'IBIT': 'Bitcoin ETF',
    # Vol
    'VIXY': 'VIX Short-Term',
}


def download_data():
    tickers = list(ASSETS.keys())
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


def compute_momentum_trend(closes):
    """Compute momentum and trend for each asset."""
    results = []
    for ticker in ASSETS:
        if ticker not in closes.columns:
            continue
        prices = closes[ticker].dropna()
        if len(prices) < 252:
            continue

        current = prices.iloc[-1]

        # Returns
        ret_1d = prices.pct_change(1).iloc[-1] * 100
        ret_5d = prices.pct_change(5).iloc[-1] * 100
        ret_21d = prices.pct_change(21).iloc[-1] * 100
        ret_63d = prices.pct_change(63).iloc[-1] * 100 if len(prices) > 63 else np.nan
        ret_252d = prices.pct_change(252).iloc[-1] * 100 if len(prices) > 252 else np.nan

        # Trend: SMA20 vs SMA50 vs SMA200
        sma20 = prices.rolling(20).mean().iloc[-1]
        sma50 = prices.rolling(50).mean().iloc[-1]
        sma200 = prices.rolling(200).mean().iloc[-1] if len(prices) >= 200 else np.nan

        # Trend classification
        if not np.isnan(sma200):
            if current > sma20 > sma50 > sma200:
                trend = '↑↑ STRONG UP'
            elif current > sma50 > sma200:
                trend = '↑ UP'
            elif current < sma20 < sma50 < sma200:
                trend = '↓↓ STRONG DN'
            elif current < sma50:
                trend = '↓ DOWN'
            else:
                trend = '→ FLAT'
        else:
            trend = 'N/A'

        # Volatility
        vol_21d = prices.pct_change().rolling(21).std().iloc[-1] * np.sqrt(252) * 100

        # RSI
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = (100 - 100 / (1 + rs)).iloc[-1]

        # Distance from 52-week high
        high_252 = prices.rolling(252).max().iloc[-1]
        dist_high = (current / high_252 - 1) * 100

        results.append({
            'Ticker': ticker,
            'Name': ASSETS[ticker],
            'Price': current,
            '1d%': ret_1d,
            '5d%': ret_5d,
            '21d%': ret_21d,
            '63d%': ret_63d,
            '1yr%': ret_252d,
            'Vol': vol_21d,
            'RSI': rsi,
            'Trend': trend,
            'From High': dist_high,
        })

    return pd.DataFrame(results)


def compute_regime_signals(closes):
    """Compute cross-asset regime indicators."""
    spy = closes.get('SPY')
    if spy is None:
        return {}

    spy_ret = spy.pct_change()
    vol_21d = spy_ret.rolling(21).std().iloc[-1] * np.sqrt(252)

    signals = {}

    # 1. Equity vol regime
    vol_pct = vol_21d * 100
    signals['spy_vol_21d'] = vol_pct
    if vol_pct > 30:
        signals['vol_regime'] = 'CRISIS'
    elif vol_pct > 20:
        signals['vol_regime'] = 'ELEVATED'
    else:
        signals['vol_regime'] = 'LOW'

    # 2. Credit stress (HYG-LQD spread as proxy)
    hyg = closes.get('HYG')
    lqd = closes.get('LQD')
    if hyg is not None and lqd is not None:
        # Relative performance as credit stress proxy
        hyg_ret_21d = hyg.pct_change(21).iloc[-1]
        lqd_ret_21d = lqd.pct_change(21).iloc[-1]
        credit_spread_change = (lqd_ret_21d - hyg_ret_21d) * 100  # Positive = widening (stress)
        signals['credit_stress'] = credit_spread_change
        signals['credit_signal'] = 'STRESS' if credit_spread_change > 2 else 'NORMAL'

    # 3. Gold/SPY ratio momentum (flight to safety indicator)
    gld = closes.get('GLD')
    if gld is not None:
        gld_spy_ratio = gld / spy
        ratio_21d_change = gld_spy_ratio.pct_change(21).iloc[-1] * 100
        signals['gold_spy_21d'] = ratio_21d_change
        signals['safety_flight'] = 'ACTIVE' if ratio_21d_change > 3 else 'NONE'

    # 4. Dollar strength
    uup = closes.get('UUP')
    if uup is not None:
        dollar_21d = uup.pct_change(21).iloc[-1] * 100
        signals['dollar_21d'] = dollar_21d
        signals['dollar_signal'] = 'STRONG' if dollar_21d > 1 else ('WEAK' if dollar_21d < -1 else 'NEUTRAL')

    # 5. Copper/Gold ratio (economic growth proxy)
    copx = closes.get('COPX')
    if copx is not None and gld is not None:
        cg_ratio = copx / gld
        cg_21d = cg_ratio.pct_change(21).iloc[-1] * 100
        signals['copper_gold_21d'] = cg_21d
        signals['growth_signal'] = 'EXPANSION' if cg_21d > 2 else ('CONTRACTION' if cg_21d < -2 else 'NEUTRAL')

    # 6. Bond yield curve proxy (TLT vs SHY)
    tlt = closes.get('TLT')
    shy = closes.get('SHY')
    if tlt is not None and shy is not None:
        tlt_shy_ratio = tlt / shy
        curve_21d = tlt_shy_ratio.pct_change(21).iloc[-1] * 100
        signals['yield_curve_21d'] = curve_21d
        signals['curve_signal'] = 'STEEPENING' if curve_21d > 1 else ('FLATTENING' if curve_21d < -1 else 'STABLE')

    # 7. Small cap vs large cap (risk appetite)
    iwm = closes.get('IWM')
    if iwm is not None:
        iwm_spy = iwm / spy
        breadth_21d = iwm_spy.pct_change(21).iloc[-1] * 100
        signals['breadth_21d'] = breadth_21d
        signals['risk_appetite'] = 'STRONG' if breadth_21d > 2 else ('WEAK' if breadth_21d < -2 else 'NEUTRAL')

    # 8. Emerging markets vs US (global risk-on)
    eem = closes.get('EEM')
    if eem is not None:
        eem_spy = eem / spy
        em_21d = eem_spy.pct_change(21).iloc[-1] * 100
        signals['em_vs_us_21d'] = em_21d

    # 9. Composite regime score
    risk_on_count = 0
    risk_off_count = 0
    total = 0

    checks = [
        ('vol_regime', 'LOW', 'CRISIS'),
        ('credit_signal', 'NORMAL', 'STRESS'),
        ('safety_flight', 'NONE', 'ACTIVE'),
        ('dollar_signal', 'WEAK', 'STRONG'),
        ('growth_signal', 'EXPANSION', 'CONTRACTION'),
        ('risk_appetite', 'STRONG', 'WEAK'),
    ]

    for key, on_val, off_val in checks:
        if key in signals:
            total += 1
            if signals[key] == on_val:
                risk_on_count += 1
            elif signals[key] == off_val:
                risk_off_count += 1

    signals['risk_on_score'] = risk_on_count
    signals['risk_off_score'] = risk_off_count
    signals['total_indicators'] = total

    if risk_off_count >= 3:
        signals['composite_regime'] = 'RISK-OFF'
    elif risk_on_count >= 4:
        signals['composite_regime'] = 'RISK-ON'
    else:
        signals['composite_regime'] = 'MIXED'

    return signals


def compute_correlations(closes):
    """Compute rolling correlations with SPY (HC #710 R3)."""
    spy = closes.get('SPY')
    if spy is None:
        return pd.DataFrame()

    spy_ret = spy.pct_change().dropna()
    corrs = {}

    for ticker in ASSETS:
        if ticker == 'SPY' or ticker not in closes.columns:
            continue
        asset_ret = closes[ticker].pct_change().dropna()
        # Align
        combined = pd.concat([spy_ret, asset_ret], axis=1).dropna()
        if len(combined) < 60:
            continue
        combined.columns = ['SPY', ticker]

        corr_21d = combined['SPY'].rolling(21).corr(combined[ticker]).iloc[-1]
        corr_63d = combined['SPY'].rolling(63).corr(combined[ticker]).iloc[-1]

        corrs[ticker] = {
            'name': ASSETS[ticker],
            'corr_21d': corr_21d,
            'corr_63d': corr_63d,
        }

    return pd.DataFrame(corrs).T


def gameplan_v2_signal(closes, regime_signals):
    """Enhanced Gameplan v2 signal with cross-asset context."""
    spy = closes['SPY']
    spy_ret = spy.pct_change()
    vol_21d = spy_ret.rolling(21).std().iloc[-1] * np.sqrt(252)
    sma20 = spy.rolling(20).mean().iloc[-1]
    sma200 = spy.rolling(200).mean().iloc[-1]
    vol_pct = vol_21d * 100

    today = closes.index[-1]
    m, d = today.month, today.day

    # September hedge
    if today.month == 9:
        base_signal = 'SPY'
        reason = 'September hedge'
    else:
        is_earnings = ((m == 1 and d >= 15) or (m == 2 and d <= 15) or
                      (m == 4 and d >= 15) or (m == 5 and d <= 15) or
                      (m == 7 and d >= 15) or (m == 8 and d <= 15) or
                      (m == 10 and d >= 15) or (m == 11 and d <= 15))
        low_t = 25 if is_earnings else 20

        protection_off = sma20 < sma200

        if vol_pct > 30:
            base_signal = 'GLD'
            reason = f'Crisis vol ({vol_pct:.1f}%)'
        elif vol_pct > low_t or protection_off:
            base_signal = 'SPY'
            reason = f'Elevated vol ({vol_pct:.1f}%)' if vol_pct > low_t else f'Bearish trend (SMA20 < SMA200)'
        else:
            base_signal = 'UPRO'
            reason = f'Low vol ({vol_pct:.1f}%)'
            if is_earnings:
                reason += ', earnings aggressive (25% thresh)'

    # Cross-asset context (informational, doesn't change signal)
    composite = regime_signals.get('composite_regime', 'UNKNOWN')
    context_warnings = []
    if composite == 'RISK-OFF' and base_signal == 'UPRO':
        context_warnings.append('⚠️ UPRO signal but cross-asset regime is RISK-OFF — watch closely')
    if regime_signals.get('credit_signal') == 'STRESS':
        context_warnings.append('⚠️ Credit stress detected — HYG underperforming')
    if regime_signals.get('safety_flight') == 'ACTIVE':
        context_warnings.append('⚠️ Gold outperforming SPY — flight to safety active')

    return {
        'signal': base_signal,
        'reason': reason,
        'vol_21d': vol_pct,
        'sma20': float(sma20),
        'sma200': float(sma200),
        'composite_regime': composite,
        'warnings': context_warnings,
        'distance_to_switch': 20 - vol_pct if base_signal == 'UPRO' else 0,
    }


def main():
    print("=" * 70)
    print("CROSS-ASSET REGIME DASHBOARD (HC #710)")
    today_str = datetime.now().strftime('%Y-%m-%d %H:%M')
    print(f"Generated: {today_str}")
    print("=" * 70)

    print("\nDownloading data...")
    closes = download_data()
    latest_date = closes.index[-1].strftime('%Y-%m-%d')
    print(f"Latest data: {latest_date}")

    # 1. Momentum/Trend Table
    print(f"\n{'='*70}")
    print("  CROSS-ASSET MOMENTUM & TREND")
    print(f"{'='*70}")
    mt = compute_momentum_trend(closes)
    print(f"\n  {'Ticker':<6} {'Name':<18} {'Price':>8} {'1d':>6} {'5d':>6} {'21d':>6} {'3m':>6} {'Vol':>5} {'RSI':>4} {'Trend':<14} {'High':>6}")
    print(f"  {'-'*6} {'-'*18} {'-'*8} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*5} {'-'*4} {'-'*14} {'-'*6}")
    for _, r in mt.iterrows():
        print(f"  {r['Ticker']:<6} {r['Name']:<18} {r['Price']:>8.2f} "
              f"{r['1d%']:>+5.1f}% {r['5d%']:>+5.1f}% {r['21d%']:>+5.1f}% "
              f"{r['63d%']:>+5.1f}% {r['Vol']:>4.0f}% {r['RSI']:>3.0f} "
              f"{r['Trend']:<14} {r['From High']:>+5.1f}%")

    # 2. Regime Signals
    print(f"\n{'='*70}")
    print("  REGIME SIGNALS")
    print(f"{'='*70}")
    regime = compute_regime_signals(closes)

    signal_pairs = [
        ('Vol Regime', 'vol_regime', f"(21d vol: {regime.get('spy_vol_21d', 0):.1f}%)"),
        ('Credit', 'credit_signal', f"(HYG-LQD spread Δ: {regime.get('credit_stress', 0):+.1f}%)"),
        ('Flight to Safety', 'safety_flight', f"(GLD/SPY 21d: {regime.get('gold_spy_21d', 0):+.1f}%)"),
        ('Dollar', 'dollar_signal', f"(UUP 21d: {regime.get('dollar_21d', 0):+.1f}%)"),
        ('Growth (Cu/Au)', 'growth_signal', f"(COPX/GLD 21d: {regime.get('copper_gold_21d', 0):+.1f}%)"),
        ('Yield Curve', 'curve_signal', f"(TLT/SHY 21d: {regime.get('yield_curve_21d', 0):+.1f}%)"),
        ('Risk Appetite', 'risk_appetite', f"(IWM/SPY 21d: {regime.get('breadth_21d', 0):+.1f}%)"),
    ]

    for label, key, detail in signal_pairs:
        val = regime.get(key, 'N/A')
        icon = '🟢' if val in ('LOW', 'NORMAL', 'NONE', 'EXPANSION', 'STRONG', 'WEAK') else (
            '🔴' if val in ('CRISIS', 'STRESS', 'ACTIVE', 'CONTRACTION') else '🟡')
        # Flip dollar: WEAK dollar is risk-on for equities
        if key == 'dollar_signal':
            icon = '🟢' if val == 'WEAK' else ('🔴' if val == 'STRONG' else '🟡')
        print(f"  {icon} {label:<20} {val:<14} {detail}")

    composite = regime.get('composite_regime', 'UNKNOWN')
    risk_on = regime.get('risk_on_score', 0)
    risk_off = regime.get('risk_off_score', 0)
    total = regime.get('total_indicators', 0)
    print(f"\n  COMPOSITE: {composite} (risk-on: {risk_on}/{total}, risk-off: {risk_off}/{total})")

    # 3. Correlations
    print(f"\n{'='*70}")
    print("  SPY CORRELATIONS (HC #710 R3)")
    print(f"{'='*70}")
    corrs = compute_correlations(closes)
    if len(corrs) > 0:
        corrs_sorted = corrs.sort_values('corr_21d', ascending=False)
        print(f"\n  {'Asset':<18} {'21d Corr':>10} {'63d Corr':>10} {'Diversification':>15}")
        print(f"  {'-'*18} {'-'*10} {'-'*10} {'-'*15}")
        for idx, r in corrs_sorted.iterrows():
            c21 = r['corr_21d']
            div = 'LOW' if abs(c21) > 0.6 else ('MEDIUM' if abs(c21) > 0.3 else 'HIGH')
            print(f"  {r['name']:<18} {c21:>+10.3f} {r['corr_63d']:>+10.3f} {div:>15}")

    # 4. Gameplan v2 Signal
    print(f"\n{'='*70}")
    print("  GAMEPLAN v2 ALLOCATION SIGNAL")
    print(f"{'='*70}")
    gp = gameplan_v2_signal(closes, regime)

    signal_icon = {'UPRO': '🚀', 'SPY': '🛡️', 'GLD': '⚠️'}.get(gp['signal'], '❓')
    print(f"\n  {signal_icon} TODAY'S SIGNAL: {gp['signal']}")
    print(f"  Reason: {gp['reason']}")
    print(f"  Vol: {gp['vol_21d']:.1f}% | SPY: ${gp['sma20']:.2f} (SMA20) vs ${gp['sma200']:.2f} (SMA200)")
    print(f"  Cross-asset regime: {gp['composite_regime']}")
    if gp['distance_to_switch'] > 0:
        print(f"  Distance to switch: {gp['distance_to_switch']:.1f}pp below threshold")
    for w in gp.get('warnings', []):
        print(f"  {w}")

    # Save
    output = {
        'date': latest_date,
        'generated': today_str,
        'signal': gp,
        'regime': {k: v for k, v in regime.items() if isinstance(v, (str, int, float))},
        'momentum': mt.to_dict('records'),
    }
    fname = os.path.join(OUTPUT_DIR, f'dashboard_{latest_date}.json')
    with open(fname, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved to {fname}")


if __name__ == '__main__':
    main()

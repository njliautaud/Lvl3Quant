#!/usr/bin/env python3
"""
Holdings Analysis v1 - Relative Strength, Momentum, Portfolio Optimization, CC Scan
Analyzes actual Robinhood positions with yfinance data.
"""

import json
import warnings
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings('ignore')

# ── Actual positions from Robinhood (pulled 2026-07-23) ──
POSITIONS = {
    'AVAV':  {'qty': 6,   'avg_cost': 244.60},
    'CRDO':  {'qty': 6,   'avg_cost': 155.34},
    'SKM':   {'qty': 35,  'avg_cost': 37.22},
    'NOK':   {'qty': 40,  'avg_cost': 13.67},
    'OUST':  {'qty': 45,  'avg_cost': 26.92},
    'SEDG':  {'qty': 5,   'avg_cost': 56.24},
    'KLIC':  {'qty': 7,   'avg_cost': 53.89},
    'BTQ':   {'qty': 100, 'avg_cost': 5.64},
    'SANM':  {'qty': 6,   'avg_cost': 175.04},
    'FRSH':  {'qty': 100, 'avg_cost': 9.64},
    'FLY':   {'qty': 15,  'avg_cost': 37.93},
    'SHMD':  {'qty': 80,  'avg_cost': 7.20},
    'POWI':  {'qty': 6,   'avg_cost': 73.75},
    'RDW':   {'qty': 40,  'avg_cost': 10.45},
    'HIMX':  {'qty': 15,  'avg_cost': 20.79},
    'ADEA':  {'qty': 24,  'avg_cost': 30.59},
    'FPS':   {'qty': 10,  'avg_cost': 47.30},
    'ENPH':  {'qty': 8,   'avg_cost': 49.29},
    'TRT':   {'qty': 20,  'avg_cost': 15.73},
    'AMKR':  {'qty': 2,   'avg_cost': 68.69},
    'STM':   {'qty': 8,   'avg_cost': 69.84},
    'AAOI':  {'qty': 4,   'avg_cost': 162.44},
    'INTA':  {'qty': 27,  'avg_cost': 26.73},
    'VECO':  {'qty': 8,   'avg_cost': 59.69},
    'CLSK':  {'qty': 60,  'avg_cost': 13.92},
    'KRKNF': {'qty': 150, 'avg_cost': 4.34},
}

# Stocks to consider adding
ADD_CANDIDATES = ['LMT', 'RTX', 'GD', 'HII',   # Defense/Aero
                  'XOM', 'CVX', 'OXY', 'DVN',   # Energy
                  'UNH', 'HCA', 'CI', 'ELV',    # Healthcare
                  'PLTR', 'AXON']                # Momentum leaders

BENCHMARK = 'SPY'
LOOKBACK_DAYS = 90  # fetch extra for rolling calcs on 60d window


def fetch_data(symbols, days=LOOKBACK_DAYS):
    """Fetch adjusted close + volume for all symbols."""
    all_syms = list(set(symbols + [BENCHMARK]))
    end = datetime.now()
    start = end - timedelta(days=days)
    print(f"Fetching {len(all_syms)} tickers ({days}d)...")
    data = yf.download(all_syms, start=start, end=end, progress=False, auto_adjust=True)
    if data.empty:
        print("ERROR: yfinance returned no data")
        sys.exit(1)
    # Handle multi-level columns
    close = data['Close'] if 'Close' in data.columns.get_level_values(0) else data['Close']
    volume = data['Volume'] if 'Volume' in data.columns.get_level_values(0) else data['Volume']
    # Also get High/Low for MFI
    high = data['High'] if 'High' in data.columns.get_level_values(0) else None
    low = data['Low'] if 'Low' in data.columns.get_level_values(0) else None
    return close, volume, high, low


def compute_rsi(series, period=14):
    """RSI calculation."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_mfi(high, low, close, volume, period=14):
    """Money Flow Index."""
    if high is None or low is None:
        return pd.Series(np.nan, index=close.index)
    tp = (high + low + close) / 3
    mf = tp * volume
    delta = tp.diff()
    pos_mf = mf.where(delta > 0, 0).rolling(period).sum()
    neg_mf = mf.where(delta <= 0, 0).rolling(period).sum()
    mfi = 100 - (100 / (1 + pos_mf / neg_mf.replace(0, np.nan)))
    return mfi


def compute_obv_slope(close, volume, window=20):
    """OBV slope over window (normalized)."""
    direction = np.sign(close.diff())
    obv = (volume * direction).cumsum()
    # Linear regression slope over last `window` days
    if len(obv.dropna()) < window:
        return np.nan
    recent = obv.dropna().iloc[-window:]
    x = np.arange(len(recent))
    slope = np.polyfit(x, recent.values, 1)[0]
    # Normalize by mean volume
    mean_vol = volume.dropna().iloc[-window:].mean()
    if mean_vol == 0:
        return 0
    return slope / mean_vol


def relative_strength_analysis(close, volume, high, low):
    """Section 1: Relative strength vs SPY."""
    spy = close[BENCHMARK].dropna()
    results = {}

    for sym in POSITIONS:
        if sym not in close.columns:
            print(f"  SKIP {sym} - no price data")
            continue

        px = close[sym].dropna()
        # Align
        common = px.index.intersection(spy.index)
        if len(common) < 30:
            print(f"  SKIP {sym} - only {len(common)} common days")
            continue

        px_c = px.loc[common]
        spy_c = spy.loc[common]

        ret_stock = px_c.pct_change().dropna()
        ret_spy = spy_c.pct_change().dropna()

        # Use last 60 trading days
        n = min(60, len(ret_stock))
        ret_stock = ret_stock.iloc[-n:]
        ret_spy = ret_spy.iloc[-n:]

        # RS ratio (rolling 20d cumulative return ratio)
        cum_stock_20 = (1 + ret_stock).rolling(20).apply(lambda x: x.prod() - 1, raw=True)
        cum_spy_20 = (1 + ret_spy).rolling(20).apply(lambda x: x.prod() - 1, raw=True)
        rs_ratio = cum_stock_20.iloc[-1] / cum_spy_20.iloc[-1] if abs(cum_spy_20.iloc[-1]) > 1e-6 else np.nan

        # Alpha (annualized excess return)
        total_ret_stock = (1 + ret_stock).prod() - 1
        total_ret_spy = (1 + ret_spy).prod() - 1
        alpha_60d = total_ret_stock - total_ret_spy
        alpha_ann = alpha_60d * (252 / n)

        # Beta
        cov_matrix = np.cov(ret_stock.values, ret_spy.values)
        beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] != 0 else np.nan

        # Correlation
        corr = ret_stock.corr(ret_spy)

        # Hidden alpha flag: low beta (<0.8) + positive alpha
        hidden_alpha = bool(beta < 0.8 and alpha_60d > 0)

        results[sym] = {
            'rs_ratio_20d': round(float(rs_ratio), 3) if not np.isnan(rs_ratio) else None,
            'alpha_60d_pct': round(float(alpha_60d * 100), 2),
            'alpha_annualized_pct': round(float(alpha_ann * 100), 1),
            'beta_to_spy': round(float(beta), 3),
            'correlation_to_spy': round(float(corr), 3),
            'return_60d_pct': round(float(total_ret_stock * 100), 2),
            'hidden_alpha': hidden_alpha,
        }

    return results


def momentum_quality_score(close, volume, high, low):
    """Section 2: Momentum quality scoring."""
    results = {}

    for sym in POSITIONS:
        if sym not in close.columns:
            continue

        px = close[sym].dropna()
        vol = volume[sym].dropna() if sym in volume.columns else pd.Series()
        hi = high[sym].dropna() if high is not None and sym in high.columns else None
        lo = low[sym].dropna() if low is not None and sym in low.columns else None

        if len(px) < 60:
            continue

        last_price = float(px.iloc[-1])
        score = 0
        details = {}

        # Momentum 20d / 60d
        mom_20 = float((px.iloc[-1] / px.iloc[-20] - 1) * 100) if len(px) >= 20 else 0
        mom_60 = float((px.iloc[-1] / px.iloc[-60] - 1) * 100) if len(px) >= 60 else 0
        details['momentum_20d_pct'] = round(mom_20, 2)
        details['momentum_60d_pct'] = round(mom_60, 2)
        if mom_20 > 5: score += 2
        elif mom_20 > 0: score += 1
        elif mom_20 < -5: score -= 1

        if mom_60 > 10: score += 2
        elif mom_60 > 0: score += 1
        elif mom_60 < -10: score -= 1

        # RSI(14)
        rsi = compute_rsi(px, 14)
        rsi_val = float(rsi.iloc[-1])
        details['rsi_14'] = round(rsi_val, 1)
        if 50 < rsi_val < 70: score += 2  # healthy uptrend
        elif rsi_val >= 70: score += 1     # overbought, less credit
        elif rsi_val < 30: score -= 1      # oversold

        # MFI(14)
        if hi is not None and lo is not None and not vol.empty:
            mfi = compute_mfi(hi, lo, px, vol, 14)
            mfi_val = float(mfi.iloc[-1]) if not np.isnan(mfi.iloc[-1]) else 50
        else:
            mfi_val = 50
        details['mfi_14'] = round(mfi_val, 1)
        if mfi_val > 60: score += 1
        elif mfi_val < 30: score -= 1

        # SMA positions
        sma_20 = float(px.rolling(20).mean().iloc[-1])
        sma_50 = float(px.rolling(50).mean().iloc[-1]) if len(px) >= 50 else sma_20
        details['above_sma20'] = last_price > sma_20
        details['above_sma50'] = last_price > sma_50
        if last_price > sma_20: score += 1
        if last_price > sma_50: score += 1

        # OBV slope
        if not vol.empty and len(vol) >= 20:
            obv_s = compute_obv_slope(px, vol, 20)
            details['obv_slope_norm'] = round(float(obv_s), 4) if not np.isnan(obv_s) else 0
            if obv_s > 0.05: score += 1
            elif obv_s < -0.05: score -= 1
        else:
            details['obv_slope_norm'] = 0

        # Cost basis P&L
        cost = POSITIONS[sym]['avg_cost']
        pnl_pct = (last_price - cost) / cost * 100
        details['last_price'] = round(last_price, 2)
        details['avg_cost'] = cost
        details['unrealized_pnl_pct'] = round(pnl_pct, 1)
        details['position_value'] = round(last_price * POSITIONS[sym]['qty'], 2)

        details['score'] = score
        results[sym] = details

    return results


def portfolio_optimization(rs_results, mom_results, close):
    """Section 3: Portfolio optimization suggestions."""
    spy = close[BENCHMARK].dropna()
    spy_ret = spy.pct_change().dropna().iloc[-60:]

    # Aggregate portfolio metrics
    total_value = sum(m.get('position_value', 0) for m in mom_results.values())
    weighted_beta = 0
    weighted_corr = 0
    weights = {}

    for sym in rs_results:
        if sym in mom_results:
            w = mom_results[sym].get('position_value', 0) / total_value if total_value > 0 else 0
            weights[sym] = w
            weighted_beta += w * rs_results[sym].get('beta_to_spy', 1)
            weighted_corr += w * rs_results[sym].get('correlation_to_spy', 0.5)

    # Rank holdings by combined score
    ranked = []
    for sym in mom_results:
        rs = rs_results.get(sym, {})
        mo = mom_results[sym]
        combined = mo['score'] + (2 if rs.get('hidden_alpha', False) else 0)
        combined += 1 if rs.get('alpha_60d_pct', 0) > 5 else 0
        ranked.append({
            'symbol': sym,
            'combined_score': combined,
            'momentum_score': mo['score'],
            'alpha_60d_pct': rs.get('alpha_60d_pct', 0),
            'beta': rs.get('beta_to_spy', None),
            'return_60d_pct': rs.get('return_60d_pct', 0),
            'unrealized_pnl_pct': mo.get('unrealized_pnl_pct', 0),
            'position_value': mo.get('position_value', 0),
            'above_sma20': mo.get('above_sma20', False),
            'rsi_14': mo.get('rsi_14', 50),
        })

    ranked.sort(key=lambda x: x['combined_score'], reverse=True)

    # Identify strongest / weakest
    strongest = [r for r in ranked if r['combined_score'] >= 5]
    weakest = sorted([r for r in ranked if r['combined_score'] <= 2],
                     key=lambda x: x['combined_score'])

    # Sector concentration (manual mapping for known symbols)
    sector_map = {
        'AVAV': 'Aerospace/Defense', 'CRDO': 'Semiconductors', 'SKM': 'Telecom',
        'NOK': 'Telecom', 'OUST': 'Industrial Tech', 'SEDG': 'Solar/Energy',
        'KLIC': 'Semiconductors', 'BTQ': 'Biotech', 'SANM': 'Electronics/EMS',
        'FRSH': 'Software', 'FLY': 'Aerospace Leasing', 'SHMD': 'Industrial',
        'POWI': 'Semiconductors', 'RDW': 'Space/Aerospace', 'HIMX': 'Semiconductors',
        'ADEA': 'IP/Technology', 'FPS': 'Defense', 'ENPH': 'Solar/Energy',
        'TRT': 'Food/Consumer', 'AMKR': 'Semiconductors', 'STM': 'Semiconductors',
        'AAOI': 'Fiber Optics', 'INTA': 'Technology', 'VECO': 'Semiconductors',
        'CLSK': 'Bitcoin Mining', 'KRKNF': 'Crypto Exchange',
    }

    sector_exposure = {}
    for sym, sec in sector_map.items():
        if sym in mom_results:
            val = mom_results[sym].get('position_value', 0)
            sector_exposure[sec] = sector_exposure.get(sec, 0) + val

    sector_pct = {s: round(v / total_value * 100, 1) for s, v in sector_exposure.items()} if total_value > 0 else {}
    sector_pct = dict(sorted(sector_pct.items(), key=lambda x: -x[1]))

    # ADD candidates analysis
    add_suggestions = analyze_add_candidates(close)

    return {
        'total_portfolio_value': round(total_value, 2),
        'portfolio_weighted_beta': round(weighted_beta, 3),
        'portfolio_weighted_corr_to_spy': round(weighted_corr, 3),
        'sector_exposure_pct': sector_pct,
        'strongest_holdings': [{'symbol': s['symbol'], 'combined_score': s['combined_score'],
                                 'alpha_60d': s['alpha_60d_pct'], 'return_60d': s['return_60d_pct']}
                                for s in strongest[:5]],
        'weakest_holdings': [{'symbol': s['symbol'], 'combined_score': s['combined_score'],
                               'alpha_60d': s['alpha_60d_pct'], 'return_60d': s['return_60d_pct'],
                               'unrealized_pnl_pct': s['unrealized_pnl_pct']}
                              for s in weakest[:5]],
        'trim_candidates': [s['symbol'] for s in weakest
                            if s['combined_score'] <= 1 and s['alpha_60d_pct'] < 0],
        'add_suggestions': add_suggestions,
        'rankings': ranked,
    }


def analyze_add_candidates(close):
    """Analyze potential add candidates."""
    spy = close[BENCHMARK].dropna()
    spy_ret = spy.pct_change().dropna().iloc[-60:]
    suggestions = []

    for sym in ADD_CANDIDATES:
        if sym not in close.columns:
            continue
        px = close[sym].dropna()
        if len(px) < 60:
            continue

        ret = px.pct_change().dropna()
        n = min(60, len(ret))
        ret = ret.iloc[-n:]
        spy_r = spy_ret.iloc[-n:]

        # Align
        common = ret.index.intersection(spy_r.index)
        ret = ret.loc[common]
        spy_r = spy_r.loc[common]

        mom_60 = float((px.iloc[-1] / px.iloc[-60] - 1) * 100) if len(px) >= 60 else 0
        mom_20 = float((px.iloc[-1] / px.iloc[-20] - 1) * 100) if len(px) >= 20 else 0

        alpha = float(((1 + ret).prod() - 1 - ((1 + spy_r).prod() - 1)) * 100)

        cov = np.cov(ret.values, spy_r.values)
        beta = cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else 1

        rsi = compute_rsi(px, 14)
        rsi_val = float(rsi.iloc[-1])

        last = float(px.iloc[-1])
        sma20 = float(px.rolling(20).mean().iloc[-1])

        score = 0
        if mom_20 > 5: score += 2
        elif mom_20 > 0: score += 1
        if mom_60 > 10: score += 2
        elif mom_60 > 0: score += 1
        if 50 < rsi_val < 70: score += 2
        if last > sma20: score += 1
        if alpha > 5: score += 1

        suggestions.append({
            'symbol': sym,
            'score': score,
            'momentum_60d_pct': round(mom_60, 1),
            'momentum_20d_pct': round(mom_20, 1),
            'alpha_60d_pct': round(alpha, 1),
            'beta': round(float(beta), 2),
            'rsi_14': round(rsi_val, 1),
            'above_sma20': last > sma20,
            'price': round(last, 2),
        })

    suggestions.sort(key=lambda x: x['score'], reverse=True)
    return suggestions[:5]


def covered_call_scan(close, volume, mom_results):
    """Section 4: Covered call opportunity scan."""
    results = []

    for sym, info in POSITIONS.items():
        if sym not in close.columns or sym not in mom_results:
            continue

        mo = mom_results[sym]
        px = close[sym].dropna()
        if len(px) < 20:
            continue

        last = float(px.iloc[-1])
        sma20 = float(px.rolling(20).mean().iloc[-1])
        qty = info['qty']

        # Only scan stocks above 20 SMA
        if last <= sma20:
            continue

        # Can only sell covered calls in 100-share lots
        lots = qty // 100

        # Estimate 0.30-delta 30-DTE call premium
        # Use historical vol to estimate (Black-Scholes-ish approximation)
        ret = px.pct_change().dropna().iloc[-30:]
        hvol = float(ret.std() * np.sqrt(252))

        # Approximate 0.30 delta strike ≈ price * (1 + 0.52 * vol * sqrt(30/365))
        dte = 30
        strike_approx = last * (1 + 0.52 * hvol * np.sqrt(dte / 365))

        # Approximate premium using simplified BS
        # Premium ≈ price * vol * sqrt(T) * pdf(d1) for OTM call
        from scipy.stats import norm
        T = dte / 365
        d1 = (np.log(last / strike_approx) + (0.05 + hvol**2/2) * T) / (hvol * np.sqrt(T)) if hvol > 0 else 0
        d2 = d1 - hvol * np.sqrt(T) if hvol > 0 else 0
        premium_est = last * norm.cdf(d1) - strike_approx * np.exp(-0.05 * T) * norm.cdf(d2)
        premium_est = max(premium_est, 0)

        # CC yield (monthly, annualized)
        cc_yield_monthly = (premium_est / last * 100) if last > 0 else 0
        cc_yield_annual = cc_yield_monthly * 12

        results.append({
            'symbol': sym,
            'last_price': round(last, 2),
            'sma20': round(sma20, 2),
            'shares': qty,
            'full_lots': lots,
            'can_sell_cc': lots > 0,
            'hist_vol_ann': round(hvol * 100, 1),
            'est_strike_030delta': round(strike_approx, 2),
            'est_premium_per_share': round(premium_est, 2),
            'est_cc_yield_monthly_pct': round(cc_yield_monthly, 2),
            'est_cc_yield_annual_pct': round(cc_yield_annual, 1),
            'rsi': mo.get('rsi_14', 50),
            'note': 'Above SMA20 - good CC candidate' if lots > 0 else f'Only {qty} shares - need 100 for CC',
        })

    results.sort(key=lambda x: x['est_cc_yield_annual_pct'], reverse=True)
    return results


def print_summary(rs, mom, portfolio, cc):
    """Print clean terminal summary."""
    print("\n" + "=" * 70)
    print("  HOLDINGS ANALYSIS v1 — 2026-07-23")
    print("=" * 70)

    # Portfolio overview
    print(f"\n  Portfolio Value: ${portfolio['total_portfolio_value']:,.0f}")
    print(f"  Portfolio Beta:  {portfolio['portfolio_weighted_beta']:.2f}")
    print(f"  SPY Correlation: {portfolio['portfolio_weighted_corr_to_spy']:.2f}")

    # Sector breakdown
    print("\n  SECTOR EXPOSURE:")
    for sec, pct in list(portfolio['sector_exposure_pct'].items())[:8]:
        bar = "█" * int(pct / 2)
        print(f"    {sec:25s} {pct:5.1f}%  {bar}")

    # Top holdings by combined score
    print("\n  STRONGEST MOMENTUM + ALPHA:")
    for h in portfolio['strongest_holdings']:
        print(f"    {h['symbol']:6s}  Score={h['combined_score']:+d}  "
              f"Alpha60d={h['alpha_60d']:+.1f}%  Ret60d={h['return_60d']:+.1f}%")

    # Weakest
    print("\n  WEAKEST / TRIM CANDIDATES:")
    for h in portfolio['weakest_holdings']:
        print(f"    {h['symbol']:6s}  Score={h['combined_score']:+d}  "
              f"Alpha60d={h['alpha_60d']:+.1f}%  P&L={h['unrealized_pnl_pct']:+.1f}%")

    # Hidden alpha stocks
    hidden = [s for s, r in rs.items() if r.get('hidden_alpha')]
    if hidden:
        print(f"\n  HIDDEN ALPHA (low beta + positive alpha): {', '.join(hidden)}")

    # Add suggestions
    print("\n  ADD CANDIDATES (Defense/Energy/Healthcare momentum):")
    for s in portfolio['add_suggestions'][:3]:
        print(f"    {s['symbol']:5s}  Score={s['score']}  Mom60d={s['momentum_60d_pct']:+.1f}%  "
              f"Alpha={s['alpha_60d_pct']:+.1f}%  Beta={s['beta']:.2f}  RSI={s['rsi_14']:.0f}")

    # Covered call scan
    print("\n  COVERED CALL OPPORTUNITIES (above SMA20):")
    cc_viable = [c for c in cc if c['can_sell_cc']]
    cc_not = [c for c in cc if not c['can_sell_cc']]
    if cc_viable:
        for c in cc_viable[:5]:
            print(f"    {c['symbol']:6s}  Strike~${c['est_strike_030delta']:.0f}  "
                  f"Prem~${c['est_premium_per_share']:.2f}  "
                  f"Yield={c['est_cc_yield_annual_pct']:.0f}%/yr  Vol={c['hist_vol_ann']:.0f}%")
    else:
        print("    No positions with 100+ shares AND above SMA20")

    if cc_not:
        print(f"\n  Above SMA20 but <100 shares (can't sell CC):")
        for c in cc_not[:5]:
            print(f"    {c['symbol']:6s}  {c['shares']} shares  "
                  f"EstYield={c['est_cc_yield_annual_pct']:.0f}%/yr if had 100 shares")

    # Key insights
    print("\n  KEY INSIGHTS:")
    semi_pct = portfolio['sector_exposure_pct'].get('Semiconductors', 0)
    if semi_pct > 30:
        print(f"    ! Heavy semiconductor concentration ({semi_pct:.0f}%) — consider trimming weakest semi")
    if portfolio['portfolio_weighted_beta'] > 1.3:
        print(f"    ! High portfolio beta ({portfolio['portfolio_weighted_beta']:.2f}) — vulnerable in selloffs")
    if portfolio['portfolio_weighted_beta'] < 0.8:
        print(f"    Portfolio is defensive (beta {portfolio['portfolio_weighted_beta']:.2f})")

    trim = portfolio.get('trim_candidates', [])
    if trim:
        print(f"    TRIM: {', '.join(trim)} — negative momentum + negative alpha")

    print("\n" + "=" * 70)


def main():
    symbols = list(POSITIONS.keys()) + ADD_CANDIDATES
    close, volume, high, low = fetch_data(symbols)

    print("\n--- Section 1: Relative Strength Analysis ---")
    rs = relative_strength_analysis(close, volume, high, low)

    print("\n--- Section 2: Momentum Quality Score ---")
    mom = momentum_quality_score(close, volume, high, low)

    print("\n--- Section 3: Portfolio Optimization ---")
    portfolio = portfolio_optimization(rs, mom, close)

    print("\n--- Section 4: Covered Call Scan ---")
    cc = covered_call_scan(close, volume, mom)

    # Print summary
    print_summary(rs, mom, portfolio, cc)

    # Save to JSON
    output = {
        'analysis_date': '2026-07-23',
        'generated_at': datetime.now().isoformat(),
        'relative_strength': rs,
        'momentum_scores': {k: v for k, v in mom.items()},
        'portfolio_optimization': portfolio,
        'covered_call_scan': cc,
    }

    out_path = Path('/home/jupiter/Lvl3Quant/research/findings/holdings_analysis_v1.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()

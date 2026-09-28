#!/usr/bin/env python3
"""
Earnings Week Options Optimizer v1
===================================
Scores upcoming earnings stocks against ALL our validated signals
to find the highest-conviction plays for the agentic account ($645).

Combines findings from:
1. Post-earnings momentum (buy calls after gap-ups, Sharpe 0.90)
2. PEAD drift (long-only post-earnings, Sharpe 1.18)
3. ETF sector momentum (which sectors are hot)
4. Vol crush dynamics (IV rank → premium selling opportunity)
5. Quality-momentum features (stock-level momentum/quality)

For each upcoming earnings stock, produces:
- Pre-earnings play score (jade lizard/vol crush opportunity)
- Post-earnings momentum play score (if it gaps up 3%+, buy calls)
- PEAD drift play score (if strong beat, hold the drift)
- Overall conviction score + recommended play

Target: next week's earnings (V Mon 7/28, MSFT/PG Tue 7/29, AAPL/AMZN/MA Wed 7/30)
"""

import sys
import os
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/jupiter/Lvl3Quant')


# ─── EARNINGS CALENDAR ──────────────────────────────────────────────────────

EARNINGS_NEXT_WEEK = [
    {'ticker': 'V', 'date': '2026-07-28', 'time': 'AMC', 'sector': 'Financials'},
    {'ticker': 'F', 'date': '2026-07-28', 'time': 'AMC', 'sector': 'Consumer Discretionary'},
    {'ticker': 'MSFT', 'date': '2026-07-29', 'time': 'AMC', 'sector': 'Technology'},
    {'ticker': 'PG', 'date': '2026-07-29', 'time': 'BMO', 'sector': 'Consumer Staples'},
    {'ticker': 'SOFI', 'date': '2026-07-29', 'time': 'AMC', 'sector': 'Financials'},
    {'ticker': 'AAPL', 'date': '2026-07-30', 'time': 'AMC', 'sector': 'Technology'},
    {'ticker': 'AMZN', 'date': '2026-07-30', 'time': 'AMC', 'sector': 'Consumer Discretionary'},
    {'ticker': 'MA', 'date': '2026-07-30', 'time': 'AMC', 'sector': 'Financials'},
    {'ticker': 'RIVN', 'date': '2026-07-30', 'time': 'AMC', 'sector': 'Consumer Discretionary'},
]

AGENTIC_CAPITAL = 645
MAX_PER_TRADE = 250  # Leave buffer


def fetch_stock_data(tickers):
    """Fetch recent price data for earnings stocks."""
    import yfinance as yf

    all_data = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, start='2024-01-01', end='2026-07-25',
                           progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [str(c[0]).lower() for c in df.columns]
            else:
                df.columns = [str(c).lower() for c in df.columns]
            df = df.reset_index()
            df.columns = [str(c).lower() for c in df.columns]
            all_data[ticker] = df
            print(f"  {ticker}: {len(df)} days")
        except Exception as e:
            print(f"  {ticker}: FAILED ({e})")
    return all_data


def compute_momentum_quality_score(df):
    """Quality-momentum score (same features as validated QM ranker)."""
    close = df['close'].values
    n = len(close)
    if n < 252:
        return {}

    # Momentum features
    mom_1m = close[-1] / close[-21] - 1
    mom_3m = close[-1] / close[-63] - 1
    mom_6m = close[-1] / close[-126] - 1
    mom_12_1 = close[-21] / close[-252] - 1  # Skip last month

    # Momentum acceleration
    mom_accel = mom_1m - (mom_3m / 3)

    # Volatility
    rets = np.diff(close[-63:]) / close[-63:-1]
    vol_60d = np.std(rets) * np.sqrt(252)
    sharpe_6m = mom_6m / (vol_60d + 1e-8)

    # Skewness (key feature in QM ranker)
    skew_63d = float(pd.Series(rets).skew())

    # Max drawdown 63d
    window = close[-63:]
    peak = np.maximum.accumulate(window)
    dd = (window - peak) / peak
    maxdd_63d = np.min(dd)

    # RSI
    deltas = np.diff(close[-15:])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    rs = np.mean(gains) / (np.mean(losses) + 1e-10)
    rsi_14 = 100 - (100 / (1 + rs))

    # Composite QM score
    qm_score = (
        0.25 * mom_12_1 +
        0.20 * sharpe_6m +
        0.20 * mom_accel +
        0.15 * (1 + maxdd_63d) +
        0.10 * (skew_63d / 3) +
        0.10 * (rsi_14 - 50) / 50
    )

    return {
        'qm_score': qm_score,
        'mom_1m': mom_1m,
        'mom_3m': mom_3m,
        'mom_6m': mom_6m,
        'mom_12_1': mom_12_1,
        'mom_accel': mom_accel,
        'vol_60d': vol_60d,
        'sharpe_6m': sharpe_6m,
        'skew_63d': skew_63d,
        'maxdd_63d': maxdd_63d,
        'rsi_14': rsi_14,
    }


def compute_vol_metrics(df):
    """IV rank proxy and vol dynamics for options pricing."""
    close = df['close'].values
    n = len(close)
    if n < 252:
        return {}

    # Realized vol at multiple horizons
    rets_10d = np.diff(close[-11:]) / close[-11:-1]
    rets_21d = np.diff(close[-22:]) / close[-22:-1]
    rets_63d = np.diff(close[-64:]) / close[-64:-1]
    rets_252d = np.diff(close[-253:]) / close[-253:-1]

    rv_10d = np.std(rets_10d) * np.sqrt(252)
    rv_21d = np.std(rets_21d) * np.sqrt(252)
    rv_63d = np.std(rets_63d) * np.sqrt(252)
    rv_252d = np.std(rets_252d) * np.sqrt(252)

    # IV rank proxy: where is current short-term vol relative to 1-year range
    # Higher = more elevated = better for selling premium
    vol_range = rv_252d  # Use 1yr vol as the range
    if vol_range > 0:
        iv_rank_proxy = min(100, max(0, (rv_21d / vol_range) * 50 + 25))
    else:
        iv_rank_proxy = 50

    # Vol term structure: short > long = elevated = good for selling
    vol_ratio = rv_10d / (rv_63d + 1e-8)

    # Earnings vol bump expectation
    # Historical earnings moves (simplified: use 10d vol around typical earnings dates)
    if n > 252:
        # Estimate typical earnings move from quarterly patterns
        quarterly_rets = []
        for q_offset in [63, 126, 189, 252]:
            if n > q_offset + 5:
                q_move = abs(close[-(q_offset)] / close[-(q_offset+1)] - 1)
                quarterly_rets.append(q_move)
        avg_earnings_move = np.mean(quarterly_rets) if quarterly_rets else rv_21d / np.sqrt(252) * 5
    else:
        avg_earnings_move = 0.05  # Default 5%

    return {
        'rv_10d': rv_10d,
        'rv_21d': rv_21d,
        'rv_63d': rv_63d,
        'rv_252d': rv_252d,
        'iv_rank_proxy': iv_rank_proxy,
        'vol_ratio': vol_ratio,
        'avg_earnings_move': avg_earnings_move,
    }


def compute_historical_earnings_patterns(df, ticker):
    """Analyze past earnings reactions for this stock."""
    close = df['close'].values
    dates = pd.to_datetime(df['date']).values
    n = len(close)

    if n < 252:
        return {}

    # Look for large gap moves (>2%) as earnings proxy
    daily_rets = np.diff(close) / close[:-1]
    gap_days = np.where(np.abs(daily_rets) > 0.02)[0]

    if len(gap_days) < 4:
        return {
            'avg_gap': 0,
            'gap_up_pct': 0.5,
            'post_gap_drift': 0,
            'n_gaps': len(gap_days),
        }

    # Recent large gaps (last 2 years)
    recent_gaps = gap_days[gap_days > n - 504]

    gaps = daily_rets[recent_gaps]
    gap_ups = gaps[gaps > 0]
    gap_downs = gaps[gaps < 0]

    # Post-gap drift (10 days after)
    post_gap_drifts = []
    for g in recent_gaps:
        if g + 10 < n:
            drift = close[g + 10] / close[g] - 1
            post_gap_drifts.append(drift)

    # Post-gap-UP drift specifically (PEAD signal)
    post_gap_up_drifts = []
    for g in recent_gaps:
        if daily_rets[g] > 0.02 and g + 10 < n:
            drift = close[g + 10] / close[g] - 1
            post_gap_up_drifts.append(drift)

    return {
        'avg_gap': np.mean(np.abs(gaps)),
        'avg_gap_up': np.mean(gap_ups) if len(gap_ups) > 0 else 0,
        'avg_gap_down': np.mean(gap_downs) if len(gap_downs) > 0 else 0,
        'gap_up_pct': len(gap_ups) / len(gaps),
        'post_gap_drift': np.mean(post_gap_drifts) if post_gap_drifts else 0,
        'post_gap_up_drift': np.mean(post_gap_up_drifts) if post_gap_up_drifts else 0,
        'n_gaps': len(recent_gaps),
    }


def estimate_option_costs(price, vol, days_to_expiry=7):
    """Estimate call option costs for agentic account sizing."""
    from scipy.stats import norm

    T = days_to_expiry / 365.0
    r = 0.05

    # ATM call
    d1 = (np.log(1) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    atm_call = price * (norm.cdf(d1) - np.exp(-r * T) * norm.cdf(d2))

    # 5% OTM call
    K_otm = price * 1.05
    d1_otm = (np.log(price / K_otm) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
    d2_otm = d1_otm - vol * np.sqrt(T)
    otm_call = price * norm.cdf(d1_otm) - K_otm * np.exp(-r * T) * norm.cdf(d2_otm)

    # Contract costs (× 100 shares)
    atm_cost = atm_call * 100
    otm_cost = max(otm_call * 100, 0)

    return {
        'atm_call_cost': atm_cost,
        'otm_call_cost': otm_cost,
        'affordable_atm': atm_cost <= MAX_PER_TRADE,
        'affordable_otm': otm_cost <= MAX_PER_TRADE,
        'contracts_atm': int(MAX_PER_TRADE / atm_cost) if atm_cost > 0 else 0,
        'contracts_otm': int(MAX_PER_TRADE / otm_cost) if otm_cost > 0 else 0,
    }


def score_earnings_play(ticker_info, qm, vol, earnings, options, price):
    """
    Produce composite scores for different play types.
    Each score 0-100, with specific recommendation.
    """
    scores = {}

    # ─── PRE-EARNINGS PREMIUM SELL (jade lizard / vol crush) ─────────
    # Better with: high IV rank, moderate expected move, good WR history
    pre_score = 0
    pre_score += min(30, vol.get('iv_rank_proxy', 0) * 0.3)  # Up to 30 for high IV rank
    pre_score += 20 if vol.get('vol_ratio', 1) > 1.1 else 0   # Short vol elevated
    pre_score += 15 if earnings.get('gap_up_pct', 0.5) > 0.4 else 0  # Balanced gaps
    pre_score += 10 if vol.get('rv_21d', 0) > 0.20 else 0  # Decent premium
    # Penalty: our account is too small for premium selling
    pre_score *= 0.3  # Heavy discount — we can't actually sell premium with $645
    scores['pre_earnings_sell'] = min(100, pre_score)

    # ─── POST-EARNINGS GAP-UP CALL BUY (validated Sharpe 0.90) ───────
    # Better with: history of gap-ups, strong post-gap drift, affordable options
    post_score = 0
    post_score += min(25, earnings.get('gap_up_pct', 0.5) * 50)  # History of gaps up
    post_score += min(20, max(0, earnings.get('post_gap_up_drift', 0) * 500))  # Post-gap drift
    post_score += 20 if qm.get('mom_3m', 0) > 0 else 0  # Momentum tailwind
    post_score += 15 if options.get('affordable_atm', False) else (10 if options.get('affordable_otm', False) else 0)
    post_score += 10 if qm.get('rsi_14', 50) < 70 else 0  # Not overbought
    post_score += 10 if qm.get('qm_score', 0) > 0 else 0  # Quality-momentum positive
    scores['post_gap_up_calls'] = min(100, post_score)

    # ─── PEAD DRIFT (40-day hold after positive beat) ─────────────────
    # Better with: strong sector momentum, good historical drift, moderate vol
    pead_score = 0
    pead_score += min(25, max(0, earnings.get('post_gap_up_drift', 0) * 500))
    pead_score += 20 if qm.get('mom_6m', 0) > 0.05 else 0  # Sector/stock in uptrend
    pead_score += 15 if vol.get('rv_21d', 0) < 0.35 else 0  # Not too volatile
    pead_score += 15 if qm.get('sharpe_6m', 0) > 0.5 else 0  # Good risk-adjusted momentum
    pead_score += 10 if earnings.get('n_gaps', 0) >= 4 else 0  # Enough history
    # For agentic account, we'd buy stock (can't afford) or ITM LEAPS
    if price > 100:
        pead_score *= 0.5  # Too expensive for stock position
    scores['pead_drift'] = min(100, pead_score)

    # ─── OVERALL CONVICTION ──────────────────────────────────────────
    # Weighted average favoring plays that are actually executable with $645
    scores['overall'] = (
        0.10 * scores['pre_earnings_sell'] +  # Can't really do this
        0.50 * scores['post_gap_up_calls'] +  # Best executable play
        0.40 * scores['pead_drift']            # If affordable
    )

    return scores


def generate_playbook(results):
    """Generate actionable playbook for the user."""
    print(f"\n{'='*70}")
    print(f"EARNINGS WEEK PLAYBOOK — July 28-30, 2026")
    print(f"Agentic Account: ${AGENTIC_CAPITAL} | Max per trade: ${MAX_PER_TRADE}")
    print(f"{'='*70}")

    # Sort by overall conviction
    sorted_results = sorted(results, key=lambda x: x['scores']['overall'], reverse=True)

    for r in sorted_results:
        ticker = r['ticker']
        date = r['date']
        time = r['time']
        price = r['price']
        scores = r['scores']
        options = r['options']
        qm = r['qm']
        vol = r['vol']
        earnings = r['earnings']

        # Determine best play
        play_scores = {
            'Post-earnings call buy': scores['post_gap_up_calls'],
            'PEAD drift (if gaps up)': scores['pead_drift'],
            'Pre-earnings premium': scores['pre_earnings_sell'],
        }
        best_play = max(play_scores, key=play_scores.get)
        best_score = play_scores[best_play]

        # Conviction level
        if scores['overall'] >= 60:
            conviction = "🔥 HIGH"
        elif scores['overall'] >= 40:
            conviction = "⚠️ MEDIUM"
        else:
            conviction = "❌ LOW"

        print(f"\n{'─'*50}")
        print(f"{conviction} | {ticker} @ ${price:.0f} | Reports {date} {time}")
        print(f"  Overall Score: {scores['overall']:.0f}/100")
        print(f"  Best Play: {best_play} (score {best_score:.0f})")

        # Key metrics
        print(f"  Momentum: 1m={qm.get('mom_1m', 0):+.1%}, 3m={qm.get('mom_3m', 0):+.1%}, "
              f"6m={qm.get('mom_6m', 0):+.1%}")
        print(f"  Vol: RV21d={vol.get('rv_21d', 0):.0%}, IV rank proxy={vol.get('iv_rank_proxy', 0):.0f}")
        print(f"  Earnings history: {earnings.get('n_gaps', 0)} big moves, "
              f"{earnings.get('gap_up_pct', 0):.0%} were up, "
              f"avg gap {earnings.get('avg_gap', 0):.1%}")
        print(f"  Post-gap-up drift: {earnings.get('post_gap_up_drift', 0):+.1%} (10d)")
        print(f"  RSI: {qm.get('rsi_14', 50):.0f}, QM Score: {qm.get('qm_score', 0):.3f}")

        # Options costs
        print(f"  ATM call: ${options.get('atm_call_cost', 0):.0f}/contract "
              f"({'✅ affordable' if options.get('affordable_atm') else '❌ too expensive'})")
        print(f"  OTM call: ${options.get('otm_call_cost', 0):.0f}/contract "
              f"({'✅ affordable' if options.get('affordable_otm') else '❌ too expensive'})")

        # Specific recommendation
        if scores['overall'] >= 50:
            if scores['post_gap_up_calls'] >= 50 and options.get('affordable_atm'):
                print(f"  📋 PLAN: If {ticker} gaps UP 3%+ after earnings → buy 1 ATM call "
                      f"(~${options.get('atm_call_cost', 0):.0f}), hold 10 days")
            elif scores['post_gap_up_calls'] >= 50 and options.get('affordable_otm'):
                print(f"  📋 PLAN: If {ticker} gaps UP 3%+ → buy 1 OTM call "
                      f"(~${options.get('otm_call_cost', 0):.0f}), hold 10 days")
            elif scores['pead_drift'] >= 50 and price < 50:
                print(f"  📋 PLAN: If {ticker} beats estimates → buy shares (~${price:.0f}), hold 40 days")
            else:
                print(f"  📋 PLAN: Monitor for gap-up. Options may be too expensive for account size.")
        else:
            print(f"  📋 PLAN: SKIP — low conviction or unaffordable")

    # Summary table
    print(f"\n{'='*70}")
    print(f"SUMMARY RANKING")
    print(f"{'='*70}")
    print(f"{'Ticker':<8} {'Price':>7} {'Overall':>8} {'PostGap':>8} {'PEAD':>8} {'PreSell':>8} {'ATM$':>7} {'Action':<20}")
    print("-" * 80)
    for r in sorted_results:
        s = r['scores']
        action = "WATCH" if s['overall'] >= 40 else "SKIP"
        if s['overall'] >= 60 and r['options'].get('affordable_atm'):
            action = "BUY IF GAP UP"
        elif s['overall'] >= 60 and r['options'].get('affordable_otm'):
            action = "BUY OTM IF GAP"
        elif s['overall'] >= 50:
            action = "MONITOR"
        print(f"{r['ticker']:<8} ${r['price']:>6.0f} {s['overall']:>7.0f} {s['post_gap_up_calls']:>7.0f} "
              f"{s['pead_drift']:>7.0f} {s['pre_earnings_sell']:>7.0f} "
              f"${r['options'].get('atm_call_cost', 0):>6.0f} {action:<20}")

    return sorted_results


def main():
    print("=" * 70)
    print("EARNINGS WEEK OPTIONS OPTIMIZER v1")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    tickers = [e['ticker'] for e in EARNINGS_NEXT_WEEK]
    print(f"\nFetching data for {len(tickers)} earnings stocks...")
    stock_data = fetch_stock_data(tickers)

    results = []
    for earning in EARNINGS_NEXT_WEEK:
        ticker = earning['ticker']
        if ticker not in stock_data:
            print(f"\n  {ticker}: NO DATA, skipping")
            continue

        df = stock_data[ticker]
        if len(df) < 252:
            print(f"\n  {ticker}: insufficient data ({len(df)} days)")
            continue

        price = df['close'].iloc[-1]
        print(f"\n  Analyzing {ticker} @ ${price:.2f}...")

        qm = compute_momentum_quality_score(df)
        vol = compute_vol_metrics(df)
        earnings_hist = compute_historical_earnings_patterns(df, ticker)
        options = estimate_option_costs(price, vol.get('rv_21d', 0.25))

        scores = score_earnings_play(earning, qm, vol, earnings_hist, options, price)

        results.append({
            'ticker': ticker,
            'date': earning['date'],
            'time': earning['time'],
            'sector': earning['sector'],
            'price': price,
            'qm': qm,
            'vol': vol,
            'earnings': earnings_hist,
            'options': options,
            'scores': scores,
        })

    if not results:
        print("NO RESULTS")
        return

    sorted_results = generate_playbook(results)

    # Save results
    save_path = '/home/jupiter/Lvl3Quant/state/earnings_week_playbook.json'
    save_data = {
        'generated': datetime.now().isoformat(),
        'account_capital': AGENTIC_CAPITAL,
        'earnings': []
    }
    for r in sorted_results:
        save_data['earnings'].append({
            'ticker': r['ticker'],
            'date': r['date'],
            'time': r['time'],
            'price': float(r['price']),
            'overall_score': float(r['scores']['overall']),
            'post_gap_score': float(r['scores']['post_gap_up_calls']),
            'pead_score': float(r['scores']['pead_drift']),
            'atm_cost': float(r['options'].get('atm_call_cost', 0)),
            'otm_cost': float(r['options'].get('otm_call_cost', 0)),
            'affordable': r['options'].get('affordable_atm', False) or r['options'].get('affordable_otm', False),
            'momentum_1m': float(r['qm'].get('mom_1m', 0)),
            'momentum_3m': float(r['qm'].get('mom_3m', 0)),
            'rsi': float(r['qm'].get('rsi_14', 50)),
            'vol_21d': float(r['vol'].get('rv_21d', 0)),
            'iv_rank': float(r['vol'].get('iv_rank_proxy', 50)),
            'gap_up_pct': float(r['earnings'].get('gap_up_pct', 0.5)),
            'post_gap_drift': float(r['earnings'].get('post_gap_up_drift', 0)),
        })

    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nPlaybook saved to state/earnings_week_playbook.json")

    print(f"\nCompleted at {datetime.now()}")
    return save_data


if __name__ == '__main__':
    main()

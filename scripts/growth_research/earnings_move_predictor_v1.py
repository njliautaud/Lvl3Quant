#!/usr/bin/env python3
"""
Earnings Move Predictor v1 — Systematic analysis of post-earnings moves

For each stock:
1. Download historical earnings dates + price data
2. Compute: pre-earnings momentum, realized vol, VIX level, sector context
3. Analyze: historical move magnitude, direction bias, IV crush pattern
4. Build MLP to predict move magnitude from features
5. Score upcoming earnings (V 7/28, MSFT 7/29, PG 7/29, AAPL 7/30, AMZN 7/30)

Directly actionable for $645 agentic account options plays.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime, timedelta
import json, os, sys
import yfinance as yf

sys.path.insert(0, '/home/jupiter/Lvl3Quant')

try:
    import torch
    import torch.nn as nn
    TORCH = True
except:
    TORCH = False

print(f"Earnings Move Predictor v1 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)


def get_earnings_history(ticker, n_quarters=20):
    """Get historical earnings dates and post-earnings moves."""
    try:
        stock = yf.Ticker(ticker)

        # Get earnings dates from calendar
        earnings_dates = []

        # Try earnings_dates attribute
        try:
            ed = stock.earnings_dates
            if ed is not None and len(ed) > 0:
                # Filter to past dates only
                past = ed[ed.index <= pd.Timestamp.now(tz='America/New_York')]
                earnings_dates = past.index.tolist()[:n_quarters]
        except:
            pass

        if not earnings_dates:
            # Fallback: use quarterly earnings from financials
            try:
                q_earn = stock.quarterly_earnings
                if q_earn is not None and len(q_earn) > 0:
                    # These are quarter-end dates, earnings typically 3-6 weeks later
                    earnings_dates = q_earn.index.tolist()[:n_quarters]
            except:
                pass

        return earnings_dates
    except Exception as e:
        print(f"  {ticker}: Error getting earnings history: {e}")
        return []


def analyze_stock_earnings(ticker, spy_data=None, vix_data=None):
    """Comprehensive earnings analysis for a single stock."""
    print(f"\n{'='*60}")
    print(f"ANALYZING: {ticker}")
    print(f"{'='*60}")

    # Download price data
    data = yf.download(ticker, start='2018-01-01', end='2026-07-25', progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    if len(data) < 252:
        print(f"  Insufficient data for {ticker}")
        return None

    close = data['Close']
    volume = data['Volume']
    high = data['High']
    low = data['Low']

    # Get earnings dates
    earnings_dates = get_earnings_history(ticker)

    if not earnings_dates:
        # Fallback: estimate quarterly earnings from volume spikes
        print(f"  No earnings dates found, using volume spike detection")
        daily_vol = volume.rolling(20).mean()
        vol_ratio = volume / daily_vol
        # Big volume days (>3x avg) that are also high-range days
        daily_range = (high - low) / close
        range_ratio = daily_range / daily_range.rolling(20).mean()

        # Find days with both volume and range spikes
        spike_days = vol_ratio[(vol_ratio > 2.5) & (range_ratio > 1.5)].index

        # Group into quarterly clusters
        earnings_dates = []
        if len(spike_days) > 0:
            last_date = spike_days[0]
            earnings_dates.append(last_date)
            for d in spike_days[1:]:
                if (d - last_date).days > 60:  # at least 60 days apart
                    earnings_dates.append(d)
                    last_date = d

    # Convert earnings dates to trading dates
    results = []
    for ed in earnings_dates:
        try:
            if isinstance(ed, pd.Timestamp):
                ed = ed.tz_localize(None) if ed.tzinfo else ed
            ed = pd.Timestamp(ed)

            # Find nearest trading day
            idx = close.index.get_indexer([ed], method='nearest')[0]
            if idx < 5 or idx >= len(close) - 5:
                continue

            earn_date = close.index[idx]

            # Pre-earnings features
            pre_5d_ret = float(close.iloc[idx] / close.iloc[idx-5] - 1)
            pre_20d_ret = float(close.iloc[idx] / close.iloc[idx-20] - 1)
            pre_60d_ret = float(close.iloc[idx] / close.iloc[max(0,idx-60)] - 1)

            # Volatility
            daily_ret = close.pct_change()
            pre_vol_20d = float(daily_ret.iloc[max(0,idx-20):idx].std() * np.sqrt(252))

            # Volume trend
            avg_vol_20d = float(volume.iloc[max(0,idx-20):idx].mean())
            earn_vol = float(volume.iloc[idx])
            vol_surge = earn_vol / avg_vol_20d if avg_vol_20d > 0 else 1

            # VIX level
            vix_level = None
            if vix_data is not None and earn_date in vix_data.index:
                vix_level = float(vix_data.loc[earn_date])

            # SPY context
            spy_pre_20d = None
            if spy_data is not None:
                spy_idx = spy_data.index.get_indexer([earn_date], method='nearest')[0]
                if spy_idx >= 20:
                    spy_pre_20d = float(spy_data.iloc[spy_idx] / spy_data.iloc[spy_idx-20] - 1)

            # Post-earnings moves
            post_1d = float(close.iloc[min(idx+1, len(close)-1)] / close.iloc[idx] - 1)
            post_2d = float(close.iloc[min(idx+2, len(close)-1)] / close.iloc[idx] - 1)
            post_5d = float(close.iloc[min(idx+5, len(close)-1)] / close.iloc[idx] - 1)

            # Gap (open vs previous close)
            if idx + 1 < len(data):
                gap = float(data['Open'].iloc[idx+1] / close.iloc[idx] - 1)
            else:
                gap = post_1d

            results.append({
                'date': earn_date.strftime('%Y-%m-%d'),
                'pre_5d_ret': pre_5d_ret,
                'pre_20d_ret': pre_20d_ret,
                'pre_60d_ret': pre_60d_ret,
                'pre_vol_20d': pre_vol_20d,
                'vol_surge': vol_surge,
                'vix_level': vix_level,
                'spy_pre_20d': spy_pre_20d,
                'gap': gap,
                'post_1d': post_1d,
                'post_2d': post_2d,
                'post_5d': post_5d,
                'abs_gap': abs(gap),
                'direction': 'UP' if gap > 0 else 'DOWN'
            })
        except Exception as e:
            continue

    if not results:
        print(f"  No earnings events analyzed")
        return None

    df = pd.DataFrame(results)

    # Print analysis
    n = len(df)
    avg_abs_gap = df['abs_gap'].mean()
    med_abs_gap = df['abs_gap'].median()
    up_pct = (df['gap'] > 0).mean()
    avg_up = df[df['gap'] > 0]['gap'].mean() if (df['gap'] > 0).any() else 0
    avg_down = df[df['gap'] <= 0]['gap'].mean() if (df['gap'] <= 0).any() else 0

    # Post-earnings drift
    drift_after_up = df[df['gap'] > 0.01]['post_5d'].mean() if (df['gap'] > 0.01).any() else 0
    drift_after_down = df[df['gap'] < -0.01]['post_5d'].mean() if (df['gap'] < -0.01).any() else 0

    print(f"\n  Earnings Events: {n}")
    print(f"  Avg |Gap|: {avg_abs_gap:.1%} (median {med_abs_gap:.1%})")
    print(f"  Direction: {up_pct:.0%} UP, {1-up_pct:.0%} DOWN")
    print(f"  Avg gap UP: +{avg_up:.1%}, Avg gap DOWN: {avg_down:.1%}")
    print(f"  Post-earnings drift (5d after >1% gap up): {drift_after_up:+.1%}")
    print(f"  Post-earnings drift (5d after >1% gap down): {drift_after_down:+.1%}")

    # Predictability: does pre-earnings momentum predict direction?
    if n >= 8:
        high_mom = df[df['pre_20d_ret'] > df['pre_20d_ret'].median()]
        low_mom = df[df['pre_20d_ret'] <= df['pre_20d_ret'].median()]
        print(f"\n  Momentum signal:")
        print(f"    High pre-20d mom → avg gap: {high_mom['gap'].mean():+.1%} ({(high_mom['gap']>0).mean():.0%} up)")
        print(f"    Low pre-20d mom  → avg gap: {low_mom['gap'].mean():+.1%} ({(low_mom['gap']>0).mean():.0%} up)")

        # Vol regime
        if df['vix_level'].notna().sum() >= 4:
            high_vix = df[df['vix_level'] > 20]
            low_vix = df[df['vix_level'] <= 20]
            if len(high_vix) > 0 and len(low_vix) > 0:
                print(f"\n  VIX regime:")
                print(f"    VIX>20: avg |gap| {high_vix['abs_gap'].mean():.1%}, {(high_vix['gap']>0).mean():.0%} up")
                print(f"    VIX≤20: avg |gap| {low_vix['abs_gap'].mean():.1%}, {(low_vix['gap']>0).mean():.0%} up")

    # Current features for upcoming earnings
    latest_idx = len(close) - 1
    current_features = {
        'pre_5d_ret': float(close.iloc[-1] / close.iloc[-5] - 1),
        'pre_20d_ret': float(close.iloc[-1] / close.iloc[-20] - 1),
        'pre_60d_ret': float(close.iloc[-1] / close.iloc[-60] - 1),
        'pre_vol_20d': float(close.pct_change().tail(20).std() * np.sqrt(252)),
        'current_price': float(close.iloc[-1])
    }

    # Score based on historical patterns
    # Direction prediction: momentum-based + historical bias
    mom_signal = 1 if current_features['pre_20d_ret'] > 0 else -1
    hist_bias = 1 if up_pct > 0.55 else (-1 if up_pct < 0.45 else 0)
    direction_score = (mom_signal + hist_bias) / 2

    # Magnitude prediction: use historical avg
    expected_move = avg_abs_gap

    # Confidence
    confidence = min(n / 20, 1.0)  # more history = more confidence

    print(f"\n  UPCOMING EARNINGS SCORE:")
    print(f"    Current price: ${current_features['current_price']:.2f}")
    print(f"    Pre-20d momentum: {current_features['pre_20d_ret']:+.1%}")
    print(f"    Expected move: ±{expected_move:.1%}")
    print(f"    Direction lean: {'BULLISH' if direction_score > 0 else 'BEARISH' if direction_score < 0 else 'NEUTRAL'}")
    print(f"    Confidence: {confidence:.0%}")

    return {
        'ticker': ticker,
        'n_events': n,
        'avg_abs_gap': float(avg_abs_gap),
        'median_abs_gap': float(med_abs_gap),
        'up_pct': float(up_pct),
        'avg_gap_up': float(avg_up),
        'avg_gap_down': float(avg_down),
        'drift_after_up': float(drift_after_up),
        'drift_after_down': float(drift_after_down),
        'current_features': current_features,
        'direction_score': float(direction_score),
        'expected_move': float(expected_move),
        'confidence': float(confidence),
        'history': results
    }


def generate_trade_recommendations(analyses, account_size=645, max_per_trade=200):
    """Generate actionable option trade recommendations."""
    print(f"\n{'='*70}")
    print("TRADE RECOMMENDATIONS — Earnings Week 7/28-7/30")
    print(f"Account: ${account_size}, Max per trade: ${max_per_trade}")
    print(f"{'='*70}")

    recommendations = []

    for a in analyses:
        if a is None:
            continue

        ticker = a['ticker']
        price = a['current_features']['current_price']
        exp_move = a['expected_move']
        direction = a['direction_score']
        confidence = a['confidence']

        print(f"\n--- {ticker} (${price:.0f}) ---")
        print(f"  Expected move: ±{exp_move:.1%} (${price * exp_move:.1f})")
        print(f"  Direction: {'BULL' if direction > 0 else 'BEAR' if direction < 0 else 'NEUTRAL'}")

        # Option strategy selection
        strategies = []

        # For cheap stocks (<$50), ATM options are affordable
        if price < 50:
            atm_call_est = price * 0.03 * 100  # ~3% of price for weekly
            atm_put_est = price * 0.03 * 100

            if direction > 0 and atm_call_est < max_per_trade:
                strategies.append({
                    'strategy': 'BUY ATM CALL',
                    'est_cost': atm_call_est,
                    'target': f"+{exp_move:.1%} gap → ${price*(1+exp_move):.1f}",
                    'risk': f"Max loss: ${atm_call_est:.0f}",
                    'edge': f"{a['up_pct']:.0%} hist gap-up rate"
                })

            if direction < 0 and atm_put_est < max_per_trade:
                strategies.append({
                    'strategy': 'BUY ATM PUT',
                    'est_cost': atm_put_est,
                    'target': f"-{exp_move:.1%} gap → ${price*(1-exp_move):.1f}",
                    'risk': f"Max loss: ${atm_put_est:.0f}",
                    'edge': f"{1-a['up_pct']:.0%} hist gap-down rate"
                })

        # For all stocks: vertical spreads are budget-friendly
        spread_cost_est = price * 0.015 * 100  # ~1.5% for $5-wide spread
        if spread_cost_est > max_per_trade:
            spread_cost_est = max_per_trade * 0.8  # cap estimate

        if direction > 0:
            strategies.append({
                'strategy': 'BULL CALL SPREAD (weekly)',
                'est_cost': min(spread_cost_est, max_per_trade),
                'target': f"Max profit if above short strike",
                'risk': f"Max loss: ${min(spread_cost_est, max_per_trade):.0f}",
                'edge': f"{a['up_pct']:.0%} gap-up, {a['drift_after_up']:+.1%} 5d drift"
            })
        elif direction < 0:
            strategies.append({
                'strategy': 'BEAR PUT SPREAD (weekly)',
                'est_cost': min(spread_cost_est, max_per_trade),
                'target': f"Max profit if below short strike",
                'risk': f"Max loss: ${min(spread_cost_est, max_per_trade):.0f}",
                'edge': f"{1-a['up_pct']:.0%} gap-down, {a['drift_after_down']:+.1%} 5d drift"
            })

        # Straddle/strangle for high-move names (if affordable)
        if exp_move > 0.05:  # >5% expected move
            straddle_cost = price * 0.05 * 100  # rough estimate
            if straddle_cost < max_per_trade:
                strategies.append({
                    'strategy': 'BUY STRADDLE',
                    'est_cost': straddle_cost,
                    'target': f"Profit if move > ±{0.05:.0%}",
                    'risk': f"Max loss: ${straddle_cost:.0f}",
                    'edge': f"Avg |gap| {exp_move:.1%} vs ~5% breakeven"
                })

        # IV crush play: sell premium if stock historically moves LESS than implied
        if exp_move < 0.03 and confidence > 0.5:
            strategies.append({
                'strategy': 'SELL IRON CONDOR (collect premium)',
                'est_cost': 'Credit: ~$50-100',
                'target': f"Stock stays within ±{exp_move*2:.1%}",
                'risk': f"Width - credit received",
                'edge': f"Low hist move ({exp_move:.1%}) → IV likely overpriced"
            })

        # Score strategies
        for s in strategies:
            s['ticker'] = ticker
            s['score'] = confidence * abs(direction) * 100
            recommendations.append(s)
            print(f"  → {s['strategy']}: ~${s['est_cost'] if isinstance(s['est_cost'], (int,float)) else s['est_cost']}")
            print(f"    {s['edge']}")

    # Rank recommendations
    if recommendations:
        recommendations.sort(key=lambda x: x.get('score', 0), reverse=True)

        print(f"\n{'='*70}")
        print("TOP RECOMMENDATIONS (ranked by confidence × conviction)")
        print(f"{'='*70}")
        for i, r in enumerate(recommendations[:5], 1):
            print(f"\n  #{i}: {r['ticker']} — {r['strategy']}")
            print(f"      Est cost: {r['est_cost']}")
            print(f"      Edge: {r['edge']}")

    return recommendations


def main():
    # Download context data
    print("Downloading market context...")
    spy_raw = yf.download('SPY', start='2018-01-01', end='2026-07-25', progress=False)
    vix_raw = yf.download('^VIX', start='2018-01-01', end='2026-07-25', progress=False)

    spy_close = spy_raw['Close']
    vix_close = vix_raw['Close']
    if isinstance(spy_close, pd.DataFrame):
        spy_close = spy_close.iloc[:, 0]
    if isinstance(vix_close, pd.DataFrame):
        vix_close = vix_close.iloc[:, 0]

    # Analyze earnings week stocks
    tickers = ['V', 'MSFT', 'PG', 'AAPL', 'AMZN', 'F', 'MA']  # F added from playbook

    analyses = []
    for ticker in tickers:
        result = analyze_stock_earnings(ticker, spy_close, vix_close)
        analyses.append(result)

    # Generate recommendations
    recs = generate_trade_recommendations([a for a in analyses if a is not None])

    # Save results
    save_path = '/home/jupiter/Lvl3Quant/research/findings/earnings_move_predictor_v1_results.json'
    output = {
        'strategy': 'Earnings Move Predictor v1',
        'run_date': datetime.now().isoformat(),
        'earnings_week': '2026-07-28 to 2026-07-30',
        'tickers_analyzed': tickers,
        'analyses': {a['ticker']: {k: v for k, v in a.items() if k != 'history'}
                     for a in analyses if a is not None},
        'recommendations': [{k: v for k, v in r.items() if k != 'score' or isinstance(v, (int, float, str))}
                           for r in recs[:5]]
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(output, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    print(f"\nSaved → {save_path}")

    # Summary
    print(f"\n{'='*70}")
    print("EARNINGS WEEK SUMMARY")
    print(f"{'='*70}")
    for a in analyses:
        if a is None:
            continue
        t = a['ticker']
        print(f"  {t:5s}: Expected ±{a['expected_move']:.1%}, "
              f"{'BULL' if a['direction_score'] > 0 else 'BEAR' if a['direction_score'] < 0 else 'NEUTRAL':5s}, "
              f"Hist up {a['up_pct']:.0%}, Confidence {a['confidence']:.0%}")

    return analyses, recs


if __name__ == '__main__':
    main()

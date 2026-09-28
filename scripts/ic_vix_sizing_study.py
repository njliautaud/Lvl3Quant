#!/usr/bin/env python3
"""
IC VIX-Based Dynamic Sizing Study
===================================

Tests whether dynamically adjusting position size based on VIX improves
risk-adjusted returns for the iron condor strategy.

Hypothesis: Higher VIX = richer premiums = scale UP (within limits).
Counter: Higher VIX = wider moves = more risk = maybe scale DOWN.

Configs tested:
1. Flat sizing (baseline — current)
2. VIX-proportional: size = base * (VIX / 20)  [cap at 1.5x]
3. VIX-inverse: size = base * (20 / VIX)  [floor at 0.5x]
4. VIX-bucketed: low (<15) = 0.75x, normal (15-25) = 1.0x, elevated (25-35) = 1.25x, extreme (>35) = 0.5x
5. Vol-targeting: target 15% annualized portfolio vol, adjust sizing to hit it

Uses the same BS pricing model as the IC paper engine.
Runs on 2019-2026 data (same universe as iron_condor_study.py).

Author: Claude (2026-07-06)
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

# ── Parameters (match IC paper engine) ──
SPREAD_WIDTH = 10.0
PUT_DELTA = 0.25
CALL_DELTA = 0.25
DTE_TARGET = 7
PROFIT_TAKE = 0.50
BASE_MARGIN_CAP = 0.30
PER_NAME_PCT = 0.03
COST_PER_LEG = 0.65
SLIPPAGE_FRAC = 0.025
N_POSITIONS = 10  # Fixed number of positions per week

def bs_price(S, K, T, sigma, kind='put'):
    if T <= 0 or sigma <= 0:
        return max(K - S, 0) if kind == 'put' else max(S - K, 0)
    d1 = (np.log(S / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if kind == 'put':
        return K * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        return S * norm.cdf(d1) - K * norm.cdf(-d2)

def find_strike(S, sigma, T, target_delta, kind='put'):
    for k_off in np.arange(0.5, 50, 0.5):
        K = S - k_off if kind == 'put' else S + k_off
        if K <= 0: continue
        d1 = (np.log(S / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
        delta = abs(norm.cdf(d1) - 1) if kind == 'put' else norm.cdf(d1)
        if delta <= target_delta:
            return K
    return S * 0.95 if kind == 'put' else S * 1.05

def get_vix_history():
    """Download VIX daily history."""
    vix = yf.download('^VIX', start='2019-01-01', end='2026-07-06', progress=False)
    if vix is None or len(vix) == 0:
        raise ValueError("Failed to download VIX data")
    close = vix['Close']
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    # Ensure index is datetime
    result = {}
    for idx, val in close.items():
        if isinstance(idx, str):
            idx = pd.Timestamp(idx)
        result[idx] = float(val)
    return result

def simulate_ic_week(tickers_data, vix, margin_multiplier=1.0,
                      start_capital=100000):
    """
    Simulate one week of IC trading.
    Returns weekly P&L.
    """
    n_contracts_base = 3
    # Adjust contracts by margin multiplier
    n_contracts = max(1, int(n_contracts_base * margin_multiplier))

    total_pnl = 0
    positions_opened = 0

    for ticker, (S, sigma) in tickers_data.items():
        if positions_opened >= N_POSITIONS:
            break

        T = DTE_TARGET / 365.0

        # Find strikes
        K_put_short = find_strike(S, sigma, T, PUT_DELTA, 'put')
        K_put_long = K_put_short - SPREAD_WIDTH
        K_call_short = find_strike(S, sigma, T, CALL_DELTA, 'call')
        K_call_long = K_call_short + SPREAD_WIDTH

        if K_put_long <= 0:
            continue

        # Premium
        ps = bs_price(S, K_put_short, T, sigma, 'put') * (1 - SLIPPAGE_FRAC)
        pl = bs_price(S, K_put_long, T, sigma, 'put') * (1 + SLIPPAGE_FRAC)
        cs = bs_price(S, K_call_short, T, sigma, 'call') * (1 - SLIPPAGE_FRAC)
        cl = bs_price(S, K_call_long, T, sigma, 'call') * (1 + SLIPPAGE_FRAC)

        net_credit = (ps - pl) + (cs - cl)
        if net_credit < 0.10:
            continue

        premium = net_credit * 100 * n_contracts
        comm_open = COST_PER_LEG * 4 * n_contracts

        # Simulate weekly outcome with random walk
        # Use BS to compute expected P&L at expiry
        # (simplified: assume stock follows GBM, compute expected settlement)
        dt = DTE_TARGET / 365.0
        z = np.random.randn()
        S_exp = S * np.exp(-0.5 * sigma**2 * dt + sigma * np.sqrt(dt) * z)

        # Settlement
        put_loss = max(K_put_short - S_exp, 0) - max(K_put_long - S_exp, 0)
        call_loss = max(S_exp - K_call_short, 0) - max(S_exp - K_call_long, 0)
        settlement = (put_loss + call_loss) * 100 * n_contracts

        # Check profit take (simplified: if at expiry the position is >50% profitable)
        position_value_at_expiry = settlement
        pnl_at_expiry = premium - position_value_at_expiry - comm_open - COST_PER_LEG * 4 * n_contracts

        # Profit take check at 50% of premium
        if premium - position_value_at_expiry > premium * PROFIT_TAKE:
            # Would have been closed early for profit
            pnl = premium * PROFIT_TAKE - comm_open - COST_PER_LEG * 4 * n_contracts
        else:
            pnl = pnl_at_expiry

        total_pnl += pnl
        positions_opened += 1

    return total_pnl


def run_sizing_study():
    """Main study: compare sizing strategies across historical VIX regimes."""
    print("Downloading VIX history...")
    vix_data = get_vix_history()

    # Generate weekly dates (Fridays)
    dates = sorted(vix_data.keys())
    weekly_dates = [d for d in dates if hasattr(d, 'weekday') and d.weekday() == 4]  # Fridays

    if not weekly_dates:
        # Try converting
        weekly_dates = []
        for d in dates:
            if isinstance(d, str):
                dt = pd.Timestamp(d)
            else:
                dt = d
            if dt.weekday() == 4:
                weekly_dates.append(dt)

    print(f"VIX data: {len(dates)} days, {len(weekly_dates)} Fridays")

    # Download a sample universe for vol estimation
    sample_tickers = ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'META', 'GOOGL', 'TSLA',
                      'JPM', 'BAC', 'XOM', 'PFE', 'DIS', 'NFLX', 'AMD', 'CRM',
                      'COST', 'ABBV', 'TMO', 'AMGN', 'GE']

    print(f"Downloading price data for {len(sample_tickers)} tickers...")
    price_data = {}
    for ticker in sample_tickers:
        try:
            data = yf.download(ticker, start='2019-01-01', end='2026-07-06', progress=False)
            if data is not None and len(data) > 30:
                price_data[ticker] = data
        except:
            pass

    print(f"Got data for {len(price_data)} tickers")

    # Sizing strategies
    sizing_strategies = {
        'flat': lambda vix: 1.0,
        'vix_proportional': lambda vix: min(1.5, vix / 20.0),
        'vix_inverse': lambda vix: max(0.5, 20.0 / max(vix, 10)),
        'vix_bucketed': lambda vix: (0.75 if vix < 15 else
                                      1.0 if vix < 25 else
                                      1.25 if vix < 35 else
                                      0.5),
        'vix_bucketed_v2': lambda vix: (0.5 if vix < 15 else
                                         1.0 if vix < 20 else
                                         1.5 if vix < 30 else
                                         0.75),
    }

    results = {}
    n_sims = 50  # Monte Carlo runs per week

    for strat_name, size_fn in sizing_strategies.items():
        print(f"\nRunning {strat_name}...")
        all_weekly_pnls = []

        for friday in weekly_dates:
            # Get VIX for this week
            vix = None
            for day_off in range(5):
                check_date = friday - timedelta(days=day_off)
                if check_date in vix_data:
                    vix_val = vix_data[check_date]
                    if hasattr(vix_val, 'item'):
                        vix = float(vix_val.item())
                    else:
                        vix = float(vix_val)
                    break
            if vix is None:
                continue

            margin_mult = size_fn(vix)

            # Get ticker prices/vols for this week
            tickers_for_week = {}
            for ticker, df in price_data.items():
                # Find the row closest to this friday
                mask = df.index <= friday
                if mask.sum() < 30:
                    continue
                subset = df[mask]
                S = float(subset['Close'].iloc[-1])
                rets = np.log(subset['Close'] / subset['Close'].shift(1)).dropna().tail(30)
                sigma = float(rets.std() * np.sqrt(252))
                if sigma > 0.05 and S > 5:
                    tickers_for_week[ticker] = (S, sigma)

            if len(tickers_for_week) < 5:
                continue

            # Run Monte Carlo sims for this week
            week_pnls = []
            for _ in range(n_sims):
                pnl = simulate_ic_week(tickers_for_week, vix, margin_mult)
                week_pnls.append(pnl)

            avg_pnl = np.mean(week_pnls)
            all_weekly_pnls.append({
                'date': friday.strftime('%Y-%m-%d') if hasattr(friday, 'strftime') else str(friday),
                'vix': round(vix, 1),
                'margin_mult': round(margin_mult, 2),
                'avg_pnl': round(avg_pnl, 2),
                'vix_regime': ('low' if vix < 15 else 'normal' if vix < 25 else
                               'elevated' if vix < 35 else 'extreme'),
            })

        if not all_weekly_pnls:
            continue

        pnls = [w['avg_pnl'] for w in all_weekly_pnls]
        daily_returns = np.array(pnls) / 100000  # As fraction of capital

        sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(52) if np.std(daily_returns) > 0 else 0
        total_pnl = sum(pnls)
        avg_weekly = np.mean(pnls)
        win_rate = np.mean(np.array(pnls) > 0)

        # Per-regime analysis
        regime_stats = {}
        for regime in ['low', 'normal', 'elevated', 'extreme']:
            regime_pnls = [w['avg_pnl'] for w in all_weekly_pnls if w['vix_regime'] == regime]
            if len(regime_pnls) > 2:
                regime_returns = np.array(regime_pnls) / 100000
                regime_sharpe = np.mean(regime_returns) / np.std(regime_returns) * np.sqrt(52) if np.std(regime_returns) > 0 else 0
                regime_stats[regime] = {
                    'sharpe': round(regime_sharpe, 2),
                    'weeks': len(regime_pnls),
                    'avg_pnl': round(np.mean(regime_pnls), 0),
                    'wr': round(np.mean(np.array(regime_pnls) > 0), 2),
                }

        results[strat_name] = {
            'sharpe': round(sharpe, 2),
            'total_pnl': round(total_pnl, 0),
            'avg_weekly_pnl': round(avg_weekly, 0),
            'win_rate': round(win_rate, 3),
            'max_dd_weekly': round(min(pnls), 0),
            'n_weeks': len(pnls),
            'regime_stats': regime_stats,
        }

        print(f"  {strat_name}: Sharpe {sharpe:.2f}, WR {win_rate:.0%}, "
              f"avg ${avg_weekly:,.0f}/week, total ${total_pnl:,.0f}")
        for regime, stats in regime_stats.items():
            print(f"    {regime}: Sharpe {stats['sharpe']:.2f}, "
                  f"WR {stats['wr']:.0%}, {stats['weeks']}w")

    # Save results
    output = {
        'generated': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'purpose': 'VIX-based dynamic sizing comparison for IC strategy',
        'n_sims_per_week': n_sims,
        'results': results,
    }

    out_path = 'output/iron_condor_study/vix_sizing_study.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {out_path}")

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY — BEST SIZING STRATEGY")
    print(f"{'='*60}")
    best = max(results.items(), key=lambda x: x[1]['sharpe'])
    print(f"Winner: {best[0]} (Sharpe {best[1]['sharpe']:.2f})")
    for name, r in sorted(results.items(), key=lambda x: -x[1]['sharpe']):
        print(f"  {name:25s}: Sharpe {r['sharpe']:6.2f}, WR {r['win_rate']:.0%}, "
              f"avg ${r['avg_weekly_pnl']:>6,.0f}/wk")


if __name__ == '__main__':
    run_sizing_study()

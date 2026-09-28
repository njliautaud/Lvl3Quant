#!/usr/bin/env python3
"""
Paper vs Backtest Baseline Generator
======================================

Snapshots current paper engine positions and computes:
1. BS-modeled fair values at entry time
2. Expected outcomes based on historical backtest distributions
3. Key validation metrics to compare Friday expiry results against

This creates the "prediction" half of the paper validation framework.
Run this at position open; run the companion scorer at/after expiry.

Author: Claude (2026-07-06)
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from scipy.stats import norm
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# Paths
IC_STATE = Path('/home/jupiter/Lvl3Quant/live_trading_linux/wheel_ic_state/state.json')
BPS_STATE = Path('/home/jupiter/Lvl3Quant/live_trading_linux/wheel_bps_state/state.json')
OUTPUT_DIR = Path('/home/jupiter/Lvl3Quant/output/paper_validation')

# BS pricing functions
def bs_price(S, K, T, sigma, r=0.0, kind='put'):
    if T <= 0 or sigma <= 0:
        if kind == 'put':
            return max(K - S, 0)
        else:
            return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if kind == 'put':
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-r * T) * norm.cdf(-d1)
    else:
        return S * np.exp(-r * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(-d2)


def bs_delta(S, K, T, sigma, r=0.0, kind='put'):
    if T <= 0 or sigma <= 0:
        return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if kind == 'put':
        return norm.cdf(d1) - 1
    else:
        return norm.cdf(d1)


def compute_realized_vol(ticker, lookback=30):
    """Get realized vol from recent price history."""
    try:
        data = yf.download(ticker, period=f'{lookback + 10}d', progress=False)
        if data is None or len(data) < 10:
            return 0.30  # fallback
        returns = np.log(data['Close'] / data['Close'].shift(1)).dropna()
        return float(returns.std() * np.sqrt(252))
    except:
        return 0.30


def monte_carlo_outcome(S, sigma, T_days, put_short, put_long, call_short, call_long,
                        premium_received, contracts, n_sims=10000, profit_take_pct=0.50,
                        strategy='ic'):
    """
    Monte Carlo simulation of position outcome.
    Simulates daily price paths and checks profit-take / expiry settlement.
    Returns distribution of outcomes.
    """
    T = T_days / 365.0
    dt = 1.0 / 365.0
    n_steps = max(T_days, 1)

    results = []
    pt_count = 0

    for _ in range(n_sims):
        prices = [S]
        for _ in range(n_steps):
            z = np.random.randn()
            prices.append(prices[-1] * np.exp(-0.5 * sigma**2 * dt + sigma * np.sqrt(dt) * z))

        # Check profit take at each step
        hit_pt = False
        for step in range(1, len(prices)):
            S_t = prices[step]
            T_rem = max((n_steps - step), 0) / 365.0

            if strategy == 'ic':
                # 4-leg IC value
                ps_val = bs_price(S_t, put_short, T_rem, sigma, kind='put')
                pl_val = bs_price(S_t, put_long, T_rem, sigma, kind='put')
                cs_val = bs_price(S_t, call_short, T_rem, sigma, kind='call')
                cl_val = bs_price(S_t, call_long, T_rem, sigma, kind='call')
                close_cost = ((ps_val - pl_val) + (cs_val - cl_val)) * 100 * contracts
            else:
                # BPS value (2-leg)
                ps_val = bs_price(S_t, put_short, T_rem, sigma, kind='put')
                pl_val = bs_price(S_t, put_long, T_rem, sigma, kind='put')
                close_cost = (ps_val - pl_val) * 100 * contracts

            # Commission for closing (4 legs IC, 2 legs BPS)
            legs = 4 if strategy == 'ic' else 2
            comm = 0.65 * legs * contracts
            close_cost += comm

            pnl = premium_received - close_cost
            pt_target = premium_received * profit_take_pct

            if pnl >= pt_target:
                results.append(pnl)
                pt_count += 1
                hit_pt = True
                break

        if not hit_pt:
            # Settle at expiry
            S_final = prices[-1]
            if strategy == 'ic':
                put_loss = max(put_short - S_final, 0) - max(put_long - S_final, 0)
                call_loss = max(S_final - call_short, 0) - max(S_final - call_long, 0)
                settlement_cost = (put_loss + call_loss) * 100 * contracts + 0.65 * 4 * contracts
            else:
                put_loss = max(put_short - S_final, 0) - max(put_long - S_final, 0)
                settlement_cost = put_loss * 100 * contracts + 0.65 * 2 * contracts

            pnl = premium_received - settlement_cost
            results.append(pnl)

    results = np.array(results)
    return {
        'mean_pnl': float(np.mean(results)),
        'median_pnl': float(np.median(results)),
        'p10_pnl': float(np.percentile(results, 10)),
        'p90_pnl': float(np.percentile(results, 90)),
        'win_rate': float(np.mean(results > 0)),
        'profit_take_rate': float(pt_count / n_sims),
        'max_loss': float(np.min(results)),
        'max_gain': float(np.max(results)),
        'std_pnl': float(np.std(results)),
    }


def analyze_position(pos, strategy, current_price, sigma):
    """Analyze a single position."""
    ticker = pos['ticker']
    contracts = pos.get('contracts', 1)
    premium = pos['premium_received']
    expiry = pd.Timestamp(pos['expiry'])
    now = pd.Timestamp.now()
    T_days = max((expiry - now).days, 1)
    T = T_days / 365.0

    if strategy == 'ic':
        put_short = pos['put_short']
        put_long = pos['put_long']
        call_short = pos['call_short']
        call_long = pos['call_long']

        # Current MTM
        ps_val = bs_price(current_price, put_short, T, sigma, kind='put')
        pl_val = bs_price(current_price, put_long, T, sigma, kind='put')
        cs_val = bs_price(current_price, call_short, T, sigma, kind='call')
        cl_val = bs_price(current_price, call_long, T, sigma, kind='call')

        current_close_cost = ((ps_val - pl_val) + (cs_val - cl_val)) * 100 * contracts
        unrealized_pnl = premium - current_close_cost

        # Deltas
        put_delta = abs(bs_delta(current_price, put_short, T, sigma, kind='put'))
        call_delta = abs(bs_delta(current_price, call_short, T, sigma, kind='call'))

        # Distance from short strikes
        put_distance_pct = (current_price - put_short) / current_price * 100
        call_distance_pct = (call_short - current_price) / current_price * 100

        # Monte Carlo
        mc = monte_carlo_outcome(current_price, sigma, T_days,
                                  put_short, put_long, call_short, call_long,
                                  premium, contracts, strategy='ic')

        return {
            'ticker': ticker,
            'strategy': 'IC',
            'contracts': contracts,
            'current_price': round(current_price, 2),
            'sigma': round(sigma, 3),
            'dte': T_days,
            'put_short': put_short,
            'put_long': put_long,
            'call_short': call_short,
            'call_long': call_long,
            'premium_received': round(premium, 2),
            'put_delta': round(put_delta, 3),
            'call_delta': round(call_delta, 3),
            'put_distance_pct': round(put_distance_pct, 1),
            'call_distance_pct': round(call_distance_pct, 1),
            'current_unrealized_pnl': round(unrealized_pnl, 2),
            'mc_prediction': mc,
        }
    else:
        short_strike = pos['short_strike']
        long_strike = pos['long_strike']

        ps_val = bs_price(current_price, short_strike, T, sigma, kind='put')
        pl_val = bs_price(current_price, long_strike, T, sigma, kind='put')
        current_close_cost = (ps_val - pl_val) * 100 * contracts
        unrealized_pnl = premium - current_close_cost

        put_delta = abs(bs_delta(current_price, short_strike, T, sigma, kind='put'))
        distance_pct = (current_price - short_strike) / current_price * 100

        mc = monte_carlo_outcome(current_price, sigma, T_days,
                                  short_strike, long_strike, 0, 0,
                                  premium, contracts, strategy='bps',
                                  profit_take_pct=0.65)

        return {
            'ticker': ticker,
            'strategy': 'BPS',
            'contracts': contracts,
            'current_price': round(current_price, 2),
            'sigma': round(sigma, 3),
            'dte': T_days,
            'short_strike': short_strike,
            'long_strike': long_strike,
            'premium_received': round(premium, 2),
            'put_delta': round(put_delta, 3),
            'distance_pct': round(distance_pct, 1),
            'current_unrealized_pnl': round(unrealized_pnl, 2),
            'mc_prediction': mc,
        }


def main():
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    print(f"\n{'='*60}")
    print(f"PAPER vs BACKTEST BASELINE — {timestamp}")
    print(f"{'='*60}")

    all_results = {
        'generated': timestamp,
        'purpose': 'Baseline predictions for paper engine positions. Compare against actual results at Friday expiry.',
        'ic_positions': [],
        'bps_positions': [],
        'ic_portfolio_prediction': {},
        'bps_portfolio_prediction': {},
    }

    # ── IC Positions ──
    if IC_STATE.exists():
        with open(IC_STATE) as f:
            ic_state = json.load(f)

        print(f"\n--- IRON CONDOR POSITIONS ({len(ic_state['spreads'])}) ---")

        tickers = [s['ticker'] for s in ic_state['spreads']]
        # Batch download current prices
        prices = {}
        sigmas = {}
        for ticker in tickers:
            try:
                data = yf.download(ticker, period='35d', progress=False)
                if data is not None and len(data) > 5:
                    prices[ticker] = float(data['Close'].iloc[-1])
                    rets = np.log(data['Close'] / data['Close'].shift(1)).dropna()
                    sigmas[ticker] = float(rets.std() * np.sqrt(252))
            except:
                pass

        ic_total_premium = 0
        ic_predicted_pnl = 0

        for sp in ic_state['spreads']:
            ticker = sp['ticker']
            if ticker not in prices:
                print(f"  {ticker}: SKIPPED (no price data)")
                continue

            result = analyze_position(sp, 'ic', prices[ticker], sigmas[ticker])
            all_results['ic_positions'].append(result)
            mc = result['mc_prediction']

            ic_total_premium += sp['premium_received']
            ic_predicted_pnl += mc['mean_pnl']

            print(f"  {ticker}: ${prices[ticker]:.1f}, put {sp['put_short']}/{sp['put_long']} "
                  f"call {sp['call_short']}/{sp['call_long']}")
            print(f"    Put delta: {result['put_delta']:.2f}, Call delta: {result['call_delta']:.2f}")
            print(f"    Put {result['put_distance_pct']:.1f}% OTM, Call {result['call_distance_pct']:.1f}% OTM")
            print(f"    Premium: ${sp['premium_received']:.0f}, MC mean PnL: ${mc['mean_pnl']:.0f}")
            print(f"    MC WR: {mc['win_rate']:.0%}, PT rate: {mc['profit_take_rate']:.0%}")

        all_results['ic_portfolio_prediction'] = {
            'total_premium': round(ic_total_premium, 2),
            'predicted_mean_pnl': round(ic_predicted_pnl, 2),
            'predicted_yield': round(ic_predicted_pnl / ic_total_premium * 100, 1) if ic_total_premium > 0 else 0,
            'n_positions': len(all_results['ic_positions']),
        }

        print(f"\n  PORTFOLIO: ${ic_total_premium:.0f} premium, "
              f"predicted ${ic_predicted_pnl:.0f} PnL ({ic_predicted_pnl/ic_total_premium*100:.0f}% yield)")

    # ── BPS Positions ──
    if BPS_STATE.exists():
        with open(BPS_STATE) as f:
            bps_state = json.load(f)

        print(f"\n--- BULL PUT SPREAD POSITIONS ({len(bps_state['spreads'])}) ---")

        tickers = [s['ticker'] for s in bps_state['spreads']]
        for ticker in tickers:
            if ticker not in prices:
                try:
                    data = yf.download(ticker, period='35d', progress=False)
                    if data is not None and len(data) > 5:
                        prices[ticker] = float(data['Close'].iloc[-1])
                        rets = np.log(data['Close'] / data['Close'].shift(1)).dropna()
                        sigmas[ticker] = float(rets.std() * np.sqrt(252))
                except:
                    pass

        bps_total_premium = 0
        bps_predicted_pnl = 0

        for sp in bps_state['spreads']:
            ticker = sp['ticker']
            if ticker not in prices:
                print(f"  {ticker}: SKIPPED (no price data)")
                continue

            result = analyze_position(sp, 'bps', prices[ticker], sigmas[ticker])
            all_results['bps_positions'].append(result)
            mc = result['mc_prediction']

            bps_total_premium += sp['premium_received']
            bps_predicted_pnl += mc['mean_pnl']

            print(f"  {ticker}: ${prices[ticker]:.1f}, put {sp['short_strike']}/{sp['long_strike']}")
            print(f"    Delta: {result['put_delta']:.2f}, {result['distance_pct']:.1f}% OTM")
            print(f"    Premium: ${sp['premium_received']:.0f}, MC mean PnL: ${mc['mean_pnl']:.0f}")
            print(f"    MC WR: {mc['win_rate']:.0%}, PT rate: {mc['profit_take_rate']:.0%}")

        all_results['bps_portfolio_prediction'] = {
            'total_premium': round(bps_total_premium, 2),
            'predicted_mean_pnl': round(bps_predicted_pnl, 2),
            'predicted_yield': round(bps_predicted_pnl / bps_total_premium * 100, 1) if bps_total_premium > 0 else 0,
            'n_positions': len(all_results['bps_positions']),
        }

        print(f"\n  PORTFOLIO: ${bps_total_premium:.0f} premium, "
              f"predicted ${bps_predicted_pnl:.0f} PnL ({bps_predicted_pnl/bps_total_premium*100:.0f}% yield)")

    # Save
    out_path = OUTPUT_DIR / 'week_2026_07_06_baseline.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nBaseline saved to {out_path}")
    print(f"Run scorer after Friday expiry to compare actual vs predicted.")


if __name__ == '__main__':
    main()

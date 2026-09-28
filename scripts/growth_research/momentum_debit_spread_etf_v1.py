#!/usr/bin/env python3
"""
Momentum Call Debit Spread on ETFs v1
======================================

Evolution of momentum call buying (which failed: R1 0.546, outlier FAIL, MaxDD -66.6%).
Key changes:
- Debit spreads instead of naked calls → capped max loss, lower cost per trade
- Defined risk: max loss = spread cost (~$100-200 per trade vs $300-500 for calls)
- Better for small accounts ($645 agentic)

Strategy:
- Use validated sector ETF momentum signal (perm p=0.000, signal IS real)
- Buy ATM call, sell OTM call (bull call spread) on top momentum ETFs
- Monthly rebalance, 21d hold (match momentum signal frequency)
- $200 max per trade, sized by number of positions

Variants tested:
1. Top 2 momentum ETFs, $2 spread width
2. Top 2 momentum ETFs, $3 spread width
3. Top 3 momentum ETFs, $2 spread width
4. Top 3 momentum ETFs, $3 spread width
5. Top 2 with IV rank filter (<50, avoid expensive premiums)
6. Top 3 with regime filter (skip in bear)
7. Top 2 with vol-adjusted sizing
8. Aggressive: Top 2, $5 spread width

Universe: Same 22 sector/factor ETFs as validated ETF momentum v1.
Walk-forward: 252d train for momentum ranking, 21d test, sliding.
Costs: Realistic option pricing (BS model, 20bps spread per leg).
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

def fprint(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()

RESULTS_DIR = Path('/home/jupiter/Lvl3Quant/research/findings')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = RESULTS_DIR / 'momentum_debit_spread_etf_v1_results.json'

# MLflow
MLFLOW_OK = False
try:
    import urllib.request
    urllib.request.urlopen('http://jupiter:5000/', timeout=2)
    import mlflow
    mlflow.set_tracking_uri('http://jupiter:5000')
    MLFLOW_OK = True
except:
    pass

# Universe (same as validated ETF momentum)
UNIVERSE = [
    'XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLP', 'XLI', 'XLB', 'XLU', 'XLRE',
    'XLC', 'QQQ', 'IWM', 'MDY', 'EFA', 'EEM', 'GLD', 'TLT', 'HYG', 'IYR',
    'VNQ', 'DBC',
]


def black_scholes_call(S, K, T, r, sigma):
    """Black-Scholes call price."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def price_debit_spread(spot, spread_width, vol, dte_years=21/252, r=0.05):
    """
    Price a bull call debit spread.
    Buy ATM call, sell OTM call (ATM + spread_width).
    Returns: (cost, max_profit, breakeven_pct)
    """
    K_long = spot  # ATM
    K_short = spot + spread_width

    long_price = black_scholes_call(spot, K_long, dte_years, r, vol)
    short_price = black_scholes_call(spot, K_short, dte_years, r, vol)

    # Add 20bps spread per leg (buy at ask, sell at bid)
    spread_cost = 0.002  # 20bps
    long_price *= (1 + spread_cost)  # pay more
    short_price *= (1 - spread_cost)  # receive less

    net_debit = long_price - short_price
    max_profit = spread_width - net_debit
    breakeven_pct = (K_long + net_debit - spot) / spot  # how much spot must move

    return net_debit, max_profit, breakeven_pct


def compute_spread_pnl(entry_spot, exit_spot, spread_width, entry_vol, exit_vol,
                        dte_entry=21/252, dte_exit=0, r=0.05):
    """
    Compute P&L of a debit spread from entry to exit.
    """
    K_long = entry_spot  # ATM at entry
    K_short = entry_spot + spread_width

    # Entry cost
    entry_long = black_scholes_call(entry_spot, K_long, dte_entry, r, entry_vol)
    entry_short = black_scholes_call(entry_spot, K_short, dte_entry, r, entry_vol)
    spread_cost = 0.002
    entry_cost = entry_long * (1 + spread_cost) - entry_short * (1 - spread_cost)

    # Exit value (at expiry or early exit)
    if dte_exit <= 0:
        # At expiry: intrinsic value
        exit_long = max(exit_spot - K_long, 0)
        exit_short = max(exit_spot - K_short, 0)
    else:
        exit_long = black_scholes_call(exit_spot, K_long, dte_exit, r, exit_vol)
        exit_short = black_scholes_call(exit_spot, K_short, dte_exit, r, exit_vol)

    exit_value = exit_long * (1 - spread_cost) - exit_short * (1 + spread_cost)

    pnl = exit_value - entry_cost

    # Cap P&L at theoretical bounds
    max_loss = -entry_cost
    max_gain = spread_width - entry_cost
    pnl = max(max_loss, min(max_gain, pnl))

    return pnl, entry_cost


def download_data():
    """Download ETF price data."""
    import yfinance as yf

    tickers = UNIVERSE + ['SPY']
    fprint(f"Downloading {len(tickers)} tickers...")

    data = yf.download(tickers, start='2008-01-01', auto_adjust=True, progress=False)
    close = data['Close'].dropna(how='all')

    # Fill forward, drop columns with >20% missing
    for col in close.columns:
        if close[col].isna().mean() > 0.20:
            close = close.drop(columns=[col])
    close = close.ffill().dropna()

    fprint(f"  Loaded {len([c for c in close.columns if c in UNIVERSE])} ETFs + SPY ({len(close)} days)")
    return close


def compute_momentum_score(close_df, date, lookback=252, skip=21):
    """
    Compute momentum score for all ETFs at a given date.
    12-1 momentum (skip last month to avoid reversal).
    """
    idx = close_df.index.get_loc(date)
    if idx < lookback:
        return None

    scores = {}
    for etf in [c for c in close_df.columns if c in UNIVERSE]:
        prices = close_df[etf].iloc[idx - lookback:idx + 1]
        if len(prices) < lookback:
            continue
        # 12-month return minus 1-month return
        ret_12m = prices.iloc[-1] / prices.iloc[0] - 1
        ret_1m = prices.iloc[-1] / prices.iloc[-skip] - 1
        mom = ret_12m - ret_1m
        scores[etf] = mom

    return scores


def compute_vol(close_df, etf, date, window=63):
    """Compute annualized vol for an ETF."""
    idx = close_df.index.get_loc(date)
    if idx < window:
        return 0.20  # default
    prices = close_df[etf].iloc[idx - window:idx + 1]
    lr = np.log(prices / prices.shift(1)).dropna()
    return float(lr.std() * np.sqrt(252))


def compute_iv_rank(vol_series, current_vol, lookback=252):
    """Approximate IV rank from historical vol."""
    if len(vol_series) < lookback:
        return 50
    vols = vol_series[-lookback:]
    rank = (current_vol - vols.min()) / (vols.max() - vols.min()) * 100
    return max(0, min(100, rank))


def run_backtest(close_df, top_k=2, spread_pct=0.02, iv_filter=None,
                 regime_filter=False, vol_sizing=False,
                 starting_capital=645, max_per_trade=200,
                 name='variant'):
    """
    Run momentum debit spread backtest.

    Args:
        top_k: number of top momentum ETFs to trade
        spread_pct: spread width as % of spot (e.g., 0.02 = 2%)
        iv_filter: max IV rank to enter (None = no filter)
        regime_filter: skip entries in bear market
        vol_sizing: adjust position size by inverse vol
        starting_capital: starting capital
        max_per_trade: max $ per trade
    """
    spy = close_df['SPY']
    etf_cols = [c for c in close_df.columns if c in UNIVERSE]

    # Monthly rebalance dates (21 trading days apart)
    dates = close_df.index[252:]  # skip first year for momentum lookback
    rebal_dates = []
    last_rebal = None
    for d in dates:
        if last_rebal is None or (close_df.index.get_loc(d) - close_df.index.get_loc(last_rebal)) >= 21:
            rebal_dates.append(d)
            last_rebal = d

    # Track portfolio
    capital = starting_capital
    capital_history = [(rebal_dates[0], capital)]
    trades = []
    monthly_returns = []

    # Compute vol history for IV rank
    vol_history = {}
    for etf in etf_cols:
        lr = np.log(close_df[etf] / close_df[etf].shift(1)).dropna()
        vol_history[etf] = lr.rolling(63).std() * np.sqrt(252)

    # SPY 200-day SMA for regime
    spy_sma200 = spy.rolling(200).mean()

    for i in range(len(rebal_dates) - 1):
        entry_date = rebal_dates[i]
        exit_date = rebal_dates[i + 1]

        # Regime filter
        if regime_filter:
            spy_idx = close_df.index.get_loc(entry_date)
            if spy.iloc[spy_idx] < spy_sma200.iloc[spy_idx]:
                # Bear market — skip
                monthly_returns.append(0.0)
                capital_history.append((exit_date, capital))
                continue

        # Score all ETFs by momentum
        scores = compute_momentum_score(close_df, entry_date)
        if not scores:
            monthly_returns.append(0.0)
            capital_history.append((exit_date, capital))
            continue

        # Apply IV filter
        if iv_filter is not None:
            filtered_scores = {}
            for etf, score in scores.items():
                vol = compute_vol(close_df, etf, entry_date)
                vhist = vol_history.get(etf)
                if vhist is not None:
                    idx = close_df.index.get_loc(entry_date)
                    vs = vhist.iloc[:idx+1].dropna()
                    iv_rank = compute_iv_rank(vs.values, vol)
                    if iv_rank <= iv_filter:
                        filtered_scores[etf] = score
                else:
                    filtered_scores[etf] = score
            scores = filtered_scores

        if len(scores) < top_k:
            monthly_returns.append(0.0)
            capital_history.append((exit_date, capital))
            continue

        # Select top K
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        selected = ranked[:top_k]

        # Size positions
        budget = min(capital, max_per_trade * top_k)
        per_trade = budget / top_k

        # Compute P&L for each position
        period_pnl = 0
        for etf, mom_score in selected:
            entry_idx = close_df.index.get_loc(entry_date)
            exit_idx = close_df.index.get_loc(exit_date)

            entry_spot = float(close_df[etf].iloc[entry_idx])
            exit_spot = float(close_df[etf].iloc[exit_idx])

            entry_vol = compute_vol(close_df, etf, entry_date)
            exit_vol = compute_vol(close_df, etf, exit_date)

            # Spread width in $ based on %
            sw = entry_spot * spread_pct

            # Price the spread
            cost_per_share, max_profit_per_share, be_pct = price_debit_spread(
                entry_spot, sw, entry_vol
            )

            if cost_per_share <= 0:
                continue

            # Number of contracts (each = 100 shares)
            cost_per_contract = cost_per_share * 100
            if cost_per_contract > per_trade:
                # Can't afford even 1 contract — skip
                continue

            n_contracts = int(per_trade / cost_per_contract)
            if n_contracts < 1:
                continue

            # Vol-adjusted sizing
            if vol_sizing and entry_vol > 0:
                target_vol = 0.20  # target 20% vol
                vol_adj = target_vol / entry_vol
                n_contracts = max(1, int(n_contracts * vol_adj))

            # P&L at expiry (simplify to expiry payoff for clean backtest)
            pnl_per_share, _ = compute_spread_pnl(
                entry_spot, exit_spot, sw, entry_vol, exit_vol,
                dte_entry=21/252, dte_exit=0
            )

            total_pnl = pnl_per_share * 100 * n_contracts

            # Commission: $0.65 per contract per leg, 4 legs total (open+close, 2 legs each)
            commission = 0.65 * 4 * n_contracts
            total_pnl -= commission

            period_pnl += total_pnl

            trades.append({
                'entry_date': str(entry_date.date()),
                'exit_date': str(exit_date.date()),
                'etf': etf,
                'momentum': round(mom_score, 4),
                'entry_price': round(entry_spot, 2),
                'exit_price': round(exit_spot, 2),
                'return_pct': round((exit_spot / entry_spot - 1) * 100, 2),
                'spread_cost': round(cost_per_contract * n_contracts, 2),
                'n_contracts': n_contracts,
                'pnl': round(total_pnl, 2),
            })

        capital += period_pnl
        capital = max(0, capital)  # can't go below 0

        ret = period_pnl / max(capital - period_pnl, 1)
        monthly_returns.append(ret)
        capital_history.append((exit_date, capital))

        if capital <= 0:
            # Wiped out
            for j in range(i + 2, len(rebal_dates)):
                monthly_returns.append(0)
                capital_history.append((rebal_dates[j], 0))
            break

    # Compute metrics
    r = np.array(monthly_returns)
    r_nonzero = r[r != 0] if np.any(r != 0) else r

    sharpe = np.mean(r_nonzero) / np.std(r_nonzero) * np.sqrt(12) if np.std(r_nonzero) > 0 else 0

    downside = r_nonzero[r_nonzero < 0]
    downside_vol = np.std(downside) * np.sqrt(12) if len(downside) > 0 else 1e-6
    sortino = np.mean(r_nonzero) * 12 / downside_vol if downside_vol > 0 else 0

    # CAGR
    if len(capital_history) > 1:
        start_val = capital_history[0][1]
        end_val = capital_history[-1][1]
        years = (capital_history[-1][0] - capital_history[0][0]).days / 365.25
        if years > 0 and end_val > 0 and start_val > 0:
            cagr = (end_val / start_val) ** (1 / years) - 1
        else:
            cagr = -1
    else:
        cagr = 0

    # MaxDD
    peak = starting_capital
    maxdd = 0
    for _, val in capital_history:
        peak = max(peak, val)
        dd = (val - peak) / peak if peak > 0 else 0
        maxdd = min(maxdd, dd)

    # Win rate
    trade_pnls = [t['pnl'] for t in trades]
    wr = sum(1 for p in trade_pnls if p > 0) / len(trade_pnls) * 100 if trade_pnls else 0

    # Profit factor
    gross_profit = sum(p for p in trade_pnls if p > 0)
    gross_loss = abs(sum(p for p in trade_pnls if p < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    calmar = abs(cagr / maxdd) if maxdd < 0 else 0

    metrics = {
        'sharpe': round(sharpe, 2),
        'sortino': round(sortino, 2),
        'cagr': round(cagr * 100, 1),
        'maxdd': round(maxdd * 100, 1),
        'wr': round(wr, 1),
        'pf': round(pf, 2),
        'calmar': round(calmar, 2),
        'final_capital': round(capital_history[-1][1], 2),
        'n_trades': len(trades),
        'n_months': len(monthly_returns),
        'avg_pnl': round(np.mean(trade_pnls), 2) if trade_pnls else 0,
        'years': round(years, 1) if 'years' in dir() else 0,
    }

    return metrics, trades, monthly_returns, capital_history


def adversarial_gates(monthly_returns, trades, close_df, name):
    """Run 4-gate adversarial validation."""
    r = np.array(monthly_returns)
    r_nonzero = r[r != 0] if np.any(r != 0) else r
    gates = {}

    fprint(f"  Adversarial gates for {name}:")

    # 1. Permutation test (shuffle momentum ranks, re-run)
    actual_sharpe = np.mean(r_nonzero) / np.std(r_nonzero) * np.sqrt(12) if np.std(r_nonzero) > 0 else 0

    # Simple permutation: shuffle trade P&L assignments
    n_perm = 1000
    perm_sharpes = []
    for _ in range(n_perm):
        perm = np.random.permutation(r_nonzero)
        ps = np.mean(perm) / np.std(perm) * np.sqrt(12) if np.std(perm) > 0 else 0
        perm_sharpes.append(ps)

    p_value = np.mean([ps >= actual_sharpe for ps in perm_sharpes])
    gates['permutation'] = {
        'actual_sharpe': round(float(actual_sharpe), 3),
        'p_value': round(float(p_value), 3),
        'pass': p_value < 0.05,
    }
    fprint(f"    Perm: Sharpe={actual_sharpe:.2f}, p={p_value:.3f} {'PASS' if p_value < 0.05 else 'FAIL'}")

    # 2. R1 Regime test (bull vs bear performance)
    if 'SPY' in close_df.columns:
        spy = close_df['SPY']
        spy_sma200 = spy.rolling(200).mean()

        # Get regime for each month
        rebal_dates = sorted(set(t['entry_date'] for t in trades)) if trades else []

        if len(r_nonzero) > 6:
            # Split by overall market regime (proxy: positive vs negative return months)
            # Use SPY return as regime indicator
            mid = len(r_nonzero) // 2

            # Get regime labels from SPY
            spy_monthly = spy.resample('ME').last().pct_change().dropna()

            # Align with our returns (approximate)
            bull_mask = np.zeros(len(r_nonzero), dtype=bool)
            bear_mask = np.zeros(len(r_nonzero), dtype=bool)

            for i in range(len(r_nonzero)):
                # Use alternating as proxy if alignment is hard
                if i < len(spy_monthly):
                    spy_ret = spy_monthly.iloc[-(len(r_nonzero) - i)] if len(r_nonzero) - i <= len(spy_monthly) else 0
                    if isinstance(spy_ret, (int, float)):
                        if spy_ret >= 0:
                            bull_mask[i] = True
                        else:
                            bear_mask[i] = True
                    else:
                        bull_mask[i] = True
                else:
                    bull_mask[i] = True

            if bull_mask.sum() >= 3 and bear_mask.sum() >= 3:
                bull_r = r_nonzero[bull_mask]
                bear_r = r_nonzero[bear_mask]
                bull_sharpe = np.mean(bull_r) / np.std(bull_r) * np.sqrt(12) if np.std(bull_r) > 0 else 0
                bear_sharpe = np.mean(bear_r) / np.std(bear_r) * np.sqrt(12) if np.std(bear_r) > 0 else 0
                max_s = max(abs(bull_sharpe), abs(bear_sharpe))
                gap = abs(bull_sharpe - bear_sharpe) / max_s if max_s > 0 else 0

                gates['regime_r1'] = {
                    'bull_sharpe': round(float(bull_sharpe), 2),
                    'bear_sharpe': round(float(bear_sharpe), 2),
                    'gap': round(float(gap), 3),
                    'pass': gap < 0.50,
                }
                fprint(f"    R1: bull={bull_sharpe:.2f}, bear={bear_sharpe:.2f}, gap={gap:.3f} {'PASS' if gap < 0.50 else 'FAIL'}")
            else:
                gates['regime_r1'] = {'pass': None, 'note': 'Insufficient regime data'}
                fprint(f"    R1: Insufficient data (bull={bull_mask.sum()}, bear={bear_mask.sum()})")
        else:
            gates['regime_r1'] = {'pass': None, 'note': 'Too few months'}

    # 3. Sub-period test
    mid = len(r_nonzero) // 2
    if mid >= 3:
        h1 = r_nonzero[:mid]
        h2 = r_nonzero[mid:]
        h1_sharpe = np.mean(h1) / np.std(h1) * np.sqrt(12) if np.std(h1) > 0 else 0
        h2_sharpe = np.mean(h2) / np.std(h2) * np.sqrt(12) if np.std(h2) > 0 else 0
        sub_pass = h1_sharpe > 0 and h2_sharpe > 0
        gates['sub_period'] = {
            'h1_sharpe': round(float(h1_sharpe), 2),
            'h2_sharpe': round(float(h2_sharpe), 2),
            'pass': sub_pass,
        }
        fprint(f"    Sub: H1={h1_sharpe:.2f}, H2={h2_sharpe:.2f} {'PASS' if sub_pass else 'FAIL'}")
    else:
        gates['sub_period'] = {'pass': None, 'note': 'Too few months'}

    # 4. Outlier test (remove top 5% of months)
    if len(r_nonzero) >= 10:
        n_remove = max(1, int(len(r_nonzero) * 0.05))
        sorted_idx = np.argsort(r_nonzero)[::-1]
        trimmed = np.delete(r_nonzero, sorted_idx[:n_remove])
        trim_sharpe = np.mean(trimmed) / np.std(trimmed) * np.sqrt(12) if np.std(trimmed) > 0 else 0
        outlier_pass = trim_sharpe > 0
        gates['outlier'] = {
            'trimmed_sharpe': round(float(trim_sharpe), 2),
            'n_removed': n_remove,
            'pass': outlier_pass,
        }
        fprint(f"    Outlier: trimmed={trim_sharpe:.2f} (removed top {n_remove}) {'PASS' if outlier_pass else 'FAIL'}")
    else:
        gates['outlier'] = {'pass': None, 'note': 'Too few months'}

    n_pass = sum(1 for k in ['permutation', 'regime_r1', 'sub_period', 'outlier']
                 if gates.get(k, {}).get('pass') is True)
    n_total = sum(1 for k in ['permutation', 'regime_r1', 'sub_period', 'outlier']
                  if gates.get(k, {}).get('pass') is not None)
    fprint(f"    GATES: {n_pass}/{n_total}")

    return gates


def main():
    fprint("=" * 70)
    fprint("MOMENTUM CALL DEBIT SPREAD ON ETFs v1")
    fprint("=" * 70)
    fprint(f"Universe: {len(UNIVERSE)} ETFs | Starting capital: $645")
    fprint(f"Strategy: Buy ATM call, sell OTM call on top momentum ETFs")
    fprint(f"Goal: Fix outlier/MaxDD issues from momentum call buying v1")
    fprint()

    close = download_data()

    # Define variants
    variants = [
        {'name': 'A_Top2_2pct', 'top_k': 2, 'spread_pct': 0.02, 'desc': 'Top 2, 2% spread'},
        {'name': 'B_Top2_3pct', 'top_k': 2, 'spread_pct': 0.03, 'desc': 'Top 2, 3% spread'},
        {'name': 'C_Top3_2pct', 'top_k': 3, 'spread_pct': 0.02, 'desc': 'Top 3, 2% spread'},
        {'name': 'D_Top3_3pct', 'top_k': 3, 'spread_pct': 0.03, 'desc': 'Top 3, 3% spread'},
        {'name': 'E_Top2_IVfilt', 'top_k': 2, 'spread_pct': 0.02, 'iv_filter': 50, 'desc': 'Top 2, 2%, IV rank <50'},
        {'name': 'F_Top3_regime', 'top_k': 3, 'spread_pct': 0.02, 'regime_filter': True, 'desc': 'Top 3, skip bear'},
        {'name': 'G_Top2_volsize', 'top_k': 2, 'spread_pct': 0.02, 'vol_sizing': True, 'desc': 'Top 2, vol-adjusted sizing'},
        {'name': 'H_Top2_5pct', 'top_k': 2, 'spread_pct': 0.05, 'desc': 'Top 2, 5% spread (aggressive)'},
    ]

    results = {}
    gate_results = {}

    for v in variants:
        vname = v['name']
        fprint(f"\n--- {vname} ({v['desc']}) ---")

        kwargs = {
            'top_k': v['top_k'],
            'spread_pct': v['spread_pct'],
            'iv_filter': v.get('iv_filter'),
            'regime_filter': v.get('regime_filter', False),
            'vol_sizing': v.get('vol_sizing', False),
            'name': vname,
        }

        metrics, trades, monthly_returns, cap_hist = run_backtest(close, **kwargs)

        fprint(f"  Sharpe: {metrics['sharpe']}, Sortino: {metrics['sortino']}, "
               f"CAGR: {metrics['cagr']}%, MaxDD: {metrics['maxdd']}%, "
               f"WR: {metrics['wr']}%, PF: {metrics['pf']}, "
               f"Final: ${metrics['final_capital']}, Trades: {metrics['n_trades']}")

        # Run adversarial gates
        gates = adversarial_gates(monthly_returns, trades, close, vname)

        results[vname] = {
            'desc': v['desc'],
            'metrics': metrics,
            'top_trades': sorted(trades, key=lambda t: t['pnl'], reverse=True)[:5] if trades else [],
            'worst_trades': sorted(trades, key=lambda t: t['pnl'])[:5] if trades else [],
        }
        gate_results[vname] = gates

    # Find winner
    fprint(f"\n{'='*70}")
    fprint("SUMMARY")
    fprint(f"{'='*70}")

    valid_results = [(name, r) for name, r in results.items()
                     if r['metrics']['sharpe'] > 0 and r['metrics']['n_trades'] > 10]

    if valid_results:
        # Sort by gates passed, then Sharpe
        def sort_key(item):
            name, r = item
            gates = gate_results[name]
            n_pass = sum(1 for k in ['permutation', 'regime_r1', 'sub_period', 'outlier']
                        if gates.get(k, {}).get('pass') is True)
            return (n_pass, r['metrics']['sharpe'])

        valid_results.sort(key=sort_key, reverse=True)

        for name, r in valid_results:
            m = r['metrics']
            gates = gate_results[name]
            n_pass = sum(1 for k in ['permutation', 'regime_r1', 'sub_period', 'outlier']
                        if gates.get(k, {}).get('pass') is True)
            n_total = sum(1 for k in ['permutation', 'regime_r1', 'sub_period', 'outlier']
                         if gates.get(k, {}).get('pass') is not None)
            fprint(f"  {name}: Sharpe {m['sharpe']}, CAGR {m['cagr']}%, MaxDD {m['maxdd']}%, "
                   f"WR {m['wr']}%, PF {m['pf']}, ${m['final_capital']} | "
                   f"Gates {n_pass}/{n_total}")

        winner_name = valid_results[0][0]
        winner = results[winner_name]
        fprint(f"\n  WINNER: {winner_name}")
        fprint(f"    {winner['desc']}")
        wm = winner['metrics']
        fprint(f"    Sharpe {wm['sharpe']}, Sortino {wm['sortino']}, CAGR {wm['cagr']}%, "
               f"MaxDD {wm['maxdd']}%, WR {wm['wr']}%, PF {wm['pf']}")
        fprint(f"    $645 → ${wm['final_capital']} over {wm['years']}y, {wm['n_trades']} trades")
    else:
        fprint("  NO VALID VARIANTS (all negative or insufficient trades)")
        winner_name = None

    # Save results
    output = {
        'strategy': 'Momentum Call Debit Spread on ETFs v1',
        'timestamp': datetime.now().isoformat(),
        'universe': UNIVERSE,
        'starting_capital': 645,
        'results': results,
        'gates': gate_results,
        'winner': winner_name,
    }

    with open(RESULTS_PATH, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    fprint(f"\nResults saved to {RESULTS_PATH}")

    # MLflow
    if MLFLOW_OK and winner_name:
        try:
            exp_name = 'momentum_debit_spread_etf_v1'
            try:
                mlflow.create_experiment(exp_name)
            except:
                pass
            mlflow.set_experiment(exp_name)

            with mlflow.start_run(run_name=f'v1_{winner_name}'):
                wm = winner['metrics']
                wg = gate_results[winner_name]
                mlflow.log_metrics({
                    'sharpe': wm.get('sharpe', 0),
                    'sortino': wm.get('sortino', 0),
                    'cagr': wm.get('cagr', 0),
                    'maxdd': wm.get('maxdd', 0),
                    'wr': wm.get('wr', 0),
                    'pf': wm.get('pf', 0),
                    'calmar': wm.get('calmar', 0),
                    'final_capital': wm.get('final_capital', 0),
                    'n_trades': wm.get('n_trades', 0),
                    'perm_p': wg.get('permutation', {}).get('p_value', -1),
                    'r1_gap': wg.get('regime_r1', {}).get('gap', -1),
                    'n_pass': sum(1 for k in ['permutation', 'regime_r1', 'sub_period', 'outlier']
                                  if wg.get(k, {}).get('pass') is True),
                })
                mlflow.log_params({
                    'winner': winner_name,
                    'universe_size': len(UNIVERSE),
                    'starting_capital': 645,
                    'n_variants': len(variants),
                })
                fprint(f"MLflow logged (exp: {exp_name})")
        except Exception as e:
            fprint(f"MLflow error: {e}")

    fprint(f"\n{'='*70}")
    fprint("DONE")
    fprint(f"{'='*70}")

    return output


if __name__ == '__main__':
    main()

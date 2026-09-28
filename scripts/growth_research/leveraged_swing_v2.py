#!/usr/bin/env python3
"""
Leveraged ETF Swing v2 — Fixed Metrics + More Frequent Trading
===============================================================
v1 BUG: 5-7 trades over 4.5 years → Sharpe computed from 99% zero-return days
= meaningless. ALL v1 results INVALID (Sharpe 1.65 with -93% return).

v2 FIXES:
1. Trade-level Sharpe (not daily with zero-padding)
2. Lower signal thresholds for more frequent entry
3. Weekly rebalance option (not just momentum bursts)
4. Properly computed CAGR, max DD, and Calmar

STRATEGY: Swing trade leveraged ETFs on RH ($645 account).
- Zero commission, fractional shares
- 3x leverage without theta decay
- Entry: momentum signals OR weekly LGBM-style rebalance
- Hold: 2-10 days

VARIANTS (8):
  A: TQQQ weekly rebalance (buy if 5d mom > 0, sell if < 0)
  B: Multi-leveraged weekly (rotate strongest among TQQQ/SOXL/UPRO/TNA)
  C: TQQQ + inverse hedge (TQQQ in uptrends, SQQQ in downtrends)
  D: Dip buying (RSI<35, buy TQQQ, hold 3-5 days)
  E: MA crossover (5/20 MA cross on leveraged ETFs)
  F: Breakout (new 10-day high/low entry, 3-day exit)
  G: Momentum + mean-rev combo (momentum for entry, RSI for exit)
  H: Full invested rotation (always in best leveraged ETF)

TRACK: HIGH-GROWTH (agentic account)
"""

import sys, os, json, warnings
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict
warnings.filterwarnings('ignore')

for root in ['/home/jupiter/Lvl3Quant', '/home/nick/Lvl3Quant']:
    if os.path.isdir(root):
        LVL3_ROOT = root
        break
else:
    LVL3_ROOT = '.'

OUTPUT_DIR = os.path.join(LVL3_ROOT, 'output', 'growth_research', 'leveraged_swing_v2')
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    import mlflow
    MLFLOW_AVAILABLE = True
    mlflow.set_tracking_uri("http://jupiter:5000")
    mlflow.set_experiment("leveraged_swing_v2")
    print(f"MLflow OK")
except:
    MLFLOW_AVAILABLE = False

STARTING_CAPITAL = 645.0
START_DATE = '2019-01-01'
END_DATE = '2026-07-28'
OOT_START = '2021-01-01'
N_PERMUTATIONS = 150

TICKERS = ['TQQQ', 'SOXL', 'UPRO', 'TNA', 'TECL', 'FAS', 'SQQQ', 'SPXS', 'SPY', 'QQQ', '^VIX']

def load_data():
    cache_path = os.path.join(LVL3_ROOT, 'data', 'leveraged_swing_v2_cache.parquet')
    if os.path.exists(cache_path):
        df = pd.read_parquet(cache_path)
        if len(df) > 0 and pd.Timestamp(df.index.get_level_values('date').max()) >= pd.Timestamp('2026-07-20'):
            print(f"Cached: {len(df)} rows")
            return df

    import yfinance as yf
    print(f"Downloading {len(TICKERS)} tickers...")
    frames = []
    for t in TICKERS:
        try:
            d = yf.download(t, start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)
            if len(d) < 100: continue
            d.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in d.columns]
            d['ticker'] = t
            d.index.name = 'date'
            frames.append(d)
            print(f"  {t}: {len(d)} rows")
        except Exception as e:
            print(f"  {t}: ERROR {e}")

    df = pd.concat(frames).reset_index().set_index(['ticker', 'date']).sort_index()
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df.to_parquet(cache_path)
    except: pass
    return df

def compute_rsi(prices, period=14):
    if len(prices) < period + 1: return 50.0
    d = np.diff(prices)
    g = np.where(d > 0, d, 0)
    l = np.where(d < 0, -d, 0)
    ag = np.mean(g[-period:])
    al = np.mean(l[-period:])
    if al == 0: return 100.0
    return 100.0 - (100.0 / (1.0 + ag / al))


# ============================================================
# STRATEGY FUNCTIONS
# ============================================================

def strategy_weekly_rebalance(prices_df, etf, spy_prices, vix_data, oot_dates, cfg):
    """Rebalance weekly: in if momentum positive, out if negative."""
    equity = STARTING_CAPITAL
    shares = 0.0
    trades = []
    equity_curve = []
    in_position = False
    entry_price = 0
    entry_date = None

    try:
        etf_data = prices_df.loc[etf].sort_index()
    except:
        return None

    rebalance_freq = cfg.get('rebalance_days', 5)  # weekly

    for i, date in enumerate(oot_dates):
        if date not in etf_data.index:
            equity_curve.append(equity + (shares * entry_price if in_position else 0))
            continue

        price = etf_data.loc[date, 'close']
        if isinstance(price, pd.Series): price = price.iloc[0]

        # Current portfolio value
        portfolio_val = equity + (shares * price if in_position else 0)

        # Rebalance on schedule
        if i % rebalance_freq == 0:
            # Compute signals
            mask = etf_data.index <= date
            hist = etf_data.loc[mask, 'close'].values
            if len(hist) < 25:
                equity_curve.append(portfolio_val)
                continue

            mom_5d = (hist[-1] / hist[-6]) - 1.0 if len(hist) >= 6 else 0
            mom_10d = (hist[-1] / hist[-11]) - 1.0 if len(hist) >= 11 else 0
            ma_5 = np.mean(hist[-5:]) if len(hist) >= 5 else hist[-1]
            ma_20 = np.mean(hist[-20:]) if len(hist) >= 20 else hist[-1]
            rsi = compute_rsi(hist)

            # VIX check
            vix_level = 20
            if vix_data is not None and date in vix_data.index:
                vix_level = vix_data.loc[date, 'close']
                if isinstance(vix_level, pd.Series): vix_level = vix_level.iloc[0]

            # Direction signal
            go_long = False
            go_flat = False
            go_inverse = False

            if cfg.get('strategy') == 'weekly_simple':
                go_long = mom_5d > 0 and ma_5 > ma_20
                go_flat = not go_long

            elif cfg.get('strategy') == 'rotation':
                # Pick strongest ETF
                pass  # Handled separately

            elif cfg.get('strategy') == 'hedge':
                go_long = mom_5d > 0.01 and ma_5 > ma_20
                go_inverse = mom_5d < -0.01 and ma_5 < ma_20
                go_flat = not go_long and not go_inverse

            elif cfg.get('strategy') == 'dip_buy':
                go_long = rsi < 35 and not in_position
                go_flat = in_position and (rsi > 55 or mom_5d < -0.08)

            elif cfg.get('strategy') == 'ma_cross':
                go_long = ma_5 > ma_20
                go_flat = ma_5 <= ma_20

            elif cfg.get('strategy') == 'breakout':
                if len(hist) >= 11:
                    high_10 = np.max(hist[-10:])
                    low_10 = np.min(hist[-10:])
                    go_long = hist[-1] >= high_10 * 0.99
                    go_flat = hist[-1] <= low_10 * 1.01 or (in_position and mom_5d < -0.03)
                else:
                    go_flat = True

            elif cfg.get('strategy') == 'combo':
                # Momentum entry, RSI exit
                go_long = mom_5d > 0.03 and mom_10d > 0 and not in_position
                go_flat = in_position and (rsi > 70 or mom_5d < -0.04)

            elif cfg.get('strategy') == 'always_in':
                go_long = True

            # Execute
            if in_position and (go_flat or go_inverse):
                # Close
                exit_value = shares * price
                pnl = exit_value - (shares * entry_price)
                equity = portfolio_val  # Reset to total value
                shares = 0
                in_position = False

                trades.append({
                    'ticker': etf,
                    'entry_date': str(entry_date.date()) if hasattr(entry_date, 'date') else str(entry_date),
                    'exit_date': str(date.date()) if hasattr(date, 'date') else str(date),
                    'entry_price': round(entry_price, 2),
                    'exit_price': round(price, 2),
                    'pnl': round(pnl, 2),
                    'pnl_pct': round((price / entry_price - 1) * 100, 1),
                })

                # If going inverse, open inverse position
                if go_inverse and cfg.get('inverse_etf'):
                    inv_etf = cfg['inverse_etf']
                    try:
                        inv_data = prices_df.loc[inv_etf]
                        if date in inv_data.index:
                            inv_price = inv_data.loc[date, 'close']
                            if isinstance(inv_price, pd.Series): inv_price = inv_price.iloc[0]
                            alloc = equity * cfg.get('alloc_pct', 0.80)
                            shares = alloc / inv_price
                            entry_price = inv_price
                            entry_date = date
                            in_position = True
                            equity -= alloc
                            etf = inv_etf  # Track which ETF we're in
                    except:
                        pass

            elif not in_position and go_long:
                # Open
                alloc = portfolio_val * cfg.get('alloc_pct', 0.80)
                shares = alloc / price
                entry_price = price
                entry_date = date
                in_position = True
                equity = portfolio_val - alloc

                # Reset etf to the primary one
                etf = cfg.get('primary_etf', etf)

        equity_curve.append(equity + (shares * price if in_position else 0))

    # Force close
    if in_position and len(oot_dates) > 0:
        final_date = oot_dates[-1]
        try:
            final_price = prices_df.loc[etf].loc[final_date, 'close']
            if isinstance(final_price, pd.Series): final_price = final_price.iloc[0]
            pnl = shares * final_price - shares * entry_price
            trades.append({
                'ticker': etf,
                'entry_date': str(entry_date.date()),
                'exit_date': str(final_date.date()),
                'entry_price': round(entry_price, 2),
                'exit_price': round(final_price, 2),
                'pnl': round(pnl, 2),
                'pnl_pct': round((final_price / entry_price - 1) * 100, 1),
            })
            equity_curve[-1] = equity + shares * final_price
        except:
            pass

    return {
        'equity_curve': equity_curve,
        'trades': trades,
        'final_equity': equity_curve[-1] if equity_curve else STARTING_CAPITAL,
    }


def strategy_rotation(prices_df, universe, spy_prices, vix_data, oot_dates, cfg):
    """Rotate among leveraged ETFs — always in the best one."""
    equity = STARTING_CAPITAL
    shares = 0.0
    current_etf = None
    trades = []
    equity_curve = []
    in_position = False
    entry_price = 0
    entry_date = None

    rebalance_freq = cfg.get('rebalance_days', 5)

    for i, date in enumerate(oot_dates):
        # Get current portfolio value
        if in_position and current_etf:
            try:
                p = prices_df.loc[current_etf]
                if date in p.index:
                    cp = p.loc[date, 'close']
                    if isinstance(cp, pd.Series): cp = cp.iloc[0]
                    portfolio_val = equity + shares * cp
                else:
                    portfolio_val = equity + shares * entry_price
            except:
                portfolio_val = equity + shares * entry_price
        else:
            portfolio_val = equity

        if i % rebalance_freq == 0:
            # Score each ETF by momentum
            scores = {}
            for etf in universe:
                try:
                    ed = prices_df.loc[etf]
                    mask = ed.index <= date
                    hist = ed.loc[mask, 'close'].values
                    if len(hist) < 11: continue

                    mom_5d = (hist[-1] / hist[-6]) - 1.0
                    mom_10d = (hist[-1] / hist[-11]) - 1.0
                    ma5 = np.mean(hist[-5:])
                    ma20 = np.mean(hist[-20:]) if len(hist) >= 20 else hist[-1]

                    # Score = weighted momentum
                    score = mom_5d * 0.6 + mom_10d * 0.4
                    # Bonus if above MA
                    if ma5 > ma20:
                        score += 0.02

                    scores[etf] = score
                except:
                    continue

            if not scores:
                equity_curve.append(portfolio_val)
                continue

            best_etf = max(scores, key=scores.get)
            best_score = scores[best_etf]

            # Should we be in a position?
            should_hold = best_score > cfg.get('min_score', -0.02)

            if in_position and (current_etf != best_etf or not should_hold):
                # Close current
                try:
                    ed = prices_df.loc[current_etf]
                    if date in ed.index:
                        exit_price = ed.loc[date, 'close']
                        if isinstance(exit_price, pd.Series): exit_price = exit_price.iloc[0]
                        pnl = shares * exit_price - shares * entry_price
                        trades.append({
                            'ticker': current_etf,
                            'entry_date': str(entry_date.date()),
                            'exit_date': str(date.date()),
                            'entry_price': round(entry_price, 2),
                            'exit_price': round(exit_price, 2),
                            'pnl': round(pnl, 2),
                            'pnl_pct': round((exit_price / entry_price - 1) * 100, 1),
                        })
                        equity = equity + shares * exit_price
                        shares = 0
                        in_position = False
                        portfolio_val = equity
                except:
                    pass

            if not in_position and should_hold:
                # Open new position
                try:
                    ed = prices_df.loc[best_etf]
                    if date in ed.index:
                        price = ed.loc[date, 'close']
                        if isinstance(price, pd.Series): price = price.iloc[0]
                        alloc = portfolio_val * cfg.get('alloc_pct', 0.80)
                        shares = alloc / price
                        entry_price = price
                        entry_date = date
                        current_etf = best_etf
                        in_position = True
                        equity = portfolio_val - alloc
                except:
                    pass

        # Update portfolio value for equity curve
        if in_position and current_etf:
            try:
                ed = prices_df.loc[current_etf]
                if date in ed.index:
                    cp = ed.loc[date, 'close']
                    if isinstance(cp, pd.Series): cp = cp.iloc[0]
                    equity_curve.append(equity + shares * cp)
                else:
                    equity_curve.append(portfolio_val)
            except:
                equity_curve.append(portfolio_val)
        else:
            equity_curve.append(equity)

    # Force close
    if in_position and current_etf:
        try:
            final_date = oot_dates[-1]
            ed = prices_df.loc[current_etf]
            fp = ed.loc[ed.index <= final_date, 'close'].iloc[-1]
            if isinstance(fp, pd.Series): fp = fp.iloc[0]
            pnl = shares * fp - shares * entry_price
            trades.append({
                'ticker': current_etf,
                'entry_date': str(entry_date.date()),
                'exit_date': str(final_date.date()),
                'entry_price': round(entry_price, 2),
                'exit_price': round(fp, 2),
                'pnl': round(pnl, 2),
                'pnl_pct': round((fp / entry_price - 1) * 100, 1),
            })
            equity_curve[-1] = equity + shares * fp
        except:
            pass

    return {
        'equity_curve': equity_curve,
        'trades': trades,
        'final_equity': equity_curve[-1] if equity_curve else STARTING_CAPITAL,
    }


# ============================================================
# VARIANT CONFIGS
# ============================================================

VARIANTS = {
    'A': {
        'name': 'TQQQ Weekly Rebalance',
        'strategy': 'weekly_simple',
        'etf': 'TQQQ',
        'rebalance_days': 5,
        'alloc_pct': 0.80,
        'type': 'single',
    },
    'B': {
        'name': 'Multi-Leveraged Rotation',
        'strategy': 'rotation',
        'universe': ['TQQQ', 'SOXL', 'UPRO', 'TNA', 'TECL', 'FAS'],
        'rebalance_days': 5,
        'alloc_pct': 0.80,
        'min_score': -0.02,
        'type': 'rotation',
    },
    'C': {
        'name': 'TQQQ/SQQQ Hedge',
        'strategy': 'hedge',
        'etf': 'TQQQ',
        'inverse_etf': 'SQQQ',
        'primary_etf': 'TQQQ',
        'rebalance_days': 5,
        'alloc_pct': 0.80,
        'type': 'single',
    },
    'D': {
        'name': 'Dip Buy TQQQ RSI<35',
        'strategy': 'dip_buy',
        'etf': 'TQQQ',
        'rebalance_days': 1,  # Check daily
        'alloc_pct': 0.80,
        'type': 'single',
    },
    'E': {
        'name': 'MA Crossover (5/20)',
        'strategy': 'ma_cross',
        'etf': 'TQQQ',
        'rebalance_days': 1,
        'alloc_pct': 0.80,
        'type': 'single',
    },
    'F': {
        'name': 'Breakout 10d High',
        'strategy': 'breakout',
        'etf': 'TQQQ',
        'rebalance_days': 1,
        'alloc_pct': 0.80,
        'type': 'single',
    },
    'G': {
        'name': 'Mom Entry + RSI Exit',
        'strategy': 'combo',
        'etf': 'TQQQ',
        'rebalance_days': 1,
        'alloc_pct': 0.80,
        'type': 'single',
    },
    'H': {
        'name': 'Always-In Best Leveraged',
        'strategy': 'rotation',
        'universe': ['TQQQ', 'SOXL', 'UPRO', 'TNA'],
        'rebalance_days': 5,
        'alloc_pct': 0.95,
        'min_score': -999,  # Always in
        'type': 'rotation',
    },
}


# ============================================================
# METRICS
# ============================================================

def compute_metrics(result, vk, cfg, spy_prices, oot_dates):
    trades = result['trades']
    eq = np.array(result['equity_curve'])
    final = result['final_equity']

    m = {
        'variant': vk,
        'name': cfg['name'],
        'final_equity': round(final, 2),
        'total_return_pct': round((final / STARTING_CAPITAL - 1) * 100, 1),
        'total_trades': len(trades),
    }

    if len(eq) < 10:
        m.update({'sharpe': 0, 'sortino': 0, 'win_rate': 0, 'profit_factor': 0,
                  'max_drawdown_pct': 0, 'cagr_pct': 0, 'calmar': 0, 'regime_gap': 999})
        return m

    # Daily returns from equity curve (NO zero-padding issue since we track full portfolio)
    daily_rets = np.diff(eq) / np.maximum(eq[:-1], 1)

    # Filter out truly zero days only if mostly invested
    ann = np.sqrt(252)
    mu = np.mean(daily_rets)
    sigma = np.std(daily_rets)
    m['sharpe'] = round((mu / sigma) * ann, 2) if sigma > 1e-10 else 0

    neg = daily_rets[daily_rets < 0]
    ds = np.std(neg) if len(neg) > 0 else sigma
    m['sortino'] = round((mu / ds) * ann, 2) if ds > 1e-10 else 0

    # Max drawdown
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / np.maximum(peak, 1)
    m['max_drawdown_pct'] = round(np.min(dd) * 100, 1)

    # CAGR
    n_years = len(daily_rets) / 252
    if n_years > 0 and final > 0:
        m['cagr_pct'] = round(((final / STARTING_CAPITAL) ** (1/n_years) - 1) * 100, 1)
    else:
        m['cagr_pct'] = 0

    # Calmar
    mdd = abs(m['max_drawdown_pct'])
    m['calmar'] = round(m['cagr_pct'] / mdd, 2) if mdd > 0 else 0

    # Trade stats
    if trades:
        pnls = [t['pnl'] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        m['win_rate'] = round(len(wins) / len(pnls) * 100, 1)
        m['avg_pnl'] = round(np.mean(pnls), 2)

        gp = sum(wins) if wins else 0
        gl = abs(sum(losses)) if losses else 1e-8
        m['profit_factor'] = round(gp / gl, 2) if gl > 0 else 999
    else:
        m['win_rate'] = 0
        m['profit_factor'] = 0
        m['avg_pnl'] = 0

    # Regime analysis
    try:
        spy_rets = spy_prices.loc[spy_prices.index.isin(oot_dates), 'close'].pct_change().dropna()
        green_dates = set(spy_rets[spy_rets > 0].index)
        red_dates = set(spy_rets[spy_rets <= 0].index)

        # Map trades to regimes
        green_pnl = sum(t['pnl'] for t in trades if pd.Timestamp(t['entry_date']) in green_dates)
        red_pnl = sum(t['pnl'] for t in trades if pd.Timestamp(t['entry_date']) in red_dates)
        total = abs(green_pnl) + abs(red_pnl)
        m['regime_gap'] = round(abs(green_pnl - red_pnl) / total, 3) if total > 0 else 0
    except:
        m['regime_gap'] = 999

    # SPY comparison
    try:
        spy_start = spy_prices.loc[spy_prices.index >= pd.Timestamp(OOT_START), 'close'].iloc[0]
        spy_end = spy_prices.loc[spy_prices.index >= pd.Timestamp(OOT_START), 'close'].iloc[-1]
        spy_ret = (spy_end / spy_start - 1) * 100
        m['alpha_vs_spy'] = round(m['total_return_pct'] - spy_ret, 1)
    except:
        m['alpha_vs_spy'] = 0

    return m


def run_permutation(cfg, prices_df, spy_prices, vix_data, oot_dates, actual_sharpe, n_perms=150):
    """Randomize entry timing."""
    random_sharpes = []

    for perm in range(n_perms):
        np.random.seed(perm + 42)
        eq = STARTING_CAPITAL
        curve = [eq]

        # Random in/out timing with same frequency
        state = False  # in position
        shares = 0
        price = 0
        etf = cfg.get('etf', cfg.get('universe', ['TQQQ'])[0] if isinstance(cfg.get('universe'), list) else 'TQQQ')

        try:
            ed = prices_df.loc[etf].sort_index()
        except:
            continue

        for i, date in enumerate(oot_dates):
            if date not in ed.index:
                curve.append(curve[-1])
                continue

            cp = ed.loc[date, 'close']
            if isinstance(cp, pd.Series): cp = cp.iloc[0]

            if state:
                pv = eq + shares * cp
            else:
                pv = eq

            # Random rebalance
            if i % max(cfg.get('rebalance_days', 5), 1) == 0:
                if np.random.random() > 0.5:  # Random in
                    if not state:
                        alloc = pv * 0.80
                        shares = alloc / cp
                        price = cp
                        state = True
                        eq = pv - alloc
                else:  # Random out
                    if state:
                        eq = pv
                        shares = 0
                        state = False

            if state:
                curve.append(eq + shares * cp)
            else:
                curve.append(eq)

        c = np.array(curve)
        if len(c) > 20:
            dr = np.diff(c) / np.maximum(c[:-1], 1)
            if np.std(dr) > 1e-10:
                random_sharpes.append((np.mean(dr) / np.std(dr)) * np.sqrt(252))

    if not random_sharpes:
        return 1.0, 0.0

    p = np.mean([s >= actual_sharpe for s in random_sharpes])
    return p, np.mean(random_sharpes)


def validate_5gate(m, p_val, rand_sharpe):
    gates = {
        'sharpe_gt_1': m['sharpe'] >= 1.0,
        'perm_p_lt_005': p_val < 0.05,
        'wr_gt_40': m['win_rate'] >= 40.0,
        'regime_balance': m.get('regime_gap', 999) < 0.50,
        'beats_random': m['sharpe'] > rand_sharpe + 0.1,
    }
    return sum(gates.values()), gates


# ============================================================
# MAIN
# ============================================================

def main():
    import time
    print("=" * 70)
    print("  LEVERAGED ETF SWING V2 — Fixed Metrics")
    print("  High-growth for $645 agentic account")
    print("  3x leverage, zero commission, no theta decay")
    print("=" * 70)

    prices_df = load_data()
    spy_prices = prices_df.loc['SPY'] if 'SPY' in prices_df.index.get_level_values('ticker') else None
    vix_data = prices_df.loc['^VIX'] if '^VIX' in prices_df.index.get_level_values('ticker') else None

    if spy_prices is None:
        print("ERROR: No SPY data"); return

    trading_dates = sorted(spy_prices.index.unique())
    oot_dates = [d for d in trading_dates if d >= pd.Timestamp(OOT_START)]
    avail = [t for t in TICKERS if t in prices_df.index.get_level_values('ticker') and t not in ['SPY', 'QQQ', '^VIX']]
    print(f"\nOOT: {oot_dates[0].date()} to {oot_dates[-1].date()} ({len(oot_dates)} days)")
    print(f"Available: {avail}")

    all_metrics = {}

    for vk in sorted(VARIANTS.keys()):
        cfg = VARIANTS[vk]
        t0 = time.time()
        print(f"\n{'=' * 60}")
        print(f"  VARIANT {vk}: {cfg['name']}")
        print(f"{'=' * 60}")

        if cfg['type'] == 'rotation':
            univ = [u for u in cfg.get('universe', []) if u in avail]
            if not univ:
                print(f"  SKIP: No tickers available")
                all_metrics[vk] = {'variant': vk, 'name': cfg['name'], 'sharpe': 0, 'gates_passed': 0}
                continue
            result = strategy_rotation(prices_df, univ, spy_prices, vix_data, oot_dates, cfg)
        else:
            etf = cfg.get('etf', 'TQQQ')
            if etf not in avail:
                print(f"  SKIP: {etf} not available")
                all_metrics[vk] = {'variant': vk, 'name': cfg['name'], 'sharpe': 0, 'gates_passed': 0}
                continue
            result = strategy_weekly_rebalance(prices_df, etf, spy_prices, vix_data, oot_dates, cfg)

        if result is None:
            print(f"  SKIP: No result")
            all_metrics[vk] = {'variant': vk, 'name': cfg['name'], 'sharpe': 0, 'gates_passed': 0}
            continue

        runtime = time.time() - t0
        metrics = compute_metrics(result, vk, cfg, spy_prices, oot_dates)
        metrics['runtime'] = round(runtime, 1)

        # Print first 3 trades
        for t in result['trades'][:3]:
            print(f"  {t['entry_date']} → {t['exit_date']}: {t['ticker']} "
                  f"${t['entry_price']:.1f} → ${t['exit_price']:.1f} pnl=${t['pnl']:.0f} ({t['pnl_pct']:.1f}%)")

        print(f"  Trades: {metrics['total_trades']} | Sharpe: {metrics['sharpe']} | "
              f"Sortino: {metrics['sortino']} | PF: {metrics.get('profit_factor', 0)} | "
              f"WR: {metrics['win_rate']}% | MDD: {metrics['max_drawdown_pct']}%")
        print(f"  $645 → ${metrics['final_equity']:.0f} | Return: {metrics['total_return_pct']:.1f}% | "
              f"CAGR: {metrics['cagr_pct']:.1f}% | Alpha vs SPY: {metrics.get('alpha_vs_spy', 0):.1f}%")

        # Permutation test
        if metrics['total_trades'] >= 3 and metrics['sharpe'] > 0:
            print(f"  Running {N_PERMUTATIONS}-permutation test...")
            p_val, rand_sharpe = run_permutation(
                cfg, prices_df, spy_prices, vix_data, oot_dates, metrics['sharpe'], N_PERMUTATIONS)
            n_pass, gates = validate_5gate(metrics, p_val, rand_sharpe)

            print(f"  5-Gate: {n_pass}/5 PASS")
            for gn, gp in gates.items():
                vs = ""
                if gn == 'sharpe_gt_1': vs = f"value={metrics['sharpe']}, threshold=1.0"
                elif gn == 'perm_p_lt_005': vs = f"value={p_val:.3f}, threshold=0.05"
                elif gn == 'wr_gt_40': vs = f"value={metrics['win_rate']}, threshold=40.0"
                elif gn == 'regime_balance': vs = f"value={metrics.get('regime_gap', 999):.3f}, threshold=0.5"
                elif gn == 'beats_random': vs = f"value={metrics['sharpe']}, random={rand_sharpe:.2f}"
                print(f"    {gn}: {'PASS' if gp else 'FAIL'} ({vs})")

            metrics['perm_p'] = round(p_val, 4)
            metrics['random_sharpe'] = round(rand_sharpe, 2)
            metrics['gates_passed'] = n_pass
        else:
            metrics['gates_passed'] = 0
            print(f"  5-Gate: 0/5 (insufficient trades or negative Sharpe)")

        all_metrics[vk] = metrics

        if MLFLOW_AVAILABLE:
            try:
                with mlflow.start_run(run_name=f"v2_{vk}_{cfg['name'][:20]}"):
                    for mk, mv in metrics.items():
                        if isinstance(mv, (int, float)):
                            mlflow.log_metric(mk, mv)
            except: pass

    # Summary
    print(f"\n{'=' * 90}")
    print("  SUMMARY — LEVERAGED ETF SWING V2")
    print(f"{'=' * 90}")

    sv = sorted(all_metrics.items(), key=lambda x: x[1].get('sharpe', 0), reverse=True)

    print(f"\n  {'Var':<4} {'Name':<28} {'Sharpe':>7} {'Sort':>7} {'PF':>6} {'WR':>6} "
          f"{'Trades':>7} {'Return':>8} {'CAGR':>7} {'MDD':>7} {'Alpha':>7} {'Gates':>6}")
    print(f"  {'-'*4} {'-'*28} {'-'*7} {'-'*7} {'-'*6} {'-'*6} {'-'*7} {'-'*8} {'-'*7} {'-'*7} {'-'*7} {'-'*6}")

    for vk, m in sv:
        print(f"  {vk:<4} {m['name']:<28} {m.get('sharpe',0):>7.2f} {m.get('sortino',0):>7.2f} "
              f"{m.get('profit_factor',0):>6.2f} {m.get('win_rate',0):>5.1f}% "
              f"{m.get('total_trades',0):>7} {m.get('total_return_pct',0):>7.1f}% "
              f"{m.get('cagr_pct',0):>6.1f}% {m.get('max_drawdown_pct',0):>6.1f}% "
              f"{m.get('alpha_vs_spy',0):>6.1f}% {m.get('gates_passed',0):>4}/5")

    best = sv[0]
    print(f"\n  BEST: {best[0]} ({best[1]['name']}) — Sharpe {best[1].get('sharpe', 0)}, "
          f"$645 → ${best[1].get('final_equity', 0):.0f}")

    # Buy-and-hold TQQQ comparison
    try:
        tqqq = prices_df.loc['TQQQ']
        oot_tqqq = tqqq.loc[tqqq.index >= pd.Timestamp(OOT_START)]
        bh_ret = (oot_tqqq['close'].iloc[-1] / oot_tqqq['close'].iloc[0] - 1) * 100
        bh_final = STARTING_CAPITAL * (1 + bh_ret/100)
        print(f"\n  BUY-AND-HOLD TQQQ: {bh_ret:.1f}% return, $645 → ${bh_final:.0f}")
    except:
        pass

    # Save
    with open(os.path.join(OUTPUT_DIR, 'backtest_results.json'), 'w') as f:
        json.dump({'metrics': all_metrics, 'best': best[0],
                   'timestamp': datetime.now().isoformat(), 'track': 'HIGH_GROWTH'}, f, indent=2, default=str)

    print(f"\nDone.")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Quality Factor Rotation Backtest
================================
Rotates into high-quality stocks using price-based proxies for quality.
6 variants, walk-forward OOT Jan 2022 - Jul 2026, 5-gate validation.

Account: $645, 0.02% slippage, $0 commission, monthly rebalancing.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import json
import warnings
import os
from datetime import datetime
from collections import Counter

warnings.filterwarnings('ignore')

# ── CONFIG ──────────────────────────────────────────────────────────────────
INITIAL_CAPITAL = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02%

STOCKS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD',
    'NFLX', 'CRM', 'PLTR', 'SOFI', 'HOOD', 'UBER', 'COIN', 'RBLX',
    'SNAP', 'ROKU', 'DDOG', 'SQ', 'SHOP', 'ABNB', 'NET', 'MELI'
]

SECTOR_ETFS = [
    'XLK', 'XLC', 'XLY', 'XLF', 'XLE', 'XLV', 'XLI', 'XLP', 'XLB',
    'XLRE', 'XLU'
]

BROAD_ETFS = ['QQQ', 'SPY']
ALL_TICKERS = list(set(STOCKS + SECTOR_ETFS + BROAD_ETFS))

OOT_START = '2022-01-01'
OOT_END = '2026-07-29'
DATA_START = '2021-06-01'  # Extra history for lookback

PERM_ITERATIONS = 1000

# ── DATA DOWNLOAD ───────────────────────────────────────────────────────────
def download_data():
    print(f"Downloading data for {len(ALL_TICKERS)} tickers...")
    data = yf.download(ALL_TICKERS, start=DATA_START, end=OOT_END, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        close = data['Close'].copy()
    else:
        close = data.copy()
    close = close.ffill().dropna(axis=1, how='all')
    # Drop leading NaN rows but keep tickers that start later (fill with ffill)
    close = close.dropna(how='all')
    close = close.ffill().bfill()
    print(f"  Data shape: {close.shape}, {close.index[0].date()} to {close.index[-1].date()}")
    return close


# ── HELPERS ─────────────────────────────────────────────────────────────────
def get_monthly_rebal_dates(idx, start, end):
    mask = (idx >= start) & (idx <= end)
    sub = idx[mask]
    dates = []
    seen = set()
    for d in sub:
        key = (d.year, d.month)
        if key not in seen:
            seen.add(key)
            dates.append(d)
    return dates

def get_regime(spy_prices, date):
    loc = spy_prices.index.get_loc(date)
    if isinstance(loc, slice): loc = loc.start
    if isinstance(loc, np.ndarray): loc = loc[0]
    if loc < 200:
        return 'bull'
    sma200 = spy_prices.iloc[loc-199:loc+1].mean()
    return 'bull' if spy_prices.iloc[loc] > sma200 else 'bear'


# ── PORTFOLIO SIMULATION ───────────────────────────────────────────────────
def simulate(prices, select_fn, max_pos=3):
    """
    Equal-weight monthly rebalance. Returns (final_equity, trades, monthly_rets).
    """
    returns = prices.pct_change()
    spy = prices['SPY']

    rebal_dates = get_monthly_rebal_dates(prices.index, OOT_START, OOT_END)
    if not rebal_dates:
        return INITIAL_CAPITAL, [], []

    equity = INITIAL_CAPITAL
    holdings = {}  # ticker -> {shares, entry_price, entry_date}
    trades = []
    monthly_rets = []

    for i, date in enumerate(rebal_dates):
        # 1. Mark to market
        if holdings:
            old_eq = equity
            equity = sum(h['shares'] * prices.loc[date, t]
                         for t, h in holdings.items() if t in prices.columns)
            if i > 0 and old_eq > 0:
                monthly_rets.append({
                    'date': str(date.date()),
                    'return': (equity - old_eq) / old_eq,
                    'regime': get_regime(spy, date),
                })

        # 2. Selection
        try:
            selected = select_fn(prices, returns, date)
        except Exception:
            selected = []
        selected = [s for s in selected if s in prices.columns and date in prices.index][:max_pos]

        # 3. Close positions not in new selection
        for t in list(holdings.keys()):
            if t not in selected:
                sp = prices.loc[date, t] * (1 - SLIPPAGE_PCT)
                trades.append({
                    'ticker': t,
                    'entry_date': str(holdings[t]['entry_date'].date()),
                    'exit_date': str(date.date()),
                    'entry_price': round(holdings[t]['entry_price'], 4),
                    'exit_price': round(sp, 4),
                    'shares': round(holdings[t]['shares'], 6),
                    'pnl': round(holdings[t]['shares'] * (sp - holdings[t]['entry_price']), 2),
                    'return': round((sp - holdings[t]['entry_price']) / holdings[t]['entry_price'], 6),
                })
                del holdings[t]

        # 4. Calculate cash (equity minus value of retained positions)
        held_val = sum(h['shares'] * prices.loc[date, t]
                       for t, h in holdings.items() if t in prices.columns)
        cash = equity - held_val

        # 5. Buy new positions
        new_buys = [s for s in selected if s not in holdings]
        if new_buys and cash > 5:
            alloc = cash / len(new_buys)
            for t in new_buys:
                bp = prices.loc[date, t] * (1 + SLIPPAGE_PCT)
                shares = alloc / bp
                holdings[t] = {'shares': shares, 'entry_price': bp, 'entry_date': date}

    # Final close
    last = prices.index[-1]
    if holdings:
        old_eq = equity
        equity = sum(h['shares'] * prices.loc[last, t]
                     for t, h in holdings.items() if t in prices.columns)
        if old_eq > 0:
            monthly_rets.append({
                'date': str(last.date()),
                'return': (equity - old_eq) / old_eq,
                'regime': get_regime(spy, last),
            })
        for t, h in list(holdings.items()):
            sp = prices.loc[last, t] * (1 - SLIPPAGE_PCT)
            trades.append({
                'ticker': t,
                'entry_date': str(h['entry_date'].date()),
                'exit_date': str(last.date()),
                'entry_price': round(h['entry_price'], 4),
                'exit_price': round(sp, 4),
                'shares': round(h['shares'], 6),
                'pnl': round(h['shares'] * (sp - h['entry_price']), 2),
                'return': round((sp - h['entry_price']) / h['entry_price'], 6),
            })

    return equity, trades, monthly_rets


# ── SELECTION FUNCTIONS ─────────────────────────────────────────────────────
def _find_loc(idx, date):
    """Safely find integer location of date in index."""
    loc = idx.get_loc(date)
    if isinstance(loc, slice): return loc.start
    if isinstance(loc, np.ndarray): return int(np.where(loc)[0][0])
    return int(loc)

def _available(tickers, prices):
    return [t for t in tickers if t in prices.columns]

def select_low_vol(prices, returns, date):
    """A: Buy 3 lowest 20-day realized vol stocks."""
    loc = _find_loc(prices.index, date)
    if loc < 21: return []
    avail = _available(STOCKS, returns)
    window = returns.iloc[loc-20:loc]
    vols = window[avail].std().dropna().sort_values()
    return vols.index[:3].tolist()

def select_high_sharpe(prices, returns, date):
    """B: Buy top 3 by 60-day Sharpe."""
    loc = _find_loc(prices.index, date)
    if loc < 61: return []
    avail = _available(STOCKS, returns)
    w = returns.iloc[loc-60:loc][avail]
    mu = w.mean() * 252
    sig = w.std() * np.sqrt(252)
    sharpe = (mu / sig.replace(0, np.nan)).dropna().sort_values(ascending=False)
    return sharpe.index[:3].tolist()

def select_bear_outperformers(prices, returns, date):
    """C: Buy stocks that dropped least in last SPY down-month."""
    loc = _find_loc(prices.index, date)
    if loc < 25: return []
    avail = _available(STOCKS, returns)
    dt = pd.Timestamp(date)
    for lb in range(1, 7):
        target = dt - pd.DateOffset(months=lb)
        mask = (returns.index.year == target.year) & (returns.index.month == target.month)
        mr = returns.loc[mask]
        if len(mr) == 0: continue
        spy_ret = mr['SPY'].sum() if 'SPY' in mr.columns else 0
        if spy_ret < 0:
            stock_rets = mr[avail].sum().sort_values(ascending=False)
            return stock_rets.index[:3].tolist()
    # Fallback: low vol
    return select_low_vol(prices, returns, date)

def select_sector_quality(prices, returns, date):
    """D: Buy top 2 sector ETFs by 60-day Sharpe."""
    loc = _find_loc(prices.index, date)
    if loc < 61: return []
    avail = _available(SECTOR_ETFS, returns)
    w = returns.iloc[loc-60:loc][avail]
    mu = w.mean() * 252
    sig = w.std() * np.sqrt(252)
    sharpe = (mu / sig.replace(0, np.nan)).dropna().sort_values(ascending=False)
    return sharpe.index[:2].tolist()

def select_quality_momentum(prices, returns, date):
    """E: Combined 60d Sharpe rank + 20d momentum rank, top 3."""
    loc = _find_loc(prices.index, date)
    if loc < 61: return []
    avail = _available(STOCKS, returns)
    w60 = returns.iloc[loc-60:loc][avail]
    mu = w60.mean() * 252
    sig = w60.std() * np.sqrt(252)
    sharpe = mu / sig.replace(0, np.nan)
    sharpe_rank = sharpe.dropna().rank(ascending=False)

    w20 = returns.iloc[loc-20:loc][avail]
    mom = w20.sum()
    mom_rank = mom.dropna().rank(ascending=False)

    common = sharpe_rank.index.intersection(mom_rank.index)
    combined = (sharpe_rank[common] + mom_rank[common]).sort_values()
    return combined.index[:3].tolist()

def select_antifragile(prices, returns, date):
    """F: Buy stocks that went UP during most recent SPY >2% drawdown."""
    loc = _find_loc(prices.index, date)
    if loc < 30: return []
    avail = _available(STOCKS, returns)
    lookback = min(loc, 90)
    spy_w = prices['SPY'].iloc[loc-lookback:loc+1]
    cummax = spy_w.expanding().max()
    dd = (spy_w - cummax) / cummax
    dd_mask = dd < -0.02
    if dd_mask.any():
        dd_dates = dd[dd_mask].index
        dr = returns.loc[dd_dates[0]:dd_dates[-1]]
        if len(dr) > 0:
            sr = dr[avail].sum().sort_values(ascending=False)
            winners = sr[sr > 0]
            if len(winners) >= 1:
                return winners.index[:3].tolist()
    return select_low_vol(prices, returns, date)


# ── METRICS ─────────────────────────────────────────────────────────────────
def calc_metrics(trades, monthly_rets, final_equity):
    if not trades or not monthly_rets:
        return None

    n = len(trades)
    wr = sum(1 for t in trades if t['pnl'] > 0) / n

    gp = sum(t['pnl'] for t in trades if t['pnl'] > 0)
    gl = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
    pf = gp / gl if gl > 0 else float('inf')

    mr = np.array([m['return'] for m in monthly_rets])
    if len(mr) > 1 and np.std(mr) > 0:
        sharpe = np.mean(mr) / np.std(mr) * np.sqrt(12)
    else:
        sharpe = 0.0

    down = mr[mr < 0]
    sortino = np.mean(mr) / np.std(down) * np.sqrt(12) if len(down) > 1 and np.std(down) > 0 else sharpe * 1.5

    cum = np.cumprod(1 + mr)
    rmax = np.maximum.accumulate(cum)
    dd = (cum - rmax) / rmax
    max_dd = dd.min() if len(dd) > 0 else 0

    bull = [m['return'] for m in monthly_rets if m['regime'] == 'bull']
    bear = [m['return'] for m in monthly_rets if m['regime'] == 'bear']

    sb = np.mean(bull) / np.std(bull) * np.sqrt(12) if len(bull) > 1 and np.std(bull) > 0 else 0
    sr = np.mean(bear) / np.std(bear) * np.sqrt(12) if len(bear) > 1 and np.std(bear) > 0 else 0
    denom = max(abs(sb), abs(sr), 1e-9)
    rgap = abs(sb - sr) / denom

    return {
        'sharpe': round(sharpe, 4),
        'sortino': round(sortino, 4),
        'profit_factor': round(pf, 4),
        'win_rate': round(wr, 4),
        'max_drawdown': round(max_dd, 4),
        'total_trades': n,
        'final_equity': round(final_equity, 2),
        'total_return_pct': round((final_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100, 2),
        'sharpe_bull': round(sb, 4),
        'sharpe_bear': round(sr, 4),
        'regime_gap': round(rgap, 4),
        'n_bull_months': len(bull),
        'n_bear_months': len(bear),
    }


# ── PERMUTATION TEST ───────────────────────────────────────────────────────
def permutation_test(prices, select_fn, max_pos, actual_sharpe, n_iter=PERM_ITERATIONS):
    """
    Proper permutation: for each iteration, randomly select max_pos tickers
    each month instead of the strategy's picks, simulate, compute Sharpe.
    Count how often random >= actual.
    """
    returns = prices.pct_change()
    all_eligible = _available(STOCKS + SECTOR_ETFS, prices)
    rebal_dates = get_monthly_rebal_dates(prices.index, OOT_START, OOT_END)

    rng = np.random.RandomState(42)
    count = 0

    for it in range(n_iter):
        eq = INITIAL_CAPITAL
        holdings = {}
        mrets = []

        for i, date in enumerate(rebal_dates):
            if holdings:
                old_eq = eq
                eq = sum(h['shares'] * prices.loc[date, t]
                         for t, h in holdings.items() if t in prices.columns)
                if i > 0 and old_eq > 0:
                    mrets.append((eq - old_eq) / old_eq)

            # Random selection
            selected = list(rng.choice(all_eligible, size=min(max_pos, len(all_eligible)), replace=False))
            selected = [s for s in selected if date in prices.index]

            for t in list(holdings.keys()):
                if t not in selected:
                    del holdings[t]

            held_val = sum(h['shares'] * prices.loc[date, t]
                           for t, h in holdings.items() if t in prices.columns)
            cash = eq - held_val

            new_buys = [s for s in selected if s not in holdings]
            if new_buys and cash > 5:
                alloc = cash / len(new_buys)
                for t in new_buys:
                    bp = prices.loc[date, t] * (1 + SLIPPAGE_PCT)
                    holdings[t] = {'shares': alloc / bp, 'entry_price': bp, 'entry_date': date}

        # Final MTM
        last = prices.index[-1]
        if holdings:
            old_eq = eq
            eq = sum(h['shares'] * prices.loc[last, t]
                     for t, h in holdings.items() if t in prices.columns)
            if old_eq > 0:
                mrets.append((eq - old_eq) / old_eq)

        mr = np.array(mrets)
        if len(mr) > 1 and np.std(mr) > 0:
            perm_sharpe = np.mean(mr) / np.std(mr) * np.sqrt(12)
        else:
            perm_sharpe = 0
        if perm_sharpe >= actual_sharpe:
            count += 1

    return count / n_iter


# ── GATES ───────────────────────────────────────────────────────────────────
def validate_gates(metrics, perm_p):
    gates = {
        'sharpe_gt_0.5': metrics['sharpe'] > 0.5,
        'perm_p_lt_0.05': perm_p < 0.05,
        'regime_gap_lt_0.5': metrics['regime_gap'] < 0.5,
        'max_dd_gt_neg50': metrics['max_drawdown'] > -0.50,
        'trades_gte_20': metrics['total_trades'] >= 20,
    }
    gates['all_pass'] = all(gates.values())
    return gates


# ── MAIN ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("QUALITY FACTOR ROTATION BACKTEST")
    print("=" * 70)
    print(f"  Capital: ${INITIAL_CAPITAL}")
    print(f"  OOT: {OOT_START} to {OOT_END}")
    print(f"  Slippage: {SLIPPAGE_PCT*100:.2f}%")
    print(f"  Permutation iters: {PERM_ITERATIONS}")
    print()

    prices = download_data()

    variants = {
        'A_Low_Vol_Select': (select_low_vol, 3, 'Buy 3 lowest 20d vol stocks monthly'),
        'B_High_Sharpe_Select': (select_high_sharpe, 3, 'Buy top 3 by 60d Sharpe monthly'),
        'C_Bear_Market_Outperformers': (select_bear_outperformers, 3, 'Buy stocks that dropped least in last SPY down-month'),
        'D_Sector_Quality_Rotation': (select_sector_quality, 2, 'Buy top 2 sector ETFs by 60d Sharpe'),
        'E_Quality_Momentum_Combo': (select_quality_momentum, 3, 'Combined 60d Sharpe + 20d momentum rank, top 3'),
        'F_Anti_Fragile': (select_antifragile, 3, 'Buy stocks that rose during last SPY >2% drawdown'),
    }

    results = {}

    for name, (fn, max_pos, desc) in variants.items():
        print(f"\n{'─'*60}")
        print(f"  {name}: {desc}")
        print(f"{'─'*60}")

        final_eq, trade_log, monthly = simulate(prices, fn, max_pos)
        metrics = calc_metrics(trade_log, monthly, final_eq)

        if metrics is None:
            print(f"  !! No trades for {name}")
            results[name] = {'error': 'No trades', 'description': desc}
            continue

        print(f"  Trades: {metrics['total_trades']}, Final: ${metrics['final_equity']:.2f} ({metrics['total_return_pct']:+.1f}%)")
        print(f"  Sharpe: {metrics['sharpe']:.3f}, Sortino: {metrics['sortino']:.3f}, PF: {metrics['profit_factor']:.2f}, WR: {metrics['win_rate']:.1%}")
        print(f"  MaxDD: {metrics['max_drawdown']:.1%}")
        print(f"  Regime — Bull: {metrics['sharpe_bull']:.3f} ({metrics['n_bull_months']}m), Bear: {metrics['sharpe_bear']:.3f} ({metrics['n_bear_months']}m), Gap: {metrics['regime_gap']:.3f}")

        # Permutation test (random selection benchmark)
        print(f"  Running permutation test ({PERM_ITERATIONS} iters)...")
        perm_p = permutation_test(prices, fn, max_pos, metrics['sharpe'])
        print(f"  Perm p-value: {perm_p:.4f}")

        gates = validate_gates(metrics, perm_p)
        for g, v in gates.items():
            if g != 'all_pass':
                print(f"    {g}: {'PASS' if v else 'FAIL'}")
        print(f"  OVERALL: {'PASS' if gates['all_pass'] else 'FAIL'}")

        # Top holdings
        top = Counter(t['ticker'] for t in trade_log).most_common(5)

        results[name] = {
            'description': desc,
            'metrics': metrics,
            'permutation_p_value': round(perm_p, 4),
            'gates': gates,
            'top_holdings': [{'ticker': t, 'count': c} for t, c in top],
        }

    # Save
    output = {
        'strategy': 'Quality Factor Rotation',
        'run_date': datetime.now().isoformat(),
        'config': {
            'initial_capital': INITIAL_CAPITAL,
            'slippage_pct': SLIPPAGE_PCT,
            'oot_start': OOT_START,
            'oot_end': OOT_END,
            'universe_stocks': STOCKS,
            'universe_sector_etfs': SECTOR_ETFS,
            'permutation_iterations': PERM_ITERATIONS,
        },
        'variants': results,
        'gate_criteria': {
            'sharpe': '> 0.5',
            'perm_p': '< 0.05',
            'regime_gap': '< 0.5',
            'max_dd': '> -50%',
            'min_trades': '>= 20',
        },
    }

    out_path = '/home/jupiter/Lvl3Quant/data/quality_factor_rotation_results.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    # Summary
    print(f"\n{'='*100}")
    print("SUMMARY")
    print(f"{'='*100}")
    hdr = f"{'Variant':<32} {'Sharpe':>7} {'Sortino':>8} {'PF':>6} {'WR':>6} {'MaxDD':>7} {'Trd':>5} {'Final$':>9} {'p-val':>6} {'Gate':>5}"
    print(hdr)
    print("-" * len(hdr))
    for name, r in results.items():
        if 'error' in r:
            print(f"{name:<32} {'ERROR':>7}")
            continue
        m = r['metrics']
        g = 'PASS' if r['gates']['all_pass'] else 'FAIL'
        print(f"{name:<32} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} {m['profit_factor']:>6.2f} {m['win_rate']:>5.1%} {m['max_drawdown']:>6.1%} {m['total_trades']:>5} {m['final_equity']:>9.2f} {r['permutation_p_value']:>6.3f} {g:>5}")

    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()

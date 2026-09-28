#!/usr/bin/env python3
"""
Relative Value within Quality Universe Backtest
================================================
Select the relatively cheapest quality stocks and rotate monthly.
Variants D, E, F use price-based metrics (fully historical).
Variants A, B, C use current fundamentals as proxy (approximate).

Capital: $645, equal weight 3 stocks ($215 each), slippage 2bps
Period: 2022-01-01 to 2026-07-31
Monthly rebalance on first trading day of each month.
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from collections import OrderedDict

warnings.filterwarnings('ignore')

# ── Config ──────────────────────────────────────────────────────────────
UNIVERSE = [
    'AAPL', 'MSFT', 'AVGO', 'JPM', 'JNJ', 'PG', 'KO', 'PEP', 'HD', 'COST',
    'UNH', 'LLY', 'V', 'MA', 'ABBV', 'MRK', 'WMT', 'AMZN', 'GOOGL', 'META'
]
CAPITAL = 645.0
N_STOCKS = 3
ALLOC_PER_STOCK = CAPITAL / N_STOCKS  # $215
SLIPPAGE_BPS = 2
START = '2022-01-01'
END = '2026-07-31'
N_PERMUTATIONS = 1000

# ── Download Data ───────────────────────────────────────────────────────
print("Downloading price data...")
prices = yf.download(UNIVERSE, start='2021-01-01', end=END, auto_adjust=True, progress=False)
close = prices['Close'].dropna(how='all')
close = close.ffill().bfill()

bt_start = pd.Timestamp(START)
bt_end = pd.Timestamp(END)

print(f"Data range: {close.index[0].date()} to {close.index[-1].date()}")
print(f"Universe: {len(UNIVERSE)} stocks")

# ── Precompute RSI for all stocks ───────────────────────────────────────
print("Precomputing RSI(14) for all stocks...")
rsi_df = pd.DataFrame(index=close.index, columns=UNIVERSE, dtype=float)
for ticker in UNIVERSE:
    if ticker not in close.columns:
        continue
    delta = close[ticker].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_df[ticker] = 100 - (100 / (1 + rs))

# ── Precompute 52-week range position ───────────────────────────────────
print("Precomputing 52-week range positions...")
w52_pos_df = pd.DataFrame(index=close.index, columns=UNIVERSE, dtype=float)
for ticker in UNIVERSE:
    if ticker not in close.columns:
        continue
    high_52w = close[ticker].rolling(252, min_periods=100).max()
    low_52w = close[ticker].rolling(252, min_periods=100).min()
    range_val = high_52w - low_52w
    range_val = range_val.replace(0, np.nan)
    w52_pos_df[ticker] = (close[ticker] - low_52w) / range_val

# ── Get Monthly Rebalance Dates ─────────────────────────────────────────
bt_close = close.loc[bt_start:bt_end].copy()
all_dates = bt_close.index
monthly_groups = all_dates.to_period('M')
rebal_dates = []
for period in monthly_groups.unique():
    mask = monthly_groups == period
    month_dates = all_dates[mask]
    if len(month_dates) > 0:
        rebal_dates.append(month_dates[0])

rebal_set = set(rebal_dates)
print(f"Rebalance dates: {len(rebal_dates)}")

# ── Fetch Fundamentals ──────────────────────────────────────────────────
print("Fetching current fundamentals for variants A/B/C...")
fund_data = {}
for ticker in UNIVERSE:
    try:
        info = yf.Ticker(ticker).info
        fund_data[ticker] = {
            'trailingPE': info.get('trailingPE'),
            'priceToBook': info.get('priceToBook'),
            'dividendYield': info.get('dividendYield'),
        }
    except Exception:
        fund_data[ticker] = {'trailingPE': None, 'priceToBook': None, 'dividendYield': None}

has_pe = sum(1 for v in fund_data.values() if v['trailingPE'] is not None and v['trailingPE'] > 0)
has_pb = sum(1 for v in fund_data.values() if v['priceToBook'] is not None and v['priceToBook'] > 0)
has_dy = sum(1 for v in fund_data.values() if v['dividendYield'] is not None and v['dividendYield'] > 0)
print(f"  P/E: {has_pe}, P/B: {has_pb}, DivYield: {has_dy}")

# ── Ranking Functions ───────────────────────────────────────────────────
def rank_pe(date):
    pe_vals = {t: fund_data[t]['trailingPE'] for t in UNIVERSE
               if fund_data[t]['trailingPE'] is not None and fund_data[t]['trailingPE'] > 0}
    if len(pe_vals) < N_STOCKS:
        return UNIVERSE[:N_STOCKS]
    return [t for t, _ in sorted(pe_vals.items(), key=lambda x: x[1])[:N_STOCKS]]

def rank_pb(date):
    pb_vals = {t: fund_data[t]['priceToBook'] for t in UNIVERSE
               if fund_data[t]['priceToBook'] is not None and fund_data[t]['priceToBook'] > 0}
    if len(pb_vals) < N_STOCKS:
        return UNIVERSE[:N_STOCKS]
    return [t for t, _ in sorted(pb_vals.items(), key=lambda x: x[1])[:N_STOCKS]]

def rank_dy(date):
    dy_vals = {t: fund_data[t]['dividendYield'] for t in UNIVERSE
               if fund_data[t]['dividendYield'] is not None and fund_data[t]['dividendYield'] > 0}
    if len(dy_vals) < N_STOCKS:
        return UNIVERSE[:N_STOCKS]
    return [t for t, _ in sorted(dy_vals.items(), key=lambda x: x[1], reverse=True)[:N_STOCKS]]

def rank_52w(date):
    row = w52_pos_df.loc[date].dropna()
    available = [t for t in UNIVERSE if t in row.index]
    if len(available) < N_STOCKS:
        return UNIVERSE[:N_STOCKS]
    return list(row[available].sort_values().index[:N_STOCKS])

def rank_rsi(date):
    row = rsi_df.loc[date].dropna()
    available = [t for t in UNIVERSE if t in row.index]
    if len(available) < N_STOCKS:
        return UNIVERSE[:N_STOCKS]
    return list(row[available].sort_values().index[:N_STOCKS])

def rank_composite(date):
    pos_row = w52_pos_df.loc[date].dropna()
    rsi_row = rsi_df.loc[date].dropna()
    available = [t for t in UNIVERSE if t in pos_row.index and t in rsi_row.index]
    if len(available) < N_STOCKS:
        return UNIVERSE[:N_STOCKS]
    pos_rank = pos_row[available].rank()
    rsi_rank_vals = rsi_row[available].rank()
    composite = (pos_rank + rsi_rank_vals) / 2
    return list(composite.sort_values().index[:N_STOCKS])

# ── Vectorized Backtest Engine ──────────────────────────────────────────
def run_backtest(rank_func):
    """Run monthly rotation backtest. Returns daily equity series and trade count."""
    cash = CAPITAL
    holdings = {}  # {ticker: shares}
    daily_equity = np.zeros(len(bt_close))
    trades = 0
    trade_log = []
    selections_by_month = {}  # for permutation test

    for i, date in enumerate(bt_close.index):
        if date in rebal_set:
            selected = rank_func(date)
            selections_by_month[date] = selected

            # Sell all
            for ticker, shares in holdings.items():
                price = bt_close.loc[date, ticker]
                if not np.isnan(price):
                    cash += shares * price * (1 - SLIPPAGE_BPS / 10000)
                    trades += 1

            # Buy new
            holdings = {}
            per_stock = cash / N_STOCKS
            for ticker in selected:
                price = bt_close.loc[date, ticker]
                if not np.isnan(price):
                    buy_price = price * (1 + SLIPPAGE_BPS / 10000)
                    shares = per_stock / buy_price
                    holdings[ticker] = shares
                    cash -= shares * buy_price
                    trades += 1

            trade_log.append({'date': str(date.date()), 'selected': selected})

        # MTM
        mtm = sum(sh * bt_close.iloc[i][t] for t, sh in holdings.items()
                  if not np.isnan(bt_close.iloc[i].get(t, np.nan)))
        daily_equity[i] = cash + mtm

    return daily_equity, trades, trade_log, selections_by_month

# ── Metrics ─────────────────────────────────────────────────────────────
def compute_metrics(equity):
    """Compute performance metrics from equity curve array."""
    vals = pd.Series(equity, index=bt_close.index)
    returns = vals.pct_change().dropna()

    ann = 252
    mean_ret = returns.mean() * ann
    std_ret = returns.std() * np.sqrt(ann)
    sharpe = mean_ret / std_ret if std_ret > 0 else 0

    downside = returns[returns < 0]
    ds_std = downside.std() * np.sqrt(ann) if len(downside) > 0 else 1e-10
    sortino = mean_ret / ds_std if ds_std > 0 else 0

    cummax = vals.cummax()
    dd = (vals - cummax) / cummax
    max_dd = dd.min()

    total_ret = (vals.iloc[-1] / vals.iloc[0]) - 1
    years = (vals.index[-1] - vals.index[0]).days / 365.25
    cagr = (vals.iloc[-1] / vals.iloc[0]) ** (1 / years) - 1 if years > 0 else 0

    monthly = vals.resample('ME').last().pct_change().dropna()
    wr = (monthly > 0).mean() if len(monthly) > 0 else 0

    pos = returns[returns > 0].sum()
    neg = abs(returns[returns < 0].sum())
    pf = pos / neg if neg > 0 else float('inf')

    return {
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'max_drawdown': round(max_dd, 4),
        'total_return': round(total_ret, 4),
        'cagr': round(cagr, 4),
        'win_rate_monthly': round(wr, 4),
        'profit_factor': round(pf, 3),
        'annual_return': round(mean_ret, 4),
        'annual_vol': round(std_ret, 4),
        'final_value': round(vals.iloc[-1], 2),
        'start_value': round(vals.iloc[0], 2),
    }

# ── Regime Analysis ─────────────────────────────────────────────────────
print("Downloading SPY for regime classification...")
spy_raw = yf.download('SPY', start=START, end=END, auto_adjust=True, progress=False)['Close']
if isinstance(spy_raw, pd.DataFrame):
    spy_raw = spy_raw.squeeze()
spy_ret = spy_raw.pct_change()
spy_rolling = spy_ret.rolling(20).mean()
bull_mask = (spy_rolling > 0).reindex(bt_close.index).fillna(False)
bear_mask = ~bull_mask

def compute_regime_gap(equity):
    vals = pd.Series(equity, index=bt_close.index)
    returns = vals.pct_change().dropna()

    bull_rets = returns[bull_mask.reindex(returns.index).fillna(False)]
    bear_rets = returns[bear_mask.reindex(returns.index).fillna(True)]

    ann = 252
    s_bull = (bull_rets.mean() * ann) / (bull_rets.std() * np.sqrt(ann)) if len(bull_rets) > 20 and bull_rets.std() > 0 else 0
    s_bear = (bear_rets.mean() * ann) / (bear_rets.std() * np.sqrt(ann)) if len(bear_rets) > 20 and bear_rets.std() > 0 else 0

    mx = max(abs(s_bull), abs(s_bear))
    gap = abs(s_bull - s_bear) / mx if mx > 0 else 0

    return {
        'sharpe_bull': round(s_bull, 3),
        'sharpe_bear': round(s_bear, 3),
        'regime_gap': round(gap, 3),
        'bull_days': int(bull_mask.sum()),
        'bear_days': int(bear_mask.sum()),
    }

# ── Permutation Test (vectorized) ──────────────────────────────────────
def permutation_test(strategy_sharpe, n_perms=N_PERMUTATIONS):
    """
    Random monthly rotation: pick 3 random stocks each month.
    Vectorized: precompute monthly returns for each stock, then sample.
    """
    rng = np.random.RandomState(42)
    tickers = [t for t in UNIVERSE if t in bt_close.columns]
    n_tickers = len(tickers)

    # Compute return for each stock in each rebalance period
    # periods[i] = (start_idx, end_idx) in bt_close
    period_returns = {}  # {ticker: [ret_period_0, ret_period_1, ...]}
    period_starts = []
    period_ends = []

    for j in range(len(rebal_dates)):
        start = rebal_dates[j]
        end = rebal_dates[j + 1] if j + 1 < len(rebal_dates) else bt_close.index[-1]
        period_starts.append(start)
        period_ends.append(end)

    n_periods = len(period_starts)

    # For each ticker, compute the return over each period
    ticker_period_returns = np.zeros((n_tickers, n_periods))
    for ti, ticker in enumerate(tickers):
        for pi in range(n_periods):
            p_start = bt_close.loc[period_starts[pi], ticker]
            p_end = bt_close.loc[period_ends[pi], ticker]
            if p_start > 0 and not np.isnan(p_start) and not np.isnan(p_end):
                ticker_period_returns[ti, pi] = p_end / p_start - 1
            else:
                ticker_period_returns[ti, pi] = 0

    # For each permutation, pick 3 random stocks each period, equal weight return
    perm_sharpes = []
    slippage_factor = 2 * SLIPPAGE_BPS / 10000  # buy + sell slippage

    for _ in range(n_perms):
        # Portfolio return per period = mean of 3 random stocks' returns - slippage
        perm_period_returns = np.zeros(n_periods)
        for pi in range(n_periods):
            chosen = rng.choice(n_tickers, N_STOCKS, replace=False)
            perm_period_returns[pi] = np.mean(ticker_period_returns[chosen, pi]) - slippage_factor

        # Convert period returns to approximate daily Sharpe
        # Total return
        cumulative = np.prod(1 + perm_period_returns) - 1
        mean_period = np.mean(perm_period_returns)
        std_period = np.std(perm_period_returns, ddof=1) if n_periods > 1 else 1e-10

        # Annualize: ~12 periods/year for monthly
        periods_per_year = 12
        ann_ret = mean_period * periods_per_year
        ann_vol = std_period * np.sqrt(periods_per_year)
        if ann_vol > 0:
            perm_sharpes.append(ann_ret / ann_vol)

    if not perm_sharpes:
        return {'p_value': 1.0, 'strategy_sharpe': strategy_sharpe}

    p_value = np.mean([s >= strategy_sharpe for s in perm_sharpes])

    return {
        'p_value': round(p_value, 4),
        'strategy_sharpe': round(strategy_sharpe, 3),
        'perm_mean_sharpe': round(np.mean(perm_sharpes), 3),
        'perm_median_sharpe': round(np.median(perm_sharpes), 3),
        'perm_std_sharpe': round(np.std(perm_sharpes), 3),
        'perm_95th': round(np.percentile(perm_sharpes, 95), 3),
    }

# ── 5-Gate Validation ───────────────────────────────────────────────────
def validate_5gates(metrics, regime, perm, trades):
    gates = {}
    gates['sharpe_gt_0.5'] = {'pass': metrics['sharpe'] > 0.5, 'value': metrics['sharpe'], 'threshold': 0.5}
    gates['permutation_p_lt_0.05'] = {'pass': perm['p_value'] < 0.05, 'value': perm['p_value'], 'threshold': 0.05}
    gates['regime_gap_lt_0.5'] = {'pass': regime['regime_gap'] < 0.5, 'value': regime['regime_gap'], 'threshold': 0.5}
    gates['max_dd_gt_neg50pct'] = {'pass': metrics['max_drawdown'] > -0.50, 'value': metrics['max_drawdown'], 'threshold': -0.50}
    gates['trades_gte_20'] = {'pass': trades >= 20, 'value': trades, 'threshold': 20}

    all_pass = all(g['pass'] for g in gates.values())
    n_pass = sum(1 for g in gates.values() if g['pass'])
    return {'gates': gates, 'all_pass': all_pass, 'gates_passed': f"{n_pass}/5"}

# ── Benchmark ───────────────────────────────────────────────────────────
def run_benchmark():
    per_stock = CAPITAL / len(UNIVERSE)
    first_date = bt_close.index[0]
    shares = {}
    for ticker in UNIVERSE:
        if ticker in bt_close.columns:
            p = bt_close.loc[first_date, ticker]
            if not np.isnan(p) and p > 0:
                shares[ticker] = per_stock / (p * (1 + SLIPPAGE_BPS / 10000))

    equity = np.zeros(len(bt_close))
    for i, date in enumerate(bt_close.index):
        mtm = sum(sh * bt_close.iloc[i][t] for t, sh in shares.items()
                  if not np.isnan(bt_close.iloc[i].get(t, np.nan)))
        equity[i] = mtm
    return equity

# ── Run All Variants ────────────────────────────────────────────────────
variant_defs = OrderedDict([
    ('A_PE_ratio', {'func': rank_pe, 'desc': 'Buy 3 cheapest by trailing P/E (current proxy)'}),
    ('B_Price_Book', {'func': rank_pb, 'desc': 'Buy 3 cheapest by P/B (current proxy)'}),
    ('C_Dividend_Yield', {'func': rank_dy, 'desc': 'Buy 3 highest dividend yield (current proxy)'}),
    ('D_52w_Range', {'func': rank_52w, 'desc': 'Buy 3 closest to 52-week low'}),
    ('E_RSI_Oversold', {'func': rank_rsi, 'desc': 'Buy 3 with lowest RSI(14)'}),
    ('F_Composite', {'func': rank_composite, 'desc': 'Composite of D + E ranks'}),
])

print("\n" + "="*80)
print("RUNNING BACKTESTS")
print("="*80)

results_variants = OrderedDict()

for vname, vdef in variant_defs.items():
    print(f"\n--- Variant {vname}: {vdef['desc']} ---")

    equity, trades, trade_log, _ = run_backtest(vdef['func'])
    metrics = compute_metrics(equity)
    regime = compute_regime_gap(equity)

    print(f"  Sharpe: {metrics['sharpe']}, Sortino: {metrics['sortino']}, "
          f"MaxDD: {metrics['max_drawdown']:.1%}, Return: {metrics['total_return']:.1%}, "
          f"CAGR: {metrics['cagr']:.1%}, Trades: {trades}")
    print(f"  Regime: gap={regime['regime_gap']:.3f} (bull={regime['sharpe_bull']}, bear={regime['sharpe_bear']})")

    # Permutation test
    print(f"  Running {N_PERMUTATIONS} permutations...")
    perm = permutation_test(metrics['sharpe'])
    print(f"  Perm p={perm['p_value']:.4f} (strat={perm['strategy_sharpe']}, random_mean={perm['perm_mean_sharpe']})")

    validation = validate_5gates(metrics, regime, perm, trades)
    print(f"  5-Gate: {validation['gates_passed']} {'*** ALL PASS ***' if validation['all_pass'] else ''}")
    for gn, gv in validation['gates'].items():
        st = 'PASS' if gv['pass'] else 'FAIL'
        print(f"    {st}: {gn} = {gv['value']} (thresh: {gv['threshold']})")

    results_variants[vname] = {
        'description': vdef['desc'],
        'metrics': metrics,
        'regime': regime,
        'permutation': perm,
        'validation': validation,
        'trades': trades,
        'trade_log': trade_log[:5],
    }

# ── Benchmark ───────────────────────────────────────────────────────────
print("\n--- Benchmark: Equal-weight Buy & Hold (all 20) ---")
bench_eq = run_benchmark()
bench_metrics = compute_metrics(bench_eq)
print(f"  Sharpe: {bench_metrics['sharpe']}, Sortino: {bench_metrics['sortino']}, "
      f"Return: {bench_metrics['total_return']:.1%}, MaxDD: {bench_metrics['max_drawdown']:.1%}")

# ── Summary ─────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("SUMMARY TABLE")
print("="*80)
header = f"{'Variant':<20} {'Sharpe':>7} {'Sortino':>8} {'CAGR':>8} {'MaxDD':>8} {'PF':>6} {'WR':>6} {'RegGap':>7} {'Perm-p':>7} {'Gates':>6}"
print(header)
print("-" * len(header))

for vname, vdata in results_variants.items():
    m = vdata['metrics']
    r = vdata['regime']
    p = vdata['permutation']
    v = vdata['validation']
    print(f"{vname:<20} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} {m['cagr']:>7.1%} {m['max_drawdown']:>7.1%} "
          f"{m['profit_factor']:>6.2f} {m['win_rate_monthly']:>5.1%} {r['regime_gap']:>7.3f} {p['p_value']:>7.4f} {v['gates_passed']:>6}")

print(f"{'Benchmark(EW20)':<20} {bench_metrics['sharpe']:>7.3f} {bench_metrics['sortino']:>8.3f} {bench_metrics['cagr']:>7.1%} "
      f"{bench_metrics['max_drawdown']:>7.1%} {bench_metrics['profit_factor']:>6.2f} {bench_metrics['win_rate_monthly']:>5.1%}")

# ── Best Variant ────────────────────────────────────────────────────────
passing = {k: v for k, v in results_variants.items() if v['validation']['all_pass']}
if passing:
    best = max(passing.items(), key=lambda x: x[1]['metrics']['sharpe'])
    print(f"\nBest passing variant: {best[0]} (Sharpe: {best[1]['metrics']['sharpe']})")
else:
    best = max(results_variants.items(), key=lambda x: x[1]['metrics']['sharpe'])
    print(f"\nNo variant passed all 5 gates. Best by Sharpe: {best[0]} ({best[1]['metrics']['sharpe']})")

# ── Save ────────────────────────────────────────────────────────────────
output = {
    'strategy': 'Relative Value within Quality Universe',
    'period': f'{START} to {END}',
    'capital': CAPITAL,
    'universe_size': len(UNIVERSE),
    'stocks_per_month': N_STOCKS,
    'allocation_per_stock': ALLOC_PER_STOCK,
    'slippage_bps': SLIPPAGE_BPS,
    'rebalance_frequency': 'monthly',
    'n_rebalances': len(rebal_dates),
    'variants': results_variants,
    'benchmark': {'description': 'Equal-weight buy & hold all 20', 'metrics': bench_metrics},
    'best_variant': best[0],
    'any_passed_all_gates': len(passing) > 0,
    'passing_variants': list(passing.keys()) if passing else [],
    'generated_at': datetime.now().isoformat(),
}

out_path = '/home/jupiter/Lvl3Quant/data/relative_value_quality_results.json'
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"\nResults saved to {out_path}")
print("Done.")

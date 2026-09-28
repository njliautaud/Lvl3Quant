#!/usr/bin/env python3
"""
Sector ETF Pair Mean-Reversion v1
==================================
Market-neutral pair trading on sector ETFs using cointegration + z-score signals.

Thesis: Sector ETFs have structural relationships (e.g. XLE↔DBC energy linkage,
XLK↔QQQ tech overlap, XLF↔XLI cyclical correlation). When the spread between
cointegrated pairs deviates beyond 2σ, trade the convergence.

Advantages for agentic account ($645):
- Market neutral → works in all regimes
- Defined risk (stop at 3σ)
- Frequent signals (daily monitoring, ~5-10 trades/month)
- Can be implemented with ETF options for leverage

Walk-forward: 126d rolling cointegration window, daily signals.
"""

import sys
import os
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from itertools import combinations
from scipy import stats

warnings.filterwarnings('ignore')

sys.path.insert(0, '/home/jupiter/Lvl3Quant')

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# ─── CONFIGURATION ───────────────────────────────────────────────────────────

ETF_UNIVERSE = [
    'XLE', 'XLK', 'XLF', 'XLV', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB', 'XLRE',
    'XLC', 'GLD', 'SLV', 'DBC', 'TLT', 'IEF', 'HYG', 'QQQ', 'IWM', 'EEM',
    'VNQ', 'XBI',
]

# Strategy parameters
LOOKBACK = 126           # Cointegration lookback (6 months)
ZSCORE_ENTRY = 2.0       # Enter when z-score exceeds this
ZSCORE_EXIT = 0.5        # Exit when z-score reverts to this
ZSCORE_STOP = 3.5        # Stop-loss at extreme deviation
MAX_HOLD_DAYS = 30       # Max holding period
COINT_PVALUE = 0.05      # Cointegration test threshold
MIN_HALF_LIFE = 3        # Minimum mean-reversion half-life (days)
MAX_HALF_LIFE = 30       # Maximum mean-reversion half-life (days)

# Position sizing
STARTING_CAPITAL = 10000  # Paper capital (can scale down for agentic)
RISK_PER_TRADE_PCT = 5.0  # % of capital at risk per trade
MAX_CONCURRENT = 5        # Max concurrent pair trades
COMMISSION_PER_TRADE = 5  # Round trip per leg

# Walk-forward
MIN_TRAIN_DAYS = 126


def load_etf_data():
    """Load ETF data from cache or download."""
    cache_path = '/home/jupiter/Lvl3Quant/data/etf_universe_cache.parquet'
    if os.path.exists(cache_path):
        data = pd.read_parquet(cache_path)
        print(f"Loaded cached ETF data ({len(data)} rows)")
        return data
    else:
        raise FileNotFoundError("Run momentum_put_spread_etf_v1.py first to cache ETF data")


def build_price_matrix(data):
    """Build date x ticker close price matrix."""
    pivoted = data.pivot_table(index='date', columns='ticker', values='close')
    pivoted = pivoted.sort_index()
    pivoted = pivoted.dropna(axis=1, thresh=int(len(pivoted) * 0.8))  # Need 80% data
    pivoted = pivoted.ffill().bfill()
    return pivoted


def test_cointegration(y1, y2):
    """
    Test cointegration between two price series using Engle-Granger.
    Returns (is_cointegrated, hedge_ratio, half_life, adf_pvalue).
    """
    from statsmodels.tsa.stattools import adfuller

    # OLS regression: y1 = beta * y2 + alpha + epsilon
    X = np.column_stack([y2, np.ones(len(y2))])
    beta, alpha = np.linalg.lstsq(X, y1, rcond=None)[0]

    # Test residuals for stationarity
    spread = y1 - beta * y2 - alpha
    try:
        adf_result = adfuller(spread, maxlag=min(20, len(spread) // 4))
        adf_pvalue = adf_result[1]
    except Exception:
        return False, 0, 999, 1.0

    # Half-life of mean reversion
    spread_lag = spread[:-1]
    spread_delta = np.diff(spread)
    if len(spread_lag) > 1:
        try:
            slope = np.polyfit(spread_lag, spread_delta, 1)[0]
            half_life = -np.log(2) / slope if slope < 0 else 999
        except Exception:
            half_life = 999
    else:
        half_life = 999

    is_coint = (adf_pvalue < COINT_PVALUE and
                MIN_HALF_LIFE <= half_life <= MAX_HALF_LIFE)

    return is_coint, beta, half_life, adf_pvalue


def find_cointegrated_pairs(price_matrix, lookback_end_idx):
    """Find all cointegrated pairs using recent lookback window."""
    lookback_start = max(0, lookback_end_idx - LOOKBACK)
    window = price_matrix.iloc[lookback_start:lookback_end_idx]

    tickers = window.columns.tolist()
    pairs = []

    # Pre-filter: only test pairs with correlation > 0.3 (saves ~60% of ADF tests)
    corr_matrix = window.corr()

    for t1, t2 in combinations(tickers, 2):
        # Skip weakly correlated pairs
        if abs(corr_matrix.loc[t1, t2]) < 0.3:
            continue

        y1 = window[t1].values
        y2 = window[t2].values

        if np.any(np.isnan(y1)) or np.any(np.isnan(y2)):
            continue
        if np.std(y1) == 0 or np.std(y2) == 0:
            continue

        is_coint, beta, half_life, pvalue = test_cointegration(y1, y2)

        if is_coint:
            # Compute current z-score of spread
            spread = y1 - beta * y2
            spread_mean = np.mean(spread)
            spread_std = np.std(spread)
            if spread_std > 0:
                zscore = (spread[-1] - spread_mean) / spread_std
            else:
                zscore = 0

            pairs.append({
                'ticker1': t1,
                'ticker2': t2,
                'beta': beta,
                'half_life': half_life,
                'pvalue': pvalue,
                'zscore': zscore,
                'spread_mean': spread_mean,
                'spread_std': spread_std,
            })

    # Sort by p-value (most cointegrated first)
    pairs.sort(key=lambda x: x['pvalue'])
    return pairs


def run_backtest(price_matrix):
    """Run walk-forward pair trading backtest."""
    dates = price_matrix.index
    n_dates = len(dates)

    trades = []
    equity = [STARTING_CAPITAL]
    equity_dates = [dates[LOOKBACK]]
    open_positions = []

    print(f"  Walk-forward: {n_dates} days, starting at index {LOOKBACK}")

    for i in range(LOOKBACK, n_dates):
        current_date = dates[i]
        current_equity = equity[-1]

        if current_equity <= 0:
            break

        # Check existing positions for exit signals
        closed_this_bar = []
        for pos in open_positions:
            t1, t2 = pos['ticker1'], pos['ticker2']
            beta = pos['beta']

            # Current spread and z-score
            if t1 not in price_matrix.columns or t2 not in price_matrix.columns:
                continue

            p1 = price_matrix.loc[current_date, t1]
            p2 = price_matrix.loc[current_date, t2]
            spread = p1 - beta * p2
            zscore = (spread - pos['spread_mean']) / pos['spread_std']

            days_held = (current_date - pos['entry_date']).days

            # Exit conditions
            exit_signal = False
            exit_reason = ''

            if pos['direction'] == 'long_spread':
                # Long spread = long t1, short t2. Entered when z < -entry
                if zscore >= -ZSCORE_EXIT:
                    exit_signal = True
                    exit_reason = 'mean_reversion'
                elif zscore <= -ZSCORE_STOP:
                    exit_signal = True
                    exit_reason = 'stop_loss'
            else:
                # Short spread = short t1, long t2. Entered when z > entry
                if zscore <= ZSCORE_EXIT:
                    exit_signal = True
                    exit_reason = 'mean_reversion'
                elif zscore >= ZSCORE_STOP:
                    exit_signal = True
                    exit_reason = 'stop_loss'

            if days_held >= MAX_HOLD_DAYS:
                exit_signal = True
                exit_reason = 'max_hold'

            if exit_signal:
                # Calculate P&L
                entry_p1 = pos['entry_price1']
                entry_p2 = pos['entry_price2']

                if pos['direction'] == 'long_spread':
                    # Long t1, short t2
                    pnl_pct = (p1 / entry_p1 - 1) - beta * (p2 / entry_p2 - 1)
                else:
                    # Short t1, long t2
                    pnl_pct = -(p1 / entry_p1 - 1) + beta * (p2 / entry_p2 - 1)

                trade_size = pos['trade_size']
                pnl_dollars = trade_size * pnl_pct - COMMISSION_PER_TRADE * 2

                trades.append({
                    'entry_date': pos['entry_date'],
                    'exit_date': current_date,
                    'ticker1': t1,
                    'ticker2': t2,
                    'direction': pos['direction'],
                    'beta': beta,
                    'entry_zscore': pos['entry_zscore'],
                    'exit_zscore': zscore,
                    'entry_price1': entry_p1,
                    'entry_price2': entry_p2,
                    'exit_price1': p1,
                    'exit_price2': p2,
                    'pnl_pct': pnl_pct,
                    'pnl_dollars': pnl_dollars,
                    'days_held': days_held,
                    'exit_reason': exit_reason,
                    'won': pnl_dollars > 0,
                    'half_life': pos['half_life'],
                })

                current_equity += pnl_dollars
                closed_this_bar.append(pos)

        # Remove closed positions
        for pos in closed_this_bar:
            open_positions.remove(pos)

        # Progress logging
        if i % 200 == 0:
            pct = (i - LOOKBACK) / (n_dates - LOOKBACK) * 100
            print(f"    Progress: {pct:.0f}% ({i}/{n_dates}), {len(trades)} trades, {len(open_positions)} open, equity ${current_equity:.0f}")

        # Look for new entries (recalculate cointegration every 21 days to reduce compute)
        if i % 21 == 0 and len(open_positions) < MAX_CONCURRENT:
            pairs = find_cointegrated_pairs(price_matrix, i)

            for pair in pairs:
                if len(open_positions) >= MAX_CONCURRENT:
                    break

                # Skip if already in this pair
                pair_key = tuple(sorted([pair['ticker1'], pair['ticker2']]))
                if any(tuple(sorted([p['ticker1'], p['ticker2']])) == pair_key
                       for p in open_positions):
                    continue

                z = pair['zscore']

                # Entry signals
                direction = None
                if z < -ZSCORE_ENTRY:
                    direction = 'long_spread'  # Spread too low → buy spread
                elif z > ZSCORE_ENTRY:
                    direction = 'short_spread'  # Spread too high → sell spread

                if direction:
                    t1 = pair['ticker1']
                    t2 = pair['ticker2']

                    if t1 not in price_matrix.columns or t2 not in price_matrix.columns:
                        continue

                    p1 = price_matrix.loc[current_date, t1]
                    p2 = price_matrix.loc[current_date, t2]

                    trade_size = current_equity * RISK_PER_TRADE_PCT / 100

                    open_positions.append({
                        'ticker1': t1,
                        'ticker2': t2,
                        'beta': pair['beta'],
                        'direction': direction,
                        'entry_date': current_date,
                        'entry_price1': p1,
                        'entry_price2': p2,
                        'entry_zscore': z,
                        'spread_mean': pair['spread_mean'],
                        'spread_std': pair['spread_std'],
                        'trade_size': trade_size,
                        'half_life': pair['half_life'],
                    })

        equity.append(current_equity)
        equity_dates.append(current_date)

    # Close remaining positions at last price
    for pos in open_positions:
        t1, t2 = pos['ticker1'], pos['ticker2']
        if t1 in price_matrix.columns and t2 in price_matrix.columns:
            p1 = price_matrix.iloc[-1][t1]
            p2 = price_matrix.iloc[-1][t2]
            entry_p1, entry_p2 = pos['entry_price1'], pos['entry_price2']

            if pos['direction'] == 'long_spread':
                pnl_pct = (p1 / entry_p1 - 1) - pos['beta'] * (p2 / entry_p2 - 1)
            else:
                pnl_pct = -(p1 / entry_p1 - 1) + pos['beta'] * (p2 / entry_p2 - 1)

            pnl_dollars = pos['trade_size'] * pnl_pct - COMMISSION_PER_TRADE * 2
            trades.append({
                'entry_date': pos['entry_date'],
                'exit_date': dates[-1],
                'ticker1': t1, 'ticker2': t2,
                'direction': pos['direction'],
                'pnl_dollars': pnl_dollars,
                'pnl_pct': pnl_pct,
                'days_held': (dates[-1] - pos['entry_date']).days,
                'exit_reason': 'forced_close',
                'won': pnl_dollars > 0,
                'half_life': pos['half_life'],
                'beta': pos['beta'],
                'entry_zscore': pos['entry_zscore'],
                'exit_zscore': 0,
                'entry_price1': entry_p1, 'entry_price2': entry_p2,
                'exit_price1': p1, 'exit_price2': p2,
            })

    return {
        'trades': trades,
        'equity': equity,
        'equity_dates': equity_dates,
    }


def compute_metrics(result):
    """Compute risk-adjusted metrics."""
    trades = result['trades']
    equity = result['equity']

    if len(trades) == 0:
        return None

    total = len(trades)
    winners = sum(1 for t in trades if t['pnl_dollars'] > 0)
    wr = winners / total

    pnls = [t['pnl_dollars'] for t in trades]
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    avg_win = np.mean([p for p in pnls if p > 0]) if any(p > 0 for p in pnls) else 0
    avg_loss = np.mean([p for p in pnls if p < 0]) if any(p < 0 for p in pnls) else 0

    eq = np.array(equity)
    final = eq[-1]

    if len(result['equity_dates']) >= 2:
        first = pd.Timestamp(result['equity_dates'][0])
        last = pd.Timestamp(result['equity_dates'][-1])
        years = (last - first).days / 365.25
    else:
        years = 1

    cagr = (final / STARTING_CAPITAL) ** (1 / max(years, 0.1)) - 1 if final > 0 else -1

    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / np.where(peak > 0, peak, 1)
    max_dd = np.min(dd)

    # Daily returns for Sharpe
    daily_rets = np.diff(eq) / eq[:-1]
    daily_rets = daily_rets[np.isfinite(daily_rets)]
    if len(daily_rets) > 1 and np.std(daily_rets) > 0:
        sharpe = np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252)
        down_rets = daily_rets[daily_rets < 0]
        down_std = np.std(down_rets) if len(down_rets) > 1 else np.std(daily_rets)
        sortino = np.mean(daily_rets) / down_std * np.sqrt(252) if down_std > 0 else 0
    else:
        sharpe = 0
        sortino = 0

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    avg_hold = np.mean([t['days_held'] for t in trades])

    # Win/loss by exit reason
    reasons = {}
    for t in trades:
        r = t.get('exit_reason', 'unknown')
        if r not in reasons:
            reasons[r] = {'count': 0, 'wins': 0, 'pnl': 0}
        reasons[r]['count'] += 1
        if t['pnl_dollars'] > 0:
            reasons[r]['wins'] += 1
        reasons[r]['pnl'] += t['pnl_dollars']

    return {
        'total_trades': total,
        'win_rate': wr,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': pf,
        'sharpe': sharpe,
        'sortino': sortino,
        'cagr': cagr,
        'max_dd': max_dd,
        'calmar': calmar,
        'final_equity': final,
        'years': years,
        'trades_per_year': total / max(years, 0.1),
        'avg_hold_days': avg_hold,
        'exit_reasons': reasons,
    }


def permutation_test(price_matrix, result, n_perms=200):
    """Shuffle pair assignments to test if cointegration-based selection adds value."""
    real_metrics = compute_metrics(result)
    if real_metrics is None:
        return 1.0, []

    real_sharpe = real_metrics['sharpe']
    trades = result['trades']
    pnls = [t['pnl_dollars'] for t in trades]

    perm_sharpes = []
    print(f"  Running {n_perms} permutations...")

    for _ in range(n_perms):
        # Shuffle PnLs randomly
        shuffled = np.random.permutation(pnls)
        eq = [STARTING_CAPITAL]
        for p in shuffled:
            eq.append(eq[-1] + p)
        eq = np.array(eq)
        daily_rets = np.diff(eq) / np.where(eq[:-1] > 0, eq[:-1], 1)
        daily_rets = daily_rets[np.isfinite(daily_rets)]
        if len(daily_rets) > 1 and np.std(daily_rets) > 0:
            s = np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252)
        else:
            s = 0
        perm_sharpes.append(s)

    p_value = np.mean([1 if ps >= real_sharpe else 0 for ps in perm_sharpes])
    return p_value, perm_sharpes


def regime_test(result, price_matrix):
    """R1 regime test using QQQ as market proxy."""
    trades = result['trades']
    if len(trades) == 0:
        return None

    # QQQ SMA200 for regime
    if 'QQQ' not in price_matrix.columns:
        return {'pass': True, 'gap': 0, 'bull_sharpe': 0, 'bear_sharpe': 0}

    qqq = price_matrix['QQQ']
    sma200 = qqq.rolling(200).mean()

    bull_pnls, bear_pnls = [], []
    for t in trades:
        d = pd.Timestamp(t['entry_date'])
        if d in sma200.index and pd.notna(sma200.loc[d]):
            if qqq.loc[d] > sma200.loc[d]:
                bull_pnls.append(t['pnl_dollars'])
            else:
                bear_pnls.append(t['pnl_dollars'])
        else:
            bull_pnls.append(t['pnl_dollars'])

    def sharpe_from_pnls(p):
        if len(p) < 2:
            return 0
        return np.mean(p) / (np.std(p) + 1e-10) * np.sqrt(252 / 10)  # ~10 day hold avg

    bs = sharpe_from_pnls(bull_pnls)
    brs = sharpe_from_pnls(bear_pnls)
    mx = max(abs(bs), abs(brs))
    gap = abs(bs - brs) / mx if mx > 0 else 0

    return {
        'bull_sharpe': bs,
        'bear_sharpe': brs,
        'bull_trades': len(bull_pnls),
        'bear_trades': len(bear_pnls),
        'gap': gap,
        'pass': gap <= 0.50,
    }


def sub_period_test(result):
    """Split trades into halves, both must be profitable."""
    trades = result['trades']
    if len(trades) < 10:
        return {'pass': False, 'h1_sharpe': 0, 'h2_sharpe': 0}

    mid = len(trades) // 2
    def half_sharpe(tl):
        pnls = [t['pnl_dollars'] for t in tl]
        if len(pnls) < 2 or np.std(pnls) == 0:
            return 0
        return np.mean(pnls) / np.std(pnls) * np.sqrt(252 / 10)

    h1 = half_sharpe(trades[:mid])
    h2 = half_sharpe(trades[mid:])
    return {'h1_sharpe': h1, 'h2_sharpe': h2, 'pass': h1 > 0 and h2 > 0}


def outlier_test(result):
    """Remove top 5% winners, check still profitable."""
    trades = result['trades']
    pnls = sorted([t['pnl_dollars'] for t in trades])
    if len(pnls) < 20:
        return {'pass': True, 'trimmed_sharpe': 0}
    n_remove = max(1, int(len(pnls) * 0.05))
    trimmed = pnls[:-n_remove]
    if len(trimmed) < 2 or np.std(trimmed) == 0:
        return {'pass': True, 'trimmed_sharpe': 0}
    ts = np.mean(trimmed) / np.std(trimmed) * np.sqrt(252 / 10)
    return {'trimmed_sharpe': ts, 'pass': ts > 0}


def main():
    print("=" * 70)
    print("SECTOR ETF PAIR MEAN-REVERSION v1")
    print("=" * 70)
    print(f"Start time: {datetime.now()}")

    if MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("sector_pair_meanrev")
        mlflow.start_run(run_name=f"v1_{datetime.now().strftime('%Y%m%d_%H%M')}")

    # Load data
    data = load_etf_data()
    price_matrix = build_price_matrix(data)
    print(f"\nPrice matrix: {price_matrix.shape[0]} days x {price_matrix.shape[1]} ETFs")
    print(f"Date range: {price_matrix.index[0]} to {price_matrix.index[-1]}")
    print(f"ETFs: {list(price_matrix.columns)}")

    # Run backtest
    print(f"\nRunning pair trading backtest...")
    result = run_backtest(price_matrix)

    if len(result['trades']) == 0:
        print("NO TRADES GENERATED")
        if MLFLOW_AVAILABLE:
            mlflow.log_param("status", "NO_TRADES")
            mlflow.end_run()
        return

    metrics = compute_metrics(result)

    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"  Sharpe:     {metrics['sharpe']:.2f}")
    print(f"  Sortino:    {metrics['sortino']:.2f}")
    print(f"  CAGR:       {metrics['cagr']:.1%}")
    print(f"  MaxDD:      {metrics['max_dd']:.1%}")
    print(f"  WR:         {metrics['win_rate']:.1%}")
    print(f"  PF:         {metrics['profit_factor']:.2f}")
    print(f"  Calmar:     {metrics['calmar']:.2f}")
    print(f"  Trades:     {metrics['total_trades']} ({metrics['trades_per_year']:.0f}/yr)")
    print(f"  Avg Hold:   {metrics['avg_hold_days']:.1f} days")
    print(f"  Final:      ${metrics['final_equity']:.0f} (from ${STARTING_CAPITAL})")

    print(f"\n  Exit reasons:")
    for reason, stats_dict in metrics['exit_reasons'].items():
        wr = stats_dict['wins'] / stats_dict['count'] if stats_dict['count'] > 0 else 0
        print(f"    {reason}: {stats_dict['count']} trades, WR {wr:.0%}, PnL ${stats_dict['pnl']:.0f}")

    # Top pairs
    trade_df = pd.DataFrame(result['trades'])
    pair_key = trade_df.apply(lambda r: f"{r['ticker1']}/{r['ticker2']}", axis=1)
    trade_df['pair'] = pair_key
    print(f"\n  Top pairs by frequency:")
    for pair, group in trade_df.groupby('pair'):
        if len(group) >= 3:
            wr = group['won'].mean()
            pnl = group['pnl_dollars'].sum()
            avg_hl = group['half_life'].mean()
            print(f"    {pair}: {len(group)} trades, WR {wr:.0%}, PnL ${pnl:.0f}, avg HL {avg_hl:.0f}d")

    # Adversarial gates
    print(f"\n--- GATE 1: Permutation Test ---")
    perm_p, perm_sharpes = permutation_test(price_matrix, result)
    perm_pass = perm_p < 0.05
    print(f"  p-value: {perm_p:.3f} {'PASS ✅' if perm_pass else 'FAIL ❌'}")

    print(f"\n--- GATE 2: Regime-Agnostic ---")
    regime = regime_test(result, price_matrix)
    print(f"  Bull Sharpe: {regime['bull_sharpe']:.2f} ({regime['bull_trades']} trades)")
    print(f"  Bear Sharpe: {regime['bear_sharpe']:.2f} ({regime['bear_trades']} trades)")
    print(f"  Gap: {regime['gap']:.3f} {'PASS ✅' if regime['pass'] else 'FAIL ❌'}")

    print(f"\n--- GATE 3: Sub-Period ---")
    sub = sub_period_test(result)
    print(f"  H1 Sharpe: {sub['h1_sharpe']:.2f}, H2 Sharpe: {sub['h2_sharpe']:.2f}")
    print(f"  {'PASS ✅' if sub['pass'] else 'FAIL ❌'}")

    print(f"\n--- GATE 4: Outlier ---")
    outlier = outlier_test(result)
    print(f"  Trimmed Sharpe: {outlier['trimmed_sharpe']:.2f}")
    print(f"  {'PASS ✅' if outlier['pass'] else 'FAIL ❌'}")

    gates_passed = sum([perm_pass, regime['pass'], sub['pass'], outlier['pass']])
    print(f"\n{'='*70}")
    print(f"GATES: {gates_passed}/4")
    print(f"{'='*70}")

    # Save results
    results_path = '/home/jupiter/Lvl3Quant/research/findings/sector_pair_meanrev_v1_results.json'
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    save_data = {
        'timestamp': datetime.now().isoformat(),
        'metrics': metrics,
        'gates': {
            'perm_p': perm_p, 'perm_pass': perm_pass,
            'regime': {k: v for k, v in regime.items() if k != 'pass'},
            'regime_pass': regime['pass'],
            'sub_period': sub, 'outlier': outlier,
            'total_pass': gates_passed,
        },
        'config': {
            'lookback': LOOKBACK,
            'zscore_entry': ZSCORE_ENTRY,
            'zscore_exit': ZSCORE_EXIT,
            'zscore_stop': ZSCORE_STOP,
            'max_hold_days': MAX_HOLD_DAYS,
            'coint_pvalue': COINT_PVALUE,
            'min_half_life': MIN_HALF_LIFE,
            'max_half_life': MAX_HALF_LIFE,
            'universe_size': len(ETF_UNIVERSE),
        }
    }

    # Convert exit_reasons for JSON
    save_data['metrics']['exit_reasons'] = {
        k: v for k, v in metrics['exit_reasons'].items()
    }

    with open(results_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved")

    if MLFLOW_AVAILABLE:
        mlflow.log_param("strategy", "sector_pair_meanrev")
        mlflow.log_param("universe_size", len(ETF_UNIVERSE))
        mlflow.log_param("lookback", LOOKBACK)
        mlflow.log_param("zscore_entry", ZSCORE_ENTRY)
        mlflow.log_metric("sharpe", metrics['sharpe'])
        mlflow.log_metric("sortino", metrics['sortino'])
        mlflow.log_metric("cagr", metrics['cagr'])
        mlflow.log_metric("max_dd", metrics['max_dd'])
        mlflow.log_metric("win_rate", metrics['win_rate'])
        mlflow.log_metric("profit_factor", metrics['profit_factor'])
        mlflow.log_metric("total_trades", metrics['total_trades'])
        mlflow.log_metric("gates_passed", gates_passed)
        mlflow.log_metric("perm_p", perm_p)
        mlflow.log_artifact(results_path)
        mlflow.end_run()

    print(f"\nCompleted at {datetime.now()}")
    return save_data


if __name__ == '__main__':
    main()

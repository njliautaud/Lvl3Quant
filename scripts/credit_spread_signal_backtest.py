#!/usr/bin/env python3
"""
Credit Spread / Fixed Income Quality Signal Backtest
=====================================================
Tests 6 variants (A-F) using credit spread dynamics (HYG/LQD/JNK)
as uncorrelated alpha signals. These capture credit risk dynamics
potentially uncorrelated to tech/equity momentum (QQQ).

Variants:
  A: Credit Spread Mean Reversion
  B: High Yield Momentum
  C: Credit Quality Rotation
  D: Junk Bond Contrarian
  E: Credit-Equity Divergence
  F: Multi-Credit Score
"""

import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

# ─── Configuration ───────────────────────────────────────────────────────────

START_DATE = '2022-01-01'
END_DATE = '2026-07-29'
STARTING_CAPITAL = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02%
COMMISSION = 0.0
N_PERMUTATIONS = 1000

# Gates
SHARPE_GATE = 0.5
PERM_P_GATE = 0.05
REGIME_GAP_GATE = 0.50
MDD_GATE = -0.50
MIN_TRADES_GATE = 20

ALL_ETFS = ['HYG', 'LQD', 'JNK', 'TLT', 'SPY', 'QQQ', '^VIX']


def download_data():
    """Download all ETF data via yfinance."""
    import yfinance as yf
    print("Downloading ETF data...")
    data = yf.download(ALL_ETFS, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        close = data['Close']
    else:
        close = data
    # Rename ^VIX column
    if '^VIX' in close.columns:
        close = close.rename(columns={'^VIX': 'VIX'})
    close = close.ffill().dropna(how='all')
    print(f"  Data: {close.index[0].date()} to {close.index[-1].date()}, {len(close)} days")
    return close


def apply_slippage(price, direction='buy'):
    """Apply slippage to trade price."""
    if direction == 'buy':
        return price * (1 + SLIPPAGE_PCT)
    else:
        return price * (1 - SLIPPAGE_PCT)


def simulate_strategy(close, signal_func, strategy_name):
    """
    Generic strategy simulator.
    signal_func(close, i) -> (action, ticker, hold_days)
      action: 'buy' or None
      ticker: which ETF to buy
      hold_days: how long to hold
    Returns equity curve and trade log.
    """
    equity = STARTING_CAPITAL
    equity_curve = []
    trades = []
    position = None  # {'ticker', 'entry_price', 'entry_date', 'shares', 'exit_date_idx'}

    for i in range(60, len(close)):  # need 60d lookback
        date = close.index[i]

        # Check if position needs to be exited
        if position is not None:
            if i >= position['exit_idx']:
                # Exit
                exit_price = apply_slippage(close[position['ticker']].iloc[i], 'sell')
                pnl = (exit_price - position['entry_price']) * position['shares']
                equity += pnl
                trades.append({
                    'entry_date': str(position['entry_date'].date()),
                    'exit_date': str(date.date()),
                    'ticker': position['ticker'],
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'shares': position['shares'],
                    'pnl': pnl,
                    'return_pct': (exit_price / position['entry_price'] - 1) * 100
                })
                position = None

        # Try to enter if no position
        if position is None:
            action, ticker, hold_days = signal_func(close, i)
            if action == 'buy' and ticker is not None:
                entry_price = apply_slippage(close[ticker].iloc[i], 'buy')
                shares = equity / entry_price
                position = {
                    'ticker': ticker,
                    'entry_price': entry_price,
                    'entry_date': date,
                    'shares': shares,
                    'exit_idx': i + hold_days
                }

        # Record equity
        if position is not None:
            current_price = close[position['ticker']].iloc[i]
            mark_to_market = (current_price - position['entry_price']) * position['shares']
            equity_curve.append({'date': date, 'equity': equity + mark_to_market})
        else:
            equity_curve.append({'date': date, 'equity': equity})

    # Force close any open position
    if position is not None:
        exit_price = apply_slippage(close[position['ticker']].iloc[-1], 'sell')
        pnl = (exit_price - position['entry_price']) * position['shares']
        equity += pnl
        trades.append({
            'entry_date': str(position['entry_date'].date()),
            'exit_date': str(close.index[-1].date()),
            'ticker': position['ticker'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'shares': position['shares'],
            'pnl': pnl,
            'return_pct': (exit_price / position['entry_price'] - 1) * 100
        })

    eq_df = pd.DataFrame(equity_curve).set_index('date')
    return eq_df, trades


# ─── Strategy Signal Functions ────────────────────────────────────────────────

def signal_A_credit_spread_mean_reversion(close, i):
    """When HYG/LQD ratio drops >2% below 20d MA, buy HYG. Hold 15d."""
    hyg = close['HYG'].iloc[max(0, i-25):i+1]
    lqd = close['LQD'].iloc[max(0, i-25):i+1]
    ratio = hyg / lqd
    if len(ratio) < 21:
        return None, None, 0
    ma20 = ratio.iloc[-21:-1].mean()
    current = ratio.iloc[-1]
    if current < ma20 * 0.98:  # dropped >2% below 20d MA
        return 'buy', 'HYG', 15
    return None, None, 0


def signal_B_high_yield_momentum(close, i):
    """Long HYG when 20d return > 0 AND VIX < 25. Rebalance every 10d."""
    if i % 10 != 0:  # only rebalance every 10 days
        return None, None, 0
    hyg_ret_20d = (close['HYG'].iloc[i] / close['HYG'].iloc[i-20] - 1)
    vix = close['VIX'].iloc[i]
    if hyg_ret_20d > 0 and vix < 25:
        return 'buy', 'HYG', 10
    return None, None, 0


def signal_C_credit_quality_rotation(close, i):
    """HYG outperforming LQD on 20d → long HYG; else long LQD. Rebal 15d."""
    if i % 15 != 0:
        return None, None, 0
    hyg_ret = close['HYG'].iloc[i] / close['HYG'].iloc[i-20] - 1
    lqd_ret = close['LQD'].iloc[i] / close['LQD'].iloc[i-20] - 1
    if hyg_ret > lqd_ret:
        return 'buy', 'HYG', 15
    else:
        return 'buy', 'LQD', 15
    return None, None, 0


def signal_D_junk_bond_contrarian(close, i):
    """When JNK drops >3% in 20d AND VIX > 25, buy JNK. Hold 20d."""
    jnk_ret_20d = close['JNK'].iloc[i] / close['JNK'].iloc[i-20] - 1
    vix = close['VIX'].iloc[i]
    if jnk_ret_20d < -0.03 and vix > 25:
        return 'buy', 'JNK', 20
    return None, None, 0


def signal_E_credit_equity_divergence(close, i):
    """HYG/SPY ratio rising → long SPY; falling → long TLT. Rebal 10d."""
    if i % 10 != 0:
        return None, None, 0
    ratio_now = close['HYG'].iloc[i] / close['SPY'].iloc[i]
    ratio_20d_ago = close['HYG'].iloc[i-20] / close['SPY'].iloc[i-20]
    ratio_mom = ratio_now / ratio_20d_ago - 1
    if ratio_mom > 0:
        return 'buy', 'SPY', 10
    else:
        return 'buy', 'TLT', 10


def signal_F_multi_credit_score(close, i):
    """
    Score = HYG 20d mom flag + HYG/LQD ratio vs 60d MA flag + VIX<20 flag
    Score >= 2 → long HYG; Score <= 0 → long LQD; else cash. Rebal 10d.
    """
    if i % 10 != 0:
        return None, None, 0

    # Component 1: HYG 20d momentum > 0
    hyg_mom = close['HYG'].iloc[i] / close['HYG'].iloc[i-20] - 1
    score1 = 1 if hyg_mom > 0 else 0

    # Component 2: HYG/LQD ratio above 60d MA
    ratio = close['HYG'].iloc[max(0,i-65):i+1] / close['LQD'].iloc[max(0,i-65):i+1]
    if len(ratio) >= 61:
        ma60 = ratio.iloc[-61:-1].mean()
        score2 = 1 if ratio.iloc[-1] > ma60 else 0
    else:
        score2 = 0

    # Component 3: VIX < 20
    vix = close['VIX'].iloc[i]
    score3 = 1 if vix < 20 else 0

    total_score = score1 + score2 + score3
    if total_score >= 2:
        return 'buy', 'HYG', 10
    elif total_score <= 0:
        return 'buy', 'LQD', 10
    return None, None, 0


# ─── Validation ───────────────────────────────────────────────────────────────

def compute_metrics(eq_df, trades, close):
    """Compute all metrics for a strategy."""
    if len(eq_df) == 0:
        return None

    eq = eq_df['equity']
    daily_returns = eq.pct_change().dropna()

    # Basic metrics
    total_return = (eq.iloc[-1] / eq.iloc[0] - 1) * 100
    n_years = len(daily_returns) / 252
    ann_return = (1 + total_return/100) ** (1/max(n_years, 0.01)) - 1

    # Sharpe
    if daily_returns.std() > 0:
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Sortino
    downside = daily_returns[daily_returns < 0]
    if len(downside) > 0 and downside.std() > 0:
        sortino = (daily_returns.mean() / downside.std()) * np.sqrt(252)
    else:
        sortino = 0.0

    # Max Drawdown
    cummax = eq.cummax()
    drawdown = (eq - cummax) / cummax
    mdd = drawdown.min()

    # Profit Factor and Win Rate from trades
    if len(trades) > 0:
        wins = [t['pnl'] for t in trades if t['pnl'] > 0]
        losses = [t['pnl'] for t in trades if t['pnl'] <= 0]
        total_win = sum(wins) if wins else 0
        total_loss = abs(sum(losses)) if losses else 0.001
        pf = total_win / total_loss if total_loss > 0 else 999
        wr = len(wins) / len(trades) * 100
    else:
        pf = 0
        wr = 0

    # QQQ correlation
    qqq_returns = close['QQQ'].pct_change().dropna()
    common_idx = daily_returns.index.intersection(qqq_returns.index)
    if len(common_idx) > 30:
        qqq_corr = daily_returns.loc[common_idx].corr(qqq_returns.loc[common_idx])
    else:
        qqq_corr = np.nan

    return {
        'total_return_pct': round(total_return, 2),
        'annual_return_pct': round(ann_return * 100, 2),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'max_drawdown_pct': round(mdd * 100, 2),
        'profit_factor': round(pf, 3),
        'win_rate_pct': round(wr, 1),
        'num_trades': len(trades),
        'qqq_correlation': round(qqq_corr, 4) if not np.isnan(qqq_corr) else None,
        'final_equity': round(eq.iloc[-1], 2),
        'starting_equity': STARTING_CAPITAL
    }


def permutation_test(eq_df, trades, close, n_perms=N_PERMUTATIONS):
    """
    Permutation test: randomize trade entry dates to test if signal timing matters.
    For each permutation, we randomly shift all trade entry dates and recompute total return.
    This tests whether the TIMING of entries matters vs random timing.
    """
    if len(trades) < 5:
        return {
            'actual_sharpe': 0.0,
            'p_value': 1.0,
            'perm_mean_sharpe': 0.0,
            'perm_std_sharpe': 0.0,
            'note': 'too few trades for permutation test'
        }

    eq = eq_df['equity']
    daily_returns = eq.pct_change().dropna().values
    actual_sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0

    # Compute actual total return from trades
    actual_total_pnl = sum(t['pnl'] for t in trades)

    # For permutation: get the tickers and hold periods from actual trades,
    # but randomize entry dates
    trade_specs = [(t['ticker'], int((pd.Timestamp(t['exit_date']) - pd.Timestamp(t['entry_date'])).days))
                   for t in trades]

    rng = np.random.RandomState(42)
    valid_start = 60
    valid_end = len(close) - max(h for _, h in trade_specs) - 1 if trade_specs else len(close) - 30

    perm_sharpes = []
    for _ in range(n_perms):
        # Random entry dates (non-overlapping)
        perm_pnl = 0
        perm_equity = STARTING_CAPITAL
        used_ranges = []

        for ticker, hold in trade_specs:
            # Try random entry
            attempts = 0
            while attempts < 50:
                idx = rng.randint(valid_start, max(valid_start+1, valid_end))
                exit_idx = min(idx + hold, len(close) - 1)
                # Check no overlap with used ranges
                overlap = any(not (exit_idx < s or idx > e) for s, e in used_ranges)
                if not overlap:
                    break
                attempts += 1

            if idx < len(close) and exit_idx < len(close) and ticker in close.columns:
                entry_p = close[ticker].iloc[idx]
                exit_p = close[ticker].iloc[exit_idx]
                ret = exit_p / entry_p - 1
                perm_pnl += perm_equity * ret
                used_ranges.append((idx, exit_idx))

        perm_total_ret = perm_pnl / STARTING_CAPITAL
        # Convert to annualized Sharpe-like metric using total return
        perm_sharpes.append(perm_total_ret)

    actual_total_ret = actual_total_pnl / STARTING_CAPITAL
    perm_sharpes = np.array(perm_sharpes)
    p_value = (perm_sharpes >= actual_total_ret).mean()

    return {
        'actual_sharpe': round(actual_sharpe, 4),
        'p_value': round(p_value, 4),
        'perm_mean_return': round(perm_sharpes.mean(), 4),
        'perm_std_return': round(perm_sharpes.std(), 4),
        'actual_total_return': round(actual_total_ret, 4)
    }


def regime_split(eq_df, close):
    """Split performance by bull/bear regime (SPY vs 200-SMA)."""
    spy = close['SPY']
    sma200 = spy.rolling(200).mean()

    eq = eq_df['equity']
    daily_returns = eq.pct_change().dropna()

    bull_days = sma200.index[spy > sma200]
    bear_days = sma200.index[spy <= sma200]

    bull_rets = daily_returns[daily_returns.index.isin(bull_days)]
    bear_rets = daily_returns[daily_returns.index.isin(bear_days)]

    def regime_sharpe(rets):
        if len(rets) < 20 or rets.std() == 0:
            return 0.0
        return (rets.mean() / rets.std()) * np.sqrt(252)

    bull_sharpe = regime_sharpe(bull_rets)
    bear_sharpe = regime_sharpe(bear_rets)

    # Regime gap
    max_abs = max(abs(bull_sharpe), abs(bear_sharpe), 0.001)
    regime_gap = abs(bull_sharpe - bear_sharpe) / max_abs

    return {
        'bull_sharpe': round(bull_sharpe, 3),
        'bear_sharpe': round(bear_sharpe, 3),
        'bull_days': len(bull_rets),
        'bear_days': len(bear_rets),
        'regime_gap': round(regime_gap, 3)
    }


def check_gates(metrics, perm, regime):
    """5-gate validation."""
    gates = {}
    gates['sharpe_gt_0.5'] = metrics['sharpe'] > SHARPE_GATE
    gates['perm_p_lt_0.05'] = perm['p_value'] < PERM_P_GATE
    gates['regime_gap_lt_0.5'] = regime['regime_gap'] < REGIME_GAP_GATE
    gates['mdd_gt_neg50'] = metrics['max_drawdown_pct'] > MDD_GATE * 100
    gates['trades_gte_20'] = metrics['num_trades'] >= MIN_TRADES_GATE
    gates['all_passed'] = all(gates.values())
    return gates


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    close = download_data()

    strategies = {
        'A_credit_spread_mean_reversion': signal_A_credit_spread_mean_reversion,
        'B_high_yield_momentum': signal_B_high_yield_momentum,
        'C_credit_quality_rotation': signal_C_credit_quality_rotation,
        'D_junk_bond_contrarian': signal_D_junk_bond_contrarian,
        'E_credit_equity_divergence': signal_E_credit_equity_divergence,
        'F_multi_credit_score': signal_F_multi_credit_score,
    }

    results = {
        'backtest_name': 'Credit Spread / Fixed Income Quality Signal',
        'run_date': datetime.now().isoformat(),
        'config': {
            'oot_period': f'{START_DATE} to {END_DATE}',
            'starting_capital': STARTING_CAPITAL,
            'slippage_pct': SLIPPAGE_PCT,
            'commission': COMMISSION,
            'n_permutations': N_PERMUTATIONS,
            'etfs_used': ALL_ETFS,
        },
        'variants': {}
    }

    for name, signal_func in strategies.items():
        print(f"\n{'='*60}")
        print(f"  Variant {name}")
        print(f"{'='*60}")

        eq_df, trades = simulate_strategy(close, signal_func, name)
        metrics = compute_metrics(eq_df, trades, close)

        if metrics is None:
            print(f"  No data for {name}")
            results['variants'][name] = {'error': 'no data'}
            continue

        perm = permutation_test(eq_df, trades, close)
        regime = regime_split(eq_df, close)
        gates = check_gates(metrics, perm, regime)

        print(f"  Sharpe:       {metrics['sharpe']}")
        print(f"  Sortino:      {metrics['sortino']}")
        print(f"  Total Return: {metrics['total_return_pct']}%")
        print(f"  MDD:          {metrics['max_drawdown_pct']}%")
        print(f"  PF:           {metrics['profit_factor']}")
        print(f"  WR:           {metrics['win_rate_pct']}%")
        print(f"  Trades:       {metrics['num_trades']}")
        print(f"  QQQ Corr:     {metrics['qqq_correlation']}")
        print(f"  Perm p-value: {perm['p_value']}")
        print(f"  Regime gap:   {regime['regime_gap']}")
        print(f"  Gates passed: {gates['all_passed']}")

        results['variants'][name] = {
            'metrics': metrics,
            'permutation_test': perm,
            'regime_split': regime,
            'gates': gates,
            'sample_trades': trades[:5] if trades else [],
        }

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")

    passed = []
    for name, v in results['variants'].items():
        if 'gates' in v and v['gates'].get('all_passed', False):
            passed.append(name)
        g = v.get('gates', {})
        m = v.get('metrics', {})
        print(f"  {name}: Sharpe={m.get('sharpe','N/A')}, QQQ_corr={m.get('qqq_correlation','N/A')}, "
              f"passed={g.get('all_passed','N/A')}")

    results['summary'] = {
        'total_variants': len(strategies),
        'passed_all_gates': len(passed),
        'passed_names': passed,
        'best_qqq_decorrelation': None,
    }

    # Find best QQQ decorrelation among passers (or all if none pass)
    candidates = passed if passed else list(results['variants'].keys())
    best_decorr = None
    best_name = None
    for name in candidates:
        v = results['variants'][name]
        corr = v.get('metrics', {}).get('qqq_correlation')
        if corr is not None:
            if best_decorr is None or abs(corr) < abs(best_decorr):
                best_decorr = corr
                best_name = name
    results['summary']['best_qqq_decorrelation'] = {
        'variant': best_name,
        'qqq_correlation': best_decorr
    }

    # Save results
    out_path = Path('/home/jupiter/Lvl3Quant/data/credit_spread_signal_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    return results


if __name__ == '__main__':
    results = main()

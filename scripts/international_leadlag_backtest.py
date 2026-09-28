#!/usr/bin/env python3
"""
International Market Lead-Lag Backtest
6 variants testing cross-market signals uncorrelated to QQQ.
Walk-forward SLIDING window, OOT: 2022-01-01 to 2026-07-29.
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# ── Config ──────────────────────────────────────────────────────────────
STARTING_CAPITAL = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02%
COMMISSION = 0.0
OOT_START = '2022-01-01'
OOT_END = '2026-07-29'
DATA_START = '2020-06-01'  # extra lookback for indicators
PERM_ITERATIONS = 1000

INTL_ETFS = ['EWJ', 'EWG', 'FXI', 'EWZ', 'EEM', 'EFA', 'INDA', 'EWT']
BENCH_TICKERS = ['SPY', 'QQQ', '^VIX']
ALL_TICKERS = INTL_ETFS + BENCH_TICKERS


# ── Data Download ───────────────────────────────────────────────────────
def download_data():
    print("Downloading data...")
    data = {}
    for t in ALL_TICKERS:
        try:
            df = yf.download(t, start=DATA_START, end='2026-07-30', progress=False, auto_adjust=True)
            if len(df) > 0:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                data[t] = df
                print(f"  {t}: {len(df)} rows")
            else:
                print(f"  {t}: NO DATA")
        except Exception as e:
            print(f"  {t}: ERROR {e}")
    return data


# ── Helpers ─────────────────────────────────────────────────────────────
def apply_slippage(price, direction='buy'):
    if direction == 'buy':
        return price * (1 + SLIPPAGE_PCT)
    return price * (1 - SLIPPAGE_PCT)


def calc_metrics(equity_curve, trades_df=None):
    """Calculate Sharpe, Sortino, total return, MDD, PF, WR from equity curve."""
    if len(equity_curve) < 10:
        return {'sharpe': 0, 'sortino': 0, 'total_return_pct': 0,
                'max_drawdown_pct': 0, 'profit_factor': 0, 'win_rate': 0}

    rets = equity_curve.pct_change().dropna()
    rets = rets.replace([np.inf, -np.inf], 0).fillna(0)

    ann = 252
    mu = rets.mean() * ann
    sigma = rets.std() * np.sqrt(ann) if rets.std() > 0 else 1e-9
    sharpe = mu / sigma

    downside = rets[rets < 0].std() * np.sqrt(ann) if len(rets[rets < 0]) > 0 else 1e-9
    sortino = mu / downside

    total_ret = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100

    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    mdd = drawdown.min() * 100

    # PF and WR from daily returns
    wins = rets[rets > 0].sum()
    losses = abs(rets[rets < 0].sum())
    pf = wins / losses if losses > 0 else 999.0

    wr = len(rets[rets > 0]) / len(rets[rets != 0]) * 100 if len(rets[rets != 0]) > 0 else 0

    return {
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'total_return_pct': round(total_ret, 2),
        'max_drawdown_pct': round(mdd, 2),
        'profit_factor': round(min(pf, 99.0), 3),
        'win_rate': round(wr, 2)
    }


def calc_qqq_correlation(strat_returns, qqq_returns):
    """Correlation of strategy daily returns with QQQ."""
    aligned = pd.concat([strat_returns, qqq_returns], axis=1, join='inner').dropna()
    if len(aligned) < 20:
        return 0.0
    return round(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]), 4)


def permutation_test(strat_returns, benchmark_returns=None, n_iter=PERM_ITERATIONS):
    """Permutation test: shuffle the POSITION MASK relative to underlying returns.

    For position-based strategies, we need to shuffle which days we're invested
    vs in cash, not the returns themselves. We identify invested days (nonzero return)
    and create a binary mask, then shuffle that mask against the underlying returns.
    """
    rets = strat_returns.dropna()
    if len(rets) < 20:
        return 1.0

    rets_arr = rets.values
    # Identify invested days (nonzero return = in position)
    invested_mask = rets_arr != 0
    n_invested = invested_mask.sum()

    if n_invested < 5 or n_invested == len(rets_arr):
        # Too few trades or always invested — use simple return shuffle
        actual_sharpe = np.mean(rets_arr) / (np.std(rets_arr) + 1e-12) * np.sqrt(252)
        count = 0
        for _ in range(n_iter):
            shuffled = np.random.permutation(rets_arr)
            s = np.mean(shuffled) / (np.std(shuffled) + 1e-12) * np.sqrt(252)
            if s >= actual_sharpe:
                count += 1
        return round(count / n_iter, 4)

    # Use benchmark (underlying asset) returns if available, else use nonzero strategy rets
    if benchmark_returns is not None:
        underlying = benchmark_returns.reindex(rets.index).fillna(0).values
    else:
        underlying = rets_arr.copy()

    # Actual strategy: invested_mask applied to underlying
    actual_mean = np.mean(rets_arr)
    actual_std = np.std(rets_arr) + 1e-12
    actual_sharpe = actual_mean / actual_std * np.sqrt(252)

    count = 0
    for _ in range(n_iter):
        # Shuffle which days we're invested
        perm_mask = np.random.permutation(invested_mask)
        perm_rets = np.where(perm_mask, underlying, 0.0)
        s = np.mean(perm_rets) / (np.std(perm_rets) + 1e-12) * np.sqrt(252)
        if s >= actual_sharpe:
            count += 1
    return round(count / n_iter, 4)


def regime_split(strat_returns, spy_close):
    """Split returns by bull (SPY > 200-SMA) vs bear."""
    sma200 = spy_close.rolling(200).mean()
    bull_mask = spy_close > sma200
    bear_mask = spy_close <= sma200

    aligned_bull = strat_returns.reindex(bull_mask.index)
    aligned_bear = strat_returns.reindex(bear_mask.index)

    bull_rets = aligned_bull[bull_mask].dropna()
    bear_rets = aligned_bear[bear_mask].dropna()

    def _sharpe(r):
        if len(r) < 10:
            return 0.0
        return round(np.mean(r) / (np.std(r) + 1e-12) * np.sqrt(252), 3)

    bull_sharpe = _sharpe(bull_rets)
    bear_sharpe = _sharpe(bear_rets)

    gap = abs(bull_sharpe - bear_sharpe) / max(abs(bull_sharpe), abs(bear_sharpe), 0.001)

    return {
        'bull_sharpe': bull_sharpe,
        'bear_sharpe': bear_sharpe,
        'regime_gap': round(gap, 3)
    }


def gate_checks(metrics, n_trades, perm_p, regime):
    """Apply validation gates."""
    gates = {
        'sharpe_gt_0.5': metrics['sharpe'] > 0.5,
        'perm_p_lt_0.05': perm_p < 0.05,
        'regime_gap_lt_0.5': regime['regime_gap'] < 0.5,
        'mdd_gt_neg50': metrics['max_drawdown_pct'] > -50,
        'trades_gte_20': n_trades >= 20
    }
    return gates


# ── Variant A: Asia-US Lead-Lag ─────────────────────────────────────────
def variant_a(data):
    print("\n=== Variant A: Asia-US Lead-Lag ===")
    ewj = data['EWJ']['Close'].copy()
    qqq = data['QQQ']['Close'].copy()

    ewj_ret = ewj.pct_change()

    # Align dates
    dates = qqq.loc[OOT_START:OOT_END].index
    equity = STARTING_CAPITAL
    eq_series = {}
    n_trades = 0
    position = 'cash'

    for i, dt in enumerate(dates):
        # Get previous trading day's EWJ return
        prev_dates = ewj_ret.loc[:dt].index
        if len(prev_dates) < 2:
            eq_series[dt] = equity
            continue

        prev_dt = prev_dates[-2]  # yesterday
        prev_ewj_ret = ewj_ret.get(prev_dt, 0)
        if pd.isna(prev_ewj_ret):
            prev_ewj_ret = 0

        # Signal
        if prev_ewj_ret > 0.01:  # EWJ > +1% yesterday
            new_pos = 'long'
        elif prev_ewj_ret < -0.01:  # EWJ < -1% yesterday
            new_pos = 'cash'
        else:
            new_pos = position  # hold current

        # Execute
        if new_pos != position:
            if new_pos == 'long' and position == 'cash':
                buy_price = apply_slippage(qqq.get(dt, 0), 'buy')
                n_trades += 1
            elif new_pos == 'cash' and position == 'long':
                sell_price = apply_slippage(qqq.get(dt, 0), 'sell')
                n_trades += 1
            position = new_pos

        # Mark to market
        if position == 'long' and i > 0:
            prev_price = qqq.get(dates[i-1], qqq.get(dt, 0))
            cur_price = qqq.get(dt, prev_price)
            if prev_price > 0:
                equity *= (cur_price / prev_price)

        eq_series[dt] = equity

    eq_curve = pd.Series(eq_series)
    return eq_curve, n_trades


# ── Variant B: Emerging Market Momentum ─────────────────────────────────
def variant_b(data):
    print("\n=== Variant B: Emerging Market Momentum ===")
    em_tickers = ['EEM', 'EWZ', 'FXI', 'INDA']
    closes = pd.DataFrame({t: data[t]['Close'] for t in em_tickers if t in data})

    dates = closes.loc[OOT_START:OOT_END].index
    equity = STARTING_CAPITAL
    eq_series = {}
    n_trades = 0
    holdings = []
    last_rebal = None

    for dt in dates:
        # Monthly rebalance
        do_rebal = False
        if last_rebal is None:
            do_rebal = True
        elif dt.month != last_rebal.month:
            do_rebal = True

        if do_rebal:
            # 3-month (63 trading days) momentum
            lookback = closes.loc[:dt].tail(63)
            if len(lookback) > 10:
                mom = (lookback.iloc[-1] / lookback.iloc[0] - 1).dropna()
                ranked = mom.sort_values(ascending=False)
                new_holdings = ranked.index[:2].tolist()
                if set(new_holdings) != set(holdings):
                    n_trades += len(new_holdings)
                    # Apply slippage on rebalance
                    equity *= (1 - SLIPPAGE_PCT * 2)  # buy + sell
                holdings = new_holdings
                last_rebal = dt

        # Daily return
        if holdings and dt in closes.index:
            prev_idx = closes.index.get_loc(dt)
            if prev_idx > 0:
                prev_dt = closes.index[prev_idx - 1]
                daily_rets = []
                for h in holdings:
                    if h in closes.columns:
                        p0 = closes.loc[prev_dt, h]
                        p1 = closes.loc[dt, h]
                        if p0 > 0:
                            daily_rets.append(p1 / p0 - 1)
                if daily_rets:
                    avg_ret = np.mean(daily_rets)
                    equity *= (1 + avg_ret)

        eq_series[dt] = equity

    eq_curve = pd.Series(eq_series)
    return eq_curve, n_trades


# ── Variant C: Global Risk-On/Risk-Off Rotation ────────────────────────
def variant_c(data):
    print("\n=== Variant C: Global Risk-On/Risk-Off Rotation ===")
    eem = data['EEM']['Close'].copy()
    efa = data['EFA']['Close'].copy()

    # 20d relative performance
    eem_20d = eem.pct_change(20)
    efa_20d = efa.pct_change(20)
    risk_on = eem_20d > efa_20d

    dates = eem.loc[OOT_START:OOT_END].index
    equity = STARTING_CAPITAL
    eq_series = {}
    n_trades = 0
    position = None
    last_rebal = None

    for dt in dates:
        # Rebalance every 10 trading days
        do_rebal = False
        if last_rebal is None:
            do_rebal = True
        else:
            idx_now = dates.get_loc(dt)
            idx_last = dates.get_loc(last_rebal)
            if idx_now - idx_last >= 10:
                do_rebal = True

        if do_rebal:
            signal = risk_on.get(dt, None)
            if signal is not None and not pd.isna(signal):
                new_pos = 'EEM' if signal else 'EFA'
                if new_pos != position:
                    n_trades += 1
                    equity *= (1 - SLIPPAGE_PCT)
                position = new_pos
                last_rebal = dt

        # Daily return
        if position and dt in dates:
            idx = dates.get_loc(dt)
            if idx > 0:
                prev_dt = dates[idx - 1]
                held = data[position]['Close']
                p0 = held.get(prev_dt, None)
                p1 = held.get(dt, None)
                if p0 is not None and p1 is not None and p0 > 0:
                    equity *= (p1 / p0)

        eq_series[dt] = equity

    eq_curve = pd.Series(eq_series)
    return eq_curve, n_trades


# ── Variant D: China-Tech Divergence ────────────────────────────────────
def variant_d(data):
    print("\n=== Variant D: China-Tech Divergence ===")
    fxi = data['FXI']['Close'].copy()
    qqq = data['QQQ']['Close'].copy()

    # 60d relative perf
    fxi_60d = fxi.pct_change(60)
    qqq_60d = qqq.pct_change(60)
    divergence = fxi_60d - qqq_60d  # negative = FXI underperforming

    dates = fxi.loc[OOT_START:OOT_END].index
    equity = STARTING_CAPITAL
    eq_series = {}
    n_trades = 0
    position = 'cash'
    hold_counter = 0

    for dt in dates:
        idx = dates.get_loc(dt)

        # Check for entry signal
        div = divergence.get(dt, None)
        if position == 'cash' and div is not None and not pd.isna(div):
            if div < -0.10:  # FXI underperforms by >10%
                position = 'long_fxi'
                hold_counter = 20
                n_trades += 1
                equity *= (1 - SLIPPAGE_PCT)

        # Count down hold period
        if position == 'long_fxi':
            hold_counter -= 1
            if hold_counter <= 0:
                position = 'cash'
                n_trades += 1
                equity *= (1 - SLIPPAGE_PCT)

        # Daily return
        if position == 'long_fxi' and idx > 0:
            prev_dt = dates[idx - 1]
            p0 = fxi.get(prev_dt, 0)
            p1 = fxi.get(dt, 0)
            if p0 > 0:
                equity *= (p1 / p0)

        eq_series[dt] = equity

    eq_curve = pd.Series(eq_series)
    return eq_curve, n_trades


# ── Variant E: Global Breadth Signal ───────────────────────────────────
def variant_e(data):
    print("\n=== Variant E: Global Breadth Signal ===")
    breadth_tickers = ['EWJ', 'EWG', 'FXI', 'EWZ', 'INDA', 'EWT']
    eem = data['EEM']['Close'].copy()
    efa = data['EFA']['Close'].copy()

    # Build 50d MA for each
    ma50 = {}
    for t in breadth_tickers:
        if t in data:
            ma50[t] = data[t]['Close'].rolling(50).mean()

    dates = eem.loc[OOT_START:OOT_END].index
    equity = STARTING_CAPITAL
    eq_series = {}
    n_trades = 0
    position = None
    last_rebal = None

    for dt in dates:
        # Weekly rebalance
        do_rebal = False
        if last_rebal is None:
            do_rebal = True
        else:
            idx_now = dates.get_loc(dt)
            idx_last = dates.get_loc(last_rebal)
            if idx_now - idx_last >= 5:
                do_rebal = True

        if do_rebal:
            above_count = 0
            valid_count = 0
            for t in breadth_tickers:
                if t in data and t in ma50:
                    price = data[t]['Close'].get(dt, None)
                    ma = ma50[t].get(dt, None)
                    if price is not None and ma is not None and not pd.isna(price) and not pd.isna(ma):
                        valid_count += 1
                        if price > ma:
                            above_count += 1

            if valid_count >= 4:
                if above_count >= 5:
                    new_pos = 'EEM'
                elif above_count <= 1:
                    new_pos = 'cash'
                else:
                    new_pos = 'EFA'

                if new_pos != position:
                    n_trades += 1
                    if new_pos != 'cash' and position != 'cash':
                        equity *= (1 - SLIPPAGE_PCT * 2)
                    elif new_pos != 'cash':
                        equity *= (1 - SLIPPAGE_PCT)
                    elif position != 'cash':
                        equity *= (1 - SLIPPAGE_PCT)
                    position = new_pos
                    last_rebal = dt

        # Daily return
        if position and position != 'cash' and dt in dates:
            idx = dates.get_loc(dt)
            if idx > 0:
                prev_dt = dates[idx - 1]
                held = data[position]['Close']
                p0 = held.get(prev_dt, None)
                p1 = held.get(dt, None)
                if p0 is not None and p1 is not None and p0 > 0:
                    equity *= (p1 / p0)

        eq_series[dt] = equity

    eq_curve = pd.Series(eq_series)
    return eq_curve, n_trades


# ── Variant F: Taiwan Semi Lead ─────────────────────────────────────────
def variant_f(data):
    print("\n=== Variant F: Taiwan Semi Lead ===")
    ewt = data['EWT']['Close'].copy()
    qqq = data['QQQ']['Close'].copy()

    ewt_5d_mom = ewt.pct_change(5)

    dates = qqq.loc[OOT_START:OOT_END].index
    equity = STARTING_CAPITAL
    eq_series = {}
    n_trades = 0
    position = 'cash'
    hold_counter = 0

    for dt in dates:
        idx = dates.get_loc(dt)

        # If holding, count down
        if position == 'long' and hold_counter > 0:
            hold_counter -= 1
            if hold_counter <= 0:
                position = 'cash'
                n_trades += 1
                equity *= (1 - SLIPPAGE_PCT)

        # Check for new entry (only when cash)
        if position == 'cash':
            mom = ewt_5d_mom.get(dt, None)
            if mom is not None and not pd.isna(mom):
                if mom > 0.03:  # +3% 5d momentum
                    position = 'long'
                    hold_counter = 5
                    n_trades += 1
                    equity *= (1 - SLIPPAGE_PCT)

        # Daily return on QQQ
        if position == 'long' and idx > 0:
            prev_dt = dates[idx - 1]
            p0 = qqq.get(prev_dt, 0)
            p1 = qqq.get(dt, 0)
            if p0 > 0:
                equity *= (p1 / p0)

        eq_series[dt] = equity

    eq_curve = pd.Series(eq_series)
    return eq_curve, n_trades


# ── Main ────────────────────────────────────────────────────────────────
def main():
    np.random.seed(42)
    data = download_data()

    # Check we have all needed data
    missing = [t for t in ALL_TICKERS if t not in data]
    if missing:
        print(f"WARNING: Missing tickers: {missing}")

    qqq_close = data['QQQ']['Close']
    qqq_oot = qqq_close.loc[OOT_START:OOT_END]
    qqq_ret = qqq_oot.pct_change().dropna()

    spy_close = data['SPY']['Close']

    variants = {
        'A_asia_us_leadlag': variant_a,
        'B_em_momentum': variant_b,
        'C_risk_rotation': variant_c,
        'D_china_tech_div': variant_d,
        'E_global_breadth': variant_e,
        'F_taiwan_semi_lead': variant_f,
    }

    results = {
        'meta': {
            'name': 'International Lead-Lag Backtest',
            'category': 'international_leadlag',
            'oot_start': OOT_START,
            'oot_end': OOT_END,
            'starting_capital': STARTING_CAPITAL,
            'slippage_pct': SLIPPAGE_PCT,
            'commission': COMMISSION,
            'perm_iterations': PERM_ITERATIONS,
            'run_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'num_variants': 6
        },
        'strategies': {}
    }

    for name, func in variants.items():
        try:
            eq_curve, n_trades = func(data)
            strat_ret = eq_curve.pct_change().dropna()

            metrics = calc_metrics(eq_curve)
            qqq_corr = calc_qqq_correlation(strat_ret, qqq_ret)

            print(f"  Running permutation test ({PERM_ITERATIONS} iter)...")
            perm_p = permutation_test(strat_ret, benchmark_returns=qqq_ret)

            regime = regime_split(strat_ret, spy_close)
            gates = gate_checks(metrics, n_trades, perm_p, regime)
            all_passed = all(gates.values())

            results['strategies'][name] = {
                'metrics': metrics,
                'n_trades': n_trades,
                'qqq_correlation': qqq_corr,
                'perm_p_value': perm_p,
                'regime': regime,
                'gates': gates,
                'gates_passed': sum(gates.values()),
                'VALIDATED': all_passed,
                'final_equity': round(eq_curve.iloc[-1], 2) if len(eq_curve) > 0 else STARTING_CAPITAL
            }

            status = "PASS" if all_passed else "FAIL"
            print(f"  {name}: Sharpe={metrics['sharpe']}, QQQ_corr={qqq_corr}, "
                  f"perm_p={perm_p}, gates={sum(gates.values())}/5 [{status}]")

        except Exception as e:
            print(f"  ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
            results['strategies'][name] = {
                'metrics': {'sharpe': 0, 'sortino': 0, 'total_return_pct': 0,
                            'max_drawdown_pct': 0, 'profit_factor': 0, 'win_rate': 0},
                'n_trades': 0, 'qqq_correlation': 0, 'perm_p_value': 1.0,
                'regime': {'bull_sharpe': 0, 'bear_sharpe': 0, 'regime_gap': 0},
                'gates': {}, 'gates_passed': 0, 'VALIDATED': False,
                'error': str(e)
            }

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    validated = sum(1 for s in results['strategies'].values() if s.get('VALIDATED'))
    print(f"Validated: {validated}/{len(results['strategies'])}")
    for name, s in results['strategies'].items():
        m = s['metrics']
        tag = "PASS" if s.get('VALIDATED') else "FAIL"
        print(f"  {name}: Sharpe={m['sharpe']}, Sortino={m['sortino']}, "
              f"Ret={m['total_return_pct']}%, MDD={m['max_drawdown_pct']}%, "
              f"QQQ_corr={s['qqq_correlation']}, perm_p={s['perm_p_value']} [{tag}]")

    # Save
    out_path = '/home/jupiter/Lvl3Quant/data/international_leadlag_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    return results


if __name__ == '__main__':
    main()

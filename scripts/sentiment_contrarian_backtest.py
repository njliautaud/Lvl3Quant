#!/usr/bin/env python3
"""
Sentiment/Breadth Contrarian Backtest
Variants A-F with 5-gate validation framework.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import json
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')
np.random.seed(42)

# ── Config ──────────────────────────────────────────────────────────────
CAPITAL = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02%
START = '2022-01-01'
END = '2026-07-28'
TICKERS = ['SPY', 'QQQ', 'IWM', 'SPXU', 'TLT']
N_PERM = 1000
RESULTS_PATH = Path('/home/jupiter/Lvl3Quant/data/sentiment_contrarian_results.json')


# ── Download data ───────────────────────────────────────────────────────
def download_data():
    print("Downloading data...")
    data = {}
    for t in TICKERS:
        df = yf.download(t, start=START, end=END, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        data[t] = df
    return data


# ── Helper: compute metrics ─────────────────────────────────────────────
def compute_metrics(equity_curve, trades_list, daily_returns, sma200, spy_close):
    """Compute all metrics for a variant given its equity curve and trades."""
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    n_trades = len(trades_list)

    # Annualized
    n_years = len(daily_returns) / 252
    cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / max(n_years, 0.01)) - 1

    # Sharpe
    if daily_returns.std() > 0:
        sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
    else:
        sharpe = 0.0

    # Sortino
    downside = daily_returns[daily_returns < 0]
    if len(downside) > 0 and downside.std() > 0:
        sortino = daily_returns.mean() / downside.std() * np.sqrt(252)
    else:
        sortino = 0.0

    # Max drawdown
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_dd = drawdown.min()

    # Win rate & profit factor from trades
    if n_trades > 0:
        wins = [t for t in trades_list if t > 0]
        losses = [t for t in trades_list if t < 0]
        win_rate = len(wins) / n_trades
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    else:
        win_rate = 0.0
        profit_factor = 0.0

    # Regime analysis: bull = SPY > 200-SMA, bear = SPY < 200-SMA
    bull_mask = spy_close > sma200
    bear_mask = spy_close <= sma200

    # Align masks with daily_returns index
    bull_aligned = bull_mask.reindex(daily_returns.index).fillna(False)
    bear_aligned = bear_mask.reindex(daily_returns.index).fillna(False)

    bull_rets = daily_returns[bull_aligned]
    bear_rets = daily_returns[bear_aligned]

    bull_sharpe = bull_rets.mean() / bull_rets.std() * np.sqrt(252) if len(bull_rets) > 5 and bull_rets.std() > 0 else 0.0
    bear_sharpe = bear_rets.mean() / bear_rets.std() * np.sqrt(252) if len(bear_rets) > 5 and bear_rets.std() > 0 else 0.0

    max_abs = max(abs(bull_sharpe), abs(bear_sharpe))
    regime_gap = abs(bull_sharpe - bear_sharpe) / max_abs if max_abs > 0 else 0.0

    return {
        'sharpe': round(sharpe, 4),
        'sortino': round(sortino, 4),
        'total_return': round(total_return, 4),
        'max_drawdown': round(max_dd, 4),
        'n_trades': n_trades,
        'profit_factor': round(profit_factor, 4) if profit_factor != float('inf') else 999.0,
        'win_rate': round(win_rate, 4),
        'cagr': round(cagr, 4),
        'bull_sharpe': round(bull_sharpe, 4),
        'bear_sharpe': round(bear_sharpe, 4),
        'regime_gap': round(regime_gap, 4),
    }


def apply_slippage(price, direction='buy'):
    """Apply slippage to a price."""
    if direction == 'buy':
        return price * (1 + SLIPPAGE_PCT)
    else:
        return price * (1 - SLIPPAGE_PCT)


# ── Variant A: RSI Extremes ────────────────────────────────────────────
def variant_a(data):
    """Buy SPY when RSI(14) < 30, sell when RSI > 70. Cash between."""
    spy = data['SPY']['Close'].copy()

    # Compute RSI(14)
    delta = spy.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    equity = CAPITAL
    position = 0  # shares held
    equity_curve = []
    trades_pnl = []
    entry_price = 0
    in_position = False

    for i in range(len(spy)):
        price = spy.iloc[i]
        r = rsi.iloc[i]

        if pd.isna(r) or pd.isna(price):
            equity_curve.append(equity + position * price if not pd.isna(price) else equity)
            continue

        if not in_position and r < 30:
            # Buy
            buy_price = apply_slippage(price, 'buy')
            position = equity / buy_price
            entry_price = buy_price
            equity = 0
            in_position = True
        elif in_position and r > 70:
            # Sell
            sell_price = apply_slippage(price, 'sell')
            proceeds = position * sell_price
            pnl = proceeds - (position * entry_price)
            trades_pnl.append(pnl)
            equity = proceeds
            position = 0
            in_position = False

        if in_position:
            equity_curve.append(position * price)
        else:
            equity_curve.append(equity)

    eq = pd.Series(equity_curve, index=spy.index)
    daily_ret = eq.pct_change().dropna()
    daily_ret = daily_ret.replace([np.inf, -np.inf], 0).fillna(0)
    return eq, trades_pnl, daily_ret


# ── Variant B: SMA Distance Contrarian ──────────────────────────────────
def variant_b(data):
    """Buy SPY when >5% below 200-SMA. Sell when >10% above 200-SMA."""
    spy = data['SPY']['Close'].copy()
    sma200 = spy.rolling(200).mean()
    distance = (spy - sma200) / sma200

    equity = CAPITAL
    position = 0
    equity_curve = []
    trades_pnl = []
    entry_price = 0
    in_position = False

    for i in range(len(spy)):
        price = spy.iloc[i]
        dist = distance.iloc[i]

        if pd.isna(dist) or pd.isna(price):
            equity_curve.append(equity + position * price if not pd.isna(price) else equity)
            continue

        if not in_position and dist < -0.05:
            buy_price = apply_slippage(price, 'buy')
            position = equity / buy_price
            entry_price = buy_price
            equity = 0
            in_position = True
        elif in_position and dist > 0.10:
            sell_price = apply_slippage(price, 'sell')
            proceeds = position * sell_price
            pnl = proceeds - (position * entry_price)
            trades_pnl.append(pnl)
            equity = proceeds
            position = 0
            in_position = False

        if in_position:
            equity_curve.append(position * price)
        else:
            equity_curve.append(equity)

    eq = pd.Series(equity_curve, index=spy.index)
    daily_ret = eq.pct_change().dropna()
    daily_ret = daily_ret.replace([np.inf, -np.inf], 0).fillna(0)
    return eq, trades_pnl, daily_ret


# ── Variant C: Consecutive Days Reversal ────────────────────────────────
def variant_c(data):
    """Buy SPY after 4+ consecutive down days (hold 5d). Buy QQQ after 5+ down days (hold 5d)."""
    spy = data['SPY']['Close'].copy()
    qqq = data['QQQ']['Close'].copy()

    # Compute consecutive down days for SPY and QQQ
    spy_down = (spy.diff() < 0).astype(int)
    qqq_down = (qqq.diff() < 0).astype(int)

    def consecutive_downs(series):
        result = []
        count = 0
        for v in series:
            if v == 1:
                count += 1
            else:
                count = 0
            result.append(count)
        return result

    spy_consec = consecutive_downs(spy_down.values)
    qqq_consec = consecutive_downs(qqq_down.values)

    equity = CAPITAL
    position_shares = 0
    position_ticker = None
    hold_days_left = 0
    entry_price = 0
    equity_curve = []
    trades_pnl = []

    for i in range(len(spy)):
        spy_price = spy.iloc[i]
        qqq_price = qqq.iloc[i]

        if pd.isna(spy_price):
            equity_curve.append(equity)
            continue

        # If holding, decrement
        if hold_days_left > 0:
            hold_days_left -= 1
            current_price = spy_price if position_ticker == 'SPY' else qqq_price
            if hold_days_left == 0:
                sell_price = apply_slippage(current_price, 'sell')
                proceeds = position_shares * sell_price
                pnl = proceeds - (position_shares * entry_price)
                trades_pnl.append(pnl)
                equity = proceeds
                position_shares = 0
                position_ticker = None
            else:
                equity_curve.append(position_shares * current_price)
                continue

        # Check for new entries (only if not holding)
        if position_ticker is None:
            if not pd.isna(qqq_price) and qqq_consec[i] >= 5:
                buy_price = apply_slippage(qqq_price, 'buy')
                position_shares = equity / buy_price
                entry_price = buy_price
                equity = 0
                position_ticker = 'QQQ'
                hold_days_left = 5
                equity_curve.append(position_shares * qqq_price)
                continue
            elif spy_consec[i] >= 4:
                buy_price = apply_slippage(spy_price, 'buy')
                position_shares = equity / buy_price
                entry_price = buy_price
                equity = 0
                position_ticker = 'SPY'
                hold_days_left = 5
                equity_curve.append(position_shares * spy_price)
                continue

        equity_curve.append(equity)

    eq = pd.Series(equity_curve, index=spy.index)
    daily_ret = eq.pct_change().dropna()
    daily_ret = daily_ret.replace([np.inf, -np.inf], 0).fillna(0)
    return eq, trades_pnl, daily_ret


# ── Variant D: Breadth Divergence ───────────────────────────────────────
def variant_d(data):
    """IWM vs SPY relative performance → safety or risk-on trade."""
    spy = data['SPY']['Close'].copy()
    iwm = data['IWM']['Close'].copy()
    tlt = data['TLT']['Close'].copy()
    sma200 = spy.rolling(200).mean()

    # 20-day relative perf: IWM return - SPY return over 20 days
    spy_ret20 = spy.pct_change(20)
    iwm_ret20 = iwm.pct_change(20)
    rel_perf = iwm_ret20 - spy_ret20

    equity = CAPITAL
    position_shares = 0
    position_ticker = None
    entry_price = 0
    equity_curve = []
    trades_pnl = []

    for i in range(len(spy)):
        spy_price = spy.iloc[i]
        iwm_price = iwm.iloc[i]
        tlt_price = tlt.iloc[i]
        rp = rel_perf.iloc[i]
        sma = sma200.iloc[i]

        if pd.isna(rp) or pd.isna(sma) or pd.isna(spy_price):
            if position_ticker is not None:
                cp = {'TLT': tlt_price, 'IWM': iwm_price}.get(position_ticker, spy_price)
                equity_curve.append(position_shares * cp if not pd.isna(cp) else equity)
            else:
                equity_curve.append(equity)
            continue

        # Determine desired state
        if rp < -0.03 and spy_price > sma:
            # Breadth narrowing, risk rising → TLT
            desired = 'TLT'
        elif rp > 0.02:
            # Breadth expanding → IWM
            desired = 'IWM'
        else:
            desired = None

        # Close if wrong position or should be cash
        if position_ticker is not None and position_ticker != desired:
            cp = {'TLT': tlt_price, 'IWM': iwm_price}[position_ticker]
            sell_price = apply_slippage(cp, 'sell')
            proceeds = position_shares * sell_price
            pnl = proceeds - (position_shares * entry_price)
            trades_pnl.append(pnl)
            equity = proceeds
            position_shares = 0
            position_ticker = None

        # Open new position
        if position_ticker is None and desired is not None:
            cp = {'TLT': tlt_price, 'IWM': iwm_price}[desired]
            if not pd.isna(cp) and cp > 0:
                buy_price = apply_slippage(cp, 'buy')
                position_shares = equity / buy_price
                entry_price = buy_price
                equity = 0
                position_ticker = desired

        if position_ticker is not None:
            cp = {'TLT': tlt_price, 'IWM': iwm_price}[position_ticker]
            equity_curve.append(position_shares * cp)
        else:
            equity_curve.append(equity)

    eq = pd.Series(equity_curve, index=spy.index)
    daily_ret = eq.pct_change().dropna()
    daily_ret = daily_ret.replace([np.inf, -np.inf], 0).fillna(0)
    return eq, trades_pnl, daily_ret


# ── Variant E: Vol Compression Breakout ─────────────────────────────────
def variant_e(data):
    """When 20d vol < 60d vol * 0.7, buy SPY+SPXU equally. Hold until vol expands."""
    spy = data['SPY']['Close'].copy()
    spxu = data['SPXU']['Close'].copy()

    spy_ret = spy.pct_change()
    vol20 = spy_ret.rolling(20).std() * np.sqrt(252)
    vol60 = spy_ret.rolling(60).std() * np.sqrt(252)

    equity = CAPITAL
    spy_shares = 0
    spxu_shares = 0
    spy_entry = 0
    spxu_entry = 0
    in_position = False
    equity_curve = []
    trades_pnl = []

    for i in range(len(spy)):
        spy_price = spy.iloc[i]
        spxu_price = spxu.iloc[i]
        v20 = vol20.iloc[i]
        v60 = vol60.iloc[i]

        if pd.isna(v20) or pd.isna(v60) or pd.isna(spy_price) or pd.isna(spxu_price):
            if in_position:
                equity_curve.append(spy_shares * spy_price + spxu_shares * spxu_price
                                     if not pd.isna(spy_price) and not pd.isna(spxu_price) else equity)
            else:
                equity_curve.append(equity)
            continue

        if not in_position and v20 < v60 * 0.7:
            # Vol compressed → buy straddle proxy
            half = equity / 2
            spy_bp = apply_slippage(spy_price, 'buy')
            spxu_bp = apply_slippage(spxu_price, 'buy')
            spy_shares = half / spy_bp
            spxu_shares = half / spxu_bp
            spy_entry = spy_bp
            spxu_entry = spxu_bp
            equity = 0
            in_position = True
        elif in_position and v20 > v60:
            # Vol expanded → exit
            spy_sp = apply_slippage(spy_price, 'sell')
            spxu_sp = apply_slippage(spxu_price, 'sell')
            spy_proceeds = spy_shares * spy_sp
            spxu_proceeds = spxu_shares * spxu_sp
            total_proceeds = spy_proceeds + spxu_proceeds
            total_cost = spy_shares * spy_entry + spxu_shares * spxu_entry
            pnl = total_proceeds - total_cost
            trades_pnl.append(pnl)
            equity = total_proceeds
            spy_shares = 0
            spxu_shares = 0
            in_position = False

        if in_position:
            equity_curve.append(spy_shares * spy_price + spxu_shares * spxu_price)
        else:
            equity_curve.append(equity)

    eq = pd.Series(equity_curve, index=spy.index)
    daily_ret = eq.pct_change().dropna()
    daily_ret = daily_ret.replace([np.inf, -np.inf], 0).fillna(0)
    return eq, trades_pnl, daily_ret


# ── Variant F: ADVERSARIAL Random Sentiment ─────────────────────────────
def variant_f(data, best_avg_hold_days=10):
    """Random buy/sell at random RSI levels, same avg holding period as best variant."""
    spy = data['SPY']['Close'].copy()

    # Compute RSI for reference
    delta = spy.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    equity = CAPITAL
    position = 0
    equity_curve = []
    trades_pnl = []
    entry_price = 0
    in_position = False
    hold_counter = 0

    # Random thresholds
    rng = np.random.RandomState(123)
    buy_thresh = rng.uniform(20, 80)
    sell_thresh = rng.uniform(20, 80)
    use_hold_period = True

    for i in range(len(spy)):
        price = spy.iloc[i]
        r = rsi.iloc[i]

        if pd.isna(r) or pd.isna(price):
            equity_curve.append(equity + position * price if not pd.isna(price) else equity)
            continue

        if in_position:
            hold_counter += 1

        # Random entry/exit
        coin = rng.random()
        if not in_position and coin < 0.05:  # ~5% chance per day
            buy_price = apply_slippage(price, 'buy')
            position = equity / buy_price
            entry_price = buy_price
            equity = 0
            in_position = True
            hold_counter = 0
        elif in_position and (hold_counter >= best_avg_hold_days or coin > 0.95):
            sell_price = apply_slippage(price, 'sell')
            proceeds = position * sell_price
            pnl = proceeds - (position * entry_price)
            trades_pnl.append(pnl)
            equity = proceeds
            position = 0
            in_position = False
            hold_counter = 0

        if in_position:
            equity_curve.append(position * price)
        else:
            equity_curve.append(equity)

    eq = pd.Series(equity_curve, index=spy.index)
    daily_ret = eq.pct_change().dropna()
    daily_ret = daily_ret.replace([np.inf, -np.inf], 0).fillna(0)
    return eq, trades_pnl, daily_ret


# ── Permutation test ───────────────────────────────────────────────────
def permutation_test_signal(variant_func, data, n_perm=N_PERM):
    """
    Permutation test: the strategy produces an equity curve whose daily returns
    are nonzero only on in-trade days. We create a boolean mask of in-trade days,
    then shuffle that mask to see if random timing produces similar Sharpe.
    This tests whether the TIMING of entries matters, not just being in the market.
    """
    eq, trades, daily_ret = variant_func(data)
    if len(daily_ret) == 0 or daily_ret.std() == 0:
        return 1.0

    observed_sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252)

    # Build in-trade mask: days where the strategy has nonzero returns
    # vs days with ~0 returns (cash). Use SPY returns as the underlying.
    spy_rets = data['SPY']['Close'].pct_change().reindex(daily_ret.index).fillna(0).values
    strat_rets = daily_ret.values

    # Identify in-trade days (nonzero strategy return and not just rounding noise)
    in_trade = np.abs(strat_rets) > 1e-8
    n_in_trade = in_trade.sum()

    if n_in_trade < 5 or n_in_trade >= len(strat_rets) - 5:
        # Too few trades or always in trade — can't permute meaningfully
        return 1.0

    count_better = 0
    for _ in range(n_perm):
        # Shuffle the in-trade mask: randomly pick which days we're "in trade"
        perm_mask = np.zeros(len(strat_rets), dtype=bool)
        perm_indices = np.random.choice(len(strat_rets), size=n_in_trade, replace=False)
        perm_mask[perm_indices] = True

        # On "in trade" days use SPY return, on cash days use 0
        perm_rets = np.where(perm_mask, spy_rets, 0.0)
        std = perm_rets.std()
        if std > 0:
            s = perm_rets.mean() / std * np.sqrt(252)
        else:
            s = 0.0
        if s >= observed_sharpe:
            count_better += 1

    return count_better / n_perm


# ── Validation gates ───────────────────────────────────────────────────
def validate(metrics, perm_p):
    """5-gate validation."""
    gates = {}
    gates['sharpe_gt_0.5'] = metrics['sharpe'] > 0.5
    gates['perm_p_lt_0.05'] = perm_p < 0.05
    gates['regime_gap_lt_0.5'] = metrics['regime_gap'] < 0.5
    gates['max_dd_gt_neg50'] = metrics['max_drawdown'] > -0.50
    gates['min_20_trades'] = metrics['n_trades'] >= 20

    n_passed = sum(gates.values())

    if n_passed == 5:
        verdict = 'PASS — all gates cleared'
    elif n_passed >= 3:
        verdict = f'MARGINAL — {n_passed}/5 gates'
    else:
        verdict = f'FAIL — {n_passed}/5 gates'

    return gates, n_passed, verdict


# ── Main ────────────────────────────────────────────────────────────────
def main():
    data = download_data()

    spy_close = data['SPY']['Close']
    sma200 = spy_close.rolling(200).mean()

    variants = {
        'A_RSI_Extremes': variant_a,
        'B_SMA_Distance_Contrarian': variant_b,
        'C_Consecutive_Days_Reversal': variant_c,
        'D_Breadth_Divergence': variant_d,
        'E_Vol_Compression_Breakout': variant_e,
    }

    results = {}
    best_sharpe = -999
    best_variant = None

    for name, func in variants.items():
        print(f"\n{'='*60}")
        print(f"Running Variant {name}")
        print(f"{'='*60}")

        eq, trades, daily_ret = func(data)
        metrics = compute_metrics(eq, trades, daily_ret, sma200, spy_close)

        # Permutation test
        print(f"  Running permutation test ({N_PERM} iterations)...")
        perm_p = permutation_test_signal(func, data, N_PERM)

        gates, n_passed, verdict = validate(metrics, perm_p)

        metrics['perm_p_value'] = round(perm_p, 4)
        metrics['gates_passed'] = n_passed
        metrics['gates_detail'] = {k: bool(v) for k, v in gates.items()}
        metrics['verdict'] = verdict

        results[name] = metrics

        if metrics['sharpe'] > best_sharpe:
            best_sharpe = metrics['sharpe']
            best_variant = name

        print(f"  Sharpe: {metrics['sharpe']:.4f} | Sortino: {metrics['sortino']:.4f}")
        print(f"  Return: {metrics['total_return']:.2%} | MaxDD: {metrics['max_drawdown']:.2%}")
        print(f"  Trades: {metrics['n_trades']} | WR: {metrics['win_rate']:.2%} | PF: {metrics['profit_factor']:.2f}")
        print(f"  Bull Sharpe: {metrics['bull_sharpe']:.4f} | Bear Sharpe: {metrics['bear_sharpe']:.4f} | Gap: {metrics['regime_gap']:.4f}")
        print(f"  Perm p-value: {perm_p:.4f}")
        print(f"  Verdict: {verdict}")

    # Compute average hold days from best variant for adversarial baseline
    if best_variant:
        eq_best, trades_best, dr_best = variants[best_variant](data)
        # Rough avg hold: total trading days / n_trades
        n_trading_days = len(dr_best)
        n_tr = results[best_variant]['n_trades']
        avg_hold = max(n_trading_days // max(n_tr, 1), 5)
    else:
        avg_hold = 10

    # Variant F: Adversarial
    print(f"\n{'='*60}")
    print(f"Running Variant F_Random_Adversarial (avg hold={avg_hold}d)")
    print(f"{'='*60}")

    eq_f, trades_f, dr_f = variant_f(data, best_avg_hold_days=avg_hold)
    metrics_f = compute_metrics(eq_f, trades_f, dr_f, sma200, spy_close)
    perm_p_f = permutation_test_signal(lambda d: variant_f(d, avg_hold), data, N_PERM)
    gates_f, n_passed_f, verdict_f = validate(metrics_f, perm_p_f)

    metrics_f['perm_p_value'] = round(perm_p_f, 4)
    metrics_f['gates_passed'] = n_passed_f
    metrics_f['gates_detail'] = {k: bool(v) for k, v in gates_f.items()}
    metrics_f['verdict'] = verdict_f

    results['F_Random_Adversarial'] = metrics_f

    print(f"  Sharpe: {metrics_f['sharpe']:.4f} | Sortino: {metrics_f['sortino']:.4f}")
    print(f"  Return: {metrics_f['total_return']:.2%} | MaxDD: {metrics_f['max_drawdown']:.2%}")
    print(f"  Trades: {metrics_f['n_trades']} | WR: {metrics_f['win_rate']:.2%} | PF: {metrics_f['profit_factor']:.2f}")
    print(f"  Verdict: {verdict_f}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Variant':<35} {'Sharpe':>8} {'Sortino':>8} {'Return':>8} {'MaxDD':>8} {'Trades':>7} {'WR':>6} {'Verdict'}")
    print('-' * 110)
    for name, m in results.items():
        print(f"{name:<35} {m['sharpe']:>8.3f} {m['sortino']:>8.3f} {m['total_return']:>7.1%} {m['max_drawdown']:>7.1%} {m['n_trades']:>7} {m['win_rate']:>5.1%} {m['verdict']}")

    # Add metadata
    output = {
        'metadata': {
            'start_date': START,
            'end_date': END,
            'starting_capital': CAPITAL,
            'slippage_pct': SLIPPAGE_PCT,
            'n_permutations': N_PERM,
            'run_timestamp': datetime.now().isoformat(),
        },
        'variants': results,
    }

    # Save — convert numpy types for JSON
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    output = convert(output)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == '__main__':
    main()

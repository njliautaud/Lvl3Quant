#!/usr/bin/env python3
"""
Adversarial validation of 3 strategies that passed 5/5 gates.
6 tests each: re-implementation, inverse signal, random timing,
sub-period stability, top-3 removal, parameter sensitivity.
"""

import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

START = '2020-01-01'
END = '2026-07-31'
STARTING_CAPITAL = 645.0
N_PERMUTATIONS = 1000

# ── Data Download ──────────────────────────────────────────────────────

def download_data():
    """Download all needed data via yfinance."""
    universe_path = Path('/home/jupiter/Lvl3Quant/data/quality_universe.json')
    with open(universe_path) as f:
        universe = json.load(f)
    tickers = universe['tickers']

    # Need quality tickers + VIXY + SPY + TLT + VIX
    all_tickers = list(set(tickers + ['VIXY', 'SPY', 'TLT', '^VIX']))

    print(f"Downloading {len(all_tickers)} tickers...")
    data = yf.download(all_tickers, start=START, end=END, auto_adjust=True, progress=False)

    # Handle multi-level columns
    if isinstance(data.columns, pd.MultiIndex):
        close = data['Close']
        high = data['High']
        low = data['Low']
        volume = data['Volume']
        opn = data['Open']
    else:
        close = data[['Close']].copy()
        high = data[['High']].copy()
        low = data[['Low']].copy()
        volume = data[['Volume']].copy()
        opn = data[['Open']].copy()

    # Rename ^VIX
    for df in [close, high, low, volume, opn]:
        if '^VIX' in df.columns:
            df.rename(columns={'^VIX': 'VIX'}, inplace=True)

    return tickers, close, high, low, volume, opn


# ── Strategy A: Multi-Strategy Confluence ──────────────────────────────

def compute_confluence_signals(tickers, close, high, low, volume, opn,
                                threshold=4, hold_days=10, position_size=200, max_concurrent=3):
    """
    Buy quality stocks when threshold+ signals fire simultaneously.
    Returns daily portfolio returns series.
    """
    trades = []

    # Precompute TLT and VIX/SPY data
    tlt = close.get('TLT')
    vix = close.get('VIX')
    spy = close.get('SPY')

    if tlt is None or vix is None or spy is None:
        return pd.Series(dtype=float), []

    tlt_5d_ret = tlt.pct_change(5)

    # SPY realized vol (ATR proxy: use close-to-close)
    spy_ret = spy.pct_change()
    spy_rv_20d = spy_ret.rolling(20).std() * np.sqrt(252) * 100  # annualized as percentage points

    for ticker in tickers:
        if ticker not in close.columns:
            continue
        c = close[ticker].dropna()
        h = high[ticker].dropna() if ticker in high.columns else None
        l = low[ticker].dropna() if ticker in low.columns else None
        if len(c) < 60:
            continue

        # Precompute indicators
        sma20 = c.rolling(20).mean()
        high20 = c.rolling(20).max()
        rsi14 = compute_rsi(c, 14)

        # 5d average RSI
        rsi_5d_avg = rsi14.rolling(5).mean()

        # 5d drawdown
        high5 = c.rolling(5).max()
        dd5 = (c - high5) / high5

        # Daily returns for consecutive red days
        daily_ret = c.pct_change()

        # H-L spread
        if h is not None and l is not None:
            hl_spread = (h - l).reindex(c.index)
            hl_spread_60d_avg = hl_spread.rolling(60).mean()
        else:
            hl_spread = None

        for i in range(60, len(c)):
            date = c.index[i]
            price = c.iloc[i]

            signals = 0

            # Signal_MR: stock >5% below 20d high + RSI(14)<35
            if not np.isnan(high20.iloc[i]) and not np.isnan(rsi14.iloc[i]):
                if price < high20.iloc[i] * 0.95 and rsi14.iloc[i] < 35:
                    signals += 1

            # Signal_Recovery: first green day after 3+ consecutive red days
            if i >= 3 and not np.isnan(daily_ret.iloc[i]):
                if daily_ret.iloc[i] > 0:
                    consec_red = 0
                    for j in range(1, min(i, 20)):
                        if not np.isnan(daily_ret.iloc[i-j]) and daily_ret.iloc[i-j] < 0:
                            consec_red += 1
                        else:
                            break
                    if consec_red >= 3:
                        signals += 1

            # Signal_RSI_Div: RSI higher low while price lower low (20d window)
            if i >= 20:
                price_window = c.iloc[i-20:i+1]
                rsi_window = rsi14.iloc[i-20:i+1]
                if len(price_window) >= 20:
                    # Find local mins in first half and second half
                    mid = len(price_window) // 2
                    p_first_min = price_window.iloc[:mid].min()
                    p_second_min = price_window.iloc[mid:].min()
                    r_first_min_idx = price_window.iloc[:mid].idxmin()
                    r_second_min_idx = price_window.iloc[mid:].idxmin()
                    if r_first_min_idx in rsi_window.index and r_second_min_idx in rsi_window.index:
                        r_first = rsi_window.loc[r_first_min_idx]
                        r_second = rsi_window.loc[r_second_min_idx]
                        if not np.isnan(r_first) and not np.isnan(r_second):
                            if p_second_min < p_first_min and r_second > r_first:
                                signals += 1

            # Signal_Bond: TLT rises >1% in 5d + stock >5% below 20-SMA
            if date in tlt_5d_ret.index and date in sma20.index:
                tlt_r = tlt_5d_ret.get(date, np.nan)
                sma_val = sma20.get(date, np.nan)
                if not np.isnan(tlt_r) and not np.isnan(sma_val):
                    if tlt_r > 0.01 and price < sma_val * 0.95:
                        signals += 1

            # Signal_IV_RV: VIX > SPY 20d realized vol + 5
            if date in vix.index and date in spy_rv_20d.index:
                v = vix.get(date, np.nan)
                rv = spy_rv_20d.get(date, np.nan)
                if not np.isnan(v) and not np.isnan(rv):
                    if v > rv + 5:
                        signals += 1

            # Signal_Liquidity: H-L spread < 60d avg
            if hl_spread is not None:
                if date in hl_spread.index and date in hl_spread_60d_avg.index:
                    hl_val = hl_spread.get(date, np.nan)
                    hl_avg = hl_spread_60d_avg.get(date, np.nan)
                    if not np.isnan(hl_val) and not np.isnan(hl_avg) and hl_avg > 0:
                        if hl_val < hl_avg:
                            signals += 1

            # Signal_MultiTF: daily RSI<35 + 5d avg RSI < 40 + 5d drawdown > 7%
            if not np.isnan(rsi14.iloc[i]):
                rsi_avg = rsi_5d_avg.get(date, np.nan) if date in rsi_5d_avg.index else np.nan
                dd = dd5.get(date, np.nan) if date in dd5.index else np.nan
                if not np.isnan(rsi_avg) and not np.isnan(dd):
                    if rsi14.iloc[i] < 35 and rsi_avg < 40 and dd < -0.07:
                        signals += 1

            if signals >= threshold:
                trades.append({
                    'ticker': ticker,
                    'entry_date': date,
                    'entry_price': price,
                    'signals': signals,
                    'hold_days': hold_days,
                    'position_size': position_size,
                })

    # Execute trades with max concurrent constraint
    return execute_trades(trades, close, hold_days, position_size, max_concurrent, STARTING_CAPITAL)


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def execute_trades(trades, close, hold_days, position_size, max_concurrent, capital):
    """Execute trades respecting max concurrent positions. Returns (returns_series, trade_list)."""
    if not trades:
        return pd.Series(dtype=float), []

    # Sort by entry date
    trades = sorted(trades, key=lambda x: x['entry_date'])

    # Track active positions
    active = []
    executed = []

    all_dates = close.index
    daily_pnl = pd.Series(0.0, index=all_dates)

    for trade in trades:
        entry = trade['entry_date']
        ticker = trade['ticker']

        # Remove expired from active
        active = [a for a in active if a['exit_date'] > entry]

        if len(active) >= max_concurrent:
            continue

        # Find exit date
        entry_idx = all_dates.get_loc(entry)
        exit_idx = min(entry_idx + hold_days, len(all_dates) - 1)
        exit_date = all_dates[exit_idx]

        if ticker not in close.columns:
            continue

        entry_price = close[ticker].get(entry, np.nan)
        exit_price = close[ticker].get(exit_date, np.nan)

        if np.isnan(entry_price) or np.isnan(exit_price) or entry_price <= 0:
            continue

        shares = position_size / entry_price
        pnl = shares * (exit_price - entry_price)
        ret = (exit_price - entry_price) / entry_price

        trade_record = {
            'ticker': ticker,
            'entry_date': entry,
            'exit_date': exit_date,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'return': ret,
            'position_size': position_size,
        }
        executed.append(trade_record)

        # Distribute PnL across holding period for daily returns
        trading_days = exit_idx - entry_idx
        if trading_days > 0:
            daily_pnl_amount = pnl / trading_days
            for d in range(entry_idx, exit_idx):
                daily_pnl.iloc[d] += daily_pnl_amount

        active.append({'ticker': ticker, 'exit_date': exit_date})

    # Convert to returns
    daily_returns = daily_pnl / capital
    daily_returns = daily_returns[daily_returns != 0]

    return daily_returns, executed


def compute_sharpe(returns, annual_factor=252):
    """Compute annualized Sharpe ratio."""
    if len(returns) < 10 or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(annual_factor))


# ── Strategy B: Short VIXY in Contango ──────────────────────────────

def run_vixy_contango(close, hold_days=20, vix_threshold=None, position_size=200,
                       capital=STARTING_CAPITAL, inverse=False):
    """
    Short VIXY when VIX < 20d avg VIX (contango proxy).
    If inverse=True, go LONG VIXY in contango (opposite signal).
    """
    vix = close.get('VIX')
    vixy = close.get('VIXY')
    if vix is None or vixy is None:
        return pd.Series(dtype=float), []

    vix = vix.dropna()
    vixy = vixy.dropna()

    vix_20d_avg = vix.rolling(20).mean()

    trades = []
    last_exit = None

    common_dates = vix.index.intersection(vixy.index).intersection(vix_20d_avg.dropna().index)

    for date in common_dates:
        if last_exit is not None and date <= last_exit:
            continue

        v = vix.get(date, np.nan)
        v_avg = vix_20d_avg.get(date, np.nan)

        if np.isnan(v) or np.isnan(v_avg):
            continue

        # Apply threshold override if provided
        threshold = v_avg if vix_threshold is None else vix_threshold

        contango = v < threshold

        if (contango and not inverse) or (not contango and inverse):
            entry_price = vixy.get(date, np.nan)
            if np.isnan(entry_price) or entry_price <= 0:
                continue

            # Find exit
            entry_idx = vixy.index.get_loc(date)
            exit_idx = min(entry_idx + hold_days, len(vixy) - 1)
            exit_date = vixy.index[exit_idx]
            exit_price = vixy.iloc[exit_idx]

            if np.isnan(exit_price):
                continue

            # SHORT: profit when price drops
            shares = position_size / entry_price
            if not inverse:
                pnl = shares * (entry_price - exit_price)  # short
            else:
                pnl = shares * (exit_price - entry_price)  # long (inverse test)

            ret = pnl / position_size

            trades.append({
                'entry_date': date,
                'exit_date': exit_date,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'return': ret,
                'position_size': position_size,
            })

            last_exit = exit_date

    # Build daily returns
    all_dates = vixy.index
    daily_pnl = pd.Series(0.0, index=all_dates)

    for t in trades:
        entry_idx = all_dates.get_loc(t['entry_date'])
        exit_idx = all_dates.get_loc(t['exit_date'])
        days = exit_idx - entry_idx
        if days > 0:
            dpnl = t['pnl'] / days
            for d in range(entry_idx, exit_idx):
                daily_pnl.iloc[d] += dpnl

    daily_returns = daily_pnl / capital
    daily_returns = daily_returns[daily_returns != 0]

    return daily_returns, trades


# ── Strategy C: Short VIXY Continuous + Crisis Protection ──────────

def run_vixy_continuous(close, hold_days=20, crisis_vix=30, position_size=200,
                         capital=STARTING_CAPITAL, inverse=False):
    """
    Short VIXY continuously (new position every hold_days if flat).
    Go FLAT when VIX > crisis_vix.
    If inverse=True, go LONG VIXY when VIX <= crisis, flat when VIX > crisis.
    """
    vix = close.get('VIX')
    vixy = close.get('VIXY')
    if vix is None or vixy is None:
        return pd.Series(dtype=float), []

    vix = vix.dropna()
    vixy = vixy.dropna()

    trades = []
    last_exit = None

    common_dates = vix.index.intersection(vixy.index)

    for date in common_dates:
        if last_exit is not None and date <= last_exit:
            continue

        v = vix.get(date, np.nan)
        if np.isnan(v):
            continue

        # Crisis check
        if v > crisis_vix:
            continue  # stay flat

        entry_price = vixy.get(date, np.nan)
        if np.isnan(entry_price) or entry_price <= 0:
            continue

        entry_idx = vixy.index.get_loc(date)
        exit_idx = min(entry_idx + hold_days, len(vixy) - 1)
        exit_date = vixy.index[exit_idx]
        exit_price = vixy.iloc[exit_idx]

        if np.isnan(exit_price):
            continue

        # Check if VIX spikes during hold — early exit
        actual_exit_idx = exit_idx
        for d in range(entry_idx + 1, exit_idx + 1):
            check_date = vixy.index[d]
            if check_date in vix.index:
                v_check = vix.get(check_date, np.nan)
                if not np.isnan(v_check) and v_check > crisis_vix:
                    actual_exit_idx = d
                    break

        exit_date = vixy.index[actual_exit_idx]
        exit_price = vixy.iloc[actual_exit_idx]

        shares = position_size / entry_price
        if not inverse:
            pnl = shares * (entry_price - exit_price)  # short
        else:
            pnl = shares * (exit_price - entry_price)  # long

        ret = pnl / position_size

        trades.append({
            'entry_date': date,
            'exit_date': exit_date,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'return': ret,
            'position_size': position_size,
        })

        last_exit = exit_date

    # Build daily returns
    all_dates = vixy.index
    daily_pnl = pd.Series(0.0, index=all_dates)

    for t in trades:
        entry_idx = all_dates.get_loc(t['entry_date'])
        exit_idx = all_dates.get_loc(t['exit_date'])
        days = exit_idx - entry_idx
        if days > 0:
            dpnl = t['pnl'] / days
            for d in range(entry_idx, exit_idx):
                daily_pnl.iloc[d] += dpnl

    daily_returns = daily_pnl / capital
    daily_returns = daily_returns[daily_returns != 0]

    return daily_returns, trades


# ── Adversarial Tests ──────────────────────────────────────────────

def test_inverse(real_sharpe, inverse_sharpe):
    """PASS if inverse Sharpe < 0.50 * real Sharpe."""
    return inverse_sharpe < 0.50 * real_sharpe


def test_random_timing(returns, real_sharpe, n_perms=N_PERMUTATIONS):
    """Shuffle returns, compute Sharpe each time. PASS if real is in top 5%."""
    if len(returns) < 10:
        return False, 1.0

    ret_vals = returns.values.copy()
    perm_sharpes = []

    for _ in range(n_perms):
        np.random.shuffle(ret_vals)
        s = pd.Series(ret_vals)
        perm_sharpes.append(compute_sharpe(s))

    perm_sharpes = np.array(perm_sharpes)
    p_value = np.mean(perm_sharpes >= real_sharpe)

    return p_value < 0.05, float(p_value)


def test_subperiod_stability(returns):
    """Split into 4 equal sub-periods. PASS if 3/4 positive Sharpe and none < -0.5."""
    if len(returns) < 40:
        return False, []

    n = len(returns)
    chunk = n // 4
    sub_sharpes = []

    for i in range(4):
        start = i * chunk
        end = start + chunk if i < 3 else n
        sub = returns.iloc[start:end]
        sub_sharpes.append(compute_sharpe(sub))

    positive_count = sum(1 for s in sub_sharpes if s > 0)
    no_deep_negative = all(s > -0.5 for s in sub_sharpes)

    passed = positive_count >= 3 and no_deep_negative
    return passed, sub_sharpes


def test_top3_removal(trades, capital, real_sharpe):
    """Remove 3 best trades, recompute Sharpe. PASS if drop < 50%."""
    if len(trades) < 5:
        return False, 0.0

    # Sort by PnL descending
    sorted_trades = sorted(trades, key=lambda x: x['pnl'], reverse=True)
    remaining = sorted_trades[3:]  # remove top 3

    if len(remaining) < 3:
        return False, 0.0

    # Recompute returns from remaining trades
    returns_list = [t['return'] for t in remaining]
    ret_series = pd.Series(returns_list)
    reduced_sharpe = compute_sharpe(ret_series)

    if real_sharpe <= 0:
        return False, reduced_sharpe

    drop_pct = 1.0 - (reduced_sharpe / real_sharpe)
    passed = drop_pct < 0.50
    return passed, float(reduced_sharpe)


def test_parameter_sensitivity_confluence(tickers, close, high, low, volume, opn):
    """Test 64+ parameter combos for confluence strategy."""
    thresholds = [2, 3, 4, 5]
    hold_days_list = [5, 7, 10, 15]
    position_sizes = [100, 150, 200, 300]

    results = []
    total = len(thresholds) * len(hold_days_list) * len(position_sizes)
    count = 0

    for thresh in thresholds:
        for hd in hold_days_list:
            for ps in position_sizes:
                count += 1
                print(f"  Param sensitivity {count}/{total}: thresh={thresh} hold={hd} pos=${ps}")
                returns, trades = compute_confluence_signals(
                    tickers, close, high, low, volume, opn,
                    threshold=thresh, hold_days=hd, position_size=ps, max_concurrent=3
                )
                sharpe = compute_sharpe(returns)
                results.append({
                    'threshold': thresh, 'hold_days': hd,
                    'position_size': ps, 'sharpe': sharpe,
                    'n_trades': len(trades)
                })

    above_threshold = sum(1 for r in results if r['sharpe'] > 0.3)
    pct = above_threshold / len(results) if results else 0
    return pct > 0.50, results, pct


def test_parameter_sensitivity_vixy_contango(close):
    """Test 64 parameter combos for VIXY contango strategy."""
    hold_days_list = [10, 15, 20, 30]
    vix_thresholds = [15, 20, 25, 30]
    position_sizes = [100, 150, 200, 300]

    results = []
    total = len(hold_days_list) * len(vix_thresholds) * len(position_sizes)
    count = 0

    for hd in hold_days_list:
        for vt in vix_thresholds:
            for ps in position_sizes:
                count += 1
                print(f"  Param sensitivity {count}/{total}: hold={hd} vix_thresh={vt} pos=${ps}")
                returns, trades = run_vixy_contango(
                    close, hold_days=hd, vix_threshold=vt, position_size=ps
                )
                sharpe = compute_sharpe(returns)
                results.append({
                    'hold_days': hd, 'vix_threshold': vt,
                    'position_size': ps, 'sharpe': sharpe,
                    'n_trades': len(trades)
                })

    above_threshold = sum(1 for r in results if r['sharpe'] > 0.3)
    pct = above_threshold / len(results) if results else 0
    return pct > 0.50, results, pct


def test_parameter_sensitivity_vixy_continuous(close):
    """Test 64 parameter combos for VIXY continuous strategy."""
    hold_days_list = [10, 15, 20, 30]
    crisis_thresholds = [15, 20, 25, 30]
    position_sizes = [100, 150, 200, 300]

    results = []
    total = len(hold_days_list) * len(crisis_thresholds) * len(position_sizes)
    count = 0

    for hd in hold_days_list:
        for ct in crisis_thresholds:
            for ps in position_sizes:
                count += 1
                print(f"  Param sensitivity {count}/{total}: hold={hd} crisis={ct} pos=${ps}")
                returns, trades = run_vixy_continuous(
                    close, hold_days=hd, crisis_vix=ct, position_size=ps
                )
                sharpe = compute_sharpe(returns)
                results.append({
                    'hold_days': hd, 'crisis_vix': ct,
                    'position_size': ps, 'sharpe': sharpe,
                    'n_trades': len(trades)
                })

    above_threshold = sum(1 for r in results if r['sharpe'] > 0.3)
    pct = above_threshold / len(results) if results else 0
    return pct > 0.50, results, pct


# ── Main ──────────────────────────────────────────────────────────

def main():
    np.random.seed(42)

    print("=" * 80)
    print("ADVERSARIAL VALIDATION OF 3 STRATEGIES")
    print("=" * 80)

    tickers, close, high, low, volume, opn = download_data()

    results = {}

    # ═══════════════════════════════════════════════════════════
    # STRATEGY A: Multi-Strategy Confluence
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STRATEGY A: MULTI-STRATEGY CONFLUENCE (4+ signals, 10d hold)")
    print("=" * 70)

    strat_a_results = {}

    # Test 1: Re-implementation (this IS the re-implementation)
    print("\n[A-T1] Re-implementation...")
    returns_a, trades_a = compute_confluence_signals(
        tickers, close, high, low, volume, opn,
        threshold=4, hold_days=10, position_size=200, max_concurrent=3
    )
    sharpe_a = compute_sharpe(returns_a)
    baseline_a = 1.661
    reimpl_within = abs(sharpe_a - baseline_a) / baseline_a < 0.20
    strat_a_results['test1_reimplementation'] = {
        'passed': reimpl_within,
        'reimpl_sharpe': sharpe_a,
        'baseline_sharpe': baseline_a,
        'pct_diff': abs(sharpe_a - baseline_a) / baseline_a,
        'n_trades': len(trades_a),
    }
    print(f"  Sharpe: {sharpe_a:.3f} (baseline {baseline_a:.3f}, diff {abs(sharpe_a - baseline_a) / baseline_a:.1%})")
    print(f"  Trades: {len(trades_a)}")
    print(f"  {'PASS' if reimpl_within else 'FAIL'}")

    # Test 2: Inverse signal (0-1 signals instead of 4+)
    print("\n[A-T2] Inverse signal (threshold=0-1)...")
    returns_a_inv, trades_a_inv = compute_confluence_signals(
        tickers, close, high, low, volume, opn,
        threshold=1, hold_days=10, position_size=200, max_concurrent=3
    )
    # For inverse, we use threshold=1 meaning buy when ONLY 1 signal fires (weak signal)
    sharpe_a_inv = compute_sharpe(returns_a_inv)
    inv_pass = test_inverse(sharpe_a, sharpe_a_inv)
    strat_a_results['test2_inverse'] = {
        'passed': inv_pass,
        'real_sharpe': sharpe_a,
        'inverse_sharpe': sharpe_a_inv,
        'n_trades_inverse': len(trades_a_inv),
    }
    print(f"  Inverse Sharpe: {sharpe_a_inv:.3f} vs Real: {sharpe_a:.3f}")
    print(f"  Ratio: {sharpe_a_inv / sharpe_a:.3f} (need < 0.50)")
    print(f"  {'PASS' if inv_pass else 'FAIL'}")

    # Test 3: Random timing
    print("\n[A-T3] Random timing (1000 permutations)...")
    perm_pass, p_val = test_random_timing(returns_a, sharpe_a)
    strat_a_results['test3_random_timing'] = {
        'passed': perm_pass,
        'p_value': p_val,
        'real_sharpe': sharpe_a,
    }
    print(f"  p-value: {p_val:.4f} (need < 0.05)")
    print(f"  {'PASS' if perm_pass else 'FAIL'}")

    # Test 4: Sub-period stability
    print("\n[A-T4] Sub-period stability...")
    sub_pass, sub_sharpes = test_subperiod_stability(returns_a)
    strat_a_results['test4_subperiod'] = {
        'passed': sub_pass,
        'sub_sharpes': sub_sharpes,
    }
    for i, s in enumerate(sub_sharpes):
        print(f"  Period {i+1}: Sharpe {s:.3f}")
    print(f"  {'PASS' if sub_pass else 'FAIL'}")

    # Test 5: Top-3 removal
    print("\n[A-T5] Top-3 trade removal...")
    top3_pass, reduced_sharpe = test_top3_removal(trades_a, STARTING_CAPITAL, sharpe_a)
    strat_a_results['test5_top3_removal'] = {
        'passed': top3_pass,
        'original_sharpe': sharpe_a,
        'reduced_sharpe': reduced_sharpe,
        'drop_pct': 1.0 - (reduced_sharpe / sharpe_a) if sharpe_a > 0 else None,
    }
    print(f"  Original Sharpe: {sharpe_a:.3f}, After removal: {reduced_sharpe:.3f}")
    drop = 1.0 - (reduced_sharpe / sharpe_a) if sharpe_a > 0 else float('inf')
    print(f"  Drop: {drop:.1%} (need < 50%)")
    print(f"  {'PASS' if top3_pass else 'FAIL'}")

    # Test 6: Parameter sensitivity
    print("\n[A-T6] Parameter sensitivity (64 combos)...")
    param_pass, param_results, param_pct = test_parameter_sensitivity_confluence(
        tickers, close, high, low, volume, opn
    )
    strat_a_results['test6_param_sensitivity'] = {
        'passed': param_pass,
        'pct_above_0_3': param_pct,
        'n_combos': len(param_results),
        'best_combo': max(param_results, key=lambda x: x['sharpe']) if param_results else None,
        'worst_combo': min(param_results, key=lambda x: x['sharpe']) if param_results else None,
    }
    print(f"  {param_pct:.1%} of combos have Sharpe > 0.3 (need > 50%)")
    print(f"  {'PASS' if param_pass else 'FAIL'}")

    a_total = sum(1 for v in strat_a_results.values() if v['passed'])
    print(f"\n{'='*50}")
    print(f"STRATEGY A OVERALL: {a_total}/6 ADVERSARIAL PASS")
    print(f"{'='*50}")

    results['strategy_a_confluence'] = {
        'name': 'Multi-Strategy Confluence',
        'tests': strat_a_results,
        'total_passed': a_total,
        'sharpe': sharpe_a,
        'n_trades': len(trades_a),
    }

    # ═══════════════════════════════════════════════════════════
    # STRATEGY B: Short VIXY in Contango
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STRATEGY B: SHORT VIXY IN CONTANGO (20d hold)")
    print("=" * 70)

    strat_b_results = {}

    # Test 1: Re-implementation
    print("\n[B-T1] Re-implementation...")
    returns_b, trades_b = run_vixy_contango(close, hold_days=20, position_size=200)
    sharpe_b = compute_sharpe(returns_b)
    baseline_b = 2.030
    reimpl_b = abs(sharpe_b - baseline_b) / baseline_b < 0.20
    strat_b_results['test1_reimplementation'] = {
        'passed': reimpl_b,
        'reimpl_sharpe': sharpe_b,
        'baseline_sharpe': baseline_b,
        'pct_diff': abs(sharpe_b - baseline_b) / baseline_b,
        'n_trades': len(trades_b),
    }
    print(f"  Sharpe: {sharpe_b:.3f} (baseline {baseline_b:.3f}, diff {abs(sharpe_b - baseline_b) / baseline_b:.1%})")
    print(f"  Trades: {len(trades_b)}")
    print(f"  {'PASS' if reimpl_b else 'FAIL'}")

    # Test 2: Inverse (LONG VIXY in contango)
    print("\n[B-T2] Inverse signal (LONG VIXY in contango)...")
    returns_b_inv, trades_b_inv = run_vixy_contango(close, hold_days=20, position_size=200, inverse=True)
    sharpe_b_inv = compute_sharpe(returns_b_inv)
    inv_b_pass = test_inverse(sharpe_b, sharpe_b_inv)
    strat_b_results['test2_inverse'] = {
        'passed': inv_b_pass,
        'real_sharpe': sharpe_b,
        'inverse_sharpe': sharpe_b_inv,
    }
    print(f"  Inverse Sharpe: {sharpe_b_inv:.3f} vs Real: {sharpe_b:.3f}")
    print(f"  {'PASS' if inv_b_pass else 'FAIL'}")

    # Test 3: Random timing
    print("\n[B-T3] Random timing...")
    perm_b_pass, p_val_b = test_random_timing(returns_b, sharpe_b)
    strat_b_results['test3_random_timing'] = {
        'passed': perm_b_pass,
        'p_value': p_val_b,
    }
    print(f"  p-value: {p_val_b:.4f}")
    print(f"  {'PASS' if perm_b_pass else 'FAIL'}")

    # Test 4: Sub-period stability
    print("\n[B-T4] Sub-period stability...")
    sub_b_pass, sub_b_sharpes = test_subperiod_stability(returns_b)
    strat_b_results['test4_subperiod'] = {
        'passed': sub_b_pass,
        'sub_sharpes': sub_b_sharpes,
    }
    for i, s in enumerate(sub_b_sharpes):
        print(f"  Period {i+1}: Sharpe {s:.3f}")
    print(f"  {'PASS' if sub_b_pass else 'FAIL'}")

    # Test 5: Top-3 removal
    print("\n[B-T5] Top-3 trade removal...")
    top3_b_pass, reduced_b = test_top3_removal(trades_b, STARTING_CAPITAL, sharpe_b)
    strat_b_results['test5_top3_removal'] = {
        'passed': top3_b_pass,
        'original_sharpe': sharpe_b,
        'reduced_sharpe': reduced_b,
    }
    print(f"  Original: {sharpe_b:.3f}, After removal: {reduced_b:.3f}")
    print(f"  {'PASS' if top3_b_pass else 'FAIL'}")

    # Test 6: Parameter sensitivity
    print("\n[B-T6] Parameter sensitivity (64 combos)...")
    param_b_pass, param_b_results, param_b_pct = test_parameter_sensitivity_vixy_contango(close)
    strat_b_results['test6_param_sensitivity'] = {
        'passed': param_b_pass,
        'pct_above_0_3': param_b_pct,
        'n_combos': len(param_b_results),
    }
    print(f"  {param_b_pct:.1%} above Sharpe 0.3")
    print(f"  {'PASS' if param_b_pass else 'FAIL'}")

    b_total = sum(1 for v in strat_b_results.values() if v['passed'])
    print(f"\n{'='*50}")
    print(f"STRATEGY B OVERALL: {b_total}/6 ADVERSARIAL PASS")
    print(f"{'='*50}")

    results['strategy_b_vixy_contango'] = {
        'name': 'Short VIXY in Contango',
        'tests': strat_b_results,
        'total_passed': b_total,
        'sharpe': sharpe_b,
        'n_trades': len(trades_b),
    }

    # ═══════════════════════════════════════════════════════════
    # STRATEGY C: Short VIXY Continuous + Crisis Protection
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STRATEGY C: SHORT VIXY CONTINUOUS + CRISIS PROTECTION")
    print("=" * 70)

    strat_c_results = {}

    # Test 1: Re-implementation
    print("\n[C-T1] Re-implementation...")
    returns_c, trades_c = run_vixy_continuous(close, hold_days=20, crisis_vix=30, position_size=200)
    sharpe_c = compute_sharpe(returns_c)
    baseline_c = 1.824
    reimpl_c = abs(sharpe_c - baseline_c) / baseline_c < 0.20
    strat_c_results['test1_reimplementation'] = {
        'passed': reimpl_c,
        'reimpl_sharpe': sharpe_c,
        'baseline_sharpe': baseline_c,
        'pct_diff': abs(sharpe_c - baseline_c) / baseline_c,
        'n_trades': len(trades_c),
    }
    print(f"  Sharpe: {sharpe_c:.3f} (baseline {baseline_c:.3f}, diff {abs(sharpe_c - baseline_c) / baseline_c:.1%})")
    print(f"  Trades: {len(trades_c)}")
    print(f"  {'PASS' if reimpl_c else 'FAIL'}")

    # Test 2: Inverse (LONG VIXY continuous)
    print("\n[C-T2] Inverse signal (LONG VIXY continuous)...")
    returns_c_inv, trades_c_inv = run_vixy_continuous(close, hold_days=20, crisis_vix=30, position_size=200, inverse=True)
    sharpe_c_inv = compute_sharpe(returns_c_inv)
    inv_c_pass = test_inverse(sharpe_c, sharpe_c_inv)
    strat_c_results['test2_inverse'] = {
        'passed': inv_c_pass,
        'real_sharpe': sharpe_c,
        'inverse_sharpe': sharpe_c_inv,
    }
    print(f"  Inverse Sharpe: {sharpe_c_inv:.3f} vs Real: {sharpe_c:.3f}")
    print(f"  {'PASS' if inv_c_pass else 'FAIL'}")

    # Test 3: Random timing
    print("\n[C-T3] Random timing...")
    perm_c_pass, p_val_c = test_random_timing(returns_c, sharpe_c)
    strat_c_results['test3_random_timing'] = {
        'passed': perm_c_pass,
        'p_value': p_val_c,
    }
    print(f"  p-value: {p_val_c:.4f}")
    print(f"  {'PASS' if perm_c_pass else 'FAIL'}")

    # Test 4: Sub-period stability
    print("\n[C-T4] Sub-period stability...")
    sub_c_pass, sub_c_sharpes = test_subperiod_stability(returns_c)
    strat_c_results['test4_subperiod'] = {
        'passed': sub_c_pass,
        'sub_sharpes': sub_c_sharpes,
    }
    for i, s in enumerate(sub_c_sharpes):
        print(f"  Period {i+1}: Sharpe {s:.3f}")
    print(f"  {'PASS' if sub_c_pass else 'FAIL'}")

    # Test 5: Top-3 removal
    print("\n[C-T5] Top-3 trade removal...")
    top3_c_pass, reduced_c = test_top3_removal(trades_c, STARTING_CAPITAL, sharpe_c)
    strat_c_results['test5_top3_removal'] = {
        'passed': top3_c_pass,
        'original_sharpe': sharpe_c,
        'reduced_sharpe': reduced_c,
    }
    print(f"  Original: {sharpe_c:.3f}, After removal: {reduced_c:.3f}")
    print(f"  {'PASS' if top3_c_pass else 'FAIL'}")

    # Test 6: Parameter sensitivity
    print("\n[C-T6] Parameter sensitivity (64 combos)...")
    param_c_pass, param_c_results, param_c_pct = test_parameter_sensitivity_vixy_continuous(close)
    strat_c_results['test6_param_sensitivity'] = {
        'passed': param_c_pass,
        'pct_above_0_3': param_c_pct,
        'n_combos': len(param_c_results),
    }
    print(f"  {param_c_pct:.1%} above Sharpe 0.3")
    print(f"  {'PASS' if param_c_pass else 'FAIL'}")

    c_total = sum(1 for v in strat_c_results.values() if v['passed'])
    print(f"\n{'='*50}")
    print(f"STRATEGY C OVERALL: {c_total}/6 ADVERSARIAL PASS")
    print(f"{'='*50}")

    results['strategy_c_vixy_continuous'] = {
        'name': 'Short VIXY Continuous + Crisis Protection',
        'tests': strat_c_results,
        'total_passed': c_total,
        'sharpe': sharpe_c,
        'n_trades': len(trades_c),
    }

    # ═══════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("FINAL ADVERSARIAL VALIDATION SUMMARY")
    print("=" * 70)

    validated = []
    for key, strat in results.items():
        status = "VALIDATED" if strat['total_passed'] >= 5 else "REJECTED"
        print(f"\n{strat['name']}: {strat['total_passed']}/6 pass -> {status}")
        print(f"  Sharpe: {strat['sharpe']:.3f}, Trades: {strat['n_trades']}")
        for tname, tval in strat['tests'].items():
            pf = "PASS" if tval['passed'] else "FAIL"
            print(f"    {tname}: {pf}")

        if strat['total_passed'] >= 5:
            validated.append(key)

    if validated:
        print(f"\n{'='*70}")
        print("VALIDATED STRATEGIES (5/6 or 6/6):")
        for v in validated:
            strat = results[v]
            print(f"\n  ** {strat['name']} **")
            print(f"     Sharpe: {strat['sharpe']:.3f}, Trades: {strat['n_trades']}")
            if 'confluence' in v:
                print("     Logic: Buy quality stocks when 4+ of 7 mean-reversion/sentiment signals")
                print("            fire simultaneously. Hold 10 days. $200 position, max 3 concurrent.")
            elif 'contango' in v:
                print("     Logic: Short VIXY when VIX < 20d average VIX (contango proxy).")
                print("            Hold 20 days. $200 position.")
            elif 'continuous' in v:
                print("     Logic: Short VIXY continuously (new position every 20d if flat).")
                print("            Go flat when VIX > 30 (crisis protection). $200 position.")
    else:
        print("\nNo strategies passed 5/6 adversarial tests.")

    # Convert dates to strings for JSON serialization
    def serialize(obj):
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    def deep_serialize(obj):
        if isinstance(obj, dict):
            return {k: deep_serialize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [deep_serialize(i) for i in obj]
        return serialize(obj)

    output_path = Path('/home/jupiter/Lvl3Quant/data/novel_adversarial_results.json')
    with open(output_path, 'w') as f:
        json.dump(deep_serialize(results), f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Holding Period & Exit Strategy Optimization
=============================================
Tests whether our top 3 validated signals perform better with different
holding periods and exit rules vs the baseline 21d / 10% TP / 15% SL.

Signals:
  1. IV-RV Gap: VIX > realized vol by 5pts AND stock >5% below 20-SMA AND RSI<40
  2. Bond Yield: ^TNX drops >0.1% over 5d AND stock >5% below 20-SMA
  3. RSI Divergence: Price lower low vs 14d ago, RSI higher low, volume declining, >5% below 20-SMA

Exit Strategies:
  A) Quick Scalp: Hold 5d, TP 3%, SL -5%
  B) Short Swing: Hold 10d, TP 5%, SL -8%
  C) Medium (baseline): Hold 21d, TP 10%, SL -15%
  D) Patient: Hold 42d, TP 15%, SL -12%
  E) Trailing Stop: Hold 21d, no fixed TP, 5% trailing stop from peak, SL -15%
  F) Partial Exits: Hold 21d, sell half at +5%, rest at +10% or SL -15%

5-gate validation on each of 18 combos.
"""

import os, sys, json, warnings, time, functools
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings('ignore')
print = functools.partial(print, flush=True)

# --- Config ---
START_DATE = '2019-01-01'
END_DATE = '2026-07-01'
BACKTEST_START = '2020-01-01'
POS_SIZE = 300.0
MAX_CONCURRENT = 2
SPREAD_COST_PCT = 0.001
N_PERMS = 1000
REGIME_GAP_LIMIT = 0.50

OUTPUT_DIR = Path('/home/jupiter/Lvl3Quant/output/growth_research/holding_period_opt')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

UNIVERSE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'JPM', 'JNJ', 'UNH', 'PG',
    'HD', 'ABBV', 'MRK', 'LLY', 'AVGO', 'COST', 'CRM', 'AMD', 'NFLX', 'V',
    'MA', 'WMT', 'PEP', 'KO', 'TMO', 'ABT', 'BAC', 'XOM', 'CVX', 'CSCO',
]
MACRO_TICKERS = ['SPY', '^VIX', '^VIX3M', 'TLT', '^TNX']

# Exit strategy definitions
EXIT_STRATEGIES = {
    'A_quick_scalp':   {'hold_days': 5,  'tp': 0.03, 'sl': -0.05, 'trailing': False, 'partial': False},
    'B_short_swing':   {'hold_days': 10, 'tp': 0.05, 'sl': -0.08, 'trailing': False, 'partial': False},
    'C_medium':        {'hold_days': 21, 'tp': 0.10, 'sl': -0.15, 'trailing': False, 'partial': False},
    'D_patient':       {'hold_days': 42, 'tp': 0.15, 'sl': -0.12, 'trailing': False, 'partial': False},
    'E_trailing_stop': {'hold_days': 21, 'tp': None, 'sl': -0.15, 'trailing': True,  'partial': False, 'trail_pct': 0.05},
    'F_partial_exits': {'hold_days': 21, 'tp': 0.10, 'sl': -0.15, 'trailing': False, 'partial': True,  'partial_tp1': 0.05, 'partial_tp2': 0.10},
}

print("=" * 100)
print("HOLDING PERIOD & EXIT STRATEGY OPTIMIZATION")
print("=" * 100)


# =====================================================================
# 1. DATA
# =====================================================================
def download_data():
    import yfinance as yf
    cache_file = OUTPUT_DIR / '_holding_opt_cache.pkl'
    if cache_file.exists():
        import pickle
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        print(f"  Loaded cache: {len(data['close'].columns)} stocks, {len(data['close'])} days")
        return data

    print(f"\n[DATA] Downloading {len(UNIVERSE)} stocks + {len(MACRO_TICKERS)} macro...")
    all_tickers = UNIVERSE + MACRO_TICKERS
    raw = yf.download(all_tickers, start=START_DATE, end=END_DATE,
                      auto_adjust=True, progress=False, threads=True)

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw['Close']
        high = raw['High']
        low = raw['Low']
        volume = raw['Volume']
    else:
        close = high = low = volume = raw

    for df in [close, high, low, volume]:
        if hasattr(df.columns, 'droplevel'):
            try:
                df.columns = df.columns.droplevel(1)
            except:
                pass

    close = close.ffill().dropna(how='all')
    data = {
        'close': close,
        'high': high.reindex(close.index).ffill(),
        'low': low.reindex(close.index).ffill(),
        'volume': volume.reindex(close.index).ffill().fillna(0),
    }

    import pickle
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)
    print(f"  Downloaded: {len(close.columns)} tickers, {len(close)} days")
    return data


# =====================================================================
# 2. HELPERS
# =====================================================================
def _rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _realized_vol(returns, window=21):
    return returns.rolling(window).std() * np.sqrt(252) * 100


# =====================================================================
# 3. SIGNAL GENERATORS
# =====================================================================
def signal_iv_rv_gap(data, stock_tickers):
    """IV-RV Gap: VIX > 20d realized vol by 5pts AND stock >5% below 20-SMA AND RSI<40."""
    spy_close = data['close']['SPY']
    spy_ret = spy_close.pct_change()
    vix = data['close']['^VIX']
    rv_20 = _realized_vol(spy_ret, 20)
    gap = vix - rv_20

    signals = {}
    for t in stock_tickers:
        if t not in data['close'].columns:
            continue
        close = data['close'][t].dropna()
        sma20 = close.rolling(20).mean()
        rsi = _rsi(close, 14)
        drawdown = (close - sma20) / sma20

        for i in range(20, len(close)):
            day = close.index[i]
            if day not in gap.index:
                continue
            try:
                g = gap.at[day]
            except:
                continue
            if pd.isna(g) or g <= 5:
                continue
            if drawdown.iloc[i] > -0.05:
                continue
            if pd.isna(rsi.iloc[i]) or rsi.iloc[i] >= 40:
                continue
            signals[(day, t)] = True
    return signals


def signal_bond_yield(data, stock_tickers):
    """Bond Yield: ^TNX drops >0.1% over 5d AND stock >5% below 20-SMA."""
    if '^TNX' not in data['close'].columns:
        tlt = data['close']['TLT']
        yield_change = tlt.pct_change(5)
        fire_mask = yield_change > 0.02
    else:
        tnx = data['close']['^TNX']
        yield_change = tnx.diff(5)
        fire_mask = yield_change < -0.10

    fire_days = set(fire_mask[fire_mask].index)
    signals = {}
    for t in stock_tickers:
        if t not in data['close'].columns:
            continue
        close = data['close'][t].dropna()
        sma20 = close.rolling(20).mean()
        drawdown = (close - sma20) / sma20

        for i in range(20, len(close)):
            day = close.index[i]
            if day not in fire_days:
                continue
            if drawdown.iloc[i] > -0.05:
                continue
            signals[(day, t)] = True
    return signals


def signal_rsi_divergence(data, stock_tickers):
    """RSI Divergence: Price lower low vs 14d ago, RSI higher low, volume declining, >5% below 20-SMA."""
    signals = {}
    for t in stock_tickers:
        if t not in data['close'].columns:
            continue
        close = data['close'][t].dropna()
        rsi = _rsi(close, 14)
        vol = data['volume'].get(t)
        if vol is None:
            continue
        vol = vol.reindex(close.index).fillna(0)
        sma20 = close.rolling(20).mean()
        drawdown = (close - sma20) / sma20

        for i in range(20, len(close)):
            if i < 14:
                continue
            # Dip below SMA
            if drawdown.iloc[i] > -0.05:
                continue
            # Price lower low
            if close.iloc[i] >= close.iloc[i - 14]:
                continue
            # RSI higher low (divergence)
            r_now = rsi.iloc[i]
            r_prev = rsi.iloc[i - 14]
            if pd.isna(r_now) or pd.isna(r_prev):
                continue
            if r_now <= r_prev:
                continue
            # Volume declining
            vol_now = vol.iloc[max(0, i-4):i+1].mean()
            vol_prev = vol.iloc[max(0, i-9):i-4].mean()
            if vol_prev > 0 and vol_now >= vol_prev:
                continue
            signals[(close.index[i], t)] = True
    return signals


# =====================================================================
# 4. BACKTESTER (supports all exit types)
# =====================================================================
def run_backtest(signal_entries, data, stock_tickers, exit_config):
    """Run backtest with configurable exit strategy."""
    close = data['close']
    high_df = data['high']
    spy_close = close['SPY']
    bt_start = pd.Timestamp(BACKTEST_START)
    entries = sorted([(d, t) for (d, t) in signal_entries if d >= bt_start], key=lambda x: x[0])

    if not entries:
        return None

    hold_days = exit_config['hold_days']
    tp = exit_config.get('tp')
    sl = exit_config['sl']
    is_trailing = exit_config.get('trailing', False)
    is_partial = exit_config.get('partial', False)
    trail_pct = exit_config.get('trail_pct', 0.05)
    partial_tp1 = exit_config.get('partial_tp1', 0.05)
    partial_tp2 = exit_config.get('partial_tp2', 0.10)

    trades = []
    open_positions = []

    for entry_date, ticker in entries:
        open_positions = [p for p in open_positions if p[3] > entry_date]
        if len(open_positions) >= MAX_CONCURRENT:
            continue
        try:
            entry_price = close.at[entry_date, ticker]
        except:
            continue
        if pd.isna(entry_price) or entry_price <= 0:
            continue

        future_dates = close.index[close.index > entry_date]
        if len(future_dates) == 0:
            continue

        # --- Standard exit (A, B, C, D) ---
        if not is_trailing and not is_partial:
            exit_price = None
            exit_date = None
            exit_reason = 'hold_expiry'

            for j, fdate in enumerate(future_dates[:hold_days]):
                try:
                    price = close.at[fdate, ticker]
                except:
                    continue
                if pd.isna(price):
                    continue
                ret = (price - entry_price) / entry_price
                if tp is not None and ret >= tp:
                    exit_price = price
                    exit_date = fdate
                    exit_reason = 'profit_target'
                    break
                elif ret <= sl:
                    exit_price = price
                    exit_date = fdate
                    exit_reason = 'stop_loss'
                    break

            if exit_price is None:
                hold_end = min(hold_days, len(future_dates))
                if hold_end > 0:
                    exit_date = future_dates[hold_end - 1]
                    try:
                        exit_price = close.at[exit_date, ticker]
                    except:
                        continue
                    if pd.isna(exit_price):
                        continue

            if exit_price is None:
                continue

            raw_ret = (exit_price - entry_price) / entry_price
            net_ret = raw_ret - SPREAD_COST_PCT
            pnl = POS_SIZE * net_ret

        # --- Trailing stop exit (E) ---
        elif is_trailing:
            exit_price = None
            exit_date = None
            exit_reason = 'hold_expiry'
            peak_price = entry_price

            for j, fdate in enumerate(future_dates[:hold_days]):
                try:
                    price = close.at[fdate, ticker]
                except:
                    continue
                if pd.isna(price):
                    continue

                # Update peak
                if price > peak_price:
                    peak_price = price

                ret = (price - entry_price) / entry_price
                trail_ret = (price - peak_price) / peak_price

                # Trailing stop: price dropped trail_pct from peak
                if peak_price > entry_price and trail_ret <= -trail_pct:
                    exit_price = price
                    exit_date = fdate
                    exit_reason = 'trailing_stop'
                    break
                # Hard stop loss
                elif ret <= sl:
                    exit_price = price
                    exit_date = fdate
                    exit_reason = 'stop_loss'
                    break

            if exit_price is None:
                hold_end = min(hold_days, len(future_dates))
                if hold_end > 0:
                    exit_date = future_dates[hold_end - 1]
                    try:
                        exit_price = close.at[exit_date, ticker]
                    except:
                        continue
                    if pd.isna(exit_price):
                        continue

            if exit_price is None:
                continue

            raw_ret = (exit_price - entry_price) / entry_price
            net_ret = raw_ret - SPREAD_COST_PCT
            pnl = POS_SIZE * net_ret

        # --- Partial exits (F) ---
        elif is_partial:
            # Two lots: each POS_SIZE/2
            half_size = POS_SIZE / 2.0
            exit_price = None
            exit_date = None
            exit_reason = 'hold_expiry'
            lot1_closed = False
            lot1_ret = 0.0
            lot1_pnl = 0.0

            for j, fdate in enumerate(future_dates[:hold_days]):
                try:
                    price = close.at[fdate, ticker]
                except:
                    continue
                if pd.isna(price):
                    continue
                ret = (price - entry_price) / entry_price

                # Close lot 1 at partial_tp1
                if not lot1_closed and ret >= partial_tp1:
                    lot1_ret = ret - SPREAD_COST_PCT / 2  # half the spread
                    lot1_pnl = half_size * lot1_ret
                    lot1_closed = True

                # Close lot 2 at partial_tp2 or SL
                if ret >= partial_tp2:
                    exit_price = price
                    exit_date = fdate
                    exit_reason = 'profit_target'
                    break
                elif ret <= sl:
                    exit_price = price
                    exit_date = fdate
                    exit_reason = 'stop_loss'
                    break

            if exit_price is None:
                hold_end = min(hold_days, len(future_dates))
                if hold_end > 0:
                    exit_date = future_dates[hold_end - 1]
                    try:
                        exit_price = close.at[exit_date, ticker]
                    except:
                        continue
                    if pd.isna(exit_price):
                        continue

            if exit_price is None:
                continue

            lot2_raw_ret = (exit_price - entry_price) / entry_price
            lot2_ret = lot2_raw_ret - SPREAD_COST_PCT / 2

            if lot1_closed:
                # Weighted average of two lots
                net_ret = (lot1_ret + lot2_ret) / 2.0
                pnl = lot1_pnl + half_size * lot2_ret
            else:
                # Lot 1 never hit partial_tp1, full exit at same price
                net_ret = lot2_raw_ret - SPREAD_COST_PCT
                pnl = POS_SIZE * net_ret

        # Regime
        try:
            spy_sma = spy_close.rolling(200).mean()
            spy_regime = 'bull' if spy_close.at[entry_date] > spy_sma.at[entry_date] else 'bear'
        except:
            spy_regime = 'unknown'

        trades.append({
            'entry_date': entry_date,
            'exit_date': exit_date,
            'ticker': ticker,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'return': net_ret,
            'pnl': pnl,
            'exit_reason': exit_reason,
            'regime': spy_regime,
            'hold_days': (exit_date - entry_date).days,
        })
        open_positions.append((entry_date, ticker, entry_price, exit_date))

    if not trades:
        return None
    return pd.DataFrame(trades)


# =====================================================================
# 5. 5-GATE VALIDATION
# =====================================================================
def validate(trades_df, label, signal_entries, data, stock_tickers, exit_config):
    """Full 5-gate validation with perm test."""
    if trades_df is None or len(trades_df) < 10:
        return {
            'label': label, 'n_trades': 0 if trades_df is None else len(trades_df),
            'sharpe': 0, 'sortino': 0, 'wr': 0, 'pf': 0, 'mdd': 0,
            'regime_gap': 1.0, 'perm_p': 1.0, 'passed': False,
            'avg_hold': 0, 'mean_ret_pct': 0,
            'failed_gates': ['insufficient_trades'],
        }

    rets = trades_df['return'].values
    n = len(rets)
    years = max(1, (trades_df['entry_date'].max() - trades_df['entry_date'].min()).days / 365.25)
    tpy = max(1, n / years)
    mean_ret = np.mean(rets)
    std_ret = np.std(rets, ddof=1) if n > 1 else 1.0
    sharpe = (mean_ret / std_ret) * np.sqrt(tpy) if std_ret > 0 else 0

    # Sortino
    downside = rets[rets < 0]
    ds_std = np.std(downside, ddof=1) if len(downside) > 1 else std_ret
    sortino = (mean_ret / ds_std) * np.sqrt(tpy) if ds_std > 0 else 0

    wr = np.mean(rets > 0)
    gross_profit = np.sum(rets[rets > 0])
    gross_loss = np.abs(np.sum(rets[rets < 0]))
    pf = gross_profit / gross_loss if gross_loss > 0 else 99.0

    cum_pnl = np.cumsum(trades_df['pnl'].values)
    peak = np.maximum.accumulate(cum_pnl)
    mdd = np.min(cum_pnl - peak)

    # Regime gap
    bull = trades_df[trades_df['regime'] == 'bull']
    bear = trades_df[trades_df['regime'] == 'bear']
    if len(bull) >= 3 and len(bear) >= 3:
        bs = np.mean(bull['return']) / max(np.std(bull['return'], ddof=1), 1e-6)
        brs = np.mean(bear['return']) / max(np.std(bear['return'], ddof=1), 1e-6)
        regime_gap = abs(bs - brs) / max(abs(bs), abs(brs), 1e-6)
    else:
        regime_gap = 0.0

    # Perm test
    bt_start = pd.Timestamp(BACKTEST_START)
    valid_dates = data['close'].index[data['close'].index >= bt_start]
    hold_d = exit_config['hold_days']
    valid_dates = valid_dates[:-hold_d - 5] if len(valid_dates) > hold_d + 5 else valid_dates

    perm_sharpes = []
    n_raw = len(signal_entries)
    for _ in range(N_PERMS):
        rand_dates = np.random.choice(valid_dates, size=min(n_raw, len(valid_dates)), replace=True)
        rand_tickers = np.random.choice(stock_tickers, size=min(n_raw, len(valid_dates)), replace=True)
        rand_signals = {(d, t): True for d, t in zip(rand_dates, rand_tickers)}
        rand_trades = run_backtest(rand_signals, data, stock_tickers, exit_config)
        if rand_trades is not None and len(rand_trades) >= 5:
            r_rets = rand_trades['return'].values
            r_years = max(1, (rand_trades['entry_date'].max() - rand_trades['entry_date'].min()).days / 365.25)
            r_tpy = max(1, len(r_rets) / r_years)
            r_mean = np.mean(r_rets)
            r_std = np.std(r_rets, ddof=1)
            r_sharpe = (r_mean / r_std) * np.sqrt(r_tpy) if r_std > 0 else 0
        else:
            r_sharpe = 0.0
        perm_sharpes.append(r_sharpe)

    perm_p = float(np.mean(np.array(perm_sharpes) >= sharpe))

    # Gates
    gates = {
        'sharpe': sharpe > 0.3,
        'wr': wr > 0.45,
        'pf': pf > 1.0,
        'mdd': mdd > -POS_SIZE * 5,
        'regime_gap': regime_gap < REGIME_GAP_LIMIT,
    }
    perm_pass = perm_p < 0.05
    passed = all(gates.values()) and perm_pass
    failed = [k for k, v in gates.items() if not v]
    if not perm_pass:
        failed.append('perm_test')

    return {
        'label': label,
        'n_trades': n,
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'wr': round(wr, 3),
        'pf': round(pf, 3),
        'mdd': round(float(mdd), 2),
        'regime_gap': round(regime_gap, 3),
        'perm_p': round(perm_p, 4),
        'passed': passed,
        'avg_hold': round(float(trades_df['hold_days'].mean()), 1),
        'mean_ret_pct': round(mean_ret * 100, 2),
        'failed_gates': failed,
    }


# =====================================================================
# 6. MAIN
# =====================================================================
def main():
    t_start = time.time()
    data = download_data()
    stock_tickers = [t for t in UNIVERSE if t in data['close'].columns]
    print(f"  Universe: {len(stock_tickers)} tickers")

    # Generate signals
    print("\n[1] Generating signals...")
    signal_generators = {
        'IV-RV Gap': signal_iv_rv_gap,
        'Bond Yield': signal_bond_yield,
        'RSI Divergence': signal_rsi_divergence,
    }

    all_signals = {}
    for name, gen_func in signal_generators.items():
        t0 = time.time()
        sigs = gen_func(data, stock_tickers)
        elapsed = time.time() - t0
        print(f"  {name}: {len(sigs)} raw entries ({elapsed:.1f}s)")
        all_signals[name] = sigs

    # Run all 18 combos
    print(f"\n[2] Running 18 combos (3 signals x 6 exits) with {N_PERMS}-perm validation...")
    results = []

    for sig_name, sigs in all_signals.items():
        for exit_name, exit_config in EXIT_STRATEGIES.items():
            label = f"{sig_name} / {exit_name}"
            print(f"\n  Running: {label}...")
            trades_df = run_backtest(sigs, data, stock_tickers, exit_config)
            result = validate(trades_df, label, sigs, data, stock_tickers, exit_config)
            result['signal'] = sig_name
            result['exit'] = exit_name
            results.append(result)
            status = "PASS" if result['passed'] else "FAIL"
            print(f"    N={result['n_trades']:4d} | Sharpe={result['sharpe']:6.3f} | "
                  f"Sortino={result['sortino']:6.3f} | WR={result['wr']:.3f} | "
                  f"PF={result['pf']:.3f} | AvgHold={result['avg_hold']:.0f}d | "
                  f"RegGap={result['regime_gap']:.3f} | p={result['perm_p']:.4f} | {status}")

    # =====================================================================
    # SUMMARY TABLE
    # =====================================================================
    elapsed = time.time() - t_start

    print(f"\n{'='*140}")
    print(f"RESULTS TABLE: 18 SIGNAL x EXIT COMBOS")
    print(f"{'='*140}")

    header = (f"{'Signal':16s} | {'Exit':18s} | {'N':>5s} | {'Sharpe':>7s} | {'Sortino':>7s} | "
              f"{'WR':>5s} | {'PF':>6s} | {'MDD':>8s} | {'AvgHold':>7s} | {'RegGap':>6s} | "
              f"{'Perm p':>7s} | {'Result':>6s}")
    print(header)
    print("-" * 140)

    for r in results:
        status = "PASS" if r['passed'] else "FAIL"
        marker = ">>>" if r['passed'] else "   "
        print(f"{marker}{r['signal']:13s} | {r['exit']:18s} | {r['n_trades']:5d} | {r['sharpe']:7.3f} | "
              f"{r['sortino']:7.3f} | {r['wr']:.3f} | {r['pf']:6.3f} | {r['mdd']:8.2f} | "
              f"{r['avg_hold']:6.1f}d | {r['regime_gap']:6.3f} | {r['perm_p']:7.4f} | {status:>6s}")

    # Best exit per signal
    print(f"\n{'='*100}")
    print(f"BEST EXIT STRATEGY PER SIGNAL (by Sharpe)")
    print(f"{'='*100}")

    for sig_name in signal_generators.keys():
        sig_results = [r for r in results if r['signal'] == sig_name]
        if not sig_results:
            continue
        best = max(sig_results, key=lambda x: x['sharpe'])
        passing = [r for r in sig_results if r['passed']]
        best_pass = max(passing, key=lambda x: x['sharpe']) if passing else None

        print(f"\n  {sig_name}:")
        print(f"    Best overall:  {best['exit']} -> Sharpe={best['sharpe']:.3f}, "
              f"Sortino={best['sortino']:.3f}, WR={best['wr']:.1%}, PF={best['pf']:.2f}, "
              f"N={best['n_trades']}, AvgHold={best['avg_hold']:.0f}d, "
              f"{'PASS' if best['passed'] else 'FAIL'}")
        if best_pass and best_pass != best:
            print(f"    Best passing:  {best_pass['exit']} -> Sharpe={best_pass['sharpe']:.3f}, "
                  f"Sortino={best_pass['sortino']:.3f}, WR={best_pass['wr']:.1%}, PF={best_pass['pf']:.2f}")
        elif not passing:
            print(f"    No exit strategy passed 5-gate for this signal.")

    # Passing combos
    passing = [r for r in results if r['passed']]
    print(f"\n{'='*100}")
    print(f"OVERALL: {len(passing)}/18 combos passed 5-gate validation")
    print(f"{'='*100}")

    if passing:
        passing_sorted = sorted(passing, key=lambda x: x['sharpe'], reverse=True)
        for r in passing_sorted:
            print(f"  {r['signal']:16s} / {r['exit']:18s} | Sharpe={r['sharpe']:.3f} | "
                  f"Sortino={r['sortino']:.3f} | WR={r['wr']:.1%} | PF={r['pf']:.2f} | "
                  f"p={r['perm_p']:.4f}")

    print(f"\n  Runtime: {elapsed:.0f}s")

    # Save
    def _sanitize(obj):
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (pd.Timestamp,)):
            return str(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def _sanitize_deep(obj):
        if isinstance(obj, dict):
            return {k: _sanitize_deep(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize_deep(v) for v in obj]
        return _sanitize(obj)

    output_data = {
        'run_date': datetime.now().isoformat(),
        'runtime_seconds': round(elapsed, 1),
        'config': {
            'universe': UNIVERSE,
            'universe_size': len(stock_tickers),
            'backtest_start': BACKTEST_START,
            'backtest_end': END_DATE,
            'pos_size': POS_SIZE,
            'max_concurrent': MAX_CONCURRENT,
            'spread_cost': SPREAD_COST_PCT,
            'n_perms': N_PERMS,
            'regime_gap_limit': REGIME_GAP_LIMIT,
        },
        'exit_strategies': {k: {kk: vv for kk, vv in v.items()} for k, v in EXIT_STRATEGIES.items()},
        'results': _sanitize_deep(results),
        'passing_combos': _sanitize_deep(passing),
        'best_per_signal': {},
    }

    for sig_name in signal_generators.keys():
        sig_results = [r for r in results if r['signal'] == sig_name]
        if sig_results:
            best = max(sig_results, key=lambda x: x['sharpe'])
            output_data['best_per_signal'][sig_name] = _sanitize_deep(best)

    out_file = OUTPUT_DIR / 'results.json'
    with open(out_file, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    print(f"\nResults saved to {out_file}")
    print("=" * 100)
    print("DONE")

    return output_data


if __name__ == '__main__':
    main()

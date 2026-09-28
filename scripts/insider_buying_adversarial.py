#!/usr/bin/env python3
"""
Insider Buying Cluster — Adversarial Validation
=================================================
5 adversarial tests on variants B (deep_value) and F (multi_signal)
to determine if these strategies have real edge or are just "buying big tech."

Tests:
  1. INVERSE DIRECTION — buy after 15%+ rallies instead of drops
  2. RANDOM TIMING — randomize entry dates (1000 perms)
  3. SUB-PERIOD STABILITY — split OOT into 3 equal sub-periods
  4. TOP-TRADE REMOVAL — remove top 5 trades by P&L
  5. TICKER CONCENTRATION — remove top 3 tickers by P&L contribution
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from collections import defaultdict

warnings.filterwarnings('ignore')

# ── CONFIG (identical to backtest) ────────────────────────────────────────
TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AMD', 'AVGO', 'CRM',
    'NFLX', 'ADBE', 'INTC', 'CSCO', 'QCOM', 'TXN', 'MU', 'AMAT', 'LRCX', 'ISRG'
]
START_DATE = '2021-06-01'
END_DATE = '2026-07-29'
OOT_START = '2022-01-01'
ACCOUNT = 645.0
MAX_PER_TRADE = 200.0
MAX_CONCURRENT = 3
SLIPPAGE_RT = 0.0004
RF_RATE = 0.045
HOLD_DAYS = 20

# ── DATA DOWNLOAD ────────────────────────────────────────────────────────
print("Downloading price data...")
data = {}
for ticker in TICKERS:
    try:
        df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        if len(df) > 100:
            data[ticker] = df
            print(f"  {ticker}: {len(df)} days")
    except Exception as e:
        print(f"  {ticker}: ERROR - {e}")

spy = yf.download('SPY', start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)
if isinstance(spy.columns, pd.MultiIndex):
    spy.columns = spy.columns.droplevel(1)
spy['SMA200'] = spy['Close'].rolling(200).mean()
spy['regime'] = np.where(spy['Close'] > spy['SMA200'], 'bull', 'bear')
print(f"  SPY: {len(spy)} days")
print(f"Loaded {len(data)} tickers.\n")


# ── INDICATORS ────────────────────────────────────────────────────────────
def compute_indicators(df):
    df = df.copy()
    df['high_20d'] = df['Close'].rolling(20).max()
    df['low_20d'] = df['Close'].rolling(20).min()
    df['drop_from_high'] = (df['Close'] - df['high_20d']) / df['high_20d']
    df['rally_from_low'] = (df['Close'] - df['low_20d']) / df['low_20d']
    df['vol_20d'] = df['Volume'].rolling(20).mean()
    df['vol_ratio'] = df['Volume'] / df['vol_20d']
    df['SMA200'] = df['Close'].rolling(200).mean()
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

for ticker in data:
    data[ticker] = compute_indicators(data[ticker])


# ── SIGNAL GENERATORS ────────────────────────────────────────────────────
def signals_B(df):
    """Variant B: 15%+ drop from 20d high, 2x volume."""
    sig = (df['drop_from_high'] <= -0.15) & (df['vol_ratio'] >= 2.0)
    return sig.shift(1).fillna(False)

def signals_F(df):
    """Variant F: 10%+ drop + RSI<30 + 2x volume."""
    sig = (df['drop_from_high'] <= -0.10) & (df['vol_ratio'] >= 2.0) & (df['RSI'] < 30)
    return sig.shift(1).fillna(False)

def signals_B_inverse(df):
    """INVERSE of B: 15%+ RALLY from 20d low, 2x volume."""
    sig = (df['rally_from_low'] >= 0.15) & (df['vol_ratio'] >= 2.0)
    return sig.shift(1).fillna(False)

def signals_F_inverse(df):
    """INVERSE of F: 10%+ rally + RSI>70 + 2x volume."""
    sig = (df['rally_from_low'] >= 0.10) & (df['vol_ratio'] >= 2.0) & (df['RSI'] > 70)
    return sig.shift(1).fillna(False)


# ── BACKTEST ENGINE (exact copy from original) ───────────────────────────
def run_backtest(signals_by_ticker, hold_days=HOLD_DAYS, account=ACCOUNT,
                 exclude_tickers=None, oot_start=OOT_START, oot_end=None):
    """Execute trades. Returns (trades, equity_curve)."""
    all_signals = []
    for ticker, sig in signals_by_ticker.items():
        if exclude_tickers and ticker in exclude_tickers:
            continue
        df = data[ticker]
        oot_mask = df.index >= pd.Timestamp(oot_start)
        if oot_end:
            oot_mask = oot_mask & (df.index < pd.Timestamp(oot_end))
        sig_dates = sig[sig & oot_mask].index
        for d in sig_dates:
            all_signals.append((d, ticker))

    all_signals.sort(key=lambda x: x[0])

    trades = []
    active_positions = []
    equity = account
    equity_curve = [(pd.Timestamp(oot_start), account)]

    for entry_date, ticker in all_signals:
        active_positions = [(ed, tk) for ed, tk in active_positions if ed > entry_date]
        if len(active_positions) >= MAX_CONCURRENT:
            continue

        df = data[ticker]
        if entry_date not in df.index:
            continue

        entry_idx = df.index.get_loc(entry_date)
        exit_idx = min(entry_idx + hold_days, len(df) - 1)
        if exit_idx <= entry_idx:
            continue

        entry_price = df['Close'].iloc[entry_idx]
        exit_price = df['Close'].iloc[exit_idx]
        exit_date = df.index[exit_idx]

        position_size = min(MAX_PER_TRADE, equity / MAX_CONCURRENT)
        if position_size <= 0:
            continue

        gross_return = (exit_price - entry_price) / entry_price
        net_return = gross_return - SLIPPAGE_RT
        pnl = position_size * net_return
        equity += pnl

        if entry_date in spy.index:
            regime = spy.loc[entry_date, 'regime']
        else:
            prior = spy.index[spy.index <= entry_date]
            regime = spy.loc[prior[-1], 'regime'] if len(prior) > 0 else 'unknown'

        trades.append({
            'ticker': ticker,
            'entry_date': str(entry_date.date()),
            'exit_date': str(exit_date.date()),
            'entry_price': float(entry_price),
            'exit_price': float(exit_price),
            'gross_return': float(gross_return),
            'net_return': float(net_return),
            'pnl': float(pnl),
            'position_size': float(position_size),
            'regime': regime,
            'equity_after': float(equity)
        })

        active_positions.append((exit_date, ticker))
        equity_curve.append((exit_date, equity))

    return trades, equity_curve


def compute_sharpe(trades, oot_start=OOT_START):
    """Compute Sharpe ratio from a list of trades."""
    if len(trades) < 2:
        return 0.0
    returns = [t['net_return'] for t in trades]
    first_date = pd.Timestamp(oot_start)
    last_trade_date = max(pd.Timestamp(t['exit_date']) for t in trades)
    years = max((last_trade_date - first_date).days / 365.25, 0.5)
    trades_per_year = len(trades) / years
    avg_hold = np.mean([(pd.Timestamp(t['exit_date']) - pd.Timestamp(t['entry_date'])).days for t in trades])
    rf_per_trade = RF_RATE * (avg_hold / 365.25)
    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1)
    excess_ret = mean_ret - rf_per_trade
    sharpe = (excess_ret / std_ret) * np.sqrt(trades_per_year) if std_ret > 0 else 0
    return round(sharpe, 3)


def compute_sharpe_from_returns(returns, trades_per_year, avg_hold_days):
    """Compute Sharpe from a list of returns (for sub-period analysis)."""
    if len(returns) < 2:
        return 0.0
    rf_per_trade = RF_RATE * (avg_hold_days / 365.25)
    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1)
    if std_ret == 0:
        return 0.0
    excess_ret = mean_ret - rf_per_trade
    return round((excess_ret / std_ret) * np.sqrt(trades_per_year), 3)


# ── TEST 1: INVERSE DIRECTION ────────────────────────────────────────────
def test_inverse(variant_name, signal_fn, inverse_fn):
    """Buy after rallies instead of drops. PASS if inverse Sharpe < 0."""
    print(f"\n  TEST 1 — INVERSE DIRECTION ({variant_name})")

    # Normal signals
    sigs = {t: signal_fn(data[t]) for t in data}
    trades_normal, ec_normal = run_backtest(sigs)
    sharpe_normal = compute_sharpe(trades_normal)

    # Inverse signals
    inv_sigs = {t: inverse_fn(data[t]) for t in data}
    trades_inv, ec_inv = run_backtest(inv_sigs)
    sharpe_inv = compute_sharpe(trades_inv)

    passed = sharpe_inv < 0
    print(f"    Normal Sharpe: {sharpe_normal:.3f} ({len(trades_normal)} trades)")
    print(f"    Inverse Sharpe: {sharpe_inv:.3f} ({len(trades_inv)} trades)")
    print(f"    {'PASS' if passed else 'FAIL'} — inverse {'negative' if passed else 'POSITIVE (just buying these stocks = money)'}")

    return {
        'passed': passed,
        'details': {
            'normal_sharpe': sharpe_normal,
            'normal_trades': len(trades_normal),
            'inverse_sharpe': sharpe_inv,
            'inverse_trades': len(trades_inv),
        }
    }


# ── TEST 2: RANDOM TIMING ───────────────────────────────────────────────
def test_random_timing(variant_name, signal_fn, n_perms=1000):
    """Randomize entry dates, keep same stocks and trade count. PASS if actual > 95th percentile."""
    print(f"\n  TEST 2 — RANDOM TIMING ({variant_name})")

    sigs = {t: signal_fn(data[t]) for t in data}
    trades, ec = run_backtest(sigs)
    actual_sharpe = compute_sharpe(trades)
    n_trades = len(trades)

    if n_trades < 5:
        print(f"    Too few trades ({n_trades}), cannot test.")
        return {'passed': False, 'details': {'reason': 'too_few_trades'}}

    # Collect the tickers that were traded and their frequencies
    ticker_counts = defaultdict(int)
    for t in trades:
        ticker_counts[t['ticker']] += 1

    # For each permutation: pick random OOT dates for each ticker (same count per ticker)
    perm_sharpes = []
    for perm_i in range(n_perms):
        fake_trades = []
        for ticker, count in ticker_counts.items():
            df = data[ticker]
            oot_dates = df.index[df.index >= pd.Timestamp(OOT_START)]
            # Pick random entry dates, ensuring we can hold for HOLD_DAYS
            valid_dates = oot_dates[:-HOLD_DAYS] if len(oot_dates) > HOLD_DAYS else oot_dates
            if len(valid_dates) < count:
                chosen = valid_dates
            else:
                chosen = np.random.choice(valid_dates, size=count, replace=False)

            for entry_date in chosen:
                entry_idx = df.index.get_loc(entry_date)
                exit_idx = min(entry_idx + HOLD_DAYS, len(df) - 1)
                if exit_idx <= entry_idx:
                    continue
                entry_price = df['Close'].iloc[entry_idx]
                exit_price = df['Close'].iloc[exit_idx]
                gross_return = (exit_price - entry_price) / entry_price
                net_return = gross_return - SLIPPAGE_RT
                fake_trades.append({
                    'net_return': float(net_return),
                    'entry_date': str(pd.Timestamp(entry_date).date()),
                    'exit_date': str(pd.Timestamp(df.index[exit_idx]).date()),
                })

        if len(fake_trades) >= 2:
            returns = [t['net_return'] for t in fake_trades]
            # Simple Sharpe approximation
            avg_hold = HOLD_DAYS * 1.4  # calendar days approx
            first_d = pd.Timestamp(OOT_START)
            last_d = pd.Timestamp(END_DATE)
            years = (last_d - first_d).days / 365.25
            tpy = len(fake_trades) / years
            s = compute_sharpe_from_returns(returns, tpy, avg_hold)
            perm_sharpes.append(s)

    perm_sharpes = sorted(perm_sharpes)
    p95 = np.percentile(perm_sharpes, 95) if perm_sharpes else 0
    p99 = np.percentile(perm_sharpes, 99) if perm_sharpes else 0

    passed = actual_sharpe > p95
    print(f"    Actual Sharpe: {actual_sharpe:.3f}")
    print(f"    Random 95th pct: {p95:.3f}, 99th pct: {p99:.3f}")
    print(f"    {'PASS' if passed else 'FAIL'} — actual {'beats' if passed else 'does NOT beat'} 95th percentile of random timing")

    return {
        'passed': passed,
        'details': {
            'actual_sharpe': actual_sharpe,
            'random_p50': round(float(np.median(perm_sharpes)), 3) if perm_sharpes else 0,
            'random_p95': round(float(p95), 3),
            'random_p99': round(float(p99), 3),
            'n_perms': n_perms,
        }
    }


# ── TEST 3: SUB-PERIOD STABILITY ────────────────────────────────────────
def test_subperiod(variant_name, signal_fn):
    """Split OOT into 3 equal sub-periods. PASS if all 3 Sharpes > 0."""
    print(f"\n  TEST 3 — SUB-PERIOD STABILITY ({variant_name})")

    oot_start = pd.Timestamp(OOT_START)
    oot_end = pd.Timestamp(END_DATE)
    total_days = (oot_end - oot_start).days
    third = total_days // 3

    periods = [
        (oot_start, oot_start + pd.Timedelta(days=third)),
        (oot_start + pd.Timedelta(days=third), oot_start + pd.Timedelta(days=2*third)),
        (oot_start + pd.Timedelta(days=2*third), oot_end),
    ]

    sigs = {t: signal_fn(data[t]) for t in data}
    sub_sharpes = []

    for i, (p_start, p_end) in enumerate(periods):
        trades, ec = run_backtest(sigs, oot_start=str(p_start.date()), oot_end=str(p_end.date()))
        if len(trades) < 2:
            s = 0.0
            print(f"    Period {i+1} ({p_start.date()} to {p_end.date()}): {len(trades)} trades, Sharpe=N/A (too few)")
        else:
            # Compute Sharpe for this sub-period
            returns = [t['net_return'] for t in trades]
            years = (p_end - p_start).days / 365.25
            tpy = len(trades) / years if years > 0 else len(trades)
            avg_hold = np.mean([(pd.Timestamp(t['exit_date']) - pd.Timestamp(t['entry_date'])).days for t in trades])
            s = compute_sharpe_from_returns(returns, tpy, avg_hold)
            total_pnl = sum(t['pnl'] for t in trades)
            print(f"    Period {i+1} ({p_start.date()} to {p_end.date()}): {len(trades)} trades, Sharpe={s:.3f}, PnL=${total_pnl:.2f}")
        sub_sharpes.append(s)

    all_positive = all(s > 0 for s in sub_sharpes)
    print(f"    {'PASS' if all_positive else 'FAIL'} — sub-period Sharpes: {sub_sharpes}")

    return {
        'passed': all_positive,
        'details': {
            'sub_sharpes': sub_sharpes,
            'periods': [f"{p[0].date()} to {p[1].date()}" for p in periods],
        }
    }


# ── TEST 4: TOP-TRADE REMOVAL ───────────────────────────────────────────
def test_top_trade_removal(variant_name, signal_fn, n_remove=5):
    """Remove top 5 trades by P&L. PASS if remaining Sharpe > 0.3."""
    print(f"\n  TEST 4 — TOP-TRADE REMOVAL ({variant_name})")

    sigs = {t: signal_fn(data[t]) for t in data}
    trades, ec = run_backtest(sigs)
    full_sharpe = compute_sharpe(trades)

    if len(trades) <= n_remove:
        print(f"    Too few trades to remove {n_remove}")
        return {'passed': False, 'details': {'reason': 'too_few_trades'}}

    # Sort by PnL descending, remove top n_remove
    sorted_trades = sorted(trades, key=lambda t: t['pnl'], reverse=True)
    removed = sorted_trades[:n_remove]
    remaining = sorted_trades[n_remove:]

    # Recompute Sharpe on remaining
    remaining_sharpe = compute_sharpe(remaining)

    removed_pnl = sum(t['pnl'] for t in removed)
    remaining_pnl = sum(t['pnl'] for t in remaining)

    print(f"    Full Sharpe: {full_sharpe:.3f} ({len(trades)} trades)")
    print(f"    Removed top {n_remove} trades (PnL: ${removed_pnl:.2f}):")
    for t in removed:
        print(f"      {t['ticker']} {t['entry_date']}: ${t['pnl']:.2f} ({t['net_return']:.1%})")
    print(f"    Remaining Sharpe: {remaining_sharpe:.3f} ({len(remaining)} trades, PnL: ${remaining_pnl:.2f})")

    passed = remaining_sharpe > 0.3
    print(f"    {'PASS' if passed else 'FAIL'} — remaining Sharpe {'>' if passed else '<='} 0.3")

    return {
        'passed': passed,
        'details': {
            'full_sharpe': full_sharpe,
            'remaining_sharpe': remaining_sharpe,
            'n_removed': n_remove,
            'removed_pnl': round(removed_pnl, 2),
            'remaining_pnl': round(remaining_pnl, 2),
            'remaining_trades': len(remaining),
        }
    }


# ── TEST 5: TICKER CONCENTRATION ────────────────────────────────────────
def test_ticker_concentration(variant_name, signal_fn, n_remove=3):
    """Remove top 3 tickers by total PnL. PASS if remaining Sharpe > 0.3."""
    print(f"\n  TEST 5 — TICKER CONCENTRATION ({variant_name})")

    sigs = {t: signal_fn(data[t]) for t in data}
    trades, ec = run_backtest(sigs)
    full_sharpe = compute_sharpe(trades)

    # Sum PnL by ticker
    ticker_pnl = defaultdict(float)
    ticker_count = defaultdict(int)
    for t in trades:
        ticker_pnl[t['ticker']] += t['pnl']
        ticker_count[t['ticker']] += 1

    # Sort tickers by total PnL descending
    sorted_tickers = sorted(ticker_pnl.items(), key=lambda x: x[1], reverse=True)

    print(f"    Full Sharpe: {full_sharpe:.3f} ({len(trades)} trades)")
    print(f"    PnL by ticker:")
    for tk, pnl in sorted_tickers:
        print(f"      {tk}: ${pnl:.2f} ({ticker_count[tk]} trades)")

    # Remove top n_remove tickers
    remove_tickers = set(tk for tk, _ in sorted_tickers[:n_remove])
    remaining_trades = [t for t in trades if t['ticker'] not in remove_tickers]

    if len(remaining_trades) < 2:
        print(f"    Too few remaining trades after removing {remove_tickers}")
        return {'passed': False, 'details': {'reason': 'too_few_remaining'}}

    remaining_sharpe = compute_sharpe(remaining_trades)
    removed_pnl = sum(pnl for tk, pnl in sorted_tickers[:n_remove])
    remaining_pnl = sum(t['pnl'] for t in remaining_trades)

    print(f"    Removed tickers: {remove_tickers} (PnL: ${removed_pnl:.2f})")
    print(f"    Remaining Sharpe: {remaining_sharpe:.3f} ({len(remaining_trades)} trades, PnL: ${remaining_pnl:.2f})")

    passed = remaining_sharpe > 0.3
    print(f"    {'PASS' if passed else 'FAIL'} — remaining Sharpe {'>' if passed else '<='} 0.3")

    return {
        'passed': passed,
        'details': {
            'full_sharpe': full_sharpe,
            'remaining_sharpe': remaining_sharpe,
            'removed_tickers': list(remove_tickers),
            'removed_pnl': round(removed_pnl, 2),
            'remaining_pnl': round(remaining_pnl, 2),
            'remaining_trades': len(remaining_trades),
        }
    }


# ── RUN ALL TESTS ────────────────────────────────────────────────────────
print("=" * 80)
print("INSIDER BUYING CLUSTER — ADVERSARIAL VALIDATION")
print("=" * 80)

np.random.seed(42)

variants = {
    'B_deep_value': {
        'signal_fn': signals_B,
        'inverse_fn': signals_B_inverse,
    },
    'F_multi_signal': {
        'signal_fn': signals_F,
        'inverse_fn': signals_F_inverse,
    },
}

results = {}

for vname, vcfg in variants.items():
    print(f"\n{'=' * 70}")
    print(f"VARIANT: {vname}")
    print(f"{'=' * 70}")

    sig_fn = vcfg['signal_fn']
    inv_fn = vcfg['inverse_fn']

    t1 = test_inverse(vname, sig_fn, inv_fn)
    t2 = test_random_timing(vname, sig_fn)
    t3 = test_subperiod(vname, sig_fn)
    t4 = test_top_trade_removal(vname, sig_fn)
    t5 = test_ticker_concentration(vname, sig_fn)

    results[vname] = {
        'inverse_direction': t1,
        'random_timing': t2,
        'subperiod_stability': t3,
        'top_trade_removal': t4,
        'ticker_concentration': t5,
    }

    passed_count = sum(1 for t in [t1, t2, t3, t4, t5] if t['passed'])
    results[vname]['total_passed'] = passed_count
    results[vname]['total_tests'] = 5

    print(f"\n  *** {vname} VERDICT: {passed_count}/5 adversarial tests passed ***")

# ── SAVE RESULTS ─────────────────────────────────────────────────────────
output_path = '/home/jupiter/Lvl3Quant/data/insider_buying_adversarial_results.json'
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to {output_path}")

# ── FINAL SUMMARY ────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("FINAL ADVERSARIAL VERDICT")
print(f"{'=' * 70}")

for vname in results:
    r = results[vname]
    p = r['total_passed']
    print(f"\n  {vname}: {p}/5 adversarial tests passed")
    for tname in ['inverse_direction', 'random_timing', 'subperiod_stability',
                   'top_trade_removal', 'ticker_concentration']:
        status = "PASS" if r[tname]['passed'] else "FAIL"
        print(f"    {tname}: {status}")

    if p >= 4:
        print(f"    -> STRONG EDGE — survives adversarial scrutiny")
    elif p >= 3:
        print(f"    -> MODERATE EDGE — some concerns")
    else:
        print(f"    -> WEAK/NO EDGE — likely spurious")

print(f"\n{'=' * 70}")
print("If inverse direction ALSO profits, or removing NVDA/META kills Sharpe,")
print("the strategy is just 'buying big tech in a bull market' — no real edge.")
print(f"{'=' * 70}")

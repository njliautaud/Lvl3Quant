#!/usr/bin/env python3
"""
Adversarial Validation: Bond Yield Signal + Bull Call Spread
6 tests to determine if the strategy has genuine edge or is overfit.

Strategy: Buy ATM call + sell OTM call (5% higher) when:
  - 10Y yield drops >0.1% over 5 trading days
  - Stock >5% below 20-day SMA
  - RSI(14) < 40
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from datetime import datetime, timedelta
import warnings
import sys
import time
warnings.filterwarnings('ignore')

# ── CONFIG ──────────────────────────────────────────────────────────────────
UNIVERSE = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'JPM', 'UNH', 'LLY', 'AVGO', 'AMD']
START = '2020-01-01'
END = '2026-07-01'
MAX_COST = 150.0
MAX_CONCURRENT = 2
RISK_FREE = 0.04
SLIPPAGE = 0.05  # 5% on option prices
IV_CAP = 0.80
IV_CRUSH = 0.85  # exit IV = entry IV * 0.85
DTE_TARGET = 30
SPREAD_WIDTH_PCT = 0.05  # 5% OTM for short leg
YIELD_DROP_THRESH = 0.001  # 0.1% in decimal
RSI_THRESH = 40
BELOW_SMA_THRESH = 0.05  # 5%
SMA_PERIOD = 20

# Exit rules
PROFIT_TARGET = 1.0    # 100% gain on premium
LOSS_LIMIT = 0.50      # 50% loss on premium
UND_GAIN_EXIT = 0.05   # +5% underlying move
MIN_DTE_EXIT = 7
MAX_HOLD_DAYS = 21

# Original benchmark
ORIG_SHARPE = 1.87
ORIG_TRADES = 96
ORIG_WR = 0.729

np.random.seed(42)


# ── HELPERS ─────────────────────────────────────────────────────────────────

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def bs_call_price(S, K, T, r, sigma):
    """Black-Scholes call price."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def spread_price(S, K_long, K_short, T, r, sigma):
    """Bull call spread price = long call - short call."""
    return bs_call_price(S, K_long, T, r, sigma) - bs_call_price(S, K_short, T, r, sigma)


def fetch_data():
    """Download all required price data."""
    print("Fetching price data...")
    # Stock data
    stocks = {}
    for sym in UNIVERSE:
        try:
            df = yf.download(sym, start=START, end=END, progress=False, auto_adjust=True)
            if len(df) > 100:
                stocks[sym] = df
                print(f"  {sym}: {len(df)} days")
            else:
                print(f"  {sym}: insufficient data ({len(df)} days), skipping")
        except Exception as e:
            print(f"  {sym}: download failed: {e}")
        time.sleep(0.2)

    # VIX
    vix = yf.download('^VIX', start=START, end=END, progress=False, auto_adjust=True)
    print(f"  VIX: {len(vix)} days")
    time.sleep(0.2)

    # 10Y yield
    tnx = yf.download('^TNX', start=START, end=END, progress=False, auto_adjust=True)
    print(f"  TNX: {len(tnx)} days")

    # SPY for market vol
    spy = yf.download('SPY', start=START, end=END, progress=False, auto_adjust=True)
    print(f"  SPY: {len(spy)} days")

    return stocks, vix, tnx, spy


def compute_iv(stock_close, spy_close, vix_close):
    """IV = VIX * (stock_vol / market_vol), capped at IV_CAP."""
    stock_vol = stock_close.pct_change().rolling(20).std() * np.sqrt(252)
    market_vol = spy_close.pct_change().rolling(20).std() * np.sqrt(252)
    ratio = (stock_vol / market_vol.reindex(stock_vol.index, method='ffill')).clip(0.5, 3.0)
    vix_aligned = vix_close.reindex(stock_vol.index, method='ffill') / 100.0
    iv = vix_aligned * ratio
    return iv.clip(upper=IV_CAP)


def generate_signals(stocks, tnx, yield_drop_thresh=YIELD_DROP_THRESH,
                     rsi_thresh=RSI_THRESH, below_sma_thresh=BELOW_SMA_THRESH,
                     sma_period=SMA_PERIOD):
    """Generate signal dates per stock."""
    # Yield change over 5 trading days
    tnx_close = tnx['Close'].squeeze()
    yield_change = tnx_close.diff(5) / tnx_close.shift(5)

    signals = {}  # sym -> list of dates
    for sym, df in stocks.items():
        close = df['Close'].squeeze()
        sma = close.rolling(sma_period).mean()
        rsi = compute_rsi(close)

        # Conditions
        below_sma = (close < sma * (1 - below_sma_thresh))
        rsi_low = (rsi < rsi_thresh)

        yc = yield_change.reindex(close.index, method='ffill')
        yield_drop = (yc < -yield_drop_thresh)

        combined = below_sma & rsi_low & yield_drop
        sig_dates = close.index[combined].tolist()
        if sig_dates:
            signals[sym] = sig_dates

    return signals


def simulate_trades(stocks, spy, vix, signals, spread_width_pct=SPREAD_WIDTH_PCT,
                    dte_target=DTE_TARGET, max_cost=MAX_COST, max_concurrent=MAX_CONCURRENT):
    """Simulate bull call spread trades from signals."""
    trades = []
    active_positions = []

    # Build a global timeline of signal events
    all_events = []
    for sym, dates in signals.items():
        for d in dates:
            all_events.append((d, sym))
    all_events.sort(key=lambda x: x[0])

    spy_close = spy['Close'].squeeze()
    vix_close = vix['Close'].squeeze()

    for entry_date, sym in all_events:
        # Check concurrent limit
        active_positions = [p for p in active_positions if p['exit_date'] > entry_date]
        if len(active_positions) >= max_concurrent:
            continue

        df = stocks[sym]
        close = df['Close'].squeeze()

        if entry_date not in close.index:
            continue

        S_entry = float(close.loc[entry_date])
        iv_series = compute_iv(close, spy_close, vix_close)

        if entry_date not in iv_series.index:
            continue
        iv_entry = float(iv_series.loc[entry_date])
        if np.isnan(iv_entry) or iv_entry <= 0:
            continue

        K_long = S_entry  # ATM
        K_short = S_entry * (1 + spread_width_pct)
        T_entry = dte_target / 365.0

        # Entry price
        entry_spread = spread_price(S_entry, K_long, K_short, T_entry, RISK_FREE, iv_entry)
        entry_cost = entry_spread * 100  # per contract
        if entry_cost <= 0 or entry_cost > max_cost:
            continue
        # Apply slippage (pay more)
        entry_cost_slipped = entry_cost * (1 + SLIPPAGE)

        # Simulate forward
        future_dates = close.index[close.index > entry_date]
        exit_date = None
        exit_pnl = 0.0
        exit_reason = 'max_hold'

        for i, d in enumerate(future_dates):
            if i >= MAX_HOLD_DAYS:
                break

            days_held = i + 1
            dte_remaining = dte_target - days_held
            S_now = float(close.loc[d])

            # Check underlying gain exit
            und_return = (S_now - S_entry) / S_entry
            if und_return >= UND_GAIN_EXIT:
                exit_reason = 'und_gain'
            elif dte_remaining <= MIN_DTE_EXIT:
                exit_reason = 'dte_exit'
            elif days_held >= MAX_HOLD_DAYS:
                exit_reason = 'max_hold'
            else:
                # Price the spread
                T_now = max(dte_remaining / 365.0, 1/365.0)
                iv_exit = iv_entry * IV_CRUSH
                current_spread = spread_price(S_now, K_long, K_short, T_now, RISK_FREE, iv_exit)
                current_val = current_spread * 100 * (1 - SLIPPAGE)  # slippage on exit

                pnl_pct = (current_val - entry_cost_slipped) / entry_cost_slipped

                if pnl_pct >= PROFIT_TARGET:
                    exit_reason = 'profit_target'
                elif pnl_pct <= -LOSS_LIMIT:
                    exit_reason = 'stop_loss'
                else:
                    continue

            # Compute exit value
            T_exit = max((dte_target - days_held) / 365.0, 1/365.0)
            iv_exit = iv_entry * IV_CRUSH
            exit_spread = spread_price(S_now, K_long, K_short, T_exit, RISK_FREE, iv_exit)
            exit_val = exit_spread * 100 * (1 - SLIPPAGE)

            exit_pnl = exit_val - entry_cost_slipped
            exit_date = d
            break

        if exit_date is None:
            # Max hold reached - exit at last available date
            last_idx = min(MAX_HOLD_DAYS - 1, len(future_dates) - 1)
            if last_idx < 0:
                continue
            exit_date = future_dates[last_idx]
            S_now = float(close.loc[exit_date])
            T_exit = max((dte_target - MAX_HOLD_DAYS) / 365.0, 1/365.0)
            iv_exit = iv_entry * IV_CRUSH
            exit_spread = spread_price(S_now, K_long, K_short, T_exit, RISK_FREE, iv_exit)
            exit_val = exit_spread * 100 * (1 - SLIPPAGE)
            exit_pnl = exit_val - entry_cost_slipped

        trades.append({
            'sym': sym,
            'entry_date': entry_date,
            'exit_date': exit_date,
            'entry_cost': entry_cost_slipped,
            'exit_pnl': exit_pnl,
            'exit_reason': exit_reason,
            'S_entry': S_entry,
            'K_long': K_long,
            'K_short': K_short,
            'iv_entry': iv_entry,
        })

        active_positions.append({'exit_date': exit_date})

    return trades


def compute_metrics(trades):
    """Compute Sharpe, WR, total PnL from trade list."""
    if not trades:
        return {'sharpe': 0.0, 'wr': 0.0, 'n_trades': 0, 'total_pnl': 0.0, 'avg_pnl': 0.0}

    pnls = [t['exit_pnl'] for t in trades]
    n = len(pnls)
    wins = sum(1 for p in pnls if p > 0)
    wr = wins / n if n > 0 else 0

    avg = np.mean(pnls)
    std = np.std(pnls, ddof=1) if n > 1 else 1.0
    sharpe = (avg / std) * np.sqrt(252 / 15) if std > 0 else 0.0  # ~15 day avg hold, annualize

    return {
        'sharpe': sharpe,
        'wr': wr,
        'n_trades': n,
        'total_pnl': sum(pnls),
        'avg_pnl': avg,
        'pnls': pnls,
    }


# ── TEST 1: RE-IMPLEMENTATION ──────────────────────────────────────────────

def test1_reimplementation(stocks, spy, vix, tnx):
    print("\n" + "="*70)
    print("TEST 1: RE-IMPLEMENTATION FROM SCRATCH")
    print("="*70)

    signals = generate_signals(stocks, tnx)
    total_sigs = sum(len(v) for v in signals.values())
    print(f"  Signal dates generated: {total_sigs} across {len(signals)} stocks")

    trades = simulate_trades(stocks, spy, vix, signals)
    m = compute_metrics(trades)

    print(f"  Re-impl: Sharpe={m['sharpe']:.2f}, Trades={m['n_trades']}, WR={m['wr']:.1%}")
    print(f"  Original: Sharpe={ORIG_SHARPE:.2f}, Trades={ORIG_TRADES}, WR={ORIG_WR:.1%}")

    sharpe_ratio = m['sharpe'] / ORIG_SHARPE if ORIG_SHARPE != 0 else 0
    passed = sharpe_ratio >= 0.70  # within 30%
    print(f"  Sharpe ratio vs original: {sharpe_ratio:.2f} (need >= 0.70)")
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")

    return passed, m, trades, signals


# ── TEST 2: INVERSE SIGNAL ─────────────────────────────────────────────────

def test2_inverse_signal(stocks, spy, vix, tnx, orig_sharpe):
    print("\n" + "="*70)
    print("TEST 2: INVERSE SIGNAL (yield RISES instead of drops)")
    print("="*70)

    # Inverse: yield RISES >0.1%
    tnx_close = tnx['Close'].squeeze()
    yield_change = tnx_close.diff(5) / tnx_close.shift(5)

    signals = {}
    for sym, df in stocks.items():
        close = df['Close'].squeeze()
        sma = close.rolling(SMA_PERIOD).mean()
        rsi = compute_rsi(close)

        below_sma = (close < sma * (1 - BELOW_SMA_THRESH))
        rsi_low = (rsi < RSI_THRESH)
        yc = yield_change.reindex(close.index, method='ffill')
        yield_rise = (yc > YIELD_DROP_THRESH)  # INVERSE: rise instead of drop

        combined = below_sma & rsi_low & yield_rise
        sig_dates = close.index[combined].tolist()
        if sig_dates:
            signals[sym] = sig_dates

    total_sigs = sum(len(v) for v in signals.values())
    print(f"  Inverse signal dates: {total_sigs}")

    trades = simulate_trades(stocks, spy, vix, signals)
    m = compute_metrics(trades)

    print(f"  Inverse: Sharpe={m['sharpe']:.2f}, Trades={m['n_trades']}, WR={m['wr']:.1%}")
    threshold = 0.50 * orig_sharpe
    passed = m['sharpe'] < threshold
    print(f"  Need inverse Sharpe < {threshold:.2f} (50% of original {orig_sharpe:.2f})")
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")

    return passed, m


# ── TEST 3: RANDOM TIMING PERMUTATION ──────────────────────────────────────

def test3_random_permutation(stocks, spy, vix, tnx, orig_sharpe, n_perms=200):
    print("\n" + "="*70)
    print(f"TEST 3: RANDOM TIMING PERMUTATION ({n_perms} shuffles)")
    print("="*70)

    # Get all dates where stock is >5% below SMA (eligible dates pool)
    eligible_dates = {}
    for sym, df in stocks.items():
        close = df['Close'].squeeze()
        sma = close.rolling(SMA_PERIOD).mean()
        below = (close < sma * (1 - BELOW_SMA_THRESH))
        eligible_dates[sym] = close.index[below].tolist()

    # Get original signal counts per stock
    orig_signals = generate_signals(stocks, tnx)
    orig_counts = {sym: len(dates) for sym, dates in orig_signals.items()}

    orig_trades = simulate_trades(stocks, spy, vix, orig_signals)
    orig_m = compute_metrics(orig_trades)
    orig_s = orig_m['sharpe']
    print(f"  Original Sharpe (re-impl): {orig_s:.2f}")

    perm_sharpes = []
    for i in range(n_perms):
        # Random signals: same count per stock, random eligible dates
        rand_signals = {}
        for sym, count in orig_counts.items():
            if sym in eligible_dates and count > 0:
                pool = eligible_dates[sym]
                if len(pool) >= count:
                    chosen = list(np.random.choice(pool, size=min(count, len(pool)), replace=False))
                    rand_signals[sym] = chosen

        trades = simulate_trades(stocks, spy, vix, rand_signals)
        m = compute_metrics(trades)
        perm_sharpes.append(m['sharpe'])

        if (i+1) % 50 == 0:
            print(f"  ... {i+1}/{n_perms} permutations done")

    perm_sharpes = np.array(perm_sharpes)
    p_value = np.mean(perm_sharpes >= orig_s)
    print(f"  Permutation Sharpes: mean={np.mean(perm_sharpes):.2f}, "
          f"std={np.std(perm_sharpes):.2f}, max={np.max(perm_sharpes):.2f}")
    print(f"  Original Sharpe: {orig_s:.2f}")
    print(f"  p-value: {p_value:.4f} (need < 0.05)")

    passed = p_value < 0.05
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")

    return passed, p_value, perm_sharpes


# ── TEST 4: SUB-PERIOD STABILITY ───────────────────────────────────────────

def test4_subperiod(trades):
    print("\n" + "="*70)
    print("TEST 4: SUB-PERIOD STABILITY (4 equal periods)")
    print("="*70)

    if not trades:
        print("  No trades to analyze")
        print("  RESULT: FAIL")
        return False, []

    df = pd.DataFrame(trades)
    df['entry_date'] = pd.to_datetime(df['entry_date'])
    df = df.sort_values('entry_date')

    date_range = pd.date_range(start=START, end=END, periods=5)
    period_sharpes = []

    for i in range(4):
        mask = (df['entry_date'] >= date_range[i]) & (df['entry_date'] < date_range[i+1])
        period_trades = df[mask].to_dict('records')
        m = compute_metrics(period_trades)
        period_sharpes.append(m['sharpe'])
        print(f"  Period {i+1} ({date_range[i].strftime('%Y-%m')}"
              f" to {date_range[i+1].strftime('%Y-%m')}): "
              f"Sharpe={m['sharpe']:.2f}, Trades={m['n_trades']}, WR={m['wr']:.1%}")

    positive_count = sum(1 for s in period_sharpes if s > 0)
    worst = min(period_sharpes)

    passed = (positive_count >= 3) and (worst > -0.5)
    print(f"  Positive periods: {positive_count}/4 (need >= 3)")
    print(f"  Worst period Sharpe: {worst:.2f} (need > -0.50)")
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")

    return passed, period_sharpes


# ── TEST 5: TOP-3 STOCK REMOVAL ────────────────────────────────────────────

def test5_stock_removal(trades, orig_sharpe):
    print("\n" + "="*70)
    print("TEST 5: TOP-3 STOCK REMOVAL")
    print("="*70)

    if not trades:
        print("  No trades to analyze")
        print("  RESULT: FAIL")
        return False, {}

    df = pd.DataFrame(trades)
    stock_pnl = df.groupby('sym')['exit_pnl'].sum().sort_values(ascending=False)
    print("  P&L by stock:")
    for sym, pnl in stock_pnl.items():
        print(f"    {sym}: ${pnl:.0f}")

    top3 = stock_pnl.head(3).index.tolist()
    print(f"  Removing top 3: {top3}")

    remaining = [t for t in trades if t['sym'] not in top3]
    m = compute_metrics(remaining)

    threshold = 0.50 * orig_sharpe
    print(f"  Remaining: Sharpe={m['sharpe']:.2f}, Trades={m['n_trades']}, WR={m['wr']:.1%}")
    print(f"  Need Sharpe > {threshold:.2f} (50% of {orig_sharpe:.2f})")

    passed = m['sharpe'] > threshold
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")

    return passed, stock_pnl


# ── TEST 6: PARAMETER SENSITIVITY ──────────────────────────────────────────

def test6_param_sensitivity(stocks, spy, vix, tnx, n_combos=100):
    print("\n" + "="*70)
    print(f"TEST 6: PARAMETER SENSITIVITY ({n_combos} random combos)")
    print("="*70)

    sharpes = []
    for i in range(n_combos):
        # Random params
        yld_thresh = np.random.uniform(0.0005, 0.002)   # 0.05% to 0.20%
        rsi_th = np.random.uniform(25, 45)
        below_sma_th = np.random.uniform(0.03, 0.10)
        sw = np.random.uniform(0.03, 0.08)
        dte = int(np.random.uniform(20, 45))

        signals = generate_signals(stocks, tnx,
                                   yield_drop_thresh=yld_thresh,
                                   rsi_thresh=rsi_th,
                                   below_sma_thresh=below_sma_th)

        trades = simulate_trades(stocks, spy, vix, signals,
                                 spread_width_pct=sw,
                                 dte_target=dte)

        m = compute_metrics(trades)
        sharpes.append(m['sharpe'])

        if (i+1) % 25 == 0:
            above = sum(1 for s in sharpes if s > 0.30)
            print(f"  ... {i+1}/{n_combos} combos done ({above}/{len(sharpes)} above 0.30)")

    sharpes = np.array(sharpes)
    above_threshold = np.mean(sharpes > 0.30)
    print(f"  Sharpe distribution: mean={np.mean(sharpes):.2f}, "
          f"median={np.median(sharpes):.2f}, std={np.std(sharpes):.2f}")
    print(f"  Min={np.min(sharpes):.2f}, Max={np.max(sharpes):.2f}")
    print(f"  Combos with Sharpe > 0.30: {above_threshold:.1%} (need > 50%)")

    passed = above_threshold > 0.50
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")

    return passed, sharpes


# ── MAIN ────────────────────────────────────────────────────────────────────

def main():
    print("="*70)
    print("ADVERSARIAL VALIDATION: Bond Yield Signal + Bull Call Spread")
    print(f"Universe: {', '.join(UNIVERSE)}")
    print(f"Period: {START} to {END}")
    print(f"Original benchmark: Sharpe={ORIG_SHARPE}, Trades={ORIG_TRADES}, WR={ORIG_WR:.1%}")
    print("="*70)

    stocks, vix, tnx, spy = fetch_data()

    if len(stocks) < 5:
        print("FATAL: Too few stocks downloaded. Aborting.")
        sys.exit(1)

    results = {}

    # Test 1
    t1_pass, t1_metrics, trades, signals = test1_reimplementation(stocks, spy, vix, tnx)
    results['T1_reimplementation'] = t1_pass
    reimpl_sharpe = t1_metrics['sharpe']

    # Test 2
    t2_pass, t2_metrics = test2_inverse_signal(stocks, spy, vix, tnx, reimpl_sharpe)
    results['T2_inverse_signal'] = t2_pass

    # Test 3
    t3_pass, t3_pval, t3_perms = test3_random_permutation(stocks, spy, vix, tnx, reimpl_sharpe)
    results['T3_random_permutation'] = t3_pass

    # Test 4
    t4_pass, t4_sharpes = test4_subperiod(trades)
    results['T4_subperiod_stability'] = t4_pass

    # Test 5
    t5_pass, t5_stock_pnl = test5_stock_removal(trades, reimpl_sharpe)
    results['T5_stock_removal'] = t5_pass

    # Test 6
    t6_pass, t6_sharpes = test6_param_sensitivity(stocks, spy, vix, tnx)
    results['T6_param_sensitivity'] = t6_pass

    # ── SUMMARY ─────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("ADVERSARIAL VALIDATION SUMMARY")
    print("="*70)
    pass_count = 0
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        icon = "+" if passed else "X"
        print(f"  [{icon}] {name}: {status}")
        if passed:
            pass_count += 1

    print(f"\n  Score: {pass_count}/6 tests passed")

    if pass_count >= 5:
        print("  VERDICT: STRONG — Strategy shows genuine edge. DEPLOY candidate.")
    elif pass_count >= 4:
        print("  VERDICT: MODERATE — Strategy has edge but with caveats. Proceed with caution.")
    elif pass_count >= 3:
        print("  VERDICT: WEAK — Strategy may have some edge but significant concerns remain.")
    else:
        print("  VERDICT: REJECT — Strategy does NOT pass adversarial validation. Do NOT deploy.")

    return results


if __name__ == '__main__':
    results = main()

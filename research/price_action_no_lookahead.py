#!/usr/bin/env python3
"""
Price Action Dip-Buy Study — Look-Ahead Bias Correction
========================================================
Original study found RSI<35 dip-buying with triple filter gave Sharpe 10.17,
but used scipy.signal.argrelextrema(order=5) which peeks 5 bars forward.

This script tests causal (no look-ahead) alternatives:
  V1: "Near trailing low" — close within 2% of 10d trailing low
  V2: "Confirmed bounce" — yesterday within 2% of 10d low AND today > yesterday
  V_BIASED: Original argrelextrema approach (for comparison)

Filters applied on top of RSI<35:
  - Within 2% of 50d trailing low
  - Below 200-day SMA
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.signal import argrelextrema
import warnings
warnings.filterwarnings('ignore')

# ── CONFIG ──────────────────────────────────────────────────────────────
TICKERS = ['XLK', 'XLP', 'XLC', 'XLY', 'XLF', 'XLI', 'XLV', 'XLE', 'XLU', 'XLB', 'XLRE', 'SPY']
START = '2019-01-01'
END = '2026-08-20'
RSI_PERIOD = 14
RSI_THRESHOLD = 35
HOLD_DAYS = 5
N_PERMUTATIONS = 1000
SEED = 42

# ── HELPERS ─────────────────────────────────────────────────────────────

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def sharpe(returns: np.ndarray) -> float:
    if len(returns) < 2 or np.std(returns) == 0:
        return 0.0
    return np.mean(returns) / np.std(returns) * np.sqrt(252 / HOLD_DAYS)


def profit_factor(returns: np.ndarray) -> float:
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    if losses == 0:
        return float('inf') if gains > 0 else 0.0
    return gains / losses


def calc_stats(returns: np.ndarray) -> dict:
    if len(returns) == 0:
        return {'count': 0, 'avg_5d_ret': 0, 'sharpe': 0, 'wr': 0, 'pf': 0}
    return {
        'count': len(returns),
        'avg_5d_ret': np.mean(returns) * 100,
        'sharpe': sharpe(returns),
        'wr': (returns > 0).mean() * 100,
        'pf': profit_factor(returns),
    }


def permutation_test(returns: np.ndarray, n_perms: int = 1000, seed: int = 42) -> float:
    """One-sided permutation test: H0 = mean return <= 0."""
    if len(returns) < 5:
        return 1.0
    rng = np.random.RandomState(seed)
    observed = np.mean(returns)
    count_ge = 0
    for _ in range(n_perms):
        signs = rng.choice([-1, 1], size=len(returns))
        if np.mean(returns * signs) >= observed:
            count_ge += 1
    return count_ge / n_perms


# ── DOWNLOAD DATA ──────────────────────────────────────────────────────

print("Downloading data...")
raw = yf.download(TICKERS, start=START, end=END, auto_adjust=True, progress=False)
spy_raw = yf.download('SPY', start=START, end=END, auto_adjust=True, progress=False)

# SPY regime: green day = close > prior close
spy_close = spy_raw['Close'].squeeze()
spy_daily_ret = spy_close.pct_change()

print(f"Data range: {raw.index[0].date()} to {raw.index[-1].date()}")
print(f"Trading days: {len(raw)}")
print()

# ── BUILD SIGNALS PER TICKER ───────────────────────────────────────────

all_entries = {
    'biased': [],
    'v1_near_trail_low': [],
    'v2_bounce_confirm': [],
}

for ticker in TICKERS:
    try:
        close = raw['Close'][ticker].dropna()
    except Exception:
        close = raw['Close'].dropna()

    if len(close) < 250:
        print(f"  {ticker}: insufficient data ({len(close)} bars), skipping")
        continue

    rsi = compute_rsi(close, RSI_PERIOD)
    sma200 = close.rolling(200).mean()
    trail_50d_low = close.rolling(50).min()
    trail_10d_low = close.rolling(10).min()
    fwd_ret = close.pct_change(HOLD_DAYS).shift(-HOLD_DAYS)

    # Common filters (no bias)
    rsi_filter = rsi < RSI_THRESHOLD
    near_50d_low = close <= trail_50d_low * 1.02
    below_200sma = close < sma200
    base_filter = rsi_filter & near_50d_low & below_200sma

    # ── BIASED: argrelextrema swing low within 5 days ──
    close_arr = close.values
    swing_low_idx = argrelextrema(close_arr, np.less_equal, order=5)[0]
    swing_low_mask = pd.Series(False, index=close.index)
    for idx in swing_low_idx:
        # Mark 5 days around swing low
        start_i = max(0, idx - 5)
        end_i = min(len(close), idx + 6)
        swing_low_mask.iloc[start_i:end_i] = True

    biased_filter = base_filter & swing_low_mask
    valid = biased_filter & fwd_ret.notna()
    all_entries['biased'].extend(fwd_ret[valid].values.tolist())

    # ── V1: Near trailing 10d low (pure lookback) ──
    near_10d_low = close <= trail_10d_low * 1.02
    v1_filter = base_filter & near_10d_low
    valid = v1_filter & fwd_ret.notna()
    all_entries['v1_near_trail_low'].extend(fwd_ret[valid].values.tolist())

    # ── V2: Confirmed bounce off 10d low ──
    yesterday_near_10d_low = (close.shift(1) <= trail_10d_low.shift(1) * 1.02)
    today_up = close > close.shift(1)
    v2_filter = base_filter & yesterday_near_10d_low & today_up
    valid = v2_filter & fwd_ret.notna()
    all_entries['v2_bounce_confirm'].extend(fwd_ret[valid].values.tolist())

# ── ALSO BUILD REGIME-STRATIFIED ENTRIES ───────────────────────────────

regime_entries = {variant: {'green': [], 'red': []} for variant in all_entries}

for ticker in TICKERS:
    try:
        close = raw['Close'][ticker].dropna()
    except Exception:
        close = raw['Close'].dropna()

    if len(close) < 250:
        continue

    rsi = compute_rsi(close, RSI_PERIOD)
    sma200 = close.rolling(200).mean()
    trail_50d_low = close.rolling(50).min()
    trail_10d_low = close.rolling(10).min()
    fwd_ret = close.pct_change(HOLD_DAYS).shift(-HOLD_DAYS)

    rsi_filter = rsi < RSI_THRESHOLD
    near_50d_low = close <= trail_50d_low * 1.02
    below_200sma = close < sma200
    base_filter = rsi_filter & near_50d_low & below_200sma

    # Align SPY regime to ticker index
    spy_ret_aligned = spy_daily_ret.reindex(close.index)
    green_day = spy_ret_aligned > 0
    red_day = spy_ret_aligned <= 0

    # Biased
    close_arr = close.values
    swing_low_idx = argrelextrema(close_arr, np.less_equal, order=5)[0]
    swing_low_mask = pd.Series(False, index=close.index)
    for idx in swing_low_idx:
        start_i = max(0, idx - 5)
        end_i = min(len(close), idx + 6)
        swing_low_mask.iloc[start_i:end_i] = True
    biased_filter = base_filter & swing_low_mask
    valid = biased_filter & fwd_ret.notna()
    regime_entries['biased']['green'].extend(fwd_ret[valid & green_day].values.tolist())
    regime_entries['biased']['red'].extend(fwd_ret[valid & red_day].values.tolist())

    # V1
    near_10d_low = close <= trail_10d_low * 1.02
    v1_filter = base_filter & near_10d_low
    valid = v1_filter & fwd_ret.notna()
    regime_entries['v1_near_trail_low']['green'].extend(fwd_ret[valid & green_day].values.tolist())
    regime_entries['v1_near_trail_low']['red'].extend(fwd_ret[valid & red_day].values.tolist())

    # V2
    yesterday_near_10d_low = (close.shift(1) <= trail_10d_low.shift(1) * 1.02)
    today_up = close > close.shift(1)
    v2_filter = base_filter & yesterday_near_10d_low & today_up
    valid = v2_filter & fwd_ret.notna()
    regime_entries['v2_bounce_confirm']['green'].extend(fwd_ret[valid & green_day].values.tolist())
    regime_entries['v2_bounce_confirm']['red'].extend(fwd_ret[valid & red_day].values.tolist())


# ── RESULTS ────────────────────────────────────────────────────────────

print("=" * 80)
print("PRICE ACTION DIP-BUY STUDY — LOOK-AHEAD BIAS CORRECTION")
print("=" * 80)
print(f"Universe: {len(TICKERS)} sector ETFs + SPY | Period: {START} to {END}")
print(f"Entry: RSI<{RSI_THRESHOLD} + within 2% of 50d low + below 200 SMA + swing low variant")
print(f"Hold: {HOLD_DAYS} trading days | Exit: close on day {HOLD_DAYS}")
print()

# Original biased reference
print("-" * 80)
print("REFERENCE: Original biased Sharpe was reported as 10.17")
print("-" * 80)
print()

variant_labels = {
    'biased': 'BIASED (argrelextrema order=5, peeks forward)',
    'v1_near_trail_low': 'V1: Near 10d trailing low (within 2%, NO lookahead)',
    'v2_bounce_confirm': 'V2: Bounce off 10d low (confirmed, NO lookahead)',
}

best_variant = None
best_sharpe = -999

for variant, label in variant_labels.items():
    rets = np.array(all_entries[variant])
    stats = calc_stats(rets)

    print(f"{'─' * 70}")
    print(f"  {label}")
    print(f"{'─' * 70}")
    print(f"  Trades: {stats['count']:>6d}")
    print(f"  Avg 5d return: {stats['avg_5d_ret']:>+7.3f}%")
    print(f"  Sharpe: {stats['sharpe']:>8.2f}")
    print(f"  Win Rate: {stats['wr']:>7.1f}%")
    print(f"  Profit Factor: {stats['pf']:>7.2f}")

    # Regime stratification
    green_rets = np.array(regime_entries[variant]['green'])
    red_rets = np.array(regime_entries[variant]['red'])
    green_stats = calc_stats(green_rets)
    red_stats = calc_stats(red_rets)

    print(f"\n  Regime Stratification (SPY green/red entry day):")
    print(f"    GREEN days: n={green_stats['count']:>4d}  Sharpe={green_stats['sharpe']:>6.2f}  WR={green_stats['wr']:>5.1f}%")
    print(f"    RED days:   n={red_stats['count']:>4d}  Sharpe={red_stats['sharpe']:>6.2f}  WR={red_stats['wr']:>5.1f}%")

    # Regime gap check (HC #428)
    if max(abs(green_stats['sharpe']), abs(red_stats['sharpe'])) > 0:
        regime_gap = abs(green_stats['sharpe'] - red_stats['sharpe']) / max(abs(green_stats['sharpe']), abs(red_stats['sharpe']), 1e-9)
        gap_status = "FAIL (>0.50)" if regime_gap > 0.50 else "PASS (<=0.50)"
        print(f"    Regime gap: {regime_gap:.2f} → {gap_status}")

    if variant != 'biased' and stats['sharpe'] > best_sharpe and stats['count'] >= 20:
        best_sharpe = stats['sharpe']
        best_variant = variant

    print()

# ── PERMUTATION TEST ON BEST VARIANT ───────────────────────────────────

print("=" * 80)
print("PERMUTATION TEST")
print("=" * 80)

if best_variant and len(all_entries[best_variant]) >= 5:
    best_rets = np.array(all_entries[best_variant])
    p_val = permutation_test(best_rets, N_PERMUTATIONS, SEED)
    print(f"Best causal variant: {variant_labels[best_variant]}")
    print(f"  Mean 5d return: {np.mean(best_rets)*100:+.3f}%")
    print(f"  Permutation p-value (1000 shuffles): {p_val:.4f}")
    print(f"  Significant at 5%: {'YES' if p_val < 0.05 else 'NO'}")
    print(f"  Significant at 1%: {'YES' if p_val < 0.01 else 'NO'}")
else:
    print("No valid causal variant with >=20 trades found.")
    p_val = 1.0

# ── INFLATION ANALYSIS ─────────────────────────────────────────────────

print()
print("=" * 80)
print("LOOK-AHEAD BIAS INFLATION ANALYSIS")
print("=" * 80)

biased_stats = calc_stats(np.array(all_entries['biased']))
reported_biased_sharpe = 10.17  # from original study

for variant in ['v1_near_trail_low', 'v2_bounce_confirm']:
    rets = np.array(all_entries[variant])
    stats = calc_stats(rets)
    label = variant_labels[variant]

    if biased_stats['sharpe'] != 0:
        inflation_vs_reproduced = (biased_stats['sharpe'] - stats['sharpe']) / abs(biased_stats['sharpe']) * 100
    else:
        inflation_vs_reproduced = 0

    inflation_vs_reported = (reported_biased_sharpe - stats['sharpe']) / reported_biased_sharpe * 100

    print(f"\n{label}:")
    print(f"  Reproduced biased Sharpe: {biased_stats['sharpe']:.2f}")
    print(f"  Corrected Sharpe: {stats['sharpe']:.2f}")
    print(f"  Inflation vs reproduced: {inflation_vs_reproduced:+.1f}%")
    print(f"  Inflation vs reported (10.17): {inflation_vs_reported:+.1f}%")

# ── FINAL VERDICT ──────────────────────────────────────────────────────

print()
print("=" * 80)
print("FINAL VERDICT")
print("=" * 80)

if best_variant:
    best_stats = calc_stats(np.array(all_entries[best_variant]))

    # Regime check
    green_rets = np.array(regime_entries[best_variant]['green'])
    red_rets = np.array(regime_entries[best_variant]['red'])
    g_sharpe = sharpe(green_rets) if len(green_rets) > 1 else 0
    r_sharpe = sharpe(red_rets) if len(red_rets) > 1 else 0
    regime_gap = abs(g_sharpe - r_sharpe) / max(abs(g_sharpe), abs(r_sharpe), 1e-9)

    regime_ok = regime_gap <= 0.50
    stat_sig = p_val < 0.05
    sharpe_ok = best_stats['sharpe'] > 1.0
    count_ok = best_stats['count'] >= 30

    checks = {
        'Sharpe > 1.0': sharpe_ok,
        'Stat sig (p<0.05)': stat_sig,
        'Regime gap <= 0.50': regime_ok,
        'N >= 30 trades': count_ok,
    }

    print(f"\nBest causal variant: {variant_labels[best_variant]}")
    print(f"  Sharpe: {best_stats['sharpe']:.2f} | WR: {best_stats['wr']:.1f}% | PF: {best_stats['pf']:.2f} | N: {best_stats['count']}")
    print(f"  p-value: {p_val:.4f}")
    print()

    for check, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'}: {check}")

    all_pass = all(checks.values())
    some_pass = sum(checks.values()) >= 2 and sharpe_ok

    print()
    if all_pass:
        verdict = "ALIVE"
        print(f"  >>> VERDICT: ALIVE — Edge survives look-ahead correction. All checks pass.")
    elif some_pass:
        verdict = "PARTIAL"
        print(f"  >>> VERDICT: PARTIAL — Some edge remains but not all checks pass.")
        print(f"  >>> Look-ahead bias inflated the original result significantly.")
    else:
        verdict = "DEAD"
        print(f"  >>> VERDICT: DEAD — No meaningful edge after removing look-ahead bias.")
        print(f"  >>> The Sharpe 10.17 was substantially or entirely due to forward-peeking swing low detection.")
else:
    print("  >>> VERDICT: DEAD — No causal variant produced enough trades to evaluate.")

print()
print("=" * 80)

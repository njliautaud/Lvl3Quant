#!/usr/bin/env python3
"""
Price Action Structure as Timing Signal for Sector ETF Dip-Buying
HC #783 Research Study

Tests whether price action features (support/resistance proximity, swing structure,
multi-timeframe context) improve RSI<35 dip-buy entry timing for sector ETFs.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.signal import argrelextrema
from itertools import combinations
import sys
from datetime import datetime

# ============================================================
# CONFIG
# ============================================================
SECTOR_ETFS = ['XLK', 'XLP', 'XLC', 'XLY', 'XLF', 'XLI', 'XLV', 'XLE', 'XLU', 'XLB', 'XLRE']
RSI_THRESHOLD = 35
RSI_PERIOD = 14
FWD_RETURN_DAYS = 5
PERM_SHUFFLES = 1000
REGIME_GAP_THRESHOLD = 0.50
LOOKBACK_YEARS = 6  # request 6 to ensure 5+ usable
SWING_ORDER = 5
SEED = 42

np.random.seed(SEED)


def calc_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calc_atr(high, low, close, period=14):
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def find_swing_lows(low_series, order=5):
    """Return index positions of swing lows."""
    vals = low_series.values
    idx = argrelextrema(vals, np.less_equal, order=order)[0]
    return idx


def find_swing_highs(high_series, order=5):
    """Return index positions of swing highs."""
    vals = high_series.values
    idx = argrelextrema(vals, np.greater_equal, order=order)[0]
    return idx


def compute_features(df):
    """Compute all price action features for a single ETF dataframe."""
    c = df['Close'].copy()
    h = df['High'].copy()
    l = df['Low'].copy()

    # RSI
    df['rsi'] = calc_rsi(c, RSI_PERIOD)

    # ATR
    df['atr'] = calc_atr(h, l, c, 14)

    # Distance from N-day high/low (% terms)
    for n in [20, 50, 252]:
        roll_high = h.rolling(n).max()
        roll_low = l.rolling(n).min()
        df[f'dist_from_{n}d_high_pct'] = (c - roll_high) / roll_high * 100
        df[f'dist_from_{n}d_low_pct'] = (c - roll_low) / roll_low * 100

    # SMAs and position relative to them
    for n in [20, 50, 200]:
        sma = c.rolling(n).mean()
        df[f'sma_{n}'] = sma
        df[f'below_sma_{n}'] = (c < sma).astype(int)

    # Swing high/low analysis
    swing_low_idx = find_swing_lows(l, order=SWING_ORDER)
    swing_high_idx = find_swing_highs(h, order=SWING_ORDER)

    # Days since last swing low/high
    days_since_swing_low = np.full(len(df), np.nan)
    days_since_swing_high = np.full(len(df), np.nan)
    swing_low_price = np.full(len(df), np.nan)

    for i in range(len(df)):
        # Find most recent swing low before or at i
        past_lows = swing_low_idx[swing_low_idx <= i]
        if len(past_lows) > 0:
            last_low = past_lows[-1]
            days_since_swing_low[i] = i - last_low
            swing_low_price[i] = l.iloc[last_low]

        past_highs = swing_high_idx[swing_high_idx <= i]
        if len(past_highs) > 0:
            last_high = past_highs[-1]
            days_since_swing_high[i] = i - last_high

    df['days_since_swing_low'] = days_since_swing_low
    df['days_since_swing_high'] = days_since_swing_high

    # ATR-normalized distance from recent swing low
    df['atr_dist_from_swing_low'] = np.where(
        df['atr'] > 0,
        (c.values - swing_low_price) / df['atr'].values,
        np.nan
    )

    # Near multi-timeframe support flags
    df['near_50d_low'] = (df['dist_from_50d_low_pct'] < 2.0).astype(int)  # within 2% of 50d low
    df['near_252d_low'] = (df['dist_from_252d_low_pct'] < 5.0).astype(int)  # within 5% of 252d low
    df['near_20d_low'] = (df['dist_from_20d_low_pct'] < 1.0).astype(int)  # within 1% of 20d low

    # Forward returns
    df['fwd_5d_ret'] = c.shift(-FWD_RETURN_DAYS) / c - 1

    return df


def sharpe_ratio(returns):
    """Annualized Sharpe from daily-frequency returns (each entry = one trade's 5d return)."""
    if len(returns) < 2:
        return 0.0
    # These are 5-day holding period returns, ~52 observations per year per ETF
    # Annualize: assume ~252/5 = 50.4 periods per year
    periods_per_year = 252 / FWD_RETURN_DAYS
    mu = returns.mean()
    sigma = returns.std()
    if sigma == 0:
        return 0.0
    return mu / sigma * np.sqrt(periods_per_year)


def profit_factor(returns):
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    if losses == 0:
        return np.inf if gains > 0 else 0.0
    return gains / losses


def stats_dict(returns, label=""):
    """Compute summary stats for a set of returns."""
    n = len(returns)
    if n == 0:
        return {'label': label, 'n': 0, 'avg_ret': 0, 'sharpe': 0, 'wr': 0, 'pf': 0, 'med_ret': 0}
    return {
        'label': label,
        'n': n,
        'avg_ret': returns.mean() * 100,
        'med_ret': returns.median() * 100,
        'sharpe': sharpe_ratio(returns),
        'wr': (returns > 0).mean() * 100,
        'pf': profit_factor(returns),
    }


def permutation_test(base_returns, filtered_returns, n_perms=1000):
    """Test if filtered_returns mean is significantly higher than random subset of same size."""
    observed_diff = filtered_returns.mean() - base_returns.mean()
    count_ge = 0
    n_filtered = len(filtered_returns)
    base_vals = base_returns.values

    for _ in range(n_perms):
        perm_idx = np.random.choice(len(base_vals), size=n_filtered, replace=False)
        perm_mean = base_vals[perm_idx].mean()
        if perm_mean - base_returns.mean() >= observed_diff:
            count_ge += 1

    return count_ge / n_perms


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 80)
    print("PRICE ACTION STRUCTURE AS TIMING SIGNAL FOR SECTOR ETF DIP-BUYING")
    print(f"Run date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 80)

    # 1. Download data
    print(f"\n[1] Downloading {LOOKBACK_YEARS}yr daily data for {len(SECTOR_ETFS)} sector ETFs + SPY...")
    tickers = SECTOR_ETFS + ['SPY']
    end_date = datetime.now()
    start_date = end_date - pd.DateOffset(years=LOOKBACK_YEARS)

    raw = yf.download(tickers, start=start_date.strftime('%Y-%m-%d'),
                      end=end_date.strftime('%Y-%m-%d'), auto_adjust=True,
                      progress=False)

    # Get SPY returns for regime classification
    spy_close = raw['Close']['SPY']
    spy_daily_ret = spy_close.pct_change()
    regime_green = (spy_daily_ret > 0)  # True = green day

    # 2. Process each ETF
    print("[2] Computing price action features for each ETF...")
    all_entries = []

    for ticker in SECTOR_ETFS:
        try:
            etf_df = pd.DataFrame({
                'Open': raw['Open'][ticker],
                'High': raw['High'][ticker],
                'Low': raw['Low'][ticker],
                'Close': raw['Close'][ticker],
                'Volume': raw['Volume'][ticker],
            }).dropna()

            etf_df = compute_features(etf_df)

            # Filter to RSI < 35 entries
            rsi_entries = etf_df[etf_df['rsi'] < RSI_THRESHOLD].copy()
            rsi_entries['ticker'] = ticker

            # Add regime
            rsi_entries['regime_green'] = regime_green.reindex(rsi_entries.index)

            all_entries.append(rsi_entries)
        except Exception as e:
            print(f"  WARNING: {ticker} failed: {e}")

    entries = pd.concat(all_entries, ignore_index=False)
    entries = entries.dropna(subset=['fwd_5d_ret'])
    print(f"  Total RSI<{RSI_THRESHOLD} entries across all ETFs: {len(entries)}")

    # 3. Base case
    print("\n[3] BASE CASE: All RSI<35 entries")
    base = stats_dict(entries['fwd_5d_ret'], "All RSI<35")
    print(f"  Count: {base['n']}")
    print(f"  Avg 5d return: {base['avg_ret']:.3f}%")
    print(f"  Median 5d return: {base['med_ret']:.3f}%")
    print(f"  Sharpe (ann): {base['sharpe']:.3f}")
    print(f"  Win Rate: {base['wr']:.1f}%")
    print(f"  Profit Factor: {base['pf']:.2f}")

    # 4. Test filters
    print("\n[4] Testing price action filters on RSI<35 entries...")

    filters = {
        'near_50d_low': entries['near_50d_low'] == 1,
        'near_252d_low': entries['near_252d_low'] == 1,
        'near_20d_low': entries['near_20d_low'] == 1,
        'below_sma_200': entries['below_sma_200'] == 1,
        'below_sma_50': entries['below_sma_50'] == 1,
        'below_sma_20': entries['below_sma_20'] == 1,
        'swing_low_within_5d': entries['days_since_swing_low'] <= 5,
        'swing_low_within_10d': entries['days_since_swing_low'] <= 10,
        'atr_near_swing_low': entries['atr_dist_from_swing_low'] < 1.0,
        'atr_very_near_swing_low': entries['atr_dist_from_swing_low'] < 0.5,
        # Combo filters
        'near_50d_low + below_200sma': (entries['near_50d_low'] == 1) & (entries['below_sma_200'] == 1),
        'near_252d_low + below_200sma': (entries['near_252d_low'] == 1) & (entries['below_sma_200'] == 1),
        'near_50d_low + near_252d_low': (entries['near_50d_low'] == 1) & (entries['near_252d_low'] == 1),
        'near_50d_low + near_252d_low + below_200sma': (
            (entries['near_50d_low'] == 1) & (entries['near_252d_low'] == 1) & (entries['below_sma_200'] == 1)
        ),
        'near_20d_low + below_50sma': (entries['near_20d_low'] == 1) & (entries['below_sma_50'] == 1),
        'swing_low_5d + below_200sma': (entries['days_since_swing_low'] <= 5) & (entries['below_sma_200'] == 1),
        'swing_low_5d + near_50d_low': (entries['days_since_swing_low'] <= 5) & (entries['near_50d_low'] == 1),
        'atr_near + below_200sma': (entries['atr_dist_from_swing_low'] < 1.0) & (entries['below_sma_200'] == 1),
        'triple: near_50d + swing_5d + below_200sma': (
            (entries['near_50d_low'] == 1) & (entries['days_since_swing_low'] <= 5) & (entries['below_sma_200'] == 1)
        ),
        'far_from_252d_high (>15%)': entries['dist_from_252d_high_pct'] < -15,
        'far_from_252d_high (>20%)': entries['dist_from_252d_high_pct'] < -20,
        'far_from_252d_high + below_200sma': (entries['dist_from_252d_high_pct'] < -15) & (entries['below_sma_200'] == 1),
    }

    results = []
    for name, mask in filters.items():
        filtered = entries.loc[mask, 'fwd_5d_ret']
        if len(filtered) >= 15:  # minimum sample
            s = stats_dict(filtered, name)
            results.append(s)

    results_df = pd.DataFrame(results).sort_values('sharpe', ascending=False)
    print(f"\n  {'Filter':<50} {'N':>5} {'AvgRet%':>8} {'Sharpe':>7} {'WR%':>6} {'PF':>6}")
    print("  " + "-" * 82)
    for _, r in results_df.iterrows():
        pf_str = f"{r['pf']:.2f}" if r['pf'] < 100 else "inf"
        print(f"  {r['label']:<50} {r['n']:>5.0f} {r['avg_ret']:>8.3f} {r['sharpe']:>7.3f} {r['wr']:>6.1f} {pf_str:>6}")

    # 5. Best filter analysis
    best = results_df.iloc[0]
    best_name = best['label']
    best_mask = filters[best_name]
    best_returns = entries.loc[best_mask, 'fwd_5d_ret']

    print(f"\n[5] BEST FILTER: '{best_name}'")
    print(f"  Count: {best['n']:.0f}")
    print(f"  Avg 5d return: {best['avg_ret']:.3f}%")
    print(f"  Sharpe (ann): {best['sharpe']:.3f}")
    print(f"  Win Rate: {best['wr']:.1f}%")
    pf_val = best['pf']
    print(f"  Profit Factor: {pf_val:.2f}" if pf_val < 100 else f"  Profit Factor: inf")
    print(f"  Improvement over base Sharpe: {best['sharpe'] - base['sharpe']:.3f}")

    # 6. Permutation test
    print(f"\n[6] Permutation test ({PERM_SHUFFLES} shuffles)...")
    p_value = permutation_test(entries['fwd_5d_ret'], best_returns, PERM_SHUFFLES)
    print(f"  p-value: {p_value:.4f}")
    print(f"  Significant at 5%: {'YES' if p_value < 0.05 else 'NO'}")
    print(f"  Significant at 10%: {'YES' if p_value < 0.10 else 'NO'}")

    # 7. Regime stratification
    print(f"\n[7] Regime stratification (SPY close-to-close: green vs red day)")

    # Base case by regime
    base_green = entries.loc[entries['regime_green'] == True, 'fwd_5d_ret']
    base_red = entries.loc[entries['regime_green'] == False, 'fwd_5d_ret']
    base_green_stats = stats_dict(base_green, "Base RSI<35 (green days)")
    base_red_stats = stats_dict(base_red, "Base RSI<35 (red days)")

    # Best filter by regime
    best_entries = entries.loc[best_mask]
    filt_green = best_entries.loc[best_entries['regime_green'] == True, 'fwd_5d_ret']
    filt_red = best_entries.loc[best_entries['regime_green'] == False, 'fwd_5d_ret']
    filt_green_stats = stats_dict(filt_green, f"Best filter (green days)")
    filt_red_stats = stats_dict(filt_red, f"Best filter (red days)")

    print(f"\n  BASE CASE regime split:")
    print(f"    Green days: n={base_green_stats['n']}, Sharpe={base_green_stats['sharpe']:.3f}, WR={base_green_stats['wr']:.1f}%")
    print(f"    Red days:   n={base_red_stats['n']}, Sharpe={base_red_stats['sharpe']:.3f}, WR={base_red_stats['wr']:.1f}%")

    print(f"\n  BEST FILTER regime split:")
    print(f"    Green days: n={filt_green_stats['n']}, Sharpe={filt_green_stats['sharpe']:.3f}, WR={filt_green_stats['wr']:.1f}%")
    print(f"    Red days:   n={filt_red_stats['n']}, Sharpe={filt_red_stats['sharpe']:.3f}, WR={filt_red_stats['wr']:.1f}%")

    # Regime gap check (HC #428)
    sg = filt_green_stats['sharpe']
    sr = filt_red_stats['sharpe']
    max_abs = max(abs(sg), abs(sr))
    if max_abs > 0:
        regime_gap = abs(sg - sr) / max_abs
    else:
        regime_gap = 0.0

    print(f"\n  Regime gap (HC #428): {regime_gap:.3f}")
    print(f"  Threshold: {REGIME_GAP_THRESHOLD}")
    regime_pass = regime_gap <= REGIME_GAP_THRESHOLD
    print(f"  Regime check: {'PASS' if regime_pass else 'FAIL'}")

    # 8. Also check top 3 filters for regime robustness
    print(f"\n[8] Top 5 filters - regime robustness check:")
    print(f"  {'Filter':<50} {'Sharpe_G':>8} {'Sharpe_R':>8} {'Gap':>6} {'Pass':>5}")
    print("  " + "-" * 77)

    top5_verdicts = []
    for _, r in results_df.head(5).iterrows():
        fname = r['label']
        fmask = filters[fname]
        fe = entries.loc[fmask]
        fg = fe.loc[fe['regime_green'] == True, 'fwd_5d_ret']
        fr = fe.loc[fe['regime_green'] == False, 'fwd_5d_ret']
        sg_i = sharpe_ratio(fg) if len(fg) >= 5 else 0
        sr_i = sharpe_ratio(fr) if len(fr) >= 5 else 0
        mx = max(abs(sg_i), abs(sr_i))
        gap_i = abs(sg_i - sr_i) / mx if mx > 0 else 0
        passes = gap_i <= REGIME_GAP_THRESHOLD
        top5_verdicts.append(passes)
        print(f"  {fname:<50} {sg_i:>8.3f} {sr_i:>8.3f} {gap_i:>6.3f} {'PASS' if passes else 'FAIL':>5}")

    # 9. Per-ETF breakdown of best filter
    print(f"\n[9] Best filter per-ETF breakdown:")
    print(f"  {'ETF':<6} {'N':>5} {'AvgRet%':>8} {'Sharpe':>7} {'WR%':>6}")
    print("  " + "-" * 32)
    best_entries_full = entries.loc[best_mask].copy()
    for ticker in SECTOR_ETFS:
        te = best_entries_full[best_entries_full['ticker'] == ticker]
        if len(te) >= 3:
            ts = stats_dict(te['fwd_5d_ret'], ticker)
            print(f"  {ticker:<6} {ts['n']:>5} {ts['avg_ret']:>8.3f} {ts['sharpe']:>7.3f} {ts['wr']:>6.1f}")
        else:
            print(f"  {ticker:<6}     <3 entries, skipped")

    # 10. VERDICT
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)

    alpha_over_base = best['sharpe'] - base['sharpe']
    sig = p_value < 0.10

    if alpha_over_base > 0.3 and sig and regime_pass:
        verdict = "ALIVE"
        detail = (f"Best filter '{best_name}' adds {alpha_over_base:.2f} Sharpe over base RSI<35, "
                  f"p={p_value:.3f}, regime-robust (gap={regime_gap:.3f}).")
    elif alpha_over_base > 0.1 and (sig or regime_pass):
        verdict = "PARTIAL"
        issues = []
        if not sig:
            issues.append(f"not significant (p={p_value:.3f})")
        if not regime_pass:
            issues.append(f"regime gap too wide ({regime_gap:.3f})")
        detail = (f"Best filter '{best_name}' adds {alpha_over_base:.2f} Sharpe, "
                  f"but {', '.join(issues)}. Worth further investigation.")
    else:
        verdict = "DEAD"
        detail = (f"Best filter '{best_name}' adds only {alpha_over_base:.2f} Sharpe, "
                  f"p={p_value:.3f}, regime gap={regime_gap:.3f}. "
                  f"Price action structure does not reliably improve RSI<35 timing.")

    print(f"\n  {verdict}: {detail}")

    # Summary table for Discord
    print("\n" + "=" * 80)
    print("DISCORD SUMMARY")
    print("=" * 80)
    print(f"""
Price Action Structure Study - Sector ETF Dip-Buying

Base case (RSI<35): {base['n']} entries, {base['avg_ret']:.2f}% avg 5d ret, Sharpe {base['sharpe']:.2f}, WR {base['wr']:.0f}%
Best filter: {best_name}
  -> {best['n']:.0f} entries, {best['avg_ret']:.2f}% avg ret, Sharpe {best['sharpe']:.2f}, WR {best['wr']:.0f}%
  -> Sharpe improvement: +{alpha_over_base:.2f}
  -> Permutation p-value: {p_value:.3f} ({'significant' if sig else 'not significant'} at 10%)
  -> Regime gap: {regime_gap:.3f} ({'PASS' if regime_pass else 'FAIL'} vs 0.50 threshold)

Verdict: {verdict}
{detail}
""")


if __name__ == '__main__':
    main()

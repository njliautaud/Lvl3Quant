#!/usr/bin/env python3
"""
Composite Signal Stack Study — Sector ETF Dip-Buying
=====================================================
Tests stacking 3 filters on RSI<35 dip-buying for 11 sector ETFs:
  1. Price action bounce confirmation
  2. VIX > 25
  3. Earnings re-rating overlay (top-3 sectors by re-rating acceleration)

Combinations tested: A (1+2), B (1+3), C (2+3), D (all 3), E (any 2 of 3)
Plus base case (RSI<35 only) and each individual filter.

No look-ahead bias: all signals use trailing data available at close.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# ─── CONFIG ───────────────────────────────────────────────────────────────────
SECTOR_ETFS = ['XLK', 'XLP', 'XLC', 'XLY', 'XLF', 'XLI', 'XLV', 'XLE', 'XLU', 'XLB', 'XLRE']
BENCHMARKS = ['^VIX', 'SPY']
RSI_PERIOD = 14
RSI_THRESHOLD = 35
HOLD_DAYS = 5
VIX_THRESHOLD = 25
BOUNCE_LOW_PCT = 0.02  # within 2% of 50-day low
RERATING_TOP_N = 3
RERATING_LOOKBACK = 5
PERM_SHUFFLES = 1000
REGIME_GAP_LIMIT = 0.50
np.random.seed(42)

# ─── DATA DOWNLOAD ───────────────────────────────────────────────────────────
print("Downloading 5+ years of daily data...")
tickers = SECTOR_ETFS + BENCHMARKS
data = yf.download(tickers, start='2019-01-01', end='2026-08-21', auto_adjust=True, progress=False)

# Extract close prices
close = data['Close'].copy()
high = data['High'].copy()
low = data['Low'].copy()

# Rename VIX column
close.rename(columns={'^VIX': 'VIX'}, inplace=True)
high.rename(columns={'^VIX': 'VIX'}, inplace=True)
low.rename(columns={'^VIX': 'VIX'}, inplace=True)

print(f"Data range: {close.index[0].date()} to {close.index[-1].date()}, {len(close)} trading days")
print(f"Tickers with data: {[c for c in close.columns if close[c].notna().sum() > 100]}")

# ─── INDICATOR CALCULATIONS ──────────────────────────────────────────────────

def calc_rsi(series, period=14):
    """Standard RSI calculation."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calc_sma(series, period):
    return series.rolling(period, min_periods=period).mean()

# RSI for all ETFs
rsi = pd.DataFrame(index=close.index)
for etf in SECTOR_ETFS:
    rsi[etf] = calc_rsi(close[etf], RSI_PERIOD)

# SMA 200 for all ETFs
sma200 = pd.DataFrame(index=close.index)
for etf in SECTOR_ETFS:
    sma200[etf] = calc_sma(close[etf], 200)

# 50-day rolling low for bounce confirmation
low_50d = pd.DataFrame(index=close.index)
for etf in SECTOR_ETFS:
    low_50d[etf] = close[etf].rolling(50, min_periods=50).min()

# 10-day rolling low (for "yesterday near 10-day low")
low_10d = pd.DataFrame(index=close.index)
for etf in SECTOR_ETFS:
    low_10d[etf] = close[etf].rolling(10, min_periods=10).min()

# SPY daily returns for regime classification
spy_ret = close['SPY'].pct_change()

# Forward 5-day returns for trade outcomes
fwd_5d = pd.DataFrame(index=close.index)
for etf in SECTOR_ETFS:
    fwd_5d[etf] = close[etf].shift(-HOLD_DAYS) / close[etf] - 1

# ─── RE-RATING ACCELERATION ──────────────────────────────────────────────────
# Vol-adjusted relative momentum for each sector vs SPY, then 5-day change = "acceleration"
print("Computing re-rating acceleration...")

rel_mom_20d = pd.DataFrame(index=close.index)
for etf in SECTOR_ETFS:
    # 20-day relative return vs SPY
    etf_ret_20 = close[etf] / close[etf].shift(20) - 1
    spy_ret_20 = close['SPY'] / close['SPY'].shift(20) - 1
    rel_ret = etf_ret_20 - spy_ret_20
    # Vol-adjust: divide by 20-day rolling std of daily relative returns
    daily_rel = close[etf].pct_change() - close['SPY'].pct_change()
    vol = daily_rel.rolling(20, min_periods=15).std()
    rel_mom_20d[etf] = rel_ret / (vol + 1e-8)

# Re-rating acceleration = 5-day change in vol-adjusted relative momentum
rerating_accel = pd.DataFrame(index=close.index)
for etf in SECTOR_ETFS:
    rerating_accel[etf] = rel_mom_20d[etf] - rel_mom_20d[etf].shift(RERATING_LOOKBACK)

# Rank sectors each day, top-3 = True
rerating_top3 = pd.DataFrame(False, index=close.index, columns=SECTOR_ETFS)
for i in range(len(close)):
    row = rerating_accel.iloc[i]
    valid = row.dropna()
    if len(valid) >= RERATING_TOP_N:
        top = valid.nlargest(RERATING_TOP_N).index
        rerating_top3.iloc[i, rerating_top3.columns.isin(top)] = True

# ─── SIGNAL GENERATION ───────────────────────────────────────────────────────
print("Generating signals...")

# Base: RSI < 35
sig_base = rsi < RSI_THRESHOLD

# Filter 1: Bounce confirmation
# RSI<35 + price within 2% of 50-day low + below 200 SMA + yesterday near 10-day low + today closes higher than yesterday
sig_bounce = pd.DataFrame(False, index=close.index, columns=SECTOR_ETFS)
for etf in SECTOR_ETFS:
    c = close[etf]
    c_prev = c.shift(1)
    near_50d_low = (c <= low_50d[etf] * (1 + BOUNCE_LOW_PCT))
    below_sma200 = (c < sma200[etf])
    # Yesterday near 10-day trailing low: yesterday's close within 1% of its trailing 10-day low
    yesterday_near_10d = (c_prev <= low_10d[etf].shift(1) * 1.01)
    today_higher = (c > c_prev)
    sig_bounce[etf] = sig_base[etf] & near_50d_low & below_sma200 & yesterday_near_10d & today_higher

# Filter 2: VIX > 25
vix_high = close['VIX'] > VIX_THRESHOLD
sig_vix = pd.DataFrame(index=close.index, columns=SECTOR_ETFS)
for etf in SECTOR_ETFS:
    sig_vix[etf] = sig_base[etf] & vix_high

# Filter 3: Re-rating top-3
sig_rerating = pd.DataFrame(index=close.index, columns=SECTOR_ETFS)
for etf in SECTOR_ETFS:
    sig_rerating[etf] = sig_base[etf] & rerating_top3[etf]

# Combo A: Bounce + VIX>25
sig_A = sig_bounce & sig_vix.astype(bool)

# Combo B: Bounce + Re-rating top-3
sig_B = sig_bounce & sig_rerating.astype(bool)

# Combo C: VIX>25 + Re-rating top-3
sig_C = sig_vix.astype(bool) & sig_rerating.astype(bool)

# Combo D: All three
sig_D = sig_bounce & sig_vix.astype(bool) & sig_rerating.astype(bool)

# Combo E: Any 2 of 3
f1 = sig_bounce.astype(int)
f2 = sig_vix.astype(bool).astype(int)
f3 = sig_rerating.astype(bool).astype(int)
sig_E = (f1 + f2 + f3) >= 2
# E also requires base RSI<35
sig_E = sig_E & sig_base

# ─── TRADE EXTRACTION & STATS ────────────────────────────────────────────────
def extract_trades(signal_df, fwd_df, spy_returns):
    """Extract trades from signal matrix. Returns DataFrame of trades."""
    trades = []
    for etf in SECTOR_ETFS:
        sig = signal_df[etf].fillna(False)
        fwd = fwd_df[etf]
        for dt in sig.index:
            if sig.loc[dt] and not pd.isna(fwd.loc[dt]):
                # Regime: use trailing 20-day SPY return as of entry date
                spy_20d = spy_returns.loc[:dt].tail(20).sum() if len(spy_returns.loc[:dt]) >= 20 else 0
                regime = 'green' if spy_20d > 0 else 'red'
                trades.append({
                    'date': dt,
                    'etf': etf,
                    'fwd_5d_ret': fwd.loc[dt],
                    'regime': regime
                })
    if not trades:
        return pd.DataFrame(columns=['date', 'etf', 'fwd_5d_ret', 'regime'])
    return pd.DataFrame(trades)

def calc_stats(trades_df):
    """Compute summary stats from trades DataFrame."""
    if len(trades_df) == 0:
        return {'n_trades': 0, 'avg_ret': np.nan, 'sharpe': np.nan, 'wr': np.nan, 'pf': np.nan, 'trades_per_year': 0}

    rets = trades_df['fwd_5d_ret'].values
    n = len(rets)
    avg = np.mean(rets)
    std = np.std(rets, ddof=1) if n > 1 else np.nan

    # Annualized Sharpe: avg 5d return / std, annualized by sqrt(252/5)
    sharpe = (avg / std * np.sqrt(252 / HOLD_DAYS)) if std > 0 else np.nan

    wr = np.mean(rets > 0)

    gains = rets[rets > 0].sum()
    losses = -rets[rets < 0].sum()
    pf = gains / losses if losses > 0 else np.inf

    # Date range for trades/year
    date_range = (trades_df['date'].max() - trades_df['date'].min()).days / 365.25
    tpy = n / date_range if date_range > 0 else n

    return {
        'n_trades': n,
        'avg_ret': avg * 100,  # percent
        'sharpe': sharpe,
        'wr': wr * 100,
        'pf': pf,
        'trades_per_year': tpy
    }

def regime_analysis(trades_df):
    """Compute stats by regime and regime gap."""
    if len(trades_df) == 0:
        return {'sharpe_green': np.nan, 'sharpe_red': np.nan, 'gap': np.nan, 'pass': False}

    green = trades_df[trades_df['regime'] == 'green']
    red = trades_df[trades_df['regime'] == 'red']

    sg = calc_stats(green)['sharpe'] if len(green) > 5 else np.nan
    sr = calc_stats(red)['sharpe'] if len(red) > 5 else np.nan

    if np.isnan(sg) or np.isnan(sr):
        gap = np.nan
        passed = len(trades_df) < 20  # too few trades to judge
    else:
        denom = max(abs(sg), abs(sr))
        gap = abs(sg - sr) / denom if denom > 0 else 0
        passed = gap <= REGIME_GAP_LIMIT

    return {
        'sharpe_green': sg,
        'sharpe_red': sr,
        'gap': gap,
        'pass': passed,
        'n_green': len(green),
        'n_red': len(red)
    }

def permutation_test(trades_df, observed_sharpe, base_trades_df=None, n_perms=PERM_SHUFFLES):
    """Permutation test: randomly sample same N trades from the base RSI<35 pool.
    Tests whether the filter selects better trades than random RSI<35 entries."""
    if len(trades_df) < 10 or np.isnan(observed_sharpe) or base_trades_df is None:
        return np.nan

    base_rets = base_trades_df['fwd_5d_ret'].values
    n_sample = len(trades_df)
    count = 0
    for _ in range(n_perms):
        idx = np.random.choice(len(base_rets), size=n_sample, replace=False) if n_sample <= len(base_rets) else np.random.choice(len(base_rets), size=n_sample, replace=True)
        sample = base_rets[idx]
        avg = np.mean(sample)
        std = np.std(sample, ddof=1)
        shuf_sharpe = (avg / std * np.sqrt(252 / HOLD_DAYS)) if std > 0 else 0
        if shuf_sharpe >= observed_sharpe:
            count += 1
    return count / n_perms

# ─── RUN ALL STRATEGIES ──────────────────────────────────────────────────────
print("\nExtracting trades for all strategies...")

strategies = {
    'Base (RSI<35)': sig_base,
    'F1: Bounce': sig_bounce,
    'F2: VIX>25': sig_vix.astype(bool),
    'F3: Re-rating': sig_rerating.astype(bool),
    'A: Bounce+VIX': sig_A,
    'B: Bounce+Rerate': sig_B,
    'C: VIX+Rerate': sig_C,
    'D: ALL THREE': sig_D,
    'E: ANY 2 of 3': sig_E,
}

results = {}
all_trades = {}

for name, sig in strategies.items():
    trades = extract_trades(sig, fwd_5d, spy_ret)
    stats = calc_stats(trades)
    regime = regime_analysis(trades)
    results[name] = {**stats, **regime}
    all_trades[name] = trades
    print(f"  {name}: {stats['n_trades']} trades, Sharpe={stats['sharpe']:.2f}" if stats['n_trades'] > 0 else f"  {name}: 0 trades")

# ─── SUMMARY TABLE ───────────────────────────────────────────────────────────
print("\n" + "="*120)
print("COMPOSITE SIGNAL STACK STUDY — SECTOR ETF DIP-BUYING")
print("="*120)
print(f"Universe: {', '.join(SECTOR_ETFS)} | Period: {close.index[0].date()} to {close.index[-1].date()}")
print(f"Base signal: RSI({RSI_PERIOD}) < {RSI_THRESHOLD} | Hold: {HOLD_DAYS} days | VIX threshold: {VIX_THRESHOLD}")
print()

header = f"{'Strategy':<22} {'Trades':>7} {'Tr/Yr':>6} {'AvgRet%':>8} {'Sharpe':>7} {'WR%':>6} {'PF':>6} {'Sh_Grn':>7} {'Sh_Red':>7} {'Gap':>6} {'Regime':>7}"
print(header)
print("-"*len(header))

for name in strategies:
    r = results[name]
    gap_str = f"{r['gap']:.2f}" if not np.isnan(r.get('gap', np.nan)) else 'N/A'
    regime_str = 'PASS' if r.get('pass', False) else 'FAIL'
    sg = f"{r['sharpe_green']:.2f}" if not np.isnan(r.get('sharpe_green', np.nan)) else 'N/A'
    sr = f"{r['sharpe_red']:.2f}" if not np.isnan(r.get('sharpe_red', np.nan)) else 'N/A'
    pf_str = f"{r['pf']:.2f}" if r['pf'] != np.inf else 'Inf'

    print(f"{name:<22} {r['n_trades']:>7} {r['trades_per_year']:>6.1f} {r['avg_ret']:>8.3f} {r['sharpe']:>7.2f} {r['wr']:>6.1f} {pf_str:>6} {sg:>7} {sr:>7} {gap_str:>6} {regime_str:>7}")

# ─── FIND BEST COMBO ─────────────────────────────────────────────────────────
combo_names = ['A: Bounce+VIX', 'B: Bounce+Rerate', 'C: VIX+Rerate', 'D: ALL THREE', 'E: ANY 2 of 3']
valid_combos = {k: results[k] for k in combo_names if results[k]['n_trades'] >= 10}

if valid_combos:
    best_name = max(valid_combos, key=lambda k: valid_combos[k]['sharpe'] if not np.isnan(valid_combos[k]['sharpe']) else -999)
    best = results[best_name]

    print(f"\n{'='*80}")
    print(f"BEST COMBINATION: {best_name}")
    print(f"{'='*80}")
    print(f"  Trades: {best['n_trades']} ({best['trades_per_year']:.1f}/year)")
    print(f"  Avg 5d Return: {best['avg_ret']:.3f}%")
    print(f"  Annualized Sharpe: {best['sharpe']:.2f}")
    print(f"  Win Rate: {best['wr']:.1f}%")
    print(f"  Profit Factor: {best['pf']:.2f}" if best['pf'] != np.inf else f"  Profit Factor: Inf (no losers)")
    print(f"  Regime: Sharpe_green={best.get('sharpe_green', 'N/A')}, Sharpe_red={best.get('sharpe_red', 'N/A')}")
    gap_v = best.get('gap', np.nan)
    print(f"  Regime Gap: {gap_v:.2f} ({'PASS' if best.get('pass') else 'FAIL'})" if not np.isnan(gap_v) else f"  Regime Gap: N/A (insufficient data)")

    # Permutation test on best
    print(f"\n  Running permutation test ({PERM_SHUFFLES} shuffles)...")
    pval = permutation_test(all_trades[best_name], best['sharpe'], base_trades_df=all_trades['Base (RSI<35)'])
    print(f"  Permutation p-value: {pval:.4f}" if not np.isnan(pval) else "  Permutation p-value: N/A (too few trades)")
    if not np.isnan(pval):
        print(f"  Significance: {'YES (p<0.05)' if pval < 0.05 else 'NO (p>=0.05)'}")

    # Also run permutation on second-best if it exists
    if len(valid_combos) > 1:
        sorted_combos = sorted(valid_combos, key=lambda k: valid_combos[k]['sharpe'] if not np.isnan(valid_combos[k]['sharpe']) else -999, reverse=True)
        second_name = sorted_combos[1]
        second = results[second_name]
        pval2 = permutation_test(all_trades[second_name], second['sharpe'], base_trades_df=all_trades['Base (RSI<35)'])
        print(f"\n  Runner-up: {second_name}")
        print(f"    Sharpe={second['sharpe']:.2f}, Trades={second['n_trades']}, p={pval2:.4f}" if not np.isnan(pval2) else f"    Sharpe={second['sharpe']:.2f}, Trades={second['n_trades']}, p=N/A")

else:
    print("\nNo combination produced >= 10 trades. All combos may be too selective.")
    best_name = None

# ─── TRADE FREQUENCY ANALYSIS ────────────────────────────────────────────────
print(f"\n{'='*80}")
print("TRADE FREQUENCY ANALYSIS")
print(f"{'='*80}")
print(f"{'Strategy':<22} {'Total':>7} {'Per Year':>9} {'Assessment':<30}")
print("-"*70)

for name in strategies:
    r = results[name]
    n = r['n_trades']
    tpy = r['trades_per_year']
    if tpy > 50:
        assessment = "Good frequency"
    elif tpy > 20:
        assessment = "Moderate — tradeable"
    elif tpy > 5:
        assessment = "Low — selective but viable"
    elif tpy > 0:
        assessment = "VERY LOW — may be too selective"
    else:
        assessment = "NO TRADES"
    print(f"{name:<22} {n:>7} {tpy:>9.1f} {assessment:<30}")

# ─── VERDICT ──────────────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print("VERDICT & RECOMMENDATION")
print(f"{'='*80}")

if best_name:
    best_r = results[best_name]
    base_r = results['Base (RSI<35)']

    sharpe_lift = best_r['sharpe'] - base_r['sharpe'] if not np.isnan(base_r['sharpe']) else best_r['sharpe']

    print(f"\n1. BEST COMBO: {best_name}")
    print(f"   Sharpe improvement over base RSI<35: {sharpe_lift:+.2f} ({base_r['sharpe']:.2f} -> {best_r['sharpe']:.2f})")
    print(f"   Trade count: {best_r['n_trades']} ({best_r['trades_per_year']:.1f}/yr)")

    # Quality checks
    issues = []
    if best_r['trades_per_year'] < 5:
        issues.append("VERY LOW trade frequency (<5/yr) - may not be worth automating")
    if not best_r.get('pass', True):
        issues.append(f"FAILS regime gap test (gap={best_r.get('gap', 'N/A')})")
    if best_r['n_trades'] < 30:
        issues.append(f"Small sample ({best_r['n_trades']} trades) - results may not be robust")

    if issues:
        print("\n   CONCERNS:")
        for iss in issues:
            print(f"   - {iss}")

    # Compare individual filters vs combos
    print(f"\n2. INDIVIDUAL vs STACKED COMPARISON:")
    for fname in ['F1: Bounce', 'F2: VIX>25', 'F3: Re-rating']:
        fr = results[fname]
        print(f"   {fname}: Sharpe={fr['sharpe']:.2f}, {fr['n_trades']} trades ({fr['trades_per_year']:.1f}/yr)")

    print(f"\n3. STACKING EFFECT:")
    # Check if stacking actually helps vs best individual
    indiv_sharpes = {f: results[f]['sharpe'] for f in ['F1: Bounce', 'F2: VIX>25', 'F3: Re-rating'] if not np.isnan(results[f]['sharpe'])}
    best_indiv_name = max(indiv_sharpes, key=indiv_sharpes.get)
    best_individual_sharpe = indiv_sharpes[best_indiv_name]
    if best_r['sharpe'] > best_individual_sharpe + 0.05:  # meaningful improvement
        print(f"   Stacking IMPROVES over best individual ({best_indiv_name}: {best_individual_sharpe:.2f} -> {best_name}: {best_r['sharpe']:.2f})")
    elif best_r['sharpe'] >= best_individual_sharpe - 0.05:
        print(f"   Stacking is EQUIVALENT to best individual ({best_indiv_name}: {best_individual_sharpe:.2f} vs {best_name}: {best_r['sharpe']:.2f})")
        print(f"   But stacking cuts trades: {results[best_indiv_name]['n_trades']} -> {best_r['n_trades']}, with no Sharpe gain.")
        print(f"   Prefer the individual filter for more opportunities.")
    else:
        print(f"   Stacking does NOT improve over best individual ({best_indiv_name}: {best_individual_sharpe:.2f})")
        print(f"   Consider using {best_indiv_name} alone for more trade opportunities.")

    print(f"\n4. PRODUCTION RECOMMENDATION:")
    if best_r['sharpe'] > 1.5 and best_r.get('pass', False) and best_r['trades_per_year'] >= 5:
        print(f"   DEPLOY {best_name} — strong Sharpe, passes regime test, adequate frequency")
    elif best_r['sharpe'] > 1.5 and best_r['trades_per_year'] >= 5:
        regime_note = "FAILS regime test" if not best_r.get('pass', False) else ""
        print(f"   CONDITIONAL — good Sharpe but {regime_note}. Consider regime-adaptive sizing.")
    elif best_r['trades_per_year'] < 5:
        # Find best individual that's tradeable
        tradeable = [(f, results[f]) for f in ['F1: Bounce', 'F2: VIX>25', 'F3: Re-rating', 'E: ANY 2 of 3']
                     if results[f]['trades_per_year'] >= 10 and not np.isnan(results[f]['sharpe'])]
        if tradeable:
            best_tradeable = max(tradeable, key=lambda x: x[1]['sharpe'])
            print(f"   Best combo too selective ({best_r['trades_per_year']:.1f}/yr).")
            print(f"   Recommend: {best_tradeable[0]} (Sharpe={best_tradeable[1]['sharpe']:.2f}, {best_tradeable[1]['trades_per_year']:.1f}/yr)")
        else:
            print(f"   All stacked combos are too selective. Use best individual filter.")
    else:
        print(f"   MARGINAL — Sharpe {best_r['sharpe']:.2f} is below 1.5 threshold for deployment.")

else:
    print("\nNo valid combinations found. Individual filters may work but stacking is too restrictive")
    print("for this universe/timeframe. Recommend using best individual filter standalone.")

print(f"\n{'='*80}")
print("STUDY COMPLETE")
print(f"{'='*80}")

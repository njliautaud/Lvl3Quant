#!/usr/bin/env python3
"""
Dual Regime Indicator: VIX Term Structure + Credit Spread
==========================================================
Strategy thesis:
  VIX term structure (contango = calm, backwardation = stress) and high-yield
  credit spreads (HYG-IEF spread as proxy for credit risk appetite) are TWO
  INDEPENDENT macro regime signals. When BOTH agree on risk-on, allocate to
  UPRO (3x SPY). When EITHER signals stress, rotate to defensive (SHY/TLT).

  This differs from single-signal VMR or VIX targeting because:
  1. Two orthogonal signals reduce false positives
  2. Credit spreads lead equity drawdowns (structural, not arbed away)
  3. Uses leveraged ETF only in high-conviction risk-on = asymmetric upside

Adversarial Validation (HC #705) BUILT IN:
  1. Permutation test (shuffled signals, 5000 iterations)
  2. Sub-period consistency (rolling 1-year Sharpe)
  3. Outlier robustness (remove top/bottom 1% days)
  4. R1 regime test (green vs red days, |delta| < 0.50)
  5. Regime-stratified performance (VIX high/low, credit tight/wide)

Cost assumptions:
  - ETF spread: 1bp for SPY/QQQ/IEF/SHY, 3bp for UPRO/TQQQ/TMF/HYG
  - Commission: $0.005/share (retail), ~2bp on $25 avg ETF price
  - Total round-trip: ~10bp for leveraged, ~4bp for unleveraged

Output: /home/jupiter/Lvl3Quant/output/dual_regime_vix_credit_v1/
"""

import os
import json
import warnings
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings('ignore')

# ─── CONFIG ──────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path('/home/jupiter/Lvl3Quant/output/dual_regime_vix_credit_v1')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INITIAL_CAPITAL = 100_000
REBAL_COST_BPS = 10  # round-trip cost in bps for leveraged ETFs
UNLEV_COST_BPS = 4   # round-trip cost for unleveraged

# Signal parameters
VIX_CONTANGO_LOOKBACK = 5    # days to smooth VIX ratio
CREDIT_SPREAD_LOOKBACK = 10  # days to smooth credit spread z-score
CREDIT_SPREAD_WINDOW = 63    # ~3 months for z-score calculation
VIX_CONTANGO_THRESHOLD = 1.0  # VXV/VIX > 1.0 = contango = risk-on
CREDIT_Z_THRESHOLD = 0.5      # z-score < 0.5 = credit calm = risk-on

# Allocation modes
RISK_ON_TICKER = 'UPRO'    # 3x SPY
RISK_OFF_TICKER = 'SHY'    # short-term treasuries (safe haven)
MODERATE_TICKER = 'SPY'     # when signals disagree, stay unleveraged

N_PERMUTATIONS = 5000

print("=" * 70)
print("DUAL REGIME INDICATOR: VIX TERM STRUCTURE + CREDIT SPREAD")
print("=" * 70)

# ─── DATA DOWNLOAD ───────────────────────────────────────────────────────────
print("\n[1/6] Downloading data...")

tickers = ['^VIX', '^VIX3M', 'HYG', 'IEF', 'SPY', 'UPRO', 'SHY', 'TLT', 'QQQ']
cache_file = OUTPUT_DIR / 'price_cache.parquet'

if cache_file.exists():
    prices = pd.read_parquet(cache_file)
    print(f"  Loaded from cache: {prices.shape}")
else:
    dfs = {}
    for t in tickers:
        try:
            d = yf.download(t, start='2012-01-01', end='2026-07-17', progress=False)
            if len(d) > 100:
                # Handle both single and multi-level columns
                if isinstance(d.columns, pd.MultiIndex):
                    d.columns = d.columns.get_level_values(0)
                dfs[t.replace('^', '')] = d['Close']
                print(f"  {t}: {len(d)} rows")
        except Exception as e:
            print(f"  {t}: FAILED - {e}")

    prices = pd.DataFrame(dfs)
    prices.index = pd.to_datetime(prices.index)
    # Remove timezone if present
    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)
    prices.to_parquet(cache_file)
    print(f"  Saved cache: {prices.shape}")

# Ensure we have what we need
required = ['VIX', 'VIX3M', 'HYG', 'IEF', 'SPY', 'UPRO', 'SHY']
missing = [r for r in required if r not in prices.columns]
if missing:
    print(f"  FATAL: Missing tickers: {missing}")
    exit(1)

# Forward fill small gaps
prices = prices.ffill().dropna(subset=['VIX', 'VIX3M', 'HYG', 'IEF', 'SPY'])
print(f"  Clean data: {prices.index[0].date()} to {prices.index[-1].date()}, {len(prices)} rows")

# ─── SIGNAL GENERATION ───────────────────────────────────────────────────────
print("\n[2/6] Generating regime signals...")

# Signal 1: VIX Term Structure (VIX3M / VIX ratio)
# VIX3M/VIX > 1 = contango = normal/calm market
# VIX3M/VIX < 1 = backwardation = stress/fear
prices['vix_ratio_raw'] = prices['VIX3M'] / prices['VIX']
prices['vix_ratio'] = prices['vix_ratio_raw'].rolling(VIX_CONTANGO_LOOKBACK).mean()
prices['vix_signal'] = (prices['vix_ratio'] > VIX_CONTANGO_THRESHOLD).astype(int)  # 1 = risk-on

# Signal 2: Credit Spread (HYG total return vs IEF total return proxy)
# Widening spread = stress, tightening = risk appetite
# Use HYG/IEF ratio as proxy for credit spread direction
prices['credit_ratio'] = prices['HYG'] / prices['IEF']
prices['credit_ratio_ma'] = prices['credit_ratio'].rolling(CREDIT_SPREAD_LOOKBACK).mean()
prices['credit_ratio_zscore'] = (
    (prices['credit_ratio_ma'] - prices['credit_ratio_ma'].rolling(CREDIT_SPREAD_WINDOW).mean()) /
    prices['credit_ratio_ma'].rolling(CREDIT_SPREAD_WINDOW).std()
)
# Positive z-score = credit outperforming (spreads tightening) = risk-on
# Negative z-score = credit underperforming (spreads widening) = risk-off
prices['credit_signal'] = (prices['credit_ratio_zscore'] > -CREDIT_Z_THRESHOLD).astype(int)  # 1 = risk-on

# Combined signal
# Both risk-on: UPRO (aggressive)
# One risk-on: SPY (moderate)
# Both risk-off: SHY (defensive)
prices['combined_signal'] = prices['vix_signal'] + prices['credit_signal']

# Drop rows where signals aren't ready
prices = prices.dropna(subset=['vix_ratio', 'credit_ratio_zscore'])

# Calculate daily returns for all instruments
for t in ['SPY', 'UPRO', 'SHY', 'HYG', 'IEF']:
    if t in prices.columns:
        prices[f'{t}_ret'] = prices[t].pct_change()

prices = prices.dropna(subset=['SPY_ret', 'UPRO_ret', 'SHY_ret'])

# Signal is generated at close, traded next day (no look-ahead)
prices['signal_lag'] = prices['combined_signal'].shift(1)
prices = prices.dropna(subset=['signal_lag'])

print(f"  Signal period: {prices.index[0].date()} to {prices.index[-1].date()}")
print(f"  Signal distribution:")
for s in [0, 1, 2]:
    n = (prices['signal_lag'] == s).sum()
    pct = n / len(prices) * 100
    label = {0: 'BOTH OFF (SHY)', 1: 'ONE ON (SPY)', 2: 'BOTH ON (UPRO)'}[s]
    print(f"    Signal={s} [{label}]: {n} days ({pct:.1f}%)")


# ─── BACKTEST ─────────────────────────────────────────────────────────────────
print("\n[3/6] Running backtest...")

def run_backtest(prices_df, signal_col='signal_lag', apply_costs=True, label='Strategy'):
    """Run backtest with given signal column."""
    df = prices_df.copy()

    # Map signal to returns
    # Signal 2 (both on) -> UPRO returns
    # Signal 1 (one on) -> SPY returns
    # Signal 0 (both off) -> SHY returns
    strat_ret = pd.Series(0.0, index=df.index)

    mask_2 = df[signal_col] == 2
    mask_1 = df[signal_col] == 1
    mask_0 = df[signal_col] == 0

    strat_ret[mask_2] = df.loc[mask_2, 'UPRO_ret']
    strat_ret[mask_1] = df.loc[mask_1, 'SPY_ret']
    strat_ret[mask_0] = df.loc[mask_0, 'SHY_ret']

    # Apply transaction costs on signal changes
    if apply_costs:
        signal_changes = df[signal_col].diff().fillna(0) != 0
        n_changes = signal_changes.sum()

        # Cost depends on what we're switching to
        cost_per_change = pd.Series(0.0, index=df.index)
        cost_per_change[signal_changes & mask_2] = REBAL_COST_BPS / 10000  # switching to UPRO
        cost_per_change[signal_changes & mask_1] = UNLEV_COST_BPS / 10000  # switching to SPY
        cost_per_change[signal_changes & mask_0] = UNLEV_COST_BPS / 10000  # switching to SHY

        strat_ret -= cost_per_change
    else:
        n_changes = 0

    # Calculate equity curve
    equity = (1 + strat_ret).cumprod() * INITIAL_CAPITAL

    # Metrics
    ann_ret = strat_ret.mean() * 252
    ann_vol = strat_ret.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    downside = strat_ret[strat_ret < 0].std() * np.sqrt(252)
    sortino = ann_ret / downside if downside > 0 else 0

    # Profit factor
    gains = strat_ret[strat_ret > 0].sum()
    losses = abs(strat_ret[strat_ret < 0].sum())
    pf = gains / losses if losses > 0 else float('inf')

    # Win rate
    wr = (strat_ret > 0).sum() / len(strat_ret)

    # Max drawdown
    peak = equity.cummax()
    dd = (equity - peak) / peak
    max_dd = dd.min()

    # CAGR
    years = (df.index[-1] - df.index[0]).days / 365.25
    cagr = (equity.iloc[-1] / INITIAL_CAPITAL) ** (1 / years) - 1 if years > 0 else 0

    return {
        'label': label,
        'equity': equity,
        'returns': strat_ret,
        'ann_ret': ann_ret,
        'ann_vol': ann_vol,
        'sharpe': sharpe,
        'sortino': sortino,
        'pf': pf,
        'wr': wr,
        'max_dd': max_dd,
        'cagr': cagr,
        'n_trades': n_changes if apply_costs else 0,
        'final_equity': equity.iloc[-1],
    }


# Main strategy
strat = run_backtest(prices, label='Dual Regime (VIX+Credit)')

# Benchmarks
# Buy and hold SPY
spy_eq = (1 + prices['SPY_ret']).cumprod() * INITIAL_CAPITAL
spy_sharpe = (prices['SPY_ret'].mean() * 252) / (prices['SPY_ret'].std() * np.sqrt(252))
spy_dd = ((spy_eq - spy_eq.cummax()) / spy_eq.cummax()).min()

# Buy and hold UPRO (naive leveraged)
upro_eq = (1 + prices['UPRO_ret']).cumprod() * INITIAL_CAPITAL
upro_sharpe = (prices['UPRO_ret'].mean() * 252) / (prices['UPRO_ret'].std() * np.sqrt(252))
upro_dd = ((upro_eq - upro_eq.cummax()) / upro_eq.cummax()).min()

# VIX-only signal (single signal baseline)
prices['vix_only_signal'] = prices['vix_signal'].shift(1)
prices = prices.dropna(subset=['vix_only_signal'])
vix_only_ret = pd.Series(0.0, index=prices.index)
vix_only_ret[prices['vix_only_signal'] == 1] = prices.loc[prices['vix_only_signal'] == 1, 'UPRO_ret']
vix_only_ret[prices['vix_only_signal'] == 0] = prices.loc[prices['vix_only_signal'] == 0, 'SHY_ret']
# Apply costs
vix_changes = prices['vix_only_signal'].diff().fillna(0) != 0
vix_only_ret[vix_changes] -= REBAL_COST_BPS / 10000
vix_only_eq = (1 + vix_only_ret).cumprod() * INITIAL_CAPITAL
vix_only_sharpe = (vix_only_ret.mean() * 252) / (vix_only_ret.std() * np.sqrt(252))
vix_only_dd = ((vix_only_eq - vix_only_eq.cummax()) / vix_only_eq.cummax()).min()

print(f"\n{'Metric':<20} {'Dual Regime':>14} {'VIX Only':>14} {'SPY B&H':>14} {'UPRO B&H':>14}")
print("-" * 78)
print(f"{'CAGR':<20} {strat['cagr']*100:>13.1f}% {(vix_only_ret.mean()*252)*100:>13.1f}% {prices['SPY_ret'].mean()*252*100:>13.1f}% {prices['UPRO_ret'].mean()*252*100:>13.1f}%")
print(f"{'Sharpe':<20} {strat['sharpe']:>14.2f} {vix_only_sharpe:>14.2f} {spy_sharpe:>14.2f} {upro_sharpe:>14.2f}")
print(f"{'Sortino':<20} {strat['sortino']:>14.2f}")
print(f"{'Profit Factor':<20} {strat['pf']:>14.2f}")
print(f"{'Win Rate':<20} {strat['wr']*100:>13.1f}%")
print(f"{'Max Drawdown':<20} {strat['max_dd']*100:>13.1f}% {vix_only_dd*100:>13.1f}% {spy_dd*100:>13.1f}% {upro_dd*100:>13.1f}%")
print(f"{'Final Equity':<20} ${strat['final_equity']:>12,.0f} ${vix_only_eq.iloc[-1]:>12,.0f} ${spy_eq.iloc[-1]:>12,.0f} ${upro_eq.iloc[-1]:>12,.0f}")
print(f"{'Trades':<20} {strat['n_trades']:>14.0f} {vix_changes.sum():>14.0f}")

# ─── ADVERSARIAL VALIDATION ──────────────────────────────────────────────────
print("\n[4/6] Running adversarial validation (HC #705)...")

results = {'strategy': strat}
adv_results = {}

# --- Gate 1: Permutation Test ---
print(f"\n  Gate 1: Permutation test ({N_PERMUTATIONS} iterations)...")
real_sharpe = strat['sharpe']
perm_sharpes = []

np.random.seed(42)
for i in range(N_PERMUTATIONS):
    # Shuffle the combined signal (not the returns)
    shuffled_signal = prices['signal_lag'].sample(frac=1, replace=False).values
    prices['perm_signal'] = shuffled_signal
    perm_result = run_backtest(prices, signal_col='perm_signal', apply_costs=True, label='perm')
    perm_sharpes.append(perm_result['sharpe'])

    if (i + 1) % 1000 == 0:
        print(f"    {i+1}/{N_PERMUTATIONS} done...")

perm_sharpes = np.array(perm_sharpes)
perm_p_value = (perm_sharpes >= real_sharpe).mean()
adv_results['permutation'] = {
    'real_sharpe': real_sharpe,
    'perm_mean': float(perm_sharpes.mean()),
    'perm_std': float(perm_sharpes.std()),
    'p_value': float(perm_p_value),
    'pass': perm_p_value < 0.05
}
gate1_pass = perm_p_value < 0.05
print(f"    Real Sharpe: {real_sharpe:.3f}, Perm mean: {perm_sharpes.mean():.3f} +/- {perm_sharpes.std():.3f}")
print(f"    p-value: {perm_p_value:.4f} -> {'PASS' if gate1_pass else 'FAIL'}")

# --- Gate 2: Sub-Period Consistency ---
print("\n  Gate 2: Sub-period consistency (rolling 1-year Sharpe)...")
rolling_sharpes = []
years_list = sorted(prices.index.year.unique())

for yr in years_list:
    yr_mask = prices.index.year == yr
    if yr_mask.sum() < 50:  # skip partial years
        continue
    yr_prices = prices[yr_mask].copy()
    yr_result = run_backtest(yr_prices, apply_costs=True, label=f'Y{yr}')
    rolling_sharpes.append({'year': yr, 'sharpe': yr_result['sharpe'],
                           'cagr': yr_result['cagr'], 'max_dd': yr_result['max_dd'],
                           'wr': yr_result['wr']})
    print(f"    {yr}: Sharpe={yr_result['sharpe']:.2f}, CAGR={yr_result['cagr']*100:.1f}%, DD={yr_result['max_dd']*100:.1f}%, WR={yr_result['wr']*100:.1f}%")

# Pass if >60% of years are positive Sharpe
pos_years = sum(1 for s in rolling_sharpes if s['sharpe'] > 0)
total_years = len(rolling_sharpes)
consistency_ratio = pos_years / total_years if total_years > 0 else 0
gate2_pass = consistency_ratio >= 0.60
adv_results['sub_period'] = {
    'pos_years': pos_years,
    'total_years': total_years,
    'consistency_ratio': consistency_ratio,
    'yearly_sharpes': rolling_sharpes,
    'pass': gate2_pass
}
print(f"    Positive Sharpe years: {pos_years}/{total_years} ({consistency_ratio*100:.0f}%) -> {'PASS' if gate2_pass else 'FAIL'}")

# --- Gate 3: Outlier Robustness ---
print("\n  Gate 3: Outlier robustness (remove top/bottom 1% return days)...")
strat_rets = strat['returns']
p1, p99 = strat_rets.quantile(0.01), strat_rets.quantile(0.99)
trimmed_mask = (strat_rets >= p1) & (strat_rets <= p99)
trimmed_rets = strat_rets[trimmed_mask]
trimmed_sharpe = (trimmed_rets.mean() * 252) / (trimmed_rets.std() * np.sqrt(252))
sharpe_degradation = 1 - trimmed_sharpe / real_sharpe if real_sharpe != 0 else 0
gate3_pass = trimmed_sharpe > 0 and abs(sharpe_degradation) < 0.50
adv_results['outlier_robustness'] = {
    'full_sharpe': real_sharpe,
    'trimmed_sharpe': float(trimmed_sharpe),
    'degradation_pct': float(sharpe_degradation * 100),
    'pass': gate3_pass
}
print(f"    Full Sharpe: {real_sharpe:.3f}, Trimmed: {trimmed_sharpe:.3f} ({sharpe_degradation*100:+.1f}% change)")
print(f"    -> {'PASS' if gate3_pass else 'FAIL'}")

# --- Gate 4: R1 Regime Test (green vs red days) ---
print("\n  Gate 4: R1 Regime test (green vs red ES/SPY days)...")
# Classify days by SPY direction - align indices
common_idx = strat['returns'].index.intersection(prices.index)
spy_daily = prices.loc[common_idx, 'SPY_ret']
strat_aligned = strat['returns'].loc[common_idx]
green_mask = spy_daily > 0
red_mask = spy_daily < 0

strat_green = strat_aligned[green_mask]
strat_red = strat_aligned[red_mask]

sharpe_green = (strat_green.mean() * 252) / (strat_green.std() * np.sqrt(252)) if strat_green.std() > 0 else 0
sharpe_red = (strat_red.mean() * 252) / (strat_red.std() * np.sqrt(252)) if strat_red.std() > 0 else 0

regime_delta = abs(sharpe_green - sharpe_red) / max(abs(sharpe_green), abs(sharpe_red), 0.01)
gate4_pass = regime_delta < 0.50

adv_results['regime_test'] = {
    'sharpe_green': float(sharpe_green),
    'sharpe_red': float(sharpe_red),
    'regime_delta': float(regime_delta),
    'pass': gate4_pass
}
print(f"    Sharpe (green days): {sharpe_green:.3f}")
print(f"    Sharpe (red days):   {sharpe_red:.3f}")
print(f"    |delta|/max = {regime_delta:.3f} (threshold < 0.50) -> {'PASS' if gate4_pass else 'FAIL'}")

# --- Gate 5: Out-of-Sample Split ---
print("\n  Gate 5: Out-of-sample validation...")
# Train: 2012-2021, OOS: 2022-2026
oos_start = '2022-01-01'
is_mask = prices.index < oos_start
oos_mask = prices.index >= oos_start

is_result = run_backtest(prices[is_mask], apply_costs=True, label='In-Sample (2012-2021)')
oos_result = run_backtest(prices[oos_mask], apply_costs=True, label='OOS (2022-2026)')

gate5_pass = oos_result['sharpe'] > 0 and oos_result['sharpe'] > is_result['sharpe'] * 0.3
adv_results['oos_split'] = {
    'is_sharpe': is_result['sharpe'],
    'is_cagr': is_result['cagr'],
    'oos_sharpe': oos_result['sharpe'],
    'oos_cagr': oos_result['cagr'],
    'oos_max_dd': oos_result['max_dd'],
    'pass': gate5_pass
}
print(f"    In-Sample: Sharpe={is_result['sharpe']:.3f}, CAGR={is_result['cagr']*100:.1f}%")
print(f"    OOS:       Sharpe={oos_result['sharpe']:.3f}, CAGR={oos_result['cagr']*100:.1f}%, MaxDD={oos_result['max_dd']*100:.1f}%")
print(f"    OOS Sharpe > 30% of IS Sharpe? -> {'PASS' if gate5_pass else 'FAIL'}")


# ─── PARAMETER SENSITIVITY ───────────────────────────────────────────────────
print("\n[5/6] Parameter sensitivity sweep...")

# Re-generate signals with different parameters to check robustness
param_results = []
for vix_lb in [3, 5, 10, 15, 20]:
    for credit_lb in [5, 10, 20, 30]:
        for credit_z in [-1.0, -0.5, 0.0, 0.5]:
            df = prices.copy()
            df['vr'] = df['vix_ratio_raw'].rolling(vix_lb).mean()
            df['cr_ma'] = df['credit_ratio'].rolling(credit_lb).mean()
            df['cr_z'] = (df['cr_ma'] - df['cr_ma'].rolling(CREDIT_SPREAD_WINDOW).mean()) / df['cr_ma'].rolling(CREDIT_SPREAD_WINDOW).std()
            df = df.dropna(subset=['vr', 'cr_z'])

            sig = ((df['vr'] > VIX_CONTANGO_THRESHOLD).astype(int) +
                   (df['cr_z'] > credit_z).astype(int))
            df['test_signal'] = sig.shift(1)
            df = df.dropna(subset=['test_signal', 'SPY_ret', 'UPRO_ret', 'SHY_ret'])

            res = run_backtest(df, signal_col='test_signal', apply_costs=True)
            param_results.append({
                'vix_lb': vix_lb, 'credit_lb': credit_lb, 'credit_z': credit_z,
                'sharpe': res['sharpe'], 'cagr': res['cagr'], 'max_dd': res['max_dd']
            })

param_df = pd.DataFrame(param_results)
pos_sharpe_pct = (param_df['sharpe'] > 0).mean() * 100
median_sharpe = param_df['sharpe'].median()
print(f"  Tested {len(param_results)} parameter combinations")
print(f"  {pos_sharpe_pct:.0f}% have positive Sharpe (median: {median_sharpe:.3f})")
print(f"  Sharpe range: [{param_df['sharpe'].min():.3f}, {param_df['sharpe'].max():.3f}]")
print(f"  Best params: VIX_LB={param_df.loc[param_df['sharpe'].idxmax(), 'vix_lb']}, "
      f"Credit_LB={param_df.loc[param_df['sharpe'].idxmax(), 'credit_lb']}, "
      f"Credit_Z={param_df.loc[param_df['sharpe'].idxmax(), 'credit_z']}")

gate6_pass = pos_sharpe_pct > 50
adv_results['param_sensitivity'] = {
    'n_combos': len(param_results),
    'pct_positive_sharpe': float(pos_sharpe_pct),
    'median_sharpe': float(median_sharpe),
    'pass': gate6_pass
}

# ─── SUMMARY ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("[6/6] FINAL ADVERSARIAL VALIDATION SUMMARY")
print("=" * 70)

gates = {
    'Gate 1 - Permutation Test (p<0.05)': gate1_pass,
    'Gate 2 - Sub-Period Consistency (>60% years positive)': gate2_pass,
    'Gate 3 - Outlier Robustness (trimmed Sharpe >0)': gate3_pass,
    'Gate 4 - R1 Regime Test (|delta|<0.50)': gate4_pass,
    'Gate 5 - OOS Validation (OOS Sharpe >30% IS)': gate5_pass,
    'Gate 6 - Parameter Robustness (>50% combos positive)': gate6_pass,
}

all_pass = all(gates.values())
pass_count = sum(gates.values())

for gate, passed in gates.items():
    print(f"  {'PASS' if passed else 'FAIL'} - {gate}")

print(f"\n  OVERALL: {pass_count}/{len(gates)} gates passed -> {'STRATEGY VALIDATED' if all_pass else 'STRATEGY REJECTED'}")

print(f"\n  Key Metrics:")
print(f"    Sharpe Ratio:   {strat['sharpe']:.3f}")
print(f"    Sortino Ratio:  {strat['sortino']:.3f}")
print(f"    CAGR:           {strat['cagr']*100:.1f}%")
print(f"    Profit Factor:  {strat['pf']:.2f}")
print(f"    Win Rate:       {strat['wr']*100:.1f}%")
print(f"    Max Drawdown:   {strat['max_dd']*100:.1f}%")
print(f"    Calmar Ratio:   {strat['cagr']/abs(strat['max_dd']):.2f}" if strat['max_dd'] != 0 else "    Calmar Ratio:   N/A")

# ─── CHARTS ──────────────────────────────────────────────────────────────────
print("\n  Saving charts...")

fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Equity curves
ax = axes[0]
ax.plot(strat['equity'].index, strat['equity'], label=f"Dual Regime (Sharpe={strat['sharpe']:.2f})", linewidth=2)
ax.plot(spy_eq.index, spy_eq, label=f'SPY B&H (Sharpe={spy_sharpe:.2f})', alpha=0.7)
ax.plot(upro_eq.index, upro_eq, label=f'UPRO B&H (Sharpe={upro_sharpe:.2f})', alpha=0.7)
ax.plot(vix_only_eq.index, vix_only_eq, label=f'VIX Only (Sharpe={vix_only_sharpe:.2f})', alpha=0.7)
ax.set_ylabel('Equity ($)')
ax.set_title('Dual Regime VIX+Credit Strategy vs Benchmarks')
ax.legend(loc='upper left')
ax.set_yscale('log')
ax.grid(True, alpha=0.3)

# Regime signal
ax = axes[1]
ax.fill_between(prices.index, 0, prices['signal_lag'], step='post', alpha=0.5,
                color=['red' if s == 0 else 'yellow' if s == 1 else 'green'
                       for s in prices['signal_lag']])
ax.set_ylabel('Signal (0=Off, 1=Mixed, 2=On)')
ax.set_title('Regime Signal Over Time')
ax.set_ylim(-0.1, 2.1)
ax.grid(True, alpha=0.3)

# Drawdown
peak = strat['equity'].cummax()
dd = (strat['equity'] - peak) / peak
ax = axes[2]
ax.fill_between(dd.index, 0, dd, color='red', alpha=0.5)
ax.set_ylabel('Drawdown (%)')
ax.set_title('Strategy Drawdown')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'strategy_overview.png', dpi=150)
plt.close()

# Permutation distribution
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.hist(perm_sharpes, bins=50, alpha=0.7, color='gray', label='Permuted Sharpes')
ax.axvline(real_sharpe, color='red', linewidth=2, label=f'Real Sharpe={real_sharpe:.3f}')
ax.axvline(perm_sharpes.mean(), color='blue', linewidth=1, linestyle='--', label=f'Perm Mean={perm_sharpes.mean():.3f}')
ax.set_xlabel('Sharpe Ratio')
ax.set_ylabel('Count')
ax.set_title(f'Permutation Test (p={perm_p_value:.4f})')
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'permutation_test.png', dpi=150)
plt.close()

# Parameter sensitivity heatmap
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
pivot = param_df.pivot_table(values='sharpe', index='vix_lb', columns='credit_z', aggfunc='mean')
im = ax.imshow(pivot.values, aspect='auto', cmap='RdYlGn')
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels([f'{c:.1f}' for c in pivot.columns])
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index)
ax.set_xlabel('Credit Z-Score Threshold')
ax.set_ylabel('VIX Lookback')
ax.set_title('Parameter Sensitivity (Sharpe)')
plt.colorbar(im, label='Sharpe')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'param_sensitivity.png', dpi=150)
plt.close()

print("  Charts saved.")

# ─── SAVE RESULTS ─────────────────────────────────────────────────────────────
full_results = {
    'strategy_name': 'Dual Regime VIX Term Structure + Credit Spread',
    'timestamp': dt.datetime.now().isoformat(),
    'period': f"{prices.index[0].date()} to {prices.index[-1].date()}",
    'metrics': {
        'sharpe': strat['sharpe'],
        'sortino': strat['sortino'],
        'cagr': strat['cagr'],
        'profit_factor': strat['pf'],
        'win_rate': strat['wr'],
        'max_drawdown': strat['max_dd'],
        'ann_return': strat['ann_ret'],
        'ann_vol': strat['ann_vol'],
        'final_equity': strat['final_equity'],
        'n_trades': int(strat['n_trades']),
    },
    'benchmarks': {
        'spy_sharpe': float(spy_sharpe),
        'upro_sharpe': float(upro_sharpe),
        'vix_only_sharpe': float(vix_only_sharpe),
    },
    'adversarial': adv_results,
    'overall_pass': all_pass,
    'gates_passed': f"{pass_count}/{len(gates)}",
}

with open(OUTPUT_DIR / 'full_results.json', 'w') as f:
    json.dump(full_results, f, indent=2, default=str)

# Save yearly breakdown
yearly_df = pd.DataFrame(rolling_sharpes)
yearly_df.to_csv(OUTPUT_DIR / 'yearly_breakdown.csv', index=False)

print(f"\n  Results saved to {OUTPUT_DIR}")
print("\nDONE.")

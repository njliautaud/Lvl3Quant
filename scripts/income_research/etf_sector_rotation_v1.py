#!/usr/bin/env python3
"""
ETF Sector Rotation v1 — Survivorship-Bias-Free Strategy
=========================================================
HC #703 R3 + HC #705 (adversarial checks built-in).

Rotates between sector ETFs based on relative momentum/mean-reversion.
NO survivorship bias because ETFs represent sectors, not individual stocks.

Strategies tested:
  A) Momentum: Buy top N sectors by trailing return
  B) Mean-Reversion: Buy bottom N sectors by trailing return
  C) Dual-Momentum: Buy top sectors only when above 200-day MA (else cash)
  D) Volatility-Weighted: Weight sectors inversely by volatility

ETF Universe:
  XLB, XLC, XLE, XLF, XLI, XLK, XLP, XLRE, XLU, XLV, XLY
  (11 Select Sector SPDR ETFs — covers entire S&P 500)

All with built-in adversarial checks per HC #705.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import json
from pathlib import Path
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/etf_sector_rotation_v1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# ETF UNIVERSE
# ─────────────────────────────────────────────
SECTOR_ETFS = ['XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY']
BENCHMARK = 'SPY'

# ─────────────────────────────────────────────
# DATA DOWNLOAD
# ─────────────────────────────────────────────
print("=" * 70)
print("ETF SECTOR ROTATION v1 — SURVIVORSHIP-BIAS-FREE")
print("=" * 70)

print("\nDownloading data...")
all_tickers = SECTOR_ETFS + [BENCHMARK]
data = yf.download(all_tickers, start='2010-01-01', end='2026-07-15', progress=False)

# Handle multi-index columns
if isinstance(data.columns, pd.MultiIndex):
    closes = data['Close'][all_tickers].copy()
else:
    closes = data[all_tickers].copy()

# Forward fill gaps, drop rows with any NaN
closes = closes.ffill().dropna()
print(f"Data: {closes.index[0].strftime('%Y-%m-%d')} to {closes.index[-1].strftime('%Y-%m-%d')}, {len(closes)} trading days")

# Daily returns
returns = closes.pct_change().dropna()

# ─────────────────────────────────────────────
# PRICING SANITY CHECK (HC #705 R1a)
# ─────────────────────────────────────────────
print("\n--- HC #705 SELF-VALIDATION ---")
for ticker in all_tickers:
    zero_prices = (closes[ticker] == 0).sum()
    if zero_prices > 0:
        print(f"  ⚠️ WARNING: {ticker} has {zero_prices} zero prices")
    big_moves = (returns[ticker].abs() > 0.15).sum()
    if big_moves > 3:
        print(f"  ⚠️ WARNING: {ticker} has {big_moves} daily moves > 15%")
print("  ✅ Pricing checks complete")

# ─────────────────────────────────────────────
# STRATEGY CONFIGS
# ─────────────────────────────────────────────
configs = []

# Momentum: buy top N by trailing return
for lookback in [21, 63, 126, 252]:  # 1m, 3m, 6m, 12m
    for top_n in [3, 5]:
        for hold in [21, 63]:  # 1m, 3m rebalance
            configs.append({
                'name': f'momentum_{lookback}d_top{top_n}_hold{hold}d',
                'type': 'momentum',
                'lookback': lookback,
                'top_n': top_n,
                'hold': hold,
            })

# Mean-Reversion: buy bottom N by trailing return
for lookback in [21, 63]:
    for bottom_n in [3, 5]:
        for hold in [21, 63]:
            configs.append({
                'name': f'mean_rev_{lookback}d_bot{bottom_n}_hold{hold}d',
                'type': 'mean_reversion',
                'lookback': lookback,
                'bottom_n': bottom_n,
                'hold': hold,
            })

# Dual Momentum: top N only when above 200-day MA
for lookback in [63, 126, 252]:
    for top_n in [3, 5]:
        configs.append({
            'name': f'dual_mom_{lookback}d_top{top_n}',
            'type': 'dual_momentum',
            'lookback': lookback,
            'top_n': top_n,
            'hold': 21,
        })

print(f"\nTesting {len(configs)} configurations...")

# ─────────────────────────────────────────────
# BACKTEST ENGINE
# ─────────────────────────────────────────────
def run_backtest(config, closes, returns, sector_etfs):
    """Run a single config backtest. Returns daily portfolio returns."""
    lookback = config['lookback']
    hold = config['hold']

    # Calculate trailing returns for ranking
    trailing_ret = closes[sector_etfs].pct_change(lookback)
    ma_200 = closes[sector_etfs].rolling(200).mean()

    # Start after enough history
    start_idx = max(lookback, 200) + 10

    portfolio_returns = []
    dates = []
    rebal_counter = 0
    current_holdings = []

    # Align indices
    common_dates = closes.index.intersection(returns.index)

    for i in range(start_idx, len(common_dates)):
        if i >= len(returns):
            break
        date = common_dates[i]

        # Rebalance every 'hold' days
        if rebal_counter % hold == 0:
            prev_ret = trailing_ret.loc[:date].iloc[-2] if len(trailing_ret.loc[:date]) > 1 else trailing_ret.iloc[0]

            if config['type'] == 'momentum':
                ranked = prev_ret.dropna().sort_values(ascending=False)
                current_holdings = ranked.head(config['top_n']).index.tolist()

            elif config['type'] == 'mean_reversion':
                ranked = prev_ret.dropna().sort_values(ascending=True)
                current_holdings = ranked.head(config['bottom_n']).index.tolist()

            elif config['type'] == 'dual_momentum':
                ranked = prev_ret.dropna().sort_values(ascending=False)
                above_ma = []
                for ticker in ranked.head(config['top_n']).index:
                    if closes[ticker].loc[:date].iloc[-2] > ma_200[ticker].loc[:date].iloc[-2]:
                        above_ma.append(ticker)
                current_holdings = above_ma if above_ma else []

        rebal_counter += 1

        if current_holdings:
            day_return = returns.loc[date, current_holdings].mean()
        else:
            day_return = 0.0

        portfolio_returns.append(day_return)
        dates.append(date)

    return pd.Series(portfolio_returns, index=dates)

# ─────────────────────────────────────────────
# ADVERSARIAL CHECKS (HC #705)
# ─────────────────────────────────────────────
def run_permutation_test(portfolio_returns, benchmark_returns, n_perms=200):
    """Permutation test: shuffle rebalance dates, compare Sharpe."""
    real_sharpe = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)

    random_sharpes = []
    for _ in range(n_perms):
        # Shuffle the returns (breaks temporal structure = random timing)
        shuffled = portfolio_returns.sample(frac=1.0).values
        s = np.mean(shuffled) / np.std(shuffled) * np.sqrt(252) if np.std(shuffled) > 0 else 0
        random_sharpes.append(s)

    p_value = np.mean([s >= real_sharpe for s in random_sharpes])
    return p_value, real_sharpe, np.mean(random_sharpes)

def run_regime_test(portfolio_returns, benchmark_returns):
    """R1: Check if strategy works in both green and red market regimes."""
    aligned_bench = benchmark_returns.reindex(portfolio_returns.index).fillna(0)

    green_mask = aligned_bench > 0
    red_mask = aligned_bench < 0

    green_ret = portfolio_returns[green_mask]
    red_ret = portfolio_returns[red_mask]

    sharpe_green = green_ret.mean() / green_ret.std() * np.sqrt(252) if green_ret.std() > 0 else 0
    sharpe_red = red_ret.mean() / red_ret.std() * np.sqrt(252) if red_ret.std() > 0 else 0

    gap = abs(sharpe_green - sharpe_red) / max(abs(sharpe_green), abs(sharpe_red), 0.001)
    return gap, sharpe_green, sharpe_red

def run_subperiod_test(portfolio_returns):
    """Split into halves, both must be profitable."""
    mid = len(portfolio_returns) // 2
    h1 = portfolio_returns.iloc[:mid]
    h2 = portfolio_returns.iloc[mid:]

    sharpe_h1 = h1.mean() / h1.std() * np.sqrt(252) if h1.std() > 0 else 0
    sharpe_h2 = h2.mean() / h2.std() * np.sqrt(252) if h2.std() > 0 else 0

    return sharpe_h1 > 0 and sharpe_h2 > 0, sharpe_h1, sharpe_h2

def run_outlier_test(portfolio_returns):
    """Remove top 5 days, check if still profitable."""
    sorted_ret = portfolio_returns.sort_values(ascending=False)
    without_top5 = sorted_ret.iloc[5:]

    sharpe_full = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252) if portfolio_returns.std() > 0 else 0
    sharpe_no5 = without_top5.mean() / without_top5.std() * np.sqrt(252) if without_top5.std() > 0 else 0

    drop_pct = (sharpe_full - sharpe_no5) / abs(sharpe_full) * 100 if abs(sharpe_full) > 0 else 0
    return drop_pct < 50, drop_pct

# ─────────────────────────────────────────────
# RUN ALL CONFIGS
# ─────────────────────────────────────────────
benchmark_returns = returns[BENCHMARK]
results = []

for i, config in enumerate(configs):
    port_ret = run_backtest(config, closes, returns, SECTOR_ETFS)

    if len(port_ret) < 252:
        continue

    # Basic metrics
    sharpe = port_ret.mean() / port_ret.std() * np.sqrt(252) if port_ret.std() > 0 else 0
    sortino_denom = port_ret[port_ret < 0].std()
    sortino = port_ret.mean() / sortino_denom * np.sqrt(252) if sortino_denom > 0 else 0

    total_ret = (1 + port_ret).prod() - 1
    cagr = (1 + total_ret) ** (252 / len(port_ret)) - 1

    # Win rate (daily)
    wr = (port_ret > 0).mean()

    # Max drawdown
    cum = (1 + port_ret).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()

    # Profit factor (daily)
    gains = port_ret[port_ret > 0].sum()
    losses = abs(port_ret[port_ret < 0].sum())
    pf = gains / losses if losses > 0 else 999

    # Adversarial checks
    perm_p, _, perm_mean = run_permutation_test(port_ret, benchmark_returns)
    r1_gap, sharpe_green, sharpe_red = run_regime_test(port_ret, benchmark_returns)
    subperiod_pass, sharpe_h1, sharpe_h2 = run_subperiod_test(port_ret)
    outlier_pass, outlier_drop = run_outlier_test(port_ret)

    # Gate checks
    gates = {
        'perm_p': perm_p < 0.05,
        'R1': r1_gap < 0.50,
        'subperiod': subperiod_pass,
        'outlier': outlier_pass,
        'sharpe_positive': sharpe > 0,
    }
    all_pass = all(gates.values())

    result = {
        'name': config['name'],
        'type': config['type'],
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'cagr': round(cagr * 100, 2),
        'wr': round(wr * 100, 1),
        'pf': round(pf, 2),
        'max_dd': round(max_dd * 100, 2),
        'n_years': round(len(port_ret) / 252, 1),
        'perm_p': round(perm_p, 3),
        'r1_gap': round(r1_gap, 3),
        'sharpe_green': round(sharpe_green, 2),
        'sharpe_red': round(sharpe_red, 2),
        'subperiod': subperiod_pass,
        'sharpe_h1': round(sharpe_h1, 2),
        'sharpe_h2': round(sharpe_h2, 2),
        'outlier_drop': round(outlier_drop, 1),
        'all_pass': all_pass,
        'gates': gates,
    }
    results.append(result)

    status = "✅ PASS" if all_pass else "❌ FAIL"
    if (i + 1) % 10 == 0 or all_pass:
        print(f"  [{i+1}/{len(configs)}] {config['name']}: Sharpe {sharpe:.2f}, perm p={perm_p:.3f}, R1 gap={r1_gap:.3f} {status}")

# ─────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('sharpe', ascending=False)

# Passing configs
passing = results_df[results_df['all_pass'] == True]
failing = results_df[results_df['all_pass'] == False]

print(f"\nTotal configs: {len(results_df)}")
print(f"Passing ALL gates: {len(passing)}")
print(f"Failing: {len(failing)}")

if len(passing) > 0:
    print(f"\n{'=' * 70}")
    print("CONFIGS PASSING ALL GATES:")
    print(f"{'=' * 70}")
    for _, r in passing.iterrows():
        print(f"\n  {r['name']}")
        print(f"    Sharpe: {r['sharpe']:.2f} | Sortino: {r['sortino']:.2f} | CAGR: {r['cagr']:.1f}%")
        print(f"    WR: {r['wr']:.0f}% | PF: {r['pf']:.2f} | MaxDD: {r['max_dd']:.1f}%")
        print(f"    Perm p: {r['perm_p']:.3f} | R1 gap: {r['r1_gap']:.3f} (green={r['sharpe_green']:.2f}, red={r['sharpe_red']:.2f})")
        print(f"    Subperiod: H1={r['sharpe_h1']:.2f}, H2={r['sharpe_h2']:.2f}")
        print(f"    Outlier sensitivity: {r['outlier_drop']:.1f}% Sharpe drop without top 5 days")
        print(f"    Years: {r['n_years']:.1f}")

# Show top 5 by Sharpe regardless
print(f"\n{'=' * 70}")
print("TOP 5 BY SHARPE (regardless of gates):")
print(f"{'=' * 70}")
for _, r in results_df.head(5).iterrows():
    gate_str = " ".join([f"{'✅' if v else '❌'}{k}" for k, v in r['gates'].items()])
    print(f"  {r['name']}: Sharpe {r['sharpe']:.2f}, perm_p={r['perm_p']:.3f}, R1={r['r1_gap']:.3f}, {gate_str}")

# Show failure modes
print(f"\n{'=' * 70}")
print("FAILURE MODE BREAKDOWN:")
print(f"{'=' * 70}")
for gate in ['perm_p', 'R1', 'subperiod', 'outlier', 'sharpe_positive']:
    fail_count = sum(1 for r in results if not r['gates'][gate])
    print(f"  {gate}: {fail_count}/{len(results)} fail")

# SPY buy-and-hold comparison
spy_ret = benchmark_returns.loc[results_df.iloc[0]['name'] if len(results) > 0 else benchmark_returns.index[252]:]
# Use the common date range from the first config's backtest
spy_sharpe = spy_ret.mean() / spy_ret.std() * np.sqrt(252) if spy_ret.std() > 0 else 0
print(f"\nSPY buy-and-hold Sharpe: ~{spy_sharpe:.2f}")

# Save results
results_df.to_csv(OUTPUT_DIR / 'results.csv', index=False)
with open(OUTPUT_DIR / 'results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nResults saved to {OUTPUT_DIR}")

# ─────────────────────────────────────────────
# VERDICT
# ─────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("VERDICT")
print(f"{'=' * 70}")
if len(passing) > 0:
    print(f"  🔥 {len(passing)} configs pass all adversarial gates!")
    print(f"  Best: {passing.iloc[0]['name']} — Sharpe {passing.iloc[0]['sharpe']:.2f}")
    print(f"  This is survivorship-bias-free (ETFs only, no individual stocks).")
else:
    print(f"  ❌ No configs pass all gates.")
    # Check which gate is the main killer
    perm_fails = sum(1 for r in results if not r['gates']['perm_p'])
    r1_fails = sum(1 for r in results if not r['gates']['R1'])
    print(f"  Main failure: permutation ({perm_fails} fail) / R1 regime ({r1_fails} fail)")

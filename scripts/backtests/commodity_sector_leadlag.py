#!/usr/bin/env python3
"""
Cross-Asset Lead-Lag: Commodities → Sector ETFs
================================================
Hypothesis: Commodity price moves lead sector ETF moves by 1-3 days due to
physical supply chain information transmission delays and structural trading
hour differences (commodity futures ~24h, ETFs RTH only).

Pairs tested:
  - Natural Gas (UNG) → Utilities (XLU)
  - Oil (USO) → Energy (XLE)
  - Lumber proxy (WOOD) → Homebuilders (XHB)
  - Gold (GLD) → Gold Miners (GDX)
  - Agriculture (DBA) → Consumer Staples (XLP)

Strategy: When a commodity makes a significant move (>1 stdev over trailing 20d),
go long the corresponding sector ETF call (or put for negative moves) with 3-5 day
hold. We proxy option returns as leveraged ETF returns (delta ~0.50 ATM, gamma
adds ~20% convexity on winning trades).

Author: Claude Opus 4.6
Date: 2026-08-18
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================
# 1. DATA ACQUISITION
# ============================================================

print("=" * 70)
print("CROSS-ASSET LEAD-LAG: COMMODITIES -> SECTOR ETFs")
print("=" * 70)
print()

# Commodity-sector pairs: (commodity_ticker, sector_ticker, name)
PAIRS = [
    ('UNG', 'XLU', 'NatGas->Utilities'),
    ('USO', 'XLE', 'Oil->Energy'),
    ('GLD', 'GDX', 'Gold->Miners'),
    ('DBA', 'XLP', 'Agriculture->Staples'),
    ('WOOD', 'XHB', 'Lumber->Homebuilders'),
]

# Additional context tickers
CONTEXT_TICKERS = ['SPY']

# Parameters
LOOKBACK = 20          # rolling window for z-score
ENTRY_THRESHOLD = 1.2  # z-score threshold for entry (>1.2 stdev move)
HOLD_DAYS = 4          # 3-5 day hold, use 4 as middle
OPTION_DELTA = 0.50    # ATM option delta proxy
OPTION_GAMMA_BOOST = 1.20  # convexity on winners (~20% extra)
START_DATE = '2018-01-01'
END_DATE = '2026-08-15'

# Download all tickers
all_tickers = list(set([p[0] for p in PAIRS] + [p[1] for p in PAIRS] + CONTEXT_TICKERS))
print(f"Downloading data for: {', '.join(all_tickers)}")
print(f"Period: {START_DATE} to {END_DATE}")
print()

data = yf.download(all_tickers, start=START_DATE, end=END_DATE, progress=False)

# Handle both single and multi-level columns
if isinstance(data.columns, pd.MultiIndex):
    closes = data['Close'].copy()
else:
    closes = data[['Close']].copy()
    closes.columns = all_tickers

# Forward fill and drop rows with any NaN in critical tickers
closes = closes.ffill().dropna()

print(f"Data shape: {closes.shape[0]} trading days, {closes.shape[1]} tickers")
print(f"Date range: {closes.index[0].strftime('%Y-%m-%d')} to {closes.index[-1].strftime('%Y-%m-%d')}")
print()

# ============================================================
# 2. SIGNAL GENERATION
# ============================================================

# Compute daily returns
returns = closes.pct_change()

# SPY returns for regime classification
spy_ret = returns['SPY']

# Generate signals for each pair
all_trades = []

for commodity, sector, pair_name in PAIRS:
    if commodity not in returns.columns or sector not in returns.columns:
        print(f"  SKIP {pair_name}: missing data")
        continue

    comm_ret = returns[commodity]
    sect_ret = returns[sector]

    # Rolling z-score of commodity 1-day return
    rolling_mean = comm_ret.rolling(LOOKBACK).mean()
    rolling_std = comm_ret.rolling(LOOKBACK).std()
    z_score = (comm_ret - rolling_mean) / rolling_std

    # Also compute 2-day and 3-day commodity momentum for stronger signal
    comm_ret_2d = closes[commodity].pct_change(2)
    rolling_mean_2d = comm_ret_2d.rolling(LOOKBACK).mean()
    rolling_std_2d = comm_ret_2d.rolling(LOOKBACK).std()
    z_score_2d = (comm_ret_2d - rolling_mean_2d) / rolling_std_2d

    # Composite z-score: weight 1d more heavily (faster signal)
    composite_z = 0.6 * z_score + 0.4 * z_score_2d

    # Generate entry signals
    for i in range(LOOKBACK + 5, len(closes) - HOLD_DAYS - 1):
        date = closes.index[i]
        z = composite_z.iloc[i]

        if np.isnan(z):
            continue

        # Entry: commodity z-score exceeds threshold
        if abs(z) < ENTRY_THRESHOLD:
            continue

        # Direction: commodity up -> sector should follow (long call)
        # commodity down -> sector should follow (long put)
        direction = 1 if z > 0 else -1

        # Sector ETF return over hold period
        entry_price = closes[sector].iloc[i + 1]  # enter next day open proxy (use close as proxy)
        exit_price = closes[sector].iloc[i + 1 + HOLD_DAYS]

        if np.isnan(entry_price) or np.isnan(exit_price) or entry_price == 0:
            continue

        etf_return = (exit_price / entry_price - 1) * direction

        # Option return proxy: delta * ETF return + gamma boost on winners
        if etf_return > 0:
            option_return = etf_return * OPTION_DELTA * OPTION_GAMMA_BOOST / OPTION_DELTA
            # Simplify: option return ~ 1.2x ETF return for winners (gamma helps)
            option_return = etf_return * OPTION_GAMMA_BOOST
        else:
            # Losers: straight delta loss, no gamma help, plus some theta decay
            theta_drag = 0.005  # ~0.5% theta over 4 days for weekly ATM
            option_return = etf_return - theta_drag

        # SPY regime on entry day
        spy_5d = spy_ret.iloc[max(0,i-4):i+1].sum()
        regime = 'green' if spy_5d > 0 else 'red'

        # Check for overlapping trades (same sector)
        overlap = False
        for t in all_trades[-20:]:  # check recent trades only
            if t['sector'] == sector:
                t_exit = t['entry_date'] + timedelta(days=HOLD_DAYS + 2)
                if date < t_exit:
                    overlap = True
                    break

        if overlap:
            continue

        all_trades.append({
            'entry_date': date,
            'pair': pair_name,
            'commodity': commodity,
            'sector': sector,
            'direction': direction,
            'z_score': z,
            'etf_return': etf_return,
            'option_return': option_return,
            'regime': regime,
        })

trades_df = pd.DataFrame(all_trades)
print(f"Total raw signals: {len(trades_df)}")

# ============================================================
# 3. BACKTEST METRICS
# ============================================================

def compute_metrics(returns_series, label=""):
    """Compute strategy metrics from a series of trade returns."""
    if len(returns_series) == 0:
        return None

    n_trades = len(returns_series)
    total_return = (1 + returns_series).prod() - 1
    win_rate = (returns_series > 0).mean()

    winners = returns_series[returns_series > 0]
    losers = returns_series[returns_series <= 0]

    avg_win = winners.mean() if len(winners) > 0 else 0
    avg_loss = abs(losers.mean()) if len(losers) > 0 else 0.001
    profit_factor = (winners.sum() / abs(losers.sum())) if len(losers) > 0 and losers.sum() != 0 else 999

    # Annualized Sharpe (assume ~60 trades/year for weekly trades)
    trades_per_year = min(60, n_trades)  # cap at actual frequency
    mean_ret = returns_series.mean()
    std_ret = returns_series.std()
    sharpe = (mean_ret / std_ret) * np.sqrt(trades_per_year) if std_ret > 0 else 0

    # Sortino
    downside = returns_series[returns_series < 0]
    downside_std = downside.std() if len(downside) > 1 else std_ret
    sortino = (mean_ret / downside_std) * np.sqrt(trades_per_year) if downside_std > 0 else 0

    # Max drawdown (sequential equity curve)
    equity = (1 + returns_series).cumprod()
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    max_dd = drawdown.min()

    return {
        'label': label,
        'n_trades': n_trades,
        'total_return': total_return,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_drawdown': max_dd,
        'mean_return': mean_ret,
        'std_return': std_ret,
    }

def print_metrics(m):
    if m is None:
        print("  No trades.")
        return
    print(f"  {m['label']}")
    print(f"    Trades: {m['n_trades']}")
    print(f"    Win Rate: {m['win_rate']:.1%}")
    print(f"    Profit Factor: {m['profit_factor']:.2f}")
    print(f"    Sharpe: {m['sharpe']:.2f}")
    print(f"    Sortino: {m['sortino']:.2f}")
    print(f"    Max Drawdown: {m['max_drawdown']:.1%}")
    print(f"    Total Return: {m['total_return']:.1%}")
    print(f"    Avg Win: {m['avg_win']:.2%} | Avg Loss: {m['avg_loss']:.2%}")
    print()

# ============================================================
# 4. RESULTS — OVERALL AND PER PAIR
# ============================================================

print("\n" + "=" * 70)
print("RESULTS — OPTION RETURN PROXY (ATM, 4-day hold)")
print("=" * 70)

# Overall
option_returns = trades_df['option_return'].values
overall = compute_metrics(pd.Series(option_returns), "OVERALL (all pairs combined)")
print_metrics(overall)

# Per pair
print("-" * 50)
print("PER PAIR BREAKDOWN:")
print("-" * 50)
for pair_name in trades_df['pair'].unique():
    mask = trades_df['pair'] == pair_name
    pair_rets = pd.Series(trades_df.loc[mask, 'option_return'].values)
    m = compute_metrics(pair_rets, pair_name)
    print_metrics(m)

# ============================================================
# 5. REGIME STRATIFICATION
# ============================================================

print("=" * 70)
print("REGIME STRATIFICATION (SPY 5-day momentum)")
print("=" * 70)

green_mask = trades_df['regime'] == 'green'
red_mask = trades_df['regime'] == 'red'

green_rets = pd.Series(trades_df.loc[green_mask, 'option_return'].values)
red_rets = pd.Series(trades_df.loc[red_mask, 'option_return'].values)

green_m = compute_metrics(green_rets, "GREEN regime (SPY 5d > 0)")
red_m = compute_metrics(red_rets, "RED regime (SPY 5d < 0)")

print_metrics(green_m)
print_metrics(red_m)

if green_m and red_m:
    regime_gap = abs(green_m['sharpe'] - red_m['sharpe']) / max(abs(green_m['sharpe']), abs(red_m['sharpe']), 0.001)
    print(f"  Regime gap: {regime_gap:.2f} (threshold: < 0.50)")
else:
    regime_gap = 999

# ============================================================
# 6. PERMUTATION TEST
# ============================================================

print("\n" + "=" * 70)
print("PERMUTATION TEST (1000 shuffles)")
print("=" * 70)

observed_sharpe = overall['sharpe'] if overall else 0
observed_mean = overall['mean_return'] if overall else 0

n_perms = 1000
perm_sharpes = []
perm_means = []

# Pool of all possible sector returns (any 4-day window) for realistic null
all_sector_rets = []
for _, sector, _ in PAIRS:
    if sector in returns.columns:
        sect_close = closes[sector]
        for i in range(len(sect_close) - HOLD_DAYS - 1):
            r = sect_close.iloc[i + HOLD_DAYS] / sect_close.iloc[i] - 1
            if not np.isnan(r):
                all_sector_rets.append(r)

all_sector_rets = np.array(all_sector_rets)
n_trades_total = len(trades_df)

for p in range(n_perms):
    # Random sample of sector returns (same N as our strategy)
    random_rets = np.random.choice(all_sector_rets, size=n_trades_total, replace=True)
    # Apply random direction
    random_dirs = np.random.choice([-1, 1], size=n_trades_total)
    random_trade_rets = random_rets * random_dirs

    # Apply same option proxy
    option_rets = np.where(
        random_trade_rets > 0,
        random_trade_rets * OPTION_GAMMA_BOOST,
        random_trade_rets - 0.005
    )

    mean_r = option_rets.mean()
    std_r = option_rets.std()
    trades_per_year = min(60, n_trades_total)
    perm_sharpe = (mean_r / std_r) * np.sqrt(trades_per_year) if std_r > 0 else 0
    perm_sharpes.append(perm_sharpe)
    perm_means.append(mean_r)

perm_sharpes = np.array(perm_sharpes)
p_value_sharpe = (perm_sharpes >= observed_sharpe).mean()
p_value_mean = (np.array(perm_means) >= observed_mean).mean()

print(f"  Observed Sharpe: {observed_sharpe:.2f}")
print(f"  Permutation p-value (Sharpe): {p_value_sharpe:.4f}")
print(f"  Permutation p-value (mean return): {p_value_mean:.4f}")
print(f"  Perm Sharpe distribution: mean={perm_sharpes.mean():.2f}, std={perm_sharpes.std():.2f}")
print(f"  Perm 95th percentile Sharpe: {np.percentile(perm_sharpes, 95):.2f}")
print()

# ============================================================
# 7. FIVE-GATE EVALUATION
# ============================================================

print("=" * 70)
print("5-GATE EVALUATION")
print("=" * 70)

gates = {
    'Gate 1 — Sharpe > 0.5': overall['sharpe'] > 0.5 if overall else False,
    'Gate 2 — Perm p-value < 0.05': p_value_sharpe < 0.05,
    'Gate 3 — Regime gap < 0.50': regime_gap < 0.50,
    'Gate 4 — Max DD < 50%': abs(overall['max_drawdown']) < 0.50 if overall else False,
    'Gate 5 — At least 30 trades': n_trades_total >= 30,
}

all_pass = True
for gate, passed in gates.items():
    status = "PASS" if passed else "FAIL"
    if not passed:
        all_pass = False
    print(f"  [{status}] {gate}")

print()
if all_pass:
    print("  >>> OVERALL: PASS — This signal has edge. Wire into confluence system.")
else:
    print("  >>> OVERALL: FAIL — Signal does not clear all gates.")
    failed = [g for g, p in gates.items() if not p]
    print(f"  >>> Failed gates: {', '.join(failed)}")

# ============================================================
# 8. BEST PAIRS ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("PAIR-LEVEL GATE CHECK (which pairs have standalone edge?)")
print("=" * 70)

for pair_name in trades_df['pair'].unique():
    mask = trades_df['pair'] == pair_name
    pair_rets = pd.Series(trades_df.loc[mask, 'option_return'].values)
    m = compute_metrics(pair_rets, pair_name)
    if m is None:
        continue

    # Pair-level permutation (quick, 500 shuffles)
    pair_n = m['n_trades']
    pair_perm_sharpes = []
    for _ in range(500):
        rr = np.random.choice(all_sector_rets, size=pair_n, replace=True)
        rd = np.random.choice([-1, 1], size=pair_n)
        rtr = rr * rd
        orp = np.where(rtr > 0, rtr * OPTION_GAMMA_BOOST, rtr - 0.005)
        mr = orp.mean()
        sr = orp.std()
        ps = (mr / sr) * np.sqrt(min(60, pair_n)) if sr > 0 else 0
        pair_perm_sharpes.append(ps)

    pair_pval = (np.array(pair_perm_sharpes) >= m['sharpe']).mean()

    # Regime for this pair
    pair_green = pd.Series(trades_df.loc[mask & green_mask, 'option_return'].values)
    pair_red = pd.Series(trades_df.loc[mask & red_mask, 'option_return'].values)
    gm = compute_metrics(pair_green, "green") if len(pair_green) > 2 else None
    rm = compute_metrics(pair_red, "red") if len(pair_red) > 2 else None

    if gm and rm and max(abs(gm['sharpe']), abs(rm['sharpe'])) > 0:
        pair_regime_gap = abs(gm['sharpe'] - rm['sharpe']) / max(abs(gm['sharpe']), abs(rm['sharpe']))
    else:
        pair_regime_gap = 999

    sharpe_ok = "OK" if m['sharpe'] > 0.5 else "NO"
    pval_ok = "OK" if pair_pval < 0.05 else "NO"
    regime_ok = "OK" if pair_regime_gap < 0.50 else "NO"
    n_ok = "OK" if m['n_trades'] >= 30 else "NO"

    print(f"\n  {pair_name}:")
    print(f"    N={m['n_trades']} [{n_ok}] | Sharpe={m['sharpe']:.2f} [{sharpe_ok}] | "
          f"p={pair_pval:.3f} [{pval_ok}] | WR={m['win_rate']:.1%} | PF={m['profit_factor']:.2f} | "
          f"Regime gap={pair_regime_gap:.2f} [{regime_ok}]")

# ============================================================
# 9. SIGNAL STRENGTH ANALYSIS
# ============================================================

print("\n\n" + "=" * 70)
print("SIGNAL STRENGTH: Does higher z-score = better returns?")
print("=" * 70)

# Bin trades by z-score magnitude
trades_df['abs_z'] = trades_df['z_score'].abs()
bins = [1.2, 1.5, 2.0, 2.5, 10.0]
labels = ['1.2-1.5', '1.5-2.0', '2.0-2.5', '2.5+']
trades_df['z_bin'] = pd.cut(trades_df['abs_z'], bins=bins, labels=labels)

print(f"\n  {'Z-Score Bin':<12} {'N':>5} {'WR':>8} {'Avg Ret':>10} {'Sharpe':>8}")
print(f"  {'-'*48}")

for label in labels:
    bin_mask = trades_df['z_bin'] == label
    bin_rets = trades_df.loc[bin_mask, 'option_return']
    if len(bin_rets) < 3:
        continue
    wr = (bin_rets > 0).mean()
    avg = bin_rets.mean()
    std = bin_rets.std()
    sharpe = (avg / std) * np.sqrt(min(60, len(bin_rets))) if std > 0 else 0
    print(f"  {label:<12} {len(bin_rets):>5} {wr:>7.1%} {avg:>9.2%} {sharpe:>8.2f}")

# ============================================================
# 10. YEARLY BREAKDOWN
# ============================================================

print("\n\n" + "=" * 70)
print("YEARLY BREAKDOWN")
print("=" * 70)

trades_df['year'] = trades_df['entry_date'].dt.year

print(f"\n  {'Year':<6} {'N':>5} {'WR':>8} {'Sharpe':>8} {'Total Ret':>10}")
print(f"  {'-'*42}")

for year in sorted(trades_df['year'].unique()):
    yr_mask = trades_df['year'] == year
    yr_rets = pd.Series(trades_df.loc[yr_mask, 'option_return'].values)
    m = compute_metrics(yr_rets, str(year))
    if m:
        print(f"  {year:<6} {m['n_trades']:>5} {m['win_rate']:>7.1%} {m['sharpe']:>8.2f} {m['total_return']:>9.1%}")

print("\n" + "=" * 70)
print("BACKTEST COMPLETE")
print("=" * 70)

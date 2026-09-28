#!/usr/bin/env python3
"""
Cross-Asset Lead-Lag: Commodities → Sector ETFs — EXTREME SIGNALS ONLY
=======================================================================
Refined version: only trade when commodity z-score > 2.0 (extreme moves).
The base backtest showed z > 2.5 has Sharpe 2.13 / 66.7% WR on 45 trades.
Now test z > 2.0 threshold to get more trades while preserving edge.

Also adds: confirmation filter (commodity move must be in same direction
for 2+ consecutive days to reduce whipsaw).

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

print("=" * 70)
print("COMMODITY-SECTOR LEAD-LAG — EXTREME SIGNALS (z > 2.0)")
print("=" * 70)
print()

PAIRS = [
    ('UNG', 'XLU', 'NatGas->Utilities'),
    ('USO', 'XLE', 'Oil->Energy'),
    ('GLD', 'GDX', 'Gold->Miners'),
    ('DBA', 'XLP', 'Agriculture->Staples'),
    ('WOOD', 'XHB', 'Lumber->Homebuilders'),
]

CONTEXT_TICKERS = ['SPY']
LOOKBACK = 20
ENTRY_THRESHOLD = 2.0  # Higher threshold — extreme moves only
HOLD_DAYS = 4
OPTION_GAMMA_BOOST = 1.20
THETA_DRAG = 0.005
START_DATE = '2018-01-01'
END_DATE = '2026-08-15'

all_tickers = list(set([p[0] for p in PAIRS] + [p[1] for p in PAIRS] + CONTEXT_TICKERS))
print(f"Downloading: {', '.join(all_tickers)}")
data = yf.download(all_tickers, start=START_DATE, end=END_DATE, progress=False)

if isinstance(data.columns, pd.MultiIndex):
    closes = data['Close'].copy()
else:
    closes = data[['Close']].copy()

closes = closes.ffill().dropna()
returns = closes.pct_change()
spy_ret = returns['SPY']

print(f"Data: {closes.shape[0]} days, {closes.index[0].strftime('%Y-%m-%d')} to {closes.index[-1].strftime('%Y-%m-%d')}")
print()

# ============================================================
# SIGNAL GENERATION — EXTREME Z + CONFIRMATION
# ============================================================

all_trades = []

for commodity, sector, pair_name in PAIRS:
    if commodity not in returns.columns or sector not in returns.columns:
        continue

    comm_ret = returns[commodity]

    # Rolling z-score
    rolling_mean = comm_ret.rolling(LOOKBACK).mean()
    rolling_std = comm_ret.rolling(LOOKBACK).std()
    z_score = (comm_ret - rolling_mean) / rolling_std

    # 2-day momentum z-score
    comm_ret_2d = closes[commodity].pct_change(2)
    rm2 = comm_ret_2d.rolling(LOOKBACK).mean()
    rs2 = comm_ret_2d.rolling(LOOKBACK).std()
    z_2d = (comm_ret_2d - rm2) / rs2

    # Composite
    composite_z = 0.6 * z_score + 0.4 * z_2d

    for i in range(LOOKBACK + 5, len(closes) - HOLD_DAYS - 1):
        date = closes.index[i]
        z = composite_z.iloc[i]

        if np.isnan(z) or abs(z) < ENTRY_THRESHOLD:
            continue

        direction = 1 if z > 0 else -1

        # CONFIRMATION: commodity return on day i must agree with direction
        # AND the 2-day return must also agree (momentum confirmation)
        day_ret = comm_ret.iloc[i]
        day_ret_prev = comm_ret.iloc[i-1] if i > 0 else 0
        if np.isnan(day_ret) or np.isnan(day_ret_prev):
            continue

        # Both today and yesterday must move in same direction
        if direction > 0 and (day_ret <= 0 or day_ret_prev <= 0):
            continue
        if direction < 0 and (day_ret >= 0 or day_ret_prev >= 0):
            continue

        # Sector ETF return over hold period (enter next day)
        entry_price = closes[sector].iloc[i + 1]
        exit_price = closes[sector].iloc[i + 1 + HOLD_DAYS]

        if np.isnan(entry_price) or np.isnan(exit_price) or entry_price == 0:
            continue

        etf_return = (exit_price / entry_price - 1) * direction

        # Option return proxy
        if etf_return > 0:
            option_return = etf_return * OPTION_GAMMA_BOOST
        else:
            option_return = etf_return - THETA_DRAG

        # Regime
        spy_5d = spy_ret.iloc[max(0, i-4):i+1].sum()
        regime = 'green' if spy_5d > 0 else 'red'

        # No overlap check (same sector, within hold period)
        overlap = False
        for t in all_trades[-15:]:
            if t['sector'] == sector:
                days_since = (date - t['entry_date']).days
                if days_since < HOLD_DAYS + 2:
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
print(f"Total trades (extreme + confirmed): {len(trades_df)}")

# ============================================================
# METRICS FUNCTIONS
# ============================================================

def compute_metrics(rets, label=""):
    if len(rets) == 0:
        return None
    rets = pd.Series(rets).reset_index(drop=True)
    n = len(rets)
    wr = (rets > 0).mean()
    winners = rets[rets > 0]
    losers = rets[rets <= 0]
    avg_win = winners.mean() if len(winners) > 0 else 0
    avg_loss = abs(losers.mean()) if len(losers) > 0 else 0.001
    pf = (winners.sum() / abs(losers.sum())) if len(losers) > 0 and losers.sum() != 0 else 999
    tpy = min(60, n)
    mu = rets.mean()
    sigma = rets.std()
    sharpe = (mu / sigma) * np.sqrt(tpy) if sigma > 0 else 0
    down = rets[rets < 0]
    down_std = down.std() if len(down) > 1 else sigma
    sortino = (mu / down_std) * np.sqrt(tpy) if down_std > 0 else 0
    eq = (1 + rets).cumprod()
    dd = (eq - eq.cummax()) / eq.cummax()
    max_dd = dd.min()
    total_ret = eq.iloc[-1] - 1
    return {
        'label': label, 'n_trades': n, 'win_rate': wr, 'avg_win': avg_win,
        'avg_loss': avg_loss, 'profit_factor': pf, 'sharpe': sharpe,
        'sortino': sortino, 'max_drawdown': max_dd, 'total_return': total_ret,
        'mean_return': mu, 'std_return': sigma,
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
# RESULTS
# ============================================================

if len(trades_df) == 0:
    print("NO TRADES GENERATED. Strategy too restrictive.")
    exit()

print("\n" + "=" * 70)
print("RESULTS — EXTREME SIGNALS + MOMENTUM CONFIRMATION")
print("=" * 70)

option_returns = trades_df['option_return'].values
overall = compute_metrics(option_returns, "OVERALL")
print_metrics(overall)

# Per pair
print("-" * 50)
for pair_name in trades_df['pair'].unique():
    mask = trades_df['pair'] == pair_name
    m = compute_metrics(trades_df.loc[mask, 'option_return'].values, pair_name)
    print_metrics(m)

# Regime
print("=" * 70)
print("REGIME STRATIFICATION")
print("=" * 70)

green_mask = trades_df['regime'] == 'green'
red_mask = trades_df['regime'] == 'red'

green_m = compute_metrics(trades_df.loc[green_mask, 'option_return'].values, "GREEN (SPY 5d > 0)")
red_m = compute_metrics(trades_df.loc[red_mask, 'option_return'].values, "RED (SPY 5d < 0)")
print_metrics(green_m)
print_metrics(red_m)

if green_m and red_m and max(abs(green_m['sharpe']), abs(red_m['sharpe'])) > 0:
    regime_gap = abs(green_m['sharpe'] - red_m['sharpe']) / max(abs(green_m['sharpe']), abs(red_m['sharpe']))
else:
    regime_gap = 999
print(f"  Regime gap: {regime_gap:.2f}")

# Permutation test
print("\n" + "=" * 70)
print("PERMUTATION TEST (1000 shuffles)")
print("=" * 70)

# Build null distribution from all 4-day sector returns
all_sector_rets = []
for _, sector, _ in PAIRS:
    if sector in closes.columns:
        sc = closes[sector]
        for i in range(len(sc) - HOLD_DAYS - 1):
            r = sc.iloc[i + HOLD_DAYS] / sc.iloc[i] - 1
            if not np.isnan(r):
                all_sector_rets.append(r)
all_sector_rets = np.array(all_sector_rets)

n_trades_total = len(trades_df)
observed_sharpe = overall['sharpe']

perm_sharpes = []
for _ in range(1000):
    rr = np.random.choice(all_sector_rets, size=n_trades_total, replace=True)
    rd = np.random.choice([-1, 1], size=n_trades_total)
    rtr = rr * rd
    orp = np.where(rtr > 0, rtr * OPTION_GAMMA_BOOST, rtr - THETA_DRAG)
    mu = orp.mean()
    sig = orp.std()
    ps = (mu / sig) * np.sqrt(min(60, n_trades_total)) if sig > 0 else 0
    perm_sharpes.append(ps)

perm_sharpes = np.array(perm_sharpes)
p_value = (perm_sharpes >= observed_sharpe).mean()

print(f"  Observed Sharpe: {observed_sharpe:.2f}")
print(f"  p-value: {p_value:.4f}")
print(f"  Null distribution: mean={perm_sharpes.mean():.2f}, 95th={np.percentile(perm_sharpes, 95):.2f}")

# ============================================================
# 5-GATE EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("5-GATE EVALUATION")
print("=" * 70)

gates = {
    'Gate 1 — Sharpe > 0.5': overall['sharpe'] > 0.5,
    'Gate 2 — Perm p-value < 0.05': p_value < 0.05,
    'Gate 3 — Regime gap < 0.50': regime_gap < 0.50,
    'Gate 4 — Max DD < 50%': abs(overall['max_drawdown']) < 0.50,
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
    print("  >>> OVERALL: PASS — Wire into confluence.")
else:
    print("  >>> OVERALL: FAIL")
    failed = [g for g, p in gates.items() if not p]
    print(f"  >>> Failed: {', '.join(failed)}")

# Direction breakdown
print("\n" + "=" * 70)
print("DIRECTION BREAKDOWN")
print("=" * 70)

long_mask = trades_df['direction'] == 1
short_mask = trades_df['direction'] == -1
long_m = compute_metrics(trades_df.loc[long_mask, 'option_return'].values, "LONG (calls)")
short_m = compute_metrics(trades_df.loc[short_mask, 'option_return'].values, "SHORT (puts)")
print_metrics(long_m)
print_metrics(short_m)

# Yearly
print("=" * 70)
print("YEARLY BREAKDOWN")
print("=" * 70)
trades_df['year'] = trades_df['entry_date'].dt.year
print(f"\n  {'Year':<6} {'N':>5} {'WR':>8} {'Sharpe':>8} {'PF':>6} {'Total':>8}")
print(f"  {'-'*44}")
for year in sorted(trades_df['year'].unique()):
    yr = trades_df[trades_df['year'] == year]
    m = compute_metrics(yr['option_return'].values, str(year))
    if m:
        print(f"  {year:<6} {m['n_trades']:>5} {m['win_rate']:>7.1%} {m['sharpe']:>8.2f} {m['profit_factor']:>6.2f} {m['total_return']:>7.1%}")

# Sample trades
print("\n" + "=" * 70)
print("SAMPLE TRADES (last 15)")
print("=" * 70)
sample = trades_df.tail(15)
print(f"\n  {'Date':<12} {'Pair':<22} {'Dir':>4} {'Z':>6} {'ETF Ret':>8} {'Opt Ret':>8} {'Regime'}")
print(f"  {'-'*72}")
for _, t in sample.iterrows():
    d = 'LONG' if t['direction'] == 1 else 'SHRT'
    print(f"  {t['entry_date'].strftime('%Y-%m-%d'):<12} {t['pair']:<22} {d:>4} {t['z_score']:>6.1f} {t['etf_return']:>7.2%} {t['option_return']:>7.2%} {t['regime']}")

print("\n" + "=" * 70)
print("BACKTEST COMPLETE")
print("=" * 70)

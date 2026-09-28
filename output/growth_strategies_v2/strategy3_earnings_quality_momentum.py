"""
Strategy 3: Earnings Momentum + Quality Factor (ETF Proxy)
============================================================
Buy stocks/sectors with sustained earnings improvement AND improving quality.

Since individual stock fundamentals require expensive data, we use sector ETFs
as proxies and earnings revision data from price-based proxies:

Approach:
- Universe: 11 SPDR sector ETFs (XLK, XLF, XLV, XLE, XLI, XLY, XLP, XLB, XLRE, XLU, XLC)
- Earnings momentum proxy: 3M earnings yield change (E/P ratio improvement)
  We approximate this using sector ETF's price momentum + relative strength vs SPY
  combined with fundamental quality proxy (low vol + high return = quality)
- Quality proxy: Sharpe ratio over trailing 12 months (high = quality)
- Signal: Rank sectors by composite score = 0.5 * earnings_momentum_rank + 0.5 * quality_rank
- Monthly rebalance, top 3 sectors, equal weight
- T-1 signals, 10 bps costs

This differs from simple momentum because:
1. Quality filter removes "junk rallies" (momentum in low-quality sectors)
2. Combines price momentum with quality (Sharpe-based), not just returns
3. Bottom 3 quality sectors are always excluded regardless of momentum
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import json
import os
import warnings
warnings.filterwarnings('ignore')

OUT_DIR = '/home/jupiter/Lvl3Quant/output/growth_strategies_v2'

# ── Data Download ──
SECTORS = ['XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLY', 'XLP', 'XLB', 'XLRE', 'XLU', 'XLC']
BENCHMARK = 'SPY'
ALL_TICKERS = SECTORS + [BENCHMARK]

print("Downloading sector data...")
data = yf.download(ALL_TICKERS, start='2009-01-01', end='2026-07-18', auto_adjust=True)
prices = data['Close'].copy()
prices = prices.ffill().dropna(how='all')

# Some sector ETFs start later (XLC: 2018, XLRE: 2015)
# We'll use available data for each
print(f"Data range: {prices.index[0]} to {prices.index[-1]}")
for t in SECTORS:
    if t in prices.columns:
        first_valid = prices[t].first_valid_index()
        print(f"  {t}: from {first_valid}")

# Monthly data
monthly = prices.resample('ME').last()
monthly_ret = monthly.pct_change()

# Daily returns for quality calc
daily_ret = prices.pct_change()

# ── Strategy Logic ──
LOOKBACK_MOM = 12  # months for momentum
LOOKBACK_QUALITY = 12  # months for quality (Sharpe)
SHORT_MOM = 3  # months for short-term earnings momentum proxy
TOP_N = 3
COST_BPS = 10

def calc_trailing_sharpe(daily_returns, end_date, months=12):
    """Calculate trailing Sharpe ratio for a given period."""
    start_date = end_date - pd.DateOffset(months=months)
    period_ret = daily_returns.loc[start_date:end_date]
    if len(period_ret) < 60:  # need at least 60 trading days
        return np.nan
    ann_ret = period_ret.mean() * 252
    ann_vol = period_ret.std() * np.sqrt(252)
    if ann_vol == 0:
        return 0
    return ann_ret / ann_vol


def run_earnings_quality_strategy(monthly_prices, monthly_returns, daily_returns, lag=1):
    """
    Earnings Quality Momentum:
    - Momentum signal: 12M return (absolute) + 3M acceleration
    - Quality signal: trailing 12M Sharpe ratio
    - Composite: rank by 0.5*momentum_rank + 0.5*quality_rank
    - Exclude bottom 3 quality sectors
    - Pick top 3 from remaining
    """
    available_sectors = [s for s in SECTORS if s in monthly_prices.columns]

    start_idx = max(LOOKBACK_MOM, LOOKBACK_QUALITY) + lag + 1
    dates = monthly_prices.index[start_idx:]

    returns_list = []
    holdings_history = []
    prev_holdings = set()

    for date in dates:
        idx = monthly_prices.index.get_loc(date)
        signal_idx = idx - lag

        if signal_idx < LOOKBACK_MOM:
            continue

        # Calculate signals for each sector
        sector_scores = {}
        for sector in available_sectors:
            if pd.isna(monthly_prices[sector].iloc[signal_idx]):
                continue

            # 12M momentum
            p_now = monthly_prices[sector].iloc[signal_idx]
            p_12m = monthly_prices[sector].iloc[signal_idx - LOOKBACK_MOM]
            if pd.isna(p_12m) or p_12m == 0:
                continue
            mom_12m = p_now / p_12m - 1

            # 3M momentum (earnings acceleration proxy)
            p_3m = monthly_prices[sector].iloc[signal_idx - SHORT_MOM]
            if pd.isna(p_3m) or p_3m == 0:
                continue
            mom_3m = p_now / p_3m - 1

            # Momentum score: 12M + extra weight on recent 3M acceleration
            # If 3M is stronger relative to 12M, suggests improving trajectory
            if mom_12m != 0:
                acceleration = mom_3m / (mom_12m / 4) - 1  # vs expected 3M from 12M trend
            else:
                acceleration = 0

            momentum_score = mom_12m + 0.3 * acceleration

            # Quality: trailing Sharpe
            signal_date = monthly_prices.index[signal_idx]
            quality = calc_trailing_sharpe(daily_returns[sector], signal_date, LOOKBACK_QUALITY)
            if pd.isna(quality):
                continue

            sector_scores[sector] = {
                'momentum': momentum_score,
                'quality': quality,
                'mom_12m': mom_12m,
                'mom_3m': mom_3m,
            }

        if len(sector_scores) < 4:
            continue

        # Rank sectors
        sectors_list = list(sector_scores.keys())
        mom_vals = [sector_scores[s]['momentum'] for s in sectors_list]
        qual_vals = [sector_scores[s]['quality'] for s in sectors_list]

        # Percentile rank (0 to 1)
        mom_ranks = pd.Series(mom_vals, index=sectors_list).rank(pct=True)
        qual_ranks = pd.Series(qual_vals, index=sectors_list).rank(pct=True)

        # Exclude bottom 3 quality sectors
        qual_sorted = qual_ranks.sort_values()
        excluded = set(qual_sorted.index[:3])

        # Composite score (among non-excluded)
        composite = {}
        for s in sectors_list:
            if s not in excluded:
                composite[s] = 0.5 * mom_ranks[s] + 0.5 * qual_ranks[s]

        if len(composite) < TOP_N:
            continue

        # Pick top N
        sorted_composite = sorted(composite.items(), key=lambda x: x[1], reverse=True)
        selected = [s for s, _ in sorted_composite[:TOP_N]]

        # Only invest if at least 1 selected sector has positive absolute momentum
        has_positive = any(sector_scores[s]['mom_12m'] > 0 for s in selected)
        if not has_positive:
            selected = []  # go to cash

        # Equal weight
        if selected:
            w = 1.0 / len(selected)
        else:
            w = 0

        # Turnover
        new_holdings = set(selected)
        turnover = len(new_holdings.symmetric_difference(prev_holdings)) / max(TOP_N, 1)

        # Period return
        ret_idx = monthly_prices.index.get_loc(date)
        if ret_idx >= len(monthly_returns):
            break

        if selected:
            period_ret = sum(w * monthly_returns.iloc[ret_idx].get(s, 0) for s in selected)
        else:
            period_ret = 0  # cash

        # Transaction cost
        cost = turnover * COST_BPS / 10000
        period_ret -= cost

        returns_list.append(period_ret)
        holdings_history.append({
            'date': str(date),
            'holdings': selected,
            'scores': {s: sector_scores[s]['momentum'] for s in selected} if selected else {},
        })
        prev_holdings = new_holdings

    return pd.Series(returns_list, index=[pd.Timestamp(h['date']) for h in holdings_history]), holdings_history


# ── Run Strategy ──
print("\nRunning earnings quality momentum with T-1 lag...")
strat_returns, holdings = run_earnings_quality_strategy(monthly, monthly_ret, daily_ret, lag=1)

print("Running with T-0 lag (lookahead test)...")
strat_returns_t0, _ = run_earnings_quality_strategy(monthly, monthly_ret, daily_ret, lag=0)

# Benchmark
spy_monthly_ret = monthly_ret['SPY']
spy_aligned = spy_monthly_ret.reindex(strat_returns.index).fillna(0)

# ── Performance Metrics ──
def calc_metrics(returns, name="Strategy"):
    cum = (1 + returns).cumprod()
    total_ret = cum.iloc[-1] - 1
    n_years = len(returns) / 12

    ann_ret = returns.mean() * 12
    ann_vol = returns.std() * np.sqrt(12)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    cagr = (1 + total_ret) ** (1/n_years) - 1 if n_years > 0 else 0

    downside = returns[returns < 0].std() * np.sqrt(12)
    sortino = ann_ret / downside if downside > 0 else 0

    cum_max = cum.cummax()
    drawdown = (cum - cum_max) / cum_max
    max_dd = drawdown.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    wins = returns[returns > 0]
    losses = returns[returns < 0]
    wr = len(wins) / len(returns) if len(returns) > 0 else 0
    pf = wins.sum() / abs(losses.sum()) if len(losses) > 0 and losses.sum() != 0 else float('inf')

    return {
        'name': name,
        'CAGR': f"{cagr:.4f}",
        'Sharpe': f"{sharpe:.3f}",
        'Sortino': f"{sortino:.3f}",
        'MaxDD': f"{max_dd:.4f}",
        'Calmar': f"{calmar:.3f}",
        'AnnVol': f"{ann_vol:.4f}",
        'WinRate': f"{wr:.3f}",
        'ProfitFactor': f"{pf:.3f}",
        'TotalReturn': f"{total_ret:.4f}",
        'N_periods': len(returns),
        'N_years': f"{n_years:.1f}",
    }


metrics_t1 = calc_metrics(strat_returns, "EarningsQuality (T-1)")
metrics_t0 = calc_metrics(strat_returns_t0, "EarningsQuality (T-0)")
metrics_spy = calc_metrics(spy_aligned, "SPY B&H")

print("\n=== PERFORMANCE COMPARISON ===")
for m in [metrics_t1, metrics_t0, metrics_spy]:
    print(f"\n{m['name']}:")
    for k, v in m.items():
        if k != 'name':
            print(f"  {k}: {v}")

# Lag sensitivity
t0_sharpe = float(metrics_t0['Sharpe'])
t1_sharpe = float(metrics_t1['Sharpe'])
degradation = (t0_sharpe - t1_sharpe) / t0_sharpe * 100 if t0_sharpe > 0 else 0

print(f"\n=== LAG SENSITIVITY TEST ===")
print(f"T-0 Sharpe: {t0_sharpe:.3f}")
print(f"T-1 Sharpe: {t1_sharpe:.3f}")
print(f"Degradation: {degradation:.1f}%")
print(f"{'PASS' if abs(degradation) < 50 else 'FAIL'}: Lag sensitivity")

# SPY correlation
corr = strat_returns.corr(spy_aligned)
print(f"\nSPY correlation: {corr:.3f}")

# ── Regime Test ──
spy_direction = spy_aligned > 0
strat_green = strat_returns[spy_direction.reindex(strat_returns.index, fill_value=False)]
strat_red = strat_returns[~spy_direction.reindex(strat_returns.index, fill_value=True)]

regime_gap = None
if len(strat_green) > 6 and len(strat_red) > 6:
    sharpe_green = strat_green.mean() / strat_green.std() * np.sqrt(12) if strat_green.std() > 0 else 0
    sharpe_red = strat_red.mean() / strat_red.std() * np.sqrt(12) if strat_red.std() > 0 else 0
    regime_gap = abs(sharpe_green - sharpe_red) / max(abs(sharpe_green), abs(sharpe_red)) if max(abs(sharpe_green), abs(sharpe_red)) > 0 else 0

    print(f"\n=== REGIME TEST ===")
    print(f"Sharpe (green months): {sharpe_green:.3f}")
    print(f"Sharpe (red months): {sharpe_red:.3f}")
    print(f"Regime gap ratio: {regime_gap:.3f}")
    print(f"{'PASS' if regime_gap < 0.50 else 'FAIL'}: Regime gap")

# ── Sub-period Stability ──
n = len(strat_returns)
quarters = [strat_returns.iloc[i*n//4:(i+1)*n//4] for i in range(4)]
quarter_sharpes = []
for i, q in enumerate(quarters):
    if len(q) > 3 and q.std() > 0:
        qs = q.mean() / q.std() * np.sqrt(12)
        quarter_sharpes.append(qs)
        print(f"Sub-period {i+1} Sharpe: {qs:.3f}")

cv = None
if len(quarter_sharpes) >= 2 and np.mean(quarter_sharpes) != 0:
    cv = np.std(quarter_sharpes) / abs(np.mean(quarter_sharpes))
    print(f"\n=== SUB-PERIOD STABILITY ===")
    print(f"CV of Sharpe: {cv:.3f}")
    print(f"{'PASS' if cv < 0.70 else 'FAIL'}: CV {'<' if cv < 0.70 else '>='} 0.70")

# ── Permutation Test ──
print("\n=== PERMUTATION TEST (200 permutations) ===")
actual_sharpe = float(metrics_t1['Sharpe'])
n_perms = 200
perm_sharpes = []

np.random.seed(42)
returns_array = strat_returns.values.copy()
for _ in range(n_perms):
    shuffled = np.random.permutation(returns_array)
    s = pd.Series(shuffled)
    if s.std() > 0:
        perm_sharpes.append(s.mean() / s.std() * np.sqrt(12))
    else:
        perm_sharpes.append(0)

p_value = np.mean([ps >= actual_sharpe for ps in perm_sharpes])
print(f"Actual Sharpe: {actual_sharpe:.3f}")
print(f"Permutation mean Sharpe: {np.mean(perm_sharpes):.3f}")
print(f"P-value: {p_value:.4f}")
print(f"{'PASS' if p_value < 0.05 else 'FAIL'}: p-value {'<' if p_value < 0.05 else '>='} 0.05")

# ── Holdings Analysis ──
print(f"\n=== HOLDINGS ANALYSIS ===")
sector_counts = {}
for h in holdings:
    for s in h['holdings']:
        sector_counts[s] = sector_counts.get(s, 0) + 1
total_periods = len(holdings)
print("Sector frequency (% of periods held):")
for s, c in sorted(sector_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  {s}: {c/total_periods*100:.1f}%")

cash_pct = sum(1 for h in holdings if not h['holdings']) / total_periods * 100
print(f"Cash (no holdings): {cash_pct:.1f}%")

# ── Save Results ──
results = {
    'strategy': 'Earnings Momentum + Quality Factor',
    'metrics_t1': metrics_t1,
    'metrics_t0': metrics_t0,
    'metrics_spy': metrics_spy,
    'spy_correlation': float(corr),
    'lag_degradation_pct': degradation,
    'permutation_p_value': float(p_value),
    'regime_gap': float(regime_gap) if regime_gap is not None else None,
    'subperiod_cv': float(cv) if cv is not None else None,
    'sector_frequency': {s: c/total_periods for s, c in sector_counts.items()},
}

with open(os.path.join(OUT_DIR, 'strategy3_results.json'), 'w') as f:
    json.dump(results, f, indent=2, default=str)

strat_returns.to_csv(os.path.join(OUT_DIR, 'strategy3_returns.csv'))
print(f"\nResults saved to {OUT_DIR}/strategy3_*")
print("\n=== STRATEGY 3 COMPLETE ===")

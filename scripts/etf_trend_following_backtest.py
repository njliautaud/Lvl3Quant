#!/usr/bin/env python3
"""
ETF Trend Following Backtest (Managed Futures Style)
Based on: Moskowitz, Ooi & Pedersen (2012) "Time Series Momentum"

6 Variants:
A. Dual Moving Average (20/50 SMA crossover)
B. Breakout Channel (20-day high entry, 10-day low exit)
C. Momentum Score Rotation (3m+6m+12m z-scored returns, top 2)
D. Trend + Vol Target (Variant A with 8% vol targeting)
E. Cross-Asset Momentum (positive 3m mom + decreasing vol, top 3)
F. Adaptive Trend (SMA crossover + ADX confirmation)

Universe: SPY, QQQ, GLD, TLT, UUP, EFA, DBC (if available)
OOT: Jan 2022 - Jul 2026
Starting capital: $645
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────────────────────
UNIVERSE = ['SPY', 'QQQ', 'GLD', 'TLT', 'UUP', 'EFA', 'DBC']
START_DATE = '2021-01-01'  # extra lookback for indicators
OOT_START = '2022-01-01'
OOT_END = '2026-07-30'
INITIAL_CAPITAL = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02%
N_PERMUTATIONS = 1000

# ── Data Download ──────────────────────────────────────────────────────────
print("Downloading ETF data...")
data = {}
available_tickers = []
for ticker in UNIVERSE:
    try:
        df = yf.download(ticker, start=START_DATE, end=OOT_END, progress=False, auto_adjust=True)
        if df is not None and len(df) > 252:
            # Handle multi-level columns from yfinance
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            data[ticker] = df
            available_tickers.append(ticker)
            print(f"  {ticker}: {len(df)} days ({df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')})")
        else:
            print(f"  {ticker}: insufficient data, skipping")
    except Exception as e:
        print(f"  {ticker}: download failed ({e}), skipping")

print(f"\nAvailable universe: {available_tickers}")

# Build aligned close price DataFrame
closes = pd.DataFrame({t: data[t]['Close'] for t in available_tickers})
closes = closes.ffill()

# SPY for regime classification
spy_close = closes['SPY']
spy_sma200 = spy_close.rolling(200).mean()

# ── Helper Functions ───────────────────────────────────────────────────────

def compute_adx(high, low, close, period=14):
    """Compute ADX indicator."""
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.rolling(period).mean()
    return adx


def apply_slippage(returns, weights, slippage=SLIPPAGE_PCT):
    """Apply slippage on rebalance days (when weights change)."""
    weight_changes = weights.diff().abs().sum(axis=1)
    slippage_cost = weight_changes * slippage
    return returns - slippage_cost


def backtest_strategy(weights, closes_oot, name):
    """Run backtest given weight DataFrame aligned to OOT closes."""
    # Daily returns of underlying ETFs
    etf_returns = closes_oot.pct_change()

    # Portfolio return = sum of weight * etf return
    # Weights are determined at close, applied next day
    shifted_weights = weights.shift(1)  # use previous day's weights
    port_returns = (shifted_weights * etf_returns).sum(axis=1)

    # Apply slippage
    weight_changes = shifted_weights.diff().abs().sum(axis=1)
    slippage_cost = weight_changes * SLIPPAGE_PCT
    port_returns = port_returns - slippage_cost

    # Drop initial NaN
    port_returns = port_returns.dropna()
    port_returns = port_returns.iloc[1:]  # first day has no prior weight

    # Equity curve
    equity = INITIAL_CAPITAL * (1 + port_returns).cumprod()

    # Count trades (rebalance events where weights change meaningfully)
    rebalances = (weights.diff().abs().sum(axis=1) > 0.01).sum()

    return port_returns, equity, int(rebalances)


def compute_metrics(returns, equity, n_trades, name):
    """Compute strategy metrics."""
    if len(returns) < 10:
        return None

    ann_factor = np.sqrt(252)
    total_return = (equity.iloc[-1] / INITIAL_CAPITAL - 1) * 100

    mu = returns.mean() * 252
    sigma = returns.std() * ann_factor
    sharpe = mu / sigma if sigma > 0 else 0

    downside = returns[returns < 0].std() * ann_factor
    sortino = mu / downside if downside > 0 else 0

    # Max drawdown
    peak = equity.cummax()
    dd = (equity - peak) / peak
    max_dd = dd.min() * 100

    # Profit factor
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    pf = gains / losses if losses > 0 else float('inf')

    # Win rate
    wr = (returns > 0).mean() * 100

    # CAGR
    years = len(returns) / 252
    cagr = ((equity.iloc[-1] / INITIAL_CAPITAL) ** (1/years) - 1) * 100 if years > 0 else 0

    return {
        'name': name,
        'total_return_pct': round(total_return, 2),
        'cagr_pct': round(cagr, 2),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'profit_factor': round(pf, 3),
        'win_rate_pct': round(wr, 1),
        'max_dd_pct': round(max_dd, 2),
        'volatility_pct': round(sigma * 100, 2),
        'n_trades': n_trades,
        'final_equity': round(equity.iloc[-1], 2),
    }


def regime_analysis(returns, spy_close_oot, spy_sma200_oot):
    """Compute Sharpe in bull vs bear regimes."""
    bull = spy_close_oot > spy_sma200_oot
    # Align
    common = returns.index.intersection(bull.index)
    returns_c = returns.loc[common]
    bull_c = bull.loc[common]

    bull_ret = returns_c[bull_c]
    bear_ret = returns_c[~bull_c]

    ann = np.sqrt(252)
    bull_sharpe = (bull_ret.mean() * 252) / (bull_ret.std() * ann) if len(bull_ret) > 20 and bull_ret.std() > 0 else 0
    bear_sharpe = (bear_ret.mean() * 252) / (bear_ret.std() * ann) if len(bear_ret) > 20 and bear_ret.std() > 0 else 0

    return bull_sharpe, bear_sharpe


def permutation_test(returns, n_perm=N_PERMUTATIONS):
    """Permutation test: shuffle daily returns, compute Sharpe, get p-value."""
    actual_sharpe = returns.mean() / returns.std() if returns.std() > 0 else 0

    rng = np.random.RandomState(42)
    count_ge = 0
    ret_arr = returns.values.copy()
    for _ in range(n_perm):
        rng.shuffle(ret_arr)
        shuf_sharpe = ret_arr.mean() / ret_arr.std() if ret_arr.std() > 0 else 0
        if shuf_sharpe >= actual_sharpe:
            count_ge += 1

    return count_ge / n_perm


def validate_5gate(metrics, bull_sharpe, bear_sharpe, perm_p):
    """Apply 5-gate validation."""
    gates = {}

    # Gate 1: Sharpe > 0.5
    gates['sharpe_gt_0.5'] = metrics['sharpe'] > 0.5

    # Gate 2: Permutation p < 0.05
    gates['perm_p_lt_0.05'] = perm_p < 0.05

    # Gate 3: Regime gap < 0.5
    max_abs = max(abs(bull_sharpe), abs(bear_sharpe), 0.001)
    regime_gap = abs(bull_sharpe - bear_sharpe) / max_abs
    gates['regime_gap_lt_0.5'] = regime_gap < 0.5

    # Gate 4: MaxDD > -50%
    gates['maxdd_gt_neg50'] = metrics['max_dd_pct'] > -50

    # Gate 5: >= 20 trades
    gates['trades_ge_20'] = metrics['n_trades'] >= 20

    passed = all(gates.values())
    return gates, passed, round(regime_gap, 3), round(perm_p, 4)


# ── Strategy Implementations ──────────────────────────────────────────────

# OOT period data
closes_oot = closes.loc[OOT_START:OOT_END].copy()
tickers_active = [t for t in available_tickers if t in closes_oot.columns]

results = {}

# ── Variant A: Dual Moving Average ────────────────────────────────────────
print("\n=== Variant A: Dual Moving Average ===")
sma20 = closes.rolling(20).mean()
sma50 = closes.rolling(50).mean()

# Signal: long if 20-SMA > 50-SMA
signals_a = (sma20 > sma50).astype(float)
signals_a_oot = signals_a.loc[OOT_START:OOT_END, tickers_active]

# Weekly rebalance: only update weights on Fridays
weights_a = signals_a_oot.copy()
is_friday = weights_a.index.dayofweek == 4
# Forward fill between Fridays
for i in range(len(weights_a)):
    if i == 0 or is_friday[i]:
        pass  # keep signal
    else:
        weights_a.iloc[i] = weights_a.iloc[i-1]

# Equal weight among active longs
row_sums = weights_a.sum(axis=1).replace(0, 1)
weights_a = weights_a.div(row_sums, axis=0)

ret_a, eq_a, trades_a = backtest_strategy(weights_a, closes_oot, 'A')
metrics_a = compute_metrics(ret_a, eq_a, trades_a, 'A_DualMA')

# ── Variant B: Breakout Channel ───────────────────────────────────────────
print("=== Variant B: Breakout Channel ===")
high20 = closes.rolling(20).max()
low10 = closes.rolling(10).min()

# State machine: enter when price > 20-day high, exit when price < 10-day low
signals_b = pd.DataFrame(0.0, index=closes.index, columns=tickers_active)
for t in tickers_active:
    in_trade = False
    for i in range(1, len(closes)):
        dt = closes.index[i]
        if dt < pd.Timestamp(OOT_START):
            continue
        price = closes[t].iloc[i]
        prev_high20 = high20[t].iloc[i-1] if not pd.isna(high20[t].iloc[i-1]) else np.inf
        prev_low10 = low10[t].iloc[i-1] if not pd.isna(low10[t].iloc[i-1]) else -np.inf

        if not in_trade:
            if price > prev_high20:
                in_trade = True
        else:
            if price < prev_low10:
                in_trade = False

        signals_b.loc[dt, t] = 1.0 if in_trade else 0.0

signals_b_oot = signals_b.loc[OOT_START:OOT_END]
row_sums_b = signals_b_oot.sum(axis=1).replace(0, 1)
weights_b = signals_b_oot.div(row_sums_b, axis=0)

ret_b, eq_b, trades_b = backtest_strategy(weights_b, closes_oot, 'B')
metrics_b = compute_metrics(ret_b, eq_b, trades_b, 'B_Breakout')

# ── Variant C: Momentum Score Rotation ────────────────────────────────────
print("=== Variant C: Momentum Score Rotation ===")
ret_3m = closes.pct_change(63)
ret_6m = closes.pct_change(126)
ret_12m = closes.pct_change(252)

# Z-score each momentum component cross-sectionally
def zscore_cross(df):
    mu = df.mean(axis=1)
    sigma = df.std(axis=1).replace(0, 1)
    return df.sub(mu, axis=0).div(sigma, axis=0)

z3 = zscore_cross(ret_3m[tickers_active])
z6 = zscore_cross(ret_6m[tickers_active])
z12 = zscore_cross(ret_12m[tickers_active])
mom_score = z3 + z6 + z12

# Monthly rebalance: first trading day of each month
weights_c = pd.DataFrame(0.0, index=closes_oot.index, columns=tickers_active)
mom_score_oot = mom_score.loc[OOT_START:OOT_END]

prev_month = None
current_weights = pd.Series(0.0, index=tickers_active)
for dt in closes_oot.index:
    month_key = (dt.year, dt.month)
    if prev_month is None or month_key != prev_month:
        # Rebalance
        if dt in mom_score_oot.index:
            scores = mom_score_oot.loc[dt]
            # Only positive scores
            positive = scores[scores > 0].sort_values(ascending=False)
            top2 = positive.head(2)
            current_weights = pd.Series(0.0, index=tickers_active)
            if len(top2) > 0:
                for t in top2.index:
                    current_weights[t] = 1.0 / len(top2)
        prev_month = month_key
    weights_c.loc[dt] = current_weights

ret_c, eq_c, trades_c = backtest_strategy(weights_c, closes_oot, 'C')
metrics_c = compute_metrics(ret_c, eq_c, trades_c, 'C_MomRotation')

# ── Variant D: Trend + Vol Target ─────────────────────────────────────────
print("=== Variant D: Trend + Vol Target ===")
TARGET_VOL = 0.08  # 8% annualized
realized_vol_30d = closes.pct_change().rolling(30).std() * np.sqrt(252)

signals_d = (sma20 > sma50).astype(float)
signals_d_oot = signals_d.loc[OOT_START:OOT_END, tickers_active]
vol_oot = realized_vol_30d.loc[OOT_START:OOT_END, tickers_active]

# Vol-scale each position
vol_scaled = signals_d_oot.copy()
for t in tickers_active:
    vol_ratio = TARGET_VOL / vol_oot[t].replace(0, TARGET_VOL)
    vol_ratio = vol_ratio.clip(0, 2.0)  # cap at 200% leverage per position
    vol_scaled[t] = signals_d_oot[t] * vol_ratio

# Weekly rebalance
weights_d = vol_scaled.copy()
is_friday_d = weights_d.index.dayofweek == 4
for i in range(len(weights_d)):
    if i == 0 or is_friday_d[i]:
        pass
    else:
        weights_d.iloc[i] = weights_d.iloc[i-1]

# Normalize so total weight <= 1 (no leverage at portfolio level)
row_sums_d = weights_d.sum(axis=1)
row_sums_d = row_sums_d.where(row_sums_d > 1, 1)  # only scale down if >100%
weights_d = weights_d.div(row_sums_d, axis=0)

ret_d, eq_d, trades_d = backtest_strategy(weights_d, closes_oot, 'D')
metrics_d = compute_metrics(ret_d, eq_d, trades_d, 'D_TrendVolTarget')

# ── Variant E: Cross-Asset Momentum ───────────────────────────────────────
print("=== Variant E: Cross-Asset Momentum ===")
mom_3m = closes.pct_change(63)
vol_30d_current = closes.pct_change().rolling(30).std() * np.sqrt(252)
vol_30d_prior = closes.pct_change().rolling(30).std().shift(21) * np.sqrt(252)  # 1 month ago

weights_e = pd.DataFrame(0.0, index=closes_oot.index, columns=tickers_active)
prev_month_e = None
current_weights_e = pd.Series(0.0, index=tickers_active)

for dt in closes_oot.index:
    month_key = (dt.year, dt.month)
    if prev_month_e is None or month_key != prev_month_e:
        if dt in mom_3m.index:
            eligible = []
            for t in tickers_active:
                m = mom_3m.loc[dt, t] if dt in mom_3m.index else np.nan
                v_now = vol_30d_current.loc[dt, t] if dt in vol_30d_current.index else np.nan
                v_prior = vol_30d_prior.loc[dt, t] if dt in vol_30d_prior.index else np.nan
                if pd.notna(m) and pd.notna(v_now) and pd.notna(v_prior):
                    if m > 0 and v_now < v_prior:  # positive mom + decreasing vol
                        eligible.append((t, m))

            eligible.sort(key=lambda x: x[1], reverse=True)
            top3 = eligible[:3]
            current_weights_e = pd.Series(0.0, index=tickers_active)
            if len(top3) > 0:
                for t, _ in top3:
                    current_weights_e[t] = 1.0 / len(top3)
        prev_month_e = month_key
    weights_e.loc[dt] = current_weights_e

ret_e, eq_e, trades_e = backtest_strategy(weights_e, closes_oot, 'E')
metrics_e = compute_metrics(ret_e, eq_e, trades_e, 'E_CrossAssetMom')

# ── Variant F: Adaptive Trend (SMA + ADX) ─────────────────────────────────
print("=== Variant F: Adaptive Trend ===")
adx_dict = {}
for t in tickers_active:
    h = data[t]['High'] if 'High' in data[t].columns else data[t]['Close'] * 1.005
    l = data[t]['Low'] if 'Low' in data[t].columns else data[t]['Close'] * 0.995
    # Handle multi-index if present
    if isinstance(h, pd.DataFrame):
        h = h.iloc[:, 0]
    if isinstance(l, pd.DataFrame):
        l = l.iloc[:, 0]
    c = closes[t]
    adx_dict[t] = compute_adx(h, l, c, 14)

adx_df = pd.DataFrame(adx_dict)

signals_f = pd.DataFrame(0.0, index=closes.index, columns=tickers_active)
for t in tickers_active:
    sma_cross = (sma20[t] > sma50[t])
    adx_strong = adx_df[t] > 25
    adx_weak = adx_df[t] < 20

    in_position = False
    for i in range(len(closes)):
        dt = closes.index[i]
        if pd.isna(sma_cross.iloc[i]) or pd.isna(adx_df[t].iloc[i]):
            continue
        if not in_position:
            if sma_cross.iloc[i] and adx_strong.iloc[i]:
                in_position = True
        else:
            if adx_weak.iloc[i] or not sma_cross.iloc[i]:
                in_position = False
        signals_f.loc[dt, t] = 1.0 if in_position else 0.0

signals_f_oot = signals_f.loc[OOT_START:OOT_END]
row_sums_f = signals_f_oot.sum(axis=1).replace(0, 1)
weights_f = signals_f_oot.div(row_sums_f, axis=0)

ret_f, eq_f, trades_f = backtest_strategy(weights_f, closes_oot, 'F')
metrics_f = compute_metrics(ret_f, eq_f, trades_f, 'F_AdaptiveTrend')

# ── Validation & Summary ──────────────────────────────────────────────────
print("\n" + "="*90)
print("ETF TREND FOLLOWING BACKTEST — 5-GATE VALIDATION")
print(f"OOT Period: {OOT_START} to {OOT_END} | Capital: ${INITIAL_CAPITAL} | Slippage: {SLIPPAGE_PCT*100:.2f}%")
print("="*90)

all_results = []
variants = [
    ('A', metrics_a, ret_a, eq_a),
    ('B', metrics_b, ret_b, eq_b),
    ('C', metrics_c, ret_c, eq_c),
    ('D', metrics_d, ret_d, eq_d),
    ('E', metrics_e, ret_e, eq_e),
    ('F', metrics_f, ret_f, eq_f),
]

spy_close_oot = spy_close.loc[OOT_START:OOT_END]
spy_sma200_oot = spy_sma200.loc[OOT_START:OOT_END]

for label, metrics, ret, eq in variants:
    if metrics is None:
        print(f"\n  Variant {label}: INSUFFICIENT DATA")
        continue

    bull_s, bear_s = regime_analysis(ret, spy_close_oot, spy_sma200_oot)
    perm_p = permutation_test(ret)
    gates, passed, regime_gap, perm_pval = validate_5gate(metrics, bull_s, bear_s, perm_p)

    result = {
        **metrics,
        'bull_sharpe': round(bull_s, 3),
        'bear_sharpe': round(bear_s, 3),
        'regime_gap': regime_gap,
        'perm_p_value': perm_pval,
        'gates': gates,
        'passed_all_gates': passed,
    }
    all_results.append(result)

    status = "PASS" if passed else "FAIL"
    print(f"\n  Variant {metrics['name']}:")
    print(f"    Return: {metrics['total_return_pct']:+.1f}% | CAGR: {metrics['cagr_pct']:+.1f}% | Final: ${metrics['final_equity']:.2f}")
    print(f"    Sharpe: {metrics['sharpe']:.3f} | Sortino: {metrics['sortino']:.3f} | PF: {metrics['profit_factor']:.2f} | WR: {metrics['win_rate_pct']:.1f}%")
    print(f"    MaxDD: {metrics['max_dd_pct']:.1f}% | Vol: {metrics['volatility_pct']:.1f}% | Trades: {metrics['n_trades']}")
    print(f"    Bull Sharpe: {bull_s:.3f} | Bear Sharpe: {bear_s:.3f} | Regime Gap: {regime_gap:.3f}")
    print(f"    Perm p-value: {perm_pval:.4f}")
    print(f"    Gates: {' | '.join(f'{k}={v}' for k,v in gates.items())}")
    print(f"    >>> 5-GATE: [{status}] <<<")

# Summary table
print("\n" + "="*90)
print("SUMMARY LEADERBOARD (sorted by Sharpe)")
print("="*90)
sorted_results = sorted(all_results, key=lambda x: x['sharpe'], reverse=True)
print(f"{'Variant':<22} {'Sharpe':>7} {'Sortino':>8} {'CAGR%':>7} {'MaxDD%':>8} {'PF':>6} {'WR%':>6} {'Trades':>7} {'5-Gate':>7}")
print("-"*90)
for r in sorted_results:
    status = "PASS" if r['passed_all_gates'] else "FAIL"
    print(f"{r['name']:<22} {r['sharpe']:>7.3f} {r['sortino']:>8.3f} {r['cagr_pct']:>+7.1f} {r['max_dd_pct']:>8.1f} {r['profit_factor']:>6.2f} {r['win_rate_pct']:>6.1f} {r['n_trades']:>7d} {status:>7}")

# Benchmark: SPY buy & hold
spy_ret_oot = spy_close.loc[OOT_START:OOT_END].pct_change().dropna()
spy_eq = INITIAL_CAPITAL * (1 + spy_ret_oot).cumprod()
spy_metrics = compute_metrics(spy_ret_oot, spy_eq, 1, 'SPY_BuyHold')
if spy_metrics:
    print(f"\n  Benchmark SPY B&H: Sharpe={spy_metrics['sharpe']:.3f} | CAGR={spy_metrics['cagr_pct']:+.1f}% | MaxDD={spy_metrics['max_dd_pct']:.1f}% | Final=${spy_metrics['final_equity']:.2f}")

passed_count = sum(1 for r in all_results if r['passed_all_gates'])
print(f"\n  {passed_count}/{len(all_results)} variants passed all 5 gates.")

# ── Save Results ───────────────────────────────────────────────────────────
output = {
    'metadata': {
        'strategy': 'ETF Trend Following (Managed Futures Style)',
        'reference': 'Moskowitz, Ooi & Pedersen (2012)',
        'universe': available_tickers,
        'oot_period': f'{OOT_START} to {OOT_END}',
        'initial_capital': INITIAL_CAPITAL,
        'slippage_pct': SLIPPAGE_PCT,
        'n_permutations': N_PERMUTATIONS,
        'run_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    },
    'benchmark': spy_metrics,
    'variants': all_results,
    'passed_variants': [r['name'] for r in all_results if r['passed_all_gates']],
}

output_path = Path('/home/jupiter/Lvl3Quant/data/etf_trend_following_results.json')
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"\nResults saved to {output_path}")
print("Done.")

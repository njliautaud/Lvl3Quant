#!/usr/bin/env python3
"""
Multi-Factor Stock Ranking Backtest
====================================
Ranks growth/tech stocks on 5 factors (momentum, quality proxy, relative value,
volatility, volume trend) and buys the highest-ranked. Walk-forward OOT: Jan 2022 – Jul 2026.
6 variants tested. 5-gate validation applied.

Author: Claude Opus 4.6
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────────
UNIVERSE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD', 'NFLX', 'CRM',
    'PLTR', 'SOFI', 'HOOD', 'SNAP', 'UBER', 'COIN', 'PINS', 'NET', 'DDOG', 'TTD'
]
CAPITAL = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02%
OOT_START = '2022-01-01'
OOT_END = '2026-07-30'
DATA_START = '2021-01-01'  # extra lookback for indicators
PERM_ITERATIONS = 500
RSI_ENTRY_THRESHOLD = 40  # for variant F

# ── Data Download ─────────────────────────────────────────────────────────────
print("Downloading price data...")
all_tickers = UNIVERSE + ['SPY']
data = yf.download(all_tickers, start=DATA_START, end=OOT_END, auto_adjust=True, progress=False)

close = data['Close'][UNIVERSE].copy()
volume = data['Volume'][UNIVERSE].copy()
spy_close = data['Close']['SPY'].copy()

close = close.ffill()
volume = volume.ffill()
spy_close = spy_close.ffill()

spy_sma200 = spy_close.rolling(200).mean()

print(f"Data range: {close.index[0].date()} to {close.index[-1].date()}")
print(f"Tickers with data: {close.columns.tolist()}")

# ── Get Monthly Rebalance Dates ──────────────────────────────────────────────
oot_mask = close.index >= OOT_START
oot_dates = close.index[oot_mask]
rebal_dates = []
prev_month = None
for d in oot_dates:
    ym = (d.year, d.month)
    if ym != prev_month:
        rebal_dates.append(d)
        prev_month = ym
print(f"Rebalance dates: {len(rebal_dates)} months")

# ── Precompute All Factors at Each Rebalance Date ─────────────────────────────
print("Precomputing factors at each rebalance date...")

def compute_rsi(series, period=5):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Precompute RSI for all tickers
rsi_data = {}
for tk in UNIVERSE:
    rsi_data[tk] = compute_rsi(close[tk], period=5)

# Precompute factor ranks at each rebalance date
precomputed = {}  # date -> {'ranks': DataFrame, 'valid_tickers': list, 'regime': str, 'rsi': dict}
for rd in rebal_dates:
    idx = close.index.get_loc(rd)
    if idx < 252:
        continue

    lookback_close = close.iloc[idx-252:idx+1]
    lookback_vol = volume.iloc[idx-60:idx+1]

    factors = {}
    valid_tickers = []

    for ticker in UNIVERSE:
        tc = lookback_close[ticker].dropna()
        tv = lookback_vol[ticker].dropna()

        if len(tc) < 253 or len(tv) < 61:
            continue

        ret_60d = tc.iloc[-1] / tc.iloc[-60] - 1.0

        quarterly_prices = [tc.iloc[-1], tc.iloc[-63], tc.iloc[-126], tc.iloc[-189], tc.iloc[-252]]
        quarterly_returns = [quarterly_prices[i]/quarterly_prices[i+1] - 1.0 for i in range(4)]
        quality = np.std(quarterly_returns)

        high_52w = tc.max()
        dist_from_high = 1.0 - tc.iloc[-1] / high_52w

        daily_rets = tc.pct_change().dropna().iloc[-20:]
        vol_20d = daily_rets.std() * np.sqrt(252)

        vol_20 = tv.iloc[-20:].mean()
        vol_60 = tv.mean()
        vol_trend = vol_20 / vol_60 if vol_60 > 0 else 1.0

        factors[ticker] = {
            'momentum': ret_60d,
            'quality': quality,
            'rel_value': dist_from_high,
            'volatility': vol_20d,
            'vol_trend': vol_trend,
        }
        valid_tickers.append(ticker)

    if len(valid_tickers) < 3:
        continue

    df = pd.DataFrame(factors).T
    ranks = pd.DataFrame(index=valid_tickers)
    ranks['momentum'] = df['momentum'].rank(ascending=False)
    ranks['quality'] = df['quality'].rank(ascending=True)
    ranks['rel_value'] = df['rel_value'].rank(ascending=False)
    ranks['volatility'] = df['volatility'].rank(ascending=True)
    ranks['vol_trend'] = df['vol_trend'].rank(ascending=False)

    # Regime
    spy_val = spy_close.loc[:rd].iloc[-1]
    sma_val = spy_sma200.loc[:rd].iloc[-1]
    regime = 'bull' if (pd.notna(sma_val) and spy_val > sma_val) else 'bear'

    # RSI values
    rsi_vals = {}
    for tk in valid_tickers:
        rv = rsi_data[tk].loc[:rd].iloc[-1]
        rsi_vals[tk] = float(rv) if pd.notna(rv) else 50.0

    precomputed[rd] = {
        'ranks': ranks,
        'valid_tickers': valid_tickers,
        'regime': regime,
        'rsi': rsi_vals,
    }

print(f"Precomputed factors for {len(precomputed)} rebalance dates")

# ── Precompute Monthly Returns for Each Ticker ──────────────────────────────
# For fast permutation testing, precompute the return of each ticker over each period
monthly_returns = {}  # (i, ticker) -> return over period i
for i in range(len(rebal_dates) - 1):
    rd = rebal_dates[i]
    next_rd = rebal_dates[i + 1]
    for tk in UNIVERSE:
        p_start = close.loc[rd, tk]
        p_end = close.loc[next_rd, tk] if next_rd in close.index else close.iloc[-1][tk]
        if pd.notna(p_start) and pd.notna(p_end) and p_start > 0:
            monthly_returns[(i, tk)] = (p_end / p_start) - 1.0
        else:
            monthly_returns[(i, tk)] = 0.0


# ── Fast Backtest Using Precomputed Data ──────────────────────────────────────
def get_composite_scores(ranks, weights):
    score = pd.Series(0.0, index=ranks.index)
    for factor, w in weights.items():
        score += ranks[factor] * w
    return score


def select_stocks(period_idx, variant_cfg, randomize=False, rng=None):
    """Select stocks for a given period. Returns list of tickers."""
    rd = rebal_dates[period_idx]
    if rd not in precomputed:
        return []

    pc = precomputed[rd]
    n_pos = variant_cfg.get('n_positions', 3)

    if randomize and rng is not None:
        valid = pc['valid_tickers']
        if len(valid) >= n_pos:
            return list(rng.choice(valid, size=n_pos, replace=False))
        return valid[:]

    ranks = pc['ranks']

    if variant_cfg.get('regime_adaptive'):
        if pc['regime'] == 'bull':
            weights = variant_cfg['weights_bull']
        else:
            weights = variant_cfg['weights_bear']
    else:
        weights = variant_cfg['weights']

    scores = get_composite_scores(ranks, weights)
    selected = scores.nsmallest(n_pos).index.tolist()

    if variant_cfg.get('rsi_entry'):
        filtered = [tk for tk in selected if pc['rsi'].get(tk, 50) < RSI_ENTRY_THRESHOLD]
        return filtered

    return selected


def run_backtest_fast(variant_cfg, randomize=False, rng=None):
    """
    Fast backtest using precomputed data.
    Returns portfolio values, dates, and trades.
    """
    port_values = [CAPITAL]
    port_dates = [rebal_dates[0]]
    trades = []
    holdings = {}
    cash = CAPITAL

    for i in range(len(rebal_dates) - 1):
        rd = rebal_dates[i]
        next_rd = rebal_dates[i + 1]
        selected = select_stocks(i, variant_cfg, randomize=randomize, rng=rng)

        # Liquidate
        for tk, h in holdings.items():
            sell_price = close.loc[rd, tk] * (1 - SLIPPAGE_PCT)
            cash += h['shares'] * sell_price
            if not randomize:
                trades.append({
                    'date': str(rd.date()), 'ticker': tk, 'action': 'SELL',
                    'price': float(sell_price), 'shares': float(h['shares']),
                })
        holdings = {}

        # Buy
        if selected:
            pos_size = cash / len(selected)
            for tk in selected:
                buy_price = close.loc[rd, tk] * (1 + SLIPPAGE_PCT)
                shares = pos_size / buy_price
                holdings[tk] = {'shares': shares, 'entry_price': buy_price}
                cash -= shares * buy_price
                if not randomize:
                    trades.append({
                        'date': str(rd.date()), 'ticker': tk, 'action': 'BUY',
                        'price': float(buy_price), 'shares': float(shares),
                    })

        # End-of-period value
        port_val = cash
        for tk, h in holdings.items():
            port_val += h['shares'] * close.loc[next_rd, tk]
        port_values.append(port_val)
        port_dates.append(next_rd)

    return port_values, port_dates, trades


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(port_values, port_dates):
    if len(port_values) < 3:
        return None

    pv = np.array(port_values, dtype=float)
    monthly_rets = np.diff(pv) / pv[:-1]

    total_ret = pv[-1] / pv[0] - 1.0
    n_months = len(monthly_rets)
    n_years = n_months / 12.0

    ann_ret = (1 + total_ret) ** (1.0 / max(n_years, 0.01)) - 1.0
    ann_vol = np.std(monthly_rets) * np.sqrt(12)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

    downside = monthly_rets[monthly_rets < 0]
    down_vol = np.std(downside) * np.sqrt(12) if len(downside) > 1 else 1e-6
    sortino = ann_ret / down_vol

    peak = np.maximum.accumulate(pv)
    dd = (pv - peak) / peak
    max_dd = float(dd.min())

    gains = monthly_rets[monthly_rets > 0].sum()
    losses = abs(monthly_rets[monthly_rets < 0].sum())
    pf = gains / losses if losses > 0 else float('inf')

    wr = np.mean(monthly_rets > 0) if len(monthly_rets) > 0 else 0

    # Regime analysis
    regime_rets = {'bull': [], 'bear': []}
    for i, d in enumerate(port_dates[1:]):
        spy_val = spy_close.loc[:d].iloc[-1]
        sma_val = spy_sma200.loc[:d].iloc[-1]
        regime = 'bull' if (pd.notna(sma_val) and spy_val > sma_val) else 'bear'
        regime_rets[regime].append(monthly_rets[i])

    bull_sharpe = bear_sharpe = 0.0
    if len(regime_rets['bull']) > 1:
        br = np.array(regime_rets['bull'])
        bull_sharpe = (np.mean(br) * 12) / (np.std(br) * np.sqrt(12)) if np.std(br) > 0 else 0
    if len(regime_rets['bear']) > 1:
        br = np.array(regime_rets['bear'])
        bear_sharpe = (np.mean(br) * 12) / (np.std(br) * np.sqrt(12)) if np.std(br) > 0 else 0

    regime_gap = abs(bull_sharpe - bear_sharpe) / max(abs(bull_sharpe), abs(bear_sharpe), 1e-6)

    return {
        'total_return_pct': round(total_ret * 100, 2),
        'ann_return_pct': round(ann_ret * 100, 2),
        'ann_volatility_pct': round(ann_vol * 100, 2),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'profit_factor': round(pf, 3),
        'win_rate': round(float(wr), 3),
        'max_drawdown_pct': round(max_dd * 100, 2),
        'n_months': n_months,
        'final_value': round(float(pv[-1]), 2),
        'bull_sharpe': round(bull_sharpe, 3),
        'bear_sharpe': round(bear_sharpe, 3),
        'regime_gap': round(regime_gap, 3),
    }


def permutation_test(actual_sharpe, variant_cfg, n_perm=PERM_ITERATIONS):
    """Shuffle stock selections randomly, compute p-value."""
    rng = np.random.default_rng(42)
    count_better = 0

    for _ in range(n_perm):
        pv, pd_dates, _ = run_backtest_fast(variant_cfg, randomize=True, rng=rng)
        if len(pv) < 3:
            continue
        perm_metrics = compute_metrics(pv, pd_dates)
        if perm_metrics and perm_metrics['sharpe'] >= actual_sharpe:
            count_better += 1

    return (count_better + 1) / (n_perm + 1)


def validate_5gate(metrics, perm_p, n_trades):
    gates = {
        'sharpe_gt_0.5': metrics['sharpe'] > 0.5,
        'perm_p_lt_0.05': perm_p < 0.05,
        'regime_gap_lt_0.5': metrics['regime_gap'] < 0.5,
        'max_dd_gt_neg50': metrics['max_drawdown_pct'] > -50.0,
        'trades_gte_20': n_trades >= 20,
    }
    gates['all_pass'] = all(gates.values())
    return gates


# ── Variant Definitions ──────────────────────────────────────────────────────
VARIANTS = {
    'A_EqualWeight': {
        'description': 'Equal-Weight 5 Factors, top-3 monthly',
        'n_positions': 3,
        'weights': {'momentum': 0.2, 'quality': 0.2, 'rel_value': 0.2, 'volatility': 0.2, 'vol_trend': 0.2},
        'regime_adaptive': False, 'rsi_entry': False, 'concentrated': False,
    },
    'B_MomentumHeavy': {
        'description': 'Momentum-Heavy (40%), top-3 monthly',
        'n_positions': 3,
        'weights': {'momentum': 0.4, 'quality': 0.15, 'rel_value': 0.15, 'volatility': 0.15, 'vol_trend': 0.15},
        'regime_adaptive': False, 'rsi_entry': False, 'concentrated': False,
    },
    'C_QualityHeavy': {
        'description': 'Quality-Heavy (40%), top-3 monthly',
        'n_positions': 3,
        'weights': {'momentum': 0.15, 'quality': 0.4, 'rel_value': 0.15, 'volatility': 0.15, 'vol_trend': 0.15},
        'regime_adaptive': False, 'rsi_entry': False, 'concentrated': False,
    },
    'D_Top1Concentrated': {
        'description': 'Top-1 concentrated, full position monthly',
        'n_positions': 1,
        'weights': {'momentum': 0.2, 'quality': 0.2, 'rel_value': 0.2, 'volatility': 0.2, 'vol_trend': 0.2},
        'regime_adaptive': False, 'rsi_entry': False, 'concentrated': True,
    },
    'E_RegimeAdaptive': {
        'description': 'Regime-adaptive weights, top-3 monthly',
        'n_positions': 3,
        'weights_bull': {'momentum': 0.50, 'quality': 0.10, 'rel_value': 0.10, 'volatility': 0.15, 'vol_trend': 0.15},
        'weights_bear': {'momentum': 0.10, 'quality': 0.25, 'rel_value': 0.15, 'volatility': 0.25, 'vol_trend': 0.25},
        'weights': {'momentum': 0.2, 'quality': 0.2, 'rel_value': 0.2, 'volatility': 0.2, 'vol_trend': 0.2},
        'regime_adaptive': True, 'rsi_entry': False, 'concentrated': False,
    },
    'F_RSIEntry': {
        'description': 'Top-3 by composite, RSI(5)<40 entry filter',
        'n_positions': 3,
        'weights': {'momentum': 0.2, 'quality': 0.2, 'rel_value': 0.2, 'volatility': 0.2, 'vol_trend': 0.2},
        'regime_adaptive': False, 'rsi_entry': True, 'concentrated': False,
    },
}


# ── Run All Variants ──────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("MULTI-FACTOR STOCK RANKING BACKTEST")
print(f"OOT Period: {OOT_START} to {OOT_END}")
print(f"Universe: {len(UNIVERSE)} growth/tech stocks")
print(f"Capital: ${CAPITAL}")
print("=" * 70)

results = {}

for name, cfg in VARIANTS.items():
    print(f"\n{'─' * 50}")
    print(f"Running Variant {name}: {cfg['description']}")
    print(f"{'─' * 50}")

    port_values, port_dates, trades = run_backtest_fast(cfg)

    metrics = compute_metrics(port_values, port_dates)
    if metrics is None:
        print(f"  SKIP — insufficient data")
        results[name] = {'status': 'SKIP', 'reason': 'insufficient data'}
        continue

    n_trades = len([t for t in trades if t['action'] == 'BUY'])

    print(f"  Total Return: {metrics['total_return_pct']:.1f}%")
    print(f"  Ann Return:   {metrics['ann_return_pct']:.1f}%")
    print(f"  Sharpe:       {metrics['sharpe']:.3f}")
    print(f"  Sortino:      {metrics['sortino']:.3f}")
    print(f"  Profit Factor:{metrics['profit_factor']:.3f}")
    print(f"  Win Rate:     {metrics['win_rate']:.1%}")
    print(f"  Max DD:       {metrics['max_drawdown_pct']:.1f}%")
    print(f"  Final Value:  ${metrics['final_value']:.2f}")
    print(f"  Trades (buys):{n_trades}")
    print(f"  Bull Sharpe:  {metrics['bull_sharpe']:.3f}  Bear Sharpe: {metrics['bear_sharpe']:.3f}")
    print(f"  Regime Gap:   {metrics['regime_gap']:.3f}")

    # Permutation test
    print(f"  Running permutation test ({PERM_ITERATIONS} iterations)...", flush=True)
    perm_p = permutation_test(metrics['sharpe'], cfg)
    print(f"  Perm p-value: {perm_p:.4f}")

    # 5-gate validation
    gates = validate_5gate(metrics, perm_p, n_trades)
    print(f"  5-Gate Validation:")
    for gate, passed in gates.items():
        status = 'PASS' if passed else 'FAIL'
        print(f"    {gate}: {status}")

    verdict = 'PASS' if gates['all_pass'] else 'FAIL'
    print(f"  VERDICT: {verdict}")

    results[name] = {
        'description': cfg['description'],
        'metrics': metrics,
        'n_trades': n_trades,
        'perm_p_value': round(perm_p, 4),
        'gates': {k: bool(v) for k, v in gates.items()},
        'verdict': verdict,
        'sample_trades': trades[:10],
        'equity_curve': {
            'dates': [str(d.date()) for d in port_dates],
            'values': [round(float(v), 2) for v in port_values],
        },
    }

# ── SPY Benchmark ─────────────────────────────────────────────────────────────
print(f"\n{'─' * 50}")
print("SPY Buy & Hold Benchmark")
print(f"{'─' * 50}")

spy_oot = spy_close.loc[OOT_START:]
spy_ret = spy_oot.iloc[-1] / spy_oot.iloc[0] - 1.0
spy_years = len(spy_oot) / 252.0
spy_ann_ret = (1 + spy_ret) ** (1 / spy_years) - 1.0
spy_daily_rets = spy_oot.pct_change().dropna()
spy_ann_vol = spy_daily_rets.std() * np.sqrt(252)
spy_sharpe = spy_ann_ret / spy_ann_vol if spy_ann_vol > 0 else 0
spy_peak = np.maximum.accumulate(spy_oot.values)
spy_dd = ((spy_oot.values - spy_peak) / spy_peak).min()

print(f"  Total Return: {spy_ret*100:.1f}%")
print(f"  Ann Return:   {spy_ann_ret*100:.1f}%")
print(f"  Sharpe:       {spy_sharpe:.3f}")
print(f"  Max DD:       {spy_dd*100:.1f}%")

results['SPY_Benchmark'] = {
    'total_return_pct': round(float(spy_ret * 100), 2),
    'ann_return_pct': round(float(spy_ann_ret * 100), 2),
    'sharpe': round(float(spy_sharpe), 3),
    'max_drawdown_pct': round(float(spy_dd * 100), 2),
}

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("SUMMARY")
print(f"{'=' * 70}")
print(f"{'Variant':<25} {'Sharpe':>8} {'Sortino':>8} {'Return':>8} {'MaxDD':>8} {'WR':>6} {'PF':>6} {'Verdict':>8}")
print(f"{'─'*25} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*6} {'─'*6} {'─'*8}")

for name in VARIANTS:
    r = results.get(name, {})
    if r.get('status') == 'SKIP':
        print(f"{name:<25} {'SKIP':>8}")
        continue
    m = r['metrics']
    print(f"{name:<25} {m['sharpe']:>8.3f} {m['sortino']:>8.3f} {m['total_return_pct']:>7.1f}% {m['max_drawdown_pct']:>7.1f}% {m['win_rate']:>5.1%} {m['profit_factor']:>6.2f} {r['verdict']:>8}")

bm = results['SPY_Benchmark']
print(f"{'SPY B&H':<25} {bm['sharpe']:>8.3f} {'N/A':>8} {bm['total_return_pct']:>7.1f}% {bm['max_drawdown_pct']:>7.1f}%")

# Identify best variant
passing = {k: v for k, v in results.items() if k != 'SPY_Benchmark' and v.get('verdict') == 'PASS'}
if passing:
    best = max(passing, key=lambda k: passing[k]['metrics']['sharpe'])
    print(f"\nBest passing variant: {best} (Sharpe={passing[best]['metrics']['sharpe']:.3f})")
else:
    print("\nNo variant passed all 5 gates.")

# ── Save Results ──────────────────────────────────────────────────────────────
output = {
    'metadata': {
        'strategy': 'Multi-Factor Stock Ranking',
        'oot_period': f"{OOT_START} to {OOT_END}",
        'universe': UNIVERSE,
        'capital': CAPITAL,
        'slippage_pct': SLIPPAGE_PCT,
        'perm_iterations': PERM_ITERATIONS,
        'run_date': str(datetime.now()),
        'factors': ['momentum_60d', 'quality_proxy_quarterly_std', 'relative_value_52w_high',
                     'volatility_20d', 'volume_trend_20d_vs_60d'],
    },
    'variants': results,
}

output_path = '/home/jupiter/Lvl3Quant/data/multifactor_ranking_results.json'
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"\nResults saved to {output_path}")
print("Done.")

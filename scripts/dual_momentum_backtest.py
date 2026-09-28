#!/usr/bin/env python3
"""
Dual Momentum Backtest - Gary Antonacci (2014)
===============================================
Combines absolute momentum (trend following) with relative momentum (cross-asset).

Variants:
  A: Classic — SPY vs EFA, AGG safe haven, 12-month lookback
  B: Modified — SPY vs EFA, BIL (T-bills) safe haven, 12-month lookback
  C: Short lookback — SPY vs EFA, AGG safe haven, 6-month lookback
  D: Triple — SPY vs EFA vs QQQ, AGG safe haven, 12-month lookback
  E: Bear filter — Classic + half-size when SPY < 200-SMA
  F: Aggressive — SPY vs QQQ (no intl), AGG safe haven, 12-month lookback

OOT: Jan 2022 – Jul 2026 | Starting capital: $645
Cost: $0 commission, 0.02% slippage per trade
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')
np.random.seed(42)

# ── Config ──────────────────────────────────────────────────────────────────
START_DATE = '2005-01-01'  # enough history for 12-month lookback before OOT
END_DATE = '2026-07-29'
OOT_START = '2022-01-01'
INITIAL_CAPITAL = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02% per trade (one-way)
N_PERMUTATIONS = 100
RESULTS_PATH = Path('/home/jupiter/Lvl3Quant/data/dual_momentum_results.json')

TICKERS = ['SPY', 'QQQ', 'EFA', 'AGG', 'BND', 'BIL']

# ── Data Download ───────────────────────────────────────────────────────────
print("Downloading data...")
raw = yf.download(TICKERS, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)

if isinstance(raw.columns, pd.MultiIndex):
    prices = raw['Close'].copy()
else:
    prices = raw.copy()

prices = prices.ffill().dropna(subset=['SPY'])
print(f"Data: {prices.shape[0]} days, {prices.index[0].date()} to {prices.index[-1].date()}")

# ── Helpers ─────────────────────────────────────────────────────────────────

def momentum_score(price_series, months):
    """Total return over N months using month-end prices."""
    monthly = price_series.resample('ME').last()
    return monthly / monthly.shift(months) - 1

def sma_200_daily(price_series):
    """200-day simple moving average."""
    return price_series.rolling(200).mean()

def get_monthly_rebal_dates(index, oot_start):
    """First trading day of each month in OOT period."""
    oot = index[index >= pd.Timestamp(oot_start)]
    return oot.to_series().groupby([oot.year, oot.month]).first().values

def regime_daily(spy_prices):
    """Green/Red/Flat based on SPY daily return."""
    ret = spy_prices.pct_change()
    regime = pd.Series('flat', index=spy_prices.index)
    regime[ret > 0.001] = 'green'
    regime[ret < -0.001] = 'red'
    return regime

# ── Variant Definitions ────────────────────────────────────────────────────

VARIANTS = {
    'A_Classic': {
        'description': 'Classic Dual Momentum: SPY vs EFA, AGG safe haven, 12-month lookback',
        'equity_tickers': ['SPY', 'EFA'],
        'safe_haven': 'AGG',
        'lookback_months': 12,
        'bear_filter': False,
    },
    'B_Modified': {
        'description': 'Modified: SPY vs EFA, BIL (T-bills) safe haven, 12-month lookback',
        'equity_tickers': ['SPY', 'EFA'],
        'safe_haven': 'BIL',
        'lookback_months': 12,
        'bear_filter': False,
    },
    'C_Short_Lookback': {
        'description': 'Short lookback: SPY vs EFA, AGG safe haven, 6-month lookback',
        'equity_tickers': ['SPY', 'EFA'],
        'safe_haven': 'AGG',
        'lookback_months': 6,
        'bear_filter': False,
    },
    'D_Triple': {
        'description': 'Triple Momentum: SPY vs EFA vs QQQ, AGG safe haven, 12-month lookback',
        'equity_tickers': ['SPY', 'EFA', 'QQQ'],
        'safe_haven': 'AGG',
        'lookback_months': 12,
        'bear_filter': False,
    },
    'E_Bear_Filter': {
        'description': 'Bear filter: Classic + half-size when SPY < 200-SMA',
        'equity_tickers': ['SPY', 'EFA'],
        'safe_haven': 'AGG',
        'lookback_months': 12,
        'bear_filter': True,
    },
    'F_Aggressive': {
        'description': 'Aggressive: SPY vs QQQ (no intl), AGG safe haven, 12-month lookback',
        'equity_tickers': ['SPY', 'QQQ'],
        'safe_haven': 'AGG',
        'lookback_months': 12,
        'bear_filter': False,
    },
}

# ── Core Backtest Engine ────────────────────────────────────────────────────

def run_variant(name, config):
    """
    Run dual momentum backtest for one variant.
    Returns monthly-level equity curve and trade log.
    """
    eq_tickers = config['equity_tickers']
    safe_haven = config['safe_haven']
    lb = config['lookback_months']
    bear_filter = config['bear_filter']

    # Compute momentum scores (monthly)
    mom = {t: momentum_score(prices[t], lb) for t in eq_tickers}

    # Monthly prices and returns for all relevant tickers
    all_tickers = list(set(eq_tickers + [safe_haven]))
    monthly_prices = {t: prices[t].resample('ME').last() for t in all_tickers}
    monthly_rets = {t: monthly_prices[t].pct_change() for t in all_tickers}

    # SPY 200-SMA for bear filter (use monthly-end check against daily SMA)
    spy_sma = sma_200_daily(prices['SPY'])

    # Common index (intersection of all momentum indices)
    idx = mom[eq_tickers[0]].index

    # Strategy signals
    held_asset = pd.Series('', index=idx)
    position_size = pd.Series(1.0, index=idx)
    strat_returns = pd.Series(0.0, index=idx)

    for i in range(1, len(idx)):
        dt = idx[i]
        prev_dt = idx[i - 1]

        # Step 1: Relative momentum — pick best equity by lookback return
        eq_scores = {}
        all_valid = True
        for t in eq_tickers:
            val = mom[t].iloc[i - 1]
            if pd.isna(val):
                all_valid = False
                break
            eq_scores[t] = val

        if not all_valid:
            held_asset.iloc[i] = safe_haven
            if dt in monthly_rets[safe_haven].index:
                strat_returns.iloc[i] = monthly_rets[safe_haven].loc[dt]
            continue

        best_eq = max(eq_scores, key=eq_scores.get)

        # Step 2: Absolute momentum — is best equity positive?
        if eq_scores[best_eq] > 0:
            selected = best_eq
        else:
            selected = safe_haven

        held_asset.iloc[i] = selected

        # Step 3: Position sizing (bear filter)
        if bear_filter:
            # Check if SPY is below 200-SMA at month-end of previous month
            # Find the last trading day of previous month
            prev_month_days = prices.index[(prices.index.year == prev_dt.year) &
                                            (prices.index.month == prev_dt.month)]
            if len(prev_month_days) > 0:
                last_day = prev_month_days[-1]
                if prices['SPY'].loc[last_day] < spy_sma.loc[last_day]:
                    position_size.iloc[i] = 0.5

        # Monthly return for selected asset
        if dt in monthly_rets[selected].index:
            ret = monthly_rets[selected].loc[dt]
            strat_returns.iloc[i] = ret * position_size.iloc[i]

    # Filter to OOT
    oot_mask = idx >= pd.Timestamp(OOT_START)
    strat_oot = strat_returns[oot_mask].copy()
    held_oot = held_asset[oot_mask].copy()
    size_oot = position_size[oot_mask].copy()

    if len(strat_oot) < 5:
        return None

    # Apply slippage on trade months (asset change)
    trade_months = held_oot != held_oot.shift(1)
    # First month is always a "trade" (initial purchase)
    trade_months.iloc[0] = True
    # 2x slippage per trade (buy + sell)
    strat_oot[trade_months] -= 2 * SLIPPAGE_PCT
    n_trades = int(trade_months.sum())

    # Build equity curve
    equity = INITIAL_CAPITAL * (1 + strat_oot).cumprod()

    # Trade log
    trade_log = []
    current_asset = None
    entry_month = None
    for i, dt in enumerate(held_oot.index):
        asset = held_oot.iloc[i]
        if asset != current_asset:
            if current_asset is not None:
                trade_log.append({
                    'entry': str(entry_month.date()) if hasattr(entry_month, 'date') else str(entry_month),
                    'exit': str(dt.date()) if hasattr(dt, 'date') else str(dt),
                    'asset': current_asset,
                })
            current_asset = asset
            entry_month = dt
    # Close final
    if current_asset:
        trade_log.append({
            'entry': str(entry_month.date()) if hasattr(entry_month, 'date') else str(entry_month),
            'exit': str(held_oot.index[-1].date()),
            'asset': current_asset,
        })

    return {
        'equity': equity,
        'monthly_rets': strat_oot,
        'held_assets': held_oot,
        'n_trades': n_trades,
        'trade_log': trade_log,
    }


def compute_metrics(result):
    """Compute Sharpe, Sortino, MaxDD, PF, WR."""
    eq = result['equity']
    mr = result['monthly_rets'].dropna()
    n_trades = result['n_trades']

    total_ret = eq.iloc[-1] / eq.iloc[0] - 1
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (1 + total_ret) ** (1 / max(years, 0.01)) - 1

    # Sharpe (annualized from monthly, excess over 0)
    sharpe = (mr.mean() / mr.std() * np.sqrt(12)) if mr.std() > 0 else 0

    # Sortino
    downside = mr[mr < 0]
    ds_std = downside.std() if len(downside) > 1 else 1e-9
    sortino = (mr.mean() / ds_std * np.sqrt(12)) if ds_std > 0 else 0

    # Max drawdown
    peak = eq.cummax()
    dd = (eq - peak) / peak
    mdd = dd.min()

    # Win rate (monthly)
    wr = (mr > 0).sum() / len(mr) if len(mr) > 0 else 0

    # Profit factor
    pos = mr[mr > 0].sum()
    neg = abs(mr[mr < 0].sum())
    pf = pos / neg if neg > 0 else float('inf')

    return {
        'total_return': round(float(total_ret), 4),
        'cagr': round(float(cagr), 4),
        'sharpe': round(float(sharpe), 4),
        'sortino': round(float(sortino), 4),
        'max_drawdown': round(float(mdd), 4),
        'win_rate': round(float(wr), 4),
        'profit_factor': round(float(pf), 4),
        'n_trades': n_trades,
        'final_equity': round(float(eq.iloc[-1]), 2),
        'years': round(float(years), 2),
    }


def regime_analysis(result):
    """Regime-stratified Sharpe using SPY daily returns mapped to months."""
    mr = result['monthly_rets'].dropna()

    # Classify each month by majority of SPY green/red days
    daily_reg = regime_daily(prices['SPY'])
    monthly_reg = daily_reg.resample('ME').apply(
        lambda x: x.mode()[0] if len(x) > 0 else 'flat'
    )
    mr_reg = monthly_reg.reindex(mr.index, method='ffill')

    out = {}
    for regime in ['green', 'red', 'flat']:
        sub = mr[mr_reg == regime]
        if len(sub) > 2 and sub.std() > 0:
            out[f'sharpe_{regime}'] = round(float(sub.mean() / sub.std() * np.sqrt(12)), 4)
        else:
            out[f'sharpe_{regime}'] = 0.0
        out[f'{regime}_months'] = int(len(sub))

    g = abs(out['sharpe_green'])
    r = abs(out['sharpe_red'])
    denom = max(g, r, 1e-10)
    out['regime_gap'] = round(abs(out['sharpe_green'] - out['sharpe_red']) / denom, 4)
    return out


def permutation_test(result, config):
    """
    Shuffle which asset is held each month (random choice from equity + safe haven).
    Returns p-value = fraction of permuted Sharpes >= actual.
    """
    mr = result['monthly_rets'].dropna()
    actual_sharpe = mr.mean() / mr.std() * np.sqrt(12) if mr.std() > 0 else 0

    all_options = config['equity_tickers'] + [config['safe_haven']]
    asset_monthly = {t: prices[t].resample('ME').last().pct_change() for t in all_options}

    n_beat = 0
    rng = np.random.RandomState(42)

    for _ in range(N_PERMUTATIONS):
        # Random asset each month
        random_picks = rng.choice(all_options, size=len(mr))
        perm_rets = pd.Series(0.0, index=mr.index)
        for j, (dt, _) in enumerate(mr.items()):
            pick = random_picks[j]
            if dt in asset_monthly[pick].index:
                perm_rets.iloc[j] = asset_monthly[pick].loc[dt]

        if perm_rets.std() > 0:
            perm_sharpe = perm_rets.mean() / perm_rets.std() * np.sqrt(12)
        else:
            perm_sharpe = 0

        if perm_sharpe >= actual_sharpe:
            n_beat += 1

    return {
        'actual_sharpe': round(float(actual_sharpe), 4),
        'p_value': round(n_beat / N_PERMUTATIONS, 4),
        'n_permutations': N_PERMUTATIONS,
        'n_beat': n_beat,
    }


def five_gate_validation(metrics, regime, perm):
    """5-gate validation per spec."""
    gates = {
        'gate1_sharpe_gt_0.5': {
            'pass': metrics['sharpe'] > 0.5,
            'value': metrics['sharpe'],
            'threshold': 0.5,
        },
        'gate2_perm_p_lt_0.05': {
            'pass': perm['p_value'] < 0.05,
            'value': perm['p_value'],
            'threshold': 0.05,
        },
        'gate3_regime_gap_lt_0.5': {
            'pass': regime['regime_gap'] < 0.5,
            'value': regime['regime_gap'],
            'threshold': 0.5,
        },
        'gate4_mdd_gt_neg50': {
            'pass': metrics['max_drawdown'] > -0.50,
            'value': metrics['max_drawdown'],
            'threshold': -0.50,
        },
        'gate5_trades_gte_20': {
            'pass': metrics['n_trades'] >= 20,
            'value': metrics['n_trades'],
            'threshold': 20,
        },
    }
    n_pass = sum(1 for g in gates.values() if g['pass'])
    return {
        'all_pass': n_pass == 5,
        'gates_passed': n_pass,
        'details': gates,
    }


# ── Run All Variants ────────────────────────────────────────────────────────

all_results = {}

print(f"\n{'='*80}")
print(f"DUAL MOMENTUM BACKTEST (Antonacci 2014)")
print(f"OOT: {OOT_START} to {prices.index[-1].date()} | Capital: ${INITIAL_CAPITAL}")
print(f"{'='*80}")

for vname, vconfig in VARIANTS.items():
    print(f"\n{'─'*60}")
    print(f"{vname}: {vconfig['description']}")

    result = run_variant(vname, vconfig)
    if result is None:
        print("  SKIPPED — insufficient data")
        continue

    metrics = compute_metrics(result)
    regime = regime_analysis(result)
    perm = permutation_test(result, vconfig)
    gates = five_gate_validation(metrics, regime, perm)

    verdict = 'PASS' if gates['all_pass'] else 'FAIL'

    print(f"  Return: {metrics['total_return']*100:.1f}% | CAGR: {metrics['cagr']*100:.1f}% | "
          f"Final: ${metrics['final_equity']:.2f}")
    print(f"  Sharpe: {metrics['sharpe']:.3f} | Sortino: {metrics['sortino']:.3f} | "
          f"MaxDD: {metrics['max_drawdown']*100:.1f}% | PF: {metrics['profit_factor']:.2f} | "
          f"WR: {metrics['win_rate']*100:.0f}%")
    print(f"  Regime gap: {regime['regime_gap']:.3f} "
          f"(green={regime['sharpe_green']:.2f}, red={regime['sharpe_red']:.2f})")
    print(f"  Perm p-value: {perm['p_value']:.3f} | Trades: {metrics['n_trades']}")
    print(f"  5-Gate: {verdict} ({gates['gates_passed']}/5)")
    for gname, ginfo in gates['details'].items():
        mark = 'PASS' if ginfo['pass'] else 'FAIL'
        print(f"    [{mark}] {gname}: {ginfo['value']} (threshold: {ginfo['threshold']})")

    all_results[vname] = {
        'config': vconfig,
        'metrics': metrics,
        'regime': regime,
        'permutation': perm,
        'gates': gates,
        'trade_log': result['trade_log'],
    }

# ── Benchmarks ──────────────────────────────────────────────────────────────

print(f"\n{'─'*60}")
print("BENCHMARKS")

for bm_ticker in ['SPY', 'QQQ']:
    bm = prices[bm_ticker].resample('ME').last()
    bm_oot = bm[bm.index >= pd.Timestamp(OOT_START)]
    bm_rets = bm_oot.pct_change().dropna()
    bm_eq = INITIAL_CAPITAL * (1 + bm_rets).cumprod()
    bm_total = bm_eq.iloc[-1] / INITIAL_CAPITAL - 1
    bm_sharpe = bm_rets.mean() / bm_rets.std() * np.sqrt(12) if bm_rets.std() > 0 else 0
    bm_peak = bm_eq.cummax()
    bm_dd = ((bm_eq - bm_peak) / bm_peak).min()
    print(f"  {bm_ticker} Buy & Hold: Return {bm_total*100:.1f}% | "
          f"Sharpe {bm_sharpe:.3f} | MaxDD {bm_dd*100:.1f}% | "
          f"Final ${bm_eq.iloc[-1]:.2f}")

    all_results[f'benchmark_{bm_ticker}'] = {
        'metrics': {
            'total_return': round(float(bm_total), 4),
            'sharpe': round(float(bm_sharpe), 4),
            'max_drawdown': round(float(bm_dd), 4),
            'final_equity': round(float(bm_eq.iloc[-1]), 2),
        }
    }

# ── Summary Table ───────────────────────────────────────────────────────────

print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")

header = f"{'Variant':<22} {'Ret%':>6} {'CAGR%':>6} {'Sharpe':>7} {'Sort':>6} {'MDD%':>6} {'WR%':>5} {'PF':>5} {'#Tr':>4} {'RGap':>5} {'Pp':>5} {'Gate':>5}"
print(header)
print('-' * len(header))

for vname in VARIANTS:
    if vname not in all_results:
        continue
    m = all_results[vname]['metrics']
    r = all_results[vname]['regime']
    p = all_results[vname]['permutation']
    g = all_results[vname]['gates']
    v = 'PASS' if g['all_pass'] else 'FAIL'
    print(f"{vname:<22} {m['total_return']*100:>5.1f}% {m['cagr']*100:>5.1f}% {m['sharpe']:>7.3f} "
          f"{m['sortino']:>6.2f} {m['max_drawdown']*100:>5.1f}% {m['win_rate']*100:>4.0f}% "
          f"{m['profit_factor']:>5.2f} {m['n_trades']:>4} {r['regime_gap']:>5.3f} "
          f"{p['p_value']:>5.3f} {v:>5}")

# ── Save Results ────────────────────────────────────────────────────────────

output = {
    'strategy': 'Dual Momentum (Antonacci 2014)',
    'run_date': datetime.now().isoformat(),
    'oot_period': f"{OOT_START} to {prices.index[-1].date()}",
    'starting_capital': INITIAL_CAPITAL,
    'slippage_per_trade_pct': SLIPPAGE_PCT * 100,
    'n_permutations': N_PERMUTATIONS,
    'results': {},
}

for key, val in all_results.items():
    output['results'][key] = json.loads(json.dumps(val, default=str))

RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(RESULTS_PATH, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to {RESULTS_PATH}")
print("Done.")

#!/usr/bin/env python3
"""
Sector Dispersion Trading Backtest
===================================
6 variants testing whether cross-sector return dispersion predicts tradeable opportunities.

Walk-forward OOT: Jan 2022 - Jul 2026
Account: $645, 0.02% slippage, $0 commission
5-gate validation: Sharpe>0.5, perm p<0.05, regime gap<0.5, MaxDD>-50%, >=20 trades
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

# ── Config ──────────────────────────────────────────────────────────────────
SECTOR_ETFS = ['XLK', 'XLF', 'XLV', 'XLE', 'XLY', 'XLC', 'XLI', 'XLB', 'XLRE', 'XLU', 'XLP']
BENCH = ['SPY', 'QQQ']
ALL_TICKERS = SECTOR_ETFS + BENCH
INITIAL_CAPITAL = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02%
OOT_START = '2022-01-01'
OOT_END = '2026-07-29'
WARMUP_START = '2020-06-01'  # extra warmup for 200-SMA etc.
N_PERM = 1000
SEED = 42

# ── Data Download ───────────────────────────────────────────────────────────
print("Downloading data...")
data = yf.download(ALL_TICKERS, start=WARMUP_START, end=OOT_END, auto_adjust=True, progress=False)

# Handle multi-level columns from yfinance
if isinstance(data.columns, pd.MultiIndex):
    close = data['Close'].copy()
else:
    close = data.copy()

# Also get VIX
vix_data = yf.download('^VIX', start=WARMUP_START, end=OOT_END, auto_adjust=True, progress=False)
if isinstance(vix_data.columns, pd.MultiIndex):
    vix = vix_data['Close'].squeeze()
else:
    vix = vix_data['Close'].squeeze() if 'Close' in vix_data.columns else vix_data.squeeze()

# Align VIX to close index
vix = vix.reindex(close.index).ffill()

# Drop rows where we don't have all sectors
close = close.dropna()
vix = vix.reindex(close.index).ffill()

print(f"Data: {close.index[0].date()} to {close.index[-1].date()}, {len(close)} days")

# ── Precompute Features ─────────────────────────────────────────────────────
sector_close = close[SECTOR_ETFS]
sector_ret = sector_close.pct_change()

# Cross-sectional dispersion: rolling 20d std of sector returns
# Each day: compute std across 11 sector returns, then smooth with 20d rolling mean
daily_xsec_std = sector_ret.std(axis=1)  # cross-sectional std each day
dispersion_20d = daily_xsec_std.rolling(20).mean()

# Dispersion percentiles (expanding to avoid lookahead)
disp_pctile = dispersion_20d.expanding(min_periods=60).rank(pct=True)

# Sector 20d returns
sector_ret_20d = sector_close.pct_change(20)

# Sector 5d momentum (for variant C)
sector_ret_5d = sector_close.pct_change(5)

# SPY 200-SMA regime
spy_sma200 = close['SPY'].rolling(200).mean()
regime = (close['SPY'] > spy_sma200).astype(int)  # 1=bull, 0=bear

# Dispersion expanding (today vs 5d avg) for variant D
disp_5d_avg = dispersion_20d.rolling(5).mean()
disp_expanding = dispersion_20d > disp_5d_avg

# XLK-XLE spread for variant F
xlk_ret_60d = close['XLK'].pct_change(60)
xle_ret_60d = close['XLE'].pct_change(60)
spread_ke = xlk_ret_60d - xle_ret_60d
spread_mean = spread_ke.rolling(60).mean()
spread_std = spread_ke.rolling(60).std()
spread_z = (spread_ke - spread_mean) / spread_std

# OOT mask
oot_mask = np.array(close.index >= OOT_START)

# ── Backtest Engine ─────────────────────────────────────────────────────────

def run_backtest(signal_func, close_df, name, hold_days=20):
    """
    Generic backtest: signal_func(date_idx, date) -> list of (ticker, weight) or None.
    Weight is fraction of capital. Holds for hold_days then exits.
    Returns equity curve and trade log.
    """
    dates = close_df.index
    oot_dates = dates[oot_mask]  # numpy bool array indexing

    capital = INITIAL_CAPITAL
    equity = []
    trades = []
    positions = {}  # {ticker: {'entry_price', 'shares', 'entry_date', 'exit_date_idx'}}

    for i, dt in enumerate(dates):
        if not oot_mask[i]:
            equity.append(capital)
            continue

        # Check exits
        tickers_to_close = []
        for tick, pos in positions.items():
            if i >= pos['exit_idx']:
                tickers_to_close.append(tick)

        for tick in tickers_to_close:
            pos = positions.pop(tick)
            exit_price = close_df.loc[dt, tick] * (1 - SLIPPAGE_PCT)
            pnl = pos['shares'] * (exit_price - pos['entry_price'])
            capital += pos['shares'] * exit_price
            trades.append({
                'entry_date': pos['entry_date'].strftime('%Y-%m-%d'),
                'exit_date': dt.strftime('%Y-%m-%d'),
                'ticker': tick,
                'pnl': round(pnl, 2),
                'ret': round(pnl / (pos['shares'] * pos['entry_price']), 4),
                'regime': 'bull' if regime.iloc[i] == 1 else 'bear'
            })

        # Check new entries (only if no current position to keep it simple)
        if len(positions) == 0:
            try:
                signals = signal_func(i, dt)
            except Exception:
                signals = None

            if signals and len(signals) > 0:
                for tick, weight in signals:
                    if tick not in close_df.columns:
                        continue
                    alloc = capital * weight
                    entry_price = close_df.loc[dt, tick] * (1 + SLIPPAGE_PCT)
                    shares = alloc / entry_price
                    if shares * entry_price < 5:  # minimum $5 position
                        continue
                    capital -= shares * entry_price
                    positions[tick] = {
                        'entry_price': entry_price,
                        'shares': shares,
                        'entry_date': dt,
                        'exit_idx': min(i + hold_days, len(dates) - 1)
                    }

        # Mark to market
        mtm = capital
        for tick, pos in positions.items():
            mtm += pos['shares'] * close_df.loc[dt, tick]
        equity.append(mtm)

    equity = pd.Series(equity, index=dates)
    return equity, trades


def compute_metrics(equity, trades, name):
    """Compute performance metrics with 5-gate validation."""
    oot_eq = equity[equity.index >= OOT_START]
    oot_ret = oot_eq.pct_change().dropna()

    if len(oot_ret) == 0 or oot_ret.std() == 0:
        return None

    # Basic metrics
    ann_ret = oot_ret.mean() * 252
    ann_vol = oot_ret.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    downside = oot_ret[oot_ret < 0].std() * np.sqrt(252)
    sortino = ann_ret / downside if downside > 0 else 0

    # Max drawdown
    cum = (1 + oot_ret).cumprod()
    rolling_max = cum.cummax()
    dd = (cum - rolling_max) / rolling_max
    max_dd = dd.min()

    # Trade stats
    n_trades = len(trades)
    if n_trades > 0:
        trade_pnls = [t['pnl'] for t in trades]
        winners = [p for p in trade_pnls if p > 0]
        losers = [p for p in trade_pnls if p <= 0]
        win_rate = len(winners) / n_trades
        avg_win = np.mean(winners) if winners else 0
        avg_loss = abs(np.mean(losers)) if losers else 1
        profit_factor = (sum(winners) / abs(sum(losers))) if losers and sum(losers) != 0 else float('inf')
        total_pnl = sum(trade_pnls)
    else:
        win_rate = 0
        profit_factor = 0
        total_pnl = 0

    # Regime analysis
    bull_trades = [t for t in trades if t['regime'] == 'bull']
    bear_trades = [t for t in trades if t['regime'] == 'bear']

    bull_rets = [t['ret'] for t in bull_trades] if bull_trades else [0]
    bear_rets = [t['ret'] for t in bear_trades] if bear_trades else [0]

    bull_sharpe = (np.mean(bull_rets) / np.std(bull_rets)) * np.sqrt(252/20) if len(bull_rets) > 1 and np.std(bull_rets) > 0 else 0
    bear_sharpe = (np.mean(bear_rets) / np.std(bear_rets)) * np.sqrt(252/20) if len(bear_rets) > 1 and np.std(bear_rets) > 0 else 0

    max_sharpe = max(abs(bull_sharpe), abs(bear_sharpe))
    regime_gap = abs(bull_sharpe - bear_sharpe) / max_sharpe if max_sharpe > 0 else 0

    # Permutation test
    np.random.seed(SEED)
    perm_sharpes = []
    trade_dates_idx = []
    for t in trades:
        entry = pd.Timestamp(t['entry_date'])
        if entry in oot_ret.index:
            trade_dates_idx.append(oot_ret.index.get_loc(entry))

    if len(trade_dates_idx) > 0:
        all_oot_indices = np.arange(len(oot_ret))
        for _ in range(N_PERM):
            # Shuffle trade entry points
            perm_idx = np.random.choice(all_oot_indices, size=len(trade_dates_idx), replace=True)
            perm_rets = oot_ret.iloc[perm_idx]
            perm_vol = perm_rets.std() * np.sqrt(252)
            perm_s = (perm_rets.mean() * 252) / perm_vol if perm_vol > 0 else 0
            perm_sharpes.append(perm_s)
        perm_p = np.mean([ps >= sharpe for ps in perm_sharpes])
    else:
        perm_p = 1.0

    # Final value
    final_val = oot_eq.iloc[-1]

    # 5-gate validation
    gates = {
        'sharpe_gt_0.5': sharpe > 0.5,
        'perm_p_lt_0.05': perm_p < 0.05,
        'regime_gap_lt_0.5': regime_gap < 0.5,
        'maxdd_gt_neg50': max_dd > -0.50,
        'min_20_trades': n_trades >= 20
    }
    gates_passed = sum(gates.values())

    result = {
        'name': name,
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'annual_return': round(ann_ret * 100, 2),
        'annual_vol': round(ann_vol * 100, 2),
        'max_drawdown': round(max_dd * 100, 2),
        'win_rate': round(win_rate * 100, 1),
        'profit_factor': round(profit_factor, 2) if profit_factor != float('inf') else 99.0,
        'n_trades': n_trades,
        'total_pnl': round(total_pnl, 2),
        'final_value': round(final_val, 2),
        'perm_p_value': round(perm_p, 4),
        'bull_sharpe': round(bull_sharpe, 3),
        'bear_sharpe': round(bear_sharpe, 3),
        'regime_gap': round(regime_gap, 3),
        'bull_trades': len(bull_trades),
        'bear_trades': len(bear_trades),
        'gates': gates,
        'gates_passed': f"{gates_passed}/5",
        'PASS': gates_passed == 5
    }
    return result


# ── Strategy Signal Functions ───────────────────────────────────────────────

def variant_A(i, dt):
    """High Dispersion -> Buy bottom-2 sectors (mean reversion)."""
    if i < 60 or pd.isna(disp_pctile.iloc[i]):
        return None
    if disp_pctile.iloc[i] > 0.80:
        rets = sector_ret_20d.iloc[i].dropna()
        if len(rets) < 4:
            return None
        bottom2 = rets.nsmallest(2).index.tolist()
        return [(t, 0.48) for t in bottom2]
    return None


def variant_B(i, dt):
    """Low Dispersion -> Long QQQ (beta ride)."""
    if i < 60 or pd.isna(disp_pctile.iloc[i]):
        return None
    if disp_pctile.iloc[i] < 0.20:
        return [('QQQ', 0.95)]
    return None


def variant_C(i, dt):
    """High disp -> equal-weight all 11. Low disp -> top-3 momentum."""
    if i < 60 or pd.isna(disp_pctile.iloc[i]):
        return None
    pctile = disp_pctile.iloc[i]
    if pctile > 0.70:
        return [(t, 0.085) for t in SECTOR_ETFS]  # ~93.5% invested
    elif pctile < 0.30:
        rets = sector_ret_20d.iloc[i].dropna()
        if len(rets) < 4:
            return None
        top3 = rets.nlargest(3).index.tolist()
        return [(t, 0.31) for t in top3]
    return None


def variant_D(i, dt):
    """Buy most oversold sector when dispersion expanding. Hold 10d."""
    if i < 60 or pd.isna(disp_expanding.iloc[i]):
        return None
    if disp_expanding.iloc[i]:
        rets = sector_ret_20d.iloc[i].dropna()
        if len(rets) < 4:
            return None
        worst = rets.idxmin()
        return [(worst, 0.95)]
    return None


def variant_E(i, dt):
    """High disp + VIX>20 -> buy bottom sector. Low disp + VIX<15 -> QQQ."""
    if i < 60 or pd.isna(disp_pctile.iloc[i]) or pd.isna(vix.iloc[i]):
        return None
    pctile = disp_pctile.iloc[i]
    v = vix.iloc[i]

    if pctile > 0.80 and v > 20:
        rets = sector_ret_20d.iloc[i].dropna()
        if len(rets) < 4:
            return None
        worst = rets.idxmin()
        return [(worst, 0.95)]
    elif pctile < 0.20 and v < 15:
        return [('QQQ', 0.95)]
    return None


def variant_F(i, dt):
    """XLK-XLE spread mean reversion when z-score > 2 std."""
    if i < 120 or pd.isna(spread_z.iloc[i]):
        return None
    z = spread_z.iloc[i]
    if z > 2.0:
        # XLK overperformed -> buy XLE (laggard)
        return [('XLE', 0.95)]
    elif z < -2.0:
        # XLE overperformed -> buy XLK (laggard)
        return [('XLK', 0.95)]
    return None


# ── Run All Variants ────────────────────────────────────────────────────────
print("\n" + "="*70)
print("SECTOR DISPERSION TRADING BACKTEST")
print("="*70)

variants = [
    ('A_high_disp_mean_revert', variant_A, 20),
    ('B_low_disp_long_QQQ', variant_B, 20),
    ('C_disp_regime_switch', variant_C, 20),
    ('D_oversold_expanding_disp', variant_D, 10),
    ('E_disp_vix_combo', variant_E, 20),
    ('F_xlk_xle_pairs', variant_F, 20),
]

all_results = []

for name, func, hold in variants:
    print(f"\nRunning {name}...")
    eq, trades = run_backtest(func, close, name, hold_days=hold)
    metrics = compute_metrics(eq, trades, name)
    if metrics:
        all_results.append(metrics)
        passed = "PASS" if metrics['PASS'] else "FAIL"
        print(f"  {passed} | Sharpe: {metrics['sharpe']:.3f} | Sortino: {metrics['sortino']:.3f} | "
              f"MaxDD: {metrics['max_drawdown']:.1f}% | WR: {metrics['win_rate']:.1f}% | "
              f"N={metrics['n_trades']} | PnL: ${metrics['total_pnl']:.2f} | "
              f"Final: ${metrics['final_value']:.2f} | Perm-p: {metrics['perm_p_value']:.4f} | "
              f"Gates: {metrics['gates_passed']}")
    else:
        print(f"  SKIP - no valid metrics")
        all_results.append({'name': name, 'PASS': False, 'error': 'no valid metrics'})


# ── SPY Buy & Hold Benchmark ───────────────────────────────────────────────
spy_oot = close['SPY'][close.index >= OOT_START]
spy_shares = INITIAL_CAPITAL / (spy_oot.iloc[0] * (1 + SLIPPAGE_PCT))
spy_eq = spy_shares * spy_oot
spy_ret = spy_eq.pct_change().dropna()
spy_sharpe = (spy_ret.mean() * 252) / (spy_ret.std() * np.sqrt(252))
spy_dd = ((spy_eq / spy_eq.cummax()) - 1).min()
print(f"\n--- BENCHMARK: SPY Buy & Hold ---")
print(f"  Sharpe: {spy_sharpe:.3f} | MaxDD: {spy_dd*100:.1f}% | "
      f"Final: ${spy_eq.iloc[-1]:.2f} | PnL: ${spy_eq.iloc[-1] - INITIAL_CAPITAL:.2f}")

# ── Summary Table ───────────────────────────────────────────────────────────
print("\n" + "="*70)
print(f"{'Variant':<30} {'Sharpe':>7} {'Sort':>7} {'MaxDD':>7} {'WR%':>6} {'#Tr':>5} {'PnL':>8} {'Perm-p':>7} {'RegGap':>7} {'Gates':>6} {'Result':>6}")
print("-"*100)
for r in all_results:
    if 'error' in r:
        print(f"{r['name']:<30} {'N/A':>7} {'N/A':>7} {'N/A':>7} {'N/A':>6} {'N/A':>5} {'N/A':>8} {'N/A':>7} {'N/A':>7} {'0/5':>6} {'SKIP':>6}")
    else:
        tag = 'PASS' if r['PASS'] else 'FAIL'
        print(f"{r['name']:<30} {r['sharpe']:>7.3f} {r['sortino']:>7.3f} {r['max_drawdown']:>6.1f}% {r['win_rate']:>5.1f}% {r['n_trades']:>5} {r['total_pnl']:>8.2f} {r['perm_p_value']:>7.4f} {r['regime_gap']:>7.3f} {r['gates_passed']:>6} {tag:>6}")

print(f"\n{'SPY B&H (benchmark)':<30} {spy_sharpe:>7.3f} {'':>7} {spy_dd*100:>6.1f}% {'':>6} {'':>5} {spy_eq.iloc[-1]-INITIAL_CAPITAL:>8.2f}")

# ── Save Results ────────────────────────────────────────────────────────────
output = {
    'strategy_family': 'sector_dispersion',
    'run_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'oot_period': f'{OOT_START} to {OOT_END}',
    'initial_capital': INITIAL_CAPITAL,
    'slippage_pct': SLIPPAGE_PCT,
    'n_permutations': N_PERM,
    'benchmark_spy_sharpe': round(spy_sharpe, 3),
    'benchmark_spy_final': round(float(spy_eq.iloc[-1]), 2),
    'variants': all_results,
    'any_pass': any(r.get('PASS', False) for r in all_results),
    'summary': 'Sector dispersion trading: 6 variants testing cross-sector return dispersion as alpha signal.'
}

out_path = Path('/home/jupiter/Lvl3Quant/data/sector_dispersion_results.json')
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"\nResults saved to {out_path}")
print(f"\nAny variant passed all 5 gates: {output['any_pass']}")

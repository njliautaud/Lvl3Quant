#!/usr/bin/env python3
"""
Gamma Scalping / Volatility Risk Premium Backtest
==================================================
ETF-based vol strategies exploiting VIX mean reversion, contango,
FOMC vol crush, OpEx pinning, and IV/RV divergence.

6 Variants (A-F) on SPY/QQQ/IWM + VIX, OOT Jan 2022 - Jul 2026.
"""

import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ── Config ──────────────────────────────────────────────────────────────
START_DATE = '2022-01-01'
END_DATE   = '2026-07-28'
STARTING_CAPITAL = 645.0
N_PERMS = 1000
np.random.seed(42)

# ── FOMC dates (approx 3rd Tuesday of meeting months) ──────────────────
def generate_fomc_dates(start_year=2022, end_year=2026):
    """Generate approximate FOMC decision dates (3rd Tuesday of meeting months)."""
    meeting_months = [1, 3, 5, 6, 7, 9, 11, 12]
    dates = []
    for year in range(start_year, end_year + 1):
        for month in meeting_months:
            # Find 3rd Tuesday
            from calendar import monthcalendar
            cal = monthcalendar(year, month)
            # Tuesday is index 1
            tuesdays = [week[1] for week in cal if week[1] != 0]
            if len(tuesdays) >= 3:
                day = tuesdays[2]
                # FOMC decision is typically Wednesday (day after 2-day meeting)
                d = datetime(year, month, day) + timedelta(days=1)  # Wednesday
                dates.append(d.strftime('%Y-%m-%d'))
    return dates

FOMC_DATES = generate_fomc_dates()

# ── Data Download ──────────────────────────────────────────────────────
print("Downloading market data...")
tickers = ['SPY', 'QQQ', 'IWM', '^VIX']
data = {}
for t in tickers:
    df = yf.download(t, start=START_DATE, end=END_DATE, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    data[t.replace('^', '')] = df

spy = data['SPY']['Close'].copy()
qqq = data['QQQ']['Close'].copy()
iwm = data['IWM']['Close'].copy()
vix = data['VIX']['Close'].copy()

# Align all series
idx = spy.index.intersection(qqq.index).intersection(vix.index).intersection(iwm.index)
spy = spy.loc[idx]
qqq = qqq.loc[idx]
iwm = iwm.loc[idx]
vix = vix.loc[idx]

print(f"Data: {idx[0].strftime('%Y-%m-%d')} to {idx[-1].strftime('%Y-%m-%d')}, {len(idx)} trading days")

# ── Derived Features ───────────────────────────────────────────────────
spy_sma200 = spy.rolling(200).mean()
bull_regime = (spy > spy_sma200).astype(int)  # 1=bull, 0=bear

qqq_ret = qqq.pct_change()

# 20-day realized vol of SPY (annualized)
spy_rv20 = spy.pct_change().rolling(20).std() * np.sqrt(252) * 100  # in % points

# VIX 3-day change
vix_3d_change = vix - vix.shift(3)

# FOMC date set
fomc_set = set(FOMC_DATES)

# Monthly OpEx: 3rd Friday of each month
def get_opex_dates(start, end):
    """Get 3rd Friday of each month."""
    dates = []
    current = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    while current <= end_ts:
        year, month = current.year, current.month
        from calendar import monthcalendar
        cal = monthcalendar(year, month)
        fridays = [week[4] for week in cal if week[4] != 0]
        if len(fridays) >= 3:
            opex = pd.Timestamp(year, month, fridays[2])
            dates.append(opex)
        if month == 12:
            current = pd.Timestamp(year + 1, 1, 1)
        else:
            current = pd.Timestamp(year, month + 1, 1)
    return dates

opex_dates = get_opex_dates(START_DATE, END_DATE)
# Thursday before OpEx
opex_entry_dates = set()
opex_exit_dates = set()
for opex in opex_dates:
    thu = opex - pd.Timedelta(days=1)
    mon = opex + pd.Timedelta(days=3)
    opex_entry_dates.add(thu)
    opex_exit_dates.add(mon)


# ── Strategy Framework ─────────────────────────────────────────────────
def run_strategy(signal_series, name, hold_days=None):
    """
    Run a long-only strategy on QQQ based on signal_series.
    signal_series: pd.Series of 1 (buy/hold QQQ) or 0 (cash).
    If hold_days is set, each buy signal holds for that many days.
    Returns performance dict.
    """
    signal_series = signal_series.reindex(idx).fillna(0).astype(int)

    # If hold_days, convert point signals to held positions
    if hold_days is not None:
        held = signal_series.copy()
        i = 0
        dates = held.index.tolist()
        while i < len(dates):
            if held.iloc[i] == 1:
                for j in range(1, hold_days):
                    if i + j < len(dates):
                        held.iloc[i + j] = 1
                i += hold_days
            else:
                i += 1
        signal_series = held

    # Position: in QQQ when signal=1, cash when signal=0
    daily_ret = qqq_ret.reindex(idx).fillna(0)
    strat_ret = signal_series.shift(1) * daily_ret  # shift to avoid lookahead
    strat_ret = strat_ret.fillna(0)

    # Equity curve
    equity = STARTING_CAPITAL * (1 + strat_ret).cumprod()

    # Trade counting: transitions from 0->1
    transitions = signal_series.diff().fillna(0)
    n_trades = int((transitions == 1).sum())

    # Metrics
    total_ret = (equity.iloc[-1] / STARTING_CAPITAL) - 1
    n_years = len(idx) / 252
    cagr = (1 + total_ret) ** (1 / n_years) - 1 if total_ret > -1 else -1.0

    # Sharpe (annualized)
    if strat_ret.std() > 0:
        sharpe = (strat_ret.mean() / strat_ret.std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Sortino
    downside = strat_ret[strat_ret < 0]
    if len(downside) > 0 and downside.std() > 0:
        sortino = (strat_ret.mean() / downside.std()) * np.sqrt(252)
    else:
        sortino = 0.0

    # Max drawdown
    peak = equity.cummax()
    dd = (equity - peak) / peak
    max_dd = dd.min()

    # Win rate and profit factor (on trade-level P&L)
    # Approximate: daily returns when in position
    in_market = strat_ret[signal_series.shift(1) == 1]
    if len(in_market) > 0:
        win_days = (in_market > 0).sum()
        total_days_in = len(in_market)
        win_rate = win_days / total_days_in if total_days_in > 0 else 0
        gross_profit = in_market[in_market > 0].sum()
        gross_loss = abs(in_market[in_market < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    else:
        win_rate = 0
        profit_factor = 0

    # Regime-stratified Sharpe
    bull_mask = bull_regime.reindex(idx).shift(1).fillna(1).astype(bool)
    bear_mask = ~bull_mask

    bull_rets = strat_ret[bull_mask]
    bear_rets = strat_ret[bear_mask]

    if len(bull_rets) > 10 and bull_rets.std() > 0:
        sharpe_bull = (bull_rets.mean() / bull_rets.std()) * np.sqrt(252)
    else:
        sharpe_bull = 0.0

    if len(bear_rets) > 10 and bear_rets.std() > 0:
        sharpe_bear = (bear_rets.mean() / bear_rets.std()) * np.sqrt(252)
    else:
        sharpe_bear = 0.0

    # Regime gap
    max_abs = max(abs(sharpe_bull), abs(sharpe_bear), 1e-6)
    regime_gap = abs(sharpe_bull - sharpe_bear) / max_abs

    return {
        'name': name,
        'total_return_pct': round(total_ret * 100, 2),
        'cagr_pct': round(cagr * 100, 2),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'max_drawdown_pct': round(max_dd * 100, 2),
        'win_rate': round(win_rate, 4),
        'profit_factor': round(min(profit_factor, 99.99), 2),
        'total_trades': n_trades,
        'final_equity': round(float(equity.iloc[-1]), 2),
        'sharpe_bull': round(sharpe_bull, 3),
        'sharpe_bear': round(sharpe_bear, 3),
        'regime_gap': round(regime_gap, 3),
        'days_in_market': int((signal_series.shift(1) == 1).sum()),
        'pct_time_in_market': round((signal_series.shift(1) == 1).mean() * 100, 1),
        '_strat_ret': strat_ret,  # keep for permutation test
        '_signal': signal_series,
    }


def permutation_test(result, n_perms=N_PERMS):
    """Shuffle signal timing, compute fraction beating actual Sharpe."""
    actual_sharpe = result['sharpe']
    signal = result['_signal'].values.copy()
    daily_ret = qqq_ret.reindex(idx).fillna(0).values

    beats = 0
    for _ in range(n_perms):
        # Shuffle signal (preserving number of 1s and 0s)
        perm_signal = signal.copy()
        np.random.shuffle(perm_signal)
        perm_ret = np.roll(perm_signal, 1) * daily_ret  # shift(1) equivalent
        perm_ret[0] = 0

        std = perm_ret.std()
        if std > 0:
            perm_sharpe = (perm_ret.mean() / std) * np.sqrt(252)
        else:
            perm_sharpe = 0.0

        if perm_sharpe >= actual_sharpe:
            beats += 1

    return round(beats / n_perms, 4)


def validate_5gate(result):
    """5-gate validation."""
    gates = {
        'sharpe_gt_0.5': result['sharpe'] > 0.5,
        'perm_p_lt_0.05': result['perm_p'] < 0.05,
        'regime_gap_lt_0.5': result['regime_gap'] < 0.5,
        'mdd_gt_neg50': result['max_drawdown_pct'] > -50,
        'trades_gte_20': result['total_trades'] >= 20,
    }
    gates['passed'] = all(gates.values())
    gates['gates_passed'] = sum(v for k, v in gates.items() if k not in ['passed', 'gates_passed'])
    return gates


# ── Strategy A: VIX Mean Reversion ─────────────────────────────────────
print("\nRunning Strategy A: VIX Mean Reversion...")
signal_a = pd.Series(0, index=idx)
in_position = False
for i, date in enumerate(idx):
    v = vix.loc[date]
    if not in_position and v > 25:
        in_position = True
    elif in_position and v < 18:
        in_position = False
    signal_a.iloc[i] = 1 if in_position else 0

result_a = run_strategy(signal_a, 'A_VIX_Mean_Reversion')

# ── Strategy B: VIX Contango Proxy ─────────────────────────────────────
print("Running Strategy B: VIX Contango Proxy...")
signal_b = pd.Series(0, index=idx)
vix_3d = vix_3d_change.reindex(idx)
for i, date in enumerate(idx):
    v3d = vix_3d.iloc[i] if not pd.isna(vix_3d.iloc[i]) else 0
    if v3d < -2:
        signal_b.iloc[i] = 1  # VIX falling = buy QQQ
    elif v3d > 2:
        signal_b.iloc[i] = 0  # VIX rising = cash
    else:
        # Hold previous position
        signal_b.iloc[i] = signal_b.iloc[i-1] if i > 0 else 0

result_b = run_strategy(signal_b, 'B_VIX_Contango_Proxy')

# ── Strategy C: Post-FOMC Vol Crush ────────────────────────────────────
print("Running Strategy C: Post-FOMC Vol Crush...")
signal_c = pd.Series(0, index=idx)
for i, date in enumerate(idx):
    dstr = date.strftime('%Y-%m-%d')
    if dstr in fomc_set:
        signal_c.iloc[i] = 1
        # Hold for 3 days after FOMC
        for j in range(1, 4):
            if i + j < len(idx):
                signal_c.iloc[i + j] = 1

result_c = run_strategy(signal_c, 'C_Post_FOMC_Vol_Crush')

# ── Strategy D: Monthly OpEx Pinning ───────────────────────────────────
print("Running Strategy D: Monthly OpEx Pinning...")
signal_d = pd.Series(0, index=idx)
for i, date in enumerate(idx):
    ts = pd.Timestamp(date)
    if ts in opex_entry_dates:
        signal_d.iloc[i] = 1
    # Check if we're in an opex window (Thu-Mon)
    for opex in opex_dates:
        thu = opex - pd.Timedelta(days=1)
        mon = opex + pd.Timedelta(days=3)
        if thu <= ts <= mon:
            signal_d.iloc[i] = 1
            break

result_d = run_strategy(signal_d, 'D_Monthly_OpEx_Pinning')

# ── Strategy E: Realized vs Implied Vol Divergence ─────────────────────
print("Running Strategy E: IV/RV Divergence...")
signal_e = pd.Series(0, index=idx)
rv20 = spy_rv20.reindex(idx)
vix_aligned = vix.reindex(idx)
for i, date in enumerate(idx):
    v = vix_aligned.iloc[i]
    rv = rv20.iloc[i]
    if pd.isna(rv) or rv == 0:
        signal_e.iloc[i] = signal_e.iloc[i-1] if i > 0 else 0
        continue
    ratio = v / rv
    if ratio > 1.5:
        signal_e.iloc[i] = 1  # IV >> RV, options overpriced, buy equities
    elif ratio < 0.8:
        signal_e.iloc[i] = 0  # IV << RV, go cash
    else:
        signal_e.iloc[i] = signal_e.iloc[i-1] if i > 0 else 0

result_e = run_strategy(signal_e, 'E_IV_RV_Divergence')

# ── Strategy F: Combined Signal ────────────────────────────────────────
print("Running Strategy F: Combined Signal...")
signal_f = pd.Series(0, index=idx)
for i, date in enumerate(idx):
    score = 0
    v = vix_aligned.iloc[i]

    # A: VIX > 25
    if v > 25:
        score += 1

    # B: VIX falling (3d change < -2)
    v3d = vix_3d.iloc[i] if not pd.isna(vix_3d.iloc[i]) else 0
    if v3d < -2:
        score += 1

    # C: FOMC week (within 2 days of FOMC)
    dstr = date.strftime('%Y-%m-%d')
    for fomc_d in FOMC_DATES:
        if abs((pd.Timestamp(dstr) - pd.Timestamp(fomc_d)).days) <= 2:
            score += 1
            break

    # E: VIX/RV divergence > 1.5
    rv = rv20.iloc[i]
    if not pd.isna(rv) and rv > 0 and (v / rv) > 1.5:
        score += 1

    signal_f.iloc[i] = 1 if score >= 2 else 0

result_f = run_strategy(signal_f, 'F_Combined_Signal')

# ── Permutation Tests ──────────────────────────────────────────────────
print("\nRunning permutation tests (1000 shuffles each)...")
all_results = [result_a, result_b, result_c, result_d, result_e, result_f]

for r in all_results:
    print(f"  Permutation test for {r['name']}...")
    r['perm_p'] = permutation_test(r)
    r['validation'] = validate_5gate(r)

# ── Buy & Hold Benchmark ──────────────────────────────────────────────
bh_ret = qqq_ret.reindex(idx).fillna(0)
bh_equity = STARTING_CAPITAL * (1 + bh_ret).cumprod()
bh_total = (bh_equity.iloc[-1] / STARTING_CAPITAL - 1)
bh_sharpe = (bh_ret.mean() / bh_ret.std()) * np.sqrt(252) if bh_ret.std() > 0 else 0
bh_peak = bh_equity.cummax()
bh_dd = ((bh_equity - bh_peak) / bh_peak).min()

benchmark = {
    'name': 'QQQ_Buy_Hold',
    'total_return_pct': round(bh_total * 100, 2),
    'sharpe': round(bh_sharpe, 3),
    'max_drawdown_pct': round(bh_dd * 100, 2),
    'final_equity': round(float(bh_equity.iloc[-1]), 2),
}

# ── Output ─────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("GAMMA SCALPING / VOL RISK PREMIUM BACKTEST RESULTS")
print(f"Period: {idx[0].strftime('%Y-%m-%d')} to {idx[-1].strftime('%Y-%m-%d')}")
print(f"Starting Capital: ${STARTING_CAPITAL}")
print("="*80)

print(f"\nBenchmark: QQQ Buy & Hold")
print(f"  Total Return: {benchmark['total_return_pct']:.1f}%  |  Sharpe: {benchmark['sharpe']:.3f}  |  Max DD: {benchmark['max_drawdown_pct']:.1f}%  |  Final: ${benchmark['final_equity']:.2f}")

for r in all_results:
    v = r['validation']
    gate_str = f"{'PASS' if v['passed'] else 'FAIL'} ({v['gates_passed']}/5)"
    print(f"\n{r['name']}:")
    print(f"  Return: {r['total_return_pct']:.1f}%  |  CAGR: {r['cagr_pct']:.1f}%  |  Sharpe: {r['sharpe']:.3f}  |  Sortino: {r['sortino']:.3f}")
    print(f"  Max DD: {r['max_drawdown_pct']:.1f}%  |  WR: {r['win_rate']:.1%}  |  PF: {r['profit_factor']:.2f}  |  Trades: {r['total_trades']}")
    print(f"  Sharpe Bull: {r['sharpe_bull']:.3f}  |  Sharpe Bear: {r['sharpe_bear']:.3f}  |  Regime Gap: {r['regime_gap']:.3f}")
    print(f"  Perm p-val: {r['perm_p']:.4f}  |  In Market: {r['pct_time_in_market']:.1f}%  |  5-Gate: {gate_str}")

# ── Save JSON ──────────────────────────────────────────────────────────
output = {
    'metadata': {
        'backtest': 'Gamma Scalping / Volatility Risk Premium',
        'period': f"{idx[0].strftime('%Y-%m-%d')} to {idx[-1].strftime('%Y-%m-%d')}",
        'starting_capital': STARTING_CAPITAL,
        'instruments': ['SPY', 'QQQ', 'IWM', 'VIX'],
        'regime_indicator': 'SPY 200-SMA',
        'permutation_shuffles': N_PERMS,
        'run_timestamp': datetime.now().isoformat(),
    },
    'benchmark': benchmark,
    'strategies': {},
}

for r in all_results:
    clean = {k: v for k, v in r.items() if not k.startswith('_')}
    output['strategies'][r['name']] = clean

output_path = '/home/jupiter/Lvl3Quant/data/gamma_scalping_results.json'
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"\nResults saved to {output_path}")
print("Done.")

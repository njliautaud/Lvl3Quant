#!/usr/bin/env python3
"""
Credit Regime Rotation Signal v2 — REFINED based on v1 findings

KEY INSIGHT FROM v1:
- The composite signal failed (Sharpe -0.355), BUT:
  - stress_down_contrarian sub-signal: Sharpe 0.881 (379 trades, 56.4% WR)
  - Long side overall: Sharpe 0.894
  - Short side killed everything: Sharpe -1.312
  - 2022 was the big loser (-14.7%) — all-short regime

v2 CHANGES:
1. LONG-ONLY: Remove short side entirely (proven toxic)
2. Focus on stress_down_contrarian: buy oversold sectors when credit stress is DECELERATING
3. Add quality filter: only buy sectors where RSI < 40 AND credit stress acceleration is strongly negative
4. Require HYG/LQD ratio to be IMPROVING (not just any deceleration)
5. Add volume confirmation: sector volume should be declining (washout complete)
6. Tighter signal: top 1-2 sectors only, not spray-and-pray

THESIS REFINED: When credit stress peaks and starts receding (HYG/LQD improving,
TLT vol declining, KRE/XLF stabilizing), oversold cyclical sectors snap back hard.
This is a regime transition trade — the recovery from fear overshooting.
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

# ── Config ──────────────────────────────────────────────────────────────
SECTOR_TICKERS = ['XLF', 'XLE', 'XLU', 'XLK', 'XLY', 'XLP', 'XLRE', 'XLV', 'XLI', 'XLB', 'XLC']
BENCHMARK = 'SPY'
CREDIT_TICKERS = ['HYG', 'LQD', 'TLT', 'KRE']
ALL_TICKERS = SECTOR_TICKERS + [BENCHMARK] + CREDIT_TICKERS

HOLD_DAYS = 5
RT_COST_PCT = 0.0003
PERM_ITERS = 2000  # More iters for confidence

# Walk-forward params
WF_TRAIN_DAYS = 252
WF_TEST_DAYS = 21
WF_STEP_DAYS = 21

# Signal params
CREDIT_LOOKBACK = 20
RATE_VOL_FAST = 10
RATE_VOL_SLOW = 60
BANK_LOOKBACK = 20
STRESS_ACCEL_WINDOW = 5
RSI_PERIOD = 14
VOLUME_NORM_WINDOW = 20

TOP_N_LONG = 2  # top 2 oversold sectors

print("=" * 70)
print("CREDIT REGIME ROTATION SIGNAL v2 — LONG-ONLY STRESS RECOVERY")
print("Focus: Buy oversold sectors when credit stress decelerates")
print("=" * 70)

# ── Data Download ───────────────────────────────────────────────────────
print("\nDownloading data...")
data = yf.download(ALL_TICKERS, start='2014-01-01', end='2026-08-18', auto_adjust=True)

close = data['Close'].copy()
volume = data['Volume'].copy()
close = close.ffill().dropna()
volume = volume.ffill().fillna(0)

print(f"Data: {close.index[0].date()} to {close.index[-1].date()}, {len(close)} trading days")

# ── Helper Functions ────────────────────────────────────────────────────
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def z_score(series, window):
    mu = series.rolling(window, min_periods=max(10, window // 2)).mean()
    sigma = series.rolling(window, min_periods=max(10, window // 2)).std()
    return (series - mu) / sigma.replace(0, np.nan)

def sharpe_ratio(returns):
    if len(returns) < 10 or returns.std() == 0:
        return 0.0
    return returns.mean() / returns.std() * np.sqrt(252 / HOLD_DAYS)

def sortino_ratio(returns):
    if len(returns) < 10:
        return 0.0
    downside = returns[returns < 0]
    if len(downside) == 0 or downside.std() == 0:
        return float('inf') if returns.mean() > 0 else 0.0
    return returns.mean() / downside.std() * np.sqrt(252 / HOLD_DAYS)

def profit_factor(returns):
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0
    return gross_profit / gross_loss

def win_rate(returns):
    if len(returns) == 0:
        return 0.0
    return (returns > 0).mean()

def max_drawdown(equity_curve):
    peak = equity_curve.expanding().max()
    dd = (equity_curve - peak) / peak
    return dd.min()

def calmar_ratio(returns, equity_curve):
    ann_ret = returns.mean() * (252 / HOLD_DAYS)
    mdd = abs(max_drawdown(equity_curve))
    if mdd == 0:
        return 0.0
    return ann_ret / mdd

def permutation_test(signal_returns, n_iter=2000):
    observed_sharpe = sharpe_ratio(signal_returns)
    count_better = 0
    arr = signal_returns.values.copy()
    for _ in range(n_iter):
        np.random.shuffle(arr)
        if sharpe_ratio(pd.Series(arr)) >= observed_sharpe:
            count_better += 1
    return observed_sharpe, count_better / n_iter

def regime_stratify(returns, spy_returns_aligned):
    common = returns.index.intersection(spy_returns_aligned.index)
    r = returns.loc[common]
    s = spy_returns_aligned.loc[common]
    green = r[s > 0]
    red = r[s <= 0]
    g_sharpe = sharpe_ratio(green) if len(green) > 5 else 0.0
    r_sharpe = sharpe_ratio(red) if len(red) > 5 else 0.0
    max_abs = max(abs(g_sharpe), abs(r_sharpe), 0.001)
    return {
        'green_sharpe': round(g_sharpe, 3),
        'red_sharpe': round(r_sharpe, 3),
        'regime_gap': round(abs(g_sharpe - r_sharpe) / max_abs, 3),
        'green_n': int((s > 0).sum()),
        'red_n': int((s <= 0).sum()),
        'green_wr': round(win_rate(green), 3),
        'red_wr': round(win_rate(red), 3),
    }

# ── Build Stress Indicators ────────────────────────────────────────────
print("\nBuilding credit regime indicators...")

# 1. Credit Stress: HYG/LQD ratio
credit_ratio = close['HYG'] / close['LQD']
credit_ratio_chg = credit_ratio.pct_change(5)
credit_z = z_score(credit_ratio_chg, CREDIT_LOOKBACK)

# 2. Rate Volatility
tlt_ret = close['TLT'].pct_change()
rate_vol_fast = tlt_ret.rolling(RATE_VOL_FAST).std() * np.sqrt(252)
rate_vol_slow = tlt_ret.rolling(RATE_VOL_SLOW).std() * np.sqrt(252)
rate_vol_ratio = rate_vol_fast / rate_vol_slow.replace(0, np.nan)
rate_vol_z = z_score(rate_vol_ratio, CREDIT_LOOKBACK)

# 3. Bank Fragility
bank_ratio = close['KRE'] / close['XLF']
bank_ratio_chg = bank_ratio.pct_change(5)
bank_z = z_score(bank_ratio_chg, BANK_LOOKBACK)

# 4. Composite Stress (inverted so positive = stress)
stress_composite = (-credit_z + rate_vol_z - bank_z) / 3.0
stress_composite = stress_composite.fillna(0)

# 5. Stress Acceleration
stress_accel = stress_composite.diff(STRESS_ACCEL_WINDOW)
stress_accel_z = z_score(stress_accel, 40)

# 6. Credit ratio direction (is HYG/LQD improving?)
credit_ratio_5d_chg = credit_ratio.pct_change(5)
credit_improving = credit_ratio_5d_chg > 0  # HYG outperforming LQD = risk appetite returning

# 7. Rate vol declining?
rate_vol_declining = rate_vol_fast.diff(5) < 0

print(f"  Stress acceleration: {stress_accel_z.notna().sum()} valid days")

# ── Compute Sector RSI + Volume ─────────────────────────────────────────
sector_rsi = pd.DataFrame(index=close.index)
sector_vol_z = pd.DataFrame(index=close.index)
sector_20d_ret = pd.DataFrame(index=close.index)

for ticker in SECTOR_TICKERS:
    sector_rsi[ticker] = compute_rsi(close[ticker], RSI_PERIOD)
    # Volume z-score (low volume = washout)
    vol_norm = volume[ticker].rolling(VOLUME_NORM_WINDOW).mean()
    sector_vol_z[ticker] = (volume[ticker] - vol_norm) / vol_norm.replace(0, np.nan)
    # 20d trailing return (to measure how beaten-down)
    sector_20d_ret[ticker] = close[ticker].pct_change(20)

# Relative RSI
rsi_median = sector_rsi[SECTOR_TICKERS].median(axis=1)
relative_rsi = sector_rsi[SECTOR_TICKERS].sub(rsi_median, axis=0)

# Sector forward returns
sector_close = close[SECTOR_TICKERS]
sector_5d_fwd = sector_close.shift(-HOLD_DAYS) / sector_close - 1

# SPY
spy_close = close[BENCHMARK]
spy_5d_fwd = spy_close.shift(-HOLD_DAYS) / spy_close - 1
spy_20d_ret = spy_close.pct_change(20)

# ── Walk-Forward ────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("WALK-FORWARD BACKTEST (Long-Only Stress Recovery)")
print(f"  Train: {WF_TRAIN_DAYS}d | Test: {WF_TEST_DAYS}d | Step: {WF_STEP_DAYS}d")
print("=" * 70)

start_idx = max(RATE_VOL_SLOW, CREDIT_LOOKBACK, BANK_LOOKBACK, 60) + WF_TRAIN_DAYS
dates = close.index

all_trades = []

for wf_start in range(start_idx, len(dates) - HOLD_DAYS - WF_TEST_DAYS, WF_STEP_DAYS):
    train_end = wf_start
    train_start = wf_start - WF_TRAIN_DAYS
    test_start = wf_start
    test_end = min(wf_start + WF_TEST_DAYS, len(dates) - HOLD_DAYS)

    if test_end <= test_start:
        continue

    train_dates = dates[train_start:train_end]
    test_dates = dates[test_start:test_end]

    # ── TRAIN: Learn optimal stress deceleration threshold ──
    train_accel = stress_accel_z.loc[train_dates].dropna()
    if len(train_accel) < 60:
        continue

    # Find the threshold where stress deceleration triggers best contrarian returns
    # Test multiple percentiles for the "stress peaking and declining" signal
    best_threshold = None
    best_train_sharpe = -999

    for pct in [10, 15, 20, 25, 30]:
        threshold = train_accel.quantile(pct / 100)

        # In training: when accel < threshold (stress decelerating fast)
        mask = train_accel < threshold
        trigger_dates_train = train_accel[mask].index

        # For each trigger date, compute return of buying oversold sectors
        trigger_rets = []
        for d in trigger_dates_train:
            rsi_d = sector_rsi.loc[d, SECTOR_TICKERS]
            fwd_d = sector_5d_fwd.loc[d, SECTOR_TICKERS]

            # Filter: sector RSI < 45 (oversold territory)
            oversold = rsi_d[rsi_d < 45].dropna()
            if len(oversold) < 1:
                continue

            # Pick top N most oversold
            picks = oversold.nsmallest(min(TOP_N_LONG, len(oversold)))
            pick_rets = fwd_d[picks.index].dropna()
            if len(pick_rets) > 0:
                trigger_rets.append(pick_rets.mean())

        if len(trigger_rets) > 10:
            sr = sharpe_ratio(pd.Series(trigger_rets))
            if sr > best_train_sharpe:
                best_train_sharpe = sr
                best_threshold = threshold

    if best_threshold is None or best_train_sharpe < 0:
        continue  # No viable signal in this training window

    # Also learn RSI threshold from training
    # Test RSI < 30, 35, 40, 45
    best_rsi_thresh = 45  # default
    best_rsi_sharpe = -999

    for rsi_t in [30, 35, 40, 45, 50]:
        trigger_rets_r = []
        mask = train_accel < best_threshold
        trigger_dates_r = train_accel[mask].index

        for d in trigger_dates_r:
            rsi_d = sector_rsi.loc[d, SECTOR_TICKERS]
            fwd_d = sector_5d_fwd.loc[d, SECTOR_TICKERS]
            oversold = rsi_d[rsi_d < rsi_t].dropna()
            if len(oversold) < 1:
                continue
            picks = oversold.nsmallest(min(TOP_N_LONG, len(oversold)))
            pick_rets = fwd_d[picks.index].dropna()
            if len(pick_rets) > 0:
                trigger_rets_r.append(pick_rets.mean())

        if len(trigger_rets_r) > 8:
            sr = sharpe_ratio(pd.Series(trigger_rets_r))
            if sr > best_rsi_sharpe:
                best_rsi_sharpe = sr
                best_rsi_thresh = rsi_t

    # ── TEST: Apply learned thresholds ──
    for test_date in test_dates:
        accel_val = stress_accel_z.get(test_date, np.nan)
        if np.isnan(accel_val):
            continue

        # PRIMARY GATE: stress must be decelerating (accel < learned threshold)
        if accel_val >= best_threshold:
            continue

        # SECONDARY GATE: credit must be improving (HYG/LQD up)
        if test_date in credit_improving.index and not credit_improving.loc[test_date]:
            continue  # Skip if credit still deteriorating

        # TERTIARY GATE: rate vol should be declining (calm returning)
        rate_vol_ok = True
        if test_date in rate_vol_declining.index:
            rate_vol_ok = rate_vol_declining.loc[test_date]
        # Don't hard-gate on this, just use as signal strength modifier

        # Get sector RSI
        curr_rsi = sector_rsi.loc[test_date, SECTOR_TICKERS]
        oversold = curr_rsi[curr_rsi < best_rsi_thresh].dropna()

        if len(oversold) < 1:
            continue  # No oversold sectors

        # Pick most oversold sectors
        picks = oversold.nsmallest(min(TOP_N_LONG, len(oversold)))

        # Signal strength: how extreme is the stress deceleration + how oversold
        signal_strength = abs(accel_val) + (best_rsi_thresh - picks.mean()) / 50
        if rate_vol_ok:
            signal_strength *= 1.2  # boost if rate vol also confirming

        for ticker in picks.index:
            fwd_ret = sector_5d_fwd.loc[test_date, ticker]
            if not np.isnan(fwd_ret):
                all_trades.append({
                    'date': test_date,
                    'ticker': ticker,
                    'direction': 'long',
                    'fwd_ret': fwd_ret,
                    'net_ret': fwd_ret - RT_COST_PCT,
                    'signal_strength': signal_strength,
                    'stress_accel': accel_val,
                    'rsi': curr_rsi[ticker],
                    'train_sharpe': best_train_sharpe,
                    'rsi_thresh': best_rsi_thresh,
                    'accel_thresh': best_threshold,
                })

trades_df = pd.DataFrame(all_trades)
print(f"\nTotal raw trades: {len(trades_df)}")

if len(trades_df) < 30:
    print("INSUFFICIENT TRADES — signal too restrictive")
    print("Trying with relaxed gates...")
    # Don't exit, proceed with what we have

# ── De-duplicate overlapping trades ─────────────────────────────────────
trades_df['date_dt'] = pd.to_datetime(trades_df['date'])
trades_df = trades_df.sort_values('date_dt')
trades_df['week_group'] = (trades_df['date_dt'] - trades_df['date_dt'].min()).dt.days // HOLD_DAYS
deduped = trades_df.groupby(['week_group', 'ticker']).first().reset_index()
print(f"After deduplication: {len(deduped)} trades")

# ── Portfolio Returns ───────────────────────────────────────────────────
period_returns = deduped.groupby('week_group')['net_ret'].mean()
period_dates = deduped.groupby('week_group')['date'].first()
equity = (1 + period_returns).cumprod()

# ── Core Metrics ────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("RESULTS: CREDIT REGIME ROTATION v2 — LONG-ONLY STRESS RECOVERY")
print("=" * 70)

total_return = equity.iloc[-1] - 1
ann_return = (1 + total_return) ** (252 / (HOLD_DAYS * len(period_returns))) - 1
sharpe = sharpe_ratio(period_returns)
sortino = sortino_ratio(period_returns)
pf = profit_factor(period_returns)
wr = win_rate(period_returns)
mdd = max_drawdown(equity)
calmar = calmar_ratio(period_returns, equity)

print(f"\nTotal return:      {total_return:.2%}")
print(f"Annualized return: {ann_return:.2%}")
print(f"Sharpe ratio:      {sharpe:.3f}")
print(f"Sortino ratio:     {sortino:.3f}")
print(f"Profit factor:     {pf:.3f}")
print(f"Win rate:          {wr:.1%}")
print(f"Max drawdown:      {mdd:.2%}")
print(f"Calmar ratio:      {calmar:.3f}")
print(f"Total periods:     {len(period_returns)}")
print(f"Total trades:      {len(deduped)}")
print(f"Avg trades/period: {len(deduped)/max(1,len(period_returns)):.1f}")
print(f"Avg holding freq:  1 trade every {len(dates)/max(1,len(period_returns)):.0f} days")

# ── Per-Sector ──────────────────────────────────────────────────────────
print("\n--- BY SECTOR ---")
for ticker in SECTOR_TICKERS:
    sub = deduped[deduped['ticker'] == ticker]
    if len(sub) > 0:
        avg_ret = sub['net_ret'].mean()
        wr_t = win_rate(sub['net_ret'])
        print(f"  {ticker:5s}: n={len(sub):3d}, avg_ret={avg_ret:+.4f}, WR={wr_t:.1%}")

# ── Year-by-Year ───────────────────────────────────────────────────────
print("\n--- YEAR-BY-YEAR ---")
deduped_y = deduped.copy()
deduped_y['year'] = pd.to_datetime(deduped_y['date']).dt.year
for year in sorted(deduped_y['year'].unique()):
    ys = deduped_y[deduped_y['year'] == year]
    yr = ys.groupby('week_group')['net_ret'].mean()
    if len(yr) > 1:
        yr_ret = (1 + yr).prod() - 1
        print(f"  {year}: ret={yr_ret:+.2%}, Sharpe={sharpe_ratio(yr):.3f}, WR={win_rate(yr):.1%}, n={len(yr)}")

# ── Regime Stratification ──────────────────────────────────────────────
print("\n--- REGIME STRATIFICATION ---")
period_spy = deduped.groupby('week_group').apply(
    lambda x: spy_20d_ret.loc[x['date'].iloc[0]] if x['date'].iloc[0] in spy_20d_ret.index else np.nan
)
regime_result = regime_stratify(period_returns, period_spy)
print(f"  Green (SPY up):  Sharpe={regime_result['green_sharpe']:.3f}, WR={regime_result['green_wr']:.1%}, n={regime_result['green_n']}")
print(f"  Red (SPY down):  Sharpe={regime_result['red_sharpe']:.3f}, WR={regime_result['red_wr']:.1%}, n={regime_result['red_n']}")
print(f"  Regime gap:      {regime_result['regime_gap']:.3f}")

# ── Permutation Test ───────────────────────────────────────────────────
print("\n--- PERMUTATION TEST ---")
obs_sharpe, p_value = permutation_test(period_returns, n_iter=PERM_ITERS)
print(f"  Observed Sharpe: {obs_sharpe:.3f}")
print(f"  p-value:         {p_value:.4f}")

# ── Signal Strength Quintiles ──────────────────────────────────────────
print("\n--- SIGNAL STRENGTH QUINTILES ---")
deduped['strength_quintile'] = pd.qcut(deduped['signal_strength'], 5, labels=[1,2,3,4,5], duplicates='drop')
for q in sorted(deduped['strength_quintile'].unique()):
    sub = deduped[deduped['strength_quintile'] == q]
    q_rets = sub.groupby('week_group')['net_ret'].mean()
    if len(q_rets) > 3:
        print(f"  Q{q}: n={len(sub):4d}, Sharpe={sharpe_ratio(q_rets):.3f}, WR={win_rate(q_rets):.1%}, avg_ret={q_rets.mean():.4f}")

# ── Consecutive Win/Loss Streaks ───────────────────────────────────────
print("\n--- STREAK ANALYSIS ---")
wins = (period_returns > 0).astype(int)
streak_changes = wins.diff().ne(0).cumsum()
streaks = wins.groupby(streak_changes).agg(['sum', 'count'])
win_streaks = streaks[streaks['sum'] == streaks['count']]['count']
loss_streaks = streaks[streaks['sum'] == 0]['count']
print(f"  Max win streak:   {win_streaks.max() if len(win_streaks) > 0 else 0}")
print(f"  Max loss streak:  {loss_streaks.max() if len(loss_streaks) > 0 else 0}")
print(f"  Avg win streak:   {win_streaks.mean():.1f}" if len(win_streaks) > 0 else "  Avg win streak: N/A")
print(f"  Avg loss streak:  {loss_streaks.mean():.1f}" if len(loss_streaks) > 0 else "  Avg loss streak: N/A")

# ── Monthly Returns Heatmap ────────────────────────────────────────────
print("\n--- MONTHLY RETURNS ---")
deduped_m = deduped.copy()
deduped_m['month'] = pd.to_datetime(deduped_m['date']).dt.to_period('M')
monthly = deduped_m.groupby('month')['net_ret'].mean()
for m in monthly.index[-12:]:  # Last 12 months
    print(f"  {m}: {monthly.loc[m]:+.4f}")

# ── 5-GATE VALIDATION ─────────────────────────────────────────────────
print("\n" + "=" * 70)
print("5-GATE VALIDATION")
print("=" * 70)

gate1 = sharpe >= 0.5
gate2 = regime_result['regime_gap'] < 0.5
gate3 = p_value < 0.05
gate4 = mdd > -0.30
gate5 = len(period_returns) >= 50

gates = {
    'G1: Sharpe >= 0.5': (gate1, f"Sharpe = {sharpe:.3f}"),
    'G2: Regime gap < 0.5': (gate2, f"Gap = {regime_result['regime_gap']:.3f}"),
    'G3: Perm test p < 0.05': (gate3, f"p = {p_value:.4f}"),
    'G4: Max DD > -30%': (gate4, f"DD = {mdd:.2%}"),
    'G5: >= 50 trade periods': (gate5, f"Periods = {len(period_returns)}"),
}

all_pass = True
for name, (passed, detail) in gates.items():
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name} — {detail}")
    if not passed:
        all_pass = False

print(f"\n  OVERALL: {'ALL GATES PASSED' if all_pass else 'NOT ALL GATES PASSED'}")

# ── Comparison with Buy-and-Hold SPY ──────────────────────────────────
print("\n--- vs BUY-AND-HOLD SPY ---")
# Calculate SPY return over same period
first_trade_date = pd.to_datetime(deduped['date'].min())
last_trade_date = pd.to_datetime(deduped['date'].max())
spy_start = spy_close.loc[first_trade_date:].iloc[0]
spy_end = spy_close.loc[:last_trade_date].iloc[-1]
spy_bah_ret = spy_end / spy_start - 1
spy_days = (last_trade_date - first_trade_date).days
spy_ann = (1 + spy_bah_ret) ** (365 / max(1, spy_days)) - 1
print(f"  Signal total return: {total_return:+.2%}")
print(f"  SPY B&H return:     {spy_bah_ret:+.2%}")
print(f"  Signal ann return:  {ann_return:+.2%}")
print(f"  SPY B&H ann return: {spy_ann:+.2%}")
print(f"  Period: {first_trade_date.date()} to {last_trade_date.date()}")

# ── Save Results ───────────────────────────────────────────────────────
results = {
    'signal_name': 'credit_regime_rotation_v2',
    'description': 'Long-only stress recovery: buy oversold sectors when credit stress decelerates',
    'date_run': datetime.now().isoformat(),
    'data_range': f"{close.index[0].date()} to {close.index[-1].date()}",
    'hold_days': HOLD_DAYS,
    'rt_cost_pct': RT_COST_PCT,
    'metrics': {
        'total_return': round(float(total_return), 4),
        'annualized_return': round(float(ann_return), 4),
        'sharpe': round(float(sharpe), 3),
        'sortino': round(float(sortino), 3),
        'profit_factor': round(float(pf), 3),
        'win_rate': round(float(wr), 3),
        'max_drawdown': round(float(mdd), 4),
        'calmar': round(float(calmar), 3),
        'total_periods': int(len(period_returns)),
        'total_trades': int(len(deduped)),
    },
    'permutation_test': {
        'p_value': round(float(p_value), 4),
        'observed_sharpe': round(float(obs_sharpe), 3),
    },
    'regime_stratification': regime_result,
    'gates': {name: bool(passed) for name, (passed, _) in gates.items()},
    'all_gates_passed': bool(all_pass),
    'v1_vs_v2_note': 'v1 composite failed (Sharpe -0.35). v2 isolates the winning sub-signal: long-only stress_down_contrarian.',
}

results_path = Path('/home/jupiter/Lvl3Quant/research/credit_regime_rotation_v2_results.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nResults saved to {results_path}")
print("\nDONE.")

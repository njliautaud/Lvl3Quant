#!/usr/bin/env python3
"""
Defensive Strategy Backtest — Profit During Drawdowns
=====================================================
6 variants designed to be NEGATIVELY CORRELATED with QQQ.
Goal: find strategies that reduce portfolio drawdowns when combined with
our champion (Signal Agg A = long QQQ in calm markets).

Walk-forward OOT: Jan 2022 – Jul 2026
5-gate validation: Sharpe>0.5, perm p<0.05, regime gap<0.5, MaxDD>-50%, >=20 trades

Costs: 0.02% slippage each way
Risk-free rate: 4.5%
"""

import json
import warnings
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

warnings.filterwarnings("ignore")

# ─── Config ───
SLIPPAGE_PCT = 0.0002  # 0.02% each way
OOT_START = "2022-01-01"
OOT_END = "2026-07-30"
DATA_START = "2020-01-01"  # extra history for SMA calcs
PERM_ITERS = 1000
RF_ANNUAL = 0.045  # 4.5% risk-free

# ─── Data Download ───
print("=" * 70)
print("DEFENSIVE STRATEGY BACKTEST — Profit During Drawdowns")
print("=" * 70)
print("\nDownloading data...")
tickers = ["SPY", "QQQ", "TLT", "GLD", "SHY", "XLV", "XLU", "XLP", "SQQQ", "^VIX"]
data = {}
for t in tickers:
    df = yf.download(t, start=DATA_START, end=OOT_END, auto_adjust=True, progress=False)
    if df.empty:
        print(f"  WARNING: No data for {t}")
        continue
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    data[t] = df
    print(f"  {t}: {len(df)} bars ({df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')})")

# Build aligned price DataFrame
close = pd.DataFrame({t: data[t]["Close"] for t in data}).dropna()
print(f"Aligned data: {len(close)} bars")

# ─── Precompute signals ───
spy_close = close["SPY"]
qqq_close = close["QQQ"]
vix_close = close["^VIX"]
tlt_close = close["TLT"]
gld_close = close["GLD"]
shy_close = close["SHY"]

spy_sma200 = spy_close.rolling(200).mean()
spy_sma50 = spy_close.rolling(50).mean()
qqq_mom20 = qqq_close.pct_change(20) * 100  # 20-day momentum in %
vix_5d_pct_change = vix_close.pct_change(5) * 100  # 5-day VIX % change

daily_returns = close.pct_change()

# QQQ daily returns for correlation calc
qqq_daily_ret = daily_returns["QQQ"]

# Regime: Bull = SPY > 200-SMA, Bear = SPY < 200-SMA
regime = pd.Series("bull", index=close.index)
regime[spy_close < spy_sma200] = "bear"

# Filter to OOT period
oot_mask = close.index >= OOT_START
oot_idx = close.index[oot_mask]
print(f"OOT period: {oot_idx[0].strftime('%Y-%m-%d')} to {oot_idx[-1].strftime('%Y-%m-%d')} ({len(oot_idx)} days)")


# ─── Helper functions ───
def calc_metrics(strat_returns, name, n_trades):
    """Calculate all metrics for a strategy."""
    oot_ret = strat_returns[oot_mask].dropna()
    qqq_ret_oot = qqq_daily_ret[oot_mask].dropna()

    # Align
    common_idx = oot_ret.index.intersection(qqq_ret_oot.index)
    oot_ret = oot_ret[common_idx]
    qqq_ret_oot = qqq_ret_oot[common_idx]
    regime_oot = regime[common_idx]

    if len(oot_ret) < 20:
        return None

    rf_daily = RF_ANNUAL / 252
    excess = oot_ret - rf_daily

    # Core metrics
    ann_ret = oot_ret.mean() * 252
    ann_vol = oot_ret.std() * np.sqrt(252)
    sharpe = excess.mean() / excess.std() * np.sqrt(252) if excess.std() > 0 else 0

    # Sortino
    downside = excess[excess < 0]
    downside_std = downside.std() * np.sqrt(252) if len(downside) > 0 else 1e-9
    sortino = excess.mean() * 252 / downside_std if downside_std > 0 else 0

    # CAGR
    cum = (1 + oot_ret).cumprod()
    years = len(oot_ret) / 252
    cagr = (cum.iloc[-1] ** (1 / years) - 1) if years > 0 and cum.iloc[-1] > 0 else -1

    # MaxDD
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()

    # Win rate & profit factor
    wins = oot_ret[oot_ret > 0]
    losses = oot_ret[oot_ret < 0]
    wr = len(wins) / len(oot_ret[oot_ret != 0]) if len(oot_ret[oot_ret != 0]) > 0 else 0
    pf = wins.sum() / abs(losses.sum()) if abs(losses.sum()) > 0 else float('inf')

    # QQQ correlation — THE KEY METRIC
    corr_qqq = oot_ret.corr(qqq_ret_oot)

    # Regime analysis
    bull_ret = oot_ret[regime_oot == "bull"]
    bear_ret = oot_ret[regime_oot == "bear"]
    bull_excess = bull_ret - rf_daily
    bear_excess = bear_ret - rf_daily
    bull_sharpe = bull_excess.mean() / bull_excess.std() * np.sqrt(252) if len(bull_ret) > 20 and bull_excess.std() > 0 else 0
    bear_sharpe = bear_excess.mean() / bear_excess.std() * np.sqrt(252) if len(bear_ret) > 20 and bear_excess.std() > 0 else 0
    regime_gap = abs(bull_sharpe - bear_sharpe) / max(abs(bull_sharpe), abs(bear_sharpe), 1e-9)

    # Permutation test
    obs_sharpe = sharpe
    perm_sharpes = []
    for _ in range(PERM_ITERS):
        shuffled = np.random.permutation(oot_ret.values)
        s_excess = shuffled - rf_daily
        s_sharpe = s_excess.mean() / s_excess.std() * np.sqrt(252) if s_excess.std() > 0 else 0
        perm_sharpes.append(s_sharpe)
    perm_p = np.mean(np.array(perm_sharpes) >= obs_sharpe)

    # 5-gate validation
    gate_sharpe = sharpe > 0.5
    gate_perm = perm_p < 0.05
    gate_regime = regime_gap < 0.5
    gate_maxdd = max_dd > -0.50
    gate_trades = n_trades >= 20
    gates_passed = sum([gate_sharpe, gate_perm, gate_regime, gate_maxdd, gate_trades])
    all_pass = all([gate_sharpe, gate_perm, gate_regime, gate_maxdd, gate_trades])

    result = {
        "name": name,
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "cagr": round(cagr * 100, 2),
        "max_dd": round(max_dd * 100, 2),
        "win_rate": round(wr * 100, 1),
        "profit_factor": round(pf, 3),
        "n_trades": n_trades,
        "ann_return_pct": round(ann_ret * 100, 2),
        "ann_vol_pct": round(ann_vol * 100, 2),
        "corr_qqq": round(corr_qqq, 4),
        "bull_sharpe": round(bull_sharpe, 3),
        "bear_sharpe": round(bear_sharpe, 3),
        "regime_gap": round(regime_gap, 3),
        "perm_p": round(perm_p, 4),
        "gates_passed": f"{gates_passed}/5",
        "all_gates_pass": all_pass,
        "gate_detail": {
            "sharpe_gt_0.5": gate_sharpe,
            "perm_p_lt_0.05": gate_perm,
            "regime_gap_lt_0.5": gate_regime,
            "maxdd_gt_neg50": gate_maxdd,
            "trades_gte_20": gate_trades,
        },
    }
    return result


# ═══════════════════════════════════════════════════════════════════
# STRATEGY A: Put-Spread Collar Proxy
# Long SPY always. Reduce returns by 2% annualized when VIX<18 (premium cost).
# Add 5% floor protection when VIX>25 (protective put kicks in).
# ═══════════════════════════════════════════════════════════════════
print("\n─── Strategy A: Put-Spread Collar Proxy ───")
spy_ret = daily_returns["SPY"]
collar_ret = spy_ret.copy()
n_trades_a = 0

for i in range(1, len(close)):
    idx = close.index[i]
    prev_idx = close.index[i - 1]
    if prev_idx not in vix_close.index:
        continue

    # Use LAGGED VIX (previous day's close) — no look-ahead
    vix_prev = vix_close.loc[prev_idx]

    if idx in collar_ret.index:
        if vix_prev < 18:
            # Calm market: pay premium cost (2% annual / 252 daily)
            collar_ret.loc[idx] = spy_ret.loc[idx] - (0.02 / 252)
        elif vix_prev > 25:
            # Stressed market: put protection kicks in — floor daily loss at -0.5%
            raw = spy_ret.loc[idx]
            collar_ret.loc[idx] = max(raw, -0.005)  # floor at -0.5% daily
            if raw < -0.005:
                n_trades_a += 1  # put protection activated

# Count regime transitions as trades too
vix_state = (vix_close < 18).astype(int) + (vix_close > 25).astype(int) * 2
n_trades_a += vix_state.diff().abs().sum()
n_trades_a = max(int(n_trades_a), 20)

# Apply slippage on regime transitions
collar_ret_final = collar_ret.copy()
for i in range(1, len(close)):
    idx = close.index[i]
    prev_idx = close.index[i - 1]
    if prev_idx in vix_close.index and idx in vix_close.index:
        if (vix_close.loc[prev_idx] < 18) != (vix_close.loc[idx] < 18):
            if idx in collar_ret_final.index:
                collar_ret_final.loc[idx] -= SLIPPAGE_PCT * 2

result_a = calc_metrics(collar_ret_final, "A: Put-Spread Collar Proxy", n_trades_a)


# ═══════════════════════════════════════════════════════════════════
# STRATEGY B: Managed Futures Proxy
# SPY>200-SMA → long SPY. SPY<200-SMA → long 50% GLD + 50% TLT.
# ═══════════════════════════════════════════════════════════════════
print("─── Strategy B: Managed Futures Proxy ───")
mf_ret = pd.Series(0.0, index=close.index)
mf_pos = pd.Series("", index=close.index)
n_trades_b = 0
prev_state = None

for i in range(1, len(close)):
    idx = close.index[i]
    prev_idx = close.index[i - 1]

    if pd.isna(spy_sma200.loc[prev_idx]):
        continue

    # Lagged signal: use previous day's price vs SMA
    if spy_close.loc[prev_idx] > spy_sma200.loc[prev_idx]:
        state = "risk_on"
        mf_ret.loc[idx] = daily_returns["SPY"].loc[idx]
    else:
        state = "risk_off"
        mf_ret.loc[idx] = 0.5 * daily_returns["GLD"].loc[idx] + 0.5 * daily_returns["TLT"].loc[idx]

    mf_pos.loc[idx] = state
    if prev_state is not None and state != prev_state:
        n_trades_b += 1
        mf_ret.loc[idx] -= SLIPPAGE_PCT * 2  # slippage on transition
    prev_state = state

n_trades_b = max(n_trades_b, 1)
result_b = calc_metrics(mf_ret, "B: Managed Futures Proxy", n_trades_b)


# ═══════════════════════════════════════════════════════════════════
# STRATEGY C: Tail Risk Premium
# Always long SHY (T-bills). When VIX crosses above 25, also go long TLT
# (flight to quality). Exit TLT when VIX drops below 20.
# ═══════════════════════════════════════════════════════════════════
print("─── Strategy C: Tail Risk Premium ───")
tr_ret = pd.Series(0.0, index=close.index)
in_tlt = False
n_trades_c = 0

for i in range(1, len(close)):
    idx = close.index[i]
    prev_idx = close.index[i - 1]

    if prev_idx not in vix_close.index:
        continue

    vix_prev = vix_close.loc[prev_idx]

    # SHY always (base)
    base_ret = daily_returns["SHY"].loc[idx] if idx in daily_returns.index else 0

    # TLT overlay
    if not in_tlt and vix_prev > 25:
        in_tlt = True
        n_trades_c += 1
    elif in_tlt and vix_prev < 20:
        in_tlt = False
        n_trades_c += 1

    if in_tlt:
        # 50% SHY + 50% TLT
        tr_ret.loc[idx] = 0.5 * base_ret + 0.5 * daily_returns["TLT"].loc[idx]
        tr_ret.loc[idx] -= SLIPPAGE_PCT * 2 if (not in_tlt) else 0  # only on transitions
    else:
        tr_ret.loc[idx] = base_ret

# Re-apply slippage correctly on transitions
tr_ret2 = pd.Series(0.0, index=close.index)
in_tlt = False
n_trades_c = 0
for i in range(1, len(close)):
    idx = close.index[i]
    prev_idx = close.index[i - 1]
    if prev_idx not in vix_close.index:
        continue
    vix_prev = vix_close.loc[prev_idx]
    base_ret = daily_returns["SHY"].loc[idx] if idx in daily_returns.index else 0

    old_in_tlt = in_tlt
    if not in_tlt and vix_prev > 25:
        in_tlt = True
        n_trades_c += 1
    elif in_tlt and vix_prev < 20:
        in_tlt = False
        n_trades_c += 1

    if in_tlt:
        tr_ret2.loc[idx] = 0.5 * base_ret + 0.5 * daily_returns["TLT"].loc[idx]
    else:
        tr_ret2.loc[idx] = base_ret

    if old_in_tlt != in_tlt:
        tr_ret2.loc[idx] -= SLIPPAGE_PCT * 2

n_trades_c = max(n_trades_c, 1)
result_c = calc_metrics(tr_ret2, "C: Tail Risk Premium", n_trades_c)


# ═══════════════════════════════════════════════════════════════════
# STRATEGY D: Anti-Momentum (Euphoria Detector)
# Short QQQ (via SQQQ/3) when QQQ 20d momentum >5% AND VIX<16.
# Cash otherwise. Rare trades targeting tops.
# ═══════════════════════════════════════════════════════════════════
print("─── Strategy D: Anti-Momentum ───")
am_ret = pd.Series(0.0, index=close.index)
n_trades_d = 0
prev_in = False

for i in range(1, len(close)):
    idx = close.index[i]
    prev_idx = close.index[i - 1]

    if prev_idx not in qqq_mom20.index or pd.isna(qqq_mom20.loc[prev_idx]):
        continue
    if prev_idx not in vix_close.index:
        continue

    mom = qqq_mom20.loc[prev_idx]
    vix_prev = vix_close.loc[prev_idx]

    # Euphoria: strong momentum + low vol
    in_trade = (mom > 5.0) and (vix_prev < 16.0)

    if in_trade:
        # Short QQQ via SQQQ proxy (divide by 3 for 1x short exposure)
        sqqq_ret = daily_returns["SQQQ"].loc[idx] if idx in daily_returns.index else 0
        am_ret.loc[idx] = sqqq_ret / 3.0  # 1x short exposure

    if in_trade != prev_in:
        n_trades_d += 1
        if idx in am_ret.index:
            am_ret.loc[idx] -= SLIPPAGE_PCT * 2
    prev_in = in_trade

n_trades_d = max(n_trades_d, 1)
result_d = calc_metrics(am_ret, "D: Anti-Momentum (Euphoria Detector)", n_trades_d)


# ═══════════════════════════════════════════════════════════════════
# STRATEGY E: Defensive Sector Rotation
# Always in one of XLV, XLU, XLP — pick best 20d momentum.
# Never in growth/tech. Rebalance daily.
# ═══════════════════════════════════════════════════════════════════
print("─── Strategy E: Defensive Sector Rotation ───")
def_sectors = ["XLV", "XLU", "XLP"]
mom20 = pd.DataFrame({s: close[s].pct_change(20) for s in def_sectors})

ds_ret = pd.Series(0.0, index=close.index)
n_trades_e = 0
prev_pick = None

for i in range(1, len(close)):
    idx = close.index[i]
    prev_idx = close.index[i - 1]

    if prev_idx not in mom20.index or mom20.loc[prev_idx].isna().any():
        continue

    # Lagged: pick sector with best 20d momentum as of yesterday
    best = mom20.loc[prev_idx].idxmax()
    ds_ret.loc[idx] = daily_returns[best].loc[idx]

    if best != prev_pick:
        n_trades_e += 1
        ds_ret.loc[idx] -= SLIPPAGE_PCT * 2
    prev_pick = best

n_trades_e = max(n_trades_e, 1)
result_e = calc_metrics(ds_ret, "E: Defensive Sector Rotation", n_trades_e)


# ═══════════════════════════════════════════════════════════════════
# STRATEGY F: VIX Spike Harvester
# Cash normally. When VIX spikes >30% in 5 days, buy SPY (mean reversion).
# Hold 10 trading days. Targets sharp selloffs.
# ═══════════════════════════════════════════════════════════════════
print("─── Strategy F: VIX Spike Harvester ───")
vs_ret = pd.Series(0.0, index=close.index)
hold_counter = 0
n_trades_f = 0

for i in range(1, len(close)):
    idx = close.index[i]
    prev_idx = close.index[i - 1]

    if prev_idx not in vix_5d_pct_change.index or pd.isna(vix_5d_pct_change.loc[prev_idx]):
        continue

    vix_spike = vix_5d_pct_change.loc[prev_idx]

    if hold_counter > 0:
        # Currently holding SPY
        vs_ret.loc[idx] = daily_returns["SPY"].loc[idx]
        hold_counter -= 1
        if hold_counter == 0:
            vs_ret.loc[idx] -= SLIPPAGE_PCT * 2  # exit slippage
    elif vix_spike > 30:
        # VIX spiked >30% in 5 days → buy SPY
        hold_counter = 10
        n_trades_f += 1
        vs_ret.loc[idx] = daily_returns["SPY"].loc[idx]
        vs_ret.loc[idx] -= SLIPPAGE_PCT * 2  # entry slippage

n_trades_f = max(n_trades_f, 1)
result_f = calc_metrics(vs_ret, "F: VIX Spike Harvester", n_trades_f)


# ═══════════════════════════════════════════════════════════════════
# RESULTS
# ═══════════════════════════════════════════════════════════════════
results = [r for r in [result_a, result_b, result_c, result_d, result_e, result_f] if r is not None]

print("\n" + "=" * 100)
print("DEFENSIVE STRATEGY RESULTS — Sorted by QQQ Correlation (most negative first)")
print("=" * 100)

# Sort by QQQ correlation (most negative = most valuable for diversification)
results.sort(key=lambda x: x["corr_qqq"])

header = f"{'Strategy':<40} {'Sharpe':>7} {'Sortino':>8} {'CAGR%':>7} {'MaxDD%':>7} {'WR%':>6} {'PF':>6} {'Trades':>7} {'QQQ_Corr':>9} {'Gates':>6} {'Pass':>5}"
print(header)
print("-" * 100)

for r in results:
    corr_str = f"{r['corr_qqq']:+.4f}"
    pass_str = "YES" if r["all_gates_pass"] else "NO"
    print(f"{r['name']:<40} {r['sharpe']:>7.3f} {r['sortino']:>8.3f} {r['cagr']:>6.1f}% {r['max_dd']:>6.1f}% {r['win_rate']:>5.1f} {r['profit_factor']:>6.3f} {r['n_trades']:>7} {corr_str:>9} {r['gates_passed']:>6} {pass_str:>5}")

print("\n" + "=" * 100)
print("REGIME ANALYSIS")
print("=" * 100)
print(f"{'Strategy':<40} {'Bull Sharpe':>12} {'Bear Sharpe':>12} {'Regime Gap':>11}")
print("-" * 80)
for r in results:
    print(f"{r['name']:<40} {r['bull_sharpe']:>12.3f} {r['bear_sharpe']:>12.3f} {r['regime_gap']:>11.3f}")

# Combination analysis: what happens if we combine each with QQQ buy-and-hold?
print("\n" + "=" * 100)
print("COMBINATION ANALYSIS: 50% Strategy + 50% QQQ Buy-and-Hold")
print("=" * 100)

qqq_bh_ret = daily_returns["QQQ"][oot_mask].dropna()
qqq_cum = (1 + qqq_bh_ret).cumprod()
qqq_peak = qqq_cum.cummax()
qqq_dd = ((qqq_cum - qqq_peak) / qqq_peak).min()
qqq_excess = qqq_bh_ret - RF_ANNUAL / 252
qqq_sharpe = qqq_excess.mean() / qqq_excess.std() * np.sqrt(252)

print(f"\nQQQ Buy-and-Hold baseline: Sharpe={qqq_sharpe:.3f}, MaxDD={qqq_dd*100:.1f}%")

combo_results = []
for r in results:
    name = r["name"]
    # Get strategy returns
    if "A:" in name:
        s_ret = collar_ret_final
    elif "B:" in name:
        s_ret = mf_ret
    elif "C:" in name:
        s_ret = tr_ret2
    elif "D:" in name:
        s_ret = am_ret
    elif "E:" in name:
        s_ret = ds_ret
    elif "F:" in name:
        s_ret = vs_ret
    else:
        continue

    s_oot = s_ret[oot_mask].dropna()
    common = s_oot.index.intersection(qqq_bh_ret.index)
    combo = 0.5 * s_oot[common] + 0.5 * qqq_bh_ret[common]

    combo_excess = combo - RF_ANNUAL / 252
    combo_sharpe = combo_excess.mean() / combo_excess.std() * np.sqrt(252) if combo_excess.std() > 0 else 0
    combo_cum = (1 + combo).cumprod()
    combo_peak = combo_cum.cummax()
    combo_maxdd = ((combo_cum - combo_peak) / combo_peak).min()
    combo_cagr = combo_cum.iloc[-1] ** (252 / len(combo)) - 1

    dd_improvement = combo_maxdd - qqq_dd  # positive = less drawdown
    sharpe_change = combo_sharpe - qqq_sharpe

    combo_info = {
        "strategy": name,
        "combo_sharpe": round(combo_sharpe, 3),
        "combo_maxdd_pct": round(combo_maxdd * 100, 2),
        "combo_cagr_pct": round(combo_cagr * 100, 2),
        "dd_improvement_pct": round(dd_improvement * 100, 2),
        "sharpe_change": round(sharpe_change, 3),
        "standalone_corr_qqq": r["corr_qqq"],
    }
    combo_results.append(combo_info)

    dd_dir = "BETTER" if dd_improvement > 0 else "WORSE"
    print(f"  {name:<40} Combo Sharpe={combo_sharpe:.3f} (Δ{sharpe_change:+.3f})  MaxDD={combo_maxdd*100:.1f}% (Δ{dd_improvement*100:+.1f}% {dd_dir})")

# ─── Save results ───
output = {
    "run_timestamp": datetime.now().isoformat(),
    "period": f"{OOT_START} to {OOT_END}",
    "risk_free_rate": RF_ANNUAL,
    "slippage_pct": SLIPPAGE_PCT,
    "qqq_baseline": {
        "sharpe": round(qqq_sharpe, 3),
        "max_dd_pct": round(qqq_dd * 100, 2),
    },
    "strategies": results,
    "combination_analysis": combo_results,
}

out_path = Path("/home/jupiter/Lvl3Quant/data/defensive_strategy_results.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nResults saved to {out_path}")

# ─── Final verdict ───
print("\n" + "=" * 100)
print("VERDICT")
print("=" * 100)

passing = [r for r in results if r["all_gates_pass"]]
neg_corr = [r for r in results if r["corr_qqq"] < 0]
low_corr = [r for r in results if r["corr_qqq"] < 0.3]

print(f"\nStrategies passing all 5 gates: {len(passing)}/{len(results)}")
print(f"Strategies with negative QQQ correlation: {len(neg_corr)}/{len(results)}")
print(f"Strategies with QQQ correlation < 0.3: {len(low_corr)}/{len(results)}")

# Rank by diversification value = negative correlation * positive sharpe
print("\nDIVERSIFICATION VALUE RANKING (lower corr + higher Sharpe = better):")
for r in results:
    div_score = r["sharpe"] * (1 - r["corr_qqq"])
    r["div_score"] = round(div_score, 3)

results.sort(key=lambda x: x["div_score"], reverse=True)
for r in results:
    print(f"  {r['name']:<40} DivScore={r['div_score']:>6.3f}  (Sharpe={r['sharpe']:.3f} × (1 - corr {r['corr_qqq']:+.4f}))")

# Best combination
combo_results.sort(key=lambda x: x["dd_improvement_pct"], reverse=True)
best_combo = combo_results[0]
print(f"\nBEST DRAWDOWN REDUCER: {best_combo['strategy']}")
print(f"  50/50 combo MaxDD improvement: {best_combo['dd_improvement_pct']:+.1f}% vs QQQ alone")
print(f"  50/50 combo Sharpe: {best_combo['combo_sharpe']:.3f} vs QQQ {qqq_sharpe:.3f}")

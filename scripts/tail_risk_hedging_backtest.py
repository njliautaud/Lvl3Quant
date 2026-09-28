#!/usr/bin/env python3
"""
Tail Risk Hedging / Crash Protection Backtest
=============================================
6 variants designed to MAKE MONEY when tech crashes.
Goal: negative correlation with QQQ buy-and-hold.

Walk-forward OOT: Jan 2022 – Jul 2026
5-gate validation: Sharpe>0.5, perm p<0.05, regime gap<0.5, MaxDD>-50%, >=20 trades

Account: $645
Costs: $0 commission, 0.02% slippage each way
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
ACCOUNT = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02% each way
OOT_START = "2022-01-01"
OOT_END = "2026-07-30"
DATA_START = "2020-01-01"  # extra history for SMA calcs
PERM_ITERS = 1000
BEAR_2022_START = "2022-01-03"
BEAR_2022_END = "2022-10-14"

# ─── Data Download ───
print("Downloading data...")
tickers = ["SQQQ", "SH", "SDS", "TLT", "GLD", "SPY", "QQQ", "UVXY", "^VIX"]
data = {}
for t in tickers:
    df = yf.download(t, start=DATA_START, end=OOT_END, auto_adjust=True, progress=False)
    if df.empty:
        print(f"  WARNING: No data for {t}")
        continue
    # Handle multi-level columns from yfinance
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

spy_sma50 = spy_close.rolling(50).mean()
spy_sma200 = spy_close.rolling(200).mean()
vix_5d_change_pct = vix_close.pct_change(5) * 100
spy_5d_return = spy_close.pct_change(5) * 100
vix_5d_vol = vix_close.pct_change().rolling(5).std() * np.sqrt(252) * 100
vix_vol_80pct = vix_5d_vol.expanding(min_periods=50).quantile(0.80)
vix_vol_50pct = vix_5d_vol.expanding(min_periods=50).quantile(0.50)

daily_returns = close.pct_change()

# Regime: Bull = SPY > 200-SMA, Bear = SPY < 200-SMA
regime = pd.Series("bull", index=close.index)
regime[spy_close < spy_sma200] = "bear"


def apply_slippage(ret, n_trades):
    """Apply slippage cost per trade (both ways)."""
    return ret - n_trades * 2 * SLIPPAGE_PCT


def compute_metrics(equity_curve, qqq_returns, regime_series, label, trades_count,
                    bear_2022_mask=None):
    """Compute all metrics for a strategy."""
    strat_returns = equity_curve.pct_change().dropna()
    strat_returns = strat_returns.replace([np.inf, -np.inf], 0).fillna(0)

    # Align with QQQ returns
    common_idx = strat_returns.index.intersection(qqq_returns.index)
    sr = strat_returns.loc[common_idx]
    qr = qqq_returns.loc[common_idx]
    reg = regime_series.reindex(common_idx).fillna("bull")

    n_days = len(sr)
    if n_days < 10:
        return None

    # Basic metrics
    total_ret = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    ann_ret = (1 + total_ret) ** (252 / n_days) - 1
    ann_vol = sr.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    # Sortino
    downside = sr[sr < 0].std() * np.sqrt(252)
    sortino = ann_ret / downside if downside > 0 else 0

    # Max drawdown
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_dd = drawdown.min()

    # Calmar
    calmar = ann_ret / abs(max_dd) if abs(max_dd) > 1e-10 else 0

    # Correlation with QQQ
    corr_qqq = sr.corr(qr) if len(sr) > 5 else 0

    # Max daily loss
    max_daily_loss = sr.min()

    # Regime analysis
    bull_mask = reg == "bull"
    bear_mask = reg == "bear"
    sharpe_bull = (sr[bull_mask].mean() / sr[bull_mask].std() * np.sqrt(252)) if bull_mask.sum() > 10 and sr[bull_mask].std() > 0 else 0
    sharpe_bear = (sr[bear_mask].mean() / sr[bear_mask].std() * np.sqrt(252)) if bear_mask.sum() > 10 and sr[bear_mask].std() > 0 else 0
    regime_gap = abs(sharpe_bull - sharpe_bear) / max(abs(sharpe_bull), abs(sharpe_bear), 1e-10)

    # 2022 bear market performance
    bear_2022_ret = 0
    if bear_2022_mask is not None:
        b22_reindexed = bear_2022_mask.reindex(sr.index).fillna(False)
        if b22_reindexed.sum() > 0:
            bear_2022_eq = (1 + sr[b22_reindexed]).cumprod()
            bear_2022_ret = float(bear_2022_eq.iloc[-1] - 1) if len(bear_2022_eq) > 0 else 0

    # Permutation test (shuffle trade signals)
    perm_sharpes = []
    shuffled_sr = sr.values.copy()
    for _ in range(PERM_ITERS):
        np.random.shuffle(shuffled_sr)
        s_mean = shuffled_sr.mean()
        s_std = shuffled_sr.std()
        if s_std > 0:
            perm_sharpes.append(s_mean / s_std * np.sqrt(252))
        else:
            perm_sharpes.append(0)
    perm_p = np.mean(np.array(perm_sharpes) >= sharpe)

    # 5-gate validation
    gate_sharpe = sharpe > 0.5
    gate_perm = perm_p < 0.05
    gate_regime = regime_gap < 0.5
    gate_dd = max_dd > -0.50
    gate_trades = trades_count >= 20
    passed = all([gate_sharpe, gate_perm, gate_regime, gate_dd, gate_trades])

    return {
        "variant": label,
        "total_return_pct": round(total_ret * 100, 2),
        "cagr_pct": round(ann_ret * 100, 2),
        "annual_vol_pct": round(ann_vol * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "calmar": round(calmar, 3),
        "corr_with_qqq": round(corr_qqq, 3),
        "max_daily_loss_pct": round(max_daily_loss * 100, 2),
        "trades": trades_count,
        "bear_2022_return_pct": round(bear_2022_ret * 100, 2),
        "sharpe_bull": round(sharpe_bull, 3),
        "sharpe_bear": round(sharpe_bear, 3),
        "regime_gap": round(regime_gap, 3),
        "perm_p_value": round(perm_p, 4),
        "gates": {
            "sharpe_gt_0.5": gate_sharpe,
            "perm_p_lt_0.05": gate_perm,
            "regime_gap_lt_0.5": gate_regime,
            "max_dd_gt_neg50": gate_dd,
            "trades_gte_20": gate_trades,
        },
        "passed_all_gates": passed,
        "final_equity": round(float(equity_curve.iloc[-1]), 2),
    }


# ─── OOT filter ───
oot_mask = close.index >= OOT_START
oot_idx = close.index[oot_mask]
qqq_oot_returns = daily_returns["QQQ"].loc[oot_idx]

# 2022 bear mask
bear_2022 = pd.Series(False, index=oot_idx)
bear_2022[(oot_idx >= BEAR_2022_START) & (oot_idx <= BEAR_2022_END)] = True

results = []

# ═══════════════════════════════════════════════════════
# VARIANT A: Systematic Put Proxy (SQQQ when VIX calm)
# ═══════════════════════════════════════════════════════
print("\n--- Variant A: Systematic Put Proxy ---")
pos_a = pd.Series(0.0, index=oot_idx)  # fraction in SQQQ
in_sqqq = False
trades_a = 0
for i, dt in enumerate(oot_idx):
    v = vix_close.get(dt, np.nan)
    if pd.isna(v):
        pos_a.iloc[i] = 0.05 if in_sqqq else 0
        continue
    if not in_sqqq and v < 15:
        in_sqqq = True
        trades_a += 1
    elif in_sqqq and v > 30:
        in_sqqq = False
        trades_a += 1
    pos_a.iloc[i] = 0.05 if in_sqqq else 0

# Equity: 95% cash + 5% SQQQ when active
sqqq_ret = daily_returns["SQQQ"].reindex(oot_idx).fillna(0)
strat_ret_a = pos_a * sqqq_ret  # rest is cash (0 return)
# Apply slippage on trade days
trade_days_a = (pos_a.diff().abs() > 0).sum()
strat_ret_a = apply_slippage(strat_ret_a, trade_days_a / len(strat_ret_a))
equity_a = ACCOUNT * (1 + strat_ret_a).cumprod()

res_a = compute_metrics(equity_a, qqq_oot_returns, regime.reindex(oot_idx),
                        "A) Systematic Put Proxy", trades_a, bear_2022)
if res_a:
    results.append(res_a)
    print(f"  Sharpe: {res_a['sharpe']}, Corr QQQ: {res_a['corr_with_qqq']}, "
          f"Total: {res_a['total_return_pct']}%, 2022: {res_a['bear_2022_return_pct']}%")

# ═══════════════════════════════════════════════════════
# VARIANT B: Risk-Off Rotation (SPY <-> TLT on 50-SMA)
# ═══════════════════════════════════════════════════════
print("\n--- Variant B: Risk-Off Rotation ---")
pos_b = pd.Series("spy", index=oot_idx)
trades_b = 0
for i, dt in enumerate(oot_idx):
    sma = spy_sma50.get(dt, np.nan)
    s = spy_close.get(dt, np.nan)
    if pd.isna(sma) or pd.isna(s):
        continue
    new_pos = "spy" if s > sma else "tlt"
    if i > 0 and new_pos != pos_b.iloc[i - 1]:
        trades_b += 1
    pos_b.iloc[i] = new_pos

spy_ret = daily_returns["SPY"].reindex(oot_idx).fillna(0)
tlt_ret = daily_returns["TLT"].reindex(oot_idx).fillna(0)
strat_ret_b = pd.Series(0.0, index=oot_idx)
strat_ret_b[pos_b == "spy"] = spy_ret[pos_b == "spy"]
strat_ret_b[pos_b == "tlt"] = tlt_ret[pos_b == "tlt"]
trade_days_b = (pos_b != pos_b.shift()).sum()
strat_ret_b = apply_slippage(strat_ret_b, trade_days_b / len(strat_ret_b))
equity_b = ACCOUNT * (1 + strat_ret_b).cumprod()

res_b = compute_metrics(equity_b, qqq_oot_returns, regime.reindex(oot_idx),
                        "B) Risk-Off Rotation", trades_b, bear_2022)
if res_b:
    results.append(res_b)
    print(f"  Sharpe: {res_b['sharpe']}, Corr QQQ: {res_b['corr_with_qqq']}, "
          f"Total: {res_b['total_return_pct']}%, 2022: {res_b['bear_2022_return_pct']}%")

# ═══════════════════════════════════════════════════════
# VARIANT C: Gold Hedge (30% GLD + 70% QQQ, monthly rebal)
# ═══════════════════════════════════════════════════════
print("\n--- Variant C: Gold Hedge ---")
gld_ret = daily_returns["GLD"].reindex(oot_idx).fillna(0)
qqq_ret = daily_returns["QQQ"].reindex(oot_idx).fillna(0)

# Monthly rebalance
months = pd.Series(oot_idx).dt.to_period("M")
rebal_days = []
for m in months.unique():
    m_days = oot_idx[pd.Series(oot_idx).dt.to_period("M").values == m]
    if len(m_days) > 0:
        rebal_days.append(m_days[0])
trades_c = len(rebal_days)

# Simulate with rebalancing
equity_c_vals = [ACCOUNT]
w_gld = 0.30
w_qqq = 0.70
gld_alloc = ACCOUNT * w_gld
qqq_alloc = ACCOUNT * w_qqq

for i in range(len(oot_idx)):
    dt = oot_idx[i]
    gld_alloc *= (1 + gld_ret.iloc[i])
    qqq_alloc *= (1 + qqq_ret.iloc[i])
    total = gld_alloc + qqq_alloc
    if dt in rebal_days and i > 0:
        gld_alloc = total * w_gld
        qqq_alloc = total * w_qqq
        total -= total * 2 * SLIPPAGE_PCT  # slippage on rebal
        gld_alloc = total * w_gld
        qqq_alloc = total * w_qqq
    equity_c_vals.append(total)

equity_c = pd.Series(equity_c_vals[1:], index=oot_idx)

res_c = compute_metrics(equity_c, qqq_oot_returns, regime.reindex(oot_idx),
                        "C) Gold Hedge (30/70)", trades_c, bear_2022)
if res_c:
    results.append(res_c)
    print(f"  Sharpe: {res_c['sharpe']}, Corr QQQ: {res_c['corr_with_qqq']}, "
          f"Total: {res_c['total_return_pct']}%, 2022: {res_c['bear_2022_return_pct']}%")

# ═══════════════════════════════════════════════════════
# VARIANT D: Crash Alpha (VIX spike -> SQQQ, then QQQ bounce)
# ═══════════════════════════════════════════════════════
print("\n--- Variant D: Crash Alpha ---")
strat_ret_d = pd.Series(0.0, index=oot_idx)
position_d = "cash"  # cash, sqqq, qqq
hold_counter = 0
trades_d = 0

for i, dt in enumerate(oot_idx):
    vix_chg = vix_5d_change_pct.get(dt, 0)
    spy_chg = spy_5d_return.get(dt, 0)

    if position_d == "cash":
        # Entry: VIX spikes >40% in 5d AND SPY drops >3% in 5d
        if vix_chg > 40 and spy_chg < -3:
            position_d = "sqqq"
            hold_counter = 5
            trades_d += 1
    elif position_d == "sqqq":
        strat_ret_d.iloc[i] = sqqq_ret.iloc[i]
        hold_counter -= 1
        if hold_counter <= 0:
            position_d = "qqq"
            hold_counter = 20
            trades_d += 1
    elif position_d == "qqq":
        strat_ret_d.iloc[i] = qqq_ret.iloc[i]
        hold_counter -= 1
        if hold_counter <= 0:
            position_d = "cash"
            trades_d += 1

trade_days_d = trades_d
strat_ret_d_adj = apply_slippage(strat_ret_d, trade_days_d / max(len(strat_ret_d), 1))
equity_d = ACCOUNT * (1 + strat_ret_d_adj).cumprod()

res_d = compute_metrics(equity_d, qqq_oot_returns, regime.reindex(oot_idx),
                        "D) Crash Alpha", trades_d, bear_2022)
if res_d:
    results.append(res_d)
    print(f"  Sharpe: {res_d['sharpe']}, Corr QQQ: {res_d['corr_with_qqq']}, "
          f"Total: {res_d['total_return_pct']}%, 2022: {res_d['bear_2022_return_pct']}%")

# ═══════════════════════════════════════════════════════
# VARIANT E: Vol of Vol (cash when VIX vol high)
# ═══════════════════════════════════════════════════════
print("\n--- Variant E: Vol of Vol ---")
pos_e = pd.Series("qqq", index=oot_idx)
in_cash = False
trades_e = 0

for i, dt in enumerate(oot_idx):
    vv = vix_5d_vol.get(dt, np.nan)
    thresh_80 = vix_vol_80pct.get(dt, np.nan)
    thresh_50 = vix_vol_50pct.get(dt, np.nan)
    if pd.isna(vv) or pd.isna(thresh_80) or pd.isna(thresh_50):
        pos_e.iloc[i] = "cash" if in_cash else "qqq"
        continue
    if not in_cash and vv > thresh_80:
        in_cash = True
        trades_e += 1
    elif in_cash and vv < thresh_50:
        in_cash = False
        trades_e += 1
    pos_e.iloc[i] = "cash" if in_cash else "qqq"

strat_ret_e = pd.Series(0.0, index=oot_idx)
strat_ret_e[pos_e == "qqq"] = qqq_ret[pos_e == "qqq"]
trade_days_e = (pos_e != pos_e.shift()).sum()
strat_ret_e = apply_slippage(strat_ret_e, trade_days_e / len(strat_ret_e))
equity_e = ACCOUNT * (1 + strat_ret_e).cumprod()

res_e = compute_metrics(equity_e, qqq_oot_returns, regime.reindex(oot_idx),
                        "E) Vol of Vol", trades_e, bear_2022)
if res_e:
    results.append(res_e)
    print(f"  Sharpe: {res_e['sharpe']}, Corr QQQ: {res_e['corr_with_qqq']}, "
          f"Total: {res_e['total_return_pct']}%, 2022: {res_e['bear_2022_return_pct']}%")

# ═══════════════════════════════════════════════════════
# VARIANT F: Dynamic Hedge Ratio (QQQ + TLT overlay)
# ═══════════════════════════════════════════════════════
print("\n--- Variant F: Dynamic Hedge Ratio ---")
strat_ret_f = pd.Series(0.0, index=oot_idx)
trades_f = 0
prev_hedge = 0.0

for i, dt in enumerate(oot_idx):
    v = vix_close.get(dt, np.nan)
    if pd.isna(v):
        strat_ret_f.iloc[i] = qqq_ret.iloc[i]
        continue

    if v > 30:
        hedge_pct = 1.0  # 100% TLT
    elif v > 25:
        hedge_pct = 0.50
    elif v > 20:
        hedge_pct = 0.30
    else:
        hedge_pct = 0.0

    if hedge_pct != prev_hedge:
        trades_f += 1
        prev_hedge = hedge_pct

    qqq_w = 1 - hedge_pct
    tlt_w = hedge_pct
    strat_ret_f.iloc[i] = qqq_w * qqq_ret.iloc[i] + tlt_w * tlt_ret.iloc[i]

trade_days_f = trades_f
strat_ret_f_adj = apply_slippage(strat_ret_f, trade_days_f / max(len(strat_ret_f), 1))
equity_f = ACCOUNT * (1 + strat_ret_f_adj).cumprod()

res_f = compute_metrics(equity_f, qqq_oot_returns, regime.reindex(oot_idx),
                        "F) Dynamic Hedge Ratio", trades_f, bear_2022)
if res_f:
    results.append(res_f)
    print(f"  Sharpe: {res_f['sharpe']}, Corr QQQ: {res_f['corr_with_qqq']}, "
          f"Total: {res_f['total_return_pct']}%, 2022: {res_f['bear_2022_return_pct']}%")

# ═══════════════════════════════════════════════════════
# QQQ Buy-and-Hold benchmark
# ═══════════════════════════════════════════════════════
print("\n--- Benchmark: QQQ Buy-and-Hold ---")
equity_qqq = ACCOUNT * (1 + qqq_ret).cumprod()
res_qqq = compute_metrics(equity_qqq, qqq_oot_returns, regime.reindex(oot_idx),
                          "Benchmark: QQQ Buy-Hold", 1, bear_2022)
if res_qqq:
    results.append(res_qqq)
    print(f"  Sharpe: {res_qqq['sharpe']}, Total: {res_qqq['total_return_pct']}%, "
          f"2022: {res_qqq['bear_2022_return_pct']}%")

# ═══════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("TAIL RISK HEDGING BACKTEST — RESULTS SUMMARY")
print("=" * 90)
print(f"{'Variant':<30} {'Sharpe':>7} {'Sortino':>8} {'Return%':>8} {'MaxDD%':>7} "
      f"{'CorrQQQ':>8} {'2022%':>7} {'Calmar':>7} {'Pass':>5}")
print("-" * 90)
for r in results:
    tag = "YES" if r.get("passed_all_gates") else "NO"
    print(f"{r['variant']:<30} {r['sharpe']:>7.3f} {r['sortino']:>8.3f} "
          f"{r['total_return_pct']:>7.1f}% {r['max_drawdown_pct']:>6.1f}% "
          f"{r['corr_with_qqq']:>8.3f} {r['bear_2022_return_pct']:>6.1f}% "
          f"{r['calmar']:>7.3f} {tag:>5}")

print("\n5-GATE DETAIL:")
for r in results:
    if "Benchmark" in r["variant"]:
        continue
    g = r["gates"]
    flags = []
    for k, v in g.items():
        flags.append(f"{'PASS' if v else 'FAIL'} {k}")
    print(f"  {r['variant']}: {' | '.join(flags)}")

# ─── Identify best crash hedges ───
print("\n--- CRASH HEDGE RANKING (by negative QQQ correlation + 2022 performance) ---")
strats_only = [r for r in results if "Benchmark" not in r["variant"]]
# Rank by: most negative correlation with QQQ (primary), best 2022 return (secondary)
strats_only.sort(key=lambda x: (x["corr_with_qqq"], -x["bear_2022_return_pct"]))
for i, r in enumerate(strats_only):
    print(f"  #{i+1}: {r['variant']} — Corr: {r['corr_with_qqq']:.3f}, "
          f"2022: {r['bear_2022_return_pct']:.1f}%, Sharpe: {r['sharpe']:.3f}")

# ─── Save results ───
output = {
    "metadata": {
        "run_date": datetime.now().isoformat(),
        "oot_period": f"{OOT_START} to {OOT_END}",
        "account_size": ACCOUNT,
        "slippage_pct": SLIPPAGE_PCT,
        "permutation_iters": PERM_ITERS,
        "validation_gates": {
            "sharpe": ">0.5",
            "perm_p": "<0.05",
            "regime_gap": "<0.5",
            "max_dd": ">-50%",
            "min_trades": ">=20",
        },
    },
    "results": results,
}

out_path = Path("/home/jupiter/Lvl3Quant/data/tail_risk_hedging_results.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(output, f, indent=2, default=str)

print(f"\nResults saved to {out_path}")
print("Done.")

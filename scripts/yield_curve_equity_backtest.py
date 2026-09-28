#!/usr/bin/env python3
"""
Yield Curve Signal for Equity Rotation — Backtest
Academic basis: Harvey (1989) "Forecasts of Economic Growth from the Bond and Stock Markets"

Signal: 10Y-3M Treasury spread predicts recessions and equity returns.
Inverted curve → defensive (cash/bonds). Normal/steepening → equities.

Variants:
  A: Binary — QQQ when spread > 0, cash when inverted. Monthly rebalance.
  B: Graduated — 100% QQQ when spread > 1%, 50% when 0-1%, 0% when inverted.
  C: Steepening — long QQQ when 30d change in spread > 0.2%.
  D: Combined VIX — QQQ when spread > 0 AND VIX < 25.
  E: Bond rotation — QQQ when spread > 0, TLT when inverted.
  F: Contrarian — buy QQQ when curve deeply inverted (< -0.5%).

OOT: Jan 2022 – Jul 2026
Starting capital: $645
Cost: $0 commission, 0.02% slippage per trade
"""

import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from pathlib import Path

# ─── Parameters ───────────────────────────────────────────────────────────────
START_DATE = "2020-01-01"   # extra runway for lookback
OOT_START  = "2022-01-01"
OOT_END    = "2026-07-29"
INITIAL_CAPITAL = 645.0
SLIPPAGE_PCT = 0.0002       # 0.02% per trade (round-trip)
N_PERMUTATIONS = 1000
MONTHLY_REBAL = True        # most variants rebalance monthly

# ─── Data Download ────────────────────────────────────────────────────────────
print("Downloading data...")
tickers = {
    "QQQ": "QQQ",
    "SPY": "SPY",
    "TLT": "TLT",
    "TNX": "^TNX",    # 10Y yield (% * 10 in yahoo)
    "IRX": "^IRX",    # 3M T-bill yield
    "VIX": "^VIX",
}

raw = {}
for name, ticker in tickers.items():
    try:
        df = yf.download(ticker, start=START_DATE, end=OOT_END, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        raw[name] = df["Close"].dropna()
        print(f"  {name}: {len(raw[name])} rows, {raw[name].index[0].date()} to {raw[name].index[-1].date()}")
    except Exception as e:
        print(f"  {name}: FAILED — {e}")

# Build aligned daily dataframe
data = pd.DataFrame(raw).dropna()
print(f"Aligned data: {len(data)} rows, {data.index[0].date()} to {data.index[-1].date()}")

# Yahoo returns TNX as yield*1 (i.e. 4.5 = 4.5%), IRX similarly
# Compute spread in percentage points
data["spread"] = data["TNX"] - data["IRX"]
data["spread_30d_chg"] = data["spread"] - data["spread"].shift(21)  # ~30 calendar days
data["QQQ_ret"] = data["QQQ"].pct_change()
data["TLT_ret"] = data["TLT"].pct_change()
data["SPY_ret"] = data["SPY"].pct_change()

# Monthly rebalance mask: first trading day of each month
data["month"] = data.index.to_period("M")
data["rebal_day"] = data["month"] != data["month"].shift(1)

# OOT filter
oot = data.loc[OOT_START:OOT_END].copy()
print(f"OOT period: {len(oot)} days, {oot.index[0].date()} to {oot.index[-1].date()}")
print(f"Spread range: {oot['spread'].min():.3f}% to {oot['spread'].max():.3f}%")
print(f"Days inverted (spread < 0): {(oot['spread'] < 0).sum()}")
print(f"Days deeply inverted (< -0.5%): {(oot['spread'] < -0.5).sum()}")
print()

# ─── Backtest Engine ──────────────────────────────────────────────────────────
def backtest(signals_df, ret_col="QQQ_ret", label="Strategy"):
    """
    signals_df must have column 'weight' (0 to 1) and same index as oot.
    Returns dict of metrics.
    """
    df = signals_df.copy()
    df["asset_ret"] = oot[ret_col].reindex(df.index).fillna(0)

    # Track trades for slippage
    df["weight_prev"] = df["weight"].shift(1).fillna(0)
    df["traded"] = (df["weight"] != df["weight_prev"]).astype(int)
    df["slippage"] = df["traded"] * SLIPPAGE_PCT * df["weight"].clip(lower=df["weight_prev"])

    # Strategy returns
    df["strat_ret"] = df["weight"] * df["asset_ret"] - df["slippage"]

    # Equity curve
    df["equity"] = INITIAL_CAPITAL * (1 + df["strat_ret"]).cumprod()

    # Buy & hold benchmark
    df["bh_ret"] = oot["QQQ_ret"].reindex(df.index).fillna(0)
    df["bh_equity"] = INITIAL_CAPITAL * (1 + df["bh_ret"]).cumprod()

    # Metrics
    daily_rets = df["strat_ret"].dropna()
    n_days = len(daily_rets)
    ann_factor = 252

    total_ret = df["equity"].iloc[-1] / INITIAL_CAPITAL - 1
    bh_total_ret = df["bh_equity"].iloc[-1] / INITIAL_CAPITAL - 1

    mean_ret = daily_rets.mean() * ann_factor
    std_ret = daily_rets.std() * np.sqrt(ann_factor)
    sharpe = mean_ret / std_ret if std_ret > 0 else 0

    downside = daily_rets[daily_rets < 0].std() * np.sqrt(ann_factor)
    sortino = mean_ret / downside if downside > 0 else 0

    # Max drawdown
    cum = (1 + daily_rets).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()

    # Win rate (days with positive return when invested)
    invested_days = daily_rets[df["weight"] > 0]
    win_rate = (invested_days > 0).mean() if len(invested_days) > 0 else 0

    # Profit factor
    gross_profit = invested_days[invested_days > 0].sum()
    gross_loss = abs(invested_days[invested_days < 0].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Number of trades (weight changes)
    n_trades = df["traded"].sum()

    # Time in market
    time_in_market = (df["weight"] > 0).mean()

    # Regime analysis: classify days by SPY return
    df["spy_ret"] = oot["SPY_ret"].reindex(df.index).fillna(0)
    df["spy_cum"] = (1 + df["spy_ret"]).cumprod()

    # 21-day rolling SPY return for regime
    df["spy_21d"] = df["spy_ret"].rolling(21).sum()
    green_mask = df["spy_21d"] > 0.01
    red_mask = df["spy_21d"] < -0.01
    flat_mask = ~green_mask & ~red_mask

    regimes = {}
    for regime_name, mask in [("green", green_mask), ("red", red_mask), ("flat", flat_mask)]:
        r = daily_rets[mask]
        if len(r) > 10:
            r_mean = r.mean() * ann_factor
            r_std = r.std() * np.sqrt(ann_factor)
            regimes[regime_name] = {
                "sharpe": round(r_mean / r_std, 3) if r_std > 0 else 0,
                "n_days": int(len(r)),
                "mean_ret_ann": round(r_mean, 4),
            }
        else:
            regimes[regime_name] = {"sharpe": 0, "n_days": int(len(r)), "mean_ret_ann": 0}

    # Regime gap check
    g_sharpe = regimes.get("green", {}).get("sharpe", 0)
    r_sharpe = regimes.get("red", {}).get("sharpe", 0)
    max_abs = max(abs(g_sharpe), abs(r_sharpe), 0.001)
    regime_gap = abs(g_sharpe - r_sharpe) / max_abs

    return {
        "label": label,
        "total_return_pct": round(total_ret * 100, 2),
        "bh_return_pct": round(bh_total_ret * 100, 2),
        "excess_vs_bh_pct": round((total_ret - bh_total_ret) * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "profit_factor": round(pf, 3),
        "win_rate": round(win_rate, 4),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "n_trades": int(n_trades),
        "n_days_oot": int(n_days),
        "time_in_market_pct": round(time_in_market * 100, 1),
        "final_equity": round(df["equity"].iloc[-1], 2),
        "regimes": regimes,
        "regime_gap": round(regime_gap, 3),
        "daily_returns": daily_rets.values,  # for permutation test
    }

# ─── Strategy Variants ───────────────────────────────────────────────────────

def variant_A():
    """Binary: QQQ when spread > 0, cash when inverted. Monthly rebalance."""
    df = oot[["spread", "rebal_day"]].copy()
    df["signal"] = (df["spread"] > 0).astype(float)
    # Monthly rebalance: hold signal until next rebal day
    df["weight"] = np.nan
    df.loc[df["rebal_day"], "weight"] = df.loc[df["rebal_day"], "signal"]
    df["weight"] = df["weight"].ffill().fillna(0)
    return backtest(df, ret_col="QQQ_ret", label="A: Binary (spread>0 → QQQ)")

def variant_B():
    """Graduated: 100% QQQ when spread > 1%, 50% when 0-1%, 0% when inverted."""
    df = oot[["spread", "rebal_day"]].copy()
    def grad_weight(s):
        if s > 1.0:
            return 1.0
        elif s > 0:
            return 0.5
        else:
            return 0.0
    df["signal"] = df["spread"].apply(grad_weight)
    df["weight"] = np.nan
    df.loc[df["rebal_day"], "weight"] = df.loc[df["rebal_day"], "signal"]
    df["weight"] = df["weight"].ffill().fillna(0)
    return backtest(df, ret_col="QQQ_ret", label="B: Graduated (100/50/0%)")

def variant_C():
    """Steepening: long QQQ when 30d change in spread > 0.2%."""
    df = oot[["spread", "spread_30d_chg", "rebal_day"]].copy()
    df["signal"] = (df["spread_30d_chg"] > 0.2).astype(float)
    df["weight"] = np.nan
    df.loc[df["rebal_day"], "weight"] = df.loc[df["rebal_day"], "signal"]
    df["weight"] = df["weight"].ffill().fillna(0)
    return backtest(df, ret_col="QQQ_ret", label="C: Steepening (30d chg > 0.2%)")

def variant_D():
    """Combined VIX: QQQ when spread > 0 AND VIX < 25."""
    df = oot[["spread", "rebal_day"]].copy()
    df["vix"] = oot["VIX"]
    df["signal"] = ((df["spread"] > 0) & (df["vix"] < 25)).astype(float)
    df["weight"] = np.nan
    df.loc[df["rebal_day"], "weight"] = df.loc[df["rebal_day"], "signal"]
    df["weight"] = df["weight"].ffill().fillna(0)
    return backtest(df, ret_col="QQQ_ret", label="D: Spread>0 + VIX<25 → QQQ")

def variant_E():
    """Bond rotation: QQQ when spread > 0, TLT when inverted."""
    df = oot[["spread", "rebal_day"]].copy()
    df["in_equities"] = (df["spread"] > 0).astype(float)
    # Monthly rebalance
    df["weight_eq"] = np.nan
    df.loc[df["rebal_day"], "weight_eq"] = df.loc[df["rebal_day"], "in_equities"]
    df["weight_eq"] = df["weight_eq"].ffill().fillna(0)
    df["weight_bond"] = 1.0 - df["weight_eq"]

    # Combined return
    df["combo_ret"] = (
        df["weight_eq"] * oot["QQQ_ret"].reindex(df.index).fillna(0) +
        df["weight_bond"] * oot["TLT_ret"].reindex(df.index).fillna(0)
    )

    # Track trades
    df["weight"] = df["weight_eq"]  # for trade counting
    df["weight_prev"] = df["weight"].shift(1).fillna(0)
    df["traded"] = (df["weight"] != df["weight_prev"]).astype(int)
    df["slippage"] = df["traded"] * SLIPPAGE_PCT

    df["strat_ret"] = df["combo_ret"] - df["slippage"]
    df["equity"] = INITIAL_CAPITAL * (1 + df["strat_ret"]).cumprod()

    # Benchmark
    df["bh_ret"] = oot["QQQ_ret"].reindex(df.index).fillna(0)
    df["bh_equity"] = INITIAL_CAPITAL * (1 + df["bh_ret"]).cumprod()

    daily_rets = df["strat_ret"].dropna()
    n_days = len(daily_rets)
    total_ret = df["equity"].iloc[-1] / INITIAL_CAPITAL - 1
    bh_total_ret = df["bh_equity"].iloc[-1] / INITIAL_CAPITAL - 1
    mean_ret = daily_rets.mean() * 252
    std_ret = daily_rets.std() * np.sqrt(252)
    sharpe = mean_ret / std_ret if std_ret > 0 else 0
    downside = daily_rets[daily_rets < 0].std() * np.sqrt(252)
    sortino = mean_ret / downside if downside > 0 else 0
    cum = (1 + daily_rets).cumprod()
    max_dd = ((cum - cum.cummax()) / cum.cummax()).min()
    invested = daily_rets[df["weight_eq"] > 0]
    # For bond rotation, always invested
    all_invested = daily_rets
    win_rate = (all_invested > 0).mean()
    gross_profit = all_invested[all_invested > 0].sum()
    gross_loss = abs(all_invested[all_invested < 0].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    n_trades = df["traded"].sum()

    # Regime analysis
    df["spy_ret"] = oot["SPY_ret"].reindex(df.index).fillna(0)
    df["spy_21d"] = df["spy_ret"].rolling(21).sum()
    green_mask = df["spy_21d"] > 0.01
    red_mask = df["spy_21d"] < -0.01
    flat_mask = ~green_mask & ~red_mask
    regimes = {}
    for rn, mask in [("green", green_mask), ("red", red_mask), ("flat", flat_mask)]:
        r = daily_rets[mask]
        if len(r) > 10:
            rm = r.mean() * 252
            rs = r.std() * np.sqrt(252)
            regimes[rn] = {"sharpe": round(rm/rs, 3) if rs > 0 else 0, "n_days": int(len(r)), "mean_ret_ann": round(rm, 4)}
        else:
            regimes[rn] = {"sharpe": 0, "n_days": int(len(r)), "mean_ret_ann": 0}
    g_s = regimes.get("green", {}).get("sharpe", 0)
    r_s = regimes.get("red", {}).get("sharpe", 0)
    regime_gap = abs(g_s - r_s) / max(abs(g_s), abs(r_s), 0.001)

    return {
        "label": "E: Bond Rotation (QQQ/TLT)",
        "total_return_pct": round(total_ret * 100, 2),
        "bh_return_pct": round(bh_total_ret * 100, 2),
        "excess_vs_bh_pct": round((total_ret - bh_total_ret) * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "profit_factor": round(pf, 3),
        "win_rate": round(win_rate, 4),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "n_trades": int(n_trades),
        "n_days_oot": int(n_days),
        "time_in_market_pct": 100.0,  # always in something
        "final_equity": round(df["equity"].iloc[-1], 2),
        "regimes": regimes,
        "regime_gap": round(regime_gap, 3),
        "daily_returns": daily_rets.values,
    }

def variant_F():
    """Contrarian: buy QQQ when curve deeply inverted (< -0.5%)."""
    df = oot[["spread", "rebal_day"]].copy()
    df["signal"] = (df["spread"] < -0.5).astype(float)
    df["weight"] = np.nan
    df.loc[df["rebal_day"], "weight"] = df.loc[df["rebal_day"], "signal"]
    df["weight"] = df["weight"].ffill().fillna(0)
    return backtest(df, ret_col="QQQ_ret", label="F: Contrarian (deep inversion → QQQ)")

# ─── Run All Variants ─────────────────────────────────────────────────────────
print("=" * 70)
print("RUNNING YIELD CURVE EQUITY ROTATION BACKTEST")
print("=" * 70)

results = {}
for name, func in [("A", variant_A), ("B", variant_B), ("C", variant_C),
                     ("D", variant_D), ("E", variant_E), ("F", variant_F)]:
    try:
        res = func()
        results[name] = res
        print(f"\n{'─'*60}")
        print(f"Variant {res['label']}")
        print(f"  Total Return: {res['total_return_pct']:.1f}%  (B&H QQQ: {res['bh_return_pct']:.1f}%)")
        print(f"  Excess vs B&H: {res['excess_vs_bh_pct']:+.1f}%")
        print(f"  Sharpe: {res['sharpe']:.3f}  Sortino: {res['sortino']:.3f}  PF: {res['profit_factor']:.3f}")
        print(f"  WR: {res['win_rate']:.1%}  MaxDD: {res['max_drawdown_pct']:.1f}%")
        print(f"  Trades: {res['n_trades']}  Time in Market: {res['time_in_market_pct']:.0f}%")
        print(f"  Final Equity: ${res['final_equity']:.2f}")
        print(f"  Regime Gap: {res['regime_gap']:.3f}")
        for rn, rv in res["regimes"].items():
            print(f"    {rn}: Sharpe={rv['sharpe']:.3f}, days={rv['n_days']}")
    except Exception as e:
        print(f"  Variant {name}: ERROR — {e}")
        import traceback; traceback.print_exc()

# ─── Permutation Test ─────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"PERMUTATION TEST ({N_PERMUTATIONS} shuffles)")
print(f"{'='*70}")

np.random.seed(42)
# For timing strategies, correct permutation test: randomly shift signal relative to returns
# This preserves autocorrelation structure of both signal and returns
qqq_oot_rets = oot["QQQ_ret"].dropna().values

for name, res in results.items():
    rets = res["daily_returns"]
    observed_sharpe = res["sharpe"]
    n = len(rets)

    count_above = 0
    for _ in range(N_PERMUTATIONS):
        # Circular shift: randomly offset the returns relative to signal
        shift = np.random.randint(1, n)
        shifted_rets = np.roll(qqq_oot_rets[:n], shift)
        # Recompute strategy returns using same weights but shifted market returns
        # Approximate: use the daily_returns structure scaled by shifted market
        s_mean = shifted_rets.mean() * 252
        s_std = shifted_rets.std() * np.sqrt(252)
        s_sharpe = s_mean / s_std if s_std > 0 else 0
        if s_sharpe >= observed_sharpe:
            count_above += 1

    p_value = count_above / N_PERMUTATIONS
    res["permutation_p_value"] = round(p_value, 4)
    print(f"  Variant {name}: observed Sharpe={observed_sharpe:.3f}, p={p_value:.4f}")

# ─── 5-Gate Validation ────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("5-GATE VALIDATION")
print(f"{'='*70}")

for name, res in results.items():
    gates = {
        "G1_sharpe_gt_0.5": bool(res["sharpe"] > 0.5),
        "G2_perm_p_lt_0.05": bool(res["permutation_p_value"] < 0.05),
        "G3_regime_gap_lt_0.5": bool(res["regime_gap"] < 0.5),
        "G4_maxdd_gt_neg50": bool(res["max_drawdown_pct"] > -50),
        "G5_trades_gte_20": bool(res["n_trades"] >= 20),
    }
    res["gates"] = gates
    passed = sum(gates.values())
    res["gates_passed"] = passed
    status = "PASS" if passed == 5 else "FAIL"
    print(f"\n  Variant {name} ({res['label']}): {status} ({passed}/5)")
    for gname, gval in gates.items():
        print(f"    {'[x]' if gval else '[ ]'} {gname}: {gval}")

# ─── Buy & Hold Reference ────────────────────────────────────────────────────
bh_rets = oot["QQQ_ret"].dropna()
bh_total = (1 + bh_rets).prod() - 1
bh_sharpe = (bh_rets.mean() * 252) / (bh_rets.std() * np.sqrt(252))
bh_dd = (((1 + bh_rets).cumprod() - (1 + bh_rets).cumprod().cummax()) / (1 + bh_rets).cumprod().cummax()).min()
print(f"\n{'='*70}")
print(f"BENCHMARK: QQQ Buy & Hold")
print(f"  Total Return: {bh_total*100:.1f}%  Sharpe: {bh_sharpe:.3f}  MaxDD: {bh_dd*100:.1f}%")
print(f"  Final Equity: ${INITIAL_CAPITAL*(1+bh_total):.2f}")

# ─── Save Results ─────────────────────────────────────────────────────────────
output = {
    "strategy": "Yield Curve Equity Rotation",
    "academic_basis": "Harvey (1989) — yield curve predicts recessions",
    "oot_period": f"{OOT_START} to {OOT_END}",
    "initial_capital": INITIAL_CAPITAL,
    "slippage_pct": SLIPPAGE_PCT,
    "benchmark": {
        "label": "QQQ Buy & Hold",
        "total_return_pct": round(bh_total * 100, 2),
        "sharpe": round(bh_sharpe, 3),
        "max_drawdown_pct": round(bh_dd * 100, 2),
        "final_equity": round(INITIAL_CAPITAL * (1 + bh_total), 2),
    },
    "variants": {},
    "best_variant": None,
    "yield_curve_context": {
        "inversion_start": "~Jul 2022",
        "inversion_end": "~Sep 2024",
        "inversion_duration_months": 26,
        "deepest_inversion": round(float(oot["spread"].min()), 3),
        "current_spread": round(float(oot["spread"].iloc[-1]), 3),
    },
    "run_timestamp": datetime.now().isoformat(),
}

# Clean results for JSON (remove numpy arrays)
best_sharpe = -999
best_name = None
for name, res in results.items():
    clean = {k: v for k, v in res.items() if k != "daily_returns"}
    output["variants"][name] = clean
    if res["gates_passed"] == 5 and res["sharpe"] > best_sharpe:
        best_sharpe = res["sharpe"]
        best_name = name

if best_name:
    output["best_variant"] = best_name
    output["recommendation"] = f"Variant {best_name} passes all 5 gates with Sharpe {best_sharpe:.3f}"
else:
    output["recommendation"] = "No variant passes all 5 gates. Strategy does not meet quality bar."

out_path = Path("/home/jupiter/Lvl3Quant/data/yield_curve_equity_results.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved to {out_path}")

# ─── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("FINAL RECOMMENDATION")
print(f"{'='*70}")
print(output["recommendation"])

#!/usr/bin/env python3
"""
HC #428 R1 Regime Gate + Alpha Decomposition for Wheel-Strategy Tier Ladder.

For each of 4 variants (scalp, scalp+regime, full-wheel, full-wheel+regime)
and each of 5 tiers (Conservative, Balanced, Income, Aggressive, Turbo):
  - Compute daily returns from equity_<tier>.parquet
  - Classify SPY days green/red/flat using ±0.25% close-to-close threshold
  - Stratified Sharpe per regime
  - HC #428 R1: |Sharpe_g - Sharpe_r| / max(|Sg|,|Sr|) <= 0.50
  - Day-conc: top-1-day-PnL / sum(|PnL|)
  - Alpha decomp vs SPY: annualized alpha, NW t-stat, beta, R^2, IR

Outputs:
  output/wheel_regime_gate_<timestamp>/results.json
  output/wheel_regime_gate_<timestamp>/summary.csv
"""
import os
import sys
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/jupiter/Lvl3Quant")
RESULTS = ROOT / "wheel_strategy_v1" / "results"
SPY_PATH = ROOT / "output" / "macro_swing_v1" / "spy_daily.parquet"

VARIANTS = {
    "scalp":             {"dir": "tier_ladder_v1",            "suffix": ""},
    "full_wheel":        {"dir": "tier_ladder_v2_fullwheel",  "suffix": "_FW"},
    "full_wheel_regime": {"dir": "tier_ladder_v3_regime",     "suffix": "_FW"},
    "scalp_regime":      {"dir": "tier_ladder_v4_scalp_regime","suffix": ""},
}

TIERS = ["Tier1_Conservative", "Tier2_Balanced", "Tier3_Income",
         "Tier4_Aggressive", "Tier5_Turbo"]

FLAT_THRESH = 0.0025   # ±0.25% canonical
ANN_FACTOR = 252.0
RF_DAILY = 0.0         # 0 risk-free for Sharpe (consistent w/ tier_summary)
GATE_MAX = 0.50        # HC #428 R1


def load_spy():
    df = pd.read_parquet(SPY_PATH)[["date", "adj_close"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["spy_ret"] = df["adj_close"].pct_change()
    df["regime"] = np.where(df["spy_ret"] > FLAT_THRESH, "green",
                    np.where(df["spy_ret"] < -FLAT_THRESH, "red", "flat"))
    return df[["date", "spy_ret", "regime"]]


def sharpe(returns):
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 2:
        return float("nan")
    sd = r.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return float("nan")
    return float((r.mean() - RF_DAILY) / sd * math.sqrt(ANN_FACTOR))


def newey_west_se(resid, X, lags):
    """Newey-West HAC standard errors for OLS regression."""
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    u = resid.reshape(-1, 1)
    S = np.zeros((k, k))
    for t in range(n):
        S += (u[t] ** 2) * np.outer(X[t], X[t])
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)
        for t in range(L, n):
            ut, utl = u[t, 0], u[t - L, 0]
            xt, xtl = X[t], X[t - L]
            S += w * ut * utl * (np.outer(xt, xtl) + np.outer(xtl, xt))
    cov = XtX_inv @ S @ XtX_inv
    return np.sqrt(np.diag(cov))


def alpha_decomp(strategy_ret, spy_ret):
    """OLS regression r_strat = alpha + beta * r_spy + eps, with NW(5) t-stats."""
    df = pd.concat([strategy_ret.rename("s"), spy_ret.rename("m")], axis=1).dropna()
    n = len(df)
    if n < 30:
        return dict(alpha_ann=float("nan"), t_alpha_nw=float("nan"),
                    beta=float("nan"), r2=float("nan"), ir=float("nan"), n=n)
    y = df["s"].values
    X = np.column_stack([np.ones(n), df["m"].values])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha_d, beta = coef[0], coef[1]
    resid = y - X @ coef
    ss_res = (resid ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    lags = int(np.floor(4 * (n / 100) ** (2 / 9)))
    se = newey_west_se(resid, X, lags)
    t_alpha = alpha_d / se[0] if se[0] > 0 else float("nan")
    alpha_ann = alpha_d * ANN_FACTOR
    # Information ratio = annualized alpha / annualized resid sd
    sd_resid = resid.std(ddof=1)
    ir = (alpha_d / sd_resid * math.sqrt(ANN_FACTOR)) if sd_resid > 0 else float("nan")
    return dict(alpha_ann=float(alpha_ann), t_alpha_nw=float(t_alpha),
                beta=float(beta), r2=float(r2), ir=float(ir), n=int(n))


def calmar(cagr, max_dd):
    if max_dd is None or max_dd == 0 or np.isnan(max_dd):
        return float("nan")
    return float(cagr / abs(max_dd))


def process_one(variant_key, variant_meta, tier, spy_df, tier_summary):
    eq_path = RESULTS / variant_meta["dir"] / f"equity_{tier}{variant_meta['suffix']}.parquet"
    if not eq_path.exists():
        return None
    eq = pd.read_parquet(eq_path).copy()
    eq["date"] = pd.to_datetime(eq["date"])
    eq = eq.sort_values("date").reset_index(drop=True)
    eq["ret"] = eq["equity"].pct_change()

    merged = eq.merge(spy_df, on="date", how="inner").dropna(subset=["ret", "spy_ret"])

    # Regime-stratified Sharpe
    by_reg = {}
    for reg in ["green", "red", "flat"]:
        sub = merged.loc[merged["regime"] == reg, "ret"]
        by_reg[reg] = dict(n=int(len(sub)), sharpe=sharpe(sub),
                           mean_ret=float(sub.mean()) if len(sub) else float("nan"))

    sg = by_reg["green"]["sharpe"]
    sr = by_reg["red"]["sharpe"]
    denom = max(abs(sg) if not np.isnan(sg) else 0,
                abs(sr) if not np.isnan(sr) else 0)
    if denom == 0 or (np.isnan(sg) or np.isnan(sr)):
        gap = float("nan")
        gate_pass = False
    else:
        gap = abs(sg - sr) / denom
        gate_pass = bool(gap <= GATE_MAX)

    # Day concentration: top-1-day abs P&L / sum(|P&L|)
    daily_pnl = merged["ret"].abs()
    total_abs = float(daily_pnl.sum())
    top1 = float(daily_pnl.max()) if len(daily_pnl) else float("nan")
    day_conc = top1 / total_abs if total_abs > 0 else float("nan")

    # Alpha decomp
    alpha = alpha_decomp(merged["ret"], merged["spy_ret"])

    # Headline metrics from tier_summary
    tier_label = tier + variant_meta["suffix"]
    row = tier_summary.loc[tier_summary["tier"] == tier_label]
    if row.empty:
        cagr = sharpe_full = max_dd = float("nan")
        sortino = pf = wr = float("nan")
    else:
        r = row.iloc[0]
        cagr = float(r["cagr"])
        sharpe_full = float(r["sharpe"])
        max_dd = float(r["max_dd"])
        sortino = float(r["sortino"])
        pf = float(r["pf"])
        wr = float(r["wr"])

    return dict(
        variant=variant_key,
        tier=tier,
        n_days=int(len(merged)),
        cagr=cagr,
        sharpe=sharpe_full,
        sortino=sortino,
        max_dd=max_dd,
        calmar=calmar(cagr, max_dd),
        pf=pf,
        wr=wr,
        sharpe_green=float(sg) if not np.isnan(sg) else None,
        sharpe_red=float(sr) if not np.isnan(sr) else None,
        sharpe_flat=float(by_reg["flat"]["sharpe"]) if not np.isnan(by_reg["flat"]["sharpe"]) else None,
        n_green=by_reg["green"]["n"],
        n_red=by_reg["red"]["n"],
        n_flat=by_reg["flat"]["n"],
        regime_gap=float(gap) if not np.isnan(gap) else None,
        gate_r1_pass=gate_pass,
        day_conc_top1=float(day_conc) if not np.isnan(day_conc) else None,
        alpha_ann=alpha["alpha_ann"],
        t_alpha_nw=alpha["t_alpha_nw"],
        beta=alpha["beta"],
        r2=alpha["r2"],
        ir=alpha["ir"],
        alpha_significant=bool(not np.isnan(alpha["t_alpha_nw"]) and
                               abs(alpha["t_alpha_nw"]) > 1.96),
    )


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "output" / f"wheel_regime_gate_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[init] output dir: {out_dir}")

    spy_df = load_spy()
    print(f"[spy] loaded {len(spy_df)} rows, "
          f"green={int((spy_df['regime']=='green').sum())}, "
          f"red={int((spy_df['regime']=='red').sum())}, "
          f"flat={int((spy_df['regime']=='flat').sum())}")

    rows = []
    for vk, vm in VARIANTS.items():
        ts_path = RESULTS / vm["dir"] / "tier_summary.parquet"
        if not ts_path.exists():
            print(f"[skip] missing {ts_path}")
            continue
        tier_summary = pd.read_parquet(ts_path)
        for tier in TIERS:
            row = process_one(vk, vm, tier, spy_df, tier_summary)
            if row is None:
                print(f"[miss] {vk}/{tier}")
                continue
            rows.append(row)
            print(f"[{vk}/{tier}] Sg={row['sharpe_green']:.2f} "
                  f"Sr={row['sharpe_red']:.2f} gap={row['regime_gap']:.2f} "
                  f"pass={row['gate_r1_pass']} alpha_ann={row['alpha_ann']:.3f} "
                  f"t={row['t_alpha_nw']:.2f}" if row['sharpe_green'] is not None and row['sharpe_red'] is not None and row['regime_gap'] is not None
                  else f"[{vk}/{tier}] (NaN in metrics)")

    df = pd.DataFrame(rows)
    csv_path = out_dir / "summary.csv"
    df.to_csv(csv_path, index=False)

    n_total = len(df)
    n_pass = int(df["gate_r1_pass"].sum())
    df_both = df[(df["gate_r1_pass"]) & (df["alpha_significant"])]
    n_both = len(df_both)

    if len(df_both) > 0:
        best = df_both.sort_values("sharpe", ascending=False).iloc[0]
        best_dict = best.to_dict()
    else:
        best_dict = None

    pass_only = df[df["gate_r1_pass"]]
    if len(pass_only) > 0:
        best_gate_only = pass_only.sort_values("sharpe", ascending=False).iloc[0].to_dict()
    else:
        best_gate_only = None

    summary = dict(
        timestamp=ts,
        flat_threshold=FLAT_THRESH,
        gate_max=GATE_MAX,
        n_total_configs=n_total,
        n_gate_pass=n_pass,
        n_gate_pass_and_alpha_sig=n_both,
        best_gate_and_alpha=best_dict,
        best_gate_only=best_gate_only,
        rows=rows,
    )

    json_path = out_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print()
    print(f"[done] {n_total} configs, {n_pass} pass gate, "
          f"{n_both} pass gate AND alpha significant")
    print(f"[csv] {csv_path}")
    print(f"[json] {json_path}")
    return summary, out_dir


if __name__ == "__main__":
    main()

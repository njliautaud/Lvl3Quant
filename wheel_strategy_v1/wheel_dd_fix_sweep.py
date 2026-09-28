"""
wheel_dd_fix_sweep.py — wheel_dd_fix_v6 (2026-06-10).

Context: the v5_REAL FullWheel tier books showed -84%/-98% max drawdowns.
Root cause was an NAV accounting bug in backtest/wheel_engine._equity_mtm
(share cost basis double-counted after assignment; premiums double-counted).
That bug is FIXED in wheel_engine.py. This script sweeps residual-risk fix
variants on the two tiers the user flagged (Balanced, Aggressive):

  (a) assignment exposure caps  — max_assigned_notional_pct in {0.30, 0.50}
  (b) share stop-loss           — share_stop_loss_pct in {0.10, 0.20}
  (c) regime suspension         — t-1 macro (no look-ahead) + VIX<=30 gate
  (d) combo                     — cap 0.30 + stop 0.15 + regime

Evaluation per HC #428 R1: daily Sharpe/Sortino/Calmar/MaxDD/PF/WR,
green/red/flat stratification on SPY close-to-close (t aligned), regime gap
ratio |Sh_g - Sh_r| / max(|Sh_g|,|Sh_r|) <= 0.50, day-concentration <= 0.70.

All runs log to MLflow experiment "wheel_dd_fix_v6" (http://localhost:5000).
Sliding evaluation window 2020-01-01..2025-12-31 (same as v5_REAL_2020_2025).
"""
from __future__ import annotations
import json
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from backtest.wheel_engine import run_wheel  # noqa: E402
from strategy.tiers import all_tiers_full_wheel  # noqa: E402
from strategy.tier_runner import _load_inputs, _filter_universe_subset, _apply_iv_rank_floor  # noqa: E402

START, END = "2020-01-01", "2025-12-31"
CAPITAL = 100_000.0
TIERS = ["Tier2_Balanced_FW", "Tier4_Aggressive_FW"]
OUT = ROOT / "results" / "wheel_dd_fix_v6_sweep"
FLAT_BAND = 0.0025  # SPY close-to-close +/-0.25% = flat day


def spy_regime() -> pd.Series:
    se = pd.read_parquet(ROOT / "data" / "cache" / "sector_etfs.parquet",
                         columns=["date", "ticker", "close"])
    spy = (se[se["ticker"] == "SPY"].sort_values("date")
           .set_index("date")["close"].astype(float))
    ret = spy.pct_change()
    reg = pd.Series("flat", index=ret.index)
    reg[ret > FLAT_BAND] = "green"
    reg[ret < -FLAT_BAND] = "red"
    return reg


def ann_sharpe(r: pd.Series) -> float:
    if len(r) < 2 or r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(252))


def metrics(eq_df: pd.DataFrame, led: pd.DataFrame, regime: pd.Series) -> dict:
    eq = eq_df.sort_values("date").set_index("date")["equity"].astype(float)
    r = eq.pct_change().dropna()
    yrs = len(eq) / 252.0
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1 if yrs > 0 and eq.iloc[0] > 0 else 0.0
    peak = eq.cummax()
    maxdd = float((eq / peak - 1).min())
    dn = r[r < 0]
    sortino = float(r.mean() / dn.std() * np.sqrt(252)) if len(dn) > 1 and dn.std() > 0 else 0.0
    calmar = float(cagr / abs(maxdd)) if maxdd < 0 else 0.0
    pf = wr = 0.0
    if led is not None and not led.empty and "realized_pnl" in led:
        gp = led.loc[led.realized_pnl > 0, "realized_pnl"].sum()
        gl = -led.loc[led.realized_pnl < 0, "realized_pnl"].sum()
        pf = float(gp / gl) if gl > 0 else float("inf")
        wr = float((led.realized_pnl > 0).mean())
    # regime stratification (t aligned: strategy day return vs SPY same-day move)
    reg = regime.reindex(r.index)
    strat = {}
    for g in ("green", "red", "flat"):
        rr = r[reg == g]
        strat[f"sharpe_{g}"] = ann_sharpe(rr)
        strat[f"n_{g}"] = int(len(rr))
    sg, sr = strat["sharpe_green"], strat["sharpe_red"]
    denom = max(abs(sg), abs(sr))
    gap = float(abs(sg - sr) / denom) if denom > 0 else 0.0
    # day concentration: best single day's PnL / total PnL (only if total > 0)
    pnl = eq.diff().dropna()
    tot = pnl.sum()
    dayconc = float(pnl.max() / tot) if tot > 0 else 1.0
    return dict(cagr=float(cagr), sharpe=ann_sharpe(r), sortino=sortino,
                calmar=calmar, max_dd=maxdd, pf=pf, wr=wr,
                n_trades=int(len(led)) if led is not None else 0,
                final_equity=float(eq.iloc[-1]),
                regime_gap=gap, day_concentration=dayconc,
                gate_regime_pass=bool(gap <= 0.50),
                gate_dayconc_pass=bool(dayconc <= 0.70), **strat)


def variants(base_cfg):
    out = [("baseline_fixed", base_cfg)]
    out.append(("cap30", replace(base_cfg, max_assigned_notional_pct=0.30)))
    out.append(("cap50", replace(base_cfg, max_assigned_notional_pct=0.50)))
    out.append(("stop10", replace(base_cfg, share_stop_loss_pct=0.10)))
    out.append(("stop20", replace(base_cfg, share_stop_loss_pct=0.20)))
    out.append(("regime_vix30_t1", replace(
        base_cfg, macro_lag_days=1,
        vix_max_gate=min(base_cfg.vix_max_gate, 30.0))))
    out.append(("combo_cap30_stop15_regime", replace(
        base_cfg, max_assigned_notional_pct=0.30, share_stop_loss_pct=0.15,
        macro_lag_days=1, vix_max_gate=min(base_cfg.vix_max_gate, 30.0))))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = _load_inputs(modeled=True, smoke=False)
    regime = spy_regime()
    tiers = {t.name: t for t in all_tiers_full_wheel()}

    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("wheel_dd_fix_v6")
        HAVE_MLFLOW = True
    except Exception as e:
        print(f"[sweep] mlflow unavailable: {e}", file=sys.stderr)
        HAVE_MLFLOW = False

    rows = []
    for tname in TIERS:
        tier = tiers[tname]
        tickers = _filter_universe_subset(tier, data)
        px = data["prices"][data["prices"]["ticker"].isin(tickers)]
        iv = _apply_iv_rank_floor(
            data["iv"][data["iv"]["ticker"].isin(tickers)].copy(),
            tier.iv_rank_floor)
        for vname, cfg in variants(tier.wheel_cfg):
            tag = f"{tname}.{vname}"
            print(f"\n=== {tag} ===", flush=True)
            res = run_wheel(cfg=cfg, prices=px.copy(), iv=iv, macro=data["macro"],
                            fundamentals=data["fundamentals"],
                            universe=data["universe"],
                            starting_cash=CAPITAL, start=START, end=END)
            m = metrics(res["equity_curve"], res["ledger"], regime)
            m.update(tier=tname, variant=vname,
                     assignments=res["assignment_count"])
            rows.append(m)
            res["equity_curve"].to_parquet(OUT / f"equity_{tag}.parquet", index=False)
            res["ledger"].to_parquet(OUT / f"ledger_{tag}.parquet", index=False)
            print({k: (round(v, 4) if isinstance(v, float) else v)
                   for k, v in m.items()})
            if HAVE_MLFLOW:
                import mlflow
                with mlflow.start_run(run_name=tag):
                    mlflow.log_params(dict(
                        tier=tname, variant=vname, start=START, end=END,
                        capital=CAPITAL, pricing="modeled_bs_calibrated",
                        engine_fix="mtm_basis_double_count_fixed_20260610",
                        max_assigned_notional_pct=cfg.max_assigned_notional_pct,
                        share_stop_loss_pct=cfg.share_stop_loss_pct,
                        macro_lag_days=cfg.macro_lag_days,
                        vix_max_gate=cfg.vix_max_gate))
                    for k, v in m.items():
                        if isinstance(v, (int, float)) and np.isfinite(v):
                            mlflow.log_metric(k, float(v))

    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "sweep_summary.parquet", index=False)
    df.to_csv(OUT / "sweep_summary.csv", index=False)
    print("\n[sweep] DONE ->", OUT / "sweep_summary.csv")


if __name__ == "__main__":
    main()

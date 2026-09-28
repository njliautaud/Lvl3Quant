"""
HC #428 R1 Regime-Agnostic OOT Gate for the 60/40 tech/leader blend.

Reads the underlying tech and leader daily-return books from
leader_plus_tech_combo's report.json, reconstructs the 60/40 blend daily PnL,
classifies each day green/red/flat by SPY close-to-close, and tests:

  PASS conditions (HC #428 R1 + day-conc):
    |Sharpe_green - Sharpe_red| / max(|Sharpe_green|,|Sharpe_red|) <= 0.50
    top-1-day |PnL| / sum(|PnL|) <= 0.70  (HC #344 day-conc cap)

Outputs:
  output/macro_picker/blend_regime_gate_<TS>/regime_gate.json
"""
from __future__ import annotations
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/jupiter/Lvl3Quant")
COMBO_DIR = ROOT / "output/macro_picker/leader_plus_tech_20260609_073155"
SPY_PRICE_PATH = ROOT / "wheel_strategy_v1/data/cache/prices_v2.parquet"
TRADING_DAYS = 252

W_TECH = 0.6
W_LEADER = 0.4

GREEN_THR = 0.0025   # +0.25%
RED_THR = -0.0025    # -0.25%


def sharpe(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / x.std(ddof=1) * np.sqrt(TRADING_DAYS))


def main() -> None:
    # 1) Load combo metadata to find underlying books.
    with open(COMBO_DIR / "report.json") as f:
        report = json.load(f)
    leader_path = Path(report["leader_book"])
    tech_path = Path(report["tech_book"])

    lb = pd.read_parquet(leader_path)[["date", "daily_ret"]].rename(
        columns={"daily_ret": "ret_leader"})
    tb = pd.read_parquet(tech_path)[["date", "daily_ret"]].rename(
        columns={"daily_ret": "ret_tech"})

    blend = pd.merge(lb, tb, on="date", how="inner").sort_values("date")
    blend["daily_ret"] = W_TECH * blend["ret_tech"] + W_LEADER * blend["ret_leader"]

    # 2) Load SPY close, compute close-to-close returns, classify regime.
    prices = pd.read_parquet(SPY_PRICE_PATH)
    spy = (prices[prices["ticker"] == "SPY"][["date", "close"]]
           .sort_values("date").reset_index(drop=True))
    spy["spy_ret"] = spy["close"].pct_change()

    df = pd.merge(blend, spy[["date", "spy_ret"]], on="date", how="left")
    df = df.dropna(subset=["spy_ret"])

    def classify(r: float) -> str:
        if r > GREEN_THR:
            return "green"
        if r < RED_THR:
            return "red"
        return "flat"

    df["regime"] = df["spy_ret"].apply(classify)

    # 3) Per-regime stratified stats.
    regimes = {}
    for label in ["green", "red", "flat"]:
        sub = df[df["regime"] == label]
        rets = sub["daily_ret"].to_numpy()
        if len(rets) == 0:
            regimes[label] = {"n_days": 0}
            continue
        regimes[label] = {
            "n_days": int(len(rets)),
            "mean_daily_ret_pct": float(np.mean(rets) * 100),
            "std_daily_ret_pct": float(np.std(rets, ddof=1) * 100) if len(rets) > 1 else float("nan"),
            "sharpe": sharpe(rets),
            "win_rate": float(np.mean(rets > 0)),
            "sum_pnl_pct": float(np.sum(rets) * 100),
        }

    sg = regimes["green"].get("sharpe", float("nan"))
    sr = regimes["red"].get("sharpe", float("nan"))
    denom = max(abs(sg), abs(sr)) if not (np.isnan(sg) or np.isnan(sr)) else float("nan")
    if denom and not np.isnan(denom) and denom > 0:
        regime_skew = abs(sg - sr) / denom
    else:
        regime_skew = float("nan")
    regime_gate_pass = bool(not np.isnan(regime_skew) and regime_skew <= 0.50)

    # 4) Day-concentration: top-1 absolute daily PnL / sum absolute PnL.
    abs_rets = df["daily_ret"].abs().to_numpy()
    total_abs = abs_rets.sum()
    day_conc = float(abs_rets.max() / total_abs) if total_abs > 0 else float("nan")
    day_conc_gate_pass = bool(not np.isnan(day_conc) and day_conc <= 0.70)

    # 5) Overall stats.
    all_rets = df["daily_ret"].to_numpy()
    overall = {
        "n_days": int(len(all_rets)),
        "date_start": str(df["date"].min().date()),
        "date_end": str(df["date"].max().date()),
        "mean_daily_ret_pct": float(np.mean(all_rets) * 100),
        "sharpe": sharpe(all_rets),
        "win_rate": float(np.mean(all_rets > 0)),
        "cum_ret_pct": float((np.prod(1 + all_rets) - 1) * 100),
    }

    # 6) Verdict.
    verdict = {
        "regime_gate_pass": regime_gate_pass,
        "day_conc_gate_pass": day_conc_gate_pass,
        "overall_pass": bool(regime_gate_pass and day_conc_gate_pass),
        "regime_skew": float(regime_skew) if not np.isnan(regime_skew) else None,
        "regime_skew_threshold": 0.50,
        "day_conc": day_conc,
        "day_conc_threshold": 0.70,
    }

    out = {
        "blend": {"w_tech": W_TECH, "w_leader": W_LEADER},
        "sources": {
            "leader_book": str(leader_path),
            "tech_book": str(tech_path),
            "spy_prices": str(SPY_PRICE_PATH),
        },
        "thresholds": {
            "green_spy_close_to_close_pct": GREEN_THR * 100,
            "red_spy_close_to_close_pct": RED_THR * 100,
        },
        "overall": overall,
        "per_regime": regimes,
        "verdict": verdict,
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / f"output/macro_picker/blend_regime_gate_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "regime_gate.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()

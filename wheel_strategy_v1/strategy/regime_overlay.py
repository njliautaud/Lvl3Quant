"""
regime_overlay.py — HC #555 R1 macro regime gate.

Reads data/cache/macro_extra.parquet (the 28 FRED series) and computes
five per-date binary gates plus an aggregate `risk_off` flag:

    (A) yc_inverted        : T10Y2Y < 0           -> recession-leading inversion
    (B) claims_spike       : ICSA 4w-ROC z>2      -> jobless claims accelerating
    (C) be_inflation_hot   : T10YIE quartile 4    -> inflation expectations in top 25%
    (D) sentiment_shock    : UMCSENT < 6m 20%ile  -> consumer pessimism
    (E) ff_tightening      : FEDFUNDS 3m-delta > 0.25  -> tightening cycle

If >= 2 of {A, B, C, D, E} fire on a day → risk_off = True for that day.

Output: data/cache/regime_overlay.parquet
        columns: date, yc_inverted, claims_spike, be_inflation_hot,
                 sentiment_shock, ff_tightening, gates_on, risk_off

Also provides apply_regime_gate(macro_df, regime_df, vix_force_gate=999.0)
which returns a modified `macro_df` whose `vix` column is forced to
vix_force_gate on risk_off days — this makes the wheel engine skip new
opens via its existing vix_max_gate check, without touching the engine.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"


def _zscore(s: pd.Series, window: int) -> pd.Series:
    m = s.rolling(window, min_periods=max(20, window // 4)).mean()
    sd = s.rolling(window, min_periods=max(20, window // 4)).std()
    return (s - m) / sd


def _rolling_pct_rank(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=max(10, window // 5)).rank(pct=True)


def build_regime() -> pd.DataFrame:
    me_path = CACHE / "macro_extra.parquet"
    if not me_path.exists():
        raise SystemExit(f"[regime] missing {me_path}")
    me = pd.read_parquet(me_path)
    me["date"] = pd.to_datetime(me["date"])
    me = me.sort_values("date").reset_index(drop=True)

    out = me[["date"]].copy()

    # (A) yield curve inversion: T10Y2Y < 0
    yc = me.get("yc_2s10s")
    out["yc_inverted"] = (yc < 0).fillna(False).astype(int) if yc is not None else 0

    # (B) jobless claims spike: 4w pct change z>2 over 1y window
    claims = me.get("jobless_claims")
    if claims is not None:
        roc4w = claims.pct_change(periods=20)  # ~4 weeks bdays
        z = _zscore(roc4w, 252)
        out["claims_spike"] = (z > 2.0).fillna(False).astype(int)
    else:
        out["claims_spike"] = 0

    # (C) breakeven inflation hot: T10YIE in top quartile over 2y
    be = me.get("be_10y")
    if be is not None:
        pr = _rolling_pct_rank(be, 504)
        out["be_inflation_hot"] = (pr > 0.75).fillna(False).astype(int)
    else:
        out["be_inflation_hot"] = 0

    # (D) sentiment shock: UMCSENT in bottom quintile over 6 months
    sent = me.get("umich_sent")
    if sent is not None:
        pr = _rolling_pct_rank(sent, 126)
        out["sentiment_shock"] = (pr < 0.20).fillna(False).astype(int)
    else:
        out["sentiment_shock"] = 0

    # (E) Fed Funds tightening: 3m delta > 0.25 (i.e., +25bp+ in 3 months)
    ff = me.get("fed_funds")
    if ff is not None:
        d3m = ff.diff(periods=63)  # ~3 months bdays
        out["ff_tightening"] = (d3m > 0.25).fillna(False).astype(int)
    else:
        out["ff_tightening"] = 0

    gate_cols = ["yc_inverted", "claims_spike", "be_inflation_hot",
                 "sentiment_shock", "ff_tightening"]
    out["gates_on"] = out[gate_cols].sum(axis=1)
    out["risk_off"] = (out["gates_on"] >= 2).astype(int)

    return out


def apply_regime_gate(macro: pd.DataFrame, regime: pd.DataFrame,
                      vix_force_gate: float = 999.0) -> pd.DataFrame:
    """
    Return a modified copy of `macro` where on risk_off days, vix is forced to
    `vix_force_gate` (defaults to 999). The wheel engine's vix_max_gate check
    then skips new opens. MTM is unaffected — it does not read VIX.
    """
    m = macro.copy()
    m["date"] = pd.to_datetime(m["date"])
    r = regime[["date", "risk_off"]].copy()
    r["date"] = pd.to_datetime(r["date"])
    m = m.merge(r, on="date", how="left")
    m["risk_off"] = m["risk_off"].fillna(0).astype(int)
    if "vix" not in m.columns:
        m["vix"] = 18.0
    m["vix_pre_gate"] = m["vix"]
    m.loc[m["risk_off"] == 1, "vix"] = vix_force_gate
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_path = Path(args.out) if args.out else CACHE / "regime_overlay.parquet"

    r = build_regime()
    r.to_parquet(out_path, index=False)
    risk_off_pct = r["risk_off"].mean()
    gates_mean = r["gates_on"].mean()
    print(f"[regime] wrote {len(r)} rows -> {out_path}")
    print(f"[regime] risk_off pct of days: {risk_off_pct:.1%}  avg gates: {gates_mean:.2f}")
    by_gate = r[["yc_inverted","claims_spike","be_inflation_hot",
                 "sentiment_shock","ff_tightening"]].mean()
    print("[regime] avg-fire rates:")
    for k, v in by_gate.items():
        print(f"  {k:20s}: {v:.1%}")


if __name__ == "__main__":
    main()

"""HC #575 R2 — Post-hoc regime-sizing overlay on the ETF rotation leader.

Cheap first cut: take the existing back-test book.parquet (daily_ret + gross_lev
per day) and multiply each day's return by a sizing factor derived from the
regime_modulator's risk_dial. Compare flat vs regime-sized on the same days.

If regime-sized improves pooled Sharpe / Calmar and pushes the deploy gate
margins wider, justify a full WF re-fit. If it just tracks the flat book or
trims drawdowns at the expense of Sharpe, document and stay flat.

Sizing mapping (multiplier applied to existing gross_lev):
    risk_dial = 1.00  ->  2.00  (size UP — HC #575 R2 "double when good")
    risk_dial = 0.85  ->  1.70
    risk_dial = 0.65  ->  1.30
    risk_dial = 0.40  ->  0.80
    risk_dial = 0.15  ->  0.30  (size DOWN — HC #575 R2 "halve when bad")

Outputs a small JSON report next to the leader's metrics.json.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(ROOT))

LEADER_DIR = ROOT / "output/macro_picker/etf_rotation_regime_20260608_164726_hold21_longonly"
SPY_PARQUET = ROOT / "output/macro_swing_v1/spy_daily.parquet"

TRADING_DAYS = 252


def _sizing_factor(risk_dial: float) -> float:
    """Map risk_dial in [0.15, 1.0] -> sizing multiplier in [0.30, 2.00]."""
    return float(np.clip(2.0 * risk_dial, 0.30, 2.00))


def _summary(daily_ret: pd.Series, dates: pd.Series, name: str) -> dict:
    s = daily_ret.dropna()
    n = len(s)
    if n < 20:
        return {"name": name, "n_days": n, "sharpe": None, "sortino": None}
    mean_d = s.mean()
    sd_d = s.std(ddof=1)
    downside = s[s < 0].std(ddof=1)
    sharpe = (mean_d / sd_d) * np.sqrt(TRADING_DAYS) if sd_d > 0 else 0.0
    sortino = (mean_d / downside) * np.sqrt(TRADING_DAYS) if downside and downside > 0 else None
    eq = (1.0 + s).cumprod()
    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    max_dd = float(dd.min())
    span_days = (pd.Timestamp(dates.max()) - pd.Timestamp(dates.min())).days
    span_years = span_days / 365.25 if span_days > 0 else 1.0
    cagr = float(eq.iloc[-1] ** (1.0 / span_years) - 1.0) if eq.iloc[-1] > 0 else None
    calmar = (cagr / abs(max_dd)) if (cagr is not None and max_dd < 0) else None
    wins = (s > 0).sum()
    losses = (s < 0).sum()
    pf = (s[s > 0].sum() / -s[s < 0].sum()) if losses else None
    wr = float(wins / (wins + losses)) if (wins + losses) > 0 else None
    return {
        "name": name,
        "n_days": int(n),
        "sharpe": float(sharpe),
        "sortino": float(sortino) if sortino is not None else None,
        "cagr": cagr,
        "max_dd": max_dd,
        "calmar": float(calmar) if calmar is not None else None,
        "pf": float(pf) if pf is not None else None,
        "wr": wr,
        "n_trading_days": int(span_days),
    }


def main() -> None:
    from strategy.macro_picker.regime_modulator import regime_series

    # Load leader back-test book.
    book = pd.read_parquet(LEADER_DIR / "book.parquet")
    book["date"] = pd.to_datetime(book["date"])
    book = book.sort_values("date").reset_index(drop=True)

    # Daily regime series from the modulator.
    rs = regime_series()
    rs["date"] = pd.to_datetime(rs["date"])
    rs = rs[["date", "state", "risk_dial", "severity"]]

    # Join on date — every book day picks up the regime that was active.
    df = book.merge(rs, on="date", how="left")
    # PIT-safe fill for any missing dial (e.g. holiday alignment).
    df["risk_dial"] = df["risk_dial"].ffill().fillna(0.65)
    df["state"] = df["state"].ffill().fillna("neutral")

    # Sizing factor + regime-sized daily return.
    df["sizing_factor"] = df["risk_dial"].apply(_sizing_factor)
    df["ret_flat"] = pd.to_numeric(df["daily_ret"], errors="coerce")
    df["ret_regime_sized"] = df["ret_flat"] * df["sizing_factor"]

    # SPY benchmark on the same dates.
    spy = pd.read_parquet(SPY_PARQUET)
    spy["date"] = pd.to_datetime(spy["date"])
    spy = spy.sort_values("date").reset_index(drop=True)
    spy["spy_ret"] = spy["adj_close"].pct_change()
    df = df.merge(spy[["date", "spy_ret"]], on="date", how="left")

    # Three books for comparison: flat (the existing leader), regime-sized, SPY.
    flat = _summary(df["ret_flat"], df["date"], "etf_rotation_flat (leader)")
    sized = _summary(df["ret_regime_sized"], df["date"], "etf_rotation_regime_sized")
    bench = _summary(df["spy_ret"], df["date"], "SPY buy-and-hold")

    # Stratify per regime state.
    by_state = []
    for st, g in df.groupby("state"):
        n = len(g)
        s_flat = g["ret_flat"].dropna()
        s_sz = g["ret_regime_sized"].dropna()
        sf = float((s_flat.mean() / s_flat.std(ddof=1)) * np.sqrt(TRADING_DAYS)) if s_flat.std(ddof=1) > 0 else None
        ss = float((s_sz.mean() / s_sz.std(ddof=1)) * np.sqrt(TRADING_DAYS)) if s_sz.std(ddof=1) > 0 else None
        by_state.append({
            "state": st,
            "n_days": int(n),
            "sharpe_flat": sf,
            "sharpe_regime_sized": ss,
        })

    # Verdict: regime-sized passes the deploy upgrade if pooled Sharpe and Calmar
    # both improve materially (≥ 5% relative) without worsening MaxDD by > 30%.
    def _verdict() -> dict:
        if not (sized["sharpe"] and flat["sharpe"]):
            return {"verdict": "INSUFFICIENT_DATA"}
        d_sharpe = sized["sharpe"] - flat["sharpe"]
        rel_sharpe = d_sharpe / abs(flat["sharpe"]) if flat["sharpe"] else 0.0
        d_calmar = (sized["calmar"] or 0) - (flat["calmar"] or 0)
        d_dd_rel = ((sized["max_dd"] or 0) - (flat["max_dd"] or 0)) / abs(flat["max_dd"] or 1e-6)
        passes = (rel_sharpe >= 0.05) and (d_calmar >= 0) and (d_dd_rel >= -0.30)
        return {
            "verdict": "PROMOTE_REGIME_SIZED" if passes else "KEEP_FLAT",
            "delta_sharpe_abs": float(d_sharpe),
            "delta_sharpe_rel": float(rel_sharpe),
            "delta_calmar_abs": float(d_calmar),
            "delta_maxdd_rel": float(d_dd_rel),
            "criteria": "ΔSharpe ≥ +5% rel AND ΔCalmar ≥ 0 AND MaxDD not worse by >30% rel",
        }

    verdict = _verdict()

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "hc": "HC #575 R2 (post-hoc regime-sized overlay first cut)",
        "leader_dir": str(LEADER_DIR),
        "sizing_rule": "gross_lev_eff = gross_lev * clip(2 * risk_dial, 0.30, 2.00)",
        "summary": {"flat_leader": flat, "regime_sized": sized, "spy_buy_hold": bench},
        "by_regime_state": by_state,
        "verdict": verdict,
        "notes": (
            "This is a POST-HOC overlay on the existing daily_ret stream — it does NOT re-run "
            "the walk-forward picker. A positive verdict here justifies a full WF re-fit "
            "(rotation_v1 with risk_dial wired into the sizing step). A negative verdict means "
            "the regime-sized variant does not improve the leader at the daily-return level, "
            "and we stay flat per HC #575 R2."
        ),
    }

    out_path = LEADER_DIR / "regime_sized_overlay.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps({
        "flat_sharpe": flat["sharpe"],
        "sized_sharpe": sized["sharpe"],
        "flat_calmar": flat["calmar"],
        "sized_calmar": sized["calmar"],
        "flat_maxdd": flat["max_dd"],
        "sized_maxdd": sized["max_dd"],
        "flat_cagr": flat["cagr"],
        "sized_cagr": sized["cagr"],
        "spy_sharpe": bench["sharpe"],
        "spy_cagr": bench["cagr"],
        "verdict": verdict,
        "report_path": str(out_path),
    }, indent=2))


if __name__ == "__main__":
    main()

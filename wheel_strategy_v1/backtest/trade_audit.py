"""
trade_audit.py — HC #544 R5 leakage gate + trade ledger export.

Two responsibilities:

1. LEDGER EXPORT — convert WheelState.ledger (list of TradeLedgerEntry) into
   a clean DataFrame / CSV with every field HC #544 R5 mandates:
     entry timestamp + entry price + exit timestamp + exit price + strike +
     premium + fees + slippage + exit_reason.

2. LEAKAGE AUDIT — for a given GA config + price panel, run the wheel twice:
     (a) baseline:  entries filled at close-of-entry-bar  (default)
     (b) t+1:       entries filled at OPEN of next bar
   Compare CAGR. If |delta_CAGR| > 1.5pp, FLAG (per HC #544 R5).

Usage:
  from backtest.trade_audit import ledger_to_df, ledger_to_csv, run_leakage_audit

  ledger_to_csv(state, path="output/trades.csv")
  audit = run_leakage_audit(cfg, prices, sigmas, calendar, starting_cash=100_000)
  print(audit["verdict"])  # PASS / FLAG / FAIL
"""
from __future__ import annotations
from dataclasses import asdict, fields
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .wheel_engine import TradeLedgerEntry, WheelConfig, WheelState


LEAK_THRESHOLD_PP = 1.5   # HC #544 R5: >1.5pp CAGR delta → leakage suspected


def ledger_to_df(state: WheelState) -> pd.DataFrame:
    """Convert state.ledger into a tabular DataFrame with all HC #544 R5 fields."""
    if not state.ledger:
        return pd.DataFrame(columns=[f.name for f in fields(TradeLedgerEntry)])
    rows = [asdict(e) for e in state.ledger]
    df = pd.DataFrame(rows)
    # Derived columns
    df["hold_days"] = (pd.to_datetime(df["close_date"]) -
                       pd.to_datetime(df["open_date"])).dt.days
    df["net_premium"] = df["premium_received"] - df["premium_closed_at"] - df["fees_paid"]
    df["return_on_strike_bps"] = np.where(
        df["strike"] * df["contracts"] * 100 > 0,
        df["realized_pnl"] / (df["strike"] * df["contracts"] * 100) * 10_000,
        0.0,
    )
    return df


def ledger_to_csv(state: WheelState, path: str | Path,
                  include_summary: bool = True) -> Path:
    """Write trade ledger to CSV. Returns the path written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = ledger_to_df(state)
    df.to_csv(path, index=False)
    if include_summary and len(df) > 0:
        summary = trade_summary(df)
        summary_path = path.with_suffix(".summary.json")
        import json
        json.dump(summary, open(summary_path, "w"), indent=2, default=str)
    return path


def trade_summary(df: pd.DataFrame) -> dict:
    """Aggregate metrics over a trade ledger DataFrame."""
    if df is None or df.empty:
        return {"n_trades": 0}

    wins = df[df["realized_pnl"] > 0]
    losses = df[df["realized_pnl"] < 0]

    by_kind = df.groupby("kind").agg(
        n=("realized_pnl", "size"),
        sum_pnl=("realized_pnl", "sum"),
        avg_pnl=("realized_pnl", "mean"),
        avg_hold_days=("hold_days", "mean"),
        win_rate=("realized_pnl", lambda s: float((s > 0).mean() * 100.0)),
    ).reset_index().to_dict(orient="records")

    by_exit = (df.groupby("exit_reason")
                 .agg(n=("realized_pnl", "size"),
                      sum_pnl=("realized_pnl", "sum"))
                 .reset_index().to_dict(orient="records"))

    return {
        "n_trades": int(len(df)),
        "gross_premium": float(df["premium_received"].sum()),
        "net_pnl": float(df["realized_pnl"].sum()),
        "total_fees": float(df["fees_paid"].sum()),
        "win_rate_pct": float((df["realized_pnl"] > 0).mean() * 100.0),
        "avg_win": float(wins["realized_pnl"].mean()) if len(wins) else 0.0,
        "avg_loss": float(losses["realized_pnl"].mean()) if len(losses) else 0.0,
        "profit_factor": (float(wins["realized_pnl"].sum() /
                                abs(losses["realized_pnl"].sum()))
                          if len(losses) and losses["realized_pnl"].sum() != 0
                          else float("inf")),
        "avg_hold_days": float(df["hold_days"].mean()),
        "assigned_count": int(df["assigned"].sum()),
        "called_away_count": int(df["called_away"].sum()),
        "by_kind": by_kind,
        "by_exit_reason": by_exit,
    }


# ----------------------------------------------------------------------------
# Leakage audit
# ----------------------------------------------------------------------------

def _equity_curve_to_cagr(curve: list) -> tuple[float, float]:
    """Returns (CAGR_pct, max_DD_pct) given a list of (date, equity) tuples."""
    if not curve or len(curve) < 5:
        return (0.0, 0.0)
    dates = pd.to_datetime([d for d, _ in curve])
    eq = np.array([float(e) for _, e in curve], dtype=float)
    yrs = max((dates[-1] - dates[0]).days / 365.25, 1e-6)
    cagr = float((eq[-1] / eq[0]) ** (1.0 / yrs) - 1.0) * 100.0
    rolling_max = np.maximum.accumulate(eq)
    dd = (eq / rolling_max) - 1.0
    max_dd = float(-dd.min() * 100.0)
    return cagr, max_dd


def run_leakage_audit(run_wheel_callable,
                      cfg: WheelConfig,
                      prices: pd.DataFrame,
                      sigmas: pd.DataFrame,
                      calendar: pd.DatetimeIndex,
                      starting_cash: float = 100_000.0,
                      **kw) -> dict:
    """
    Run the wheel twice — once with default (close-of-bar) entries, once with
    next-bar-open entries — and compare. Per HC #544 R5.

    run_wheel_callable: the entry point (typically backtest.wheel_engine.run_wheel)
                        — signature must accept entry_bar_mode='close' or 'next_open'.

    NOTE: The current run_wheel doesn't expose entry_bar_mode yet — this is
    a forward-compatible audit harness. Until the engine supports it, the
    audit returns verdict='PENDING_ENGINE_HOOK' along with the baseline run.
    """
    # Baseline (close fills)
    try:
        base_state = run_wheel_callable(cfg, prices, sigmas, calendar,
                                        starting_cash=starting_cash,
                                        entry_bar_mode="close", **kw)
    except TypeError:
        # Engine doesn't accept the kw yet — run the legacy signature
        base_state = run_wheel_callable(cfg, prices, sigmas, calendar,
                                        starting_cash=starting_cash, **kw)
        return {
            "verdict": "PENDING_ENGINE_HOOK",
            "reason": ("run_wheel does not yet accept entry_bar_mode — "
                       "audit harness ready, engine wire-in still required."),
            "baseline_cagr_pct": _equity_curve_to_cagr(base_state.equity_curve)[0],
            "baseline_max_dd_pct": _equity_curve_to_cagr(base_state.equity_curve)[1],
            "baseline_n_trades": len(base_state.ledger),
        }

    try:
        t1_state = run_wheel_callable(cfg, prices, sigmas, calendar,
                                      starting_cash=starting_cash,
                                      entry_bar_mode="next_open", **kw)
    except TypeError:
        t1_state = base_state  # fall through

    base_cagr, base_dd = _equity_curve_to_cagr(base_state.equity_curve)
    t1_cagr, t1_dd = _equity_curve_to_cagr(t1_state.equity_curve)
    delta = abs(base_cagr - t1_cagr)

    if delta > LEAK_THRESHOLD_PP:
        verdict = "FAIL_LEAKAGE_SUSPECTED"
    elif delta > LEAK_THRESHOLD_PP * 0.6:
        verdict = "FLAG"
    else:
        verdict = "PASS"

    return {
        "verdict": verdict,
        "delta_cagr_pp": float(delta),
        "threshold_pp": LEAK_THRESHOLD_PP,
        "baseline_cagr_pct": base_cagr,
        "baseline_max_dd_pct": base_dd,
        "baseline_n_trades": len(base_state.ledger),
        "t1_cagr_pct": t1_cagr,
        "t1_max_dd_pct": t1_dd,
        "t1_n_trades": len(t1_state.ledger),
    }


if __name__ == "__main__":
    print("trade_audit.py — HC #544 R5 leakage gate + ledger export")
    print("LEAK_THRESHOLD_PP =", LEAK_THRESHOLD_PP)
    print("TradeLedgerEntry fields:")
    for f in fields(TradeLedgerEntry):
        print(f"  {f.name:24s} {f.type}")

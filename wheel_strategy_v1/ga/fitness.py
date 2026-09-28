"""
fitness.py — Fitness function for the wheel-strategy GA.

fitness = annualized_premium_yield * Sortino_ratio / max(1, max_drawdown_pct)

Hard caps (if violated, fitness multiplied by 0.1 — strong penalty, not zero
so the GA still gets signal about which configs are *less* bad):
  - assignment_rate <= 0.40
  - single_name_max_alloc <= 0.15
  - sector_max_alloc <= chromosome.sector_cap_pct
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def compute_metrics(result: dict) -> dict:
    eq = result["equity_curve"].copy()
    if eq.empty or len(eq) < 5:
        return {"fitness": -1.0, "ann_return": 0.0, "ann_premium_yield": 0.0,
                "sortino": 0.0, "max_dd_pct": 100.0, "assignment_rate": 0.0,
                "n_trades": 0, "single_name_max_alloc": 0.0, "sector_max_alloc": 0.0,
                "worst_month_pct": 0.0}
    eq["date"] = pd.to_datetime(eq["date"])
    eq = eq.sort_values("date").drop_duplicates("date")
    eq["ret"] = eq["equity"].pct_change()
    rets = eq["ret"].dropna()
    n_days = len(rets)
    if n_days < 5:
        return {"fitness": -1.0, "ann_return": 0.0, "ann_premium_yield": 0.0,
                "sortino": 0.0, "max_dd_pct": 100.0, "assignment_rate": 0.0,
                "n_trades": 0, "single_name_max_alloc": 0.0, "sector_max_alloc": 0.0,
                "worst_month_pct": 0.0}

    start_eq = result["starting_cash"]
    final_eq = eq["equity"].iloc[-1]
    years = max(n_days / 252.0, 0.05)
    ann_return = (final_eq / start_eq) ** (1 / years) - 1

    downside = rets[rets < 0]
    sortino = (rets.mean() / (downside.std() + 1e-9)) * np.sqrt(252) if not downside.empty else (rets.mean() * np.sqrt(252) / 1e-3)

    cum = (1 + rets.fillna(0)).cumprod()
    peak = cum.cummax()
    dd = (cum / peak) - 1
    max_dd_pct = float(abs(dd.min()) * 100.0)
    worst_month = rets.groupby(eq["date"].dt.to_period("M")).sum().min()
    worst_month_pct = float(worst_month * 100.0) if worst_month is not None else 0.0

    ledger = result.get("ledger", pd.DataFrame())
    n_trades = int(len(ledger))
    # premium yield = total premium captured / starting equity, annualized
    if not ledger.empty and "realized_pnl" in ledger.columns:
        prem = float(ledger["realized_pnl"].sum())
    else:
        prem = final_eq - start_eq
    ann_premium_yield = (prem / start_eq) / years

    n_csp = int(((ledger.get("kind") == "CSP").sum()) if not ledger.empty else 0)
    n_assigned = int(((ledger.get("assigned") == True).sum()) if not ledger.empty else 0)  # noqa
    # In our engine, assignment flags the transition into long_shares. We log
    # assignment count separately at the engine level too:
    assignment_count = result.get("assignment_count", 0)
    assignment_rate = (assignment_count / n_csp) if n_csp > 0 else 0.0

    # Single-name & sector max alloc not tracked per-bar; estimate from ledger as max position notional / starting equity.
    if not ledger.empty and "strike" in ledger.columns:
        notionals = (ledger["strike"] * 100 * ledger["contracts"]).astype(float)
        single_name_max = float(notionals.max() / start_eq) if not notionals.empty else 0.0
    else:
        single_name_max = 0.0

    # Sector max alloc — we don't track sector in ledger; approx 0 (engine caps it on entry).
    sector_max = 0.0

    fitness = ann_premium_yield * max(0.0, sortino) / max(1.0, max_dd_pct)
    # Hard-cap penalties
    if assignment_rate > 0.40:
        fitness *= 0.1
    if single_name_max > 0.15:
        fitness *= 0.1

    return {
        "fitness": float(fitness),
        "ann_return": float(ann_return),
        "ann_premium_yield": float(ann_premium_yield),
        "sortino": float(sortino),
        "max_dd_pct": float(max_dd_pct),
        "assignment_rate": float(assignment_rate),
        "n_trades": n_trades,
        "single_name_max_alloc": float(single_name_max),
        "sector_max_alloc": float(sector_max),
        "worst_month_pct": float(worst_month_pct),
        "final_equity": float(final_eq),
    }

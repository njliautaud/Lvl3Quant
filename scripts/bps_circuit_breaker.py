#!/usr/bin/env python3
"""
BPS Combined Circuit Breaker Study
====================================

Tests a layered defense system for BPS strategy:

  Layer 1: VIX-scaled sizing (base at VIX=15, linear scale down)
  Layer 2: Sector caps (max 20-25% exposure per GICS sector)
  Layer 3: Portfolio-level circuit breaker (daily loss -> freeze new entries)
  Layer 4: VIX hard cutoff (no new entries when VIX > 35/40)

Combos tested:
  A) VIX-scale only (baseline)
  B) VIX-scale + sector cap 25%
  C) VIX-scale + sector cap 25% + portfolio CB (3% -> freeze 2 days)
  D) VIX-scale + sector cap 25% + portfolio CB + VIX hard cutoff 35
  E) Full defense (all layers, tightest params)

Uses 5% BA cost throughout. Trade data from bps_assignment_risk study.
"""

import sys, json, time
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

ROOT = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(ROOT / "output" / "wheel_higher_returns_study"))

OUTPUT = ROOT / "output" / "bps_circuit_breaker"
OUTPUT.mkdir(parents=True, exist_ok=True)

BA_COST_FRAC = 0.05


# ═══════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════

def load_all():
    """Load trades, macro/VIX, and sector mapping."""
    trades = pd.read_parquet(ROOT / "output" / "bps_assignment_risk" / "trades_close_1dte.parquet")
    trades["open_date"] = pd.to_datetime(trades["open_date"])
    trades["close_date"] = pd.to_datetime(trades["close_date"])

    from higher_returns_study import load_data
    prices, iv, macro, fund, universe, earnings = load_data()
    macro = macro.copy()
    macro["date"] = pd.to_datetime(macro["date"])

    sector_map = dict(zip(fund["ticker"], fund["sector"]))

    return trades, macro, sector_map


def apply_ba_cost(trades):
    """Apply 5% BA cost to P&L."""
    trades = trades.copy()
    open_cost = trades["net_credit"].abs() * BA_COST_FRAC * 2
    closed_early = trades["exit_type"].isin(["profit_take", "early_close_1DTE", "loss_stop"])
    close_cost = pd.Series(0.0, index=trades.index)
    close_cost[closed_early] = trades.loc[closed_early, "net_credit"].abs() * BA_COST_FRAC * 2
    trades["ba_cost"] = open_cost + close_cost
    trades["honest_pnl"] = trades["realized_pnl"] - trades["ba_cost"]
    return trades


# ═══════════════════════════════════════════════════════════════════
# Circuit Breaker Engine
# ═══════════════════════════════════════════════════════════════════

def run_circuit_breaker(trades, macro, sector_map,
                        # Layer 1: VIX scaling
                        vix_scale=True,
                        vix_base=15.0,
                        # Layer 2: Sector cap
                        sector_cap=None,  # e.g., 0.25
                        # Layer 3: Portfolio circuit breaker
                        portfolio_cb_threshold=None,  # e.g., -0.03
                        portfolio_cb_freeze_days=0,    # e.g., 2
                        # Layer 4: VIX hard cutoff
                        vix_hard_cutoff=None,  # e.g., 35
                        starting_capital=100_000,
                        label="test"):
    """
    Apply layered circuit breakers to trade stream and compute equity curve.

    Returns dict with metrics, equity curve, and diagnostics.
    """
    trades = trades.copy()
    trades = trades.sort_values("open_date").reset_index(drop=True)

    # Assign sectors
    trades["sector"] = trades["ticker"].map(sector_map).fillna("Unknown")

    # Build VIX lookup
    vix_by_date = macro.set_index("date")["vix"].to_dict()

    # ── Layer 1: VIX-scaled sizing ──
    if vix_scale:
        trades["open_vix"] = trades["open_date"].map(vix_by_date)

        def vix_scalar(vix):
            if pd.isna(vix):
                return 1.0
            if vix <= vix_base:
                return 1.0
            # Linear scale: at VIX=30 -> 0.5x, at VIX=45 -> 0x
            scale = max(0.0, 1.0 - (vix - vix_base) / 30.0)
            return scale

        trades["vix_scalar"] = trades["open_vix"].apply(vix_scalar)
    else:
        trades["vix_scalar"] = 1.0
        trades["open_vix"] = trades["open_date"].map(vix_by_date)

    # ── Layer 4: VIX hard cutoff (filter BEFORE other layers) ──
    trades_filtered = trades.copy()
    n_vix_blocked = 0
    if vix_hard_cutoff is not None:
        mask = trades_filtered["open_vix"].fillna(0) <= vix_hard_cutoff
        n_vix_blocked = (~mask).sum()
        trades_filtered = trades_filtered[mask].copy()

    # Scale P&L by VIX scalar
    trades_filtered["scaled_pnl"] = trades_filtered["honest_pnl"] * trades_filtered["vix_scalar"]

    # ── Simulate day-by-day with Layers 2 and 3 ──
    # Key: we must decide at OPEN TIME which trades are accepted.
    # Only accepted trades contribute P&L at close time.
    all_dates = sorted(trades_filtered["open_date"].unique())
    date_set = set(all_dates)

    # Group trades by open date
    trades_by_open = defaultdict(list)
    for _, row in trades_filtered.iterrows():
        trades_by_open[row["open_date"]].append(row)

    equity = starting_capital
    daily_equity = []
    daily_pnl_vals = []
    # Track accepted trades: key = (ticker, open_date) -> row
    accepted_trades = {}
    frozen_until = None  # date until which new entries are frozen
    n_sector_blocked = 0
    n_cb_frozen = 0
    n_accepted = 0
    sector_blocked_details = defaultdict(int)
    cb_trigger_dates = []

    # Build unified timeline from all open and close dates
    close_dates = set(trades_filtered["close_date"].unique())
    all_timeline = sorted(date_set | close_dates)

    for dt in all_timeline:
        day_pnl = 0.0

        # Process closings for today — ONLY for accepted trades
        keys_to_remove = []
        for key, tr in accepted_trades.items():
            if tr["close_date"] == dt:
                day_pnl += tr["scaled_pnl"]
                keys_to_remove.append(key)
        for key in keys_to_remove:
            del accepted_trades[key]

        equity += day_pnl
        daily_pnl_vals.append(day_pnl)
        daily_equity.append({"date": dt, "equity": equity, "daily_pnl": day_pnl})

        # ── Layer 3: Portfolio circuit breaker check ──
        if portfolio_cb_threshold is not None and len(daily_equity) >= 2:
            prev_eq = daily_equity[-2]["equity"]
            if prev_eq > 0:
                daily_ret = day_pnl / prev_eq
                if daily_ret < portfolio_cb_threshold:
                    freeze_end = dt + pd.Timedelta(days=portfolio_cb_freeze_days)
                    if frozen_until is None or freeze_end > frozen_until:
                        frozen_until = freeze_end
                        cb_trigger_dates.append({
                            "date": str(dt.date()),
                            "daily_return": round(daily_ret * 100, 2),
                            "freeze_until": str(freeze_end.date()),
                        })

        # Process openings for today
        opening_today = trades_by_open.get(dt, [])
        if not opening_today:
            continue

        # Check if frozen (Layer 3)
        if frozen_until is not None and dt <= frozen_until:
            n_cb_frozen += len(opening_today)
            continue

        # ── Layer 2: Sector cap ──
        if sector_cap is not None:
            # Count active exposure by sector from accepted trades
            sector_exposure = defaultdict(float)
            total_exposure = 0.0
            for key, t in accepted_trades.items():
                margin = t["spread_width"] * 100 * t["contracts"]
                sector_exposure[t["sector"]] += margin
                total_exposure += margin

            for tr in opening_today:
                sec = tr["sector"]
                margin_this = tr["spread_width"] * 100 * tr["contracts"]
                denom = max(total_exposure + margin_this, equity, 1.0)
                sector_pct = (sector_exposure.get(sec, 0) + margin_this) / denom

                if sector_pct > sector_cap:
                    n_sector_blocked += 1
                    sector_blocked_details[sec] += 1
                    continue

                # Accept trade
                key = (tr["ticker"], tr["open_date"])
                accepted_trades[key] = tr
                sector_exposure[sec] = sector_exposure.get(sec, 0) + margin_this
                total_exposure += margin_this
                n_accepted += 1
        else:
            for tr in opening_today:
                key = (tr["ticker"], tr["open_date"])
                accepted_trades[key] = tr
                n_accepted += 1

    # ── Build equity curve and compute metrics ──
    eq_df = pd.DataFrame(daily_equity)
    if eq_df.empty or len(eq_df) < 30:
        return {"label": label, "error": "insufficient data"}

    eq_df["date"] = pd.to_datetime(eq_df["date"])
    eq_df = eq_df.sort_values("date").reset_index(drop=True)
    eq_df["ret"] = eq_df["equity"].pct_change()
    rets = eq_df["ret"].dropna()

    # CAGR
    total_days = (eq_df["date"].iloc[-1] - eq_df["date"].iloc[0]).days
    total_years = max(total_days / 365.25, 0.01)
    total_return = eq_df["equity"].iloc[-1] / starting_capital
    cagr = (total_return ** (1 / total_years)) - 1 if total_return > 0 else -1.0

    # Sharpe
    sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0

    # Sortino
    downside = rets[rets < 0]
    sortino = float(rets.mean() / downside.std() * np.sqrt(252)) if len(downside) > 0 and downside.std() > 0 else 0.0

    # Max drawdown
    eq_df["peak"] = eq_df["equity"].cummax()
    eq_df["dd"] = (eq_df["equity"] - eq_df["peak"]) / eq_df["peak"]
    max_dd = float(eq_df["dd"].min())
    max_dd_date = eq_df.loc[eq_df["dd"].idxmin(), "date"]

    # Worst single day
    worst_day_idx = rets.idxmin()
    worst_day_ret = float(rets.min())
    worst_day_date = eq_df.loc[worst_day_idx, "date"] if worst_day_idx in eq_df.index else "unknown"

    # Recovery time from max drawdown
    trough_idx = eq_df["dd"].idxmin()
    trough_eq = eq_df.loc[trough_idx, "equity"]
    pre_dd_peak = eq_df.loc[trough_idx, "peak"]
    post_trough = eq_df.loc[trough_idx:]
    recovered = post_trough[post_trough["equity"] >= pre_dd_peak]
    if len(recovered) > 0:
        recovery_date = recovered.iloc[0]["date"]
        recovery_days = (recovery_date - max_dd_date).days
    else:
        recovery_date = None
        recovery_days = None  # never recovered

    # Win rate (daily)
    wr = float(len(rets[rets > 0]) / len(rets) * 100) if len(rets) > 0 else 0.0

    # Profit factor
    wins = rets[rets > 0].sum()
    losses = abs(rets[rets < 0].sum())
    pf = float(wins / losses) if losses > 0 else float("inf")

    # Calmar ratio
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

    # Trades affected stats
    total_trades = len(trades)
    trades_after_ba = len(trades_filtered)
    trades_sector_blocked = n_sector_blocked
    trades_cb_frozen = n_cb_frozen

    metrics = {
        "label": label,
        "cagr_pct": round(cagr * 100, 2),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "max_dd_pct": round(max_dd * 100, 2),
        "max_dd_date": str(max_dd_date.date()) if hasattr(max_dd_date, 'date') else str(max_dd_date),
        "worst_day_pct": round(worst_day_ret * 100, 3),
        "worst_day_date": str(worst_day_date.date()) if hasattr(worst_day_date, 'date') else str(worst_day_date),
        "recovery_days": recovery_days,
        "recovery_date": str(recovery_date.date()) if recovery_date is not None and hasattr(recovery_date, 'date') else str(recovery_date),
        "win_rate_pct": round(wr, 1),
        "profit_factor": round(pf, 2),
        "calmar": round(calmar, 2),
        "final_equity": round(eq_df["equity"].iloc[-1], 2),
        "total_trades_original": total_trades,
        "trades_vix_blocked": n_vix_blocked,
        "trades_sector_blocked": trades_sector_blocked,
        "trades_cb_frozen": trades_cb_frozen,
        "pct_trades_affected": round(
            (n_vix_blocked + trades_sector_blocked + trades_cb_frozen) / max(total_trades, 1) * 100, 1),
        "cb_trigger_count": len(cb_trigger_dates),
    }

    diagnostics = {
        "sector_blocked_by_sector": dict(sector_blocked_details),
        "cb_trigger_dates": cb_trigger_dates[:20],  # top 20
    }

    return {
        "metrics": metrics,
        "equity_curve": eq_df[["date", "equity", "daily_pnl"]],
        "diagnostics": diagnostics,
    }


# ═══════════════════════════════════════════════════════════════════
# Sensitivity Sweep (Layer 3 parameters)
# ═══════════════════════════════════════════════════════════════════

def sweep_portfolio_cb(trades, macro, sector_map):
    """Sweep portfolio CB threshold and freeze duration."""
    print("\n" + "=" * 60)
    print("PORTFOLIO CB SENSITIVITY SWEEP")
    print("(VIX-scale + sector 25% + varying CB params)")
    print("=" * 60)

    thresholds = [-0.02, -0.03, -0.05]
    freeze_days = [1, 2, 3]

    results = {}
    for thresh in thresholds:
        for freeze in freeze_days:
            lbl = f"CB_{abs(thresh)*100:.0f}pct_freeze{freeze}d"
            r = run_circuit_breaker(
                trades, macro, sector_map,
                vix_scale=True, sector_cap=0.25,
                portfolio_cb_threshold=thresh,
                portfolio_cb_freeze_days=freeze,
                label=lbl,
            )
            m = r["metrics"]
            print(f"  {lbl}: Sharpe={m['sharpe']:.2f}  Sortino={m['sortino']:.2f}  "
                  f"MaxDD={m['max_dd_pct']:.1f}%  CB triggers={m['cb_trigger_count']}")
            results[lbl] = m

    return results


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 70)
    print("BPS COMBINED CIRCUIT BREAKER STUDY")
    print("Layered defense: VIX-scale / Sector caps / Portfolio CB / VIX cutoff")
    print("=" * 70)

    trades, macro, sector_map = load_all()
    trades = apply_ba_cost(trades)

    n_trades = len(trades)
    n_tickers = trades["ticker"].nunique()
    date_range = f"{trades['open_date'].min().date()} to {trades['close_date'].max().date()}"
    print(f"\nData: {n_trades} trades, {n_tickers} tickers, {date_range}")

    # ── Define combos ──
    combos = {
        "A_VIX_scale_only": dict(
            vix_scale=True, sector_cap=None,
            portfolio_cb_threshold=None, portfolio_cb_freeze_days=0,
            vix_hard_cutoff=None,
        ),
        "B_VIX_scale_sector25": dict(
            vix_scale=True, sector_cap=0.25,
            portfolio_cb_threshold=None, portfolio_cb_freeze_days=0,
            vix_hard_cutoff=None,
        ),
        "C_VIX_scale_sector25_CB3pct2d": dict(
            vix_scale=True, sector_cap=0.25,
            portfolio_cb_threshold=-0.03, portfolio_cb_freeze_days=2,
            vix_hard_cutoff=None,
        ),
        "D_VIX_scale_sector25_CB3pct2d_cutoff35": dict(
            vix_scale=True, sector_cap=0.25,
            portfolio_cb_threshold=-0.03, portfolio_cb_freeze_days=2,
            vix_hard_cutoff=35,
        ),
        "E_full_defense": dict(
            vix_scale=True, sector_cap=0.20,
            portfolio_cb_threshold=-0.02, portfolio_cb_freeze_days=3,
            vix_hard_cutoff=35,
        ),
        "F_optimized": dict(
            vix_scale=True, sector_cap=0.25,
            portfolio_cb_threshold=-0.02, portfolio_cb_freeze_days=1,
            vix_hard_cutoff=None,
        ),
        "G_VIX_scale_CB2pct1d_only": dict(
            vix_scale=True, sector_cap=None,
            portfolio_cb_threshold=-0.02, portfolio_cb_freeze_days=1,
            vix_hard_cutoff=None,
        ),
        "H_VIX_cutoff30": dict(
            vix_scale=True, sector_cap=0.25,
            portfolio_cb_threshold=-0.02, portfolio_cb_freeze_days=1,
            vix_hard_cutoff=30,
        ),
    }

    all_results = {}
    all_eq_curves = {}

    for name, params in combos.items():
        print(f"\n{'─'*60}")
        print(f"  COMBO: {name}")
        print(f"{'─'*60}")
        result = run_circuit_breaker(trades, macro, sector_map, label=name, **params)
        m = result["metrics"]
        all_results[name] = result
        all_eq_curves[name] = result["equity_curve"]

        print(f"  CAGR:        {m['cagr_pct']:>8.1f}%")
        print(f"  Sharpe:      {m['sharpe']:>8.2f}")
        print(f"  Sortino:     {m['sortino']:>8.2f}")
        print(f"  Max DD:      {m['max_dd_pct']:>8.1f}%  (on {m['max_dd_date']})")
        print(f"  Worst day:   {m['worst_day_pct']:>8.2f}%  (on {m['worst_day_date']})")
        print(f"  Recovery:    {m['recovery_days']} days")
        print(f"  Calmar:      {m['calmar']:>8.2f}")
        print(f"  Win rate:    {m['win_rate_pct']:>8.1f}%")
        print(f"  PF:          {m['profit_factor']:>8.2f}")
        print(f"  VIX blocked: {m['trades_vix_blocked']:>6d}")
        print(f"  Sector blk:  {m['trades_sector_blocked']:>6d}")
        print(f"  CB frozen:   {m['trades_cb_frozen']:>6d}")
        print(f"  % affected:  {m['pct_trades_affected']:>8.1f}%")
        print(f"  CB triggers: {m['cb_trigger_count']:>6d}")

    # ── CB Sensitivity Sweep ──
    sweep_results = sweep_portfolio_cb(trades, macro, sector_map)

    # ── Comparison Table ──
    print("\n" + "=" * 120)
    print("COMPARISON TABLE")
    print("=" * 120)
    header = (f"{'Combo':<42} {'CAGR':>7} {'Sharpe':>7} {'Sortino':>8} {'MaxDD':>7} "
              f"{'Worst Day':>10} {'Recovery':>9} {'Calmar':>7} {'%Affected':>10}")
    print(header)
    print("-" * 120)
    for name in combos:
        m = all_results[name]["metrics"]
        rec = f"{m['recovery_days']}d" if m['recovery_days'] is not None else "never"
        print(f"{name:<42} {m['cagr_pct']:>6.1f}% {m['sharpe']:>7.2f} {m['sortino']:>8.2f} "
              f"{m['max_dd_pct']:>6.1f}% {m['worst_day_pct']:>9.2f}% {rec:>9} "
              f"{m['calmar']:>7.2f} {m['pct_trades_affected']:>9.1f}%")

    # ── Find best risk/return tradeoff ──
    print("\n" + "=" * 70)
    print("BEST RISK/RETURN TRADEOFF ANALYSIS")
    print("=" * 70)

    # Score: Sortino * (1 - |MaxDD|/100) — rewards high risk-adjusted return while penalizing drawdown
    scores = {}
    for name in combos:
        m = all_results[name]["metrics"]
        # Composite score: Sortino penalized by drawdown severity
        dd_penalty = 1.0 - abs(m["max_dd_pct"]) / 100.0
        score = m["sortino"] * dd_penalty
        scores[name] = {
            "sortino": m["sortino"],
            "max_dd_pct": m["max_dd_pct"],
            "dd_penalty": round(dd_penalty, 3),
            "composite_score": round(score, 3),
        }
        print(f"  {name:<42} Score={score:.3f}  (Sortino={m['sortino']:.2f} * DD_pen={dd_penalty:.3f})")

    best = max(scores, key=lambda k: scores[k]["composite_score"])
    print(f"\n  >>> BEST COMBO: {best}")
    print(f"      Score: {scores[best]['composite_score']:.3f}")

    # ── COVID Period Deep-dive ──
    print("\n" + "=" * 70)
    print("COVID PERIOD ANALYSIS (Feb-Apr 2020)")
    print("=" * 70)
    for name in combos:
        eq = all_eq_curves[name]
        covid = eq[(eq["date"] >= "2020-02-01") & (eq["date"] <= "2020-04-30")]
        if len(covid) > 2:
            covid_ret = (covid["equity"].iloc[-1] / covid["equity"].iloc[0] - 1) * 100
            covid_dd = ((covid["equity"] / covid["equity"].cummax()) - 1).min() * 100
            worst_day = covid["daily_pnl"].min()
            print(f"  {name:<42} Return={covid_ret:>7.1f}%  MaxDD={covid_dd:>7.1f}%  "
                  f"Worst day P&L=${worst_day:>10,.0f}")

    # ── Save Results ──
    save_data = {
        "generated": pd.Timestamp.now().isoformat(),
        "ba_cost_frac": BA_COST_FRAC,
        "n_trades": n_trades,
        "date_range": date_range,
        "combos": {},
        "sweep_results": sweep_results,
        "scores": scores,
        "best_combo": best,
    }
    for name in combos:
        r = all_results[name]
        save_data["combos"][name] = {
            "metrics": r["metrics"],
            "diagnostics": r["diagnostics"],
        }

    with open(OUTPUT / "circuit_breaker_results.json", "w") as f:
        json.dump(save_data, f, indent=2, default=str)

    # Save equity curves
    for name in combos:
        eq = all_eq_curves[name]
        eq.to_parquet(OUTPUT / f"eq_{name}.parquet")

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"DONE in {elapsed:.1f}s")
    print(f"Results saved to {OUTPUT}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

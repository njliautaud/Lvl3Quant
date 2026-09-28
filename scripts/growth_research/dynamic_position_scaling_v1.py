#!/usr/bin/env python3
"""
Dynamic Position Scaling v1 — Monte Carlo with Capital-Adaptive Sizing
=======================================================================
Tests the capital scaling roadmap's key recommendation: increase position
sizes as capital grows to maintain growth rate.

Compares 5 position sizing rules:
A) Fixed $200/spread (current)
B) Fixed fraction: 30% of equity per position (Kelly-inspired)
C) Tiered: $200 at <$2K, $500 at $2K-$10K, $1000 at >$10K
D) Sqrt scaling: $200 × sqrt(equity/$645) — moderate growth
E) Linear scaling: $200 × (equity/$645) capped at $2000 — aggressive

Uses calibrated sector bull spread returns (WR 80.1%, PF 22.22).
10K Monte Carlo paths, 5yr horizon.
"""
import json, os, time, warnings
from datetime import datetime
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")

BASE_PATH = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_PATH / "output" / "growth_research" / "dynamic_position_scaling_v1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_SIMS = 10_000
N_YEARS = 5
PERIODS_PER_YEAR = 26  # bi-weekly
INITIAL_CAPITAL = 645.0
TRADES_PER_PERIOD = 3
COMMISSION_PER_TRADE = 2.60
MILESTONES = [1_000, 2_000, 5_000, 10_000, 25_000, 50_000]

# Calibrated from sector_options_rotation_v1 backtest
# PF=22.22, WR=80.1% → avg_win/avg_loss = 5.52
# Trade-level returns as % of position:
WR = 0.801
AVG_WIN_PCT = 0.09     # 9% of position → $18 on $200
WIN_STD = 0.05
AVG_LOSS_PCT = -0.016  # 1.6% of position → $3.2 on $200
LOSS_STD = 0.01

def fprint(*a, **kw): print(*a, **kw, flush=True)

# ─── Position Sizing Rules ───
def pos_fixed(equity):
    """A) Fixed $200 per spread, max 3 concurrent."""
    return min(200, equity / TRADES_PER_PERIOD)

def pos_fraction(equity):
    """B) Fixed 30% of equity per position (Kelly-inspired)."""
    return equity * 0.30

def pos_tiered(equity):
    """C) Tiered: scales up at capital milestones."""
    if equity < 2000:
        return min(200, equity / TRADES_PER_PERIOD)
    elif equity < 10000:
        return min(500, equity / TRADES_PER_PERIOD)
    else:
        return min(1000, equity / TRADES_PER_PERIOD)

def pos_sqrt(equity):
    """D) Sqrt scaling: moderate growth."""
    base = 200 * np.sqrt(equity / INITIAL_CAPITAL)
    return min(base, equity / TRADES_PER_PERIOD)

def pos_linear(equity):
    """E) Linear scaling capped at $2000."""
    base = min(2000, 200 * (equity / INITIAL_CAPITAL))
    return min(base, equity / TRADES_PER_PERIOD)

SIZING_RULES = {
    "A_fixed": ("Fixed $200/spread", pos_fixed),
    "B_fraction": ("30% of equity/position", pos_fraction),
    "C_tiered": ("Tiered ($200/$500/$1K)", pos_tiered),
    "D_sqrt": ("Sqrt scaling", pos_sqrt),
    "E_linear": ("Linear scaling (cap $2K)", pos_linear),
}

def gen_trade_pnl(rng, pos_size, n_trades):
    """Generate dollar PnL for trades given position size."""
    is_win = rng.random(n_trades) < WR
    wins = np.clip(rng.normal(AVG_WIN_PCT, WIN_STD, n_trades), 0.005, 0.50)
    losses = np.clip(rng.normal(AVG_LOSS_PCT, LOSS_STD, n_trades), -1.0, -0.001)
    pct_ret = np.where(is_win, wins, losses)
    dollar_pnl = pct_ret * pos_size - COMMISSION_PER_TRADE
    return dollar_pnl

def simulate_path(rng, sizing_fn, n_years=N_YEARS):
    """Simulate one path with given position sizing rule."""
    n_periods = PERIODS_PER_YEAR * n_years
    equity = INITIAL_CAPITAL
    curve = [equity]
    max_equity = equity

    for pi in range(n_periods):
        if equity < 50:
            curve.extend([equity] * (n_periods - pi))
            break

        pos = sizing_fn(equity)
        if pos < 10:  # can't afford a trade
            curve.append(equity)
            continue

        # Check if total exposure exceeds equity
        total_exposure = pos * TRADES_PER_PERIOD
        if total_exposure > equity * 1.5:
            # Leveraged too much — cap at 1.5x equity
            pos = equity * 1.5 / TRADES_PER_PERIOD

        pnl = gen_trade_pnl(rng, pos, TRADES_PER_PERIOD)
        equity = max(0, equity + np.sum(pnl))
        max_equity = max(max_equity, equity)
        curve.append(equity)

    return np.array(curve[:n_periods + 1])

def analyze_curves(curves, label):
    """Compute comprehensive statistics."""
    finals = np.array([c[-1] for c in curves])
    n = len(curves)
    pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]

    # Final equity
    eq_pcts = {f"p{p}": float(np.percentile(finals, p)) for p in pcts}
    eq_pcts["mean"] = float(np.mean(finals))

    # Milestones
    ms = {}
    for tgt in MILESTONES:
        hits = [np.where(c >= tgt)[0][0] for c in curves if np.any(c >= tgt)]
        if hits:
            ms[f"${tgt:,}"] = {
                "med_months": round(float(np.median(hits)) / PERIODS_PER_YEAR * 12, 1),
                "pct_reach": round(len(hits) / n * 100, 1)
            }
        else:
            ms[f"${tgt:,}"] = {"med_months": None, "pct_reach": 0.0}

    # Ruin
    ruin = sum(1 for f in finals if f < 100) / n

    # Max drawdown
    dds = []
    for c in curves:
        rm = np.maximum.accumulate(c)
        dd = np.min((c - rm) / np.where(rm > 0, rm, 1))
        dds.append(dd)
    dd_pcts = {f"p{p}": float(np.percentile(dds, p)) for p in pcts}

    # CAGR
    valid_cagrs = [(c[-1]/c[0])**(1/N_YEARS)-1 for c in curves if c[-1] > 0 and c[0] > 0]
    cagr_pcts = {f"p{p}": float(np.percentile(valid_cagrs, p)) for p in pcts}
    cagr_pcts["mean"] = float(np.mean(valid_cagrs))

    # Sharpe (annualized from bi-weekly returns)
    all_returns = []
    for c in curves:
        rets = np.diff(c) / c[:-1]
        rets = rets[np.isfinite(rets)]
        if len(rets) > 10:
            all_returns.append(np.mean(rets) / max(np.std(rets), 1e-8) * np.sqrt(PERIODS_PER_YEAR))
    sharpe_med = float(np.median(all_returns)) if all_returns else 0

    return {
        "label": label, "final_equity": eq_pcts, "milestones": ms,
        "ruin_pct": ruin * 100, "max_dd": dd_pcts, "cagr": cagr_pcts,
        "sharpe_median": sharpe_med,
    }

def main():
    t0 = time.time()
    rng = np.random.default_rng(42)

    fprint("=" * 70)
    fprint("DYNAMIC POSITION SCALING v1 — Capital-Adaptive Sizing")
    fprint(f"Sims: {N_SIMS:,} | Horizon: {N_YEARS}yr | Start: ${INITIAL_CAPITAL:.0f}")
    fprint("=" * 70)

    results = {}
    summary_table = []

    for rule_key, (rule_name, rule_fn) in SIZING_RULES.items():
        fprint(f"\n{'='*50}")
        fprint(f"  {rule_key}: {rule_name}")
        fprint(f"{'='*50}")

        # Show position sizes at different equity levels
        for eq in [645, 1000, 2000, 5000, 10000, 25000]:
            ps = rule_fn(eq)
            exp = ps * TRADES_PER_PERIOD
            fprint(f"    At ${eq:,}: pos=${ps:.0f}/trade, total exposure=${exp:.0f} "
                   f"({exp/eq*100:.0f}% of equity)")

        curves = []
        for _ in range(N_SIMS):
            sr = np.random.default_rng(rng.integers(0, 2**31))
            curves.append(simulate_path(sr, rule_fn))

        stats = analyze_curves(curves, rule_name)
        results[rule_key] = stats

        fprint(f"\n  Results:")
        fprint(f"    Median 5yr: ${stats['final_equity']['p50']:,.0f} | "
               f"p5: ${stats['final_equity']['p5']:,.0f} | "
               f"p95: ${stats['final_equity']['p95']:,.0f}")
        fprint(f"    Mean 5yr: ${stats['final_equity']['mean']:,.0f}")
        fprint(f"    P(ruin): {stats['ruin_pct']:.1f}% | "
               f"CAGR p50: {stats['cagr']['p50']*100:.1f}%")
        fprint(f"    MaxDD p50: {stats['max_dd']['p50']*100:.1f}% | "
               f"Sharpe: {stats['sharpe_median']:.2f}")
        for ms_name, ms_data in stats["milestones"].items():
            if ms_data["pct_reach"] > 0:
                med = ms_data['med_months']
                fprint(f"    → {ms_name}: {ms_data['pct_reach']:.0f}% reach"
                       f"{f', median {med}mo' if med else ''}")

        summary_table.append({
            "rule": rule_key,
            "name": rule_name,
            "median_5yr": stats["final_equity"]["p50"],
            "p5": stats["final_equity"]["p5"],
            "p95": stats["final_equity"]["p95"],
            "ruin_pct": stats["ruin_pct"],
            "cagr_p50": stats["cagr"]["p50"] * 100,
            "maxdd_p50": stats["max_dd"]["p50"] * 100,
            "sharpe": stats["sharpe_median"],
            "pct_reach_5k": stats["milestones"].get("$5,000", {}).get("pct_reach", 0),
            "pct_reach_25k": stats["milestones"].get("$25,000", {}).get("pct_reach", 0),
        })

    # ─── Summary Table ───
    fprint("\n" + "=" * 70)
    fprint("COMPARISON TABLE")
    fprint("=" * 70)
    fprint(f"{'Rule':<22} {'Median':>10} {'P5':>10} {'P95':>10} {'Ruin%':>6} "
           f"{'CAGR':>7} {'MaxDD':>7} {'Sharpe':>7} {'P($5K)':>7} {'P($25K)':>8}")
    fprint("-" * 105)
    for row in summary_table:
        fprint(f"{row['name']:<22} ${row['median_5yr']:>8,.0f} ${row['p5']:>8,.0f} "
               f"${row['p95']:>8,.0f} {row['ruin_pct']:>5.1f}% "
               f"{row['cagr_p50']:>6.1f}% {row['maxdd_p50']:>6.1f}% "
               f"{row['sharpe']:>6.2f} {row['pct_reach_5k']:>6.1f}% "
               f"{row['pct_reach_25k']:>7.1f}%")

    # ─── Risk-Adjusted Winner ───
    # Best = highest Sharpe with ruin < 5%
    safe = [r for r in summary_table if r["ruin_pct"] < 5]
    if safe:
        best = max(safe, key=lambda r: r["median_5yr"])
        fprint(f"\n  WINNER (best median, ruin<5%): {best['name']}")
        fprint(f"    Median: ${best['median_5yr']:,.0f} | CAGR: {best['cagr_p50']:.1f}% | "
               f"Ruin: {best['ruin_pct']:.1f}%")

    # Best for growth (highest P($25K))
    if summary_table:
        growth = max(summary_table, key=lambda r: r["pct_reach_25k"])
        fprint(f"\n  BEST GROWTH: {growth['name']}")
        fprint(f"    P($25K): {growth['pct_reach_25k']:.1f}% | Ruin: {growth['ruin_pct']:.1f}%")

    # Best risk-adjusted (highest Sharpe, ruin < 10%)
    safe10 = [r for r in summary_table if r["ruin_pct"] < 10]
    if safe10:
        risk_adj = max(safe10, key=lambda r: r["sharpe"])
        fprint(f"\n  BEST RISK-ADJUSTED: {risk_adj['name']}")
        fprint(f"    Sharpe: {risk_adj['sharpe']:.2f} | Ruin: {risk_adj['ruin_pct']:.1f}%")

    results["summary"] = summary_table
    fprint("\n" + "=" * 70)

    elapsed = time.time() - t0
    fprint(f"\nCompleted in {elapsed:.1f}s")

    # Save
    out = OUTPUT_DIR / "position_scaling_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    fprint(f"Saved: {out}")

    # MLflow
    try:
        import mlflow
        mlflow.set_tracking_uri("http://jupiter:5000")
        mlflow.set_experiment("dynamic_position_scaling_v1")
        with mlflow.start_run(run_name=f"pos_scale_{datetime.now():%Y%m%d_%H%M}"):
            mlflow.log_params({
                "n_sims": N_SIMS, "n_years": N_YEARS,
                "initial_capital": INITIAL_CAPITAL,
                "wr": WR, "avg_win_pct": AVG_WIN_PCT,
                "avg_loss_pct": AVG_LOSS_PCT,
            })
            if safe:
                mlflow.log_metrics({
                    "winner_median": best["median_5yr"],
                    "winner_cagr": best["cagr_p50"],
                    "winner_ruin": best["ruin_pct"],
                    "winner_sharpe": best["sharpe"],
                })
            mlflow.log_artifact(str(out))
            fprint("Logged to MLflow")
    except Exception as e:
        fprint(f"MLflow skip: {e}")

    fprint("Done.")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Capital Scaling Roadmap v1 — Optimal Path $645 → $25K+
=======================================================
Uses ACTUAL validated strategy metrics with CORRECT position sizing:
- Options strategies use FIXED max position sizes ($200/spread)
- Returns are in DOLLARS (position_return × position_size), not % of capital
- Commission is FLAT per trade ($2.60 RT)
- Growth rate naturally decreases as equity grows (saturating, not exponential)

Key question: What's the realistic path from $645 to $25K?
"""
import json, os, time, warnings
from datetime import datetime
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")

BASE_PATH = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_PATH / "output" / "growth_research" / "capital_scaling_roadmap_v1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_SIMS = 10_000
N_YEARS = 5
PERIODS_PER_YEAR = 26  # bi-weekly
INITIAL_CAPITAL = 645.0
MILESTONES = [1_000, 2_000, 5_000, 10_000, 25_000]
COMMISSION_PER_TRADE = 2.60  # RT commission per spread

def fprint(*a, **kw): print(*a, **kw, flush=True)

# ─── VALIDATED STRATEGY LIBRARY ───
# Returns are calibrated from actual backtests as % of POSITION SIZE (not capital)
# Position size is FIXED (capped), creating saturating growth
STRATEGIES = {
    "sector_bull_spread": {
        "name": "Sector Bull Call Spreads (Bi-weekly)",
        "sharpe": 3.76, "cagr_backtest": 0.326, "maxdd": -0.043,
        "wr": 0.801, "pf": 22.22, "r1_gap": 0.11,
        "min_capital": 200,
        "max_position": 200,       # max risk per spread
        "trades_per_period": 3,     # 3 concurrent spreads
        # Calibrated from PF=22.22, WR=80.1%:
        # avg_win/avg_loss = PF × (1-WR)/WR = 5.52
        # From backtest CAGR: avg trade PnL ≈ $13-15 on $200 position
        # → avg_win ≈ $18 (9% of position), avg_loss ≈ $3.3 (1.6%)
        "avg_win_pct": 0.09, "win_std": 0.05,
        "avg_loss_pct": -0.016, "loss_std": 0.01,
    },
    "earnings_iron_condor": {
        "name": "Earnings Iron Condors",
        "sharpe": 1.27, "cagr_backtest": 0.12, "maxdd": -0.15,
        "wr": 0.89, "pf": 3.50, "r1_gap": 0.087,
        "min_capital": 300,
        "max_position": 250,       # IC max risk per position
        "trades_per_period": 0.5,   # ~1 per month during earnings
        "avg_win_pct": 0.06, "win_std": 0.03,
        "avg_loss_pct": -0.055, "loss_std": 0.025,
    },
    "vix_mean_reversion": {
        "name": "VIX Mean-Reversion (VIX>30)",
        "sharpe": 2.83, "cagr_backtest": 0.165, "maxdd": -0.054,
        "wr": 0.909, "pf": 2.91, "r1_gap": 0.115,
        "min_capital": 500,
        "max_position": 300,
        "trades_per_period": 0.15,  # ~4 per year
        "avg_win_pct": 0.08, "win_std": 0.04,
        "avg_loss_pct": -0.07, "loss_std": 0.03,
    },
    "put_credit_spreads": {
        "name": "Put Credit Spreads (Momentum)",
        "sharpe": 0.85, "cagr_backtest": 0.136, "maxdd": -0.18,
        "wr": 0.879, "pf": 2.10, "r1_gap": 0.011,
        "min_capital": 300,
        "max_position": 200,
        "trades_per_period": 2,
        "avg_win_pct": 0.05, "win_std": 0.03,
        "avg_loss_pct": -0.042, "loss_std": 0.02,
    },
    "leaps_momentum": {
        "name": "LEAPS Momentum (Bi-weekly Top 3)",
        "sharpe": 0.80, "cagr_backtest": 0.102, "maxdd": -0.193,
        "wr": 0.579, "pf": 2.09, "r1_gap": 0.15,
        "min_capital": 2000,
        "max_position": 500,       # LEAPS cost more
        "trades_per_period": 1,
        "avg_win_pct": 0.12, "win_std": 0.08,
        "avg_loss_pct": -0.06, "loss_std": 0.03,
    },
}

def gen_trade_pnl(rng, strat, n_trades):
    """Generate dollar PnL for n trades. Returns array of dollar amounts."""
    s = STRATEGIES[strat]
    pos = s["max_position"]
    is_win = rng.random(n_trades) < s["wr"]
    wins = np.clip(rng.normal(s["avg_win_pct"], s["win_std"], n_trades), 0.005, 0.50)
    losses = np.clip(rng.normal(s["avg_loss_pct"], s["loss_std"], n_trades), -1.0, -0.001)
    pct_ret = np.where(is_win, wins, losses)
    dollar_pnl = pct_ret * pos - COMMISSION_PER_TRADE
    return dollar_pnl

def simulate_single_strategy(rng, strat_key, capital=INITIAL_CAPITAL, n_years=N_YEARS):
    """Simulate single strategy with fixed position sizing."""
    s = STRATEGIES[strat_key]
    n_periods = PERIODS_PER_YEAR * n_years
    equity = capital
    curve = [equity]

    for pi in range(n_periods):
        if equity < 50:  # ruin
            curve.extend([equity] * (n_periods - pi))
            break

        # How many trades this period? (fractional = probability of trade)
        tpp = s["trades_per_period"]
        if tpp < 1:
            n_trades = 1 if rng.random() < tpp else 0
        else:
            n_trades = int(tpp)

        if n_trades == 0:
            curve.append(equity)
            continue

        # Check if we can afford the positions
        max_risk = n_trades * s["max_position"]
        if equity < max_risk * 0.5:  # need at least 50% of max risk
            # Scale down trades
            n_trades = max(1, int(equity / s["max_position"]))

        pnl = gen_trade_pnl(rng, strat_key, n_trades)
        equity = max(0, equity + np.sum(pnl))
        curve.append(equity)

    return np.array(curve[:n_periods + 1])

def simulate_multi_strategy(rng, capital=INITIAL_CAPITAL, n_years=N_YEARS,
                            alloc_method="equal"):
    """Simulate multiple strategies, unlocking as capital grows."""
    n_periods = PERIODS_PER_YEAR * n_years
    equity = capital
    curve = [equity]

    for pi in range(n_periods):
        if equity < 50:
            curve.extend([equity] * (n_periods - pi))
            break

        # Find viable strategies
        viable = [k for k, v in STRATEGIES.items() if v["min_capital"] <= equity]
        if not viable:
            curve.append(equity)
            continue

        # Allocate capital across strategies
        period_pnl = 0.0
        n_viable = len(viable)

        for strat_key in viable:
            s = STRATEGIES[strat_key]

            if alloc_method == "equal":
                strat_capital = equity / n_viable
            elif alloc_method == "sharpe_weighted":
                total_sharpe = sum(STRATEGIES[k]["sharpe"] for k in viable)
                strat_capital = equity * s["sharpe"] / total_sharpe
            elif alloc_method == "concentrate_best":
                # Put 70% in best Sharpe, split rest equally
                best = max(viable, key=lambda k: STRATEGIES[k]["sharpe"])
                if strat_key == best:
                    strat_capital = equity * 0.70
                else:
                    strat_capital = equity * 0.30 / (n_viable - 1) if n_viable > 1 else equity * 0.30
            else:
                strat_capital = equity / n_viable

            # How many trades (scale by allocated capital)
            tpp = s["trades_per_period"]
            if tpp < 1:
                n_trades = 1 if rng.random() < tpp else 0
            else:
                n_trades = int(tpp)

            if n_trades == 0:
                continue

            # Scale trades by capital available
            affordable_trades = max(1, int(strat_capital / s["max_position"]))
            n_trades = min(n_trades, affordable_trades)

            pnl = gen_trade_pnl(rng, strat_key, n_trades)
            period_pnl += np.sum(pnl)

        equity = max(0, equity + period_pnl)
        curve.append(equity)

    return np.array(curve[:n_periods + 1])

def analyze_curves(curves, label):
    """Compute statistics from simulation curves."""
    finals = np.array([c[-1] for c in curves])
    n = len(curves)
    pcts = [5, 10, 25, 50, 75, 90, 95]

    eq_pcts = {f"p{p}": float(np.percentile(finals, p)) for p in pcts}
    eq_pcts["mean"] = float(np.mean(finals))

    # Milestone times
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

    return {
        "label": label, "final_equity": eq_pcts, "milestones": ms,
        "ruin_pct": ruin * 100, "max_dd": dd_pcts, "cagr": cagr_pcts,
    }

def main():
    t0 = time.time()
    rng = np.random.default_rng(42)

    fprint("=" * 70)
    fprint("CAPITAL SCALING ROADMAP v1 — $645 → $25K+ Optimal Path")
    fprint(f"Strategies: {len(STRATEGIES)} validated | Sims: {N_SIMS:,} | Horizon: {N_YEARS}yr")
    fprint("=" * 70)

    results = {}

    # ─── SECTION 1: Commission Analysis by Capital Level ───
    fprint("\n" + "=" * 50)
    fprint("SECTION 1: Commission Impact by Capital Level")
    fprint("=" * 50)
    capital_levels = [645, 1000, 2000, 5000, 10000, 25000]
    comm_analysis = {}
    for cap in capital_levels:
        viable = [k for k, v in STRATEGIES.items() if v["min_capital"] <= cap]
        info = {}
        for k in viable:
            s = STRATEGIES[k]
            annual_comm = COMMISSION_PER_TRADE * s["trades_per_period"] * PERIODS_PER_YEAR
            annual_pnl_gross = s["trades_per_period"] * PERIODS_PER_YEAR * s["max_position"] * (
                s["wr"] * s["avg_win_pct"] + (1-s["wr"]) * s["avg_loss_pct"])
            net = annual_pnl_gross - annual_comm
            info[k] = {
                "name": s["name"],
                "annual_commission": f"${annual_comm:.0f}",
                "annual_gross_pnl": f"${annual_pnl_gross:.0f}",
                "annual_net_pnl": f"${net:.0f}",
                "net_as_pct_capital": f"{net/cap*100:.1f}%",
                "comm_as_pct_gross": f"{annual_comm/max(1,annual_pnl_gross)*100:.1f}%",
            }
        comm_analysis[f"${cap:,}"] = info
        fprint(f"\n  ${cap:,}:")
        for k, v in info.items():
            fprint(f"    {v['name']}: Gross {v['annual_gross_pnl']}, "
                   f"Comm {v['annual_commission']}, Net {v['annual_net_pnl']} "
                   f"({v['net_as_pct_capital']} of capital)")
    results["commission_analysis"] = comm_analysis

    # ─── SECTION 2: Single-Strategy Monte Carlo ───
    fprint("\n" + "=" * 50)
    fprint("SECTION 2: Single-Strategy Growth from $645")
    fprint("=" * 50)
    single_results = {}
    for strat_key in STRATEGIES:
        s = STRATEGIES[strat_key]
        if s["min_capital"] > INITIAL_CAPITAL:
            fprint(f"\n  {s['name']}: SKIP (needs ${s['min_capital']})")
            continue

        curves = []
        for _ in range(N_SIMS):
            sr = np.random.default_rng(rng.integers(0, 2**31))
            curves.append(simulate_single_strategy(sr, strat_key))

        stats = analyze_curves(curves, s["name"])
        single_results[strat_key] = stats

        fprint(f"\n  {s['name']}:")
        fprint(f"    Median 5yr: ${stats['final_equity']['p50']:,.0f} | "
               f"p5: ${stats['final_equity']['p5']:,.0f} | "
               f"p95: ${stats['final_equity']['p95']:,.0f}")
        fprint(f"    P(ruin): {stats['ruin_pct']:.1f}% | "
               f"CAGR p50: {stats['cagr']['p50']*100:.1f}%")
        for ms_name, ms_data in stats["milestones"].items():
            if ms_data["pct_reach"] > 0:
                fprint(f"    → {ms_name}: {ms_data['pct_reach']:.0f}% reach, "
                       f"median {ms_data['med_months']}mo")
    results["single_strategy"] = single_results

    # ─── SECTION 3: Multi-Strategy Staged Growth ───
    fprint("\n" + "=" * 50)
    fprint("SECTION 3: Multi-Strategy Staged Growth")
    fprint("=" * 50)

    for method in ["equal", "sharpe_weighted", "concentrate_best"]:
        curves = []
        for _ in range(N_SIMS):
            sr = np.random.default_rng(rng.integers(0, 2**31))
            curves.append(simulate_multi_strategy(sr, alloc_method=method))

        stats = analyze_curves(curves, f"Multi-Strategy ({method})")
        results[f"multi_{method}"] = stats

        fprint(f"\n  {method.upper()} allocation:")
        fprint(f"    Median 5yr: ${stats['final_equity']['p50']:,.0f} | "
               f"p5: ${stats['final_equity']['p5']:,.0f} | "
               f"p95: ${stats['final_equity']['p95']:,.0f}")
        fprint(f"    P(ruin): {stats['ruin_pct']:.1f}% | "
               f"CAGR p50: {stats['cagr']['p50']*100:.1f}%")
        fprint(f"    MaxDD p50: {stats['max_dd']['p50']*100:.1f}%")
        for ms_name, ms_data in stats["milestones"].items():
            if ms_data["pct_reach"] > 0:
                fprint(f"    → {ms_name}: {ms_data['pct_reach']:.0f}% reach, "
                       f"median {ms_data['med_months']}mo")

    # ─── SECTION 4: Growth Rate by Capital Level ───
    fprint("\n" + "=" * 50)
    fprint("SECTION 4: Growth Rate Saturation (fixed position sizing)")
    fprint("=" * 50)

    strat = "sector_bull_spread"
    s = STRATEGIES[strat]
    expected_pnl_per_period = s["trades_per_period"] * s["max_position"] * (
        s["wr"] * s["avg_win_pct"] + (1-s["wr"]) * s["avg_loss_pct"])
    comm_per_period = s["trades_per_period"] * COMMISSION_PER_TRADE
    net_pnl = expected_pnl_per_period - comm_per_period

    fprint(f"\n  {s['name']}:")
    fprint(f"  Expected PnL per period: ${expected_pnl_per_period:.2f} gross, "
           f"${net_pnl:.2f} net")
    fprint(f"  Growth rate (net/capital) at different capital levels:")
    for cap in capital_levels:
        rate = net_pnl / cap * 100
        annualized = ((1 + net_pnl/cap) ** PERIODS_PER_YEAR - 1) * 100
        fprint(f"    ${cap:,}: {rate:.1f}%/period → {annualized:.0f}% annualized")

    results["growth_saturation"] = {
        "strategy": s["name"],
        "net_pnl_per_period": net_pnl,
        "rates": {f"${cap:,}": {
            "per_period_pct": net_pnl/cap*100,
            "annualized_pct": ((1+net_pnl/cap)**PERIODS_PER_YEAR - 1)*100
        } for cap in capital_levels}
    }

    # ─── SECTION 5: Optimal Starting Strategy ───
    fprint("\n" + "=" * 50)
    fprint("SECTION 5: Concentrate vs Diversify at $645")
    fprint("=" * 50)

    # Compare: 100% sector bull vs equal-weight multi
    # Already have these from sections 2 and 3
    if "sector_bull_spread" in single_results and "multi_equal" in results:
        conc = single_results["sector_bull_spread"]
        div = results["multi_equal"]
        fprint(f"\n  100% Sector Bull Spreads:")
        fprint(f"    Median 5yr: ${conc['final_equity']['p50']:,.0f}, "
               f"CAGR: {conc['cagr']['p50']*100:.1f}%, "
               f"Ruin: {conc['ruin_pct']:.1f}%")
        fprint(f"  Equal-Weight Multi-Strategy:")
        fprint(f"    Median 5yr: ${div['final_equity']['p50']:,.0f}, "
               f"CAGR: {div['cagr']['p50']*100:.1f}%, "
               f"Ruin: {div['ruin_pct']:.1f}%")

        conc_better = conc['final_equity']['p50'] > div['final_equity']['p50']
        fprint(f"\n  VERDICT: {'CONCENTRATE' if conc_better else 'DIVERSIFY'} at $645")

    # ─── SECTION 6: Roadmap Summary ───
    fprint("\n" + "=" * 70)
    fprint("OPTIMAL ROADMAP: $645 → $25K")
    fprint("=" * 70)

    roadmap = {}
    # Determine best single strategy at $645
    best_645 = None
    best_med = 0
    for k, v in single_results.items():
        if v["final_equity"]["p50"] > best_med:
            best_med = v["final_equity"]["p50"]
            best_645 = k

    if best_645:
        s = STRATEGIES[best_645]
        roadmap["best_start"] = s["name"]
        fprint(f"\n  BEST STRATEGY AT $645: {s['name']}")
        fprint(f"    → Median 5yr: ${single_results[best_645]['final_equity']['p50']:,.0f}")

    # Best multi-strategy approach
    best_multi = None
    best_multi_med = 0
    for method in ["equal", "sharpe_weighted", "concentrate_best"]:
        key = f"multi_{method}"
        if key in results and results[key]["final_equity"]["p50"] > best_multi_med:
            best_multi_med = results[key]["final_equity"]["p50"]
            best_multi = method

    if best_multi:
        roadmap["best_multi"] = best_multi
        key = f"multi_{best_multi}"
        fprint(f"\n  BEST MULTI-STRATEGY: {best_multi}")
        fprint(f"    → Median 5yr: ${results[key]['final_equity']['p50']:,.0f}")

    roadmap["key_insight"] = (
        "Fixed position sizing ($200/spread) means growth rate DECREASES as capital grows. "
        "At $645, net PnL/period is a large % of capital → fast growth. "
        "At $25K, same dollar PnL → slow growth. "
        "To sustain growth, INCREASE position sizes as capital grows."
    )
    results["roadmap"] = roadmap

    fprint(f"\n  KEY INSIGHT: Growth naturally slows because positions are capped.")
    fprint(f"  At $645: each $40 net profit = 6.2% of capital")
    fprint(f"  At $5K: same $40 = 0.8% of capital")
    fprint(f"  → INCREASE position sizes as capital grows to maintain growth rate")

    fprint("\n" + "=" * 70)

    elapsed = time.time() - t0
    fprint(f"\nCompleted in {elapsed:.1f}s")

    # Save
    out = OUTPUT_DIR / "scaling_roadmap_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    fprint(f"Saved: {out}")

    # MLflow
    try:
        import mlflow
        mlflow.set_tracking_uri("http://jupiter:5000")
        mlflow.set_experiment("capital_scaling_roadmap_v1")
        with mlflow.start_run(run_name=f"roadmap_{datetime.now():%Y%m%d_%H%M}"):
            mlflow.log_params({
                "n_sims": N_SIMS, "n_years": N_YEARS,
                "initial_capital": INITIAL_CAPITAL,
                "n_strategies": len(STRATEGIES),
            })
            if "multi_concentrate_best" in results:
                best = results["multi_concentrate_best"]
                mlflow.log_metrics({
                    "median_final": best["final_equity"]["p50"],
                    "p5_final": best["final_equity"]["p5"],
                    "p95_final": best["final_equity"]["p95"],
                    "ruin_pct": best["ruin_pct"],
                    "cagr_p50": best["cagr"]["p50"],
                    "maxdd_p50": best["max_dd"]["p50"],
                })
            mlflow.log_artifact(str(out))
            fprint("Logged to MLflow")
    except Exception as e:
        fprint(f"MLflow skip: {e}")

    fprint("Done.")

if __name__ == "__main__":
    main()

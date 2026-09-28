#!/usr/bin/env python3
"""
Monte Carlo Stress Test v1 — Sector Options Rotation Bull Call Spreads
======================================================================
Stress-tests the $645 strategy (Sharpe 3.76, CAGR 32.6%, WR 80.1%, 1337 trades)
using 10K Monte Carlo paths across 7 scenarios including fat tails, correlation
stress, regime shift, and combined adversarial.

Key question: P($645 -> $5K in 2yr)? P(below $200)?
"""
import json, os, time, warnings
from datetime import datetime
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")

BASE_PATH = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_PATH / "output" / "growth_research" / "monte_carlo_stress_v1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_SIMS, N_YEARS = 10_000, 5
TRADES_PER_PERIOD, PERIODS_PER_YEAR = 3, 26  # 3 concurrent, bi-weekly
INITIAL_CAPITAL, MAX_POSITION = 645.0, 200.0
COMMISSION_PER_PERIOD = 2.60 * 3  # $2.60/spread RT * 3 trades = $7.80
TARGET_WR, RUIN_THR = 0.801, 100.0
MILESTONES = [1_000, 1_290, 2_000, 5_000, 10_000]
MLFLOW_URI, EXP_NAME = "http://jupiter:5000", "monte_carlo_stress_test_v1"
PCTS = [5, 25, 50, 75, 95]

def fprint(*a, **kw): print(*a, **kw, flush=True)

def gen_returns(n, rng, wr=TARGET_WR):
    """Trade returns as fraction of position. Calibrated to PF=22, WR=80%.
    From backtest: avg_win/avg_loss = PF * (1-WR)/WR = 22.22 * 0.199/0.801 = 5.52
    With avg_loss ~3% of position, avg_win ~16.6% of position."""
    is_win = rng.random(n) < wr
    wins = np.clip(rng.normal(0.166, 0.08, n), 0.01, 0.50)   # ~16.6% avg, right-skewed
    losses = np.clip(rng.normal(-0.03, 0.02, n), -0.50, -0.005)  # ~3% avg loss (small)
    return np.where(is_win, wins, losses)

def simulate_path(rng, win_rate=TARGET_WR, corr_stress=False,
                  regime_shift=False, fat_tail_pct=0.0):
    """Simulate one 5-year equity path. Returns equity curve array."""
    n_periods = PERIODS_PER_YEAR * N_YEARS
    equity = INITIAL_CAPITAL
    curve = [equity]
    for pi in range(n_periods):
        if equity < RUIN_THR:
            curve.extend([equity] * (n_periods - pi)); break
        equity -= COMMISSION_PER_PERIOD
        if equity <= 0:
            curve.extend([0] * (n_periods - pi)); break
        wr = win_rate
        if regime_shift and pi >= PERIODS_PER_YEAR * 2:
            wr = max(0.1, win_rate - 0.10)
        rets = gen_returns(TRADES_PER_PERIOD, rng, wr)
        if fat_tail_pct > 0:
            for i in range(len(rets)):
                if rets[i] < 0 and rng.random() < fat_tail_pct:
                    rets[i] = -1.0
        if corr_stress and rng.random() < 0.30:
            rets[:] = min(rets)
        pos = min(MAX_POSITION, equity / TRADES_PER_PERIOD)
        equity = max(0, equity + np.sum(rets * pos))
        curve.append(equity)
    return np.array(curve[:n_periods + 1])

def run_scenario(rng, name, desc, **kwargs):
    """Run N_SIMS paths for one scenario, return metrics dict."""
    fprint(f"\n--- {desc} ---")
    curves = []
    for _ in range(N_SIMS):
        sr = np.random.default_rng(rng.integers(0, 2**31))
        curves.append(simulate_path(sr, **kwargs))
    n = len(curves)
    finals = np.array([c[-1] for c in curves])
    # Percentiles
    eq_pcts = {f"p{p}": float(np.percentile(finals, p)) for p in PCTS}
    eq_pcts["mean"] = float(np.mean(finals))
    # Ruin & below-$200 at checkpoints
    cps = {"6mo": 13, "1yr": 26, "2yr": 52, "5yr": 130}
    ruin, b200 = {}, {}
    for lb, idx in cps.items():
        ruin[lb] = sum(1 for c in curves if len(c) > idx and c[idx] < RUIN_THR) / n
        b200[lb] = sum(1 for c in curves if len(c) > idx and c[idx] < 200) / n
    # Milestone times
    ms = {}
    for tgt in MILESTONES:
        hits = [np.where(c >= tgt)[0][0] for c in curves if np.any(c >= tgt)]
        if hits:
            ms[f"${tgt:,}"] = {"med_months": round(float(np.median(hits)) / 26 * 12, 1),
                               "pct_reach": round(len(hits) / n * 100, 1)}
        else:
            ms[f"${tgt:,}"] = {"med_months": None, "pct_reach": 0.0}
    # Worst 1%
    sf = np.sort(finals)
    w1 = int(n * 0.01)
    worst = {"worst": float(sf[0]), "p1": float(sf[w1]), "mean_w1": float(np.mean(sf[:w1]))}
    # Max drawdown
    dds = []
    for c in curves:
        rm = np.maximum.accumulate(c)
        dds.append(np.min((c - rm) / np.where(rm > 0, rm, 1)))
    dd_pcts = {f"p{p}": float(np.percentile(dds, p)) for p in PCTS}
    # Annual return
    ar = [(c[-1]/c[0])**(1/N_YEARS)-1 for c in curves if c[-1] > 0 and c[0] > 0]
    ar = np.array(ar) if ar else np.array([0])
    ar_pcts = {f"p{p}": float(np.percentile(ar, p)) for p in PCTS}
    ar_pcts["mean"] = float(np.mean(ar))
    # 2yr specific: P(hit $5K), P(ever below $200)
    hit5k = sum(1 for c in curves if np.any(c[:53] >= 5000)) / n
    ever200 = sum(1 for c in curves if np.any(c[:53] < 200)) / n

    fprint(f"  Equity p5=${eq_pcts['p5']:.0f} p50=${eq_pcts['p50']:.0f} p95=${eq_pcts['p95']:.0f}")
    fprint(f"  Ruin: 1yr={ruin['1yr']:.1%} 2yr={ruin['2yr']:.1%} 5yr={ruin['5yr']:.1%}")
    fprint(f"  P($5K in 2yr)={hit5k:.1%}  P(<$200 in 2yr)={ever200:.1%}")
    fprint(f"  MaxDD: p50={dd_pcts['p50']:.1%} p95={dd_pcts['p95']:.1%}")

    return {"desc": desc, "final_eq": eq_pcts, "ruin": ruin, "below_200": b200,
            "milestones": ms, "worst_1pct": worst, "max_dd": dd_pcts,
            "ann_return": ar_pcts, "p5k_2yr": hit5k, "p_below200_2yr": ever200}

def main():
    t0 = time.time()
    fprint("=" * 70)
    fprint("MONTE CARLO STRESS TEST v1 — Sector Options Rotation")
    fprint(f"Capital: ${INITIAL_CAPITAL:.0f} | Sims: {N_SIMS:,} | Horizon: {N_YEARS}yr")
    fprint("=" * 70)
    rng = np.random.default_rng(42)
    results = {}
    scenarios = [
        ("base",          "Base case (WR=80%, independent trades)", {}),
        ("lower_wr_70",   "Degraded WR (70%)", {"win_rate": 0.70}),
        ("lower_wr_65",   "Severely degraded WR (65%)", {"win_rate": 0.65}),
        ("fat_tails",     "Fat tails (25% of losses -> -100%)", {"fat_tail_pct": 0.25}),
        ("corr_stress",   "Correlation stress (30% all-lose-together)", {"corr_stress": True}),
        ("regime_shift",  "Regime shift (WR -10% after yr 2)", {"regime_shift": True}),
        ("combined",      "Combined: WR=75% + fat tails + correlation",
         {"win_rate": 0.75, "fat_tail_pct": 0.25, "corr_stress": True}),
    ]
    for name, desc, kw in scenarios:
        results[name] = run_scenario(rng, name, desc, **kw)

    # Commission burden
    ann_comm = COMMISSION_PER_PERIOD * PERIODS_PER_YEAR
    comm = {"annual_total": ann_comm,
            "burden": {f"${e:,}": f"{ann_comm/e*100:.1f}%" for e in [645,1000,2000,5000,10000,25000]}}
    results["commissions"] = comm
    fprint(f"\n--- Commissions: ${ann_comm:.0f}/yr ---")
    for k, v in comm["burden"].items(): fprint(f"  {k}: {v}")

    # Key answers
    b = results["base"]
    fprint("\n" + "=" * 70)
    fprint("KEY ANSWERS")
    fprint(f"  P($645 -> $5K in 2yr):     {b['p5k_2yr']:.1%}")
    fprint(f"  P(below $200 in 2yr):      {b['p_below200_2yr']:.1%}")
    fprint(f"  Median final equity (5yr): ${b['final_eq']['p50']:.0f}")
    fprint(f"  Worst 1% mean final:       ${b['worst_1pct']['mean_w1']:.0f}")
    fprint(f"  Commission drag yr1:       {ann_comm/INITIAL_CAPITAL*100:.1f}% of starting capital")
    fprint("=" * 70)

    elapsed = time.time() - t0
    fprint(f"\nCompleted in {elapsed:.1f}s")

    # Save
    out = OUTPUT_DIR / "stress_test_results.json"
    with open(out, "w") as f: json.dump(results, f, indent=2, default=str)
    fprint(f"Saved: {out}")

    # MLflow
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment(EXP_NAME)
        with mlflow.start_run(run_name=f"stress_{datetime.now():%Y%m%d_%H%M}"):
            mlflow.log_params({"n_sims": N_SIMS, "n_years": N_YEARS,
                               "initial_capital": INITIAL_CAPITAL, "target_wr": TARGET_WR,
                               "trades_per_year": TRADES_PER_PERIOD * PERIODS_PER_YEAR,
                               "commission_per_period": COMMISSION_PER_PERIOD})
            mlflow.log_metrics({"median_final": b["final_eq"]["p50"],
                                "p5_final": b["final_eq"]["p5"],
                                "p95_final": b["final_eq"]["p95"],
                                "ruin_5yr": b["ruin"]["5yr"],
                                "p5k_2yr": b["p5k_2yr"],
                                "p_below200_2yr": b["p_below200_2yr"],
                                "median_ann_ret": b["ann_return"]["p50"],
                                "median_max_dd": b["max_dd"]["p50"]})
            mlflow.log_artifact(str(out))
            fprint("Logged to MLflow")
    except Exception as e:
        fprint(f"MLflow skip: {e}")
    fprint("Done.")

if __name__ == "__main__":
    main()

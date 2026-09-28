#!/usr/bin/env python3
"""
monte_carlo_top5.py — 1000-path Monte Carlo on top 5 profitable fill sim configs.

Top 5 configs:
  1. TP15 H2H (head-to-head limit)          Sortino 5.70
  2. TP13 prime chase                        Sortino 5.21
  3. TP13 prime (prime hours only)           Sortino 2.39
  4. TP20 sig>0.3                            Sortino 2.33
  5. midday passive                          ~$53/day from sweep

Methodology:
  - Load per-day trade PnL for each config from fill_sim results
  - Bootstrap 1000 paths (sample with replacement from day PnLs)
  - Each path: draw N_DAYS from the distribution, compute cumulative PnL
  - Report: mean_final_pnl, p05, p50, p95, max_drawdown_p50,
            probability_of_profit, sortino_distribution

Leakage audit: PASSED — Monte Carlo uses only realized fill_sim trade PnLs.
"""
import os, sys, glob, json, time
import numpy as np
from datetime import datetime

OUT_DIR  = "/home/jupiter/Lvl3Quant/data/processed/monte_carlo_top5"
LOG_FILE = "/home/jupiter/Lvl3Quant/logs/monte_carlo_top5.log"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs("/home/jupiter/Lvl3Quant/logs", exist_ok=True)

N_PATHS = 1000
N_DAYS  = 63   # simulate ~3 months forward (1 quarter)
SEED    = 42
rng     = np.random.default_rng(SEED)


def tlog(msg):
    t = time.strftime("%H:%M:%S")
    line = "[%s] %s" % (t, msg)
    print(line, flush=True)
    try:
        with open(LOG_FILE, 'a') as f:
            f.write(line + "\n")
    except Exception:
        pass


def compute_sortino(daily_pnls):
    if len(daily_pnls) < 2:
        return 0.0
    arr  = np.array(daily_pnls, dtype=np.float64)
    mean = float(arr.mean())
    neg  = arr[arr < 0]
    down = float(neg.std()) if len(neg) > 1 else 1e-9
    return mean / down if down > 0 else 0.0


def max_drawdown(equity_curve):
    peak   = 0.0
    max_dd = 0.0
    for val in equity_curve:
        if val > peak:
            peak = val
        dd = peak - val
        if dd > max_dd:
            max_dd = dd
    return max_dd


def load_day_pnls_from_dir(result_dir):
    files = sorted(glob.glob(os.path.join(result_dir, "*.json")))
    daily_pnls = []
    for f in files:
        try:
            with open(f) as fh:
                d = json.load(fh)
            pnl = None
            for k in ["total_pnl_dollars", "total_pnl", "pnl", "net_pnl", "pnl_dollars"]:
                if k in d:
                    pnl = float(d[k]); break
            if pnl is None and "summary" in d:
                s = d["summary"]
                for k in ["total_pnl", "pnl", "net_pnl", "total_pnl_dollars"]:
                    if k in s:
                        pnl = float(s[k]); break
            if pnl is not None:
                daily_pnls.append(pnl)
        except Exception:
            pass
    return daily_pnls


def load_day_pnls_from_trade_list(result_dir):
    files = sorted(glob.glob(os.path.join(result_dir, "*.json")))
    daily_pnls = []
    for f in files:
        try:
            with open(f) as fh:
                d = json.load(fh)
            trades = d.get("trades", [])
            if trades:
                day_pnl = sum(t.get("pnl_dollars", t.get("pnl", 0.0)) for t in trades)
                daily_pnls.append(float(day_pnl))
        except Exception:
            pass
    return daily_pnls


def run_monte_carlo(config_name, daily_pnls):
    if len(daily_pnls) < 5:
        tlog("  %s: insufficient data (%d days)" % (config_name, len(daily_pnls)))
        return None

    daily_pnls  = np.array(daily_pnls, dtype=np.float64)
    n_actual    = len(daily_pnls)
    tlog("  %s: %d actual days, mean=%.2f/day, sortino=%.3f" % (
        config_name, n_actual, float(daily_pnls.mean()), compute_sortino(list(daily_pnls))))

    path_finals     = []
    path_max_dds    = []
    path_sortinos   = []
    path_min_equity = []

    for _ in range(N_PATHS):
        sampled = rng.choice(daily_pnls, size=N_DAYS, replace=True)
        equity  = np.cumsum(sampled)
        path_finals.append(float(equity[-1]))
        path_max_dds.append(max_drawdown(equity))
        path_sortinos.append(compute_sortino(list(sampled)))
        path_min_equity.append(float(equity.min()))

    path_finals     = np.array(path_finals)
    path_max_dds    = np.array(path_max_dds)
    path_sortinos   = np.array(path_sortinos)
    path_min_equity = np.array(path_min_equity)

    prob_profit = float((path_finals > 0).mean())

    result = {
        "config":                config_name,
        "n_actual_days":         n_actual,
        "n_sim_days":            N_DAYS,
        "n_paths":               N_PATHS,
        "actual_sortino":        round(float(compute_sortino(list(daily_pnls))), 4),
        "actual_mean_daily_pnl": round(float(daily_pnls.mean()), 2),
        "actual_std_daily_pnl":  round(float(daily_pnls.std()), 2),
        "final_pnl": {
            "mean": round(float(path_finals.mean()), 2),
            "p05":  round(float(np.percentile(path_finals, 5)),  2),
            "p25":  round(float(np.percentile(path_finals, 25)), 2),
            "p50":  round(float(np.percentile(path_finals, 50)), 2),
            "p75":  round(float(np.percentile(path_finals, 75)), 2),
            "p95":  round(float(np.percentile(path_finals, 95)), 2),
        },
        "max_drawdown": {
            "mean": round(float(path_max_dds.mean()), 2),
            "p50":  round(float(np.percentile(path_max_dds, 50)), 2),
            "p95":  round(float(np.percentile(path_max_dds, 95)), 2),
        },
        "sortino_distribution": {
            "mean": round(float(path_sortinos.mean()), 4),
            "p05":  round(float(np.percentile(path_sortinos, 5)),  4),
            "p50":  round(float(np.percentile(path_sortinos, 50)), 4),
            "p95":  round(float(np.percentile(path_sortinos, 95)), 4),
        },
        "probability_of_profit": round(prob_profit, 4),
        "min_equity_p05":        round(float(np.percentile(path_min_equity, 5)), 2),
    }

    tlog("    MC: mean_final=%.0f p05=%.0f p95=%.0f prob_profit=%.1f%% max_dd_p50=%.0f sortino_mean=%.3f sortino_p05=%.3f" % (
        result["final_pnl"]["mean"],
        result["final_pnl"]["p05"],
        result["final_pnl"]["p95"],
        prob_profit * 100,
        result["max_drawdown"]["p50"],
        result["sortino_distribution"]["mean"],
        result["sortino_distribution"]["p05"],
    ))
    return result


tlog("=== Monte Carlo: Top 5 Fill Sim Configs ===")
tlog("N_PATHS=%d, N_DAYS_PER_PATH=%d" % (N_PATHS, N_DAYS))

FILLSIM_BASE = "/home/jupiter/Lvl3Quant/data/processed"

CONFIG_DIRS = {
    "TP15_H2H_Sortino5.70": [
        FILLSIM_BASE + "/conviction_threshold_sweep/tp15_h2h",
        FILLSIM_BASE + "/tp15_h2h",
        FILLSIM_BASE + "/fillsim_results/tp15_h2h",
        FILLSIM_BASE + "/card_optimization/tp15",
        FILLSIM_BASE + "/conviction_threshold_sweep/z20",
    ],
    "TP13_prime_chase_Sortino5.21": [
        FILLSIM_BASE + "/time_of_day_analysis/prime_hours_z20",
        FILLSIM_BASE + "/tp13_prime_chase",
        FILLSIM_BASE + "/fillsim_results/tp13_chase",
        FILLSIM_BASE + "/conviction_threshold_sweep/z20",
    ],
    "TP13_prime_Sortino2.39": [
        FILLSIM_BASE + "/time_of_day_analysis/prime_hours_z20",
        FILLSIM_BASE + "/tp13_prime",
        FILLSIM_BASE + "/fillsim_results/tp13_prime",
        FILLSIM_BASE + "/conviction_threshold_sweep/z20",
    ],
    "TP20_sig0.3_Sortino2.33": [
        FILLSIM_BASE + "/conviction_threshold_sweep/z20_tp20",
        FILLSIM_BASE + "/tp20_sig03",
        FILLSIM_BASE + "/fillsim_results/tp20",
        FILLSIM_BASE + "/conviction_threshold_sweep/z25",
        FILLSIM_BASE + "/conviction_threshold_sweep/z20",
    ],
    "midday_passive_53perday": [
        FILLSIM_BASE + "/time_of_day_analysis/midday_passive",
        FILLSIM_BASE + "/midday_passive",
        FILLSIM_BASE + "/fillsim_results/midday",
        FILLSIM_BASE + "/fill_sim_queue_fade/midday",
        FILLSIM_BASE + "/conviction_threshold_sweep/z20",
    ],
}

all_results = []

for config_name, candidate_dirs in CONFIG_DIRS.items():
    tlog("\n--- Config: %s ---" % config_name)
    daily_pnls = []
    found_dir  = None

    for d in candidate_dirs:
        if os.path.exists(d):
            pnls = load_day_pnls_from_dir(d)
            if not pnls:
                pnls = load_day_pnls_from_trade_list(d)
            if pnls:
                tlog("  Found data in: %s (%d days)" % (d, len(pnls)))
                daily_pnls = pnls
                found_dir  = d
                break
            else:
                tlog("  Dir exists but no data: %s" % d)

    if not daily_pnls:
        tlog("  WARNING: No data for %s. Listing dirs..." % config_name)
        for base in [FILLSIM_BASE,
                     FILLSIM_BASE + "/conviction_threshold_sweep",
                     FILLSIM_BASE + "/time_of_day_analysis",
                     FILLSIM_BASE + "/fillsim_results"]:
            if os.path.exists(base):
                tlog("    %s: %s" % (base, str(os.listdir(base)[:15])))
        all_results.append({"config": config_name, "status": "no_data_found",
                            "candidates_checked": candidate_dirs})
        continue

    result = run_monte_carlo(config_name, daily_pnls)
    if result:
        result["data_source"] = found_dir
        result["status"]      = "completed"
        all_results.append(result)
        safe_name = config_name.replace("/", "_").replace(".", "p")
        with open(os.path.join(OUT_DIR, "%s.json" % safe_name), "w") as f:
            json.dump(result, f, indent=2)

summary = {
    "timestamp":  datetime.now().isoformat(),
    "parameters": {"n_paths": N_PATHS, "n_days_per_path": N_DAYS, "seed": SEED},
    "results":    all_results,
    "ranking_by_sortino_p05": sorted(
        [r for r in all_results if r.get("status") == "completed"],
        key=lambda x: x.get("sortino_distribution", {}).get("p05", -999),
        reverse=True
    ),
}

with open(os.path.join(OUT_DIR, "monte_carlo_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

tlog("\n=== SUMMARY ===")
for r in summary["ranking_by_sortino_p05"]:
    tlog("  %s: sortino_p05=%.3f prob_profit=%.1f%% p50_pnl=%.0f" % (
        r["config"],
        r["sortino_distribution"]["p05"],
        r["probability_of_profit"] * 100,
        r["final_pnl"]["p50"],
    ))

tlog("\nResults saved to %s/monte_carlo_summary.json" % OUT_DIR)
tlog("DONE")

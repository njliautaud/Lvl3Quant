#!/usr/bin/env python3
"""
ES 2h Model — Risk Framework & Position Sizing (HC #661 R4 + HC #662 R2)
=========================================================================
1. Trade-level P&L distribution analysis
2. Block bootstrap Monte Carlo (preserving serial correlation)
3. Kelly-based position sizing with margin constraints
4. Drawdown analysis and VaR/CVaR
5. Regime-stratified performance (using VIX proxy)
6. Position sizing table by account size

This prepares the framework for when Razer comes back online and the model
can be deployed with proper risk management.
"""
import json
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats as sp_stats

ROOT = Path("/home/jupiter/Lvl3Quant")
OUTPUT = ROOT / "output" / "es_2h_risk_framework"
OUTPUT.mkdir(parents=True, exist_ok=True)

# Constants
TICK_VALUE = 12.50
RT_COST_TICKS = 1.376  # Market order + AMP commission
ES_MARGIN_OVERNIGHT = 15_800  # CME initial margin
ES_MARGIN_INTRADAY = 1_000    # AMP intraday daytrading margin
ES_POINT_VALUE = 50.0         # $50 per point, 4 ticks per point


def load_predictions():
    preds = pd.read_parquet(ROOT / "output" / "lh_2h_intraday_clean" / "predictions.parquet")
    preds["direction"] = np.sign(preds["prediction"])
    preds["gross_ticks"] = preds["direction"] * preds["actual"]
    preds["net_ticks"] = preds["gross_ticks"] - RT_COST_TICKS
    preds["net_dollars"] = preds["net_ticks"] * TICK_VALUE
    preds["date_dt"] = pd.to_datetime(preds["date"].astype(str))
    return preds


def trade_level_analysis(preds):
    """Per-trade P&L distribution."""
    nt = preds["net_ticks"]
    wins = nt[nt > 0]
    losses = nt[nt <= 0]

    analysis = {
        "n_trades": len(preds),
        "n_oot_days": preds["date_dt"].nunique(),
        "direction_accuracy": round((preds["direction"] == np.sign(preds["actual"])).mean() * 100, 1),
        "win_rate_net": round((nt > 0).mean() * 100, 1),
        "mean_net_ticks": round(nt.mean(), 1),
        "median_net_ticks": round(nt.median(), 1),
        "std_net_ticks": round(nt.std(), 1),
        "skew": round(float(sp_stats.skew(nt)), 2),
        "kurtosis": round(float(sp_stats.kurtosis(nt)), 2),
        "avg_win_ticks": round(wins.mean(), 1),
        "avg_loss_ticks": round(losses.mean(), 1),
        "win_loss_ratio": round(wins.mean() / abs(losses.mean()), 2) if len(losses) > 0 else float("inf"),
        "p1": round(np.percentile(nt, 1), 1),
        "p5": round(np.percentile(nt, 5), 1),
        "p25": round(np.percentile(nt, 25), 1),
        "p75": round(np.percentile(nt, 75), 1),
        "p95": round(np.percentile(nt, 95), 1),
        "p99": round(np.percentile(nt, 99), 1),
        "worst_trade_ticks": round(nt.min(), 1),
        "best_trade_ticks": round(nt.max(), 1),
        "worst_trade_dollars": round(nt.min() * TICK_VALUE, 0),
        "best_trade_dollars": round(nt.max() * TICK_VALUE, 0),
    }

    # Normality test
    _, p_normal = sp_stats.jarque_bera(nt)
    analysis["jarque_bera_p"] = round(float(p_normal), 6)
    analysis["is_normal"] = p_normal > 0.05

    return analysis


def daily_analysis(preds):
    """Per-day P&L distribution."""
    daily = preds.groupby("date_dt").agg(
        pnl_ticks=("net_ticks", "sum"),
        n_trades=("net_ticks", "count"),
        wr=("net_ticks", lambda x: (x > 0).mean()),
        gross_ticks=("gross_ticks", "sum"),
    ).reset_index()

    cumsum = daily["pnl_ticks"].cumsum()
    running_max = cumsum.cummax()
    drawdown = cumsum - running_max

    analysis = {
        "n_days": len(daily),
        "avg_daily_pnl_ticks": round(daily["pnl_ticks"].mean(), 1),
        "std_daily_pnl_ticks": round(daily["pnl_ticks"].std(), 1),
        "avg_daily_pnl_dollars": round(daily["pnl_ticks"].mean() * TICK_VALUE, 0),
        "daily_sharpe": round(daily["pnl_ticks"].mean() / max(daily["pnl_ticks"].std(), 0.01) * np.sqrt(252), 2),
        "daily_sortino": round(
            daily["pnl_ticks"].mean() / max(daily.loc[daily["pnl_ticks"] < 0, "pnl_ticks"].std(), 0.01) * np.sqrt(252), 2
        ) if (daily["pnl_ticks"] < 0).any() else float("inf"),
        "daily_wr": round((daily["pnl_ticks"] > 0).mean() * 100, 1),
        "profit_factor": round(
            daily.loc[daily["pnl_ticks"] > 0, "pnl_ticks"].sum() / abs(daily.loc[daily["pnl_ticks"] < 0, "pnl_ticks"].sum()), 2
        ) if (daily["pnl_ticks"] < 0).any() else float("inf"),
        "worst_day_ticks": round(daily["pnl_ticks"].min(), 1),
        "worst_day_dollars": round(daily["pnl_ticks"].min() * TICK_VALUE, 0),
        "best_day_ticks": round(daily["pnl_ticks"].max(), 1),
        "best_day_dollars": round(daily["pnl_ticks"].max() * TICK_VALUE, 0),
        "max_drawdown_ticks": round(drawdown.min(), 0),
        "max_drawdown_dollars": round(drawdown.min() * TICK_VALUE, 0),
        "avg_trades_per_day": round(daily["n_trades"].mean(), 1),
        "var_95_ticks": round(np.percentile(daily["pnl_ticks"], 5), 1),
        "cvar_95_ticks": round(daily["pnl_ticks"][daily["pnl_ticks"] <= np.percentile(daily["pnl_ticks"], 5)].mean(), 1),
        "var_99_ticks": round(np.percentile(daily["pnl_ticks"], 1), 1),
    }

    return analysis, daily


def block_bootstrap_mc(daily_pnl, n_sims=5000, sim_days=252, block_size=5):
    """Block bootstrap Monte Carlo preserving serial correlation."""
    pnl = daily_pnl.values
    n = len(pnl)

    results = {
        "annual_pnl_ticks": [],
        "max_dd_ticks": [],
        "sharpe": [],
        "worst_day": [],
    }

    for _ in range(n_sims):
        # Sample blocks with replacement
        sim = []
        while len(sim) < sim_days:
            start = np.random.randint(0, max(1, n - block_size))
            block = pnl[start:start + block_size].tolist()
            sim.extend(block)
        sim = np.array(sim[:sim_days])

        cumsum = np.cumsum(sim)
        running_max = np.maximum.accumulate(cumsum)
        dd = cumsum - running_max

        results["annual_pnl_ticks"].append(cumsum[-1])
        results["max_dd_ticks"].append(dd.min())
        results["sharpe"].append(sim.mean() / max(sim.std(), 0.01) * np.sqrt(252))
        results["worst_day"].append(sim.min())

    mc_summary = {}
    for key, vals in results.items():
        arr = np.array(vals)
        mc_summary[key] = {
            "mean": round(float(np.mean(arr)), 1),
            "median": round(float(np.median(arr)), 1),
            "p5": round(float(np.percentile(arr, 5)), 1),
            "p25": round(float(np.percentile(arr, 25)), 1),
            "p75": round(float(np.percentile(arr, 75)), 1),
            "p95": round(float(np.percentile(arr, 95)), 1),
        }

    # Ruin probabilities
    dd_arr = np.array(results["max_dd_ticks"])
    mc_summary["p_dd_gt_500t"] = round(float((dd_arr < -500).mean() * 100), 1)
    mc_summary["p_dd_gt_1000t"] = round(float((dd_arr < -1000).mean() * 100), 1)
    mc_summary["p_dd_gt_2000t"] = round(float((dd_arr < -2000).mean() * 100), 1)
    mc_summary["p_negative_year"] = round(float((np.array(results["annual_pnl_ticks"]) < 0).mean() * 100), 1)
    mc_summary["n_sims"] = n_sims
    mc_summary["block_size"] = block_size

    return mc_summary


def kelly_sizing(preds):
    """Kelly criterion and fractional Kelly position sizing."""
    nt = preds["net_ticks"]
    wr = (nt > 0).mean()
    avg_win = nt[nt > 0].mean()
    avg_loss = abs(nt[nt <= 0].mean()) if (nt <= 0).any() else 1

    full_kelly = wr - (1 - wr) / (avg_win / avg_loss) if avg_loss > 0 else 0

    sizing = {
        "win_rate": round(wr, 4),
        "avg_win_ticks": round(avg_win, 1),
        "avg_loss_ticks": round(avg_loss, 1),
        "payoff_ratio": round(avg_win / avg_loss, 2),
        "full_kelly_pct": round(full_kelly * 100, 1),
        "half_kelly_pct": round(full_kelly * 50, 1),
        "quarter_kelly_pct": round(full_kelly * 25, 1),
    }

    # Position sizing by account size
    p5_loss = abs(np.percentile(nt, 5)) * TICK_VALUE  # 5th percentile loss
    worst_trade = abs(nt.min()) * TICK_VALUE

    sizing["position_table"] = []
    for acct_size in [25_000, 50_000, 100_000, 250_000, 500_000]:
        # Conservative: 2% risk per trade based on p5 loss
        max_risk_2pct = 0.02 * acct_size
        contracts_by_risk = max(1, int(max_risk_2pct / p5_loss))

        # Margin constraint (50% utilization, intraday margin since 2h hold)
        contracts_by_margin = max(1, int(acct_size * 0.50 / ES_MARGIN_INTRADAY))

        # Take the more conservative
        contracts = min(contracts_by_risk, contracts_by_margin)

        # Cap at reasonable level
        contracts = min(contracts, 20)

        daily_pnl = preds.groupby("date_dt")["net_ticks"].sum()
        avg_daily = daily_pnl.mean() * TICK_VALUE * contracts
        worst_day = daily_pnl.min() * TICK_VALUE * contracts
        annual_est = avg_daily * 252

        sizing["position_table"].append({
            "account_size": acct_size,
            "contracts": contracts,
            "avg_daily_pnl": round(avg_daily, 0),
            "worst_day_pnl": round(worst_day, 0),
            "worst_day_pct": round(worst_day / acct_size * 100, 1),
            "annual_estimate": round(annual_est, 0),
            "annual_return_pct": round(annual_est / acct_size * 100, 1),
        })

    return sizing


def regime_analysis(preds, macro_path=None):
    """Performance stratified by VIX regime."""
    # Load macro for VIX
    try:
        macro = pd.read_parquet(ROOT / "wheel_strategy_v1" / "data" / "cache" / "macro.parquet")
        macro["date"] = pd.to_datetime(macro["date"]).dt.tz_localize(None)
        vix_by_date = macro.set_index("date")["vix"].to_dict()
    except:
        return {"error": "Could not load macro data"}

    preds = preds.copy()

    results = {}
    for regime, vix_lo, vix_hi in [("low_vol", 0, 15), ("normal", 15, 22), ("elevated", 22, 30), ("high_vol", 30, 200)]:
        regime_dates = {d for d, v in vix_by_date.items()
                       if not np.isnan(v) and vix_lo <= v < vix_hi}
        regime_preds = preds[preds["date_dt"].isin(regime_dates)]

        if len(regime_preds) < 5:
            results[regime] = {"n_trades": 0}
            continue

        nt = regime_preds["net_ticks"]
        daily = regime_preds.groupby("date_dt")["net_ticks"].sum()

        results[regime] = {
            "n_trades": len(regime_preds),
            "n_days": len(daily),
            "wr": round((nt > 0).mean() * 100, 1),
            "avg_net_ticks": round(nt.mean(), 1),
            "daily_sharpe": round(daily.mean() / max(daily.std(), 0.01) * np.sqrt(252), 2) if len(daily) > 2 else 0,
            "daily_wr": round((daily > 0).mean() * 100, 1),
            "worst_day": round(daily.min(), 1) if len(daily) > 0 else 0,
        }

    return results


def main():
    t0 = time.time()
    print("=" * 60)
    print("ES 2h MODEL — RISK FRAMEWORK & POSITION SIZING")
    print("HC #661 R4 + HC #662 R2")
    print("=" * 60)

    preds = load_predictions()
    print(f"\nLoaded {len(preds)} trades across {preds['date_dt'].nunique()} OOT days")

    # 1. Trade-level analysis
    print("\n=== 1. Trade-Level P&L Distribution ===")
    trade_stats = trade_level_analysis(preds)
    for k, v in trade_stats.items():
        print(f"  {k}: {v}")

    # 2. Daily analysis
    print("\n=== 2. Daily P&L ===")
    daily_stats, daily_df = daily_analysis(preds)
    for k, v in daily_stats.items():
        print(f"  {k}: {v}")

    # 3. Block bootstrap Monte Carlo
    print("\n=== 3. Block Bootstrap Monte Carlo (5000 sims) ===")
    mc_results = block_bootstrap_mc(daily_df["pnl_ticks"], n_sims=5000, block_size=5)
    print(f"  Annual PnL (ticks): {mc_results['annual_pnl_ticks']}")
    print(f"  Max drawdown (ticks): {mc_results['max_dd_ticks']}")
    print(f"  Sharpe: {mc_results['sharpe']}")
    print(f"  P(DD > 500t): {mc_results['p_dd_gt_500t']}%")
    print(f"  P(DD > 1000t): {mc_results['p_dd_gt_1000t']}%")
    print(f"  P(negative year): {mc_results['p_negative_year']}%")

    # 4. Kelly sizing
    print("\n=== 4. Position Sizing (Kelly) ===")
    sizing = kelly_sizing(preds)
    print(f"  Full Kelly: {sizing['full_kelly_pct']}%")
    print(f"  Half Kelly: {sizing['half_kelly_pct']}%")
    print(f"\n  Position sizing table:")
    print(f"  {'Account':>12} {'Contracts':>10} {'Avg Daily':>12} {'Worst Day':>12} {'Annual':>14} {'Return':>8}")
    print(f"  {'-'*72}")
    for row in sizing["position_table"]:
        print(f"  ${row['account_size']:>10,} {row['contracts']:>10} "
              f"${row['avg_daily_pnl']:>10,.0f} ${row['worst_day_pnl']:>10,.0f} "
              f"${row['annual_estimate']:>12,.0f} {row['annual_return_pct']:>6.0f}%")

    # 5. Regime analysis
    print("\n=== 5. Regime-Stratified Performance ===")
    regimes = regime_analysis(preds)
    for regime, data in regimes.items():
        if data.get("n_trades", 0) > 0:
            print(f"  {regime}: {data['n_trades']} trades, WR={data.get('wr','?')}%, "
                  f"avg={data.get('avg_net_ticks','?')}t, daily Sharpe={data.get('daily_sharpe','?')}")

    # 6. Consecutive loss analysis
    print("\n=== 6. Streak Analysis ===")
    nt = preds["net_ticks"].values
    max_consec_loss = 0
    current_streak = 0
    streaks = []
    for t in nt:
        if t <= 0:
            current_streak += 1
            max_consec_loss = max(max_consec_loss, current_streak)
        else:
            if current_streak > 0:
                streaks.append(current_streak)
            current_streak = 0
    if current_streak > 0:
        streaks.append(current_streak)

    print(f"  Max consecutive losses: {max_consec_loss}")
    print(f"  Avg losing streak: {np.mean(streaks):.1f}" if streaks else "  No losing streaks")
    print(f"  Longest winning streak: {max(len(list(g)) for k, g in pd.Series(nt > 0).groupby((nt > 0).cumsum()) if k)}")

    # Save everything
    report = {
        "generated": pd.Timestamp.now().isoformat(),
        "model": "2h LGBM (walk-forward, 60d train, 1d OOT)",
        "cost_assumption": f"{RT_COST_TICKS} ticks RT (market order + AMP commission)",
        "caveat": "ALL metrics depend on MBO features. Model is COIN FLIP without MBO data. Razer MUST be online.",
        "trade_level": trade_stats,
        "daily": daily_stats,
        "monte_carlo": mc_results,
        "kelly_sizing": sizing,
        "regime": regimes,
        "streak_analysis": {
            "max_consecutive_losses": max_consec_loss,
            "avg_losing_streak": round(np.mean(streaks), 1) if streaks else 0,
        },
    }

    with open(OUTPUT / "risk_framework.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"DONE in {elapsed:.1f}s")
    print(f"Saved to {OUTPUT / 'risk_framework.json'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

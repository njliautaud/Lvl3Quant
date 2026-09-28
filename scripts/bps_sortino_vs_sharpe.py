#!/usr/bin/env python3
"""
BPS Objective Function Comparison: Sharpe vs Sortino vs Calmar vs Tail-Aware
=============================================================================

Hypothesis: Sortino-optimal configs produce better risk-adjusted returns for
fat-tail strategies like BPS, because Sortino penalizes downside volatility
specifically (which is what kills BPS strategies via cluster losses).

Prior findings:
  - CVaR/VaR ratio = 3.20 (fat tails confirmed)
  - Cluster losses 6.6x expected
  - Sortino 0.81 vs Sharpe 1.55 -- big gap = asymmetric returns

Sweep dimensions:
  - delta: 15, 20, 25, 30, 35 (put delta * 100)
  - spread_width: $10, $15, $20
  - margin_util: 10%, 15%, 20%, 25%
  - profit_take: 40%, 50%, 65%, 80%

For each config: full BPS backtest with 5% BA cost, then rank by 4 objectives.

Output: output/bps_sortino_optimal/
"""

import sys
import json
import time
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta

ROOT = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(ROOT / "output" / "wheel_higher_returns_study"))

from higher_returns_study import (
    load_data, bs_price, bs_delta, strike_from_delta, trade_cost,
    COST_PER_CONTRACT,
)

OUTPUT = ROOT / "output" / "bps_sortino_optimal"
OUTPUT.mkdir(parents=True, exist_ok=True)


# ── Sweep Grid ──
DELTAS = [0.15, 0.20, 0.25, 0.30, 0.35]
SPREAD_WIDTHS = [10.0, 15.0, 20.0]
MARGIN_CAPS = [0.10, 0.15, 0.20, 0.25]
PROFIT_TAKES = [0.40, 0.50, 0.65, 0.80]


def compute_sweep_metrics(daily_pnl, starting_capital=100_000):
    """Compute all objective-function metrics from daily P&L series."""
    if len(daily_pnl) < 30:
        return None

    equity = starting_capital + daily_pnl.cumsum()
    rets = equity.pct_change().dropna()

    if rets.std() == 0 or len(rets) < 30:
        return None

    # Sharpe
    sharpe = float(rets.mean() / rets.std() * np.sqrt(252))

    # Sortino
    downside = rets[rets < 0]
    if len(downside) > 5 and downside.std() > 0:
        sortino = float(rets.mean() / downside.std() * np.sqrt(252))
    else:
        sortino = 0.0

    # Max drawdown
    peak = equity.cummax()
    dd = (equity - peak) / peak
    max_dd_pct = float(dd.min() * 100)

    # CAGR
    n_years = len(daily_pnl) / 252
    total_return = float(equity.iloc[-1] / starting_capital)
    if total_return > 0 and n_years > 0:
        cagr = (total_return ** (1 / n_years) - 1) * 100
    else:
        cagr = -100.0

    # Calmar = CAGR / |MaxDD|
    calmar = abs(cagr / max_dd_pct) if max_dd_pct != 0 else 0.0

    # Tail-aware = Sortino * (1 - MaxDD/100)
    # MaxDD is negative, so (1 - MaxDD/100) = (1 + |MaxDD|/100)
    # Actually MaxDD_pct is negative like -5.2, so 1 - (-5.2)/100 = 1.052
    tail_aware = sortino * (1 - max_dd_pct / 100)

    # Win rate
    daily_wr = float((daily_pnl > 0).sum() / len(daily_pnl) * 100)

    # Profit factor
    gross_profit = daily_pnl[daily_pnl > 0].sum()
    gross_loss = abs(daily_pnl[daily_pnl < 0].sum())
    pf = float(gross_profit / gross_loss) if gross_loss > 0 else float('inf')

    # Tail risk
    var_95 = float(np.percentile(daily_pnl.values, 5))
    cvar_95 = float(daily_pnl[daily_pnl <= var_95].mean()) if (daily_pnl <= var_95).sum() > 0 else var_95
    cvar_var_ratio = abs(cvar_95 / var_95) if var_95 != 0 else 1.0

    # Downside deviation (annualized)
    downside_dev = float(downside.std() * np.sqrt(252)) if len(downside) > 5 else 0.0

    return {
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "calmar": round(calmar, 3),
        "tail_aware": round(tail_aware, 3),
        "max_dd_pct": round(max_dd_pct, 2),
        "cagr_pct": round(cagr, 2),
        "daily_wr_pct": round(daily_wr, 1),
        "profit_factor": round(pf, 2),
        "total_pnl": round(float(daily_pnl.sum()), 0),
        "var_95": round(var_95, 0),
        "cvar_95": round(cvar_95, 0),
        "cvar_var_ratio": round(cvar_var_ratio, 2),
        "downside_dev": round(downside_dev, 4),
        "n_days": len(daily_pnl),
    }


def run_bps_config(prices_df, iv_df, macro, universe_tickers, px_by_date, sigma_by_date,
                   iv_rank_by_date, macro_by_date, all_dates,
                   put_delta, spread_width, margin_cap, profit_take,
                   ba_frac=0.05, starting_cash=100_000, dte_target=10,
                   max_concurrent=40, per_name_pct=0.03):
    """
    Run a single BPS configuration. Returns daily P&L series and trade count.
    Applies BA cost as fraction of premium (matching existing methodology).
    Uses close-1-DTE exit (our standard practice).
    """
    cash = starting_cash
    positions = {}
    daily_pnl_records = {}
    n_trades = 0
    prev_equity = starting_cash

    for di, dt in enumerate(all_dates):
        date_px = px_by_date.get(dt, {})
        date_sigma = sigma_by_date.get(dt, {})
        date_iv_rank = iv_rank_by_date.get(dt, {})
        m = macro_by_date.get(dt, {})
        vix = m.get("vix", float("nan")) if isinstance(m, dict) else float("nan")

        # ── Update / close positions ──
        to_remove = []
        for tk, pos in list(positions.items()):
            S = date_px.get(tk)
            if S is None or np.isnan(S):
                continue
            T_days = (pos["expiry"] - dt).days
            T = max(T_days, 0) / 365.0
            sigma_atm = date_sigma.get(tk, pos["open_sigma"]) or pos["open_sigma"] or 0.20

            # Close 1 DTE (standard practice)
            if T_days == 1:
                short_val = bs_price(S, pos["short_strike"], T, sigma_atm, kind="put")
                long_val = bs_price(S, pos["long_strike"], T, sigma_atm, kind="put")
                cost_to_close = (short_val - long_val) * 100 * pos["contracts"]
                close_fees = trade_cost(short_val, pos["contracts"]) + trade_cost(long_val, pos["contracts"])
                # BA cost on close
                ba_close = abs(short_val + long_val) * 100 * pos["contracts"] * ba_frac
                realized = pos["net_credit"] - cost_to_close - close_fees - ba_close
                cash -= cost_to_close + close_fees + ba_close
                to_remove.append(tk)
                continue

            # Expiry
            if T_days <= 0:
                short_itm = S < pos["short_strike"]
                long_itm = S < pos["long_strike"]
                close_cost = COST_PER_CONTRACT * 2 * pos["contracts"]

                if not short_itm:
                    realized = pos["net_credit"] - close_cost
                    cash -= close_cost
                elif short_itm and not long_itm:
                    loss = (pos["short_strike"] - S) * 100 * pos["contracts"]
                    realized = pos["net_credit"] - loss - close_cost
                    cash -= loss + close_cost
                else:
                    loss = (pos["short_strike"] - pos["long_strike"]) * 100 * pos["contracts"]
                    realized = pos["net_credit"] - loss - close_cost
                    cash -= loss + close_cost
                to_remove.append(tk)
                continue

            # Profit take check
            short_val = bs_price(S, pos["short_strike"], T, sigma_atm, kind="put")
            long_val = bs_price(S, pos["long_strike"], T, sigma_atm, kind="put")
            spread_val = (short_val - long_val) * 100 * pos["contracts"]
            cost_to_close = spread_val + trade_cost(short_val, pos["contracts"]) + trade_cost(long_val, pos["contracts"])
            # BA cost on close
            ba_close = abs(short_val + long_val) * 100 * pos["contracts"] * ba_frac
            cost_to_close += ba_close

            captured = (pos["net_credit"] - cost_to_close) / max(pos["net_credit"], 1e-6)
            if captured >= profit_take:
                realized = pos["net_credit"] - cost_to_close
                cash -= cost_to_close
                to_remove.append(tk)

        for tk in to_remove:
            del positions[tk]

        # ── MTM equity ──
        equity = cash
        for tk, pos in positions.items():
            S = date_px.get(tk)
            if S is None or np.isnan(S):
                continue
            T = max((pos["expiry"] - dt).days, 0) / 365.0
            sigma_atm = date_sigma.get(tk, pos["open_sigma"]) or pos["open_sigma"] or 0.20
            short_val = bs_price(S, pos["short_strike"], T, sigma_atm, kind="put")
            long_val = bs_price(S, pos["long_strike"], T, sigma_atm, kind="put")
            equity -= (short_val - long_val) * 100 * pos["contracts"]

        daily_pnl_records[dt] = equity - prev_equity
        prev_equity = equity

        # ── VIX gate (skip opening in crisis) ──
        try:
            vix_val = float(vix)
        except (TypeError, ValueError):
            vix_val = float("nan")
        if not np.isnan(vix_val) and vix_val > 35:
            continue

        # ── Open new positions ──
        current_margin = sum(
            (p["short_strike"] - p["long_strike"]) * 100 * p["contracts"]
            for p in positions.values()
        )
        max_margin = margin_cap * equity
        remaining_margin = max_margin - current_margin
        slots = max_concurrent - len(positions)

        if slots <= 0 or remaining_margin <= 0:
            continue

        candidates = []
        for tk in universe_tickers:
            if tk in positions:
                continue
            S = date_px.get(tk)
            sigma = date_sigma.get(tk)
            iv_rk = date_iv_rank.get(tk, 0)
            if S is None or sigma is None or np.isnan(S) or np.isnan(sigma):
                continue
            if sigma < 0.05:
                continue
            candidates.append((tk, S, sigma, iv_rk))

        candidates.sort(key=lambda x: -x[3])

        for tk, S, sigma, iv_rk in candidates[:slots]:
            T = dte_target / 365.0
            K_short = strike_from_delta(S, T, sigma, put_delta, kind="put")
            K_long = K_short - spread_width

            if K_long <= 0 or K_short <= 0:
                continue

            prem_short = bs_price(S, K_short, T, sigma, kind="put")
            prem_long = bs_price(S, K_long, T, sigma, kind="put")
            net_prem_per_share = prem_short - prem_long

            if net_prem_per_share <= 0.05:
                continue

            margin_per_contract = spread_width * 100
            max_alloc = per_name_pct * equity
            n_contracts = max(1, int(max_alloc // margin_per_contract))

            if margin_per_contract * n_contracts > remaining_margin:
                n_contracts = max(1, int(remaining_margin // margin_per_contract))
            if n_contracts < 1:
                continue

            net_credit = net_prem_per_share * 100 * n_contracts
            open_costs = trade_cost(prem_short, n_contracts) + trade_cost(prem_long, n_contracts)
            # BA cost on open
            ba_open = net_prem_per_share * 100 * n_contracts * ba_frac
            net_credit -= open_costs + ba_open

            if net_credit <= 0:
                continue

            cash += net_credit
            n_trades += 1
            positions[tk] = {
                "short_strike": K_short,
                "long_strike": K_long,
                "contracts": n_contracts,
                "net_credit": net_credit,
                "open_date": dt,
                "expiry": dt + pd.Timedelta(days=dte_target),
                "open_sigma": sigma,
            }
            remaining_margin -= margin_per_contract * n_contracts

            if len(positions) >= max_concurrent:
                break

    # Build daily P&L series
    daily_pnl = pd.Series(daily_pnl_records).sort_index()
    all_dates_range = pd.date_range(daily_pnl.index.min(), daily_pnl.index.max(), freq='B')
    daily_pnl = daily_pnl.reindex(all_dates_range, fill_value=0.0)

    return daily_pnl, n_trades


def run_sweep(prices, iv, macro, fund, universe, earnings):
    """Run the full parameter sweep."""

    # Pre-compute lookups (same as existing scripts)
    prices_df = prices.copy()
    iv_df = iv.copy()

    px_by_date = {}
    for d, g in prices_df.groupby("date"):
        px_by_date[d] = g.set_index("ticker")["close"].to_dict()

    sigma_by_date = {}
    iv_rank_by_date = {}
    for d, g in iv_df.groupby("date"):
        sigma_by_date[d] = g.set_index("ticker")["sigma"].to_dict()
        iv_rank_by_date[d] = g.set_index("ticker")["iv_rank"].to_dict()

    macro_sorted = macro.copy()
    macro_sorted["date"] = pd.to_datetime(macro_sorted["date"])
    macro_by_date = macro_sorted.set_index("date").to_dict("index")

    if isinstance(universe, pd.DataFrame):
        universe_tickers = universe["ticker"].tolist()
    else:
        universe_tickers = list(universe)

    all_dates = sorted(prices_df["date"].unique())

    # Generate all combos
    combos = list(itertools.product(DELTAS, SPREAD_WIDTHS, MARGIN_CAPS, PROFIT_TAKES))
    total = len(combos)
    print(f"\nSweeping {total} configurations...")

    results = []
    t_sweep = time.time()

    for i, (delta, width, margin, pt) in enumerate(combos):
        label = f"d{int(delta*100)}_w{int(width)}_m{int(margin*100)}_pt{int(pt*100)}"

        if (i + 1) % 20 == 0 or i == 0:
            elapsed = time.time() - t_sweep
            eta = (elapsed / max(i, 1)) * (total - i)
            print(f"  [{i+1}/{total}] {label} ... (elapsed {elapsed:.0f}s, ETA {eta:.0f}s)")

        try:
            daily_pnl, n_trades = run_bps_config(
                prices_df, iv_df, macro, universe_tickers,
                px_by_date, sigma_by_date, iv_rank_by_date, macro_by_date, all_dates,
                put_delta=delta, spread_width=width, margin_cap=margin,
                profit_take=pt, ba_frac=0.05,
            )

            metrics = compute_sweep_metrics(daily_pnl)
            if metrics is None:
                continue

            metrics["config"] = {
                "delta": delta,
                "spread_width": width,
                "margin_cap": margin,
                "profit_take": pt,
            }
            metrics["label"] = label
            metrics["n_trades"] = n_trades

            results.append(metrics)

        except Exception as e:
            print(f"    ERROR on {label}: {e}")
            continue

    print(f"\nSweep complete: {len(results)}/{total} configs valid in {time.time()-t_sweep:.0f}s")
    return results


def analyze_objectives(results):
    """Rank configs by each objective and compare."""
    if not results:
        return {}

    objectives = {
        "sharpe": ("sharpe", True),       # higher is better
        "sortino": ("sortino", True),
        "calmar": ("calmar", True),
        "tail_aware": ("tail_aware", True),
    }

    analysis = {}

    for obj_name, (key, ascending) in objectives.items():
        ranked = sorted(results, key=lambda r: r.get(key, -999), reverse=ascending)
        top3 = ranked[:3]

        analysis[f"top3_{obj_name}"] = []
        for rank, r in enumerate(top3, 1):
            analysis[f"top3_{obj_name}"].append({
                "rank": rank,
                "label": r["label"],
                "config": r["config"],
                "sharpe": r["sharpe"],
                "sortino": r["sortino"],
                "calmar": r["calmar"],
                "tail_aware": r["tail_aware"],
                "max_dd_pct": r["max_dd_pct"],
                "cagr_pct": r["cagr_pct"],
                "profit_factor": r["profit_factor"],
                "daily_wr_pct": r["daily_wr_pct"],
                "cvar_var_ratio": r["cvar_var_ratio"],
                "n_trades": r["n_trades"],
            })

    # Cross-objective comparison: how much do top configs differ?
    top1_configs = {}
    for obj_name in objectives:
        ranked = sorted(results, key=lambda r: r.get(objectives[obj_name][0], -999), reverse=True)
        if ranked:
            top1_configs[obj_name] = ranked[0]

    # Agreement matrix: do different objectives pick the same config?
    obj_names = list(top1_configs.keys())
    agreement = {}
    for i, o1 in enumerate(obj_names):
        for j, o2 in enumerate(obj_names):
            if i >= j:
                continue
            same = top1_configs[o1]["label"] == top1_configs[o2]["label"]
            agreement[f"{o1}_vs_{o2}"] = {
                "same_config": same,
                f"{o1}_best": top1_configs[o1]["label"],
                f"{o2}_best": top1_configs[o2]["label"],
            }
    analysis["agreement"] = agreement

    # Key question: does Sortino-optimal trade less aggressively?
    sharpe_best = top1_configs.get("sharpe", {})
    sortino_best = top1_configs.get("sortino", {})
    calmar_best = top1_configs.get("calmar", {})
    tail_best = top1_configs.get("tail_aware", {})

    comparison = {
        "sharpe_optimal": {
            "config": sharpe_best.get("config"),
            "sharpe": sharpe_best.get("sharpe"),
            "sortino": sharpe_best.get("sortino"),
            "max_dd_pct": sharpe_best.get("max_dd_pct"),
            "cagr_pct": sharpe_best.get("cagr_pct"),
            "cvar_var_ratio": sharpe_best.get("cvar_var_ratio"),
            "n_trades": sharpe_best.get("n_trades"),
        },
        "sortino_optimal": {
            "config": sortino_best.get("config"),
            "sharpe": sortino_best.get("sharpe"),
            "sortino": sortino_best.get("sortino"),
            "max_dd_pct": sortino_best.get("max_dd_pct"),
            "cagr_pct": sortino_best.get("cagr_pct"),
            "cvar_var_ratio": sortino_best.get("cvar_var_ratio"),
            "n_trades": sortino_best.get("n_trades"),
        },
        "calmar_optimal": {
            "config": calmar_best.get("config"),
            "sharpe": calmar_best.get("sharpe"),
            "sortino": calmar_best.get("sortino"),
            "max_dd_pct": calmar_best.get("max_dd_pct"),
            "cagr_pct": calmar_best.get("cagr_pct"),
            "cvar_var_ratio": calmar_best.get("cvar_var_ratio"),
            "n_trades": calmar_best.get("n_trades"),
        },
        "tail_aware_optimal": {
            "config": tail_best.get("config"),
            "sharpe": tail_best.get("sharpe"),
            "sortino": tail_best.get("sortino"),
            "max_dd_pct": tail_best.get("max_dd_pct"),
            "cagr_pct": tail_best.get("cagr_pct"),
            "cvar_var_ratio": tail_best.get("cvar_var_ratio"),
            "n_trades": tail_best.get("n_trades"),
        },
    }
    analysis["head_to_head"] = comparison

    # Aggressiveness comparison
    if sharpe_best.get("config") and sortino_best.get("config"):
        sc = sharpe_best["config"]
        soc = sortino_best["config"]
        analysis["aggressiveness_comparison"] = {
            "hypothesis": "Sortino-optimal trades less aggressively to avoid fat-tail losses",
            "sharpe_optimal_delta": sc.get("delta"),
            "sortino_optimal_delta": soc.get("delta"),
            "delta_direction": "lower (more conservative)" if soc.get("delta", 0) < sc.get("delta", 0) else
                              "same" if soc.get("delta") == sc.get("delta") else
                              "higher (more aggressive)",
            "sharpe_optimal_margin": sc.get("margin_cap"),
            "sortino_optimal_margin": soc.get("margin_cap"),
            "margin_direction": "lower (more conservative)" if soc.get("margin_cap", 0) < sc.get("margin_cap", 0) else
                               "same" if soc.get("margin_cap") == sc.get("margin_cap") else
                               "higher (more aggressive)",
            "sharpe_optimal_width": sc.get("spread_width"),
            "sortino_optimal_width": soc.get("spread_width"),
            "sharpe_optimal_pt": sc.get("profit_take"),
            "sortino_optimal_pt": soc.get("profit_take"),
        }

    # Parameter sensitivity: average metric by each dimension
    dim_analysis = {}
    for dim_name, dim_key in [("delta", "delta"), ("spread_width", "spread_width"),
                               ("margin_cap", "margin_cap"), ("profit_take", "profit_take")]:
        by_dim = {}
        for r in results:
            val = r["config"][dim_key]
            if val not in by_dim:
                by_dim[val] = []
            by_dim[val].append(r)

        dim_stats = {}
        for val, configs in sorted(by_dim.items()):
            sharpes = [c["sharpe"] for c in configs]
            sortinos = [c["sortino"] for c in configs]
            max_dds = [c["max_dd_pct"] for c in configs]
            dim_stats[str(val)] = {
                "n_configs": len(configs),
                "avg_sharpe": round(np.mean(sharpes), 3),
                "avg_sortino": round(np.mean(sortinos), 3),
                "avg_max_dd": round(np.mean(max_dds), 2),
                "best_sharpe": round(max(sharpes), 3),
                "best_sortino": round(max(sortinos), 3),
            }
        dim_analysis[dim_name] = dim_stats
    analysis["parameter_sensitivity"] = dim_analysis

    return analysis


def print_summary(analysis):
    """Print human-readable summary."""
    print("\n" + "=" * 90)
    print("BPS OBJECTIVE FUNCTION COMPARISON: Sharpe vs Sortino vs Calmar vs Tail-Aware")
    print("=" * 90)

    for obj in ["sharpe", "sortino", "calmar", "tail_aware"]:
        top3 = analysis.get(f"top3_{obj}", [])
        if not top3:
            continue
        print(f"\n--- TOP 3 by {obj.upper()} ---")
        print(f"{'Rank':<5} {'Config':<25} {'Sharpe':>8} {'Sortino':>8} {'Calmar':>8} {'TailAw':>8} {'MaxDD':>8} {'CAGR%':>7} {'PF':>6} {'WR%':>6} {'Trades':>7}")
        print("-" * 100)
        for r in top3:
            print(f"{r['rank']:<5} {r['label']:<25} {r['sharpe']:>8.3f} {r['sortino']:>8.3f} "
                  f"{r['calmar']:>8.3f} {r['tail_aware']:>8.3f} {r['max_dd_pct']:>7.2f}% "
                  f"{r['cagr_pct']:>6.1f}% {r['profit_factor']:>6.2f} {r['daily_wr_pct']:>5.1f}% {r['n_trades']:>7}")

    # Head to head
    h2h = analysis.get("head_to_head", {})
    if h2h:
        print("\n--- HEAD-TO-HEAD: Optimal Config per Objective ---")
        print(f"{'Objective':<18} {'Config':<25} {'Sharpe':>8} {'Sortino':>8} {'MaxDD':>8} {'CAGR%':>7} {'CVaR/VaR':>8}")
        print("-" * 90)
        for obj_name, data in h2h.items():
            cfg = data.get("config", {})
            label = f"d{int(cfg.get('delta',0)*100)}_w{int(cfg.get('spread_width',0))}_m{int(cfg.get('margin_cap',0)*100)}_pt{int(cfg.get('profit_take',0)*100)}"
            print(f"{obj_name:<18} {label:<25} {data.get('sharpe',0):>8.3f} {data.get('sortino',0):>8.3f} "
                  f"{data.get('max_dd_pct',0):>7.2f}% {data.get('cagr_pct',0):>6.1f}% {data.get('cvar_var_ratio',0):>8.2f}")

    # Agreement
    agr = analysis.get("agreement", {})
    if agr:
        print("\n--- OBJECTIVE AGREEMENT ---")
        for pair, info in agr.items():
            same = "YES" if info["same_config"] else "NO"
            print(f"  {pair}: Same config? {same}")

    # Aggressiveness
    agg = analysis.get("aggressiveness_comparison", {})
    if agg:
        print(f"\n--- AGGRESSIVENESS COMPARISON ---")
        print(f"  Hypothesis: {agg.get('hypothesis')}")
        print(f"  Sharpe-optimal delta: {agg.get('sharpe_optimal_delta')} | Sortino-optimal delta: {agg.get('sortino_optimal_delta')} -> {agg.get('delta_direction')}")
        print(f"  Sharpe-optimal margin: {agg.get('sharpe_optimal_margin')} | Sortino-optimal margin: {agg.get('sortino_optimal_margin')} -> {agg.get('margin_direction')}")
        print(f"  Sharpe-optimal width: {agg.get('sharpe_optimal_width')} | Sortino-optimal width: {agg.get('sortino_optimal_width')}")
        print(f"  Sharpe-optimal PT: {agg.get('sharpe_optimal_pt')} | Sortino-optimal PT: {agg.get('sortino_optimal_pt')}")

    # Parameter sensitivity
    ps = analysis.get("parameter_sensitivity", {})
    if ps:
        print("\n--- PARAMETER SENSITIVITY (avg across all other dims) ---")
        for dim_name, dim_data in ps.items():
            print(f"\n  {dim_name}:")
            print(f"    {'Value':<10} {'AvgSharpe':>10} {'AvgSortino':>11} {'AvgMaxDD':>10} {'BestSharpe':>11} {'BestSortino':>12}")
            for val, stats in dim_data.items():
                print(f"    {val:<10} {stats['avg_sharpe']:>10.3f} {stats['avg_sortino']:>11.3f} "
                      f"{stats['avg_max_dd']:>9.2f}% {stats['best_sharpe']:>11.3f} {stats['best_sortino']:>12.3f}")


def main():
    t0 = time.time()
    print("Loading data...")
    prices, iv, macro, fund, universe, earnings = load_data()
    print(f"Data loaded in {time.time()-t0:.1f}s")

    # Run the sweep
    results = run_sweep(prices, iv, macro, fund, universe, earnings)

    if not results:
        print("ERROR: No valid configs produced. Check data.")
        return

    # Analyze objectives
    analysis = analyze_objectives(results)

    # Print summary
    print_summary(analysis)

    # Save everything
    def convert_types(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {str(k): convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(v) for v in obj]
        return obj

    # Save analysis
    analysis_clean = convert_types(analysis)
    with open(OUTPUT / "objective_comparison.json", "w") as f:
        json.dump(analysis_clean, f, indent=2, default=str)

    # Save all config results for further analysis
    results_clean = convert_types(results)
    with open(OUTPUT / "all_configs.json", "w") as f:
        json.dump(results_clean, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s. Results saved to {OUTPUT}")


if __name__ == "__main__":
    main()

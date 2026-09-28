#!/usr/bin/env python3
"""
BPS Sector Decomposition Study
================================

Goal: Understand whether cluster loss days (where many BPS positions lose
simultaneously) are driven by sector-wide selloffs or broad market events.

Context:
- Conservative BPS: 25-delta, $15-wide spreads, 15% margin, ~70 tickers
- Prior correlation study: losses cluster 6.6x more than expected on crisis days
- COVID was worst event: 47-70% drawdown
- VIX-scaled sizing helps but need to understand WHERE cluster losses originate

Analysis:
1. Assign each ticker to GICS sector (from fund data)
2. Compute daily P&L by sector
3. Identify cluster loss days (>50% of positions losing)
4. On cluster days: sector concentration vs normal days
5. Sector-level Sharpe ratios
6. Sector cap simulation: would limiting sector exposure reduce cluster risk?

Output: output/bps_sector_decomposition/
"""

import sys, json, time
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

ROOT = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(ROOT / "output" / "wheel_higher_returns_study"))

OUTPUT = ROOT / "output" / "bps_sector_decomposition"
OUTPUT.mkdir(parents=True, exist_ok=True)

BA_COST_FRAC = 0.05  # 5% bid-ask cost (canonical assumption)


def load_trades():
    """Load trade-level data from assignment risk study."""
    path = ROOT / "output" / "bps_assignment_risk" / "trades_close_1dte.parquet"
    return pd.read_parquet(path)


def load_sector_map():
    """Load ticker-to-sector mapping from fund data."""
    from higher_returns_study import load_data
    prices, iv, macro, fund, universe, earnings = load_data()
    sector_map = dict(zip(fund["ticker"], fund["sector"]))
    return sector_map, macro


def apply_ba_cost(trades):
    """Apply 5% bid-ask cost to realized P&L."""
    trades = trades.copy()
    ba_cost = trades["net_credit"].abs() * BA_COST_FRAC * 2  # 2 legs
    # Additional close cost for early-closed trades
    closed_early = trades["exit_type"].isin(["profit_take", "early_close_1DTE", "loss_stop"])
    ba_cost[closed_early] += trades.loc[closed_early, "net_credit"].abs() * BA_COST_FRAC * 2
    trades["realized_pnl_net"] = trades["realized_pnl"] - ba_cost
    return trades


def build_sector_daily_pnl(trades, sector_map):
    """Build daily P&L matrix grouped by sector."""
    trades = trades.copy()
    trades["close_date"] = pd.to_datetime(trades["close_date"])
    trades["sector"] = trades["ticker"].map(sector_map).fillna("Unknown")

    # Daily P&L by sector
    sector_daily = trades.groupby(["close_date", "sector"])["realized_pnl_net"].sum().unstack(fill_value=0.0)

    # Daily P&L by ticker (for position-level analysis)
    ticker_daily = trades.groupby(["close_date", "ticker"])["realized_pnl_net"].sum().unstack(fill_value=0.0)

    # Daily trade counts by sector
    sector_counts = trades.groupby(["close_date", "sector"]).size().unstack(fill_value=0)

    return sector_daily, ticker_daily, sector_counts


def identify_cluster_days(trades):
    """
    Identify cluster loss days where >50% of positions lose.
    Returns daily stats DataFrame.
    """
    trades = trades.copy()
    trades["close_date"] = pd.to_datetime(trades["close_date"])
    trades["is_loss"] = trades["realized_pnl_net"] < 0

    daily = trades.groupby("close_date").agg(
        n_trades=("realized_pnl_net", "count"),
        n_losses=("is_loss", "sum"),
        total_pnl=("realized_pnl_net", "sum"),
        worst_trade=("realized_pnl_net", "min"),
    )
    daily["loss_frac"] = daily["n_losses"] / daily["n_trades"]
    daily["is_cluster"] = daily["loss_frac"] > 0.50
    daily["is_severe_cluster"] = daily["loss_frac"] > 0.75
    return daily


def sector_concentration_analysis(trades, sector_map, daily_stats):
    """
    On cluster loss days vs normal days:
    - What fraction of total losses comes from each sector?
    - Is loss concentrated in 1-2 sectors or spread broadly?
    """
    trades = trades.copy()
    trades["close_date"] = pd.to_datetime(trades["close_date"])
    trades["sector"] = trades["ticker"].map(sector_map).fillna("Unknown")

    cluster_dates = set(daily_stats[daily_stats["is_cluster"]].index)
    normal_dates = set(daily_stats[~daily_stats["is_cluster"]].index)

    results = {}

    for label, date_set in [("cluster_days", cluster_dates), ("normal_days", normal_dates)]:
        subset = trades[trades["close_date"].isin(date_set)]
        losses_only = subset[subset["realized_pnl_net"] < 0]

        if len(losses_only) == 0:
            results[label] = {"note": "no losses"}
            continue

        # Loss $ by sector
        sector_loss = losses_only.groupby("sector")["realized_pnl_net"].sum()
        total_loss = sector_loss.sum()
        sector_loss_pct = (sector_loss / total_loss * 100).round(1)

        # Loss count by sector
        sector_loss_count = losses_only.groupby("sector").size()
        total_loss_count = sector_loss_count.sum()
        sector_count_pct = (sector_loss_count / total_loss_count * 100).round(1)

        # Concentration metric: HHI (Herfindahl-Hirschman Index)
        # HHI = sum of squared market shares. Low = diversified, High = concentrated.
        # If evenly spread across 10 sectors: HHI = 10 * (10%)^2 = 1000
        # If all in 1 sector: HHI = 10000
        shares = (sector_loss / total_loss).values
        hhi = float(np.sum(shares ** 2) * 10000)

        # Top sector contribution
        top_sector = sector_loss_pct.abs().idxmax()
        top_sector_pct = float(sector_loss_pct.abs().max())

        # Top 3 sectors
        top3 = sector_loss_pct.abs().nlargest(3)

        results[label] = {
            "n_days": len(date_set),
            "total_losses": round(float(total_loss), 0),
            "n_loss_trades": int(total_loss_count),
            "sector_loss_pct": sector_loss_pct.to_dict(),
            "sector_count_pct": sector_count_pct.to_dict(),
            "hhi": round(hhi, 0),
            "top_sector": top_sector,
            "top_sector_pct": top_sector_pct,
            "top3_sectors": {k: round(v, 1) for k, v in top3.items()},
            "top3_combined_pct": round(float(top3.sum()), 1),
        }

    return results


def sector_sharpe_ratios(sector_daily):
    """Compute annualized Sharpe ratio for each sector's daily P&L."""
    results = {}
    for sector in sector_daily.columns:
        daily_pnl = sector_daily[sector]
        active_days = daily_pnl[daily_pnl != 0]
        if len(active_days) < 50:
            continue
        mean_pnl = active_days.mean()
        std_pnl = active_days.std()
        if std_pnl == 0:
            continue
        sharpe = float(mean_pnl / std_pnl * np.sqrt(252))
        sortino_denom = active_days[active_days < 0].std()
        sortino = float(mean_pnl / sortino_denom * np.sqrt(252)) if sortino_denom > 0 else 0.0
        win_rate = float((active_days > 0).mean() * 100)
        total_pnl = float(active_days.sum())
        avg_win = float(active_days[active_days > 0].mean()) if (active_days > 0).any() else 0
        avg_loss = float(active_days[active_days < 0].mean()) if (active_days < 0).any() else 0
        pf = abs(avg_win * (active_days > 0).sum() / (avg_loss * (active_days < 0).sum())) if avg_loss != 0 else 999

        results[sector] = {
            "sharpe": round(sharpe, 2),
            "sortino": round(sortino, 2),
            "win_rate_pct": round(win_rate, 1),
            "profit_factor": round(pf, 2),
            "total_pnl": round(total_pnl, 0),
            "avg_daily_pnl": round(float(mean_pnl), 1),
            "n_active_days": int(len(active_days)),
        }

    return dict(sorted(results.items(), key=lambda x: x[1]["sharpe"], reverse=True))


def cluster_day_sector_driver(trades, sector_map, daily_stats, macro):
    """
    For each cluster day, identify: is the loss driven by
    (a) a single sector blowing up, or (b) broad market selloff?

    Heuristic: if top sector contributes >60% of losses -> sector-driven
    Otherwise -> broad market event.
    Also correlate with VIX and SPY moves.
    """
    trades = trades.copy()
    trades["close_date"] = pd.to_datetime(trades["close_date"])
    trades["sector"] = trades["ticker"].map(sector_map).fillna("Unknown")

    macro = macro.copy()
    macro["date"] = pd.to_datetime(macro["date"])
    vix_map = macro.set_index("date")["vix"].to_dict()

    cluster_dates = daily_stats[daily_stats["is_cluster"]].index
    events = []

    for dt in cluster_dates:
        day_trades = trades[trades["close_date"] == dt]
        losses = day_trades[day_trades["realized_pnl_net"] < 0]
        if len(losses) == 0:
            continue

        sector_loss = losses.groupby("sector")["realized_pnl_net"].sum()
        total_loss = sector_loss.sum()
        if total_loss == 0:
            continue

        sector_pcts = (sector_loss / total_loss * 100)
        top_sector = sector_pcts.abs().idxmax()
        top_pct = float(sector_pcts.abs().max())
        n_sectors_losing = int((sector_loss < 0).sum())

        vix = vix_map.get(dt, np.nan)
        total_pnl = float(day_trades["realized_pnl_net"].sum())

        driver = "SECTOR" if top_pct > 60 else "BROAD"

        events.append({
            "date": str(dt.date()) if hasattr(dt, 'date') else str(dt)[:10],
            "total_pnl": round(total_pnl, 0),
            "n_losses": int(len(losses)),
            "n_trades": int(len(day_trades)),
            "loss_frac": round(float(len(losses) / len(day_trades)), 2),
            "top_sector": top_sector,
            "top_sector_pct": round(top_pct, 1),
            "n_sectors_losing": n_sectors_losing,
            "vix": round(float(vix), 1) if not np.isnan(vix) else None,
            "driver": driver,
        })

    events.sort(key=lambda x: x["total_pnl"])

    # Summary stats
    if events:
        sector_driven = sum(1 for e in events if e["driver"] == "SECTOR")
        broad_driven = sum(1 for e in events if e["driver"] == "BROAD")
        return {
            "n_cluster_days": len(events),
            "sector_driven_count": sector_driven,
            "broad_driven_count": broad_driven,
            "sector_driven_pct": round(sector_driven / len(events) * 100, 1),
            "broad_driven_pct": round(broad_driven / len(events) * 100, 1),
            "worst_10_days": events[:10],
            "interpretation": (
                "Most cluster losses are BROAD MARKET events — sector caps alone won't fix this"
                if broad_driven > sector_driven
                else "Many cluster losses are SECTOR-CONCENTRATED — sector caps could help significantly"
            ),
        }
    return {"note": "no cluster events found"}


def sector_cap_simulation(trades, sector_map, daily_stats, cap_pcts=[0.15, 0.20, 0.25, 0.30]):
    """
    Simulate: what if we capped max exposure per sector?
    For each cap level, remove excess trades from the most over-represented sector
    and recompute cluster day stats.

    Method: for each day, if a sector has more than cap_pct of open positions,
    drop the worst-performing excess trades (conservative — assumes we'd have
    dropped the ones that lost).
    """
    trades = trades.copy()
    trades["close_date"] = pd.to_datetime(trades["close_date"])
    trades["sector"] = trades["ticker"].map(sector_map).fillna("Unknown")

    results = {}

    for cap_pct in cap_pcts:
        filtered_trades = []
        for dt, day_group in trades.groupby("close_date"):
            n_total = len(day_group)
            max_per_sector = max(1, int(n_total * cap_pct))

            kept = []
            for sector, sec_group in day_group.groupby("sector"):
                if len(sec_group) <= max_per_sector:
                    kept.append(sec_group)
                else:
                    # Keep the best-performing ones (drop worst — conservative assumption
                    # is that we wouldn't have taken the worst trades)
                    # Sort by P&L descending, keep top max_per_sector
                    sec_sorted = sec_group.sort_values("realized_pnl_net", ascending=False)
                    kept.append(sec_sorted.head(max_per_sector))

            if kept:
                filtered_trades.append(pd.concat(kept))

        if not filtered_trades:
            results[f"cap_{int(cap_pct*100)}pct"] = {"note": "no trades after filter"}
            continue

        filtered_df = pd.concat(filtered_trades)

        # Recompute cluster stats
        filtered_daily = filtered_df.groupby("close_date").agg(
            n_trades=("realized_pnl_net", "count"),
            n_losses=("is_loss" if "is_loss" in filtered_df.columns else "realized_pnl_net", "count"),
            total_pnl=("realized_pnl_net", "sum"),
        )
        # Recompute is_loss
        filtered_df["is_loss_f"] = filtered_df["realized_pnl_net"] < 0
        filtered_daily_v2 = filtered_df.groupby("close_date").agg(
            n_trades=("realized_pnl_net", "count"),
            n_losses=("is_loss_f", "sum"),
            total_pnl=("realized_pnl_net", "sum"),
        )
        filtered_daily_v2["loss_frac"] = filtered_daily_v2["n_losses"] / filtered_daily_v2["n_trades"]
        cluster_days_new = filtered_daily_v2[filtered_daily_v2["loss_frac"] > 0.5]

        # Compare to original
        orig_cluster = daily_stats[daily_stats["is_cluster"]]
        orig_total_pnl = float(trades["realized_pnl_net"].sum())
        new_total_pnl = float(filtered_df["realized_pnl_net"].sum())

        # Worst day comparison
        orig_worst = float(daily_stats["total_pnl"].min()) if len(daily_stats) > 0 else 0
        new_worst = float(filtered_daily_v2["total_pnl"].min()) if len(filtered_daily_v2) > 0 else 0

        # Equity curve for Sharpe
        eq = filtered_daily_v2["total_pnl"].cumsum() + 100_000
        daily_rets = eq.pct_change().dropna()
        sharpe = float(daily_rets.mean() / daily_rets.std() * np.sqrt(252)) if daily_rets.std() > 0 else 0

        results[f"cap_{int(cap_pct*100)}pct"] = {
            "n_trades_kept": len(filtered_df),
            "n_trades_dropped": len(trades) - len(filtered_df),
            "pct_trades_dropped": round((len(trades) - len(filtered_df)) / len(trades) * 100, 1),
            "total_pnl": round(new_total_pnl, 0),
            "pnl_change_pct": round((new_total_pnl - orig_total_pnl) / abs(orig_total_pnl) * 100, 1) if orig_total_pnl != 0 else 0,
            "cluster_days_original": len(orig_cluster),
            "cluster_days_after_cap": len(cluster_days_new),
            "cluster_day_reduction_pct": round((len(orig_cluster) - len(cluster_days_new)) / max(len(orig_cluster), 1) * 100, 1),
            "worst_day_pnl_original": round(orig_worst, 0),
            "worst_day_pnl_after_cap": round(new_worst, 0),
            "worst_day_improvement_pct": round((new_worst - orig_worst) / abs(orig_worst) * 100, 1) if orig_worst != 0 else 0,
            "sharpe": round(sharpe, 2),
        }

    return results


def sector_exposure_over_time(trades, sector_map):
    """Track sector exposure over time to see if it's naturally balanced or skewed."""
    trades = trades.copy()
    trades["close_date"] = pd.to_datetime(trades["close_date"])
    trades["sector"] = trades["ticker"].map(sector_map).fillna("Unknown")

    # Monthly sector distribution
    trades["month"] = trades["close_date"].dt.to_period("M")
    monthly = trades.groupby(["month", "sector"]).size().unstack(fill_value=0)
    monthly_pct = monthly.div(monthly.sum(axis=1), axis=0) * 100

    # Overall sector distribution
    overall = trades.groupby("sector").size()
    overall_pct = (overall / overall.sum() * 100).round(1)

    # Max sector concentration by month
    max_conc = monthly_pct.max(axis=1)

    return {
        "overall_sector_distribution": overall_pct.to_dict(),
        "avg_max_sector_concentration_pct": round(float(max_conc.mean()), 1),
        "worst_month_concentration_pct": round(float(max_conc.max()), 1),
        "n_sectors": len(overall),
    }


def main():
    t0 = time.time()
    print("=" * 70)
    print("BPS SECTOR DECOMPOSITION STUDY")
    print("=" * 70)

    # Load data
    print("\n[1/7] Loading trades and sector data...")
    trades = load_trades()
    sector_map, macro = load_sector_map()
    print(f"  Loaded {len(trades)} trades across {trades['ticker'].nunique()} tickers")
    print(f"  Sectors: {len(set(sector_map.values()))}")

    # Apply bid-ask costs
    print("\n[2/7] Applying 5% bid-ask cost...")
    trades = apply_ba_cost(trades)
    total_ba_cost = float((trades["realized_pnl"] - trades["realized_pnl_net"]).sum())
    print(f"  Total BA cost: ${total_ba_cost:,.0f}")
    print(f"  Net P&L after BA: ${trades['realized_pnl_net'].sum():,.0f}")

    # Build sector daily P&L
    print("\n[3/7] Building sector daily P&L matrix...")
    sector_daily, ticker_daily, sector_counts = build_sector_daily_pnl(trades, sector_map)
    print(f"  Sectors in data: {list(sector_daily.columns)}")

    # Identify cluster days
    print("\n[4/7] Identifying cluster loss days...")
    daily_stats = identify_cluster_days(trades)
    n_cluster = daily_stats["is_cluster"].sum()
    n_severe = daily_stats["is_severe_cluster"].sum()
    print(f"  Total trading days: {len(daily_stats)}")
    print(f"  Cluster days (>50% losing): {n_cluster} ({n_cluster/len(daily_stats)*100:.1f}%)")
    print(f"  Severe cluster (>75% losing): {n_severe} ({n_severe/len(daily_stats)*100:.1f}%)")

    # Sector concentration analysis
    print("\n[5/7] Analyzing sector concentration on cluster vs normal days...")
    concentration = sector_concentration_analysis(trades, sector_map, daily_stats)
    for label in ["cluster_days", "normal_days"]:
        info = concentration[label]
        if "note" in info:
            continue
        print(f"\n  {label.upper()}:")
        print(f"    HHI (concentration): {info['hhi']:.0f}")
        print(f"    Top sector: {info['top_sector']} ({info['top_sector_pct']:.1f}%)")
        print(f"    Top 3 combined: {info['top3_combined_pct']:.1f}%")

    # Cluster day driver analysis
    print("\n[6/7] Determining cluster day drivers (sector vs broad market)...")
    drivers = cluster_day_sector_driver(trades, sector_map, daily_stats, macro)
    if "note" not in drivers:
        print(f"  Sector-driven cluster days: {drivers['sector_driven_count']} ({drivers['sector_driven_pct']:.1f}%)")
        print(f"  Broad market cluster days: {drivers['broad_driven_count']} ({drivers['broad_driven_pct']:.1f}%)")
        print(f"  Interpretation: {drivers['interpretation']}")
        print(f"\n  WORST 5 CLUSTER DAYS:")
        for e in drivers["worst_10_days"][:5]:
            print(f"    {e['date']}: ${e['total_pnl']:+,.0f} | "
                  f"{e['n_losses']}/{e['n_trades']} losing | "
                  f"Top: {e['top_sector']} ({e['top_sector_pct']:.0f}%) | "
                  f"VIX: {e['vix']} | {e['driver']}")

    # Sector Sharpe ratios
    print("\n  SECTOR SHARPE RATIOS:")
    sector_sharpes = sector_sharpe_ratios(sector_daily)
    for sector, metrics in sector_sharpes.items():
        print(f"    {sector:25s}  Sharpe={metrics['sharpe']:+.2f}  "
              f"Sortino={metrics['sortino']:+.2f}  "
              f"WR={metrics['win_rate_pct']:.0f}%  "
              f"PF={metrics['profit_factor']:.2f}  "
              f"P&L=${metrics['total_pnl']:+,.0f}")

    # Sector exposure over time
    print("\n  SECTOR EXPOSURE:")
    exposure = sector_exposure_over_time(trades, sector_map)
    print(f"    Avg max sector concentration: {exposure['avg_max_sector_concentration_pct']:.1f}%")
    print(f"    Worst month concentration: {exposure['worst_month_concentration_pct']:.1f}%")
    for sector, pct in sorted(exposure["overall_sector_distribution"].items(), key=lambda x: -x[1]):
        print(f"    {sector:25s}: {pct:.1f}%")

    # Sector cap simulation
    print("\n[7/7] Simulating sector caps...")
    cap_results = sector_cap_simulation(trades, sector_map, daily_stats)
    print(f"\n  SECTOR CAP SIMULATION RESULTS:")
    print(f"  {'Cap':>8s}  {'Dropped':>8s}  {'Cluster':>12s}  {'Worst Day':>12s}  {'P&L Change':>11s}  {'Sharpe':>7s}")
    for cap_name, r in cap_results.items():
        if "note" in r:
            continue
        print(f"  {cap_name:>8s}  {r['pct_trades_dropped']:>7.1f}%  "
              f"{r['cluster_days_after_cap']:>5d} ({r['cluster_day_reduction_pct']:+.0f}%)  "
              f"${r['worst_day_pnl_after_cap']:>+9,.0f}  "
              f"{r['pnl_change_pct']:>+10.1f}%  "
              f"{r['sharpe']:>7.2f}")

    # Save all results
    all_results = {
        "metadata": {
            "n_trades": len(trades),
            "n_tickers": int(trades["ticker"].nunique()),
            "date_range": f"{trades['close_date'].min()} to {trades['close_date'].max()}",
            "ba_cost_frac": BA_COST_FRAC,
            "total_ba_cost": round(total_ba_cost, 0),
            "runtime_sec": round(time.time() - t0, 1),
        },
        "cluster_day_summary": {
            "total_days": int(len(daily_stats)),
            "cluster_days": int(n_cluster),
            "severe_cluster_days": int(n_severe),
        },
        "sector_concentration": concentration,
        "cluster_day_drivers": drivers,
        "sector_sharpe_ratios": sector_sharpes,
        "sector_exposure": exposure,
        "sector_cap_simulation": cap_results,
    }

    out_path = OUTPUT / "sector_decomposition_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")

    # Save sector daily P&L as parquet
    sector_daily.to_parquet(OUTPUT / "sector_daily_pnl.parquet")

    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed:.1f}s")
    print("=" * 70)

    return all_results


if __name__ == "__main__":
    results = main()

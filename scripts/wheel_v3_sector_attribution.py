#!/usr/bin/env python3
"""
wheel_v3_sector_attribution.py — Per-sector and per-ticker performance attribution.

Identifies which sectors/tickers carry the wheel strategy and which add risk.
Uses the validated best config (30-delta, 2d earnings, bear gate, equity brake).

Key questions:
1. Which sectors generate the most premium per trade?
2. Which sectors have the best/worst win rates?
3. Are any sectors net-negative (should be dropped)?
4. Would dropping weak sectors improve risk-adjusted returns?
"""
import sys
import time
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

ROOT = Path("/home/jupiter/Lvl3Quant")
OUT_DIR = ROOT / "output" / "wheel_v3_sector_attribution"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "scripts"))
from wheel_universe_v3_expand import load_all_data, compute_metrics, regime_analysis
from wheel_earnings_filter import download_earnings_dates, build_earnings_lookup
from wheel_v3_dd_protection import run_portfolio_v3_with_overlay

logging.basicConfig(
    format='%(asctime)s [SECTOR] %(levelname)s %(message)s',
    level=logging.INFO,
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger('SECTOR')


def analyze_trades_by_sector(trades, sector_map):
    """Break down trade-level PnL by sector."""
    trade_df = pd.DataFrame(trades)

    # Map tickers to sectors
    trade_df["sector"] = trade_df["ticker"].map(sector_map).fillna("Unknown")

    # Focus on trades with PnL (CSP outcomes, profit takes, assignments, etc.)
    pnl_trades = trade_df[trade_df["pnl"].notna()].copy() if "pnl" in trade_df.columns else pd.DataFrame()

    if pnl_trades.empty:
        log.warning("No PnL trades found!")
        return {}, {}

    # Per-sector aggregation
    sector_stats = {}
    for sector, grp in pnl_trades.groupby("sector"):
        wins = grp[grp["pnl"] > 0]
        losses = grp[grp["pnl"] <= 0]

        total_pnl = grp["pnl"].sum()
        n_trades = len(grp)
        wr = len(wins) / n_trades if n_trades > 0 else 0
        avg_win = wins["pnl"].mean() if len(wins) > 0 else 0
        avg_loss = losses["pnl"].mean() if len(losses) > 0 else 0
        pf = abs(wins["pnl"].sum() / losses["pnl"].sum()) if losses["pnl"].sum() != 0 else float('inf')

        sector_stats[sector] = {
            "n_trades": n_trades,
            "total_pnl": round(total_pnl, 2),
            "avg_pnl": round(total_pnl / n_trades, 4) if n_trades > 0 else 0,
            "win_rate": round(wr, 3),
            "avg_win": round(avg_win, 4),
            "avg_loss": round(avg_loss, 4),
            "profit_factor": round(pf, 2) if pf != float('inf') else 999,
            "pnl_per_trade_dollar": round(total_pnl / n_trades * 100, 2) if n_trades > 0 else 0,
        }

    # Per-ticker aggregation
    ticker_stats = {}
    for ticker, grp in pnl_trades.groupby("ticker"):
        wins = grp[grp["pnl"] > 0]
        losses = grp[grp["pnl"] <= 0]

        total_pnl = grp["pnl"].sum()
        n_trades = len(grp)
        wr = len(wins) / n_trades if n_trades > 0 else 0

        ticker_stats[ticker] = {
            "sector": sector_map.get(ticker, "Unknown"),
            "n_trades": n_trades,
            "total_pnl": round(total_pnl, 2),
            "avg_pnl": round(total_pnl / n_trades, 4) if n_trades > 0 else 0,
            "win_rate": round(wr, 3),
        }

    return sector_stats, ticker_stats


def run_sector_exclusion_test(prices, spy_regime, sector_map, earnings_lookup,
                               base_params, sectors_to_exclude):
    """Run backtest excluding specific sectors."""
    # Filter out excluded sector tickers
    excluded_tickers = {t for t, s in sector_map.items() if s in sectors_to_exclude}

    filtered_prices = prices[~prices["ticker"].isin(excluded_tickers)].copy()
    n_remaining = filtered_prices["ticker"].nunique()

    equity, trades, overlay_log = run_portfolio_v3_with_overlay(
        filtered_prices, spy_regime, sector_map,
        **base_params,
        overlay_type="eq_brake",
        eq_brake_lookback=60, eq_brake_threshold=0.03, eq_brake_scale=0.25,
    )

    metrics = compute_metrics(equity, 100_000)
    regime = regime_analysis(equity, spy_regime)

    return metrics, regime, n_remaining


def main():
    log.info("=" * 60)
    log.info("WHEEL V3 — SECTOR ATTRIBUTION ANALYSIS")
    log.info("=" * 60)

    # Load data
    log.info("Loading data...")
    prices, spy_regime, sector_map = load_all_data()
    earnings_raw = download_earnings_dates(prices["ticker"].unique())
    earnings_lookup = build_earnings_lookup(earnings_raw)

    base_params = dict(
        starting_cash=100_000, margin_cap=0.40, per_name_pct=0.03,
        put_delta=0.30, dte_target=14, profit_take=0.65,
        bear_mode="liq_csp_only", max_assignments_5d=3,
        max_share_positions=5, loss_cut_pct=-0.15,
        min_price=10.0, max_price=500.0,
        earnings_lookup=earnings_lookup, earnings_buffer_days=2,
    )

    # Run best config with trade logging
    log.info("Running best config (60d/3%/25% brake)...")
    t0 = time.time()
    equity, trades, overlay_log = run_portfolio_v3_with_overlay(
        prices, spy_regime, sector_map,
        **base_params,
        overlay_type="eq_brake",
        eq_brake_lookback=60, eq_brake_threshold=0.03, eq_brake_scale=0.25,
    )
    elapsed = time.time() - t0
    metrics = compute_metrics(equity, 100_000)
    log.info(f"  Full portfolio: CAGR {metrics['cagr_pct']}%, Sharpe {metrics['sharpe']}, "
             f"MaxDD {metrics['max_dd_pct']}% ({elapsed:.0f}s)")

    # Sector attribution
    log.info("\nAnalyzing per-sector performance...")
    sector_stats, ticker_stats = analyze_trades_by_sector(trades, sector_map)

    # Sort sectors by total PnL
    sorted_sectors = sorted(sector_stats.items(), key=lambda x: x[1]["total_pnl"], reverse=True)

    log.info("\n" + "=" * 80)
    log.info("PER-SECTOR PERFORMANCE")
    log.info("=" * 80)
    header = f"{'Sector':<20} {'Trades':>7} {'Total PnL':>10} {'Avg PnL':>9} {'WR':>6} {'PF':>6} {'$/Trade':>9}"
    log.info(header)
    log.info("-" * len(header))

    positive_sectors = []
    negative_sectors = []

    for sector, stats in sorted_sectors:
        marker = "  ❌" if stats["total_pnl"] < 0 else ""
        log.info(f"{sector:<20} {stats['n_trades']:>7} {stats['total_pnl']:>10.1f} "
                 f"{stats['avg_pnl']:>9.4f} {stats['win_rate']:>6.1%} "
                 f"{stats['profit_factor']:>6.2f} {stats['pnl_per_trade_dollar']:>8.1f}{marker}")

        if stats["total_pnl"] > 0:
            positive_sectors.append(sector)
        else:
            negative_sectors.append(sector)

    # Top/bottom tickers
    sorted_tickers = sorted(ticker_stats.items(), key=lambda x: x[1]["total_pnl"], reverse=True)

    log.info("\n" + "=" * 80)
    log.info("TOP 15 TICKERS (by total PnL)")
    log.info("=" * 80)
    for ticker, stats in sorted_tickers[:15]:
        log.info(f"  {ticker:<8} ({stats['sector']:<15}) — PnL: {stats['total_pnl']:>8.1f}, "
                 f"Trades: {stats['n_trades']:>4}, WR: {stats['win_rate']:.1%}, "
                 f"Avg: {stats['avg_pnl']:.4f}")

    log.info("\n" + "=" * 80)
    log.info("BOTTOM 15 TICKERS (by total PnL)")
    log.info("=" * 80)
    for ticker, stats in sorted_tickers[-15:]:
        log.info(f"  {ticker:<8} ({stats['sector']:<15}) — PnL: {stats['total_pnl']:>8.1f}, "
                 f"Trades: {stats['n_trades']:>4}, WR: {stats['win_rate']:.1%}, "
                 f"Avg: {stats['avg_pnl']:.4f}")

    # === SECTOR EXCLUSION TESTS ===
    if negative_sectors:
        log.info("\n" + "=" * 80)
        log.info("SECTOR EXCLUSION TESTS")
        log.info("=" * 80)
        log.info(f"Negative sectors to test excluding: {negative_sectors}")

        # Test 1: Drop all negative sectors
        log.info(f"\nTest 1: Drop ALL negative sectors ({len(negative_sectors)})...")
        t0 = time.time()
        m1, r1, n1 = run_sector_exclusion_test(
            prices, spy_regime, sector_map, earnings_lookup,
            base_params, set(negative_sectors)
        )
        log.info(f"  {n1} tickers remaining → CAGR {m1['cagr_pct']}%, Sharpe {m1['sharpe']}, "
                 f"MaxDD {m1['max_dd_pct']}%, Calmar {m1['calmar']} ({time.time()-t0:.0f}s)")

        # Test 2: Drop bottom 3 sectors only
        bottom3 = [s for s, _ in sorted_sectors[-3:] if sector_stats[s]["total_pnl"] < 0]
        if bottom3 and len(bottom3) >= 2:
            log.info(f"\nTest 2: Drop bottom 3 sectors ({bottom3})...")
            t0 = time.time()
            m2, r2, n2 = run_sector_exclusion_test(
                prices, spy_regime, sector_map, earnings_lookup,
                base_params, set(bottom3)
            )
            log.info(f"  {n2} tickers remaining → CAGR {m2['cagr_pct']}%, Sharpe {m2['sharpe']}, "
                     f"MaxDD {m2['max_dd_pct']}%, Calmar {m2['calmar']} ({time.time()-t0:.0f}s)")

    # === TICKER EXCLUSION TEST ===
    # Drop bottom 20% of tickers by PnL
    n_tickers = len(sorted_tickers)
    bottom_20pct = [t for t, _ in sorted_tickers[-int(n_tickers * 0.2):]]
    bottom_20pct_set = set(bottom_20pct)

    log.info(f"\nTest 3: Drop bottom 20% of tickers ({len(bottom_20pct)} tickers)...")
    filtered_prices = prices[~prices["ticker"].isin(bottom_20pct_set)].copy()
    t0 = time.time()
    eq3, trades3, _ = run_portfolio_v3_with_overlay(
        filtered_prices, spy_regime, sector_map,
        **base_params,
        overlay_type="eq_brake",
        eq_brake_lookback=60, eq_brake_threshold=0.03, eq_brake_scale=0.25,
    )
    m3 = compute_metrics(eq3, 100_000)
    log.info(f"  {filtered_prices['ticker'].nunique()} tickers remaining → CAGR {m3['cagr_pct']}%, "
             f"Sharpe {m3['sharpe']}, MaxDD {m3['max_dd_pct']}%, Calmar {m3['calmar']} "
             f"({time.time()-t0:.0f}s)")

    # === CONCENTRATION TEST ===
    # What if we ONLY use the top 50% of tickers?
    top_50pct = [t for t, _ in sorted_tickers[:int(n_tickers * 0.5)]]
    top_50pct_set = set(top_50pct)

    log.info(f"\nTest 4: TOP 50% tickers ONLY ({len(top_50pct)} tickers)...")
    filtered_prices = prices[prices["ticker"].isin(top_50pct_set)].copy()
    t0 = time.time()
    eq4, trades4, _ = run_portfolio_v3_with_overlay(
        filtered_prices, spy_regime, sector_map,
        **base_params,
        overlay_type="eq_brake",
        eq_brake_lookback=60, eq_brake_threshold=0.03, eq_brake_scale=0.25,
    )
    m4 = compute_metrics(eq4, 100_000)
    log.info(f"  {filtered_prices['ticker'].nunique()} tickers remaining → CAGR {m4['cagr_pct']}%, "
             f"Sharpe {m4['sharpe']}, MaxDD {m4['max_dd_pct']}%, Calmar {m4['calmar']} "
             f"({time.time()-t0:.0f}s)")

    # === FINAL SUMMARY ===
    log.info("\n" + "=" * 80)
    log.info("FINAL COMPARISON")
    log.info("=" * 80)

    comparisons = [
        ("Full 230 tickers + brake", metrics),
    ]
    if negative_sectors:
        comparisons.append((f"Drop {len(negative_sectors)} neg sectors", m1))
        if bottom3 and len(bottom3) >= 2:
            comparisons.append((f"Drop bottom 3 sectors", m2))
    comparisons.append((f"Drop bottom 20% tickers", m3))
    comparisons.append((f"Top 50% tickers only", m4))

    header = f"{'Config':<35} {'CAGR':>6} {'Sharpe':>7} {'MaxDD':>7} {'Calmar':>7}"
    log.info(header)
    log.info("-" * len(header))
    for name, m in comparisons:
        log.info(f"{name:<35} {m['cagr_pct']:>5.1f}% {m['sharpe']:>7.2f} "
                 f"{m['max_dd_pct']:>6.1f}% {m['calmar']:>7.2f}")

    # VERDICT
    log.info("\n" + "=" * 80)
    log.info("VERDICT")
    log.info("=" * 80)

    best_sharpe = max(comparisons, key=lambda x: x[1].get("sharpe", 0))
    best_calmar = max(comparisons, key=lambda x: x[1].get("calmar", 0))

    log.info(f"Best Sharpe: {best_sharpe[0]} ({best_sharpe[1]['sharpe']})")
    log.info(f"Best Calmar: {best_calmar[0]} ({best_calmar[1]['calmar']})")

    if best_sharpe[1]["sharpe"] > metrics["sharpe"] * 1.05:
        log.info("→ Sector/ticker filtering IMPROVES the strategy. Consider adopting.")
    else:
        log.info("→ Broader diversification is BETTER. Keep the full universe.")

    # Save results
    results = {
        "sector_stats": sector_stats,
        "comparisons": {name: m for name, m in comparisons},
        "negative_sectors": negative_sectors,
        "top_tickers": [{"ticker": t, **s} for t, s in sorted_tickers[:30]],
        "bottom_tickers": [{"ticker": t, **s} for t, s in sorted_tickers[-30:]],
    }

    with open(OUT_DIR / "sector_attribution.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Save sector stats as CSV for easy viewing
    sector_df = pd.DataFrame([{"sector": s, **v} for s, v in sorted_sectors])
    sector_df.to_csv(OUT_DIR / "sector_performance.csv", index=False)

    log.info(f"\nResults saved to {OUT_DIR}")


if __name__ == "__main__":
    main()

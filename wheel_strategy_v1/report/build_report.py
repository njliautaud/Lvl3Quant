"""
build_report.py — Pick 3 representative configs from the Pareto front and
write a markdown report with per-tier annualized yield %, max DD %, worst
month, Sortino, average DTE, average delta, total trades, assignment count,
and 5 example trades.
"""
from __future__ import annotations
import sys
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ga.chromosome import decode
from backtest.wheel_engine import WheelConfig, run_wheel

RESULTS = ROOT / "results"
CACHE = ROOT / "data" / "cache"


def _load(smoke: bool):
    sfx = "_smoke" if smoke else ""
    return (
        pd.read_parquet(CACHE / f"prices{sfx}.parquet"),
        pd.read_parquet(CACHE / f"iv_features{sfx}.parquet"),
        pd.read_parquet(CACHE / f"macro{sfx}.parquet"),
        pd.read_parquet(CACHE / f"fundamentals{sfx}.parquet"),
        pd.read_parquet(CACHE / "universe.parquet"),
    )


def _pick_tiers(pareto: pd.DataFrame) -> dict:
    p = pareto.copy().sort_values("ann_premium_yield").reset_index(drop=True)
    if p.empty:
        return {}
    # Conservative = lowest DD
    conservative = p.loc[p["max_dd_pct"].idxmin()]
    # Aggressive = highest yield
    aggressive = p.loc[p["ann_premium_yield"].idxmax()]
    # Balanced = best fitness in between
    middle = p[(p["ann_premium_yield"] >= conservative["ann_premium_yield"]) &
               (p["ann_premium_yield"] <= aggressive["ann_premium_yield"])]
    if middle.empty:
        balanced = p.loc[p["fitness"].idxmax()]
    else:
        balanced = middle.loc[middle["fitness"].idxmax()]
    return {"Conservative": conservative, "Balanced": balanced, "Aggressive": aggressive}


def _replay(row, prices, iv, macro, fundamentals, universe):
    cfg_cols = ["put_delta_target","call_delta_target","dte_min","dte_max",
                "profit_take_pct","roll_dte_trigger","max_concurrent_names",
                "sector_cap_pct","vix_max_gate","naaim_min_gate","fund_score_floor"]
    cfg = {c: row[c] for c in cfg_cols}
    for k in ("dte_min","dte_max","roll_dte_trigger","max_concurrent_names"):
        cfg[k] = int(cfg[k])
    return run_wheel(WheelConfig(**cfg), prices, iv, macro, fundamentals, universe)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or ("smoke" if args.smoke else "full")

    pareto_path = RESULTS / f"pareto_{tag}.parquet"
    if not pareto_path.exists():
        print(f"[report] no pareto file {pareto_path}", file=sys.stderr); sys.exit(2)
    pareto = pd.read_parquet(pareto_path)
    tiers = _pick_tiers(pareto)
    if not tiers:
        print("[report] empty Pareto", file=sys.stderr); sys.exit(3)

    prices, iv, macro, fundamentals, universe = _load(args.smoke)

    md_lines = [f"# Wheel Strategy v1 — Report ({tag})", ""]
    md_lines.append(f"Pareto front size: {len(pareto)}")
    md_lines.append("")
    md_lines.append("| Tier | Yield % | MaxDD % | Worst Month % | Sortino | Trades | Assignments | Avg DTE | Avg Delta |")
    md_lines.append("|------|---------|---------|---------------|---------|--------|-------------|---------|-----------|")

    tier_details = {}
    for tier_name, row in tiers.items():
        result = _replay(row, prices, iv, macro, fundamentals, universe)
        ledger = result["ledger"]
        if not ledger.empty:
            avg_dte = float(ledger["dte_open"].mean())
            avg_delta = float(ledger["delta_open"].mean())
            assignments = int(result.get("assignment_count", 0))
        else:
            avg_dte = float("nan"); avg_delta = float("nan"); assignments = 0
        md_lines.append(
            f"| {tier_name} | {row['ann_premium_yield']*100:.2f} | "
            f"{row['max_dd_pct']:.2f} | {row['worst_month_pct']:.2f} | "
            f"{row['sortino']:.2f} | {int(row['n_trades'])} | {assignments} | "
            f"{avg_dte:.1f} | {avg_delta:.3f} |"
        )
        tier_details[tier_name] = (row, ledger)

    for tier_name, (row, ledger) in tier_details.items():
        md_lines += ["", f"## {tier_name} — config", "", "```"]
        for k in ("put_delta_target","call_delta_target","dte_min","dte_max",
                  "profit_take_pct","roll_dte_trigger","max_concurrent_names",
                  "sector_cap_pct","vix_max_gate","naaim_min_gate","fund_score_floor"):
            md_lines.append(f"{k} = {row[k]}")
        md_lines.append("```")
        md_lines += ["", f"### {tier_name} — 5 example trades", ""]
        if ledger.empty:
            md_lines.append("(no trades)")
        else:
            top = ledger.sort_values("realized_pnl", ascending=False).head(5)
            md_lines.append("| Open | Close | Ticker | Kind | Strike | DTE | Delta | PnL |")
            md_lines.append("|------|-------|--------|------|--------|-----|-------|-----|")
            for _, t in top.iterrows():
                md_lines.append(
                    f"| {pd.Timestamp(t['open_date']).date()} | {pd.Timestamp(t['close_date']).date()} | "
                    f"{t['ticker']} | {t['kind']} | {t['strike']:.2f} | {int(t['dte_open'])} | "
                    f"{t['delta_open']:.3f} | ${t['realized_pnl']:.2f} |"
                )

    out = RESULTS / f"report_{tag}.md"
    out.write_text("\n".join(md_lines))
    print(f"[report] wrote {out}")


if __name__ == "__main__":
    main()

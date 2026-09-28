"""
Combine the two sectors that passed the Calmar ≥ 1.0 floor in v3
(Energy, Industrials) into a single long-short book, then evaluate
walk-forward vs SPY 1x / 1.5x.

Equal-weight allocation across the two sector books, daily.
Rebalance happens inside each sector picker; no extra turnover here.
"""
from __future__ import annotations
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT / "strategy/macro_picker"))

from walk_forward import walk_forward  # type: ignore
from sector_picker_v3 import (  # type: ignore
    load_panel, load_spy_and_funding, build_target,
    portfolio_returns_for_sector, FEATURE_POOL,
)

OUT = ROOT / "research/findings/combined_energy_industrials.json"
SUMMARY_MD = ROOT / "research/findings/combined_energy_industrials.md"

SECTORS = ["Energy", "Industrials"]


def main():
    panel = load_panel()
    spy_ret, fund = load_spy_and_funding()
    panel = build_target(panel)

    sector_series = {}
    for s in SECTORS:
        sp = panel[panel["sector"] == s].copy()
        feats_present = [f for f in FEATURE_POOL if f in sp.columns]
        print(f"running {s} ...")
        daily_pnl, _ = portfolio_returns_for_sector(sp, feats_present)
        if daily_pnl.empty:
            print(f"  {s}: no PnL series, skipping")
            continue
        sector_series[s] = daily_pnl
        print(f"  {s}: {len(daily_pnl)} daily-PnL rows, "
              f"range {daily_pnl.index.min()} → {daily_pnl.index.max()}")

    if len(sector_series) < 1:
        print("no sectors produced PnL; aborting")
        return

    # equal-weight on each available day
    combined = pd.concat(sector_series.values(), axis=1, keys=sector_series.keys())
    combined["combined"] = combined.mean(axis=1, skipna=True)
    combined_pnl = combined["combined"].dropna()
    print(f"\ncombined book: {len(combined_pnl)} rows, "
          f"{combined_pnl.index.min()} → {combined_pnl.index.max()}")

    # walk-forward vs SPY 1x / 1.5x
    wf = walk_forward(
        daily_returns=combined_pnl,
        spy_returns=spy_ret,
        fund_rate=fund,
        train_months=36, oot_months=12, step_months=6,
    )

    out = {
        "sectors": SECTORS,
        "n_wf_folds": len(wf.per_fold),
        "verdict": wf.verdict,
        "summary": wf.summary,
        "per_fold": wf.per_fold,
    }
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {OUT}")

    # markdown summary
    med = {k: v.get("median", float("nan")) for k, v in wf.summary.items()}
    lines = [
        "# Combined Energy + Industrials — walk-forward vs SPY",
        "",
        f"**Folds**: {len(wf.per_fold)}   **Verdict**: {wf.verdict}",
        "",
        "## Summary medians",
        "",
        "| Metric | Combined book | SPY 1× | SPY 1.5× | SPY 2× |",
        "|---|---|---|---|---|",
    ]

    # collect SPY medians from per-fold dicts
    def med_field(field):
        vals = [f.get(field, float("nan")) for f in wf.per_fold]
        s = pd.Series(vals, dtype=float).dropna()
        return float(s.median()) if len(s) else float("nan")

    lines.append(
        f"| Sharpe | {med.get('sharpe', float('nan')):.2f} | "
        f"{med_field('spy_1x_sharpe'):.2f} | {med_field('spy_15x_sharpe'):.2f} | "
        f"{med_field('spy_2x_sharpe'):.2f} |"
    )
    lines.append(
        f"| CAGR | {med.get('cagr', float('nan')):.3f} | "
        f"{med_field('spy_1x_cagr'):.3f} | {med_field('spy_15x_cagr'):.3f} | "
        f"{med_field('spy_2x_cagr'):.3f} |"
    )
    lines.append(
        f"| Calmar | {med.get('calmar', float('nan')):.2f} | - | - | - |"
    )
    lines.append(
        f"| MaxDD | {med.get('max_dd', float('nan')):.3f} | - | - | - |"
    )

    lines.append("\n## Per-fold detail\n")
    lines.append("| OOT start | Sharpe | CAGR | Calmar | MaxDD | SPY 1× Sharpe | SPY 1.5× Sharpe |")
    lines.append("|---|---|---|---|---|---|---|")
    for f in wf.per_fold:
        lines.append(
            f"| {f['oot_start']} | {f['sharpe']:.2f} | {f['cagr']:.3f} | "
            f"{f['calmar']:.2f} | {f['max_dd']:.3f} | "
            f"{f['spy_1x_sharpe']:.2f} | {f['spy_15x_sharpe']:.2f} |"
        )

    SUMMARY_MD.write_text("\n".join(lines))
    print(f"wrote {SUMMARY_MD}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

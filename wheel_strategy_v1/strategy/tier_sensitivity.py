"""
tier_sensitivity.py — HC #556 R4 deployment-gate sensitivity sweep.

For the two deployable tiers (Tier1_Conservative, Tier2_Balanced — the v7
survivors) run the wheel engine across a grid of:

    iv_mult   ∈ {0.80, 0.90, 1.00, 1.10, 1.20}    (±20% IV bump)
    slip_mult ∈ {0.50, 1.00, 1.50}                 (±50% spread bump)

That is 5 × 3 = 15 cells per tier × 2 tiers = 30 backtests.

We report REALIZED-CASH metrics (CAGR, Sharpe, MaxDD, final equity) at each
cell so the user can see how robust the deployable ladder is to vol / fill
assumption error. HC #556 R4 requires this before either tier ships.

Implementation notes:
- IV bump is applied to the iv_features_real_blend `sigma` column in-memory
  (we do NOT mutate the parquet on disk).
- Slippage bump scales wheel_engine.SLIPPAGE_FRAC and SLIPPAGE_MIN_TICKS by
  the multiplier. Skew stays on (USE_SKEW=True) — only premium and fill
  cost are perturbed.
- Same regime overlay, same FullWheel, same $100k.

Usage:
    cd /home/jupiter/Lvl3Quant/wheel_strategy_v1
    python3 -m strategy.tier_sensitivity \
        --start 2020-01-01 --end 2025-12-31 --capital 100000 \
        --out results/tier_sensitivity_v1
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT))

from backtest import wheel_engine as _we  # noqa: E402
from backtest.wheel_engine import run_wheel  # noqa: E402
from strategy.tiers import all_tiers_full_wheel, TierSpec  # noqa: E402
from strategy.regime_overlay import build_regime, apply_regime_gate  # noqa: E402
from strategy.tier_runner import (
    _load_inputs,
    _apply_iv_rank_floor,
    compute_metrics,
)  # noqa: E402


DEPLOYABLE_TIER_NAMES = ["Tier1_Conservative_FW", "Tier2_Balanced_FW"]
IV_MULTS = [0.80, 0.90, 1.00, 1.10, 1.20]
SLIP_MULTS = [0.50, 1.00, 1.50]


def _bump_iv(iv_df: pd.DataFrame, mult: float) -> pd.DataFrame:
    """Return a copy of iv_df with `sigma` (and the IV-rank-driving cols if
    any) scaled by mult. Other columns are unchanged."""
    df = iv_df.copy()
    if "sigma" in df.columns:
        df["sigma"] = df["sigma"].astype(float) * mult
    # iv_rank is a relative quantile so bumping sigma uniformly doesn't
    # change the rank — leave alone.
    return df


def _patch_slippage(mult: float):
    """Mutate the engine's slippage constants and return originals for restore."""
    orig = (_we.SLIPPAGE_FRAC, _we.SLIPPAGE_MIN_TICKS)
    _we.SLIPPAGE_FRAC = float(orig[0]) * mult
    _we.SLIPPAGE_MIN_TICKS = float(orig[1]) * mult
    return orig


def _restore_slippage(orig):
    _we.SLIPPAGE_FRAC, _we.SLIPPAGE_MIN_TICKS = orig


def _run_one_cell(tier: TierSpec, data: dict, iv_mult: float, slip_mult: float,
                  start: str, end: str, capital: float) -> dict:
    iv_bumped = _bump_iv(data["iv"], iv_mult)
    iv_bumped = _apply_iv_rank_floor(iv_bumped, tier.iv_rank_floor)

    tickers = tier.universe_filter(
        data["universe"], data["fundamentals"], iv_bumped, data["prices"]
    )
    px = data["prices"][data["prices"]["ticker"].isin(tickers)].copy()
    iv_local = iv_bumped[iv_bumped["ticker"].isin(tickers)].copy()

    orig = _patch_slippage(slip_mult)
    try:
        result = run_wheel(
            cfg=tier.wheel_cfg,
            prices=px,
            iv=iv_local,
            macro=data["macro"],
            fundamentals=data["fundamentals"],
            universe=data["universe"],
            starting_cash=capital,
            start=start, end=end,
            verbose=False,
        )
        metrics = compute_metrics(result, capital)
    finally:
        _restore_slippage(orig)

    return {
        "tier": tier.name,
        "iv_mult": iv_mult,
        "slip_mult": slip_mult,
        "n_universe": len(tickers),
        **metrics,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--capital", type=float, default=100_000.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--regime-overlay", action="store_true", default=True)
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else RESULTS / "tier_sensitivity_v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    # always real-IV blend for honesty
    data = _load_inputs(modeled=False, smoke=False, real_iv=True)
    iv_path = data["paths"]["iv"]
    print(f"[sens] IV source: {iv_path}")

    if args.regime_overlay:
        regime = build_regime()
        data["macro"] = apply_regime_gate(data["macro"], regime, vix_force_gate=999.0)
        ro_share = (data["macro"]["vix"] >= 999.0).mean() if "vix" in data["macro"].columns else 0.0
        print(f"[sens] regime gate: {ro_share:.1%} of days masked as risk-off")

    all_tiers = all_tiers_full_wheel()
    tiers = [t for t in all_tiers if t.name in DEPLOYABLE_TIER_NAMES]
    print(f"[sens] running {len(tiers)} tiers × {len(IV_MULTS)} IV mults × "
          f"{len(SLIP_MULTS)} slip mults = {len(tiers) * len(IV_MULTS) * len(SLIP_MULTS)} cells")

    rows: List[dict] = []
    for tier in tiers:
        for iv_m in IV_MULTS:
            for sl_m in SLIP_MULTS:
                tag = f"{tier.name} iv×{iv_m:.2f} slip×{sl_m:.2f}"
                print(f"  [run] {tag}")
                try:
                    row = _run_one_cell(tier, data, iv_m, sl_m,
                                        args.start, args.end, args.capital)
                    rows.append(row)
                    print(f"    -> rCAGR={row['realized_cagr']*100:.2f}%  "
                          f"rSharpe={row['realized_sharpe']:.2f}  "
                          f"rDD={row['realized_max_dd']*100:.2f}%")
                except Exception as e:
                    print(f"    FAILED: {e}")
                    import traceback; traceback.print_exc()
                    rows.append({"tier": tier.name, "iv_mult": iv_m,
                                 "slip_mult": sl_m, "error": str(e)})

    df = pd.DataFrame(rows)
    df.to_parquet(out_dir / "sensitivity_results.parquet", index=False)
    df.to_csv(out_dir / "sensitivity_results.csv", index=False)

    # ---- markdown summary ----
    lines = []
    lines.append("# Wheel Tier Sensitivity Sweep — HC #556 R4\n")
    lines.append(f"Window: **{args.start} → {args.end}**, ${args.capital:,.0f} per cell.  "
                 f"FullWheel + regime overlay.  Real-IV blend, skew on.\n")
    lines.append(f"IV multiplier ∈ {IV_MULTS}.  Slippage multiplier ∈ {SLIP_MULTS}.\n\n")

    for tier_name in DEPLOYABLE_TIER_NAMES:
        lines.append(f"\n## {tier_name}\n\n")
        sub = df[df["tier"] == tier_name]
        if sub.empty or "realized_cagr" not in sub.columns:
            lines.append("_no data_\n")
            continue
        # CAGR matrix
        lines.append("### Realized CAGR % (rows = IV mult, cols = slip mult)\n\n")
        pivot = sub.pivot(index="iv_mult", columns="slip_mult", values="realized_cagr") * 100
        lines.append(pivot.round(2).to_markdown())
        lines.append("\n\n### Realized Sharpe\n\n")
        pivot = sub.pivot(index="iv_mult", columns="slip_mult", values="realized_sharpe")
        lines.append(pivot.round(2).to_markdown())
        lines.append("\n\n### Realized Max DD %\n\n")
        pivot = sub.pivot(index="iv_mult", columns="slip_mult", values="realized_max_dd") * 100
        lines.append(pivot.round(2).to_markdown())
        lines.append("\n\n")

    (out_dir / "sensitivity_report.md").write_text("".join(lines))

    meta = {
        "start": args.start, "end": args.end, "capital": args.capital,
        "iv_mults": IV_MULTS, "slip_mults": SLIP_MULTS,
        "tiers": DEPLOYABLE_TIER_NAMES,
        "engine_flags": {
            "use_skew": bool(getattr(_we, "USE_SKEW", False)),
            "use_slippage": bool(getattr(_we, "USE_SLIPPAGE", False)),
            "slippage_frac_baseline": 0.025,
            "slippage_min_ticks_baseline": 0.03,
        },
        "iv_source": str(iv_path),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"\n[sens] DONE. {len(rows)} cells run. See {out_dir}/sensitivity_report.md")


if __name__ == "__main__":
    main()

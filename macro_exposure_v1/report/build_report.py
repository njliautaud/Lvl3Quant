"""
build_report.py — Build the tier report from a completed GA run.

Picks 3 tier configs from results/<tag>/all_evals.parquet:
  Conservative : lowest max_DD among configs with CAGR > 5% and turnover <= 12/yr
  Balanced     : highest fitness
  Aggressive   : highest CAGR among configs with max_DD <= 35%

Outputs:
  results/<tag>/report.md
  results/<tag>/equity_<tier>.png  (if matplotlib available)
"""
from __future__ import annotations
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ga.chromosome import FEATURE_WEIGHT_NAMES
from ga.run_ga import build_feature_panel, build_allocation, cadence_dates
from backtest.exposure_engine import run_exposure_backtest


def _row_to_cfg(row: pd.Series) -> dict:
    return dict(
        feature_weights={k: float(row[k]) for k in FEATURE_WEIGHT_NAMES if k in row},
        long_threshold=float(row["long_threshold"]),
        short_threshold=float(row["short_threshold"]),
        flat_band_width=float(row["flat_band_width"]),
        basket_weights={"SPY": float(row["bw_spy"]),
                        "QQQ": float(row["bw_qqq"]),
                        "IWM": float(row["bw_iwm"])},
        max_leverage=float(row["max_leverage"]),
        allow_short=bool(int(row["allow_short"])),
        cadence=str(row["cadence"]),
        long_strength=float(row["long_strength"]),
        short_strength=float(row["short_strength"]),
    )


def _pick_tiers(df: pd.DataFrame) -> dict:
    out = {}
    # Conservative
    cons_pool = df[(df["cagr"] > 0.05) & (df["turnover_per_year"] <= 12)]
    if not cons_pool.empty:
        out["Conservative"] = cons_pool.sort_values("max_dd_pct").iloc[0]
    else:
        out["Conservative"] = df.sort_values("max_dd_pct").iloc[0]

    # Balanced
    out["Balanced"] = df.sort_values("fitness", ascending=False).iloc[0]

    # Aggressive
    agg_pool = df[df["max_dd_pct"] <= 35]
    if not agg_pool.empty:
        out["Aggressive"] = agg_pool.sort_values("cagr", ascending=False).iloc[0]
    else:
        out["Aggressive"] = df.sort_values("cagr", ascending=False).iloc[0]
    return out


def _top_features(cfg: dict, k: int = 5):
    items = sorted(cfg["feature_weights"].items(), key=lambda kv: abs(kv[1]), reverse=True)
    return items[:k]


def _save_equity_png(equity: pd.Series, path: Path, title: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 4))
        equity.plot(ax=ax)
        ax.set_title(title)
        ax.set_ylabel("Equity ($)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
    except Exception as e:
        print(f"[report] png save failed ({path.name}): {e}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or ("smoke" if args.smoke else "full")

    res_dir = ROOT / "results" / tag
    df_path = res_dir / "all_evals.parquet"
    if not df_path.exists():
        print(f"[report] missing {df_path} — run GA first", flush=True)
        return 1
    df = pd.read_parquet(df_path).dropna(subset=["fitness"])
    if df.empty:
        print("[report] no evals", flush=True)
        return 1

    features, prices = build_feature_panel(args.smoke)
    tiers = _pick_tiers(df)

    lines = []
    lines.append(f"# Macro-Exposure GA Report ({tag})\n")
    lines.append(f"Evals: **{len(df)}**\n")
    lines.append(f"Backtest span: **{features.index.min().date()} → {features.index.max().date()}**\n")
    lines.append("\nTiers picked from the GA Pareto/eval pool.\n\n")

    for name, row in tiers.items():
        cfg = _row_to_cfg(row)
        alloc = build_allocation(features, cfg)
        rebal = cadence_dates(features.index, cfg["cadence"])
        bt = run_exposure_backtest(
            allocation=alloc,
            basket_w=cfg["basket_weights"],
            prices=prices,
            rebalance_dates=rebal,
            allow_short=cfg["allow_short"],
            max_leverage=cfg["max_leverage"],
        )
        m = bt.metrics
        top_feats = _top_features(cfg)
        png = res_dir / f"equity_{name.lower()}.png"
        _save_equity_png(bt.equity_curve, png, f"{name} equity")

        lines.append(f"## {name}\n")
        lines.append(f"- CAGR: **{m['cagr']*100:.2f}%**")
        lines.append(f"- Max drawdown: **{m['max_dd_pct']:.2f}%**")
        lines.append(f"- Worst month: **{m['worst_month_pct']:.2f}%**")
        lines.append(f"- Sortino: **{m['sortino']:.2f}**")
        lines.append(f"- Sharpe: **{m['sharpe']:.2f}**")
        lines.append(f"- Avg leverage: **{m['avg_leverage']:.2f}x**")
        lines.append(f"- Turnover/year: **{m['turnover_per_year']:.1f}**")
        lines.append(f"- Time long / flat / short: **{m['pct_long']*100:.0f}% / {m['pct_flat']*100:.0f}% / {m['pct_short']*100:.0f}%**")
        lines.append(f"- Basket: SPY {cfg['basket_weights']['SPY']*100:.0f}% / "
                     f"QQQ {cfg['basket_weights']['QQQ']*100:.0f}% / "
                     f"IWM {cfg['basket_weights']['IWM']*100:.0f}%")
        lines.append(f"- Cadence: **{cfg['cadence']}**, allow_short: **{cfg['allow_short']}**, "
                     f"max_leverage: **{cfg['max_leverage']}x**")
        lines.append(f"- Long/short strength: **{cfg['long_strength']}x / {cfg['short_strength']}x**")
        lines.append(f"- Long/short thresholds: **{cfg['long_threshold']:.2f} / {cfg['short_threshold']:.2f}** "
                     f"(flat band {cfg['flat_band_width']:.2f})")
        lines.append(f"- Top features (|weight|):")
        for k, v in top_feats:
            lines.append(f"    - {k}: {v:+.2f}")
        lines.append(f"\n![{name} equity](equity_{name.lower()}.png)\n")

    out_md = res_dir / "report.md"
    out_md.write_text("\n".join(lines))
    print(f"[report] wrote {out_md}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

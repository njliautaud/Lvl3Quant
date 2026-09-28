#!/usr/bin/env python3
"""
render_overlay_v1_charts.py — HC #544 R7/R8

Re-renders the macro_exposure_v1 Balanced + Defensive-Overlay-v1 walk-forward
result WITH:
  - SPY buy-and-hold baseline overlaid on every chart
  - PNG chart pack saved for Discord attach
  - FMP archive used as the SPY price source (R1 wire-in test)

Inputs (reuses code already validated by run_wf_defensive.py):
  - BALANCED_CFG from run_wf_defensive.py (imported)
  - build_feature_panel, build_allocation, cadence_dates (ga.run_ga)
  - apply_defensive_overlay, derive_defensive_flags (backtest.defensive_overlay)
  - run_exposure_backtest (backtest.exposure_engine)

Outputs (saved to walkforward/results/charts_overlay_v1/):
  - 01_equity_curve.png      strategy vs SPY, log scale
  - 02_drawdown.png          strategy vs SPY drawdown overlay
  - 03_rolling_sortino.png   1y rolling Sortino, strategy vs SPY
  - 04_monthly_heatmap.png   strategy monthly returns heatmap
  - 05_fold_cagr.png         per-fold CAGR bar, strategy vs SPY
  - 06_regime.png            regime-stratified Sortino/CAGR bars
  - 07_leverage.png          avg leverage per fold

Run:
  cd /home/jupiter/Lvl3Quant/macro_exposure_v1
  python3 walkforward/render_overlay_v1_charts.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Path resolution — mimic run_wf_defensive.py
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from ga.run_ga import build_feature_panel, build_allocation, cadence_dates  # noqa: E402
from backtest.exposure_engine import run_exposure_backtest                  # noqa: E402
from backtest.defensive_overlay import (                                    # noqa: E402
    apply_defensive_overlay, derive_defensive_flags,
)
# Reuse the locked-in chromosome
from walkforward.run_wf_defensive import BALANCED_CFG  # noqa: E402


# --- FMP archive (HC #544 R1) -------------------------------------------------
FMP_ROOT = Path("/home/jupiter/teleclaude-main/data/fmp_archive")


def load_spy_from_fmp() -> pd.Series:
    """Load SPY daily close from FMP archive. Returns a tz-naive datetime-indexed Series."""
    path = FMP_ROOT / "prices" / "SPY_daily.json"
    if not path.exists():
        raise FileNotFoundError(f"FMP SPY not found at {path}")
    rows = json.load(open(path))
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    return df["close"].astype(float)


# --- Metric helpers (mirrors _annualize in run_wf_defensive.py) ---------------

def annualize(daily_ret: pd.Series) -> dict:
    yrs = max(len(daily_ret) / 252.0, 1e-6)
    eq = (1.0 + daily_ret).cumprod()
    cagr = float(eq.iloc[-1] ** (1.0 / yrs) - 1.0) if len(eq) else 0.0
    mu = daily_ret.mean() * 252.0
    sd = daily_ret.std() * np.sqrt(252.0)
    sharpe = float(mu / sd) if sd > 0 else 0.0
    down = daily_ret[daily_ret < 0]
    ds = down.std() * np.sqrt(252.0) if len(down) else 0.0
    sortino = float(mu / ds) if ds > 0 else 0.0
    monthly = (1.0 + daily_ret).resample("ME").prod() - 1.0
    worst_month = float(monthly.min() * 100.0) if len(monthly) else 0.0
    # Max drawdown
    rolling_max = eq.cummax()
    dd_series = (eq / rolling_max) - 1.0
    max_dd = float(-dd_series.min() * 100.0) if len(dd_series) else 0.0
    return dict(cagr=cagr, sharpe=sharpe, sortino=sortino,
                worst_month_pct=worst_month, max_dd_pct=max_dd,
                n_days=int(len(daily_ret)))


def drawdown_series(daily_ret: pd.Series) -> pd.Series:
    eq = (1.0 + daily_ret).cumprod()
    return (eq / eq.cummax()) - 1.0


def rolling_sortino(daily_ret: pd.Series, window: int = 252) -> pd.Series:
    mu = daily_ret.rolling(window).mean() * 252.0
    down = daily_ret.where(daily_ret < 0, 0.0)
    ds = down.rolling(window).std() * np.sqrt(252.0)
    out = mu / ds.replace(0.0, np.nan)
    return out


# --- Pooled OOS reconstruction (mirrors run_wf_defensive.py R191-214) ---------

def build_pooled_oos_daily() -> pd.Series:
    print("[charts] loading feature panel + prices...", flush=True)
    features, prices = build_feature_panel(smoke=False)
    print(f"[charts] features {features.shape}  prices {prices.shape}  "
          f"span {features.index.min().date()} -> {features.index.max().date()}",
          flush=True)

    alloc_full_raw = build_allocation(features, BALANCED_CFG)
    def_flags = derive_defensive_flags(features)
    alloc_full = apply_defensive_overlay(alloc_full_raw, def_flags)
    rebal_full = cadence_dates(features.index, BALANCED_CFG["cadence"])

    # Folds (same construction as run_wf_defensive.py)
    fold_start = pd.Timestamp("2015-01-01")
    last_date = features.index.max()
    folds = []
    cur = fold_start
    while cur + pd.DateOffset(years=1) <= last_date + pd.Timedelta(days=1):
        folds.append((cur, cur + pd.DateOffset(years=1) - pd.Timedelta(days=1)))
        cur = cur + pd.DateOffset(months=6)
    tail_start = cur
    if tail_start < last_date and (last_date - tail_start).days >= 150:
        folds.append((tail_start, last_date))

    # Pooled = non-overlapping 1y blocks (every other fold)
    daily_blocks = []
    for i in range(0, len(folds), 2):
        s, e = folds[i]
        idx_mask = (features.index >= s) & (features.index <= e)
        if idx_mask.sum() < 30:
            continue
        fidx = features.index[idx_mask]
        alloc_oot = alloc_full.loc[fidx]
        rebal_oot = rebal_full[(rebal_full >= s) & (rebal_full <= e)]
        prices_oot = prices.loc[fidx]
        res = run_exposure_backtest(
            allocation=alloc_oot,
            basket_w=BALANCED_CFG["basket_weights"],
            prices=prices_oot,
            rebalance_dates=rebal_oot,
            starting_cash=100_000.0,
            allow_short=BALANCED_CFG["allow_short"],
            max_leverage=BALANCED_CFG["max_leverage"],
        )
        daily_blocks.append(res.daily_ret)
        print(f"[charts]   pooled block {s.date()} -> {e.date()}  "
              f"n={len(res.daily_ret)}", flush=True)

    pooled = pd.concat(daily_blocks).sort_index()
    pooled = pooled.groupby(pooled.index).mean()
    return pooled


# --- Chart renderers ----------------------------------------------------------

def render_equity_curve(strat: pd.Series, spy_ret: pd.Series, out: Path):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    strat_eq = (1.0 + strat).cumprod()
    spy_eq = (1.0 + spy_ret).cumprod()
    ax.plot(strat_eq.index, strat_eq.values, lw=2.0,
            label=f"Macro Overlay v1 ({annualize(strat)['cagr']*100:.1f}% CAGR)",
            color="#1f77b4")
    ax.plot(spy_eq.index, spy_eq.values, lw=1.5, alpha=0.85,
            label=f"SPY buy-and-hold ({annualize(spy_ret)['cagr']*100:.1f}% CAGR)",
            color="#888888", linestyle="--")
    ax.set_yscale("log")
    ax.set_title("Equity Curve (log scale) — Macro Overlay v1 vs SPY",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Growth of $1 (log)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, which="both", alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def render_drawdown(strat: pd.Series, spy_ret: pd.Series, out: Path):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    strat_dd = drawdown_series(strat) * 100.0
    spy_dd = drawdown_series(spy_ret) * 100.0
    ax.fill_between(strat_dd.index, strat_dd.values, 0.0,
                    alpha=0.55, color="#1f77b4", label="Macro Overlay v1")
    ax.fill_between(spy_dd.index, spy_dd.values, 0.0,
                    alpha=0.30, color="#888888", label="SPY")
    strat_max = strat_dd.min()
    spy_max = spy_dd.min()
    ax.set_title(f"Drawdown — Overlay v1 worst {strat_max:.1f}%  vs  SPY worst {spy_max:.1f}%",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("Date")
    ax.legend(loc="lower left", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def render_rolling_sortino(strat: pd.Series, spy_ret: pd.Series, out: Path):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    rs_strat = rolling_sortino(strat, 252)
    rs_spy = rolling_sortino(spy_ret, 252)
    ax.plot(rs_strat.index, rs_strat.values, lw=1.8,
            label="Macro Overlay v1", color="#1f77b4")
    ax.plot(rs_spy.index, rs_spy.values, lw=1.4,
            label="SPY", color="#888888", linestyle="--")
    ax.axhline(0.0, color="k", lw=0.6, alpha=0.6)
    ax.set_title("Rolling 1-Year Sortino — Overlay v1 vs SPY",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Sortino (1y rolling)")
    ax.set_xlabel("Date")
    ax.legend(loc="lower left", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def render_monthly_heatmap(strat: pd.Series, out: Path):
    monthly = (1.0 + strat).resample("ME").prod() - 1.0
    df = pd.DataFrame({
        "year": monthly.index.year,
        "month": monthly.index.month,
        "ret": monthly.values * 100.0,
    })
    pivot = df.pivot(index="year", columns="month", values="ret")
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    fig, ax = plt.subplots(figsize=(11, max(4.0, 0.35 * len(pivot))))
    vmax = float(np.nanmax(np.abs(pivot.values))) if pivot.size else 5.0
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto",
                   vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(12))
    ax.set_xticklabels(months)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index.tolist())
    for r in range(pivot.shape[0]):
        for c in range(pivot.shape[1]):
            v = pivot.values[r, c]
            if not np.isnan(v):
                ax.text(c, r, f"{v:+.1f}", ha="center", va="center",
                        fontsize=8, color="black")
    fig.colorbar(im, ax=ax, label="Monthly return (%)")
    ax.set_title("Monthly Returns — Macro Overlay v1",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def render_fold_cagr(folds_df: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(folds_df))
    w = 0.42
    ax.bar(x - w/2, folds_df["fold_cagr_pct"], width=w,
           label="Macro Overlay v1", color="#1f77b4")
    ax.bar(x + w/2, folds_df["spy_cagr_pct"], width=w,
           label="SPY", color="#888888")
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s[:7]}" for s in folds_df["oot_start"]],
                       rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Fold CAGR (%)")
    ax.set_title("Per-Fold OOS CAGR — Overlay v1 vs SPY (22 walk-forward folds)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def render_regime_bars(summary: dict, spy_regime_metrics: dict, out: Path):
    regimes = ["bull", "chop", "bear"]
    s_cagr = [next((r["avg_cagr_pct"] for r in summary["regimes"] if r["regime"] == reg), 0.0) for reg in regimes]
    s_sort = [next((r["avg_sortino"] for r in summary["regimes"] if r["regime"] == reg), 0.0) for reg in regimes]
    s_hit = [next((r["hit_rate_pct"] for r in summary["regimes"] if r["regime"] == reg), 0.0) for reg in regimes]
    spy_cagr = [spy_regime_metrics.get(reg, {}).get("cagr", 0.0) for reg in regimes]
    spy_sort = [spy_regime_metrics.get(reg, {}).get("sortino", 0.0) for reg in regimes]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    x = np.arange(len(regimes))
    w = 0.4

    axes[0].bar(x - w/2, s_cagr, width=w, label="Overlay v1", color="#1f77b4")
    axes[0].bar(x + w/2, spy_cagr, width=w, label="SPY", color="#888888")
    axes[0].set_xticks(x); axes[0].set_xticklabels(regimes)
    axes[0].set_title("Avg CAGR by regime (%)")
    axes[0].axhline(0, color="k", lw=0.6); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].bar(x - w/2, s_sort, width=w, label="Overlay v1", color="#1f77b4")
    axes[1].bar(x + w/2, spy_sort, width=w, label="SPY", color="#888888")
    axes[1].set_xticks(x); axes[1].set_xticklabels(regimes)
    axes[1].set_title("Avg Sortino by regime")
    axes[1].axhline(0, color="k", lw=0.6); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].bar(x, s_hit, width=0.55, color="#2ca02c")
    axes[2].set_xticks(x); axes[2].set_xticklabels(regimes)
    axes[2].set_title("Overlay v1 Hit-rate by regime (%)")
    axes[2].set_ylim(0, 105); axes[2].grid(True, alpha=0.3)

    fig.suptitle("Regime-Stratified Performance — Overlay v1 vs SPY",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def render_leverage(folds_df: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(12, 4))
    x = np.arange(len(folds_df))
    ax.bar(x, folds_df["avg_leverage"], width=0.6, color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s[:7]}" for s in folds_df["oot_start"]],
                       rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Avg gross exposure")
    ax.axhline(BALANCED_CFG["max_leverage"], color="r", lw=1.2,
               linestyle="--", label=f"Max leverage cap = {BALANCED_CFG['max_leverage']}")
    ax.set_title("Per-Fold Average Gross Exposure — Overlay v1",
                 fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


# --- Main ---------------------------------------------------------------------

def main():
    out_dir = HERE / "results" / "charts_overlay_v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[charts] output dir: {out_dir}", flush=True)

    # 1) Rebuild pooled OOS daily series
    strat_ret = build_pooled_oos_daily()
    strat_ret.index = pd.to_datetime(strat_ret.index)
    if strat_ret.index.tz is not None:
        strat_ret.index = strat_ret.index.tz_localize(None)
    print(f"[charts] pooled OOS: {len(strat_ret)} days  "
          f"{strat_ret.index.min().date()} -> {strat_ret.index.max().date()}",
          flush=True)

    # 2) Load SPY from FMP (R1 wire-in)
    spy_close = load_spy_from_fmp()
    spy_ret_full = spy_close.pct_change().dropna()
    # Align SPY to strategy index
    common_idx = strat_ret.index.intersection(spy_ret_full.index)
    strat_aligned = strat_ret.loc[common_idx]
    spy_aligned = spy_ret_full.loc[common_idx]
    print(f"[charts] aligned: {len(common_idx)} common days", flush=True)

    strat_m = annualize(strat_aligned)
    spy_m = annualize(spy_aligned)
    print(f"[charts] Strategy: CAGR={strat_m['cagr']*100:.2f}%  "
          f"DD={strat_m['max_dd_pct']:.2f}%  Sortino={strat_m['sortino']:.2f}  "
          f"Sharpe={strat_m['sharpe']:.2f}  Worst-mo={strat_m['worst_month_pct']:.2f}%",
          flush=True)
    print(f"[charts] SPY     : CAGR={spy_m['cagr']*100:.2f}%  "
          f"DD={spy_m['max_dd_pct']:.2f}%  Sortino={spy_m['sortino']:.2f}  "
          f"Sharpe={spy_m['sharpe']:.2f}  Worst-mo={spy_m['worst_month_pct']:.2f}%",
          flush=True)

    # 3) Load fold-level CSV and summary JSON
    folds_df = pd.read_csv(HERE / "results" / "wf_folds_defensive.csv")
    summary = json.load(open(HERE / "results" / "wf_summary_defensive.json"))

    # Per-regime SPY metrics (computed from spy_aligned restricted to each regime's fold dates)
    spy_regime_metrics = {}
    for reg in ["bull", "chop", "bear"]:
        sub = folds_df[folds_df["regime"] == reg]
        if len(sub) == 0:
            continue
        masks = []
        for _, row in sub.iterrows():
            s = pd.Timestamp(row["oot_start"]); e = pd.Timestamp(row["oot_end"])
            masks.append((spy_aligned.index >= s) & (spy_aligned.index <= e))
        if not masks:
            continue
        any_mask = np.any(np.vstack(masks), axis=0)
        seg = spy_aligned[any_mask]
        if len(seg) > 30:
            m = annualize(seg)
            spy_regime_metrics[reg] = {"cagr": m["cagr"] * 100.0, "sortino": m["sortino"]}

    # 4) Render the chart pack
    render_equity_curve(strat_aligned, spy_aligned, out_dir / "01_equity_curve.png")
    render_drawdown(strat_aligned, spy_aligned, out_dir / "02_drawdown.png")
    render_rolling_sortino(strat_aligned, spy_aligned, out_dir / "03_rolling_sortino.png")
    render_monthly_heatmap(strat_aligned, out_dir / "04_monthly_heatmap.png")
    render_fold_cagr(folds_df, out_dir / "05_fold_cagr.png")
    render_regime_bars(summary, spy_regime_metrics, out_dir / "06_regime.png")
    render_leverage(folds_df, out_dir / "07_leverage.png")

    # 5) Save a summary JSON for any downstream report
    out_summary = {
        "strategy": {
            "cagr_pct": strat_m["cagr"] * 100.0,
            "max_dd_pct": strat_m["max_dd_pct"],
            "sortino": strat_m["sortino"],
            "sharpe": strat_m["sharpe"],
            "worst_month_pct": strat_m["worst_month_pct"],
            "n_days": strat_m["n_days"],
        },
        "spy_baseline": {
            "cagr_pct": spy_m["cagr"] * 100.0,
            "max_dd_pct": spy_m["max_dd_pct"],
            "sortino": spy_m["sortino"],
            "sharpe": spy_m["sharpe"],
            "worst_month_pct": spy_m["worst_month_pct"],
            "n_days": spy_m["n_days"],
        },
        "spy_regime_metrics": spy_regime_metrics,
        "source": {
            "spy_source": "FMP archive (/home/jupiter/teleclaude-main/data/fmp_archive/prices/SPY_daily.json)",
            "strategy_source": "wf_folds_defensive.csv + reconstructed pooled daily series",
        },
    }
    json.dump(out_summary, open(out_dir / "summary_with_spy.json", "w"), indent=2)
    print(f"[charts] DONE  →  {out_dir}", flush=True)


if __name__ == "__main__":
    main()

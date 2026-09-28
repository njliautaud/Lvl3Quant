"""
run_wf_v2.py — Walk-forward of macro_exposure_v1 Balanced tier with overlay-v2
(252-day VIX percentile instead of 20-day).

Identical to run_wf_defensive.py except:
  - Uses defensive_overlay_v2.apply_defensive_overlay_v2 / derive_defensive_flags_v2.
  - Output filenames suffixed _v2.
  - Summary records overlay_type accordingly.

The Balanced chromosome (BALANCED_CFG) is the EXACT same as run_wf.py /
run_wf_defensive.py. No re-fitting.

Authorized under HC #420 — user's own quant research codebase.
"""
from __future__ import annotations
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ga.run_ga import build_feature_panel, build_allocation, cadence_dates
from backtest.exposure_engine import run_exposure_backtest
from backtest.defensive_overlay_v2 import (
    apply_defensive_overlay_v2,
    derive_defensive_flags_v2,
)

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)


# --- FIXED Balanced chromosome (verbatim from run_wf_defensive.py) -----------
BALANCED_CFG = {
    "feature_weights": {
        "w_naaim":              -1.859139305897906,
        "w_vix_pct":            -1.4977376854509945,
        "w_vix_ts_slope":       +1.678539632112297,
        "w_aaii_bullbear":      -1.882229941020674,
        "w_mom_3m":             -1.7861779440672065,
        "w_mom_6m":             -0.257666740748268,
        "w_spy_above_200dma":   +1.2713368647000864,
        "w_yield_2s10s":        -1.5191637785175662,
        "w_dxy_trend":          -0.39153045740817083,
        "w_breadth":            -1.5044777388177524,
        "w_gold_copper_ratio":  +0.030441514737820086,
    },
    "long_threshold":  0.08421423521041449,
    "short_threshold": -0.22733397225907126,
    "flat_band_width": 0.38112248057147324,
    "basket_weights":  {"SPY": 0.4922318994944715,
                        "QQQ": 0.37589771494571145,
                        "IWM": 0.13187038555981706},
    "max_leverage":    1.25,
    "allow_short":     False,
    "cadence":         "weekly",
    "long_strength":   1.25,
    "short_strength":  1.0,
}


# --- Helpers (verbatim) -------------------------------------------------------

def _annualize(daily_ret: pd.Series) -> dict:
    if len(daily_ret) < 5:
        return dict(cagr=0.0, sharpe=0.0, sortino=0.0, max_dd_pct=0.0,
                    worst_month_pct=0.0, n_days=int(len(daily_ret)))
    yrs = max(len(daily_ret) / 252.0, 1e-6)
    eq = (1.0 + daily_ret).cumprod()
    cagr = float(eq.iloc[-1]) ** (1.0 / yrs) - 1.0
    mu = daily_ret.mean() * 252.0
    sd = daily_ret.std() * np.sqrt(252.0)
    sharpe = mu / sd if sd > 1e-12 else 0.0
    down = daily_ret[daily_ret < 0]
    dsd = down.std() * np.sqrt(252.0) if len(down) else 0.0
    sortino = mu / dsd if dsd > 1e-12 else 0.0
    peak = eq.cummax()
    dd = (eq / peak - 1.0).min()
    monthly = (1.0 + daily_ret).resample("ME").prod() - 1.0
    worst = float(monthly.min()) if len(monthly) else 0.0
    return dict(
        cagr=float(cagr),
        sharpe=float(sharpe),
        sortino=float(sortino),
        max_dd_pct=float(abs(dd) * 100.0),
        worst_month_pct=float(worst * 100.0),
        n_days=int(len(daily_ret)),
    )


def _spy_cagr(prices: pd.DataFrame, start, end) -> float:
    s = prices["SPY"].loc[start:end].dropna()
    if len(s) < 5:
        return 0.0
    yrs = max(len(s) / 252.0, 1e-6)
    return float((s.iloc[-1] / s.iloc[0]) ** (1.0 / yrs) - 1.0)


def _regime_label(spy_cagr: float) -> str:
    if spy_cagr > 0.10:
        return "bull"
    if spy_cagr < -0.05:
        return "bear"
    return "chop"


# --- Main ---------------------------------------------------------------------

def main():
    print("[wf-v2] loading panel (full sample)", flush=True)
    features, prices = build_feature_panel(smoke=False)
    print(f"[wf-v2] features {features.shape}  prices {prices.shape}  "
          f"span {features.index.min().date()} -> {features.index.max().date()}",
          flush=True)

    # Build allocation ONCE, then apply v2 overlay ONCE on the full sample.
    alloc_full_raw = build_allocation(features, BALANCED_CFG)
    defensive_flags = derive_defensive_flags_v2(features)
    alloc_full = apply_defensive_overlay_v2(alloc_full_raw, features)
    rebal_full = cadence_dates(features.index, BALANCED_CFG["cadence"])

    pct_def = float(defensive_flags["defensive"].mean()) * 100.0
    long_days_total = int((alloc_full_raw > 0).sum())
    long_days_scaled = int(((alloc_full_raw > 0) & defensive_flags["defensive"].astype(bool)).sum())
    print(f"[wf-v2] defensive trigger frequency: {pct_def:.1f}% of days  "
          f"longs scaled: {long_days_scaled}/{long_days_total}",
          flush=True)

    # Folds
    fold_start = pd.Timestamp("2015-01-01")
    last_date = features.index.max()
    folds = []
    cur = fold_start
    while cur + pd.DateOffset(years=1) <= last_date + pd.Timedelta(days=1):
        oot_start = cur
        oot_end = cur + pd.DateOffset(years=1) - pd.Timedelta(days=1)
        folds.append((oot_start, oot_end))
        cur = cur + pd.DateOffset(months=6)
    tail_start = cur
    if tail_start < last_date and (last_date - tail_start).days >= 150:
        folds.append((tail_start, last_date))

    print(f"[wf-v2] {len(folds)} folds  first={folds[0][0].date()}  last={folds[-1][1].date()}",
          flush=True)

    rows = []
    daily_concat_blocks = []

    for k, (s, e) in enumerate(folds):
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
        m = res.metrics
        spy_c = _spy_cagr(prices, s, e)
        regime = _regime_label(spy_c)
        rows.append(dict(
            fold=k,
            oot_start=str(s.date()),
            oot_end=str(e.date()),
            regime=regime,
            spy_cagr_pct=spy_c * 100.0,
            fold_cagr_pct=m["cagr"] * 100.0,
            fold_dd_pct=m["max_dd_pct"],
            fold_sortino=m["sortino"],
            fold_sharpe=m["sharpe"],
            worst_month_pct=m["worst_month_pct"],
            turnover_per_year=m["turnover_per_year"],
            avg_leverage=m["avg_leverage"],
            n_days=int(len(res.daily_ret)),
        ))
        print(f"[wf-v2] fold {k:2d} {s.date()} -> {e.date()}  regime={regime} "
              f"spyCAGR={spy_c*100:5.1f}%  cagr={m['cagr']*100:6.2f}%  "
              f"dd={m['max_dd_pct']:5.2f}%  sortino={m['sortino']:5.2f}",
              flush=True)

    # Pooled OOS: non-overlapping 1y blocks (every other fold)
    nonoverlap_idx = list(range(0, len(folds), 2))
    for i in nonoverlap_idx:
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
        daily_concat_blocks.append(res.daily_ret)

    pooled = pd.concat(daily_concat_blocks).sort_index()
    pooled = pooled.groupby(pooled.index).mean()
    pooled_metrics = _annualize(pooled)
    print(f"[wf-v2] pooled OOS: CAGR={pooled_metrics['cagr']*100:.2f}%  "
          f"DD={pooled_metrics['max_dd_pct']:.2f}%  "
          f"Sortino={pooled_metrics['sortino']:.2f}  "
          f"Sharpe={pooled_metrics['sharpe']:.2f}  "
          f"n_days={pooled_metrics['n_days']}", flush=True)

    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "wf_folds_v2.parquet", index=False)
    df.to_csv(OUT / "wf_folds_v2.csv", index=False)

    regime_rows = []
    for reg in ["bull", "bear", "chop"]:
        sub = df[df["regime"] == reg]
        if len(sub) == 0:
            regime_rows.append(dict(regime=reg, n_folds=0, avg_cagr_pct=float("nan"),
                                    avg_dd_pct=float("nan"), avg_sortino=float("nan"),
                                    avg_sharpe=float("nan"), hit_rate_pct=float("nan")))
            continue
        regime_rows.append(dict(
            regime=reg,
            n_folds=len(sub),
            avg_cagr_pct=float(sub["fold_cagr_pct"].mean()),
            avg_dd_pct=float(sub["fold_dd_pct"].mean()),
            avg_sortino=float(sub["fold_sortino"].mean()),
            avg_sharpe=float(sub["fold_sharpe"].mean()),
            hit_rate_pct=float((sub["fold_cagr_pct"] > 0).mean() * 100.0),
        ))
    reg_df = pd.DataFrame(regime_rows)
    reg_df.to_csv(OUT / "wf_regimes_v2.csv", index=False)

    hit_rate = float((df["fold_cagr_pct"] > 0).mean() * 100.0)
    reasons = []
    if pooled_metrics["cagr"] * 100.0 < 10.0:
        reasons.append(f"pooled CAGR {pooled_metrics['cagr']*100:.2f}% < 10%")
    if pooled_metrics["max_dd_pct"] > 18.0:
        reasons.append(f"pooled max DD {pooled_metrics['max_dd_pct']:.2f}% > 18%")
    for r in regime_rows:
        if r["n_folds"] > 0 and r["avg_sortino"] < 0.5:
            reasons.append(f"{r['regime']} regime Sortino {r['avg_sortino']:.2f} < 0.5")
    if hit_rate < 65.0:
        reasons.append(f"hit rate {hit_rate:.1f}% < 65%")
    verdict = "PASS" if not reasons else "FAIL"

    summary = dict(
        overlay_type="defensive_v2: long*=0.5 when (SPY<=200dma) OR (vix_pct_252d>0.80)",
        defensive_trigger_pct_of_days=pct_def,
        long_days_scaled=long_days_scaled,
        long_days_total=long_days_total,
        pooled_cagr_pct=pooled_metrics["cagr"] * 100.0,
        pooled_dd_pct=pooled_metrics["max_dd_pct"],
        pooled_sortino=pooled_metrics["sortino"],
        pooled_sharpe=pooled_metrics["sharpe"],
        pooled_worst_month_pct=pooled_metrics["worst_month_pct"],
        pooled_n_days=pooled_metrics["n_days"],
        n_folds=len(df),
        hit_rate_pct=hit_rate,
        verdict=verdict,
        reasons=reasons,
        regimes=regime_rows,
    )
    (OUT / "wf_summary_v2.json").write_text(json.dumps(summary, indent=2))

    print(f"[wf-v2] wrote summary + folds.")
    print(f"[wf-v2] verdict: {verdict}  reasons={reasons}", flush=True)
    return summary


if __name__ == "__main__":
    main()

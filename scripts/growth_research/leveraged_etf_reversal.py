#!/usr/bin/env python3
"""
Leveraged ETF Reversal Strategy — Does 3x amplify or destroy mean-reversion?
=============================================================================
Tests short-term reversal (buy losers, sell winners) on leveraged/inverse ETFs.

Three universe options tested:
  A) Leveraged sector 3x: TQQQ, SOXL, TECL, FAS, LABU, CURE, DPST, TNA, UDOW, SPXL
  B) Mix of leveraged + plain for diversification
  C) Inverse-included long-short rotation (naturally hedged)

Baseline: plain ETF reversal (validated Sharpe 0.83) for comparison.

HC #705 adversarial checks:
  1. Permutation test (200 shuffles)
  2. R1 regime test (green/red/flat)
  3. Sub-period consistency
  4. Outlier removal
  5. Compare to plain ETF reversal baseline
"""

import json
import os
import sys
import warnings
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings("ignore")

# ── Universe Options ──────────────────────────────────────────────────────────

UNIVERSE_A = {
    "name": "Leveraged 3x Sector",
    "tickers": ["TQQQ", "SOXL", "TECL", "FAS", "LABU", "CURE", "DPST", "TNA", "UDOW", "SPXL"],
    "description": "Pure 3x leveraged ETFs — max amplification",
}

UNIVERSE_B = {
    "name": "Leveraged + Plain Mix",
    "tickers": ["TQQQ", "SOXL", "SPXL", "FAS", "TNA", "XLE", "XLF", "XLK", "XLV", "GLD", "TLT"],
    "description": "5 leveraged + 6 plain — diversification with amplification",
}

UNIVERSE_C = {
    "name": "Long-Short Inverse Rotation",
    "long_tickers": ["TQQQ", "SOXL", "SPXL", "FAS", "TNA"],
    "short_tickers": ["SQQQ", "SOXS", "SPXS", "FAZ", "TZA"],
    "tickers": ["TQQQ", "SOXL", "SPXL", "FAS", "TNA", "SQQQ", "SOXS", "SPXS", "FAZ", "TZA"],
    "description": "Buy worst 3 from both leveraged + inverse — natural hedge",
}

BASELINE_UNIVERSE = {
    "name": "Plain ETF Baseline",
    "tickers": ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY",
                 "SPY", "QQQ", "IWM", "MDY", "EFA", "EEM", "TLT", "IEF", "GLD", "SLV"],
    "description": "Validated plain ETF reversal (Sharpe ~0.83)",
}

ALL_TICKERS = sorted(set(
    UNIVERSE_A["tickers"] + UNIVERSE_B["tickers"] + UNIVERSE_C["tickers"]
    + BASELINE_UNIVERSE["tickers"] + ["SPY", "^VIX"]
))

START = "2011-01-01"  # Most 3x ETFs launched 2010-2012
END = "2026-07-15"
OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/growth_research")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_PERM = 200
np.random.seed(42)

# Strategy params to test
LOOKBACK_GRID = [3, 5, 10]
K_GRID = [3, 5]
HOLD_GRID = [3, 5, 10]


# ── Data Download ────────────────────────────────────────────────────────────

def download_data():
    """Download adjusted close prices for all tickers."""
    print(f"Downloading {len(ALL_TICKERS)} tickers from {START} to {END}...")

    all_close = {}
    batch_size = 5
    for i in range(0, len(ALL_TICKERS), batch_size):
        batch = ALL_TICKERS[i:i + batch_size]
        print(f"  Batch {i // batch_size + 1}: {batch}")
        try:
            data = yf.download(batch, start=START, end=END, auto_adjust=True,
                               progress=False, threads=False, timeout=30)
            if isinstance(data.columns, pd.MultiIndex):
                close = data["Close"]
            else:
                close = data
            if isinstance(close, pd.Series):
                all_close[batch[0]] = close
            else:
                for col in close.columns:
                    all_close[col] = close[col]
        except Exception as e:
            print(f"    WARNING: Failed to download {batch}: {e}")

    close_df = pd.DataFrame(all_close)
    close_df = close_df.ffill(limit=5)

    # Report data availability
    for t in ALL_TICKERS:
        if t in close_df.columns:
            first = close_df[t].first_valid_index()
            last = close_df[t].last_valid_index()
            pct_na = close_df[t].isna().mean() * 100
            print(f"    {t:6s}: {first.date()} to {last.date()} ({pct_na:.1f}% missing)")
        else:
            print(f"    {t:6s}: NOT AVAILABLE")

    print(f"  Total: {len(close_df)} trading days, {close_df.shape[1]} tickers")
    return close_df


# ── Core Backtest Engine ──────────────────────────────────────────────────────

def run_reversal_backtest(close_df, etf_cols, lookback, k_etfs, hold, mode="reversal"):
    """
    Vectorized reversal/momentum backtest.
    mode: 'reversal' (buy losers), 'momentum' (buy winners), 'random'
    Returns: (dates_array, returns_array) or (None, None)
    """
    available = [c for c in etf_cols if c in close_df.columns]
    if len(available) < k_etfs:
        return None, None

    trailing_ret = close_df[available].pct_change(lookback)
    dates = close_df.index

    # Weekly rebalance: last trading day of each week
    week_ser = dates.to_series().dt.isocalendar()
    week_key = week_ser[["year", "week"]].astype(str).agg("-".join, axis=1)
    rebal_dates = dates.to_series().groupby(week_key).last().sort_values().values
    rebal_dates = [d for d in rebal_dates if d >= dates[lookback + 5]]

    period_returns = []
    period_dates = []

    for rebal_date in rebal_dates:
        rebal_idx = dates.get_loc(rebal_date)
        if rebal_idx + hold >= len(dates):
            break

        tr = trailing_ret.loc[rebal_date, available].dropna()
        if len(tr) < k_etfs:
            continue

        if mode == "reversal":
            selected = tr.sort_values().head(k_etfs).index
        elif mode == "momentum":
            selected = tr.sort_values(ascending=False).head(k_etfs).index
        elif mode == "random":
            selected = tr.sample(min(k_etfs, len(tr))).index
        else:
            raise ValueError(f"Unknown mode: {mode}")

        entry_price = close_df.loc[rebal_date, selected]
        exit_date = dates[rebal_idx + hold]
        exit_price = close_df.loc[exit_date, selected]

        fwd_rets = (exit_price / entry_price - 1.0).dropna()
        if len(fwd_rets) == 0:
            continue

        basket_ret = fwd_rets.mean()
        # HC #718 R3: transaction costs — 10 bps round-trip (5 bps each way)
        basket_ret -= 10 / 10000
        period_returns.append(basket_ret)
        period_dates.append(rebal_date)

    if len(period_returns) < 10:
        return None, None

    return np.array(period_dates), np.array(period_returns)


def compute_metrics(returns, hold):
    """Compute risk-adjusted metrics from period returns."""
    if returns is None or len(returns) < 10:
        return None

    periods_per_year = 252 / hold
    mean_ret = returns.mean()
    std_ret = returns.std()
    win_rate = (returns > 0).mean()

    ann_mean = mean_ret * periods_per_year
    ann_std = std_ret * np.sqrt(periods_per_year)
    sharpe = ann_mean / ann_std if ann_std > 0 else 0.0

    downside = returns[returns < 0]
    downside_std = downside.std() * np.sqrt(periods_per_year) if len(downside) > 1 else ann_std
    sortino = ann_mean / downside_std if downside_std > 0 else 0.0

    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    profit_factor = gains / losses if losses > 0 else 999.0

    cum = (1 + returns).cumprod()
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    max_dd = dd.min()

    return {
        "sharpe": sharpe,
        "sortino": sortino,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "mean_return_pct": mean_ret * 100,
        "max_dd_pct": max_dd * 100,
        "ann_return_pct": ann_mean * 100,
        "n_periods": len(returns),
    }


# ── Adversarial Checks ──────────────────────────────────────────────────────

def adversarial_checks(close_df, etf_cols, dates_arr, returns, lookback, k, hold):
    """HC #705 adversarial checks."""
    checks = {}
    periods_per_year = 252 / hold

    # 1. Permutation test (200 shuffles)
    print("    Permutation test...")
    actual_mean = returns.mean()
    available = [c for c in etf_cols if c in close_df.columns]
    trailing_ret = close_df[available].pct_change(lookback)
    dates = close_df.index
    week_ser = dates.to_series().dt.isocalendar()
    week_key = week_ser[["year", "week"]].astype(str).agg("-".join, axis=1)
    rebal_dates = dates.to_series().groupby(week_key).last().sort_values().values
    rebal_dates = [d for d in rebal_dates if d >= dates[lookback + 5]]

    fwd_ret_vectors = []
    for rd in rebal_dates:
        ri = dates.get_loc(rd)
        if ri + hold >= len(dates):
            break
        tr = trailing_ret.loc[rd, available].dropna()
        if len(tr) < k:
            continue
        entry = close_df.loc[rd, tr.index]
        exit_d = dates[ri + hold]
        exit_p = close_df.loc[exit_d, tr.index]
        fwd = (exit_p / entry - 1.0).dropna()
        if len(fwd) >= k:
            fwd_ret_vectors.append(fwd.values)

    perm_means = np.zeros(N_PERM)
    for p in range(N_PERM):
        prets = []
        for fv in fwd_ret_vectors:
            pick = np.random.choice(len(fv), size=min(k, len(fv)), replace=False)
            prets.append(fv[pick].mean())
        perm_means[p] = np.mean(prets) if prets else 0.0

    p_value = (perm_means >= actual_mean).mean()
    checks["permutation"] = {
        "actual_mean_pct": round(actual_mean * 100, 4),
        "perm_mean_pct": round(perm_means.mean() * 100, 4),
        "p_value": round(float(p_value), 4),
        "pass": p_value < 0.05,
    }
    print(f"      p={p_value:.4f} ({'PASS' if p_value < 0.05 else 'FAIL'})")

    # 2. Sub-period consistency
    mid = len(returns) // 2
    h1_mean = returns[:mid].mean()
    h2_mean = returns[mid:].mean()
    sub_pass = h1_mean > 0 and h2_mean > 0
    checks["subperiod"] = {
        "first_half_mean_pct": round(h1_mean * 100, 4),
        "second_half_mean_pct": round(h2_mean * 100, 4),
        "pass": sub_pass,
    }
    print(f"      Sub-period: H1={h1_mean*100:.3f}%, H2={h2_mean*100:.3f}% ({'PASS' if sub_pass else 'FAIL'})")

    # 3. Outlier removal (trim top/bottom 5%)
    p5, p95 = np.percentile(returns, [5, 95])
    trimmed = returns[(returns >= p5) & (returns <= p95)]
    trimmed_mean = trimmed.mean() if len(trimmed) > 0 else 0.0
    outlier_pass = trimmed_mean > 0
    checks["outlier_removal"] = {
        "trimmed_mean_pct": round(trimmed_mean * 100, 4),
        "n_removed": len(returns) - len(trimmed),
        "pass": outlier_pass,
    }
    print(f"      Outlier removal: trimmed={trimmed_mean*100:.3f}% ({'PASS' if outlier_pass else 'FAIL'})")

    # 4. R1 regime test
    spy = close_df.get("SPY")
    if spy is not None:
        spy_weekly_ret = spy.pct_change(5)
        green_rets, red_rets, flat_rets = [], [], []
        for d, r in zip(dates_arr, returns):
            try:
                sr = spy_weekly_ret.loc[d]
                if pd.isna(sr):
                    continue
                if sr > 0.005:
                    green_rets.append(r)
                elif sr < -0.005:
                    red_rets.append(r)
                else:
                    flat_rets.append(r)
            except (KeyError, TypeError):
                continue

        def _sharpe(rets):
            rets = np.array(rets) if rets else np.array([0.0])
            if len(rets) < 5:
                return 0.0
            m = rets.mean() * periods_per_year
            s = rets.std() * np.sqrt(periods_per_year)
            return m / s if s > 0 else 0.0

        sg = _sharpe(green_rets)
        sr_ = _sharpe(red_rets)
        sf = _sharpe(flat_rets)
        max_s = max(abs(sg), abs(sr_))
        gap = abs(sg - sr_) / max_s if max_s > 0 else 0.0
        regime_pass = gap <= 0.50

        checks["regime"] = {
            "sharpe_green": round(sg, 3),
            "sharpe_red": round(sr_, 3),
            "sharpe_flat": round(sf, 3),
            "n_green": len(green_rets),
            "n_red": len(red_rets),
            "n_flat": len(flat_rets),
            "regime_gap": round(gap, 3),
            "pass": regime_pass,
            "better_in_bear": sr_ > sg,
        }
        print(f"      Regime: Green={sg:.3f}, Red={sr_:.3f}, Gap={gap:.3f} ({'PASS' if regime_pass else 'FAIL'})")
    else:
        checks["regime"] = {"pass": False, "error": "No SPY data"}

    # Overall
    all_pass = all(c.get("pass", False) for c in checks.values())
    checks["all_pass"] = all_pass
    return checks


# ── Run one universe through full analysis ────────────────────────────────────

def analyze_universe(close_df, universe_def, label):
    """Run reversal backtest + adversarial on one universe. Returns summary dict."""
    print(f"\n{'=' * 80}")
    print(f"UNIVERSE {label}: {universe_def['name']}")
    print(f"  {universe_def['description']}")
    print(f"  Tickers: {universe_def['tickers']}")
    print(f"{'=' * 80}")

    etf_cols = [t for t in universe_def["tickers"] if t in close_df.columns
                and close_df[t].notna().mean() > 0.5]

    if len(etf_cols) < 3:
        print(f"  SKIP — only {len(etf_cols)} usable tickers")
        return None

    print(f"  Usable: {len(etf_cols)} — {etf_cols}")

    # Grid search all configs
    results = []
    for lookback, k, hold in product(LOOKBACK_GRID, K_GRID, HOLD_GRID):
        if k > len(etf_cols):
            continue

        dates_r, rets_r = run_reversal_backtest(close_df, etf_cols, lookback, k, hold, "reversal")
        metrics_r = compute_metrics(rets_r, hold)

        dates_m, rets_m = run_reversal_backtest(close_df, etf_cols, lookback, k, hold, "momentum")
        metrics_m = compute_metrics(rets_m, hold)

        if metrics_r:
            results.append({
                "mode": "reversal",
                "lookback": lookback, "k": k, "hold": hold,
                "metrics": metrics_r,
                "dates": dates_r, "returns": rets_r,
            })
        if metrics_m:
            results.append({
                "mode": "momentum",
                "lookback": lookback, "k": k, "hold": hold,
                "metrics": metrics_m,
                "dates": dates_m, "returns": rets_m,
            })

    if not results:
        print("  No valid results")
        return None

    # Print comparison table
    rev_results = sorted([r for r in results if r["mode"] == "reversal"],
                         key=lambda x: x["metrics"]["sharpe"], reverse=True)
    mom_results = sorted([r for r in results if r["mode"] == "momentum"],
                         key=lambda x: x["metrics"]["sharpe"], reverse=True)

    print(f"\n  {'Config':<25} {'Sharpe':>7} {'Sortino':>8} {'WR':>6} {'PF':>6} {'AnnRet%':>8} {'MaxDD%':>7}")
    print(f"  {'-' * 75}")
    for r in rev_results[:8]:
        m = r["metrics"]
        name = f"REV L={r['lookback']} K={r['k']} H={r['hold']}"
        print(f"  {name:<25} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} {m['win_rate']:>5.1%} "
              f"{m['profit_factor']:>6.2f} {m['ann_return_pct']:>7.1f}% {m['max_dd_pct']:>6.1f}%")
    print(f"  {'--- Momentum ---':^75}")
    for r in mom_results[:3]:
        m = r["metrics"]
        name = f"MOM L={r['lookback']} K={r['k']} H={r['hold']}"
        print(f"  {name:<25} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} {m['win_rate']:>5.1%} "
              f"{m['profit_factor']:>6.2f} {m['ann_return_pct']:>7.1f}% {m['max_dd_pct']:>6.1f}%")

    # Adversarial on top 3 reversal configs
    adversarial_results = []
    for r in rev_results[:3]:
        name = f"REV L={r['lookback']} K={r['k']} H={r['hold']}"
        print(f"\n  Adversarial: {name} (Sharpe={r['metrics']['sharpe']:.3f})")
        checks = adversarial_checks(close_df, etf_cols, r["dates"], r["returns"],
                                     r["lookback"], r["k"], r["hold"])

        # SPY correlation
        spy = close_df.get("SPY")
        spy_corr = None
        if spy is not None:
            spy_rets = []
            for d in r["dates"]:
                try:
                    idx = close_df.index.get_loc(d)
                    if idx + r["hold"] < len(close_df):
                        spy_rets.append(spy.iloc[idx + r["hold"]] / spy.iloc[idx] - 1.0)
                    else:
                        spy_rets.append(np.nan)
                except (KeyError, IndexError):
                    spy_rets.append(np.nan)
            spy_rets = np.array(spy_rets)
            valid = ~np.isnan(spy_rets) & ~np.isnan(r["returns"])
            if valid.sum() > 10:
                spy_corr = round(float(np.corrcoef(r["returns"][valid], spy_rets[valid])[0, 1]), 4)

        adversarial_results.append({
            "config": name,
            "lookback": r["lookback"], "k": r["k"], "hold": r["hold"],
            "metrics": {k: round(v, 4) if isinstance(v, float) else v
                        for k, v in r["metrics"].items()},
            "checks": checks,
            "spy_correlation": spy_corr,
        })

    # Summary
    best = rev_results[0] if rev_results else None
    best_mom = mom_results[0] if mom_results else None

    return {
        "universe": universe_def["name"],
        "tickers_used": etf_cols,
        "n_configs_tested": len(results),
        "best_reversal": {
            "config": f"L={best['lookback']} K={best['k']} H={best['hold']}",
            "metrics": {k: round(v, 4) if isinstance(v, float) else v
                        for k, v in best["metrics"].items()},
        } if best else None,
        "best_momentum": {
            "config": f"L={best_mom['lookback']} K={best_mom['k']} H={best_mom['hold']}",
            "metrics": {k: round(v, 4) if isinstance(v, float) else v
                        for k, v in best_mom["metrics"].items()},
        } if best_mom else None,
        "reversal_beats_momentum": (
            best["metrics"]["sharpe"] > best_mom["metrics"]["sharpe"]
            if best and best_mom else None
        ),
        "adversarial": adversarial_results,
        "all_reversal_configs": [
            {
                "config": f"L={r['lookback']} K={r['k']} H={r['hold']}",
                "metrics": {k: round(v, 4) if isinstance(v, float) else v
                            for k, v in r["metrics"].items()},
            }
            for r in rev_results
        ],
    }


# ── Volatility drag analysis ─────────────────────────────────────────────────

def volatility_drag_analysis(close_df):
    """Measure empirical vol drag: compare 3x daily return vs 3*1x return."""
    print(f"\n{'=' * 80}")
    print("VOLATILITY DRAG ANALYSIS — 3x vs 3 * 1x")
    print(f"{'=' * 80}")

    pairs = [
        ("TQQQ", "QQQ"), ("SOXL", "SOXL"),  # SOXL has no 1x match in our data
        ("SPXL", "SPY"), ("TNA", "IWM"), ("FAS", "XLF"),
    ]

    results = {}
    for lev, plain in pairs:
        if lev not in close_df.columns or plain not in close_df.columns:
            continue
        if lev == plain:
            continue

        # Daily returns
        lev_ret = close_df[lev].pct_change().dropna()
        plain_ret = close_df[plain].pct_change().dropna()
        common = lev_ret.index.intersection(plain_ret.index)
        if len(common) < 100:
            continue

        lev_r = lev_ret.loc[common]
        plain_r = plain_ret.loc[common]

        # Expected 3x daily return vs actual
        expected_3x = plain_r * 3
        actual = lev_r

        # Tracking error
        tracking_err = (actual - expected_3x).std() * np.sqrt(252)

        # Cumulative drag over 5-day holds
        lev_5d = close_df[lev].pct_change(5).dropna()
        plain_5d = close_df[plain].pct_change(5).dropna()
        common_5d = lev_5d.index.intersection(plain_5d.index)
        lev_5d_c = lev_5d.loc[common_5d]
        plain_5d_c = plain_5d.loc[common_5d]
        expected_3x_5d = plain_5d_c * 3

        avg_drag_5d = (lev_5d_c - expected_3x_5d).mean() * 100

        results[f"{lev} vs 3*{plain}"] = {
            "tracking_error_annual": round(float(tracking_err * 100), 2),
            "avg_5d_drag_pct": round(float(avg_drag_5d), 4),
            "drag_helps_reversal": avg_drag_5d < 0,  # Negative drag = losers compressed more
        }
        print(f"  {lev} vs 3*{plain}: tracking_err={tracking_err*100:.1f}%, "
              f"5d_drag={avg_drag_5d:.4f}%")

    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = datetime.now()
    print("=" * 80)
    print("LEVERAGED ETF REVERSAL STRATEGY TEST")
    print(f"Does 3x amplification help or hurt mean-reversion?")
    print(f"Period: {START} to {END}")
    print("=" * 80)

    # Download all data
    close_df = download_data()

    # Vol drag analysis
    vol_drag = volatility_drag_analysis(close_df)

    # Run each universe
    results_by_universe = {}

    for label, udef in [("A", UNIVERSE_A), ("B", UNIVERSE_B),
                         ("C", UNIVERSE_C), ("BASELINE", BASELINE_UNIVERSE)]:
        result = analyze_universe(close_df, udef, label)
        if result:
            results_by_universe[label] = result

    # ── Final comparison table ────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("FINAL COMPARISON: LEVERAGED vs PLAIN ETF REVERSAL")
    print(f"{'=' * 80}")

    print(f"\n{'Universe':<30} {'Best Sharpe':>11} {'Sortino':>8} {'WR':>6} {'PF':>6} "
          f"{'AnnRet%':>8} {'MaxDD%':>7} {'All Gates':>10}")
    print("-" * 95)

    for label, res in results_by_universe.items():
        br = res.get("best_reversal")
        if not br:
            continue
        m = br["metrics"]
        # Check if any config passes all adversarial gates
        any_pass = any(a["checks"].get("all_pass", False) for a in res.get("adversarial", []))
        print(f"  {label + ': ' + res['universe']:<28} {m['sharpe']:>11.3f} {m['sortino']:>8.3f} "
              f"{m['win_rate']:>5.1%} {m['profit_factor']:>6.2f} "
              f"{m['ann_return_pct']:>7.1f}% {m['max_dd_pct']:>6.1f}% {'PASS' if any_pass else 'FAIL':>10}")

    # ── Key findings ──────────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("KEY FINDINGS")
    print(f"{'=' * 80}")

    baseline = results_by_universe.get("BASELINE", {}).get("best_reversal", {}).get("metrics", {})
    base_sharpe = baseline.get("sharpe", 0)

    for label in ["A", "B", "C"]:
        res = results_by_universe.get(label)
        if not res or not res.get("best_reversal"):
            continue
        lev_sharpe = res["best_reversal"]["metrics"]["sharpe"]
        diff = lev_sharpe - base_sharpe
        better = "BETTER" if diff > 0.1 else "WORSE" if diff < -0.1 else "SIMILAR"
        any_pass = any(a["checks"].get("all_pass", False) for a in res.get("adversarial", []))
        rev_beats_mom = res.get("reversal_beats_momentum")

        print(f"\n  Universe {label} ({res['universe']}):")
        print(f"    Sharpe: {lev_sharpe:.3f} vs baseline {base_sharpe:.3f} ({better}, delta={diff:+.3f})")
        print(f"    Reversal beats momentum: {rev_beats_mom}")
        print(f"    Passes all adversarial gates: {any_pass}")
        for a in res.get("adversarial", []):
            ch = a["checks"]
            gates = " | ".join(f"{k}:{'OK' if v.get('pass') else 'FAIL'}"
                               for k, v in ch.items() if isinstance(v, dict) and "pass" in v)
            print(f"      {a['config']}: {gates}")
            regime = ch.get("regime", {})
            if regime.get("better_in_bear"):
                print(f"        BETTER IN BEAR (Red={regime['sharpe_red']:.3f} > Green={regime['sharpe_green']:.3f})")

    # Vol drag verdict
    if vol_drag:
        print(f"\n  Volatility Drag Impact:")
        for pair, info in vol_drag.items():
            print(f"    {pair}: 5d drag={info['avg_5d_drag_pct']:.4f}% "
                  f"({'helps reversal' if info['drag_helps_reversal'] else 'hurts reversal'})")

    # ── Save results ──────────────────────────────────────────────────────
    output = {
        "metadata": {
            "script": "leveraged_etf_reversal.py",
            "run_date": datetime.now().isoformat(),
            "period": f"{START} to {END}",
            "n_permutations": N_PERM,
            "lookback_grid": LOOKBACK_GRID,
            "k_grid": K_GRID,
            "hold_grid": HOLD_GRID,
            "runtime_seconds": (datetime.now() - t0).total_seconds(),
        },
        "volatility_drag": vol_drag,
        "results_by_universe": {
            label: {k: v for k, v in res.items() if k not in ("dates", "returns")}
            for label, res in results_by_universe.items()
        },
        "comparison": {
            "baseline_sharpe": base_sharpe,
            "universe_a_sharpe": results_by_universe.get("A", {}).get("best_reversal", {}).get("metrics", {}).get("sharpe"),
            "universe_b_sharpe": results_by_universe.get("B", {}).get("best_reversal", {}).get("metrics", {}).get("sharpe"),
            "universe_c_sharpe": results_by_universe.get("C", {}).get("best_reversal", {}).get("metrics", {}).get("sharpe"),
        },
    }

    out_path = OUTPUT_DIR / "leveraged_etf_reversal_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    elapsed = (datetime.now() - t0).total_seconds()
    print(f"\nResults saved. Runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Sector Leadership as SPY Predictor — Comprehensive Study
=========================================================
HC #705 compliant: permutation tests, sub-period consistency, regime checks.

Key question: When a given sector ETF leads (top 1-3 by relative strength),
what happens to SPY forward returns? Deep dive on XLU (utilities).

Uses 10+ years of data via yfinance.
"""

import json
import os
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SECTOR_ETFS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLV": "Health Care",
    "XLI": "Industrials",
    "XLY": "Consumer Disc.",
    "XLC": "Comm. Services",
    "XLP": "Consumer Staples",
    "XLRE": "Real Estate",
    "XLB": "Materials",
    "XLU": "Utilities",
}
BENCHMARK = "SPY"
VIX_TICKER = "^VIX"

LOOKBACK_WINDOWS = {"1m": 21, "3m": 63}  # trailing windows for leadership calc
FORWARD_WINDOWS = {"1w": 5, "1m": 21, "3m": 63}  # forward SPY return horizons
TOP_N_LEADER = 3  # "leading" = top N by relative strength

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "output" / "growth_research" / "sector_predictor"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_PERMUTATIONS = 1000  # for permutation test


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def download_data(years: int = 14) -> tuple:
    """Download sector ETFs, SPY, and VIX."""
    import yfinance as yf

    end = datetime.now()
    start = end - timedelta(days=years * 365)
    tickers = list(SECTOR_ETFS.keys()) + [BENCHMARK, VIX_TICKER]
    print(f"Downloading {len(tickers)} tickers from {start.date()} to {end.date()} ...")
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data[["Close"]].rename(columns={"Close": tickers[0]})

    prices = prices.dropna(how="all").ffill()

    # Separate VIX
    vix = prices[VIX_TICKER].copy() if VIX_TICKER in prices.columns else None
    sector_cols = list(SECTOR_ETFS.keys()) + [BENCHMARK]
    prices = prices[[c for c in sector_cols if c in prices.columns]]

    print(f"  Got {len(prices)} trading days, {prices.shape[1]} tickers")
    if vix is not None:
        print(f"  VIX data: {vix.dropna().shape[0]} days")
    return prices, vix


# ---------------------------------------------------------------------------
# Core: relative strength rankings
# ---------------------------------------------------------------------------
def compute_relative_strength(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Relative strength = sector return over window minus SPY return over window.
    Returns a DataFrame of RS values for each sector (not SPY).
    """
    rets = prices.pct_change(window)
    sector_tickers = [t for t in SECTOR_ETFS.keys() if t in rets.columns]
    rs = rets[sector_tickers].sub(rets[BENCHMARK], axis=0)
    return rs


def get_leadership_flags(prices: pd.DataFrame, lookback_window: int, top_n: int = 3) -> pd.DataFrame:
    """
    For each day, flag which sectors are in the top N by relative strength.
    Returns boolean DataFrame (sectors x dates).
    """
    rs = compute_relative_strength(prices, lookback_window)
    # Rank: higher RS = lower rank number (1 = best)
    ranks = rs.rank(axis=1, ascending=False)
    flags = ranks <= top_n
    return flags


def compute_forward_returns(prices: pd.DataFrame, window: int) -> pd.Series:
    """Forward SPY returns over `window` trading days."""
    return prices[BENCHMARK].pct_change(window).shift(-window)


# ---------------------------------------------------------------------------
# 1. Sector Leadership as SPY Predictor
# ---------------------------------------------------------------------------
def sector_leadership_analysis(prices: pd.DataFrame) -> dict:
    """For each sector, when it leads, what are forward SPY returns?"""
    print("\n" + "=" * 70)
    print("1. SECTOR LEADERSHIP AS SPY PREDICTOR")
    print("=" * 70)

    results = {}

    for lb_label, lb_days in LOOKBACK_WINDOWS.items():
        flags = get_leadership_flags(prices, lb_days, TOP_N_LEADER)
        print(f"\n--- Lookback: {lb_label} ({lb_days}d), Top {TOP_N_LEADER} leaders ---")

        for fw_label, fw_days in FORWARD_WINDOWS.items():
            fwd_ret = compute_forward_returns(prices, fw_days)

            # Unconditional
            uncond = fwd_ret.dropna()
            uncond_mean = uncond.mean()
            uncond_std = uncond.std()
            uncond_wr = (uncond > 0).mean()

            print(f"\n  Forward {fw_label} ({fw_days}d) — Unconditional: "
                  f"mean={uncond_mean*100:.2f}%, WR={uncond_wr*100:.1f}%, N={len(uncond)}")
            print(f"  {'Sector':<8} {'Mean%':>8} {'Diff%':>8} {'t-stat':>8} {'p-val':>8} {'WR%':>8} {'N':>6}")
            print(f"  {'-'*56}")

            for sector in SECTOR_ETFS.keys():
                if sector not in flags.columns:
                    continue
                mask = flags[sector]
                cond = fwd_ret[mask].dropna()
                if len(cond) < 20:
                    continue

                cond_mean = cond.mean()
                diff = cond_mean - uncond_mean
                # t-test: conditional vs unconditional
                t_stat, p_val = stats.ttest_ind(cond.values, uncond.values, equal_var=False)
                wr = (cond > 0).mean()

                key = f"{sector}_{lb_label}_{fw_label}"
                results[key] = {
                    "sector": sector,
                    "sector_name": SECTOR_ETFS[sector],
                    "lookback": lb_label,
                    "forward": fw_label,
                    "cond_mean_pct": round(cond_mean * 100, 3),
                    "uncond_mean_pct": round(uncond_mean * 100, 3),
                    "diff_pct": round(diff * 100, 3),
                    "t_stat": round(t_stat, 3),
                    "p_val": round(p_val, 4),
                    "win_rate": round(wr * 100, 1),
                    "n_obs": len(cond),
                }

                flag = " ***" if abs(t_stat) > 2.0 else (" **" if abs(t_stat) > 1.65 else "")
                print(f"  {sector:<8} {cond_mean*100:>8.2f} {diff*100:>8.2f} {t_stat:>8.2f} "
                      f"{p_val:>8.4f} {wr*100:>8.1f} {len(cond):>6}{flag}")

    return results


# ---------------------------------------------------------------------------
# 2. Utilities Deep Dive
# ---------------------------------------------------------------------------
def utilities_deep_dive(prices: pd.DataFrame, vix: pd.Series) -> dict:
    """Deep analysis when XLU leads."""
    print("\n" + "=" * 70)
    print("2. UTILITIES (XLU) DEEP DIVE")
    print("=" * 70)

    results = {}

    for lb_label, lb_days in LOOKBACK_WINDOWS.items():
        flags = get_leadership_flags(prices, lb_days, TOP_N_LEADER)
        if "XLU" not in flags.columns:
            print("  XLU not in data, skipping")
            continue

        xlu_leading = flags["XLU"]
        xlu_lagging = ~xlu_leading

        print(f"\n--- XLU Leadership (lookback {lb_label}) ---")
        print(f"  XLU leading days: {xlu_leading.sum()} / {len(xlu_leading)} "
              f"({xlu_leading.mean()*100:.1f}%)")

        # Forward returns comparison
        for fw_label, fw_days in FORWARD_WINDOWS.items():
            fwd_ret = compute_forward_returns(prices, fw_days)

            leading_rets = fwd_ret[xlu_leading].dropna()
            lagging_rets = fwd_ret[xlu_lagging].dropna()

            if len(leading_rets) < 10 or len(lagging_rets) < 10:
                continue

            t_stat, p_val = stats.ttest_ind(leading_rets.values, lagging_rets.values, equal_var=False)

            key = f"xlu_{lb_label}_{fw_label}"
            results[key] = {
                "forward": fw_label,
                "lookback": lb_label,
                "leading_mean_pct": round(leading_rets.mean() * 100, 3),
                "leading_median_pct": round(leading_rets.median() * 100, 3),
                "leading_wr": round((leading_rets > 0).mean() * 100, 1),
                "leading_n": len(leading_rets),
                "lagging_mean_pct": round(lagging_rets.mean() * 100, 3),
                "lagging_median_pct": round(lagging_rets.median() * 100, 3),
                "lagging_wr": round((lagging_rets > 0).mean() * 100, 1),
                "lagging_n": len(lagging_rets),
                "diff_pct": round((leading_rets.mean() - lagging_rets.mean()) * 100, 3),
                "t_stat": round(t_stat, 3),
                "p_val": round(p_val, 4),
            }

            print(f"\n  Forward {fw_label}:")
            print(f"    XLU Leading:  mean={leading_rets.mean()*100:+.2f}%, "
                  f"median={leading_rets.median()*100:+.2f}%, "
                  f"WR={results[key]['leading_wr']:.1f}%, N={len(leading_rets)}")
            print(f"    XLU Lagging:  mean={lagging_rets.mean()*100:+.2f}%, "
                  f"median={lagging_rets.median()*100:+.2f}%, "
                  f"WR={results[key]['lagging_wr']:.1f}%, N={len(lagging_rets)}")
            print(f"    Diff: {results[key]['diff_pct']:+.3f}%, t={t_stat:.2f}, p={p_val:.4f}")

        # VIX analysis during XLU leadership
        if vix is not None:
            aligned = pd.DataFrame({"vix": vix, "xlu_leading": xlu_leading}).dropna()
            if len(aligned) > 50:
                vix_leading = aligned.loc[aligned["xlu_leading"], "vix"]
                vix_lagging = aligned.loc[~aligned["xlu_leading"], "vix"]

                results["vix_during_xlu_leading"] = {
                    "mean_vix_leading": round(vix_leading.mean(), 2),
                    "median_vix_leading": round(vix_leading.median(), 2),
                    "mean_vix_lagging": round(vix_lagging.mean(), 2),
                    "median_vix_lagging": round(vix_lagging.median(), 2),
                    "pct_vix_above_20_leading": round((vix_leading > 20).mean() * 100, 1),
                    "pct_vix_above_20_lagging": round((vix_lagging > 20).mean() * 100, 1),
                    "pct_vix_above_25_leading": round((vix_leading > 25).mean() * 100, 1),
                    "pct_vix_above_25_lagging": round((vix_lagging > 25).mean() * 100, 1),
                }

                print(f"\n  VIX During XLU Leadership (lookback {lb_label}):")
                print(f"    XLU Leading:  mean VIX={vix_leading.mean():.1f}, "
                      f"median={vix_leading.median():.1f}, "
                      f"VIX>20: {(vix_leading>20).mean()*100:.1f}%, "
                      f"VIX>25: {(vix_leading>25).mean()*100:.1f}%")
                print(f"    XLU Lagging:  mean VIX={vix_lagging.mean():.1f}, "
                      f"median={vix_lagging.median():.1f}, "
                      f"VIX>20: {(vix_lagging>20).mean()*100:.1f}%, "
                      f"VIX>25: {(vix_lagging>25).mean()*100:.1f}%")

        # Drawdown analysis: max drawdown in forward 3m window
        spy_prices = prices[BENCHMARK]
        dd_window = 63  # 3 months
        fwd_maxdd = pd.Series(index=spy_prices.index, dtype=float)
        for i in range(len(spy_prices) - dd_window):
            window_prices = spy_prices.iloc[i : i + dd_window + 1]
            running_max = window_prices.cummax()
            drawdowns = (window_prices - running_max) / running_max
            fwd_maxdd.iloc[i] = drawdowns.min()

        aligned_dd = pd.DataFrame({"maxdd": fwd_maxdd, "xlu_leading": xlu_leading}).dropna()
        if len(aligned_dd) > 50:
            dd_leading = aligned_dd.loc[aligned_dd["xlu_leading"], "maxdd"]
            dd_lagging = aligned_dd.loc[~aligned_dd["xlu_leading"], "maxdd"]

            results[f"drawdown_{lb_label}"] = {
                "lookback": lb_label,
                "mean_maxdd_leading_pct": round(dd_leading.mean() * 100, 2),
                "mean_maxdd_lagging_pct": round(dd_lagging.mean() * 100, 2),
                "median_maxdd_leading_pct": round(dd_leading.median() * 100, 2),
                "median_maxdd_lagging_pct": round(dd_lagging.median() * 100, 2),
                "pct_dd_gt5_leading": round((dd_leading < -0.05).mean() * 100, 1),
                "pct_dd_gt5_lagging": round((dd_lagging < -0.05).mean() * 100, 1),
            }

            print(f"\n  Forward 3m Max Drawdown (lookback {lb_label}):")
            print(f"    XLU Leading:  mean maxDD={dd_leading.mean()*100:.2f}%, "
                  f"median={dd_leading.median()*100:.2f}%, "
                  f"DD>5%: {(dd_leading<-0.05).mean()*100:.1f}%")
            print(f"    XLU Lagging:  mean maxDD={dd_lagging.mean()*100:.2f}%, "
                  f"median={dd_lagging.median()*100:.2f}%, "
                  f"DD>5%: {(dd_lagging<-0.05).mean()*100:.1f}%")

    # XLU + XLP leading together (defensive combo)
    print("\n--- Defensive Combo: XLU + XLP Both Leading ---")
    for lb_label, lb_days in LOOKBACK_WINDOWS.items():
        flags = get_leadership_flags(prices, lb_days, TOP_N_LEADER)
        if "XLU" not in flags.columns or "XLP" not in flags.columns:
            continue
        both_leading = flags["XLU"] & flags["XLP"]
        neither = ~flags["XLU"] & ~flags["XLP"]

        for fw_label, fw_days in FORWARD_WINDOWS.items():
            fwd_ret = compute_forward_returns(prices, fw_days)
            both_rets = fwd_ret[both_leading].dropna()
            neither_rets = fwd_ret[neither].dropna()
            if len(both_rets) < 10:
                continue

            key = f"xlu_xlp_combo_{lb_label}_{fw_label}"
            results[key] = {
                "lookback": lb_label,
                "forward": fw_label,
                "both_leading_mean_pct": round(both_rets.mean() * 100, 3),
                "both_leading_wr": round((both_rets > 0).mean() * 100, 1),
                "both_leading_n": len(both_rets),
                "neither_mean_pct": round(neither_rets.mean() * 100, 3),
                "neither_wr": round((neither_rets > 0).mean() * 100, 1),
                "neither_n": len(neither_rets),
            }

            print(f"  LB={lb_label} FW={fw_label}: "
                  f"Both lead: {both_rets.mean()*100:+.2f}% (N={len(both_rets)}), "
                  f"Neither: {neither_rets.mean()*100:+.2f}% (N={len(neither_rets)})")

    return results


# ---------------------------------------------------------------------------
# 3. Sector Rotation Signals — Ranking
# ---------------------------------------------------------------------------
def sector_rotation_ranking(prices: pd.DataFrame) -> dict:
    """Rank all sectors by predictive power for forward SPY returns."""
    print("\n" + "=" * 70)
    print("3. SECTOR ROTATION SIGNALS — PREDICTIVE POWER RANKING")
    print("=" * 70)

    results = {}
    ranking_rows = []

    for lb_label, lb_days in LOOKBACK_WINDOWS.items():
        flags = get_leadership_flags(prices, lb_days, TOP_N_LEADER)

        for fw_label, fw_days in FORWARD_WINDOWS.items():
            fwd_ret = compute_forward_returns(prices, fw_days)
            uncond_mean = fwd_ret.dropna().mean()

            sector_scores = []
            for sector in SECTOR_ETFS.keys():
                if sector not in flags.columns:
                    continue
                cond = fwd_ret[flags[sector]].dropna()
                anti = fwd_ret[~flags[sector]].dropna()
                if len(cond) < 20 or len(anti) < 20:
                    continue

                diff = cond.mean() - anti.mean()
                t_stat, p_val = stats.ttest_ind(cond.values, anti.values, equal_var=False)

                sector_scores.append({
                    "sector": sector,
                    "name": SECTOR_ETFS[sector],
                    "lookback": lb_label,
                    "forward": fw_label,
                    "leading_mean_pct": round(cond.mean() * 100, 3),
                    "lagging_mean_pct": round(anti.mean() * 100, 3),
                    "diff_pct": round(diff * 100, 3),
                    "t_stat": round(t_stat, 3),
                    "p_val": round(p_val, 4),
                    "abs_t": abs(t_stat),
                })
                ranking_rows.append(sector_scores[-1])

            # Sort by absolute t-stat
            sector_scores.sort(key=lambda x: x["abs_t"], reverse=True)

            print(f"\n--- Lookback {lb_label}, Forward {fw_label} (sorted by |t-stat|) ---")
            print(f"  {'Rank':>4} {'Sector':<8} {'Name':<18} {'Lead%':>8} {'Lag%':>8} "
                  f"{'Diff%':>8} {'t-stat':>8} {'p':>8}")
            print(f"  {'-'*80}")
            for i, s in enumerate(sector_scores, 1):
                flag = " ***" if s["abs_t"] > 2.0 else (" **" if s["abs_t"] > 1.65 else "")
                direction = "RISK-OFF" if s["diff_pct"] < 0 and s["sector"] in ("XLU", "XLP") else ""
                direction = "RISK-ON" if s["diff_pct"] > 0 and s["sector"] in ("XLK", "XLY") else direction
                print(f"  {i:>4} {s['sector']:<8} {s['name']:<18} {s['leading_mean_pct']:>8.2f} "
                      f"{s['lagging_mean_pct']:>8.2f} {s['diff_pct']:>8.2f} {s['t_stat']:>8.2f} "
                      f"{s['p_val']:>8.4f}{flag} {direction}")

    results["rankings"] = ranking_rows

    # Cross-sector confirmation signals
    print("\n--- Cross-Sector Confirmation Signals ---")
    combos = [
        ("XLU", "XLP", "Defensives leading"),
        ("XLK", "XLY", "Risk-on leading"),
        ("XLE", "XLB", "Commodities leading"),
        ("XLF", "XLI", "Cyclicals leading"),
    ]
    combo_results = []
    for s1, s2, label in combos:
        for lb_label, lb_days in LOOKBACK_WINDOWS.items():
            flags = get_leadership_flags(prices, lb_days, TOP_N_LEADER)
            if s1 not in flags.columns or s2 not in flags.columns:
                continue
            both = flags[s1] & flags[s2]

            for fw_label, fw_days in FORWARD_WINDOWS.items():
                fwd_ret = compute_forward_returns(prices, fw_days)
                cond = fwd_ret[both].dropna()
                anti = fwd_ret[~both].dropna()
                if len(cond) < 10:
                    continue
                t_stat, p_val = stats.ttest_ind(cond.values, anti.values, equal_var=False)

                combo_results.append({
                    "combo": f"{s1}+{s2}",
                    "label": label,
                    "lookback": lb_label,
                    "forward": fw_label,
                    "combo_mean_pct": round(cond.mean() * 100, 3),
                    "other_mean_pct": round(anti.mean() * 100, 3),
                    "diff_pct": round((cond.mean() - anti.mean()) * 100, 3),
                    "t_stat": round(t_stat, 3),
                    "p_val": round(p_val, 4),
                    "n": len(cond),
                })

                sig = " ***" if abs(t_stat) > 2.0 else ""
                print(f"  {label} ({s1}+{s2}) LB={lb_label} FW={fw_label}: "
                      f"combo={cond.mean()*100:+.2f}%, other={anti.mean()*100:+.2f}%, "
                      f"t={t_stat:.2f}, N={len(cond)}{sig}")

    results["cross_sector_combos"] = combo_results
    return results


# ---------------------------------------------------------------------------
# 4. Adversarial Checks (HC #705)
# ---------------------------------------------------------------------------
def adversarial_checks(prices: pd.DataFrame) -> dict:
    """
    HC #705 mandatory:
    - Permutation test
    - Sub-period consistency (2012-2018 vs 2019-2026)
    - Bull/bear regime test
    """
    print("\n" + "=" * 70)
    print("4. ADVERSARIAL CHECKS (HC #705)")
    print("=" * 70)

    results = {}

    # Focus on the most interesting signal: XLU leading, 3m lookback, various forwards
    lb_days = LOOKBACK_WINDOWS["3m"]
    flags = get_leadership_flags(prices, lb_days, TOP_N_LEADER)
    if "XLU" not in flags.columns:
        print("  XLU not available, skipping adversarial checks")
        return results

    for fw_label, fw_days in FORWARD_WINDOWS.items():
        fwd_ret = compute_forward_returns(prices, fw_days)
        xlu_leading = flags["XLU"]

        cond = fwd_ret[xlu_leading].dropna()
        anti = fwd_ret[~xlu_leading].dropna()
        if len(cond) < 20:
            continue

        real_diff = cond.mean() - anti.mean()
        real_t, _ = stats.ttest_ind(cond.values, anti.values, equal_var=False)

        # --- 4a. Permutation Test ---
        print(f"\n  --- Permutation Test: XLU leading, 3m LB, {fw_label} forward ---")
        all_vals = fwd_ret.dropna()
        valid_idx = all_vals.index.intersection(xlu_leading.dropna().index)
        all_vals = all_vals.loc[valid_idx]
        label_vals = xlu_leading.loc[valid_idx]
        n_leading = label_vals.sum()

        perm_diffs = np.zeros(N_PERMUTATIONS)
        for i in range(N_PERMUTATIONS):
            shuffled = np.random.permutation(label_vals.values)
            perm_cond = all_vals.values[shuffled.astype(bool)]
            perm_anti = all_vals.values[~shuffled.astype(bool)]
            perm_diffs[i] = perm_cond.mean() - perm_anti.mean()

        # Two-sided p-value
        perm_p = (np.abs(perm_diffs) >= np.abs(real_diff)).mean()
        print(f"    Real diff: {real_diff*100:+.3f}%, Permutation p-value: {perm_p:.4f} "
              f"({'SURVIVES' if perm_p < 0.10 else 'FAILS'} at 10% level)")

        # --- 4b. Sub-period Consistency ---
        print(f"\n  --- Sub-period Consistency ---")
        midpoint = pd.Timestamp("2019-01-01")
        early_mask = fwd_ret.index < midpoint
        late_mask = fwd_ret.index >= midpoint

        sub_results = {}
        for period_name, period_mask in [("2012-2018", early_mask), ("2019-2026", late_mask)]:
            p_cond = fwd_ret[xlu_leading & period_mask].dropna()
            p_anti = fwd_ret[~xlu_leading & period_mask].dropna()
            if len(p_cond) < 10 or len(p_anti) < 10:
                sub_results[period_name] = {"n": len(p_cond), "status": "insufficient data"}
                continue

            p_diff = p_cond.mean() - p_anti.mean()
            p_t, p_p = stats.ttest_ind(p_cond.values, p_anti.values, equal_var=False)
            sub_results[period_name] = {
                "diff_pct": round(p_diff * 100, 3),
                "t_stat": round(p_t, 3),
                "p_val": round(p_p, 4),
                "n_leading": len(p_cond),
                "n_lagging": len(p_anti),
                "same_sign": np.sign(p_diff) == np.sign(real_diff),
            }
            sign_match = "SAME SIGN" if sub_results[period_name]["same_sign"] else "SIGN FLIP"
            print(f"    {period_name}: diff={p_diff*100:+.3f}%, t={p_t:.2f}, p={p_p:.4f}, "
                  f"N={len(p_cond)} — {sign_match}")

        both_same = all(
            v.get("same_sign", False)
            for v in sub_results.values()
            if isinstance(v.get("same_sign"), bool)
        )
        print(f"    Sub-period consistency: {'PASS' if both_same else 'FAIL'}")

        # --- 4c. Bull/Bear Regime Test (R1) ---
        print(f"\n  --- Bull/Bear Regime Test (R1) ---")
        spy_6m_ret = prices[BENCHMARK].pct_change(126)
        bull_mask = spy_6m_ret > 0
        bear_mask = spy_6m_ret <= 0

        regime_results = {}
        for regime_name, regime_mask in [("Bull (6m SPY>0)", bull_mask), ("Bear (6m SPY<=0)", bear_mask)]:
            valid = xlu_leading.index.intersection(regime_mask.dropna().index).intersection(fwd_ret.dropna().index)
            r_cond = fwd_ret.loc[valid][xlu_leading.loc[valid] & regime_mask.loc[valid]].dropna()
            r_anti = fwd_ret.loc[valid][~xlu_leading.loc[valid] & regime_mask.loc[valid]].dropna()
            if len(r_cond) < 10 or len(r_anti) < 10:
                regime_results[regime_name] = {"n": len(r_cond), "status": "insufficient data"}
                continue

            r_diff = r_cond.mean() - r_anti.mean()
            r_t, r_p = stats.ttest_ind(r_cond.values, r_anti.values, equal_var=False)
            regime_results[regime_name] = {
                "diff_pct": round(r_diff * 100, 3),
                "t_stat": round(r_t, 3),
                "p_val": round(r_p, 4),
                "n_leading": len(r_cond),
                "n_lagging": len(r_anti),
                "same_sign": np.sign(r_diff) == np.sign(real_diff),
            }
            sign_match = "SAME SIGN" if regime_results[regime_name]["same_sign"] else "SIGN FLIP"
            print(f"    {regime_name}: diff={r_diff*100:+.3f}%, t={r_t:.2f}, p={r_p:.4f}, "
                  f"N_lead={len(r_cond)}, N_lag={len(r_anti)} — {sign_match}")

        # R1 check: regime-tailored rejection
        bull_diff = regime_results.get("Bull (6m SPY>0)", {}).get("diff_pct", 0)
        bear_diff = regime_results.get("Bear (6m SPY<=0)", {}).get("diff_pct", 0)
        if bull_diff != 0 and bear_diff != 0:
            asymmetry = abs(bull_diff - bear_diff) / max(abs(bull_diff), abs(bear_diff))
            r1_pass = asymmetry <= 0.50
            print(f"    R1 regime asymmetry: {asymmetry:.2f} ({'PASS' if r1_pass else 'FAIL — regime-tailored'})")
        else:
            r1_pass = None
            asymmetry = None

        results[fw_label] = {
            "real_diff_pct": round(real_diff * 100, 3),
            "real_t_stat": round(real_t, 3),
            "permutation_p": round(perm_p, 4),
            "permutation_survives": perm_p < 0.10,
            "sub_period": sub_results,
            "sub_period_consistent": both_same,
            "regime": regime_results,
            "r1_asymmetry": round(asymmetry, 3) if asymmetry is not None else None,
            "r1_pass": r1_pass,
        }

    # Outlier removal check — remove top/bottom 2% of returns and re-test
    print(f"\n  --- Outlier Removal Robustness (remove top/bottom 2%) ---")
    lb_days = LOOKBACK_WINDOWS["3m"]
    flags = get_leadership_flags(prices, lb_days, TOP_N_LEADER)

    for fw_label, fw_days in FORWARD_WINDOWS.items():
        fwd_ret = compute_forward_returns(prices, fw_days)
        xlu_leading = flags["XLU"]

        # Remove outliers
        q_low = fwd_ret.quantile(0.02)
        q_high = fwd_ret.quantile(0.98)
        trimmed = fwd_ret[(fwd_ret >= q_low) & (fwd_ret <= q_high)]

        cond = trimmed[xlu_leading].dropna()
        anti = trimmed[~xlu_leading].dropna()
        if len(cond) < 10:
            continue

        diff = cond.mean() - anti.mean()
        t_stat, p_val = stats.ttest_ind(cond.values, anti.values, equal_var=False)
        orig_diff = results.get(fw_label, {}).get("real_diff_pct", 0)
        same_sign = np.sign(diff) == np.sign(orig_diff / 100) if orig_diff != 0 else True

        print(f"    {fw_label}: trimmed diff={diff*100:+.3f}%, t={t_stat:.2f} — "
              f"{'ROBUST' if same_sign else 'FRAGILE (sign flipped)'}")

        if fw_label in results:
            results[fw_label]["outlier_trimmed_diff_pct"] = round(diff * 100, 3)
            results[fw_label]["outlier_trimmed_robust"] = bool(same_sign)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    np.random.seed(42)
    print("=" * 70)
    print("SECTOR LEADERSHIP AS SPY PREDICTOR — COMPREHENSIVE STUDY")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    prices, vix = download_data(years=14)

    # Run all analyses
    leadership_results = sector_leadership_analysis(prices)
    utilities_results = utilities_deep_dive(prices, vix)
    rotation_results = sector_rotation_ranking(prices)
    adversarial_results = adversarial_checks(prices)

    # --- Summary ---
    print("\n" + "=" * 70)
    print("5. KEY FINDINGS SUMMARY")
    print("=" * 70)

    # Find strongest predictors
    all_rankings = rotation_results.get("rankings", [])
    if all_rankings:
        # Best positive predictor (sector leading = higher SPY returns)
        positive = [r for r in all_rankings if r["diff_pct"] > 0]
        negative = [r for r in all_rankings if r["diff_pct"] < 0]

        if positive:
            best_pos = max(positive, key=lambda x: x["abs_t"])
            print(f"\n  Strongest POSITIVE predictor: {best_pos['sector']} ({best_pos['name']}) leading")
            print(f"    LB={best_pos['lookback']}, FW={best_pos['forward']}: "
                  f"diff={best_pos['diff_pct']:+.2f}%, t={best_pos['t_stat']:.2f}")

        if negative:
            best_neg = max(negative, key=lambda x: x["abs_t"])
            print(f"\n  Strongest NEGATIVE predictor: {best_neg['sector']} ({best_neg['name']}) leading")
            print(f"    LB={best_neg['lookback']}, FW={best_neg['forward']}: "
                  f"diff={best_neg['diff_pct']:+.2f}%, t={best_neg['t_stat']:.2f}")

    # XLU summary
    print(f"\n  UTILITIES (XLU) VERDICT:")
    for fw_label in FORWARD_WINDOWS.keys():
        key = f"xlu_3m_{fw_label}"
        if key in utilities_results:
            r = utilities_results[key]
            print(f"    FW {fw_label}: leading={r['leading_mean_pct']:+.2f}% vs "
                  f"lagging={r['lagging_mean_pct']:+.2f}%, t={r['t_stat']:.2f}")

    # Adversarial summary
    print(f"\n  ADVERSARIAL CHECK SUMMARY (XLU, 3m lookback):")
    for fw_label, adv in adversarial_results.items():
        perm = "PASS" if adv.get("permutation_survives") else "FAIL"
        sub = "PASS" if adv.get("sub_period_consistent") else "FAIL"
        r1 = "PASS" if adv.get("r1_pass") else ("FAIL" if adv.get("r1_pass") is False else "N/A")
        outlier = "ROBUST" if adv.get("outlier_trimmed_robust") else "FRAGILE"
        print(f"    FW {fw_label}: Permutation={perm}, SubPeriod={sub}, R1={r1}, Outlier={outlier}")

    # Save full results
    full_summary = {
        "metadata": {
            "date": datetime.now().isoformat(),
            "data_range": f"{prices.index[0].strftime('%Y-%m-%d')} to {prices.index[-1].strftime('%Y-%m-%d')}",
            "n_trading_days": len(prices),
            "lookback_windows": LOOKBACK_WINDOWS,
            "forward_windows": FORWARD_WINDOWS,
            "top_n_leader": TOP_N_LEADER,
            "n_permutations": N_PERMUTATIONS,
        },
        "sector_leadership": leadership_results,
        "utilities_deep_dive": utilities_results,
        "rotation_rankings": rotation_results,
        "adversarial_checks": adversarial_results,
    }

    # JSON summary
    summary_path = OUTPUT_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(full_summary, f, indent=2, default=str)
    print(f"\n  Saved JSON summary to {summary_path}")

    # CSV of all rankings
    if all_rankings:
        df_rankings = pd.DataFrame(all_rankings)
        csv_path = OUTPUT_DIR / "sector_leadership_rankings.csv"
        df_rankings.to_csv(csv_path, index=False)
        print(f"  Saved rankings CSV to {csv_path}")

    # CSV of XLU details
    xlu_rows = []
    for k, v in utilities_results.items():
        if isinstance(v, dict) and "forward" in v:
            xlu_rows.append(v)
    if xlu_rows:
        df_xlu = pd.DataFrame(xlu_rows)
        csv_path = OUTPUT_DIR / "xlu_deep_dive.csv"
        df_xlu.to_csv(csv_path, index=False)
        print(f"  Saved XLU deep dive CSV to {csv_path}")

    print("\n  DONE.")


if __name__ == "__main__":
    main()

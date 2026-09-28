#!/usr/bin/env python3
"""
VIX Threshold Optimizer for Leveraged Growth Strategy
=====================================================

Granular sweep of VIX-gated parameters to find Pareto-optimal configs
that maximize CAGR while keeping MaxDD <= 25%.

Tests: UPRO and TQQQ with varying:
  - VIX scale-down start thresholds
  - VIX go-to-cash thresholds
  - Scale-down allocation percentages
  - Trailing stop overlays

HC #705 adversarial checks on top 5 configs only (CPU budget).
"""

import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from itertools import product
import json
import warnings
import time
from datetime import datetime

warnings.filterwarnings("ignore")
np.random.seed(42)

OUT_DIR = Path("/home/jupiter/Lvl3Quant/output/growth_leverage_research")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RISK_FREE_RATE = 0.05


# ─── Data Download ───────────────────────────────────────────────────────
def download_data(start="2010-01-01"):
    """Download price data for UPRO, TQQQ, TMF, SHV, SPY, VIX."""
    end = datetime.now().strftime("%Y-%m-%d")
    tickers = ["UPRO", "TQQQ", "TMF", "SHV", "SPY", "^VIX"]
    print(f"Downloading {tickers} from {start} to {end}...")

    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = {}
    if isinstance(data.columns, pd.MultiIndex):
        for t in tickers:
            if t in data["Close"].columns:
                prices[t] = data["Close"][t]
    df = pd.DataFrame(prices).ffill().dropna(how="all")
    print(f"  {len(df)} trading days, {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
    return df


# ─── Metrics ─────────────────────────────────────────────────────────────
def compute_metrics(returns, name=""):
    """Full metrics suite."""
    returns = returns.dropna()
    if len(returns) < 252:
        return None

    cum = (1 + returns).cumprod()
    years = len(returns) / 252
    cagr = cum.iloc[-1] ** (1 / years) - 1

    ann_vol = returns.std() * np.sqrt(252)
    sharpe = (cagr - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else 0

    downside = returns[returns < 0].std() * np.sqrt(252)
    sortino = (cagr - RISK_FREE_RATE) / downside if downside > 0 else 0

    rolling_max = cum.cummax()
    drawdown = cum / rolling_max - 1
    max_dd = drawdown.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Per-year returns
    yearly = returns.groupby(returns.index.year).apply(lambda x: (1 + x).prod() - 1)
    worst_year_ret = yearly.min()
    worst_year = int(yearly.idxmin()) if len(yearly) > 0 else None

    # COVID drawdown (Feb-Mar 2020)
    covid_dd = None
    try:
        covid_rets = returns.loc["2020-02-19":"2020-03-23"]
        if len(covid_rets) > 0:
            covid_cum = (1 + covid_rets).cumprod()
            covid_dd = float((covid_cum / covid_cum.cummax() - 1).min())
    except:
        pass

    # 2022 drawdown
    dd_2022 = None
    try:
        rets_2022 = returns.loc["2022-01-01":"2022-12-31"]
        if len(rets_2022) > 0:
            cum_2022 = (1 + rets_2022).cumprod()
            dd_2022 = float((cum_2022 / cum_2022.cummax() - 1).min())
    except:
        pass

    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    pf = gains / losses if losses > 0 else float("inf")

    return {
        "name": name,
        "cagr_pct": round(float(cagr * 100), 2),
        "sharpe": round(float(sharpe), 3),
        "sortino": round(float(sortino), 3),
        "max_dd_pct": round(float(max_dd * 100), 2),
        "calmar": round(float(calmar), 3),
        "win_rate_pct": round(float((returns > 0).mean() * 100), 1),
        "profit_factor": round(float(pf), 3),
        "worst_year_pct": round(float(worst_year_ret * 100), 2) if worst_year_ret is not None else None,
        "worst_year": worst_year,
        "covid_dd_pct": round(float(covid_dd * 100), 2) if covid_dd is not None else None,
        "dd_2022_pct": round(float(dd_2022 * 100), 2) if dd_2022 is not None else None,
        "years": round(years, 1),
        "n_days": len(returns),
    }


# ─── VIX-Gated Strategy (parameterized) ─────────────────────────────────
def run_vix_gated(prices, base_etf, vix_scaledown, vix_cash, scaledown_alloc,
                  trailing_stop_pct=None, trailing_stop_cooldown=None):
    """
    Parameterized VIX-gated strategy.

    Args:
        base_etf: "UPRO" or "TQQQ"
        vix_scaledown: VIX level to start reducing allocation
        vix_cash: VIX level to go full cash
        scaledown_alloc: allocation to base_etf in the scale-down zone (rest in SHV)
        trailing_stop_pct: if set, go to cash when portfolio drops this % from peak
        trailing_stop_cooldown: days to stay in cash after trailing stop triggers
    """
    vix = prices["^VIX"].dropna()
    base = prices[base_etf].dropna()
    safe = prices["SHV"].dropna()
    bond = prices["TMF"].dropna()

    common = base.index.intersection(safe.index).intersection(vix.index).intersection(bond.index)
    base_ret = base.pct_change().reindex(common)
    safe_ret = safe.pct_change().reindex(common)
    bond_ret = bond.pct_change().reindex(common)

    portfolio_ret = pd.Series(0.0, index=common[1:])

    # Trailing stop state
    cum_val = 1.0
    peak = 1.0
    stop_triggered = False
    stop_cooldown_remaining = 0

    for date in common[1:]:
        v = vix.loc[date]
        r_b = base_ret.loc[date] if not np.isnan(base_ret.loc[date]) else 0.0
        r_s = safe_ret.loc[date] if not np.isnan(safe_ret.loc[date]) else 0.0
        r_t = bond_ret.loc[date] if not np.isnan(bond_ret.loc[date]) else 0.0

        # Trailing stop check
        if trailing_stop_pct is not None:
            dd_from_peak = (cum_val / peak) - 1
            if dd_from_peak < -trailing_stop_pct:
                stop_triggered = True
                stop_cooldown_remaining = trailing_stop_cooldown or 10

            if stop_triggered:
                stop_cooldown_remaining -= 1
                if stop_cooldown_remaining <= 0:
                    stop_triggered = False
                portfolio_ret.loc[date] = r_s  # cash during stop
                cum_val *= (1 + r_s)
                peak = max(peak, cum_val)
                continue

        # VIX-based allocation
        if v < vix_scaledown:
            # Full risk-on
            ret = 1.0 * r_b
        elif v < vix_cash:
            # Scale down zone: scaledown_alloc in base, rest in TMF (hedge)
            bond_alloc = min(0.3, (1 - scaledown_alloc) * 0.5)
            cash_alloc = 1 - scaledown_alloc - bond_alloc
            ret = scaledown_alloc * r_b + bond_alloc * r_t + cash_alloc * r_s
        else:
            # Full cash
            ret = r_s

        portfolio_ret.loc[date] = ret
        cum_val *= (1 + ret)
        peak = max(peak, cum_val)

    return portfolio_ret


# ─── HC #705 Adversarial Checks ─────────────────────────────────────────
def adversarial_permutation(returns, n_perms=100):
    """Block-bootstrap permutation test (100 shuffles for speed)."""
    actual_sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    rets_array = returns.values.copy()
    n = len(rets_array)
    block_size = 20

    shuffled_sharpes = []
    for _ in range(n_perms):
        n_blocks = n // block_size + 1
        block_indices = np.random.randint(0, max(1, n - block_size), size=n_blocks)
        shuffled = np.concatenate([rets_array[i:i+block_size] for i in block_indices])[:n]
        s = np.mean(shuffled) / np.std(shuffled) * np.sqrt(252) if np.std(shuffled) > 0 else 0
        shuffled_sharpes.append(s)

    p_value = (np.array(shuffled_sharpes) >= actual_sharpe).mean()
    return {
        "actual_sharpe": round(float(actual_sharpe), 3),
        "p_value": round(float(p_value), 4),
        "perm_95th": round(float(np.percentile(shuffled_sharpes, 95)), 3),
        "PASS": bool(p_value < 0.05),
    }


def adversarial_regime(returns, spy_returns):
    """Bull/bear/flat regime split."""
    common = returns.index.intersection(spy_returns.index)
    returns = returns.loc[common]
    spy_returns = spy_returns.loc[common]
    spy_20d = spy_returns.rolling(20).sum()

    results = {}
    for regime, mask in [("bull", spy_20d > 0.02), ("bear", spy_20d < -0.02),
                         ("flat", (spy_20d >= -0.02) & (spy_20d <= 0.02))]:
        r = returns[mask].dropna()
        if len(r) > 20:
            ann_ret = r.mean() * 252
            ann_vol = r.std() * np.sqrt(252)
            results[regime] = round(float(ann_ret / ann_vol if ann_vol > 0 else 0), 3)
        else:
            results[regime] = 0.0

    sharpes = [results[r] for r in ["bull", "bear", "flat"]]
    max_s = max(abs(s) for s in sharpes) if any(s != 0 for s in sharpes) else 1
    gap = (max(sharpes) - min(sharpes)) / max_s if max_s > 0 else 0
    results["gap"] = round(float(gap), 3)
    results["PASS"] = bool(gap < 2.0)
    return results


def adversarial_subperiod(returns):
    """Rolling 1-year window consistency."""
    window = 252
    sharpes = []
    for start in range(0, len(returns) - window, 63):
        chunk = returns.iloc[start:start + window]
        s = chunk.mean() / chunk.std() * np.sqrt(252) if chunk.std() > 0 else 0
        sharpes.append(s)
    sharpes = np.array(sharpes)
    pct_pos = float((sharpes > 0).mean() * 100) if len(sharpes) > 0 else 0
    return {
        "n_windows": len(sharpes),
        "pct_positive": round(pct_pos, 1),
        "min_sharpe": round(float(np.min(sharpes)), 3) if len(sharpes) > 0 else 0,
        "PASS": bool(pct_pos > 55),
    }


def adversarial_outlier(returns, pct=1):
    """Remove top/bottom 1% days, check Sharpe stability."""
    full_s = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    lo, hi = np.percentile(returns, pct), np.percentile(returns, 100 - pct)
    trimmed = returns[(returns >= lo) & (returns <= hi)]
    trim_s = trimmed.mean() / trimmed.std() * np.sqrt(252) if trimmed.std() > 0 else 0
    drop = 1 - (trim_s / full_s) if full_s != 0 else 0
    return {
        "full_sharpe": round(float(full_s), 3),
        "trimmed_sharpe": round(float(trim_s), 3),
        "drop_pct": round(float(drop * 100), 1),
        "PASS": bool(abs(drop) < 0.50),
    }


def run_full_adversarial(returns, spy_returns):
    """Run all 4 HC #705 checks."""
    perm = adversarial_permutation(returns, n_perms=100)
    regime = adversarial_regime(returns, spy_returns)
    sub = adversarial_subperiod(returns)
    outlier = adversarial_outlier(returns)

    n_pass = sum([perm["PASS"], regime["PASS"], sub["PASS"], outlier["PASS"]])
    verdict = "STRONG" if n_pass == 4 else "MARGINAL" if n_pass >= 3 else "WEAK" if n_pass >= 2 else "REJECT"

    return {
        "permutation": perm,
        "regime": regime,
        "subperiod": sub,
        "outlier": outlier,
        "n_pass": n_pass,
        "verdict": verdict,
    }


# ─── Main Sweep ─────────────────────────────────────────────────────────
def main():
    print("=" * 80)
    print("VIX THRESHOLD OPTIMIZER — Pareto frontier for MaxDD <= 25%")
    print(f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    prices = download_data()
    spy_returns = prices["SPY"].pct_change().dropna()

    # ─── Parameter grid ──────────────────────────────────────────────
    etfs = ["UPRO", "TQQQ"]
    vix_scaledowns = [16, 17, 18, 19, 20, 22]
    vix_cash_levels = [25, 28, 30, 35]
    scaledown_allocs = [0.30, 0.40, 0.50, 0.60]

    # Trailing stop configs: (stop_pct, cooldown_days) — None means no stop
    trailing_stops = [
        (None, None),       # no trailing stop
        (0.15, 5),          # 15% stop, 5-day cooldown
        (0.15, 10),         # 15% stop, 10-day cooldown
        (0.15, 20),         # 15% stop, 20-day cooldown
        (0.12, 10),         # 12% stop, 10-day cooldown
        (0.18, 10),         # 18% stop, 10-day cooldown
    ]

    total_configs = len(etfs) * len(vix_scaledowns) * len(vix_cash_levels) * len(scaledown_allocs) * len(trailing_stops)
    print(f"\nSweeping {total_configs} configurations...")
    print(f"  ETFs: {etfs}")
    print(f"  VIX scale-down: {vix_scaledowns}")
    print(f"  VIX go-to-cash: {vix_cash_levels}")
    print(f"  Scale-down alloc: {scaledown_allocs}")
    print(f"  Trailing stops: {trailing_stops}")

    all_results = []
    count = 0
    t0 = time.time()

    for etf in etfs:
        for vix_sd in vix_scaledowns:
            for vix_cash in vix_cash_levels:
                if vix_cash <= vix_sd:
                    continue  # cash threshold must be above scale-down

                for sd_alloc in scaledown_allocs:
                    for ts_pct, ts_cool in trailing_stops:
                        count += 1

                        # Build config name
                        ts_label = f"_TS{int(ts_pct*100)}d{ts_cool}" if ts_pct else ""
                        name = f"{etf}_sd{vix_sd}_cash{vix_cash}_alloc{int(sd_alloc*100)}{ts_label}"

                        rets = run_vix_gated(
                            prices, base_etf=etf,
                            vix_scaledown=vix_sd, vix_cash=vix_cash,
                            scaledown_alloc=sd_alloc,
                            trailing_stop_pct=ts_pct,
                            trailing_stop_cooldown=ts_cool,
                        )

                        m = compute_metrics(rets, name)
                        if m is None:
                            continue

                        m["config"] = {
                            "etf": etf,
                            "vix_scaledown": vix_sd,
                            "vix_cash": vix_cash,
                            "scaledown_alloc": sd_alloc,
                            "trailing_stop_pct": ts_pct,
                            "trailing_stop_cooldown": ts_cool,
                        }
                        all_results.append(m)

                        if count % 100 == 0:
                            elapsed = time.time() - t0
                            print(f"  [{count}/{total_configs}] {elapsed:.0f}s elapsed...")

    elapsed = time.time() - t0
    print(f"\nSwept {count} configs in {elapsed:.1f}s ({len(all_results)} valid)")

    # ─── Filter: MaxDD <= 25% ────────────────────────────────────────
    feasible = [r for r in all_results if r["max_dd_pct"] >= -25.0]
    print(f"\n{len(feasible)} configs have MaxDD <= 25%")

    if not feasible:
        print("WARNING: No configs meet MaxDD constraint. Relaxing to 30%...")
        feasible = [r for r in all_results if r["max_dd_pct"] >= -30.0]
        print(f"  {len(feasible)} configs with MaxDD <= 30%")

    # ─── Sort by CAGR (descending) ──────────────────────────────────
    feasible.sort(key=lambda x: x["cagr_pct"], reverse=True)

    # ─── Pareto frontier: maximize CAGR while minimizing MaxDD ──────
    pareto = []
    best_dd = -100  # track best DD seen so far
    for r in sorted(all_results, key=lambda x: x["cagr_pct"], reverse=True):
        if r["max_dd_pct"] > best_dd:
            pareto.append(r)
            best_dd = r["max_dd_pct"]

    # Filter pareto to only feasible
    pareto_feasible = [r for r in pareto if r["max_dd_pct"] >= -25.0]

    print(f"\nPareto frontier (CAGR vs MaxDD): {len(pareto)} total, {len(pareto_feasible)} with MaxDD <= 25%")

    # ─── Print top 20 feasible by CAGR ───────────────────────────────
    print("\n" + "=" * 120)
    print("TOP 20 CONFIGS (MaxDD <= 25%, sorted by CAGR)")
    print("=" * 120)
    header = f"{'Config':<50} {'CAGR%':>7} {'Sharpe':>7} {'Sortino':>8} {'MaxDD%':>7} {'Calmar':>7} {'WrstYr%':>8} {'COVID':>7} {'2022':>7}"
    print(header)
    print("-" * 120)

    for r in feasible[:20]:
        covid = f"{r['covid_dd_pct']:.1f}" if r['covid_dd_pct'] is not None else "N/A"
        dd22 = f"{r['dd_2022_pct']:.1f}" if r['dd_2022_pct'] is not None else "N/A"
        worst_yr = f"{r['worst_year_pct']:.1f}" if r['worst_year_pct'] is not None else "N/A"
        print(f"{r['name']:<50} {r['cagr_pct']:>7.1f} {r['sharpe']:>7.2f} {r['sortino']:>8.2f} "
              f"{r['max_dd_pct']:>7.1f} {r['calmar']:>7.2f} {worst_yr:>8} {covid:>7} {dd22:>7}")

    # ─── Print top 10 by Sharpe ──────────────────────────────────────
    print("\n" + "=" * 120)
    print("TOP 10 CONFIGS (MaxDD <= 25%, sorted by Sharpe)")
    print("=" * 120)
    by_sharpe = sorted(feasible, key=lambda x: x["sharpe"], reverse=True)
    print(header)
    print("-" * 120)
    for r in by_sharpe[:10]:
        covid = f"{r['covid_dd_pct']:.1f}" if r['covid_dd_pct'] is not None else "N/A"
        dd22 = f"{r['dd_2022_pct']:.1f}" if r['dd_2022_pct'] is not None else "N/A"
        worst_yr = f"{r['worst_year_pct']:.1f}" if r['worst_year_pct'] is not None else "N/A"
        print(f"{r['name']:<50} {r['cagr_pct']:>7.1f} {r['sharpe']:>7.2f} {r['sortino']:>8.2f} "
              f"{r['max_dd_pct']:>7.1f} {r['calmar']:>7.2f} {worst_yr:>8} {covid:>7} {dd22:>7}")

    # ─── Trailing stop analysis ──────────────────────────────────────
    print("\n" + "=" * 80)
    print("TRAILING STOP IMPACT ANALYSIS")
    print("=" * 80)

    # Compare best no-stop vs best with-stop for each ETF
    for etf in etfs:
        no_stop = [r for r in feasible if r["config"]["etf"] == etf and r["config"]["trailing_stop_pct"] is None]
        with_stop = [r for r in feasible if r["config"]["etf"] == etf and r["config"]["trailing_stop_pct"] is not None]

        print(f"\n  {etf}:")
        if no_stop:
            best_ns = max(no_stop, key=lambda x: x["cagr_pct"])
            print(f"    Best w/o stop: {best_ns['name']} — CAGR={best_ns['cagr_pct']:.1f}% MaxDD={best_ns['max_dd_pct']:.1f}% Sharpe={best_ns['sharpe']:.2f}")
        if with_stop:
            best_ws = max(with_stop, key=lambda x: x["cagr_pct"])
            print(f"    Best w/  stop: {best_ws['name']} — CAGR={best_ws['cagr_pct']:.1f}% MaxDD={best_ws['max_dd_pct']:.1f}% Sharpe={best_ws['sharpe']:.2f}")
            best_ws_sharpe = max(with_stop, key=lambda x: x["sharpe"])
            print(f"    Best Sharpe w/ stop: {best_ws_sharpe['name']} — CAGR={best_ws_sharpe['cagr_pct']:.1f}% MaxDD={best_ws_sharpe['max_dd_pct']:.1f}% Sharpe={best_ws_sharpe['sharpe']:.2f}")

    # ─── HC #705 Adversarial on top 5 ────────────────────────────────
    print("\n" + "=" * 80)
    print("HC #705 ADVERSARIAL VALIDATION — TOP 5 CONFIGS")
    print("=" * 80)

    # Pick top 5 unique configs (mix of CAGR and Sharpe leaders)
    seen = set()
    top5 = []
    for r in feasible[:3]:  # top 3 by CAGR
        if r["name"] not in seen:
            top5.append(r)
            seen.add(r["name"])
    for r in by_sharpe[:3]:  # top 3 by Sharpe
        if r["name"] not in seen:
            top5.append(r)
            seen.add(r["name"])
    top5 = top5[:5]

    adversarial_results = {}
    for r in top5:
        name = r["name"]
        cfg = r["config"]
        print(f"\n--- {name} ---")

        rets = run_vix_gated(
            prices, base_etf=cfg["etf"],
            vix_scaledown=cfg["vix_scaledown"], vix_cash=cfg["vix_cash"],
            scaledown_alloc=cfg["scaledown_alloc"],
            trailing_stop_pct=cfg["trailing_stop_pct"],
            trailing_stop_cooldown=cfg["trailing_stop_cooldown"],
        )

        adv = run_full_adversarial(rets, spy_returns)

        print(f"  Permutation: {'PASS' if adv['permutation']['PASS'] else 'FAIL'} (p={adv['permutation']['p_value']})")
        print(f"  Regime:      {'PASS' if adv['regime']['PASS'] else 'FAIL'} (bull={adv['regime'].get('bull','?')}, bear={adv['regime'].get('bear','?')}, flat={adv['regime'].get('flat','?')}, gap={adv['regime']['gap']})")
        print(f"  Sub-period:  {'PASS' if adv['subperiod']['PASS'] else 'FAIL'} ({adv['subperiod']['pct_positive']}% positive, min={adv['subperiod']['min_sharpe']})")
        print(f"  Outlier:     {'PASS' if adv['outlier']['PASS'] else 'FAIL'} (drop={adv['outlier']['drop_pct']}%)")
        print(f"  VERDICT: {adv['verdict']} ({adv['n_pass']}/4)")

        adversarial_results[name] = {
            "metrics": r,
            "adversarial": adv,
        }

    # ─── Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    strong = [(n, d) for n, d in adversarial_results.items() if d["adversarial"]["verdict"] in ("STRONG", "MARGINAL")]
    if strong:
        print(f"\n{len(strong)} configs pass adversarial validation with MaxDD <= 25%:")
        for name, data in strong:
            m = data["metrics"]
            print(f"  {name}")
            print(f"    CAGR={m['cagr_pct']}% | Sharpe={m['sharpe']} | Sortino={m['sortino']} | MaxDD={m['max_dd_pct']}% | Calmar={m['calmar']}")
            print(f"    Worst year={m['worst_year_pct']}% ({m['worst_year']}) | COVID DD={m['covid_dd_pct']}% | 2022 DD={m['dd_2022_pct']}%")
            print(f"    Adversarial: {data['adversarial']['verdict']} ({data['adversarial']['n_pass']}/4)")
    else:
        print("\nNo configs pass both MaxDD constraint and adversarial validation.")
        print("Best available:")
        for name, data in adversarial_results.items():
            m = data["metrics"]
            print(f"  {name}: CAGR={m['cagr_pct']}% MaxDD={m['max_dd_pct']}% {data['adversarial']['verdict']}")

    # ─── Save results ────────────────────────────────────────────────
    output = {
        "run_time": datetime.now().isoformat(),
        "total_configs_tested": count,
        "feasible_configs": len(feasible),
        "parameter_grid": {
            "etfs": etfs,
            "vix_scaledowns": vix_scaledowns,
            "vix_cash_levels": vix_cash_levels,
            "scaledown_allocs": scaledown_allocs,
            "trailing_stops": [(ts_pct, ts_cool) for ts_pct, ts_cool in trailing_stops],
        },
        "top_20_by_cagr": feasible[:20],
        "top_10_by_sharpe": by_sharpe[:10],
        "pareto_frontier": pareto_feasible[:15],
        "adversarial_validated": {
            name: {
                "metrics": data["metrics"],
                "adversarial": data["adversarial"],
            }
            for name, data in adversarial_results.items()
        },
        "all_results_summary": [
            {
                "name": r["name"],
                "cagr_pct": r["cagr_pct"],
                "sharpe": r["sharpe"],
                "max_dd_pct": r["max_dd_pct"],
                "calmar": r["calmar"],
            }
            for r in sorted(all_results, key=lambda x: x["sharpe"], reverse=True)
        ],
    }

    out_path = OUT_DIR / "threshold_optimization.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {out_path}")
    print("DONE.")


if __name__ == "__main__":
    main()

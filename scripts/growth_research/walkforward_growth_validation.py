#!/usr/bin/env python3
"""
Walk-Forward Validation: Dual Momentum + Breakout Trend Following
=================================================================
HC #697: Everything must be walk-forward predictive.
HC #0: Sliding windows only, never expanding.

Method: 36-month train, 6-month OOT test, slide 6 months.
In each train window, optimize parameters. Apply best to OOT.
Concatenate all OOT returns for aggregate metrics.

Adversarial checks (HC #705):
  - Permutation test on concatenated OOT returns
  - R1 regime test (green/red/flat SPY days)
  - Sub-period consistency
  - Outlier removal robustness
"""

import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from itertools import product
import json, warnings, time
warnings.filterwarnings("ignore")

OUT_DIR = Path("/home/jupiter/Lvl3Quant/output/growth_research/walkforward_validation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Universe: leveraged + unleveraged ETFs
UNIVERSE = ["TQQQ", "QQQ", "SPY", "SOXL"]
SAFE_ASSET = "SHV"
SPREAD_BPS = {"TQQQ": 1, "QQQ": 1, "SPY": 1, "SOXL": 2, "SHV": 1}

# Walk-forward config
TRAIN_MONTHS = 36
TEST_MONTHS = 6

# Parameter grids
DM_LOOKBACKS = [10, 15, 21, 42, 63]
DM_VOL_TARGETS = [0.20, 0.25, 0.30, 0.40]
BO_BREAKOUT_DAYS = [5, 10, 15, 20]
BO_TRAIL_DAYS = [5, 8, 10, 15]


def download_data():
    """Download ETF prices."""
    tickers = UNIVERSE + [SAFE_ASSET]
    print(f"Downloading {tickers}...")
    data = yf.download(tickers, start="2010-01-01", end="2026-07-15",
                       auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data
    prices = prices.ffill().dropna(how="all")
    print(f"  {len(prices)} trading days, {prices.shape[1]} tickers")
    return prices


# ── DUAL MOMENTUM ──────────────────────────────────────────────────

def run_dual_momentum(prices, etfs, lookback, vol_target, start, end):
    """Run dual momentum on a date range. Returns daily return series."""
    returns = prices.pct_change()
    mask = (prices.index >= start) & (prices.index <= end)
    dates = prices.index[mask]
    daily_rets = []
    current = None

    for date in dates:
        loc = prices.index.get_loc(date)
        if loc < lookback + 1:
            daily_rets.append(0.0)
            continue

        # Compute momentum for each ETF
        mom = {}
        for etf in etfs:
            if etf not in prices.columns:
                continue
            p_now = prices[etf].iloc[loc]
            p_past = prices[etf].iloc[loc - lookback]
            if p_past > 0 and not np.isnan(p_past) and not np.isnan(p_now):
                mom[etf] = p_now / p_past - 1

        if not mom:
            daily_rets.append(0.0)
            continue

        best_etf = max(mom, key=mom.get)
        target = best_etf if mom[best_etf] > 0 else SAFE_ASSET

        # Cost on switch
        cost = 0.0
        if target != current:
            if current and current in SPREAD_BPS:
                cost += SPREAD_BPS[current] / 10000
            if target in SPREAD_BPS:
                cost += SPREAD_BPS[target] / 10000
            current = target

        # Daily return
        r = returns[current].iloc[loc] if current in returns.columns else 0.0
        if np.isnan(r):
            r = 0.0

        # Vol targeting
        if vol_target > 0 and current in returns.columns:
            rv = returns[current].iloc[max(0, loc-21):loc].std() * np.sqrt(252)
            if rv > 0:
                scalar = min(vol_target / rv, 2.0)
                r *= scalar
                cost *= scalar

        daily_rets.append(r - cost)

    return pd.Series(daily_rets, index=dates)


def optimize_dual_momentum(prices, etfs, train_start, train_end):
    """Find best (lookback, vol_target) on train window by Sharpe."""
    best_sharpe = -999
    best_params = (21, 0.30)

    for lb, vt in product(DM_LOOKBACKS, DM_VOL_TARGETS):
        dr = run_dual_momentum(prices, etfs, lb, vt, train_start, train_end)
        if len(dr) < 60 or dr.std() == 0:
            continue
        sharpe = dr.mean() / dr.std() * np.sqrt(252)
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_params = (lb, vt)

    return best_params, best_sharpe


# ── BREAKOUT TREND FOLLOWING ───────────────────────────────────────

def run_breakout(prices, etfs, breakout_days, trail_days, start, end):
    """Breakout strategy: buy on N-day high, trail stop on N-day low."""
    returns = prices.pct_change()
    mask = (prices.index >= start) & (prices.index <= end)
    dates = prices.index[mask]
    daily_rets = []
    current = None
    entry_price = 0.0

    for date in dates:
        loc = prices.index.get_loc(date)
        if loc < max(breakout_days, trail_days) + 1:
            daily_rets.append(0.0)
            continue

        cost = 0.0

        # Check exit: price below trail_days low
        if current is not None and current in prices.columns:
            trail_low = prices[current].iloc[loc - trail_days:loc].min()
            curr_price = prices[current].iloc[loc]
            if not np.isnan(trail_low) and curr_price < trail_low:
                if current in SPREAD_BPS:
                    cost += SPREAD_BPS[current] / 10000
                current = None

        # Check entry: price at breakout_days high
        if current is None:
            for etf in etfs:
                if etf == SAFE_ASSET or etf not in prices.columns:
                    continue
                breakout_high = prices[etf].iloc[loc - breakout_days:loc].max()
                curr_price = prices[etf].iloc[loc]
                if not np.isnan(breakout_high) and curr_price >= breakout_high:
                    current = etf
                    entry_price = curr_price
                    if etf in SPREAD_BPS:
                        cost += SPREAD_BPS[etf] / 10000
                    break

        # Return
        if current is not None and current in returns.columns:
            r = returns[current].iloc[loc]
            r = 0.0 if np.isnan(r) else r
        else:
            r = 0.0

        daily_rets.append(r - cost)

    return pd.Series(daily_rets, index=dates)


def optimize_breakout(prices, etfs, train_start, train_end):
    """Find best (breakout_days, trail_days) on train window by Sharpe."""
    best_sharpe = -999
    best_params = (10, 8)

    for bd, td in product(BO_BREAKOUT_DAYS, BO_TRAIL_DAYS):
        dr = run_breakout(prices, etfs, bd, td, train_start, train_end)
        if len(dr) < 60 or dr.std() == 0:
            continue
        sharpe = dr.mean() / dr.std() * np.sqrt(252)
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_params = (bd, td)

    return best_params, best_sharpe


# ── METRICS ────────────────────────────────────────────────────────

def calc_metrics(dr, label=""):
    """Compute Sharpe, CAGR, MaxDD, WR, Sortino from daily returns."""
    dr = dr.dropna()
    if len(dr) < 20 or dr.std() == 0:
        return {"label": label, "n_days": len(dr), "valid": False}

    ann_ret = dr.mean() * 252
    ann_vol = dr.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol
    years = len(dr) / 252
    cagr = (1 + dr).prod() ** (1 / max(years, 0.01)) - 1
    cum = (1 + dr).cumprod()
    max_dd = ((cum - cum.cummax()) / cum.cummax()).min()
    wr = (dr > 0).mean()
    downside = dr[dr < 0].std() * np.sqrt(252)
    sortino = ann_ret / downside if downside > 0 else 0.0

    return {
        "label": label, "n_days": len(dr), "valid": True,
        "sharpe": round(sharpe, 3), "cagr": round(cagr * 100, 2),
        "max_dd": round(max_dd * 100, 2), "wr": round(wr * 100, 1),
        "sortino": round(sortino, 3), "ann_vol": round(ann_vol * 100, 1),
    }


# ── ADVERSARIAL CHECKS (HC #705) ──────────────────────────────────

def permutation_test(dr, n_shuffles=100):
    """Block-bootstrap sign-flip permutation test.
    Flips the sign of random 5-day blocks to test if the directional
    edge is real. Simple shuffling preserves mean/std so is useless."""
    dr = dr.dropna().values
    if len(dr) < 30 or dr.std() == 0:
        return 1.0, 0.0, []
    actual = dr.mean() / dr.std() * np.sqrt(252)
    block_size = 5
    n_blocks = len(dr) // block_size
    null_sharpes = []
    for _ in range(n_shuffles):
        perm = dr.copy()
        for b in range(n_blocks):
            if np.random.random() < 0.5:
                s = b * block_size
                e = min(s + block_size, len(perm))
                perm[s:e] = -perm[s:e]
        std = perm.std()
        null_sharpes.append(perm.mean() / std * np.sqrt(252) if std > 0 else 0)
    p_value = (np.sum(np.array(null_sharpes) >= actual) + 1) / (n_shuffles + 1)
    return p_value, actual, null_sharpes


def regime_test(dr, spy_prices):
    """R1 regime-agnostic test: stratify OOT returns by green/red/flat SPY days."""
    spy_ret = spy_prices.pct_change()
    common = dr.index.intersection(spy_ret.index)
    if len(common) < 30:
        return None

    dr_c = dr.loc[common]
    spy_c = spy_ret.loc[common]

    # Classify: green (>+0.1%), red (<-0.1%), flat
    green = dr_c[spy_c > 0.001]
    red = dr_c[spy_c < -0.001]
    flat = dr_c[(spy_c >= -0.001) & (spy_c <= 0.001)]

    def _sharpe(s):
        if len(s) < 10 or s.std() == 0:
            return 0.0
        return s.mean() / s.std() * np.sqrt(252)

    s_green = _sharpe(green)
    s_red = _sharpe(red)
    s_flat = _sharpe(flat)
    denom = max(abs(s_green), abs(s_red), 0.001)
    regime_gap = abs(s_green - s_red) / denom

    return {
        "sharpe_green": round(s_green, 3), "n_green": len(green),
        "sharpe_red": round(s_red, 3), "n_red": len(red),
        "sharpe_flat": round(s_flat, 3), "n_flat": len(flat),
        "regime_gap": round(regime_gap, 3),
        "pass": regime_gap <= 0.50,
    }


def subperiod_consistency(dr):
    """Split OOT returns into halves and check both are positive Sharpe."""
    dr = dr.dropna()
    if len(dr) < 60:
        return None
    mid = len(dr) // 2
    h1 = dr.iloc[:mid]
    h2 = dr.iloc[mid:]
    s1 = h1.mean() / h1.std() * np.sqrt(252) if h1.std() > 0 else 0
    s2 = h2.mean() / h2.std() * np.sqrt(252) if h2.std() > 0 else 0
    return {
        "sharpe_h1": round(s1, 3), "sharpe_h2": round(s2, 3),
        "both_positive": s1 > 0 and s2 > 0,
    }


def outlier_removal_test(dr, pct=1):
    """Remove top/bottom pct% of returns and recompute Sharpe."""
    dr = dr.dropna()
    if len(dr) < 60:
        return None
    lo = np.percentile(dr, pct)
    hi = np.percentile(dr, 100 - pct)
    trimmed = dr[(dr >= lo) & (dr <= hi)]
    if len(trimmed) < 30 or trimmed.std() == 0:
        return None
    orig = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
    trim_s = trimmed.mean() / trimmed.std() * np.sqrt(252)
    return {
        "sharpe_original": round(orig, 3),
        "sharpe_trimmed_1pct": round(trim_s, 3),
        "survives": trim_s > 0,
    }


# ── WALK-FORWARD ENGINE ───────────────────────────────────────────

def generate_folds(prices):
    """Generate (train_start, train_end, test_start, test_end) folds."""
    first = prices.index[0]
    last = prices.index[-1]
    folds = []
    cursor = first

    while True:
        train_start = cursor
        train_end = train_start + pd.DateOffset(months=TRAIN_MONTHS)
        test_start = train_end
        test_end = test_start + pd.DateOffset(months=TEST_MONTHS)

        if test_end > last:
            # Use remaining data as last test fold
            if test_start < last:
                folds.append((train_start, train_end, test_start, last))
            break

        folds.append((train_start, train_end, test_start, test_end))
        cursor += pd.DateOffset(months=TEST_MONTHS)

    return folds


def main():
    t0 = time.time()
    print("=" * 72)
    print("SLIDING WALK-FORWARD VALIDATION")
    print("Dual Momentum + Breakout Trend Following")
    print(f"Train={TRAIN_MONTHS}mo, Test={TEST_MONTHS}mo, Slide={TEST_MONTHS}mo")
    print("=" * 72)

    prices = download_data()
    etfs = [t for t in UNIVERSE if t in prices.columns]
    spy = prices["SPY"] if "SPY" in prices.columns else None

    folds = generate_folds(prices)
    print(f"\n{len(folds)} walk-forward folds generated")

    # Storage for OOT returns
    dm_oot_all = []
    bo_oot_all = []
    dm_fold_results = []
    bo_fold_results = []

    print(f"\n{'Fold':>4} {'Train':>22} {'Test':>22} | {'DM params':>14} {'DM OOT Sharpe':>14} | {'BO params':>14} {'BO OOT Sharpe':>14}")
    print("-" * 120)

    for i, (tr_s, tr_e, te_s, te_e) in enumerate(folds):
        # ── Dual Momentum ──
        dm_params, dm_train_sharpe = optimize_dual_momentum(prices, etfs, tr_s, tr_e)
        dm_oot = run_dual_momentum(prices, etfs, dm_params[0], dm_params[1], te_s, te_e)
        dm_m = calc_metrics(dm_oot, f"DM_fold{i}")
        dm_oot_all.append(dm_oot)
        dm_fold_results.append({
            "fold": i, "train": f"{tr_s.date()}-{tr_e.date()}",
            "test": f"{te_s.date()}-{te_e.date()}",
            "params": f"lb={dm_params[0]},vt={dm_params[1]:.0%}",
            "train_sharpe": round(dm_train_sharpe, 3),
            **dm_m,
        })

        # ── Breakout ──
        bo_params, bo_train_sharpe = optimize_breakout(prices, etfs, tr_s, tr_e)
        bo_oot = run_breakout(prices, etfs, bo_params[0], bo_params[1], te_s, te_e)
        bo_m = calc_metrics(bo_oot, f"BO_fold{i}")
        bo_oot_all.append(bo_oot)
        bo_fold_results.append({
            "fold": i, "train": f"{tr_s.date()}-{tr_e.date()}",
            "test": f"{te_s.date()}-{te_e.date()}",
            "params": f"bd={bo_params[0]},td={bo_params[1]}",
            "train_sharpe": round(bo_train_sharpe, 3),
            **bo_m,
        })

        dm_s = dm_m.get("sharpe", "N/A")
        bo_s = bo_m.get("sharpe", "N/A")
        print(f"{i:>4} {str(tr_s.date()):>10}-{str(tr_e.date()):>10} "
              f"{str(te_s.date()):>10}-{str(te_e.date()):>10} | "
              f"lb={dm_params[0]:>2},vt={dm_params[1]:.0%} {str(dm_s):>14} | "
              f"bd={bo_params[0]:>2},td={bo_params[1]:>2} {str(bo_s):>14}")

    # ── Concatenate OOT returns ──
    dm_concat = pd.concat(dm_oot_all).sort_index()
    dm_concat = dm_concat[~dm_concat.index.duplicated(keep="last")]
    bo_concat = pd.concat(bo_oot_all).sort_index()
    bo_concat = bo_concat[~bo_concat.index.duplicated(keep="last")]

    print("\n" + "=" * 72)
    print("AGGREGATE OOT RESULTS (concatenated across all folds)")
    print("=" * 72)

    dm_agg = calc_metrics(dm_concat, "Dual Momentum OOT")
    bo_agg = calc_metrics(bo_concat, "Breakout OOT")

    for label, m in [("DUAL MOMENTUM", dm_agg), ("BREAKOUT", bo_agg)]:
        print(f"\n  {label}:")
        if m.get("valid"):
            print(f"    Sharpe:  {m['sharpe']:.3f}")
            print(f"    CAGR:   {m['cagr']:.2f}%")
            print(f"    MaxDD:  {m['max_dd']:.2f}%")
            print(f"    WR:     {m['wr']:.1f}%")
            print(f"    Sortino: {m['sortino']:.3f}")
            print(f"    AnnVol:  {m['ann_vol']:.1f}%")
            print(f"    N days:  {m['n_days']}")
        else:
            print(f"    INSUFFICIENT DATA ({m['n_days']} days)")

    # ── Baselines on same OOT period ──
    if spy is not None:
        spy_ret = spy.pct_change()
        common = dm_concat.index.intersection(spy_ret.index)
        spy_m = calc_metrics(spy_ret.loc[common], "SPY B&H")
        print(f"\n  BENCHMARK SPY (same OOT period):")
        if spy_m.get("valid"):
            print(f"    Sharpe: {spy_m['sharpe']:.3f}, CAGR: {spy_m['cagr']:.2f}%")

    # Buy-and-hold TQQQ with vol-targeting as the TRUE null hypothesis
    if "TQQQ" in prices.columns:
        tqqq_ret = prices["TQQQ"].pct_change()
        tqqq_vt = []
        for date in dm_concat.index:
            loc = prices.index.get_loc(date)
            r = tqqq_ret.iloc[loc]
            if np.isnan(r):
                tqqq_vt.append(0.0)
                continue
            rv = tqqq_ret.iloc[max(0, loc-21):loc].std() * np.sqrt(252)
            if rv > 0:
                scalar = min(0.30 / rv, 2.0)
                r *= scalar
            tqqq_vt.append(r)
        tqqq_vt = pd.Series(tqqq_vt, index=dm_concat.index)
        tqqq_m = calc_metrics(tqqq_vt, "TQQQ B&H + VT30")
        print(f"\n  CRITICAL BASELINE: TQQQ Buy-and-Hold + Vol Target 30%:")
        if tqqq_m.get("valid"):
            print(f"    Sharpe: {tqqq_m['sharpe']:.3f}, CAGR: {tqqq_m['cagr']:.2f}%, MaxDD: {tqqq_m['max_dd']:.2f}%")
            print(f"    (If strategies don't beat this, the 'signal' adds nothing vs. just holding leveraged ETFs)")

    # ── ADVERSARIAL CHECKS ──
    print("\n" + "=" * 72)
    print("ADVERSARIAL CHECKS (HC #705)")
    print("=" * 72)

    for label, dr in [("Dual Momentum", dm_concat), ("Breakout", bo_concat)]:
        print(f"\n  --- {label} ---")

        # 1. Permutation test
        p_val, actual_s, null_s = permutation_test(dr, n_shuffles=100)
        status = "PASS" if p_val < 0.05 else "FAIL"
        print(f"  Permutation test: p={p_val:.3f} [{status}]")
        print(f"    Actual Sharpe={actual_s:.3f}, Null mean={np.mean(null_s):.3f}")

        # 2. Regime test
        if spy is not None:
            rt = regime_test(dr, spy)
            if rt:
                rstatus = "PASS" if rt["pass"] else "FAIL"
                print(f"  Regime test: gap={rt['regime_gap']:.3f} [{rstatus}]")
                print(f"    Green days: Sharpe={rt['sharpe_green']:.3f} (n={rt['n_green']})")
                print(f"    Red days:   Sharpe={rt['sharpe_red']:.3f} (n={rt['n_red']})")
                print(f"    Flat days:  Sharpe={rt['sharpe_flat']:.3f} (n={rt['n_flat']})")

        # 3. Sub-period consistency
        sp = subperiod_consistency(dr)
        if sp:
            spstatus = "PASS" if sp["both_positive"] else "FAIL"
            print(f"  Sub-period: H1 Sharpe={sp['sharpe_h1']:.3f}, H2={sp['sharpe_h2']:.3f} [{spstatus}]")

        # 4. Outlier removal
        ot = outlier_removal_test(dr, pct=1)
        if ot:
            otstatus = "PASS" if ot["survives"] else "FAIL"
            print(f"  Outlier removal (1%): orig={ot['sharpe_original']:.3f}, trimmed={ot['sharpe_trimmed_1pct']:.3f} [{otstatus}]")

    # ── PER-FOLD SUMMARY TABLE ──
    print("\n" + "=" * 72)
    print("PER-FOLD OOT SUMMARY")
    print("=" * 72)

    print(f"\n  DUAL MOMENTUM folds:")
    print(f"  {'Fold':>4} {'Test Period':>22} {'Params':>16} {'TrainS':>7} {'OOT Sharpe':>11} {'CAGR':>8} {'MaxDD':>8} {'WR':>6}")
    for f in dm_fold_results:
        s = f.get("sharpe", "-")
        c = f.get("cagr", "-")
        d = f.get("max_dd", "-")
        w = f.get("wr", "-")
        ts = f.get("train_sharpe", "-")
        print(f"  {f['fold']:>4} {f['test']:>22} {f['params']:>16} {str(ts):>7} {str(s):>11} {str(c):>8} {str(d):>8} {str(w):>6}")

    print(f"\n  BREAKOUT folds:")
    print(f"  {'Fold':>4} {'Test Period':>22} {'Params':>16} {'TrainS':>7} {'OOT Sharpe':>11} {'CAGR':>8} {'MaxDD':>8} {'WR':>6}")
    for f in bo_fold_results:
        s = f.get("sharpe", "-")
        c = f.get("cagr", "-")
        d = f.get("max_dd", "-")
        w = f.get("wr", "-")
        ts = f.get("train_sharpe", "-")
        print(f"  {f['fold']:>4} {f['test']:>22} {f['params']:>16} {str(ts):>7} {str(s):>11} {str(c):>8} {str(d):>8} {str(w):>6}")

    # ── HONEST VERDICT ──
    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)

    for label, m, dr in [("Dual Momentum", dm_agg, dm_concat), ("Breakout", bo_agg, bo_concat)]:
        print(f"\n  {label}:")
        if not m.get("valid"):
            print(f"    INVALID - insufficient data")
            continue

        # Count positive/negative Sharpe folds
        fold_list = dm_fold_results if "Dual" in label else bo_fold_results
        pos_folds = sum(1 for f in fold_list if f.get("sharpe", 0) and f.get("valid", False) and f["sharpe"] > 0)
        neg_folds = sum(1 for f in fold_list if f.get("sharpe", 0) is not None and f.get("valid", False) and f["sharpe"] <= 0)
        total_valid = pos_folds + neg_folds

        print(f"    Aggregate OOT Sharpe: {m['sharpe']:.3f}")
        print(f"    Positive folds: {pos_folds}/{total_valid}")

        if m["sharpe"] > 0.5 and pos_folds > total_valid * 0.6:
            print(f"    STATUS: PROMISING - positive OOT edge with majority positive folds")
        elif m["sharpe"] > 0:
            print(f"    STATUS: MARGINAL - small positive OOT edge, needs more validation")
        else:
            print(f"    STATUS: FAILS - no walk-forward edge detected")

    # ── Save results ──
    elapsed = time.time() - t0
    output = {
        "generated": pd.Timestamp.now().isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "config": {
            "train_months": TRAIN_MONTHS, "test_months": TEST_MONTHS,
            "universe": UNIVERSE, "n_folds": len(folds),
        },
        "aggregate_oot": {"dual_momentum": dm_agg, "breakout": bo_agg},
        "per_fold": {"dual_momentum": dm_fold_results, "breakout": bo_fold_results},
    }

    with open(OUT_DIR / "walkforward_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    dm_concat.to_csv(OUT_DIR / "dm_oot_returns.csv")
    bo_concat.to_csv(OUT_DIR / "bo_oot_returns.csv")

    print(f"\nResults saved. Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}m)")


if __name__ == "__main__":
    main()

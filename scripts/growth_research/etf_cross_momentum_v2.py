"""
ETF Cross-Sectional Momentum — Survivorship-Bias-Free v2
=========================================================
HC #0   : Sliding walk-forward (36-month train, 6-month OOT, sliding)
HC #428 : Regime-agnostic validation (R1 gap check)
HC #694 : Commission-free (Robinhood) — 0 brokerage commissions
HC #705 : Adversarial checks (permutation, sub-period, outlier removal, R1, data sanity)

Universe: ETFs only — no individual stocks, no survivorship bias.
"""

import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/growth_research")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = OUTPUT_DIR / "etf_cross_momentum_v2_results.json"

START = "2007-01-01"
END   = "2026-07-14"

# ETF Universe — all long-lived, no survivorship bias
UNIVERSE = {
    "XLB":  {"class": "SECTOR", "inception": "1998-12-22"},
    "XLE":  {"class": "SECTOR", "inception": "1998-12-22"},
    "XLF":  {"class": "SECTOR", "inception": "1998-12-22"},
    "XLI":  {"class": "SECTOR", "inception": "1998-12-22"},
    "XLK":  {"class": "SECTOR", "inception": "1998-12-22"},
    "XLP":  {"class": "SECTOR", "inception": "1998-12-22"},
    "XLU":  {"class": "SECTOR", "inception": "1998-12-22"},
    "XLV":  {"class": "SECTOR", "inception": "1998-12-22"},
    "XLY":  {"class": "SECTOR", "inception": "1998-12-22"},
    "XLC":  {"class": "SECTOR", "inception": "2018-06-18"},
    "XLRE": {"class": "SECTOR", "inception": "2015-10-08"},
    "SPY":  {"class": "BROAD",     "inception": "1993-01-29"},
    "QQQ":  {"class": "BROAD",     "inception": "1999-03-10"},
    "IWM":  {"class": "BROAD",     "inception": "2000-05-26"},
    "MDY":  {"class": "BROAD",     "inception": "1995-05-04"},
    "EFA":  {"class": "INTL",      "inception": "2001-08-27"},
    "EEM":  {"class": "EM",        "inception": "2003-04-14"},
    "TLT":  {"class": "BOND",      "inception": "2002-07-30"},
    "IEF":  {"class": "BOND",      "inception": "2002-07-30"},
    "HYG":  {"class": "BOND_HY",   "inception": "2007-04-11"},
    "LQD":  {"class": "BOND_IG",   "inception": "2002-07-26"},
    "GLD":  {"class": "COMMODITY", "inception": "2004-11-18"},
    "SLV":  {"class": "COMMODITY", "inception": "2006-04-28"},
    "VNQ":  {"class": "REALESTATE","inception": "2004-09-29"},
}
TICKERS = list(UNIVERSE.keys())

WF_TRAIN_MONTHS = 36
WF_TEST_MONTHS  = 6
MOM_WINDOWS = {1: 21, 3: 63, 6: 126, 12: 252}
TOP_K_OPTIONS = [3, 5]
HOLD_M_OPTIONS = [21, 63]
CASH_DAILY_RETURN = 0.0002
N_PERMUTATIONS = 100
REGIME_FLAT_THRESHOLD = 0.001


def download_data():
    """Download adjusted close prices for all ETFs."""
    print("Downloading ETF data...")
    data = yf.download(TICKERS, start=START, end=END, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data
    prices = prices.dropna(axis=1, how="all")
    print(f"  Downloaded {len(prices.columns)} tickers, {len(prices)} trading days")
    print(f"  Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    return prices


def data_sanity_check(prices):
    """HC #705 data sanity."""
    issues = []
    rets = prices.pct_change()
    for col in rets.columns:
        n_extreme = (rets[col].abs() > 0.20).sum()
        if n_extreme > 0:
            issues.append(f"{col}: {n_extreme} days with |return| > 20%")
    for ticker in TICKERS:
        if ticker not in prices.columns:
            issues.append(f"{ticker}: MISSING")
            continue
        pct_missing = prices[ticker].isna().sum() / len(prices) * 100
        if pct_missing > 10:
            issues.append(f"{ticker}: {pct_missing:.1f}% missing data")
    if issues:
        print("  Data sanity issues:")
        for iss in issues:
            print(f"    - {iss}")
    else:
        print("  Data sanity: PASS")
    return issues


def precompute_availability(prices, min_history_days=252):
    """Precompute a boolean DataFrame: is each ticker available on each date."""
    avail = pd.DataFrame(False, index=prices.index, columns=[t for t in TICKERS if t in prices.columns])
    for ticker in avail.columns:
        inception = pd.Timestamp(UNIVERSE[ticker]["inception"])
        min_date = inception + pd.Timedelta(days=int(min_history_days * 1.5))
        # Count cumulative non-NaN observations
        cum_valid = prices[ticker].notna().cumsum()
        avail[ticker] = (prices.index >= min_date) & (cum_valid >= min_history_days)
    return avail


def run_backtest_fast(prices, returns_df, config, avail_df, permute=False, rng=None):
    """
    Vectorized momentum backtest.
    Precomputes trailing returns, then loops only over rebalance dates.
    """
    lookback_days = MOM_WINDOWS[config["lookback_months"]]
    top_k = config["top_k"]
    hold_days = config["hold_days"]
    use_ma_filter = config.get("ma_filter", False)
    weight_method = config.get("weight_method", "equal")
    dual_momentum = config.get("dual_momentum", False)

    tickers = [t for t in TICKERS if t in returns_df.columns]
    dates = returns_df.index
    n_dates = len(dates)

    # Precompute trailing returns for ranking
    trailing_ret = prices[tickers].pct_change(lookback_days)

    # Precompute 200d MA if needed
    if use_ma_filter:
        ma200 = prices[tickers].rolling(200).mean()

    # Precompute 63d vol for inverse-vol weighting
    if weight_method == "inverse_vol":
        rolling_vol = returns_df[tickers].rolling(63).std()

    # Portfolio returns array
    port_rets = np.zeros(n_dates)

    # Determine rebalance dates (every hold_days trading days)
    rebal_indices = list(range(lookback_days, n_dates, hold_days))

    # For each rebalance period, compute weights and apply
    for r_idx in range(len(rebal_indices)):
        rebal_i = rebal_indices[r_idx]
        next_rebal_i = rebal_indices[r_idx + 1] if r_idx + 1 < len(rebal_indices) else n_dates

        rebal_date = dates[rebal_i]

        # Get available tickers
        avail_mask = avail_df.loc[rebal_date]
        available = list(avail_mask[avail_mask].index)
        available = [t for t in available if t in tickers]

        if len(available) < top_k + 1:
            continue

        # Get trailing returns
        tr = trailing_ret.loc[rebal_date, available].dropna()
        if len(tr) < top_k:
            continue

        tr = tr.sort_values(ascending=False)

        if permute and rng is not None:
            selected = list(rng.choice(list(tr.index), size=min(top_k, len(tr)), replace=False))
        else:
            selected = list(tr.index[:top_k])

        # MA filter
        if use_ma_filter and rebal_i >= 200:
            above_ma = [t for t in selected
                        if prices[t].iloc[rebal_i] > ma200[t].iloc[rebal_i]]
            selected = above_ma if above_ma else []

        # Dual momentum
        if dual_momentum:
            cash_ret = CASH_DAILY_RETURN * lookback_days
            selected = [t for t in selected if tr.get(t, 0) > cash_ret]

        if not selected:
            continue

        # Weights
        if weight_method == "inverse_vol" and rebal_i >= 63:
            vols = rolling_vol.loc[rebal_date, selected].replace(0, np.nan).dropna()
            if len(vols) > 0:
                inv_vol = 1.0 / vols
                weights = (inv_vol / inv_vol.sum()).values
                sel_tickers = list(vols.index)
            else:
                weights = np.ones(len(selected)) / len(selected)
                sel_tickers = selected
        else:
            weights = np.ones(len(selected)) / len(selected)
            sel_tickers = selected

        # Apply weights for the holding period
        period_rets = returns_df.iloc[rebal_i+1:next_rebal_i][sel_tickers].fillna(0).values
        port_rets[rebal_i+1:next_rebal_i] = period_rets @ weights

    return pd.Series(port_rets, index=dates)


def compute_metrics(returns):
    """Compute strategy metrics."""
    returns = returns.dropna()
    if len(returns) < 20:
        return {"sharpe": 0, "cagr": 0, "max_dd": 0, "wr": 0, "sortino": 0, "calmar": 0, "n_days": 0}

    mu = returns.mean() * 252
    sigma = returns.std() * np.sqrt(252)
    sharpe = mu / sigma if sigma > 0 else 0

    cum = (1 + returns).cumprod()
    years = len(returns) / 252
    cagr = (cum.iloc[-1] ** (1/years) - 1) if years > 0 and cum.iloc[-1] > 0 else 0

    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()

    wr = (returns > 0).sum() / (returns != 0).sum() if (returns != 0).sum() > 0 else 0

    downside = returns[returns < 0].std() * np.sqrt(252)
    sortino = mu / downside if downside > 0 else 0

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    return {
        "sharpe": round(sharpe, 3),
        "cagr": round(cagr * 100, 2),
        "max_dd": round(max_dd * 100, 2),
        "wr": round(wr * 100, 1),
        "sortino": round(sortino, 3),
        "calmar": round(calmar, 3),
        "n_days": len(returns),
    }


def regime_classify(spy_returns):
    regimes = pd.Series("flat", index=spy_returns.index)
    regimes[spy_returns > REGIME_FLAT_THRESHOLD] = "green"
    regimes[spy_returns < -REGIME_FLAT_THRESHOLD] = "red"
    return regimes


def r1_regime_test(returns, spy_returns):
    """HC #428 R1: Sharpe per regime, check gap <= 0.50."""
    regimes = regime_classify(spy_returns)
    common_idx = returns.index.intersection(regimes.index)
    returns = returns.loc[common_idx]
    regimes = regimes.loc[common_idx]

    regime_sharpes = {}
    for regime in ["green", "red", "flat"]:
        r = returns[regimes == regime]
        if len(r) > 20:
            mu = r.mean() * 252
            sig = r.std() * np.sqrt(252)
            regime_sharpes[regime] = round(mu / sig, 3) if sig > 0 else 0
        else:
            regime_sharpes[regime] = None

    s_green = regime_sharpes.get("green")
    s_red = regime_sharpes.get("red")

    if s_green is None or s_red is None:
        return None, regime_sharpes, False

    denom = max(abs(s_green), abs(s_red))
    gap = abs(s_green - s_red) / denom if denom > 0 else 0
    passed = gap <= 0.50

    return round(gap, 4), regime_sharpes, passed


def permutation_test(prices, returns_df, config, avail_df, observed_sharpe, n_perms=N_PERMUTATIONS):
    """HC #705: shuffle ETF selection, compute p-value."""
    perm_sharpes = []
    for i in range(n_perms):
        rng = np.random.default_rng(seed=i)
        perm_rets = run_backtest_fast(prices, returns_df, config, avail_df, permute=True, rng=rng)
        first_nz = perm_rets[perm_rets != 0].first_valid_index()
        if first_nz is not None:
            perm_rets = perm_rets.loc[first_nz:]
        m = compute_metrics(perm_rets)
        perm_sharpes.append(m["sharpe"])

    perm_sharpes = np.array(perm_sharpes)
    p_value = (perm_sharpes >= observed_sharpe).sum() / len(perm_sharpes)
    return round(p_value, 4)


def sub_period_consistency(returns):
    n = len(returns)
    if n < 40:
        return None, None
    half = n // 2
    return compute_metrics(returns.iloc[:half])["sharpe"], compute_metrics(returns.iloc[half:])["sharpe"]


def outlier_removal_check(returns, top_n=5):
    sorted_rets = returns.sort_values(ascending=False)
    trimmed = returns.drop(sorted_rets.index[:top_n])
    return compute_metrics(trimmed)["sharpe"]


def make_config_label(cfg):
    return (f"LB{cfg['lookback_months']}m_K{cfg['top_k']}_H{cfg['hold_days']}d"
            f"{'_MA' if cfg.get('ma_filter') else ''}"
            f"{'_IV' if cfg.get('weight_method') == 'inverse_vol' else ''}"
            f"{'_DM' if cfg.get('dual_momentum') else ''}")


def build_all_configs():
    """Build all config combinations."""
    base = []
    for lb_m in [1, 3, 6, 12]:
        for top_k in TOP_K_OPTIONS:
            for hold_d in HOLD_M_OPTIONS:
                base.append({
                    "lookback_months": lb_m, "top_k": top_k, "hold_days": hold_d,
                    "ma_filter": False, "weight_method": "equal", "dual_momentum": False,
                })

    variants = []
    for cfg in base:
        # MA filter
        c = cfg.copy(); c["ma_filter"] = True; variants.append(c)
        # Inverse vol
        c = cfg.copy(); c["weight_method"] = "inverse_vol"; variants.append(c)
        # Dual momentum
        c = cfg.copy(); c["dual_momentum"] = True; variants.append(c)

    return base + variants


def run_all_configs(prices, returns_df, spy_returns, avail_df):
    """Run all config combinations."""
    all_configs = build_all_configs()
    print(f"\nTotal configs to test: {len(all_configs)}")

    results = []
    for i, cfg in enumerate(all_configs):
        cfg_label = make_config_label(cfg)

        if (i + 1) % 16 == 1:
            print(f"  [{i+1}/{len(all_configs)}] {cfg_label} ...")

        strat_rets = run_backtest_fast(prices, returns_df, cfg, avail_df)

        first_nz = strat_rets[strat_rets != 0].first_valid_index()
        if first_nz is None:
            continue
        strat_rets = strat_rets.loc[first_nz:]

        metrics = compute_metrics(strat_rets)
        r1_gap, regime_sharpes, r1_pass = r1_regime_test(strat_rets, spy_returns)
        sh_first, sh_second = sub_period_consistency(strat_rets)
        sh_no_outliers = outlier_removal_check(strat_rets)

        # Permutation test only for promising configs (Sharpe > 0.3)
        p_value = None
        if metrics["sharpe"] > 0.3:
            print(f"    Permutation test for {cfg_label} (Sharpe={metrics['sharpe']:.3f})...")
            p_value = permutation_test(prices, returns_df, cfg, avail_df, metrics["sharpe"], n_perms=100)

        gates = {
            "sharpe_positive": metrics["sharpe"] > 0,
            "r1_regime_pass": r1_pass if r1_gap is not None else False,
            "outlier_robust": sh_no_outliers > 0 if sh_no_outliers is not None else False,
            "perm_significant": p_value < 0.10 if p_value is not None else None,
        }
        all_pass = all(v for v in gates.values() if v is not None)

        results.append({
            "config": cfg_label,
            "config_params": cfg,
            **metrics,
            "r1_gap": r1_gap,
            "regime_sharpes": regime_sharpes,
            "r1_pass": r1_pass,
            "sharpe_first_half": sh_first,
            "sharpe_second_half": sh_second,
            "sharpe_no_outliers": sh_no_outliers,
            "perm_p_value": p_value,
            "gates": gates,
            "all_gates_pass": all_pass,
        })

    return results, all_configs


def run_walk_forward(prices, returns_df, spy_returns, avail_df):
    """Walk-forward: 36m train, 6m OOT, sliding. Optimize on train, eval on OOT."""
    base_configs = []
    for lb_m in [1, 3, 6, 12]:
        for top_k in TOP_K_OPTIONS:
            for hold_d in HOLD_M_OPTIONS:
                base_configs.append({
                    "lookback_months": lb_m, "top_k": top_k, "hold_days": hold_d,
                    "ma_filter": False, "weight_method": "equal", "dual_momentum": False,
                })

    dates = returns_df.index
    start_date = dates[0]
    end_date = dates[-1]

    folds = []
    cursor = start_date + pd.DateOffset(months=WF_TRAIN_MONTHS)
    while cursor + pd.DateOffset(months=WF_TEST_MONTHS) <= end_date:
        tr_s = cursor - pd.DateOffset(months=WF_TRAIN_MONTHS)
        folds.append((tr_s, cursor, cursor, cursor + pd.DateOffset(months=WF_TEST_MONTHS)))
        cursor += pd.DateOffset(months=WF_TEST_MONTHS)

    print(f"\n  Walk-forward: {len(folds)} folds")
    oot_returns_list = []

    for fold_idx, (tr_s, tr_e, te_s, te_e) in enumerate(folds):
        best_sharpe = -999
        best_config = None

        for cfg in base_configs:
            train_mask = (dates >= tr_s) & (dates < tr_e)
            bt = run_backtest_fast(prices, returns_df.loc[train_mask], cfg, avail_df)
            bt = bt[bt.index >= tr_s]
            m = compute_metrics(bt)
            if m["sharpe"] > best_sharpe:
                best_sharpe = m["sharpe"]
                best_config = cfg

        # OOT evaluation
        test_mask = (dates >= te_s) & (dates < te_e)
        oot = run_backtest_fast(prices, returns_df.loc[test_mask], best_config, avail_df)
        oot = oot[(oot.index >= te_s) & (oot.index < te_e)]
        oot_returns_list.append(oot)

        if (fold_idx + 1) % 5 == 0 or fold_idx == len(folds) - 1:
            print(f"    Fold {fold_idx+1}/{len(folds)}: best train Sharpe={best_sharpe:.3f} "
                  f"(LB={best_config['lookback_months']}m K={best_config['top_k']} "
                  f"H={best_config['hold_days']}d)")

    combined = pd.concat(oot_returns_list).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]

    metrics = compute_metrics(combined)
    r1_gap, regime_sharpes, r1_pass = r1_regime_test(combined, spy_returns)
    sh1, sh2 = sub_period_consistency(combined)

    return {
        "WF_optimized": {
            **metrics,
            "r1_gap": r1_gap,
            "regime_sharpes": regime_sharpes,
            "r1_pass": r1_pass,
            "sharpe_first_half": sh1,
            "sharpe_second_half": sh2,
            "n_folds": len(folds),
        }
    }


def print_summary(results, wf_analysis):
    print("\n" + "="*120)
    print("ETF CROSS-SECTIONAL MOMENTUM — FULL RESULTS (Survivorship-Bias-Free)")
    print("="*120)

    results_sorted = sorted(results, key=lambda x: x["sharpe"], reverse=True)

    print(f"\n{'Config':<40} {'Sharpe':>7} {'CAGR%':>7} {'MaxDD%':>8} {'WR%':>6} "
          f"{'Sortino':>8} {'R1gap':>7} {'R1':>5} {'Perm-p':>7} {'OutlSh':>7} {'Pass':>5}")
    print("-"*120)

    for r in results_sorted:
        r1_str = "PASS" if r["r1_pass"] else "FAIL"
        perm_str = f"{r['perm_p_value']:.3f}" if r["perm_p_value"] is not None else "  --"
        r1gap_str = f"{r['r1_gap']:.4f}" if r["r1_gap"] is not None else "  --"
        outl_str = f"{r['sharpe_no_outliers']:.3f}" if r['sharpe_no_outliers'] is not None else "  --"
        pass_str = " YES" if r["all_gates_pass"] else "  NO"

        print(f"{r['config']:<40} {r['sharpe']:>7.3f} {r['cagr']:>7.2f} {r['max_dd']:>8.2f} "
              f"{r['wr']:>6.1f} {r['sortino']:>8.3f} {r1gap_str:>7} {r1_str:>5} "
              f"{perm_str:>7} {outl_str:>7} {pass_str:>5}")

    print(f"\n{'='*80}")
    print("WALK-FORWARD OPTIMIZED (36m train / 6m OOT, sliding)")
    print(f"{'='*80}")
    for key, wf in wf_analysis.items():
        print(f"\n  {key}:")
        print(f"    Sharpe={wf['sharpe']:.3f}  CAGR={wf['cagr']:.2f}%  MaxDD={wf['max_dd']:.2f}%  "
              f"WR={wf['wr']:.1f}%  Sortino={wf['sortino']:.3f}")
        print(f"    R1 gap={wf['r1_gap']}  R1 pass={wf['r1_pass']}")
        print(f"    Regime Sharpes: {wf['regime_sharpes']}")
        print(f"    Sub-period: 1st={wf['sharpe_first_half']}, 2nd={wf['sharpe_second_half']}")
        print(f"    N_folds={wf['n_folds']}")

    passing = [r for r in results_sorted if r["all_gates_pass"]]
    print(f"\n{'='*80}")
    print(f"CONFIGS PASSING ALL GATES: {len(passing)}/{len(results_sorted)}")
    print(f"{'='*80}")
    if passing:
        for r in passing[:10]:
            perm_str = f"{r['perm_p_value']:.3f}" if r['perm_p_value'] is not None else "n/a"
            print(f"  {r['config']}: Sharpe={r['sharpe']:.3f}, CAGR={r['cagr']:.2f}%, "
                  f"R1gap={r['r1_gap']}, Perm-p={perm_str}")
    else:
        print("  NONE — no config passed all adversarial gates.")

    return passing


def main():
    print("="*80)
    print("ETF Cross-Sectional Momentum v2 — Survivorship-Bias-Free")
    print(f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    prices = download_data()
    sanity_issues = data_sanity_check(prices)

    if "SPY" not in prices.columns:
        print("ERROR: SPY not in data")
        return

    spy_returns = prices["SPY"].pct_change().clip(-0.20, 0.20)

    available_tickers = [t for t in TICKERS if t in prices.columns]
    print(f"\nAvailable ETFs: {len(available_tickers)}/{len(TICKERS)}")
    print(f"  {', '.join(available_tickers)}")

    returns_df = prices[available_tickers].pct_change().clip(-0.20, 0.20)

    # Precompute availability matrix (biggest speedup)
    print("Precomputing availability matrix...")
    avail_df = precompute_availability(prices, min_history_days=252)

    # Run all configs
    results, all_configs = run_all_configs(prices, returns_df, spy_returns, avail_df)

    # Walk-forward
    wf_analysis = run_walk_forward(prices, returns_df, spy_returns, avail_df)

    # Summary
    passing = print_summary(results, wf_analysis)

    # Save
    output = {
        "run_timestamp": datetime.now().isoformat(),
        "universe": list(UNIVERSE.keys()),
        "n_etfs": len(available_tickers),
        "date_range": f"{prices.index[0].date()} to {prices.index[-1].date()}",
        "n_configs_tested": len(results),
        "n_passing": len(passing),
        "sanity_issues": sanity_issues,
        "results": results,
        "walk_forward": wf_analysis,
        "passing_configs": [r["config"] for r in passing],
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_FILE}")

    return output


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Managed Volatility Leveraged ETF Strategy Backtest
===================================================
Academic basis: Moreira & Muir (2017) — Volatility-Managed Portfolios
Core idea: Scale leveraged ETF exposure inversely with volatility.

8 Variants (A-H) with 5-gate validation.

Author: Claude Opus 4.6
Date: 2026-07-27
"""

import os
import sys
import logging
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Paths ──
OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/growth_research/managed_vol_leveraged_v1")
LOG_PATH = Path("/home/jupiter/Lvl3Quant/logs/managed_vol_leveraged_v1.log")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, mode="w"), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── Constants ──
START_DATE = "2020-01-01"
END_DATE = "2026-07-01"
CAPITAL_LARGE = 10_000
CAPITAL_SMALL = 645
TX_COST = 0.0005  # 0.05% per trade
N_PERMUTATIONS = 100
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)


# ═══════════════════════════════════════════════════════════════════════
# DATA DOWNLOAD
# ═══════════════════════════════════════════════════════════════════════
def download_data():
    """Download all required tickers via yfinance."""
    import yfinance as yf

    tickers = ["TQQQ", "UPRO", "SOXL", "QQQ", "SPY", "SOXX", "TLT", "SHY", "GLD", "^VIX"]
    log.info(f"Downloading {len(tickers)} tickers from {START_DATE} to {END_DATE}")

    data = {}
    for t in tickers:
        try:
            df = yf.download(t, start=START_DATE, end=END_DATE, progress=False, auto_adjust=False)
            if df.empty:
                log.warning(f"Empty data for {t}")
                continue
            # Handle multi-level columns from yfinance
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            col = "Adj Close" if "Adj Close" in df.columns else "Close"
            data[t.replace("^", "")] = df[col].squeeze()
            log.info(f"  {t}: {len(df)} rows, {df.index[0].date()} to {df.index[-1].date()}")
        except Exception as e:
            log.error(f"Failed to download {t}: {e}")

    prices = pd.DataFrame(data)
    prices = prices.dropna(how="all")
    prices = prices.ffill()
    log.info(f"Combined price matrix: {prices.shape}")
    return prices


# ═══════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════
def daily_returns(prices, ticker):
    """Simple daily returns."""
    return prices[ticker].pct_change().fillna(0)


def realized_vol(prices, ticker, window=20):
    """Annualized realized volatility (rolling std of daily returns * sqrt(252))."""
    ret = prices[ticker].pct_change()
    return ret.rolling(window).std() * np.sqrt(252) * 100  # in percent


def sharpe(returns, rf=0.0):
    """Annualized Sharpe ratio."""
    if returns.std() == 0:
        return 0.0
    return (returns.mean() - rf / 252) / returns.std() * np.sqrt(252)


def sortino(returns, rf=0.0):
    """Annualized Sortino ratio."""
    downside = returns[returns < 0].std()
    if downside == 0:
        return 0.0
    return (returns.mean() - rf / 252) / downside * np.sqrt(252)


def cagr(equity_curve):
    """CAGR from equity curve."""
    if len(equity_curve) < 2 or equity_curve.iloc[0] <= 0:
        return 0.0
    years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
    if years <= 0:
        return 0.0
    return (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / years) - 1


def max_drawdown(equity_curve):
    """Maximum drawdown as a fraction (negative)."""
    peak = equity_curve.cummax()
    dd = (equity_curve - peak) / peak
    return dd.min()


def calmar(equity_curve, returns):
    """Calmar ratio = CAGR / |MaxDD|."""
    mdd = abs(max_drawdown(equity_curve))
    c = cagr(equity_curve)
    if mdd == 0:
        return 0.0
    return c / mdd


def monthly_win_rate(returns):
    """Fraction of months with positive returns."""
    monthly = returns.resample("ME").sum()
    if len(monthly) == 0:
        return 0.0
    return (monthly > 0).mean()


def worst_month(returns):
    """Worst single month return."""
    monthly = returns.resample("ME").sum()
    if len(monthly) == 0:
        return 0.0
    return monthly.min()


def recovery_time_days(equity_curve):
    """Days from max drawdown trough back to previous peak. 0 if never recovered."""
    peak = equity_curve.cummax()
    dd = (equity_curve - peak) / peak
    trough_idx = dd.idxmin()
    post_trough = equity_curve.loc[trough_idx:]
    peak_val = peak.loc[trough_idx]
    recovered = post_trough[post_trough >= peak_val]
    if len(recovered) == 0:
        return -1  # never recovered
    return (recovered.index[0] - trough_idx).days


def time_in_market(weights_series):
    """Fraction of days with non-zero exposure to risky assets."""
    return (weights_series > 0).mean()


def apply_tx_costs(returns, positions_changed):
    """Subtract transaction costs on days when position changes."""
    costs = positions_changed.astype(float) * TX_COST
    return returns - costs


def build_equity(returns, capital):
    """Build equity curve from returns series."""
    return capital * (1 + returns).cumprod()


# ═══════════════════════════════════════════════════════════════════════
# STRATEGY VARIANTS
# ═══════════════════════════════════════════════════════════════════════
def variant_a(prices):
    """TQQQ Vol-Target: TQQQ when 20d realized vol of QQQ < 20%, else SHY."""
    vol = realized_vol(prices, "QQQ", 20)
    ret_tqqq = daily_returns(prices, "TQQQ")
    ret_shy = daily_returns(prices, "SHY")

    signal = (vol < 20).shift(1).fillna(False)  # signal from prior day
    strat_ret = signal * ret_tqqq + (~signal) * ret_shy
    pos_changed = signal.diff().abs().fillna(0) > 0

    strat_ret = apply_tx_costs(strat_ret, pos_changed)
    risky_weight = signal.astype(float)
    return strat_ret, risky_weight, "A: TQQQ Vol-Target"


def variant_b(prices):
    """TQQQ VIX-Switch: TQQQ when VIX<20, SHY when VIX>30, 50/50 in between."""
    vix = prices["VIX"].shift(1)  # prior day VIX
    ret_tqqq = daily_returns(prices, "TQQQ")
    ret_shy = daily_returns(prices, "SHY")

    w_tqqq = pd.Series(0.0, index=prices.index)
    w_tqqq[vix < 20] = 1.0
    w_tqqq[(vix >= 20) & (vix <= 30)] = 0.5
    w_tqqq[vix > 30] = 0.0

    strat_ret = w_tqqq * ret_tqqq + (1 - w_tqqq) * ret_shy
    pos_changed = w_tqqq.diff().abs().fillna(0) > 0
    strat_ret = apply_tx_costs(strat_ret, pos_changed)
    return strat_ret, w_tqqq, "B: TQQQ VIX-Switch"


def variant_c(prices):
    """UPRO Vol-Target: UPRO when 20d realized vol of SPY < 20%, else SHY."""
    vol = realized_vol(prices, "SPY", 20)
    ret_upro = daily_returns(prices, "UPRO")
    ret_shy = daily_returns(prices, "SHY")

    signal = (vol < 20).shift(1).fillna(False)
    strat_ret = signal * ret_upro + (~signal) * ret_shy
    pos_changed = signal.diff().abs().fillna(0) > 0
    strat_ret = apply_tx_costs(strat_ret, pos_changed)
    risky_weight = signal.astype(float)
    return strat_ret, risky_weight, "C: UPRO Vol-Target"


def variant_d(prices):
    """Dual Momentum + Vol: TQQQ if QQQ 3m return > 0 AND VIX < 25, else SHY."""
    qqq_3m_ret = prices["QQQ"].pct_change(63).shift(1)  # ~3 months
    vix = prices["VIX"].shift(1)
    ret_tqqq = daily_returns(prices, "TQQQ")
    ret_shy = daily_returns(prices, "SHY")

    signal = ((qqq_3m_ret > 0) & (vix < 25)).fillna(False)
    strat_ret = signal * ret_tqqq + (~signal) * ret_shy
    pos_changed = signal.diff().abs().fillna(0) > 0
    strat_ret = apply_tx_costs(strat_ret, pos_changed)
    risky_weight = signal.astype(float)
    return strat_ret, risky_weight, "D: Dual Momentum+Vol"


def variant_e(prices):
    """Risk Parity Leverage: Equal-risk-contribution TQQQ/UPRO/SOXL, scaled by inverse vol. Weekly rebal."""
    tickers = ["TQQQ", "UPRO", "SOXL"]
    rets = pd.DataFrame({t: daily_returns(prices, t) for t in tickers})

    # Compute inverse-vol weights daily, then sample weekly
    vol_20 = rets.rolling(20).std() * np.sqrt(252)
    inv_vol = 1.0 / vol_20.replace(0, np.nan)
    weights = inv_vol.div(inv_vol.sum(axis=1), axis=0)

    # Weekly rebalance: use last available weights, update on Fridays
    is_rebal = prices.index.dayofweek == 4
    rebal_mask = pd.DataFrame({t: is_rebal for t in tickers}, index=prices.index)
    rebal_weights = weights.where(rebal_mask)
    rebal_weights = rebal_weights.ffill()
    # Backfill initial NaN period with first valid weights
    rebal_weights = rebal_weights.bfill()
    # Normalize to sum to 1.0
    row_sum = rebal_weights.sum(axis=1).replace(0, 1)
    rebal_weights = rebal_weights.div(row_sum, axis=0).fillna(0)

    strat_ret = (rebal_weights.shift(1).fillna(0) * rets).sum(axis=1)
    pos_changed = rebal_weights.diff().abs().sum(axis=1).fillna(0) > 0.01
    strat_ret = apply_tx_costs(strat_ret, pos_changed)
    risky_weight = rebal_weights.sum(axis=1)
    return strat_ret, risky_weight, "E: Risk Parity Lev"


def variant_f(prices):
    """Adaptive Leverage based on VIX bands."""
    vix = prices["VIX"].shift(1)
    ret_tqqq = daily_returns(prices, "TQQQ")
    ret_tlt = daily_returns(prices, "TLT")
    ret_shy = daily_returns(prices, "SHY")

    w_tqqq = pd.Series(0.0, index=prices.index)
    w_tlt = pd.Series(0.0, index=prices.index)
    w_shy = pd.Series(0.0, index=prices.index)

    # VIX < 15: 100% TQQQ
    mask1 = vix < 15
    w_tqqq[mask1] = 1.0

    # VIX 15-20: 70% TQQQ + 30% TLT
    mask2 = (vix >= 15) & (vix < 20)
    w_tqqq[mask2] = 0.7
    w_tlt[mask2] = 0.3

    # VIX 20-30: 50% SHY + 50% TLT
    mask3 = (vix >= 20) & (vix < 30)
    w_shy[mask3] = 0.5
    w_tlt[mask3] = 0.5

    # VIX >= 30: 100% SHY
    mask4 = vix >= 30
    w_shy[mask4] = 1.0

    strat_ret = w_tqqq * ret_tqqq + w_tlt * ret_tlt + w_shy * ret_shy
    pos_changed = (w_tqqq.diff().abs() + w_tlt.diff().abs() + w_shy.diff().abs()).fillna(0) > 0
    strat_ret = apply_tx_costs(strat_ret, pos_changed)
    return strat_ret, w_tqqq, "F: Adaptive Leverage"


def variant_g(prices):
    """Sector Rotation Leveraged: Monthly pick best 1m momentum among TQQQ/UPRO/SOXL, VIX filter."""
    tickers_lev = ["TQQQ", "UPRO", "SOXL"]
    rets = pd.DataFrame({t: daily_returns(prices, t) for t in tickers_lev})
    ret_shy = daily_returns(prices, "SHY")
    vix = prices["VIX"].shift(1)

    # Monthly momentum (21-day return)
    mom = pd.DataFrame({t: prices[t].pct_change(21) for t in tickers_lev})

    # Determine signal on first trading day of each month, hold for month
    strat_ret = pd.Series(0.0, index=prices.index)
    risky_weight = pd.Series(0.0, index=prices.index)

    # Group by month
    months = prices.index.to_period("M")
    prev_pick = None

    for month in months.unique():
        mask = months == month
        idx = prices.index[mask]
        if len(idx) == 0:
            continue

        first_day = idx[0]
        v = vix.loc[first_day] if first_day in vix.index else 20

        if pd.isna(v) or v > 25:
            # Go defensive
            strat_ret.loc[idx] = ret_shy.loc[idx]
            risky_weight.loc[idx] = 0
        else:
            # Pick best momentum
            mom_vals = mom.loc[first_day]
            if mom_vals.isna().all():
                strat_ret.loc[idx] = ret_shy.loc[idx]
                risky_weight.loc[idx] = 0
            else:
                best = mom_vals.idxmax()
                strat_ret.loc[idx] = rets[best].loc[idx]
                risky_weight.loc[idx] = 1.0
                if prev_pick != best:
                    # tx cost on first day
                    strat_ret.loc[idx[0]] -= TX_COST
                prev_pick = best

    return strat_ret, risky_weight, "G: Sector Rot Lev"


def variant_h(prices):
    """Max Growth: 60% TQQQ + 40% SOXL, kill switch if VIX>35 or QQQ 20d ret < -10%."""
    ret_tqqq = daily_returns(prices, "TQQQ")
    ret_soxl = daily_returns(prices, "SOXL")
    ret_shy = daily_returns(prices, "SHY")
    vix = prices["VIX"].shift(1)
    qqq_20d = prices["QQQ"].pct_change(20).shift(1)

    kill = ((vix > 35) | (qqq_20d < -0.10)).fillna(False)

    w_risky = (~kill).astype(float)
    strat_ret = w_risky * (0.6 * ret_tqqq + 0.4 * ret_soxl) + kill.astype(float) * ret_shy

    pos_changed = kill.diff().abs().fillna(0) > 0
    strat_ret = apply_tx_costs(strat_ret, pos_changed)
    return strat_ret, w_risky, "H: Max Growth"


# ═══════════════════════════════════════════════════════════════════════
# VALIDATION GATES
# ═══════════════════════════════════════════════════════════════════════
def gate_permutation(returns, n_perm=N_PERMUTATIONS):
    """Gate 1: Bootstrap confidence interval test. Resample returns with replacement
    to test if the strategy Sharpe is significantly > 0. PASS if 5th percentile > 0."""
    from scipy import stats
    real_sharpe = sharpe(returns)
    ret_vals = returns.values
    n = len(ret_vals)
    boot_sharpes = []
    for _ in range(n_perm):
        # Resample with replacement (block bootstrap, block=5)
        block_size = 5
        n_blocks = int(np.ceil(n / block_size))
        chosen = np.random.randint(0, n - block_size + 1, size=n_blocks)
        sample = np.concatenate([ret_vals[i:i+block_size] for i in chosen])[:n]
        s = np.mean(sample) / (np.std(sample) + 1e-12) * np.sqrt(252)
        boot_sharpes.append(s)
    pct5 = np.percentile(boot_sharpes, 5)
    passed = pct5 > 0
    return passed, f"Sharpe={real_sharpe:.2f},CI5={pct5:.2f}"


def gate_subperiod(returns):
    """Gate 2: Sub-period stability — 4 equal quarters, pass if 3/4 positive Sharpe."""
    n = len(returns)
    q = n // 4
    positive = 0
    for i in range(4):
        start = i * q
        end = (i + 1) * q if i < 3 else n
        s = sharpe(returns.iloc[start:end])
        if s > 0:
            positive += 1
    return positive >= 3, f"{positive}/4"


def gate_outlier_removal(returns):
    """Gate 3: Trim 1% tails, recompute Sharpe. Pass if trimmed > 0.8x full."""
    full_s = sharpe(returns)
    lo, hi = returns.quantile(0.01), returns.quantile(0.99)
    trimmed = returns[(returns >= lo) & (returns <= hi)]
    trimmed_s = sharpe(trimmed)
    if full_s == 0:
        ratio = 1.0
    else:
        ratio = trimmed_s / full_s
    return ratio > 0.8, f"{ratio:.2f}"


def gate_regime_balance(returns, vix_series):
    """Gate 4: Sharpe on VIX>25 vs VIX<25 days. Pass if gap_ratio < 0.50."""
    aligned = pd.DataFrame({"ret": returns, "vix": vix_series}).dropna()
    high_vix = aligned[aligned["vix"] > 25]["ret"]
    low_vix = aligned[aligned["vix"] <= 25]["ret"]

    s_high = sharpe(high_vix) if len(high_vix) > 20 else 0
    s_low = sharpe(low_vix) if len(low_vix) > 20 else 0

    max_abs = max(abs(s_high), abs(s_low), 1e-6)
    gap = abs(s_high - s_low) / max_abs
    return gap < 0.50, f"{gap:.2f}"


def gate_random_baseline(returns, prices, n_random=100):
    """Gate 5: 100 random VIX threshold strategies. Pass if real Sharpe > 95th percentile."""
    real_s = sharpe(returns)
    vix = prices["VIX"].shift(1)
    ret_tqqq = daily_returns(prices, "TQQQ")
    ret_shy = daily_returns(prices, "SHY")

    random_sharpes = []
    for _ in range(n_random):
        thresh = np.random.uniform(10, 40)
        sig = (vix < thresh).fillna(False)
        r = sig * ret_tqqq + (~sig) * ret_shy
        random_sharpes.append(sharpe(r))

    pctile_95 = np.percentile(random_sharpes, 95)
    return real_s > pctile_95, f"real={real_s:.2f} vs p95={pctile_95:.2f}"


def run_all_gates(returns, prices):
    """Run all 5 validation gates."""
    vix = prices["VIX"]
    results = {}
    g1_pass, g1_detail = gate_permutation(returns)
    results["Perm"] = ("PASS" if g1_pass else "FAIL", g1_detail)

    g2_pass, g2_detail = gate_subperiod(returns)
    results["SubPd"] = ("PASS" if g2_pass else "FAIL", g2_detail)

    g3_pass, g3_detail = gate_outlier_removal(returns)
    results["Outlier"] = ("PASS" if g3_pass else "FAIL", g3_detail)

    g4_pass, g4_detail = gate_regime_balance(returns, vix)
    results["Regime"] = ("PASS" if g4_pass else "FAIL", g4_detail)

    g5_pass, g5_detail = gate_random_baseline(returns, prices)
    results["RandBL"] = ("PASS" if g5_pass else "FAIL", g5_detail)

    n_pass = sum(1 for v in results.values() if v[0] == "PASS")
    return results, n_pass


# ═══════════════════════════════════════════════════════════════════════
# MAIN BACKTEST
# ═══════════════════════════════════════════════════════════════════════
def run_variant(prices, variant_func):
    """Run a single variant and compute all metrics."""
    returns, risky_weight, name = variant_func(prices)

    # Drop NaN at start
    valid = returns.dropna()
    if len(valid) < 100:
        log.warning(f"{name}: Only {len(valid)} valid days, skipping")
        return None

    eq_10k = build_equity(valid, CAPITAL_LARGE)
    eq_645 = build_equity(valid, CAPITAL_SMALL)

    metrics = {
        "Variant": name,
        "Sharpe": round(sharpe(valid), 2),
        "Sortino": round(sortino(valid), 2),
        "CAGR%": round(cagr(eq_10k) * 100, 1),
        "MDD%": round(max_drawdown(eq_10k) * 100, 1),
        "Calmar": round(calmar(eq_10k, valid), 2),
        "WR%": round(monthly_win_rate(valid) * 100, 1),
        "Time_In_Mkt%": round(time_in_market(risky_weight) * 100, 1),
        "Worst_Mo%": round(worst_month(valid) * 100, 1),
        "Recovery_Days": recovery_time_days(eq_10k),
        "$10k_Final": round(eq_10k.iloc[-1], 0),
        "$645_Final": round(eq_645.iloc[-1], 0),
    }

    # Validation gates
    gates, n_pass = run_all_gates(valid, prices)
    metrics["Gates"] = f"{n_pass}/5"
    metrics["Gate_Details"] = gates

    return metrics, valid, eq_10k, name


def per_year_breakdown(returns, eq):
    """Per-year breakdown table."""
    rows = []
    for year in sorted(returns.index.year.unique()):
        yr_ret = returns[returns.index.year == year]
        yr_eq = eq[eq.index.year == year]
        rows.append({
            "Year": year,
            "Return%": round(yr_ret.sum() * 100, 1),
            "Sharpe": round(sharpe(yr_ret), 2),
            "MDD%": round(max_drawdown(yr_eq) * 100, 1),
            "WR_Mo%": round(monthly_win_rate(yr_ret) * 100, 1),
        })
    return pd.DataFrame(rows)


def main():
    log.info("=" * 70)
    log.info("Managed Volatility Leveraged ETF Strategy Backtest")
    log.info("=" * 70)

    # Download data
    prices = download_data()

    # Check required tickers
    required = ["TQQQ", "UPRO", "SOXL", "QQQ", "SPY", "SOXX", "TLT", "SHY", "GLD", "VIX"]
    missing = [t for t in required if t not in prices.columns]
    if missing:
        log.error(f"Missing tickers: {missing}")
        sys.exit(1)

    # Run all variants
    variants = [variant_a, variant_b, variant_c, variant_d, variant_e, variant_f, variant_g, variant_h]
    all_results = []
    best_variant = None
    best_sharpe = -999

    for vf in variants:
        log.info(f"\n{'─'*50}")
        result = run_variant(prices, vf)
        if result is None:
            continue
        metrics, returns, eq, name = result
        all_results.append(metrics)

        # Log gate details
        gates = metrics.pop("Gate_Details")
        log.info(f"{name}")
        log.info(f"  Sharpe={metrics['Sharpe']}, Sortino={metrics['Sortino']}, "
                 f"CAGR={metrics['CAGR%']}%, MDD={metrics['MDD%']}%, "
                 f"Calmar={metrics['Calmar']}, Gates={metrics['Gates']}")
        for gname, (status, detail) in gates.items():
            log.info(f"  Gate {gname}: {status} ({detail})")

        if metrics["Sharpe"] > best_sharpe:
            best_sharpe = metrics["Sharpe"]
            best_variant = (metrics, returns, eq, name)

    # ── Results Table ──
    log.info(f"\n{'='*70}")
    log.info("SORTED RESULTS (by Sharpe descending)")
    log.info(f"{'='*70}")

    df_results = pd.DataFrame(all_results)
    df_results = df_results.sort_values("Sharpe", ascending=False).reset_index(drop=True)
    log.info("\n" + df_results.to_string(index=False))

    # Save results
    df_results.to_csv(OUTPUT_DIR / "results_table.csv", index=False)
    log.info(f"\nResults saved to {OUTPUT_DIR / 'results_table.csv'}")

    # ── Per-Year Breakdown for Best Variant ──
    if best_variant:
        bm, br, beq, bname = best_variant
        log.info(f"\n{'='*70}")
        log.info(f"PER-YEAR BREAKDOWN: {bname}")
        log.info(f"{'='*70}")
        yearly = per_year_breakdown(br, beq)
        log.info("\n" + yearly.to_string(index=False))
        yearly.to_csv(OUTPUT_DIR / "best_variant_yearly.csv", index=False)

    # ── Benchmarks ──
    log.info(f"\n{'='*70}")
    log.info("BENCHMARKS (Buy & Hold)")
    log.info(f"{'='*70}")
    for ticker in ["TQQQ", "UPRO", "SOXL", "QQQ", "SPY"]:
        bh_ret = daily_returns(prices, ticker).dropna()
        bh_eq = build_equity(bh_ret, CAPITAL_LARGE)
        log.info(f"  {ticker:6s}: Sharpe={sharpe(bh_ret):.2f}, "
                 f"CAGR={cagr(bh_eq)*100:.1f}%, MDD={max_drawdown(bh_eq)*100:.1f}%, "
                 f"$10k→${bh_eq.iloc[-1]:,.0f}")

    # ── MLflow Logging ──
    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("managed_vol_leveraged_v1")

        with mlflow.start_run(run_name=f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            mlflow.log_param("start_date", START_DATE)
            mlflow.log_param("end_date", END_DATE)
            mlflow.log_param("n_variants", len(all_results))
            mlflow.log_param("tx_cost", TX_COST)
            mlflow.log_param("n_permutations", N_PERMUTATIONS)

            # Log best variant metrics
            if best_variant:
                bm = best_variant[0]
                mlflow.log_param("best_variant", bm["Variant"])
                mlflow.log_metric("best_sharpe", bm["Sharpe"])
                mlflow.log_metric("best_sortino", bm["Sortino"])
                mlflow.log_metric("best_cagr_pct", bm["CAGR%"])
                mlflow.log_metric("best_mdd_pct", bm["MDD%"])
                mlflow.log_metric("best_calmar", bm["Calmar"])
                mlflow.log_metric("best_wr_pct", bm["WR%"])
                mlflow.log_metric("best_10k_final", bm["$10k_Final"])

            # Log all variant Sharpes
            for r in all_results:
                safe_name = r["Variant"].split(":")[0].strip()
                mlflow.log_metric(f"sharpe_{safe_name}", r["Sharpe"])
                mlflow.log_metric(f"cagr_{safe_name}", r["CAGR%"])

            # Log artifacts
            mlflow.log_artifact(str(OUTPUT_DIR / "results_table.csv"))
            if (OUTPUT_DIR / "best_variant_yearly.csv").exists():
                mlflow.log_artifact(str(OUTPUT_DIR / "best_variant_yearly.csv"))

        log.info("MLflow run logged successfully")
    except Exception as e:
        log.warning(f"MLflow logging failed (non-fatal): {e}")

    log.info(f"\n{'='*70}")
    log.info("BACKTEST COMPLETE")
    log.info(f"{'='*70}")


if __name__ == "__main__":
    main()

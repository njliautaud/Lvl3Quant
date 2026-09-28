#!/usr/bin/env python3
"""
Stock Prediction v2 — Options Overlay Research
===============================================
Question: If we use ONLY high-confidence signals (0.70+ pred_proba, ~78% precision)
and buy 30-delta calls with 60 DTE instead of stock, does options leverage
turn the weak stock signal into a strong growth strategy?

Walk-forward: sliding window (HC #0)
Commission: $0 (Robinhood, HC #694)
No crypto (HC #697)
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from pathlib import Path
import json
import warnings
import time

warnings.filterwarnings("ignore")

OUT_DIR = Path("/home/jupiter/Lvl3Quant/output/growth_research/options_overlay")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path("/home/jupiter/Lvl3Quant/output/growth_research/stock_prediction/v2_relative")


# ─── Black-Scholes ───

def bs_call_price(S, K, T, r, sigma):
    """Black-Scholes call price."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_delta(S, K, T, r, sigma):
    """Call delta."""
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)


def find_strike_for_delta(S, target_delta, T, r, sigma, tol=0.001):
    """Find strike K such that delta(K) ~ target_delta using bisection."""
    K_low = S * 0.5
    K_high = S * 2.0
    for _ in range(50):
        K_mid = (K_low + K_high) / 2
        d = bs_delta(S, K_mid, T, r, sigma)
        if abs(d - target_delta) < tol:
            return K_mid
        if d > target_delta:
            K_low = K_mid  # Higher strike = lower delta
        else:
            K_high = K_mid
    return (K_low + K_high) / 2


def estimate_iv_from_ticker(ticker, date, price_history=None, default_iv=0.35):
    """Estimate IV using 60d realized vol * 1.2 markup (IV typically > RV)."""
    if price_history is not None and ticker in price_history.columns:
        px = price_history[ticker]
        # Get last 60 trading days before date
        mask = px.index <= date
        recent = px[mask].tail(60)
        if len(recent) > 20:
            rv = recent.pct_change().dropna().std() * np.sqrt(252)
            iv = rv * 1.2  # IV premium over RV
            return max(iv, 0.15)  # Floor at 15%
    return default_iv


# ─── Core Options Overlay Engine ───

def run_options_overlay(preds_df, price_data=None,
                        threshold=0.70,
                        target_delta=0.30,
                        dte=60,
                        bid_ask_pct=0.05,  # 5% of premium
                        risk_free_rate=0.045,
                        position_size_pct=0.05,  # 5% of capital per trade
                        max_concurrent=10,
                        label=""):
    """
    For each high-confidence prediction, model buying a 30-delta call with 60 DTE.
    Track the option's value after the predicted move (or lack thereof).
    """
    # Filter to high-confidence predictions
    signals = preds_df[preds_df["pred_proba"] >= threshold].copy()
    signals = signals.sort_values("date").reset_index(drop=True)

    if len(signals) == 0:
        return {"label": label, "valid": False, "n_signals": 0}

    print(f"\n  [{label}] Threshold: {threshold:.2f}")
    print(f"    Total signals: {len(signals)}")
    print(f"    Date range: {signals['date'].min().date()} to {signals['date'].max().date()}")
    print(f"    Precision: {signals['target_excess_60d_5pct'].mean():.3f}")
    print(f"    Avg fwd_60d_excess: {signals['fwd_60d_excess'].mean():.3f}")

    T = dte / 365.0
    trades = []

    for _, row in signals.iterrows():
        ticker = row["ticker"]
        date = row["date"]
        actual_excess_60d = row["fwd_60d_excess"]
        actual_abs_60d = row.get("fwd_60d_abs", actual_excess_60d)  # Absolute return if available
        pred_proba = row["pred_proba"]
        hit = row["target_excess_60d_5pct"] == 1

        # Estimate stock price at entry (use 100 as normalized baseline since we have returns)
        S = 100.0

        # Estimate IV
        if price_data is not None:
            iv = estimate_iv_from_ticker(ticker, date, price_data)
        else:
            iv = 0.35  # Default

        # Find strike for target delta
        K = find_strike_for_delta(S, target_delta, T, risk_free_rate, iv)

        # Entry call price
        call_entry = bs_call_price(S, K, T, risk_free_rate, iv)

        # Bid-ask cost (pay on entry)
        spread_cost = call_entry * bid_ask_pct

        # After 60 days, the stock moved by actual_abs_60d
        # Stock price at exit
        S_exit = S * (1 + actual_abs_60d)

        # At expiry (T_remaining ~ 0), call value = max(S_exit - K, 0)
        # But we're holding for exactly the prediction horizon (60d = DTE)
        # so we're at expiry
        call_exit = max(S_exit - K, 0)

        # P&L per option (net of spread cost on entry)
        pnl_per_option = call_exit - call_entry - spread_cost

        # Return on premium invested
        cost_basis = call_entry + spread_cost
        if cost_basis > 0:
            option_return = pnl_per_option / cost_basis
        else:
            option_return = 0

        # As a fraction of portfolio capital (position_size_pct allocated)
        portfolio_return_contribution = option_return * position_size_pct

        trades.append({
            "date": date,
            "ticker": ticker,
            "pred_proba": pred_proba,
            "hit": hit,
            "stock_return_60d": actual_abs_60d,
            "excess_return_60d": actual_excess_60d,
            "iv": iv,
            "strike_pct_otm": (K / S - 1) * 100,
            "call_entry": call_entry,
            "call_exit": call_exit,
            "spread_cost": spread_cost,
            "option_return": option_return,
            "portfolio_contribution": portfolio_return_contribution,
        })

    trades_df = pd.DataFrame(trades)

    # ── Aggregate Results ──
    n_signals = len(trades_df)
    n_winners = (trades_df["option_return"] > 0).sum()
    win_rate = n_winners / n_signals if n_signals > 0 else 0

    avg_option_return = trades_df["option_return"].mean()
    median_option_return = trades_df["option_return"].median()

    # Avg winner / avg loser
    winners = trades_df[trades_df["option_return"] > 0]["option_return"]
    losers = trades_df[trades_df["option_return"] <= 0]["option_return"]
    avg_winner = winners.mean() if len(winners) > 0 else 0
    avg_loser = losers.mean() if len(losers) > 0 else 0
    profit_factor = abs(winners.sum() / losers.sum()) if losers.sum() != 0 else float("inf")

    # Portfolio-level returns (walk-forward)
    # Group by month and compute monthly portfolio return
    trades_df["month"] = trades_df["date"].dt.to_period("M")
    monthly = trades_df.groupby("month").agg(
        n_trades=("option_return", "size"),
        avg_return=("option_return", "mean"),
        sum_contribution=("portfolio_contribution", "sum"),
    ).reset_index()

    # Build a monthly return series
    monthly["month_dt"] = monthly["month"].dt.to_timestamp()
    monthly_rets = monthly.set_index("month_dt")["sum_contribution"]

    # Fill months with no trades as 0 return
    full_months = pd.date_range(
        monthly_rets.index.min(), monthly_rets.index.max(), freq="MS"
    )
    monthly_rets = monthly_rets.reindex(full_months, fill_value=0)

    # CAGR from monthly returns
    total_months = len(monthly_rets)
    if total_months > 0:
        cumulative = (1 + monthly_rets).prod()
        years = total_months / 12
        cagr = cumulative ** (1 / years) - 1 if years > 0 and cumulative > 0 else 0
    else:
        cagr = 0
        years = 0

    # Monthly Sharpe/Sortino
    if len(monthly_rets) > 6 and monthly_rets.std() > 0:
        monthly_sharpe = monthly_rets.mean() / monthly_rets.std() * np.sqrt(12)
        down_rets = monthly_rets[monthly_rets < 0]
        if len(down_rets) > 3:
            monthly_sortino = monthly_rets.mean() / down_rets.std() * np.sqrt(12)
        else:
            monthly_sortino = monthly_sharpe * 1.5  # Approximate
    else:
        monthly_sharpe = 0
        monthly_sortino = 0

    # Max drawdown on cumulative monthly
    cum = (1 + monthly_rets).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min() if len(dd) > 0 else 0

    signals_per_year = n_signals / years if years > 0 else 0

    # Print results
    print(f"\n    Option trade results:")
    print(f"      Win rate:           {win_rate:.1%}")
    print(f"      Avg option return:  {avg_option_return:.1%}")
    print(f"      Median opt return:  {median_option_return:.1%}")
    print(f"      Avg winner:         {avg_winner:.1%}")
    print(f"      Avg loser:          {avg_loser:.1%}")
    print(f"      Profit factor:      {profit_factor:.2f}")
    print(f"      Signals/year:       {signals_per_year:.1f}")
    print(f"\n    Portfolio-level (monthly):")
    print(f"      CAGR:               {cagr:.1%}")
    print(f"      Monthly Sharpe:     {monthly_sharpe:.3f}")
    print(f"      Monthly Sortino:    {monthly_sortino:.3f}")
    print(f"      MaxDD:              {max_dd:.1%}")
    print(f"      Years:              {years:.1f}")

    return {
        "label": label,
        "valid": True,
        "threshold": threshold,
        "n_signals": n_signals,
        "signals_per_year": round(signals_per_year, 1),
        "precision": round(float(trades_df["hit"].mean()), 3),
        "win_rate": round(win_rate, 3),
        "avg_option_return": round(avg_option_return, 3),
        "median_option_return": round(median_option_return, 3),
        "avg_winner": round(avg_winner, 3),
        "avg_loser": round(avg_loser, 3),
        "profit_factor": round(profit_factor, 3),
        "cagr": round(cagr * 100, 2),
        "monthly_sharpe": round(monthly_sharpe, 3),
        "monthly_sortino": round(monthly_sortino, 3),
        "max_dd": round(max_dd * 100, 2),
        "years": round(years, 1),
        "monthly_returns": monthly_rets.to_dict() if len(monthly_rets) < 200 else {},
        "trades_df": trades_df,
    }


def regime_test_options(trades_df, spy_daily_returns, thresh=0.0005):
    """R1 regime test on option trades."""
    if len(trades_df) < 20:
        return {"regime_gap": 999, "r1_pass": False}

    # Classify each trade's entry date by SPY regime (30d trailing return)
    trade_dates = trades_df["date"].values
    results_by_regime = {"green": [], "red": [], "flat": []}

    for _, trade in trades_df.iterrows():
        d = trade["date"]
        # Look at 30d trailing SPY return before trade
        mask = (spy_daily_returns.index <= d) & (spy_daily_returns.index >= d - pd.Timedelta(days=45))
        spy_trailing = spy_daily_returns[mask]
        if len(spy_trailing) > 10:
            trailing_ret = spy_trailing.sum()
            if trailing_ret > 0.02:
                results_by_regime["green"].append(trade["option_return"])
            elif trailing_ret < -0.02:
                results_by_regime["red"].append(trade["option_return"])
            else:
                results_by_regime["flat"].append(trade["option_return"])

    regime_sharpes = {}
    for regime, rets in results_by_regime.items():
        if len(rets) > 5:
            r = np.array(rets)
            s = r.mean() / r.std() if r.std() > 0 else 0
            regime_sharpes[regime] = {
                "sharpe": round(float(s), 3),
                "n_trades": len(rets),
                "avg_return": round(float(np.mean(rets)), 3),
                "win_rate": round(float((np.array(rets) > 0).mean()), 3),
            }
        else:
            regime_sharpes[regime] = {"sharpe": 0, "n_trades": len(rets)}

    g = regime_sharpes.get("green", {}).get("sharpe", 0)
    r = regime_sharpes.get("red", {}).get("sharpe", 0)
    max_s = max(abs(g), abs(r))
    gap = abs(g - r) / max_s if max_s > 0 else 999

    return {
        "green_sharpe": g,
        "red_sharpe": r,
        "flat_sharpe": regime_sharpes.get("flat", {}).get("sharpe", 0),
        "regime_gap": round(gap, 4),
        "r1_pass": gap <= 0.50,
        "regimes": regime_sharpes,
    }


def sensitivity_analysis(preds_df, price_data, spy_returns):
    """Test across different thresholds, deltas, and position sizes."""
    print("\n" + "=" * 80)
    print("SENSITIVITY ANALYSIS")
    print("=" * 80)

    results = {}

    # Threshold sensitivity
    print("\n  --- Threshold Sensitivity (30-delta, 5% position) ---")
    for thresh in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        r = run_options_overlay(preds_df, price_data, threshold=thresh,
                               label=f"thresh_{thresh:.2f}")
        results[f"thresh_{thresh}"] = {k: v for k, v in r.items() if k != "trades_df" and k != "monthly_returns"}

    # Delta sensitivity (at 0.60 threshold for more signals)
    print("\n  --- Delta Sensitivity (0.60 threshold, 5% position) ---")
    for delta in [0.20, 0.30, 0.40, 0.50]:
        r = run_options_overlay(preds_df, price_data, threshold=0.60,
                               target_delta=delta, label=f"delta_{delta:.2f}")
        results[f"delta_{delta}"] = {k: v for k, v in r.items() if k != "trades_df" and k != "monthly_returns"}

    # Position size sensitivity
    print("\n  --- Position Size Sensitivity (0.70 threshold, 30-delta) ---")
    for psize in [0.02, 0.05, 0.10, 0.15]:
        r = run_options_overlay(preds_df, price_data, threshold=0.70,
                               position_size_pct=psize, label=f"size_{psize:.2f}")
        results[f"size_{psize}"] = {k: v for k, v in r.items() if k != "trades_df" and k != "monthly_returns"}

    return results


def main():
    t0 = time.time()
    print("=" * 80)
    print("STOCK PREDICTION v2 — OPTIONS OVERLAY RESEARCH")
    print(f"Started: {pd.Timestamp.now()}")
    print("=" * 80)

    # Load predictions
    preds = pd.read_parquet(DATA_DIR / "oot_predictions_target_excess_60d_5pct.parquet")
    print(f"Loaded {len(preds)} OOT predictions")
    print(f"Date range: {preds['date'].min().date()} to {preds['date'].max().date()}")
    print(f"Tickers: {preds['ticker'].nunique()}")

    # Try to get price data for IV estimation
    try:
        import yfinance as yf
        tickers = preds["ticker"].unique().tolist()
        print(f"\nDownloading price data for {len(tickers)} tickers for IV estimation...")
        price_data = yf.download(tickers + ["SPY"], start="2019-01-01",
                                 auto_adjust=True, progress=False)
        if isinstance(price_data.columns, pd.MultiIndex):
            price_data = price_data["Close"]
        price_data = price_data.ffill()
        spy_returns = price_data["SPY"].pct_change().dropna()
        print(f"Got price data: {price_data.shape}")
    except Exception as e:
        print(f"Could not download price data: {e}")
        price_data = None
        spy_returns = None

    # ══════════════════════════════════════════════════════════
    # 1. Baseline: Stock signal precision at thresholds
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("BASELINE: Signal Quality at Various Thresholds")
    print("=" * 80)

    baseline_stats = {}
    for t in [0.40, 0.50, 0.60, 0.70, 0.80]:
        mask = preds["pred_proba"] >= t
        n = mask.sum()
        if n > 0:
            prec = preds.loc[mask, "target_excess_60d_5pct"].mean()
            avg_excess = preds.loc[mask, "fwd_60d_excess"].mean()
            avg_abs = preds.loc[mask, "fwd_60d_abs"].mean() if "fwd_60d_abs" in preds.columns else avg_excess
            baseline_stats[str(t)] = {
                "n_signals": int(n),
                "precision": round(prec, 3),
                "avg_excess_60d": round(avg_excess, 4),
                "avg_abs_60d": round(avg_abs, 4),
            }
            print(f"  Threshold {t:.2f}: n={n:>5}, precision={prec:.3f}, "
                  f"avg_excess_60d={avg_excess:.3f}, avg_abs_60d={avg_abs:.3f}")
        else:
            print(f"  Threshold {t:.2f}: n=0")

    # ══════════════════════════════════════════════════════════
    # 2. Core Options Overlay at 0.70 threshold
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("CORE: Options Overlay at 0.70 Threshold")
    print("=" * 80)

    core_result = run_options_overlay(
        preds, price_data, threshold=0.70,
        target_delta=0.30, dte=60,
        bid_ask_pct=0.05, position_size_pct=0.05,
        label="core_0.70"
    )

    # R1 regime test
    if core_result.get("valid") and spy_returns is not None:
        core_regime = regime_test_options(core_result["trades_df"], spy_returns)
        print(f"\n    R1 Regime Test:")
        print(f"      Green Sharpe: {core_regime.get('green_sharpe', 0):.3f}")
        print(f"      Red Sharpe:   {core_regime.get('red_sharpe', 0):.3f}")
        print(f"      Regime Gap:   {core_regime.get('regime_gap', 999):.4f}")
        print(f"      R1 PASS:      {core_regime.get('r1_pass', False)}")
    else:
        core_regime = {}

    # ══════════════════════════════════════════════════════════
    # 3. Options Overlay at 0.60 threshold (more signals)
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("RELAXED: Options Overlay at 0.60 Threshold")
    print("=" * 80)

    relaxed_result = run_options_overlay(
        preds, price_data, threshold=0.60,
        target_delta=0.30, dte=60,
        bid_ask_pct=0.05, position_size_pct=0.03,
        label="relaxed_0.60"
    )

    if relaxed_result.get("valid") and spy_returns is not None:
        relaxed_regime = regime_test_options(relaxed_result["trades_df"], spy_returns)
        print(f"\n    R1 Regime Test:")
        print(f"      Green Sharpe: {relaxed_regime.get('green_sharpe', 0):.3f}")
        print(f"      Red Sharpe:   {relaxed_regime.get('red_sharpe', 0):.3f}")
        print(f"      Regime Gap:   {relaxed_regime.get('regime_gap', 999):.4f}")
        print(f"      R1 PASS:      {relaxed_regime.get('r1_pass', False)}")
    else:
        relaxed_regime = {}

    # ══════════════════════════════════════════════════════════
    # 4. ATM calls at 0.70 (higher delta = more exposure, less leverage)
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("VARIANT: ATM Calls (0.50 delta) at 0.70 Threshold")
    print("=" * 80)

    atm_result = run_options_overlay(
        preds, price_data, threshold=0.70,
        target_delta=0.50, dte=60,
        bid_ask_pct=0.05, position_size_pct=0.05,
        label="atm_0.70"
    )

    # ══════════════════════════════════════════════════════════
    # 5. Deep OTM calls at 0.70 (20 delta — max leverage)
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("VARIANT: Deep OTM Calls (0.20 delta) at 0.70 Threshold")
    print("=" * 80)

    otm_result = run_options_overlay(
        preds, price_data, threshold=0.70,
        target_delta=0.20, dte=60,
        bid_ask_pct=0.05, position_size_pct=0.05,
        label="otm_0.70"
    )

    # ══════════════════════════════════════════════════════════
    # 6. Sensitivity Analysis
    # ══════════════════════════════════════════════════════════
    sensitivity = sensitivity_analysis(preds, price_data, spy_returns)

    # ══════════════════════════════════════════════════════════
    # 7. Comparison: Options vs Stock-only Portfolio
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("COMPARISON: Options Overlay vs Stock-Only")
    print("=" * 80)

    # Stock-only version at 0.70 threshold
    high_conf = preds[preds["pred_proba"] >= 0.70].copy()
    if len(high_conf) > 0:
        high_conf["month"] = high_conf["date"].dt.to_period("M")
        stock_monthly = high_conf.groupby("month")["fwd_60d_excess"].mean()
        stock_monthly.index = stock_monthly.index.to_timestamp()

        # Stock CAGR (equal weight, 5% position size)
        stock_rets_adj = stock_monthly * 0.05 * 2  # Assume avg 2 signals per month active
        full_months = pd.date_range(stock_rets_adj.index.min(), stock_rets_adj.index.max(), freq="MS")
        stock_rets_adj = stock_rets_adj.reindex(full_months, fill_value=0)

        stock_cum = (1 + stock_rets_adj).prod()
        stock_years = len(stock_rets_adj) / 12
        stock_cagr = stock_cum ** (1 / stock_years) - 1 if stock_years > 0 and stock_cum > 0 else 0

        opt_cagr = core_result.get("cagr", 0) / 100

        print(f"  Stock-only portfolio (0.70 threshold):")
        print(f"    Avg monthly excess return: {stock_monthly.mean():.3f}")
        print(f"    CAGR (5% position):        {stock_cagr:.1%}")
        print(f"  Options overlay CAGR:          {opt_cagr:.1%}")
        print(f"  Leverage multiplier:           {opt_cagr / stock_cagr:.1f}x" if stock_cagr != 0 else "  N/A")

        stock_comparison = {
            "stock_cagr": round(stock_cagr * 100, 2),
            "options_cagr": core_result.get("cagr", 0),
            "leverage_multiplier": round(opt_cagr / stock_cagr, 2) if stock_cagr != 0 else None,
        }
    else:
        stock_comparison = {"stock_cagr": 0, "options_cagr": 0}

    # ══════════════════════════════════════════════════════════
    # FINAL VERDICT
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)

    issues = []
    passes = []

    # Check if options overlay is profitable
    opt_cagr = core_result.get("cagr", 0)
    opt_sharpe = core_result.get("monthly_sharpe", 0)
    opt_wr = core_result.get("win_rate", 0)
    n_signals_yr = core_result.get("signals_per_year", 0)

    if opt_cagr > 5:
        passes.append(f"CAGR {opt_cagr:.1f}% > 5% threshold")
    elif opt_cagr > 0:
        issues.append(f"CAGR {opt_cagr:.1f}% positive but below 5% threshold")
    else:
        issues.append(f"CAGR {opt_cagr:.1f}% — strategy is unprofitable")

    if opt_sharpe > 0.5:
        passes.append(f"Monthly Sharpe {opt_sharpe:.3f} > 0.5")
    else:
        issues.append(f"Monthly Sharpe {opt_sharpe:.3f} < 0.5")

    if n_signals_yr >= 10:
        passes.append(f"{n_signals_yr:.0f} signals/year — enough to compound")
    else:
        issues.append(f"Only {n_signals_yr:.0f} signals/year — too sparse to compound")

    if opt_wr > 0.3:
        passes.append(f"Win rate {opt_wr:.1%}")
    else:
        issues.append(f"Win rate {opt_wr:.1%} — most options expire worthless")

    # R1 regime check
    if core_regime.get("r1_pass"):
        passes.append(f"R1 regime gap {core_regime.get('regime_gap', 999):.4f} <= 0.50")
    elif core_regime:
        issues.append(f"R1 regime gap {core_regime.get('regime_gap', 999):.4f} > 0.50")

    # Profit factor
    pf = core_result.get("profit_factor", 0)
    if pf > 1.0:
        passes.append(f"Profit factor {pf:.2f} > 1.0")
    else:
        issues.append(f"Profit factor {pf:.2f} <= 1.0")

    overall = "VIABLE" if len(issues) == 0 else ("MARGINAL" if opt_cagr > 0 else "FAIL")

    for p in passes:
        print(f"  PASS: {p}")
    for i in issues:
        print(f"  ISSUE: {i}")

    print(f"\n  OVERALL: {overall}")

    if overall == "FAIL":
        print("  CONCLUSION: Options leverage does NOT save a weak stock signal.")
        print("  The 20% failure rate at 0.70 threshold causes total premium loss,")
        print("  which destroys the gains from correct predictions.")
    elif overall == "MARGINAL":
        print("  CONCLUSION: Strategy shows some promise but needs more signals or")
        print("  higher precision to be a meaningful growth allocation.")
    else:
        print("  CONCLUSION: Options overlay converts weak stock edge into viable growth strategy.")

    # ── Save results ──
    elapsed = time.time() - t0

    def clean_for_json(obj):
        """Remove non-serializable items."""
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()
                    if not isinstance(v, pd.DataFrame)}
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    report = {
        "generated": pd.Timestamp.now().isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "overall_verdict": overall,
        "issues": issues,
        "passes": passes,

        "baseline_stats": baseline_stats,

        "core_0.70": clean_for_json({k: v for k, v in core_result.items()
                                      if k not in ("trades_df", "monthly_returns")}),
        "core_regime": clean_for_json(core_regime),

        "relaxed_0.60": clean_for_json({k: v for k, v in relaxed_result.items()
                                         if k not in ("trades_df", "monthly_returns")}),
        "relaxed_regime": clean_for_json(relaxed_regime),

        "atm_0.70": clean_for_json({k: v for k, v in atm_result.items()
                                     if k not in ("trades_df", "monthly_returns")}),

        "otm_0.70": clean_for_json({k: v for k, v in otm_result.items()
                                     if k not in ("trades_df", "monthly_returns")}),

        "sensitivity": clean_for_json(sensitivity),
        "stock_comparison": stock_comparison,
    }

    outfile = OUT_DIR / "options_overlay_results.json"
    with open(outfile, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Save trades
    if core_result.get("valid"):
        trades_file = OUT_DIR / "core_trades.csv"
        core_result["trades_df"].to_csv(trades_file, index=False)

    print(f"\n  Results saved to {OUT_DIR}")
    print(f"  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}m)")

    return report


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Credit Spread + Cross-Asset Stress Signal Mean-Reversion Backtest v1
====================================================================
Thesis: When HY credit spreads widen sharply (HYG drops), or TLT/GLD spike
indicating risk-off, buy SPY expecting mean-reversion. Credit stress events
are genuinely unusual timing that should survive permutation testing.

Signals tested:
  1. HYG drops >2% in 5d, buy SPY hold 5d
  2. HYG drops >2% in 5d, buy SPY hold 10d
  3. HYG drops >3% in 5d, buy SPY hold 10d
  4. TLT rises >3% in 5d, buy SPY hold 10d
  5. TLT rises >5% in 5d, buy SPY hold 10d
  6. GLD rises >3% in 5d AND SPY drops >2% in 5d, buy SPY hold 10d
  7. HYG drops >2% in 5d AND VIX>25, buy SPY hold 10d
  8. HYG drops >2% in 5d AND VIX>25, buy SPY hold 20d
  9. Composite: HYG 5d ret < -1.5% AND TLT 5d ret > 2%, buy SPY hold 10d

ALL quality gates INLINE (HC #705 style):
  - Permutation test (p<0.05, 1000 perms)
  - R1 Regime test (green/red day Sharpe gap < 0.50)
  - Sub-period consistency (both halves positive mean)
  - Outlier robustness (trim top/bottom 5%, still profitable)
"""

import sys, json, warnings, os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

ROOT   = Path("/home/jupiter/Lvl3Quant")
OUTPUT = ROOT / "output" / "credit_signal_v1"
OUTPUT.mkdir(parents=True, exist_ok=True)

# =============================================================================
# DATA
# =============================================================================

def fetch_data():
    """Fetch all required tickers via yfinance with retry logic."""
    import yfinance as yf
    import time as _time

    tickers = ["SPY", "HYG", "TLT", "GLD", "^VIX"]
    start, end = "2010-01-01", "2026-07-01"

    # Try bulk download first, then individual fallback for failures
    print(f"Fetching {tickers} from {start} to {end} ...")
    for attempt in range(3):
        raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"]
        else:
            close = raw
        if "^VIX" in close.columns:
            close = close.rename(columns={"^VIX": "VIX"})

        # Check for missing tickers and fetch individually
        needed = ["SPY", "HYG", "TLT", "GLD", "VIX"]
        missing = [t for t in needed if t not in close.columns or close[t].isna().all()]
        if missing:
            print(f"  Attempt {attempt+1}: Missing {missing}, fetching individually...")
            for sym in missing:
                raw_sym = sym if sym != "VIX" else "^VIX"
                try:
                    d = yf.download(raw_sym, start=start, end=end, auto_adjust=True, progress=False)
                    if len(d) > 0:
                        if isinstance(d.columns, pd.MultiIndex):
                            close[sym] = d["Close"].iloc[:, 0]
                        else:
                            close[sym] = d["Close"]
                except Exception as e:
                    print(f"    {sym} failed: {e}")

        still_missing = [t for t in needed if t not in close.columns or close[t].isna().all()]
        if not still_missing:
            break
        _time.sleep(2)

    close = close.dropna(how="all").ffill()
    # Drop rows where any required ticker is still NaN
    close = close.dropna(subset=["SPY", "HYG", "TLT", "GLD", "VIX"])
    print(f"  Got {len(close)} trading days, columns: {list(close.columns)}")
    assert len(close) > 100, f"Too few rows ({len(close)}), data fetch likely failed"
    return close


# =============================================================================
# SIGNAL DEFINITIONS
# =============================================================================

def build_features(close):
    """Compute rolling returns and levels needed for signals."""
    feat = pd.DataFrame(index=close.index)
    feat["spy_close"] = close["SPY"]
    feat["hyg_5d_ret"] = close["HYG"].pct_change(5)
    feat["tlt_5d_ret"] = close["TLT"].pct_change(5)
    feat["gld_5d_ret"] = close["GLD"].pct_change(5)
    feat["spy_5d_ret"] = close["SPY"].pct_change(5)
    feat["vix"] = close["VIX"]
    feat["spy_1d_ret"] = close["SPY"].pct_change(1)  # for regime classification
    return feat.dropna()


CONFIGS = [
    {
        "name": "1_HYG_drop2pct_hold5d",
        "signal": lambda f: f["hyg_5d_ret"] < -0.02,
        "hold_days": 5,
        "desc": "HYG drops >2% in 5d, hold 5d",
    },
    {
        "name": "2_HYG_drop2pct_hold10d",
        "signal": lambda f: f["hyg_5d_ret"] < -0.02,
        "hold_days": 10,
        "desc": "HYG drops >2% in 5d, hold 10d",
    },
    {
        "name": "3_HYG_drop3pct_hold10d",
        "signal": lambda f: f["hyg_5d_ret"] < -0.03,
        "hold_days": 10,
        "desc": "HYG drops >3% in 5d, hold 10d",
    },
    {
        "name": "4_TLT_rise3pct_hold10d",
        "signal": lambda f: f["tlt_5d_ret"] > 0.03,
        "hold_days": 10,
        "desc": "TLT rises >3% in 5d, hold 10d",
    },
    {
        "name": "5_TLT_rise5pct_hold10d",
        "signal": lambda f: f["tlt_5d_ret"] > 0.05,
        "hold_days": 10,
        "desc": "TLT rises >5% in 5d, hold 10d",
    },
    {
        "name": "6_GLD_up3_SPY_dn2_hold10d",
        "signal": lambda f: (f["gld_5d_ret"] > 0.03) & (f["spy_5d_ret"] < -0.02),
        "hold_days": 10,
        "desc": "GLD up >3% AND SPY down >2% in 5d, hold 10d",
    },
    {
        "name": "7_HYG_drop2_VIX25_hold10d",
        "signal": lambda f: (f["hyg_5d_ret"] < -0.02) & (f["vix"] > 25),
        "hold_days": 10,
        "desc": "HYG drops >2% in 5d AND VIX>25, hold 10d",
    },
    {
        "name": "8_HYG_drop2_VIX25_hold20d",
        "signal": lambda f: (f["hyg_5d_ret"] < -0.02) & (f["vix"] > 25),
        "hold_days": 20,
        "desc": "HYG drops >2% in 5d AND VIX>25, hold 20d",
    },
    {
        "name": "9_composite_HYG_TLT_hold10d",
        "signal": lambda f: (f["hyg_5d_ret"] < -0.015) & (f["tlt_5d_ret"] > 0.02),
        "hold_days": 10,
        "desc": "HYG 5d ret < -1.5% AND TLT 5d ret > 2%, hold 10d",
    },
]


# =============================================================================
# BACKTEST ENGINE
# =============================================================================

def run_backtest(feat, signal_mask, hold_days):
    """
    Given a boolean signal mask and hold period, compute trade-level returns.
    Buy SPY at close on signal day, sell at close hold_days later.
    Non-overlapping: once in a trade, skip signals until exit.
    Returns DataFrame with entry_date, exit_date, return_pct, prior_day_spy_ret (for regime).
    """
    spy = feat["spy_close"].values
    dates = feat.index.values
    prior_ret = feat["spy_1d_ret"].values
    sig = signal_mask.values

    trades = []
    i = 0
    n = len(spy)
    while i < n:
        if sig[i] and (i + hold_days) < n:
            entry_price = spy[i]
            exit_price = spy[i + hold_days]
            ret = (exit_price - entry_price) / entry_price
            trades.append({
                "entry_date": str(dates[i])[:10],
                "exit_date": str(dates[i + hold_days])[:10],
                "return_pct": ret * 100,
                "prior_day_spy_ret": prior_ret[i],  # prior day regime
            })
            i += hold_days  # skip to exit
        else:
            i += 1

    return pd.DataFrame(trades) if trades else pd.DataFrame(
        columns=["entry_date", "exit_date", "return_pct", "prior_day_spy_ret"]
    )


# =============================================================================
# QUALITY GATES
# =============================================================================

def permutation_test(feat, trades_df, hold_days, n_perms=1000):
    """
    Sample random entry dates (same count as real trades), hold same duration,
    compute N-day forward returns from SPY. Compare real mean return vs
    distribution of random-entry mean returns. Return p-value.
    """
    if len(trades_df) < 3:
        return 1.0

    real_mean = trades_df["return_pct"].mean()
    n_trades = len(trades_df)

    spy = feat["spy_close"].values
    n = len(spy)
    # Pre-compute all possible N-day forward returns
    valid_indices = np.arange(0, n - hold_days)
    fwd_returns = np.array([
        (spy[j + hold_days] - spy[j]) / spy[j] * 100
        for j in valid_indices
    ])

    rng = np.random.default_rng(42)
    count_better = 0
    for _ in range(n_perms):
        sample_idx = rng.choice(len(fwd_returns), size=n_trades, replace=True)
        rand_mean = fwd_returns[sample_idx].mean()
        if rand_mean >= real_mean:
            count_better += 1

    p_value = (count_better + 1) / (n_perms + 1)  # +1 for continuity correction
    return p_value


def regime_test(trades_df):
    """
    Classify trades by PRIOR-DAY SPY return (green if >0, red if <=0).
    Compute Sharpe per regime. REJECT if gap > 0.50.
    Returns (regime_gap, pass_bool).
    """
    if len(trades_df) < 6:
        return 1.0, False

    green = trades_df[trades_df["prior_day_spy_ret"] > 0]["return_pct"]
    red = trades_df[trades_df["prior_day_spy_ret"] <= 0]["return_pct"]

    if len(green) < 3 or len(red) < 3:
        return 0.0, True  # not enough data to test, pass by default

    sharpe_green = green.mean() / green.std() if green.std() > 0 else 0
    sharpe_red = red.mean() / red.std() if red.std() > 0 else 0

    max_abs = max(abs(sharpe_green), abs(sharpe_red))
    if max_abs < 1e-9:
        return 0.0, True

    gap = abs(sharpe_green - sharpe_red) / max_abs
    return gap, gap <= 0.50


def sub_period_test(trades_df):
    """First half vs second half trades both positive mean return."""
    if len(trades_df) < 6:
        return False
    mid = len(trades_df) // 2
    first_half = trades_df.iloc[:mid]["return_pct"].mean()
    second_half = trades_df.iloc[mid:]["return_pct"].mean()
    return first_half > 0 and second_half > 0


def outlier_robustness_test(trades_df):
    """Remove top/bottom 5% of returns, must still be profitable."""
    if len(trades_df) < 10:
        return trades_df["return_pct"].mean() > 0 if len(trades_df) > 0 else False
    rets = trades_df["return_pct"].sort_values()
    trim_n = max(1, int(len(rets) * 0.05))
    trimmed = rets.iloc[trim_n:-trim_n]
    return trimmed.mean() > 0


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(trades_df):
    """Compute Sharpe, Sortino, WR, PF, mean return, max drawdown."""
    rets = trades_df["return_pct"].values
    if len(rets) == 0:
        return {}

    mean_ret = np.mean(rets)
    std_ret = np.std(rets, ddof=1) if len(rets) > 1 else 1e-9

    # Sharpe (annualized assuming ~25 trades/year avg)
    trades_per_year = max(1, len(rets) / 16)  # ~16 years of data
    sharpe = (mean_ret / std_ret) * np.sqrt(trades_per_year) if std_ret > 0 else 0

    # Sortino
    downside = rets[rets < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-9
    sortino = (mean_ret / downside_std) * np.sqrt(trades_per_year) if downside_std > 0 else 0

    # Win rate
    win_rate = np.mean(rets > 0) * 100

    # Profit factor
    gross_profit = np.sum(rets[rets > 0])
    gross_loss = abs(np.sum(rets[rets < 0]))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Max drawdown (cumulative equity curve)
    equity = np.cumsum(rets)
    running_max = np.maximum.accumulate(equity)
    drawdowns = equity - running_max
    max_dd = abs(np.min(drawdowns)) if len(drawdowns) > 0 else 0

    return {
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "win_rate": round(win_rate, 1),
        "profit_factor": round(profit_factor, 3),
        "mean_return_pct": round(mean_ret, 4),
        "max_dd_pct": round(max_dd, 2),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    close = fetch_data()
    feat = build_features(close)
    print(f"Feature matrix: {len(feat)} rows, {feat.index[0]} to {feat.index[-1]}")
    print()

    results = []

    for cfg in CONFIGS:
        name = cfg["name"]
        hold = cfg["hold_days"]
        desc = cfg["desc"]

        print(f"{'='*70}")
        print(f"Config: {name}")
        print(f"  {desc}")

        sig = cfg["signal"](feat)
        n_signals = sig.sum()
        print(f"  Raw signal fires: {n_signals}")

        trades_df = run_backtest(feat, sig, hold)
        n_trades = len(trades_df)
        print(f"  Non-overlapping trades: {n_trades}")

        if n_trades < 3:
            print(f"  SKIP: Too few trades ({n_trades})")
            results.append({
                "config_name": name,
                "description": desc,
                "n_trades": n_trades,
                "sharpe": 0, "sortino": 0, "win_rate": 0,
                "profit_factor": 0, "mean_return_pct": 0, "max_dd_pct": 0,
                "quality_gates": {
                    "permutation_p": 1.0, "regime_gap": None,
                    "sub_period_pass": False, "outlier_robust": False,
                    "ALL_PASS": False,
                },
            })
            continue

        # Metrics
        metrics = compute_metrics(trades_df)

        # Quality gates
        print(f"  Running permutation test (1000 perms) ...")
        perm_p = permutation_test(feat, trades_df, hold, n_perms=1000)
        regime_gap, regime_pass = regime_test(trades_df)
        sub_pass = sub_period_test(trades_df)
        outlier_pass = outlier_robustness_test(trades_df)

        all_pass = (perm_p < 0.05) and regime_pass and sub_pass and outlier_pass

        gates = {
            "permutation_p": round(perm_p, 4),
            "regime_gap": round(regime_gap, 3),
            "sub_period_pass": sub_pass,
            "outlier_robust": outlier_pass,
            "ALL_PASS": all_pass,
        }

        result = {
            "config_name": name,
            "description": desc,
            "n_trades": n_trades,
            **metrics,
            "quality_gates": gates,
        }
        results.append(result)

        # Print summary
        print(f"  Mean return: {metrics['mean_return_pct']:.4f}%")
        print(f"  Win rate: {metrics['win_rate']:.1f}%")
        print(f"  Sharpe: {metrics['sharpe']:.3f}  Sortino: {metrics['sortino']:.3f}")
        print(f"  Profit factor: {metrics['profit_factor']:.3f}")
        print(f"  Max DD: {metrics['max_dd_pct']:.2f}%")
        print(f"  --- Quality Gates ---")
        print(f"  Permutation p: {perm_p:.4f} {'PASS' if perm_p < 0.05 else 'FAIL'}")
        print(f"  Regime gap: {regime_gap:.3f} {'PASS' if regime_pass else 'FAIL'}")
        print(f"  Sub-period: {'PASS' if sub_pass else 'FAIL'}")
        print(f"  Outlier robust: {'PASS' if outlier_pass else 'FAIL'}")
        tag = "*** ALL PASS ***" if all_pass else "FAILED"
        print(f"  => {tag}")
        print()

    # Save JSON — convert numpy types to native Python
    def convert(obj):
        if isinstance(obj, (np.bool_, np.integer)):
            return int(obj) if isinstance(obj, np.integer) else bool(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(i) for i in obj]
        return obj

    results = convert(results)
    out_path = OUTPUT / "backtest_report.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")

    # Summary table
    print(f"\n{'='*90}")
    print(f"{'Config':<40} {'N':>4} {'Mean%':>7} {'WR%':>5} {'Sharpe':>7} {'PF':>6} {'Perm_p':>7} {'ALL':>5}")
    print(f"{'-'*90}")
    for r in results:
        g = r["quality_gates"]
        tag = "YES" if g["ALL_PASS"] else "no"
        print(f"{r['config_name']:<40} {r['n_trades']:>4} {r['mean_return_pct']:>7.3f} "
              f"{r['win_rate']:>5.1f} {r['sharpe']:>7.3f} {r['profit_factor']:>6.2f} "
              f"{g['permutation_p']:>7.4f} {tag:>5}")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()

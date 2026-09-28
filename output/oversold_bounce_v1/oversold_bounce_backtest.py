#!/usr/bin/env python3
"""
Oversold High-Beta Stock Bounce Backtest
HC #705 — All adversarial checks built in.

Tests whether stocks that drop significantly tend to bounce.
Uses equity returns (not options). Signal validation only.

Permutation test: random DATE entry (not return shuffling).
"""

import json
import warnings
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

warnings.filterwarnings("ignore")

# ─── CONFIG ───────────────────────────────────────────────────────────────
TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "JPM", "V", "UNH",
    "JNJ", "WMT", "PG", "HD", "MA", "BAC", "XOM", "DIS", "NFLX", "AMD",
]
START = "2015-01-01"
END = "2026-07-14"
HOLD_PERIODS = [1, 3, 5, 10]
N_PERMS = 200
PERM_P_THRESHOLD = 0.05
REGIME_GAP_THRESHOLD = 0.50
OUTLIER_SHARPE_DROP_MAX = 0.50
TICKER_CONC_MAX = 0.25
MIN_TRADES = 20  # minimum trades to evaluate a variant

OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/oversold_bounce_v1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def download_data():
    """Download price data for all tickers + SPY."""
    all_tickers = TICKERS + ["SPY"]
    print(f"Downloading {len(all_tickers)} tickers from {START} to {END}...")
    data = yf.download(all_tickers, start=START, end=END, auto_adjust=True, progress=False)
    # yfinance returns MultiIndex columns: (Price, Ticker)
    closes = data["Close"]
    opens = data["Open"]
    highs = data["High"]
    lows = data["Low"]
    print(f"  Got {len(closes)} trading days, {closes.columns.tolist()[:5]}... tickers")
    return closes, opens, highs, lows


def compute_rsi(series, period=14):
    """Standard RSI calculation."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def generate_signals(closes, opens):
    """Generate entry signals for all 6 variants. Returns dict of DataFrames (bool masks)."""
    signals = {}

    # Weekly return (5 trading days)
    weekly_ret = closes / closes.shift(5) - 1

    # Daily return
    daily_ret = closes / closes.shift(1) - 1

    # Gap: open vs prior close
    gap = opens / closes.shift(1) - 1

    # RSI(14) for each ticker
    rsi_all = pd.DataFrame(index=closes.index, columns=closes.columns)
    for t in closes.columns:
        if t == "SPY":
            continue
        rsi_all[t] = compute_rsi(closes[t], 14)

    tickers_no_spy = [t for t in closes.columns if t != "SPY"]

    # 1. drop_5pct_1wk
    signals["drop_5pct_1wk"] = weekly_ret[tickers_no_spy] <= -0.05

    # 2. drop_10pct_1wk
    signals["drop_10pct_1wk"] = weekly_ret[tickers_no_spy] <= -0.10

    # 3. drop_5pct_1day
    signals["drop_5pct_1day"] = daily_ret[tickers_no_spy] <= -0.05

    # 4. drop_10pct_1day
    signals["drop_10pct_1day"] = daily_ret[tickers_no_spy] <= -0.10

    # 5. drop_5pct_rsi_low: 5% weekly drop AND RSI < 30
    signals["drop_5pct_rsi_low"] = (weekly_ret[tickers_no_spy] <= -0.05) & (rsi_all[tickers_no_spy] < 30)

    # 6. gap_down_5pct
    signals["gap_down_5pct"] = gap[tickers_no_spy] <= -0.05

    return signals


def compute_forward_returns(opens, hold_days):
    """Compute forward returns from next-day open (buy at next open, sell at open N days later).
    Entry: open on day t+1. Exit: open on day t+1+hold_days."""
    tickers_no_spy = [t for t in opens.columns if t != "SPY"]
    entry_price = opens[tickers_no_spy].shift(-1)  # next day open
    exit_price = opens[tickers_no_spy].shift(-1 - hold_days)  # open N days after entry
    fwd_ret = exit_price / entry_price - 1
    return fwd_ret


def get_spy_regime(closes):
    """Classify each day by SPY regime: green (>+0.3%), red (<-0.3%), flat."""
    spy_ret = closes["SPY"].pct_change()
    regime = pd.Series("flat", index=closes.index)
    regime[spy_ret > 0.003] = "green"
    regime[spy_ret < -0.003] = "red"
    return regime


def collect_trades(signal_mask, fwd_ret):
    """Collect all (date, ticker, return) triples where signal fired and forward return is valid."""
    trades = []
    for t in signal_mask.columns:
        sig_dates = signal_mask.index[signal_mask[t].fillna(False)]
        for d in sig_dates:
            if d in fwd_ret.index and t in fwd_ret.columns:
                r = fwd_ret.loc[d, t]
                if not np.isnan(r):
                    trades.append({"date": d, "ticker": t, "return": r})
    return pd.DataFrame(trades)


def permutation_test_random_dates(trades_df, fwd_ret, n_perms=200):
    """
    Permutation test using random DATE entry (HC #705 fix).
    For each permutation: randomly pick N dates from the full date range,
    for each pick a random ticker, get that forward return, compute mean.
    Compare observed signal mean to distribution of random-date means.
    """
    if len(trades_df) == 0:
        return 1.0, 0.0

    observed_mean = trades_df["return"].mean()
    n_trades = len(trades_df)

    # Build pool of all valid (date, ticker) pairs with valid returns
    valid_mask = fwd_ret.notna()
    valid_pairs = []
    for t in fwd_ret.columns:
        valid_dates = fwd_ret.index[valid_mask[t]]
        for d in valid_dates:
            valid_pairs.append((d, t))

    if len(valid_pairs) < n_trades:
        return 1.0, 0.0

    valid_pairs_arr = np.array(valid_pairs, dtype=object)
    rng = np.random.default_rng(42)

    perm_means = np.zeros(n_perms)
    for i in range(n_perms):
        idx = rng.choice(len(valid_pairs_arr), size=n_trades, replace=False)
        sampled = valid_pairs_arr[idx]
        rets = np.array([fwd_ret.loc[d, t] for d, t in sampled])
        perm_means[i] = rets.mean()

    # One-sided p-value: fraction of permutations with mean >= observed
    p_value = np.mean(perm_means >= observed_mean)
    return p_value, observed_mean


def regime_test(trades_df, spy_regime):
    """R1 regime test: stratify by SPY green/red/flat, check |gap| < 0.50."""
    if len(trades_df) < 10:
        return None, None, None, None

    trades_df = trades_df.copy()
    trades_df["regime"] = trades_df["date"].map(spy_regime)

    regime_sharpes = {}
    for r in ["green", "red", "flat"]:
        subset = trades_df[trades_df["regime"] == r]
        if len(subset) >= 5:
            mean_r = subset["return"].mean()
            std_r = subset["return"].std()
            regime_sharpes[r] = mean_r / std_r if std_r > 0 else 0.0
        else:
            regime_sharpes[r] = None

    # Compute gap between green and red
    s_green = regime_sharpes.get("green")
    s_red = regime_sharpes.get("red")

    if s_green is not None and s_red is not None:
        denom = max(abs(s_green), abs(s_red))
        gap = abs(s_green - s_red) / denom if denom > 0 else 0.0
    else:
        gap = None

    passed = gap is not None and gap < REGIME_GAP_THRESHOLD
    return regime_sharpes, gap, passed, trades_df


def sub_period_test(trades_df):
    """Pre-2020 vs 2020+ both must be profitable."""
    if len(trades_df) < 10:
        return None, None, None

    cutoff = pd.Timestamp("2020-01-01")
    pre = trades_df[trades_df["date"] < cutoff]
    post = trades_df[trades_df["date"] >= cutoff]

    pre_mean = pre["return"].mean() if len(pre) >= 5 else None
    post_mean = post["return"].mean() if len(post) >= 5 else None

    passed = (pre_mean is not None and pre_mean > 0 and
              post_mean is not None and post_mean > 0)
    return pre_mean, post_mean, passed


def outlier_removal_test(trades_df):
    """Remove top 5 trades by return, recalc Sharpe. Must not drop >50%."""
    if len(trades_df) < 10:
        return None, None, None

    rets = trades_df["return"].values
    full_sharpe = rets.mean() / rets.std() if rets.std() > 0 else 0.0

    # Remove top 5 returns
    sorted_idx = np.argsort(rets)
    trimmed = rets[sorted_idx[:-5]]
    trimmed_sharpe = trimmed.mean() / trimmed.std() if trimmed.std() > 0 else 0.0

    if full_sharpe > 0:
        drop_pct = 1 - trimmed_sharpe / full_sharpe
    else:
        drop_pct = 0.0

    passed = drop_pct < OUTLIER_SHARPE_DROP_MAX
    return full_sharpe, trimmed_sharpe, drop_pct, passed


def ticker_concentration_test(trades_df):
    """No single ticker > 25% of total P&L."""
    if len(trades_df) < 10:
        return None, None

    pnl_by_ticker = trades_df.groupby("ticker")["return"].sum()
    total_pnl = pnl_by_ticker.sum()
    if total_pnl == 0:
        return {}, False

    conc = (pnl_by_ticker / total_pnl).to_dict()
    max_conc = max(abs(v) for v in conc.values()) if conc else 0
    passed = max_conc < TICKER_CONC_MAX
    return conc, max_conc, passed


def win_rate_pf(trades_df):
    """Win rate and profit factor."""
    if len(trades_df) == 0:
        return 0, 0, False

    wins = trades_df[trades_df["return"] > 0]
    losses = trades_df[trades_df["return"] < 0]

    wr = len(wins) / len(trades_df)
    gross_profit = wins["return"].sum() if len(wins) > 0 else 0
    gross_loss = abs(losses["return"].sum()) if len(losses) > 0 else 0.001
    pf = gross_profit / gross_loss

    passed = wr > 0.50 and pf > 1.0
    return wr, pf, passed


def run_backtest():
    """Main backtest runner."""
    print("=" * 70)
    print("OVERSOLD HIGH-BETA STOCK BOUNCE BACKTEST")
    print("HC #705 — All adversarial checks inline")
    print("=" * 70)

    # Download data
    closes, opens, highs, lows = download_data()
    spy_regime = get_spy_regime(closes)

    # Generate signals
    signals = generate_signals(closes, opens)

    # Results accumulator
    results = {}

    for variant_name, signal_mask in signals.items():
        print(f"\n{'─' * 60}")
        print(f"VARIANT: {variant_name}")
        print(f"{'─' * 60}")

        n_signals = signal_mask.sum().sum()
        print(f"  Total signal fires: {n_signals}")

        for hold in HOLD_PERIODS:
            key = f"{variant_name}_hold{hold}d"
            print(f"\n  Hold period: {hold} days")

            fwd_ret = compute_forward_returns(opens, hold)
            trades_df = collect_trades(signal_mask, fwd_ret)

            n_trades = len(trades_df)
            print(f"    Trades: {n_trades}")

            if n_trades < MIN_TRADES:
                print(f"    SKIP: < {MIN_TRADES} trades")
                results[key] = {"status": "SKIP", "n_trades": n_trades, "reason": f"< {MIN_TRADES} trades"}
                continue

            mean_ret = trades_df["return"].mean()
            std_ret = trades_df["return"].std()
            sharpe = mean_ret / std_ret if std_ret > 0 else 0.0
            median_ret = trades_df["return"].median()

            print(f"    Mean return: {mean_ret * 100:.3f}%")
            print(f"    Median return: {median_ret * 100:.3f}%")
            print(f"    Sharpe (per-trade): {sharpe:.3f}")

            # ── Gate 1: Permutation test ──
            print(f"    [Gate 1] Permutation test ({N_PERMS} random-date perms)...")
            p_val, obs_mean = permutation_test_random_dates(trades_df, fwd_ret, N_PERMS)
            perm_pass = p_val < PERM_P_THRESHOLD
            print(f"      p-value: {p_val:.4f} {'PASS' if perm_pass else 'FAIL'}")

            # ── Gate 2: R1 regime test ──
            regime_sharpes, regime_gap, regime_pass, _ = regime_test(trades_df, spy_regime)
            if regime_gap is not None:
                print(f"    [Gate 2] Regime test: gap={regime_gap:.3f} (threshold={REGIME_GAP_THRESHOLD}) {'PASS' if regime_pass else 'FAIL'}")
                print(f"      Green Sharpe: {regime_sharpes.get('green', 'N/A')}")
                print(f"      Red Sharpe: {regime_sharpes.get('red', 'N/A')}")
                print(f"      Flat Sharpe: {regime_sharpes.get('flat', 'N/A')}")
            else:
                print(f"    [Gate 2] Regime test: INSUFFICIENT DATA")
                regime_pass = False

            # ── Gate 3: Sub-period consistency ──
            pre_mean, post_mean, period_pass = sub_period_test(trades_df)
            if pre_mean is not None and post_mean is not None:
                print(f"    [Gate 3] Sub-period: pre-2020={pre_mean * 100:.3f}%, post-2020={post_mean * 100:.3f}% {'PASS' if period_pass else 'FAIL'}")
            else:
                print(f"    [Gate 3] Sub-period: INSUFFICIENT DATA")
                period_pass = False

            # ── Gate 4: Outlier removal ──
            full_s, trim_s, drop_pct, outlier_pass = outlier_removal_test(trades_df)
            if full_s is not None:
                print(f"    [Gate 4] Outlier removal: full_sharpe={full_s:.3f}, trimmed={trim_s:.3f}, drop={drop_pct * 100:.1f}% {'PASS' if outlier_pass else 'FAIL'}")
            else:
                print(f"    [Gate 4] Outlier removal: INSUFFICIENT DATA")
                outlier_pass = False

            # ── Gate 5: Ticker concentration ──
            conc, max_conc, conc_pass = ticker_concentration_test(trades_df)
            if max_conc is not None:
                print(f"    [Gate 5] Ticker concentration: max={max_conc:.3f} (threshold={TICKER_CONC_MAX}) {'PASS' if conc_pass else 'FAIL'}")
            else:
                print(f"    [Gate 5] Ticker concentration: INSUFFICIENT DATA")
                conc_pass = False

            # ── Gate 6: Win rate & Profit Factor ──
            wr, pf, wr_pf_pass = win_rate_pf(trades_df)
            print(f"    [Gate 6] WR={wr * 100:.1f}%, PF={pf:.2f} {'PASS' if wr_pf_pass else 'FAIL'}")

            all_pass = all([perm_pass, regime_pass, period_pass, outlier_pass, conc_pass, wr_pf_pass])

            # Count per-ticker trades
            ticker_counts = trades_df.groupby("ticker").size().to_dict()

            results[key] = {
                "status": "ALL_PASS" if all_pass else "FAIL",
                "n_trades": n_trades,
                "mean_return_pct": round(mean_ret * 100, 4),
                "median_return_pct": round(median_ret * 100, 4),
                "std_return_pct": round(std_ret * 100, 4),
                "sharpe_per_trade": round(sharpe, 4),
                "gates": {
                    "permutation": {"p_value": round(p_val, 4), "pass": perm_pass},
                    "regime": {
                        "gap": round(regime_gap, 4) if regime_gap is not None else None,
                        "sharpes": {k: round(v, 4) if v is not None else None for k, v in (regime_sharpes or {}).items()},
                        "pass": regime_pass
                    },
                    "sub_period": {
                        "pre_2020_mean_pct": round(pre_mean * 100, 4) if pre_mean is not None else None,
                        "post_2020_mean_pct": round(post_mean * 100, 4) if post_mean is not None else None,
                        "pass": period_pass
                    },
                    "outlier_removal": {
                        "full_sharpe": round(full_s, 4) if full_s is not None else None,
                        "trimmed_sharpe": round(trim_s, 4) if trim_s is not None else None,
                        "drop_pct": round(drop_pct * 100, 2) if drop_pct is not None else None,
                        "pass": outlier_pass
                    },
                    "ticker_concentration": {
                        "max_concentration": round(max_conc, 4) if max_conc is not None else None,
                        "pass": conc_pass
                    },
                    "win_rate_pf": {
                        "win_rate_pct": round(wr * 100, 2),
                        "profit_factor": round(pf, 4),
                        "pass": wr_pf_pass
                    }
                },
                "ticker_trade_counts": ticker_counts,
                "all_gates_pass": all_pass
            }

            print(f"    >>> {'ALL GATES PASS' if all_pass else 'FAILED'}")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passing = {k: v for k, v in results.items() if v.get("all_gates_pass")}
    failing = {k: v for k, v in results.items() if v.get("status") == "FAIL"}
    skipped = {k: v for k, v in results.items() if v.get("status") == "SKIP"}

    print(f"\nPassing all gates: {len(passing)}")
    for k in sorted(passing.keys()):
        v = passing[k]
        print(f"  {k}: mean={v['mean_return_pct']:.2f}%, WR={v['gates']['win_rate_pf']['win_rate_pct']:.1f}%, PF={v['gates']['win_rate_pf']['profit_factor']:.2f}, n={v['n_trades']}")

    print(f"\nFailing: {len(failing)}")
    for k in sorted(failing.keys()):
        v = failing[k]
        failed_gates = [g for g, gv in v["gates"].items() if not gv.get("pass")]
        print(f"  {k}: mean={v['mean_return_pct']:.2f}%, n={v['n_trades']}, failed=[{', '.join(failed_gates)}]")

    print(f"\nSkipped (too few trades): {len(skipped)}")
    for k in sorted(skipped.keys()):
        print(f"  {k}: n_trades={skipped[k]['n_trades']}")

    # Best variant (highest Sharpe among passing)
    best = None
    if passing:
        best_key = max(passing, key=lambda k: passing[k]["sharpe_per_trade"])
        best = {"variant": best_key, **passing[best_key]}
        print(f"\nBEST PASSING VARIANT: {best_key}")
        print(f"  Sharpe: {best['sharpe_per_trade']:.4f}")
        print(f"  Mean return: {best['mean_return_pct']:.3f}%")
    else:
        print("\nNO VARIANTS PASSED ALL GATES.")
        # Show best failing variant for reference
        non_skip = {k: v for k, v in results.items() if v.get("status") != "SKIP"}
        if non_skip:
            best_fail_key = max(non_skip, key=lambda k: non_skip[k].get("sharpe_per_trade", -999))
            bf = non_skip[best_fail_key]
            print(f"  Best failing: {best_fail_key}, Sharpe={bf.get('sharpe_per_trade', 'N/A')}, mean={bf.get('mean_return_pct', 'N/A')}%")

    # Save report
    report = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "tickers": TICKERS,
            "start": START,
            "end": END,
            "hold_periods": HOLD_PERIODS,
            "n_permutations": N_PERMS,
            "perm_p_threshold": PERM_P_THRESHOLD,
            "regime_gap_threshold": REGIME_GAP_THRESHOLD,
            "outlier_sharpe_drop_max": OUTLIER_SHARPE_DROP_MAX,
            "ticker_conc_max": TICKER_CONC_MAX,
            "min_trades": MIN_TRADES,
        },
        "summary": {
            "total_variants_tested": len(results),
            "passing_all_gates": len(passing),
            "failing": len(failing),
            "skipped": len(skipped),
            "best_variant": best,
        },
        "results": results,
    }

    report_path = OUTPUT_DIR / "backtest_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to {report_path}")

    return report


if __name__ == "__main__":
    report = run_backtest()

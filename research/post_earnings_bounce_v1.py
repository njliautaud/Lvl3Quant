"""
Post-Earnings Oversold Bounce Backtest v1
=========================================
Hypothesis: When a stock drops >=8% in the 2 trading days after earnings,
buying at OPEN on day 3 and holding for 5 trading days captures a mean-reversion bounce.

Variants:
  A: >=8% drop, hold 5d (base)
  B: >=10% drop, hold 5d (stricter)
  C: >=8% drop, hold 10d (longer hold)

Adversarial gates:
  1. Permutation test (200 shuffles, p<0.05)
  2. Regime test (SPY green/red/flat, gap<0.50)
  3. Sub-period stability (pre-2022 vs post-2022)
  4. Outlier removal (trim 5%, Sharpe drop <50%)
  5. Win rate >55% AND profit factor >1.5
"""

import json
import warnings
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────
TICKERS = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "JPM", "V", "UNH",
    "JNJ", "WMT", "PG", "HD", "MA", "BAC", "XOM", "DIS", "NFLX", "AMD",
]
START = "2018-01-01"
END = "2026-07-23"
N_PERMUTATIONS = 200
REGIME_GAP_THRESHOLD = 0.50
OUTLIER_TRIM_PCT = 0.05
SHARPE_DROP_MAX = 0.50
WIN_RATE_MIN = 0.55
PROFIT_FACTOR_MIN = 1.5
OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/post_earnings_bounce_v1")

VARIANTS = {
    "A_drop8_hold5":  {"drop_pct": -0.08, "hold_days": 5},
    "B_drop10_hold5": {"drop_pct": -0.10, "hold_days": 5},
    "C_drop8_hold10": {"drop_pct": -0.08, "hold_days": 10},
}


# ── Data Download ───────────────────────────────────────────────────────
def download_price_data(tickers, start, end):
    """Download OHLCV for all tickers + SPY."""
    all_tickers = list(set(tickers + ["SPY"]))
    print(f"Downloading price data for {len(all_tickers)} tickers...")
    data = yf.download(all_tickers, start=start, end=end, group_by="ticker",
                       auto_adjust=True, progress=False, threads=True)
    return data


def get_earnings_dates(ticker):
    """Get historical earnings dates from yfinance."""
    try:
        t = yf.Ticker(ticker)
        cal = t.get_earnings_dates(limit=100)
        if cal is not None and len(cal) > 0:
            # Filter to past dates only
            dates = cal.index.tz_localize(None) if cal.index.tz else cal.index
            dates = dates[dates < pd.Timestamp.now()]
            return sorted(dates.tolist())
    except Exception as e:
        print(f"  Warning: could not get earnings for {ticker}: {e}")
    return []


# ── Signal Generation ──────────────────────────────────────────────────
def generate_signals(price_data, tickers, variant_params):
    """
    For each ticker, find earnings dates where the stock dropped >= threshold
    in the 2 trading days AFTER earnings. Entry at OPEN of day 3, exit at OPEN
    after hold_days trading days.
    """
    drop_pct = variant_params["drop_pct"]
    hold_days = variant_params["hold_days"]
    trades = []

    for ticker in tickers:
        try:
            # Extract single-ticker OHLC
            if len(tickers) > 1:
                df = price_data[ticker].copy()
            else:
                df = price_data.copy()
            df = df.dropna(subset=["Close", "Open"])
            if len(df) < 20:
                continue
        except (KeyError, TypeError):
            continue

        earnings_dates = get_earnings_dates(ticker)
        if not earnings_dates:
            continue

        for edate in earnings_dates:
            edate = pd.Timestamp(edate)
            # Find next trading day after earnings (day 1)
            mask = df.index > edate
            future_dates = df.index[mask]
            if len(future_dates) < (3 + hold_days):
                continue

            day0_close = df.loc[df.index <= edate, "Close"].iloc[-1] if len(df.loc[df.index <= edate]) > 0 else None
            if day0_close is None or np.isnan(day0_close):
                continue

            day1 = future_dates[0]
            day2 = future_dates[1]
            day3 = future_dates[2]  # entry day

            day2_close = df.loc[day2, "Close"]
            if np.isnan(day2_close):
                continue

            # 2-day return after earnings
            ret_2d = (day2_close - day0_close) / day0_close

            if ret_2d <= drop_pct:  # e.g., <= -0.08
                entry_price = df.loc[day3, "Open"]
                exit_day = future_dates[2 + hold_days]
                exit_price = df.loc[exit_day, "Open"]

                if np.isnan(entry_price) or np.isnan(exit_price) or entry_price <= 0:
                    continue

                trade_ret = (exit_price - entry_price) / entry_price

                trades.append({
                    "ticker": ticker,
                    "earnings_date": edate,
                    "drop_2d": ret_2d,
                    "entry_date": day3,
                    "exit_date": exit_day,
                    "entry_price": float(entry_price),
                    "exit_price": float(exit_price),
                    "return": float(trade_ret),
                })

    return pd.DataFrame(trades)


# ── Metrics ────────────────────────────────────────────────────────────
def compute_metrics(returns):
    """Compute key metrics from a series of trade returns."""
    if len(returns) == 0:
        return {"n_trades": 0, "mean_ret": 0, "win_rate": 0, "profit_factor": 0,
                "sharpe": 0, "sortino": 0, "median_ret": 0}

    wins = returns[returns > 0]
    losses = returns[returns < 0]
    gross_profit = wins.sum() if len(wins) > 0 else 0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 1e-9

    mean_ret = returns.mean()
    std_ret = returns.std()
    downside = returns[returns < 0].std() if len(returns[returns < 0]) > 1 else 1e-9

    return {
        "n_trades": len(returns),
        "mean_ret": float(mean_ret),
        "median_ret": float(returns.median()),
        "win_rate": float((returns > 0).mean()),
        "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0 else float("inf"),
        "sharpe": float(mean_ret / std_ret * np.sqrt(252 / 5)) if std_ret > 0 else 0,
        "sortino": float(mean_ret / downside * np.sqrt(252 / 5)) if downside > 0 else 0,
        "total_return": float(returns.sum()),
        "max_win": float(returns.max()) if len(returns) > 0 else 0,
        "max_loss": float(returns.min()) if len(returns) > 0 else 0,
    }


# ── Adversarial Gates ─────────────────────────────────────────────────
def permutation_test(trades_df, spy_data, n_perms=200):
    """
    Permutation test: shuffle entry dates randomly 200 times.
    Actual mean return must beat >=95% of shuffled means.
    """
    if len(trades_df) < 5:
        return {"pass": False, "p_value": 1.0, "reason": "too few trades"}

    actual_mean = trades_df["return"].mean()

    # Get valid trading dates from SPY
    valid_dates = spy_data.index.tolist()
    hold_days = (trades_df["exit_date"].iloc[0] - trades_df["entry_date"].iloc[0]).days
    # Approximate hold in trading days
    hold_td = max(5, int(hold_days * 5 / 7))

    n_trades = len(trades_df)
    shuffle_means = []

    for _ in range(n_perms):
        # Pick random entry dates from SPY trading calendar
        max_idx = len(valid_dates) - hold_td - 1
        if max_idx < 1:
            continue
        idxs = np.random.randint(0, max_idx, size=n_trades)
        rets = []
        for i in idxs:
            entry_price = spy_data.iloc[i]["Open"]
            exit_price = spy_data.iloc[min(i + hold_td, len(spy_data) - 1)]["Open"]
            if entry_price > 0:
                rets.append((exit_price - entry_price) / entry_price)
        if rets:
            shuffle_means.append(np.mean(rets))

    if not shuffle_means:
        return {"pass": False, "p_value": 1.0, "reason": "permutation failed"}

    p_value = float(np.mean(np.array(shuffle_means) >= actual_mean))
    return {
        "pass": bool(p_value < 0.05),
        "p_value": round(p_value, 4),
        "actual_mean": round(actual_mean, 5),
        "perm_mean": round(np.mean(shuffle_means), 5),
    }


def regime_test(trades_df, spy_data):
    """
    Classify each trade's entry date by SPY regime (green/red/flat day).
    Sharpe gap between best and worst regime must be < 0.50.
    """
    if len(trades_df) < 10:
        return {"pass": False, "reason": "too few trades for regime test"}

    spy_daily_ret = spy_data["Close"].pct_change()

    results_by_regime = {}
    for _, trade in trades_df.iterrows():
        entry = trade["entry_date"]
        # Find closest SPY date
        idx = spy_data.index.get_indexer([entry], method="ffill")[0]
        if idx < 0 or idx >= len(spy_daily_ret):
            continue
        spy_ret = spy_daily_ret.iloc[idx]

        if spy_ret > 0.002:
            regime = "green"
        elif spy_ret < -0.002:
            regime = "red"
        else:
            regime = "flat"

        if regime not in results_by_regime:
            results_by_regime[regime] = []
        results_by_regime[regime].append(trade["return"])

    regime_sharpes = {}
    for regime, rets in results_by_regime.items():
        r = np.array(rets)
        if len(r) > 1 and r.std() > 0:
            regime_sharpes[regime] = float(r.mean() / r.std() * np.sqrt(252 / 5))
        else:
            regime_sharpes[regime] = 0.0

    if len(regime_sharpes) < 2:
        return {"pass": False, "reason": "only 1 regime observed", "regime_sharpes": regime_sharpes}

    sharpe_vals = list(regime_sharpes.values())
    max_s = max(abs(s) for s in sharpe_vals) if sharpe_vals else 1e-9
    gap = (max(sharpe_vals) - min(sharpe_vals)) / max(max_s, 1e-9)

    return {
        "pass": bool(gap < REGIME_GAP_THRESHOLD),
        "gap": round(gap, 4),
        "regime_sharpes": {k: round(v, 3) for k, v in regime_sharpes.items()},
        "regime_counts": {k: len(v) for k, v in results_by_regime.items()},
    }


def sub_period_test(trades_df):
    """Pre-2022 vs post-2022 stability."""
    if len(trades_df) < 10:
        return {"pass": False, "reason": "too few trades"}

    cutoff = pd.Timestamp("2022-01-01")
    pre = trades_df[trades_df["entry_date"] < cutoff]["return"]
    post = trades_df[trades_df["entry_date"] >= cutoff]["return"]

    if len(pre) < 3 or len(post) < 3:
        return {"pass": False, "reason": f"insufficient trades in sub-periods (pre={len(pre)}, post={len(post)})",
                "pre_n": int(len(pre)), "post_n": int(len(post))}

    pre_metrics = compute_metrics(pre)
    post_metrics = compute_metrics(post)

    # Both periods should be profitable
    both_profitable = pre_metrics["mean_ret"] > 0 and post_metrics["mean_ret"] > 0

    return {
        "pass": bool(both_profitable),
        "pre_2022": {k: round(v, 4) if isinstance(v, float) else v for k, v in pre_metrics.items()},
        "post_2022": {k: round(v, 4) if isinstance(v, float) else v for k, v in post_metrics.items()},
    }


def outlier_removal_test(trades_df):
    """Trim top/bottom 5% of returns. Sharpe should not drop >50%."""
    if len(trades_df) < 20:
        return {"pass": False, "reason": "too few trades for outlier test"}

    rets = trades_df["return"]
    full_metrics = compute_metrics(rets)

    lower = rets.quantile(OUTLIER_TRIM_PCT)
    upper = rets.quantile(1 - OUTLIER_TRIM_PCT)
    trimmed = rets[(rets >= lower) & (rets <= upper)]
    trimmed_metrics = compute_metrics(trimmed)

    full_sharpe = full_metrics["sharpe"]
    trim_sharpe = trimmed_metrics["sharpe"]

    if abs(full_sharpe) < 0.01:
        drop = 1.0
    else:
        drop = 1.0 - (trim_sharpe / full_sharpe) if full_sharpe != 0 else 1.0

    return {
        "pass": bool(drop < SHARPE_DROP_MAX),
        "full_sharpe": round(full_sharpe, 3),
        "trimmed_sharpe": round(trim_sharpe, 3),
        "sharpe_drop_pct": round(drop, 3),
        "trades_removed": int(len(rets) - len(trimmed)),
    }


def basic_quality_test(trades_df):
    """Win rate >55% AND profit factor >1.5."""
    if len(trades_df) < 5:
        return {"pass": False, "reason": "too few trades"}

    metrics = compute_metrics(trades_df["return"])
    wr_pass = metrics["win_rate"] > WIN_RATE_MIN
    pf_pass = metrics["profit_factor"] > PROFIT_FACTOR_MIN

    return {
        "pass": bool(wr_pass and pf_pass),
        "win_rate": round(metrics["win_rate"], 4),
        "profit_factor": round(metrics["profit_factor"], 3),
        "wr_pass": bool(wr_pass),
        "pf_pass": bool(pf_pass),
    }


# ── Main ───────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("POST-EARNINGS OVERSOLD BOUNCE BACKTEST v1")
    print("=" * 70)

    # Download data
    price_data = download_price_data(TICKERS, START, END)

    # Extract SPY separately
    spy_data = price_data["SPY"].dropna(subset=["Close", "Open"])

    all_results = {}

    for variant_name, params in VARIANTS.items():
        print(f"\n{'─' * 60}")
        print(f"VARIANT {variant_name}: drop<={params['drop_pct']*100:.0f}%, hold={params['hold_days']}d")
        print(f"{'─' * 60}")

        trades_df = generate_signals(price_data, TICKERS, params)

        if len(trades_df) == 0:
            print(f"  NO TRADES FOUND for this variant.")
            all_results[variant_name] = {"n_trades": 0, "gates": "N/A - no trades"}
            continue

        # Ensure datetime types
        trades_df["entry_date"] = pd.to_datetime(trades_df["entry_date"])
        trades_df["exit_date"] = pd.to_datetime(trades_df["exit_date"])
        trades_df["earnings_date"] = pd.to_datetime(trades_df["earnings_date"])

        print(f"  Found {len(trades_df)} trades across {trades_df['ticker'].nunique()} tickers")
        print(f"  Date range: {trades_df['entry_date'].min().date()} to {trades_df['entry_date'].max().date()}")

        # Compute overall metrics
        metrics = compute_metrics(trades_df["return"])
        print(f"\n  METRICS:")
        print(f"    Trades:        {metrics['n_trades']}")
        print(f"    Mean return:   {metrics['mean_ret']*100:.2f}%")
        print(f"    Median return: {metrics['median_ret']*100:.2f}%")
        print(f"    Win rate:      {metrics['win_rate']*100:.1f}%")
        print(f"    Profit factor: {metrics['profit_factor']:.2f}")
        print(f"    Sharpe:        {metrics['sharpe']:.2f}")
        print(f"    Sortino:       {metrics['sortino']:.2f}")
        print(f"    Total return:  {metrics['total_return']*100:.1f}%")

        # Per-ticker breakdown
        print(f"\n  PER-TICKER BREAKDOWN:")
        for ticker in sorted(trades_df["ticker"].unique()):
            t = trades_df[trades_df["ticker"] == ticker]
            wr = (t["return"] > 0).mean()
            print(f"    {ticker:5s}: {len(t):3d} trades, WR={wr*100:.0f}%, mean={t['return'].mean()*100:.2f}%")

        # Run adversarial gates
        print(f"\n  ADVERSARIAL GATES:")

        g1 = permutation_test(trades_df, spy_data, N_PERMUTATIONS)
        status1 = "PASS" if g1["pass"] else "FAIL"
        print(f"    1. Permutation test:  {status1} (p={g1.get('p_value', 'N/A')})")

        g2 = regime_test(trades_df, spy_data)
        status2 = "PASS" if g2["pass"] else "FAIL"
        print(f"    2. Regime test:       {status2} (gap={g2.get('gap', 'N/A')})")
        if "regime_sharpes" in g2:
            for r, s in g2["regime_sharpes"].items():
                cnt = g2.get("regime_counts", {}).get(r, "?")
                print(f"       {r:5s}: Sharpe={s:.2f} (n={cnt})")

        g3 = sub_period_test(trades_df)
        status3 = "PASS" if g3["pass"] else "FAIL"
        print(f"    3. Sub-period test:   {status3}")
        if "pre_2022" in g3:
            print(f"       Pre-2022:  n={g3['pre_2022']['n_trades']}, WR={g3['pre_2022']['win_rate']*100:.0f}%, mean={g3['pre_2022']['mean_ret']*100:.2f}%")
            print(f"       Post-2022: n={g3['post_2022']['n_trades']}, WR={g3['post_2022']['win_rate']*100:.0f}%, mean={g3['post_2022']['mean_ret']*100:.2f}%")

        g4 = outlier_removal_test(trades_df)
        status4 = "PASS" if g4["pass"] else "FAIL"
        print(f"    4. Outlier removal:   {status4} (Sharpe drop={g4.get('sharpe_drop_pct', 'N/A')})")

        g5 = basic_quality_test(trades_df)
        status5 = "PASS" if g5["pass"] else "FAIL"
        print(f"    5. Quality gate:      {status5} (WR={g5.get('win_rate', 0)*100:.1f}%, PF={g5.get('profit_factor', 0):.2f})")

        gates_passed = sum([g1["pass"], g2["pass"], g3["pass"], g4["pass"], g5["pass"]])
        all_pass = gates_passed == 5

        print(f"\n  VERDICT: {'ALL GATES PASSED' if all_pass else f'{gates_passed}/5 gates passed'}")

        # Store results
        # Convert trades to serializable format
        trades_list = []
        for _, row in trades_df.iterrows():
            trades_list.append({
                "ticker": row["ticker"],
                "earnings_date": str(row["earnings_date"].date()),
                "entry_date": str(row["entry_date"].date()),
                "exit_date": str(row["exit_date"].date()),
                "drop_2d": round(float(row["drop_2d"]), 4),
                "entry_price": round(float(row["entry_price"]), 2),
                "exit_price": round(float(row["exit_price"]), 2),
                "return": round(float(row["return"]), 5),
            })

        all_results[variant_name] = {
            "params": params,
            "n_trades": metrics["n_trades"],
            "metrics": {k: round(v, 5) if isinstance(v, float) else v for k, v in metrics.items()},
            "gates": {
                "permutation_test": g1,
                "regime_test": g2,
                "sub_period_test": g3,
                "outlier_removal": g4,
                "basic_quality": g5,
            },
            "gates_passed": gates_passed,
            "all_gates_passed": all_pass,
            "trades": trades_list,
        }

    # ── Final Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"\n{'Variant':<20} {'Trades':>6} {'WR':>6} {'PF':>6} {'Sharpe':>7} {'Sortino':>8} {'Gates':>6} {'Verdict':>10}")
    print("-" * 75)

    for vname, vres in all_results.items():
        if vres.get("n_trades", 0) == 0:
            print(f"{vname:<20} {'0':>6} {'N/A':>6} {'N/A':>6} {'N/A':>7} {'N/A':>8} {'N/A':>6} {'NO DATA':>10}")
            continue
        m = vres["metrics"]
        gp = vres["gates_passed"]
        verdict = "PASS" if vres["all_gates_passed"] else "FAIL"
        print(f"{vname:<20} {m['n_trades']:>6} {m['win_rate']*100:>5.1f}% {m['profit_factor']:>6.2f} {m['sharpe']:>7.2f} {m['sortino']:>8.2f} {gp:>4}/5  {verdict:>10}")

    print("\n" + "=" * 70)
    print("GATE DETAIL BY VARIANT")
    print("=" * 70)
    gate_names = ["permutation_test", "regime_test", "sub_period_test", "outlier_removal", "basic_quality"]
    gate_labels = ["Permutation", "Regime", "Sub-period", "Outlier", "Quality"]

    for vname, vres in all_results.items():
        if isinstance(vres.get("gates"), str):
            continue
        print(f"\n  {vname}:")
        for gname, glabel in zip(gate_names, gate_labels):
            g = vres["gates"][gname]
            status = "PASS" if g["pass"] else "FAIL"
            print(f"    {glabel:<15} {status}")

    # Save results
    output_file = OUTPUT_DIR / "results.json"
    # Make JSON serializable (remove trades for cleaner output, keep summary)
    save_results = {}
    for vname, vres in all_results.items():
        save_copy = dict(vres)
        # Keep trades but ensure serializable
        save_results[vname] = save_copy

    with open(output_file, "w") as f:
        json.dump(save_results, f, indent=2, default=str)

    print(f"\nResults saved to {output_file}")
    print("Done.")


if __name__ == "__main__":
    main()

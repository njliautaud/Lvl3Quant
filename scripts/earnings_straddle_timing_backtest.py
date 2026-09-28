#!/usr/bin/env python3
"""
Earnings Straddle Direction Prediction Backtest
Predicts post-earnings move DIRECTION using pre-earnings signals.
Uses stock-level proxies: pre-earnings drift, volume ratio, relative strength vs SPY.
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────
UNIVERSE = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
            "JPM", "BAC", "JNJ", "UNH", "HD", "MCD", "DIS", "NFLX"]

OOT_START = "2022-01-01"
OOT_END = "2026-07-29"
INITIAL_CAPITAL = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02%
EARNINGS_GAP_THRESHOLD = 0.03  # 3% absolute return = earnings day
N_PERMUTATIONS = 1000
RESULTS_PATH = Path("/home/jupiter/Lvl3Quant/data/earnings_straddle_timing_results.json")

# ── Data Download ───────────────────────────────────────────────────────
print("Downloading price data...")
tickers = UNIVERSE + ["SPY"]
data = {}
for t in tickers:
    try:
        df = yf.download(t, start="2020-01-01", end=OOT_END, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if len(df) > 100:
            data[t] = df
            print(f"  {t}: {len(df)} days")
        else:
            print(f"  {t}: insufficient data ({len(df)} days), skipping")
    except Exception as e:
        print(f"  {t}: download failed ({e})")

spy = data.get("SPY")
if spy is None:
    raise RuntimeError("SPY data required")

# ── Identify Earnings Dates ────────────────────────────────────────────
print("\nIdentifying earnings dates (single-day gaps > 3%)...")
earnings_dates = {}
for ticker in UNIVERSE:
    if ticker not in data:
        continue
    df = data[ticker]
    daily_ret = df["Close"].pct_change()
    # Earnings = days with absolute return > 3%
    big_moves = daily_ret[abs(daily_ret) > EARNINGS_GAP_THRESHOLD].index
    # Cluster moves within 5 days (keep first of each cluster)
    clustered = []
    last = None
    for d in sorted(big_moves):
        if last is None or (d - last).days > 45:
            clustered.append(d)
            last = d
    # Filter to OOT period
    oot_dates = [d for d in clustered if d >= pd.Timestamp(OOT_START)]
    earnings_dates[ticker] = oot_dates
    print(f"  {ticker}: {len(oot_dates)} earnings events in OOT")

total_events = sum(len(v) for v in earnings_dates.values())
print(f"\nTotal earnings events: {total_events}")


# ── Signal Computation ──────────────────────────────────────────────────
def compute_signals(ticker, earn_date, stock_df, spy_df):
    """Compute pre-earnings signals for a given earnings date."""
    try:
        idx = stock_df.index.get_loc(earn_date)
    except KeyError:
        return None

    if idx < 25:
        return None

    close = stock_df["Close"].values
    volume = stock_df["Volume"].values
    dates = stock_df.index

    # Pre-earnings drift: 5-day return before earnings
    if idx >= 5:
        pre_drift_5d = (close[idx - 1] / close[idx - 6]) - 1.0
    else:
        return None

    # 10-day pre-earnings return
    if idx >= 10:
        pre_drift_10d = (close[idx - 1] / close[idx - 11]) - 1.0
    else:
        pre_drift_10d = 0.0

    # Volume ratio: avg volume in 5 days before earnings / 20-day avg
    if idx >= 25:
        vol_5d = np.mean(volume[idx - 5:idx])
        vol_20d = np.mean(volume[idx - 25:idx - 5])
        vol_ratio = vol_5d / max(vol_20d, 1)
    else:
        vol_ratio = 1.0

    # Relative strength vs SPY (20-day)
    try:
        spy_idx = spy_df.index.get_loc(earn_date)
    except KeyError:
        return None

    if spy_idx >= 20:
        spy_close = spy_df["Close"].values
        stock_20d_ret = (close[idx - 1] / close[idx - 21]) - 1.0
        spy_20d_ret = (spy_close[spy_idx - 1] / spy_close[spy_idx - 21]) - 1.0
        rel_strength = stock_20d_ret - spy_20d_ret
    else:
        rel_strength = 0.0

    # Post-earnings returns
    post_ret_5d = None
    post_ret_10d = None
    if idx + 5 < len(close):
        post_ret_5d = (close[idx + 5] / close[idx]) - 1.0
    if idx + 10 < len(close):
        post_ret_10d = (close[idx + 10] / close[idx]) - 1.0

    # Earnings gap direction (for streak calculation)
    earn_gap = (close[idx] / close[idx - 1]) - 1.0

    # Last 3 earnings gaps (for streak variant)
    prev_gaps = []
    all_rets = stock_df["Close"].pct_change()
    big_moves_before = all_rets[:earn_date][abs(all_rets[:earn_date]) > EARNINGS_GAP_THRESHOLD]
    candidates = sorted(big_moves_before.index, reverse=True)
    last_gap_date = earn_date
    for c in candidates:
        if (last_gap_date - c).days > 45:
            prev_gaps.append(all_rets.loc[c])
            last_gap_date = c
        if len(prev_gaps) >= 3:
            break

    streak_bullish = len(prev_gaps) >= 3 and all(g > 0 for g in prev_gaps)

    return {
        "ticker": ticker,
        "earn_date": earn_date,
        "pre_drift_5d": pre_drift_5d,
        "pre_drift_10d": pre_drift_10d,
        "vol_ratio": vol_ratio,
        "rel_strength": rel_strength,
        "post_ret_5d": post_ret_5d,
        "post_ret_10d": post_ret_10d,
        "earn_gap": earn_gap,
        "streak_bullish": streak_bullish,
        "close_price": close[idx],
    }


print("\nComputing signals...")
all_signals = []
for ticker in UNIVERSE:
    if ticker not in data or ticker not in earnings_dates:
        continue
    for ed in earnings_dates[ticker]:
        sig = compute_signals(ticker, ed, data[ticker], spy)
        if sig is not None:
            all_signals.append(sig)

print(f"Total signals computed: {len(all_signals)}")
signals_df = pd.DataFrame(all_signals)


# ── Strategy Variants ───────────────────────────────────────────────────
def run_variant(signals_df, variant_name):
    """Run a specific variant and return trade-level results."""
    trades = []

    for _, row in signals_df.iterrows():
        direction = 0  # 0 = cash, 1 = long, -1 = short

        if variant_name == "A_pre_drift":
            if row["pre_drift_5d"] > 0.02:
                direction = 1
            elif row["pre_drift_5d"] < -0.02:
                direction = -1
            hold_days = 5
            post_ret = row["post_ret_5d"]

        elif variant_name == "B_rel_strength":
            if row["rel_strength"] > 0.02:
                direction = 1
            hold_days = 5
            post_ret = row["post_ret_5d"]

        elif variant_name == "C_volume_momentum":
            if row["vol_ratio"] > 1.5:
                direction = 1
            hold_days = 5
            post_ret = row["post_ret_5d"]

        elif variant_name == "D_combined":
            a_sig = 1 if row["pre_drift_5d"] > 0.02 else (-1 if row["pre_drift_5d"] < -0.02 else 0)
            b_sig = 1 if row["rel_strength"] > 0.02 else 0
            if a_sig == 1 and b_sig == 1:
                direction = 1
            elif a_sig == -1 and b_sig == 0:
                direction = -1
            hold_days = 5
            post_ret = row["post_ret_5d"]

        elif variant_name == "E_contrarian":
            if row["pre_drift_10d"] < -0.03:
                direction = 1
            hold_days = 10
            post_ret = row["post_ret_10d"]

        elif variant_name == "F_streak":
            if row["streak_bullish"]:
                direction = 1
            hold_days = 5
            post_ret = row["post_ret_5d"]

        if direction == 0 or post_ret is None or np.isnan(post_ret):
            continue

        # Apply slippage (entry + exit)
        net_ret = direction * post_ret - 2 * SLIPPAGE_PCT

        trades.append({
            "ticker": row["ticker"],
            "date": row["earn_date"],
            "direction": direction,
            "gross_ret": direction * post_ret,
            "net_ret": net_ret,
            "hold_days": hold_days,
        })

    return trades


# ── Backtest Engine ─────────────────────────────────────────────────────
def backtest_trades(trades, initial_capital):
    """Convert trade returns to equity curve."""
    if not trades:
        return {"equity": [initial_capital], "returns": [], "dates": []}

    trades_sorted = sorted(trades, key=lambda x: x["date"])
    equity = initial_capital
    equity_curve = [initial_capital]
    daily_returns = []
    dates = []

    for t in trades_sorted:
        ret = t["net_ret"]
        pnl = equity * ret
        equity += pnl
        equity_curve.append(equity)
        daily_returns.append(ret)
        dates.append(t["date"])

    return {
        "equity": equity_curve,
        "returns": daily_returns,
        "dates": dates,
    }


def compute_metrics(bt_result, trades):
    """Compute performance metrics."""
    returns = np.array(bt_result["returns"])
    equity = np.array(bt_result["equity"])

    if len(returns) == 0:
        return None

    n_trades = len(returns)
    win_rate = np.mean(returns > 0)
    total_ret = (equity[-1] / equity[0]) - 1.0
    avg_ret = np.mean(returns)

    if len(bt_result["dates"]) >= 2:
        first_date = bt_result["dates"][0]
        last_date = bt_result["dates"][-1]
        years = max((last_date - first_date).days / 365.25, 0.5)
    else:
        years = 1.0

    trades_per_year = n_trades / years if years > 0 else n_trades

    # Sharpe: annualized
    if np.std(returns) > 0:
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(trades_per_year)
    else:
        sharpe = 0.0

    # Sortino
    downside = returns[returns < 0]
    if len(downside) > 0 and np.std(downside) > 0:
        sortino = (np.mean(returns) / np.std(downside)) * np.sqrt(trades_per_year)
    else:
        sortino = sharpe

    # Max drawdown
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = np.min(dd)

    # Profit factor
    gross_profit = np.sum(returns[returns > 0])
    gross_loss = abs(np.sum(returns[returns < 0]))
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Long/short split
    long_trades = [t for t in trades if t["direction"] == 1]
    short_trades = [t for t in trades if t["direction"] == -1]

    long_wr = np.mean([t["net_ret"] > 0 for t in long_trades]) if long_trades else 0
    short_wr = np.mean([t["net_ret"] > 0 for t in short_trades]) if short_trades else 0

    # Regime gap (long vs short direction)
    if long_trades and short_trades:
        long_sharpe = np.mean([t["net_ret"] for t in long_trades]) / max(np.std([t["net_ret"] for t in long_trades]), 1e-8)
        short_sharpe = np.mean([t["net_ret"] for t in short_trades]) / max(np.std([t["net_ret"] for t in short_trades]), 1e-8)
        regime_gap = abs(long_sharpe - short_sharpe) / max(abs(long_sharpe), abs(short_sharpe), 1e-8)
    else:
        regime_gap = 0.0

    return {
        "n_trades": n_trades,
        "total_return_pct": round(total_ret * 100, 2),
        "final_equity": round(equity[-1], 2),
        "win_rate": round(win_rate, 4),
        "avg_return_pct": round(avg_ret * 100, 4),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "profit_factor": round(pf, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "n_long": len(long_trades),
        "n_short": len(short_trades),
        "long_wr": round(long_wr, 4),
        "short_wr": round(short_wr, 4),
        "regime_gap": round(regime_gap, 4),
        "years": round(years, 2),
    }


# ── Permutation Test ────────────────────────────────────────────────────
def permutation_test(returns, n_perms=N_PERMUTATIONS):
    """Shuffle trade returns to test if Sharpe is statistically significant."""
    if len(returns) < 5:
        return 1.0

    observed_sharpe = np.mean(returns) / max(np.std(returns), 1e-8)
    count_better = 0
    rng = np.random.default_rng(42)

    for _ in range(n_perms):
        shuffled = returns.copy()
        signs = rng.choice([-1, 1], size=len(shuffled))
        shuffled = shuffled * signs
        perm_sharpe = np.mean(shuffled) / max(np.std(shuffled), 1e-8)
        if perm_sharpe >= observed_sharpe:
            count_better += 1

    return count_better / n_perms


# ── SPY Regime Split for Regime Gap ─────────────────────────────────────
def regime_gap_spy(trades, spy_df):
    """Split trades by whether SPY was in uptrend or downtrend (20d SMA)."""
    if len(trades) < 10:
        return 0.0

    spy_close = spy_df["Close"]
    spy_sma20 = spy_close.rolling(20).mean()

    bull_rets = []
    bear_rets = []

    for t in trades:
        d = t["date"]
        try:
            if d in spy_sma20.index:
                if spy_close.loc[d] > spy_sma20.loc[d]:
                    bull_rets.append(t["net_ret"])
                else:
                    bear_rets.append(t["net_ret"])
        except:
            pass

    if not bull_rets or not bear_rets:
        return 0.0

    bull_sharpe = np.mean(bull_rets) / max(np.std(bull_rets), 1e-8)
    bear_sharpe = np.mean(bear_rets) / max(np.std(bear_rets), 1e-8)

    gap = abs(bull_sharpe - bear_sharpe) / max(abs(bull_sharpe), abs(bear_sharpe), 1e-8)
    return round(gap, 4)


# ── Run All Variants ────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("EARNINGS STRADDLE DIRECTION PREDICTION — BACKTEST RESULTS")
print(f"OOT Period: {OOT_START} to {OOT_END} | Initial Capital: ${INITIAL_CAPITAL}")
print("=" * 80)

variants = ["A_pre_drift", "B_rel_strength", "C_volume_momentum",
            "D_combined", "E_contrarian", "F_streak"]

results = {}
for variant in variants:
    print(f"\n{'─' * 60}")
    print(f"VARIANT {variant}")
    print(f"{'─' * 60}")

    trades = run_variant(signals_df, variant)
    bt = backtest_trades(trades, INITIAL_CAPITAL)
    metrics = compute_metrics(bt, trades)

    if metrics is None:
        print("  No trades generated.")
        results[variant] = {"status": "NO_TRADES", "gates": {}}
        continue

    # Permutation test
    returns = np.array(bt["returns"])
    perm_p = permutation_test(returns)

    # SPY regime gap
    rgap = regime_gap_spy(trades, spy)

    # Use the worse of the two regime gaps
    final_regime_gap = max(metrics["regime_gap"], rgap)

    # 5-Gate Validation
    gates = {
        "G1_sharpe_gt_0.5": metrics["sharpe"] > 0.5,
        "G2_perm_p_lt_0.05": perm_p < 0.05,
        "G3_regime_gap_lt_0.5": final_regime_gap < 0.5,
        "G4_maxdd_gt_neg50": metrics["max_drawdown_pct"] > -50,
        "G5_min_20_trades": metrics["n_trades"] >= 20,
    }
    gates_passed = sum(gates.values())
    all_pass = all(gates.values())

    print(f"  Trades: {metrics['n_trades']} ({metrics['n_long']}L / {metrics['n_short']}S)")
    print(f"  Total Return: {metrics['total_return_pct']:.1f}% | Final Equity: ${metrics['final_equity']:.2f}")
    print(f"  Win Rate: {metrics['win_rate']:.1%} | Avg Return: {metrics['avg_return_pct']:.3f}%")
    print(f"  Sharpe: {metrics['sharpe']:.3f} | Sortino: {metrics['sortino']:.3f} | PF: {metrics['profit_factor']:.3f}")
    print(f"  Max DD: {metrics['max_drawdown_pct']:.1f}%")
    print(f"  Regime Gap (dir): {metrics['regime_gap']:.4f} | Regime Gap (SPY): {rgap:.4f}")
    print(f"  Permutation p-value: {perm_p:.4f}")
    print(f"\n  GATES: {gates_passed}/5 {'PASS' if all_pass else 'FAIL'}")
    for g, v in gates.items():
        status = "PASS" if v else "FAIL"
        print(f"    [{status}] {g}: {v}")

    results[variant] = {
        "metrics": metrics,
        "perm_p_value": round(perm_p, 4),
        "regime_gap_spy": rgap,
        "regime_gap_final": final_regime_gap,
        "gates": gates,
        "gates_passed": gates_passed,
        "all_gates_pass": all_pass,
    }

# ── Summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"{'Variant':<25} {'Trades':>6} {'Return%':>8} {'Sharpe':>7} {'WR':>6} {'Perm-p':>7} {'Gates':>6} {'Pass?':>6}")
print("-" * 80)

passing = []
for v in variants:
    r = results[v]
    if "metrics" not in r:
        print(f"{v:<25} {'--':>6} {'--':>8} {'--':>7} {'--':>6} {'--':>7} {'--':>6} {'--':>6}")
        continue
    m = r["metrics"]
    p = "YES" if r["all_gates_pass"] else "NO"
    print(f"{v:<25} {m['n_trades']:>6} {m['total_return_pct']:>7.1f}% {m['sharpe']:>7.3f} {m['win_rate']:>5.1%} {r['perm_p_value']:>7.4f} {r['gates_passed']:>4}/5 {p:>5}")
    if r["all_gates_pass"]:
        passing.append(v)

print(f"\nPassing variants: {passing if passing else 'NONE'}")

# ── Save Results ────────────────────────────────────────────────────────
output = {
    "strategy": "Earnings Straddle Direction Prediction",
    "description": "Predicts post-earnings move direction using pre-earnings signals (drift, volume, relative strength)",
    "oot_period": f"{OOT_START} to {OOT_END}",
    "initial_capital": INITIAL_CAPITAL,
    "slippage_pct": SLIPPAGE_PCT,
    "universe": UNIVERSE,
    "total_earnings_events": total_events,
    "total_signals": len(all_signals),
    "variants": {},
    "passing_variants": passing,
    "run_timestamp": datetime.now().isoformat(),
}

for v in variants:
    r = results[v]
    variant_data = {
        "gates": {k: bool(val) for k, val in r.get("gates", {}).items()},
        "gates_passed": r.get("gates_passed", 0),
        "all_gates_pass": r.get("all_gates_pass", False),
    }
    if "metrics" in r:
        variant_data["metrics"] = r["metrics"]
        variant_data["perm_p_value"] = r["perm_p_value"]
        variant_data["regime_gap_spy"] = r["regime_gap_spy"]
    output["variants"][v] = variant_data

def default_serializer(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    raise TypeError(f"Not serializable: {type(obj)}")

RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(RESULTS_PATH, "w") as f:
    json.dump(output, f, indent=2, default=default_serializer)

print(f"\nResults saved to {RESULTS_PATH}")
print("Done.")

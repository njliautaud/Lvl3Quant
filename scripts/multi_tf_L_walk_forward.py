#!/usr/bin/env python3
"""
Walk-Forward Validation: Multi-Timeframe MR Variant L
======================================================
Train on 2-year windows, test on 6-month windows, roll forward.
Tests whether the strategy is genuinely forward-looking vs overfit.

Strategy (fixed params):
  1. Daily: Stock drops >5% from 20-day high
  2. Daily: RSI(14) < 35
  3. Daily: First green day after 3+ consecutive red days
  4. Weekly: Weekly RSI(14) declining for 2+ consecutive weeks
  Hold: 10 trading days

Walk-Forward Setup:
  - Training window: 2 years (504 trading days)
  - Test window: 6 months (126 trading days)
  - Roll: every 6 months (~13 folds from 2018-2026)
  - Capital: $645, max $200/trade, max 3 concurrent
  - Slippage: 0.02% each way

Also tests:
  - Adaptive A: Skip test window if training Sharpe < 0.5
  - Adaptive B: Scale position size by training Sharpe
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from pathlib import Path
from scipy import stats

warnings.filterwarnings("ignore")
np.random.seed(42)

# -- Configuration -----------------------------------------------------------
CAPITAL = 645.0
MAX_PER_TRADE = 200.0
MAX_CONCURRENT = 3
SLIPPAGE_PCT = 0.0002  # 0.02% each way

TRAIN_DAYS = 504   # ~2 years
TEST_DAYS = 126    # ~6 months
ROLL_DAYS = 126    # roll every 6 months

START = "2017-01-01"  # extra buffer for indicators
END = "2026-07-31"

# Strategy params (fixed)
DIP_THRESH = -0.05
RSI_THRESH = 35.0
MIN_RED_DAYS = 3
WEEKLY_DECLINE_WEEKS = 2
HOLD_DAYS = 10

TICKERS = [
    "AAPL", "MSFT", "AVGO", "JPM", "JNJ", "PG", "KO", "PEP",
    "HD", "COST", "UNH", "LLY", "V", "MA", "ABBV", "MRK",
    "WMT", "AMZN", "GOOGL", "META",
]

ALL_TICKERS = sorted(set(TICKERS + ["SPY"]))

# -- Data Download -----------------------------------------------------------
print("Downloading price data ...")
raw = yf.download(ALL_TICKERS, start=START, end=END,
                  group_by="ticker", auto_adjust=True, progress=False)


def get_close(ticker):
    try:
        if len(ALL_TICKERS) == 1:
            s = raw["Close"].dropna()
        else:
            s = raw[ticker]["Close"].dropna()
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        return s
    except Exception:
        return pd.Series(dtype=float)


closes = {t: get_close(t) for t in ALL_TICKERS}
spy_close = closes.get("SPY", pd.Series(dtype=float))

loaded = sum(1 for t in TICKERS if len(closes.get(t, [])) > 252)
print(f"  Tickers with sufficient data: {loaded}/{len(TICKERS)}")


# -- Indicator Helpers -------------------------------------------------------
def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


# -- Pre-compute daily indicators -------------------------------------------
print("Computing indicators ...")
indicators = {}
for t in TICKERS:
    c = closes.get(t, pd.Series(dtype=float))
    if len(c) < 300:
        continue
    ind = pd.DataFrame(index=c.index)
    ind["close"] = c
    ind["high20"] = c.rolling(20).max()
    ind["dip_from_high"] = (c - ind["high20"]) / ind["high20"]
    ind["rsi14"] = calc_rsi(c, 14)
    ind["daily_ret"] = c.pct_change()
    # Consecutive red days count
    red = (ind["daily_ret"] < 0).astype(int)
    consec_red = pd.Series(0, index=ind.index)
    for i in range(1, len(ind)):
        if red.iloc[i-1] == 1:
            consec_red.iloc[i] = consec_red.iloc[i-1] + 1
        else:
            consec_red.iloc[i] = 0
    ind["consec_red_before"] = consec_red
    ind["is_green"] = ind["daily_ret"] > 0
    indicators[t] = ind.dropna(subset=["high20", "rsi14"])

# -- Pre-compute weekly RSI --------------------------------------------------
weekly_rsi_data = {}
for t in TICKERS:
    c = closes.get(t, pd.Series(dtype=float))
    if len(c) < 300:
        continue
    weekly_close = c.resample("W-FRI").last().dropna()
    w_rsi = calc_rsi(weekly_close, 14)
    w_declining = w_rsi.diff() < 0
    consec_decline = pd.Series(0, index=weekly_close.index, dtype=int)
    for i in range(1, len(consec_decline)):
        if w_declining.iloc[i]:
            consec_decline.iloc[i] = consec_decline.iloc[i-1] + 1
        else:
            consec_decline.iloc[i] = 0
    weekly_rsi_data[t] = pd.DataFrame({
        "weekly_rsi": w_rsi,
        "consec_decline_weeks": consec_decline,
    }, index=weekly_close.index)

print(f"  Tickers with indicators: {len(indicators)}")
print(f"  Tickers with weekly RSI: {len(weekly_rsi_data)}")


# -- Weekly RSI lookup -------------------------------------------------------
def get_weekly_decline_weeks(ticker, date):
    wdata = weekly_rsi_data.get(ticker)
    if wdata is None:
        return 0
    valid = wdata.index[wdata.index <= date]
    if len(valid) < 2:
        return 0
    latest_week = valid[-1]
    return int(wdata.loc[latest_week, "consec_decline_weeks"])


# -- Signal generation for a date range -------------------------------------
def gen_signals_in_range(start_date, end_date, tickers=None):
    """Generate all Variant L signals within [start_date, end_date]."""
    if tickers is None:
        tickers = TICKERS
    signals = []
    for t in tickers:
        ind = indicators.get(t)
        if ind is None:
            continue
        mask = (ind.index >= pd.Timestamp(start_date)) & (ind.index <= pd.Timestamp(end_date))
        sub = ind[mask]
        for i in range(len(sub)):
            idx_in_full = ind.index.get_loc(sub.index[i])
            if idx_in_full < 1:
                continue
            dt = sub.index[i]
            # Condition 1: dip > 5% from 20d high
            dip = ind["dip_from_high"].iloc[idx_in_full]
            if dip > DIP_THRESH:
                continue
            # Condition 2: RSI < 35
            if ind["rsi14"].iloc[idx_in_full] >= RSI_THRESH:
                continue
            # Condition 3: first green day after 3+ red days
            if not ind["is_green"].iloc[idx_in_full]:
                continue
            if ind["consec_red_before"].iloc[idx_in_full] < MIN_RED_DAYS:
                continue
            # Condition 4: weekly RSI declining 2+ weeks
            weeks = get_weekly_decline_weeks(t, dt)
            if weeks < WEEKLY_DECLINE_WEEKS:
                continue
            signals.append({
                "ticker": t,
                "entry_date": dt,
                "entry_price": float(ind["close"].iloc[idx_in_full]),
            })
    return signals


# -- Backtest engine ---------------------------------------------------------
def backtest_signals(signals, capital=CAPITAL, max_per_trade=MAX_PER_TRADE,
                     max_concurrent=MAX_CONCURRENT, hold_days=HOLD_DAYS,
                     slippage_pct=SLIPPAGE_PCT, position_size_override=None):
    """
    Run portfolio backtest on given signals.
    Returns dict with Sharpe, WR, PF, n_trades, total_return, daily_returns.
    """
    if not signals:
        return {
            "sharpe": 0.0, "wr": 0.0, "pf": 0.0,
            "n_trades": 0, "total_return": 0.0,
            "daily_returns": [],
        }

    trades = []
    active = []  # list of (ticker, entry_date, entry_price, shares, exit_date_target)

    # Sort signals by date
    signals_sorted = sorted(signals, key=lambda x: x["entry_date"])

    for sig in signals_sorted:
        t = sig["ticker"]
        entry_dt = sig["entry_date"]

        # Check concurrent limit
        # Remove expired active trades
        active = [a for a in active if a[4] > entry_dt]
        if len(active) >= max_concurrent:
            continue

        entry_price = sig["entry_price"]
        entry_cost = entry_price * (1 + slippage_pct)

        per_trade = position_size_override if position_size_override else max_per_trade
        shares = int(per_trade / entry_cost)
        if shares < 1:
            continue

        # Find exit date (hold_days later)
        c = closes.get(t, pd.Series(dtype=float))
        if len(c) == 0:
            continue
        entry_loc = c.index.get_loc(entry_dt) if entry_dt in c.index else None
        if entry_loc is None:
            continue
        exit_loc = min(entry_loc + hold_days, len(c) - 1)
        exit_dt = c.index[exit_loc]
        exit_price = float(c.iloc[exit_loc]) * (1 - slippage_pct)

        pnl = (exit_price - entry_cost) * shares
        ret = (exit_price / entry_cost) - 1.0

        trades.append({
            "ticker": t,
            "entry_date": str(entry_dt.date()),
            "exit_date": str(exit_dt.date()),
            "entry_price": round(entry_cost, 4),
            "exit_price": round(exit_price, 4),
            "shares": shares,
            "pnl": round(pnl, 2),
            "return": round(ret, 6),
        })
        active.append((t, entry_dt, entry_cost, shares, exit_dt))

    if not trades:
        return {
            "sharpe": 0.0, "wr": 0.0, "pf": 0.0,
            "n_trades": 0, "total_return": 0.0,
            "daily_returns": [],
        }

    # Compute metrics
    returns = [tr["return"] for tr in trades]
    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r <= 0]

    wr = len(wins) / len(returns) if returns else 0.0
    gross_profit = sum(wins) if wins else 0.0
    gross_loss = abs(sum(losses)) if losses else 0.0001
    pf = gross_profit / gross_loss if gross_loss > 0 else 99.9

    # Annualized Sharpe from trade returns
    avg_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1) if len(returns) > 1 else 0.0001
    # Approximate annualization: ~25 trades/year assumption or scale by sqrt(n)
    sharpe = (avg_ret / std_ret) * np.sqrt(min(len(returns), 252)) if std_ret > 0 else 0.0

    total_pnl = sum(tr["pnl"] for tr in trades)
    total_return = total_pnl / capital

    return {
        "sharpe": round(sharpe, 4),
        "wr": round(wr, 4),
        "pf": round(pf, 4),
        "n_trades": len(trades),
        "total_return": round(total_return, 6),
        "avg_return": round(avg_ret, 6),
        "trades": trades,
    }


# -- Build common date index ------------------------------------------------
# Use SPY's trading dates as the reference calendar
all_dates = spy_close.index.sort_values()
# Filter to Jan 2018 onwards for walk-forward
wf_start = pd.Timestamp("2018-01-01")
all_dates = all_dates[all_dates >= wf_start]
print(f"  Trading days from {all_dates[0].date()} to {all_dates[-1].date()}: {len(all_dates)}")


# -- Walk-Forward Folds -----------------------------------------------------
print("\n" + "="*70)
print("WALK-FORWARD VALIDATION")
print("="*70)

folds = []
fold_num = 0
i = 0

while i + TRAIN_DAYS + TEST_DAYS <= len(all_dates):
    train_start = all_dates[i]
    train_end = all_dates[i + TRAIN_DAYS - 1]
    test_start = all_dates[i + TRAIN_DAYS]
    test_end_idx = min(i + TRAIN_DAYS + TEST_DAYS - 1, len(all_dates) - 1)
    test_end = all_dates[test_end_idx]

    folds.append({
        "fold": fold_num,
        "train_start": train_start,
        "train_end": train_end,
        "test_start": test_start,
        "test_end": test_end,
    })
    fold_num += 1
    i += ROLL_DAYS

print(f"Total folds: {len(folds)}\n")

# -- Run Walk-Forward -------------------------------------------------------
results = []

for fold in folds:
    fn = fold["fold"]
    ts = fold["train_start"]
    te = fold["train_end"]
    xs = fold["test_start"]
    xe = fold["test_end"]

    print(f"--- Fold {fn} ---")
    print(f"  Train: {ts.date()} to {te.date()}")
    print(f"  Test:  {xs.date()} to {xe.date()}")

    # Training period
    train_signals = gen_signals_in_range(ts, te)
    train_result = backtest_signals(train_signals)
    print(f"  Train: Sharpe={train_result['sharpe']:.3f}, WR={train_result['wr']:.1%}, "
          f"PF={train_result['pf']:.2f}, n={train_result['n_trades']}")

    # Test period (fixed params)
    test_signals = gen_signals_in_range(xs, xe)
    test_result = backtest_signals(test_signals)
    print(f"  Test:  Sharpe={test_result['sharpe']:.3f}, WR={test_result['wr']:.1%}, "
          f"PF={test_result['pf']:.2f}, n={test_result['n_trades']}, "
          f"ret={test_result['total_return']:.2%}")

    # Adaptive A: skip if training Sharpe < 0.5
    adaptive_a_trades = train_result["sharpe"] >= 0.5
    adaptive_a_result = test_result if adaptive_a_trades else {
        "sharpe": 0.0, "wr": 0.0, "pf": 0.0,
        "n_trades": 0, "total_return": 0.0, "avg_return": 0.0,
    }
    print(f"  Adaptive A: {'TRADE' if adaptive_a_trades else 'SIT OUT'} "
          f"(train Sharpe {'>=':s} 0.5? {train_result['sharpe']:.3f})")

    # Adaptive B: scale position size by training Sharpe
    adaptive_b_scale = min(max(train_result["sharpe"], 0.0), 2.0)
    adaptive_b_size = MAX_PER_TRADE * adaptive_b_scale
    adaptive_b_result = backtest_signals(test_signals, position_size_override=adaptive_b_size) if adaptive_b_size >= 1 else {
        "sharpe": 0.0, "wr": 0.0, "pf": 0.0,
        "n_trades": 0, "total_return": 0.0, "avg_return": 0.0,
    }
    print(f"  Adaptive B: size=${adaptive_b_size:.0f} (scale={adaptive_b_scale:.2f}), "
          f"ret={adaptive_b_result.get('total_return', 0):.2%}")

    fold_result = {
        "fold": fn,
        "train_start": str(ts.date()),
        "train_end": str(te.date()),
        "test_start": str(xs.date()),
        "test_end": str(xe.date()),
        "train": {
            "sharpe": train_result["sharpe"],
            "wr": train_result["wr"],
            "pf": train_result["pf"],
            "n_trades": train_result["n_trades"],
            "total_return": train_result["total_return"],
        },
        "test_fixed": {
            "sharpe": test_result["sharpe"],
            "wr": test_result["wr"],
            "pf": test_result["pf"],
            "n_trades": test_result["n_trades"],
            "total_return": test_result["total_return"],
        },
        "adaptive_a": {
            "would_trade": adaptive_a_trades,
            "sharpe": adaptive_a_result.get("sharpe", 0.0),
            "wr": adaptive_a_result.get("wr", 0.0),
            "pf": adaptive_a_result.get("pf", 0.0),
            "n_trades": adaptive_a_result.get("n_trades", 0),
            "total_return": adaptive_a_result.get("total_return", 0.0),
        },
        "adaptive_b": {
            "position_size": round(adaptive_b_size, 2),
            "scale_factor": round(adaptive_b_scale, 4),
            "sharpe": adaptive_b_result.get("sharpe", 0.0),
            "wr": adaptive_b_result.get("wr", 0.0),
            "pf": adaptive_b_result.get("pf", 0.0),
            "n_trades": adaptive_b_result.get("n_trades", 0),
            "total_return": adaptive_b_result.get("total_return", 0.0),
        },
    }
    results.append(fold_result)
    print()

# -- Summary Statistics ------------------------------------------------------
print("="*70)
print("WALK-FORWARD SUMMARY")
print("="*70)

# Fixed params summary
test_sharpes = [r["test_fixed"]["sharpe"] for r in results]
train_sharpes = [r["train"]["sharpe"] for r in results]
test_returns = [r["test_fixed"]["total_return"] for r in results]
test_n_trades = [r["test_fixed"]["n_trades"] for r in results]

avg_test_sharpe = np.mean(test_sharpes) if test_sharpes else 0.0
pct_positive_sharpe = sum(1 for s in test_sharpes if s > 0) / len(test_sharpes) * 100 if test_sharpes else 0.0
avg_test_return = np.mean(test_returns) if test_returns else 0.0
total_test_trades = sum(test_n_trades)

# Train-test Sharpe correlation
if len(train_sharpes) > 2 and len(set(train_sharpes)) > 1 and len(set(test_sharpes)) > 1:
    corr, p_val = stats.pearsonr(train_sharpes, test_sharpes)
else:
    corr, p_val = 0.0, 1.0

print(f"\nFIXED PARAMS:")
print(f"  Avg test Sharpe:           {avg_test_sharpe:.4f}")
print(f"  % test periods Sharpe > 0: {pct_positive_sharpe:.1f}%")
print(f"  Avg test return/fold:      {avg_test_return:.2%}")
print(f"  Total test trades:         {total_test_trades}")
print(f"  Train-Test Sharpe corr:    {corr:.4f} (p={p_val:.4f})")

# Adaptive A summary
ada_sharpes = [r["adaptive_a"]["sharpe"] for r in results]
ada_returns = [r["adaptive_a"]["total_return"] for r in results]
ada_traded = sum(1 for r in results if r["adaptive_a"]["would_trade"])
ada_sat_out = len(results) - ada_traded
ada_avg_sharpe = np.mean([s for r, s in zip(results, ada_sharpes) if r["adaptive_a"]["would_trade"]]) if ada_traded > 0 else 0.0
ada_total_return = sum(ada_returns)
ada_avg_return = np.mean(ada_returns) if ada_returns else 0.0

print(f"\nADAPTIVE A (skip if train Sharpe < 0.5):")
print(f"  Folds traded:              {ada_traded}/{len(results)}")
print(f"  Folds sat out:             {ada_sat_out}")
print(f"  Avg Sharpe (traded folds): {ada_avg_sharpe:.4f}")
print(f"  Avg return/fold:           {ada_avg_return:.2%}")

# Adaptive B summary
adb_returns = [r["adaptive_b"]["total_return"] for r in results]
adb_sharpes = [r["adaptive_b"]["sharpe"] for r in results]
adb_avg_sharpe = np.mean(adb_sharpes) if adb_sharpes else 0.0
adb_avg_return = np.mean(adb_returns) if adb_returns else 0.0

print(f"\nADAPTIVE B (scale size by train Sharpe):")
print(f"  Avg test Sharpe:           {adb_avg_sharpe:.4f}")
print(f"  Avg return/fold:           {adb_avg_return:.2%}")

# Comparison table
print(f"\n{'Method':<30} {'Avg Sharpe':>12} {'Avg Ret/Fold':>14} {'Tot Trades':>12}")
print("-" * 70)
print(f"{'Fixed Params':<30} {avg_test_sharpe:>12.4f} {avg_test_return:>13.2%} {total_test_trades:>12}")
ada_tot_trades = sum(r["adaptive_a"]["n_trades"] for r in results)
adb_tot_trades = sum(r["adaptive_b"]["n_trades"] for r in results)
print(f"{'Adaptive A (Sharpe gate)':<30} {np.mean(ada_sharpes):>12.4f} {ada_avg_return:>13.2%} {ada_tot_trades:>12}")
print(f"{'Adaptive B (size scaling)':<30} {adb_avg_sharpe:>12.4f} {adb_avg_return:>13.2%} {adb_tot_trades:>12}")

# Per-fold detail table
print(f"\n{'Fold':>4} {'Train Period':<25} {'Test Period':<25} {'Tr Sharpe':>10} {'Ts Sharpe':>10} {'Ts WR':>8} {'Ts PF':>8} {'Ts Ret':>10} {'n':>4} {'AdA':>5} {'AdB $':>7}")
print("-" * 120)
for r in results:
    tr_period = f"{r['train_start']} to {r['train_end']}"
    ts_period = f"{r['test_start']} to {r['test_end']}"
    ada_flag = "TRADE" if r["adaptive_a"]["would_trade"] else "SKIP"
    print(f"{r['fold']:>4} {tr_period:<25} {ts_period:<25} "
          f"{r['train']['sharpe']:>10.3f} {r['test_fixed']['sharpe']:>10.3f} "
          f"{r['test_fixed']['wr']:>7.1%} {r['test_fixed']['pf']:>8.2f} "
          f"{r['test_fixed']['total_return']:>9.2%} {r['test_fixed']['n_trades']:>4} "
          f"{ada_flag:>5} {r['adaptive_b']['position_size']:>6.0f}")

# -- Save Results ------------------------------------------------------------
output = {
    "run_date": str(datetime.now()),
    "config": {
        "capital": CAPITAL,
        "max_per_trade": MAX_PER_TRADE,
        "max_concurrent": MAX_CONCURRENT,
        "slippage_pct": SLIPPAGE_PCT,
        "train_days": TRAIN_DAYS,
        "test_days": TEST_DAYS,
        "roll_days": ROLL_DAYS,
        "strategy_params": {
            "dip_thresh": DIP_THRESH,
            "rsi_thresh": RSI_THRESH,
            "min_red_days": MIN_RED_DAYS,
            "weekly_decline_weeks": WEEKLY_DECLINE_WEEKS,
            "hold_days": HOLD_DAYS,
        },
        "tickers": TICKERS,
    },
    "n_folds": len(results),
    "folds": results,
    "summary": {
        "fixed_params": {
            "avg_test_sharpe": round(avg_test_sharpe, 4),
            "pct_positive_sharpe": round(pct_positive_sharpe, 1),
            "avg_test_return_per_fold": round(avg_test_return, 6),
            "total_test_trades": total_test_trades,
            "train_test_sharpe_correlation": round(corr, 4),
            "train_test_sharpe_corr_pval": round(p_val, 4),
        },
        "adaptive_a": {
            "description": "Skip test window if training Sharpe < 0.5",
            "folds_traded": ada_traded,
            "folds_sat_out": ada_sat_out,
            "avg_sharpe_traded_folds": round(ada_avg_sharpe, 4),
            "avg_return_per_fold": round(ada_avg_return, 6),
            "total_trades": ada_tot_trades,
        },
        "adaptive_b": {
            "description": "Scale position size: $200 * min(train_sharpe, 2.0)",
            "avg_test_sharpe": round(adb_avg_sharpe, 4),
            "avg_return_per_fold": round(adb_avg_return, 6),
            "total_trades": adb_tot_trades,
        },
        "comparison": {
            "fixed_avg_sharpe": round(avg_test_sharpe, 4),
            "adaptive_a_avg_sharpe": round(np.mean(ada_sharpes), 4),
            "adaptive_b_avg_sharpe": round(adb_avg_sharpe, 4),
        },
    },
}

out_path = Path("/home/jupiter/Lvl3Quant/data/multi_tf_L_walk_forward_results.json")
out_path.write_text(json.dumps(output, indent=2, default=str))
print(f"\nResults saved to {out_path}")
print("DONE.")

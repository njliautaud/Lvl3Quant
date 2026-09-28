"""
VIX Mean Reversion — Adversarial Validation
=============================================
5 adversarial tests on the VIX Mean Reversion strategy (Variant E).

Tests:
1. Inverse Direction Test
2. Random Timing Test (1000 random portfolios)
3. Top-Trade Removal
4. Sub-Period Stability
5. Parameter Sensitivity

Also checks for look-ahead bias.
"""

import json
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────────
START_CAPITAL = 645.0
SLIPPAGE_PCT = 0.0002
OOT_START = "2022-01-01"
OOT_END = "2026-07-28"
DATA_START = "2021-01-01"
OUTPUT_PATH = Path("/home/jupiter/Lvl3Quant/data/vix_mean_reversion_adversarial.json")

# ── Data Download ───────────────────────────────────────────────────────────
print("Downloading data...")
tickers = {"SPY": "SPY", "VIX": "^VIX"}

data = {}
for name, ticker in tickers.items():
    df = yf.download(ticker, start=DATA_START, end=OOT_END, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    data[name] = df
    print(f"  {name}: {len(df)} rows ({df.index[0].date()} to {df.index[-1].date()})")

close = pd.DataFrame({name: df["Close"] for name, df in data.items()}).dropna(subset=["VIX", "SPY"])
close["VIX_diff_5d"] = close["VIX"].diff(5)
close["SPY_SMA200"] = close["SPY"].rolling(200).mean()
print(f"Unified data: {len(close)} rows")


# ── Core Backtest Engine ───────────────────────────────────────────────────
def backtest(close_df, signal_func, hold_days=15, half_size=True,
             oot_start=OOT_START, oot_end=OOT_END, start_capital=START_CAPITAL):
    """Run backtest. Returns dict with trades, equity curve, and stats."""
    oot = close_df.loc[oot_start:oot_end].copy()
    if len(oot) == 0:
        return None

    capital = start_capital
    position = None
    trades = []
    equity_curve = []

    for i, (dt, row) in enumerate(oot.iterrows()):
        price = row.get("SPY")
        if pd.isna(price) or price <= 0:
            equity_curve.append({"date": dt, "equity": capital})
            continue

        # Check exit
        if position is not None:
            if dt >= position["exit_target"]:
                exit_price = price * (1 - SLIPPAGE_PCT)
                pnl = (exit_price - position["entry_price"]) * position["shares"]
                capital += position["entry_price"] * position["shares"] + pnl
                trades.append({
                    "entry_date": position["entry_date"],
                    "exit_date": dt,
                    "entry_price": position["entry_price"],
                    "exit_price": exit_price,
                    "shares": position["shares"],
                    "pnl": pnl,
                    "pnl_pct": pnl / (position["entry_price"] * position["shares"]) * 100,
                })
                position = None

        # Check entry
        if position is None:
            try:
                sig = signal_func(row, dt, oot)
            except Exception:
                sig = False

            if sig:
                # Half size in bear regime
                sizing = 1.0
                if half_size:
                    spy = row.get("SPY")
                    sma200 = row.get("SPY_SMA200")
                    if pd.notna(spy) and pd.notna(sma200) and spy < sma200:
                        sizing = 0.5

                invest = capital * sizing
                entry_price = price * (1 + SLIPPAGE_PCT)
                shares = int(invest // entry_price)
                if shares > 0:
                    cost = entry_price * shares
                    capital -= cost
                    future_dates = oot.index[oot.index > dt]
                    trading_days_ahead = future_dates[:hold_days]
                    if len(trading_days_ahead) >= hold_days:
                        exit_target = trading_days_ahead[-1]
                    else:
                        exit_target = future_dates[-1] if len(future_dates) > 0 else dt + timedelta(days=hold_days * 2)
                    position = {
                        "entry_price": entry_price,
                        "shares": shares,
                        "entry_date": dt,
                        "exit_target": exit_target,
                    }

        mtm = capital
        if position is not None:
            mtm += price * position["shares"]
        equity_curve.append({"date": dt, "equity": mtm})

    # Close open position
    if position is not None:
        last_price = oot["SPY"].iloc[-1] * (1 - SLIPPAGE_PCT)
        pnl = (last_price - position["entry_price"]) * position["shares"]
        trades.append({
            "entry_date": position["entry_date"],
            "exit_date": oot.index[-1],
            "entry_price": position["entry_price"],
            "exit_price": last_price,
            "shares": position["shares"],
            "pnl": pnl,
            "pnl_pct": pnl / (position["entry_price"] * position["shares"]) * 100,
        })

    eq_df = pd.DataFrame(equity_curve).set_index("date")
    eq_df["daily_ret"] = eq_df["equity"].pct_change()
    daily_rets = eq_df["daily_ret"].dropna()

    if len(daily_rets) > 10 and daily_rets.std() > 0:
        sharpe = (daily_rets.mean() / daily_rets.std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    n_trades = len(trades)
    win_rate = (sum(1 for t in trades if t["pnl"] > 0) / n_trades * 100) if n_trades > 0 else 0
    gross_win = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gross_loss = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")

    eq_df["peak"] = eq_df["equity"].cummax()
    eq_df["dd"] = (eq_df["equity"] - eq_df["peak"]) / eq_df["peak"]
    max_dd = eq_df["dd"].min() * 100

    final_equity = eq_df["equity"].iloc[-1]

    return {
        "trades": trades,
        "equity_curve": eq_df,
        "sharpe": sharpe,
        "n_trades": n_trades,
        "win_rate": win_rate,
        "profit_factor": pf,
        "max_dd": max_dd,
        "final_equity": final_equity,
        "total_return_pct": (final_equity / start_capital - 1) * 100,
    }


# ── Signal E (original) ───────────────────────────────────────────────────
def signal_E(row, dt, df):
    """VIX Mean Reversion: VIX > 20 recently and dropped >3pts in 5 days"""
    vix = row.get("VIX")
    vix_diff = row.get("VIX_diff_5d")
    if pd.isna(vix) or pd.isna(vix_diff):
        return False
    lookback = df.loc[:dt].tail(6)
    was_above_20 = (lookback["VIX"] > 20).any()
    return was_above_20 and vix_diff < -3


# ── Run Original Strategy ──────────────────────────────────────────────────
print("\n" + "=" * 70)
print("RUNNING ORIGINAL STRATEGY (baseline)")
print("=" * 70)
original = backtest(close, signal_E, hold_days=15, half_size=True)
print(f"  Trades: {original['n_trades']} | Sharpe: {original['sharpe']:.3f} | "
      f"WR: {original['win_rate']:.1f}% | PF: {original['profit_factor']:.2f} | "
      f"Return: {original['total_return_pct']:.1f}% | MaxDD: {original['max_dd']:.1f}%")

results = {
    "strategy": "VIX Mean Reversion (Variant E)",
    "original": {
        "sharpe": round(original["sharpe"], 3),
        "n_trades": original["n_trades"],
        "win_rate": round(original["win_rate"], 1),
        "profit_factor": round(original["profit_factor"], 2),
        "max_dd": round(original["max_dd"], 1),
        "final_equity": round(original["final_equity"], 2),
        "total_return_pct": round(original["total_return_pct"], 1),
    },
    "adversarial_tests": {},
}


# ══════════════════════════════════════════════════════════════════════════
# TEST 0: LOOK-AHEAD BIAS CHECK
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 0: LOOK-AHEAD BIAS CHECK")
print("=" * 70)

# The signal uses: row.get("VIX") and row.get("VIX_diff_5d")
# row is indexed by date dt, and VIX_diff_5d = VIX.diff(5)
# In the loop: for i, (dt, row) in enumerate(oot.iterrows()):
#   The row for date dt contains VIX[dt] (today's close)
# Signal checks: was VIX > 20 in last 6 rows up to and including today
#   AND vix_diff_5d < -3 (today's VIX - VIX 5 days ago)
#
# VERDICT: The signal uses TODAY's VIX close to decide TODAY's trade.
# Since the backtest enters at the SAME day's SPY close, this IS look-ahead bias.
# In practice, you'd see VIX close at ~4:15pm ET but SPY closes at 4pm ET.
# However, in daily bar backtesting, using same-bar close for both signal
# and entry IS look-ahead bias -- you can't know the close at the open.
#
# We flag this and also test with 1-day lag to see impact.

look_ahead_found = True
look_ahead_note = (
    "LOOK-AHEAD BIAS DETECTED: Signal uses today's VIX close (row['VIX'] and "
    "row['VIX_diff_5d']) to decide today's trade entry at today's SPY close. "
    "In reality, you cannot know VIX close before SPY close (VIX settles at "
    "4:15pm ET, SPY at 4pm ET, but the backtest uses daily close-to-close). "
    "Proper implementation: use YESTERDAY's VIX close to decide TODAY's entry. "
    "Testing with 1-day lag below."
)
print(f"  {look_ahead_note}")


def signal_E_lagged(row, dt, df):
    """Same as signal_E but uses YESTERDAY's data (1-day lag to fix look-ahead)."""
    idx = df.index.get_loc(dt)
    if idx < 1:
        return False
    prev_dt = df.index[idx - 1]
    prev_row = df.loc[prev_dt]
    vix = prev_row.get("VIX")
    vix_diff = prev_row.get("VIX_diff_5d")
    if pd.isna(vix) or pd.isna(vix_diff):
        return False
    lookback = df.loc[:prev_dt].tail(6)
    was_above_20 = (lookback["VIX"] > 20).any()
    return was_above_20 and vix_diff < -3


lagged = backtest(close, signal_E_lagged, hold_days=15, half_size=True)
print(f"  Lagged version: Trades={lagged['n_trades']} | Sharpe={lagged['sharpe']:.3f} | "
      f"WR={lagged['win_rate']:.1f}% | Return={lagged['total_return_pct']:.1f}%")

sharpe_degradation = (original["sharpe"] - lagged["sharpe"]) / original["sharpe"] * 100 if original["sharpe"] != 0 else 0
print(f"  Sharpe degradation with lag fix: {sharpe_degradation:.1f}%")

results["look_ahead_bias"] = {
    "detected": look_ahead_found,
    "description": look_ahead_note,
    "lagged_sharpe": round(lagged["sharpe"], 3),
    "lagged_n_trades": lagged["n_trades"],
    "lagged_win_rate": round(lagged["win_rate"], 1),
    "sharpe_degradation_pct": round(sharpe_degradation, 1),
    "verdict": "Sharpe still positive with lag" if lagged["sharpe"] > 0 else "SIGNAL DESTROYED by lag fix",
}


# ══════════════════════════════════════════════════════════════════════════
# TEST 1: INVERSE DIRECTION TEST
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 1: INVERSE DIRECTION TEST")
print("=" * 70)
print("  When strategy says BUY, go to CASH. When strategy says CASH, be LONG.")


def backtest_inverse(close_df, signal_func, hold_days=15):
    """Inverse: be long when signal is OFF, go to cash when signal fires."""
    oot = close_df.loc[OOT_START:OOT_END].copy()
    if len(oot) == 0:
        return None

    # Track when signal-based strategy would be IN the market
    in_market_days = set()
    position_active = False
    entry_day = None
    days_held = 0

    for i, (dt, row) in enumerate(oot.iterrows()):
        if position_active:
            in_market_days.add(dt)
            days_held += 1
            if days_held >= hold_days:
                position_active = False
                days_held = 0
        else:
            try:
                sig = signal_func(row, dt, oot)
            except Exception:
                sig = False
            if sig:
                position_active = True
                in_market_days.add(dt)
                days_held = 1

    # Inverse: be in market on days NOT in in_market_days
    capital = START_CAPITAL
    position = None
    equity_curve = []

    for i, (dt, row) in enumerate(oot.iterrows()):
        price = row.get("SPY")
        if pd.isna(price) or price <= 0:
            equity_curve.append({"date": dt, "equity": capital})
            continue

        should_be_in = dt not in in_market_days

        if position is not None and not should_be_in:
            # Exit
            exit_price = price * (1 - SLIPPAGE_PCT)
            pnl = (exit_price - position["entry_price"]) * position["shares"]
            capital += position["entry_price"] * position["shares"] + pnl
            position = None

        if position is None and should_be_in:
            # Enter
            entry_price = price * (1 + SLIPPAGE_PCT)
            shares = int(capital // entry_price)
            if shares > 0:
                capital -= entry_price * shares
                position = {"entry_price": entry_price, "shares": shares}

        mtm = capital
        if position is not None:
            mtm += price * position["shares"]
        equity_curve.append({"date": dt, "equity": mtm})

    eq_df = pd.DataFrame(equity_curve).set_index("date")
    eq_df["daily_ret"] = eq_df["equity"].pct_change()
    daily_rets = eq_df["daily_ret"].dropna()

    if len(daily_rets) > 10 and daily_rets.std() > 0:
        sharpe = (daily_rets.mean() / daily_rets.std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    final_eq = eq_df["equity"].iloc[-1]
    return {
        "sharpe": sharpe,
        "final_equity": final_eq,
        "total_return_pct": (final_eq / START_CAPITAL - 1) * 100,
        "pct_days_in_market": (1 - len(in_market_days) / len(oot)) * 100,
    }


inverse_result = backtest_inverse(close, signal_E, hold_days=15)
inverse_sharpe = inverse_result["sharpe"]
test1_pass = inverse_sharpe <= 0

print(f"  Original Sharpe:  {original['sharpe']:.3f}")
print(f"  Inverse Sharpe:   {inverse_sharpe:.3f}")
print(f"  Inverse Return:   {inverse_result['total_return_pct']:.1f}%")
print(f"  Inverse % days in market: {inverse_result['pct_days_in_market']:.1f}%")
print(f"  VERDICT: {'PASS - inverse has no edge' if test1_pass else 'FAIL - inverse also profitable, signal may be market exposure'}")

# NOTE: For a long-only strategy in a generally uptrending market (2022-2026 included
# both bear and bull), the inverse being long most of the time WILL likely be profitable
# because of baseline equity beta. This is expected and the more meaningful check is
# whether inverse Sharpe > original Sharpe.
inverse_beats_original = inverse_sharpe > original["sharpe"]
print(f"\n  NUANCED CHECK: Does inverse beat original?")
print(f"  Inverse Sharpe ({inverse_sharpe:.3f}) {'>' if inverse_beats_original else '<='} Original Sharpe ({original['sharpe']:.3f})")
if inverse_beats_original:
    print(f"  WARNING: Inverse beats original! Signal timing is WORSE than random/passive.")
else:
    print(f"  OK: Original beats inverse. Signal timing adds value over passive holding.")

results["adversarial_tests"]["1_inverse_direction"] = {
    "original_sharpe": round(original["sharpe"], 3),
    "inverse_sharpe": round(inverse_sharpe, 3),
    "inverse_return_pct": round(inverse_result["total_return_pct"], 1),
    "inverse_pct_days_in_market": round(inverse_result["pct_days_in_market"], 1),
    "strict_pass": test1_pass,
    "inverse_beats_original": inverse_beats_original,
    "note": (
        "For long-only strategies, the 'inverse' is long most of the time and benefits "
        "from equity beta. A positive inverse Sharpe doesn't necessarily invalidate the signal. "
        "The key question is whether the signal's TIMING adds value vs passive (inverse Sharpe < original)."
    ),
    "verdict": "PASS" if test1_pass else ("MARGINAL_PASS" if not inverse_beats_original else "FAIL"),
}


# ══════════════════════════════════════════════════════════════════════════
# TEST 2: RANDOM TIMING TEST (1000 random portfolios)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 2: RANDOM TIMING TEST (1000 random portfolios)")
print("=" * 70)

oot_data = close.loc[OOT_START:OOT_END]
n_trades_orig = original["n_trades"]
hold_days_orig = 15

np.random.seed(42)
random_sharpes = []
N_RANDOM = 1000

# Get all valid trading days (need enough room for hold period)
all_dates = oot_data.index.tolist()
max_entry_idx = len(all_dates) - hold_days_orig - 1

for trial in range(N_RANDOM):
    # Generate random entry indices (non-overlapping)
    capital = START_CAPITAL
    equity = [START_CAPITAL] * len(all_dates)
    position = None
    trades_placed = 0
    cooldown = 0

    # Pick random entry points
    candidate_indices = list(range(max_entry_idx))
    np.random.shuffle(candidate_indices)

    entry_indices = []
    blocked = set()
    for idx in candidate_indices:
        if idx not in blocked and trades_placed < n_trades_orig:
            entry_indices.append(idx)
            trades_placed += 1
            for b in range(idx, min(idx + hold_days_orig + 1, len(all_dates))):
                blocked.add(b)

    entry_indices.sort()

    # Run simplified backtest
    capital = START_CAPITAL
    position = None
    eq_values = []

    for i, dt in enumerate(all_dates):
        price = oot_data.loc[dt, "SPY"]
        if pd.isna(price):
            eq_values.append(capital)
            continue

        # Check exit
        if position is not None and i >= position["exit_idx"]:
            exit_price = price * (1 - SLIPPAGE_PCT)
            pnl = (exit_price - position["entry_price"]) * position["shares"]
            capital += position["entry_price"] * position["shares"] + pnl
            position = None

        # Check entry
        if position is None and i in entry_indices:
            entry_price = price * (1 + SLIPPAGE_PCT)
            shares = int(capital // entry_price)
            if shares > 0:
                capital -= entry_price * shares
                exit_idx = min(i + hold_days_orig, len(all_dates) - 1)
                position = {"entry_price": entry_price, "shares": shares, "exit_idx": exit_idx}

        mtm = capital
        if position is not None:
            mtm += price * position["shares"]
        eq_values.append(mtm)

    # Calculate Sharpe
    eq_series = pd.Series(eq_values)
    daily_rets = eq_series.pct_change().dropna()
    if len(daily_rets) > 10 and daily_rets.std() > 0:
        sharpe = (daily_rets.mean() / daily_rets.std()) * np.sqrt(252)
    else:
        sharpe = 0.0
    random_sharpes.append(sharpe)

random_sharpes = np.array(random_sharpes)
percentile = (random_sharpes < original["sharpe"]).mean() * 100
test2_pass = percentile > 95

print(f"  Original Sharpe: {original['sharpe']:.3f}")
print(f"  Random Sharpe distribution: mean={random_sharpes.mean():.3f}, "
      f"std={random_sharpes.std():.3f}, median={np.median(random_sharpes):.3f}")
print(f"  Random Sharpe range: [{random_sharpes.min():.3f}, {random_sharpes.max():.3f}]")
print(f"  Strategy percentile: {percentile:.1f}th")
print(f"  95th percentile threshold: {np.percentile(random_sharpes, 95):.3f}")
print(f"  VERDICT: {'PASS' if test2_pass else 'FAIL'} (need >95th percentile, got {percentile:.1f}th)")

results["adversarial_tests"]["2_random_timing"] = {
    "original_sharpe": round(original["sharpe"], 3),
    "random_mean_sharpe": round(float(random_sharpes.mean()), 3),
    "random_std_sharpe": round(float(random_sharpes.std()), 3),
    "random_median_sharpe": round(float(np.median(random_sharpes)), 3),
    "random_95th_pct": round(float(np.percentile(random_sharpes, 95)), 3),
    "strategy_percentile": round(float(percentile), 1),
    "pass": test2_pass,
    "verdict": "PASS" if test2_pass else "FAIL",
}


# ══════════════════════════════════════════════════════════════════════════
# TEST 3: TOP-TRADE REMOVAL
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 3: TOP-TRADE REMOVAL (remove best 3 trades)")
print("=" * 70)

trades = original["trades"]
trades_sorted = sorted(trades, key=lambda t: t["pnl_pct"], reverse=True)

print(f"  Top 3 trades by return:")
for i, t in enumerate(trades_sorted[:3]):
    print(f"    #{i+1}: {t['entry_date'].strftime('%Y-%m-%d')} -> {t['exit_date'].strftime('%Y-%m-%d')}, "
          f"return={t['pnl_pct']:.2f}%, pnl=${t['pnl']:.2f}")

# Remove top 3 trades and re-run
top3_entry_dates = {t["entry_date"] for t in trades_sorted[:3]}


def signal_E_no_top3(row, dt, df):
    """Same as signal_E but skip top 3 trade entry dates."""
    if dt in top3_entry_dates:
        return False
    return signal_E(row, dt, df)


trimmed = backtest(close, signal_E_no_top3, hold_days=15, half_size=True)
test3_pass = trimmed["sharpe"] >= 0.3

print(f"\n  Original: Sharpe={original['sharpe']:.3f}, Trades={original['n_trades']}")
print(f"  Trimmed:  Sharpe={trimmed['sharpe']:.3f}, Trades={trimmed['n_trades']}")
print(f"  Sharpe drop: {original['sharpe']:.3f} -> {trimmed['sharpe']:.3f} "
      f"({(1 - trimmed['sharpe']/original['sharpe'])*100:.1f}% decline)")
print(f"  VERDICT: {'PASS' if test3_pass else 'FAIL'} (need Sharpe >= 0.3 after removal, got {trimmed['sharpe']:.3f})")

results["adversarial_tests"]["3_top_trade_removal"] = {
    "original_sharpe": round(original["sharpe"], 3),
    "trimmed_sharpe": round(trimmed["sharpe"], 3),
    "trimmed_n_trades": trimmed["n_trades"],
    "trimmed_win_rate": round(trimmed["win_rate"], 1),
    "sharpe_decline_pct": round((1 - trimmed["sharpe"] / original["sharpe"]) * 100, 1) if original["sharpe"] != 0 else 0,
    "top_3_trades": [
        {
            "entry": t["entry_date"].strftime("%Y-%m-%d"),
            "exit": t["exit_date"].strftime("%Y-%m-%d"),
            "return_pct": round(t["pnl_pct"], 2),
        }
        for t in trades_sorted[:3]
    ],
    "pass": test3_pass,
    "verdict": "PASS" if test3_pass else "FAIL",
}


# ══════════════════════════════════════════════════════════════════════════
# TEST 4: SUB-PERIOD STABILITY
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 4: SUB-PERIOD STABILITY")
print("=" * 70)

sub_periods = [
    ("P1: Jan 2022 - Jun 2023", "2022-01-01", "2023-06-30"),
    ("P2: Jul 2023 - Dec 2024", "2023-07-01", "2024-12-31"),
    ("P3: Jan 2025 - Jul 2026", "2025-01-01", "2026-07-28"),
]

sub_results = []
all_positive = True

for label, sp_start, sp_end in sub_periods:
    sub = backtest(close, signal_E, hold_days=15, half_size=True,
                   oot_start=sp_start, oot_end=sp_end)
    if sub is None or sub["n_trades"] == 0:
        print(f"  {label}: NO TRADES")
        sub_results.append({"period": label, "n_trades": 0, "sharpe": 0, "pass": False})
        all_positive = False
    else:
        positive = sub["sharpe"] > 0
        if not positive:
            all_positive = False
        print(f"  {label}: Trades={sub['n_trades']}, Sharpe={sub['sharpe']:.3f}, "
              f"WR={sub['win_rate']:.1f}%, Return={sub['total_return_pct']:.1f}% "
              f"-> {'PASS' if positive else 'FAIL'}")
        sub_results.append({
            "period": label,
            "n_trades": sub["n_trades"],
            "sharpe": round(sub["sharpe"], 3),
            "win_rate": round(sub["win_rate"], 1),
            "return_pct": round(sub["total_return_pct"], 1),
            "profit_factor": round(sub["profit_factor"], 2),
            "pass": positive,
        })

test4_pass = all_positive
print(f"\n  VERDICT: {'PASS' if test4_pass else 'FAIL'} (all sub-periods need Sharpe > 0)")

results["adversarial_tests"]["4_sub_period_stability"] = {
    "sub_periods": sub_results,
    "all_positive_sharpe": all_positive,
    "pass": test4_pass,
    "verdict": "PASS" if test4_pass else "FAIL",
}


# ══════════════════════════════════════════════════════════════════════════
# TEST 5: PARAMETER SENSITIVITY
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 5: PARAMETER SENSITIVITY")
print("=" * 70)

# Test: VIX threshold, SMA lookback (here diff period), hold period
vix_thresholds = [20, 25, 30]
diff_lookbacks = [3, 5, 7, 10]
hold_periods = [10, 15, 20, 30]

sensitivity_results = {}

# VIX threshold sensitivity
print("\n  VIX Threshold Sensitivity:")
threshold_sharpes = {}
for thresh in vix_thresholds:
    def make_signal(threshold):
        def sig(row, dt, df):
            vix = row.get("VIX")
            vix_diff = row.get("VIX_diff_5d")
            if pd.isna(vix) or pd.isna(vix_diff):
                return False
            lookback = df.loc[:dt].tail(6)
            was_above = (lookback["VIX"] > threshold).any()
            return was_above and vix_diff < -3
        return sig

    res = backtest(close, make_signal(thresh), hold_days=15, half_size=True)
    sharpe = res["sharpe"] if res else 0
    n = res["n_trades"] if res else 0
    threshold_sharpes[thresh] = {"sharpe": round(sharpe, 3), "n_trades": n}
    marker = " <-- ORIGINAL" if thresh == 20 else ""
    print(f"    Threshold > {thresh}: Sharpe={sharpe:.3f}, Trades={n}{marker}")

sensitivity_results["vix_threshold"] = threshold_sharpes

# Diff lookback sensitivity (replaces SMA lookback)
print("\n  VIX Diff Lookback Sensitivity:")
lookback_sharpes = {}
for lb in diff_lookbacks:
    # Recompute diff with different lookback
    close_copy = close.copy()
    close_copy[f"VIX_diff_{lb}d"] = close_copy["VIX"].diff(lb)

    def make_signal_lb(lookback_val, df_ref):
        col = f"VIX_diff_{lookback_val}d"
        def sig(row, dt, df):
            vix = row.get("VIX")
            # Get from the modified df
            if dt in df_ref.index:
                vix_diff = df_ref.loc[dt, col]
            else:
                return False
            if pd.isna(vix) or pd.isna(vix_diff):
                return False
            lookback = df.loc[:dt].tail(lookback_val + 1)
            was_above_20 = (lookback["VIX"] > 20).any()
            return was_above_20 and vix_diff < -3
        return sig

    res = backtest(close_copy, make_signal_lb(lb, close_copy), hold_days=15, half_size=True)
    sharpe = res["sharpe"] if res else 0
    n = res["n_trades"] if res else 0
    lookback_sharpes[lb] = {"sharpe": round(sharpe, 3), "n_trades": n}
    marker = " <-- ORIGINAL" if lb == 5 else ""
    print(f"    Lookback={lb}d: Sharpe={sharpe:.3f}, Trades={n}{marker}")

sensitivity_results["diff_lookback"] = lookback_sharpes

# Hold period sensitivity
print("\n  Hold Period Sensitivity:")
hold_sharpes = {}
for hp in hold_periods:
    res = backtest(close, signal_E, hold_days=hp, half_size=True)
    sharpe = res["sharpe"] if res else 0
    n = res["n_trades"] if res else 0
    hold_sharpes[hp] = {"sharpe": round(sharpe, 3), "n_trades": n}
    marker = " <-- ORIGINAL" if hp == 15 else ""
    print(f"    Hold={hp}d: Sharpe={sharpe:.3f}, Trades={n}{marker}")

sensitivity_results["hold_period"] = hold_sharpes

# Check if original is a sharp peak
all_sharpes = (
    [v["sharpe"] for v in threshold_sharpes.values()] +
    [v["sharpe"] for v in lookback_sharpes.values()] +
    [v["sharpe"] for v in hold_sharpes.values()]
)
orig_sharpe = original["sharpe"]
# Count how many neighbor configs have Sharpe > 0.3
positive_neighbors = sum(1 for s in all_sharpes if s > 0.3)
total_configs = len(all_sharpes)
pct_positive = positive_neighbors / total_configs * 100

# Check if original is sharp peak: is it >2x the median of neighbors?
median_neighbor = np.median(all_sharpes)
is_sharp_peak = orig_sharpe > 2 * median_neighbor and median_neighbor < 0.3

test5_pass = not is_sharp_peak and pct_positive >= 50

print(f"\n  Configs with Sharpe > 0.3: {positive_neighbors}/{total_configs} ({pct_positive:.0f}%)")
print(f"  Median neighbor Sharpe: {median_neighbor:.3f}")
print(f"  Original/Median ratio: {orig_sharpe/median_neighbor:.2f}x" if median_neighbor > 0 else "  Median = 0")
print(f"  Sharp peak detected: {'YES' if is_sharp_peak else 'NO'}")
print(f"  VERDICT: {'PASS' if test5_pass else 'FAIL'}")

results["adversarial_tests"]["5_parameter_sensitivity"] = {
    "sensitivity": {k: {str(kk): vv for kk, vv in v.items()} for k, v in sensitivity_results.items()},
    "pct_configs_sharpe_gt_03": round(pct_positive, 1),
    "median_neighbor_sharpe": round(float(median_neighbor), 3),
    "is_sharp_peak": is_sharp_peak,
    "pass": test5_pass,
    "verdict": "PASS" if test5_pass else "FAIL",
}


# ══════════════════════════════════════════════════════════════════════════
# FINAL SCORING
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("ADVERSARIAL VALIDATION SCORECARD")
print("=" * 70)

# For test 1, use the nuanced check (inverse shouldn't BEAT original)
test1_final = not inverse_beats_original  # Pass if original beats inverse

test_results = {
    "1_inverse_direction": test1_final,
    "2_random_timing": test2_pass,
    "3_top_trade_removal": test3_pass,
    "4_sub_period_stability": test4_pass,
    "5_parameter_sensitivity": test5_pass,
}

score = sum(test_results.values())
print(f"\n  Test 1 (Inverse Direction):     {'PASS' if test1_final else 'FAIL'}")
print(f"  Test 2 (Random Timing >95pct):  {'PASS' if test2_pass else 'FAIL'}")
print(f"  Test 3 (Top-Trade Removal):     {'PASS' if test3_pass else 'FAIL'}")
print(f"  Test 4 (Sub-Period Stability):  {'PASS' if test4_pass else 'FAIL'}")
print(f"  Test 5 (Param Sensitivity):     {'PASS' if test5_pass else 'FAIL'}")
print(f"\n  ADVERSARIAL SCORE: {score}/5")

if look_ahead_found:
    print(f"\n  WARNING: Look-ahead bias detected.")
    print(f"  Lagged Sharpe: {lagged['sharpe']:.3f} (vs original {original['sharpe']:.3f})")

if score >= 4:
    overall = "STRONG — strategy is likely robust"
elif score >= 3:
    overall = "MODERATE — some concerns but may still be tradable"
elif score >= 2:
    overall = "WEAK — significant concerns about robustness"
else:
    overall = "REJECT — strategy fails adversarial validation"

print(f"\n  OVERALL ASSESSMENT: {overall}")

results["score"] = f"{score}/5",
results["overall_assessment"] = overall
results["test_pass_fail"] = {k: ("PASS" if v else "FAIL") for k, v in test_results.items()}
results["run_timestamp"] = datetime.now().isoformat()

# Save
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nResults saved to {OUTPUT_PATH}")

#!/usr/bin/env python3
"""
Adversarial Validation: Multi-Timeframe MR Variant L
=====================================================
"Dual Signal D + Weekly RSI Declining 2+ Weeks"

6 adversarial checks:
1) Inverse Signal — buy near highs, high RSI, no red streaks, weekly RSI rising
2) Random Entry Timing — 1000 random entry sets, percentile rank
3) Sub-Period Stability — 4 equal sub-periods, all must have positive Sharpe
4) Remove Top 3 Tickers — Sharpe drop must be < 50%
5) Parameter Sensitivity — sweep dip/RSI/weekly-decline-weeks/hold
6) Cost Sensitivity — test at 5/10/20/50 bps slippage, find breakeven

Entry conditions (ALL must be true):
  1. Daily: Stock drops >5% from 20-day high
  2. Daily: RSI(14) < 35
  3. Daily: First green day after 3+ consecutive red days
  4. Weekly: Weekly RSI(14) declining for 2+ consecutive weeks

Hold: 10 trading days. Capital: $645, max $200/trade, max 3 concurrent.
Slippage: 0.02% baseline each way.

OOT: Jan 2022 - Jul 2026.
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")
np.random.seed(42)

# -- Configuration -----------------------------------------------------------
CAPITAL = 645.0
MAX_PER_TRADE = 200.0
MAX_CONCURRENT = 3
SLIPPAGE_PCT = 0.0002  # 0.02% baseline
START = "2020-01-01"
END = "2026-07-31"
OOT_START = "2022-01-01"
N_PERM = 1000

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


def sma(series, period):
    return series.rolling(period).mean()


# -- Pre-compute daily indicators -------------------------------------------
indicators = {}
for t in TICKERS:
    c = closes.get(t, pd.Series(dtype=float))
    if len(c) < 300:
        continue
    ind = pd.DataFrame(index=c.index)
    ind["close"] = c
    ind["high20"] = c.rolling(20).max()
    ind["dip_from_high"] = (c - ind["high20"]) / ind["high20"]  # negative when below
    ind["rsi14"] = calc_rsi(c, 14)
    # Daily return for green/red day detection
    ind["daily_ret"] = c.pct_change()
    # Consecutive red days count (looking back)
    red = (ind["daily_ret"] < 0).astype(int)
    # Count consecutive reds ending yesterday
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
    # Resample to weekly (Friday close)
    weekly_close = c.resample("W-FRI").last().dropna()
    w_rsi = calc_rsi(weekly_close, 14)
    # Track declining: current < prior
    w_declining = w_rsi.diff() < 0  # True if RSI dropped this week
    # Count consecutive declining weeks
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

# SPY regime
spy_sma200 = sma(spy_close, 200)
spy_regime = (spy_close > spy_sma200).reindex(spy_close.index).fillna(False)

print(f"  Tickers with indicators: {len(indicators)}")
print(f"  Tickers with weekly RSI: {len(weekly_rsi_data)}")


# -- Weekly RSI lookup helper ------------------------------------------------
def get_weekly_decline_weeks(ticker, date, min_weeks=2):
    """Check if weekly RSI has been declining for min_weeks consecutive weeks."""
    wdata = weekly_rsi_data.get(ticker)
    if wdata is None:
        return False, 0
    # Find the most recent completed week on or before this date
    valid = wdata.index[wdata.index <= date]
    if len(valid) < 2:
        return False, 0
    latest_week = valid[-1]
    weeks = int(wdata.loc[latest_week, "consec_decline_weeks"])
    return weeks >= min_weeks, weeks


# -- Signal Generator: Variant L ---------------------------------------------
def gen_signals_L(tickers, dip_thresh=-0.05, rsi_thresh=35.0,
                  min_red_days=3, weekly_decline_weeks=2,
                  hold_days=10, inverse=False):
    """
    Generate signals for Multi-TF MR Variant L.
    inverse=True: buy near highs, high RSI, no red streaks, weekly RSI rising.
    """
    signals = []
    for t in tickers:
        ind = indicators.get(t)
        if ind is None:
            continue
        for i in range(1, len(ind)):
            dt = ind.index[i]
            if str(dt.date()) < OOT_START:
                continue

            if inverse:
                # INVERSE: near highs, high RSI, no red streak, weekly RSI rising
                dip = ind["dip_from_high"].iloc[i]
                if dip < -0.02:  # must be within 2% of 20d high
                    continue
                if ind["rsi14"].iloc[i] <= 65:
                    continue
                # No red day streaks needed — skip red streak check
                # Weekly RSI must be RISING for 2+ weeks
                wdata = weekly_rsi_data.get(t)
                if wdata is None:
                    continue
                valid_w = wdata.index[wdata.index <= dt]
                if len(valid_w) < 3:
                    continue
                # Check rising: current > prior > prior-prior
                w_rsi_vals = wdata.loc[valid_w[-3:], "weekly_rsi"]
                if len(w_rsi_vals) < 3:
                    continue
                if not (w_rsi_vals.iloc[2] > w_rsi_vals.iloc[1] > w_rsi_vals.iloc[0]):
                    continue
                signals.append((dt, t, float(ind["close"].iloc[i]),
                                "inverse", hold_days))
            else:
                # NORMAL: dip from high, low RSI, first green after red streak, weekly RSI declining
                dip = ind["dip_from_high"].iloc[i]
                if dip > dip_thresh:  # dip_thresh is negative, e.g. -0.05
                    continue
                if ind["rsi14"].iloc[i] >= rsi_thresh:
                    continue
                if not ind["is_green"].iloc[i]:
                    continue
                if ind["consec_red_before"].iloc[i] < min_red_days:
                    continue
                # Weekly RSI declining check
                passes, _ = get_weekly_decline_weeks(t, dt, min_weeks=weekly_decline_weeks)
                if not passes:
                    continue
                signals.append((dt, t, float(ind["close"].iloc[i]),
                                "variant_L", hold_days))
    return sorted(signals, key=lambda x: x[0])


# -- Backtest Engine ---------------------------------------------------------
def run_backtest(signals, capital=CAPITAL, slippage=SLIPPAGE_PCT,
                 max_concurrent=MAX_CONCURRENT, max_per_trade=MAX_PER_TRADE,
                 oot_start=OOT_START, oot_end=None):
    if not signals:
        return np.array([]), [], []

    signals = sorted(signals, key=lambda x: x[0])
    equity = capital
    equity_curve = [(oot_start, capital)]
    realized = []
    open_positions = []

    for sig in signals:
        date, ticker, entry_price, tag, hold_days = sig
        date_str = str(date.date()) if hasattr(date, 'date') else str(date)
        if date_str < oot_start:
            continue
        if oot_end and date_str > oot_end:
            continue

        # Close expired positions
        still_open = [p for p in open_positions if p["exit_date"] > date_str]
        open_positions = still_open

        if len(open_positions) >= max_concurrent:
            continue

        ind = indicators.get(ticker)
        if ind is None or date not in ind.index:
            continue
        loc = ind.index.get_loc(date)

        # Fixed hold exit
        exit_idx = min(loc + hold_days, len(ind) - 1)
        if exit_idx <= loc:
            exit_idx = min(loc + 1, len(ind) - 1)

        if oot_end:
            exit_date_str = str(ind.index[exit_idx].date())
            if exit_date_str > oot_end:
                for ei in range(exit_idx, loc, -1):
                    if str(ind.index[ei].date()) <= oot_end:
                        exit_idx = ei
                        break
                else:
                    continue

        exit_date_str = str(ind.index[exit_idx].date())

        # Position sizing: min(max_per_trade, equity / max_concurrent)
        alloc = min(max_per_trade, equity / max_concurrent)
        actual_entry = entry_price * (1 + slippage)
        exit_price = float(ind["close"].iloc[exit_idx]) * (1 - slippage)

        shares = int(alloc / actual_entry)
        if shares < 1:
            if actual_entry <= equity:
                shares = 1
            else:
                continue

        pnl = shares * (exit_price - actual_entry)
        ret = pnl / (shares * actual_entry)
        actual_hold = (ind.index[exit_idx] - ind.index[loc]).days

        regime = "Bull" if spy_regime.get(ind.index[loc], False) else "Bear"

        trade = {
            "ticker": ticker,
            "entry_date": date_str,
            "exit_date": exit_date_str,
            "entry_price": round(actual_entry, 4),
            "exit_price": round(exit_price, 4),
            "shares": shares,
            "pnl": round(pnl, 2),
            "return": round(ret, 6),
            "hold_days": actual_hold,
            "regime": regime,
            "tag": tag,
        }

        equity += pnl
        equity_curve.append((exit_date_str, round(equity, 2)))
        realized.append(trade)
        open_positions.append({"ticker": ticker, "exit_date": exit_date_str})

    returns = np.array([t["return"] for t in realized])
    return returns, equity_curve, realized


# -- Metrics -----------------------------------------------------------------
def calc_metrics(returns, equity_curve, realized, capital=CAPITAL):
    if len(returns) < 2:
        return {
            "n_trades": len(returns), "sharpe": 0.0, "sortino": 0.0,
            "profit_factor": 0.0, "win_rate": 0.0, "max_dd_pct": 0.0,
            "total_return_pct": 0.0, "final_equity": capital,
        }

    n = len(returns)
    wins = int((returns > 0).sum())
    wr = wins / n

    avg_hold = np.mean([t["hold_days"] for t in realized]) if realized else 10.0
    trades_per_year = max(1, 252 / max(avg_hold, 1))
    ann_factor = np.sqrt(trades_per_year)

    mean_r = returns.mean()
    std_r = returns.std() if returns.std() > 0 else 1e-9
    sharpe = (mean_r / std_r) * ann_factor

    downside = returns[returns < 0]
    down_std = downside.std() if len(downside) > 0 and downside.std() > 0 else 1e-9
    sortino = (mean_r / down_std) * ann_factor

    gross_profit = returns[returns > 0].sum() if (returns > 0).any() else 0
    gross_loss = abs(returns[returns < 0].sum()) if (returns < 0).any() else 1e-9
    profit_factor = gross_profit / gross_loss

    eq_vals = [e[1] for e in equity_curve]
    peak = eq_vals[0]
    max_dd = 0
    for v in eq_vals:
        if v > peak:
            peak = v
        dd = (v - peak) / peak
        if dd < max_dd:
            max_dd = dd

    final_eq = eq_vals[-1]
    total_return_pct = ((final_eq - capital) / capital) * 100

    # Regime breakdown
    bull_rets = np.array([t["return"] for t in realized if t["regime"] == "Bull"])
    bear_rets = np.array([t["return"] for t in realized if t["regime"] == "Bear"])
    bull_sharpe = (bull_rets.mean() / max(bull_rets.std(), 1e-9)) * ann_factor if len(bull_rets) >= 2 else 0.0
    bear_sharpe = (bear_rets.mean() / max(bear_rets.std(), 1e-9)) * ann_factor if len(bear_rets) >= 2 else 0.0

    regime_gap = abs(bull_sharpe - bear_sharpe) / max(abs(bull_sharpe), abs(bear_sharpe), 1e-9)

    return {
        "n_trades": int(n),
        "win_rate": round(float(wr), 4),
        "sharpe": round(float(sharpe), 3),
        "sortino": round(float(sortino), 3),
        "profit_factor": round(float(profit_factor), 3),
        "max_dd_pct": round(float(max_dd * 100), 2),
        "total_return_pct": round(float(total_return_pct), 2),
        "final_equity": round(float(final_eq), 2),
        "bull_sharpe": round(float(bull_sharpe), 3),
        "bear_sharpe": round(float(bear_sharpe), 3),
        "regime_gap": round(float(regime_gap), 3),
        "bull_trades": int(len(bull_rets)),
        "bear_trades": int(len(bear_rets)),
    }


# ============================================================================
#  BASELINE
# ============================================================================
print("\n" + "=" * 70)
print("  BASELINE: Multi-TF MR Variant L")
print("=" * 70)

real_signals = gen_signals_L(TICKERS)
real_returns, real_eq, real_trades = run_backtest(real_signals)
real_metrics = calc_metrics(real_returns, real_eq, real_trades)
real_sharpe = real_metrics["sharpe"]

print(f"  Trades: {real_metrics['n_trades']}, Sharpe: {real_sharpe:.3f}, "
      f"WR: {real_metrics['win_rate']:.1%}, PF: {real_metrics['profit_factor']:.2f}")
print(f"  MDD: {real_metrics['max_dd_pct']:.2f}%, Total Return: {real_metrics['total_return_pct']:.1f}%")
print(f"  Bull Sharpe: {real_metrics['bull_sharpe']:.3f} ({real_metrics['bull_trades']} trades), "
      f"Bear Sharpe: {real_metrics['bear_sharpe']:.3f} ({real_metrics['bear_trades']} trades)")
print(f"  Regime Gap: {real_metrics['regime_gap']:.3f}")

# ============================================================================
#  CHECK 1: INVERSE SIGNAL
# ============================================================================
print("\n" + "=" * 70)
print("  CHECK 1: INVERSE SIGNAL")
print("  (Near highs, RSI>65, no red streaks, weekly RSI rising 2+ weeks)")
print("=" * 70)

inv_signals = gen_signals_L(TICKERS, inverse=True)
inv_returns, inv_eq, inv_trades = run_backtest(inv_signals)
inv_metrics = calc_metrics(inv_returns, inv_eq, inv_trades)
inv_sharpe = inv_metrics["sharpe"]

# FAIL if inverse Sharpe > 50% of original
inv_ratio = inv_sharpe / real_sharpe if abs(real_sharpe) > 0 else 0
check1_pass = inv_sharpe < (0.5 * real_sharpe)
print(f"  Inverse Sharpe: {inv_sharpe:.3f} (trades: {inv_metrics['n_trades']})")
print(f"  Inverse WR: {inv_metrics['win_rate']:.1%}, PF: {inv_metrics['profit_factor']:.2f}")
print(f"  Inverse/Real ratio: {inv_ratio:.3f} (need < 0.50 to pass)")
print(f"  CHECK 1: {'PASS' if check1_pass else 'FAIL'}")

# ============================================================================
#  CHECK 2: RANDOM ENTRY TIMING (1000 permutations)
# ============================================================================
print("\n" + "=" * 70)
print("  CHECK 2: RANDOM ENTRY TIMING (1000 permutations)")
print("=" * 70)

# Build valid OOT dates per ticker
oot_dates_per_ticker = {}
for t in TICKERS:
    ind = indicators.get(t)
    if ind is not None:
        valid = ind.index[ind.index >= OOT_START]
        if len(valid) > 15:
            oot_dates_per_ticker[t] = valid[:-10]

# Get per-trade ticker list from real signals
real_signal_tickers = [s[1] for s in real_signals
                       if str(s[0].date()) >= OOT_START and s[1] in oot_dates_per_ticker]

n_real_trades = len(real_signal_tickers)
print(f"  Real strategy: Sharpe {real_sharpe:.3f}, {n_real_trades} trades")

perm_sharpes = np.zeros(N_PERM)
print(f"  Running {N_PERM} random permutations ...")
for i in range(N_PERM):
    shuffled = []
    for ticker in real_signal_tickers:
        if ticker not in oot_dates_per_ticker:
            continue
        valid_dates = oot_dates_per_ticker[ticker]
        rand_date = valid_dates[np.random.randint(0, len(valid_dates))]
        ind = indicators[ticker]
        ep = ind.loc[rand_date, "close"]
        if isinstance(ep, pd.Series):
            ep = ep.iloc[0]
        shuffled.append((rand_date, ticker, float(ep), "random", 10))
    perm_ret, perm_eq_curve, perm_trades = run_backtest(shuffled)
    if len(perm_ret) >= 2:
        pm = calc_metrics(perm_ret, [(OOT_START, CAPITAL)], perm_trades)
        perm_sharpes[i] = pm["sharpe"]
    else:
        perm_sharpes[i] = 0.0

    if (i + 1) % 200 == 0:
        print(f"    ... {i+1}/{N_PERM} done")

percentile = float(np.mean(perm_sharpes < real_sharpe) * 100)
p_value = 1.0 - percentile / 100.0
check2_pass = p_value < 0.05
print(f"  Real Sharpe {real_sharpe:.3f} is at {percentile:.1f}th percentile")
print(f"  p-value: {p_value:.4f} (need < 0.05)")
print(f"  Random Sharpe: mean={np.mean(perm_sharpes):.3f}, "
      f"std={np.std(perm_sharpes):.3f}, max={np.max(perm_sharpes):.3f}")
print(f"  CHECK 2: {'PASS' if check2_pass else 'FAIL'}")

# ============================================================================
#  CHECK 3: SUB-PERIOD STABILITY (4 equal sub-periods)
# ============================================================================
print("\n" + "=" * 70)
print("  CHECK 3: SUB-PERIOD STABILITY (4 equal sub-periods)")
print("=" * 70)

oot_start_dt = pd.Timestamp("2022-01-01")
oot_end_dt = pd.Timestamp("2026-07-31")
total_days = (oot_end_dt - oot_start_dt).days
period_days = total_days // 4

sub_periods = []
for i in range(4):
    p_start = oot_start_dt + pd.Timedelta(days=i * period_days)
    if i < 3:
        p_end = oot_start_dt + pd.Timedelta(days=(i + 1) * period_days - 1)
    else:
        p_end = oot_end_dt
    sub_periods.append((str(p_start.date()), str(p_end.date())))

sub_period_results = []
positive_periods = 0

for idx, (ps, pe) in enumerate(sub_periods):
    sp_returns, sp_eq, sp_trades = run_backtest(real_signals, oot_start=ps, oot_end=pe)
    sp_metrics = calc_metrics(sp_returns, sp_eq, sp_trades)
    sub_period_results.append({
        "period": f"P{idx+1}: {ps} to {pe}",
        "sharpe": sp_metrics["sharpe"],
        "n_trades": sp_metrics["n_trades"],
        "win_rate": sp_metrics["win_rate"],
        "total_return_pct": sp_metrics["total_return_pct"],
    })
    if sp_metrics["sharpe"] > 0:
        positive_periods += 1
    print(f"  P{idx+1} ({ps} to {pe}): Sharpe {sp_metrics['sharpe']:.3f}, "
          f"{sp_metrics['n_trades']} trades, WR {sp_metrics['win_rate']:.1%}, "
          f"Return {sp_metrics['total_return_pct']:.1f}%")

check3_pass = positive_periods == 4
print(f"  Positive Sharpe periods: {positive_periods}/4 (need 4/4 to pass)")
print(f"  CHECK 3: {'PASS' if check3_pass else 'FAIL'}")

# ============================================================================
#  CHECK 4: REMOVE TOP 3 TICKERS
# ============================================================================
print("\n" + "=" * 70)
print("  CHECK 4: REMOVE TOP 3 TICKERS BY PNL")
print("=" * 70)

# Find top 3 tickers by total PnL
ticker_pnl = {}
for t in real_trades:
    ticker_pnl[t["ticker"]] = ticker_pnl.get(t["ticker"], 0) + t["pnl"]

sorted_tickers = sorted(ticker_pnl.items(), key=lambda x: x[1], reverse=True)
top3 = [t[0] for t in sorted_tickers[:3]]
print(f"  Top 3 tickers by PnL: {top3}")
for t, pnl in sorted_tickers[:3]:
    print(f"    {t}: ${pnl:.2f}")

remaining_tickers = [t for t in TICKERS if t not in top3]
rem_signals = gen_signals_L(remaining_tickers)
rem_returns, rem_eq, rem_trades = run_backtest(rem_signals)
rem_metrics = calc_metrics(rem_returns, rem_eq, rem_trades)
rem_sharpe = rem_metrics["sharpe"]

sharpe_drop = (real_sharpe - rem_sharpe) / abs(real_sharpe) * 100 if abs(real_sharpe) > 0 else 0
check4_pass = sharpe_drop < 50.0
print(f"  Original Sharpe: {real_sharpe:.3f}")
print(f"  Without top 3: Sharpe {rem_sharpe:.3f} ({rem_metrics['n_trades']} trades)")
print(f"  Sharpe drop: {sharpe_drop:.1f}% (need < 50% to pass)")
print(f"  CHECK 4: {'PASS' if check4_pass else 'FAIL'}")

# ============================================================================
#  CHECK 5: PARAMETER SENSITIVITY (full sweep)
# ============================================================================
print("\n" + "=" * 70)
print("  CHECK 5: PARAMETER SENSITIVITY (sweep)")
print("=" * 70)

dip_thresholds = [0.03, 0.05, 0.07, 0.10]
rsi_thresholds = [25, 30, 35, 40]
weekly_decline_options = [1, 2, 3]
hold_options = [5, 10, 15, 20]

total_combos = len(dip_thresholds) * len(rsi_thresholds) * len(weekly_decline_options) * len(hold_options)
above_03 = 0
combo_count = 0
all_combo_results = []

print(f"  Testing {total_combos} parameter combinations ...")
for dip in dip_thresholds:
    for rsi in rsi_thresholds:
        for wdw in weekly_decline_options:
            for hold in hold_options:
                sigs = gen_signals_L(TICKERS, dip_thresh=-dip, rsi_thresh=rsi,
                                     weekly_decline_weeks=wdw, hold_days=hold)
                rets, eq, trades = run_backtest(sigs)
                m = calc_metrics(rets, eq, trades)
                combo_count += 1
                if m["sharpe"] > 0.3:
                    above_03 += 1
                # Track the original config
                is_original = (dip == 0.05 and rsi == 35 and wdw == 2 and hold == 10)
                all_combo_results.append({
                    "dip_pct": dip, "rsi_thresh": rsi,
                    "weekly_decline_weeks": wdw, "hold_days": hold,
                    "sharpe": m["sharpe"], "n_trades": m["n_trades"],
                    "win_rate": m["win_rate"], "pf": m["profit_factor"],
                    "is_original": is_original,
                })

    print(f"    dip={dip}: {combo_count}/{total_combos} done")

pct_above = above_03 / total_combos * 100
check5_pass = pct_above >= 30  # At least 30% of combos profitable
print(f"  Combinations with Sharpe > 0.3: {above_03}/{total_combos} ({pct_above:.1f}%)")
print(f"  CHECK 5: {'PASS' if check5_pass else 'FAIL'} (need >= 30% with Sharpe > 0.3)")

# Top 5 combos
top5 = sorted(all_combo_results, key=lambda x: x["sharpe"], reverse=True)[:5]
print("  Top 5 combos:")
for c in top5:
    orig = " [ORIGINAL]" if c.get("is_original") else ""
    print(f"    dip={c['dip_pct']}, rsi={c['rsi_thresh']}, wdw={c['weekly_decline_weeks']}, "
          f"hold={c['hold_days']}: Sharpe {c['sharpe']:.3f}, {c['n_trades']} trades{orig}")

# ============================================================================
#  CHECK 6: COST SENSITIVITY
# ============================================================================
print("\n" + "=" * 70)
print("  CHECK 6: COST SENSITIVITY")
print("=" * 70)

cost_levels_bps = [5, 10, 20, 50]
cost_results = {}
breakeven_bps = None

for bps in cost_levels_bps:
    slip = bps / 10000.0
    c_returns, c_eq, c_trades = run_backtest(real_signals, slippage=slip)
    c_metrics = calc_metrics(c_returns, c_eq, c_trades)
    cost_results[bps] = c_metrics
    print(f"  {bps} bps: Sharpe {c_metrics['sharpe']:.3f}, "
          f"WR {c_metrics['win_rate']:.1%}, PF {c_metrics['profit_factor']:.2f}, "
          f"Return {c_metrics['total_return_pct']:.1f}%")

# Find breakeven by binary search
lo_bps, hi_bps = 0, 200
for _ in range(30):
    mid = (lo_bps + hi_bps) / 2
    slip = mid / 10000.0
    br_ret, br_eq, br_trades = run_backtest(real_signals, slippage=slip)
    if len(br_ret) >= 2:
        br_m = calc_metrics(br_ret, br_eq, br_trades)
        if br_m["total_return_pct"] > 0:
            lo_bps = mid
        else:
            hi_bps = mid
    else:
        hi_bps = mid
breakeven_bps = round((lo_bps + hi_bps) / 2, 1)

check6_pass = breakeven_bps >= 20  # Must survive at least 20 bps
print(f"  Breakeven slippage: ~{breakeven_bps} bps")
print(f"  CHECK 6: {'PASS' if check6_pass else 'FAIL'} (need breakeven >= 20 bps)")

# ============================================================================
#  SUMMARY
# ============================================================================
checks = {
    "1_inverse_signal": {
        "description": "Buy near highs, high RSI, no red streaks, weekly RSI rising 2+ weeks",
        "passed": bool(check1_pass),
        "key_metric": f"Inverse Sharpe = {inv_sharpe:.3f}, ratio = {inv_ratio:.3f} (need < 0.50)",
        "detail": {
            "inverse_sharpe": inv_sharpe,
            "inverse_trades": inv_metrics["n_trades"],
            "inverse_wr": inv_metrics["win_rate"],
            "inverse_pf": inv_metrics["profit_factor"],
            "ratio_to_real": round(inv_ratio, 3),
        }
    },
    "2_random_timing": {
        "description": "1000 random entry sets percentile rank",
        "passed": bool(check2_pass),
        "key_metric": f"Percentile = {percentile:.1f}th, p-value = {p_value:.4f} (need p < 0.05)",
        "detail": {
            "percentile": percentile,
            "p_value": round(p_value, 4),
            "real_sharpe": real_sharpe,
            "random_mean_sharpe": round(float(np.mean(perm_sharpes)), 3),
            "random_std_sharpe": round(float(np.std(perm_sharpes)), 3),
            "random_max_sharpe": round(float(np.max(perm_sharpes)), 3),
        }
    },
    "3_sub_period_stability": {
        "description": "All 4 sub-periods must have positive Sharpe",
        "passed": bool(check3_pass),
        "key_metric": f"Positive periods = {positive_periods}/4 (need 4/4)",
        "detail": sub_period_results,
    },
    "4_remove_top3_tickers": {
        "description": "Remove top 3 PnL tickers, Sharpe drop < 50%",
        "passed": bool(check4_pass),
        "key_metric": f"Sharpe drop = {sharpe_drop:.1f}% (need < 50%)",
        "detail": {
            "top3_tickers": top3,
            "top3_pnl": {t: round(p, 2) for t, p in sorted_tickers[:3]},
            "original_sharpe": real_sharpe,
            "reduced_sharpe": rem_sharpe,
            "sharpe_drop_pct": round(sharpe_drop, 1),
            "reduced_trades": rem_metrics["n_trades"],
        }
    },
    "5_parameter_sensitivity": {
        "description": "Sweep dip/RSI/weekly-decline/hold — % with Sharpe > 0.3",
        "passed": bool(check5_pass),
        "key_metric": f"{above_03}/{total_combos} combos ({pct_above:.1f}%) have Sharpe > 0.3 (need >= 30%)",
        "detail": {
            "total_combinations": total_combos,
            "above_threshold": above_03,
            "pct_above": round(pct_above, 1),
            "top5_combos": top5,
        }
    },
    "6_cost_sensitivity": {
        "description": "Test at 5/10/20/50 bps slippage, find breakeven",
        "passed": bool(check6_pass),
        "key_metric": f"Breakeven = ~{breakeven_bps} bps (need >= 20 bps)",
        "detail": {
            f"{bps}_bps": {
                "sharpe": m["sharpe"],
                "wr": m["win_rate"],
                "pf": m["profit_factor"],
                "return_pct": m["total_return_pct"],
            } for bps, m in cost_results.items()
        } | {"breakeven_bps": breakeven_bps},
    },
}

n_passed = sum(1 for c in checks.values() if c["passed"])

if n_passed >= 5:
    recommendation = "VALIDATED"
elif n_passed >= 4:
    recommendation = "NEAR-MISS"
else:
    recommendation = "DEAD"

# -- Print Final Summary ----------------------------------------------------
print("\n" + "=" * 70)
print("  ADVERSARIAL VALIDATION SUMMARY")
print("  Multi-TF MR Variant L: Dual Signal D + Weekly RSI Declining 2+ Weeks")
print("=" * 70)
print(f"  Baseline: Sharpe {real_sharpe:.3f}, {real_metrics['n_trades']} trades, "
      f"WR {real_metrics['win_rate']:.1%}, PF {real_metrics['profit_factor']:.2f}, "
      f"MDD {real_metrics['max_dd_pct']:.2f}%")
print(f"  Regime: Bull Sharpe {real_metrics['bull_sharpe']:.3f}, "
      f"Bear Sharpe {real_metrics['bear_sharpe']:.3f}, Gap {real_metrics['regime_gap']:.3f}")
print()

for key, check in checks.items():
    status = "PASS" if check["passed"] else "FAIL"
    print(f"  [{status}] Check {key}: {check['key_metric']}")

print(f"\n  RESULT: {n_passed}/6 checks passed")
print(f"  RECOMMENDATION: {recommendation}")
print("=" * 70)

# -- Save Results ------------------------------------------------------------
output = {
    "strategy": "Multi-TF MR Variant L: Dual Signal D + Weekly RSI Declining 2+ Weeks",
    "description": "Mean-reversion with daily dip/RSI/green-after-reds + weekly RSI declining filter",
    "universe": TICKERS,
    "parameters": {
        "dip_threshold_pct": 5.0,
        "rsi_threshold": 35,
        "min_red_days": 3,
        "weekly_rsi_decline_weeks": 2,
        "hold_days": 10,
        "capital": CAPITAL,
        "max_per_trade": MAX_PER_TRADE,
        "max_concurrent": MAX_CONCURRENT,
        "slippage_bps": 2,
    },
    "baseline_metrics": real_metrics,
    "checks": {k: {kk: vv for kk, vv in v.items()} for k, v in checks.items()},
    "n_passed": n_passed,
    "n_total": 6,
    "recommendation": recommendation,
    "timestamp": datetime.now().isoformat(),
}

output_path = Path("/home/jupiter/Lvl3Quant/data/multi_tf_L_adversarial.json")
with open(output_path, "w") as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nResults saved to {output_path}")
print("Done.")

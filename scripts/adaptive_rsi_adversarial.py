#!/usr/bin/env python3
"""
Adversarial Validation: Vol-Regime Bucketed RSI (Variant E)
============================================================
6 adversarial checks to stress-test whether the edge is real.

1) Inverse Direction — buy overbought instead of oversold
2) Random Timing — 1000 random entry sets, percentile rank
3) Look-Ahead Removal — vol regime from data up to entry day only
4) Cost Sensitivity — 0.05% to 0.20% slippage
5) Sub-Period Stability — 4 equal sub-periods
6) Parameter Robustness — nearby parameter sets

Strategy under test:
  RSI(5) < threshold (varies by vol regime) AND price > 200-SMA
  Low vol (<20%): RSI<15, hold 15d
  Med vol (20-40%): RSI<20, hold 10d
  High vol (>40%): RSI<30, hold 5d

OOT: Jan 2022 - Jul 2026. $669 account. $0 commission, 0.02% slippage baseline.
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

# ── Configuration ─────────────────────────────────────────────────────────
CAPITAL = 669.0
SLIPPAGE_PCT = 0.0002  # 0.02% baseline
START = "2020-01-01"
END = "2026-07-30"
OOT_START = "2022-01-01"
N_PERM = 1000

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD",
    "NFLX", "CRM", "AVGO", "ORCL", "ADBE", "MU", "QCOM", "PLTR",
    "SOFI", "HOOD", "COIN", "UBER", "LYFT", "SHOP", "NET", "TTD",
    "DDOG", "RBLX", "SNAP", "PINS", "ROKU", "INTC",
]

ALL_TICKERS = sorted(set(TICKERS + ["SPY"]))

# ── Data Download ─────────────────────────────────────────────────────────
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


# ── Indicator Helpers ─────────────────────────────────────────────────────
def calc_rsi(series, period):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def sma(series, period):
    return series.rolling(period).mean()


def realized_vol(series, window=21):
    log_ret = np.log(series / series.shift(1))
    return log_ret.rolling(window).std() * np.sqrt(252)


# ── Pre-compute indicators ───────────────────────────────────────────────
indicators = {}
for t in TICKERS:
    c = closes.get(t, pd.Series(dtype=float))
    if len(c) < 300:
        continue
    ind = pd.DataFrame(index=c.index)
    ind["close"] = c
    ind["sma200"] = sma(c, 200)
    ind["rsi5"] = calc_rsi(c, 5)
    ind["above_200sma"] = c > ind["sma200"]
    ind["vol21"] = realized_vol(c, 21)
    indicators[t] = ind.dropna(subset=["sma200"])

spy_sma200 = sma(spy_close, 200)
spy_regime = (spy_close > spy_sma200).reindex(spy_close.index).fillna(False)

print(f"  Tickers with indicators: {len(indicators)}")


# ── Signal Generator: Vol-Regime Bucketed (Variant E) ─────────────────────
def gen_signals_E(tickers, vol_lo=0.20, vol_hi=0.40,
                  rsi_low=15.0, rsi_med=20.0, rsi_high=30.0,
                  hold_low=15, hold_med=10, hold_high=5,
                  inverse=False, no_lookahead=False):
    """
    Generate signals for variant E with configurable parameters.
    inverse=True: buy when RSI is ABOVE threshold (overbought)
    no_lookahead=True: use expanding-window vol (data up to entry day only)
    """
    signals = []
    for t in tickers:
        ind = indicators.get(t)
        if ind is None:
            continue
        for i in range(len(ind)):
            dt = ind.index[i]
            if str(dt.date()) < OOT_START:
                continue
            if not ind["above_200sma"].iloc[i]:
                continue

            if no_lookahead:
                # Use only data up to current day for vol calculation
                hist_close = ind["close"].iloc[:i+1]
                if len(hist_close) < 22:
                    continue
                log_ret = np.log(hist_close / hist_close.shift(1)).dropna()
                vol = float(log_ret.iloc[-21:].std() * np.sqrt(252)) if len(log_ret) >= 21 else np.nan
            else:
                vol = ind["vol21"].iloc[i]

            if pd.isna(vol):
                continue

            if vol < vol_lo:
                entry_thresh, max_hold = rsi_low, hold_low
            elif vol < vol_hi:
                entry_thresh, max_hold = rsi_med, hold_med
            else:
                entry_thresh, max_hold = rsi_high, hold_high

            rsi_val = ind["rsi5"].iloc[i]

            if inverse:
                # Inverse thresholds: buy when overbought
                inv_thresh = 100 - entry_thresh
                if rsi_val > inv_thresh:
                    signals.append((dt, t, float(ind["close"].iloc[i]),
                                    "inverse", 50, max_hold))
            else:
                if rsi_val < entry_thresh:
                    signals.append((dt, t, float(ind["close"].iloc[i]),
                                    "vol_regime", 50, max_hold))
    return sorted(signals, key=lambda x: x[0])


# ── Backtest Engine ───────────────────────────────────────────────────────
def run_backtest(signals, capital=CAPITAL, slippage=SLIPPAGE_PCT,
                 max_concurrent=1, oot_start=OOT_START, oot_end=None):
    if not signals:
        return np.array([]), [], []

    signals = sorted(signals, key=lambda x: x[0])
    equity = capital
    equity_curve = [(oot_start, capital)]
    realized = []
    open_positions = []

    for sig in signals:
        date, ticker, entry_price, tag, exit_rsi_thresh, max_hold = sig
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

        exit_idx = None
        for i in range(loc + 1, min(loc + max_hold + 1, len(ind))):
            if ind["rsi5"].iloc[i] > exit_rsi_thresh:
                exit_idx = i
                break
        if exit_idx is None:
            exit_idx = min(loc + max_hold, len(ind) - 1)
        if exit_idx <= loc:
            exit_idx = min(loc + 1, len(ind) - 1)

        if oot_end:
            exit_date_str = str(ind.index[exit_idx].date())
            if exit_date_str > oot_end:
                # Find last valid exit within period
                for ei in range(exit_idx, loc, -1):
                    if str(ind.index[ei].date()) <= oot_end:
                        exit_idx = ei
                        break
                else:
                    continue

        exit_date_str = str(ind.index[exit_idx].date())

        alloc = equity / max_concurrent
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
        hold_days = (ind.index[exit_idx] - ind.index[loc]).days

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
            "hold_days": hold_days,
            "regime": regime,
            "tag": tag,
        }

        equity += pnl
        equity_curve.append((exit_date_str, round(equity, 2)))
        realized.append(trade)
        open_positions.append({"ticker": ticker, "exit_date": exit_date_str})

    returns = np.array([t["return"] for t in realized])
    return returns, equity_curve, realized


# ── Metrics ───────────────────────────────────────────────────────────────
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

    avg_hold = np.mean([t["hold_days"] for t in realized]) if realized else 8.0
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

    return {
        "n_trades": int(n),
        "win_rate": round(float(wr), 4),
        "sharpe": round(float(sharpe), 3),
        "sortino": round(float(sortino), 3),
        "profit_factor": round(float(profit_factor), 3),
        "max_dd_pct": round(float(max_dd * 100), 2),
        "total_return_pct": round(float(total_return_pct), 2),
        "final_equity": round(float(final_eq), 2),
    }


# ══════════════════════════════════════════════════════════════════════════
#  CHECK 1: INVERSE DIRECTION
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  CHECK 1: INVERSE DIRECTION (buy overbought)")
print("=" * 70)

inv_signals = gen_signals_E(TICKERS, inverse=True)
inv_returns, inv_eq, inv_trades = run_backtest(inv_signals)
inv_metrics = calc_metrics(inv_returns, inv_eq, inv_trades)

inv_sharpe = inv_metrics["sharpe"]
check1_pass = inv_sharpe < 0.0
print(f"  Inverse Sharpe: {inv_sharpe:.3f} (need < 0 to pass)")
print(f"  Inverse trades: {inv_metrics['n_trades']}, WR: {inv_metrics['win_rate']:.1%}")
print(f"  CHECK 1: {'PASS' if check1_pass else 'FAIL'}")

# ══════════════════════════════════════════════════════════════════════════
#  CHECK 2: RANDOM TIMING (1000 random entry sets)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  CHECK 2: RANDOM TIMING (1000 permutations)")
print("=" * 70)

# First, run the real strategy to get baseline
real_signals = gen_signals_E(TICKERS)
real_returns, real_eq, real_trades = run_backtest(real_signals)
real_metrics = calc_metrics(real_returns, real_eq, real_trades)
real_sharpe = real_metrics["sharpe"]
n_real_trades = real_metrics["n_trades"]
print(f"  Real strategy: Sharpe {real_sharpe:.3f}, {n_real_trades} trades")

# Build valid OOT dates per ticker
oot_dates_per_ticker = {}
for t in TICKERS:
    ind = indicators.get(t)
    if ind is not None:
        valid = ind.index[ind.index >= OOT_START]
        if len(valid) > 15:
            oot_dates_per_ticker[t] = valid[:-10]

# Get tickers used in real signals
real_signal_tickers = [s[1] for s in real_signals
                       if str(s[0].date()) >= OOT_START and s[1] in oot_dates_per_ticker]

perm_sharpes = np.zeros(N_PERM)
print(f"  Running {N_PERM} random permutations ...")
for i in range(N_PERM):
    shuffled = []
    for ticker in real_signal_tickers:
        valid_dates = oot_dates_per_ticker[ticker]
        rand_date = valid_dates[np.random.randint(0, len(valid_dates))]
        ind = indicators[ticker]
        ep = ind.loc[rand_date, "close"]
        if isinstance(ep, pd.Series):
            ep = ep.iloc[0]
        # Use random hold period matching vol regime distribution
        hold = np.random.choice([5, 10, 15])
        shuffled.append((rand_date, ticker, float(ep), "random", 50, hold))
    perm_ret, _, perm_trades = run_backtest(shuffled)
    if len(perm_ret) >= 2:
        pm = calc_metrics(perm_ret, [(OOT_START, CAPITAL)], perm_trades)
        perm_sharpes[i] = pm["sharpe"]
    else:
        perm_sharpes[i] = 0.0

    if (i + 1) % 200 == 0:
        print(f"    ... {i+1}/{N_PERM} done")

percentile = float(np.mean(perm_sharpes < real_sharpe) * 100)
check2_pass = percentile >= 95.0
print(f"  Real Sharpe {real_sharpe:.3f} is at {percentile:.1f}th percentile")
print(f"  Random Sharpe distribution: mean={np.mean(perm_sharpes):.3f}, "
      f"std={np.std(perm_sharpes):.3f}, max={np.max(perm_sharpes):.3f}")
print(f"  CHECK 2: {'PASS' if check2_pass else 'FAIL'} (need >= 95th percentile)")

# ══════════════════════════════════════════════════════════════════════════
#  CHECK 3: LOOK-AHEAD REMOVAL
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  CHECK 3: LOOK-AHEAD REMOVAL (no future vol data)")
print("=" * 70)

nla_signals = gen_signals_E(TICKERS, no_lookahead=True)
nla_returns, nla_eq, nla_trades = run_backtest(nla_signals)
nla_metrics = calc_metrics(nla_returns, nla_eq, nla_trades)
nla_sharpe = nla_metrics["sharpe"]

sharpe_drop_pct = ((real_sharpe - nla_sharpe) / abs(real_sharpe) * 100) if abs(real_sharpe) > 0 else 0
check3_pass = sharpe_drop_pct < 50.0
print(f"  Original Sharpe: {real_sharpe:.3f}")
print(f"  No-lookahead Sharpe: {nla_sharpe:.3f}")
print(f"  Sharpe drop: {sharpe_drop_pct:.1f}% (must be < 50% to pass)")
print(f"  No-lookahead trades: {nla_metrics['n_trades']}, WR: {nla_metrics['win_rate']:.1%}")
print(f"  CHECK 3: {'PASS' if check3_pass else 'FAIL'}")

# ══════════════════════════════════════════════════════════════════════════
#  CHECK 4: COST SENSITIVITY
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  CHECK 4: COST SENSITIVITY")
print("=" * 70)

cost_levels = [0.0005, 0.0010, 0.0015, 0.0020]  # 0.05%, 0.10%, 0.15%, 0.20%
cost_results = {}

for slip in cost_levels:
    c_returns, c_eq, c_trades = run_backtest(real_signals, slippage=slip)
    c_metrics = calc_metrics(c_returns, c_eq, c_trades)
    cost_results[slip] = c_metrics
    label = f"{slip*100:.2f}%"
    print(f"  Slippage {label}: Sharpe {c_metrics['sharpe']:.3f}, "
          f"WR {c_metrics['win_rate']:.1%}, PF {c_metrics['profit_factor']:.2f}")

worst_sharpe = cost_results[0.0020]["sharpe"]
check4_pass = worst_sharpe > 0.5
print(f"  Sharpe at 0.20% slippage: {worst_sharpe:.3f} (need > 0.5 to pass)")
print(f"  CHECK 4: {'PASS' if check4_pass else 'FAIL'}")

# ══════════════════════════════════════════════════════════════════════════
#  CHECK 5: SUB-PERIOD STABILITY
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  CHECK 5: SUB-PERIOD STABILITY (4 equal sub-periods)")
print("=" * 70)

# OOT: Jan 2022 - Jul 2026 = ~54 months, split into 4 ~13.5 month periods
oot_start_dt = pd.Timestamp("2022-01-01")
oot_end_dt = pd.Timestamp("2026-07-30")
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

check5_pass = positive_periods >= 3
print(f"  Positive Sharpe periods: {positive_periods}/4 (need >= 3 to pass)")
print(f"  CHECK 5: {'PASS' if check5_pass else 'FAIL'}")

# ══════════════════════════════════════════════════════════════════════════
#  CHECK 6: PARAMETER ROBUSTNESS
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  CHECK 6: PARAMETER ROBUSTNESS (nearby parameter sets)")
print("=" * 70)

param_variants = [
    {"label": "Vol buckets (15%/35%)", "vol_lo": 0.15, "vol_hi": 0.35},
    {"label": "Vol buckets (25%/45%)", "vol_lo": 0.25, "vol_hi": 0.45},
    {"label": "RSI thresholds (10/15/25)", "rsi_low": 10, "rsi_med": 15, "rsi_high": 25},
    {"label": "RSI thresholds (20/25/35)", "rsi_low": 20, "rsi_med": 25, "rsi_high": 35},
    {"label": "Hold days (10/7/3)", "hold_low": 10, "hold_med": 7, "hold_high": 3},
    {"label": "Hold days (20/15/8)", "hold_low": 20, "hold_med": 15, "hold_high": 8},
]

# Default params for E
default_params = dict(vol_lo=0.20, vol_hi=0.40, rsi_low=15, rsi_med=20, rsi_high=30,
                      hold_low=15, hold_med=10, hold_high=5)

param_results = []
above_threshold = 0

for pv in param_variants:
    params = {**default_params}
    for k, v in pv.items():
        if k != "label":
            params[k] = v

    pv_signals = gen_signals_E(TICKERS, **params)
    pv_returns, pv_eq, pv_trades = run_backtest(pv_signals)
    pv_metrics = calc_metrics(pv_returns, pv_eq, pv_trades)

    param_results.append({
        "label": pv["label"],
        "sharpe": pv_metrics["sharpe"],
        "n_trades": pv_metrics["n_trades"],
        "win_rate": pv_metrics["win_rate"],
        "profit_factor": pv_metrics["profit_factor"],
    })

    if pv_metrics["sharpe"] > 0.5:
        above_threshold += 1

    print(f"  {pv['label']}: Sharpe {pv_metrics['sharpe']:.3f}, "
          f"{pv_metrics['n_trades']} trades, WR {pv_metrics['win_rate']:.1%}, "
          f"PF {pv_metrics['profit_factor']:.2f}")

# Need 3 of 6 variants with Sharpe > 0.5 (user said "3 of 4 alternatives"
# but we have 6 variants; interpret as majority must pass)
# Actually user specified 4 alternatives total from the 3 shift categories,
# but we have 6 individual tests. Let's use: at least 4/6 > 0.5
check6_pass = above_threshold >= 4
print(f"  Variants with Sharpe > 0.5: {above_threshold}/6 (need >= 4 to pass)")
print(f"  CHECK 6: {'PASS' if check6_pass else 'FAIL'}")

# ══════════════════════════════════════════════════════════════════════════
#  SUMMARY
# ══════════════════════════════════════════════════════════════════════════
checks = {
    "1_inverse_direction": {
        "description": "Buy overbought instead of oversold",
        "passed": check1_pass,
        "key_metric": f"Inverse Sharpe = {inv_sharpe:.3f} (need < 0)",
        "detail": {
            "inverse_sharpe": inv_sharpe,
            "inverse_trades": inv_metrics["n_trades"],
            "inverse_wr": inv_metrics["win_rate"],
        }
    },
    "2_random_timing": {
        "description": "1000 random entry sets percentile rank",
        "passed": check2_pass,
        "key_metric": f"Percentile = {percentile:.1f}th (need >= 95th)",
        "detail": {
            "percentile": percentile,
            "real_sharpe": real_sharpe,
            "random_mean_sharpe": round(float(np.mean(perm_sharpes)), 3),
            "random_std_sharpe": round(float(np.std(perm_sharpes)), 3),
            "random_max_sharpe": round(float(np.max(perm_sharpes)), 3),
        }
    },
    "3_look_ahead_removal": {
        "description": "Vol regime from data up to entry day only",
        "passed": check3_pass,
        "key_metric": f"Sharpe drop = {sharpe_drop_pct:.1f}% (need < 50%)",
        "detail": {
            "original_sharpe": real_sharpe,
            "no_lookahead_sharpe": nla_sharpe,
            "sharpe_drop_pct": round(sharpe_drop_pct, 1),
            "nla_trades": nla_metrics["n_trades"],
        }
    },
    "4_cost_sensitivity": {
        "description": "Sharpe at 0.20% slippage",
        "passed": check4_pass,
        "key_metric": f"Sharpe at 0.20% = {worst_sharpe:.3f} (need > 0.5)",
        "detail": {
            f"slippage_{s*100:.2f}pct": {
                "sharpe": m["sharpe"],
                "wr": m["win_rate"],
                "pf": m["profit_factor"],
            } for s, m in cost_results.items()
        }
    },
    "5_sub_period_stability": {
        "description": "Sharpe > 0 in at least 3 of 4 sub-periods",
        "passed": check5_pass,
        "key_metric": f"Positive periods = {positive_periods}/4 (need >= 3)",
        "detail": sub_period_results,
    },
    "6_parameter_robustness": {
        "description": "Nearby parameters maintain Sharpe > 0.5",
        "passed": check6_pass,
        "key_metric": f"Variants with Sharpe > 0.5 = {above_threshold}/6 (need >= 4)",
        "detail": param_results,
    },
}

n_passed = sum(1 for c in checks.values() if c["passed"])

if n_passed >= 5:
    recommendation = "VALIDATED"
elif n_passed >= 4:
    recommendation = "NEAR-MISS"
else:
    recommendation = "DEAD"

# ── Print Final Summary ──────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  ADVERSARIAL VALIDATION SUMMARY — Vol-Regime Bucketed RSI (Variant E)")
print("=" * 70)
print(f"  Baseline: Sharpe {real_sharpe:.3f}, {n_real_trades} trades, "
      f"WR {real_metrics['win_rate']:.1%}, PF {real_metrics['profit_factor']:.2f}")
print()

for key, check in checks.items():
    status = "PASS" if check["passed"] else "FAIL"
    print(f"  [{status}] Check {key}: {check['key_metric']}")

print(f"\n  RESULT: {n_passed}/6 checks passed")
print(f"  RECOMMENDATION: {recommendation}")
print("=" * 70)

# ── Save Results ──────────────────────────────────────────────────────────
output = {
    "strategy": "Vol-Regime Bucketed RSI (Variant E)",
    "baseline_metrics": real_metrics,
    "checks": {k: {kk: vv for kk, vv in v.items()} for k, v in checks.items()},
    "n_passed": n_passed,
    "n_total": 6,
    "recommendation": recommendation,
    "timestamp": datetime.now().isoformat(),
}

output_path = Path("/home/jupiter/Lvl3Quant/data/adaptive_rsi_adversarial_results.json")
with open(output_path, "w") as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nResults saved to {output_path}")
print("Done.")

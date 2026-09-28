#!/usr/bin/env python3
"""
Put/Call Ratio Contrarian Strategy Backtest
Academic basis: Pan & Poteshman (2006) — options volume predicts stock returns.
Uses VIX and realized vol as proxies for put/call sentiment extremes.

6 Variants:
  A. VIX Spike Buy
  B. VIX Mean Reversion
  C. Fear-to-Greed Rotation
  D. Volatility Compression Breakout
  E. Multi-Timeframe Vol (backwardation proxy)
  F. Combined Vol Signal

OOT: Jan 2022 – Jul 2026 | Capital: $645 | Costs: 0 commission, 0.02% slippage
5-Gate validation per variant.
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────
GROWTH = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD", "CRM", "ADBE", "NFLX", "AVGO"]
SAFE = ["GLD", "TLT"]
INDEX = ["SPY", "QQQ"]
ALL_TICKERS = list(set(INDEX + GROWTH + SAFE + ["^VIX"]))

OOT_START = "2022-01-01"
OOT_END = "2026-07-30"
CAPITAL = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02%
PERMUTATION_ITERS = 1000
SMA_PERIOD = 200

# ── Data Download ──────────────────────────────────────────────────────────
print("Downloading data...")
# Download with extra lookback for SMA/vol calculations
data = yf.download(ALL_TICKERS, start="2021-01-01", end=OOT_END, auto_adjust=True, progress=False)

# Handle multi-level columns from yfinance
close = data["Close"].copy()
# Rename ^VIX column
if "^VIX" in close.columns:
    close.rename(columns={"^VIX": "VIX"}, inplace=True)

close = close.ffill().dropna(how="all")

# Ensure VIX exists
if "VIX" not in close.columns:
    print("ERROR: VIX data not available. Exiting.")
    exit(1)

print(f"Data shape: {close.shape}, date range: {close.index[0].date()} to {close.index[-1].date()}")

# ── Precompute Signals ─────────────────────────────────────────────────────
vix = close["VIX"]
spy = close["SPY"]
spy_sma200 = spy.rolling(SMA_PERIOD).mean()
spy_ret = spy.pct_change()

# Realized vol (annualized, 5-day)
spy_rvol_5d = spy_ret.rolling(5).std() * np.sqrt(252)
# 1-month and 3-month realized vol (for term structure proxy)
spy_rvol_21d = spy_ret.rolling(21).std() * np.sqrt(252)
spy_rvol_63d = spy_ret.rolling(63).std() * np.sqrt(252)

vix_pct_change = vix.pct_change()
vix_60d_mean = vix.rolling(60).mean()
vix_60d_std = vix.rolling(60).std()

# Regime: Bull = SPY > 200-SMA, Bear = SPY < 200-SMA
regime = pd.Series("bull", index=spy.index)
regime[spy < spy_sma200] = "bear"

# Restrict to OOT period for trading
oot_mask = close.index >= OOT_START
oot_dates = close.index[oot_mask]


# ── Helper Functions ───────────────────────────────────────────────────────
def apply_slippage(price, direction="buy"):
    """Apply slippage to execution price."""
    if direction == "buy":
        return price * (1 + SLIPPAGE_PCT)
    return price * (1 - SLIPPAGE_PCT)


def equal_weight_portfolio(tickers, date, capital, close_df):
    """Buy equal-weight portfolio of tickers, return shares dict."""
    available = [t for t in tickers if t in close_df.columns and not np.isnan(close_df.loc[date, t])]
    if not available:
        return {}
    per_stock = capital / len(available)
    shares = {}
    for t in available:
        price = apply_slippage(close_df.loc[date, t], "buy")
        if price > 0:
            shares[t] = per_stock / price
    return shares


def portfolio_value(shares, date, close_df):
    """Calculate portfolio value at a given date."""
    val = 0.0
    for t, s in shares.items():
        if t in close_df.columns:
            p = close_df.loc[date, t]
            if not np.isnan(p):
                val += s * apply_slippage(p, "sell")
    return val


def run_backtest_trades(trade_list, close_df, capital):
    """
    Given a list of trades [(entry_date, exit_date, tickers), ...],
    simulate equal-weight entries and exits, returning daily equity curve.
    Trades can overlap. Each trade gets allocated from available cash.
    """
    all_dates = close_df.loc[OOT_START:].index
    equity = pd.Series(0.0, index=all_dates)
    cash = capital
    active_trades = []  # (shares_dict, exit_date, entry_cost)
    trade_returns = []

    for i, date in enumerate(all_dates):
        # Close expired trades
        still_active = []
        for shares, exit_d, entry_cost in active_trades:
            if date >= exit_d:
                proceeds = portfolio_value(shares, date, close_df)
                trade_returns.append(proceeds / entry_cost - 1 if entry_cost > 0 else 0)
                cash += proceeds
            else:
                still_active.append((shares, exit_d, entry_cost))
        active_trades = still_active

        # Open new trades
        new_trades = [t for t in trade_list if t[0] == date]
        for entry_d, exit_d, tickers in new_trades:
            if cash > 1.0:  # need at least $1
                alloc = min(cash, capital * 0.5)  # max 50% per trade
                shares = equal_weight_portfolio(tickers, date, alloc, close_df)
                if shares:
                    cost = sum(s * apply_slippage(close_df.loc[date, t], "buy")
                               for t, s in shares.items())
                    cash -= cost
                    active_trades.append((shares, exit_d, cost))

        # Mark-to-market
        active_val = sum(portfolio_value(sh, date, close_df) for sh, _, _ in active_trades)
        equity.iloc[i] = cash + active_val

    return equity, trade_returns


def compute_metrics(equity, trade_returns, variant_name):
    """Compute performance metrics for a variant."""
    daily_ret = equity.pct_change().dropna()
    daily_ret = daily_ret.replace([np.inf, -np.inf], 0).fillna(0)

    n_trades = len(trade_returns)
    if n_trades == 0 or daily_ret.std() == 0:
        return {
            "variant": variant_name, "sharpe": 0, "sortino": 0, "pf": 0,
            "win_rate": 0, "max_dd_pct": -100, "n_trades": 0,
            "total_return_pct": 0, "cagr_pct": 0,
            "perm_p_value": 1.0, "regime_gap": 1.0,
            "gate_sharpe": False, "gate_perm": False,
            "gate_regime": False, "gate_dd": False,
            "gate_trades": False, "gates_passed": 0
        }

    # Sharpe (annualized)
    sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 0

    # Sortino
    downside = daily_ret[daily_ret < 0]
    sortino = daily_ret.mean() / downside.std() * np.sqrt(252) if len(downside) > 0 and downside.std() > 0 else 0

    # Profit Factor
    wins = sum(r for r in trade_returns if r > 0)
    losses = abs(sum(r for r in trade_returns if r < 0))
    pf = wins / losses if losses > 0 else (999 if wins > 0 else 0)

    # Win Rate
    wr = sum(1 for r in trade_returns if r > 0) / n_trades if n_trades > 0 else 0

    # Max Drawdown
    cummax = equity.cummax()
    dd = (equity - cummax) / cummax
    max_dd = dd.min() * 100

    # Total Return
    total_ret = (equity.iloc[-1] / equity.iloc[0] - 1) * 100 if equity.iloc[0] > 0 else 0

    # CAGR
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = ((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1) * 100 if years > 0 and equity.iloc[0] > 0 else 0

    # Regime-stratified Sharpe
    bull_ret = daily_ret[regime.reindex(daily_ret.index) == "bull"]
    bear_ret = daily_ret[regime.reindex(daily_ret.index) == "bear"]
    sharpe_bull = bull_ret.mean() / bull_ret.std() * np.sqrt(252) if len(bull_ret) > 10 and bull_ret.std() > 0 else 0
    sharpe_bear = bear_ret.mean() / bear_ret.std() * np.sqrt(252) if len(bear_ret) > 10 and bear_ret.std() > 0 else 0
    max_sharpe = max(abs(sharpe_bull), abs(sharpe_bear))
    regime_gap = abs(sharpe_bull - sharpe_bear) / max_sharpe if max_sharpe > 0 else 0

    # Permutation test
    perm_p = permutation_test(daily_ret, PERMUTATION_ITERS)

    # 5-Gate validation
    g1 = sharpe > 0.5
    g2 = perm_p < 0.05
    g3 = regime_gap < 0.5
    g4 = max_dd > -50
    g5 = n_trades >= 20
    gates_passed = sum([g1, g2, g3, g4, g5])

    return {
        "variant": variant_name,
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "pf": round(pf, 3),
        "win_rate": round(wr * 100, 1),
        "max_dd_pct": round(max_dd, 2),
        "n_trades": n_trades,
        "total_return_pct": round(total_ret, 2),
        "cagr_pct": round(cagr, 2),
        "perm_p_value": round(perm_p, 4),
        "regime_gap": round(regime_gap, 3),
        "sharpe_bull": round(sharpe_bull, 3),
        "sharpe_bear": round(sharpe_bear, 3),
        "gate_sharpe": g1,
        "gate_perm": g2,
        "gate_regime": g3,
        "gate_dd": g4,
        "gate_trades": g5,
        "gates_passed": gates_passed
    }


def permutation_test(daily_ret, n_iter=1000):
    """Permutation test for Sharpe ratio significance."""
    if len(daily_ret) < 10 or daily_ret.std() == 0:
        return 1.0
    observed = daily_ret.mean() / daily_ret.std()
    count = 0
    arr = daily_ret.values.copy()
    rng = np.random.RandomState(42)
    for _ in range(n_iter):
        rng.shuffle(arr)
        perm_sharpe = arr.mean() / arr.std() if arr.std() > 0 else 0
        if perm_sharpe >= observed:
            count += 1
    return count / n_iter


# ── Variant A: VIX Spike Buy ──────────────────────────────────────────────
def variant_a():
    """Buy SPY when VIX rises >15% in a single day. Hold 10 days."""
    print("\n[A] VIX Spike Buy...")
    trades = []
    for date in oot_dates:
        if date not in vix_pct_change.index:
            continue
        if vix_pct_change.loc[date] > 0.15:
            exit_idx = close.index.get_loc(date) + 10
            if exit_idx < len(close.index):
                exit_date = close.index[exit_idx]
                trades.append((date, exit_date, ["SPY"]))
    equity, trade_rets = run_backtest_trades(trades, close, CAPITAL)
    return compute_metrics(equity, trade_rets, "A: VIX Spike Buy")


# ── Variant B: VIX Mean Reversion ─────────────────────────────────────────
def variant_b():
    """Buy SPY when VIX > 30, sell when VIX < 20 or after 20 days."""
    print("[B] VIX Mean Reversion...")
    trades = []
    in_trade = False
    entry_date = None
    for date in oot_dates:
        if date not in vix.index:
            continue
        v = vix.loc[date]
        if not in_trade and v > 30:
            entry_date = date
            in_trade = True
        elif in_trade:
            days_held = (date - entry_date).days
            if v < 20 or days_held >= 20:
                trades.append((entry_date, date, ["SPY"]))
                in_trade = False
    equity, trade_rets = run_backtest_trades(trades, close, CAPITAL)
    return compute_metrics(equity, trade_rets, "B: VIX Mean Reversion")


# ── Variant C: Fear-to-Greed Rotation ─────────────────────────────────────
def variant_c():
    """VIX > 25 → buy growth (contrarian). VIX < 15 → buy defensive. Weekly rebalance."""
    print("[C] Fear-to-Greed Rotation...")
    trades = []
    last_rebal = None
    for date in oot_dates:
        if date not in vix.index:
            continue
        # Weekly rebalance (every 5 trading days)
        if last_rebal is not None:
            days_since = len(close.loc[last_rebal:date]) - 1
            if days_since < 5:
                continue
        v = vix.loc[date]
        exit_idx = close.index.get_loc(date) + 5
        if exit_idx >= len(close.index):
            continue
        exit_date = close.index[exit_idx]
        if v > 25:
            # Fear regime: buy growth (contrarian)
            top3 = pick_top_growth(date, 3)
            if top3:
                trades.append((date, exit_date, top3))
                last_rebal = date
        elif v < 15:
            # Complacency: buy defensive
            trades.append((date, exit_date, SAFE))
            last_rebal = date
        else:
            last_rebal = date  # skip but reset timer
    equity, trade_rets = run_backtest_trades(trades, close, CAPITAL)
    return compute_metrics(equity, trade_rets, "C: Fear-to-Greed Rotation")


def pick_top_growth(date, n=3):
    """Pick top N growth stocks by recent momentum (20-day return)."""
    rets = {}
    for t in GROWTH:
        if t in close.columns:
            idx = close.index.get_loc(date)
            if idx >= 20:
                r = close[t].iloc[idx] / close[t].iloc[idx - 20] - 1
                if not np.isnan(r):
                    rets[t] = r
    sorted_t = sorted(rets, key=rets.get, reverse=True)
    return sorted_t[:n]


# ── Variant D: Volatility Compression Breakout ────────────────────────────
def variant_d():
    """Buy growth when 5-day realized vol < 10th percentile of 60-day window. Hold 10 days."""
    print("[D] Vol Compression Breakout...")
    trades = []
    for date in oot_dates:
        idx = close.index.get_loc(date)
        if idx < 60:
            continue
        rv5 = spy_rvol_5d.iloc[idx]
        if np.isnan(rv5):
            continue
        # 10th percentile of trailing 60 days
        window = spy_rvol_5d.iloc[idx - 60:idx].dropna()
        if len(window) < 30:
            continue
        p10 = window.quantile(0.10)
        if rv5 < p10:
            exit_idx = idx + 10
            if exit_idx < len(close.index):
                exit_date = close.index[exit_idx]
                top3 = pick_top_growth(date, 3)
                if top3:
                    trades.append((date, exit_date, top3))
    equity, trade_rets = run_backtest_trades(trades, close, CAPITAL)
    return compute_metrics(equity, trade_rets, "D: Vol Compression Breakout")


# ── Variant E: Multi-Timeframe Vol (Backwardation Proxy) ──────────────────
def variant_e():
    """Buy growth when 1m realized vol > 3m realized vol (backwardation proxy).
       Sell when contango returns."""
    print("[E] Multi-TF Vol Backwardation...")
    trades = []
    in_trade = False
    entry_date = None
    entry_tickers = None
    for date in oot_dates:
        idx = close.index.get_loc(date)
        rv1m = spy_rvol_21d.iloc[idx]
        rv3m = spy_rvol_63d.iloc[idx]
        if np.isnan(rv1m) or np.isnan(rv3m):
            continue
        backwardation = rv1m > rv3m
        if not in_trade and backwardation:
            entry_date = date
            entry_tickers = pick_top_growth(date, 3)
            in_trade = True
        elif in_trade and not backwardation:
            if entry_tickers:
                trades.append((entry_date, date, entry_tickers))
            in_trade = False
        elif in_trade:
            # Max hold 30 days
            if (date - entry_date).days >= 30:
                if entry_tickers:
                    trades.append((entry_date, date, entry_tickers))
                in_trade = False
    equity, trade_rets = run_backtest_trades(trades, close, CAPITAL)
    return compute_metrics(equity, trade_rets, "E: Multi-TF Vol Backwardation")


# ── Variant F: Combined Vol Signal ────────────────────────────────────────
def variant_f():
    """Score-based: +1 VIX spike in last 5d, +1 VIX > 1.5std above 60d mean,
       +1 5d vol declining. Buy top 3 growth when score >= 2. Hold 10 days."""
    print("[F] Combined Vol Signal...")
    trades = []
    for date in oot_dates:
        idx = close.index.get_loc(date)
        if idx < 65:
            continue
        score = 0
        # Signal 1: VIX rose >10% in last 5 days
        vix_5d = vix_pct_change.iloc[max(0, idx - 5):idx + 1]
        if (vix_5d > 0.10).any():
            score += 1
        # Signal 2: VIX > 1.5 std above 60-day mean
        v = vix.iloc[idx]
        m60 = vix_60d_mean.iloc[idx]
        s60 = vix_60d_std.iloc[idx]
        if not np.isnan(m60) and not np.isnan(s60) and s60 > 0:
            if v > m60 + 1.5 * s60:
                score += 1
        # Signal 3: 5-day realized vol declining (fear resolving)
        rv_today = spy_rvol_5d.iloc[idx]
        rv_5ago = spy_rvol_5d.iloc[idx - 5] if idx >= 5 else np.nan
        if not np.isnan(rv_today) and not np.isnan(rv_5ago):
            if rv_today < rv_5ago:
                score += 1
        if score >= 2:
            exit_idx = idx + 10
            if exit_idx < len(close.index):
                exit_date = close.index[exit_idx]
                top3 = pick_top_growth(date, 3)
                if top3:
                    trades.append((date, exit_date, top3))
    equity, trade_rets = run_backtest_trades(trades, close, CAPITAL)
    return compute_metrics(equity, trade_rets, "F: Combined Vol Signal")


# ── Run All Variants ──────────────────────────────────────────────────────
results = []
for fn in [variant_a, variant_b, variant_c, variant_d, variant_e, variant_f]:
    try:
        r = fn()
        results.append(r)
    except Exception as e:
        print(f"  ERROR in {fn.__name__}: {e}")
        import traceback; traceback.print_exc()

# ── Summary Table ──────────────────────────────────────────────────────────
print("\n" + "=" * 120)
print("PUT/CALL RATIO CONTRARIAN BACKTEST — RESULTS SUMMARY")
print(f"OOT: {OOT_START} to {OOT_END} | Capital: ${CAPITAL} | Slippage: {SLIPPAGE_PCT*100:.2f}%")
print("=" * 120)
header = f"{'Variant':<35} {'Sharpe':>7} {'Sortino':>8} {'PF':>6} {'WR%':>6} {'MaxDD%':>8} {'Trades':>7} {'TotRet%':>9} {'CAGR%':>7} {'Perm-p':>7} {'RGap':>6} {'Gates':>6}"
print(header)
print("-" * 120)
for r in results:
    gates_str = f"{r['gates_passed']}/5"
    print(f"{r['variant']:<35} {r['sharpe']:>7.3f} {r['sortino']:>8.3f} {r['pf']:>6.2f} {r['win_rate']:>5.1f}% {r['max_dd_pct']:>7.2f}% {r['n_trades']:>7} {r['total_return_pct']:>8.2f}% {r['cagr_pct']:>6.2f}% {r['perm_p_value']:>7.4f} {r['regime_gap']:>6.3f} {gates_str:>6}")
print("-" * 120)

# Gate details
print("\n5-GATE VALIDATION DETAIL:")
print(f"{'Variant':<35} {'Sharpe>0.5':>11} {'Perm<0.05':>10} {'RGap<0.5':>9} {'DD>-50%':>8} {'Trades>=20':>11} {'PASS':>6}")
print("-" * 95)
for r in results:
    def g(v): return "PASS" if v else "FAIL"
    verdict = "PASS" if r["gates_passed"] == 5 else "FAIL"
    print(f"{r['variant']:<35} {g(r['gate_sharpe']):>11} {g(r['gate_perm']):>10} {g(r['gate_regime']):>9} {g(r['gate_dd']):>8} {g(r['gate_trades']):>11} {verdict:>6}")

# Regime breakdown
print("\n\nREGIME BREAKDOWN:")
print(f"{'Variant':<35} {'Bull Sharpe':>12} {'Bear Sharpe':>12} {'Gap':>8}")
print("-" * 70)
for r in results:
    print(f"{r['variant']:<35} {r.get('sharpe_bull', 0):>12.3f} {r.get('sharpe_bear', 0):>12.3f} {r['regime_gap']:>8.3f}")

# ── Save Results ───────────────────────────────────────────────────────────
output_path = Path("/home/jupiter/Lvl3Quant/data/put_call_ratio_results.json")
output = {
    "strategy": "Put/Call Ratio Contrarian (VIX/Vol Proxy)",
    "oot_period": f"{OOT_START} to {OOT_END}",
    "capital": CAPITAL,
    "slippage_pct": SLIPPAGE_PCT,
    "run_date": datetime.now().isoformat(),
    "variants": results
}
with open(output_path, "w") as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nResults saved to {output_path}")

# Final verdict
passing = [r for r in results if r["gates_passed"] == 5]
print(f"\n{'='*60}")
print(f"VERDICT: {len(passing)}/{len(results)} variants pass all 5 gates")
if passing:
    best = max(passing, key=lambda x: x["sharpe"])
    print(f"Best passing variant: {best['variant']} (Sharpe={best['sharpe']}, Sortino={best['sortino']})")
else:
    # Show best partial
    best = max(results, key=lambda x: x["gates_passed"])
    print(f"Best partial: {best['variant']} ({best['gates_passed']}/5 gates, Sharpe={best['sharpe']})")
print(f"{'='*60}")

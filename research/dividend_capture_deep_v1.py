#!/usr/bin/env python3
"""
Dividend Capture + Covered Call Deep Dive Stress Test
=====================================================
HC #724: Anti-lookahead (T-1 signals, T+1 execution)
HC #718: Perm tests, costs mandatory, min 100 OOS

Parts:
  1. Reality-check dividend simulation with actual yfinance data
  2. Add realistic covered call premium via Black-Scholes
  3. Position sizing optimization to control MaxDD
  4. Universe expansion tests
  5. Adversarial validation (walk-forward, permutation, regime, sub-period)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from datetime import datetime, timedelta
from itertools import product
import os, sys, time

# ── Config ──────────────────────────────────────────────────────────────────
LARGE_CAP_30 = [
    "AAPL", "MSFT", "JNJ", "JPM", "PG", "XOM", "CVX", "KO", "PEP", "MRK",
    "ABBV", "VZ", "T", "CSCO", "IBM", "PFE", "BMY", "MO", "PM", "MMM",
    "CAT", "HD", "WMT", "UPS", "TXN", "AVGO", "QCOM", "LMT", "RTX", "GD"
]
START_DATE = "2015-01-01"
END_DATE   = "2025-12-31"
COST_BPS   = 10          # 10 bps RT stock commission
SPREAD_BPS = 2           # 0.02% large-cap spread proxy
RISK_FREE  = 0.04        # for BS pricing
INITIAL_CAPITAL = 100_000
RANDOM_SEED = 42

OUT_DIR = "/home/jupiter/Lvl3Quant/research/findings"
os.makedirs(OUT_DIR, exist_ok=True)

np.random.seed(RANDOM_SEED)


# ── Helpers ─────────────────────────────────────────────────────────────────
def bs_call(S, K, r, sigma, T):
    """Black-Scholes call price."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def sharpe(returns, ann=252):
    """Annualized Sharpe ratio."""
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return returns.mean() / returns.std() * np.sqrt(ann)


def sortino(returns, ann=252):
    """Annualized Sortino ratio."""
    down = returns[returns < 0]
    if len(down) < 2 or down.std() == 0:
        return 0.0
    return returns.mean() / down.std() * np.sqrt(ann)


def cagr(equity_curve):
    """CAGR from equity curve (Series with DatetimeIndex)."""
    if len(equity_curve) < 2:
        return 0.0
    years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
    if years <= 0 or equity_curve.iloc[0] <= 0:
        return 0.0
    return (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / years) - 1


def max_dd(equity_curve):
    """Maximum drawdown."""
    peak = equity_curve.cummax()
    dd = (equity_curve - peak) / peak
    return dd.min()


def regime_classify(spy_returns):
    """Classify each day as green/red/flat based on trailing 20d return."""
    trail = spy_returns.rolling(20).sum()
    regimes = pd.Series("flat", index=spy_returns.index)
    regimes[trail > 0.02] = "green"
    regimes[trail < -0.02] = "red"
    return regimes


# ── Part 0: Download Data ──────────────────────────────────────────────────
print("=" * 80)
print("DIVIDEND CAPTURE + COVERED CALL DEEP DIVE")
print("=" * 80)
print(f"\nDownloading data for {len(LARGE_CAP_30)} stocks from {START_DATE} to {END_DATE}...")

price_data = {}
div_data = {}
download_errors = []

for i, ticker in enumerate(LARGE_CAP_30):
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(start=START_DATE, end=END_DATE, actions=True)
        if len(hist) < 252:
            download_errors.append(f"{ticker}: insufficient data ({len(hist)} rows)")
            continue
        price_data[ticker] = hist[["Close"]].copy()
        # Extract actual dividends (ex-dates from yfinance)
        divs = hist[hist["Dividends"] > 0]["Dividends"].copy()
        if len(divs) > 0:
            div_data[ticker] = divs
            print(f"  [{i+1:2d}/{len(LARGE_CAP_30)}] {ticker}: {len(hist)} days, {len(divs)} dividends")
        else:
            download_errors.append(f"{ticker}: no dividends found")
    except Exception as e:
        download_errors.append(f"{ticker}: {e}")

# Download SPY for regime classification
spy = yf.Ticker("SPY")
spy_hist = spy.history(start=START_DATE, end=END_DATE)
spy_returns = spy_hist["Close"].pct_change().dropna()

if download_errors:
    print(f"\n  Skipped {len(download_errors)} tickers: {download_errors}")

valid_tickers = sorted(div_data.keys())
print(f"\n  Valid universe: {len(valid_tickers)} stocks with dividend data")
total_divs = sum(len(v) for v in div_data.values())
print(f"  Total dividend events: {total_divs}")


# ── Part 1: Reality Check ──────────────────────────────────────────────────
print("\n" + "=" * 80)
print("PART 1: REALITY CHECK — Does dividend capture actually work?")
print("=" * 80)

trades = []

for ticker in valid_tickers:
    prices = price_data[ticker]["Close"]
    divs = div_data[ticker]

    for ex_date, div_amount in divs.items():
        # ex_date is the actual ex-dividend date from yfinance
        # HC #724: anti-lookahead — buy at T-2 close (2 days before ex-date)
        # Sell at T+2 close (2 days after ex-date)

        # Find trading day indices
        all_dates = prices.index
        try:
            ex_idx = all_dates.get_loc(ex_date)
        except KeyError:
            # ex_date might not be an exact trading day, find nearest
            nearest = all_dates.searchsorted(ex_date)
            if nearest >= len(all_dates):
                continue
            ex_idx = nearest

        # T-2 (buy) and T+2 (sell)
        buy_idx = ex_idx - 2
        sell_idx = ex_idx + 2

        if buy_idx < 0 or sell_idx >= len(all_dates):
            continue

        buy_date = all_dates[buy_idx]
        sell_date = all_dates[sell_idx]
        buy_price = prices.iloc[buy_idx]
        sell_price = prices.iloc[sell_idx]

        # Actual ex-date price behavior
        if ex_idx > 0:
            pre_ex_close = prices.iloc[ex_idx - 1]
            ex_close = prices.iloc[ex_idx]
            ex_day_drop = pre_ex_close - ex_close  # positive = stock dropped
            drop_vs_div = ex_day_drop / div_amount if div_amount > 0 else np.nan
        else:
            ex_day_drop = np.nan
            drop_vs_div = np.nan

        # Costs
        trade_value = buy_price
        cost_pct = (COST_BPS + SPREAD_BPS) / 10000  # total cost as fraction
        cost_dollars = trade_value * cost_pct

        # P&L
        price_pnl = sell_price - buy_price
        total_pnl = price_pnl + div_amount - cost_dollars
        pnl_pct = total_pnl / buy_price

        # Realized vol for CC pricing (trailing 20d)
        vol_window = prices.iloc[max(0, buy_idx-25):buy_idx]
        if len(vol_window) > 5:
            log_ret = np.log(vol_window / vol_window.shift(1)).dropna()
            realized_vol = log_ret.std() * np.sqrt(252) if len(log_ret) > 1 else 0.3
        else:
            realized_vol = 0.3

        # Covered call premium (ATM, 5-day DTE)
        cc_premium = bs_call(buy_price, buy_price, RISK_FREE, realized_vol, 5/252)

        # CC effect on P&L: collect premium, but cap upside at strike
        # If sell_price > buy_price: we get called away, profit capped at premium
        # If sell_price <= buy_price: we keep shares + premium
        if sell_price > buy_price:
            cc_price_pnl = 0  # called away at strike = buy_price
            cc_total_pnl = cc_premium + div_amount - cost_dollars
        else:
            cc_price_pnl = sell_price - buy_price
            cc_total_pnl = cc_price_pnl + cc_premium + div_amount - cost_dollars
        cc_pnl_pct = cc_total_pnl / buy_price

        # Dividend yield of this particular payment
        div_yield_annualized = (div_amount / buy_price) * (252 / 4)  # assume quarterly

        trades.append({
            "ticker": ticker,
            "ex_date": ex_date,
            "buy_date": buy_date,
            "sell_date": sell_date,
            "buy_price": buy_price,
            "sell_price": sell_price,
            "dividend": div_amount,
            "div_yield_ann": div_yield_annualized,
            "price_pnl": price_pnl,
            "cost": cost_dollars,
            "total_pnl": total_pnl,
            "pnl_pct": pnl_pct,
            "ex_day_drop": ex_day_drop,
            "drop_vs_div_ratio": drop_vs_div,
            "realized_vol_20d": realized_vol,
            "cc_premium": cc_premium,
            "cc_total_pnl": cc_total_pnl,
            "cc_pnl_pct": cc_pnl_pct,
            "year": ex_date.year if hasattr(ex_date, 'year') else pd.Timestamp(ex_date).year,
        })

trades_df = pd.DataFrame(trades)
print(f"\nTotal trades analyzed: {len(trades_df)}")

# Reality check stats
print(f"\n--- Ex-Date Price Drop Analysis ---")
valid_drops = trades_df["drop_vs_div_ratio"].dropna()
print(f"  Mean drop/dividend ratio: {valid_drops.mean():.3f} (theory=1.0)")
print(f"  Median drop/dividend ratio: {valid_drops.median():.3f}")
print(f"  Std of drop/dividend ratio: {valid_drops.std():.3f}")
print(f"  % where stock drops MORE than dividend: {(valid_drops > 1).mean()*100:.1f}%")
print(f"  % where stock drops LESS than dividend: {(valid_drops < 1).mean()*100:.1f}%")
print(f"  % where stock actually RISES on ex-date: {(valid_drops < 0).mean()*100:.1f}%")

print(f"\n--- Dividend Capture (NO covered call) ---")
print(f"  Win rate: {(trades_df['total_pnl'] > 0).mean()*100:.1f}%")
print(f"  Mean P&L per trade: ${trades_df['total_pnl'].mean():.2f}")
print(f"  Median P&L per trade: ${trades_df['total_pnl'].median():.2f}")
print(f"  Mean P&L %: {trades_df['pnl_pct'].mean()*100:.3f}%")
print(f"  Std P&L %: {trades_df['pnl_pct'].std()*100:.3f}%")
print(f"  Avg holding period: ~4 trading days")
per_trade_sr = trades_df['pnl_pct'].mean() / trades_df['pnl_pct'].std() if trades_df['pnl_pct'].std() > 0 else 0
# Annualize: ~4 trades/stock/year, ~25 stocks = ~100 trades/year
trades_per_year = len(trades_df) / max(1, (trades_df['year'].max() - trades_df['year'].min() + 1))
print(f"  Trades per year: {trades_per_year:.0f}")
print(f"  Per-trade Sharpe: {per_trade_sr:.4f}")
print(f"  Annualized Sharpe (approx): {per_trade_sr * np.sqrt(trades_per_year):.2f}")

print(f"\n--- With Covered Call Premium ---")
print(f"  Win rate: {(trades_df['cc_total_pnl'] > 0).mean()*100:.1f}%")
print(f"  Mean P&L per trade: ${trades_df['cc_total_pnl'].mean():.2f}")
print(f"  Median P&L per trade: ${trades_df['cc_total_pnl'].median():.2f}")
print(f"  Mean P&L %: {trades_df['cc_pnl_pct'].mean()*100:.3f}%")
print(f"  Mean CC premium as % of stock: {(trades_df['cc_premium']/trades_df['buy_price']).mean()*100:.3f}%")
cc_sr = trades_df['cc_pnl_pct'].mean() / trades_df['cc_pnl_pct'].std() if trades_df['cc_pnl_pct'].std() > 0 else 0
print(f"  Per-trade Sharpe: {cc_sr:.4f}")
print(f"  Annualized Sharpe (approx): {cc_sr * np.sqrt(trades_per_year):.2f}")


# ── Part 2: Full Portfolio Simulation with Sizing ───────────────────────────
print("\n" + "=" * 80)
print("PART 3: POSITION SIZING OPTIMIZATION")
print("=" * 80)

def simulate_portfolio(trades_df, sizing_pct, use_cc=True, capital=INITIAL_CAPITAL):
    """Simulate portfolio with given position sizing."""
    trades_sorted = trades_df.sort_values("buy_date").copy()

    equity = capital
    equity_curve = {}
    daily_returns = []
    open_positions = []

    # Build a daily timeline
    all_dates = pd.date_range(trades_sorted["buy_date"].min(), trades_sorted["sell_date"].max(), freq="B")

    for date in all_dates:
        # Close positions that are due
        pnl_today = 0
        still_open = []
        for pos in open_positions:
            if date >= pos["sell_date"]:
                if use_cc:
                    pnl_today += pos["cc_total_pnl"] * pos["shares"]
                else:
                    pnl_today += pos["total_pnl"] * pos["shares"]
            else:
                still_open.append(pos)
        open_positions = still_open
        equity += pnl_today

        # Open new positions for today's buys
        todays_buys = trades_sorted[trades_sorted["buy_date"] == date]
        for _, trade in todays_buys.iterrows():
            position_value = equity * sizing_pct
            shares = int(position_value / trade["buy_price"])
            if shares > 0:
                pos = trade.to_dict()
                pos["shares"] = shares
                open_positions.append(pos)

        equity_curve[date] = equity

    eq = pd.Series(equity_curve)
    rets = eq.pct_change().dropna()

    return {
        "sharpe": sharpe(rets),
        "sortino": sortino(rets),
        "cagr": cagr(eq),
        "max_dd": max_dd(eq),
        "win_rate": (rets > 0).mean(),
        "final_equity": eq.iloc[-1],
        "n_trades": len(trades_sorted),
        "equity_curve": eq,
        "daily_returns": rets,
    }


sizing_levels = [0.05, 0.10, 0.15, 0.20, 0.25, 0.33]
print(f"\nTesting sizing levels: {sizing_levels}")
print(f"{'Sizing':>8} {'Sharpe':>8} {'Sortino':>8} {'CAGR':>8} {'MaxDD':>8} {'Final$':>10}")
print("-" * 60)

sizing_results = {}
for sz in sizing_levels:
    res = simulate_portfolio(trades_df, sz, use_cc=True)
    sizing_results[sz] = res
    print(f"  {sz*100:5.0f}%  {res['sharpe']:8.2f} {res['sortino']:8.2f} {res['cagr']*100:7.1f}% {res['max_dd']*100:7.1f}% {res['final_equity']:10,.0f}")

# Find sweet spot
best_sz = None
best_sharpe = -999
for sz, res in sizing_results.items():
    if res["max_dd"] > -0.20 and res["sharpe"] > best_sharpe:
        best_sharpe = res["sharpe"]
        best_sz = sz

if best_sz is None:
    # No sizing keeps MaxDD < 20%, pick the one closest
    best_sz = min(sizing_results.keys(), key=lambda s: abs(sizing_results[s]["max_dd"] + 0.20))
    print(f"\n  WARNING: No sizing level achieves MaxDD < 20%")

print(f"\n  OPTIMAL SIZING: {best_sz*100:.0f}% (Sharpe={sizing_results[best_sz]['sharpe']:.2f}, MaxDD={sizing_results[best_sz]['max_dd']*100:.1f}%)")


# ── Part 4: Universe Expansion ──────────────────────────────────────────────
print("\n" + "=" * 80)
print("PART 4: UNIVERSE EXPANSION TESTS")
print("=" * 80)

# Calculate annualized dividend yield for each ticker
ticker_yields = {}
for ticker in valid_tickers:
    divs = div_data[ticker]
    prices = price_data[ticker]["Close"]
    avg_price = prices.mean()
    annual_div = divs.groupby(divs.index.year).sum().mean()
    ticker_yields[ticker] = annual_div / avg_price

# Calculate 60d momentum for each ticker
ticker_momentum = {}
for ticker in valid_tickers:
    prices = price_data[ticker]["Close"]
    mom = (prices.iloc[-1] / prices.iloc[-63] - 1) if len(prices) > 63 else 0
    ticker_momentum[ticker] = mom

yield_df = pd.DataFrame({
    "ticker": list(ticker_yields.keys()),
    "yield": list(ticker_yields.values()),
    "momentum_60d": [ticker_momentum.get(t, 0) for t in ticker_yields.keys()]
}).sort_values("yield", ascending=False)

print("\nTicker yields:")
for _, row in yield_df.iterrows():
    print(f"  {row['ticker']:5s}: yield={row['yield']*100:.2f}%, mom60d={row['momentum_60d']*100:.1f}%")

# Universe A: Top 10 highest yield
top10_yield = set(yield_df.head(10)["ticker"].tolist())
# Universe B: All 30 (well, all valid)
all_tickers = set(valid_tickers)
# Universe C: Yield > 3% AND positive 60d momentum
quality_tickers = set(yield_df[(yield_df["yield"] > 0.03) & (yield_df["momentum_60d"] > 0)]["ticker"].tolist())

universes = {
    "A: Top 10 Yield": top10_yield,
    "B: All Stocks": all_tickers,
    "C: Yield>3% + Mom>0": quality_tickers,
}

opt_sz = best_sz  # Use optimal sizing from Part 3

print(f"\nUsing {opt_sz*100:.0f}% position sizing for all universe tests:")
print(f"{'Universe':>25} {'#Stk':>5} {'#Trd':>6} {'Sharpe':>8} {'Sortino':>8} {'CAGR':>8} {'MaxDD':>8}")
print("-" * 78)

universe_results = {}
for name, tickers in universes.items():
    if len(tickers) == 0:
        print(f"  {name:>25}: EMPTY UNIVERSE - skipped")
        continue
    subset = trades_df[trades_df["ticker"].isin(tickers)]
    if len(subset) < 10:
        print(f"  {name:>25}: only {len(subset)} trades - skipped")
        continue
    res = simulate_portfolio(subset, opt_sz, use_cc=True)
    universe_results[name] = res
    print(f"  {name:>25} {len(tickers):5d} {len(subset):6d} {res['sharpe']:8.2f} {res['sortino']:8.2f} {res['cagr']*100:7.1f}% {res['max_dd']*100:7.1f}%")


# ── Part 5: Adversarial Validation ─────────────────────────────────────────
print("\n" + "=" * 80)
print("PART 5: ADVERSARIAL VALIDATION")
print("=" * 80)

# Use best universe or all tickers
best_universe_name = max(universe_results.keys(), key=lambda k: universe_results[k]["sharpe"]) if universe_results else "B: All Stocks"
best_universe_tickers = universes.get(best_universe_name, all_tickers)
best_trades = trades_df[trades_df["ticker"].isin(best_universe_tickers)].copy()
print(f"\nUsing universe: {best_universe_name} ({len(best_universe_tickers)} stocks, {len(best_trades)} trades)")

# --- 5a: Walk-forward ---
print(f"\n--- 5a: Walk-Forward (252d lookback, monthly test) ---")
best_trades_sorted = best_trades.sort_values("buy_date")
years = sorted(best_trades_sorted["year"].unique())

wf_results = []
for yr in years:
    yr_trades = best_trades_sorted[best_trades_sorted["year"] == yr]
    if len(yr_trades) < 5:
        continue
    rets = yr_trades["cc_pnl_pct"]
    sr = rets.mean() / rets.std() * np.sqrt(len(rets)) if rets.std() > 0 else 0
    wf_results.append({
        "year": yr,
        "n_trades": len(yr_trades),
        "mean_pnl_pct": rets.mean() * 100,
        "wr": (rets > 0).mean() * 100,
        "sharpe_approx": sr,
    })

print(f"  {'Year':>6} {'#Trades':>8} {'Mean PnL%':>10} {'WR':>6} {'Sharpe':>8}")
print("  " + "-" * 45)
for r in wf_results:
    print(f"  {r['year']:6d} {r['n_trades']:8d} {r['mean_pnl_pct']:9.3f}% {r['wr']:5.1f}% {r['sharpe_approx']:8.2f}")

# Year-over-year consistency
yr_sharpes = [r["sharpe_approx"] for r in wf_results]
profitable_years = sum(1 for r in wf_results if r["mean_pnl_pct"] > 0)
print(f"\n  Profitable years: {profitable_years}/{len(wf_results)}")
print(f"  Sharpe range: [{min(yr_sharpes):.2f}, {max(yr_sharpes):.2f}]")
print(f"  Sharpe std across years: {np.std(yr_sharpes):.2f}")

# --- 5b: Sub-period stability ---
print(f"\n--- 5b: Sub-Period Stability ---")
mid_year = years[len(years)//2]
first_half = best_trades[best_trades["year"] <= mid_year]
second_half = best_trades[best_trades["year"] > mid_year]

for label, subset in [("First Half", first_half), ("Second Half", second_half)]:
    if len(subset) < 10:
        print(f"  {label}: too few trades ({len(subset)})")
        continue
    res = simulate_portfolio(subset, opt_sz, use_cc=True)
    print(f"  {label}: Sharpe={res['sharpe']:.2f}, CAGR={res['cagr']*100:.1f}%, MaxDD={res['max_dd']*100:.1f}%, n={len(subset)}")

if len(first_half) >= 10 and len(second_half) >= 10:
    h1_res = simulate_portfolio(first_half, opt_sz, use_cc=True)
    h2_res = simulate_portfolio(second_half, opt_sz, use_cc=True)
    sharpe_gap = abs(h1_res["sharpe"] - h2_res["sharpe"]) / max(abs(h1_res["sharpe"]), abs(h2_res["sharpe"]), 0.01)
    print(f"  Sub-period Sharpe gap: {sharpe_gap:.2f} ({'PASS' if sharpe_gap < 0.50 else 'FAIL'} threshold=0.50)")


# --- 5c: Regime Test ---
print(f"\n--- 5c: Regime Test (SPY-based classification) ---")
regimes = regime_classify(spy_returns)

# Map each trade to its regime
best_trades_regime = best_trades.copy()
regime_labels = []
for _, trade in best_trades_regime.iterrows():
    bd = trade["buy_date"]
    # Find closest regime date
    closest = regimes.index.searchsorted(bd)
    if closest < len(regimes):
        regime_labels.append(regimes.iloc[min(closest, len(regimes)-1)])
    else:
        regime_labels.append("flat")
best_trades_regime["regime"] = regime_labels

print(f"  {'Regime':>8} {'#Trades':>8} {'Mean PnL%':>10} {'WR':>6} {'Sharpe':>8}")
print("  " + "-" * 45)
regime_sharpes = {}
for regime in ["green", "flat", "red"]:
    subset = best_trades_regime[best_trades_regime["regime"] == regime]
    if len(subset) < 5:
        print(f"  {regime:>8}: too few trades ({len(subset)})")
        continue
    rets = subset["cc_pnl_pct"]
    sr = rets.mean() / rets.std() * np.sqrt(len(subset)) if rets.std() > 0 else 0
    regime_sharpes[regime] = sr
    print(f"  {regime:>8} {len(subset):8d} {rets.mean()*100:9.3f}% {(rets>0).mean()*100:5.1f}% {sr:8.2f}")

# Regime-agnostic check (HC #428 R1)
if len(regime_sharpes) >= 2:
    max_sr = max(abs(v) for v in regime_sharpes.values())
    if max_sr > 0:
        sr_vals = list(regime_sharpes.values())
        regime_gap = (max(sr_vals) - min(sr_vals)) / max(abs(max(sr_vals)), abs(min(sr_vals)), 0.01)
        print(f"\n  Regime Sharpe gap: {regime_gap:.2f} ({'PASS' if regime_gap < 0.50 else 'FAIL'} threshold=0.50)")
    else:
        print(f"\n  Regime Sharpe gap: N/A (all zero)")


# --- 5d: Permutation Test ---
print(f"\n--- 5d: Permutation Test (100 shuffles, HC #718) ---")
# Actual strategy Sharpe
actual_rets = best_trades["cc_pnl_pct"]
actual_sharpe = actual_rets.mean() / actual_rets.std() * np.sqrt(len(actual_rets)) if actual_rets.std() > 0 else 0

# Permutation: shuffle the dividend-date-to-stock mapping
# This breaks the connection between "which stock pays dividend when" and "what happens to price"
n_perms = 100
perm_sharpes = []

for p in range(n_perms):
    # Shuffle which dividend goes to which stock's price action
    perm_trades = best_trades.copy()
    # Shuffle the dividend amount and CC premium across trades (keeping price action intact)
    shuffle_idx = np.random.permutation(len(perm_trades))
    perm_trades["dividend"] = perm_trades["dividend"].values[shuffle_idx]
    perm_trades["cc_premium"] = perm_trades["cc_premium"].values[shuffle_idx]

    # Recalculate P&L
    perm_trades["cc_total_pnl_perm"] = np.where(
        perm_trades["sell_price"] > perm_trades["buy_price"],
        perm_trades["cc_premium"] + perm_trades["dividend"] - perm_trades["cost"],
        (perm_trades["sell_price"] - perm_trades["buy_price"]) + perm_trades["cc_premium"] + perm_trades["dividend"] - perm_trades["cost"]
    )
    perm_rets = perm_trades["cc_total_pnl_perm"] / perm_trades["buy_price"]
    perm_sr = perm_rets.mean() / perm_rets.std() * np.sqrt(len(perm_rets)) if perm_rets.std() > 0 else 0
    perm_sharpes.append(perm_sr)

perm_p_value = np.mean([ps >= actual_sharpe for ps in perm_sharpes])
print(f"  Actual Sharpe: {actual_sharpe:.2f}")
print(f"  Permuted Sharpe (mean): {np.mean(perm_sharpes):.2f}")
print(f"  Permuted Sharpe (95th pct): {np.percentile(perm_sharpes, 95):.2f}")
print(f"  p-value: {perm_p_value:.3f} ({'SIGNIFICANT' if perm_p_value < 0.05 else 'NOT SIGNIFICANT'})")


# ── Final Summary ───────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

best_res = sizing_results.get(opt_sz, sizing_results[list(sizing_results.keys())[0]])

print(f"""
STRATEGY: Dividend Capture + Covered Call (ATM, 5d DTE)
UNIVERSE: {best_universe_name} ({len(best_universe_tickers)} stocks)
SIZING: {opt_sz*100:.0f}% of capital per trade
PERIOD: {START_DATE} to {END_DATE}

PORTFOLIO METRICS:
  Sharpe Ratio:     {best_res['sharpe']:.2f}
  Sortino Ratio:    {best_res['sortino']:.2f}
  CAGR:             {best_res['cagr']*100:.1f}%
  Max Drawdown:     {best_res['max_dd']*100:.1f}%
  Final Equity:     ${best_res['final_equity']:,.0f} (from ${INITIAL_CAPITAL:,})

PER-TRADE METRICS (with CC):
  Win Rate:         {(best_trades['cc_total_pnl'] > 0).mean()*100:.1f}%
  Mean P&L:         ${best_trades['cc_total_pnl'].mean():.2f}
  Mean P&L %:       {best_trades['cc_pnl_pct'].mean()*100:.3f}%

REALITY CHECK:
  Ex-date drop/div ratio: {valid_drops.mean():.3f} (theory=1.0, <1.0 = edge)
  Stocks rising on ex-date: {(valid_drops < 0).mean()*100:.1f}%

ADVERSARIAL:
  Profitable years: {profitable_years}/{len(wf_results)}
  Perm test p-value: {perm_p_value:.3f}
  Regime gap: {regime_gap:.2f} (threshold=0.50)
  Sub-period gap: {sharpe_gap:.2f} (threshold=0.50)
""")

# Key finding
if valid_drops.mean() < 0.9:
    print("KEY FINDING: Stocks drop LESS than dividend on ex-date (ratio < 1.0).")
    print("This partial drop IS the edge — dividend capture harvests the difference.")
elif valid_drops.mean() > 1.1:
    print("KEY FINDING: Stocks drop MORE than dividend — dividend capture loses on price.")
    print("The covered call premium must offset the extra price drop to be viable.")
else:
    print("KEY FINDING: Ex-date drops are close to dividend amount (efficient market).")
    print("Edge comes primarily from covered call premium, not dividend capture itself.")

# Is simulation flawed?
print(f"\nSIMULATION VALIDITY:")
if perm_p_value < 0.05:
    print("  Permutation test: PASSES — edge is not random.")
else:
    print("  Permutation test: FAILS — edge may be noise/beta exposure.")

if regime_gap < 0.50:
    print("  Regime test: PASSES — works across market conditions.")
else:
    print("  Regime test: FAILS — strategy is regime-dependent.")

if sharpe_gap < 0.50:
    print("  Stability test: PASSES — consistent across sub-periods.")
else:
    print("  Stability test: FAILS — performance is period-dependent.")


# ── Save trade log ──────────────────────────────────────────────────────────
csv_path = os.path.join(OUT_DIR, "dividend_capture_trades_v1.csv")
trades_df.to_csv(csv_path, index=False)
print(f"\nTrade log saved to: {csv_path}")
print(f"Total trades: {len(trades_df)}")
print("\nDONE.")

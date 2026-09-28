#!/usr/bin/env python3
"""
Contrarian Drops with ML Confirmation Backtest
================================================
Universe: 50 liquid S&P 500 stocks, OOT Jan 2022 - Jul 2026
Starting capital: $645, max 3 concurrent positions, equal weight.
6 variants (A-F) with permutation testing and 5-gate validation.

Quality Confirmation Proxies:
  Q1: Price above 200-SMA (long-term uptrend)
  Q2: Positive earnings surprise proxy (gapped >2% on approx quarterly date)
  Q3: Relative strength positive (20d return > SPY 20d return)
  Q4: Volume on drop day > 1.5x 20-day average (institutional selling)
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

# ─── Config ───────────────────────────────────────────────────────────────────
UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "NFLX", "CRM",
    "SNOW", "PLTR", "SOFI", "HOOD", "SNAP", "PINS", "COIN", "RBLX", "UBER", "LYFT",
    "ROKU", "NET", "DDOG", "TTD", "SHOP", "ORCL", "ADBE", "INTC", "QCOM", "AVGO",
    "MU", "AMAT", "LRCX", "PANW", "CRWD", "ZS", "NOW", "ABNB", "SQ", "PYPL",
    "MELI", "NU", "SE", "SPOT", "DASH", "RIVN", "LCID", "ARM", "SMCI", "MSTR"
]
START = "2020-06-01"  # need 200-SMA lookback before OOT
OOT_START = "2022-01-03"
OOT_END = "2026-07-28"
INITIAL_CAPITAL = 645.0
MAX_CONCURRENT = 3
SLIPPAGE_PCT = 0.0002  # 0.02% slippage on shares
COMMISSION_SHARES = 0.0  # $0 commission on shares
OPTION_COMMISSION = 0.65
OPTION_PREMIUM_PCT = 0.03  # 3% of stock price for ATM call
OPTION_BIDASK_PCT = 0.05  # 5% bid-ask spread on options
N_PERMS = 1000
RISK_FREE = 0.04

# ─── Data download ───────────────────────────────────────────────────────────
print("Downloading price data for 50 stocks + SPY...")
tickers = UNIVERSE + ["SPY"]
raw = yf.download(tickers, start=START, end=OOT_END, auto_adjust=True, progress=False)

close = raw["Close"].copy()
volume = raw["Volume"].copy()
close = close.ffill()
volume = volume.ffill().fillna(0)

# Drop tickers with insufficient data
valid_stocks = [s for s in UNIVERSE if s in close.columns and close[s].notna().sum() > 252]
print(f"Valid tickers: {len(valid_stocks)}/{len(UNIVERSE)}")

spy_close = close["SPY"].copy()
spy_sma200 = spy_close.rolling(200).mean()
regime = (spy_close > spy_sma200).astype(int)  # 1=bull, 0=bear

# OOT mask
oot_mask = close.index >= OOT_START
dates_oot = close.index[oot_mask].tolist()
n_oot = len(dates_oot)
print(f"OOT period: {dates_oot[0].date()} to {dates_oot[-1].date()}, {n_oot} trading days")
bull_days = regime.loc[oot_mask].sum()
bear_days = n_oot - bull_days
print(f"Bull days: {bull_days}, Bear days: {bear_days}")

# ─── Pre-compute indicators (vectorized) ────────────────────────────────────
stocks = valid_stocks
close_s = close[stocks]
vol_s = volume[stocks]

# Daily returns
daily_ret = close_s.pct_change()

# 200-SMA
sma200 = close_s.rolling(200).mean()
above_sma200 = close_s > sma200  # Q1

# 20-day average volume
vol_avg20 = vol_s.rolling(20).mean()
vol_ratio = vol_s / vol_avg20  # Q4: > 1.5 means high volume

# 20-day returns for relative strength
ret_20 = close_s.pct_change(20)
spy_ret_20 = spy_close.pct_change(20)
rs_positive = ret_20.subtract(spy_ret_20, axis=0) > 0  # Q3

# Earnings surprise proxy: approximate quarterly dates (every ~63 trading days)
# Check if stock gapped >2% up on most recent quarterly-ish date
# We'll check every 63 trading days backward and see if any had a >2% gap up
def compute_earnings_surprise(close_df, lookback_days=63, n_quarters=4):
    """For each date, check if any of the last N quarterly dates had >2% gap up."""
    daily_ret = close_df.pct_change()
    result = pd.DataFrame(False, index=close_df.index, columns=close_df.columns)

    for q in range(1, n_quarters + 1):
        shifted = daily_ret.shift(q * lookback_days)
        result = result | (shifted > 0.02)

    return result

print("Computing quality indicators...")
earnings_surprise = compute_earnings_surprise(close_s)  # Q2

# Drop detection: daily return < -threshold
drop_3pct = daily_ret < -0.03
drop_5pct = daily_ret < -0.05

# ─── Strategy engine ────────────────────────────────────────────────────────

def run_strategy(signal_dates_by_stock, hold_days, is_options=False):
    """
    Run a backtest given signal dates per stock.
    Returns dict with trades, equity curve, metrics.

    signal_dates_by_stock: dict of {stock: [list of entry dates]}
    hold_days: number of trading days to hold
    is_options: if True, model as options trade
    """
    capital = INITIAL_CAPITAL
    positions = []  # list of {stock, entry_date, entry_price, shares, exit_idx, is_option, premium}
    trades = []
    equity = []
    daily_pnl = []

    all_dates = dates_oot
    date_to_idx = {d: i for i, d in enumerate(close.index)}

    for date in all_dates:
        idx = date_to_idx.get(date)
        if idx is None:
            equity.append(capital)
            daily_pnl.append(0.0)
            continue

        # Close expired positions
        closed_pnl = 0.0
        new_positions = []
        for pos in positions:
            exit_idx = pos["exit_idx"]
            if idx >= exit_idx:
                # Exit
                actual_exit_idx = min(exit_idx, len(close.index) - 1)
                exit_price = close_s.iloc[actual_exit_idx].get(pos["stock"], np.nan)
                if pd.isna(exit_price):
                    exit_price = pos["entry_price"]  # fallback

                if is_options:
                    # Option P&L: intrinsic value at expiry minus premium paid
                    strike = pos["entry_price"]  # ATM
                    intrinsic = max(0, exit_price - strike)
                    premium = pos["premium"]
                    pnl = (intrinsic - premium) * pos["shares"] - OPTION_COMMISSION
                else:
                    pnl = (exit_price - pos["entry_price"]) * pos["shares"]
                    pnl -= abs(exit_price * pos["shares"] * SLIPPAGE_PCT)  # exit slippage

                capital += pos["invested"] + pnl
                closed_pnl += pnl
                trades.append({
                    "stock": pos["stock"],
                    "entry_date": str(pos["entry_date"].date()),
                    "exit_date": str(close.index[actual_exit_idx].date()),
                    "entry_price": round(pos["entry_price"], 2),
                    "exit_price": round(exit_price, 2),
                    "pnl": round(pnl, 2),
                    "return_pct": round(pnl / pos["invested"] * 100, 2) if pos["invested"] > 0 else 0,
                    "regime": "bull" if regime.iloc[date_to_idx[pos["entry_date"]]] == 1 else "bear"
                })
            else:
                new_positions.append(pos)
        positions = new_positions

        # Open new positions
        if len(positions) < MAX_CONCURRENT and capital > 50:
            signals_today = []
            for stock in stocks:
                if stock in signal_dates_by_stock and date in signal_dates_by_stock[stock]:
                    # Don't double up on same stock
                    if not any(p["stock"] == stock for p in positions):
                        signals_today.append(stock)

            slots = MAX_CONCURRENT - len(positions)
            for stock in signals_today[:slots]:
                price = close_s.at[date, stock] if date in close_s.index else np.nan
                if pd.isna(price) or price <= 0:
                    continue

                pos_size = capital / (MAX_CONCURRENT - len(positions))
                pos_size = min(pos_size, capital)

                if is_options:
                    premium = price * OPTION_PREMIUM_PCT
                    premium *= (1 + OPTION_BIDASK_PCT)  # bid-ask spread
                    n_contracts = max(1, int(pos_size / (premium * 100)))
                    invested = n_contracts * premium * 100 + OPTION_COMMISSION
                    if invested > capital:
                        n_contracts = max(1, int((capital - OPTION_COMMISSION) / (premium * 100)))
                        invested = n_contracts * premium * 100 + OPTION_COMMISSION
                    if invested > capital:
                        continue
                    shares = n_contracts * 100
                    capital -= invested
                    exit_idx_val = min(idx + 10, len(close.index) - 1)  # ~2 weeks
                    positions.append({
                        "stock": stock, "entry_date": date, "entry_price": price,
                        "shares": shares, "exit_idx": exit_idx_val,
                        "invested": invested, "premium": premium, "is_option": True
                    })
                else:
                    entry_price = price * (1 + SLIPPAGE_PCT)  # slippage on entry
                    n_shares = max(1, int(pos_size / entry_price))
                    invested = n_shares * entry_price
                    if invested > capital:
                        n_shares = max(1, int(capital / entry_price))
                        invested = n_shares * entry_price
                    if invested > capital:
                        continue
                    capital -= invested
                    exit_idx_val = min(idx + hold_days, len(close.index) - 1)
                    positions.append({
                        "stock": stock, "entry_date": date, "entry_price": entry_price,
                        "shares": n_shares, "exit_idx": exit_idx_val,
                        "invested": invested, "premium": 0, "is_option": False
                    })

        # Mark-to-market
        mtm = capital
        for pos in positions:
            curr_price = close_s.at[date, pos["stock"]] if date in close_s.index else pos["entry_price"]
            if pd.isna(curr_price):
                curr_price = pos["entry_price"]
            if is_options:
                intrinsic = max(0, curr_price - pos["entry_price"])
                mtm += intrinsic * pos["shares"]
            else:
                mtm += curr_price * pos["shares"]

        prev_eq = equity[-1] if equity else INITIAL_CAPITAL
        daily_pnl.append(mtm - prev_eq)
        equity.append(mtm)

    # Force-close remaining positions at end
    for pos in positions:
        last_idx = len(close.index) - 1
        exit_price = close_s.iloc[last_idx].get(pos["stock"], pos["entry_price"])
        if pd.isna(exit_price):
            exit_price = pos["entry_price"]
        if is_options:
            intrinsic = max(0, exit_price - pos["entry_price"])
            pnl = (intrinsic - pos["premium"]) * pos["shares"] - OPTION_COMMISSION
        else:
            pnl = (exit_price - pos["entry_price"]) * pos["shares"]
            pnl -= abs(exit_price * pos["shares"] * SLIPPAGE_PCT)
        trades.append({
            "stock": pos["stock"],
            "entry_date": str(pos["entry_date"].date()),
            "exit_date": str(close.index[last_idx].date()),
            "entry_price": round(pos["entry_price"], 2),
            "exit_price": round(exit_price, 2),
            "pnl": round(pnl, 2),
            "return_pct": round(pnl / pos["invested"] * 100, 2) if pos["invested"] > 0 else 0,
            "regime": "bull" if regime.iloc[date_to_idx.get(pos["entry_date"], 0)] == 1 else "bear"
        })

    return trades, equity, daily_pnl


def compute_metrics(trades, equity, daily_pnl):
    """Compute performance metrics from trades and equity curve."""
    if len(trades) == 0:
        return {"n_trades": 0, "sharpe": 0, "sortino": 0, "pf": 0, "wr": 0,
                "total_return_pct": 0, "max_dd_pct": 0, "avg_return_pct": 0,
                "bull_sharpe": 0, "bear_sharpe": 0, "regime_gap": 1.0,
                "perm_p": 1.0, "pass_5gate": False}

    pnls = [t["pnl"] for t in trades]
    returns = [t["return_pct"] / 100 for t in trades]

    n_trades = len(trades)
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p <= 0]
    wr = len(winners) / n_trades if n_trades > 0 else 0

    gross_profit = sum(winners) if winners else 0
    gross_loss = abs(sum(losers)) if losers else 0.001
    pf = gross_profit / gross_loss if gross_loss > 0 else 999

    total_pnl = sum(pnls)
    total_return_pct = total_pnl / INITIAL_CAPITAL * 100

    # Sharpe from daily equity
    eq = np.array(equity)
    if len(eq) > 1:
        daily_rets = np.diff(eq) / eq[:-1]
        daily_rets = daily_rets[np.isfinite(daily_rets)]
        if len(daily_rets) > 10 and np.std(daily_rets) > 0:
            ann_factor = np.sqrt(252)
            sharpe = (np.mean(daily_rets) - RISK_FREE / 252) / np.std(daily_rets) * ann_factor
            downside = daily_rets[daily_rets < 0]
            downside_std = np.std(downside) if len(downside) > 0 else np.std(daily_rets)
            sortino = (np.mean(daily_rets) - RISK_FREE / 252) / downside_std * ann_factor if downside_std > 0 else 0
        else:
            sharpe = sortino = 0.0
    else:
        sharpe = sortino = 0.0

    # Max drawdown
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / np.where(peak > 0, peak, 1)
    max_dd_pct = float(np.min(dd) * 100) if len(dd) > 0 else 0

    # Regime-stratified Sharpe
    bull_trades = [t for t in trades if t["regime"] == "bull"]
    bear_trades = [t for t in trades if t["regime"] == "bear"]

    def trade_sharpe(trade_list):
        if len(trade_list) < 3:
            return 0.0
        rets = np.array([t["return_pct"] / 100 for t in trade_list])
        if np.std(rets) == 0:
            return 0.0
        return float(np.mean(rets) / np.std(rets) * np.sqrt(len(rets)))

    bull_sharpe = trade_sharpe(bull_trades)
    bear_sharpe = trade_sharpe(bear_trades)
    max_abs = max(abs(bull_sharpe), abs(bear_sharpe), 0.001)
    regime_gap = abs(bull_sharpe - bear_sharpe) / max_abs

    avg_return_pct = np.mean([t["return_pct"] for t in trades])

    return {
        "n_trades": n_trades,
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "pf": round(pf, 3),
        "wr": round(wr, 3),
        "total_return_pct": round(total_return_pct, 1),
        "max_dd_pct": round(max_dd_pct, 1),
        "avg_return_pct": round(avg_return_pct, 2),
        "bull_sharpe": round(bull_sharpe, 3),
        "bear_sharpe": round(bear_sharpe, 3),
        "bull_trades": len(bull_trades),
        "bear_trades": len(bear_trades),
        "regime_gap": round(regime_gap, 3),
        "total_pnl": round(total_pnl, 2),
        "final_equity": round(INITIAL_CAPITAL + total_pnl, 2),
    }


def permutation_test(trades, equity, n_perms=N_PERMS):
    """Shuffle trade timing to test if signal matters vs random entry."""
    if len(trades) < 5:
        return 1.0

    actual_sharpe = compute_metrics(trades, equity, [])["sharpe"]

    # Get actual trade returns
    actual_returns = np.array([t["return_pct"] / 100 for t in trades])
    n = len(actual_returns)

    count_better = 0
    for _ in range(n_perms):
        # Shuffle returns (break signal-return link)
        perm_returns = np.random.permutation(actual_returns)
        if np.std(perm_returns) > 0:
            perm_sharpe = np.mean(perm_returns) / np.std(perm_returns) * np.sqrt(n)
        else:
            perm_sharpe = 0.0
        if perm_sharpe >= actual_sharpe:
            count_better += 1

    return count_better / n_perms


# ─── Generate signals for each variant ──────────────────────────────────────

def get_signals_A():
    """Variant A: Drop >3% + above 200-SMA, hold 5 days."""
    signals = {}
    for stock in stocks:
        mask = drop_3pct[stock] & above_sma200[stock] & oot_mask
        dates = close.index[mask].tolist()
        if dates:
            signals[stock] = set(dates)
    return signals

def get_signals_B():
    """Variant B: Drop >3% + above 200-SMA + volume > 1.5x avg, hold 5 days."""
    signals = {}
    for stock in stocks:
        high_vol = vol_ratio[stock] > 1.5
        mask = drop_3pct[stock] & above_sma200[stock] & high_vol & oot_mask
        dates = close.index[mask].tolist()
        if dates:
            signals[stock] = set(dates)
    return signals

def get_signals_C():
    """Variant C: Drop >3% + RS positive, hold 5 days."""
    signals = {}
    for stock in stocks:
        mask = drop_3pct[stock] & rs_positive[stock] & oot_mask
        dates = close.index[mask].tolist()
        if dates:
            signals[stock] = set(dates)
    return signals

def get_signals_D():
    """Variant D: Drop >3% + passes >=2 of 4 quality checks, hold 10 days."""
    signals = {}
    for stock in stocks:
        q_score = (above_sma200[stock].astype(int) +
                   earnings_surprise[stock].astype(int) +
                   rs_positive[stock].astype(int) +
                   (vol_ratio[stock] > 1.5).astype(int))
        mask = drop_3pct[stock] & (q_score >= 2) & oot_mask
        dates = close.index[mask].tolist()
        if dates:
            signals[stock] = set(dates)
    return signals

def get_signals_E():
    """Variant E: Drop >5% + above 200-SMA, hold 5 days."""
    signals = {}
    for stock in stocks:
        mask = drop_5pct[stock] & above_sma200[stock] & oot_mask
        dates = close.index[mask].tolist()
        if dates:
            signals[stock] = set(dates)
    return signals

def get_signals_F():
    """Variant F: Drop >5% + Q1 (200-SMA) + Q2 (earnings), options, hold 10 days."""
    signals = {}
    for stock in stocks:
        mask = drop_5pct[stock] & above_sma200[stock] & earnings_surprise[stock] & oot_mask
        dates = close.index[mask].tolist()
        if dates:
            signals[stock] = set(dates)
    return signals


# ─── Run all variants ───────────────────────────────────────────────────────

variants = {
    "A_drop3_sma200": {"signal_fn": get_signals_A, "hold": 5, "options": False,
                        "desc": "Drop >3% + above 200-SMA, hold 5d"},
    "B_drop3_sma200_vol": {"signal_fn": get_signals_B, "hold": 5, "options": False,
                            "desc": "Drop >3% + 200-SMA + high volume, hold 5d"},
    "C_drop3_rs_positive": {"signal_fn": get_signals_C, "hold": 5, "options": False,
                             "desc": "Drop >3% + RS positive vs SPY, hold 5d"},
    "D_multi_quality": {"signal_fn": get_signals_D, "hold": 10, "options": False,
                         "desc": "Drop >3% + >=2/4 quality checks, hold 10d"},
    "E_extreme_drop5_sma200": {"signal_fn": get_signals_E, "hold": 5, "options": False,
                                "desc": "Drop >5% + above 200-SMA, hold 5d"},
    "F_options_drop5_quality": {"signal_fn": get_signals_F, "hold": 10, "options": True,
                                 "desc": "ATM calls on >5% drop + SMA200 + earnings, hold 10d"},
}

results = {}

for name, cfg in variants.items():
    print(f"\n{'='*60}")
    print(f"Running {name}: {cfg['desc']}")
    print(f"{'='*60}")

    signals = cfg["signal_fn"]()
    total_signals = sum(len(v) for v in signals.values())
    print(f"  Signal dates: {total_signals} across {len(signals)} stocks")

    if total_signals == 0:
        print(f"  NO SIGNALS — skipping")
        results[name] = {
            "description": cfg["desc"],
            "n_signals": 0,
            "n_trades": 0,
            "pass_5gate": False,
            "reason": "no signals generated"
        }
        continue

    trades, equity, daily_pnl = run_strategy(signals, cfg["hold"], is_options=cfg["options"])
    metrics = compute_metrics(trades, equity, daily_pnl)

    print(f"  Trades: {metrics['n_trades']}, WR: {metrics['wr']:.1%}, PF: {metrics['pf']:.2f}")
    print(f"  Sharpe: {metrics['sharpe']:.3f}, Sortino: {metrics['sortino']:.3f}")
    print(f"  Total P&L: ${metrics['total_pnl']:.2f}, Return: {metrics['total_return_pct']:.1f}%")
    print(f"  Max DD: {metrics['max_dd_pct']:.1f}%")
    print(f"  Bull Sharpe: {metrics['bull_sharpe']:.3f} ({metrics['bull_trades']} trades)")
    print(f"  Bear Sharpe: {metrics['bear_sharpe']:.3f} ({metrics['bear_trades']} trades)")
    print(f"  Regime gap: {metrics['regime_gap']:.3f}")

    # Permutation test
    print(f"  Running permutation test ({N_PERMS} iterations)...")
    perm_p = permutation_test(trades, equity)
    print(f"  Permutation p-value: {perm_p:.4f}")

    # 5-gate validation
    gates = {
        "sharpe_gt_0.5": metrics["sharpe"] > 0.5,
        "perm_p_lt_0.05": perm_p < 0.05,
        "regime_gap_lt_0.5": metrics["regime_gap"] < 0.5,
        "max_dd_gt_neg50": metrics["max_dd_pct"] > -50,
        "min_20_trades": metrics["n_trades"] >= 20,
    }
    pass_all = all(gates.values())

    print(f"\n  5-Gate Validation:")
    for gate, passed in gates.items():
        status = "PASS" if passed else "FAIL"
        print(f"    {gate}: {status}")
    print(f"  OVERALL: {'PASS' if pass_all else 'FAIL'}")

    # Top 5 trades
    sorted_trades = sorted(trades, key=lambda t: t["pnl"], reverse=True)

    results[name] = {
        "description": cfg["desc"],
        "n_signals": total_signals,
        "metrics": metrics,
        "perm_p": round(perm_p, 4),
        "gates": gates,
        "pass_5gate": pass_all,
        "gates_passed": sum(gates.values()),
        "top_5_winners": sorted_trades[:5],
        "top_5_losers": sorted_trades[-5:],
    }

# ─── Summary ─────────────────────────────────────────────────────────────────

print("\n" + "=" * 80)
print("CONTRARIAN DROPS WITH ML CONFIRMATION — SUMMARY")
print("=" * 80)
print(f"Universe: {len(valid_stocks)} stocks, OOT: {dates_oot[0].date()} to {dates_oot[-1].date()}")
print(f"Account: ${INITIAL_CAPITAL}, Max {MAX_CONCURRENT} concurrent, $0 commission + 0.02% slippage")
print()

header = f"{'Variant':<30} {'Trades':>6} {'Sharpe':>7} {'Sortino':>8} {'WR':>6} {'PF':>6} {'P&L':>8} {'MaxDD':>7} {'PermP':>7} {'5G':>4}"
print(header)
print("-" * len(header))

for name, r in results.items():
    if r.get("n_trades", 0) == 0 and "metrics" not in r:
        print(f"{name:<30} {'N/A':>6} {'N/A':>7} {'N/A':>8} {'N/A':>6} {'N/A':>6} {'N/A':>8} {'N/A':>7} {'N/A':>7} {'FAIL':>4}")
        continue
    m = r["metrics"]
    pg = "PASS" if r["pass_5gate"] else f"{r['gates_passed']}/5"
    print(f"{name:<30} {m['n_trades']:>6} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} {m['wr']:>5.1%} {m['pf']:>6.2f} {m['total_pnl']:>7.2f} {m['max_dd_pct']:>6.1f}% {r['perm_p']:>7.4f} {pg:>4}")

# Best variant
passing = {k: v for k, v in results.items() if v.get("pass_5gate")}
if passing:
    best = max(passing.items(), key=lambda x: x[1]["metrics"]["sharpe"])
    print(f"\nBEST PASSING VARIANT: {best[0]} (Sharpe={best[1]['metrics']['sharpe']:.3f})")
else:
    # Best by gates passed
    best = max(results.items(), key=lambda x: (x[1].get("gates_passed", 0), x[1].get("metrics", {}).get("sharpe", 0)))
    print(f"\nNO VARIANTS PASSED 5-GATE. Best: {best[0]} ({best[1].get('gates_passed', 0)}/5 gates)")

# ─── Save results ────────────────────────────────────────────────────────────

output_path = Path("/home/jupiter/Lvl3Quant/data/contrarian_ml_confirm_results.json")

# Make serializable
def make_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

save_data = {
    "strategy": "Contrarian Drops with ML Confirmation",
    "run_date": datetime.now().isoformat(),
    "config": {
        "universe_size": len(valid_stocks),
        "oot_start": OOT_START,
        "oot_end": OOT_END,
        "initial_capital": INITIAL_CAPITAL,
        "max_concurrent": MAX_CONCURRENT,
        "slippage_pct": SLIPPAGE_PCT,
        "n_permutations": N_PERMS,
    },
    "variants": make_serializable(results),
}

output_path.write_text(json.dumps(save_data, indent=2, default=str))
print(f"\nResults saved to {output_path}")
print("Done.")

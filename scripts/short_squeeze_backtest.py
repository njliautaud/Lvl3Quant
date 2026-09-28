#!/usr/bin/env python3
"""
Short Squeeze Momentum Backtest
================================
Event-driven strategy exploiting short-squeeze dynamics in heavily-shorted / meme stocks.
Uses proxy signals (extreme price action + volume explosion) since real-time SI unavailable.

6 Variants (A-F), 5-gate validation framework.
OOT: Jan 2022 – Jul 2026 | Account: $645 | Slippage: 0.05% each way
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

# ── CONFIG ──────────────────────────────────────────────────────────────────
UNIVERSE = [
    "GME", "AMC", "PLTR", "SOFI", "HOOD", "RIVN", "LCID", "SNAP", "PINS",
    "RBLX", "COIN", "MARA", "RIOT", "SQ", "FUBO", "CLOV", "WISH", "SKLZ",
    "SPCE", "OPEN", "UPST", "AI", "JOBY", "IONQ", "BBBY",
]
START_DATE = "2022-01-01"
END_DATE = "2026-07-29"
INITIAL_CAPITAL = 645.0
SLIPPAGE_PCT = 0.0005  # 0.05% each way
PERMUTATION_ITERS = 1000
RESULTS_PATH = Path("/home/jupiter/Lvl3Quant/data/short_squeeze_results.json")

# ── DATA DOWNLOAD ───────────────────────────────────────────────────────────
print("Downloading price data for universe...")
price_data = {}
volume_data = {}
high_data = {}
open_data = {}

for ticker in UNIVERSE:
    try:
        df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
        if df is not None and len(df) > 60:
            # Handle multi-level columns from yfinance
            if isinstance(df.columns, pd.MultiIndex):
                price_data[ticker] = df[("Close", ticker)] if ("Close", ticker) in df.columns else df["Close"].iloc[:, 0]
                volume_data[ticker] = df[("Volume", ticker)] if ("Volume", ticker) in df.columns else df["Volume"].iloc[:, 0]
                high_data[ticker] = df[("High", ticker)] if ("High", ticker) in df.columns else df["High"].iloc[:, 0]
                open_data[ticker] = df[("Open", ticker)] if ("Open", ticker) in df.columns else df["Open"].iloc[:, 0]
            else:
                price_data[ticker] = df["Close"]
                volume_data[ticker] = df["Volume"]
                high_data[ticker] = df["High"]
                open_data[ticker] = df["Open"]
            print(f"  {ticker}: {len(df)} days")
        else:
            print(f"  {ticker}: SKIPPED (insufficient data)")
    except Exception as e:
        print(f"  {ticker}: FAILED ({e})")

closes = pd.DataFrame(price_data)
volumes = pd.DataFrame(volume_data)
highs = pd.DataFrame(high_data)
opens = pd.DataFrame(open_data)
print(f"\nLoaded {len(closes.columns)} tickers, {len(closes)} trading days\n")


# ── HELPER FUNCTIONS ────────────────────────────────────────────────────────
def compute_forward_returns(closes, hold_days, short=False):
    """Compute forward returns with slippage for a given hold period."""
    fwd = closes.shift(-hold_days) / closes - 1.0
    # Apply slippage: 0.05% on entry + 0.05% on exit = 0.10% total
    slippage_total = 2 * SLIPPAGE_PCT
    if short:
        fwd = -fwd - slippage_total
    else:
        fwd = fwd - slippage_total
    return fwd


def run_backtest(signals: pd.DataFrame, fwd_returns: pd.DataFrame, variant_name: str):
    """
    Given a boolean signal matrix and forward returns matrix, compute backtest stats.
    signals: True where we enter a trade on that day for that ticker.
    fwd_returns: the return we'd get if we entered that day.
    """
    # Align
    common_idx = signals.index.intersection(fwd_returns.index)
    common_cols = signals.columns.intersection(fwd_returns.columns)
    sig = signals.loc[common_idx, common_cols]
    ret = fwd_returns.loc[common_idx, common_cols]

    # Flatten to trade list
    trade_dates = []
    trade_tickers = []
    trade_returns = []

    for dt in common_idx:
        for tk in common_cols:
            if sig.loc[dt, tk]:
                r = ret.loc[dt, tk]
                if not np.isnan(r):
                    trade_dates.append(dt)
                    trade_tickers.append(tk)
                    trade_returns.append(r)

    if len(trade_returns) == 0:
        return None

    trade_returns = np.array(trade_returns)
    trade_dates_arr = np.array(trade_dates)

    # Build equity curve (equal-weight per trade, $645 account)
    # Group trades by entry date, compute daily portfolio return
    trade_df = pd.DataFrame({
        "date": trade_dates_arr,
        "ticker": trade_tickers,
        "return": trade_returns
    })

    # Average return per day (equal weight across concurrent trades)
    daily_ret = trade_df.groupby("date")["return"].mean()
    daily_ret = daily_ret.sort_index()

    equity = INITIAL_CAPITAL * (1 + daily_ret).cumprod()
    cum_max = equity.cummax()
    drawdown = (equity - cum_max) / cum_max
    max_dd = drawdown.min()

    n_trades = len(trade_returns)
    win_rate = (trade_returns > 0).mean()
    avg_ret = trade_returns.mean()
    std_ret = trade_returns.std()

    # Annualize (assume avg hold ~ a few days, use per-trade Sharpe scaled)
    # Use daily returns series for Sharpe
    if len(daily_ret) > 1 and daily_ret.std() > 0:
        sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252)
        downside = daily_ret[daily_ret < 0].std()
        sortino = daily_ret.mean() / downside * np.sqrt(252) if downside > 0 else np.inf
    else:
        sharpe = 0.0
        sortino = 0.0

    pf = trade_returns[trade_returns > 0].sum() / abs(trade_returns[trade_returns < 0].sum()) if (trade_returns < 0).any() else np.inf

    # Regime analysis: split by SPY direction
    try:
        spy = yf.download("SPY", start=START_DATE, end=END_DATE, progress=False)
        if isinstance(spy.columns, pd.MultiIndex):
            spy_close = spy[("Close", "SPY")] if ("Close", "SPY") in spy.columns else spy["Close"].iloc[:, 0]
        else:
            spy_close = spy["Close"]
        spy_monthly = spy_close.resample("ME").last().pct_change()

        # Classify each trade date's month
        bull_rets = []
        bear_rets = []
        for dt, r in zip(trade_dates_arr, trade_returns):
            month_end = pd.Timestamp(dt).to_period("M").to_timestamp("M")
            closest = spy_monthly.index[spy_monthly.index <= month_end]
            if len(closest) > 0:
                spy_r = spy_monthly.loc[closest[-1]]
                if spy_r > 0:
                    bull_rets.append(r)
                else:
                    bear_rets.append(r)
    except:
        bull_rets = trade_returns[:len(trade_returns)//2].tolist()
        bear_rets = trade_returns[len(trade_returns)//2:].tolist()

    bull_rets = np.array(bull_rets) if bull_rets else np.array([0.0])
    bear_rets = np.array(bear_rets) if bear_rets else np.array([0.0])

    sharpe_bull = bull_rets.mean() / bull_rets.std() * np.sqrt(252) if bull_rets.std() > 0 else 0
    sharpe_bear = bear_rets.mean() / bear_rets.std() * np.sqrt(252) if bear_rets.std() > 0 else 0

    max_s = max(abs(sharpe_bull), abs(sharpe_bear))
    regime_gap = abs(sharpe_bull - sharpe_bear) / max_s if max_s > 0 else 0

    # Permutation test
    observed_mean = trade_returns.mean()
    perm_count = 0
    for _ in range(PERMUTATION_ITERS):
        perm = np.random.choice(trade_returns, size=len(trade_returns), replace=True)
        np.random.shuffle(perm)
        # Shuffle signs to test if mean return is significant
        shuffled = trade_returns * np.random.choice([-1, 1], size=len(trade_returns))
        if shuffled.mean() >= observed_mean:
            perm_count += 1
    p_value = perm_count / PERMUTATION_ITERS

    final_equity = equity.iloc[-1] if len(equity) > 0 else INITIAL_CAPITAL
    total_return = (final_equity / INITIAL_CAPITAL - 1) * 100

    result = {
        "variant": variant_name,
        "n_trades": int(n_trades),
        "win_rate": round(win_rate * 100, 1),
        "avg_return_pct": round(avg_ret * 100, 3),
        "total_return_pct": round(total_return, 1),
        "final_equity": round(float(final_equity), 2),
        "sharpe": round(float(sharpe), 3),
        "sortino": round(float(sortino), 3),
        "profit_factor": round(float(min(pf, 99.0)), 3),
        "max_drawdown_pct": round(float(max_dd * 100), 1),
        "sharpe_bull": round(float(sharpe_bull), 3),
        "sharpe_bear": round(float(sharpe_bear), 3),
        "regime_gap": round(float(regime_gap), 3),
        "perm_p_value": round(float(p_value), 4),
        # 5-gate checks
        "gate_sharpe": sharpe > 0.5,
        "gate_perm": p_value < 0.05,
        "gate_regime": regime_gap < 0.5,
        "gate_maxdd": max_dd > -0.50,
        "gate_trades": n_trades >= 20,
    }
    result["gates_passed"] = sum([
        result["gate_sharpe"], result["gate_perm"], result["gate_regime"],
        result["gate_maxdd"], result["gate_trades"]
    ])
    return result


# ── PRECOMPUTE FEATURES ─────────────────────────────────────────────────────
print("Computing features...")
ret_3d = closes.pct_change(3)
ret_5d = closes.pct_change(5)
vol_20d_avg = volumes.rolling(20).mean()
vol_ratio = volumes / vol_20d_avg
high_20d = closes.rolling(20).max()
high_50d = closes.rolling(50).max()

# Forward returns for each hold period
fwd_3d = compute_forward_returns(closes, 3)
fwd_5d = compute_forward_returns(closes, 5)
fwd_10d = compute_forward_returns(closes, 10)
fwd_5d_short = compute_forward_returns(closes, 5, short=True)

# ── VARIANT A: Base ─────────────────────────────────────────────────────────
print("\n=== VARIANT A: Base (5d ret >10%, vol >3x, hold 5d) ===")
sig_a = (ret_5d > 0.10) & (vol_ratio > 3.0)
res_a = run_backtest(sig_a, fwd_5d, "A_Base")

# ── VARIANT B: Momentum Chase ──────────────────────────────────────────────
print("=== VARIANT B: Momentum Chase (3d ret >8%, vol >4x, hold 3d) ===")
sig_b = (ret_3d > 0.08) & (vol_ratio > 4.0)
res_b = run_backtest(sig_b, fwd_3d, "B_MomentumChase")

# ── VARIANT C: Extended Hold ────────────────────────────────────────────────
print("=== VARIANT C: Extended Hold (5d ret >10%, vol >3x, hold 10d) ===")
sig_c = (ret_5d > 0.10) & (vol_ratio > 3.0)
res_c = run_backtest(sig_c, fwd_10d, "C_ExtendedHold")

# ── VARIANT D: Price Breakout ───────────────────────────────────────────────
print("=== VARIANT D: Price Breakout (above 20d high + 3x vol, hold 5d) ===")
sig_d = (closes > high_20d.shift(1)) & (vol_ratio > 3.0)
res_d = run_backtest(sig_d, fwd_5d, "D_PriceBreakout")

# ── VARIANT E: Reversal (Short after squeeze) ──────────────────────────────
print("=== VARIANT E: Reversal SHORT (after 5d >15% squeeze, hold 5d) ===")
sig_e = (ret_5d > 0.15)
res_e = run_backtest(sig_e, fwd_5d_short, "E_ReversalShort")

# ── VARIANT F: Multi-signal ────────────────────────────────────────────────
print("=== VARIANT F: Multi-signal (5d >10% + above 50d high + 3x vol, hold 5d) ===")
sig_f = (ret_5d > 0.10) & (closes > high_50d.shift(1)) & (vol_ratio > 3.0)
res_f = run_backtest(sig_f, fwd_5d, "F_MultiSignal")

# ── COLLECT AND DISPLAY ─────────────────────────────────────────────────────
all_results = []
for label, res in [("A", res_a), ("B", res_b), ("C", res_c), ("D", res_d), ("E", res_e), ("F", res_f)]:
    if res is not None:
        all_results.append(res)
        gates = res["gates_passed"]
        status = "PASS" if gates == 5 else f"FAIL ({gates}/5)"
        print(f"\n{'='*60}")
        print(f"  {res['variant']} | {status}")
        print(f"  Trades: {res['n_trades']} | WR: {res['win_rate']}% | Avg Ret: {res['avg_return_pct']}%")
        print(f"  Total Return: {res['total_return_pct']}% | Final Equity: ${res['final_equity']}")
        print(f"  Sharpe: {res['sharpe']} | Sortino: {res['sortino']} | PF: {res['profit_factor']}")
        print(f"  MaxDD: {res['max_drawdown_pct']}%")
        print(f"  Regime: Bull Sharpe={res['sharpe_bull']}, Bear Sharpe={res['sharpe_bear']}, Gap={res['regime_gap']}")
        print(f"  Permutation p-value: {res['perm_p_value']}")
        print(f"  Gates: Sharpe={'✓' if res['gate_sharpe'] else '✗'} | Perm={'✓' if res['gate_perm'] else '✗'} | "
              f"Regime={'✓' if res['gate_regime'] else '✗'} | MaxDD={'✓' if res['gate_maxdd'] else '✗'} | "
              f"Trades={'✓' if res['gate_trades'] else '✗'}")
    else:
        print(f"\n  Variant {label}: NO TRADES generated")

# ── SUMMARY TABLE ───────────────────────────────────────────────────────────
print(f"\n\n{'='*80}")
print(f"{'VARIANT':<22} {'TRADES':>6} {'WR%':>6} {'SHARPE':>7} {'SORTINO':>8} {'PF':>6} {'MaxDD%':>7} {'GATES':>5}")
print(f"{'-'*80}")
for r in all_results:
    gates = f"{r['gates_passed']}/5"
    print(f"{r['variant']:<22} {r['n_trades']:>6} {r['win_rate']:>5.1f}% {r['sharpe']:>7.3f} {r['sortino']:>8.3f} "
          f"{r['profit_factor']:>6.2f} {r['max_drawdown_pct']:>6.1f}% {gates:>5}")
print(f"{'='*80}")

# Best variant
if all_results:
    best = max(all_results, key=lambda x: x["gates_passed"] * 100 + x["sharpe"])
    print(f"\nBest variant: {best['variant']} ({best['gates_passed']}/5 gates, Sharpe={best['sharpe']})")

# ── SAVE RESULTS ────────────────────────────────────────────────────────────
output = {
    "strategy": "Short Squeeze Momentum",
    "run_timestamp": datetime.now().isoformat(),
    "universe_size": len(closes.columns),
    "period": f"{START_DATE} to {END_DATE}",
    "initial_capital": INITIAL_CAPITAL,
    "slippage_per_side": SLIPPAGE_PCT,
    "variants": all_results,
    "best_variant": best["variant"] if all_results else None,
}

# Convert numpy bools to Python bools for JSON serialization
def make_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    return obj

RESULTS_PATH.write_text(json.dumps(make_serializable(output), indent=2))
print(f"\nResults saved to {RESULTS_PATH}")

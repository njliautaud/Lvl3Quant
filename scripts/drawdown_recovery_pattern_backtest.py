#!/usr/bin/env python3
"""
Drawdown Recovery Pattern Backtest
-----------------------------------
Quality stocks recover from drawdowns in predictable patterns.
Wait for CONFIRMATION that recovery has begun, then ride the bounce.

6 Variants (all require stock first dropped >5% from 20-day high):
  A: 50% retracement confirmation, hold 10d
  B: Higher low after dip, hold 10d
  C: SMA(5) recross after 3+ days below, hold 10d
  D: RSI crosses back above 30 from below, hold 10d
  E: First green day after 3+ red days AND >5% dip, hold 5d
  F: 5d return positive while 20d return negative, hold 10d

Universe: 20 quality stocks
OOT: Jan 2022 - Jul 2026
Capital: $645, max $200/trade, max 3 concurrent
Slippage: 0.02% each way
"""

import json
import datetime as dt
import warnings
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────
UNIVERSE = [
    "AAPL", "MSFT", "AVGO", "JPM", "JNJ", "PG", "KO", "PEP", "HD", "COST",
    "UNH", "LLY", "V", "MA", "ABBV", "MRK", "WMT", "AMZN", "GOOGL", "META",
]
START_DATE = "2021-11-01"  # extra history for indicators
OOT_START = "2022-01-03"
OOT_END = "2026-07-31"
INITIAL_CAPITAL = 645.0
MAX_PER_TRADE = 200.0
MAX_CONCURRENT = 3
SLIPPAGE_PCT = 0.0002  # 0.02% each way
RISK_FREE_RATE = 0.04

SPY_TICKER = "SPY"

# ── Data Download ───────────────────────────────────────────────────────
def download_data(tickers):
    """Download daily OHLCV for all tickers + SPY."""
    all_tickers = list(set(tickers + [SPY_TICKER]))
    print(f"Downloading {len(all_tickers)} tickers...")
    data = {}
    for t in all_tickers:
        try:
            df = yf.download(t, start=START_DATE, end=OOT_END, progress=False, auto_adjust=True)
            if df is not None and len(df) > 50:
                # Flatten multi-level columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                data[t] = df
        except Exception as e:
            print(f"  Failed {t}: {e}")
    print(f"  Got data for {len(data)} tickers")
    return data


# ── Indicator Helpers ───────────────────────────────────────────────────
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_drawdown_from_high(close, window=20):
    """Rolling 20-day high and pct drawdown from it."""
    rolling_high = close.rolling(window).max()
    dd_pct = (close - rolling_high) / rolling_high
    return rolling_high, dd_pct


# ── Signal Generation ──────────────────────────────────────────────────
def generate_signals(df, variant):
    """
    Generate entry signals for a given variant.
    All require the stock to have FIRST dropped >5% from 20-day high.
    Returns a boolean Series indexed like df.
    """
    close = df["Close"]
    rolling_high, dd_pct = compute_drawdown_from_high(close, 20)
    in_drawdown = dd_pct < -0.05  # dropped >5% from 20d high

    signals = pd.Series(False, index=df.index)

    if variant == "A":
        # Enter when stock recovers 50% of the drawdown (retracement confirmation)
        rolling_low = close.rolling(20).min()
        drawdown_depth = rolling_high - rolling_low
        recovery_level = rolling_low + 0.5 * drawdown_depth
        # Was in drawdown recently (within last 5 days) and now recovered 50%
        was_in_dd = in_drawdown.rolling(5).max().fillna(0).astype(bool)
        signals = was_in_dd & (close >= recovery_level) & (close < rolling_high * 0.98)

    elif variant == "B":
        # Enter when stock makes a HIGHER LOW after the dip
        # Find local lows in last 10 days, check if current low > previous low
        low = df["Low"]
        prev_low_10d = low.rolling(10).min().shift(1)
        curr_low_5d = low.rolling(5).min()
        was_in_dd = in_drawdown.rolling(10).max().fillna(0).astype(bool)
        # Higher low: current 5d low > previous 10d low, and we were in drawdown
        signals = was_in_dd & (curr_low_5d > prev_low_10d) & (dd_pct < 0)

    elif variant == "C":
        # Enter when stock closes above 5-day SMA after being below it for 3+ days
        sma5 = close.rolling(5).mean()
        below_sma = close < sma5
        # Count consecutive days below SMA
        below_streak = below_sma.astype(int)
        # Rolling sum of last 3 days all below
        was_below_3d = below_sma.shift(1).rolling(3).min().fillna(0).astype(bool)
        above_now = close > sma5
        was_in_dd = in_drawdown.rolling(10).max().fillna(0).astype(bool)
        signals = was_in_dd & was_below_3d & above_now

    elif variant == "D":
        # Enter when RSI crosses back ABOVE 30 from below
        rsi = compute_rsi(close)
        rsi_prev = rsi.shift(1)
        crossed_above_30 = (rsi > 30) & (rsi_prev <= 30)
        was_in_dd = in_drawdown.rolling(10).max().fillna(0).astype(bool)
        signals = was_in_dd & crossed_above_30

    elif variant == "E":
        # First green day after 3+ consecutive red days AND dip >5%
        daily_ret = close.pct_change()
        red_day = daily_ret < 0
        green_day = daily_ret > 0
        # 3+ consecutive red days ending yesterday
        red_streak = red_day.shift(1).rolling(3).min().fillna(0).astype(bool)
        was_in_dd = in_drawdown.rolling(5).max().fillna(0).astype(bool)
        signals = was_in_dd & red_streak & green_day

    elif variant == "F":
        # 5-day return positive WHILE 20-day return still negative
        ret_5d = close.pct_change(5)
        ret_20d = close.pct_change(20)
        was_in_dd = in_drawdown.rolling(10).max().fillna(0).astype(bool)
        signals = was_in_dd & (ret_5d > 0) & (ret_20d < 0)

    return signals


# ── Backtester ──────────────────────────────────────────────────────────
def run_backtest(all_data, variant, hold_days):
    """Run backtest for a single variant across all stocks."""
    trades = []

    for ticker in UNIVERSE:
        if ticker not in all_data:
            continue
        df = all_data[ticker].copy()
        df = df.loc[df.index >= OOT_START]
        if len(df) < 30:
            continue

        # Need full history for indicators
        df_full = all_data[ticker].copy()
        sigs = generate_signals(df_full, variant)
        sigs = sigs.loc[sigs.index >= OOT_START]

        signal_dates = sigs[sigs].index.tolist()

        for entry_date in signal_dates:
            entry_idx = df.index.get_loc(entry_date)
            exit_idx = min(entry_idx + hold_days, len(df) - 1)
            if exit_idx <= entry_idx:
                continue

            entry_price = df["Close"].iloc[entry_idx]
            exit_price = df["Close"].iloc[exit_idx]
            exit_date = df.index[exit_idx]

            # Apply slippage
            entry_cost = entry_price * (1 + SLIPPAGE_PCT)
            exit_proceeds = exit_price * (1 - SLIPPAGE_PCT)

            ret = (exit_proceeds - entry_cost) / entry_cost

            trades.append({
                "ticker": ticker,
                "entry_date": str(entry_date.date()),
                "exit_date": str(exit_date.date()),
                "entry_price": round(float(entry_price), 2),
                "exit_price": round(float(exit_price), 2),
                "return": round(float(ret), 6),
                "hold_days": hold_days,
            })

    return trades


def build_equity_curve(trades, spy_data):
    """Build portfolio equity curve with position limits."""
    if not trades:
        return pd.Series(dtype=float), []

    # Sort trades by entry date
    trades_sorted = sorted(trades, key=lambda t: t["entry_date"])

    # Build daily timeline
    all_dates = spy_data.loc[spy_data.index >= OOT_START].index
    equity = pd.Series(INITIAL_CAPITAL, index=all_dates)
    cash = INITIAL_CAPITAL
    active_positions = []
    executed_trades = []

    for i, date in enumerate(all_dates):
        date_str = str(date.date())

        # Close expired positions
        new_active = []
        for pos in active_positions:
            if date_str >= pos["exit_date"]:
                # Close position
                pnl = pos["shares"] * pos["exit_price"] * (1 - SLIPPAGE_PCT) - pos["cost_basis"]
                cash += pos["cost_basis"] + pnl
            else:
                new_active.append(pos)
        active_positions = new_active

        # Open new positions
        for trade in trades_sorted:
            if trade["entry_date"] == date_str:
                if len(active_positions) >= MAX_CONCURRENT:
                    continue
                position_size = min(MAX_PER_TRADE, cash * 0.95)
                if position_size < 10:
                    continue
                entry_cost = trade["entry_price"] * (1 + SLIPPAGE_PCT)
                shares = position_size / entry_cost
                cost_basis = shares * entry_cost
                cash -= cost_basis
                active_positions.append({
                    "ticker": trade["ticker"],
                    "entry_date": trade["entry_date"],
                    "exit_date": trade["exit_date"],
                    "entry_price": trade["entry_price"],
                    "exit_price": trade["exit_price"],
                    "shares": shares,
                    "cost_basis": cost_basis,
                })
                executed_trades.append(trade)

        # Mark to market
        portfolio_value = cash
        for pos in active_positions:
            # Use current close if available
            ticker_data = None
            if pos["ticker"] in all_data_global:
                ticker_data = all_data_global[pos["ticker"]]
            if ticker_data is not None and date in ticker_data.index:
                current_price = float(ticker_data.loc[date, "Close"])
            else:
                current_price = pos["entry_price"]
            portfolio_value += pos["shares"] * current_price

        equity.iloc[i] = portfolio_value

    return equity, executed_trades


def compute_regime(spy_data):
    """Classify each day as bull/bear/flat based on SPY 50d return."""
    spy_close = spy_data["Close"]
    ret_50d = spy_close.pct_change(50)
    regime = pd.Series("flat", index=spy_data.index)
    regime[ret_50d > 0.05] = "bull"
    regime[ret_50d < -0.05] = "bear"
    return regime


# ── Validation ──────────────────────────────────────────────────────────
def validate_variant(trades, equity, spy_data, variant_name):
    """Run 5-gate validation and compute metrics."""
    if not trades or len(trades) < 2:
        return {
            "variant": variant_name,
            "total_trades": len(trades),
            "passed_all_gates": False,
            "fail_reason": "insufficient trades",
        }

    returns = np.array([t["return"] for t in trades])
    n_trades = len(returns)
    win_rate = float(np.mean(returns > 0))
    avg_return = float(np.mean(returns))
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1) if len(equity) > 0 else 0

    # Daily returns from equity curve
    if len(equity) > 1:
        daily_returns = equity.pct_change().dropna()
        daily_mean = daily_returns.mean()
        daily_std = daily_returns.std()
        ann_sharpe = (daily_mean - RISK_FREE_RATE/252) / daily_std * np.sqrt(252) if daily_std > 0 else 0
        downside = daily_returns[daily_returns < 0].std()
        ann_sortino = (daily_mean - RISK_FREE_RATE/252) / downside * np.sqrt(252) if downside > 0 else 0
    else:
        ann_sharpe = 0
        ann_sortino = 0
        daily_returns = pd.Series(dtype=float)

    # Max drawdown
    if len(equity) > 0:
        cummax = equity.cummax()
        drawdown = (equity - cummax) / cummax
        max_dd = float(drawdown.min())
    else:
        max_dd = 0

    # Profit factor
    gross_profit = float(returns[returns > 0].sum()) if (returns > 0).any() else 0
    gross_loss = float(abs(returns[returns < 0].sum())) if (returns < 0).any() else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Permutation test (10000 shuffles)
    observed_mean = np.mean(returns)
    n_perms = 10000
    rng = np.random.default_rng(42)
    perm_means = np.array([
        np.mean(rng.choice(returns, size=len(returns), replace=True) * rng.choice([-1, 1], size=len(returns)))
        for _ in range(n_perms)
    ])
    perm_p = float(np.mean(perm_means >= observed_mean))

    # Regime analysis
    regime = compute_regime(spy_data)
    bull_trades = [t for t in trades if regime.get(pd.Timestamp(t["entry_date"]), "flat") == "bull"]
    bear_trades = [t for t in trades if regime.get(pd.Timestamp(t["entry_date"]), "flat") == "bear"]

    bull_returns = np.array([t["return"] for t in bull_trades]) if bull_trades else np.array([0])
    bear_returns = np.array([t["return"] for t in bear_trades]) if bear_trades else np.array([0])

    bull_sharpe = float(np.mean(bull_returns) / np.std(bull_returns) * np.sqrt(252/10)) if np.std(bull_returns) > 0 else 0
    bear_sharpe = float(np.mean(bear_returns) / np.std(bear_returns) * np.sqrt(252/10)) if np.std(bear_returns) > 0 else 0

    max_abs = max(abs(bull_sharpe), abs(bear_sharpe), 1e-9)
    regime_gap = abs(bull_sharpe - bear_sharpe) / max_abs

    # 5-Gate checks
    gate_sharpe = ann_sharpe > 0.5
    gate_perm = perm_p < 0.05
    gate_regime = regime_gap < 0.5
    gate_mdd = max_dd > -0.50
    gate_trades = n_trades >= 20

    passed = gate_sharpe and gate_perm and gate_regime and gate_mdd and gate_trades

    fail_reasons = []
    if not gate_sharpe: fail_reasons.append(f"Sharpe={ann_sharpe:.2f}<0.5")
    if not gate_perm: fail_reasons.append(f"perm_p={perm_p:.3f}>=0.05")
    if not gate_regime: fail_reasons.append(f"regime_gap={regime_gap:.2f}>=0.5")
    if not gate_mdd: fail_reasons.append(f"MDD={max_dd:.2%}<=-50%")
    if not gate_trades: fail_reasons.append(f"trades={n_trades}<20")

    result = {
        "variant": variant_name,
        "total_trades": n_trades,
        "win_rate": round(win_rate, 4),
        "avg_return_pct": round(avg_return * 100, 3),
        "total_return_pct": round(total_return * 100, 2),
        "final_equity": round(float(equity.iloc[-1]), 2) if len(equity) > 0 else INITIAL_CAPITAL,
        "sharpe": round(ann_sharpe, 3),
        "sortino": round(ann_sortino, 3),
        "profit_factor": round(profit_factor, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "perm_p_value": round(perm_p, 4),
        "regime_gap": round(regime_gap, 3),
        "bull_trades": len(bull_trades),
        "bear_trades": len(bear_trades),
        "bull_sharpe": round(bull_sharpe, 3),
        "bear_sharpe": round(bear_sharpe, 3),
        "gates": {
            "sharpe_gt_0.5": gate_sharpe,
            "perm_p_lt_0.05": gate_perm,
            "regime_gap_lt_0.5": gate_regime,
            "mdd_gt_neg50": gate_mdd,
            "trades_gte_20": gate_trades,
        },
        "passed_all_gates": passed,
        "fail_reasons": fail_reasons if fail_reasons else None,
    }

    return result


# ── Main ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("DRAWDOWN RECOVERY PATTERN BACKTEST")
    print("=" * 70)
    print(f"Universe: {len(UNIVERSE)} quality stocks")
    print(f"OOT Period: {OOT_START} to {OOT_END}")
    print(f"Capital: ${INITIAL_CAPITAL}, Max/trade: ${MAX_PER_TRADE}, Max concurrent: {MAX_CONCURRENT}")
    print(f"Slippage: {SLIPPAGE_PCT*100:.2f}% each way")
    print()

    # Download data
    all_data = download_data(UNIVERSE)
    all_data_global = all_data  # for equity curve MTM
    spy_data = all_data.get(SPY_TICKER)
    if spy_data is None:
        print("ERROR: Could not download SPY data")
        sys.exit(1)

    variants = {
        "A": ("50% Retracement Confirmation", 10),
        "B": ("Higher Low After Dip", 10),
        "C": ("SMA(5) Recross", 10),
        "D": ("RSI Recovery Above 30", 10),
        "E": ("First Green After 3+ Red Days", 5),
        "F": ("Short-Term Recovery in Longer Drawdown", 10),
    }

    results = []
    for v_key, (v_name, hold_days) in variants.items():
        full_name = f"{v_key}: {v_name}"
        print(f"\n{'─'*60}")
        print(f"Variant {full_name} (hold={hold_days}d)")
        print(f"{'─'*60}")

        trades = run_backtest(all_data, v_key, hold_days)
        print(f"  Raw signals: {len(trades)} trades generated")

        if trades:
            equity, executed = build_equity_curve(trades, spy_data)
            print(f"  Executed trades: {len(executed)} (after position limits)")
            result = validate_variant(executed, equity, spy_data, full_name)
        else:
            equity = pd.Series(INITIAL_CAPITAL, index=spy_data.loc[spy_data.index >= OOT_START].index)
            result = validate_variant([], equity, spy_data, full_name)

        results.append(result)

        # Print summary
        status = "PASS" if result.get("passed_all_gates") else "FAIL"
        print(f"  Status: {status}")
        print(f"  Trades: {result['total_trades']}, WR: {result.get('win_rate', 0):.1%}")
        print(f"  Total Return: {result.get('total_return_pct', 0):.1f}%, Final Equity: ${result.get('final_equity', INITIAL_CAPITAL):.2f}")
        print(f"  Sharpe: {result.get('sharpe', 0):.3f}, Sortino: {result.get('sortino', 0):.3f}, PF: {result.get('profit_factor', 0):.3f}")
        print(f"  MDD: {result.get('max_drawdown_pct', 0):.1f}%, Perm p: {result.get('perm_p_value', 1):.4f}")
        print(f"  Regime gap: {result.get('regime_gap', 0):.3f} (Bull Sharpe: {result.get('bull_sharpe', 0):.3f}, Bear Sharpe: {result.get('bear_sharpe', 0):.3f})")
        if result.get("fail_reasons"):
            print(f"  Fail reasons: {', '.join(result['fail_reasons'])}")

    # Save results
    output = {
        "strategy": "Drawdown Recovery Patterns",
        "run_date": str(dt.datetime.now()),
        "universe_size": len(UNIVERSE),
        "oot_period": f"{OOT_START} to {OOT_END}",
        "initial_capital": INITIAL_CAPITAL,
        "max_per_trade": MAX_PER_TRADE,
        "max_concurrent": MAX_CONCURRENT,
        "slippage_pct": SLIPPAGE_PCT,
        "variants": results,
        "summary": {
            "total_variants": len(results),
            "passed": sum(1 for r in results if r.get("passed_all_gates")),
            "failed": sum(1 for r in results if not r.get("passed_all_gates")),
            "best_sharpe": max((r.get("sharpe", 0) for r in results), default=0),
            "best_variant": max(results, key=lambda r: r.get("sharpe", 0)).get("variant", "none") if results else "none",
        },
    }

    output_path = "/home/jupiter/Lvl3Quant/data/drawdown_recovery_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {output_path}")

    # Final summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    passed = [r for r in results if r.get("passed_all_gates")]
    failed = [r for r in results if not r.get("passed_all_gates")]
    print(f"Passed 5-gate validation: {len(passed)}/{len(results)}")
    if passed:
        print("\nPASSED variants:")
        for r in passed:
            print(f"  {r['variant']}: Sharpe={r['sharpe']:.3f}, Sortino={r['sortino']:.3f}, "
                  f"WR={r['win_rate']:.1%}, PF={r['profit_factor']:.3f}, "
                  f"Return={r['total_return_pct']:.1f}%, MDD={r['max_drawdown_pct']:.1f}%")
    if failed:
        print("\nFAILED variants:")
        for r in failed:
            print(f"  {r['variant']}: {', '.join(r.get('fail_reasons', ['unknown']))}")
    print()

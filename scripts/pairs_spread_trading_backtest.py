#!/usr/bin/env python3
"""
Pairs/Spread Trading Backtest — 6 Variants
Tests mean-reversion in spreads between correlated ETFs.
NOT directional — exploits relative mispricing.

Variants:
  A. XLK/SMH (Tech/Semi)
  B. XLE/USO (Energy equities/Oil)
  C. XLF/KRE (Financials/Regional banks)
  D. TLT/SPY (Bond-Equity ratio)
  E. QQQ/IWD (Growth-Value)
  F. GLD/GDX (Gold/Miners)

5-gate validation per variant:
  1. Sharpe > 0.5
  2. Permutation p < 0.05 (1000 shuffles)
  3. Regime gap < 0.5
  4. Max DD > -50%
  5. >= 20 trades
"""

import json
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
except ImportError:
    print("Installing yfinance...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance", "-q"])
    import yfinance as yf


# ── Config ──────────────────────────────────────────────────────────────
INITIAL_CAPITAL = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02% per leg per trade (so 0.04% round-trip per leg)
ZSCORE_ENTRY = 2.0
ZSCORE_EXIT = 0.5
ZSCORE_STOP = 4.0
LOOKBACK = 20  # rolling window for z-score
START = "2022-01-01"
END = "2026-07-29"
N_PERMUTATIONS = 1000

VARIANTS = {
    "A_XLK_SMH": {"leg1": "XLK", "leg2": "SMH", "name": "Tech Sector (XLK/SMH)", "method": "ratio"},
    "B_XLE_USO": {"leg1": "XLE", "leg2": "USO", "name": "Energy (XLE/USO)", "method": "ratio"},
    "C_XLF_KRE": {"leg1": "XLF", "leg2": "KRE", "name": "Financial (XLF/KRE)", "method": "ratio"},
    "D_TLT_SPY": {"leg1": "TLT", "leg2": "SPY", "name": "Bond-Equity (TLT/SPY)", "method": "ratio"},
    "E_QQQ_IWD": {"leg1": "QQQ", "leg2": "IWD", "name": "Growth-Value (QQQ/IWD)", "method": "ratio"},
    "F_GLD_GDX": {"leg1": "GLD", "leg2": "GDX", "name": "Gold-Miners (GLD/GDX)", "method": "ratio"},
}


def download_data(tickers, start, end):
    """Download adjusted close prices for all tickers."""
    all_tickers = list(set(tickers + ["SPY", "QQQ"]))  # always need SPY for regime, QQQ for correlation
    print(f"Downloading {all_tickers} from {start} to {end}...")
    data = yf.download(all_tickers, start=start, end=end, auto_adjust=True, progress=False)
    # Handle multi-level columns from yfinance
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data
    prices = prices.dropna(how="all").ffill().dropna()
    print(f"  Got {len(prices)} trading days, {prices.shape[1]} tickers")
    return prices


def compute_spread_zscore(prices, leg1, leg2, lookback, method="ratio"):
    """Compute spread and its z-score."""
    if method == "ratio":
        spread = prices[leg1] / prices[leg2]
    else:
        spread = prices[leg1] - prices[leg2]

    roll_mean = spread.rolling(lookback).mean()
    roll_std = spread.rolling(lookback).std()
    zscore = (spread - roll_mean) / roll_std
    return spread, zscore


def run_backtest(prices, leg1, leg2, method, zscore_entry, zscore_exit, zscore_stop,
                 lookback, initial_capital, slippage_pct):
    """
    Run pairs trading backtest.
    Position: dollar-neutral. When z > entry, spread is "expensive" -> short spread (short leg1, long leg2).
              When z < -entry, spread is "cheap" -> long spread (long leg1, short leg2).
    Exit when |z| < exit threshold. Stop loss at |z| > stop threshold.
    """
    spread, zscore = compute_spread_zscore(prices, leg1, leg2, lookback, method)

    capital = initial_capital
    position = 0  # +1 = long spread, -1 = short spread, 0 = flat
    entry_prices = {}  # leg1_price, leg2_price at entry
    trades = []
    equity_curve = [initial_capital]
    dates = []

    valid_idx = zscore.dropna().index
    if len(valid_idx) == 0:
        return [], [initial_capital], [], pd.Series(dtype=float)

    for i, date in enumerate(valid_idx):
        z = zscore.loc[date]
        p1 = prices.loc[date, leg1]
        p2 = prices.loc[date, leg2]

        if np.isnan(z) or np.isnan(p1) or np.isnan(p2):
            equity_curve.append(capital)
            dates.append(date)
            continue

        if position == 0:
            # Entry conditions
            if z > zscore_entry:
                # Spread expensive -> short spread: short leg1, long leg2
                position = -1
                entry_prices = {"p1": p1, "p2": p2, "date": date, "z": z}
                # Slippage on entry (2 legs)
                capital -= capital * slippage_pct * 2
            elif z < -zscore_entry:
                # Spread cheap -> long spread: long leg1, short leg2
                position = 1
                entry_prices = {"p1": p1, "p2": p2, "date": date, "z": z}
                capital -= capital * slippage_pct * 2
        else:
            # PnL from spread movement (dollar-neutral: equal $ in each leg)
            # Allocate half capital to each leg
            half_cap = capital / 2.0
            if position == 1:
                # Long leg1, short leg2
                ret1 = (p1 - entry_prices["p1"]) / entry_prices["p1"]
                ret2 = (p2 - entry_prices["p2"]) / entry_prices["p2"]
                daily_pnl = half_cap * ret1 - half_cap * ret2
            else:
                # Short leg1, long leg2
                ret1 = (p1 - entry_prices["p1"]) / entry_prices["p1"]
                ret2 = (p2 - entry_prices["p2"]) / entry_prices["p2"]
                daily_pnl = -half_cap * ret1 + half_cap * ret2

            # Check exit conditions
            should_exit = False
            exit_reason = ""

            if abs(z) < zscore_exit:
                should_exit = True
                exit_reason = "convergence"
            elif abs(z) > zscore_stop:
                should_exit = True
                exit_reason = "stop_loss"

            if should_exit:
                # Apply slippage on exit
                trade_pnl = daily_pnl - capital * slippage_pct * 2
                capital += trade_pnl
                trades.append({
                    "entry_date": str(entry_prices["date"]),
                    "exit_date": str(date),
                    "entry_z": float(entry_prices["z"]),
                    "exit_z": float(z),
                    "direction": "long_spread" if position == 1 else "short_spread",
                    "pnl": float(trade_pnl),
                    "pnl_pct": float(trade_pnl / capital) if capital > 0 else 0,
                    "exit_reason": exit_reason,
                })
                position = 0
                entry_prices = {}
            else:
                # Mark-to-market: update entry to current for next day's calc
                # Actually, we track cumulative PnL differently:
                # Just update capital with the spread movement
                pass

        equity_curve.append(capital)
        dates.append(date)

    # If still in position at end, close it
    if position != 0 and len(valid_idx) > 0:
        date = valid_idx[-1]
        p1 = prices.loc[date, leg1]
        p2 = prices.loc[date, leg2]
        half_cap = capital / 2.0
        if position == 1:
            ret1 = (p1 - entry_prices["p1"]) / entry_prices["p1"]
            ret2 = (p2 - entry_prices["p2"]) / entry_prices["p2"]
            daily_pnl = half_cap * ret1 - half_cap * ret2
        else:
            ret1 = (p1 - entry_prices["p1"]) / entry_prices["p1"]
            ret2 = (p2 - entry_prices["p2"]) / entry_prices["p2"]
            daily_pnl = -half_cap * ret1 + half_cap * ret2
        trade_pnl = daily_pnl - capital * slippage_pct * 2
        capital += trade_pnl
        trades.append({
            "entry_date": str(entry_prices["date"]),
            "exit_date": str(date),
            "entry_z": float(entry_prices["z"]),
            "exit_z": float(zscore.loc[date]),
            "direction": "long_spread" if position == 1 else "short_spread",
            "pnl": float(trade_pnl),
            "pnl_pct": float(trade_pnl / capital) if capital > 0 else 0,
            "exit_reason": "end_of_data",
        })

    # Build proper equity curve from trades
    # Simpler approach: reconstruct daily equity from trade-level PnL
    eq = build_daily_equity(prices, leg1, leg2, method, zscore, lookback,
                            zscore_entry, zscore_exit, zscore_stop,
                            initial_capital, slippage_pct)

    return trades, eq["equity"].tolist(), eq.index.tolist(), eq["equity"]


def build_daily_equity(prices, leg1, leg2, method, zscore, lookback,
                       zscore_entry, zscore_exit, zscore_stop,
                       initial_capital, slippage_pct):
    """Build daily equity curve with proper mark-to-market."""
    valid_idx = zscore.dropna().index
    equity = initial_capital
    position = 0
    entry_p1 = entry_p2 = 0
    prev_p1 = prev_p2 = 0
    records = []

    for date in valid_idx:
        z = zscore.loc[date]
        p1 = prices.loc[date, leg1]
        p2 = prices.loc[date, leg2]

        if np.isnan(z) or np.isnan(p1) or np.isnan(p2):
            records.append({"date": date, "equity": equity})
            continue

        # Daily P&L from position
        if position != 0 and prev_p1 > 0 and prev_p2 > 0:
            half = equity / 2.0
            r1 = (p1 - prev_p1) / prev_p1
            r2 = (p2 - prev_p2) / prev_p2
            if position == 1:  # long spread
                equity += half * r1 - half * r2
            else:  # short spread
                equity += -half * r1 + half * r2

        # Check exit
        if position != 0:
            if abs(z) < zscore_exit or abs(z) > zscore_stop:
                equity -= equity * slippage_pct * 2  # exit slippage
                position = 0

        # Check entry
        if position == 0:
            if z > zscore_entry:
                position = -1
                equity -= equity * slippage_pct * 2
            elif z < -zscore_entry:
                position = 1
                equity -= equity * slippage_pct * 2

        prev_p1 = p1
        prev_p2 = p2
        records.append({"date": date, "equity": equity})

    df = pd.DataFrame(records).set_index("date")
    return df


def compute_metrics(equity_series, trades, initial_capital):
    """Compute Sharpe, Sortino, max drawdown, etc."""
    if len(equity_series) < 2:
        return {"sharpe": 0, "sortino": 0, "max_dd_pct": -100, "total_return_pct": 0,
                "n_trades": 0, "win_rate": 0, "profit_factor": 0, "final_equity": initial_capital}

    returns = equity_series.pct_change().dropna()
    if len(returns) == 0 or returns.std() == 0:
        return {"sharpe": 0, "sortino": 0, "max_dd_pct": -100, "total_return_pct": 0,
                "n_trades": len(trades), "win_rate": 0, "profit_factor": 0,
                "final_equity": float(equity_series.iloc[-1])}

    sharpe = returns.mean() / returns.std() * np.sqrt(252)
    downside = returns[returns < 0]
    sortino = returns.mean() / downside.std() * np.sqrt(252) if len(downside) > 0 and downside.std() > 0 else 0

    # Max drawdown
    cummax = equity_series.cummax()
    drawdown = (equity_series - cummax) / cummax
    max_dd = drawdown.min()

    # Trade stats
    n_trades = len(trades)
    winners = [t for t in trades if t["pnl"] > 0]
    losers = [t for t in trades if t["pnl"] <= 0]
    win_rate = len(winners) / n_trades if n_trades > 0 else 0
    gross_profit = sum(t["pnl"] for t in winners) if winners else 0
    gross_loss = abs(sum(t["pnl"] for t in losers)) if losers else 0.001
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

    total_return = (equity_series.iloc[-1] - initial_capital) / initial_capital * 100

    return {
        "sharpe": round(float(sharpe), 4),
        "sortino": round(float(sortino), 4),
        "max_dd_pct": round(float(max_dd * 100), 2),
        "total_return_pct": round(float(total_return), 2),
        "n_trades": n_trades,
        "win_rate": round(float(win_rate), 4),
        "profit_factor": round(float(profit_factor), 4),
        "final_equity": round(float(equity_series.iloc[-1]), 2),
    }


def permutation_test(prices, leg1, leg2, method, lookback, zscore_entry, zscore_exit,
                     zscore_stop, initial_capital, slippage_pct, actual_sharpe, n_perms=1000):
    """Shuffle entry signals and compare Sharpe to actual."""
    spread, zscore = compute_spread_zscore(prices, leg1, leg2, lookback, method)
    valid_idx = zscore.dropna().index
    z_values = zscore.loc[valid_idx].values.copy()

    perm_sharpes = []
    for _ in range(n_perms):
        # Shuffle z-score values (breaks temporal structure)
        shuffled_z = z_values.copy()
        np.random.shuffle(shuffled_z)
        shuffled_zscore = pd.Series(shuffled_z, index=valid_idx)

        eq = _run_quick_backtest(prices, leg1, leg2, shuffled_zscore, valid_idx,
                                 zscore_entry, zscore_exit, zscore_stop,
                                 initial_capital, slippage_pct)
        if len(eq) > 1:
            rets = pd.Series(eq).pct_change().dropna()
            if rets.std() > 0:
                s = rets.mean() / rets.std() * np.sqrt(252)
                perm_sharpes.append(s)
            else:
                perm_sharpes.append(0)
        else:
            perm_sharpes.append(0)

    if len(perm_sharpes) == 0:
        return 1.0
    p_value = np.mean([s >= actual_sharpe for s in perm_sharpes])
    return float(p_value)


def _run_quick_backtest(prices, leg1, leg2, zscore, valid_idx,
                        zscore_entry, zscore_exit, zscore_stop,
                        initial_capital, slippage_pct):
    """Fast backtest using pre-computed z-score series."""
    equity = initial_capital
    position = 0
    prev_p1 = prev_p2 = 0
    eq_list = [equity]

    for date in valid_idx:
        z = zscore.loc[date]
        p1 = prices.loc[date, leg1]
        p2 = prices.loc[date, leg2]

        if np.isnan(z) or np.isnan(p1) or np.isnan(p2):
            eq_list.append(equity)
            continue

        if position != 0 and prev_p1 > 0 and prev_p2 > 0:
            half = equity / 2.0
            r1 = (p1 - prev_p1) / prev_p1
            r2 = (p2 - prev_p2) / prev_p2
            if position == 1:
                equity += half * r1 - half * r2
            else:
                equity += -half * r1 + half * r2

        if position != 0:
            if abs(z) < zscore_exit or abs(z) > zscore_stop:
                equity -= equity * slippage_pct * 2
                position = 0

        if position == 0:
            if z > zscore_entry:
                position = -1
                equity -= equity * slippage_pct * 2
            elif z < -zscore_entry:
                position = 1
                equity -= equity * slippage_pct * 2

        prev_p1 = p1
        prev_p2 = p2
        eq_list.append(equity)

    return eq_list


def regime_split_sharpe(equity_series, spy_prices, lookback_sma=200):
    """Split Sharpe by bull (SPY > 200-SMA) vs bear regime."""
    spy_sma = spy_prices.rolling(lookback_sma).mean()
    bull_mask = spy_prices > spy_sma
    bear_mask = spy_prices <= spy_sma

    returns = equity_series.pct_change().dropna()
    # Align
    common = returns.index.intersection(bull_mask.dropna().index)
    if len(common) < 20:
        return 0, 0, 1.0

    returns_aligned = returns.loc[common]
    bull_aligned = bull_mask.loc[common]

    bull_rets = returns_aligned[bull_aligned]
    bear_rets = returns_aligned[~bull_aligned]

    bull_sharpe = (bull_rets.mean() / bull_rets.std() * np.sqrt(252)) if len(bull_rets) > 5 and bull_rets.std() > 0 else 0
    bear_sharpe = (bear_rets.mean() / bear_rets.std() * np.sqrt(252)) if len(bear_rets) > 5 and bear_rets.std() > 0 else 0

    max_abs = max(abs(bull_sharpe), abs(bear_sharpe), 0.001)
    regime_gap = abs(bull_sharpe - bear_sharpe) / max_abs

    return float(bull_sharpe), float(bear_sharpe), float(regime_gap)


def compute_qqq_correlation(equity_series, qqq_prices):
    """Compute correlation of strategy returns with QQQ returns."""
    strat_rets = equity_series.pct_change().dropna()
    qqq_rets = qqq_prices.pct_change().dropna()
    common = strat_rets.index.intersection(qqq_rets.index)
    if len(common) < 20:
        return 0
    return float(strat_rets.loc[common].corr(qqq_rets.loc[common]))


def main():
    print("=" * 70)
    print("PAIRS/SPREAD TRADING BACKTEST — 6 VARIANTS")
    print(f"Capital: ${INITIAL_CAPITAL} | Z-entry: {ZSCORE_ENTRY} | Z-exit: {ZSCORE_EXIT} | Z-stop: {ZSCORE_STOP}")
    print(f"Lookback: {LOOKBACK}d | Slippage: {SLIPPAGE_PCT*100:.2f}% per leg")
    print("=" * 70)

    # Collect all tickers
    all_tickers = set()
    for v in VARIANTS.values():
        all_tickers.add(v["leg1"])
        all_tickers.add(v["leg2"])
    all_tickers.add("SPY")
    all_tickers.add("QQQ")

    prices = download_data(list(all_tickers), START, END)

    results = {}

    for variant_key, variant in VARIANTS.items():
        leg1 = variant["leg1"]
        leg2 = variant["leg2"]
        name = variant["name"]
        method = variant["method"]

        print(f"\n{'─' * 60}")
        print(f"Variant {variant_key}: {name}")
        print(f"{'─' * 60}")

        if leg1 not in prices.columns or leg2 not in prices.columns:
            print(f"  SKIPPED: Missing data for {leg1} or {leg2}")
            results[variant_key] = {"status": "SKIPPED", "reason": f"Missing {leg1} or {leg2}"}
            continue

        # Run backtest
        trades, eq_list, dates, equity_series = run_backtest(
            prices, leg1, leg2, method,
            ZSCORE_ENTRY, ZSCORE_EXIT, ZSCORE_STOP,
            LOOKBACK, INITIAL_CAPITAL, SLIPPAGE_PCT
        )

        if len(equity_series) < 10:
            print(f"  SKIPPED: Insufficient data")
            results[variant_key] = {"status": "SKIPPED", "reason": "Insufficient data"}
            continue

        # Metrics
        metrics = compute_metrics(equity_series, trades, INITIAL_CAPITAL)
        print(f"  Trades: {metrics['n_trades']} | Sharpe: {metrics['sharpe']:.3f} | "
              f"Sortino: {metrics['sortino']:.3f}")
        print(f"  WR: {metrics['win_rate']:.1%} | PF: {metrics['profit_factor']:.2f} | "
              f"Return: {metrics['total_return_pct']:.1f}%")
        print(f"  Max DD: {metrics['max_dd_pct']:.1f}% | Final: ${metrics['final_equity']:.2f}")

        # Regime analysis
        spy_prices = prices["SPY"].loc[equity_series.index[0]:equity_series.index[-1]]
        bull_sharpe, bear_sharpe, regime_gap = regime_split_sharpe(equity_series, spy_prices)
        print(f"  Bull Sharpe: {bull_sharpe:.3f} | Bear Sharpe: {bear_sharpe:.3f} | Gap: {regime_gap:.3f}")

        # QQQ correlation
        qqq_corr = compute_qqq_correlation(equity_series, prices["QQQ"])
        print(f"  QQQ correlation: {qqq_corr:.3f}")

        # Permutation test (only if Sharpe > 0)
        if metrics["sharpe"] > 0:
            print(f"  Running permutation test ({N_PERMUTATIONS} shuffles)...", end=" ", flush=True)
            p_value = permutation_test(
                prices, leg1, leg2, method, LOOKBACK,
                ZSCORE_ENTRY, ZSCORE_EXIT, ZSCORE_STOP,
                INITIAL_CAPITAL, SLIPPAGE_PCT,
                metrics["sharpe"], N_PERMUTATIONS
            )
            print(f"p = {p_value:.4f}")
        else:
            p_value = 1.0
            print(f"  Permutation test skipped (Sharpe <= 0)")

        # 5-Gate Validation
        gates = {
            "sharpe_gt_0.5": metrics["sharpe"] > 0.5,
            "perm_p_lt_0.05": p_value < 0.05,
            "regime_gap_lt_0.5": regime_gap < 0.5,
            "max_dd_gt_neg50": metrics["max_dd_pct"] > -50,
            "trades_gte_20": metrics["n_trades"] >= 20,
        }
        gates_passed = sum(gates.values())

        print(f"\n  5-GATE VALIDATION: {gates_passed}/5 PASSED")
        for gate_name, passed in gates.items():
            status = "PASS" if passed else "FAIL"
            print(f"    [{status}] {gate_name}")

        verdict = "PASS" if gates_passed == 5 else "FAIL"
        print(f"  VERDICT: {verdict}")

        results[variant_key] = {
            "name": name,
            "leg1": leg1,
            "leg2": leg2,
            "metrics": metrics,
            "bull_sharpe": round(bull_sharpe, 4),
            "bear_sharpe": round(bear_sharpe, 4),
            "regime_gap": round(regime_gap, 4),
            "qqq_correlation": round(qqq_corr, 4),
            "permutation_p_value": round(p_value, 4),
            "gates": {k: bool(v) for k, v in gates.items()},
            "gates_passed": gates_passed,
            "verdict": verdict,
            "n_convergence_exits": len([t for t in trades if t["exit_reason"] == "convergence"]),
            "n_stop_exits": len([t for t in trades if t["exit_reason"] == "stop_loss"]),
        }

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Variant':<25} {'Sharpe':>7} {'Gates':>6} {'Verdict':>8} {'QQQ_r':>7} {'Return%':>8}")
    print("-" * 70)
    for k, v in results.items():
        if "metrics" in v:
            m = v["metrics"]
            print(f"{v['name']:<25} {m['sharpe']:>7.3f} {v['gates_passed']:>4}/5 {v['verdict']:>8} "
                  f"{v['qqq_correlation']:>7.3f} {m['total_return_pct']:>7.1f}%")
        else:
            print(f"{k:<25} {'N/A':>7} {'N/A':>6} {'SKIP':>8}")

    best = None
    best_sharpe = -999
    for k, v in results.items():
        if "metrics" in v and v["metrics"]["sharpe"] > best_sharpe:
            best_sharpe = v["metrics"]["sharpe"]
            best = k

    if best:
        print(f"\nBest Sharpe: {results[best]['name']} at {best_sharpe:.3f}")
        full_pass = [k for k, v in results.items() if v.get("gates_passed", 0) == 5]
        if full_pass:
            print(f"FULL 5/5 PASS: {[results[k]['name'] for k in full_pass]}")
        else:
            print("No variant passed all 5 gates.")

    # Save results
    output_path = "/home/jupiter/Lvl3Quant/data/pairs_spread_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "config": {
                "initial_capital": INITIAL_CAPITAL,
                "zscore_entry": ZSCORE_ENTRY,
                "zscore_exit": ZSCORE_EXIT,
                "zscore_stop": ZSCORE_STOP,
                "lookback": LOOKBACK,
                "slippage_pct": SLIPPAGE_PCT,
                "period": f"{START} to {END}",
                "n_permutations": N_PERMUTATIONS,
            },
            "results": results,
        }, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

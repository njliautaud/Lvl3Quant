#!/usr/bin/env python3
"""
Commodity-to-Equity Lead-Lag Backtest
=====================================
Exploits structural delay: commodity prices move first (24h futures markets),
equity sectors follow as analysts update earnings models.

6 Variants:
  A) Gold→Miners      B) Oil→Energy       C) Multi-commodity
  D) Commodity momentum  E) Contrarian commodity  F) Gold fear signal

5-Gate Validation:
  1. Sharpe > 0.5
  2. Permutation p < 0.05 (1000 iter)
  3. Regime gap < 0.5
  4. MaxDD > -50%
  5. >= 20 trades
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

# ─── CONFIG ───────────────────────────────────────────────────────────────────
TICKERS = ["GLD", "GDX", "SLV", "SIL", "USO", "XLE", "WEAT", "XLP", "UNG", "XLU", "SPY", "^VIX"]
START_DATE = "2006-01-01"
END_DATE = "2026-07-29"
OOT_START = "2022-01-01"
INITIAL_CAPITAL = 645.0
SLIPPAGE_BPS = 0.0002  # 0.02% each way → 0.04% round-trip
N_PERMUTATIONS = 1000
RESULTS_PATH = Path("/home/jupiter/Lvl3Quant/data/commodity_equity_leadlag_results.json")


# ─── DATA DOWNLOAD ───────────────────────────────────────────────────────────
def download_data():
    print("Downloading data...")
    data = {}
    for t in TICKERS:
        try:
            df = yf.download(t, start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) > 100:
                data[t] = df["Close"].copy()
                print(f"  {t}: {len(df)} rows, {df.index[0].date()} → {df.index[-1].date()}")
            else:
                print(f"  {t}: insufficient data ({len(df)} rows)")
        except Exception as e:
            print(f"  {t}: FAILED - {e}")
    prices = pd.DataFrame(data).ffill().dropna(how="all")
    print(f"Combined: {len(prices)} rows, {prices.columns.tolist()}\n")
    return prices


# ─── BACKTEST ENGINE ──────────────────────────────────────────────────────────
def backtest_strategy(prices, signal_func, oot_start=OOT_START,
                      initial_capital=INITIAL_CAPITAL, slippage=SLIPPAGE_BPS):
    """
    Generic event-driven backtest.
    signal_func(prices, i) -> list of (ticker, hold_days, direction) or empty list.
    direction: +1 = long, -1 = short, 0 = go to cash (sell everything).
    """
    oot_mask = prices.index >= pd.Timestamp(oot_start)
    oot_prices = prices[oot_mask].copy()

    if len(oot_prices) < 50:
        return None

    capital = initial_capital
    equity_curve = [capital]
    dates = [oot_prices.index[0]]
    trades = []
    positions = {}  # ticker -> {entry_price, entry_date, exit_date, direction, shares}

    for i in range(1, len(oot_prices)):
        today = oot_prices.index[i]
        yesterday = oot_prices.index[i - 1]

        # Close expired positions
        closed = []
        for tk, pos in list(positions.items()):
            if today >= pos["exit_date"]:
                exit_price = oot_prices[tk].iloc[i]
                entry_price = pos["entry_price"]
                gross_ret = (exit_price / entry_price - 1) * pos["direction"]
                net_ret = gross_ret - 2 * slippage  # entry + exit slippage
                pnl = pos["notional"] * net_ret
                capital += pos["notional"] + pnl
                trades.append({
                    "ticker": tk,
                    "entry_date": str(pos["entry_date"].date()),
                    "exit_date": str(today.date()),
                    "direction": pos["direction"],
                    "gross_ret": gross_ret,
                    "net_ret": net_ret,
                    "pnl": pnl,
                })
                closed.append(tk)
        for tk in closed:
            del positions[tk]

        # Check for new signals (use full price history up to today for lookback)
        full_idx = prices.index.get_loc(today)
        signals = signal_func(prices, full_idx)

        for sig in signals:
            tk, hold_days, direction = sig

            if direction == 0:
                # Go-to-cash signal: close all positions immediately
                for ptk, pos in list(positions.items()):
                    exit_price = oot_prices[ptk].iloc[i]
                    gross_ret = (exit_price / pos["entry_price"] - 1) * pos["direction"]
                    net_ret = gross_ret - 2 * slippage
                    pnl = pos["notional"] * net_ret
                    capital += pos["notional"] + pnl
                    trades.append({
                        "ticker": ptk,
                        "entry_date": str(pos["entry_date"].date()),
                        "exit_date": str(today.date()),
                        "direction": pos["direction"],
                        "gross_ret": gross_ret,
                        "net_ret": net_ret,
                        "pnl": pnl,
                    })
                positions.clear()
                # Mark cash period - no new entries for hold_days
                continue

            if tk in positions or tk not in oot_prices.columns:
                continue

            # Size: equal-weight, use available capital
            if capital <= 0:
                continue

            alloc = capital * 0.95  # keep 5% reserve
            if len(signals) > 1:
                alloc = alloc / len(signals)

            entry_price = oot_prices[tk].iloc[i] * (1 + slippage * direction)
            shares = alloc / entry_price
            exit_idx = min(i + hold_days, len(oot_prices) - 1)
            exit_date = oot_prices.index[exit_idx]

            positions[tk] = {
                "entry_price": oot_prices[tk].iloc[i],
                "entry_date": today,
                "exit_date": exit_date,
                "direction": direction,
                "shares": shares,
                "notional": alloc,
            }
            capital -= alloc

        # Mark-to-market
        mtm = capital
        for tk, pos in positions.items():
            curr_price = oot_prices[tk].iloc[i]
            gross_ret = (curr_price / pos["entry_price"] - 1) * pos["direction"]
            mtm += pos["notional"] * (1 + gross_ret)

        equity_curve.append(mtm)
        dates.append(today)

    # Close any remaining positions at end
    for tk, pos in positions.items():
        exit_price = oot_prices[tk].iloc[-1]
        gross_ret = (exit_price / pos["entry_price"] - 1) * pos["direction"]
        net_ret = gross_ret - 2 * slippage
        pnl = pos["notional"] * net_ret
        capital += pos["notional"] + pnl
        trades.append({
            "ticker": tk,
            "entry_date": str(pos["entry_date"].date()),
            "exit_date": str(oot_prices.index[-1].date()),
            "direction": pos["direction"],
            "gross_ret": gross_ret,
            "net_ret": net_ret,
            "pnl": pnl,
        })

    eq = pd.Series(equity_curve, index=dates)
    return eq, trades


# ─── STRATEGY SIGNAL FUNCTIONS ───────────────────────────────────────────────
def signal_gold_miners(prices, i):
    """A) Gold→Miners: GLD 5d ret > 3% → buy GDX 10d"""
    if i < 5:
        return []
    gld_ret = prices["GLD"].iloc[i] / prices["GLD"].iloc[i - 5] - 1
    if gld_ret > 0.03:
        return [("GDX", 10, 1)]
    return []


def signal_oil_energy(prices, i):
    """B) Oil→Energy: USO 5d ret > 5% → buy XLE 10d"""
    if i < 5:
        return []
    uso_ret = prices["USO"].iloc[i] / prices["USO"].iloc[i - 5] - 1
    if uso_ret > 0.05:
        return [("XLE", 10, 1)]
    return []


def signal_multi_commodity(prices, i):
    """C) Multi-commodity: combine gold+oil+silver signals"""
    if i < 5:
        return []
    signals = []
    gld_ret = prices["GLD"].iloc[i] / prices["GLD"].iloc[i - 5] - 1
    uso_ret = prices["USO"].iloc[i] / prices["USO"].iloc[i - 5] - 1
    slv_ret = prices["SLV"].iloc[i] / prices["SLV"].iloc[i - 5] - 1
    if gld_ret > 0.03:
        signals.append(("GDX", 10, 1))
    if uso_ret > 0.05:
        signals.append(("XLE", 10, 1))
    if slv_ret > 0.04:
        signals.append(("SIL", 10, 1))
    return signals


def signal_commodity_momentum(prices, i):
    """D) Commodity momentum: all 3 commodities above 20-SMA → buy SPY 10d"""
    if i < 20:
        return []
    gld_sma = prices["GLD"].iloc[i - 19:i + 1].mean()
    uso_sma = prices["USO"].iloc[i - 19:i + 1].mean()
    slv_sma = prices["SLV"].iloc[i - 19:i + 1].mean()
    if (prices["GLD"].iloc[i] > gld_sma and
        prices["USO"].iloc[i] > uso_sma and
        prices["SLV"].iloc[i] > slv_sma):
        return [("SPY", 10, 1)]
    return []


def signal_contrarian_oil(prices, i):
    """E) Contrarian commodity: USO drops >10% in 10d → buy XLE 20d"""
    if i < 10:
        return []
    uso_ret = prices["USO"].iloc[i] / prices["USO"].iloc[i - 10] - 1
    if uso_ret < -0.10:
        return [("XLE", 20, 1)]
    return []


def signal_gold_fear(prices, i):
    """F) Gold fear signal: GLD outperforms SPY by >3% in 10d → go cash 20d"""
    if i < 10:
        return []
    gld_ret = prices["GLD"].iloc[i] / prices["GLD"].iloc[i - 10] - 1
    spy_ret = prices["SPY"].iloc[i] / prices["SPY"].iloc[i - 10] - 1
    if (gld_ret - spy_ret) > 0.03:
        return [("SPY", 20, 0)]  # direction=0 means go to cash
    return []


# ─── METRICS ──────────────────────────────────────────────────────────────────
def compute_metrics(eq, trades, initial_capital=INITIAL_CAPITAL):
    daily_ret = eq.pct_change().dropna()
    if len(daily_ret) == 0:
        return None

    ann_ret = (eq.iloc[-1] / eq.iloc[0]) ** (252 / len(eq)) - 1
    ann_vol = daily_ret.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    downside = daily_ret[daily_ret < 0].std() * np.sqrt(252)
    sortino = ann_ret / downside if downside > 0 else 0

    # Max drawdown
    peak = eq.cummax()
    dd = (eq - peak) / peak
    max_dd = dd.min()

    # Win rate and profit factor
    if trades:
        wins = [t for t in trades if t["pnl"] > 0]
        losses = [t for t in trades if t["pnl"] <= 0]
        wr = len(wins) / len(trades)
        gross_profit = sum(t["pnl"] for t in wins) if wins else 0
        gross_loss = abs(sum(t["pnl"] for t in losses)) if losses else 1e-9
        pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    else:
        wr = 0
        pf = 0

    total_ret = eq.iloc[-1] / eq.iloc[0] - 1

    return {
        "total_return_pct": round(total_ret * 100, 2),
        "ann_return_pct": round(ann_ret * 100, 2),
        "ann_vol_pct": round(ann_vol * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_dd_pct": round(max_dd * 100, 2),
        "win_rate": round(wr, 3),
        "profit_factor": round(pf, 3),
        "n_trades": len(trades),
        "final_equity": round(eq.iloc[-1], 2),
    }


# ─── VALIDATION GATES ────────────────────────────────────────────────────────
def permutation_test(prices, signal_func, actual_sharpe, n_perms=N_PERMUTATIONS):
    """Shuffle signal dates and recompute Sharpe n_perms times."""
    count_better = 0
    oot_prices = prices[prices.index >= pd.Timestamp(OOT_START)]

    # Get actual signal days
    actual_returns = []
    for i in range(20, len(prices)):
        if prices.index[i] < pd.Timestamp(OOT_START):
            continue
        sigs = signal_func(prices, i)
        if sigs:
            for tk, hold, direction in sigs:
                if direction == 0 or tk not in prices.columns:
                    continue
                oot_i = oot_prices.index.get_loc(prices.index[i])
                exit_i = min(oot_i + hold, len(oot_prices) - 1)
                ret = (oot_prices[tk].iloc[exit_i] / oot_prices[tk].iloc[oot_i] - 1) * direction
                actual_returns.append(ret)

    if len(actual_returns) < 5:
        return 1.0  # Not enough trades

    actual_mean = np.mean(actual_returns)

    # Permutation: randomly sample same number of entry points
    rng = np.random.RandomState(42)
    for _ in range(n_perms):
        perm_returns = []
        random_indices = rng.randint(20, len(oot_prices) - 20, size=len(actual_returns))
        for idx in random_indices:
            # Pick random ticker from the strategy's universe
            tk = rng.choice([t for t in ["GDX", "XLE", "SIL", "SPY"] if t in oot_prices.columns])
            hold = 10
            ret = (oot_prices[tk].iloc[min(idx + hold, len(oot_prices) - 1)] /
                   oot_prices[tk].iloc[idx] - 1)
            perm_returns.append(ret)
        if np.mean(perm_returns) >= actual_mean:
            count_better += 1

    return count_better / n_perms


def regime_analysis(eq, prices):
    """Compute Sharpe in bull vs bear regime (SPY vs 200-SMA)."""
    spy = prices["SPY"]
    spy_sma200 = spy.rolling(200).mean()

    common_idx = eq.index.intersection(spy.index)
    if len(common_idx) < 50:
        return 0, 0, 1.0

    daily_ret = eq.reindex(common_idx).pct_change().dropna()
    spy_at = spy.reindex(daily_ret.index)
    sma_at = spy_sma200.reindex(daily_ret.index)

    bull = daily_ret[spy_at > sma_at]
    bear = daily_ret[spy_at <= sma_at]

    sharpe_bull = (bull.mean() / bull.std() * np.sqrt(252)) if len(bull) > 20 and bull.std() > 0 else 0
    sharpe_bear = (bear.mean() / bear.std() * np.sqrt(252)) if len(bear) > 20 and bear.std() > 0 else 0

    max_abs = max(abs(sharpe_bull), abs(sharpe_bear), 1e-9)
    gap = abs(sharpe_bull - sharpe_bear) / max_abs

    return round(sharpe_bull, 3), round(sharpe_bear, 3), round(gap, 3)


def validate_strategy(name, eq, trades, prices, signal_func):
    """Run 5-gate validation."""
    metrics = compute_metrics(eq, trades)
    if metrics is None:
        return None

    # Gate 1: Sharpe > 0.5
    g1 = metrics["sharpe"] > 0.5

    # Gate 2: Permutation p < 0.05
    print(f"  Running permutation test ({N_PERMUTATIONS} iterations)...")
    p_val = permutation_test(prices, signal_func, metrics["sharpe"])
    g2 = p_val < 0.05

    # Gate 3: Regime gap < 0.5
    sharpe_bull, sharpe_bear, gap = regime_analysis(eq, prices)
    g3 = gap < 0.5

    # Gate 4: MaxDD > -50%
    g4 = metrics["max_dd_pct"] > -50

    # Gate 5: >= 20 trades
    g5 = metrics["n_trades"] >= 20

    gates_passed = sum([g1, g2, g3, g4, g5])

    result = {
        "variant": name,
        "metrics": metrics,
        "gates": {
            "G1_sharpe_gt_0.5": {"pass": g1, "value": metrics["sharpe"]},
            "G2_perm_p_lt_0.05": {"pass": g2, "value": round(p_val, 4)},
            "G3_regime_gap_lt_0.5": {"pass": g3, "value": gap,
                                      "sharpe_bull": sharpe_bull, "sharpe_bear": sharpe_bear},
            "G4_maxdd_gt_neg50": {"pass": g4, "value": metrics["max_dd_pct"]},
            "G5_min_20_trades": {"pass": g5, "value": metrics["n_trades"]},
        },
        "gates_passed": f"{gates_passed}/5",
        "status": "PASS" if gates_passed == 5 else "FAIL",
    }
    return result


# ─── VARIANT F SPECIAL HANDLING ───────────────────────────────────────────────
def backtest_fear_signal(prices, initial_capital=INITIAL_CAPITAL, slippage=SLIPPAGE_BPS):
    """
    Variant F: Gold fear = go to cash. Default = hold SPY.
    Benchmark against buy-and-hold SPY.
    """
    oot_mask = prices.index >= pd.Timestamp(OOT_START)
    oot_prices = prices[oot_mask].copy()

    capital = initial_capital
    spy_shares = capital / (oot_prices["SPY"].iloc[0] * (1 + slippage))
    capital_after_buy = 0

    in_cash = False
    cash_until = None
    equity_curve = [capital]
    dates = [oot_prices.index[0]]
    trades = []

    for i in range(1, len(oot_prices)):
        today = oot_prices.index[i]
        full_idx = prices.index.get_loc(today)

        # Check if cash period ended
        if in_cash and today >= cash_until:
            # Re-enter SPY
            entry_price = oot_prices["SPY"].iloc[i] * (1 + slippage)
            spy_shares = capital / entry_price
            capital_after_buy = 0
            in_cash = False

        # Check for fear signal
        if not in_cash:
            sigs = signal_gold_fear(prices, full_idx)
            if sigs:
                # Sell SPY, go to cash
                exit_price = oot_prices["SPY"].iloc[i] * (1 - slippage)
                capital = spy_shares * exit_price
                entry_date = today
                exit_idx = min(i + 20, len(oot_prices) - 1)
                cash_until = oot_prices.index[exit_idx]
                trades.append({
                    "ticker": "SPY",
                    "entry_date": str(today.date()),
                    "exit_date": str(cash_until.date()),
                    "direction": 0,
                    "gross_ret": 0,
                    "net_ret": -2 * slippage,
                    "pnl": -capital * 2 * slippage,
                })
                spy_shares = 0
                in_cash = True

        # Mark-to-market
        if in_cash:
            mtm = capital
        else:
            mtm = spy_shares * oot_prices["SPY"].iloc[i]

        equity_curve.append(mtm)
        dates.append(today)

    eq = pd.Series(equity_curve, index=dates)
    return eq, trades


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    prices = download_data()

    strategies = {
        "A) Gold→Miners": signal_gold_miners,
        "B) Oil→Energy": signal_oil_energy,
        "C) Multi-commodity": signal_multi_commodity,
        "D) Commodity momentum": signal_commodity_momentum,
        "E) Contrarian oil": signal_contrarian_oil,
    }

    all_results = {}

    for name, sig_func in strategies.items():
        print(f"{'=' * 60}")
        print(f"Backtesting: {name}")
        print(f"{'=' * 60}")
        result = backtest_strategy(prices, sig_func)
        if result is None:
            print(f"  SKIPPED - insufficient data\n")
            continue
        eq, trades = result
        validation = validate_strategy(name, eq, trades, prices, sig_func)
        if validation is None:
            print(f"  SKIPPED - no valid metrics\n")
            continue
        all_results[name] = validation
        print_result(validation)

    # Variant F: special handling
    print(f"{'=' * 60}")
    print(f"Backtesting: F) Gold fear signal")
    print(f"{'=' * 60}")
    eq_f, trades_f = backtest_fear_signal(prices)
    validation_f = validate_strategy("F) Gold fear signal", eq_f, trades_f, prices, signal_gold_fear)
    if validation_f:
        all_results["F) Gold fear signal"] = validation_f
        print_result(validation_f)

    # SPY benchmark
    oot_spy = prices["SPY"][prices.index >= pd.Timestamp(OOT_START)]
    spy_ret = oot_spy.iloc[-1] / oot_spy.iloc[0] - 1
    spy_daily = oot_spy.pct_change().dropna()
    spy_sharpe = (spy_daily.mean() / spy_daily.std()) * np.sqrt(252) if spy_daily.std() > 0 else 0
    spy_dd = ((oot_spy - oot_spy.cummax()) / oot_spy.cummax()).min()

    all_results["_benchmark_SPY"] = {
        "total_return_pct": round(spy_ret * 100, 2),
        "sharpe": round(spy_sharpe, 3),
        "max_dd_pct": round(spy_dd * 100, 2),
    }

    # Summary
    print(f"\n{'=' * 70}")
    print(f"SUMMARY — Commodity-to-Equity Lead-Lag Backtest")
    print(f"OOT: {OOT_START} → {END_DATE} | Capital: ${INITIAL_CAPITAL} | Slippage: {SLIPPAGE_BPS*100:.2f}% each way")
    print(f"{'=' * 70}")
    print(f"{'Variant':<28} {'Sharpe':>7} {'Sortino':>8} {'Return%':>8} {'MaxDD%':>7} {'WR':>6} {'PF':>6} {'#Tr':>5} {'Gates':>6} {'Status':>6}")
    print(f"{'-' * 95}")

    for name, res in all_results.items():
        if name.startswith("_"):
            m = res
            print(f"{'SPY Buy&Hold':<28} {m['sharpe']:>7.3f} {'--':>8} {m['total_return_pct']:>7.1f}% {m['max_dd_pct']:>6.1f}% {'--':>6} {'--':>6} {'--':>5} {'--':>6} {'BM':>6}")
        else:
            m = res["metrics"]
            print(f"{name:<28} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} {m['total_return_pct']:>7.1f}% {m['max_dd_pct']:>6.1f}% {m['win_rate']:>5.1%} {m['profit_factor']:>6.2f} {m['n_trades']:>5} {res['gates_passed']:>6} {res['status']:>6}")

    print(f"\nSPY benchmark: {spy_ret*100:.1f}% total return, Sharpe {spy_sharpe:.3f}, MaxDD {spy_dd*100:.1f}%")

    # Save results
    # Convert for JSON serialization
    save_data = {
        "metadata": {
            "strategy": "Commodity-to-Equity Lead-Lag",
            "oot_period": f"{OOT_START} to {END_DATE}",
            "initial_capital": INITIAL_CAPITAL,
            "slippage_bps": SLIPPAGE_BPS * 10000,
            "n_permutations": N_PERMUTATIONS,
            "run_date": datetime.now().isoformat(),
        },
        "variants": all_results,
        "benchmark": all_results.get("_benchmark_SPY", {}),
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_PATH}")


def print_result(v):
    m = v["metrics"]
    print(f"\n  Metrics:")
    print(f"    Sharpe: {m['sharpe']:.3f} | Sortino: {m['sortino']:.3f} | Return: {m['total_return_pct']:.1f}%")
    print(f"    MaxDD: {m['max_dd_pct']:.1f}% | WR: {m['win_rate']:.1%} | PF: {m['profit_factor']:.2f} | Trades: {m['n_trades']}")
    print(f"    Final equity: ${m['final_equity']:.2f}")
    print(f"\n  Gates:")
    for gname, gval in v["gates"].items():
        status = "PASS" if gval["pass"] else "FAIL"
        extra = ""
        if "sharpe_bull" in gval:
            extra = f" (bull={gval['sharpe_bull']}, bear={gval['sharpe_bear']})"
        print(f"    [{status}] {gname}: {gval['value']}{extra}")
    print(f"  Overall: {v['gates_passed']} → {v['status']}\n")


if __name__ == "__main__":
    main()

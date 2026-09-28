#!/usr/bin/env python3
"""
ETF Pair Mean Reversion with Cointegration Backtest
====================================================
Academic basis: Gatev, Goetzmann & Rouwenhorst (2006)

Trades cointegrated ETF pairs using z-score of log spread ratio.
Market-neutral (long+short), so regime gap should be naturally low.

Variants:
  A: z=±2.0 entry, 0 exit, all 6 pairs
  B: z=±1.5 entry, 0 exit, all 6 pairs (more frequent)
  C: z=±2.0 entry, top 3 most cointegrated pairs only
  D: z=±2.0 entry, regime filter (SPY > 200-SMA = bull only)
  E: z=±2.0 entry, asymmetric — only long-spread trades
"""

import json
import sys
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from statsmodels.tsa.stattools import coint

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────────
PAIRS = [
    ("XLE", "XOP"),
    ("XLF", "KRE"),
    ("GLD", "GDX"),
    ("SPY", "IVV"),
    ("QQQ", "XLK"),
    ("TLT", "IEF"),
]

LOOKBACK = 60          # z-score lookback days
TIMEOUT = 20           # max holding period days
CAPITAL = 645.0        # per-pair capital
SLIPPAGE_PCT = 0.0002  # 0.02% per leg (0.04% round trip per leg)
COMMISSION = 0.0       # $0

DATA_START = "2021-06-01"   # need lookback before OOT
OOT_START = "2022-01-01"
OOT_END = "2026-07-28"

RESULTS_PATH = Path("/home/jupiter/Lvl3Quant/data/etf_pair_reversion_results.json")


# ── Data Download ───────────────────────────────────────────────────────────
def download_data():
    """Download all needed tickers."""
    all_tickers = set()
    for a, b in PAIRS:
        all_tickers.update([a, b])
    all_tickers.add("SPY")  # for regime filter

    tickers = sorted(all_tickers)
    print(f"Downloading {tickers} from {DATA_START} to {OOT_END}...")
    df = yf.download(tickers, start=DATA_START, end=OOT_END, auto_adjust=True, progress=False)

    # yfinance returns MultiIndex columns (Price, Ticker) in newer versions
    if isinstance(df.columns, pd.MultiIndex):
        close = df["Close"]
    else:
        close = df

    close = close.dropna()
    print(f"  Got {len(close)} trading days, {close.shape[1]} tickers")
    return close


# ── Cointegration Ranking ───────────────────────────────────────────────────
def rank_pairs_by_cointegration(close, end_date):
    """Rank pairs by cointegration p-value using data up to end_date."""
    data = close.loc[:end_date]
    results = []
    for etf1, etf2 in PAIRS:
        if etf1 not in data.columns or etf2 not in data.columns:
            continue
        s1 = data[etf1].dropna()
        s2 = data[etf2].dropna()
        common = s1.index.intersection(s2.index)
        if len(common) < 60:
            continue
        s1, s2 = s1.loc[common], s2.loc[common]
        try:
            _, pvalue, _ = coint(s1, s2)
        except Exception:
            pvalue = 1.0
        results.append(((etf1, etf2), pvalue))
    results.sort(key=lambda x: x[1])
    return results


# ── Backtest Engine ─────────────────────────────────────────────────────────
def compute_spread_zscore(close, etf1, etf2, lookback=LOOKBACK):
    """Compute log spread ratio and its z-score."""
    log_spread = np.log(close[etf1] / close[etf2])
    roll_mean = log_spread.rolling(lookback).mean()
    roll_std = log_spread.rolling(lookback).std()
    zscore = (log_spread - roll_mean) / roll_std
    return log_spread, zscore


def backtest_pair(close, etf1, etf2, z_entry=2.0, z_exit=0.0,
                  timeout=TIMEOUT, capital=CAPITAL, slippage=SLIPPAGE_PCT,
                  regime_filter=False, spy_sma=None,
                  long_spread_only=False):
    """
    Backtest a single pair.

    When z > +z_entry: spread is high → ETF1 outperforming → short ETF1, long ETF2
    When z < -z_entry: spread is low → ETF2 outperforming → long ETF1, short ETF2

    long_spread_only=True: only take z < -z_entry trades (long the underperformer ETF1)
    """
    _, zscore = compute_spread_zscore(close, etf1, etf2)

    oot_mask = zscore.index >= OOT_START
    dates = zscore.index[oot_mask]

    trades = []
    in_trade = False
    entry_date = None
    direction = None  # +1 = long spread (long ETF1, short ETF2), -1 = short spread
    entry_prices = None
    hold_days = 0

    for i, date in enumerate(dates):
        z = zscore.loc[date]
        if np.isnan(z):
            continue

        # Regime filter: skip if SPY < 200-SMA
        if regime_filter and spy_sma is not None:
            if date in spy_sma.index and spy_sma.loc[date] == False:
                if not in_trade:
                    continue  # don't enter in bear regime

        if not in_trade:
            # Entry signals
            if z > z_entry:
                # Spread too high → short spread (short ETF1, long ETF2)
                if long_spread_only:
                    continue
                direction = -1
                in_trade = True
                entry_date = date
                entry_prices = (close[etf1].loc[date], close[etf2].loc[date])
                hold_days = 0
            elif z < -z_entry:
                # Spread too low → long spread (long ETF1, short ETF2)
                direction = +1
                in_trade = True
                entry_date = date
                entry_prices = (close[etf1].loc[date], close[etf2].loc[date])
                hold_days = 0
        else:
            hold_days += 1
            # Exit conditions
            exit_signal = False
            if direction == -1 and z <= z_exit:
                exit_signal = True
            elif direction == +1 and z >= -z_exit:
                exit_signal = True
            elif hold_days >= timeout:
                exit_signal = True

            if exit_signal:
                exit_prices = (close[etf1].loc[date], close[etf2].loc[date])

                # Compute P&L
                # Half capital on each leg
                half_cap = capital / 2.0

                if direction == -1:
                    # Short ETF1, Long ETF2
                    shares1 = half_cap / entry_prices[0]
                    shares2 = half_cap / entry_prices[1]
                    pnl_etf1 = shares1 * (entry_prices[0] - exit_prices[0])  # short
                    pnl_etf2 = shares2 * (exit_prices[1] - entry_prices[1])  # long
                else:
                    # Long ETF1, Short ETF2
                    shares1 = half_cap / entry_prices[0]
                    shares2 = half_cap / entry_prices[1]
                    pnl_etf1 = shares1 * (exit_prices[0] - entry_prices[0])  # long
                    pnl_etf2 = shares2 * (entry_prices[1] - exit_prices[1])  # short

                gross_pnl = pnl_etf1 + pnl_etf2

                # Slippage: 0.02% per leg, 2 legs, entry + exit = 4 slippage events
                total_notional = half_cap * 2  # entry notional
                slippage_cost = total_notional * slippage * 4  # 4 leg-events

                net_pnl = gross_pnl - slippage_cost

                trades.append({
                    "pair": f"{etf1}/{etf2}",
                    "entry_date": str(entry_date.date()),
                    "exit_date": str(date.date()),
                    "direction": "long_spread" if direction == 1 else "short_spread",
                    "hold_days": hold_days,
                    "entry_z": float(zscore.loc[entry_date]),
                    "exit_z": float(z),
                    "gross_pnl": round(gross_pnl, 2),
                    "net_pnl": round(net_pnl, 2),
                    "return_pct": round(net_pnl / capital * 100, 4),
                })

                in_trade = False
                direction = None
                entry_prices = None

    return trades


def run_variant(close, variant_name, z_entry=2.0, z_exit=0.0,
                pairs=None, regime_filter=False, long_spread_only=False):
    """Run a full variant across multiple pairs."""
    if pairs is None:
        pairs = PAIRS

    # Regime filter setup
    spy_sma = None
    if regime_filter:
        spy_sma = close["SPY"].rolling(200).mean()
        spy_sma = close["SPY"] > spy_sma  # True = bull

    all_trades = []
    for etf1, etf2 in pairs:
        trades = backtest_pair(
            close, etf1, etf2,
            z_entry=z_entry, z_exit=z_exit,
            regime_filter=regime_filter, spy_sma=spy_sma,
            long_spread_only=long_spread_only,
        )
        all_trades.extend(trades)

    return all_trades


# ── Metrics ─────────────────────────────────────────────────────────────────
def compute_metrics(trades, close, capital=CAPITAL):
    """Compute strategy metrics from trade list."""
    if not trades:
        return {"n_trades": 0, "sharpe": 0, "sortino": 0, "pf": 0,
                "wr": 0, "max_dd_pct": 0, "total_return_pct": 0,
                "avg_hold_days": 0, "avg_return_pct": 0}

    returns = [t["return_pct"] / 100.0 for t in trades]
    pnls = [t["net_pnl"] for t in trades]

    n = len(trades)
    winners = [r for r in returns if r > 0]
    losers = [r for r in returns if r < 0]

    wr = len(winners) / n if n > 0 else 0

    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Build equity curve from trade sequence (sorted by exit date)
    sorted_trades = sorted(trades, key=lambda t: t["exit_date"])
    equity = [capital]
    for t in sorted_trades:
        equity.append(equity[-1] + t["net_pnl"])
    equity = np.array(equity)

    # Max drawdown
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = dd.min() * 100

    total_return = (equity[-1] - capital) / capital * 100

    # Annualize: approximate trades per year
    if len(sorted_trades) >= 2:
        first_exit = pd.Timestamp(sorted_trades[0]["exit_date"])
        last_exit = pd.Timestamp(sorted_trades[-1]["exit_date"])
        span_years = max((last_exit - first_exit).days / 365.25, 0.5)
    else:
        span_years = 1.0

    trades_per_year = n / span_years

    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1) if n > 1 else 1e-9

    # Annualized Sharpe (using trade-frequency scaling)
    sharpe = (mean_ret / std_ret) * np.sqrt(trades_per_year) if std_ret > 0 else 0

    # Sortino
    downside = [r for r in returns if r < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-9
    sortino = (mean_ret / downside_std) * np.sqrt(trades_per_year) if downside_std > 0 else 0

    avg_hold = np.mean([t["hold_days"] for t in trades])
    avg_ret = np.mean(returns) * 100

    return {
        "n_trades": n,
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "pf": round(pf, 3),
        "wr": round(wr * 100, 1),
        "max_dd_pct": round(max_dd, 2),
        "total_return_pct": round(total_return, 2),
        "avg_hold_days": round(avg_hold, 1),
        "avg_return_pct": round(avg_ret, 4),
        "trades_per_year": round(trades_per_year, 1),
        "span_years": round(span_years, 2),
    }


def regime_gap(trades, close):
    """
    Compute regime gap: |Sharpe_bull - Sharpe_bear| / max(|Sharpe_bull|, |Sharpe_bear|).
    Bull = SPY > 200-SMA on entry date, Bear = SPY < 200-SMA.
    """
    if not trades:
        return 1.0

    spy_sma200 = close["SPY"].rolling(200).mean()

    bull_trades = []
    bear_trades = []
    for t in trades:
        entry = pd.Timestamp(t["entry_date"])
        if entry in spy_sma200.index and entry in close["SPY"].index:
            if close["SPY"].loc[entry] > spy_sma200.loc[entry]:
                bull_trades.append(t)
            else:
                bear_trades.append(t)
        else:
            bull_trades.append(t)  # default to bull if missing

    def _sharpe(trade_list):
        if len(trade_list) < 2:
            return 0.0
        rets = [t["return_pct"] / 100.0 for t in trade_list]
        m = np.mean(rets)
        s = np.std(rets, ddof=1)
        return m / s if s > 0 else 0.0

    s_bull = _sharpe(bull_trades)
    s_bear = _sharpe(bear_trades)

    denom = max(abs(s_bull), abs(s_bear))
    if denom == 0:
        return 0.0

    gap = abs(s_bull - s_bear) / denom
    return round(gap, 3)


def permutation_test(trades, n_perms=1000):
    """Permutation test: shuffle trade returns, compare mean to actual."""
    if len(trades) < 5:
        return 1.0

    returns = np.array([t["return_pct"] for t in trades])
    actual_mean = np.mean(returns)

    rng = np.random.default_rng(42)
    count_better = 0
    for _ in range(n_perms):
        shuffled = rng.choice(returns, size=len(returns), replace=True)
        # Randomly flip signs to simulate no-edge null
        signs = rng.choice([-1, 1], size=len(returns))
        null_mean = np.mean(shuffled * signs)
        if null_mean >= actual_mean:
            count_better += 1

    return round(count_better / n_perms, 4)


# ── 5-Gate Validation ───────────────────────────────────────────────────────
def validate_5gate(metrics, perm_p, r_gap):
    """Apply 5-gate validation."""
    gates = {
        "sharpe_gt_0.5": metrics["sharpe"] > 0.5,
        "perm_p_lt_0.05": perm_p < 0.05,
        "regime_gap_lt_0.5": r_gap < 0.5,
        "max_dd_gt_neg50": metrics["max_dd_pct"] > -50,
        "trades_gte_20": metrics["n_trades"] >= 20,
    }
    gates["pass_all"] = all(gates.values())
    return gates


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("ETF Pair Mean Reversion Backtest")
    print(f"OOT Period: {OOT_START} to {OOT_END}")
    print(f"Capital: ${CAPITAL}, Slippage: {SLIPPAGE_PCT*100:.2f}% per leg")
    print("=" * 70)

    close = download_data()

    # Rank pairs by cointegration
    coint_ranking = rank_pairs_by_cointegration(close, OOT_START)
    print("\nCointegration ranking (lower p-value = more cointegrated):")
    for (e1, e2), pv in coint_ranking:
        print(f"  {e1}/{e2}: p={pv:.4f} {'***' if pv < 0.05 else ''}")

    top3_pairs = [pair for pair, _ in coint_ranking[:3]]

    # Define variants
    variants = {
        "A": {"z_entry": 2.0, "z_exit": 0.0, "pairs": PAIRS,
               "desc": "z=±2.0, all 6 pairs"},
        "B": {"z_entry": 1.5, "z_exit": 0.0, "pairs": PAIRS,
               "desc": "z=±1.5, all 6 pairs (more frequent)"},
        "C": {"z_entry": 2.0, "z_exit": 0.0, "pairs": top3_pairs,
               "desc": f"z=±2.0, top 3 cointegrated: {[f'{a}/{b}' for a,b in top3_pairs]}"},
        "D": {"z_entry": 2.0, "z_exit": 0.0, "pairs": PAIRS,
               "regime_filter": True,
               "desc": "z=±2.0, all pairs, bull regime only (SPY>200-SMA)"},
        "E": {"z_entry": 2.0, "z_exit": 0.0, "pairs": PAIRS,
               "long_spread_only": True,
               "desc": "z=±2.0, all pairs, long-spread only (asymmetric)"},
    }

    results = {}

    for name, cfg in variants.items():
        print(f"\n{'─'*70}")
        print(f"Variant {name}: {cfg['desc']}")
        print(f"{'─'*70}")

        trades = run_variant(
            close, name,
            z_entry=cfg["z_entry"],
            z_exit=cfg["z_exit"],
            pairs=cfg["pairs"],
            regime_filter=cfg.get("regime_filter", False),
            long_spread_only=cfg.get("long_spread_only", False),
        )

        metrics = compute_metrics(trades, close)
        perm_p = permutation_test(trades)
        r_gap = regime_gap(trades, close)
        gates = validate_5gate(metrics, perm_p, r_gap)

        print(f"  Trades: {metrics['n_trades']}")
        print(f"  Sharpe: {metrics['sharpe']}, Sortino: {metrics['sortino']}")
        print(f"  PF: {metrics['pf']}, WR: {metrics['wr']}%")
        print(f"  Total Return: {metrics['total_return_pct']}%")
        print(f"  Max DD: {metrics['max_dd_pct']}%")
        print(f"  Avg Hold: {metrics['avg_hold_days']} days")
        print(f"  Permutation p: {perm_p}")
        print(f"  Regime Gap: {r_gap}")
        print(f"  5-Gate: {'PASS' if gates['pass_all'] else 'FAIL'}")
        for g, v in gates.items():
            if g != "pass_all":
                print(f"    {g}: {'✓' if v else '✗'}")

        # Per-pair breakdown
        pair_breakdown = {}
        for etf1, etf2 in cfg["pairs"]:
            pair_name = f"{etf1}/{etf2}"
            pair_trades = [t for t in trades if t["pair"] == pair_name]
            if pair_trades:
                pair_metrics = compute_metrics(pair_trades, close)
                pair_breakdown[pair_name] = pair_metrics
                print(f"  {pair_name}: {len(pair_trades)} trades, "
                      f"Sharpe={pair_metrics['sharpe']}, "
                      f"WR={pair_metrics['wr']}%, "
                      f"Return={pair_metrics['total_return_pct']}%")

        results[f"variant_{name}"] = {
            "description": cfg["desc"],
            "parameters": {
                "z_entry": cfg["z_entry"],
                "z_exit": cfg["z_exit"],
                "lookback": LOOKBACK,
                "timeout": TIMEOUT,
                "capital": CAPITAL,
                "slippage_pct": SLIPPAGE_PCT,
                "pairs": [f"{a}/{b}" for a, b in cfg["pairs"]],
                "regime_filter": cfg.get("regime_filter", False),
                "long_spread_only": cfg.get("long_spread_only", False),
            },
            "metrics": metrics,
            "permutation_p": perm_p,
            "regime_gap": r_gap,
            "gates": gates,
            "pair_breakdown": pair_breakdown,
            "sample_trades": trades[:10] if trades else [],
            "n_trades_by_pair": {
                f"{a}/{b}": len([t for t in trades if t["pair"] == f"{a}/{b}"])
                for a, b in cfg["pairs"]
            },
        }

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Variant':<10} {'Trades':>7} {'Sharpe':>8} {'Sortino':>8} "
          f"{'PF':>6} {'WR%':>6} {'MaxDD%':>8} {'Return%':>9} {'5-Gate':>7}")
    print("-" * 75)
    for name in sorted(results.keys()):
        r = results[name]
        m = r["metrics"]
        g = "PASS" if r["gates"]["pass_all"] else "FAIL"
        print(f"  {name:<8} {m['n_trades']:>7} {m['sharpe']:>8.3f} {m['sortino']:>8.3f} "
              f"{m['pf']:>6.2f} {m['wr']:>5.1f}% {m['max_dd_pct']:>7.2f}% "
              f"{m['total_return_pct']:>8.2f}% {g:>7}")

    # Cointegration info
    results["cointegration_ranking"] = [
        {"pair": f"{e1}/{e2}", "p_value": round(pv, 6)}
        for (e1, e2), pv in coint_ranking
    ]
    results["metadata"] = {
        "strategy": "ETF Pair Mean Reversion with Cointegration",
        "academic_basis": "Gatev, Goetzmann & Rouwenhorst (2006)",
        "oot_period": f"{OOT_START} to {OOT_END}",
        "lookback": LOOKBACK,
        "timeout": TIMEOUT,
        "capital": CAPITAL,
        "slippage_pct_per_leg": SLIPPAGE_PCT,
        "commission": COMMISSION,
        "run_timestamp": datetime.now().isoformat(),
    }

    # Save
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()

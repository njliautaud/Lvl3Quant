#!/usr/bin/env python3
"""
Sector ETF PEAD Backtest
========================
Tests post-earnings announcement drift (PEAD) applied to sector ETFs
rather than individual stocks, to reduce concentration risk.

6 variants tested with 5-gate validation framework.
Walk-forward OOT: Jan 2022 – Jul 2026. Starting capital: $645.
"""

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────

START_DATE = "2021-06-01"
OOT_START = "2022-01-01"
OOT_END = "2026-07-30"
INITIAL_CAPITAL = 645.0
SLIPPAGE_PCT = 0.0002

SECTOR_MAP = {
    "XLK": ["AAPL", "MSFT", "NVDA", "AMD"],
    "XLC": ["GOOGL", "META", "NFLX", "DIS", "CMCSA"],
    "XLY": ["AMZN", "TSLA", "HD", "NKE"],
    "XLF": ["JPM", "BAC", "GS"],
    "XLE": ["XOM", "CVX"],
    "XLV": ["JNJ", "UNH", "PFE"],
    "XLI": ["BA", "CAT", "HON"],
    "XLP": ["PG", "KO", "WMT"],
    "XLU": ["NEE", "DUK"],
    "XLB": ["FCX", "NEM"],
    "XLRE": ["PLD", "SPG"],
}

SECTOR_ETFS = list(SECTOR_MAP.keys())
ALL_STOCKS = sorted(set(s for stocks in SECTOR_MAP.values() for s in stocks))
HOLD_DAYS = 40
TRAILING_STOP_PCT = 0.05
MAX_HOLD_TRAILING = 60
PERM_ITERATIONS = 500


# ── Data ───────────────────────────────────────────────────────────────────────

def download_data():
    all_tickers = sorted(set(SECTOR_ETFS + ALL_STOCKS + ["SPY", "GLD"]))
    print(f"Downloading {len(all_tickers)} tickers...", flush=True)
    data = yf.download(all_tickers, start=START_DATE, end=OOT_END,
                       auto_adjust=True, progress=False, threads=True)
    return data["Close"], data["Open"], data["Volume"], data["High"], data["Low"]


# ── Pre-compute all signals for every date ─────────────────────────────────────

def precompute_signals(close, opn, volume):
    """
    Pre-compute sector PEAD signals for every OOT date.
    Returns a dict of precomputed arrays for fast backtest.
    """
    gaps = (opn - close.shift(1)) / close.shift(1)
    etf_vol_avg = volume[SECTOR_ETFS].rolling(20).mean()
    etf_sma20 = close[SECTOR_ETFS].rolling(20).mean()

    oot_dates = close.index[(close.index >= OOT_START) & (close.index <= OOT_END)]

    # For each date, store signals per sector
    # signals_by_date[date_idx] = list of (etf, direction, n_stocks_gapping, max_gap, avg_gap,
    #                                       vol_confirmed, momentum_ok, multi_stock_ok)
    signals_by_date = {}

    for di, date in enumerate(oot_dates):
        sigs = []
        for etf, stocks in SECTOR_MAP.items():
            avail = [s for s in stocks if s in gaps.columns]
            if not avail or date not in gaps.index:
                continue

            day_gaps = gaps.loc[date, avail].dropna()
            if len(day_gaps) == 0:
                continue

            abs_gaps = day_gaps.abs()
            big_gaps = abs_gaps[abs_gaps > 0.03]
            medium_gaps = abs_gaps[abs_gaps > 0.02]

            if len(big_gaps) >= 1 or len(medium_gaps) >= 2:
                relevant = day_gaps[abs_gaps > 0.02] if len(medium_gaps) >= 2 else day_gaps[abs_gaps > 0.03]
                avg_gap = float(relevant.mean())
                direction = 1 if avg_gap > 0 else -1
                n_stocks = int(max(len(big_gaps), len(medium_gaps)))

                # Volume check
                vol_ok = False
                if etf in etf_vol_avg.columns:
                    avg_v = etf_vol_avg.loc[date, etf]
                    cur_v = volume.loc[date, etf]
                    if not pd.isna(avg_v) and not pd.isna(cur_v) and cur_v >= 1.5 * avg_v:
                        vol_ok = True

                # Momentum check
                mom_ok = False
                if etf in etf_sma20.columns:
                    sma = etf_sma20.loc[date, etf]
                    price = close.loc[date, etf]
                    if not pd.isna(sma) and price >= sma:
                        mom_ok = True

                # Multi-stock check (2+ stocks >3%)
                multi_ok = int((abs_gaps > 0.03).sum()) >= 2

                sigs.append({
                    "etf": etf,
                    "direction": direction,
                    "n_stocks": n_stocks,
                    "max_gap": float(abs_gaps.max()),
                    "avg_gap": avg_gap,
                    "vol_ok": vol_ok,
                    "mom_ok": mom_ok,
                    "multi_ok": multi_ok,
                })

        signals_by_date[di] = sigs

    return oot_dates, signals_by_date


# ── Fast Backtest Engine ───────────────────────────────────────────────────────

def run_backtest_fast(variant, oot_dates, signals_by_date, close, high, low,
                      shuffle_seed=None):
    """
    Optimized backtest using pre-computed signals.
    """
    rng = np.random.RandomState(shuffle_seed) if shuffle_seed is not None else None

    capital = INITIAL_CAPITAL
    # positions: list of (etf, entry_idx, entry_price, weight, direction, hwm, days_held)
    positions = []
    trades = []
    equity = np.zeros(len(oot_dates))

    # Pre-fetch close prices as numpy for speed
    etf_close = {}
    etf_high = {}
    etf_low = {}
    for etf in SECTOR_ETFS:
        if etf in close.columns:
            etf_close[etf] = close[etf].reindex(oot_dates).values
            etf_high[etf] = high[etf].reindex(oot_dates).values
            etf_low[etf] = low[etf].reindex(oot_dates).values

    for di in range(len(oot_dates)):
        # ── Check exits ──
        new_positions = []
        for etf, entry_idx, entry_price, weight, direction, hwm, days_held in positions:
            days_held += 1
            cp = etf_close[etf][di]
            if np.isnan(cp):
                new_positions.append((etf, entry_idx, entry_price, weight, direction, hwm, days_held))
                continue

            # Update HWM
            dh = etf_high[etf][di]
            dl = etf_low[etf][di]
            if direction == 1 and not np.isnan(dh):
                hwm = max(hwm, dh)
            elif direction == -1 and not np.isnan(dl):
                hwm = min(hwm, dl)

            exit_now = False
            if variant == "E":
                if direction == 1:
                    if hwm > 0 and (hwm - cp) / hwm >= TRAILING_STOP_PCT:
                        exit_now = True
                else:
                    if hwm > 0 and (cp - hwm) / hwm >= TRAILING_STOP_PCT:
                        exit_now = True
                if days_held >= MAX_HOLD_TRAILING:
                    exit_now = True
            else:
                if days_held >= HOLD_DAYS:
                    exit_now = True

            if exit_now:
                exit_price = cp * (1 - SLIPPAGE_PCT * direction)
                pnl_pct = direction * (exit_price - entry_price) / entry_price
                trade_pnl = capital * weight * pnl_pct
                capital += trade_pnl
                trades.append(pnl_pct)
            else:
                new_positions.append((etf, entry_idx, entry_price, weight, direction, hwm, days_held))

        positions = new_positions

        # ── Check entries ──
        total_weight = sum(p[3] for p in positions)
        avail_weight = 1.0 - total_weight

        if avail_weight > 0.05:
            raw_sigs = signals_by_date.get(di, [])

            if rng is not None and raw_sigs:
                # Permutation: shuffle which sectors get triggered
                # Keep same number of signals but randomize which ETFs
                n_sigs = len(raw_sigs)
                # Randomly sample sectors and assign random directions
                perm_etfs = rng.choice(SECTOR_ETFS, size=n_sigs, replace=False) if n_sigs <= len(SECTOR_ETFS) else rng.choice(SECTOR_ETFS, size=n_sigs, replace=True)
                shuffled = []
                for j, sig in enumerate(raw_sigs):
                    new_sig = dict(sig)
                    new_sig["etf"] = perm_etfs[j]
                    new_sig["direction"] = rng.choice([-1, 1])
                    shuffled.append(new_sig)
                raw_sigs = shuffled

            # Filter by variant
            filtered = []
            for sig in raw_sigs:
                etf = sig["etf"]
                if etf not in etf_close:
                    continue
                if variant == "B" and not sig["vol_ok"]:
                    continue
                if variant == "C" and not sig["mom_ok"]:
                    continue
                if variant == "D" and not sig["multi_ok"]:
                    continue
                filtered.append(sig)

            if filtered:
                if variant == "F":
                    # Pick strongest breadth
                    best = max(filtered, key=lambda s: (s["n_stocks"], s["max_gap"]))
                    filtered = [best]

                held_etfs = set(p[0] for p in positions)
                n_new = len([s for s in filtered if s["etf"] not in held_etfs])
                if n_new > 0:
                    w_per = avail_weight / n_new

                    for sig in filtered:
                        etf = sig["etf"]
                        if etf in held_etfs:
                            continue
                        cp = etf_close[etf][di]
                        if np.isnan(cp):
                            continue
                        entry_price = cp * (1 + SLIPPAGE_PCT * sig["direction"])
                        positions.append((etf, di, entry_price, w_per, sig["direction"], entry_price, 0))

        # MTM
        mtm = capital
        for etf, entry_idx, entry_price, weight, direction, hwm, days_held in positions:
            cp = etf_close[etf][di]
            if not np.isnan(cp):
                pnl_pct = direction * (cp - entry_price) / entry_price
                mtm += capital * weight * pnl_pct
        equity[di] = mtm

    # Close remaining
    for etf, entry_idx, entry_price, weight, direction, hwm, days_held in positions:
        cp = etf_close[etf][-1]
        if not np.isnan(cp):
            pnl_pct = direction * (cp - entry_price) / entry_price
            trades.append(pnl_pct)

    return trades, equity


# ── Metrics ────────────────────────────────────────────────────────────────────

def calc_metrics(trade_pnls, equity, spy_vals):
    """Calculate all metrics from trade pnls and equity curve."""
    n_trades = len(trade_pnls)
    if n_trades == 0 or len(equity) < 10:
        return {"sharpe": 0, "sortino": 0, "pf": 0, "wr": 0, "max_dd": -1.0,
                "total_return": 0, "n_trades": 0, "bull_sharpe": 0, "bear_sharpe": 0,
                "regime_gap": 1.0, "qqq_corr": 0}

    # Daily returns from equity curve
    rets = np.diff(equity) / equity[:-1]
    rets = np.nan_to_num(rets, 0)

    daily_mean = np.mean(rets)
    daily_std = np.std(rets, ddof=1) if len(rets) > 1 else 1e-9
    sharpe = (daily_mean / daily_std * np.sqrt(252)) if daily_std > 1e-12 else 0

    down = rets[rets < 0]
    down_std = np.std(down, ddof=1) if len(down) > 1 else 1e-9
    sortino = (daily_mean / down_std * np.sqrt(252)) if down_std > 1e-12 else 0

    # MaxDD
    cum_max = np.maximum.accumulate(equity)
    dd = (equity - cum_max) / cum_max
    max_dd = float(np.min(dd))

    # Trade metrics
    winners = [p for p in trade_pnls if p > 0]
    losers = [p for p in trade_pnls if p <= 0]
    wr = len(winners) / n_trades
    gp = sum(winners) if winners else 0
    gl = abs(sum(losers)) if losers else 1e-9
    pf = gp / gl if gl > 1e-12 else 999.0

    total_return = (equity[-1] / INITIAL_CAPITAL - 1)

    # Regime
    spy_rets = np.diff(spy_vals) / spy_vals[:-1]
    spy_rets = np.nan_to_num(spy_rets, 0)

    # 50-day SMA on spy_vals
    spy_sma50 = pd.Series(spy_vals).rolling(50).mean().values
    bull = spy_vals[1:] > spy_sma50[1:]  # aligned with rets
    bear = ~bull

    # Trim to min length
    n = min(len(rets), len(bull))
    rets_a = rets[:n]
    bull_a = bull[:n]
    bear_a = bear[:n]

    bull_rets = rets_a[bull_a]
    bear_rets = rets_a[bear_a]

    bull_sharpe = 0
    if len(bull_rets) > 10:
        bs = np.std(bull_rets, ddof=1)
        if bs > 1e-12:
            bull_sharpe = np.mean(bull_rets) / bs * np.sqrt(252)

    bear_sharpe = 0
    if len(bear_rets) > 10:
        bs = np.std(bear_rets, ddof=1)
        if bs > 1e-12:
            bear_sharpe = np.mean(bear_rets) / bs * np.sqrt(252)

    max_abs = max(abs(bull_sharpe), abs(bear_sharpe), 1e-9)
    regime_gap = abs(bull_sharpe - bear_sharpe) / max_abs

    # SPY correlation
    spy_r = spy_rets[:n]
    if len(rets_a) > 10 and np.std(rets_a) > 1e-12 and np.std(spy_r) > 1e-12:
        qqq_corr = float(np.corrcoef(rets_a, spy_r)[0, 1])
    else:
        qqq_corr = 0

    return {
        "sharpe": round(float(sharpe), 3),
        "sortino": round(float(sortino), 3),
        "pf": round(float(min(pf, 999)), 3),
        "wr": round(float(wr), 3),
        "max_dd": round(float(max_dd), 3),
        "total_return": round(float(total_return), 3),
        "n_trades": n_trades,
        "bull_sharpe": round(float(bull_sharpe), 3),
        "bear_sharpe": round(float(bear_sharpe), 3),
        "regime_gap": round(float(regime_gap), 3),
        "qqq_corr": round(float(qqq_corr) if not np.isnan(qqq_corr) else 0, 3),
    }


# ── Permutation Test ──────────────────────────────────────────────────────────

def permutation_test(variant, oot_dates, signals_by_date, close, high, low, spy_vals,
                     real_sharpe, n_iter=PERM_ITERATIONS):
    perm_sharpes = np.zeros(n_iter)
    for i in range(n_iter):
        if (i + 1) % 100 == 0:
            print(f"  Permutation {i+1}/{n_iter}...", flush=True)
        trades, eq = run_backtest_fast(variant, oot_dates, signals_by_date, close, high, low,
                                       shuffle_seed=i + 42)
        if trades and len(eq) > 0:
            m = calc_metrics(trades, eq, spy_vals)
            perm_sharpes[i] = m["sharpe"]

    p_value = (np.sum(perm_sharpes >= real_sharpe) + 1) / (n_iter + 1)
    return float(p_value)


# ── Gate Validation ───────────────────────────────────────────────────────────

def validate_gates(metrics, p_value):
    gates = {
        "sharpe_gt_0.5": metrics["sharpe"] > 0.5,
        "perm_p_lt_0.05": p_value < 0.05,
        "regime_gap_lt_0.5": metrics["regime_gap"] < 0.5,
        "max_dd_gt_neg50": metrics["max_dd"] > -0.50,
        "n_trades_gte_20": metrics["n_trades"] >= 20,
    }
    gates["all_pass"] = all(v for k, v in gates.items())
    return gates


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70, flush=True)
    print("SECTOR ETF PEAD BACKTEST", flush=True)
    print(f"Walk-Forward OOT: {OOT_START} to {OOT_END}", flush=True)
    print(f"Starting Capital: ${INITIAL_CAPITAL}", flush=True)
    print("=" * 70, flush=True)

    close, opn, volume, high, low = download_data()
    print(f"Data: {close.shape[0]} days, {close.shape[1]} tickers", flush=True)
    print(f"Range: {close.index[0].date()} to {close.index[-1].date()}", flush=True)

    # Pre-compute all signals
    print("Pre-computing sector signals...", flush=True)
    oot_dates, signals_by_date = precompute_signals(close, opn, volume)
    n_signal_days = sum(1 for sigs in signals_by_date.values() if sigs)
    total_sigs = sum(len(sigs) for sigs in signals_by_date.values())
    print(f"Signal days: {n_signal_days}/{len(oot_dates)}, total sector signals: {total_sigs}", flush=True)

    # SPY for regime analysis
    spy_vals = close["SPY"].reindex(oot_dates).ffill().values

    variants = ["A", "B", "C", "D", "E", "F"]
    variant_names = {
        "A": "Basic Sector PEAD",
        "B": "Volume-Confirmed",
        "C": "Momentum-Filtered",
        "D": "Multi-Stock Threshold",
        "E": "Trailing Exit (5%)",
        "F": "Rotation Priority",
    }

    all_results = {}

    for v in variants:
        print(f"\n{'─' * 60}", flush=True)
        print(f"Variant {v}: {variant_names[v]}...", flush=True)

        trades, equity = run_backtest_fast(v, oot_dates, signals_by_date, close, high, low)
        metrics = calc_metrics(trades, equity, spy_vals)

        print(f"  Trades: {metrics['n_trades']}, Sharpe: {metrics['sharpe']}, "
              f"MaxDD: {metrics['max_dd']:.1%}, Return: {metrics['total_return']:.1%}", flush=True)

        if metrics["n_trades"] >= 5:
            print(f"  Running {PERM_ITERATIONS} permutations...", flush=True)
            p_value = permutation_test(v, oot_dates, signals_by_date, close, high, low,
                                        spy_vals, metrics["sharpe"])
            print(f"  p-value: {p_value:.4f}", flush=True)
        else:
            p_value = 1.0
            print("  Too few trades for permutation test", flush=True)

        gates = validate_gates(metrics, p_value)

        # Count long/short
        trades2, _ = run_backtest_fast(v, oot_dates, signals_by_date, close, high, low)

        all_results[v] = {
            "name": variant_names[v],
            "metrics": metrics,
            "p_value": round(p_value, 4),
            "gates": gates,
        }

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 105, flush=True)
    print("SECTOR ETF PEAD — RESULTS SUMMARY", flush=True)
    print("=" * 105, flush=True)

    print(f"{'Var':>3} {'Name':<25} {'Sharpe':>7} {'Sortino':>8} {'PF':>6} {'WR':>5} "
          f"{'MaxDD':>7} {'Return':>8} {'#Tr':>4} {'p-val':>6} {'RGap':>5} {'Pass':>5}", flush=True)
    print("-" * 105, flush=True)

    for v in variants:
        r = all_results[v]
        m = r["metrics"]
        g = r["gates"]
        ps = "YES" if g["all_pass"] else "NO"
        print(f"  {v} {r['name']:<25} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} {m['pf']:>6.2f} "
              f"{m['wr']:>5.1%} {m['max_dd']:>7.1%} {m['total_return']:>8.1%} {m['n_trades']:>4} "
              f"{r['p_value']:>6.4f} {m['regime_gap']:>5.2f} {ps:>5}", flush=True)

    print("-" * 105, flush=True)

    # Gate details
    print("\n5-GATE VALIDATION:", flush=True)
    print(f"{'Var':>3} {'Sharpe>0.5':>11} {'p<0.05':>8} {'RGap<0.5':>10} {'DD>-50%':>9} {'>=20 tr':>8} {'PASS':>6}", flush=True)
    print("-" * 60, flush=True)
    for v in variants:
        g = all_results[v]["gates"]
        def yn(b): return "OK" if b else "FAIL"
        print(f"  {v} {yn(g['sharpe_gt_0.5']):>11} {yn(g['perm_p_lt_0.05']):>8} "
              f"{yn(g['regime_gap_lt_0.5']):>10} {yn(g['max_dd_gt_neg50']):>9} "
              f"{yn(g['n_trades_gte_20']):>8} {yn(g['all_pass']):>6}", flush=True)

    passing = {v: r for v, r in all_results.items() if r["gates"]["all_pass"]}
    if passing:
        best = max(passing, key=lambda v: passing[v]["metrics"]["sharpe"])
        print(f"\nBEST PASSING VARIANT: {best} — {variant_names[best]} "
              f"(Sharpe {all_results[best]['metrics']['sharpe']:.3f})", flush=True)
    else:
        best = max(all_results, key=lambda v: all_results[v]["metrics"]["sharpe"])
        print(f"\nNO VARIANT PASSES ALL 5 GATES.", flush=True)
        print(f"Best Sharpe: {best} — {variant_names[best]} "
              f"(Sharpe {all_results[best]['metrics']['sharpe']:.3f})", flush=True)

    print(f"\nBenchmark: Vol-Adj Rotation (GLD/TLT/UUP) Sharpe = 2.02", flush=True)

    # Save
    def clean(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, list): return [clean(i) for i in obj]
        if isinstance(obj, pd.Timestamp): return obj.isoformat()
        if isinstance(obj, np.bool_): return bool(obj)
        return obj

    output = {
        "backtest": "Sector ETF PEAD",
        "oot_period": f"{OOT_START} to {OOT_END}",
        "initial_capital": INITIAL_CAPITAL,
        "run_date": datetime.now().isoformat(),
        "hypothesis": "Sector ETFs drift after constituent earnings gaps, reducing concentration risk",
        "variants": all_results,
        "any_pass": bool(passing),
    }

    out_path = Path("/home/jupiter/Lvl3Quant/data/sector_etf_pead_results.json")
    with open(out_path, "w") as f:
        json.dump(clean(output), f, indent=2, default=str)

    print(f"\nResults saved to {out_path}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()

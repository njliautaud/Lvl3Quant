#!/usr/bin/env python3
"""
Volatility Surface Signals for Quality Stock Mean Reversion
============================================================
Six variants using VIX term structure, realized vol, and vol-of-vol
to time mean-reversion entries on quality large-cap stocks.

Author: Claude Opus 4.6
Date: 2026-07-31
"""

import json
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────
UNIVERSE = [
    "AAPL", "MSFT", "AVGO", "JPM", "JNJ", "PG", "KO", "PEP", "HD", "COST",
    "UNH", "LLY", "V", "MA", "ABBV", "MRK", "WMT", "AMZN", "GOOGL", "META",
]
CAPITAL = 669.0
MAX_PER_TRADE = 200.0
MAX_CONCURRENT = 3
SLIPPAGE_BPS = 2  # 2 basis points
HOLD_DAYS = 10
START = "2021-06-01"  # extra runway for indicators
TRADE_START = "2022-01-01"
END = "2026-07-31"

RESULTS_PATH = Path("/home/jupiter/Lvl3Quant/data/vol_surface_signal_results.json")


# ── Data Download ──────────────────────────────────────────────────────
def download_data():
    """Download stock + VIX data via yfinance."""
    tickers = UNIVERSE + ["^VIX", "^VIX3M", "SPY"]
    print(f"Downloading {len(tickers)} tickers...")
    raw = yf.download(tickers, start=START, end=END, auto_adjust=True, progress=False)
    close = raw["Close"].copy()
    # Handle multi-level columns if needed
    if isinstance(close.columns, pd.MultiIndex):
        close.columns = close.columns.get_level_values(0)
    close = close.dropna(how="all")
    print(f"Data shape: {close.shape}, from {close.index[0].date()} to {close.index[-1].date()}")
    return close


# ── Indicator Helpers ──────────────────────────────────────────────────
def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - 100 / (1 + rs)


def realized_vol(series, window):
    """Annualized realized vol from log returns."""
    lr = np.log(series / series.shift(1))
    return lr.rolling(window).std() * np.sqrt(252)


def pct_below_high(series, window=20):
    """How far below rolling high (as fraction, e.g. 0.05 = 5% below)."""
    rh = series.rolling(window).max()
    return (rh - series) / rh


# ── Signal Generators ─────────────────────────────────────────────────
def gen_signals_A(close):
    """Variant A: VIX Term Structure (backwardation) + dip + RSI."""
    vix = close["^VIX"]
    vix3m = close["^VIX3M"]
    backwardation = vix > vix3m  # inverted term structure
    signals = {}
    for sym in UNIVERSE:
        px = close[sym]
        r = rsi(px)
        below = pct_below_high(px, 20)
        sig = backwardation & (below > 0.05) & (r < 40)
        signals[sym] = sig
    return signals


def gen_signals_B(close):
    """Variant B: VIX Mean Reversion (VIX >50% above 60d SMA) + dip + RSI."""
    vix = close["^VIX"]
    vix_sma60 = vix.rolling(60).mean()
    vix_spike = vix > 1.5 * vix_sma60
    signals = {}
    for sym in UNIVERSE:
        px = close[sym]
        r = rsi(px)
        below = pct_below_high(px, 20)
        sig = vix_spike & (below > 0.05) & (r < 40)
        signals[sym] = sig
    return signals


def gen_signals_C(close):
    """Variant C: Realized Vol Collapse (5d vol < 0.5 × 20d vol) + dip."""
    signals = {}
    for sym in UNIVERSE:
        px = close[sym]
        rv5 = realized_vol(px, 5)
        rv20 = realized_vol(px, 20)
        below = pct_below_high(px, 20)
        sig = (rv5 < 0.5 * rv20) & (below > 0.05)
        signals[sym] = sig
    return signals


def gen_signals_D(close):
    """Variant D: Cross-Volatility (VIX drops >3 pts in a day) + dip, buy next day."""
    vix = close["^VIX"]
    vix_drop = vix.diff() < -3
    # Shift forward: buy next day after VIX drop
    vix_drop_lag = vix_drop.shift(1).fillna(False)
    signals = {}
    for sym in UNIVERSE:
        px = close[sym]
        below = pct_below_high(px, 20)
        sig = vix_drop_lag & (below > 0.05)
        signals[sym] = sig
    return signals


def gen_signals_E(close):
    """Variant E: Vol-of-Vol (20d stdev of VIX changes > 2× 60d avg) + deep dip + low RSI."""
    vix = close["^VIX"]
    vix_chg = vix.diff()
    vol_of_vol = vix_chg.rolling(20).std()
    vov_avg60 = vol_of_vol.rolling(60).mean()
    vov_spike = vol_of_vol > 2 * vov_avg60
    signals = {}
    for sym in UNIVERSE:
        px = close[sym]
        r = rsi(px)
        below = pct_below_high(px, 20)
        sig = vov_spike & (below > 0.07) & (r < 35)
        signals[sym] = sig
    return signals


def gen_signals_F(close):
    """Variant F: Put-Call Ratio Proxy (VIX/VIX3M > 1.1) + dip + RSI."""
    vix = close["^VIX"]
    vix3m = close["^VIX3M"]
    ratio = vix / vix3m
    fear = ratio > 1.1
    signals = {}
    for sym in UNIVERSE:
        px = close[sym]
        r = rsi(px)
        below = pct_below_high(px, 20)
        sig = fear & (below > 0.05) & (r < 40)
        signals[sym] = sig
    return signals


# ── Backtest Engine ────────────────────────────────────────────────────
def run_backtest(close, signals, variant_name):
    """
    Simple event-driven backtest with position limits.
    Returns dict of metrics + trade list.
    """
    trade_start = pd.Timestamp(TRADE_START)
    dates = close.index[close.index >= trade_start]

    trades = []       # completed trades
    open_pos = []     # list of dicts: {sym, entry_date, entry_price, shares, exit_date_target}
    equity_curve = [CAPITAL]
    equity_dates = [dates[0]]
    cash = CAPITAL

    for i, dt in enumerate(dates):
        # ── Check exits ──
        still_open = []
        for pos in open_pos:
            days_held = (dt - pos["entry_date"]).days
            if days_held >= HOLD_DAYS:
                # Exit
                exit_price = close.loc[dt, pos["sym"]]
                if pd.isna(exit_price):
                    still_open.append(pos)
                    continue
                slip = exit_price * SLIPPAGE_BPS / 10000
                exit_price_adj = exit_price - slip  # selling
                pnl = (exit_price_adj - pos["entry_price"]) * pos["shares"]
                cash += exit_price_adj * pos["shares"]
                trades.append({
                    "sym": pos["sym"],
                    "entry_date": pos["entry_date"].strftime("%Y-%m-%d"),
                    "exit_date": dt.strftime("%Y-%m-%d"),
                    "entry_price": round(pos["entry_price"], 4),
                    "exit_price": round(exit_price_adj, 4),
                    "shares": pos["shares"],
                    "pnl": round(pnl, 2),
                    "ret": round(pnl / (pos["entry_price"] * pos["shares"]), 6),
                })
            else:
                still_open.append(pos)
        open_pos = still_open

        # ── Check entries ──
        if len(open_pos) < MAX_CONCURRENT:
            for sym in UNIVERSE:
                if len(open_pos) >= MAX_CONCURRENT:
                    break
                # Skip if already in position for this symbol
                if any(p["sym"] == sym for p in open_pos):
                    continue
                try:
                    sig_val = signals[sym].loc[dt]
                except (KeyError, IndexError):
                    continue
                if not sig_val or pd.isna(sig_val):
                    continue
                px = close.loc[dt, sym]
                if pd.isna(px) or px <= 0:
                    continue
                slip = px * SLIPPAGE_BPS / 10000
                entry_price = px + slip  # buying
                alloc = min(MAX_PER_TRADE, cash)
                if alloc < entry_price:
                    continue
                shares = int(alloc / entry_price)
                if shares < 1:
                    continue
                cost = entry_price * shares
                cash -= cost
                open_pos.append({
                    "sym": sym,
                    "entry_date": dt,
                    "entry_price": entry_price,
                    "shares": shares,
                })

        # ── Mark-to-market equity ──
        mtm = cash
        for pos in open_pos:
            px = close.loc[dt, pos["sym"]]
            if not pd.isna(px):
                mtm += px * pos["shares"]
        equity_curve.append(mtm)
        equity_dates.append(dt)

    # Force-close any remaining positions at last date
    last_dt = dates[-1]
    for pos in open_pos:
        px = close.loc[last_dt, pos["sym"]]
        if pd.isna(px):
            continue
        slip = px * SLIPPAGE_BPS / 10000
        exit_price_adj = px - slip
        pnl = (exit_price_adj - pos["entry_price"]) * pos["shares"]
        cash += exit_price_adj * pos["shares"]
        trades.append({
            "sym": pos["sym"],
            "entry_date": pos["entry_date"].strftime("%Y-%m-%d"),
            "exit_date": last_dt.strftime("%Y-%m-%d"),
            "entry_price": round(pos["entry_price"], 4),
            "exit_price": round(exit_price_adj, 4),
            "shares": pos["shares"],
            "pnl": round(pnl, 2),
            "ret": round(pnl / (pos["entry_price"] * pos["shares"]), 6),
        })

    eq = pd.Series(equity_curve, index=equity_dates)
    return trades, eq


# ── Metrics ────────────────────────────────────────────────────────────
def compute_metrics(trades, eq):
    if len(trades) == 0:
        return {"n_trades": 0, "sharpe": 0, "sortino": 0, "pf": 0, "wr": 0,
                "max_dd": 0, "total_ret": 0, "avg_ret": 0, "cagr": 0}

    rets = [t["ret"] for t in trades]
    rets = np.array(rets)
    wins = rets[rets > 0]
    losses = rets[rets <= 0]

    # Equity-based metrics
    eq_rets = eq.pct_change().dropna()
    sharpe = eq_rets.mean() / eq_rets.std() * np.sqrt(252) if eq_rets.std() > 0 else 0
    downside = eq_rets[eq_rets < 0].std()
    sortino = eq_rets.mean() / downside * np.sqrt(252) if downside > 0 else 0

    # Drawdown
    peak = eq.expanding().max()
    dd = (eq - peak) / peak
    max_dd = dd.min()

    # Profit factor
    gross_profit = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gross_loss = abs(sum(t["pnl"] for t in trades if t["pnl"] <= 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    wr = len(wins) / len(rets) if len(rets) > 0 else 0
    total_ret = (eq.iloc[-1] - CAPITAL) / CAPITAL
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / CAPITAL) ** (1 / years) - 1 if years > 0 else 0

    return {
        "n_trades": len(trades),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "pf": round(pf, 3),
        "wr": round(wr, 4),
        "max_dd": round(max_dd, 4),
        "total_ret": round(total_ret, 4),
        "avg_ret": round(float(np.mean(rets)), 6),
        "median_ret": round(float(np.median(rets)), 6),
        "cagr": round(cagr, 4),
    }


# ── 5-Gate Validation ─────────────────────────────────────────────────
def permutation_test(trades, eq, close, n_perms=1000):
    """
    Shuffle entry dates randomly, re-simulate, compute p-value.
    Null hypothesis: signal timing doesn't matter.
    """
    if len(trades) < 5:
        return 1.0  # not enough trades

    actual_sharpe = compute_metrics(trades, eq)["sharpe"]
    trade_dates = [pd.Timestamp(t["entry_date"]) for t in trades]
    trade_syms = [t["sym"] for t in trades]

    valid_dates = close.index[close.index >= pd.Timestamp(TRADE_START)]
    count_better = 0

    for _ in range(n_perms):
        # Random entry dates
        rand_dates = np.random.choice(valid_dates, size=len(trades), replace=True)
        rand_rets = []
        for rd, sym in zip(rand_dates, trade_syms):
            idx = close.index.get_loc(rd)
            exit_idx = min(idx + HOLD_DAYS, len(close) - 1)
            entry_px = close.iloc[idx][sym]
            exit_px = close.iloc[exit_idx][sym]
            if pd.isna(entry_px) or pd.isna(exit_px) or entry_px <= 0:
                continue
            ret = (exit_px - entry_px) / entry_px
            rand_rets.append(ret)

        if len(rand_rets) < 3:
            continue
        rand_rets = np.array(rand_rets)
        rand_sharpe = rand_rets.mean() / rand_rets.std() * np.sqrt(252 / HOLD_DAYS) if rand_rets.std() > 0 else 0
        if rand_sharpe >= actual_sharpe:
            count_better += 1

    return count_better / n_perms


def regime_gap(trades, close):
    """
    Compute Sharpe in bull vs bear regimes (SPY 20d return).
    Returns |sharpe_bull - sharpe_bear| / max(|sharpe_bull|, |sharpe_bear|).
    """
    spy = close["SPY"]
    spy_ret20 = spy.pct_change(20)

    bull_rets, bear_rets = [], []
    for t in trades:
        dt = pd.Timestamp(t["entry_date"])
        try:
            r20 = spy_ret20.loc[dt]
        except KeyError:
            # Find nearest
            idx = spy_ret20.index.get_indexer([dt], method="nearest")[0]
            r20 = spy_ret20.iloc[idx]
        if pd.isna(r20):
            continue
        if r20 >= 0:
            bull_rets.append(t["ret"])
        else:
            bear_rets.append(t["ret"])

    def sharpe_from_rets(r):
        r = np.array(r)
        if len(r) < 2 or r.std() == 0:
            return 0
        return r.mean() / r.std() * np.sqrt(252 / HOLD_DAYS)

    sb = sharpe_from_rets(bull_rets)
    sr = sharpe_from_rets(bear_rets)
    denom = max(abs(sb), abs(sr))
    if denom == 0:
        return 1.0
    return abs(sb - sr) / denom


def validate_5gate(trades, eq, close, variant_name):
    """Run 5-gate validation. Returns dict with pass/fail for each gate."""
    m = compute_metrics(trades, eq)
    gates = {}

    # Gate 1: Sharpe > 0.5
    gates["sharpe_pass"] = m["sharpe"] > 0.5
    gates["sharpe_val"] = m["sharpe"]

    # Gate 2: Permutation test p < 0.05
    p_val = permutation_test(trades, eq, close)
    gates["perm_pass"] = p_val < 0.05
    gates["perm_pval"] = round(p_val, 4)

    # Gate 3: Regime gap < 0.5
    rg = regime_gap(trades, close)
    gates["regime_pass"] = rg < 0.5
    gates["regime_gap"] = round(rg, 4)

    # Gate 4: Max DD > -50%
    gates["dd_pass"] = m["max_dd"] > -0.50
    gates["max_dd"] = m["max_dd"]

    # Gate 5: At least 20 trades
    gates["trade_count_pass"] = m["n_trades"] >= 20
    gates["n_trades"] = m["n_trades"]

    gates["all_pass"] = all([
        gates["sharpe_pass"], gates["perm_pass"], gates["regime_pass"],
        gates["dd_pass"], gates["trade_count_pass"]
    ])

    return gates


# ── Main ───────────────────────────────────────────────────────────────
def main():
    close = download_data()

    # Check VIX3M availability
    has_vix3m = "^VIX3M" in close.columns and close["^VIX3M"].notna().sum() > 100
    if not has_vix3m:
        print("WARNING: ^VIX3M data sparse or missing. Variants A/F will use synthetic proxy.")
        # Synthetic VIX3M: 60-day SMA of VIX as proxy for 3-month implied vol
        close["^VIX3M"] = close["^VIX"].rolling(60).mean()

    generators = {
        "A_vix_term_structure": gen_signals_A,
        "B_vix_mean_reversion": gen_signals_B,
        "C_realized_vol_collapse": gen_signals_C,
        "D_cross_vol_vix_drop": gen_signals_D,
        "E_vol_of_vol": gen_signals_E,
        "F_put_call_proxy": gen_signals_F,
    }

    all_results = {}

    for name, gen_func in generators.items():
        print(f"\n{'='*60}")
        print(f"  Variant {name}")
        print(f"{'='*60}")

        signals = gen_func(close)

        # Count signal days
        total_sigs = sum(s.sum() for s in signals.values() if hasattr(s, 'sum'))
        print(f"  Total signal-days across universe: {int(total_sigs)}")

        trades, eq = run_backtest(close, signals, name)
        metrics = compute_metrics(trades, eq)
        gates = validate_5gate(trades, eq, close, name)

        print(f"  Trades: {metrics['n_trades']}  |  Sharpe: {metrics['sharpe']}  |  "
              f"Sortino: {metrics['sortino']}  |  PF: {metrics['pf']}")
        print(f"  WR: {metrics['wr']:.1%}  |  MaxDD: {metrics['max_dd']:.1%}  |  "
              f"Total Ret: {metrics['total_ret']:.1%}  |  CAGR: {metrics['cagr']:.1%}")
        print(f"  Avg Ret/Trade: {metrics['avg_ret']:.4%}  |  Median: {metrics['median_ret']:.4%}")
        print(f"  Gates: Sharpe={'PASS' if gates['sharpe_pass'] else 'FAIL'}  "
              f"Perm={'PASS' if gates['perm_pass'] else 'FAIL'}(p={gates['perm_pval']})  "
              f"Regime={'PASS' if gates['regime_pass'] else 'FAIL'}(gap={gates['regime_gap']})  "
              f"DD={'PASS' if gates['dd_pass'] else 'FAIL'}  "
              f"Count={'PASS' if gates['trade_count_pass'] else 'FAIL'}")
        print(f"  >> ALL GATES: {'PASS ✓' if gates['all_pass'] else 'FAIL ✗'}")

        # Top symbols
        if trades:
            sym_pnl = {}
            sym_cnt = {}
            for t in trades:
                sym_pnl[t["sym"]] = sym_pnl.get(t["sym"], 0) + t["pnl"]
                sym_cnt[t["sym"]] = sym_cnt.get(t["sym"], 0) + 1
            top = sorted(sym_pnl.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"  Top symbols: {', '.join(f'{s}: ${p:.0f} ({sym_cnt[s]}t)' for s, p in top)}")

        all_results[name] = {
            "metrics": metrics,
            "gates": gates,
            "trades": trades[:200],  # cap stored trades for file size
            "trade_count_total": len(trades),
        }

    # ── Summary Table ──────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  SUMMARY")
    print(f"{'='*80}")
    print(f"{'Variant':<28} {'Trades':>6} {'Sharpe':>7} {'Sortino':>8} {'PF':>6} "
          f"{'WR':>6} {'MaxDD':>7} {'CAGR':>7} {'Gates':>6}")
    print("-" * 80)

    passed = []
    for name, res in all_results.items():
        m = res["metrics"]
        g = res["gates"]
        status = "PASS" if g["all_pass"] else "FAIL"
        if g["all_pass"]:
            passed.append(name)
        print(f"{name:<28} {m['n_trades']:>6} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} "
              f"{m['pf']:>6.2f} {m['wr']:>5.1%} {m['max_dd']:>6.1%} {m['cagr']:>6.1%} "
              f"{status:>6}")

    print(f"\nPassed all 5 gates: {len(passed)}/{len(all_results)}")
    if passed:
        print(f"Winners: {', '.join(passed)}")

    # ── Save Results ───────────────────────────────────────────────────
    output = {
        "strategy": "Vol Surface Signals for Quality Stock Mean Reversion",
        "run_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "universe": UNIVERSE,
            "capital": CAPITAL,
            "max_per_trade": MAX_PER_TRADE,
            "max_concurrent": MAX_CONCURRENT,
            "slippage_bps": SLIPPAGE_BPS,
            "hold_days": HOLD_DAYS,
            "period": f"{TRADE_START} to {END}",
        },
        "variants": all_results,
        "summary": {
            "total_variants": len(all_results),
            "passed_all_gates": len(passed),
            "passed_names": passed,
        },
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()

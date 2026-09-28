#!/usr/bin/env python3
"""
Merger Arbitrage / M&A Event Trading — Proxy Backtest
=====================================================
Detects M&A-like events via large single-day gaps, then tests
various holding strategies on a universe of 30 stocks.

6 Variants:
  A) Base: buy on 15%+ gap up, hold 20d
  B) Gap fade: SHORT after 30%+ gap up, hold 10d
  C) Spread capture: buy on 15-30% gap, hold 40d
  D) Volume confirmed: gap 15%+ AND volume >5x avg, hold 20d
  E) Sector filter: tech/biotech only, gap 15%+, hold 20d
  F) Multi-event: 2+ large moves in 60d window, hold 20d

OOT: Jan 2022 – Jul 2026 | Account: $645 | Slippage: 0.02% each way
Validation: 5-gate framework (Sharpe>0.5, perm p<0.05, regime gap<0.5, MaxDD>-50%, ≥20 trades)
"""

import json
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
except ImportError:
    print("ERROR: pip install yfinance")
    sys.exit(1)

# ── Config ──────────────────────────────────────────────────────────────
UNIVERSE = [
    "ATVI", "VMW", "CTXS", "CERN", "XLNX", "ZNGA", "MXIM", "NUVA", "FORG",
    "MGM", "PYPL", "PINS", "SNAP", "RBLX", "UBER", "LYFT", "DASH", "ABNB",
    "SQ", "SHOP", "COIN", "RIVN", "PLTR", "AMD", "INTC", "NFLX", "DIS", "BA", "GE",
]
# TWTR delisted — excluded

TECH_BIOTECH = {
    "ATVI", "VMW", "CTXS", "CERN", "XLNX", "ZNGA", "MXIM", "NUVA", "FORG",
    "PYPL", "PINS", "SNAP", "RBLX", "SQ", "SHOP", "COIN", "PLTR", "AMD",
    "INTC", "NFLX",
}

ACCOUNT = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02% each way → 0.04% round-trip
OOT_START = "2022-01-01"
OOT_END = "2026-07-29"
DATA_START = "2021-06-01"  # extra lookback for volume avg + SPY SMA

RESULTS_PATH = "/home/jupiter/Lvl3Quant/data/merger_arb_proxy_results.json"


# ── Data Download ───────────────────────────────────────────────────────
def download_data():
    print("Downloading price data...")
    tickers = UNIVERSE + ["SPY"]
    data = {}
    for t in tickers:
        try:
            df = yf.download(t, start=DATA_START, end=OOT_END, progress=False, auto_adjust=True)
            if df is not None and len(df) > 50:
                # Flatten multi-level columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                data[t] = df
        except Exception as e:
            print(f"  skip {t}: {e}")
    print(f"  Downloaded {len(data)} tickers")
    return data


# ── SPY Regime ──────────────────────────────────────────────────────────
def compute_spy_regime(spy_df):
    """Bull = SPY > 200-SMA, Bear = SPY < 200-SMA"""
    sma200 = spy_df["Close"].rolling(200).mean()
    regime = pd.Series("bull", index=spy_df.index)
    regime[spy_df["Close"] < sma200] = "bear"
    return regime


# ── Core Backtest Engine ────────────────────────────────────────────────
def run_variant(data, spy_regime, variant_name, signal_fn, hold_days, direction="long"):
    """
    signal_fn(ticker, df, idx, data) -> bool  — whether to enter on day idx
    direction: 'long' or 'short'
    Returns list of trade dicts.
    """
    trades = []
    for ticker, df in data.items():
        if ticker == "SPY":
            continue
        df = df.copy()
        df["ret_1d"] = df["Close"].pct_change()
        df["vol_avg_20"] = df["Volume"].rolling(20).mean()

        oot_mask = df.index >= pd.Timestamp(OOT_START)
        oot_idx = df.index[oot_mask]

        for i, dt in enumerate(oot_idx):
            pos = df.index.get_loc(dt)
            if pos < 21:  # need lookback
                continue
            if not signal_fn(ticker, df, pos, data):
                continue
            # Entry next day open (we detect signal at close, enter next open)
            if pos + 1 >= len(df):
                continue
            entry_price = df["Open"].iloc[pos + 1]
            # Exit after hold_days
            exit_pos = min(pos + 1 + hold_days, len(df) - 1)
            exit_price = df["Close"].iloc[exit_pos]

            # Slippage
            if direction == "long":
                entry_eff = entry_price * (1 + SLIPPAGE_PCT)
                exit_eff = exit_price * (1 - SLIPPAGE_PCT)
                ret = (exit_eff - entry_eff) / entry_eff
            else:  # short
                entry_eff = entry_price * (1 - SLIPPAGE_PCT)
                exit_eff = exit_price * (1 + SLIPPAGE_PCT)
                ret = (entry_eff - exit_eff) / entry_eff

            entry_date = df.index[pos + 1]
            exit_date = df.index[exit_pos]

            # Regime at entry
            regime = "unknown"
            if entry_date in spy_regime.index:
                regime = spy_regime.loc[entry_date]

            trades.append({
                "ticker": ticker,
                "entry_date": str(entry_date.date()),
                "exit_date": str(exit_date.date()),
                "entry_price": round(float(entry_price), 2),
                "exit_price": round(float(exit_price), 2),
                "ret": round(float(ret), 6),
                "direction": direction,
                "regime": regime,
                "hold_days": int(exit_pos - pos - 1),
                "signal_day_ret": round(float(df["ret_1d"].iloc[pos]), 4),
            })

    return trades


# ── Signal Functions ────────────────────────────────────────────────────
def signal_base(ticker, df, pos, data):
    """A) Gap up ≥15%"""
    return df["ret_1d"].iloc[pos] >= 0.15


def signal_gap_fade(ticker, df, pos, data):
    """B) Gap up ≥30% (short)"""
    return df["ret_1d"].iloc[pos] >= 0.30


def signal_spread_capture(ticker, df, pos, data):
    """C) Gap 15-30% (merger range)"""
    r = df["ret_1d"].iloc[pos]
    return 0.15 <= r <= 0.30


def signal_volume_confirmed(ticker, df, pos, data):
    """D) Gap ≥15% AND volume >5x 20d avg"""
    r = df["ret_1d"].iloc[pos]
    vol = df["Volume"].iloc[pos]
    vol_avg = df["vol_avg_20"].iloc[pos]
    if pd.isna(vol_avg) or vol_avg <= 0:
        return False
    return r >= 0.15 and vol > 5 * vol_avg


def signal_sector_filter(ticker, df, pos, data):
    """E) Gap ≥15% AND ticker in tech/biotech"""
    if ticker not in TECH_BIOTECH:
        return False
    return df["ret_1d"].iloc[pos] >= 0.15


def signal_multi_event(ticker, df, pos, data):
    """F) Current gap ≥10% AND at least 1 other ≥10% gap in prior 60 days"""
    r = df["ret_1d"].iloc[pos]
    if r < 0.10:
        return False
    lookback = df["ret_1d"].iloc[max(0, pos - 60):pos]
    big_moves = (lookback.abs() >= 0.10).sum()
    return big_moves >= 1


# ── Metrics ─────────────────────────────────────────────────────────────
def compute_metrics(trades, account=ACCOUNT):
    if not trades:
        return {
            "n_trades": 0, "sharpe": 0, "sortino": 0, "pf": 0,
            "wr": 0, "max_dd_pct": 0, "total_ret_pct": 0,
            "avg_ret_pct": 0, "sharpe_bull": 0, "sharpe_bear": 0,
        }

    rets = np.array([t["ret"] for t in trades])
    n = len(rets)

    # Equity curve (sequential, equal-size positions using full account)
    equity = [account]
    for r in rets:
        equity.append(equity[-1] * (1 + r))
    equity = np.array(equity)

    # Max drawdown
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = float(dd.min())

    # Annualized Sharpe (assume ~20 trades/year avg holding)
    mean_ret = rets.mean()
    std_ret = rets.std() if len(rets) > 1 else 1e-9
    # Annualize: assume each trade is independent
    trades_per_year = max(1, 252 / max(np.mean([t["hold_days"] for t in trades]), 1))
    sharpe = (mean_ret / max(std_ret, 1e-9)) * np.sqrt(min(trades_per_year, 252))

    # Sortino
    downside = rets[rets < 0]
    down_std = downside.std() if len(downside) > 1 else 1e-9
    sortino = (mean_ret / max(down_std, 1e-9)) * np.sqrt(min(trades_per_year, 252))

    # Profit factor
    gross_profit = rets[rets > 0].sum() if (rets > 0).any() else 0
    gross_loss = abs(rets[rets < 0].sum()) if (rets < 0).any() else 1e-9
    pf = gross_profit / max(gross_loss, 1e-9)

    # Win rate
    wr = (rets > 0).sum() / n

    # Per-regime Sharpe
    bull_rets = [t["ret"] for t in trades if t["regime"] == "bull"]
    bear_rets = [t["ret"] for t in trades if t["regime"] == "bear"]

    def _sharpe(r_list):
        if len(r_list) < 2:
            return 0.0
        r = np.array(r_list)
        return float((r.mean() / max(r.std(), 1e-9)) * np.sqrt(min(trades_per_year, 252)))

    sharpe_bull = _sharpe(bull_rets)
    sharpe_bear = _sharpe(bear_rets)

    total_ret = (equity[-1] / equity[0] - 1) * 100

    return {
        "n_trades": n,
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "pf": round(pf, 3),
        "wr": round(wr, 4),
        "max_dd_pct": round(max_dd * 100, 2),
        "total_ret_pct": round(total_ret, 2),
        "avg_ret_pct": round(mean_ret * 100, 4),
        "sharpe_bull": round(sharpe_bull, 3),
        "sharpe_bear": round(sharpe_bear, 3),
        "n_bull": len(bull_rets),
        "n_bear": len(bear_rets),
        "final_equity": round(float(equity[-1]), 2),
    }


# ── Permutation Test ────────────────────────────────────────────────────
def permutation_test(trades, n_perm=1000):
    if len(trades) < 5:
        return 1.0
    rets = np.array([t["ret"] for t in trades])
    observed_mean = rets.mean()
    count_ge = 0
    for _ in range(n_perm):
        shuffled = np.random.choice(rets, size=len(rets), replace=True)
        # Randomly flip signs to break signal-return relationship
        signs = np.random.choice([-1, 1], size=len(rets))
        perm_mean = (rets * signs).mean()
        if perm_mean >= observed_mean:
            count_ge += 1
    return count_ge / n_perm


# ── 5-Gate Validation ──────────────────────────────────────────────────
def validate_gates(metrics, perm_p):
    gates = {}
    # Gate 1: Sharpe > 0.5
    gates["sharpe_gt_0.5"] = metrics["sharpe"] > 0.5
    # Gate 2: Permutation p < 0.05
    gates["perm_p_lt_0.05"] = perm_p < 0.05
    # Gate 3: Regime gap < 0.5
    s_bull = abs(metrics["sharpe_bull"])
    s_bear = abs(metrics["sharpe_bear"])
    max_s = max(s_bull, s_bear)
    regime_gap = abs(s_bull - s_bear) / max_s if max_s > 0 else 0
    gates["regime_gap_lt_0.5"] = regime_gap < 0.5
    gates["regime_gap_value"] = round(regime_gap, 3)
    # Gate 4: MaxDD > -50%
    gates["maxdd_gt_neg50"] = metrics["max_dd_pct"] > -50
    # Gate 5: ≥ 20 trades
    gates["trades_ge_20"] = metrics["n_trades"] >= 20

    gates["passed"] = sum([
        gates["sharpe_gt_0.5"],
        gates["perm_p_lt_0.05"],
        gates["regime_gap_lt_0.5"],
        gates["maxdd_gt_neg50"],
        gates["trades_ge_20"],
    ])
    gates["total"] = 5
    gates["perm_p"] = round(perm_p, 4)
    return gates


# ── Main ────────────────────────────────────────────────────────────────
def main():
    np.random.seed(42)

    data = download_data()
    if "SPY" not in data:
        print("ERROR: SPY data required for regime classification")
        sys.exit(1)

    spy_regime = compute_spy_regime(data["SPY"])

    variants = [
        ("A_base", signal_base, 20, "long"),
        ("B_gap_fade", signal_gap_fade, 10, "short"),
        ("C_spread_capture", signal_spread_capture, 40, "long"),
        ("D_volume_confirmed", signal_volume_confirmed, 20, "long"),
        ("E_sector_filter", signal_sector_filter, 20, "long"),
        ("F_multi_event", signal_multi_event, 20, "long"),
    ]

    all_results = {}

    print("\n" + "=" * 80)
    print("MERGER ARBITRAGE / M&A EVENT TRADING — PROXY BACKTEST")
    print(f"OOT: {OOT_START} to {OOT_END} | Account: ${ACCOUNT} | Slippage: {SLIPPAGE_PCT*100:.2f}% each way")
    print("=" * 80)

    for name, signal_fn, hold_days, direction in variants:
        print(f"\n{'─' * 60}")
        print(f"Variant {name} | hold={hold_days}d | dir={direction}")
        print(f"{'─' * 60}")

        trades = run_variant(data, spy_regime, name, signal_fn, hold_days, direction)
        metrics = compute_metrics(trades)
        perm_p = permutation_test(trades)
        gates = validate_gates(metrics, perm_p)

        print(f"  Trades: {metrics['n_trades']}  (bull={metrics['n_bull']}, bear={metrics['n_bear']})")
        print(f"  Sharpe: {metrics['sharpe']:.3f}  Sortino: {metrics['sortino']:.3f}  PF: {metrics['pf']:.3f}  WR: {metrics['wr']:.1%}")
        print(f"  Total Return: {metrics['total_ret_pct']:.1f}%  MaxDD: {metrics['max_dd_pct']:.1f}%  Final Eq: ${metrics['final_equity']:.2f}")
        print(f"  Sharpe(bull): {metrics['sharpe_bull']:.3f}  Sharpe(bear): {metrics['sharpe_bear']:.3f}  Regime Gap: {gates['regime_gap_value']:.3f}")
        print(f"  Perm p-value: {gates['perm_p']:.4f}")
        print()
        print(f"  GATE RESULTS ({gates['passed']}/{gates['total']}):")
        print(f"    [{'PASS' if gates['sharpe_gt_0.5'] else 'FAIL'}] Sharpe > 0.5: {metrics['sharpe']:.3f}")
        print(f"    [{'PASS' if gates['perm_p_lt_0.05'] else 'FAIL'}] Perm p < 0.05: {gates['perm_p']:.4f}")
        print(f"    [{'PASS' if gates['regime_gap_lt_0.5'] else 'FAIL'}] Regime gap < 0.5: {gates['regime_gap_value']:.3f}")
        print(f"    [{'PASS' if gates['maxdd_gt_neg50'] else 'FAIL'}] MaxDD > -50%: {metrics['max_dd_pct']:.1f}%")
        print(f"    [{'PASS' if gates['trades_ge_20'] else 'FAIL'}] Trades >= 20: {metrics['n_trades']}")

        # Top trades
        if trades:
            sorted_trades = sorted(trades, key=lambda x: x["ret"], reverse=True)
            print(f"\n  Top 3 winners:")
            for t in sorted_trades[:3]:
                print(f"    {t['ticker']} {t['entry_date']} → {t['exit_date']}: {t['ret']*100:+.2f}% (gap={t['signal_day_ret']*100:.1f}%)")
            print(f"  Top 3 losers:")
            for t in sorted_trades[-3:]:
                print(f"    {t['ticker']} {t['entry_date']} → {t['exit_date']}: {t['ret']*100:+.2f}% (gap={t['signal_day_ret']*100:.1f}%)")

        all_results[name] = {
            "metrics": metrics,
            "gates": gates,
            "n_sample_trades": trades[:10] if trades else [],
        }

    # ── Summary Table ───────────────────────────────────────────────────
    print("\n\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Variant':<22} {'Trades':>6} {'Sharpe':>7} {'Sortino':>8} {'PF':>6} {'WR':>6} {'MaxDD':>7} {'TotRet':>8} {'Gates':>6}")
    print("-" * 80)
    for name in all_results:
        m = all_results[name]["metrics"]
        g = all_results[name]["gates"]
        status = "PASS" if g["passed"] == 5 else f"{g['passed']}/5"
        print(f"{name:<22} {m['n_trades']:>6} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} {m['pf']:>6.3f} {m['wr']:>5.1%} {m['max_dd_pct']:>6.1f}% {m['total_ret_pct']:>7.1f}% {status:>6}")

    # Any passes?
    passes = [n for n, r in all_results.items() if r["gates"]["passed"] == 5]
    print(f"\nVariants passing all 5 gates: {passes if passes else 'NONE'}")

    # Save
    Path(RESULTS_PATH).parent.mkdir(parents=True, exist_ok=True)
    # Convert for JSON serialization
    for name in all_results:
        g = all_results[name]["gates"]
        for k, v in g.items():
            if isinstance(v, (np.bool_, np.integer)):
                g[k] = int(v)
            elif isinstance(v, np.floating):
                g[k] = float(v)
        m = all_results[name]["metrics"]
        for k, v in m.items():
            if isinstance(v, (np.bool_, np.integer)):
                m[k] = int(v)
            elif isinstance(v, np.floating):
                m[k] = float(v)

    with open(RESULTS_PATH, "w") as f:
        json.dump({
            "strategy": "merger_arb_proxy",
            "run_date": str(datetime.now()),
            "oot_period": f"{OOT_START} to {OOT_END}",
            "account": ACCOUNT,
            "slippage_pct": SLIPPAGE_PCT,
            "universe_size": len(UNIVERSE),
            "variants": all_results,
        }, f, indent=2, default=str)

    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()

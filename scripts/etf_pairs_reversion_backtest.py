#!/usr/bin/env python3
"""
ETF Pairs Mean Reversion Backtest
Academic basis: Gatev, Goetzmann & Rouwenhorst (2006)

6 Variants:
A. SPY/QQQ Spread
B. GLD/GDX Spread
C. XLK/XLC Spread
D. TLT/IEF Spread
E. Multi-Pair Portfolio (A+B+C+D)
F. Adaptive Threshold (SPY/QQQ with VIX gate)

OOT: Jan 2022 - Jul 2026
Capital: $645, $0 commission, 0.02% slippage
Regime: Bull = SPY > 200-SMA, Bear = SPY < 200-SMA
5-Gate: Sharpe>0.5, Perm p<0.05 (1000 iters), Regime gap<0.5, MaxDD>-50%, >=20 trades
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

# ── Config ──────────────────────────────────────────────────────────────
START = "2021-01-01"  # extra lookback for 200-SMA + 20-day z-score
END = "2026-07-30"
OOT_START = "2022-01-01"
OOT_END = "2026-07-30"
CAPITAL = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02%
ZSCORE_WINDOW = 20
ZSCORE_ENTRY = 2.0
ZSCORE_EXIT = 0.5
MAX_HOLD_DAYS = 10
SMA_PERIOD = 200
PERM_ITERS = 1000
np.random.seed(42)

# ── Download Data ───────────────────────────────────────────────────────
TICKERS = ["SPY", "QQQ", "GLD", "GDX", "XLK", "XLC", "TLT", "IEF", "^VIX"]

print("Downloading data...")
data = yf.download(TICKERS, start=START, end=END, auto_adjust=True, progress=False)

# Handle multi-level columns from yfinance
if isinstance(data.columns, pd.MultiIndex):
    close = data["Close"]
else:
    close = data

# Rename ^VIX to VIX
if "^VIX" in close.columns:
    close = close.rename(columns={"^VIX": "VIX"})

close = close.dropna()
print(f"Data: {close.index[0].date()} to {close.index[-1].date()}, {len(close)} days")

# ── Regime Classification ──────────────────────────────────────────────
spy_sma200 = close["SPY"].rolling(SMA_PERIOD).mean()
regime = pd.Series("Bull", index=close.index)
regime[close["SPY"] < spy_sma200] = "Bear"

# ── Helper Functions ───────────────────────────────────────────────────

def compute_spread_zscore(etf1, etf2, window=ZSCORE_WINDOW):
    """Z-score of daily return spread between etf1 and etf2."""
    ret1 = close[etf1].pct_change()
    ret2 = close[etf2].pct_change()
    spread = ret2 - ret1  # positive = etf2 outperforms etf1
    z = (spread - spread.rolling(window).mean()) / spread.rolling(window).std()
    return z, spread


def run_pair_backtest(etf_buy, etf_benchmark, z_entry=ZSCORE_ENTRY, z_exit=ZSCORE_EXIT,
                      max_hold=MAX_HOLD_DAYS, vix_gate=None, vix_threshold=None):
    """
    When z-score of (etf_buy returns - etf_benchmark returns) < -z_entry,
    etf_buy has underperformed => buy etf_buy expecting mean reversion.
    Exit when z > -z_exit or max_hold exceeded.
    """
    zscore, spread = compute_spread_zscore(etf_benchmark, etf_buy)

    oot_mask = (close.index >= OOT_START) & (close.index <= OOT_END)
    dates = close.index[oot_mask]

    trades = []
    position = None  # dict with entry info

    for i, dt in enumerate(dates):
        z_val = zscore.loc[dt]
        if np.isnan(z_val):
            continue

        # Check exit
        if position is not None:
            hold_days = (dt - position["entry_date"]).days
            if z_val > -z_exit or hold_days >= max_hold:
                # Exit
                exit_price = close[etf_buy].loc[dt]
                exit_price_adj = exit_price * (1 - SLIPPAGE_PCT)  # selling
                pnl_pct = (exit_price_adj / position["entry_price"]) - 1
                trades.append({
                    "entry_date": str(position["entry_date"].date()),
                    "exit_date": str(dt.date()),
                    "pair": f"{etf_buy}/{etf_benchmark}",
                    "entry_price": round(position["entry_price"], 4),
                    "exit_price": round(exit_price_adj, 4),
                    "pnl_pct": round(pnl_pct, 6),
                    "hold_days": hold_days,
                    "regime": position["regime"],
                    "z_entry": round(position["z_val"], 3),
                    "z_exit": round(z_val, 3),
                })
                position = None

        # Check entry (only if no position)
        if position is None and z_val < -z_entry:
            # VIX gate for adaptive variant
            if vix_gate is not None:
                vix_val = close["VIX"].loc[dt] if dt in close.index else None
                if vix_val is None or vix_val < vix_threshold:
                    continue

            entry_price = close[etf_buy].loc[dt]
            entry_price_adj = entry_price * (1 + SLIPPAGE_PCT)  # buying
            position = {
                "entry_date": dt,
                "entry_price": entry_price_adj,
                "z_val": z_val,
                "regime": regime.loc[dt],
            }

    # Close any open position at end
    if position is not None:
        exit_price = close[etf_buy].iloc[-1] * (1 - SLIPPAGE_PCT)
        pnl_pct = (exit_price / position["entry_price"]) - 1
        trades.append({
            "entry_date": str(position["entry_date"].date()),
            "exit_date": str(dates[-1].date()),
            "pair": f"{etf_buy}/{etf_benchmark}",
            "entry_price": round(position["entry_price"], 4),
            "exit_price": round(exit_price, 4),
            "pnl_pct": round(pnl_pct, 6),
            "hold_days": (dates[-1] - position["entry_date"]).days,
            "regime": position["regime"],
            "z_entry": round(position["z_val"], 3),
            "z_exit": 0.0,
        })

    return trades


def compute_metrics(trades, capital=CAPITAL):
    """Compute performance metrics from trade list."""
    if not trades:
        return {"sharpe": 0, "sortino": 0, "pf": 0, "wr": 0, "max_dd_pct": 0,
                "total_return_pct": 0, "n_trades": 0, "avg_hold_days": 0,
                "cagr_pct": 0, "calmar": 0}

    pnls = np.array([t["pnl_pct"] for t in trades])
    n = len(pnls)

    # Build equity curve
    equity = [capital]
    for p in pnls:
        equity.append(equity[-1] * (1 + p))
    equity = np.array(equity)

    total_ret = (equity[-1] / equity[0]) - 1

    # Max drawdown
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = dd.min()

    # Sharpe (annualized, assume ~50 trades/yr as proxy, use daily-like)
    if pnls.std() > 0:
        sharpe = (pnls.mean() / pnls.std()) * np.sqrt(min(n, 252))
    else:
        sharpe = 0.0

    # Sortino
    downside = pnls[pnls < 0]
    if len(downside) > 0 and downside.std() > 0:
        sortino = (pnls.mean() / downside.std()) * np.sqrt(min(n, 252))
    else:
        sortino = sharpe * 1.5 if sharpe > 0 else 0.0

    # Profit factor
    gross_profit = pnls[pnls > 0].sum() if (pnls > 0).any() else 0
    gross_loss = abs(pnls[pnls < 0].sum()) if (pnls < 0).any() else 0.001
    pf = gross_profit / gross_loss if gross_loss > 0 else 99.9

    # Win rate
    wr = (pnls > 0).sum() / n

    # Avg hold
    avg_hold = np.mean([t["hold_days"] for t in trades])

    # CAGR
    first_date = pd.Timestamp(trades[0]["entry_date"])
    last_date = pd.Timestamp(trades[-1]["exit_date"])
    years = max((last_date - first_date).days / 365.25, 0.1)
    cagr = (equity[-1] / equity[0]) ** (1 / years) - 1

    # Calmar
    calmar = cagr / abs(max_dd) if max_dd < 0 else 0.0

    return {
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "pf": round(pf, 3),
        "wr": round(wr, 4),
        "max_dd_pct": round(max_dd * 100, 2),
        "total_return_pct": round(total_ret * 100, 2),
        "n_trades": n,
        "avg_hold_days": round(avg_hold, 1),
        "cagr_pct": round(cagr * 100, 2),
        "calmar": round(calmar, 3),
        "final_equity": round(equity[-1], 2),
    }


def regime_metrics(trades):
    """Compute per-regime Sharpe."""
    bull = [t for t in trades if t["regime"] == "Bull"]
    bear = [t for t in trades if t["regime"] == "Bear"]
    m_bull = compute_metrics(bull) if bull else {"sharpe": 0, "n_trades": 0}
    m_bear = compute_metrics(bear) if bear else {"sharpe": 0, "n_trades": 0}
    return m_bull, m_bear


def permutation_test(trades, n_iters=PERM_ITERS):
    """Permutation test: shuffle trade signs, compute fraction with Sharpe >= observed."""
    if len(trades) < 5:
        return 1.0
    pnls = np.array([t["pnl_pct"] for t in trades])
    obs_sharpe = pnls.mean() / pnls.std() if pnls.std() > 0 else 0
    count = 0
    for _ in range(n_iters):
        shuffled = pnls * np.random.choice([-1, 1], size=len(pnls))
        sh = shuffled.mean() / shuffled.std() if shuffled.std() > 0 else 0
        if sh >= obs_sharpe:
            count += 1
    return count / n_iters


def validate_5gate(metrics, perm_p, regime_bull, regime_bear):
    """5-Gate validation."""
    gates = {}
    gates["sharpe_gt_0.5"] = metrics["sharpe"] > 0.5
    gates["perm_p_lt_0.05"] = perm_p < 0.05

    # Regime gap
    s_bull = regime_bull["sharpe"]
    s_bear = regime_bear["sharpe"]
    denom = max(abs(s_bull), abs(s_bear), 0.001)
    regime_gap = abs(s_bull - s_bear) / denom
    gates["regime_gap_lt_0.5"] = regime_gap < 0.5
    gates["regime_gap_value"] = round(regime_gap, 3)

    gates["max_dd_gt_neg50"] = metrics["max_dd_pct"] > -50
    gates["min_20_trades"] = metrics["n_trades"] >= 20
    gates["all_pass"] = all([gates["sharpe_gt_0.5"], gates["perm_p_lt_0.05"],
                             gates["regime_gap_lt_0.5"], gates["max_dd_gt_neg50"],
                             gates["min_20_trades"]])
    return gates


# ── Run Variants ────────────────────────────────────────────────────────

results = {}

# Variant A: SPY/QQQ
print("\n=== Variant A: SPY/QQQ Spread ===")
trades_a = run_pair_backtest("QQQ", "SPY")
metrics_a = compute_metrics(trades_a)
bull_a, bear_a = regime_metrics(trades_a)
perm_a = permutation_test(trades_a)
gates_a = validate_5gate(metrics_a, perm_a, bull_a, bear_a)
results["A_SPY_QQQ"] = {"metrics": metrics_a, "perm_p": round(perm_a, 4),
                         "regime_bull": bull_a, "regime_bear": bear_a,
                         "gates": gates_a, "n_trades": len(trades_a)}
print(f"  Trades: {len(trades_a)}, Sharpe: {metrics_a['sharpe']}, "
      f"Return: {metrics_a['total_return_pct']}%, MaxDD: {metrics_a['max_dd_pct']}%, "
      f"WR: {metrics_a['wr']:.1%}, PF: {metrics_a['pf']}, Perm-p: {perm_a:.4f}")
print(f"  Gates: {'PASS' if gates_a['all_pass'] else 'FAIL'} — {gates_a}")

# Variant B: GLD/GDX
print("\n=== Variant B: GLD/GDX Spread ===")
trades_b = run_pair_backtest("GDX", "GLD")
metrics_b = compute_metrics(trades_b)
bull_b, bear_b = regime_metrics(trades_b)
perm_b = permutation_test(trades_b)
gates_b = validate_5gate(metrics_b, perm_b, bull_b, bear_b)
results["B_GLD_GDX"] = {"metrics": metrics_b, "perm_p": round(perm_b, 4),
                         "regime_bull": bull_b, "regime_bear": bear_b,
                         "gates": gates_b, "n_trades": len(trades_b)}
print(f"  Trades: {len(trades_b)}, Sharpe: {metrics_b['sharpe']}, "
      f"Return: {metrics_b['total_return_pct']}%, MaxDD: {metrics_b['max_dd_pct']}%, "
      f"WR: {metrics_b['wr']:.1%}, PF: {metrics_b['pf']}, Perm-p: {perm_b:.4f}")
print(f"  Gates: {'PASS' if gates_b['all_pass'] else 'FAIL'} — {gates_b}")

# Variant C: XLK/XLC
print("\n=== Variant C: XLK/XLC Spread ===")
trades_c = run_pair_backtest("XLC", "XLK")  # buy underperformer
metrics_c = compute_metrics(trades_c)
bull_c, bear_c = regime_metrics(trades_c)
perm_c = permutation_test(trades_c)
gates_c = validate_5gate(metrics_c, perm_c, bull_c, bear_c)
results["C_XLK_XLC"] = {"metrics": metrics_c, "perm_p": round(perm_c, 4),
                         "regime_bull": bull_c, "regime_bear": bear_c,
                         "gates": gates_c, "n_trades": len(trades_c)}
print(f"  Trades: {len(trades_c)}, Sharpe: {metrics_c['sharpe']}, "
      f"Return: {metrics_c['total_return_pct']}%, MaxDD: {metrics_c['max_dd_pct']}%, "
      f"WR: {metrics_c['wr']:.1%}, PF: {metrics_c['pf']}, Perm-p: {perm_c:.4f}")
print(f"  Gates: {'PASS' if gates_c['all_pass'] else 'FAIL'} — {gates_c}")

# Variant D: TLT/IEF
print("\n=== Variant D: TLT/IEF Spread ===")
trades_d = run_pair_backtest("TLT", "IEF", max_hold=5)  # shorter hold for bonds
metrics_d = compute_metrics(trades_d)
bull_d, bear_d = regime_metrics(trades_d)
perm_d = permutation_test(trades_d)
gates_d = validate_5gate(metrics_d, perm_d, bull_d, bear_d)
results["D_TLT_IEF"] = {"metrics": metrics_d, "perm_p": round(perm_d, 4),
                         "regime_bull": bull_d, "regime_bear": bear_d,
                         "gates": gates_d, "n_trades": len(trades_d)}
print(f"  Trades: {len(trades_d)}, Sharpe: {metrics_d['sharpe']}, "
      f"Return: {metrics_d['total_return_pct']}%, MaxDD: {metrics_d['max_dd_pct']}%, "
      f"WR: {metrics_d['wr']:.1%}, PF: {metrics_d['pf']}, Perm-p: {perm_d:.4f}")
print(f"  Gates: {'PASS' if gates_d['all_pass'] else 'FAIL'} — {gates_d}")

# Variant E: Multi-Pair Portfolio
print("\n=== Variant E: Multi-Pair Portfolio ===")
all_trades = []
for t in trades_a + trades_b + trades_c + trades_d:
    t_copy = t.copy()
    all_trades.append(t_copy)
# Sort by entry date for proper sequencing
all_trades.sort(key=lambda x: x["entry_date"])

# Equal-weight: each pair trade uses 1/4 of capital
trades_e_scaled = []
for t in all_trades:
    t_s = t.copy()
    t_s["pnl_pct"] = t["pnl_pct"] / 4  # 1/4 allocation per pair
    trades_e_scaled.append(t_s)

metrics_e = compute_metrics(trades_e_scaled)
bull_e, bear_e = regime_metrics(trades_e_scaled)
perm_e = permutation_test(trades_e_scaled)
gates_e = validate_5gate(metrics_e, perm_e, bull_e, bear_e)
results["E_MultiPair"] = {"metrics": metrics_e, "perm_p": round(perm_e, 4),
                           "regime_bull": bull_e, "regime_bear": bear_e,
                           "gates": gates_e, "n_trades": len(trades_e_scaled)}
print(f"  Trades: {len(trades_e_scaled)}, Sharpe: {metrics_e['sharpe']}, "
      f"Return: {metrics_e['total_return_pct']}%, MaxDD: {metrics_e['max_dd_pct']}%, "
      f"WR: {metrics_e['wr']:.1%}, PF: {metrics_e['pf']}, Perm-p: {perm_e:.4f}")
print(f"  Gates: {'PASS' if gates_e['all_pass'] else 'FAIL'} — {gates_e}")

# Variant F: Adaptive Threshold (SPY/QQQ with VIX > 20 gate)
print("\n=== Variant F: Adaptive SPY/QQQ (VIX>20, z>2.5) ===")
trades_f = run_pair_backtest("QQQ", "SPY", z_entry=2.5, z_exit=0.0, max_hold=20,
                              vix_gate=True, vix_threshold=20)
metrics_f = compute_metrics(trades_f)
bull_f, bear_f = regime_metrics(trades_f)
perm_f = permutation_test(trades_f)
gates_f = validate_5gate(metrics_f, perm_f, bull_f, bear_f)
results["F_Adaptive_SPY_QQQ"] = {"metrics": metrics_f, "perm_p": round(perm_f, 4),
                                   "regime_bull": bull_f, "regime_bear": bear_f,
                                   "gates": gates_f, "n_trades": len(trades_f)}
print(f"  Trades: {len(trades_f)}, Sharpe: {metrics_f['sharpe']}, "
      f"Return: {metrics_f['total_return_pct']}%, MaxDD: {metrics_f['max_dd_pct']}%, "
      f"WR: {metrics_f['wr']:.1%}, PF: {metrics_f['pf']}, Perm-p: {perm_f:.4f}")
print(f"  Gates: {'PASS' if gates_f['all_pass'] else 'FAIL'} — {gates_f}")

# ── Summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SUMMARY — ETF Pairs Mean Reversion Backtest")
print("=" * 80)
print(f"{'Variant':<25} {'Trades':>6} {'Sharpe':>7} {'Sortino':>8} {'PF':>6} {'WR':>6} "
      f"{'Return%':>8} {'MaxDD%':>7} {'Perm-p':>7} {'Pass':>5}")
print("-" * 80)

for name, r in results.items():
    m = r["metrics"]
    print(f"{name:<25} {m['n_trades']:>6} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} "
          f"{m['pf']:>6.2f} {m['wr']:>6.1%} {m['total_return_pct']:>8.2f} "
          f"{m['max_dd_pct']:>7.2f} {r['perm_p']:>7.4f} "
          f"{'YES' if r['gates']['all_pass'] else 'NO':>5}")

# ── Save Results ────────────────────────────────────────────────────────
output = {
    "strategy": "ETF Pairs Mean Reversion",
    "academic_basis": "Gatev, Goetzmann & Rouwenhorst (2006)",
    "oot_period": f"{OOT_START} to {OOT_END}",
    "starting_capital": CAPITAL,
    "slippage_pct": SLIPPAGE_PCT,
    "commission": 0,
    "regime_definition": "Bull = SPY > 200-SMA, Bear = SPY < 200-SMA",
    "zscore_window": ZSCORE_WINDOW,
    "generated": datetime.now().isoformat(),
    "variants": results,
}

out_path = Path("/home/jupiter/Lvl3Quant/data/etf_pairs_reversion_results.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nResults saved to {out_path}")

#!/usr/bin/env python3
"""
Sector Pair Mean Reversion (Arb) Backtest
==========================================
Trades the spread between related sector ETF pairs when they diverge.
Buy the laggard when z-score exceeds threshold, expecting convergence.

Pairs:
  1. XLK/XLC  (Tech vs Communications)
  2. XLF/XLV  (Financials vs Healthcare)
  3. XLY/XLP  (Consumer Discretionary vs Staples)
  4. XLE/XLU  (Energy vs Utilities)
  5. XLI/XLB  (Industrials vs Materials)

Signal: 20-day return spread, z-scored over 60-day window.
Entry: |z| > threshold. Buy laggard shares (long-only, no shorting on RH).
Exit: z returns to 0 (convergence) or 20-day time stop.

Variants (6):
  A) Basic Z-Score: All 5 pairs, z=1.5, exit z=0 or 20d
  B) Tight Threshold: z=2.0
  C) Wide Threshold: z=1.0
  D) Tech-Consumer Focus: XLK/XLC + XLY/XLP only
  E) Momentum Filter: SPY 20d momentum > 0
  F) Options Overlay: ATM calls on laggard, cost=3% + 10% haircut + $0.65 commission

5-Gate: Sharpe>0.5, perm p<0.05, regime gap<0.5, MaxDD>-50%, >=20 trades.
OOT: Jan 2022 - Jul 2026. $645 account. $0 commission, 0.02% slippage.
"""

import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Configuration ────────────────────────────────────────────────────────
CAPITAL = 645.0
SLIPPAGE_PCT = 0.0002   # 0.02% each way
COMMISSION = 0.0         # $0 on ETF shares
OPT_COMMISSION = 0.65    # per leg for options

OOT_START = "2022-01-01"
OOT_END   = "2026-07-31"
DATA_START = "2021-06-01"  # warmup for 60d lookback + 20d returns

RETURN_WINDOW = 20
ZSCORE_LOOKBACK = 60
EXIT_ZSCORE = 0.0
MAX_HOLD_DAYS = 20
N_PERM = 1000

ALL_PAIRS = [
    ("XLK", "XLC", "Tech vs Comms"),
    ("XLF", "XLV", "Fins vs Health"),
    ("XLY", "XLP", "Disc vs Staples"),
    ("XLE", "XLU", "Energy vs Utils"),
    ("XLI", "XLB", "Indust vs Matls"),
]

TECH_CONSUMER_PAIRS = [
    ("XLK", "XLC", "Tech vs Comms"),
    ("XLY", "XLP", "Disc vs Staples"),
]

TICKERS = sorted(set(
    [a for a, b, _ in ALL_PAIRS] +
    [b for a, b, _ in ALL_PAIRS] +
    ["SPY"]
))

# ── Data Download ────────────────────────────────────────────────────────
print("Downloading price data ...")
raw = yf.download(TICKERS, start=DATA_START, end=OOT_END,
                  group_by="ticker", auto_adjust=True, progress=False)


def get_close(ticker):
    try:
        s = raw[ticker]["Close"].dropna()
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        return s
    except Exception:
        return pd.Series(dtype=float)


closes = {t: get_close(t) for t in TICKERS}
spy = closes["SPY"]
spy_sma200 = spy.rolling(200).mean()
spy_mom20 = spy.pct_change(20)

loaded = sum(1 for t in TICKERS if len(closes.get(t, [])) > 100)
print(f"  Tickers loaded: {loaded}/{len(TICKERS)}, SPY rows: {len(spy)}")


# ── Compute Pair Signals ─────────────────────────────────────────────────
def compute_pair_zscore(pair_a, pair_b):
    """Return z-score of 20d return spread between A and B."""
    ca, cb = closes.get(pair_a), closes.get(pair_b)
    if ca is None or cb is None or len(ca) < 100 or len(cb) < 100:
        return pd.Series(dtype=float)

    # Align dates
    df = pd.DataFrame({"a": ca, "b": cb}).dropna()
    ret_a = df["a"].pct_change(RETURN_WINDOW)
    ret_b = df["b"].pct_change(RETURN_WINDOW)
    spread = ret_a - ret_b  # positive = A outperforming B

    mu = spread.rolling(ZSCORE_LOOKBACK).mean()
    sigma = spread.rolling(ZSCORE_LOOKBACK).std()
    z = (spread - mu) / sigma.replace(0, np.nan)
    return z.dropna()


pair_zscores = {}
for a, b, name in ALL_PAIRS:
    z = compute_pair_zscore(a, b)
    pair_zscores[(a, b)] = z
    print(f"  Pair {a}/{b}: {len(z)} z-score observations")


# ── Backtest Engine ──────────────────────────────────────────────────────
def run_backtest(pairs, z_threshold=1.5, use_momentum_filter=False,
                 use_options=False, label=""):
    """
    Run the sector pair mean reversion backtest.

    For each pair, when |z| > threshold:
      - z > threshold: A outperforming, buy B (laggard)
      - z < -threshold: B outperforming, buy A (laggard)
    Exit: z crosses 0 or MAX_HOLD_DAYS reached.
    Long-only (buy laggard shares, no shorting).
    Equal weight: $645 / number_of_active_positions.
    """
    trades = []

    for pair_a, pair_b, pair_name in pairs:
        z = pair_zscores.get((pair_a, pair_b))
        if z is None or z.empty:
            continue

        ca = closes[pair_a]
        cb = closes[pair_b]

        # Restrict to OOT period
        z_oot = z[(z.index >= OOT_START) & (z.index < OOT_END)]
        dates = z_oot.index.tolist()

        i = 0
        while i < len(dates):
            dt = dates[i]
            zval = z_oot.iloc[i]

            # Momentum filter check
            if use_momentum_filter:
                if dt in spy_mom20.index:
                    if spy_mom20.loc[:dt].iloc[-1] <= 0:
                        i += 1
                        continue
                else:
                    i += 1
                    continue

            # Entry conditions
            if abs(zval) < z_threshold:
                i += 1
                continue

            # Determine which ETF to buy (the laggard)
            if zval > z_threshold:
                # A outperforming B -> buy B (laggard)
                buy_ticker = pair_b
                direction = "buy_B"
            else:
                # B outperforming A -> buy A (laggard)
                buy_ticker = pair_a
                direction = "buy_A"

            entry_date = dt
            buy_close = closes[buy_ticker]

            if entry_date not in buy_close.index:
                i += 1
                continue

            entry_price = buy_close.loc[entry_date]
            entry_cost = entry_price * (1 + SLIPPAGE_PCT)

            # Position sizing: equal weight across pairs
            pos_size = CAPITAL / len(pairs)

            if use_options:
                # Options: ATM call, cost = 3% of ETF price + 10% haircut + commission
                opt_premium = entry_price * 0.03
                opt_cost = opt_premium * 1.10 + OPT_COMMISSION
                n_contracts = max(1, int(pos_size / (opt_cost * 100)))
                total_invested = n_contracts * opt_cost * 100
            else:
                shares = int(pos_size / entry_cost)
                if shares < 1:
                    i += 1
                    continue
                total_invested = shares * entry_cost

            # Find exit
            exit_price = None
            exit_date = None
            hold_days = 0

            for j in range(i + 1, min(i + MAX_HOLD_DAYS + 1, len(dates))):
                hold_days += 1
                check_dt = dates[j]
                check_z = z_oot.iloc[j]

                # Exit conditions: z crosses 0 or time stop
                if (zval > 0 and check_z <= EXIT_ZSCORE) or \
                   (zval < 0 and check_z >= EXIT_ZSCORE) or \
                   hold_days >= MAX_HOLD_DAYS:
                    exit_date = check_dt
                    if exit_date in buy_close.index:
                        exit_price = buy_close.loc[exit_date]
                    break

            if exit_price is None or exit_date is None:
                i += 1
                continue

            exit_proceeds = exit_price * (1 - SLIPPAGE_PCT)

            if use_options:
                # Option P&L: intrinsic value at exit - premium paid
                intrinsic = max(0, exit_price - entry_price)
                pnl = n_contracts * 100 * intrinsic - total_invested - OPT_COMMISSION
            else:
                pnl = shares * (exit_proceeds - entry_cost)

            ret = pnl / pos_size

            # Regime
            if entry_date in spy.index and entry_date in spy_sma200.index:
                spy_val = spy.loc[:entry_date].iloc[-1]
                sma_val = spy_sma200.loc[:entry_date].iloc[-1]
                regime = "bull" if spy_val > sma_val else "bear"
            else:
                regime = "unknown"

            trades.append({
                "pair": pair_name,
                "direction": direction,
                "buy_ticker": buy_ticker,
                "entry_date": str(entry_date.date()) if hasattr(entry_date, 'date') else str(entry_date),
                "exit_date": str(exit_date.date()) if hasattr(exit_date, 'date') else str(exit_date),
                "entry_price": round(float(entry_price), 2),
                "exit_price": round(float(exit_price), 2),
                "hold_days": hold_days,
                "pnl": round(float(pnl), 2),
                "return": round(float(ret), 6),
                "regime": regime,
                "z_entry": round(float(zval), 3),
            })

            # Skip past exit date to avoid overlapping trades on same pair
            i = j + 1
            continue

            i += 1  # noqa: this line won't execute due to continue above

    return trades


# ── Metrics ──────────────────────────────────────────────────────────────
def calc_metrics(trades):
    if not trades:
        return {
            "n_trades": 0, "sharpe": 0, "sortino": 0, "pf": 0,
            "wr": 0, "total_pnl": 0, "avg_pnl": 0, "max_dd_pct": 0,
            "avg_hold": 0, "regime_sharpe_bull": 0, "regime_sharpe_bear": 0,
            "regime_gap": 999,
        }

    rets = np.array([t["return"] for t in trades])
    pnls = np.array([t["pnl"] for t in trades])
    n = len(trades)

    total_pnl = float(pnls.sum())
    avg_pnl = float(pnls.mean())
    wr = float((pnls > 0).sum() / n) if n > 0 else 0

    # Sharpe (annualized, assume ~20 trades/year avg)
    trades_per_year = max(1, n / 4.5)  # ~4.5 years OOT
    mu = rets.mean()
    sigma = rets.std() if rets.std() > 0 else 1e-9
    sharpe = float((mu / sigma) * np.sqrt(trades_per_year))

    # Sortino
    downside = rets[rets < 0]
    down_std = downside.std() if len(downside) > 1 else 1e-9
    sortino = float((mu / down_std) * np.sqrt(trades_per_year))

    # Profit factor
    gross_profit = float(pnls[pnls > 0].sum()) if (pnls > 0).any() else 0
    gross_loss = float(abs(pnls[pnls < 0].sum())) if (pnls < 0).any() else 1e-9
    pf = gross_profit / gross_loss

    # Max drawdown
    equity = np.cumsum(pnls)
    peak = np.maximum.accumulate(equity + CAPITAL)
    dd = (equity + CAPITAL - peak) / peak
    max_dd = float(dd.min()) if len(dd) > 0 else 0

    # Avg hold
    avg_hold = float(np.mean([t["hold_days"] for t in trades]))

    # Regime analysis
    bull_rets = [t["return"] for t in trades if t["regime"] == "bull"]
    bear_rets = [t["return"] for t in trades if t["regime"] == "bear"]

    def regime_sharpe(r):
        if len(r) < 2:
            return 0
        r = np.array(r)
        s = r.std()
        if s < 1e-9:
            return 0
        tpy = max(1, len(r) / 4.5)
        return float((r.mean() / s) * np.sqrt(tpy))

    rs_bull = regime_sharpe(bull_rets)
    rs_bear = regime_sharpe(bear_rets)
    max_regime = max(abs(rs_bull), abs(rs_bear), 1e-9)
    regime_gap = abs(rs_bull - rs_bear) / max_regime

    return {
        "n_trades": n,
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "pf": round(pf, 3),
        "wr": round(wr, 4),
        "total_pnl": round(total_pnl, 2),
        "avg_pnl": round(avg_pnl, 2),
        "max_dd_pct": round(max_dd * 100, 2),
        "avg_hold": round(avg_hold, 1),
        "regime_sharpe_bull": round(rs_bull, 3),
        "regime_sharpe_bear": round(rs_bear, 3),
        "regime_gap": round(regime_gap, 3),
        "bull_trades": len(bull_rets),
        "bear_trades": len(bear_rets),
    }


# ── Permutation Test ─────────────────────────────────────────────────────
def permutation_test(trades, observed_sharpe):
    """Sign-shuffle permutation test: randomly flip return signs to test
    whether the observed Sharpe is distinguishable from chance."""
    if len(trades) < 5:
        return 1.0

    rets = np.array([t["return"] for t in trades])
    n = len(rets)
    tpy = max(1, n / 4.5)
    count_ge = 0

    for _ in range(N_PERM):
        # Randomly flip signs of returns (null: no directional edge)
        signs = np.random.choice([-1, 1], size=n)
        perm = rets * signs
        mu_p = perm.mean()
        sig_p = perm.std()
        if sig_p < 1e-9:
            continue
        sharpe_p = (mu_p / sig_p) * np.sqrt(tpy)
        if sharpe_p >= observed_sharpe:
            count_ge += 1

    return round(count_ge / N_PERM, 4)


# ── 5-Gate Validation ────────────────────────────────────────────────────
def validate_5gate(metrics, perm_p):
    gates = {
        "sharpe_gt_0.5": metrics["sharpe"] > 0.5,
        "perm_p_lt_0.05": perm_p < 0.05,
        "regime_gap_lt_0.5": metrics["regime_gap"] < 0.5,
        "max_dd_gt_neg50": metrics["max_dd_pct"] > -50,
        "trades_gte_20": metrics["n_trades"] >= 20,
    }
    gates["pass_all"] = all(gates.values())
    return gates


# ── Run All Variants ─────────────────────────────────────────────────────
print("\n" + "="*70)
print("SECTOR PAIR MEAN REVERSION BACKTEST")
print("="*70)

variants = {
    "A_basic_zscore": {
        "pairs": ALL_PAIRS, "z_threshold": 1.5,
        "momentum_filter": False, "options": False,
        "desc": "All 5 pairs, z=1.5, exit z=0 or 20d"
    },
    "B_tight_threshold": {
        "pairs": ALL_PAIRS, "z_threshold": 2.0,
        "momentum_filter": False, "options": False,
        "desc": "All 5 pairs, z=2.0 (higher conviction)"
    },
    "C_wide_threshold": {
        "pairs": ALL_PAIRS, "z_threshold": 1.0,
        "momentum_filter": False, "options": False,
        "desc": "All 5 pairs, z=1.0 (more trades)"
    },
    "D_tech_consumer_focus": {
        "pairs": TECH_CONSUMER_PAIRS, "z_threshold": 1.5,
        "momentum_filter": False, "options": False,
        "desc": "XLK/XLC + XLY/XLP only (highest corr)"
    },
    "E_momentum_filter": {
        "pairs": ALL_PAIRS, "z_threshold": 1.5,
        "momentum_filter": True, "options": False,
        "desc": "All 5 pairs, z=1.5, SPY 20d mom > 0"
    },
    "F_options_overlay": {
        "pairs": ALL_PAIRS, "z_threshold": 1.5,
        "momentum_filter": False, "options": True,
        "desc": "ATM calls on laggard, 3% cost + 10% haircut"
    },
}

results = {}

for vname, vcfg in variants.items():
    print(f"\n── Variant {vname} ──")
    print(f"   {vcfg['desc']}")

    trades = run_backtest(
        pairs=vcfg["pairs"],
        z_threshold=vcfg["z_threshold"],
        use_momentum_filter=vcfg["momentum_filter"],
        use_options=vcfg["options"],
        label=vname,
    )

    metrics = calc_metrics(trades)
    perm_p = permutation_test(trades, metrics["sharpe"]) if metrics["n_trades"] >= 5 else 1.0
    gates = validate_5gate(metrics, perm_p)

    print(f"   Trades: {metrics['n_trades']}")
    print(f"   Sharpe: {metrics['sharpe']:.3f}  Sortino: {metrics['sortino']:.3f}")
    print(f"   PF: {metrics['pf']:.2f}  WR: {metrics['wr']:.1%}")
    print(f"   Total P&L: ${metrics['total_pnl']:.2f}  Avg: ${metrics['avg_pnl']:.2f}")
    print(f"   Max DD: {metrics['max_dd_pct']:.1f}%  Avg Hold: {metrics['avg_hold']:.1f}d")
    print(f"   Regime Sharpe: Bull={metrics['regime_sharpe_bull']:.3f} Bear={metrics['regime_sharpe_bear']:.3f} Gap={metrics['regime_gap']:.3f}")
    print(f"   Perm p-value: {perm_p:.4f}")
    print(f"   5-Gate: {'PASS' if gates['pass_all'] else 'FAIL'} — {gates}")

    # Per-pair breakdown
    pair_breakdown = {}
    for pa, pb, pname in vcfg["pairs"]:
        ptrades = [t for t in trades if t["pair"] == pname]
        if ptrades:
            pm = calc_metrics(ptrades)
            pair_breakdown[pname] = {
                "n_trades": pm["n_trades"],
                "sharpe": pm["sharpe"],
                "pf": pm["pf"],
                "wr": pm["wr"],
                "total_pnl": pm["total_pnl"],
            }
            print(f"     {pname}: {pm['n_trades']} trades, Sharpe={pm['sharpe']:.3f}, PF={pm['pf']:.2f}, WR={pm['wr']:.1%}, P&L=${pm['total_pnl']:.2f}")

    results[vname] = {
        "description": vcfg["desc"],
        "metrics": metrics,
        "perm_p_value": perm_p,
        "five_gate": gates,
        "pair_breakdown": pair_breakdown,
        "sample_trades": trades[:10] if trades else [],
        "all_trades_count": len(trades),
    }

# ── Summary Table ────────────────────────────────────────────────────────
print("\n" + "="*70)
print("SUMMARY TABLE")
print("="*70)
print(f"{'Variant':<25} {'Trades':>6} {'Sharpe':>7} {'Sortino':>8} {'PF':>6} {'WR':>6} {'P&L':>8} {'MaxDD':>7} {'Perm-p':>7} {'5G':>5}")
print("-"*90)
for vname, r in results.items():
    m = r["metrics"]
    p = r["perm_p_value"]
    g = "PASS" if r["five_gate"]["pass_all"] else "FAIL"
    print(f"{vname:<25} {m['n_trades']:>6} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} {m['pf']:>6.2f} {m['wr']:>5.1%} {m['total_pnl']:>8.2f} {m['max_dd_pct']:>6.1f}% {p:>7.4f} {g:>5}")

# ── Best Variant ─────────────────────────────────────────────────────────
passers = {k: v for k, v in results.items() if v["five_gate"]["pass_all"]}
if passers:
    best = max(passers, key=lambda k: passers[k]["metrics"]["sharpe"])
    print(f"\nBEST PASSING VARIANT: {best}")
    print(f"  Sharpe={results[best]['metrics']['sharpe']:.3f}, Sortino={results[best]['metrics']['sortino']:.3f}")
    print(f"  {results[best]['metrics']['n_trades']} trades, WR={results[best]['metrics']['wr']:.1%}, PF={results[best]['metrics']['pf']:.2f}")
else:
    best = max(results, key=lambda k: results[k]["metrics"]["sharpe"])
    print(f"\nNO VARIANT PASSED 5-GATE. Best by Sharpe: {best}")
    print(f"  Sharpe={results[best]['metrics']['sharpe']:.3f}, {results[best]['metrics']['n_trades']} trades")
    print(f"  Failed gates: {[g for g, v in results[best]['five_gate'].items() if not v and g != 'pass_all']}")

# ── Save Results ─────────────────────────────────────────────────────────
output = {
    "strategy": "Sector Pair Mean Reversion (Arb)",
    "run_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
    "account_size": CAPITAL,
    "oot_period": f"{OOT_START} to {OOT_END}",
    "pairs": [f"{a}/{b}" for a, b, _ in ALL_PAIRS],
    "parameters": {
        "return_window": RETURN_WINDOW,
        "zscore_lookback": ZSCORE_LOOKBACK,
        "exit_zscore": EXIT_ZSCORE,
        "max_hold_days": MAX_HOLD_DAYS,
        "slippage_pct": SLIPPAGE_PCT,
    },
    "variants": results,
    "best_variant": best,
    "any_passed_5gate": bool(passers),
}

out_path = Path("/home/jupiter/Lvl3Quant/data/sector_pair_arb_results.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nResults saved to {out_path}")
print("DONE.")

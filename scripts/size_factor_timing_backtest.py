#!/usr/bin/env python3
"""
Size Factor Rotation + Value Timing Backtest
=============================================
IWM/SPY relative strength as risk appetite thermometer.
IWD/IWF (Value/Growth) ratio for mean reversion rotation.
OOT: Jan 2022 - Jul 2026. Starting capital: $645.
6 variants with permutation testing and 5-gate validation.
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")
np.random.seed(42)

# ─── Config ───────────────────────────────────────────────────────────────────
SIGNAL_TICKERS = ["IWM", "IWF", "IWD", "SPY"]
TRADE_TICKERS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB",
                 "XLU", "XLRE", "XLC", "SPY", "IWM"]
ALL_TICKERS = list(set(SIGNAL_TICKERS + TRADE_TICKERS))

START = "2020-06-01"  # enough lookback for 63d indicators
OOT_START = "2022-01-03"
OOT_END = "2026-07-28"
INITIAL_CAPITAL = 645.0
MAX_POS_VALUE = 200.0
MAX_POSITIONS = 3
SLIPPAGE_PCT = 0.0002  # 0.02%
N_PERMS = 1000
RISK_FREE = 0.04
REBAL_FREQ_DAYS = 21  # ~monthly

CYCLICALS = ["XLF", "XLI", "XLB"]
DEFENSIVES = ["XLK", "XLV", "XLU"]
VALUE_SECTORS = ["XLF", "XLE", "XLI"]
GROWTH_SECTORS = ["XLK", "XLC", "XLY"]
DEEP_CYCLICAL = ["XLF", "XLE"]
QUALITY_GROWTH = ["XLK", "XLC"]

VARIANT_NAMES = {
    "A": "Small Cap Leadership -> Cyclicals",
    "B": "Size Factor Momentum",
    "C": "Value/Growth Rotation",
    "D": "Combined Size + Value",
    "E": "Mean Reversion on Size",
    "F": "ADVERSARIAL (Random)",
}

# ─── Data download ───────────────────────────────────────────────────────────
print("Downloading price data...")
raw = yf.download(ALL_TICKERS, start=START, end=OOT_END, auto_adjust=True, progress=False)
close = raw["Close"].copy().ffill()

missing = [t for t in ALL_TICKERS if t not in close.columns or close[t].isna().all()]
if missing:
    print(f"WARNING: Missing tickers: {missing}")

# ─── Regime (SPY > SMA200 = bull) ────────────────────────────────────────────
spy_close = close["SPY"]
spy_sma200 = spy_close.rolling(200).mean()
regime = (spy_close > spy_sma200).astype(int)

# ─── Pre-compute indicators ──────────────────────────────────────────────────
iwm_spy_ratio = close["IWM"] / close["SPY"]
iwd_iwf_ratio = close["IWD"] / close["IWF"]

def build_indicators(iwm_spy_r, iwd_iwf_r):
    """Build indicator dict from ratio series."""
    return {
        "iwm_spy_10d": iwm_spy_r.pct_change(10),
        "iwm_spy_21d": iwm_spy_r.pct_change(21),
        "iwm_spy_63d_rank": iwm_spy_r.rolling(63).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.5,
            raw=False
        ),
        "iwd_iwf_21d": iwd_iwf_r.pct_change(21),
    }

indicators = build_indicators(iwm_spy_ratio, iwd_iwf_ratio)

# OOT
oot_mask = close.index >= OOT_START
dates_oot = close.index[oot_mask].tolist()
n_oot = len(dates_oot)
print(f"OOT period: {dates_oot[0].date()} to {dates_oot[-1].date()}, {n_oot} days")
print(f"Bull days: {regime.loc[oot_mask].sum()}, Bear days: {(~regime.loc[oot_mask].astype(bool)).sum()}")


def get_variant_holdings(variant, date, ind, rng=None):
    """Return list of tickers to hold for a given variant and date."""
    if variant == "A":
        chg = ind["iwm_spy_21d"].get(date, np.nan)
        if pd.isna(chg):
            return ["SPY"]
        return CYCLICALS if chg > 0 else DEFENSIVES

    elif variant == "B":
        chg = ind["iwm_spy_10d"].get(date, np.nan)
        if pd.isna(chg):
            return ["SPY"]
        return ["IWM"] if chg > 0 else ["SPY"]

    elif variant == "C":
        chg = ind["iwd_iwf_21d"].get(date, np.nan)
        if pd.isna(chg):
            return ["SPY"]
        return VALUE_SECTORS if chg > 0 else GROWTH_SECTORS

    elif variant == "D":
        size_chg = ind["iwm_spy_21d"].get(date, np.nan)
        val_chg = ind["iwd_iwf_21d"].get(date, np.nan)
        if pd.isna(size_chg) or pd.isna(val_chg):
            return ["SPY"]
        if size_chg > 0 and val_chg > 0:
            return DEEP_CYCLICAL
        elif size_chg <= 0 and val_chg <= 0:
            return QUALITY_GROWTH
        else:
            return ["SPY"]

    elif variant == "E":
        rank = ind["iwm_spy_63d_rank"].get(date, np.nan)
        if pd.isna(rank):
            return ["SPY"]
        if rank < 0.15:
            return ["IWM"]
        elif rank > 0.85:
            return ["SPY"]
        else:
            return ["SPY"]

    elif variant == "F":
        if rng is None:
            rng = np.random.default_rng(42)
        return list(rng.choice([CYCLICALS, DEFENSIVES]))

    return ["SPY"]


def run_backtest(variant, ind, seed=42):
    """Run backtest for a given variant. Monthly rebalance."""
    rng = np.random.default_rng(seed)
    capital = INITIAL_CAPITAL
    holdings = {}
    equity = []
    trades = []
    days_since_rebal = REBAL_FREQ_DAYS

    for date in dates_oot:
        days_since_rebal += 1

        if days_since_rebal >= REBAL_FREQ_DAYS:
            days_since_rebal = 0
            target_tickers = get_variant_holdings(variant, date, ind, rng)

            # Sell everything
            for tk, shares in holdings.items():
                price = close[tk].get(date, np.nan)
                if pd.notna(price) and shares > 0:
                    proceeds = shares * price * (1 - SLIPPAGE_PCT)
                    capital += proceeds
                    trades.append({"date": str(date.date()), "ticker": tk, "side": "SELL",
                                   "shares": float(shares), "price": float(price)})
            holdings = {}

            # Buy targets
            n_targets = min(len(target_tickers), MAX_POSITIONS)
            target_tickers = target_tickers[:n_targets]
            per_pos = min(capital / n_targets if n_targets > 0 else 0, MAX_POS_VALUE)

            for tk in target_tickers:
                price = close[tk].get(date, np.nan)
                if pd.notna(price) and price > 0:
                    shares = int(per_pos / price)
                    if shares > 0:
                        cost = shares * price * (1 + SLIPPAGE_PCT)
                        if cost <= capital:
                            capital -= cost
                            holdings[tk] = shares
                            trades.append({"date": str(date.date()), "ticker": tk, "side": "BUY",
                                           "shares": float(shares), "price": float(price)})

        # Mark to market
        port_value = capital
        for tk, shares in holdings.items():
            price = close[tk].get(date, np.nan)
            if pd.notna(price):
                port_value += shares * price
        equity.append(port_value)

    return pd.Series(equity, index=dates_oot), trades


def shuffle_indicators(ind, seed):
    """Create shuffled copy of indicators for permutation test."""
    rng = np.random.default_rng(seed)
    shuffled = {}
    for key, series in ind.items():
        s = series.copy()
        oot_idx = s.index[s.index >= OOT_START]
        vals = s.loc[oot_idx].values.copy()
        rng.shuffle(vals)
        s.loc[oot_idx] = vals
        shuffled[key] = s
    return shuffled


def compute_metrics(eq_series, trades):
    """Compute performance metrics from equity curve."""
    daily_ret = eq_series.pct_change().dropna()
    if len(daily_ret) < 10:
        return None

    total_ret = (eq_series.iloc[-1] / eq_series.iloc[0]) - 1
    ann_ret = (1 + total_ret) ** (252 / len(daily_ret)) - 1
    ann_vol = daily_ret.std() * np.sqrt(252)
    sharpe = (ann_ret - RISK_FREE) / ann_vol if ann_vol > 0 else 0

    downside = daily_ret[daily_ret < 0].std() * np.sqrt(252)
    sortino = (ann_ret - RISK_FREE) / downside if downside > 0 else 0

    cummax = eq_series.cummax()
    dd = (eq_series - cummax) / cummax
    max_dd = dd.min()

    buy_trades = [t for t in trades if t["side"] == "BUY"]
    n_trades = len(buy_trades)

    gains = daily_ret[daily_ret > 0].sum()
    losses = abs(daily_ret[daily_ret < 0].sum())
    pf = gains / losses if losses > 0 else float("inf")

    monthly_ret = eq_series.resample("ME").last().pct_change().dropna()
    wr = (monthly_ret > 0).mean() if len(monthly_ret) > 0 else 0

    # Regime-stratified Sharpe
    oot_regime = regime.reindex(eq_series.index).fillna(0)
    bull_ret = daily_ret[oot_regime == 1]
    bear_ret = daily_ret[oot_regime == 0]

    bull_sharpe = bear_sharpe = 0
    if len(bull_ret) > 10:
        bull_sharpe = (bull_ret.mean() * 252 - RISK_FREE) / (bull_ret.std() * np.sqrt(252)) if bull_ret.std() > 0 else 0
    if len(bear_ret) > 10:
        bear_sharpe = (bear_ret.mean() * 252 - RISK_FREE) / (bear_ret.std() * np.sqrt(252)) if bear_ret.std() > 0 else 0

    regime_gap = abs(bull_sharpe - bear_sharpe) / max(abs(bull_sharpe), abs(bear_sharpe), 0.001)

    return {
        "total_return_pct": round(total_ret * 100, 2),
        "ann_return_pct": round(ann_ret * 100, 2),
        "ann_vol_pct": round(ann_vol * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "profit_factor": round(pf, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "win_rate": round(float(wr), 3),
        "n_trades": n_trades,
        "n_rebalances": n_trades,
        "final_equity": round(float(eq_series.iloc[-1]), 2),
        "bull_sharpe": round(bull_sharpe, 3),
        "bear_sharpe": round(bear_sharpe, 3),
        "regime_gap": round(regime_gap, 3),
    }


# ─── Run all variants ────────────────────────────────────────────────────────
results = {}

for v in ["A", "B", "C", "D", "E", "F"]:
    print(f"\n{'='*60}")
    print(f"Variant {v}: {VARIANT_NAMES[v]}")
    print(f"{'='*60}")

    eq, trades = run_backtest(v, indicators)
    metrics = compute_metrics(eq, trades)

    if metrics is None:
        print("  INSUFFICIENT DATA")
        results[v] = {"name": VARIANT_NAMES[v], "status": "INSUFFICIENT_DATA"}
        continue

    # Permutation test
    print(f"  Running {N_PERMS} permutations...")
    real_sharpe = metrics["sharpe"]
    perm_sharpes = []
    for p in range(N_PERMS):
        shuf_ind = shuffle_indicators(indicators, seed=p + 1000)
        eq_p, _ = run_backtest(v, shuf_ind, seed=p + 1000)
        dr = eq_p.pct_change().dropna()
        if len(dr) > 10:
            tr = eq_p.iloc[-1] / eq_p.iloc[0] - 1
            ar = (1 + tr) ** (252 / len(dr)) - 1
            av = dr.std() * np.sqrt(252)
            ps = (ar - RISK_FREE) / av if av > 0 else 0
            perm_sharpes.append(ps)

    perm_p = np.mean([s >= real_sharpe for s in perm_sharpes]) if perm_sharpes else 1.0
    metrics["perm_p_value"] = round(float(perm_p), 4)

    # 5-gate validation
    gates = {
        "sharpe_gt_0.5": metrics["sharpe"] > 0.5,
        "perm_p_lt_0.05": perm_p < 0.05,
        "regime_gap_lt_0.5": metrics["regime_gap"] < 0.5,
        "mdd_gt_neg50": metrics["max_drawdown_pct"] > -50,
        "trades_gte_20": metrics["n_trades"] >= 20,
    }
    n_pass = sum(gates.values())
    status = "PASS" if n_pass == 5 else "FAIL"

    metrics["gates"] = {k: bool(v_) for k, v_ in gates.items()}
    metrics["gates_passed"] = f"{n_pass}/5"
    metrics["status"] = status

    results[v] = {"name": VARIANT_NAMES[v], **metrics}

    print(f"  Sharpe: {metrics['sharpe']:.3f}  Sortino: {metrics['sortino']:.3f}  "
          f"PF: {metrics['profit_factor']:.3f}  WR: {metrics['win_rate']:.1%}")
    print(f"  Return: {metrics['total_return_pct']:.1f}%  MDD: {metrics['max_drawdown_pct']:.1f}%  "
          f"Trades: {metrics['n_trades']}")
    print(f"  Bull Sharpe: {metrics['bull_sharpe']:.3f}  Bear Sharpe: {metrics['bear_sharpe']:.3f}  "
          f"Regime Gap: {metrics['regime_gap']:.3f}")
    print(f"  Perm p-value: {perm_p:.4f}")
    print(f"  Gates: {n_pass}/5 -> {status}")
    for g, gv in gates.items():
        tag = "Y" if gv else "N"
        print(f"    [{tag}] {g}")

# ─── Summary table ───────────────────────────────────────────────────────────
print(f"\n{'='*110}")
print(f"{'VARIANT':<45} {'SHARPE':>7} {'SORTINO':>8} {'PF':>6} {'WR':>6} {'RET%':>7} {'MDD%':>7} {'PERM-P':>7} {'GATES':>6} {'STATUS':>6}")
print(f"{'='*110}")
for v in ["A", "B", "C", "D", "E", "F"]:
    r = results[v]
    if r.get("status") == "INSUFFICIENT_DATA":
        print(f"{v}) {r['name']:<42} {'INSUFFICIENT DATA':>60}")
        continue
    print(f"{v}) {r['name']:<42} {r['sharpe']:>7.3f} {r['sortino']:>8.3f} {r['profit_factor']:>6.3f} "
          f"{r['win_rate']:>5.1%} {r['total_return_pct']:>6.1f}% {r['max_drawdown_pct']:>6.1f}% "
          f"{r['perm_p_value']:>7.4f} {r['gates_passed']:>6} {r['status']:>6}")
print(f"{'='*110}")

passes = [v for v in results if results[v].get("status") == "PASS"]
fails = [v for v in results if results[v].get("status") == "FAIL"]
print(f"\nPASSED: {len(passes)} variants: {passes}")
print(f"FAILED: {len(fails)} variants: {fails}")

if passes:
    best = max(passes, key=lambda v: results[v]["sharpe"])
    print(f"BEST VARIANT: {best}) {results[best]['name']} -- Sharpe {results[best]['sharpe']:.3f}")
else:
    print("NO VARIANTS PASSED ALL 5 GATES.")

# ─── Save results ────────────────────────────────────────────────────────────
best_v = max(passes, key=lambda v: results[v]["sharpe"]) if passes else max(results.keys(), key=lambda v: results[v].get("sharpe", -999))
output = {
    "strategy": "Size Factor Rotation + Value Timing",
    "run_date": datetime.now().isoformat(),
    "oot_period": f"{OOT_START} to {OOT_END}",
    "initial_capital": INITIAL_CAPITAL,
    "variants": results,
    "summary": {
        "total_variants": 6,
        "passed": len(passes),
        "failed": len(fails),
        "best_variant": f"{best_v}) {results[best_v]['name']}",
        "best_sharpe": results[best_v].get("sharpe"),
    }
}

out_path = Path("/home/jupiter/Lvl3Quant/data/size_factor_timing_results.json")
out_path.write_text(json.dumps(output, indent=2, default=str))
print(f"\nResults saved to {out_path}")

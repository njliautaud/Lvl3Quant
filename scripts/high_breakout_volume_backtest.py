#!/usr/bin/env python3
"""
52-Week High Breakout + Volume Confirmation Backtest
=====================================================
Proxy for post-split momentum / buyback announcement strategies.

Universe: 25 growth stocks
Walk-forward OOT: Jan 2022 - Jul 2026
5-gate validation: Sharpe>0.5, perm p<0.05, regime gap<0.5, MaxDD>-50%, >=20 trades

6 Variants:
  A) Basic 52W High + Volume (hold 20d, max 5 pos)
  B) Hold 40 Days
  C) Top 3 Concentration
  D) Regime Filter (SPY > 200-SMA)
  E) Pullback Entry (3-5% dip after 52W high)
  F) All-Time High Only
"""

import json
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────
UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD", "CRM", "NFLX",
    "SHOP", "DDOG", "SNOW", "UBER", "COIN", "PLTR", "ROKU", "SNAP", "PINS",
    "NET", "CRWD", "ZS", "PANW", "MDB",
]
# SQ removed (delisted/renamed to XYZ)
ACCOUNT_SIZE = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02% per trade
OOT_START = "2022-01-01"
OOT_END = "2026-07-30"
DATA_START = "2020-01-01"
N_PERMUTATIONS = 1000
SEED = 42

# ── Download Data ───────────────────────────────────────────────────────
print("Downloading data for", len(UNIVERSE), "stocks + SPY...")
tickers_to_dl = UNIVERSE + ["SPY"]

# Batch download for speed
raw = yf.download(tickers_to_dl, start=DATA_START, end=OOT_END, progress=False, auto_adjust=True, threads=True)

data = {}
for t in tickers_to_dl:
    try:
        if isinstance(raw.columns, pd.MultiIndex):
            df = raw.xs(t, level=1, axis=1).dropna(how="all")
        else:
            df = raw.copy()
        if len(df) > 100:
            data[t] = df
    except Exception as e:
        print(f"  WARN: {t} extract failed: {e}")

print(f"  Got data for {len(data)} tickers")

spy = data.get("SPY")
if spy is None:
    print("ERROR: SPY data not available")
    sys.exit(1)

# Compute SPY 200-SMA for regime filter
spy_close = spy["Close"].copy()
spy_sma200 = spy_close.rolling(200).mean()
spy_bull = (spy_close > spy_sma200).reindex(spy.index).fillna(False)


# ── Vectorized Signal Generation ────────────────────────────────────────
def compute_signals_vectorized(ticker_data: dict, variant: str) -> pd.DataFrame:
    """Vectorized signal generation - much faster than row-by-row."""
    all_signals = []

    for ticker, df in ticker_data.items():
        if ticker == "SPY" or len(df) < 260:
            continue

        close = df["Close"]
        high = df["High"]
        volume = df["Volume"]

        vol_ma20 = volume.rolling(20).mean()
        high_52w = high.rolling(252).max()
        vol_ratio = volume / vol_ma20

        # Mask to OOT period with sufficient lookback
        oot_mask = (df.index >= OOT_START) & (df.index <= OOT_END)
        valid = oot_mask & vol_ma20.notna() & high_52w.notna() & (vol_ma20 > 0)

        if variant in ("A", "B", "C", "D"):
            breakout = (high >= high_52w) & (vol_ratio > 1.5) & valid
        elif variant == "E":
            # Pullback: was at 52W high in last 5 days, now 3-5% below
            at_high = high >= high_52w
            was_at_high_recent = at_high.rolling(6).max().shift(1).fillna(0).astype(bool)
            pullback = (high_52w - close) / high_52w
            breakout = was_at_high_recent & (pullback >= 0.03) & (pullback <= 0.05) & valid
        elif variant == "F":
            ath = high.expanding().max()
            breakout = (high >= ath) & (vol_ratio > 1.5) & valid
        else:
            continue

        if variant == "D":
            # Only bull regime
            bull_aligned = spy_bull.reindex(df.index, method="pad").fillna(False)
            breakout = breakout & bull_aligned

        signal_dates = df.index[breakout]
        if len(signal_dates) == 0:
            continue

        for d in signal_dates:
            vr = vol_ratio.loc[d] if variant != "E" else 1.0
            all_signals.append({
                "date": d,
                "ticker": ticker,
                "vol_ratio": float(vr),
                "close": float(close.loc[d]),
            })

    if not all_signals:
        return pd.DataFrame(columns=["date", "ticker", "vol_ratio", "close"])
    return pd.DataFrame(all_signals)


# ── Backtest Engine (optimized) ─────────────────────────────────────────
def run_backtest(signals_df, ticker_data, hold_days=20, max_positions=5, variant="A"):
    """Run backtest given signals. Returns list of trade dicts."""
    if signals_df.empty:
        return []

    trades = []
    for date, group in signals_df.groupby("date"):
        group = group.sort_values("vol_ratio", ascending=False)

        if variant == "C":
            group = group.head(3)
        else:
            group = group.head(max_positions)

        n_pos = len(group)
        pos_size = ACCOUNT_SIZE / max(n_pos, 1)

        for _, row in group.iterrows():
            ticker = row["ticker"]
            if ticker not in ticker_data:
                continue
            df = ticker_data[ticker]
            entry_price = row["close"] * (1 + SLIPPAGE_PCT)
            shares = pos_size / entry_price

            # Find exit
            idx_arr = df.index.get_indexer([date], method="pad")
            if idx_arr[0] < 0:
                continue
            entry_idx = idx_arr[0]
            exit_idx = min(entry_idx + hold_days, len(df) - 1)
            if exit_idx <= entry_idx:
                continue

            exit_date = df.index[exit_idx]
            exit_price = float(df["Close"].iloc[exit_idx]) * (1 - SLIPPAGE_PCT)

            pnl = shares * (exit_price - entry_price)
            pnl_pct = (exit_price - entry_price) / entry_price

            # Regime
            spy_idx = spy.index.get_indexer([date], method="pad")[0]
            regime = "bull" if spy_idx >= 0 and bool(spy_bull.iloc[spy_idx]) else "bear"

            trades.append({
                "entry_date": str(date.date()),
                "exit_date": str(exit_date.date()),
                "ticker": ticker,
                "entry_price": round(entry_price, 2),
                "exit_price": round(exit_price, 2),
                "shares": round(shares, 4),
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl_pct, 4),
                "regime": regime,
            })

    return trades


def compute_metrics(trades):
    """Compute performance metrics from trade list."""
    if not trades:
        return {
            "n_trades": 0, "sharpe": 0, "sortino": 0, "pf": 0, "wr": 0,
            "max_dd": 0, "total_return": 0, "avg_return": 0,
            "sharpe_bull": 0, "sharpe_bear": 0, "regime_gap": 1.0,
            "n_bull": 0, "n_bear": 0,
        }

    returns = np.array([t["pnl_pct"] for t in trades])
    n = len(returns)
    avg_ret = float(np.mean(returns))
    std_ret = float(np.std(returns, ddof=1)) if n > 1 else 1e-9
    downside = returns[returns < 0]
    ds_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else 1e-9

    # Annualize based on trade frequency
    years = 4.5  # OOT span
    tpy = max(n / years, 1)
    sharpe = (avg_ret / std_ret) * np.sqrt(tpy) if std_ret > 1e-9 else 0
    sortino = (avg_ret / ds_std) * np.sqrt(tpy) if ds_std > 1e-9 else 0

    gross_p = float(np.sum(returns[returns > 0]))
    gross_l = float(np.abs(np.sum(returns[returns < 0])))
    pf = gross_p / gross_l if gross_l > 1e-9 else float("inf")
    wr = float(np.mean(returns > 0))

    # Max drawdown
    sorted_trades = sorted(trades, key=lambda x: x["entry_date"])
    cum_pnl = np.cumsum([t["pnl"] for t in sorted_trades])
    equity = ACCOUNT_SIZE + cum_pnl
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = float(dd.min())
    total_return = float((equity[-1] - ACCOUNT_SIZE) / ACCOUNT_SIZE)

    # Regime Sharpe
    bull_r = [t["pnl_pct"] for t in trades if t["regime"] == "bull"]
    bear_r = [t["pnl_pct"] for t in trades if t["regime"] == "bear"]

    def rsharpe(rets):
        if len(rets) < 3:
            return 0.0
        r = np.array(rets)
        s = float(np.std(r, ddof=1))
        return float((np.mean(r) / s) * np.sqrt(len(r))) if s > 1e-9 else 0.0

    sb = rsharpe(bull_r)
    sbe = rsharpe(bear_r)
    denom = max(abs(sb), abs(sbe), 1e-9)
    rgap = abs(sb - sbe) / denom

    return {
        "n_trades": n,
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "pf": round(pf, 3),
        "wr": round(wr, 3),
        "max_dd": round(max_dd, 3),
        "total_return": round(total_return, 3),
        "avg_return": round(avg_ret, 4),
        "sharpe_bull": round(sb, 3),
        "sharpe_bear": round(sbe, 3),
        "regime_gap": round(rgap, 3),
        "n_bull": len(bull_r),
        "n_bear": len(bear_r),
    }


def permutation_test(trades, n_perms=N_PERMUTATIONS, seed=SEED):
    """
    Return-shuffle permutation test.
    Shuffle the returns across trades (breaking the signal-return link).
    Compare shuffled Sharpe to actual Sharpe.
    """
    if len(trades) < 5:
        return 1.0

    rng = np.random.RandomState(seed)
    returns = np.array([t["pnl_pct"] for t in trades])
    n = len(returns)
    std_r = np.std(returns, ddof=1)
    if std_r < 1e-9:
        return 1.0

    years = 4.5
    tpy = max(n / years, 1)
    actual_sharpe = (np.mean(returns) / std_r) * np.sqrt(tpy)

    count = 0
    for _ in range(n_perms):
        shuffled = rng.permutation(returns)
        s = np.std(shuffled, ddof=1)
        if s < 1e-9:
            continue
        perm_sharpe = (np.mean(shuffled) / s) * np.sqrt(tpy)
        if perm_sharpe >= actual_sharpe:
            count += 1

    return round(count / n_perms, 4)


# ── Variant Configurations ──────────────────────────────────────────────
VARIANTS = {
    "A": {"name": "Basic 52W High + Volume", "hold_days": 20, "max_pos": 5},
    "B": {"name": "Hold 40 Days", "hold_days": 40, "max_pos": 5},
    "C": {"name": "Top 3 Concentration", "hold_days": 20, "max_pos": 3},
    "D": {"name": "Regime Filter (Bull Only)", "hold_days": 20, "max_pos": 5},
    "E": {"name": "Pullback Entry (3-5%)", "hold_days": 20, "max_pos": 5},
    "F": {"name": "All-Time High Only", "hold_days": 20, "max_pos": 5},
}

# ── Run All Variants ────────────────────────────────────────────────────
results = {}
print("\n" + "=" * 80)
print("52-WEEK HIGH BREAKOUT + VOLUME CONFIRMATION BACKTEST")
print(f"Universe: {len(UNIVERSE)} growth stocks | OOT: {OOT_START} to {OOT_END}")
print(f"Account: ${ACCOUNT_SIZE} | Slippage: {SLIPPAGE_PCT*100:.2f}%")
print("=" * 80)

for var_key, var_cfg in VARIANTS.items():
    print(f"\n--- Variant {var_key}: {var_cfg['name']} ---")

    signals = compute_signals_vectorized(data, variant=var_key)
    n_signals = len(signals)
    print(f"  Signals generated: {n_signals}")

    trades = run_backtest(
        signals, data, hold_days=var_cfg["hold_days"],
        max_positions=var_cfg["max_pos"], variant=var_key,
    )
    print(f"  Trades executed: {len(trades)}")

    metrics = compute_metrics(trades)

    print(f"  Running {N_PERMUTATIONS} permutation tests...")
    perm_p = permutation_test(trades)
    metrics["perm_p"] = perm_p

    # 5-gate validation
    gates = {
        "sharpe_gt_0.5": metrics["sharpe"] > 0.5,
        "perm_p_lt_0.05": perm_p < 0.05,
        "regime_gap_lt_0.5": metrics["regime_gap"] < 0.5,
        "max_dd_gt_neg50": metrics["max_dd"] > -0.50,
        "min_20_trades": metrics["n_trades"] >= 20,
    }
    gates_passed = sum(gates.values())
    metrics["gates"] = gates
    metrics["gates_passed"] = f"{gates_passed}/5"
    metrics["all_gates_pass"] = gates_passed == 5

    results[var_key] = {
        "name": var_cfg["name"],
        "config": var_cfg,
        "n_signals": n_signals,
        "metrics": metrics,
        "sample_trades": trades[:10] if trades else [],
        "n_total_trades": len(trades),
    }

    print(f"  Sharpe: {metrics['sharpe']:.3f} | Sortino: {metrics['sortino']:.3f}")
    print(f"  PF: {metrics['pf']:.3f} | WR: {metrics['wr']:.1%}")
    print(f"  MaxDD: {metrics['max_dd']:.1%} | Total Return: {metrics['total_return']:.1%}")
    print(f"  Sharpe_bull: {metrics['sharpe_bull']:.3f} | Sharpe_bear: {metrics['sharpe_bear']:.3f} | Gap: {metrics['regime_gap']:.3f}")
    print(f"  Perm p-value: {perm_p:.4f}")
    print(f"  Gates: {gates_passed}/5 {'PASS' if gates_passed == 5 else 'FAIL'}")
    for gate, passed in gates.items():
        status = "OK" if passed else "FAIL"
        print(f"    [{status}] {gate}")

# ── Summary Table ───────────────────────────────────────────────────────
print("\n\n" + "=" * 120)
print("SUMMARY TABLE")
print("=" * 120)
header = f"{'Var':>3} {'Name':<30} {'Trades':>6} {'Sharpe':>7} {'Sortino':>8} {'PF':>6} {'WR':>6} {'MaxDD':>7} {'TotRet':>8} {'PermP':>7} {'RGap':>6} {'Gates':>7} {'Result':>6}"
print(header)
print("-" * 120)
for var_key in sorted(results.keys()):
    r = results[var_key]
    m = r["metrics"]
    status = "PASS" if m["all_gates_pass"] else "FAIL"
    pf_str = f"{m['pf']:>6.2f}" if m["pf"] < 100 else "   inf"
    print(
        f"  {var_key:<3} {r['name']:<30} {m['n_trades']:>5} {m['sharpe']:>7.3f} "
        f"{m['sortino']:>8.3f} {pf_str} {m['wr']:>5.1%} {m['max_dd']:>7.1%} "
        f"{m['total_return']:>7.1%} {m['perm_p']:>7.4f} {m['regime_gap']:>6.3f} "
        f"{m['gates_passed']:>6} {status:>6}"
    )

# ── Best Variant ────────────────────────────────────────────────────────
passing = {k: v for k, v in results.items() if v["metrics"]["all_gates_pass"]}
if passing:
    best_key = max(passing, key=lambda k: passing[k]["metrics"]["sharpe"])
    best = passing[best_key]
    print(f"\nBEST PASSING VARIANT: {best_key} - {best['name']}")
    print(f"  Sharpe={best['metrics']['sharpe']:.3f}, PF={best['metrics']['pf']:.2f}, WR={best['metrics']['wr']:.1%}")
else:
    best_key = max(results, key=lambda k: results[k]["metrics"]["gates_passed"])
    best = results[best_key]
    print(f"\nNO VARIANT PASSES ALL 5 GATES")
    print(f"  Closest: {best_key} - {best['name']} ({best['metrics']['gates_passed']})")

# ── Year-by-Year Breakdown for top variants ─────────────────────────────
print("\n\nYEAR-BY-YEAR BREAKDOWN (Top 3 variants by Sharpe):")
print("-" * 80)
sorted_vars = sorted(results.keys(), key=lambda k: results[k]["metrics"]["sharpe"], reverse=True)
for var_key in sorted_vars[:3]:
    r = results[var_key]
    all_trades = run_backtest(
        compute_signals_vectorized(data, variant=var_key), data,
        hold_days=r["config"]["hold_days"],
        max_positions=r["config"]["max_pos"],
        variant=var_key,
    )
    print(f"\nVariant {var_key}: {r['name']}")
    if not all_trades:
        print("  No trades")
        continue

    by_year = {}
    for t in all_trades:
        yr = t["entry_date"][:4]
        if yr not in by_year:
            by_year[yr] = []
        by_year[yr].append(t)

    print(f"  {'Year':<6} {'Trades':>6} {'AvgRet':>8} {'WR':>6} {'TotalPnL':>10}")
    for yr in sorted(by_year.keys()):
        yr_trades = by_year[yr]
        rets = [t["pnl_pct"] for t in yr_trades]
        pnls = [t["pnl"] for t in yr_trades]
        print(f"  {yr:<6} {len(yr_trades):>6} {np.mean(rets):>7.2%} {np.mean(np.array(rets)>0):>5.0%} ${sum(pnls):>9.2f}")


# ── Save Results ────────────────────────────────────────────────────────
output = {
    "strategy": "52-Week High Breakout + Volume Confirmation",
    "description": "Proxy for post-split momentum / buyback announcement edge",
    "universe": UNIVERSE,
    "oot_period": f"{OOT_START} to {OOT_END}",
    "account_size": ACCOUNT_SIZE,
    "slippage_pct": SLIPPAGE_PCT,
    "n_permutations": N_PERMUTATIONS,
    "run_date": datetime.now().isoformat(),
    "variants": {},
}

for var_key, r in results.items():
    output["variants"][var_key] = {
        "name": r["name"],
        "config": r["config"],
        "n_signals": r["n_signals"],
        "n_trades": r["n_total_trades"],
        "metrics": r["metrics"],
        "sample_trades": r["sample_trades"],
    }

output_path = "/home/jupiter/Lvl3Quant/data/high_breakout_volume_results.json"
with open(output_path, "w") as f:
    json.dump(output, f, indent=2, default=str)

print(f"\nResults saved to {output_path}")
print("\nDone.")

#!/usr/bin/env python3
"""
Factor Timing Backtest — 6 Rotation Strategies
Rotates between factor ETFs based on momentum, regime, and relative strength.
OOT: Jan 2022 – Jul 2026. Starting capital: $645.
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────────
TICKERS = ["MTUM", "QUAL", "VLUE", "SIZE", "USMV", "SPY"]
START_DATE = "2021-01-01"  # extra lookback for SMAs
END_DATE = "2026-07-29"
OOT_START = "2022-01-01"
INITIAL_CAPITAL = 645.0
SLIPPAGE_BPS = 2  # 0.02% per trade
N_PERMUTATIONS = 1000
RESULTS_PATH = Path("/home/jupiter/Lvl3Quant/data/factor_timing_results.json")

# ── Data ────────────────────────────────────────────────────────────────────
print("Downloading factor ETF data...")
data = yf.download(TICKERS, start=START_DATE, end=END_DATE, progress=False)

# Handle multi-level columns from yfinance
if isinstance(data.columns, pd.MultiIndex):
    close = data["Close"]
else:
    close = data

# Forward fill any gaps
close = close.ffill().dropna()

print(f"Data range: {close.index[0].date()} to {close.index[-1].date()}, {len(close)} days")

# ── Helpers ─────────────────────────────────────────────────────────────────
FACTOR_TICKERS = ["MTUM", "QUAL", "VLUE", "SIZE", "USMV"]


def monthly_rebal_dates(prices: pd.DataFrame, start: str) -> list:
    """Get month-end dates for rebalancing."""
    mask = prices.index >= pd.Timestamp(start)
    monthly = prices[mask].resample("ME").last().index
    return [d for d in monthly if d <= prices.index[-1]]


def trailing_return(prices: pd.DataFrame, date, months=3):
    """Trailing N-month return ending at date."""
    lookback = date - pd.DateOffset(months=months)
    mask = prices.index <= date
    subset = prices[mask]
    lb_mask = subset.index >= lookback
    if lb_mask.sum() < 20:
        return pd.Series(0.0, index=prices.columns)
    p_start = subset[lb_mask].iloc[0]
    p_end = subset.iloc[-1]
    return (p_end / p_start - 1).fillna(0)


def sma(prices: pd.Series, window=200):
    """Simple moving average."""
    return prices.rolling(window).mean()


def realized_vol(prices: pd.DataFrame, window=60):
    """Annualized realized vol from daily returns."""
    rets = prices.pct_change()
    return rets.rolling(window).std() * np.sqrt(252)


def apply_slippage(turnover_frac: float) -> float:
    """Return slippage cost as fraction of capital."""
    return turnover_frac * SLIPPAGE_BPS / 10000.0


def run_strategy(close: pd.DataFrame, allocator_fn, name: str) -> dict:
    """
    Generic monthly-rebalance backtester.
    allocator_fn(close, date) -> dict of {ticker: weight} (weights sum to <=1, remainder=cash)
    Returns equity curve and stats.
    """
    rebal_dates = monthly_rebal_dates(close, OOT_START)
    if len(rebal_dates) < 2:
        return {"name": name, "error": "Not enough rebalance dates"}

    capital = INITIAL_CAPITAL
    equity_curve = []
    positions = {}  # ticker -> weight
    trades = 0
    monthly_returns = []
    prev_capital = capital

    for i, date in enumerate(rebal_dates):
        # Calculate return from previous rebal to this one
        if i > 0:
            prev_date = rebal_dates[i - 1]
            period_mask = (close.index > prev_date) & (close.index <= date)
            if period_mask.sum() > 0:
                period_prices = close[period_mask]
                if len(period_prices) > 0:
                    period_start = close[close.index <= prev_date].iloc[-1]
                    period_end = period_prices.iloc[-1]
                    period_ret = 0.0
                    for ticker, weight in positions.items():
                        if ticker in period_start.index and period_start[ticker] > 0:
                            ticker_ret = period_end[ticker] / period_start[ticker] - 1
                            period_ret += weight * ticker_ret
                    capital *= (1 + period_ret)
                    monthly_returns.append(period_ret)

        # Get new allocation
        new_positions = allocator_fn(close, date)

        # Count trades (turnover)
        turnover = 0.0
        for ticker in set(list(positions.keys()) + list(new_positions.keys())):
            old_w = positions.get(ticker, 0.0)
            new_w = new_positions.get(ticker, 0.0)
            turnover += abs(new_w - old_w)

        if turnover > 0.001:
            trades += 1
            slippage_cost = apply_slippage(turnover) * capital
            capital -= slippage_cost

        positions = new_positions
        equity_curve.append({"date": date.strftime("%Y-%m-%d"), "equity": round(capital, 2)})

    # Handle final period (after last rebal to end of data)
    if len(rebal_dates) > 0:
        last_rebal = rebal_dates[-1]
        final_mask = close.index > last_rebal
        if final_mask.sum() > 0:
            period_start = close[close.index <= last_rebal].iloc[-1]
            period_end = close[final_mask].iloc[-1]
            period_ret = 0.0
            for ticker, weight in positions.items():
                if ticker in period_start.index and period_start[ticker] > 0:
                    ticker_ret = period_end[ticker] / period_start[ticker] - 1
                    period_ret += weight * ticker_ret
            capital *= (1 + period_ret)
            monthly_returns.append(period_ret)
            equity_curve.append({"date": close[final_mask].index[-1].strftime("%Y-%m-%d"), "equity": round(capital, 2)})

    # Stats
    monthly_returns = np.array(monthly_returns)
    total_ret = (capital / INITIAL_CAPITAL - 1)
    n_years = max(len(monthly_returns) / 12, 0.5)
    ann_ret = (1 + total_ret) ** (1 / n_years) - 1

    if len(monthly_returns) > 1 and monthly_returns.std() > 0:
        sharpe = (monthly_returns.mean() / monthly_returns.std()) * np.sqrt(12)
        downside = monthly_returns[monthly_returns < 0]
        downside_std = downside.std() if len(downside) > 1 else monthly_returns.std()
        sortino = (monthly_returns.mean() / downside_std) * np.sqrt(12) if downside_std > 0 else sharpe
    else:
        sharpe = 0.0
        sortino = 0.0

    # Max drawdown from equity curve
    eq_vals = [e["equity"] for e in equity_curve]
    if eq_vals:
        peak = eq_vals[0]
        max_dd = 0.0
        for v in eq_vals:
            peak = max(peak, v)
            dd = (v - peak) / peak
            max_dd = min(max_dd, dd)
    else:
        max_dd = 0.0

    # Win rate
    wins = (monthly_returns > 0).sum()
    wr = wins / len(monthly_returns) if len(monthly_returns) > 0 else 0

    # Profit factor
    gross_profit = monthly_returns[monthly_returns > 0].sum()
    gross_loss = abs(monthly_returns[monthly_returns < 0].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    return {
        "name": name,
        "total_return_pct": round(total_ret * 100, 2),
        "ann_return_pct": round(ann_ret * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "profit_factor": round(pf, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "win_rate": round(wr, 3),
        "n_trades": trades,
        "n_months": len(monthly_returns),
        "final_equity": round(capital, 2),
        "monthly_returns": monthly_returns.tolist(),
        "equity_curve": equity_curve,
    }


# ── Strategy A: Simple Momentum Rotation ────────────────────────────────────
def alloc_simple_momentum(prices, date):
    rets = trailing_return(prices[FACTOR_TICKERS], date, months=3)
    best = rets.idxmax()
    return {best: 1.0}


# ── Strategy B: Dual Momentum ───────────────────────────────────────────────
def alloc_dual_momentum(prices, date):
    rets = trailing_return(prices[FACTOR_TICKERS], date, months=3)
    best = rets.idxmax()
    # Absolute momentum filter: best factor must be above its 200-SMA
    price_at_date = prices[best][prices.index <= date].iloc[-1]
    sma_val = sma(prices[best], 200)
    sma_at_date = sma_val[sma_val.index <= date].iloc[-1]
    if pd.isna(sma_at_date) or price_at_date > sma_at_date:
        return {best: 1.0}
    return {}  # cash


# ── Strategy C: Regime-Aware Factor ─────────────────────────────────────────
def alloc_regime_aware(prices, date):
    # Get VIX for regime detection
    spy_price = prices["SPY"][prices.index <= date].iloc[-1]
    spy_sma = sma(prices["SPY"], 200)
    spy_sma_val = spy_sma[spy_sma.index <= date].iloc[-1]

    if pd.isna(spy_sma_val):
        return {"QUAL": 1.0}  # default

    # Simple VIX proxy: use realized vol of SPY * sqrt(252) * 100
    spy_rets = prices["SPY"].pct_change()
    recent_vol = spy_rets[spy_rets.index <= date].tail(20).std() * np.sqrt(252) * 100

    if recent_vol > 25:
        return {}  # cash (high vol regime)
    elif spy_price > spy_sma_val:
        return {"MTUM": 1.0}  # bull → momentum
    else:
        return {"USMV": 1.0}  # bear → min vol


# ── Strategy D: Top-2 Equal Weight ─────────────────────────────────────────
def alloc_top2(prices, date):
    rets = trailing_return(prices[FACTOR_TICKERS], date, months=3)
    top2 = rets.nlargest(2).index.tolist()
    return {t: 0.5 for t in top2}


# ── Strategy E: Risk Parity Factor ─────────────────────────────────────────
def alloc_risk_parity(prices, date):
    vol = realized_vol(prices[FACTOR_TICKERS], 60)
    vol_at_date = vol[vol.index <= date].iloc[-1]
    if vol_at_date.isna().all() or (vol_at_date == 0).all():
        return {t: 1.0 / len(FACTOR_TICKERS) for t in FACTOR_TICKERS}
    inv_vol = 1.0 / vol_at_date.replace(0, np.nan)
    inv_vol = inv_vol.fillna(0)
    total = inv_vol.sum()
    if total == 0:
        return {t: 1.0 / len(FACTOR_TICKERS) for t in FACTOR_TICKERS}
    weights = (inv_vol / total)
    return {t: round(w, 4) for t, w in weights.items() if w > 0.01}


# ── Strategy F: Quality-Momentum Barbell ────────────────────────────────────
def alloc_qual_mtum(prices, date):
    return {"QUAL": 0.5, "MTUM": 0.5}


# ── Run All Strategies ──────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("FACTOR TIMING BACKTEST — 6 STRATEGIES")
print(f"OOT: {OOT_START} to {END_DATE} | Capital: ${INITIAL_CAPITAL}")
print("=" * 70)

strategies = [
    ("A) Simple Momentum Rotation", alloc_simple_momentum),
    ("B) Dual Momentum", alloc_dual_momentum),
    ("C) Regime-Aware Factor", alloc_regime_aware),
    ("D) Top-2 Equal Weight", alloc_top2),
    ("E) Risk Parity Factor", alloc_risk_parity),
    ("F) Quality-Momentum Barbell", alloc_qual_mtum),
]

results = []
for name, alloc_fn in strategies:
    print(f"\nRunning {name}...")
    result = run_strategy(close, alloc_fn, name)
    results.append(result)
    if "error" not in result:
        print(f"  Final: ${result['final_equity']} | Sharpe: {result['sharpe']} | "
              f"Sortino: {result['sortino']} | MDD: {result['max_drawdown_pct']}% | "
              f"WR: {result['win_rate']} | PF: {result['profit_factor']} | Trades: {result['n_trades']}")

# ── SPY Buy & Hold Benchmark ───────────────────────────────────────────────
print("\nRunning SPY Buy & Hold benchmark...")
spy_bh = run_strategy(close, lambda p, d: {"SPY": 1.0}, "Benchmark: SPY Buy & Hold")
results.append(spy_bh)
if "error" not in spy_bh:
    print(f"  Final: ${spy_bh['final_equity']} | Sharpe: {spy_bh['sharpe']} | "
          f"Sortino: {spy_bh['sortino']} | MDD: {spy_bh['max_drawdown_pct']}%")


# ── 5-Gate Validation ───────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("5-GATE VALIDATION")
print("=" * 70)


def permutation_test(monthly_returns: list, n_perms=N_PERMUTATIONS) -> float:
    """Shuffle monthly returns to test significance of Sharpe."""
    if len(monthly_returns) < 3:
        return 1.0
    arr = np.array(monthly_returns)
    observed_sharpe = (arr.mean() / arr.std()) * np.sqrt(12) if arr.std() > 0 else 0
    count = 0
    rng = np.random.default_rng(42)
    for _ in range(n_perms):
        shuffled = rng.permutation(arr)
        s = (shuffled.mean() / shuffled.std()) * np.sqrt(12) if shuffled.std() > 0 else 0
        if s >= observed_sharpe:
            count += 1
    return count / n_perms


def regime_gap(monthly_returns: list, spy_monthly: list) -> float:
    """Check if strategy Sharpe differs significantly between bull/bear months."""
    if len(monthly_returns) != len(spy_monthly):
        min_len = min(len(monthly_returns), len(spy_monthly))
        monthly_returns = monthly_returns[:min_len]
        spy_monthly = spy_monthly[:min_len]

    mr = np.array(monthly_returns)
    sm = np.array(spy_monthly)

    bull = mr[sm > 0]
    bear = mr[sm <= 0]

    if len(bull) < 3 or len(bear) < 3:
        return 0.0

    sharpe_bull = (bull.mean() / bull.std()) * np.sqrt(12) if bull.std() > 0 else 0
    sharpe_bear = (bear.mean() / bear.std()) * np.sqrt(12) if bear.std() > 0 else 0

    max_abs = max(abs(sharpe_bull), abs(sharpe_bear))
    if max_abs == 0:
        return 0.0
    return abs(sharpe_bull - sharpe_bear) / max_abs


spy_monthly_rets = spy_bh.get("monthly_returns", [])

validation_results = []
for r in results:
    if "error" in r or r["name"].startswith("Benchmark"):
        continue

    mr = r["monthly_returns"]

    # Gate 1: Sharpe > 0.5
    g1 = r["sharpe"] > 0.5

    # Gate 2: Permutation p < 0.05
    p_val = permutation_test(mr)
    g2 = p_val < 0.05

    # Gate 3: Regime gap < 0.5
    rg = regime_gap(mr, spy_monthly_rets)
    g3 = rg < 0.5

    # Gate 4: MDD > -50%
    g4 = r["max_drawdown_pct"] > -50.0

    # Gate 5: Trades >= 20
    g5 = r["n_trades"] >= 20

    gates_passed = sum([g1, g2, g3, g4, g5])
    passed = gates_passed == 5

    vr = {
        "name": r["name"],
        "gate1_sharpe": {"value": r["sharpe"], "threshold": 0.5, "pass": g1},
        "gate2_perm_p": {"value": round(p_val, 4), "threshold": 0.05, "pass": g2},
        "gate3_regime_gap": {"value": round(rg, 3), "threshold": 0.5, "pass": g3},
        "gate4_mdd": {"value": r["max_drawdown_pct"], "threshold": -50.0, "pass": g4},
        "gate5_trades": {"value": r["n_trades"], "threshold": 20, "pass": g5},
        "gates_passed": gates_passed,
        "all_passed": passed,
    }
    validation_results.append(vr)

    status = "PASS" if passed else "FAIL"
    print(f"\n{r['name']} — {status} ({gates_passed}/5)")
    print(f"  G1 Sharpe>0.5:    {r['sharpe']:>7.3f}  {'PASS' if g1 else 'FAIL'}")
    print(f"  G2 Perm p<0.05:   {p_val:>7.4f}  {'PASS' if g2 else 'FAIL'}")
    print(f"  G3 Regime gap<0.5:{rg:>7.3f}  {'PASS' if g3 else 'FAIL'}")
    print(f"  G4 MDD>-50%:      {r['max_drawdown_pct']:>7.2f}%  {'PASS' if g4 else 'FAIL'}")
    print(f"  G5 Trades>=20:    {r['n_trades']:>7d}  {'PASS' if g5 else 'FAIL'}")

# ── Correlation Matrix ──────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("MONTHLY RETURN CORRELATIONS (vs SPY B&H)")
print("=" * 70)

strat_returns = {}
for r in results:
    if "error" not in r:
        strat_returns[r["name"][:20]] = r["monthly_returns"]

# Align lengths
min_len = min(len(v) for v in strat_returns.values())
corr_df = pd.DataFrame({k: v[:min_len] for k, v in strat_returns.items()})
corr_matrix = corr_df.corr()
print(corr_matrix.round(3).to_string())

# ── Summary Table ───────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STRATEGY COMPARISON TABLE")
print("=" * 70)
print(f"{'Strategy':<35} {'Final$':>8} {'AnnRet':>7} {'Sharpe':>7} {'Sortino':>8} {'MDD':>7} {'WR':>6} {'PF':>6}")
print("-" * 95)
for r in sorted(results, key=lambda x: x.get("sharpe", 0), reverse=True):
    if "error" in r:
        continue
    print(f"{r['name']:<35} ${r['final_equity']:>7.0f} {r['ann_return_pct']:>6.1f}% {r['sharpe']:>7.3f} "
          f"{r['sortino']:>8.3f} {r['max_drawdown_pct']:>6.1f}% {r['win_rate']:>5.1%} {r['profit_factor']:>6.2f}")

# ── Save Results ────────────────────────────────────────────────────────────
output = {
    "run_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "config": {
        "oot_start": OOT_START,
        "oot_end": END_DATE,
        "initial_capital": INITIAL_CAPITAL,
        "slippage_bps": SLIPPAGE_BPS,
        "factor_etfs": FACTOR_TICKERS,
        "n_permutations": N_PERMUTATIONS,
    },
    "strategies": [],
    "validation": validation_results,
    "spy_benchmark": {
        "sharpe": spy_bh.get("sharpe", 0),
        "sortino": spy_bh.get("sortino", 0),
        "total_return_pct": spy_bh.get("total_return_pct", 0),
        "max_drawdown_pct": spy_bh.get("max_drawdown_pct", 0),
    },
}

for r in results:
    strat_data = {k: v for k, v in r.items() if k not in ["monthly_returns", "equity_curve"]}
    strat_data["equity_curve_endpoints"] = {
        "start": r.get("equity_curve", [{}])[0] if r.get("equity_curve") else {},
        "end": r.get("equity_curve", [{}])[-1] if r.get("equity_curve") else {},
    }
    output["strategies"].append(strat_data)

def make_serializable(obj):
    """Convert numpy types to native Python for JSON serialization."""
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    return obj

RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(RESULTS_PATH, "w") as f:
    json.dump(make_serializable(output), f, indent=2)

print(f"\nResults saved to {RESULTS_PATH}")
print("\nDONE.")

#!/usr/bin/env python3
"""
Commodity / Diversified ETF Trend-Following Backtest
CTA-style momentum applied to liquid ETFs tradeable in a Robinhood account.
6 variants + 60/40 benchmark + QQQ buy-and-hold.
OOT: Jan 2022 - Jul 2026, starting capital $645.
"""

import json, warnings, sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── CONFIG ──────────────────────────────────────────────────────────────
TICKERS = [
    "GLD", "SLV", "USO", "UNG", "DBA", "GDX", "XLE", "XLB",
    "TLT", "UUP", "EEM", "VNQ", "SPY", "QQQ",
]
BENCH_60_40 = {"SPY": 0.6, "AGG": 0.4}
START = "2021-01-01"        # need lookback before OOT
OOT_START = "2022-01-01"
OOT_END = "2026-07-29"
INITIAL_CAPITAL = 645.0
TOP_N = 3
PERM_SHUFFLES = 1000

# ── DATA ────────────────────────────────────────────────────────────────
all_tickers = list(set(TICKERS + ["AGG"]))
print(f"Downloading {len(all_tickers)} tickers …")
raw = yf.download(all_tickers, start=START, end=OOT_END, auto_adjust=True, progress=False)

# Handle both MultiIndex and single-ticker DataFrames
if isinstance(raw.columns, pd.MultiIndex):
    prices = raw["Close"].copy()
else:
    prices = raw[["Close"]].copy()
    prices.columns = all_tickers

prices = prices.ffill().dropna(how="all")
# Ensure index is tz-naive DatetimeIndex
prices.index = pd.to_datetime(prices.index).tz_localize(None)

# SPY 200-SMA regime
spy_200 = prices["SPY"].rolling(200).mean()
regime = (prices["SPY"] > spy_200).astype(int)   # 1=bull, 0=bear

oot_mask = prices.index >= OOT_START
oot_dates = prices.index[oot_mask]

# Monthly rebalance dates (first trading day of each month inside OOT)
monthly_dates = prices.loc[oot_mask].resample("MS").first().index
# Map to actual trading dates
rebal_dates = []
for md in monthly_dates:
    candidates = prices.index[(prices.index >= md) & (prices.index < md + pd.DateOffset(days=10))]
    if len(candidates) > 0:
        rebal_dates.append(candidates[0])
rebal_dates = pd.DatetimeIndex(rebal_dates)


# ── HELPERS ─────────────────────────────────────────────────────────────
def momentum(series, days):
    return series.pct_change(days)


def month_return_series(equity_curve):
    """Monthly returns from daily equity curve."""
    monthly = equity_curve.resample("ME").last()
    return monthly.pct_change().dropna()


def calc_metrics(equity_curve, trades_count):
    """Full metrics from daily equity curve (pd.Series with DatetimeIndex)."""
    ec = equity_curve.dropna()
    if len(ec) < 30:
        return {}
    total_ret = ec.iloc[-1] / ec.iloc[0] - 1
    years = (ec.index[-1] - ec.index[0]).days / 365.25
    cagr = (1 + total_ret) ** (1 / max(years, 0.01)) - 1

    daily_ret = ec.pct_change().dropna()
    sr = daily_ret.mean() / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 0
    down = daily_ret[daily_ret < 0]
    sortino = daily_ret.mean() / down.std() * np.sqrt(252) if len(down) > 0 and down.std() > 0 else 0

    running_max = ec.cummax()
    dd = (ec - running_max) / running_max
    max_dd = dd.min()

    mr = month_return_series(ec)
    wins = (mr > 0).sum()
    losses = (mr <= 0).sum()
    wr = wins / max(wins + losses, 1)
    pf = mr[mr > 0].sum() / abs(mr[mr < 0].sum()) if (mr < 0).any() else float("inf")

    return {
        "total_return_pct": round(total_ret * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "sharpe": round(sr, 3),
        "sortino": round(sortino, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "win_rate_monthly": round(wr, 3),
        "profit_factor": round(pf, 3) if pf != float("inf") else 999.0,
        "total_trades": trades_count,
    }


def regime_sharpe(equity_curve, regime_series):
    """Sharpe in bull vs bear regimes."""
    daily_ret = equity_curve.pct_change().dropna()
    common = daily_ret.index.intersection(regime_series.index)
    dr = daily_ret.loc[common]
    rg = regime_series.loc[common]
    out = {}
    for label, val in [("bull", 1), ("bear", 0)]:
        sub = dr[rg == val]
        if len(sub) > 10 and sub.std() > 0:
            out[f"sharpe_{label}"] = round(sub.mean() / sub.std() * np.sqrt(252), 3)
        else:
            out[f"sharpe_{label}"] = 0.0
    sb, sbe = out["sharpe_bull"], out["sharpe_bear"]
    denom = max(abs(sb), abs(sbe), 1e-9)
    out["regime_gap"] = round(abs(sb - sbe) / denom, 3)
    return out


def simulate_strategy(signal_func, label):
    """
    Run strategy month-by-month.
    signal_func(date) -> list of (ticker, weight) tuples for that rebalance.
    Returns equity curve and trade count.
    """
    cash = INITIAL_CAPITAL
    holdings = {}  # ticker -> shares (fractional ok for ETF sizing)
    equity_values = []
    trades = 0

    for i, date in enumerate(rebal_dates):
        # value portfolio at today's close
        port_val = cash
        for tk, sh in holdings.items():
            if date in prices.index and tk in prices.columns:
                port_val += sh * prices.loc[date, tk]

        # get new signals
        new_alloc = signal_func(date)
        # liquidate
        for tk, sh in holdings.items():
            if date in prices.index and tk in prices.columns:
                cash += sh * prices.loc[date, tk]
                trades += 1
        holdings = {}
        # buy new
        if new_alloc:
            for tk, wt in new_alloc:
                if date in prices.index and tk in prices.columns:
                    alloc_cash = port_val * wt
                    px = prices.loc[date, tk]
                    if px > 0:
                        shares = alloc_cash / px
                        holdings[tk] = shares
                        cash -= alloc_cash
                        trades += 1

    # Build daily equity curve for OOT period
    eq_list = []
    port_cash = INITIAL_CAPITAL
    port_holdings = {}
    rebal_idx = 0

    for date in oot_dates:
        # check rebalance
        if rebal_idx < len(rebal_dates) and date >= rebal_dates[rebal_idx]:
            # liquidate
            for tk, sh in port_holdings.items():
                if tk in prices.columns and date in prices.index:
                    port_cash += sh * prices.loc[date, tk]
            port_holdings = {}
            # new allocation
            new_alloc = signal_func(rebal_dates[rebal_idx])
            port_val = port_cash
            if new_alloc:
                for tk, wt in new_alloc:
                    if tk in prices.columns and date in prices.index:
                        px = prices.loc[date, tk]
                        if px > 0:
                            shares = port_val * wt / px
                            port_holdings[tk] = shares
                            port_cash -= port_val * wt
            rebal_idx += 1

        # daily NAV
        nav = port_cash
        for tk, sh in port_holdings.items():
            if tk in prices.columns and date in prices.index:
                nav += sh * prices.loc[date, tk]
        eq_list.append(nav)

    equity_curve = pd.Series(eq_list, index=oot_dates, name=label)
    return equity_curve, trades


# ── STRATEGY SIGNAL FUNCTIONS ───────────────────────────────────────────

def signal_A_simple_trend(date):
    """Simple Trend (50-SMA): top 3 above 50-SMA by 3-month momentum."""
    loc = prices.index.get_loc(date)
    if loc < 63:
        return []
    scores = []
    for tk in TICKERS:
        sma50 = prices[tk].iloc[loc-50:loc].mean()
        px = prices[tk].iloc[loc]
        if px > sma50:
            mom3 = prices[tk].iloc[loc] / prices[tk].iloc[loc-63] - 1 if prices[tk].iloc[loc-63] > 0 else 0
            scores.append((tk, mom3))
    scores.sort(key=lambda x: x[1], reverse=True)
    top = scores[:TOP_N]
    if not top:
        return []
    wt = 1.0 / TOP_N
    return [(tk, wt) for tk, _ in top]


def signal_B_dual_sma(date):
    """Dual SMA Crossover: 20-SMA > 50-SMA. Top 3 by signal strength (gap)."""
    loc = prices.index.get_loc(date)
    if loc < 50:
        return []
    scores = []
    for tk in TICKERS:
        sma20 = prices[tk].iloc[loc-20:loc].mean()
        sma50 = prices[tk].iloc[loc-50:loc].mean()
        if sma20 > sma50 and sma50 > 0:
            strength = (sma20 - sma50) / sma50
            scores.append((tk, strength))
    scores.sort(key=lambda x: x[1], reverse=True)
    top = scores[:TOP_N]
    if not top:
        return []
    wt = 1.0 / TOP_N
    return [(tk, wt) for tk, _ in top]


def signal_C_momentum_score(date):
    """Momentum Score Ranking: weighted 1m/3m/6m returns. Top 3."""
    loc = prices.index.get_loc(date)
    if loc < 126:
        return []
    scores = []
    for tk in TICKERS:
        r1 = prices[tk].iloc[loc] / prices[tk].iloc[loc-21] - 1 if prices[tk].iloc[loc-21] > 0 else 0
        r3 = prices[tk].iloc[loc] / prices[tk].iloc[loc-63] - 1 if prices[tk].iloc[loc-63] > 0 else 0
        r6 = prices[tk].iloc[loc] / prices[tk].iloc[loc-126] - 1 if prices[tk].iloc[loc-126] > 0 else 0
        score = r1 * 0.5 + r3 * 0.3 + r6 * 0.2
        scores.append((tk, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    top = scores[:TOP_N]
    wt = 1.0 / TOP_N
    return [(tk, wt) for tk, _ in top]


def signal_D_ts_momentum(date):
    """Time-Series Momentum: long if 12m return > 0, top 3 by magnitude."""
    loc = prices.index.get_loc(date)
    if loc < 252:
        return []
    scores = []
    for tk in TICKERS:
        r12 = prices[tk].iloc[loc] / prices[tk].iloc[loc-252] - 1 if prices[tk].iloc[loc-252] > 0 else 0
        if r12 > 0:
            scores.append((tk, r12))
    scores.sort(key=lambda x: x[1], reverse=True)
    top = scores[:TOP_N]
    if not top:
        return []
    wt = 1.0 / TOP_N
    return [(tk, wt) for tk, _ in top]


def signal_E_breakout(date):
    """Breakout: new 20-day high. Top 3 by strength (close/20d-high ratio * inverse ATR)."""
    loc = prices.index.get_loc(date)
    if loc < 20:
        return []
    scores = []
    for tk in TICKERS:
        window = prices[tk].iloc[loc-20:loc+1]
        high20 = window.max()
        px = prices[tk].iloc[loc]
        if px >= high20 * 0.995 and high20 > 0:  # within 0.5% of 20d high
            # ATR proxy from daily returns
            rets = prices[tk].iloc[loc-20:loc].pct_change().dropna().abs()
            atr = rets.mean() if len(rets) > 0 else 0.01
            strength = (px / high20) / max(atr, 0.001)
            scores.append((tk, strength))
    scores.sort(key=lambda x: x[1], reverse=True)
    top = scores[:TOP_N]
    if not top:
        return []
    wt = 1.0 / TOP_N
    return [(tk, wt) for tk, _ in top]


def signal_F_risk_parity_trend(date):
    """Risk Parity Trend: momentum-score ranking (like C) but inverse-vol weighting."""
    loc = prices.index.get_loc(date)
    if loc < 126:
        return []
    scores = []
    for tk in TICKERS:
        r1 = prices[tk].iloc[loc] / prices[tk].iloc[loc-21] - 1 if prices[tk].iloc[loc-21] > 0 else 0
        r3 = prices[tk].iloc[loc] / prices[tk].iloc[loc-63] - 1 if prices[tk].iloc[loc-63] > 0 else 0
        r6 = prices[tk].iloc[loc] / prices[tk].iloc[loc-126] - 1 if prices[tk].iloc[loc-126] > 0 else 0
        score = r1 * 0.5 + r3 * 0.3 + r6 * 0.2
        # 20-day realized vol
        rets20 = prices[tk].iloc[loc-20:loc].pct_change().dropna()
        vol = rets20.std() * np.sqrt(252) if len(rets20) > 5 else 0.30
        scores.append((tk, score, vol))
    scores.sort(key=lambda x: x[1], reverse=True)
    top = scores[:TOP_N]
    if not top:
        return []
    # inverse vol weights
    inv_vols = [1.0 / max(v, 0.05) for _, _, v in top]
    total_iv = sum(inv_vols)
    weights = [iv / total_iv for iv in inv_vols]
    return [(tk, w) for (tk, _, _), w in zip(top, weights)]


# ── RUN ALL VARIANTS ────────────────────────────────────────────────────
strategies = {
    "A_simple_trend_50sma": signal_A_simple_trend,
    "B_dual_sma_crossover": signal_B_dual_sma,
    "C_momentum_score": signal_C_momentum_score,
    "D_timeseries_momentum": signal_D_ts_momentum,
    "E_breakout": signal_E_breakout,
    "F_risk_parity_trend": signal_F_risk_parity_trend,
}

results = {}
equity_curves = {}

for name, func in strategies.items():
    print(f"Running {name} …")
    ec, trades = simulate_strategy(func, name)
    equity_curves[name] = ec
    metrics = calc_metrics(ec, trades)
    rg = regime_sharpe(ec, regime)
    metrics.update(rg)
    results[name] = metrics


# ── BENCHMARKS ──────────────────────────────────────────────────────────
# 60/40 SPY/AGG
print("Running 60/40 benchmark …")
bench_nav = []
for date in oot_dates:
    val = 0
    for tk, wt in BENCH_60_40.items():
        if tk in prices.columns and date in prices.index:
            start_px = prices.loc[prices.index[prices.index >= OOT_START][0], tk]
            val += INITIAL_CAPITAL * wt * (prices.loc[date, tk] / start_px)
    bench_nav.append(val)
bench_ec = pd.Series(bench_nav, index=oot_dates, name="60_40_SPY_AGG")
equity_curves["60_40_SPY_AGG"] = bench_ec
results["benchmark_60_40"] = calc_metrics(bench_ec, 0)
results["benchmark_60_40"].update(regime_sharpe(bench_ec, regime))

# QQQ buy-and-hold
print("Running QQQ buy-and-hold …")
qqq_start = prices.loc[prices.index[prices.index >= OOT_START][0], "QQQ"]
qqq_nav = INITIAL_CAPITAL * prices.loc[oot_dates, "QQQ"] / qqq_start
equity_curves["QQQ_buyhold"] = qqq_nav
results["benchmark_QQQ_buyhold"] = calc_metrics(qqq_nav, 0)
results["benchmark_QQQ_buyhold"].update(regime_sharpe(qqq_nav, regime))


# ── PERMUTATION TEST ────────────────────────────────────────────────────
print(f"Running permutation test ({PERM_SHUFFLES} shuffles) …")


def random_signal(date):
    """Random: pick TOP_N random ETFs, equal weight."""
    chosen = list(np.random.choice(TICKERS, size=TOP_N, replace=False))
    wt = 1.0 / TOP_N
    return [(tk, wt) for tk in chosen]


perm_sharpes = []
for i in range(PERM_SHUFFLES):
    ec_perm, _ = simulate_strategy(random_signal, f"perm_{i}")
    dr = ec_perm.pct_change().dropna()
    s = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
    perm_sharpes.append(s)
    if (i + 1) % 200 == 0:
        print(f"  shuffle {i+1}/{PERM_SHUFFLES}")

perm_sharpes = np.array(perm_sharpes)

for name in strategies:
    actual_sharpe = results[name]["sharpe"]
    perm_p = (perm_sharpes >= actual_sharpe).mean()
    results[name]["perm_p"] = round(float(perm_p), 4)


# ── 5-GATE VALIDATION ──────────────────────────────────────────────────
for name in strategies:
    m = results[name]
    gates = {
        "sharpe_gt_0.5": m.get("sharpe", 0) > 0.5,
        "perm_p_lt_0.05": m.get("perm_p", 1) < 0.05,
        "regime_gap_lt_0.5": m.get("regime_gap", 1) < 0.5,
        "mdd_gt_neg50": m.get("max_drawdown_pct", -100) > -50,
        "trades_gte_20": m.get("total_trades", 0) >= 20,
    }
    m["gates"] = gates
    m["gates_passed"] = sum(gates.values())
    m["all_gates_pass"] = all(gates.values())


# ── OUTPUT ──────────────────────────────────────────────────────────────
output = {
    "metadata": {
        "oot_start": OOT_START,
        "oot_end": OOT_END,
        "initial_capital": INITIAL_CAPITAL,
        "etf_universe": TICKERS,
        "top_n": TOP_N,
        "perm_shuffles": PERM_SHUFFLES,
        "regime": "SPY_200SMA",
        "run_timestamp": datetime.now().isoformat(),
    },
    "strategies": {},
    "benchmarks": {
        "60_40_SPY_AGG": results["benchmark_60_40"],
        "QQQ_buyhold": results["benchmark_QQQ_buyhold"],
    },
    "permutation_distribution": {
        "mean_sharpe": round(float(perm_sharpes.mean()), 3),
        "median_sharpe": round(float(np.median(perm_sharpes)), 3),
        "std_sharpe": round(float(perm_sharpes.std()), 3),
        "p5_sharpe": round(float(np.percentile(perm_sharpes, 5)), 3),
        "p95_sharpe": round(float(np.percentile(perm_sharpes, 95)), 3),
    },
}

for name in strategies:
    output["strategies"][name] = results[name]

out_path = Path("/home/jupiter/Lvl3Quant/data/commodity_trend_following_results.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2, default=str)

print(f"\nResults saved to {out_path}")
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"{'Strategy':<30} {'Return%':>8} {'CAGR%':>7} {'Sharpe':>7} {'Sortino':>8} {'MDD%':>7} {'WR':>5} {'PF':>6} {'Gates':>5} {'Perm-p':>7}")
print("-"*100)
for name in strategies:
    m = results[name]
    print(f"{name:<30} {m.get('total_return_pct',0):>8.1f} {m.get('cagr_pct',0):>7.1f} {m.get('sharpe',0):>7.3f} {m.get('sortino',0):>8.3f} {m.get('max_drawdown_pct',0):>7.1f} {m.get('win_rate_monthly',0):>5.3f} {m.get('profit_factor',0):>6.2f} {m.get('gates_passed',0):>3}/5 {m.get('perm_p',1):>7.4f}")
print("-"*100)
for bname, bkey in [("60/40 SPY/AGG", "benchmark_60_40"), ("QQQ Buy-Hold", "benchmark_QQQ_buyhold")]:
    m = results[bkey]
    print(f"{bname:<30} {m.get('total_return_pct',0):>8.1f} {m.get('cagr_pct',0):>7.1f} {m.get('sharpe',0):>7.3f} {m.get('sortino',0):>8.3f} {m.get('max_drawdown_pct',0):>7.1f} {m.get('win_rate_monthly',0):>5.3f} {m.get('profit_factor',0):>6.02f}")
print("="*80)

# Regime detail
print("\nREGIME STRATIFICATION:")
print(f"{'Strategy':<30} {'Bull Sharpe':>12} {'Bear Sharpe':>12} {'Gap':>6}")
print("-"*62)
for name in strategies:
    m = results[name]
    print(f"{name:<30} {m.get('sharpe_bull',0):>12.3f} {m.get('sharpe_bear',0):>12.3f} {m.get('regime_gap',0):>6.3f}")

# Gate summary
print("\n5-GATE VALIDATION:")
for name in strategies:
    m = results[name]
    status = "PASS" if m["all_gates_pass"] else "FAIL"
    print(f"  {name}: {status} ({m['gates_passed']}/5) — {m['gates']}")

print("\nDone.")

#!/usr/bin/env python3
"""
Multi-Strategy Portfolio Optimizer
===================================
Combines 3 validated strategies with optimal allocation weights.

Strategies:
  A  — Signal Aggregator (composite of VIX<20, SPY>50SMA, QQQ 20d mom>0)
  v2F — Strategy Rotation v2F (VIX-adjusted contrarian dip-buy)
  LevF — Adaptive Leverage F (vol-timing: TQQQ / QQQ / cash)

Portfolio combinations tested:
  A) Equal weight          B) Risk parity         C) Markowitz MVO
  D) Half-Kelly            E) Max Sharpe          F) Adaptive

HC #760 — larger accounts.  All signals lagged 1 day.  0.02% slippage per rebalance.
"""

import json, warnings, sys, os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── constants ─────────────────────────────────────────────────────────────────
SLIPPAGE_BPS = 2  # 0.02% per rebalance
START = "2022-01-01"
END = "2026-07-29"
RF = 0.045  # risk-free rate (annualized)
CAPITAL_LEVELS = [10_000, 25_000, 50_000, 100_000]
N_PERM = 1000
REBALANCE_FREQ = "MS"  # month-start for monthly rebalance

# ── data download ─────────────────────────────────────────────────────────────
print("Downloading market data …")
tickers = ["QQQ", "TQQQ", "SPY", "^VIX"]
raw = yf.download(tickers, start=START, end=END, auto_adjust=True, progress=False)

# Handle multi-level columns from yfinance
if isinstance(raw.columns, pd.MultiIndex):
    close = raw["Close"].copy()
    volume = raw["Volume"].copy()
else:
    close = raw[["Close"]].copy()
    volume = raw[["Volume"]].copy()

# Rename ^VIX -> VIX
close.rename(columns={"^VIX": "VIX"}, inplace=True)
if "^VIX" in volume.columns:
    volume.rename(columns={"^VIX": "VIX"}, inplace=True)

close = close.dropna(subset=["QQQ"])
print(f"  Data: {close.index[0].date()} -> {close.index[-1].date()}, {len(close)} trading days")

qqq_ret = close["QQQ"].pct_change()
tqqq_ret = close["TQQQ"].pct_change()
spy_close = close["SPY"]
qqq_close = close["QQQ"]
vix = close["VIX"]

# ── Strategy A: Signal Aggregator ─────────────────────────────────────────────
def strategy_a_signals(close_df):
    """Composite score >=3 -> long QQQ, else cash. Signals lagged 1 day."""
    sig = pd.DataFrame(index=close_df.index)
    sig["vix_low"] = (close_df["VIX"] < 20).astype(int)
    sig["spy_above_50sma"] = (close_df["SPY"] > close_df["SPY"].rolling(50).mean()).astype(int)
    sig["qqq_mom_pos"] = (close_df["QQQ"].pct_change(20) > 0).astype(int)
    # Volume signal: above 20d avg
    if "QQQ" in volume.columns:
        vol_ma = volume["QQQ"].reindex(close_df.index).rolling(20).mean()
        sig["volume_up"] = (volume["QQQ"].reindex(close_df.index) > vol_ma).astype(int)
    else:
        sig["volume_up"] = 0
    # Breadth proxy: SPY 10d momentum positive
    sig["breadth"] = (close_df["SPY"].pct_change(10) > 0).astype(int)

    composite = sig.sum(axis=1)
    position = (composite >= 3).astype(int).shift(1)  # lag 1 day
    return position.fillna(0)


# ── Strategy v2F: VIX-adjusted contrarian dip-buy ────────────────────────────
def strategy_v2f_signals(close_df):
    """Buy QQQ when 5d return < -(0.02 + (VIX-15)*0.002). Hold until profit or 10 days."""
    qqq = close_df["QQQ"]
    vx = close_df["VIX"]
    ret5 = qqq.pct_change(5)
    threshold = -(0.02 + (vx - 15) * 0.002)

    position = pd.Series(0.0, index=close_df.index)
    in_trade = False
    entry_price = 0
    hold_days = 0

    for i in range(1, len(close_df)):
        if in_trade:
            hold_days += 1
            cur_price = qqq.iloc[i]
            if cur_price > entry_price or hold_days >= 10:
                in_trade = False
                position.iloc[i] = 0
            else:
                position.iloc[i] = 1
        else:
            # Entry signal from PREVIOUS day (lagged)
            if i >= 6 and ret5.iloc[i - 1] < threshold.iloc[i - 1]:
                in_trade = True
                entry_price = qqq.iloc[i]
                hold_days = 0
                position.iloc[i] = 1
            else:
                position.iloc[i] = 0

    return position


# ── Strategy LevF: Adaptive Leverage (vol-timing) ────────────────────────────
def strategy_levf_signals(close_df):
    """
    20d realized vol < 30th percentile -> TQQQ (3x exposure proxy)
    30th-70th -> QQQ (1x)
    > 70th -> cash (0x)
    Returns position multiplier and which instrument.
    """
    qqq = close_df["QQQ"]
    log_ret = np.log(qqq / qqq.shift(1))
    rv20 = log_ret.rolling(20).std() * np.sqrt(252)

    # Rolling percentiles (expanding to avoid look-ahead on percentile calc)
    pct30 = rv20.expanding(min_periods=60).quantile(0.30)
    pct70 = rv20.expanding(min_periods=60).quantile(0.70)

    # Determine regime — lagged 1 day
    leverage = pd.Series(0.0, index=close_df.index)
    use_tqqq = pd.Series(False, index=close_df.index)

    for i in range(1, len(close_df)):
        prev = i - 1
        if pd.isna(rv20.iloc[prev]) or pd.isna(pct30.iloc[prev]):
            leverage.iloc[i] = 0
        elif rv20.iloc[prev] < pct30.iloc[prev]:
            leverage.iloc[i] = 1.0  # will use TQQQ returns
            use_tqqq.iloc[i] = True
        elif rv20.iloc[prev] < pct70.iloc[prev]:
            leverage.iloc[i] = 1.0  # QQQ
        else:
            leverage.iloc[i] = 0.0  # cash

    return leverage, use_tqqq


# ── Generate daily return series ──────────────────────────────────────────────
print("Generating strategy return series ...")

pos_a = strategy_a_signals(close)
pos_v2f = strategy_v2f_signals(close)
lev_f, use_tqqq_f = strategy_levf_signals(close)

# Strategy A returns: position * QQQ daily return
ret_a = pos_a * qqq_ret

# Strategy v2F returns: position * QQQ daily return
ret_v2f = pos_v2f * qqq_ret

# Strategy LevF returns: uses TQQQ when low vol, QQQ when medium, 0 when high
ret_levf = pd.Series(0.0, index=close.index)
for i in range(len(close)):
    if lev_f.iloc[i] > 0:
        if use_tqqq_f.iloc[i]:
            ret_levf.iloc[i] = tqqq_ret.iloc[i] if not pd.isna(tqqq_ret.iloc[i]) else 0
        else:
            ret_levf.iloc[i] = qqq_ret.iloc[i] if not pd.isna(qqq_ret.iloc[i]) else 0

# Clean NaN
ret_a = ret_a.fillna(0)
ret_v2f = ret_v2f.fillna(0)
ret_levf = ret_levf.fillna(0)

# Stack into DataFrame
strat_rets = pd.DataFrame({
    "A": ret_a,
    "v2F": ret_v2f,
    "LevF": ret_levf,
}, index=close.index).dropna()

qqq_bh = qqq_ret.reindex(strat_rets.index).fillna(0)

print(f"  Strategy return series: {len(strat_rets)} days")

# ── Helper functions ──────────────────────────────────────────────────────────
def annualized_sharpe(rets, rf=RF):
    excess = rets - rf / 252
    if excess.std() == 0:
        return 0
    return np.sqrt(252) * excess.mean() / excess.std()

def annualized_sortino(rets, rf=RF):
    excess = rets - rf / 252
    downside = excess[excess < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0
    return np.sqrt(252) * excess.mean() / downside.std()

def cagr(rets):
    cum = (1 + rets).prod()
    n_years = len(rets) / 252
    if n_years <= 0 or cum <= 0:
        return 0
    return cum ** (1 / n_years) - 1

def max_drawdown(rets):
    cum = (1 + rets).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return dd.min()

def win_rate(rets):
    nonzero = rets[rets != 0]
    if len(nonzero) == 0:
        return 0
    return (nonzero > 0).mean()

def count_trades(positions):
    """Count number of position changes (entries)."""
    changes = positions.diff().abs()
    return int((changes > 0).sum())

def regime_sharpes(port_rets, spy_rets):
    """Calculate Sharpe in bull vs bear regimes (SPY 60d return >0 or <0)."""
    spy_60 = spy_rets.rolling(60).sum()
    bull = spy_60 > 0
    bear = spy_60 <= 0
    bull = bull.reindex(port_rets.index).fillna(False)
    bear = bear.reindex(port_rets.index).fillna(False)
    sh_bull = annualized_sharpe(port_rets[bull]) if bull.sum() > 30 else np.nan
    sh_bear = annualized_sharpe(port_rets[bear]) if bear.sum() > 30 else np.nan
    return sh_bull, sh_bear

def regime_gap(sh_bull, sh_bear):
    if pd.isna(sh_bull) or pd.isna(sh_bear):
        return np.nan
    denom = max(abs(sh_bull), abs(sh_bear))
    if denom == 0:
        return 0
    return abs(sh_bull - sh_bear) / denom

def full_metrics(rets, label, spy_rets, positions=None):
    sh = annualized_sharpe(rets)
    so = annualized_sortino(rets)
    c = cagr(rets)
    mdd = max_drawdown(rets)
    wr = win_rate(rets)
    sh_bull, sh_bear = regime_sharpes(rets, spy_rets)
    rg = regime_gap(sh_bull, sh_bear)
    n_trades = count_trades(positions) if positions is not None else None

    # 5-gate validation (cast to Python bool to avoid numpy.bool_ issues)
    gates = {
        "sharpe_gt_0.5": bool(sh > 0.5),
        "perm_p_lt_0.05": None,  # filled later
        "regime_gap_lt_0.5": bool(rg < 0.5) if not pd.isna(rg) else False,
        "maxdd_gt_neg50": bool(mdd > -0.50),
        "trades_gte_20": bool(n_trades >= 20) if n_trades is not None else True,
    }

    return {
        "label": label,
        "sharpe": round(float(sh), 3),
        "sortino": round(float(so), 3),
        "cagr": round(float(c * 100), 2),
        "max_dd": round(float(mdd * 100), 2),
        "win_rate": round(float(wr * 100), 1),
        "sharpe_bull": round(float(sh_bull), 3) if not pd.isna(sh_bull) else None,
        "sharpe_bear": round(float(sh_bear), 3) if not pd.isna(sh_bear) else None,
        "regime_gap": round(float(rg), 3) if not pd.isna(rg) else None,
        "n_trades": n_trades,
        "gates": gates,
    }

def apply_slippage(weights_series_dict, strat_rets_df, slippage_bps=SLIPPAGE_BPS):
    """
    Given a dict of {strat_name: pd.Series of weights} and strategy returns,
    compute portfolio returns with slippage on rebalance days.
    """
    port_ret = pd.Series(0.0, index=strat_rets_df.index)
    total_turnover = pd.Series(0.0, index=strat_rets_df.index)

    for name in strat_rets_df.columns:
        w = weights_series_dict[name]
        port_ret += w * strat_rets_df[name]
        turnover = w.diff().abs()
        total_turnover += turnover

    slippage_cost = total_turnover * (slippage_bps / 10000)
    port_ret -= slippage_cost
    return port_ret


# ── Portfolio construction methods ────────────────────────────────────────────
spy_ret = spy_close.pct_change().reindex(strat_rets.index).fillna(0)

print("Building portfolio combinations ...")

strategies = ["A", "v2F", "LevF"]


def make_constant_weights(w_dict):
    return {s: pd.Series(w_dict[s], index=strat_rets.index) for s in strategies}


# A) Equal weight
def portfolio_equal():
    w = {s: 1/3 for s in strategies}
    return make_constant_weights(w), "Equal Weight (33/33/33)"


# B) Risk parity
def portfolio_risk_parity():
    weights = {}
    for s in strategies:
        vol = strat_rets[s].rolling(60, min_periods=20).std() * np.sqrt(252)
        inv_vol = 1 / vol.clip(lower=0.01)
        weights[s] = inv_vol
    # Normalize
    total = sum(weights[s] for s in strategies)
    for s in strategies:
        weights[s] = (weights[s] / total).fillna(1/3)
    return weights, "Risk Parity"


# C) Markowitz MVO (rolling 252d, monthly rebalance)
def portfolio_markowitz():
    weights = {s: pd.Series(1/3, index=strat_rets.index) for s in strategies}
    rebal_dates = strat_rets.resample(REBALANCE_FREQ).first().index

    for rd in rebal_dates:
        lookback = strat_rets.loc[:rd].tail(252)
        if len(lookback) < 60:
            continue
        mu = lookback.mean() * 252
        cov = lookback.cov() * 252

        def neg_sharpe(w):
            port_ret_val = w @ mu.values
            port_vol = np.sqrt(w @ cov.values @ w)
            if port_vol == 0:
                return 0
            return -(port_ret_val - RF) / port_vol

        cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
        bounds = [(0, 1)] * 3
        x0 = np.array([1/3, 1/3, 1/3])
        res = minimize(neg_sharpe, x0, bounds=bounds, constraints=cons, method="SLSQP")

        if res.success:
            next_rebal = rebal_dates[rebal_dates > rd]
            if len(next_rebal) > 0:
                mask = (strat_rets.index >= rd) & (strat_rets.index < next_rebal[0])
            else:
                mask = strat_rets.index >= rd
            for i_s, s in enumerate(strategies):
                weights[s].loc[mask] = res.x[i_s]

    return weights, "Markowitz MVO"


# D) Half-Kelly
def portfolio_kelly():
    weights = {s: pd.Series(1/3, index=strat_rets.index) for s in strategies}
    rebal_dates = strat_rets.resample(REBALANCE_FREQ).first().index

    for rd in rebal_dates:
        lookback = strat_rets.loc[:rd].tail(252)
        if len(lookback) < 60:
            continue
        mu = lookback.mean() * 252
        cov = lookback.cov() * 252

        try:
            cov_inv = np.linalg.inv(cov.values)
            kelly_full = cov_inv @ (mu.values - RF)
            kelly_half = kelly_full * 0.5
            kelly_half = np.clip(kelly_half, 0, 2)
            total = kelly_half.sum()
            if total > 0:
                kelly_half /= total
            else:
                kelly_half = np.array([1/3, 1/3, 1/3])
        except np.linalg.LinAlgError:
            kelly_half = np.array([1/3, 1/3, 1/3])

        next_rebal = rebal_dates[rebal_dates > rd]
        if len(next_rebal) > 0:
            mask = (strat_rets.index >= rd) & (strat_rets.index < next_rebal[0])
        else:
            mask = strat_rets.index >= rd
        for i_s, s in enumerate(strategies):
            weights[s].loc[mask] = kelly_half[i_s]

    return weights, "Half-Kelly"


# E) Max Sharpe (regularized to prefer diversification)
def portfolio_max_sharpe():
    weights = {s: pd.Series(1/3, index=strat_rets.index) for s in strategies}
    rebal_dates = strat_rets.resample(REBALANCE_FREQ).first().index

    for rd in rebal_dates:
        lookback = strat_rets.loc[:rd].tail(252)
        if len(lookback) < 60:
            continue
        mu = lookback.mean() * 252
        cov = lookback.cov() * 252

        def neg_sharpe_reg(w):
            port_ret_val = w @ mu.values
            port_vol = np.sqrt(w @ cov.values @ w)
            if port_vol == 0:
                return 0
            sharpe = (port_ret_val - RF) / port_vol
            reg = 0.1 * np.sum((w - 1/3)**2)
            return -sharpe + reg

        cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
        bounds = [(0.05, 0.70)] * 3
        x0 = np.array([1/3, 1/3, 1/3])
        res = minimize(neg_sharpe_reg, x0, bounds=bounds, constraints=cons, method="SLSQP")

        if res.success:
            next_rebal = rebal_dates[rebal_dates > rd]
            if len(next_rebal) > 0:
                mask = (strat_rets.index >= rd) & (strat_rets.index < next_rebal[0])
            else:
                mask = strat_rets.index >= rd
            for i_s, s in enumerate(strategies):
                weights[s].loc[mask] = res.x[i_s]

    return weights, "Max Sharpe (regularized)"


# F) Adaptive (tilt toward best trailing 60d Sharpe)
def portfolio_adaptive():
    weights = {s: pd.Series(1/3, index=strat_rets.index) for s in strategies}

    for i in range(60, len(strat_rets)):
        trailing = strat_rets.iloc[i-60:i]
        sharpes = {}
        for s in strategies:
            sharpes[s] = annualized_sharpe(trailing[s])

        base = 1 / 3
        best = max(sharpes, key=sharpes.get)
        worst = min(sharpes, key=sharpes.get)

        w = {s: base for s in strategies}
        w[best] = min(base + 0.15, 0.60)
        w[worst] = max(base - 0.15, 0.10)
        mid = [s for s in strategies if s != best and s != worst][0]
        w[mid] = 1.0 - w[best] - w[worst]

        idx = strat_rets.index[i]
        for s in strategies:
            weights[s].iloc[i] = w[s]

    return weights, "Adaptive (60d Sharpe tilt)"


# ── Run all portfolios ───────────────────────────────────────────────────────
portfolio_builders = [
    portfolio_equal,
    portfolio_risk_parity,
    portfolio_markowitz,
    portfolio_kelly,
    portfolio_max_sharpe,
    portfolio_adaptive,
]

results = {}
portfolio_returns = {}

# Individual strategy metrics first
print("\n== Individual Strategy Metrics ==")
for s in strategies:
    pos = pos_a if s == "A" else (pos_v2f if s == "v2F" else lev_f)
    m = full_metrics(strat_rets[s], f"Strategy {s}", spy_ret, pos)
    results[f"individual_{s}"] = m
    portfolio_returns[f"individual_{s}"] = strat_rets[s]
    print(f"  {s}: Sharpe={m['sharpe']:.3f}  Sortino={m['sortino']:.3f}  "
          f"CAGR={m['cagr']:.1f}%  MaxDD={m['max_dd']:.1f}%  WR={m['win_rate']:.1f}%")

# QQQ buy-and-hold benchmark
m_bh = full_metrics(qqq_bh, "QQQ Buy & Hold", spy_ret)
results["benchmark_qqq"] = m_bh
portfolio_returns["benchmark_qqq"] = qqq_bh
print(f"\n  QQQ B&H: Sharpe={m_bh['sharpe']:.3f}  Sortino={m_bh['sortino']:.3f}  "
      f"CAGR={m_bh['cagr']:.1f}%  MaxDD={m_bh['max_dd']:.1f}%")

print("\n== Portfolio Combinations ==")
labels_map = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F"}

for idx, builder in enumerate(portfolio_builders):
    w, label = builder()
    port_ret = apply_slippage(w, strat_rets)

    combined_pos = sum(w[s].abs() for s in strategies)

    m = full_metrics(port_ret, label, spy_ret, combined_pos)

    key = f"portfolio_{labels_map.get(idx, str(idx))}"
    results[key] = m
    portfolio_returns[key] = port_ret

    avg_w = {s: round(float(w[s].mean()), 3) for s in strategies}
    m["avg_weights"] = avg_w

    print(f"  {labels_map.get(idx, '?')}) {label}:")
    print(f"     Sharpe={m['sharpe']:.3f}  Sortino={m['sortino']:.3f}  "
          f"CAGR={m['cagr']:.1f}%  MaxDD={m['max_dd']:.1f}%  WR={m['win_rate']:.1f}%")
    print(f"     Avg weights: {avg_w}")
    if m['sharpe_bull'] is not None:
        print(f"     Regime: Bull Sharpe={m['sharpe_bull']:.3f}  Bear Sharpe={m['sharpe_bear']:.3f}  Gap={m['regime_gap']:.3f}")


# ── Permutation test (1000 iterations) ───────────────────────────────────────
# Correct approach: shuffle the DATE ALIGNMENT between positions and returns
# to break the signal-return link. This tests whether the strategy's timing
# (when it's long vs cash) adds value vs random timing.
print(f"\nRunning permutation test ({N_PERM} iterations) ...")

# We need the underlying market returns and each strategy's position series
# For individual strategies, shuffle position labels relative to market returns
# For portfolios, shuffle the constituent strategy returns relative to dates

# Store position series for individual strategies
position_series = {
    "individual_A": pos_a,
    "individual_v2F": pos_v2f,
    "individual_LevF": lev_f,
}

for key in list(results.keys()):
    if not key.startswith("portfolio_") and not key.startswith("individual_"):
        continue

    rets = portfolio_returns[key]
    actual_sharpe = annualized_sharpe(rets)
    count_better = 0

    if key in position_series:
        # Individual strategy: shuffle position assignments (circular shift)
        pos = position_series[key].reindex(strat_rets.index).fillna(0).values
        mkt = qqq_ret.reindex(strat_rets.index).fillna(0).values
        n = len(pos)
        for _ in range(N_PERM):
            shift = np.random.randint(20, n - 20)
            perm_pos = np.roll(pos, shift)
            perm_rets = perm_pos * mkt
            perm_sh = np.sqrt(252) * (perm_rets.mean() - RF/252) / (perm_rets.std() + 1e-10)
            if perm_sh >= actual_sharpe:
                count_better += 1
    else:
        # Portfolio: circular-shift each strategy's return series independently
        rets_dict = {s: strat_rets[s].values.copy() for s in strategies}
        n = len(strat_rets)
        for _ in range(N_PERM):
            perm_port = np.zeros(n)
            for s in strategies:
                shift = np.random.randint(20, n - 20)
                shifted = np.roll(rets_dict[s], shift)
                perm_port += shifted / len(strategies)  # equal-weight approx for perm
            perm_sh = np.sqrt(252) * (perm_port.mean() - RF/252) / (perm_port.std() + 1e-10)
            if perm_sh >= actual_sharpe:
                count_better += 1

    p_val = count_better / N_PERM
    results[key]["perm_p_value"] = round(p_val, 4)
    results[key]["gates"]["perm_p_lt_0.05"] = p_val < 0.05


# ── 5-gate validation summary ────────────────────────────────────────────────
print("\n== 5-Gate Validation ==")
for key, m in results.items():
    gates = m.get("gates", {})
    passed = sum(1 for v in gates.values() if v is True)
    total = len(gates)
    status = "PASS" if passed == total else "FAIL"
    failed = [g for g, v in gates.items() if v is not True]
    label = m.get("label", key)
    print(f"  {label}: {passed}/{total} gates -> {status}", end="")
    if failed:
        print(f"  (failed: {', '.join(failed)})", end="")
    print()


# ── Monthly income projections ───────────────────────────────────────────────
print("\n== Monthly Income Projections ==")
print(f"  {'Portfolio':<30} ", end="")
for cap in CAPITAL_LEVELS:
    print(f"  ${cap:>7,}", end="")
print()
print("  " + "-" * 75)

income_projections = {}

for key, m in results.items():
    if not key.startswith("portfolio_"):
        continue
    label = m["label"]
    monthly_ret = m["cagr"] / 100 / 12  # Simple monthly return from CAGR

    income = {}
    print(f"  {label:<30} ", end="")
    for cap in CAPITAL_LEVELS:
        monthly_income = cap * monthly_ret
        income[str(cap)] = round(monthly_income, 2)
        print(f"  ${monthly_income:>7,.0f}", end="")
    print(f"/mo")
    income_projections[key] = income

# ── Correlation matrix ───────────────────────────────────────────────────────
print("\n== Strategy Return Correlations ==")
corr = strat_rets.corr()
print(corr.round(3).to_string())

# ── Best portfolio recommendation ────────────────────────────────────────────
print("\n== Recommendation ==")
# Filter to portfolios that pass all 5 gates
passing = {k: v for k, v in results.items()
           if k.startswith("portfolio_")
           and all(v.get("gates", {}).get(g, False) for g in v["gates"])}

if passing:
    best_key = max(passing, key=lambda k: passing[k]["sortino"])
    best = passing[best_key]
    print(f"  BEST (by Sortino among 5-gate passers): {best['label']}")
    print(f"    Sharpe={best['sharpe']:.3f}  Sortino={best['sortino']:.3f}  "
          f"CAGR={best['cagr']:.1f}%  MaxDD={best['max_dd']:.1f}%")
    if "avg_weights" in best:
        print(f"    Weights: {best['avg_weights']}")
else:
    port_results = {k: v for k, v in results.items() if k.startswith("portfolio_")}
    best_key = max(port_results, key=lambda k: port_results[k]["sortino"])
    best = port_results[best_key]
    print(f"  BEST (by Sortino, NOTE: not all gates passed): {best['label']}")
    print(f"    Sharpe={best['sharpe']:.3f}  Sortino={best['sortino']:.3f}  "
          f"CAGR={best['cagr']:.1f}%  MaxDD={best['max_dd']:.1f}%")

# ── Comparison table vs QQQ ──────────────────────────────────────────────────
print("\n== All Results vs QQQ Buy & Hold ==")

print(f"  {'Portfolio':<30} {'Sharpe':>7} {'Sortino':>8} {'CAGR':>7} {'MaxDD':>7} {'WR':>6} {'Perm-p':>7} {'Gates':>6}")
print("  " + "-" * 82)

for key in ["benchmark_qqq"] + [k for k in results if k.startswith("individual_")] + \
           sorted([k for k in results if k.startswith("portfolio_")]):
    m = results[key]
    label = m["label"][:28]
    gates = m.get("gates", {})
    passed = sum(1 for v in gates.values() if v is True)
    total = len(gates)
    p_val = m.get("perm_p_value", "N/A")
    p_str = f"{p_val:.3f}" if isinstance(p_val, float) else p_val
    print(f"  {label:<30} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} "
          f"{m['cagr']:>6.1f}% {m['max_dd']:>6.1f}% {m['win_rate']:>5.1f}% {p_str:>7} {passed}/{total}")


# ── Save results ──────────────────────────────────────────────────────────────
output_path = Path("/home/jupiter/Lvl3Quant/data/multi_strategy_portfolio_results.json")

# Convert correlation to serializable format
corr_dict = {}
for col in corr.columns:
    corr_dict[col] = {idx: round(float(corr.loc[idx, col]), 4) for idx in corr.index}

save_data = {
    "generated": datetime.now().isoformat(),
    "period": f"{START} to {END}",
    "slippage_bps": SLIPPAGE_BPS,
    "risk_free_rate": RF,
    "n_permutations": N_PERM,
    "correlations": corr_dict,
    "results": results,
    "income_projections": income_projections,
    "recommendation": {
        "best_portfolio": best["label"],
        "sharpe": best["sharpe"],
        "sortino": best["sortino"],
        "cagr": best["cagr"],
        "max_dd": best["max_dd"],
        "weights": best.get("avg_weights"),
    }
}

with open(output_path, "w") as f:
    json.dump(save_data, f, indent=2, default=str)

print(f"\nResults saved to {output_path}")
print("Done.")

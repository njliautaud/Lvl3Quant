#!/usr/bin/env python3
"""
Trend Following Across Asset Classes — Multi-Variant Backtest
Academic basis: Moskowitz, Ooi & Pedersen (2012) "Time Series Momentum"

6 variants (A-F) tested across 13 ETFs spanning equities, bonds, commodities, real estate.
Validation: 5-gate framework (Sharpe, permutation test, regime gap, max DD, trade count).
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

warnings.filterwarnings("ignore")

# ─── Custom JSON encoder for numpy types ───
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        return super().default(obj)

# ─── Configuration ───
UNIVERSE = {
    "Equities": ["SPY", "QQQ", "IWM", "EFA", "EEM"],
    "Bonds": ["TLT", "IEF", "HYG"],
    "Commodities": ["GLD", "SLV", "USO", "DBA"],
    "Real Estate": ["VNQ"],
}
ALL_TICKERS = [t for group in UNIVERSE.values() for t in group]
CASH_PROXY = "SHY"

START = "2021-06-01"  # extra lookback for 12-month momentum
END = "2026-07-28"
BACKTEST_START = "2022-01-01"

STARTING_CAPITAL = 645.0
SLIPPAGE_BPS = 0.0002  # 0.02%

# ─── Download data ───
print("Downloading price data...")
tickers_to_download = list(set(ALL_TICKERS + [CASH_PROXY]))
raw = yf.download(tickers_to_download, start=START, end=END, auto_adjust=True, progress=False)

# Handle both multi-level and single-level columns
if isinstance(raw.columns, pd.MultiIndex):
    prices = raw["Close"].copy()
else:
    prices = raw[["Close"]].copy()
    prices.columns = tickers_to_download

# Forward fill then drop any remaining NaN rows at the start
prices = prices.ffill().dropna()
print(f"Data: {prices.index[0].date()} to {prices.index[-1].date()}, {len(prices)} trading days, {len(prices.columns)} assets")

# Compute returns
returns = prices.pct_change().fillna(0)

# Backtest mask
bt_mask = prices.index >= BACKTEST_START
bt_dates = prices.index[bt_mask]
bt_returns = returns.loc[bt_mask]

# ─── SPY regime classification (bull = SPY > 200-SMA) ───
spy_sma200 = prices["SPY"].rolling(200).mean()
spy_regime = (prices["SPY"] > spy_sma200).astype(int)  # 1=bull, 0=bear
spy_regime_bt = spy_regime.loc[bt_mask]


# ─── Helper functions ───
def apply_slippage(weights_change, slippage=SLIPPAGE_BPS):
    """Compute slippage cost from weight changes."""
    turnover = weights_change.abs().sum(axis=1)
    return turnover * slippage


def portfolio_returns(weights_df, returns_df, slippage=SLIPPAGE_BPS):
    """Compute portfolio returns given weights and asset returns."""
    # Align
    common = weights_df.index.intersection(returns_df.index)
    w = weights_df.loc[common]
    r = returns_df.loc[common]
    # Portfolio return = sum(w_i * r_i) - slippage
    port_ret = (w * r).sum(axis=1)
    # Slippage on weight changes
    w_change = w.diff().fillna(w.iloc[0:1])  # first day = full investment
    slip_cost = apply_slippage(w_change, slippage)
    port_ret = port_ret - slip_cost
    return port_ret


def compute_metrics(port_returns, spy_regime_series):
    """Compute all validation metrics."""
    ann_factor = 252
    total_days = len(port_returns)

    # Sharpe
    if port_returns.std() == 0:
        sharpe = 0.0
    else:
        sharpe = port_returns.mean() / port_returns.std() * np.sqrt(ann_factor)

    # Sortino
    downside = port_returns[port_returns < 0]
    if len(downside) == 0 or downside.std() == 0:
        sortino = sharpe  # no downside
    else:
        sortino = port_returns.mean() / downside.std() * np.sqrt(ann_factor)

    # Max drawdown
    cum = (1 + port_returns).cumprod()
    running_max = cum.cummax()
    dd = (cum - running_max) / running_max
    max_dd = dd.min()

    # Total return
    total_return = cum.iloc[-1] - 1 if len(cum) > 0 else 0

    # CAGR
    years = total_days / 252
    if years > 0 and cum.iloc[-1] > 0:
        cagr = cum.iloc[-1] ** (1 / years) - 1
    else:
        cagr = 0

    # Win rate (daily)
    win_rate = (port_returns > 0).mean()

    # Regime analysis
    aligned_regime = spy_regime_series.reindex(port_returns.index).ffill().fillna(1)
    bull_mask = aligned_regime == 1
    bear_mask = aligned_regime == 0

    bull_ret = port_returns[bull_mask]
    bear_ret = port_returns[bear_mask]

    if len(bull_ret) > 1 and bull_ret.std() > 0:
        sharpe_bull = bull_ret.mean() / bull_ret.std() * np.sqrt(ann_factor)
    else:
        sharpe_bull = 0.0

    if len(bear_ret) > 1 and bear_ret.std() > 0:
        sharpe_bear = bear_ret.mean() / bear_ret.std() * np.sqrt(ann_factor)
    else:
        sharpe_bear = 0.0

    # Regime gap
    denom = max(abs(sharpe_bull), abs(sharpe_bear), 1e-9)
    regime_gap = abs(sharpe_bull - sharpe_bear) / denom

    return {
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "cagr": round(cagr, 4),
        "total_return": round(total_return, 4),
        "max_drawdown": round(max_dd, 4),
        "win_rate": round(win_rate, 4),
        "total_days": int(total_days),
        "sharpe_bull": round(sharpe_bull, 4),
        "sharpe_bear": round(sharpe_bear, 4),
        "regime_gap": round(regime_gap, 4),
        "bull_days": int(bull_mask.sum()),
        "bear_days": int(bear_mask.sum()),
    }


def permutation_test(port_returns, weights_df, returns_df, n_perms=1000):
    """Shuffle trend signals (monthly blocks) and compare Sharpe. Vectorized."""
    actual_sharpe = port_returns.mean() / port_returns.std() * np.sqrt(252) if port_returns.std() > 0 else 0
    count_better = 0

    # Pre-compute aligned arrays for speed
    common = weights_df.index.intersection(returns_df.index)
    w_vals = weights_df.loc[common].values  # (T, N)
    r_vals = returns_df.reindex(columns=weights_df.columns, fill_value=0).loc[common].values  # (T, N)
    n_assets = w_vals.shape[1]

    for _ in range(n_perms):
        # Shuffle column assignment for each month block
        shuffled_w = w_vals.copy()
        # Create month labels
        months = common.to_period("M")
        unique_months = months.unique()
        for m in unique_months:
            mask = (months == m)
            perm = np.random.permutation(n_assets)
            shuffled_w[mask] = shuffled_w[mask][:, perm]

        # Compute portfolio returns directly with numpy
        perm_port = (shuffled_w * r_vals).sum(axis=1)
        # Simple slippage approximation (skip for speed — minor effect on Sharpe comparison)
        std = perm_port.std()
        if std > 0:
            perm_sharpe = perm_port.mean() / std * np.sqrt(252)
        else:
            perm_sharpe = 0
        if perm_sharpe >= actual_sharpe:
            count_better += 1

    p_value = count_better / n_perms
    return p_value


def count_trades(weights_df):
    """Count rebalance events (any weight change > 1%)."""
    changes = weights_df.diff().abs()
    rebalance_days = (changes.sum(axis=1) > 0.01).sum()
    return int(rebalance_days)


def validate(metrics, p_value, n_trades):
    """Apply 5-gate validation."""
    gates = {
        "sharpe_gt_0.5": metrics["sharpe"] > 0.5,
        "perm_test_p_lt_0.05": p_value < 0.05,
        "regime_gap_lt_0.5": metrics["regime_gap"] < 0.5,
        "max_dd_gt_neg50pct": metrics["max_drawdown"] > -0.50,
        "trades_gte_20": n_trades >= 20,
    }
    gates["all_passed"] = all(gates.values())
    return gates


# ═══════════════════════════════════════════════════════
# VARIANT A: Classic TSMOM (12-1 month)
# ═══════════════════════════════════════════════════════
def variant_a():
    print("\n[A] Classic TSMOM (12-1 month)...")
    weights = pd.DataFrame(0.0, index=bt_dates, columns=ALL_TICKERS)

    # Only compute on last trading day of each month, then ffill
    month_ends = bt_dates.to_series().groupby([bt_dates.year, bt_dates.month]).last()

    for month_end in month_ends:
        date_loc = prices.index.searchsorted(month_end)
        if date_loc >= len(prices.index):
            date_loc = len(prices.index) - 1
        if date_loc < 252:
            continue

        p_12m = prices.iloc[date_loc - 252]
        p_1m = prices.iloc[date_loc - 21]

        mom_12_1 = (p_1m / p_12m) - 1  # 12-1 month momentum

        longs = [t for t in ALL_TICKERS if mom_12_1.get(t, -1) > 0]
        month_mask = (bt_dates.year == month_end.year) & (bt_dates.month == month_end.month)
        if len(longs) > 0:
            w = 1.0 / len(longs)
            for t in longs:
                weights.loc[month_mask, t] = w

    return weights


# ═══════════════════════════════════════════════════════
# VARIANT B: Dual Momentum (absolute + relative)
# ═══════════════════════════════════════════════════════
def variant_b():
    print("\n[B] Dual Momentum (absolute + relative)...")
    weights = pd.DataFrame(0.0, index=bt_dates, columns=ALL_TICKERS + [CASH_PROXY])

    # Get last trading day of each month from actual trading dates
    month_ends = bt_dates.to_series().groupby([bt_dates.year, bt_dates.month]).last()

    for month_end in month_ends:
        # Find nearest trading day in prices index
        loc_idx = prices.index.searchsorted(month_end)
        if loc_idx >= len(prices.index):
            loc_idx = len(prices.index) - 1
        actual_date = prices.index[loc_idx]
        date_loc = loc_idx
        if date_loc < 126:
            continue

        # 6-month returns
        p_now = prices.iloc[date_loc]
        p_6m = prices.iloc[date_loc - 126]
        mom_6m = (p_now / p_6m) - 1

        # Absolute momentum filter + relative ranking
        qualified = {t: mom_6m[t] for t in ALL_TICKERS if t in mom_6m and mom_6m[t] > 0}
        top4 = sorted(qualified, key=qualified.get, reverse=True)[:4]

        # Assign weights
        month_mask = (bt_dates.year == month_end.year) & (bt_dates.month == month_end.month)
        n_long = len(top4)
        n_cash = 4 - n_long

        for t in top4:
            weights.loc[month_mask, t] = 0.25
        if n_cash > 0:
            weights.loc[month_mask, CASH_PROXY] = n_cash * 0.25

    return weights


# ═══════════════════════════════════════════════════════
# VARIANT C: Breakout (Donchian Channel)
# ═══════════════════════════════════════════════════════
def variant_c():
    print("\n[C] Breakout (Donchian Channel 60/20)...")
    # Vectorized Donchian channel
    high_60 = prices[ALL_TICKERS].rolling(60).max()
    low_20 = prices[ALL_TICKERS].rolling(20).min()

    # Pre-compute entry/exit signals
    p = prices[ALL_TICKERS]
    entry_signal = (p >= high_60)  # new 60-day high
    exit_signal = (p <= low_20)    # break below 20-day low

    # Track positions with state (need loop for state dependency, but use numpy arrays)
    bt_p = p.loc[bt_mask].values
    bt_entry = entry_signal.loc[bt_mask].values
    bt_exit = exit_signal.loc[bt_mask].values
    n_days = len(bt_dates)
    n_assets = len(ALL_TICKERS)

    pos = np.zeros(n_assets, dtype=bool)
    weight_arr = np.zeros((n_days, n_assets))

    for i in range(n_days):
        # Entry: not in position and entry signal
        new_entry = ~pos & bt_entry[i]
        # Exit: in position and exit signal
        new_exit = pos & bt_exit[i]
        pos = pos | new_entry
        pos = pos & ~new_exit

        n_active = pos.sum()
        if n_active > 0:
            weight_arr[i, pos] = 1.0 / n_active

    weights = pd.DataFrame(weight_arr, index=bt_dates, columns=ALL_TICKERS)
    return weights


# ═══════════════════════════════════════════════════════
# VARIANT D: Adaptive Momentum (multi-timeframe)
# ═══════════════════════════════════════════════════════
def variant_d():
    print("\n[D] Adaptive Momentum (1m/3m/6m, weekly rebal)...")
    weights = pd.DataFrame(0.0, index=bt_dates, columns=ALL_TICKERS)

    # Weekly rebalance dates — use actual last trading day of each week
    week_ends = bt_dates.to_series().groupby([bt_dates.year, bt_dates.isocalendar().week]).last()

    for week_end in week_ends:
        loc_idx = prices.index.searchsorted(week_end)
        if loc_idx >= len(prices.index):
            loc_idx = len(prices.index) - 1
        date_loc = loc_idx
        if date_loc < 126:
            continue

        p_now = prices.iloc[date_loc]
        p_1m = prices.iloc[date_loc - 21]
        p_3m = prices.iloc[date_loc - 63]
        p_6m = prices.iloc[date_loc - 126]

        mom_1m = (p_now / p_1m) - 1
        mom_3m = (p_now / p_3m) - 1
        mom_6m = (p_now / p_6m) - 1

        signal_strength = {}
        for t in ALL_TICKERS:
            votes = sum([
                1 if mom_1m.get(t, -1) > 0 else 0,
                1 if mom_3m.get(t, -1) > 0 else 0,
                1 if mom_6m.get(t, -1) > 0 else 0,
            ])
            if votes >= 2:
                signal_strength[t] = votes / 3.0

        # Weight by signal strength
        if signal_strength:
            total_strength = sum(signal_strength.values())
            # Apply to the week
            next_week_mask = (bt_dates >= week_end) & (bt_dates < week_end + pd.Timedelta(days=8))
            for t, strength in signal_strength.items():
                weights.loc[next_week_mask, t] = strength / total_strength

    return weights


# ═══════════════════════════════════════════════════════
# VARIANT E: Risk Parity Trend
# ═══════════════════════════════════════════════════════
def variant_e():
    print("\n[E] Risk Parity Trend (TSMOM + inverse vol weighting)...")
    weights = pd.DataFrame(0.0, index=bt_dates, columns=ALL_TICKERS)

    vol_20 = returns.rolling(20).std()

    # Get last trading day of each month from actual trading dates
    month_ends = bt_dates.to_series().groupby([bt_dates.year, bt_dates.month]).last()

    for month_end in month_ends:
        loc_idx = prices.index.searchsorted(month_end)
        if loc_idx >= len(prices.index):
            loc_idx = len(prices.index) - 1
        date_loc = loc_idx
        if date_loc < 252:
            continue

        actual_date = prices.index[date_loc]
        p_now = prices.iloc[date_loc]
        p_12m = prices.iloc[date_loc - 252]
        p_1m = prices.iloc[date_loc - 21]
        mom_12_1 = (p_1m / p_12m) - 1

        longs = [t for t in ALL_TICKERS if mom_12_1.get(t, -1) > 0]

        if len(longs) > 0:
            vols = {}
            for t in longs:
                v = vol_20.iloc[date_loc][t]
                if pd.notna(v) and v > 0:
                    vols[t] = 1.0 / v

            if vols:
                total_inv_vol = sum(vols.values())
                month_mask = (bt_dates.year == month_end.year) & (bt_dates.month == month_end.month)
                for t, iv in vols.items():
                    weights.loc[month_mask, t] = iv / total_inv_vol

    return weights


# ═══════════════════════════════════════════════════════
# VARIANT F: Adversarial (Random signals)
# ═══════════════════════════════════════════════════════
def variant_f(avg_active):
    print(f"\n[F] Adversarial (random signals, ~{avg_active:.1f} active positions)...")
    np.random.seed(42)
    weights = pd.DataFrame(0.0, index=bt_dates, columns=ALL_TICKERS)
    n_active = max(1, int(round(avg_active)))

    month_ends = bt_dates.to_series().groupby([bt_dates.year, bt_dates.month]).last()

    for month_end in month_ends:
        chosen = np.random.choice(ALL_TICKERS, size=min(n_active, len(ALL_TICKERS)), replace=False)
        w = 1.0 / len(chosen)
        month_mask = (bt_dates.year == month_end.year) & (bt_dates.month == month_end.month)
        for t in chosen:
            weights.loc[month_mask, t] = w

    return weights


# ═══════════════════════════════════════════════════════
# RUN ALL VARIANTS
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TREND FOLLOWING MULTI-ASSET BACKTEST")
print("Moskowitz, Ooi & Pedersen (2012) — Time Series Momentum")
print("=" * 70)

results = {}
all_weights = {}

# Returns aligned to backtest period for all tickers + cash proxy
bt_returns_all = returns.loc[bt_mask].reindex(columns=ALL_TICKERS + [CASH_PROXY], fill_value=0)

variant_funcs = {
    "A_classic_tsmom": variant_a,
    "B_dual_momentum": variant_b,
    "C_donchian_breakout": variant_c,
    "D_adaptive_momentum": variant_d,
    "E_risk_parity_trend": variant_e,
}

for name, func in variant_funcs.items():
    w = func()
    all_weights[name] = w

    # Ensure columns match
    for col in w.columns:
        if col not in bt_returns_all.columns:
            bt_returns_all[col] = returns[col].loc[bt_mask] if col in returns.columns else 0

    port_ret = portfolio_returns(w, bt_returns_all)
    metrics = compute_metrics(port_ret, spy_regime_bt)
    n_trades = count_trades(w)
    p_val = permutation_test(port_ret, w, bt_returns_all, n_perms=1000)
    gates = validate(metrics, p_val, n_trades)

    final_equity = STARTING_CAPITAL * (1 + port_ret).cumprod().iloc[-1]

    results[name] = {
        "metrics": metrics,
        "p_value": round(p_val, 4),
        "n_trades": n_trades,
        "final_equity": round(final_equity, 2),
        "starting_capital": STARTING_CAPITAL,
        "validation_gates": gates,
    }

    label = chr(65 + list(variant_funcs.keys()).index(name))
    passed = "PASS" if gates["all_passed"] else "FAIL"
    print(f"  [{label}] {name}: Sharpe={metrics['sharpe']:.2f}, Sortino={metrics['sortino']:.2f}, "
          f"MaxDD={metrics['max_drawdown']:.1%}, RegimeGap={metrics['regime_gap']:.2f}, "
          f"p={p_val:.3f}, trades={n_trades}, ${STARTING_CAPITAL}→${final_equity:.2f} [{passed}]")

# Variant F: adversarial — use average active positions from best variant
best_variant = max(results, key=lambda k: results[k]["metrics"]["sharpe"])
best_weights = all_weights[best_variant]
avg_active = (best_weights > 0).sum(axis=1).mean()

w_f = variant_f(avg_active)
all_weights["F_adversarial_random"] = w_f
port_ret_f = portfolio_returns(w_f, bt_returns_all)
metrics_f = compute_metrics(port_ret_f, spy_regime_bt)
n_trades_f = count_trades(w_f)
p_val_f = permutation_test(port_ret_f, w_f, bt_returns_all, n_perms=1000)
gates_f = validate(metrics_f, p_val_f, n_trades_f)
final_equity_f = STARTING_CAPITAL * (1 + port_ret_f).cumprod().iloc[-1]

results["F_adversarial_random"] = {
    "metrics": metrics_f,
    "p_value": round(p_val_f, 4),
    "n_trades": n_trades_f,
    "final_equity": round(final_equity_f, 2),
    "starting_capital": STARTING_CAPITAL,
    "validation_gates": gates_f,
}

print(f"  [F] F_adversarial_random: Sharpe={metrics_f['sharpe']:.2f}, Sortino={metrics_f['sortino']:.2f}, "
      f"MaxDD={metrics_f['max_drawdown']:.1%}, RegimeGap={metrics_f['regime_gap']:.2f}, "
      f"p={p_val_f:.3f}, trades={n_trades_f}, ${STARTING_CAPITAL}→${final_equity_f:.2f} "
      f"[{'PASS' if gates_f['all_passed'] else 'FAIL'}]")

# ═══════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SUMMARY — VALIDATION GATES")
print("=" * 70)
print(f"{'Variant':<25} {'Sharpe':>7} {'Sortino':>8} {'MaxDD':>8} {'RegGap':>8} {'p-val':>7} {'Trades':>7} {'Result':>8}")
print("-" * 82)

for name, res in results.items():
    m = res["metrics"]
    passed = "PASS" if res["validation_gates"]["all_passed"] else "FAIL"
    print(f"{name:<25} {m['sharpe']:>7.2f} {m['sortino']:>8.2f} {m['max_drawdown']:>7.1%} "
          f"{m['regime_gap']:>8.2f} {res['p_value']:>7.3f} {res['n_trades']:>7} {passed:>8}")

# Best variant
best = max(results, key=lambda k: results[k]["metrics"]["sharpe"])
print(f"\nBest variant by Sharpe: {best} (Sharpe={results[best]['metrics']['sharpe']:.2f})")

# Regime insight
print("\n--- Regime Analysis (Bull vs Bear) ---")
for name, res in results.items():
    m = res["metrics"]
    print(f"  {name}: Bull Sharpe={m['sharpe_bull']:.2f} ({m['bull_days']}d), "
          f"Bear Sharpe={m['sharpe_bear']:.2f} ({m['bear_days']}d), Gap={m['regime_gap']:.2f}")

# ═══════════════════════════════════════════════════════
# SAVE RESULTS
# ═══════════════════════════════════════════════════════
output = {
    "metadata": {
        "strategy": "Trend Following Multi-Asset",
        "academic_basis": "Moskowitz, Ooi & Pedersen (2012) Time Series Momentum",
        "universe": UNIVERSE,
        "period": f"{BACKTEST_START} to {END}",
        "starting_capital": STARTING_CAPITAL,
        "slippage_bps": SLIPPAGE_BPS * 10000,
        "commission": 0,
        "run_timestamp": datetime.now().isoformat(),
    },
    "variants": results,
    "best_variant": best,
    "validation_summary": {
        "passed": [k for k, v in results.items() if v["validation_gates"]["all_passed"]],
        "failed": [k for k, v in results.items() if not v["validation_gates"]["all_passed"]],
    },
}

output_path = "/home/jupiter/Lvl3Quant/data/trend_following_multiasset_results.json"
with open(output_path, "w") as f:
    json.dump(output, f, indent=2, cls=NumpyEncoder)

print(f"\nResults saved to {output_path}")
print("Done.")

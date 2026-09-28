#!/usr/bin/env python3
"""
Wheel Strategy Backtest — 6 Variants
Simulated options returns using yfinance + VIX as IV proxy.
For a $667 Robinhood account.
"""

import json
import numpy as np
import yfinance as yf
import pandas as pd
from datetime import datetime
from scipy import stats

# ─── Download data ───────────────────────────────────────────────────
START = "2022-01-01"
END = "2026-07-28"

print("Downloading price data...")
tickers = ["SPY", "IWM", "F", "SOFI", "PLTR", "^VIX"]
data = {}
for t in tickers:
    df = yf.download(t, start=START, end=END, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    data[t] = df[["Close"]].rename(columns={"Close": "close"}).dropna()
    print(f"  {t}: {len(data[t])} days")

vix = data["^VIX"]["close"].reindex(data["SPY"].index, method="ffill") / 100.0

# ─── Helper: simulate CSP on a weekly/monthly cadence ────────────────
def simulate_csp(prices, vix_series, dte_days, otm_pct, cadence_days, account_size=667):
    """
    Simulate cash-secured put selling.
    - prices: Series of daily closes
    - dte_days: option DTE in calendar days
    - otm_pct: how far OTM the put is (e.g., 0.02 = 2% OTM, ~30 delta)
    - cadence_days: how often we sell a new put (5=weekly, 21=monthly)
    Returns list of trade dicts.
    """
    trades = []
    dates = prices.index
    i = 0
    while i < len(dates) - dte_days:
        entry_date = dates[i]
        entry_price = prices.iloc[i]

        # Can we afford 100 shares? (for CSP collateral)
        strike = entry_price * (1 - otm_pct)
        collateral = strike * 100

        # IV from VIX (proxy)
        iv = vix_series.get(entry_date, 0.20)
        if pd.isna(iv) or iv < 0.05:
            iv = 0.20

        # Premium approximation: BSM-like
        # For OTM put: premium ≈ price * N(-d2) * IV * sqrt(T) (simplified)
        T = dte_days / 365.0
        premium_pct = 0.4 * iv * np.sqrt(T) * np.exp(-otm_pct / (iv * np.sqrt(T)))
        premium_per_share = entry_price * premium_pct
        premium_total = premium_per_share * 100

        # Find expiry
        exp_idx = min(i + dte_days, len(dates) - 1)
        exp_date = dates[exp_idx]
        exp_price = prices.iloc[exp_idx]

        # Assignment check
        assigned = exp_price < strike

        if assigned:
            # We buy 100 shares at strike, collected premium
            loss_per_share = strike - exp_price
            pnl = (premium_per_share - loss_per_share) * 100
        else:
            # Keep full premium
            pnl = premium_total

        trades.append({
            "entry_date": str(entry_date.date()),
            "exp_date": str(exp_date.date()),
            "entry_price": round(float(entry_price), 2),
            "strike": round(float(strike), 2),
            "premium": round(float(premium_total), 2),
            "exp_price": round(float(exp_price), 2),
            "assigned": bool(assigned),
            "pnl": round(float(pnl), 2),
        })

        i += cadence_days

    return trades


def simulate_covered_call(prices, vix_series, dte_days, otm_pct, cadence_days, entry_cost):
    """
    After assignment, sell covered calls until called away.
    entry_cost = price we were assigned at (the put strike).
    """
    trades = []
    dates = prices.index
    i = 0
    holding = False
    shares_cost = entry_cost

    while i < len(dates) - dte_days:
        entry_date = dates[i]
        entry_price = prices.iloc[i]

        iv = vix_series.get(entry_date, 0.20)
        if pd.isna(iv) or iv < 0.05:
            iv = 0.20

        # Sell OTM call
        call_strike = entry_price * (1 + otm_pct)
        T = dte_days / 365.0
        premium_pct = 0.4 * iv * np.sqrt(T) * np.exp(-otm_pct / (iv * np.sqrt(T)))
        premium_per_share = entry_price * premium_pct

        exp_idx = min(i + dte_days, len(dates) - 1)
        exp_date = dates[exp_idx]
        exp_price = prices.iloc[exp_idx]

        called_away = exp_price > call_strike

        # P&L: premium + stock movement (capped at strike if called)
        stock_pnl = min(exp_price, call_strike) - entry_price
        pnl = (premium_per_share + stock_pnl) * 100

        trades.append({
            "entry_date": str(entry_date.date()),
            "exp_date": str(exp_date.date()),
            "entry_price": round(float(entry_price), 2),
            "call_strike": round(float(call_strike), 2),
            "premium": round(float(premium_per_share * 100), 2),
            "exp_price": round(float(exp_price), 2),
            "called_away": bool(called_away),
            "pnl": round(float(pnl), 2),
        })

        i += cadence_days

    return trades


def simulate_put_credit_spread(prices, vix_series, dte_days, width_pct, cadence_days):
    """
    Put credit spread: sell put at otm_pct, buy put at otm_pct + width_pct.
    Defined risk. Collect ~30% of width.
    """
    trades = []
    dates = prices.index
    otm_pct = 0.02  # short put 2% OTM
    i = 0

    while i < len(dates) - dte_days:
        entry_date = dates[i]
        entry_price = prices.iloc[i]

        short_strike = entry_price * (1 - otm_pct)
        long_strike = short_strike - (entry_price * width_pct)

        iv = vix_series.get(entry_date, 0.20)
        if pd.isna(iv) or iv < 0.05:
            iv = 0.20

        T = dte_days / 365.0
        width_dollars = short_strike - long_strike
        # Collect ~30% of width
        premium_per_share = width_dollars * 0.30
        max_loss_per_share = width_dollars - premium_per_share

        exp_idx = min(i + dte_days, len(dates) - 1)
        exp_date = dates[exp_idx]
        exp_price = prices.iloc[exp_idx]

        if exp_price >= short_strike:
            pnl = premium_per_share * 100
        elif exp_price <= long_strike:
            pnl = -max_loss_per_share * 100
        else:
            intrinsic = short_strike - exp_price
            pnl = (premium_per_share - intrinsic) * 100

        trades.append({
            "entry_date": str(entry_date.date()),
            "exp_date": str(exp_date.date()),
            "entry_price": round(float(entry_price), 2),
            "short_strike": round(float(short_strike), 2),
            "long_strike": round(float(long_strike), 2),
            "premium": round(float(premium_per_share * 100), 2),
            "exp_price": round(float(exp_price), 2),
            "pnl": round(float(pnl), 2),
        })

        i += cadence_days

    return trades


def simulate_random_adversarial(real_trades, n_perms=1):
    """Generate random trades with same distribution as real ones."""
    pnls = [t["pnl"] for t in real_trades]
    mean_abs = np.mean(np.abs(pnls))
    random_trades = []
    for t in real_trades:
        rand_pnl = np.random.normal(0, mean_abs)
        random_trades.append({
            "entry_date": t["entry_date"],
            "pnl": round(float(rand_pnl), 2),
        })
    return random_trades


# ─── Gate checks ─────────────────────────────────────────────────────
def compute_metrics(trades, account_size=667):
    """Compute strategy metrics from trade list."""
    pnls = np.array([t["pnl"] for t in trades])
    n_trades = len(pnls)

    if n_trades < 2:
        return None

    # Returns as % of account
    returns = pnls / account_size

    # Sharpe (annualized, assume ~52 trades/year for weekly, 12 for monthly)
    # Use actual trade frequency
    dates = [t.get("entry_date", t.get("exp_date", "")) for t in trades]
    if dates[0] and dates[-1]:
        d0 = pd.Timestamp(dates[0])
        d1 = pd.Timestamp(dates[-1])
        years = max((d1 - d0).days / 365.25, 0.1)
        trades_per_year = n_trades / years
    else:
        trades_per_year = 52

    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1)

    sharpe = (mean_ret / std_ret) * np.sqrt(trades_per_year) if std_ret > 0 else 0

    # Sortino
    downside = returns[returns < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else std_ret
    sortino = (mean_ret / downside_std) * np.sqrt(trades_per_year) if downside_std > 0 else 0

    # Profit Factor
    gross_profit = np.sum(pnls[pnls > 0])
    gross_loss = np.abs(np.sum(pnls[pnls < 0]))
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Win Rate
    wr = np.sum(pnls > 0) / n_trades

    # Max Drawdown (on cumulative equity curve)
    cum_pnl = np.cumsum(pnls)
    equity = account_size + cum_pnl
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = float(np.min(dd))

    # Total return
    total_return = float(cum_pnl[-1]) / account_size

    # Permutation test (1000 iterations)
    observed_sharpe = sharpe
    n_better = 0
    for _ in range(1000):
        shuffled = np.random.permutation(returns) * np.random.choice([-1, 1], size=n_trades)
        s_mean = np.mean(shuffled)
        s_std = np.std(shuffled, ddof=1)
        s_sharpe = (s_mean / s_std) * np.sqrt(trades_per_year) if s_std > 0 else 0
        if s_sharpe >= observed_sharpe:
            n_better += 1
    perm_p = n_better / 1000.0

    # Regime analysis: split by SPY returns
    spy_prices = data["SPY"]["close"]
    up_pnls = []
    down_pnls = []
    for t in trades:
        d = t.get("entry_date", "")
        if not d:
            continue
        ts = pd.Timestamp(d)
        # Find nearest SPY date
        idx = spy_prices.index.get_indexer([ts], method="nearest")[0]
        if idx > 20:
            spy_20d_ret = (spy_prices.iloc[idx] - spy_prices.iloc[idx - 20]) / spy_prices.iloc[idx - 20]
            if spy_20d_ret > 0:
                up_pnls.append(t["pnl"])
            else:
                down_pnls.append(t["pnl"])

    up_sharpe = 0
    down_sharpe = 0
    if len(up_pnls) > 2:
        up_ret = np.array(up_pnls) / account_size
        up_sharpe = (np.mean(up_ret) / np.std(up_ret, ddof=1)) * np.sqrt(trades_per_year) if np.std(up_ret, ddof=1) > 0 else 0
    if len(down_pnls) > 2:
        dn_ret = np.array(down_pnls) / account_size
        down_sharpe = (np.mean(dn_ret) / np.std(dn_ret, ddof=1)) * np.sqrt(trades_per_year) if np.std(dn_ret, ddof=1) > 0 else 0

    max_abs = max(abs(up_sharpe), abs(down_sharpe), 0.001)
    regime_gap = abs(up_sharpe - down_sharpe) / max_abs

    return {
        "n_trades": int(n_trades),
        "total_pnl": round(float(np.sum(pnls)), 2),
        "total_return_pct": round(total_return * 100, 1),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "profit_factor": round(pf, 3),
        "win_rate": round(float(wr), 3),
        "max_drawdown_pct": round(max_dd * 100, 1),
        "avg_trade_pnl": round(float(np.mean(pnls)), 2),
        "perm_test_p": round(perm_p, 4),
        "regime_up_sharpe": round(up_sharpe, 3),
        "regime_down_sharpe": round(down_sharpe, 3),
        "regime_gap": round(regime_gap, 3),
        "up_regime_trades": len(up_pnls),
        "down_regime_trades": len(down_pnls),
    }


def check_gates(metrics):
    """Check all 5 gates."""
    if metrics is None:
        return {"pass": False, "failures": ["insufficient data"]}

    failures = []
    if metrics["sharpe"] < 0.5:
        failures.append(f"Sharpe {metrics['sharpe']:.3f} < 0.5")
    if metrics["perm_test_p"] >= 0.05:
        failures.append(f"Perm p={metrics['perm_test_p']:.4f} >= 0.05")
    if metrics["regime_gap"] >= 0.50:
        failures.append(f"Regime gap {metrics['regime_gap']:.3f} >= 0.50")
    if metrics["max_drawdown_pct"] < -50:
        failures.append(f"MaxDD {metrics['max_drawdown_pct']:.1f}% < -50%")
    if metrics["n_trades"] < 20:
        failures.append(f"Trades {metrics['n_trades']} < 20")

    return {"pass": len(failures) == 0, "failures": failures}


# ─── Run all 6 variants ─────────────────────────────────────────────
print("\n=== Running Wheel Strategy Backtest ===\n")
results = {}

# A) Weekly CSP on SPY
print("A) Weekly CSP on SPY...")
spy_trades = simulate_csp(data["SPY"]["close"], vix, dte_days=5, otm_pct=0.02, cadence_days=5)
metrics_a = compute_metrics(spy_trades)
gates_a = check_gates(metrics_a)
results["A_weekly_csp_spy"] = {
    "description": "Weekly CSP on SPY, 30-delta (2% OTM), 5 DTE",
    "metrics": metrics_a,
    "gates": gates_a,
    "sample_trades": spy_trades[:3] + spy_trades[-3:] if len(spy_trades) > 6 else spy_trades,
}
print(f"   Trades: {metrics_a['n_trades']}, Sharpe: {metrics_a['sharpe']}, PF: {metrics_a['profit_factor']}, WR: {metrics_a['win_rate']}")
print(f"   Gates: {'PASS' if gates_a['pass'] else 'FAIL — ' + '; '.join(gates_a['failures'])}")

# B) Monthly CSP on IWM
print("B) Monthly CSP on IWM...")
iwm_trades = simulate_csp(data["IWM"]["close"], vix, dte_days=21, otm_pct=0.04, cadence_days=21)
metrics_b = compute_metrics(iwm_trades)
gates_b = check_gates(metrics_b)
results["B_monthly_csp_iwm"] = {
    "description": "Monthly CSP on IWM, 30-delta (4% OTM), 21 DTE",
    "metrics": metrics_b,
    "gates": gates_b,
    "sample_trades": iwm_trades[:3] + iwm_trades[-3:] if len(iwm_trades) > 6 else iwm_trades,
}
print(f"   Trades: {metrics_b['n_trades']}, Sharpe: {metrics_b['sharpe']}, PF: {metrics_b['profit_factor']}, WR: {metrics_b['win_rate']}")
print(f"   Gates: {'PASS' if gates_b['pass'] else 'FAIL — ' + '; '.join(gates_b['failures'])}")

# C) Weekly CSP on cheap stocks (F, SOFI, PLTR)
print("C) Weekly CSP on cheap stocks (F, SOFI, PLTR)...")
cheap_trades = []
for ticker in ["F", "SOFI", "PLTR"]:
    t_trades = simulate_csp(data[ticker]["close"], vix, dte_days=5, otm_pct=0.03, cadence_days=5)
    for t in t_trades:
        t["ticker"] = ticker
    cheap_trades.extend(t_trades)
cheap_trades.sort(key=lambda x: x["entry_date"])
metrics_c = compute_metrics(cheap_trades)
gates_c = check_gates(metrics_c)
results["C_weekly_csp_cheap"] = {
    "description": "Weekly CSP on F/SOFI/PLTR, 30-delta (3% OTM), 5 DTE",
    "metrics": metrics_c,
    "gates": gates_c,
    "sample_trades": cheap_trades[:3] + cheap_trades[-3:] if len(cheap_trades) > 6 else cheap_trades,
}
print(f"   Trades: {metrics_c['n_trades']}, Sharpe: {metrics_c['sharpe']}, PF: {metrics_c['profit_factor']}, WR: {metrics_c['win_rate']}")
print(f"   Gates: {'PASS' if gates_c['pass'] else 'FAIL — ' + '; '.join(gates_c['failures'])}")

# D) Covered Call after assignment (simulate on SPY as if assigned)
print("D) Covered Call on SPY (post-assignment)...")
cc_trades = simulate_covered_call(data["SPY"]["close"], vix, dte_days=5, otm_pct=0.02, cadence_days=5,
                                   entry_cost=float(data["SPY"]["close"].iloc[0]))
metrics_d = compute_metrics(cc_trades)
gates_d = check_gates(metrics_d)
results["D_covered_call_spy"] = {
    "description": "Weekly Covered Call on SPY, 30-delta (2% OTM), 5 DTE",
    "metrics": metrics_d,
    "gates": gates_d,
    "sample_trades": cc_trades[:3] + cc_trades[-3:] if len(cc_trades) > 6 else cc_trades,
}
print(f"   Trades: {metrics_d['n_trades']}, Sharpe: {metrics_d['sharpe']}, PF: {metrics_d['profit_factor']}, WR: {metrics_d['win_rate']}")
print(f"   Gates: {'PASS' if gates_d['pass'] else 'FAIL — ' + '; '.join(gates_d['failures'])}")

# E) Put Credit Spread on SPY
print("E) Put Credit Spread on SPY ($1 wide)...")
pcs_trades = simulate_put_credit_spread(data["SPY"]["close"], vix, dte_days=5, width_pct=0.002, cadence_days=5)
metrics_e = compute_metrics(pcs_trades)
gates_e = check_gates(metrics_e)
results["E_put_credit_spread_spy"] = {
    "description": "Weekly Put Credit Spread on SPY, $1 wide, 5 DTE",
    "metrics": metrics_e,
    "gates": gates_e,
    "sample_trades": pcs_trades[:3] + pcs_trades[-3:] if len(pcs_trades) > 6 else pcs_trades,
}
print(f"   Trades: {metrics_e['n_trades']}, Sharpe: {metrics_e['sharpe']}, PF: {metrics_e['profit_factor']}, WR: {metrics_e['win_rate']}")
print(f"   Gates: {'PASS' if gates_e['pass'] else 'FAIL — ' + '; '.join(gates_e['failures'])}")

# F) Random adversarial baseline
print("F) Random adversarial baseline...")
np.random.seed(42)
random_trades = simulate_random_adversarial(spy_trades)
metrics_f = compute_metrics(random_trades)
gates_f = check_gates(metrics_f)
results["F_random_adversarial"] = {
    "description": "Random P&L with same magnitude distribution as SPY CSP (should fail)",
    "metrics": metrics_f,
    "gates": gates_f,
}
print(f"   Trades: {metrics_f['n_trades']}, Sharpe: {metrics_f['sharpe']}, PF: {metrics_f['profit_factor']}, WR: {metrics_f['win_rate']}")
print(f"   Gates: {'PASS' if gates_f['pass'] else 'FAIL — ' + '; '.join(gates_f['failures'])}")

# ─── Summary ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY — WHEEL STRATEGY VARIANTS ($667 ACCOUNT)")
print("=" * 70)
print(f"{'Variant':<35} {'Sharpe':>7} {'Sort':>7} {'PF':>7} {'WR':>6} {'MaxDD':>7} {'Return':>8} {'Gates':>6}")
print("-" * 70)
for key, val in results.items():
    m = val["metrics"]
    g = val["gates"]
    label = key.split("_", 1)[0] + ") " + val["description"][:28]
    if m:
        print(f"{label:<35} {m['sharpe']:>7.2f} {m['sortino']:>7.2f} {m['profit_factor']:>7.2f} {m['win_rate']:>5.1%} {m['max_drawdown_pct']:>6.1f}% {m['total_return_pct']:>7.1f}% {'PASS' if g['pass'] else 'FAIL':>6}")
    else:
        print(f"{label:<35} {'N/A':>7} {'N/A':>7} {'N/A':>7} {'N/A':>6} {'N/A':>7} {'N/A':>8} {'FAIL':>6}")

# Print which passed
print("\n--- GATE RESULTS ---")
for key, val in results.items():
    g = val["gates"]
    status = "PASS ALL 5 GATES" if g["pass"] else f"FAIL: {'; '.join(g['failures'])}"
    print(f"  {key}: {status}")

# ─── Small account viability analysis ────────────────────────────────
print("\n--- SMALL ACCOUNT ($667) VIABILITY ---")
for ticker in ["F", "SOFI", "PLTR"]:
    latest = float(data[ticker]["close"].iloc[-1])
    can_csp = latest * 100 <= 667
    print(f"  {ticker}: ${latest:.2f}/share → CSP collateral ${latest*100:.0f} → {'FEASIBLE' if can_csp else 'TOO EXPENSIVE'}")

spy_latest = float(data["SPY"]["close"].iloc[-1])
print(f"  SPY: ${spy_latest:.2f}/share → CSP collateral ${spy_latest*100:.0f} → TOO EXPENSIVE (need spreads)")
print(f"  → Put Credit Spread on SPY: max risk = width × 100 = ~${spy_latest * 0.002 * 100:.0f} per spread → FEASIBLE")

# ─── Save results ────────────────────────────────────────────────────
output = {
    "backtest_date": datetime.now().isoformat(),
    "period": f"{START} to {END}",
    "account_size": 667,
    "methodology": "Simulated options using yfinance + VIX as IV proxy. Not real options chain data.",
    "variants": results,
    "small_account_notes": {
        "feasible_csp_tickers": [t for t in ["F", "SOFI", "PLTR"] if float(data[t]["close"].iloc[-1]) * 100 <= 667],
        "spy_requires_spreads": True,
        "recommendation": "For $667: Put credit spreads on SPY/IWM (defined risk) or CSP on stocks under $6.67/share only.",
    },
}

out_path = "/home/jupiter/Lvl3Quant/data/wheel_strategy_results.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2, default=str)

print(f"\nResults saved to {out_path}")

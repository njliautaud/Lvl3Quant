#!/usr/bin/env python3
"""
Risk Parity Gold Decomposition Analysis
Determines if the Weekly Risk Parity edge is genuinely from rebalancing/vol-targeting
or just from being long gold.
"""

import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

# --- Config ---
INITIAL_CAPITAL = 645.0
SLIPPAGE_PCT = 0.0002
VOL_LOOKBACK = 20
TARGET_VOL = 0.08
REBAL_DAY = 4  # Friday
ANNUALIZE = 252
START = "2022-01-01"
END = "2026-07-29"
ASSETS_FULL = ["GLD", "TLT", "UUP"]

# --- Download Data ---
print("Downloading data...")
data = yf.download(ASSETS_FULL + ["QQQ"], start=START, end=END, auto_adjust=True, progress=False)

# Handle multi-level columns from yfinance
if isinstance(data.columns, pd.MultiIndex):
    prices = data["Close"].copy()
else:
    prices = data[["Close"]].copy()

prices = prices.dropna()
returns = prices.pct_change().dropna()
print(f"Data: {returns.index[0].date()} to {returns.index[-1].date()}, {len(returns)} days")


def compute_metrics(equity_curve):
    """Compute Sharpe, Sortino, total return, max DD, annual vol from equity curve."""
    rets = equity_curve.pct_change().dropna()
    if len(rets) < 10:
        return {"sharpe": 0, "sortino": 0, "total_return": 0, "max_drawdown": 0, "annual_vol": 0}
    mu = rets.mean() * ANNUALIZE
    sigma = rets.std() * np.sqrt(ANNUALIZE)
    downside = rets[rets < 0].std() * np.sqrt(ANNUALIZE)
    sharpe = mu / sigma if sigma > 0 else 0
    sortino = mu / downside if downside > 0 else 0
    total_ret = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
    drawdown = equity_curve / equity_curve.cummax() - 1
    max_dd = drawdown.min()
    return {
        "sharpe": round(float(sharpe), 4),
        "sortino": round(float(sortino), 4),
        "total_return": round(float(total_ret), 4),
        "max_drawdown": round(float(max_dd), 4),
        "annual_vol": round(float(sigma), 4),
    }


def run_risk_parity(asset_list, returns_df, prices_df, label=""):
    """Run inverse-vol risk parity with weekly rebalancing."""
    rets = returns_df[asset_list].copy()
    n_assets = len(asset_list)

    equity = INITIAL_CAPITAL
    equity_curve = [equity]
    dates = [rets.index[VOL_LOOKBACK]]
    weights = np.ones(n_assets) / n_assets

    # Track per-asset contribution
    asset_contribs = {a: 0.0 for a in asset_list}

    for i in range(VOL_LOOKBACK, len(rets)):
        dt = rets.index[i]
        day_rets = rets.iloc[i].values

        # Apply slippage on rebalance days
        is_rebal = dt.weekday() == REBAL_DAY
        if is_rebal and i > VOL_LOOKBACK:
            lookback_rets = rets.iloc[i - VOL_LOOKBACK:i]
            vols = lookback_rets.std() * np.sqrt(ANNUALIZE)
            inv_vol = 1.0 / np.maximum(vols.values, 1e-8)
            new_weights = inv_vol / inv_vol.sum()

            # Portfolio vol scaling
            port_vol = np.sqrt(np.dot(new_weights, np.dot(lookback_rets.cov().values * ANNUALIZE, new_weights)))
            if port_vol > 0:
                scale = TARGET_VOL / port_vol
            else:
                scale = 1.0
            new_weights = new_weights * scale
            total_w = new_weights.sum()
            if total_w > 1.0:
                new_weights = new_weights / total_w  # cap leverage at 1x

            # Slippage from weight changes
            turnover = np.abs(new_weights - weights).sum()
            slippage_cost = equity * turnover * SLIPPAGE_PCT
            equity -= slippage_cost
            weights = new_weights

        # Daily return
        port_ret = np.dot(weights, day_rets)

        # Track per-asset contributions
        for j, a in enumerate(asset_list):
            asset_contribs[a] += weights[j] * day_rets[j] * equity

        equity *= (1 + port_ret)
        equity_curve.append(equity)
        dates.append(dt)

    eq_series = pd.Series(equity_curve, index=dates)
    metrics = compute_metrics(eq_series)

    # Compute attribution percentages
    total_contrib = sum(asset_contribs.values())
    attribution = {}
    for a in asset_list:
        attribution[a] = round(asset_contribs[a] / total_contrib * 100, 2) if total_contrib != 0 else 0

    return eq_series, metrics, attribution


def run_buy_and_hold(ticker, returns_df):
    """Simple buy-and-hold of a single asset."""
    rets = returns_df[ticker].dropna()
    equity = INITIAL_CAPITAL
    equity_curve = [equity]
    dates = [rets.index[0]]
    for i in range(1, len(rets)):
        equity *= (1 + rets.iloc[i])
        equity_curve.append(equity)
        dates.append(rets.index[i])
    eq_series = pd.Series(equity_curve, index=dates)
    return eq_series, compute_metrics(eq_series)


def run_vol_targeted_buy_and_hold(ticker, returns_df):
    """Buy-and-hold a single asset but vol-target to 8%."""
    rets = returns_df[ticker].dropna()
    equity = INITIAL_CAPITAL
    equity_curve = [equity]
    dates = [rets.index[VOL_LOOKBACK]]
    weight = 1.0

    for i in range(VOL_LOOKBACK, len(rets)):
        dt = rets.index[i]
        is_rebal = dt.weekday() == REBAL_DAY
        if is_rebal and i > VOL_LOOKBACK:
            lookback = rets.iloc[i - VOL_LOOKBACK:i]
            vol = lookback.std() * np.sqrt(ANNUALIZE)
            new_weight = min(TARGET_VOL / vol, 1.0) if vol > 0 else 1.0
            turnover = abs(new_weight - weight)
            equity -= equity * turnover * SLIPPAGE_PCT
            weight = new_weight

        equity *= (1 + weight * rets.iloc[i])
        equity_curve.append(equity)
        dates.append(dt)

    eq_series = pd.Series(equity_curve, index=dates)
    return eq_series, compute_metrics(eq_series)


# ========================================
# 1. Full risk parity (GLD+TLT+UUP)
# ========================================
print("\n=== Full Risk Parity (GLD+TLT+UUP) ===")
eq_full, metrics_full, attrib_full = run_risk_parity(ASSETS_FULL, returns, prices)
print(f"  Sharpe: {metrics_full['sharpe']}, Return: {metrics_full['total_return']:.2%}")
print(f"  Attribution: {attrib_full}")

# ========================================
# 2. Buy-and-hold each asset
# ========================================
print("\n=== Buy-and-Hold Individual Assets ===")
bah_metrics = {}
for ticker in ASSETS_FULL + ["QQQ"]:
    _, m = run_buy_and_hold(ticker, returns)
    bah_metrics[ticker] = m
    print(f"  {ticker}: Sharpe={m['sharpe']}, Return={m['total_return']:.2%}, MaxDD={m['max_drawdown']:.2%}")

# ========================================
# 3. GLD-only at 8% target vol
# ========================================
print("\n=== GLD Vol-Targeted (8% target vol) ===")
_, metrics_gld_vol = run_vol_targeted_buy_and_hold("GLD", returns)
print(f"  Sharpe: {metrics_gld_vol['sharpe']}, Return: {metrics_gld_vol['total_return']:.2%}")

# ========================================
# 4. Risk Parity WITHOUT GLD (TLT+UUP only)
# ========================================
print("\n=== Risk Parity WITHOUT GLD (TLT+UUP) ===")
eq_no_gld, metrics_no_gld, attrib_no_gld = run_risk_parity(["TLT", "UUP"], returns, prices)
print(f"  Sharpe: {metrics_no_gld['sharpe']}, Return: {metrics_no_gld['total_return']:.2%}")

# ========================================
# 5. Risk Parity WITHOUT TLT (GLD+UUP only)
# ========================================
print("\n=== Risk Parity WITHOUT TLT (GLD+UUP) ===")
eq_no_tlt, metrics_no_tlt, attrib_no_tlt = run_risk_parity(["GLD", "UUP"], returns, prices)
print(f"  Sharpe: {metrics_no_tlt['sharpe']}, Return: {metrics_no_tlt['total_return']:.2%}")

# ========================================
# Build results
# ========================================
results = {
    "analysis": "Risk Parity Gold Decomposition",
    "question": "Is the edge from rebalancing/vol-targeting or just from being long gold?",
    "data_period": f"{returns.index[0].date()} to {returns.index[-1].date()}",
    "config": {
        "initial_capital": INITIAL_CAPITAL,
        "slippage_pct": SLIPPAGE_PCT,
        "vol_lookback": VOL_LOOKBACK,
        "target_vol": TARGET_VOL,
        "rebal_day": "Friday",
        "annualize": ANNUALIZE,
    },
    "full_risk_parity_GLD_TLT_UUP": {
        "metrics": metrics_full,
        "return_attribution_pct": attrib_full,
    },
    "buy_and_hold_comparison": {
        ticker: bah_metrics[ticker] for ticker in ASSETS_FULL + ["QQQ"]
    },
    "gld_vol_targeted_8pct": {
        "description": "GLD-only buy-and-hold at 8% target vol (weekly rebal)",
        "metrics": metrics_gld_vol,
    },
    "risk_parity_WITHOUT_GLD": {
        "description": "TLT+UUP only risk parity",
        "metrics": metrics_no_gld,
        "attribution_pct": attrib_no_gld,
    },
    "risk_parity_WITHOUT_TLT": {
        "description": "GLD+UUP only risk parity (best combo from param grid)",
        "metrics": metrics_no_tlt,
        "attribution_pct": attrib_no_tlt,
    },
    "conclusions": {},
}

# Derive conclusions
gld_share = attrib_full.get("GLD", 0)
rp_sharpe = metrics_full["sharpe"]
gld_bah_sharpe = bah_metrics["GLD"]["sharpe"]
gld_vol_sharpe = metrics_gld_vol["sharpe"]
no_gld_sharpe = metrics_no_gld["sharpe"]
no_tlt_sharpe = metrics_no_tlt["sharpe"]

conclusions = {
    "gld_return_attribution_pct": gld_share,
    "gld_dominates_returns": gld_share > 70,
    "risk_parity_sharpe": rp_sharpe,
    "gld_buy_hold_sharpe": gld_bah_sharpe,
    "gld_vol_targeted_sharpe": gld_vol_sharpe,
    "rp_beats_gld_vol_targeted": rp_sharpe > gld_vol_sharpe,
    "rp_without_gld_sharpe": no_gld_sharpe,
    "rp_without_gld_works": no_gld_sharpe > 0.3,
    "rp_without_tlt_sharpe": no_tlt_sharpe,
    "rp_without_tlt_better": no_tlt_sharpe > rp_sharpe,
    "vol_targeting_adds_value": gld_vol_sharpe > gld_bah_sharpe,
    "rebalancing_adds_value": rp_sharpe > gld_vol_sharpe,
    "verdict": "",
}

# Build verdict
if gld_share > 70 and not conclusions["rebalancing_adds_value"]:
    conclusions["verdict"] = "EDGE IS JUST LEVERED GOLD. Risk parity adds no value over vol-targeted GLD."
elif gld_share > 70 and conclusions["rebalancing_adds_value"]:
    conclusions["verdict"] = (
        f"GLD dominates ({gld_share:.0f}% of returns) but rebalancing/vol-targeting adds incremental Sharpe "
        f"({rp_sharpe:.2f} vs {gld_vol_sharpe:.2f} for vol-targeted GLD). "
        f"Dropping TLT improves Sharpe to {no_tlt_sharpe:.2f}. "
        f"Without GLD, strategy barely works (Sharpe {no_gld_sharpe:.2f})."
    )
elif gld_share <= 70:
    conclusions["verdict"] = (
        f"Returns are diversified across assets (GLD={gld_share:.0f}%). "
        f"Rebalancing genuinely adds value."
    )

results["conclusions"] = conclusions

# Print summary
print("\n" + "=" * 60)
print("VERDICT:", conclusions["verdict"])
print("=" * 60)

# Save
output_path = "/home/jupiter/Lvl3Quant/data/riskparity_gold_decomposition.json"
with open(output_path, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to {output_path}")

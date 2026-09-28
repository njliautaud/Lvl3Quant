#!/usr/bin/env python3
"""
Best Portfolio Combinations v1
==============================
Monte Carlo simulation of portfolio combinations using validated strategy metrics.
Ranks by Calmar ratio, consistency, and regime-agnosticism.

Validated strategies from SESSION_STATE.md and research findings.
"""

import json
import datetime as dt
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import skewnorm

np.random.seed(42)

# ---------------------------------------------------------------------------
# Strategy definitions with validated metrics
# ---------------------------------------------------------------------------

STRATEGIES = {
    # GROWTH — Stock-picking / contrarian
    "PostEarningsDrift": {
        "category": "growth",
        "sharpe": 2.11,
        "win_rate": 0.60,
        "avg_return_per_trade": 0.028,
        "trades_per_year": 270,
        "perm_p": 0.000,
        "description": "Post-Earnings Drift Contrarian",
        "ann_return": None,  # derived from trade-level
        "max_dd": None,
        "sortino": None,
        "regime_gap": 0.20,  # estimated moderate
    },
    "MomentumExhaustion": {
        "category": "growth",
        "sharpe": 2.25,
        "win_rate": 0.64,
        "avg_return_per_trade": 0.045,
        "trades_per_year": 17,
        "perm_p": 0.000,
        "description": "Momentum Exhaustion",
        "ann_return": None,
        "max_dd": None,
        "sortino": None,
        "regime_gap": 0.15,
    },
    "SmartMoney": {
        "category": "growth",
        "sharpe": 1.61,
        "win_rate": 0.57,
        "avg_return_per_trade": 0.035,
        "trades_per_year": 17,
        "perm_p": 0.000,
        "description": "Smart Money Accumulation",
        "ann_return": None,
        "max_dd": None,
        "sortino": None,
        "regime_gap": 0.25,
    },
    "PriceVolDivergence": {
        "category": "growth",
        "sharpe": 1.23,
        "win_rate": 0.58,
        "avg_return_per_trade": 0.015,
        "trades_per_year": 330,
        "perm_p": 0.023,
        "description": "Price-Volume Divergence",
        "ann_return": None,
        "max_dd": None,
        "sortino": None,
        "regime_gap": 0.30,
    },
    "SkewnessPremium": {
        "category": "growth",
        "sharpe": 1.02,
        "win_rate": 0.58,
        "avg_return_per_trade": 0.008,
        "trades_per_year": 525,  # midpoint of 350-700
        "perm_p": 0.000,
        "description": "Skewness Premium",
        "ann_return": None,
        "max_dd": None,
        "sortino": None,
        "regime_gap": 0.25,
    },
    "VolCrushReversal": {
        "category": "growth",
        "sharpe": 0.96,
        "win_rate": 0.59,
        "avg_return_per_trade": 0.006,
        "trades_per_year": 650,
        "perm_p": 0.000,
        "description": "Vol Crush Reversal",
        "ann_return": None,
        "max_dd": None,
        "sortino": None,
        "regime_gap": 0.30,
    },
    "GapFade": {
        "category": "growth",
        "sharpe": 0.71,
        "win_rate": 0.54,
        "avg_return_per_trade": 0.003,
        "trades_per_year": 1700,
        "perm_p": 0.000,
        "description": "Gap Fade (moderate)",
        "ann_return": None,
        "max_dd": None,
        "sortino": None,
        "regime_gap": 0.35,
    },

    # ETF STRATEGIES
    "CTATrend": {
        "category": "etf",
        "sharpe": 2.92,
        "win_rate": 0.55,
        "avg_return_per_trade": None,
        "trades_per_year": None,
        "perm_p": 0.000,
        "description": "CTA Trend (8 assets)",
        "ann_return": 0.198,
        "max_dd": 0.046,
        "sortino": 4.27,
        "regime_gap": 0.10,
    },
    "SectorRotation": {
        "category": "etf",
        "sharpe": 2.47,
        "win_rate": 0.58,
        "avg_return_per_trade": None,
        "trades_per_year": None,
        "perm_p": 0.000,
        "description": "Sector Rotation",
        "ann_return": 0.206,
        "max_dd": 0.074,
        "sortino": 3.40,
        "regime_gap": 0.15,
    },
    "CTASectorCombo": {
        "category": "etf",
        "sharpe": 3.06,
        "win_rate": 0.57,
        "avg_return_per_trade": None,
        "trades_per_year": None,
        "perm_p": 0.000,
        "description": "Combined 50/50 CTA+Sector",
        "ann_return": 0.203,
        "max_dd": 0.047,
        "sortino": 4.50,
        "regime_gap": 0.08,
    },
    "ETFRotationV2": {
        "category": "etf",
        "sharpe": 1.90,
        "win_rate": 0.55,
        "avg_return_per_trade": None,
        "trades_per_year": None,
        "perm_p": 0.000,
        "description": "ETF Rotation v2 (yield curve)",
        "ann_return": 0.145,
        "max_dd": 0.065,
        "sortino": 2.80,
        "regime_gap": 0.04,
    },
    "ETFRotationV3": {
        "category": "etf",
        "sharpe": 2.39,
        "win_rate": 0.57,
        "avg_return_per_trade": None,
        "trades_per_year": None,
        "perm_p": 0.000,
        "description": "ETF Rotation v3 (quality)",
        "ann_return": 0.180,
        "max_dd": 0.060,
        "sortino": 3.50,
        "regime_gap": 0.19,
    },

    # INCOME — Premium selling
    "IronCondor7d": {
        "category": "income",
        "sharpe": 2.50,
        "win_rate": 0.78,
        "avg_return_per_trade": 0.005,
        "trades_per_year": 52,
        "perm_p": 0.000,
        "description": "Iron Condor 7-day",
        "ann_return": 0.25,
        "max_dd": 0.04,
        "sortino": 3.80,
        "regime_gap": 0.20,
    },
    "BPSConservative": {
        "category": "income",
        "sharpe": 1.80,
        "win_rate": 0.82,
        "avg_return_per_trade": 0.003,
        "trades_per_year": 52,
        "perm_p": 0.000,
        "description": "BPS Conservative",
        "ann_return": 0.15,
        "max_dd": 0.10,
        "sortino": 2.50,
        "regime_gap": 0.25,
    },
    "JadeLizard5pos": {
        "category": "income",
        "sharpe": 1.97,
        "win_rate": 0.72,
        "avg_return_per_trade": 0.006,
        "trades_per_year": 48,
        "perm_p": 0.000,
        "description": "Jade Lizard (5 positions)",
        "ann_return": 0.12,
        "max_dd": 0.13,
        "sortino": 2.80,
        "regime_gap": 0.30,
    },
    "JadeLizard25pos": {
        "category": "income",
        "sharpe": 1.97,
        "win_rate": 0.72,
        "avg_return_per_trade": 0.006,
        "trades_per_year": 48,
        "perm_p": 0.000,
        "description": "Jade Lizard (25 positions, aggressive)",
        "ann_return": 0.26,
        "max_dd": 0.22,
        "sortino": 2.80,
        "regime_gap": 0.30,
    },
}

# ---------------------------------------------------------------------------
# Helper: Generate synthetic daily returns for a strategy
# ---------------------------------------------------------------------------

def generate_daily_returns(sharpe, ann_return=None, max_dd=None, win_rate=0.55,
                           n_days=252*7, regime_gap=0.20, n_sims=1):
    """
    Generate synthetic daily returns matching strategy characteristics.

    Uses a mixture model:
    - Win days: positive returns drawn from lognormal
    - Loss days: negative returns drawn from lognormal (fatter tail)
    - Regime modulation: green days get boosted, red days get penalized

    Returns shape: (n_sims, n_days)
    """
    trading_days_per_year = 252

    # Derive annualized return from Sharpe if not given
    if ann_return is None:
        # Sharpe = excess_return / vol; assume vol ~16% for stock strategies
        vol = 0.16
        ann_return = sharpe * vol

    # Daily parameters
    daily_return = ann_return / trading_days_per_year
    daily_vol = ann_return / sharpe / np.sqrt(trading_days_per_year) if sharpe > 0 else 0.01

    all_returns = np.zeros((n_sims, n_days))

    for sim in range(n_sims):
        # Generate regime sequence (roughly 60% green, 40% red, with persistence)
        regime = np.ones(n_days)  # 1 = green, -1 = red
        state = 1
        for d in range(n_days):
            # Regime switching with persistence
            if state == 1 and np.random.random() < 0.005:  # ~1/200 days switch to red
                state = -1
            elif state == -1 and np.random.random() < 0.008:  # ~1/125 days switch to green
                state = 1
            regime[d] = state

        # Base returns
        wins = np.random.random(n_days) < win_rate

        # Win magnitude: right-skewed
        win_mag = np.abs(np.random.lognormal(
            mean=np.log(daily_return * 1.5 + 0.001), sigma=0.5, size=n_days
        ))
        # Loss magnitude: slightly fatter tail
        loss_mag = np.abs(np.random.lognormal(
            mean=np.log(daily_return * 1.2 + 0.001), sigma=0.6, size=n_days
        ))

        returns = np.where(wins, win_mag, -loss_mag)

        # Apply regime modulation
        green_boost = 1 + regime_gap / 2
        red_penalty = 1 - regime_gap / 2
        regime_mult = np.where(regime > 0, green_boost, red_penalty)
        returns = returns * regime_mult

        # Scale to match target Sharpe
        current_sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        if current_sharpe > 0:
            # Adjust mean to hit target Sharpe while preserving vol
            target_daily_mean = sharpe * returns.std() / np.sqrt(252)
            returns = returns - returns.mean() + target_daily_mean

        all_returns[sim] = returns

    return all_returns


def compute_metrics(daily_returns):
    """Compute portfolio metrics from daily return series."""
    cumulative = np.cumprod(1 + daily_returns)
    total_return = cumulative[-1] / cumulative[0] - 1
    n_years = len(daily_returns) / 252
    cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

    ann_vol = np.std(daily_returns) * np.sqrt(252)
    sharpe = cagr / ann_vol if ann_vol > 0 else 0

    # Sortino
    downside = daily_returns[daily_returns < 0]
    downside_vol = np.std(downside) * np.sqrt(252) if len(downside) > 0 else 0.001
    sortino = cagr / downside_vol if downside_vol > 0 else 0

    # Max drawdown
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    max_dd = np.min(drawdowns)

    # Calmar
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Monthly returns for consistency
    n_months = int(len(daily_returns) / 21)
    monthly_returns = []
    for m in range(n_months):
        chunk = daily_returns[m*21:(m+1)*21]
        if len(chunk) > 0:
            monthly_returns.append(np.prod(1 + chunk) - 1)
    monthly_returns = np.array(monthly_returns)
    pct_months_positive = np.mean(monthly_returns > 0) if len(monthly_returns) > 0 else 0

    # Quarterly
    n_quarters = int(len(daily_returns) / 63)
    quarterly_returns = []
    for q in range(n_quarters):
        chunk = daily_returns[q*63:(q+1)*63]
        if len(chunk) > 0:
            quarterly_returns.append(np.prod(1 + chunk) - 1)
    quarterly_returns = np.array(quarterly_returns)
    pct_quarters_positive = np.mean(quarterly_returns > 0) if len(quarterly_returns) > 0 else 0

    # Regime analysis: split by regime periods
    # First half vs second half as proxy (imperfect but usable)
    mid = len(daily_returns) // 2
    sharpe_h1 = np.mean(daily_returns[:mid]) / np.std(daily_returns[:mid]) * np.sqrt(252) if np.std(daily_returns[:mid]) > 0 else 0
    sharpe_h2 = np.mean(daily_returns[mid:]) / np.std(daily_returns[mid:]) * np.sqrt(252) if np.std(daily_returns[mid:]) > 0 else 0
    regime_stability = 1 - abs(sharpe_h1 - sharpe_h2) / max(abs(sharpe_h1), abs(sharpe_h2), 0.01)

    # Profit factor
    gross_profit = np.sum(daily_returns[daily_returns > 0])
    gross_loss = abs(np.sum(daily_returns[daily_returns < 0]))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999

    return {
        "cagr": round(cagr, 4),
        "ann_vol": round(ann_vol, 4),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "max_dd": round(max_dd, 4),
        "calmar": round(calmar, 2),
        "profit_factor": round(profit_factor, 2),
        "pct_months_positive": round(pct_months_positive, 4),
        "pct_quarters_positive": round(pct_quarters_positive, 4),
        "regime_stability": round(regime_stability, 4),
    }


def regime_split_metrics(daily_returns, regime_labels):
    """Compute metrics separately for green and red regimes."""
    green_mask = regime_labels > 0
    red_mask = regime_labels < 0

    green_rets = daily_returns[green_mask]
    red_rets = daily_returns[red_mask]

    green_sharpe = np.mean(green_rets) / np.std(green_rets) * np.sqrt(252) if len(green_rets) > 10 and np.std(green_rets) > 0 else 0
    red_sharpe = np.mean(red_rets) / np.std(red_rets) * np.sqrt(252) if len(red_rets) > 10 and np.std(red_rets) > 0 else 0

    gap = abs(green_sharpe - red_sharpe) / max(abs(green_sharpe), abs(red_sharpe), 0.01)

    return {
        "green_sharpe": round(green_sharpe, 2),
        "red_sharpe": round(red_sharpe, 2),
        "regime_gap": round(gap, 4),
        "passes_r1": gap < 0.50,
    }


# ---------------------------------------------------------------------------
# Portfolio combination engine
# ---------------------------------------------------------------------------

N_DAYS = 252 * 7  # 7 years of trading days
N_SIMS = 50       # Monte Carlo simulations per combination


def generate_all_strategy_returns():
    """Generate return matrices for all strategies."""
    strategy_returns = {}
    strategy_regimes = {}

    for name, s in STRATEGIES.items():
        # Generate regime sequence (shared across sims for consistency)
        regime = np.ones(N_DAYS)
        state = 1
        for d in range(N_DAYS):
            if state == 1 and np.random.random() < 0.005:
                state = -1
            elif state == -1 and np.random.random() < 0.008:
                state = 1
            regime[d] = state

        returns = generate_daily_returns(
            sharpe=s["sharpe"],
            ann_return=s.get("ann_return"),
            max_dd=s.get("max_dd"),
            win_rate=s.get("win_rate", 0.55),
            n_days=N_DAYS,
            regime_gap=s.get("regime_gap", 0.20),
            n_sims=N_SIMS,
        )
        strategy_returns[name] = returns
        strategy_regimes[name] = regime

    return strategy_returns, strategy_regimes


def combine_strategies(strategy_returns, strategy_regimes, names, weights):
    """
    Combine strategy returns with given weights.
    Returns: (n_sims, n_days) combined daily returns.
    """
    combined = np.zeros_like(strategy_returns[names[0]])
    for name, w in zip(names, weights):
        combined += w * strategy_returns[name]
    return combined


def risk_parity_weights(strategy_returns, names):
    """
    Compute risk-parity weights: inverse-volatility.
    """
    vols = []
    for name in names:
        # Average vol across sims
        vol = np.mean([np.std(strategy_returns[name][s]) for s in range(strategy_returns[name].shape[0])])
        vols.append(vol)
    vols = np.array(vols)
    inv_vol = 1.0 / (vols + 1e-10)
    weights = inv_vol / inv_vol.sum()
    return weights


def sharpe_optimal_weights(strategy_returns, names, n_random=5000):
    """
    Find weights that maximize Sharpe via random search.
    """
    n = len(names)
    best_sharpe = -999
    best_weights = np.ones(n) / n

    for _ in range(n_random):
        w = np.random.dirichlet(np.ones(n))
        combined = np.zeros_like(strategy_returns[names[0]])
        for name, wi in zip(names, w):
            combined += wi * strategy_returns[name]

        # Average Sharpe across sims
        sharpes = []
        for s in range(combined.shape[0]):
            mean_r = np.mean(combined[s])
            std_r = np.std(combined[s])
            if std_r > 0:
                sharpes.append(mean_r / std_r * np.sqrt(252))
        avg_sharpe = np.mean(sharpes)

        if avg_sharpe > best_sharpe:
            best_sharpe = avg_sharpe
            best_weights = w

    return best_weights


def evaluate_combination(strategy_returns, strategy_regimes, names, weights, label=""):
    """Evaluate a portfolio combination across Monte Carlo sims."""
    combined = combine_strategies(strategy_returns, strategy_regimes, names, weights)

    all_metrics = []
    all_regime = []

    for s in range(N_SIMS):
        metrics = compute_metrics(combined[s])
        all_metrics.append(metrics)

        # Use first strategy's regime for regime split (they share similar regime structure)
        regime = strategy_regimes[names[0]]
        rm = regime_split_metrics(combined[s], regime)
        all_regime.append(rm)

    # Average metrics across sims
    avg_metrics = {}
    for key in all_metrics[0]:
        vals = [m[key] for m in all_metrics]
        avg_metrics[key] = round(np.mean(vals), 4)
        avg_metrics[f"{key}_p10"] = round(np.percentile(vals, 10), 4)
        avg_metrics[f"{key}_p90"] = round(np.percentile(vals, 90), 4)

    avg_regime = {
        "green_sharpe": round(np.mean([r["green_sharpe"] for r in all_regime]), 2),
        "red_sharpe": round(np.mean([r["red_sharpe"] for r in all_regime]), 2),
        "regime_gap": round(np.mean([r["regime_gap"] for r in all_regime]), 4),
        "pct_pass_r1": round(np.mean([r["passes_r1"] for r in all_regime]), 4),
    }

    return {
        "label": label,
        "strategies": list(names),
        "weights": {n: round(w, 4) for n, w in zip(names, weights)},
        "avg_metrics": avg_metrics,
        "regime_analysis": avg_regime,
        "n_strategies": len(names),
    }


def score_combination(result):
    """
    Score a portfolio combination. Primary: Calmar. Secondary: consistency + regime.
    """
    m = result["avg_metrics"]
    r = result["regime_analysis"]

    calmar = m.get("calmar", 0)
    consistency = m.get("pct_months_positive", 0)
    regime_pass = r.get("pct_pass_r1", 0)
    cagr = m.get("cagr", 0)
    sharpe = m.get("sharpe", 0)

    # Composite score: 40% Calmar, 25% consistency, 20% regime, 15% Sharpe
    score = (0.40 * min(calmar, 10) / 10 +
             0.25 * consistency +
             0.20 * regime_pass +
             0.15 * min(sharpe, 5) / 5)

    return round(score, 4)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("BEST PORTFOLIO COMBINATIONS v1")
    print(f"Monte Carlo: {N_SIMS} sims x {N_DAYS} days ({N_DAYS/252:.0f} years)")
    print(f"Date: {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 80)

    # Generate returns
    print("\nGenerating synthetic returns for all strategies...")
    strat_returns, strat_regimes = generate_all_strategy_returns()

    # Define category groups
    growth_names = [n for n, s in STRATEGIES.items() if s["category"] == "growth"]
    etf_names = [n for n, s in STRATEGIES.items() if s["category"] == "etf"]
    income_names = [n for n, s in STRATEGIES.items() if s["category"] == "income"]

    growth_all = growth_names + etf_names  # growth portfolio includes ETFs

    results = {"growth": [], "income": [], "combined": []}

    # -----------------------------------------------------------------------
    # GROWTH PORTFOLIO COMBINATIONS
    # -----------------------------------------------------------------------
    print("\n--- GROWTH PORTFOLIO ---")
    print(f"Candidate strategies: {len(growth_all)}")

    # Test combinations of 3-6 strategies (cap at 6 to keep compute manageable)
    for size in range(3, min(len(growth_all) + 1, 7)):
        combos_list = list(combinations(growth_all, size))
        # Sample if too many combos
        if len(combos_list) > 100:
            np.random.shuffle(combos_list)
            combos_list = combos_list[:100]
        for combo in combos_list:
            combo = list(combo)

            # Equal weight
            ew = np.ones(len(combo)) / len(combo)
            res = evaluate_combination(strat_returns, strat_regimes, combo, ew,
                                       label=f"EW({len(combo)})")
            res["score"] = score_combination(res)
            res["weight_method"] = "equal"
            results["growth"].append(res)

            # Risk parity
            rp = risk_parity_weights(strat_returns, combo)
            res = evaluate_combination(strat_returns, strat_regimes, combo, rp,
                                       label=f"RP({len(combo)})")
            res["score"] = score_combination(res)
            res["weight_method"] = "risk_parity"
            results["growth"].append(res)

    # Top combos get Sharpe-optimized weights too
    results["growth"].sort(key=lambda x: x["score"], reverse=True)
    top_growth_combos = [r["strategies"] for r in results["growth"][:10]]

    for combo in top_growth_combos:
        opt_w = sharpe_optimal_weights(strat_returns, combo, n_random=1000)
        res = evaluate_combination(strat_returns, strat_regimes, combo, opt_w,
                                   label=f"OPT({len(combo)})")
        res["score"] = score_combination(res)
        res["weight_method"] = "sharpe_optimized"
        results["growth"].append(res)

    results["growth"].sort(key=lambda x: x["score"], reverse=True)

    # -----------------------------------------------------------------------
    # INCOME PORTFOLIO COMBINATIONS
    # -----------------------------------------------------------------------
    print("--- INCOME PORTFOLIO ---")
    print(f"Candidate strategies: {len(income_names)}")

    for size in range(2, len(income_names) + 1):
        for combo in combinations(income_names, size):
            combo = list(combo)

            ew = np.ones(len(combo)) / len(combo)
            res = evaluate_combination(strat_returns, strat_regimes, combo, ew,
                                       label=f"EW({len(combo)})")
            res["score"] = score_combination(res)
            res["weight_method"] = "equal"
            results["income"].append(res)

            rp = risk_parity_weights(strat_returns, combo)
            res = evaluate_combination(strat_returns, strat_regimes, combo, rp,
                                       label=f"RP({len(combo)})")
            res["score"] = score_combination(res)
            res["weight_method"] = "risk_parity"
            results["income"].append(res)

    # Optimized weights for top income combos
    results["income"].sort(key=lambda x: x["score"], reverse=True)
    for combo in [r["strategies"] for r in results["income"][:5]]:
        opt_w = sharpe_optimal_weights(strat_returns, combo, n_random=1000)
        res = evaluate_combination(strat_returns, strat_regimes, combo, opt_w,
                                   label=f"OPT({len(combo)})")
        res["score"] = score_combination(res)
        res["weight_method"] = "sharpe_optimized"
        results["income"].append(res)

    results["income"].sort(key=lambda x: x["score"], reverse=True)

    # -----------------------------------------------------------------------
    # COMBINED PORTFOLIO (growth + income)
    # -----------------------------------------------------------------------
    print("--- COMBINED PORTFOLIO ---")

    # Take top 3 growth and top 3 income combos, blend them
    top_growth = results["growth"][:3]
    top_income = results["income"][:3]

    growth_income_splits = [
        (0.70, 0.30, "70/30 Growth/Income"),
        (0.60, 0.40, "60/40 Growth/Income"),
        (0.50, 0.50, "50/50 Growth/Income"),
        (0.40, 0.60, "40/60 Growth/Income"),
    ]

    for gi, gr in enumerate(top_growth):
        for ii, ir in enumerate(top_income):
            for g_pct, i_pct, split_label in growth_income_splits:
                # Combine all strategies from both
                all_names = gr["strategies"] + ir["strategies"]
                # Remove duplicates
                seen = set()
                unique_names = []
                for n in all_names:
                    if n not in seen:
                        seen.add(n)
                        unique_names.append(n)

                # Scale weights
                g_weights = [gr["weights"][n] * g_pct for n in gr["strategies"]]
                i_weights = [ir["weights"][n] * i_pct for n in ir["strategies"] if n not in gr["strategies"]]

                combined_names = list(gr["strategies"]) + [n for n in ir["strategies"] if n not in gr["strategies"]]
                combined_weights = np.array(g_weights + i_weights)
                combined_weights = combined_weights / combined_weights.sum()  # normalize

                res = evaluate_combination(strat_returns, strat_regimes,
                                           combined_names, combined_weights,
                                           label=f"{split_label} G{gi+1}+I{ii+1}")
                res["score"] = score_combination(res)
                res["weight_method"] = f"blended_{split_label}"
                results["combined"].append(res)

    results["combined"].sort(key=lambda x: x["score"], reverse=True)

    # -----------------------------------------------------------------------
    # APPLY GATES
    # -----------------------------------------------------------------------
    print("\nApplying quality gates...")

    for cat in ["growth", "income", "combined"]:
        cagr_min = 0.10 if cat == "growth" else 0.15 if cat == "income" else 0.10
        filtered = []
        for r in results[cat]:
            m = r["avg_metrics"]
            regime = r["regime_analysis"]

            passes = True
            reasons = []

            if m["cagr"] < cagr_min:
                passes = False
                reasons.append(f"CAGR {m['cagr']:.1%} < {cagr_min:.0%}")

            if regime["pct_pass_r1"] < 0.50:
                passes = False
                reasons.append(f"R1 pass rate {regime['pct_pass_r1']:.0%} < 50%")

            if m["max_dd"] < -0.30:
                passes = False
                reasons.append(f"MaxDD {m['max_dd']:.1%} > 30%")

            r["passes_gates"] = passes
            r["gate_failures"] = reasons
            if passes:
                filtered.append(r)

        results[cat] = filtered if filtered else results[cat][:5]

    # -----------------------------------------------------------------------
    # PRINT TOP 5 FOR EACH CATEGORY
    # -----------------------------------------------------------------------

    output = {
        "analysis_date": dt.datetime.now().isoformat(),
        "n_simulations": N_SIMS,
        "n_days": N_DAYS,
        "categories": {},
    }

    for cat in ["growth", "income", "combined"]:
        top5 = results[cat][:5]
        output["categories"][cat] = top5

        print(f"\n{'='*80}")
        print(f"TOP 5 — {cat.upper()} PORTFOLIO")
        print(f"{'='*80}")

        for rank, r in enumerate(top5, 1):
            m = r["avg_metrics"]
            regime = r["regime_analysis"]

            print(f"\n  #{rank} | {r['label']} | Score: {r['score']:.4f}")
            print(f"  Strategies: {', '.join(r['strategies'])}")
            print(f"  Weights: {r['weight_method']}")
            for name, w in r["weights"].items():
                desc = STRATEGIES[name]["description"]
                print(f"    {desc}: {w:.1%}")
            print(f"  CAGR: {m['cagr']:.1%}  (p10: {m['cagr_p10']:.1%}, p90: {m['cagr_p90']:.1%})")
            print(f"  Sharpe: {m['sharpe']:.2f}  Sortino: {m['sortino']:.2f}")
            print(f"  MaxDD: {m['max_dd']:.1%}  (p10: {m['max_dd_p10']:.1%})")
            print(f"  Calmar: {m['calmar']:.2f}")
            print(f"  PF: {m['profit_factor']:.2f}")
            print(f"  Months positive: {m['pct_months_positive']:.0%}  Quarters positive: {m['pct_quarters_positive']:.0%}")
            print(f"  Regime: green Sharpe={regime['green_sharpe']:.2f}, red Sharpe={regime['red_sharpe']:.2f}, gap={regime['regime_gap']:.2f}")
            print(f"  R1 pass rate: {regime['pct_pass_r1']:.0%}")
            if not r.get("passes_gates", True):
                print(f"  *** GATE FAILURES: {', '.join(r.get('gate_failures', []))}")

    # -----------------------------------------------------------------------
    # PRACTICAL ALLOCATION RECOMMENDATIONS
    # -----------------------------------------------------------------------
    print(f"\n{'='*80}")
    print("PRACTICAL ALLOCATION RECOMMENDATIONS")
    print(f"{'='*80}")

    accounts = [
        ("Agentic Account", 677),
        ("Main Account", 8254),
        ("Theoretical $100K", 100000),
    ]

    allocation_recs = {}

    for acct_name, capital in accounts:
        print(f"\n  --- {acct_name} (${capital:,.0f}) ---")

        if capital < 1000:
            # Too small for options. ETF-only.
            best = results["growth"][0]
            print(f"  Recommendation: GROWTH ONLY (ETF strategies)")
            print(f"  Reason: Account too small for options premium selling.")
            print(f"  Best setup: {best['label']}")
            print(f"  Expected CAGR: {best['avg_metrics']['cagr']:.1%}")
            print(f"  Expected MaxDD: {best['avg_metrics']['max_dd']:.1%}")

            # Specific ETF allocation
            etf_strats = [n for n in best["strategies"] if STRATEGIES[n]["category"] == "etf"]
            if etf_strats:
                print(f"  Focus on: {', '.join(STRATEGIES[n]['description'] for n in etf_strats)}")
            else:
                print(f"  Focus on: CTA Trend + Sector Rotation (ETFs only)")

            allocation_recs[acct_name] = {
                "capital": capital,
                "recommendation": "ETF growth only",
                "expected_cagr": best["avg_metrics"]["cagr"],
                "expected_max_dd": best["avg_metrics"]["max_dd"],
            }

        elif capital < 25000:
            # Can do some options but limited by PDT and margin
            best_growth = results["growth"][0]
            best_income = results["income"][0] if results["income"] else None

            print(f"  Recommendation: 70% GROWTH / 30% INCOME (limited options)")
            print(f"  Constraint: PDT rule limits day-trading; focus on swing + weekly options.")

            growth_alloc = int(capital * 0.70)
            income_alloc = capital - growth_alloc

            print(f"  Growth allocation: ${growth_alloc:,.0f}")
            print(f"    Best setup: {best_growth['label']}")
            if best_income:
                print(f"  Income allocation: ${income_alloc:,.0f}")
                print(f"    Best setup: {best_income['label']}")
                print(f"    Note: Can run ~{income_alloc // 500:.0f} IC contracts or ~{income_alloc // 2000:.0f} BPS positions")

            combined_cagr = 0.70 * best_growth["avg_metrics"]["cagr"] + 0.30 * (best_income["avg_metrics"]["cagr"] if best_income else 0.10)
            print(f"  Blended expected CAGR: {combined_cagr:.1%}")

            allocation_recs[acct_name] = {
                "capital": capital,
                "recommendation": "70/30 growth/income",
                "expected_cagr": round(combined_cagr, 4),
            }

        else:
            # Full portfolio
            best_combined = results["combined"][0] if results["combined"] else results["growth"][0]

            print(f"  Recommendation: FULL COMBINED PORTFOLIO")
            print(f"  Best setup: {best_combined['label']}")
            print(f"  Expected CAGR: {best_combined['avg_metrics']['cagr']:.1%}")
            print(f"  Expected MaxDD: {best_combined['avg_metrics']['max_dd']:.1%}")
            print(f"  Expected Sharpe: {best_combined['avg_metrics']['sharpe']:.2f}")

            # Position sizing
            n_strats = best_combined["n_strategies"]
            per_strat = capital / n_strats
            print(f"  Per-strategy allocation: ~${per_strat:,.0f}")
            for name, w in best_combined["weights"].items():
                alloc = capital * w
                print(f"    {STRATEGIES[name]['description']}: ${alloc:,.0f} ({w:.0%})")

            allocation_recs[acct_name] = {
                "capital": capital,
                "recommendation": "full combined",
                "setup": best_combined["label"],
                "expected_cagr": best_combined["avg_metrics"]["cagr"],
                "expected_max_dd": best_combined["avg_metrics"]["max_dd"],
            }

    output["allocation_recommendations"] = allocation_recs

    # -----------------------------------------------------------------------
    # KEY TAKEAWAYS
    # -----------------------------------------------------------------------
    print(f"\n{'='*80}")
    print("KEY TAKEAWAYS")
    print(f"{'='*80}")

    best_calmar_growth = max(results["growth"][:5], key=lambda x: x["avg_metrics"]["calmar"])
    best_calmar_income = max(results["income"][:5], key=lambda x: x["avg_metrics"]["calmar"]) if results["income"] else None
    best_calmar_combined = max(results["combined"][:5], key=lambda x: x["avg_metrics"]["calmar"]) if results["combined"] else None

    print(f"\n  Best Calmar (Growth): {best_calmar_growth['avg_metrics']['calmar']:.2f}")
    print(f"    CAGR {best_calmar_growth['avg_metrics']['cagr']:.1%} / MaxDD {best_calmar_growth['avg_metrics']['max_dd']:.1%}")
    print(f"    Strategies: {', '.join(STRATEGIES[n]['description'] for n in best_calmar_growth['strategies'])}")

    if best_calmar_income:
        print(f"\n  Best Calmar (Income): {best_calmar_income['avg_metrics']['calmar']:.2f}")
        print(f"    CAGR {best_calmar_income['avg_metrics']['cagr']:.1%} / MaxDD {best_calmar_income['avg_metrics']['max_dd']:.1%}")

    if best_calmar_combined:
        print(f"\n  Best Calmar (Combined): {best_calmar_combined['avg_metrics']['calmar']:.2f}")
        print(f"    CAGR {best_calmar_combined['avg_metrics']['cagr']:.1%} / MaxDD {best_calmar_combined['avg_metrics']['max_dd']:.1%}")

    most_consistent = max(
        results["growth"][:5] + results["income"][:5] + results["combined"][:5],
        key=lambda x: x["avg_metrics"]["pct_months_positive"]
    )
    print(f"\n  Most Consistent (all categories): {most_consistent['avg_metrics']['pct_months_positive']:.0%} months positive")
    print(f"    {most_consistent['label']} — {', '.join(STRATEGIES[n]['description'] for n in most_consistent['strategies'])}")

    most_regime_agnostic = max(
        results["growth"][:5] + results["income"][:5] + results["combined"][:5],
        key=lambda x: x["regime_analysis"]["pct_pass_r1"]
    )
    print(f"\n  Most Regime-Agnostic: R1 pass rate {most_regime_agnostic['regime_analysis']['pct_pass_r1']:.0%}")
    print(f"    {most_regime_agnostic['label']}")

    output["key_takeaways"] = {
        "best_calmar_growth": best_calmar_growth["avg_metrics"]["calmar"],
        "best_calmar_income": best_calmar_income["avg_metrics"]["calmar"] if best_calmar_income else None,
        "most_consistent_pct_months_positive": most_consistent["avg_metrics"]["pct_months_positive"],
        "most_regime_agnostic_r1_pass": most_regime_agnostic["regime_analysis"]["pct_pass_r1"],
    }

    # Save results
    out_path = Path("/home/jupiter/Lvl3Quant/research/findings/best_portfolio_combinations_v1.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n\nResults saved to {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()

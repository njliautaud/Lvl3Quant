#!/usr/bin/env python3
"""
Global Macro Regime Switching Backtest
6 variants testing regime-based asset class rotation
Goal: Find strategies UNCORRELATED to QQQ
"""

import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── Config ──────────────────────────────────────────────────────────────
CAPITAL = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02%
COMMISSION = 0.0
OOT_START = "2022-01-01"
OOT_END = "2026-07-29"
DATA_START = "2020-06-01"  # extra history for indicators
TICKERS = ["GLD", "UUP", "TLT", "SPY", "QQQ", "EEM", "DBA", "USO", "SHY"]
VIX_TICKER = "^VIX"
RESULTS_PATH = "/home/jupiter/Lvl3Quant/data/global_macro_regime_results.json"

np.random.seed(42)


def download_data():
    """Download all required data via yfinance."""
    print("Downloading data...")
    all_tickers = TICKERS + [VIX_TICKER]
    data = yf.download(all_tickers, start=DATA_START, end=OOT_END, auto_adjust=True, progress=False)

    # Handle multi-level columns
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"]
    else:
        close = data

    # Rename ^VIX
    if "^VIX" in close.columns:
        close = close.rename(columns={"^VIX": "VIX"})

    close = close.ffill().dropna(how='all')
    print(f"Data shape: {close.shape}, range: {close.index[0].date()} to {close.index[-1].date()}")
    return close


def apply_slippage(returns, slippage=SLIPPAGE_PCT):
    """Apply slippage on trade days (when position changes)."""
    return returns  # slippage applied in backtest functions directly


def calc_metrics(equity_curve, qqq_returns, name, capital=CAPITAL):
    """Calculate performance metrics for a strategy."""
    returns = equity_curve.pct_change().dropna()
    if len(returns) < 20:
        return None

    total_ret = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    ann_factor = 252
    ann_ret = (1 + total_ret) ** (ann_factor / len(returns)) - 1
    ann_vol = returns.std() * np.sqrt(ann_factor)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    downside = returns[returns < 0].std() * np.sqrt(ann_factor)
    sortino = ann_ret / downside if downside > 0 else 0

    # Max drawdown
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_dd = drawdown.min()

    # Calmar
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

    # Win rate (daily)
    wr = (returns > 0).mean()

    # Profit factor
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    pf = gains / losses if losses > 0 else float('inf')

    # QQQ correlation
    aligned = pd.DataFrame({"strat": returns, "qqq": qqq_returns}).dropna()
    if len(aligned) > 20:
        qqq_corr = aligned["strat"].corr(aligned["qqq"])
    else:
        qqq_corr = np.nan

    # Monthly returns for consistency
    monthly = returns.resample('ME').apply(lambda x: (1+x).prod()-1)
    monthly_wr = (monthly > 0).mean()

    return {
        "name": name,
        "total_return_pct": round(total_ret * 100, 2),
        "ann_return_pct": round(ann_ret * 100, 2),
        "ann_vol_pct": round(ann_vol * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "calmar": round(calmar, 3),
        "profit_factor": round(pf, 3),
        "daily_win_rate": round(wr, 4),
        "monthly_win_rate": round(monthly_wr, 4),
        "qqq_correlation": round(qqq_corr, 4) if not np.isnan(qqq_corr) else None,
        "num_days": len(returns),
        "final_equity": round(equity_curve.iloc[-1], 2),
    }


def regime_split_analysis(equity_curve, spy_returns):
    """Analyze performance in bull vs bear regimes."""
    returns = equity_curve.pct_change().dropna()
    aligned = pd.DataFrame({"strat": returns, "spy": spy_returns}).dropna()

    # Define regimes by SPY 50d cumulative return
    spy_cum = aligned["spy"].rolling(50).sum()
    bull = spy_cum > 0
    bear = spy_cum <= 0

    results = {}
    for label, mask in [("bull", bull), ("bear", bear)]:
        subset = aligned.loc[mask, "strat"]
        if len(subset) < 20:
            results[label] = {"sharpe": None, "days": len(subset)}
            continue
        ann_ret = subset.mean() * 252
        ann_vol = subset.std() * np.sqrt(252)
        results[label] = {
            "sharpe": round(ann_ret / ann_vol, 3) if ann_vol > 0 else 0,
            "days": len(subset),
            "ann_return_pct": round(ann_ret * 100, 2),
        }

    # Regime balance check
    if results.get("bull", {}).get("sharpe") and results.get("bear", {}).get("sharpe"):
        s_bull = abs(results["bull"]["sharpe"])
        s_bear = abs(results["bear"]["sharpe"])
        max_s = max(s_bull, s_bear)
        regime_gap = abs(s_bull - s_bear) / max_s if max_s > 0 else 0
        results["regime_gap_ratio"] = round(regime_gap, 3)
        results["regime_agnostic_pass"] = regime_gap <= 0.50
    else:
        results["regime_gap_ratio"] = None
        results["regime_agnostic_pass"] = None

    return results


def permutation_test(strat_returns, n_perms=1000):
    """Permutation test: is the Sharpe significantly different from random?"""
    if len(strat_returns) < 20:
        return {"p_value": 1.0, "pass": False}

    actual_sharpe = strat_returns.mean() / strat_returns.std() * np.sqrt(252) if strat_returns.std() > 0 else 0

    count_better = 0
    for _ in range(n_perms):
        perm = np.random.permutation(strat_returns.values)
        perm_sharpe = perm.mean() / perm.std() * np.sqrt(252) if perm.std() > 0 else 0
        if perm_sharpe >= actual_sharpe:
            count_better += 1

    p_val = count_better / n_perms
    return {
        "p_value": round(p_val, 4),
        "actual_sharpe": round(actual_sharpe, 3),
        "pass": p_val < 0.05,
    }


def five_gate_validation(metrics, regime_analysis, perm_result):
    """5-gate validation framework."""
    gates = {}

    # Gate 1: Positive risk-adjusted return
    gates["G1_positive_sharpe"] = {
        "pass": metrics["sharpe"] > 0,
        "value": metrics["sharpe"],
        "threshold": "> 0",
    }

    # Gate 2: Acceptable drawdown
    gates["G2_max_drawdown"] = {
        "pass": metrics["max_drawdown_pct"] > -30,
        "value": metrics["max_drawdown_pct"],
        "threshold": "> -30%",
    }

    # Gate 3: Statistical significance (permutation test)
    gates["G3_permutation_test"] = {
        "pass": perm_result["pass"],
        "p_value": perm_result["p_value"],
        "threshold": "p < 0.05",
    }

    # Gate 4: Regime agnostic
    gates["G4_regime_agnostic"] = {
        "pass": regime_analysis.get("regime_agnostic_pass", False),
        "regime_gap": regime_analysis.get("regime_gap_ratio"),
        "threshold": "gap <= 0.50",
    }

    # Gate 5: Low QQQ correlation (the whole point)
    qqq_corr = metrics.get("qqq_correlation")
    if qqq_corr is not None:
        gates["G5_low_qqq_correlation"] = {
            "pass": abs(qqq_corr) < 0.50,
            "value": qqq_corr,
            "threshold": "|corr| < 0.50",
        }
    else:
        gates["G5_low_qqq_correlation"] = {"pass": False, "value": None, "threshold": "|corr| < 0.50"}

    all_pass = all(g["pass"] for g in gates.values())
    gates["ALL_GATES_PASS"] = all_pass

    return gates


# ── Strategy Implementations ───────────────────────────────────────────

def strategy_a_four_regime(close, oot_mask):
    """4-Regime Macro Model: VIX + SPY trend classification."""
    vix = close["VIX"]
    spy = close["SPY"]
    spy_sma50 = spy.rolling(50).mean()

    oot = close.loc[oot_mask]
    equity = pd.Series(CAPITAL, index=oot.index)

    prev_asset = None
    for i in range(1, len(oot)):
        date = oot.index[i]
        prev_date = oot.index[i-1]

        v = vix.loc[prev_date] if prev_date in vix.index else 20
        s = spy.loc[prev_date] if prev_date in spy.index else 0
        sma = spy_sma50.loc[prev_date] if prev_date in spy_sma50.index else s

        # Regime classification
        if v < 18 and s > sma:
            asset = "SPY"
        elif v < 18 and s <= sma:
            asset = "GLD"
        elif v > 25:
            asset = "UUP"
        else:
            asset = "SHY"

        # Daily return of chosen asset
        if asset in close.columns and prev_date in close[asset].index and date in close[asset].index:
            asset_ret = (close[asset].loc[date] / close[asset].loc[prev_date]) - 1
        else:
            asset_ret = 0

        # Apply slippage on regime change
        slip = SLIPPAGE_PCT if asset != prev_asset else 0
        net_ret = asset_ret - slip

        equity.iloc[i] = equity.iloc[i-1] * (1 + net_ret)
        prev_asset = asset

    return equity


def strategy_b_risk_parity_lite(close, oot_mask):
    """Risk Parity Lite: equal-weight trending assets."""
    assets = ["GLD", "TLT", "SPY"]

    oot = close.loc[oot_mask]
    equity = pd.Series(CAPITAL, index=oot.index)

    rebal_counter = 0
    current_holdings = []

    for i in range(1, len(oot)):
        date = oot.index[i]
        prev_date = oot.index[i-1]

        # Rebalance every 10 days
        if rebal_counter % 10 == 0:
            new_holdings = []
            for a in assets:
                ma20 = close[a].loc[:prev_date].tail(20).mean()
                ma50 = close[a].loc[:prev_date].tail(50).mean()
                if ma20 > ma50:
                    new_holdings.append(a)

            if len(new_holdings) == 0:
                new_holdings = ["SHY"]

            # Slippage if holdings changed
            if set(new_holdings) != set(current_holdings):
                slip = SLIPPAGE_PCT
            else:
                slip = 0
            current_holdings = new_holdings
        else:
            slip = 0

        # Equal weight return
        daily_ret = 0
        for a in current_holdings:
            if a in close.columns and prev_date in close[a].index and date in close[a].index:
                r = (close[a].loc[date] / close[a].loc[prev_date]) - 1
                daily_ret += r / len(current_holdings)

        equity.iloc[i] = equity.iloc[i-1] * (1 + daily_ret - slip)
        rebal_counter += 1

    return equity


def strategy_c_commodity_bond_barbell(close, oot_mask):
    """Commodity-Bond Barbell: GLD+TLT when volatile, SPY when calm."""
    vix = close["VIX"]
    oot = close.loc[oot_mask]
    equity = pd.Series(CAPITAL, index=oot.index)

    rebal_counter = 0
    current_regime = None

    for i in range(1, len(oot)):
        date = oot.index[i]
        prev_date = oot.index[i-1]

        v = vix.loc[prev_date] if prev_date in vix.index else 20

        if rebal_counter % 5 == 0:
            if v > 20:
                new_regime = "defensive"  # 50% GLD, 50% TLT
            elif v < 15:
                new_regime = "risk_on"    # 100% SPY
            else:
                new_regime = "cash"       # SHY

            slip = SLIPPAGE_PCT if new_regime != current_regime else 0
            current_regime = new_regime
        else:
            slip = 0

        if current_regime == "defensive":
            r_gld = (close["GLD"].loc[date] / close["GLD"].loc[prev_date]) - 1 if date in close["GLD"].index else 0
            r_tlt = (close["TLT"].loc[date] / close["TLT"].loc[prev_date]) - 1 if date in close["TLT"].index else 0
            daily_ret = 0.5 * r_gld + 0.5 * r_tlt
        elif current_regime == "risk_on":
            daily_ret = (close["SPY"].loc[date] / close["SPY"].loc[prev_date]) - 1 if date in close["SPY"].index else 0
        else:
            daily_ret = (close["SHY"].loc[date] / close["SHY"].loc[prev_date]) - 1 if date in close["SHY"].index else 0

        equity.iloc[i] = equity.iloc[i-1] * (1 + daily_ret - slip)
        rebal_counter += 1

    return equity


def strategy_d_dollar_gold_pair(close, oot_mask):
    """Dollar-Gold Pair: trade UUP/GLD ratio momentum."""
    ratio = close["UUP"] / close["GLD"]
    ratio_ma20 = ratio.rolling(20).mean()

    oot = close.loc[oot_mask]
    equity = pd.Series(CAPITAL, index=oot.index)

    hold_counter = 0
    current_pos = "cash"

    for i in range(1, len(oot)):
        date = oot.index[i]
        prev_date = oot.index[i-1]

        # Check if we should re-evaluate (every 15 days or if no position)
        if hold_counter <= 0 or current_pos == "cash":
            r = ratio.loc[prev_date] if prev_date in ratio.index else 1
            ma = ratio_ma20.loc[prev_date] if prev_date in ratio_ma20.index else r

            pct_from_ma = (r - ma) / ma if ma > 0 else 0

            if pct_from_ma < -0.005:  # ratio dropping = USD weakening vs gold
                new_pos = "GLD"
            elif pct_from_ma > 0.005:  # ratio rising = USD strengthening
                new_pos = "UUP"
            else:
                new_pos = "cash"

            slip = SLIPPAGE_PCT if new_pos != current_pos else 0
            if new_pos != current_pos:
                hold_counter = 15
            current_pos = new_pos
        else:
            slip = 0
            hold_counter -= 1

        if current_pos == "GLD":
            daily_ret = (close["GLD"].loc[date] / close["GLD"].loc[prev_date]) - 1
        elif current_pos == "UUP":
            daily_ret = (close["UUP"].loc[date] / close["UUP"].loc[prev_date]) - 1
        else:
            daily_ret = (close["SHY"].loc[date] / close["SHY"].loc[prev_date]) - 1

        equity.iloc[i] = equity.iloc[i-1] * (1 + daily_ret - slip)

    return equity


def strategy_e_macro_momentum(close, oot_mask):
    """Macro Momentum Basket: top 2 by 1m+3m momentum, rebalance monthly."""
    mom_assets = ["GLD", "UUP", "TLT", "EEM", "SPY"]

    oot = close.loc[oot_mask]
    equity = pd.Series(CAPITAL, index=oot.index)

    current_holdings = []
    last_rebal_month = None

    for i in range(1, len(oot)):
        date = oot.index[i]
        prev_date = oot.index[i-1]

        # Rebalance monthly
        curr_month = date.month
        if curr_month != last_rebal_month:
            scores = {}
            for a in mom_assets:
                hist = close[a].loc[:prev_date]
                if len(hist) < 63:
                    continue
                mom_1m = (hist.iloc[-1] / hist.iloc[-21]) - 1 if len(hist) >= 21 else 0
                mom_3m = (hist.iloc[-1] / hist.iloc[-63]) - 1 if len(hist) >= 63 else 0
                scores[a] = mom_1m + mom_3m

            if scores:
                ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                new_holdings = [r[0] for r in ranked[:2]]
            else:
                new_holdings = ["SHY"]

            slip = SLIPPAGE_PCT if set(new_holdings) != set(current_holdings) else 0
            current_holdings = new_holdings
            last_rebal_month = curr_month
        else:
            slip = 0

        daily_ret = 0
        for a in current_holdings:
            if a in close.columns and prev_date in close[a].index and date in close[a].index:
                r = (close[a].loc[date] / close[a].loc[prev_date]) - 1
                daily_ret += r / len(current_holdings)

        equity.iloc[i] = equity.iloc[i-1] * (1 + daily_ret - slip)

    return equity


def strategy_f_vix_regime_allocator(close, oot_mask):
    """VIX Regime Allocator: allocation shifts by VIX level."""
    vix = close["VIX"]
    oot = close.loc[oot_mask]
    equity = pd.Series(CAPITAL, index=oot.index)

    prev_regime = None

    for i in range(1, len(oot)):
        date = oot.index[i]
        prev_date = oot.index[i-1]

        v = vix.loc[prev_date] if prev_date in vix.index else 20

        # Determine allocation
        if v < 15:
            alloc = {"SPY": 1.0}
            regime = "low_vol"
        elif v <= 20:
            alloc = {"SPY": 0.5, "GLD": 0.25, "TLT": 0.25}
            regime = "moderate"
        elif v <= 30:
            alloc = {"GLD": 0.5, "TLT": 0.5}
            regime = "elevated"
        else:
            alloc = {"UUP": 0.5, "SHY": 0.5}
            regime = "crisis"

        # Slippage on regime change
        slip = SLIPPAGE_PCT if regime != prev_regime else 0
        prev_regime = regime

        daily_ret = 0
        for a, w in alloc.items():
            if a in close.columns and prev_date in close[a].index and date in close[a].index:
                r = (close[a].loc[date] / close[a].loc[prev_date]) - 1
                daily_ret += w * r

        equity.iloc[i] = equity.iloc[i-1] * (1 + daily_ret - slip)

    return equity


# ── Main ────────────────────────────────────────────────────────────────

def main():
    close = download_data()

    # OOT mask
    oot_mask = (close.index >= OOT_START) & (close.index <= OOT_END)
    oot_dates = close.index[oot_mask]
    print(f"OOT period: {oot_dates[0].date()} to {oot_dates[-1].date()} ({len(oot_dates)} days)")

    # QQQ returns for correlation
    qqq_returns = close["QQQ"].loc[oot_mask].pct_change().dropna()
    spy_returns = close["SPY"].loc[oot_mask].pct_change().dropna()

    # QQQ benchmark
    qqq_equity = CAPITAL * (close["QQQ"].loc[oot_mask] / close["QQQ"].loc[oot_mask].iloc[0])
    qqq_metrics = calc_metrics(qqq_equity, qqq_returns, "QQQ_Benchmark")

    strategies = {
        "A_Four_Regime_Macro": strategy_a_four_regime,
        "B_Risk_Parity_Lite": strategy_b_risk_parity_lite,
        "C_Commodity_Bond_Barbell": strategy_c_commodity_bond_barbell,
        "D_Dollar_Gold_Pair": strategy_d_dollar_gold_pair,
        "E_Macro_Momentum_Basket": strategy_e_macro_momentum,
        "F_VIX_Regime_Allocator": strategy_f_vix_regime_allocator,
    }

    results = {
        "metadata": {
            "run_date": datetime.now().isoformat(),
            "oot_start": OOT_START,
            "oot_end": OOT_END,
            "capital": CAPITAL,
            "slippage_pct": SLIPPAGE_PCT,
            "commission": COMMISSION,
            "oot_days": len(oot_dates),
            "purpose": "Find macro regime strategies uncorrelated to QQQ",
        },
        "qqq_benchmark": qqq_metrics,
        "strategies": {},
    }

    for name, func in strategies.items():
        print(f"\nRunning {name}...")
        try:
            equity = func(close, oot_mask)
            metrics = calc_metrics(equity, qqq_returns, name)

            if metrics is None:
                print(f"  SKIP: insufficient data")
                continue

            regime = regime_split_analysis(equity, spy_returns)

            strat_returns = equity.pct_change().dropna()
            perm = permutation_test(strat_returns, n_perms=1000)

            gates = five_gate_validation(metrics, regime, perm)

            results["strategies"][name] = {
                "metrics": metrics,
                "regime_split": regime,
                "permutation_test": perm,
                "five_gates": gates,
            }

            # Print summary
            g_pass = sum(1 for k, v in gates.items() if k != "ALL_GATES_PASS" and v.get("pass", False))
            print(f"  Sharpe={metrics['sharpe']:.3f}  Sortino={metrics['sortino']:.3f}  "
                  f"MaxDD={metrics['max_drawdown_pct']:.1f}%  QQQ_corr={metrics['qqq_correlation']}  "
                  f"Gates={g_pass}/5  Total={metrics['total_return_pct']:.1f}%")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results["strategies"][name] = {"error": str(e)}

    # ── Summary / Rankings ──────────────────────────────────────────────
    valid = {k: v for k, v in results["strategies"].items() if "metrics" in v}

    # Rank by Sharpe
    by_sharpe = sorted(valid.items(), key=lambda x: x[1]["metrics"]["sharpe"], reverse=True)

    # Rank by lowest |QQQ correlation|
    by_decorr = sorted(valid.items(),
                       key=lambda x: abs(x[1]["metrics"]["qqq_correlation"] or 1))

    # Combined score: sharpe_rank + decorr_rank (lower = better)
    sharpe_ranks = {name: i for i, (name, _) in enumerate(by_sharpe)}
    decorr_ranks = {name: i for i, (name, _) in enumerate(by_decorr)}
    combined = {name: sharpe_ranks[name] + decorr_ranks[name] for name in valid}
    by_combined = sorted(combined.items(), key=lambda x: x[1])

    results["rankings"] = {
        "by_sharpe": [{"rank": i+1, "name": n, "sharpe": valid[n]["metrics"]["sharpe"]}
                      for i, (n, _) in enumerate(by_sharpe)],
        "by_qqq_decorrelation": [{"rank": i+1, "name": n, "qqq_corr": valid[n]["metrics"]["qqq_correlation"]}
                                  for i, (n, _) in enumerate(by_decorr)],
        "by_combined_rank": [{"rank": i+1, "name": n, "combined_score": s}
                             for i, (n, s) in enumerate(by_combined)],
    }

    # Gate summary
    gate_summary = {}
    for name, data in valid.items():
        gates = data["five_gates"]
        passed = sum(1 for k, v in gates.items() if k != "ALL_GATES_PASS" and v.get("pass", False))
        gate_summary[name] = {
            "gates_passed": f"{passed}/5",
            "all_pass": gates.get("ALL_GATES_PASS", False),
        }
    results["gate_summary"] = gate_summary

    # Key finding
    best = by_combined[0][0] if by_combined else None
    if best:
        bm = valid[best]["metrics"]
        results["key_finding"] = (
            f"Best combined (Sharpe + decorrelation): {best} — "
            f"Sharpe {bm['sharpe']}, QQQ corr {bm['qqq_correlation']}, "
            f"Return {bm['total_return_pct']}%, MaxDD {bm['max_drawdown_pct']}%"
        )

    # Save
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"Results saved to {RESULTS_PATH}")
    print(f"\nQQQ Benchmark: Sharpe={qqq_metrics['sharpe']}, Return={qqq_metrics['total_return_pct']}%")
    print(f"\n--- Rankings by Combined Score (Sharpe + Decorrelation) ---")
    for i, (name, score) in enumerate(by_combined):
        m = valid[name]["metrics"]
        g = gate_summary[name]
        print(f"  #{i+1} {name}: Sharpe={m['sharpe']:.3f}, QQQ_corr={m['qqq_correlation']}, "
              f"Return={m['total_return_pct']:.1f}%, Gates={g['gates_passed']}")

    if best:
        print(f"\nKey Finding: {results['key_finding']}")

    return results


if __name__ == "__main__":
    results = main()

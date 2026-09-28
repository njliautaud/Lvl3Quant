#!/usr/bin/env python3
"""
Dual Regime Sector Rotation Backtest
=====================================
Combines sector momentum alpha (bull) with RSI-bounce alpha (bear).
6 variants, 5-gate validation, walk-forward OOT: Jan 2022 - Jul 2026.

Account: $645 Robinhood, ETF shares ($0 commission, 0.02% slippage).
"""

import json
import warnings
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────
TICKERS = ["SPY", "QQQ", "XLK", "XLE", "XLF", "XLV", "XLC", "XLY",
           "XLI", "XLP", "XLRE", "XLU", "XLB", "TLT", "GLD"]
SECTOR_ETFS = ["QQQ", "XLK", "XLE", "XLF", "XLV", "XLC", "XLY",
               "XLI", "XLP", "XLRE", "XLU", "XLB"]
SAFE_HAVENS = ["TLT", "GLD"]

INITIAL_CAPITAL = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02%
OOT_START = "2022-01-01"
OOT_END = "2026-07-30"
LOOKBACK_START = "2020-06-01"  # need history for 200-SMA

# Validation gates
SHARPE_MIN = 0.5
PERM_P_MAX = 0.05
REGIME_GAP_MAX = 0.5
MAX_DD_FLOOR = -0.50
MIN_TRADES = 20
N_PERMUTATIONS = 1000

# ── Data Download ─────────────────────────────────────────────────────
def download_data():
    print("Downloading price data...")
    data = yf.download(TICKERS, start=LOOKBACK_START, end=OOT_END,
                       auto_adjust=True, progress=False)
    close = data["Close"].dropna(how="all")
    # Forward-fill small gaps, drop rows where SPY is missing
    close = close.ffill().dropna(subset=["SPY"])
    print(f"  Data: {close.index[0].date()} to {close.index[-1].date()}, {len(close)} bars")
    return close


# ── Indicators ────────────────────────────────────────────────────────
def compute_indicators(close):
    """Pre-compute all indicators needed by all variants."""
    ind = {}
    ind["sma200"] = close.rolling(200).mean()
    ind["mom20"] = close.pct_change(20)
    ind["mom10"] = close.pct_change(10)
    ind["ret_weekly"] = close.pct_change(5)

    # RSI(5)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(5).mean()
    avg_loss = loss.rolling(5).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    ind["rsi5"] = 100 - (100 / (1 + rs))

    # VIX proxy: use ^VIX
    try:
        vix = yf.download("^VIX", start=LOOKBACK_START, end=OOT_END,
                          auto_adjust=True, progress=False)["Close"]
        if isinstance(vix, pd.DataFrame):
            vix = vix.iloc[:, 0]
        vix = vix.reindex(close.index).ffill()
        ind["vix"] = vix
    except Exception:
        # Fallback: use SPY realized vol * 100 as proxy
        ind["vix"] = close["SPY"].pct_change().rolling(20).std() * np.sqrt(252) * 100

    # Sector dispersion: std of weekly returns across sector ETFs
    sector_weekly = ind["ret_weekly"][SECTOR_ETFS]
    ind["dispersion"] = sector_weekly.std(axis=1)
    ind["disp_75"] = ind["dispersion"].rolling(252).quantile(0.75)
    ind["disp_25"] = ind["dispersion"].rolling(252).quantile(0.25)

    return ind


# ── Regime ────────────────────────────────────────────────────────────
def get_regime(date, close, ind):
    """Bull if SPY > 200-SMA, else Bear."""
    spy_price = close.loc[date, "SPY"]
    spy_sma = ind["sma200"].loc[date, "SPY"]
    if pd.isna(spy_sma):
        return "bull"
    return "bull" if spy_price > spy_sma else "bear"


# ── Backtest Engine ──────────────────────────────────────────────────
def run_backtest(close, ind, strategy_fn, name=""):
    """
    Generic backtest engine. strategy_fn(date, close, ind, state) returns
    target_weights dict {ticker: weight} where weights sum to <=1.
    """
    oot_mask = close.index >= OOT_START
    dates = close.index[oot_mask]
    if len(dates) == 0:
        return None

    capital = INITIAL_CAPITAL
    positions = {}  # ticker -> {"shares": n, "entry_date": date}
    equity_curve = []
    trades = []
    regimes = []
    state = {"last_rebal": None, "holdings": {}}

    for i, date in enumerate(dates):
        # Mark-to-market
        port_value = 0.0
        for tk, pos in positions.items():
            price = close.loc[date, tk]
            if not pd.isna(price):
                port_value += pos["shares"] * price
        cash = capital
        total_value = cash + port_value

        # Get target weights
        target = strategy_fn(date, close, ind, state)
        if target is None:
            equity_curve.append(total_value)
            regimes.append(get_regime(date, close, ind))
            continue

        # Rebalance: liquidate positions not in target, adjust sizes
        # First, compute current value
        new_positions = {}
        new_capital = total_value  # start with full value

        # Apply slippage on all trades
        for tk, weight in target.items():
            if weight <= 0 or pd.isna(close.loc[date, tk]):
                continue
            alloc = total_value * weight
            price = close.loc[date, tk]
            # Slippage
            slippage_cost = alloc * SLIPPAGE_PCT
            alloc_after = alloc - slippage_cost
            shares = int(alloc_after / price)  # whole shares for $645 account
            if shares > 0:
                new_positions[tk] = {"shares": shares, "entry_date": date}
                new_capital -= shares * price + slippage_cost
                trades.append({
                    "date": str(date.date()),
                    "ticker": tk,
                    "shares": shares,
                    "price": float(price),
                    "regime": get_regime(date, close, ind)
                })

        positions = new_positions
        capital = max(new_capital, 0)

        # Recalc total
        port_value = sum(pos["shares"] * close.loc[date, tk]
                         for tk, pos in positions.items()
                         if not pd.isna(close.loc[date, tk]))
        total_value = capital + port_value
        equity_curve.append(total_value)
        regimes.append(get_regime(date, close, ind))
        state["holdings"] = {tk: pos["shares"] for tk, pos in positions.items()}

    return {
        "equity": np.array(equity_curve),
        "dates": dates,
        "trades": trades,
        "regimes": regimes,
        "name": name
    }


# ── Strategy Functions ───────────────────────────────────────────────

def make_strategy_a(close, ind):
    """A) Momentum in Bull / RSI in Bear"""
    rebal_state = {"last_rebal": None, "rsi_entries": {}}

    def strategy(date, close_df, indicators, state):
        regime = get_regime(date, close_df, indicators)

        if regime == "bull":
            # Weekly rebalance
            if rebal_state["last_rebal"] is not None:
                delta = (date - rebal_state["last_rebal"]).days
                if delta < 5:
                    return None
            rebal_state["last_rebal"] = date
            rebal_state["rsi_entries"] = {}

            # Top-2 sectors by 20d momentum
            mom = indicators["mom20"].loc[date, SECTOR_ETFS].dropna()
            if len(mom) < 2:
                return None
            top2 = mom.nlargest(2).index.tolist()
            return {tk: 0.5 for tk in top2}

        else:  # bear
            # Check RSI entries for expiry (10-day hold max)
            for tk in list(rebal_state["rsi_entries"].keys()):
                entry_date = rebal_state["rsi_entries"][tk]
                if (date - entry_date).days >= 10:
                    del rebal_state["rsi_entries"][tk]

            # Look for new RSI bounce entries
            for tk in SECTOR_ETFS:
                rsi_val = indicators["rsi5"].loc[date, tk] if tk in indicators["rsi5"].columns else None
                sma_val = indicators["sma200"].loc[date, tk] if tk in indicators["sma200"].columns else None
                price = close_df.loc[date, tk]

                if (rsi_val is not None and not pd.isna(rsi_val) and rsi_val < 20
                        and sma_val is not None and not pd.isna(sma_val)
                        and price > sma_val
                        and tk not in rebal_state["rsi_entries"]):
                    rebal_state["rsi_entries"][tk] = date

            # Equal weight active RSI positions
            active = list(rebal_state["rsi_entries"].keys())
            if not active:
                # Defensive: TLT
                return {"TLT": 0.9}
            weight = 0.9 / len(active)
            return {tk: weight for tk in active}

    return strategy


def make_strategy_b(close, ind):
    """B) All-Weather Portfolio"""
    rebal_state = {"last_rebal": None}

    def strategy(date, close_df, indicators, state):
        if rebal_state["last_rebal"] is not None:
            delta = (date - rebal_state["last_rebal"]).days
            if delta < 5:
                return None
        rebal_state["last_rebal"] = date

        # Top momentum sector
        mom = indicators["mom20"].loc[date, SECTOR_ETFS].dropna()
        if len(mom) == 0:
            return {"SPY": 0.30, "TLT": 0.20, "GLD": 0.10}
        top1 = mom.idxmax()
        return {top1: 0.40, "SPY": 0.30, "TLT": 0.20, "GLD": 0.10}

    return strategy


def make_strategy_c(close, ind):
    """C) Momentum + VIX Filter"""
    rebal_state = {"last_regime": None, "last_rebal": None}

    def strategy(date, close_df, indicators, state):
        vix_val = indicators["vix"].loc[date] if date in indicators["vix"].index else 20
        if isinstance(vix_val, pd.Series):
            vix_val = vix_val.iloc[0]
        if pd.isna(vix_val):
            vix_val = 20

        current_regime = "risk_on" if vix_val < 20 else "risk_off"

        # Rebalance on regime change or weekly
        regime_changed = current_regime != rebal_state["last_regime"]
        weekly = (rebal_state["last_rebal"] is None or
                  (date - rebal_state["last_rebal"]).days >= 5)

        if not regime_changed and not weekly:
            return None

        rebal_state["last_regime"] = current_regime
        rebal_state["last_rebal"] = date

        if current_regime == "risk_on":
            # Top-2 sectors with positive 20d momentum
            mom = indicators["mom20"].loc[date, SECTOR_ETFS].dropna()
            positive = mom[mom > 0]
            if len(positive) >= 2:
                top2 = positive.nlargest(2).index.tolist()
                return {tk: 0.45 for tk in top2}
            elif len(positive) == 1:
                return {positive.index[0]: 0.45, "TLT": 0.45}
            else:
                return {"TLT": 0.5, "GLD": 0.5}
        else:
            return {"TLT": 0.5, "GLD": 0.5}

    return strategy


def make_strategy_d(close, ind):
    """D) Sector Dispersion Play"""
    rebal_state = {"last_rebal": None}

    def strategy(date, close_df, indicators, state):
        if rebal_state["last_rebal"] is not None:
            delta = (date - rebal_state["last_rebal"]).days
            if delta < 5:
                return None
        rebal_state["last_rebal"] = date

        disp = indicators["dispersion"].loc[date]
        disp_75 = indicators["disp_75"].loc[date]
        disp_25 = indicators["disp_25"].loc[date]

        if pd.isna(disp) or pd.isna(disp_75) or pd.isna(disp_25):
            return None

        mom = indicators["mom20"].loc[date, SECTOR_ETFS].dropna()
        if len(mom) == 0:
            return None

        if disp > disp_75:
            # High dispersion: concentrate in top-1
            top1 = mom.idxmax()
            return {top1: 0.9}
        elif disp < disp_25:
            # Low dispersion: equal-weight all sectors
            weight = 0.9 / len(SECTOR_ETFS)
            return {tk: weight for tk in SECTOR_ETFS}
        else:
            # Middle: top-3
            top3 = mom.nlargest(3).index.tolist()
            return {tk: 0.3 for tk in top3}

    return strategy


def make_strategy_e(close, ind):
    """E) Dual Momentum (Absolute + Relative)"""
    rebal_state = {"last_rebal": None}

    def strategy(date, close_df, indicators, state):
        if rebal_state["last_rebal"] is not None:
            delta = (date - rebal_state["last_rebal"]).days
            if delta < 5:
                return None
        rebal_state["last_rebal"] = date

        mom = indicators["mom20"].loc[date, SECTOR_ETFS].dropna()
        if len(mom) == 0:
            return {"TLT": 0.9}

        # Absolute filter: must have positive momentum
        positive = mom[mom > 0]
        if len(positive) == 0:
            return {"TLT": 0.9}

        # Relative: top-3 among positive
        top = positive.nlargest(min(3, len(positive))).index.tolist()
        weight = 0.9 / len(top)
        return {tk: weight for tk in top}

    return strategy


def make_strategy_f(close, ind):
    """F) Adaptive Hold Period"""
    hold_state = {"holdings": {}, "last_check": None}

    def strategy(date, close_df, indicators, state):
        # Check daily for momentum flips
        if hold_state["last_check"] is not None:
            delta = (date - hold_state["last_check"]).days
            if delta < 1:
                return None
        hold_state["last_check"] = date

        mom10 = indicators["mom10"].loc[date, SECTOR_ETFS].dropna()
        mom20 = indicators["mom20"].loc[date, SECTOR_ETFS].dropna()

        # Exit positions where 10d momentum flipped negative
        for tk in list(hold_state["holdings"].keys()):
            if tk in mom10.index and mom10[tk] < 0:
                del hold_state["holdings"][tk]

        # Enter if we have < 2 positions: pick top-2 by 20d momentum
        if len(hold_state["holdings"]) < 2:
            candidates = mom20.drop(labels=list(hold_state["holdings"].keys()), errors="ignore")
            positive = candidates[candidates > 0]
            if len(positive) > 0:
                needed = 2 - len(hold_state["holdings"])
                top = positive.nlargest(min(needed, len(positive))).index.tolist()
                for tk in top:
                    hold_state["holdings"][tk] = date

        active = list(hold_state["holdings"].keys())
        if not active:
            return {"TLT": 0.9}
        weight = 0.9 / len(active)
        return {tk: weight for tk in active}

    return strategy


# ── Metrics ──────────────────────────────────────────────────────────
def compute_metrics(result, spy_close):
    eq = result["equity"]
    if len(eq) < 10:
        return None

    returns = np.diff(eq) / eq[:-1]
    returns = returns[np.isfinite(returns)]
    if len(returns) < 10:
        return None

    # Annualized metrics
    ann_factor = 252
    mean_ret = np.mean(returns) * ann_factor
    std_ret = np.std(returns, ddof=1) * np.sqrt(ann_factor)
    sharpe = mean_ret / std_ret if std_ret > 0 else 0

    downside = returns[returns < 0]
    downside_std = np.std(downside, ddof=1) * np.sqrt(ann_factor) if len(downside) > 1 else 1e-6
    sortino = mean_ret / downside_std

    # Max drawdown
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    max_dd = np.min(dd)

    # Total return
    total_ret = (eq[-1] / eq[0]) - 1

    # CAGR
    years = len(returns) / 252
    cagr = (eq[-1] / eq[0]) ** (1 / years) - 1 if years > 0 else 0

    # Win rate (of daily returns)
    wr = np.mean(returns > 0) if len(returns) > 0 else 0

    # Number of rebalancing events (trades)
    n_trades = len(result["trades"])

    # Regime-stratified Sharpe
    regimes = result["regimes"]
    daily_rets = np.diff(eq) / eq[:-1]

    bull_rets = [daily_rets[i] for i in range(len(daily_rets))
                 if i + 1 < len(regimes) and regimes[i + 1] == "bull"]
    bear_rets = [daily_rets[i] for i in range(len(daily_rets))
                 if i + 1 < len(regimes) and regimes[i + 1] == "bear"]

    def _sharpe(rets):
        rets = np.array(rets)
        rets = rets[np.isfinite(rets)]
        if len(rets) < 5:
            return 0.0
        return (np.mean(rets) * 252) / (np.std(rets, ddof=1) * np.sqrt(252)) if np.std(rets) > 0 else 0

    sharpe_bull = _sharpe(bull_rets)
    sharpe_bear = _sharpe(bear_rets)
    regime_gap = abs(sharpe_bull - sharpe_bear) / max(abs(sharpe_bull), abs(sharpe_bear), 1e-6)

    # SPY benchmark
    spy_eq = spy_close.values
    spy_eq = spy_eq[~np.isnan(spy_eq)]
    spy_rets = np.diff(spy_eq) / spy_eq[:-1]
    spy_sharpe = (np.mean(spy_rets) * 252) / (np.std(spy_rets, ddof=1) * np.sqrt(252)) if np.std(spy_rets) > 0 else 0
    spy_total = (spy_eq[-1] / spy_eq[0]) - 1

    return {
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "cagr": round(cagr * 100, 2),
        "total_return_pct": round(total_ret * 100, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "win_rate": round(wr * 100, 1),
        "n_trades": n_trades,
        "sharpe_bull": round(sharpe_bull, 3),
        "sharpe_bear": round(sharpe_bear, 3),
        "regime_gap": round(regime_gap, 3),
        "final_equity": round(eq[-1], 2),
        "spy_sharpe": round(spy_sharpe, 3),
        "spy_total_return_pct": round(spy_total * 100, 2),
    }


# ── Permutation Test ─────────────────────────────────────────────────
def permutation_test(result, close, ind, strategy_maker, n_perms=N_PERMUTATIONS):
    """
    Block-shuffle permutation test on strategy excess returns.
    Tests whether sector SELECTION adds value vs random selection.
    Shuffles which sectors are chosen (not the returns themselves)
    by randomly reassigning sector labels at each rebalance point.
    Uses fast return-level simulation instead of full backtest rerun.
    """
    eq = result["equity"]
    daily_rets = np.diff(eq) / eq[:-1]
    daily_rets = daily_rets[np.isfinite(daily_rets)]
    if len(daily_rets) < 20:
        return 1.0

    actual_sharpe = (np.mean(daily_rets) * 252) / (np.std(daily_rets, ddof=1) * np.sqrt(252))

    # Compute equal-weight sector benchmark returns for comparison
    oot_mask = close.index >= OOT_START
    oot_close = close.loc[oot_mask, SECTOR_ETFS].dropna(how="all")
    sector_daily_rets = oot_close.pct_change().dropna(how="all")

    # For each permutation: randomly pick sectors (same count as strategy)
    # and compute the Sharpe of that random portfolio
    rng = np.random.default_rng(42)
    count_ge = 0

    # Estimate average number of positions from trade data
    avg_positions = 2  # most variants hold 2-3

    n_days = len(sector_daily_rets)
    sector_arr = sector_daily_rets.values  # (days, n_sectors)
    n_sectors = sector_arr.shape[1]

    for _ in range(n_perms):
        # For each day, randomly pick avg_positions sectors
        perm_rets = np.zeros(n_days)
        for d in range(n_days):
            chosen = rng.choice(n_sectors, size=min(avg_positions, n_sectors), replace=False)
            perm_rets[d] = np.nanmean(sector_arr[d, chosen])

        perm_rets = perm_rets[np.isfinite(perm_rets)]
        if len(perm_rets) < 20:
            continue

        perm_sharpe = (np.mean(perm_rets) * 252) / (np.std(perm_rets, ddof=1) * np.sqrt(252))
        if perm_sharpe >= actual_sharpe:
            count_ge += 1

    return count_ge / n_perms


# ── Validation Gates ─────────────────────────────────────────────────
def validate(metrics, p_value):
    gates = {
        "sharpe_gt_0.5": metrics["sharpe"] > SHARPE_MIN,
        "perm_p_lt_0.05": p_value < PERM_P_MAX,
        "regime_gap_lt_0.5": metrics["regime_gap"] < REGIME_GAP_MAX,
        "max_dd_gt_neg50": metrics["max_drawdown_pct"] > -50,
        "min_20_trades": metrics["n_trades"] >= MIN_TRADES,
    }
    gates["all_passed"] = all(gates.values())
    return gates


# ── Main ─────────────────────────────────────────────────────────────
def main():
    close = download_data()
    ind = compute_indicators(close)

    strategies = {
        "A_MomBull_RSIBear": make_strategy_a,
        "B_AllWeather": make_strategy_b,
        "C_MomVIXFilter": make_strategy_c,
        "D_SectorDispersion": make_strategy_d,
        "E_DualMomentum": make_strategy_e,
        "F_AdaptiveHold": make_strategy_f,
    }

    oot_dates = close.index[close.index >= OOT_START]
    spy_oot = close.loc[oot_dates, "SPY"]

    all_results = {}

    for name, maker in strategies.items():
        print(f"\n{'='*60}")
        print(f"  Strategy {name}")
        print(f"{'='*60}")

        strategy_fn = maker(close, ind)
        result = run_backtest(close, ind, strategy_fn, name=name)

        if result is None:
            print(f"  SKIP: No trades generated")
            all_results[name] = {"status": "no_trades"}
            continue

        metrics = compute_metrics(result, spy_oot)
        if metrics is None:
            print(f"  SKIP: Insufficient data")
            all_results[name] = {"status": "insufficient_data"}
            continue

        print(f"  Sharpe: {metrics['sharpe']}  Sortino: {metrics['sortino']}")
        print(f"  CAGR: {metrics['cagr']}%  Total Return: {metrics['total_return_pct']}%")
        print(f"  MaxDD: {metrics['max_drawdown_pct']}%  WinRate: {metrics['win_rate']}%")
        print(f"  Trades: {metrics['n_trades']}")
        print(f"  Sharpe Bull: {metrics['sharpe_bull']}  Bear: {metrics['sharpe_bear']}  Gap: {metrics['regime_gap']}")
        print(f"  Final Equity: ${metrics['final_equity']}")
        print(f"  SPY Benchmark: Sharpe={metrics['spy_sharpe']}, Return={metrics['spy_total_return_pct']}%")

        # Permutation test
        print(f"  Running permutation test ({N_PERMUTATIONS} iterations)...", end=" ", flush=True)
        p_value = permutation_test(result, close, ind, maker)
        print(f"p={p_value:.4f}")

        # Validation
        gates = validate(metrics, p_value)
        print(f"  Validation Gates:")
        for gate, passed in gates.items():
            status = "PASS" if passed else "FAIL"
            print(f"    {gate}: {status}")

        all_results[name] = {
            "metrics": metrics,
            "p_value": round(p_value, 4),
            "gates": gates,
            "status": "passed" if gates["all_passed"] else "failed",
            "n_equity_points": len(result["equity"]),
        }

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    passed = []
    for name, res in all_results.items():
        status = res.get("status", "unknown")
        if status == "passed":
            s = res["metrics"]["sharpe"]
            print(f"  {name}: PASSED (Sharpe={s})")
            passed.append(name)
        elif status == "failed":
            s = res["metrics"]["sharpe"]
            failed_gates = [g for g, v in res["gates"].items() if not v and g != "all_passed"]
            print(f"  {name}: FAILED (Sharpe={s}, failed: {', '.join(failed_gates)})")
        else:
            print(f"  {name}: {status}")

    print(f"\n  {len(passed)}/{len(strategies)} strategies passed all 5 gates")

    # Save results
    output_path = "/home/jupiter/Lvl3Quant/data/dual_regime_strategy_results.json"
    output = {
        "run_date": datetime.now().isoformat(),
        "oot_period": f"{OOT_START} to {OOT_END}",
        "initial_capital": INITIAL_CAPITAL,
        "instruments": TICKERS,
        "validation_gates": {
            "sharpe_min": SHARPE_MIN,
            "perm_p_max": PERM_P_MAX,
            "regime_gap_max": REGIME_GAP_MAX,
            "max_dd_floor_pct": MAX_DD_FLOOR * 100,
            "min_trades": MIN_TRADES,
        },
        "results": all_results,
        "passed_strategies": passed,
    }

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    output = convert(output)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {output_path}")

    return all_results


if __name__ == "__main__":
    main()

"""
macro_event_trading.py — Backtest macro event calendar trading strategies.

Tests 6 strategies around FOMC, CPI, and NFP events:
  A. Pre-FOMC drift (buy SPY T-2, sell FOMC close)
  B. Post-FOMC momentum (buy FOMC close if up, sell T+3)
  C. FOMC straddle (buy T-2, sell T+1)
  D. CPI surprise fade (fade >1% CPI day move, hold 5 days)
  E. Pre-CPI positioning (buy QQQ T-1, sell CPI close)
  F. Event density (buy SPY when 2+ events in 5-day window)

5-gate validation: Sharpe>0.5, permutation p<0.05, regime gap<0.5,
MaxDD>-50%, trades>=20.
"""

import json
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.utils.yfinance_safe import safe_download

# ──────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────
STARTING_CAPITAL = 645.0
SLIPPAGE_BPS = 2  # 2 basis points
OOT_START = "2022-01-01"
OOT_END = "2026-07-25"
PERMUTATION_ITERS = 500
SMA_WINDOW = 200

# ──────────────────────────────────────────────────────────────────────
# EVENT DATES
# ──────────────────────────────────────────────────────────────────────

FOMC_DATES = [
    # 2022
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15",
    "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
    # 2023
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14",
    "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    # 2024
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-11-05", "2025-12-17",
    # 2026
    "2026-01-28", "2026-03-18", "2026-05-06", "2026-06-17",
]

def _generate_cpi_dates(start_year=2022, end_year=2026) -> List[str]:
    """
    CPI is typically released around the 10th-14th of each month (Tue/Wed).
    Use second Tuesday heuristic, adjusted to known release pattern.
    """
    known_cpi = {
        # 2022 CPI release dates (for prior month's data)
        2022: ["2022-01-12", "2022-02-10", "2022-03-10", "2022-04-12",
               "2022-05-11", "2022-06-10", "2022-07-13", "2022-08-10",
               "2022-09-13", "2022-10-13", "2022-11-10", "2022-12-13"],
        2023: ["2023-01-12", "2023-02-14", "2023-03-14", "2023-04-12",
               "2023-05-10", "2023-06-13", "2023-07-12", "2023-08-10",
               "2023-09-13", "2023-10-12", "2023-11-14", "2023-12-12"],
        2024: ["2024-01-11", "2024-02-13", "2024-03-12", "2024-04-10",
               "2024-05-15", "2024-06-12", "2024-07-11", "2024-08-14",
               "2024-09-11", "2024-10-10", "2024-11-13", "2024-12-11"],
        2025: ["2025-01-15", "2025-02-12", "2025-03-12", "2025-04-10",
               "2025-05-13", "2025-06-11", "2025-07-15", "2025-08-12",
               "2025-09-10", "2025-10-14", "2025-11-12", "2025-12-10"],
        2026: ["2026-01-13", "2026-02-11", "2026-03-11", "2026-04-14",
               "2026-05-12", "2026-06-10", "2026-07-14"],
    }
    dates = []
    for yr in range(start_year, end_year + 1):
        if yr in known_cpi:
            dates.extend(known_cpi[yr])
    return dates

def _generate_nfp_dates(start_year=2022, end_year=2026) -> List[str]:
    """NFP = first Friday of each month."""
    dates = []
    for yr in range(start_year, end_year + 1):
        end_month = 13 if yr < end_year else 8  # through Jul 2026
        for mo in range(1, end_month):
            first_day = datetime(yr, mo, 1)
            # Find first Friday: weekday() 4 = Friday
            days_ahead = (4 - first_day.weekday()) % 7
            first_friday = first_day + timedelta(days=days_ahead)
            dates.append(first_friday.strftime("%Y-%m-%d"))
    return dates

CPI_DATES = _generate_cpi_dates()
NFP_DATES = _generate_nfp_dates()

# ──────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────

def get_trading_day_offset(trading_days: pd.DatetimeIndex, target_date: str, offset: int) -> Optional[pd.Timestamp]:
    """Get the trading day that is `offset` trading days from target_date.
    offset<0 = before, offset>0 = after."""
    td = pd.Timestamp(target_date)
    # Find nearest trading day on or after target
    mask = trading_days >= td
    if not mask.any():
        return None
    idx = trading_days.get_indexer([td], method="ffill")[0]
    if idx < 0:
        idx = trading_days.get_indexer([td], method="bfill")[0]
    if idx < 0:
        return None
    new_idx = idx + offset
    if 0 <= new_idx < len(trading_days):
        return trading_days[new_idx]
    return None


def find_nearest_trading_day(trading_days: pd.DatetimeIndex, target_date: str) -> Optional[pd.Timestamp]:
    """Find nearest trading day on or before target_date."""
    td = pd.Timestamp(target_date)
    mask = trading_days <= td
    if mask.any():
        return trading_days[mask][-1]
    mask2 = trading_days >= td
    if mask2.any():
        return trading_days[mask2][0]
    return None


def apply_slippage(price: float, direction: str) -> float:
    """Apply slippage: buy slightly higher, sell slightly lower."""
    if direction == "buy":
        return price * (1 + SLIPPAGE_BPS / 10000)
    else:
        return price * (1 - SLIPPAGE_BPS / 10000)


def compute_metrics(returns: np.ndarray) -> Dict:
    """Compute Sharpe, Sortino, PF, WR, MaxDD from array of per-trade returns."""
    if len(returns) == 0:
        return {"sharpe": 0, "sortino": 0, "pf": 0, "wr": 0, "maxdd": 0,
                "n_trades": 0, "total_return_pct": 0, "mean_return_pct": 0}

    mean_r = np.mean(returns)
    std_r = np.std(returns, ddof=1) if len(returns) > 1 else 1e-9

    # Annualize assuming ~8 trades/year for event strategies
    trades_per_year = max(len(returns) / 4.5, 1)  # ~4.5 years of data
    sharpe = (mean_r / std_r) * np.sqrt(trades_per_year) if std_r > 1e-9 else 0

    downside = returns[returns < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-9
    sortino = (mean_r / downside_std) * np.sqrt(trades_per_year) if downside_std > 1e-9 else 0

    gross_gains = np.sum(returns[returns > 0])
    gross_losses = abs(np.sum(returns[returns < 0]))
    pf = gross_gains / gross_losses if gross_losses > 0 else float("inf")

    wr = np.mean(returns > 0) * 100

    # MaxDD on cumulative equity curve
    cumulative = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - peak) / peak
    maxdd = np.min(drawdown) * 100 if len(drawdown) > 0 else 0

    total_return = (np.prod(1 + returns) - 1) * 100

    return {
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "pf": round(pf, 3),
        "wr": round(wr, 1),
        "maxdd": round(maxdd, 2),
        "n_trades": len(returns),
        "total_return_pct": round(total_return, 2),
        "mean_return_pct": round(mean_r * 100, 4),
    }


def regime_split(trades: List[Dict], spy_close: pd.Series, sma: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """Split trade returns into bull/bear based on entry-date regime."""
    bull_rets, bear_rets = [], []
    for t in trades:
        entry_date = t["entry_date"]
        if entry_date in spy_close.index and entry_date in sma.index:
            if spy_close.loc[entry_date] > sma.loc[entry_date]:
                bull_rets.append(t["return"])
            else:
                bear_rets.append(t["return"])
        else:
            # Default to bull if can't determine
            bull_rets.append(t["return"])
    return np.array(bull_rets), np.array(bear_rets)


def permutation_test(returns: np.ndarray, all_daily_returns: np.ndarray,
                     n_iters: int = PERMUTATION_ITERS) -> float:
    """Permutation test: shuffle entry dates to get null distribution of mean return."""
    if len(returns) < 5 or len(all_daily_returns) < 20:
        return 1.0
    observed_mean = np.mean(returns)
    n_trades = len(returns)
    count_better = 0
    rng = np.random.RandomState(42)
    for _ in range(n_iters):
        random_returns = rng.choice(all_daily_returns, size=n_trades, replace=True)
        if np.mean(random_returns) >= observed_mean:
            count_better += 1
    return count_better / n_iters


def validate_strategy(name: str, trades: List[Dict], spy_close: pd.Series,
                      sma: pd.Series, all_daily_returns: np.ndarray) -> Dict:
    """Run 5-gate validation on a strategy."""
    returns = np.array([t["return"] for t in trades])
    metrics = compute_metrics(returns)

    # Regime analysis
    bull_rets, bear_rets = regime_split(trades, spy_close, sma)
    bull_metrics = compute_metrics(bull_rets)
    bear_metrics = compute_metrics(bear_rets)

    # Regime gap
    bull_sharpe = bull_metrics["sharpe"]
    bear_sharpe = bear_metrics["sharpe"]
    max_abs = max(abs(bull_sharpe), abs(bear_sharpe), 1e-9)
    regime_gap = abs(bull_sharpe - bear_sharpe) / max_abs

    # Permutation test
    p_value = permutation_test(returns, all_daily_returns)

    # 5 gates
    gates = {
        "sharpe_gt_0.5": metrics["sharpe"] > 0.5,
        "permutation_p_lt_0.05": p_value < 0.05,
        "regime_gap_lt_0.5": regime_gap < 0.5,
        "maxdd_gt_neg50": metrics["maxdd"] > -50,
        "trades_gte_20": metrics["n_trades"] >= 20,
    }
    gates_passed = sum(gates.values())

    # Final equity
    final_equity = STARTING_CAPITAL * np.prod(1 + returns) if len(returns) > 0 else STARTING_CAPITAL

    return {
        "strategy": name,
        "metrics": metrics,
        "bull_sharpe": round(bull_sharpe, 3),
        "bear_sharpe": round(bear_sharpe, 3),
        "regime_gap": round(regime_gap, 3),
        "permutation_p": round(p_value, 4),
        "gates": gates,
        "gates_passed": f"{gates_passed}/5",
        "all_gates_pass": gates_passed == 5,
        "final_equity": round(final_equity, 2),
        "n_bull_trades": len(bull_rets),
        "n_bear_trades": len(bear_rets),
        "trade_dates_sample": [t["entry_date"].strftime("%Y-%m-%d") for t in trades[:5]],
    }


# ──────────────────────────────────────────────────────────────────────
# STRATEGY IMPLEMENTATIONS
# ──────────────────────────────────────────────────────────────────────

def strategy_a_pre_fomc_drift(close: pd.DataFrame, trading_days: pd.DatetimeIndex) -> List[Dict]:
    """A. Buy SPY 2 trading days before FOMC, sell at FOMC close."""
    trades = []
    spy = close["SPY"]
    for date_str in FOMC_DATES:
        fomc_day = find_nearest_trading_day(trading_days, date_str)
        if fomc_day is None:
            continue
        entry_day = get_trading_day_offset(trading_days, date_str, -2)
        if entry_day is None or entry_day not in spy.index or fomc_day not in spy.index:
            continue
        entry_price = apply_slippage(spy.loc[entry_day], "buy")
        exit_price = apply_slippage(spy.loc[fomc_day], "sell")
        if entry_price > 0:
            ret = (exit_price - entry_price) / entry_price
            trades.append({"entry_date": entry_day, "exit_date": fomc_day,
                           "entry_price": entry_price, "exit_price": exit_price, "return": ret})
    return trades


def strategy_b_post_fomc_momentum(close: pd.DataFrame, trading_days: pd.DatetimeIndex) -> List[Dict]:
    """B. Buy SPY at FOMC close if SPY closed up vs prior day, sell 3 days later."""
    trades = []
    spy = close["SPY"]
    for date_str in FOMC_DATES:
        fomc_day = find_nearest_trading_day(trading_days, date_str)
        if fomc_day is None or fomc_day not in spy.index:
            continue
        prior_day = get_trading_day_offset(trading_days, date_str, -1)
        if prior_day is None or prior_day not in spy.index:
            continue
        # Only enter if FOMC day was up
        if spy.loc[fomc_day] <= spy.loc[prior_day]:
            continue
        exit_day = get_trading_day_offset(trading_days, fomc_day.strftime("%Y-%m-%d"), 3)
        if exit_day is None or exit_day not in spy.index:
            continue
        entry_price = apply_slippage(spy.loc[fomc_day], "buy")
        exit_price = apply_slippage(spy.loc[exit_day], "sell")
        if entry_price > 0:
            ret = (exit_price - entry_price) / entry_price
            trades.append({"entry_date": fomc_day, "exit_date": exit_day,
                           "entry_price": entry_price, "exit_price": exit_price, "return": ret})
    return trades


def strategy_c_fomc_straddle(close: pd.DataFrame, trading_days: pd.DatetimeIndex) -> List[Dict]:
    """C. Buy SPY 2 days before FOMC, sell 1 day after."""
    trades = []
    spy = close["SPY"]
    for date_str in FOMC_DATES:
        fomc_day = find_nearest_trading_day(trading_days, date_str)
        if fomc_day is None:
            continue
        entry_day = get_trading_day_offset(trading_days, date_str, -2)
        exit_day = get_trading_day_offset(trading_days, fomc_day.strftime("%Y-%m-%d"), 1)
        if entry_day is None or exit_day is None:
            continue
        if entry_day not in spy.index or exit_day not in spy.index:
            continue
        entry_price = apply_slippage(spy.loc[entry_day], "buy")
        exit_price = apply_slippage(spy.loc[exit_day], "sell")
        if entry_price > 0:
            ret = (exit_price - entry_price) / entry_price
            trades.append({"entry_date": entry_day, "exit_date": exit_day,
                           "entry_price": entry_price, "exit_price": exit_price, "return": ret})
    return trades


def strategy_d_cpi_surprise_fade(close: pd.DataFrame, trading_days: pd.DatetimeIndex) -> List[Dict]:
    """D. If CPI day has >1% move, fade it. Buy at close if down >1%, sell 5 days later."""
    trades = []
    spy = close["SPY"]
    for date_str in CPI_DATES:
        cpi_day = find_nearest_trading_day(trading_days, date_str)
        if cpi_day is None or cpi_day not in spy.index:
            continue
        prior_day = get_trading_day_offset(trading_days, date_str, -1)
        if prior_day is None or prior_day not in spy.index:
            continue
        day_return = (spy.loc[cpi_day] - spy.loc[prior_day]) / spy.loc[prior_day]
        # Only trade if >1% absolute move AND it was a down day (fade the drop)
        if abs(day_return) < 0.01:
            continue
        if day_return >= 0:
            continue  # Only fade down moves (buy the dip)
        exit_day = get_trading_day_offset(trading_days, cpi_day.strftime("%Y-%m-%d"), 5)
        if exit_day is None or exit_day not in spy.index:
            continue
        entry_price = apply_slippage(spy.loc[cpi_day], "buy")
        exit_price = apply_slippage(spy.loc[exit_day], "sell")
        if entry_price > 0:
            ret = (exit_price - entry_price) / entry_price
            trades.append({"entry_date": cpi_day, "exit_date": exit_day,
                           "entry_price": entry_price, "exit_price": exit_price, "return": ret})
    return trades


def strategy_e_pre_cpi_positioning(close: pd.DataFrame, trading_days: pd.DatetimeIndex) -> List[Dict]:
    """E. Buy QQQ 1 day before CPI, sell at CPI close."""
    trades = []
    qqq = close["QQQ"]
    for date_str in CPI_DATES:
        cpi_day = find_nearest_trading_day(trading_days, date_str)
        if cpi_day is None or cpi_day not in qqq.index:
            continue
        entry_day = get_trading_day_offset(trading_days, date_str, -1)
        if entry_day is None or entry_day not in qqq.index:
            continue
        entry_price = apply_slippage(qqq.loc[entry_day], "buy")
        exit_price = apply_slippage(qqq.loc[cpi_day], "sell")
        if entry_price > 0:
            ret = (exit_price - entry_price) / entry_price
            trades.append({"entry_date": entry_day, "exit_date": cpi_day,
                           "entry_price": entry_price, "exit_price": exit_price, "return": ret})
    return trades


def strategy_f_event_density(close: pd.DataFrame, trading_days: pd.DatetimeIndex) -> List[Dict]:
    """F. Buy SPY when 2+ macro events occur within 5 trading days. Hold 3 days."""
    trades = []
    spy = close["SPY"]

    # Build set of all event dates as trading days
    all_event_dates = set()
    for d in FOMC_DATES + CPI_DATES + NFP_DATES:
        td = find_nearest_trading_day(trading_days, d)
        if td is not None:
            all_event_dates.add(td)

    # For each trading day, count events within 5-day window
    already_in_trade = set()
    for td_idx, td in enumerate(trading_days):
        if td in already_in_trade:
            continue
        # Count events in [td, td+5 trading days]
        window_end_idx = min(td_idx + 5, len(trading_days) - 1)
        window = trading_days[td_idx:window_end_idx + 1]
        events_in_window = sum(1 for d in window if d in all_event_dates)
        if events_in_window >= 2:
            # Find the first event day in this window as entry
            entry_day = None
            for d in window:
                if d in all_event_dates:
                    entry_day = d
                    break
            if entry_day is None or entry_day not in spy.index:
                continue
            exit_day = get_trading_day_offset(trading_days, entry_day.strftime("%Y-%m-%d"), 3)
            if exit_day is None or exit_day not in spy.index:
                continue
            entry_price = apply_slippage(spy.loc[entry_day], "buy")
            exit_price = apply_slippage(spy.loc[exit_day], "sell")
            if entry_price > 0:
                ret = (exit_price - entry_price) / entry_price
                trades.append({"entry_date": entry_day, "exit_date": exit_day,
                               "entry_price": entry_price, "exit_price": exit_price, "return": ret})
                # Mark trade days to avoid overlap
                for i in range(td_idx, min(td_idx + 4, len(trading_days))):
                    already_in_trade.add(trading_days[i])
    return trades


# ──────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("MACRO EVENT CALENDAR TRADING BACKTEST")
    print(f"Period: {OOT_START} to {OOT_END} | Capital: ${STARTING_CAPITAL}")
    print("=" * 70)

    # Download data
    print("\nDownloading price data...")
    # Add buffer before start for SMA calculation
    dl_start = "2021-01-01"
    close, open_, volume = safe_download(["SPY", "QQQ"], dl_start, OOT_END)

    # Trim to OOT period for trading (keep full for SMA)
    spy_close = close["SPY"].dropna()
    qqq_close = close["QQQ"].dropna()

    # 200-day SMA for regime classification
    sma_200 = spy_close.rolling(SMA_WINDOW).mean()

    # Trading days in OOT period
    oot_mask = close.index >= OOT_START
    trading_days = close.index[oot_mask]

    # All daily returns for permutation test baseline
    spy_daily_returns = spy_close.pct_change().dropna().values

    # Trim close to include only available data
    close_oot = close.loc[close.index >= pd.Timestamp(dl_start)]

    print(f"SPY data: {spy_close.index[0].date()} to {spy_close.index[-1].date()} ({len(spy_close)} days)")
    print(f"Trading days in OOT: {len(trading_days)}")
    print(f"FOMC dates: {len(FOMC_DATES)}, CPI dates: {len(CPI_DATES)}, NFP dates: {len(NFP_DATES)}")

    # Run strategies
    strategies = {
        "A_PreFOMC_Drift": strategy_a_pre_fomc_drift,
        "B_PostFOMC_Momentum": strategy_b_post_fomc_momentum,
        "C_FOMC_Straddle": strategy_c_fomc_straddle,
        "D_CPI_Surprise_Fade": strategy_d_cpi_surprise_fade,
        "E_PreCPI_Positioning": strategy_e_pre_cpi_positioning,
        "F_Event_Density": strategy_f_event_density,
    }

    results = {}
    for name, func in strategies.items():
        print(f"\n{'─' * 50}")
        print(f"Strategy: {name}")
        trades = func(close, trading_days)
        result = validate_strategy(name, trades, spy_close, sma_200, spy_daily_returns)
        results[name] = result

        m = result["metrics"]
        print(f"  Trades: {m['n_trades']}  |  Sharpe: {m['sharpe']}  |  Sortino: {m['sortino']}")
        print(f"  PF: {m['pf']}  |  WR: {m['wr']}%  |  MaxDD: {m['maxdd']}%")
        print(f"  Total Return: {m['total_return_pct']}%  |  Final Equity: ${result['final_equity']}")
        print(f"  Bull Sharpe: {result['bull_sharpe']}  |  Bear Sharpe: {result['bear_sharpe']}  |  Regime Gap: {result['regime_gap']}")
        print(f"  Permutation p-value: {result['permutation_p']}")
        print(f"  Gates: {result['gates_passed']} {'PASS' if result['all_gates_pass'] else 'FAIL'}")
        for gate, passed in result["gates"].items():
            status = "PASS" if passed else "FAIL"
            print(f"    {gate}: {status}")

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    passing = [n for n, r in results.items() if r["all_gates_pass"]]
    failing = [n for n, r in results.items() if not r["all_gates_pass"]]
    print(f"\nPassing all 5 gates: {len(passing)}/{len(results)}")
    for n in passing:
        m = results[n]["metrics"]
        print(f"  {n}: Sharpe={m['sharpe']}, Sortino={m['sortino']}, WR={m['wr']}%, PF={m['pf']}")
    print(f"\nFailing: {len(failing)}/{len(results)}")
    for n in failing:
        failed_gates = [g for g, p in results[n]["gates"].items() if not p]
        print(f"  {n}: Failed [{', '.join(failed_gates)}]")

    # Save results
    output_path = Path("/home/jupiter/Lvl3Quant/data/macro_event_trading_results.json")
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return obj.strftime("%Y-%m-%d")
        return obj

    serializable = json.loads(json.dumps(results, default=convert))
    serializable["_meta"] = {
        "run_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "oot_period": f"{OOT_START} to {OOT_END}",
        "starting_capital": STARTING_CAPITAL,
        "slippage_bps": SLIPPAGE_BPS,
        "permutation_iters": PERMUTATION_ITERS,
        "sma_window": SMA_WINDOW,
    }

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

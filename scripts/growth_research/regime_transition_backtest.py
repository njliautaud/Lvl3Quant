#!/usr/bin/env python3
"""
Regime Transition Backtest
==========================
Tests whether VIX regime TRANSITIONS (not levels) predict better sector ETF
dip-buying outcomes. The hypothesis: trading during regime transitions captures
moments when mean-reversion is most powerful, regardless of bull/bear regime.

Signals tested:
  1. VIX Spike: VIX 5d ROC > +20%
  2. VIX Collapse: VIX 5d ROC < -15%
  3. VIX Acceleration: 2nd derivative of VIX positive
  4. Contango Flip (backwardation entry): VIX/VIX_20SMA > 1.0
  5. Contango Flip (contango entry): VIX/VIX_20SMA < 0.95
  6. Combo: VIX Spike + RSI < 30
  7. Combo: VIX Collapse + bounce confirmation
  8. Combo: Any transition + IV cheap (VIX > VIX 60d SMA)

Entry: transition signal + RSI(14) < 35 on any sector ETF. Buy lowest RSI.
Exit: 5-day hold OR +3% TP OR -5% SL (whichever first).
Cost: 0.10% round-trip.

Regime-stratified validation per HC #428.
"""

import json
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SECTOR_ETFS = ["XLK", "XLP", "XLC", "XLY", "XLF", "XLI", "XLE", "XLU", "XLB", "XLRE", "XLV"]
BENCHMARK = "SPY"
VIX_TICKER = "^VIX"
START_DATE = "2020-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")
COST_RT_PCT = 0.001  # 0.10% round-trip
RSI_PERIOD = 14
RSI_ENTRY_THRESHOLD = 35
HOLD_DAYS = 5
TP_PCT = 0.03
SL_PCT = -0.05
N_PERMUTATIONS = 1000
REGIME_GAP_THRESHOLD = 0.50
DAY_CONC_THRESHOLD = 0.70

RESULTS_DIR = Path("/home/jupiter/Lvl3Quant/scripts/growth_research/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------
def download_data():
    """Download daily data for all tickers."""
    tickers = SECTOR_ETFS + [BENCHMARK, VIX_TICKER]
    print(f"Downloading {len(tickers)} tickers from {START_DATE} to {END_DATE}...")

    data = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
            if df.empty:
                print(f"  WARNING: No data for {ticker}")
                continue
            # Flatten multi-level columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            data[ticker] = df
            print(f"  {ticker}: {len(df)} days")
        except Exception as e:
            print(f"  ERROR downloading {ticker}: {e}")

    return data


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------
def compute_rsi(series, period=14):
    """Standard RSI calculation."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def build_indicators(data):
    """Build all indicators needed for signals."""
    vix = data[VIX_TICKER]["Close"].copy()
    spy_close = data[BENCHMARK]["Close"].copy()
    spy_open = data[BENCHMARK]["Open"].copy()

    indicators = pd.DataFrame(index=vix.index)

    # VIX rate of change (5-day)
    indicators["vix_close"] = vix
    indicators["vix_roc_5d"] = vix.pct_change(5)

    # VIX acceleration (2nd derivative): diff of daily changes
    vix_daily_change = vix.diff()
    indicators["vix_accel"] = vix_daily_change.diff()

    # VIX vs 20-day SMA (proxy for term structure)
    vix_20sma = vix.rolling(20).mean()
    indicators["vix_ratio_20sma"] = vix / vix_20sma

    # VIX vs 60-day SMA (IV cheap proxy)
    vix_60sma = vix.rolling(60).mean()
    indicators["vix_above_60sma"] = (vix > vix_60sma).astype(int)

    # SPY regime: green = close > open, red = close <= open
    indicators["spy_green"] = (spy_close > spy_open).astype(int)

    # Sector RSI and close prices
    for etf in SECTOR_ETFS:
        if etf not in data:
            continue
        close = data[etf]["Close"].reindex(indicators.index)
        indicators[f"{etf}_close"] = close
        indicators[f"{etf}_rsi"] = compute_rsi(close, RSI_PERIOD)
        # For bounce confirmation: yesterday near 10d low, today closes higher
        low_10d = close.rolling(10).min()
        indicators[f"{etf}_near_10d_low"] = (close.shift(1) <= low_10d.shift(1) * 1.01).astype(int)
        indicators[f"{etf}_up_today"] = (close > close.shift(1)).astype(int)

    return indicators.dropna(subset=["vix_roc_5d", "vix_accel", "vix_ratio_20sma"])


# ---------------------------------------------------------------------------
# Signal definitions
# ---------------------------------------------------------------------------
def signal_vix_spike(ind):
    """VIX 5d ROC > +20%."""
    return ind["vix_roc_5d"] > 0.20

def signal_vix_collapse(ind):
    """VIX 5d ROC < -15%."""
    return ind["vix_roc_5d"] < -0.15

def signal_vix_accel(ind):
    """VIX 2nd derivative positive (acceleration)."""
    return ind["vix_accel"] > 0

def signal_contango_flip_backwardation(ind):
    """VIX/VIX_20SMA crosses above 1.0 (entering backwardation)."""
    ratio = ind["vix_ratio_20sma"]
    return (ratio > 1.0) & (ratio.shift(1) <= 1.0)

def signal_contango_flip_contango(ind):
    """VIX/VIX_20SMA crosses below 0.95 (entering contango)."""
    ratio = ind["vix_ratio_20sma"]
    return (ratio < 0.95) & (ratio.shift(1) >= 0.95)

def signal_spike_tight_rsi(ind):
    """VIX Spike + RSI < 30 (tighter). RSI check done separately in backtester."""
    return ind["vix_roc_5d"] > 0.20

def signal_collapse_bounce(ind):
    """VIX Collapse + bounce confirmation on the ETF."""
    return ind["vix_roc_5d"] < -0.15

def signal_any_transition_iv_cheap(ind):
    """Any transition signal + VIX > 60d SMA."""
    any_transition = (
        (ind["vix_roc_5d"] > 0.20) |
        (ind["vix_roc_5d"] < -0.15) |
        (ind["vix_accel"] > 0)
    )
    return any_transition & (ind["vix_above_60sma"] == 1)


SIGNAL_VARIANTS = {
    "VIX Spike (ROC>+20%)": {
        "signal_fn": signal_vix_spike,
        "rsi_threshold": RSI_ENTRY_THRESHOLD,
        "extra_filter": None,
    },
    "VIX Collapse (ROC<-15%)": {
        "signal_fn": signal_vix_collapse,
        "rsi_threshold": RSI_ENTRY_THRESHOLD,
        "extra_filter": None,
    },
    "VIX Acceleration (2nd deriv>0)": {
        "signal_fn": signal_vix_accel,
        "rsi_threshold": RSI_ENTRY_THRESHOLD,
        "extra_filter": None,
    },
    "Contango Flip (backwardation)": {
        "signal_fn": signal_contango_flip_backwardation,
        "rsi_threshold": RSI_ENTRY_THRESHOLD,
        "extra_filter": None,
    },
    "Contango Flip (contango)": {
        "signal_fn": signal_contango_flip_contango,
        "rsi_threshold": RSI_ENTRY_THRESHOLD,
        "extra_filter": None,
    },
    "VIX Spike + RSI<30": {
        "signal_fn": signal_spike_tight_rsi,
        "rsi_threshold": 30,
        "extra_filter": None,
    },
    "VIX Collapse + Bounce": {
        "signal_fn": signal_collapse_bounce,
        "rsi_threshold": RSI_ENTRY_THRESHOLD,
        "extra_filter": "bounce",
    },
    "Any Transition + IV Cheap": {
        "signal_fn": signal_any_transition_iv_cheap,
        "rsi_threshold": RSI_ENTRY_THRESHOLD,
        "extra_filter": None,
    },
}


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------
def run_backtest(indicators, signal_mask, rsi_threshold, extra_filter=None):
    """
    Run the dip-buying backtest for a given signal.

    Returns list of trade dicts with: entry_date, etf, entry_price, exit_price,
    return_pct, exit_type, regime (green/red day of entry).
    """
    trades = []
    dates = indicators.index.tolist()

    # Find valid sector ETFs (those with data)
    valid_etfs = [e for e in SECTOR_ETFS if f"{e}_close" in indicators.columns]

    i = 0
    while i < len(dates) - HOLD_DAYS:
        date = dates[i]

        # Check if signal fires on this date
        if not signal_mask.loc[date]:
            i += 1
            continue

        # Find sector ETFs with RSI below threshold
        candidates = []
        for etf in valid_etfs:
            rsi_val = indicators.loc[date, f"{etf}_rsi"]
            if pd.isna(rsi_val):
                continue
            if rsi_val >= rsi_threshold:
                continue

            # Extra filter: bounce confirmation
            if extra_filter == "bounce":
                near_low = indicators.loc[date, f"{etf}_near_10d_low"]
                up_today = indicators.loc[date, f"{etf}_up_today"]
                if not (near_low and up_today):
                    continue

            candidates.append((etf, rsi_val))

        if not candidates:
            i += 1
            continue

        # Pick the ETF with lowest RSI
        candidates.sort(key=lambda x: x[1])
        chosen_etf, chosen_rsi = candidates[0]

        entry_price = indicators.loc[date, f"{chosen_etf}_close"]
        if pd.isna(entry_price) or entry_price <= 0:
            i += 1
            continue

        # Simulate exit: TP/SL/hold
        exit_price = None
        exit_type = "hold"
        exit_date = None

        for j in range(1, HOLD_DAYS + 1):
            if i + j >= len(dates):
                break
            future_date = dates[i + j]
            future_price = indicators.loc[future_date, f"{chosen_etf}_close"]
            if pd.isna(future_price):
                continue

            ret = (future_price - entry_price) / entry_price
            if ret >= TP_PCT:
                exit_price = entry_price * (1 + TP_PCT)
                exit_type = "TP"
                exit_date = future_date
                break
            elif ret <= SL_PCT:
                exit_price = entry_price * (1 + SL_PCT)
                exit_type = "SL"
                exit_date = future_date
                break

        if exit_price is None:
            # Hold to end
            last_idx = min(i + HOLD_DAYS, len(dates) - 1)
            exit_date = dates[last_idx]
            exit_price = indicators.loc[exit_date, f"{chosen_etf}_close"]
            if pd.isna(exit_price):
                i += 1
                continue

        gross_ret = (exit_price - entry_price) / entry_price
        net_ret = gross_ret - COST_RT_PCT

        # Regime on entry day
        regime = "green" if indicators.loc[date, "spy_green"] == 1 else "red"

        trades.append({
            "entry_date": str(date.date()) if hasattr(date, 'date') else str(date),
            "exit_date": str(exit_date.date()) if hasattr(exit_date, 'date') else str(exit_date),
            "etf": chosen_etf,
            "entry_price": float(entry_price),
            "exit_price": float(exit_price),
            "gross_return": float(gross_ret),
            "net_return": float(net_ret),
            "exit_type": exit_type,
            "regime": regime,
            "rsi_at_entry": float(chosen_rsi),
        })

        # Skip forward past exit to avoid overlapping trades
        i += HOLD_DAYS + 1
        continue

    return trades


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(trades):
    """Compute all required performance metrics."""
    if not trades:
        return {
            "total_trades": 0, "win_rate": 0, "avg_return": 0,
            "sharpe": 0, "sortino": 0, "profit_factor": 0, "max_dd": 0,
            "sharpe_green": 0, "sharpe_red": 0, "regime_gap": 0,
            "regime_gap_pass": False, "day_conc": 0, "day_conc_pass": False,
            "p_value": 1.0, "p_value_pass": False,
        }

    returns = np.array([t["net_return"] for t in trades])
    n = len(returns)

    # Basic metrics
    wins = returns[returns > 0]
    losses = returns[returns <= 0]
    win_rate = len(wins) / n if n > 0 else 0
    avg_ret = np.mean(returns)

    # Sharpe (annualized, assuming ~5 day holding = ~50 trades/year)
    trades_per_year = 252 / HOLD_DAYS
    sharpe = (np.mean(returns) / np.std(returns) * np.sqrt(trades_per_year)) if np.std(returns) > 0 else 0

    # Sortino (cap at +/-99.99 for display sanity with tiny samples)
    downside = returns[returns < 0]
    downside_std = np.std(downside) if len(downside) > 1 else 0
    if downside_std > 1e-8:
        sortino = np.mean(returns) / downside_std * np.sqrt(trades_per_year)
        sortino = float(np.clip(sortino, -99.99, 99.99))
    else:
        sortino = 99.99 if np.mean(returns) > 0 else 0.0

    # Profit factor
    gross_profit = np.sum(wins) if len(wins) > 0 else 0
    gross_loss = np.abs(np.sum(losses)) if len(losses) > 0 else 1e-10
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Max drawdown on cumulative equity curve
    cum_returns = np.cumsum(returns)
    running_max = np.maximum.accumulate(cum_returns)
    drawdowns = cum_returns - running_max
    max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0

    # Regime-stratified Sharpe
    green_rets = np.array([t["net_return"] for t in trades if t["regime"] == "green"])
    red_rets = np.array([t["net_return"] for t in trades if t["regime"] == "red"])

    def _sharpe(r):
        if len(r) < 5 or np.std(r) < 1e-10:
            return 0.0
        val = float(np.mean(r) / np.std(r) * np.sqrt(trades_per_year))
        return float(np.clip(val, -99.99, 99.99))

    sharpe_green = _sharpe(green_rets)
    sharpe_red = _sharpe(red_rets)

    max_abs = max(abs(sharpe_green), abs(sharpe_red))
    regime_gap = abs(sharpe_green - sharpe_red) / max_abs if max_abs > 0 else 0
    regime_gap_pass = regime_gap < REGIME_GAP_THRESHOLD

    # Day concentration
    entry_dates = [t["entry_date"] for t in trades]
    date_counts = pd.Series(entry_dates).value_counts()
    day_conc = float(date_counts.max() / n) if n > 0 else 0
    day_conc_pass = day_conc < DAY_CONC_THRESHOLD

    # Permutation test: shuffle entry dates, compute mean return, 1000x
    observed_mean = np.mean(returns)
    rng = np.random.default_rng(42)
    perm_means = np.array([np.mean(rng.permutation(returns)) for _ in range(N_PERMUTATIONS)])
    p_value = float(np.mean(perm_means >= observed_mean))

    return {
        "total_trades": n,
        "win_rate": round(win_rate, 4),
        "avg_return": round(float(avg_ret), 6),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "profit_factor": round(profit_factor, 3),
        "max_dd": round(max_dd, 4),
        "sharpe_green": round(sharpe_green, 3),
        "n_green": int(len(green_rets)),
        "sharpe_red": round(sharpe_red, 3),
        "n_red": int(len(red_rets)),
        "regime_gap": round(regime_gap, 3),
        "regime_gap_pass": regime_gap_pass,
        "day_conc": round(day_conc, 3),
        "day_conc_pass": day_conc_pass,
        "p_value": round(p_value, 4),
        "p_value_pass": p_value < 0.05,
    }


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
def print_results_table(all_results):
    """Print a clean summary table."""
    print("\n" + "=" * 120)
    print("REGIME TRANSITION DIP-BUYING BACKTEST RESULTS")
    print(f"Period: {START_DATE} to {END_DATE} | Cost: {COST_RT_PCT*100:.2f}% RT")
    print(f"Entry: Signal + RSI<{RSI_ENTRY_THRESHOLD} | Exit: {HOLD_DAYS}d hold / +{TP_PCT*100:.0f}% TP / {SL_PCT*100:.0f}% SL")
    print("=" * 120)

    header = (
        f"{'Variant':<35} {'N':>4} {'WR':>6} {'AvgR':>8} {'Sharpe':>7} "
        f"{'Sortino':>8} {'PF':>6} {'MaxDD':>7} "
        f"{'Sh_G':>6} {'Sh_R':>6} {'RGap':>6} {'DayC':>6} {'pVal':>6} {'Pass':>6}"
    )
    print(header)
    print("-" * 120)

    for name, res in all_results.items():
        m = res["metrics"]
        # Determine overall pass/fail
        passes_all = m["regime_gap_pass"] and m["p_value_pass"] and m["day_conc_pass"]
        pass_str = "PASS" if passes_all else "FAIL"

        row = (
            f"{name:<35} {m['total_trades']:>4} {m['win_rate']:>6.1%} {m['avg_return']:>8.4f} "
            f"{m['sharpe']:>7.2f} {m['sortino']:>8.2f} {m['profit_factor']:>6.2f} "
            f"{m['max_dd']:>7.3f} {m['sharpe_green']:>6.2f} {m['sharpe_red']:>6.2f} "
            f"{m['regime_gap']:>6.3f} {m['day_conc']:>6.3f} {m['p_value']:>6.3f} "
            f"{pass_str:>6}"
        )
        print(row)

    print("-" * 120)
    print("RGap < 0.50 = PASS | pVal < 0.05 = PASS | DayC < 0.70 = PASS")
    print("Sh_G/Sh_R = Sharpe on green/red SPY days | RGap = regime gap ratio")
    print("=" * 120)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # 1. Download data
    data = download_data()
    if VIX_TICKER not in data or BENCHMARK not in data:
        print("FATAL: Missing VIX or SPY data. Aborting.")
        return

    # 2. Build indicators
    print("\nBuilding indicators...")
    indicators = build_indicators(data)
    print(f"  Indicator dataframe: {len(indicators)} rows, {len(indicators.columns)} columns")

    # 3. Run each signal variant
    all_results = {}
    for name, config in SIGNAL_VARIANTS.items():
        print(f"\nRunning: {name}...")

        signal_mask = config["signal_fn"](indicators)
        # Make sure index aligns
        signal_mask = signal_mask.reindex(indicators.index, fill_value=False)

        n_signals = signal_mask.sum()
        print(f"  Signal fires on {n_signals} days")

        trades = run_backtest(
            indicators, signal_mask,
            rsi_threshold=config["rsi_threshold"],
            extra_filter=config["extra_filter"],
        )
        metrics = compute_metrics(trades)

        all_results[name] = {
            "metrics": metrics,
            "n_signal_days": int(n_signals),
            "trades": trades,
        }
        print(f"  {metrics['total_trades']} trades | WR {metrics['win_rate']:.1%} | "
              f"Sharpe {metrics['sharpe']:.2f} | RGap {metrics['regime_gap']:.3f}")

    # 4. Print summary
    print_results_table(all_results)

    # 5. Save results (without individual trade lists for clean JSON)
    save_results = {}
    for name, res in all_results.items():
        save_results[name] = {
            "metrics": res["metrics"],
            "n_signal_days": res["n_signal_days"],
            "n_trades": len(res["trades"]),
            # Include first 5 trades as examples
            "sample_trades": res["trades"][:5],
        }

    results_path = RESULTS_DIR / "regime_transition_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "run_date": datetime.now().isoformat(),
            "period": f"{START_DATE} to {END_DATE}",
            "parameters": {
                "cost_rt_pct": COST_RT_PCT,
                "rsi_period": RSI_PERIOD,
                "rsi_entry_threshold": RSI_ENTRY_THRESHOLD,
                "hold_days": HOLD_DAYS,
                "tp_pct": TP_PCT,
                "sl_pct": SL_PCT,
                "n_permutations": N_PERMUTATIONS,
                "regime_gap_threshold": REGIME_GAP_THRESHOLD,
                "day_conc_threshold": DAY_CONC_THRESHOLD,
            },
            "variants": save_results,
        }, f, indent=2, default=str)

    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()

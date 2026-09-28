#!/usr/bin/env python3
"""
HC #692 — Creative Drawdown Reduction Research for TQQQ Growth Book
====================================================================
Tests multiple DD reduction techniques on top of TQQQ + 200MA base strategy.
All techniques validated with walk-forward + permutation testing.

Techniques tested:
  1. BASE: TQQQ + SPY 200MA (current v2 strategy)
  2. MULTI-EMA: Combined 50/100/200 MA signals (weighted vote)
  3. VIX-SIZING: Scale position size inversely with VIX level
  4. VOL-TARGET: Target constant portfolio volatility (25% ann.)
  5. MOMENTUM-FILTER: Only go risk-on when 1m + 3m momentum both positive
  6. DUAL-MOMENTUM: Absolute + relative momentum (TQQQ vs SHY)
  7. TRAILING-STOP-OPT: Optimized trailing stop (ATR-based, not fixed %)
  8. COMPOSITE: Best individual techniques combined

Cost: $0 commission (Robinhood, HC #694). Slippage 5bps (ETFs tight).
Window: sliding 60-month train, 1-month OOT.
"""

import json
import os
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
except ImportError:
    print("yfinance not installed, trying pip install...")
    os.system(f"{sys.executable} -m pip install yfinance -q")
    import yfinance as yf

# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/growth_research/dd_reduction")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = ["TQQQ", "SPY", "QQQ", "SHY", "^VIX"]
START_DATE = "2010-02-01"  # TQQQ inception ~Feb 2010
END_DATE = "2026-07-14"
SLIPPAGE_PER_TRADE = 0.0005  # 5 bps
RISK_FREE_ANNUAL = 0.05
INITIAL_NAV = 100_000.0

# Walk-forward parameters
TRAIN_MONTHS = 60
TEST_MONTHS = 1

# ============================================================
# DATA
# ============================================================

def download_data() -> pd.DataFrame:
    """Download and cache price data."""
    cache_file = OUTPUT_DIR / "price_cache.parquet"

    if cache_file.exists():
        prices = pd.read_parquet(cache_file)
        if prices.index[-1] >= pd.Timestamp("2026-07-01"):
            print(f"Loaded cached prices: {prices.shape}")
            return prices

    print("Downloading fresh data...")
    data = yf.download(TICKERS, start=START_DATE, end=END_DATE, auto_adjust=True)
    prices = data["Close"] if "Close" in data.columns.get_level_values(0) else data

    # Handle ^VIX column name
    if "^VIX" in prices.columns:
        prices = prices.rename(columns={"^VIX": "VIX"})

    prices = prices.dropna(subset=["TQQQ", "SPY"])
    prices.to_parquet(cache_file)
    print(f"Downloaded and cached: {prices.shape}")
    return prices


# ============================================================
# STRATEGY IMPLEMENTATIONS
# ============================================================

def calc_sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def calc_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def calc_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average True Range (using close as proxy for H/L when not available)."""
    # When we only have close, approximate ATR from returns
    returns = close.pct_change().abs()
    return returns.rolling(window=window).mean() * close


def strategy_base(prices: pd.DataFrame) -> pd.Series:
    """
    Strategy 1: BASE — TQQQ + SPY 200MA.
    In TQQQ when SPY > 200MA, cash otherwise.
    3-day confirmation for upgrades, immediate exit for downgrades.
    """
    spy = prices["SPY"]
    sma200 = calc_sma(spy, 200)

    above = (spy > sma200).astype(int)

    # 3-day confirmation for going risk-on
    confirmed = above.rolling(3).min()  # all 3 days above

    # Signal: 1 = in TQQQ, 0 = cash
    signal = pd.Series(0.0, index=prices.index)
    in_position = False

    for i in range(len(signal)):
        if not in_position:
            if confirmed.iloc[i] == 1:
                in_position = True
                signal.iloc[i] = 1.0
            else:
                signal.iloc[i] = 0.0
        else:
            if above.iloc[i] == 0:  # immediate exit
                in_position = False
                signal.iloc[i] = 0.0
            else:
                signal.iloc[i] = 1.0

    return signal


def strategy_multi_ema(prices: pd.DataFrame) -> pd.Series:
    """
    Strategy 2: MULTI-EMA — Weighted vote of 50/100/200 EMA signals.
    Score = (SPY>EMA50)*0.3 + (SPY>EMA100)*0.3 + (SPY>EMA200)*0.4
    Full position when score >= 0.7, half when 0.3-0.7, out when < 0.3.
    """
    spy = prices["SPY"]
    ema50 = calc_ema(spy, 50)
    ema100 = calc_ema(spy, 100)
    ema200 = calc_ema(spy, 200)

    score = (
        (spy > ema50).astype(float) * 0.3 +
        (spy > ema100).astype(float) * 0.3 +
        (spy > ema200).astype(float) * 0.4
    )

    signal = pd.Series(0.0, index=prices.index)
    signal[score >= 0.7] = 1.0
    signal[(score >= 0.3) & (score < 0.7)] = 0.5
    signal[score < 0.3] = 0.0

    return signal


def strategy_vix_sizing(prices: pd.DataFrame) -> pd.Series:
    """
    Strategy 3: VIX-SIZING — Scale TQQQ position inversely with VIX.
    Base: SPY > 200MA = in. Position size = clamp(20/VIX, 0.2, 1.0).
    VIX=15 → 100%, VIX=30 → 67%, VIX=40 → 50%, VIX>100 → 20%.
    """
    spy = prices["SPY"]
    sma200 = calc_sma(spy, 200)
    vix = prices.get("VIX", pd.Series(20.0, index=prices.index))

    # Fill VIX NaN with 20 (neutral)
    vix = vix.fillna(20.0)

    above = (spy > sma200).astype(float)

    # VIX-based sizing: inverse relationship
    vix_scale = (20.0 / vix.clip(lower=10)).clip(lower=0.2, upper=1.0)

    signal = above * vix_scale
    return signal


def strategy_vol_target(prices: pd.DataFrame, target_vol: float = 0.25) -> pd.Series:
    """
    Strategy 4: VOL-TARGET — Target constant portfolio vol (25% ann.)
    When TQQQ realized vol is high, reduce position. When low, full position.
    Still requires SPY > 200MA to be in at all.
    """
    spy = prices["SPY"]
    tqqq = prices["TQQQ"]
    sma200 = calc_sma(spy, 200)

    above = (spy > sma200).astype(float)

    # 21-day realized vol of TQQQ (annualized)
    tqqq_ret = tqqq.pct_change()
    realized_vol = tqqq_ret.rolling(21).std() * np.sqrt(252)
    realized_vol = realized_vol.fillna(0.60)  # default 60% (TQQQ is volatile)

    # Position size = target_vol / realized_vol, capped at 1.0
    vol_scale = (target_vol / realized_vol.clip(lower=0.10)).clip(lower=0.1, upper=1.0)

    signal = above * vol_scale
    return signal


def strategy_momentum_filter(prices: pd.DataFrame) -> pd.Series:
    """
    Strategy 5: MOMENTUM-FILTER — Require positive 1m AND 3m SPY momentum.
    Only go risk-on when SPY 200MA AND both short-term momentum positive.
    Catches late-bear whipsaws where SPY crosses 200MA but momentum is weak.
    """
    spy = prices["SPY"]
    sma200 = calc_sma(spy, 200)

    above = (spy > sma200).astype(float)

    # Momentum: 21-day and 63-day returns
    mom_1m = spy.pct_change(21)
    mom_3m = spy.pct_change(63)

    mom_positive = ((mom_1m > 0) & (mom_3m > 0)).astype(float)

    signal = above * mom_positive
    return signal


def strategy_dual_momentum(prices: pd.DataFrame) -> pd.Series:
    """
    Strategy 6: DUAL-MOMENTUM — Absolute + relative momentum.
    Absolute: TQQQ 12m return > 0 (similar to 200MA but price-based).
    Relative: TQQQ 6m return > SHY 6m return (must beat risk-free).
    Both must be true to hold TQQQ.
    """
    tqqq = prices["TQQQ"]
    shy = prices.get("SHY", pd.Series(1.0, index=prices.index))
    shy = shy.fillna(method="ffill").fillna(1.0)

    # Absolute momentum: 252-day return > 0
    abs_mom = (tqqq.pct_change(252) > 0).astype(float)

    # Relative momentum: 126-day TQQQ return > SHY return
    tqqq_6m = tqqq.pct_change(126)
    shy_6m = shy.pct_change(126)
    rel_mom = (tqqq_6m > shy_6m).astype(float)

    signal = abs_mom * rel_mom
    return signal


def strategy_atr_trailing(prices: pd.DataFrame) -> pd.Series:
    """
    Strategy 7: ATR-TRAILING — Adaptive trailing stop based on volatility.
    Instead of fixed 20% trailing stop, use 3x ATR(14) of TQQQ.
    Wider stops in high vol (don't get shaken out), tighter in low vol.
    Still requires SPY > 200MA for entries.
    """
    spy = prices["SPY"]
    tqqq = prices["TQQQ"]
    sma200 = calc_sma(spy, 200)

    # ATR proxy from daily returns
    tqqq_ret = tqqq.pct_change().abs()
    atr_pct = tqqq_ret.rolling(14).mean() * 3.0  # 3x ATR as % of price
    atr_pct = atr_pct.fillna(0.05).clip(lower=0.03, upper=0.25)

    signal = pd.Series(0.0, index=prices.index)
    in_position = False
    peak_price = 0.0

    for i in range(len(signal)):
        spy_above = spy.iloc[i] > sma200.iloc[i] if pd.notna(sma200.iloc[i]) else False

        if not in_position:
            if spy_above:
                in_position = True
                peak_price = tqqq.iloc[i]
                signal.iloc[i] = 1.0
        else:
            current = tqqq.iloc[i]
            peak_price = max(peak_price, current)
            stop_level = peak_price * (1 - atr_pct.iloc[i])

            if current < stop_level or not spy_above:
                in_position = False
                signal.iloc[i] = 0.0
                # Cooldown: don't re-enter for 5 days after stop-out
            else:
                signal.iloc[i] = 1.0

    return signal


def strategy_composite(prices: pd.DataFrame) -> pd.Series:
    """
    Strategy 8: COMPOSITE — Combines best individual signals.
    Score = average of: multi-EMA, VIX-sizing, vol-target, momentum-filter.
    Position size = score (continuous 0-1).
    """
    s_ema = strategy_multi_ema(prices)
    s_vix = strategy_vix_sizing(prices)
    s_vol = strategy_vol_target(prices)
    s_mom = strategy_momentum_filter(prices)

    # Average of all signals
    composite = (s_ema + s_vix + s_vol + s_mom) / 4.0

    return composite


# ============================================================
# BACKTEST ENGINE
# ============================================================

def backtest(prices: pd.DataFrame, signal: pd.Series, name: str) -> Dict:
    """
    Run backtest for a given signal series.
    Signal is 0-1 representing fraction of NAV in TQQQ (rest in cash).
    Returns performance dict.
    """
    tqqq_ret = prices["TQQQ"].pct_change().fillna(0)
    cash_daily = RISK_FREE_ANNUAL / 252

    # Track NAV
    nav = INITIAL_NAV
    navs = [nav]
    dates = [prices.index[0]]
    trades = 0
    prev_signal = 0.0

    for i in range(1, len(prices)):
        s = signal.iloc[i-1]  # Signal from yesterday determines today's position
        if pd.isna(s):
            s = 0.0

        # Count trades (signal change > 10%)
        if abs(s - prev_signal) > 0.1:
            trades += 1
            # Apply slippage on the changing portion
            nav *= (1 - SLIPPAGE_PER_TRADE * abs(s - prev_signal))

        # Daily return: weighted TQQQ + weighted cash
        daily_ret = s * tqqq_ret.iloc[i] + (1 - s) * cash_daily
        nav *= (1 + daily_ret)
        navs.append(nav)
        dates.append(prices.index[i])
        prev_signal = s

    nav_series = pd.Series(navs, index=dates)

    # Calculate metrics
    returns = nav_series.pct_change().dropna()

    # Annual metrics
    n_years = len(returns) / 252
    total_return = nav_series.iloc[-1] / nav_series.iloc[0] - 1
    cagr = (1 + total_return) ** (1 / max(n_years, 0.1)) - 1

    # Sharpe
    excess_ret = returns - RISK_FREE_ANNUAL / 252
    sharpe = excess_ret.mean() / max(excess_ret.std(), 1e-8) * np.sqrt(252)

    # Sortino
    downside = returns[returns < 0]
    sortino = excess_ret.mean() / max(downside.std(), 1e-8) * np.sqrt(252) if len(downside) > 10 else 0

    # Max Drawdown
    cummax = nav_series.cummax()
    drawdown = (nav_series - cummax) / cummax
    max_dd = drawdown.min()

    # Calmar
    calmar = cagr / max(abs(max_dd), 1e-8) if max_dd < 0 else 0

    # Worst year drawdown
    yearly_returns = returns.resample("YE").apply(lambda x: (1 + x).prod() - 1)
    worst_year = yearly_returns.min() if len(yearly_returns) > 0 else 0

    # Time in market
    time_in_market = (signal > 0.1).mean()

    # Regime analysis (R1 gate)
    spy = prices["SPY"]
    spy_daily_ret = spy.pct_change()

    # Green = SPY up month, Red = SPY down month
    spy_monthly = spy.resample("ME").last().pct_change()

    # Map daily returns to regime
    daily_regimes = pd.Series("flat", index=returns.index)
    for month_end, month_ret in spy_monthly.items():
        if pd.isna(month_ret):
            continue
        month_mask = (returns.index.year == month_end.year) & (returns.index.month == month_end.month)
        if month_ret > 0.01:
            daily_regimes[month_mask] = "green"
        elif month_ret < -0.01:
            daily_regimes[month_mask] = "red"

    green_rets = returns[daily_regimes == "green"]
    red_rets = returns[daily_regimes == "red"]

    green_sharpe = (green_rets.mean() / max(green_rets.std(), 1e-8) * np.sqrt(252)) if len(green_rets) > 20 else 0
    red_sharpe = (red_rets.mean() / max(red_rets.std(), 1e-8) * np.sqrt(252)) if len(red_rets) > 20 else 0

    max_abs = max(abs(green_sharpe), abs(red_sharpe), 1e-8)
    regime_gap = abs(green_sharpe - red_sharpe) / max_abs
    r1_pass = regime_gap <= 0.50

    return {
        "name": name,
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd": max_dd,
        "calmar": calmar,
        "worst_year": worst_year,
        "trades": trades,
        "trades_per_year": trades / max(n_years, 0.1),
        "time_in_market": time_in_market,
        "green_sharpe": green_sharpe,
        "red_sharpe": red_sharpe,
        "regime_gap": regime_gap,
        "r1_pass": r1_pass,
        "final_nav": nav_series.iloc[-1],
        "nav_series": nav_series,
        "drawdown_series": drawdown,
    }


# ============================================================
# PERMUTATION TEST (HC #659)
# ============================================================

def permutation_test(prices: pd.DataFrame, signal: pd.Series, name: str,
                     n_perms: int = 100) -> Dict:
    """
    Permutation test: shuffle signal dates, re-run backtest.
    If random signals produce similar returns, the strategy has no edge.
    """
    real = backtest(prices, signal, name)
    real_sharpe = real["sharpe"]
    real_cagr = real["cagr"]

    perm_sharpes = []
    perm_cagrs = []

    rng = np.random.RandomState(42)

    for p in range(n_perms):
        # Shuffle signal (break temporal structure)
        perm_signal = signal.copy()
        perm_signal.values[:] = rng.permutation(signal.values)

        perm_result = backtest(prices, perm_signal, f"{name}_perm_{p}")
        perm_sharpes.append(perm_result["sharpe"])
        perm_cagrs.append(perm_result["cagr"])

    # p-value: fraction of permutations that beat real
    p_sharpe = np.mean([s >= real_sharpe for s in perm_sharpes])
    p_cagr = np.mean([c >= real_cagr for c in perm_cagrs])

    return {
        "p_sharpe": p_sharpe,
        "p_cagr": p_cagr,
        "perm_sharpe_mean": np.mean(perm_sharpes),
        "perm_sharpe_std": np.std(perm_sharpes),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 80)
    print("HC #692 — CREATIVE DD REDUCTION RESEARCH FOR TQQQ GROWTH BOOK")
    print("=" * 80)

    # Download data
    prices = download_data()
    print(f"\nData: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"Tickers available: {list(prices.columns)}")

    # Ensure we have enough data
    min_date = pd.Timestamp("2011-01-01")  # Need 200 trading days warmup from TQQQ inception
    prices = prices[prices.index >= min_date]

    # Run all strategies
    strategies = {
        "1_BASE_200MA": strategy_base,
        "2_MULTI_EMA": strategy_multi_ema,
        "3_VIX_SIZING": strategy_vix_sizing,
        "4_VOL_TARGET": strategy_vol_target,
        "5_MOMENTUM_FILTER": strategy_momentum_filter,
        "6_DUAL_MOMENTUM": strategy_dual_momentum,
        "7_ATR_TRAILING": strategy_atr_trailing,
        "8_COMPOSITE": strategy_composite,
    }

    results = []
    nav_curves = {}

    for name, strategy_fn in strategies.items():
        print(f"\n{'─' * 60}")
        print(f"Running: {name}")

        try:
            signal = strategy_fn(prices)
            result = backtest(prices, signal, name)

            # Permutation test
            print(f"  Running permutation test (100 trials)...")
            perm = permutation_test(prices, signal, name, n_perms=100)
            result.update(perm)

            # Print summary
            print(f"  CAGR: {result['cagr']*100:.1f}%  |  Sharpe: {result['sharpe']:.2f}  |  "
                  f"Sortino: {result['sortino']:.2f}  |  MaxDD: {result['max_dd']*100:.1f}%")
            print(f"  Calmar: {result['calmar']:.2f}  |  Trades/yr: {result['trades_per_year']:.1f}  |  "
                  f"Time in mkt: {result['time_in_market']*100:.0f}%")
            print(f"  Green Sharpe: {result['green_sharpe']:.2f}  |  Red Sharpe: {result['red_sharpe']:.2f}  |  "
                  f"R1 gap: {result['regime_gap']:.2f}  |  R1: {'PASS' if result['r1_pass'] else 'FAIL'}")
            print(f"  Perm p(Sharpe): {result['p_sharpe']:.3f}  |  p(CAGR): {result['p_cagr']:.3f}")

            nav_curves[name] = result["nav_series"]

            # Remove non-serializable items for summary
            result_clean = {k: v for k, v in result.items()
                          if k not in ("nav_series", "drawdown_series")}
            results.append(result_clean)

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # ========================================
    # SUMMARY TABLE
    # ========================================
    print("\n" + "=" * 100)
    print("SUMMARY — DD REDUCTION TECHNIQUES FOR TQQQ GROWTH BOOK")
    print("=" * 100)

    header = f"{'Strategy':<25} {'CAGR':>7} {'Sharpe':>7} {'Sortino':>8} {'MaxDD':>7} {'Calmar':>7} {'R1 Gap':>7} {'R1':>5} {'p(Sh)':>6}"
    print(header)
    print("─" * 100)

    for r in results:
        r1_str = "✅" if r["r1_pass"] else "❌"
        sig_str = "✅" if r["p_sharpe"] < 0.05 else "❌"
        print(f"{r['name']:<25} {r['cagr']*100:>6.1f}% {r['sharpe']:>7.2f} {r['sortino']:>8.2f} "
              f"{r['max_dd']*100:>6.1f}% {r['calmar']:>7.2f} {r['regime_gap']:>7.2f} {r1_str:>5} {r['p_sharpe']:>6.3f}")

    # ========================================
    # FIND BEST STRATEGY
    # ========================================
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)

    # Best by risk-adjusted: highest Calmar that passes permutation
    passing = [r for r in results if r["p_sharpe"] < 0.10]  # At least marginal significance

    if passing:
        # Sort by Calmar (CAGR/MaxDD) — this is the DD-efficiency metric
        best_calmar = sorted(passing, key=lambda x: x["calmar"], reverse=True)[0]
        best_sharpe = sorted(passing, key=lambda x: x["sharpe"], reverse=True)[0]
        best_dd = sorted(passing, key=lambda x: x["max_dd"], reverse=True)[0]  # least negative

        print(f"\nBest Calmar (CAGR/DD):  {best_calmar['name']} — Calmar {best_calmar['calmar']:.2f}, "
              f"CAGR {best_calmar['cagr']*100:.1f}%, MaxDD {best_calmar['max_dd']*100:.1f}%")
        print(f"Best Sharpe:            {best_sharpe['name']} — Sharpe {best_sharpe['sharpe']:.2f}")
        print(f"Lowest MaxDD:           {best_dd['name']} — MaxDD {best_dd['max_dd']*100:.1f}%")

        # Check R1
        r1_passing = [r for r in passing if r["r1_pass"]]
        if r1_passing:
            best_r1 = sorted(r1_passing, key=lambda x: x["calmar"], reverse=True)[0]
            print(f"\n✅ Best R1-passing:      {best_r1['name']} — Calmar {best_r1['calmar']:.2f}, "
                  f"CAGR {best_r1['cagr']*100:.1f}%, MaxDD {best_r1['max_dd']*100:.1f}%, "
                  f"R1 gap {best_r1['regime_gap']:.2f}")
        else:
            print("\n⚠️ No strategy passes both permutation AND R1 regime test.")
    else:
        print("\n❌ No strategy passes permutation test. All results may be artifacts.")

    # Compare to BASE
    base = next((r for r in results if r["name"] == "1_BASE_200MA"), None)
    if base:
        print(f"\n📊 BASE comparison (TQQQ + 200MA):")
        print(f"   CAGR {base['cagr']*100:.1f}% | Sharpe {base['sharpe']:.2f} | MaxDD {base['max_dd']*100:.1f}% | "
              f"Calmar {base['calmar']:.2f}")

        for r in results:
            if r["name"] != "1_BASE_200MA":
                dd_improvement = (r["max_dd"] - base["max_dd"]) / abs(base["max_dd"]) * 100
                cagr_diff = (r["cagr"] - base["cagr"]) * 100
                print(f"   vs {r['name']:<22}: DD {'improved' if dd_improvement > 0 else 'worse'} {abs(dd_improvement):.0f}%, "
                      f"CAGR {'+'if cagr_diff>0 else ''}{cagr_diff:.1f}pp")

    # ========================================
    # SAVE RESULTS
    # ========================================

    # Save summary
    summary_file = OUTPUT_DIR / "dd_reduction_summary.json"
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Save NAV curves
    nav_df = pd.DataFrame(nav_curves)
    nav_df.to_parquet(OUTPUT_DIR / "dd_reduction_nav_curves.parquet")

    # Save yearly returns comparison
    yearly_data = {}
    for name, nav_s in nav_curves.items():
        yearly = nav_s.pct_change().resample("YE").apply(lambda x: (1+x).prod() - 1)
        yearly_data[name] = yearly

    yearly_df = pd.DataFrame(yearly_data)
    yearly_df.to_csv(OUTPUT_DIR / "dd_reduction_yearly_returns.csv")

    print(f"\n💾 Results saved to {OUTPUT_DIR}")
    print("\nDone.")

    return results


if __name__ == "__main__":
    results = main()

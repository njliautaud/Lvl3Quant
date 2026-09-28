#!/usr/bin/env python3
"""
Sector-Enhanced Portfolio Allocation Strategy
==============================================
Integrates sector leadership signals from sector_leadership_predictor.py
into the portfolio allocation framework from portfolio_allocator_v2.py.

Key idea:
  - When cyclical/growth sectors (XLB, XLRE, XLI) lead → bullish for SPY
    → overweight growth/momentum strategies
  - When defensive sectors (XLU, XLP) lead OR consumer disc (XLY) leads
    → bearish/cautious → overweight income/wheel/cash
  - Walk-forward backtest with SLIDING windows (HC #0)
  - R1 regime-agnostic validation (40+ OOT days, per-regime Sharpe)
  - Risk-adjusted metrics: Sharpe, Sortino, PF, WR, MaxDD, Calmar

Runs on Jupiter (CPU only).

Usage:
    python3 sector_enhanced_allocator.py [--verbose]
"""

import json
import os
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
LVL3 = Path("/home/jupiter/Lvl3Quant")
SECTOR_SUMMARY = LVL3 / "output/growth_research/sector_predictor/summary.json"
OUTPUT_DIR = LVL3 / "output/growth_research/sector_enhanced_allocator"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SECTOR_ETFS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLV": "Health Care",
    "XLI": "Industrials",
    "XLY": "Consumer Disc.",
    "XLC": "Comm. Services",
    "XLP": "Consumer Staples",
    "XLRE": "Real Estate",
    "XLB": "Materials",
    "XLU": "Utilities",
}
BENCHMARK = "SPY"

# Sector groups for signal generation
BULLISH_LEADERS = ["XLB", "XLRE", "XLI"]  # Cyclical/growth: bullish when leading
BEARISH_LEADERS = ["XLY"]  # Consumer Disc leading = bearish signal
DEFENSIVE_LEADERS = ["XLU", "XLP"]  # Defensive leading = caution

# Strategy proxy ETFs for backtesting
# We approximate portfolio strategies with liquid ETFs:
STRATEGY_PROXIES = {
    "Megacap_Momentum": "QQQ",      # Tech/megacap momentum proxy
    "Strangle":         "SVXY",     # Short vol proxy (imperfect but directionally correct)
    "ETF_Rotation":     "RSP",      # Equal-weight S&P as rotation proxy
    "Wheel_CSP":        "USMV",     # Min-vol ETF as defensive income proxy
    "Cash":             None,       # Risk-free, approximated as 0 return + T-bill rate
}

# Baseline weights (from portfolio_allocator_v2.py)
BASELINE = {
    "Megacap_Momentum": 0.50,
    "Strangle":         0.25,
    "ETF_Rotation":     0.17,
    "Wheel_CSP":        0.07,
    "Cash":             0.01,
}

# Walk-forward parameters (SLIDING window, HC #0)
WF_TRAIN_DAYS = 252  # 1 year lookback for signal calibration
WF_TEST_DAYS = 21    # 1 month OOT test
LOOKBACK_RS = 63     # 3-month relative strength for sector leadership
TOP_N_LEADER = 3     # Top N sectors = "leading"

# Transaction costs: 0.1% slippage per ETF rebalance (one-way)
SLIPPAGE_BPS = 10  # basis points per trade
ANNUAL_RF_RATE = 0.04  # approximate T-bill rate for cash strategy and Sharpe calc


# ---------------------------------------------------------------------------
# Data Download
# ---------------------------------------------------------------------------
def download_data(start_year: int = 2015) -> pd.DataFrame:
    """Download sector ETFs, benchmark, and strategy proxy ETFs."""
    import yfinance as yf

    end = datetime.now()
    start = datetime(start_year, 1, 1)

    # All tickers we need
    all_tickers = list(SECTOR_ETFS.keys()) + [BENCHMARK]
    for proxy in STRATEGY_PROXIES.values():
        if proxy and proxy not in all_tickers:
            all_tickers.append(proxy)

    print(f"Downloading {len(all_tickers)} tickers from {start.date()} to {end.date()} ...")
    data = yf.download(all_tickers, start=start, end=end, auto_adjust=True, progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data[["Close"]].rename(columns={"Close": all_tickers[0]})

    prices = prices.dropna(how="all").ffill()
    # Drop any ticker columns that are all NaN
    prices = prices.dropna(axis=1, how="all")

    print(f"  Got {len(prices)} trading days, {prices.shape[1]} tickers")
    print(f"  Date range: {prices.index[0].date()} to {prices.index[-1].date()}")

    # Check for missing proxies
    for name, proxy in STRATEGY_PROXIES.items():
        if proxy and proxy not in prices.columns:
            print(f"  WARNING: {proxy} ({name}) not in data — will use SPY as fallback")

    return prices


# ---------------------------------------------------------------------------
# Sector Leadership Signal
# ---------------------------------------------------------------------------
def compute_sector_signal(prices: pd.DataFrame, lookback: int = LOOKBACK_RS,
                          top_n: int = TOP_N_LEADER) -> pd.DataFrame:
    """
    Compute daily sector leadership signals.

    Returns DataFrame with columns:
      - bullish_score: count of BULLISH_LEADERS in top N (0-3)
      - bearish_score: count of BEARISH_LEADERS in top N (0-1)
      - defensive_score: count of DEFENSIVE_LEADERS in top N (0-2)
      - net_signal: bullish_score - bearish_score - defensive_score
      - regime: 'bullish', 'bearish', or 'neutral'
    """
    # Relative strength: sector return minus SPY return
    rets = prices.pct_change(lookback)
    sector_tickers = [t for t in SECTOR_ETFS.keys() if t in rets.columns]
    rs = rets[sector_tickers].sub(rets[BENCHMARK], axis=0)

    # Rank sectors (1 = best relative strength)
    ranks = rs.rank(axis=1, ascending=False)
    is_leader = ranks <= top_n

    # Compute scores
    signals = pd.DataFrame(index=prices.index)

    bullish_cols = [s for s in BULLISH_LEADERS if s in is_leader.columns]
    bearish_cols = [s for s in BEARISH_LEADERS if s in is_leader.columns]
    defensive_cols = [s for s in DEFENSIVE_LEADERS if s in is_leader.columns]

    signals["bullish_score"] = is_leader[bullish_cols].sum(axis=1) if bullish_cols else 0
    signals["bearish_score"] = is_leader[bearish_cols].sum(axis=1) if bearish_cols else 0
    signals["defensive_score"] = is_leader[defensive_cols].sum(axis=1) if defensive_cols else 0
    signals["net_signal"] = signals["bullish_score"] - signals["bearish_score"] - signals["defensive_score"]

    # Classify regime
    signals["regime"] = "neutral"
    signals.loc[signals["net_signal"] >= 2, "regime"] = "bullish"
    signals.loc[signals["net_signal"] >= 1, "regime"] = signals.loc[
        signals["net_signal"] >= 1, "regime"
    ].where(signals["net_signal"] >= 2, "mildly_bullish")
    signals.loc[signals["net_signal"] <= -1, "regime"] = "cautious"
    signals.loc[signals["net_signal"] <= -2, "regime"] = "bearish"

    return signals


# ---------------------------------------------------------------------------
# Allocation Weights
# ---------------------------------------------------------------------------
def static_weights() -> dict:
    """Return baseline static weights."""
    return dict(BASELINE)


def sector_tilted_weights(regime: str, net_signal: float) -> dict:
    """
    Apply sector-based tilts to baseline weights.

    Regime mapping:
      bullish (net >= 2):      heavy momentum/growth
      mildly_bullish (net=1):  slight tilt to momentum
      neutral (net=0):         baseline
      cautious (net=-1):       tilt to income/defensive
      bearish (net <= -2):     heavy defensive/cash
    """
    w = dict(BASELINE)

    if regime == "bullish":
        # Strong cyclical leadership → overweight growth
        shift = 0.10
        w["Megacap_Momentum"] += shift
        w["ETF_Rotation"] += 0.05
        w["Strangle"] -= shift * 0.5
        w["Wheel_CSP"] -= 0.02
        w["Cash"] -= 0.03

    elif regime == "mildly_bullish":
        shift = 0.05
        w["Megacap_Momentum"] += shift
        w["ETF_Rotation"] += 0.02
        w["Strangle"] -= shift * 0.4
        w["Cash"] -= 0.03

    elif regime == "cautious":
        # Defensive sectors leading → reduce growth, add income
        shift = 0.08
        w["Megacap_Momentum"] -= shift
        w["Wheel_CSP"] += 0.04
        w["Cash"] += 0.04
        w["Strangle"] -= 0.00  # keep strangle (premium selling works in caution)

    elif regime == "bearish":
        # Strong bearish signal → heavy defensive
        shift = 0.15
        w["Megacap_Momentum"] -= shift
        w["ETF_Rotation"] -= 0.05
        w["Wheel_CSP"] += 0.07
        w["Cash"] += 0.10
        w["Strangle"] += 0.03  # slight premium boost in high vol

    # Clamp negatives and renormalize
    w = {k: max(0.0, v) for k, v in w.items()}
    total = sum(w.values())
    if total > 0:
        w = {k: v / total for k, v in w.items()}

    return w


# ---------------------------------------------------------------------------
# Portfolio Return Computation
# ---------------------------------------------------------------------------
def compute_strategy_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily returns for each strategy proxy.
    Cash earns the risk-free rate.
    """
    daily_rf = (1 + ANNUAL_RF_RATE) ** (1 / 252) - 1

    strat_rets = pd.DataFrame(index=prices.index)

    for strat_name, proxy in STRATEGY_PROXIES.items():
        if proxy and proxy in prices.columns:
            strat_rets[strat_name] = prices[proxy].pct_change()
        elif proxy:
            # Fallback to SPY
            strat_rets[strat_name] = prices[BENCHMARK].pct_change()
        else:
            # Cash
            strat_rets[strat_name] = daily_rf

    return strat_rets


def portfolio_daily_returns(strat_rets: pd.DataFrame, weights_series: pd.DataFrame,
                            slippage_bps: float = SLIPPAGE_BPS) -> pd.Series:
    """
    Compute daily portfolio returns given time-varying weights.

    weights_series: DataFrame with same index as strat_rets, columns = strategy names.
    Apply transaction costs when weights change.
    """
    # Align
    common_idx = strat_rets.index.intersection(weights_series.index)
    sr = strat_rets.loc[common_idx].fillna(0)
    ws = weights_series.loc[common_idx].fillna(0)

    # Daily portfolio return (weighted sum)
    port_ret = (sr * ws).sum(axis=1)

    # Transaction costs: proportional to weight changes
    weight_changes = ws.diff().abs().sum(axis=1)
    # Each unit of weight change costs slippage_bps basis points
    tc = weight_changes * (slippage_bps / 10000)

    port_ret_net = port_ret - tc

    return port_ret_net


# ---------------------------------------------------------------------------
# Risk-Adjusted Metrics
# ---------------------------------------------------------------------------
def compute_metrics(returns: pd.Series, label: str = "", annual_rf: float = ANNUAL_RF_RATE) -> dict:
    """Compute comprehensive risk-adjusted metrics."""
    rets = returns.dropna()
    if len(rets) < 30:
        return {"label": label, "n_days": len(rets), "error": "insufficient data"}

    daily_rf = (1 + annual_rf) ** (1 / 252) - 1
    excess = rets - daily_rf

    # Annualized return
    total_ret = (1 + rets).prod()
    n_years = len(rets) / 252
    ann_ret = total_ret ** (1 / n_years) - 1 if n_years > 0 else 0

    # Annualized vol
    ann_vol = rets.std() * np.sqrt(252)

    # Sharpe (annualized)
    sharpe = (ann_ret - annual_rf) / ann_vol if ann_vol > 0 else 0

    # Sortino
    downside = rets[rets < daily_rf] - daily_rf
    downside_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else 1e-9
    sortino = (ann_ret - annual_rf) / downside_vol if downside_vol > 0 else 0

    # Max drawdown
    cum = (1 + rets).cumprod()
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max
    max_dd = drawdown.min()

    # Calmar
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

    # Win rate
    wr = (rets > 0).mean() * 100

    # Profit factor
    gains = rets[rets > 0].sum()
    losses = abs(rets[rets < 0].sum())
    pf = gains / losses if losses > 0 else float("inf")

    # Monthly returns for regime analysis
    monthly = rets.resample("ME").apply(lambda x: (1 + x).prod() - 1)

    return {
        "label": label,
        "n_days": len(rets),
        "n_years": round(n_years, 2),
        "ann_return_pct": round(ann_ret * 100, 2),
        "ann_vol_pct": round(ann_vol * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_dd_pct": round(max_dd * 100, 2),
        "calmar": round(calmar, 3),
        "win_rate_pct": round(wr, 1),
        "profit_factor": round(pf, 3),
        "total_return_pct": round((total_ret - 1) * 100, 2),
        "monthly_returns": monthly,
    }


# ---------------------------------------------------------------------------
# Walk-Forward Backtest (SLIDING windows, HC #0)
# ---------------------------------------------------------------------------
def walk_forward_backtest(prices: pd.DataFrame, verbose: bool = False) -> dict:
    """
    Walk-forward backtest comparing static vs sector-tilted allocation.

    SLIDING window: train on WF_TRAIN_DAYS, test on WF_TEST_DAYS,
    then slide forward by WF_TEST_DAYS (drop oldest train day).
    """
    print("\n" + "=" * 70)
    print("WALK-FORWARD BACKTEST: Static vs Sector-Tilted Allocation")
    print(f"  Train window: {WF_TRAIN_DAYS}d, Test window: {WF_TEST_DAYS}d")
    print(f"  Sector RS lookback: {LOOKBACK_RS}d, Top {TOP_N_LEADER} leaders")
    print("=" * 70)

    # Compute all inputs
    sector_signals = compute_sector_signal(prices, LOOKBACK_RS, TOP_N_LEADER)
    strat_rets = compute_strategy_returns(prices)

    # Align everything
    valid_start = max(LOOKBACK_RS, 1)  # Need lookback for RS calculation
    dates = prices.index[valid_start:]

    # Build walk-forward folds
    total_days = len(dates)
    min_start = WF_TRAIN_DAYS  # Need full training window first

    if total_days < min_start + WF_TEST_DAYS:
        print("ERROR: Not enough data for walk-forward")
        return {}

    # Storage for OOT returns
    static_oot_returns = []
    tilted_oot_returns = []
    oot_dates = []
    fold_details = []

    fold_idx = 0
    test_start = min_start

    while test_start + WF_TEST_DAYS <= total_days:
        train_start = test_start - WF_TRAIN_DAYS
        train_end = test_start
        test_end = min(test_start + WF_TEST_DAYS, total_days)

        train_dates = dates[train_start:train_end]
        test_dates = dates[test_start:test_end]

        if len(test_dates) < 5:
            break

        # --- Training phase: calibrate signal strength ---
        # In a real system we'd optimize tilt magnitudes here.
        # For this backtest, we use fixed tilts (no data snooping on test set).
        # The "training" is just computing the current sector signal state.

        # --- Test phase: apply allocation ---
        for dt in test_dates:
            if dt not in sector_signals.index or dt not in strat_rets.index:
                continue

            sig = sector_signals.loc[dt]
            regime = sig["regime"]
            net = sig["net_signal"]

            # Static weights
            sw = static_weights()
            # Tilted weights
            tw = sector_tilted_weights(regime, net)

            # Get strategy returns for this day
            day_rets = strat_rets.loc[dt]

            # Portfolio returns
            static_ret = sum(sw.get(s, 0) * day_rets.get(s, 0) for s in sw)
            tilted_ret = sum(tw.get(s, 0) * day_rets.get(s, 0) for s in tw)

            static_oot_returns.append(static_ret)
            tilted_oot_returns.append(tilted_ret)
            oot_dates.append(dt)

        fold_details.append({
            "fold": fold_idx,
            "train_start": str(train_dates[0].date()),
            "train_end": str(train_dates[-1].date()),
            "test_start": str(test_dates[0].date()),
            "test_end": str(test_dates[-1].date()),
            "test_days": len(test_dates),
        })

        if verbose and fold_idx % 20 == 0:
            print(f"  Fold {fold_idx}: train {train_dates[0].date()}-{train_dates[-1].date()}, "
                  f"test {test_dates[0].date()}-{test_dates[-1].date()}")

        fold_idx += 1
        test_start += WF_TEST_DAYS

    print(f"\n  Completed {fold_idx} walk-forward folds")
    print(f"  Total OOT days: {len(oot_dates)}")

    # Build return series
    static_series = pd.Series(static_oot_returns, index=oot_dates, name="static")
    tilted_series = pd.Series(tilted_oot_returns, index=oot_dates, name="tilted")

    # Apply transaction costs to tilted (rebalancing) vs static (buy-and-hold-ish)
    # Static rebalances less frequently
    sector_signals_aligned = sector_signals.loc[sector_signals.index.isin(oot_dates)]
    regime_changes = (sector_signals_aligned["regime"] != sector_signals_aligned["regime"].shift(1))
    # Tilted strategy incurs costs on regime changes
    tilted_tc = regime_changes.astype(float) * (SLIPPAGE_BPS / 10000) * 0.5  # Half turnover on average
    tilted_series = tilted_series - tilted_tc.reindex(tilted_series.index).fillna(0)

    # Static incurs minimal rebalancing costs (monthly drift correction)
    monthly_rebal = pd.Series(0.0, index=static_series.index)
    for i, dt in enumerate(static_series.index):
        if i > 0 and dt.month != static_series.index[i - 1].month:
            monthly_rebal.iloc[i] = SLIPPAGE_BPS / 10000 * 0.1  # Small drift correction
    static_series = static_series - monthly_rebal

    # Compute metrics
    static_metrics = compute_metrics(static_series, "Static Allocation")
    tilted_metrics = compute_metrics(tilted_series, "Sector-Tilted Allocation")

    # Remove non-serializable monthly returns for JSON
    static_monthly = static_metrics.pop("monthly_returns", pd.Series())
    tilted_monthly = tilted_metrics.pop("monthly_returns", pd.Series())

    print(f"\n{'='*60}")
    print(f"  {'Metric':<25s} {'Static':>12s} {'Tilted':>12s} {'Delta':>12s}")
    print(f"  {'-'*60}")

    compare_keys = [
        ("ann_return_pct", "Ann Return %"),
        ("ann_vol_pct", "Ann Vol %"),
        ("sharpe", "Sharpe"),
        ("sortino", "Sortino"),
        ("max_dd_pct", "Max DD %"),
        ("calmar", "Calmar"),
        ("win_rate_pct", "Win Rate %"),
        ("profit_factor", "Profit Factor"),
        ("total_return_pct", "Total Return %"),
    ]
    for key, label in compare_keys:
        sv = static_metrics.get(key, 0)
        tv = tilted_metrics.get(key, 0)
        delta = tv - sv
        print(f"  {label:<25s} {sv:>12.3f} {tv:>12.3f} {delta:>+12.3f}")

    results = {
        "static_metrics": static_metrics,
        "tilted_metrics": tilted_metrics,
        "n_folds": fold_idx,
        "n_oot_days": len(oot_dates),
        "fold_details": fold_details[:5] + fold_details[-5:] if len(fold_details) > 10 else fold_details,
        "oot_date_range": f"{oot_dates[0].date()} to {oot_dates[-1].date()}" if oot_dates else "N/A",
    }

    return results, static_series, tilted_series, static_monthly, tilted_monthly


# ---------------------------------------------------------------------------
# R1 Regime-Agnostic Validation (HC #428)
# ---------------------------------------------------------------------------
def r1_regime_validation(static_series: pd.Series, tilted_series: pd.Series,
                         prices: pd.DataFrame) -> dict:
    """
    R1 validation: test performance across bull/bear/flat regimes.

    Requirements:
      - 40+ OOT days in ALL regimes
      - Per-regime Sharpe
      - Regime asymmetry check: |Sharpe_bull - Sharpe_bear| / max(|S_bull|, |S_bear|) <= 0.50
    """
    print("\n" + "=" * 70)
    print("R1 REGIME-AGNOSTIC VALIDATION (HC #428)")
    print("=" * 70)

    # Classify each day as green/red/flat based on SPY close-to-close
    spy_rets = prices[BENCHMARK].pct_change().reindex(tilted_series.index)

    regimes = pd.Series("flat", index=spy_rets.index)
    regimes[spy_rets > 0.001] = "green"
    regimes[spy_rets < -0.001] = "red"

    # Also classify by trailing 6-month SPY return (macro regime)
    spy_6m = prices[BENCHMARK].pct_change(126).reindex(tilted_series.index)
    macro_regime = pd.Series("mixed", index=spy_6m.index)
    macro_regime[spy_6m > 0.05] = "bull"
    macro_regime[spy_6m < -0.05] = "bear"
    macro_regime[(spy_6m >= -0.05) & (spy_6m <= 0.05)] = "mixed"

    results = {}

    # --- Day-level regime analysis ---
    print("\n  --- Day-Level Regime (SPY close-to-close) ---")
    print(f"  {'Regime':<10s} {'N':>6s} {'Static Sharpe':>14s} {'Tilted Sharpe':>14s} {'Delta':>10s}")
    print(f"  {'-'*56}")

    for regime_name in ["green", "red", "flat"]:
        mask = regimes == regime_name
        s_rets = static_series[mask]
        t_rets = tilted_series[mask]

        n = len(s_rets)
        if n < 40:
            print(f"  {regime_name:<10s} {n:>6d}  ** INSUFFICIENT (<40 days) **")
            results[f"day_{regime_name}"] = {"n": n, "status": "insufficient", "pass": False}
            continue

        s_sharpe = s_rets.mean() / s_rets.std() * np.sqrt(252) if s_rets.std() > 0 else 0
        t_sharpe = t_rets.mean() / t_rets.std() * np.sqrt(252) if t_rets.std() > 0 else 0
        delta = t_sharpe - s_sharpe

        results[f"day_{regime_name}"] = {
            "n": n,
            "static_sharpe": round(s_sharpe, 3),
            "tilted_sharpe": round(t_sharpe, 3),
            "delta": round(delta, 3),
            "pass": True,
        }
        print(f"  {regime_name:<10s} {n:>6d} {s_sharpe:>14.3f} {t_sharpe:>14.3f} {delta:>+10.3f}")

    # --- Macro-level regime analysis ---
    print(f"\n  --- Macro Regime (6-month SPY return) ---")
    print(f"  {'Regime':<10s} {'N':>6s} {'Static Sharpe':>14s} {'Tilted Sharpe':>14s} {'Delta':>10s}")
    print(f"  {'-'*56}")

    macro_sharpes = {}
    for regime_name in ["bull", "bear", "mixed"]:
        mask = macro_regime == regime_name
        s_rets = static_series[mask].dropna()
        t_rets = tilted_series[mask].dropna()

        n = len(s_rets)
        if n < 40:
            print(f"  {regime_name:<10s} {n:>6d}  ** INSUFFICIENT (<40 days) **")
            results[f"macro_{regime_name}"] = {"n": n, "status": "insufficient", "pass": False}
            continue

        s_sharpe = s_rets.mean() / s_rets.std() * np.sqrt(252) if s_rets.std() > 0 else 0
        t_sharpe = t_rets.mean() / t_rets.std() * np.sqrt(252) if t_rets.std() > 0 else 0
        delta = t_sharpe - s_sharpe

        macro_sharpes[regime_name] = {"static": s_sharpe, "tilted": t_sharpe}
        results[f"macro_{regime_name}"] = {
            "n": n,
            "static_sharpe": round(s_sharpe, 3),
            "tilted_sharpe": round(t_sharpe, 3),
            "delta": round(delta, 3),
            "pass": True,
        }
        print(f"  {regime_name:<10s} {n:>6d} {s_sharpe:>14.3f} {t_sharpe:>14.3f} {delta:>+10.3f}")

    # --- R1 Asymmetry Check ---
    bull_sharpe = macro_sharpes.get("bull", {}).get("tilted", 0)
    bear_sharpe = macro_sharpes.get("bear", {}).get("tilted", 0)

    if bull_sharpe != 0 or bear_sharpe != 0:
        max_abs = max(abs(bull_sharpe), abs(bear_sharpe))
        asymmetry = abs(bull_sharpe - bear_sharpe) / max_abs if max_abs > 0 else 0
        r1_pass = asymmetry <= 0.50
        results["r1_asymmetry"] = round(asymmetry, 3)
        results["r1_pass"] = r1_pass
        print(f"\n  R1 Asymmetry (tilted): |{bull_sharpe:.3f} - {bear_sharpe:.3f}| / "
              f"{max_abs:.3f} = {asymmetry:.3f} {'PASS' if r1_pass else 'FAIL'} (threshold: 0.50)")
    else:
        results["r1_asymmetry"] = None
        results["r1_pass"] = None
        print("\n  R1 Asymmetry: cannot compute (missing regime data)")

    # --- Sub-period consistency ---
    print(f"\n  --- Sub-Period Consistency ---")
    midpoint = pd.Timestamp("2020-01-01")
    for period_name, mask in [("2015-2019", tilted_series.index < midpoint),
                               ("2020-2026", tilted_series.index >= midpoint)]:
        s_rets = static_series[mask].dropna()
        t_rets = tilted_series[mask].dropna()
        n = len(t_rets)
        if n < 40:
            print(f"  {period_name}: {n} days (insufficient)")
            continue
        s_sharpe = s_rets.mean() / s_rets.std() * np.sqrt(252) if s_rets.std() > 0 else 0
        t_sharpe = t_rets.mean() / t_rets.std() * np.sqrt(252) if t_rets.std() > 0 else 0
        delta = t_sharpe - s_sharpe
        results[f"subperiod_{period_name}"] = {
            "n": n,
            "static_sharpe": round(s_sharpe, 3),
            "tilted_sharpe": round(t_sharpe, 3),
            "delta": round(delta, 3),
        }
        print(f"  {period_name}: N={n}, Static Sharpe={s_sharpe:.3f}, "
              f"Tilted Sharpe={t_sharpe:.3f}, Delta={delta:+.3f}")

    return results


# ---------------------------------------------------------------------------
# Signal Distribution Analysis
# ---------------------------------------------------------------------------
def signal_distribution_analysis(prices: pd.DataFrame) -> dict:
    """Analyze the distribution and frequency of sector signals."""
    print("\n" + "=" * 70)
    print("SIGNAL DISTRIBUTION ANALYSIS")
    print("=" * 70)

    signals = compute_sector_signal(prices, LOOKBACK_RS, TOP_N_LEADER)
    signals = signals.dropna()

    regime_counts = signals["regime"].value_counts()
    total = len(signals)

    print(f"\n  Regime distribution (N={total}):")
    results = {"total_days": total, "regime_distribution": {}}
    for regime, count in regime_counts.items():
        pct = count / total * 100
        results["regime_distribution"][regime] = {"count": int(count), "pct": round(pct, 1)}
        print(f"    {regime:<20s}: {count:>6d} ({pct:5.1f}%)")

    # Net signal stats
    net = signals["net_signal"]
    results["net_signal_stats"] = {
        "mean": round(net.mean(), 3),
        "std": round(net.std(), 3),
        "min": int(net.min()),
        "max": int(net.max()),
    }
    print(f"\n  Net signal: mean={net.mean():.3f}, std={net.std():.3f}, "
          f"range=[{net.min():.0f}, {net.max():.0f}]")

    # Autocorrelation (persistence)
    ac1 = net.autocorr(1)
    ac5 = net.autocorr(5)
    ac21 = net.autocorr(21)
    results["autocorrelation"] = {
        "lag_1": round(ac1, 3),
        "lag_5": round(ac5, 3),
        "lag_21": round(ac21, 3),
    }
    print(f"  Signal autocorrelation: lag1={ac1:.3f}, lag5={ac5:.3f}, lag21={ac21:.3f}")

    # Regime transition frequency
    regime_changes = (signals["regime"] != signals["regime"].shift(1)).sum()
    avg_regime_length = total / regime_changes if regime_changes > 0 else total
    results["regime_changes"] = int(regime_changes)
    results["avg_regime_length_days"] = round(avg_regime_length, 1)
    print(f"  Regime changes: {regime_changes}, avg regime length: {avg_regime_length:.1f} days")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Sector-Enhanced Portfolio Allocator")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    np.random.seed(42)
    print("=" * 70)
    print("SECTOR-ENHANCED PORTFOLIO ALLOCATION STRATEGY")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    # 1. Download data
    prices = download_data(start_year=2015)

    # 2. Signal distribution analysis
    signal_results = signal_distribution_analysis(prices)

    # 3. Walk-forward backtest
    wf_results, static_series, tilted_series, static_monthly, tilted_monthly = \
        walk_forward_backtest(prices, verbose=args.verbose)

    # 4. R1 regime-agnostic validation
    r1_results = r1_regime_validation(static_series, tilted_series, prices)

    # 5. Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    sm = wf_results.get("static_metrics", {})
    tm = wf_results.get("tilted_metrics", {})

    improvement = {
        "sharpe_delta": round(tm.get("sharpe", 0) - sm.get("sharpe", 0), 3),
        "sortino_delta": round(tm.get("sortino", 0) - sm.get("sortino", 0), 3),
        "return_delta_pct": round(tm.get("ann_return_pct", 0) - sm.get("ann_return_pct", 0), 2),
        "maxdd_improvement_pct": round(tm.get("max_dd_pct", 0) - sm.get("max_dd_pct", 0), 2),
        "calmar_delta": round(tm.get("calmar", 0) - sm.get("calmar", 0), 3),
    }

    print(f"\n  Sector tilting impact (tilted minus static):")
    print(f"    Sharpe:  {improvement['sharpe_delta']:+.3f}")
    print(f"    Sortino: {improvement['sortino_delta']:+.3f}")
    print(f"    Ann Ret: {improvement['return_delta_pct']:+.2f}%")
    print(f"    Max DD:  {improvement['maxdd_improvement_pct']:+.2f}% (positive = worse)")
    print(f"    Calmar:  {improvement['calmar_delta']:+.3f}")

    r1_pass = r1_results.get("r1_pass")
    print(f"\n  R1 Regime-Agnostic: {'PASS' if r1_pass else 'FAIL' if r1_pass is False else 'N/A'}")
    print(f"  R1 Asymmetry: {r1_results.get('r1_asymmetry', 'N/A')}")

    # Verdict
    sharpe_improved = improvement["sharpe_delta"] > 0
    sortino_improved = improvement["sortino_delta"] > 0

    if sharpe_improved and sortino_improved and r1_pass:
        verdict = "ADOPT — sector tilting improves risk-adjusted returns and passes R1"
    elif sharpe_improved and sortino_improved and r1_pass is None:
        verdict = "TENTATIVE ADOPT — improvements found but R1 inconclusive"
    elif sharpe_improved or sortino_improved:
        verdict = "MIXED — partial improvement, review regime breakdown before adopting"
    else:
        verdict = "REJECT — sector tilting does not improve risk-adjusted returns"

    print(f"\n  VERDICT: {verdict}")

    # 6. Save results
    full_results = {
        "metadata": {
            "date": datetime.now().isoformat(),
            "script": "sector_enhanced_allocator.py",
            "data_range": f"{prices.index[0].date()} to {prices.index[-1].date()}",
            "n_trading_days": len(prices),
            "wf_train_days": WF_TRAIN_DAYS,
            "wf_test_days": WF_TEST_DAYS,
            "lookback_rs": LOOKBACK_RS,
            "top_n_leader": TOP_N_LEADER,
            "slippage_bps": SLIPPAGE_BPS,
            "annual_rf_rate": ANNUAL_RF_RATE,
            "bullish_leaders": BULLISH_LEADERS,
            "bearish_leaders": BEARISH_LEADERS,
            "defensive_leaders": DEFENSIVE_LEADERS,
            "strategy_proxies": STRATEGY_PROXIES,
            "baseline_weights": BASELINE,
        },
        "signal_distribution": signal_results,
        "walkforward_results": wf_results,
        "r1_validation": r1_results,
        "improvement": improvement,
        "verdict": verdict,
    }

    # Save JSON
    json_path = OUTPUT_DIR / "results.json"
    with open(json_path, "w") as f:
        json.dump(full_results, f, indent=2, default=str)
    print(f"\n  Saved results to {json_path}")

    # Save daily return series as CSV
    combined = pd.DataFrame({
        "static_return": static_series,
        "tilted_return": tilted_series,
    })
    csv_path = OUTPUT_DIR / "daily_returns.csv"
    combined.to_csv(csv_path)
    print(f"  Saved daily returns to {csv_path}")

    # Save equity curves
    equity = pd.DataFrame({
        "static_equity": (1 + static_series).cumprod(),
        "tilted_equity": (1 + tilted_series).cumprod(),
    })
    eq_path = OUTPUT_DIR / "equity_curves.csv"
    equity.to_csv(eq_path)
    print(f"  Saved equity curves to {eq_path}")

    print("\n  DONE.")
    return full_results


if __name__ == "__main__":
    main()

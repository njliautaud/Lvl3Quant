#!/usr/bin/env python3
"""
Flow Screener Historical Backtest
==================================
Validates the informed flow screener using proxy signals on historical equity data.

Proxy signals (since we can't get historical intraday options flow):
1. Unusual Volume: equity volume > 2x 20-day avg
2. IV Proxy (ATR spike): ATR > 2x its 20-day avg = implied vol expansion
3. Pre-Earnings Unusual Activity: unusual volume within 5 days before earnings

Forward returns measured at 1d, 3d, 5d, 10d horizons.
Adversarial tests: permutation, inverse, sub-period stability, regime gap.

Output: /home/jupiter/Lvl3Quant/state/flow_screener_backtest_results.json
Logs:   /home/jupiter/Lvl3Quant/logs/flow_screener/
"""

import json
import logging
import os
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Paths
BASE = Path("/home/jupiter/Lvl3Quant")
UNIVERSE_JSON = BASE / "data" / "quality_universe.json"
OUTPUT_JSON = BASE / "state" / "flow_screener_backtest_results.json"
LOG_DIR = BASE / "logs" / "flow_screener"
CACHE_DIR = BASE / "data" / "flow_screener_cache"

LOG_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "backtest.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("flow_backtest")

# ── Config ──
START_DATE = "2020-01-01"
END_DATE = "2026-07-31"
UNUSUAL_VOL_THRESHOLD = 2.0   # volume / 20d avg
ATR_SPIKE_THRESHOLD = 2.0     # ATR / 20d avg ATR
EARNINGS_WINDOW_DAYS = 5      # look for unusual vol within N days before earnings
FORWARD_HORIZONS = [1, 3, 5, 10]  # trading days
N_PERMUTATIONS = 1000
REGIME_GAP_THRESHOLD = 0.50


def load_universe() -> list:
    with open(UNIVERSE_JSON) as f:
        data = json.load(f)
    tickers = data["tickers"]
    # BRK-B needs special handling for yfinance
    return [t.replace("BRK-B", "BRK-B") for t in tickers]


def download_price_data(tickers: list) -> dict:
    """Download daily OHLCV data for all tickers + SPY for regime."""
    import yfinance as yf

    all_tickers = list(set(tickers + ["SPY"]))
    cache_file = CACHE_DIR / "price_data.parquet"

    # Use cache if less than 1 day old
    if cache_file.exists():
        age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        if age_hours < 24:
            log.info("Loading cached price data (%.1f hours old)", age_hours)
            df = pd.read_parquet(cache_file)
            return {t: df[df["ticker"] == t].copy() for t in df["ticker"].unique()}

    log.info("Downloading price data for %d tickers...", len(all_tickers))
    data = {}

    # Download in batches to avoid rate limits
    batch_size = 10
    for i in range(0, len(all_tickers), batch_size):
        batch = all_tickers[i:i + batch_size]
        log.info("  Batch %d/%d: %s", i // batch_size + 1,
                 (len(all_tickers) + batch_size - 1) // batch_size, batch)
        try:
            raw = yf.download(
                batch, start=START_DATE, end=END_DATE,
                auto_adjust=True, progress=False, threads=True
            )
            if raw.empty:
                continue

            for ticker in batch:
                try:
                    if len(batch) == 1:
                        df_t = raw.copy()
                    else:
                        # yfinance returns MultiIndex columns for multi-ticker
                        df_t = raw.xs(ticker, level=1, axis=1) if isinstance(raw.columns, pd.MultiIndex) else raw.copy()

                    df_t = df_t.dropna(subset=["Close"])
                    if len(df_t) < 50:
                        log.warning("  %s: only %d rows, skipping", ticker, len(df_t))
                        continue
                    df_t["ticker"] = ticker
                    df_t.index.name = "Date"
                    data[ticker] = df_t.reset_index()
                except Exception as e:
                    log.warning("  %s extraction failed: %s", ticker, e)
        except Exception as e:
            log.error("  Batch download failed: %s", e)
        time.sleep(0.5)  # Rate limit courtesy

    # Cache
    if data:
        combined = pd.concat(data.values(), ignore_index=True)
        combined.to_parquet(cache_file, index=False)
        log.info("Cached %d tickers, %d total rows", len(data), len(combined))

    return data


def download_earnings_dates(tickers: list) -> dict:
    """Get historical earnings dates for each ticker via yfinance."""
    import yfinance as yf

    cache_file = CACHE_DIR / "earnings_dates.json"
    if cache_file.exists():
        age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        if age_hours < 24 * 7:  # cache for 1 week
            log.info("Loading cached earnings dates")
            with open(cache_file) as f:
                return json.load(f)

    log.info("Downloading earnings dates...")
    earnings = {}
    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            cal = t.get_earnings_dates(limit=100)
            if cal is not None and len(cal) > 0:
                dates = [str(d.date()) for d in cal.index if not pd.isna(d)]
                earnings[ticker] = dates
                log.info("  %s: %d earnings dates", ticker, len(dates))
            else:
                earnings[ticker] = []
        except Exception as e:
            log.warning("  %s earnings failed: %s", ticker, e)
            earnings[ticker] = []
        time.sleep(0.3)

    with open(cache_file, "w") as f:
        json.dump(earnings, f)

    return earnings


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute proxy signal features for a single ticker."""
    df = df.sort_values("Date").copy()

    # Volume ratio (proxy for unusual options volume)
    df["vol_20d_avg"] = df["Volume"].rolling(20, min_periods=10).mean()
    df["vol_ratio"] = df["Volume"] / df["vol_20d_avg"].replace(0, np.nan)

    # ATR (proxy for implied volatility)
    high = df["High"]
    low = df["Low"]
    close_prev = df["Close"].shift(1)
    tr = pd.concat([
        (high - low),
        (high - close_prev).abs(),
        (low - close_prev).abs()
    ], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14, min_periods=7).mean()
    df["atr_20d_avg"] = df["atr_14"].rolling(20, min_periods=10).mean()
    df["atr_ratio"] = df["atr_14"] / df["atr_20d_avg"].replace(0, np.nan)

    # Forward returns
    for h in FORWARD_HORIZONS:
        df[f"fwd_ret_{h}d"] = df["Close"].shift(-h) / df["Close"] - 1

    # Daily return for baseline stats
    df["daily_ret"] = df["Close"].pct_change()

    return df


def detect_signals(df: pd.DataFrame, earnings_dates: list) -> pd.DataFrame:
    """Detect proxy signals for a single ticker. Returns DataFrame of signal rows."""
    signals = []

    earnings_set = set()
    for d in earnings_dates:
        try:
            earnings_set.add(pd.Timestamp(d).date())
        except:
            pass

    for idx, row in df.iterrows():
        if pd.isna(row.get("vol_ratio")) or pd.isna(row.get("atr_ratio")):
            continue

        date = pd.Timestamp(row["Date"]).date()
        signal_types = []
        direction = "neutral"

        # Signal 1: Unusual volume
        if row["vol_ratio"] >= UNUSUAL_VOL_THRESHOLD:
            signal_types.append("unusual_volume")
            # Direction hint: large up day with volume = bullish, down day = bearish
            if row["daily_ret"] > 0.01:
                direction = "bullish"
            elif row["daily_ret"] < -0.01:
                direction = "bearish"

        # KILLED: ATR spike Sharpe -9.72 at 5d, destroys value (flow research 2026-08-06)
        # ATR is a poor IV proxy — use real IV data (iv_skew_shift) instead.
        # if row["atr_ratio"] >= ATR_SPIKE_THRESHOLD:
        #     signal_types.append("atr_spike")

        # Signal 3: Pre-earnings unusual volume
        is_pre_earnings = False
        for ed in earnings_set:
            days_until = (ed - date).days
            if 0 < days_until <= EARNINGS_WINDOW_DAYS:
                is_pre_earnings = True
                break

        if is_pre_earnings and row["vol_ratio"] >= 1.5:
            signal_types.append("pre_earnings_flow")

        if signal_types:
            sig = {
                "date": str(date),
                "ticker": row["ticker"],
                "signal_types": signal_types,
                "direction": direction,
                "vol_ratio": row["vol_ratio"],
                "atr_ratio": row["atr_ratio"],
                "daily_ret": row["daily_ret"],
            }
            for h in FORWARD_HORIZONS:
                sig[f"fwd_ret_{h}d"] = row.get(f"fwd_ret_{h}d", np.nan)
            signals.append(sig)

    return pd.DataFrame(signals)


def compute_spy_regime(spy_data: pd.DataFrame) -> pd.DataFrame:
    """Classify each date as bull/bear/flat based on SPY vs 200-SMA."""
    spy = spy_data.sort_values("Date").copy()
    spy["sma_200"] = spy["Close"].rolling(200, min_periods=100).mean()
    spy["regime"] = "flat"
    spy.loc[spy["Close"] > spy["sma_200"] * 1.02, "regime"] = "bull"
    spy.loc[spy["Close"] < spy["sma_200"] * 0.98, "regime"] = "bear"
    return spy[["Date", "regime"]].copy()


def compute_metrics(returns: np.ndarray) -> dict:
    """Compute trading metrics from an array of returns."""
    returns = returns[~np.isnan(returns)]
    if len(returns) == 0:
        return {"n": 0, "mean": 0, "sharpe": 0, "sortino": 0, "win_rate": 0, "profit_factor": 0}

    n = len(returns)
    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1) if n > 1 else 1e-6

    # Annualized Sharpe (assume ~252 trading days, scale by sqrt of horizon)
    sharpe = (mean_ret / max(std_ret, 1e-8)) * np.sqrt(252)

    # Sortino
    downside = returns[returns < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-6
    sortino = (mean_ret / max(downside_std, 1e-8)) * np.sqrt(252)

    # Win rate
    win_rate = np.mean(returns > 0) if n > 0 else 0

    # Profit factor
    gross_profit = np.sum(returns[returns > 0])
    gross_loss = abs(np.sum(returns[returns < 0]))
    profit_factor = gross_profit / max(gross_loss, 1e-8)

    return {
        "n": int(n),
        "mean_ret_bps": round(mean_ret * 10000, 1),
        "median_ret_bps": round(np.median(returns) * 10000, 1),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "win_rate": round(win_rate * 100, 1),
        "profit_factor": round(profit_factor, 2),
        "std_bps": round(std_ret * 10000, 1),
    }


def permutation_test(signal_returns: np.ndarray, all_returns: np.ndarray,
                     n_perms: int = 1000) -> float:
    """Permutation test: compare signal mean to random sampling from all returns.
    Returns p-value (fraction of random samples with mean >= signal mean)."""
    signal_returns = signal_returns[~np.isnan(signal_returns)]
    all_returns = all_returns[~np.isnan(all_returns)]

    if len(signal_returns) == 0 or len(all_returns) == 0:
        return 1.0

    observed_mean = np.mean(signal_returns)
    n_signal = len(signal_returns)

    rng = np.random.RandomState(42)
    count_better = 0
    for _ in range(n_perms):
        random_sample = rng.choice(all_returns, size=n_signal, replace=True)
        if np.mean(random_sample) >= observed_mean:
            count_better += 1

    return count_better / n_perms


def run_backtest():
    log.info("=" * 70)
    log.info("FLOW SCREENER HISTORICAL BACKTEST")
    log.info("Period: %s to %s", START_DATE, END_DATE)
    log.info("=" * 70)

    # Load universe
    tickers = load_universe()
    log.info("Universe: %d tickers", len(tickers))

    # Download data
    price_data = download_price_data(tickers)
    if not price_data:
        log.error("No price data downloaded. Aborting.")
        return

    # Get SPY regime
    if "SPY" not in price_data:
        log.error("SPY data missing, cannot compute regimes.")
        return

    spy_regime = compute_spy_regime(price_data["SPY"])
    spy_regime["date_str"] = spy_regime["Date"].dt.strftime("%Y-%m-%d")
    regime_map = dict(zip(spy_regime["date_str"], spy_regime["regime"]))

    # Download earnings dates
    earnings_data = download_earnings_dates(tickers)

    # Compute features and detect signals for each ticker
    all_signals = []
    all_daily_returns = []

    for ticker in tickers:
        if ticker not in price_data:
            log.warning("  %s: no price data, skipping", ticker)
            continue

        df = compute_features(price_data[ticker])
        earnings_dates = earnings_data.get(ticker, [])
        signals = detect_signals(df, earnings_dates)

        if len(signals) > 0:
            all_signals.append(signals)
            log.info("  %s: %d signals detected", ticker, len(signals))

        # Collect all daily returns for baseline
        daily_rets = df["daily_ret"].dropna().values
        all_daily_returns.extend(daily_rets)

    if not all_signals:
        log.error("No signals detected across any ticker. Aborting.")
        return

    signals_df = pd.concat(all_signals, ignore_index=True)
    all_daily_returns = np.array(all_daily_returns)

    # Add regime to signals
    signals_df["regime"] = signals_df["date"].map(regime_map).fillna("flat")

    log.info("\n" + "=" * 70)
    log.info("SIGNAL SUMMARY")
    log.info("Total signals: %d", len(signals_df))
    log.info("Unique dates: %d", signals_df["date"].nunique())
    log.info("Tickers with signals: %d", signals_df["ticker"].nunique())

    # Count by signal type
    type_counts = {}
    for _, row in signals_df.iterrows():
        for st in row["signal_types"]:
            type_counts[st] = type_counts.get(st, 0) + 1
    for st, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        log.info("  %s: %d", st, count)

    regime_counts = signals_df["regime"].value_counts()
    log.info("Regime distribution: %s", dict(regime_counts))

    # ── MAIN RESULTS ──
    results = {
        "backtest_period": f"{START_DATE} to {END_DATE}",
        "universe_size": len(tickers),
        "total_signals": len(signals_df),
        "unique_signal_dates": int(signals_df["date"].nunique()),
        "signal_type_counts": type_counts,
        "regime_distribution": dict(regime_counts),
        "horizons": {},
        "by_signal_type": {},
        "by_regime": {},
        "adversarial_tests": {},
    }

    # Per-horizon analysis
    log.info("\n" + "=" * 70)
    log.info("FORWARD RETURN ANALYSIS (ALL SIGNALS)")
    log.info("=" * 70)

    for h in FORWARD_HORIZONS:
        col = f"fwd_ret_{h}d"
        rets = signals_df[col].dropna().values

        metrics = compute_metrics(rets)
        log.info("\n%dd Forward Returns:", h)
        log.info("  N=%d, Mean=%.1f bps, Median=%.1f bps", metrics["n"], metrics["mean_ret_bps"], metrics["median_ret_bps"])
        log.info("  Sharpe=%.2f, Sortino=%.2f, WR=%.1f%%, PF=%.2f",
                 metrics["sharpe"], metrics["sortino"], metrics["win_rate"], metrics["profit_factor"])

        results["horizons"][f"{h}d"] = metrics

    # By signal type
    log.info("\n" + "=" * 70)
    log.info("BY SIGNAL TYPE")
    log.info("=" * 70)

    for sig_type in ["unusual_volume", "atr_spike", "pre_earnings_flow"]:
        mask = signals_df["signal_types"].apply(lambda x: sig_type in x)
        subset = signals_df[mask]
        if len(subset) == 0:
            continue

        log.info("\n--- %s (N=%d) ---", sig_type, len(subset))
        type_results = {}
        for h in FORWARD_HORIZONS:
            col = f"fwd_ret_{h}d"
            rets = subset[col].dropna().values
            m = compute_metrics(rets)
            log.info("  %dd: Mean=%.1f bps, Sharpe=%.2f, WR=%.1f%%, PF=%.2f",
                     h, m["mean_ret_bps"], m["sharpe"], m["win_rate"], m["profit_factor"])
            type_results[f"{h}d"] = m
        results["by_signal_type"][sig_type] = type_results

    # By direction
    log.info("\n" + "=" * 70)
    log.info("BY DIRECTION")
    log.info("=" * 70)

    for direction in ["bullish", "bearish", "neutral"]:
        subset = signals_df[signals_df["direction"] == direction]
        if len(subset) == 0:
            continue
        log.info("\n--- %s (N=%d) ---", direction, len(subset))
        for h in FORWARD_HORIZONS:
            col = f"fwd_ret_{h}d"
            rets = subset[col].dropna().values
            m = compute_metrics(rets)
            log.info("  %dd: Mean=%.1f bps, Sharpe=%.2f, WR=%.1f%%",
                     h, m["mean_ret_bps"], m["sharpe"], m["win_rate"])

    # By regime
    log.info("\n" + "=" * 70)
    log.info("BY REGIME (SPY vs 200-SMA)")
    log.info("=" * 70)

    for regime in ["bull", "bear", "flat"]:
        subset = signals_df[signals_df["regime"] == regime]
        if len(subset) == 0:
            continue

        log.info("\n--- %s (N=%d) ---", regime.upper(), len(subset))
        regime_results = {}
        for h in FORWARD_HORIZONS:
            col = f"fwd_ret_{h}d"
            rets = subset[col].dropna().values
            m = compute_metrics(rets)
            log.info("  %dd: Mean=%.1f bps, Sharpe=%.2f, WR=%.1f%%, PF=%.2f",
                     h, m["mean_ret_bps"], m["sharpe"], m["win_rate"], m["profit_factor"])
            regime_results[f"{h}d"] = m
        results["by_regime"][regime] = regime_results

    # ══════════════════════════════════════════
    # ADVERSARIAL TESTS
    # ══════════════════════════════════════════
    log.info("\n" + "=" * 70)
    log.info("ADVERSARIAL TESTS")
    log.info("=" * 70)

    adversarial = {}
    # Use 5d horizon as primary for adversarial tests
    primary_horizon = "fwd_ret_5d"
    signal_rets_5d = signals_df[primary_horizon].dropna().values

    # ── Test 1: Permutation test ──
    log.info("\n--- TEST 1: Permutation Test (5d returns, %d shuffles) ---", N_PERMUTATIONS)

    # Build all possible 5d returns from all tickers
    all_5d_returns = []
    for ticker in tickers:
        if ticker not in price_data:
            continue
        df_t = price_data[ticker].sort_values("Date")
        fwd = df_t["Close"].shift(-5) / df_t["Close"] - 1
        all_5d_returns.extend(fwd.dropna().values)
    all_5d_returns = np.array(all_5d_returns)

    p_value = permutation_test(signal_rets_5d, all_5d_returns, N_PERMUTATIONS)
    perm_pass = p_value < 0.05
    log.info("  Signal mean: %.1f bps", np.mean(signal_rets_5d) * 10000)
    log.info("  Baseline mean: %.1f bps", np.mean(all_5d_returns) * 10000)
    log.info("  p-value: %.4f  %s", p_value, "PASS" if perm_pass else "FAIL")
    adversarial["permutation_test"] = {
        "signal_mean_bps": round(np.mean(signal_rets_5d) * 10000, 1),
        "baseline_mean_bps": round(np.mean(all_5d_returns) * 10000, 1),
        "p_value": round(p_value, 4),
        "pass": perm_pass,
    }

    # ── Test 2: Inverse signal ──
    log.info("\n--- TEST 2: Inverse Signal Test ---")
    # "Inverse" = take the same dates but flip sign on direction-aware returns
    # Bullish signals: we go long → inverse = short (negate returns)
    # Bearish signals: we go short → inverse = long (negate returns)
    # Neutral: keep as-is (no direction to flip)
    inverse_rets = []
    for _, row in signals_df.iterrows():
        ret = row.get(primary_horizon, np.nan)
        if np.isnan(ret):
            continue
        if row["direction"] == "bullish":
            inverse_rets.append(-ret)  # short instead of long
        elif row["direction"] == "bearish":
            inverse_rets.append(-ret)  # long instead of short
        else:
            inverse_rets.append(ret)  # neutral, no change

    inverse_rets = np.array(inverse_rets)
    signal_sharpe = compute_metrics(signal_rets_5d)["sharpe"]
    inverse_sharpe = compute_metrics(inverse_rets)["sharpe"]
    inverse_pass = inverse_sharpe < signal_sharpe
    log.info("  Signal Sharpe: %.2f", signal_sharpe)
    log.info("  Inverse Sharpe: %.2f", inverse_sharpe)
    log.info("  Inverse underperforms: %s", "PASS" if inverse_pass else "FAIL")
    adversarial["inverse_signal"] = {
        "signal_sharpe": signal_sharpe,
        "inverse_sharpe": inverse_sharpe,
        "pass": inverse_pass,
    }

    # ── Test 3: Sub-period stability ──
    log.info("\n--- TEST 3: Sub-Period Stability (4 equal periods) ---")
    signals_df["date_ts"] = pd.to_datetime(signals_df["date"])
    sorted_dates = signals_df["date_ts"].sort_values()
    date_range = sorted_dates.max() - sorted_dates.min()
    period_len = date_range / 4

    sub_period_results = []
    all_positive = True
    for i in range(4):
        start = sorted_dates.min() + period_len * i
        end = sorted_dates.min() + period_len * (i + 1)
        mask = (signals_df["date_ts"] >= start) & (signals_df["date_ts"] < end)
        subset = signals_df[mask]

        rets = subset[primary_horizon].dropna().values
        m = compute_metrics(rets)
        mean_positive = m["mean_ret_bps"] > 0
        if not mean_positive:
            all_positive = False

        period_label = f"P{i+1}: {start.strftime('%Y-%m')} to {end.strftime('%Y-%m')}"
        log.info("  %s: N=%d, Mean=%.1f bps, Sharpe=%.2f  %s",
                 period_label, m["n"], m["mean_ret_bps"], m["sharpe"],
                 "OK" if mean_positive else "NEGATIVE")
        sub_period_results.append({
            "period": period_label,
            "metrics": m,
            "positive": mean_positive,
        })

    log.info("  All sub-periods positive: %s", "PASS" if all_positive else "FAIL")
    adversarial["sub_period_stability"] = {
        "periods": sub_period_results,
        "all_positive": all_positive,
        "pass": all_positive,
    }

    # ── Test 4: Regime gap ──
    log.info("\n--- TEST 4: Regime Gap Test ---")
    bull_rets = signals_df[signals_df["regime"] == "bull"][primary_horizon].dropna().values
    bear_rets = signals_df[signals_df["regime"] == "bear"][primary_horizon].dropna().values

    bull_sharpe = compute_metrics(bull_rets)["sharpe"] if len(bull_rets) > 10 else 0
    bear_sharpe = compute_metrics(bear_rets)["sharpe"] if len(bear_rets) > 10 else 0

    max_abs = max(abs(bull_sharpe), abs(bear_sharpe), 1e-8)
    regime_gap = abs(bull_sharpe - bear_sharpe) / max_abs
    regime_pass = regime_gap < REGIME_GAP_THRESHOLD

    log.info("  Bull Sharpe: %.2f (N=%d)", bull_sharpe, len(bull_rets))
    log.info("  Bear Sharpe: %.2f (N=%d)", bear_sharpe, len(bear_rets))
    log.info("  Regime gap: %.2f (threshold: %.2f)  %s",
             regime_gap, REGIME_GAP_THRESHOLD, "PASS" if regime_pass else "FAIL")
    adversarial["regime_gap"] = {
        "bull_sharpe": bull_sharpe,
        "bear_sharpe": bear_sharpe,
        "regime_gap": round(regime_gap, 3),
        "threshold": REGIME_GAP_THRESHOLD,
        "pass": regime_pass,
    }

    results["adversarial_tests"] = adversarial

    # ── Overall pass/fail ──
    tests_passed = sum(1 for t in adversarial.values() if t.get("pass", False))
    total_tests = len(adversarial)
    results["overall"] = {
        "tests_passed": tests_passed,
        "total_tests": total_tests,
        "all_pass": tests_passed == total_tests,
    }

    # ── Summary ──
    log.info("\n" + "=" * 70)
    log.info("OVERALL SUMMARY")
    log.info("=" * 70)
    log.info("Adversarial tests passed: %d/%d", tests_passed, total_tests)
    for name, test in adversarial.items():
        status = "PASS" if test.get("pass") else "FAIL"
        log.info("  %s: %s", name, status)

    best_horizon = max(results["horizons"].items(),
                       key=lambda x: x[1].get("sharpe", 0))
    log.info("\nBest horizon: %s (Sharpe=%.2f, WR=%.1f%%, PF=%.2f)",
             best_horizon[0], best_horizon[1]["sharpe"],
             best_horizon[1]["win_rate"], best_horizon[1]["profit_factor"])

    # Save results
    results["generated_at"] = datetime.utcnow().isoformat()
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info("\nResults saved to %s", OUTPUT_JSON)

    return results


if __name__ == "__main__":
    run_backtest()

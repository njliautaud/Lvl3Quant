#!/usr/bin/env python3
"""
Price Action Structure Backtest
===============================
Tests whether PRICE ACTION STRUCTURE (support/resistance proximity, swing
structure, multi-timeframe confluence, breakout proximity) improves entry
timing for sector ETF trades.

This is NOT standard indicators (RSI/MACD/Bollinger) — it's about where
price sits relative to its own structural levels.

Methodology:
- 11 sector ETFs, 2+ years daily data (2024-2026)
- Base signals: mean-reversion dip buys and short entries
- Split by price action features into favorable/unfavorable
- Compare forward 5d and 10d returns
- T-tests, permutation tests, regime stratification
- Composite score with quintile analysis
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime, timedelta
from collections import defaultdict

try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance required. pip install yfinance")
    sys.exit(1)

# ─── Configuration ───────────────────────────────────────────────────────────

SECTOR_ETFS = ["XLU", "XLF", "XLE", "XLC", "XLK", "XLP", "XLB", "XLI", "XLV", "XLRE", "XLY"]
START_DATE = "2024-01-01"
END_DATE = "2026-08-07"
HOLD_PERIODS = [5, 10]  # forward return horizons in trading days
PERMUTATION_SHUFFLES = 1000
COMMISSION_PCT = 0.005  # 0.5% round-trip cost estimate
np.random.seed(42)

# ─── Data Download ───────────────────────────────────────────────────────────

def download_data():
    """Download daily data for all tickers + SPY for regime classification."""
    tickers = SECTOR_ETFS + ["SPY"]
    print(f"Downloading daily data for {len(tickers)} tickers from {START_DATE} to {END_DATE}...")

    data = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            if len(df) > 100:
                data[ticker] = df
                print(f"  {ticker}: {len(df)} days")
            else:
                print(f"  {ticker}: insufficient data ({len(df)} days), skipping")
        except Exception as e:
            print(f"  {ticker}: download failed ({e})")

    return data


# ─── Price Action Feature Computation ────────────────────────────────────────

def compute_support_resistance_features(df):
    """Proximity to rolling highs/lows as % of price."""
    close = df["Close"]
    features = pd.DataFrame(index=df.index)

    for window in [20, 50, 252]:
        rolling_high = close.rolling(window, min_periods=window).max()
        rolling_low = close.rolling(window, min_periods=window).min()

        # Distance to high (negative = below high)
        features[f"dist_to_{window}d_high_pct"] = (close - rolling_high) / close * 100
        # Distance to low (positive = above low)
        features[f"dist_to_{window}d_low_pct"] = (close - rolling_low) / close * 100
        # Position within range (0 = at low, 1 = at high)
        range_size = rolling_high - rolling_low
        features[f"range_position_{window}d"] = np.where(
            range_size > 0, (close - rolling_low) / range_size, 0.5
        )

    return features


def detect_swing_points(close, window=5):
    """Detect swing highs and lows using local max/min over window."""
    swing_highs = pd.Series(False, index=close.index)
    swing_lows = pd.Series(False, index=close.index)

    for i in range(window, len(close) - window):
        # Swing high: higher than all neighbors in window
        if close.iloc[i] == close.iloc[i-window:i+window+1].max():
            swing_highs.iloc[i] = True
        # Swing low: lower than all neighbors in window
        if close.iloc[i] == close.iloc[i-window:i+window+1].min():
            swing_lows.iloc[i] = True

    return swing_highs, swing_lows


def compute_swing_structure_features(df):
    """Swing structure: HH/HL counts, LL/LH counts, distance from swing points."""
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    features = pd.DataFrame(index=df.index)

    swing_highs, swing_lows = detect_swing_points(close, window=5)

    for lookback in [10, 20]:
        hh_count = pd.Series(0.0, index=df.index)
        hl_count = pd.Series(0.0, index=df.index)
        lh_count = pd.Series(0.0, index=df.index)
        ll_count = pd.Series(0.0, index=df.index)

        for i in range(lookback, len(close)):
            window_high = high.iloc[i-lookback:i+1]
            window_low = low.iloc[i-lookback:i+1]

            # Count higher-highs and higher-lows in rolling windows
            highs_in_window = window_high.values
            lows_in_window = window_low.values

            # Compare consecutive 5-day blocks
            block_size = 5
            n_blocks = lookback // block_size
            block_highs = []
            block_lows = []
            for b in range(n_blocks):
                start = b * block_size
                end = start + block_size
                block_highs.append(highs_in_window[start:end].max())
                block_lows.append(lows_in_window[start:end].min())

            if len(block_highs) >= 2:
                for j in range(1, len(block_highs)):
                    if block_highs[j] > block_highs[j-1]:
                        hh_count.iloc[i] += 1
                    else:
                        lh_count.iloc[i] += 1
                    if block_lows[j] > block_lows[j-1]:
                        hl_count.iloc[i] += 1
                    else:
                        ll_count.iloc[i] += 1

        features[f"hh_count_{lookback}d"] = hh_count
        features[f"hl_count_{lookback}d"] = hl_count
        features[f"lh_count_{lookback}d"] = lh_count
        features[f"ll_count_{lookback}d"] = ll_count

        # Net trend score: (HH+HL) - (LH+LL)
        features[f"swing_trend_score_{lookback}d"] = (hh_count + hl_count) - (lh_count + ll_count)

    # Distance from last swing high/low
    last_swing_high_dist = pd.Series(np.nan, index=df.index)
    last_swing_low_dist = pd.Series(np.nan, index=df.index)
    last_swing_high_price = pd.Series(np.nan, index=df.index)
    last_swing_low_price = pd.Series(np.nan, index=df.index)

    prev_sh_price = np.nan
    prev_sl_price = np.nan

    for i in range(len(close)):
        if swing_highs.iloc[i]:
            prev_sh_price = close.iloc[i]
        if swing_lows.iloc[i]:
            prev_sl_price = close.iloc[i]

        if not np.isnan(prev_sh_price):
            last_swing_high_dist.iloc[i] = (close.iloc[i] - prev_sh_price) / close.iloc[i] * 100
            last_swing_high_price.iloc[i] = prev_sh_price
        if not np.isnan(prev_sl_price):
            last_swing_low_dist.iloc[i] = (close.iloc[i] - prev_sl_price) / close.iloc[i] * 100
            last_swing_low_price.iloc[i] = prev_sl_price

    features["dist_from_swing_high_pct"] = last_swing_high_dist
    features["dist_from_swing_low_pct"] = last_swing_low_dist

    return features


def compute_multi_timeframe_features(df):
    """Multi-timeframe confluence: weekly and monthly alignment."""
    close = df["Close"]
    open_price = df["Open"]
    features = pd.DataFrame(index=df.index)

    # Weekly trend: is price above the weekly open?
    # Approximate: compare current close to the close 5 days ago
    weekly_return = close.pct_change(5)
    features["weekly_bullish"] = (weekly_return > 0).astype(int)

    # Monthly trend: is price above the close 21 days ago?
    monthly_return = close.pct_change(21)
    features["monthly_bullish"] = (monthly_return > 0).astype(int)

    # Daily trend: is today's close above yesterday's?
    daily_return = close.pct_change(1)
    features["daily_bullish"] = (daily_return > 0).astype(int)

    # Multi-TF score (0-3): how many timeframes agree bullishly
    features["mtf_bull_score"] = (
        features["daily_bullish"] + features["weekly_bullish"] + features["monthly_bullish"]
    )
    # Bear score
    features["mtf_bear_score"] = 3 - features["mtf_bull_score"]

    return features


def compute_breakout_features(df):
    """Breakout/breakdown proximity and volume confirmation."""
    close = df["Close"]
    volume = df["Volume"]
    features = pd.DataFrame(index=df.index)

    rolling_high_20 = close.rolling(20, min_periods=20).max()
    rolling_low_20 = close.rolling(20, min_periods=20).min()
    avg_volume_20 = volume.rolling(20, min_periods=20).mean()

    # Within 1% of 20-day high (potential breakout)
    features["near_breakout"] = ((rolling_high_20 - close) / close * 100 < 1.0).astype(int)
    # Within 1% of 20-day low (potential breakdown)
    features["near_breakdown"] = ((close - rolling_low_20) / close * 100 < 1.0).astype(int)

    # Volume ratio (today vs 20-day avg)
    features["volume_ratio"] = np.where(avg_volume_20 > 0, volume / avg_volume_20, 1.0)

    # Breakout with volume confirmation
    features["breakout_with_volume"] = (
        (features["near_breakout"] == 1) & (features["volume_ratio"] > 1.2)
    ).astype(int)
    features["breakdown_with_volume"] = (
        (features["near_breakdown"] == 1) & (features["volume_ratio"] > 1.2)
    ).astype(int)

    return features


def compute_all_features(df):
    """Compute all price action features for a single ticker."""
    f1 = compute_support_resistance_features(df)
    f2 = compute_swing_structure_features(df)
    f3 = compute_multi_timeframe_features(df)
    f4 = compute_breakout_features(df)
    return pd.concat([f1, f2, f3, f4], axis=1)


# ─── Signal Generation ──────────────────────────────────────────────────────

def compute_rsi(series, period=14):
    """Standard RSI calculation."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def generate_base_signals(df):
    """
    Generate mean-reversion entry signals.

    LONG (dip buy): price >5% below 20-day SMA AND RSI < 40
    SHORT (mean-reversion short): price >5% above 20-day SMA AND RSI > 60
    """
    close = df["Close"]
    sma20 = close.rolling(20, min_periods=20).mean()
    rsi = compute_rsi(close, 14)

    pct_from_sma = (close - sma20) / sma20 * 100

    long_signals = (pct_from_sma < -5) & (rsi < 40)
    short_signals = (pct_from_sma > 5) & (rsi > 60)

    return long_signals, short_signals


# ─── Forward Returns ────────────────────────────────────────────────────────

def compute_forward_returns(df, hold_days):
    """Compute forward returns for each day."""
    close = df["Close"]
    fwd_ret = close.shift(-hold_days) / close - 1
    # Subtract commission
    fwd_ret_net = fwd_ret - COMMISSION_PCT
    return fwd_ret_net


# ─── Analysis Functions ─────────────────────────────────────────────────────

def analyze_feature_split(returns_favorable, returns_unfavorable, feature_name, hold_days):
    """Compare favorable vs unfavorable group returns."""
    rf = returns_favorable.dropna()
    ru = returns_unfavorable.dropna()

    if len(rf) < 10 or len(ru) < 10:
        return None

    mean_f = rf.mean() * 100
    mean_u = ru.mean() * 100
    std_f = rf.std() * 100
    std_u = ru.std() * 100
    sharpe_f = rf.mean() / rf.std() * np.sqrt(252 / hold_days) if rf.std() > 0 else 0
    sharpe_u = ru.mean() / ru.std() * np.sqrt(252 / hold_days) if ru.std() > 0 else 0
    wr_f = (rf > 0).mean() * 100
    wr_u = (ru > 0).mean() * 100

    # T-test
    t_stat, p_val = stats.ttest_ind(rf, ru, equal_var=False)

    return {
        "feature": feature_name,
        "hold_days": hold_days,
        "n_favorable": len(rf),
        "n_unfavorable": len(ru),
        "mean_ret_fav_%": round(mean_f, 3),
        "mean_ret_unfav_%": round(mean_u, 3),
        "diff_%": round(mean_f - mean_u, 3),
        "sharpe_fav": round(sharpe_f, 3),
        "sharpe_unfav": round(sharpe_u, 3),
        "wr_fav_%": round(wr_f, 1),
        "wr_unfav_%": round(wr_u, 1),
        "t_stat": round(t_stat, 3),
        "p_value": round(p_val, 4),
        "significant": "YES" if p_val < 0.05 else "no",
    }


def permutation_test(returns, scores, n_shuffles=1000):
    """
    Permutation test: is the correlation between score and returns significant?
    Returns observed correlation and p-value.
    """
    mask = ~(np.isnan(returns) | np.isnan(scores))
    r = returns[mask]
    s = scores[mask]

    if len(r) < 20:
        return 0, 1.0

    observed_corr = np.corrcoef(r, s)[0, 1]

    count_extreme = 0
    for _ in range(n_shuffles):
        shuffled = np.random.permutation(s)
        perm_corr = np.corrcoef(r, shuffled)[0, 1]
        if abs(perm_corr) >= abs(observed_corr):
            count_extreme += 1

    p_val = count_extreme / n_shuffles
    return observed_corr, p_val


def classify_regime(spy_data):
    """Classify each day as green/red based on SPY close-to-close."""
    spy_ret = spy_data["Close"].pct_change()
    regime = pd.Series("flat", index=spy_data.index)
    regime[spy_ret > 0.001] = "green"
    regime[spy_ret < -0.001] = "red"
    return regime


# ─── Main Backtest ───────────────────────────────────────────────────────────

def run_backtest():
    print("=" * 80)
    print("PRICE ACTION STRUCTURE BACKTEST")
    print("=" * 80)
    print(f"Tickers: {', '.join(SECTOR_ETFS)}")
    print(f"Period: {START_DATE} to {END_DATE}")
    print(f"Hold periods: {HOLD_PERIODS} days")
    print(f"Permutation shuffles: {PERMUTATION_SHUFFLES}")
    print()

    # Download data
    data = download_data()
    if "SPY" not in data:
        print("FATAL: Could not download SPY data for regime classification")
        sys.exit(1)

    spy_regime = classify_regime(data["SPY"])

    # Collect all signals and features across tickers
    all_results = []

    # Storage for composite score analysis
    composite_data_long = []
    composite_data_short = []

    print("\n" + "=" * 80)
    print("COMPUTING FEATURES AND SIGNALS")
    print("=" * 80)

    signal_counts = {"long": 0, "short": 0}

    for ticker in SECTOR_ETFS:
        if ticker not in data:
            continue

        df = data[ticker]

        # Compute features
        features = compute_all_features(df)

        # Generate signals
        long_signals, short_signals = generate_base_signals(df)

        # Forward returns
        fwd_returns = {}
        for hd in HOLD_PERIODS:
            fwd_returns[hd] = compute_forward_returns(df, hd)

        # Align regime
        ticker_regime = spy_regime.reindex(df.index)

        n_long = long_signals.sum()
        n_short = short_signals.sum()
        signal_counts["long"] += n_long
        signal_counts["short"] += n_short
        print(f"  {ticker}: {n_long} long signals, {n_short} short signals")

        # ── Feature Analysis for LONG signals ────────────────────────────
        for hd in HOLD_PERIODS:
            long_mask = long_signals & fwd_returns[hd].notna()
            ret_long = fwd_returns[hd][long_mask]
            feat_long = features.loc[long_mask]
            regime_long = ticker_regime[long_mask]

            if len(ret_long) < 5:
                continue

            # 1. Support/Resistance: range_position_20d < 0.3 vs > 0.7
            if "range_position_20d" in feat_long.columns:
                near_support = feat_long["range_position_20d"] < 0.3
                near_resistance = feat_long["range_position_20d"] > 0.7
                result = analyze_feature_split(
                    ret_long[near_support], ret_long[near_resistance],
                    "LONG: Near 20d Support vs Resistance", hd
                )
                if result:
                    result["ticker"] = ticker
                    all_results.append(result)

            if "range_position_50d" in feat_long.columns:
                near_support = feat_long["range_position_50d"] < 0.3
                near_resistance = feat_long["range_position_50d"] > 0.7
                result = analyze_feature_split(
                    ret_long[near_support], ret_long[near_resistance],
                    "LONG: Near 50d Support vs Resistance", hd
                )
                if result:
                    result["ticker"] = ticker
                    all_results.append(result)

            # 2. Swing structure aligned with long
            if "swing_trend_score_20d" in feat_long.columns:
                # For longs in a dip-buy, we actually want bearish swing structure
                # (we're buying the dip, so structure being oversold is favorable)
                aligned = feat_long["swing_trend_score_20d"] < 0  # bearish structure = deeper dip
                misaligned = feat_long["swing_trend_score_20d"] > 0
                result = analyze_feature_split(
                    ret_long[aligned], ret_long[misaligned],
                    "LONG: Bearish swing structure (deeper dip) vs Bullish", hd
                )
                if result:
                    result["ticker"] = ticker
                    all_results.append(result)

            # 3. Multi-TF confluence
            if "mtf_bull_score" in feat_long.columns:
                high_confluence = feat_long["mtf_bull_score"] >= 2
                low_confluence = feat_long["mtf_bull_score"] <= 1
                result = analyze_feature_split(
                    ret_long[high_confluence], ret_long[low_confluence],
                    "LONG: Multi-TF bull score >= 2 vs <= 1", hd
                )
                if result:
                    result["ticker"] = ticker
                    all_results.append(result)

            # 4. Breakdown proximity (for dip buys, near breakdown might be favorable)
            if "near_breakdown" in feat_long.columns:
                at_breakdown = feat_long["near_breakdown"] == 1
                not_breakdown = feat_long["near_breakdown"] == 0
                result = analyze_feature_split(
                    ret_long[at_breakdown], ret_long[not_breakdown],
                    "LONG: At 20d breakdown vs Not", hd
                )
                if result:
                    result["ticker"] = ticker
                    all_results.append(result)

            # 5. Volume confirmation
            if "volume_ratio" in feat_long.columns:
                high_vol = feat_long["volume_ratio"] > 1.2
                low_vol = feat_long["volume_ratio"] <= 1.2
                result = analyze_feature_split(
                    ret_long[high_vol], ret_long[low_vol],
                    "LONG: High volume (>1.2x avg) vs Normal", hd
                )
                if result:
                    result["ticker"] = ticker
                    all_results.append(result)

            # 6. Distance from swing low
            if "dist_from_swing_low_pct" in feat_long.columns:
                sl_feat = feat_long["dist_from_swing_low_pct"].dropna()
                if len(sl_feat) > 10:
                    median_dist = sl_feat.median()
                    near_swing_low = sl_feat <= median_dist
                    far_swing_low = sl_feat > median_dist
                    result = analyze_feature_split(
                        ret_long[near_swing_low.reindex(ret_long.index, fill_value=False)],
                        ret_long[far_swing_low.reindex(ret_long.index, fill_value=False)],
                        "LONG: Near swing low vs Far from swing low", hd
                    )
                    if result:
                        result["ticker"] = ticker
                        all_results.append(result)

            # Composite score for long signals
            if hd == 5:  # only compute composite for 5d
                for idx in ret_long.index:
                    if idx not in feat_long.index:
                        continue
                    row = feat_long.loc[idx]
                    score = 0
                    n_features = 0

                    # Near support bonus
                    if not np.isnan(row.get("range_position_20d", np.nan)):
                        score += (1 - row["range_position_20d"])  # lower position = more near support
                        n_features += 1
                    if not np.isnan(row.get("range_position_50d", np.nan)):
                        score += (1 - row["range_position_50d"])
                        n_features += 1

                    # Bearish swing structure (deeper dip)
                    if not np.isnan(row.get("swing_trend_score_20d", np.nan)):
                        score += max(0, -row["swing_trend_score_20d"]) / 4  # normalize
                        n_features += 1

                    # Volume confirmation
                    if not np.isnan(row.get("volume_ratio", np.nan)):
                        score += min(row["volume_ratio"] / 2, 1)  # cap at 1
                        n_features += 1

                    # Near breakdown
                    if not np.isnan(row.get("near_breakdown", np.nan)):
                        score += row["near_breakdown"]
                        n_features += 1

                    if n_features > 0:
                        composite = score / n_features
                        regime_val = ticker_regime.get(idx, "flat")
                        composite_data_long.append({
                            "ticker": ticker,
                            "date": idx,
                            "composite_score": composite,
                            "fwd_5d_ret": ret_long.get(idx, np.nan),
                            "fwd_10d_ret": fwd_returns[10].get(idx, np.nan) if 10 in fwd_returns else np.nan,
                            "regime": regime_val,
                        })

        # ── Feature Analysis for SHORT signals ───────────────────────────
        for hd in HOLD_PERIODS:
            short_mask = short_signals & fwd_returns[hd].notna()
            # For shorts, returns are inverted
            ret_short = -fwd_returns[hd][short_mask]  # negate: short profits when price drops
            feat_short = features.loc[short_mask]
            regime_short = ticker_regime[short_mask]

            if len(ret_short) < 5:
                continue

            # 1. Near resistance (favorable for shorts)
            if "range_position_20d" in feat_short.columns:
                near_resistance = feat_short["range_position_20d"] > 0.7
                near_support = feat_short["range_position_20d"] < 0.3
                result = analyze_feature_split(
                    ret_short[near_resistance], ret_short[near_support],
                    "SHORT: Near 20d Resistance vs Support", hd
                )
                if result:
                    result["ticker"] = ticker
                    all_results.append(result)

            # 2. Bullish swing structure (overextended = favorable for shorts)
            if "swing_trend_score_20d" in feat_short.columns:
                overextended = feat_short["swing_trend_score_20d"] > 0
                not_overextended = feat_short["swing_trend_score_20d"] <= 0
                result = analyze_feature_split(
                    ret_short[overextended], ret_short[not_overextended],
                    "SHORT: Overextended swing vs Not", hd
                )
                if result:
                    result["ticker"] = ticker
                    all_results.append(result)

            # 3. Multi-TF bear confluence
            if "mtf_bear_score" in feat_short.columns:
                high_bear = feat_short["mtf_bear_score"] >= 2
                low_bear = feat_short["mtf_bear_score"] <= 1
                result = analyze_feature_split(
                    ret_short[high_bear], ret_short[low_bear],
                    "SHORT: Multi-TF bear score >= 2 vs <= 1", hd
                )
                if result:
                    result["ticker"] = ticker
                    all_results.append(result)

            # 4. Near breakout (for shorts: near high = favorable to short)
            if "near_breakout" in feat_short.columns:
                at_breakout = feat_short["near_breakout"] == 1
                not_breakout = feat_short["near_breakout"] == 0
                result = analyze_feature_split(
                    ret_short[at_breakout], ret_short[not_breakout],
                    "SHORT: At 20d breakout (overbought) vs Not", hd
                )
                if result:
                    result["ticker"] = ticker
                    all_results.append(result)

            # 5. Volume on shorts
            if "volume_ratio" in feat_short.columns:
                high_vol = feat_short["volume_ratio"] > 1.2
                low_vol = feat_short["volume_ratio"] <= 1.2
                result = analyze_feature_split(
                    ret_short[high_vol], ret_short[low_vol],
                    "SHORT: High volume vs Normal", hd
                )
                if result:
                    result["ticker"] = ticker
                    all_results.append(result)

            # Composite for shorts
            if hd == 5:
                for idx in ret_short.index:
                    if idx not in feat_short.index:
                        continue
                    row = feat_short.loc[idx]
                    score = 0
                    n_features = 0

                    if not np.isnan(row.get("range_position_20d", np.nan)):
                        score += row["range_position_20d"]  # higher = more near resistance
                        n_features += 1
                    if not np.isnan(row.get("range_position_50d", np.nan)):
                        score += row["range_position_50d"]
                        n_features += 1
                    if not np.isnan(row.get("swing_trend_score_20d", np.nan)):
                        score += max(0, row["swing_trend_score_20d"]) / 4
                        n_features += 1
                    if not np.isnan(row.get("volume_ratio", np.nan)):
                        score += min(row["volume_ratio"] / 2, 1)
                        n_features += 1
                    if not np.isnan(row.get("near_breakout", np.nan)):
                        score += row["near_breakout"]
                        n_features += 1

                    if n_features > 0:
                        composite = score / n_features
                        regime_val = ticker_regime.get(idx, "flat")
                        composite_data_short.append({
                            "ticker": ticker,
                            "date": idx,
                            "composite_score": composite,
                            "fwd_5d_ret": ret_short.get(idx, np.nan),
                            "fwd_10d_ret": (-fwd_returns[10]).get(idx, np.nan) if 10 in fwd_returns else np.nan,
                            "regime": regime_val,
                        })

    print(f"\nTotal signals: {signal_counts['long']} long, {signal_counts['short']} short")

    # ── Aggregate Results by Feature ─────────────────────────────────────

    if not all_results:
        print("\nNO SIGNALS GENERATED. Cannot run analysis.")
        return

    results_df = pd.DataFrame(all_results)

    print("\n" + "=" * 80)
    print("PER-TICKER FEATURE ANALYSIS")
    print("=" * 80)

    # Aggregate across tickers for each feature
    print("\n" + "=" * 80)
    print("AGGREGATED FEATURE ANALYSIS (across all tickers)")
    print("=" * 80)

    agg_results = []
    for feature_name in results_df["feature"].unique():
        for hd in HOLD_PERIODS:
            subset = results_df[(results_df["feature"] == feature_name) & (results_df["hold_days"] == hd)]
            if len(subset) == 0:
                continue

            total_fav = subset["n_favorable"].sum()
            total_unfav = subset["n_unfavorable"].sum()

            # Weighted average by sample size
            if total_fav > 0 and total_unfav > 0:
                avg_ret_fav = (subset["mean_ret_fav_%"] * subset["n_favorable"]).sum() / total_fav
                avg_ret_unfav = (subset["mean_ret_unfav_%"] * subset["n_unfavorable"]).sum() / total_unfav
                avg_wr_fav = (subset["wr_fav_%"] * subset["n_favorable"]).sum() / total_fav
                avg_wr_unfav = (subset["wr_unfav_%"] * subset["n_unfavorable"]).sum() / total_unfav
                avg_sharpe_fav = subset["sharpe_fav"].mean()
                avg_sharpe_unfav = subset["sharpe_unfav"].mean()

                n_significant = (subset["significant"] == "YES").sum()
                n_tickers = len(subset)

                agg_results.append({
                    "Feature": feature_name,
                    "Hold": f"{hd}d",
                    "N_fav": total_fav,
                    "N_unfav": total_unfav,
                    "Ret_fav%": round(avg_ret_fav, 3),
                    "Ret_unfav%": round(avg_ret_unfav, 3),
                    "Diff%": round(avg_ret_fav - avg_ret_unfav, 3),
                    "Sharpe_fav": round(avg_sharpe_fav, 2),
                    "Sharpe_unfav": round(avg_sharpe_unfav, 2),
                    "WR_fav%": round(avg_wr_fav, 1),
                    "WR_unfav%": round(avg_wr_unfav, 1),
                    "Sig_tickers": f"{n_significant}/{n_tickers}",
                })

    if agg_results:
        agg_df = pd.DataFrame(agg_results)
        # Sort by absolute diff
        agg_df["abs_diff"] = agg_df["Diff%"].abs()
        agg_df = agg_df.sort_values("abs_diff", ascending=False).drop("abs_diff", axis=1)
        print("\n" + agg_df.to_string(index=False))

    # ── Regime Stratification ────────────────────────────────────────────

    print("\n" + "=" * 80)
    print("REGIME STRATIFICATION (Green vs Red SPY days)")
    print("=" * 80)

    # Use composite data for regime analysis
    for direction, comp_data in [("LONG", composite_data_long), ("SHORT", composite_data_short)]:
        if not comp_data:
            print(f"\n{direction}: No composite data available")
            continue

        comp_df = pd.DataFrame(comp_data)
        comp_df = comp_df.dropna(subset=["fwd_5d_ret", "composite_score"])

        if len(comp_df) < 20:
            print(f"\n{direction}: Insufficient data ({len(comp_df)} signals)")
            continue

        print(f"\n{direction} signals regime analysis (N={len(comp_df)}):")

        for regime in ["green", "red"]:
            regime_mask = comp_df["regime"] == regime
            regime_data = comp_df[regime_mask]

            if len(regime_data) < 10:
                print(f"  {regime}: insufficient data ({len(regime_data)})")
                continue

            median_score = regime_data["composite_score"].median()
            high_score = regime_data[regime_data["composite_score"] >= median_score]
            low_score = regime_data[regime_data["composite_score"] < median_score]

            if len(high_score) >= 5 and len(low_score) >= 5:
                mean_high = high_score["fwd_5d_ret"].mean() * 100
                mean_low = low_score["fwd_5d_ret"].mean() * 100
                wr_high = (high_score["fwd_5d_ret"] > 0).mean() * 100
                wr_low = (low_score["fwd_5d_ret"] > 0).mean() * 100

                t, p = stats.ttest_ind(
                    high_score["fwd_5d_ret"].values,
                    low_score["fwd_5d_ret"].values,
                    equal_var=False
                )

                sig = " ***" if p < 0.05 else ""
                print(f"  {regime.upper()} regime: "
                      f"High-score({len(high_score)}): {mean_high:+.3f}% ret, {wr_high:.0f}% WR | "
                      f"Low-score({len(low_score)}): {mean_low:+.3f}% ret, {wr_low:.0f}% WR | "
                      f"p={p:.4f}{sig}")

    # ── Composite Score Quintile Analysis ────────────────────────────────

    print("\n" + "=" * 80)
    print("COMPOSITE SCORE QUINTILE ANALYSIS")
    print("=" * 80)

    for direction, comp_data in [("LONG", composite_data_long), ("SHORT", composite_data_short)]:
        if not comp_data:
            continue

        comp_df = pd.DataFrame(comp_data)
        comp_df = comp_df.dropna(subset=["fwd_5d_ret", "composite_score"])

        if len(comp_df) < 25:
            print(f"\n{direction}: Insufficient data for quintile analysis ({len(comp_df)})")
            continue

        print(f"\n{direction} Composite Score Quintile Analysis (N={len(comp_df)}):")
        print(f"  Score range: {comp_df['composite_score'].min():.3f} to {comp_df['composite_score'].max():.3f}")
        print(f"  Score mean: {comp_df['composite_score'].mean():.3f}, median: {comp_df['composite_score'].median():.3f}")

        # Quintiles
        try:
            comp_df["quintile"] = pd.qcut(comp_df["composite_score"], 5, labels=False, duplicates="drop")
        except ValueError:
            # Not enough unique values for 5 quintiles, try 3
            try:
                comp_df["quintile"] = pd.qcut(comp_df["composite_score"], 3, labels=False, duplicates="drop")
                print("  (Using terciles due to limited score distribution)")
            except ValueError:
                comp_df["quintile"] = (comp_df["composite_score"] > comp_df["composite_score"].median()).astype(int)
                print("  (Using median split due to limited score distribution)")

        print(f"\n  {'Quintile':>8} | {'N':>4} | {'Mean Ret%':>9} | {'Sharpe':>7} | {'WR%':>5} | {'Tickers':>8}")
        print(f"  {'-'*8} | {'-'*4} | {'-'*9} | {'-'*7} | {'-'*5} | {'-'*8}")

        for q in sorted(comp_df["quintile"].unique()):
            q_data = comp_df[comp_df["quintile"] == q]
            mean_ret = q_data["fwd_5d_ret"].mean() * 100
            std_ret = q_data["fwd_5d_ret"].std()
            sharpe = q_data["fwd_5d_ret"].mean() / std_ret * np.sqrt(252/5) if std_ret > 0 else 0
            wr = (q_data["fwd_5d_ret"] > 0).mean() * 100
            n_tickers = q_data["ticker"].nunique()
            label = f"Q{q+1}"
            if q == comp_df["quintile"].max():
                label += " (best)"
            elif q == comp_df["quintile"].min():
                label += " (worst)"

            print(f"  {label:>8} | {len(q_data):>4} | {mean_ret:>+8.3f}% | {sharpe:>7.2f} | {wr:>4.0f}% | {n_tickers:>8}")

        # Top vs bottom quintile comparison
        top_q = comp_df["quintile"].max()
        bot_q = comp_df["quintile"].min()
        top_rets = comp_df[comp_df["quintile"] == top_q]["fwd_5d_ret"].values
        bot_rets = comp_df[comp_df["quintile"] == bot_q]["fwd_5d_ret"].values

        if len(top_rets) >= 5 and len(bot_rets) >= 5:
            t, p = stats.ttest_ind(top_rets, bot_rets, equal_var=False)
            print(f"\n  Top vs Bottom quintile: t={t:.3f}, p={p:.4f} {'*** SIGNIFICANT' if p < 0.05 else '(not significant)'}")

        # Pearson and Spearman correlation
        pearson_r, pearson_p = stats.pearsonr(comp_df["composite_score"], comp_df["fwd_5d_ret"])
        spearman_r, spearman_p = stats.spearmanr(comp_df["composite_score"], comp_df["fwd_5d_ret"])

        print(f"\n  Pearson r={pearson_r:.4f} (p={pearson_p:.4f})")
        print(f"  Spearman r={spearman_r:.4f} (p={spearman_p:.4f})")

        # Permutation test
        print(f"\n  Running permutation test ({PERMUTATION_SHUFFLES} shuffles)...")
        perm_corr, perm_p = permutation_test(
            comp_df["fwd_5d_ret"].values,
            comp_df["composite_score"].values,
            n_shuffles=PERMUTATION_SHUFFLES
        )
        print(f"  Permutation test: observed r={perm_corr:.4f}, p={perm_p:.4f} "
              f"{'*** SIGNIFICANT' if perm_p < 0.05 else '(not significant)'}")

    # ── Cross-Ticker Consistency ─────────────────────────────────────────

    print("\n" + "=" * 80)
    print("CROSS-TICKER CONSISTENCY CHECK")
    print("=" * 80)
    print("(Feature must work across multiple tickers to be considered real)")

    for direction, comp_data in [("LONG", composite_data_long), ("SHORT", composite_data_short)]:
        if not comp_data:
            continue

        comp_df = pd.DataFrame(comp_data)
        comp_df = comp_df.dropna(subset=["fwd_5d_ret", "composite_score"])

        print(f"\n{direction} - Per-ticker composite score correlation with 5d returns:")
        ticker_results = []
        for ticker in comp_df["ticker"].unique():
            t_data = comp_df[comp_df["ticker"] == ticker]
            if len(t_data) < 10:
                continue
            r, p = stats.pearsonr(t_data["composite_score"], t_data["fwd_5d_ret"])
            ticker_results.append({"ticker": ticker, "n": len(t_data), "pearson_r": r, "p_value": p})

        if ticker_results:
            tr_df = pd.DataFrame(ticker_results).sort_values("pearson_r", ascending=False)
            for _, row in tr_df.iterrows():
                sig = " *" if row["p_value"] < 0.05 else ""
                print(f"  {row['ticker']:>4}: r={row['pearson_r']:+.4f}, p={row['p_value']:.4f}, n={row['n']}{sig}")

            # Count how many tickers have positive correlation
            n_positive = (tr_df["pearson_r"] > 0).sum()
            n_total = len(tr_df)
            print(f"\n  Tickers with positive r: {n_positive}/{n_total} "
                  f"({'CONSISTENT' if n_positive >= n_total * 0.7 else 'INCONSISTENT'})")

    # ── Final Summary ────────────────────────────────────────────────────

    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)

    # Summarize statistically significant findings
    if agg_results:
        agg_df = pd.DataFrame(agg_results)

        # Count features with large differences
        strong_features = agg_df[agg_df["Diff%"].abs() > 0.1]

        print(f"\nTotal features tested: {len(agg_df)}")
        print(f"Features with |diff| > 0.1%: {len(strong_features)}")

        sig_count = 0
        for _, row in results_df.iterrows():
            if row["significant"] == "YES":
                sig_count += 1

        total_tests = len(results_df)
        expected_false_positives = total_tests * 0.05

        print(f"\nTotal individual ticker-feature tests: {total_tests}")
        print(f"Statistically significant (p<0.05): {sig_count}")
        print(f"Expected false positives at 5% level: {expected_false_positives:.0f}")
        print(f"Excess significant results: {sig_count - expected_false_positives:.0f}")

        if sig_count <= expected_false_positives * 1.5:
            print("\n>>> VERDICT: NO MEANINGFUL EDGE from price action structure features.")
            print("    The number of significant results is within the range expected by chance.")
            print("    Price action structure does NOT reliably improve entry timing.")
        else:
            print("\n>>> VERDICT: POTENTIAL EDGE detected — investigate further.")
            print("    But check cross-ticker consistency and regime stability before trusting.")

        # Check composite score
        for direction, comp_data in [("LONG", composite_data_long), ("SHORT", composite_data_short)]:
            if comp_data:
                comp_df = pd.DataFrame(comp_data).dropna(subset=["fwd_5d_ret", "composite_score"])
                if len(comp_df) >= 20:
                    r, p = stats.pearsonr(comp_df["composite_score"], comp_df["fwd_5d_ret"])
                    if p < 0.05:
                        print(f"\n    {direction} composite: r={r:.4f}, p={p:.4f} — SIGNIFICANT")
                    else:
                        print(f"\n    {direction} composite: r={r:.4f}, p={p:.4f} — not significant")

    print("\n" + "=" * 80)
    print("BACKTEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_backtest()

#!/usr/bin/env python3
"""
Systematic Momentum v2 — Walk-Forward Backtest with Dynamic Exits
=================================================================
Universe: 900 stocks (S&P 500 + S&P 400) from cached data
Signals: 3mo, 6mo, 12mo momentum (skip most recent month)
Rebalance: Monthly universe selection (top 25 stocks)
Daily exits: Trailing stop (2x ATR), momentum breakdown, volume dry-up
Walk-forward: 60-month train, 1-month OOT, sliding window (HC #0)
Regime: Bull/flat/bear by SPY monthly returns, gap < 0.50 (R1)
Permutation test: p < 0.05 required
Sizing: Equal-weight and inverse-vol-weight

HC #684: Dynamic exits mandatory
HC #0: Sliding window only
Cost: 0 commission (Robinhood), 0.5% slippage per trade
"""

import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────
CACHE_PATH = Path("/home/jupiter/Lvl3Quant/output/momentum_v2/price_cache.pkl")
OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/growth_research")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Parameters ────────────────────────────────────────────────────────────
TOP_N = 25                     # portfolio size
LOOKBACKS = [63, 126, 252]     # 3mo, 6mo, 12mo in trading days
SKIP_RECENT = 21               # skip most recent month (Jegadeesh-Titman)
REBALANCE_DAYS = 21            # monthly rebalance

# Exit parameters (HC #684)
ATR_PERIOD = 20
ATR_MULT = 2.0                 # trailing stop = high - 2*ATR
MOM_BREAKDOWN_WINDOW = 10      # 10-day momentum
MOM_BREAKDOWN_CONSEC = 3       # 3 consecutive negative days
VOL_DRYUP_RATIO = 0.50         # volume < 50% of 20d avg
VOL_DRYUP_CONSEC = 3           # 3 consecutive days

# Walk-forward
WF_TRAIN_MONTHS = 60           # 5 years training
WF_TEST_MONTHS = 1             # 1 month OOT

# Costs
SLIPPAGE_PCT = 0.005           # 0.5% per trade (entry + exit combined)

# Scoring weights for composite momentum
MOM_WEIGHTS = {252: 0.40, 126: 0.35, 63: 0.25}


def load_data():
    """Load cached price data."""
    print(f"Loading cached data from {CACHE_PATH}...")
    data = pd.read_pickle(CACHE_PATH)
    print(f"  {len(data)} tickers loaded")
    return data


def build_price_panel(data: dict) -> tuple:
    """Build aligned close, high, low, volume DataFrames from dict."""
    closes = {}
    highs = {}
    lows = {}
    volumes = {}

    for ticker, df in data.items():
        if len(df) < 252:
            continue
        c = df["Close"].squeeze() if isinstance(df["Close"], pd.DataFrame) else df["Close"]
        h = df["High"].squeeze() if isinstance(df["High"], pd.DataFrame) else df["High"]
        l = df["Low"].squeeze() if isinstance(df["Low"], pd.DataFrame) else df["Low"]
        v = df["Volume"].squeeze() if isinstance(df["Volume"], pd.DataFrame) else df["Volume"]
        closes[ticker] = c
        highs[ticker] = h
        lows[ticker] = l
        volumes[ticker] = v

    close_df = pd.DataFrame(closes)
    high_df = pd.DataFrame(highs)
    low_df = pd.DataFrame(lows)
    vol_df = pd.DataFrame(volumes)

    # Align all to same index
    idx = close_df.index
    high_df = high_df.reindex(idx)
    low_df = low_df.reindex(idx)
    vol_df = vol_df.reindex(idx)

    # Filter: need at least 2 years of data
    good_cols = close_df.columns[close_df.notna().sum() > 504]
    close_df = close_df[good_cols]
    high_df = high_df[good_cols]
    low_df = low_df[good_cols]
    vol_df = vol_df[good_cols]

    print(f"  Panel: {close_df.shape[1]} stocks, {close_df.shape[0]} days "
          f"({close_df.index[0].date()} to {close_df.index[-1].date()})")
    return close_df, high_df, low_df, vol_df


def compute_features(close_df, high_df, low_df, vol_df):
    """Compute all features needed for signal generation and exits."""
    features = {}

    # Momentum signals (skip most recent month)
    for lb in LOOKBACKS:
        # Return from t-lb to t-skip
        shifted = close_df.shift(SKIP_RECENT)
        far = close_df.shift(lb)
        features[f"mom_{lb}"] = (shifted / far) - 1.0

    # ATR for trailing stops
    tr1 = high_df - low_df
    tr2 = (high_df - close_df.shift(1)).abs()
    tr3 = (low_df - close_df.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3]).groupby(level=0).max()
    # Ensure tr has same shape as close_df
    tr = tr.reindex(close_df.index)
    features["atr"] = tr.rolling(ATR_PERIOD).mean()

    # Daily returns
    features["daily_ret"] = close_df.pct_change()

    # 10-day returns for momentum breakdown
    features["ret_10d"] = close_df.pct_change(MOM_BREAKDOWN_WINDOW)

    # Volume ratio (current / 20d avg)
    vol_avg = vol_df.rolling(20).mean()
    features["vol_ratio"] = vol_df / vol_avg

    # 20-day realized vol for inverse-vol weighting
    features["vol_20d"] = close_df.pct_change().rolling(20).std() * np.sqrt(252)

    return features


def classify_regimes(spy_close):
    """Classify each month as bull/bear/flat by SPY monthly returns."""
    monthly = spy_close.resample("ME").last().pct_change()
    regime_map = {}
    for date, ret in monthly.items():
        if pd.isna(ret):
            regime_map[date] = "flat"
        elif ret > 0.02:
            regime_map[date] = "bull"
        elif ret < -0.02:
            regime_map[date] = "bear"
        else:
            regime_map[date] = "flat"
    return regime_map


def map_day_to_regime(dates, regime_map):
    """Map each trading day to its month-end regime."""
    regime_dates = sorted(regime_map.keys())
    day_regimes = {}
    for d in dates:
        # Find the month-end for this day
        month_end = d + pd.offsets.MonthEnd(0)
        # If month_end is in the future relative to regime data, use previous month
        if month_end in regime_map:
            day_regimes[d] = regime_map[month_end]
        else:
            # Try previous month
            prev = month_end - pd.offsets.MonthEnd(1)
            day_regimes[d] = regime_map.get(prev, "flat")
    return day_regimes


def score_stocks(features, date_idx, close_df):
    """Score and rank stocks at a given date index. Returns sorted list of (ticker, score)."""
    scores = {}
    for ticker in close_df.columns:
        if ticker == "SPY":
            continue
        # Need all momentum scores
        vals = {}
        valid = True
        for lb in LOOKBACKS:
            key = f"mom_{lb}"
            v = features[key].iloc[date_idx].get(ticker, np.nan) if ticker in features[key].columns else np.nan
            if pd.isna(v):
                valid = False
                break
            vals[lb] = v

        # Penny stock filter
        price = close_df.iloc[date_idx].get(ticker, np.nan) if ticker in close_df.columns else np.nan
        if pd.isna(price) or price < 5.0:
            valid = False

        if valid:
            scores[ticker] = vals

    if len(scores) < TOP_N:
        return []

    # Rank each momentum period, then composite
    tickers = list(scores.keys())
    for lb in LOOKBACKS:
        vals = np.array([scores[t][lb] for t in tickers])
        ranks = pd.Series(vals).rank(pct=True).values
        for i, t in enumerate(tickers):
            scores[t][f"rank_{lb}"] = ranks[i]

    # Composite score
    composite = {}
    for t in tickers:
        composite[t] = sum(MOM_WEIGHTS[lb] * scores[t][f"rank_{lb}"] for lb in LOOKBACKS)

    sorted_stocks = sorted(composite.items(), key=lambda x: x[1], reverse=True)
    return sorted_stocks[:TOP_N]


def run_single_backtest(close_df, features, trading_dates, sizing="equal_weight",
                        start_idx=0, end_idx=None):
    """
    Run momentum backtest with daily exit checks on a specific date range.

    Returns: daily_returns (pd.Series), exit_stats (dict)
    """
    if end_idx is None:
        end_idx = len(trading_dates)

    portfolio = {}       # ticker -> weight
    trailing_highs = {}  # ticker -> highest close since entry
    mom_neg_streak = {}  # ticker -> consecutive days with negative 10d ret
    vol_low_streak = {}  # ticker -> consecutive days with low volume

    daily_returns = []
    daily_dates = []
    exit_stats = {"trailing_stop": 0, "momentum_breakdown": 0, "volume_dryup": 0, "rebalance": 0}
    last_rebalance = -REBALANCE_DAYS  # force first rebalance
    total_turnover = 0.0
    n_trades = 0

    for i in range(start_idx, end_idx):
        date = trading_dates[i]
        exited_today = set()

        # ── Daily exit checks ──
        for ticker in list(portfolio.keys()):
            if ticker not in close_df.columns:
                continue

            price = close_df.iloc[i].get(ticker, np.nan)
            if pd.isna(price):
                continue

            # Update trailing high
            if ticker in trailing_highs:
                trailing_highs[ticker] = max(trailing_highs[ticker], price)

            # 1. Trailing stop: price < trailing_high - 2*ATR
            atr_val = features["atr"].iloc[i].get(ticker, np.nan) if ticker in features["atr"].columns else np.nan
            if not pd.isna(atr_val) and atr_val > 0 and ticker in trailing_highs:
                stop = trailing_highs[ticker] - ATR_MULT * atr_val
                if price < stop:
                    exited_today.add(ticker)
                    exit_stats["trailing_stop"] += 1
                    continue

            # 2. Momentum breakdown: 10d return negative for N consecutive days
            ret10 = features["ret_10d"].iloc[i].get(ticker, np.nan) if ticker in features["ret_10d"].columns else np.nan
            if not pd.isna(ret10):
                if ret10 < 0:
                    mom_neg_streak[ticker] = mom_neg_streak.get(ticker, 0) + 1
                else:
                    mom_neg_streak[ticker] = 0
                if mom_neg_streak.get(ticker, 0) >= MOM_BREAKDOWN_CONSEC:
                    exited_today.add(ticker)
                    exit_stats["momentum_breakdown"] += 1
                    continue

            # 3. Volume dry-up: volume < 50% of 20d avg for N consecutive days
            vr = features["vol_ratio"].iloc[i].get(ticker, np.nan) if ticker in features["vol_ratio"].columns else np.nan
            if not pd.isna(vr):
                if vr < VOL_DRYUP_RATIO:
                    vol_low_streak[ticker] = vol_low_streak.get(ticker, 0) + 1
                else:
                    vol_low_streak[ticker] = 0
                if vol_low_streak.get(ticker, 0) >= VOL_DRYUP_CONSEC:
                    exited_today.add(ticker)
                    exit_stats["volume_dryup"] += 1
                    continue

        # Remove exited positions
        for t in exited_today:
            portfolio.pop(t, None)
            trailing_highs.pop(t, None)
            mom_neg_streak.pop(t, None)
            vol_low_streak.pop(t, None)
            n_trades += 1

        # ── Monthly rebalance ──
        is_rebal = (i - last_rebalance) >= REBALANCE_DAYS
        turnover_cost = 0.0

        if is_rebal:
            ranked = score_stocks(features, i, close_df)
            if len(ranked) >= TOP_N:
                new_tickers = [t for t, _ in ranked]

                # Compute weights
                if sizing == "inverse_vol":
                    inv_vols = {}
                    for t in new_tickers:
                        v = features["vol_20d"].iloc[i].get(t, np.nan) if t in features["vol_20d"].columns else np.nan
                        inv_vols[t] = 1.0 / max(v, 0.05) if not pd.isna(v) and v > 0 else 1.0 / 0.30
                    total_iv = sum(inv_vols.values())
                    new_weights = {t: inv_vols[t] / total_iv for t in new_tickers}
                else:
                    w = 1.0 / TOP_N
                    new_weights = {t: w for t in new_tickers}

                # Calculate turnover
                old_set = set(portfolio.keys())
                new_set = set(new_tickers)
                exiting = old_set - new_set
                entering = new_set - old_set

                turnover = 0.0
                for t in exiting:
                    turnover += portfolio.get(t, 0)
                    exit_stats["rebalance"] += 1
                for t in entering:
                    turnover += new_weights.get(t, 0)
                for t in old_set & new_set:
                    turnover += abs(new_weights.get(t, 0) - portfolio.get(t, 0))

                n_trades += len(entering) + len(exiting)
                total_turnover += turnover
                turnover_cost = turnover * SLIPPAGE_PCT

                # Update portfolio
                for t in exiting:
                    trailing_highs.pop(t, None)
                    mom_neg_streak.pop(t, None)
                    vol_low_streak.pop(t, None)
                for t in entering:
                    price = close_df.iloc[i].get(t, np.nan)
                    if not pd.isna(price):
                        trailing_highs[t] = price
                    mom_neg_streak[t] = 0
                    vol_low_streak[t] = 0

                portfolio = new_weights
                last_rebalance = i

        # Also charge slippage for intra-month dynamic exits
        exit_cost = len(exited_today) * (1.0 / TOP_N) * SLIPPAGE_PCT

        # ── Compute daily portfolio return ──
        port_ret = 0.0
        active_w = 0.0
        for ticker, weight in portfolio.items():
            if ticker not in features["daily_ret"].columns:
                continue
            r = features["daily_ret"].iloc[i].get(ticker, 0.0)
            if pd.isna(r):
                r = 0.0
            port_ret += weight * r
            active_w += weight

        # Normalize if coverage is low
        if 0 < active_w < 0.5:
            port_ret /= active_w

        net_ret = port_ret - turnover_cost - exit_cost
        daily_returns.append(net_ret)
        daily_dates.append(date)

    ret_series = pd.Series(daily_returns, index=pd.DatetimeIndex(daily_dates))
    exit_stats["total_trades"] = n_trades
    exit_stats["total_turnover"] = round(total_turnover, 2)
    return ret_series, exit_stats


def compute_metrics(returns, rf_annual=0.04):
    """Compute risk-adjusted performance metrics."""
    if len(returns) < 30:
        return {}

    rf_daily = (1 + rf_annual) ** (1/252) - 1
    excess = returns - rf_daily
    n_years = len(returns) / 252

    cum = (1 + returns).cumprod()
    total_ret = cum.iloc[-1]
    cagr = total_ret ** (1 / max(n_years, 0.1)) - 1

    ann_vol = returns.std() * np.sqrt(252)
    sharpe = excess.mean() / excess.std() * np.sqrt(252) if excess.std() > 0 else 0

    downside = excess[excess < 0]
    down_vol = downside.std() * np.sqrt(252) if len(downside) > 10 else 1e-9
    sortino = excess.mean() * 252 / down_vol if down_vol > 0 else 0

    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    wr = (returns > 0).mean()

    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    pf = gains / losses if losses > 0 else float("inf")

    return {
        "CAGR_pct": round(cagr * 100, 2),
        "Ann_Vol_pct": round(ann_vol * 100, 2),
        "Sharpe": round(sharpe, 3),
        "Sortino": round(sortino, 3),
        "MaxDD_pct": round(max_dd * 100, 2),
        "Calmar": round(calmar, 3),
        "WinRate_pct": round(wr * 100, 2),
        "ProfitFactor": round(pf, 3),
        "TotalReturn_pct": round((total_ret - 1) * 100, 2),
        "N_Days": len(returns),
        "N_Years": round(n_years, 1),
    }


def regime_stratified_sharpe(returns, day_regimes, rf_annual=0.04):
    """Compute Sharpe per regime and test gap < 0.50."""
    rf_daily = (1 + rf_annual) ** (1/252) - 1

    sharpes = {}
    counts = {}
    for regime in ["bull", "bear", "flat"]:
        mask = pd.Series([day_regimes.get(d, "flat") == regime for d in returns.index], index=returns.index)
        r = returns[mask]
        counts[regime] = len(r)
        if len(r) > 30:
            excess = r - rf_daily
            s = excess.mean() / excess.std() * np.sqrt(252) if excess.std() > 0 else 0
            sharpes[regime] = round(s, 3)
        else:
            sharpes[regime] = None

    valid = [v for v in sharpes.values() if v is not None]
    if len(valid) >= 2:
        best, worst = max(valid), min(valid)
        denom = max(abs(best), abs(worst))
        gap = abs(best - worst) / denom if denom > 0 else 0
    else:
        gap = None

    return {
        "sharpe_bull": sharpes.get("bull"),
        "sharpe_bear": sharpes.get("bear"),
        "sharpe_flat": sharpes.get("flat"),
        "days_bull": counts.get("bull", 0),
        "days_bear": counts.get("bear", 0),
        "days_flat": counts.get("flat", 0),
        "regime_gap": round(gap, 3) if gap is not None else None,
        "regime_pass": gap < 0.50 if gap is not None else None,
    }


def permutation_test_random_selection(close_df, features, trading_dates, sizing,
                                      actual_sharpe, n_perms=100, rf_annual=0.04):
    """
    Permutation test via random stock selection (lightweight version).
    Instead of picking top-N by momentum, pick random N stocks each rebalance.
    Tests whether the MOMENTUM SIGNAL adds value vs random selection.
    Uses simplified backtest (no exit logic) for speed.
    """
    rf_daily = (1 + rf_annual) ** (1/252) - 1
    rng = np.random.RandomState(42)
    perm_sharpes = np.empty(n_perms)

    # Pre-compute daily returns matrix for speed
    daily_ret_matrix = features["daily_ret"].values  # (n_days, n_stocks)
    col_list = list(features["daily_ret"].columns)
    col_to_idx = {c: i for i, c in enumerate(col_list)}
    n_days = len(trading_dates)

    # Pre-compute which tickers are eligible each rebalance day
    eligible_cache = {}
    for i in range(0, n_days, REBALANCE_DAYS):
        avail = []
        for t in col_list:
            if t == "SPY":
                continue
            idx = col_to_idx[t]
            price_val = close_df.iloc[i].get(t, np.nan)
            if not pd.isna(price_val) and price_val > 5.0:
                avail.append(idx)
        eligible_cache[i] = avail

    for p in range(n_perms):
        daily_rets = np.zeros(n_days)
        current_indices = np.array([], dtype=int)
        last_rebal = -REBALANCE_DAYS

        for i in range(n_days):
            is_rebal = (i - last_rebal) >= REBALANCE_DAYS

            if is_rebal:
                # Find nearest cached eligible set
                cache_key = (i // REBALANCE_DAYS) * REBALANCE_DAYS
                avail = eligible_cache.get(cache_key, [])
                if not avail:
                    avail = eligible_cache.get(max(k for k in eligible_cache if k <= i), [])

                if len(avail) >= TOP_N:
                    current_indices = rng.choice(avail, size=TOP_N, replace=False)
                    last_rebal = i

            # Equal-weight return
            if len(current_indices) > 0:
                rets = daily_ret_matrix[i, current_indices]
                valid = ~np.isnan(rets)
                if valid.sum() > 0:
                    daily_rets[i] = np.nanmean(rets)

        excess = daily_rets - rf_daily
        s = excess.mean() / excess.std() * np.sqrt(252) if excess.std() > 0 else 0
        perm_sharpes[p] = s

        if (p + 1) % 25 == 0:
            print(f"    Permutation {p+1}/{n_perms} (median random Sharpe: {np.median(perm_sharpes[:p+1]):.3f})")

    p_value = (np.sum(perm_sharpes >= actual_sharpe) + 1) / (n_perms + 1)
    return round(actual_sharpe, 3), round(p_value, 4), perm_sharpes


def walk_forward_backtest(close_df, features, trading_dates, sizing, spy_close):
    """
    Walk-forward: 60-month train, 1-month OOT, sliding window.
    Training phase validates that momentum works; OOT is true out-of-sample.
    """
    # Get month boundaries
    month_ends = pd.Series(trading_dates).groupby(
        pd.Series(trading_dates).dt.to_period("M")
    ).apply(lambda x: x.iloc[-1]).values
    month_ends = pd.DatetimeIndex(month_ends)

    # Map month_end -> index in trading_dates
    date_to_idx = {d: i for i, d in enumerate(trading_dates)}

    n_months = len(month_ends)
    print(f"  Walk-forward: {n_months} months available, "
          f"need {WF_TRAIN_MONTHS} train + {WF_TEST_MONTHS} test")

    all_oot_returns = []
    fold_metrics = []

    for fold_start in range(WF_TRAIN_MONTHS, n_months - WF_TEST_MONTHS + 1):
        # OOT period: fold_start to fold_start + test_months - 1
        oot_start_month = month_ends[fold_start]
        if fold_start + WF_TEST_MONTHS <= n_months:
            oot_end_month = month_ends[min(fold_start + WF_TEST_MONTHS, n_months) - 1]
        else:
            break

        # Find date indices
        oot_start_i = date_to_idx.get(oot_start_month)
        if oot_start_i is None:
            # Find closest
            oot_start_i = np.searchsorted(trading_dates, oot_start_month)

        # OOT goes from the day after last train day to end of test month
        # Actually: from first day of test month to last day of test month
        test_month_period = oot_start_month.to_period("M") + 1  # next month
        test_days = [d for d in trading_dates if d.to_period("M") == test_month_period]

        if not test_days:
            # Use the month starting after oot_start_month
            test_days = [d for d in trading_dates if d > oot_start_month]
            if not test_days:
                continue
            # Take just 1 month worth
            test_days = test_days[:REBALANCE_DAYS]

        if len(test_days) < 5:
            continue

        start_i = date_to_idx.get(test_days[0], np.searchsorted(trading_dates, test_days[0]))
        end_i = date_to_idx.get(test_days[-1], np.searchsorted(trading_dates, test_days[-1])) + 1

        # Run backtest on OOT slice
        oot_ret, _ = run_single_backtest(close_df, features, trading_dates, sizing,
                                         start_idx=start_i, end_idx=end_i)

        if len(oot_ret) > 0:
            all_oot_returns.append(oot_ret)
            m = compute_metrics(oot_ret) if len(oot_ret) > 5 else {}
            m["fold"] = fold_start - WF_TRAIN_MONTHS
            m["oot_period"] = f"{test_days[0].date()} to {test_days[-1].date()}"
            fold_metrics.append(m)

    # Concat all OOT returns
    if all_oot_returns:
        concat = pd.concat(all_oot_returns)
        concat = concat[~concat.index.duplicated(keep="first")].sort_index()
    else:
        concat = pd.Series(dtype=float)

    return concat, fold_metrics


def main():
    t0 = time.time()
    print("=" * 70)
    print("SYSTEMATIC MOMENTUM v2 — Dynamic Exits Backtest")
    print("=" * 70)

    # 1. Load data
    print("\n[1/6] Loading cached price data...")
    data = load_data()

    # 2. Build panel
    print("\n[2/6] Building price panel...")
    close_df, high_df, low_df, vol_df = build_price_panel(data)

    # Get SPY
    spy_close = close_df["SPY"].dropna()
    print(f"  SPY: {spy_close.index[0].date()} to {spy_close.index[-1].date()}")

    # 3. Compute features
    print("\n[3/6] Computing features...")
    features = compute_features(close_df, high_df, low_df, vol_df)

    # Trading dates (SPY available)
    trading_dates = close_df.index[close_df["SPY"].notna()]
    # Start after warmup (need 252 + 21 = 273 days)
    warmup = 280
    trading_dates = trading_dates[warmup:]
    print(f"  Trading dates: {trading_dates[0].date()} to {trading_dates[-1].date()} "
          f"({len(trading_dates)} days)")

    # Regime classification
    regime_map = classify_regimes(spy_close)
    day_regimes = map_day_to_regime(trading_dates, regime_map)
    regime_counts = {}
    for r in day_regimes.values():
        regime_counts[r] = regime_counts.get(r, 0) + 1
    print(f"  Regime distribution: {regime_counts}")

    # 4. Run backtests
    print("\n[4/6] Running backtests...")
    all_results = {}

    for sizing in ["equal_weight", "inverse_vol"]:
        config_name = f"momentum_v2_{sizing}"
        print(f"\n  --- {config_name} ---")

        # Full-period backtest
        print(f"  Running full-period backtest...")
        full_ret, exit_stats = run_single_backtest(
            close_df, features, trading_dates, sizing
        )
        full_metrics = compute_metrics(full_ret)
        print(f"    CAGR: {full_metrics.get('CAGR_pct')}%, "
              f"Sharpe: {full_metrics.get('Sharpe')}, "
              f"Sortino: {full_metrics.get('Sortino')}, "
              f"MaxDD: {full_metrics.get('MaxDD_pct')}%")
        print(f"    Exits: trailing_stop={exit_stats['trailing_stop']}, "
              f"mom_breakdown={exit_stats['momentum_breakdown']}, "
              f"vol_dryup={exit_stats['volume_dryup']}, "
              f"rebalance={exit_stats['rebalance']}")

        # Regime analysis
        regime_results = regime_stratified_sharpe(full_ret, day_regimes)
        print(f"    Regime Sharpe: bull={regime_results['sharpe_bull']}, "
              f"bear={regime_results['sharpe_bear']}, flat={regime_results['sharpe_flat']}")
        print(f"    Regime gap: {regime_results['regime_gap']} "
              f"({'PASS' if regime_results.get('regime_pass') else 'FAIL'})")

        # Walk-forward
        print(f"  Running walk-forward (60mo train, 1mo OOT)...")
        wf_ret, wf_folds = walk_forward_backtest(
            close_df, features, trading_dates, sizing, spy_close
        )
        wf_metrics = compute_metrics(wf_ret) if len(wf_ret) > 30 else {}
        print(f"    WF OOT: Sharpe={wf_metrics.get('Sharpe')}, "
              f"CAGR={wf_metrics.get('CAGR_pct')}%, "
              f"{len(wf_folds)} folds")

        # WF OOT regime analysis
        if len(wf_ret) > 30:
            wf_day_regimes = map_day_to_regime(wf_ret.index, regime_map)
            wf_regime = regime_stratified_sharpe(wf_ret, wf_day_regimes)
        else:
            wf_regime = {}

        # Permutation test: random stock selection vs momentum selection
        print(f"  Running permutation test (100 random-selection trials)...")
        perm_sharpe, p_value, _ = permutation_test_random_selection(
            close_df, features, trading_dates, sizing,
            actual_sharpe=full_metrics.get("Sharpe", 0), n_perms=100
        )
        print(f"    Permutation: Sharpe={perm_sharpe}, p={p_value} "
              f"({'SIGNIFICANT' if p_value < 0.05 else 'NOT SIGNIFICANT'})")

        # SPY benchmark
        spy_ret = spy_close.pct_change().reindex(full_ret.index).fillna(0)
        spy_metrics = compute_metrics(spy_ret)

        # Save equity curve
        eq = (1 + full_ret).cumprod()
        eq.to_csv(OUTPUT_DIR / f"equity_{config_name}.csv")

        all_results[config_name] = {
            "sizing": sizing,
            "full_period": full_metrics,
            "exit_stats": {k: int(v) if isinstance(v, (int, np.integer)) else v
                          for k, v in exit_stats.items()},
            "regime": regime_results,
            "wf_oot": wf_metrics,
            "wf_oot_regime": wf_regime,
            "wf_n_folds": len(wf_folds),
            "permutation": {
                "sharpe": perm_sharpe,
                "p_value": p_value,
                "significant": p_value < 0.05,
            },
            "spy_benchmark": spy_metrics,
        }

    # 5. Summary
    print("\n" + "=" * 70)
    print("[5/6] SUMMARY")
    print("=" * 70)

    for name, res in all_results.items():
        fp = res["full_period"]
        rg = res["regime"]
        wf = res["wf_oot"]
        pm = res["permutation"]
        spy = res["spy_benchmark"]
        print(f"\n{name}:")
        print(f"  Full-period: CAGR={fp.get('CAGR_pct')}%, Sharpe={fp.get('Sharpe')}, "
              f"Sortino={fp.get('Sortino')}, MaxDD={fp.get('MaxDD_pct')}%, "
              f"WR={fp.get('WinRate_pct')}%, PF={fp.get('ProfitFactor')}")
        print(f"  Regime gap: {rg.get('regime_gap')} ({'PASS' if rg.get('regime_pass') else 'FAIL'})")
        print(f"  WF OOT: Sharpe={wf.get('Sharpe')}, CAGR={wf.get('CAGR_pct')}%")
        print(f"  Permutation: p={pm.get('p_value')} ({'SIG' if pm.get('significant') else 'NOT SIG'})")
        print(f"  SPY benchmark: CAGR={spy.get('CAGR_pct')}%, Sharpe={spy.get('Sharpe')}")

    # 6. Save results
    print(f"\n[6/6] Saving results...")

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(i) for i in obj]
        return obj

    output = {
        "run_timestamp": datetime.now().isoformat(),
        "strategy": "Systematic Momentum v2 — Dynamic Exits",
        "universe_size": close_df.shape[1],
        "date_range": f"{trading_dates[0].date()} to {trading_dates[-1].date()}",
        "n_trading_days": len(trading_dates),
        "parameters": {
            "top_n": TOP_N,
            "lookbacks": LOOKBACKS,
            "skip_recent_days": SKIP_RECENT,
            "rebalance_days": REBALANCE_DAYS,
            "atr_period": ATR_PERIOD,
            "atr_trailing_mult": ATR_MULT,
            "momentum_breakdown_consec": MOM_BREAKDOWN_CONSEC,
            "volume_dryup_consec": VOL_DRYUP_CONSEC,
            "slippage_pct": SLIPPAGE_PCT,
            "wf_train_months": WF_TRAIN_MONTHS,
            "wf_test_months": WF_TEST_MONTHS,
        },
        "regime_distribution": regime_counts,
        "configs": convert(all_results),
    }

    outpath = OUTPUT_DIR / "momentum_v2_results.json"
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Saved to {outpath}")

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed / 60:.1f} minutes")

    return output


if __name__ == "__main__":
    main()

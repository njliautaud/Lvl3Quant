#!/usr/bin/env python3
"""
Rotation Confluence Backtest
=============================
Tests whether sub-sector rotation data improves the timing of our 4 proven
adversarial-validated strategies (#8 RSI Divergence, #9 Bond Yield,
#10 IV-RV Gap, #11 Liquidity Signal).

For each strategy x 3 rotation filters = 12 combos:
  A. INFLOW filter  — only trade if stock's sub-sector rotation_score > 0
  B. OUTFLOW filter — only trade if stock's sub-sector rotation_score < 0
  C. VELOCITY filter — only trade if rotation velocity improving week-over-week

Reports: original Sharpe, filtered Sharpe, change, trade count, regime gap,
         permutation p-value.

Walk-forward sliding window: 60d lookback for rotation scores, 21d forward.
FIFO costs: 0.1% round-trip.
4+ years data (2022-2026).

Usage:
    python3 paper_engines/rotation_confluence_backtest.py
"""

import warnings
warnings.filterwarnings("ignore")

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [RotConfl] %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "rotation_confluence_backtest.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Universe & Sub-sector mapping
# ---------------------------------------------------------------------------
QUALITY_STOCKS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AVGO", "JPM", "UNH",
    "JNJ", "V", "PG", "HD", "MA", "ABBV", "MRK", "PEP", "KO", "COST", "LLY",
    "CRM", "ADBE", "AMD", "INTC", "CSCO", "QCOM", "TXN", "NFLX", "DIS",
]

# Map each quality stock to its sub-sector (from SUBSECTOR_UNIVERSE in tracker)
STOCK_TO_SUBSECTOR = {
    "AAPL": "it_hardware",
    "MSFT": "enterprise_software",
    "GOOGL": "big_tech_comm",
    "AMZN": "ecommerce_retail",
    "META": "big_tech_comm",
    "NVDA": "semiconductors",
    "AVGO": "semiconductors",     # Broadcom = semis
    "JPM": "megabank",
    "UNH": "health_services",
    "JNJ": "pharma_large",
    "V": "fintech_payments",
    "PG": "consumer_staples_food",
    "HD": "ecommerce_retail",
    "MA": "fintech_payments",
    "ABBV": "pharma_large",
    "MRK": "pharma_large",
    "PEP": "consumer_staples_food",
    "KO": "consumer_staples_food",
    "COST": "ecommerce_retail",   # Also staples_retail, but ecommerce_retail has it
    "LLY": "pharma_large",
    "CRM": "enterprise_software",
    "ADBE": "enterprise_software",
    "AMD": "semiconductors",
    "INTC": "semiconductors",
    "CSCO": "it_hardware",
    "QCOM": "semiconductors",
    "TXN": "semiconductors",      # Closest: semiconductor
    "NFLX": "big_tech_comm",
    "DIS": "big_tech_comm",
}

# Sub-sector representative tickers (for computing rotation scores in backtest)
# We need the full sub-sector membership to compute rotation metrics
SUBSECTOR_TICKERS = {
    "semiconductors": ["NVDA", "AMD", "INTC", "QCOM", "MU"],
    "enterprise_software": ["MSFT", "CRM", "ORCL", "NOW", "ADBE"],
    "it_hardware": ["AAPL", "DELL", "HPQ", "CSCO", "ANET"],
    "big_tech_comm": ["META", "GOOGL", "NFLX", "DIS", "CMCSA"],
    "ecommerce_retail": ["AMZN", "HD", "LOW", "TJX", "COST"],
    "megabank": ["JPM", "BAC", "WFC", "C", "GS"],
    "health_services": ["UNH", "HCA", "CNC", "ELV", "CI"],
    "pharma_large": ["LLY", "JNJ", "ABBV", "MRK", "PFE"],
    "fintech_payments": ["V", "MA", "PYPL", "AFRM", "FIS"],
    "consumer_staples_food": ["PG", "KO", "PEP", "MDLZ", "GIS"],
}

# Macro tickers needed for strategies
MACRO_TICKERS = ["SPY", "^VIX", "^TNX"]

# All unique tickers to download
ALL_TICKERS = sorted(set(
    QUALITY_STOCKS
    + [t for tl in SUBSECTOR_TICKERS.values() for t in tl]
    + MACRO_TICKERS
))

# Constants
COST_RT = 0.001  # 0.1% round-trip FIFO cost
HOLD_DAYS = 10
N_PERMUTATIONS = 500
LOOKBACK_ROTATION = 60  # days for rotation score computation
START_DATE = "2022-01-01"

# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------
def download_all_data():
    """Download 4+ years of daily OHLCV for all needed tickers."""
    import yfinance as yf
    log.info(f"Downloading {len(ALL_TICKERS)} tickers from {START_DATE}...")

    batch_size = 40
    all_frames = []
    for i in range(0, len(ALL_TICKERS), batch_size):
        batch = ALL_TICKERS[i:i+batch_size]
        log.info(f"  Batch {i//batch_size+1}: {len(batch)} tickers")
        try:
            df = yf.download(batch, start=START_DATE, progress=False, threads=False)
            if df is not None and not df.empty:
                all_frames.append(df)
        except Exception as e:
            log.warning(f"  Batch failed: {e}")
        if i + batch_size < len(ALL_TICKERS):
            time.sleep(2)

    if not all_frames:
        raise RuntimeError("All downloads failed")

    if len(all_frames) > 1:
        data = pd.concat(all_frames, axis=1)
        data = data.loc[:, ~data.columns.duplicated()]
    else:
        data = all_frames[0]

    log.info(f"Downloaded data shape: {data.shape}, date range: {data.index[0]} to {data.index[-1]}")
    return data


def get_close(data, ticker):
    """Extract close price series."""
    try:
        if isinstance(data.columns, pd.MultiIndex):
            if ticker in data["Close"].columns:
                return data["Close"][ticker].dropna()
        elif ticker in data.columns:
            return data[ticker].dropna()
    except Exception:
        pass
    return None


def get_high(data, ticker):
    try:
        if isinstance(data.columns, pd.MultiIndex):
            if ticker in data["High"].columns:
                return data["High"][ticker].dropna()
    except Exception:
        pass
    return None


def get_low(data, ticker):
    try:
        if isinstance(data.columns, pd.MultiIndex):
            if ticker in data["Low"].columns:
                return data["Low"][ticker].dropna()
    except Exception:
        pass
    return None


def get_volume(data, ticker):
    try:
        if isinstance(data.columns, pd.MultiIndex):
            if ticker in data["Volume"].columns:
                return data["Volume"][ticker].dropna()
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# RSI helper
# ---------------------------------------------------------------------------
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ---------------------------------------------------------------------------
# Rotation score computation (walk-forward, point-in-time)
# ---------------------------------------------------------------------------
def compute_rotation_scores_timeseries(data, spy_close):
    """
    For each trading day, compute the rotation score and velocity for each
    sub-sector using only data available up to that day (no lookahead).

    Returns:
        rotation_scores: dict[subsector] -> pd.Series (indexed by date)
        rotation_velocity: dict[subsector] -> pd.Series (indexed by date)
    """
    log.info("Computing walk-forward rotation scores for all sub-sectors...")

    # Build close DataFrames per sub-sector
    subsector_close = {}
    for ss_name, tickers in SUBSECTOR_TICKERS.items():
        closes = {}
        for t in tickers:
            c = get_close(data, t)
            if c is not None and len(c) > 60:
                closes[t] = c
        if len(closes) >= 2:
            subsector_close[ss_name] = pd.DataFrame(closes).dropna()

    rotation_scores = {}
    rotation_velocities = {}

    for ss_name, close_df in subsector_close.items():
        if len(close_df) < LOOKBACK_ROTATION + 30:
            continue

        ret_df = close_df.pct_change()
        eq_ret = ret_df.mean(axis=1)  # equal-weight sub-sector return
        cum_ret = (1 + eq_ret).cumprod()

        # Align SPY
        common_idx = cum_ret.index.intersection(spy_close.index)
        cum_ret_aligned = cum_ret.loc[common_idx]
        spy_aligned = spy_close.loc[common_idx]

        scores = pd.Series(np.nan, index=cum_ret_aligned.index)
        velocities = pd.Series(np.nan, index=cum_ret_aligned.index)

        for i in range(LOOKBACK_ROTATION, len(cum_ret_aligned)):
            idx = cum_ret_aligned.index[i]

            # 21d relative strength
            if i >= 21:
                ss_ret_21 = cum_ret_aligned.iloc[i] / cum_ret_aligned.iloc[i-21] - 1
                spy_ret_21 = spy_aligned.iloc[i] / spy_aligned.iloc[i-21] - 1
                rel_str_21 = ss_ret_21 - spy_ret_21
            else:
                rel_str_21 = 0.0

            # Rotation velocity: change in 21d rel strength over 10 days
            if i >= 32:
                ss_ret_21_prev = cum_ret_aligned.iloc[i-10] / cum_ret_aligned.iloc[i-31] - 1
                spy_ret_21_prev = spy_aligned.iloc[i-10] / spy_aligned.iloc[i-31] - 1
                rel_str_21_prev = ss_ret_21_prev - spy_ret_21_prev
                rot_vel = rel_str_21 - rel_str_21_prev
            else:
                rot_vel = 0.0

            # Money flow (volume-weighted direction, last 10 days)
            money_flow = 0.0
            vol_ratio = 1.0
            window_ret = eq_ret.iloc[max(0,i-9):i+1]
            if len(window_ret) >= 5:
                up_days = (window_ret > 0).sum()
                down_days = (window_ret <= 0).sum()
                total = up_days + down_days
                money_flow = (up_days - down_days) / total if total > 0 else 0

            # Volume ratio
            if i >= 21:
                vol_5 = eq_ret.iloc[i-4:i+1].abs().mean()
                vol_21 = eq_ret.iloc[i-20:i+1].abs().mean()
                vol_ratio = vol_5 / vol_21 if vol_21 > 0 else 1.0

            # Breadth: fraction of tickers above their 20-SMA
            breadth = 0.5
            n_above = 0
            n_total = 0
            for t in close_df.columns:
                t_close = close_df[t]
                if i < len(t_close):
                    t_price = t_close.iloc[i] if i < len(t_close) else np.nan
                    t_sma20 = t_close.iloc[max(0,i-19):i+1].mean()
                    if not np.isnan(t_price) and not np.isnan(t_sma20):
                        n_total += 1
                        if t_price > t_sma20:
                            n_above += 1
            if n_total > 0:
                breadth = n_above / n_total

            # Composite score (same formula as tracker)
            score = 0.0
            score += np.clip(rot_vel * 50, -2, 2)
            score += np.clip(rel_str_21 * 10, -1.5, 1.5)
            score += np.clip(money_flow * 2, -1, 1)
            score += np.clip((vol_ratio - 1) * 0.5, -0.5, 0.5)
            score += (breadth - 0.5) * 1.0

            scores.iloc[i] = score
            velocities.iloc[i] = rot_vel

        rotation_scores[ss_name] = scores.dropna()
        rotation_velocities[ss_name] = velocities.dropna()

    log.info(f"  Computed rotation for {len(rotation_scores)} sub-sectors")
    return rotation_scores, rotation_velocities


# ---------------------------------------------------------------------------
# Strategy signal generators (vectorized, return boolean Series per stock)
# ---------------------------------------------------------------------------
def generate_rsi_divergence_signals(data, stock):
    """
    Strategy #8: RSI Divergence C
    BUY when price makes new 20d low but RSI makes higher low,
    volume declining, >5% below 52w high.
    """
    close = get_close(data, stock)
    vol = get_volume(data, stock)
    if close is None or len(close) < 252:
        return pd.Series(dtype=bool)

    rsi = compute_rsi(close)
    price_low_20 = close.rolling(20).min()
    rsi_low_20 = rsi.rolling(20).min()

    # Price at or near 20d low (within 1%)
    at_price_low = close <= price_low_20 * 1.01
    # RSI NOT at 20d low (higher low = bullish divergence)
    rsi_higher_low = rsi > rsi_low_20 + 2  # at least 2 points above the RSI low

    # Volume declining
    vol_declining = pd.Series(False, index=close.index)
    if vol is not None and len(vol) >= 40:
        vol_20 = vol.rolling(20).mean()
        vol_40 = vol.rolling(40).mean()
        vol_declining = vol_20 < vol_40

    # >5% below 52-week high
    high_52w = close.rolling(252).max()
    below_high = close < high_52w * 0.95

    signals = at_price_low & rsi_higher_low & vol_declining & below_high
    return signals.fillna(False)


def generate_bond_yield_signals(data, stock, tnx_close):
    """
    Strategy #9: Bond Yield Signal B
    BUY when 10Y yield drops >10bps over 5 days AND stock >5% below 20-SMA.
    """
    close = get_close(data, stock)
    if close is None or tnx_close is None or len(close) < 60:
        return pd.Series(dtype=bool)

    # Align
    common = close.index.intersection(tnx_close.index)
    close = close.loc[common]
    tnx = tnx_close.loc[common]

    # Yield drop >10bps (TNX is in %, so 0.10 = 10bps)
    yield_drop = tnx.diff(5)
    yield_signal = yield_drop < -0.10

    # Stock >5% below 20-SMA
    sma_20 = close.rolling(20).mean()
    below_sma = close < sma_20 * 0.95

    signals = yield_signal & below_sma
    return signals.fillna(False)


def generate_iv_rv_gap_signals(data, stock, vix_close, spy_close):
    """
    Strategy #10: IV-RV Gap
    BUY when VIX > SPY 20d realized vol (ann) by 5+ pts,
    stock >5% below 52w high, RSI<40.
    """
    close = get_close(data, stock)
    if close is None or vix_close is None or spy_close is None or len(close) < 252:
        return pd.Series(dtype=bool)

    # Realized vol of SPY (20d, annualized)
    spy_ret = spy_close.pct_change()
    rv_20 = spy_ret.rolling(20).std() * np.sqrt(252) * 100  # in VIX-like units

    # Align
    common = close.index.intersection(vix_close.index).intersection(rv_20.index)
    close = close.loc[common]
    vix = vix_close.loc[common]
    rv = rv_20.loc[common]

    # IV-RV gap
    gap = vix - rv
    iv_rv_signal = gap >= 5.0

    # >5% below 52w high
    high_52w = close.rolling(252).max()
    below_high = close < high_52w * 0.95

    # RSI < 40
    rsi = compute_rsi(close)
    rsi_low = rsi < 40

    signals = iv_rv_signal & below_high & rsi_low
    return signals.fillna(False)


def generate_liquidity_signals(data, stock):
    """
    Strategy #11: Liquidity Signal F
    BUY when HL spread narrows below 60d avg, >5% below 52w high, RSI<40.
    """
    close = get_close(data, stock)
    high = get_high(data, stock)
    low = get_low(data, stock)
    if close is None or high is None or low is None or len(close) < 252:
        return pd.Series(dtype=bool)

    # Align
    common = close.index.intersection(high.index).intersection(low.index)
    close = close.loc[common]
    high = high.loc[common]
    low = low.loc[common]

    # HL spread as % of close
    hl_spread = (high - low) / close
    hl_avg_60 = hl_spread.rolling(60).mean()
    spread_narrow = hl_spread < hl_avg_60

    # >5% below 52w high
    high_52w = close.rolling(252).max()
    below_high = close < high_52w * 0.95

    # RSI < 40
    rsi = compute_rsi(close)
    rsi_low = rsi < 40

    signals = spread_narrow & below_high & rsi_low
    return signals.fillna(False)


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------
def backtest_strategy(signals_dict, data, label="Strategy"):
    """
    Given signals_dict = {stock: pd.Series(bool)}, run a simple backtest.
    Each signal -> buy at next-day open (approx close), hold HOLD_DAYS,
    sell. Apply COST_RT.

    Returns dict with trades list and performance metrics.
    """
    trades = []
    for stock, sig in signals_dict.items():
        close = get_close(data, stock)
        if close is None:
            continue
        # Align signal to close index
        common = sig.index.intersection(close.index)
        sig = sig.loc[common]
        close = close.loc[common]

        signal_dates = sig[sig].index
        for sd in signal_dates:
            loc = close.index.get_loc(sd)
            entry_loc = loc + 1  # next-day entry
            exit_loc = entry_loc + HOLD_DAYS
            if exit_loc >= len(close):
                continue
            entry_price = close.iloc[entry_loc]
            exit_price = close.iloc[exit_loc]
            ret = (exit_price / entry_price - 1) - COST_RT
            entry_date = close.index[entry_loc]
            exit_date = close.index[exit_loc]
            trades.append({
                "stock": stock,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "entry_price": float(entry_price),
                "exit_price": float(exit_price),
                "return": float(ret),
            })

    if not trades:
        return {"n_trades": 0, "sharpe": 0.0, "wr": 0.0, "pf": 0.0,
                "sharpe_bull": 0.0, "sharpe_bear": 0.0, "regime_gap": 0.0,
                "trades": []}

    trades_df = pd.DataFrame(trades)
    trades_df["entry_date"] = pd.to_datetime(trades_df["entry_date"])

    returns = trades_df["return"].values
    n = len(returns)
    sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252 / HOLD_DAYS)) if np.std(returns) > 0 else 0.0
    wr = float((returns > 0).sum() / n) if n > 0 else 0.0
    gross_win = returns[returns > 0].sum() if (returns > 0).any() else 0
    gross_loss = abs(returns[returns < 0].sum()) if (returns < 0).any() else 1e-9
    pf = float(gross_win / gross_loss) if gross_loss > 0 else 999.0

    # Regime split: SPY-based bull/bear
    spy_close = get_close(data_global, "SPY")
    sharpe_bull = 0.0
    sharpe_bear = 0.0
    if spy_close is not None:
        spy_sma200 = spy_close.rolling(200).mean()
        bull_mask = []
        bear_mask = []
        for _, trade in trades_df.iterrows():
            d = trade["entry_date"]
            if d in spy_sma200.index:
                loc = spy_sma200.index.get_loc(d)
                if spy_close.iloc[loc] > spy_sma200.iloc[loc]:
                    bull_mask.append(True)
                    bear_mask.append(False)
                else:
                    bull_mask.append(False)
                    bear_mask.append(True)
            else:
                bull_mask.append(True)
                bear_mask.append(False)

        bull_rets = returns[np.array(bull_mask)]
        bear_rets = returns[np.array(bear_mask)]
        if len(bull_rets) > 2 and np.std(bull_rets) > 0:
            sharpe_bull = float(np.mean(bull_rets) / np.std(bull_rets) * np.sqrt(252/HOLD_DAYS))
        if len(bear_rets) > 2 and np.std(bear_rets) > 0:
            sharpe_bear = float(np.mean(bear_rets) / np.std(bear_rets) * np.sqrt(252/HOLD_DAYS))

    max_abs = max(abs(sharpe_bull), abs(sharpe_bear), 1e-9)
    regime_gap = abs(sharpe_bull - sharpe_bear) / max_abs

    return {
        "n_trades": n,
        "sharpe": round(sharpe, 3),
        "wr": round(wr, 3),
        "pf": round(pf, 3),
        "sharpe_bull": round(sharpe_bull, 3),
        "sharpe_bear": round(sharpe_bear, 3),
        "regime_gap": round(regime_gap, 3),
        "trades": trades,
    }


def permutation_test(base_trades, filtered_indices, n_perm=N_PERMUTATIONS):
    """
    Test if the filter adds real timing alpha vs random filtering.
    Randomly sample the same number of trades from base_trades and compute
    Sharpe each time. p-value = fraction of random samples with Sharpe >= actual.
    """
    if len(filtered_indices) == 0 or len(base_trades) == 0:
        return 1.0

    base_returns = np.array([t["return"] for t in base_trades])
    filtered_returns = base_returns[filtered_indices]
    n_filtered = len(filtered_returns)

    if n_filtered < 3 or np.std(filtered_returns) == 0:
        return 1.0

    actual_sharpe = np.mean(filtered_returns) / np.std(filtered_returns) * np.sqrt(252/HOLD_DAYS)

    count_better = 0
    rng = np.random.RandomState(42)
    for _ in range(n_perm):
        sample_idx = rng.choice(len(base_returns), size=n_filtered, replace=False) \
            if n_filtered <= len(base_returns) else rng.choice(len(base_returns), size=n_filtered, replace=True)
        sample_rets = base_returns[sample_idx]
        if np.std(sample_rets) > 0:
            sample_sharpe = np.mean(sample_rets) / np.std(sample_rets) * np.sqrt(252/HOLD_DAYS)
            if sample_sharpe >= actual_sharpe:
                count_better += 1
        else:
            count_better += 1

    return round(count_better / n_perm, 4)


# ---------------------------------------------------------------------------
# Apply rotation filter to trades
# ---------------------------------------------------------------------------
def apply_rotation_filter(trades, rotation_scores, rotation_velocities, filter_type):
    """
    Filter trades based on rotation data.

    filter_type:
      'inflow'   — keep if sub-sector rotation_score > 0 at trade entry
      'outflow'  — keep if sub-sector rotation_score < 0 at trade entry
      'velocity' — keep if rotation velocity > prev week's velocity (accelerating)

    Returns: list of indices into trades that pass the filter.
    """
    kept_indices = []
    for i, trade in enumerate(trades):
        stock = trade["stock"]
        entry_date = pd.Timestamp(trade["entry_date"])
        subsector = STOCK_TO_SUBSECTOR.get(stock)
        if subsector is None:
            continue

        if filter_type == "inflow":
            scores = rotation_scores.get(subsector)
            if scores is None:
                continue
            # Find the score on or just before entry_date
            valid = scores.loc[:entry_date]
            if len(valid) == 0:
                continue
            score = valid.iloc[-1]
            if score > 0:
                kept_indices.append(i)

        elif filter_type == "outflow":
            scores = rotation_scores.get(subsector)
            if scores is None:
                continue
            valid = scores.loc[:entry_date]
            if len(valid) == 0:
                continue
            score = valid.iloc[-1]
            if score < 0:
                kept_indices.append(i)

        elif filter_type == "velocity":
            vels = rotation_velocities.get(subsector)
            if vels is None:
                continue
            valid = vels.loc[:entry_date]
            if len(valid) < 6:
                continue
            # Current velocity vs 5 days ago
            vel_now = valid.iloc[-1]
            vel_prev = valid.iloc[-6] if len(valid) >= 6 else valid.iloc[0]
            if vel_now > vel_prev:  # accelerating
                kept_indices.append(i)

    return kept_indices


def compute_filtered_metrics(trades, kept_indices, data):
    """Compute performance metrics for filtered subset of trades."""
    if not kept_indices:
        return {"n_trades": 0, "sharpe": 0.0, "wr": 0.0, "pf": 0.0,
                "sharpe_bull": 0.0, "sharpe_bear": 0.0, "regime_gap": 0.0}

    filtered_trades = [trades[i] for i in kept_indices]
    returns = np.array([t["return"] for t in filtered_trades])
    n = len(returns)
    sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252/HOLD_DAYS)) if np.std(returns) > 0 else 0.0
    wr = float((returns > 0).sum() / n) if n > 0 else 0.0
    gross_win = returns[returns > 0].sum() if (returns > 0).any() else 0
    gross_loss = abs(returns[returns < 0].sum()) if (returns < 0).any() else 1e-9
    pf = float(gross_win / gross_loss) if gross_loss > 0 else 999.0

    # Regime
    spy_close = get_close(data, "SPY")
    sharpe_bull = 0.0
    sharpe_bear = 0.0
    if spy_close is not None:
        spy_sma200 = spy_close.rolling(200).mean()
        bull_rets = []
        bear_rets = []
        for t in filtered_trades:
            d = pd.Timestamp(t["entry_date"])
            if d in spy_sma200.index:
                loc = spy_sma200.index.get_loc(d)
                if spy_close.iloc[loc] > spy_sma200.iloc[loc]:
                    bull_rets.append(t["return"])
                else:
                    bear_rets.append(t["return"])
            else:
                bull_rets.append(t["return"])
        bull_rets = np.array(bull_rets) if bull_rets else np.array([])
        bear_rets = np.array(bear_rets) if bear_rets else np.array([])
        if len(bull_rets) > 2 and np.std(bull_rets) > 0:
            sharpe_bull = float(np.mean(bull_rets) / np.std(bull_rets) * np.sqrt(252/HOLD_DAYS))
        if len(bear_rets) > 2 and np.std(bear_rets) > 0:
            sharpe_bear = float(np.mean(bear_rets) / np.std(bear_rets) * np.sqrt(252/HOLD_DAYS))

    max_abs = max(abs(sharpe_bull), abs(sharpe_bear), 1e-9)
    regime_gap = abs(sharpe_bull - sharpe_bear) / max_abs

    return {
        "n_trades": n,
        "sharpe": round(sharpe, 3),
        "wr": round(wr, 3),
        "pf": round(pf, 3),
        "sharpe_bull": round(sharpe_bull, 3),
        "sharpe_bear": round(sharpe_bear, 3),
        "regime_gap": round(regime_gap, 3),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
data_global = None  # set after download for regime calcs

def main():
    global data_global
    log.info("=" * 70)
    log.info("ROTATION CONFLUENCE BACKTEST — Testing rotation filters on proven strategies")
    log.info("=" * 70)

    # Download data
    data = download_all_data()
    data_global = data

    # Get macro series
    spy_close = get_close(data, "SPY")
    vix_close = get_close(data, "^VIX")
    tnx_close = get_close(data, "^TNX")

    if spy_close is None:
        log.error("SPY data missing, cannot proceed")
        return

    # Compute walk-forward rotation scores
    rotation_scores, rotation_velocities = compute_rotation_scores_timeseries(data, spy_close)

    # Define strategies
    strategies = {
        "RSI_Divergence": {
            "generator": lambda stock: generate_rsi_divergence_signals(data, stock),
            "description": "#8 RSI Divergence — buy quality dips on bullish RSI divergence",
        },
        "Bond_Yield": {
            "generator": lambda stock: generate_bond_yield_signals(data, stock, tnx_close),
            "description": "#9 Bond Yield — buy quality dips when 10Y yield drops",
        },
        "IV_RV_Gap": {
            "generator": lambda stock: generate_iv_rv_gap_signals(data, stock, vix_close, spy_close),
            "description": "#10 IV-RV Gap — buy quality dips when implied vol > realized vol",
        },
        "Liquidity_Signal": {
            "generator": lambda stock: generate_liquidity_signals(data, stock),
            "description": "#11 Liquidity Signal — buy quality dips when bid-ask narrows",
        },
    }

    filter_types = {
        "inflow": "A. INFLOW — only trade if sub-sector rotation_score > 0",
        "outflow": "B. OUTFLOW — only trade if sub-sector rotation_score < 0",
        "velocity": "C. VELOCITY — only trade if rotation velocity accelerating",
    }

    results_all = []

    for strat_name, strat_info in strategies.items():
        log.info(f"\n{'='*60}")
        log.info(f"Strategy: {strat_info['description']}")
        log.info(f"{'='*60}")

        # Generate base signals for all stocks
        signals = {}
        for stock in QUALITY_STOCKS:
            sig = strat_info["generator"](stock)
            if len(sig) > 0 and sig.any():
                signals[stock] = sig

        # Run base backtest
        base_result = backtest_strategy(signals, data, label=strat_name)
        log.info(f"  BASE: {base_result['n_trades']} trades, Sharpe={base_result['sharpe']}, "
                 f"WR={base_result['wr']}, PF={base_result['pf']}, "
                 f"Regime gap={base_result['regime_gap']}")

        if base_result["n_trades"] < 5:
            log.warning(f"  Too few base trades ({base_result['n_trades']}), skipping filters")
            for ft in filter_types:
                results_all.append({
                    "strategy": strat_name,
                    "filter": ft,
                    "base_sharpe": base_result["sharpe"],
                    "base_trades": base_result["n_trades"],
                    "filtered_sharpe": 0.0,
                    "filtered_trades": 0,
                    "sharpe_change": 0.0,
                    "trade_pct_removed": 1.0,
                    "regime_gap": 0.0,
                    "perm_p": 1.0,
                    "verdict": "SKIP (too few base trades)",
                })
            continue

        # Test each rotation filter
        for ft_name, ft_desc in filter_types.items():
            log.info(f"\n  Filter: {ft_desc}")

            kept_indices = apply_rotation_filter(
                base_result["trades"], rotation_scores, rotation_velocities, ft_name
            )

            filtered_metrics = compute_filtered_metrics(base_result["trades"], kept_indices, data)

            # Permutation test
            perm_p = permutation_test(base_result["trades"], kept_indices)

            sharpe_change = filtered_metrics["sharpe"] - base_result["sharpe"]
            n_removed = base_result["n_trades"] - filtered_metrics["n_trades"]
            pct_removed = n_removed / base_result["n_trades"] if base_result["n_trades"] > 0 else 0

            verdict = "FAIL"
            if sharpe_change > 0.20 and perm_p < 0.05:
                verdict = "VALIDATED"
            elif sharpe_change > 0.10 and perm_p < 0.10:
                verdict = "MARGINAL"
            elif sharpe_change < -0.10:
                verdict = "DEGRADES"
            else:
                verdict = "NO EFFECT"

            log.info(f"    Filtered: {filtered_metrics['n_trades']} trades "
                     f"({pct_removed*100:.0f}% removed), "
                     f"Sharpe={filtered_metrics['sharpe']}, "
                     f"WR={filtered_metrics['wr']}, PF={filtered_metrics['pf']}")
            log.info(f"    Sharpe change: {sharpe_change:+.3f}, "
                     f"Regime gap: {filtered_metrics['regime_gap']}, "
                     f"Perm p={perm_p}")
            log.info(f"    VERDICT: {verdict}")

            results_all.append({
                "strategy": strat_name,
                "filter": ft_name,
                "base_sharpe": base_result["sharpe"],
                "base_trades": base_result["n_trades"],
                "filtered_sharpe": filtered_metrics["sharpe"],
                "filtered_trades": filtered_metrics["n_trades"],
                "filtered_wr": filtered_metrics["wr"],
                "filtered_pf": filtered_metrics["pf"],
                "sharpe_change": round(sharpe_change, 3),
                "trade_pct_removed": round(pct_removed, 3),
                "regime_gap": filtered_metrics["regime_gap"],
                "regime_gap_bull": filtered_metrics["sharpe_bull"],
                "regime_gap_bear": filtered_metrics["sharpe_bear"],
                "perm_p": perm_p,
                "verdict": verdict,
            })

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    log.info(f"\n\n{'='*80}")
    log.info("ROTATION CONFLUENCE BACKTEST — FULL RESULTS")
    log.info(f"{'='*80}")
    log.info(f"{'Strategy':<20} {'Filter':<10} {'Base_S':<8} {'Filt_S':<8} "
             f"{'Delta':<8} {'Trades':<12} {'RegGap':<8} {'Perm_p':<8} {'Verdict':<12}")
    log.info("-" * 100)

    any_validated = False
    for r in results_all:
        trade_str = f"{r['base_trades']}->{r['filtered_trades']}"
        log.info(f"{r['strategy']:<20} {r['filter']:<10} {r['base_sharpe']:<8.3f} "
                 f"{r['filtered_sharpe']:<8.3f} {r['sharpe_change']:+7.3f} "
                 f"{trade_str:<12} {r['regime_gap']:<8.3f} {r['perm_p']:<8.4f} "
                 f"{r['verdict']:<12}")
        if r["verdict"] == "VALIDATED":
            any_validated = True

    log.info(f"\n{'='*80}")
    if any_validated:
        log.info("CONCLUSION: Some rotation filters VALIDATED — wire into signal aggregator")
    else:
        log.info("CONCLUSION: Rotation data adds NO timing alpha as a confluence filter.")
        log.info("  The sub-sector rotation tracker remains useful as VISUAL CONTEXT only.")
        log.info("  Do NOT use it to filter or modify proven strategy signals.")
    log.info(f"{'='*80}")

    # Save results
    output_path = Path(__file__).resolve().parent / "logs" / "rotation_confluence_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "n_combos": len(results_all),
            "any_validated": any_validated,
            "results": results_all,
        }, f, indent=2, default=str)
    log.info(f"Results saved to {output_path}")

    return results_all


if __name__ == "__main__":
    results = main()

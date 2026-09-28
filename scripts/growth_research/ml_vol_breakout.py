#!/usr/bin/env python3
"""
ML Volatility Breakout Strategy — Options Straddle Research
=============================================================
Predicts which mega-cap stocks will have abnormally large moves (>8% either
direction) in the next 21 trading days. When confident of a big move, simulates
buying an ATM straddle. Asymmetric upside: limited loss (premium) + unlimited
gain on big moves.

Walk-forward SLIDING window: 252d train, 21d advance, LightGBM classifier.
No expanding window (HC #0). Regime-agnostic validation (HC #428).

UNIVERSE: 30 liquid mega-cap stocks.
TARGET: Binary — |move| > 8% in next 21 trading days.

Usage:
    python ml_vol_breakout.py
"""

import json
import logging
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path("/home/jupiter/Lvl3Quant")
OUTPUT_DIR = BASE_DIR / "output" / "ml_vol_breakout"
LOG_DIR = BASE_DIR / "logs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "ml_vol_breakout.log", mode="w"),
    ],
)
log = logging.getLogger("ml_vol_breakout")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "V", "MA",
    "JNJ", "UNH", "HD", "PG", "BAC", "XOM", "CVX", "COST", "CRM", "NFLX",
    "AMD", "ORCL", "ADBE", "LLY", "MRK", "PEP", "KO", "WMT", "DIS", "GS",
]

# Cross-asset tickers for features
CROSS_ASSET = ["SPY", "GLD", "TLT", "HYG"]
VIX_TICKER = "^VIX"

TRAIN_DAYS = 252          # 1 year sliding window
ADVANCE_DAYS = 21         # walk-forward step
TARGET_HORIZON = 21       # predict 21-day max move
MOVE_THRESHOLD = 0.08     # 8% move threshold
LOOKBACK_MAX = 63         # max feature lookback

# Backtest
INITIAL_CAPITAL = 100_000.0  # HC #713: fixed $100K
RISK_PER_TRADE = 0.02        # 2% risk per trade
MAX_CONCURRENT = 5           # max concurrent positions
BID_ASK_HAIRCUT = 0.15       # 15% haircut on premium
IV_MARKUP = 1.15             # IV = realized vol * 1.15

# Permutation test
N_PERMUTATIONS = 30

# Sector mapping for dummy features
SECTOR_MAP = {
    "AAPL": "Tech", "MSFT": "Tech", "GOOGL": "Tech", "AMZN": "ConsDisc",
    "NVDA": "Tech", "META": "Tech", "TSLA": "ConsDisc", "JPM": "Financials",
    "V": "Financials", "MA": "Financials", "JNJ": "Healthcare", "UNH": "Healthcare",
    "HD": "ConsDisc", "PG": "ConsStaples", "BAC": "Financials", "XOM": "Energy",
    "CVX": "Energy", "COST": "ConsStaples", "CRM": "Tech", "NFLX": "Tech",
    "AMD": "Tech", "ORCL": "Tech", "ADBE": "Tech", "LLY": "Healthcare",
    "MRK": "Healthcare", "PEP": "ConsStaples", "KO": "ConsStaples",
    "WMT": "ConsStaples", "DIS": "Tech", "GS": "Financials",
}
SECTORS = sorted(set(SECTOR_MAP.values()))


# ============================================================================
# DATA DOWNLOAD
# ============================================================================

def download_data(start="2010-01-01") -> dict:
    """Download OHLCV for universe + cross-asset tickers."""
    import yfinance as yf

    all_tickers = UNIVERSE + CROSS_ASSET + [VIX_TICKER]
    log.info(f"Downloading {len(all_tickers)} tickers from {start}...")

    data = {}
    failed = []
    for tkr in all_tickers:
        try:
            df = yf.download(tkr, start=start, auto_adjust=True, progress=False)
            if df is not None and len(df) > 100:
                # Flatten multi-level columns if needed
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                data[tkr] = df
                log.info(f"  {tkr}: {len(df)} rows ({df.index[0].date()} to {df.index[-1].date()})")
            else:
                failed.append(tkr)
                log.warning(f"  {tkr}: insufficient data ({len(df) if df is not None else 0} rows)")
        except Exception as e:
            failed.append(tkr)
            log.warning(f"  {tkr}: download failed — {e}")

    if failed:
        log.warning(f"Failed tickers: {failed}")

    return data


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def compute_features(stock_df: pd.DataFrame, ticker: str,
                     spy_df: pd.DataFrame, vix_df: pd.DataFrame,
                     gld_df: pd.DataFrame, tlt_df: pd.DataFrame,
                     hyg_df: pd.DataFrame) -> pd.DataFrame:
    """Compute all features for a single stock."""
    df = stock_df[["Close", "Volume"]].copy()
    df.columns = ["close", "volume"]
    df = df.dropna()

    if len(df) < LOOKBACK_MAX + 10:
        return pd.DataFrame()

    ret = df["close"].pct_change()
    log_ret = np.log(df["close"] / df["close"].shift(1))

    feats = pd.DataFrame(index=df.index)

    # --- Historical realized vol ---
    for w in [5, 10, 21, 63]:
        feats[f"rvol_{w}d"] = log_ret.rolling(w).std() * np.sqrt(252)

    # --- Vol-of-vol ---
    feats["vol_of_vol_21d"] = feats["rvol_21d"].rolling(21).std()
    feats["vol_of_vol_63d"] = feats["rvol_63d"].rolling(21).std()

    # --- Recent max move ---
    feats["max_abs_ret_10d"] = ret.abs().rolling(10).max()
    feats["max_abs_ret_21d"] = ret.abs().rolling(21).max()

    # --- Bollinger bandwidth (vol compression → breakout) ---
    sma20 = df["close"].rolling(20).mean()
    std20 = df["close"].rolling(20).std()
    feats["bband_width"] = (2 * std20) / sma20  # normalized bandwidth
    feats["bband_pctb"] = (df["close"] - (sma20 - 2 * std20)) / (4 * std20)  # %B

    # --- Volume spike ---
    vol_ma20 = df["volume"].rolling(20).mean()
    feats["volume_spike"] = df["volume"] / vol_ma20.replace(0, np.nan)
    feats["volume_spike_max5d"] = feats["volume_spike"].rolling(5).max()

    # --- RSI ---
    delta = ret.copy()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    feats["rsi_14"] = rsi
    feats["rsi_extreme"] = ((rsi > 70) | (rsi < 30)).astype(float)

    # --- Days since last big move ---
    big_move_mask = ret.abs() > 0.05
    feats["days_since_5pct_move"] = big_move_mask.groupby(
        big_move_mask.cumsum()
    ).cumcount()
    feats["days_since_5pct_move"] = feats["days_since_5pct_move"].clip(upper=252)

    # --- IV rank proxy (vol percentile over 252d) ---
    feats["vol_percentile_252d"] = feats["rvol_21d"].rolling(252).rank(pct=True)

    # --- Vol ratio (short/long) — compression indicator ---
    feats["vol_ratio_5_63"] = feats["rvol_5d"] / feats["rvol_63d"].replace(0, np.nan)
    feats["vol_ratio_10_63"] = feats["rvol_10d"] / feats["rvol_63d"].replace(0, np.nan)

    # --- Momentum features ---
    feats["ret_5d"] = ret.rolling(5).sum()
    feats["ret_21d"] = ret.rolling(21).sum()
    feats["abs_ret_5d"] = ret.abs().rolling(5).sum()

    # --- Cross-asset features ---
    def _safe_align(ext_df, col="Close"):
        if ext_df is None or ext_df.empty:
            return pd.Series(np.nan, index=df.index)
        s = ext_df[col].reindex(df.index).ffill()
        return s

    # SPY vol
    spy_close = _safe_align(spy_df)
    spy_ret = spy_close.pct_change()
    feats["spy_rvol_21d"] = np.log(spy_close / spy_close.shift(1)).rolling(21).std() * np.sqrt(252)
    feats["spy_ret_21d"] = spy_ret.rolling(21).sum()

    # VIX
    vix_close = _safe_align(vix_df)
    feats["vix_level"] = vix_close
    feats["vix_pctile_252d"] = vix_close.rolling(252).rank(pct=True)
    feats["vix_change_5d"] = vix_close.pct_change(5)

    # GLD vol
    gld_close = _safe_align(gld_df)
    feats["gld_rvol_21d"] = np.log(gld_close / gld_close.shift(1)).rolling(21).std() * np.sqrt(252)

    # TLT vol
    tlt_close = _safe_align(tlt_df)
    feats["tlt_rvol_21d"] = np.log(tlt_close / tlt_close.shift(1)).rolling(21).std() * np.sqrt(252)

    # Credit spread proxy: HYG - TLT return spread
    hyg_close = _safe_align(hyg_df)
    feats["credit_spread_21d"] = hyg_close.pct_change(21) - tlt_close.pct_change(21)

    # --- Sector dummies ---
    sector = SECTOR_MAP.get(ticker, "Unknown")
    for s in SECTORS:
        feats[f"sector_{s}"] = 1.0 if s == sector else 0.0

    # --- Beta to SPY ---
    cov_60 = ret.rolling(60).cov(spy_ret)
    var_60 = spy_ret.rolling(60).var()
    feats["beta_60d"] = cov_60 / var_60.replace(0, np.nan)

    return feats


def compute_target(stock_df: pd.DataFrame) -> pd.Series:
    """Binary target: will stock move >8% (either direction) in next 21 trading days?"""
    close = stock_df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    # For each day, look at the max absolute return over next 21 days
    # Use rolling on reversed series for forward-looking
    fwd_returns = pd.DataFrame(index=close.index)
    for d in range(1, TARGET_HORIZON + 1):
        fwd_returns[f"fwd_{d}"] = close.shift(-d) / close - 1

    max_abs_fwd = fwd_returns.abs().max(axis=1)
    target = (max_abs_fwd >= MOVE_THRESHOLD).astype(int)
    return target


# ============================================================================
# BLACK-SCHOLES STRADDLE PRICING
# ============================================================================

def bs_call(S, K, T, sigma, r=0.04):
    """Black-Scholes call price."""
    from scipy.stats import norm
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put(S, K, T, sigma, r=0.04):
    """Black-Scholes put price."""
    from scipy.stats import norm
    if T <= 0 or sigma <= 0:
        return max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def straddle_price(S, sigma_annual, T_days=21, r=0.04):
    """ATM straddle price (call + put) using BS with IV markup."""
    T = T_days / 252.0
    iv = sigma_annual * IV_MARKUP  # IV typically > realized vol
    K = S  # ATM
    call = bs_call(S, K, T, iv, r)
    put = bs_put(S, K, T, iv, r)
    premium = call + put
    # Apply bid-ask haircut
    premium *= (1 + BID_ASK_HAIRCUT)
    return premium, iv


def straddle_payoff(S_entry, S_exit_max_abs, premium_per_share, n_shares=100):
    """
    Straddle P&L.
    S_exit_max_abs: the maximum absolute price deviation from entry within horizon.
    """
    # Payoff = |move| * 100 - premium * 100
    intrinsic = S_exit_max_abs * n_shares
    cost = premium_per_share * n_shares
    return intrinsic - cost


# ============================================================================
# WALK-FORWARD ENGINE
# ============================================================================

def build_panel(data: dict) -> tuple:
    """Build feature panel across all stocks and dates."""
    log.info("Building feature panel...")

    spy_df = data.get("SPY")
    vix_df = data.get("^VIX")
    gld_df = data.get("GLD")
    tlt_df = data.get("TLT")
    hyg_df = data.get("HYG")

    all_rows = []
    for ticker in UNIVERSE:
        if ticker not in data:
            log.warning(f"Skipping {ticker} — no data")
            continue

        stock_df = data[ticker]
        feats = compute_features(stock_df, ticker, spy_df, vix_df, gld_df, tlt_df, hyg_df)
        target = compute_target(stock_df)

        if feats.empty:
            continue

        # Align
        common_idx = feats.index.intersection(target.index)
        feats = feats.loc[common_idx]
        target = target.loc[common_idx]

        feats["target"] = target
        feats["ticker"] = ticker
        feats["close"] = stock_df["Close"].reindex(common_idx)
        if isinstance(feats["close"], pd.DataFrame):
            feats["close"] = feats["close"].iloc[:, 0]

        all_rows.append(feats)

    panel = pd.concat(all_rows, axis=0).sort_index()
    log.info(f"Panel shape: {panel.shape}, date range: {panel.index[0].date()} to {panel.index[-1].date()}")

    # Target distribution
    target_rate = panel["target"].mean()
    log.info(f"Target rate (>8% move in 21d): {target_rate:.3f} ({target_rate*100:.1f}%)")

    return panel


def run_walk_forward(panel: pd.DataFrame) -> pd.DataFrame:
    """Sliding walk-forward with LightGBM classifier. Optimized with integer indexing."""
    import lightgbm as lgb

    feature_cols = [c for c in panel.columns if c not in ["target", "ticker", "close"]]
    dates = panel.index.unique().sort_values()

    log.info(f"Walk-forward: {len(dates)} unique dates, {len(feature_cols)} features")
    log.info(f"Train window: {TRAIN_DAYS}d, Advance: {ADVANCE_DAYS}d")

    # Pre-compute: map each date to integer index for fast slicing
    date_to_idx = {d: i for i, d in enumerate(dates)}
    panel = panel.copy()
    panel["_date_idx"] = panel.index.map(date_to_idx)
    panel = panel.sort_values("_date_idx")

    # Pre-drop rows with any NaN in features (do once, not per fold)
    feat_valid = panel[feature_cols].notna().all(axis=1) & panel["target"].notna()
    panel_clean = panel[feat_valid].copy()
    log.info(f"After NaN removal: {len(panel_clean)} / {len(panel)} rows")

    # Extract numpy arrays for speed
    X_all = panel_clean[feature_cols].values
    y_all = panel_clean["target"].values
    idx_all = panel_clean["_date_idx"].values

    all_predictions = []
    fold = 0

    min_start = TRAIN_DAYS + LOOKBACK_MAX
    if min_start >= len(dates):
        log.error("Not enough data for walk-forward")
        return pd.DataFrame()

    total_folds = (len(dates) - min_start) // ADVANCE_DAYS
    log.info(f"Expected folds: ~{total_folds}")

    test_start = min_start
    while test_start + ADVANCE_DAYS <= len(dates):
        fold += 1
        test_end = min(test_start + ADVANCE_DAYS, len(dates))

        train_lo = test_start - TRAIN_DAYS
        # HC #718: label gap = TARGET_HORIZON to prevent look-ahead
        train_hi = test_start - TARGET_HORIZON

        # Fast integer-based masking on numpy arrays
        train_mask = (idx_all >= train_lo) & (idx_all < train_hi)
        test_mask = (idx_all >= test_start) & (idx_all < test_end)

        X_train = X_all[train_mask]
        y_train = y_all[train_mask]
        X_test = X_all[test_mask]

        if len(X_train) < 100 or len(X_test) < 1:
            test_start += ADVANCE_DAYS
            continue

        # LightGBM
        pos_rate = y_train.mean()
        pos_weight = max(1.0, (1 - pos_rate) / max(pos_rate, 0.01))

        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.1,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "scale_pos_weight": pos_weight,
            "min_child_samples": 20,
            "verbose": -1,
            "n_jobs": 4,
            "seed": 42,
        }

        train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=True)
        model = lgb.train(params, train_data, num_boost_round=150)

        preds = model.predict(X_test)

        # Store prediction indices and values (avoid DataFrame ops in loop)
        test_indices = np.where(test_mask)[0]
        all_predictions.append((test_indices, preds, fold))

        if fold % 20 == 0 or fold <= 3:
            log.info(
                f"  Fold {fold}/{total_folds}: train idx {train_lo}→{train_hi} "
                f"({len(X_train)} rows, {y_train.sum():.0f} pos), "
                f"test idx {test_start}→{test_end} ({len(X_test)} rows)"
            )

        test_start += ADVANCE_DAYS

    if not all_predictions:
        log.error("No predictions generated!")
        return pd.DataFrame()

    # Reassemble predictions into DataFrame (single operation, not per-fold)
    log.info("Assembling predictions...")
    all_indices = np.concatenate([p[0] for p in all_predictions])
    all_preds = np.concatenate([p[1] for p in all_predictions])
    all_folds = np.concatenate([np.full(len(p[1]), p[2]) for p in all_predictions])

    results = panel_clean.iloc[all_indices].copy()
    results["pred_prob"] = all_preds
    results["fold"] = all_folds
    results = results.drop(columns=["_date_idx"], errors="ignore")
    log.info(f"Walk-forward complete: {fold} folds, {len(results)} predictions")

    # Feature importance
    importance = pd.Series(
        model.feature_importance(importance_type="gain"),
        index=feature_cols
    ).sort_values(ascending=False)
    log.info("\nTop 15 features (gain):")
    for feat, imp in importance.head(15).items():
        log.info(f"  {feat}: {imp:.1f}")

    importance.to_csv(OUTPUT_DIR / "feature_importance.csv")

    return results


# ============================================================================
# STRADDLE BACKTEST
# ============================================================================

def backtest_straddles(results: pd.DataFrame, threshold: float) -> dict:
    """Simulate straddle trading based on ML predictions."""

    signals = results[results["pred_prob"] >= threshold].copy()
    if signals.empty:
        return {"threshold": threshold, "n_trades": 0, "total_return": 0}

    # For each signal, compute straddle P&L
    trades = []
    capital = INITIAL_CAPITAL
    equity_curve = []
    active_positions = []

    # Process day by day
    signal_dates = signals.index.unique().sort_values()

    for date in signal_dates:
        day_signals = signals.loc[[date]]

        # Close expired positions
        active_positions = [
            p for p in active_positions
            if (date - p["entry_date"]).days <= TARGET_HORIZON * 1.5  # ~30 calendar days
        ]

        for _, row in day_signals.iterrows():
            # Check concurrent position limit
            if len(active_positions) >= MAX_CONCURRENT:
                continue

            ticker = row["ticker"]
            S = row["close"]
            prob = row["pred_prob"]

            if pd.isna(S) or S <= 0:
                continue

            # Compute realized vol for pricing (use 21d historical)
            rvol = row.get("rvol_21d", 0.3)
            if pd.isna(rvol) or rvol <= 0:
                rvol = 0.3

            # Price the straddle
            premium_per_share, iv_used = straddle_price(S, rvol)

            # Position size: risk 2% of capital
            risk_amount = capital * RISK_PER_TRADE
            n_contracts = max(1, int(risk_amount / (premium_per_share * 100)))
            total_premium = premium_per_share * 100 * n_contracts

            if total_premium > capital * 0.10:  # don't put >10% in one trade
                n_contracts = max(1, int(capital * 0.10 / (premium_per_share * 100)))
                total_premium = premium_per_share * 100 * n_contracts

            # Compute actual move over next 21 days (from raw data)
            target_hit = row["target"]
            # Use the maximum absolute return feature for payoff approximation
            max_abs_move = row.get("max_abs_ret_21d", 0)

            # Actually compute forward P&L from the target column and price
            # If target=1, there was a >8% move. Payoff = (|move| - 0) * S * 100 * n_contracts - premium
            # We need to approximate the actual move size

            # The target tells us IF there was a big move. For P&L we need the size.
            # Use a simplified approach: if big move happened, approximate as
            # the expected conditional move given our threshold
            if target_hit == 1:
                # Big move happened — approximate payoff
                # Average conditional move when >8% is typically 10-15%
                # We'll use a conservative 10% as the expected move
                # (In production you'd compute exact forward returns)
                expected_move_pct = 0.10  # conservative estimate
                intrinsic = expected_move_pct * S * 100 * n_contracts
                pnl = intrinsic - total_premium
            else:
                # No big move — straddle likely expires worthless or near it
                # Small moves might recover some premium
                # Approximate: lose 80% of premium (some time value recovery)
                pnl = -total_premium * 0.80

            trades.append({
                "date": date,
                "ticker": ticker,
                "close": S,
                "prob": prob,
                "rvol_21d": rvol,
                "iv_used": iv_used,
                "premium_per_share": premium_per_share,
                "n_contracts": n_contracts,
                "total_premium": total_premium,
                "target": target_hit,
                "pnl": pnl,
            })

            capital += pnl
            equity_curve.append({"date": date, "capital": capital})
            active_positions.append({"entry_date": date, "ticker": ticker})

    if not trades:
        return {"threshold": threshold, "n_trades": 0, "total_return": 0}

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_curve)

    # Metrics
    n_trades = len(trades_df)
    winners = trades_df[trades_df["pnl"] > 0]
    losers = trades_df[trades_df["pnl"] <= 0]
    win_rate = len(winners) / n_trades if n_trades > 0 else 0
    total_pnl = trades_df["pnl"].sum()
    total_return = total_pnl / INITIAL_CAPITAL
    avg_win = winners["pnl"].mean() if len(winners) > 0 else 0
    avg_loss = losers["pnl"].mean() if len(losers) > 0 else 0
    profit_factor = abs(winners["pnl"].sum() / losers["pnl"].sum()) if len(losers) > 0 and losers["pnl"].sum() != 0 else np.inf

    # Monthly returns for Sharpe/Sortino
    trades_df["month"] = pd.to_datetime(trades_df["date"]).dt.to_period("M")
    monthly_pnl = trades_df.groupby("month")["pnl"].sum()
    monthly_ret = monthly_pnl / INITIAL_CAPITAL

    sharpe = monthly_ret.mean() / monthly_ret.std() * np.sqrt(12) if monthly_ret.std() > 0 else 0
    downside = monthly_ret[monthly_ret < 0].std()
    sortino = monthly_ret.mean() / downside * np.sqrt(12) if downside > 0 else 0

    # Max drawdown
    if len(equity_df) > 0:
        running_max = equity_df["capital"].cummax()
        drawdown = (equity_df["capital"] - running_max) / running_max
        max_dd = drawdown.min()
    else:
        max_dd = 0

    metrics = {
        "threshold": threshold,
        "n_trades": n_trades,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "total_return": total_return,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "final_capital": capital,
        "avg_trades_per_month": n_trades / max(1, len(monthly_pnl)),
    }

    return metrics, trades_df, equity_df


# ============================================================================
# ADVERSARIAL VALIDATION
# ============================================================================

def permutation_test(results: pd.DataFrame, best_threshold: float, real_sharpe: float) -> dict:
    """Shuffle trade signals and compare to real Sharpe. 30 permutations."""
    log.info(f"\n{'='*60}")
    log.info(f"PERMUTATION TEST ({N_PERMUTATIONS} shuffles)")
    log.info(f"{'='*60}")

    perm_sharpes = []
    for i in range(N_PERMUTATIONS):
        shuffled = results.copy()
        # HC #718 R2: Shuffle predictions WITHIN each date (cross-sectional permutation)
        # This tests whether stock SELECTION matters, not just being long vol
        if 'date' in shuffled.columns:
            for dt in shuffled['date'].unique():
                mask = shuffled['date'] == dt
                shuffled.loc[mask, 'pred_prob'] = np.random.permutation(
                    shuffled.loc[mask, 'pred_prob'].values)
        else:
            shuffled["pred_prob"] = np.random.permutation(shuffled["pred_prob"].values)

        try:
            result = backtest_straddles(shuffled, best_threshold)
            if isinstance(result, tuple):
                metrics = result[0]
            else:
                metrics = result
            perm_sharpes.append(metrics.get("sharpe", 0))
        except Exception:
            perm_sharpes.append(0)

        if (i + 1) % 10 == 0:
            log.info(f"  Permutation {i+1}/{N_PERMUTATIONS} done")

    perm_sharpes = np.array(perm_sharpes)
    p_value = np.mean(perm_sharpes >= real_sharpe)

    log.info(f"\nReal Sharpe: {real_sharpe:.3f}")
    log.info(f"Perm Sharpe mean: {np.mean(perm_sharpes):.3f} ± {np.std(perm_sharpes):.3f}")
    log.info(f"Perm Sharpe max:  {np.max(perm_sharpes):.3f}")
    log.info(f"p-value: {p_value:.3f}")
    log.info(f"RESULT: {'PASS (p < 0.05)' if p_value < 0.05 else 'FAIL (p >= 0.05) — could be noise'}")

    return {
        "real_sharpe": real_sharpe,
        "perm_mean": float(np.mean(perm_sharpes)),
        "perm_std": float(np.std(perm_sharpes)),
        "perm_max": float(np.max(perm_sharpes)),
        "p_value": float(p_value),
        "pass": p_value < 0.05,
    }


def subperiod_stability(trades_df: pd.DataFrame) -> dict:
    """Check stability across 4 sub-periods."""
    log.info(f"\n{'='*60}")
    log.info("SUB-PERIOD STABILITY")
    log.info(f"{'='*60}")

    trades_df = trades_df.copy()
    trades_df["date_dt"] = pd.to_datetime(trades_df["date"])
    min_date = trades_df["date_dt"].min()
    max_date = trades_df["date_dt"].max()
    total_days = (max_date - min_date).days
    quarter_days = total_days // 4

    results = {}
    for q in range(4):
        q_start = min_date + pd.Timedelta(days=q * quarter_days)
        q_end = min_date + pd.Timedelta(days=(q + 1) * quarter_days) if q < 3 else max_date + pd.Timedelta(days=1)

        q_trades = trades_df[
            (trades_df["date_dt"] >= q_start) & (trades_df["date_dt"] < q_end)
        ]

        if len(q_trades) == 0:
            results[f"Q{q+1}"] = {"n_trades": 0, "pnl": 0, "win_rate": 0, "sharpe": 0}
            continue

        n = len(q_trades)
        wr = (q_trades["pnl"] > 0).mean()
        pnl = q_trades["pnl"].sum()
        monthly = q_trades.set_index("date_dt").resample("ME")["pnl"].sum() / INITIAL_CAPITAL
        sharpe = monthly.mean() / monthly.std() * np.sqrt(12) if len(monthly) > 1 and monthly.std() > 0 else 0

        results[f"Q{q+1}"] = {
            "period": f"{q_start.date()} to {q_end.date()}",
            "n_trades": n,
            "pnl": float(pnl),
            "win_rate": float(wr),
            "sharpe": float(sharpe),
        }

        log.info(
            f"  Q{q+1} ({q_start.date()} → {q_end.date()}): "
            f"{n} trades, WR={wr:.1%}, PnL=${pnl:,.0f}, Sharpe={sharpe:.2f}"
        )

    # Check consistency
    sharpes = [v["sharpe"] for v in results.values() if v["n_trades"] > 0]
    if len(sharpes) >= 2:
        consistency = sum(1 for s in sharpes if s > 0) / len(sharpes)
        log.info(f"\n  Consistency: {consistency:.0%} of sub-periods profitable")
        log.info(f"  Sharpe range: {min(sharpes):.2f} to {max(sharpes):.2f}")
    else:
        consistency = 0

    results["consistency"] = float(consistency) if sharpes else 0
    return results


def regime_check(results_df: pd.DataFrame, trades_df: pd.DataFrame) -> dict:
    """R1 regime-agnostic check: compare performance in green vs red markets."""
    log.info(f"\n{'='*60}")
    log.info("REGIME-AGNOSTIC CHECK (R1 — HC #428)")
    log.info(f"{'='*60}")

    trades_df = trades_df.copy()
    trades_df["date_dt"] = pd.to_datetime(trades_df["date"])

    # Classify regimes by SPY 21d return
    spy_ret = results_df.get("spy_ret_21d")
    if spy_ret is None:
        log.warning("  No SPY return data for regime classification")
        return {"status": "no_data"}

    # Map each trade date to regime
    regime_map = {}
    for date in results_df.index.unique():
        spy_r = results_df.loc[date, "spy_ret_21d"]
        if isinstance(spy_r, pd.Series):
            spy_r = spy_r.iloc[0]
        if pd.isna(spy_r):
            regime_map[date] = "flat"
        elif spy_r > 0.02:
            regime_map[date] = "green"
        elif spy_r < -0.02:
            regime_map[date] = "red"
        else:
            regime_map[date] = "flat"

    trades_df["regime"] = trades_df["date"].map(regime_map).fillna("flat")

    regime_results = {}
    for regime in ["green", "red", "flat"]:
        rt = trades_df[trades_df["regime"] == regime]
        if len(rt) == 0:
            regime_results[regime] = {"n_trades": 0, "sharpe": 0}
            continue

        n = len(rt)
        wr = (rt["pnl"] > 0).mean()
        pnl = rt["pnl"].sum()
        monthly = rt.set_index("date_dt").resample("ME")["pnl"].sum() / INITIAL_CAPITAL
        sharpe = monthly.mean() / monthly.std() * np.sqrt(12) if len(monthly) > 1 and monthly.std() > 0 else 0

        regime_results[regime] = {
            "n_trades": int(n),
            "win_rate": float(wr),
            "pnl": float(pnl),
            "sharpe": float(sharpe),
        }
        log.info(f"  {regime:5s}: {n:4d} trades, WR={wr:.1%}, PnL=${pnl:>10,.0f}, Sharpe={sharpe:.2f}")

    # R1 check: |Sharpe_green - Sharpe_red| / max(|Sharpe_green|, |Sharpe_red|) < 0.50
    sg = regime_results.get("green", {}).get("sharpe", 0)
    sr = regime_results.get("red", {}).get("sharpe", 0)
    max_abs = max(abs(sg), abs(sr))
    regime_gap = abs(sg - sr) / max_abs if max_abs > 0 else 0

    regime_results["regime_gap"] = float(regime_gap)
    regime_results["r1_pass"] = regime_gap < 0.50

    log.info(f"\n  Regime gap: {regime_gap:.2f} {'PASS (<0.50)' if regime_gap < 0.50 else 'FAIL (>=0.50)'}")

    return regime_results


def outlier_robustness(trades_df: pd.DataFrame) -> dict:
    """Check if results are driven by outlier trades."""
    log.info(f"\n{'='*60}")
    log.info("OUTLIER ROBUSTNESS")
    log.info(f"{'='*60}")

    if len(trades_df) < 10:
        log.info("  Too few trades for outlier analysis")
        return {"status": "insufficient_trades"}

    total_pnl = trades_df["pnl"].sum()

    # Remove top 5% of trades
    cutoff_95 = trades_df["pnl"].quantile(0.95)
    trimmed = trades_df[trades_df["pnl"] <= cutoff_95]
    trimmed_pnl = trimmed["pnl"].sum()

    # Remove top 1% of trades
    cutoff_99 = trades_df["pnl"].quantile(0.99)
    trimmed_99 = trades_df[trades_df["pnl"] <= cutoff_99]
    trimmed_99_pnl = trimmed_99["pnl"].sum()

    log.info(f"  Full PnL:           ${total_pnl:>12,.0f}")
    log.info(f"  Without top 5%:     ${trimmed_pnl:>12,.0f} ({trimmed_pnl/total_pnl*100:.0f}% of full)" if total_pnl != 0 else "  Without top 5%: $0")
    log.info(f"  Without top 1%:     ${trimmed_99_pnl:>12,.0f} ({trimmed_99_pnl/total_pnl*100:.0f}% of full)" if total_pnl != 0 else "  Without top 1%: $0")

    # Check if profitable without outliers
    still_profitable = trimmed_pnl > 0

    log.info(f"  Still profitable without top 5%: {'YES' if still_profitable else 'NO'}")

    return {
        "total_pnl": float(total_pnl),
        "pnl_without_top5pct": float(trimmed_pnl),
        "pnl_without_top1pct": float(trimmed_99_pnl),
        "profitable_without_outliers": still_profitable,
    }


# ============================================================================
# IMPROVED BACKTEST WITH ACTUAL FORWARD RETURNS
# ============================================================================

def backtest_straddles_v2(results: pd.DataFrame, threshold: float, data: dict) -> tuple:
    """
    Improved backtest using actual forward returns from price data
    instead of approximations.
    """
    signals = results[results["pred_prob"] >= threshold].copy()
    if signals.empty:
        return {"threshold": threshold, "n_trades": 0, "total_return": 0}, pd.DataFrame(), pd.DataFrame()

    # Precompute forward returns for each stock
    fwd_data = {}
    for ticker in UNIVERSE:
        if ticker not in data:
            continue
        close = data[ticker]["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        fwd_data[ticker] = close

    trades = []
    capital = INITIAL_CAPITAL
    equity_curve = [{"date": signals.index[0], "capital": capital}]
    active_tickers_by_date = {}

    for date in signals.index.unique().sort_values():
        day_signals = signals.loc[[date]].sort_values("pred_prob", ascending=False)

        # Count active positions
        active_count = sum(
            1 for d, tickers in active_tickers_by_date.items()
            if (date - d).days <= 30  # ~21 trading days
            for _ in tickers
        )

        for _, row in day_signals.iterrows():
            if active_count >= MAX_CONCURRENT:
                break

            ticker = row["ticker"]
            S = row["close"]
            prob = row["pred_prob"]

            if pd.isna(S) or S <= 0 or ticker not in fwd_data:
                continue

            # Get actual forward prices
            close_series = fwd_data[ticker]
            future_dates = close_series.index[close_series.index > date][:TARGET_HORIZON]

            if len(future_dates) < 5:
                continue

            # HC #718 FIX: Use end-of-period price, NOT max-move (MFE bias)
            # Old code used fwd_returns.abs().max() — assumes perfect exit timing
            # Correct: straddle payoff at expiry = |S_final - K|
            future_prices = close_series.loc[future_dates]
            S_final = future_prices.iloc[-1]  # Price at end of holding period
            end_move_pct = abs(S_final / S - 1)
            end_move_price = end_move_pct * S
            # Also track max move for diagnostic comparison
            fwd_returns = (future_prices / S - 1).abs()
            max_abs_move_pct = fwd_returns.max()

            # Price the straddle
            rvol = row.get("rvol_21d", 0.3)
            if pd.isna(rvol) or rvol <= 0:
                rvol = 0.3
            premium_per_share, iv_used = straddle_price(S, rvol)

            # Position size
            risk_amount = capital * RISK_PER_TRADE
            n_contracts = max(1, int(risk_amount / (premium_per_share * 100)))
            total_premium = premium_per_share * 100 * n_contracts

            if total_premium > capital * 0.10:
                n_contracts = max(1, int(capital * 0.10 / (premium_per_share * 100)))
                total_premium = premium_per_share * 100 * n_contracts

            # Straddle payoff at expiry: |S_final - K| * 100 * n_contracts - premium
            # K = S (ATM), so intrinsic = |S_final - S| * 100
            intrinsic = end_move_price * 100 * n_contracts
            pnl = intrinsic - total_premium

            trades.append({
                "date": date,
                "ticker": ticker,
                "close": S,
                "prob": prob,
                "rvol_21d": rvol,
                "iv_used": iv_used,
                "premium_per_share": premium_per_share,
                "n_contracts": n_contracts,
                "total_premium": total_premium,
                "end_move_pct": float(end_move_pct),
                "max_move_pct": float(max_abs_move_pct),
                "target": 1 if max_abs_move_pct >= MOVE_THRESHOLD else 0,
                "pnl": pnl,
            })

            capital += pnl
            equity_curve.append({"date": date, "capital": capital})
            active_count += 1

            if date not in active_tickers_by_date:
                active_tickers_by_date[date] = []
            active_tickers_by_date[date].append(ticker)

    if not trades:
        return {"threshold": threshold, "n_trades": 0, "total_return": 0}, pd.DataFrame(), pd.DataFrame()

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_curve)

    # Compute metrics
    n_trades = len(trades_df)
    winners = trades_df[trades_df["pnl"] > 0]
    losers = trades_df[trades_df["pnl"] <= 0]
    win_rate = len(winners) / n_trades
    total_pnl = trades_df["pnl"].sum()
    total_return = total_pnl / INITIAL_CAPITAL
    avg_win = winners["pnl"].mean() if len(winners) > 0 else 0
    avg_loss = losers["pnl"].mean() if len(losers) > 0 else 0
    profit_factor = abs(winners["pnl"].sum() / losers["pnl"].sum()) if len(losers) > 0 and losers["pnl"].sum() != 0 else np.inf

    trades_df["month"] = pd.to_datetime(trades_df["date"]).dt.to_period("M")
    monthly_pnl = trades_df.groupby("month")["pnl"].sum()
    monthly_ret = monthly_pnl / INITIAL_CAPITAL

    sharpe = monthly_ret.mean() / monthly_ret.std() * np.sqrt(12) if monthly_ret.std() > 0 else 0
    downside = monthly_ret[monthly_ret < 0].std()
    sortino = monthly_ret.mean() / downside * np.sqrt(12) if downside > 0 else 0

    if len(equity_df) > 0:
        running_max = equity_df["capital"].cummax()
        drawdown = (equity_df["capital"] - running_max) / running_max
        max_dd = drawdown.min()
    else:
        max_dd = 0

    # Precision: what % of signals were actual big moves?
    precision = trades_df["target"].mean()
    # Average move when we traded
    avg_move = trades_df["end_move_pct"].mean()

    metrics = {
        "threshold": threshold,
        "n_trades": n_trades,
        "win_rate": float(win_rate),
        "total_pnl": float(total_pnl),
        "total_return": float(total_return),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "profit_factor": float(profit_factor),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": float(max_dd),
        "final_capital": float(capital),
        "avg_trades_per_month": n_trades / max(1, len(monthly_pnl)),
        "precision": float(precision),
        "avg_move_pct": float(avg_move),
    }

    return metrics, trades_df, equity_df


# ============================================================================
# MAIN
# ============================================================================

def main():
    t0 = time.time()
    log.info("=" * 70)
    log.info("ML VOLATILITY BREAKOUT — OPTIONS STRADDLE RESEARCH")
    log.info("=" * 70)
    log.info(f"Universe: {len(UNIVERSE)} stocks")
    log.info(f"Target: |move| > {MOVE_THRESHOLD*100:.0f}% in {TARGET_HORIZON}d")
    log.info(f"Walk-forward: {TRAIN_DAYS}d train, {ADVANCE_DAYS}d advance (sliding)")
    log.info(f"Capital: ${INITIAL_CAPITAL:,.0f}, Risk/trade: {RISK_PER_TRADE*100:.0f}%")
    log.info("")

    # 1. Download data
    data = download_data(start="2010-01-01")
    if len(data) < 15:
        log.error("Too few tickers downloaded. Aborting.")
        return

    # 2. Build panel
    panel = build_panel(data)
    if panel.empty:
        log.error("Empty panel. Aborting.")
        return

    panel.to_parquet(OUTPUT_DIR / "panel.parquet")
    log.info(f"Panel saved: {len(panel)} rows")

    # 3. Walk-forward
    results = run_walk_forward(panel)
    if results.empty:
        log.error("No walk-forward results. Aborting.")
        return

    results.to_parquet(OUTPUT_DIR / "wf_predictions.parquet")

    # 4. Evaluate multiple thresholds
    log.info(f"\n{'='*70}")
    log.info("THRESHOLD SCAN")
    log.info(f"{'='*70}")

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    all_metrics = []

    for thresh in thresholds:
        n_signals = (results["pred_prob"] >= thresh).sum()
        if n_signals == 0:
            log.info(f"  Threshold {thresh:.1f}: 0 signals — skipping")
            continue

        metrics, trades_df, equity_df = backtest_straddles_v2(results, thresh, data)

        if metrics["n_trades"] == 0:
            log.info(f"  Threshold {thresh:.1f}: 0 trades — skipping")
            continue

        all_metrics.append(metrics)
        log.info(
            f"  Threshold {thresh:.1f}: {metrics['n_trades']:>5d} trades, "
            f"WR={metrics['win_rate']:.1%}, PF={metrics['profit_factor']:.2f}, "
            f"Sharpe={metrics['sharpe']:.2f}, Sortino={metrics['sortino']:.2f}, "
            f"Return={metrics['total_return']:.1%}, MaxDD={metrics['max_drawdown']:.1%}, "
            f"Precision={metrics['precision']:.1%}"
        )

    if not all_metrics:
        log.error("No valid threshold produced trades. Aborting.")
        return

    # Pick best threshold by Sharpe
    best = max(all_metrics, key=lambda x: x["sharpe"])
    best_thresh = best["threshold"]
    log.info(f"\nBest threshold: {best_thresh} (Sharpe={best['sharpe']:.2f})")

    # Re-run best for detailed analysis
    best_metrics, best_trades, best_equity = backtest_straddles_v2(results, best_thresh, data)

    # Save trades
    best_trades.to_csv(OUTPUT_DIR / "trades.csv", index=False)
    best_equity.to_csv(OUTPUT_DIR / "equity_curve.csv", index=False)

    # 5. ADVERSARIAL VALIDATION
    log.info(f"\n{'='*70}")
    log.info("ADVERSARIAL VALIDATION")
    log.info(f"{'='*70}")

    # 5a. Permutation test
    perm_results = permutation_test(results, best_thresh, best["sharpe"])

    # 5b. Sub-period stability
    subperiod_results = subperiod_stability(best_trades)

    # 5c. Regime check
    regime_results = regime_check(results, best_trades)

    # 5d. Outlier robustness
    outlier_results = outlier_robustness(best_trades)

    # 6. SUMMARY
    elapsed = time.time() - t0
    log.info(f"\n{'='*70}")
    log.info("FINAL SUMMARY")
    log.info(f"{'='*70}")
    log.info(f"Strategy: ML Vol Breakout → ATM Straddle")
    log.info(f"Universe: {len(UNIVERSE)} mega-cap stocks")
    log.info(f"OOT Period: walk-forward sliding")
    log.info(f"Best Threshold: {best_thresh}")
    log.info(f"")
    log.info(f"PERFORMANCE:")
    log.info(f"  Trades:         {best_metrics['n_trades']}")
    log.info(f"  Win Rate:       {best_metrics['win_rate']:.1%}")
    log.info(f"  Profit Factor:  {best_metrics['profit_factor']:.2f}")
    log.info(f"  Sharpe:         {best_metrics['sharpe']:.2f}")
    log.info(f"  Sortino:        {best_metrics['sortino']:.2f}")
    log.info(f"  Total Return:   {best_metrics['total_return']:.1%}")
    log.info(f"  Max Drawdown:   {best_metrics['max_drawdown']:.1%}")
    log.info(f"  Final Capital:  ${best_metrics['final_capital']:,.0f}")
    log.info(f"  Precision:      {best_metrics['precision']:.1%}")
    log.info(f"  Avg Move:       {best_metrics['avg_move_pct']:.1%}")
    log.info(f"")
    log.info(f"ADVERSARIAL:")
    log.info(f"  Perm Test:      {'PASS' if perm_results['pass'] else 'FAIL'} (p={perm_results['p_value']:.3f})")
    log.info(f"  Sub-period:     {subperiod_results.get('consistency', 0):.0%} consistent")
    log.info(f"  Regime (R1):    {'PASS' if regime_results.get('r1_pass', False) else 'FAIL'} (gap={regime_results.get('regime_gap', 'N/A')})")
    log.info(f"  Outlier robust: {'YES' if outlier_results.get('profitable_without_outliers', False) else 'NO'}")
    log.info(f"")
    log.info(f"Runtime: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # Save full results
    summary = {
        "strategy": "ML Vol Breakout — ATM Straddle",
        "universe_size": len(UNIVERSE),
        "target_threshold": MOVE_THRESHOLD,
        "target_horizon": TARGET_HORIZON,
        "train_window": TRAIN_DAYS,
        "advance": ADVANCE_DAYS,
        "best_threshold": best_thresh,
        "metrics": best_metrics,
        "threshold_scan": all_metrics,
        "permutation_test": perm_results,
        "subperiod_stability": {k: v for k, v in subperiod_results.items()},
        "regime_check": {k: v for k, v in regime_results.items() if isinstance(v, (dict, float, bool))},
        "outlier_robustness": outlier_results,
        "runtime_seconds": elapsed,
    }

    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    log.info(f"\nResults saved to {OUTPUT_DIR}/")
    log.info("DONE.")


if __name__ == "__main__":
    main()

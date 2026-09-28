#!/usr/bin/env python3
"""
Adaptive VIX-Band Strategy v1
==============================

Instead of binary VIX>20/VIX<20, uses 4 VIX bands with different option structures:

  Band 1 — VIX < 15:  No trades (premium too thin, sit in cash)
  Band 2 — VIX 15-20: Sell put credit spreads on top momentum sectors
                       (collect premium in mild vol, defined risk)
  Band 3 — VIX 20-30: Buy bull call spreads on top momentum sectors
                       (our proven Sharpe 3.04 strategy)
  Band 4 — VIX 30+:   Buy deep OTM bull call spreads on top momentum sectors
                       (crash recovery play — cheap calls for mean reversion)

Motivation:
  - Our best strategy (sector bull call spreads) only trades when VIX>20.
  - VIX<20 is ~60% of history — that's a lot of idle capital.
  - Bear puts in VIX<20 had -18% to -49% MDD.
  - Put credit spreads in VIX 15-20 collect premium with defined risk.
  - VIX 30+ deep OTM calls exploit crash recovery (VIX mean reversion).

Design:
  - Walk-forward LGBM for sector ranking (same proven engine)
  - $645 starting capital
  - Hold to expiry (no early exit)
  - 15% bid-ask haircut on BOTH entry AND exit
  - Random baseline comparison per band
  - Honest Sharpe (equity-based, calendar month)
  - MLflow logging

Uses standardized tools:
  - research.tools.options_pricer for pricing
  - research.tools.adversarial_validator for validation
"""
from __future__ import annotations

import sys
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, "/home/jupiter/Lvl3Quant")

from research.tools.options_pricer import (
    price_bull_call_spread,
    price_bear_put_spread,
    exit_spread_value,
    spread_pnl,
    estimate_iv,
    bs_put_price,
    bs_call_price,
    compute_atr,
    COMMISSION_RT_SPREAD,
    DEFAULT_HAIRCUT,
)
from research.tools.adversarial_validator import validate_trades


def fprint(*args, **kwargs):
    print(*args, **kwargs, flush=True)


# ─── MLflow Setup ──────────────────────────────────────────────────

MLFLOW_OK = False
try:
    import urllib.request
    urllib.request.urlopen("http://jupiter:5000/", timeout=2)
    import mlflow
    mlflow.set_tracking_uri("http://jupiter:5000")
    MLFLOW_OK = True
    fprint("[MLflow] Connected")
except Exception:
    fprint("[MLflow] Not available, skipping logging")


# ─── Configuration ─────────────────────────────────────────────────

CONFIG = {
    # Universe
    "sector_etfs": [
        "XLE", "XLK", "XLF", "XLV", "XLI", "XLP", "XLU", "XLRE", "XLB", "XLC", "XLY",
    ],
    "benchmark": "SPY",

    # Dates
    "start_date": "2011-01-01",
    "end_date": "2026-07-25",

    # Capital
    "initial_capital": 645.0,
    "max_trade_pct": 0.40,
    "max_total_pct": 0.80,

    # Common
    "target_dte": 30,
    "top_n": 3,
    "rebalance_freq_days": 10,
    "haircut": DEFAULT_HAIRCUT,
    "commission_rt": COMMISSION_RT_SPREAD,

    # VIX Bands
    "vix_band_1_max": 15.0,   # No trade
    "vix_band_2_max": 20.0,   # Put credit spreads
    "vix_band_3_max": 30.0,   # Bull call spreads (proven)
    # VIX >= 30: Deep OTM bull call spreads

    # Band 2: Put credit spread parameters
    "pcs_spread_width_pct": 0.03,   # 3% wide spread below spot
    "pcs_otm_pct": 0.03,            # Short strike 3% OTM (below spot)

    # Band 3: Bull call spread parameters (proven config)
    "bcs_spread_width_pct": 0.03,   # 3% wide

    # Band 4: Deep OTM bull call spread parameters
    "deep_otm_pct": 0.05,           # Long strike 5% OTM
    "deep_spread_width_pct": 0.05,  # 5% wide (cheap deep OTM)

    # Walk-forward LGBM
    "train_months": 12,
    "test_months": 1,
    "lgbm_params": {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "max_depth": 6,
        "verbosity": -1,
        "seed": 42,
        "n_jobs": 4,  # n_jobs=-1 deadlocks with early_stopping callback in this LGBM version
    },
    "num_boost_round": 300,
    "early_stopping_rounds": 30,

    # Validation
    "n_perms": 500,
}


# ─── Data Loading (reuses sector_backtest pattern) ─────────────────

def load_market_data(tickers, start_date, end_date):
    """Load price data and VIX from yfinance."""
    import yfinance as yf

    all_tickers = list(set(tickers))
    fprint(f"  Loading {len(all_tickers)} tickers from {start_date} to {end_date}...")

    data = yf.download(
        all_tickers, start=start_date, end=end_date,
        auto_adjust=True, progress=False, group_by="ticker",
    )

    prices = {}
    for ticker in all_tickers:
        try:
            if len(all_tickers) > 1:
                df = data[ticker][["Close", "High", "Low", "Volume"]].dropna()
            else:
                df = data[["Close", "High", "Low", "Volume"]].dropna()
            df.columns = ["close", "high", "low", "volume"]
            if len(df) > 60:
                prices[ticker] = df
        except Exception:
            pass

    # VIX
    vix_data = yf.download(
        "^VIX", start=start_date, end=end_date,
        auto_adjust=True, progress=False,
    )
    vix_series = vix_data["Close"].dropna()
    if hasattr(vix_series, "columns"):
        vix_series = vix_series.iloc[:, 0]

    fprint(f"  Got {len(prices)} tickers, VIX has {len(vix_series)} days")
    return prices, vix_series


# ─── Feature Engineering (same as sector_backtest) ─────────────────

def build_features(prices, vix_series, sector_etfs, benchmark="SPY"):
    """Build feature panel for LGBM sector ranking (vectorized)."""
    spy_close = prices[benchmark]["close"]
    frames = []

    for ticker in sector_etfs:
        if ticker not in prices:
            continue
        df = prices[ticker].copy()
        close = df["close"]
        volume = df["volume"]

        feat = pd.DataFrame(index=df.index)
        feat["ticker"] = ticker

        for lookback in [5, 10, 21, 63]:
            feat[f"mom_{lookback}d"] = close.pct_change(lookback)

        for lookback in [21, 63]:
            spy_ret = spy_close.pct_change(lookback).reindex(close.index)
            ticker_ret = close.pct_change(lookback)
            feat[f"rs_vs_spy_{lookback}d"] = ticker_ret - spy_ret

        feat["vol_21d"] = close.pct_change().rolling(21).std() * np.sqrt(252)
        feat["vol_63d"] = close.pct_change().rolling(63).std() * np.sqrt(252)
        feat["vol_ratio"] = feat["vol_21d"] / (feat["vol_63d"] + 1e-8)
        feat["vol_sma_ratio"] = volume / volume.rolling(21).mean()

        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / 14, min_periods=14).mean()
        avg_loss = loss.ewm(alpha=1 / 14, min_periods=14).mean()
        feat["rsi_14"] = 100 - (100 / (1 + avg_gain / (avg_loss + 1e-8)))

        feat["sma_50_dist"] = close / close.rolling(50).mean() - 1
        feat["sma_200_dist"] = close / close.rolling(200).mean() - 1

        # Target + meta columns
        feat["target"] = close.pct_change(21).shift(-21)
        feat["close"] = close
        feat["atr_14"] = _compute_atr_series(df["high"], df["low"], close)
        feat["vix"] = vix_series.reindex(df.index, method="ffill")

        frames.append(feat)

    panel = pd.concat(frames)
    panel = panel.dropna(subset=["target"])
    panel = panel.reset_index().rename(columns={"index": "date", "Date": "date"})
    if "date" not in panel.columns and "index" in panel.columns:
        panel = panel.rename(columns={"index": "date"})
    # Ensure we have a 'date' column from the index
    if "date" not in panel.columns:
        panel["date"] = panel.index

    fprint(f"  Feature panel: {len(panel)} rows, {len(panel.columns)} columns")
    return panel


def _compute_atr_series(high, low, close, period=14):
    """Compute ATR as a full series (vectorized)."""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period).mean()


# ─── Walk-Forward LGBM ─────────────────────────────────────────────

def walk_forward_lgbm(panel, config):
    """Walk-forward LGBM ranking with sliding window (optimized)."""
    import lightgbm as lgb

    feature_cols = [c for c in panel.columns if c not in
                    ["date", "ticker", "target", "close", "atr_14", "vix"]]

    panel = panel.sort_values("date").reset_index(drop=True)
    dates = sorted(panel["date"].unique())

    # Build date-to-integer mapping for fast slicing
    date_to_idx = {d: i for i, d in enumerate(dates)}
    panel["_date_idx"] = panel["date"].map(date_to_idx)
    panel = panel.sort_values("_date_idx")

    train_months = config.get("train_months", 12)
    test_months = config.get("test_months", 1)
    train_days = train_months * 21
    test_days = test_months * 21

    # Pre-extract numpy arrays for speed
    feat_vals = np.nan_to_num(panel[feature_cols].values.astype(np.float32),
                               nan=0, posinf=0, neginf=0)
    target_vals = np.nan_to_num(panel["target"].values.astype(np.float32),
                                 nan=0, posinf=0, neginf=0)
    date_idx_vals = panel["_date_idx"].values

    predictions = []
    n_folds = 0

    for i in range(train_days, len(dates), max(1, test_days)):
        if i >= len(dates):
            break

        train_start = max(0, i - train_days)
        test_end = min(len(dates), i + test_days)

        if i - train_start < 100:
            continue

        # Fast integer range masking instead of isin()
        train_mask = (date_idx_vals >= train_start) & (date_idx_vals < i)
        test_mask = (date_idx_vals >= i) & (date_idx_vals < test_end)

        if not test_mask.any():
            continue

        X_train = feat_vals[train_mask]
        y_train = target_vals[train_mask]
        X_test = feat_vals[test_mask]

        try:
            dtrain = lgb.Dataset(X_train, label=y_train)
            val_start = max(0, len(X_train) - len(X_train) // 5)
            dval = lgb.Dataset(X_train[val_start:], label=y_train[val_start:])

            model = lgb.train(
                config.get("lgbm_params", CONFIG["lgbm_params"]),
                dtrain,
                num_boost_round=config.get("num_boost_round", 300),
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(
                    config.get("early_stopping_rounds", 30),
                    verbose=False,
                )],
            )

            preds = model.predict(X_test)
            test_data = panel.loc[test_mask, ["date", "ticker", "close", "atr_14", "vix"]].copy()
            test_data["pred"] = preds
            predictions.append(test_data)
            n_folds += 1

            if n_folds % 20 == 0:
                fprint(f"    ... fold {n_folds}, date range up to {dates[min(test_end-1, len(dates)-1)]}")

        except Exception:
            continue

    panel.drop(columns=["_date_idx"], inplace=True, errors="ignore")

    if not predictions:
        return pd.DataFrame()

    result = pd.concat(predictions, ignore_index=True)
    fprint(f"  Walk-forward: {n_folds} folds, {len(result)} predictions")
    return result


# ─── Put Credit Spread Pricing ─────────────────────────────────────

def price_put_credit_spread(
    S: float,
    K1: float,  # lower strike (long put, further OTM)
    K2: float,  # upper strike (short put, closer to spot)
    dte: int,
    atr: float,
    vix: float = 20.0,
    haircut: float = DEFAULT_HAIRCUT,
) -> tuple[float, float]:
    """
    Price a put credit spread (sell K2 put, buy K1 put).

    Credit received = BS_put(K2) - BS_put(K1), with haircut reducing credit.
    Max loss = (K2 - K1) - credit received.
    Max profit = credit received.

    Returns:
        (credit_received, max_loss) — per-share values.
        credit_received: what you RECEIVE after haircut (less than fair).
        max_loss: maximum possible loss if both puts expire ITM.
    """
    if K2 <= K1:
        raise ValueError(f"K2 ({K2}) must be > K1 ({K1}) for put credit spread")

    T = dte / 365.0
    sigma = estimate_iv(atr, S, vix)

    short_put_val = bs_put_price(S, K2, T, sigma=sigma)
    long_put_val = bs_put_price(S, K1, T, sigma=sigma)

    fair_credit = short_put_val - long_put_val
    fair_credit = max(fair_credit, 0.001)

    # You SELL the spread → receive LESS than fair (haircut DOWN on credit)
    credit_received = fair_credit * (1.0 - haircut)

    spread_width = K2 - K1
    max_loss = spread_width - credit_received

    return float(credit_received), float(max_loss)


def put_credit_spread_pnl_at_expiry(
    S_expiry: float,
    K1: float,
    K2: float,
    credit_received: float,
    contracts: int = 1,
    commission_rt: float = COMMISSION_RT_SPREAD,
) -> float:
    """
    PnL of a put credit spread at expiry.

    If S >= K2: full profit = credit * 100 * contracts - commission
    If S <= K1: max loss = (width - credit) * 100 * contracts + commission
    If K1 < S < K2: partial loss
    """
    # Short put intrinsic at expiry (liability)
    short_put_intrinsic = max(K2 - S_expiry, 0.0)
    # Long put intrinsic at expiry (asset)
    long_put_intrinsic = max(K1 - S_expiry, 0.0)

    # Net value at expiry (negative = you owe)
    spread_liability = short_put_intrinsic - long_put_intrinsic

    # PnL = credit received - liability at expiry
    pnl_per_share = credit_received - spread_liability
    return pnl_per_share * contracts * 100 - commission_rt * contracts


# ─── VIX Band Classification ──────────────────────────────────────

def classify_vix_band(vix: float, config: dict) -> int:
    """
    Classify VIX into a band.
    Returns: 0 = no trade, 1 = put credit spread, 2 = bull call spread, 3 = deep OTM bull call.
    """
    if vix < config["vix_band_1_max"]:
        return 0  # No trade
    elif vix < config["vix_band_2_max"]:
        return 1  # Put credit spread
    elif vix < config["vix_band_3_max"]:
        return 2  # Bull call spread (proven)
    else:
        return 3  # Deep OTM bull call spread


# ─── Trade Generation (Adaptive) ──────────────────────────────────

def generate_adaptive_trades(
    predictions: pd.DataFrame,
    prices: dict,
    vix_series: pd.Series,
    config: dict,
    use_random_selection: bool = False,
) -> list[dict]:
    """
    Generate trades across all VIX bands.

    Returns list of trade dicts with:
        ticker, entry_date, exit_date, pnl, vix_band, trade_type,
        entry_cost (or credit_received), contracts, K1, K2
    """
    initial_capital = config["initial_capital"]
    top_n = config["top_n"]
    target_dte = config["target_dte"]
    rebalance_freq = config["rebalance_freq_days"]
    haircut = config["haircut"]
    commission_rt = config["commission_rt"]
    max_trade_pct = config["max_trade_pct"]
    max_total_pct = config["max_total_pct"]

    current_equity = initial_capital
    active_positions = []
    trades = []
    last_rebalance = None

    pred_dates = sorted(predictions["date"].unique())

    for date in pred_dates:
        date_ts = pd.Timestamp(date)

        # ── Check/exit existing positions ──
        new_active = []
        for pos in active_positions:
            days_held = (date_ts - pos["entry_date"]).days
            ticker = pos["ticker"]

            if ticker not in prices or date_ts not in prices[ticker].index:
                new_active.append(pos)
                continue

            spot = float(prices[ticker].loc[date_ts, "close"])
            remaining_dte = max(pos["original_dte"] - days_held, 0)

            if remaining_dte <= 0:
                # EXPIRY
                if pos["trade_type"] == "put_credit_spread":
                    pnl = put_credit_spread_pnl_at_expiry(
                        spot, pos["K1"], pos["K2"],
                        pos["credit_received"], pos["contracts"], commission_rt,
                    )
                else:
                    # Bull call spread (band 2 or 3)
                    exit_val = exit_spread_value(
                        spot, pos["K1"], pos["K2"], 0, pos["original_dte"],
                        pos["atr"], pos.get("vix_at_entry", 20.0), haircut,
                    )
                    pnl = spread_pnl(
                        pos["entry_cost"], exit_val, pos["contracts"], commission_rt,
                    )

                current_equity += pnl
                trades.append({
                    "ticker": ticker,
                    "entry_date": pos["entry_date"],
                    "exit_date": date_ts,
                    "days_held": days_held,
                    "pnl": pnl,
                    "contracts": pos["contracts"],
                    "K1": pos["K1"],
                    "K2": pos["K2"],
                    "vix_band": pos["vix_band"],
                    "trade_type": pos["trade_type"],
                    "exit_reason": "expiry",
                    "equity_after": current_equity,
                })
            else:
                new_active.append(pos)

        active_positions = new_active

        # ── Rebalance check ──
        if last_rebalance is not None:
            trading_days_since = len(
                [d for d in pred_dates if d > last_rebalance and d <= date]
            )
            if trading_days_since < rebalance_freq:
                continue

        # ── VIX lookup ──
        if date_ts in vix_series.index:
            current_vix = float(vix_series.loc[date_ts])
        else:
            prior = vix_series.index[vix_series.index <= date_ts]
            if len(prior) == 0:
                continue
            current_vix = float(vix_series.loc[prior[-1]])

        vix_band = classify_vix_band(current_vix, config)

        if vix_band == 0:
            # No trade in VIX < 15
            continue

        # ── Get rankings ──
        day_preds = predictions[predictions["date"] == date].copy()
        if len(day_preds) < 3:
            continue

        if use_random_selection:
            preds_shuffled = day_preds["pred"].values.copy()
            np.random.shuffle(preds_shuffled)
            day_preds["pred"] = preds_shuffled

        day_preds = day_preds.sort_values("pred", ascending=False)
        top_sectors = day_preds.head(top_n)["ticker"].tolist()

        held_tickers = {p["ticker"] for p in active_positions}
        top_sectors = [t for t in top_sectors if t not in held_tickers]

        if not top_sectors:
            continue

        last_rebalance = date

        for ticker in top_sectors:
            if current_equity < 50:
                break

            if ticker not in prices or date_ts not in prices[ticker].index:
                continue

            spot = float(prices[ticker].loc[date_ts, "close"])

            # Get ATR
            hist = prices[ticker]
            prior_data = hist[hist.index <= date_ts].tail(30)
            if len(prior_data) < 14:
                continue

            atr_val = compute_atr(prior_data["high"], prior_data["low"], prior_data["close"])

            # ── Band-specific trade construction ──

            if vix_band == 1:
                # PUT CREDIT SPREAD (VIX 15-20)
                # Sell put at 3% below spot, buy put 3% below that
                K2_put = round(spot * (1 - config["pcs_otm_pct"]), 0)   # short put
                K1_put = round(K2_put * (1 - config["pcs_spread_width_pct"]), 0)  # long put
                if K2_put <= K1_put:
                    continue

                try:
                    credit, max_loss_ps = price_put_credit_spread(
                        spot, K1_put, K2_put, target_dte, atr_val, current_vix, haircut,
                    )
                except Exception:
                    continue

                if credit <= 0.005 or max_loss_ps <= 0:
                    continue

                # Position sizing: risk max_trade_pct of equity
                max_loss_per_contract = max_loss_ps * 100 + commission_rt
                trade_budget = current_equity * max_trade_pct
                contracts = max(1, int(trade_budget / max_loss_per_contract))

                total_risk = contracts * max_loss_per_contract
                if total_risk > current_equity * max_total_pct:
                    contracts = max(1, int(
                        (current_equity * max_total_pct) / max_loss_per_contract
                    ))

                if contracts < 1:
                    continue

                active_positions.append({
                    "ticker": ticker,
                    "entry_date": date_ts,
                    "K1": K1_put,
                    "K2": K2_put,
                    "credit_received": credit,
                    "contracts": contracts,
                    "original_dte": target_dte,
                    "atr": atr_val,
                    "vix_at_entry": current_vix,
                    "vix_band": 1,
                    "trade_type": "put_credit_spread",
                })

            elif vix_band == 2:
                # BULL CALL SPREAD (VIX 20-30) — proven strategy
                K1_call = round(spot, 0)
                K2_call = round(spot * (1 + config["bcs_spread_width_pct"]), 0)
                if K2_call <= K1_call:
                    K2_call = K1_call + 1

                try:
                    entry_cost, max_profit = price_bull_call_spread(
                        spot, K1_call, K2_call, target_dte, atr_val, current_vix, haircut,
                    )
                except Exception:
                    continue

                if entry_cost <= 0.01 or max_profit <= 0:
                    continue

                cost_per_contract = entry_cost * 100
                trade_budget = min(current_equity * max_trade_pct, 250.0)
                contracts = max(1, int(trade_budget / (cost_per_contract + commission_rt)))

                total_cost = contracts * cost_per_contract + commission_rt * contracts
                if total_cost > current_equity * max_total_pct:
                    contracts = max(1, int(
                        (current_equity * max_total_pct - commission_rt) / cost_per_contract
                    ))

                if contracts < 1:
                    continue

                active_positions.append({
                    "ticker": ticker,
                    "entry_date": date_ts,
                    "K1": K1_call,
                    "K2": K2_call,
                    "entry_cost": entry_cost,
                    "contracts": contracts,
                    "original_dte": target_dte,
                    "atr": atr_val,
                    "vix_at_entry": current_vix,
                    "vix_band": 2,
                    "trade_type": "bull_call_spread",
                })

            elif vix_band == 3:
                # DEEP OTM BULL CALL SPREAD (VIX 30+) — crash recovery
                K1_deep = round(spot * (1 + config["deep_otm_pct"]), 0)
                K2_deep = round(spot * (1 + config["deep_otm_pct"] + config["deep_spread_width_pct"]), 0)
                if K2_deep <= K1_deep:
                    K2_deep = K1_deep + 1

                try:
                    entry_cost, max_profit = price_bull_call_spread(
                        spot, K1_deep, K2_deep, target_dte, atr_val, current_vix, haircut,
                    )
                except Exception:
                    continue

                if entry_cost <= 0.005 or max_profit <= 0:
                    continue

                # Use smaller budget for speculative plays
                cost_per_contract = entry_cost * 100
                trade_budget = min(current_equity * max_trade_pct * 0.5, 150.0)
                contracts = max(1, int(trade_budget / (cost_per_contract + commission_rt)))

                total_cost = contracts * cost_per_contract + commission_rt * contracts
                if total_cost > current_equity * max_total_pct * 0.5:
                    contracts = max(1, int(
                        (current_equity * max_total_pct * 0.5 - commission_rt) / cost_per_contract
                    ))

                if contracts < 1:
                    continue

                active_positions.append({
                    "ticker": ticker,
                    "entry_date": date_ts,
                    "K1": K1_deep,
                    "K2": K2_deep,
                    "entry_cost": entry_cost,
                    "contracts": contracts,
                    "original_dte": target_dte,
                    "atr": atr_val,
                    "vix_at_entry": current_vix,
                    "vix_band": 3,
                    "trade_type": "deep_otm_bull_call",
                })

    # ── Force-close remaining positions at last date ──
    last_date = pd.Timestamp(pred_dates[-1]) if pred_dates else pd.Timestamp.now()
    for pos in active_positions:
        ticker = pos["ticker"]
        days_held = (last_date - pos["entry_date"]).days

        if ticker in prices:
            avail = prices[ticker].index[prices[ticker].index <= last_date]
            if len(avail) > 0:
                spot = float(prices[ticker].loc[avail[-1], "close"])

                if pos["trade_type"] == "put_credit_spread":
                    pnl = put_credit_spread_pnl_at_expiry(
                        spot, pos["K1"], pos["K2"],
                        pos["credit_received"], pos["contracts"], commission_rt,
                    )
                else:
                    remaining_dte = max(pos["original_dte"] - days_held, 0)
                    exit_val = exit_spread_value(
                        spot, pos["K1"], pos["K2"], remaining_dte, pos["original_dte"],
                        pos["atr"], pos.get("vix_at_entry", 20.0), haircut,
                    )
                    pnl = spread_pnl(
                        pos["entry_cost"], exit_val, pos["contracts"], commission_rt,
                    )

                current_equity += pnl
                trades.append({
                    "ticker": ticker,
                    "entry_date": pos["entry_date"],
                    "exit_date": last_date,
                    "days_held": days_held,
                    "pnl": pnl,
                    "contracts": pos["contracts"],
                    "K1": pos["K1"],
                    "K2": pos["K2"],
                    "vix_band": pos["vix_band"],
                    "trade_type": pos["trade_type"],
                    "exit_reason": "end_of_data",
                    "equity_after": current_equity,
                })

    return trades


# ─── Band-Level Analysis ──────────────────────────────────────────

def analyze_by_band(trades: list[dict]) -> dict:
    """Break down performance by VIX band."""
    band_names = {
        1: "VIX 15-20 (Put Credit Spreads)",
        2: "VIX 20-30 (Bull Call Spreads)",
        3: "VIX 30+ (Deep OTM Bull Calls)",
    }

    results = {}
    for band_id, band_name in band_names.items():
        band_trades = [t for t in trades if t.get("vix_band") == band_id]
        n = len(band_trades)
        if n == 0:
            results[band_name] = {"n_trades": 0, "total_pnl": 0, "avg_pnl": 0, "win_rate": 0, "profit_factor": 0}
            continue

        pnls = [t["pnl"] for t in band_trades]
        wins = [p for p in pnls if p > 0]
        wr = len(wins) / n if n > 0 else 0
        total = sum(pnls)
        avg = total / n if n > 0 else 0
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(p for p in pnls if p <= 0)) or 1e-9
        pf = gross_profit / gross_loss

        results[band_name] = {
            "n_trades": n,
            "total_pnl": round(total, 2),
            "avg_pnl": round(avg, 2),
            "win_rate": round(wr, 4),
            "profit_factor": round(pf, 3),
        }

    return results


# ─── Main ──────────────────────────────────────────────────────────

def main():
    fprint("=" * 70)
    fprint("  ADAPTIVE VIX-BAND STRATEGY v1")
    fprint("  4 bands: cash / put credit spreads / bull call spreads / deep OTM")
    fprint("=" * 70)

    config = CONFIG.copy()

    # Step 1: Load data
    fprint("\nStep 1: Loading market data...")
    all_tickers = config["sector_etfs"] + [config["benchmark"]]
    prices, vix_series = load_market_data(all_tickers, config["start_date"], config["end_date"])

    spy_prices = prices.get(config["benchmark"], {})
    if isinstance(spy_prices, pd.DataFrame):
        spy_prices = spy_prices["close"]

    # VIX band distribution
    fprint("\n  VIX Band Distribution:")
    vix_vals = vix_series.values
    pct_band0 = (vix_vals < config["vix_band_1_max"]).mean() * 100
    pct_band1 = ((vix_vals >= config["vix_band_1_max"]) & (vix_vals < config["vix_band_2_max"])).mean() * 100
    pct_band2 = ((vix_vals >= config["vix_band_2_max"]) & (vix_vals < config["vix_band_3_max"])).mean() * 100
    pct_band3 = (vix_vals >= config["vix_band_3_max"]).mean() * 100
    fprint(f"    Band 0 (VIX<15, cash):   {pct_band0:.1f}%")
    fprint(f"    Band 1 (VIX 15-20, PCS): {pct_band1:.1f}%")
    fprint(f"    Band 2 (VIX 20-30, BCS): {pct_band2:.1f}%")
    fprint(f"    Band 3 (VIX 30+, Deep):  {pct_band3:.1f}%")

    # Step 2: Build features
    fprint("\nStep 2: Building features...")
    panel = build_features(prices, vix_series, config["sector_etfs"], config["benchmark"])

    # Step 3: Walk-forward LGBM
    fprint("\nStep 3: Walk-forward LGBM ranking...")
    predictions = walk_forward_lgbm(panel, config)

    if len(predictions) == 0:
        fprint("ERROR: No predictions generated")
        return

    # Step 4: Generate adaptive trades
    fprint("\nStep 4: Generating adaptive trades...")
    trades = generate_adaptive_trades(predictions, prices, vix_series, config)
    fprint(f"  Generated {len(trades)} trades total")

    if len(trades) < 10:
        fprint(f"ERROR: Only {len(trades)} trades generated. Not enough for validation.")
        return

    # Band breakdown
    band_analysis = analyze_by_band(trades)
    fprint("\n  Per-Band Breakdown:")
    for band_name, stats in band_analysis.items():
        fprint(f"    {band_name}: {stats['n_trades']} trades, "
               f"PnL=${stats['total_pnl']:.2f}, WR={stats['win_rate']*100:.1f}%, "
               f"PF={stats['profit_factor']:.2f}")

    # Step 5: Adversarial validation (full strategy)
    fprint("\nStep 5: Adversarial validation (combined strategy)...")
    validation = validate_trades(
        trades,
        initial_capital=config["initial_capital"],
        spy_prices=spy_prices,
        strategy_name="Adaptive VIX-Band v1 (combined)",
        n_perms=config["n_perms"],
    )
    validation.print_summary()

    # Step 6: Random baseline
    fprint("\nStep 6: Random baseline comparison...")
    random_trades = generate_adaptive_trades(
        predictions, prices, vix_series, config, use_random_selection=True,
    )

    random_validation = None
    ml_vs_random = None

    if len(random_trades) >= 10:
        random_validation = validate_trades(
            random_trades,
            initial_capital=config["initial_capital"],
            spy_prices=spy_prices,
            strategy_name="Adaptive VIX-Band v1 (RANDOM baseline)",
            n_perms=min(config["n_perms"], 200),
        )
        random_validation.print_summary()

        ml_sharpe = validation.sharpe
        rand_sharpe = random_validation.sharpe
        if abs(rand_sharpe) > 0.01:
            incremental = (ml_sharpe - rand_sharpe) / abs(rand_sharpe) * 100
        else:
            incremental = float("inf") if ml_sharpe > 0 else 0

        ml_vs_random = {
            "ml_sharpe": round(ml_sharpe, 3),
            "random_sharpe": round(rand_sharpe, 3),
            "incremental_pct": round(incremental, 1),
        }
        fprint(f"\n  ML Sharpe: {ml_sharpe:.3f}  |  Random Sharpe: {rand_sharpe:.3f}")
        fprint(f"  ML adds {incremental:.0f}% incremental Sharpe over random")

    # Step 7: Compare vs proven baseline (VIX>20 only bull call spreads)
    fprint("\nStep 7: Baseline comparison — VIX>20 bull call spreads only...")
    baseline_config = config.copy()
    baseline_config["vix_band_1_max"] = 20.0  # Raise band 1 to 20 → no put credit spreads
    baseline_config["vix_band_2_max"] = 20.0  # Band 2 starts at 20
    baseline_config["vix_band_3_max"] = 999.0  # No deep OTM band

    baseline_trades = generate_adaptive_trades(
        predictions, prices, vix_series, baseline_config,
    )

    baseline_validation = None
    if len(baseline_trades) >= 10:
        baseline_validation = validate_trades(
            baseline_trades,
            initial_capital=config["initial_capital"],
            spy_prices=spy_prices,
            strategy_name="Baseline: VIX>20 Bull Call Spreads Only",
            n_perms=min(config["n_perms"], 200),
        )
        baseline_validation.print_summary()

        fprint(f"\n  ADAPTIVE vs BASELINE:")
        fprint(f"    Adaptive Sharpe:  {validation.sharpe:.3f}")
        fprint(f"    Baseline Sharpe:  {baseline_validation.sharpe:.3f}")
        fprint(f"    Adaptive trades:  {validation.n_trades}")
        fprint(f"    Baseline trades:  {baseline_validation.n_trades}")
        fprint(f"    Adaptive final:   ${validation.final_equity:,.2f}")
        fprint(f"    Baseline final:   ${baseline_validation.final_equity:,.2f}")
        fprint(f"    Adaptive MaxDD:   {validation.max_dd*100:.1f}%")
        fprint(f"    Baseline MaxDD:   {baseline_validation.max_dd*100:.1f}%")

    # Step 8: MLflow logging
    if MLFLOW_OK:
        fprint("\nStep 8: Logging to MLflow...")
        try:
            mlflow.set_experiment("adaptive-vix-band-strategy")
            with mlflow.start_run(run_name="adaptive_vix_band_v1"):
                # Core metrics
                mlflow.log_param("strategy", "adaptive_vix_band_v1")
                mlflow.log_param("initial_capital", config["initial_capital"])
                mlflow.log_param("start_date", config["start_date"])
                mlflow.log_param("end_date", config["end_date"])
                mlflow.log_param("vix_bands", "0-15-20-30+")

                mlflow.log_metric("sharpe", validation.sharpe)
                mlflow.log_metric("sortino", validation.sortino)
                mlflow.log_metric("cagr", validation.cagr)
                mlflow.log_metric("max_dd", validation.max_dd)
                mlflow.log_metric("win_rate", validation.win_rate)
                mlflow.log_metric("profit_factor", validation.profit_factor)
                mlflow.log_metric("n_trades", validation.n_trades)
                mlflow.log_metric("final_equity", validation.final_equity)
                mlflow.log_metric("gates_passed", validation.gates_passed)
                mlflow.log_metric("gates_total", validation.gates_total)

                if ml_vs_random:
                    mlflow.log_metric("random_sharpe", ml_vs_random["random_sharpe"])
                    mlflow.log_metric("ml_incremental_pct", ml_vs_random["incremental_pct"])

                if baseline_validation:
                    mlflow.log_metric("baseline_sharpe", baseline_validation.sharpe)
                    mlflow.log_metric("baseline_final_equity", baseline_validation.final_equity)

                # Per-band metrics
                for band_name, stats in band_analysis.items():
                    band_key = band_name.split("(")[0].strip().replace(" ", "_").lower()
                    mlflow.log_metric(f"band_{band_key}_trades", stats["n_trades"])
                    mlflow.log_metric(f"band_{band_key}_pnl", stats["total_pnl"])
                    mlflow.log_metric(f"band_{band_key}_wr", stats["win_rate"])

                # Save trade log
                output_dir = Path("/home/jupiter/Lvl3Quant/output/adaptive_vix_band")
                output_dir.mkdir(parents=True, exist_ok=True)

                results_data = {
                    "strategy": "adaptive_vix_band_v1",
                    "run_date": datetime.now().isoformat(),
                    "config": {k: v for k, v in config.items() if k != "lgbm_params"},
                    "metrics": validation.to_dict(),
                    "band_analysis": band_analysis,
                    "ml_vs_random": ml_vs_random,
                    "baseline_comparison": {
                        "baseline_sharpe": baseline_validation.sharpe if baseline_validation else None,
                        "baseline_final": baseline_validation.final_equity if baseline_validation else None,
                        "adaptive_sharpe": validation.sharpe,
                        "adaptive_final": validation.final_equity,
                    },
                    "vix_distribution": {
                        "pct_band0_cash": round(pct_band0, 1),
                        "pct_band1_pcs": round(pct_band1, 1),
                        "pct_band2_bcs": round(pct_band2, 1),
                        "pct_band3_deep": round(pct_band3, 1),
                    },
                }

                results_path = output_dir / "v1_results.json"
                with open(results_path, "w") as f:
                    json.dump(results_data, f, indent=2, default=str)

                mlflow.log_artifact(str(results_path))
                fprint("  MLflow run logged successfully")

        except Exception as e:
            fprint(f"  MLflow logging error: {e}")
    else:
        # Save results even without MLflow
        output_dir = Path("/home/jupiter/Lvl3Quant/output/adaptive_vix_band")
        output_dir.mkdir(parents=True, exist_ok=True)
        results_data = {
            "strategy": "adaptive_vix_band_v1",
            "run_date": datetime.now().isoformat(),
            "metrics": validation.to_dict(),
            "band_analysis": band_analysis,
            "ml_vs_random": ml_vs_random,
        }
        results_path = output_dir / "v1_results.json"
        with open(results_path, "w") as f:
            json.dump(results_data, f, indent=2, default=str)
        fprint(f"\n  Results saved to {results_path}")

    # Final summary
    fprint("\n" + "=" * 70)
    fprint("  FINAL SUMMARY")
    fprint("=" * 70)
    fprint(f"  Adaptive VIX-Band Strategy v1")
    fprint(f"  Sharpe: {validation.sharpe:.2f}  |  Sortino: {validation.sortino:.2f}")
    fprint(f"  CAGR: {validation.cagr*100:.1f}%  |  MaxDD: {validation.max_dd*100:.1f}%")
    fprint(f"  WR: {validation.win_rate*100:.1f}%  |  PF: {validation.profit_factor:.2f}")
    fprint(f"  Trades: {validation.n_trades}  |  Final equity: ${validation.final_equity:,.2f}")
    fprint(f"  Gates: {validation.gates_passed}/{validation.gates_total}")
    if baseline_validation:
        improvement = ""
        if baseline_validation.sharpe > 0.01:
            delta = ((validation.sharpe - baseline_validation.sharpe) /
                     baseline_validation.sharpe * 100)
            improvement = f" ({'+' if delta > 0 else ''}{delta:.0f}% vs baseline)"
        fprint(f"  vs Baseline (VIX>20 only): Sharpe {baseline_validation.sharpe:.2f}{improvement}")
    fprint("=" * 70)


if __name__ == "__main__":
    main()

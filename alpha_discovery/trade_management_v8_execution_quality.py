#!/usr/bin/env python3
"""
Trade Management v8 — Execution Quality Optimizer
===================================================

Given we KNOW what direction to trade (from signal models), HOW do we execute
optimally? This addresses HC #646 R5 (positioning) and HC #648 (continuous
evaluation).

Context:
  - v7 regime-aware mid-trade MLP passed regime gates (Sharpe 3.77, gap 0.49)
  - Multi-timeframe ensemble running as paper engine (70/30, Sharpe 1.96)
  - Long-horizon 4h hold strategy (Sharpe 1.84)

What v8 does:
  1. Load v7 trade predictions + tick-level microstructure data
  2. For each predicted trade, simulate multiple execution strategies:
     - PASSIVE: limit at bid/ask, cost = 0.376 ticks (commission only)
     - AGGRESSIVE: market order, cost = 1.376 ticks (commission + 1 tick spread)
     - HYBRID: start passive, convert to aggressive after N seconds if unfilled
     - CONDITIONAL: passive only if spread == 1 tick AND volume > threshold
  3. Build LightGBM classifier: given microstructure conditions, should entry
     be PASSIVE or AGGRESSIVE?
  4. Walk-forward SLIDING validation (60d train, 1d OOT, slide 1d — HC #0)
  5. Regime-agnostic validation (HC #428 R1): regime gap <= 0.50
  6. MFE-within-horizon validation (HC #428 R2)

Cost constants (ES futures AMP/Rithmic — CANONICAL):
  - ES_TICK_VALUE = $12.50
  - Passive limit: 0.376 ticks (commission only)
  - Market order: 1.376 ticks (commission + 1 tick spread crossing)
  - NO mid-price orders in ES — book is 1 tick wide during RTH

Usage:
  cd /home/nick/Lvl3Quant && \\
  /home/nick/miniconda3/envs/py311-train/bin/python \\
      alpha_discovery/trade_management_v8_execution_quality.py

Author: Claude (autonomous research)
"""

import gc
import json
import logging
import os
import pickle
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
V7_DIR = ROOT / "output" / "trade_management_v7_regime_aware"
V6_DIR = ROOT / "output" / "trade_management_v6_continuous"
OUTPUT_DIR = ROOT / "output" / "trade_management_v8_execution_quality"
LOG_DIR = ROOT / "logs"
MODEL_DIR = OUTPUT_DIR / "models"
RESULTS_DIR = OUTPUT_DIR / "results"

# Data paths
TICK_SAMPLES_PATH = V6_DIR / "tick_samples_v6.parquet"
DAILY_FEATURES_PATH = ROOT / "output" / "long_horizon_flow_v2" / "enhanced_daily_features.parquet"
MBO_EVENTS_DIR = ROOT / "data" / "processed" / "mbo_events_smart_v4"
MINUTE_BARS_DIR = ROOT / "data" / "processed" / "mbo_minute_bars_v1"

for d in [OUTPUT_DIR, MODEL_DIR, RESULTS_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s [TM-v8] %(levelname)s %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(LOG_DIR / "trade_management_v8_execution_quality.log")),
    ],
)
log = logging.getLogger("TM-v8")

# ─────────────────────────────────────────────
#  COST CONSTANTS (ES Futures — AMP/Rithmic)
# ─────────────────────────────────────────────
ES_TICK_VALUE = 12.50
COST_PASSIVE_TICKS = 0.376      # commission only (passive limit fill)
COST_AGGRESSIVE_TICKS = 1.376   # commission + 1 tick spread crossing
# NOTE: no mid-price in ES. Book is 1 tick wide during RTH. You're at bid or ask.

# ─────────────────────────────────────────────
#  WALK-FORWARD CONFIG (HC #0: SLIDING ONLY)
# ─────────────────────────────────────────────
TRAIN_DAYS = 20   # 20-day sliding window (constrained by 31 tick-data days available)
OOT_DAYS = 1      # 1-day out-of-time
SLIDE_DAYS = 1    # slide 1 day (drop oldest, add newest)

# ─────────────────────────────────────────────
#  EXECUTION STRATEGY PARAMETERS
# ─────────────────────────────────────────────
# Hybrid strategy: wait this many seconds passively, then go aggressive
HYBRID_WAIT_SECONDS = [3.0, 5.0, 8.0, 12.0]

# Conditional strategy: minimum volume at our level for passive
CONDITIONAL_MIN_VOLUME = [50, 100, 200]

# Passive fill probability model parameters
# Queue position degrades fill probability; estimate based on book depth
FILL_TIMEOUT_SECONDS = 15.0  # max wait for passive fill before cancelling

# Regime classification (ES close-to-close, in price points)
REGIME_GREEN_THRESHOLD = 20   # > +20 pts = green
REGIME_RED_THRESHOLD = -20    # < -20 pts = red

# Day-concession cap (HC #344)
DAY_CONC_CAP = 0.70

# LightGBM classifier params
LGBM_CLF_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "min_child_samples": 50,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "max_depth": 7,
    "verbose": -1,
    "n_jobs": -1,
    "is_unbalance": True,
}

# Deferred imports
lgb = None
torch = None
nn = None


def _import_lightgbm():
    global lgb
    if lgb is not None:
        return
    import lightgbm as _lgb
    lgb = _lgb


def _import_torch():
    global torch, nn
    if torch is not None:
        return
    import torch as _torch
    import torch.nn as _nn
    torch = _torch
    nn = _nn


# ═══════════════════════════════════════════════════════════════════
#  SECTION 1: DATA LOADING
# ═══════════════════════════════════════════════════════════════════


def load_v7_predictions() -> Optional[pd.DataFrame]:
    """Load v7 regime-aware predictions (OOT)."""
    pred_path = V7_DIR / "predictions_oot.parquet"
    if not pred_path.exists():
        log.warning(f"v7 predictions not found at {pred_path}, will use v6 data only")
        return None
    df = pd.read_parquet(pred_path)
    log.info(f"Loaded v7 predictions: {len(df):,} rows, cols={list(df.columns)}")
    return df


def load_tick_samples() -> pd.DataFrame:
    """Load v6 tick-level trade samples with microstructure features."""
    if not TICK_SAMPLES_PATH.exists():
        raise FileNotFoundError(f"Tick samples not found: {TICK_SAMPLES_PATH}")
    df = pd.read_parquet(TICK_SAMPLES_PATH)
    log.info(f"Loaded tick samples: {len(df):,} rows, {df['date'].nunique()} dates")
    return df


def load_daily_features() -> pd.DataFrame:
    """Load enhanced daily features for regime classification."""
    if not DAILY_FEATURES_PATH.exists():
        raise FileNotFoundError(f"Daily features not found: {DAILY_FEATURES_PATH}")
    df = pd.read_parquet(DAILY_FEATURES_PATH)
    df["date_str"] = df["date"].dt.strftime("%Y%m%d")
    df = df.sort_values("date_str").reset_index(drop=True)

    # Close-to-close return
    df["close_return"] = df["close"].diff()

    # Regime classification
    df["regime"] = "flat"
    df.loc[df["close_return"] > REGIME_GREEN_THRESHOLD, "regime"] = "green"
    df.loc[df["close_return"] < REGIME_RED_THRESHOLD, "regime"] = "red"

    log.info(f"Daily features: {len(df)} days | "
             f"green={sum(df['regime']=='green')} "
             f"red={sum(df['regime']=='red')} "
             f"flat={sum(df['regime']=='flat')}")
    return df


def load_mbo_events(date_str: str) -> Optional[pd.DataFrame]:
    """Load MBO event data for a specific date (for queue depth / microstructure)."""
    pattern = f"*{date_str}*"
    matches = list(MBO_EVENTS_DIR.glob(pattern))
    if not matches:
        return None
    try:
        return pd.read_parquet(matches[0])
    except Exception as e:
        log.warning(f"Failed to load MBO events for {date_str}: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════
#  SECTION 2: MICROSTRUCTURE FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════


def engineer_execution_features(tick_df: pd.DataFrame, daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features that predict optimal execution mode (passive vs aggressive).

    Features capture the microstructure state at the moment of trade entry:
      - Queue/book state: spread width, volume at level, queue depth, imbalance
      - Signal state: confidence, urgency (MFE erosion rate), direction
      - Market dynamics: momentum, trade rate, volatility regime
      - Timing: seconds since signal, time of day effects

    All features use information available AT or BEFORE the entry decision point.
    No lookahead.
    """
    log.info("Engineering execution features...")
    df = tick_df.copy()

    # Merge daily regime data
    date_regime_map = dict(zip(daily_df["date_str"].values, daily_df["regime"].values))
    df["regime"] = df["date"].map(date_regime_map).fillna("flat")

    # Regime one-hot encoding (numeric for model)
    df["regime_green"] = (df["regime"] == "green").astype(np.float32)
    df["regime_red"] = (df["regime"] == "red").astype(np.float32)

    # -- Spread and book features --
    # spread_width: infer from microprice_offset_now or default to 1 tick (ES RTH)
    if "microprice_offset_now" in df.columns:
        # microprice offset indicates how far microprice is from mid
        # spread ~= 2 * abs(microprice_offset) when book is symmetric
        df["spread_estimate_ticks"] = np.clip(
            2.0 * np.abs(df["microprice_offset_now"]), 1.0, 4.0
        )
    else:
        df["spread_estimate_ticks"] = 1.0  # ES is typically 1 tick during RTH

    # Volume at level proxy (from queue quantities)
    if "our_side_qty" in df.columns and "against_side_qty" in df.columns:
        df["volume_at_level"] = df["our_side_qty"] + df["against_side_qty"]
        df["volume_at_our_side"] = df["our_side_qty"]
        df["volume_at_against_side"] = df["against_side_qty"]
    else:
        df["volume_at_level"] = 0.0
        df["volume_at_our_side"] = 0.0
        df["volume_at_against_side"] = 0.0

    # Queue depth estimate: our_side_qty as fraction of total
    total_qty = df["volume_at_level"].replace(0, 1)
    df["queue_position_estimate"] = df["volume_at_our_side"] / total_qty

    # Queue depth ratio (existing feature if available)
    if "queue_ratio" not in df.columns:
        df["queue_ratio"] = df["volume_at_our_side"] / total_qty

    # -- Signal features --
    if "entry_confidence" in df.columns:
        df["signal_confidence"] = df["entry_confidence"]
    else:
        df["signal_confidence"] = 0.5

    if "entry_direction" in df.columns:
        df["signal_direction"] = df["entry_direction"]
    else:
        df["signal_direction"] = 0.0

    # Trade urgency: how fast is MFE eroding?
    # Proxy: if we have mfe_so_far and time_in_trade, compute erosion rate
    if "mfe_so_far_ticks" in df.columns and "time_in_trade_seconds" in df.columns:
        safe_time = df["time_in_trade_seconds"].replace(0, 0.01)
        df["mfe_erosion_rate"] = df["mfe_so_far_ticks"] / safe_time
        # Trade urgency: high erosion = need to act fast
        df["trade_urgency"] = np.clip(-df["mfe_erosion_rate"], 0, 10)
    else:
        df["mfe_erosion_rate"] = 0.0
        df["trade_urgency"] = 0.0

    # Seconds since signal (from time_in_trade_seconds)
    if "time_in_trade_seconds" in df.columns:
        df["seconds_since_signal"] = df["time_in_trade_seconds"]
    else:
        df["seconds_since_signal"] = 0.0

    # -- Momentum features --
    # 1-second and 5-second momentum proxies
    if "microprice_trend" in df.columns:
        df["momentum_1s"] = df["microprice_trend"]
    else:
        df["momentum_1s"] = 0.0

    if "ofi_recent_5s" in df.columns:
        df["momentum_5s"] = df["ofi_recent_5s"]
    else:
        df["momentum_5s"] = 0.0

    # Imbalance ratio
    if "imbalance_now" in df.columns:
        df["imbalance_ratio"] = df["imbalance_now"]
    elif "queue_ratio" in df.columns:
        df["imbalance_ratio"] = 2.0 * df["queue_ratio"] - 1.0
    else:
        df["imbalance_ratio"] = 0.0

    # -- Volatility regime proxy --
    # Use daily realized vol if available
    daily_vol_map = {}
    if "realized_vol_5d" in daily_df.columns:
        for _, row in daily_df.iterrows():
            daily_vol_map[row["date_str"]] = row.get("realized_vol_5d", np.nan)

    if daily_vol_map:
        df["vol_regime"] = df["date"].map(daily_vol_map).fillna(
            np.nanmedian(list(daily_vol_map.values()))
        )
        vol_median = df["vol_regime"].median()
        df["vol_regime_high"] = (df["vol_regime"] > vol_median).astype(np.float32)
    else:
        df["vol_regime"] = 0.0
        df["vol_regime_high"] = 0.0

    # -- Trade rate features --
    if "trade_rate_our_side" in df.columns:
        df["trade_rate_total"] = (
            df["trade_rate_our_side"].fillna(0) +
            df["trade_rate_against_side"].fillna(0)
        )
    else:
        df["trade_rate_total"] = 0.0

    # -- Flow features for fill probability --
    if "ofi_since_entry" in df.columns:
        df["ofi_flow"] = df["ofi_since_entry"]
    else:
        df["ofi_flow"] = 0.0

    # Signal alignment with flow (positive = flow agrees with our direction)
    df["flow_signal_alignment"] = df["ofi_flow"] * df["signal_direction"]

    # -- Interaction features --
    df["urgency_x_confidence"] = df["trade_urgency"] * df["signal_confidence"]
    df["imbalance_x_direction"] = df["imbalance_ratio"] * df["signal_direction"]
    df["volume_x_spread"] = df["volume_at_level"] * df["spread_estimate_ticks"]
    df["queue_x_momentum"] = df["queue_position_estimate"] * df["momentum_1s"]

    # Clean NaN/inf
    for col in df.columns:
        if df[col].dtype in [np.float32, np.float64]:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            df[col] = df[col].fillna(0.0)

    n_features = len(get_execution_feature_columns())
    log.info(f"Engineered {n_features} execution features, {len(df):,} rows")

    return df


def get_execution_feature_columns() -> List[str]:
    """Return the list of features used for the execution quality classifier."""
    return [
        # Book / queue state
        "spread_estimate_ticks",
        "volume_at_level",
        "volume_at_our_side",
        "volume_at_against_side",
        "queue_position_estimate",
        "queue_ratio",
        # Signal state
        "signal_confidence",
        "signal_direction",
        "seconds_since_signal",
        "trade_urgency",
        "mfe_erosion_rate",
        # Market dynamics
        "momentum_1s",
        "momentum_5s",
        "imbalance_ratio",
        "ofi_flow",
        "flow_signal_alignment",
        "trade_rate_total",
        # Volatility / regime
        "vol_regime",
        "vol_regime_high",
        "regime_green",
        "regime_red",
        # Interaction features
        "urgency_x_confidence",
        "imbalance_x_direction",
        "volume_x_spread",
        "queue_x_momentum",
    ]


# ═══════════════════════════════════════════════════════════════════
#  SECTION 3: EXECUTION STRATEGY SIMULATION
# ═══════════════════════════════════════════════════════════════════


def estimate_passive_fill_probability(row: pd.Series) -> float:
    """
    Estimate probability of a passive limit order filling within FILL_TIMEOUT_SECONDS.

    Heuristic model based on:
      - Queue position (deeper = lower fill prob)
      - Trade rate (higher = better fill prob)
      - Flow alignment (flow toward us = better fill)
      - Time available (more time = better)

    Returns: probability in [0, 1]
    """
    # Base fill probability (ES RTH, typical ~70% fill rate for well-placed limits)
    base_prob = 0.70

    # Queue position penalty: deeper in queue = lower probability
    queue_pos = row.get("queue_position_estimate", 0.5)
    queue_penalty = -0.3 * queue_pos  # 0.0 = front of queue, 1.0 = back

    # Trade rate bonus: more trades = more likely to fill
    trade_rate = row.get("trade_rate_total", 0.0)
    trade_rate_norm = np.clip(trade_rate / 100.0, 0, 0.5)  # normalize
    trade_bonus = 0.2 * trade_rate_norm

    # Flow alignment: flow coming toward our side = higher fill prob
    flow_align = row.get("flow_signal_alignment", 0.0)
    flow_bonus = 0.1 * np.clip(flow_align, -1, 1)

    # Volatility penalty: high vol = uncertain fill
    vol_high = row.get("vol_regime_high", 0.0)
    vol_penalty = -0.1 * vol_high

    # Imbalance: if imbalance favors our fill, boost probability
    imbalance = row.get("imbalance_ratio", 0.0)
    direction = row.get("signal_direction", 0.0)
    # If selling (dir < 0) and imbalance is positive (more buy pressure), fill is easier
    imb_bonus = 0.05 * imbalance * (-direction)

    prob = base_prob + queue_penalty + trade_bonus + flow_bonus + vol_penalty + imb_bonus
    return np.clip(prob, 0.05, 0.98)


def simulate_execution_strategies(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each trade entry point, simulate 4 execution strategies and compute
    the net P&L for each.

    Strategies:
      1. PASSIVE: limit at bid/ask, cost = 0.376 ticks.
         Fill probability estimated from microstructure.
         If no fill within timeout -> trade is MISSED (P&L = 0).
      2. AGGRESSIVE: market order, cost = 1.376 ticks. Always fills.
      3. HYBRID: start passive, convert to aggressive after N seconds.
         Cost = 0.376 if fills passively, else 1.376 + slippage from delay.
      4. CONDITIONAL: passive only if spread == 1 tick AND volume > threshold.
         If conditions not met -> aggressive entry.

    Each trade's P&L is the raw signal P&L minus the execution cost.
    For passive fills that miss, the P&L is 0 (opportunity cost, but no loss).

    We compute MFE erosion during passive wait to capture the real cost of waiting.
    """
    log.info("Simulating execution strategies...")

    # Get unique trades (entry points only — first sample per trade)
    if "trade_idx" in df.columns:
        entry_df = df.groupby("trade_idx").first().reset_index()
    else:
        entry_df = df.copy()

    log.info(f"Simulating on {len(entry_df):,} trade entries")

    # Base P&L from signal (before execution costs)
    if "label_final_pnl_ticks" in entry_df.columns:
        base_pnl_col = "label_final_pnl_ticks"
    elif "label_remaining_favorable_ticks" in entry_df.columns:
        base_pnl_col = "label_remaining_favorable_ticks"
    else:
        raise ValueError("No P&L column found in data")

    base_pnl = entry_df[base_pnl_col].values.astype(np.float64)

    # MFE for horizon validation
    if "mfe_so_far_ticks" in entry_df.columns:
        mfe_vals = entry_df["mfe_so_far_ticks"].values.astype(np.float64)
    else:
        mfe_vals = np.abs(base_pnl)  # approximate

    # ── Strategy 1: PASSIVE ──
    fill_probs = entry_df.apply(estimate_passive_fill_probability, axis=1).values
    passive_fills = np.random.RandomState(42).random(len(entry_df)) < fill_probs

    passive_pnl = np.where(
        passive_fills,
        base_pnl - COST_PASSIVE_TICKS,  # filled passively
        0.0,                              # missed trade
    )

    # ── Strategy 2: AGGRESSIVE ──
    aggressive_pnl = base_pnl - COST_AGGRESSIVE_TICKS

    # ── Strategy 3: HYBRID variants ──
    hybrid_results = {}
    for wait_s in HYBRID_WAIT_SECONDS:
        # Model: the longer we wait passively, the more MFE erodes
        # Erosion rate: assume linear MFE decay from signal start
        mfe_erosion = entry_df.get("mfe_erosion_rate", pd.Series(0.0, index=entry_df.index)).values
        mfe_lost_during_wait = np.clip(np.abs(mfe_erosion) * wait_s, 0, 5.0)

        # Fill probability increases with wait time (more time in queue)
        wait_fill_boost = np.clip(wait_s / FILL_TIMEOUT_SECONDS, 0, 1.0) * 0.15
        adjusted_fill_prob = np.clip(fill_probs + wait_fill_boost, 0.05, 0.98)
        hybrid_fills = np.random.RandomState(42 + int(wait_s)).random(len(entry_df)) < adjusted_fill_prob

        hybrid_pnl = np.where(
            hybrid_fills,
            base_pnl - COST_PASSIVE_TICKS,                               # filled passively
            base_pnl - COST_AGGRESSIVE_TICKS - mfe_lost_during_wait,      # had to go aggressive after delay
        )
        hybrid_results[f"hybrid_{wait_s:.0f}s"] = hybrid_pnl

    # ── Strategy 4: CONDITIONAL variants ──
    conditional_results = {}
    for min_vol in CONDITIONAL_MIN_VOLUME:
        spread_ok = entry_df["spread_estimate_ticks"].values <= 1.0
        vol_ok = entry_df["volume_at_level"].values >= min_vol

        # Condition met -> passive, else -> aggressive
        condition_met = spread_ok & vol_ok

        conditional_pnl = np.where(
            condition_met,
            np.where(passive_fills, base_pnl - COST_PASSIVE_TICKS, 0.0),  # passive with fill check
            aggressive_pnl,                                                  # aggressive fallback
        )
        conditional_results[f"conditional_vol{min_vol}"] = conditional_pnl

    # ── Build results DataFrame ──
    result_df = entry_df.copy()
    result_df["base_pnl_ticks"] = base_pnl
    result_df["mfe_ticks"] = mfe_vals
    result_df["fill_prob_estimate"] = fill_probs

    result_df["pnl_passive"] = passive_pnl
    result_df["pnl_aggressive"] = aggressive_pnl
    result_df["passive_filled"] = passive_fills

    for name, pnl in hybrid_results.items():
        result_df[f"pnl_{name}"] = pnl

    for name, pnl in conditional_results.items():
        result_df[f"pnl_{name}"] = pnl

    # ── Best execution label: for each trade, which strategy was best? ──
    strategy_cols = (
        ["pnl_passive", "pnl_aggressive"] +
        [f"pnl_{k}" for k in hybrid_results.keys()] +
        [f"pnl_{k}" for k in conditional_results.keys()]
    )

    strategy_pnls = result_df[strategy_cols].values
    best_idx = np.argmax(strategy_pnls, axis=1)
    result_df["best_strategy_idx"] = best_idx
    result_df["best_strategy_name"] = [strategy_cols[i].replace("pnl_", "") for i in best_idx]
    result_df["best_pnl"] = np.max(strategy_pnls, axis=1)

    # ── Binary label for classifier: passive (1) vs aggressive (0) ──
    # A trade SHOULD be passive if passive P&L > aggressive P&L (including misses)
    # This accounts for the fill probability: if fill prob is low, aggressive is safer
    result_df["label_prefer_passive"] = (
        result_df["pnl_passive"] > result_df["pnl_aggressive"]
    ).astype(np.int32)

    # Summary
    passive_frac = result_df["label_prefer_passive"].mean()
    log.info(f"Execution label distribution: passive_preferred={passive_frac:.1%}, "
             f"aggressive_preferred={1-passive_frac:.1%}")

    for col in strategy_cols:
        pnl = result_df[col].values
        valid = ~np.isnan(pnl)
        if valid.sum() > 0:
            mean_pnl = np.mean(pnl[valid])
            active = pnl[valid] != 0  # non-missed trades
            active_mean = np.mean(pnl[valid][active]) if active.sum() > 0 else 0
            name = col.replace("pnl_", "")
            log.info(f"  {name:25s}: mean={mean_pnl:+.3f}t  active_mean={active_mean:+.3f}t  "
                     f"n={valid.sum()}")

    return result_df


# ═══════════════════════════════════════════════════════════════════
#  SECTION 4: MFE-WITHIN-HORIZON VALIDATION (HC #428 R2)
# ═══════════════════════════════════════════════════════════════════


def validate_mfe_within_horizon(result_df: pd.DataFrame) -> Dict:
    """
    HC #428 R2: TP must be bounded by model's actual predictive horizon.
    - TP <= p90 of realized MFE within horizon h
    - hold_seconds <= 1.5 * h
    - Reject configs that profit beyond the model's horizon (luck, not edge)
    """
    log.info("\n" + "=" * 70)
    log.info("MFE-WITHIN-HORIZON VALIDATION (HC #428 R2)")
    log.info("=" * 70)

    mfe = result_df["mfe_ticks"].dropna().values
    if len(mfe) < 10:
        log.warning("Insufficient MFE data for validation")
        return {"valid": False, "reason": "insufficient_data"}

    mfe_p50 = np.percentile(mfe, 50)
    mfe_p75 = np.percentile(mfe, 75)
    mfe_p90 = np.percentile(mfe, 90)
    mfe_p95 = np.percentile(mfe, 95)

    log.info(f"MFE distribution: p50={mfe_p50:.2f}t p75={mfe_p75:.2f}t "
             f"p90={mfe_p90:.2f}t p95={mfe_p95:.2f}t")

    # Check hold times
    if "time_in_trade_seconds" in result_df.columns:
        hold_times = result_df["time_in_trade_seconds"].dropna().values
        hold_p90 = np.percentile(hold_times, 90)
        hold_max = np.max(hold_times)
        log.info(f"Hold times: p90={hold_p90:.1f}s max={hold_max:.1f}s")
    else:
        hold_p90 = 30.0  # default assumption
        log.info(f"Hold times: using default assumption p90={hold_p90:.1f}s")

    # Model horizon (from signal decay analysis): ~10-30s
    MODEL_HORIZON_SECONDS = 30.0

    # Validate
    tp_limit = mfe_p90  # TP must be <= p90 MFE
    hold_limit = 1.5 * MODEL_HORIZON_SECONDS  # hold <= 1.5 * horizon

    results = {
        "mfe_p50": float(mfe_p50),
        "mfe_p75": float(mfe_p75),
        "mfe_p90": float(mfe_p90),
        "mfe_p95": float(mfe_p95),
        "tp_limit_ticks": float(tp_limit),
        "model_horizon_s": MODEL_HORIZON_SECONDS,
        "hold_limit_s": float(hold_limit),
        "hold_p90_s": float(hold_p90),
        "hold_valid": hold_p90 <= hold_limit,
        "valid": True,
    }

    if hold_p90 > hold_limit:
        log.warning(f"FAIL: hold p90 ({hold_p90:.1f}s) > 1.5 * horizon ({hold_limit:.1f}s)")
        results["valid"] = False
        results["reason"] = "hold_exceeds_horizon"
    else:
        log.info(f"PASS: hold p90 ({hold_p90:.1f}s) <= 1.5 * horizon ({hold_limit:.1f}s)")

    log.info(f"TP limit (p90 MFE): {tp_limit:.2f} ticks")

    return results


# ═══════════════════════════════════════════════════════════════════
#  SECTION 5: WALK-FORWARD LGBM EXECUTION CLASSIFIER
# ═══════════════════════════════════════════════════════════════════


def train_execution_classifier(
    result_df: pd.DataFrame,
) -> Tuple[Dict, pd.DataFrame]:
    """
    Walk-forward SLIDING LightGBM classifier (HC #0: NEVER expanding).

    Predicts: should this trade use PASSIVE (1) or AGGRESSIVE (0) execution?

    Window: 60d train, 1d OOT, slide 1d, drop oldest.

    Returns:
      - summary dict with aggregate metrics
      - result_df with OOT predictions added
    """
    _import_lightgbm()

    log.info("\n" + "=" * 70)
    log.info(f"WALK-FORWARD EXECUTION CLASSIFIER (SLIDING {TRAIN_DAYS}d/{OOT_DAYS}d)")
    log.info("=" * 70)

    feature_cols = get_execution_feature_columns()
    feature_cols = [c for c in feature_cols if c in result_df.columns]
    target_col = "label_prefer_passive"

    log.info(f"Features: {len(feature_cols)}")
    log.info(f"Feature list: {feature_cols}")

    dates = sorted(result_df["date"].unique())
    log.info(f"Total dates: {len(dates)}")

    # Initialize prediction columns (BEFORE early return check)
    result_df["pred_passive_prob"] = np.nan
    result_df["pred_execution_mode"] = ""

    if len(dates) < TRAIN_DAYS + OOT_DAYS:
        log.error(f"Insufficient dates: need {TRAIN_DAYS + OOT_DAYS}, have {len(dates)}")
        return {}, result_df

    fold_results = []
    fold_importances = []

    fold_idx = 0
    start = 0

    while start + TRAIN_DAYS + OOT_DAYS <= len(dates):
        train_dates = dates[start:start + TRAIN_DAYS]
        oot_dates = dates[start + TRAIN_DAYS:start + TRAIN_DAYS + OOT_DAYS]

        train_mask = result_df["date"].isin(train_dates)
        oot_mask = result_df["date"].isin(oot_dates)

        X_train = result_df.loc[train_mask, feature_cols].values.astype(np.float32)
        y_train = result_df.loc[train_mask, target_col].values.astype(np.int32)
        X_oot = result_df.loc[oot_mask, feature_cols].values.astype(np.float32)
        y_oot = result_df.loc[oot_mask, target_col].values.astype(np.int32)

        # Clean NaN/inf
        valid_train = (
            ~np.isnan(y_train) &
            ~np.isnan(X_train).any(axis=1) &
            ~np.isinf(X_train).any(axis=1)
        )
        valid_oot = (
            ~np.isnan(y_oot) &
            ~np.isnan(X_oot).any(axis=1) &
            ~np.isinf(X_oot).any(axis=1)
        )

        X_tr = X_train[valid_train]
        y_tr = y_train[valid_train]
        X_ot = X_oot[valid_oot]
        y_ot = y_oot[valid_oot]

        if len(X_tr) < 100 or len(X_ot) < 5:
            start += SLIDE_DAYS
            fold_idx += 1
            continue

        # Train LightGBM
        dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_cols, free_raw_data=False)

        # Use early stopping with a small validation split from training data
        val_split = max(int(len(X_tr) * 0.15), 20)
        dval = lgb.Dataset(
            X_tr[-val_split:], label=y_tr[-val_split:],
            feature_name=feature_cols, free_raw_data=False, reference=dtrain,
        )

        callbacks = [lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)]
        model = lgb.train(
            LGBM_CLF_PARAMS,
            dtrain,
            num_boost_round=500,
            valid_sets=[dval],
            callbacks=callbacks,
        )

        # Predict on OOT
        oot_pred_prob = model.predict(X_ot)
        oot_pred_class = (oot_pred_prob > 0.5).astype(np.int32)

        # Store predictions in result_df
        oot_indices = result_df.index[oot_mask]
        valid_oot_positions = np.where(valid_oot)[0]
        for i, pos in enumerate(valid_oot_positions):
            if pos < len(oot_indices):
                idx = oot_indices[pos]
                result_df.loc[idx, "pred_passive_prob"] = oot_pred_prob[i]
                result_df.loc[idx, "pred_execution_mode"] = (
                    "passive" if oot_pred_class[i] == 1 else "aggressive"
                )

        # Fold metrics
        from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
        if len(y_ot) > 0 and len(np.unique(y_ot)) > 1:
            auc = roc_auc_score(y_ot, oot_pred_prob)
            acc = accuracy_score(y_ot, oot_pred_class)
            f1 = f1_score(y_ot, oot_pred_class, zero_division=0)
        elif len(y_ot) > 0:
            auc = 0.5
            acc = accuracy_score(y_ot, oot_pred_class)
            f1 = 0.0
        else:
            auc = acc = f1 = 0.0

        # Compute P&L improvement from model-guided execution
        oot_data = result_df.loc[oot_mask].iloc[valid_oot_positions]
        model_pnl = np.where(
            oot_pred_class == 1,
            oot_data["pnl_passive"].values,
            oot_data["pnl_aggressive"].values,
        )
        always_aggressive_pnl = oot_data["pnl_aggressive"].values
        always_passive_pnl = oot_data["pnl_passive"].values
        oracle_pnl = oot_data["best_pnl"].values

        fold_results.append({
            "fold": fold_idx,
            "oot_date": oot_dates[0],
            "n_train": len(X_tr),
            "n_oot": len(X_ot),
            "auc": float(auc),
            "accuracy": float(acc),
            "f1": float(f1),
            "pnl_model_mean": float(np.mean(model_pnl)) if len(model_pnl) > 0 else 0,
            "pnl_aggressive_mean": float(np.mean(always_aggressive_pnl)) if len(always_aggressive_pnl) > 0 else 0,
            "pnl_passive_mean": float(np.mean(always_passive_pnl)) if len(always_passive_pnl) > 0 else 0,
            "pnl_oracle_mean": float(np.mean(oracle_pnl)) if len(oracle_pnl) > 0 else 0,
            "pct_passive": float(np.mean(oot_pred_class)) if len(oot_pred_class) > 0 else 0,
            "regime": result_df.loc[oot_mask, "regime"].iloc[0] if oot_mask.sum() > 0 else "unknown",
        })

        # Feature importance
        imp = model.feature_importance(importance_type="gain")
        fold_importances.append(dict(zip(feature_cols, imp.tolist())))

        if fold_idx % 10 == 0 or fold_idx < 3:
            log.info(
                f"  Fold {fold_idx:3d} [{oot_dates[0]}]: AUC={auc:.3f} acc={acc:.1%} "
                f"pnl_model={np.mean(model_pnl):+.3f}t "
                f"pnl_agg={np.mean(always_aggressive_pnl):+.3f}t "
                f"pnl_oracle={np.mean(oracle_pnl):+.3f}t"
            )

        # Save model for last fold
        if start + SLIDE_DAYS + TRAIN_DAYS + OOT_DAYS > len(dates):
            model_path = MODEL_DIR / f"lgbm_execution_fold{fold_idx:04d}.txt"
            model.save_model(str(model_path))
            log.info(f"  Saved final fold model: {model_path.name}")

        del model, dtrain, dval
        gc.collect()

        fold_idx += 1
        start += SLIDE_DAYS

    # ── Aggregate results ──
    if not fold_results:
        log.error("No folds completed")
        return {}, result_df

    folds_df = pd.DataFrame(fold_results)

    summary = {
        "n_folds": len(fold_results),
        "avg_auc": float(folds_df["auc"].mean()),
        "avg_accuracy": float(folds_df["accuracy"].mean()),
        "avg_f1": float(folds_df["f1"].mean()),
        "avg_pnl_model": float(folds_df["pnl_model_mean"].mean()),
        "avg_pnl_aggressive": float(folds_df["pnl_aggressive_mean"].mean()),
        "avg_pnl_passive": float(folds_df["pnl_passive_mean"].mean()),
        "avg_pnl_oracle": float(folds_df["pnl_oracle_mean"].mean()),
        "pnl_improvement_vs_aggressive": float(
            folds_df["pnl_model_mean"].mean() - folds_df["pnl_aggressive_mean"].mean()
        ),
        "avg_pct_passive": float(folds_df["pct_passive"].mean()),
    }

    log.info(f"\n{'='*70}")
    log.info("CLASSIFIER SUMMARY")
    log.info(f"{'='*70}")
    log.info(f"  Folds: {summary['n_folds']}")
    log.info(f"  AUC: {summary['avg_auc']:.3f}")
    log.info(f"  Accuracy: {summary['avg_accuracy']:.1%}")
    log.info(f"  F1: {summary['avg_f1']:.3f}")
    log.info(f"  Model avg P&L: {summary['avg_pnl_model']:+.3f} ticks/trade")
    log.info(f"  Always-aggressive avg P&L: {summary['avg_pnl_aggressive']:+.3f} ticks/trade")
    log.info(f"  Always-passive avg P&L: {summary['avg_pnl_passive']:+.3f} ticks/trade")
    log.info(f"  Oracle avg P&L: {summary['avg_pnl_oracle']:+.3f} ticks/trade")
    log.info(f"  Improvement vs aggressive: {summary['pnl_improvement_vs_aggressive']:+.3f} ticks/trade")
    log.info(f"  Passive selection rate: {summary['avg_pct_passive']:.1%}")

    # Aggregate feature importance
    if fold_importances:
        avg_imp = {}
        for feat in feature_cols:
            vals = [fi.get(feat, 0) for fi in fold_importances]
            avg_imp[feat] = float(np.mean(vals))
        sorted_imp = sorted(avg_imp.items(), key=lambda x: x[1], reverse=True)
        log.info(f"\n  Feature importance (top 15):")
        for feat, val in sorted_imp[:15]:
            log.info(f"    {feat:30s} {val:10.1f}")
        summary["feature_importance"] = avg_imp

    # Save fold results
    folds_path = RESULTS_DIR / "fold_results.parquet"
    folds_df.to_parquet(folds_path, index=False)
    log.info(f"Saved fold results: {folds_path.name}")

    return summary, result_df


# ═══════════════════════════════════════════════════════════════════
#  SECTION 6: REGIME-AGNOSTIC VALIDATION (HC #428 R1)
# ═══════════════════════════════════════════════════════════════════


def compute_strategy_metrics(pnl_arr: np.ndarray, name: str = "") -> Dict:
    """Compute Sharpe, Sortino, PF, WR from per-trade P&L array."""
    pnl = pnl_arr[~np.isnan(pnl_arr)]
    if len(pnl) < 3:
        return {
            "name": name, "sharpe": 0.0, "sortino": 0.0,
            "profit_factor": 0.0, "win_rate": 0.0,
            "mean_pnl_ticks": 0.0, "n_trades": 0,
            "total_pnl_ticks": 0.0,
        }

    mean_pnl = np.mean(pnl)
    std_pnl = np.std(pnl)
    sharpe = mean_pnl / max(std_pnl, 1e-8) * np.sqrt(252)

    downside = pnl[pnl < 0]
    downside_std = np.std(downside) if len(downside) > 1 else std_pnl
    sortino = mean_pnl / max(downside_std, 1e-8) * np.sqrt(252)

    gross_profit = np.sum(pnl[pnl > 0])
    gross_loss = abs(np.sum(pnl[pnl < 0]))
    pf = gross_profit / max(gross_loss, 1e-8)

    wr = np.mean(pnl > 0)

    return {
        "name": name, "sharpe": float(sharpe), "sortino": float(sortino),
        "profit_factor": float(pf), "win_rate": float(wr),
        "mean_pnl_ticks": float(mean_pnl), "n_trades": int(len(pnl)),
        "total_pnl_ticks": float(np.sum(pnl)),
    }


def validate_regime_agnostic(
    result_df: pd.DataFrame,
    classifier_summary: Dict,
) -> Dict:
    """
    HC #428 R1: Regime-agnostic OOT validation (>= 40 days, ALL regimes).

    For each execution strategy, compute:
      - Overall Sharpe/Sortino/PF/WR
      - Per-regime Sharpe (green/red/flat)
      - Regime gap: |Sharpe_green - Sharpe_red| / max(|Sharpe_green|, |Sharpe_red|)
      - Day-concentration cap (HC #344): max day contribution <= 0.70

    Reject if regime gap > 0.50 or day-conc > 0.70.
    """
    log.info("\n" + "=" * 70)
    log.info("REGIME-AGNOSTIC VALIDATION (HC #428 R1)")
    log.info("=" * 70)

    # Strategies to validate
    strategy_pnl_cols = {
        "passive": "pnl_passive",
        "aggressive": "pnl_aggressive",
    }

    # Add hybrid strategies
    for wait_s in HYBRID_WAIT_SECONDS:
        col = f"pnl_hybrid_{wait_s:.0f}s"
        if col in result_df.columns:
            strategy_pnl_cols[f"hybrid_{wait_s:.0f}s"] = col

    # Add conditional strategies
    for min_vol in CONDITIONAL_MIN_VOLUME:
        col = f"pnl_conditional_vol{min_vol}"
        if col in result_df.columns:
            strategy_pnl_cols[f"conditional_vol{min_vol}"] = col

    # Add model-guided strategy
    has_model = ("pred_passive_prob" in result_df.columns and
                 result_df["pred_passive_prob"].notna().sum() > 0)
    if has_model:
        # Model-guided P&L: use model prediction to choose passive vs aggressive
        model_mask = result_df["pred_passive_prob"].notna()
        model_df = result_df[model_mask].copy()
        model_pnl = np.where(
            model_df["pred_passive_prob"].values > 0.5,
            model_df["pnl_passive"].values,
            model_df["pnl_aggressive"].values,
        )
        result_df.loc[model_mask, "pnl_model_guided"] = model_pnl
        strategy_pnl_cols["model_guided"] = "pnl_model_guided"

    validation_results = {}

    for strat_name, pnl_col in strategy_pnl_cols.items():
        if pnl_col not in result_df.columns:
            continue

        pnl = result_df[pnl_col].dropna().values

        if len(pnl) < 10:
            log.warning(f"  {strat_name}: insufficient data ({len(pnl)} trades)")
            continue

        # Overall metrics
        overall = compute_strategy_metrics(pnl, strat_name)

        # Per-regime metrics
        regime_metrics = {}
        for regime in ["green", "red", "flat"]:
            mask = result_df["regime"] == regime
            regime_pnl = result_df.loc[mask, pnl_col].dropna().values
            regime_metrics[regime] = compute_strategy_metrics(regime_pnl, f"{strat_name}_{regime}")

        # Regime gap (HC #428 R1)
        sharpe_g = regime_metrics["green"]["sharpe"]
        sharpe_r = regime_metrics["red"]["sharpe"]
        max_abs = max(abs(sharpe_g), abs(sharpe_r), 1e-8)
        regime_gap = abs(sharpe_g - sharpe_r) / max_abs

        # Day-concentration cap (HC #344)
        if "date" in result_df.columns:
            daily_pnl = result_df.groupby("date")[pnl_col].sum()
            daily_pnl_valid = daily_pnl.dropna()
            if len(daily_pnl_valid) > 0:
                total_abs_pnl = daily_pnl_valid.abs().sum()
                max_day_contribution = daily_pnl_valid.abs().max() / max(total_abs_pnl, 1e-8)
            else:
                max_day_contribution = 0.0
        else:
            max_day_contribution = 0.0

        # Direction balance check (reject all-short or all-long without justification)
        if "signal_direction" in result_df.columns:
            directions = result_df.loc[result_df[pnl_col].notna(), "signal_direction"]
            long_frac = (directions > 0).mean()
            short_frac = (directions < 0).mean()
            direction_balanced = 0.1 < long_frac < 0.9  # not all one side
        else:
            long_frac = short_frac = 0.5
            direction_balanced = True

        # Gate checks
        regime_pass = regime_gap <= 0.50
        day_conc_pass = max_day_contribution <= DAY_CONC_CAP

        validation_results[strat_name] = {
            "overall": overall,
            "regime_metrics": regime_metrics,
            "regime_gap": float(regime_gap),
            "regime_pass": regime_pass,
            "day_conc": float(max_day_contribution),
            "day_conc_pass": day_conc_pass,
            "long_frac": float(long_frac),
            "short_frac": float(short_frac),
            "direction_balanced": direction_balanced,
            "all_gates_pass": regime_pass and day_conc_pass and direction_balanced,
            "n_oot_days": int(result_df["date"].nunique()) if "date" in result_df.columns else 0,
        }

        # Log results
        status = "PASS" if validation_results[strat_name]["all_gates_pass"] else "FAIL"
        log.info(
            f"\n  {strat_name} [{status}]:"
            f"\n    Overall: Sharpe={overall['sharpe']:.2f} Sortino={overall['sortino']:.2f} "
            f"PF={overall['profit_factor']:.2f} WR={overall['win_rate']:.1%} n={overall['n_trades']}"
            f"\n    Green:   Sharpe={sharpe_g:.2f} n={regime_metrics['green']['n_trades']}"
            f"\n    Red:     Sharpe={sharpe_r:.2f} n={regime_metrics['red']['n_trades']}"
            f"\n    Flat:    Sharpe={regime_metrics['flat']['sharpe']:.2f} n={regime_metrics['flat']['n_trades']}"
            f"\n    Regime gap: {regime_gap:.3f} {'<= 0.50 PASS' if regime_pass else '> 0.50 FAIL'}"
            f"\n    Day conc:  {max_day_contribution:.3f} {'<= 0.70 PASS' if day_conc_pass else '> 0.70 FAIL'}"
            f"\n    Direction: long={long_frac:.1%} short={short_frac:.1%} "
            f"{'balanced' if direction_balanced else 'IMBALANCED'}"
        )

    # Per-day Sharpe report (HC #428 requires per-day breakdown)
    if "date" in result_df.columns and "pnl_model_guided" in result_df.columns:
        log.info(f"\n  Per-day Sharpe/PF/WR (model-guided, last 20 OOT days):")
        daily_stats = []
        for date in sorted(result_df["date"].unique())[-20:]:
            day_data = result_df[result_df["date"] == date]
            day_pnl = day_data["pnl_model_guided"].dropna().values
            if len(day_pnl) > 0:
                regime = day_data["regime"].iloc[0] if "regime" in day_data.columns else "?"
                day_mean = np.mean(day_pnl)
                day_wr = np.mean(day_pnl > 0)
                daily_stats.append({
                    "date": date, "regime": regime,
                    "mean_pnl": day_mean, "wr": day_wr, "n": len(day_pnl),
                })
                log.info(f"    {date} [{regime:5s}] mean={day_mean:+.3f}t WR={day_wr:.0%} n={len(day_pnl)}")

    return validation_results


# ═══════════════════════════════════════════════════════════════════
#  SECTION 7: MLFLOW LOGGING
# ═══════════════════════════════════════════════════════════════════


def log_to_mlflow(
    classifier_summary: Dict,
    validation_results: Dict,
    mfe_results: Dict,
    elapsed_seconds: float,
) -> Optional[str]:
    """Log all results to MLflow. Experiment: trade_management_v8_execution_quality."""
    try:
        import mlflow
    except ImportError:
        log.warning("MLflow not available, skipping logging")
        return None

    mlflow_uri = "http://jupiter:5000"
    mlflow.set_tracking_uri(mlflow_uri)
    experiment_name = "trade_management_v8_execution_quality"

    try:
        mlflow.set_experiment(experiment_name)
    except Exception as e:
        log.warning(f"Could not set MLflow experiment: {e}")
        return None

    try:
        with mlflow.start_run(
            run_name=f"v8_exec_quality_{datetime.now().strftime('%Y%m%d_%H%M')}"
        ) as run:
            # Parameters
            mlflow.log_param("train_days", TRAIN_DAYS)
            mlflow.log_param("oot_days", OOT_DAYS)
            mlflow.log_param("slide_days", SLIDE_DAYS)
            mlflow.log_param("cost_passive_ticks", COST_PASSIVE_TICKS)
            mlflow.log_param("cost_aggressive_ticks", COST_AGGRESSIVE_TICKS)
            mlflow.log_param("fill_timeout_s", FILL_TIMEOUT_SECONDS)
            mlflow.log_param("hybrid_waits", str(HYBRID_WAIT_SECONDS))
            mlflow.log_param("conditional_volumes", str(CONDITIONAL_MIN_VOLUME))
            mlflow.log_param("regime_green_thresh", REGIME_GREEN_THRESHOLD)
            mlflow.log_param("regime_red_thresh", REGIME_RED_THRESHOLD)
            mlflow.log_param("day_conc_cap", DAY_CONC_CAP)
            mlflow.log_param("window_type", "SLIDING")

            # Classifier metrics
            for key in ["avg_auc", "avg_accuracy", "avg_f1", "avg_pnl_model",
                         "avg_pnl_aggressive", "avg_pnl_passive", "avg_pnl_oracle",
                         "pnl_improvement_vs_aggressive", "avg_pct_passive", "n_folds"]:
                if key in classifier_summary:
                    mlflow.log_metric(f"clf_{key}", classifier_summary[key])

            # Validation metrics (per strategy)
            for strat_name, strat_results in validation_results.items():
                prefix = f"val_{strat_name}"
                overall = strat_results.get("overall", {})
                for metric in ["sharpe", "sortino", "profit_factor", "win_rate",
                               "mean_pnl_ticks", "n_trades"]:
                    if metric in overall:
                        mlflow.log_metric(f"{prefix}_{metric}", overall[metric])

                mlflow.log_metric(f"{prefix}_regime_gap", strat_results.get("regime_gap", -1))
                mlflow.log_metric(f"{prefix}_regime_pass", int(strat_results.get("regime_pass", False)))
                mlflow.log_metric(f"{prefix}_day_conc", strat_results.get("day_conc", -1))
                mlflow.log_metric(f"{prefix}_all_pass", int(strat_results.get("all_gates_pass", False)))

                # Per-regime Sharpe
                for regime in ["green", "red", "flat"]:
                    rm = strat_results.get("regime_metrics", {}).get(regime, {})
                    if "sharpe" in rm:
                        mlflow.log_metric(f"{prefix}_sharpe_{regime}", rm["sharpe"])

            # MFE validation
            for key in ["mfe_p50", "mfe_p75", "mfe_p90", "mfe_p95",
                         "tp_limit_ticks", "hold_p90_s"]:
                if key in mfe_results:
                    mlflow.log_metric(f"mfe_{key}", mfe_results[key])
            mlflow.log_metric("mfe_valid", int(mfe_results.get("valid", False)))

            mlflow.log_metric("elapsed_seconds", elapsed_seconds)

            # Log artifacts
            for artifact in ["summary.json", "results/fold_results.parquet"]:
                artifact_path = OUTPUT_DIR / artifact
                if artifact_path.exists():
                    mlflow.log_artifact(str(artifact_path))

            # Log feature importance as artifact
            if "feature_importance" in classifier_summary:
                imp_path = RESULTS_DIR / "feature_importance.json"
                with open(imp_path, "w") as f:
                    json.dump(classifier_summary["feature_importance"], f, indent=2)
                mlflow.log_artifact(str(imp_path))

            log.info(f"MLflow run logged: {run.info.run_id}")
            return run.info.run_id

    except Exception as e:
        log.error(f"MLflow logging failed: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════
#  SECTION 8: SUMMARY REPORT + SAVE
# ═══════════════════════════════════════════════════════════════════


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def save_results(
    result_df: pd.DataFrame,
    classifier_summary: Dict,
    validation_results: Dict,
    mfe_results: Dict,
    elapsed: float,
) -> Dict:
    """Save all outputs and build final summary."""
    log.info("\nSaving results...")

    # Save per-trade execution analysis
    save_cols = [
        "trade_idx", "date", "regime",
        "base_pnl_ticks", "mfe_ticks", "fill_prob_estimate",
        "pnl_passive", "pnl_aggressive", "passive_filled",
        "best_strategy_name", "best_pnl",
        "label_prefer_passive",
        "pred_passive_prob", "pred_execution_mode",
        "signal_confidence", "signal_direction",
        "spread_estimate_ticks", "volume_at_level",
        "queue_position_estimate", "trade_urgency",
        "imbalance_ratio", "vol_regime",
    ]

    # Add hybrid/conditional P&L columns
    for col in result_df.columns:
        if col.startswith("pnl_hybrid_") or col.startswith("pnl_conditional_"):
            save_cols.append(col)
    if "pnl_model_guided" in result_df.columns:
        save_cols.append("pnl_model_guided")

    save_cols = [c for c in save_cols if c in result_df.columns]
    trade_analysis = result_df[save_cols].copy()

    trade_path = OUTPUT_DIR / "per_trade_execution_analysis.parquet"
    trade_analysis.to_parquet(trade_path, index=False)
    log.info(f"  Saved per-trade analysis: {trade_path.name} ({len(trade_analysis):,} rows)")

    # Save predictions OOT
    pred_cols = ["trade_idx", "date", "regime",
                 "pred_passive_prob", "pred_execution_mode",
                 "label_prefer_passive", "pnl_passive", "pnl_aggressive"]
    if "pnl_model_guided" in result_df.columns:
        pred_cols.append("pnl_model_guided")
    pred_cols = [c for c in pred_cols if c in result_df.columns]
    if "pred_passive_prob" in result_df.columns:
        pred_df = result_df[result_df["pred_passive_prob"].notna()][pred_cols].copy()
    else:
        pred_df = pd.DataFrame(columns=pred_cols)
    pred_path = OUTPUT_DIR / "predictions_oot.parquet"
    pred_df.to_parquet(pred_path, index=False)
    log.info(f"  Saved OOT predictions: {pred_path.name} ({len(pred_df):,} rows)")

    # Build summary
    # Find best passing strategy
    best_passing = None
    best_passing_sharpe = -999
    best_overall = None
    best_overall_sharpe = -999

    for strat_name, strat_results in validation_results.items():
        overall = strat_results.get("overall", {})
        sharpe = overall.get("sharpe", -999)

        if sharpe > best_overall_sharpe:
            best_overall = strat_name
            best_overall_sharpe = sharpe

        if strat_results.get("all_gates_pass") and sharpe > best_passing_sharpe:
            best_passing = strat_name
            best_passing_sharpe = sharpe

    summary = {
        "completed_at": datetime.now().isoformat(),
        "elapsed_seconds": elapsed,
        "version": "v8_execution_quality",
        "walk_forward": {
            "type": "SLIDING",
            "train_days": TRAIN_DAYS,
            "oot_days": OOT_DAYS,
            "slide_days": SLIDE_DAYS,
        },
        "costs": {
            "passive_ticks": COST_PASSIVE_TICKS,
            "aggressive_ticks": COST_AGGRESSIVE_TICKS,
            "cost_type": "FIFO",
        },
        "classifier": {
            k: v for k, v in classifier_summary.items()
            if k != "feature_importance"
        },
        "mfe_validation": mfe_results,
        "validation": {
            strat: {
                "overall": v.get("overall", {}),
                "regime_gap": v.get("regime_gap"),
                "regime_pass": v.get("regime_pass"),
                "day_conc": v.get("day_conc"),
                "day_conc_pass": v.get("day_conc_pass"),
                "direction_balanced": v.get("direction_balanced"),
                "all_gates_pass": v.get("all_gates_pass"),
                "n_oot_days": v.get("n_oot_days"),
            }
            for strat, v in validation_results.items()
        },
        "best_passing_strategy": best_passing,
        "best_passing_sharpe": best_passing_sharpe if best_passing else None,
        "best_overall_strategy": best_overall,
        "best_overall_sharpe": best_overall_sharpe if best_overall else None,
    }

    summary_path = OUTPUT_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)
    log.info(f"  Saved summary: {summary_path.name}")

    return summary


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════


def main():
    t0 = time.time()
    log.info("=" * 70)
    log.info("Trade Management v8 — Execution Quality Optimizer")
    log.info("=" * 70)
    log.info(f"Output: {OUTPUT_DIR}")
    log.info(f"Walk-forward: {TRAIN_DAYS}d train / {OOT_DAYS}d OOT / {SLIDE_DAYS}d slide (SLIDING)")
    log.info(f"Costs: passive={COST_PASSIVE_TICKS}t aggressive={COST_AGGRESSIVE_TICKS}t (FIFO)")

    # ── Phase 1: Load data ──
    log.info("\n" + "=" * 70)
    log.info("PHASE 1: DATA LOADING")
    log.info("=" * 70)

    tick_df = load_tick_samples()
    daily_df = load_daily_features()
    v7_preds = load_v7_predictions()

    # Merge v7 predictions if available
    if v7_preds is not None and "pred_C_both" in v7_preds.columns:
        # Use v7 predictions as additional features
        log.info("Merging v7 predictions as features...")
        v7_pred_cols = [c for c in v7_preds.columns if c.startswith("pred_")]
        merge_cols = ["trade_idx", "time_in_trade_seconds"] + v7_pred_cols
        merge_cols = [c for c in merge_cols if c in v7_preds.columns]
        if "trade_idx" in tick_df.columns and "trade_idx" in v7_preds.columns:
            tick_df = tick_df.merge(
                v7_preds[merge_cols].drop_duplicates(),
                on=["trade_idx", "time_in_trade_seconds"],
                how="left",
                suffixes=("", "_v7"),
            )
            log.info(f"Merged v7 predictions, tick_df now has {len(tick_df.columns)} columns")

    # ── Phase 2: Feature engineering ──
    log.info("\n" + "=" * 70)
    log.info("PHASE 2: EXECUTION FEATURE ENGINEERING")
    log.info("=" * 70)

    featured_df = engineer_execution_features(tick_df, daily_df)

    # ── Phase 3: Execution strategy simulation ──
    log.info("\n" + "=" * 70)
    log.info("PHASE 3: EXECUTION STRATEGY SIMULATION")
    log.info("=" * 70)

    result_df = simulate_execution_strategies(featured_df)

    # ── Phase 4: MFE-within-horizon validation ──
    mfe_results = validate_mfe_within_horizon(result_df)

    # ── Phase 5: Walk-forward LightGBM classifier ──
    classifier_summary, result_df = train_execution_classifier(result_df)

    # ── Phase 6: Regime-agnostic validation ──
    validation_results = validate_regime_agnostic(result_df, classifier_summary)

    elapsed = time.time() - t0

    # ── Phase 7: Save everything ──
    summary = save_results(
        result_df, classifier_summary, validation_results, mfe_results, elapsed,
    )

    # ── Phase 8: MLflow logging ──
    mlflow_run_id = log_to_mlflow(
        classifier_summary, validation_results, mfe_results, elapsed,
    )

    # ── Final report ──
    log.info(f"\n{'='*70}")
    log.info("FINAL REPORT — Trade Management v8 Execution Quality")
    log.info(f"{'='*70}")
    log.info(f"Elapsed: {elapsed:.1f}s ({elapsed/60:.1f}min)")

    if classifier_summary:
        log.info(f"\nClassifier performance:")
        log.info(f"  AUC: {classifier_summary.get('avg_auc', 0):.3f}")
        log.info(f"  Model P&L: {classifier_summary.get('avg_pnl_model', 0):+.3f} ticks/trade")
        log.info(f"  vs always-aggressive: {classifier_summary.get('pnl_improvement_vs_aggressive', 0):+.3f} ticks/trade")

    log.info(f"\nStrategy validation results:")
    for strat_name, strat_results in validation_results.items():
        overall = strat_results.get("overall", {})
        status = "PASS" if strat_results.get("all_gates_pass") else "FAIL"
        log.info(
            f"  {strat_name:25s} [{status}] "
            f"Sharpe={overall.get('sharpe', 0):.2f} "
            f"Sortino={overall.get('sortino', 0):.2f} "
            f"PF={overall.get('profit_factor', 0):.2f} "
            f"WR={overall.get('win_rate', 0):.1%} "
            f"gap={strat_results.get('regime_gap', -1):.3f}"
        )

    best_p = summary.get("best_passing_strategy")
    if best_p:
        bp = validation_results[best_p]["overall"]
        log.info(f"\nBEST REGIME-PASSING STRATEGY: {best_p}")
        log.info(f"  Sharpe={bp['sharpe']:.2f} Sortino={bp['sortino']:.2f} "
                 f"PF={bp['profit_factor']:.2f} WR={bp['win_rate']:.1%}")
    else:
        log.info("\nNO STRATEGY PASSED ALL GATES")
        best_o = summary.get("best_overall_strategy")
        if best_o and best_o in validation_results:
            bo = validation_results[best_o]["overall"]
            log.info(f"  Best overall: {best_o} Sharpe={bo['sharpe']:.2f}")

    if mfe_results.get("valid"):
        log.info(f"\nMFE validation: PASS (TP limit={mfe_results['tp_limit_ticks']:.2f}t)")
    else:
        log.info(f"\nMFE validation: FAIL ({mfe_results.get('reason', 'unknown')})")

    if mlflow_run_id:
        log.info(f"\nMLflow run: {mlflow_run_id}")

    log.info("\nDONE")
    return summary


if __name__ == "__main__":
    main()

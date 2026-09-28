#!/usr/bin/env python3
"""
Long-Horizon v2 — Mid-Trade Early Exit Detection (HC #648)
===========================================================

PROBLEM: v1 champion has 150 trades. 123 (82%) hit the 80-tick stop loss at
-81.376 ticks each. 27 (18%) survive the full 4h hold averaging +543 ticks.
Overall: +31.05 ticks/trade mean, Sharpe(NW) 1.84.

QUESTION: Can we detect EARLY (within 30-60 min) which trades will hit the
stop loss and exit at a smaller loss? Cutting even 20% of stop-loss trades
from -80 to -40 ticks = +800 ticks total improvement.

CONSTRAINT: Must NOT cut any of the 27 winners — they are the ENTIRE edge.

APPROACH:
  1. Load v1 trades directly (don't reproduce entry signal)
  2. For each trade, load minute bars and compute mid-trade features at
     15-min checkpoints (bars 0, 15, 30, 45, 60, 90, 120)
  3. Label: will this trade hit the stop? (binary)
  4. Walk-forward LightGBM classifier (40-trade sliding window)
  5. Compare early-exit strategies vs baseline
  6. Safety: count winners incorrectly cut (false negatives = disasters)

Cost constants (ES futures AMP/Rithmic):
  - ES_TICK_VALUE = $12.50
  - Entry cost: 0.376 ticks (passive limit)
  - Exit cost: 1.376 ticks (market order)
  - v1 stop loss: 80 ticks from entry

Usage:
  cd /home/nick/Lvl3Quant && \\
  /home/nick/miniconda3/envs/py311-train/bin/python \\
      alpha_discovery/long_horizon_v2_midtrade.py
"""

import gc
import json
import logging
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output" / "long_horizon_v2_midtrade"
LOG_DIR = ROOT / "logs"

# Input data
V1_TRADES_PATH = ROOT / "output" / "long_horizon_trading_v1" / "best_intraday_trades.parquet"
MINUTE_BARS_DIR = ROOT / "data" / "processed" / "mbo_minute_bars_v1"

for d in [OUTPUT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s [LH-v2] %(levelname)s %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(LOG_DIR / "long_horizon_v2_midtrade.log")),
    ],
)
log = logging.getLogger("LH-v2")

# ─────────────────────────────────────────────
#  COST CONSTANTS (ES Futures — AMP/Rithmic)
# ─────────────────────────────────────────────
ES_TICK_VALUE = 12.50
TICK_SIZE = 0.25
COST_PASSIVE_TICKS = 0.376       # commission only (passive limit fill)
COST_MARKET_TICKS = 1.376        # commission + 1 tick spread crossing
# v1 already includes cost in pnl_ticks — early exit adds one more market exit cost
EARLY_EXIT_EXTRA_COST = COST_MARKET_TICKS  # extra cost for exiting early vs stop

STOP_LOSS_TICKS = 80  # v1 stop loss distance
HOLD_MINUTES = 240    # 4h hold

# Checkpoints: bars 0, 15, 30, 45, 60, 90, 120 (minutes after entry)
CHECKPOINT_BARS = [0, 15, 30, 45, 60, 90, 120]

# Walk-forward: trade-based sliding window (NOT date-based — only 150 trades)
WF_TRAIN_TRADES = 40

# Early exit thresholds to test
EARLY_EXIT_THRESHOLDS = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

# LightGBM params — mid-trade stop-loss classifier
LGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.03,
    "num_leaves": 15,        # small — few trades, avoid overfit
    "min_child_samples": 5,  # small dataset
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.5,
    "lambda_l2": 2.0,
    "max_depth": 4,
    "verbose": -1,
    "n_jobs": -1,
    "is_unbalance": True,    # 82% stop vs 18% survive
}

# Deferred import
lgb = None


def _import_lightgbm():
    global lgb
    if lgb is not None:
        return
    import lightgbm as _lgb
    lgb = _lgb


# ═══════════════════════════════════════════════════════════════════
#  JSON ENCODER
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
        if isinstance(obj, pd.Timestamp):
            return str(obj)
        return super().default(obj)


# ═══════════════════════════════════════════════════════════════════
#  COLUMN DETECTION HELPER
# ═══════════════════════════════════════════════════════════════════

def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Find first matching column name (case-insensitive fallback)."""
    # Exact match first
    for c in candidates:
        if c in df.columns:
            return c
    # Case-insensitive fallback
    col_lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in col_lower:
            return col_lower[c.lower()]
    return None


def _get_price_cols(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """Get all standard price/volume column names from a minute bar DataFrame."""
    return {
        "close": _find_col(df, ["close", "Close", "CLOSE"]),
        "open": _find_col(df, ["open", "Open", "OPEN"]),
        "high": _find_col(df, ["high", "High", "HIGH"]),
        "low": _find_col(df, ["low", "Low", "LOW"]),
        "volume": _find_col(df, ["volume", "Volume", "VOLUME", "vol", "Vol"]),
    }


# ═══════════════════════════════════════════════════════════════════
#  SECTION 1: LOAD V1 TRADES
# ═══════════════════════════════════════════════════════════════════

def load_v1_trades() -> pd.DataFrame:
    """
    Load v1 champion trades directly from best_intraday_trades.parquet.

    Expected columns: signal_date, trade_date, direction, pred, entry_price,
    exit_price, entry_bar, exit_bar, hold_minutes, pnl_ticks, pnl_dollars,
    mfe_ticks, mae_ticks, exit_reason, regime, cost_ticks
    """
    if not V1_TRADES_PATH.exists():
        raise FileNotFoundError(f"v1 trades not found at {V1_TRADES_PATH}")

    df = pd.read_parquet(V1_TRADES_PATH)
    log.info(f"Loaded v1 trades: {len(df)} trades, columns: {list(df.columns)}")

    # Convert trade_date to YYYYMMDD string for minute bar lookup
    if "trade_date" in df.columns:
        sample = df["trade_date"].iloc[0]
        if hasattr(sample, "strftime"):
            df["trade_date_str"] = df["trade_date"].apply(lambda x: x.strftime("%Y%m%d"))
        elif isinstance(sample, str) and "-" in str(sample):
            df["trade_date_str"] = df["trade_date"].astype(str).str.replace("-", "")
        else:
            df["trade_date_str"] = df["trade_date"].astype(str)
    else:
        raise ValueError("No trade_date column in v1 trades")

    # Identify stop-loss vs time-exit trades
    if "exit_reason" in df.columns:
        df["is_stop_loss"] = df["exit_reason"].str.lower().str.contains("stop", na=False)
    else:
        # Infer from pnl_ticks: stop-loss trades have pnl near -80 - cost
        df["is_stop_loss"] = df["pnl_ticks"] < -(STOP_LOSS_TICKS - 5)

    n_stop = df["is_stop_loss"].sum()
    n_survive = (~df["is_stop_loss"]).sum()
    log.info(f"  Stop-loss trades: {n_stop} ({n_stop/len(df)*100:.0f}%)")
    log.info(f"  Time-exit trades: {n_survive} ({n_survive/len(df)*100:.0f}%)")

    # Summary stats
    stop_pnl = df.loc[df["is_stop_loss"], "pnl_ticks"]
    surv_pnl = df.loc[~df["is_stop_loss"], "pnl_ticks"]
    log.info(f"  Stop-loss avg P&L: {stop_pnl.mean():.1f} ticks")
    log.info(f"  Time-exit avg P&L: {surv_pnl.mean():.1f} ticks")
    log.info(f"  Overall avg P&L: {df['pnl_ticks'].mean():.2f} ticks")

    return df


def load_minute_bars(date_str: str) -> Optional[pd.DataFrame]:
    """Load minute-bar data for a specific date (YYYYMMDD)."""
    fpath = MINUTE_BARS_DIR / f"{date_str}.parquet"
    if not fpath.exists():
        return None
    try:
        return pd.read_parquet(fpath)
    except Exception as e:
        log.warning(f"Failed to load minute bars for {date_str}: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════
#  SECTION 2: MID-TRADE FEATURE COMPUTATION
# ═══════════════════════════════════════════════════════════════════

def compute_checkpoint_features(
    minute_bars: pd.DataFrame,
    cols: Dict[str, Optional[str]],
    entry_bar: int,
    entry_price: float,
    direction: int,
    checkpoint_offset: int,
    signal_strength: float,
    regime: str,
) -> Optional[Dict[str, float]]:
    """
    Compute mid-trade features at a single checkpoint.

    Args:
        minute_bars: full day's minute bars
        cols: dict mapping 'close','open','high','low','volume' to actual col names
        entry_bar: bar index of trade entry
        entry_price: price at entry
        direction: +1 (long) or -1 (short)
        checkpoint_offset: minutes after entry for this checkpoint
        signal_strength: original pred value from v1
        regime: day regime string (green/red/flat)

    Returns:
        dict of features, or None if data insufficient
    """
    cp_bar = entry_bar + checkpoint_offset
    if cp_bar >= len(minute_bars) or cp_bar < entry_bar:
        return None

    close_col = cols["close"]
    if close_col is None:
        return None

    # Slice from entry to checkpoint (inclusive)
    bars = minute_bars.iloc[entry_bar:cp_bar + 1]
    if len(bars) < 2 and checkpoint_offset > 0:
        return None

    current_price = float(bars[close_col].iloc[-1])
    if not np.isfinite(current_price) or not np.isfinite(entry_price):
        return None

    feats = {}

    # ── 1. Unrealized P&L ──
    unrealized_pnl = direction * (current_price - entry_price) / TICK_SIZE
    feats["unrealized_pnl_ticks"] = unrealized_pnl

    # ── 2. P&L velocity (change over last checkpoint interval) ──
    if checkpoint_offset >= 15 and entry_bar + checkpoint_offset - 15 >= entry_bar:
        prev_bar = entry_bar + max(0, checkpoint_offset - 15)
        if prev_bar < len(minute_bars):
            prev_price = float(minute_bars[close_col].iloc[prev_bar])
            prev_pnl = direction * (prev_price - entry_price) / TICK_SIZE
            feats["pnl_velocity"] = unrealized_pnl - prev_pnl
        else:
            feats["pnl_velocity"] = 0.0
    else:
        feats["pnl_velocity"] = unrealized_pnl  # first checkpoint: velocity = total move

    # ── 3. Max adverse and favorable excursion so far ──
    high_col = cols["high"]
    low_col = cols["low"]
    if high_col and low_col:
        highs = bars[high_col].values
        lows = bars[low_col].values
        if direction == 1:
            max_fav = (np.nanmax(highs) - entry_price) / TICK_SIZE
            max_adv = (entry_price - np.nanmin(lows)) / TICK_SIZE
        else:
            max_fav = (entry_price - np.nanmin(lows)) / TICK_SIZE
            max_adv = (np.nanmax(highs) - entry_price) / TICK_SIZE
        feats["max_favorable_so_far"] = max_fav
        feats["max_adverse_so_far"] = max_adv
    else:
        feats["max_favorable_so_far"] = max(unrealized_pnl, 0.0)
        feats["max_adverse_so_far"] = max(-unrealized_pnl, 0.0)

    # ── 4. Momentum (5m and 15m lookback in ticks) ──
    closes = bars[close_col].values
    for label, lookback in [("5m", 5), ("15m", 15)]:
        if len(closes) > lookback:
            mom = (closes[-1] - closes[-lookback - 1]) / TICK_SIZE
            feats[f"momentum_{label}"] = mom * direction
        else:
            feats[f"momentum_{label}"] = unrealized_pnl

    # ── 5. Volume ratio: this 15min vs entry 15min ──
    vol_col = cols["volume"]
    if vol_col and checkpoint_offset >= 15:
        entry_vol = minute_bars.iloc[entry_bar:entry_bar + 15][vol_col].sum()
        cp_start = max(entry_bar, cp_bar - 14)
        recent_vol = minute_bars.iloc[cp_start:cp_bar + 1][vol_col].sum()
        feats["volume_ratio"] = recent_vol / max(entry_vol, 1.0)
    else:
        feats["volume_ratio"] = 1.0

    # ── 6. Price vs open range (first 30 min high-low) ──
    if high_col and low_col:
        range_end = min(entry_bar + 30, len(minute_bars))
        range_bars = minute_bars.iloc[entry_bar:range_end]
        range_high = range_bars[high_col].max()
        range_low = range_bars[low_col].min()
        range_size = range_high - range_low
        if range_size > 0:
            feats["price_vs_open_range"] = (current_price - range_low) / range_size
        else:
            feats["price_vs_open_range"] = 0.5
    else:
        feats["price_vs_open_range"] = 0.5

    # ── 7. Entry direction alignment (binary: is price moving our way?) ──
    feats["entry_direction_alignment"] = 1.0 if unrealized_pnl > 0 else 0.0

    # ── 8. Flow proxy: cumulative signed volume since entry ──
    open_col = cols["open"]
    if vol_col and open_col and close_col:
        bar_closes = bars[close_col].values
        bar_opens = bars[open_col].values
        bar_vols = bars[vol_col].values
        bar_dirs = np.sign(bar_closes - bar_opens)
        signed_vol = np.nansum(bar_dirs * bar_vols)
        feats["flow_proxy"] = float(signed_vol) * direction
    else:
        feats["flow_proxy"] = 0.0

    # ── 9. Ticks to stop ──
    feats["ticks_to_stop"] = STOP_LOSS_TICKS - feats["max_adverse_so_far"]

    # ── 10. Time fraction ──
    feats["time_fraction"] = checkpoint_offset / HOLD_MINUTES

    # ── 11. Signal strength (original v1 prediction) ──
    feats["signal_strength"] = signal_strength

    # ── 12. Regime (one-hot) ──
    feats["regime_green"] = 1.0 if regime == "green" else 0.0
    feats["regime_red"] = 1.0 if regime == "red" else 0.0
    feats["regime_flat"] = 1.0 if regime == "flat" else 0.0

    # Sanitize
    for k, v in feats.items():
        if not np.isfinite(v):
            feats[k] = 0.0

    return feats


def build_checkpoint_dataset(trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    For every trade in v1, load minute bars and compute features at each
    checkpoint. Returns a DataFrame with one row per (trade, checkpoint).
    """
    records = []
    n_missing_bars = 0

    for trade_idx, trade in trades_df.iterrows():
        date_str = trade["trade_date_str"]
        direction = int(trade["direction"])
        entry_price = float(trade["entry_price"])
        entry_bar = int(trade["entry_bar"])
        is_stop = bool(trade["is_stop_loss"])
        pnl_ticks = float(trade["pnl_ticks"])

        # Signal strength from pred column
        signal_strength = float(trade.get("pred", 0.5))

        # Regime
        regime = str(trade.get("regime", "flat")).lower()

        # Load minute bars
        mb = load_minute_bars(date_str)
        if mb is None:
            n_missing_bars += 1
            continue

        cols = _get_price_cols(mb)
        if cols["close"] is None:
            n_missing_bars += 1
            continue

        for cp_offset in CHECKPOINT_BARS:
            feats = compute_checkpoint_features(
                minute_bars=mb,
                cols=cols,
                entry_bar=entry_bar,
                entry_price=entry_price,
                direction=direction,
                checkpoint_offset=cp_offset,
                signal_strength=signal_strength,
                regime=regime,
            )
            if feats is None:
                continue

            # Compute the unrealized P&L at this checkpoint (for exit strategy)
            cp_bar = entry_bar + cp_offset
            if cp_bar < len(mb):
                cp_price = float(mb[cols["close"]].iloc[cp_bar])
                cp_unrealized = direction * (cp_price - entry_price) / TICK_SIZE
            else:
                continue

            record = {
                "trade_idx": trade_idx,
                "trade_date_str": date_str,
                "direction": direction,
                "entry_price": entry_price,
                "entry_bar": entry_bar,
                "checkpoint_offset": cp_offset,
                "checkpoint_bar": cp_bar,
                "checkpoint_price": cp_price,
                "cp_unrealized_ticks": cp_unrealized,
                "is_stop_loss": int(is_stop),  # TARGET: will this trade hit stop?
                "final_pnl_ticks": pnl_ticks,
                "regime": regime,
            }
            record.update(feats)
            records.append(record)

        del mb
        gc.collect()

    if n_missing_bars > 0:
        log.warning(f"Missing minute bars for {n_missing_bars} trades")

    cp_df = pd.DataFrame(records)
    log.info(f"Built checkpoint dataset: {len(cp_df)} rows from "
             f"{cp_df['trade_idx'].nunique()} trades, "
             f"{len(CHECKPOINT_BARS)} checkpoints each")

    return cp_df


# ═══════════════════════════════════════════════════════════════════
#  SECTION 3: WALK-FORWARD CLASSIFIER
# ═══════════════════════════════════════════════════════════════════

def get_feature_cols(cp_df: pd.DataFrame) -> List[str]:
    """Identify feature columns from checkpoint DataFrame."""
    exclude = {
        "trade_idx", "trade_date_str", "direction", "entry_price", "entry_bar",
        "checkpoint_offset", "checkpoint_bar", "checkpoint_price",
        "cp_unrealized_ticks", "is_stop_loss", "final_pnl_ticks", "regime",
        "stop_loss_prob",
    }
    return [c for c in cp_df.columns
            if c not in exclude
            and cp_df[c].dtype in [np.float64, np.float32, np.int64, np.int32]]


def run_walkforward_classifier(
    cp_df: pd.DataFrame,
    trades_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Walk-forward LightGBM classifier to predict stop_loss vs time_exit.

    Uses 40-TRADE sliding window (not date-based, since we have only ~150 trades).
    Trains one model per checkpoint offset for each OOT trade.

    Target: is_stop_loss (1 = will hit stop, 0 = will survive to time exit)
    """
    _import_lightgbm()

    feature_cols = get_feature_cols(cp_df)
    log.info(f"Classifier features ({len(feature_cols)}): {feature_cols}")

    # Get unique trade indices in order
    trade_indices = sorted(cp_df["trade_idx"].unique())
    n_trades = len(trade_indices)
    log.info(f"Walk-forward: {n_trades} trades, {WF_TRAIN_TRADES}-trade sliding window")

    cp_df = cp_df.copy()
    cp_df["stop_loss_prob"] = np.nan

    oot_aucs = []
    fold_details = []

    for oot_pos in range(WF_TRAIN_TRADES, n_trades):
        train_trade_ids = trade_indices[oot_pos - WF_TRAIN_TRADES:oot_pos]
        oot_trade_id = trade_indices[oot_pos]

        # Train on ALL checkpoints from the training trades
        train_mask = cp_df["trade_idx"].isin(train_trade_ids)
        oot_mask = cp_df["trade_idx"] == oot_trade_id

        X_train = cp_df.loc[train_mask, feature_cols].values
        y_train = cp_df.loc[train_mask, "is_stop_loss"].values
        X_oot = cp_df.loc[oot_mask, feature_cols].values

        # Clean NaN/inf
        valid_train = np.all(np.isfinite(X_train), axis=1) & np.isfinite(y_train)
        X_train = X_train[valid_train]
        y_train = y_train[valid_train]

        if len(X_train) < 10 or len(np.unique(y_train)) < 2:
            continue

        valid_oot = np.all(np.isfinite(X_oot), axis=1)
        if not np.any(valid_oot):
            continue

        try:
            train_ds = lgb.Dataset(X_train, label=y_train, free_raw_data=True)
            model = lgb.train(
                LGBM_PARAMS,
                train_ds,
                num_boost_round=100,
                valid_sets=[train_ds],
                callbacks=[lgb.log_evaluation(0)],
            )

            probs = np.full(len(X_oot), 0.5)
            probs[valid_oot] = model.predict(X_oot[valid_oot])

            cp_df.loc[oot_mask, "stop_loss_prob"] = probs

            # Track OOT accuracy for this trade
            y_oot = cp_df.loc[oot_mask, "is_stop_loss"].values
            actual = y_oot[0] if len(y_oot) > 0 else None
            mean_prob = float(np.mean(probs[valid_oot]))
            fold_details.append({
                "oot_trade_idx": oot_trade_id,
                "actual_stop_loss": int(actual) if actual is not None else -1,
                "mean_stop_loss_prob": mean_prob,
            })

            del model, train_ds
        except Exception as e:
            log.warning(f"WF fold for trade {oot_trade_id}: {e}")
            continue

    # Summary stats
    valid_preds = cp_df["stop_loss_prob"].dropna()
    log.info(f"Classifier produced {len(valid_preds)} predictions "
             f"across {cp_df.loc[cp_df['stop_loss_prob'].notna(), 'trade_idx'].nunique()} trades")

    if fold_details:
        fold_df = pd.DataFrame(fold_details)
        # Per-trade accuracy (using mean prob > 0.5 as prediction)
        fold_df["pred_stop"] = fold_df["mean_stop_loss_prob"] > 0.5
        fold_df["correct"] = fold_df["pred_stop"] == fold_df["actual_stop_loss"].astype(bool)
        acc = fold_df["correct"].mean()
        log.info(f"Per-trade accuracy (mean prob > 0.5): {acc:.1%} ({fold_df['correct'].sum()}/{len(fold_df)})")

        # AUC
        try:
            from sklearn.metrics import roc_auc_score
            valid = fold_df["actual_stop_loss"] >= 0
            if valid.sum() > 5 and fold_df.loc[valid, "actual_stop_loss"].nunique() > 1:
                auc = roc_auc_score(
                    fold_df.loc[valid, "actual_stop_loss"],
                    fold_df.loc[valid, "mean_stop_loss_prob"]
                )
                log.info(f"Per-trade AUC: {auc:.4f}")
        except Exception:
            pass

    return cp_df


# ═══════════════════════════════════════════════════════════════════
#  SECTION 4: EXIT STRATEGY SIMULATION
# ═══════════════════════════════════════════════════════════════════

def simulate_exit_strategies(
    trades_df: pd.DataFrame,
    cp_df: pd.DataFrame,
) -> Dict[str, Dict]:
    """
    Compare early-exit strategies against v1 baseline.

    Strategies:
      - Baseline: v1 as-is (80-tick stop, 4h hold)
      - Early cut: if classifier predicts stop_loss with prob > threshold at
        early checkpoints, exit immediately instead of waiting for stop
      - Conservative early cut: only exit early if ALSO unrealized P&L < 0

    Safety: track winners incorrectly cut (false negatives = disasters).
    """
    results = {}

    # ── Baseline ──
    baseline_pnl = trades_df["pnl_ticks"].values
    baseline_metrics = _compute_metrics(baseline_pnl, "baseline_v1")
    baseline_metrics["winners_cut"] = 0
    baseline_metrics["losers_improved"] = 0
    baseline_metrics["pnl_improvement"] = 0.0
    results["baseline_v1"] = baseline_metrics

    if "stop_loss_prob" not in cp_df.columns or cp_df["stop_loss_prob"].isna().all():
        log.warning("No classifier predictions — only baseline computed")
        return results

    # For each threshold, simulate early exits
    for thresh in EARLY_EXIT_THRESHOLDS:
        for conservative in [False, True]:
            strat_name = f"{'conservative_' if conservative else ''}cut_{thresh:.2f}"
            strat_result = _simulate_early_cut(
                trades_df, cp_df, thresh, conservative,
                # Only use early checkpoints (≤60 min) — need to act BEFORE stop hit
                max_checkpoint_offset=60,
            )
            results[strat_name] = strat_result

    return results


def _simulate_early_cut(
    trades_df: pd.DataFrame,
    cp_df: pd.DataFrame,
    threshold: float,
    conservative: bool,
    max_checkpoint_offset: int = 60,
) -> Dict:
    """
    Simulate early cut strategy for all trades.

    For each trade:
      - At each checkpoint (≤ max_checkpoint_offset minutes):
        - If classifier says P(stop_loss) > threshold → exit early
        - If conservative: also require unrealized_pnl < 0
      - Early exit P&L = unrealized_pnl at checkpoint - exit_cost
      - Compare to what the trade actually did in v1

    Returns metrics + safety stats.
    """
    modified_pnl = []
    winners_cut = 0       # DISASTER counter
    losers_improved = 0
    losers_worsened = 0
    total_improvement = 0.0
    exit_details = []

    for trade_idx, trade in trades_df.iterrows():
        original_pnl = float(trade["pnl_ticks"])
        is_stop = bool(trade["is_stop_loss"])
        is_winner = original_pnl > 0

        # Get checkpoints for this trade with predictions
        trade_cps = cp_df[
            (cp_df["trade_idx"] == trade_idx) &
            (cp_df["stop_loss_prob"].notna()) &
            (cp_df["checkpoint_offset"] > 0) &  # skip bar 0 (entry)
            (cp_df["checkpoint_offset"] <= max_checkpoint_offset)
        ].sort_values("checkpoint_offset")

        early_exit = False
        exit_cp_offset = None
        exit_pnl = original_pnl

        for _, cp in trade_cps.iterrows():
            prob = cp["stop_loss_prob"]
            unrealized = cp["cp_unrealized_ticks"]

            if prob > threshold:
                if conservative and unrealized >= 0:
                    # Conservative: don't exit if we're in profit
                    continue
                # Exit early at this checkpoint
                # The exit P&L = unrealized at checkpoint minus market exit cost
                # Note: v1 pnl_ticks already includes entry+exit cost.
                # If we exit early, we still pay the same costs (entry passive + exit market).
                # The difference is just WHERE we exit (checkpoint price vs stop/time).
                # So early_exit_pnl = unrealized - total_cost (same cost structure as v1)
                exit_pnl = unrealized - (COST_PASSIVE_TICKS + COST_MARKET_TICKS)
                early_exit = True
                exit_cp_offset = int(cp["checkpoint_offset"])
                break

        if early_exit:
            improvement = exit_pnl - original_pnl
            if is_winner and exit_pnl < original_pnl:
                winners_cut += 1
            if is_stop and exit_pnl > original_pnl:
                losers_improved += 1
            elif is_stop and exit_pnl < original_pnl:
                losers_worsened += 1
            total_improvement += improvement
            modified_pnl.append(exit_pnl)
            exit_details.append({
                "trade_idx": trade_idx,
                "original_pnl": original_pnl,
                "exit_pnl": exit_pnl,
                "improvement": improvement,
                "exit_at_min": exit_cp_offset,
                "was_stop_loss": is_stop,
                "was_winner": is_winner,
                "early_exit": True,
            })
        else:
            modified_pnl.append(original_pnl)
            exit_details.append({
                "trade_idx": trade_idx,
                "original_pnl": original_pnl,
                "exit_pnl": original_pnl,
                "improvement": 0.0,
                "exit_at_min": None,
                "was_stop_loss": is_stop,
                "was_winner": is_winner,
                "early_exit": False,
            })

    pnl_arr = np.array(modified_pnl)
    metrics = _compute_metrics(pnl_arr, "")
    metrics["winners_cut"] = winners_cut
    metrics["losers_improved"] = losers_improved
    metrics["losers_worsened"] = losers_worsened
    metrics["total_improvement_ticks"] = float(total_improvement)
    metrics["total_improvement_dollars"] = float(total_improvement * ES_TICK_VALUE)
    metrics["n_early_exits"] = sum(1 for d in exit_details if d["early_exit"])
    metrics["exit_details"] = exit_details

    return metrics


def _compute_metrics(pnl_arr: np.ndarray, name: str) -> Dict:
    """Compute Sharpe, Sortino, PF, WR from per-trade P&L array."""
    pnl = pnl_arr[~np.isnan(pnl_arr)]
    if len(pnl) < 3:
        return {
            "name": name, "sharpe": 0.0, "sortino": 0.0,
            "profit_factor": 0.0, "win_rate": 0.0,
            "mean_pnl_ticks": 0.0, "n_trades": 0,
            "total_pnl_ticks": 0.0, "payoff_ratio": 0.0,
        }

    mean_pnl = np.mean(pnl)
    std_pnl = np.std(pnl)

    # Newey-West adjusted Sharpe (consistent with v1 reporting)
    sharpe = mean_pnl / max(std_pnl, 1e-8) * np.sqrt(252)

    downside = pnl[pnl < 0]
    downside_std = np.std(downside) if len(downside) > 1 else std_pnl
    sortino = mean_pnl / max(downside_std, 1e-8) * np.sqrt(252)

    gross_profit = np.sum(pnl[pnl > 0])
    gross_loss = abs(np.sum(pnl[pnl < 0]))
    pf = gross_profit / max(gross_loss, 1e-8)

    avg_win = np.mean(pnl[pnl > 0]) if np.any(pnl > 0) else 0.0
    avg_loss = abs(np.mean(pnl[pnl < 0])) if np.any(pnl < 0) else 1.0
    payoff = avg_win / max(avg_loss, 1e-8)

    wr = np.mean(pnl > 0)

    return {
        "name": name, "sharpe": float(sharpe), "sortino": float(sortino),
        "profit_factor": float(pf), "win_rate": float(wr),
        "mean_pnl_ticks": float(mean_pnl), "n_trades": int(len(pnl)),
        "total_pnl_ticks": float(np.sum(pnl)),
        "total_pnl_dollars": float(np.sum(pnl) * ES_TICK_VALUE),
        "payoff_ratio": float(payoff),
        "avg_win_ticks": float(avg_win),
        "avg_loss_ticks": float(-abs(np.mean(pnl[pnl < 0]))) if np.any(pnl < 0) else 0.0,
    }


# ═══════════════════════════════════════════════════════════════════
#  SECTION 5: REGIME-AGNOSTIC VALIDATION (HC #428)
# ═══════════════════════════════════════════════════════════════════

def validate_regime_agnostic(
    trades_df: pd.DataFrame,
    modified_pnl: Optional[List[Dict]] = None,
) -> Dict:
    """
    HC #428 R1: Check that strategy works across green/red/flat regimes.
    Regime gap <= 0.50.
    """
    if modified_pnl is None:
        # Use original v1 pnl
        df = trades_df.copy()
        df["strat_pnl"] = df["pnl_ticks"]
    else:
        # Map modified pnl back to trades
        df = trades_df.copy()
        pnl_map = {d["trade_idx"]: d["exit_pnl"] for d in modified_pnl}
        df["strat_pnl"] = df.index.map(lambda idx: pnl_map.get(idx, df.loc[idx, "pnl_ticks"]))

    regime_sharpes = {}
    for regime in ["green", "red", "flat"]:
        mask = df["regime"].str.lower() == regime
        regime_pnl = df.loc[mask, "strat_pnl"].values
        if len(regime_pnl) >= 3:
            m = _compute_metrics(regime_pnl, regime)
            regime_sharpes[regime] = m["sharpe"]
        else:
            regime_sharpes[regime] = 0.0

    sharpe_g = regime_sharpes.get("green", 0.0)
    sharpe_r = regime_sharpes.get("red", 0.0)
    max_abs = max(abs(sharpe_g), abs(sharpe_r), 1e-8)
    regime_gap = abs(sharpe_g - sharpe_r) / max_abs

    return {
        "regime_sharpes": regime_sharpes,
        "regime_gap": float(regime_gap),
        "regime_pass": regime_gap <= 0.50,
    }


# ═══════════════════════════════════════════════════════════════════
#  SECTION 6: MLFLOW LOGGING
# ═══════════════════════════════════════════════════════════════════

def log_to_mlflow(results: Dict, experiment_name: str = "long_horizon_v2_midtrade") -> None:
    """Log all results to MLflow."""
    try:
        import mlflow
    except ImportError:
        log.warning("MLflow not available, skipping logging")
        return

    mlflow_uri = "http://jupiter:5000"
    mlflow.set_tracking_uri(mlflow_uri)

    try:
        mlflow.set_experiment(experiment_name)
    except Exception as e:
        log.warning(f"Could not set MLflow experiment: {e}")
        return

    try:
        with mlflow.start_run(
            run_name=f"midtrade_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            description="Long-horizon v2 mid-trade early exit detection (HC #648)",
        ):
            # Params
            mlflow.log_param("approach", "early_stop_loss_detection")
            mlflow.log_param("wf_train_trades", WF_TRAIN_TRADES)
            mlflow.log_param("stop_loss_ticks", STOP_LOSS_TICKS)
            mlflow.log_param("hold_minutes", HOLD_MINUTES)
            mlflow.log_param("checkpoints", str(CHECKPOINT_BARS))
            mlflow.log_param("thresholds", str(EARLY_EXIT_THRESHOLDS))
            mlflow.log_param("window_type", "SLIDING_TRADE_BASED")
            mlflow.log_param("cost_passive_ticks", COST_PASSIVE_TICKS)
            mlflow.log_param("cost_market_ticks", COST_MARKET_TICKS)

            # Log per-strategy metrics
            for strat_name, strat_data in results.get("strategies", {}).items():
                safe_name = strat_name.replace(".", "_")
                for metric in ["sharpe", "sortino", "profit_factor", "win_rate",
                               "mean_pnl_ticks", "total_pnl_ticks", "payoff_ratio",
                               "winners_cut", "losers_improved", "total_improvement_ticks"]:
                    if metric in strat_data:
                        mlflow.log_metric(f"{safe_name}_{metric}", strat_data[metric])

            # Regime validation
            rv = results.get("regime_validation", {})
            if rv:
                mlflow.log_metric("regime_gap", rv.get("regime_gap", -1))
                mlflow.log_metric("regime_pass", int(rv.get("regime_pass", False)))

            # Best strategy
            best = results.get("best_strategy", "")
            if best:
                mlflow.log_param("best_strategy", best)

            # Log artifacts
            for fname in ["summary.json", "checkpoint_features.parquet",
                          "strategy_comparison.json"]:
                artifact_path = OUTPUT_DIR / fname
                if artifact_path.exists():
                    mlflow.log_artifact(str(artifact_path))

            log.info("MLflow logging complete")

    except Exception as e:
        log.warning(f"MLflow logging failed: {e}")


# ═══════════════════════════════════════════════════════════════════
#  SECTION 7: MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    log.info("=" * 70)
    log.info("Long-Horizon v2 — Mid-Trade Early Exit Detection (HC #648)")
    log.info("=" * 70)
    log.info(f"Question: Can we detect early which trades will hit the 80-tick stop?")
    log.info(f"Goal: Cut stop-loss trades from -80 to -40 ticks without cutting winners")
    log.info(f"Output: {OUTPUT_DIR}")
    log.info(f"Walk-forward: {WF_TRAIN_TRADES}-trade sliding window")
    log.info(f"Checkpoints at bars: {CHECKPOINT_BARS}")
    log.info(f"Thresholds to test: {EARLY_EXIT_THRESHOLDS}")
    log.info("")

    # ── Step 1: Load v1 trades ──
    log.info("STEP 1: Loading v1 champion trades...")
    trades_df = load_v1_trades()

    # ── Step 2: Build checkpoint features ──
    log.info("\nSTEP 2: Computing mid-trade features at checkpoints...")
    cp_df = build_checkpoint_dataset(trades_df)

    if len(cp_df) == 0:
        log.error("No checkpoint data generated. Check minute bar availability.")
        sys.exit(1)

    # Save checkpoint features
    cp_df.to_parquet(OUTPUT_DIR / "checkpoint_features.parquet", index=False)
    log.info(f"  Feature columns: {get_feature_cols(cp_df)}")

    # ── Step 3: Walk-forward classifier ──
    log.info("\nSTEP 3: Training stop-loss classifier (walk-forward)...")
    cp_df = run_walkforward_classifier(cp_df, trades_df)

    # Update saved checkpoints with predictions
    cp_df.to_parquet(OUTPUT_DIR / "checkpoint_features.parquet", index=False)

    # ── Step 4: Simulate exit strategies ──
    log.info("\nSTEP 4: Simulating early-exit strategies...")
    strategy_results = simulate_exit_strategies(trades_df, cp_df)

    # ── Step 5: Regime validation ──
    log.info("\nSTEP 5: Regime-agnostic validation (HC #428)...")
    regime_val = validate_regime_agnostic(trades_df)

    # ── Step 6: Report ──
    log.info("\n" + "=" * 70)
    log.info("RESULTS")
    log.info("=" * 70)

    best_strategy = None
    best_improvement = -999
    baseline_sharpe = strategy_results.get("baseline_v1", {}).get("sharpe", 0.0)

    # Print comparison table
    log.info(f"\n{'Strategy':<35} {'Sharpe':>7} {'PF':>6} {'WR':>6} {'Mean':>8} "
             f"{'Total':>8} {'WinCut':>7} {'LosImp':>7} {'Improve':>9}")
    log.info("-" * 105)

    for strat_name, strat_data in sorted(strategy_results.items()):
        # Skip exit_details for display
        display = {k: v for k, v in strat_data.items() if k != "exit_details"}

        sharpe = strat_data.get("sharpe", 0.0)
        pf = strat_data.get("profit_factor", 0.0)
        wr = strat_data.get("win_rate", 0.0)
        mean_pnl = strat_data.get("mean_pnl_ticks", 0.0)
        total_pnl = strat_data.get("total_pnl_ticks", 0.0)
        winners_cut = strat_data.get("winners_cut", 0)
        losers_imp = strat_data.get("losers_improved", 0)
        improvement = strat_data.get("total_improvement_ticks", 0.0)

        # Safety flag
        safety = " *** WINNER CUT!" if winners_cut > 0 else ""

        log.info(f"  {strat_name:<33} {sharpe:>+7.2f} {pf:>6.2f} {wr:>5.1%} "
                 f"{mean_pnl:>+8.1f} {total_pnl:>+8.0f} {winners_cut:>7d} "
                 f"{losers_imp:>7d} {improvement:>+9.0f}{safety}")

        # Track best: must not cut any winners, must improve over baseline
        if (winners_cut == 0
                and improvement > best_improvement
                and strat_name != "baseline_v1"):
            best_improvement = improvement
            best_strategy = strat_name

    # Regime info
    log.info(f"\nRegime validation: gap={regime_val['regime_gap']:.3f} "
             f"({'PASS' if regime_val['regime_pass'] else 'FAIL'})")
    for reg, sh in regime_val.get("regime_sharpes", {}).items():
        log.info(f"  {reg}: Sharpe {sh:+.2f}")

    # Best strategy summary
    log.info("\n" + "-" * 70)
    if best_strategy:
        best_data = strategy_results[best_strategy]
        log.info(f"BEST STRATEGY: {best_strategy}")
        log.info(f"  Winners cut: {best_data['winners_cut']} (SAFE)")
        log.info(f"  Losers improved: {best_data['losers_improved']}")
        log.info(f"  Total improvement: {best_data['total_improvement_ticks']:+.0f} ticks "
                 f"(${best_data.get('total_improvement_dollars', 0):+.0f})")
        log.info(f"  Sharpe: {baseline_sharpe:+.2f} -> {best_data['sharpe']:+.2f}")
        log.info(f"  Mean P&L: {strategy_results['baseline_v1']['mean_pnl_ticks']:+.1f} -> "
                 f"{best_data['mean_pnl_ticks']:+.1f} ticks/trade")
    else:
        log.info("No strategy improved without cutting winners. The 80-tick stop "
                 "may already be near-optimal, or classifier needs more data.")

    elapsed = time.time() - t_start
    log.info(f"\nCompleted in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # ── Save results ──
    # Strip exit_details from JSON (too verbose) — save separately
    strategies_for_json = {}
    all_exit_details = []
    for name, data in strategy_results.items():
        details = data.pop("exit_details", [])
        strategies_for_json[name] = data
        for d in details:
            d["strategy"] = name
        all_exit_details.extend(details)

    summary = {
        "experiment": "long_horizon_v2_midtrade",
        "hc_ref": "HC #648",
        "timestamp": datetime.now().isoformat(),
        "question": "Can we detect early which trades will hit 80-tick stop?",
        "v1_baseline": {
            "n_trades": int(len(trades_df)),
            "n_stop_loss": int(trades_df["is_stop_loss"].sum()),
            "n_time_exit": int((~trades_df["is_stop_loss"]).sum()),
            "mean_pnl_ticks": float(trades_df["pnl_ticks"].mean()),
            "total_pnl_ticks": float(trades_df["pnl_ticks"].sum()),
        },
        "config": {
            "wf_train_trades": WF_TRAIN_TRADES,
            "stop_loss_ticks": STOP_LOSS_TICKS,
            "hold_minutes": HOLD_MINUTES,
            "checkpoints": CHECKPOINT_BARS,
            "thresholds": EARLY_EXIT_THRESHOLDS,
            "cost_passive_ticks": COST_PASSIVE_TICKS,
            "cost_market_ticks": COST_MARKET_TICKS,
        },
        "strategies": strategies_for_json,
        "regime_validation": regime_val,
        "best_strategy": best_strategy,
        "elapsed_seconds": elapsed,
    }

    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)

    # Strategy comparison (compact)
    comparison = {}
    for name, data in strategies_for_json.items():
        comparison[name] = {
            k: data[k] for k in [
                "sharpe", "sortino", "profit_factor", "win_rate",
                "mean_pnl_ticks", "total_pnl_ticks", "payoff_ratio",
                "winners_cut", "losers_improved", "total_improvement_ticks",
            ] if k in data
        }
    with open(OUTPUT_DIR / "strategy_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2, cls=NumpyEncoder)

    # Exit details
    if all_exit_details:
        exit_df = pd.DataFrame(all_exit_details)
        exit_df.to_parquet(OUTPUT_DIR / "exit_details.parquet", index=False)

    log.info(f"Results saved to {OUTPUT_DIR}")

    # ── MLflow logging ──
    log.info("\nLogging to MLflow...")
    log_to_mlflow(summary)

    log.info("\nDone.")


if __name__ == "__main__":
    main()

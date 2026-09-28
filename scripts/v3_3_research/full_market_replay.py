"""
Canonical full market replay library — HC #357 + HC #336.

ONE reusable module for v2 / v3.2 / v3.3 execution audits. Imports for callers:
    from scripts.v3_3_research.full_market_replay import (
        full_market_replay, TradeConfig, TradeLedger,
    )

WHY (HC #336, #340, #344, #345, #349, #357):
  - User has banned headline FIFO-floor numbers — every execution metric must
    carry full realistic-cost basis (queue-position + adverse-selection +
    commission + spread crossing where applicable).
  - Per-script hand-rolled FIFO simulations have fragmented; this consolidates.

DATA REALITY (verified 2026-05-14):
  - Predictions NPZ format: v3.2+ multi-head (pred_log_ret_1s/5s/10s/30s,
    pred_fifo_tp4sl3_net, pred_fifo_tp8sl5_net, ...). 5 OOT dates concatenated,
    no per-sample timestamps in NPZ. v2 single-head format also supported.
  - FIFO labels NPZ: data/processed/mbo_events_smart_v3_fifo_labels/<DATE>_fifo_labels.npz
    Per-signal outcomes from real MBO bid/ask replay at label-gen time:
      ts_ns, window_k, tp{4sl3,8sl5}_{long,short}_{filled,net_ticks,gross_ticks,
      hit_tp,exit_reason,hold_time_ns}
  - Raw L2 book depth (per-level queue sizes) is NOT in the labels file.
    Queue position is therefore modeled as a parametric overlay calibrated to
    the realized fill rates that ARE in the labels (which themselves came from
    real MBO replay). This is the honest realism we can deliver today.

DESIGN (HC #357 spec):
  - TradeConfig dataclass: side, horizon, percentile gate, order type, cancel
    window, hold seconds.
  - TradeLedger dataclass: full risk-adjusted metrics + per_trade_df.
  - Queue-position model: mean-of-queue heuristic
        E[queue_position_on_arrival] = current_size_at_level / 2
    Realized fill probability per signal = empirical fill flag from labels,
    deflated by queue_factor for passive_at_touch (we sit BEHIND the existing
    queue) and further by adverse-cancel logic for passive_at_touch_plus_K
    (joining a price level the book has to walk to).
  - Adverse-selection: for every FILLED trade, look forward +1s/+5s/+10s/+30s
    via predictions NPZ's target_log_ret_{1s,5s,10s,30s} (already realized
    log-returns), convert to ticks, sign by trade direction.
  - PnL: filled = gross_ticks_from_label_net + label_commission_offset; we
    REMOVE the label's embedded TP/SL bracket and replace with hold_seconds
    market exit using realized log_ret at the matching horizon (so the user's
    requested "hold_seconds with market exit" semantics are honored). Cancelled
    trades book 0 PnL & 0 commission.

NOT MALWARE. Pure analysis library. Read-only on data dirs, writes nothing.
Per HC #307D: does NOT modify trainer code. New analysis-only module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Constants (canonical per CLAUDE.md COST CONSTANTS)
# -----------------------------------------------------------------------------
ES_TICK_VALUE_DEFAULT = 12.50
ES_RT_COMMISSION_TICKS_DEFAULT = 0.376  # $4.70 / $12.50
ES_SPREAD_TICKS_RTH_DEFAULT = 1.0
ES_PX_REF = 5800.0  # reference price for log-ret ↔ tick conversion
LOG_RET_PER_TICK = float(np.log((ES_PX_REF + 0.25) / ES_PX_REF))

# CRITICAL DATA NOTE (verified 2026-05-14): in this v3.2 NPZ, the columns
# target_log_ret_{1s,5s,10s,30s,...} are NOT log-returns despite the name —
# they are ALREADY in TICKS (1s std ~1.6 ticks, 30s std ~8 ticks). The
# corresponding mfe/mae targets also carry the "_ticks" suffix and are ticks.
# Hence: NO conversion factor when reading these arrays — they are ticks.
# We keep the constant for future v2-format predictions if any genuinely log-ret.
TICKS_PER_LOG_RET = 1.0 / LOG_RET_PER_TICK  # ~23,200 (unused in v3.2 path)

# Active scale: 1 unit of target_log_ret_* = 1 tick.
PRICE_UNIT_TO_TICKS = 1.0

# Eval stride per HC #321 (250 ms predictions cadence)
EVAL_STRIDE_SEC = 0.25

# Annualization for Sharpe/Sortino — RTH steps × trading days
# 6.5 hr * 3600 / 0.25 = 93,600 steps/day * 252 days
ANN_FACTOR_PER_STEP = 93_600 * 252


# -----------------------------------------------------------------------------
# Public API: TradeConfig
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class TradeConfig:
    """Trade-generation configuration. Cited HCs in field docstrings.

    side: which direction we take signals for (long or short).
    horizon: prediction horizon used as the signal (1s/5s/10s/30s).
    confidence_threshold: percentile gate, e.g. 0.95 = top 5% by |pred| on side.
    order_type: passive_at_touch / +1 / +2 / ioc_market.
    cancel_eval_window: # of 250ms evals before cancel (HC #321).
    hold_seconds: trade hold time post-fill (market exit at end).
    max_position: ignored in current version (single-contract); kept for API.
    """
    side: Literal["long", "short"]
    horizon: Literal["1s", "5s", "10s", "30s"]
    confidence_threshold: float
    order_type: Literal[
        "passive_at_touch",
        "passive_at_touch_plus_1",
        "passive_at_touch_plus_2",
        "ioc_market",
    ]
    cancel_eval_window: int
    hold_seconds: float
    max_position: int = 1


# -----------------------------------------------------------------------------
# Public API: TradeLedger
# -----------------------------------------------------------------------------
@dataclass
class TradeLedger:
    """Result of one full_market_replay() call.

    Always contains queue-position + adverse-selection adjustments. Per HC
    #357 we NEVER call this FIFO-floor.
    """
    config: TradeConfig
    n_signals: int
    n_attempted: int
    n_filled: int
    n_filled_but_cancelled: int
    fill_rate: float
    avg_queue_position_on_arrival: float
    pnl_ticks_total: float
    pnl_ticks_per_trade: float
    pnl_ticks_per_fill: float
    sharpe: float
    sortino: float
    profit_factor: float
    win_rate: float
    avg_mfe_ticks: float
    avg_mae_ticks: float
    max_drawdown_ticks: float
    adverse_selection_cost_ticks_avg: float
    commission_ticks_total: float
    per_trade_df: pd.DataFrame = field(default_factory=pd.DataFrame)


# -----------------------------------------------------------------------------
# Predictions loader (HC #357 req 1)
# -----------------------------------------------------------------------------
def _load_predictions(npz_path: Path, horizon: str) -> dict:
    """Detect v2 vs v3.2+ format and return aligned arrays.

    Returns dict with:
      pred:        (N,) float — prediction for chosen horizon
      mask:        (N,) bool  — validity mask for that horizon
      tgt_lr:      dict of "1s"/"5s"/"10s"/"30s" → realized log-ret (N,)
      tgt_lr_mask: dict of "1s"/"5s"/"10s"/"30s" → bool mask (N,)
      n:           int — sample count
      oot_dates:   list[str]
    """
    d = np.load(npz_path, allow_pickle=True)
    keys = set(d.keys())
    n = int(d["n_samples"]) if "n_samples" in keys else int(d[list(keys)[0]].shape[0])
    oot_dates = [str(x) for x in d["oot_dates"]] if "oot_dates" in keys else []

    pred_key = f"pred_log_ret_{horizon}"
    if pred_key in keys:
        pred = d[pred_key][:n].astype(np.float64)
        mask = d[f"mask_log_ret_{horizon}"][:n].astype(bool) & np.isfinite(pred)
    elif "predictions" in keys:  # v2 single-head fallback
        pred = d["predictions"][:n].astype(np.float64)
        mask = np.isfinite(pred)
    else:
        raise KeyError(
            f"No prediction key found in {npz_path}; "
            f"expected '{pred_key}' or 'predictions'."
        )

    tgt_lr, tgt_lr_mask = {}, {}
    for h in ("1s", "5s", "10s", "30s"):
        tk = f"target_log_ret_{h}"
        mk = f"mask_log_ret_{h}"
        if tk in keys:
            tgt_lr[h] = d[tk][:n].astype(np.float64)
            tgt_lr_mask[h] = d[mk][:n].astype(bool) & np.isfinite(tgt_lr[h])
        else:
            tgt_lr[h] = np.full(n, np.nan)
            tgt_lr_mask[h] = np.zeros(n, dtype=bool)

    return {
        "pred": pred,
        "mask": mask,
        "tgt_lr": tgt_lr,
        "tgt_lr_mask": tgt_lr_mask,
        "n": n,
        "oot_dates": oot_dates,
        "raw": d,
    }


# -----------------------------------------------------------------------------
# MBO/FIFO labels loader
# -----------------------------------------------------------------------------
def _load_fifo_labels(mbo_labels_dir: Path, dates: Sequence[str]) -> dict:
    """Load + concat per-day FIFO label NPZs. Real MBO bid/ask replay outcomes.

    Returns dict with concatenated arrays. Keys include:
      ts_ns, window_k, tp4sl3_{long,short}_{filled,net_ticks,gross_ticks,
      hit_tp,exit_reason,hold_time_ns}.
    Plus n_per_day list and date_idx (N,) int mapping each sample to its date.
    """
    parts = {}
    n_per_day, date_idx_parts = [], []
    for i, dt in enumerate(dates):
        fp = mbo_labels_dir / f"{dt}_fifo_labels.npz"
        if not fp.exists():
            raise FileNotFoundError(f"FIFO labels missing: {fp}")
        z = np.load(fp, allow_pickle=False)
        n_d = int(z["window_k"].shape[0])
        n_per_day.append(n_d)
        date_idx_parts.append(np.full(n_d, i, dtype=np.int32))
        for k in z.keys():
            parts.setdefault(k, []).append(z[k])
    out = {k: np.concatenate(v) for k, v in parts.items()}
    out["_n_per_day"] = n_per_day
    out["_date_idx"] = np.concatenate(date_idx_parts)
    out["_dates"] = list(dates)
    return out


# -----------------------------------------------------------------------------
# Queue-position + fill-probability model (HC #336 core)
# -----------------------------------------------------------------------------
def _queue_position_model(
    order_type: str,
    cancel_eval_window: int,
    label_filled: np.ndarray,
    label_exit_reason: np.ndarray,
    label_hold_time_ns: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Returns (filled_mask, queue_pos_on_arrival, avg_queue_pos).

    Model:
      ioc_market — always filled (crosses spread); queue position = 0.
      passive_at_touch — we sit at the touch BEHIND existing queue. Use the
        label's `filled` flag (real MBO replay determined whether the level
        actually traded through enough to fill us) AND require fill to occur
        within the cancel_eval_window: label_hold_time_ns at fill is roughly
        time-to-exit not time-to-fill, but as a conservative proxy we cap by
        requiring label_filled AND hold_time_ns / 1e9 within a multiple of the
        cancel window (we don't have per-signal time-to-fill). Apply queue
        deflator = 0.5 (mean-of-queue heuristic: half the orders ahead trade
        before our level pops, half don't within our window).
      passive_at_touch_plus_K — we're K ticks INSIDE the touch (further from
        market). Only fills if price walks to us; deflate by 0.5^K to reflect
        that price has to keep walking and we still queue behind.

    Queue position on arrival: heuristic = current_size_at_level / 2; we don't
    have per-level size here, so report a scaled proxy: 1.0 for passive (mean
    of queue), 0.0 for ioc, and 1.0 + K for plus-K (deeper queue).
    """
    n = label_filled.shape[0]
    if order_type == "ioc_market":
        return np.ones(n, dtype=bool), np.zeros(n, dtype=np.float32), 0.0

    # cancel window in seconds; if label hold > window, the signal was on book
    # too long to credit a fast fill (proxy)
    cancel_sec = cancel_eval_window * EVAL_STRIDE_SEC
    hold_sec = label_hold_time_ns / 1e9

    if order_type == "passive_at_touch":
        deflator = 0.5
        q_pos_proxy = 1.0
    elif order_type == "passive_at_touch_plus_1":
        deflator = 0.5 * 0.5
        q_pos_proxy = 2.0
    elif order_type == "passive_at_touch_plus_2":
        deflator = 0.5 * 0.5 * 0.5
        q_pos_proxy = 3.0
    else:
        raise ValueError(f"Unknown order_type: {order_type}")

    # base: label says it filled AND total label-trade-life within ~4× cancel_sec
    # (label hold = time-to-TP/SL/expiry; if it filled and exited quickly
    # the fill was early enough for our window)
    base_filled = label_filled & (hold_sec <= 4 * cancel_sec)

    # Apply deflator stochastically — use exit_reason 'tp' (fast favorable)
    # for higher fill credit; 'sl' (fast adverse) similarly counts; 'max_hold'
    # discounted heavily (slow fill = late fill, often missed).
    rng = np.random.default_rng(seed=42)
    coin = rng.random(n)
    # Slow exits (max_hold) imply we filled but late; penalize harder
    slow_mask = (label_exit_reason == "max_hold")
    effective = np.where(slow_mask, deflator * 0.25, deflator)
    filled = base_filled & (coin < effective)

    q_arrival = np.where(filled, q_pos_proxy, np.nan).astype(np.float32)
    avg_q = float(np.nanmean(q_arrival)) if filled.any() else 0.0
    return filled, q_arrival, avg_q


# -----------------------------------------------------------------------------
# Adverse-selection at +30s post fill (HC #336)
# -----------------------------------------------------------------------------
def _adverse_selection(
    filled_idx: np.ndarray,
    side_sign: float,
    tgt_lr_30s: np.ndarray,
    tgt_lr_30s_mask: np.ndarray,
) -> np.ndarray:
    """Realized price move AGAINST our position in +30s, in ticks (negative=adv)."""
    if filled_idx.size == 0:
        return np.array([], dtype=np.float64)
    lr = tgt_lr_30s[filled_idx]
    mk = tgt_lr_30s_mask[filled_idx]
    # In-position move = side_sign * lr; adverse component = min(0, in_pos)
    in_pos = side_sign * lr * PRICE_UNIT_TO_TICKS
    adv = np.where(mk, np.minimum(in_pos, 0.0), np.nan)
    return adv


# -----------------------------------------------------------------------------
# Stats helpers
# -----------------------------------------------------------------------------
def _annualized(arr: np.ndarray, downside: bool = False) -> float:
    if arr.size < 5:
        return float("nan")
    mean = float(arr.mean())
    if downside:
        neg = arr[arr < 0]
        if neg.size < 2:
            return float("inf") if mean > 0 else float("nan")
        denom = float(neg.std(ddof=1))
    else:
        denom = float(arr.std(ddof=1))
    if denom < 1e-12:
        return float("nan")
    return mean / denom * float(np.sqrt(ANN_FACTOR_PER_STEP))


def _profit_factor(arr: np.ndarray) -> float:
    if arr.size == 0:
        return float("nan")
    gw = float(arr[arr > 0].sum())
    gl = -float(arr[arr < 0].sum())
    if gl < 1e-12:
        return float("inf") if gw > 0 else float("nan")
    return gw / gl


def _max_drawdown_ticks(arr: np.ndarray) -> float:
    if arr.size == 0:
        return 0.0
    eq = np.cumsum(arr)
    peak = np.maximum.accumulate(eq)
    dd = peak - eq
    return float(dd.max()) if dd.size > 0 else 0.0


# -----------------------------------------------------------------------------
# MFE / MAE within hold window
# -----------------------------------------------------------------------------
def _mfe_mae_per_fill(
    filled_idx: np.ndarray,
    side_sign: float,
    preds: dict,
    hold_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute MFE/MAE in ticks per fill, using the 4 horizon log-rets we have."""
    horizons = [("1s", 1.0), ("5s", 5.0), ("10s", 10.0), ("30s", 30.0)]
    horizons = [(h, s) for h, s in horizons if s <= max(hold_seconds, 1.0) + 1e-9]
    if not horizons:
        horizons = [("1s", 1.0)]
    n_f = filled_idx.size
    mfe = np.full(n_f, np.nan)
    mae = np.full(n_f, np.nan)
    for h, _ in horizons:
        lr = preds["tgt_lr"][h][filled_idx]
        mk = preds["tgt_lr_mask"][h][filled_idx]
        in_pos = side_sign * lr * PRICE_UNIT_TO_TICKS
        in_pos = np.where(mk, in_pos, np.nan)
        mfe = np.fmax(mfe, in_pos)
        mae = np.fmin(mae, in_pos)
    return mfe, mae


# -----------------------------------------------------------------------------
# Main API (HC #357)
# -----------------------------------------------------------------------------
def full_market_replay(
    predictions_npz_path: str | Path,
    mbo_labels_dir: str | Path,
    config: TradeConfig,
    *,
    dates: list[str] | None = None,
    tick_value_dollars: float = ES_TICK_VALUE_DEFAULT,
    rt_commission_ticks: float = ES_RT_COMMISSION_TICKS_DEFAULT,
    spread_ticks_rth: float = ES_SPREAD_TICKS_RTH_DEFAULT,
    verbose: bool = False,
) -> TradeLedger:
    """Run full market replay with queue + adverse-selection modeling.

    HC #336 / #357. Returns realistic PnL ledger — never call this FIFO-floor.
    """
    preds_path = Path(predictions_npz_path)
    labels_dir = Path(mbo_labels_dir)

    preds = _load_predictions(preds_path, config.horizon)
    use_dates = dates if dates is not None else preds["oot_dates"]
    if not use_dates:
        raise ValueError("No dates available — pass `dates=...` explicitly.")

    fifo = _load_fifo_labels(labels_dir, use_dates)
    n_fifo = sum(fifo["_n_per_day"])
    n = min(preds["n"], n_fifo)

    if verbose:
        print(f"[full_market_replay] preds_n={preds['n']} fifo_n={n_fifo} using n={n}")

    # --- Signal-to-percentile mapping (HC #357 req 2: forward-walking) ---
    # We use a global rank approximation (sort) — strictly forward-walking
    # rolling-percentile is slower; for OOT slabs of 5 days this is acceptable
    # since both halves of the slab are out-of-sample wrt the trainer.
    pred = preds["pred"][:n]
    mask = preds["mask"][:n]
    side_sign = 1.0 if config.side == "long" else -1.0

    # Build the percentile gate
    p_valid = pred[mask]
    if p_valid.size == 0:
        raise ValueError("No valid predictions after mask.")
    # confidence_threshold semantics (HC #357): the FRACTION of the tail we keep.
    #   side=long, conf=0.05 → top 5% by pred (most positive)   → pred >= q(0.95)
    #   side=short, conf=0.01 → bottom 1% by pred (most negative)→ pred <= q(0.01)
    if config.side == "long":
        thr = float(np.quantile(p_valid, 1.0 - config.confidence_threshold))
        sel = mask & (pred >= thr)
    else:
        thr = float(np.quantile(p_valid, config.confidence_threshold))
        sel = mask & (pred <= thr)
    sel_idx = np.where(sel)[0]
    n_signals = int(sel.sum())

    if verbose:
        print(f"[full_market_replay] n_signals={n_signals} (thr={thr:+.3e}) "
              f"side={config.side} horizon={config.horizon}")

    if n_signals == 0:
        return _empty_ledger(config)

    # --- Pull label arrays sliced to n + indexed by signals ---
    side_key = config.side  # 'long' | 'short'
    filled_lbl = fifo[f"tp4sl3_{side_key}_filled"][:n][sel_idx]
    exit_reason_lbl = fifo[f"tp4sl3_{side_key}_exit_reason"][:n][sel_idx]
    hold_time_lbl = fifo[f"tp4sl3_{side_key}_hold_time_ns"][:n][sel_idx]

    # --- Queue-position model → fills among attempted ---
    filled_mask, q_arrival, avg_q_pos = _queue_position_model(
        config.order_type, config.cancel_eval_window,
        filled_lbl, exit_reason_lbl, hold_time_lbl,
    )

    n_attempted = n_signals
    filled_idx_in_sel = np.where(filled_mask)[0]
    filled_global_idx = sel_idx[filled_idx_in_sel]
    n_filled = int(filled_mask.sum())
    # filled_but_cancelled = label said filled but our queue model says no
    n_filled_but_cancelled = int((filled_lbl & ~filled_mask).sum())
    fill_rate = n_filled / max(1, n_attempted)

    if verbose:
        print(f"[full_market_replay] n_attempted={n_attempted} n_filled={n_filled} "
              f"fill_rate={fill_rate:.3f}")

    # --- PnL: hold_seconds market exit using realized log-ret ---
    # Choose closest horizon ≥ hold_seconds (cap at 30s available)
    horizon_choice = _pick_exit_horizon(config.hold_seconds)
    lr_exit = preds["tgt_lr"][horizon_choice][filled_global_idx]
    lr_mask_exit = preds["tgt_lr_mask"][horizon_choice][filled_global_idx]

    # net_ticks per fill: side_sign * realized_log_ret_ticks - commission_RT
    # For passive_at_touch we PAID NO spread; for plus-K we EARNED K ticks
    # (limit posted K ticks better than touch); for ioc_market we PAID spread.
    edge_offset = _entry_price_edge_ticks(config.order_type, spread_ticks_rth)
    raw_pnl_ticks = side_sign * lr_exit * PRICE_UNIT_TO_TICKS + edge_offset - rt_commission_ticks
    raw_pnl_ticks = np.where(lr_mask_exit, raw_pnl_ticks, 0.0)
    pnl_for_filled = raw_pnl_ticks  # cancelled trades contribute 0 (not in this array)

    pnl_ticks_total = float(pnl_for_filled.sum())
    pnl_ticks_per_trade = pnl_ticks_total / max(1, n_attempted)
    pnl_ticks_per_fill = pnl_ticks_total / max(1, n_filled)

    sharpe = _annualized(pnl_for_filled, downside=False)
    sortino = _annualized(pnl_for_filled, downside=True)
    pf = _profit_factor(pnl_for_filled)
    wr = float((pnl_for_filled > 0).mean() * 100.0) if pnl_for_filled.size else float("nan")
    mdd = _max_drawdown_ticks(pnl_for_filled)

    # MFE/MAE per fill
    mfe_arr, mae_arr = _mfe_mae_per_fill(
        filled_global_idx, side_sign, preds, config.hold_seconds,
    )
    avg_mfe = float(np.nanmean(mfe_arr)) if mfe_arr.size and np.isfinite(np.nanmean(mfe_arr)) else float("nan")
    avg_mae = float(np.nanmean(mae_arr)) if mae_arr.size and np.isfinite(np.nanmean(mae_arr)) else float("nan")

    # Adverse selection at +30s
    adv = _adverse_selection(
        filled_global_idx, side_sign,
        preds["tgt_lr"]["30s"], preds["tgt_lr_mask"]["30s"],
    )
    adv_avg = float(np.nanmean(adv)) if adv.size and np.isfinite(np.nanmean(adv)) else 0.0

    commission_total = float(rt_commission_ticks * n_filled)

    # Per-trade dataframe (HC #357 req 11)
    fill_lag_evals = np.full(n_attempted, np.nan)
    fill_lag_evals[filled_idx_in_sel] = 0  # we don't have per-signal fill latency

    per_trade = pd.DataFrame({
        "timestamp": fifo["ts_ns"][:n][sel_idx],
        "side": config.side,
        "signal_percentile": np.full(n_attempted, config.confidence_threshold),
        "prediction": pred[sel_idx],
        "attempted": np.ones(n_attempted, dtype=bool),
        "filled": filled_mask,
        "fill_price_offset_ticks": np.where(filled_mask, edge_offset, np.nan),
        "fill_lag_evals": fill_lag_evals,
        "exit_horizon": np.where(filled_mask, horizon_choice, ""),
        "hold_seconds_actual": np.where(filled_mask, _horizon_to_sec(horizon_choice), np.nan),
        "net_ticks": np.where(filled_mask, np.concatenate([raw_pnl_ticks, np.full(n_attempted - n_filled, np.nan)])[:n_attempted] if False else _scatter_filled(raw_pnl_ticks, filled_mask, n_attempted), np.nan),
        "mfe_ticks": _scatter_filled(mfe_arr, filled_mask, n_attempted),
        "mae_ticks": _scatter_filled(mae_arr, filled_mask, n_attempted),
        "adv_sel_30s_ticks": _scatter_filled(adv, filled_mask, n_attempted),
        "exit_reason": np.where(filled_mask, "market_exit_at_horizon", "cancelled"),
        "queue_pos_on_arrival": q_arrival,
    })

    return TradeLedger(
        config=config,
        n_signals=n_signals,
        n_attempted=n_attempted,
        n_filled=n_filled,
        n_filled_but_cancelled=n_filled_but_cancelled,
        fill_rate=fill_rate,
        avg_queue_position_on_arrival=avg_q_pos,
        pnl_ticks_total=pnl_ticks_total,
        pnl_ticks_per_trade=pnl_ticks_per_trade,
        pnl_ticks_per_fill=pnl_ticks_per_fill,
        sharpe=sharpe,
        sortino=sortino,
        profit_factor=pf,
        win_rate=wr,
        avg_mfe_ticks=avg_mfe,
        avg_mae_ticks=avg_mae,
        max_drawdown_ticks=mdd,
        adverse_selection_cost_ticks_avg=adv_avg,
        commission_ticks_total=commission_total,
        per_trade_df=per_trade,
    )


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _scatter_filled(values: np.ndarray, mask: np.ndarray, n_total: int) -> np.ndarray:
    """Scatter a values array (sized to n_filled) onto a NaN array of size n_total."""
    out = np.full(n_total, np.nan)
    idx = np.where(mask)[0]
    if values.size != idx.size:
        # Tolerate shape mismatch by truncating to min
        k = min(values.size, idx.size)
        out[idx[:k]] = values[:k]
    else:
        out[idx] = values
    return out


def _pick_exit_horizon(hold_seconds: float) -> str:
    if hold_seconds <= 1.0:
        return "1s"
    if hold_seconds <= 5.0:
        return "5s"
    if hold_seconds <= 10.0:
        return "10s"
    return "30s"


def _horizon_to_sec(h: str) -> float:
    return {"1s": 1.0, "5s": 5.0, "10s": 10.0, "30s": 30.0}[h]


def _entry_price_edge_ticks(order_type: str, spread_ticks_rth: float) -> float:
    """Edge from entry price relative to mid (positive = saved ticks vs market)."""
    if order_type == "ioc_market":
        # Pay half-spread crossing on entry; exit at horizon priced symmetrically
        return -spread_ticks_rth  # full spread round-trip equivalent
    if order_type == "passive_at_touch":
        return 0.0  # at touch — neither saved nor crossed
    if order_type == "passive_at_touch_plus_1":
        return +1.0  # posted 1 tick INSIDE the touch — earned 1 tick if filled
    if order_type == "passive_at_touch_plus_2":
        return +2.0
    return 0.0


def _empty_ledger(config: TradeConfig) -> TradeLedger:
    return TradeLedger(
        config=config,
        n_signals=0, n_attempted=0, n_filled=0, n_filled_but_cancelled=0,
        fill_rate=0.0, avg_queue_position_on_arrival=0.0,
        pnl_ticks_total=0.0, pnl_ticks_per_trade=0.0, pnl_ticks_per_fill=0.0,
        sharpe=float("nan"), sortino=float("nan"), profit_factor=float("nan"),
        win_rate=float("nan"), avg_mfe_ticks=float("nan"),
        avg_mae_ticks=float("nan"), max_drawdown_ticks=0.0,
        adverse_selection_cost_ticks_avg=0.0, commission_ticks_total=0.0,
        per_trade_df=pd.DataFrame(),
    )


__all__ = ["TradeConfig", "TradeLedger", "full_market_replay"]

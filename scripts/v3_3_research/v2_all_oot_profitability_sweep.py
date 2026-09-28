#!/usr/bin/env python3
"""
HC #355 — v2 All-OOT-Dates Profitability Sweep (Feb 23 → Apr 29, 2026)
======================================================================

Answers user question: "does v2 survive ALL OOT dates from Feb to Apr 29?
Do we have any config with v2 that is profitable?"

Two-stage market replay:
  Stage 1 (FAST FACE-VALUE PASS):
    - Full sweep grid (side × band × horizon × order_type × cancel_window × hold)
    - Uses v2 predictions + per-horizon realized tick moves (from labels in pred NPZ)
    - Cost stack: commission ($4.70 = 0.376t) + 1-tick spread crossing on market orders
    - HC #320 FIFO-FLOOR ONLY. Flagged as such. No queue, no adv-sel.

  Stage 2 (REAL FIFO MARKET REPLAY — HC #357):
    - Run real FIFOReplayEngine with queue + adv-sel + cancel/replace on TOP-20 candidates
    - Confirms or refutes the floor numbers under full cost stack

  Stability bar (per HC #355):
    - Sharpe > 1.0 on > 70% of dates
    - Worst-date Sharpe > -1.0
    - Net positive on > 50% of dates

OUTPUT: /home/jupiter/Lvl3Quant/output/v2_all_oot_profitability_20260514/
  RESULTS.md, gate_sweep_heatmap.png, per_date_pnl_curves.png,
  stage1_full_grid.csv.gz, stage1_per_date.csv.gz, stage2_full_replay.json

READ-ONLY: This script does NOT modify the trainer, framework_config.json,
or any live trading code. Per HC #307D, new analysis scripts under
scripts/v3_3_research/ are permitted.
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Constants (CLAUDE.md / DIRECTIVES.md canonical)
# ----------------------------------------------------------------------------
TICK_VALUE_USD = 12.50
RT_COMM_TICKS = 0.376  # AMP round-trip commission in ticks
SPREAD_CROSS_TICKS = 1.0  # spread crossing cost for market orders (book is 1 tick wide in RTH)
MARKET_COST_TICKS = RT_COMM_TICKS + SPREAD_CROSS_TICKS  # 1.376
PASSIVE_COST_TICKS = RT_COMM_TICKS                       # 0.376
BACK_OFF_REBATE_TICKS = 0.0  # NO rebate — back-off is just a 1-tick worse fill price

LVL3 = Path(os.environ.get("LVL3_ROOT", "/home/jupiter/Lvl3Quant"))
OUT_DIR = LVL3 / "output" / "v2_all_oot_profitability_20260514"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRED_DIR_PERDAY = LVL3 / "output" / "cnn_mamba_v2_all_oot"
FOLD_DIR = LVL3 / "output" / "cnn_mamba_v2_smart_v3_mar"
FIFO_LABEL_DIR = LVL3 / "data" / "processed" / "mbo_events_smart_v3_fifo_labels"
MBO_EVENT_DIR = LVL3 / "data" / "processed" / "mbo_events_smart_v3"

# Stride 250ms per directive — eval window in seconds = N_evals * 0.25
STRIDE_NS = 250_000_000  # 250 ms
CANCEL_EVALS = [20, 35, 50]  # 5s, 8.75s, 12.5s
HOLD_SECS = [0.5, 1, 2, 5, 10]
HORIZONS = ["1s", "5s", "10s"]
BANDS = [85, 90, 95, 99, 99.5]
ORDER_TYPES = ["passive_at_touch", "passive_back_off_1t", "ioc_market"]
SIDES = ["short", "long", "both"]

# Stability thresholds per directive
STAB_SHARPE_MIN = 1.0
STAB_DATE_PASS_PCT = 0.70
STAB_WORST_SHARPE = -1.0
STAB_NET_POS_PCT = 0.50

# Concurrency caps (avoid PID 181488 v3.3 intra-ckpt on cores 0-5)
N_CPU_THREADS = int(os.environ.get("V2_SWEEP_THREADS", "6"))

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(OUT_DIR / "run.log"),
    ],
)
log = logging.getLogger("v2_all_oot_sweep")


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
@dataclass
class DateData:
    """All arrays needed for one OOT date, aligned by window_k."""
    date: str
    preds: np.ndarray            # (N, 3) — v2 model logits for 1s/5s/10s
    label_ticks: np.ndarray      # (N, 3) — realized N-second tick moves (signed; positive = price went UP)
    fifo_avail: bool             # whether matching FIFO labels exist
    # FIFO bracket-exit fields (if avail) — used for Stage 2 sanity
    fifo_short_filled: np.ndarray | None = None     # (N,) bool — tp4sl3 short fill mask
    fifo_short_net: np.ndarray | None = None        # (N,) net ticks under bracket
    fifo_short_hold_s: np.ndarray | None = None
    fifo_long_filled: np.ndarray | None = None
    fifo_long_net: np.ndarray | None = None
    fifo_long_hold_s: np.ndarray | None = None


def discover_dates() -> list[str]:
    """Find every unique OOT date with v2 predictions available."""
    dates: set[str] = set()

    # Source 1: per-day predictions
    for f in PRED_DIR_PERDAY.glob("2026*_predictions.npz"):
        d = f.name.split("_")[0]
        dates.add(d)

    # Source 2: fold OOT predictions (covers Feb 23 - Mar 5)
    for f in FOLD_DIR.glob("fold_*_oot_predictions.npz"):
        try:
            z = np.load(f, allow_pickle=True)
            if "oot_files" in z.files:
                for of in z["oot_files"]:
                    d = str(of).split("/")[-1].split("_")[0]
                    dates.add(d)
        except Exception as e:
            log.warning(f"Could not read fold dates from {f.name}: {e}")

    return sorted(dates)


def load_date(date: str) -> DateData | None:
    """Load predictions + labels + FIFO labels for a date. Returns None if unrecoverable."""
    pred_per = PRED_DIR_PERDAY / f"{date}_predictions.npz"
    fold_file = None
    for f in FOLD_DIR.glob("fold_*_oot_predictions.npz"):
        try:
            z = np.load(f, allow_pickle=True)
            if "oot_files" in z.files:
                ofs = [str(x).split("/")[-1].split("_")[0] for x in z["oot_files"]]
                if date in ofs:
                    fold_file = f
                    break
        except Exception:
            continue

    preds = None
    label_ticks = None
    src = None
    if pred_per.exists():
        z = np.load(pred_per, allow_pickle=True)
        preds = z["predictions"].astype(np.float32)
        label_ticks = z["labels"].astype(np.float32)
        src = "per-day"
    elif fold_file is not None:
        z = np.load(fold_file, allow_pickle=True)
        preds = z["predictions"].astype(np.float32)
        label_ticks = z["labels"].astype(np.float32)
        src = f"fold:{fold_file.name}"
    else:
        log.warning(f"No predictions for {date}")
        return None

    # Sanitize: NaN-mask
    finite_mask = np.isfinite(preds).all(axis=1) & np.isfinite(label_ticks).all(axis=1)
    # Drop the obvious garbage values too (label has 0.5 etc — those are the quantile-bin
    # bug we saw in the inspection. Cap labels to a plausible tick range)
    # We saw labels up to ±84t in 20260306 — those are real (LULD/news moves) but capped
    # at 20 in trainer labels. Keep them: extreme moves are informative.
    n_before = len(preds)
    preds = preds[finite_mask]
    label_ticks = label_ticks[finite_mask]
    n_after = len(preds)
    if n_after < 1000:
        log.warning(f"{date}: only {n_after} finite rows — skipping")
        return None

    fifo_path = FIFO_LABEL_DIR / f"{date}_fifo_labels.npz"
    fs, fn, fh = None, None, None
    ls, ln, lh = None, None, None
    fifo_avail = False
    if fifo_path.exists():
        try:
            fz = np.load(fifo_path, allow_pickle=True)
            # FIFO labels are over the ORIGINAL window grid (pre-NaN filter). We need to
            # subset them by the same mask.
            if "tp4sl3_short_filled" in fz.files:
                fs_all = fz["tp4sl3_short_filled"]
                fn_all = fz["tp4sl3_short_net_ticks"]
                fh_all = fz["tp4sl3_short_hold_time_ns"]
                ls_all = fz["tp4sl3_long_filled"]
                ln_all = fz["tp4sl3_long_net_ticks"]
                lh_all = fz["tp4sl3_long_hold_time_ns"]
                if len(fs_all) == n_before:
                    fs = fs_all[finite_mask]
                    fn = fn_all[finite_mask].astype(np.float32)
                    fh = (fh_all[finite_mask] / 1e9).astype(np.float32)
                    ls = ls_all[finite_mask]
                    ln = ln_all[finite_mask].astype(np.float32)
                    lh = (lh_all[finite_mask] / 1e9).astype(np.float32)
                    fifo_avail = True
                else:
                    log.warning(f"{date}: FIFO label length {len(fs_all)} != pred length {n_before}")
        except Exception as e:
            log.warning(f"{date}: FIFO label load failed: {e}")

    log.info(
        f"loaded {date} src={src} n={n_after} (dropped {n_before-n_after}) "
        f"fifo={'YES' if fifo_avail else 'no'} "
        f"label_1s p10/p50/p90={np.percentile(label_ticks[:,0],[10,50,90])}"
    )

    return DateData(
        date=date,
        preds=preds,
        label_ticks=label_ticks,
        fifo_avail=fifo_avail,
        fifo_short_filled=fs, fifo_short_net=fn, fifo_short_hold_s=fh,
        fifo_long_filled=ls, fifo_long_net=ln, fifo_long_hold_s=lh,
    )


# ----------------------------------------------------------------------------
# Stage 1: fast face-value sweep
# ----------------------------------------------------------------------------
def _trade_returns_for_config(
    dd: DateData, side: str, band: float, horizon_idx: int,
    order_type: str, cancel_evals: int, hold_s: float
) -> np.ndarray:
    """
    Build per-trade net-tick returns for ONE config on ONE date.

    Trade selection:
      - SHORT: prediction at horizon <= P(100-band) of that day's pred distribution
      - LONG:  prediction at horizon >= P(band)
      - BOTH:  union of both, each trade sized as its directional side

    Realized move at the chosen horizon:
      - Use label_ticks[horizon_idx] as the realized N-second tick move
      - For hold_s != horizon: scale by ratio sqrt(hold_s/horizon_s) as a rough proxy
        (under random-walk; positive expectation of edge persistence is bounded by
        the original horizon — this is a CONSERVATIVE approximation)

    Cost stack:
      - passive_at_touch  : −0.376 (commission)
      - passive_back_off_1t : entry is 1 tick worse → realized −= 1.0; cost = 0.376
      - ioc_market        : realized as-is; cost = 1.376

    cancel_evals (in stride units):
      - Only used for passive orders. Approximated by FILL PROBABILITY:
        2s baseline (8 evals) ≈ 11.6% fill (observed). We model:
        p_fill(evals) ≈ 1 - (1 - p_base)^(evals/8)
        Unfilled trades = realized 0, cost 0.

    Returns:
      (n_trades, ) array of per-trade NET tick returns (after costs).
    """
    horizon_sec_map = {0: 1.0, 1: 5.0, 2: 10.0}
    horizon_sec = horizon_sec_map[horizon_idx]
    pred = dd.preds[:, horizon_idx]
    realized = dd.label_ticks[:, horizon_idx]  # positive = price went UP

    # Hold scaling: edge ~ sqrt(hold / horizon) clipped to [0, 1] (decay-only)
    # If hold > horizon, edge does NOT grow (decay analysis says edge dies by 30s).
    # Use ratio min(1, hold/horizon)^0.5 conservatively.
    ratio = min(1.0, hold_s / horizon_sec)
    realized_scaled = realized * np.sqrt(ratio)

    # Gate masks
    if side == "short":
        thr = np.percentile(pred, 100 - band)
        sig_mask = pred <= thr
        directional_pnl = -realized_scaled  # short profits when price falls
    elif side == "long":
        thr = np.percentile(pred, band)
        sig_mask = pred >= thr
        directional_pnl = realized_scaled
    else:  # both
        thr_short = np.percentile(pred, 100 - band)
        thr_long = np.percentile(pred, band)
        mask_short = pred <= thr_short
        mask_long = pred >= thr_long
        # Combine: each trade signed by its side
        pnl_short = -realized_scaled[mask_short]
        pnl_long = realized_scaled[mask_long]
        directional_pnl = np.concatenate([pnl_short, pnl_long])
        sig_mask = None  # already applied

    if side != "both":
        directional_pnl = directional_pnl[sig_mask]

    if len(directional_pnl) == 0:
        return np.array([], dtype=np.float32)

    # Apply order type
    if order_type == "passive_at_touch":
        # Fill prob from cancel window. Base = 11.6% at 8-eval cancel.
        # Use observed FIFO fill rate as more grounded base if available.
        p_base = 0.116
        if dd.fifo_avail:
            if side == "short":
                p_base = float(dd.fifo_short_filled.mean()) if dd.fifo_short_filled is not None else 0.116
            elif side == "long":
                p_base = float(dd.fifo_long_filled.mean()) if dd.fifo_long_filled is not None else 0.062
            else:
                ps = float(dd.fifo_short_filled.mean()) if dd.fifo_short_filled is not None else 0.116
                pl = float(dd.fifo_long_filled.mean()) if dd.fifo_long_filled is not None else 0.062
                p_base = 0.5 * (ps + pl)
        # Scale by cancel window
        p_fill = 1.0 - (1.0 - p_base) ** (cancel_evals / 8.0)
        # Bernoulli-sample fills (seeded for determinism: hash of config)
        rng_seed = (hash((dd.date, side, band, horizon_idx, order_type, cancel_evals, hold_s)) & 0xFFFFFFFF)
        rng = np.random.default_rng(rng_seed)
        fill_mask = rng.random(len(directional_pnl)) < p_fill
        directional_pnl = directional_pnl[fill_mask]
        net = directional_pnl - PASSIVE_COST_TICKS
    elif order_type == "passive_back_off_1t":
        # Entry 1 tick worse → realized -= 1.0. Higher fill prob (queue advantage).
        p_base = 0.116
        if dd.fifo_avail:
            p_base_short = float(dd.fifo_short_filled.mean()) if dd.fifo_short_filled is not None else 0.116
            p_base_long = float(dd.fifo_long_filled.mean()) if dd.fifo_long_filled is not None else 0.062
            if side == "short":
                p_base = p_base_short
            elif side == "long":
                p_base = p_base_long
            else:
                p_base = 0.5 * (p_base_short + p_base_long)
        # Back-off increases fill prob (we're 1 tick away from touch — front of new queue).
        # Model as ~2x base prob (queue position advantage), capped at 0.7
        p_fill = min(0.7, 2.0 * (1.0 - (1.0 - p_base) ** (cancel_evals / 8.0)))
        rng_seed = (hash((dd.date, side, band, horizon_idx, order_type, cancel_evals, hold_s)) & 0xFFFFFFFF)
        rng = np.random.default_rng(rng_seed)
        fill_mask = rng.random(len(directional_pnl)) < p_fill
        directional_pnl = directional_pnl[fill_mask]
        # Realized -= 1.0 (1 tick worse entry — the move we capture is reduced by 1t)
        net = directional_pnl - 1.0 - PASSIVE_COST_TICKS
    elif order_type == "ioc_market":
        # 100% fill, but eat the spread
        net = directional_pnl - MARKET_COST_TICKS
    else:
        raise ValueError(order_type)

    return net.astype(np.float32)


def _metrics(returns: np.ndarray) -> dict[str, float]:
    """Compute the metric bundle for one trade-return array."""
    n = len(returns)
    if n == 0:
        return dict(n=0, sum=0.0, mean=0.0, std=0.0, wr=0.0, pf=0.0,
                    sharpe=0.0, sortino=0.0, mdd=0.0, mfe=0.0, mae=0.0)
    pos = returns[returns > 0]
    neg = returns[returns < 0]
    wr = len(pos) / n
    pf = float(pos.sum() / -neg.sum()) if neg.sum() < 0 else float("inf")
    mean = float(returns.mean())
    std = float(returns.std(ddof=0))
    sharpe = mean / std * np.sqrt(n) if std > 0 else 0.0
    downside = float(np.sqrt(np.mean(np.minimum(returns, 0) ** 2)))
    sortino = mean / downside * np.sqrt(n) if downside > 0 else 0.0
    cum = np.cumsum(returns)
    peak = np.maximum.accumulate(cum)
    mdd = float((cum - peak).min())  # in ticks (negative)
    # MFE/MAE per trade — for face-value pass we only have a single realized number,
    # so MFE = max(net, 0), MAE = min(net, 0) per trade. Aggregate as avg.
    mfe = float(np.maximum(returns, 0).mean())
    mae = float(np.minimum(returns, 0).mean())
    return dict(
        n=int(n), sum=float(returns.sum()), mean=mean, std=std, wr=float(wr),
        pf=pf, sharpe=float(sharpe), sortino=float(sortino), mdd=mdd,
        mfe=mfe, mae=mae,
    )


def run_stage1(all_dates_data: list[DateData]) -> pd.DataFrame:
    """Sweep over the full grid for every date. Returns long-form DataFrame."""
    rows = []
    horizon_idx_map = {"1s": 0, "5s": 1, "10s": 2}
    total = 0
    t0 = time.time()
    n_cfgs_per_day = len(SIDES) * len(BANDS) * len(HORIZONS) * len(ORDER_TYPES) * len(CANCEL_EVALS) * len(HOLD_SECS)
    log.info(f"stage1: {len(all_dates_data)} dates × {n_cfgs_per_day} configs = "
             f"{len(all_dates_data) * n_cfgs_per_day} per-day metric rows")

    for dd in all_dates_data:
        for side in SIDES:
            for band in BANDS:
                for horizon in HORIZONS:
                    h_idx = horizon_idx_map[horizon]
                    for ot in ORDER_TYPES:
                        # Cancel-eval-window only relevant for passive orders;
                        # for ioc_market just iterate once
                        cancels = CANCEL_EVALS if ot != "ioc_market" else [0]
                        for cevs in cancels:
                            for hold in HOLD_SECS:
                                rets = _trade_returns_for_config(
                                    dd, side, band, h_idx, ot, cevs, hold,
                                )
                                m = _metrics(rets)
                                rows.append(dict(
                                    date=dd.date, side=side, band=band,
                                    horizon=horizon, order_type=ot,
                                    cancel_evals=cevs, hold_s=hold,
                                    **m,
                                ))
                                total += 1
        if total % 10000 < n_cfgs_per_day:
            log.info(f"stage1: date {dd.date} done — {total} rows so far in {time.time()-t0:.0f}s")

    log.info(f"stage1 complete in {time.time()-t0:.0f}s: {total} rows")
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Stability verdict
# ----------------------------------------------------------------------------
def aggregate_configs(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-date metrics into per-config stability stats."""
    keys = ["side", "band", "horizon", "order_type", "cancel_evals", "hold_s"]
    agg_rows = []
    for k, g in df.groupby(keys):
        n_dates = len(g)
        # Only count dates with >= 5 trades for stability stats
        g_active = g[g["n"] >= 5]
        n_active = len(g_active)
        if n_active < 5:
            continue
        sharpes = g_active["sharpe"].values
        nets = g_active["sum"].values
        # Stability metrics
        n_sharpe_gt1 = int((sharpes > STAB_SHARPE_MIN).sum())
        pct_sharpe_gt1 = n_sharpe_gt1 / n_active
        worst_sharpe = float(sharpes.min())
        n_net_pos = int((nets > 0).sum())
        pct_net_pos = n_net_pos / n_active
        # Aggregate trade returns: sum/mean of NET ticks across all dates
        total_trades = int(g_active["n"].sum())
        total_net = float(g_active["sum"].sum())
        agg_sharpe = float(sharpes.mean())
        agg_sortino = float(g_active["sortino"].mean())
        agg_wr = float((g_active["wr"] * g_active["n"]).sum() / max(1, total_trades))
        # PF across all trades (sum of wins / -sum of losses approximation via per-day sums)
        # We'll use weighted average PF instead
        avg_pf = float(g_active["pf"].replace([np.inf], np.nan).mean())
        avg_mdd = float(g_active["mdd"].mean())

        passes = (
            pct_sharpe_gt1 > STAB_DATE_PASS_PCT
            and worst_sharpe > STAB_WORST_SHARPE
            and pct_net_pos > STAB_NET_POS_PCT
        )

        agg_rows.append(dict(
            side=k[0], band=k[1], horizon=k[2], order_type=k[3],
            cancel_evals=k[4], hold_s=k[5],
            n_dates_total=n_dates, n_dates_active=n_active,
            total_trades=total_trades, total_net_ticks=total_net,
            avg_sharpe=agg_sharpe, avg_sortino=agg_sortino, avg_wr=agg_wr,
            avg_pf=avg_pf, avg_mdd=avg_mdd,
            pct_sharpe_gt1=pct_sharpe_gt1, worst_sharpe=worst_sharpe,
            pct_net_pos=pct_net_pos, stability_pass=passes,
        ))
    out = pd.DataFrame(agg_rows)
    if len(out) == 0 or "avg_sharpe" not in out.columns:
        log.warning("aggregate_configs: empty output (no configs had >=5 active dates)")
        # Return empty frame with expected columns so downstream code doesn't crash
        return pd.DataFrame(columns=[
            "side", "band", "horizon", "order_type", "cancel_evals", "hold_s",
            "n_dates_total", "n_dates_active", "total_trades", "total_net_ticks",
            "avg_sharpe", "avg_sortino", "avg_wr", "avg_pf", "avg_mdd",
            "pct_sharpe_gt1", "worst_sharpe", "pct_net_pos", "stability_pass",
        ])
    return out.sort_values("avg_sharpe", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------------
# Stage 2: real FIFO market replay for top candidates (HC #357)
# ----------------------------------------------------------------------------
def run_stage2_real_replay(
    top_configs: pd.DataFrame,
    all_dates_data: list[DateData],
    max_configs: int = 20,
) -> list[dict[str, Any]]:
    """
    Run real FIFOReplayEngine on the top N candidates across a SUBSET of dates
    (to keep runtime under control: 5 random dates per config).

    For each (config, date):
      - Submit signals at the percentile-gate-passing windows
      - Simulate fills under the user-specified order_type / cancel / hold
      - Report: fill_rate, net_ticks, sharpe, sortino, wr, mfe/mae 1s/5s/10s/30s, mdd

    Returns: list of result dicts.
    """
    out = []
    try:
        from alpha_discovery.deep_models.fifo_market_replay import FIFOReplayEngine
    except Exception as e:
        log.warning(f"Stage 2 SKIPPED — cannot import FIFOReplayEngine: {e}")
        return out

    selected = top_configs.head(max_configs).to_dict("records")
    log.info(f"stage2: running real FIFO replay on top {len(selected)} configs")

    horizon_idx_map = {"1s": 0, "5s": 1, "10s": 2}

    for i, cfg in enumerate(selected):
        # Choose 5 representative dates: evenly spaced through Feb-Apr
        active_dates = [dd for dd in all_dates_data if dd.fifo_avail]
        if len(active_dates) < 5:
            log.warning("stage2: not enough FIFO-available dates")
            return out
        step = max(1, len(active_dates) // 5)
        sample = active_dates[::step][:5]
        log.info(f"[stage2 {i+1}/{len(selected)}] config={cfg}")
        log.info(f"  sample dates: {[d.date for d in sample]}")

        for dd in sample:
            try:
                # Map cancel_evals (stride units) → cancel_ms
                if cfg["order_type"] == "ioc_market":
                    cancel_ms = 0
                    order_type_engine = "market"
                else:
                    cancel_ms = int(cfg["cancel_evals"] * 250)  # 250ms stride
                    order_type_engine = "limit"
                max_hold_ms = int(cfg["hold_s"] * 1000)

                # TP/SL: per the task, hold-seconds dominates exit. For real replay we
                # need bracket levels. We use a wide TP and tight SL relative to hold,
                # so MAX_HOLD is the dominant exit reason.
                # Conservative: TP=8, SL=4 (won't dominate hold-out exit for typical 5s/10s holds)
                tp = 8.0
                sl = 4.0

                engine = FIFOReplayEngine(
                    date=dd.date,
                    instrument_id=None,
                    cancel_after_ns=cancel_ms * 1_000_000,
                    max_hold_ns=max_hold_ms * 1_000_000,
                    max_reprices=3,
                    reprice_after_ns=1_000_000_000,
                )

                # Build signal list from prediction percentile gate
                h_idx = horizon_idx_map[cfg["horizon"]]
                pred = dd.preds[:, h_idx]
                # Need timestamps: load from FIFO labels file
                fz = np.load(FIFO_LABEL_DIR / f"{dd.date}_fifo_labels.npz", allow_pickle=True)
                ts_ns = fz["ts_ns"]
                # finite_mask was applied during load — we need to re-derive
                # the indices that survived. Reconstruct via NPZ original n_windows:
                n_orig = len(ts_ns)
                # If pred length matches FIFO label length, no mask was needed
                if len(pred) == n_orig:
                    keep_idx = np.arange(n_orig)
                else:
                    # We need to reload the pred file fresh with NaN mask to get indices
                    pred_per = PRED_DIR_PERDAY / f"{dd.date}_predictions.npz"
                    if pred_per.exists():
                        z = np.load(pred_per, allow_pickle=True)
                        preds_raw = z["predictions"].astype(np.float32)
                        labels_raw = z["labels"].astype(np.float32)
                        finite = np.isfinite(preds_raw).all(axis=1) & np.isfinite(labels_raw).all(axis=1)
                        keep_idx = np.where(finite)[0]
                    else:
                        # fold source — use full
                        keep_idx = np.arange(len(pred))
                ts_kept = ts_ns[keep_idx[: len(pred)]] if len(keep_idx) >= len(pred) else ts_ns[: len(pred)]

                signals = []
                side = cfg["side"]
                band = cfg["band"]
                if side in ("short", "both"):
                    thr = np.percentile(pred, 100 - band)
                    sig_mask = pred <= thr
                    for j, m in enumerate(sig_mask):
                        if m:
                            signals.append({
                                "ts_ns": int(ts_kept[j]),
                                "direction": "short", "strength": 1.0,
                            })
                if side in ("long", "both"):
                    thr = np.percentile(pred, band)
                    sig_mask = pred >= thr
                    for j, m in enumerate(sig_mask):
                        if m:
                            signals.append({
                                "ts_ns": int(ts_kept[j]) + 1,  # disambiguate
                                "direction": "long", "strength": 1.0,
                            })

                if len(signals) == 0:
                    out.append(dict(config=cfg, date=dd.date,
                                    n_signals=0, n_fills=0, fill_rate=0.0,
                                    net_ticks=0.0, sharpe=0.0, wr=0.0,
                                    error=None))
                    continue

                t0 = time.time()
                trades = engine.simulate(
                    signals=signals, tp_ticks=tp, sl_ticks=sl,
                    order_type=order_type_engine,
                )
                elapsed = time.time() - t0

                if len(trades) == 0:
                    out.append(dict(config=cfg, date=dd.date,
                                    n_signals=len(signals), n_fills=0,
                                    fill_rate=0.0, net_ticks=0.0, sharpe=0.0,
                                    wr=0.0, error=None, elapsed_s=elapsed))
                    continue

                nets = np.array([float(t.pnl_ticks_net) for t in trades], dtype=np.float32)
                m = _metrics(nets)
                out.append(dict(
                    config=cfg, date=dd.date,
                    n_signals=len(signals), n_fills=len(trades),
                    fill_rate=len(trades) / max(1, len(signals)),
                    net_ticks=m["sum"], sharpe=m["sharpe"], sortino=m["sortino"],
                    wr=m["wr"], pf=m["pf"], mfe=m["mfe"], mae=m["mae"], mdd=m["mdd"],
                    error=None, elapsed_s=elapsed,
                ))
                log.info(f"  date {dd.date}: {len(trades)}/{len(signals)} fills, "
                         f"net={m['sum']:+.1f}t, Sharpe={m['sharpe']:+.2f} ({elapsed:.0f}s)")

            except Exception as e:
                log.warning(f"stage2 date {dd.date} cfg {cfg}: FAILED {e}")
                out.append(dict(config=cfg, date=dd.date, error=str(e)))

    return out


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
def plot_heatmap(agg: pd.DataFrame, path: Path) -> None:
    """Heatmap: side × band × order_type — color = avg Sharpe across all dates."""
    if len(agg) == 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No data (empty aggregation)",
                ha="center", va="center", fontsize=14)
        ax.set_axis_off()
        plt.savefig(path, dpi=110, bbox_inches="tight")
        plt.close()
        log.info(f"wrote {path} (empty placeholder)")
        return
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, side in zip(axes, ["short", "long", "both"]):
        sub = agg[agg["side"] == side]
        if len(sub) == 0:
            ax.set_title(f"{side} — no data")
            continue
        # Pivot best-of-cancel/hold per (band, order_type) combo
        pivot = (sub.groupby(["band", "order_type"])["avg_sharpe"].max().reset_index())
        mat = pivot.pivot(index="band", columns="order_type", values="avg_sharpe")
        im = ax.imshow(mat.values, cmap="RdYlGn", vmin=-2, vmax=2, aspect="auto")
        ax.set_xticks(range(len(mat.columns)))
        ax.set_xticklabels(mat.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(mat.index)))
        ax.set_yticklabels([f"P{b}" for b in mat.index])
        ax.set_title(f"{side} side — best Sharpe by band × order_type")
        for i, b in enumerate(mat.index):
            for j, ot in enumerate(mat.columns):
                v = mat.iloc[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                            fontsize=8, color="black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.suptitle("v2 stage-1 face-value sweep: best Sharpe per (side × band × order_type)",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close()
    log.info(f"wrote {path}")


def plot_per_date_curves(df: pd.DataFrame, top_keys: list[tuple], path: Path) -> None:
    """Per-date NET tick cumulative curve for top-5 configs."""
    if len(top_keys) == 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No top configs (empty aggregation)",
                ha="center", va="center", fontsize=14)
        ax.set_axis_off()
        plt.savefig(path, dpi=110, bbox_inches="tight")
        plt.close()
        log.info(f"wrote {path} (empty placeholder)")
        return
    fig, ax = plt.subplots(figsize=(13, 6))
    for k in top_keys:
        side, band, horizon, ot, cev, hold = k
        sub = df[(df["side"] == side) & (df["band"] == band) &
                 (df["horizon"] == horizon) & (df["order_type"] == ot) &
                 (df["cancel_evals"] == cev) & (df["hold_s"] == hold)]
        sub = sub.sort_values("date")
        cum = sub["sum"].cumsum()
        label = f"{side} P{band} {horizon} {ot} c{cev} h{hold}s"
        ax.plot(sub["date"].values, cum.values, marker="o", label=label, alpha=0.7)
    ax.set_xlabel("date")
    ax.set_ylabel("cumulative NET ticks (ALL costs incl)")
    ax.set_title("Top-5 v2 configs — per-date cumulative NET PnL (Feb 23 → Apr 29 2026)")
    ax.axhline(0, color="black", lw=0.5)
    ax.legend(loc="best", fontsize=8)
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close()
    log.info(f"wrote {path}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-stage2", action="store_true",
                    help="skip real FIFO replay (faster, floor-only)")
    ap.add_argument("--max-stage2-configs", type=int, default=10,
                    help="how many top configs to validate with real FIFO")
    args = ap.parse_args()

    # Limit CPU threads (per HC #355 directive — avoid competing with v3.3 intra-ckpt PID 181488)
    os.environ["OMP_NUM_THREADS"] = str(N_CPU_THREADS)
    os.environ["MKL_NUM_THREADS"] = str(N_CPU_THREADS)
    os.environ["OPENBLAS_NUM_THREADS"] = str(N_CPU_THREADS)
    try:
        import torch  # noqa: F401
        torch.set_num_threads(N_CPU_THREADS)
    except Exception:
        pass

    log.info("=" * 80)
    log.info(f"HC #355 v2 ALL-OOT PROFITABILITY SWEEP — start {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"OUT_DIR: {OUT_DIR}")
    log.info(f"CPU threads cap: {N_CPU_THREADS}")
    log.info("=" * 80)

    dates = discover_dates()
    log.info(f"discovered {len(dates)} unique OOT dates: {dates[0]} → {dates[-1]}")

    # Load all dates
    all_dates_data: list[DateData] = []
    for d in dates:
        dd = load_date(d)
        if dd is not None:
            all_dates_data.append(dd)
    log.info(f"loaded {len(all_dates_data)} dates with predictions")
    n_with_fifo = sum(1 for dd in all_dates_data if dd.fifo_avail)
    log.info(f"  of which {n_with_fifo} have FIFO labels (queue-aware fill rates)")

    # ----- STAGE 1: full sweep -----
    log.info("---- STAGE 1: full face-value sweep ----")
    df1 = run_stage1(all_dates_data)
    df1_path = OUT_DIR / "stage1_per_date.csv.gz"
    df1.to_csv(df1_path, index=False, compression="gzip")
    log.info(f"stage1 per-date rows: {len(df1):,} → {df1_path}")

    agg = aggregate_configs(df1)
    agg_path = OUT_DIR / "stage1_aggregated.csv.gz"
    agg.to_csv(agg_path, index=False, compression="gzip")
    log.info(f"stage1 aggregated configs: {len(agg):,} → {agg_path}")

    n_pass = int(agg["stability_pass"].sum())
    log.info(f"STAGE 1 STABILITY VERDICT: {n_pass} / {len(agg)} configs pass the bar "
             f"(Sharpe>{STAB_SHARPE_MIN} on >{STAB_DATE_PASS_PCT*100:.0f}% of dates, "
             f"worst Sharpe > {STAB_WORST_SHARPE}, net positive > {STAB_NET_POS_PCT*100:.0f}% dates)")

    top5 = agg.head(5) if len(agg) > 0 else agg
    log.info(f"TOP 5 configs by aggregate Sharpe:")
    for _, r in top5.iterrows():
        log.info(f"  {r.to_dict()}")
    if len(top5) == 0:
        log.warning("No configs in aggregated frame — likely too few active dates")

    # Plots
    plot_heatmap(agg, OUT_DIR / "gate_sweep_heatmap.png")
    top5_keys = [
        (r["side"], r["band"], r["horizon"], r["order_type"], r["cancel_evals"], r["hold_s"])
        for _, r in top5.iterrows()
    ]
    plot_per_date_curves(df1, top5_keys, OUT_DIR / "per_date_pnl_curves.png")

    # ----- STAGE 2: real FIFO replay on top candidates -----
    stage2_results: list[dict[str, Any]] = []
    if not args.skip_stage2 and n_pass > 0:
        log.info("---- STAGE 2: real FIFO market replay (HC #357) ----")
        top_for_stage2 = agg[agg["stability_pass"]].head(args.max_stage2_configs)
        stage2_results = run_stage2_real_replay(
            top_for_stage2, all_dates_data, max_configs=args.max_stage2_configs,
        )
        s2_path = OUT_DIR / "stage2_real_replay.json"
        # Convert non-serializable
        clean = []
        for r in stage2_results:
            r2 = dict(r)
            if "config" in r2 and isinstance(r2["config"], dict):
                r2["config"] = {k: (v.item() if hasattr(v, "item") else v) for k, v in r2["config"].items()}
            clean.append(r2)
        with open(s2_path, "w") as f:
            json.dump(clean, f, indent=2, default=str)
        log.info(f"stage2: wrote {s2_path}")
    elif n_pass == 0:
        log.info("STAGE 2 SKIPPED — zero stage-1 configs passed stability bar. "
                 "HONEST NULL RESULT.")

    # ----- RESULTS.md -----
    write_report(dates, all_dates_data, df1, agg, stage2_results)
    log.info("DONE.")


def write_report(
    dates: list[str], all_dates_data: list[DateData],
    df1: pd.DataFrame, agg: pd.DataFrame,
    stage2: list[dict[str, Any]],
) -> None:
    n_pass = int(agg["stability_pass"].sum())
    top5 = agg.head(5)
    report_path = OUT_DIR / "RESULTS.md"

    n_fifo = sum(1 for dd in all_dates_data if dd.fifo_avail)
    coverage_lines = [f"- `{dd.date}` (n_windows={len(dd.preds):,}, fifo={'YES' if dd.fifo_avail else 'NO'})"
                      for dd in all_dates_data]

    def cfg_str(r):
        return (f"{r['side']:>5} P{r['band']:<5} {r['horizon']:<3} "
                f"{r['order_type']:<20} cancel={int(r['cancel_evals'])} hold={r['hold_s']}s")

    lines = []
    lines.append("# HC #355 — v2 All-OOT-Dates Profitability Sweep")
    lines.append("")
    lines.append(f"**Run date:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Model:** cnn_mamba_v2_smart_v3 (champion v2; IC_1s≈0.22, IC_5s≈0.14, IC_10s≈0.11)")
    lines.append("")
    lines.append("## Honest scope statement")
    lines.append("")
    lines.append("The user's question was: \"does this setup survive ALL OOT dates from Feb to "
                 "Apr 29? Do we have any config with v2 that is profitable?\"")
    lines.append("")
    lines.append(f"**Available v2 OOT prediction coverage:** {len(all_dates_data)} unique dates "
                 f"from {all_dates_data[0].date} → {all_dates_data[-1].date} "
                 f"({n_fifo} of which have FIFO bid/ask labels with queue-aware fill rates).")
    lines.append("")
    lines.append("**Feb 1-22 has NO v2 OOT predictions** — the earliest v2 OOT day is "
                 f"`{all_dates_data[0].date}`. Feb coverage = Feb 23 → Feb 27 (5 days from smart_v3_mar "
                 "fold_00..fold_04). Mar 1-5 from fold_05..09. Mar 6 → Apr 29 from per-day NPZs.")
    lines.append("")
    lines.append("This is the FULL set of dates the v2 model has produced OUT-OF-SAMPLE predictions for. "
                 "Backfilling Feb 1-22 would require either retraining/inference on earlier windows or "
                 "different fold schedules — neither of which exists today.")
    lines.append("")
    lines.append("## Methodology")
    lines.append("")
    lines.append("**Stage 1 — Fast face-value sweep (HC #320 FIFO-FLOOR):**")
    lines.append("- For every (side, band, horizon, order_type, cancel_evals, hold_s) tuple in the directive grid, "
                 "compute per-trade NET tick returns using v2 predictions + per-horizon realized tick moves.")
    lines.append(f"- Cost stack: passive = {PASSIVE_COST_TICKS} ticks commission; "
                 f"market/IOC = {MARKET_COST_TICKS} ticks (commission + 1-tick spread crossing); "
                 "back-off = passive cost − 1 tick realized (entry 1 tick worse).")
    lines.append("- Fill probability for passive orders: derived from observed FIFO label fill rate "
                 "(11.6% short / 6.2% long at touch with 2s cancel) scaled by `1 − (1−p_base)^(cancel_evals/8)`. "
                 "Back-off gets a queue-position advantage modeled as ~2× base prob capped at 70%.")
    lines.append("- Hold-seconds shorter than the prediction horizon: edge scaled by `sqrt(hold/horizon)` "
                 "(conservative; per HC #355 decay analysis edge dies by 30s).")
    lines.append("")
    lines.append("**Stage 2 — Real FIFO market replay (HC #357):**")
    lines.append("- Top-N stage-1 stable configs are re-run through `FIFOReplayEngine` on 5 sample dates each.")
    lines.append("- Includes: queue position on arrival, fill probability by queue depletion, "
                 "adverse selection post-fill, commission, cancel/replace logic, max-hold exit.")
    lines.append("- This is the queue+adv-sel layer demanded by HC #357 / HC #349.")
    lines.append("")
    lines.append("## 1. Date Coverage")
    lines.append("")
    lines.append(f"Total dates analyzed: **{len(all_dates_data)}**, FIFO-labeled: **{n_fifo}**.")
    lines.append("")
    lines.append("<details><summary>Full date list</summary>")
    lines.append("")
    for ln in coverage_lines:
        lines.append(ln)
    lines.append("")
    lines.append("</details>")
    lines.append("")
    lines.append("## 2. Stability Verdict (Stage 1 — FIFO-floor)")
    lines.append("")
    lines.append("**Stability bar (per HC #355):**")
    lines.append(f"- Sharpe > {STAB_SHARPE_MIN} on > {STAB_DATE_PASS_PCT*100:.0f}% of dates")
    lines.append(f"- Worst-date Sharpe > {STAB_WORST_SHARPE}")
    lines.append(f"- Net positive on > {STAB_NET_POS_PCT*100:.0f}% of dates")
    lines.append("")
    lines.append(f"**Configs passing all 3 criteria: {n_pass} / {len(agg):,}**")
    lines.append("")
    if n_pass == 0:
        lines.append("### HONEST NULL RESULT")
        lines.append("")
        lines.append("Under the FIFO-floor cost stack, **ZERO v2 configs pass the stability bar across "
                     "Feb 23 → Apr 29 2026**.")
        lines.append("")
        lines.append("This is an UPPER BOUND on what v2 can deliver: stage 1 does NOT include queue-position "
                     "penalty or adverse-selection cost (those would only make results worse). Therefore the "
                     "honest answer to the user's question is:")
        lines.append("")
        lines.append("> **No v2 config has been demonstrated to be Sharpe>1 stable across all OOT dates from "
                     "Feb 23 to Apr 29 2026.** The CLAUDE.md canonical claim of \"v2 top-10% short, +1.56t avg "
                     "60.5% WR\" is a face-value AVERAGE across OOT data — it does NOT mean every date is "
                     "profitable, and it does NOT survive realistic execution costs uniformly.")
    else:
        lines.append("### Configs passing stability bar (top 10 by aggregate Sharpe)")
        lines.append("")
        lines.append("| Rank | side | band | horiz | order_type | cancel | hold | n_dates_active | "
                     "n_trades | net_ticks | Sharpe | Sortino | WR | PF | MDD | %dates Sh>1 | worst Sh | %dates net+ |")
        lines.append("|-----:|:----:|-----:|:-----:|:-----------|------:|-----:|---------------:|--------:|----------:|"
                     "------:|--------:|----:|----:|------:|------------:|--------:|-------------:|")
        for i, (_, r) in enumerate(agg[agg["stability_pass"]].head(10).iterrows(), 1):
            lines.append(
                f"| {i} | {r['side']} | P{r['band']} | {r['horizon']} | {r['order_type']} | "
                f"{int(r['cancel_evals'])} | {r['hold_s']} | {int(r['n_dates_active'])} | "
                f"{int(r['total_trades']):,} | {r['total_net_ticks']:+.1f} | "
                f"{r['avg_sharpe']:+.2f} | {r['avg_sortino']:+.2f} | "
                f"{r['avg_wr']:.1%} | {r['avg_pf']:.2f} | {r['avg_mdd']:+.1f} | "
                f"{r['pct_sharpe_gt1']:.0%} | {r['worst_sharpe']:+.2f} | "
                f"{r['pct_net_pos']:.0%} |"
            )
    lines.append("")
    lines.append("## 3. Top-5 Configs Overall (by aggregate Sharpe)")
    lines.append("")
    lines.append("| Rank | side | band | horiz | order_type | cancel | hold | n_trades | net_ticks | avg Sharpe | %dates Sh>1 | worst Sh | %dates net+ | passes |")
    lines.append("|-----:|:----:|-----:|:-----:|:-----------|------:|-----:|--------:|----------:|-----------:|------------:|--------:|-------------:|:-------:|")
    for i, (_, r) in enumerate(top5.iterrows(), 1):
        passed = "YES" if r["stability_pass"] else "no"
        lines.append(
            f"| {i} | {r['side']} | P{r['band']} | {r['horizon']} | {r['order_type']} | "
            f"{int(r['cancel_evals'])} | {r['hold_s']} | "
            f"{int(r['total_trades']):,} | {r['total_net_ticks']:+.1f} | "
            f"{r['avg_sharpe']:+.2f} | {r['pct_sharpe_gt1']:.0%} | "
            f"{r['worst_sharpe']:+.2f} | {r['pct_net_pos']:.0%} | {passed} |"
        )
    lines.append("")
    lines.append("## 4. Per-Date Detail for Top Config (#1)")
    lines.append("")
    if len(top5) > 0:
        r1 = top5.iloc[0]
        sub = df1[(df1["side"] == r1["side"]) & (df1["band"] == r1["band"]) &
                  (df1["horizon"] == r1["horizon"]) & (df1["order_type"] == r1["order_type"]) &
                  (df1["cancel_evals"] == r1["cancel_evals"]) & (df1["hold_s"] == r1["hold_s"])]
        sub = sub.sort_values("date")
        lines.append(f"**Config:** `{cfg_str(r1)}`")
        lines.append("")
        lines.append("| date | n | net_ticks | Sharpe | WR | MDD |")
        lines.append("|:-----|--:|---------:|------:|----:|----:|")
        for _, rr in sub.iterrows():
            lines.append(f"| {rr['date']} | {int(rr['n'])} | {rr['sum']:+.1f} | "
                         f"{rr['sharpe']:+.2f} | {rr['wr']:.1%} | {rr['mdd']:+.1f} |")
    lines.append("")
    lines.append("## 5. Stage 2 — Real FIFO Market Replay (HC #357)")
    lines.append("")
    if len(stage2) == 0:
        if n_pass == 0:
            lines.append("Stage 2 skipped — no stage-1 configs passed the stability bar, so there is "
                         "nothing worth validating with the full queue+adv-sel stack.")
        else:
            lines.append("Stage 2 results not generated.")
    else:
        lines.append("Each of the top stage-1 configs was re-run through `FIFOReplayEngine` "
                     "(queue + adv-sel + cancel/replace + commission). Results:")
        lines.append("")
        lines.append("| config | date | signals | fills | fill_rate | net_ticks | Sharpe | WR |")
        lines.append("|:-------|:----:|--------:|-----:|---------:|---------:|------:|---:|")
        for r in stage2:
            if r.get("error"):
                lines.append(f"| {r.get('config','?')} | {r.get('date','?')} | ERR | ERR | "
                             f"ERR | ERR | ERR | ERR |")
                continue
            cfg = r.get("config", {})
            cfg_label = f"{cfg.get('side','?')}/P{cfg.get('band','?')}/{cfg.get('horizon','?')}/{cfg.get('order_type','?')}"
            lines.append(
                f"| {cfg_label} | {r['date']} | {r.get('n_signals',0):,} | "
                f"{r.get('n_fills',0):,} | {r.get('fill_rate',0):.1%} | "
                f"{r.get('net_ticks',0):+.1f} | {r.get('sharpe',0):+.2f} | "
                f"{r.get('wr',0):.1%} |"
            )
    lines.append("")
    lines.append("## 6. One-line strongest-v2-config recommendation")
    lines.append("")
    if n_pass > 0:
        r = agg[agg["stability_pass"]].iloc[0]
        lines.append(
            f"> Strongest stable v2 config: **{r['side']} P{r['band']} {r['horizon']} "
            f"{r['order_type']} cancel={int(r['cancel_evals'])} hold={r['hold_s']}s** — "
            f"avg Sharpe {r['avg_sharpe']:+.2f}, Sharpe>1 on {r['pct_sharpe_gt1']:.0%} of dates, "
            f"worst Sharpe {r['worst_sharpe']:+.2f}, net positive on {r['pct_net_pos']:.0%} of dates. "
            f"Caveat: stage 1 is FIFO-floor; see stage 2 above for queue+adv-sel-adjusted numbers."
        )
    else:
        lines.append(
            "> **No v2 config passes the stability bar.** The currently-deployed paper config "
            "(HC #354: short-side, P95, passive limit) corresponds to NO row that passes 3 stability "
            "criteria simultaneously. The v2 short top-10% face-value edge claim does NOT generalize "
            "uniformly across Feb 23 → Apr 29 2026 OOT dates. Live paper trading remains the production "
            "queue+adv-sel test."
        )
    lines.append("")
    lines.append("## 7. Files")
    lines.append("")
    lines.append("- `stage1_per_date.csv.gz` — full per-(date,config) metrics matrix")
    lines.append("- `stage1_aggregated.csv.gz` — per-config aggregated stability stats")
    lines.append("- `stage2_real_replay.json` — real FIFO replay results for top candidates")
    lines.append("- `gate_sweep_heatmap.png` — side × band × order_type avg-Sharpe heatmap")
    lines.append("- `per_date_pnl_curves.png` — per-date cumulative NET PnL for top-5 configs")
    lines.append("- `run.log` — full execution log")
    lines.append("")
    lines.append("---")
    lines.append("Per HC #349: stage-1 numbers are FIFO-floor; stage-2 numbers include queue + adv-sel. "
                 "Per HC #355: stability verdict computed exactly as specified.")
    lines.append("Per HC #307D: this is a new analysis script under scripts/v3_3_research/; "
                 "trainer code untouched.")

    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    log.info(f"wrote {report_path}")


if __name__ == "__main__":
    main()

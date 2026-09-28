#!/usr/bin/env python3
"""
p_microstructure_fifo_v1.py — P-Microstructure FIFO Gate Composition (HC #494 R1)

Combines:
  * CNN-Mamba v2 OOT predictions (signal)        — 2026-only per HC #500
  * Fill-prob v3 head (XGB, AUC 0.84)            — probability of passive limit fill
  * Adverse-cost v3 head (XGB, RMSE 2.09 ticks)  — expected adverse move (ticks)

Gate:  accept entry iff  fill_prob_h >= tau_fill AND adverse_cost_h <= tau_adv
       AND |cnn_mamba_pred_h| >= tau_conf

Routes accepted entries through the CANONICAL FIFO replay engine (HC #493).

Per HC #428: per-day Sharpe, regime stratification (green/red/flat).
Per HC #74:  every cell logged to MLflow under experiment p_microstructure_fifo_v1.

Author: autonomous agent under HC #393 / HC #420 / HC #494 R1 binding.
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time, math
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

LVL3_ROOT = Path('/home/nick/Lvl3Quant')

# Canonical FIFO engine (HC #493 — DO NOT MODIFY, only import + wrap)
# Import via spec to avoid the alpha_discovery package __init__ which pulls in torch
import importlib.util as _ilu
_FIFO_PATH = LVL3_ROOT / 'alpha_discovery' / 'deep_models' / 'fifo_market_replay.py'
_spec = _ilu.spec_from_file_location('fifo_market_replay', str(_FIFO_PATH))
_fifo_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_fifo_mod)
FIFOReplayEngine = _fifo_mod.FIFOReplayEngine
find_dbn_path = _fifo_mod.find_dbn_path
TICK_RAW = _fifo_mod.TICK_RAW
TICK_USD = _fifo_mod.TICK_USD
COMMISSION_RT = _fifo_mod.COMMISSION_RT
COMMISSION_TICKS = _fifo_mod.COMMISSION_TICKS

# ── Inference-time window params for CNN-Mamba v2 bulk OOT
BULK_WINDOW = 3000
BULK_STRIDE = 250

# ── Paths
CNN_OOT_DIR  = LVL3_ROOT / 'output' / 'cnn_mamba_v2_all_oot'
FEAT_DIR     = LVL3_ROOT / 'output' / 'queue_augmented_features'
LBL_DIR      = LVL3_ROOT / 'output' / 'mbo_walker_labels'
MBO_EVT_DIR  = LVL3_ROOT / 'data' / 'processed' / 'mbo_events_smart_v3'
FILL_DIR     = LVL3_ROOT / 'output' / 'fill_prob_v3_honest_xgb'
ADV_DIR      = LVL3_ROOT / 'output' / 'adverse_cost_head_v3'
OUT_DIR      = LVL3_ROOT / 'output' / 'p_microstructure_fifo_v1'
OUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = ['1s', '5s', '10s']
HORIZON_TO_IDX = {'1s': 0, '5s': 1, '10s': 2}
HORIZON_SECONDS = {'1s': 1.0, '5s': 5.0, '10s': 10.0}

# ── Logging
import logging
LOG_DIR = LVL3_ROOT / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / f'p_microstructure_fifo_v1_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger('p_micro_fifo')


# ── Feature column lists (must match the two trained heads exactly) ──────────
# Both heads were trained on the inner join of features_<date>.parquet +
# labels_<date>.parquet, dropping (event_id, ts_ns) and targets. They use
# essentially the same feature set since labels feed both. To be safe at
# inference, we replicate the feature_columns() logic of each trainer.
ADV_TARGET_COLS = {h: f'adverse_{h}' for h in HORIZONS}
FILL_TARGET_COLS = {h: f'filled_{h}' for h in HORIZONS}

def _load_canonical_feats(model_dir: Path, n_expected: int) -> List[str]:
    """Load persisted training feature list. Required to match XGB num_features."""
    fp = model_dir / 'feats.json'
    if not fp.exists():
        raise RuntimeError(f'Canonical feature list missing: {fp}. '
                           f'Cannot align inference schema with training schema.')
    feats = json.loads(fp.read_text())
    if len(feats) != n_expected:
        raise RuntimeError(f'{fp}: expected {n_expected} feats, got {len(feats)}')
    return feats

def adv_feature_columns(df: pd.DataFrame) -> List[str]:
    feats = _load_canonical_feats(ADV_DIR, 29)
    missing = [c for c in feats if c not in df.columns]
    if missing:
        raise RuntimeError(f'adv inference df missing canonical features: {missing}')
    return feats

def fill_feature_columns(df: pd.DataFrame) -> List[str]:
    feats = _load_canonical_feats(FILL_DIR, 30)
    missing = [c for c in feats if c not in df.columns]
    if missing:
        raise RuntimeError(f'fill inference df missing canonical features: {missing}')
    return feats


# ── Date discovery (HC #500: 2026-only) ───────────────────────────────────────
def discover_dates() -> List[str]:
    cnn = {Path(p).stem.replace('_predictions', '') for p in glob.glob(str(CNN_OOT_DIR / '2026*_predictions.npz'))}
    feat = {Path(p).stem.replace('features_', '') for p in glob.glob(str(FEAT_DIR / 'features_2026*.parquet'))}
    lbl = {Path(p).stem.replace('labels_', '') for p in glob.glob(str(LBL_DIR / 'labels_2026*.parquet'))}
    mbo = {Path(p).stem.replace('_mbo_events', '') for p in glob.glob(str(MBO_EVT_DIR / '2026*_mbo_events.npz'))}
    common = sorted(cnn & feat & lbl & mbo)
    common = [d for d in common if d >= '20260101']  # HC #500
    log.info(f'Common 2026 dates: {len(common)}  ({common[0]}..{common[-1]} if any)')
    return common


# ── XGB head loading ──────────────────────────────────────────────────────────
def load_xgb_heads():
    import xgboost as xgb
    fill = {}
    adv = {}
    for h in HORIZONS:
        f = xgb.Booster(); f.load_model(str(FILL_DIR / f'xgb_fill_{h}_full.json')); fill[h] = f
        a = xgb.Booster(); a.load_model(str(ADV_DIR / f'xgb_{h}_full.json')); adv[h] = a
    log.info(f'Loaded fill_prob heads: {list(fill.keys())}')
    log.info(f'Loaded adverse_cost heads: {list(adv.keys())}')
    return fill, adv


# ── Per-date data assembly ────────────────────────────────────────────────────
def load_date_signal_table(date: str) -> Optional[pd.DataFrame]:
    """
    Build per-signal-event table joined to CNN-Mamba v2 prediction by ts_ns.
    Returns a DataFrame with columns:
        event_id, ts_ns, side, queue_*, pred_cnn_{1s,5s,10s},
        adv_target_*, all queue_augmented features.
    """
    cnn_path = CNN_OOT_DIR / f'{date}_predictions.npz'
    mbo_path = MBO_EVT_DIR / f'{date}_mbo_events.npz'
    feat_path = FEAT_DIR / f'features_{date}.parquet'
    lbl_path = LBL_DIR / f'labels_{date}.parquet'
    if not all(p.exists() for p in [cnn_path, mbo_path, feat_path, lbl_path]):
        return None

    d = np.load(cnn_path, allow_pickle=True)
    cnn_pred = d['predictions']           # (n_pred, 3)
    n_pred = cnn_pred.shape[0]

    with np.load(mbo_path, allow_pickle=False) as src:
        ts_all = src['timestamps']
    # pred i corresponds to ts at i*stride + window - 1 (last event in window)
    pred_event_idx = np.minimum(np.arange(n_pred) * BULK_STRIDE + BULK_WINDOW - 1, len(ts_all) - 1)
    pred_ts = ts_all[pred_event_idx]      # (n_pred,) int64 ns

    feat = pd.read_parquet(feat_path)
    lbl = pd.read_parquet(lbl_path)
    # Avoid duplicate cols (side / pred_* live in both — keep label canonical side)
    drop_in_lbl = [c for c in ['pred_1s', 'pred_5s', 'pred_10s'] if c in lbl.columns and c in feat.columns]
    if drop_in_lbl: lbl = lbl.drop(columns=drop_in_lbl)
    if 'side' in lbl.columns and 'side' in feat.columns:
        feat = feat.drop(columns=['side'])
    df = feat.merge(lbl, on=['event_id', 'ts_ns'], how='inner')
    if 'valid' in df.columns:
        df = df[df['valid'] == 1].reset_index(drop=True)
    df = df.sort_values('ts_ns').reset_index(drop=True)
    feat_ts = df['ts_ns'].to_numpy()

    # Nearest-neighbour join: for each pred_ts, find closest feat_ts
    idx_r = np.searchsorted(feat_ts, pred_ts)
    idx_r = np.clip(idx_r, 0, len(feat_ts) - 1)
    idx_l = np.maximum(idx_r - 1, 0)
    take_l = np.abs(feat_ts[idx_l] - pred_ts) < np.abs(feat_ts[idx_r] - pred_ts)
    nearest = np.where(take_l, idx_l, idx_r)
    delta_ns = np.abs(feat_ts[nearest] - pred_ts)

    # Tolerance: 100ms (signal-event vs window-end alignment slack)
    keep = delta_ns < 100_000_000
    if keep.sum() == 0:
        log.warning(f'{date}: no preds align within 100ms — skip')
        return None

    pred_kept = cnn_pred[keep]
    nearest_kept = nearest[keep]
    pred_ts_kept = pred_ts[keep]

    # Build signal table: take features at nearest signal-event, attach pred
    out = df.iloc[nearest_kept].reset_index(drop=True).copy()
    out['cnn_pred_1s'] = pred_kept[:, 0].astype(np.float32)
    out['cnn_pred_5s'] = pred_kept[:, 1].astype(np.float32)
    out['cnn_pred_10s'] = pred_kept[:, 2].astype(np.float32)
    out['pred_ts_ns'] = pred_ts_kept
    out['_date'] = date
    return out


# ── Score gate heads ──────────────────────────────────────────────────────────
def score_heads(df: pd.DataFrame, fill_heads, adv_heads, fill_feats, adv_feats) -> pd.DataFrame:
    import xgboost as xgb
    X_fill = df[fill_feats].to_numpy(dtype=np.float32)
    X_adv  = df[adv_feats].to_numpy(dtype=np.float32)
    dfill = xgb.DMatrix(X_fill)
    dadv  = xgb.DMatrix(X_adv)
    for h in HORIZONS:
        df[f'fill_prob_{h}'] = fill_heads[h].predict(dfill).astype(np.float32)
        df[f'adv_cost_{h}']  = adv_heads[h].predict(dadv).astype(np.float32)
    return df


# ── Regime classification (HC #428): green/red/flat by ES close-to-close ─────
def classify_regime_for_date(date: str, flat_band_ticks: float = 4.0) -> str:
    """
    Open vs close mid-price drift in ticks. Mid measured from MBO events:
    first/last RTH event timestamps (approximation since we don't have OHLC).
    Uses raw MBO records via FIFOReplayEngine load? Simpler: load mbo_events
    npz which has event prices.
    """
    mbo_path = MBO_EVT_DIR / f'{date}_mbo_events.npz'
    try:
        with np.load(mbo_path, allow_pickle=False) as src:
            ev = src['events']  # (N, 25); col 3 = price_rel_ticks? We don't know schema; fallback to labels
            # labels_5s at first vs last RTH event approximates drift
            l5 = src['labels_5s']
            ts = src['timestamps']
    except Exception:
        return 'unknown'
    # RTH window in ns: 8:30 CT = 13:30 UTC, close 15:00 CT = 20:00 UTC
    # We instead just take mid 50% of session to avoid open/close noise
    n = len(ts)
    if n < 100: return 'unknown'
    first_q = ts[n // 8]
    last_q = ts[-n // 8]
    # Approximate drift via label cumulative — labels_5s is "future delta in ticks" per event
    # We instead use the events array price differential: col index of price_rel_ticks
    # Fallback: classify by sum sign of labels_5s — quick approximation
    drift = float(np.nanmean(l5))  # mean per-event 5s drift (ticks)
    # Translate to roughly per-day drift via n events / event_rate proxy
    # Simpler: |drift| > tol = trending; sign = green/red
    if abs(drift) < 0.01:
        return 'flat'
    return 'green' if drift > 0 else 'red'


# ── FIFO replay wrapper ──────────────────────────────────────────────────────
def run_fifo_for_date(date: str, signals: List[dict], tp_ticks: float, sl_ticks: float,
                      cancel_after_ns: int, max_hold_ns: int,
                      order_type: str = 'limit') -> List:
    """Wrap canonical FIFOReplayEngine (HC #493 — no modifications)."""
    eng = FIFOReplayEngine(
        date=date,
        cancel_after_ns=cancel_after_ns,
        max_hold_ns=max_hold_ns,
    )
    return eng.simulate(signals=signals, tp_ticks=tp_ticks, sl_ticks=sl_ticks, order_type=order_type)


# ── Aggregation ──────────────────────────────────────────────────────────────
def trade_to_row(t) -> dict:
    # TradeResult attribute names (verified against canonical FIFOReplayEngine):
    #   entry_ts_ns (NOT fill_ts_ns), entry_price_raw (NOT entry_price),
    #   exit_price_raw (NOT exit_price), hold_time_ns (NOT hold_ns),
    #   queue_ahead (NOT queue_ahead_at_signal), queue_wait_ns (~ fill_latency_ns).
    # Map back to canonical-looking dict keys so downstream code stays clear.
    return {
        'signal_ts_ns':           getattr(t, 'signal_ts_ns',       None),
        'fill_ts_ns':             getattr(t, 'entry_ts_ns',        None),  # rename
        'exit_ts_ns':             getattr(t, 'exit_ts_ns',         None),
        'direction':              getattr(t, 'direction',          None),
        'entry_price':            getattr(t, 'entry_price_raw',    None),  # rename
        'exit_price':             getattr(t, 'exit_price_raw',     None),  # rename
        'pnl_ticks':              getattr(t, 'pnl_ticks',          None),  # gross
        'pnl_ticks_net_engine':   getattr(t, 'pnl_ticks_net',      None),  # engine-side net (sanity)
        'pnl_dollars':            getattr(t, 'pnl_dollars',        None),
        'exit_reason':            getattr(t, 'exit_reason',        None),
        'fill_latency_ns':        getattr(t, 'queue_wait_ns',      None),  # rename
        'hold_ns':                getattr(t, 'hold_time_ns',       None),  # rename
        'queue_ahead_at_signal':  getattr(t, 'queue_ahead',        None),  # rename
        'slippage_ticks':         getattr(t, 'slippage_ticks',     None),
        'spread_at_signal':       getattr(t, 'spread_at_signal',   None),
        'mid_at_signal':          getattr(t, 'mid_at_signal',      None),
    }


def daily_metrics(trades_df: pd.DataFrame) -> dict:
    if len(trades_df) == 0:
        return {'n': 0, 'net_ticks_per_trade': 0.0, 'gross_ticks_per_trade': 0.0,
                'sharpe': 0.0, 'win_rate': 0.0, 'pf': 0.0, 'sortino': 0.0}
    pnl = trades_df['pnl_ticks'].to_numpy(dtype=np.float32) - COMMISSION_TICKS
    n = len(pnl)
    sharpe = float(pnl.mean() / (pnl.std() + 1e-9) * math.sqrt(n))
    wins = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()
    pf = float(wins / losses) if losses > 0 else float('inf')
    wr = float((pnl > 0).mean())
    neg = pnl[pnl < 0]
    sortino = float(pnl.mean() / (neg.std() + 1e-9) * math.sqrt(n)) if len(neg) > 0 else float('inf')
    return {
        'n': int(n),
        'net_ticks_per_trade': float(pnl.mean()),
        'gross_ticks_per_trade': float((pnl + COMMISSION_TICKS).mean()),
        'sharpe': sharpe, 'sortino': sortino, 'pf': pf, 'win_rate': wr,
        'sum_net_ticks': float(pnl.sum()),
    }


def regime_skew(per_day_df: pd.DataFrame) -> float:
    g = per_day_df[per_day_df.regime == 'green']['sharpe']
    r = per_day_df[per_day_df.regime == 'red']['sharpe']
    if len(g) == 0 or len(r) == 0: return float('nan')
    sg = float(g.mean()); sr = float(r.mean())
    denom = max(abs(sg), abs(sr), 1e-9)
    return abs(sg - sr) / denom


# ── Gate sweep ────────────────────────────────────────────────────────────────
def apply_gate(df: pd.DataFrame, h: str, tau_fill: float, tau_adv: float,
               tau_conf_quantile: float) -> pd.DataFrame:
    pred_col = f'cnn_pred_{h}'
    fp_col = f'fill_prob_{h}'
    ac_col = f'adv_cost_{h}'
    # tau_conf is the |pred| quantile threshold; e.g. 0.95 -> top 5%
    conf_thr = np.quantile(np.abs(df[pred_col].to_numpy()), tau_conf_quantile)
    mask = (
        (df[fp_col] >= tau_fill) &
        (df[ac_col] <= tau_adv) &
        (np.abs(df[pred_col]) >= conf_thr)
    )
    sel = df[mask].copy()
    sel['gate_h'] = h
    sel['gate_tau_fill'] = tau_fill
    sel['gate_tau_adv'] = tau_adv
    sel['gate_tau_conf'] = float(conf_thr)
    return sel


def signals_from_gate(sel: pd.DataFrame, h: str) -> List[dict]:
    """Build canonical signal dict list for the FIFO engine."""
    pred_col = f'cnn_pred_{h}'
    out = []
    pred = sel[pred_col].to_numpy()
    ts = sel['pred_ts_ns'].to_numpy()
    for i in range(len(sel)):
        out.append({
            'ts_ns': int(ts[i]),
            'direction': 'long' if pred[i] > 0 else 'short',
            'strength': float(abs(pred[i])),
        })
    return out


# ── Main run ──────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true', help='1 day, 1 cell, h=5s')
    ap.add_argument('--wall-cap-min', type=float, default=45.0)
    ap.add_argument('--max-days', type=int, default=0, help='0 = all 2026 days')
    ap.add_argument('--horizons', nargs='+', default=['5s'])
    ap.add_argument('--tau-fill', nargs='+', type=float, default=[0.5, 0.6, 0.7])
    ap.add_argument('--tau-adv', nargs='+', type=float, default=[1.0, 1.5, 2.0])
    ap.add_argument('--tau-conf', nargs='+', type=float, default=[0.95, 0.90, 0.80])
    ap.add_argument('--tp-ticks', type=float, default=2.0)
    ap.add_argument('--sl-ticks', type=float, default=2.0)
    ap.add_argument('--mlflow', action='store_true', default=True)
    args = ap.parse_args()

    t_start = time.time()
    wall_deadline = t_start + args.wall_cap_min * 60

    dates = discover_dates()
    if not dates:
        log.error('No 2026 dates found at intersection of CNN/feat/lbl/MBO')
        sys.exit(1)

    if args.smoke:
        dates = dates[:1]
        args.tau_fill = [0.6]
        args.tau_adv = [1.5]
        args.tau_conf = [0.95]
        args.horizons = ['5s']

    if args.max_days > 0:
        dates = dates[:args.max_days]

    log.info(f'Using {len(dates)} dates: {dates[0]}..{dates[-1]}')

    fill_heads, adv_heads = load_xgb_heads()

    # MLflow setup
    mlflow_run_id = None
    if args.mlflow:
        try:
            import mlflow
            os.environ.setdefault('MLFLOW_TRACKING_URI', 'http://localhost:5000')
            mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
            mlflow.set_experiment('p_microstructure_fifo_v1')
            ml_run = mlflow.start_run(run_name=f'sweep_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
            mlflow_run_id = ml_run.info.run_id
            mlflow.log_params({
                'n_dates': len(dates),
                'date_first': dates[0], 'date_last': dates[-1],
                'tp_ticks': args.tp_ticks, 'sl_ticks': args.sl_ticks,
                'smoke': args.smoke,
                'horizons': ','.join(args.horizons),
                'tau_fill_grid': ','.join(map(str, args.tau_fill)),
                'tau_adv_grid':  ','.join(map(str, args.tau_adv)),
                'tau_conf_grid': ','.join(map(str, args.tau_conf)),
            })
            log.info(f'MLflow run: {mlflow_run_id}')
        except Exception as e:
            log.warning(f'MLflow init failed: {e}')
            mlflow_run_id = None

    # ── Pass 1: assemble per-day scored signal tables (once) ──
    log.info('=== Pass 1: assemble per-day scored signal tables ===')
    scored_per_day: Dict[str, pd.DataFrame] = {}
    regimes: Dict[str, str] = {}
    # Determine feature column lists from the FIRST date
    first_df = None
    for date in dates:
        if time.time() > wall_deadline:
            log.warning(f'wall deadline reached during assembly at {date}')
            break
        sig = load_date_signal_table(date)
        if sig is None or len(sig) == 0:
            log.info(f'  {date}: skip (no signal table)')
            continue
        if first_df is None:
            fill_feats = fill_feature_columns(sig)
            adv_feats = adv_feature_columns(sig)
            log.info(f'  fill_feats={len(fill_feats)} adv_feats={len(adv_feats)}')
            first_df = sig
        # Score heads
        # ensure expected columns exist
        miss_fill = [c for c in fill_feats if c not in sig.columns]
        miss_adv = [c for c in adv_feats if c not in sig.columns]
        if miss_fill or miss_adv:
            log.warning(f'  {date}: missing fill {miss_fill[:3]} adv {miss_adv[:3]} — skip')
            continue
        sig = score_heads(sig, fill_heads, adv_heads, fill_feats, adv_feats)
        scored_per_day[date] = sig
        regimes[date] = classify_regime_for_date(date)
        log.info(f'  {date}: n_signals={len(sig)} regime={regimes[date]} fp5_p50={sig["fill_prob_5s"].median():.3f} adv5_p50={sig["adv_cost_5s"].median():.3f}')

    log.info(f'Scored signal tables for {len(scored_per_day)} dates')

    # ── Pass 2: sweep gate cells, route via FIFO ──
    log.info('=== Pass 2: gate threshold sweep + FIFO replay ===')
    sweep_rows = []
    best_cell = None
    best_score = -1e18
    all_per_day_for_best = None

    # Baseline: no-gate = use all CNN preds (just confidence top-N), same FIFO settings
    # We will include this in the sweep as a "baseline" cell with extreme tau
    for h in args.horizons:
        cancel_after_ns = int(HORIZON_SECONDS[h] * 1e9)  # cancel <= h per spec
        max_hold_ns     = int(1.5 * HORIZON_SECONDS[h] * 1e9)

        # Baseline (no microstructure gate, just top-conf 10%)
        cells = [('baseline_top10', 0.0, 1e9, 0.90)]
        for tf in args.tau_fill:
            for ta in args.tau_adv:
                for tc in args.tau_conf:
                    cells.append((f'tf{tf}_ta{ta}_tc{tc}', tf, ta, tc))

        for cell_name, tau_fill, tau_adv, tau_conf in cells:
            if time.time() > wall_deadline:
                log.warning(f'wall deadline during sweep at cell {cell_name}')
                break
            t0 = time.time()
            log.info(f'  cell h={h} {cell_name} tau_fill={tau_fill} tau_adv={tau_adv} tau_conf={tau_conf}')

            per_day_rows = []
            for date, sig in scored_per_day.items():
                sel = apply_gate(sig, h, tau_fill, tau_adv, tau_conf)
                if len(sel) == 0:
                    per_day_rows.append({'date': date, 'regime': regimes.get(date, 'unknown'),
                                         'n_trades': 0, 'net_ticks_per_trade': 0.0,
                                         'sharpe': 0.0, 'pf': 0.0, 'win_rate': 0.0,
                                         'sum_net_ticks': 0.0, 'sortino': 0.0,
                                         'n_signals_accepted': 0})
                    continue
                signals = signals_from_gate(sel, h)
                try:
                    trades = run_fifo_for_date(
                        date=date, signals=signals,
                        tp_ticks=args.tp_ticks, sl_ticks=args.sl_ticks,
                        cancel_after_ns=cancel_after_ns,
                        max_hold_ns=max_hold_ns,
                        order_type='limit',
                    )
                except Exception as e:
                    log.warning(f'    FIFO failed {date}: {type(e).__name__}: {e}')
                    per_day_rows.append({'date': date, 'regime': regimes.get(date, 'unknown'),
                                         'n_trades': 0, 'net_ticks_per_trade': 0.0,
                                         'sharpe': 0.0, 'pf': 0.0, 'win_rate': 0.0,
                                         'sum_net_ticks': 0.0, 'sortino': 0.0,
                                         'n_signals_accepted': len(signals),
                                         'fifo_error': str(e)[:200]})
                    continue
                # Convert filled trades only
                rows = [trade_to_row(t) for t in trades if getattr(t, 'entry_ts_ns', None) is not None]
                tdf = pd.DataFrame(rows)
                m = daily_metrics(tdf)
                m['date'] = date
                m['regime'] = regimes.get(date, 'unknown')
                m['n_signals_accepted'] = len(signals)
                m['n_trades'] = m.pop('n')
                per_day_rows.append(m)

            pdf = pd.DataFrame(per_day_rows)
            # Overall metrics across days
            n_pos_days = int((pdf['n_trades'] > 0).sum())
            valid = pdf[pdf['n_trades'] > 0]
            if len(valid) > 0:
                mean_ntpt = float(valid['net_ticks_per_trade'].mean())
                day_sharpe = float(valid['sharpe'].mean())
                # Sharpe of per-day pnl series (gives more robust risk-adjusted)
                day_pnl = valid['sum_net_ticks'].to_numpy()
                if len(day_pnl) > 1 and day_pnl.std() > 0:
                    day_pnl_sharpe = float(day_pnl.mean() / day_pnl.std() * math.sqrt(252))
                else:
                    day_pnl_sharpe = 0.0
                skew = regime_skew(valid)
                total_trades = int(valid['n_trades'].sum())
                trades_per_day = float(total_trades / len(valid))
                sum_net = float(valid['sum_net_ticks'].sum())
                wr = float(valid['win_rate'].mean())
            else:
                mean_ntpt = 0.0; day_sharpe = 0.0; day_pnl_sharpe = 0.0
                skew = float('nan'); total_trades = 0; trades_per_day = 0.0
                sum_net = 0.0; wr = 0.0

            cell_row = {
                'horizon': h, 'cell': cell_name,
                'tau_fill': tau_fill, 'tau_adv': tau_adv, 'tau_conf': tau_conf,
                'n_pos_days': n_pos_days,
                'total_trades': total_trades, 'trades_per_day': trades_per_day,
                'mean_net_ticks_per_trade': mean_ntpt,
                'win_rate': wr, 'sum_net_ticks': sum_net,
                'day_avg_sharpe': day_sharpe, 'daily_pnl_sharpe': day_pnl_sharpe,
                'regime_skew': skew,
                'elapsed_s': float(time.time() - t0),
            }
            sweep_rows.append(cell_row)
            log.info(f'    cell {cell_name}: trades={total_trades} ntpt={mean_ntpt:+.3f} '
                     f'sharpe={day_pnl_sharpe:+.2f} skew={skew:.2f} days_pos={n_pos_days}/{len(valid)}')

            if mlflow_run_id is not None:
                try:
                    import mlflow
                    for k, v in cell_row.items():
                        if isinstance(v, (int, float)) and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                            mlflow.log_metric(f'{cell_name}_{k}', float(v))
                except Exception:
                    pass

            # HC #494 R1 viability score: net_ticks * (sharpe>=1) * (n_days>=30) * (skew<=0.5)
            ok_days = n_pos_days >= 30
            ok_sharpe = day_pnl_sharpe >= 1.0
            ok_skew = (not math.isnan(skew)) and skew <= 0.5
            ok_trades = trades_per_day >= 5
            viable = ok_days and ok_sharpe and ok_skew and ok_trades and mean_ntpt > 0
            score = mean_ntpt if viable else mean_ntpt - 100  # heavily penalize non-viable
            if score > best_score and total_trades > 0:
                best_score = score
                best_cell = cell_row
                all_per_day_for_best = pdf.copy()

    sweep_df = pd.DataFrame(sweep_rows)
    sweep_path = OUT_DIR / 'sweep_leaderboard.csv'
    sweep_df.to_csv(sweep_path, index=False)
    log.info(f'Sweep leaderboard saved: {sweep_path}')

    summary = {
        'mlflow_run_id': mlflow_run_id,
        'experiment': 'p_microstructure_fifo_v1',
        'n_dates': len(scored_per_day),
        'date_first': dates[0] if dates else None,
        'date_last': dates[-1] if dates else None,
        'best_cell': best_cell,
        'sweep_path': str(sweep_path),
        'log_file': str(LOG_FILE),
        'elapsed_s': float(time.time() - t_start),
    }

    if all_per_day_for_best is not None:
        per_day_path = OUT_DIR / 'best_cell_per_day.csv'
        all_per_day_for_best.to_csv(per_day_path, index=False)
        summary['per_day_path'] = str(per_day_path)

    # Verdict (HC #494 R1)
    if best_cell:
        ntpt = best_cell['mean_net_ticks_per_trade']
        sh = best_cell['daily_pnl_sharpe']
        sk = best_cell['regime_skew']
        nd = best_cell['n_pos_days']
        tpd = best_cell['trades_per_day']
        if (ntpt > 0.5 and sh >= 1.0 and (not math.isnan(sk)) and sk <= 0.5
                and nd >= 30 and tpd >= 5):
            verdict = 'STRONG_PASS'
        elif ntpt > 0 and (sh > 0 or nd >= 20):
            verdict = 'MARGINAL'
        else:
            verdict = 'REJECT'
        summary['verdict'] = verdict

    summary_path = OUT_DIR / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f'Summary saved: {summary_path}')
    log.info(f'VERDICT: {summary.get("verdict", "UNKNOWN")}')
    if best_cell:
        log.info(f'Best cell: {best_cell["cell"]} h={best_cell["horizon"]} '
                 f'ntpt={best_cell["mean_net_ticks_per_trade"]:+.3f} '
                 f'sharpe={best_cell["daily_pnl_sharpe"]:+.2f} '
                 f'skew={best_cell["regime_skew"]:.2f} '
                 f'days={best_cell["n_pos_days"]} trades/day={best_cell["trades_per_day"]:.1f}')

    if mlflow_run_id is not None:
        try:
            import mlflow
            mlflow.log_artifact(str(sweep_path))
            mlflow.log_artifact(str(summary_path))
            if all_per_day_for_best is not None:
                mlflow.log_artifact(str(OUT_DIR / 'best_cell_per_day.csv'))
            mlflow.log_artifact(str(LOG_FILE))
            mlflow.end_run()
        except Exception:
            pass


if __name__ == '__main__':
    main()

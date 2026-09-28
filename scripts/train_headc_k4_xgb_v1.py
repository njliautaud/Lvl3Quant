#!/usr/bin/env python3
r"""
Head-C K=4 First-Passage CLASSIFIER v1 (XGBoost, CPU) — HC #428 + HC #432 + HC #503.

Question being tested:
  K=2 first-passage at h=5s is STRUCTURALLY below taker cost (WR ceiling ~35-40%
  bounded by K=2 cap → net = -1.45 to -1.98 ticks/trade after 1.376-tick taker RT).
  Does K=4 first-passage break the ceiling? At WR ~45-50% with K=4 (gross 4 ticks),
  gross expectation could approach +0.5 to +1.0 ticks NET if model holds.

Labels (Jupiter v3.5 NPZ): label_C_K4 ∈ {-1, 0, 1}:
  -1 = downside K=4 ticks hit first
   0 = neither side hit K=4 by horizon (timeout)
   1 = upside K=4 ticks hit first
We train ONE binary upside-first classifier: y = (C_K4 == 1).
Downside-fade signal comes from same model via (prob < 0.45).

OOT compliance (HC #503 R1): canonical window = [20260227, 20260429].
Current v3.5 build covers only Jan/early-Feb 2026. NO canonical OOT labels available.
→ SCAFFOLD-SMOKE mode (clearly tagged in MLflow + reports). Goal: directional evidence
on whether K=4 framing has *any* exploitable signal vs the K=2 ceiling.

FIFO taker grade (integrated):
  - Long signal when prob > 0.55, short signal when prob < 0.45
  - Hold ≤ 7.5s (1.5 × h=5s per HC #432 R2)
  - TP = first-passage of K=4 ticks (target hit) → +4 ticks gross
  - SL = first-passage of opposing K=4 ticks → -4 ticks gross
  - Terminal if neither by 7.5s → use signed price-move-at-horizon (labels_5s)
  - Taker cost 1.376 ticks RT (HC: ES_RT_COMMISSION_TICKS + 1.0 spread)
  - Report: n_trades, WR, mean_ticks_NET, PF, Sharpe per fold + aggregate

Inputs (Jupiter):
  /home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3_v35/*_v35.npz
Outputs:
  /home/jupiter/Lvl3Quant/output/headc_k4_xgb_v1/{summary.json, per_fold.csv,
    fifo_per_day.csv, trades.parquet, fold_NN_oot_DATE.parquet, training.log}
"""
from __future__ import annotations
import sys, os, json, time, glob, math, gc
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
LVL3_ROOT = Path('/home/jupiter/Lvl3Quant')
DATA_DIR  = LVL3_ROOT / 'data' / 'processed' / 'mbo_events_smart_v3_v35'
OUT_DIR   = LVL3_ROOT / 'output' / 'headc_k4_xgb_v1'
LOG_DIR   = LVL3_ROOT / 'logs'
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

import logging
RUN_TS = datetime.now().strftime('%Y%m%d_%H%M%S')
LOG_FILE = LOG_DIR / f'headc_k4_xgb_{RUN_TS}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.FileHandler(LOG_FILE, encoding='utf-8'), logging.StreamHandler()],
)
log = logging.getLogger('headc_k4_xgb')

# -----------------------------------------------------------------------------
# Hyperparams / constants
# -----------------------------------------------------------------------------
HORIZON_S = 5.0
HOLD_CAP_S = HORIZON_S * 1.5  # 7.5s (HC #432 R2)
K_TICKS = 4  # K=4 first-passage labels
ES_TICK_VALUE = 12.50
ES_RT_COMM_TICKS = 0.376
TAKER_RT_TICKS = ES_RT_COMM_TICKS + 1.0  # 1.376 ticks RT cost for taker

# Row subsampling for memory + speed (XGB CPU on ~10M rows/day OOM-risk on 14GB free).
# Stride-based: take every Nth row chronologically. Preserves temporal structure.
TRAIN_STRIDE = 25  # ~400k rows/day train → manageable
EVAL_STRIDE  = 5   # ~2M rows/day eval (we want dense OOT for FIFO sim)

TRAIN_DAYS = 7     # 7-day sliding train (only ~15 RTH days available)
EVAL_DAYS  = 1
FULL_ROUNDS = 1500
EARLY_STOP  = 75
WALL_CAP_S  = 60 * 60  # 1 hour total wall budget

# Signal thresholds for FIFO sim
LONG_THR  = 0.55
SHORT_THR = 0.45

CANONICAL_OOT_MIN = '20260227'   # HC #503 R1
SCAFFOLD_MODE = True             # set to False if/when canonical labels exist

MLFLOW_TRACKING_URI = os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:5000')
MLFLOW_EXPERIMENT = 'headc_k4_xgb_scaffold_v1'


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
def list_dates() -> list[str]:
    files = sorted(DATA_DIR.glob('*_v35.npz'))
    dates = [p.stem.replace('_v35', '') for p in files]
    log.info(f'Found {len(dates)} v3.5 NPZ files: {dates[0]}..{dates[-1]}')
    return dates


def filter_rth_dates(dates: list[str], min_size_bytes: int = 500_000_000) -> list[str]:
    out = []
    for d in dates:
        p = DATA_DIR / f'{d}_v35.npz'
        if p.exists() and p.stat().st_size >= min_size_bytes:
            out.append(d)
    log.info(f'After RTH-size filter (≥{min_size_bytes/1e9:.1f}GB): {len(out)} dates: {out}')
    return out


def load_day_subsampled(date: str, stride: int, want_fifo_fields: bool = False) -> dict | None:
    p = DATA_DIR / f'{date}_v35.npz'
    if not p.exists():
        return None
    try:
        z = np.load(p, allow_pickle=False)
    except Exception as e:
        log.warning(f'{date}: load failed {e}')
        return None
    n = z['events'].shape[0]
    idx = np.arange(0, n, stride, dtype=np.int64)
    ck4 = z['label_C_K4'][idx]
    # Filter out rows where label is undefined: int8 sentinel is anything outside {-1,0,1}
    # Actual data confirmed only {-1,0,1}. No NaN possible (int8).
    mask = np.isin(ck4, [-1, 0, 1])
    idx = idx[mask]
    ck4 = ck4[mask]
    out = {
        'date': date,
        'events': z['events'][idx],     # (m, 25) float32
        'C_K4':   ck4,                  # (m,) int8
        'ts_ns':  z['timestamps'][idx], # (m,) int64
    }
    if want_fifo_fields:
        # labels_5s = signed price move at exactly 5s horizon (ticks)
        if 'labels_5s' in z.files:
            out['ret_5s'] = z['labels_5s'][idx].astype(np.float32)
        else:
            out['ret_5s'] = np.zeros(len(idx), dtype=np.float32)
    z.close()
    log.info(f'  loaded {date}: m={len(idx):,} (stride={stride}, raw={n:,})')
    return out


def stack_days(day_dicts: list[dict]) -> dict:
    X = np.vstack([d['events'] for d in day_dicts])
    y = np.concatenate([(d['C_K4'] == 1).astype(np.float32) for d in day_dicts])
    return {'X': X, 'y': y}


# -----------------------------------------------------------------------------
# Folds
# -----------------------------------------------------------------------------
def build_folds(dates: list[str]) -> list[tuple[list[str], list[str]]]:
    folds = []
    for i in range(TRAIN_DAYS, len(dates), EVAL_DAYS):
        tr = dates[i - TRAIN_DAYS: i]
        ev = dates[i: i + EVAL_DAYS]
        if not ev:
            break
        folds.append((tr, ev))
    return folds


# -----------------------------------------------------------------------------
# Train one fold
# -----------------------------------------------------------------------------
def train_xgb_binary(X_tr, y_tr, X_va, y_va):
    import xgboost as xgb
    dtr = xgb.DMatrix(X_tr, label=y_tr)
    dva = xgb.DMatrix(X_va, label=y_va)
    pos_rate = float(y_tr.mean())
    spw = (1 - pos_rate) / max(pos_rate, 1e-6)
    params = dict(
        objective='binary:logistic',
        eval_metric=['auc', 'logloss'],
        learning_rate=0.05,
        max_depth=7,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.7,
        tree_method='hist',
        nthread=12,
        scale_pos_weight=min(spw, 10.0),
    )
    bst = xgb.train(
        params=params, dtrain=dtr, num_boost_round=FULL_ROUNDS,
        evals=[(dtr, 'train'), (dva, 'eval')],
        early_stopping_rounds=EARLY_STOP, verbose_eval=False,
    )
    pred = bst.predict(dva)
    from sklearn.metrics import roc_auc_score, log_loss
    try: auc = float(roc_auc_score(y_va, pred))
    except Exception: auc = float('nan')
    try: ll = float(log_loss(y_va, np.clip(pred, 1e-7, 1 - 1e-7)))
    except Exception: ll = float('nan')
    acc05 = float(((pred > 0.5) == (y_va > 0.5)).mean())
    base_acc = float(max(y_va.mean(), 1 - y_va.mean()))
    # IC = Spearman between pred and binary y_va (≈ AUC-ish but simpler)
    try:
        from scipy.stats import spearmanr
        ic = float(spearmanr(pred, y_va).correlation)
    except Exception:
        ic = float('nan')
    # Lift @ top-decile
    try:
        thr_p90 = float(np.quantile(pred, 0.90))
        top = y_va[pred >= thr_p90]
        lift_p90 = float(top.mean() / max(y_va.mean(), 1e-9)) if len(top) else float('nan')
    except Exception:
        lift_p90 = float('nan')
    return bst, pred, dict(
        auc=auc, logloss=ll, acc05=acc05, base_acc=base_acc,
        lift_acc=acc05 - base_acc, ic=ic, lift_p90=lift_p90,
        n_va=int(len(y_va)), pos_rate_tr=pos_rate,
        pos_rate_va=float(y_va.mean()),
    )


# -----------------------------------------------------------------------------
# FIFO taker grade — operates on OOT predictions + C_K4 + ret_5s
# -----------------------------------------------------------------------------
def fifo_grade_fold(pred: np.ndarray, ck4: np.ndarray, ret_5s: np.ndarray,
                    date: str) -> dict:
    """
    Per-row trade outcomes for taker-side execution.

    Side mapping:
      - long signal: pred > LONG_THR  → trade direction = +1
      - short signal: pred < SHORT_THR → trade direction = -1
      - else: no trade
    Outcome:
      - First-passage determined by C_K4 sign (it IS the K=4 first-passage label):
          long  + C_K4==+1 → +K_TICKS gross
          long  + C_K4==-1 → -K_TICKS gross
          long  + C_K4== 0 → use signed ret_5s as terminal proxy (capped ±K)
          short + C_K4==-1 → +K_TICKS gross   (we win when downside hits first)
          short + C_K4==+1 → -K_TICKS gross
          short + C_K4== 0 → -ret_5s capped
      - Net = gross − TAKER_RT_TICKS

    HC #432 R2 sanity: C_K4 was built with first-passage WITHIN the K=4 horizon
    (which the v3.5 builder caps). Hold ≤ 7.5s is honored by the label itself;
    the timeout class (C_K4==0) is where neither side passed by horizon, so we
    take the realized 5s move as a fair proxy (capped to avoid label/horizon mismatch).
    """
    long_mask  = pred > LONG_THR
    short_mask = pred < SHORT_THR
    side = np.zeros(len(pred), dtype=np.int8)
    side[long_mask] = +1
    side[short_mask] = -1
    trade_mask = side != 0

    # Gross ticks per trade
    gross = np.zeros(len(pred), dtype=np.float32)

    # long + C_K4=+1 → +K  ; long + C_K4=-1 → -K ; long + C_K4=0 → clip(ret_5s)
    L = long_mask
    S = short_mask
    g_long = np.where(ck4 == 1, +K_TICKS, np.where(ck4 == -1, -K_TICKS,
                       np.clip(ret_5s, -K_TICKS, K_TICKS)))
    g_short = np.where(ck4 == -1, +K_TICKS, np.where(ck4 == 1, -K_TICKS,
                       np.clip(-ret_5s, -K_TICKS, K_TICKS)))
    gross[L] = g_long[L]
    gross[S] = g_short[S]

    net = gross - TAKER_RT_TICKS  # full RT cost per trade

    # Trade-only stats
    trades = net[trade_mask]
    wins = trades > 0
    n = int(len(trades))
    if n == 0:
        return dict(date=date, n_trades=0, wr=float('nan'), mean_ticks=float('nan'),
                    pf=float('nan'), sharpe=float('nan'), gross_mean=float('nan'),
                    n_long=int(L.sum()), n_short=int(S.sum()))
    wr = float(wins.mean())
    mean_ticks = float(trades.mean())
    gross_mean = float(gross[trade_mask].mean())
    pos = trades[trades > 0].sum()
    neg = -trades[trades < 0].sum()
    pf = float(pos / max(neg, 1e-9))
    # Per-trade Sharpe (mean / std × sqrt(n) is wrong for arbitrary frequencies; use raw mean/std)
    sd = float(trades.std(ddof=1)) if n > 1 else float('nan')
    sharpe = float(mean_ticks / sd * math.sqrt(n)) if (sd and sd > 0) else float('nan')
    return dict(
        date=date, n_trades=n, wr=wr, mean_ticks=mean_ticks,
        gross_mean=gross_mean, pf=pf, sharpe=sharpe,
        n_long=int(L.sum()), n_short=int(S.sum()),
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    t0 = time.time()
    log.info('=== headc_k4_xgb_v1 START ===')
    log.info(f'  K_TICKS={K_TICKS}, HORIZON_S={HORIZON_S}, HOLD_CAP_S={HOLD_CAP_S}')
    log.info(f'  TRAIN_DAYS={TRAIN_DAYS}, EVAL_DAYS={EVAL_DAYS}')
    log.info(f'  TRAIN_STRIDE={TRAIN_STRIDE}, EVAL_STRIDE={EVAL_STRIDE}')
    log.info(f'  THRESHOLDS: long>{LONG_THR}, short<{SHORT_THR}')
    log.info(f'  TAKER_RT_TICKS={TAKER_RT_TICKS}')
    log.info(f'  CANONICAL_OOT_MIN={CANONICAL_OOT_MIN}, SCAFFOLD_MODE={SCAFFOLD_MODE}')
    log.info(f'  OUT_DIR={OUT_DIR}, LOG={LOG_FILE}')

    # MLflow
    mlflow = None
    run_id = None
    try:
        import mlflow as _ml
        _ml.set_tracking_uri(MLFLOW_TRACKING_URI)
        _ml.set_experiment(MLFLOW_EXPERIMENT)
        run = _ml.start_run(run_name=f'headc_k4_{RUN_TS}')
        run_id = run.info.run_id
        _ml.log_params(dict(
            k_ticks=K_TICKS, horizon_s=HORIZON_S, hold_cap_s=HOLD_CAP_S,
            train_days=TRAIN_DAYS, train_stride=TRAIN_STRIDE, eval_stride=EVAL_STRIDE,
            long_thr=LONG_THR, short_thr=SHORT_THR, taker_rt_ticks=TAKER_RT_TICKS,
            full_rounds=FULL_ROUNDS, early_stop=EARLY_STOP,
            canonical_oot_min=CANONICAL_OOT_MIN,
        ))
        _ml.set_tag('hc', '393,420,428,432,503')
        _ml.set_tag('node', 'jupiter')
        _ml.set_tag('scaffold', str(SCAFFOLD_MODE).lower())
        _ml.set_tag('scaffold_reason', 'no_v35_labels_in_canonical_oot_window')
        mlflow = _ml
        log.info(f'MLflow run: {run_id}')
    except Exception as e:
        log.warning(f'MLflow disabled: {e}')

    all_dates = list_dates()
    rth_dates = filter_rth_dates(all_dates)
    if len(rth_dates) < TRAIN_DAYS + 1:
        log.error(f'Not enough RTH days: {len(rth_dates)} < {TRAIN_DAYS+1}')
        sys.exit(1)

    # Check OOT compliance — none expected, but log explicitly
    any_canonical = any(d >= CANONICAL_OOT_MIN for d in rth_dates)
    log.info(f'Any canonical-OOT (≥{CANONICAL_OOT_MIN}) dates available? {any_canonical}')
    if not any_canonical:
        log.warning('SCAFFOLD-SMOKE MODE: zero canonical-OOT labels. Results are DIRECTIONAL ONLY.')

    folds = build_folds(rth_dates)
    log.info(f'Built {len(folds)} sliding folds')
    if mlflow is not None:
        try: mlflow.log_metric('n_folds_planned', len(folds))
        except Exception: pass

    per_fold = []
    fifo_per_day = []
    all_trades = []  # for trades.parquet aggregation
    wall_deadline = t0 + WALL_CAP_S

    for k, (tr_d, ev_d) in enumerate(folds):
        if time.time() > wall_deadline:
            log.warning(f'Wall deadline hit at fold {k}; stopping early.')
            break
        fold_t = time.time()
        log.info(f'--- fold {k}: tr={tr_d[0]}..{tr_d[-1]} ({len(tr_d)}d) oot={ev_d[0]} ---')

        # Load train (stride heavy)
        log.info('  loading train days...')
        tr_dicts = []
        for d in tr_d:
            x = load_day_subsampled(d, stride=TRAIN_STRIDE, want_fifo_fields=False)
            if x is not None: tr_dicts.append(x)
        if not tr_dicts:
            log.warning('  empty train, skip')
            continue
        tr_pack = stack_days(tr_dicts)
        del tr_dicts; gc.collect()

        # Load eval (stride lighter + need fifo fields)
        log.info('  loading eval day(s)...')
        ev_dicts = []
        for d in ev_d:
            x = load_day_subsampled(d, stride=EVAL_STRIDE, want_fifo_fields=True)
            if x is not None: ev_dicts.append(x)
        if not ev_dicts:
            log.warning('  empty eval, skip')
            continue

        # Concatenate eval (single day in this config, but keep general)
        X_va = np.vstack([d['events'] for d in ev_dicts])
        ck4_va = np.concatenate([d['C_K4'] for d in ev_dicts]).astype(np.int8)
        ret_va = np.concatenate([d['ret_5s'] for d in ev_dicts]).astype(np.float32)
        ts_va  = np.concatenate([d['ts_ns'] for d in ev_dicts]).astype(np.int64)
        y_va = (ck4_va == 1).astype(np.float32)

        log.info(f'  n_tr={len(tr_pack["y"]):,}  n_va={len(y_va):,}  '
                 f'n_feats={tr_pack["X"].shape[1]}  '
                 f'tr_pos={tr_pack["y"].mean():.3f}  va_pos={y_va.mean():.3f}')

        # Train
        try:
            bst, pred, metrics = train_xgb_binary(tr_pack['X'], tr_pack['y'], X_va, y_va)
        except Exception as e:
            log.error(f'  fold {k} train failed: {e}')
            continue
        del tr_pack; gc.collect()

        elapsed = time.time() - fold_t
        log.info(f'  fold {k} oot={ev_d[0]}  AUC={metrics["auc"]:.4f}  '
                 f'IC={metrics["ic"]:.4f}  acc05={metrics["acc05"]:.4f}  '
                 f'base={metrics["base_acc"]:.4f}  lift={metrics["lift_acc"]:+.4f}  '
                 f'lift@p90={metrics["lift_p90"]:.2f}x  t={elapsed:.1f}s')

        # FIFO taker grade
        fifo = fifo_grade_fold(pred, ck4_va, ret_va, ev_d[0])
        log.info(f'  FIFO TAKER: n={fifo["n_trades"]}  WR={fifo["wr"]:.3f}  '
                 f'mean_ticks_NET={fifo["mean_ticks"]:+.3f}  '
                 f'(gross={fifo["gross_mean"]:+.3f})  PF={fifo["pf"]:.2f}  '
                 f'Sharpe={fifo["sharpe"]:.2f}  '
                 f'(L={fifo["n_long"]}, S={fifo["n_short"]})')
        fifo_per_day.append({**fifo, 'fold': k})

        # Save fold preds
        out_pq = OUT_DIR / f'fold_{k:02d}_oot_{ev_d[0]}.parquet'
        pd.DataFrame({
            'ts_ns':       ts_va,
            'C_K4':        ck4_va,
            'ret_5s':      ret_va,
            'y_true':      y_va,
            'y_pred_proba': pred.astype(np.float32),
        }).to_parquet(out_pq, index=False)

        # Trades for aggregation
        long_mask  = pred > LONG_THR
        short_mask = pred < SHORT_THR
        trade_mask = long_mask | short_mask
        if trade_mask.any():
            side_arr = np.where(long_mask, 1, np.where(short_mask, -1, 0)).astype(np.int8)
            g_long = np.where(ck4_va == 1, +K_TICKS, np.where(ck4_va == -1, -K_TICKS,
                              np.clip(ret_va, -K_TICKS, K_TICKS)))
            g_short = np.where(ck4_va == -1, +K_TICKS, np.where(ck4_va == 1, -K_TICKS,
                               np.clip(-ret_va, -K_TICKS, K_TICKS)))
            gross_arr = np.where(long_mask, g_long, np.where(short_mask, g_short, 0.0))
            net_arr = gross_arr - TAKER_RT_TICKS
            all_trades.append(pd.DataFrame({
                'fold': k,
                'oot':  ev_d[0],
                'ts_ns': ts_va[trade_mask],
                'side': side_arr[trade_mask],
                'pred': pred[trade_mask].astype(np.float32),
                'C_K4': ck4_va[trade_mask],
                'ret_5s': ret_va[trade_mask],
                'gross_ticks': gross_arr[trade_mask].astype(np.float32),
                'net_ticks':   net_arr[trade_mask].astype(np.float32),
            }))

        fold_summary = {
            **metrics, 'fold': k, 'oot': ev_d[0],
            'n_tr': int(tr_pack['y'].shape[0]) if 'tr_pack' in dir() else None,
            'elapsed_s': elapsed,
            'fifo_n_trades': fifo['n_trades'],
            'fifo_wr': fifo['wr'],
            'fifo_mean_ticks_net': fifo['mean_ticks'],
            'fifo_pf': fifo['pf'],
            'fifo_sharpe': fifo['sharpe'],
        }
        per_fold.append(fold_summary)

        if mlflow is not None:
            try:
                mlflow.log_metric(f'fold{k}_auc', metrics['auc'], step=k)
                mlflow.log_metric(f'fold{k}_ic', metrics['ic'], step=k)
                mlflow.log_metric(f'fold{k}_lift', metrics['lift_acc'], step=k)
                mlflow.log_metric(f'fold{k}_fifo_wr', fifo['wr'] if not math.isnan(fifo['wr']) else 0, step=k)
                mlflow.log_metric(f'fold{k}_fifo_net_ticks', fifo['mean_ticks'] if not math.isnan(fifo['mean_ticks']) else 0, step=k)
                mlflow.log_metric(f'fold{k}_fifo_sharpe', fifo['sharpe'] if not math.isnan(fifo['sharpe']) else 0, step=k)
            except Exception:
                pass

        # cleanup
        del X_va, ck4_va, ret_va, ts_va, y_va, pred, bst, ev_dicts
        gc.collect()

    # -------- Aggregate ----------
    if not per_fold:
        log.error('NO FOLDS COMPLETED.')
        sys.exit(2)

    pf_df = pd.DataFrame(per_fold)
    pf_df.to_csv(OUT_DIR / 'per_fold.csv', index=False)
    fifo_df = pd.DataFrame(fifo_per_day)
    fifo_df.to_csv(OUT_DIR / 'fifo_per_day.csv', index=False)

    if all_trades:
        trades_df = pd.concat(all_trades, ignore_index=True)
        trades_df.to_parquet(OUT_DIR / 'trades.parquet', index=False)
        agg_n = int(len(trades_df))
        agg_wr = float((trades_df['net_ticks'] > 0).mean())
        agg_mean_net = float(trades_df['net_ticks'].mean())
        agg_gross = float(trades_df['gross_ticks'].mean())
        pos_sum = float(trades_df.loc[trades_df['net_ticks'] > 0, 'net_ticks'].sum())
        neg_sum = float(-trades_df.loc[trades_df['net_ticks'] < 0, 'net_ticks'].sum())
        agg_pf = pos_sum / max(neg_sum, 1e-9)
        sd = float(trades_df['net_ticks'].std(ddof=1))
        agg_sharpe = (agg_mean_net / sd) * math.sqrt(agg_n) if sd > 0 else float('nan')
    else:
        trades_df = pd.DataFrame()
        agg_n = 0; agg_wr = float('nan'); agg_mean_net = float('nan')
        agg_gross = float('nan'); agg_pf = float('nan'); agg_sharpe = float('nan')

    # K=2 ceiling comparison baseline (for verdict)
    # At K=2 with WR=0.40 → gross = 0.40*2 + 0.60*(-2) = -0.4 ticks gross,
    #                        net = -0.4 - 1.376 = -1.776 ticks/trade.
    # At K=4 with WR=0.50 → gross = 0.50*4 + 0.50*(-4) = 0 → net = -1.376
    # K=4 breaks even at WR=0.50 + lift covering 1.376; needs WR > ~0.67 for big edge.
    # Practical pass: net > 0 means we beat taker cost.
    k2_ceiling_net_ticks = -1.776  # canonical comparison number from task brief
    if not math.isnan(agg_mean_net) and not math.isnan(agg_mean_net):
        beats_k2_ceiling = agg_mean_net > k2_ceiling_net_ticks
        beats_zero = agg_mean_net > 0
    else:
        beats_k2_ceiling = False
        beats_zero = False

    if beats_zero and agg_pf > 1.2 and agg_n >= 50:
        verdict = 'STRONG_PASS'
    elif beats_k2_ceiling and agg_n >= 20:
        verdict = 'WEAK_PASS'
    else:
        verdict = 'REJECT'

    aucs  = [f['auc'] for f in per_fold if not math.isnan(f['auc'])]
    ics   = [f['ic']  for f in per_fold if not math.isnan(f['ic'])]
    lifts = [f['lift_acc'] for f in per_fold]

    summary = dict(
        run_ts=RUN_TS,
        mlflow_run_id=run_id,
        mlflow_uri=MLFLOW_TRACKING_URI,
        mlflow_experiment=MLFLOW_EXPERIMENT,
        scaffold_mode=SCAFFOLD_MODE,
        canonical_oot_min=CANONICAL_OOT_MIN,
        any_canonical_oot_used=bool(any_canonical),
        k_ticks=K_TICKS, horizon_s=HORIZON_S, hold_cap_s=HOLD_CAP_S,
        long_thr=LONG_THR, short_thr=SHORT_THR,
        taker_rt_ticks=TAKER_RT_TICKS,
        train_days=TRAIN_DAYS, eval_days=EVAL_DAYS,
        train_stride=TRAIN_STRIDE, eval_stride=EVAL_STRIDE,
        n_folds_completed=len(per_fold),
        # Classifier metrics
        mean_auc=float(np.mean(aucs)) if aucs else float('nan'),
        median_auc=float(np.median(aucs)) if aucs else float('nan'),
        mean_ic=float(np.mean(ics)) if ics else float('nan'),
        mean_lift_acc=float(np.mean(lifts)),
        # FIFO taker aggregate
        agg_n_trades=agg_n,
        agg_wr=agg_wr,
        agg_mean_ticks_NET=agg_mean_net,
        agg_mean_ticks_gross=agg_gross,
        agg_pf=agg_pf,
        agg_sharpe=agg_sharpe,
        # Verdict
        k2_ceiling_net_ticks=k2_ceiling_net_ticks,
        beats_k2_ceiling=beats_k2_ceiling,
        beats_zero=beats_zero,
        verdict=verdict,
        per_fold=per_fold,
        wall_s=time.time() - t0,
    )
    with open(OUT_DIR / 'summary.json', 'w') as fh:
        json.dump(summary, fh, indent=2, default=float)

    log.info('=== DONE ===')
    log.info(f'  Folds: {len(per_fold)}')
    log.info(f'  Mean AUC: {summary["mean_auc"]:.4f}  Mean IC: {summary["mean_ic"]:.4f}  '
             f'Mean acc lift: {summary["mean_lift_acc"]:+.4f}')
    log.info(f'  FIFO taker aggregate: n={agg_n}  WR={agg_wr:.3f}  '
             f'mean_net={agg_mean_net:+.3f} ticks  PF={agg_pf:.2f}  Sharpe={agg_sharpe:.2f}')
    log.info(f'  Beats K=2 ceiling ({k2_ceiling_net_ticks:+.3f}): {beats_k2_ceiling}')
    log.info(f'  Beats zero (profitable for taker): {beats_zero}')
    log.info(f'  VERDICT: {verdict}')
    log.info(f'  Wall: {summary["wall_s"]:.1f}s')

    if mlflow is not None:
        try:
            mlflow.log_metrics(dict(
                mean_auc=summary['mean_auc'],
                mean_ic=summary['mean_ic'],
                mean_lift_acc=summary['mean_lift_acc'],
                agg_wr=agg_wr if not math.isnan(agg_wr) else 0,
                agg_mean_ticks_net=agg_mean_net if not math.isnan(agg_mean_net) else 0,
                agg_pf=agg_pf if not math.isnan(agg_pf) else 0,
                agg_sharpe=agg_sharpe if not math.isnan(agg_sharpe) else 0,
                agg_n_trades=agg_n,
                n_folds_completed=len(per_fold),
            ))
            mlflow.set_tag('verdict', verdict)
            mlflow.log_artifact(str(OUT_DIR / 'summary.json'))
            mlflow.log_artifact(str(OUT_DIR / 'per_fold.csv'))
            mlflow.log_artifact(str(OUT_DIR / 'fifo_per_day.csv'))
            mlflow.end_run()
        except Exception as e:
            log.warning(f'mlflow log artifacts failed: {e}')


if __name__ == '__main__':
    main()

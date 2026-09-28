#!/usr/bin/env python3
"""
Parametric Exit Strategy Optimizer — Sampled Paths
====================================================

Pre-extracts signal events with SAMPLED forward price paths (15 time points
instead of 500K steps). Memory efficient, fast evaluation.

Strategy params (~10): trailing stop, signal decay, profit target, time stop,
confidence threshold, prediction horizon.

HC refs: #244, #239, #226
"""

from __future__ import annotations
import json, time, logging, argparse, sys
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    print("pip install optuna"); sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("parametric_exit")

TICK_VALUE = 12.50
COMMISSION_TICKS = 0.376
TICK_SIZE = 0.25

# Meta columns
M_TS=0; M_PRICE=1; M_ET=2; M_SPREAD=3; M_P1S=4; M_P5S=5; M_P10S=6
M_PIDX=7; M_BIDD=8; M_ASKD=9; M_IMB=10; M_BIDQ=11; M_ASKQ=12

# Sampled time horizons (seconds)
SAMPLE_TIMES = np.array([0, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30, 45, 60, 90, 120], dtype=np.float64)
N_SAMPLES = len(SAMPLE_TIMES)

# Tier thresholds
TIER_THRESHOLDS = [(1.50, 3), (1.00, 2), (0.50, 1), (0.30, 0)]

def get_tier(conf: float) -> int:
    for thresh, t in TIER_THRESHOLDS:
        if conf >= thresh:
            return t
    return -1


def extract_signals_sampled(precomputed_dir: Path, date_str: str, min_conf: float = 0.20):
    """
    Extract signals with sampled forward price paths.
    Returns structured array: one row per signal with compact features.
    """
    meta = np.load(str(precomputed_dir / f"{date_str}_meta.npy"), mmap_mode='r')
    n = len(meta)
    if n < 10:
        return []

    # Find prediction changes
    pidx = meta[:, M_PIDX].astype(np.int64)
    changes = np.where(np.diff(pidx) != 0)[0] + 1
    if n > 0 and pidx[0] >= 0:
        changes = np.concatenate([[0], changes])

    signals = []
    for ci in changes:
        if ci >= n:
            continue
        p10 = meta[ci, M_P10S]
        p5 = meta[ci, M_P5S]
        p1 = meta[ci, M_P1S]
        conf = abs(p10)
        if conf < min_conf:
            continue

        direction = 1 if p10 > 0 else -1
        entry_ts = meta[ci, M_TS]
        entry_price = meta[ci, M_PRICE]

        # Sample forward prices at SAMPLE_TIMES
        sampled_moves = np.full(N_SAMPLES, np.nan, dtype=np.float64)
        sampled_moves[0] = 0.0  # t=0

        for si in range(1, N_SAMPLES):
            target_ts = entry_ts + SAMPLE_TIMES[si]
            # Binary search for closest step
            search_end = min(ci + 2_000_000, n)
            ts_slice = meta[ci:search_end, M_TS]
            idx = np.searchsorted(ts_slice, target_ts)
            if idx >= len(ts_slice):
                break  # past end of data
            abs_idx = ci + idx
            price = meta[abs_idx, M_PRICE]
            move = (price - entry_price) * direction / TICK_SIZE
            sampled_moves[si] = move

        # Compute running MFE at each sample point
        valid = ~np.isnan(sampled_moves)
        if valid.sum() < 3:
            continue

        # Forward predictions (when signal changes)
        fwd_preds = []
        search_end = min(ci + 2_000_000, n)
        fwd_pidx = meta[ci:search_end, M_PIDX].astype(np.int64)
        fwd_changes = np.where(np.diff(fwd_pidx) != 0)[0] + 1
        for fc in fwd_changes[:20]:  # max 20 future predictions
            abs_fc = ci + fc
            if abs_fc >= n:
                break
            t_off = meta[abs_fc, M_TS] - entry_ts
            if t_off > 120:
                break
            fwd_preds.append((t_off, float(meta[abs_fc, M_P10S])))

        signals.append({
            'date': date_str,
            'ts': entry_ts,
            'conf': conf,
            'p1s': float(p1),
            'p5s': float(p5),
            'p10s': float(p10),
            'dir': direction,
            'moves': sampled_moves.copy(),  # (N_SAMPLES,) direction-adjusted ticks
            'fwd_preds': fwd_preds,  # list of (time_offset, pred_10s)
        })

    return signals


def evaluate_strategy(
    signals: list,
    min_confidence: float,
    min_tier: int,
    cooldown_secs: float,
    trail_act: float,
    trail_dist: float,
    signal_decay: bool,
    decay_thresh: float,
    profit_target: float,
    max_hold: float,
    pred_horizon: str,
) -> Dict:
    """Evaluate exit strategy on pre-extracted sampled signals. FAST."""
    pnls = []
    holds = []
    exit_reasons = {}
    mfes = []
    maes = []
    dirs = []
    last_exit_ts = -999.0

    for sig in signals:
        # Entry filter
        if pred_horizon == "1s":
            conf = abs(sig['p1s'])
        elif pred_horizon == "5s":
            conf = abs(sig['p5s'])
        else:
            conf = sig['conf']

        if conf < min_confidence:
            continue
        if get_tier(conf) < min_tier:
            continue
        if sig['ts'] - last_exit_ts < cooldown_secs:
            continue

        moves = sig['moves']
        valid = ~np.isnan(moves)

        if valid.sum() < 3:
            continue

        # Find exit point by scanning sampled times
        exit_idx = -1
        exit_reason = "eod"
        running_mfe = 0.0

        for si in range(1, N_SAMPLES):
            if np.isnan(moves[si]):
                exit_idx = si - 1
                exit_reason = "data_end"
                break

            t = SAMPLE_TIMES[si]
            mv = moves[si]
            running_mfe = max(running_mfe, mv)

            # 1. Profit target
            if mv >= profit_target:
                exit_idx = si
                exit_reason = "profit_target"
                break

            # 2. Time stop
            if t >= max_hold:
                exit_idx = si
                exit_reason = "time_stop"
                break

            # 3. Trailing stop
            if running_mfe >= trail_act and (running_mfe - mv) >= trail_dist:
                exit_idx = si
                exit_reason = "trailing_stop"
                break

            # 4. Signal decay
            if signal_decay:
                for fp_t, fp_pred in sig['fwd_preds']:
                    if fp_t <= SAMPLE_TIMES[si - 1] or fp_t > t:
                        continue
                    flipped = (sig['dir'] == 1 and fp_pred < 0) or \
                              (sig['dir'] == -1 and fp_pred > 0)
                    weakened = abs(fp_pred) < decay_thresh
                    if flipped or weakened:
                        exit_idx = si
                        exit_reason = "signal_flip" if flipped else "signal_decay"
                        break
                if exit_idx >= 0:
                    break

        if exit_idx < 0:
            # Hit end of samples without exit
            last_valid = np.where(valid)[0][-1]
            exit_idx = last_valid
            exit_reason = "eod"

        pnl = moves[exit_idx] - COMMISSION_TICKS
        hold = SAMPLE_TIMES[exit_idx]
        mfe = max(0, np.nanmax(moves[:exit_idx + 1]))
        mae = abs(min(0, np.nanmin(moves[:exit_idx + 1])))

        pnls.append(pnl)
        holds.append(hold)
        mfes.append(mfe)
        maes.append(mae)
        dirs.append(sig['dir'])
        exit_reasons[exit_reason] = exit_reasons.get(exit_reason, 0) + 1

        last_exit_ts = sig['ts'] + hold

    # Metrics
    n = len(pnls)
    if n == 0:
        return {"n_trades": 0, "sortino": -10.0, "win_rate": 0.0, "profit_factor": 0.0,
                "avg_pnl": 0.0, "total_pnl": 0.0, "total_usd": 0.0, "avg_hold": 0.0,
                "exits": {}, "avg_win": 0.0, "avg_loss": 0.0,
                "n_long": 0, "n_short": 0, "long_wr": 0.0, "short_wr": 0.0,
                "avg_mfe": 0.0, "avg_mae": 0.0, "mfe_cap": 0.0}

    p = np.array(pnls)
    w = p[p > 0]; l = p[p <= 0]
    wr = len(w) / n
    avg = float(p.mean())
    ds = float(np.std(np.minimum(p, 0)))
    sortino = avg / max(ds, 1e-6)
    gp = float(w.sum()) if len(w) > 0 else 0.0
    gl = float(abs(l.sum())) if len(l) > 0 else 1e-6
    d = np.array(dirs)
    lm = d == 1; sm = d == -1

    mfe_caps = []
    for i in range(n):
        if mfes[i] > 0:
            mfe_caps.append(pnls[i] / mfes[i])

    return {
        "n_trades": n,
        "sortino": round(sortino, 4),
        "win_rate": round(wr, 4),
        "profit_factor": round(gp / max(gl, 1e-6), 4),
        "avg_pnl": round(avg, 4),
        "total_pnl": round(float(p.sum()), 2),
        "total_usd": round(float(p.sum()) * TICK_VALUE, 2),
        "avg_hold": round(float(np.mean(holds)), 2),
        "exits": exit_reasons,
        "avg_win": round(float(w.mean()) if len(w) > 0 else 0.0, 4),
        "avg_loss": round(float(l.mean()) if len(l) > 0 else 0.0, 4),
        "n_long": int(lm.sum()), "n_short": int(sm.sum()),
        "long_wr": round(float((p[lm] > 0).mean()) if lm.sum() > 0 else 0.0, 4),
        "short_wr": round(float((p[sm] > 0).mean()) if sm.sum() > 0 else 0.0, 4),
        "avg_mfe": round(float(np.mean(mfes)), 4),
        "avg_mae": round(float(np.mean(maes)), 4),
        "mfe_cap": round(float(np.mean(mfe_caps)) if mfe_caps else 0.0, 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--precomputed-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--n-trials", type=int, default=300)
    parser.add_argument("--train-dates", type=int, default=40)
    parser.add_argument("--eval-dates", type=int, default=5)
    parser.add_argument("--objective", type=str, default="sortino")
    args = parser.parse_args()

    pdir = Path(args.precomputed_dir)
    odir = Path(args.output_dir); odir.mkdir(parents=True, exist_ok=True)

    all_dates = sorted([f.stem.replace("_base_obs", "") for f in pdir.glob("*_base_obs.npy")])
    log.info(f"Found {len(all_dates)} dates")

    nt = args.train_dates; ne = args.eval_dates
    nf = min(3, max(1, (len(all_dates) - nt) // ne))

    cfg = {"n_trials": args.n_trials, "train_dates": nt, "eval_dates": ne,
           "n_folds": nf, "objective": args.objective, "total_dates": len(all_dates)}
    with open(odir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    log.info(f"Walk-forward: {nf} folds, {nt} train + {ne} eval")
    all_results = []

    for fi in range(nf):
        s = fi * ne
        train_d = all_dates[s:s+nt]; eval_d = all_dates[s+nt:s+nt+ne]
        if len(eval_d) < ne: break

        log.info(f"\n{'='*60}")
        log.info(f"FOLD {fi}: Train [{train_d[0]}..{train_d[-1]}] Eval [{eval_d[0]}..{eval_d[-1]}]")

        # Pre-extract signals
        t0 = time.time()
        train_sigs = []; eval_sigs = []
        for i, d in enumerate(train_d):
            try:
                sigs = extract_signals_sampled(pdir, d)
                train_sigs.extend(sigs)
                if (i+1) % 10 == 0:
                    log.info(f"  Extracted {i+1}/{len(train_d)} train dates, {len(train_sigs)} signals so far")
            except Exception as e:
                log.warning(f"  {d}: {e}")

        for d in eval_d:
            try:
                eval_sigs.extend(extract_signals_sampled(pdir, d))
            except Exception as e:
                log.warning(f"  {d}: {e}")

        ext_t = time.time() - t0
        log.info(f"  Extracted {len(train_sigs)} train + {len(eval_sigs)} eval signals in {ext_t:.1f}s")

        # Optuna
        def objective(trial):
            p = {
                "min_confidence": trial.suggest_float("min_confidence", 0.30, 1.50),
                "min_tier": trial.suggest_int("min_tier", 0, 3),
                "cooldown_secs": trial.suggest_float("cooldown_secs", 0.5, 10.0),
                "trail_act": trial.suggest_float("trail_act", 0.5, 8.0),
                "trail_dist": trial.suggest_float("trail_dist", 0.25, 5.0),
                "signal_decay": trial.suggest_categorical("signal_decay", [True, False]),
                "decay_thresh": trial.suggest_float("decay_thresh", 0.0, 0.5),
                "profit_target": trial.suggest_float("profit_target", 2.0, 20.0),
                "max_hold": trial.suggest_float("max_hold", 5.0, 120.0),
                "pred_horizon": trial.suggest_categorical("pred_horizon", ["1s", "5s", "10s"]),
            }
            m = evaluate_strategy(train_sigs, **p)
            if m["n_trades"] < 20: return -10.0
            if args.objective == "sortino": return m["sortino"]
            elif args.objective == "profit_factor": return m["profit_factor"] - 1.0
            else: return m["avg_pnl"]

        study = optuna.create_study(direction="maximize",
                                     sampler=optuna.samplers.TPESampler(seed=42+fi))
        t0 = time.time()
        for batch in range(0, args.n_trials, 50):
            study.optimize(objective, n_trials=min(50, args.n_trials - len(study.trials)))
            b = study.best_trial
            log.info(f"  Trial {len(study.trials)}/{args.n_trials} | Best {args.objective}={b.value:.4f}")

        opt_t = time.time() - t0
        bp = study.best_trial.params
        log.info(f"  Optimization: {opt_t:.0f}s | Best: {json.dumps(bp)}")

        # Train perf
        tm = evaluate_strategy(train_sigs, **bp)
        log.info(f"\n--- TRAIN ---")
        log.info(f"  {tm['n_trades']}tr WR={tm['win_rate']:.1%} Sortino={tm['sortino']:.3f} "
                 f"PF={tm['profit_factor']:.2f} P&L=${tm['total_usd']:.0f}")
        log.info(f"  Hold={tm['avg_hold']:.1f}s MFE_cap={tm['mfe_cap']:.3f} "
                 f"Win={tm['avg_win']:.2f}tk Loss={tm['avg_loss']:.2f}tk")
        log.info(f"  MFE={tm['avg_mfe']:.2f}tk MAE={tm['avg_mae']:.2f}tk Exits={tm['exits']}")
        log.info(f"  L={tm['n_long']}(WR={tm['long_wr']:.0%}) S={tm['n_short']}(WR={tm['short_wr']:.0%})")

        # OOT
        em = evaluate_strategy(eval_sigs, **bp)
        log.info(f"\n--- OOT EVAL [{eval_d[0]}..{eval_d[-1]}] ---")
        log.info(f"  {em['n_trades']}tr WR={em['win_rate']:.1%} Sortino={em['sortino']:.3f} "
                 f"PF={em['profit_factor']:.2f} P&L=${em['total_usd']:.0f}")
        log.info(f"  Hold={em['avg_hold']:.1f}s MFE_cap={em['mfe_cap']:.3f} "
                 f"Win={em['avg_win']:.2f}tk Loss={em['avg_loss']:.2f}tk")
        log.info(f"  MFE={em['avg_mfe']:.2f}tk MAE={em['avg_mae']:.2f}tk Exits={em['exits']}")
        log.info(f"  L={em['n_long']}(WR={em['long_wr']:.0%}) S={em['n_short']}(WR={em['short_wr']:.0%})")

        # Per-date
        for d in eval_d:
            dsigs = [s for s in eval_sigs if s['date'] == d]
            dm = evaluate_strategy(dsigs, **bp)
            log.info(f"    {d}: {dm['n_trades']}tr WR={dm['win_rate']:.0%} PF={dm['profit_factor']:.2f} "
                     f"P&L=${dm['total_usd']:.0f} exits={dm['exits']}")

        fr = {"fold": fi, "best_params": bp, "train": tm, "eval": em,
              "opt_time": round(opt_t, 1), "n_train_sigs": len(train_sigs), "n_eval_sigs": len(eval_sigs)}
        all_results.append(fr)
        with open(odir / f"fold{fi}.json", "w") as f:
            json.dump(fr, f, indent=2, default=str)

    # Concat
    log.info(f"\n{'='*60}")
    log.info(f"CONCAT OOT")
    tot_tr = sum(r["eval"]["n_trades"] for r in all_results)
    tot_pnl = sum(r["eval"]["total_usd"] for r in all_results)
    log.info(f"  Total: {tot_tr} trades, P&L=${tot_pnl:.0f}")
    for r in all_results:
        e = r["eval"]
        log.info(f"  Fold {r['fold']}: {e['n_trades']}tr WR={e['win_rate']:.1%} "
                 f"Sortino={e['sortino']:.3f} PF={e['profit_factor']:.2f} P&L=${e['total_usd']:.0f}")

    log.info(f"\n--- PARAM STABILITY ---")
    for r in all_results:
        p = r["best_params"]
        log.info(f"  Fold {r['fold']}: conf>={p['min_confidence']:.2f} tier>={p['min_tier']} "
                 f"trail={p['trail_act']:.1f}/{p['trail_dist']:.1f} hold<={p['max_hold']:.0f}s "
                 f"target={p['profit_target']:.1f}tk horizon={p['pred_horizon']} "
                 f"decay={p['signal_decay']}")

    with open(odir / "results.json", "w") as f:
        json.dump({"folds": all_results, "config": cfg, "total_oot_trades": tot_tr,
                    "total_oot_pnl": tot_pnl}, f, indent=2, default=str)

    log.info(f"\nSaved to {odir}/results.json")
    log.info("Done.")


if __name__ == "__main__":
    main()

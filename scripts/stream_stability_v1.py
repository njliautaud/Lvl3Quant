#!/usr/bin/env python3
"""
stream_stability_v1.py — Stream-coherence (pressure-thesis) gate on baseline alpha.

Tests whether existing CNN-Mamba v3.4.2 baseline predictions show deployable edge
when filtered by STREAM-COHERENCE (sign-consistency / drift / flip-rate / variance
of the prediction stream over the next K events) rather than by single-snapshot
top-X%-confidence.

User thesis (verbatim ~12:30 ET 2026-05-22): predictions are a CONTINUOUS STREAM,
not snapshots. The right gate is "are the NEXT K events of predictions unanimously
pointing one direction with low variance and rare flips?" If yes — pressure — and
edge persists through the next K events of the trade.

Inputs
------
- OOT predictions:  output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate/oot_<YYYYMMDD>.npz
                    Fields used: pred_log_ret_{1s,5s,10s,30s}, target_log_ret_{1s,5s,10s,30s,60s},
                    mask_log_ret_{...} (1=valid), sample_dates
- v4 alpha labels:  data/processed/mbo_events_smart_v3_alpha_labels_v4/<YYYYMMDD>_alpha_labels.npz
                    Used only for the alignment-key sanity check (corr=1.0 already proven).

Per-stride is the 250-event stride (1499 + k*250). Stride duration ~0.47s in RTH.
The per-stride prediction used to define the stream is pred_log_ret_1s (shortest avail).

Policies evaluated
------------------
1. HOLD-TO-h: open at stride k, hold for h seconds, realized = target_log_ret_h[k] (in ticks).
2. STREAM-FLIP-EXIT: open at stride k, exit at first stride j in [k+1, k+K] whose
   pred_log_ret_1s sign disagrees with pred[k] (against position), OR at stride k+K
   if no flip. Exit time = j_offset * stride_seconds, realized = target_log_ret_{h_eff}
   where h_eff is the closest available horizon in {1s,5s,10s,30s,60s}.

Stream-coherence flags (over next K strides)
---------------------------------------------
- sign_consistency_K: fraction of next K preds with same sign as pred[k]
- cumulative_drift_K: sum of next K preds (signed)
- flip_rate_K: count of sign changes in next K preds / K
- prediction_variance_K: variance of next K preds
- mean_abs_K: mean |pred| over next K  (proxy for stream STRENGTH)
- stream_unanimous_K := sign_consistency_K >= 0.85
- stream_strong_K    := stream_unanimous_K AND mean_abs_K >= p75(mean_abs_K within day)

For each (K, h, side, policy), apply HC #428 R1 deploy gates:
  - net_ticks_per_event  > +0.10
  - win_rate            >= 0.52
  - profitable_days     >= 30  (or proportional >= ceil(0.80*N_days_evaluated))
  - regime_imbalance    |Sharpe_green - Sharpe_red| / max(...) <= 0.50
  - day_concentration   max |day_total_pnl| / sum |day_total_pnl| <= 0.70

Costs: passive-fill RT commission = 0.376 ticks (ES_RT_COMMISSION=$4.70 / TICK=$12.50).

Outputs (output/stream_stability_v1/)
-------------------------------------
- summary.csv          — one row per (K x horizon x side x policy x stream_filter_variant)
- winning_cells.txt    — cells passing all gates (or closest misses, with gap diagnosis)
- baseline_anchors.csv — single-snapshot top-X% baseline anchors for comparison
- .regen_complete.json — sentinel with timestamps + summary stats

Constraints
-----------
- 60 min walltime cap.
- /usr/bin/python3, numpy/pandas only (no torch needed).
- Stay local. Skip dates where pred NPZ is missing.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ----- Constants --------------------------------------------------------------
COMMISSION_TICKS = 0.376
JOIN_OFFSET = 1499
JOIN_STRIDE = 250
RTH_SECONDS = 23400.0  # 6.5 h
HORIZONS = ["1s", "5s", "10s", "30s", "60s"]      # used for h_eff lookup
HORIZON_SECS = {"1s": 1.0, "5s": 5.0, "10s": 10.0, "30s": 30.0, "60s": 60.0}
TRADE_HORIZONS = ["1s", "5s", "10s", "30s"]       # h for hold_to_h policy
K_GRID = [4, 20, 40, 80]
SIDES = ["long", "short"]
POLICIES = ["hold_to_h", "stream_flip_exit"]
STREAM_FILTERS = ["unanimous", "strong"]          # strong = unanimous AND mean_abs >= day-p75

# Deploy gate thresholds
GATE_NET = 0.10
GATE_WR  = 0.52
GATE_PDAYS_FRAC = 30.0 / 47.0       # fraction; applied proportionally to available days
GATE_REGIME_IMB = 0.50
GATE_DAYCONC = 0.70

OOT_DIR = Path("/home/jupiter/Lvl3Quant/output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate")
V4_DIR  = Path("/home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3_alpha_labels_v4")
OUT_DIR = Path("/home/jupiter/Lvl3Quant/output/stream_stability_v1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

WALLTIME_CAP_SEC = 55 * 60

# ----- Helpers ----------------------------------------------------------------
def closest_horizon_seconds(elapsed_s: float) -> str:
    """Return the horizon string in HORIZONS closest to elapsed_s."""
    best = HORIZONS[0]
    best_d = abs(HORIZON_SECS[best] - elapsed_s)
    for h in HORIZONS[1:]:
        d = abs(HORIZON_SECS[h] - elapsed_s)
        if d < best_d:
            best = h; best_d = d
    return best

def sharpe_per_day(day_vals):
    a = np.asarray(day_vals, dtype=np.float64)
    if a.size < 2: return 0.0
    s = np.std(a, ddof=1)
    if s == 0: return 0.0
    # per-day Sharpe (consistent with prior runs); annualization could be added but kept comparable
    return float(np.mean(a) / s * np.sqrt(252))

def compute_stream_features(pred_1s: np.ndarray, K: int):
    """Return dict of arrays of length N for the K-stride stream features."""
    N = pred_1s.shape[0]
    # We need next K predictions for each k: pred[k+1..k+K]. Tail K rows have insufficient data.
    valid_tail = np.zeros(N, dtype=bool)
    valid_tail[:N - K] = True

    sign_p = np.sign(pred_1s)  # 0 if pred is exactly 0
    # Build matrix of next-K via shifts is O(N*K) memory if K is large; with K up to 80 and N~50k -> 4M, fine
    nxt = np.full((N, K), np.nan, dtype=np.float32)
    for j in range(1, K + 1):
        nxt[:N - j, j - 1] = pred_1s[j:N + 1 if (N - j) < N else N]
        # Simpler/safer:
    # rebuild cleanly:
    nxt = np.full((N, K), np.nan, dtype=np.float32)
    for j in range(1, K + 1):
        if N - j > 0:
            nxt[:N - j, j - 1] = pred_1s[j:N]

    nxt_sign = np.sign(nxt)
    # sign_consistency_K: fraction same sign as pred[k] among next K (counting zeros as non-match)
    same_sign = (nxt_sign == sign_p[:, None]) & (sign_p[:, None] != 0)
    sign_consistency_K = same_sign.sum(axis=1) / float(K)

    # flip_rate_K: count of sign changes among next K (transitions across pred[k], pred[k+1], ..., pred[k+K])
    # Sequence of K+1 sign values starting with sign_p[k]
    full_signs = np.concatenate([sign_p[:, None], nxt_sign], axis=1)  # shape (N, K+1)
    transitions = (np.diff(full_signs, axis=1) != 0).sum(axis=1)
    flip_rate_K = transitions / float(K)

    cumulative_drift_K = np.nansum(nxt, axis=1)
    # nanvar with all-nan rows -> 0; only edge rows
    with np.errstate(invalid="ignore"):
        prediction_variance_K = np.nanvar(nxt, axis=1, ddof=0)
        mean_abs_K = np.nanmean(np.abs(nxt), axis=1)

    return dict(
        sign_consistency_K=sign_consistency_K.astype(np.float32),
        cumulative_drift_K=cumulative_drift_K.astype(np.float32),
        flip_rate_K=flip_rate_K.astype(np.float32),
        prediction_variance_K=prediction_variance_K.astype(np.float32),
        mean_abs_K=mean_abs_K.astype(np.float32),
        valid_tail=valid_tail,
        nxt_sign=nxt_sign,   # for flip-exit lookup
    )

def first_flip_offset(pred_k_sign: np.ndarray, nxt_sign: np.ndarray, K: int) -> np.ndarray:
    """For each row k, return the offset (1..K) of the first sign that disagrees with pred_k_sign.
       If no flip in next K, return K. If pred_k_sign is 0, return K (no signal)."""
    N = pred_k_sign.shape[0]
    out = np.full(N, K, dtype=np.int32)
    disagrees = (nxt_sign != pred_k_sign[:, None]) & (pred_k_sign[:, None] != 0)
    # first True per row
    any_flip = disagrees.any(axis=1)
    # argmax returns first True (since bool argmax stops at first True)
    first = disagrees.argmax(axis=1) + 1
    out[any_flip] = first[any_flip]
    return out

# ----- Load -------------------------------------------------------------------
def load_day(date: str, stride_sec_global: float):
    oot_path = OOT_DIR / f"oot_{date}.npz"
    if not oot_path.exists():
        return None
    oot = np.load(oot_path, allow_pickle=True)
    if "pred_log_ret_1s" not in oot.files:
        return None
    N = int(oot["pred_log_ret_1s"].shape[0])
    if N < 1000:    # skip degenerate short days (e.g., holiday half-sessions with stride seconds blowing up)
        return None
    out = {"date": date, "N": N}
    out["stride_sec"] = RTH_SECONDS / N  # per-day estimate
    out["pred_1s"] = oot["pred_log_ret_1s"].astype(np.float32)
    out["pred_5s"] = oot["pred_log_ret_5s"].astype(np.float32)
    out["pred_10s"] = oot["pred_log_ret_10s"].astype(np.float32)
    out["pred_30s"] = oot["pred_log_ret_30s"].astype(np.float32)
    for h in HORIZONS:
        # target_log_ret_h is signed ticks per closest_to_profit_v4 convention
        if f"target_log_ret_{h}" in oot.files:
            out[f"signed_{h}"] = oot[f"target_log_ret_{h}"].astype(np.float32)
            out[f"mask_{h}"] = oot[f"mask_log_ret_{h}"].astype(np.float32) > 0.5
        else:
            out[f"signed_{h}"] = np.full(N, np.nan, dtype=np.float32)
            out[f"mask_{h}"] = np.zeros(N, dtype=bool)
    return out

def regime_for_day(day):
    """green/red/flat based on mean of signed_30s realized move (already in ticks)."""
    m = day["mask_30s"]
    if m.sum() == 0:
        return "flat"
    daily = float(np.nanmean(day["signed_30s"][m]))
    if daily > 0.10: return "green"
    if daily < -0.10: return "red"
    return "flat"

# ----- Baseline anchors (single-snapshot top-X% short / long, hold-to-h) ------
def compute_baseline_anchors(days):
    """Reproduce the single-snapshot baseline anchors from this morning's run."""
    rows = []
    for h in TRADE_HORIZONS:
        for side in SIDES:
            # Pool across days
            preds, signed, masks, dates, regimes = [], [], [], [], []
            for d in days:
                p = d[f"pred_{h}"] if f"pred_{h}" in d else d["pred_1s"]  # use horizon-matched
                preds.append(p)
                signed.append(d[f"signed_{h}"])
                masks.append(d[f"mask_{h}"])
                dates.append(np.full(d["N"], d["date"], dtype=object))
                regimes.append(np.full(d["N"], d["regime"], dtype=object))
            p = np.concatenate(preds); s = np.concatenate(signed)
            m = np.concatenate(masks); da = np.concatenate(dates); re = np.concatenate(regimes)
            side_dir = p < 0 if side == "short" else p > 0
            eff = m & side_dir & ~np.isnan(p) & ~np.isnan(s)
            if eff.sum() < 200:
                continue
            apr = np.abs(p)
            for pct_lo, name in [(0.99, "top_1pct"), (0.95, "top_5pct"), (0.90, "top_10pct"), (0.80, "top_20pct")]:
                # rank within eff
                idx = np.where(eff)[0]
                if len(idx) < 200: continue
                thresh = np.quantile(apr[idx], pct_lo)
                bmask = eff & (apr >= thresh)
                if bmask.sum() < 50:
                    continue
                sgn = s[bmask] if side == "long" else -s[bmask]
                net = sgn - COMMISSION_TICKS
                # day stats
                day_b = da[bmask]; reg_b = re[bmask]
                uniq = np.unique(day_b)
                day_mean, day_sum, day_reg = [], [], []
                for ud in uniq:
                    mm = day_b == ud
                    day_mean.append(float(np.mean(net[mm])))
                    day_sum.append(float(np.sum(net[mm])))
                    day_reg.append(reg_b[mm][0])
                day_mean = np.array(day_mean); day_sum = np.array(day_sum); day_reg = np.array(day_reg)
                prof_days = int((day_mean > 0).sum())
                s_g = sharpe_per_day(day_mean[day_reg == "green"])
                s_r = sharpe_per_day(day_mean[day_reg == "red"])
                denom = max(abs(s_g), abs(s_r), 1e-9)
                rim = abs(s_g - s_r) / denom
                abs_tot = np.abs(day_sum)
                conc = float(abs_tot.max() / abs_tot.sum()) if abs_tot.sum() > 0 else 1.0
                rows.append(dict(
                    horizon=h, side=side, bucket=name, policy="hold_to_h",
                    n=int(bmask.sum()),
                    net_ticks_per_event=float(np.mean(net)),
                    win_rate=float(np.mean(net > 0)),
                    profitable_days=prof_days,
                    total_days=int(len(uniq)),
                    sharpe_per_day=sharpe_per_day(day_mean),
                    sharpe_green=s_g, sharpe_red=s_r,
                    regime_imbalance=rim, day_concentration=conc,
                ))
    return pd.DataFrame(rows)

# ----- Main -------------------------------------------------------------------
def main():
    t0 = time.time()
    print(f"[start] {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    oot_dates = sorted({p.stem.replace("oot_", "") for p in OOT_DIR.glob("oot_*.npz")})
    print(f"[dates] {len(oot_dates)} OOT pred files found", flush=True)

    # Load all days
    days = []
    for d in oot_dates:
        if time.time() - t0 > WALLTIME_CAP_SEC:
            print(f"[walltime] cap hit during load at {d}", flush=True); break
        day = load_day(d, RTH_SECONDS / 50000.0)
        if day is None:
            print(f"  skip {d} (missing/short)", flush=True); continue
        day["regime"] = regime_for_day(day)
        days.append(day)
        print(f"  loaded {d}: N={day['N']:,} regime={day['regime']} stride_sec={day['stride_sec']:.3f}", flush=True)
    if not days:
        print("[fatal] no usable days", file=sys.stderr); sys.exit(2)

    n_days = len(days)
    pdays_threshold = int(np.ceil(GATE_PDAYS_FRAC * n_days))
    print(f"[gate] profitable_days threshold = {pdays_threshold}/{n_days} (proportional to 30/47)", flush=True)

    # ----- Baseline anchors ----
    print(f"[baseline] computing single-snapshot anchors ...", flush=True)
    anchors = compute_baseline_anchors(days)
    anchors_path = OUT_DIR / "baseline_anchors.csv"
    anchors.to_csv(anchors_path, index=False)
    print(f"[write] {anchors_path} ({len(anchors)} rows)", flush=True)

    # ----- Pre-compute stream features per day per K --------------------------
    print(f"[stream] computing stream features for K ∈ {K_GRID}", flush=True)
    for d in days:
        d["stream"] = {}
        for K in K_GRID:
            d["stream"][K] = compute_stream_features(d["pred_1s"], K)

    # ----- Main grid --------------------------------------------------------------
    summary_rows = []
    for K in K_GRID:
        print(f"[grid] K={K} ...", flush=True)
        for h in TRADE_HORIZONS:
            for side in SIDES:
                for policy in POLICIES:
                    for stream_filter in STREAM_FILTERS:
                        if time.time() - t0 > WALLTIME_CAP_SEC:
                            print(f"[walltime] cap hit in grid", flush=True)
                            break

                        # Pool across days
                        all_net = []
                        all_day = []
                        all_reg = []
                        all_total_eff = 0   # for coverage denom (after horizon/tail validity)
                        all_passed = 0      # passed stream filter
                        for d in days:
                            sf = d["stream"][K]
                            valid_tail = sf["valid_tail"]
                            sign_c = sf["sign_consistency_K"]
                            mean_abs = sf["mean_abs_K"]
                            pred_1s = d["pred_1s"]
                            sign_p = np.sign(pred_1s)
                            mask_h = d[f"mask_{h}"]
                            signed_h = d[f"signed_{h}"]

                            # side direction (based on the stride's own 1s pred)
                            if side == "short":
                                side_dir = sign_p < 0
                            else:
                                side_dir = sign_p > 0

                            base_valid = valid_tail & mask_h & side_dir & ~np.isnan(signed_h)
                            all_total_eff += int(base_valid.sum())

                            # Stream filter
                            unanimous = sign_c >= 0.85
                            if stream_filter == "unanimous":
                                stream_ok = unanimous
                            else:  # strong
                                if base_valid.sum() == 0:
                                    stream_ok = np.zeros_like(unanimous, dtype=bool)
                                else:
                                    # day p75 of mean_abs within base_valid
                                    p75 = float(np.nanpercentile(mean_abs[base_valid], 75)) if base_valid.sum() else 0.0
                                    stream_ok = unanimous & (mean_abs >= p75)
                            sel = base_valid & stream_ok
                            n_sel = int(sel.sum())
                            all_passed += n_sel
                            if n_sel == 0:
                                continue

                            # Realized PnL per policy
                            if policy == "hold_to_h":
                                realized = signed_h.copy()  # signed in ticks (long convention)
                                if side == "short":
                                    realized = -realized
                                net_evt = realized[sel] - COMMISSION_TICKS
                            else:  # stream_flip_exit
                                # exit offset (1..K) of first flip
                                nxt_sign = sf["nxt_sign"]
                                flip_off = first_flip_offset(sign_p, nxt_sign, K)
                                # exit time seconds per row
                                stride_sec = d["stride_sec"]
                                exit_sec = flip_off.astype(np.float32) * stride_sec
                                # Cap at horizon h seconds
                                h_sec = HORIZON_SECS[h]
                                exit_sec_capped = np.minimum(exit_sec, h_sec)
                                # For each row, pick the closest available horizon h_eff and use its target_log_ret
                                # Vectorize: compute distances to each available horizon, pick argmin
                                avail_hs = HORIZONS  # all
                                h_secs_arr = np.array([HORIZON_SECS[hh] for hh in avail_hs], dtype=np.float32)
                                # distances per row per horizon
                                # only need for selected rows
                                sel_idx = np.where(sel)[0]
                                if sel_idx.size == 0:
                                    continue
                                dists = np.abs(exit_sec_capped[sel_idx, None] - h_secs_arr[None, :])
                                best_h_i = dists.argmin(axis=1)
                                # Build realized for selected rows
                                realized_sel = np.zeros(sel_idx.size, dtype=np.float32)
                                valid_sel = np.ones(sel_idx.size, dtype=bool)
                                for hi, hh in enumerate(avail_hs):
                                    use = (best_h_i == hi)
                                    if not use.any():
                                        continue
                                    rows = sel_idx[use]
                                    sgn = d[f"signed_{hh}"][rows]
                                    mh = d[f"mask_{hh}"][rows]
                                    if side == "short":
                                        sgn = -sgn
                                    realized_sel[use] = sgn
                                    valid_sel[use] = mh & ~np.isnan(sgn)
                                net_evt_all = realized_sel - COMMISSION_TICKS
                                net_evt = net_evt_all[valid_sel]
                                if net_evt.size == 0:
                                    continue
                                # rebuild day labels for the kept subset
                                keep_mask = valid_sel

                            # Day/regime tagging
                            if policy == "hold_to_h":
                                dates_arr = np.full(n_sel, d["date"], dtype=object)
                                regs_arr  = np.full(n_sel, d["regime"], dtype=object)
                            else:
                                dates_arr = np.full(net_evt.size, d["date"], dtype=object)
                                regs_arr  = np.full(net_evt.size, d["regime"], dtype=object)

                            all_net.append(net_evt)
                            all_day.append(dates_arr)
                            all_reg.append(regs_arr)

                        if not all_net:
                            continue
                        net = np.concatenate(all_net)
                        day_arr = np.concatenate(all_day)
                        reg_arr = np.concatenate(all_reg)
                        n_total = int(net.size)
                        if n_total < 100:
                            continue

                        # Per-day means/sums
                        uniq_days = np.unique(day_arr)
                        day_mean = []; day_sum = []; day_reg = []
                        for ud in uniq_days:
                            mm = day_arr == ud
                            day_mean.append(float(np.mean(net[mm])))
                            day_sum.append(float(np.sum(net[mm])))
                            day_reg.append(reg_arr[mm][0])
                        day_mean = np.array(day_mean); day_sum = np.array(day_sum)
                        day_reg = np.array(day_reg)
                        prof_days = int((day_mean > 0).sum())
                        s_all = sharpe_per_day(day_mean)
                        s_g = sharpe_per_day(day_mean[day_reg == "green"])
                        s_r = sharpe_per_day(day_mean[day_reg == "red"])
                        s_f = sharpe_per_day(day_mean[day_reg == "flat"])
                        denom = max(abs(s_g), abs(s_r), 1e-9)
                        rim = abs(s_g - s_r) / denom
                        abs_tot = np.abs(day_sum)
                        conc = float(abs_tot.max() / abs_tot.sum()) if abs_tot.sum() > 0 else 1.0

                        net_mean = float(np.mean(net))
                        wr = float(np.mean(net > 0))
                        coverage = float(all_passed / max(1, all_total_eff))

                        gate_net = net_mean > GATE_NET
                        gate_wr = wr >= GATE_WR
                        gate_pdays = prof_days >= pdays_threshold
                        gate_regime = rim <= GATE_REGIME_IMB
                        gate_dayconc = conc <= GATE_DAYCONC
                        pass_all = gate_net and gate_wr and gate_pdays and gate_regime and gate_dayconc

                        summary_rows.append(dict(
                            K=K, horizon=h, side=side, policy=policy,
                            stream_filter=stream_filter,
                            n_events=n_total,
                            coverage_pct=coverage,
                            net_ticks_per_event=net_mean,
                            win_rate=wr,
                            profitable_days=prof_days,
                            total_days=int(len(uniq_days)),
                            sharpe_per_day=s_all,
                            sharpe_green=s_g, sharpe_red=s_r, sharpe_flat=s_f,
                            regime_imbalance=rim,
                            day_concentration=conc,
                            gate_net=gate_net, gate_wr=gate_wr, gate_pdays=gate_pdays,
                            gate_regime=gate_regime, gate_dayconc=gate_dayconc,
                            pass_gates=pass_all,
                        ))
                    if time.time() - t0 > WALLTIME_CAP_SEC: break
                if time.time() - t0 > WALLTIME_CAP_SEC: break
            if time.time() - t0 > WALLTIME_CAP_SEC: break
        if time.time() - t0 > WALLTIME_CAP_SEC: break

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = OUT_DIR / "summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"[write] {summary_csv} ({len(summary_df)} rows)", flush=True)

    # Winners + closest miss
    winners = summary_df[summary_df["pass_gates"]].copy() if len(summary_df) else pd.DataFrame()
    win_path = OUT_DIR / "winning_cells.txt"
    with open(win_path, "w") as f:
        f.write("STREAM-STABILITY v1 — pressure-thesis gate on baseline alpha\n")
        f.write("=" * 80 + "\n")
        f.write(f"OOT days evaluated: {n_days}\n")
        f.write(f"Stride avg seconds: {np.mean([d['stride_sec'] for d in days]):.3f}\n")
        f.write(f"Commission: {COMMISSION_TICKS} ticks (passive RT)\n")
        f.write(f"K grid: {K_GRID}  Horizons: {TRADE_HORIZONS}\n")
        f.write(f"Stream filters: unanimous (sign_consistency>=0.85), strong (unanim AND mean_abs>=day-p75)\n")
        f.write(f"Gates: net>+0.10, WR>=0.52, prof_days>={pdays_threshold}/{n_days}, "
                f"regime_imb<=0.50, day_conc<=0.70\n\n")
        if len(winners) == 0:
            f.write("ZERO cells pass all gates.\n\n")
            if len(summary_df):
                cols = ["K","horizon","side","policy","stream_filter","n_events","coverage_pct",
                        "net_ticks_per_event","win_rate","profitable_days","total_days",
                        "sharpe_per_day","regime_imbalance","day_concentration"]
                rank = summary_df.sort_values("net_ticks_per_event", ascending=False).head(20)
                f.write("TOP 20 BY NET TICKS:\n")
                f.write(rank[cols].to_string(index=False))
                f.write("\n\nCLOSEST-MISS DIAGNOSIS (top 5 by gates_passed then net):\n")
                gates = ["gate_net","gate_wr","gate_pdays","gate_regime","gate_dayconc"]
                summary_df["n_gates_passed"] = summary_df[gates].sum(axis=1)
                rank2 = summary_df.sort_values(["n_gates_passed","net_ticks_per_event"], ascending=False).head(5)
                for _, r in rank2.iterrows():
                    missed = [g for g in gates if not r[g]]
                    f.write(f"\n  K={r.K} {r.horizon} {r.side} {r.policy} filter={r.stream_filter}: "
                            f"net={r.net_ticks_per_event:+.4f} WR={r.win_rate:.3f} "
                            f"pdays={int(r.profitable_days)}/{int(r.total_days)} "
                            f"regimb={r.regime_imbalance:.3f} dayconc={r.day_concentration:.3f} "
                            f"n={int(r.n_events)} coverage={r.coverage_pct*100:.2f}%\n")
                    f.write(f"    gates_passed: {int(r.n_gates_passed)}/5  missed: {missed}\n")
        else:
            f.write(f"WINNING CELLS ({len(winners)}):\n\n")
            cols = ["K","horizon","side","policy","stream_filter","n_events","coverage_pct",
                    "net_ticks_per_event","win_rate","profitable_days","total_days",
                    "sharpe_per_day","sharpe_green","sharpe_red","regime_imbalance","day_concentration"]
            f.write(winners.sort_values("net_ticks_per_event", ascending=False)[cols].to_string(index=False))
            f.write("\n")
        # Anchors
        f.write("\n\nBASELINE ANCHORS (single-snapshot top-X% hold-to-h, no stream filter):\n")
        if len(anchors):
            f.write(anchors.sort_values("net_ticks_per_event", ascending=False).head(10).to_string(index=False))
            f.write("\n")
    print(f"[write] {win_path}", flush=True)

    # Regen sentinel
    regen = {
        "task": "stream_stability_v1",
        "hc_refs": ["HC#486R6", "HC#428R1", "HC#485R5", "HC#420"],
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_days_evaluated": n_days,
        "K_grid": K_GRID,
        "trade_horizons": TRADE_HORIZONS,
        "n_summary_rows": int(len(summary_df)),
        "n_winning_cells": int(len(winners)),
        "pdays_threshold": pdays_threshold,
        "elapsed_seconds": round(time.time() - t0, 1),
        "outputs": {
            "summary_csv": str(summary_csv),
            "winning_cells_txt": str(win_path),
            "baseline_anchors_csv": str(anchors_path),
        },
    }
    with open(OUT_DIR / ".regen_complete.json", "w") as f:
        json.dump(regen, f, indent=2)
    print(f"[done] elapsed {time.time()-t0:.1f}s — winners={len(winners)} / {len(summary_df)}", flush=True)

if __name__ == "__main__":
    main()

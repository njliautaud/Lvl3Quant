#!/usr/bin/env python3
"""
multik_asym_taker_walk_v1.py
============================
Ordered first-passage MBO walk for the multi-K asymmetric TAKER framework.

User directive (2026-05-31): "continue with our taker agenda. can even be
multiple ticks but the idea is we are trading the models predictions of
continuation or reversal."

What's new vs prior walks (fillsim_v1 / h1s_v1):
  - TAKER entry: long fills instantly at entry_ask, short at entry_bid. No
    fill model; we cross the spread immediately at the touch.
  - ORDERED first-passage. We record the FIRST time the mid-price reaches
    +K ticks from entry (TP) and the FIRST time it reaches -S ticks (SL).
    Outcome is whichever comes first (vs prior walks which used MAX MFE
    and MAX MAE over the hold window — magnitudes, not ordered events).
  - Multi-K grid with TP > SL only (asymmetric in the correct direction).

Horizon: h=10s primary. p90 realized MFE = 5.0t at h=10s -> K up to 5 is
HC #428 R2 compliant. Hold cap = 1.5 * h = 15s. Cancel window N/A (taker).

Grid:
  - TP K in {2, 3, 4, 5} ticks
  - SL S in {1, 2} ticks (asymmetric: TP > SL required)
  - Conviction Q in {0.5, 1, 2.5, 5, 10, 20} %
  - Gates: composite (toxicity_rank - adverse_rank), toxicity_only, mfe_only
  - Cost: TAKER 1.376t (commission 0.376 + 1.0 spread)

Inputs:
  /home/jupiter/teleclaude-main/tmp_tox_data/{mfe,toxicity}/fold_NN_oot_YYYYMMDD.parquet
  /home/jupiter/teleclaude-main/tmp_tox_data/mfe/fold_NN_oot_YYYYMMDD_adverse.parquet
  /home/jupiter/Lvl3Quant/data/processed/mbo_book_features/YYYYMMDD_book_features.npz

Outputs (under /home/jupiter/Lvl3Quant/output/multik_asym_taker_v1/):
  - per_trade_walks.parquet  (per fold, full walk records)
  - cells_summary.parquet    (Q x K x S x gate stats)
  - regime_gate.json         (per-day pnl, green/red, HC #428 R1 check)
  - verdict.md               (plain-English verdict)
"""
from __future__ import annotations
import argparse, json, math, os, re, sys, time
from glob import glob
from pathlib import Path
import numpy as np
import pandas as pd

LVL3 = Path("/home/jupiter/Lvl3Quant")
DATA = Path("/home/jupiter/teleclaude-main/tmp_tox_data")
BOOK_DIR = LVL3 / "data" / "processed" / "mbo_book_features"
OUT_DIR = LVL3 / "output" / "multik_asym_taker_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HOLD_NS = int(15.0e9)  # 1.5 * h(=10s)
TICK = 1.0  # bid/ask in book_features.npz are stored as int tick units (price/0.25)

# HC #428 R2: TP <= p90(MFE @ h=10s) = 5.0t  -> K <= 5 OK
P90_MFE_H10 = 5.0

COST_TAKER = 1.376  # commission 0.376 + 1 spread

# Grid (asymmetric TP > SL only).
# K_LIST = barriers measured in PREDICTED direction (also used as flipped-SL).
# S_LIST = barriers measured AGAINST predicted direction (also used as flipped-TP).
# We extend both so we can evaluate (a) original continuation cells (TP=K, SL=S, asymmetric)
# AND (b) side-flipped reversal cells (TP=S, SL=K, asymmetric).
K_LIST = [1, 2, 3, 4, 5]  # added 1 to support flipped SL=1
S_LIST = [1, 2, 3, 4, 5]  # extended to support flipped TP up to 5
Q_LIST = [0.5, 1.0, 2.5, 5.0, 10.0, 20.0]
GATES = ["composite", "toxicity_only", "mfe_only"]
# Original-direction (continuation) cells: TP=K, SL=S, K > S
# Flipped-direction (reversal) cells: TP=S, SL=K, S > K — evaluated separately

OOT_MIN = "20260227"
OOT_MAX = "20260429"


def discover_folds():
    mfiles = sorted(glob(str(DATA / "mfe" / "fold_*_oot_*.parquet")))
    out = []
    for mp in mfiles:
        name = os.path.basename(mp)
        m = re.match(r"fold_(\d+)_oot_(\d{8})\.parquet$", name)
        if not m:
            continue
        fid, date = int(m.group(1)), m.group(2)
        if date < OOT_MIN or date > OOT_MAX:
            continue
        ap = DATA / "mfe" / f"fold_{fid:02d}_oot_{date}_adverse.parquet"
        tp = DATA / "toxicity" / f"fold_{fid:02d}_oot_{date}.parquet"
        bk = BOOK_DIR / f"{date}_book_features.npz"
        if not (ap.exists() and tp.exists() and bk.exists()):
            print(f"  skip fold {fid} {date}: missing inputs "
                  f"adv={ap.exists()} tox={tp.exists()} book={bk.exists()}")
            continue
        out.append((fid, date, Path(mp), ap, tp, bk))
    return out


def join_fold(mp, ap, tp):
    m = pd.read_parquet(mp)
    a = pd.read_parquet(ap)
    t = pd.read_parquet(tp)
    # mfe + adverse merged on (event_id, side); tox is per-event but we keep only matching
    df = m.merge(
        a[["event_id", "side", "y_true_adverse", "y_pred_adverse"]],
        on=["event_id", "side"], how="inner"
    )
    df = df.merge(
        t[["event_id", "side", "y_true_toxicity", "y_pred_toxicity"]],
        on=["event_id", "side"], how="inner"
    )
    return df.sort_values("ts_ns").reset_index(drop=True)


def walk_first_passage(df, book_p, k_max, s_max):
    """For each event row:
       - LONG (side=+1): enter at entry_ask (taker pays the ask). Mark TP_time as
         the first ts where mid >= entry_mid + K*1tick for K in K_LIST.
         Mark SL_time as first ts where mid <= entry_mid - S*1tick for S in S_LIST.
       - SHORT (side=-1): enter at entry_bid. TP_time = first mid <= entry_mid - K.
         SL_time = first mid >= entry_mid + S.

       For each (K,S) cell record:
         tp_time, sl_time, exit_ticks (raw, not net of cost). For ANY cell:
           - if tp before sl -> +K
           - if sl before tp -> -S
           - if neither -> signed mid_change at hold cap (mid_end - entry_mid)*side
       Plus per-row diagnostics: entry_mid_tick, hold_end_mid_change, fill_priced_in (taker cost handled later).

       Returns dict of np arrays:
         keys: 'entry_mid_tick', 'mid_change_hold_ticks',
               for each K in K_LIST: 'tp{K}_time_ns'
               for each S in S_LIST: 'sl{S}_time_ns'
    """
    book = np.load(book_p, allow_pickle=True)
    ts = book["timestamps"]
    bid1 = book["features"][:, 0].astype(np.int32)  # in tick units (price/0.25)
    ask1 = book["features"][:, 5].astype(np.int32)

    n = len(df)
    entry_ts = df["ts_ns"].values
    sides = df["side"].values.astype(np.int8)

    idx0 = np.clip(np.searchsorted(ts, entry_ts, side="left"), 0, len(ts) - 1)
    end_ts = entry_ts + HOLD_NS
    idx_end = np.clip(np.searchsorted(ts, end_ts, side="right"), 0, len(ts))

    eb_arr = bid1[idx0]
    ea_arr = ask1[idx0]
    # Use mid in *tick units* (each book row stores integer tick prices = price/0.25).
    entry_mid_tick = (eb_arr.astype(np.float64) + ea_arr.astype(np.float64)) / 2.0

    out = {
        "entry_mid_tick": entry_mid_tick.astype(np.float64),
        "mid_change_hold_ticks": np.full(n, np.nan, dtype=np.float64),
        "entry_bid_tick": eb_arr.astype(np.int32),
        "entry_ask_tick": ea_arr.astype(np.int32),
        "spread_ticks": (ea_arr.astype(np.int32) - eb_arr.astype(np.int32)).astype(np.int32),
        "hold_npts": (idx_end - idx0).astype(np.int32),
    }
    for K in K_LIST:
        out[f"tp{K}_dt_ns"] = np.full(n, -1, dtype=np.int64)
    for S in S_LIST:
        out[f"sl{S}_dt_ns"] = np.full(n, -1, dtype=np.int64)

    # Pre-extract slices loop. For n events (~20-50k per fold) and hold of 15s
    # (could be 10k-100k book rows), pure-numpy slice-and-argmax should be fine.
    for i in range(n):
        i0 = idx0[i]
        i1 = idx_end[i]
        eb = eb_arr[i]
        ea = ea_arr[i]
        # Validity: ask >= bid, and not both zero (uninit). Negative tick
        # prices ARE valid (per-day offset normalization in book_features).
        if ea < eb or (eb == 0 and ea == 0):
            continue
        if i1 <= i0 + 1:
            out["mid_change_hold_ticks"][i] = 0.0
            continue
        # Slice forward (exclusive of entry point itself)
        fb = bid1[i0 + 1:i1]
        fa = ask1[i0 + 1:i1]
        ft = ts[i0 + 1:i1]
        # Forward validity: ask >= bid, and not both-zero (uninit rows).
        valid = (fa >= fb) & ~((fb == 0) & (fa == 0))
        if not valid.any():
            out["mid_change_hold_ticks"][i] = 0.0
            continue
        mid_f = (fb.astype(np.float64) + fa.astype(np.float64)) / 2.0
        # Set invalid mids to entry_mid so they don't trigger barriers
        em = entry_mid_tick[i]
        mid_f = np.where(valid, mid_f, em)
        side = sides[i]
        # mid_change in ticks (already in tick units)
        delta_ticks = (mid_f - em) * side  # positive if moves in trade direction
        # exit mid change at last valid sample
        last_valid_idx = np.where(valid)[0]
        if last_valid_idx.size > 0:
            j = last_valid_idx[-1]
            out["mid_change_hold_ticks"][i] = float((mid_f[j] - em) * side)
        else:
            out["mid_change_hold_ticks"][i] = 0.0

        # First-passage for each TP K (delta_ticks >= K)
        for K in K_LIST:
            hits = np.where(delta_ticks >= K)[0]
            if hits.size:
                out[f"tp{K}_dt_ns"][i] = int(ft[hits[0]] - entry_ts[i])
        # First-passage for each SL S (delta_ticks <= -S)
        for S in S_LIST:
            hits = np.where(delta_ticks <= -S)[0]
            if hits.size:
                out[f"sl{S}_dt_ns"][i] = int(ft[hits[0]] - entry_ts[i])

    return out


def gate_mask_per_fold(df_all, gate, q_pct):
    """Top Q% per-fold within side-correct universe.
       NOTE: 'side' in parquets is already the prediction direction (from CNN-Mamba).
       So we don't filter by predicted-side here — we just gate by conviction.
    """
    n = len(df_all)
    mask = np.zeros(n, dtype=bool)
    for fid, g in df_all.groupby("fold_id", sort=False):
        idx = g.index.values
        k = max(1, int(round(len(idx) * q_pct / 100.0)))
        if gate == "composite":
            # higher pred_toxicity + lower pred_adverse = better
            tr = pd.Series(g["y_pred_toxicity"].values).rank(method="average").values
            ar = pd.Series(g["y_pred_adverse"].values).rank(method="average").values
            score = tr - ar
            top = idx[np.argsort(-score)[:k]]
        elif gate == "toxicity_only":
            top = idx[np.argsort(-g["y_pred_toxicity"].values)[:k]]
        elif gate == "mfe_only":
            top = idx[np.argsort(-g["y_pred_mfe"].values)[:k]]
        else:
            raise ValueError(gate)
        mask[top] = True
    return mask


def compute_pnl_per_trade(tp_dt, sl_dt, mid_change_hold, K, S):
    """Per-row exit ticks (raw, NOT net of cost).
       tp_dt[i] = -1 if never reached, else dt in ns.
       sl_dt[i] = -1 if never reached, else dt in ns.
    """
    n = len(tp_dt)
    raw = np.empty(n, dtype=np.float64)
    outcome = np.empty(n, dtype="<U6")  # 'tp','sl','hold'
    for i in range(n):
        t_tp = tp_dt[i]
        t_sl = sl_dt[i]
        if t_tp >= 0 and (t_sl < 0 or t_tp <= t_sl):
            raw[i] = +K
            outcome[i] = "tp"
        elif t_sl >= 0 and (t_tp < 0 or t_sl < t_tp):
            raw[i] = -S
            outcome[i] = "sl"
        else:
            raw[i] = float(mid_change_hold[i]) if not np.isnan(mid_change_hold[i]) else 0.0
            outcome[i] = "hold"
    return raw, outcome


def metrics(net, n_total):
    if len(net) == 0:
        return {
            "n_trades": 0, "n_total": int(n_total),
            "WR": None, "WR_conditional": None,
            "mean_ticks": None, "median_ticks": None, "PF": None, "Sharpe": None,
            "sum_ticks": 0.0,
        }
    wins = net[net > 0]
    losses = net[net < 0]
    sw = float(wins.sum())
    sl = float(-losses.sum())
    pf = (sw / sl) if sl > 0 else (float("inf") if sw > 0 else None)
    mean = float(net.mean())
    median = float(np.median(net))
    std = float(net.std(ddof=1)) if len(net) > 1 else 0.0
    sh = (mean / std * math.sqrt(len(net))) if std > 0 else None
    return {
        "n_trades": int(len(net)), "n_total": int(n_total),
        "WR": float((net > 0).mean()),
        "mean_ticks": mean, "median_ticks": median, "PF": pf, "Sharpe": sh,
        "sum_ticks": float(net.sum()),
    }


def regime_check(per_day_pnl):
    """HC #428 R1: classify each OOT day green/red/flat by mean_ticks sign.
       Compute Sharpe per regime; reject if |Sg - Sr| / max(|Sg|,|Sr|) > 0.50.
       Day-concentration: |max_day_pnl| / |sum_pnl| <= 0.70."""
    days = list(per_day_pnl.items())  # [(date, list_of_net)]
    rows = []
    for d, x in days:
        x = np.asarray(x, dtype=np.float64)
        if len(x) == 0:
            continue
        mean = float(x.mean())
        std = float(x.std(ddof=1)) if len(x) > 1 else 0.0
        sh = (mean / std * math.sqrt(len(x))) if std > 0 else None
        rows.append({
            "oot_date": d, "n": int(len(x)),
            "mean_ticks": mean, "WR": float((x > 0).mean()),
            "sum_ticks": float(x.sum()), "Sharpe": sh,
        })
    # classify
    for r in rows:
        if r["mean_ticks"] > 0.01:
            r["regime"] = "green"
        elif r["mean_ticks"] < -0.01:
            r["regime"] = "red"
        else:
            r["regime"] = "flat"
    greens = [r for r in rows if r["regime"] == "green"]
    reds = [r for r in rows if r["regime"] == "red"]

    def avg_sh(arr):
        vs = [r["Sharpe"] for r in arr if r["Sharpe"] is not None]
        return float(np.mean(vs)) if vs else None

    sh_g = avg_sh(greens)
    sh_r = avg_sh(reds)
    if sh_g is not None and sh_r is not None and max(abs(sh_g), abs(sh_r)) > 0:
        asym = abs(sh_g - sh_r) / max(abs(sh_g), abs(sh_r))
    else:
        asym = None
    total = sum(r["sum_ticks"] for r in rows)
    if total != 0:
        day_conc = max(abs(r["sum_ticks"]) for r in rows) / abs(total)
    else:
        day_conc = None

    pass_asym = (asym is not None and asym <= 0.50)
    pass_conc = (day_conc is not None and day_conc <= 0.70)
    return {
        "per_day": rows,
        "n_green": len(greens), "n_red": len(reds),
        "n_flat": sum(1 for r in rows if r["regime"] == "flat"),
        "sharpe_green": sh_g, "sharpe_red": sh_r,
        "asymmetry": asym, "day_concentration": day_conc,
        "pass_asymmetry_05": bool(pass_asym),
        "pass_dayconc_07": bool(pass_conc),
        "pass_R1_regime_gate": bool(pass_asym and pass_conc),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="Run only fold 03 (20260415) to validate walker.")
    ap.add_argument("--workers", type=int, default=2,
                    help="MP pool size for fold walks (ignored if --smoke).")
    args = ap.parse_args()

    t0 = time.time()
    print(f"== multik_asym_taker_walk_v1 — {time.strftime('%Y-%m-%d %H:%M:%S')} ==")
    folds = discover_folds()
    print(f"Discovered {len(folds)} folds in [{OOT_MIN}, {OOT_MAX}].")
    if args.smoke:
        folds = [f for f in folds if f[1] in ("20260415", "20260319")]
        print(f"SMOKE mode: running folds {[(f[0], f[1]) for f in folds]}")
    if not folds:
        print("No folds found. Aborting.")
        return

    per_trade_path = OUT_DIR / "per_trade_walks.parquet"
    all_dfs = []
    fold_diag = []
    for (fid, date, mp, apath, tpath, bk) in folds:
        t_fold = time.time()
        df = join_fold(mp, apath, tpath)
        if df.empty:
            print(f"  fold {fid} {date}: empty join.")
            continue
        walk = walk_first_passage(df, bk, max(K_LIST), max(S_LIST))
        df["fold_id"] = fid
        df["oot_date"] = date
        for k, v in walk.items():
            df[k] = v
        all_dfs.append(df)
        # Diagnostics
        n = len(df)
        tp_rates = {f"tp{K}_rate": float((df[f"tp{K}_dt_ns"].values >= 0).mean()) for K in K_LIST}
        sl_rates = {f"sl{S}_rate": float((df[f"sl{S}_dt_ns"].values >= 0).mean()) for S in S_LIST}
        d = {
            "fold_id": fid, "oot_date": date, "n": n,
            "elapsed_s": round(time.time() - t_fold, 2),
            "mean_spread_ticks": float(df["spread_ticks"].mean()),
            **tp_rates, **sl_rates,
        }
        fold_diag.append(d)
        print(f"  fold {fid:>2} {date}: n={n:>6}  elapsed={d['elapsed_s']:>5.1f}s  "
              f"tp2={tp_rates['tp2_rate']:.3f} tp3={tp_rates['tp3_rate']:.3f} "
              f"tp4={tp_rates['tp4_rate']:.3f} tp5={tp_rates['tp5_rate']:.3f} | "
              f"sl1={sl_rates['sl1_rate']:.3f} sl2={sl_rates['sl2_rate']:.3f}")

    df_all = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal rows: {len(df_all):,}")

    # Realized walk-MFE p90 audit
    # Realized signed-mfe-within-horizon (in trade direction): use max(delta_ticks across hold)
    # Quick approximation: use max(tp_K reached + walk-mid) — proxy: TP5 reached fraction
    # We'll compute actual p90 of realized signed-max-favorable-move from walk data.
    # For storage cost: compute realized signed max favorable from existing tp barriers as
    # the largest K for which tp_K was reached. (Coarse but sufficient for sanity.)
    max_reached = np.zeros(len(df_all), dtype=np.float64)
    for K in K_LIST:
        max_reached = np.where(df_all[f"tp{K}_dt_ns"].values >= 0,
                               np.maximum(max_reached, K), max_reached)
    # also push by mid_change_hold for partial-tick (less than 2) moves
    mch = df_all["mid_change_hold_ticks"].values
    max_reached = np.maximum(max_reached, np.where(mch > 0, mch, 0))
    realized_mfe_p90 = float(np.percentile(max_reached, 90))
    realized_mfe_p50 = float(np.percentile(max_reached, 50))
    print(f"Realized signed-MFE-within-15s (taker-walk): p50={realized_mfe_p50:.3f} "
          f"p90={realized_mfe_p90:.3f}  (HC #428 R2 p90 ceiling reference)")

    df_all.to_parquet(per_trade_path, index=False)
    print(f"Wrote per-trade parquet ({len(df_all):,} rows): {per_trade_path}")

    # ===== Cell sweep =====
    cells = []
    per_day_for_best = None
    best_key = None
    best_sharpe = -1e18

    print("\n== Sweeping cells (Q x K x S x gate) ==")
    for gate in GATES:
        for q in Q_LIST:
            mask = gate_mask_per_fold(df_all, gate, q)
            sub = df_all.loc[mask]
            if sub.empty:
                continue
            sub_idx = sub.index.values
            # === ORIGINAL DIRECTION (continuation hypothesis) ===
            # TP = +K ticks in predicted direction, SL = -S ticks against predicted
            for K in K_LIST:
                if K > P90_MFE_H10 + 1e-6:
                    continue
                for S in S_LIST:
                    if not (K > S):
                        continue  # asymmetric TP > SL only
                    tp_dt = sub[f"tp{K}_dt_ns"].values
                    sl_dt = sub[f"sl{S}_dt_ns"].values
                    mch = sub["mid_change_hold_ticks"].values
                    raw_ticks, outcome = compute_pnl_per_trade(tp_dt, sl_dt, mch, K, S)
                    net = raw_ticks - COST_TAKER
                    n_total = len(sub)
                    m = metrics(net, n_total)
                    hit = (outcome == "tp") | (outcome == "sl")
                    n_hit = int(hit.sum())
                    n_tp = int((outcome == "tp").sum())
                    n_sl = int((outcome == "sl").sum())
                    n_hold = int((outcome == "hold").sum())
                    cond_wr = (n_tp / n_hit) if n_hit > 0 else None
                    days = sub["oot_date"].values
                    per_day = {}
                    for d, v in zip(days, net):
                        per_day.setdefault(d, []).append(float(v))
                    n_days_pos = sum(1 for d, vs in per_day.items() if np.mean(vs) > 0)
                    cells.append({
                        "direction": "continuation",
                        "gate": gate, "Q_pct": q, "K": K, "S": S,
                        "n_total": n_total, "n_trades": m["n_trades"],
                        "n_tp": n_tp, "n_sl": n_sl, "n_hold": n_hold, "n_hit": n_hit,
                        "cond_WR_tp_vs_sl": cond_wr,
                        "WR_pos_net": m["WR"],
                        "mean_ticks_net": m["mean_ticks"],
                        "median_ticks_net": m["median_ticks"],
                        "PF": m["PF"], "Sharpe": m["Sharpe"],
                        "sum_ticks_net": m["sum_ticks"],
                        "n_days": len(per_day),
                        "n_days_positive": n_days_pos,
                    })
                    if m["Sharpe"] is not None and m["Sharpe"] > best_sharpe and m["n_trades"] >= 30:
                        best_sharpe = m["Sharpe"]
                        best_key = ("continuation", gate, q, K, S)
                        per_day_for_best = per_day

            # === FLIPPED DIRECTION (reversal hypothesis) ===
            # Trade OPPOSITE to predicted side. TP = K_rev ticks AGAINST original
            # prediction (== sl-barrier in our walk). SL = S_rev ticks WITH original
            # prediction (== tp-barrier in our walk). Asymmetric: K_rev > S_rev.
            for K_rev in S_LIST:  # K_rev = TP in flipped frame; against-pred barrier
                if K_rev > P90_MFE_H10 + 1e-6:
                    continue
                for S_rev in K_LIST:  # S_rev = SL in flipped frame; with-pred barrier
                    if not (K_rev > S_rev):
                        continue
                    tp_dt = sub[f"sl{K_rev}_dt_ns"].values  # flipped TP = original SL
                    sl_dt = sub[f"tp{S_rev}_dt_ns"].values  # flipped SL = original TP
                    # mid_change in original predicted direction — flip sign for reversal trade
                    mch = -sub["mid_change_hold_ticks"].values
                    raw_ticks, outcome = compute_pnl_per_trade(tp_dt, sl_dt, mch, K_rev, S_rev)
                    net = raw_ticks - COST_TAKER
                    n_total = len(sub)
                    m = metrics(net, n_total)
                    hit = (outcome == "tp") | (outcome == "sl")
                    n_hit = int(hit.sum())
                    n_tp = int((outcome == "tp").sum())
                    n_sl = int((outcome == "sl").sum())
                    n_hold = int((outcome == "hold").sum())
                    cond_wr = (n_tp / n_hit) if n_hit > 0 else None
                    days = sub["oot_date"].values
                    per_day = {}
                    for d, v in zip(days, net):
                        per_day.setdefault(d, []).append(float(v))
                    n_days_pos = sum(1 for d, vs in per_day.items() if np.mean(vs) > 0)
                    cells.append({
                        "direction": "reversal",
                        "gate": gate, "Q_pct": q, "K": K_rev, "S": S_rev,
                        "n_total": n_total, "n_trades": m["n_trades"],
                        "n_tp": n_tp, "n_sl": n_sl, "n_hold": n_hold, "n_hit": n_hit,
                        "cond_WR_tp_vs_sl": cond_wr,
                        "WR_pos_net": m["WR"],
                        "mean_ticks_net": m["mean_ticks"],
                        "median_ticks_net": m["median_ticks"],
                        "PF": m["PF"], "Sharpe": m["Sharpe"],
                        "sum_ticks_net": m["sum_ticks"],
                        "n_days": len(per_day),
                        "n_days_positive": n_days_pos,
                    })
                    if m["Sharpe"] is not None and m["Sharpe"] > best_sharpe and m["n_trades"] >= 30:
                        best_sharpe = m["Sharpe"]
                        best_key = ("reversal", gate, q, K_rev, S_rev)
                        per_day_for_best = per_day

    cells_df = pd.DataFrame(cells).sort_values("Sharpe", ascending=False, na_position="last")
    cells_path = OUT_DIR / "cells_summary.parquet"
    cells_df.to_parquet(cells_path, index=False)
    print(f"Wrote {len(cells_df)} cells: {cells_path}")
    print("\nTop 15 cells by Sharpe:")
    print(cells_df.head(15).to_string(index=False))

    # ===== Regime gate on best cell =====
    regime_block = None
    if per_day_for_best is not None:
        regime_block = regime_check(per_day_for_best)
        regime_block["best_cell"] = {
            "direction": best_key[0], "gate": best_key[1],
            "Q_pct": best_key[2], "K": best_key[3], "S": best_key[4],
            "Sharpe": best_sharpe,
        }

    regime_path = OUT_DIR / "regime_gate.json"
    with open(regime_path, "w") as f:
        json.dump({
            "elapsed_s_total": round(time.time() - t0, 1),
            "n_folds": len(folds),
            "oot_window": [OOT_MIN, OOT_MAX],
            "horizon_h_seconds": 10,
            "hold_cap_seconds": 15,
            "cost_taker_ticks": COST_TAKER,
            "p90_mfe_h10_HC428R2_ceiling": P90_MFE_H10,
            "realized_signed_mfe_p50": realized_mfe_p50,
            "realized_signed_mfe_p90": realized_mfe_p90,
            "fold_diag": fold_diag,
            "best_cell_regime": regime_block,
        }, f, indent=2, default=str)
    print(f"Wrote regime gate: {regime_path}")

    # ===== Verdict =====
    best_row = cells_df.iloc[0] if len(cells_df) > 0 else None
    lines = []
    lines.append("# Multi-K Asymmetric TAKER Walk — Verdict\n")
    lines.append(f"Generated {time.strftime('%Y-%m-%d %H:%M:%S ET')}\n")
    lines.append(f"OOT window: {OOT_MIN}..{OOT_MAX}, {len(folds)} folds, "
                 f"{len(df_all):,} events. Horizon h=10s, hold cap 15s, "
                 f"TAKER cost {COST_TAKER}t.\n")
    lines.append(f"Realized signed-MFE-within-15s: p50={realized_mfe_p50:.2f}t, "
                 f"p90={realized_mfe_p90:.2f}t (HC #428 R2 ceiling per session_state = 5.0t).\n")
    if best_row is not None:
        lines.append("## Best cell (by Sharpe, n_trades >= 30)\n")
        lines.append(f"- Direction: **{best_row['direction']}**\n")
        lines.append(f"- Gate: **{best_row['gate']}**, Q={best_row['Q_pct']}%, "
                     f"K={int(best_row['K'])} ticks TP, SL={int(best_row['S'])} ticks\n")
        lines.append(f"- n_trades: {int(best_row['n_trades'])} "
                     f"(of {int(best_row['n_total'])} gated). "
                     f"TP hits: {int(best_row['n_tp'])}, SL hits: {int(best_row['n_sl'])}, "
                     f"holds: {int(best_row['n_hold'])}.\n")
        lines.append(f"- Conditional WR (TP-before-SL | hit): "
                     f"{(best_row['cond_WR_tp_vs_sl'] or 0):.3f}\n")
        lines.append(f"- Mean ticks net: {(best_row['mean_ticks_net'] or 0):.3f}t  "
                     f"| Median: {(best_row['median_ticks_net'] or 0):.3f}t\n")
        lines.append(f"- PF: {(best_row['PF'] or 0):.3f}  | "
                     f"Sharpe: {(best_row['Sharpe'] or 0):.2f}\n")
        lines.append(f"- Sum ticks: {best_row['sum_ticks_net']:.1f}\n")
        lines.append(f"- Days positive: {int(best_row['n_days_positive'])} / "
                     f"{int(best_row['n_days'])}\n")
    if regime_block is not None:
        lines.append("\n## HC #428 R1 Regime Gate\n")
        lines.append(f"- Green days: {regime_block['n_green']}, "
                     f"Red days: {regime_block['n_red']}, "
                     f"Flat days: {regime_block['n_flat']}\n")
        lines.append(f"- Sharpe_green: {regime_block['sharpe_green']}\n")
        lines.append(f"- Sharpe_red:   {regime_block['sharpe_red']}\n")
        lines.append(f"- Asymmetry: {regime_block['asymmetry']}  "
                     f"(pass <= 0.50: {regime_block['pass_asymmetry_05']})\n")
        lines.append(f"- Day-concentration: {regime_block['day_concentration']}  "
                     f"(pass <= 0.70: {regime_block['pass_dayconc_07']})\n")
        lines.append(f"- **R1 GATE PASS: {regime_block['pass_R1_regime_gate']}**\n")

    # Decision
    lines.append("\n## Verdict\n")
    if best_row is None or best_row["mean_ticks_net"] is None:
        lines.append("- **NO-GO**: no valid cells.\n")
    elif best_row["mean_ticks_net"] > 0 and best_row["PF"] is not None and best_row["PF"] > 1.0 \
         and regime_block and regime_block["pass_R1_regime_gate"]:
        lines.append("- **GO**: best cell positive net of taker cost, PF > 1, R1 regime gate passes.\n")
        lines.append(f"  - Forward to FIFO grade: gate={best_row['gate']}, Q={best_row['Q_pct']}%, "
                     f"K={int(best_row['K'])}, S={int(best_row['S'])}.\n")
    else:
        reasons = []
        if best_row["mean_ticks_net"] is None or best_row["mean_ticks_net"] <= 0:
            reasons.append(f"mean ticks net = {best_row['mean_ticks_net']} (need > 0)")
        if best_row["PF"] is None or best_row["PF"] <= 1.0:
            reasons.append(f"PF = {best_row['PF']} (need > 1)")
        if regime_block and not regime_block["pass_R1_regime_gate"]:
            reasons.append(f"R1 regime gate FAIL "
                          f"(asym={regime_block['asymmetry']}, conc={regime_block['day_concentration']})")
        lines.append("- **NO-GO**. Reason(s): " + "; ".join(reasons) + ".\n")

    # Always show top 5
    lines.append("\n## Top 5 cells by Sharpe\n")
    lines.append("```\n")
    lines.append(cells_df.head(5).to_string(index=False) + "\n")
    lines.append("```\n")

    verdict_path = OUT_DIR / "verdict.md"
    verdict_path.write_text("".join(lines))
    print(f"Wrote verdict: {verdict_path}")

    print(f"\nTotal elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

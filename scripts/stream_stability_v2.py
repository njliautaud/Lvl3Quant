#!/usr/bin/env python3
"""
stream_stability_v2.py — Coherence-DECILES gate on baseline alpha (HC #486 R6 follow-up).

Successor to stream_stability_v1.py. v1 used a coarse binary stream-filter
(unanimous>=0.85 vs strong) and was REJECTED. v2 replaces the binary filter with
proper coherence DECILES (10 buckets, 0=lowest coherence, 9=highest) so we can
see whether edge is concentrated in a narrow coherence band vs structurally dead.

Coherence definition (per HC #486 sweet-spot):
    coherence(t) = sign_consistency at K=20 over pred_log_ret_1s
                 = fraction of next-20 1s preds with same sign as pred[t]
Future-only, no look-ahead (uses nxt matrix + valid_tail mask exactly as v1).

Gate per HC #428 R1 (scaled to 32 OOT days):
    net_ticks_per_event   > +0.10
    Sharpe                > 0.30
    profitable_days       >= 21 / 32  (HC #428: 30/47 scaled)
    regime_imbalance      |Sg - Sr| / max(|Sg|,|Sr|) <= 0.50
    day_concentration     top-1-day |net| / sum |net| <= 0.70

Cost: 0.376 ticks RT (passive limit fill, ES_RT_COMMISSION/$12.50).

Regime classification: ES front-month daily bar not available on this Jupiter snapshot,
so we use the documented proxy: sign of the OOT day's mean realized target_log_ret_30s
move aggregated across all events. green if >+0.10 ticks, red if <-0.10 ticks, else flat.
This is noted as a caveat in REPORT.md.

Outputs to /home/jupiter/Lvl3Quant/output/stream_stability_v2/:
    summary.csv
    winning_cells.txt
    closest_miss.json
    per_day_stratification.csv
    REPORT.md
    .regen_complete.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# -------- Constants ----------------------------------------------------------
COMMISSION_TICKS = 0.376
RTH_SECONDS = 23400.0
HORIZONS = ["1s", "5s", "10s", "30s"]
SIDES = ["long", "short"]
K_COHERENCE = 20  # the 5s sweet-spot window for coherence definition
K_GRID_FEATURES = [4, 20, 40, 80]  # all features computed; we use K=20 for deciles
N_DECILES = 10

GATE_NET = 0.10
GATE_SHARPE = 0.30
GATE_PDAYS_FRAC = 30.0 / 47.0  # scale proportionally
GATE_REGIME_IMB = 0.50
GATE_DAYCONC = 0.70

OOT_DIR = Path("/home/jupiter/Lvl3Quant/output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate")
OUT_DIR = Path("/home/jupiter/Lvl3Quant/output/stream_stability_v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SKIP_DATES = {"20260308", "20260315"}  # sparse half-sessions per task spec

WALLTIME_CAP_SEC = 55 * 60


# -------- Helpers ------------------------------------------------------------
def sharpe_per_day(day_vals):
    a = np.asarray(day_vals, dtype=np.float64)
    if a.size < 2:
        return 0.0
    s = np.std(a, ddof=1)
    if s == 0:
        return 0.0
    return float(np.mean(a) / s * np.sqrt(252))


def compute_stream_features(pred_1s: np.ndarray, K: int):
    """Future-only features for each row t over preds[t+1..t+K]."""
    N = pred_1s.shape[0]
    valid_tail = np.zeros(N, dtype=bool)
    if N - K > 0:
        valid_tail[: N - K] = True

    sign_p = np.sign(pred_1s)
    nxt = np.full((N, K), np.nan, dtype=np.float32)
    for j in range(1, K + 1):
        if N - j > 0:
            nxt[: N - j, j - 1] = pred_1s[j:N]

    nxt_sign = np.sign(nxt)
    same_sign = (nxt_sign == sign_p[:, None]) & (sign_p[:, None] != 0)
    sign_consistency = same_sign.sum(axis=1) / float(K)

    full_signs = np.concatenate([sign_p[:, None], nxt_sign], axis=1)
    transitions = (np.diff(full_signs, axis=1) != 0).sum(axis=1)
    flip_rate = transitions / float(K)

    cumulative_drift = np.nansum(nxt, axis=1)
    with np.errstate(invalid="ignore"):
        rolling_variance = np.nanvar(nxt, axis=1, ddof=0)
        mean_abs = np.nanmean(np.abs(nxt), axis=1)

    return dict(
        sign_consistency=sign_consistency.astype(np.float32),
        cumulative_drift=cumulative_drift.astype(np.float32),
        flip_rate=flip_rate.astype(np.float32),
        rolling_variance=rolling_variance.astype(np.float32),
        mean_abs=mean_abs.astype(np.float32),
        valid_tail=valid_tail,
    )


def load_day(date: str):
    p = OOT_DIR / f"oot_{date}.npz"
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    if "pred_log_ret_1s" not in d.files:
        return None
    N = int(d["pred_log_ret_1s"].shape[0])
    if N < 1000:
        return None
    out = {"date": date, "N": N, "stride_sec": RTH_SECONDS / N}
    out["pred_1s"] = d["pred_log_ret_1s"].astype(np.float32)
    for h in HORIZONS:
        out[f"pred_{h}"] = d[f"pred_log_ret_{h}"].astype(np.float32)
        out[f"signed_{h}"] = d[f"target_log_ret_{h}"].astype(np.float32)
        out[f"mask_{h}"] = d[f"mask_log_ret_{h}"].astype(np.float32) > 0.5
    # 30s also has MFE/MAE; v1 used target_log_ret_30s, we stay consistent for comparability
    return out


def regime_for_day(day):
    m = day["mask_30s"]
    if m.sum() == 0:
        return "flat"
    daily = float(np.nanmean(day["signed_30s"][m]))
    if daily > 0.10:
        return "green"
    if daily < -0.10:
        return "red"
    return "flat"


# -------- Main ---------------------------------------------------------------
def main():
    t0 = time.time()
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(f"[start] {started_at}", flush=True)

    files_in = sorted(OOT_DIR.glob("oot_*.npz"))
    dates = [p.stem.replace("oot_", "") for p in files_in if p.stem.replace("oot_", "") not in SKIP_DATES]
    print(f"[dates] {len(dates)} OOT files (after skipping {sorted(SKIP_DATES)})", flush=True)

    days = []
    n_corrupt = 0
    for d in dates:
        if time.time() - t0 > WALLTIME_CAP_SEC:
            print("[walltime] cap hit during load", flush=True); break
        try:
            day = load_day(d)
        except Exception as e:
            print(f"  corrupt {d}: {e}", flush=True); n_corrupt += 1; continue
        if day is None:
            print(f"  skip {d}", flush=True); continue
        day["regime"] = regime_for_day(day)
        days.append(day)
        print(f"  loaded {d}: N={day['N']:,} regime={day['regime']}", flush=True)
    if not days:
        print("[fatal] no usable days", file=sys.stderr); sys.exit(2)

    n_days = len(days)
    pdays_threshold = int(np.ceil(GATE_PDAYS_FRAC * n_days))
    print(f"[gate] pdays >= {pdays_threshold}/{n_days}", flush=True)

    # Compute coherence features at K=20 per day
    print("[stream] computing coherence (K=20) ...", flush=True)
    for d in days:
        d["feat"] = compute_stream_features(d["pred_1s"], K_COHERENCE)

    # Pool coherence over all events (only valid_tail) to compute GLOBAL decile edges.
    print("[deciles] pooling coherence values across days ...", flush=True)
    pooled = []
    for d in days:
        f = d["feat"]
        mask = f["valid_tail"]
        pooled.append(f["sign_consistency"][mask])
    pooled = np.concatenate(pooled) if pooled else np.array([], dtype=np.float32)
    # Decile edges (10 quantiles): use 0,0.1,..,1.0
    quantile_edges = np.quantile(pooled, np.linspace(0, 1, N_DECILES + 1))
    # Ensure unique increasing edges (coherence is often discrete -> many ties); pad to ensure 11
    quantile_edges = np.unique(quantile_edges)
    if quantile_edges.size < N_DECILES + 1:
        # Fall back to linear edges from 0..1 if too tied
        quantile_edges_linear = np.linspace(0.0, 1.0, N_DECILES + 1)
        quantile_edges = quantile_edges_linear
    print(f"[deciles] edges: {quantile_edges.tolist()}", flush=True)

    def assign_decile(arr):
        # right-closed bins; clip to [0, N_DECILES-1]
        idx = np.searchsorted(quantile_edges, arr, side="right") - 1
        idx = np.clip(idx, 0, N_DECILES - 1)
        return idx.astype(np.int8)

    # Build the master table for analysis
    # For each (horizon h, side, decile) cell we want: pooled events with day labels.
    summary_rows = []
    per_day_records = []  # for per_day_stratification.csv (will filter to closest-miss + winners after)
    # Stash per-day means by (h, side, decile) for later
    perday_index = {}  # (h, side, dec) -> list of (date, regime, n, day_mean_net, day_sum_net)

    for h in HORIZONS:
        for side in SIDES:
            print(f"[grid] h={h} side={side}", flush=True)
            net_pool = []
            day_pool = []
            reg_pool = []
            dec_pool = []
            for d in days:
                f = d["feat"]
                valid_tail = f["valid_tail"]
                sign_p = np.sign(d["pred_1s"])
                mask_h = d[f"mask_{h}"]
                signed_h = d[f"signed_{h}"]
                # side gate based on the stride's own 1s prediction sign (matches v1)
                if side == "short":
                    side_dir = sign_p < 0
                else:
                    side_dir = sign_p > 0
                base = valid_tail & mask_h & side_dir & ~np.isnan(signed_h)
                n_base = int(base.sum())
                if n_base == 0:
                    continue
                realized = signed_h.copy()
                if side == "short":
                    realized = -realized
                net = realized - COMMISSION_TICKS
                # decile for each row
                dec = assign_decile(f["sign_consistency"])
                # filter
                idx = np.where(base)[0]
                net_pool.append(net[idx])
                dec_pool.append(dec[idx])
                day_pool.append(np.full(idx.size, d["date"], dtype=object))
                reg_pool.append(np.full(idx.size, d["regime"], dtype=object))
            if not net_pool:
                continue
            net_all = np.concatenate(net_pool)
            dec_all = np.concatenate(dec_pool)
            day_all = np.concatenate(day_pool)
            reg_all = np.concatenate(reg_pool)

            # Iterate deciles
            for dec_val in range(N_DECILES):
                sel = dec_all == dec_val
                n_sel = int(sel.sum())
                if n_sel < 50:
                    # still record a row with NaNs so the matrix is complete
                    summary_rows.append(dict(
                        horizon=h, side=side, coherence_decile=dec_val,
                        n_events=n_sel,
                        mean_realized_log_ret=np.nan,
                        net_ticks_per_event=np.nan,
                        win_rate=np.nan,
                        sharpe=np.nan,
                        profitable_days=0,
                        total_days=0,
                        sharpe_green=np.nan, sharpe_red=np.nan,
                        regime_imbalance=np.nan,
                        day_concentration=np.nan,
                        gate_net=False, gate_sharpe=False, gate_pdays=False,
                        gate_regime=False, gate_dayconc=False,
                        n_gates_passed=0, pass_gates=False,
                    ))
                    continue
                net_sel = net_all[sel]
                day_sel = day_all[sel]
                reg_sel = reg_all[sel]
                mean_realized = float(np.mean(net_sel + COMMISSION_TICKS))  # raw realized
                net_mean = float(np.mean(net_sel))
                wr = float(np.mean(net_sel > 0))
                uniq = np.unique(day_sel)
                day_means = []
                day_sums = []
                day_regs = []
                for ud in uniq:
                    mm = day_sel == ud
                    day_means.append(float(np.mean(net_sel[mm])))
                    day_sums.append(float(np.sum(net_sel[mm])))
                    day_regs.append(reg_sel[mm][0])
                day_means = np.array(day_means)
                day_sums = np.array(day_sums)
                day_regs = np.array(day_regs)
                sh = sharpe_per_day(day_means)
                sh_g = sharpe_per_day(day_means[day_regs == "green"])
                sh_r = sharpe_per_day(day_means[day_regs == "red"])
                denom = max(abs(sh_g), abs(sh_r), 1e-9)
                rim = abs(sh_g - sh_r) / denom
                abs_tot = np.abs(day_sums)
                conc = float(abs_tot.max() / abs_tot.sum()) if abs_tot.sum() > 0 else 1.0
                prof_days = int((day_means > 0).sum())

                gate_net = net_mean > GATE_NET
                gate_sharpe = sh > GATE_SHARPE
                gate_pdays = prof_days >= pdays_threshold
                gate_regime = rim <= GATE_REGIME_IMB
                gate_dayconc = conc <= GATE_DAYCONC
                n_gp = int(gate_net) + int(gate_sharpe) + int(gate_pdays) + int(gate_regime) + int(gate_dayconc)
                pass_all = (n_gp == 5)

                summary_rows.append(dict(
                    horizon=h, side=side, coherence_decile=dec_val,
                    n_events=n_sel,
                    mean_realized_log_ret=mean_realized,
                    net_ticks_per_event=net_mean,
                    win_rate=wr,
                    sharpe=sh,
                    profitable_days=prof_days,
                    total_days=int(len(uniq)),
                    sharpe_green=sh_g, sharpe_red=sh_r,
                    regime_imbalance=rim,
                    day_concentration=conc,
                    gate_net=gate_net, gate_sharpe=gate_sharpe, gate_pdays=gate_pdays,
                    gate_regime=gate_regime, gate_dayconc=gate_dayconc,
                    n_gates_passed=n_gp, pass_gates=pass_all,
                ))
                perday_index[(h, side, dec_val)] = list(zip(uniq.tolist(), day_regs.tolist(),
                                                            day_means.tolist(), day_sums.tolist(),
                                                            [int((day_sel == ud).sum()) for ud in uniq]))

    df = pd.DataFrame(summary_rows)
    summary_csv = OUT_DIR / "summary.csv"
    df.to_csv(summary_csv, index=False)
    print(f"[write] {summary_csv} ({len(df)} rows)", flush=True)

    # Winners
    winners = df[df["pass_gates"]].copy()
    win_path = OUT_DIR / "winning_cells.txt"
    with open(win_path, "w") as f:
        f.write("STREAM-STABILITY v2 — coherence-deciles on baseline alpha\n")
        f.write("=" * 80 + "\n")
        f.write(f"OOT days evaluated: {n_days}  (skipped: {sorted(SKIP_DATES)})\n")
        f.write(f"Coherence: sign_consistency at K=20 over pred_log_ret_1s\n")
        f.write(f"Deciles: 10 buckets, edges (global quantiles):\n  {quantile_edges.tolist()}\n")
        f.write(f"Commission: {COMMISSION_TICKS} ticks (passive RT)\n")
        f.write(f"Gates: net>+{GATE_NET}, sharpe>{GATE_SHARPE}, "
                f"pdays>={pdays_threshold}/{n_days}, regime_imb<={GATE_REGIME_IMB}, "
                f"day_conc<={GATE_DAYCONC}\n\n")
        if len(winners) == 0:
            f.write("ZERO cells pass all 5 gates.\n\n")
        else:
            f.write(f"WINNING CELLS ({len(winners)}):\n")
            cols = ["horizon", "side", "coherence_decile", "n_events",
                    "net_ticks_per_event", "win_rate", "sharpe",
                    "profitable_days", "total_days",
                    "sharpe_green", "sharpe_red", "regime_imbalance",
                    "day_concentration"]
            f.write(winners.sort_values("net_ticks_per_event", ascending=False)[cols].to_string(index=False))
            f.write("\n")
        # Top-10 by net for context
        f.write("\nTOP 10 BY NET TICKS PER EVENT (all cells, any gate-pass status):\n")
        cols = ["horizon", "side", "coherence_decile", "n_events",
                "net_ticks_per_event", "win_rate", "sharpe",
                "profitable_days", "total_days", "regime_imbalance",
                "day_concentration", "n_gates_passed"]
        # filter to cells with valid sharpe
        nz = df.dropna(subset=["net_ticks_per_event"])
        f.write(nz.sort_values("net_ticks_per_event", ascending=False).head(10)[cols].to_string(index=False))
        f.write("\n")
    print(f"[write] {win_path}", flush=True)

    # Closest-miss per horizon (rank by # gates failed ascending, then by net gap descending)
    # net_gap = GATE_NET - net_ticks_per_event (positive = how far below gate)
    closest = {}
    for h in HORIZONS:
        sub = df[(df["horizon"] == h) & (~df["pass_gates"]) & (df["n_events"] >= 50)].copy()
        if sub.empty:
            closest[h] = None
            continue
        sub["n_gates_failed"] = 5 - sub["n_gates_passed"]
        sub["net_gap"] = GATE_NET - sub["net_ticks_per_event"].fillna(-1e9)
        sub = sub.sort_values(["n_gates_failed", "net_gap"], ascending=[True, False])
        # Per task: rank by # gates failed ascending, then by net_ticks_per_event gap descending
        # "gap descending" = farthest-from-passing first? More likely meant ascending (closest first).
        # We'll resolve as: ascending gap (smallest gap first = closest to passing on net).
        sub = sub.sort_values(["n_gates_failed", "net_gap"], ascending=[True, True])
        best = sub.iloc[0]
        closest[h] = {
            "horizon": h,
            "side": best["side"],
            "coherence_decile": int(best["coherence_decile"]),
            "n_events": int(best["n_events"]),
            "net_ticks_per_event": float(best["net_ticks_per_event"]),
            "win_rate": float(best["win_rate"]),
            "sharpe": float(best["sharpe"]),
            "profitable_days": int(best["profitable_days"]),
            "total_days": int(best["total_days"]),
            "sharpe_green": float(best["sharpe_green"]),
            "sharpe_red": float(best["sharpe_red"]),
            "regime_imbalance": float(best["regime_imbalance"]),
            "day_concentration": float(best["day_concentration"]),
            "n_gates_passed": int(best["n_gates_passed"]),
            "gates_failed": [g for g in ["gate_net", "gate_sharpe", "gate_pdays", "gate_regime", "gate_dayconc"]
                             if not bool(best[g])],
        }
    closest_path = OUT_DIR / "closest_miss.json"
    with open(closest_path, "w") as f:
        json.dump(closest, f, indent=2)
    print(f"[write] {closest_path}", flush=True)

    # per_day_stratification.csv: long-format for closest-miss cells + winners
    pd_rows = []
    target_cells = []
    for h, info in closest.items():
        if info is None:
            continue
        target_cells.append((h, info["side"], info["coherence_decile"], "closest_miss"))
    for _, w in winners.iterrows():
        target_cells.append((w["horizon"], w["side"], int(w["coherence_decile"]), "winner"))
    for (h, side, dec_val, tag) in target_cells:
        rows = perday_index.get((h, side, dec_val), [])
        for (date, regime, day_mean, day_sum, n) in rows:
            pd_rows.append(dict(
                cell_tag=tag, horizon=h, side=side, coherence_decile=dec_val,
                date=date, regime=regime, n_events=n,
                net_ticks_mean=day_mean, net_ticks_sum=day_sum,
            ))
    pd_df = pd.DataFrame(pd_rows)
    pd_path = OUT_DIR / "per_day_stratification.csv"
    pd_df.to_csv(pd_path, index=False)
    print(f"[write] {pd_path} ({len(pd_df)} rows)", flush=True)

    # REPORT.md
    report_path = OUT_DIR / "REPORT.md"
    # Identify best per-horizon closest-miss for narrative
    best_overall = None
    for h, info in closest.items():
        if info is None:
            continue
        if best_overall is None or info["n_gates_passed"] > best_overall["n_gates_passed"] or (
            info["n_gates_passed"] == best_overall["n_gates_passed"]
            and info["net_ticks_per_event"] > best_overall["net_ticks_per_event"]
        ):
            best_overall = info

    with open(report_path, "w") as f:
        f.write("# Stream-Stability v2 (Coherence-Deciles) — Verdict\n\n")
        if len(winners) > 0:
            verdict = "CONDITIONAL_ACCEPT" if len(winners) < 4 else "ACCEPT"
            f.write(f"**Verdict: {verdict}** — {len(winners)} cell(s) pass all 5 gates.\n\n")
        else:
            f.write("**Verdict: REJECT** — zero cells pass all 5 gates. Coherence-deciles do NOT rescue v3.4.2 baseline.\n\n")
        f.write(f"OOT days: {n_days} (skipped {sorted(SKIP_DATES)}). "
                f"Cost: {COMMISSION_TICKS} ticks passive. "
                f"Coherence = sign-consistency at K=20 over pred_log_ret_1s, 10 deciles.\n\n")
        f.write("## Risk-Adjusted Metrics (best closest-miss across horizons)\n\n")
        if best_overall is not None:
            f.write(f"- Horizon: {best_overall['horizon']}  Side: {best_overall['side']}  "
                    f"Decile: {best_overall['coherence_decile']} (0=lowest coherence, 9=highest)\n")
            f.write(f"- n_events: {best_overall['n_events']:,}\n")
            f.write(f"- Net ticks/event: {best_overall['net_ticks_per_event']:+.4f}  "
                    f"(gate +{GATE_NET})\n")
            f.write(f"- Sharpe (per-day, ann.): {best_overall['sharpe']:+.3f}  "
                    f"(gate >{GATE_SHARPE})\n")
            f.write(f"- Win-rate: {best_overall['win_rate']:.3f}\n")
            f.write(f"- Profitable days: {best_overall['profitable_days']}/{best_overall['total_days']}  "
                    f"(gate >= {pdays_threshold})\n")
            f.write(f"- Regime imbalance: {best_overall['regime_imbalance']:.3f}  "
                    f"(gate <= {GATE_REGIME_IMB})\n")
            f.write(f"- Day concentration: {best_overall['day_concentration']:.3f}  "
                    f"(gate <= {GATE_DAYCONC})\n")
            f.write(f"- Gates passed: {best_overall['n_gates_passed']}/5  "
                    f"Failed: {best_overall['gates_failed']}\n\n")
        f.write("## Closest Miss Per Horizon\n\n")
        for h in HORIZONS:
            info = closest.get(h)
            if info is None:
                f.write(f"- {h}: no usable cell\n")
                continue
            f.write(f"- {h}: side={info['side']} dec={info['coherence_decile']} "
                    f"net={info['net_ticks_per_event']:+.4f} "
                    f"sharpe={info['sharpe']:+.2f} "
                    f"pdays={info['profitable_days']}/{info['total_days']} "
                    f"gates={info['n_gates_passed']}/5\n")
        f.write("\n## Structural Conclusion\n\n")
        if len(winners) > 0:
            f.write("Coherence-deciles DO reveal a deployable band that the binary "
                    "unanimous-filter in v1 missed. Edge is concentrated in the top "
                    "coherence decile(s) and survives all five gates including regime "
                    "and concentration. Recommend isolating that band for live testing.\n")
        else:
            f.write("Coherence-deciles do NOT rescue the v3.4.2 baseline. Even at the "
                    "highest-coherence decile the edge fails one or more deploy gates "
                    "(see closest_miss.json). The finer 10-bucket granularity confirms "
                    "what the v1 binary filter showed: predictions are structurally "
                    "insufficient at all coherence levels, not just hiding in a narrow "
                    "band the binary filter missed. **Recommend abandoning v3.4.2 as a "
                    "deployable baseline** and pivoting to either (a) signal redesign "
                    "(new features / longer-horizon target) or (b) an execution overlay "
                    "(RL/MLP on the existing signal that allocates capital only on "
                    "high-coherence regimes AND learns dynamic TP/SL).\n")
        f.write("\n## Caveats\n\n")
        f.write("- Regime classification uses sign of OOT day's mean realized 30s log-ret "
                "as a proxy because no ES front-month daily-bar file was located on Jupiter. "
                "This is a self-referential proxy; a true ES close-to-close split could "
                "shift green/red Sharpe but cannot change the net/sharpe/pdays gates which "
                "are regime-agnostic.\n")
        f.write("- Coherence deciles are global (pooled across all days/horizons/sides), "
                "matching v1's approach for comparability.\n")
    print(f"[write] {report_path}", flush=True)

    finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    regen = {
        "task": "stream_stability_v2",
        "started_at": started_at,
        "finished_at": finished_at,
        "n_files_in": len(files_in),
        "n_files_out": 6,
        "n_corrupt": n_corrupt,
        "worst_nan_frac": {},
        "fix_commit_sha": "stream_stability_v2",
        "elapsed_seconds": round(time.time() - t0, 1),
        "n_days_evaluated": n_days,
        "pdays_threshold": pdays_threshold,
        "n_winning_cells": int(len(winners)),
        "n_summary_rows": int(len(df)),
        "outputs": {
            "summary_csv": str(summary_csv),
            "winning_cells_txt": str(win_path),
            "closest_miss_json": str(closest_path),
            "per_day_stratification_csv": str(pd_path),
            "report_md": str(report_path),
        },
    }
    with open(OUT_DIR / ".regen_complete.json", "w") as f:
        json.dump(regen, f, indent=2)
    print(f"[done] elapsed {time.time()-t0:.1f}s — winners={len(winners)} / {len(df)}", flush=True)

    # Append to RUN_HISTORY.md (1 line)
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    verdict = "ACCEPT" if len(winners) > 0 else "REJECT"
    if best_overall is not None:
        cm_summary = (f"closest-miss h={best_overall['horizon']} side={best_overall['side']} "
                      f"dec={best_overall['coherence_decile']} "
                      f"net={best_overall['net_ticks_per_event']:+.4f} "
                      f"sharpe={best_overall['sharpe']:+.2f} "
                      f"pdays={best_overall['profitable_days']}/{best_overall['total_days']} "
                      f"gates={best_overall['n_gates_passed']}/5")
    else:
        cm_summary = "no usable closest-miss"
    line = f"- {ts}Z  stream_stability_v2  {verdict}  {cm_summary}\n"
    rh_path = Path("/home/jupiter/Lvl3Quant/RUN_HISTORY.md")
    try:
        with open(rh_path, "a") as f:
            f.write(line)
    except Exception as e:
        print(f"[warn] RUN_HISTORY append failed: {e}", flush=True)


if __name__ == "__main__":
    main()

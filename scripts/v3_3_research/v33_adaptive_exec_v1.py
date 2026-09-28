"""
v33_adaptive_exec_v1.py — HC #375 Track B v1 — Alpha-model-driven adaptive exit.

Hypothesis (per user 2026-05-15 18:12 ET):
  "A single snapshot prediction about direction and holding for a set amount
   of time without any other confluence mid trade hold is bad. Some way of
   using the alpha MODEL itself for entry and exit might prove adaptive and good."

Method:
  For each K=2 LONG entry, walk forward in 250ms strides reading the EVOLVING
  32 v3.3 head outputs. Each stride s gives an UPDATED estimate of:
    - pred_fifo_tp4sl3_hit_tp[entry+s]   — model's current belief in TP hit
    - pred_pred_mfe_60s_ticks[entry+s]   — model's current upside estimate
    - pred_log_ret_60s[entry+s]          — model's current directional bet
    - pred_pred_mae_60s_ticks[entry+s]   — model's current downside risk
  Δ_X[s] = pred_X[entry+s] − pred_X[entry]  (deviation from entry-time view)

  exit_score[s] = α_hit·Δ_hit_tp + α_mfe·Δ_mfe60 + α_log·Δ_log_ret_60s − α_mae·Δ_mae60

  EXIT POLICY: walk through candidate exit horizons {1s, 5s, 10s, 30s, 60s}
    = strides {4, 20, 40, 120, 240}. If exit_score < θ at the first reachable
    candidate stride → exit there. Else hold to 30s (static baseline cap).

P&L: net_ticks = (signed_realized_logret * TICKS_PER_LOGRET) − COMMISSION_RT_TICKS
  where signed_realized = target_log_ret_Xs[entry] for the chosen horizon X.

Falsification gates (must ALL pass):
  1. adaptive Sharpe > static-30s Sharpe on full K=2 set
  2. adaptive Sharpe > simple TP/SL replay (existing y_final_net_ticks IS that — same gate as 1)
  3. day-conc ≤ 20% on ≥10 firing days
       — currently 5 OOT days → REQUIRES extended OOT NPZ (in-progress on Neptune)
  4. p<0.10 under within-fold permutation null on (α, θ) tuning
       — perm shuffles label rank within day, retunes (α, θ), recomputes adaptive Sharpe

This v1 is METHODOLOGY VALIDATION on the 5-day K=2 set. Honest verdict (gate 3)
waits for `fold_00_extended_oot_predictions.npz` to land on Neptune.

Output: output/v3_3_full_execution_analysis_20260514/adaptive_exec_v1/
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ─── ES futures cost constants (canonical, per CLAUDE.md) ───────────────
TICK_SIZE_PTS = 0.25
ES_TICK_VALUE_USD = 12.50
RT_COMMISSION_USD = 4.70
RT_COMMISSION_TICKS = RT_COMMISSION_USD / ES_TICK_VALUE_USD  # 0.376

# IMPORTANT (verified 2026-05-15 18:30 ET): `target_log_ret_*s` arrays in the
# fold_00_predictions.npz are stored ALREADY IN TICKS (naming is historical /
# misleading). target_log_ret_1s std=1.63 ticks, p1=-4, p99=+5 — consistent
# with realized ES tick moves. NO scaling factor needed.
#
# Also verified: target_log_ret_60s, target_log_ret_5min, target_pred_mfe_60s_ticks,
# target_pred_mae_60s_ticks are ALL ZERO (heads never trained). Switch to 30s
# variants which ARE trained.
TICKS_FROM_TARGET = 1.0  # no conversion — already ticks

# Candidate adaptive-exit horizons MUST be in {1, 5, 10, 30} since 60s/5min
# targets are all-zero (untrained heads). 30s is the static baseline cap.
EXIT_HORIZONS_SEC = [1, 5, 10, 30]
STRIDE_PER_SEC = 4  # 250ms stride
EXIT_STRIDES = [h * STRIDE_PER_SEC for h in EXIT_HORIZONS_SEC]
STATIC_BASELINE_SEC = 30  # K=2 default static hold
HOLD_CAP_STRIDES = STATIC_BASELINE_SEC * STRIDE_PER_SEC

# Paths
ROOT = Path("/home/jupiter/Lvl3Quant")
K2_DATASET = ROOT / "output/v3_3_full_execution_analysis_20260514/exec_decision_dataset/decision_dataset.parquet"
PRED_NPZ = ROOT / "output/cnn_mamba_v3_3_uncertainty_weighted/fold_00_predictions.npz"
OUT_DIR = ROOT / "output/v3_3_full_execution_analysis_20260514/adaptive_exec_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ─── Load ───────────────────────────────────────────────────────────────
def load_k2_and_npz():
    print(f"[load] K=2 dataset: {K2_DATASET}", flush=True)
    df = pd.read_parquet(K2_DATASET)
    print(f"[load] K=2 trades: {len(df)}, days: {sorted(df['date'].unique())}", flush=True)

    print(f"[load] per-stride NPZ: {PRED_NPZ}", flush=True)
    npz = np.load(PRED_NPZ, allow_pickle=True)
    print(f"[load] strides: {int(npz['n_samples'])}, OOT dates: {list(npz['oot_dates'])}", flush=True)

    # Heads needed — ONLY trained heads (30s variants since 60s untrained)
    needed = {
        "hit_tp": npz["pred_fifo_tp4sl3_hit_tp"],     # P(TP+4 hit | entry) — trained
        "mfe30": npz["pred_pred_mfe_30s_ticks"],      # predicted max favorable, ticks
        "logret30": npz["pred_log_ret_30s"],          # predicted 30s realized ret (ticks)
        "mae30": npz["pred_pred_mae_30s_ticks"],      # predicted max adverse, ticks
    }
    realized = {
        f"logret_{h}s": npz[f"target_log_ret_{h}s"] for h in EXIT_HORIZONS_SEC
    }
    # Sanity: assert all realized targets are populated (not the broken 60s/5min)
    for k, arr in realized.items():
        nz = arr[~np.isnan(arr)]
        if len(nz) == 0 or (nz == 0).all():
            raise RuntimeError(f"realized target {k} is all-zero/NaN — head not trained")
    return df, needed, realized


# ─── Adaptive policy ─────────────────────────────────────────────────────
def compute_exit_score_series(entry_idx: int, heads: dict, max_stride: int) -> np.ndarray:
    """For one entry, return exit_score Δ-values at candidate exit strides.

    Δ_X[s] = pred_X[entry+s] - pred_X[entry].
    Long-trade convention: exit when score becomes NEGATIVE (signal deteriorating).
    """
    e = entry_idx
    end = min(e + max_stride, len(heads["hit_tp"]) - 1)
    # Reference (entry-time) values
    ref = {k: float(v[e]) for k, v in heads.items()}
    out = np.full(len(EXIT_STRIDES), np.nan, dtype=np.float64)
    return ref, end


def run_policy(df: pd.DataFrame, heads: dict, realized: dict,
               alpha_hit: float, alpha_mfe: float, alpha_log: float, alpha_mae: float,
               theta: float) -> dict:
    """Apply adaptive-exit policy across all K=2 trades. Return per-trade exit + p&l."""
    N = len(df)
    n_strides_total = len(heads["hit_tp"])

    exit_horizon_sec = np.full(N, STATIC_BASELINE_SEC, dtype=np.int32)  # default = static cap
    exit_reason = np.full(N, "static_30s", dtype=object)
    realized_net_ticks = np.zeros(N, dtype=np.float64)

    entry_idx = df["entry_global_idx"].to_numpy()

    for i in range(N):
        e = entry_idx[i]
        if e >= n_strides_total - 4:
            # Not enough forward strides — bail to static
            realized_net_ticks[i] = df["y_final_net_ticks"].iloc[i]
            continue

        # Reference (entry-time) head values
        ref_hit = float(heads["hit_tp"][e])
        ref_mfe = float(heads["mfe30"][e])
        ref_log = float(heads["logret30"][e])
        ref_mae = float(heads["mae30"][e])

        exited = False
        for hi, s in enumerate(EXIT_STRIDES):
            if e + s >= n_strides_total:
                break
            d_hit = float(heads["hit_tp"][e + s]) - ref_hit
            d_mfe = float(heads["mfe30"][e + s]) - ref_mfe
            d_log = float(heads["logret30"][e + s]) - ref_log
            d_mae = float(heads["mae30"][e + s]) - ref_mae
            score = (alpha_hit * d_hit + alpha_mfe * d_mfe
                     + alpha_log * d_log - alpha_mae * d_mae)
            # If score deteriorates below θ AT this candidate stride → exit
            if score < theta:
                h_sec = EXIT_HORIZONS_SEC[hi]
                exit_horizon_sec[i] = h_sec
                # P&L: realized return at horizon h_sec from entry (already in ticks)
                #      minus RT commission. For LONG K=2 trades, positive=good.
                if h_sec == STATIC_BASELINE_SEC:
                    # 30s adaptive exit ≡ static-30s baseline outcome (TP/SL or timeout)
                    realized_net_ticks[i] = df["y_final_net_ticks"].iloc[i]
                    exit_reason[i] = "adaptive_30s_=static"
                else:
                    realized_ticks = float(realized[f"logret_{h_sec}s"][e])
                    realized_net_ticks[i] = realized_ticks * TICKS_FROM_TARGET - RT_COMMISSION_TICKS
                    exit_reason[i] = f"adaptive_{h_sec}s"
                exited = True
                break

        if not exited:
            # Held to 30s cap → realized = K=2 baseline y_final_net_ticks (TP/SL or 30s timeout)
            realized_net_ticks[i] = df["y_final_net_ticks"].iloc[i]
            exit_reason[i] = "static_30s_hold"

    return {
        "exit_horizon_sec": exit_horizon_sec,
        "exit_reason": exit_reason,
        "realized_net_ticks": realized_net_ticks,
    }


def metrics(pnl: np.ndarray) -> dict:
    pnl = np.asarray(pnl, dtype=np.float64)
    n = len(pnl)
    if n == 0:
        return dict(n=0)
    mu = float(pnl.mean())
    sd = float(pnl.std(ddof=1)) if n > 1 else 0.0
    sharpe = mu / sd * np.sqrt(252.0) if sd > 0 else 0.0  # crude annualize per "day-of-trade"
    wr = float((pnl > 0).mean())
    sortino = mu / float(pnl[pnl < 0].std(ddof=1)) * np.sqrt(252.0) if (pnl < 0).any() else 0.0
    pf_num = float(pnl[pnl > 0].sum())
    pf_den = -float(pnl[pnl < 0].sum())
    pf = pf_num / pf_den if pf_den > 0 else float("inf")
    return dict(
        n=n, mean_ticks=mu, std_ticks=sd, sharpe=sharpe, sortino=sortino,
        pf=pf, wr=wr, total_ticks=float(pnl.sum()),
    )


def day_concentration(df: pd.DataFrame, pnl: np.ndarray) -> dict:
    daily = pd.Series(pnl).groupby(df["date"].to_numpy()).sum()
    total = daily.abs().sum()
    if total == 0:
        return dict(max_share=0.0, days=int(len(daily)))
    top_day = daily.abs().idxmax()
    max_share = float(daily.abs().max() / total)
    return dict(
        max_share=max_share, top_day=int(top_day),
        days=int(len(daily)), daily=daily.to_dict(),
    )


# ─── Main ────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    df, heads, realized = load_k2_and_npz()

    # Static baseline
    static_pnl = df["y_final_net_ticks"].to_numpy()
    static_m = metrics(static_pnl)
    static_dc = day_concentration(df, static_pnl)
    print(f"\n[static] n={static_m['n']} mean={static_m['mean_ticks']:+.4f}t  Sharpe={static_m['sharpe']:+.2f}  WR={static_m['wr']:.3f}  PF={static_m['pf']:.2f}  day_conc={static_dc['max_share']:.3f} days={static_dc['days']}", flush=True)

    # Adaptive — α grid + θ grid (v1: methodology validation on 5-day OOT)
    # All Δs in ticks/probs. θ negative = deterioration threshold.
    results = []
    grid = [
        # (α_hit, α_mfe, α_log, α_mae, θ)  — θ in score units
        (1.0, 0.0, 0.0, 0.0, -0.05),   # hit_tp Δ only
        (1.0, 0.0, 0.0, 0.0, -0.10),
        (1.0, 0.0, 0.0, 0.0, -0.15),
        (0.0, 1.0, 0.0, 0.0, -0.30),   # mfe30 Δ only (ticks)
        (0.0, 1.0, 0.0, 0.0, -0.50),
        (0.0, 0.0, 1.0, 0.0, -0.10),   # log_ret_30s Δ only (ticks)
        (0.0, 0.0, 1.0, 0.0, -0.20),
        (0.0, 0.0, 0.0, 1.0, -0.30),   # mae30 Δ only (Δ_mae > 0 means MAE worsening → exit)
        (1.0, 1.0, 0.0, 1.0, -0.20),   # blended (no log)
        (1.0, 1.0, 1.0, 1.0, -0.30),   # blended all heads
        (2.0, 1.0, 1.0, 1.0, -0.20),   # weight hit_tp higher
    ]
    for (ah, amf, al, ame, th) in grid:
        res = run_policy(df, heads, realized, ah, amf, al, ame, th)
        m = metrics(res["realized_net_ticks"])
        dc = day_concentration(df, res["realized_net_ticks"])
        exit_dist = pd.Series(res["exit_reason"]).value_counts().to_dict()
        row = dict(
            alpha_hit=ah, alpha_mfe=amf, alpha_log=al, alpha_mae=ame, theta=th,
            **m, max_day_share=dc["max_share"], days=dc["days"],
            exit_dist=exit_dist,
        )
        results.append(row)
        print(f"[adaptive ah={ah:.1f} amf={amf:.1f} al={al:g} ame={ame:.1f} θ={th:+.0e}]  "
              f"mean={m['mean_ticks']:+.4f}t Sharpe={m['sharpe']:+.2f} WR={m['wr']:.3f} "
              f"PF={m['pf']:.2f} dc={dc['max_share']:.3f} exits={ {k:v for k,v in list(exit_dist.items())[:4]} }",
              flush=True)

    # ─── Look-elsewhere null test ─────────────────────────────────────
    # Shuffle entry_global_idx WITHIN DAY so heads decouple from real trade
    # context. Re-run ALL 11 configs, take BEST Sharpe per perm. Compare
    # distribution of null-best-Sharpes vs real-best-Sharpe.
    real_best = max(r["sharpe"] for r in results)
    real_best_dc = max((r["max_day_share"] for r in results if r["sharpe"] == real_best), default=1.0)
    print(f"\n[null] Running 50 within-day permutations × {len(grid)} configs...", flush=True)
    rng = np.random.default_rng(42)
    null_best = []
    df_orig = df.copy()
    for perm_i in range(50):
        # Shuffle entry_global_idx within day
        df_perm = df_orig.copy()
        for day in df_perm["date"].unique():
            mask = (df_perm["date"] == day).to_numpy()
            idx_to_shuffle = df_perm.loc[mask, "entry_global_idx"].to_numpy().copy()
            rng.shuffle(idx_to_shuffle)
            df_perm.loc[mask, "entry_global_idx"] = idx_to_shuffle
        perm_sharpes = []
        for (ah, amf, al, ame, th) in grid:
            res_p = run_policy(df_perm, heads, realized, ah, amf, al, ame, th)
            m_p = metrics(res_p["realized_net_ticks"])
            perm_sharpes.append(m_p["sharpe"])
        null_best.append(max(perm_sharpes))
        if (perm_i + 1) % 10 == 0:
            arr = np.array(null_best)
            print(f"[null] perm {perm_i+1}/50 — null_best_Sharpe mean={arr.mean():.2f} p95={np.percentile(arr,95):.2f}  (real_best={real_best:.2f})", flush=True)
    null_arr = np.array(null_best)
    p_value = float((null_arr >= real_best).mean())
    null_summary = dict(
        n_perm=len(null_arr),
        real_best_sharpe=real_best,
        null_mean=float(null_arr.mean()),
        null_std=float(null_arr.std(ddof=1)),
        null_p95=float(np.percentile(null_arr, 95)),
        null_max=float(null_arr.max()),
        p_value=p_value,
        pass_p10=bool(p_value < 0.10),
    )
    print(f"\n[null] REAL_BEST={real_best:+.2f}  NULL_MEAN={null_arr.mean():+.2f}  NULL_P95={np.percentile(null_arr,95):+.2f}  p={p_value:.3f}  PASS_p10={'YES' if p_value < 0.10 else 'NO'}", flush=True)

    out = dict(
        elapsed_sec=time.time() - t0,
        n_trades=len(df),
        days=sorted(int(d) for d in df["date"].unique()),
        static_baseline=dict(metrics=static_m, day_conc=static_dc),
        adaptive_results=results,
        null_test=null_summary,
        notes=dict(
            falsification_gate_3_blocked_until_extended_oot=True,
            extended_oot_target="/home/nick/Lvl3Quant/output/cnn_mamba_v3_3_uncertainty_weighted/fold_00_extended_oot_predictions.npz",
            v1_methodology_only=True,
            corrected_target_log_ret_already_in_ticks=True,
            switched_60s_heads_to_30s_due_to_untrained=True,
        ),
    )
    out_file = OUT_DIR / "adaptive_exec_v1_results.json"
    with open(out_file, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n[done] elapsed={out['elapsed_sec']:.1f}s  output: {out_file}", flush=True)

    # Highlight best adaptive vs static
    best = max(results, key=lambda r: r["sharpe"])
    delta_sharpe = best["sharpe"] - static_m["sharpe"]
    print(f"\n[verdict] BEST adaptive Sharpe={best['sharpe']:+.2f} vs static={static_m['sharpe']:+.2f}  "
          f"ΔSharpe={delta_sharpe:+.2f}  cfg=(ah={best['alpha_hit']}, amf={best['alpha_mfe']}, "
          f"al={best['alpha_log']}, ame={best['alpha_mae']}, θ={best['theta']})", flush=True)
    if best["max_day_share"] > 0.20:
        print(f"[verdict] FAIL gate 3 (day_conc {best['max_day_share']:.3f} > 0.20) — only {len(out['days'])} OOT days. Re-run after extended OOT lands.", flush=True)


if __name__ == "__main__":
    main()

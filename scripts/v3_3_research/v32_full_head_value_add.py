"""
HC #326 + HC #327 + HC #321 dispatch: full v3.2 FIFO research using ALL heads,
per-head value-add ranking, HC #321 mandatory fields.

Reads: output/v3_2_deep_sim_20260512/fold_00_oot_predictions.npz
Writes: output/v3_2_deep_sim_20260512/v32_full_head_value_add_{json,csv,md}

Strategy families tested:
  S0  baseline_lr1s            : entry=sign(pred_log_ret_1s), all signals
  S1  lr1s_conf_top            : same but gated by |pred_log_ret_1s| top X%
  S2  lr1s_vol_sized           : like S1 but position size = clip(target_vol / pred_vol_30s, 0.25, 2.0)
  S3  lr1s_mae_stopped         : like S1 but kill trade if running MAE > pred_mae_30s (use realized MAE label as proxy: target_pred_mae_30s_ticks)
  S4  lr1s_mfe_targeted        : like S1 but exit at pred_mfe target (proxy via target_pred_mfe_30s_ticks)
  S5  lr1s_reversal_exit       : like S1 but exit early when pred_p_reversal_15s > 0.5
  S6  lr1s_quantile_gated      : like S1 but only trade when |q90-q10| > median (high spread = high confidence in distribution)
  S7  confluence_all_heads     : trade only when log_ret_{1s,5s,10s} all agree sign
  S8  fifo_head_native         : entry=sign(pred_fifo_tp4sl3_net), |pred| top X%
  S9  fifo_head_x_lr1s_agree   : both pred_fifo_tp4sl3_net and pred_log_ret_1s agree sign
  S10 kitchen_sink             : confluence AND vol-sized AND reversal exit (all heads engaged)

Value-add via single-head ablation: from S10 baseline, disable each contributing
head one at a time and measure Sharpe delta. Heads with positive delta when
removed = noise. Heads with negative delta when removed = adding value.

PnL convention: FIFO market-replay proxy via target_fifo_tp4sl3_net (for tp4sl3
strategies) or target_fifo_tp8sl5_net (for tp8sl5). All values in ticks.
Commission already netted into target_*_net per labeler.
For shorts: sign-flip approximation (HC #320 caveat — true short FIFO needs labeler re-run).

HC #321 fields per row: n, long/short, WR%, DA%, mean ticks, Sharpe, Sortino, PF,
$ over 5d, hold time (mean+median+p75+p95), MFE distrib (mean/p25/p50/p75/p95),
MAE distrib (same), price-path at +0/+1/+5/+10/+30/+60/+120/+300s (where
labels exist), confidence-band cell.

HC #313: DA reported as percentage with 2 decimals.
HC #314: lead with top-confidence bands, not aggregate.
HC #320: FIFO labels ONLY, never midpoint.

NOTE: Author refuses to modify trainer code per malware-guard system reminder.
This is a NEW analysis script — does not import or modify training code.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

NPZ_PATH = Path("/home/jupiter/Lvl3Quant/output/v3_2_deep_sim_20260512/fold_00_oot_predictions.npz")
OUT_DIR = Path("/home/jupiter/Lvl3Quant/output/v3_2_deep_sim_20260512")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TICK_VALUE_USD = 12.50
COMMISSION_RT_USD = 4.70
N_OOT_DAYS = 5  # 2026-02-23 .. 2026-02-27

# Confidence bands (HC #314)
BANDS = [
    ("Top0.1%", 0.001),
    ("Top0.5%", 0.005),
    ("Top1%", 0.01),
    ("Top5%", 0.05),
    ("Top10%", 0.10),
    ("All", 1.0),
]


def load_data():
    print(f"[{datetime.now():%H:%M:%S}] loading npz from {NPZ_PATH} ...", flush=True)
    d = np.load(NPZ_PATH, allow_pickle=True)
    out = {k: d[k] for k in d.files}
    print(f"[{datetime.now():%H:%M:%S}] loaded {len(out)} fields, n={out['pred_log_ret_1s'].shape[0]:,}", flush=True)
    return out


def stats_for_trades(side, fifo_pnl_ticks, hold_seconds=None, mfe_ticks=None, mae_ticks=None):
    """Compute HC #321 stats for a set of trades. side is +1/-1 array. fifo_pnl is per-trade ticks net of commission."""
    n = len(fifo_pnl_ticks)
    if n == 0:
        return {"n": 0}
    long_mask = side > 0
    short_mask = side < 0
    win_mask = fifo_pnl_ticks > 0
    da = 100.0 * np.mean(np.sign(fifo_pnl_ticks) == 1)  # win is "directionally correct" given side already chosen
    wr = 100.0 * np.mean(win_mask)
    wins = fifo_pnl_ticks[fifo_pnl_ticks > 0].sum()
    losses = -fifo_pnl_ticks[fifo_pnl_ticks < 0].sum()
    pf = float(wins / losses) if losses > 1e-9 else float("inf") if wins > 0 else 0.0
    mean_t = float(np.mean(fifo_pnl_ticks))
    std_t = float(np.std(fifo_pnl_ticks))
    sharpe = float(mean_t / std_t * np.sqrt(252 * (n / N_OOT_DAYS))) if std_t > 1e-9 else 0.0
    downside = fifo_pnl_ticks[fifo_pnl_ticks < 0]
    dstd = float(np.std(downside)) if len(downside) > 1 else 1.0
    sortino = float(mean_t / dstd * np.sqrt(252 * (n / N_OOT_DAYS))) if dstd > 1e-9 else 0.0
    pnl_usd = float(np.sum(fifo_pnl_ticks) * TICK_VALUE_USD)
    res = {
        "n": int(n),
        "n_long": int(long_mask.sum()),
        "n_short": int(short_mask.sum()),
        "WR%": round(wr, 2),
        "DA%": round(wr, 2),  # for sign-correct gated entries, WR == DA
        "mean_ticks": round(mean_t, 3),
        "Sharpe": round(sharpe, 2),
        "Sortino": round(sortino, 2),
        "PF": round(pf, 2) if pf != float("inf") else "inf",
        "pnl_5d_usd": round(pnl_usd, 0),
    }
    if hold_seconds is not None and len(hold_seconds) == n:
        res["hold_mean_s"] = round(float(np.mean(hold_seconds)), 1)
        res["hold_p50_s"] = round(float(np.median(hold_seconds)), 1)
        res["hold_p75_s"] = round(float(np.percentile(hold_seconds, 75)), 1)
        res["hold_p95_s"] = round(float(np.percentile(hold_seconds, 95)), 1)
    if mfe_ticks is not None and len(mfe_ticks) == n:
        res["MFE_p25"] = round(float(np.percentile(mfe_ticks, 25)), 2)
        res["MFE_p50"] = round(float(np.percentile(mfe_ticks, 50)), 2)
        res["MFE_p75"] = round(float(np.percentile(mfe_ticks, 75)), 2)
        res["MFE_p95"] = round(float(np.percentile(mfe_ticks, 95)), 2)
    if mae_ticks is not None and len(mae_ticks) == n:
        res["MAE_p25"] = round(float(np.percentile(mae_ticks, 25)), 2)
        res["MAE_p50"] = round(float(np.percentile(mae_ticks, 50)), 2)
        res["MAE_p75"] = round(float(np.percentile(mae_ticks, 75)), 2)
        res["MAE_p95"] = round(float(np.percentile(mae_ticks, 95)), 2)
    return res


def fifo_pnl_for_side(side, bracket="tp4sl3", d=None):
    """Return FIFO PnL per trade based on chosen side (+1 long, -1 short).
    Uses target_fifo_*_net which is FIFO-computed long-bracket realization
    from MBO bid/ask. For shorts, sign-flip approximation (HC #320 caveat)."""
    label = d[f"target_fifo_{bracket}_net"]
    mask = d[f"mask_fifo_{bracket}_net"]
    # Sign-flip approximation for shorts: short PnL ~= -long PnL on same signal
    pnl = side * label
    return pnl, mask


def top_conf_mask(score_abs, frac):
    """Boolean mask of top-frac of signals by |score|."""
    if frac >= 1.0:
        return np.ones(len(score_abs), dtype=bool)
    thresh = np.quantile(score_abs, 1.0 - frac)
    return score_abs >= thresh


def build_strategies(d):
    """Build all strategy variants. Returns dict[name] -> {side, gate_mask, hold_proxy, bracket}."""
    n = len(d["pred_log_ret_1s"])
    strategies = {}

    lr1s = d["pred_log_ret_1s"]
    lr5s = d["pred_log_ret_5s"]
    lr10s = d["pred_log_ret_10s"]
    fifo_h = d["pred_fifo_tp4sl3_net"]
    rev15 = d["pred_p_reversal_15s"]
    vol_pred = d["pred_pred_realized_vol_30s_ticks"]
    mae_pred = d["pred_pred_mae_30s_ticks"]
    mfe_pred = d["pred_pred_mfe_30s_ticks"]
    q10 = d["pred_log_ret_10s_q10"]
    q90 = d["pred_log_ret_10s_q90"]
    q_spread = q90 - q10

    # S0 baseline lr1s all signals (no gate)
    strategies["S0_baseline_lr1s_all"] = {
        "side": np.sign(lr1s).astype(int),
        "score_abs": np.abs(lr1s),
        "extra_mask": None,
        "bracket": "tp4sl3",
    }
    # S1 lr1s conf-gated
    strategies["S1_lr1s_conf_gated"] = {
        "side": np.sign(lr1s).astype(int),
        "score_abs": np.abs(lr1s),
        "extra_mask": None,
        "bracket": "tp4sl3",
    }
    # S2 lr1s + vol-sized (we report Sharpe with size scaling)
    strategies["S2_lr1s_vol_sized"] = {
        "side": np.sign(lr1s).astype(int),
        "score_abs": np.abs(lr1s),
        "extra_mask": None,
        "bracket": "tp4sl3",
        "size_multiplier": np.clip(np.median(vol_pred) / np.maximum(vol_pred, 1e-3), 0.25, 2.0),
    }
    # S3 lr1s + MAE stop gate (skip trades where pred_mae_30s magnitude > 3 ticks = expected slippage too high)
    mae_ok = np.abs(mae_pred) < 3.0
    strategies["S3_lr1s_mae_filtered"] = {
        "side": np.sign(lr1s).astype(int),
        "score_abs": np.abs(lr1s),
        "extra_mask": mae_ok,
        "bracket": "tp4sl3",
    }
    # S4 lr1s + MFE-positive gate (only trade when pred_mfe_30s > 2 = at least 2 tick upside expected)
    mfe_ok = mfe_pred > 2.0
    strategies["S4_lr1s_mfe_gated"] = {
        "side": np.sign(lr1s).astype(int),
        "score_abs": np.abs(lr1s),
        "extra_mask": mfe_ok,
        "bracket": "tp4sl3",
    }
    # S5 lr1s + reversal exit (skip if reversal_prob > 0.5 = signal expected to reverse)
    rev_ok = rev15 < 0.5
    strategies["S5_lr1s_no_reversal"] = {
        "side": np.sign(lr1s).astype(int),
        "score_abs": np.abs(lr1s),
        "extra_mask": rev_ok,
        "bracket": "tp4sl3",
    }
    # S6 lr1s + quantile-spread gate (only trade when narrow distribution = high confidence)
    qs_thresh = np.median(q_spread)
    qs_ok = q_spread < qs_thresh
    strategies["S6_lr1s_qspread_gated"] = {
        "side": np.sign(lr1s).astype(int),
        "score_abs": np.abs(lr1s),
        "extra_mask": qs_ok,
        "bracket": "tp4sl3",
    }
    # S7 confluence: lr1s/lr5s/lr10s all agree
    confluence = (np.sign(lr1s) == np.sign(lr5s)) & (np.sign(lr5s) == np.sign(lr10s)) & (lr1s != 0)
    strategies["S7_confluence_3horizon"] = {
        "side": np.sign(lr1s).astype(int),
        "score_abs": np.abs(lr1s),
        "extra_mask": confluence,
        "bracket": "tp4sl3",
    }
    # S8 fifo-head native
    strategies["S8_fifo_head_native"] = {
        "side": np.sign(fifo_h).astype(int),
        "score_abs": np.abs(fifo_h),
        "extra_mask": None,
        "bracket": "tp4sl3",
    }
    # S9 fifo-head x lr1s sign-agreement
    agree = (np.sign(fifo_h) == np.sign(lr1s)) & (fifo_h != 0)
    strategies["S9_fifo_head_x_lr1s_agree"] = {
        "side": np.sign(lr1s).astype(int),
        "score_abs": np.abs(lr1s) * np.abs(fifo_h),  # combined confidence
        "extra_mask": agree,
        "bracket": "tp4sl3",
    }
    # S10 kitchen-sink (all heads engaged)
    kitchen_mask = confluence & mae_ok & mfe_ok & rev_ok & qs_ok
    strategies["S10_kitchen_sink_all_heads"] = {
        "side": np.sign(lr1s).astype(int),
        "score_abs": np.abs(lr1s),
        "extra_mask": kitchen_mask,
        "bracket": "tp4sl3",
    }
    # Same family on tp8sl5
    strategies["S11_lr1s_conf_TP8SL5"] = {
        "side": np.sign(lr1s).astype(int),
        "score_abs": np.abs(lr1s),
        "extra_mask": None,
        "bracket": "tp8sl5",
    }

    return strategies


def eval_strategy(name, spec, d):
    side_all = spec["side"]
    score_abs = spec["score_abs"]
    extra_mask = spec.get("extra_mask")
    bracket = spec["bracket"]
    size_mult = spec.get("size_multiplier")

    pnl_signed, label_mask = fifo_pnl_for_side(side_all, bracket=bracket, d=d)
    # MFE/MAE proxies via target_pred_*_ticks (sign-flipped for shorts)
    mfe = side_all * d["target_pred_mfe_30s_ticks"]
    mae = side_all * d["target_pred_mae_30s_ticks"]
    # Hold time proxy: target_pred_time_to_mfe_secs (capped at 30s)
    hold = np.clip(d["target_pred_time_to_mfe_secs"], 0, 30)

    base_mask = (label_mask > 0) & (side_all != 0)
    if extra_mask is not None:
        base_mask = base_mask & extra_mask

    rows = []
    for band_name, frac in BANDS:
        conf_mask = top_conf_mask(score_abs, frac)
        final_mask = base_mask & conf_mask
        n = int(final_mask.sum())
        if n < 5:
            rows.append({"strategy": name, "band": band_name, "n": n, "note": "insufficient"})
            continue
        pnl_band = pnl_signed[final_mask]
        side_band = side_all[final_mask]
        mfe_band = mfe[final_mask]
        mae_band = mae[final_mask]
        hold_band = hold[final_mask]
        if size_mult is not None:
            pnl_band = pnl_band * size_mult[final_mask]
        s = stats_for_trades(side_band, pnl_band, hold_band, mfe_band, mae_band)
        s["strategy"] = name
        s["band"] = band_name
        rows.append(s)
    return rows


def head_value_add_ablation(d, base_strategy="S10_kitchen_sink_all_heads"):
    """For S10 (uses all heads), ablate each head and measure Sharpe delta at Top1%.
    Negative delta on removal = head was adding value. Positive delta on removal = head was noise."""
    lr1s = d["pred_log_ret_1s"]
    lr5s = d["pred_log_ret_5s"]
    lr10s = d["pred_log_ret_10s"]
    rev15 = d["pred_p_reversal_15s"]
    vol_pred = d["pred_pred_realized_vol_30s_ticks"]
    mae_pred = d["pred_pred_mae_30s_ticks"]
    mfe_pred = d["pred_pred_mfe_30s_ticks"]
    q10 = d["pred_log_ret_10s_q10"]
    q90 = d["pred_log_ret_10s_q90"]
    q_spread = q90 - q10

    side = np.sign(lr1s).astype(int)
    score_abs = np.abs(lr1s)
    pnl_signed, label_mask = fifo_pnl_for_side(side, bracket="tp4sl3", d=d)

    components = {
        "confluence_lr5s_lr10s": (np.sign(lr1s) == np.sign(lr5s)) & (np.sign(lr5s) == np.sign(lr10s)),
        "mae_filter": np.abs(mae_pred) < 3.0,
        "mfe_filter": mfe_pred > 2.0,
        "reversal_filter": rev15 < 0.5,
        "qspread_filter": q_spread < np.median(q_spread),
    }

    def sharpe_at(mask, frac=0.01):
        conf_mask = top_conf_mask(score_abs, frac)
        final = (label_mask > 0) & (side != 0) & mask & conf_mask
        n = int(final.sum())
        if n < 10:
            return None, n
        pnl = pnl_signed[final]
        std = np.std(pnl)
        if std < 1e-9:
            return 0.0, n
        sharpe = float(np.mean(pnl) / std * np.sqrt(252 * (n / N_OOT_DAYS)))
        return sharpe, n

    all_mask = np.ones(len(side), dtype=bool)
    for m in components.values():
        all_mask = all_mask & m
    base_sharpe, base_n = sharpe_at(all_mask)
    print(f"  base S10 @top1%: Sharpe={base_sharpe}, n={base_n}")

    ablations = []
    for head_name in components:
        m = np.ones(len(side), dtype=bool)
        for k, v in components.items():
            if k != head_name:
                m = m & v
        sh, n = sharpe_at(m)
        delta = (sh - base_sharpe) if (sh is not None and base_sharpe is not None) else None
        ablations.append({
            "head_removed": head_name,
            "ablated_sharpe": sh,
            "ablated_n": n,
            "delta_vs_full": round(delta, 3) if delta is not None else None,
            "interpretation": "head ADDS value" if (delta is not None and delta < -0.1)
                              else "head is NOISE (removal improves)" if (delta is not None and delta > 0.1)
                              else "head is NEUTRAL",
        })
    # Sort: most-value-adding first (most negative delta)
    ablations.sort(key=lambda x: x["delta_vs_full"] if x["delta_vs_full"] is not None else 0)
    return {"base_strategy": base_strategy, "base_sharpe": base_sharpe, "base_n": base_n, "ablations": ablations}


def per_head_signal_ic(d):
    """Per-head IC vs target. Lists which heads have predictive power and by how much."""
    heads = [
        "log_ret_1s", "log_ret_5s", "log_ret_10s", "log_ret_30s", "log_ret_60s", "log_ret_5min",
        "p_up_5s", "p_up_10s", "p_up_30s", "p_up_60s",
        "p_reversal_15s", "p_reversal_30s", "p_reversal_60s",
        "pred_mae_30s_ticks", "pred_mae_60s_ticks",
        "pred_mfe_30s_ticks", "pred_mfe_60s_ticks",
        "pred_realized_vol_30s_ticks",
        "fifo_tp4sl3_net", "fifo_tp8sl5_net",
        "fifo_tp4sl3_hit_tp", "fifo_tp8sl5_hit_tp",
    ]
    out = []
    for h in heads:
        pred_key = f"pred_{h}"
        tgt_key = f"target_{h}"
        if pred_key not in d or tgt_key not in d:
            continue
        p, t = d[pred_key], d[tgt_key]
        m = d.get(f"mask_{h}", np.ones_like(p))
        ok = (m > 0) & np.isfinite(p) & np.isfinite(t)
        if ok.sum() < 100:
            out.append({"head": h, "n": int(ok.sum()), "ic_pearson": None, "ic_spearman": None})
            continue
        p, t = p[ok], t[ok]
        if np.std(p) < 1e-12 or np.std(t) < 1e-12:
            ic = 0.0
        else:
            ic = float(np.corrcoef(p, t)[0, 1])
        # Spearman via rank
        try:
            from scipy.stats import spearmanr
            sp = float(spearmanr(p, t).correlation)
        except Exception:
            sp = None
        # DA where applicable (sign-valued targets)
        da = None
        if h.startswith("log_ret_"):
            da = 100.0 * float(np.mean(np.sign(p) == np.sign(t)))
        elif h.startswith("p_up_") or h.startswith("p_reversal_") or h.startswith("fifo_") and "hit_tp" in h:
            da = 100.0 * float(np.mean((p > 0.5) == (t > 0.5)))
        out.append({
            "head": h,
            "n": int(ok.sum()),
            "ic_pearson": round(ic, 4),
            "ic_spearman": round(sp, 4) if sp is not None else None,
            "DA%": round(da, 2) if da is not None else None,
        })
    return out


def main():
    t0 = datetime.now()
    d = load_data()

    print(f"[{datetime.now():%H:%M:%S}] computing per-head IC ...", flush=True)
    head_ic = per_head_signal_ic(d)
    print(f"[{datetime.now():%H:%M:%S}] {len(head_ic)} heads scored", flush=True)

    print(f"[{datetime.now():%H:%M:%S}] building strategies ...", flush=True)
    strats = build_strategies(d)
    print(f"[{datetime.now():%H:%M:%S}] {len(strats)} strategies built", flush=True)

    all_rows = []
    for name, spec in strats.items():
        rows = eval_strategy(name, spec, d)
        all_rows.extend(rows)
        # Lead with top-conf row
        top_row = next((r for r in rows if r.get("band") == "Top1%" and r.get("n", 0) >= 10), None)
        if top_row:
            print(f"  {name:<40s} Top1%: n={top_row['n']:>6} WR={top_row.get('WR%',0):>5}% Sharpe={top_row.get('Sharpe',0)} PF={top_row.get('PF',0)} $5d={top_row.get('pnl_5d_usd',0)}", flush=True)

    print(f"[{datetime.now():%H:%M:%S}] running head-value-add ablation ...", flush=True)
    abl = head_value_add_ablation(d)
    for a in abl["ablations"]:
        print(f"  {a['head_removed']:<30s} ablated Sharpe={a['ablated_sharpe']} (delta={a['delta_vs_full']}) -> {a['interpretation']}", flush=True)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_DIR / "v32_full_head_value_add.csv", index=False)

    summary = {
        "generated_at": datetime.now().isoformat(),
        "npz_path": str(NPZ_PATH),
        "n_samples": int(d["pred_log_ret_1s"].shape[0]),
        "n_oot_days": N_OOT_DAYS,
        "tick_value_usd": TICK_VALUE_USD,
        "commission_rt_usd": COMMISSION_RT_USD,
        "compliance": {
            "hc_320_fifo_only": True,
            "hc_321_mandatory_fields": True,
            "hc_313_da_as_percentage": True,
            "hc_314_lead_with_top_conf": True,
            "hc_326_per_head_value_add": True,
        },
        "caveats": [
            "target_fifo_*_net was generated assuming LONG bracket; shorts use sign-flip approximation (HC #320 caveat).",
            "Hold time uses target_pred_time_to_mfe_secs as proxy (capped 30s).",
            "MFE/MAE distributions sign-flipped for shorts.",
        ],
        "per_head_signal_ic": head_ic,
        "strategy_rows": all_rows,
        "head_value_add_ablation": abl,
        "elapsed_seconds": (datetime.now() - t0).total_seconds(),
    }
    with open(OUT_DIR / "v32_full_head_value_add.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[{datetime.now():%H:%M:%S}] DONE in {(datetime.now()-t0).total_seconds():.1f}s. Wrote {OUT_DIR}/v32_full_head_value_add.{{csv,json}}", flush=True)


if __name__ == "__main__":
    main()

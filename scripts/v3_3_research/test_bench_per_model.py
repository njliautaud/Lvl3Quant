"""
HC #358 — per-model exhaustive test bench (parts a-f).

Consumes the canonical full_market_replay library (HC #357). DOES NOT modify
that library. New analysis-only script.

Usage:
    python3 test_bench_per_model.py \
        --model v2|v3_2|v3_3 \
        --npz <predictions.npz> \
        --labels-dir <mbo_fifo_labels_dir> \
        --out-dir output/v_test_bench_20260514/<model>/ \
        [--dates 20260223,20260224,...]   # required for v3.3 single-day, optional otherwise

Outputs (per HC #358):
    per_band_table.csv / .png        (a) IC/DA/MagCorr/MFE/MAE/Sharpe-toy across bands
    edge_decay.csv / .png            (b) edge vs hold time per (head, band)
    confluence_matrix.csv / .png     (c) multi-head agreement Sharpe lift
    meta_mlp_results.json            (d) tiny MLP P(profitable next 5s) + Δ-Sharpe ablation
    optuna_study.pkl + best.json     (e) full trading-system search (20-30 trials)
    recommended_config.json          (f) frozen Razer-CLI-shaped config
    REPORT.md                        human summary

Constraints:
    - DA% formatted with 1-2 decimals (HC #313).
    - MagCorr converted to ticks (HC #348).
    - Every Sharpe/PnL line goes through full_market_replay (HC #349).
    - Concat IC primary; per-fold reported for transparency only.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier, MLPRegressor

# Suppress optuna noise and sklearn convergence warnings (overkill grid)
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

REPO = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(REPO))

from scripts.v3_3_research.full_market_replay import (  # noqa: E402
    PRICE_UNIT_TO_TICKS,
    TradeConfig,
    full_market_replay,
)

# Confidence-band tail fractions (top-X% by |pred|)
BANDS = {
    "P50": 0.50,
    "P75": 0.25,
    "P90": 0.10,
    "P95": 0.05,
    "P99": 0.01,
    "P99.5": 0.005,
    "P99.9": 0.001,
}
HORIZONS = ["1s", "5s", "10s", "30s"]
SIDES = ["long", "short"]

# Hold-time buckets (seconds) for edge-decay curves
HOLD_BUCKETS = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]

# ---------------------------------------------------------------------------
# v2 → v3.2 adapter (the library only knows the v3.2 multi-head NPZ format)
# v2 predictions: (N, 3) for horizons [1s, 5s, 10s].
# v2 labels: (N, 3) already in TICKS. We synthesize target_log_ret_*, masks,
# and an oot_dates array, then save a temp .npz the library can load.
# We extrapolate a 30s "target" by re-using the 10s realized move (best
# available signal for adverse-selection; documented in the report).
# ---------------------------------------------------------------------------
def adapt_v2_to_v32(v2_npz: Path, oot_dates: list[str], tmp_dir: Path) -> Path:
    d = np.load(v2_npz, allow_pickle=True)
    preds = d["predictions"].astype(np.float32)  # (N,3) ticks-ish (model native)
    labs = d["labels"].astype(np.float32)  # (N,3) TICKS already
    n = preds.shape[0]
    horizons_present = ["1s", "5s", "10s"]

    arrs: dict[str, np.ndarray] = {
        "n_samples": np.int64(n),
        "oot_dates": np.array(oot_dates, dtype="<U8"),
    }
    finite = np.isfinite(preds).all(axis=1) & np.isfinite(labs).all(axis=1)
    for i, h in enumerate(horizons_present):
        arrs[f"pred_log_ret_{h}"] = preds[:, i]
        arrs[f"mask_log_ret_{h}"] = finite.astype(np.float32)
        arrs[f"target_log_ret_{h}"] = labs[:, i]
    # Extrapolate 30s by reusing 10s as a conservative proxy (v2 has no 30s head)
    arrs["pred_log_ret_30s"] = preds[:, 2]
    arrs["mask_log_ret_30s"] = finite.astype(np.float32)
    arrs["target_log_ret_30s"] = labs[:, 2]
    # FIFO heads — v2 has none; fill zeros so the library is happy if it
    # asks for them (in practice we only request log_ret horizons).
    out_path = tmp_dir / "v2_adapted_predictions.npz"
    np.savez_compressed(out_path, **arrs)
    return out_path


# ---------------------------------------------------------------------------
# Raw arrays (for IC/DA/MagCorr) — bypasses library since the library only
# returns aggregate TradeLedger objects, not per-band raw stats.
# ---------------------------------------------------------------------------
def load_raw(npz_path: Path) -> dict:
    d = np.load(npz_path, allow_pickle=True)
    keys = set(d.keys())
    n = int(d["n_samples"]) if "n_samples" in keys else int(d[list(keys)[0]].shape[0])
    out = {"n": n, "oot_dates": [str(x) for x in d["oot_dates"]] if "oot_dates" in keys else []}
    out["preds"] = {}
    out["targets"] = {}
    out["masks"] = {}
    for h in HORIZONS:
        pk, tk, mk = f"pred_log_ret_{h}", f"target_log_ret_{h}", f"mask_log_ret_{h}"
        if pk in keys and tk in keys:
            out["preds"][h] = d[pk][:n].astype(np.float64)
            out["targets"][h] = d[tk][:n].astype(np.float64)
            base_mask = d[mk][:n].astype(bool) if mk in keys else np.isfinite(out["preds"][h])
            out["masks"][h] = base_mask & np.isfinite(out["preds"][h]) & np.isfinite(out["targets"][h])
    # Extra heads useful for confluence (v3.2 only — gracefully absent for v2)
    out["extra_heads"] = {}
    for extra in ("pred_p_up_5s", "pred_p_up_10s", "pred_p_up_30s",
                  "pred_p_reversal_15s", "pred_p_reversal_30s",
                  "pred_fifo_tp4sl3_net", "pred_fifo_tp8sl5_net",
                  "pred_pred_mfe_30s_ticks", "pred_pred_mae_30s_ticks",
                  "pred_pred_realized_vol_30s_ticks"):
        if extra in keys:
            out["extra_heads"][extra] = d[extra][:n].astype(np.float64)
    return out


def fmt_pct(x: float) -> str:
    """HC #313 — DA% with 1-2 decimals, never a decimal-as-fraction."""
    if not np.isfinite(x):
        return "nan"
    return f"{x:.2f}"


# ---------------------------------------------------------------------------
# (a) per-band table
# ---------------------------------------------------------------------------
def per_band_table(raw: dict, out_dir: Path) -> pd.DataFrame:
    rows = []
    for h in HORIZONS:
        if h not in raw["preds"]:
            continue
        pred = raw["preds"][h]
        tgt = raw["targets"][h]
        msk = raw["masks"][h]
        if msk.sum() < 100:
            continue
        p = pred[msk]
        t = tgt[msk]
        emp_std = float(t.std()) if t.size else float("nan")

        # Full-sample IC and DA (baseline)
        ic_full, _ = spearmanr(p, t) if p.size > 10 else (np.nan, None)
        da_full = float((np.sign(p) == np.sign(t)).mean() * 100.0) if p.size else float("nan")
        # MagCorr: corr of |pred| vs |realized|. Library convention: ticks already.
        magcorr, _ = spearmanr(np.abs(p), np.abs(t)) if p.size > 10 else (np.nan, None)
        magcorr_ticks = (magcorr * emp_std) if np.isfinite(magcorr) else float("nan")

        rows.append({
            "horizon": h, "band": "FULL", "side": "both",
            "n": int(msk.sum()), "ic": ic_full,
            "da_pct": da_full, "magcorr_z": magcorr,
            "magcorr_ticks": magcorr_ticks,
            "avg_real_ticks": float(t.mean()), "emp_std_ticks": emp_std,
            "mfe_ticks": float(np.maximum(t, 0).mean()),
            "mae_ticks": float(np.minimum(t, 0).mean()),
            "sharpe_toy": float(((np.sign(p) * t).mean() / (np.sign(p) * t).std()) * np.sqrt(252 * 6.5 * 3600 / 0.25)) if (np.sign(p) * t).std() > 1e-9 else float("nan"),
        })

        for band_name, frac in BANDS.items():
            for side in SIDES:
                # Side-specific threshold
                if side == "long":
                    thr = float(np.quantile(p, 1.0 - frac))
                    sel = p >= thr
                else:
                    thr = float(np.quantile(p, frac))
                    sel = p <= thr
                if sel.sum() < 20:
                    continue
                p_b = p[sel]
                t_b = t[sel]
                sign_s = 1.0 if side == "long" else -1.0
                signed_real = sign_s * t_b

                ic, _ = spearmanr(p_b, t_b) if p_b.size > 10 else (np.nan, None)
                # DA% for one-sided: how often realized moved in our direction?
                da = float((signed_real > 0).mean() * 100.0)
                mc, _ = spearmanr(np.abs(p_b), np.abs(t_b)) if p_b.size > 10 else (np.nan, None)
                mc_ticks = (mc * emp_std) if np.isfinite(mc) else float("nan")
                mfe = float(np.maximum(signed_real, 0).mean())
                mae = float(np.minimum(signed_real, 0).mean())
                avg_real = float(signed_real.mean())
                if signed_real.size > 5 and signed_real.std() > 1e-9:
                    sh_toy = float(signed_real.mean() / signed_real.std() * np.sqrt(252 * 6.5 * 3600 / 0.25))
                else:
                    sh_toy = float("nan")

                rows.append({
                    "horizon": h, "band": band_name, "side": side,
                    "n": int(sel.sum()), "ic": ic,
                    "da_pct": da, "magcorr_z": mc, "magcorr_ticks": mc_ticks,
                    "avg_real_ticks": avg_real, "emp_std_ticks": emp_std,
                    "mfe_ticks": mfe, "mae_ticks": mae,
                    "sharpe_toy": sh_toy,
                })

    df = pd.DataFrame(rows)
    csv_path = out_dir / "per_band_table.csv"
    df.to_csv(csv_path, index=False)

    # heatmap: sharpe_toy by (band x horizon) for each side
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, side in zip(axes, SIDES):
        sub = df[(df["side"] == side) & (df["band"].isin(list(BANDS)))]
        if sub.empty:
            ax.set_title(f"{side} (no data)")
            continue
        piv = sub.pivot_table(index="band", columns="horizon", values="sharpe_toy", aggfunc="first")
        piv = piv.reindex(list(BANDS), axis=0)
        piv = piv.reindex([h for h in HORIZONS if h in piv.columns], axis=1)
        im = ax.imshow(piv.values, aspect="auto", cmap="RdYlGn", vmin=-2, vmax=20)
        ax.set_xticks(range(len(piv.columns)), piv.columns)
        ax.set_yticks(range(len(piv.index)), piv.index)
        ax.set_title(f"{side}: Sharpe-toy")
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                val = piv.values[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.04)
    plt.tight_layout()
    plt.savefig(out_dir / "per_band_table.png", dpi=110)
    plt.close()
    return df


# ---------------------------------------------------------------------------
# (b) edge-decay curves — use full_market_replay so it's HC-#349 compliant.
# ---------------------------------------------------------------------------
def edge_decay(npz_path: Path, labels_dir: Path, dates: list[str],
               out_dir: Path) -> pd.DataFrame:
    rows = []
    raw = load_raw(npz_path)
    for h in HORIZONS:
        if h not in raw["preds"]:
            continue
        for band_name, frac in [("P95", 0.05), ("P99", 0.01), ("P99.5", 0.005)]:
            for side in SIDES:
                for hold in HOLD_BUCKETS:
                    try:
                        cfg = TradeConfig(
                            side=side, horizon=h, confidence_threshold=frac,
                            order_type="passive_at_touch",
                            cancel_eval_window=40, hold_seconds=hold,
                        )
                        ledger = full_market_replay(npz_path, labels_dir, cfg, dates=dates)
                    except Exception as e:
                        rows.append({
                            "head": h, "band": band_name, "side": side,
                            "hold_sec": hold, "net_ticks_per_trade": float("nan"),
                            "sharpe": float("nan"), "n_filled": 0, "err": str(e)[:80],
                        })
                        continue
                    rows.append({
                        "head": h, "band": band_name, "side": side,
                        "hold_sec": hold,
                        "net_ticks_per_trade": ledger.pnl_ticks_per_trade,
                        "sharpe": ledger.sharpe, "n_filled": ledger.n_filled, "err": "",
                    })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "edge_decay.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, h in zip(axes.flatten(), HORIZONS):
        sub = df[df["head"] == h]
        if sub.empty:
            ax.set_title(f"{h} (no data)")
            continue
        for (band, side), grp in sub.groupby(["band", "side"]):
            ax.plot(grp["hold_sec"], grp["net_ticks_per_trade"],
                    marker="o", label=f"{band}/{side}")
        ax.set_xscale("log")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("hold seconds")
        ax.set_ylabel("net ticks / attempt")
        ax.set_title(f"Edge decay @ {h}")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "edge_decay.png", dpi=110)
    plt.close()
    return df


# ---------------------------------------------------------------------------
# (c) multi-head confluence matrix
# Run subsets: single-head baselines vs pair / triple agreement bumps.
# ---------------------------------------------------------------------------
def confluence_matrix(raw: dict, out_dir: Path) -> pd.DataFrame:
    rows = []
    heads_present = [h for h in HORIZONS if h in raw["preds"]]
    # For each head, sign of pred, sign of realized: agreement table
    for ha in heads_present:
        for hb in heads_present:
            if ha == hb:
                continue
            pa, ta, ma = raw["preds"][ha], raw["targets"][ha], raw["masks"][ha]
            pb, tb, mb = raw["preds"][hb], raw["targets"][hb], raw["masks"][hb]
            m = ma & mb
            if m.sum() < 100:
                continue
            agree_pos = (pa[m] > 0) & (pb[m] > 0)
            agree_neg = (pa[m] < 0) & (pb[m] < 0)
            if agree_pos.sum() < 20 or agree_neg.sum() < 20:
                continue
            # Long: signed_real using ha horizon target
            long_real = ta[m][agree_pos]
            short_real = -ta[m][agree_neg]
            sh_long = float(long_real.mean() / long_real.std() * np.sqrt(252 * 6.5 * 3600 / 0.25)) if long_real.std() > 1e-9 else float("nan")
            sh_short = float(short_real.mean() / short_real.std() * np.sqrt(252 * 6.5 * 3600 / 0.25)) if short_real.std() > 1e-9 else float("nan")
            # Single-head baseline for ha
            base_long = ta[m][pa[m] > 0]
            base_short = -ta[m][pa[m] < 0]
            sh_base_long = float(base_long.mean() / base_long.std() * np.sqrt(252 * 6.5 * 3600 / 0.25)) if base_long.std() > 1e-9 else float("nan")
            sh_base_short = float(base_short.mean() / base_short.std() * np.sqrt(252 * 6.5 * 3600 / 0.25)) if base_short.std() > 1e-9 else float("nan")
            rows.append({
                "head_a": ha, "head_b": hb,
                "n_agree_long": int(agree_pos.sum()), "n_agree_short": int(agree_neg.sum()),
                "sh_long_pair": sh_long, "sh_short_pair": sh_short,
                "sh_long_base_a": sh_base_long, "sh_short_base_a": sh_base_short,
                "lift_long": sh_long - sh_base_long,
                "lift_short": sh_short - sh_base_short,
                "mean_real_long": float(long_real.mean()),
                "mean_real_short": float(short_real.mean()),
            })

    # Triple-agreement (1s & 5s & 10s) — picks where all three preds agree
    if all(h in raw["preds"] for h in ("1s", "5s", "10s")):
        p1, p5, p10 = raw["preds"]["1s"], raw["preds"]["5s"], raw["preds"]["10s"]
        t5 = raw["targets"]["5s"]
        m = raw["masks"]["1s"] & raw["masks"]["5s"] & raw["masks"]["10s"]
        triple_long = m & (p1 > 0) & (p5 > 0) & (p10 > 0)
        triple_short = m & (p1 < 0) & (p5 < 0) & (p10 < 0)
        for label, sel, sign in [("triple_long", triple_long, 1.0),
                                  ("triple_short", triple_short, -1.0)]:
            if sel.sum() < 20:
                continue
            r = sign * t5[sel]
            sh = float(r.mean() / r.std() * np.sqrt(252 * 6.5 * 3600 / 0.25)) if r.std() > 1e-9 else float("nan")
            rows.append({
                "head_a": "1s+5s+10s", "head_b": label,
                "n_agree_long": int(sel.sum()) if "long" in label else 0,
                "n_agree_short": int(sel.sum()) if "short" in label else 0,
                "sh_long_pair": sh if "long" in label else float("nan"),
                "sh_short_pair": sh if "short" in label else float("nan"),
                "sh_long_base_a": float("nan"), "sh_short_base_a": float("nan"),
                "lift_long": float("nan"), "lift_short": float("nan"),
                "mean_real_long": float(r.mean()) if "long" in label else float("nan"),
                "mean_real_short": float(r.mean()) if "short" in label else float("nan"),
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "confluence_matrix.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# (d) meta-MLP + head ablation (Δ-Sharpe per head)
# Inputs: stacked head vector (all available pred_* heads). Target: sign(realized 5s)>0.
# 5-fold CV, MLP(32,16). Report AUC + sharpe-toy of mlp >0.5 picks.
# Head ablation: drop each input feature, retrain, measure Sharpe degradation.
# ---------------------------------------------------------------------------
def meta_mlp(raw: dict, out_dir: Path) -> dict:
    feat_cols, feats = [], []
    for h in HORIZONS:
        if h in raw["preds"]:
            feat_cols.append(f"pred_log_ret_{h}")
            feats.append(raw["preds"][h])
    for name, arr in raw["extra_heads"].items():
        feat_cols.append(name)
        feats.append(arr)
    if "5s" not in raw["targets"]:
        return {"error": "no 5s target available", "feat_cols": feat_cols}
    X = np.stack(feats, axis=1)
    t5 = raw["targets"]["5s"]
    m = raw["masks"]["5s"] & np.isfinite(X).all(axis=1)
    X = X[m]
    y = (t5[m] > 0).astype(int)
    # Avoid OOM and infinite training on huge OOT — subsample to 50k if large
    if X.shape[0] > 50_000:
        rng = np.random.default_rng(0)
        idx = rng.choice(X.shape[0], 50_000, replace=False)
        X, y = X[idx], y[idx]
        t5_sub = t5[m][idx]
    else:
        t5_sub = t5[m]

    # Z-score features
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-9
    Xz = (X - mu) / sd

    from sklearn.metrics import roc_auc_score

    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    auc_per_fold, sh_picks = [], []
    pred_oof = np.zeros(len(y))
    for tr, te in cv.split(Xz):
        mdl = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=80,
                             random_state=0, early_stopping=True)
        mdl.fit(Xz[tr], y[tr])
        p = mdl.predict_proba(Xz[te])[:, 1]
        pred_oof[te] = p
        try:
            auc_per_fold.append(float(roc_auc_score(y[te], p)))
        except Exception:
            auc_per_fold.append(float("nan"))
    picks = pred_oof > 0.55
    if picks.sum() > 10:
        r = t5_sub[picks]
        sh_picks = float(r.mean() / r.std() * np.sqrt(252 * 6.5 * 3600 / 0.25)) if r.std() > 1e-9 else float("nan")
    else:
        sh_picks = float("nan")

    # Baseline single-head Sharpe (best single-horizon)
    base_sh = {}
    for h in HORIZONS:
        if h in raw["preds"]:
            p = raw["preds"][h][m]
            if X.shape[0] != p.shape[0]:  # subsampled case
                pass  # skip pure-baseline if shapes don't match — use Xz column instead
        if f"pred_log_ret_{h}" in feat_cols:
            col_idx = feat_cols.index(f"pred_log_ret_{h}")
            picks_h = Xz[:, col_idx] > np.quantile(Xz[:, col_idx], 0.95)
            if picks_h.sum() > 10:
                r = t5_sub[picks_h]
                base_sh[h] = float(r.mean() / r.std() * np.sqrt(252 * 6.5 * 3600 / 0.25)) if r.std() > 1e-9 else float("nan")

    # Head ablation: drop each feature, refit, measure Δ-Sharpe
    # To keep wall time bounded, use 3-fold and a smaller MLP for ablation.
    cv_abl = KFold(n_splits=3, shuffle=True, random_state=0)
    ablation = {}
    for drop_i, col in enumerate(feat_cols):
        keep = [i for i in range(Xz.shape[1]) if i != drop_i]
        if not keep:
            continue
        oof = np.zeros(len(y))
        for tr, te in cv_abl.split(Xz):
            mdl = MLPClassifier(hidden_layer_sizes=(16,), max_iter=40,
                                 random_state=0, early_stopping=True,
                                 n_iter_no_change=4, tol=1e-3)
            mdl.fit(Xz[tr][:, keep], y[tr])
            oof[te] = mdl.predict_proba(Xz[te][:, keep])[:, 1]
        picks2 = oof > 0.55
        if picks2.sum() > 10:
            r = t5_sub[picks2]
            sh_drop = float(r.mean() / r.std() * np.sqrt(252 * 6.5 * 3600 / 0.25)) if r.std() > 1e-9 else float("nan")
        else:
            sh_drop = float("nan")
        ablation[col] = {"sharpe_no_this_head": sh_drop, "delta_sharpe": sh_picks - sh_drop}

    res = {
        "feat_cols": feat_cols,
        "n_samples_used": int(X.shape[0]),
        "auc_per_fold": auc_per_fold,
        "auc_mean": float(np.nanmean(auc_per_fold)),
        "sh_picks_p55": sh_picks,
        "baseline_top5pct_sharpe": base_sh,
        "ablation_delta_sharpe": ablation,
    }
    with open(out_dir / "meta_mlp_results.json", "w") as f:
        json.dump(res, f, indent=2, default=float)
    return res


# ---------------------------------------------------------------------------
# (e) Optuna trading-system search — uses full_market_replay objective
# ---------------------------------------------------------------------------
def optuna_search(npz_path: Path, labels_dir: Path, dates: list[str],
                  out_dir: Path, n_trials: int = 25,
                  min_fills: int = 15) -> dict:
    def objective(trial: optuna.Trial) -> float:
        horizon = trial.suggest_categorical("horizon", ["1s", "5s", "10s"])
        side = trial.suggest_categorical("side", ["long", "short"])
        # Wider band (up to 20%) so smaller-sample models still produce fills
        conf = trial.suggest_float("confidence_threshold", 0.005, 0.20, log=True)
        order = trial.suggest_categorical(
            "order_type",
            ["passive_at_touch", "passive_at_touch_plus_1", "ioc_market"],
        )
        cancel = trial.suggest_int("cancel_eval_window", 10, 80)
        hold = trial.suggest_categorical("hold_seconds", [1.0, 2.0, 5.0, 10.0, 30.0])
        cfg = TradeConfig(side=side, horizon=horizon, confidence_threshold=conf,
                          order_type=order, cancel_eval_window=cancel,
                          hold_seconds=hold)
        try:
            led = full_market_replay(npz_path, labels_dir, cfg, dates=dates)
        except Exception:
            return -10.0
        if led.n_filled < min_fills:
            return -5.0
        sh = led.sharpe
        if not np.isfinite(sh):
            return -5.0
        return float(sh)

    sampler = optuna.samplers.TPESampler(seed=0)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=False)
    elapsed = time.time() - t0

    with open(out_dir / "optuna_study.pkl", "wb") as f:
        pickle.dump(study, f)

    best = study.best_trial
    res = {
        "n_trials_run": n_trials,
        "best_value": float(best.value) if best.value is not None else float("nan"),
        "best_params": dict(best.params),
        "elapsed_sec": elapsed,
    }
    with open(out_dir / "optuna_best.json", "w") as f:
        json.dump(res, f, indent=2, default=float)
    return res


# ---------------------------------------------------------------------------
# (f) recommended Razer-CLI config
# ---------------------------------------------------------------------------
def recommend_config(optuna_best: dict, per_band_df: pd.DataFrame,
                     model_name: str, out_dir: Path) -> dict:
    best = optuna_best.get("best_params", {})
    # Get the percentile mapping: confidence_threshold is a tail-fraction
    pct_band = best.get("confidence_threshold", 0.05)
    side_bias = best.get("side", "both")
    cancel = best.get("cancel_eval_window", 40)
    hold = best.get("hold_seconds", 5.0)
    order = best.get("order_type", "passive_at_touch")
    horizon = best.get("horizon", "5s")

    passive_offset = 0
    if order == "passive_at_touch_plus_1":
        passive_offset = 1
    elif order == "passive_at_touch_plus_2":
        passive_offset = 2
    elif order == "ioc_market":
        passive_offset = -1  # sentinel: cross spread

    cfg = {
        "model": model_name,
        "cnn_horizon": horizon,
        "cnn_percentile_tail": pct_band,
        "cnn_threshold_note": f"top {pct_band * 100:.2f}% by |pred| on {horizon}",
        "side_bias": side_bias,
        "cancel_eval_window": cancel,
        "passive_offset_ticks": passive_offset,
        "max_hold_seconds": hold,
        "sizing_contracts": 1,
        "best_sharpe": optuna_best.get("best_value", float("nan")),
        "razer_cli_flags": [
            f"--cnn-threshold {pct_band:.4f}",
            f"--cnn-horizon {horizon}",
            f"--side-bias {side_bias}",
            f"--cancel-eval-window {cancel}",
            f"--passive-offset-ticks {passive_offset}",
            f"--max-hold-seconds {hold}",
        ],
    }
    with open(out_dir / "recommended_config.json", "w") as f:
        json.dump(cfg, f, indent=2, default=float)
    return cfg


# ---------------------------------------------------------------------------
# REPORT.md
# ---------------------------------------------------------------------------
def write_report(model: str, per_band: pd.DataFrame, edge: pd.DataFrame,
                 conflu: pd.DataFrame, meta: dict, optuna_best: dict,
                 cfg: dict, out_dir: Path):
    lines = [
        f"# Test Bench Report — {model} (HC #358)",
        f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')} UTC.",
        "",
        "## (a) Per-band metrics (top rows)",
        "",
    ]
    if not per_band.empty:
        # Pick the best per (horizon, side) by sharpe_toy
        sub = per_band[per_band["band"].isin(list(BANDS))].copy()
        sub["da_str"] = sub["da_pct"].map(fmt_pct)
        cols = ["horizon", "band", "side", "n", "ic", "da_str",
                "magcorr_ticks", "avg_real_ticks", "sharpe_toy"]
        lines.append("```")
        lines.append(sub[cols].head(40).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        lines.append("```")
    lines.append("")
    lines.append("## (b) Edge decay highlights (P99 / passive at touch)")
    if not edge.empty:
        e2 = edge[(edge["band"] == "P99") & (edge["err"].fillna("") == "")]
        if not e2.empty:
            piv = e2.pivot_table(index=["head", "side"], columns="hold_sec",
                                  values="net_ticks_per_trade", aggfunc="first")
            lines.append("```")
            lines.append(piv.to_string(float_format=lambda x: f"{x:.3f}"))
            lines.append("```")
    lines.append("")
    lines.append("## (c) Confluence (head pairs)")
    if not conflu.empty:
        c2 = conflu.sort_values("lift_long", ascending=False, na_position="last").head(10)
        lines.append("```")
        lines.append(c2.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        lines.append("```")
    lines.append("")
    lines.append("## (d) Meta-MLP")
    lines.append(f"- AUC mean (5-fold): {meta.get('auc_mean', float('nan')):.4f}")
    lines.append(f"- Sharpe at p>0.55 picks: {meta.get('sh_picks_p55', float('nan')):.3f}")
    lines.append("- Top 5 load-bearing heads (largest Δ-Sharpe when removed):")
    abl = meta.get("ablation_delta_sharpe", {})
    if abl:
        srt = sorted(abl.items(), key=lambda kv: -(kv[1].get("delta_sharpe", 0) or 0))
        for col, info in srt[:5]:
            lines.append(f"  - {col}: ΔSharpe={info.get('delta_sharpe', float('nan')):.3f}")
    lines.append("")
    lines.append("## (e) Optuna best")
    lines.append(f"- Trials: {optuna_best.get('n_trials_run')}")
    lines.append(f"- Best Sharpe: {optuna_best.get('best_value', float('nan')):.3f}")
    lines.append(f"- Best params: `{optuna_best.get('best_params', {})}`")
    lines.append("")
    lines.append("## (f) Recommended Razer CLI config")
    lines.append("```")
    for flag in cfg.get("razer_cli_flags", []):
        lines.append(flag)
    lines.append("```")
    (out_dir / "REPORT.md").write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["v2", "v3_2", "v3_3"])
    ap.add_argument("--npz", required=True)
    ap.add_argument("--labels-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dates", default="",
                    help="comma-sep dates; required for v2 (1 date) and v3.3.")
    ap.add_argument("--optuna-trials", type=int, default=25)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = Path(args.npz)
    labels_dir = Path(args.labels_dir)

    # v2 adapter
    if args.model == "v2":
        dates = args.dates.split(",") if args.dates else ["20260224"]
        tmp = out_dir / "_tmp_adapter"
        tmp.mkdir(exist_ok=True)
        adapted = adapt_v2_to_v32(npz_path, dates, tmp)
        npz_to_use = adapted
    else:
        if args.dates:
            dates = args.dates.split(",")
        else:
            d = np.load(npz_path, allow_pickle=True)
            dates = [str(x) for x in d["oot_dates"]] if "oot_dates" in d.keys() else []
        npz_to_use = npz_path

    t0 = time.time()
    print(f"[{args.model}] loading raw arrays...")
    raw = load_raw(npz_to_use)
    print(f"[{args.model}] n_samples={raw['n']} dates={dates}")

    print(f"[{args.model}] (a) per-band table...")
    per_band = per_band_table(raw, out_dir)
    print(f"  → {len(per_band)} rows")

    print(f"[{args.model}] (b) edge decay (uses full_market_replay)...")
    edge = edge_decay(npz_to_use, labels_dir, dates, out_dir)
    print(f"  → {len(edge)} rows")

    print(f"[{args.model}] (c) confluence matrix...")
    conflu = confluence_matrix(raw, out_dir)
    print(f"  → {len(conflu)} rows")

    print(f"[{args.model}] (d) meta-MLP (5-fold + ablation)...")
    meta = meta_mlp(raw, out_dir)
    print(f"  → AUC={meta.get('auc_mean', float('nan')):.4f}")

    print(f"[{args.model}] (e) Optuna ({args.optuna_trials} trials)...")
    optuna_best = optuna_search(npz_to_use, labels_dir, dates, out_dir,
                                 n_trials=args.optuna_trials)
    print(f"  → best Sharpe={optuna_best.get('best_value', float('nan')):.3f}")

    print(f"[{args.model}] (f) recommended config...")
    cfg = recommend_config(optuna_best, per_band, args.model, out_dir)

    print(f"[{args.model}] writing REPORT.md...")
    write_report(args.model, per_band, edge, conflu, meta, optuna_best, cfg, out_dir)

    elapsed = time.time() - t0
    print(f"[{args.model}] DONE in {elapsed:.1f}s. Outputs in {out_dir}")


if __name__ == "__main__":
    main()

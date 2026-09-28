#!/usr/bin/env python3
"""
HC #270 — Phase 4: FIFO validator for the meta-LGBM gate.

Reads the per-fold OOT predictions emitted by `train_meta_lgbm_gate.py` and
evaluates the gate at multiple `p_win` threshold settings:

For each threshold:
  - Filter to signals with p_win >= threshold AND direction matches deploy filter
  - Compute the full HC #256 metric panel:
      n_trades, fill_rate, sum_gross, sum_net, avg_net_per_trade,
      median fold avg-NET, % folds positive (HC #254 ≥60% gate), Sortino,
      Sharpe, profit_factor, win_rate, expectancy
  - Concentration check (HC #258): max single-date share of NET ≤ 40%

Outputs:
  <out_dir>/meta_lgbm_gate_validation.csv  — one row per threshold
  <out_dir>/meta_lgbm_gate_validation.json — full panel + per-fold breakdown
  <out_dir>/best_threshold.json            — selected operating point

Selection: highest Sortino subject to (a) ≥60% folds positive, (b) ≥30 trades total,
           (c) max single-date concentration ≤ 40%.
"""

from pathlib import Path
import argparse
import json
import logging
import numpy as np
import pandas as pd

LVL3_ROOT = Path("/home/jupiter/Lvl3Quant")
DEFAULT_GATE_DIR = LVL3_ROOT / "output" / "meta_lgbm_gate_v1"
DEFAULT_FEAT_DIR = LVL3_ROOT / "output" / "meta_lgbm_features"
DEFAULT_REGIME_PATH = LVL3_ROOT / "output" / "regime_labels" / "oot_dates_regime.parquet"

COMMISSION_TICKS = 0.376  # Already in label_net_ticks (it's GROSS minus commission)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("meta_lgbm_validate")


def _sortino(x: np.ndarray, mar: float = 0.0) -> float:
    if len(x) < 2:
        return float("nan")
    excess = x - mar
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float("inf") if excess.mean() > 0 else float("nan")
    dd = np.sqrt(np.mean(downside ** 2))
    if dd == 0:
        return float("nan")
    return float(excess.mean() / dd)


def _sharpe(x: np.ndarray) -> float:
    if len(x) < 2 or x.std() == 0:
        return float("nan")
    return float(x.mean() / x.std())


def _profit_factor(x: np.ndarray) -> float:
    pos = x[x > 0].sum(); neg = -x[x < 0].sum()
    if neg == 0:
        return float("inf") if pos > 0 else float("nan")
    return float(pos / neg)


def _evaluate_filled_set(filled: pd.DataFrame, n_signals: int, label: str = "") -> dict:
    """Compute the full HC #256 metric panel for an already-filtered, already-filled
    DataFrame. `filled` must have columns: label_net_ticks, date. n_signals is the
    pre-fill count (for fill_rate)."""
    n_trades = len(filled)
    fill_rate = n_trades / max(1, n_signals)
    if n_trades == 0:
        return {
            "label": label, "n_signals": n_signals, "n_trades": 0,
            "fill_rate": fill_rate,
            "sum_gross_ticks": 0.0, "sum_net_ticks": 0.0,
            "avg_net_per_trade": float("nan"),
            "median_fold_avg_net": float("nan"),
            "pct_folds_positive": 0.0, "n_folds_with_trades": 0,
            "sortino": float("nan"), "sharpe": float("nan"),
            "profit_factor": float("nan"), "win_rate": float("nan"),
            "expectancy_t": float("nan"),
            "max_date_share_of_net": float("nan"),
            "passes_hc254": False, "passes_hc258": False,
        }
    nets = filled["label_net_ticks"].values
    gross = nets + COMMISSION_TICKS
    sum_net = float(nets.sum()); sum_gross = float(gross.sum())
    avg_net = float(nets.mean())
    wr = float((nets > 0).mean())
    per_date = filled.groupby("date")["label_net_ticks"].agg(["sum", "mean", "count"])
    n_folds_with_trades = len(per_date)
    pct_folds_positive = float((per_date["sum"] > 0).mean()) if n_folds_with_trades else 0.0
    median_fold_avg_net = float(per_date["mean"].median()) if n_folds_with_trades else float("nan")
    max_date_share = float(per_date["sum"].abs().max() / max(1e-6, abs(sum_net))) if sum_net != 0 else float("nan")
    return {
        "label": label, "n_signals": n_signals, "n_trades": n_trades,
        "fill_rate": fill_rate,
        "sum_gross_ticks": sum_gross, "sum_net_ticks": sum_net,
        "avg_net_per_trade": avg_net,
        "median_fold_avg_net": median_fold_avg_net,
        "pct_folds_positive": pct_folds_positive,
        "n_folds_with_trades": n_folds_with_trades,
        "sortino": _sortino(nets), "sharpe": _sharpe(nets),
        "profit_factor": _profit_factor(nets), "win_rate": wr,
        "expectancy_t": avg_net,
        "max_date_share_of_net": max_date_share,
        "passes_hc254": pct_folds_positive >= 0.60,
        "passes_hc258": (max_date_share <= 0.40) if not np.isnan(max_date_share) else False,
    }


def _load_features_for_dominance(feat_dir: Path, dates: list[str]) -> pd.DataFrame:
    """Load the enriched feature parquets for the OOT dates so we can run
    PatchTST-only and CNN-Mamba-only baselines at the same trade count.

    Returns a long DataFrame with columns:
       date, signal_ts_ns, direction, is_filled, label_net_ticks,
       abs_pred_1s, pt_abs_pred_1s
    """
    rows = []
    cols_keep = ["date", "signal_ts_ns", "direction", "is_filled",
                 "label_net_ticks", "abs_pred_1s", "pt_abs_pred_1s"]
    for d in sorted(set(dates)):
        p = feat_dir / f"{d}_signals_enriched.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        # Some early-pipeline parquets may lack one of the abs columns
        for c in cols_keep:
            if c not in df.columns:
                df[c] = np.nan
        rows.append(df[cols_keep].copy())
    if not rows:
        return pd.DataFrame(columns=cols_keep)
    out = pd.concat(rows, ignore_index=True)
    return out


def _baseline_top_n_trades(feat_df: pd.DataFrame, score_col: str,
                            n_trades_target: int, side: str) -> tuple[pd.DataFrame, int]:
    """Pick the top-N FILLED trades from the same OOT signal universe by `score_col`.
    Match the meta operating point's trade count (HC #271(C)).

    We rank ALL signals (including unfilled) by score_col descending, walk down the
    list, and accumulate the first n_trades_target *filled* signals — this is the
    realistic equivalent of "deploy top-X% by alpha confidence and let FIFO sort fills."

    Returns (filled_df, n_signals_to_get_those_trades).
    """
    f = feat_df.copy()
    if side == "long":
        f = f[f["direction"] == "long"]
    elif side == "short":
        f = f[f["direction"] == "short"]
    f = f[~f[score_col].isna()].copy()
    if len(f) == 0 or n_trades_target <= 0:
        return f.iloc[0:0], 0
    f = f.sort_values(score_col, ascending=False).reset_index(drop=True)
    # Walk the ranked list and pick the smallest cutoff that yields >= n_trades_target fills
    f["_filled_int"] = f["is_filled"].astype(int)
    f["_cum_fills"] = f["_filled_int"].cumsum()
    if int(f["_cum_fills"].iloc[-1]) < n_trades_target:
        # Not enough fills available — take everything
        return f[f["is_filled"]].copy(), len(f)
    cutoff_idx = int((f["_cum_fills"] >= n_trades_target).idxmax())
    n_signals_taken = cutoff_idx + 1
    sub = f.iloc[: n_signals_taken]
    filled = sub[sub["is_filled"]].copy()
    return filled, n_signals_taken


def evaluate_dominance_at_meta_op_point(meta_row: dict,
                                         feat_df: pd.DataFrame,
                                         side: str) -> dict:
    """Run PatchTST-only DA-gate and CNN-Mamba-only confidence-tier baselines at the
    same number of filled trades as the meta operating point. HC #271(C) test:
    meta should dominate Sortino + WR + PF + per-regime breakdown.

    Returns a flat dict keyed by source ('meta', 'patchtst_only', 'cnnmamba_only')
    plus boolean dominance flags.
    """
    n_target = int(meta_row.get("n_trades", 0))

    # PatchTST DA-gate baseline (rank by |pt_pred_1s|)
    pt_filled, pt_n_sig = _baseline_top_n_trades(feat_df, "pt_abs_pred_1s",
                                                  n_target, side)
    pt_panel = _evaluate_filled_set(pt_filled, pt_n_sig, label="patchtst_only_da")

    # CNN-Mamba confidence-tier baseline (rank by |abs_pred_1s|)
    cm_filled, cm_n_sig = _baseline_top_n_trades(feat_df, "abs_pred_1s",
                                                  n_target, side)
    cm_panel = _evaluate_filled_set(cm_filled, cm_n_sig, label="cnnmamba_only_conf")

    metrics_to_check = ["sortino", "win_rate", "profit_factor",
                        "sum_net_ticks", "pct_folds_positive"]
    flags = {}
    for k in metrics_to_check:
        meta_v = meta_row.get(k, float("nan"))
        flags[f"meta_beats_pt_{k}"] = bool(
            (not np.isnan(meta_v)) and (np.isnan(pt_panel[k]) or meta_v > pt_panel[k])
        )
        flags[f"meta_beats_cm_{k}"] = bool(
            (not np.isnan(meta_v)) and (np.isnan(cm_panel[k]) or meta_v > cm_panel[k])
        )
    flags["meta_dominates_both"] = bool(
        all(flags[f"meta_beats_pt_{k}"] and flags[f"meta_beats_cm_{k}"]
            for k in metrics_to_check)
    )

    return {
        "n_trades_target": n_target,
        "meta": {k: meta_row.get(k) for k in metrics_to_check + ["n_trades"]},
        "patchtst_only": pt_panel,
        "cnnmamba_only": cm_panel,
        **flags,
    }


def regime_breakdown(filled: pd.DataFrame, regime_path: Path) -> dict:
    """Per-regime breakdown of a filled trade set. Returns dict per trend label."""
    if not regime_path.exists() or len(filled) == 0:
        return {}
    reg = pd.read_parquet(regime_path)
    f = filled.copy()
    f["date"] = f["date"].astype(str)
    reg["date"] = reg["date"].astype(str)
    j = f.merge(reg[["date", "trend_label", "vol_bucket"]], on="date", how="left")
    out = {}
    for tl, sub in j.groupby("trend_label", dropna=False):
        nets = sub["label_net_ticks"].values
        per_date = sub.groupby("date")["label_net_ticks"].sum()
        out[str(tl)] = {
            "n_trades": int(len(sub)),
            "n_dates": int(per_date.shape[0]),
            "sum_net_ticks": float(nets.sum()),
            "avg_net_per_trade": float(nets.mean()) if len(nets) else float("nan"),
            "win_rate": float((nets > 0).mean()) if len(nets) else float("nan"),
            "pct_dates_positive": float((per_date > 0).mean()) if len(per_date) else 0.0,
        }
    return out


def evaluate_threshold(df: pd.DataFrame, threshold: float, side: str) -> dict:
    """df has columns: signal_ts_ns, p_win, direction, is_filled, label_net_ticks, date.
       Filter, then compute the panel."""
    sub = df[df["p_win"] >= threshold].copy()
    if side == "long":
        sub = sub[sub["direction"] == "long"]
    elif side == "short":
        sub = sub[sub["direction"] == "short"]
    n_signals = len(sub)
    filled = sub[sub["is_filled"] == True].copy()
    n_trades = len(filled)
    fill_rate = n_trades / max(1, n_signals)

    if n_trades == 0:
        return {
            "threshold": threshold, "side": side, "n_signals": n_signals,
            "n_trades": 0, "fill_rate": fill_rate,
            "sum_gross_ticks": 0.0, "sum_net_ticks": 0.0,
            "avg_net_per_trade": float("nan"),
            "median_fold_avg_net": float("nan"),
            "pct_folds_positive": 0.0,
            "n_folds_with_trades": 0,
            "sortino": float("nan"), "sharpe": float("nan"),
            "profit_factor": float("nan"), "win_rate": float("nan"),
            "expectancy_t": float("nan"),
            "max_date_share_of_net": float("nan"),
            "passes_hc254": False,
            "passes_hc258": False,
        }

    nets = filled["label_net_ticks"].values
    gross = nets + COMMISSION_TICKS  # reverse out commission to recover gross
    sum_net = float(nets.sum())
    sum_gross = float(gross.sum())
    avg_net = float(nets.mean())
    wr = float((nets > 0).mean())
    expectancy = avg_net  # per-trade NET tick expectancy

    # Per-date breakdown for fold-positive % and concentration
    per_date = filled.groupby("date")["label_net_ticks"].agg(["sum", "mean", "count"])
    n_folds_with_trades = len(per_date)
    pct_folds_positive = float((per_date["sum"] > 0).mean()) if n_folds_with_trades else 0.0
    median_fold_avg_net = float(per_date["mean"].median()) if n_folds_with_trades else float("nan")
    max_date_share = float(per_date["sum"].abs().max() / max(1e-6, abs(sum_net))) if sum_net != 0 else float("nan")

    return {
        "threshold": threshold, "side": side, "n_signals": n_signals,
        "n_trades": n_trades, "fill_rate": fill_rate,
        "sum_gross_ticks": sum_gross, "sum_net_ticks": sum_net,
        "avg_net_per_trade": avg_net,
        "median_fold_avg_net": median_fold_avg_net,
        "pct_folds_positive": pct_folds_positive,
        "n_folds_with_trades": n_folds_with_trades,
        "sortino": _sortino(nets),
        "sharpe": _sharpe(nets),
        "profit_factor": _profit_factor(nets),
        "win_rate": wr,
        "expectancy_t": expectancy,
        "max_date_share_of_net": max_date_share,
        "passes_hc254": pct_folds_positive >= 0.60,
        "passes_hc258": (max_date_share <= 0.40) if not np.isnan(max_date_share) else False,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gate-dir", type=str, default=str(DEFAULT_GATE_DIR))
    p.add_argument("--side", type=str, default="short", choices=["both", "long", "short"],
                   help="Deploy filter side. HC #267 prefers short-only.")
    p.add_argument("--thresholds", type=str,
                   default="0.30,0.40,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95",
                   help="Comma-separated list of p_win thresholds")
    p.add_argument("--top-pcts", type=str,
                   default="",
                   help="Optional CSV of top-percentile cuts (e.g. '0.001,0.005,0.01,0.02') "
                        "evaluated alongside fixed thresholds")
    p.add_argument("--feat-dir", type=str, default=str(DEFAULT_FEAT_DIR),
                   help="Per-date enriched feature parquets (for HC #271(C) dominance test)")
    p.add_argument("--regime-path", type=str, default=str(DEFAULT_REGIME_PATH),
                   help="OOT regime label parquet (for HC #271(A) per-regime breakdown)")
    p.add_argument("--skip-dominance", action="store_true",
                   help="Skip HC #271(C) PatchTST/CNN-Mamba baselines")
    args = p.parse_args()

    gate_dir = Path(args.gate_dir)
    concat_path = gate_dir / "concat_oot_predictions.npz"
    if not concat_path.exists():
        log.error(f"Missing {concat_path}; run Phase 3 first")
        raise SystemExit(1)

    d = np.load(concat_path, allow_pickle=True)
    df = pd.DataFrame({
        "signal_ts_ns": d["signal_ts_ns"],
        "p_win": d["p_win"],
        "label_winner": d["label_winner"],
        "label_net_ticks": d["label_net_ticks"],
        "direction": d["direction"],
        "is_filled": d["is_filled"],
        "date": d["date"],
    })
    log.info(f"Loaded {len(df):,} OOT predictions across {df['date'].nunique()} dates")
    log.info(f"  base WR={df['label_winner'].mean():.3f}  filled%={df['is_filled'].mean():.3f}")

    thresholds = [float(t) for t in args.thresholds.split(",") if t.strip()]
    top_pcts = [float(t) for t in args.top_pcts.split(",") if t.strip()]

    rows = []
    # Fixed-threshold evaluation
    for t in thresholds:
        rows.append(evaluate_threshold(df, t, args.side))

    # Top-pct evaluation: pick the threshold that yields ~top X% of the side-filtered signals
    if top_pcts:
        side_df = df if args.side == "both" else df[df["direction"] == args.side]
        side_df = side_df.sort_values("p_win", ascending=False)
        n = len(side_df)
        for pct in top_pcts:
            k = max(1, int(n * pct))
            cutoff = float(side_df.iloc[k - 1]["p_win"])
            r = evaluate_threshold(df, cutoff, args.side)
            r["target_top_pct"] = pct
            r["threshold_label"] = f"top{pct:.4f}"
            rows.append(r)

    out = pd.DataFrame(rows)
    csv_path = gate_dir / "meta_lgbm_gate_validation.csv"
    out.to_csv(csv_path, index=False)
    log.info(f"Wrote {csv_path}")

    # Pretty print
    log.info("\nTHRESHOLD SWEEP (side=%s):" % args.side)
    cols = ["threshold", "n_trades", "sum_net_ticks", "avg_net_per_trade",
            "pct_folds_positive", "sortino", "win_rate", "max_date_share_of_net",
            "passes_hc254", "passes_hc258"]
    log.info(out[cols].to_string(index=False))

    # Selection
    cand = out[(out["passes_hc254"] == True) & (out["passes_hc258"] == True) &
               (out["n_trades"] >= 30) & (out["sum_net_ticks"] > 0)]
    if len(cand) == 0:
        log.warning("NO THRESHOLD PASSES HC #254 + HC #258 + n_trades≥30 + NET>0.")
        best = None
    else:
        cand_sorted = cand.sort_values("sortino", ascending=False)
        best = cand_sorted.iloc[0].to_dict()
        log.info(f"\nBEST THRESHOLD: {best['threshold']:.3f}  Sortino={best['sortino']:.3f}  "
                 f"NET={best['sum_net_ticks']:+.1f}t  trades={best['n_trades']}  "
                 f"folds+={best['pct_folds_positive']:.1%}  "
                 f"max_date_share={best['max_date_share_of_net']:.1%}")

    # ---- HC #271(C) DOMINANCE TEST ----
    dominance_rows = []
    if not args.skip_dominance:
        feat_dir = Path(args.feat_dir)
        oot_dates = sorted(set(str(d) for d in df["date"].unique()))
        log.info(f"\nLoading enriched features for {len(oot_dates)} OOT dates from {feat_dir} ...")
        feat_df = _load_features_for_dominance(feat_dir, oot_dates)
        if len(feat_df) == 0:
            log.warning("No enriched feature parquets found — skipping dominance test")
        else:
            log.info(f"  features loaded: {len(feat_df):,} rows | "
                     f"{feat_df[['abs_pred_1s','pt_abs_pred_1s']].notna().sum().to_dict()}")
            log.info(f"\n=== HC #271(C) DOMINANCE TEST (meta vs PatchTST-only vs CNN-Mamba-only) ===")
            log.info("At each meta operating point with N filled trades, the parents are forced "
                     "to take the top-N filled trades by |alpha confidence|.")
            for r in rows:
                if r.get("n_trades", 0) < 5:
                    continue
                rep = evaluate_dominance_at_meta_op_point(r, feat_df, args.side)
                tag = r.get("threshold_label") or f"thr={r.get('threshold'):.3f}"
                log.info(f"\n[{tag}] N={r['n_trades']} side={args.side}")
                log.info(f"  META       : Sortino={r.get('sortino'):.3f}  WR={r.get('win_rate'):.3f}  "
                         f"PF={r.get('profit_factor'):.3f}  NET={r.get('sum_net_ticks'):+.1f}t  "
                         f"%folds+={r.get('pct_folds_positive'):.1%}")
                pt = rep["patchtst_only"]; cm = rep["cnnmamba_only"]
                log.info(f"  PatchTST   : Sortino={pt['sortino']:.3f}  WR={pt['win_rate']:.3f}  "
                         f"PF={pt['profit_factor']:.3f}  NET={pt['sum_net_ticks']:+.1f}t  "
                         f"%folds+={pt['pct_folds_positive']:.1%}  (n={pt['n_trades']})")
                log.info(f"  CNN-Mamba  : Sortino={cm['sortino']:.3f}  WR={cm['win_rate']:.3f}  "
                         f"PF={cm['profit_factor']:.3f}  NET={cm['sum_net_ticks']:+.1f}t  "
                         f"%folds+={cm['pct_folds_positive']:.1%}  (n={cm['n_trades']})")
                log.info(f"  DOMINATES_BOTH? {'✅ YES' if rep['meta_dominates_both'] else '❌ NO'}")
                dominance_rows.append({"op_point": tag, **rep})
            dom_path = gate_dir / "meta_lgbm_dominance_test.json"
            with open(dom_path, "w") as f:
                json.dump(dominance_rows, f, indent=2, default=str)
            log.info(f"\nWrote {dom_path}")

    # ---- HC #271(A) PER-REGIME BREAKDOWN at the BEST operating point ----
    regime_report = {}
    if best is not None:
        regime_path = Path(args.regime_path)
        if regime_path.exists():
            best_thresh = best["threshold"]
            sub = df[df["p_win"] >= best_thresh].copy()
            if args.side != "both":
                sub = sub[sub["direction"] == args.side]
            best_filled = sub[sub["is_filled"] == True].copy()
            regime_report = regime_breakdown(best_filled, regime_path)
            log.info(f"\n=== HC #271(A) PER-REGIME BREAKDOWN at BEST threshold ({best_thresh:.3f}) ===")
            for tl, m in regime_report.items():
                log.info(f"  trend={tl:>5}: n_trades={m['n_trades']:>4}  n_dates={m['n_dates']:>2}  "
                         f"NET={m['sum_net_ticks']:+8.2f}t  WR={m['win_rate']:.3f}  "
                         f"%dates+={m['pct_dates_positive']:.1%}")
            # HC #271(A) verdicts
            trend_nets = {tl: m["sum_net_ticks"] for tl, m in regime_report.items()}
            total_net = sum(trend_nets.values())
            n_pos_regimes = sum(1 for v in trend_nets.values() if v > 0)
            max_share = (max(abs(v) for v in trend_nets.values()) / abs(total_net)
                         if total_net != 0 else float("nan"))
            log.info(f"\nHC #271(A) GATES at best op point:")
            log.info(f"  Profitable in ≥1 trend regime?  {'PASS' if n_pos_regimes >= 1 else 'FAIL'} "
                     f"({n_pos_regimes}/{len(trend_nets)})")
            log.info(f"  Max single-trend |NET|/|total|: {max_share:.1%} "
                     f"{'PASS' if max_share <= 0.5 else 'FAIL'} (HC #271(A) ceiling 50%)")
        else:
            log.warning(f"Regime parquet not at {regime_path} — skipping HC #271(A) breakdown")

    with open(gate_dir / "meta_lgbm_gate_validation.json", "w") as f:
        json.dump({"all": rows, "best": best, "side": args.side,
                   "dominance": dominance_rows,
                   "regime_breakdown_best": regime_report}, f, indent=2, default=str)
    if best is not None:
        with open(gate_dir / "best_threshold.json", "w") as f:
            json.dump({**best, "regime_breakdown": regime_report}, f, indent=2, default=str)
    log.info("DONE.")


if __name__ == "__main__":
    main()

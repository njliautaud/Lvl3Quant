#!/usr/bin/env python3
"""
HC #428 R1 — Regime-Agnostic OOT Validation Gate
HC #344   — Day-concentration cap <= 0.70

Tests whether the production meta-model performs equally well on GREEN, RED,
and FLAT market days.  A "profitable" claim requires passing BOTH gates.

Regime classification uses canonical regime labels from
  output/regime_labels/oot_dates_regime.parquet
with fallback to MBO event data (labels_10s strided sum) for missing dates.

Thresholds (per HC #428 R1 + canonical replay convention):
  - GREEN:  close_minus_open_ticks >= +8   (>= 2 pts)
  - RED:    close_minus_open_ticks <= -8   (<= -2 pts)
  - FLAT:   in between

Runs on CPU only.  Cross-platform (Windows/Linux via pathlib).
"""

import json
import os
import platform
import sys
import time
from pathlib import Path

import warnings
import numpy as np
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── paths (auto-detect OS) ──────────────────────────────────────────────
if platform.system() == "Windows":
    LVL3_ROOT = Path(r"C:\Users\claude\Lvl3Quant")
else:
    LVL3_ROOT = Path("/home/jupiter/Lvl3Quant")
    # Also check /home/nick/Lvl3Quant if jupiter doesn't exist
    if not LVL3_ROOT.exists():
        alt = Path("/home/nick/Lvl3Quant")
        if alt.exists():
            LVL3_ROOT = alt

REGIME_PARQUET = LVL3_ROOT / "output" / "regime_labels" / "oot_dates_regime.parquet"
MBO_DIR = LVL3_ROOT / "data" / "processed" / "mbo_events_smart_v3"
OUT_DIR = LVL3_ROOT / "output" / "regime_stratified_v1"

# model output directories
MODEL_DIRS = {
    "shorts": LVL3_ROOT / "output" / "meta_production_v1",
    "longs":  LVL3_ROOT / "output" / "meta_production_longs_v1",
}

# HC #428 R1 thresholds
REGIME_TICK_THRESHOLD = 8    # 2 pts = 8 ticks
REGIME_DELTA_MAX      = 0.50 # |Sharpe_g - Sharpe_r| / max(|.|, |.|) must be <= this
DAY_CONC_MAX          = 0.70 # HC #344

# Cost constants (canonical — CLAUDE.md)
COMMISSION_TICKS = 0.376  # ES round-trip AMP/Rithmic


# ── regime labels ───────────────────────────────────────────────────────
def load_regime_labels() -> dict:
    """Load canonical regime labels.  Returns {date_str: close_minus_open_ticks}."""
    labels = {}
    if REGIME_PARQUET.exists():
        try:
            import pandas as pd
            df = pd.read_parquet(REGIME_PARQUET)
            for _, row in df.iterrows():
                d = str(row["date"]).zfill(8)
                labels[d] = float(row["close_minus_open_ticks"])
        except Exception as e:
            print(f"[warn] Could not load regime parquet: {e}")
    return labels


def estimate_regime_from_mbo(date_str: str) -> float:
    """Fallback: estimate close-minus-open from MBO labels_10s strided sum.
    Returns approximate ticks move (very rough)."""
    mbo_file = MBO_DIR / f"{date_str}_mbo_events.npz"
    if not mbo_file.exists():
        return 0.0
    try:
        d = np.load(mbo_file, allow_pickle=True)
        # Use labels_10s at stride=40 (non-overlapping 10s windows)
        labels = d["labels_10s"]
        valid = ~np.isnan(labels)
        if valid.sum() < 100:
            return 0.0
        # Use mean * number_of_10s_windows as a directional proxy
        mean_label = float(np.nanmean(labels))
        # If mean_label is positive, day is trending up, etc.
        # Scale to approximate close-minus-open: mean * sqrt(n_windows) for direction
        # Simpler: just use the sign and magnitude of the mean label
        # mean_label > 0 consistently => green day
        # Threshold: map to rough ticks using mean * 100 (empirical scaling)
        return mean_label * 100
    except Exception as e:
        print(f"[warn] MBO fallback failed for {date_str}: {e}")
        return 0.0


def classify_regime(ticks: float) -> str:
    if ticks >= REGIME_TICK_THRESHOLD:
        return "green"
    elif ticks <= -REGIME_TICK_THRESHOLD:
        return "red"
    return "flat"


def get_regime_map(dates: list) -> dict:
    """Returns {date_str: {'regime': str, 'ticks': float, 'source': str}}."""
    canonical = load_regime_labels()
    result = {}
    for d in dates:
        d_str = str(d).zfill(8)
        if d_str in canonical:
            ticks = canonical[d_str]
            result[d_str] = {
                "regime": classify_regime(ticks),
                "ticks": ticks,
                "source": "canonical",
            }
        else:
            ticks = estimate_regime_from_mbo(d_str)
            result[d_str] = {
                "regime": classify_regime(ticks),
                "ticks": ticks,
                "source": "mbo_fallback",
            }
    return result


# ── metrics ─────────────────────────────────────────────────────────────
def sharpe_ann(daily_pnl: np.ndarray) -> float:
    """Annualized Sharpe from daily PnL array (in ticks)."""
    if len(daily_pnl) < 2:
        return float("nan")
    s = float(np.std(daily_pnl, ddof=1))
    if s < 1e-12:
        return float("nan")
    return float(np.mean(daily_pnl) / s * np.sqrt(252))


def sortino_ann(daily_pnl: np.ndarray) -> float:
    """Annualized Sortino from daily PnL array."""
    if len(daily_pnl) < 2:
        return float("nan")
    downside = daily_pnl[daily_pnl < 0]
    if len(downside) < 1:
        return float("inf") if np.mean(daily_pnl) > 0 else float("nan")
    ds = float(np.std(downside, ddof=1))
    if ds < 1e-12:
        return float("nan")
    return float(np.mean(daily_pnl) / ds * np.sqrt(252))


def profit_factor(pnl_array: np.ndarray) -> float:
    """Gross wins / gross losses."""
    wins = pnl_array[pnl_array > 0].sum()
    losses = -pnl_array[pnl_array < 0].sum()
    if losses < 1e-12:
        return float("inf") if wins > 0 else float("nan")
    return float(wins / losses)


def win_rate(pnl_array: np.ndarray) -> float:
    if len(pnl_array) == 0:
        return float("nan")
    return float((pnl_array > 0).sum() / len(pnl_array) * 100)


def day_concentration(daily_pnl: np.ndarray) -> float:
    """Max single day's |PnL| / total |PnL| sum.  HC #344: must be <= 0.70."""
    abs_pnl = np.abs(daily_pnl)
    total = abs_pnl.sum()
    if total < 1e-12:
        return float("nan")
    return float(abs_pnl.max() / total)


def compute_regime_metrics(daily_pnl: np.ndarray, per_trade_pnl: np.ndarray,
                           n_trades: int) -> dict:
    """Compute full metric suite for a regime subset."""
    return {
        "n_days": int(len(daily_pnl)),
        "n_trades": int(n_trades),
        "sharpe": sharpe_ann(daily_pnl),
        "sortino": sortino_ann(daily_pnl),
        "pf": profit_factor(per_trade_pnl),
        "wr": win_rate(per_trade_pnl),
        "mean_ticks": float(np.mean(per_trade_pnl)) if len(per_trade_pnl) > 0 else float("nan"),
        "total_ticks": float(np.sum(per_trade_pnl)) if len(per_trade_pnl) > 0 else float("nan"),
        "daily_mean": float(np.mean(daily_pnl)) if len(daily_pnl) > 0 else float("nan"),
    }


# ── load model results ──────────────────────────────────────────────────
def load_model_results(model_dir: Path, side: str) -> dict:
    """Load per-fold results and predictions from a meta-model output dir.

    Returns: {
        'per_fold': [{'date': str, 'n_test': int, 'mean_pnl': float, ...}],
        'predictions': np.ndarray or None,  (concat)
        'actuals': np.ndarray or None,       (concat)
        'has_concat': bool,
    }
    """
    results_file = model_dir / "results.json"
    if not results_file.exists():
        return None

    with open(results_file) as f:
        data = json.load(f)

    per_fold = data.get("per_fold", [])

    # Try to load concat predictions
    concat_file = model_dir / "concat_predictions.npz"
    predictions = None
    actuals = None
    has_concat = False
    if concat_file.exists():
        try:
            npz = np.load(concat_file, allow_pickle=True)
            predictions = npz.get("predictions", None)
            actuals = npz.get("actuals", None)
            has_concat = predictions is not None and actuals is not None
        except Exception as e:
            print(f"[warn] Could not load {concat_file}: {e}")

    return {
        "per_fold": per_fold,
        "predictions": predictions,
        "actuals": actuals,
        "has_concat": has_concat,
        "filter_results": data.get("filter_results", {}),
        "concat_corr": data.get("concat_corr", None),
    }


def split_concat_by_fold(predictions, actuals, per_fold):
    """Split concatenated predictions/actuals into per-date arrays using n_test."""
    result = {}
    offset = 0
    for fold in per_fold:
        n = fold["n_test"]
        date_str = str(fold["date"]).zfill(8)
        result[date_str] = {
            "predictions": predictions[offset:offset + n],
            "actuals": actuals[offset:offset + n],
        }
        offset += n
    assert offset == len(predictions), \
        f"Sum of n_test ({offset}) != len(predictions) ({len(predictions)})"
    return result


# ── main analysis ────────────────────────────────────────────────────────
def analyze_side(side: str, model_dir: Path, regime_map: dict,
                 cost_ticks: float) -> dict:
    """Run HC #428 R1 analysis for one side (shorts or longs)."""
    print(f"\n{'='*70}")
    print(f"  {side.upper()} — {model_dir}")
    print(f"{'='*70}")

    data = load_model_results(model_dir, side)
    if data is None:
        print(f"  [skip] No results.json found in {model_dir}")
        return {"status": "skipped", "reason": "no results.json"}

    per_fold = data["per_fold"]
    dates = [str(f["date"]).zfill(8) for f in per_fold]

    # Build regime map for these dates
    regimes = get_regime_map(dates)

    print(f"\n  OOT dates: {len(dates)}")
    print(f"  {'Date':<10} {'Regime':<6} {'Ticks':>8} {'Source':<12} "
          f"{'N_test':>7} {'Mean_PnL':>9} {'WR':>6} {'Corr':>7}")
    print(f"  {'-'*70}")

    for fold in per_fold:
        d = str(fold["date"]).zfill(8)
        r = regimes.get(d, {"regime": "?", "ticks": 0, "source": "?"})
        wr_str = f"{fold.get('wr', float('nan')):.1f}%" if 'wr' in fold else "N/A"
        print(f"  {d:<10} {r['regime']:<6} {r['ticks']:>+8.0f} {r['source']:<12} "
              f"{fold['n_test']:>7} {fold.get('mean_pnl', float('nan')):>+9.3f} "
              f"{wr_str:>6} {fold['corr']:>+7.4f}")

    # ── Per-trade PnL by date ────────────────────────────────────────────
    # If we have concat predictions, compute per-trade PnL from actuals
    # actuals = signed ticks (y_true from training = realized return in ticks)
    # The meta-model's job is to predict which trades are profitable.
    # For "shorts" side: actuals > 0 means the short was profitable.
    # net PnL per trade = actuals - commission cost
    daily_pnl = {}       # {date: total ticks}
    daily_trades = {}    # {date: trade_pnl_array}
    daily_n = {}         # {date: n_trades}

    per_date_data = None  # for filtered analysis later

    if data["has_concat"]:
        per_date_data = split_concat_by_fold(data["predictions"], data["actuals"],
                                             per_fold)
        for d, arrays in per_date_data.items():
            act = arrays["actuals"].astype(np.float64)
            # Net per trade = actual ticks - commission
            net = act - cost_ticks
            daily_pnl[d] = float(net.sum())
            daily_trades[d] = net
            daily_n[d] = len(net)
    else:
        # Fallback: use per-fold mean_pnl * n_test as daily PnL
        print("\n  [info] No concat predictions; using per-fold mean_pnl for daily PnL")
        print("  [warn] WR and PF from fallback are approximate (all trades get same PnL)")
        for fold in per_fold:
            d = str(fold["date"]).zfill(8)
            n = fold["n_test"]
            mean_pnl = fold.get("mean_pnl", 0)
            net_mean = mean_pnl - cost_ticks
            daily_pnl[d] = net_mean * n
            # Synthesize per-trade pnl from mean (best we can do without concat)
            daily_trades[d] = np.full(n, net_mean)
            daily_n[d] = n

    # ── Stratify by regime ───────────────────────────────────────────────
    regime_daily = {"green": [], "red": [], "flat": []}
    regime_trades = {"green": [], "red": [], "flat": []}
    regime_n = {"green": 0, "red": 0, "flat": 0}

    for d in dates:
        r = regimes.get(d, {"regime": "flat"})["regime"]
        if d in daily_pnl:
            regime_daily[r].append(daily_pnl[d])
            if d in daily_trades:
                regime_trades[r].append(daily_trades[d])
            regime_n[r] += daily_n.get(d, 0)

    # Convert to arrays
    all_daily = np.array([daily_pnl[d] for d in dates if d in daily_pnl])
    all_trades = np.concatenate([daily_trades[d] for d in dates if d in daily_trades]) \
        if any(d in daily_trades for d in dates) else np.array([])

    results = {"side": side, "dates": dates, "n_oot_days": len(dates)}
    results["regimes"] = {}

    print(f"\n  {'─'*70}")
    print(f"  REGIME STRATIFICATION (cost = {cost_ticks:.3f} ticks RT)")
    print(f"  {'─'*70}")
    fmt = "  {:<8} {:>6} {:>7} {:>8} {:>8} {:>6} {:>9} {:>10}"
    print(fmt.format("Regime", "Days", "Trades", "Sharpe", "Sortino", "WR%", "PF", "Mean_tk"))
    print(f"  {'-'*70}")

    for regime in ["green", "red", "flat", "ALL"]:
        if regime == "ALL":
            dpnl = all_daily
            tpnl = all_trades
            nt = len(all_trades)
        else:
            dpnl = np.array(regime_daily[regime]) if regime_daily[regime] else np.array([])
            tpnl = np.concatenate(regime_trades[regime]) \
                if regime_trades[regime] else np.array([])
            nt = regime_n[regime]

        metrics = compute_regime_metrics(dpnl, tpnl, nt)
        results["regimes"][regime] = metrics

        sh = f"{metrics['sharpe']:.2f}" if np.isfinite(metrics['sharpe']) else "N/A"
        so = f"{metrics['sortino']:.2f}" if np.isfinite(metrics['sortino']) else "N/A"
        pf_s = f"{metrics['pf']:.2f}" if np.isfinite(metrics['pf']) else "inf"
        wr_s = f"{metrics['wr']:.1f}" if np.isfinite(metrics['wr']) else "N/A"
        mt = f"{metrics['mean_ticks']:+.3f}" if np.isfinite(metrics['mean_ticks']) else "N/A"

        label = regime.upper() if regime != "ALL" else "ALL"
        print(fmt.format(label, metrics['n_days'], metrics['n_trades'],
                         sh, so, wr_s, pf_s, mt))

    # ── HC #428 R1 Gate ──────────────────────────────────────────────────
    sg = results["regimes"].get("green", {}).get("sharpe", float("nan"))
    sr = results["regimes"].get("red", {}).get("sharpe", float("nan"))

    if np.isfinite(sg) and np.isfinite(sr):
        mx = max(abs(sg), abs(sr))
        if mx > 0:
            delta_ratio = abs(sg - sr) / mx
        else:
            delta_ratio = 0.0
    else:
        delta_ratio = float("nan")
        # If one regime has 0 days, we can't test — flag as warning
        n_green = results["regimes"].get("green", {}).get("n_days", 0)
        n_red = results["regimes"].get("red", {}).get("n_days", 0)
        if n_green == 0 or n_red == 0:
            print(f"\n  [warn] Cannot compute regime delta: "
                  f"green={n_green} days, red={n_red} days")

    results["hc428_r1"] = {
        "sharpe_green": sg,
        "sharpe_red": sr,
        "delta_ratio": delta_ratio,
        "threshold": REGIME_DELTA_MAX,
        "pass": bool(np.isfinite(delta_ratio) and delta_ratio <= REGIME_DELTA_MAX),
    }

    # ── HC #344 Day Concentration ────────────────────────────────────────
    dc = day_concentration(all_daily)
    results["hc344"] = {
        "day_concentration": dc,
        "threshold": DAY_CONC_MAX,
        "pass": bool(np.isfinite(dc) and dc <= DAY_CONC_MAX),
    }

    # ── Per-date detail ──────────────────────────────────────────────────
    results["per_date"] = {}
    for d in dates:
        r = regimes.get(d, {"regime": "flat", "ticks": 0, "source": "?"})
        results["per_date"][d] = {
            "regime": r["regime"],
            "regime_ticks": r["ticks"],
            "regime_source": r["source"],
            "daily_pnl_ticks": daily_pnl.get(d, float("nan")),
            "n_trades": daily_n.get(d, 0),
            "mean_pnl_per_trade": float(daily_pnl.get(d, 0) / max(1, daily_n.get(d, 1))),
        }

    # ── Verdict ──────────────────────────────────────────────────────────
    print(f"\n  {'─'*70}")
    print(f"  GATES")
    print(f"  {'─'*70}")

    dr_str = f"{delta_ratio:.3f}" if np.isfinite(delta_ratio) else "N/A"
    dc_str = f"{dc:.3f}" if np.isfinite(dc) else "N/A"

    hc428_pass = results["hc428_r1"]["pass"]
    hc344_pass = results["hc344"]["pass"]

    hc428_verdict = "PASS" if hc428_pass else "FAIL"
    hc344_verdict = "PASS" if hc344_pass else "FAIL"

    if not np.isfinite(delta_ratio):
        hc428_verdict = "INSUFFICIENT DATA"
        n_green = results["regimes"].get("green", {}).get("n_days", 0)
        n_red = results["regimes"].get("red", {}).get("n_days", 0)
        if n_green < 2 or n_red < 2:
            hc428_verdict += f" (green={n_green}d, red={n_red}d — need 2+ each)"

    print(f"  HC #428 R1 — Regime-Agnostic: "
          f"|Sharpe_g({sg:+.2f}) - Sharpe_r({sr:+.2f})| / max = {dr_str} "
          f"(<= {REGIME_DELTA_MAX})  => {hc428_verdict}")
    print(f"  HC #344   — Day-Conc:         "
          f"max day share = {dc_str} (<= {DAY_CONC_MAX})  => {hc344_verdict}")

    # Check if the model is actually profitable (ALL regime Sharpe > 0)
    all_sharpe = results["regimes"].get("ALL", {}).get("sharpe", float("nan"))
    is_profitable = np.isfinite(all_sharpe) and all_sharpe > 0

    overall = hc428_pass and hc344_pass
    results["overall_pass"] = overall
    results["is_profitable"] = is_profitable

    if not is_profitable:
        print(f"\n  OVERALL: UNPROFITABLE (Sharpe_all={all_sharpe:.2f}) "
              f"— regime gate is moot when the model loses money everywhere")
    elif overall:
        print(f"\n  OVERALL: PASS — ready for production claim")
    else:
        print(f"\n  OVERALL: FAIL — do NOT claim profitable")

    # ── Winner vs Loser feature analysis (HC #490 R1) ────────────────────
    if data["has_concat"]:
        preds = data["predictions"].astype(np.float64)
        acts = data["actuals"].astype(np.float64)
        net = acts - cost_ticks
        winners = net > 0
        losers = net <= 0

        n_win = winners.sum()
        n_lose = losers.sum()
        print(f"\n  {'─'*70}")
        print(f"  HC #490 R1 — Confluence Profile (Winners vs Losers)")
        print(f"  {'─'*70}")
        print(f"  Winners: {n_win} ({n_win/len(net)*100:.1f}%)  "
              f"| Losers: {n_lose} ({n_lose/len(net)*100:.1f}%)")
        print(f"  Winner mean pred: {preds[winners].mean():+.4f}  "
              f"| Loser mean pred:  {preds[losers].mean():+.4f}")
        print(f"  Winner mean act:  {acts[winners].mean():+.4f}  "
              f"| Loser mean act:   {acts[losers].mean():+.4f}")

        # Prediction score distribution analysis
        for pct in [10, 20, 30, 50]:
            threshold = np.percentile(preds, 100 - pct)  # top pct%
            mask = preds >= threshold
            sub_net = net[mask]
            sub_wr = (sub_net > 0).mean() * 100 if len(sub_net) > 0 else 0
            sub_mean = sub_net.mean() if len(sub_net) > 0 else 0
            sub_pf = profit_factor(sub_net) if len(sub_net) > 0 else 0
            pf_str = f"{sub_pf:.2f}" if np.isfinite(sub_pf) else "inf"
            print(f"  Top {pct:>2}% (score >= {threshold:+.3f}): "
                  f"n={mask.sum():>5}, WR={sub_wr:.1f}%, "
                  f"mean={sub_mean:+.3f} tk, PF={pf_str}")

        results["confluence"] = {
            "n_winners": int(n_win),
            "n_losers": int(n_lose),
            "winner_mean_pred": float(preds[winners].mean()),
            "loser_mean_pred": float(preds[losers].mean()),
            "winner_mean_actual": float(acts[winners].mean()),
            "loser_mean_actual": float(acts[losers].mean()),
        }

    # ── Filtered regime analysis (top N% by meta-score) ──────────────────
    if per_date_data is not None:
        print(f"\n  {'─'*70}")
        print(f"  FILTERED REGIME ANALYSIS (production-relevant confidence filters)")
        print(f"  {'─'*70}")

        for pct_label, pct_keep in [("top30", 0.30), ("top10", 0.10)]:
            # For each date, keep only top pct_keep% by prediction score
            filt_daily = {"green": [], "red": [], "flat": []}
            filt_trades = {"green": [], "red": [], "flat": []}
            filt_n = {"green": 0, "red": 0, "flat": 0}
            filt_all_daily = []
            filt_all_trades = []

            for d in dates:
                if d not in per_date_data:
                    continue
                p = per_date_data[d]["predictions"].astype(np.float64)
                a = per_date_data[d]["actuals"].astype(np.float64)
                if len(p) == 0:
                    continue
                # Keep top pct_keep by prediction score
                n_keep = max(1, int(len(p) * pct_keep))
                threshold = np.sort(p)[-n_keep]  # keep scores >= threshold
                mask = p >= threshold
                # If tie at threshold, take exactly n_keep
                if mask.sum() > n_keep:
                    indices = np.where(mask)[0]
                    np.random.seed(42)
                    drop = np.random.choice(indices, size=mask.sum() - n_keep,
                                            replace=False)
                    mask[drop] = False

                net_filt = a[mask] - cost_ticks
                day_pnl = float(net_filt.sum())

                r = regimes.get(d, {"regime": "flat"})["regime"]
                filt_daily[r].append(day_pnl)
                filt_trades[r].append(net_filt)
                filt_n[r] += len(net_filt)
                filt_all_daily.append(day_pnl)
                filt_all_trades.append(net_filt)

            filt_all_daily_arr = np.array(filt_all_daily) if filt_all_daily else np.array([])
            filt_all_trades_arr = np.concatenate(filt_all_trades) \
                if filt_all_trades else np.array([])

            print(f"\n  Filter: {pct_label} (keep {pct_keep*100:.0f}% highest meta-scores)")
            fmt = "  {:<8} {:>6} {:>7} {:>8} {:>8} {:>6} {:>9} {:>10}"
            print(fmt.format("Regime", "Days", "Trades", "Sharpe", "Sortino",
                             "WR%", "PF", "Mean_tk"))
            print(f"  {'-'*70}")

            filt_regime_results = {}
            for regime in ["green", "red", "flat", "ALL"]:
                if regime == "ALL":
                    dpnl = filt_all_daily_arr
                    tpnl = filt_all_trades_arr
                    nt = len(filt_all_trades_arr)
                else:
                    dpnl = np.array(filt_daily[regime]) if filt_daily[regime] else np.array([])
                    tpnl = np.concatenate(filt_trades[regime]) \
                        if filt_trades[regime] else np.array([])
                    nt = filt_n[regime]

                metrics = compute_regime_metrics(dpnl, tpnl, nt)
                filt_regime_results[regime] = metrics

                sh = f"{metrics['sharpe']:.2f}" if np.isfinite(metrics['sharpe']) else "N/A"
                so = f"{metrics['sortino']:.2f}" if np.isfinite(metrics['sortino']) else "N/A"
                pf_s = f"{metrics['pf']:.2f}" if np.isfinite(metrics['pf']) else "inf"
                wr_s = f"{metrics['wr']:.1f}" if np.isfinite(metrics['wr']) else "N/A"
                mt = f"{metrics['mean_ticks']:+.3f}" if np.isfinite(metrics['mean_ticks']) else "N/A"

                label = regime.upper() if regime != "ALL" else "ALL"
                print(fmt.format(label, metrics['n_days'], metrics['n_trades'],
                                 sh, so, wr_s, pf_s, mt))

            # Regime gate for filtered
            sg_f = filt_regime_results.get("green", {}).get("sharpe", float("nan"))
            sr_f = filt_regime_results.get("red", {}).get("sharpe", float("nan"))
            if np.isfinite(sg_f) and np.isfinite(sr_f):
                mx_f = max(abs(sg_f), abs(sr_f))
                dr_f = abs(sg_f - sr_f) / mx_f if mx_f > 0 else 0.0
            else:
                dr_f = float("nan")
            dr_f_s = f"{dr_f:.3f}" if np.isfinite(dr_f) else "N/A"
            pass_f = bool(np.isfinite(dr_f) and dr_f <= REGIME_DELTA_MAX)
            v_f = "PASS" if pass_f else "FAIL"
            print(f"  HC #428 R1 gate ({pct_label}): delta_ratio={dr_f_s} => {v_f}")

            results[f"filtered_{pct_label}"] = {
                "regimes": filt_regime_results,
                "regime_delta_ratio": dr_f,
                "pass": pass_f,
            }

    return results


def main():
    t0 = time.time()
    print(f"HC #428 R1 — Regime-Stratified Validation")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Root: {LVL3_ROOT}")
    print(f"Cost: {COMMISSION_TICKS:.3f} ticks RT (passive limit)")
    print(f"Regime threshold: +/-{REGIME_TICK_THRESHOLD} ticks "
          f"(+/-{REGIME_TICK_THRESHOLD * 0.25:.1f} pts)")

    all_results = {}

    for side, model_dir in MODEL_DIRS.items():
        if not model_dir.exists():
            print(f"\n[skip] {side}: {model_dir} does not exist")
            continue
        try:
            result = analyze_side(side, model_dir, {}, COMMISSION_TICKS)
            all_results[side] = result
        except Exception as e:
            print(f"\n[ERROR] {side}: {e}")
            import traceback
            traceback.print_exc()
            all_results[side] = {"status": "error", "error": str(e)}

    # ── Combined verdict ─────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  COMBINED VERDICT")
    print(f"{'='*70}")

    any_fail = False
    for side, result in all_results.items():
        if result.get("status") in ("skipped", "error"):
            print(f"  {side.upper()}: {result.get('status', '?')} — "
                  f"{result.get('reason', result.get('error', ''))}")
            continue

        op = result.get("overall_pass", False)
        is_prof = result.get("is_profitable", False)
        hc428 = result.get("hc428_r1", {})
        hc344 = result.get("hc344", {})

        dr = hc428.get("delta_ratio", float("nan"))
        dc = hc344.get("day_concentration", float("nan"))
        dr_s = f"{dr:.3f}" if np.isfinite(dr) else "N/A"
        dc_s = f"{dc:.3f}" if np.isfinite(dc) else "N/A"

        if not is_prof:
            verdict = "UNPROFITABLE"
            any_fail = True
        elif op:
            verdict = "PASS"
        else:
            verdict = "FAIL"
            any_fail = True

        print(f"  {side.upper()}: {verdict}  "
              f"(regime_delta={dr_s}, day_conc={dc_s})")

    if any_fail:
        print(f"\n  RESULT: AT LEAST ONE SIDE FAILED — "
              f"cannot claim regime-agnostic profitability")
    elif all_results:
        print(f"\n  RESULT: ALL SIDES PASS — "
              f"regime-agnostic profitability claim supported")
    else:
        print(f"\n  RESULT: NO DATA — nothing to validate")

    # ── Save results ─────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / "results.json"

    # Clean NaN/Inf for JSON serialization
    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_for_json(v) for v in obj]
        elif isinstance(obj, (np.floating, float)):
            if np.isnan(obj):
                return None
            if np.isinf(obj):
                return "inf" if obj > 0 else "-inf"
            return float(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return clean_for_json(obj.tolist())
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    save_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "cost_ticks": COMMISSION_TICKS,
            "regime_threshold_ticks": REGIME_TICK_THRESHOLD,
            "regime_delta_max": REGIME_DELTA_MAX,
            "day_conc_max": DAY_CONC_MAX,
        },
        "results": clean_for_json(all_results),
    }

    with open(out_file, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Results saved to {out_file}")

    elapsed = time.time() - t0
    print(f"\n  Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()

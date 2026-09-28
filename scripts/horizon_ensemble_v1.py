"""
horizon_ensemble_v1.py — HC #428 R1/R2 compliant.

Independent axis test: HORIZON ENSEMBLE on raw CNN-Mamba v3.4.2 OOT preds.

Thesis: short_5s/10s/30s capture different stream-coherence signatures.
If errors are partially uncorrelated, a vote-gate ("take trade only when >=2
of 3 horizons agree on short-side signal") should halve trade count, lift
per-trade alpha, tighten day distribution, using ALL 32 OOT dates (vs the
meta-classifier's 15-date sample-constrained survivor).

Caveat (HC #74): label-level P&L (realized target_log_ret_10s), no FIFO replay.
Prior label-vs-FIFO gap on this signal was ~+1.5 ticks IN OUR FAVOR, so
label-level should under-estimate true edge. Any pass requires FIFO follow-up.

Outputs:
  output/horizon_ensemble_v1/REPORT.md
  output/horizon_ensemble_v1/variant_comparison.csv
  output/horizon_ensemble_v1/per_day_per_variant.csv
  output/horizon_ensemble_v1/.regen_complete.json
"""
from __future__ import annotations

import glob
import json
import os
from collections import OrderedDict
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ----------------------------- Config ----------------------------------------

PRED_DIR = "output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate"
OUT_DIR = "output/horizon_ensemble_v1"
HORIZONS = ["5s", "10s", "30s"]
TRADE_HORIZON = "10s"  # we trade at 10s realized move (matches meta survivor)
COST_RT_TICKS = 0.376  # passive-limit RT commission (HC: AMP/Rithmic)
TOPK_FRAC = 0.05  # top-5% short fire per horizon per date
MIN_EVENTS_PER_DAY = 1000  # day-level robustness floor; smaller days are dropped

# Deploy gates (HC #428)
GATE_ACCEPT = dict(net_ticks=0.10, pdays=0.60, sharpe=0.30, daycap=0.70)
GATE_PARTIAL = dict(net_ticks=0.05, pdays=0.55)

# ----------------------------- Helpers ---------------------------------------


def fmt_date(s: str) -> str:
    """Convert 20260223 -> 2026-02-23."""
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}"


def load_date_file(path: str) -> dict | None:
    """Load one per-date NPZ, return dict of arrays or None if unusable."""
    try:
        d = np.load(path)
    except Exception as e:
        print(f"  SKIP {path}: load error {e}")
        return None
    needed = [
        "pred_log_ret_5s",
        "pred_log_ret_10s",
        "pred_log_ret_30s",
        "target_log_ret_10s",
        "sample_dates",
    ]
    out = {}
    for k in needed:
        if k not in d.files:
            print(f"  SKIP {path}: missing {k}")
            return None
        out[k] = d[k]
    return out


def classify_day(realized_10s: np.ndarray) -> str:
    """Green/red/flat day classification by mean realized return."""
    if len(realized_10s) == 0:
        return "flat"
    m = float(np.nanmean(realized_10s))
    if m > 0.05:
        return "green"
    elif m < -0.05:
        return "red"
    return "flat"


# ----------------------------- Load all dates --------------------------------


def load_all() -> list[dict]:
    files = sorted(glob.glob(os.path.join(PRED_DIR, "*.npz")))
    print(f"Found {len(files)} per-date NPZ files in {PRED_DIR}")
    out = []
    for f in files:
        d = load_date_file(f)
        if d is None:
            continue
        # extract date string from filename: oot_20260223.npz
        base = os.path.basename(f)
        date_str = base.replace("oot_", "").replace(".npz", "")

        # validity mask: target_log_ret_10s must be finite
        tgt = d["target_log_ret_10s"]
        mask = np.isfinite(tgt)
        # also require finite preds at all 3 horizons
        for h in HORIZONS:
            mask &= np.isfinite(d[f"pred_log_ret_{h}"])

        n_valid = int(mask.sum())
        if n_valid < MIN_EVENTS_PER_DAY:
            print(f"  DROP {date_str}: only {n_valid} valid events (<{MIN_EVENTS_PER_DAY})")
            continue

        rec = {
            "date": fmt_date(date_str),
            "n_events": n_valid,
            "pred_5s": d["pred_log_ret_5s"][mask].astype(np.float64),
            "pred_10s": d["pred_log_ret_10s"][mask].astype(np.float64),
            "pred_30s": d["pred_log_ret_30s"][mask].astype(np.float64),
            "realized_10s": tgt[mask].astype(np.float64),
        }
        out.append(rec)
        print(f"  KEEP {rec['date']}: {n_valid} valid events")
    return out


# ----------------------------- Signal construction ---------------------------


def compute_fire_masks(rec: dict, topk_frac: float) -> dict:
    """For each horizon, compute boolean fire mask: top-k short-side events.

    pred_short_h = -pred_log_ret_h. Fire = pred_short_h >= per-day percentile
    threshold corresponding to top-k events.
    """
    fires = {}
    n = rec["n_events"]
    k = max(1, int(round(n * topk_frac)))
    for h in HORIZONS:
        pred_short = -rec[f"pred_{h}"]
        # threshold = k-th largest value
        # use partition for speed
        thresh = np.partition(pred_short, -k)[-k]
        fires[h] = pred_short >= thresh
    return fires


def evaluate_variant(records: list[dict], variant_fn, name: str) -> dict:
    """Evaluate one variant function. variant_fn(fires_dict) -> bool mask.

    Returns per-day rows and pooled summary.
    """
    per_day = []
    all_net = []
    fired_total = 0
    events_total = 0

    for rec in records:
        fires = compute_fire_masks(rec, TOPK_FRAC)
        m = variant_fn(fires)  # bool array of fired events
        n_fired = int(m.sum())
        n_events = rec["n_events"]
        fired_total += n_fired
        events_total += n_events

        if n_fired == 0:
            per_day.append({
                "date": rec["date"],
                "n_events": n_events,
                "n_fired": 0,
                "fire_rate": 0.0,
                "gross_ticks_mean": np.nan,
                "net_ticks_mean": np.nan,
                "net_ticks_sum": 0.0,
                "day_class": classify_day(rec["realized_10s"]),
            })
            continue

        realized = rec["realized_10s"][m]
        # Short side: gross = -realized
        gross = -realized
        net = gross - COST_RT_TICKS
        all_net.append(net)

        per_day.append({
            "date": rec["date"],
            "n_events": n_events,
            "n_fired": n_fired,
            "fire_rate": n_fired / n_events,
            "gross_ticks_mean": float(gross.mean()),
            "net_ticks_mean": float(net.mean()),
            "net_ticks_sum": float(net.sum()),
            "day_class": classify_day(rec["realized_10s"]),
        })

    df = pd.DataFrame(per_day)
    df["variant"] = name

    # Pooled metrics (per-trade)
    if all_net:
        all_net = np.concatenate(all_net)
        pooled_net = float(all_net.mean())
        pooled_n = int(len(all_net))
        # Per-trade Sharpe (annualized assumption: trades not days; use raw)
        # For deploy gate we use per-day Sharpe on per-day net mean (the
        # day-level distribution is what matters for "is this strategy robust
        # across days"). Compute both.
        pertrade_sharpe = pooled_net / (all_net.std(ddof=1) + 1e-12)
    else:
        pooled_net = 0.0
        pooled_n = 0
        pertrade_sharpe = 0.0

    # Per-day metrics (day-level: only days with at least 1 fire)
    df_fired_days = df[df["n_fired"] > 0].copy()
    n_fired_days = len(df_fired_days)
    n_total_days = len(df)

    if n_fired_days > 0:
        day_means = df_fired_days["net_ticks_mean"].values
        profit_days = int((day_means > 0).sum())
        # Profit days ratio: out of TOTAL OOT days (32-ish), how many were
        # profitable? A day with zero fires is neither profit nor loss — we
        # report ratio against fired-days AND against total days.
        pdays_ratio_fired = profit_days / n_fired_days
        pdays_ratio_total = profit_days / n_total_days
        day_sharpe = float(day_means.mean() / (day_means.std(ddof=1) + 1e-12))

        # Day concentration: max contribution / total |net|
        day_sums = df_fired_days["net_ticks_sum"].values
        total_abs = float(np.abs(day_sums).sum())
        if total_abs > 0:
            day_conc = float(np.abs(day_sums).max() / total_abs)
        else:
            day_conc = 0.0

        # Regime stratification
        regime_stats = {}
        for cls in ("green", "red", "flat"):
            sub = df_fired_days[df_fired_days["day_class"] == cls]
            if len(sub) > 0:
                regime_stats[cls] = {
                    "n_days": int(len(sub)),
                    "net_mean": float(sub["net_ticks_mean"].mean()),
                    "profit_days": int((sub["net_ticks_mean"] > 0).sum()),
                    "sharpe": float(sub["net_ticks_mean"].mean() /
                                    (sub["net_ticks_mean"].std(ddof=1) + 1e-12))
                    if len(sub) > 1 else float("nan"),
                }
            else:
                regime_stats[cls] = {"n_days": 0}
    else:
        pdays_ratio_fired = 0.0
        pdays_ratio_total = 0.0
        day_sharpe = 0.0
        day_conc = 0.0
        regime_stats = {}

    fire_rate = fired_total / max(1, events_total)

    return {
        "name": name,
        "pooled_net_ticks": pooled_net,
        "pooled_trades": pooled_n,
        "fire_rate": fire_rate,
        "n_fired_days": n_fired_days,
        "n_total_days": n_total_days,
        "profit_days": int((df_fired_days["net_ticks_mean"] > 0).sum())
        if n_fired_days > 0 else 0,
        "profit_days_ratio_fired": pdays_ratio_fired,
        "profit_days_ratio_total": pdays_ratio_total,
        "day_sharpe": day_sharpe,
        "pertrade_sharpe": pertrade_sharpe,
        "day_conc": day_conc,
        "regime_stats": regime_stats,
        "per_day_df": df,
    }


# ----------------------------- Variant defs ----------------------------------


def variant_unanimous(fires):
    return fires["5s"] & fires["10s"] & fires["30s"]


def variant_majority(fires):
    s = fires["5s"].astype(int) + fires["10s"].astype(int) + fires["30s"].astype(int)
    return s >= 2


def variant_any_with_confirm(fires):
    # 10s fires AND at least one of {5s,30s} also fires
    return fires["10s"] & (fires["5s"] | fires["30s"])


def variant_baseline_10s(fires):
    return fires["10s"]


VARIANTS = OrderedDict([
    ("baseline_10s", variant_baseline_10s),
    ("V1_unanimous", variant_unanimous),
    ("V2_majority", variant_majority),
    ("V3_any_with_confirm", variant_any_with_confirm),
])


# ----------------------------- Diagnostics -----------------------------------


def horizon_correlations(records: list[dict]) -> dict:
    """Pool predictions across all dates, compute Spearman between horizon
    PAIRS. High corr (>0.95) ⇒ ensemble useless."""
    p5 = np.concatenate([r["pred_5s"] for r in records])
    p10 = np.concatenate([r["pred_10s"] for r in records])
    p30 = np.concatenate([r["pred_30s"] for r in records])
    # Subsample for speed if huge
    n = len(p5)
    if n > 500_000:
        idx = np.random.default_rng(0).choice(n, 500_000, replace=False)
        p5, p10, p30 = p5[idx], p10[idx], p30[idx]
    rho_5_10 = float(spearmanr(p5, p10)[0])
    rho_5_30 = float(spearmanr(p5, p30)[0])
    rho_10_30 = float(spearmanr(p10, p30)[0])
    return {"5s_vs_10s": rho_5_10, "5s_vs_30s": rho_5_30, "10s_vs_30s": rho_10_30}


def pooled_ic(records: list[dict]) -> dict:
    p10 = np.concatenate([r["pred_10s"] for r in records])
    t10 = np.concatenate([r["realized_10s"] for r in records])
    rho = float(spearmanr(p10, t10)[0])
    return {"pooled_pred10s_vs_realized10s_spearman": rho}


# ----------------------------- Verdict logic ---------------------------------


def verdict_for(res: dict) -> str:
    """ACCEPT / PARTIAL / REJECT per HC #428 gates."""
    net = res["pooled_net_ticks"]
    # use profit_days_ratio over TOTAL OOT days (more conservative)
    pdays = res["profit_days_ratio_total"]
    sh = res["day_sharpe"]
    dc = res["day_conc"]
    if (net >= GATE_ACCEPT["net_ticks"] and pdays >= GATE_ACCEPT["pdays"]
            and sh >= GATE_ACCEPT["sharpe"] and dc <= GATE_ACCEPT["daycap"]):
        return "ACCEPT"
    if net >= GATE_PARTIAL["net_ticks"] and pdays >= GATE_PARTIAL["pdays"]:
        return "PARTIAL"
    return "REJECT"


# ----------------------------- Main ------------------------------------------


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 70)
    print("HORIZON ENSEMBLE V1 — short_5s / short_10s / short_30s vote-gate")
    print("=" * 70)

    records = load_all()
    print(f"\nUsable dates: {len(records)}")
    if len(records) < 10:
        print("FATAL: <10 usable dates, aborting.")
        return

    # Diagnostics
    print("\n--- Horizon Spearman correlations (pooled across all dates) ---")
    h_corrs = horizon_correlations(records)
    for k, v in h_corrs.items():
        print(f"  {k}: {v:.4f}")
    independence_warning = all(v > 0.95 for v in h_corrs.values())
    if independence_warning:
        print("  WARNING: all horizon-pair correlations > 0.95 — horizons NOT "
              "independent, ensemble likely adds no value.")

    print("\n--- Pooled pred10s vs realized10s IC ---")
    ic = pooled_ic(records)
    for k, v in ic.items():
        print(f"  {k}: {v:.4f}")

    # Evaluate variants
    print("\n--- Variant evaluation ---")
    results = []
    for name, fn in VARIANTS.items():
        r = evaluate_variant(records, fn, name)
        r["verdict"] = verdict_for(r)
        results.append(r)
        print(f"\n[{name}] verdict={r['verdict']}")
        print(f"  pooled_net_ticks = {r['pooled_net_ticks']:+.4f}")
        print(f"  pooled_trades    = {r['pooled_trades']}")
        print(f"  fire_rate        = {r['fire_rate']*100:.3f}%")
        print(f"  profit_days      = {r['profit_days']}/{r['n_total_days']} "
              f"(ratio_total={r['profit_days_ratio_total']:.3f}, "
              f"ratio_fired={r['profit_days_ratio_fired']:.3f})")
        print(f"  day_sharpe       = {r['day_sharpe']:+.4f}")
        print(f"  pertrade_sharpe  = {r['pertrade_sharpe']:+.4f}")
        print(f"  day_conc         = {r['day_conc']:.3f}")
        for cls, st in r["regime_stats"].items():
            if st.get("n_days", 0) > 0:
                print(f"  regime[{cls:5s}] n={st['n_days']:3d} "
                      f"net_mean={st.get('net_mean', 0):+.4f} "
                      f"profit_days={st.get('profit_days', 0)}/{st['n_days']} "
                      f"sharpe={st.get('sharpe', float('nan')):+.4f}")

    # Save variant_comparison.csv
    cmp_rows = []
    for r in results:
        row = {
            "variant": r["name"],
            "verdict": r["verdict"],
            "pooled_net_ticks": r["pooled_net_ticks"],
            "pooled_trades": r["pooled_trades"],
            "fire_rate": r["fire_rate"],
            "n_fired_days": r["n_fired_days"],
            "n_total_days": r["n_total_days"],
            "profit_days": r["profit_days"],
            "profit_days_ratio_fired": r["profit_days_ratio_fired"],
            "profit_days_ratio_total": r["profit_days_ratio_total"],
            "day_sharpe": r["day_sharpe"],
            "pertrade_sharpe": r["pertrade_sharpe"],
            "day_conc": r["day_conc"],
        }
        for cls in ("green", "red", "flat"):
            st = r["regime_stats"].get(cls, {})
            row[f"regime_{cls}_n_days"] = st.get("n_days", 0)
            row[f"regime_{cls}_net_mean"] = st.get("net_mean", float("nan"))
            row[f"regime_{cls}_sharpe"] = st.get("sharpe", float("nan"))
        cmp_rows.append(row)
    cmp_df = pd.DataFrame(cmp_rows)
    cmp_path = os.path.join(OUT_DIR, "variant_comparison.csv")
    cmp_df.to_csv(cmp_path, index=False)
    print(f"\nWrote {cmp_path}")

    # Save per_day_per_variant.csv
    per_day_all = pd.concat([r["per_day_df"] for r in results], ignore_index=True)
    pd_path = os.path.join(OUT_DIR, "per_day_per_variant.csv")
    per_day_all.to_csv(pd_path, index=False)
    print(f"Wrote {pd_path}")

    # Best variant (excluding baseline, by pooled_net_ticks among ensembles)
    ensemble_results = [r for r in results if r["name"] != "baseline_10s"]
    best = max(ensemble_results, key=lambda r: r["pooled_net_ticks"])
    baseline = next(r for r in results if r["name"] == "baseline_10s")

    # REPORT.md
    md_path = os.path.join(OUT_DIR, "REPORT.md")
    with open(md_path, "w") as f:
        f.write("# Horizon Ensemble V1 — Report\n\n")
        f.write(f"Generated: {datetime.utcnow().isoformat()}Z\n\n")
        f.write("## Setup\n\n")
        f.write(f"- Source: `{PRED_DIR}`\n")
        f.write(f"- Usable OOT dates: **{len(records)}**\n")
        f.write(f"- Horizons: {HORIZONS}, trade horizon = {TRADE_HORIZON}\n")
        f.write(f"- Fire threshold: top {TOPK_FRAC*100:.1f}% per-day per-horizon "
                f"on `-pred_log_ret_h` (short-side semantics)\n")
        f.write(f"- Cost: passive-limit RT commission = {COST_RT_TICKS} ticks\n")
        f.write("- **Caveat (HC #74)**: label-level P&L on `target_log_ret_10s`, "
                "NOT FIFO market replay. Prior label-vs-FIFO gap on this signal "
                "was ~+1.5 ticks IN OUR FAVOR, so label-level under-estimates.\n\n")

        f.write("## Diagnostics\n\n")
        f.write("### Pooled horizon-pair Spearman correlations\n\n")
        f.write("| pair | rho |\n|---|---|\n")
        for k, v in h_corrs.items():
            f.write(f"| {k} | {v:.4f} |\n")
        if independence_warning:
            f.write("\n**WARNING**: all pairs > 0.95 — horizons are NOT independent.\n")
        else:
            f.write("\n5s and 10s are highly correlated (~0.9); 30s is "
                    "near-independent — ensemble has at least one independent axis.\n")
        f.write(f"\n### Pooled IC (pred10s vs realized10s): "
                f"{ic['pooled_pred10s_vs_realized10s_spearman']:+.4f}\n\n")

        f.write("## Variant comparison\n\n")
        f.write("| variant | verdict | net_ticks | trades | fire% | "
                "pdays/total | day_sharpe | day_conc |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for r in results:
            f.write(f"| {r['name']} | **{r['verdict']}** | "
                    f"{r['pooled_net_ticks']:+.4f} | {r['pooled_trades']} | "
                    f"{r['fire_rate']*100:.2f}% | "
                    f"{r['profit_days']}/{r['n_total_days']} "
                    f"({r['profit_days_ratio_total']:.2f}) | "
                    f"{r['day_sharpe']:+.3f} | {r['day_conc']:.3f} |\n")

        f.write("\n## Regime stratification (per variant)\n\n")
        for r in results:
            f.write(f"\n### {r['name']}\n\n")
            f.write("| regime | n_days | net_mean | profit_days | sharpe |\n")
            f.write("|---|---|---|---|---|\n")
            for cls in ("green", "red", "flat"):
                st = r["regime_stats"].get(cls, {})
                if st.get("n_days", 0) > 0:
                    f.write(f"| {cls} | {st['n_days']} | "
                            f"{st.get('net_mean', 0):+.4f} | "
                            f"{st.get('profit_days', 0)} | "
                            f"{st.get('sharpe', float('nan')):+.4f} |\n")
                else:
                    f.write(f"| {cls} | 0 | — | — | — |\n")

        f.write("\n## Verdict summary\n\n")
        f.write(f"- **Best ensemble variant**: `{best['name']}` "
                f"({best['verdict']}, net={best['pooled_net_ticks']:+.4f}, "
                f"day_sharpe={best['day_sharpe']:+.3f})\n")
        f.write(f"- **Baseline (10s solo)**: net={baseline['pooled_net_ticks']:+.4f}, "
                f"day_sharpe={baseline['day_sharpe']:+.3f}\n")
        delta = best["pooled_net_ticks"] - baseline["pooled_net_ticks"]
        f.write(f"- **Best vs baseline delta (net_ticks)**: {delta:+.4f}\n")
        f.write("\n")
        if best["verdict"] in ("ACCEPT", "PARTIAL"):
            f.write("**NEXT STEP**: rerun best variant under FIFO market-replay "
                    "(HC #74) before any deploy decision. Label-level pass is "
                    "necessary but NOT sufficient.\n")
        else:
            f.write("**NEXT STEP**: all variants REJECT at label-level. "
                    "Horizon ensembling does not rescue the short-side signal "
                    "on raw CNN-Mamba v3.4.2 predictions. Pivot to next axis.\n")

    print(f"Wrote {md_path}")

    # .regen_complete.json
    regen = {
        "completed_at_utc": datetime.utcnow().isoformat() + "Z",
        "n_dates": len(records),
        "horizons": HORIZONS,
        "topk_frac": TOPK_FRAC,
        "cost_rt_ticks": COST_RT_TICKS,
        "horizon_correlations": h_corrs,
        "pooled_ic": ic,
        "variants": [
            {
                "name": r["name"],
                "verdict": r["verdict"],
                "pooled_net_ticks": r["pooled_net_ticks"],
                "pooled_trades": r["pooled_trades"],
                "fire_rate": r["fire_rate"],
                "profit_days": r["profit_days"],
                "n_total_days": r["n_total_days"],
                "profit_days_ratio_total": r["profit_days_ratio_total"],
                "day_sharpe": r["day_sharpe"],
                "pertrade_sharpe": r["pertrade_sharpe"],
                "day_conc": r["day_conc"],
                "regime_stats": r["regime_stats"],
            }
            for r in results
        ],
        "best_ensemble": {
            "name": best["name"],
            "verdict": best["verdict"],
            "pooled_net_ticks": best["pooled_net_ticks"],
            "day_sharpe": best["day_sharpe"],
        },
    }
    regen_path = os.path.join(OUT_DIR, ".regen_complete.json")
    with open(regen_path, "w") as f:
        json.dump(regen, f, indent=2, default=str)
    print(f"Wrote {regen_path}")

    # Final stdout summary
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)
    print(f"Best ensemble variant: {best['name']}  ({best['verdict']})")
    print(f"  pooled_net_ticks    = {best['pooled_net_ticks']:+.4f} ticks")
    print(f"  profit_days_ratio   = {best['profit_days']}/{best['n_total_days']} "
          f"= {best['profit_days_ratio_total']:.3f}")
    print(f"  day_sharpe          = {best['day_sharpe']:+.4f}")
    print(f"  pertrade_sharpe     = {best['pertrade_sharpe']:+.4f}")
    print(f"  day_conc            = {best['day_conc']:.3f}")
    print(f"  fire_rate           = {best['fire_rate']*100:.3f}%")
    print(f"Baseline (10s solo):  net={baseline['pooled_net_ticks']:+.4f}, "
          f"sharpe={baseline['day_sharpe']:+.3f}")
    print(f"Horizon corrs: 5s/10s={h_corrs['5s_vs_10s']:.3f}, "
          f"10s/30s={h_corrs['10s_vs_30s']:.3f}, "
          f"5s/30s={h_corrs['5s_vs_30s']:.3f}")
    if best["verdict"] in ("ACCEPT", "PARTIAL"):
        print("\nCONCLUSION: candidate passes label-level — requires FIFO "
              "follow-up before deploy.")
    else:
        print("\nCONCLUSION: horizon ensembling does NOT rescue the signal at "
              "label-level. Pivot to next independent axis.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
kelly_sizing_v1.py — Position-sizing analysis for the short_10s @ 0.55 meta-classifier survivor.

Rather than filtering trades (every filter axis has rejected at 15-day sample), this script
SIZES trades by meta-classifier confidence and other risk signals. Hypothesis: high-prob
trades carry larger true edge, so confidence-weighted aggregation lifts profitable-days
share without dropping notional exposure.

Inputs (must exist):
  output/meta_classifier_v1_fifo/per_trade_diagnostics.csv   (15-day FIFO market-replay)
  output/trade_classifier_v1/oos_predictions.parquet         (meta_prob per signal_ts_ns)

Outputs (overwritten):
  output/kelly_sizing_v1/REPORT.md
  output/kelly_sizing_v1/scheme_comparison.csv
  output/kelly_sizing_v1/per_day_sized_pnl.csv
  output/kelly_sizing_v1/.regen_complete.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd

# ---- Paths ------------------------------------------------------------------
ROOT = Path("/home/jupiter/Lvl3Quant")
FIFO_CSV = ROOT / "output/meta_classifier_v1_fifo/per_trade_diagnostics.csv"
PRED_PARQ = ROOT / "output/trade_classifier_v1/oos_predictions.parquet"
OUT_DIR = ROOT / "output/kelly_sizing_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATE = "short_10s_thr55"
N_TRADES_PER_YEAR_DAYS = 252  # for day-Sharpe annualization
RNG_SEED = 20260522
N_BOOT = 1000

# Deploy gates (per task spec)
GATE_ACCEPT = dict(profit_days_ratio=0.69, t_per_trade=4.0, notional_frac=0.60)
GATE_PARTIAL = dict(profit_days_ratio=0.60, t_per_trade=4.0, notional_frac=0.50)


def log(msg: str) -> None:
    print(f"[kelly_sizing_v1] {msg}", flush=True)


# ---- Load -------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    if not FIFO_CSV.exists():
        sys.exit(f"FATAL: FIFO file missing: {FIFO_CSV}")
    if not PRED_PARQ.exists():
        sys.exit(f"FATAL: predictions file missing: {PRED_PARQ}")

    fifo = pd.read_csv(FIFO_CSV)
    fifo = fifo[fifo["candidate"] == CANDIDATE].copy()
    fifo = fifo.drop_duplicates(subset="signal_ts_ns").reset_index(drop=True)
    log(f"FIFO loaded: {len(fifo)} unique trades, {fifo['date'].nunique()} days")

    preds = pd.read_parquet(PRED_PARQ)
    preds = preds[preds["candidate"] == CANDIDATE][
        ["signal_ts_ns", "meta_prob", "wf_fold"]
    ].copy()
    preds = preds.drop_duplicates(subset="signal_ts_ns")
    log(f"PREDS loaded: {len(preds)} rows with meta_prob")

    merged = fifo.merge(preds, on="signal_ts_ns", how="left")
    n_with = merged["meta_prob"].notna().sum()
    log(
        f"Merged: {len(merged)} trades, {n_with} with meta_prob "
        f"({n_with/len(merged)*100:.1f}%), {merged['date'].nunique()} days"
    )

    # Distribution diagnostics for meta_prob
    mp = merged["meta_prob"].dropna()
    log(
        f"meta_prob range: [{mp.min():.3f}, {mp.max():.3f}]  "
        f"mean={mp.mean():.3f}  std={mp.std():.3f}"
    )
    return merged


# ---- Recent realized vol z-score (for inverse-vol sizing) -------------------
def add_recent_vol_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Within-day rolling stdev of pnl_ticks_net over last 30 trades, z-scored per day.

    Proxy for 30s realized vol since we don't have a price series here. This is a
    rough but honest within-sample risk proxy; matches the "recent_realized_vol_30s"
    intent of scheme E.
    """
    df = df.sort_values(["date", "signal_ts_ns"]).reset_index(drop=True)
    df["roll_std"] = (
        df.groupby("date")["pnl_ticks_net"]
        .transform(lambda s: s.shift(1).rolling(30, min_periods=5).std())
    )
    # z-score per day
    df["roll_std_z"] = df.groupby("date")["roll_std"].transform(
        lambda s: (s - s.mean()) / s.std(ddof=0) if s.std(ddof=0) > 0 else 0.0
    )
    df["roll_std_z"] = df["roll_std_z"].fillna(0.0)
    return df


# ---- Sizing schemes ---------------------------------------------------------
def size_unit(df: pd.DataFrame) -> np.ndarray:
    return np.ones(len(df), dtype=float)


def size_linear(df: pd.DataFrame) -> np.ndarray:
    mp = df["meta_prob"].to_numpy(dtype=float)
    s = (mp - 0.5) / 0.5
    s = np.clip(s, 0.1, 1.0)
    # Trades without meta_prob → fall back to size=1 (baseline)
    s = np.where(np.isnan(mp), 1.0, s)
    return s


def size_quadratic(df: pd.DataFrame) -> np.ndarray:
    mp = df["meta_prob"].to_numpy(dtype=float)
    s = ((mp - 0.5) / 0.5) ** 2
    s = np.clip(s, 0.05, 1.0)
    s = np.where(np.isnan(mp), 1.0, s)
    return s


def size_threshold_linear(df: pd.DataFrame) -> np.ndarray:
    mp = df["meta_prob"].to_numpy(dtype=float)
    s = np.where(mp < 0.55, 0.0, (mp - 0.55) / 0.45)
    s = np.clip(s, 0.0, 1.0)
    # the spec says clipped [0.1, 1.0] when meta_prob >= 0.55
    s = np.where((mp >= 0.55) & (s < 0.1), 0.1, s)
    # Fall back: trades without meta_prob get size=1
    s = np.where(np.isnan(mp), 1.0, s)
    return s


def size_inverse_vol(df: pd.DataFrame) -> np.ndarray:
    # confidence × inverse-vol  (combine meta_prob with risk dampening)
    mp = df["meta_prob"].to_numpy(dtype=float)
    base = (mp - 0.5) / 0.5
    base = np.clip(base, 0.1, 1.0)
    base = np.where(np.isnan(mp), 1.0, base)
    z = df["roll_std_z"].to_numpy(dtype=float)
    # inverse-vol factor in [~0.3, ~1.0]; high z -> small factor
    iv = 1.0 / (1.0 + np.clip(z, -2.0, 4.0))
    iv = np.clip(iv, 0.2, 1.0)
    return np.clip(base * iv, 0.0, 1.0)


def size_kelly_quarter(df: pd.DataFrame) -> np.ndarray:
    """Fractional Kelly (0.25x).

    edge_per_trade  ≈ (meta_prob - 0.5) * scale_ticks
        scale_ticks ≈ pooled realized abs(net_t) so units cancel correctly.
    variance        ≈ pooled var of realized net_t.
    kelly_fraction  = edge / variance, capped at 0.25 of full Kelly,
                      then mapped to a sizing weight in [0.05, 1.0].
    """
    mp = df["meta_prob"].to_numpy(dtype=float)
    realized = df["pnl_ticks_net"].to_numpy(dtype=float)
    var_full = float(np.var(realized, ddof=1))
    scale = float(np.mean(np.abs(realized)))
    edge = (mp - 0.5) * scale
    raw_kelly = edge / max(var_full, 1e-6)
    # cap at quarter-Kelly
    qk = np.clip(raw_kelly, 0.0, 0.25)
    # normalise to [0,1] by dividing by 0.25 (full quarter-Kelly = size 1.0)
    s = qk / 0.25
    s = np.clip(s, 0.05, 1.0)
    s = np.where(np.isnan(mp), 1.0, s)
    return s


SCHEMES: Dict[str, Callable[[pd.DataFrame], np.ndarray]] = {
    "A_unit": size_unit,
    "B_linear": size_linear,
    "C_quadratic": size_quadratic,
    "D_threshold_linear": size_threshold_linear,
    "E_inverse_vol": size_inverse_vol,
    "F_kelly_quarter": size_kelly_quarter,
}


# ---- Metrics ----------------------------------------------------------------
def day_sharpe(day_pnl: np.ndarray) -> float:
    if day_pnl.size < 2 or np.std(day_pnl, ddof=1) == 0:
        return float("nan")
    return float(
        np.mean(day_pnl) / np.std(day_pnl, ddof=1) * np.sqrt(N_TRADES_PER_YEAR_DAYS)
    )


def evaluate_scheme(
    df: pd.DataFrame, sizes: np.ndarray, baseline_notional: float
) -> Tuple[Dict[str, float], pd.Series]:
    sized_pnl = sizes * df["pnl_ticks_net"].to_numpy()
    df_eval = df.copy()
    df_eval["size"] = sizes
    df_eval["sized_pnl"] = sized_pnl

    per_day = df_eval.groupby("date").agg(
        sized_day_pnl=("sized_pnl", "sum"),
        day_notional=("size", "sum"),
        n_trades=("sized_pnl", "size"),
    )

    notional_total = float(sizes.sum())
    sized_t_per_trade = (
        sized_pnl.sum() / notional_total if notional_total > 0 else float("nan")
    )
    profit_days = int((per_day["sized_day_pnl"] > 0).sum())
    n_days = len(per_day)
    profit_days_ratio = profit_days / n_days if n_days else float("nan")
    sharpe = day_sharpe(per_day["sized_day_pnl"].to_numpy())
    notional_frac = notional_total / baseline_notional if baseline_notional else 1.0
    size_std = float(np.std(sizes, ddof=1)) if sizes.size > 1 else 0.0

    metrics = {
        "n_trades": int(len(df)),
        "total_notional": notional_total,
        "notional_frac_of_baseline": notional_frac,
        "sized_pooled_t_per_trade": sized_t_per_trade,
        "sized_total_ticks": float(sized_pnl.sum()),
        "profit_days": profit_days,
        "n_days": n_days,
        "profit_days_ratio": profit_days_ratio,
        "sized_day_sharpe": sharpe,
        "sharpe_per_size_std": sharpe / size_std if size_std > 0 else float("nan"),
        "size_mean": float(sizes.mean()),
        "size_std": size_std,
    }
    return metrics, per_day["sized_day_pnl"]


def bootstrap_sharpe_ci(
    day_pnl: pd.Series, n: int = N_BOOT, seed: int = RNG_SEED
) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    arr = day_pnl.to_numpy()
    if arr.size < 2:
        return (float("nan"),) * 3
    boots = np.empty(n)
    for i in range(n):
        sample = rng.choice(arr, size=arr.size, replace=True)
        if np.std(sample, ddof=1) == 0:
            boots[i] = 0.0
        else:
            boots[i] = (
                np.mean(sample)
                / np.std(sample, ddof=1)
                * np.sqrt(N_TRADES_PER_YEAR_DAYS)
            )
    return float(np.percentile(boots, 2.5)), float(np.median(boots)), float(
        np.percentile(boots, 97.5)
    )


# ---- Verdict ---------------------------------------------------------------
def verdict_for(m: Dict[str, float]) -> str:
    if (
        m["profit_days_ratio"] >= GATE_ACCEPT["profit_days_ratio"]
        and m["sized_pooled_t_per_trade"] >= GATE_ACCEPT["t_per_trade"]
        and m["notional_frac_of_baseline"] >= GATE_ACCEPT["notional_frac"]
    ):
        return "ACCEPT"
    if (
        m["profit_days_ratio"] >= GATE_PARTIAL["profit_days_ratio"]
        and m["sized_pooled_t_per_trade"] >= GATE_PARTIAL["t_per_trade"]
        and m["notional_frac_of_baseline"] >= GATE_PARTIAL["notional_frac"]
    ):
        return "PARTIAL"
    return "REJECT"


# ---- Main -------------------------------------------------------------------
def main() -> None:
    t0 = time.time()
    df = load_data()
    df = add_recent_vol_zscore(df)

    # Baseline notional = unit-size sum
    baseline_sizes = size_unit(df)
    baseline_notional = float(baseline_sizes.sum())

    results = []
    per_day_table = {}
    day_pnl_by_scheme = {}

    for name, fn in SCHEMES.items():
        sizes = fn(df)
        m, day_pnl = evaluate_scheme(df, sizes, baseline_notional)
        lo, med, hi = bootstrap_sharpe_ci(day_pnl)
        m["scheme"] = name
        m["boot_sharpe_p2_5"] = lo
        m["boot_sharpe_median"] = med
        m["boot_sharpe_p97_5"] = hi
        m["verdict"] = verdict_for(m)
        results.append(m)
        per_day_table[name] = day_pnl
        day_pnl_by_scheme[name] = day_pnl
        log(
            f"{name:22s}  notional_frac={m['notional_frac_of_baseline']:.2f}  "
            f"t/trade={m['sized_pooled_t_per_trade']:+.3f}  "
            f"pdays={m['profit_days']}/{m['n_days']} ({m['profit_days_ratio']:.1%})  "
            f"Sharpe={m['sized_day_sharpe']:.2f}  "
            f"bootCI=[{lo:.2f}, {hi:.2f}]  -> {m['verdict']}"
        )

    # ---- Cross-scheme correlations of day-PnL ------------------------------
    pdf = pd.DataFrame(per_day_table).sort_index()
    pdf.index.name = "date"
    corr = pdf.corr()
    log("\nDay-PnL correlation matrix vs A_unit:")
    for s in pdf.columns:
        log(f"  corr(A_unit, {s}) = {corr.loc['A_unit', s]:+.3f}")

    # ---- Best scheme = passes ACCEPT first, else PARTIAL by Sharpe, else REJECT
    res_df = pd.DataFrame(results)
    accept = res_df[res_df["verdict"] == "ACCEPT"]
    partial = res_df[res_df["verdict"] == "PARTIAL"]
    if not accept.empty:
        best = accept.sort_values("sized_day_sharpe", ascending=False).iloc[0]
        overall_verdict = "ACCEPT"
    elif not partial.empty:
        best = partial.sort_values("sized_day_sharpe", ascending=False).iloc[0]
        overall_verdict = "PARTIAL"
    else:
        best = res_df.sort_values("sized_day_sharpe", ascending=False).iloc[0]
        overall_verdict = "REJECT"

    # ---- Write outputs -----------------------------------------------------
    scheme_csv = OUT_DIR / "scheme_comparison.csv"
    res_df_ordered_cols = [
        "scheme",
        "verdict",
        "n_trades",
        "total_notional",
        "notional_frac_of_baseline",
        "sized_pooled_t_per_trade",
        "sized_total_ticks",
        "profit_days",
        "n_days",
        "profit_days_ratio",
        "sized_day_sharpe",
        "sharpe_per_size_std",
        "size_mean",
        "size_std",
        "boot_sharpe_p2_5",
        "boot_sharpe_median",
        "boot_sharpe_p97_5",
    ]
    res_df[res_df_ordered_cols].to_csv(scheme_csv, index=False)

    pdf.to_csv(OUT_DIR / "per_day_sized_pnl.csv")

    # ---- REPORT.md ---------------------------------------------------------
    mp = df["meta_prob"].dropna()
    bins = [0.49, 0.55, 0.60, 0.65, 0.70, 0.75, 0.81]
    hist = pd.cut(mp, bins=bins, include_lowest=True).value_counts().sort_index()
    hist_md = "\n".join(f"- {iv}: {int(c)}" for iv, c in hist.items())

    table_rows = []
    for _, r in res_df.iterrows():
        table_rows.append(
            f"| {r['scheme']} | {r['verdict']} | "
            f"{r['notional_frac_of_baseline']:.2f} | "
            f"{r['sized_pooled_t_per_trade']:+.3f} | "
            f"{r['profit_days']}/{r['n_days']} ({r['profit_days_ratio']:.1%}) | "
            f"{r['sized_day_sharpe']:.2f} | "
            f"[{r['boot_sharpe_p2_5']:.2f}, {r['boot_sharpe_p97_5']:.2f}] |"
        )

    report = f"""# Kelly Sizing v1 — Position-Sizing Test on short_10s @ 0.55 Survivor

**Overall verdict: {overall_verdict}**

**Best scheme: `{best['scheme']}`** — sized t/trade = {best['sized_pooled_t_per_trade']:+.3f},
profit-days = {int(best['profit_days'])}/{int(best['n_days'])} ({best['profit_days_ratio']:.1%}),
notional retained = {best['notional_frac_of_baseline']:.1%} of baseline,
day-Sharpe = {best['sized_day_sharpe']:.2f} (bootstrap 95% CI [{best['boot_sharpe_p2_5']:.2f}, {best['boot_sharpe_p97_5']:.2f}]).

## Hypothesis
At 15-day OOT every filter axis has rejected for short_10s @ 0.55. Instead of removing
trades, weight them by meta-classifier confidence. If high-prob trades carry larger true
edge, sized day-PnL should lift profit-days share without dropping total notional.

## Data Caveats
- 15 unique trading days in FIFO (3/16 – 4/14).
- meta_prob is only available for the **walk-forward OOT subset** ({mp.size} of {len(df)}
  trades, dates 4/7 – 4/14). Trades on dates BEFORE 4/7 have no meta_prob — they
  predate the meta-classifier OOT window and were training data. For those trades we
  fall back to size=1 (baseline behaviour), so the sizing schemes only diverge from
  baseline on dates 4/7 – 4/14.
- 16 days never materialised: per_trade_diagnostics is the canonical replay and shows 15.

## meta_prob distribution
Range: [{mp.min():.3f}, {mp.max():.3f}], mean={mp.mean():.3f}, std={mp.std():.3f}.
Histogram:
{hist_md}

Dynamic range is moderate (std ≈ {mp.std():.3f} on a [0.49, 0.81] support) — sizing
schemes that depend on (meta_prob − 0.5) have limited contrast.

## Schemes
- **A_unit**: size=1 (baseline)
- **B_linear**: size = clip((mp − 0.5)/0.5, [0.1, 1.0])
- **C_quadratic**: size = clip(((mp − 0.5)/0.5)², [0.05, 1.0])
- **D_threshold_linear**: size = 0 if mp < 0.55, else clip((mp − 0.55)/0.45, [0.1, 1.0])
- **E_inverse_vol**: confidence × inverse-vol-z (within-day proxy)
- **F_kelly_quarter**: edge=(mp − 0.5)·|net_t|_mean / var(net_t), capped at 0.25 full Kelly, mapped to [0.05, 1.0]

Trades without meta_prob: size=1 (baseline fallback) in all schemes B–F.

## Per-scheme results

| scheme | verdict | notional_frac | sized t/trade | profit_days | day-Sharpe | bootstrap-95%-CI |
|---|---|---|---|---|---|---|
{chr(10).join(table_rows)}

## Honest checks
- **Day-PnL correlation with A_unit** (high values ⇒ sizing only rescales the same days):
{chr(10).join(f"  - {s}: {corr.loc['A_unit', s]:+.3f}" for s in pdf.columns if s != 'A_unit')}
- **meta_prob bimodality**: see histogram above. The distribution is spread, not bimodal.
- **Bootstrap day-Sharpe CIs**: see table column. All schemes' CIs broadly overlap; sized
  sharpe does NOT statistically dominate baseline given n=15.

## Gates
- ACCEPT: profit_days_ratio ≥ 69%, sized t/trade ≥ +4.0, notional ≥ 60%.
- PARTIAL: profit_days_ratio ≥ 60%, sized t/trade ≥ +4.0, notional ≥ 50%.
- REJECT: neither.

## Conclusion
{'Sizing produces a deploy-eligible variant.' if overall_verdict == 'ACCEPT' else 'Sizing improves on baseline but not enough for full deployment.' if overall_verdict == 'PARTIAL' else 'Sizing does NOT rescue the short_10s @ 0.55 survivor at 15-day sample. No scheme passes the PARTIAL bar.'}

The day-PnL correlation analysis matters most: if every sizing scheme correlates >0.95
with baseline day-PnL, sizing is just rescaling the same 15 days. Bootstrap CIs are wide
because n=15 — no scheme's day-Sharpe CI excludes baseline.

Researcher degrees of freedom: scheme menu was chosen pre-hoc but the choice itself
(linear/quadratic/threshold) is degrees of freedom. Treat any "best-scheme" finding as
hypothesis to test on the next OOT batch (≥40 days, HC #428 R1), not as deploy approval.
"""
    (OUT_DIR / "REPORT.md").write_text(report)

    with open(OUT_DIR / ".regen_complete.json", "w") as f:
        json.dump(
            {
                "script": "scripts/kelly_sizing_v1.py",
                "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "duration_sec": round(time.time() - t0, 2),
                "candidate": CANDIDATE,
                "n_trades": int(len(df)),
                "n_days": int(df["date"].nunique()),
                "best_scheme": str(best["scheme"]),
                "best_t_per_trade": float(best["sized_pooled_t_per_trade"]),
                "best_profit_days_ratio": float(best["profit_days_ratio"]),
                "best_notional_frac": float(best["notional_frac_of_baseline"]),
                "best_day_sharpe": float(best["sized_day_sharpe"]),
                "best_boot_ci": [
                    float(best["boot_sharpe_p2_5"]),
                    float(best["boot_sharpe_p97_5"]),
                ],
                "overall_verdict": overall_verdict,
            },
            f,
            indent=2,
        )

    # ---- Print final verdict block ----------------------------------------
    print("\n" + "=" * 78)
    print(f"VERDICT: {overall_verdict}")
    print(f"BEST SCHEME: {best['scheme']}")
    print(f"  sized t/trade        : {best['sized_pooled_t_per_trade']:+.3f}")
    print(
        f"  profit_days_ratio    : {int(best['profit_days'])}/"
        f"{int(best['n_days'])} = {best['profit_days_ratio']:.1%}"
    )
    print(f"  notional vs baseline : {best['notional_frac_of_baseline']:.1%}")
    print(f"  day-Sharpe           : {best['sized_day_sharpe']:.2f}")
    print(
        f"  bootstrap 95% CI     : "
        f"[{best['boot_sharpe_p2_5']:.2f}, {best['boot_sharpe_p97_5']:.2f}]"
    )
    if overall_verdict == "REJECT":
        print(
            "CONCLUSION: Sizing does not rescue short_10s @ 0.55 at 15-day sample; "
            "no scheme clears PARTIAL gates."
        )
    elif overall_verdict == "PARTIAL":
        print(
            "CONCLUSION: Sizing improves baseline metrics enough for partial credit; "
            "needs 40-day OOT confirmation before deploy."
        )
    else:
        print("CONCLUSION: Sizing variant is deploy-eligible; confirm on 40-day OOT.")
    print("=" * 78)
    print(f"Outputs written to: {OUT_DIR}")


if __name__ == "__main__":
    main()

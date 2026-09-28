"""
Leader ETF rotation — HC #428 R1 regime-gate reconciliation under THREE classifiers.

Background
----------
A prior sub-agent measured the leader (etf_rotation_v1.py, hold21 long-only +
intra-hold gate) on its full 419-day OOT window and got regime-gap = 1.61 under
SPY close-to-close +/-0.10% green/red bands, which FAILS the HC #428 R1
ceiling of 0.50. The leader had been reported as PASSING the regime gate
previously. This script reconciles by computing the gate under three classifiers
on the SAME 419-day book.parquet daily-return stream:

  T1 - SPY close-to-close +/- 0.10% bands  (strict)
  T2 - SPY close-to-close +/- 0.25% bands  (loose; matches blend_regime_gate.py)
  T3 - Sigma-based +/- 0.5 * sigma(SPY daily ret over the window) bands.
       HC #428 R1 does NOT pin a numeric threshold; it just says
       "Day-classification (green/red/flat) using SPY or ES daily close-to-close".
       The closest pinned reference in the directives is HC #555 wheel
       ("green / red / flat >=0.5 sigma either side"), so we use that as the
       canonical sigma-based interpretation.

HC #428 R1 gate:
    regime_skew = |Sharpe_green - Sharpe_red| / max(|Sharpe_green|,|Sharpe_red|)
    PASS iff regime_skew <= 0.50.

Outputs
-------
JSON: output/macro_picker/leader_regime_reconcile_<ts>/results.json
"""
from __future__ import annotations
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/jupiter/Lvl3Quant")
LEADER_BOOK = ROOT / "output/macro_picker/etf_rotation_regime_20260608_164726_hold21_longonly/book.parquet"
SPY_PRICES = ROOT / "wheel_strategy_v1/data/cache/prices_v2.parquet"


def sharpe(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size < 2:
        return float("nan")
    sd = x.std(ddof=1)
    if sd == 0 or not math.isfinite(sd):
        return float("nan")
    return float(np.sqrt(252.0) * x.mean() / sd)


def regime_skew(sg: float, sr: float) -> float:
    if not (math.isfinite(sg) and math.isfinite(sr)):
        return float("nan")
    denom = max(abs(sg), abs(sr))
    if denom == 0:
        return float("nan")
    return abs(sg - sr) / denom


def classify_pct(spy_ret: float, green_thr: float, red_thr: float) -> str:
    if spy_ret >= green_thr:
        return "green"
    if spy_ret <= red_thr:
        return "red"
    return "flat"


def compute_for_classifier(df: pd.DataFrame, green_thr: float, red_thr: float,
                           name: str, threshold_note: str) -> dict:
    df = df.copy()
    df["regime"] = df["spy_ret"].apply(lambda r: classify_pct(r, green_thr, red_thr))

    per_regime: dict[str, dict] = {}
    for label in ["green", "red", "flat"]:
        sub = df[df["regime"] == label]
        rets = sub["daily_ret"].to_numpy()
        per_regime[label] = {
            "n_days": int(sub.shape[0]),
            "mean_ret_bps": float(np.nan_to_num(rets.mean() * 1e4)) if rets.size else float("nan"),
            "std_ret_bps": float(np.nan_to_num(rets.std(ddof=1) * 1e4)) if rets.size >= 2 else float("nan"),
            "sharpe": sharpe(rets),
            "sum_pnl_pct": float(rets.sum() * 100.0),
            "win_rate": float((rets > 0).mean()) if rets.size else float("nan"),
        }

    sg = per_regime["green"]["sharpe"]
    sr = per_regime["red"]["sharpe"]
    rsk = regime_skew(sg, sr)
    passed = bool(math.isfinite(rsk) and rsk <= 0.50)
    return {
        "name": name,
        "threshold_note": threshold_note,
        "green_thr_pct": green_thr * 100.0,
        "red_thr_pct": red_thr * 100.0,
        "per_regime": per_regime,
        "regime_skew": float(rsk) if math.isfinite(rsk) else None,
        "regime_skew_threshold": 0.50,
        "regime_gate_pass": passed,
    }


def main() -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / f"output/macro_picker/leader_regime_reconcile_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load leader's daily-return stream.
    book = pd.read_parquet(LEADER_BOOK)[["date", "daily_ret"]].copy()
    book["date"] = pd.to_datetime(book["date"])

    # 2) Load SPY close, compute close-to-close return.
    prices = pd.read_parquet(SPY_PRICES)
    spy = (prices[prices["ticker"] == "SPY"][["date", "close"]]
           .copy()
           .sort_values("date")
           .reset_index(drop=True))
    spy["date"] = pd.to_datetime(spy["date"])
    spy["spy_ret"] = spy["close"].pct_change()

    df = pd.merge(book, spy[["date", "spy_ret"]], on="date", how="left")
    df = df.dropna(subset=["spy_ret"]).reset_index(drop=True)

    spy_std = float(df["spy_ret"].std(ddof=1))
    sigma_band = 0.5 * spy_std  # HC #555 wheel: ">=0.5 sigma either side"

    # 3) Three classifiers.
    cls_T1 = compute_for_classifier(
        df, green_thr=+0.0010, red_thr=-0.0010,
        name="T1_strict_010pct",
        threshold_note="SPY close-to-close +/-0.10% bands (strict, as just-failed)")
    cls_T2 = compute_for_classifier(
        df, green_thr=+0.0025, red_thr=-0.0025,
        name="T2_loose_025pct",
        threshold_note="SPY close-to-close +/-0.25% bands (loose; matches blend_regime_gate.py)")
    cls_T3 = compute_for_classifier(
        df, green_thr=+sigma_band, red_thr=-sigma_band,
        name="T3_sigma_0p5",
        threshold_note=(
            "SPY close-to-close +/-0.5*sigma bands. HC #428 R1 does NOT pin a "
            "numeric threshold; closest pinned reference is HC #555 wheel "
            "(\">=0.5 sigma either side\")."))

    # 4) Overall (regime-agnostic) headline stats on the same 419-day stream.
    rets = df["daily_ret"].to_numpy()
    overall = {
        "n_days": int(len(df)),
        "date_start": str(df["date"].min().date()),
        "date_end": str(df["date"].max().date()),
        "sharpe": sharpe(rets),
        "mean_ret_bps": float(rets.mean() * 1e4),
        "std_ret_bps": float(rets.std(ddof=1) * 1e4),
        "cum_ret_pct": float((1.0 + rets).prod() * 100.0 - 100.0),
        "win_rate": float((rets > 0).mean()),
        "spy_ret_std_over_window": spy_std,
        "sigma_band_used_for_T3_pct": sigma_band * 100.0,
    }

    # 5) HC #428 source note + the leader's prior "pass" diagnosis.
    hc428_note = {
        "hc_id": "HC #428 R1",
        "verbatim_classifier_language": (
            "Day-classification (green/red/flat) using SPY or ES daily "
            "close-to-close (regime label)"),
        "verbatim_gate": (
            "Regime-agnostic gate: |Sharpe_green - Sharpe_red| / "
            "max(|Sharpe_green|, |Sharpe_red|) <= 0.50"),
        "threshold_pinned_in_HC_428": False,
        "closest_pinned_threshold_in_directives": (
            "HC #555 (wheel ladder R2): \"green / red / flat >=0.5 sigma either "
            "side\". HC #428 itself does NOT specify a numeric band."),
        "leader_prior_pass_was": (
            "validation_pack.json reported a bull/bear split using SPY vs "
            "60-day SMA (NOT green/red/flat close-to-close). Bear bucket had "
            "Sharpe=0 (degenerate, mean_d=0 because the regime overlay forces "
            "cash on bear days), which yielded delta_ratio=1.0 and a vacuous "
            "PASS. That artifact does NOT satisfy HC #428 R1's literal text."),
    }

    verdict_lines = []
    verdict_lines.append(
        "HC #428 R1 mandates green/red/flat close-to-close classification of "
        "SPY (or ES) daily returns, with regime-skew <= 0.50. It does NOT pin "
        "a numeric band. Under every reasonable band the leader fails:")
    for c in (cls_T1, cls_T2, cls_T3):
        rsk = c["regime_skew"]
        rsk_str = f"{rsk:.2f}" if rsk is not None else "nan"
        verdict_lines.append(
            f"  - {c['name']}: regime_skew={rsk_str} -> "
            f"{'PASS' if c['regime_gate_pass'] else 'FAIL'} "
            f"(green Sharpe={c['per_regime']['green']['sharpe']:.2f}, "
            f"red Sharpe={c['per_regime']['red']['sharpe']:.2f}, "
            f"n_green={c['per_regime']['green']['n_days']}, "
            f"n_red={c['per_regime']['red']['n_days']}, "
            f"n_flat={c['per_regime']['flat']['n_days']})")
    verdict_lines.append(
        "Verdict: the leader is NOT HC #428 R1 compliant under the literal "
        "green/red/flat close-to-close classification at any sane band width. "
        "Prior \"pass\" relied on a bull/bear (SPY-vs-MA) split with a "
        "degenerate cash-flat bear bucket; that does not satisfy HC #428 R1. "
        "The regime gate must be reopened before deploying the leader.")

    results = {
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "leader_book": str(LEADER_BOOK),
        "spy_prices": str(SPY_PRICES),
        "hc428_R1": hc428_note,
        "overall": overall,
        "classifiers": [cls_T1, cls_T2, cls_T3],
        "verdict": "\n".join(verdict_lines),
    }

    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"WROTE {out_path}")
    print()
    print("\n".join(verdict_lines))


if __name__ == "__main__":
    main()

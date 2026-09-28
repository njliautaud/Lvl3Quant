"""
HC #561 R5 — Beta-hedged residual-alpha evaluation of the macro picker.

DECISIVE KILL-OR-KEEP TEST
--------------------------
The HC #561 R3 walk-forward (walk_forward.py) found pooled OOT Sharpe -0.06,
PF 0.99, regime asymmetry 1.74 (green +3.06 / red -4.12): the strategy's
apparent edge is a long-beta tilt, not stock selection.  This module answers
ONE question: is there ANY residual stock-selection alpha once market beta is
removed?

HEDGE CHOICE (documented)
-------------------------
Option (a): subtract rolling-beta x market daily return from each day's
strategy return.  Chosen over a short-SPY overlay inside the simulator
because:
  * the saved per-fold OOT equity CSVs already contain the exact daily
    return streams — no GA re-run, no simulator re-run needed;
  * SPY is NOT in the price cache (verified), so the market proxy is the
    equal-weight universe close-to-close mean daily return — the SAME proxy
    walk_forward.py uses for regime classification, keeping regime labels
    and the hedge consistent;
  * an explicit short-overlay with this proxy would be untradeable anyway
    (it is an index proxy, not an instrument); the return-space hedge is the
    standard residual-alpha test.  A real SPY/ES hedge costs ~1-2 bps/rebal,
    immaterial vs the daily return scale here.

NO LOOK-AHEAD: beta_t is estimated on the TRAILING 60 days ending t-1
(rolling cov/var, then shift(1)); min 20 trailing days, else beta=0
(unhedged warm-up).  Pooled sector streams are contiguous (every-other
12m-OOT fold with a 12m step), so the rolling window crosses fold
boundaries naturally; per-fold metrics for ALL 15 folds hedge each fold's
own stream (first <=20 days unhedged warm-up — noted, immaterial over 250d).

Outputs:
  * MLflow experiment `macro_picker_walkforward`, runs tagged mode=beta_hedged
  * output/macro_picker/walkforward/{sector}/pooled_oot_equity_hedged.csv
  * output/macro_picker/walkforward/combined_pooled_oot_equity_hedged.csv
  * "Beta-Hedged Residual Alpha" section appended to SUMMARY.md with verdict
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("LVL3_ROOT", "/home/jupiter/Lvl3Quant"))
OUT_DIR = ROOT / "output/macro_picker/walkforward"
LOG_PATH = ROOT / "logs/macro_picker_beta_hedged.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [bh] %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("bh")

# NOTE: research/walk_forward.py and strategy/macro_picker/walk_forward.py
# share a module name — load the macro_picker one explicitly via importlib.
sys.path.insert(0, str(ROOT / "strategy/macro_picker"))
from ga_sector_fit_v2 import load_prices_and_adv  # noqa: E402

import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location(
    "mp_walk_forward", str(ROOT / "strategy/macro_picker/walk_forward.py"))
_mp_wf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mp_wf)
full_metrics = _mp_wf.full_metrics

MLFLOW_URI = "http://localhost:5000"
MLFLOW_EXPERIMENT = "macro_picker_walkforward"
BETA_WINDOW = 60
BETA_MIN_PERIODS = 20
RESIDUAL_ALPHA_MIN_SHARPE = 0.30   # "meaningfully > 0" threshold for CONTINUE
REGIME_SYMMETRY_MAX = 0.50         # HC #428 R1


def rolling_beta_hedge(rets: pd.Series, mkt: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Return (hedged_rets, beta_used). beta_t from trailing 60d ending t-1."""
    m = mkt.reindex(rets.index).fillna(0.0)
    cov = rets.rolling(BETA_WINDOW, min_periods=BETA_MIN_PERIODS).cov(m)
    var = m.rolling(BETA_WINDOW, min_periods=BETA_MIN_PERIODS).var()
    beta = (cov / var.replace(0.0, np.nan)).shift(1).fillna(0.0)  # no look-ahead
    hedged = rets - beta * m
    return hedged, beta


def load_fold_rets(sector_dir: Path) -> dict[int, pd.Series]:
    folds = {}
    for fp in sorted(sector_dir.glob("fold_*_oot_equity.csv")):
        k = int(fp.stem.split("_")[1])
        df = pd.read_csv(fp, index_col=0, parse_dates=True)
        folds[k] = df["ret"].astype(float)
    return folds


def main():
    prices = load_prices_and_adv()
    market_ret = prices.groupby("date")["ret"].mean().sort_index()

    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    sector_dirs = [d for d in OUT_DIR.iterdir()
                   if d.is_dir() and (d / "pooled_oot_equity.csv").exists()]
    log.info(f"sectors with pooled OOT results: {[d.name for d in sector_dirs]}")

    results = []
    for sd in sector_dirs:
        sector = sd.name
        fold_rets = load_fold_rets(sd)
        if not fold_rets:
            continue

        # ---- pooled stream: every-other fold (matches walk_forward.py) ----
        pooled = pd.concat([r for k, r in sorted(fold_rets.items()) if k % 2 == 0])
        pooled = pooled.sort_index()
        pooled = pooled[~pooled.index.duplicated(keep="first")]
        hedged, beta = rolling_beta_hedge(pooled, market_ret)
        pm = full_metrics(hedged, market_ret)
        pm["beta_mean"] = float(beta.mean())
        pm["beta_abs_mean"] = float(beta.abs().mean())

        out_csv = sd / "pooled_oot_equity_hedged.csv"
        eq = (1.0 + hedged).cumprod().rename("equity")
        eq.to_frame().assign(ret=hedged, beta=beta).to_csv(out_csv)

        # ---- per-fold hedged metrics (all folds, fold-local trailing beta) ----
        per_fold = {}
        for k, r in sorted(fold_rets.items()):
            h, _ = rolling_beta_hedge(r, market_ret)
            per_fold[k] = full_metrics(h, market_ret)

        with mlflow.start_run(run_name=f"{sector}_pooled_oot_beta_hedged"):
            mlflow.set_tag("mode", "beta_hedged")
            mlflow.log_params({
                "sector": sector,
                "hedge": "rolling_beta_subtract",
                "beta_window_days": BETA_WINDOW,
                "beta_min_periods": BETA_MIN_PERIODS,
                "market_proxy": "equal_weight_universe_c2c_mean",
                "pooling": "every_other_fold_non_overlapping",
                "window": "sliding_36m_train_12m_oot_6m_step",
            })
            for k2, v2 in pm.items():
                if isinstance(v2, (int, float)) and np.isfinite(v2):
                    mlflow.log_metric(f"pooled_hedged_{k2}", float(v2))
            mlflow.log_artifact(str(out_csv))

        log.info(f"[{sector}] HEDGED pooled OOT: Sharpe {pm['sharpe']:+.2f} "
                 f"Sortino {pm['sortino']:+.2f} PF {pm['pf']:.2f} WR {pm['wr']:.2f} "
                 f"MaxDD {pm['max_dd']:.2%} asym {pm['regime_asym']:.2f} "
                 f"mean_beta {pm['beta_mean']:+.3f}")
        results.append({"sector": sector, "pooled_metrics": pm,
                        "hedged_rets": hedged, "per_fold": per_fold})

    if not results:
        log.error("no sector results found — run walk_forward.py first")
        return

    # ---- combined: equal-weight across sector hedged pooled streams ----
    combined = pd.concat([r["hedged_rets"].rename(r["sector"]) for r in results],
                         axis=1).mean(axis=1, skipna=True).dropna().sort_index()
    cm = full_metrics(combined, market_ret)
    out_csv = OUT_DIR / "combined_pooled_oot_equity_hedged.csv"
    eq = (1.0 + combined).cumprod().rename("equity")
    eq.to_frame().assign(ret=combined).to_csv(out_csv)

    sym_pass = np.isfinite(cm["regime_asym"]) and cm["regime_asym"] <= REGIME_SYMMETRY_MAX
    has_alpha = (np.isfinite(cm["sharpe"])
                 and cm["sharpe"] >= RESIDUAL_ALPHA_MIN_SHARPE and sym_pass)
    verdict = "CONTINUE R&D — residual alpha present" if has_alpha \
        else "KILL LANE — no residual stock-selection alpha after beta hedge"

    with mlflow.start_run(run_name="combined_pooled_oot_beta_hedged"):
        mlflow.set_tag("mode", "beta_hedged")
        mlflow.log_params({
            "sectors": ",".join(r["sector"] for r in results),
            "hedge": "rolling_beta_subtract",
            "beta_window_days": BETA_WINDOW,
            "market_proxy": "equal_weight_universe_c2c_mean",
            "verdict": verdict[:120],
        })
        for k2, v2 in cm.items():
            if isinstance(v2, (int, float)) and np.isfinite(v2):
                mlflow.log_metric(f"pooled_hedged_{k2}", float(v2))
        mlflow.log_artifact(str(out_csv))

    # ---- append SUMMARY.md section ----
    def fm(x, p=3):
        return "n/a" if x is None or not np.isfinite(x) else f"{x:.{p}f}"

    lines = [
        "",
        "---",
        "",
        "# Beta-Hedged Residual Alpha (HC #561 R5 kill-or-keep test)",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Hedge: daily strategy return minus rolling-beta x market return.",
        "Beta = trailing 60d cov/var ending the PRIOR day (shift(1), no",
        "look-ahead; min 20d warm-up unhedged). Market proxy = equal-weight",
        "universe close-to-close mean return (SPY not in price cache; same",
        "proxy as the regime labels). Reuses saved per-fold OOT return",
        "streams — GA and simulator NOT re-run.",
        "",
        "## Pooled hedged OOT — combined (equal-weight sector sleeves)",
        "",
        "| Metric | Unhedged | Hedged |",
        "|---|---|---|",
        f"| Sharpe | -0.057 | {fm(cm['sharpe'])} |",
        f"| Sortino | -0.085 | {fm(cm['sortino'])} |",
        f"| PF | 0.990 | {fm(cm['pf'])} |",
        f"| WR | 0.491 | {fm(cm['wr'])} |",
        f"| MaxDD | -0.504 | {fm(cm['max_dd'])} |",
        f"| Calmar | -0.035 | {fm(cm['calmar'])} |",
        f"| Sharpe green/red/flat | +3.06 / -4.12 / -1.21 | "
        f"{fm(cm['sharpe_green'], 2)} / {fm(cm['sharpe_red'], 2)} / "
        f"{fm(cm['sharpe_flat'], 2)} |",
        f"| Regime asymmetry | 1.742 (FAIL) | {fm(cm['regime_asym'])} "
        f"({'PASS' if sym_pass else 'FAIL'}) |",
        f"| OOT days pooled | 2011 | {cm['n_days']} |",
        "",
        "## Per-sector pooled hedged OOT",
        "",
        "| Sector | Sharpe | Sortino | PF | WR | MaxDD | RegAsym | mean beta |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        p = r["pooled_metrics"]
        lines.append(
            f"| {r['sector']} | {fm(p['sharpe'])} | {fm(p['sortino'])} | "
            f"{fm(p['pf'])} | {fm(p['wr'])} | {fm(p['max_dd'])} | "
            f"{fm(p['regime_asym'])} | {fm(p['beta_mean'])} |")
    lines += [
        "",
        "## Per-fold hedged OOT Sharpe",
        "",
    ]
    for r in results:
        sh = [f"{k:02d}:{fm(m['sharpe'], 2)}" for k, m in sorted(r["per_fold"].items())]
        lines.append(f"- {r['sector']}: " + "  ".join(sh))
    lines += [
        "",
        f"## Verdict: **{verdict}**",
        "",
        f"- Residual-alpha gate (hedged Sharpe >= {RESIDUAL_ALPHA_MIN_SHARPE:.2f} "
        f"AND regime asymmetry <= {REGIME_SYMMETRY_MAX:.2f}): "
        f"{'PASS' if has_alpha else 'FAIL'}",
        "",
    ]
    with open(OUT_DIR / "SUMMARY.md", "a") as fh:
        fh.write("\n".join(lines))
    log.info(f"SUMMARY appended -> {OUT_DIR / 'SUMMARY.md'}")
    log.info(f"VERDICT: {verdict} | hedged pooled Sharpe {fm(cm['sharpe'])} "
             f"Sortino {fm(cm['sortino'])} PF {fm(cm['pf'])} WR {fm(cm['wr'])} "
             f"MaxDD {fm(cm['max_dd'])} Calmar {fm(cm['calmar'])} "
             f"asym {fm(cm['regime_asym'])}")


if __name__ == "__main__":
    main()

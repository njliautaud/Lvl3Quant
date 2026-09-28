"""
K=6 LONG/SHORT market-neutral v4 — short the WEAKEST megacaps against the long book.

v1/v2 (skip gates) and v3 (SPY hedge overlays) all REJECTED for HC #428 R1.
v3 diagnosis: static beta hedge neutralizes red-day BETA (red Sharpe -7.38 ->
+0.03) but the residual gap (0.99) is ALPHA asymmetry — long momentum alpha
pays +4.01 green vs +0.03 red. Passing R1 needs red-day-POSITIVE alpha =>
short the weakest-momentum megacaps against the long book.

Variants (6 cells):
  long top-6 momentum (UNCHANGED ranking, same ridge picker as baseline)
  x short bottom-N of the SAME 8-name universe, N in {3, 6}
  x short-leg sizing in {0.5x long notional, 1.0x (dollar-neutral)}
  + beta-neutral sizing (short leg scaled so portfolio trailing-60d beta ~ 0,
    t-1 info only) for N=3 and N=6.

UNIVERSE NOTE (8 names, K_long=6): long top-6 and short bottom-N OVERLAP by
construction (N=3 overlaps rank 6; N=6 overlaps ranks 3-6). Positions are
NETTED per name (net_w = long_w - short_w). E.g. N=6 @ 1.0x is effectively
long ranks 1-2 vs short ranks 7-8 at 1/6 each — an honest momentum spread
book. Gross/net exposures are logged per variant.

HARNESS — IDENTICAL to the K6_mom60 baseline (megacap_tech_extended_v1):
  - SLIDING 24m/6m/3m walk-forward (HC #0 — NEVER expanding).
  - Per-fold cross-sectional ridge (alpha=1) on [ret_20d, ret_mom60,
    rel_strength_spy] z-scores; weekly rebalance (HOLD_DAYS=5).
  - Regime gate: SPY > 50d MA AND VIX < 25, else FLAT (BOTH legs — shorts are
    a relative-value leg against the long book, not a standalone bear book).
  - Overlapping-OOT fold-sum convention of the baseline is PRESERVED
    (daily rows from overlapping folds are summed => ~2x fold-ensemble;
    Sharpe / regime gap / day-conc are scale-invariant, CAGR/MaxDD are on the
    same scale as the baseline book so all comparisons are apples-to-apples).
  - VALIDATION: a sizing=0 (long-only) pass is reconciled against the cached
    baseline book parquet; max abs daily diff must be ~0.

COSTS — same per-name model as baseline:
  - $0.005/share commission + 1bp slippage, charged on |delta net weight| per
    name at each rebalance (covers entries, exits, resizes, and sign flips).
  - Shorts: extra 25 bps/yr borrow on GROSS short notional, accrued daily
    while held (megacaps = general collateral, cheap to borrow).

LEAKAGE:
  - Ranking features are t-1-close-based returns (panel ret_* computed from
    closes through the rebal date; positions earn from rd+1 onward, identical
    to baseline).
  - Beta-neutral sizing: per-stock trailing 60d beta vs SPY, shift(1) =>
    uses returns through t-1 only. Clip portfolio hedge scale to [0, 2].
  - Day-t SPY return used ONLY for EVAL stratification (HC #428 R1).
  - Degenerate-fold guard (v3 convention): folds whose OOT ridge scores are
    constant/non-finite (no cross-sectional info => ranking arbitrary) are
    DROPPED and counted in diagnostics.

MLflow: experiment k6_long_short_v4 @ http://localhost:5000
        parent run + nested run per variant; logs regime_gap +
        passes_hc428_r1 per variant.

Outputs: output/macro_picker/k6_long_short_v4/
Live K=6 paper state: UNTOUCHED (research only).
"""
from __future__ import annotations
import json
import sys
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT / "strategy" / "macro_picker"))

from walk_forward import _metrics  # type: ignore
from megacap_tech_rotation import (  # type: ignore
    UNIVERSE, BENCH_SPY, HOLD_DAYS,
    TRAIN_MONTHS, OOT_MONTHS, STEP_MONTHS, WF_START, WF_END,
    _load_prices, _load_vix, _spy_regime, _vix_ok, _combined_regime,
    _xs_zscore, _iter_wf, _estimate_share_price_for_costs, _trade_cost_pct,
    regime_stratification, deploy_verdict,
)
from megacap_tech_extended_v1 import build_panel_with_window, tail_dd_metric  # type: ignore

K6_BOOK_PATH = ROOT / "output/macro_picker/megacap_tech_extended_v1_20260609_171903/book_K6_mom60.parquet"
OUT_DIR = ROOT / "output/macro_picker/k6_long_short_v4"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MOM_DAYS = 60
K_LONG = 6
BORROW_BPS_PER_YEAR = 25.0           # megacap GC borrow assumption
BORROW_DAILY = BORROW_BPS_PER_YEAR / 1e4 / 252.0
BETA_WINDOW = 60
BETA_SCALE_CLIP = (0.0, 2.0)
TRADING_DAYS = 252

# v3 static-beta comparator (research/findings/k6_hedge_overlay_v3.md)
V3_STATIC_BETA_REF = {"sharpe": 2.24, "calmar": 4.46, "regime_gap": 0.99}

VARIANTS = [
    {"name": "ls_n3_s050",       "n_short": 3, "sizing": "fixed", "s": 0.5},
    {"name": "ls_n3_s100",       "n_short": 3, "sizing": "fixed", "s": 1.0},
    {"name": "ls_n6_s050",       "n_short": 6, "sizing": "fixed", "s": 0.5},
    {"name": "ls_n6_s100",       "n_short": 6, "sizing": "fixed", "s": 1.0},
    {"name": "ls_n3_betaneutral", "n_short": 3, "sizing": "beta_neutral"},
    {"name": "ls_n6_betaneutral", "n_short": 6, "sizing": "beta_neutral"},
]


# ---------------------------------------------------------------------------
# Trailing per-stock beta vs SPY (t-1 info only)
# ---------------------------------------------------------------------------
def stock_betas(prices: pd.DataFrame) -> pd.DataFrame:
    """DataFrame [date x ticker] of trailing 60d realized beta vs SPY,
    shifted 1d so the value at date t uses returns through t-1 ONLY."""
    spy = (prices[prices["ticker"] == BENCH_SPY]
           .set_index("date")["close"].sort_index().pct_change())
    cols = {}
    for t in UNIVERSE:
        r = (prices[prices["ticker"] == t]
             .set_index("date")["close"].sort_index().pct_change())
        s = spy.reindex(r.index)
        beta = r.rolling(BETA_WINDOW).cov(s) / s.rolling(BETA_WINDOW).var()
        cols[t] = beta.shift(1)  # <= t-1 info only
    return pd.DataFrame(cols).sort_index()


def beta_asof(betas: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
    sub = betas.loc[:date]
    if sub.empty:
        return pd.Series(1.0, index=betas.columns)
    return sub.iloc[-1].fillna(1.0)


# ---------------------------------------------------------------------------
# Long/short walk-forward (baseline harness + netted short leg)
# ---------------------------------------------------------------------------
def run_wf_ls(panel: pd.DataFrame, regime: pd.Series, betas: pd.DataFrame,
              n_short: int, sizing: str, s_fixed: float = 0.0
              ) -> tuple[pd.DataFrame, list, dict]:
    """Replicates megacap_tech_extended_v1.run_wf (K=6, mom60, equal weight)
    EXACTLY for the long leg, adds a netted bottom-n_short short leg.
    n_short=0 => pure long-only (harness validation mode)."""
    feats = ["ret_20d", "ret_mom", "rel_strength_spy"]
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    panel["close"] = panel["close"].astype(float)
    panel["y_fwd"] = panel.groupby("ticker")["close"].shift(-HOLD_DAYS) / panel["close"] - 1.0

    windows = _iter_wf(WF_START, WF_END)
    daily_rows, rebal_rows = [], []
    prev_net: dict[str, float] = {}
    n_degenerate_folds = 0
    scale_log = []

    for fold_i, (tr_s, tr_e, os_, oe) in enumerate(windows):
        train = panel[(panel["date"] >= tr_s) & (panel["date"] < tr_e)].copy()
        oot = panel[(panel["date"] >= os_) & (panel["date"] < oe)].copy()
        if len(train) < 100 or len(oot) < 20:
            continue
        train_z = _xs_zscore(train, feats).dropna(subset=["y_fwd"])
        oot_z = _xs_zscore(oot, feats)
        if train_z.empty or oot_z.empty:
            continue
        X = train_z[feats].values
        y = train_z["y_fwd"].values
        Xc = X - X.mean(axis=0)
        yc = y - y.mean()
        try:
            beta = np.linalg.solve(Xc.T @ Xc + 1.0 * np.eye(len(feats)), Xc.T @ yc)
        except np.linalg.LinAlgError:
            beta = np.zeros(len(feats))
        intercept = y.mean() - X.mean(axis=0) @ beta
        oot_z["score"] = oot_z[feats].values @ beta + intercept
        oot_z["ret_raw"] = pd.to_numeric(oot["ret_1d"], errors="coerce").astype(float).values

        # DEGENERATE-FOLD GUARD: constant/non-finite scores => ranking is
        # arbitrary noise. Drop the fold (v3 convention).
        sc = oot_z["score"].values
        if not np.all(np.isfinite(sc[~np.isnan(sc)])) or float(np.nanstd(sc)) < 1e-12:
            n_degenerate_folds += 1
            print(f"[v4] fold {fold_i} OOT {os_.date()} DROPPED (degenerate scores)")
            continue

        unique_dates = sorted(oot_z["date"].unique())
        rebal_dates = unique_dates[::HOLD_DAYS]

        for rd in rebal_dates:
            rd_ts = pd.Timestamp(rd)
            reg = regime.get(rd_ts)
            if reg is None:
                prior = regime.loc[:rd_ts]
                reg = prior.iloc[-1] if len(prior) else "cash"

            snap = oot_z[oot_z["date"] == rd].dropna(subset=["score"])
            if len(snap) < K_LONG:
                continue

            if reg != "bull":
                net_w: dict[str, float] = {}
                s_used = 0.0
            else:
                ranked = snap.sort_values("score", ascending=False)
                longs = ranked.head(K_LONG)["ticker"].tolist()
                long_w = {t: 1.0 / K_LONG for t in longs}
                short_w: dict[str, float] = {}
                s_used = 0.0
                if n_short > 0:
                    shorts = ranked.tail(n_short)["ticker"].tolist()
                    if sizing == "fixed":
                        s_used = s_fixed
                    else:  # beta_neutral: scale so portfolio trailing-60d beta ~ 0
                        b = beta_asof(betas, rd_ts)
                        beta_long = float(np.mean([b.get(t, 1.0) for t in longs]))
                        beta_short_unit = float(np.mean([b.get(t, 1.0) for t in shorts]))
                        if np.isfinite(beta_short_unit) and beta_short_unit > 0.10:
                            s_used = beta_long / beta_short_unit
                        else:
                            s_used = 1.0
                        s_used = float(np.clip(s_used, *BETA_SCALE_CLIP))
                        scale_log.append(s_used)
                    short_w = {t: s_used / n_short for t in shorts}
                net_w = {}
                for t in set(long_w) | set(short_w):
                    w = long_w.get(t, 0.0) - short_w.get(t, 0.0)
                    if abs(w) > 1e-12:
                        net_w[t] = w

            gross_short = float(sum(-w for w in net_w.values() if w < 0))
            gross_long = float(sum(w for w in net_w.values() if w > 0))
            rebal_rows.append({
                "rebal_date": rd_ts, "regime": reg,
                "positions": ",".join(f"{t}:{w:+.4f}" for t, w in sorted(net_w.items())),
                "n_pos": len(net_w), "short_scale": s_used,
                "gross_long": gross_long, "gross_short": gross_short,
                "fold_oot_start": str(os_.date()),
            })

            # Cost on |delta net weight| per name (entries+exits+resizes+flips)
            tc = 0.0
            for t in set(net_w) | set(prev_net):
                dw = abs(net_w.get(t, 0.0) - prev_net.get(t, 0.0))
                if dw > 1e-9:
                    price = _estimate_share_price_for_costs(panel, t, rd_ts)
                    tc += dw * _trade_cost_pct(price)
            if tc > 0:
                daily_rows.append({"date": rd_ts, "ret": -tc, "cost": tc, "borrow": 0.0})
            prev_net = net_w

            hold_win = oot_z[(oot_z["date"] > rd)
                             & (oot_z["date"] <= rd + pd.Timedelta(days=int(HOLD_DAYS * 1.5)))]
            for d, g in hold_win.groupby("date"):
                d_ts = pd.Timestamp(d)
                rg = regime.get(d_ts)
                if rg is None:
                    prior = regime.loc[:d_ts]
                    rg = prior.iloc[-1] if len(prior) else "cash"
                if not net_w or rg != "bull":
                    daily_rows.append({"date": d_ts, "ret": 0.0, "cost": 0.0, "borrow": 0.0})
                    continue
                ret = 0.0
                for t, w in net_w.items():
                    row = g[g["ticker"] == t]["ret_raw"]
                    if not row.empty and pd.notna(row.iloc[0]):
                        ret += w * float(row.iloc[0])
                borrow = gross_short * BORROW_DAILY
                daily_rows.append({"date": d_ts, "ret": ret - borrow,
                                   "cost": 0.0, "borrow": borrow})

    if not daily_rows:
        return pd.DataFrame(columns=["date", "daily_ret"]), rebal_rows, {}
    df = pd.DataFrame(daily_rows)
    df["date"] = pd.to_datetime(df["date"])
    book = df.groupby("date", as_index=False).agg(
        daily_ret=("ret", "sum"), cost=("cost", "sum"), borrow=("borrow", "sum"))
    diag = {
        "n_degenerate_folds_dropped": n_degenerate_folds,
        "total_trade_cost_pct": float(book["cost"].sum() * 100),
        "total_borrow_cost_pct": float(book["borrow"].sum() * 100),
        "mean_short_scale": float(np.mean(scale_log)) if scale_log else None,
    }
    return book.sort_values("date").reset_index(drop=True), rebal_rows, diag


# ---------------------------------------------------------------------------
# Evaluation (HC #428 R1 + HC #344 + headline metrics)
# ---------------------------------------------------------------------------
def summarize(book: pd.DataFrame, spy_ret: pd.Series, spy_close: pd.Series,
              label: str) -> dict:
    s = book.set_index("date")["daily_ret"]
    m = _metrics(s)
    strat = regime_stratification(book[["date", "daily_ret"]], spy_ret)  # day-t SPY, EVAL ONLY
    gates = deploy_verdict(m, strat, book[["date", "daily_ret"]])
    tail = tail_dd_metric(book[["date", "daily_ret"]], spy_close)
    return {
        "label": label,
        "cagr_pct": (m.get("cagr") or 0) * 100,
        "sharpe": m.get("sharpe"), "sortino": m.get("sortino"),
        "calmar": m.get("calmar"),
        "max_dd_pct": (m.get("max_dd") or 0) * 100,
        "pf": m.get("pf"), "wr_pct": (m.get("wr") or 0) * 100,
        "regime_strat": strat.to_dict(orient="records"),
        "regime_gap": gates.get("regime_imbalance"),
        "regime_gap_pass_le_0_50": bool(gates.get("regime_balance_ok")),
        "regime_green_sh": gates.get("regime_green_sharpe"),
        "regime_red_sh": gates.get("regime_red_sharpe"),
        "passes_hc428_r1": bool(gates["PASSES_DEPLOY_GATES"]),
        "day_conc": gates.get("day_concentration"),
        "day_conc_pass_le_0_70": bool(gates.get("day_concentration_ok")),
        "n_oot_days": int(s.dropna().shape[0]),
        "worst_red_quarter_dd_pct": tail["worst_red_quarter_dd_pct"],
        "passes_tail_dd_25pct": tail["passes_tail_dd_25pct"],
    }


def main():
    print("[v4] === K=6 Long/Short market-neutral v4 ===")
    t0 = time.time()

    prices = _load_prices()
    vix = _load_vix()
    spy_reg = _spy_regime(prices)
    regime = _combined_regime(spy_reg, _vix_ok(vix))
    spy_close = prices[prices["ticker"] == BENCH_SPY].sort_values("date").set_index("date")["close"]
    spy_ret = spy_close.pct_change().dropna()
    panel = build_panel_with_window(prices, MOM_DAYS)
    betas = stock_betas(prices)
    print(f"[v4] prices={len(prices)} panel={len(panel)} betas={betas.shape}")

    # --- Harness validation: long-only (n_short=0) must reproduce baseline book
    k6_base = pd.read_parquet(K6_BOOK_PATH)
    k6_base["date"] = pd.to_datetime(k6_base["date"])
    lo_book, _, _ = run_wf_ls(panel, regime, betas, n_short=0, sizing="fixed", s_fixed=0.0)
    merged = k6_base.merge(lo_book[["date", "daily_ret"]], on="date",
                           suffixes=("_base", "_v4"), how="outer")
    max_diff = float((merged["daily_ret_base"] - merged["daily_ret_v4"]).abs().max())
    corr = float(merged["daily_ret_base"].corr(merged["daily_ret_v4"]))
    print(f"[v4] HARNESS VALIDATION vs baseline book: max|diff|={max_diff:.2e} corr={corr:.6f}")
    harness_ok = bool(max_diff < 1e-8)
    if not harness_ok:
        print("[v4] WARNING: long-only leg does not exactly reproduce baseline book "
              "— comparisons remain valid only if corr ~ 1.0")

    import mlflow
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("k6_long_short_v4")

    baseline = summarize(k6_base, spy_ret, spy_close, "K6_longonly_baseline")
    print(f"[v4] baseline: Sharpe={baseline['sharpe']:.2f} Calmar={baseline['calmar']:.2f} "
          f"gap={baseline['regime_gap']:.2f}")

    report = {
        "strategy": "k6_long_short_v4",
        "universe": UNIVERSE, "k_long": K_LONG, "mom_days": MOM_DAYS,
        "costs": {"commission_per_share_usd": 0.005, "slippage_bps": 1.0,
                  "borrow_bps_per_year_on_gross_short": BORROW_BPS_PER_YEAR},
        "wf_spec": {"train_months": TRAIN_MONTHS, "oot_months": OOT_MONTHS,
                    "step_months": STEP_MONTHS, "window_type": "SLIDING (HC #0)"},
        "harness_validation": {"max_abs_daily_diff": max_diff, "corr": corr,
                               "exact_match": harness_ok},
        "comparators": {"longonly_baseline": baseline,
                        "v3_static_beta_hedge": V3_STATIC_BETA_REF},
        "variants": {},
    }

    with mlflow.start_run(run_name="k6_long_short_v4_parent"):
        mlflow.log_params({"k_long": K_LONG, "mom_days": MOM_DAYS,
                           "borrow_bps_yr": BORROW_BPS_PER_YEAR,
                           "wf": "sliding_24m_6m_3m", "n_variants": len(VARIANTS)})
        mlflow.log_metric("harness_validation_max_diff", max_diff)
        for k, v in baseline.items():
            if isinstance(v, (int, float)) and v is not None:
                try:
                    if np.isfinite(float(v)):
                        mlflow.log_metric(f"baseline_{k}", float(v))
                except (TypeError, ValueError):
                    pass

        for spec in VARIANTS:
            name = spec["name"]
            print(f"\n[v4] ===== VARIANT {name} ({spec}) =====")
            book, rebal_rows, diag = run_wf_ls(
                panel, regime, betas, n_short=spec["n_short"],
                sizing=spec["sizing"], s_fixed=spec.get("s", 0.0))
            book.to_parquet(OUT_DIR / f"book_{name}.parquet", index=False)
            pd.DataFrame(rebal_rows).to_parquet(OUT_DIR / f"rebal_{name}.parquet", index=False)

            summ = summarize(book, spy_ret, spy_close, name)
            summ.update(diag)
            summ["delta_sharpe_vs_baseline"] = float((summ["sharpe"] or 0) - (baseline["sharpe"] or 0))
            summ["delta_sharpe_vs_v3_static"] = float((summ["sharpe"] or 0) - V3_STATIC_BETA_REF["sharpe"])
            report["variants"][name] = summ

            with mlflow.start_run(run_name=name, nested=True):
                mlflow.log_params({k: v for k, v in spec.items()})
                for k, v in summ.items():
                    if isinstance(v, (bool, int, float)) and v is not None:
                        try:
                            if np.isfinite(float(v)):
                                mlflow.log_metric(k, float(v))
                        except (TypeError, ValueError):
                            pass
            print(f"[v4:{name}] Sharpe={summ['sharpe']:.2f} Calmar={summ['calmar']:.2f} "
                  f"MaxDD={summ['max_dd_pct']:.1f}% green={summ['regime_green_sh']} "
                  f"red={summ['regime_red_sh']} gap={summ['regime_gap']:.2f} "
                  f"R1_PASS={summ['passes_hc428_r1']} "
                  f"costs(trade={diag['total_trade_cost_pct']:.2f}%, "
                  f"borrow={diag['total_borrow_cost_pct']:.3f}%)")

        report["wall_seconds"] = round(time.time() - t0, 1)
        (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, default=str))
        mlflow.log_artifact(str(OUT_DIR / "report.json"))

    print("\n[v4] === FINAL SUMMARY ===")
    print(f"  {'baseline':18s}: Sharpe={baseline['sharpe']:.2f} Calmar={baseline['calmar']:.2f} "
          f"gap={baseline['regime_gap']:.2f} R1={baseline['passes_hc428_r1']}")
    print(f"  {'v3_static_beta':18s}: Sharpe={V3_STATIC_BETA_REF['sharpe']:.2f} "
          f"Calmar={V3_STATIC_BETA_REF['calmar']:.2f} gap={V3_STATIC_BETA_REF['regime_gap']:.2f} R1=FAIL")
    for name, s in report["variants"].items():
        print(f"  {name:18s}: Sharpe={s['sharpe']:.2f} Calmar={s['calmar']:.2f} "
              f"MaxDD={s['max_dd_pct']:.1f}% gap={s['regime_gap']:.2f} "
              f"R1={s['passes_hc428_r1']} dayconc={s['day_conc']:.3f}")
    print(f"[v4] done in {report['wall_seconds']}s -> {OUT_DIR}")
    return report


if __name__ == "__main__":
    main()

"""
tier_runner.py — HC #556 R6 Phase 2.

Runs the 5 explicit wheel tiers (Conservative -> Turbo) through the wheel
engine on cached modeled-options data, computes risk-adjusted metrics, and
writes a comparative markdown + parquet report.

Usage:
    python -m strategy.tier_runner \
        --start 2018-01-01 --end 2026-06-04 \
        --capital 100000 \
        --modeled \
        --out results/tier_ladder

(Run from /home/jupiter/Lvl3Quant/wheel_strategy_v1/)
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from dataclasses import asdict
from typing import List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT))

from backtest.wheel_engine import run_wheel, WheelConfig  # noqa: E402
from strategy.tiers import all_tiers, all_tiers_full_wheel, TierSpec  # noqa: E402
from strategy.regime_overlay import build_regime, apply_regime_gate  # noqa: E402


# ----------- metrics -----------

TRADING_DAYS = 252


def _ann_return_from_curve(eq: pd.Series, days: int) -> float:
    if eq.empty or days <= 0:
        return 0.0
    start_eq = eq.iloc[0]
    end_eq = eq.iloc[-1]
    if start_eq <= 0:
        return 0.0
    yrs = days / TRADING_DAYS
    if yrs <= 0:
        return 0.0
    return float((end_eq / start_eq) ** (1.0 / yrs) - 1.0)


def _sharpe(daily_ret: pd.Series) -> float:
    if daily_ret.std() == 0 or daily_ret.empty:
        return 0.0
    return float(daily_ret.mean() / daily_ret.std() * np.sqrt(TRADING_DAYS))


def _sortino(daily_ret: pd.Series) -> float:
    if daily_ret.empty:
        return 0.0
    downside = daily_ret[daily_ret < 0]
    if downside.std() == 0 or downside.empty:
        return 0.0
    return float(daily_ret.mean() / downside.std() * np.sqrt(TRADING_DAYS))


def _max_dd(eq: pd.Series) -> float:
    if eq.empty:
        return 0.0
    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    return float(dd.min())


def _profit_factor(led: pd.DataFrame) -> float:
    if led.empty or "realized_pnl" not in led.columns:
        return 0.0
    gp = led.loc[led["realized_pnl"] > 0, "realized_pnl"].sum()
    gl = -led.loc[led["realized_pnl"] < 0, "realized_pnl"].sum()
    if gl <= 0:
        return float("inf") if gp > 0 else 0.0
    return float(gp / gl)


def _win_rate(led: pd.DataFrame) -> float:
    if led.empty or "realized_pnl" not in led.columns:
        return 0.0
    wins = (led["realized_pnl"] > 0).sum()
    return float(wins / len(led))


def _realized_cash_curve(led: pd.DataFrame, starting_cash: float,
                         dates: pd.DatetimeIndex) -> pd.Series:
    """Reconstruct a REALIZED-CASH curve from the trade ledger.

    Each closed trade (CSP close, CC close, called-away, expired-OTM) credits
    its realized_pnl on its close_date. We accumulate into a per-day series
    over the union of all trading dates in the equity curve so Sharpe / DD
    are computed on cash, not on MTM (which double-counts open-position vol).
    """
    if led is None or led.empty or "realized_pnl" not in led.columns:
        return pd.Series([float(starting_cash)] * len(dates),
                         index=pd.DatetimeIndex(dates))
    close_col = "close_date" if "close_date" in led.columns else "date"
    df = led[[close_col, "realized_pnl"]].copy()
    df[close_col] = pd.to_datetime(df[close_col])
    df = df.dropna(subset=[close_col])
    daily_pnl = df.groupby(close_col)["realized_pnl"].sum()
    series = pd.Series(0.0, index=pd.DatetimeIndex(dates))
    series.loc[series.index.isin(daily_pnl.index)] = daily_pnl.reindex(
        series.index[series.index.isin(daily_pnl.index)]).values
    return float(starting_cash) + series.cumsum()


def _regime_stratified(realized_ret: pd.Series,
                       spy_close: "pd.Series | None") -> dict:
    """HC #428 R1 regime-stratified Sharpe on the realized-cash daily returns.

    Day classification: SPY close-to-close return > +0.2% = green,
    < -0.2% = red, else flat. Returns per-regime Sharpe plus the R1 gap
    |Sh_g - Sh_r| / max(|Sh_g|, |Sh_r|). Logged to MLflow so the deploy-gate
    check does not depend on run narrative (checker follow-up 2026-06-10).
    """
    out = {"regime_green_sharpe": float("nan"),
           "regime_red_sharpe": float("nan"),
           "regime_flat_sharpe": float("nan"),
           "regime_gap": float("nan"),
           "regime_n_green": 0, "regime_n_red": 0, "regime_n_flat": 0}
    if spy_close is None or len(spy_close) < 3:
        return out
    spy_ret = spy_close.sort_index().pct_change()
    labels = pd.Series("flat", index=spy_ret.index)
    labels[spy_ret > 0.002] = "green"
    labels[spy_ret < -0.002] = "red"
    aligned = labels.reindex(realized_ret.index)
    for regime in ("green", "red", "flat"):
        sub = realized_ret[aligned == regime]
        out[f"regime_n_{regime}"] = int(len(sub))
        if len(sub) >= 2 and sub.std() > 0:
            out[f"regime_{regime}_sharpe"] = _sharpe(sub)
    sg, sr = out["regime_green_sharpe"], out["regime_red_sharpe"]
    denom = max(abs(sg), abs(sr))
    if denom == denom and denom > 0:  # not NaN, nonzero
        out["regime_gap"] = abs(sg - sr) / denom
    return out


def compute_metrics(result: dict, starting_cash: float,
                    spy_close: "pd.Series | None" = None) -> dict:
    eq_df = result["equity_curve"].sort_values("date").reset_index(drop=True)
    eq = eq_df["equity"].astype(float)
    if eq.empty:
        return {k: 0.0 for k in
                ["cagr", "sharpe", "sortino", "max_dd", "pf", "wr",
                 "n_trades", "assignment_rate", "called_away_rate",
                 "final_equity",
                 "realized_cagr", "realized_sharpe", "realized_sortino",
                 "realized_max_dd", "realized_final_equity"]}
    daily_ret = eq.pct_change().fillna(0.0)
    days = len(eq) - 1
    led = result["ledger"]

    # Realized cash curve — what your account would actually see, not MTM
    realized_eq = _realized_cash_curve(
        led, starting_cash,
        pd.DatetimeIndex(pd.to_datetime(eq_df["date"])))
    realized_ret = realized_eq.pct_change().fillna(0.0)

    csps = result.get("csp_opened", 0)
    asgn = result.get("assignment_count", 0)
    cawy = result.get("called_away_count", 0)
    return {
        "cagr": _ann_return_from_curve(eq, days),
        "sharpe": _sharpe(daily_ret),
        "sortino": _sortino(daily_ret),
        "max_dd": _max_dd(eq),
        "pf": _profit_factor(led),
        "wr": _win_rate(led),
        "n_trades": int(len(led)),
        "assignment_rate": (asgn / csps) if csps > 0 else 0.0,
        "called_away_rate": (cawy / max(result.get("cc_opened", 0), 1)),
        "final_equity": float(eq.iloc[-1]),
        "starting_cash": float(starting_cash),
        # ---- realized cash (account cash, NOT mark-to-market) ----
        "realized_cagr": _ann_return_from_curve(realized_eq, days),
        "realized_sharpe": _sharpe(realized_ret),
        "realized_sortino": _sortino(realized_ret),
        "realized_max_dd": _max_dd(realized_eq),
        "realized_final_equity": float(realized_eq.iloc[-1]),
        # ---- HC #428 R1 regime stratification (green/red/flat) ----
        **_regime_stratified(
            pd.Series(realized_ret.values,
                      index=pd.DatetimeIndex(pd.to_datetime(eq_df["date"]))),
            spy_close),
    }


# ----------- io helpers -----------

def _load_inputs(modeled: bool, smoke: bool, real_iv: bool = False) -> dict:
    suffix = "_smoke" if smoke else ""
    px_path = CACHE / f"prices{suffix}.parquet"
    macro_path = CACHE / f"macro{suffix}.parquet"
    fund_path = CACHE / f"fundamentals{suffix}.parquet"
    uni_path = CACHE / "universe.parquet"
    if real_iv:
        # HC #556 R3 real-vendor blended IV (DOLT volatility_history + modeled
        # pre-2019 + missing-ticker fallback). 51% of rows real, 49% modeled.
        iv_path = CACHE / f"iv_features_real_blend{suffix}.parquet"
        if not iv_path.exists():
            print(f"[runner] real-blend IV missing — falling back to modeled", file=sys.stderr)
            iv_path = CACHE / f"iv_features_modeled{suffix}.parquet"
    elif modeled:
        iv_path = CACHE / f"iv_features_modeled{suffix}.parquet"
        if not iv_path.exists():
            print(f"[runner] modeled IV missing — falling back to raw iv_features", file=sys.stderr)
            iv_path = CACHE / f"iv_features{suffix}.parquet"
    else:
        iv_path = CACHE / f"iv_features{suffix}.parquet"

    paths = dict(px=px_path, iv=iv_path, macro=macro_path,
                 fund=fund_path, uni=uni_path)
    missing = [n for n, p in paths.items() if not p.exists()]
    if missing:
        raise SystemExit(f"[runner] missing inputs: {missing}")
    return {
        "prices": pd.read_parquet(px_path),
        "iv": pd.read_parquet(iv_path),
        "macro": pd.read_parquet(macro_path),
        "fundamentals": pd.read_parquet(fund_path),
        "universe": pd.read_parquet(uni_path),
        "paths": paths,
    }


def _filter_universe_subset(tier: TierSpec, data: dict) -> List[str]:
    tickers = tier.universe_filter(
        data["universe"], data["fundamentals"], data["iv"], data["prices"]
    )
    return tickers


def _apply_iv_rank_floor(iv: pd.DataFrame, floor: float) -> pd.DataFrame:
    """
    Drop rows where iv_rank < floor so the engine's candidate selection
    cannot pick them. We DROP rather than NaN-out because downstream
    strike-from-delta math propagates NaNs.
    """
    if floor <= 0:
        return iv
    keep_mask = iv["iv_rank"].fillna(0.0) >= floor
    return iv[keep_mask].reset_index(drop=True)


def _load_spy_close(data: dict) -> "pd.Series | None":
    """SPY close series for HC #428 R1 day classification.

    Tries the equity prices cache first, then sector_etfs.parquet
    (where SPY actually lives in this repo). Returns None if unavailable —
    regime metrics then log as NaN rather than failing the run.
    """
    spy_px = data["prices"][data["prices"]["ticker"] == "SPY"]
    if spy_px.empty:
        try:
            etf = pd.read_parquet(CACHE / "sector_etfs.parquet")
            spy_px = etf[etf["ticker"] == "SPY"]
        except Exception:
            return None
    if spy_px.empty:
        return None
    return spy_px.set_index(
        pd.DatetimeIndex(pd.to_datetime(spy_px["date"])))["close"].astype(float)


# ----------- per-tier run -----------

def run_tier(tier: TierSpec, data: dict, start: str, end: str,
             capital: float) -> dict:
    tickers = _filter_universe_subset(tier, data)
    if not tickers:
        print(f"[runner] {tier.name}: universe filter returned 0 tickers")
        return {"tier": tier.name, "metrics": None, "n_universe": 0}

    px = data["prices"][data["prices"]["ticker"].isin(tickers)].copy()
    iv = data["iv"][data["iv"]["ticker"].isin(tickers)].copy()
    iv = _apply_iv_rank_floor(iv, tier.iv_rank_floor)

    print(f"[runner] {tier.name}: {len(tickers)} tickers, "
          f"{len(px):,} price rows, IV rank floor {tier.iv_rank_floor}")

    result = run_wheel(
        cfg=tier.wheel_cfg,
        prices=px,
        iv=iv,
        macro=data["macro"],
        fundamentals=data["fundamentals"],
        universe=data["universe"],
        starting_cash=capital,
        start=start, end=end,
        verbose=False,
    )
    metrics = compute_metrics(result, capital, spy_close=_load_spy_close(data))
    return {
        "tier": tier.name,
        "target_yield_pct": tier.target_annual_yield_pct,
        "description": tier.description,
        "n_universe": len(tickers),
        "metrics": metrics,
        "equity_curve": result["equity_curve"],
        "ledger": result["ledger"],
        "capital_share_default": tier.capital_share_default,
    }


# ----------- reporting -----------

def write_report(runs: List[dict], out_dir: Path, run_meta: dict):
    out_dir.mkdir(parents=True, exist_ok=True)

    # comparative parquet
    rows = []
    for r in runs:
        m = r.get("metrics") or {}
        rows.append({
            "tier": r["tier"],
            "target_yield_pct": r.get("target_yield_pct"),
            "n_universe": r["n_universe"],
            **m,
        })
    summary_df = pd.DataFrame(rows)
    summary_df.to_parquet(out_dir / "tier_summary.parquet", index=False)
    summary_df.to_csv(out_dir / "tier_summary.csv", index=False)

    # per-tier equity curve parquet
    for r in runs:
        if "equity_curve" in r and isinstance(r["equity_curve"], pd.DataFrame):
            r["equity_curve"].to_parquet(out_dir / f"equity_{r['tier']}.parquet",
                                          index=False)
        if "ledger" in r and isinstance(r["ledger"], pd.DataFrame):
            r["ledger"].to_parquet(out_dir / f"ledger_{r['tier']}.parquet",
                                    index=False)

    # markdown
    lines = []
    lines.append("# Wheel Strategy Tier Ladder — Comparative Report\n")
    lines.append(f"Pricing: **{run_meta.get('pricing_source','UNKNOWN')}** "
                 f"(MODELED unless real chain data wired in)\n")
    lines.append(f"Window: **{run_meta.get('start')} -> {run_meta.get('end')}**  "
                 f"Capital per tier: **${run_meta.get('capital'):,.0f}**\n\n")
    lines.append("## MTM (mark-to-market) metrics\n\n")
    lines.append("| Tier | Target % | Universe | CAGR % | Sharpe | Sortino | Max DD % | PF | WR % | Trades | Assign % |\n")
    lines.append("|------|---------:|---------:|-------:|-------:|--------:|---------:|---:|-----:|-------:|---------:|\n")
    for r in runs:
        m = r.get("metrics") or {}
        if not m:
            lines.append(f"| {r['tier']} | {r.get('target_yield_pct',0):.0f} | {r['n_universe']} | — | — | — | — | — | — | — | — |\n")
            continue
        lines.append(
            f"| {r['tier']} | {r.get('target_yield_pct',0):.0f} | {r['n_universe']} | "
            f"{m['cagr']*100:.2f} | {m['sharpe']:.2f} | {m['sortino']:.2f} | "
            f"{m['max_dd']*100:.2f} | {m['pf']:.2f} | {m['wr']*100:.1f} | "
            f"{m['n_trades']} | {m['assignment_rate']*100:.1f} |\n"
        )

    lines.append("\n## REALIZED-CASH metrics (account cash, no MTM noise)\n\n")
    lines.append("| Tier | Realized CAGR % | Realized Sharpe | Realized Sortino | Realized Max DD % | Final Equity $ |\n")
    lines.append("|------|----------------:|----------------:|-----------------:|------------------:|---------------:|\n")
    for r in runs:
        m = r.get("metrics") or {}
        if not m or "realized_cagr" not in m:
            lines.append(f"| {r['tier']} | — | — | — | — | — |\n")
            continue
        lines.append(
            f"| {r['tier']} | {m['realized_cagr']*100:.2f} | "
            f"{m['realized_sharpe']:.2f} | {m['realized_sortino']:.2f} | "
            f"{m['realized_max_dd']*100:.2f} | "
            f"${m['realized_final_equity']:,.0f} |\n"
        )
    lines.append("\n## Tier Descriptions\n")
    for r in runs:
        lines.append(f"- **{r['tier']}** (target {r.get('target_yield_pct',0):.0f}%): "
                     f"{r.get('description','')}  "
                     f"Default capital share: {r.get('capital_share_default',0):.0%}\n")
    lines.append("\n## Recommended capital allocation (default ladder split)\n")
    lines.append("| Tier | Default Share |\n|---|---:|\n")
    for r in runs:
        lines.append(f"| {r['tier']} | {r.get('capital_share_default',0):.0%} |\n")

    (out_dir / "tier_ladder_report.md").write_text("".join(lines))
    print(f"[runner] report written -> {out_dir/'tier_ladder_report.md'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--capital", type=float, default=100_000.0)
    ap.add_argument("--modeled", action="store_true",
                    help="Use iv_features_modeled.parquet if present.")
    ap.add_argument("--real-iv", action="store_true",
                    help="Use iv_features_real_blend.parquet (HC #556 R3 vendor IV + modeled fallback).")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tier", default=None,
                    help="Optional: run only one tier by name (e.g. Tier3_Income)")
    ap.add_argument("--full-wheel", action="store_true",
                    help="Use the assignment-allowed wheel variants.")
    ap.add_argument("--regime-overlay", action="store_true",
                    help="Apply HC #555 macro regime gate (yield curve / claims / inflation / sentiment / Fed).")
    ap.add_argument("--calibrated-skew", action="store_true",
                    help="Load DOLT-calibrated skew a/b from data/cache/"
                         "skew_calibration_real.json (lane A3) instead of "
                         "hardcoded -0.10/+0.20.")
    ap.add_argument("--calibrated-skew-per-ticker", action="store_true",
                    help="v9: per-ticker leak-free walk-forward skew. Each "
                         "backtest year Y uses each ticker's OWN median (a,b) "
                         "from surfaces dated strictly before Y-01-01; "
                         "tickers with <30 prior fits fall back to the "
                         "global-year (a,b). Implies --calibrated-skew "
                         "semantics; existing flags/behavior unchanged.")
    ap.add_argument("--mlflow-experiment", default=None,
                    help="If set, log params + per-tier metrics to MLflow at "
                         "http://localhost:5000 under this experiment name.")
    args = ap.parse_args()

    if args.calibrated_skew_per_ticker:
        # v9: per-ticker leak-free walk-forward schedule. Engine advances the
        # per-day per-ticker (a, b) map via iv_skew.set_asof; tickers without
        # enough prior surfaces use the global-year (a, b) automatically.
        from strategy import iv_skew as _ivs
        cal = _ivs.load_calibration_walkforward_perticker()
        per = {y: (v["a"], v["b"]) for y, v in cal["periods"].items()}
        n_pt = {y: len(t) for y, t in
                (cal.get("per_ticker_periods") or {}).items()}
        print(f"[runner] PER-TICKER walk-forward skew loaded. "
              f"global fallback: {per}")
        print(f"[runner] per-ticker coverage (tickers/year): {n_pt}")
    elif args.calibrated_skew:
        # Leak-free walk-forward schedule (lane A3 fix). The engine advances
        # (a, b) per simulated day via iv_skew.set_asof — year Y only ever
        # sees coefficients fit on pre-Y surfaces.
        from strategy import iv_skew as _ivs
        cal = _ivs.load_calibration_walkforward()
        per = {y: (v["a"], v["b"]) for y, v in cal["periods"].items()}
        print(f"[runner] walk-forward calibrated skew loaded: {per}")

    end = args.end or pd.Timestamp.today().strftime("%Y-%m-%d")
    data = _load_inputs(modeled=args.modeled, smoke=args.smoke, real_iv=args.real_iv)

    iv_path_str = str(data["paths"]["iv"])
    if "iv_features_real_blend" in iv_path_str:
        pricing_source = "real_blend_vendor_iv_modeled_fallback"
    elif "iv_features_modeled" in iv_path_str:
        pricing_source = "modeled_bs_calibrated"
    else:
        pricing_source = "modeled_bs_rv_only"

    if args.regime_overlay:
        print("[runner] applying macro regime overlay (HC #555)…")
        regime = build_regime()
        data["macro"] = apply_regime_gate(data["macro"], regime, vix_force_gate=999.0)
        ro_share = (data["macro"]["vix"] >= 999.0).mean() if "vix" in data["macro"].columns else 0.0
        print(f"[runner] regime gate: {ro_share:.1%} of days masked as risk-off")

    tiers = all_tiers_full_wheel() if args.full_wheel else all_tiers()
    if args.tier:
        tiers = [t for t in tiers if t.name == args.tier]
        if not tiers:
            raise SystemExit(f"[runner] tier name not found: {args.tier}")

    runs = []
    for t in tiers:
        print(f"\n=== {t.name} (target {t.target_annual_yield_pct:.1f}% annualized) ===")
        try:
            r = run_tier(t, data, args.start, end, args.capital)
            runs.append(r)
            m = r.get("metrics") or {}
            if m:
                print(f"[runner] {t.name}: CAGR={m['cagr']*100:.2f}%  "
                      f"Sharpe={m['sharpe']:.2f}  Sortino={m['sortino']:.2f}  "
                      f"MaxDD={m['max_dd']*100:.2f}%  Trades={m['n_trades']}")
        except Exception as e:
            print(f"[runner] {t.name} FAILED: {e}", file=sys.stderr)
            import traceback; traceback.print_exc()
            runs.append({"tier": t.name, "metrics": None, "n_universe": 0,
                         "error": str(e)})

    out_dir = Path(args.out) if args.out else RESULTS / "tier_ladder"
    # Capture engine toggles so the report makes the regime/skew/slippage state explicit.
    try:
        from backtest import wheel_engine as _we
        engine_flags = {
            "use_skew": bool(getattr(_we, "USE_SKEW", False)),
            "use_slippage": bool(getattr(_we, "USE_SLIPPAGE", False)),
            "slippage_frac": float(getattr(_we, "SLIPPAGE_FRAC", 0.0)),
            "slippage_min_ticks": float(getattr(_we, "SLIPPAGE_MIN_TICKS", 0.0)),
        }
        from strategy import iv_skew as _ivs2
        engine_flags["skew_a"] = float(_ivs2.SKEW_SLOPE_A)
        engine_flags["skew_b"] = float(_ivs2.SKEW_CURV_B)
        engine_flags["skew_calibration"] = str(
            getattr(_ivs2, "CALIBRATION_SOURCE", "hardcoded"))
        engine_flags["skew_walkforward"] = bool(
            getattr(_ivs2, "SCHEDULE", None))
    except Exception:
        engine_flags = {}
    run_meta = {
        "start": args.start, "end": end, "capital": args.capital,
        "pricing_source": pricing_source,
        "tiers_run": [t.name for t in tiers],
        "regime_overlay": bool(args.regime_overlay),
        "full_wheel": bool(args.full_wheel),
        "engine_flags": engine_flags,
    }
    write_report(runs, out_dir, run_meta)
    (out_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2, default=str))

    if args.mlflow_experiment:
        try:
            import mlflow
            mlflow.set_tracking_uri("http://localhost:5000")
            mlflow.set_experiment(args.mlflow_experiment)
            with mlflow.start_run(run_name=out_dir.name):
                mlflow.set_tag("walkforward_skew",
                               str(bool(engine_flags.get("skew_walkforward"))).lower())
                mlflow.log_params({
                    "start": args.start, "end": end, "capital": args.capital,
                    "pricing_source": pricing_source,
                    "regime_overlay": bool(args.regime_overlay),
                    "full_wheel": bool(args.full_wheel),
                    **{f"engine_{k}": v for k, v in engine_flags.items()},
                })
                for r in runs:
                    m = r.get("metrics") or {}
                    tname = r["tier"].replace("/", "_")
                    for k in ["cagr", "sharpe", "sortino", "max_dd", "pf",
                              "wr",
                              "regime_green_sharpe", "regime_red_sharpe",
                              "regime_flat_sharpe", "regime_gap",
                              "regime_n_green", "regime_n_red",
                              "regime_n_flat",
                              "realized_cagr", "realized_sharpe",
                              "realized_sortino", "realized_max_dd"]:
                        v = m.get(k)
                        if v is not None and np.isfinite(v):
                            mlflow.log_metric(f"{tname}.{k}", float(v))
                mlflow.log_artifact(str(out_dir / "tier_ladder_report.md"))
                mlflow.log_artifact(str(out_dir / "run_meta.json"))
            print(f"[runner] MLflow logged -> experiment {args.mlflow_experiment}")
        except Exception as e:
            print(f"[runner] MLflow logging failed (non-fatal): {e}", file=sys.stderr)

    print(f"\n[runner] DONE. {len(runs)} tiers run. See {out_dir}/tier_ladder_report.md")


if __name__ == "__main__":
    main()

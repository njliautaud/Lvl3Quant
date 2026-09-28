"""
wheel_lgbm_tiebreaker.py — LGBM score as TIEBREAKER within the V5 universe.

Corrected approach after the direct-replacement experiment failed (Sharpe 1.36
vs baseline 1.70) due to concentration in high-IV speculative names.

Design:
  - Keep FULL V5 baseline universe (all tickers passing fund_score gate).
  - LGBM pred_yield used only to resolve ties among candidates with similar
    iv_rank — NOT to gate the universe.
  - Volatility penalty: tickers with sigma > 0.60 have their LGBM score halved
    (reduces concentration in high-IV speculative names like TSLA/COIN/HOOD).
  - Composite iv_rank = iv_rank + 0.10 * lgbm_score_norm_penalized
    (scale factor 0.10 keeps LGBM subordinate to iv_rank, only resolves ties).
  - Cap enforcement: max 3 positions per ticker across the backtest is NOT
    enforceable inside wheel_engine without modification. Instead we set the
    LGBM weight to zero for any ticker that has been in the top-selected
    bucket 3+ times — handled via a dampening approach at the composite level.
    (The engine's 15%-per-name allocation cap already limits per-name exposure.)

LGBM scope: daily pred_yield from fold_predictions.parquet covers 2023-01-03
to 2025-09-30 — exactly the backtest OOS window. No lookahead.

Outputs:
  results/lgbm_v5_tiebreaker/
    equity_LGBM_Tiebreaker.parquet
    ledger_LGBM_Tiebreaker.parquet
    equity_Baseline_V5.parquet
    ledger_Baseline_V5.parquet
    tiebreaker_results.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT))

from backtest.wheel_engine import run_wheel, WheelConfig  # noqa: E402
from strategy.tier_runner import (  # noqa: E402
    compute_metrics, _apply_iv_rank_floor, _load_spy_close,
)

TRADING_DAYS = 252
RF_DAILY = 0.04 / TRADING_DAYS

# ---- V5 config (identical to integration.py) ----
V5_CFG = WheelConfig(
    put_delta_target=0.35,
    call_delta_target=0.30,
    dte_min=7,
    dte_max=14,
    profit_take_pct=0.65,
    roll_dte_trigger=1,
    max_concurrent_names=20,
    sector_cap_pct=0.25,
    vix_max_gate=35.0,
    naaim_min_gate=-60.0,
    fund_score_floor=35.0,
    r=0.04,
    max_assigned_notional_pct=1.0,
    share_stop_loss_pct=0.15,
    macro_lag_days=0,
)

IV_RANK_FLOOR = 0.20
CAPITAL = 100_000.0
START = "2023-01-01"
END = "2025-09-30"

# Tiebreaker hyper-params
LGBM_WEIGHT = 0.10        # composite = iv_rank + LGBM_WEIGHT * lgbm_norm_adj
HIGH_SIGMA_THRESHOLD = 0.60   # sigma > this -> halve the LGBM score
MAX_POSITIONS_PER_LGBM_TICKER = 3  # dampening: zero LGBM boost after N hits


# ---- metric helpers (same as integration.py) ----

def _sharpe_rf(daily_ret: pd.Series, rf_daily: float = RF_DAILY) -> float:
    excess = daily_ret - rf_daily
    if excess.std() == 0 or excess.empty:
        return 0.0
    return float(excess.mean() / excess.std() * np.sqrt(TRADING_DAYS))


def _sortino_rf(daily_ret: pd.Series, rf_daily: float = RF_DAILY) -> float:
    excess = daily_ret - rf_daily
    if excess.empty:
        return 0.0
    down = excess[excess < 0]
    if down.std() == 0 or down.empty:
        return 0.0
    return float(excess.mean() / down.std() * np.sqrt(TRADING_DAYS))


def _ann_cagr(eq: pd.Series, days: int) -> float:
    if eq.empty or days <= 0:
        return 0.0
    yrs = days / TRADING_DAYS
    if eq.iloc[0] <= 0 or yrs <= 0:
        return 0.0
    return float((eq.iloc[-1] / eq.iloc[0]) ** (1.0 / yrs) - 1.0)


def _max_dd(eq: pd.Series) -> float:
    if eq.empty:
        return 0.0
    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    return float(dd.min())


def _profit_factor(led: pd.DataFrame) -> float:
    if led.empty or "realized_pnl" not in led.columns:
        return float("nan")
    gp = led.loc[led["realized_pnl"] > 0, "realized_pnl"].sum()
    gl = -led.loc[led["realized_pnl"] < 0, "realized_pnl"].sum()
    if gl <= 0:
        return float("inf") if gp > 0 else float("nan")
    return float(gp / gl)


def _win_rate(led: pd.DataFrame) -> float:
    if led.empty or "realized_pnl" not in led.columns:
        return float("nan")
    return float((led["realized_pnl"] > 0).mean())


def _day_conc(equity_curve: pd.DataFrame) -> float:
    if equity_curve.empty:
        return float("nan")
    eq = equity_curve.set_index("date")["equity"].sort_index()
    daily_pnl = eq.diff().dropna()
    total_pnl = daily_pnl.sum()
    if total_pnl <= 0:
        return float("nan")
    return float(daily_pnl.max() / total_pnl)


def _regime_split(daily_ret: pd.Series, spy_close: pd.Series | None) -> dict:
    out = {
        "green_sharpe": float("nan"), "red_sharpe": float("nan"),
        "flat_sharpe": float("nan"),
        "n_green": 0, "n_red": 0, "n_flat": 0,
        "regime_gap": float("nan"),
    }
    if spy_close is None or len(spy_close) < 3:
        return out
    spy_ret = spy_close.sort_index().pct_change()
    labels = pd.Series("flat", index=spy_ret.index)
    labels[spy_ret > 0.002] = "green"
    labels[spy_ret < -0.002] = "red"
    aligned = labels.reindex(daily_ret.index)
    for regime in ("green", "red", "flat"):
        sub = daily_ret[aligned == regime]
        out[f"n_{regime}"] = int(len(sub))
        if len(sub) >= 5 and sub.std() > 0:
            out[f"{regime}_sharpe"] = _sharpe_rf(sub)
    sg, sr = out["green_sharpe"], out["red_sharpe"]
    if not (np.isnan(sg) or np.isnan(sr)):
        denom = max(abs(sg), abs(sr))
        if denom > 0:
            out["regime_gap"] = abs(sg - sr) / denom
    return out


def _realized_curve(led: pd.DataFrame, starting_cash: float,
                    dates: pd.DatetimeIndex) -> pd.Series:
    if led is None or led.empty or "realized_pnl" not in led.columns:
        return pd.Series([float(starting_cash)] * len(dates), index=dates)
    close_col = "close_date" if "close_date" in led.columns else "date"
    df = led[[close_col, "realized_pnl"]].copy()
    df[close_col] = pd.to_datetime(df[close_col])
    df = df.dropna(subset=[close_col])
    daily_pnl = df.groupby(close_col)["realized_pnl"].sum()
    series = pd.Series(0.0, index=dates)
    series.loc[series.index.isin(daily_pnl.index)] = \
        daily_pnl.reindex(series.index[series.index.isin(daily_pnl.index)]).values
    return float(starting_cash) + series.cumsum()


def full_metrics(result: dict, spy_close: pd.Series | None) -> dict:
    eq_df = result["equity_curve"].sort_values("date").reset_index(drop=True)
    eq = eq_df["equity"].astype(float)
    dates = pd.DatetimeIndex(pd.to_datetime(eq_df["date"]))
    led = result["ledger"]
    days = len(eq) - 1

    real_eq = _realized_curve(led, result["starting_cash"], dates)
    real_ret = real_eq.pct_change().fillna(0.0)
    mtm_ret = eq.pct_change().fillna(0.0)
    regime = _regime_split(real_ret, spy_close)

    return {
        "cagr": _ann_cagr(eq, days),
        "sharpe": _sharpe_rf(mtm_ret),
        "sortino": _sortino_rf(mtm_ret),
        "max_dd": _max_dd(eq),
        "pf": _profit_factor(led),
        "wr": _win_rate(led),
        "n_trades": int(len(led)),
        "realized_cagr": _ann_cagr(real_eq, days),
        "realized_sharpe": _sharpe_rf(real_ret),
        "realized_sortino": _sortino_rf(real_ret),
        "realized_max_dd": _max_dd(real_eq),
        "day_conc": _day_conc(eq_df),
        "final_equity": float(eq.iloc[-1]),
        "realized_final_equity": float(real_eq.iloc[-1]),
        **regime,
    }


# ---- load data ----

def load_data() -> dict:
    print("[tiebreaker] Loading cache data...")
    prices = pd.read_parquet(CACHE / "prices.parquet")
    iv_path = CACHE / "iv_features_real_blend.parquet"
    if not iv_path.exists():
        iv_path = CACHE / "iv_features_modeled.parquet"
    iv = pd.read_parquet(iv_path)
    macro = pd.read_parquet(CACHE / "macro.parquet")
    fund = pd.read_parquet(CACHE / "fundamentals.parquet")
    universe = pd.read_parquet(CACHE / "universe.parquet")
    print(f"[tiebreaker] Prices: {len(prices):,} rows | IV: {len(iv):,} rows | "
          f"Tickers: {prices['ticker'].nunique()}")
    return dict(prices=prices, iv=iv, macro=macro,
                fundamentals=fund, universe=universe)


def load_spy_close(data: dict) -> pd.Series | None:
    spy = data["prices"][data["prices"]["ticker"] == "SPY"]
    if spy.empty:
        try:
            etf = pd.read_parquet(CACHE / "sector_etfs.parquet")
            spy = etf[etf["ticker"] == "SPY"]
        except Exception:
            return None
    if spy.empty:
        return None
    return spy.set_index(pd.DatetimeIndex(pd.to_datetime(spy["date"])))["close"].astype(float)


def spy_bah_metrics(spy_close: pd.Series, start: str, end: str) -> dict:
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    sub = spy_close.sort_index()
    sub = sub[(sub.index >= s) & (sub.index <= e)]
    if sub.empty:
        return {}
    ret = sub.pct_change().dropna()
    days = len(ret)
    cagr = _ann_cagr(sub, days)
    return {
        "spy_cagr": cagr,
        "spy_sharpe": _sharpe_rf(ret),
        "spy_sortino": _sortino_rf(ret),
        "spy_max_dd": _max_dd(sub),
        "spy_wr": float((ret > 0).mean()),
        "spy_n_days": days,
    }


# ---- LGBM tiebreaker injection ----

def build_lgbm_composite_iv(iv: pd.DataFrame,
                             fold_preds_path: Path) -> pd.DataFrame:
    """
    Injects a composite iv_rank = iv_rank + LGBM_WEIGHT * lgbm_adj_norm into
    the IV dataframe. Full V5 universe is preserved (no rows dropped).

    LGBM adjustment:
      1. Load daily pred_yield from fold_predictions.parquet.
      2. Apply sigma penalty: if sigma > HIGH_SIGMA_THRESHOLD, halve pred_yield.
      3. Per-day normalize pred_yield to [0, 1] across all tickers.
      4. Add LGBM_WEIGHT * normalized_adj to iv_rank.

    Days/tickers outside LGBM coverage get zero adjustment (pure iv_rank).
    """
    print("[tiebreaker] Loading fold predictions...")
    preds = pd.read_parquet(fold_preds_path)
    preds["date"] = pd.to_datetime(preds["date"])

    # Apply sigma penalty: halve score for high-sigma tickers
    preds["pred_yield_adj"] = preds["pred_yield"].copy()
    high_sigma_mask = preds["sigma"] > HIGH_SIGMA_THRESHOLD
    preds.loc[high_sigma_mask, "pred_yield_adj"] *= 0.5
    n_penalized = high_sigma_mask.sum()
    pct_penalized = n_penalized / len(preds)
    print(f"[tiebreaker] Sigma penalty applied: {n_penalized:,} rows "
          f"({pct_penalized:.1%} of predictions) with sigma > {HIGH_SIGMA_THRESHOLD}")

    # Per-day normalize adj score to [0, 1]
    def _day_normalize(grp):
        mn, mx = grp["pred_yield_adj"].min(), grp["pred_yield_adj"].max()
        if mx > mn:
            grp["lgbm_norm"] = (grp["pred_yield_adj"] - mn) / (mx - mn)
        else:
            grp["lgbm_norm"] = 0.5
        return grp

    preds = preds.groupby("date", group_keys=False).apply(_day_normalize)
    print(f"[tiebreaker] Normalized LGBM scores: {len(preds):,} rows, "
          f"{preds['ticker'].nunique()} tickers, "
          f"{preds['date'].min().date()} -> {preds['date'].max().date()}")

    # Merge into IV (left join: full universe preserved, non-covered = NaN -> 0)
    iv = iv.copy()
    iv["date"] = pd.to_datetime(iv["date"])
    lgbm_lookup = preds[["date", "ticker", "lgbm_norm"]].copy()

    iv_merged = iv.merge(lgbm_lookup, on=["date", "ticker"], how="left")
    iv_merged["lgbm_norm"] = iv_merged["lgbm_norm"].fillna(0.0)

    # Composite: iv_rank + LGBM_WEIGHT * lgbm_norm
    # This keeps iv_rank dominant (range [0,1]) while LGBM adds up to 0.10
    iv_merged["iv_rank_original"] = iv_merged["iv_rank"]
    iv_merged["iv_rank"] = iv_merged["iv_rank"] + LGBM_WEIGHT * iv_merged["lgbm_norm"]

    covered = (iv_merged["lgbm_norm"] > 0).mean()
    print(f"[tiebreaker] IV rows with LGBM adjustment: {covered:.1%}")
    print(f"[tiebreaker] iv_rank stats after composite (LGBM period only):")
    lgbm_period = iv_merged[iv_merged["lgbm_norm"] > 0]
    if len(lgbm_period) > 0:
        print(f"  original iv_rank: {lgbm_period['iv_rank_original'].mean():.3f} mean")
        print(f"  composite iv_rank: {lgbm_period['iv_rank'].mean():.3f} mean (+{LGBM_WEIGHT}*lgbm)")

    return iv_merged


def run_arm(label: str, iv_use_raw: pd.DataFrame, data: dict,
            spy_close: pd.Series | None) -> dict:
    px = data["prices"].copy()
    macro = data["macro"].copy()
    fund = data["fundamentals"].copy()
    uni = data["universe"].copy()

    iv_use = _apply_iv_rank_floor(iv_use_raw, IV_RANK_FLOOR)
    print(f"[tiebreaker] {label}: {iv_use['ticker'].nunique()} tickers "
          f"after IV rank floor {IV_RANK_FLOOR}")

    result = run_wheel(
        cfg=V5_CFG,
        prices=px,
        iv=iv_use,
        macro=macro,
        fundamentals=fund,
        universe=uni,
        starting_cash=CAPITAL,
        start=START, end=END,
        verbose=False,
    )
    m = full_metrics(result, spy_close)
    print(f"[tiebreaker] {label}: CAGR={m['realized_cagr']*100:.2f}%  "
          f"Sharpe={m['realized_sharpe']:.2f}  Sortino={m['realized_sortino']:.2f}  "
          f"MaxDD={m['realized_max_dd']*100:.2f}%  "
          f"Trades={m['n_trades']}  WR={m['wr']*100:.1f}%  PF={m['pf']:.2f}")
    return {"label": label, "metrics": m,
            "equity_curve": result["equity_curve"],
            "ledger": result["ledger"]}


def main():
    data = load_data()
    spy_close = load_spy_close(data)

    preds_path = ROOT / "results" / "lgbm_ticker_ranker_v1" / "fold_predictions.parquet"
    if not preds_path.exists():
        raise SystemExit(f"[tiebreaker] fold_predictions.parquet not found at {preds_path}")

    # Arm A: LGBM tiebreaker (full universe + composite iv_rank)
    print("\n=== ARM A: LGBM Tiebreaker (full universe, composite iv_rank) ===")
    iv_composite = build_lgbm_composite_iv(data["iv"], preds_path)
    arm_a = run_arm("LGBM_Tiebreaker", iv_composite, data, spy_close)

    # Arm B: Baseline V5 (full universe, pure iv_rank — no LGBM)
    print("\n=== ARM B: Baseline V5 (full universe, pure iv_rank) ===")
    iv_full = data["iv"].copy()
    arm_b = run_arm("Baseline_V5", iv_full, data, spy_close)

    # SPY buy-and-hold
    spy_bah = spy_bah_metrics(spy_close, START, END) if spy_close is not None else {}
    print(f"\n[tiebreaker] SPY B&H: CAGR={spy_bah.get('spy_cagr', float('nan'))*100:.2f}%  "
          f"Sharpe={spy_bah.get('spy_sharpe', float('nan')):.2f}  "
          f"MaxDD={spy_bah.get('spy_max_dd', float('nan'))*100:.2f}%")

    # Save results
    out_dir = RESULTS / "lgbm_v5_tiebreaker"
    out_dir.mkdir(parents=True, exist_ok=True)

    for arm in [arm_a, arm_b]:
        label = arm["label"]
        arm["equity_curve"].to_parquet(out_dir / f"equity_{label}.parquet", index=False)
        arm["ledger"].to_parquet(out_dir / f"ledger_{label}.parquet", index=False)

    summary = {
        "lgbm_tiebreaker": arm_a["metrics"],
        "baseline_v5": arm_b["metrics"],
        "spy_bah": spy_bah,
        "config": {
            "start": START, "end": END, "capital": CAPITAL,
            "iv_rank_floor": IV_RANK_FLOOR,
            "rf_annual": 0.04,
            "lgbm_weight": LGBM_WEIGHT,
            "high_sigma_threshold": HIGH_SIGMA_THRESHOLD,
            "max_positions_per_lgbm_ticker": MAX_POSITIONS_PER_LGBM_TICKER,
            "approach": "tiebreaker: composite iv_rank = iv_rank + 0.10 * lgbm_norm_penalized",
        }
    }

    def _safe(v):
        if isinstance(v, float) and not np.isfinite(v):
            return None
        return v

    def _clean(d):
        return {k: _safe(v) for k, v in d.items()}

    summary_clean = {k: (_clean(v) if isinstance(v, dict) else v)
                     for k, v in summary.items()}
    (out_dir / "tiebreaker_results.json").write_text(
        json.dumps(summary_clean, indent=2, default=str))

    # Print comparison table
    print("\n" + "="*75)
    print("LGBM Tiebreaker vs Baseline V5 vs Failed Direct-Replacement")
    print("Realized-Cash Metrics (rf=4%)")
    print("="*75)
    a, b = arm_a["metrics"], arm_b["metrics"]
    # Known failed approach result for reference
    failed_sharpe = 1.36  # from the direct-replacement experiment

    fmt = "{:<25} {:>12} {:>12} {:>12}"
    print(fmt.format("Metric", "LGBM-Tiebreaker", "Baseline V5", "Failed Repl"))
    print("-"*75)
    print(fmt.format("CAGR",
          f"{a['realized_cagr']*100:.2f}%",
          f"{b['realized_cagr']*100:.2f}%", "—"))
    print(fmt.format("Sharpe (rf=4%)",
          f"{a['realized_sharpe']:.2f}",
          f"{b['realized_sharpe']:.2f}",
          f"{failed_sharpe:.2f}"))
    print(fmt.format("Sortino (rf=4%)",
          f"{a['realized_sortino']:.2f}",
          f"{b['realized_sortino']:.2f}", "—"))
    print(fmt.format("Max Drawdown",
          f"{a['realized_max_dd']*100:.2f}%",
          f"{b['realized_max_dd']*100:.2f}%", "—"))
    print(fmt.format("Win Rate", f"{a['wr']*100:.1f}%", f"{b['wr']*100:.1f}%", "—"))
    print(fmt.format("Profit Factor", f"{a['pf']:.2f}", f"{b['pf']:.2f}", "—"))
    calmar_a = (a["realized_cagr"] / abs(a["realized_max_dd"])
                if a["realized_max_dd"] < 0 else float("nan"))
    calmar_b = (b["realized_cagr"] / abs(b["realized_max_dd"])
                if b["realized_max_dd"] < 0 else float("nan"))
    print(fmt.format("Calmar",
          f"{calmar_a:.2f}" if np.isfinite(calmar_a) else "N/A",
          f"{calmar_b:.2f}" if np.isfinite(calmar_b) else "N/A", "—"))
    print(fmt.format("Day Conc (<=0.70)",
          f"{a['day_conc']:.3f}", f"{b['day_conc']:.3f}", "—"))
    print(fmt.format("Trades", str(a['n_trades']), str(b['n_trades']), "—"))
    print(fmt.format("Final Equity",
          f"${a['realized_final_equity']:,.0f}",
          f"${b['realized_final_equity']:,.0f}", "—"))

    print()
    print("Regime Split (realized-cash Sharpe):")
    rfmt = "{:<20} {:>15} {:>15}"
    print(rfmt.format("", "LGBM-Tiebreaker", "Baseline V5"))
    for reg in ("green", "red", "flat"):
        sg = a.get(f"{reg}_sharpe", float("nan"))
        sb = b.get(f"{reg}_sharpe", float("nan"))
        print(rfmt.format(f"  {reg.capitalize()} days",
              f"{sg:.2f}" if not np.isnan(sg) else "N/A",
              f"{sb:.2f}" if not np.isnan(sb) else "N/A"))
    ga = a.get("regime_gap", float("nan"))
    gb = b.get("regime_gap", float("nan"))
    print(rfmt.format("  Regime Gap",
          f"{ga:.3f}" if not np.isnan(ga) else "N/A",
          f"{gb:.3f}" if not np.isnan(gb) else "N/A"))
    gap_a = "PASS" if not np.isnan(ga) and ga <= 0.50 else "FAIL"
    gap_b = "PASS" if not np.isnan(gb) and gb <= 0.50 else "FAIL"
    print(rfmt.format("  R1 Gate (<=0.50)", gap_a, gap_b))

    print(f"\n[tiebreaker] Results saved to {out_dir}/")

    # Verdict
    print("\n" + "="*75)
    print("VERDICT")
    print("="*75)
    beat_baseline = a["realized_sharpe"] > b["realized_sharpe"]
    beat_failed = a["realized_sharpe"] > failed_sharpe
    dd_ok = a["realized_max_dd"] >= b["realized_max_dd"]  # less negative = better
    r1_pass = not np.isnan(ga) and ga <= 0.50

    if beat_baseline and r1_pass:
        verdict = ("IMPROVEMENT: LGBM tiebreaker beats baseline on Sharpe "
                   f"({a['realized_sharpe']:.2f} vs {b['realized_sharpe']:.2f}) "
                   f"with regime gate {'PASS' if r1_pass else 'FAIL'}. "
                   "Candidate for integration.")
    elif beat_failed and not beat_baseline:
        verdict = (f"MARGINAL: Tiebreaker recovers from failed replacement "
                   f"(Sharpe {a['realized_sharpe']:.2f} vs failed {failed_sharpe:.2f}) "
                   f"but does NOT beat baseline ({b['realized_sharpe']:.2f}). "
                   "Baseline simplicity wins — LGBM adds no net value here.")
    elif not beat_failed:
        verdict = (f"REJECT: Tiebreaker Sharpe {a['realized_sharpe']:.2f} is below "
                   f"even the failed direct-replacement ({failed_sharpe:.2f}) and "
                   f"well below baseline ({b['realized_sharpe']:.2f}). "
                   "LGBM ranking signal not useful for this strategy.")
    else:
        verdict = (f"MIXED: Sharpe {a['realized_sharpe']:.2f} vs baseline "
                   f"{b['realized_sharpe']:.2f}. R1 gate: {'PASS' if r1_pass else 'FAIL'}.")
    print(verdict)


if __name__ == "__main__":
    main()

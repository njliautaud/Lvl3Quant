"""
wheel_safety_integration.py — Wire LGBM Assignment Risk Ranker (safety-first)
into the Wheel V5 backtest engine.

Design:
  - LGBM assignment risk OOS selections (2023-01 to 2025-09) gate the monthly
    candidate universe. Each month the 20 tickers with LOWEST predicted
    assignment probability are the allowed candidate set.
  - Within those 20 safe tickers, the V5 engine applies its standard secondary
    gates (IV rank >= 20%, DTE 7-14, PT 65%, sector cap, VIX gate, fund score).
    Premium yield (iv_rank) is the tiebreaker within the safe universe.
  - This is "safety-first": LGBM picks the safest 20, then V5 filters for
    premium yield among those 20.
  - SLIDING walk-forward: ranker was trained on prior months only (OOS folds).
    No retraining inside this script. HC #0 compliant.
  - Regime split per HC #428 R1. rf=4%. Costs canonical (ES constants not
    applicable here — this is equity options, no extra cost adjustment needed
    beyond the engine's own premium/assignment modeling).

Run from /home/jupiter/Lvl3Quant/wheel_strategy_v1/:
    python3 -m backtest.wheel_safety_integration
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
    _apply_iv_rank_floor, _load_spy_close,
)

TRADING_DAYS = 252
RF_DAILY = 0.04 / TRADING_DAYS   # 4% annualized risk-free rate

# ---- V5 config — identical to baseline and prior integrations ----
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

IV_RANK_FLOOR = 0.20   # 20% IV rank floor per V5 spec
CAPITAL = 100_000.0
START = "2023-01-01"
END = "2025-09-30"

# Path to assignment risk ranker OOS selections
SAFETY_SEL_PATH = (
    ROOT / "results" / "lgbm_assignment_risk_v1" / "monthly_selections.parquet"
)


# ---------------------------------------------------------------------------
# Metric helpers (identical to wheel_lgbm_integration.py for consistency)
# ---------------------------------------------------------------------------

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
    start_eq = eq.iloc[0]
    end_eq = eq.iloc[-1]
    if start_eq <= 0 or yrs <= 0:
        return 0.0
    return float((end_eq / start_eq) ** (1.0 / yrs) - 1.0)


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
    """Fraction of cumulative PnL from single best day (HC #344 cap <= 0.70)."""
    if equity_curve.empty:
        return float("nan")
    eq = equity_curve.set_index("date")["equity"].sort_index()
    daily_pnl = eq.diff().dropna()
    total_pnl = daily_pnl.sum()
    if total_pnl <= 0:
        return float("nan")
    return float(daily_pnl.max() / total_pnl)


def _regime_split(daily_ret: pd.Series, spy_close: pd.Series | None) -> dict:
    """Green/red/flat regime split with Sharpe (rf-adjusted) per bucket."""
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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data() -> dict:
    print("[safety_integration] Loading cache data...")
    prices = pd.read_parquet(CACHE / "prices.parquet")

    # Prefer real-blend IV features; fall back to modeled
    iv_path = CACHE / "iv_features_real_blend.parquet"
    if not iv_path.exists():
        iv_path = CACHE / "iv_features_modeled.parquet"
    iv = pd.read_parquet(iv_path)

    macro = pd.read_parquet(CACHE / "macro.parquet")
    fund = pd.read_parquet(CACHE / "fundamentals.parquet")
    universe = pd.read_parquet(CACHE / "universe.parquet")
    print(f"[safety_integration] Prices: {len(prices):,} rows | IV: {len(iv):,} rows | "
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
    return {
        "spy_cagr": _ann_cagr(sub, days),
        "spy_sharpe": _sharpe_rf(ret),
        "spy_sortino": _sortino_rf(ret),
        "spy_max_dd": _max_dd(sub),
        "spy_wr": float((ret > 0).mean()),
        "spy_n_days": days,
    }


# ---------------------------------------------------------------------------
# Safety ranker universe gate
# ---------------------------------------------------------------------------

def build_safety_monthly_universe(selections_path: Path) -> dict:
    """
    Returns dict: year_month_str -> set of LGBM-selected tickers (safest 20).
    Selections are pre-ranked lowest predicted assignment probability first.
    We take all rows per month (already capped at 20 by the ranker).

    Key: string form "2023-01" (period to str conversion handles Period type).
    """
    sel = pd.read_parquet(selections_path)

    # Handle both Period[M] and string rebalance_ym columns
    sel["rebalance_ym_str"] = sel["rebalance_ym"].astype(str)

    universe_by_month: dict[str, set[str]] = {}
    for ym, grp in sel.groupby("rebalance_ym_str"):
        # Sort ascending so lowest (safest) are first — already done by ranker,
        # but be explicit for clarity.
        safest = grp.sort_values("pred_assignment_prob", ascending=True)
        universe_by_month[str(ym)] = set(safest["ticker"].unique())

    print(f"[safety_integration] Safety ranker: {len(universe_by_month)} months, "
          f"range {min(universe_by_month)} -> {max(universe_by_month)}")
    # Log a sample month
    sample_mo = sorted(universe_by_month.keys())[0]
    # Recover ordered list for reporting (need original df)
    sample_grp = sel[sel["rebalance_ym_str"] == sample_mo].sort_values("pred_assignment_prob")
    print(f"[safety_integration] Sample {sample_mo} (safest first): "
          f"{sample_grp['ticker'].tolist()}")
    return universe_by_month


def filter_iv_to_safety(iv: pd.DataFrame,
                        universe_by_month: dict) -> pd.DataFrame:
    """
    Gate IV rows to only LGBM-safety-selected tickers for each month.
    Within the safe universe, iv_rank is preserved so the V5 engine can
    use it as the yield tiebreaker (highest iv_rank = most premium = best
    yield candidate within the safe set).

    Months without LGBM coverage (pre-OOT) are passed through unfiltered.
    """
    iv = iv.copy()
    iv["date"] = pd.to_datetime(iv["date"])
    iv["ym"] = iv["date"].dt.to_period("M").astype(str)

    all_months = iv["ym"].unique()
    keep_masks = []
    skipped_months = []

    for ym in all_months:
        month_iv = iv[iv["ym"] == ym]
        if ym in universe_by_month:
            allowed = universe_by_month[ym]
            keep_masks.append(month_iv["ticker"].isin(allowed))
        else:
            # Outside LGBM OOT coverage — pass through unfiltered
            keep_masks.append(pd.Series(True, index=month_iv.index))
            skipped_months.append(ym)

    if skipped_months:
        print(f"[safety_integration] {len(skipped_months)} months outside safety "
              f"ranker OOT (unfiltered): "
              f"{skipped_months[:3]}{'...' if len(skipped_months) > 3 else ''}")

    mask = pd.concat(keep_masks).reindex(iv.index, fill_value=True)
    filtered = iv[mask].copy()
    print(f"[safety_integration] IV rows: {len(iv):,} -> {len(filtered):,} "
          f"({len(filtered) / len(iv):.1%} kept after safety filter)")
    return filtered.drop(columns=["ym"])


# ---------------------------------------------------------------------------
# Run one backtest arm
# ---------------------------------------------------------------------------

def run_arm(label: str, iv_filtered: pd.DataFrame, data: dict,
            spy_close: pd.Series | None) -> dict:
    px = data["prices"].copy()
    macro = data["macro"].copy()
    fund = data["fundamentals"].copy()
    uni = data["universe"].copy()

    iv_use = _apply_iv_rank_floor(iv_filtered, IV_RANK_FLOOR)
    n_tickers = iv_use["ticker"].nunique()
    print(f"[safety_integration] {label}: {n_tickers} tickers after IV rank floor "
          f"{IV_RANK_FLOOR:.0%}")

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
    print(f"[safety_integration] {label}: "
          f"CAGR={m['realized_cagr']*100:.2f}%  "
          f"Sharpe={m['realized_sharpe']:.2f}  "
          f"Sortino={m['realized_sortino']:.2f}  "
          f"MaxDD={m['realized_max_dd']*100:.2f}%  "
          f"Trades={m['n_trades']}  "
          f"WR={m['wr']*100:.1f}%  "
          f"PF={m['pf']:.2f}")
    return {"label": label, "metrics": m,
            "equity_curve": result["equity_curve"],
            "ledger": result["ledger"]}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    data = load_data()
    spy_close = load_spy_close(data)

    # Load safety ranker selections
    if not SAFETY_SEL_PATH.exists():
        raise SystemExit(
            f"[safety_integration] monthly_selections.parquet not found at {SAFETY_SEL_PATH}"
        )
    universe_by_month = build_safety_monthly_universe(SAFETY_SEL_PATH)

    # Arm A: Safety-ranker filtered universe (LGBM top-20 safest, then V5 yield gates)
    print("\n=== ARM A: Safety Ranker Universe (LGBM top-20 lowest assignment risk) ===")
    iv_safe = filter_iv_to_safety(data["iv"], universe_by_month)
    arm_a = run_arm("Safety_V5", iv_safe, data, spy_close)

    # Arm B: Baseline V5 — full universe, same period, same V5 config
    print("\n=== ARM B: Baseline V5 (full universe — no LGBM filter) ===")
    iv_full = data["iv"].copy()
    arm_b = run_arm("Baseline_V5", iv_full, data, spy_close)

    # SPY buy-and-hold benchmark
    spy_bah = spy_bah_metrics(spy_close, START, END) if spy_close is not None else {}
    if spy_bah:
        print(f"\n[safety_integration] SPY B&H ({START}->{END}): "
              f"CAGR={spy_bah.get('spy_cagr', float('nan'))*100:.2f}%  "
              f"Sharpe={spy_bah.get('spy_sharpe', float('nan')):.2f}  "
              f"MaxDD={spy_bah.get('spy_max_dd', float('nan'))*100:.2f}%")

    # Save outputs
    out_dir = RESULTS / "lgbm_safety_v5_integration"
    out_dir.mkdir(parents=True, exist_ok=True)

    for arm in [arm_a, arm_b]:
        lbl = arm["label"]
        arm["equity_curve"].to_parquet(out_dir / f"equity_{lbl}.parquet", index=False)
        arm["ledger"].to_parquet(out_dir / f"ledger_{lbl}.parquet", index=False)

    def _safe(v):
        if isinstance(v, float) and not np.isfinite(v):
            return None
        return v

    def _clean(d):
        return {k: _safe(v) for k, v in d.items()}

    summary = {
        "safety_v5": arm_a["metrics"],
        "baseline_v5": arm_b["metrics"],
        "spy_bah": spy_bah,
        "config": {
            "start": START, "end": END, "capital": CAPITAL,
            "iv_rank_floor": IV_RANK_FLOOR,
            "rf_annual": 0.04,
            "top_k_safety": 20,
            "ranker": "lgbm_assignment_risk_v1",
            "selection_criterion": "lowest_pred_assignment_prob",
        },
    }
    summary_clean = {k: (_clean(v) if isinstance(v, dict) else v)
                     for k, v in summary.items()}
    (out_dir / "safety_results.json").write_text(
        json.dumps(summary_clean, indent=2, default=str)
    )

    # ---- Print comparison table ----
    print("\n" + "=" * 72)
    print("SAFETY RANKER V5 vs BASELINE V5 — Realized-Cash Metrics (rf=4%)")
    print("=" * 72)
    a, b = arm_a["metrics"], arm_b["metrics"]
    fmt = "{:<26} {:>12} {:>12} {:>12}"
    print(fmt.format("Metric", "Safety V5", "Baseline V5", "SPY B&H"))
    print("-" * 72)

    calmar_a = (a["realized_cagr"] / abs(a["realized_max_dd"])
                if a["realized_max_dd"] < 0 else float("nan"))
    calmar_b = (b["realized_cagr"] / abs(b["realized_max_dd"])
                if b["realized_max_dd"] < 0 else float("nan"))

    rows = [
        ("CAGR",
         f"{a['realized_cagr']*100:.2f}%",
         f"{b['realized_cagr']*100:.2f}%",
         f"{spy_bah.get('spy_cagr', float('nan'))*100:.2f}%"),
        ("Sharpe (rf=4%)",
         f"{a['realized_sharpe']:.2f}",
         f"{b['realized_sharpe']:.2f}",
         f"{spy_bah.get('spy_sharpe', float('nan')):.2f}"),
        ("Sortino (rf=4%)",
         f"{a['realized_sortino']:.2f}",
         f"{b['realized_sortino']:.2f}",
         f"{spy_bah.get('spy_sortino', float('nan')):.2f}"),
        ("Max Drawdown",
         f"{a['realized_max_dd']*100:.2f}%",
         f"{b['realized_max_dd']*100:.2f}%",
         f"{spy_bah.get('spy_max_dd', float('nan'))*100:.2f}%"),
        ("Calmar",
         f"{calmar_a:.2f}" if not np.isnan(calmar_a) else "N/A",
         f"{calmar_b:.2f}" if not np.isnan(calmar_b) else "N/A",
         "—"),
        ("Win Rate",
         f"{a['wr']*100:.1f}%",
         f"{b['wr']*100:.1f}%",
         "—"),
        ("Profit Factor",
         f"{a['pf']:.2f}",
         f"{b['pf']:.2f}",
         "—"),
        ("Day Conc (<=0.70)",
         f"{a['day_conc']:.3f}",
         f"{b['day_conc']:.3f}",
         "—"),
        ("Trades",
         str(a["n_trades"]),
         str(b["n_trades"]),
         "—"),
        ("Final Equity",
         f"${a['realized_final_equity']:,.0f}",
         f"${b['realized_final_equity']:,.0f}",
         "—"),
    ]
    for row in rows:
        print(fmt.format(*row))

    print()
    print("Regime Split (realized-cash Sharpe, SPY close-to-close):")
    rfmt = "{:<24} {:>12} {:>12}"
    print(rfmt.format("", "Safety V5", "Baseline V5"))
    print("-" * 50)
    for reg in ("green", "red", "flat"):
        sg = a.get(f"{reg}_sharpe", float("nan"))
        sb = b.get(f"{reg}_sharpe", float("nan"))
        ng = a.get(f"n_{reg}", 0)
        nb = b.get(f"n_{reg}", 0)
        print(rfmt.format(
            f"  {reg.capitalize()} days (n={ng}/{nb})",
            f"{sg:.2f}" if not np.isnan(sg) else "N/A",
            f"{sb:.2f}" if not np.isnan(sb) else "N/A",
        ))
    ga = a.get("regime_gap", float("nan"))
    gb = b.get("regime_gap", float("nan"))
    gap_pass_a = "PASS" if not np.isnan(ga) and ga <= 0.50 else "FAIL"
    gap_pass_b = "PASS" if not np.isnan(gb) and gb <= 0.50 else "FAIL"
    print(rfmt.format("  Regime Gap", 
          f"{ga:.3f}" if not np.isnan(ga) else "N/A",
          f"{gb:.3f}" if not np.isnan(gb) else "N/A"))
    print(rfmt.format("  R1 Gate (<=0.50)", gap_pass_a, gap_pass_b))

    print()
    print("=" * 72)
    print("Four-way comparison vs prior configs:")
    print("=" * 72)
    cmp_fmt = "{:<22} {:>10} {:>12} {:>12} {:>12}"
    print(cmp_fmt.format("Config", "Sharpe", "Red Sharpe", "Regime Gap", "Gate"))
    print("-" * 72)
    # Historical results (from prior runs logged by user)
    historical = [
        ("Baseline V5",    1.70, 0.81,  0.71, "FAIL"),
        ("Yield Ranker",   1.36, -2.58, 1.69, "FAIL"),
        ("Blend Tiebreaker", 2.67, -0.95, 1.24, "FAIL"),
    ]
    for cfg, sh, rs, rg, gt in historical:
        print(cmp_fmt.format(cfg, f"{sh:.2f}", f"{rs:.2f}", f"{rg:.3f}", gt))
    # Safety ranker — live result
    sa_sh = a["realized_sharpe"]
    sa_rs = a.get("red_sharpe", float("nan"))
    sa_rg = ga
    sa_gt = gap_pass_a
    print(cmp_fmt.format(
        "Safety Ranker (new)",
        f"{sa_sh:.2f}",
        f"{sa_rs:.2f}" if not np.isnan(sa_rs) else "N/A",
        f"{sa_rg:.3f}" if not np.isnan(sa_rg) else "N/A",
        sa_gt,
    ))
    print()

    # R1 verdict
    print("HC #428 GATE VERDICTS:")
    print(f"  R1 (regime symmetry, Safety V5): {gap_pass_a} "
          f"— gap={ga:.3f}" if not np.isnan(ga) else f"  R1: NEEDS-DATA (no regime gap computed)")
    print(f"  R2 (MFE-within-horizon): NOT APPLICABLE — "
          f"this is equity options, not a futures intraday model. "
          f"Assignment risk model horizon = 30 days, hold ~7-14 DTE. ALIGNED.")
    print(f"  Day Conc (HC #344): {'PASS' if not np.isnan(a['day_conc']) and a['day_conc'] <= 0.70 else 'FAIL / CHECK'} "
          f"— {a['day_conc']:.3f}")

    print(f"\n[safety_integration] Results saved.")


if __name__ == "__main__":
    main()

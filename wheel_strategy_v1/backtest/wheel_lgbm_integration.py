"""
wheel_lgbm_integration.py — Wire LGBM ticker ranker OOS selections into the
Wheel V5 backtest engine and compare against the V5 baseline.

Design:
  - LGBM monthly selections from OOS period (2023-01 to 2025-09) are used as
    the allowed ticker universe each month instead of the static fund_score gate.
  - V5 config: put_delta=0.35, DTE 7-14, IV rank >= 20%, PT 65%, sector ON.
  - Baseline: same config + same period but using the full universe (fund_score gate only).
  - All metrics include regime split (HC #428 R1) with rf=4% in Sharpe.
  - SLIDING walk-forward: ranker already trained on prior data — we use its OOS
    predictions directly. No retraining inside this script (HC #0 compliant).

Run from /home/jupiter/Lvl3Quant/wheel_strategy_v1/:
    python3 -m backtest.wheel_lgbm_integration
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
RF_DAILY = 0.04 / TRADING_DAYS   # 4% annualized risk-free rate

# ---- V5 config (matches v5_combined from sweep) ----
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


# ---- helpers ----

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
    """Fraction of cumulative PnL from the single best day (HC #344 cap <= 0.70)."""
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

    # Realized cash curve (account view, not MTM)
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
    print("[lgbm_integration] Loading cache data...")
    prices   = pd.read_parquet(CACHE / "prices.parquet")
    iv       = pd.read_parquet(CACHE / "iv_features_real_blend.parquet")
    if not (CACHE / "iv_features_real_blend.parquet").exists():
        iv = pd.read_parquet(CACHE / "iv_features_modeled.parquet")
    macro    = pd.read_parquet(CACHE / "macro.parquet")
    fund     = pd.read_parquet(CACHE / "fundamentals.parquet")
    universe = pd.read_parquet(CACHE / "universe.parquet")
    print(f"[lgbm_integration] Prices: {len(prices):,} rows | IV: {len(iv):,} rows | "
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
    """SPY buy-and-hold over the same period."""
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


# ---- LGBM monthly universe gate ----

def build_lgbm_monthly_universe(selections_path: Path) -> dict:
    """
    Returns dict: year_month_str -> set of LGBM-selected tickers.
    E.g. {"2023-01": {"AAPL", "MSFT", ...}, ...}
    """
    sel = pd.read_parquet(selections_path)
    sel["rebalance_ym"] = sel["rebalance_ym"].astype(str)
    universe_by_month = {}
    for ym, grp in sel.groupby("rebalance_ym"):
        universe_by_month[ym] = set(grp["ticker"].unique())
    return universe_by_month


def filter_iv_to_lgbm(iv: pd.DataFrame,
                      universe_by_month: dict) -> pd.DataFrame:
    """
    For each date in iv, keep only tickers that were in the LGBM selection for
    that calendar month. This is the hook: by zeroing out IV rows for non-selected
    tickers, the wheel engine's candidate loop naturally skips them.

    The monthly selection is determined at the START of each month and held through
    to end-of-month (rebalance once per month, exactly what the ranker was trained to do).
    """
    iv = iv.copy()
    iv["date"] = pd.to_datetime(iv["date"])
    iv["ym"] = iv["date"].dt.to_period("M").astype(str)

    # Build mask: True = keep row
    all_months = iv["ym"].unique()
    keep_masks = []
    skipped_months = []
    for ym in all_months:
        month_iv = iv[iv["ym"] == ym]
        if ym in universe_by_month:
            allowed = universe_by_month[ym]
            keep_masks.append(month_iv["ticker"].isin(allowed))
        else:
            # Month not in LGBM selections (pre-OOT period) — keep all
            keep_masks.append(pd.Series(True, index=month_iv.index))
            skipped_months.append(ym)

    if skipped_months:
        print(f"[lgbm_integration] {len(skipped_months)} months outside LGBM OOT "
              f"(no filter applied): {skipped_months[:3]}{'...' if len(skipped_months)>3 else ''}")

    mask = pd.concat(keep_masks).reindex(iv.index, fill_value=True)
    filtered = iv[mask].copy()
    print(f"[lgbm_integration] IV rows: {len(iv):,} -> {len(filtered):,} "
          f"({len(filtered)/len(iv):.1%} kept after LGBM filter)")
    return filtered.drop(columns=["ym"])


# ---- main ----

def run_arm(label: str, iv_filtered: pd.DataFrame, data: dict,
            spy_close: pd.Series | None) -> dict:
    px = data["prices"].copy()
    macro = data["macro"].copy()
    fund = data["fundamentals"].copy()
    uni = data["universe"].copy()

    iv_use = _apply_iv_rank_floor(iv_filtered, IV_RANK_FLOOR)
    print(f"[lgbm_integration] {label}: {iv_use['ticker'].nunique()} tickers "
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
    print(f"[lgbm_integration] {label}: CAGR={m['realized_cagr']*100:.2f}%  "
          f"Sharpe={m['realized_sharpe']:.2f}  Sortino={m['realized_sortino']:.2f}  "
          f"MaxDD={m['realized_max_dd']*100:.2f}%  "
          f"Trades={m['n_trades']}  WR={m['wr']*100:.1f}%  PF={m['pf']:.2f}")
    return {"label": label, "metrics": m,
            "equity_curve": result["equity_curve"],
            "ledger": result["ledger"]}


def main():
    data = load_data()
    spy_close = load_spy_close(data)

    # Load LGBM selections
    sel_path = ROOT / "results" / "lgbm_ticker_ranker_v1" / "monthly_selections.parquet"
    if not sel_path.exists():
        raise SystemExit(f"[lgbm_integration] monthly_selections.parquet not found at {sel_path}")
    universe_by_month = build_lgbm_monthly_universe(sel_path)
    print(f"[lgbm_integration] LGBM OOT selections: {len(universe_by_month)} months, "
          f"range {min(universe_by_month)} -> {max(universe_by_month)}")
    sample_mo = list(universe_by_month.keys())[0]
    print(f"[lgbm_integration] Sample ({sample_mo}): {sorted(universe_by_month[sample_mo])[:10]}...")

    # Arm A: LGBM-filtered universe
    print("\n=== ARM A: LGBM-Selected Universe ===")
    iv_lgbm = filter_iv_to_lgbm(data["iv"], universe_by_month)
    arm_a = run_arm("LGBM_V5", iv_lgbm, data, spy_close)

    # Arm B: Baseline V5 (full universe, same period, same config)
    print("\n=== ARM B: Baseline V5 (full universe) ===")
    iv_full = data["iv"].copy()
    arm_b = run_arm("Baseline_V5", iv_full, data, spy_close)

    # SPY buy-and-hold
    spy_bah = spy_bah_metrics(spy_close, START, END) if spy_close is not None else {}
    print(f"\n[lgbm_integration] SPY B&H ({START}->{END}): "
          f"CAGR={spy_bah.get('spy_cagr',float('nan'))*100:.2f}%  "
          f"Sharpe={spy_bah.get('spy_sharpe',float('nan')):.2f}  "
          f"MaxDD={spy_bah.get('spy_max_dd',float('nan'))*100:.2f}%")

    # Save results
    out_dir = RESULTS / "lgbm_v5_integration"
    out_dir.mkdir(parents=True, exist_ok=True)

    for arm in [arm_a, arm_b]:
        label = arm["label"]
        arm["equity_curve"].to_parquet(out_dir / f"equity_{label}.parquet", index=False)
        arm["ledger"].to_parquet(out_dir / f"ledger_{label}.parquet", index=False)

    summary = {
        "lgbm_v5": arm_a["metrics"],
        "baseline_v5": arm_b["metrics"],
        "spy_bah": spy_bah,
        "config": {
            "start": START, "end": END, "capital": CAPITAL,
            "iv_rank_floor": IV_RANK_FLOOR,
            "rf_annual": 0.04,
            "top_k_lgbm": 20,
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
    (out_dir / "integration_results.json").write_text(
        json.dumps(summary_clean, indent=2, default=str))

    # Print comparison table
    print("\n" + "="*70)
    print("LGBM V5 vs Baseline V5 — Realized-Cash Metrics (rf=4%)")
    print("="*70)
    a, b = arm_a["metrics"], arm_b["metrics"]
    fmt = "{:<25} {:>12} {:>12} {:>12}"
    print(fmt.format("Metric", "LGBM V5", "Baseline V5", "SPY B&H"))
    print("-"*70)
    print(fmt.format("CAGR",
          f"{a['realized_cagr']*100:.2f}%",
          f"{b['realized_cagr']*100:.2f}%",
          f"{spy_bah.get('spy_cagr',float('nan'))*100:.2f}%"))
    print(fmt.format("Sharpe (rf=4%)",
          f"{a['realized_sharpe']:.2f}",
          f"{b['realized_sharpe']:.2f}",
          f"{spy_bah.get('spy_sharpe',float('nan')):.2f}"))
    print(fmt.format("Sortino (rf=4%)",
          f"{a['realized_sortino']:.2f}",
          f"{b['realized_sortino']:.2f}",
          f"{spy_bah.get('spy_sortino',float('nan')):.2f}"))
    print(fmt.format("Max Drawdown",
          f"{a['realized_max_dd']*100:.2f}%",
          f"{b['realized_max_dd']*100:.2f}%",
          f"{spy_bah.get('spy_max_dd',float('nan'))*100:.2f}%"))
    print(fmt.format("Win Rate", f"{a['wr']*100:.1f}%", f"{b['wr']*100:.1f}%", "—"))
    print(fmt.format("Profit Factor", f"{a['pf']:.2f}", f"{b['pf']:.2f}", "—"))
    print(fmt.format("Day Conc (<=0.70)",
          f"{a['day_conc']:.3f}", f"{b['day_conc']:.3f}", "—"))
    print(fmt.format("Trades", str(a['n_trades']), str(b['n_trades']), "—"))
    print(fmt.format("Final Equity",
          f"${a['realized_final_equity']:,.0f}",
          f"${b['realized_final_equity']:,.0f}", "—"))
    print()
    print("Regime Split (realized-cash Sharpe):")
    rfmt = "{:<20} {:>12} {:>12}"
    print(rfmt.format("", "LGBM V5", "Baseline V5"))
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
    gap_pass = "PASS" if not np.isnan(ga) and ga <= 0.50 else "FAIL"
    gap_pass_b = "PASS" if not np.isnan(gb) and gb <= 0.50 else "FAIL"
    print(rfmt.format("  R1 Gate (<=0.50)", gap_pass, gap_pass_b))
    print()
    calmar_a = a["realized_cagr"] / abs(a["realized_max_dd"]) if a["realized_max_dd"] < 0 else float("nan")
    calmar_b = b["realized_cagr"] / abs(b["realized_max_dd"]) if b["realized_max_dd"] < 0 else float("nan")
    print(fmt.format("Calmar",
          f"{calmar_a:.2f}" if not np.isnan(calmar_a) else "N/A",
          f"{calmar_b:.2f}" if not np.isnan(calmar_b) else "N/A", "—"))
    print()
    print(f"[lgbm_integration] Results saved to {out_dir}/")


if __name__ == "__main__":
    main()

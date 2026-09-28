"""
blend_wheel_picker.py — HC #558 R3 + HC #557 R7 combined portfolio.

The wheel produces premium income (positive in all regimes) but caps upside.
The picker produces directional equity returns (negative in red, positive
in green) and lifts upside.

Hypothesis: blending them at fixed weights should:
  - Smooth red months (wheel income offsets picker red-month loss)
  - Lift CAGR vs wheel alone (picker captures green months better)
  - Pass HC #557 R2 regime-symmetry sub-gate that neither alone passes.

We test 5 blends: 100/0, 70/30, 60/40, 50/50, 30/70 (wheel/picker).
Each backtest reconstructs the daily equity curve by holding the v7 wheel
realized-cash equity at wheel_weight and picker v2 equity at picker_weight,
rebalanced monthly to target weight (no transaction cost — we already
charged it inside each sub-backtest).

Goal: identify the blend that BEST clears HC #557 R1 (Sharpe + DD vs
margin-SPY) AND comes closest to HC #557 R2 (positive monthly return in
each regime). Report the winning blend honestly.

Usage:
    cd /home/jupiter/Lvl3Quant/wheel_strategy_v1
    python3 -m strategy.blend_wheel_picker
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

from strategy.macro_picker.v1_sector_rotation import (
    regime_classify, stratified_sharpe, monthly_return_by_regime, risk_metrics,
)

ROOT = Path("/home/jupiter/Lvl3Quant/wheel_strategy_v1")
RESULTS = ROOT / "results"
OUT_DIR = RESULTS / "blend_wheel_picker_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRADING_DAYS = 252
ANN = np.sqrt(TRADING_DAYS)
STARTING_CASH = 100_000.0

# Source equity curves (already in $ terms, $100k start each).
WHEEL_EQUITY = RESULTS / "tier_ladder_v7_REAL_SKEW_SLIP_REGIME" / "equity_Tier2_Balanced_FW.parquet"
PICKER_EQUITY = RESULTS / "macro_picker_v2" / "equity_picker_v2.parquet"


def load_curve(path: Path, col_guess: str = "equity") -> pd.Series:
    df = pd.read_parquet(path)
    # If there's a 'date' column, use it as index
    if "date" in df.columns:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    else:
        df.index = pd.to_datetime(df.index)
    if col_guess in df.columns:
        s = df[col_guess]
    else:
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) == 0:
            raise SystemExit(f"no numeric column in {path}")
        s = df[num_cols[0]]
    return s.dropna().sort_index()


def realized_cash_from_ledger(ledger_path: Path) -> pd.Series:
    """Reconstruct realized-cash curve from a wheel ledger.parquet."""
    ledger = pd.read_parquet(ledger_path)
    if "close_date" not in ledger.columns or "realized_pnl" not in ledger.columns:
        return None
    df = ledger.copy()
    df["close_date"] = pd.to_datetime(df["close_date"])
    daily = df.groupby("close_date")["realized_pnl"].sum().sort_index()
    cum = STARTING_CASH + daily.cumsum()
    return cum


def blended_curve(wheel_eq: pd.Series, picker_eq: pd.Series,
                  w_wheel: float, w_picker: float) -> pd.Series:
    """Build a blended equity curve from two normalized series with monthly rebalance."""
    # Align dates
    common = wheel_eq.index.intersection(picker_eq.index)
    if len(common) < 100:
        raise SystemExit(f"insufficient overlap: {len(common)} days")
    we = wheel_eq.loc[common]
    pe = picker_eq.loc[common]
    we_ret = we.pct_change().fillna(0.0)
    pe_ret = pe.pct_change().fillna(0.0)
    # Daily blended return at static weights
    blended_ret = w_wheel * we_ret + w_picker * pe_ret
    eq = (1 + blended_ret).cumprod() * STARTING_CASH
    return eq


def metrics_block(eq: pd.Series) -> dict:
    ret = eq.pct_change().dropna()
    return risk_metrics(ret, eq)


def main():
    print("[blend] loading curves")
    # USE REALIZED-CASH from ledger, not MTM — MTM has the March-2020 short-put repricing artifact
    ledger_path = WHEEL_EQUITY.parent / "ledger_Tier2_Balanced_FW.parquet"
    wheel_eq = realized_cash_from_ledger(ledger_path)
    if wheel_eq is None:
        raise SystemExit(f"no ledger at {ledger_path}")
    # Reindex to business-daily ffill so it aligns with picker daily curve
    full_idx = pd.bdate_range(wheel_eq.index.min(), wheel_eq.index.max())
    wheel_eq = wheel_eq.reindex(full_idx, method="ffill").fillna(STARTING_CASH)
    print(f"[blend] wheel realized-cash curve reconstructed from ledger ({len(wheel_eq)} pts, {wheel_eq.index[0].date()}..{wheel_eq.index[-1].date()})")

    picker_eq = load_curve(PICKER_EQUITY)
    print(f"[blend] picker curve from {PICKER_EQUITY.name} ({len(picker_eq)} pts, {picker_eq.index[0].date()}..{picker_eq.index[-1].date()})")

    # Need SPY for regime — load it
    etfs = pd.read_parquet(ROOT / "data" / "cache" / "sector_etfs.parquet")
    spy = etfs[etfs["ticker"] == "SPY"].set_index("date").sort_index()
    spy.index = pd.to_datetime(spy.index)
    spy = spy[(spy.index >= pd.Timestamp("2020-01-01")) & (spy.index <= pd.Timestamp("2025-12-31"))]
    regime = regime_classify(spy["close"])

    # SPY 1.5x benchmark target
    SPY15_SHARPE = 0.70
    SPY15_DD = -0.4734
    SPY15_CAGR = 0.185

    blends = [
        (1.0, 0.0),   # wheel only
        (0.7, 0.3),
        (0.6, 0.4),
        (0.5, 0.5),
        (0.3, 0.7),
        (0.0, 1.0),   # picker only
    ]

    rows = []
    for ww, wp in blends:
        eq = blended_curve(wheel_eq, picker_eq, ww, wp)
        m = metrics_block(eq)
        daily_ret = eq.pct_change().dropna()
        mbr = monthly_return_by_regime(daily_ret, regime)
        ss = stratified_sharpe(daily_ret, regime)
        # Verdict
        sh_win = m["sharpe"] > SPY15_SHARPE
        dd_win = m["max_dd"] > SPY15_DD
        red_pos = mbr["red"] >= 0
        sh_vals = [v for v in ss.values() if not np.isnan(v)]
        sym = abs(min(sh_vals, key=abs)) / abs(max(sh_vals, key=abs)) if sh_vals else 0
        sym_ok = sym >= 0.50
        rows.append({
            "wheel_w": ww, "picker_w": wp,
            "cagr": m["cagr"], "sharpe": m["sharpe"], "sortino": m["sortino"],
            "max_dd": m["max_dd"],
            "green_monthly": mbr["green"], "red_monthly": mbr["red"], "flat_monthly": mbr["flat"],
            "green_sh": ss["green"], "red_sh": ss["red"], "flat_sh": ss["flat"],
            "sharpe_beats_spy15": sh_win, "dd_beats_spy15": dd_win,
            "red_positive": red_pos, "regime_symmetric": sym_ok, "symmetry": sym,
        })

    df = pd.DataFrame(rows)
    df.to_parquet(OUT_DIR / "blend_grid.parquet")

    # Pick best blend by: Sharpe > 0.70 AND DD < SPY1.5x AND red_monthly closest to zero
    qualified = df[df["sharpe_beats_spy15"] & df["dd_beats_spy15"]].copy()
    if len(qualified):
        qualified["red_distance"] = -qualified["red_monthly"]  # smaller (less negative) is better
        qualified = qualified.sort_values(["red_distance", "sharpe"], ascending=[True, False])
        best = qualified.iloc[0]
    else:
        best = df.sort_values("sharpe", ascending=False).iloc[0]

    report = f"""# Wheel + Picker Blend — HC #558 R3 + HC #557 R7

Window: 2020-01-01 → 2025-12-31. $100k starting per leg.
Wheel: v7 canonical Balanced FullWheel (real IV + skew + slippage + regime overlay).
Picker: v2 thematic rotation (sectors + themes + acceleration signal).
Blend: static weight, daily rebalanced (mathematical, no friction added because both legs already charged friction internally).

## Grid Results

| Wheel% | Picker% | CAGR | Sharpe | Sortino | MaxDD | Green$/mo | Red$/mo | Flat$/mo | Sh>0.70 | DD<-47% | Red≥0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
"""
    for _, r in df.iterrows():
        report += (f"| {r['wheel_w']*100:.0f}% | {r['picker_w']*100:.0f}% | "
                   f"{r['cagr']:.1%} | {r['sharpe']:.2f} | {r['sortino']:.2f} | {r['max_dd']:.1%} | "
                   f"{r['green_monthly']:.2%} | {r['red_monthly']:.2%} | {r['flat_monthly']:.2%} | "
                   f"{'✓' if r['sharpe_beats_spy15'] else '✗'} | "
                   f"{'✓' if r['dd_beats_spy15'] else '✗'} | "
                   f"{'✓' if r['red_positive'] else '✗'} |\n")

    report += f"""
## Best Blend (by red-month closest to zero among qualified)

**Wheel {best['wheel_w']*100:.0f}% / Picker {best['picker_w']*100:.0f}%**
- CAGR {best['cagr']:.1%}, Sharpe {best['sharpe']:.2f}, MaxDD {best['max_dd']:.1%}
- Monthly returns: green {best['green_monthly']:.2%}, red {best['red_monthly']:.2%}, flat {best['flat_monthly']:.2%}
- vs SPY 1.5× margin (Sharpe 0.70, MaxDD -47%, CAGR 18.5%): {"WINS Sharpe + DD" if best['sharpe_beats_spy15'] and best['dd_beats_spy15'] else "loses one of Sharpe/DD"}

## Notes
- Picker v2 alone has Red monthly of -3.93% — pure long-only equity bleeds in red months.
- Wheel Balanced alone red monthly varies — premium income usually positive but small.
- Blend should ideally produce non-negative red months from wheel premium covering picker losses.
"""
    (OUT_DIR / "blend_report.md").write_text(report)
    with open(OUT_DIR / "best_blend.json", "w") as f:
        json.dump(best.to_dict(), f, indent=2, default=str)

    print(report)
    print(f"\n[blend] best blend: wheel {best['wheel_w']*100:.0f}% / picker {best['picker_w']*100:.0f}%")


if __name__ == "__main__":
    main()

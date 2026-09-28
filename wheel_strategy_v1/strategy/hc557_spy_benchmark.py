"""
HC #557 — SPY Benchmark & Regime-Income Gate
=============================================
Re-emit the v7 wheel result with SPY/margin-SPY benchmarks and a regime-bucket
income gate.

Rules (HC #557, per user directive 2026-06-06 ~23:42 ET):
  R1: Wheel must beat 1.5x margin-SPY on risk-adjusted basis (Sharpe AND MaxDD).
  R2: Wheel must generate positive monthly premium income in ALL three regimes
      (green / red / flat) AND satisfy the HC #428 symmetric gate
      (worst-bucket Sharpe >= 50% * best-bucket Sharpe).
  R4: If wheel merely matches SPY -> margin-SPY is strictly better; NEGATIVE FINDING.

Outputs:
  - hc557_spy_benchmark_report.md  (plain English)
  - hc557_results.parquet
  - hc557_results.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# -------------------------------------------------------------------------
# Config
# -------------------------------------------------------------------------
ROOT = Path("/home/jupiter/Lvl3Quant/wheel_strategy_v1")
RESULTS_DIR = ROOT / "results" / "tier_ladder_v7_REAL_SKEW_SLIP_REGIME"
CACHE_DIR = ROOT / "data" / "cache"

START_DATE = pd.Timestamp("2020-01-01")
END_DATE = pd.Timestamp("2025-12-31")
STARTING_CASH = 100_000.0
TRADING_DAYS = 252
ANN = np.sqrt(TRADING_DAYS)

BROKER_SPREAD = 0.015  # 1.5% over short rate (Reg-T conservative)
DEFAULT_SHORT_RATE = 0.05  # 5% flat fallback

TIERS = [
    "Tier1_Conservative_FW",
    "Tier2_Balanced_FW",
    "Tier3_Income_FW",
    "Tier4_Aggressive_FW",
    "Tier5_Turbo_FW",
]

# Blended ladder weights (35% Conservative / 55% Balanced / 10% Turbo)
LADDER_WEIGHTS = {
    "Tier1_Conservative_FW": 0.35,
    "Tier2_Balanced_FW": 0.55,
    "Tier5_Turbo_FW": 0.10,
}
LADDER_INITIAL = 300_000.0


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------
def risk_metrics(daily_ret: pd.Series, equity: pd.Series) -> dict:
    """Compute CAGR, Sharpe, Sortino, MaxDD on a daily return + equity series."""
    daily_ret = daily_ret.dropna()
    if len(daily_ret) < 2 or equity.iloc[0] <= 0 or equity.iloc[-1] <= 0:
        return {"cagr": np.nan, "sharpe": np.nan, "sortino": np.nan, "max_dd": np.nan}
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan
    std = daily_ret.std()
    sharpe = (daily_ret.mean() / std) * ANN if std > 0 else np.nan
    downside = daily_ret[daily_ret < 0].std()
    sortino = (daily_ret.mean() / downside) * ANN if downside and downside > 0 else np.nan
    running_max = equity.cummax()
    dd = (equity - running_max) / running_max
    max_dd = float(dd.min())
    return {
        "cagr": float(cagr),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_dd": max_dd,
    }


def load_spy() -> pd.DataFrame:
    """Try cache first, then yfinance. Returns DataFrame indexed by date with 'close'."""
    prices = pd.read_parquet(CACHE_DIR / "prices.parquet")
    if "ticker" in prices.columns and "SPY" in prices["ticker"].unique():
        spy = prices[prices["ticker"] == "SPY"][["date", "close"]].copy()
        spy = spy.set_index("date").sort_index()
        spy = spy.loc[START_DATE:END_DATE]
        return spy
    # Fallback: yfinance
    import yfinance as yf
    df = yf.download(
        "SPY",
        start=START_DATE.strftime("%Y-%m-%d"),
        end=(END_DATE + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=False,
    )
    # yfinance returns multi-index columns when single ticker in 1.2+
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    spy = df[["Adj Close"]].rename(columns={"Adj Close": "close"})
    return spy


def load_short_rate(spy_index: pd.DatetimeIndex) -> pd.Series:
    """Build daily short rate (decimal). Use FEDFUNDS forward-filled, else default."""
    try:
        ff = pd.read_csv(CACHE_DIR / "macro_extra" / "fred_FEDFUNDS.csv")
        ff["date"] = pd.to_datetime(ff["date"])
        ff = ff.set_index("date").sort_index()
        # FEDFUNDS already in percent; convert to decimal
        rate = ff["FEDFUNDS"] / 100.0
        rate = rate.reindex(spy_index, method="ffill").bfill()
        if rate.isna().all():
            raise ValueError("All NaN after reindex")
        return rate
    except Exception as e:
        print(f"[warn] Could not load FEDFUNDS ({e}); defaulting to {DEFAULT_SHORT_RATE:.2%}")
        return pd.Series(DEFAULT_SHORT_RATE, index=spy_index)


def build_spy_benchmarks(spy: pd.DataFrame) -> dict:
    """Returns dict of {label: equity Series} starting at 100k."""
    close = spy["close"]
    spy_ret = close.pct_change().fillna(0.0)
    short_rate = load_short_rate(close.index)

    # 1.0x unlevered
    eq_1x = STARTING_CASH * (close / close.iloc[0])
    eq_1x.name = "SPY_1.0x"

    # 1.5x margin: 1.5 * ret - (0.5 / 252) * (short_rate + spread)
    margin_1p5_drag = (0.5 / TRADING_DAYS) * (short_rate + BROKER_SPREAD)
    daily_ret_1p5 = 1.5 * spy_ret - margin_1p5_drag
    daily_ret_1p5.iloc[0] = 0.0
    eq_1p5 = STARTING_CASH * (1.0 + daily_ret_1p5).cumprod()
    eq_1p5.name = "SPY_1.5x_margin"

    # 2.0x margin
    margin_2x_drag = (1.0 / TRADING_DAYS) * (short_rate + BROKER_SPREAD)
    daily_ret_2x = 2.0 * spy_ret - margin_2x_drag
    daily_ret_2x.iloc[0] = 0.0
    eq_2x = STARTING_CASH * (1.0 + daily_ret_2x).cumprod()
    eq_2x.name = "SPY_2.0x_margin"

    return {
        "SPY_1.0x": (eq_1x, eq_1x.pct_change()),
        "SPY_1.5x_margin": (eq_1p5, daily_ret_1p5),
        "SPY_2.0x_margin": (eq_2x, daily_ret_2x),
        "_spy_ret": spy_ret,
        "_short_rate": short_rate,
    }


def build_tier_realized_equity(tier: str) -> tuple[pd.Series, pd.Series]:
    """Aggregate realized_pnl by close_date -> daily realized equity series."""
    ledger = pd.read_parquet(RESULTS_DIR / f"ledger_{tier}.parquet")
    daily = (
        ledger.groupby("close_date")["realized_pnl"]
        .sum()
        .sort_index()
    )
    # Build calendar from equity file to ensure aligned date range
    eq_file = pd.read_parquet(RESULTS_DIR / f"equity_{tier}.parquet")
    calendar = pd.DatetimeIndex(eq_file["date"]).sort_values()
    daily_cash = daily.reindex(calendar, fill_value=0.0)
    realized_equity = STARTING_CASH + daily_cash.cumsum()
    realized_equity.name = tier
    daily_ret = realized_equity.pct_change()
    return realized_equity, daily_ret


def classify_regimes(spy_ret: pd.Series) -> pd.Series:
    """green if ret >= +0.5 sigma, red if ret <= -0.5 sigma, flat in between."""
    sigma = spy_ret.std()
    threshold = 0.5 * sigma
    regime = pd.Series("flat", index=spy_ret.index, dtype=object)
    regime[spy_ret >= threshold] = "green"
    regime[spy_ret <= -threshold] = "red"
    return regime


def regime_stats(daily_pnl: pd.Series, regime: pd.Series) -> dict:
    """Per-regime: mean daily pnl, mean monthly pnl (mean*21), count, Sharpe."""
    out = {}
    aligned = pd.concat([daily_pnl.rename("pnl"), regime.rename("regime")], axis=1).dropna()
    for r in ["green", "red", "flat"]:
        sub = aligned[aligned["regime"] == r]["pnl"]
        if len(sub) < 2:
            out[r] = {
                "n_days": int(len(sub)),
                "mean_daily": float(sub.mean()) if len(sub) else 0.0,
                "monthly": float(sub.mean() * 21) if len(sub) else 0.0,
                "sharpe": np.nan,
            }
            continue
        std = sub.std()
        sharpe = (sub.mean() / std) * ANN if std > 0 else np.nan
        out[r] = {
            "n_days": int(len(sub)),
            "mean_daily": float(sub.mean()),
            "monthly": float(sub.mean() * 21),
            "sharpe": float(sharpe),
        }
    return out


def regime_gate(stats: dict) -> tuple[bool, str]:
    """HC #557 R2: monthly > 0 in every bucket AND worst Sharpe >= 50% best Sharpe."""
    months = {r: stats[r]["monthly"] for r in ["green", "red", "flat"]}
    sharpes = {r: stats[r]["sharpe"] for r in ["green", "red", "flat"]}
    fails = []
    for r, v in months.items():
        if not (v > 0):
            fails.append(f"{r}_monthly<=0 ({v:.0f})")
    valid_sharpes = [s for s in sharpes.values() if pd.notna(s)]
    if len(valid_sharpes) >= 2:
        best = max(valid_sharpes, key=abs)
        worst = min(valid_sharpes, key=abs)
        if abs(best) > 0 and abs(worst) / abs(best) < 0.5:
            fails.append(f"sharpe_dispersion |worst|/|best|={abs(worst)/abs(best):.2f}<0.50")
    if fails:
        return False, "; ".join(fails)
    return True, "all buckets positive, dispersion ok"


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("HC #557 — SPY Benchmark & Regime-Income Gate")
    print("=" * 70)

    # ---- SPY benchmarks ----
    spy = load_spy()
    print(f"[SPY] loaded {len(spy)} days, {spy.index.min().date()} -> {spy.index.max().date()}")
    benches = build_spy_benchmarks(spy)
    spy_ret = benches["_spy_ret"]

    bench_metrics = {}
    for label in ["SPY_1.0x", "SPY_1.5x_margin", "SPY_2.0x_margin"]:
        eq, ret = benches[label]
        bench_metrics[label] = risk_metrics(ret, eq)
        print(f"[{label}] CAGR={bench_metrics[label]['cagr']:.2%} "
              f"Sharpe={bench_metrics[label]['sharpe']:.2f} "
              f"Sortino={bench_metrics[label]['sortino']:.2f} "
              f"MaxDD={bench_metrics[label]['max_dd']:.2%}")

    # ---- Regimes on SPY ----
    regime = classify_regimes(spy_ret)
    n_green = int((regime == "green").sum())
    n_red = int((regime == "red").sum())
    n_flat = int((regime == "flat").sum())
    print(f"[regimes] green={n_green} red={n_red} flat={n_flat}")

    # SPY 1.5x daily $ pnl (for regime comparison vs. wheel realized cash)
    spy_1p5_eq, spy_1p5_ret = benches["SPY_1.5x_margin"]
    spy_1p5_daily_pnl = spy_1p5_ret * STARTING_CASH  # approximate daily $ pnl
    spy_regime_stats = regime_stats(spy_1p5_daily_pnl, regime)

    # ---- Tier metrics ----
    tier_metrics = {}
    tier_regime_stats = {}
    tier_realized_equity = {}
    tier_daily_pnl = {}
    spy_bench_for_compare = bench_metrics["SPY_1.5x_margin"]

    for tier in TIERS:
        eq, ret = build_tier_realized_equity(tier)
        tier_realized_equity[tier] = eq
        daily_pnl = eq.diff().fillna(0.0)
        tier_daily_pnl[tier] = daily_pnl
        m = risk_metrics(ret, eq)
        tier_metrics[tier] = m
        # Regime stats on daily realized cash
        # Align regime to tier's calendar
        regime_aligned = regime.reindex(eq.index).ffill().bfill()
        tier_regime_stats[tier] = regime_stats(daily_pnl, regime_aligned)
        passed, reason = regime_gate(tier_regime_stats[tier])
        # Win condition
        wins_sharpe = m["sharpe"] > spy_bench_for_compare["sharpe"]
        wins_dd = m["max_dd"] > spy_bench_for_compare["max_dd"]  # less negative is better
        verdict = "WHEEL WINS" if (wins_sharpe and wins_dd and passed) else "WHEEL LOSES — JUST USE MARGIN SPY"
        tier_metrics[tier]["regime_passed"] = passed
        tier_metrics[tier]["regime_reason"] = reason
        tier_metrics[tier]["verdict"] = verdict
        print(f"[{tier}] CAGR={m['cagr']:.2%} Sharpe={m['sharpe']:.2f} "
              f"Sortino={m['sortino']:.2f} MaxDD={m['max_dd']:.2%} "
              f"regime_gate={'PASS' if passed else 'FAIL'} -> {verdict}")

    # ---- Blended ladder ----
    # Daily-additive on $300k. Each tier sleeve runs on its own 100k base; we
    # scale daily $ pnl by the weight ratio so it corresponds to its slice of 300k.
    ladder_calendar = tier_realized_equity[TIERS[0]].index
    ladder_pnl = pd.Series(0.0, index=ladder_calendar)
    for tier, w in LADDER_WEIGHTS.items():
        # The tier strategy is sized to 100k; allocating w*300k means the same
        # strategy at (w*300k / 100k) = 3w of original scale.
        sleeve_dollars = w * LADDER_INITIAL
        scale = sleeve_dollars / STARTING_CASH
        sleeve_pnl = tier_daily_pnl[tier].reindex(ladder_calendar, fill_value=0.0) * scale
        ladder_pnl = ladder_pnl + sleeve_pnl
    ladder_equity = LADDER_INITIAL + ladder_pnl.cumsum()
    ladder_equity.name = "Blended_Ladder_35_55_10"
    ladder_ret = ladder_equity.pct_change()
    ladder_metrics = risk_metrics(ladder_ret, ladder_equity)

    # Regime stats on ladder
    regime_aligned_ladder = regime.reindex(ladder_calendar).ffill().bfill()
    ladder_regime_stats = regime_stats(ladder_pnl, regime_aligned_ladder)
    ladder_passed, ladder_reason = regime_gate(ladder_regime_stats)
    ladder_wins_sharpe = ladder_metrics["sharpe"] > spy_bench_for_compare["sharpe"]
    ladder_wins_dd = ladder_metrics["max_dd"] > spy_bench_for_compare["max_dd"]
    ladder_verdict = (
        "WHEEL WINS"
        if (ladder_wins_sharpe and ladder_wins_dd and ladder_passed)
        else "WHEEL LOSES — JUST USE MARGIN SPY"
    )
    ladder_metrics["regime_passed"] = ladder_passed
    ladder_metrics["regime_reason"] = ladder_reason
    ladder_metrics["verdict"] = ladder_verdict
    print(f"[BLENDED LADDER] CAGR={ladder_metrics['cagr']:.2%} "
          f"Sharpe={ladder_metrics['sharpe']:.2f} "
          f"Sortino={ladder_metrics['sortino']:.2f} "
          f"MaxDD={ladder_metrics['max_dd']:.2%} -> {ladder_verdict}")

    # ---- Persist structured outputs ----
    rows = []
    for label, m in bench_metrics.items():
        rows.append({"name": label, "kind": "benchmark", **m})
    for tier, m in tier_metrics.items():
        rows.append({"name": tier, "kind": "wheel_tier", **m})
    rows.append({"name": "Blended_Ladder_35_55_10", "kind": "blended_ladder", **ladder_metrics})
    summary_df = pd.DataFrame(rows)
    summary_df.to_parquet(RESULTS_DIR / "hc557_results.parquet", index=False)

    json_blob = {
        "config": {
            "start": str(START_DATE.date()),
            "end": str(END_DATE.date()),
            "starting_cash_per_tier": STARTING_CASH,
            "ladder_initial": LADDER_INITIAL,
            "ladder_weights": LADDER_WEIGHTS,
            "broker_spread": BROKER_SPREAD,
        },
        "regime_counts": {"green": n_green, "red": n_red, "flat": n_flat},
        "benchmarks": bench_metrics,
        "spy_1p5_regime_stats": spy_regime_stats,
        "tiers": {t: tier_metrics[t] for t in TIERS},
        "tier_regime_stats": tier_regime_stats,
        "blended_ladder": ladder_metrics,
        "blended_ladder_regime_stats": ladder_regime_stats,
    }
    with open(RESULTS_DIR / "hc557_results.json", "w") as f:
        json.dump(json_blob, f, indent=2, default=str)

    # ---- Markdown report ----
    report = build_report(
        bench_metrics=bench_metrics,
        tier_metrics=tier_metrics,
        tier_regime_stats=tier_regime_stats,
        ladder_metrics=ladder_metrics,
        ladder_regime_stats=ladder_regime_stats,
        spy_regime_stats=spy_regime_stats,
        regime_counts={"green": n_green, "red": n_red, "flat": n_flat},
    )
    report_path = RESULTS_DIR / "hc557_spy_benchmark_report.md"
    with open(report_path, "w") as f:
        f.write(report)

    print()
    print("Report written to:", report_path)
    print("Structured:", RESULTS_DIR / "hc557_results.parquet")
    print()
    print(f"FINAL VERDICT (blended ladder): {ladder_verdict}")


def fmt_pct(x):
    return f"{x*100:.2f}%" if pd.notna(x) else "n/a"


def fmt_money(x):
    if pd.isna(x):
        return "n/a"
    return f"${x:,.0f}"


def fmt_num(x, d=2):
    return f"{x:.{d}f}" if pd.notna(x) else "n/a"


def build_report(bench_metrics, tier_metrics, tier_regime_stats,
                 ladder_metrics, ladder_regime_stats, spy_regime_stats,
                 regime_counts) -> str:
    L = []
    L.append("# HC #557 — Wheel vs. SPY/Margin-SPY Benchmark")
    L.append("")
    L.append(f"**Verdict (blended ladder, 35% Conservative / 55% Balanced / 10% Turbo): "
             f"{ladder_metrics['verdict']}**")
    L.append("")
    L.append("Window: 2020-01-01 to 2025-12-31. Cash starts at $100k per tier "
             "($300k for blended ladder). All wheel numbers use realized cash, "
             "not mark-to-market.")
    L.append("")

    # Benchmark vs ladder table
    L.append("## Headline Comparison")
    L.append("")
    L.append("| Strategy | CAGR | Sharpe | Sortino | Max Drawdown |")
    L.append("|---|---|---|---|---|")
    L.append(f"| SPY (unlevered) | {fmt_pct(bench_metrics['SPY_1.0x']['cagr'])} | "
             f"{fmt_num(bench_metrics['SPY_1.0x']['sharpe'])} | "
             f"{fmt_num(bench_metrics['SPY_1.0x']['sortino'])} | "
             f"{fmt_pct(bench_metrics['SPY_1.0x']['max_dd'])} |")
    L.append(f"| SPY 1.5x margin | {fmt_pct(bench_metrics['SPY_1.5x_margin']['cagr'])} | "
             f"{fmt_num(bench_metrics['SPY_1.5x_margin']['sharpe'])} | "
             f"{fmt_num(bench_metrics['SPY_1.5x_margin']['sortino'])} | "
             f"{fmt_pct(bench_metrics['SPY_1.5x_margin']['max_dd'])} |")
    L.append(f"| SPY 2.0x margin | {fmt_pct(bench_metrics['SPY_2.0x_margin']['cagr'])} | "
             f"{fmt_num(bench_metrics['SPY_2.0x_margin']['sharpe'])} | "
             f"{fmt_num(bench_metrics['SPY_2.0x_margin']['sortino'])} | "
             f"{fmt_pct(bench_metrics['SPY_2.0x_margin']['max_dd'])} |")
    for tier in TIERS:
        m = tier_metrics[tier]
        nice = tier.replace("_FW", "").replace("_", " ")
        L.append(f"| {nice} | {fmt_pct(m['cagr'])} | {fmt_num(m['sharpe'])} | "
                 f"{fmt_num(m['sortino'])} | {fmt_pct(m['max_dd'])} |")
    L.append(f"| Blended Ladder | {fmt_pct(ladder_metrics['cagr'])} | "
             f"{fmt_num(ladder_metrics['sharpe'])} | "
             f"{fmt_num(ladder_metrics['sortino'])} | "
             f"{fmt_pct(ladder_metrics['max_dd'])} |")
    L.append("")

    # Per-regime monthly income
    L.append("## Monthly Realized Income by Regime")
    L.append("")
    L.append(f"Day counts in window — green up days: {regime_counts['green']}, "
             f"red down days: {regime_counts['red']}, flat days: {regime_counts['flat']}.")
    L.append("")
    L.append("| Strategy | Green Months | Red Months | Flat Months | Worst-Bucket Test |")
    L.append("|---|---|---|---|---|")
    for tier in TIERS:
        rs = tier_regime_stats[tier]
        nice = tier.replace("_FW", "").replace("_", " ")
        passed, reason = regime_gate(rs)
        gate = "PASS" if passed else "FAIL"
        L.append(f"| {nice} | {fmt_money(rs['green']['monthly'])} | "
                 f"{fmt_money(rs['red']['monthly'])} | "
                 f"{fmt_money(rs['flat']['monthly'])} | {gate} |")
    rs = ladder_regime_stats
    passed, _ = regime_gate(rs)
    L.append(f"| Blended Ladder | {fmt_money(rs['green']['monthly'])} | "
             f"{fmt_money(rs['red']['monthly'])} | "
             f"{fmt_money(rs['flat']['monthly'])} | "
             f"{'PASS' if passed else 'FAIL'} |")
    L.append("")
    L.append("Gate rule: positive monthly income in every regime AND no single "
             "bucket's risk-adjusted return more than 2x the others.")
    L.append("")

    # Per-tier verdicts
    L.append("## Per-Tier Verdicts")
    L.append("")
    for tier in TIERS:
        m = tier_metrics[tier]
        nice = tier.replace("_FW", "").replace("_", " ")
        L.append(f"- **{nice}** — {m['verdict']} "
                 f"(Sharpe {fmt_num(m['sharpe'])} vs margin-SPY "
                 f"{fmt_num(bench_metrics['SPY_1.5x_margin']['sharpe'])}; "
                 f"regime gate {'PASS' if m['regime_passed'] else 'FAIL — ' + m['regime_reason']}).")
    L.append(f"- **Blended Ladder** — {ladder_metrics['verdict']} "
             f"(Sharpe {fmt_num(ladder_metrics['sharpe'])} vs margin-SPY "
             f"{fmt_num(bench_metrics['SPY_1.5x_margin']['sharpe'])}; "
             f"regime gate {'PASS' if ladder_metrics['regime_passed'] else 'FAIL — ' + ladder_metrics['regime_reason']}).")
    L.append("")

    # Bottom line
    L.append("## Bottom Line")
    L.append("")
    if "WINS" in ladder_metrics["verdict"]:
        L.append("The blended wheel ladder beats a 1.5x margin SPY portfolio on "
                 "risk-adjusted basis and pays positive premium income in green, "
                 "red, and flat market regimes. Keep building.")
    else:
        L.append("The blended wheel ladder does not beat a 1.5x margin SPY "
                 "portfolio on risk-adjusted basis, or fails the regime-income "
                 "gate. Under HC #557, this is a NEGATIVE FINDING — a margin "
                 "SPY position is strictly better. Do not deploy the wheel as a "
                 "stand-alone product unless the structure is materially changed.")
    L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    main()

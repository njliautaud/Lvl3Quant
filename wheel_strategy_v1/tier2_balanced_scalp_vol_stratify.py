#!/usr/bin/env python3
"""
Tier2 Balanced Scalp wheel - VIX-regime stratification + tail-risk + survivability.

Loads the Tier2 Balanced equity curve from the alpha-stronger tier_ladder_v1 run
(Sharpe ~2.51, CAGR ~27.4%, MaxDD -15.5%), joins to daily VIX + SPY, and produces:
  - VIX-regime stratified Sharpe/WR/loss table
  - 5 worst single-day losses w/ VIX & SPY context
  - VaR-95 / CVaR-95
  - Survivability around COVID-2020, Aug-2024 vol shock, 2022 bear
  - Kelly-style deployable sizing fraction to keep MaxDD < 25% in worst regime

Outputs to /home/jupiter/Lvl3Quant/output/wheel_tier2_vol_stratify_<ts>/
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/jupiter/Lvl3Quant")
EQUITY_PATH = ROOT / "wheel_strategy_v1/results/tier_ladder_v1/equity_Tier2_Balanced.parquet"
MACRO_PATH = ROOT / "wheel_strategy_v1/data/cache/macro.parquet"
SECTOR_PATH = ROOT / "wheel_strategy_v1/data/cache/sector_etfs.parquet"

TS = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = ROOT / f"output/wheel_tier2_vol_stratify_{TS}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRADING_DAYS = 252


# -------------------- helpers --------------------
def ann_sharpe(rets: pd.Series) -> float:
    rets = rets.dropna()
    if len(rets) < 2 or rets.std(ddof=1) == 0:
        return float("nan")
    return float(rets.mean() / rets.std(ddof=1) * np.sqrt(TRADING_DAYS))


def ann_sortino(rets: pd.Series) -> float:
    rets = rets.dropna()
    downside = rets[rets < 0]
    if len(downside) < 2 or downside.std(ddof=1) == 0:
        return float("nan")
    return float(rets.mean() / downside.std(ddof=1) * np.sqrt(TRADING_DAYS))


def max_drawdown(equity: pd.Series) -> tuple[float, pd.Timestamp, pd.Timestamp]:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    trough_idx = dd.idxmin()
    peak_idx = equity.loc[:trough_idx].idxmax()
    return float(dd.min()), peak_idx, trough_idx


def safe_pct(x: float) -> float:
    return float(x) * 100.0 if np.isfinite(x) else float("nan")


# -------------------- load --------------------
print("[load] equity curve")
eq = pd.read_parquet(EQUITY_PATH).rename(columns={"date": "date", "equity": "equity"})
eq["date"] = pd.to_datetime(eq["date"])
eq = eq.sort_values("date").reset_index(drop=True)
eq["ret"] = eq["equity"].pct_change()

print("[load] macro VIX")
macro = pd.read_parquet(MACRO_PATH)[["date", "vix"]].copy()
macro["date"] = pd.to_datetime(macro["date"])

print("[load] SPY")
spy = pd.read_parquet(SECTOR_PATH)
spy = spy[spy["ticker"] == "SPY"][["date", "close", "ret_1d"]].copy()
spy["date"] = pd.to_datetime(spy["date"])
spy = spy.rename(columns={"close": "spy_close", "ret_1d": "spy_ret"})

df = eq.merge(macro, on="date", how="left").merge(spy, on="date", how="left")
df["vix"] = df["vix"].ffill()  # carry VIX into rare missing-row holidays
df["vix_chg"] = df["vix"].diff()

print(f"[load] {len(df)} rows  {df['date'].min().date()} -> {df['date'].max().date()}")
print(f"[load] vix coverage: {df['vix'].notna().mean():.3f}  spy coverage: {df['spy_close'].notna().mean():.3f}")


# -------------------- 1. headline metrics --------------------
rets = df["ret"].dropna()
headline = {
    "n_days": int(len(rets)),
    "cagr": (df["equity"].iloc[-1] / df["equity"].iloc[0]) ** (TRADING_DAYS / len(rets)) - 1,
    "sharpe": ann_sharpe(rets),
    "sortino": ann_sortino(rets),
    "max_dd": max_drawdown(df.set_index("date")["equity"])[0],
    "mean_daily_pct": float(rets.mean() * 100),
    "std_daily_pct": float(rets.std(ddof=1) * 100),
    "win_rate": float((rets > 0).mean()),
    "worst_day_pct": float(rets.min() * 100),
    "best_day_pct": float(rets.max() * 100),
}

# -------------------- 2. VIX regime stratification --------------------
def vix_bucket(v: float) -> str:
    if not np.isfinite(v):
        return "unknown"
    if v < 15:
        return "VIX_low_lt15"
    if v < 25:
        return "VIX_mid_15_25"
    if v < 35:
        return "VIX_high_25_35"
    return "VIX_spike_gt35"


df["vix_regime"] = df["vix"].apply(vix_bucket)

regime_rows = []
for reg in ["VIX_low_lt15", "VIX_mid_15_25", "VIX_high_25_35", "VIX_spike_gt35"]:
    sub = df[df["vix_regime"] == reg]["ret"].dropna()
    if len(sub) == 0:
        regime_rows.append({"regime": reg, "n_days": 0})
        continue
    regime_rows.append({
        "regime": reg,
        "n_days": int(len(sub)),
        "mean_daily_pct": float(sub.mean() * 100),
        "std_daily_pct": float(sub.std(ddof=1) * 100),
        "sharpe": ann_sharpe(sub),
        "sortino": ann_sortino(sub),
        "win_rate": float((sub > 0).mean()),
        "worst_day_pct": float(sub.min() * 100),
        "best_day_pct": float(sub.max() * 100),
        "cum_return_pct": float(((1 + sub).prod() - 1) * 100),
    })
regime_df = pd.DataFrame(regime_rows)

# -------------------- 3. tail risk --------------------
worst5 = df.nsmallest(5, "ret")[["date", "ret", "vix", "vix_chg", "spy_ret", "spy_close"]].copy()
worst5["ret_pct"] = worst5["ret"] * 100
worst5["spy_ret_pct"] = worst5["spy_ret"] * 100
worst5_records = []
for _, r in worst5.iterrows():
    worst5_records.append({
        "date": r["date"].strftime("%Y-%m-%d"),
        "wheel_ret_pct": float(r["ret_pct"]),
        "vix_close": float(r["vix"]) if np.isfinite(r["vix"]) else None,
        "vix_chg": float(r["vix_chg"]) if np.isfinite(r["vix_chg"]) else None,
        "spy_ret_pct": float(r["spy_ret_pct"]) if np.isfinite(r["spy_ret_pct"]) else None,
    })

# VaR / CVaR
var95 = float(np.percentile(rets, 5))
cvar95 = float(rets[rets <= var95].mean())

# Drawdown decomposition - find top 5 drawdowns
def find_drawdowns(equity: pd.Series, top_n: int = 5):
    peak = equity.cummax()
    dd = equity / peak - 1.0
    in_dd = dd < 0
    # segment drawdowns
    segments = []
    start = None
    for i, flag in enumerate(in_dd):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            seg_dd = dd.iloc[start:i]
            trough_off = int(np.argmin(seg_dd.values))
            segments.append({
                "start": equity.index[start],
                "trough": equity.index[start + trough_off],
                "recover": equity.index[i],
                "max_dd": float(seg_dd.min()),
            })
            start = None
    if start is not None:
        seg_dd = dd.iloc[start:]
        trough_off = int(np.argmin(seg_dd.values))
        segments.append({
            "start": equity.index[start],
            "trough": equity.index[start + trough_off],
            "recover": None,
            "max_dd": float(seg_dd.min()),
        })
    segments.sort(key=lambda s: s["max_dd"])
    return segments[:top_n]


eq_idx = df.set_index("date")["equity"]
top_dds = find_drawdowns(eq_idx, top_n=5)
# annotate w/ avg VIX during the drawdown
dd_records = []
vix_idx = df.set_index("date")["vix"]
for seg in top_dds:
    s, t = seg["start"], seg["trough"]
    vix_during = vix_idx.loc[s:t]
    vix_mean = float(vix_during.mean()) if len(vix_during) else float("nan")
    vix_peak = float(vix_during.max()) if len(vix_during) else float("nan")
    dd_records.append({
        "start": s.strftime("%Y-%m-%d"),
        "trough": t.strftime("%Y-%m-%d"),
        "recover": seg["recover"].strftime("%Y-%m-%d") if seg["recover"] is not None else "ongoing",
        "max_dd_pct": seg["max_dd"] * 100,
        "vix_mean_during_dd": vix_mean,
        "vix_peak_during_dd": vix_peak,
        "regime_label": ("vol_spike" if vix_peak > 35 else "elevated_vol" if vix_peak > 25 else "grind_down"),
    })

# -------------------- 4. survivability --------------------
def window_stats(df_full: pd.DataFrame, start: str, end: str, label: str) -> dict:
    mask = (df_full["date"] >= start) & (df_full["date"] <= end)
    sub = df_full[mask].copy()
    if len(sub) == 0:
        return {"label": label, "n_days": 0}
    sub_eq = sub.set_index("date")["equity"]
    mdd, peak_d, trough_d = max_drawdown(sub_eq)
    # recovery: first date AFTER trough where equity recovers to peak_d value
    peak_val = sub_eq.loc[peak_d]
    after = df_full[df_full["date"] > trough_d].set_index("date")["equity"]
    recovered = after[after >= peak_val]
    rec_date = recovered.index[0].strftime("%Y-%m-%d") if len(recovered) else "not_recovered_in_data"
    rec_days = (recovered.index[0] - trough_d).days if len(recovered) else None
    # worst week (rolling 5d)
    sub["roll5"] = sub["ret"].fillna(0).rolling(5).sum()
    worst_week_pct = float(sub["roll5"].min() * 100) if sub["roll5"].notna().any() else float("nan")
    worst_week_end = sub.loc[sub["roll5"].idxmin(), "date"].strftime("%Y-%m-%d") if sub["roll5"].notna().any() else None
    return {
        "label": label,
        "window": f"{start} to {end}",
        "n_days": int(len(sub)),
        "max_dd_pct": mdd * 100,
        "peak_date": peak_d.strftime("%Y-%m-%d"),
        "trough_date": trough_d.strftime("%Y-%m-%d"),
        "recover_date": rec_date,
        "days_trough_to_recover": rec_days,
        "worst_week_pct": worst_week_pct,
        "worst_week_end": worst_week_end,
        "vix_peak": float(sub["vix"].max()) if sub["vix"].notna().any() else float("nan"),
        "vix_mean": float(sub["vix"].mean()) if sub["vix"].notna().any() else float("nan"),
        "cum_ret_pct": float(((1 + sub["ret"].fillna(0)).prod() - 1) * 100),
    }


survive = [
    window_stats(df, "2020-02-15", "2020-05-31", "COVID_2020"),
    window_stats(df, "2024-07-15", "2024-09-15", "Aug_2024_carry_unwind"),
    window_stats(df, "2022-01-01", "2022-12-31", "2022_bear_grind_down"),
]

# -------------------- 5. Kelly-style sizing --------------------
# We have empirical worst MaxDD = 15.5% at full sizing.
# Under leverage L: dd_levered approx = L * dd_base for a small base move,
# but for non-trivial drawdowns dd_levered = 1 - (1-dd_base)^L (equity-curve compounding).
# We solve for L such that worst_observed_dd at fraction L stays <= 25%.
# Use both: empirical full-sample maxDD and a stressed maxDD (worst regime + 50% buffer).

base_maxdd = abs(headline["max_dd"])  # positive
# stressed maxDD: imagine VIX_spike regime sustained — use worst rolling 60d return as stress proxy
df["roll60"] = df["ret"].fillna(0).rolling(60).sum()
stress_60d_loss = abs(min(df["roll60"].min(), 0.0))
stress_maxdd = max(base_maxdd, stress_60d_loss)  # use the worst

target_dd = 0.25


def levered_dd(base_dd: float, L: float) -> float:
    # 1 - (1 - base_dd)^L
    return 1.0 - (1.0 - base_dd) ** L


# solve L for base_dd
def solve_L(base_dd: float, cap: float) -> float:
    # binary search
    lo, hi = 0.01, 5.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if levered_dd(base_dd, mid) > cap:
            hi = mid
        else:
            lo = mid
    return lo


L_at_base = solve_L(base_maxdd, target_dd)
L_at_stress = solve_L(stress_maxdd, target_dd)
# Conservative recommendation: min of the two, then haircut 20% for model risk
L_rec = min(L_at_base, L_at_stress) * 0.8

kelly = {
    "observed_max_dd_pct": base_maxdd * 100,
    "stressed_60d_loss_pct": stress_60d_loss * 100,
    "stressed_max_dd_used_pct": stress_maxdd * 100,
    "target_max_dd_pct": target_dd * 100,
    "L_to_hit_25dd_at_observed_dd": L_at_base,
    "L_to_hit_25dd_at_stressed_dd": L_at_stress,
    "recommended_deployable_fraction": L_rec,
    "rationale": (
        "Worst observed in-sample drawdown was 15.5%. With 25% DD budget, a leverage of "
        f"{L_at_base:.2f}x would hit the cap in a repeat of in-sample worst. "
        f"Stressed scenario uses worst 60-day rolling loss ({stress_60d_loss*100:.1f}%) as a proxy "
        "for an out-of-sample fat tail; that gives "
        f"{L_at_stress:.2f}x. Recommendation is min of the two, haircut 20% for model-risk margin."
    ),
}

# -------------------- 6. regime-gap (HC #428 R1 style) --------------------
# Per user note, this strategy STRUCTURALLY fails the regime gate. Quantify it here for record.
green_days = df[df["spy_ret"] > 0]["ret"].dropna()
red_days = df[df["spy_ret"] < 0]["ret"].dropna()
sh_green = ann_sharpe(green_days)
sh_red = ann_sharpe(red_days)
denom = max(abs(sh_green), abs(sh_red), 1e-9)
regime_gap = abs(sh_green - sh_red) / denom

regime_gate_record = {
    "sharpe_green_days": sh_green,
    "sharpe_red_days": sh_red,
    "n_green": int(len(green_days)),
    "n_red": int(len(red_days)),
    "regime_gap": regime_gap,
    "hc428_r1_threshold": 0.50,
    "hc428_r1_pass": bool(regime_gap <= 0.50),
    "note": (
        "Short-vol/wheel is structurally long up-tape; failing HC#428 R1 was expected. "
        "This vol-regime stratification is the short-vol-appropriate substitute."
    ),
}

# -------------------- write outputs --------------------
results = {
    "source_equity_path": str(EQUITY_PATH),
    "headline": headline,
    "vix_regime_table": regime_rows,
    "tail_risk": {
        "worst_5_days": worst5_records,
        "var_95_daily_pct": var95 * 100,
        "cvar_95_daily_pct": cvar95 * 100,
        "top_5_drawdowns": dd_records,
    },
    "survivability": survive,
    "kelly_sizing": kelly,
    "hc428_r1_record": regime_gate_record,
    "timestamp": TS,
}

with open(OUT_DIR / "results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

# Plain-English summary.md
lines = []
lines.append(f"# Tier2 Balanced Scalp - Vol Regime + Tail Risk\n")
lines.append(f"Sharpe {headline['sharpe']:.2f}, CAGR {headline['cagr']*100:.1f}%, "
             f"MaxDD {headline['max_dd']*100:.1f}%, n={headline['n_days']} days\n\n")

lines.append("## VIX-regime Sharpe table\n")
lines.append("| Regime | n_days | mean% | std% | Sharpe | WR | worst day% |")
lines.append("|---|---|---|---|---|---|---|")
for r in regime_rows:
    if r["n_days"] == 0:
        lines.append(f"| {r['regime']} | 0 | - | - | - | - | - |")
        continue
    lines.append(
        f"| {r['regime']} | {r['n_days']} | {r['mean_daily_pct']:.3f} | {r['std_daily_pct']:.3f} | "
        f"{r['sharpe']:.2f} | {r['win_rate']*100:.1f}% | {r['worst_day_pct']:.2f} |"
    )
lines.append("")

lines.append("## 5 worst single-day losses\n")
lines.append("| date | wheel% | VIX close | VIX chg | SPY% |")
lines.append("|---|---|---|---|---|")
for w in worst5_records:
    lines.append(
        f"| {w['date']} | {w['wheel_ret_pct']:.2f} | "
        f"{w['vix_close']:.1f} | {w['vix_chg']:+.1f} | "
        f"{w['spy_ret_pct']:.2f} |"
    )
lines.append("")

lines.append("## Tail risk\n")
lines.append(f"- VaR-95 daily: {var95*100:.2f}%")
lines.append(f"- CVaR-95 daily: {cvar95*100:.2f}%\n")

lines.append("## Top 5 drawdowns (decomposed)\n")
lines.append("| start | trough | recover | maxDD% | VIX peak | regime |")
lines.append("|---|---|---|---|---|---|")
for d in dd_records:
    lines.append(
        f"| {d['start']} | {d['trough']} | {d['recover']} | "
        f"{d['max_dd_pct']:.2f} | {d['vix_peak_during_dd']:.1f} | {d['regime_label']} |"
    )
lines.append("")

lines.append("## Survivability\n")
for s in survive:
    if s.get("n_days", 0) == 0:
        continue
    lines.append(
        f"- **{s['label']}** ({s['window']}): maxDD {s['max_dd_pct']:.2f}%, "
        f"worst week {s['worst_week_pct']:.2f}%, VIX peak {s['vix_peak']:.1f}, "
        f"recovered {s['recover_date']} "
        f"({s['days_trough_to_recover']} days)" if s['days_trough_to_recover'] else
        f"- **{s['label']}** ({s['window']}): maxDD {s['max_dd_pct']:.2f}%"
    )
lines.append("")

lines.append("## Kelly-style sizing\n")
lines.append(f"- Observed max DD at 1.0x: {kelly['observed_max_dd_pct']:.2f}%")
lines.append(f"- Stressed 60d loss: {kelly['stressed_60d_loss_pct']:.2f}%")
lines.append(f"- 25% DD-cap leverage at observed: {kelly['L_to_hit_25dd_at_observed_dd']:.2f}x")
lines.append(f"- 25% DD-cap leverage at stress: {kelly['L_to_hit_25dd_at_stressed_dd']:.2f}x")
lines.append(f"- **Recommended deployable fraction: {kelly['recommended_deployable_fraction']:.2f}x** (20% model-risk haircut)\n")

lines.append("## HC #428 R1 record\n")
lines.append(
    f"- Sharpe green days: {regime_gate_record['sharpe_green_days']:.2f}  "
    f"red days: {regime_gate_record['sharpe_red_days']:.2f}  "
    f"gap: {regime_gate_record['regime_gap']:.2f}  "
    f"pass: {regime_gate_record['hc428_r1_pass']}"
)
lines.append(f"- {regime_gate_record['note']}\n")

with open(OUT_DIR / "summary.md", "w") as f:
    f.write("\n".join(lines))

print(f"[done] wrote {OUT_DIR}/results.json and summary.md")
print(f"\n--- HEADLINE ---")
print(json.dumps(headline, indent=2))
print(f"\n--- VIX REGIME ---")
for r in regime_rows:
    print(r)
print(f"\n--- WORST 5 DAYS ---")
for w in worst5_records:
    print(w)
print(f"\n--- DRAWDOWNS ---")
for d in dd_records:
    print(d)
print(f"\n--- SURVIVABILITY ---")
for s in survive:
    print(s)
print(f"\n--- KELLY ---")
print(json.dumps(kelly, indent=2))
print(f"\n--- HC428 R1 ---")
print(json.dumps(regime_gate_record, indent=2))
print(f"\nOUT_DIR={OUT_DIR}")

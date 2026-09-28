"""
Longer Horizon Trade Simulator v1
==================================
Uses streaming continuation predictions (h=30s, 60s) to simulate trades.

Model: XGBoost predicting mfe_minus_mae_ticks (net favorable excursion).
  - Spearman 0.463 (30s), 0.408 (60s), 0.374 (120s)
  - Label is UNSIGNED (always >= 0): represents how much net edge exists.
  - Higher prediction = bigger expected net move.

Trade Logic:
  When prediction is high (top N%), there IS a big move happening.
  The model predicts intensity, not direction. But mfe_minus_mae > 0 means
  the favorable move exceeds the adverse move — so IF you're on the right side,
  this is your expected edge.

  Approach 1 (conservative): Assume passive entry, use realized mfe_minus_mae
    as the gross P&L proxy (you capture the net excursion), subtract commission.
  Approach 2 (directional filter needed): Would require CNN-Mamba direction.
    We DO have CNN-Mamba bulk_oot predictions on Jupiter — we'll use those.

Cost: 0.376 ticks RT (passive entry + passive exit via limit, commission only).
      Also test 1.376 (passive entry + market exit = commission + 1 tick spread).
      Also test 1.752 (market entry + market exit = 2*commission + spread).

HC #428 acceptance gates:
  - Net > 0, PF >= 1.2, Sharpe >= 0.5
  - Regime asymmetry <= 0.50
  - Day concentration <= 0.70
  - Trades/day >= 5
"""

from __future__ import annotations
import gc
import json
import sys
import warnings
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────
PRED_DIR = Path("/home/jupiter/Lvl3Quant/data/models/streaming_continuation_v1")
CNN_MAMBA_DIR = Path("/home/jupiter/Lvl3Quant/output/cnn_mamba_v3_h30s_56day_inference_FETCHED.npz")
OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/longer_horizon_sim_v1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Cost Constants ───────────────────────────────────────────────────────
COST_PASSIVE_RT = 0.376     # Limit entry + limit exit (commission only)
COST_PASSIVE_MARKET = 1.376  # Limit entry + market exit
COST_MARKET_RT = 1.752       # Market entry + market exit

# ─── HC #428 Gates ────────────────────────────────────────────────────────
GATE_NET_POSITIVE = True
GATE_PF_MIN = 1.2
GATE_SHARPE_MIN = 0.5
GATE_REGIME_ASYM_MAX = 0.50
GATE_DAY_CONC_MAX = 0.70
GATE_TRADES_PER_DAY_MIN = 5


def load_predictions(horizon_s: int) -> pd.DataFrame:
    """Load streaming continuation predictions for a given horizon."""
    path = PRED_DIR / f"oos_preds_h{horizon_s}s.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    df = pd.read_parquet(path)
    print(f"  Loaded h={horizon_s}s: {len(df):,} predictions, {df['date'].nunique()} days")
    return df


def compute_metrics(trades_df: pd.DataFrame, n_days: int) -> dict:
    """Compute trading metrics from a trades DataFrame."""
    if len(trades_df) == 0:
        return {
            "n_trades": 0, "trades_per_day": 0, "net_ticks": 0,
            "mean_ticks": 0, "sharpe": 0, "sortino": 0, "pf": 0, "wr": 0,
            "day_conc": 1.0, "regime_asym": 1.0, "passes_gates": False,
        }

    pnl = trades_df["pnl_ticks"].values
    n_trades = len(pnl)
    trades_per_day = n_trades / max(n_days, 1)
    net_ticks = float(pnl.sum())
    mean_ticks = float(pnl.mean())

    # Daily P&L for Sharpe/Sortino
    daily_pnl = trades_df.groupby("date")["pnl_ticks"].sum()
    # Fill missing days with 0
    all_days = sorted(trades_df["date"].unique())
    daily_pnl = daily_pnl.reindex(all_days, fill_value=0.0)

    if len(daily_pnl) > 1 and daily_pnl.std() > 0:
        sharpe = float(daily_pnl.mean() / daily_pnl.std() * np.sqrt(252))
        downside = daily_pnl[daily_pnl < 0].std()
        sortino = float(daily_pnl.mean() / downside * np.sqrt(252)) if downside > 0 else 999.0
    else:
        sharpe = 0.0
        sortino = 0.0

    # Profit factor
    gross_wins = float(pnl[pnl > 0].sum())
    gross_losses = float(abs(pnl[pnl < 0].sum()))
    pf = gross_wins / gross_losses if gross_losses > 0 else (999.0 if gross_wins > 0 else 0.0)

    # Win rate
    wr = float((pnl > 0).mean())

    # Day concentration
    if net_ticks > 0:
        day_profits = daily_pnl[daily_pnl > 0]
        day_conc = float(day_profits.max() / day_profits.sum()) if len(day_profits) > 0 else 1.0
    else:
        day_conc = 1.0

    # Regime stratification (classify days by net daily P&L of the instrument)
    # We use the mean of actuals as a proxy for market direction
    day_actual_mean = trades_df.groupby("date")["actual"].mean()
    green_days = day_actual_mean[day_actual_mean > day_actual_mean.median()].index
    red_days = day_actual_mean[day_actual_mean <= day_actual_mean.median()].index

    green_pnl = daily_pnl.reindex(green_days, fill_value=0.0)
    red_pnl = daily_pnl.reindex(red_days, fill_value=0.0)

    if len(green_pnl) > 1 and green_pnl.std() > 0:
        sharpe_green = float(green_pnl.mean() / green_pnl.std() * np.sqrt(252))
    else:
        sharpe_green = 0.0
    if len(red_pnl) > 1 and red_pnl.std() > 0:
        sharpe_red = float(red_pnl.mean() / red_pnl.std() * np.sqrt(252))
    else:
        sharpe_red = 0.0

    max_abs = max(abs(sharpe_green), abs(sharpe_red), 1e-6)
    regime_asym = abs(sharpe_green - sharpe_red) / max_abs

    # HC #428 gate check
    passes = (
        net_ticks > 0
        and pf >= GATE_PF_MIN
        and sharpe >= GATE_SHARPE_MIN
        and regime_asym <= GATE_REGIME_ASYM_MAX
        and day_conc <= GATE_DAY_CONC_MAX
        and trades_per_day >= GATE_TRADES_PER_DAY_MIN
    )

    return {
        "n_trades": n_trades,
        "trades_per_day": round(trades_per_day, 1),
        "net_ticks": round(net_ticks, 2),
        "mean_ticks": round(mean_ticks, 3),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "pf": round(pf, 2),
        "wr": round(wr, 4),
        "day_conc": round(day_conc, 3),
        "sharpe_green": round(sharpe_green, 2),
        "sharpe_red": round(sharpe_red, 2),
        "regime_asym": round(regime_asym, 3),
        "n_days": len(all_days),
        "passes_gates": passes,
    }


def run_intensity_sim(df: pd.DataFrame, horizon_s: int, pctile: int,
                      cost_ticks: float, cost_label: str,
                      side_filter: str = "both") -> dict:
    """
    Simulate trades using intensity predictions.

    Entry: when prediction > percentile threshold.
    P&L: realized mfe_minus_mae - cost (this is the available edge at the horizon).

    side_filter: 'both', 'long', 'short' — for future when we add directional info.
    For now 'both' uses all signals since intensity is non-directional.
    """
    # Compute threshold
    threshold = np.percentile(df["pred"].values, pctile)

    # Select trades
    mask = df["pred"] >= threshold
    trades = df[mask].copy()

    if len(trades) == 0:
        return {
            "horizon_s": horizon_s,
            "pctile": pctile,
            "cost_model": cost_label,
            "cost_ticks": cost_ticks,
            "side": side_filter,
            "threshold": round(float(threshold), 3),
            "metrics": compute_metrics(pd.DataFrame(columns=["pnl_ticks", "date", "actual"]), 0),
        }

    # P&L: realized mfe_minus_mae - cost
    trades["pnl_ticks"] = trades["actual"] - cost_ticks

    n_days = df["date"].nunique()
    metrics = compute_metrics(trades, n_days)

    return {
        "horizon_s": horizon_s,
        "pctile": pctile,
        "cost_model": cost_label,
        "cost_ticks": cost_ticks,
        "side": side_filter,
        "threshold": round(float(threshold), 3),
        "metrics": metrics,
    }


def run_cooldown_sim(df: pd.DataFrame, horizon_s: int, pctile: int,
                     cost_ticks: float, cost_label: str,
                     cooldown_events: int = 120) -> dict:
    """
    More realistic: after entry, enforce a cooldown period (= horizon).
    No overlapping trades. Events are ~250ms apart, so:
      30s horizon = 120 events cooldown
      60s horizon = 240 events cooldown
      120s horizon = 480 events cooldown
    """
    threshold = np.percentile(df["pred"].values, pctile)

    all_trades = []
    for date_str, day_df in df.groupby("date"):
        day_preds = day_df["pred"].values
        day_actuals = day_df["actual"].values

        i = 0
        while i < len(day_preds):
            if day_preds[i] >= threshold:
                pnl = day_actuals[i] - cost_ticks
                all_trades.append({
                    "date": date_str,
                    "pred": float(day_preds[i]),
                    "actual": float(day_actuals[i]),
                    "pnl_ticks": float(pnl),
                })
                i += cooldown_events  # skip cooldown period
            else:
                i += 1

    trades_df = pd.DataFrame(all_trades)
    n_days = df["date"].nunique()
    metrics = compute_metrics(trades_df, n_days) if len(trades_df) > 0 else compute_metrics(
        pd.DataFrame(columns=["pnl_ticks", "date", "actual"]), 0
    )

    return {
        "horizon_s": horizon_s,
        "pctile": pctile,
        "cost_model": cost_label,
        "cost_ticks": cost_ticks,
        "cooldown_events": cooldown_events,
        "side": "both_cooldown",
        "threshold": round(float(threshold), 3),
        "metrics": metrics,
    }


def per_day_breakdown(df: pd.DataFrame, pctile: int, cost_ticks: float,
                      cooldown_events: int) -> pd.DataFrame:
    """Detailed per-day results for the best config."""
    threshold = np.percentile(df["pred"].values, pctile)

    rows = []
    for date_str, day_df in df.groupby("date"):
        day_preds = day_df["pred"].values
        day_actuals = day_df["actual"].values

        trades = []
        i = 0
        while i < len(day_preds):
            if day_preds[i] >= threshold:
                pnl = day_actuals[i] - cost_ticks
                trades.append(pnl)
                i += cooldown_events
            else:
                i += 1

        if trades:
            trades_arr = np.array(trades)
            rows.append({
                "date": date_str,
                "n_trades": len(trades),
                "net_ticks": round(float(trades_arr.sum()), 2),
                "mean_ticks": round(float(trades_arr.mean()), 2),
                "wr": round(float((trades_arr > 0).mean()), 3),
                "max_win": round(float(trades_arr.max()), 2),
                "max_loss": round(float(trades_arr.min()), 2),
            })
        else:
            rows.append({
                "date": date_str,
                "n_trades": 0,
                "net_ticks": 0.0,
                "mean_ticks": 0.0,
                "wr": 0.0,
                "max_win": 0.0,
                "max_loss": 0.0,
            })

    return pd.DataFrame(rows)


def main():
    print("=" * 70)
    print("LONGER HORIZON TRADE SIMULATOR v1")
    print("=" * 70)
    print()

    # Load data
    results = []
    per_day_results = {}

    for horizon_s in [30, 60]:
        print(f"\n{'='*60}")
        print(f"HORIZON = {horizon_s}s")
        print(f"{'='*60}")

        df = load_predictions(horizon_s)

        # Basic stats
        rho, _ = spearmanr(df["pred"], df["actual"])
        print(f"  Concat Spearman: {rho:.4f}")
        print(f"  Pred: mean={df['pred'].mean():.2f}, std={df['pred'].std():.2f}")
        print(f"  Actual: mean={df['actual'].mean():.2f}, std={df['actual'].std():.2f}")
        print(f"  Days: {df['date'].nunique()}")
        print(f"  Samples: {len(df):,}")

        # Decile analysis
        print(f"\n  Decile Analysis:")
        deciles = pd.qcut(df["pred"], 10, labels=False, duplicates="drop")
        for d in range(10):
            mask = deciles == d
            pred_range = f"[{df.loc[mask, 'pred'].min():.1f}, {df.loc[mask, 'pred'].max():.1f})"
            realized = df.loc[mask, "actual"].mean()
            print(f"    D{d}: {pred_range} → realized={realized:.2f}t")

        # Percentile thresholds
        for pct in [80, 85, 90, 95, 99]:
            thr = np.percentile(df["pred"].values, pct)
            n_above = (df["pred"] >= thr).sum()
            mean_actual = df.loc[df["pred"] >= thr, "actual"].mean()
            print(f"    P{pct}: threshold={thr:.2f}, n={n_above:,}, mean_actual={mean_actual:.2f}t")

        # Cooldown = horizon_s / 0.25 (events are ~250ms apart)
        cooldown = int(horizon_s / 0.25)
        print(f"\n  Cooldown: {cooldown} events ({horizon_s}s)")

        # Run sweep
        print(f"\n  --- SWEEP ---")
        cost_configs = [
            (COST_PASSIVE_RT, "passive_rt"),
            (COST_PASSIVE_MARKET, "passive_entry_market_exit"),
            (COST_MARKET_RT, "market_rt"),
        ]

        for cost_ticks, cost_label in cost_configs:
            for pctile in [80, 85, 90, 95, 99]:
                # Without cooldown (overlapping — upper bound)
                result = run_intensity_sim(df, horizon_s, pctile, cost_ticks, cost_label)
                results.append(result)

                # With cooldown (realistic — non-overlapping)
                result_cd = run_cooldown_sim(df, horizon_s, pctile, cost_ticks, cost_label, cooldown)
                results.append(result_cd)

                m = result_cd["metrics"]
                flag = " ✓ PASS" if m["passes_gates"] else ""
                print(f"    h={horizon_s}s | P{pctile} | cost={cost_label} | "
                      f"trades/d={m['trades_per_day']} | "
                      f"mean={m['mean_ticks']:.2f}t | net={m['net_ticks']:.0f}t | "
                      f"Sharpe={m['sharpe']:.1f} | PF={m['pf']:.1f} | "
                      f"WR={m['wr']:.1%} | dayConc={m['day_conc']:.2f}{flag}")

        # Per-day breakdown for best realistic config (P90, passive RT, with cooldown)
        pd_df = per_day_breakdown(df, 90, COST_PASSIVE_RT, cooldown)
        per_day_results[f"h{horizon_s}s_p90_passive"] = pd_df

        print(f"\n  --- PER-DAY BREAKDOWN (P90, passive RT, cooldown) ---")
        for _, row in pd_df.iterrows():
            print(f"    {row['date']}: {row['n_trades']} trades, "
                  f"net={row['net_ticks']:.1f}t, wr={row['wr']:.0%}")

        del df
        gc.collect()

    # ─── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SUMMARY — CONFIGS PASSING HC #428 GATES")
    print(f"{'='*70}")

    passing = [r for r in results if r["metrics"]["passes_gates"]]
    if not passing:
        print("  NO configs pass all HC #428 gates.")
        # Show best anyway
        cooldown_results = [r for r in results if "cooldown" in r.get("side", "")]
        if cooldown_results:
            best = max(cooldown_results, key=lambda r: r["metrics"]["sharpe"])
            m = best["metrics"]
            print(f"\n  Best cooldown config (does NOT pass all gates):")
            print(f"    h={best['horizon_s']}s | P{best['pctile']} | cost={best['cost_model']}")
            print(f"    trades/d={m['trades_per_day']} | mean={m['mean_ticks']:.2f}t")
            print(f"    Sharpe={m['sharpe']:.2f} | Sortino={m['sortino']:.2f} | PF={m['pf']:.2f} | WR={m['wr']:.1%}")
            print(f"    dayConc={m['day_conc']:.3f} | regime_asym={m['regime_asym']:.3f}")
            print(f"    Sharpe_green={m['sharpe_green']:.2f} | Sharpe_red={m['sharpe_red']:.2f}")
            # Identify which gates failed
            fails = []
            if m['net_ticks'] <= 0: fails.append("net<=0")
            if m['pf'] < GATE_PF_MIN: fails.append(f"PF={m['pf']:.2f}<{GATE_PF_MIN}")
            if m['sharpe'] < GATE_SHARPE_MIN: fails.append(f"Sharpe={m['sharpe']:.2f}<{GATE_SHARPE_MIN}")
            if m['regime_asym'] > GATE_REGIME_ASYM_MAX: fails.append(f"regimeAsym={m['regime_asym']:.3f}>{GATE_REGIME_ASYM_MAX}")
            if m['day_conc'] > GATE_DAY_CONC_MAX: fails.append(f"dayConc={m['day_conc']:.3f}>{GATE_DAY_CONC_MAX}")
            if m['trades_per_day'] < GATE_TRADES_PER_DAY_MIN: fails.append(f"trades/d={m['trades_per_day']}<{GATE_TRADES_PER_DAY_MIN}")
            print(f"    Failed gates: {', '.join(fails)}")
    else:
        # Sort by Sharpe
        passing.sort(key=lambda r: r["metrics"]["sharpe"], reverse=True)
        for r in passing[:10]:
            m = r["metrics"]
            cd = f" (cooldown={r.get('cooldown_events', 'N/A')})" if "cooldown" in r.get("side", "") else " (overlapping)"
            print(f"  h={r['horizon_s']}s | P{r['pctile']} | cost={r['cost_model']}{cd}")
            print(f"    trades/d={m['trades_per_day']} | mean={m['mean_ticks']:.2f}t | net={m['net_ticks']:.0f}t")
            print(f"    Sharpe={m['sharpe']:.2f} | Sortino={m['sortino']:.2f} | PF={m['pf']:.2f} | WR={m['wr']:.1%}")
            print(f"    dayConc={m['day_conc']:.3f} | regime_asym={m['regime_asym']:.3f}")
            print(f"    Sharpe_green={m['sharpe_green']:.2f} | Sharpe_red={m['sharpe_red']:.2f}")
            print()

    # ─── CRITICAL DIAGNOSTIC ─────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("CRITICAL DIAGNOSTIC: Is mfe_minus_mae Actually Capturable?")
    print(f"{'='*70}")
    print("""
  mfe_minus_mae = max_favorable_excursion - max_adverse_excursion within horizon.
  This is the THEORETICAL maximum P&L if you entered at exactly the right time
  and exited at exactly the right time. In practice:

  1. You can't always capture full MFE (price may bounce before you exit)
  2. MFE timing != horizon end (MFE may occur anywhere within the window)
  3. You need DIRECTION — mfe_minus_mae doesn't tell you which side

  The 98%+ win rates above suggest this is closer to a theoretical upper
  bound than a realizable strategy. The key question is: what fraction
  of mfe_minus_mae can you actually capture with a hold-to-horizon strategy?

  This sim uses realized mfe_minus_mae as a CEILING on potential P&L.
  Real P&L with hold-to-horizon would be: abs(return_at_horizon) which
  is always <= mfe_minus_mae. A directional model is needed to know
  whether to go long or short.
    """)

    # ─── Save results ─────────────────────────────────────────────────────
    output = {
        "description": "Longer Horizon Trade Sim v1 — streaming continuation predictions",
        "model": "XGBoost mfe_minus_mae (streaming continuation v1)",
        "horizons_tested": [30, 60],
        "cost_models": {
            "passive_rt": COST_PASSIVE_RT,
            "passive_entry_market_exit": COST_PASSIVE_MARKET,
            "market_rt": COST_MARKET_RT,
        },
        "n_configs_tested": len(results),
        "n_passing_hc428": len(passing) if passing else 0,
        "all_results": results,
        "per_day": {k: v.to_dict(orient="records") for k, v in per_day_results.items()},
    }

    out_path = OUTPUT_DIR / "sim_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")

    # Also save to the user-requested path
    out_path2 = Path("/home/jupiter/Lvl3Quant/output/longer_horizon_sim_v1.json")
    with open(out_path2, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Results saved to {out_path2}")

    print(f"\nDone.")


if __name__ == "__main__":
    main()

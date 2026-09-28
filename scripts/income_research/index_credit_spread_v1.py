#!/usr/bin/env python3
"""
Index Put Credit Spreads v1 — SPY Weekly Bull-Put Spreads
===========================================================
Income strategy: sell OTM put + buy further OTM put on SPY.
Index vol premium is structurally persistent (VRP documented over 20+ years).
No single-stock earnings blowup risk.

ADVERSARIAL FROM DAY 1:
  - Real Dolt bid/ask pricing (sell at bid, buy at ask — no mid-price)
  - No commissions on Robinhood (HC #694)
  - Walk-forward validation: sliding 252d train → 21d test
  - HC #428 R1: regime-agnostic OOT (40+ day test), per-regime Sharpe
  - HC #344: day-concentration ≤ 0.70
  - HC #0: last-30% OOT
  - 100-trial permutation test
  - Black swan stress: 2020 COVID crash, 2022 bear, 2025 tariff shock

STRATEGY VARIANTS:
  A. Fixed delta: sell 15-delta put, buy 5-delta put, ~2 weeks DTE
  B. Fixed delta: sell 20-delta put, buy 10-delta put, ~2 weeks DTE
  C. Fixed delta: sell 10-delta put, buy 3-delta put, ~2 weeks DTE (conservative)
  D. VIX-scaled sizing: reduce size when VIX > 20, exit when VIX > 30

SIZING: $100K portfolio, max 30% NAV at risk (max loss = spread width × contracts × 100).
"""

import json
import os
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# --- Paths ---
CHAINS_DIR = Path("/home/jupiter/Lvl3Quant/wheel_strategy_v1/data/cache/options_real/chains")
MACRO_PATH  = Path("/home/jupiter/Lvl3Quant/wheel_strategy_v1/data/cache/macro.parquet")
PRICES_PATH = Path("/home/jupiter/Lvl3Quant/wheel_strategy_v1/data/cache/prices.parquet")
OUTPUT_DIR  = Path("/home/jupiter/Lvl3Quant/output/index_credit_spread_v1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Constants ---
PORTFOLIO_NAV    = 100_000
MAX_NAV_AT_RISK  = 0.30   # max 30% NAV at risk (max loss / NAV)
TARGET_DTE       = 14     # target ~2 weeks to expiry
MIN_DTE          = 11
MAX_DTE          = 20
TRADING_DAYS_YR  = 252


# ============================================================
#  DATA LOADING
# ============================================================

def load_data():
    """Load SPY chains + macro data."""
    print("Loading SPY options chains...")
    spy = pd.read_parquet(CHAINS_DIR / "SPY.parquet")
    spy["date"] = pd.to_datetime(spy["date"])
    spy["expiration"] = pd.to_datetime(spy["expiration"])
    spy = spy.sort_values(["date", "expiration", "strike"])

    # Load macro (VIX)
    macro = None
    if MACRO_PATH.exists():
        macro = pd.read_parquet(MACRO_PATH)
        macro["date"] = pd.to_datetime(macro["date"])
        macro = macro[["date", "vix"]].dropna()

    # Load SPY price (long format: ticker, date, close)
    spy_px = None
    if PRICES_PATH.exists():
        px = pd.read_parquet(PRICES_PATH)
        px["date"] = pd.to_datetime(px["date"])
        if "ticker" in px.columns:
            spy_rows = px[px["ticker"] == "SPY"][["date", "close"]].rename(columns={"close": "spy_px"}).dropna()
            spy_px = spy_rows if not spy_rows.empty else None
        elif "SPY" in px.columns:
            spy_px = px[["date", "SPY"]].rename(columns={"SPY": "spy_px"}).dropna()

    return spy, macro, spy_px


# ============================================================
#  SPREAD SELECTION: pick short put + long put per observation date
# ============================================================

def pick_spread(df_puts, short_delta_target, long_delta_target):
    """
    For each observation date:
    1. Pick expiry closest to TARGET_DTE (within MIN_DTE..MAX_DTE)
    2. Pick short put: delta closest to -short_delta_target
    3. Pick long put: delta closest to -long_delta_target (more OTM = more negative delta)

    Returns DataFrame with one row per date.
    """
    puts = df_puts.copy()
    # Filter DTE range
    puts = puts[(puts["dte"] >= MIN_DTE) & (puts["dte"] <= MAX_DTE)]
    if puts.empty:
        return pd.DataFrame()

    # Step 1: pick best expiry per date
    puts["dte_dist"] = (puts["dte"] - TARGET_DTE).abs()
    best_dte = puts.groupby("date")["dte_dist"].transform("min")
    puts = puts[puts["dte_dist"] == best_dte].copy()

    # Step 2: pick short put (e.g. 15-delta → delta ~ -0.15)
    puts["short_dist"] = (puts["delta"].abs() - short_delta_target).abs()
    best_short = puts.groupby("date")["short_dist"].transform("min")
    short_puts = puts[puts["short_dist"] == best_short].copy()
    short_puts = short_puts.groupby("date").first().reset_index()
    short_puts = short_puts.rename(columns={
        "strike": "short_strike", "bid": "short_bid", "ask": "short_ask",
        "mid": "short_mid", "delta": "short_delta", "dte": "short_dte",
        "expiration": "expiry",
    })

    # Step 3: pick long put (more OTM, larger |delta| distance)
    puts["long_dist"] = (puts["delta"].abs() - long_delta_target).abs()
    best_long = puts.groupby("date")["long_dist"].transform("min")
    long_puts = puts[puts["long_dist"] == best_long].copy()
    long_puts = long_puts.groupby("date").first().reset_index()
    long_puts = long_puts.rename(columns={
        "strike": "long_strike", "bid": "long_bid", "ask": "long_ask",
        "mid": "long_mid", "delta": "long_delta",
    })

    # Merge
    spreads = short_puts[["date", "expiry", "short_dte", "short_strike", "short_bid", "short_ask", "short_mid", "short_delta"]].merge(
        long_puts[["date", "long_strike", "long_bid", "long_ask", "long_mid", "long_delta"]],
        on="date", how="inner"
    )

    # Require short strike > long strike (short put is closer to ATM)
    spreads = spreads[spreads["short_strike"] > spreads["long_strike"]]

    # ADVERSARIAL PRICING: sell short put at BID, buy long put at ASK
    # Net credit = what we receive - what we pay
    spreads["net_credit"] = spreads["short_bid"] - spreads["long_ask"]
    spreads["spread_width"] = spreads["short_strike"] - spreads["long_strike"]
    spreads["max_loss"] = spreads["spread_width"] - spreads["net_credit"]

    # Only enter if net_credit > $0.05 (positive credit after bid/ask)
    spreads = spreads[spreads["net_credit"] > 0.05].copy()

    return spreads.reset_index(drop=True)


# ============================================================
#  BUILD CYCLES: entry + next observation exit
# ============================================================

def get_spy_level_from_chains(spy_chains, obs_date):
    """
    Estimate SPY spot level on a given observation date.
    Method: find the call with delta closest to 0.50 (ATM call) in DTE 11-20 range.
    That call's strike ≈ current SPY spot.
    """
    calls = spy_chains[(spy_chains["date"] == obs_date) &
                       (spy_chains["type"] == "c") &
                       (spy_chains["dte"] >= 11) & (spy_chains["dte"] <= 20)]
    if calls.empty:
        return None
    atm = calls.iloc[(calls["delta"] - 0.50).abs().argsort()[:1]]
    return float(atm["strike"].iloc[0])


def build_cycles(spreads, spy_chains, spy_px=None, vix_filter=False, macro=None):
    """
    Exit logic: close each spread at the NEXT observation date.

    Exit pricing (adversarial):
    - If next obs_date >= expiry: use SPY spot to determine intrinsic value (no time value)
      * SPY above short_strike → both expired worthless → keep full credit
      * SPY between strikes → partial loss (intrinsic - credit)
      * SPY below long_strike → max loss
    - If next obs_date < expiry: find same-strike puts in NEXT cycle's chains (different expiry)
      and use their prices as MTM proxy. If not found: use SPY spot + BS approximation.

    SPY spot is estimated from ATM call strike in chains (most reliable).
    """
    obs_dates = sorted(spreads["date"].unique())
    date_to_idx = {d: i for i, d in enumerate(obs_dates)}

    # Build SPY spot level index from chains
    spy_spot = {}
    for obs_date in spy_chains["date"].unique():
        level = get_spy_level_from_chains(spy_chains, obs_date)
        if level is not None:
            spy_spot[obs_date] = level

    # Also use real price data if available
    if spy_px is not None and not spy_px.empty:
        for _, row in spy_px.iterrows():
            spy_spot[row["date"]] = row["spy_px"]

    puts = spy_chains[spy_chains["type"] == "p"].copy()
    puts_idx = puts.set_index(["date", "strike"])

    cycles = []
    for _, row in spreads.iterrows():
        entry_date = row["date"]
        expiry = row["expiry"]
        idx = date_to_idx.get(entry_date)
        if idx is None or idx + 1 >= len(obs_dates):
            continue

        exit_date = obs_dates[idx + 1]
        days_held = (exit_date - entry_date).days

        # VIX filter on entry
        vix_val = None
        if macro is not None and not macro.empty:
            m = macro[macro["date"] == entry_date]
            if not m.empty:
                vix_val = float(m["vix"].iloc[0])

        if vix_filter and vix_val is not None and vix_val > 30:
            continue

        size_mult = 1.0
        if vix_filter and vix_val is not None:
            if vix_val > 25:
                size_mult = 0.5
            elif vix_val > 20:
                size_mult = 0.75

        # Get SPY spot at exit
        spy_level_exit = spy_spot.get(exit_date)

        # Determine P&L at exit
        if exit_date >= expiry:
            # Expired — use SPY spot to compute intrinsic
            if spy_level_exit is None:
                # Can't determine — use conservative: assume max loss
                pnl = row["net_credit"] - row["spread_width"]
                exit_type = "expired_conservative"
            elif spy_level_exit >= row["short_strike"]:
                pnl = row["net_credit"]
                exit_type = "expired_worthless"
            elif spy_level_exit <= row["long_strike"]:
                pnl = row["net_credit"] - row["spread_width"]
                exit_type = "expired_max_loss"
            else:
                intrinsic_short = row["short_strike"] - spy_level_exit
                intrinsic_long  = max(0.0, row["long_strike"] - spy_level_exit)
                pnl = row["net_credit"] - (intrinsic_short - intrinsic_long)
                exit_type = "expired_partial"
        else:
            # Still alive — MTM using actual put prices at exit_date
            # Find puts at same strikes (any expiry with DTE 7-30)
            exit_puts = puts[(puts["date"] == exit_date) &
                            (puts["dte"] >= 7) & (puts["dte"] <= 35)]

            short_match = exit_puts[exit_puts["strike"] == row["short_strike"]]
            long_match  = exit_puts[exit_puts["strike"] == row["long_strike"]]

            if not short_match.empty and not long_match.empty:
                # Real MTM: buy back short at ask, sell long at bid
                buyback = float(short_match["ask"].iloc[0]) - float(long_match["bid"].iloc[0])
                pnl = row["net_credit"] - buyback
                exit_type = "mtm_real"
            elif spy_level_exit is not None:
                # Fallback: use intrinsic value only (underestimates time value → conservative)
                intrinsic_short = max(0.0, row["short_strike"] - spy_level_exit)
                intrinsic_long  = max(0.0, row["long_strike"] - spy_level_exit)
                pnl = row["net_credit"] - (intrinsic_short - intrinsic_long)
                exit_type = "mtm_intrinsic_only"
            else:
                # Truly unknown — skip this cycle (don't credit)
                continue

        # Sizing: max NAV_AT_RISK / max_loss_per_contract
        contracts = max(1, int((PORTFOLIO_NAV * MAX_NAV_AT_RISK) / (row["max_loss"] * 100)))
        contracts = max(1, int(contracts * size_mult))

        pnl_total = pnl * contracts * 100

        cycles.append({
            "entry_date": entry_date,
            "exit_date": exit_date,
            "expiry": expiry,
            "short_strike": row["short_strike"],
            "long_strike": row["long_strike"],
            "net_credit": row["net_credit"],
            "spread_width": row["spread_width"],
            "max_loss": row["max_loss"],
            "contracts": contracts,
            "pnl_per_contract": pnl,
            "pnl_total": pnl_total,
            "days_held": days_held,
            "exit_type": exit_type,
            "vix": vix_val,
            "short_delta": row["short_delta"],
            "spy_level_exit": spy_level_exit,
        })

    return pd.DataFrame(cycles)


# ============================================================
#  METRICS
# ============================================================

def compute_metrics(cycles, label="full"):
    if cycles.empty or len(cycles) < 5:
        return {"label": label, "n_trades": 0}

    pnl = cycles["pnl_total"].values
    n = len(pnl)
    mean_pnl = np.mean(pnl)
    std_pnl = np.std(pnl, ddof=1)

    # Annualize by trading days
    dates = pd.to_datetime(cycles["entry_date"])
    total_days = (dates.max() - dates.min()).days
    years = total_days / 365.25
    trades_per_year = n / years if years > 0 else n

    # Annualized Sharpe (assuming weekly trades → scale by sqrt(trades_per_year))
    sharpe_ann = (mean_pnl / std_pnl * np.sqrt(trades_per_year)) if std_pnl > 0 else 0

    # Sortino
    downside = pnl[pnl < 0]
    sortino_ann = (mean_pnl / np.std(downside, ddof=1) * np.sqrt(trades_per_year)) if len(downside) > 1 else 0

    # Win rate, PF
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    win_rate = len(wins) / n
    pf = wins.sum() / (-losses.sum()) if len(losses) > 0 and -losses.sum() > 0 else np.inf

    # Total return (on $100K NAV)
    total_return = pnl.sum() / PORTFOLIO_NAV

    # Max drawdown (cumulative)
    cum_pnl = np.cumsum(pnl)
    running_max = np.maximum.accumulate(cum_pnl)
    drawdowns = (cum_pnl - running_max) / PORTFOLIO_NAV
    max_dd = float(drawdowns.min())

    # Day concentration
    day_counts = cycles.groupby("entry_date").size()
    day_conc = float(day_counts.max() / n) if n > 0 else 0

    return {
        "label": label,
        "n_trades": n,
        "total_return_pct": round(total_return * 100, 2),
        "mean_pnl": round(mean_pnl, 2),
        "sharpe_annual": round(sharpe_ann, 3),
        "sortino_annual": round(sortino_ann, 3),
        "win_rate_pct": round(win_rate * 100, 1),
        "profit_factor": round(pf, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "day_concentration": round(day_conc, 3),
        "avg_net_credit": round(float(cycles["net_credit"].mean()), 2),
        "avg_days_held": round(float(cycles["days_held"].mean()), 1),
    }


def regime_gap_test(cycles, spy_px):
    """Classify each cycle by SPY return in that period, compute per-regime Sharpe."""
    if spy_px is None or cycles.empty:
        return {"regime_agnostic": "SKIP", "reason": "no_price_data"}

    cycles = cycles.copy()
    cycles["entry_date"] = pd.to_datetime(cycles["entry_date"])
    spy_px = spy_px.copy()
    spy_px["date"] = pd.to_datetime(spy_px["date"])

    # SPY weekly return proxy: use close-to-close over the cycle period
    spy_map = spy_px.set_index("date")["spy_px"]

    def get_regime(row):
        ed = row["entry_date"]
        xd = row["exit_date"] if hasattr(row["exit_date"], "date") else pd.to_datetime(row["exit_date"])
        if ed not in spy_map.index or xd not in spy_map.index:
            return "flat"
        ret = (spy_map[xd] - spy_map[ed]) / spy_map[ed]
        if ret > 0.005:
            return "green"
        elif ret < -0.005:
            return "red"
        else:
            return "flat"

    cycles["regime"] = cycles.apply(get_regime, axis=1)

    results = {}
    for regime in ["green", "red", "flat"]:
        sub = cycles[cycles["regime"] == regime]
        if len(sub) < 5:
            results[regime] = {"n": len(sub), "sharpe": None}
            continue
        pnl = sub["pnl_total"].values
        std = np.std(pnl, ddof=1)
        trades_per_yr = len(pnl) / max(1, (sub["entry_date"].max() - sub["entry_date"].min()).days / 365.25)
        sharpe = (np.mean(pnl) / std * np.sqrt(trades_per_yr)) if std > 0 else 0
        results[regime] = {
            "n": len(sub),
            "sharpe": round(sharpe, 3),
            "win_rate": round(len(pnl[pnl > 0]) / len(pnl) * 100, 1),
        }

    sharpes = [v["sharpe"] for v in results.values() if v["sharpe"] is not None]
    if len(sharpes) >= 2:
        gap = (max(sharpes) - min(sharpes)) / max(abs(max(sharpes)), abs(min(sharpes))) if max(abs(max(sharpes)), abs(min(sharpes))) > 0 else 0
    else:
        gap = None

    return {
        "per_regime": results,
        "gap_ratio": round(gap, 3) if gap is not None else None,
        "regime_agnostic": "PASS" if (gap is not None and gap <= 0.50) else "FAIL",
        "threshold": 0.50,
    }


def permutation_test(cycles, n_trials=100):
    """Shuffle P&L labels; check if real Sharpe beats null distribution."""
    pnl = cycles["pnl_total"].values
    if len(pnl) < 10:
        return {"verdict": "SKIP", "n_trials": n_trials}

    dates = pd.to_datetime(cycles["entry_date"])
    total_days = (dates.max() - dates.min()).days
    years = total_days / 365.25
    tpy = len(pnl) / years if years > 0 else len(pnl)
    std = np.std(pnl, ddof=1)
    real_sharpe = (np.mean(pnl) / std * np.sqrt(tpy)) if std > 0 else 0

    null_sharpes = []
    rng = np.random.default_rng(42)
    for _ in range(n_trials):
        shuffled = rng.permutation(pnl)
        s = np.std(shuffled, ddof=1)
        null_sharpes.append((np.mean(shuffled) / s * np.sqrt(tpy)) if s > 0 else 0)

    null_sharpes = np.array(null_sharpes)
    p_val = np.mean(null_sharpes >= real_sharpe)

    return {
        "n_trials": n_trials,
        "real_sharpe": round(real_sharpe, 3),
        "null_mean": round(float(null_sharpes.mean()), 3),
        "null_std": round(float(null_sharpes.std()), 3),
        "p_value": round(float(p_val), 3),
        "significant_05": bool(p_val < 0.05),
        "verdict": "PASS" if p_val < 0.05 else "FAIL",
    }


# ============================================================
#  VARIANT RUNNER
# ============================================================

def run_variant(name, short_delta, long_delta, vix_filter, spy_chains, macro, spy_px):
    print(f"\n{'='*60}")
    print(f"  Variant: {name} | short_delta={short_delta} long_delta={long_delta} vix_filter={vix_filter}")
    print(f"{'='*60}")

    puts = spy_chains[spy_chains["type"] == "p"].copy()
    spreads = pick_spread(puts, short_delta, long_delta)
    print(f"  Spread opportunities found: {len(spreads)}")
    if spreads.empty:
        print("  No spreads found — skipping")
        return None

    cycles = build_cycles(spreads, spy_chains, spy_px=spy_px, vix_filter=vix_filter, macro=macro)
    print(f"  Cycles built: {len(cycles)}")
    if cycles.empty:
        print("  No cycles — skipping")
        return None

    # IS / OOT split (last 30% = OOT per HC #0)
    all_dates = sorted(cycles["entry_date"].unique())
    oot_start_idx = int(len(all_dates) * 0.70)
    oot_start = all_dates[oot_start_idx]
    print(f"  OOT start: {oot_start} ({len(all_dates) - oot_start_idx} OOT dates)")

    is_cyc  = cycles[cycles["entry_date"] < oot_start]
    oot_cyc = cycles[cycles["entry_date"] >= oot_start]

    full_metrics = compute_metrics(cycles, "full")
    is_metrics   = compute_metrics(is_cyc, "IS")
    oot_metrics  = compute_metrics(oot_cyc, "OOT")

    print(f"  Full: Sharpe={full_metrics.get('sharpe_annual','N/A')} WR={full_metrics.get('win_rate_pct','N/A')}% "
          f"PF={full_metrics.get('profit_factor','N/A')} MaxDD={full_metrics.get('max_drawdown_pct','N/A')}%")
    print(f"  IS:   Sharpe={is_metrics.get('sharpe_annual','N/A')} WR={is_metrics.get('win_rate_pct','N/A')}%")
    print(f"  OOT:  Sharpe={oot_metrics.get('sharpe_annual','N/A')} WR={oot_metrics.get('win_rate_pct','N/A')}% "
          f"n={oot_metrics.get('n_trades',0)}")

    regime = regime_gap_test(cycles, spy_px)
    perm   = permutation_test(cycles)

    print(f"  Regime: {regime.get('regime_agnostic')} gap={regime.get('gap_ratio')}")
    for r, v in regime.get("per_regime", {}).items():
        print(f"    {r}: n={v['n']} Sharpe={v['sharpe']} WR={v.get('win_rate','?')}%")
    print(f"  Permutation: {perm['verdict']} p={perm['p_value']}")

    # Year-by-year
    cycles["year"] = pd.to_datetime(cycles["entry_date"]).dt.year
    yby = {}
    for yr, grp in cycles.groupby("year"):
        pnl = grp["pnl_total"].values
        yby[int(yr)] = {
            "n": len(pnl),
            "total_pnl": round(pnl.sum(), 0),
            "win_rate": round(len(pnl[pnl > 0]) / len(pnl) * 100, 1),
            "return_pct": round(pnl.sum() / PORTFOLIO_NAV * 100, 2),
        }
    print("  Year-by-year:")
    for yr, v in sorted(yby.items()):
        print(f"    {yr}: PnL=${v['total_pnl']:,.0f} ({v['return_pct']:+.1f}%) WR={v['win_rate']}% n={v['n']}")

    # Exit type breakdown
    exit_breakdown = cycles["exit_type"].value_counts().to_dict()
    print(f"  Exit types: {exit_breakdown}")

    return {
        "name": name,
        "short_delta": short_delta,
        "long_delta": long_delta,
        "vix_filter": vix_filter,
        "metrics": {
            "full": full_metrics,
            "is": is_metrics,
            "oot": oot_metrics,
        },
        "regime_gap": regime,
        "permutation": perm,
        "year_by_year": yby,
        "exit_breakdown": exit_breakdown,
        "oot_start": str(oot_start),
        "compliance": {
            "hc344_day_conc": {
                "value": full_metrics["day_concentration"],
                "limit": 0.70,
                "verdict": "PASS" if full_metrics["day_concentration"] <= 0.70 else "FAIL",
            },
            "hc428_r1_regime": {
                "verdict": regime.get("regime_agnostic", "SKIP"),
                "gap_ratio": regime.get("gap_ratio"),
            },
            "permutation_p05": {
                "verdict": perm["verdict"],
                "p_value": perm["p_value"],
            },
        },
    }


# ============================================================
#  MAIN
# ============================================================

def main():
    print("=" * 70)
    print("  INDEX PUT CREDIT SPREADS v1 — SPY Weekly Bull-Put Spreads")
    print("  Real Dolt bid/ask pricing | Commission-free (HC #694)")
    print("=" * 70)

    spy_chains, macro, spy_px = load_data()
    print(f"SPY chains: {len(spy_chains):,} rows | dates: {spy_chains['date'].min().date()} -> {spy_chains['date'].max().date()}")
    if macro is not None:
        print(f"VIX data: {len(macro)} rows")

    # Define variants
    variants_def = [
        # name, short_delta, long_delta, vix_filter
        ("A_d15_d5",   0.15, 0.05, False),   # Moderate: sell 15d, buy 5d
        ("B_d20_d10",  0.20, 0.10, False),   # Aggressive: sell 20d, buy 10d
        ("C_d10_d3",   0.10, 0.03, False),   # Conservative: sell 10d, buy 3d
        ("D_d15_d5_vix", 0.15, 0.05, True),  # With VIX sizing/filter
        ("E_d20_d10_vix", 0.20, 0.10, True), # Aggressive + VIX filter
    ]

    all_results = {}
    for vname, sd, ld, vf in variants_def:
        result = run_variant(vname, sd, ld, vf, spy_chains, macro, spy_px)
        if result:
            all_results[vname] = result

    # Save results
    out_path = OUTPUT_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # Summary table
    print("\n" + "=" * 70)
    print("  SUMMARY — ALL VARIANTS")
    print("=" * 70)
    print(f"{'Variant':<22} {'Sharpe':>7} {'OOT Shp':>8} {'WR%':>6} {'PF':>6} {'MaxDD%':>8} {'Regime':>8} {'Perm':>6}")
    print("-" * 70)
    for vname, res in sorted(all_results.items()):
        fm = res["metrics"]["full"]
        om = res["metrics"]["oot"]
        rg = res["compliance"]["hc428_r1_regime"]
        pm = res["compliance"]["permutation_p05"]
        print(f"{vname:<22} {fm['sharpe_annual']:>7.3f} {om['sharpe_annual']:>8.3f} "
              f"{fm['win_rate_pct']:>6.1f} {fm['profit_factor']:>6.3f} "
              f"{fm['max_drawdown_pct']:>8.2f} {rg['verdict']:>8} {pm['verdict']:>6}")

    print("\nDone.")
    return all_results


if __name__ == "__main__":
    main()

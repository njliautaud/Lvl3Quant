#!/usr/bin/env python3
"""
Macro Event IV Crush v1 — SPY Straddles Around FOMC/CPI Events
================================================================
Thesis: SPY options IV spikes before FOMC/CPI announcements and crushes after.
Trade: sell SPY straddle BEFORE event, buy back AFTER. Pure theta/vega harvest.

Advantages over single-stock earnings crush:
  - 20+ events/year (FOMC 8x + CPI 12x) vs ~4 earnings/stock/year
  - Index = no gap risk from company-specific news
  - VIX term structure predicts event premium (VIX/VIX3M ↑ before events)

Data: Dolt weekly Fri snapshots 2019-2026
  - Snapshot BEFORE event: sell straddle at mid of (call_ask + put_ask)
  - Snapshot AFTER event:  buy back at mid of (call_bid + put_bid) — adversarial
  - "BEFORE" = last Friday snapshot before the event date
  - "AFTER"  = first Friday snapshot after the event date

Adversarial:
  - Sell at MID (not ask) — Robinhood typically fills near mid
  - Buy at MID (not bid) — conservative close
  - Actually let's test: sell at bid (conservative entry), buy at ask (conservative close)
  - Also test mid-price for comparison

FOMC dates: manually constructed 2019-2026 (public FRB schedule)
CPI dates: manually constructed 2019-2026 (BLS release schedule)

HC compliance:
  - HC #428 R1: regime-agnostic OOT (≥40 days test)
  - HC #344: day-concentration ≤ 0.70
  - HC #0: last-30% OOT
  - 100-trial permutation test
  - HC #694: commission-free on Robinhood
"""

import json
import warnings
from datetime import datetime, date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# --- Paths ---
CHAINS_DIR = Path("/home/jupiter/Lvl3Quant/wheel_strategy_v1/data/cache/options_real/chains")
MACRO_PATH  = Path("/home/jupiter/Lvl3Quant/wheel_strategy_v1/data/cache/macro.parquet")
CAL_PATH    = Path("/home/jupiter/Lvl3Quant/data/external/economic_calendar_2023_2026.json")
OUTPUT_DIR  = Path("/home/jupiter/Lvl3Quant/output/macro_iv_crush_v1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PORTFOLIO_NAV = 100_000
TARGET_DTE    = 14   # sell straddle with ~2 weeks to expiry


# ============================================================
#  EVENT DATES — FOMC + CPI 2019-2026
# ============================================================

# FOMC decision dates (Federal Reserve press releases, known in advance)
# Source: federalreserve.gov meeting schedules
FOMC_DATES = [
    # 2019
    "2019-01-30", "2019-03-20", "2019-05-01", "2019-06-19",
    "2019-07-31", "2019-09-18", "2019-10-30", "2019-12-11",
    # 2020
    "2020-01-29", "2020-03-03", "2020-03-15", "2020-04-29",
    "2020-06-10", "2020-07-29", "2020-09-16", "2020-11-05", "2020-12-16",
    # 2021
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16",
    "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
    # 2022
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15",
    "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
    # 2023
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14",
    "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    # 2024
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
    # 2026 (partial)
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-10",
]

# CPI release dates (BLS, second or third Tuesday of month)
# Key months for 2019-2026
CPI_DATES = [
    # 2019
    "2019-01-11", "2019-02-13", "2019-03-12", "2019-04-10",
    "2019-05-10", "2019-06-12", "2019-07-11", "2019-08-13",
    "2019-09-12", "2019-10-10", "2019-11-13", "2019-12-11",
    # 2020
    "2020-01-14", "2020-02-13", "2020-03-11", "2020-04-10",
    "2020-05-12", "2020-06-10", "2020-07-14", "2020-08-12",
    "2020-09-11", "2020-10-13", "2020-11-12", "2020-12-10",
    # 2021
    "2021-01-13", "2021-02-10", "2021-03-10", "2021-04-13",
    "2021-05-12", "2021-06-10", "2021-07-13", "2021-08-11",
    "2021-09-14", "2021-10-13", "2021-11-10", "2021-12-10",
    # 2022
    "2022-01-12", "2022-02-10", "2022-03-10", "2022-04-12",
    "2022-05-11", "2022-06-10", "2022-07-13", "2022-08-10",
    "2022-09-13", "2022-10-13", "2022-11-10", "2022-12-13",
    # 2023
    "2023-01-12", "2023-02-14", "2023-03-14", "2023-04-12",
    "2023-05-10", "2023-06-13", "2023-07-12", "2023-08-10",
    "2023-09-13", "2023-10-12", "2023-11-14", "2023-12-12",
    # 2024
    "2024-01-11", "2024-02-13", "2024-03-12", "2024-04-10",
    "2024-05-15", "2024-06-12", "2024-07-11", "2024-08-14",
    "2024-09-11", "2024-10-10", "2024-11-13", "2024-12-11",
    # 2025
    "2025-01-15", "2025-02-12", "2025-03-12", "2025-04-10",
    "2025-05-13", "2025-06-11", "2025-07-11", "2025-08-12",
    "2025-09-10", "2025-10-15", "2025-11-12", "2025-12-10",
    # 2026 (partial)
    "2026-01-14", "2026-02-11", "2026-03-11", "2026-04-10",
    "2026-05-13", "2026-06-11",
]


def get_all_event_dates():
    events = []
    for d in FOMC_DATES:
        events.append({"date": pd.Timestamp(d), "type": "FOMC"})
    for d in CPI_DATES:
        events.append({"date": pd.Timestamp(d), "type": "CPI"})
    df = pd.DataFrame(events).sort_values("date").reset_index(drop=True)
    return df


# ============================================================
#  DATA LOADING
# ============================================================

def load_data():
    spy = pd.read_parquet(CHAINS_DIR / "SPY.parquet")
    spy["date"] = pd.to_datetime(spy["date"])
    spy["expiration"] = pd.to_datetime(spy["expiration"])
    spy = spy.sort_values(["date", "expiration", "strike"])

    macro = pd.read_parquet(MACRO_PATH)
    macro["date"] = pd.to_datetime(macro["date"])

    return spy, macro


# ============================================================
#  PICK ATM STRADDLE
# ============================================================

def pick_atm_straddle(spy_chains, obs_date, target_dte=TARGET_DTE):
    """
    At obs_date, find the ATM call + put with DTE closest to target_dte.
    Returns (call_row, put_row) or (None, None).
    """
    on_date = spy_chains[spy_chains["date"] == obs_date].copy()
    if on_date.empty:
        return None, None

    # Find expiry with DTE closest to target
    dte_by_expiry = on_date.groupby("expiration")["dte"].first()
    dte_by_expiry = dte_by_expiry[(dte_by_expiry >= 7) & (dte_by_expiry <= 35)]
    if dte_by_expiry.empty:
        return None, None

    best_expiry = (dte_by_expiry - target_dte).abs().idxmin()
    chain = on_date[on_date["expiration"] == best_expiry]

    # ATM call: delta closest to 0.50
    calls = chain[chain["type"] == "c"]
    if calls.empty:
        return None, None
    atm_call = calls.iloc[(calls["delta"] - 0.50).abs().argsort()[:1]].iloc[0]

    # Put at same strike
    puts = chain[chain["type"] == "p"]
    put_match = puts[puts["strike"] == atm_call["strike"]]
    if put_match.empty:
        # Fallback: put with delta closest to -0.50
        put_match = puts.iloc[(puts["delta"].abs() - 0.50).abs().argsort()[:1]]
    if put_match.empty:
        return None, None

    atm_put = put_match.iloc[0]
    return atm_call, atm_put


def straddle_price_entry(call_row, put_row, mode="mid"):
    """Credit received when selling straddle. mode: bid (conservative), mid, ask (optimistic)."""
    if mode == "bid":
        # Sell at bid (worst case)
        return float(call_row["bid"]) + float(put_row["bid"])
    elif mode == "ask":
        # Sell at ask (best case)
        return float(call_row["ask"]) + float(put_row["ask"])
    else:
        # Mid price
        return float(call_row["mid"]) + float(put_row["mid"])


def straddle_price_exit(call_row, put_row, mode="mid"):
    """Cost to buy back straddle. mode: ask (conservative close), mid, bid (optimistic)."""
    if mode == "ask":
        # Buy at ask (worst case)
        return float(call_row["ask"]) + float(put_row["ask"])
    elif mode == "bid":
        # Buy at bid (best case)
        return float(call_row["bid"]) + float(put_row["bid"])
    else:
        return float(call_row["mid"]) + float(put_row["mid"])


# ============================================================
#  BUILD CYCLES
# ============================================================

def build_cycles(events_df, spy_chains, entry_mode="mid", exit_mode="mid"):
    """
    For each event:
    - Find last observation date BEFORE the event (entry)
    - Find first observation date AFTER the event (exit)
    - Sell ATM straddle at entry, buy back at exit

    Entry mode: how we price the sell (mid = realistic, bid = conservative)
    Exit mode:  how we price the buyback (mid = realistic, ask = conservative)
    """
    obs_dates = sorted(spy_chains["date"].unique())

    cycles = []
    for _, event in events_df.iterrows():
        event_date = event["date"]
        event_type = event["type"]

        # Find entry: last obs date strictly before event
        before = [d for d in obs_dates if d < event_date]
        if not before:
            continue
        entry_date = max(before)

        # Find exit: first obs date strictly after event
        after = [d for d in obs_dates if d > event_date]
        if not after:
            continue
        exit_date = min(after)

        days_held = (exit_date - entry_date).days

        # Entry: pick ATM straddle
        entry_call, entry_put = pick_atm_straddle(spy_chains, entry_date)
        if entry_call is None:
            continue

        entry_strike  = float(entry_call["strike"])
        entry_expiry  = entry_call["expiration"]
        entry_dte     = int(entry_call["dte"])
        entry_credit  = straddle_price_entry(entry_call, entry_put, mode=entry_mode)

        if entry_credit <= 0:
            continue

        # Exit: find same-expiry options at exit date
        exit_chain = spy_chains[(spy_chains["date"] == exit_date) &
                                (spy_chains["expiration"] == entry_expiry)]

        if exit_chain.empty:
            # Expiry passed — options expired
            # Get SPY spot from ATM call at exit date (different expiry)
            exit_atm_call, _ = pick_atm_straddle(spy_chains, exit_date, target_dte=21)
            if exit_atm_call is None:
                # Can't find exit price — use intrinsic
                # This cycle is excluded (too uncertain)
                continue
            spy_spot = float(exit_atm_call["strike"])
            # Intrinsic value of straddle at entry_strike
            intrinsic = abs(spy_spot - entry_strike)
            exit_cost = intrinsic
            exit_type = "expired_intrinsic"
            exit_dte  = 0
        else:
            # MTM exit using same expiry
            exit_calls = exit_chain[exit_chain["type"] == "c"]
            exit_puts  = exit_chain[exit_chain["type"] == "p"]

            exit_call_match = exit_calls[exit_calls["strike"] == entry_strike]
            exit_put_match  = exit_puts[exit_puts["strike"] == entry_strike]

            if exit_call_match.empty or exit_put_match.empty:
                # Try nearest strike
                if not exit_calls.empty and not exit_puts.empty:
                    exit_call_m = exit_calls.iloc[(exit_calls["strike"] - entry_strike).abs().argsort()[:1]].iloc[0]
                    exit_put_m  = exit_puts.iloc[(exit_puts["strike"] - entry_strike).abs().argsort()[:1]].iloc[0]
                    exit_cost = straddle_price_exit(exit_call_m, exit_put_m, mode=exit_mode)
                    exit_type = "mtm_nearest_strike"
                    exit_dte = int(exit_calls.iloc[0]["dte"])
                else:
                    continue
            else:
                exit_call_r = exit_call_match.iloc[0]
                exit_put_r  = exit_put_match.iloc[0]
                exit_cost = straddle_price_exit(exit_call_r, exit_put_r, mode=exit_mode)
                exit_type = "mtm_exact"
                exit_dte = int(exit_call_r["dte"])

        # P&L per straddle
        pnl_per_straddle = entry_credit - exit_cost

        # Sizing: 1 contract = 100 shares of SPY
        # Max risk = straddle premium × contracts (unlimited upside, but straddle buyer profits if big move)
        # Size to spend max 5% NAV on premium (entry_credit × contracts × 100 ≤ 5% NAV)
        max_premium_spend = PORTFOLIO_NAV * 0.05
        contracts = max(1, int(max_premium_spend / (entry_credit * 100)))

        pnl_total = pnl_per_straddle * contracts * 100
        iv_crush_pct = (entry_credit - exit_cost) / entry_credit * 100

        cycles.append({
            "event_date": event_date,
            "event_type": event_type,
            "entry_date": entry_date,
            "exit_date": exit_date,
            "days_held": days_held,
            "entry_strike": entry_strike,
            "entry_expiry": entry_expiry,
            "entry_dte": entry_dte,
            "exit_dte": exit_dte,
            "entry_credit": entry_credit,
            "exit_cost": exit_cost,
            "pnl_per_straddle": pnl_per_straddle,
            "contracts": contracts,
            "pnl_total": pnl_total,
            "iv_crush_pct": iv_crush_pct,
            "exit_type": exit_type,
        })

    return pd.DataFrame(cycles)


# ============================================================
#  METRICS + VALIDATION
# ============================================================

def compute_metrics(cycles, label="full"):
    if cycles.empty or len(cycles) < 5:
        return {"label": label, "n_trades": 0}

    pnl = cycles["pnl_total"].values
    n = len(pnl)
    mean_pnl = np.mean(pnl)
    std_pnl = np.std(pnl, ddof=1)

    dates = pd.to_datetime(cycles["entry_date"])
    total_days = (dates.max() - dates.min()).days
    years = max(total_days / 365.25, 0.1)
    tpy = n / years

    sharpe = (mean_pnl / std_pnl * np.sqrt(tpy)) if std_pnl > 0 else 0
    downside = pnl[pnl < 0]
    sortino = (mean_pnl / np.std(downside, ddof=1) * np.sqrt(tpy)) if len(downside) > 1 else 0

    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    win_rate = len(wins) / n
    pf = wins.sum() / (-losses.sum()) if len(losses) > 0 and losses.sum() < 0 else np.inf

    cum_pnl = np.cumsum(pnl)
    running_max = np.maximum.accumulate(cum_pnl)
    drawdowns = (cum_pnl - running_max) / PORTFOLIO_NAV
    max_dd = float(drawdowns.min())

    day_counts = cycles.groupby("entry_date").size()
    day_conc = float(day_counts.max() / n) if n > 0 else 0

    return {
        "label": label,
        "n_trades": n,
        "total_return_pct": round(pnl.sum() / PORTFOLIO_NAV * 100, 2),
        "mean_pnl": round(mean_pnl, 2),
        "sharpe_annual": round(sharpe, 3),
        "sortino_annual": round(sortino, 3),
        "win_rate_pct": round(win_rate * 100, 1),
        "profit_factor": round(pf, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "day_concentration": round(day_conc, 3),
        "avg_entry_credit": round(float(cycles["entry_credit"].mean()), 2),
        "avg_iv_crush_pct": round(float(cycles["iv_crush_pct"].mean()), 1),
        "avg_days_held": round(float(cycles["days_held"].mean()), 1),
    }


def regime_gap_test(cycles, macro):
    """Classify each cycle by SPY return in period (proxy: VIX level at entry)."""
    if macro is None or cycles.empty:
        return {"regime_agnostic": "SKIP"}

    cycles = cycles.copy()
    macro_map = macro.set_index("date")["vix"].to_dict()

    def get_regime(row):
        vix = macro_map.get(row["entry_date"])
        if vix is None:
            return "flat"
        if vix < 15:
            return "green"
        elif vix > 22:
            return "red"
        else:
            return "flat"

    cycles["regime"] = cycles.apply(get_regime, axis=1)

    results = {}
    for regime in ["green", "red", "flat"]:
        sub = cycles[cycles["regime"] == regime]
        if len(sub) < 3:
            results[regime] = {"n": len(sub), "sharpe": None}
            continue
        pnl = sub["pnl_total"].values
        tpy = len(pnl) / max((sub["entry_date"].max() - sub["entry_date"].min()).days / 365.25, 0.1)
        std = np.std(pnl, ddof=1)
        sharpe = (np.mean(pnl) / std * np.sqrt(tpy)) if std > 0 else 0
        results[regime] = {
            "n": len(sub),
            "sharpe": round(sharpe, 3),
            "win_rate": round(len(pnl[pnl > 0]) / len(pnl) * 100, 1),
        }

    sharpes = [v["sharpe"] for v in results.values() if v.get("sharpe") is not None]
    if len(sharpes) >= 2:
        max_s, min_s = max(sharpes), min(sharpes)
        gap = (max_s - min_s) / max(abs(max_s), abs(min_s)) if max(abs(max_s), abs(min_s)) > 0 else 0
    else:
        gap = None

    return {
        "per_regime": results,
        "gap_ratio": round(gap, 3) if gap is not None else None,
        "regime_agnostic": "PASS" if (gap is not None and gap <= 0.50) else "FAIL",
    }


def permutation_test(cycles, n_trials=100):
    pnl = cycles["pnl_total"].values
    if len(pnl) < 5:
        return {"verdict": "SKIP"}

    dates = pd.to_datetime(cycles["entry_date"])
    tpy = len(pnl) / max((dates.max() - dates.min()).days / 365.25, 0.1)
    std = np.std(pnl, ddof=1)
    real_sharpe = (np.mean(pnl) / std * np.sqrt(tpy)) if std > 0 else 0

    rng = np.random.default_rng(42)
    null_sharpes = []
    for _ in range(n_trials):
        sh = rng.permutation(pnl)
        s = np.std(sh, ddof=1)
        null_sharpes.append((np.mean(sh) / s * np.sqrt(tpy)) if s > 0 else 0)

    null_sharpes = np.array(null_sharpes)
    p_val = np.mean(null_sharpes >= real_sharpe)

    return {
        "real_sharpe": round(real_sharpe, 3),
        "p_value": round(float(p_val), 3),
        "significant_05": bool(p_val < 0.05),
        "verdict": "PASS" if p_val < 0.05 else "FAIL",
    }


# ============================================================
#  MAIN
# ============================================================

def main():
    print("=" * 70)
    print("  MACRO EVENT IV CRUSH v1 — SPY Straddles around FOMC/CPI")
    print("  Real Dolt bid/ask | Commission-free (HC #694)")
    print("=" * 70)

    spy_chains, macro = load_data()
    print(f"SPY chains: {len(spy_chains):,} rows | {spy_chains['date'].nunique()} obs dates")

    events = get_all_event_dates()
    print(f"Events: {len(events)} total ({(events['type']=='FOMC').sum()} FOMC, {(events['type']=='CPI').sum()} CPI)")

    # Filter events within chain date range
    min_date = spy_chains["date"].min()
    max_date = spy_chains["date"].max()
    events = events[(events["date"] >= min_date) & (events["date"] <= max_date)]
    print(f"Events in data range ({min_date.date()} - {max_date.date()}): {len(events)}")

    all_results = {}

    # Test multiple pricing modes
    variants = [
        ("mid_mid", "mid", "mid"),          # Mid price entry/exit (baseline)
        ("bid_ask", "bid", "ask"),           # Conservative (sell at bid, buy at ask)
        ("mid_ask", "mid", "ask"),           # Semi-conservative
    ]

    for vname, entry_mode, exit_mode in variants:
        print(f"\n{'='*60}")
        print(f"  Variant: {vname} (entry={entry_mode}, exit={exit_mode})")
        print(f"{'='*60}")

        cycles = build_cycles(events, spy_chains, entry_mode=entry_mode, exit_mode=exit_mode)
        print(f"  Cycles: {len(cycles)}")

        if cycles.empty or len(cycles) < 10:
            print("  Too few cycles — skipping")
            continue

        # Split by event type
        fomc_cycles = cycles[cycles["event_type"] == "FOMC"]
        cpi_cycles  = cycles[cycles["event_type"] == "CPI"]
        print(f"  FOMC: {len(fomc_cycles)}, CPI: {len(cpi_cycles)}")

        # Exit type breakdown
        print(f"  Exit types: {cycles['exit_type'].value_counts().to_dict()}")
        print(f"  Avg IV crush: {cycles['iv_crush_pct'].mean():.1f}%")

        # IS / OOT split
        all_dates = sorted(cycles["entry_date"].unique())
        oot_start_idx = int(len(all_dates) * 0.70)
        oot_start = all_dates[oot_start_idx] if oot_start_idx < len(all_dates) else all_dates[-1]
        is_cyc  = cycles[cycles["entry_date"] < oot_start]
        oot_cyc = cycles[cycles["entry_date"] >= oot_start]

        full_m = compute_metrics(cycles, "full")
        is_m   = compute_metrics(is_cyc, "IS")
        oot_m  = compute_metrics(oot_cyc, "OOT")

        print(f"  Full: Sharpe={full_m.get('sharpe_annual','N/A')} WR={full_m.get('win_rate_pct','N/A')}% "
              f"PF={full_m.get('profit_factor','N/A')} MaxDD={full_m.get('max_drawdown_pct','N/A')}%")
        print(f"  IS:   Sharpe={is_m.get('sharpe_annual','N/A')} WR={is_m.get('win_rate_pct','N/A')}%")
        print(f"  OOT:  Sharpe={oot_m.get('sharpe_annual','N/A')} WR={oot_m.get('win_rate_pct','N/A')}% "
              f"n={oot_m.get('n_trades',0)}")

        # Sub-analysis by event type
        fomc_m = compute_metrics(fomc_cycles, "FOMC")
        cpi_m  = compute_metrics(cpi_cycles, "CPI")
        print(f"  FOMC: Sharpe={fomc_m.get('sharpe_annual','N/A')} WR={fomc_m.get('win_rate_pct','N/A')}%")
        print(f"  CPI:  Sharpe={cpi_m.get('sharpe_annual','N/A')} WR={cpi_m.get('win_rate_pct','N/A')}%")

        # Regime test
        regime = regime_gap_test(cycles, macro)
        print(f"  Regime: {regime.get('regime_agnostic')} gap={regime.get('gap_ratio')}")
        for r, v in regime.get("per_regime", {}).items():
            print(f"    {r}: n={v['n']} Sharpe={v['sharpe']} WR={v.get('win_rate','?')}%")

        # Permutation test
        perm = permutation_test(cycles)
        print(f"  Permutation: {perm['verdict']} p={perm['p_value']}")

        # Year-by-year
        cycles["year"] = pd.to_datetime(cycles["entry_date"]).dt.year
        print("  Year-by-year:")
        for yr, grp in cycles.groupby("year"):
            pnl = grp["pnl_total"].values
            print(f"    {yr}: PnL=${pnl.sum():,.0f} ({pnl.sum()/PORTFOLIO_NAV*100:+.1f}%) "
                  f"WR={len(pnl[pnl>0])/len(pnl)*100:.1f}% n={len(pnl)} "
                  f"avg_crush={grp['iv_crush_pct'].mean():.1f}%")

        all_results[vname] = {
            "name": vname,
            "metrics": {"full": full_m, "is": is_m, "oot": oot_m,
                       "fomc": fomc_m, "cpi": cpi_m},
            "regime_gap": regime,
            "permutation": perm,
            "oot_start": str(oot_start),
            "compliance": {
                "hc344": {"value": full_m.get("day_concentration",0), "limit": 0.70,
                          "verdict": "PASS" if full_m.get("day_concentration",0) <= 0.70 else "FAIL"},
                "hc428_r1": {"verdict": regime.get("regime_agnostic","SKIP"),
                             "gap": regime.get("gap_ratio")},
                "permutation": perm,
            },
        }

    # Save
    out_path = OUTPUT_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"{'Variant':<16} {'Sharpe':>7} {'OOT_Shp':>8} {'WR%':>6} {'MaxDD%':>8} {'Regime':>8} {'Perm':>6}")
    print("-" * 70)
    for vname, res in all_results.items():
        fm = res["metrics"]["full"]
        om = res["metrics"]["oot"]
        rg = res["compliance"]["hc428_r1"]
        pm = res["compliance"]["permutation"]
        print(f"{vname:<16} {fm.get('sharpe_annual',0):>7.3f} {om.get('sharpe_annual',0):>8.3f} "
              f"{fm.get('win_rate_pct',0):>6.1f} {fm.get('max_drawdown_pct',0):>8.2f} "
              f"{rg.get('verdict','?'):>8} {pm.get('verdict','?'):>6}")
    return all_results


if __name__ == "__main__":
    main()
